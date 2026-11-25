use destream::{de, en};
use fensor::SparseCell::{Key, Payload};
use fensor::{MAX_BLOCK_CAPACITY, SparseNode};
use futures::{TryStreamExt, stream};

async fn transcode<'en, In, Out>(value: In, json: bool) -> Result<Out, String>
where
    In: en::IntoStream<'en> + 'en,
    Out: de::FromStream<Context = ()>,
{
    let chunks = if json {
        destream_json::en::encode(value)
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap()
    } else {
        tbon::en::encode(value)
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap()
    };

    // Split scalar and container boundaries independently of encoder chunking.
    let input = stream::iter(chunks.into_iter().flat_map(|chunk| {
        (0..chunk.len()).map(move |i| Ok::<_, std::io::Error>(chunk.slice(i..i + 1)))
    }));
    if json {
        destream_json::de::try_decode((), input)
            .await
            .map_err(|cause| cause.to_string())
    } else {
        tbon::de::try_decode((), input)
            .await
            .map_err(|cause| cause.to_string())
    }
}

#[tokio::test]
async fn sparse_nodes_use_native_tree_framing_and_typed_values() {
    for json in [false, true] {
        let node = SparseNode(b_tree::Node::Leaf(vec![vec![
            Key(u64::MAX),
            Payload(vec![i64::MIN, i64::MAX]),
        ]]));
        let decoded: b_tree::Node<Vec<(u64, Vec<i64>)>> = transcode(&node, json).await.unwrap();
        let b_tree::Node::Leaf(rows) = decoded else {
            panic!("expected a native leaf");
        };

        assert_eq!(rows, vec![(u64::MAX, vec![i64::MIN, i64::MAX])]);

        let children = vec![Default::default(); 2];
        let key_only = b_tree::Node::Index(vec![(2u64,)], children.clone());
        let SparseNode(b_tree::Node::Index(rows, actual)) =
            transcode::<_, SparseNode<f64>>(&key_only, json)
                .await
                .unwrap()
        else {
            panic!("expected a sparse index");
        };

        assert_eq!(rows, vec![vec![Key(2)]]);
        assert_eq!(actual, children);

        let mut expected = vec![1., -0.];
        if !json {
            expected.extend([f64::NAN, f64::NEG_INFINITY]);
        }

        let index = b_tree::Node::Index(vec![(2u64, expected.clone())], children.clone());
        let SparseNode(b_tree::Node::Index(rows, actual)) =
            transcode::<_, SparseNode<f64>>(&index, json).await.unwrap()
        else {
            panic!("expected a sparse index with a complete separator row");
        };

        assert_eq!(actual, children);
        let [Key(2), Payload(values)] = rows[0].as_slice() else {
            panic!("expected a complete separator row");
        };

        assert_eq!(values.len(), expected.len());
        assert_eq!(values[0], 1.);
        assert_eq!(values[1], 0.);
        assert!(values[1].is_sign_negative());
        if !json {
            assert!(values[2].is_nan());
            assert_eq!(values[3], f64::NEG_INFINITY);
        }
    }
}

#[cfg(feature = "complex")]
#[tokio::test]
async fn sparse_complex_values_use_ordered_typed_components() {
    use fensor::complex::{Complex32, Complex64};

    type Components = (f32, f32);

    let c32 = Complex32::new(f32::NAN, -0.);
    let node = SparseNode(b_tree::Node::Leaf(vec![vec![Key(0), Payload(vec![c32])]]));
    let decoded: b_tree::Node<Vec<(u64, Vec<Components>)>> = transcode(&node, false).await.unwrap();
    let b_tree::Node::Leaf(rows) = decoded else {
        panic!("expected a native leaf");
    };

    let (re, im) = rows[0].1[0];
    assert!(re.is_nan());
    assert_eq!(im, 0.);
    assert!(im.is_sign_negative());

    let node = b_tree::Node::Leaf(vec![(0u64, vec![(2.25f64, -3.5f64)])]);
    let SparseNode(b_tree::Node::Leaf(rows)) = transcode::<_, SparseNode<Complex64>>(&node, false)
        .await
        .unwrap()
    else {
        panic!("expected a sparse leaf");
    };

    assert_eq!(
        rows,
        vec![vec![Key(0), Payload(vec![Complex64::new(2.25, -3.5)])]]
    );
}

#[tokio::test]
async fn sparse_node_decoding_enforces_shape_and_allocation_limits() {
    async fn rejects<'en>(value: impl en::IntoStream<'en> + 'en) {
        assert!(transcode::<_, SparseNode<f64>>(value, false).await.is_err());
    }

    for node in [
        SparseNode(b_tree::Node::Leaf(
            (0..4)
                .map(|id| vec![Key(id), Payload(vec![1.; MAX_BLOCK_CAPACITY])])
                .collect(),
        )),
        SparseNode(b_tree::Node::Leaf(
            (0..1024)
                .map(|id| vec![Key(id), Payload(vec![1.])])
                .collect(),
        )),
        SparseNode(b_tree::Node::Index(
            (0..1023).map(|id| vec![Key(id)]).collect(),
            vec![Default::default(); 1024],
        )),
    ] {
        transcode::<_, SparseNode<f64>>(&node, false).await.unwrap();
    }

    let b_tree::Node::Index(_, children) =
        b_tree::Node::<()>::Index((), vec![Default::default(); 1025])
    else {
        unreachable!()
    };

    rejects((
        false,
        (0..1024u64).map(|id| (id,)).collect::<Vec<_>>(),
        &children[..],
    ))
    .await;

    for count in [0, 1, 3] {
        rejects((false, vec![(0u64,)], &children[..count])).await;
    }

    rejects((false, Vec::<(u64,)>::new(), &children[..1])).await;
    rejects((true, vec![(0u64,)])).await;
    rejects((false, vec![Vec::<u64>::new()], &children[..2])).await;
    rejects((false, vec![("bad block",)], &children[..2])).await;
    rejects((true, vec![(0u64, vec![1f64], 1u64)])).await;
    rejects((true, vec![(0u64, vec![1f64])], &children[..1])).await;
    rejects((true, vec![(0u64, vec!["bad value"])])).await;

    for count in [0, MAX_BLOCK_CAPACITY + 1] {
        rejects((true, vec![(0u64, vec![1f64; count])])).await;
    }

    for (rows, values) in [(1025, 1), (5, MAX_BLOCK_CAPACITY)] {
        let rows = (0..rows)
            .map(|id| (id as u64, vec![1f64; values]))
            .collect::<Vec<_>>();

        rejects((true, rows)).await;
    }
}
