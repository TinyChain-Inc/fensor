use super::*;
use crate::complex::Complex64;
use crate::test_support::{cleanup, fixture};

#[tokio::test]
async fn complete_groups_pack_without_materialization() {
    let width = 7;
    let count = MAX_BATCH_ELEMENTS / width + 1;
    let (root, tensor) = fixture::source(
        "fft_packs",
        smallvec::smallvec![count as u64, width as u64],
        Layout::Dense,
        31,
        2048,
        std::iter::repeat_n(Complex64::ONE, count * width),
    )
    .await;
    tensor.sync().await.unwrap();

    fn files(path: &std::path::Path) -> std::collections::BTreeSet<std::path::PathBuf> {
        let mut paths = std::collections::BTreeSet::new();

        for entry in std::fs::read_dir(path).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                paths.extend(files(&path));
            } else {
                paths.insert(path);
            }
        }
        paths
    }

    let before = files(&root);
    let view = tensor.view().fft().await.unwrap();
    let coords = (0..count)
        .map(|i| vec![i as u64, 0])
        .chain([vec![0, 0], vec![0, 1]])
        .collect();
    let request = BatchRequest::explicit(coords).unwrap();

    crate::read_metrics::CURRENT
        .scope(Default::default(), async {
            let batch = expression::evaluate_batch(&view, &request).await.unwrap();
            assert!(
                batch.values[..count + 1]
                    .iter()
                    .all(|&v| v == Complex64::new(width as f64, 0.))
            );
            let ku = 32. * width as f64 * f64::EPSILON / 2.;
            assert!(batch.values[count + 1].norm() <= ku / (1. - ku) * width as f64);
            crate::read_metrics::CURRENT.with(|m| {
                let m = m.borrow();
                assert_eq!(m.slice_requests, 3); // Outer request plus two complete packs.
                assert_eq!(m.requested, count * width); // Duplicate outputs share source reads.
            });
        })
        .await;
    assert!(
        crate::expression::traversal::preferred(&view, view.shape())
            .unwrap()
            .is_none()
    );
    assert_eq!(
        files(&root),
        before,
        "evaluation must not create result storage"
    );
    cleanup(&root).await;
}

#[test]
fn invalid_axes_fail_before_execution() {
    for shape in [&[][..], &[0][..]] {
        assert!(matches!(
            validate_axes(shape, 1, "fft"),
            Err(Error::InvalidLayout(_))
        ));
    }
    assert!(validate_axes(&[MAX_BATCH_ELEMENTS as u64], 1, "fft").is_ok());
    assert!(matches!(
        validate_axes(&[MAX_BATCH_ELEMENTS as u64 + 1], 1, "fft"),
        Err(Error::Unsupported(_))
    ));
    assert!(validate_axes(&[u64::MAX, 2], 1, "fft").is_err());
}
