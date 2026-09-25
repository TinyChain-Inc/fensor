use fensor::{
    AxisRange, Error, Layout, Tensor, TensorRead, TensorSchema, TensorWrite, TensorWriteBulk,
};
use ha_ndarray::{range, shape};
use number_general::DType;

mod common;
use common::{FsEntry, new_dir};

#[tokio::test]
async fn writes_validate_cardinality_before_mutation() {
    for layout in [Layout::Dense, Layout::Sparse { axis: None }] {
        let (_, dir) = new_dir("bounded_writes").await;
        let tensor = Tensor::<FsEntry, u8>::create(
            dir,
            TensorSchema::new(u8::dtype(), shape![6]).unwrap(),
            layout,
            2,
        )
        .await
        .unwrap();
        tensor
            .write_values(range![AxisRange::In(0, 6, 1)], vec![1, 2, 3, 4, 5, 6])
            .await
            .unwrap();
        for values in [vec![9], vec![9; 7]] {
            assert!(matches!(
                tensor
                    .write_values(range![AxisRange::In(0, 6, 1)], values)
                    .await,
                Err(Error::DataMismatch(_))
            ));
            for i in 0..6 {
                assert_eq!(tensor.read_value(&[i]).await.unwrap(), i as u8 + 1);
            }
        }
        tensor
            .write_values(range![AxisRange::Of(vec![5, 1, 5])], vec![3, 4, 7])
            .await
            .unwrap();
        assert_eq!(tensor.read_value(&[5]).await.unwrap(), 7);
        assert_eq!(tensor.read_value(&[1]).await.unwrap(), 4);
        tensor
            .write_values(range![AxisRange::In(0, 6, 2)], vec![8, 9, 10])
            .await
            .unwrap();
        assert_eq!(tensor.read_value(&[4]).await.unwrap(), 10);
    }
}

#[tokio::test]
async fn huge_mismatched_write_does_not_expand_coordinates() {
    tokio::time::timeout(std::time::Duration::from_secs(30), async {
        let (_, dir) = new_dir("huge_bounded_write").await;
        let tensor = Tensor::<FsEntry, u8>::create(
            dir,
            TensorSchema::new(u8::dtype(), shape![1_000_000_000]).unwrap(),
            Layout::Sparse { axis: None },
            2,
        )
        .await
        .unwrap();
        tensor.write_value(&[0], 127).await.unwrap();
        assert!(matches!(
            tensor
                .write_values(range![AxisRange::In(0, 1_000_000_000, 1)], vec![1])
                .await,
            Err(Error::DataMismatch(_))
        ));
        assert_eq!(tensor.read_value(&[0]).await.unwrap(), 127);
    })
    .await
    .expect("length validation must not traverse or allocate the full range");
}
