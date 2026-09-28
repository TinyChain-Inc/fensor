//! Storage contracts are parameterized independently of the numerical matrix.

use fensor::{
    Error, Layout, Tensor, TensorArray, TensorElement, TensorFileEntry, TensorGeometry, TensorRead,
    TensorSchema, TensorSparseIndex, TensorTransform, TensorWrite, TensorWriteBulk,
};
use ha_ndarray::shape;

use common::{FsEntry, cleanup, fixture, numbers::same};

mod common;

async fn storage<T: TensorElement>()
where
    FsEntry: TensorFileEntry<T>,
{
    for layout in [Layout::Dense, Layout::Sparse { axis: None }] {
        // Dense payloads exceed the cache even for u8. Sparse fixtures also
        // budget for their many distinct axis keys; index memory is a separate cost.
        let len = if matches!(layout, Layout::Dense) {
            4097
        } else {
            65
        };
        let mut expected: Vec<T> = (0..len)
            .map(|i| if i % 3 == 0 { T::ZERO } else { T::ONE })
            .collect();
        let cache = if matches!(layout, Layout::Dense) {
            2048
        } else {
            1_000_000
        };
        let (root, tensor) = fixture::source(
            "dtype_storage",
            shape![len as u64],
            layout,
            31,
            cache,
            expected.clone(),
        )
        .await;
        assert_eq!(tensor.dtype(), T::dtype());
        assert_eq!(tensor.view().dtype(), T::dtype());
        assert_eq!(tensor.schema().dtype(), T::dtype());

        tensor
            .view()
            .flip(0)
            .unwrap()
            .write_value(&[0], T::ONE)
            .await
            .unwrap();
        expected[len - 1] = T::ONE;
        fixture::consumers(&tensor.view(), &expected, same, same).await;
        tensor.write_value(&[1], T::ZERO).await.unwrap();
        expected[1] = T::ZERO;
        tensor.sync().await.unwrap();
        drop(tensor);
        let dir = common::open_dir(&root).unwrap();
        // A mismatched concrete payload type must never reinterpret metadata.
        if T::dtype() != <u8 as number_general::DType>::dtype() {
            assert!(Tensor::<FsEntry, u8>::load(dir.clone()).await.is_err());
        } else {
            assert!(Tensor::<FsEntry, f64>::load(dir.clone()).await.is_err());
        }
        let tensor = Tensor::<FsEntry, T>::load(dir.clone()).await.unwrap();
        assert_eq!(tensor.schema().dtype(), T::dtype());
        fixture::blocks(&tensor, &expected, same).await;

        let blocks = dir.read().await.get_dir("blocks").unwrap().clone();
        let metadata = blocks.read().await.get_file("metadata").unwrap().clone();
        assert_eq!(
            metadata
                .read::<fensor::TensorMetadata<T>>()
                .await
                .unwrap()
                .shape()
                .as_slice(),
            &[len as u64]
        );

        // Corrupt a required block through its owning filesystem adapter. Reads
        // and scalar writes must reject the length before indexing or mutation.
        let id = if matches!(layout, Layout::Dense) {
            0
        } else {
            tensor.lookup_block_id(&[2, 2]).await.unwrap().unwrap()
        };
        let file = blocks
            .read()
            .await
            .get_file(&id.to_string())
            .unwrap()
            .clone();
        {
            let mut values = file.write::<Vec<T>>().await.unwrap();
            values.clear();
        }
        assert!(matches!(
            tensor.read_value(&[2]).await,
            Err(Error::InvalidLayout(_))
        ));
        assert!(matches!(
            tensor.write_value(&[2], T::ONE).await,
            Err(Error::InvalidLayout(_))
        ));
        assert!(file.read::<Vec<T>>().await.unwrap().is_empty());
        drop(tensor);
        drop(dir);
        cleanup(&root).await;
    }

    // All dtypes share the same elementwise zero lifecycle, including through
    // geometric and bulk writes. Keep persistence permutations in the fixture above.
    let (root, tensor) = fixture::source(
        "dtype_shared_key",
        shape![2, 2],
        Layout::Sparse { axis: None },
        4,
        1_000_000,
        [T::ONE, T::ONE, T::ZERO, T::ZERO],
    )
    .await;
    tensor
        .view()
        .transpose(None)
        .unwrap()
        .write_value(&[0, 0], T::ZERO)
        .await
        .unwrap();
    assert_eq!(tensor.read_value(&[0, 1]).await.unwrap(), T::ONE);
    tensor
        .write_values(
            smallvec::smallvec![fensor::AxisRange::At(0), fensor::AxisRange::In(0, 2, 1)],
            vec![T::ZERO, T::ZERO],
        )
        .await
        .unwrap();
    assert_eq!(tensor.read_value(&[0, 1]).await.unwrap(), T::ZERO);
    cleanup(&root).await;
}

macro_rules! storage_case {
    ($name:ident,$t:ty) => {
        #[tokio::test]
        async fn $name() {
            storage::<$t>().await;
        }
    };
}

storage_case!(u8_storage, u8);
storage_case!(u16_storage, u16);
storage_case!(u32_storage, u32);
storage_case!(u64_storage, u64);
storage_case!(i8_storage, i8);
storage_case!(i16_storage, i16);
storage_case!(i32_storage, i32);
storage_case!(i64_storage, i64);
storage_case!(f32_storage, f32);
storage_case!(f64_storage, f64);

#[cfg(feature = "complex")]
storage_case!(complex32_storage, fensor::complex::Complex32);
#[cfg(feature = "complex")]
storage_case!(complex64_storage, fensor::complex::Complex64);

#[test]
fn concrete_schema_feature_boundary() {
    use number_general::{ComplexType, NumberType};
    for class in [ComplexType::C32, ComplexType::C64] {
        let schema = TensorSchema::new(NumberType::Complex(class), shape![2]);
        assert_eq!(schema.is_ok(), cfg!(feature = "complex"));
    }
}

#[cfg(feature = "complex")]
#[tokio::test]
async fn malformed_complex_components_are_rejected_by_adapter() {
    // These are adapter-owned payload tags, not a fensor wire format.
    for tag in [21u8, 23] {
        let truncated = (tag, vec![(1f64,)]);
        assert!(
            tbon::de::try_decode::<_, _, FsEntry>((), tbon::en::encode(&truncated).unwrap())
                .await
                .is_err()
        );
        let excess = (tag, vec![(1f64, 2f64, 3f64)]);
        assert!(
            tbon::de::try_decode::<_, _, FsEntry>((), tbon::en::encode(&excess).unwrap())
                .await
                .is_err()
        );
    }
}
