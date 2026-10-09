use super::*;

#[test]
fn validated_layout_selects_its_axis() {
    for (layout, expected) in [
        (Layout::Dense, 0),
        (Layout::Sparse { axis: None }, 0),
        (Layout::Sparse { axis: Some(0) }, 0),
        (Layout::Sparse { axis: Some(1) }, 1),
        (Layout::Sparse { axis: Some(2) }, 2),
    ] {
        let geometry = crate::StorageGeometry::new(
            TensorSchema::new(
                number_general::NumberType::Float(number_general::FloatType::F32),
                vec![2, 2, 2].into(),
            )
            .unwrap(),
            layout,
            vec![1, 1, 1].into(),
        )
        .unwrap();
        assert_eq!(geometry.sparse_axis().unwrap_or(0), expected);
    }
}
