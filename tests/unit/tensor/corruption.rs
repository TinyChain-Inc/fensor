use crate::sparse::SparseCell;
use crate::{Tensor, TensorElement, TensorFileEntry};

async fn corrupt_sparse_row<F: TensorFileEntry<T>, T: TensorElement>(
    tensor: &Tensor<F, T>,
    id: u64,
    mutate: impl FnOnce(&mut Vec<SparseCell<T>>),
) {
    let owner = tensor.storage.sparse().unwrap();
    let _guard = owner.gate.write().await;
    let values = tensor
        .directory
        .read()
        .await
        .get_dir("values")
        .cloned()
        .unwrap();
    let primary = values.read().await.get_dir("primary").cloned().unwrap();
    let primary = primary.read().await;
    for (_, entry) in primary.iter() {
        let mut node = entry
            .as_file()
            .unwrap()
            .write::<crate::SparseNode<T>>(0)
            .await
            .unwrap();
        if let b_table::Node::Leaf(rows) = &mut *node {
            for row in rows {
                if row.first() == Some(&SparseCell::Key(id)) {
                    mutate(row);
                    return;
                }
            }
        }
    }
    panic!("missing sparse corruption fixture row {id}");
}

pub(crate) async fn corrupt_sparse_key<F: TensorFileEntry<T>, T: TensorElement>(
    tensor: &Tensor<F, T>,
    id: u64,
    replacement: u64,
) {
    corrupt_sparse_row(tensor, id, |row| row[0] = SparseCell::Key(replacement)).await;
}

/// Clear a required dense payload while retaining its native row and key.
pub(crate) async fn corrupt_sparse_payload<F: TensorFileEntry<T>, T: TensorElement>(
    tensor: &Tensor<F, T>,
    id: u64,
) {
    corrupt_sparse_row(tensor, id, |row| {
        let SparseCell::Payload(values) = &mut row[1] else {
            panic!("native sparse payload")
        };
        values.clear();
    })
    .await;
}
