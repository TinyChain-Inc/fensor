use crate::sparse::SparseCell;
use crate::{Tensor, TensorElement, TensorFileEntry};

pub(in crate::tensor) async fn sparse_row<F: TensorFileEntry<T>, T: TensorElement>(
    tensor: &Tensor<F, T>,
    id: u64,
) -> (freqfs::FileReadGuardOwned<F, crate::SparseNode<T>>, usize) {
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
        let node = entry
            .as_file()
            .unwrap()
            .clone()
            .into_read::<crate::SparseNode<T>>()
            .await
            .unwrap();

        let b_tree::Node::Leaf(rows) = &*node else {
            continue;
        };

        if let Some(slot) = rows
            .iter()
            .position(|row| row.first() == Some(&SparseCell::Key(id)))
        {
            return (node, slot);
        }
    }

    panic!("missing sparse fixture row {id}");
}

pub(in crate::tensor) async fn corrupt_sparse_row<F: TensorFileEntry<T>, T: TensorElement>(
    tensor: &Tensor<F, T>,
    id: u64,
    mutate: impl FnOnce(&mut Vec<SparseCell<T>>),
) {
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
            .write::<crate::SparseNode<T>>()
            .await
            .unwrap();

        if let b_tree::Node::Leaf(rows) = &mut *node {
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

pub(in crate::tensor) fn block_index<F, T>(
    owner: &super::SparseBlocks<F, T>,
) -> &super::Blocks<F, T> {
    &owner.inner.blocks
}

pub(in crate::tensor) fn files<F, T>(
    storage: &crate::tensor::Storage<F, T>,
) -> &freqfs::DirLock<F> {
    match storage {
        crate::tensor::Storage::Dense(dense) => {
            crate::tensor::dense_storage::test_support::files(dense)
        }
        crate::tensor::Storage::Sparse(sparse) => &sparse.inner.metadata_dir,
    }
}
