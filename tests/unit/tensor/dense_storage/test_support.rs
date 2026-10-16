pub(in crate::tensor) fn files<F>(owner: &super::DenseStorage<F>) -> &freqfs::DirLock<F> {
    &owner.blocks
}
