use std::cell::RefCell;

#[derive(Default, Debug)]
pub(crate) struct Metrics {
    pub groups: usize,
    pub replaced_blocks: usize,
    pub descriptor_writes: usize,
    pub max_descriptor_batch: usize,
    pub block_updates: usize,
    pub constructed_blocks: usize,
    pub staged_elements: usize,
    pub max_staging_batch: usize,
}

tokio::task_local! {
    pub(crate) static CURRENT: RefCell<Metrics>;
}

pub(crate) fn record(update: impl FnOnce(&mut Metrics)) {
    let _ = CURRENT.try_with(|metrics| update(&mut metrics.borrow_mut()));
}
