//! Test-only per-consumption observations; absent from production builds.

use std::cell::RefCell;

#[derive(Default, Debug)]
pub(crate) struct Metrics {
    pub occupancy_analyses: usize,
    pub descriptor_lookups: usize,
    pub padding_walks: usize,
    pub mapped_runs: usize,
    pub slice_requests: usize,
    pub index_entries: usize,
    pub reduction_calls: usize,
    pub runs: usize,
    pub coordinate_resolutions: usize,
    pub expanded_coordinates: usize,
    pub boundary_decodes: usize,
    pub borrows: usize,
    pub logical_payload_reads: usize,
    pub requested: usize,
    pub borrowed: usize,
    pub backend_calls: usize,
}

tokio::task_local! {
    pub(crate) static CURRENT: RefCell<Metrics>;
}

pub(crate) fn record(update: impl FnOnce(&mut Metrics)) {
    let _ = CURRENT.try_with(|m| update(&mut m.borrow_mut()));
}
