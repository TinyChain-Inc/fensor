//! Test-only per-consumption observations; absent from production builds.

use std::cell::RefCell;
use std::time::{Duration, Instant};

#[derive(Default, Debug)]
pub(crate) struct Metrics {
    pub slice_requests: usize,
    pub index_entries: usize,
    pub reduction_calls: usize,
    pub runs: usize,
    pub coordinate_resolutions: usize,
    pub boundary_decodes: usize,
    pub lookups: usize,
    pub borrows: usize,
    pub requested: usize,
    pub borrowed: usize,
    pub backend_calls: usize,
    pub mapping: Duration,
    pub index: Duration,
    pub access: Duration,
    pub scatter: Duration,
    pub planning: Duration,
    pub operands: Duration,
    pub backend: Duration,
    pub accumulate: Duration,
}

tokio::task_local! {
    pub(crate) static CURRENT: RefCell<Metrics>;
}

pub(crate) fn record(update: impl FnOnce(&mut Metrics)) {
    let _ = CURRENT.try_with(|m| update(&mut m.borrow_mut()));
}

pub(crate) struct Timer(Instant, fn(&mut Metrics) -> &mut Duration);

impl Timer {
    pub fn new(field: fn(&mut Metrics) -> &mut Duration) -> Self {
        Self(Instant::now(), field)
    }
}

impl Drop for Timer {
    fn drop(&mut self) {
        record(|m| *(self.1)(m) += self.0.elapsed());
    }
}
