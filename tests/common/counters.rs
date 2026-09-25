//! Test observations only; these counters never synchronize tensor storage.
//!
//! Adapter traffic is process-wide: reset/snapshot is for isolated benchmarks
//! run with one test thread. Concurrent correctness tests must not assert totals.
//! Structural assertions use local counters or the existing task-local metrics.

use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Default)]
pub struct Counter(AtomicU64);

impl Counter {
    pub const fn new() -> Self {
        Self(AtomicU64::new(0))
    }

    /// Increment an observation or sequence, returning its previous value.
    pub fn increment(&self) -> u64 {
        self.0.fetch_add(1, Ordering::Relaxed)
    }

    pub fn read(&self) -> u64 {
        self.0.load(Ordering::Relaxed)
    }

    pub fn reset(&self) {
        self.0.store(0, Ordering::Relaxed);
    }
}

static LOADS: Counter = Counter::new();

static SAVES: Counter = Counter::new();

pub fn record_load() {
    LOADS.increment();
}

pub fn record_save() {
    SAVES.increment();
}

pub fn reset_traffic() {
    LOADS.reset();
    SAVES.reset();
}

pub struct Traffic {
    pub loads: u64,
    pub saves: u64,
}

/// Observe independent totals; this is not a transactional snapshot.
pub fn snapshot_traffic() -> Traffic {
    Traffic {
        loads: LOADS.read(),
        saves: SAVES.read(),
    }
}
