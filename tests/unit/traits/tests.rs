use std::cell::Cell;

use super::coordinate_batches;
use crate::expression::MAX_BATCH_ELEMENTS;

#[test]
fn batching_only_consumes_its_bounded_prefix() {
    let consumed = Cell::new(0);
    let coords = (0..1_000_000_000u64).map(|i| {
        consumed.set(consumed.get() + 1);
        vec![i]
    });
    let mut batches = coordinate_batches(coords);
    assert_eq!(consumed.get(), 0);
    assert_eq!(batches.next().unwrap().len(), MAX_BATCH_ELEMENTS);
    assert_eq!(consumed.get(), MAX_BATCH_ELEMENTS);
    drop(batches);
    assert_eq!(consumed.get(), MAX_BATCH_ELEMENTS);
}
