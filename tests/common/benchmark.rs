//! Shared benchmark stream consumption; callers own timing and observation.

use fensor::TensorRead;
use futures::TryStreamExt;

pub async fn consume<V: TensorRead>(view: &V, mode: &str) -> usize {
    let mut count = 0;
    match mode {
        "row" => {
            let mut stream = view.read_blocks().unwrap();

            while let Some(values) = stream.try_next().await.unwrap() {
                count += values.len();
                std::hint::black_box(values);
            }
        }
        "coordinate" => {
            let mut stream = view.read_coordinate_blocks().unwrap();

            while let Some((coords, values)) = stream.try_next().await.unwrap() {
                count += values.len();
                std::hint::black_box((coords, values));
            }
        }
        _ => panic!("unknown benchmark stream mode {mode}"),
    }
    count
}
