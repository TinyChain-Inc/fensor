//! Resident-cache and cache-pressure reads; OS page-cache state is uncontrolled.

use fensor::{AxisRange, TensorGeometry, TensorRead, TensorReduceAll, TensorTransform};
use futures::TryStreamExt;
use ha_ndarray::range;

use super::{benchmark, benchmark_source};

pub async fn run() {
    let smoke = std::env::var_os("FENSOR_BENCH_SMOKE").is_some();
    let rows = if smoke { 4 } else { 64 };
    let columns = if smoke { 7 } else { 257 };
    let repeats = if smoke { 1 } else { 16 };

    for capacity in [128, 4096] {
        for cache in [32768, 1_000_000] {
            let (root, tensor) =
                benchmark_source::source(rows, columns, false, capacity, 1, cache).await;
            let name = format!("read_blocks_{capacity}_cache{cache}");
            let temperature = if cache == 1_000_000 {
                "resident"
            } else {
                "streaming"
            };
            let transposed = tensor.view().transpose(None).unwrap();
            let selected = tensor
                .view()
                .slice(range![
                    AxisRange::In(0, rows as u64, 1),
                    AxisRange::In(0, 1, 1)
                ])
                .unwrap();

            for operation in ["row", "transpose", "selection", "sum"] {
                benchmark::measure(&name, operation, temperature, async {
                    let result = async {
                        let mut count = 0;
                        for _ in 0..repeats {
                            count += match operation {
                                "row" => consume(&tensor).await?,
                                "transpose" => consume(&transposed).await?,
                                "selection" => consume(&selected).await?,
                                "sum" => {
                                    std::hint::black_box(tensor.sum_all().await?);
                                    1
                                }
                                _ => unreachable!(),
                            };
                        }
                        Ok::<_, fensor::Error>(count)
                    }
                    .await;
                    let (admitted, count) = match result {
                        Ok(count) => (true, count),
                        Err(fensor::Error::Io(error))
                            if error.kind() == std::io::ErrorKind::ResourceBusy =>
                        {
                            (false, 0)
                        }
                        Err(error) => panic!("read failed for {name}: {error}"),
                    };

                    benchmark::record(
                        &name,
                        operation,
                        temperature,
                        "admitted",
                        "count",
                        u128::from(admitted),
                    );
                    benchmark::record(
                        &name,
                        operation,
                        temperature,
                        "capacity_rejected",
                        "count",
                        u128::from(!admitted),
                    );
                    count
                })
                .await;
            }

            benchmark::copy(&name, &transposed, 128, 1_000_000, true).await;

            // Small point requests remain a borrowed native path.
            benchmark::measure(&name, "point", temperature, async {
                for _ in 0..repeats {
                    std::hint::black_box(tensor.read_value(&[0, 0]).await.unwrap());
                }
                repeats
            })
            .await;

            assert_eq!(tensor.size().unwrap(), (rows * columns) as u64);
            drop(selected);
            drop(transposed);
            drop(tensor);
            drop(root);
        }
    }
}

async fn consume<V: TensorRead>(view: &V) -> fensor::Result<usize> {
    let mut blocks = view.read_blocks()?;
    let mut count = 0;

    while let Some(values) = blocks.try_next().await? {
        count += values.len();
        std::hint::black_box(values);
    }

    assert_eq!(count as u64, view.size()?);
    Ok(count)
}
