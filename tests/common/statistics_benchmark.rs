//! Warm-cache statistics including expression construction and full consumption.

use std::hint::black_box;
use std::time::Instant;

use fensor::{StatisticsElement, TensorFileEntry, TensorStatistics};
use ha_ndarray::{axes, shape};

use super::benchmark;
use super::common::{FsEntry, fixture};

async fn case<T: StatisticsElement>(dtype: &str, value: impl Fn(u64) -> T)
where
    FsEntry: TensorFileEntry<T>,
{
    for sparse in [false, true] {
        for capacity in [256, 4096] {
            let name = format!("statistics-{dtype}-sparse-{sparse}-capacity-{capacity}");
            let values = (0..65536)
                .map(|i| {
                    if sparse && i % 8191 != 0 {
                        T::ZERO
                    } else {
                        value(i)
                    }
                })
                .collect::<Vec<_>>();

            let (_root, tensor) = fixture::source(
                &name,
                shape![16, 4096],
                fixture::layout(sparse),
                capacity,
                64 * 1024 * 1024,
                values,
            )
            .await;

            let source = tensor.view();
            macro_rules! measure {
                ($operation:literal, $body:expr) => {{
                    black_box($body);
                    let mut samples = Vec::with_capacity(7);
                    for _ in 0..7 {
                        let start = Instant::now();
                        black_box($body);
                        samples.push(start.elapsed().as_nanos());
                    }

                    samples.sort_unstable();
                    benchmark::record(&name, $operation, "warm", "elapsed", "ns", samples[3]);
                }};
            }

            measure!("mean-all", source.mean_all().await.unwrap());
            measure!("norm-all", source.norm_all().await.unwrap());
            measure!("std-all", source.std_all().await.unwrap());
            measure!("mean-axis", {
                let result = source.mean(axes![1], false).await.unwrap();
                benchmark::consume(&result, "row").await
            });

            measure!("norm-axis", {
                let result = source.norm(axes![1], false).await.unwrap();
                benchmark::consume(&result, "row").await
            });

            measure!("std-axis", {
                let result = source.std(axes![1], false).await.unwrap();
                benchmark::consume(&result, "row").await
            });
        }
    }
}

#[tokio::test]
#[ignore = "isolated release measurements, not a timing gate"]
async fn statistical_reductions() {
    case("f64", |i| (i % 17) as f64 - 8.).await;
    #[cfg(feature = "complex")]
    case("c64", |i| {
        fensor::complex::Complex64::new((i % 17) as f64 - 8., (i % 11) as f64 - 5.)
    })
    .await;
}
