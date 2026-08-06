use std::{env, hint::black_box, time::Duration, time::Instant};

use criterion::{BenchmarkId, Criterion, Throughput};
use jolt_field::AkitaField;
use jolt_kernels::metal::solinas::{
    evaluate_product_uniskip_extensions_cpu, product_uniskip_reference, ProductRemainderRow,
    ProductUniskipConfig, SolinasMetal,
};
use jolt_poly::EqPolynomial;
use rayon::prelude::*;

const DEFAULT_ELEMENTS: [usize; 3] = [1 << 16, 1 << 20, 1 << 22];
const VALIDATION_ELEMENTS: usize = 1 << 8;

pub fn bench(c: &mut Criterion, context: &SolinasMetal) {
    validate(context);

    let threads = setting("JOLT_METAL_PRODUCT_UNISKIP_THREADS", 128);
    let warmup_ms = setting("JOLT_METAL_PRODUCT_UNISKIP_WARMUP_MS", 2_000);
    let measurement_ms = setting("JOLT_METAL_PRODUCT_UNISKIP_MEASUREMENT_MS", 5_000);
    let metal_only = env::var_os("JOLT_METAL_PRODUCT_UNISKIP_METAL_ONLY").is_some();
    let config = ProductUniskipConfig {
        threads_per_threadgroup: Some(threads),
    };
    let mut group = c.benchmark_group("metal_sumcheck/product_uniskip");
    let _ = group
        .sample_size(10)
        .warm_up_time(Duration::from_millis(warmup_ms as u64))
        .measurement_time(Duration::from_millis(measurement_ms as u64));

    for elements in cases() {
        let rows = rows(elements);
        let (e_in, e_out) = equality_weights(elements);
        let upload_start = Instant::now();
        let resident = context
            .prepare_product_remainder_rows(&rows)
            .expect("product rows should prepare once");
        let upload_wall = upload_start.elapsed();
        let invocation = context
            .prepare_product_uniskip(&resident, &e_in, &e_out, config)
            .expect("product uni-skip should prepare");
        assert_eq!(
            invocation.row_allocation_identity(),
            resident.allocation_identity()
        );
        assert_eq!(invocation.execute_device_buffer_allocations(), 0);

        let cpu_start = Instant::now();
        let expected = evaluate_product_uniskip_extensions_cpu(&rows, &e_in, &e_out)
            .expect("optimized CPU two-node evaluation should succeed");
        let cpu_one_shot = cpu_start.elapsed();
        let metal_start = Instant::now();
        let (actual, metal_active) = invocation
            .execute_timed()
            .expect("product uni-skip should execute");
        let metal_wall = metal_start.elapsed();
        assert_eq!(actual, expected);

        eprintln!(
            "product-uniskip n={elements} upload={upload_wall:?} cpu-one-shot={cpu_one_shot:?} metal-wall={metal_wall:?} metal-active={metal_active:?} resident-bytes={} row-allocation={} useful-multiplications={}",
            invocation.storage_layout().resident_bytes(),
            invocation.row_allocation_identity(),
            invocation.useful_multiplications(),
        );

        let suffix = format!("n{elements}_tg{threads}");
        let _ = group.throughput(Throughput::Elements(
            invocation.useful_multiplications() as u64
        ));
        let cpu_first = env::var("JOLT_SOLINAS_BENCH_ORDER").as_deref() == Ok("cpu-first");
        let add_cpu =
            |group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>| {
                let _ = group.bench_function(
                    BenchmarkId::new("cpu_optimized_two_node", &suffix),
                    |bench| {
                        bench.iter(|| {
                            black_box(
                                evaluate_product_uniskip_extensions_cpu(&rows, &e_in, &e_out)
                                    .expect("optimized CPU two-node evaluation should succeed"),
                            )
                        });
                    },
                );
            };
        let add_metal = |group: &mut criterion::BenchmarkGroup<
            '_,
            criterion::measurement::WallTime,
        >| {
            let _ =
                group.bench_function(BenchmarkId::new("metal_wall_resident", &suffix), |bench| {
                    bench.iter(|| {
                        black_box(
                            invocation
                                .execute()
                                .expect("product uni-skip should execute"),
                        )
                    });
                });
            let _ = group.bench_function(
                BenchmarkId::new("metal_active_resident", &suffix),
                |bench| {
                    bench.iter_custom(|iterations| {
                        let mut active = Duration::ZERO;
                        for _ in 0..iterations {
                            active += invocation
                                .execute_timed()
                                .expect("timed product uni-skip should execute")
                                .1;
                        }
                        active
                    });
                },
            );
        };
        if metal_only {
            add_metal(&mut group);
        } else if cpu_first {
            add_cpu(&mut group);
            add_metal(&mut group);
        } else {
            add_metal(&mut group);
            add_cpu(&mut group);
        }
    }
    group.finish();
}

fn validate(context: &SolinasMetal) {
    let rows = rows(VALIDATION_ELEMENTS);
    let (e_in, e_out) = equality_weights(VALIDATION_ELEMENTS);
    let oracle = product_uniskip_reference::extended_node_values(&rows, &e_in, &e_out)
        .expect("independent product uni-skip oracle should succeed");
    assert_eq!(
        evaluate_product_uniskip_extensions_cpu(&rows, &e_in, &e_out)
            .expect("optimized CPU product uni-skip should succeed"),
        oracle
    );
    let resident = context
        .prepare_product_remainder_rows(&rows)
        .expect("validation rows should prepare");
    let invocation = context
        .prepare_product_uniskip(&resident, &e_in, &e_out, ProductUniskipConfig::default())
        .expect("validation invocation should prepare");
    assert_eq!(
        invocation
            .execute()
            .expect("validation invocation should execute"),
        oracle
    );
}

fn equality_weights(elements: usize) -> (Vec<AkitaField>, Vec<AkitaField>) {
    let point = values(elements.ilog2() as usize, 0x6a09_e667_f3bc_c909);
    let split = point.len() / 2;
    (
        EqPolynomial::<AkitaField>::evals(&point[split..], None),
        EqPolynomial::<AkitaField>::evals(&point[..split], None),
    )
}

fn rows(elements: usize) -> Vec<ProductRemainderRow> {
    (0..elements)
        .into_par_iter()
        .map(|index| {
            let right_input = match index % 5 {
                0 => i128::MIN,
                1 => i128::MAX,
                2 => -1,
                3 => 0,
                _ => {
                    let low = splitmix(index as u64 ^ 0x243f_6a88_85a3_08d3) as u128;
                    let high = splitmix(index as u64 ^ 0x1319_8a2e_0370_7344) as u128;
                    ((high << 64) | low) as i128
                }
            };
            let selector = splitmix(index as u64 ^ 0xa409_3822_299f_31d0);
            ProductRemainderRow::new(
                splitmix(index as u64),
                right_input,
                selector & 1 != 0,
                selector & 2 != 0,
                splitmix(index as u64 ^ 0x082e_fa98_ec4e_6c89),
                selector & 4 != 0,
                selector & 8 != 0,
                selector & 16 != 0,
            )
        })
        .collect()
}

fn values(count: usize, seed: u64) -> Vec<AkitaField> {
    (0..count)
        .map(|index| AkitaField::from_u64(splitmix(seed ^ index as u64) & ((1u64 << 56) - 1)))
        .collect()
}

fn splitmix(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn setting(name: &str, default: usize) -> usize {
    env::var(name).map_or(default, |value| {
        value
            .parse()
            .unwrap_or_else(|_| panic!("{name} should be a positive integer"))
    })
}

fn cases() -> Vec<usize> {
    let cases = env::var("JOLT_SOLINAS_BENCH_ELEMENTS").map_or_else(
        |_| DEFAULT_ELEMENTS.to_vec(),
        |value| {
            vec![value
                .parse()
                .expect("JOLT_SOLINAS_BENCH_ELEMENTS should be a positive integer")]
        },
    );
    assert!(
        cases
            .iter()
            .all(|elements| elements.is_power_of_two() && *elements >= VALIDATION_ELEMENTS),
        "product uni-skip benchmark sizes must be powers of two at least 2^8"
    );
    cases
}
