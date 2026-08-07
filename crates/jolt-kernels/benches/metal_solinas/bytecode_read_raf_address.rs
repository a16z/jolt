use std::{env, hint::black_box, time::Duration, time::Instant};

use criterion::{BenchmarkId, Criterion, Throughput};
use jolt_field::AkitaField;
use jolt_kernels::metal::solinas::{
    bytecode_read_raf_address::{
        carrier::AddressMajorShape, oracle::HostAddressMajorCarrier, BytecodeAddressMajorConfig,
        BYTECODE_ADDRESS_MAJOR_STAGES,
    },
    SolinasMetal,
};

pub fn bench(c: &mut Criterion, context: &SolinasMetal) {
    let support = setting("JOLT_METAL_BYTECODE_ADDRESS_SUPPORT", 10);
    let warmup_ms = setting("JOLT_METAL_BYTECODE_ADDRESS_WARMUP_MS", 1_000);
    let measurement_ms = setting("JOLT_METAL_BYTECODE_ADDRESS_MEASUREMENT_MS", 3_000);
    let mut group = c.benchmark_group("metal_sumcheck/bytecode_read_raf_address_major");
    let _ = group
        .sample_size(10)
        .warm_up_time(Duration::from_millis(warmup_ms as u64))
        .measurement_time(Duration::from_millis(measurement_ms as u64));

    for log_rows in logs() {
        let shape = AddressMajorShape::production(log_rows as u32)
            .expect("bytecode address-major benchmark shape should be valid");
        let carrier = HostAddressMajorCarrier::balanced_probe(shape, support, 3)
            .expect("balanced bytecode address-major carrier should build");
        let (e_lo, e_hi) = tables(shape);
        let outer_length = shape
            .outer_length()
            .expect("outer length should be representable");
        let rows = shape.rows().expect("row count should be representable");
        let useful_products = 4usize
            .checked_mul(rows)
            .and_then(|products| {
                9usize
                    .checked_mul(support * outer_length)
                    .and_then(|outer_products| products.checked_add(outer_products))
            })
            .expect("useful-product count should fit");
        let mut reference = None;

        for outer_tiles in tiles().into_iter().filter(|tiles| *tiles <= outer_length) {
            let setup_started = Instant::now();
            let invocation = context
                .prepare_bytecode_address_major_probe(
                    &carrier,
                    &e_lo,
                    &e_hi,
                    BytecodeAddressMajorConfig { outer_tiles },
                )
                .expect("bytecode address-major worker should prepare");
            let setup_wall = setup_started.elapsed();
            let wall_started = Instant::now();
            let (output, gpu_active) = invocation
                .execute_timed()
                .expect("bytecode address-major worker should execute");
            let first_wall = wall_started.elapsed();
            if let Some(expected) = &reference {
                assert_eq!(&output, expected, "outer tiling changed the exact output");
            } else {
                reference = Some(output);
            }

            eprintln!(
                "bytecode-address-major log={log_rows} support={support} tiles={outer_tiles} setup={setup_wall:?} first-wall={first_wall:?} gpu-active={gpu_active:?} owned-bytes={} carrier-bytes={} partial-bytes={} tg-bytes={} static-tg-bytes={} max-threads={}",
                invocation.storage().owned_bytes,
                invocation.storage().carrier_bytes,
                invocation.storage().partial_bytes,
                invocation.threadgroup_memory_bytes(),
                invocation
                    .worker_pipeline_limits()
                    .static_threadgroup_memory_length,
                invocation
                    .worker_pipeline_limits()
                    .max_total_threads_per_threadgroup,
            );

            let case = format!("log{log_rows}_support{support}_tiles{outer_tiles}");
            let _ = group.throughput(Throughput::Elements(useful_products as u64));
            let _ = group.bench_function(BenchmarkId::new("metal_wall_resident", &case), |bench| {
                bench.iter(|| {
                    black_box(
                        invocation
                            .execute()
                            .expect("bytecode address-major worker should execute"),
                    )
                });
            });
            let _ =
                group.bench_function(BenchmarkId::new("metal_active_resident", &case), |bench| {
                    bench.iter_custom(|iterations| {
                        let mut active = Duration::ZERO;
                        for _ in 0..iterations {
                            active += invocation
                                .execute_timed()
                                .expect("bytecode address-major worker should execute")
                                .1;
                        }
                        active
                    });
                });
        }
    }
    group.finish();
}

fn tables(shape: AddressMajorShape) -> (Vec<Vec<AkitaField>>, Vec<Vec<AkitaField>>) {
    let inner_length = shape
        .inner_length()
        .expect("inner length should be representable");
    let outer_length = shape
        .outer_length()
        .expect("outer length should be representable");
    let e_lo = (0..BYTECODE_ADDRESS_MAJOR_STAGES)
        .map(|stage| {
            (0..inner_length)
                .map(|inner| {
                    AkitaField::from_u64((1 + 17 * stage as u64 + 5 * inner as u64) & 0x000f_ffff)
                })
                .collect()
        })
        .collect();
    let e_hi = (0..BYTECODE_ADDRESS_MAJOR_STAGES)
        .map(|stage| {
            (0..outer_length)
                .map(|outer| {
                    AkitaField::from_u64((3 + 19 * stage as u64 + 7 * outer as u64) & 0x000f_ffff)
                })
                .collect()
        })
        .collect();
    (e_lo, e_hi)
}

fn logs() -> Vec<usize> {
    list_setting("JOLT_METAL_BYTECODE_ADDRESS_LOGS", &[20, 22])
}

fn tiles() -> Vec<usize> {
    list_setting("JOLT_METAL_BYTECODE_ADDRESS_TILES", &[1, 2, 4, 8])
}

fn list_setting(name: &str, default: &[usize]) -> Vec<usize> {
    env::var(name)
        .ok()
        .map(|value| {
            value
                .split(',')
                .map(|entry| {
                    entry
                        .parse::<usize>()
                        .unwrap_or_else(|_| panic!("{name} entries must be integers"))
                })
                .collect()
        })
        .unwrap_or_else(|| default.to_vec())
}

fn setting(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .map(|value| {
            value
                .parse::<usize>()
                .unwrap_or_else(|_| panic!("{name} must be an integer"))
        })
        .unwrap_or(default)
}
