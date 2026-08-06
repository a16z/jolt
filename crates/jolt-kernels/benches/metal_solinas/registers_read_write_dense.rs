use std::{env, hint::black_box, time::Duration, time::Instant};

use criterion::{BenchmarkId, Criterion, Throughput};
use jolt_field::AkitaField;
use jolt_kernels::metal::solinas::{
    Fp128, RegistersRwDenseStateWords, SolinasMetal, REGISTERS_RW_DENSE_COLUMNS,
};

const DEFAULT_TRACE_ELEMENTS: usize = 1 << 26;
const DENSE_PREFIX_BINDS: usize = 8;
const E_IN_LENGTH: usize = 8;
const ACTIVE_80_PERCENT_CAP: Duration = Duration::from_nanos(6_704_877);

pub fn bench(c: &mut Criterion, context: &SolinasMetal) {
    let trace_elements = trace_elements();
    let source_rows = trace_elements >> DENSE_PREFIX_BINDS;
    let pair_count = source_rows / 4;
    assert_eq!(pair_count % E_IN_LENGTH, 0);
    let e_out_length = pair_count / E_IN_LENGTH;

    let state = RegistersRwDenseStateWords {
        val: Fp128::from_u128(1),
        ra: Fp128::from_u128(2),
        wa: Fp128::from_u128(3),
    };
    let source = vec![state; source_rows * REGISTERS_RW_DENSE_COLUMNS];
    let source_inc = vec![AkitaField::from_u64(4); source_rows];
    let e_in = vec![AkitaField::one(); E_IN_LENGTH];
    let e_out = vec![AkitaField::one(); e_out_length];
    let challenge = AkitaField::from_u64(0xfeed_beef);

    let setup_start = Instant::now();
    let invocation = context
        .prepare_registers_rw_dense_round(&source, &source_inc, &e_in, &e_out, challenge)
        .expect("registers read/write dense round should prepare");
    let setup_wall = setup_start.elapsed();
    drop((source, source_inc, e_in, e_out));

    let (message, cold_active) = invocation
        .execute_timed()
        .expect("registers read/write dense round should execute");
    let expected_zero = AkitaField::from_u64((pair_count * REGISTERS_RW_DENSE_COLUMNS * 17) as u64);
    assert_eq!(message, [expected_zero, AkitaField::zero()]);
    assert_eq!(invocation.execute_device_buffer_allocations(), 0);
    assert_ne!(
        invocation.source_state_allocation_identity(),
        invocation.destination_state_allocation_identity()
    );

    eprintln!(
        "registers-rw-dense trace_elements={trace_elements} source_rows={source_rows} \
         storage_bytes={} setup_ms={:.3} cold_active_ms={:.3} active_cap_ms={:.3} \
         tew={} max_threads={} tg={} tg_mem={}",
        invocation.storage().total_bytes,
        setup_wall.as_secs_f64() * 1e3,
        cold_active.as_secs_f64() * 1e3,
        ACTIVE_80_PERCENT_CAP.as_secs_f64() * 1e3,
        invocation.message_pipeline_limits().thread_execution_width,
        invocation
            .message_pipeline_limits()
            .max_total_threads_per_threadgroup,
        invocation.threads_per_threadgroup(),
        invocation.dynamic_threadgroup_memory_bytes(),
    );

    let warmup_ms = setting("JOLT_METAL_RW_DENSE_WARMUP_MS", 500);
    let measurement_ms = setting("JOLT_METAL_RW_DENSE_MEASUREMENT_MS", 1_500);
    let suffix = format!("trace_n{trace_elements}_round9_tg128");
    let mut group = c.benchmark_group("metal_sumcheck/registers_read_write_dense_round");
    let _ = group
        .sample_size(10)
        .warm_up_time(Duration::from_millis(warmup_ms))
        .measurement_time(Duration::from_millis(measurement_ms))
        .throughput(Throughput::Elements(trace_elements as u64));
    let _ = group.bench_function(BenchmarkId::new("metal_wall_resident", &suffix), |bench| {
        bench.iter(|| {
            black_box(
                invocation
                    .execute()
                    .expect("registers read/write dense round should execute"),
            )
        });
    });
    let _ = group.bench_function(
        BenchmarkId::new("metal_active_resident", &suffix),
        |bench| {
            bench.iter_custom(|iterations| {
                let mut active = Duration::ZERO;
                for _ in 0..iterations {
                    let (message, elapsed) = invocation
                        .execute_timed()
                        .expect("timed registers read/write dense round should execute");
                    let _ = black_box(message);
                    active += elapsed;
                }
                active
            });
        },
    );
    group.finish();
}

fn trace_elements() -> usize {
    let elements =
        env::var("JOLT_SOLINAS_BENCH_ELEMENTS").map_or(DEFAULT_TRACE_ELEMENTS, |value| {
            value
                .parse()
                .expect("JOLT_SOLINAS_BENCH_ELEMENTS should be a positive integer")
        });
    assert!(
        elements.is_power_of_two() && elements >= 1 << 18,
        "dense registers read/write trace size must be a power of two at least 2^18"
    );
    elements
}

fn setting(name: &str, default: u64) -> u64 {
    env::var(name).map_or(default, |value| {
        value
            .parse()
            .unwrap_or_else(|_| panic!("{name} should be a nonnegative integer"))
    })
}
