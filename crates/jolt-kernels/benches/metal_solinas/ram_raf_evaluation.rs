use std::{env, hint::black_box, mem::size_of, time::Duration, time::Instant};

use criterion::{BenchmarkId, Criterion, Throughput};
use jolt_field::AkitaField;
use jolt_kernels::metal::solinas::{
    dense_pushforward_oracle, RamRafAffineTail, RamRafConfig, SolinasMetal, RAM_RAF_ADDRESS_DOMAIN,
    RAM_RAF_NO_ACCESS, RAM_RAF_TARGET_CPU_NS,
};
use jolt_poly::EqPolynomial;
use rayon::prelude::*;

const DEFAULT_ELEMENTS: [usize; 3] = [1 << 20, 1 << 22, 1 << 24];
const VALIDATION_ROWS: usize = 1 << 16;

pub fn bench(c: &mut Criterion, context: &SolinasMetal) {
    validate(context);

    let warmup_ms = setting("JOLT_METAL_RAM_RAF_WARMUP_MS", 1_000);
    let measurement_ms = setting("JOLT_METAL_RAM_RAF_MEASUREMENT_MS", 3_000);
    let threads = setting("JOLT_METAL_RAM_RAF_THREADS", 1_024);
    let topology = env::var("JOLT_METAL_RAM_RAF_TOPOLOGY").unwrap_or_else(|_| "random".into());
    let observe_only = env::var_os("JOLT_METAL_RAM_RAF_OBSERVE_ONLY").is_some();
    let mut group = c.benchmark_group("metal_sumcheck/ram_raf_evaluation");
    let _ = group
        .sample_size(10)
        .warm_up_time(Duration::from_millis(warmup_ms as u64))
        .measurement_time(Duration::from_millis(measurement_ms as u64));

    for rows in cases() {
        let log_t = rows.ilog2() as usize;
        let config = RamRafConfig {
            threads,
            trace_cutoff: 1 << 15,
            ..RamRafConfig::default()
        };
        let generation_start = Instant::now();
        let addresses = addresses(rows, &topology);
        let generation_wall = generation_start.elapsed();
        let point = values(log_t, 0x243f_6a88_85a3_08d3);

        let upload_start = Instant::now();
        let plane = context
            .prepare_ram_raf_addresses(&addresses, config)
            .expect("RAM RAF address plane should prepare");
        let upload_wall = upload_start.elapsed();
        let address_storage_id = plane.storage_id();
        let setup_start = Instant::now();
        let sequence = context
            .prepare_ram_raf_sequence(plane, &point, config)
            .expect("RAM RAF sequence should prepare");
        let setup_wall = setup_start.elapsed();
        assert_eq!(sequence.address_storage_id(), address_storage_id);
        assert_eq!(sequence.round_device_buffer_allocations(), 0);

        let cold_start = Instant::now();
        let cold = sequence
            .execute_timed()
            .expect("RAM RAF cold pushforward should execute");
        let cold_wall = cold_start.elapsed();
        let warm_start = Instant::now();
        let warm = sequence
            .execute_timed()
            .expect("RAM RAF warm pushforward should execute");
        let warm_wall = warm_start.elapsed();
        assert_eq!(warm.masses, cold.masses);
        assert_eq!(warm.counters, cold.counters);

        let challenges = values(
            RAM_RAF_ADDRESS_DOMAIN.ilog2() as usize,
            0x1319_8a2e_0370_7344,
        );
        let tail_start = Instant::now();
        let tail_output = run_tail(warm.masses.clone(), &challenges);
        let tail_wall = tail_start.elapsed();
        let hybrid_no_fs_wall = warm_wall + tail_wall;
        let _ = black_box(tail_output);
        eprintln!(
            "ram-raf n={rows} topology={topology} generation={generation_wall:?} upload={upload_wall:?} setup={setup_wall:?} cold-wall={cold_wall:?} cold-active={:?} warm-wall={warm_wall:?} warm-active={:?} tail-no-fs={tail_wall:?} hybrid-no-fs={hybrid_no_fs_wall:?} accessed={} live-subtotals={} resident-address-bytes={} sequence-bytes={} fold-tew={} fold-max-threads={} fold-static-tgmem={} frozen-cpu-ms={:.6} hybrid-no-fs-ratio={:.6}",
            cold.gpu_active,
            warm.gpu_active,
            warm.counters.accessed_rows,
            warm.counters.nonzero_subtotals,
            rows * size_of::<u32>(),
            sequence.storage_plan().sequence_owned_bytes,
            sequence.fold_pipeline_limits().thread_execution_width,
            sequence
                .fold_pipeline_limits()
                .max_total_threads_per_threadgroup,
            sequence
                .fold_pipeline_limits()
                .static_threadgroup_memory_length,
            RAM_RAF_TARGET_CPU_NS as f64 / 1e6,
            RAM_RAF_TARGET_CPU_NS as f64 / hybrid_no_fs_wall.as_nanos() as f64,
        );

        if observe_only {
            continue;
        }
        let suffix = format!("n{rows}_{topology}_t{}", sequence.fold_threads());
        let _ = group.throughput(Throughput::Elements(rows as u64));
        let _ = group.bench_function(
            BenchmarkId::new("metal_wall_pushforward", &suffix),
            |bench| {
                bench.iter(|| {
                    black_box(
                        sequence
                            .execute_timed()
                            .expect("RAM RAF pushforward should execute")
                            .masses,
                    )
                });
            },
        );
        let _ = group.bench_function(
            BenchmarkId::new("metal_active_pushforward", &suffix),
            |bench| {
                bench.iter_custom(|iterations| {
                    let mut active = Duration::ZERO;
                    for _ in 0..iterations {
                        active += sequence
                            .execute_timed()
                            .expect("RAM RAF pushforward should execute")
                            .gpu_active;
                    }
                    active
                });
            },
        );
        let _ = group.bench_function(
            BenchmarkId::new("metal_wall_hybrid_no_fs", &suffix),
            |bench| {
                bench.iter(|| {
                    let observation = sequence
                        .execute_timed()
                        .expect("RAM RAF pushforward should execute");
                    black_box(run_tail(observation.masses, &challenges))
                });
            },
        );
    }
    group.finish();
}

fn validate(context: &SolinasMetal) {
    let rows = VALIDATION_ROWS;
    let config = RamRafConfig {
        trace_cutoff: 1 << 15,
        ..RamRafConfig::default()
    };
    let addresses = addresses(rows, "mixed");
    let point = values(rows.ilog2() as usize, 0xa409_3822_299f_31d0);
    let plane = context
        .prepare_ram_raf_addresses(&addresses, config)
        .expect("validation address plane should prepare");
    let sequence = context
        .prepare_ram_raf_sequence(plane, &point, config)
        .expect("validation sequence should prepare");
    let observation = sequence
        .execute_timed()
        .expect("validation pushforward should execute");
    let equality = EqPolynomial::<AkitaField>::evals(&point, None);
    let expected = dense_pushforward_oracle(&addresses, &equality, RAM_RAF_ADDRESS_DOMAIN)
        .expect("validation oracle should execute");
    assert_eq!(observation.masses, expected);
    let challenges = values(
        RAM_RAF_ADDRESS_DOMAIN.ilog2() as usize,
        0x082e_fa98_ec4e_6c89,
    );
    let _ = run_tail(observation.masses, &challenges);
}

fn run_tail(
    masses: Vec<AkitaField>,
    challenges: &[AkitaField],
) -> jolt_kernels::metal::solinas::RamRafTailOutput<AkitaField> {
    let mut tail = RamRafAffineTail::new(masses, 0).expect("RAM RAF tail should initialize");
    assert_eq!(tail.remaining_rounds(), challenges.len());
    let mut claim = tail.input_claim();
    for &challenge in challenges {
        let coefficients = tail
            .message(claim)
            .expect("RAM RAF tail message should match its claim")
            .coefficients();
        claim = coefficients[0] + challenge * (coefficients[1] + challenge * coefficients[2]);
        tail.bind(challenge)
            .expect("RAM RAF tail should have another round");
    }
    let output = tail.output().expect("RAM RAF tail should be fully bound");
    assert_eq!(output.unmap_address * output.ram_ra, claim);
    output
}

fn addresses(rows: usize, topology: &str) -> Vec<u32> {
    (0..rows)
        .into_par_iter()
        .map(|index| match topology {
            "none" => RAM_RAF_NO_ACCESS,
            "one-hot" => 0,
            "mixed" if index % 8 == 0 => RAM_RAF_NO_ACCESS,
            "mixed" => ((splitmix(index as u64) as usize) % RAM_RAF_ADDRESS_DOMAIN) as u32,
            "random" => ((splitmix(index as u64) as usize) % RAM_RAF_ADDRESS_DOMAIN) as u32,
            other => panic!("unknown RAM RAF topology {other}"),
        })
        .collect()
}

fn values(count: usize, seed: u64) -> Vec<AkitaField> {
    (0..count)
        .map(|index| AkitaField::from_u64(splitmix(seed ^ index as u64) & ((1 << 56) - 1)))
        .collect()
}

fn splitmix(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn cases() -> Vec<usize> {
    env::var("JOLT_SOLINAS_BENCH_ELEMENTS").map_or_else(
        |_| DEFAULT_ELEMENTS.to_vec(),
        |value| {
            vec![value
                .parse()
                .expect("JOLT_SOLINAS_BENCH_ELEMENTS must be an integer")]
        },
    )
}

fn setting(name: &str, default: usize) -> usize {
    env::var(name).map_or(default, |value| {
        value
            .parse()
            .unwrap_or_else(|_| panic!("{name} must be an integer"))
    })
}
