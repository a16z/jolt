use std::{env, hint::black_box, time::Duration, time::Instant};

use criterion::{BenchmarkId, Criterion, Throughput};
use jolt_field::AkitaField;
use jolt_kernels::metal::solinas::{
    ram_ra_claim_reduction::{
        RamRaClaimConfig, RamRaClaimQAccumulator, RamRaClaimQInvocation,
        RAM_RA_CLAIM_ADDRESS_DOMAIN, RAM_RA_CLAIM_NO_ACCESS, RAM_RA_CLAIM_TARGET_ACCESSED_ROWS,
        RAM_RA_CLAIM_TARGET_CPU_NS, RAM_RA_CLAIM_TARGET_FIVE_X_NS, RAM_RA_CLAIM_TARGET_ROWS,
    },
    SolinasMetal,
};

pub fn bench(c: &mut Criterion, context: &SolinasMetal) {
    let rows = trace_rows();
    let accessed_rows = accessed_rows(rows);
    let addresses = uniform_addresses(rows, accessed_rows);
    let r_address = point(RAM_RA_CLAIM_ADDRESS_DOMAIN.ilog2() as usize, 0xa11c_e001);
    let cycle_points = [
        point(rows.ilog2() as usize, 0xc1c1_0001),
        point(rows.ilog2() as usize, 0xc1c1_0002),
        point(rows.ilog2() as usize, 0xc1c1_0003),
    ];
    let cycle_refs = [
        cycle_points[0].as_slice(),
        cycle_points[1].as_slice(),
        cycle_points[2].as_slice(),
    ];
    let config = RamRaClaimConfig {
        trace_cutoff: rows,
        q_partitions: q_partitions(),
        q_accumulator: accumulator(),
        ..RamRaClaimConfig::default()
    };

    let setup_started = Instant::now();
    let resident = context
        .prepare_ram_ra_claim_addresses(&addresses)
        .expect("RAM RA claim address plane should prepare");
    drop(addresses);
    let invocation = context
        .prepare_ram_ra_claim_q(&resident, &r_address, cycle_refs, config)
        .expect("RAM RA claim Q invocation should prepare");
    let control = paired_control().then(|| {
        context
            .prepare_ram_ra_claim_q(
                &resident,
                &r_address,
                cycle_refs,
                RamRaClaimConfig {
                    q_accumulator: RamRaClaimQAccumulator::Explicit,
                    ..config
                },
            )
            .expect("RAM RA claim array control should prepare")
    });
    drop(resident);
    let setup_wall = setup_started.elapsed();

    let cold = invocation
        .execute_timed()
        .expect("RAM RA claim Q should execute");
    let warm = invocation
        .execute_timed()
        .expect("warm RAM RA claim Q should execute");
    assert_eq!(cold.q, warm.q);
    assert_eq!(cold.checksum, warm.checksum);
    if let Some(control) = &control {
        let observation = control
            .execute_timed()
            .expect("RAM RA claim array control should execute");
        assert_eq!(observation.q, warm.q);
        report_paired_screen(&invocation, control);
    }
    assert_eq!(warm.counters.q_accessed_rows as usize, accessed_rows);
    assert_eq!(warm.counters.q_invalid_rows, 0);
    assert_eq!(warm.counters.gather_invalid_rows, 0);
    assert_eq!(warm.counters.unsupported_dispatches, 0);
    assert_eq!(invocation.execute_device_buffer_allocations(), 0);
    assert_ne!(
        invocation.source_allocation_identity(),
        invocation.output_allocation_identity()
    );

    let plan = invocation.plan();
    let projection = invocation.projection();
    eprintln!(
        "ram-ra-claim-q rows={rows} accessed={accessed_rows} accumulator={} setup-ms={:.3} \
         cold-active-ms={:.3} warm-active-ms={:.3} warm-wall-ms={:.3} \
         useful-products={} perfect-cache-bytes={} shader-requested-bytes={} \
         conservative-product-floor-ms={:.3} traffic-floor-ms={:.3} no-cache-floor-ms={:.3} \
         conservative-q-80pct-cap-ms={:.3} member-5x-cap-ms={:.3} cpu-member-ms={:.3} \
         resident-bytes={} compact-resident-bytes={} readback-bytes={} producer-groups={} reducer-groups={} \
         producer-tew={} reducer-tew={}",
        config.q_accumulator.name(),
        setup_wall.as_secs_f64() * 1e3,
        cold.gpu_active.as_secs_f64() * 1e3,
        warm.gpu_active.as_secs_f64() * 1e3,
        warm.resident_wall.as_secs_f64() * 1e3,
        projection.q_full_width_products,
        projection.q_perfect_cache_bytes,
        projection.q_shader_logical_bytes,
        projection.q_product_floor_ns as f64 / 1e6,
        projection.q_perfect_cache_traffic_floor_ns as f64 / 1e6,
        projection.q_no_cache_request_floor_ns as f64 / 1e6,
        projection.q_pursuit_ns as f64 / 1e6,
        RAM_RA_CLAIM_TARGET_FIVE_X_NS as f64 / 1e6,
        RAM_RA_CLAIM_TARGET_CPU_NS as f64 / 1e6,
        plan.storage.total_resident_bytes,
        invocation.compact_resident_bytes(),
        plan.storage.readback_bytes,
        plan.producer_dispatch.threadgroups,
        plan.reducer_dispatch.threadgroups,
        invocation.producer_pipeline_limits().thread_execution_width,
        invocation.reducer_pipeline_limits().thread_execution_width,
    );

    let suffix = format!(
        "n{rows}_a{accessed_rows}_{}_p{}",
        config.q_accumulator.name(),
        config.q_partitions
    );
    let mut group = c.benchmark_group("metal_sumcheck/ram_ra_claim_q");
    let _ = group
        .sample_size(10)
        .warm_up_time(Duration::from_millis(setting(
            "JOLT_METAL_RAM_RA_CLAIM_WARMUP_MS",
            200,
        )))
        .measurement_time(Duration::from_millis(setting(
            "JOLT_METAL_RAM_RA_CLAIM_MEASUREMENT_MS",
            1_000,
        )))
        .throughput(Throughput::Elements(projection.q_full_width_products));
    let _ = group.bench_function(BenchmarkId::new("resident_active", &suffix), |bench| {
        bench.iter_custom(|iterations| {
            let mut active = Duration::ZERO;
            for _ in 0..iterations {
                let observation = invocation
                    .execute_timed()
                    .expect("timed RAM RA claim Q should execute");
                let _ = black_box(observation.checksum);
                active += observation.gpu_active;
            }
            active
        });
    });
    let _ = group.bench_function(BenchmarkId::new("resident_wall", &suffix), |bench| {
        bench.iter_custom(|iterations| {
            let mut wall = Duration::ZERO;
            for _ in 0..iterations {
                let observation = invocation
                    .execute_timed()
                    .expect("timed RAM RA claim Q should execute");
                let _ = black_box(observation.checksum);
                wall += observation.resident_wall;
            }
            wall
        });
    });
    group.finish();
}

fn uniform_addresses(rows: usize, accessed_rows: usize) -> Vec<u32> {
    (0..rows)
        .map(|row| {
            let before = row * accessed_rows / rows;
            let after = (row + 1) * accessed_rows / rows;
            if after == before {
                RAM_RA_CLAIM_NO_ACCESS
            } else {
                (splitmix64(row as u64) as usize % RAM_RA_CLAIM_ADDRESS_DOMAIN) as u32
            }
        })
        .collect()
}

fn point(length: usize, seed: u64) -> Vec<AkitaField> {
    (0..length)
        .map(|index| AkitaField::from_u64(splitmix64(seed.wrapping_add(index as u64))))
        .collect()
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn trace_rows() -> usize {
    let rows = env::var("JOLT_SOLINAS_BENCH_ELEMENTS").map_or(RAM_RA_CLAIM_TARGET_ROWS, |value| {
        value
            .parse()
            .expect("JOLT_SOLINAS_BENCH_ELEMENTS should be a positive integer")
    });
    assert!(
        rows.is_power_of_two() && rows >= 1 << 12,
        "RAM RA claim trace size must be a power of two at least 2^12"
    );
    rows
}

fn accessed_rows(rows: usize) -> usize {
    env::var("JOLT_METAL_RAM_RA_CLAIM_ACCESSED_ROWS").map_or_else(
        |_| {
            if rows == RAM_RA_CLAIM_TARGET_ROWS {
                RAM_RA_CLAIM_TARGET_ACCESSED_ROWS
            } else {
                rows * RAM_RA_CLAIM_TARGET_ACCESSED_ROWS / RAM_RA_CLAIM_TARGET_ROWS
            }
        },
        |value| {
            value
                .parse()
                .expect("JOLT_METAL_RAM_RA_CLAIM_ACCESSED_ROWS should be a nonnegative integer")
        },
    )
}

fn setting(name: &str, default: u64) -> u64 {
    env::var(name).map_or(default, |value| {
        value
            .parse()
            .unwrap_or_else(|_| panic!("{name} should be a nonnegative integer"))
    })
}

fn accumulator() -> RamRaClaimQAccumulator {
    match env::var("JOLT_METAL_RAM_RA_CLAIM_ACCUMULATOR").as_deref() {
        Ok("array") => RamRaClaimQAccumulator::Array,
        Ok("explicit") => RamRaClaimQAccumulator::Explicit,
        Ok("compact") | Err(_) => RamRaClaimQAccumulator::Compact,
        Ok(value) => panic!(
            "JOLT_METAL_RAM_RA_CLAIM_ACCUMULATOR must be `array`, `explicit`, or `compact`, got `{value}`"
        ),
    }
}

fn paired_control() -> bool {
    setting("JOLT_METAL_RAM_RA_CLAIM_PAIRED", 0) != 0
}

fn q_partitions() -> usize {
    setting("JOLT_METAL_RAM_RA_CLAIM_Q_PARTITIONS", 8) as usize
}

fn report_paired_screen(candidate: &RamRaClaimQInvocation, control: &RamRaClaimQInvocation) {
    for _ in 0..8 {
        let _ = candidate
            .execute_timed()
            .expect("paired RAM RA candidate warmup should execute");
        let _ = control
            .execute_timed()
            .expect("paired RAM RA control warmup should execute");
    }
    let mut candidate_active = Vec::with_capacity(10);
    let mut candidate_wall = Vec::with_capacity(10);
    let mut control_active = Vec::with_capacity(10);
    let mut control_wall = Vec::with_capacity(10);
    for pair in 0usize..10 {
        let sample = |invocation: &RamRaClaimQInvocation| {
            invocation
                .execute_timed()
                .expect("paired RAM RA claim Q should execute")
        };
        let (candidate_sample, control_sample) = if pair.is_multiple_of(2) {
            (sample(candidate), sample(control))
        } else {
            let control_sample = sample(control);
            (sample(candidate), control_sample)
        };
        assert_eq!(candidate_sample.checksum, control_sample.checksum);
        candidate_active.push(candidate_sample.gpu_active);
        candidate_wall.push(candidate_sample.resident_wall);
        control_active.push(control_sample.gpu_active);
        control_wall.push(control_sample.resident_wall);
    }
    let candidate_active_median = median(&candidate_active);
    let candidate_wall_median = median(&candidate_wall);
    let control_active_median = median(&control_active);
    let control_wall_median = median(&control_wall);
    eprintln!(
        "ram-ra-claim-q-paired candidate-active={candidate_active:?} control-active={control_active:?} \
         candidate-wall={candidate_wall:?} control-wall={control_wall:?} \
         candidate-active-median={candidate_active_median:?} control-active-median={control_active_median:?} \
         candidate-wall-median={candidate_wall_median:?} control-wall-median={control_wall_median:?} \
         active-speedup={} wall-speedup={}",
        control_active_median.as_secs_f64() / candidate_active_median.as_secs_f64(),
        control_wall_median.as_secs_f64() / candidate_wall_median.as_secs_f64(),
    );
}

fn median(samples: &[Duration]) -> Duration {
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    sorted[sorted.len() / 2]
}
