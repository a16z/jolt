use std::{env, hint::black_box, time::Duration, time::Instant};

use criterion::{BenchmarkId, Criterion, Throughput};
use jolt_field::AkitaField;
use jolt_kernels::metal::solinas::{
    registers_claim_reduction::{
        RegistersClaimAccumulator, RegistersClaimKernelConfig, RegistersClaimLinearQInvocation,
        RegistersClaimRoofRates, REGISTERS_CLAIM_FIVE_X_GATE_NS,
    },
    SolinasMetal,
};

const DEFAULT_TRACE_ELEMENTS: usize = 1 << 26;

pub fn bench(c: &mut Criterion, context: &SolinasMetal) {
    let elements = trace_elements();
    let config = RegistersClaimKernelConfig {
        build_threads_per_threadgroup: setting("JOLT_METAL_REGISTERS_CLAIM_BUILD_THREADS", 128)
            as usize,
        accumulator: accumulator(),
        ..RegistersClaimKernelConfig::default()
    };
    let rd_value = u64::MAX;
    let rs1_value = (1_u64 << 32) + 1;
    let rs2_value = (1_u64 << 32) - 1;
    let rd = vec![rd_value; elements];
    let rs1 = vec![rs1_value; elements];
    let rs2 = vec![rs2_value; elements];
    let tau = (0..elements.trailing_zeros() as usize)
        .map(|index| AkitaField::from_u64(0x1_0000_01b3_u64.wrapping_mul(index as u64 + 1)))
        .collect::<Vec<_>>();
    let gamma = -AkitaField::from_u64(0xfeed_face_cafe_beef);

    let setup_started = Instant::now();
    let resident = context
        .prepare_registers_claim_resident_planes(&rd, &rs1, &rs2)
        .expect("registers claim resident planes should prepare");
    drop((rd, rs1, rs2));
    let invocation = context
        .prepare_registers_claim_linear_q(&resident, &tau, gamma, config)
        .expect("registers claim linear-q invocation should prepare");
    let prefix_challenges = tau[..tau.len() - tau.len() / 2].to_vec();
    let fold_invocation = context
        .prepare_registers_claim_direct_fold(&resident, &prefix_challenges, config)
        .expect("registers claim direct-fold invocation should prepare");
    let control = paired_control().then(|| {
        context
            .prepare_registers_claim_linear_q(
                &resident,
                &tau,
                gamma,
                RegistersClaimKernelConfig {
                    accumulator: RegistersClaimAccumulator::Deferred224,
                    ..config
                },
            )
            .expect("registers claim deferred control should prepare")
    });
    drop(resident);
    let setup_wall = setup_started.elapsed();

    let expected = AkitaField::from_u64(rd_value)
        + gamma * AkitaField::from_u64(rs1_value)
        + gamma * gamma * AkitaField::from_u64(rs2_value);
    let cold = invocation
        .execute_timed()
        .expect("registers claim linear-q should execute");
    assert!(cold.q.iter().all(|&value| value == expected));
    if let Some(control) = &control {
        let observation = control
            .execute_timed()
            .expect("registers claim deferred control should execute");
        assert_eq!(observation.q, cold.q);
    }
    assert_eq!(invocation.execute_device_buffer_allocations(), 0);
    assert!(!invocation
        .source_allocation_identities()
        .contains(&invocation.output_allocation_identity()));

    let warm = invocation
        .execute_timed()
        .expect("warm registers claim linear-q should execute");
    assert_eq!(cold.checksum, warm.checksum);
    let fold_cold = fold_invocation
        .execute_timed()
        .expect("registers claim direct fold should execute");
    assert!(fold_cold
        .outputs
        .rd_write_value
        .iter()
        .all(|&value| value == AkitaField::from_u64(rd_value)));
    assert!(fold_cold
        .outputs
        .rs1_value
        .iter()
        .all(|&value| value == AkitaField::from_u64(rs1_value)));
    assert!(fold_cold
        .outputs
        .rs2_value
        .iter()
        .all(|&value| value == AkitaField::from_u64(rs2_value)));
    if let Some(control) = &control {
        report_paired_screen(&invocation, control);
    }
    let work = invocation.plan().work().expect("checked work should fit");
    let ceiling = work
        .calibrated_ceiling(RegistersClaimRoofRates::CONSERVATIVE, 80)
        .expect("calibrated ceiling should fit");
    eprintln!(
        "registers-claim-linear-q n={elements} accumulator={} resident-bytes={} setup-ms={:.3} \
         cold-active-ms={:.3} warm-active-ms={:.3} warm-wall-ms={:.3} \
         half-width-terms={} full-products={} traffic-floor-ms={:.3} \
         arithmetic-floor-ms={:.3} active-80pct-cap-ms={:.3} five-x-member-gate-ms={:.3} \
         tew={} max-threads={} tg={}",
        config.accumulator.name(),
        invocation.plan().storage.total_resident_bytes,
        setup_wall.as_secs_f64() * 1e3,
        cold.gpu_active.as_secs_f64() * 1e3,
        warm.gpu_active.as_secs_f64() * 1e3,
        warm.resident_wall.as_secs_f64() * 1e3,
        work.half_width_terms,
        work.full_products,
        ceiling.traffic_floor_ns as f64 / 1e6,
        ceiling.arithmetic_floor_ns as f64 / 1e6,
        ceiling.utilization_cap_ns as f64 / 1e6,
        REGISTERS_CLAIM_FIVE_X_GATE_NS as f64 / 1e6,
        invocation.pipeline_limits().thread_execution_width,
        invocation
            .pipeline_limits()
            .max_total_threads_per_threadgroup,
        invocation.threads_per_threadgroup(),
    );
    eprintln!(
        "registers-claim-direct-fold n={elements} active-ms={:.3} wall-ms={:.3} \
         half-width-terms={} threadgroups={} tew={} max-threads={} tg={} tg-bytes={}",
        fold_cold.gpu_active.as_secs_f64() * 1e3,
        fold_cold.resident_wall.as_secs_f64() * 1e3,
        fold_cold.useful_half_width_terms,
        fold_cold.threadgroups,
        fold_invocation.pipeline_limits().thread_execution_width,
        fold_invocation
            .pipeline_limits()
            .max_total_threads_per_threadgroup,
        fold_invocation.threads_per_threadgroup(),
        fold_invocation.dynamic_threadgroup_bytes(),
    );

    let suffix = format!(
        "n{elements}_{}_tg{}",
        config.accumulator.name(),
        invocation.threads_per_threadgroup()
    );
    let mut group = c.benchmark_group("metal_sumcheck/registers_claim_linear_q");
    let _ = group
        .sample_size(10)
        .warm_up_time(Duration::from_millis(setting(
            "JOLT_METAL_REGISTERS_CLAIM_WARMUP_MS",
            200,
        )))
        .measurement_time(Duration::from_millis(setting(
            "JOLT_METAL_REGISTERS_CLAIM_MEASUREMENT_MS",
            1_000,
        )))
        .throughput(Throughput::Elements(work.half_width_terms));
    let _ = group.bench_function(BenchmarkId::new("resident_active", &suffix), |bench| {
        bench.iter_custom(|iterations| {
            let mut active = Duration::ZERO;
            for _ in 0..iterations {
                let observation = invocation
                    .execute_timed()
                    .expect("timed registers claim linear-q should execute");
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
                    .expect("timed registers claim linear-q should execute");
                let _ = black_box(observation.checksum);
                wall += observation.resident_wall;
            }
            wall
        });
    });
    group.finish();

    let fold_suffix = format!(
        "n{elements}_tg{}",
        fold_invocation.threads_per_threadgroup()
    );
    let mut fold_group = c.benchmark_group("metal_sumcheck/registers_claim_direct_fold");
    let _ = fold_group
        .sample_size(10)
        .warm_up_time(Duration::from_millis(setting(
            "JOLT_METAL_REGISTERS_CLAIM_WARMUP_MS",
            200,
        )))
        .measurement_time(Duration::from_millis(setting(
            "JOLT_METAL_REGISTERS_CLAIM_MEASUREMENT_MS",
            1_000,
        )))
        .throughput(Throughput::Elements(fold_cold.useful_half_width_terms));
    let _ = fold_group.bench_function(BenchmarkId::new("resident_active", &fold_suffix), |bench| {
        bench.iter_custom(|iterations| {
            let mut active = Duration::ZERO;
            for _ in 0..iterations {
                let observation = fold_invocation
                    .execute_timed()
                    .expect("timed registers claim direct fold should execute");
                let _ = black_box(observation.outputs.rd_write_value[0]);
                active += observation.gpu_active;
            }
            active
        });
    });
    let _ = fold_group.bench_function(BenchmarkId::new("resident_wall", &fold_suffix), |bench| {
        bench.iter_custom(|iterations| {
            let mut wall = Duration::ZERO;
            for _ in 0..iterations {
                let observation = fold_invocation
                    .execute_timed()
                    .expect("timed registers claim direct fold should execute");
                let _ = black_box(observation.outputs.rd_write_value[0]);
                wall += observation.resident_wall;
            }
            wall
        });
    });
    fold_group.finish();
}

fn trace_elements() -> usize {
    let elements =
        env::var("JOLT_SOLINAS_BENCH_ELEMENTS").map_or(DEFAULT_TRACE_ELEMENTS, |value| {
            value
                .parse()
                .expect("JOLT_SOLINAS_BENCH_ELEMENTS should be a positive integer")
        });
    assert!(
        elements.is_power_of_two() && elements >= 1 << 12,
        "registers claim trace size must be a power of two at least 2^12"
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

fn accumulator() -> RegistersClaimAccumulator {
    match env::var("JOLT_METAL_REGISTERS_CLAIM_ACCUMULATOR").as_deref() {
        Ok("deferred224") => RegistersClaimAccumulator::Deferred224,
        Ok("canonical128") | Err(_) => RegistersClaimAccumulator::Canonical128,
        Ok(value) => panic!(
            "JOLT_METAL_REGISTERS_CLAIM_ACCUMULATOR must be `deferred224` or `canonical128`, got `{value}`"
        ),
    }
}

fn paired_control() -> bool {
    setting("JOLT_METAL_REGISTERS_CLAIM_PAIRED", 0) != 0
}

fn report_paired_screen(
    candidate: &RegistersClaimLinearQInvocation,
    control: &RegistersClaimLinearQInvocation,
) {
    let mut candidate_active = Vec::with_capacity(10);
    let mut candidate_wall = Vec::with_capacity(10);
    let mut control_active = Vec::with_capacity(10);
    let mut control_wall = Vec::with_capacity(10);
    for pair in 0usize..10 {
        let sample = |invocation: &RegistersClaimLinearQInvocation| {
            invocation
                .execute_timed()
                .expect("paired registers claim invocation should execute")
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
        "registers-claim-linear-q-paired candidate-active={candidate_active:?} \
         control-active={control_active:?} candidate-wall={candidate_wall:?} \
         control-wall={control_wall:?} candidate-active-median={candidate_active_median:?} \
         control-active-median={control_active_median:?} candidate-wall-median={candidate_wall_median:?} \
         control-wall-median={control_wall_median:?} active-speedup={} wall-speedup={}",
        control_active_median.as_secs_f64() / candidate_active_median.as_secs_f64(),
        control_wall_median.as_secs_f64() / candidate_wall_median.as_secs_f64(),
    );
}

fn median(samples: &[Duration]) -> Duration {
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    sorted[sorted.len() / 2]
}
