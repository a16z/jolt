use std::{env, hint::black_box, time::Duration};

use criterion::{BenchmarkId, Criterion, Throughput};
use jolt_kernels::metal::solinas::half_width_probe::reference_outputs;
use jolt_kernels::metal::solinas::{
    DispatchConfig, Fp128, HalfWidthDomain, HalfWidthOperand, HalfWidthProbe, Probe, SolinasMetal,
    TARGET_CHAIN_ELEMENTS, TARGET_CHAIN_ITERATIONS,
};

pub fn bench(c: &mut Criterion, context: &SolinasMetal) {
    let elements = setting_usize("JOLT_METAL_HALF_WIDTH_ELEMENTS", TARGET_CHAIN_ELEMENTS);
    let iterations = setting_u32("JOLT_METAL_HALF_WIDTH_ITERATIONS", TARGET_CHAIN_ITERATIONS);
    assert!(elements.is_power_of_two() && elements >= 1 << 10);
    let coefficients = coefficients(elements);
    let threadgroup_widths = threadgroup_widths();
    let ilps = selected_ilps();
    let mut group = c.benchmark_group("metal_solinas/half_width_chain_active");
    let _ = group
        .sample_size(10)
        .warm_up_time(Duration::from_millis(setting_u64(
            "JOLT_METAL_HALF_WIDTH_WARMUP_MS",
            100,
        )))
        .measurement_time(Duration::from_millis(setting_u64(
            "JOLT_METAL_HALF_WIDTH_MEASUREMENT_MS",
            300,
        )));

    for domain in [
        HalfWidthDomain::Unsigned,
        HalfWidthDomain::SignedMagnitude,
        HalfWidthDomain::UnsignedDelta,
    ] {
        let operands = operands(domain, elements);
        for &ilp in &ilps {
            let probe = probe(domain, ilp);
            for &threads in &threadgroup_widths {
                let invocation = context
                    .prepare_half_width_probe(
                        probe,
                        &coefficients,
                        &operands,
                        iterations,
                        Some(threads),
                    )
                    .expect("half-width chain should prepare");
                invocation
                    .execute()
                    .expect("half-width chain should execute");
                let output = invocation
                    .read_output()
                    .expect("half-width output should be readable");
                validate_edges(probe, &coefficients, &operands, iterations, &output);
                assert_eq!(invocation.execute_device_buffer_allocations(), 0);
                eprintln!(
                    "half-width probe={} operations={} tg={} tew={} max_threads={} live_word_floor={}",
                    probe.name(),
                    invocation.shape().operation_count(),
                    threads,
                    invocation.pipeline_limits().thread_execution_width,
                    invocation
                        .pipeline_limits()
                        .max_total_threads_per_threadgroup,
                    jolt_kernels::metal::solinas::half_width_probe::HalfWidthRegisterFloor::for_probe(probe)
                        .minimum_live_words,
                );
                let _ =
                    group.throughput(Throughput::Elements(invocation.shape().operation_count()));
                let suffix = format!("{}_n{elements}_iter{iterations}_tg{threads}", probe.name());
                let _ = group.bench_function(BenchmarkId::new("half", suffix), |bench| {
                    bench.iter_custom(|samples| {
                        let mut active = Duration::ZERO;
                        for _ in 0..samples {
                            active += invocation
                                .execute_timed()
                                .expect("timed half-width chain should execute");
                        }
                        active
                    });
                });
            }
        }
    }

    let rhs = (0..elements)
        .map(|index| Fp128::from_u128((index as u64).wrapping_mul(29).wrapping_add(3) as u128))
        .collect::<Vec<_>>();
    for &ilp in &ilps {
        let probe = full_probe(ilp);
        for &threads in &threadgroup_widths {
            let invocation = context
                .prepare(
                    probe,
                    &coefficients,
                    &rhs,
                    DispatchConfig {
                        iterations,
                        threads_per_threadgroup: Some(threads),
                    },
                )
                .expect("full-width control should prepare");
            invocation
                .execute()
                .expect("full-width control should execute");
            let _ = invocation
                .read_output()
                .expect("full-width output should be readable");
            let operations = invocation.field_operation_count();
            let _ = group.throughput(Throughput::Elements(operations));
            let suffix = format!("{}_n{elements}_iter{iterations}_tg{threads}", probe.name());
            let _ = group.bench_function(BenchmarkId::new("full", suffix), |bench| {
                bench.iter_custom(|samples| {
                    let mut active = Duration::ZERO;
                    for _ in 0..samples {
                        active += invocation
                            .execute_timed()
                            .expect("timed full-width control should execute");
                    }
                    active
                });
            });
        }
    }
    group.finish();
}

fn coefficients(elements: usize) -> Vec<Fp128> {
    (0..elements)
        .map(|index| {
            let low = (index as u64)
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let high = low.rotate_left(29) ^ 0xa5a5_5a5a_1234_5678;
            Fp128::from_u128((u128::from(high) << 64) | u128::from(low))
        })
        .collect()
}

fn operands(domain: HalfWidthDomain, elements: usize) -> Vec<HalfWidthOperand> {
    (0..elements)
        .map(|index| {
            let primary = (index as u64)
                .wrapping_mul(1_442_695_040_888_963_407)
                .wrapping_add(6_364_136_223_846_793_005);
            match domain {
                HalfWidthDomain::Unsigned => HalfWidthOperand::unsigned(primary),
                HalfWidthDomain::SignedMagnitude => {
                    HalfWidthOperand::signed_magnitude(primary, index % 2 != 0)
                }
                HalfWidthDomain::UnsignedDelta => {
                    HalfWidthOperand::delta(primary, primary.rotate_left(31))
                }
            }
        })
        .collect()
}

fn probe(domain: HalfWidthDomain, ilp: usize) -> HalfWidthProbe {
    match (domain, ilp) {
        (HalfWidthDomain::Unsigned, 1) => HalfWidthProbe::ChainU64Ilp1,
        (HalfWidthDomain::Unsigned, 2) => HalfWidthProbe::ChainU64Ilp2,
        (HalfWidthDomain::Unsigned, 4) => HalfWidthProbe::ChainU64Ilp4,
        (HalfWidthDomain::Unsigned, 8) => HalfWidthProbe::ChainU64Ilp8,
        (HalfWidthDomain::SignedMagnitude, 1) => HalfWidthProbe::ChainSignedU64Ilp1,
        (HalfWidthDomain::SignedMagnitude, 2) => HalfWidthProbe::ChainSignedU64Ilp2,
        (HalfWidthDomain::SignedMagnitude, 4) => HalfWidthProbe::ChainSignedU64Ilp4,
        (HalfWidthDomain::SignedMagnitude, 8) => HalfWidthProbe::ChainSignedU64Ilp8,
        (HalfWidthDomain::UnsignedDelta, 1) => HalfWidthProbe::ChainU64DeltaIlp1,
        (HalfWidthDomain::UnsignedDelta, 2) => HalfWidthProbe::ChainU64DeltaIlp2,
        (HalfWidthDomain::UnsignedDelta, 4) => HalfWidthProbe::ChainU64DeltaIlp4,
        (HalfWidthDomain::UnsignedDelta, 8) => HalfWidthProbe::ChainU64DeltaIlp8,
        _ => panic!("half-width ILP should be one of 1, 2, 4, or 8"),
    }
}

fn full_probe(ilp: usize) -> Probe {
    match ilp {
        1 => Probe::ChainWide1,
        2 => Probe::ChainWide2,
        4 => Probe::ChainWide4,
        8 => Probe::ChainWide8,
        _ => panic!("full-width ILP should be one of 1, 2, 4, or 8"),
    }
}

fn selected_ilps() -> Vec<usize> {
    env::var("JOLT_METAL_HALF_WIDTH_ILP").map_or_else(
        |_| vec![1, 2, 4, 8],
        |value| {
            value
                .split(',')
                .map(|ilp| {
                    let ilp = ilp
                        .parse()
                        .expect("half-width ILPs should be comma-separated integers");
                    assert!([1, 2, 4, 8].contains(&ilp));
                    ilp
                })
                .collect()
        },
    )
}

fn validate_edges(
    probe: HalfWidthProbe,
    coefficients: &[Fp128],
    operands: &[HalfWidthOperand],
    iterations: u32,
    output: &[Fp128],
) {
    let ilp = probe.independent_chains();
    for start in [0, coefficients.len() - ilp] {
        let end = start + ilp;
        let expected = reference_outputs(
            probe,
            &coefficients[start..end],
            &operands[start..end],
            iterations,
        )
        .expect("half-width edge oracle should evaluate");
        assert_eq!(&output[start..end], expected);
    }
    let _ = black_box(output[0]);
}

fn threadgroup_widths() -> Vec<usize> {
    env::var("JOLT_METAL_HALF_WIDTH_THREADS").map_or_else(
        |_| vec![256],
        |value| {
            value
                .split(',')
                .map(|width| {
                    width
                        .parse()
                        .expect("threadgroup widths should be comma-separated integers")
                })
                .collect()
        },
    )
}

fn setting_u64(name: &str, default: u64) -> u64 {
    env::var(name).map_or(default, |value| {
        value
            .parse()
            .unwrap_or_else(|_| panic!("{name} should be a nonnegative integer"))
    })
}

fn setting_usize(name: &str, default: usize) -> usize {
    setting_u64(name, default as u64) as usize
}

fn setting_u32(name: &str, default: u32) -> u32 {
    setting_u64(name, u64::from(default)) as u32
}
