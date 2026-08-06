use std::{env, hint::black_box, time::Duration, time::Instant};

use criterion::{BenchmarkId, Criterion, Throughput};
use jolt_field::AkitaField;
use jolt_kernels::metal::solinas::spartan_shift::{
    bind_dense_state, bind_prefix_tables, dense_round_endpoints, prefix_round_endpoints,
    SpartanShiftDenseState, SpartanShiftFlagWord, SpartanShiftGeometry, SpartanShiftKernelConfig,
    SpartanShiftOutputs, SpartanShiftPrefixStrategy, SpartanShiftPrefixTables,
    SpartanShiftResidentRows, SPARTAN_SHIFT_CPU_MEDIAN_NS, SPARTAN_SHIFT_FIVE_X_CAP_NS,
};
use jolt_kernels::metal::solinas::SolinasMetal;
use jolt_poly::EqPlusOnePrefixSuffix;
use rayon::prelude::*;

const DEFAULT_ROWS: usize = 1 << 20;

pub fn bench(c: &mut Criterion, context: &SolinasMetal) {
    let rows = setting("JOLT_SOLINAS_BENCH_ELEMENTS", DEFAULT_ROWS);
    let defaults = SpartanShiftKernelConfig::default();
    let config = SpartanShiftKernelConfig {
        build_threads_per_threadgroup: setting(
            "JOLT_METAL_SPARTAN_SHIFT_BUILD_THREADS",
            defaults.build_threads_per_threadgroup,
        ),
        high_tile_elements: setting(
            "JOLT_METAL_SPARTAN_SHIFT_HIGH_TILE",
            defaults.high_tile_elements,
        ),
        fold_threads_per_threadgroup: setting(
            "JOLT_METAL_SPARTAN_SHIFT_FOLD_THREADS",
            defaults.fold_threads_per_threadgroup,
        ),
    };
    let warmup_ms = setting("JOLT_METAL_SPARTAN_SHIFT_WARMUP_MS", 1_000);
    let measurement_ms = setting("JOLT_METAL_SPARTAN_SHIFT_MEASUREMENT_MS", 3_000);
    let observe_only = env::var_os("JOLT_METAL_SPARTAN_SHIFT_OBSERVE_ONLY").is_some();
    let hybrid_only =
        env::var("JOLT_METAL_SPARTAN_SHIFT_CRITERION").is_ok_and(|value| value == "hybrid-mixed");
    let retained_only =
        hybrid_only || env::var_os("JOLT_METAL_SPARTAN_SHIFT_RETAINED_ONLY").is_some();
    let geometry = SpartanShiftGeometry::new(rows).expect("Spartan shift geometry should be valid");
    let generation_start = Instant::now();
    let (unexpanded_pc, pc, flags) = native_planes(geometry);
    let generation_wall = generation_start.elapsed();
    let upload_start = Instant::now();
    let resident = context
        .prepare_spartan_shift_rows(&unexpanded_pc, &pc, &flags, true)
        .expect("Spartan shift resident planes should prepare");
    let upload_wall = upload_start.elapsed();
    let source_allocations = resident.allocation_identities();
    let r_outer = values(geometry.log_t(), 0xA11C_E001);
    let r_product = values(geometry.log_t(), 0xB22D_F002);
    let gamma = AkitaField::from_u64(0xC33E_1003);
    let prefix_challenges = values(geometry.prefix_vars(), 0xD44F_2004);
    let suffix_challenges = values(geometry.suffix_vars(), 0xE550_3005);
    if env::var_os("JOLT_METAL_SPARTAN_SHIFT_SWEEP").is_some() {
        sweep(
            context,
            &resident,
            &r_outer,
            &r_product,
            gamma,
            &prefix_challenges,
        );
        return;
    }

    let setup_start = Instant::now();
    let host_factors = SpartanShiftHostFactors::new(&r_outer, &r_product);
    let mixed = context
        .prepare_spartan_shift_prefix(
            &resident,
            &r_outer,
            &r_product,
            gamma,
            config,
            SpartanShiftPrefixStrategy::Mixed,
        )
        .expect("mixed Spartan shift prefix should prepare");
    let expanded = (!retained_only).then(|| {
        context
            .prepare_spartan_shift_prefix(
                &resident,
                &r_outer,
                &r_product,
                gamma,
                config,
                SpartanShiftPrefixStrategy::ExpandedHalfWidth,
            )
            .expect("expanded Spartan shift prefix should prepare")
    });
    let fold = context
        .prepare_spartan_shift_fold(&resident, &prefix_challenges, config)
        .expect("Spartan shift native fold should prepare");
    let setup_wall = setup_start.elapsed();
    for allocations in [
        mixed.source_allocation_identities(),
        fold.source_allocation_identities(),
    ] {
        assert_eq!(allocations, source_allocations);
    }
    if let Some(expanded) = expanded.as_ref() {
        assert_eq!(expanded.source_allocation_identities(), source_allocations);
        assert_eq!(expanded.execute_device_buffer_allocations(), 0);
    }
    assert_eq!(mixed.execute_device_buffer_allocations(), 0);
    assert_eq!(fold.execute_device_buffer_allocations(), 0);

    let mixed_cold_start = Instant::now();
    let mixed_cold = mixed
        .execute()
        .expect("mixed Spartan shift prefix should execute");
    let mixed_cold_wall = mixed_cold_start.elapsed();
    let expanded_cold = expanded.as_ref().map(|expanded| {
        let started = Instant::now();
        let observation = expanded
            .execute()
            .expect("expanded Spartan shift prefix should execute");
        assert_eq!(mixed_cold.q, observation.q);
        (started.elapsed(), observation.gpu_active)
    });
    let fold_cold_start = Instant::now();
    let fold_cold = fold
        .execute()
        .expect("Spartan shift native fold should execute");
    let fold_cold_wall = fold_cold_start.elapsed();

    let hybrid_start = Instant::now();
    let mixed_warm_start = Instant::now();
    let mixed_warm = mixed
        .execute()
        .expect("mixed Spartan shift prefix should execute");
    let mixed_warm_wall = mixed_warm_start.elapsed();
    let prefix_host_start = Instant::now();
    let (prefix_host, bound_prefix) =
        run_prefix_host(mixed_warm.q, &host_factors, &prefix_challenges);
    let prefix_host_wall = prefix_host_start.elapsed();
    let fold_warm_start = Instant::now();
    let folded = fold
        .execute()
        .expect("Spartan shift native fold should execute");
    let fold_warm_wall = fold_warm_start.elapsed();
    let fold_warm_active = folded.gpu_active;
    let suffix_host_start = Instant::now();
    let outputs = run_suffix_host(
        geometry,
        folded.outputs,
        &host_factors,
        bound_prefix,
        &suffix_challenges,
        gamma,
    );
    let suffix_host_wall = suffix_host_start.elapsed();
    let hybrid_wall = hybrid_start.elapsed();
    let _ = black_box((prefix_host, outputs));
    let ratio = if rows == 1 << 26 {
        SPARTAN_SHIFT_CPU_MEDIAN_NS as f64 / hybrid_wall.as_nanos() as f64
    } else {
        f64::NAN
    };

    eprintln!(
        "spartan-shift rows={rows} generation={generation_wall:?} upload={upload_wall:?} setup={setup_wall:?} mixed-cold-wall={mixed_cold_wall:?} mixed-cold-active={:?} expanded-cold={expanded_cold:?} fold-cold-wall={fold_cold_wall:?} fold-cold-active={:?} mixed-warm-wall={mixed_warm_wall:?} mixed-warm-active={:?} prefix-host={prefix_host_wall:?} fold-warm-wall={fold_warm_wall:?} fold-warm-active={fold_warm_active:?} suffix-host={suffix_host_wall:?} hybrid-warm-wall={hybrid_wall:?} target-cpu-ms={:.6} five-x-cap-ms={:.6} target-ratio={ratio:.6} resident-bytes={} mixed-private-bytes={} expanded-private-bytes={:?} fold-private-bytes={} build-threads={} high-tile={} fold-threads={}",
        mixed_cold.gpu_active,
        fold_cold.gpu_active,
        mixed_warm.gpu_active,
        SPARTAN_SHIFT_CPU_MEDIAN_NS as f64 / 1e6,
        SPARTAN_SHIFT_FIVE_X_CAP_NS as f64 / 1e6,
        resident.resident_bytes(),
        mixed.plan().storage.total_resident_bytes - resident.resident_bytes(),
        expanded.as_ref().map(|invocation| {
            invocation.plan().storage.total_resident_bytes - resident.resident_bytes()
        }),
        fold.plan().storage.low_weight_bytes + fold.plan().storage.dense_output_bytes,
        config.build_threads_per_threadgroup,
        config.high_tile_elements,
        config.fold_threads_per_threadgroup,
    );

    if observe_only {
        return;
    }

    let mut group = c.benchmark_group("metal_sumcheck/spartan_shift");
    let _ = group
        .sample_size(10)
        .warm_up_time(Duration::from_millis(warmup_ms as u64))
        .measurement_time(Duration::from_millis(measurement_ms as u64))
        .throughput(Throughput::Elements(rows as u64));
    let suffix = format!(
        "n{rows}_bt{}_h{}_ft{}",
        config.build_threads_per_threadgroup,
        config.high_tile_elements,
        config.fold_threads_per_threadgroup
    );
    if !hybrid_only {
        for (name, invocation) in std::iter::once(("mixed", &mixed))
            .chain(expanded.as_ref().map(|invocation| ("expanded", invocation)))
        {
            let _ =
                group.bench_function(BenchmarkId::new(format!("{name}_wall"), &suffix), |bench| {
                    bench.iter(|| black_box(invocation.execute().expect("prefix should execute")));
                });
            let _ = group.bench_function(
                BenchmarkId::new(format!("{name}_active"), &suffix),
                |bench| {
                    bench.iter_custom(|iterations| {
                        let mut active = Duration::ZERO;
                        for _ in 0..iterations {
                            active += invocation
                                .execute()
                                .expect("prefix should execute")
                                .gpu_active;
                        }
                        active
                    });
                },
            );
        }
        let _ = group.bench_function(BenchmarkId::new("fold_wall", &suffix), |bench| {
            bench.iter(|| black_box(fold.execute().expect("fold should execute")));
        });
    }
    let _ = group.bench_function(
        BenchmarkId::new("resident_hybrid_mixed_wall", &suffix),
        |bench| {
            bench.iter(|| {
                let prefix = mixed.execute().expect("prefix should execute");
                let (prefix_host, bound_prefix) =
                    run_prefix_host(prefix.q, &host_factors, &prefix_challenges);
                let folded = fold.execute().expect("fold should execute");
                let suffix_host = run_suffix_host(
                    geometry,
                    folded.outputs,
                    &host_factors,
                    bound_prefix,
                    &suffix_challenges,
                    gamma,
                );
                black_box((prefix_host, suffix_host))
            });
        },
    );
    group.finish();
}

struct SpartanShiftHostFactors {
    p: [Vec<AkitaField>; 4],
    suffix: [Vec<AkitaField>; 4],
}

impl SpartanShiftHostFactors {
    fn new(r_outer: &[AkitaField], r_product: &[AkitaField]) -> Self {
        let outer = EqPlusOnePrefixSuffix::new(r_outer);
        let product = EqPlusOnePrefixSuffix::new(r_product);
        Self {
            p: [
                outer.prefix_0,
                outer.prefix_1,
                product.prefix_0,
                product.prefix_1,
            ],
            suffix: [
                outer.suffix_0,
                outer.suffix_1,
                product.suffix_0,
                product.suffix_1,
            ],
        }
    }
}

fn run_prefix_host(
    q: [Vec<AkitaField>; 4],
    factors: &SpartanShiftHostFactors,
    challenges: &[AkitaField],
) -> ([AkitaField; 2], [AkitaField; 4]) {
    let mut tables = SpartanShiftPrefixTables {
        p: factors.p.clone(),
        q,
    };
    let mut endpoints = [AkitaField::from_u64(0); 2];
    for &challenge in challenges {
        endpoints = prefix_round_endpoints(&tables).expect("prefix round should execute");
        bind_prefix_tables(&mut tables, challenge).expect("prefix bind should execute");
    }
    assert_eq!(tables.p[0].len(), 1);
    (endpoints, std::array::from_fn(|pair| tables.p[pair][0]))
}

fn run_suffix_host(
    geometry: SpartanShiftGeometry,
    outputs: SpartanShiftOutputs<Vec<AkitaField>>,
    factors: &SpartanShiftHostFactors,
    bound_prefix: [AkitaField; 4],
    suffix_challenges: &[AkitaField],
    gamma: AkitaField,
) -> SpartanShiftOutputs<AkitaField> {
    let combine = |first: usize, second: usize| {
        factors.suffix[first]
            .iter()
            .zip(&factors.suffix[second])
            .map(|(&s0, &s1)| bound_prefix[first] * s0 + bound_prefix[second] * s1)
            .collect()
    };
    let mut state = SpartanShiftDenseState {
        eq_plus_one_outer: combine(0, 1),
        eq_plus_one_product: combine(2, 3),
        unexpanded_pc: outputs.unexpanded_pc,
        pc: outputs.pc,
        is_virtual: outputs.is_virtual,
        is_first_in_sequence: outputs.is_first_in_sequence,
        is_noop: outputs.is_noop,
    };
    for &challenge in suffix_challenges {
        let _ =
            black_box(dense_round_endpoints(&state, gamma).expect("dense round should execute"));
        bind_dense_state(&mut state, challenge).expect("dense bind should execute");
    }
    assert_eq!(suffix_challenges.len(), geometry.suffix_vars());
    jolt_kernels::metal::solinas::spartan_shift::final_outputs(&state)
        .expect("dense state should finish")
}

fn native_planes(
    geometry: SpartanShiftGeometry,
) -> (Vec<u64>, Vec<u64>, Vec<SpartanShiftFlagWord>) {
    let unexpanded_pc = (0..geometry.rows())
        .into_par_iter()
        .map(|row| splitmix(row as u64 ^ 0x243F_6A88_85A3_08D3))
        .collect();
    let pc = (0..geometry.rows())
        .into_par_iter()
        .map(|row| splitmix(row as u64 ^ 0x1319_8A2E_0370_7344))
        .collect();
    let flags = (0..geometry.flag_words())
        .into_par_iter()
        .map(|word| {
            let start = word * 32;
            let end = (start + 32).min(geometry.rows());
            let mut flags = SpartanShiftFlagWord::default();
            for row in start..end {
                let bit = 1u32 << (row - start);
                flags.is_virtual |= u32::from(row % 5 == 1) * bit;
                flags.is_first_in_sequence |= u32::from(row % 17 == 3) * bit;
                flags.is_noop |= u32::from(row % 7 == 0) * bit;
            }
            flags
        })
        .collect();
    (unexpanded_pc, pc, flags)
}

fn values(count: usize, seed: u64) -> Vec<AkitaField> {
    (0..count)
        .map(|index| AkitaField::from_u64(splitmix(seed ^ index as u64)))
        .collect()
}

fn sweep(
    context: &SolinasMetal,
    resident: &SpartanShiftResidentRows,
    r_outer: &[AkitaField],
    r_product: &[AkitaField],
    gamma: AkitaField,
    prefix_challenges: &[AkitaField],
) {
    for strategy in [
        SpartanShiftPrefixStrategy::Mixed,
        SpartanShiftPrefixStrategy::ExpandedHalfWidth,
    ] {
        for build_threads in [64, 128, 256, 512] {
            for high_tile_elements in [32, 64, 128, 256, 512] {
                let config = SpartanShiftKernelConfig {
                    build_threads_per_threadgroup: build_threads,
                    high_tile_elements,
                    fold_threads_per_threadgroup: 32,
                };
                let invocation = context
                    .prepare_spartan_shift_prefix(
                        resident, r_outer, r_product, gamma, config, strategy,
                    )
                    .expect("prefix sweep invocation should prepare");
                let mut samples = (0..5)
                    .map(|_| {
                        let started = Instant::now();
                        let observation = invocation
                            .execute()
                            .expect("prefix sweep invocation should execute");
                        (started.elapsed(), observation.gpu_active)
                    })
                    .collect::<Vec<_>>();
                samples.sort_unstable_by_key(|sample| sample.0);
                let (wall, active) = samples[samples.len() / 2];
                eprintln!(
                    "spartan-shift-sweep phase={strategy:?} build-threads={build_threads} high-tile={high_tile_elements} wall={wall:?} active={active:?} private-bytes={}",
                    invocation.plan().storage.total_resident_bytes - resident.resident_bytes(),
                );
            }
        }
    }
    for fold_threads in [32, 64, 128, 256, 512, 1024] {
        let config = SpartanShiftKernelConfig {
            fold_threads_per_threadgroup: fold_threads,
            ..SpartanShiftKernelConfig::default()
        };
        let invocation = context
            .prepare_spartan_shift_fold(resident, prefix_challenges, config)
            .expect("fold sweep invocation should prepare");
        let mut samples = (0..5)
            .map(|_| {
                let started = Instant::now();
                let observation = invocation
                    .execute()
                    .expect("fold sweep invocation should execute");
                (started.elapsed(), observation.gpu_active)
            })
            .collect::<Vec<_>>();
        samples.sort_unstable_by_key(|sample| sample.0);
        let (wall, active) = samples[samples.len() / 2];
        eprintln!(
            "spartan-shift-sweep phase=fold fold-threads={fold_threads} wall={wall:?} active={active:?} tgmem={}",
            invocation.dynamic_threadgroup_bytes(),
        );
    }
}

fn splitmix(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

fn setting(name: &str, default: usize) -> usize {
    env::var(name).map_or(default, |value| {
        value
            .parse()
            .unwrap_or_else(|_| panic!("{name} should be a positive integer"))
    })
}
