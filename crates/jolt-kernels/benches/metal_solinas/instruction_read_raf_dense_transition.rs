use std::{env, hint::black_box, time::Duration};

use criterion::{BenchmarkId, Criterion, Throughput};
use jolt_kernels::metal::solinas::{
    DenseTransitionTile, Fp128, Product5Config, SolinasMetal, PRODUCT5_FACTORS,
};

use super::reference::values;

const DEFAULT_ELEMENTS: usize = 1 << 20;

pub fn bench(c: &mut Criterion, context: &SolinasMetal) {
    let elements = requested_elements();
    let message_pairs = elements / 4;
    let inner_log2 = requested_inner_log2(message_pairs.trailing_zeros() as usize);
    let e_in = values(1 << inner_log2);
    let e_out = values(message_pairs / e_in.len());
    let source = values(PRODUCT5_FACTORS * elements);
    let challenge = Fp128::from_u128(0x243f_6a88_85a3_08d3_1319_8a2e_0370_7344);
    let useful_products = 8 * elements as u64 + PRODUCT5_FACTORS as u64 * e_out.len() as u64;
    let mut group = c.benchmark_group("metal_solinas/product5_tiled_transition");
    let _ = group
        .sample_size(20)
        .warm_up_time(Duration::from_secs(2))
        .measurement_time(Duration::from_secs(4))
        .throughput(Throughput::Elements(useful_products));

    {
        let invocation = context
            .prepare_product5_fused_transition(
                &source,
                elements,
                challenge,
                &e_in,
                &e_out,
                Product5Config::default(),
            )
            .expect("retained Product5 transition should prepare");
        let _ = group.bench_function(
            BenchmarkId::new("retained_wall", format!("n{elements}_inner{inner_log2}")),
            |bench| {
                bench.iter(|| {
                    invocation
                        .execute_timed()
                        .expect("retained Product5 transition should execute")
                });
            },
        );
        let _ = group.bench_function(
            BenchmarkId::new("retained_active", format!("n{elements}_inner{inner_log2}")),
            |bench| {
                bench.iter_custom(|iterations| {
                    let mut active = Duration::ZERO;
                    for _ in 0..iterations {
                        active += invocation
                            .execute_timed()
                            .expect("retained Product5 transition should execute");
                    }
                    active
                });
            },
        );
    }

    for tile in [
        DenseTransitionTile::Pairs32,
        DenseTransitionTile::Pairs64,
        DenseTransitionTile::Pairs128,
    ] {
        let invocation = context
            .prepare_product5_tiled_transition(&source, elements, challenge, &e_in, &e_out, tile)
            .expect("dense transition candidate should prepare");
        let params = invocation.params();
        eprintln!(
            "instruction-read-raf dense transition tile={} groups={} dynamic_tgmem={} allocated={} logical={} useful={} main_limits={:?} weight_limits={:?} reduction_limits={:?}",
            tile.pairs(),
            params.total_tiles,
            invocation.dynamic_threadgroup_bytes(),
            invocation
                .execute_timed()
                .expect("dense transition warmup should execute")
                .allocated_bytes,
            invocation
                .logical_bytes()
                .expect("dense transition traffic should fit"),
            invocation.useful_products(),
            invocation.main_pipeline_limits(),
            invocation.weight_pipeline_limits(),
            invocation.reduction_pipeline_limits(),
        );
        let suffix = format!("tile{}_n{elements}_inner{inner_log2}", tile.pairs());
        let _ = group.bench_function(BenchmarkId::new("candidate_wall", &suffix), |bench| {
            bench.iter(|| {
                black_box(
                    invocation
                        .execute_timed()
                        .expect("dense transition candidate should execute"),
                )
            });
        });
        let _ = group.bench_function(BenchmarkId::new("candidate_active", suffix), |bench| {
            bench.iter_custom(|iterations| {
                let mut active = Duration::ZERO;
                for _ in 0..iterations {
                    active += invocation
                        .execute_timed()
                        .expect("dense transition candidate should execute")
                        .gpu_active;
                }
                active
            });
        });
    }
    group.finish();
}

fn requested_elements() -> usize {
    let elements = env::var("JOLT_SOLINAS_BENCH_ELEMENTS").map_or(DEFAULT_ELEMENTS, |value| {
        value
            .parse::<usize>()
            .expect("JOLT_SOLINAS_BENCH_ELEMENTS should be a positive integer")
    });
    assert!(
        elements >= 256 && elements.is_power_of_two(),
        "dense transition elements must be a power of two of at least 256"
    );
    elements
}

fn requested_inner_log2(log_pairs: usize) -> usize {
    let inner_log2 = env::var("JOLT_SOLINAS_DENSE_TRANSITION_INNER_LOG2").map_or_else(
        |_| (log_pairs / 2).clamp(6, 10).min(log_pairs),
        |value| {
            value
                .parse::<usize>()
                .expect("JOLT_SOLINAS_DENSE_TRANSITION_INNER_LOG2 should be an integer")
        },
    );
    assert!(
        (6..=log_pairs).contains(&inner_log2),
        "dense transition inner log2 must be in 6..=log2(message pairs)"
    );
    inner_log2
}
