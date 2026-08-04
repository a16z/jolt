use std::{env, hint::black_box, mem::size_of, time::Duration, time::Instant};

use criterion::{measurement::WallTime, BenchmarkGroup, BenchmarkId, Criterion, Throughput};
use jolt_field::AkitaField;
use jolt_kernels::metal::solinas::{
    BytecodeCycleSequenceConfig, BytecodeCycleTables, BytecodeCycleTablesMut, SolinasMetal,
    BYTECODE_CYCLE_TABLES,
};
use rayon::prelude::*;

const DEFAULT_ELEMENTS: [usize; 3] = [1 << 16, 1 << 20, 1 << 22];

pub fn bench(c: &mut Criterion, context: &SolinasMetal) {
    let config = BytecodeCycleSequenceConfig {
        message_threads_per_threadgroup: Some(env_usize(
            "JOLT_METAL_BYTECODE_MESSAGE_THREADS",
            256,
        )),
        transition_threads_per_threadgroup: Some(env_usize(
            "JOLT_METAL_BYTECODE_TRANSITION_THREADS",
            128,
        )),
        max_threadgroups: env_usize("JOLT_METAL_BYTECODE_MAX_THREADGROUPS", 1 << 13),
    };
    validate(context, config);

    let mut group = c.benchmark_group("metal_sumcheck/bytecode_cycle_dense_q10");
    let _ = group
        .sample_size(10)
        .warm_up_time(Duration::from_secs(2))
        .measurement_time(Duration::from_secs(5));
    for elements in cases() {
        assert!(elements >= 4 && elements.is_power_of_two());
        assert!(
            (elements as u64) * size_of::<AkitaField>() as u64
                <= context.device_info().max_buffer_length,
            "one Bytecode factor exceeds the Metal buffer limit"
        );
        let tables = test_tables(elements);
        let _ = group.throughput(Throughput::Elements(elements as u64));
        bench_message(&mut group, context, &tables, elements, config);
        bench_transition(&mut group, context, &tables, elements, config);
    }
    group.finish();
}

fn bench_message(
    group: &mut BenchmarkGroup<'_, WallTime>,
    context: &SolinasMetal,
    tables: &[AkitaField],
    elements: usize,
    config: BytecodeCycleSequenceConfig,
) {
    let suffix = format!("n{elements}");
    let _ = group.bench_function(BenchmarkId::new("cpu_q10_message", &suffix), |bench| {
        bench.iter(|| black_box(cpu_message(black_box(tables), elements)));
    });

    let mut sequence = context
        .prepare_bytecode_cycle_sequence(table_views(tables, elements), config)
        .expect("Bytecode dense sequence should prepare");
    let expected = cpu_message(tables, elements);
    assert_eq!(
        sequence.message().expect("validation message should run"),
        expected
    );
    let _ = group.bench_function(BenchmarkId::new("metal_wall_message", &suffix), |bench| {
        bench.iter(|| black_box(sequence.message().expect("Metal message should run")));
    });
    let _ = group.bench_function(BenchmarkId::new("metal_active_message", &suffix), |bench| {
        bench.iter_custom(|iterations| {
            let before = sequence.gpu_active_time();
            for _ in 0..iterations {
                let _ = black_box(sequence.message().expect("Metal message should run"));
            }
            sequence
                .gpu_active_time()
                .checked_sub(before)
                .expect("GPU active time should be monotonic")
        });
    });
}

fn bench_transition(
    group: &mut BenchmarkGroup<'_, WallTime>,
    context: &SolinasMetal,
    tables: &[AkitaField],
    elements: usize,
    config: BytecodeCycleSequenceConfig,
) {
    let challenge = -AkitaField::from_u64(0x9e37_79b9);
    let suffix = format!("n{elements}");
    let mut bound = vec![AkitaField::zero(); BYTECODE_CYCLE_TABLES * elements / 2];
    let _ = group.bench_function(BenchmarkId::new("cpu_bind_q10_message", &suffix), |bench| {
        bench.iter_custom(|iterations| {
            let mut measured = Duration::ZERO;
            for _ in 0..iterations {
                let start = Instant::now();
                cpu_bind(tables, &mut bound, elements, challenge);
                let message = black_box(cpu_message(&bound, elements / 2));
                measured += start.elapsed();
                let _ = black_box(message);
            }
            measured
        });
    });

    let mut sequence = context
        .prepare_bytecode_cycle_sequence(table_views(tables, elements), config)
        .expect("Bytecode dense sequence should prepare");
    cpu_bind(tables, &mut bound, elements, challenge);
    assert_eq!(
        sequence
            .bind_and_message(challenge)
            .expect("validation transition should run"),
        cpu_message(&bound, elements / 2)
    );
    let mut actual_bound = vec![AkitaField::zero(); bound.len()];
    sequence
        .read_current_tables(table_views_mut(&mut actual_bound, elements / 2))
        .expect("validation readback should succeed");
    assert_eq!(actual_bound, bound);
    sequence
        .rewind_initial_state()
        .expect("first transition should rewind without a copy");
    let _ = group.bench_function(
        BenchmarkId::new("metal_wall_bind_message", &suffix),
        |bench| {
            bench.iter_custom(|iterations| {
                let mut measured = Duration::ZERO;
                for _ in 0..iterations {
                    let start = Instant::now();
                    let message = black_box(
                        sequence
                            .bind_and_message(challenge)
                            .expect("Metal transition should run"),
                    );
                    measured += start.elapsed();
                    sequence
                        .rewind_initial_state()
                        .expect("first transition should rewind without a copy");
                    let _ = black_box(message);
                }
                measured
            });
        },
    );
    let _ = group.bench_function(
        BenchmarkId::new("metal_active_bind_message", &suffix),
        |bench| {
            bench.iter_custom(|iterations| {
                let before = sequence.gpu_active_time();
                for _ in 0..iterations {
                    let message = black_box(
                        sequence
                            .bind_and_message(challenge)
                            .expect("Metal transition should run"),
                    );
                    sequence
                        .rewind_initial_state()
                        .expect("first transition should rewind without a copy");
                    let _ = black_box(message);
                }
                sequence
                    .gpu_active_time()
                    .checked_sub(before)
                    .expect("GPU active time should be monotonic")
            });
        },
    );
}

fn validate(context: &SolinasMetal, config: BytecodeCycleSequenceConfig) {
    let elements = 1 << 13;
    let tables = test_tables(elements);
    let challenge = AkitaField::from_u64(0xbb67_ae85);
    let mut expected_bound = vec![AkitaField::zero(); BYTECODE_CYCLE_TABLES * elements / 2];
    cpu_bind(&tables, &mut expected_bound, elements, challenge);
    let mut sequence = context
        .prepare_bytecode_cycle_sequence(table_views(&tables, elements), config)
        .expect("validation Bytecode sequence should prepare");
    assert_eq!(
        sequence.message().expect("validation message should run"),
        cpu_message(&tables, elements)
    );
    assert_eq!(
        sequence
            .bind_and_message(challenge)
            .expect("validation transition should run"),
        cpu_message(&expected_bound, elements / 2)
    );
    let mut actual_bound = vec![AkitaField::zero(); expected_bound.len()];
    sequence
        .read_current_tables(table_views_mut(&mut actual_bound, elements / 2))
        .expect("validation readback should succeed");
    assert_eq!(actual_bound, expected_bound);
}

fn test_tables(elements: usize) -> Vec<AkitaField> {
    (0..BYTECODE_CYCLE_TABLES)
        .flat_map(|table| {
            (0..elements).map(move |index| {
                let value = AkitaField::from_u64(19 + 97 * table as u64 + 131 * index as u64);
                if table == 2 && index % 3 == 0 {
                    -value
                } else {
                    value
                }
            })
        })
        .collect()
}

fn table_views(tables: &[AkitaField], elements: usize) -> BytecodeCycleTables<'_> {
    assert_eq!(tables.len(), BYTECODE_CYCLE_TABLES * elements);
    let mut planes = tables.chunks_exact(elements);
    BytecodeCycleTables {
        combined: planes.next().expect("combined plane should exist"),
        fused_combined: planes.next().expect("fused combined plane should exist"),
        fused_inc: planes.next().expect("fused increment plane should exist"),
        ra0: planes.next().expect("RA0 plane should exist"),
        ra1: planes.next().expect("RA1 plane should exist"),
    }
}

fn table_views_mut(tables: &mut [AkitaField], elements: usize) -> BytecodeCycleTablesMut<'_> {
    assert_eq!(tables.len(), BYTECODE_CYCLE_TABLES * elements);
    let mut planes = tables.chunks_exact_mut(elements);
    BytecodeCycleTablesMut {
        combined: planes.next().expect("combined plane should exist"),
        fused_combined: planes.next().expect("fused combined plane should exist"),
        fused_inc: planes.next().expect("fused increment plane should exist"),
        ra0: planes.next().expect("RA0 plane should exist"),
        ra1: planes.next().expect("RA1 plane should exist"),
    }
}

fn cpu_bind(
    source: &[AkitaField],
    bound: &mut [AkitaField],
    elements: usize,
    challenge: AkitaField,
) {
    let bound_elements = elements / 2;
    bound
        .par_chunks_mut(bound_elements)
        .enumerate()
        .for_each(|(table, output)| {
            let source = &source[table * elements..(table + 1) * elements];
            output
                .par_iter_mut()
                .enumerate()
                .for_each(|(index, output)| {
                    let lo = source[2 * index];
                    let hi = source[2 * index + 1];
                    *output = lo + challenge * (hi - lo);
                });
        });
}

fn cpu_message(tables: &[AkitaField], elements: usize) -> [AkitaField; 4] {
    (0..elements / 2)
        .into_par_iter()
        .fold(
            || [AkitaField::zero(); 4],
            |mut acc, pair| {
                let lo = std::array::from_fn(|table| tables[table * elements + 2 * pair]);
                let hi = std::array::from_fn(|table| tables[table * elements + 2 * pair + 1]);
                for (acc, value) in acc.iter_mut().zip(q10(lo, hi)) {
                    *acc += value;
                }
                acc
            },
        )
        .reduce(
            || [AkitaField::zero(); 4],
            |mut left, right| {
                for (left, right) in left.iter_mut().zip(right) {
                    *left += right;
                }
                left
            },
        )
}

fn q10(lo: [AkitaField; 5], hi: [AkitaField; 5]) -> [AkitaField; 4] {
    let ra = grid_from_anchors(
        lo[3] * lo[4],
        hi[3] * hi[4],
        (hi[3] - lo[3]) * (hi[4] - lo[4]),
    );
    let coefficient = grid_from_anchors(
        lo[0] + lo[2] * lo[1],
        hi[0] + hi[2] * hi[1],
        (hi[2] - lo[2]) * (hi[1] - lo[1]),
    );
    std::array::from_fn(|sample| ra[sample] * coefficient[sample])
}

fn grid_from_anchors(
    at_zero: AkitaField,
    at_one: AkitaField,
    leading: AkitaField,
) -> [AkitaField; 4] {
    let second_difference = leading + leading;
    let delta_two = at_one - at_zero + second_difference;
    let at_two = at_one + delta_two;
    let delta_three = delta_two + second_difference;
    let at_three = at_two + delta_three;
    [
        at_zero,
        at_two,
        at_three,
        at_three + delta_three + second_difference,
    ]
}

fn cases() -> Vec<usize> {
    env::var("JOLT_SOLINAS_BENCH_ELEMENTS").map_or_else(
        |_| DEFAULT_ELEMENTS.to_vec(),
        |value| {
            vec![value
                .parse()
                .expect("JOLT_SOLINAS_BENCH_ELEMENTS should be a positive integer")]
        },
    )
}

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name).map_or(default, |value| {
        value
            .parse()
            .unwrap_or_else(|_| panic!("{name} should be a positive integer"))
    })
}
