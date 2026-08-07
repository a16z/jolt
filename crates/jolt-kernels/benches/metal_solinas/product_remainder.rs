use std::{
    env,
    hint::black_box,
    time::{Duration, Instant},
};

use criterion::{BenchmarkId, Criterion, Throughput};
use jolt_field::AkitaField;
use jolt_kernels::metal::solinas::{
    ProductRemainderRow, ProductRemainderSequenceConfig, SolinasMetal,
    PRODUCT_REMAINDER_MESSAGE_COLUMNS, PRODUCT_REMAINDER_OPENINGS,
};
use rayon::prelude::*;

const DEFAULT_ELEMENTS: [usize; 3] = [1 << 16, 1 << 20, 1 << 22];
const VALIDATION_ELEMENTS: usize = 1 << 8;

pub fn bench(c: &mut Criterion, context: &SolinasMetal) {
    validate(context);

    let materialize_threads = setting("JOLT_METAL_PRODUCT_REMAINDER_MATERIALIZE_THREADS", 128);
    let transition_threads = setting("JOLT_METAL_PRODUCT_REMAINDER_TRANSITION_THREADS", 64);
    let openings_threads = setting("JOLT_METAL_PRODUCT_REMAINDER_OPENINGS_THREADS", 128);
    let cutoff = setting("JOLT_METAL_PRODUCT_REMAINDER_CUTOFF", 1 << 16);
    let service_only = env::var_os("JOLT_METAL_PRODUCT_REMAINDER_SERVICE_ONLY").is_some();
    let config = ProductRemainderSequenceConfig {
        uniskip_threads_per_threadgroup: Some(64),
        materialize_threads_per_threadgroup: Some(materialize_threads),
        transition_threads_per_threadgroup: Some(transition_threads),
        openings_threads_per_threadgroup: Some(openings_threads),
    };
    let mut group = c.benchmark_group("metal_sumcheck/product_remainder");
    let _ = group
        .sample_size(10)
        .warm_up_time(Duration::from_secs(2))
        .measurement_time(Duration::from_secs(5));

    for elements in cases() {
        let rows = rows(elements);
        let lagrange = lagrange();
        let (e_in_capacity, e_out_capacity) = split_factors(elements);
        let setup_start = Instant::now();
        let mut sequence = context
            .prepare_product_remainder_sequence(
                &rows,
                lagrange,
                e_in_capacity,
                e_out_capacity,
                config,
            )
            .expect("product remainder sequence should prepare");
        let setup_wall = setup_start.elapsed();
        assert_eq!(sequence.round_device_buffer_allocations(), 0);

        let (materialize_e_in_length, materialize_e_out_length) = split_factors(elements / 2);
        let materialize_e_in = weights(materialize_e_in_length, 1);
        let materialize_e_out = weights(materialize_e_out_length, 2);
        let materialize_start = Instant::now();
        let (first_message, first_active) = sequence
            .message_timed(&materialize_e_in, &materialize_e_out)
            .expect("product remainder first message should execute");
        let mut service_wall = materialize_start.elapsed();
        let mut gated_active = first_active;
        assert_eq!(
            sequence
                .replay_materialize_message_timed(&materialize_e_in, &materialize_e_out)
                .expect("product remainder first message should replay")
                .0,
            first_message
        );

        let suffix = format!("n{elements}_tg{materialize_threads}_cutoff{cutoff}");
        let materialize_products = 5 * elements + 2 * materialize_e_out.len();
        if !service_only {
            let _ = group.throughput(Throughput::Elements(materialize_products as u64));
            let _ = group.bench_function(
                BenchmarkId::new("metal_wall_materialize", &suffix),
                |bench| {
                    bench.iter(|| {
                        black_box(
                            sequence
                                .replay_materialize_message_timed(
                                    &materialize_e_in,
                                    &materialize_e_out,
                                )
                                .expect("product remainder first message should replay")
                                .0,
                        )
                    });
                },
            );
            let _ = group.bench_function(
                BenchmarkId::new("metal_active_materialize", &suffix),
                |bench| {
                    bench.iter_custom(|iterations| {
                        let mut active = Duration::ZERO;
                        for _ in 0..iterations {
                            active += sequence
                                .replay_materialize_message_timed(
                                    &materialize_e_in,
                                    &materialize_e_out,
                                )
                                .expect("timed product remainder first message should replay")
                                .1;
                        }
                        active
                    });
                },
            );
        }

        let mut round = 0usize;
        while sequence.current_elements() > 2 {
            let source_elements = sequence.current_elements();
            let (e_in_length, e_out_length) = split_factors(source_elements / 4);
            let e_in = weights(e_in_length, 3 + 2 * round);
            let e_out = weights(e_out_length, 4 + 2 * round);
            let challenge =
                AkitaField::from_u64(0x9e37_79b9_7f4a_7c15_u64.wrapping_mul(round as u64 + 1));
            let replay = sequence
                .replay_current_bind_and_message_timed(challenge, &e_in, &e_out)
                .expect("product remainder transition should replay")
                .0;

            if source_elements > cutoff && !service_only {
                let transition_suffix =
                    format!("r{round}_n{source_elements}_tg{transition_threads}_cutoff{cutoff}");
                let products = 2 * source_elements + 2 * e_out.len();
                let _ = group.throughput(Throughput::Elements(products as u64));
                let _ = group.bench_function(
                    BenchmarkId::new("metal_wall_transition", &transition_suffix),
                    |bench| {
                        bench.iter(|| {
                            black_box(
                                sequence
                                    .replay_current_bind_and_message_timed(challenge, &e_in, &e_out)
                                    .expect("product remainder transition should replay")
                                    .0,
                            )
                        });
                    },
                );
                let _ = group.bench_function(
                    BenchmarkId::new("metal_active_transition", &transition_suffix),
                    |bench| {
                        bench.iter_custom(|iterations| {
                            let mut active = Duration::ZERO;
                            for _ in 0..iterations {
                                active += sequence
                                    .replay_current_bind_and_message_timed(challenge, &e_in, &e_out)
                                    .expect("timed product remainder transition should replay")
                                    .1;
                            }
                            active
                        });
                    },
                );
            }

            let transition_start = Instant::now();
            let (message, active) = sequence
                .bind_and_message_timed(challenge, &e_in, &e_out)
                .expect("product remainder transition should advance");
            service_wall += transition_start.elapsed();
            assert_eq!(message, replay);
            if source_elements > cutoff {
                gated_active += active;
            }
            round += 1;
        }

        let (opening_e_in_length, opening_e_out_length) = split_factors(elements);
        let opening_e_in = weights(opening_e_in_length, 101);
        let opening_e_out = weights(opening_e_out_length, 102);
        let opening_start = Instant::now();
        let (opening_values, opening_active) = sequence
            .openings_timed(&opening_e_in, &opening_e_out)
            .expect("product remainder openings should execute");
        service_wall += opening_start.elapsed();
        gated_active += opening_active;
        assert_eq!(
            sequence
                .replay_openings_timed(&opening_e_in, &opening_e_out)
                .expect("product remainder openings should replay")
                .0,
            opening_values
        );

        let opening_suffix = format!("n{elements}_tg{openings_threads}_cutoff{cutoff}");
        let opening_products = 3 * elements + PRODUCT_REMAINDER_OPENINGS * opening_e_out.len();
        if !service_only {
            let _ = group.throughput(Throughput::Elements(opening_products as u64));
            let _ = group.bench_function(
                BenchmarkId::new("metal_wall_openings", &opening_suffix),
                |bench| {
                    bench.iter(|| {
                        black_box(
                            sequence
                                .replay_openings_timed(&opening_e_in, &opening_e_out)
                                .expect("product remainder openings should replay")
                                .0,
                        )
                    });
                },
            );
            let _ = group.bench_function(
                BenchmarkId::new("metal_active_openings", &opening_suffix),
                |bench| {
                    bench.iter_custom(|iterations| {
                        let mut active = Duration::ZERO;
                        for _ in 0..iterations {
                            active += sequence
                                .replay_openings_timed(&opening_e_in, &opening_e_out)
                                .expect("timed product remainder openings should replay")
                                .1;
                        }
                        active
                    });
                },
            );
        }

        let cold_full_active = sequence.gpu_active_time();
        let warm_start = Instant::now();
        let phase_start = Instant::now();
        let (warm_first, mut warm_full_active) = sequence
            .restart_message_timed(&materialize_e_in, &materialize_e_out)
            .expect("product remainder warm sequence should restart");
        let mut warm_gated_wall = phase_start.elapsed();
        let mut warm_gated_active = warm_full_active;
        assert_eq!(warm_first, first_message);
        let mut warm_round = 0usize;
        while sequence.current_elements() > 2 {
            let source_elements = sequence.current_elements();
            let (e_in_length, e_out_length) = split_factors(source_elements / 4);
            let e_in = weights(e_in_length, 3 + 2 * warm_round);
            let e_out = weights(e_out_length, 4 + 2 * warm_round);
            let challenge =
                AkitaField::from_u64(0x9e37_79b9_7f4a_7c15_u64.wrapping_mul(warm_round as u64 + 1));
            let phase_start = Instant::now();
            let (_, active) = sequence
                .bind_and_message_timed(challenge, &e_in, &e_out)
                .expect("product remainder warm transition should execute");
            let phase_wall = phase_start.elapsed();
            warm_full_active += active;
            if source_elements > cutoff {
                warm_gated_wall += phase_wall;
                warm_gated_active += active;
            }
            warm_round += 1;
        }
        let phase_start = Instant::now();
        let (warm_openings, active) = sequence
            .openings_timed(&opening_e_in, &opening_e_out)
            .expect("product remainder warm openings should execute");
        warm_gated_wall += phase_start.elapsed();
        warm_full_active += active;
        warm_gated_active += active;
        let warm_full_wall = warm_start.elapsed();
        assert_eq!(warm_openings, opening_values);

        eprintln!(
            "product-remainder n={elements} setup={setup_wall:?} cold-wall={service_wall:?} cold-gated-active={gated_active:?} cold-full-active={:?} warm-gated-wall={warm_gated_wall:?} warm-gated-active={warm_gated_active:?} warm-full-wall={warm_full_wall:?} warm-full-active={warm_full_active:?} resident-bytes={} rounds={round}",
            cold_full_active,
            sequence.storage_layout().resident_bytes(),
        );
    }
    group.finish();
}

fn validate(context: &SolinasMetal) {
    let elements = VALIDATION_ELEMENTS;
    let rows = rows(elements);
    let lagrange = lagrange();
    let (capacity_in, capacity_out) = split_factors(elements);
    let mut sequence = context
        .prepare_product_remainder_sequence(
            &rows,
            lagrange,
            capacity_in,
            capacity_out,
            ProductRemainderSequenceConfig::default(),
        )
        .expect("product remainder validation sequence should prepare");
    let (e_in_length, e_out_length) = split_factors(elements / 2);
    let e_in = weights(e_in_length, 1);
    let e_out = weights(e_out_length, 2);
    let (mut state, expected) = oracle_materialize(&rows, lagrange, &e_in, &e_out);
    assert_eq!(
        sequence
            .message(&e_in, &e_out)
            .expect("product remainder validation message should execute"),
        expected
    );

    let mut round = 0usize;
    while sequence.current_elements() > 2 {
        let source_elements = sequence.current_elements();
        let (e_in_length, e_out_length) = split_factors(source_elements / 4);
        let e_in = weights(e_in_length, 3 + 2 * round);
        let e_out = weights(e_out_length, 4 + 2 * round);
        let challenge =
            AkitaField::from_u64(0x9e37_79b9_7f4a_7c15_u64.wrapping_mul(round as u64 + 1));
        let (next_state, expected) =
            oracle_transition(&state, source_elements, challenge, &e_in, &e_out);
        assert_eq!(
            sequence
                .bind_and_message(challenge, &e_in, &e_out)
                .expect("product remainder validation transition should execute"),
            expected
        );
        state = next_state;
        round += 1;
    }

    let (e_in_length, e_out_length) = split_factors(elements);
    let e_in = weights(e_in_length, 101);
    let e_out = weights(e_out_length, 102);
    assert_eq!(
        sequence
            .openings(&e_in, &e_out)
            .expect("product remainder validation openings should execute"),
        oracle_openings(&rows, &e_in, &e_out)
    );
}

fn oracle_materialize(
    rows: &[ProductRemainderRow],
    lagrange: [AkitaField; 3],
    e_in: &[AkitaField],
    e_out: &[AkitaField],
) -> (
    Vec<AkitaField>,
    [AkitaField; PRODUCT_REMAINDER_MESSAGE_COLUMNS],
) {
    let mut state = vec![AkitaField::zero(); 2 * rows.len()];
    let mut endpoints = [AkitaField::zero(); PRODUCT_REMAINDER_MESSAGE_COLUMNS];
    for (x_out, &outer_weight) in e_out.iter().enumerate() {
        let mut inner = [AkitaField::zero(); PRODUCT_REMAINDER_MESSAGE_COLUMNS];
        for (x_in, &inner_weight) in e_in.iter().enumerate() {
            let pair = x_out * e_in.len() + x_in;
            let low = 2 * pair;
            let high = low + 1;
            let (left_low, right_low) = rows[low].relation_values(&lagrange);
            let (left_high, right_high) = rows[high].relation_values(&lagrange);
            state[low] = left_low;
            state[high] = left_high;
            state[rows.len() + low] = right_low;
            state[rows.len() + high] = right_high;
            inner[0] += inner_weight * left_low * right_low;
            inner[1] += inner_weight * (left_high - left_low) * (right_high - right_low);
        }
        for (endpoint, value) in endpoints.iter_mut().zip(inner) {
            *endpoint += outer_weight * value;
        }
    }
    (state, endpoints)
}

fn oracle_transition(
    state: &[AkitaField],
    source_elements: usize,
    challenge: AkitaField,
    e_in: &[AkitaField],
    e_out: &[AkitaField],
) -> (
    Vec<AkitaField>,
    [AkitaField; PRODUCT_REMAINDER_MESSAGE_COLUMNS],
) {
    let bound_elements = source_elements / 2;
    let mut bound = vec![AkitaField::zero(); 2 * bound_elements];
    let mut endpoints = [AkitaField::zero(); PRODUCT_REMAINDER_MESSAGE_COLUMNS];
    let bind = |low: AkitaField, high: AkitaField| low + challenge * (high - low);
    for (x_out, &outer_weight) in e_out.iter().enumerate() {
        let mut inner = [AkitaField::zero(); PRODUCT_REMAINDER_MESSAGE_COLUMNS];
        for (x_in, &inner_weight) in e_in.iter().enumerate() {
            let pair = x_out * e_in.len() + x_in;
            let source = 4 * pair;
            let destination = 2 * pair;
            let left_0 = bind(state[source], state[source + 1]);
            let left_1 = bind(state[source + 2], state[source + 3]);
            let right_0 = bind(
                state[source_elements + source],
                state[source_elements + source + 1],
            );
            let right_1 = bind(
                state[source_elements + source + 2],
                state[source_elements + source + 3],
            );
            bound[destination] = left_0;
            bound[destination + 1] = left_1;
            bound[bound_elements + destination] = right_0;
            bound[bound_elements + destination + 1] = right_1;
            inner[0] += inner_weight * left_0 * right_0;
            inner[1] += inner_weight * (left_1 - left_0) * (right_1 - right_0);
        }
        for (endpoint, value) in endpoints.iter_mut().zip(inner) {
            *endpoint += outer_weight * value;
        }
    }
    (bound, endpoints)
}

fn oracle_openings(
    rows: &[ProductRemainderRow],
    e_in: &[AkitaField],
    e_out: &[AkitaField],
) -> [AkitaField; PRODUCT_REMAINDER_OPENINGS] {
    let mut sums = [AkitaField::zero(); PRODUCT_REMAINDER_OPENINGS];
    for (x_out, &outer_weight) in e_out.iter().enumerate() {
        for (x_in, &inner_weight) in e_in.iter().enumerate() {
            let fields = rows[x_out * e_in.len() + x_in].fields::<AkitaField>();
            let weight = outer_weight * inner_weight;
            for (sum, value) in sums.iter_mut().zip(fields) {
                *sum += weight * value;
            }
        }
    }
    sums
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
                _ => index as i128 * 0x1_0000_0001 - 0x1234_5678,
            };
            ProductRemainderRow::new(
                (index as u64)
                    .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                    .rotate_left((index % 63) as u32),
                right_input,
                index & 1 != 0,
                index % 3 == 0,
                (!(index as u64)).wrapping_mul(0xbf58_476d_1ce4_e5b9),
                index % 5 == 0,
                index % 7 == 0,
                index % 11 == 0,
            )
        })
        .collect()
}

fn lagrange() -> [AkitaField; 3] {
    [
        AkitaField::from_u64(0x101),
        AkitaField::from_u64(0x1001),
        AkitaField::from_u64(0x1_0001),
    ]
}

fn weights(length: usize, salt: usize) -> Vec<AkitaField> {
    (0..length)
        .map(|index| {
            AkitaField::from_u64(
                (index as u64)
                    .wrapping_mul(0x94d0_49bb_1331_11eb)
                    .wrapping_add(salt as u64),
            )
        })
        .collect()
}

fn split_factors(elements: usize) -> (usize, usize) {
    assert!(elements.is_power_of_two());
    let e_in = 1usize << (elements.ilog2() as usize / 2);
    (e_in, elements / e_in)
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
            .all(|elements| elements.is_power_of_two() && *elements >= 1 << 8),
        "product remainder benchmark sizes must be powers of two at least 2^8"
    );
    cases
}
