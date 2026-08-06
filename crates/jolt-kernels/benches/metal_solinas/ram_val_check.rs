use std::{env, hint::black_box, time::Duration, time::Instant};

use criterion::{BenchmarkId, Criterion, Throughput};
use jolt_field::AkitaField;
use jolt_kernels::metal::solinas::{
    ram_val_check_oracle, RamValCheckConfig, RamValCheckDenseRow, RamValCheckNativeRow,
    RamValCheckPlan, SolinasMetal, RAM_VAL_CHECK_TARGET_CPU_NS,
};
use jolt_poly::{EqPolynomial, LtPolynomial};
use rayon::prelude::*;

const DEFAULT_ELEMENTS: [usize; 3] = [1 << 16, 1 << 20, 1 << 22];
const VALIDATION_LOG_T: usize = 12;

pub fn bench(c: &mut Criterion, context: &SolinasMetal) {
    validate(context);

    let warmup_ms = setting("JOLT_METAL_RAM_VAL_CHECK_WARMUP_MS", 2_000);
    let measurement_ms = setting("JOLT_METAL_RAM_VAL_CHECK_MEASUREMENT_MS", 5_000);
    let first_threads = setting("JOLT_METAL_RAM_VAL_CHECK_FIRST_THREADS", 32);
    let native_threads = setting("JOLT_METAL_RAM_VAL_CHECK_NATIVE_THREADS", 32);
    let dense_threads = setting("JOLT_METAL_RAM_VAL_CHECK_DENSE_THREADS", 32);
    let service_only = env::var_os("JOLT_METAL_RAM_VAL_CHECK_SERVICE_ONLY").is_some();
    let observe_only = env::var_os("JOLT_METAL_RAM_VAL_CHECK_OBSERVE_ONLY").is_some();
    let mut group = c.benchmark_group("metal_sumcheck/ram_val_check");
    let _ = group
        .sample_size(10)
        .warm_up_time(Duration::from_millis(warmup_ms as u64))
        .measurement_time(Duration::from_millis(measurement_ms as u64));

    for elements in cases() {
        let log_t = elements.ilog2() as usize;
        let log_k = 13.min(log_t / 2);
        let gpu_binds = setting("JOLT_METAL_RAM_VAL_CHECK_GPU_BINDS", 10.min(log_t / 2 - 1));
        let config = RamValCheckConfig {
            first_message_threads: first_threads,
            native_transition_threads: native_threads,
            dense_transition_threads: dense_threads,
            cpu_tail_elements: elements >> gpu_binds,
        };
        let plan = RamValCheckPlan::new(log_t, log_k, config)
            .expect("RAM value-check benchmark plan should be valid");
        let rows = rows(elements, plan.address_domain());
        let (eq_address, lt_lo, lt_hi, eq_hi) = factor_tables(log_t, log_k, log_t / 2);
        let all_challenges = challenges(log_t);
        let prefix_challenges = &all_challenges[..plan.gpu_bind_rounds()];
        let bound_lt_tables = bound_lt_tables(lt_lo.clone(), prefix_challenges);

        let upload_start = Instant::now();
        let resident_rows = context
            .prepare_ram_val_check_rows(&rows, plan.address_domain())
            .expect("RAM value-check rows should prepare");
        let upload_wall = upload_start.elapsed();
        let setup_start = Instant::now();
        let mut sequence = context
            .prepare_ram_val_check_sequence(
                resident_rows.clone(),
                &eq_address,
                &lt_lo,
                &lt_hi,
                &eq_hi,
                plan,
            )
            .expect("RAM value-check sequence should prepare");
        let setup_wall = setup_start.elapsed();
        assert_eq!(sequence.round_device_buffer_allocations(), 0);
        assert_eq!(
            sequence.row_allocation_identity(),
            resident_rows.allocation_identity()
        );

        let first_start = Instant::now();
        let (first_message, first_active) = sequence
            .message_timed()
            .expect("RAM value-check first message should execute");
        let mut cold_wall = first_start.elapsed();
        assert_eq!(
            sequence
                .replay_first_message_timed()
                .expect("RAM value-check first message should replay")
                .0,
            first_message
        );

        let suffix =
            format!("n{elements}_first{first_threads}_native{native_threads}_dense{dense_threads}");
        if !service_only {
            let _ = group.throughput(Throughput::Elements(elements as u64));
            let _ = group.bench_function(
                BenchmarkId::new("metal_wall_first_message", &suffix),
                |bench| {
                    bench.iter(|| {
                        black_box(
                            sequence
                                .replay_first_message_timed()
                                .expect("RAM value-check first message should replay")
                                .0,
                        )
                    });
                },
            );
            let _ = group.bench_function(
                BenchmarkId::new("metal_active_first_message", &suffix),
                |bench| {
                    bench.iter_custom(|iterations| {
                        let mut active = Duration::ZERO;
                        for _ in 0..iterations {
                            active += sequence
                                .replay_first_message_timed()
                                .expect("timed RAM value-check first message should replay")
                                .1;
                        }
                        active
                    });
                },
            );
        }

        let mut phase_active = vec![first_active];
        for (round, (&challenge, bound_lt_lo)) in
            prefix_challenges.iter().zip(&bound_lt_tables).enumerate()
        {
            let replay = sequence
                .replay_current_bind_and_message_timed(challenge, bound_lt_lo)
                .expect("RAM value-check transition should replay")
                .0;
            if !service_only {
                let phase = if round == 0 { "native" } else { "dense" };
                let phase_suffix = format!("{phase}_r{round}_{suffix}");
                let _ = group.bench_function(
                    BenchmarkId::new("metal_wall_transition", &phase_suffix),
                    |bench| {
                        bench.iter(|| {
                            black_box(
                                sequence
                                    .replay_current_bind_and_message_timed(challenge, bound_lt_lo)
                                    .expect("RAM value-check transition should replay")
                                    .0,
                            )
                        });
                    },
                );
                let _ = group.bench_function(
                    BenchmarkId::new("metal_active_transition", &phase_suffix),
                    |bench| {
                        bench.iter_custom(|iterations| {
                            let mut active = Duration::ZERO;
                            for _ in 0..iterations {
                                active += sequence
                                    .replay_current_bind_and_message_timed(challenge, bound_lt_lo)
                                    .expect("timed RAM value-check transition should replay")
                                    .1;
                            }
                            active
                        });
                    },
                );
            }
            let phase_start = Instant::now();
            let (message, active) = sequence
                .bind_and_message_timed(challenge, bound_lt_lo)
                .expect("RAM value-check transition should advance");
            cold_wall += phase_start.elapsed();
            assert_eq!(message, replay);
            phase_active.push(active);
        }
        let handoff_start = Instant::now();
        let mut tail = vec![RamValCheckDenseRow::default(); plan.cpu_tail_elements()];
        sequence
            .read_current_state_into(&mut tail)
            .expect("RAM value-check tail should read");
        cold_wall += handoff_start.elapsed();
        assert_eq!(tail.len(), plan.cpu_tail_elements());

        let warm_start = Instant::now();
        let (_, warm_first_active) = sequence
            .restart_message_timed()
            .expect("RAM value-check prefix should restart");
        let mut warm_phase_active = vec![warm_first_active];
        for (&challenge, bound_lt_lo) in prefix_challenges.iter().zip(&bound_lt_tables) {
            let (_, active) = sequence
                .bind_and_message_timed(challenge, bound_lt_lo)
                .expect("RAM value-check warm transition should execute");
            warm_phase_active.push(active);
        }
        let mut warm_tail = vec![RamValCheckDenseRow::default(); plan.cpu_tail_elements()];
        sequence
            .read_current_state_into(&mut warm_tail)
            .expect("RAM value-check warm tail should read");
        let warm_wall = warm_start.elapsed();
        assert_eq!(warm_tail, tail);
        let warm_active = sequence.gpu_active_time();
        let tail_start = Instant::now();
        let dense_lt = dense_lt_table(
            bound_lt_tables
                .last()
                .expect("RAM value-check prefix must bind at least once"),
            &lt_hi,
            &eq_hi,
        );
        let tail_result = cpu_tail(
            warm_tail.clone(),
            dense_lt,
            &all_challenges[plan.gpu_bind_rounds()..],
        );
        let tail_wall = tail_start.elapsed();
        let hybrid_no_fs_wall = warm_wall + tail_wall;
        let _ = black_box(tail_result);

        eprintln!(
            "ram-val-check n={elements} gpu-binds={} upload={upload_wall:?} setup={setup_wall:?} cold-wall={cold_wall:?} warm-prefix-wall={warm_wall:?} warm-active={warm_active:?} cpu-tail-wall={tail_wall:?} hybrid-no-fs-wall={hybrid_no_fs_wall:?} cold-phase-active={phase_active:?} warm-phase-active={warm_phase_active:?} resident-bytes={} tail-bytes={} frozen-cpu-ms={:.6} resident-prefix-ratio={:.6} hybrid-no-fs-ratio={:.6}",
            plan.gpu_bind_rounds(),
            sequence.storage_layout().resident_bytes(),
            sequence.storage_layout().tail_handoff_bytes(),
            RAM_VAL_CHECK_TARGET_CPU_NS as f64 / 1e6,
            RAM_VAL_CHECK_TARGET_CPU_NS as f64 / warm_wall.as_nanos() as f64,
            RAM_VAL_CHECK_TARGET_CPU_NS as f64 / hybrid_no_fs_wall.as_nanos() as f64,
        );

        if observe_only {
            continue;
        }

        let _ = group.bench_function(
            BenchmarkId::new("metal_wall_resident_prefix_prebound_lt", &suffix),
            |bench| {
                bench.iter(|| {
                    let mut last = sequence
                        .restart_message_timed()
                        .expect("RAM value-check prefix should restart")
                        .0;
                    for (&challenge, bound_lt_lo) in prefix_challenges.iter().zip(&bound_lt_tables)
                    {
                        last = sequence
                            .bind_and_message(challenge, bound_lt_lo)
                            .expect("RAM value-check prefix transition should execute");
                    }
                    sequence
                        .read_current_state_into(&mut warm_tail)
                        .expect("RAM value-check prefix tail should read");
                    let _tail = black_box(&warm_tail);
                    black_box(last)
                });
            },
        );
        let _ = group.bench_function(
            BenchmarkId::new("metal_active_resident_prefix", &suffix),
            |bench| {
                bench.iter_custom(|iterations| {
                    let mut total = Duration::ZERO;
                    for _ in 0..iterations {
                        let (_, first) = sequence
                            .restart_message_timed()
                            .expect("timed RAM value-check prefix should restart");
                        total += first;
                        for (&challenge, bound_lt_lo) in
                            prefix_challenges.iter().zip(&bound_lt_tables)
                        {
                            total += sequence
                                .bind_and_message_timed(challenge, bound_lt_lo)
                                .expect("timed RAM value-check transition should execute")
                                .1;
                        }
                    }
                    total
                });
            },
        );
        let _ = group.bench_function(
            BenchmarkId::new("metal_wall_hybrid_no_fs", &suffix),
            |bench| {
                bench.iter(|| {
                    let mut last = sequence
                        .restart_message_timed()
                        .expect("RAM value-check prefix should restart")
                        .0;
                    for (&challenge, bound_lt_lo) in prefix_challenges.iter().zip(&bound_lt_tables)
                    {
                        last = sequence
                            .bind_and_message(challenge, bound_lt_lo)
                            .expect("RAM value-check prefix transition should execute");
                    }
                    sequence
                        .read_current_state_into(&mut warm_tail)
                        .expect("RAM value-check prefix tail should read");
                    let dense_lt = dense_lt_table(
                        bound_lt_tables
                            .last()
                            .expect("RAM value-check prefix must bind at least once"),
                        &lt_hi,
                        &eq_hi,
                    );
                    let tail = cpu_tail(
                        warm_tail.clone(),
                        dense_lt,
                        &all_challenges[plan.gpu_bind_rounds()..],
                    );
                    black_box((last, tail))
                });
            },
        );
    }
    group.finish();
}

fn validate(context: &SolinasMetal) {
    let log_t = VALIDATION_LOG_T;
    let log_k = 5;
    let config = RamValCheckConfig {
        first_message_threads: 32,
        native_transition_threads: 32,
        dense_transition_threads: 64,
        cpu_tail_elements: 1 << 7,
    };
    let plan = RamValCheckPlan::new(log_t, log_k, config).expect("validation plan should be valid");
    let rows = rows(plan.cycles(), plan.address_domain());
    let (eq_address, mut lt_lo, lt_hi, eq_hi) = factor_tables(log_t, log_k, log_t / 2);
    let resident = context
        .prepare_ram_val_check_rows(&rows, plan.address_domain())
        .expect("validation rows should prepare");
    let mut sequence = context
        .prepare_ram_val_check_sequence(resident, &eq_address, &lt_lo, &lt_hi, &eq_hi, plan)
        .expect("validation sequence should prepare");
    assert_eq!(
        sequence.message().expect("first message should execute"),
        ram_val_check_oracle::first_message(&rows, &eq_address, &lt_lo, &lt_hi, &eq_hi)
            .expect("first-message oracle should succeed")
    );
    let mut state: Option<Vec<RamValCheckDenseRow<AkitaField>>> = None;
    for challenge in challenges(plan.gpu_bind_rounds()) {
        bind_table(&mut lt_lo, challenge);
        let expected = if let Some(state) = state.as_ref() {
            ram_val_check_oracle::dense_bind_and_message(state, challenge, &lt_lo, &lt_hi, &eq_hi)
                .expect("dense oracle should succeed")
        } else {
            ram_val_check_oracle::native_bind_and_message(
                &rows,
                &eq_address,
                challenge,
                &lt_lo,
                &lt_hi,
                &eq_hi,
            )
            .expect("native oracle should succeed")
        };
        assert_eq!(
            sequence
                .bind_and_message(challenge, &lt_lo)
                .expect("validation transition should execute"),
            expected.evals
        );
        state = Some(expected.state);
    }
    let tail = sequence
        .read_current_state()
        .expect("validation tail should read");
    assert_eq!(
        tail,
        state.expect("validation prefix should produce dense state")
    );

    let all_challenges = challenges(log_t);
    let dense_lt = dense_lt_table(&lt_lo, &lt_hi, &eq_hi);
    let (_, tail_row, tail_lt) =
        cpu_tail(tail, dense_lt, &all_challenges[plan.gpu_bind_rounds()..]);
    let mut expected_rows: Vec<_> = rows
        .iter()
        .copied()
        .map(|row| RamValCheckDenseRow {
            increment: row.increment_field(),
            ram_ra: row
                .ram_ra(&eq_address)
                .expect("validation RAM address should be in range"),
        })
        .collect();
    let (_, full_lt_lo, full_lt_hi, full_eq_hi) = factor_tables(log_t, log_k, log_t / 2);
    let mut expected_lt = dense_lt_table(&full_lt_lo, &full_lt_hi, &full_eq_hi);
    for challenge in all_challenges {
        expected_rows = bind_rows(&expected_rows, challenge);
        expected_lt = bind_fields(&expected_lt, challenge);
    }
    assert_eq!(tail_row, expected_rows[0]);
    assert_eq!(tail_lt, expected_lt[0]);
}

fn dense_lt_table(
    lt_lo: &[AkitaField],
    lt_hi: &[AkitaField],
    eq_hi: &[AkitaField],
) -> Vec<AkitaField> {
    assert_eq!(lt_hi.len(), eq_hi.len());
    (0..lt_hi.len() * lt_lo.len())
        .into_par_iter()
        .map(|index| {
            let high = index / lt_lo.len();
            let low = index % lt_lo.len();
            lt_hi[high] + eq_hi[high] * lt_lo[low]
        })
        .collect()
}

fn cpu_tail(
    mut rows: Vec<RamValCheckDenseRow<AkitaField>>,
    mut lt: Vec<AkitaField>,
    challenges: &[AkitaField],
) -> (
    Vec<[AkitaField; 3]>,
    RamValCheckDenseRow<AkitaField>,
    AkitaField,
) {
    assert_eq!(rows.len(), lt.len());
    assert_eq!(challenges.len(), rows.len().ilog2() as usize);
    let mut messages = Vec::with_capacity(challenges.len().saturating_sub(1));
    for (round, &challenge) in challenges.iter().enumerate() {
        rows = bind_rows(&rows, challenge);
        lt = bind_fields(&lt, challenge);
        if round + 1 < challenges.len() {
            messages.push(cpu_message(&rows, &lt));
        }
    }
    assert_eq!(rows.len(), 1);
    assert_eq!(lt.len(), 1);
    (messages, rows[0], lt[0])
}

fn bind_rows(
    rows: &[RamValCheckDenseRow<AkitaField>],
    challenge: AkitaField,
) -> Vec<RamValCheckDenseRow<AkitaField>> {
    (0..rows.len() / 2)
        .into_par_iter()
        .map(|index| {
            let low = rows[2 * index];
            let high = rows[2 * index + 1];
            RamValCheckDenseRow {
                increment: low.increment + challenge * (high.increment - low.increment),
                ram_ra: low.ram_ra + challenge * (high.ram_ra - low.ram_ra),
            }
        })
        .collect()
}

fn bind_fields(values: &[AkitaField], challenge: AkitaField) -> Vec<AkitaField> {
    (0..values.len() / 2)
        .into_par_iter()
        .map(|index| {
            let low = values[2 * index];
            low + challenge * (values[2 * index + 1] - low)
        })
        .collect()
}

fn cpu_message(rows: &[RamValCheckDenseRow<AkitaField>], lt: &[AkitaField]) -> [AkitaField; 3] {
    assert_eq!(rows.len(), lt.len());
    (0..rows.len() / 2)
        .into_par_iter()
        .map(|index| {
            let low = rows[2 * index];
            let high = rows[2 * index + 1];
            let increment_delta = high.increment - low.increment;
            let ra_delta = high.ram_ra - low.ram_ra;
            let lt_delta = lt[2 * index + 1] - lt[2 * index];
            let increment_2 = high.increment + increment_delta;
            let ra_2 = high.ram_ra + ra_delta;
            let lt_2 = lt[2 * index + 1] + lt_delta;
            [
                low.increment * low.ram_ra * lt[2 * index],
                increment_2 * ra_2 * lt_2,
                (increment_2 + increment_delta) * (ra_2 + ra_delta) * (lt_2 + lt_delta),
            ]
        })
        .reduce(
            || [AkitaField::default(); 3],
            |left, right| [left[0] + right[0], left[1] + right[1], left[2] + right[2]],
        )
}

fn factor_tables(
    log_t: usize,
    log_k: usize,
    low_bits: usize,
) -> (
    Vec<AkitaField>,
    Vec<AkitaField>,
    Vec<AkitaField>,
    Vec<AkitaField>,
) {
    let r_address = values(log_k, 0x243f_6a88_85a3_08d3);
    let r_cycle = values(log_t, 0x1319_8a2e_0370_7344);
    let eq_address = EqPolynomial::<AkitaField>::evals(&r_address, None);
    let (r_hi, r_lo) = r_cycle.split_at(log_t - low_bits);
    let lt_lo = LtPolynomial::<AkitaField>::evaluations(r_lo);
    let gamma = AkitaField::from_u64(0xfeed_beef_cafe_babe);
    let lt_hi = LtPolynomial::<AkitaField>::evaluations(r_hi)
        .into_iter()
        .map(|value| value + gamma)
        .collect();
    let eq_hi = EqPolynomial::<AkitaField>::evals(r_hi, None);
    (eq_address, lt_lo, lt_hi, eq_hi)
}

fn bound_lt_tables(mut lt_lo: Vec<AkitaField>, challenges: &[AkitaField]) -> Vec<Vec<AkitaField>> {
    challenges
        .iter()
        .map(|&challenge| {
            bind_table(&mut lt_lo, challenge);
            lt_lo.clone()
        })
        .collect()
}

fn bind_table(table: &mut Vec<AkitaField>, challenge: AkitaField) {
    let bound = table.len() / 2;
    for index in 0..bound {
        table[index] = table[2 * index] + challenge * (table[2 * index + 1] - table[2 * index]);
    }
    table.truncate(bound);
}

fn challenges(count: usize) -> Vec<AkitaField> {
    (0..count)
        .map(|index| AkitaField::from_u64(0x9e37_79b9_7f4a_7c15_u64.wrapping_mul(index as u64 + 1)))
        .collect()
}

fn rows(elements: usize, address_domain: usize) -> Vec<RamValCheckNativeRow> {
    (0..elements)
        .into_par_iter()
        .map(|index| {
            let selector = splitmix(index as u64 ^ 0xa409_3822_299f_31d0);
            let (address, increment) = if selector.trailing_zeros() >= 4 {
                (None, 0)
            } else {
                let magnitude = splitmix(index as u64 ^ 0x082e_fa98_ec4e_6c89) as i64;
                (
                    Some((selector as usize % address_domain) as u32),
                    i128::from(magnitude),
                )
            };
            RamValCheckNativeRow::new(address, increment)
                .expect("synthetic RAM value-check row should be valid")
        })
        .collect()
}

fn values(count: usize, seed: u64) -> Vec<AkitaField> {
    (0..count)
        .map(|index| AkitaField::from_u64(splitmix(seed ^ index as u64) & ((1u64 << 56) - 1)))
        .collect()
}

fn splitmix(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
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
            .all(|elements| elements.is_power_of_two() && *elements >= 1 << VALIDATION_LOG_T),
        "RAM value-check benchmark sizes must be powers of two at least 2^12"
    );
    cases
}
