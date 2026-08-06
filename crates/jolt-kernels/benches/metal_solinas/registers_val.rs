use std::{
    env,
    hint::black_box,
    time::{Duration, Instant},
};

use criterion::{BenchmarkId, Criterion, Throughput};
use jolt_field::{AdditiveAccumulator, AkitaAccumulator, AkitaField, RingAccumulator};
use jolt_kernels::metal::solinas::{
    RegistersValDenseConfig, RegistersValFirstMessageConfig, RegistersValTransitionConfig,
    SolinasMetal,
};
use jolt_poly::{EqPolynomial, LtPolynomial};
use rayon::prelude::*;

const DEFAULT_ELEMENTS: [usize; 3] = [1 << 16, 1 << 20, 1 << 22];

struct SplitLt {
    lo: Vec<AkitaField>,
    hi: Vec<AkitaField>,
    eq_hi: Vec<AkitaField>,
}

impl SplitLt {
    fn new(point: &[AkitaField]) -> Self {
        let mid = point.len() / 2;
        let (r_hi, r_lo) = point.split_at(point.len() - mid);
        Self {
            lo: LtPolynomial::<AkitaField>::evaluations(r_lo),
            hi: LtPolynomial::<AkitaField>::evaluations(r_hi),
            eq_hi: EqPolynomial::<AkitaField>::evals(r_hi, None),
        }
    }

    fn pair(&self, pair: usize) -> [AkitaField; 2] {
        let first = 2 * pair;
        let hi = first / self.lo.len();
        let lo = first % self.lo.len();
        let base = self.hi[hi];
        let scale = self.eq_hi[hi];
        [base + scale * self.lo[lo], base + scale * self.lo[lo + 1]]
    }

    fn bind_lo(&self, challenge: AkitaField) -> Self {
        Self {
            lo: self
                .lo
                .chunks_exact(2)
                .map(|pair| pair[0] + challenge * (pair[1] - pair[0]))
                .collect(),
            hi: self.hi.clone(),
            eq_hi: self.eq_hi.clone(),
        }
    }
}

pub fn bench(c: &mut Criterion, context: &SolinasMetal) {
    let threads_per_threadgroup =
        env::var("JOLT_METAL_REGISTERS_VAL_THREADS").map_or(32, |value| {
            value
                .parse()
                .expect("JOLT_METAL_REGISTERS_VAL_THREADS should be a positive integer")
        });
    let dense_threads_per_threadgroup =
        env::var("JOLT_METAL_REGISTERS_VAL_DENSE_THREADS").map_or(64, |value| {
            value
                .parse()
                .expect("JOLT_METAL_REGISTERS_VAL_DENSE_THREADS should be a positive integer")
        });
    let cpu_threads = std::thread::available_parallelism().map_or(1, |count| count.get());
    let mut group = c.benchmark_group("metal_sumcheck/registers_val_first_message");
    let _ = group
        .sample_size(10)
        .warm_up_time(Duration::from_secs(2))
        .measurement_time(Duration::from_secs(5));

    for elements in cases() {
        let inc = (0..elements)
            .into_par_iter()
            .map(|index| {
                AkitaField::from_u64(
                    (index as u64)
                        .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                        .rotate_left((index % 63) as u32),
                )
            })
            .collect::<Vec<_>>();
        let rd = (0..elements)
            .into_par_iter()
            .map(|index| {
                if index % 11 == 0 {
                    u8::MAX
                } else {
                    ((37 * index + 19) & 127) as u8
                }
            })
            .collect::<Vec<_>>();
        let r_address = (0..7)
            .map(|index| AkitaField::from_u64(0x101 + 17 * index as u64))
            .collect::<Vec<_>>();
        let r_cycle = (0..elements.ilog2() as usize)
            .map(|index| AkitaField::from_u64(0x1001 + 29 * index as u64))
            .collect::<Vec<_>>();
        let eq_address = EqPolynomial::<AkitaField>::evals(&r_address, None);
        let lt = SplitLt::new(&r_cycle);
        let invocation = context
            .prepare_registers_val_first_message(
                &inc,
                &rd,
                &r_address,
                &r_cycle,
                RegistersValFirstMessageConfig {
                    threads_per_threadgroup: Some(threads_per_threadgroup),
                },
            )
            .expect("registers value first message should prepare");
        let expected = cpu_message(&inc, &rd, &eq_address, &lt);
        invocation
            .execute()
            .expect("registers value first message should execute");
        assert_eq!(
            invocation
                .read_message()
                .expect("registers value first message should be readable"),
            expected
        );
        let challenge = AkitaField::from_u64(0xfeed_beef_cafe_babe);
        let bound_lt = lt.bind_lo(challenge);
        let transition_source = context
            .prepare_registers_val_first_message(
                &inc,
                &rd,
                &r_address,
                &r_cycle,
                RegistersValFirstMessageConfig {
                    threads_per_threadgroup: Some(threads_per_threadgroup),
                },
            )
            .expect("registers value transition source should prepare");
        transition_source
            .execute()
            .expect("registers value transition source should execute");
        let transition = transition_source
            .into_first_transition(
                &bound_lt.lo,
                RegistersValTransitionConfig {
                    threads_per_threadgroup: Some(threads_per_threadgroup),
                },
            )
            .expect("registers value first transition should prepare");
        let transition_expected =
            cpu_transition_message(&inc, &rd, &eq_address, challenge, &bound_lt);
        transition
            .execute(challenge)
            .expect("registers value first transition should execute");
        assert_eq!(
            transition
                .read_message()
                .expect("registers value second message should be readable"),
            transition_expected
        );

        let _ = group.throughput(Throughput::Elements(elements as u64));
        let suffix = format!(
            "n{elements}_tg{}_cpu{cpu_threads}",
            invocation.threads_per_threadgroup()
        );
        let cpu_first = env::var("JOLT_SOLINAS_BENCH_ORDER").as_deref() == Ok("cpu-first");
        let add_cpu =
            |group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>| {
                let _ = group.bench_function(BenchmarkId::new("cpu_optimized", &suffix), |bench| {
                    bench.iter(|| black_box(cpu_message(&inc, &rd, &eq_address, &lt)));
                });
            };
        let add_gpu = |group: &mut criterion::BenchmarkGroup<
            '_,
            criterion::measurement::WallTime,
        >| {
            let _ =
                group.bench_function(BenchmarkId::new("metal_wall_resident", &suffix), |bench| {
                    bench.iter(|| {
                        invocation
                            .execute()
                            .expect("registers value first message should execute");
                        black_box(
                            invocation
                                .read_message()
                                .expect("registers value first message should be readable"),
                        )
                    });
                });
            let _ = group.bench_function(
                BenchmarkId::new("metal_active_resident", &suffix),
                |bench| {
                    bench.iter_custom(|iterations| {
                        let mut active = Duration::ZERO;
                        for _ in 0..iterations {
                            active += invocation
                                .execute_timed()
                                .expect("timed registers value first message should execute");
                        }
                        active
                    });
                },
            );
        };
        if cpu_first {
            add_cpu(&mut group);
            add_gpu(&mut group);
        } else {
            add_gpu(&mut group);
            add_cpu(&mut group);
        }
        let _ = group.bench_function(
            BenchmarkId::new("cpu_message_only_first_transition", &suffix),
            |bench| {
                bench.iter(|| {
                    black_box(cpu_transition_message(
                        &inc,
                        &rd,
                        &eq_address,
                        challenge,
                        &bound_lt,
                    ))
                });
            },
        );
        let _ = group.bench_function(
            BenchmarkId::new("metal_wall_resident_first_transition", &suffix),
            |bench| {
                bench.iter(|| {
                    transition
                        .execute(challenge)
                        .expect("registers value first transition should execute");
                    black_box(
                        transition
                            .read_message()
                            .expect("registers value second message should be readable"),
                    )
                });
            },
        );
        let _ = group.bench_function(
            BenchmarkId::new("metal_active_resident_first_transition", &suffix),
            |bench| {
                bench.iter_custom(|iterations| {
                    let mut active = Duration::ZERO;
                    for _ in 0..iterations {
                        active += transition
                            .execute_timed(challenge)
                            .expect("timed registers value first transition should execute");
                    }
                    active
                });
            },
        );

        let cutoff = env::var("JOLT_METAL_REGISTERS_VAL_CUTOFF").map_or(1 << 16, |value| {
            value
                .parse()
                .expect("JOLT_METAL_REGISTERS_VAL_CUTOFF should be a positive integer")
        });
        let mut sequence = transition
            .into_sequence(RegistersValDenseConfig {
                threads_per_threadgroup: Some(dense_threads_per_threadgroup),
            })
            .expect("registers value dense sequence should prepare");
        let mut current_lt = bound_lt;
        let mut dense_round = 0_u64;
        let mut dense_ladder_wall = Duration::ZERO;
        while sequence.current_elements() > cutoff && current_lt.lo.len() >= 4 {
            let dense_challenge =
                AkitaField::from_u64(0x9e37_79b9_7f4a_7c15_u64.wrapping_mul(dense_round + 1));
            let host_start = Instant::now();
            let next_lt = current_lt.bind_lo(dense_challenge);
            dense_ladder_wall += host_start.elapsed();
            let current_elements = sequence.current_elements();
            let _ = group.throughput(Throughput::Elements(current_elements as u64));
            let dense_suffix = format!(
                "r{dense_round}_n{current_elements}_tg{}_cpu{cpu_threads}",
                sequence.threads_per_threadgroup()
            );
            let replay = sequence
                .replay_current_bind_and_message_timed(dense_challenge, &next_lt.lo)
                .expect("registers value dense transition should replay")
                .0;
            let _ = group.bench_function(
                BenchmarkId::new("metal_wall_resident_dense_transition", &dense_suffix),
                |bench| {
                    bench.iter(|| {
                        black_box(
                            sequence
                                .replay_current_bind_and_message_timed(dense_challenge, &next_lt.lo)
                                .expect("registers value dense transition should replay")
                                .0,
                        )
                    });
                },
            );
            let _ = group.bench_function(
                BenchmarkId::new("metal_active_resident_dense_transition", &dense_suffix),
                |bench| {
                    bench.iter_custom(|iterations| {
                        let mut active = Duration::ZERO;
                        for _ in 0..iterations {
                            active += sequence
                                .replay_current_bind_and_message_timed(dense_challenge, &next_lt.lo)
                                .expect("timed registers value dense transition should replay")
                                .1;
                        }
                        active
                    });
                },
            );
            let round_start = Instant::now();
            let advanced = sequence
                .bind_and_message(dense_challenge, &next_lt.lo)
                .expect("registers value dense transition should advance");
            dense_ladder_wall += round_start.elapsed();
            assert_eq!(advanced, replay);
            current_lt = next_lt;
            dense_round += 1;
        }
        eprintln!(
            "registers-val dense ladder n={elements} cutoff={} rounds={dense_round} active={:?} wall={dense_ladder_wall:?}",
            sequence.current_elements(),
            sequence.gpu_active_time(),
        );
    }
    group.finish();
}

fn cpu_message(
    inc: &[AkitaField],
    rd: &[u8],
    eq_address: &[AkitaField],
    lt: &SplitLt,
) -> [AkitaField; 3] {
    let accumulate = |pair: usize, sums: &mut [AkitaAccumulator; 3]| {
        let first = 2 * pair;
        let inc_pair = [inc[first], inc[first + 1]];
        let wa_pair = [rd[first], rd[first + 1]].map(|index| {
            if index == u8::MAX {
                AkitaField::from_u64(0)
            } else {
                eq_address[index as usize]
            }
        });
        let lt_pair = lt.pair(pair);
        let deltas = [
            inc_pair[1] - inc_pair[0],
            wa_pair[1] - wa_pair[0],
            lt_pair[1] - lt_pair[0],
        ];
        let at_2 = [
            inc_pair[1] + deltas[0],
            wa_pair[1] + deltas[1],
            lt_pair[1] + deltas[2],
        ];
        sums[0].fmadd(inc_pair[0] * wa_pair[0], lt_pair[0]);
        sums[1].fmadd(at_2[0] * at_2[1], at_2[2]);
        sums[2].fmadd(
            (at_2[0] + deltas[0]) * (at_2[1] + deltas[1]),
            at_2[2] + deltas[2],
        );
    };
    (0..inc.len() / 2)
        .into_par_iter()
        .fold(
            || [AkitaAccumulator::default(); 3],
            |mut sums, pair| {
                accumulate(pair, &mut sums);
                sums
            },
        )
        .map(|sums| sums.map(AkitaAccumulator::reduce))
        .reduce(
            || [AkitaField::from_u64(0); 3],
            |left, right| [left[0] + right[0], left[1] + right[1], left[2] + right[2]],
        )
}

fn cpu_transition_message(
    inc: &[AkitaField],
    rd: &[u8],
    eq_address: &[AkitaField],
    challenge: AkitaField,
    bound_lt: &SplitLt,
) -> [AkitaField; 3] {
    let accumulate = |pair: usize, sums: &mut [AkitaAccumulator; 3]| {
        let source = 4 * pair;
        let bind = |low: AkitaField, high: AkitaField| low + challenge * (high - low);
        let wa = [0, 1, 2, 3].map(|offset| {
            let register = rd[source + offset];
            if register == u8::MAX {
                AkitaField::from_u64(0)
            } else {
                eq_address[register as usize]
            }
        });
        let inc_pair = [
            bind(inc[source], inc[source + 1]),
            bind(inc[source + 2], inc[source + 3]),
        ];
        let wa_pair = [bind(wa[0], wa[1]), bind(wa[2], wa[3])];
        let lt_pair = bound_lt.pair(pair);
        let deltas = [
            inc_pair[1] - inc_pair[0],
            wa_pair[1] - wa_pair[0],
            lt_pair[1] - lt_pair[0],
        ];
        let at_2 = [
            inc_pair[1] + deltas[0],
            wa_pair[1] + deltas[1],
            lt_pair[1] + deltas[2],
        ];
        sums[0].fmadd(inc_pair[0] * wa_pair[0], lt_pair[0]);
        sums[1].fmadd(at_2[0] * at_2[1], at_2[2]);
        sums[2].fmadd(
            (at_2[0] + deltas[0]) * (at_2[1] + deltas[1]),
            at_2[2] + deltas[2],
        );
    };
    (0..inc.len() / 4)
        .into_par_iter()
        .fold(
            || [AkitaAccumulator::default(); 3],
            |mut sums, pair| {
                accumulate(pair, &mut sums);
                sums
            },
        )
        .map(|sums| sums.map(AkitaAccumulator::reduce))
        .reduce(
            || [AkitaField::from_u64(0); 3],
            |left, right| [left[0] + right[0], left[1] + right[1], left[2] + right[2]],
        )
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
            .all(|elements| elements.is_power_of_two() && *elements >= 1 << 2),
        "registers value benchmark sizes must be powers of two at least 2^2"
    );
    cases
}
