use std::{env, hint::black_box, time::Duration};

use criterion::{BenchmarkId, Criterion, Throughput};
use jolt_field::{AdditiveAccumulator, AkitaAccumulator, AkitaField, RingAccumulator};
use jolt_kernels::metal::solinas::{
    RegisterAccessRow, RegistersReadWriteMessageConfig, SolinasMetal,
};
use jolt_poly::{BindingOrder, GruenSplitEqPolynomial};
use rayon::prelude::*;

const DEFAULT_ELEMENTS: [usize; 3] = [1 << 16, 1 << 20, 1 << 22];

#[derive(Clone, Copy)]
struct Cell {
    value: AkitaField,
    prev_value: u64,
    next_value: u64,
    ra: AkitaField,
    read: bool,
    write: bool,
}

#[derive(Clone, Copy)]
struct BoundCell {
    value: AkitaField,
    prev_value: u64,
    next_value: u64,
    ra: AkitaField,
    wa: AkitaField,
}

pub fn bench(c: &mut Criterion, context: &SolinasMetal) {
    let threads_per_threadgroup =
        env::var("JOLT_METAL_REGISTERS_RW_THREADS").map_or(128, |value| {
            value
                .parse()
                .expect("JOLT_METAL_REGISTERS_RW_THREADS should be a positive integer")
        });
    let cpu_threads = std::thread::available_parallelism().map_or(1, |count| count.get());
    let gamma = AkitaField::from_u64(0x1234_5678_9abc_def0);
    let mut group = c.benchmark_group("metal_sumcheck/registers_read_write_first_message");
    let _ = group
        .sample_size(10)
        .warm_up_time(Duration::from_secs(2))
        .measurement_time(Duration::from_secs(5));

    for elements in cases() {
        let rows = rows(elements);
        let inc = (0..elements)
            .into_par_iter()
            .map(|row| AkitaField::from_u64((17 * row + 3) as u64))
            .collect::<Vec<_>>();
        let point = (0..elements.ilog2() as usize)
            .map(|round| AkitaField::from_u64((1009 + 37 * round) as u64))
            .collect::<Vec<_>>();
        let mut gruen = GruenSplitEqPolynomial::new(&point, BindingOrder::LowToHigh);
        let first_e_in = gruen.e_in_current().to_vec();
        let first_e_out = gruen.e_out_current().to_vec();
        let invocation = context
            .prepare_registers_read_write_first_message(
                &rows,
                &inc,
                gamma,
                &first_e_in,
                &first_e_out,
                RegistersReadWriteMessageConfig {
                    threads_per_threadgroup: Some(threads_per_threadgroup),
                },
            )
            .expect("registers read/write first message should prepare");
        let expected = cpu_message(&rows, &inc, gamma, &first_e_in, &first_e_out);
        invocation
            .execute()
            .expect("registers read/write first message should execute");
        assert_eq!(
            invocation
                .read_message()
                .expect("registers read/write first message should be readable"),
            expected
        );
        let first_challenge = AkitaField::from_u64(0xfeed_beef_cafe_babe);
        gruen.bind(first_challenge);
        let second = invocation
            .prepare_second_message(
                gruen.e_in_current(),
                gruen.e_out_current(),
                RegistersReadWriteMessageConfig {
                    threads_per_threadgroup: Some(threads_per_threadgroup),
                },
            )
            .expect("second registers read/write message should prepare");
        let second_expected = cpu_second_message(
            &rows,
            &inc,
            gamma,
            first_challenge,
            gruen.e_in_current(),
            gruen.e_out_current(),
        );
        second
            .execute(first_challenge)
            .expect("second registers read/write message should execute");
        assert_eq!(
            second
                .read_message()
                .expect("second registers read/write message should be readable"),
            second_expected
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
                    bench.iter(|| {
                        black_box(cpu_message(&rows, &inc, gamma, &first_e_in, &first_e_out))
                    });
                });
            };
        let add_gpu =
            |group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>| {
                let _ =
                    group.bench_function(
                        BenchmarkId::new("metal_wall_resident", &suffix),
                        |bench| {
                            bench.iter(|| {
                                invocation
                                    .execute()
                                    .expect("registers read/write first message should execute");
                                black_box(invocation.read_message().expect(
                                    "registers read/write first message should be readable",
                                ))
                            });
                        },
                    );
                let _ = group.bench_function(
                    BenchmarkId::new("metal_active_resident", &suffix),
                    |bench| {
                        bench.iter_custom(|iterations| {
                            let mut active = Duration::ZERO;
                            for _ in 0..iterations {
                                active += invocation
                                    .execute_timed()
                                    .expect("timed registers read/write message should execute");
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
            BenchmarkId::new("cpu_optimized_second_message", &suffix),
            |bench| {
                bench.iter(|| {
                    black_box(cpu_second_message(
                        &rows,
                        &inc,
                        gamma,
                        first_challenge,
                        gruen.e_in_current(),
                        gruen.e_out_current(),
                    ))
                });
            },
        );
        let _ = group.bench_function(
            BenchmarkId::new("metal_wall_resident_second_message", &suffix),
            |bench| {
                bench.iter(|| {
                    second
                        .execute(first_challenge)
                        .expect("second registers read/write message should execute");
                    black_box(
                        second
                            .read_message()
                            .expect("second registers read/write message should be readable"),
                    )
                });
            },
        );
        let _ = group.bench_function(
            BenchmarkId::new("metal_active_resident_second_message", &suffix),
            |bench| {
                bench.iter_custom(|iterations| {
                    let mut active = Duration::ZERO;
                    for _ in 0..iterations {
                        active += second
                            .execute_timed(first_challenge)
                            .expect("timed second registers read/write message should execute");
                    }
                    active
                });
            },
        );
    }
    group.finish();
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
            .all(|elements| elements.is_power_of_two() && *elements >= 1 << 6),
        "registers read/write benchmark sizes must be powers of two at least 2^6"
    );
    cases
}

fn rows(elements: usize) -> Vec<RegisterAccessRow> {
    let mut state = [0u64; 128];
    (0..elements)
        .map(|cycle| {
            let rs1_index = ((13 * cycle) & 127) as u8;
            let rs2_index = if cycle % 7 == 0 {
                rs1_index
            } else {
                ((29 * cycle + 5) & 127) as u8
            };
            let rd_index = if cycle % 5 == 0 {
                rs2_index
            } else {
                ((47 * cycle + 11) & 127) as u8
            };
            let rs1 = (cycle % 9 != 0).then_some((rs1_index, state[rs1_index as usize]));
            let rs2 = (cycle % 6 != 0).then_some((rs2_index, state[rs2_index as usize]));
            let rd = if cycle % 4 == 0 {
                None
            } else {
                let pre = state[rd_index as usize];
                let post = pre.wrapping_add((cycle as u64) | 1);
                state[rd_index as usize] = post;
                Some((rd_index, pre, post))
            };
            RegisterAccessRow::new(rs1, rs2, rd)
        })
        .collect()
}

fn cell(row: RegisterAccessRow, column: u8, gamma: AkitaField) -> Option<Cell> {
    let rs1 = row.rs1().filter(|(index, _)| *index == column);
    let rs2 = row.rs2().filter(|(index, _)| *index == column);
    let rd = row.rd().filter(|(index, ..)| *index == column);
    if rs1.is_none() && rs2.is_none() && rd.is_none() {
        return None;
    }
    let value = rs1
        .map(|(_, value)| value)
        .or_else(|| rs2.map(|(_, value)| value))
        .unwrap_or_else(|| rd.expect("present cell has an access").1);
    let mut ra = AkitaField::from_u64(0);
    if rs1.is_some() {
        ra += gamma;
    }
    if rs2.is_some() {
        ra += gamma * gamma;
    }
    Some(Cell {
        value: AkitaField::from_u64(value),
        prev_value: value,
        next_value: rd.map_or(value, |(_, _, post)| post),
        ra,
        read: rs1.is_some() || rs2.is_some(),
        write: rd.is_some(),
    })
}

fn bind(lo: AkitaField, hi: AkitaField, challenge: AkitaField) -> AkitaField {
    lo + challenge * (hi - lo)
}

fn bound_cell(
    lo: RegisterAccessRow,
    hi: RegisterAccessRow,
    column: u8,
    challenge: AkitaField,
    gamma: AkitaField,
) -> Option<BoundCell> {
    let even = cell(lo, column, gamma);
    let odd = cell(hi, column, gamma);
    if even.is_none() && odd.is_none() {
        return None;
    }
    let even_value = even.map_or_else(
        || AkitaField::from_u64(odd.expect("odd cell is present").prev_value),
        |cell| cell.value,
    );
    let odd_value = odd.map_or_else(
        || AkitaField::from_u64(even.expect("even cell is present").next_value),
        |cell| cell.value,
    );
    let zero = AkitaField::from_u64(0);
    Some(BoundCell {
        value: bind(even_value, odd_value, challenge),
        prev_value: even.map_or_else(
            || odd.expect("odd cell is present").prev_value,
            |cell| cell.prev_value,
        ),
        next_value: odd.map_or_else(
            || even.expect("even cell is present").next_value,
            |cell| cell.next_value,
        ),
        ra: bind(
            even.map_or(zero, |cell| cell.ra),
            odd.map_or(zero, |cell| cell.ra),
            challenge,
        ),
        wa: bind(
            even.map_or(zero, |cell| AkitaField::from_u64(u64::from(cell.write))),
            odd.map_or(zero, |cell| AkitaField::from_u64(u64::from(cell.write))),
            challenge,
        ),
    })
}

fn second_pair_endpoints(
    rows: [RegisterAccessRow; 4],
    inc_zero: AkitaField,
    inc_infinity: AkitaField,
    challenge: AkitaField,
    gamma: AkitaField,
) -> [AkitaField; 2] {
    let candidates = [
        rows[0].rs1().map(|access| access.0),
        rows[0].rs2().map(|access| access.0),
        rows[0].rd().map(|access| access.0),
        rows[1].rs1().map(|access| access.0),
        rows[1].rs2().map(|access| access.0),
        rows[1].rd().map(|access| access.0),
        rows[2].rs1().map(|access| access.0),
        rows[2].rs2().map(|access| access.0),
        rows[2].rd().map(|access| access.0),
        rows[3].rs1().map(|access| access.0),
        rows[3].rs2().map(|access| access.0),
        rows[3].rd().map(|access| access.0),
    ];
    let mut q = [AkitaAccumulator::default(); 2];
    for (slot, candidate) in candidates.iter().enumerate() {
        let Some(column) = candidate else {
            continue;
        };
        if candidates[..slot].contains(candidate) {
            continue;
        }
        let even = bound_cell(rows[0], rows[1], *column, challenge, gamma);
        let odd = bound_cell(rows[2], rows[3], *column, challenge, gamma);
        if let Some(even) = even {
            q[0].fmadd(even.ra, even.value);
            q[0].fmadd(even.wa, even.value + inc_zero);
        }
        let value_infinity = match (even, odd) {
            (Some(even), Some(odd)) => odd.value - even.value,
            (Some(even), None) => AkitaField::from_u64(even.next_value) - even.value,
            (None, Some(odd)) => odd.value - AkitaField::from_u64(odd.prev_value),
            (None, None) => unreachable!("candidate came from one of the rows"),
        };
        let zero = AkitaField::from_u64(0);
        q[1].fmadd(
            odd.map_or(zero, |cell| cell.ra) - even.map_or(zero, |cell| cell.ra),
            value_infinity,
        );
        q[1].fmadd(
            odd.map_or(zero, |cell| cell.wa) - even.map_or(zero, |cell| cell.wa),
            value_infinity + inc_infinity,
        );
    }
    q.map(AdditiveAccumulator::reduce)
}

fn pair_endpoints(
    lo: RegisterAccessRow,
    hi: RegisterAccessRow,
    inc_zero: AkitaField,
    inc_infinity: AkitaField,
    gamma: AkitaField,
) -> [AkitaField; 2] {
    let candidates = [
        lo.rs1().map(|access| access.0),
        lo.rs2().map(|access| access.0),
        lo.rd().map(|access| access.0),
        hi.rs1().map(|access| access.0),
        hi.rs2().map(|access| access.0),
        hi.rd().map(|access| access.0),
    ];
    let mut q = [AkitaAccumulator::default(); 2];
    for (slot, candidate) in candidates.iter().enumerate() {
        let Some(column) = candidate else {
            continue;
        };
        if candidates[..slot].contains(candidate) {
            continue;
        }
        let even = cell(lo, *column, gamma);
        let odd = cell(hi, *column, gamma);
        if let Some(even) = even {
            if even.read {
                q[0].fmadd(even.ra, even.value);
            }
            if even.write {
                q[0].add(even.value + inc_zero);
            }
        }
        let value_infinity = match (even, odd) {
            (Some(even), Some(odd)) => odd.value - even.value,
            (Some(even), None) => AkitaField::from_u64(even.next_value) - even.value,
            (None, Some(_)) => AkitaField::from_u64(0),
            (None, None) => unreachable!("candidate came from one of the rows"),
        };
        let even_ra = even.map_or(AkitaField::from_u64(0), |cell| cell.ra);
        let odd_ra = odd.map_or(AkitaField::from_u64(0), |cell| cell.ra);
        if even.is_some_and(|cell| cell.read) || odd.is_some_and(|cell| cell.read) {
            q[1].fmadd(odd_ra - even_ra, value_infinity);
        }
        let even_write = even.is_some_and(|cell| cell.write);
        let odd_write = odd.is_some_and(|cell| cell.write);
        let write_term = value_infinity + inc_infinity;
        if odd_write && !even_write {
            q[1].add(write_term);
        } else if even_write && !odd_write {
            q[1].add(-write_term);
        }
    }
    q.map(AdditiveAccumulator::reduce)
}

fn cpu_message(
    rows: &[RegisterAccessRow],
    inc: &[AkitaField],
    gamma: AkitaField,
    e_in: &[AkitaField],
    e_out: &[AkitaField],
) -> [AkitaField; 2] {
    assert_eq!(rows.len(), inc.len());
    assert_eq!(rows.len() / 2, e_in.len() * e_out.len());
    (0..e_out.len())
        .into_par_iter()
        .map(|x_out| {
            let mut lanes = [AkitaAccumulator::default(); 2];
            for (x_in, &inner_weight) in e_in.iter().enumerate() {
                let pair = x_out * e_in.len() + x_in;
                let q = pair_endpoints(
                    rows[2 * pair],
                    rows[2 * pair + 1],
                    inc[2 * pair],
                    inc[2 * pair + 1] - inc[2 * pair],
                    gamma,
                );
                lanes[0].fmadd(inner_weight, q[0]);
                lanes[1].fmadd(inner_weight, q[1]);
            }
            lanes.map(|lane| e_out[x_out] * lane.reduce())
        })
        .reduce(
            || [AkitaField::from_u64(0); 2],
            |lhs, rhs| [lhs[0] + rhs[0], lhs[1] + rhs[1]],
        )
}

fn cpu_second_message(
    rows: &[RegisterAccessRow],
    inc: &[AkitaField],
    gamma: AkitaField,
    challenge: AkitaField,
    e_in: &[AkitaField],
    e_out: &[AkitaField],
) -> [AkitaField; 2] {
    assert_eq!(rows.len(), inc.len());
    assert_eq!(rows.len() / 4, e_in.len() * e_out.len());
    (0..e_out.len())
        .into_par_iter()
        .map(|x_out| {
            let mut lanes = [AkitaAccumulator::default(); 2];
            for (x_in, &inner_weight) in e_in.iter().enumerate() {
                let pair = x_out * e_in.len() + x_in;
                let base = 4 * pair;
                let inc_zero = bind(inc[base], inc[base + 1], challenge);
                let inc_one = bind(inc[base + 2], inc[base + 3], challenge);
                let q = second_pair_endpoints(
                    [rows[base], rows[base + 1], rows[base + 2], rows[base + 3]],
                    inc_zero,
                    inc_one - inc_zero,
                    challenge,
                    gamma,
                );
                lanes[0].fmadd(inner_weight, q[0]);
                lanes[1].fmadd(inner_weight, q[1]);
            }
            lanes.map(|lane| e_out[x_out] * lane.reduce())
        })
        .reduce(
            || [AkitaField::from_u64(0); 2],
            |lhs, rhs| [lhs[0] + rhs[0], lhs[1] + rhs[1]],
        )
}
