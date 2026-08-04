use std::{env, hint::black_box, time::Duration};

use criterion::{BenchmarkId, Criterion, Throughput};
use jolt_field::{AdditiveAccumulator, AkitaAccumulator, AkitaField, RingAccumulator};
use jolt_kernels::metal::solinas::{InstructionRaFirstMessageConfig, SolinasMetal};
use jolt_poly::{BindingOrder, GruenSplitEqPolynomial};
use rayon::prelude::*;

const DEFAULT_ELEMENTS: [usize; 3] = [1 << 16, 1 << 20, 1 << 22];
const FACTORS: usize = 16;
const GROUPS: usize = 4;
const FACTORS_PER_GROUP: usize = 4;
const BINS: usize = 256;

pub fn bench(c: &mut Criterion, context: &SolinasMetal) {
    let threads_per_threadgroup =
        env::var("JOLT_METAL_INSTRUCTION_RA_THREADS").map_or(128, |value| {
            value
                .parse()
                .expect("JOLT_METAL_INSTRUCTION_RA_THREADS should be a positive integer")
        });
    let cpu_threads = std::thread::available_parallelism().map_or(1, |count| count.get());
    let gamma = AkitaField::from_u64(7);
    let tables = chunk_tables(gamma);
    let mut group = c.benchmark_group("metal_sumcheck/instruction_ra_first_message");
    let _ = group
        .sample_size(10)
        .warm_up_time(Duration::from_secs(2))
        .measurement_time(Duration::from_secs(5));

    for elements in cases() {
        let log_n = elements.ilog2() as usize;
        let (cycle_order, table_major, inverse) = lookup_layout(elements, log_n, 1);
        let point = (0..log_n)
            .map(|round| AkitaField::from_u64((1009 + 37 * round) as u64))
            .collect::<Vec<_>>();
        let gruen = GruenSplitEqPolynomial::new(&point, BindingOrder::LowToHigh);
        let invocation = context
            .prepare_instruction_ra_first_message(
                &table_major,
                &inverse,
                &tables,
                gruen.e_in_current(),
                gruen.e_out_current(),
                InstructionRaFirstMessageConfig {
                    threads_per_threadgroup: Some(threads_per_threadgroup),
                },
            )
            .expect("Instruction RA first message should prepare");
        let expected = cpu_message(
            &cycle_order,
            &tables,
            gruen.e_in_current(),
            gruen.e_out_current(),
        );
        invocation
            .execute()
            .expect("Instruction RA first message should execute");
        assert_eq!(
            invocation
                .read_message()
                .expect("Instruction RA first message should be readable"),
            expected
        );
        assert_eq!(invocation.execute_device_buffer_allocations(), 0);

        let _ = group.throughput(Throughput::Elements(invocation.useful_multiplications()));
        let suffix = format!(
            "n{elements}_tg{}_cpu{cpu_threads}",
            invocation.threads_per_threadgroup()
        );
        let cpu_first = env::var("JOLT_SOLINAS_BENCH_ORDER").as_deref() == Ok("cpu-first");
        let add_cpu =
            |group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>| {
                let _ = group.bench_function(BenchmarkId::new("cpu_optimized", &suffix), |bench| {
                    bench.iter(|| {
                        black_box(cpu_message(
                            &cycle_order,
                            &tables,
                            gruen.e_in_current(),
                            gruen.e_out_current(),
                        ))
                    });
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
                            .expect("Instruction RA first message should execute");
                        black_box(
                            invocation
                                .read_message()
                                .expect("Instruction RA first message should be readable"),
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
                                .expect("timed Instruction RA first message should execute");
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
        "Instruction RA benchmark sizes must be powers of two at least 2^6"
    );
    cases
}

fn lookup_layout(rows: usize, log_n: usize, seed: u64) -> (Vec<u128>, Vec<u128>, Vec<u32>) {
    let cycle_order = (0..rows)
        .into_par_iter()
        .map(|cycle| lookup(cycle, seed))
        .collect::<Vec<_>>();
    let table_major = (0..rows)
        .into_par_iter()
        .map(|slot| lookup(permute(slot, log_n), seed))
        .collect::<Vec<_>>();
    let inverse = (0..rows)
        .into_par_iter()
        .map(|cycle| permute(cycle, log_n) as u32)
        .collect::<Vec<_>>();
    (cycle_order, table_major, inverse)
}

fn lookup(cycle: usize, seed: u64) -> u128 {
    let counter = seed.wrapping_add(2 * cycle as u64);
    u128::from(splitmix(counter)) | (u128::from(splitmix(counter + 1)) << 64)
}

fn permute(index: usize, log_n: usize) -> usize {
    index.reverse_bits() >> (usize::BITS as usize - log_n)
}

fn chunk_tables(gamma: AkitaField) -> Vec<AkitaField> {
    let mut gamma_power = AkitaField::from_u64(1);
    let mut tables = Vec::with_capacity(FACTORS * BINS);
    for factor in 0..FACTORS {
        for bin in 0..BINS {
            let mut value = AkitaField::from_u64((2 + 17 * factor + 31 * bin) as u64);
            if factor.is_multiple_of(FACTORS_PER_GROUP) {
                value *= gamma_power;
            }
            tables.push(value);
        }
        if (factor + 1).is_multiple_of(FACTORS_PER_GROUP) {
            gamma_power *= gamma;
        }
    }
    tables
}

fn cpu_message(
    lookups: &[u128],
    tables: &[AkitaField],
    e_in: &[AkitaField],
    e_out: &[AkitaField],
) -> [AkitaField; 4] {
    assert_eq!(lookups.len() / 2, e_in.len() * e_out.len());
    (0..e_out.len())
        .into_par_iter()
        .map(|x_out| {
            let mut lanes = [AkitaAccumulator::default(); 4];
            for (x_in, &inner_weight) in e_in.iter().enumerate() {
                let pair = x_out * e_in.len() + x_in;
                let mut row = [AkitaAccumulator::default(); 4];
                for group in 0..GROUPS {
                    let mut factors = [(AkitaField::from_u64(0), AkitaField::from_u64(0)); 4];
                    for (offset, pair_values) in factors.iter_mut().enumerate() {
                        let factor = group * FACTORS_PER_GROUP + offset;
                        let shift = 8 * (FACTORS - 1 - factor);
                        let index = |cycle: usize| ((lookups[cycle] >> shift) & 0xff) as usize;
                        *pair_values = (
                            tables[factor * BINS + index(2 * pair)],
                            tables[factor * BINS + index(2 * pair + 1)],
                        );
                    }
                    let left = quadratic_grid(factors[0], factors[1]);
                    let right = quadratic_grid(factors[2], factors[3]);
                    for ((lane, left), right) in row.iter_mut().zip(left).zip(right) {
                        lane.fmadd(left, right);
                    }
                }
                for (lane, row) in lanes.iter_mut().zip(row) {
                    lane.fmadd(inner_weight, row.reduce());
                }
            }
            let mut output = [AkitaField::from_u64(0); 4];
            for (output, lane) in output.iter_mut().zip(lanes) {
                *output = e_out[x_out] * lane.reduce();
            }
            output
        })
        .reduce(
            || [AkitaField::from_u64(0); 4],
            |mut lhs, rhs| {
                for (lhs, rhs) in lhs.iter_mut().zip(rhs) {
                    *lhs += rhs;
                }
                lhs
            },
        )
}

fn quadratic_grid(
    first: (AkitaField, AkitaField),
    second: (AkitaField, AkitaField),
) -> [AkitaField; 4] {
    let at_zero = first.0 * second.0;
    let at_one = first.1 * second.1;
    let at_infinity = (first.1 - first.0) * (second.1 - second.0);
    let twice_at_infinity = at_infinity + at_infinity;
    let at_two = at_one + at_one - at_zero + twice_at_infinity;
    let at_three = at_two + at_one - at_zero + twice_at_infinity + twice_at_infinity;
    [at_one, at_two, at_three, at_infinity]
}

fn splitmix(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}
