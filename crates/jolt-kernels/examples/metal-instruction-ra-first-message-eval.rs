#![expect(
    clippy::print_stdout,
    reason = "the evaluator emits one machine-readable result"
)]

use std::env;
use std::error::Error;
use std::hint::black_box;
use std::time::{Duration, Instant};

use jolt_field::{AdditiveAccumulator, AkitaAccumulator, AkitaField, RingAccumulator};
use jolt_kernels::metal::solinas::{InstructionRaFirstMessageConfig, SolinasMetal};
use jolt_poly::{BindingOrder, GruenSplitEqPolynomial};
use rayon::prelude::*;
use serde_json::json;

type EvalResult<T> = Result<T, Box<dyn Error>>;

const FACTORS: usize = 16;
const BINS: usize = 256;
const GROUPS: usize = 4;
const FACTORS_PER_GROUP: usize = 4;

fn splitmix(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn lookup(cycle: usize, seed: u64) -> u128 {
    match cycle {
        0 => 0x0001_0203_0405_0607_0809_0a0b_0c0d_0e0f,
        1 => 0xf0e1_d2c3_b4a5_9687_7869_5a4b_3c2d_1e0f,
        2 => 0xff00_aa55_cc33_9966_1234_5678_9abc_def0,
        _ => {
            let counter = seed.wrapping_add(2 * cycle as u64);
            u128::from(splitmix(counter)) | (u128::from(splitmix(counter + 1)) << 64)
        }
    }
}

fn permute(index: usize, log_n: usize) -> usize {
    index.reverse_bits() >> (usize::BITS as usize - log_n)
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

fn env_usize(name: &str, default: usize) -> EvalResult<usize> {
    match env::var(name) {
        Ok(value) => Ok(value.parse()?),
        Err(env::VarError::NotPresent) => Ok(default),
        Err(error) => Err(error.into()),
    }
}

fn median(values: &mut [Duration]) -> Duration {
    values.sort_unstable();
    values[values.len() / 2]
}

fn main() -> EvalResult<()> {
    let log_n = env_usize("JOLT_METAL_EVAL_LOG_N", 22)?;
    let validation_log_n = env_usize("JOLT_METAL_EVAL_VALIDATE_LOG_N", 12)?;
    let repeats = env_usize("JOLT_METAL_EVAL_REPEATS", 3)?;
    let seed = env_usize("JOLT_METAL_EVAL_SEED", 1)? as u64;
    let threads = env_usize("JOLT_METAL_INSTRUCTION_RA_THREADS", 128)?;
    if !(6..=28).contains(&log_n)
        || !(6..=20).contains(&validation_log_n)
        || repeats < 3
        || repeats.is_multiple_of(2)
    {
        return Err("log sizes or repeat count are outside the evaluator domain".into());
    }

    let context = SolinasMetal::for_akita()?;
    let gamma = AkitaField::from_u64(7);
    let tables = chunk_tables(gamma);

    let validation_rows = 1usize << validation_log_n;
    let (validation_cycle, validation_table_major, validation_inverse) =
        lookup_layout(validation_rows, validation_log_n, seed);
    let validation_point = (0..validation_log_n)
        .map(|round| AkitaField::from_u64((1009 + 37 * round) as u64))
        .collect::<Vec<_>>();
    let validation_gruen = GruenSplitEqPolynomial::new(&validation_point, BindingOrder::LowToHigh);
    let validation_expected = cpu_message(
        &validation_cycle,
        &tables,
        validation_gruen.e_in_current(),
        validation_gruen.e_out_current(),
    );
    let validation = context.prepare_instruction_ra_first_message(
        &validation_table_major,
        &validation_inverse,
        &tables,
        validation_gruen.e_in_current(),
        validation_gruen.e_out_current(),
        InstructionRaFirstMessageConfig {
            threads_per_threadgroup: Some(threads),
        },
    )?;
    validation.execute()?;
    let exact_q_evals = validation.read_message()? == validation_expected;
    if !exact_q_evals {
        return Err("Instruction RA first message differs from the CPU oracle".into());
    }

    let rows = 1usize << log_n;
    let preparation_started = Instant::now();
    let (cycle_order, table_major, inverse) = lookup_layout(rows, log_n, seed);
    let point = (0..log_n)
        .map(|round| AkitaField::from_u64((1009 + 37 * round) as u64))
        .collect::<Vec<_>>();
    let gruen = GruenSplitEqPolynomial::new(&point, BindingOrder::LowToHigh);
    let invocation = context.prepare_instruction_ra_first_message(
        &table_major,
        &inverse,
        &tables,
        gruen.e_in_current(),
        gruen.e_out_current(),
        InstructionRaFirstMessageConfig {
            threads_per_threadgroup: Some(threads),
        },
    )?;
    let preparation = preparation_started.elapsed();
    invocation.execute()?;

    let mut cpu_times = Vec::with_capacity(repeats);
    let mut gpu_wall_times = Vec::with_capacity(repeats);
    let mut gpu_active_times = Vec::with_capacity(repeats);
    for repeat in 0..repeats {
        if repeat.is_multiple_of(2) {
            let started = Instant::now();
            let _ = black_box(cpu_message(
                &cycle_order,
                &tables,
                gruen.e_in_current(),
                gruen.e_out_current(),
            ));
            cpu_times.push(started.elapsed());
        }
        let started = Instant::now();
        let active = invocation.execute_timed()?;
        gpu_wall_times.push(started.elapsed());
        gpu_active_times.push(active);
        if !repeat.is_multiple_of(2) {
            let started = Instant::now();
            let _ = black_box(cpu_message(
                &cycle_order,
                &tables,
                gruen.e_in_current(),
                gruen.e_out_current(),
            ));
            cpu_times.push(started.elapsed());
        }
    }

    let cpu_median = median(&mut cpu_times);
    let gpu_wall_median = median(&mut gpu_wall_times);
    let gpu_active_median = median(&mut gpu_active_times);
    let useful_multiplications = invocation.useful_multiplications();
    let output = json!({
        "schema_version": 1,
        "kernel": "instruction_ra_first_message",
        "metrics": {
            "resident_speedup": cpu_median.as_secs_f64() / gpu_wall_median.as_secs_f64(),
            "gpu_active_gmul_per_second": useful_multiplications as f64 / gpu_active_median.as_secs_f64() / 1e9
        },
        "timings": {
            "preparation_seconds": preparation.as_secs_f64(),
            "cpu_median_seconds": cpu_median.as_secs_f64(),
            "gpu_wall_median_seconds": gpu_wall_median.as_secs_f64(),
            "gpu_active_median_seconds": gpu_active_median.as_secs_f64(),
            "repeats": repeats
        },
        "guards": {
            "exact_q_evals": exact_q_evals,
            "no_execute_allocations": invocation.execute_device_buffer_allocations() == 0
        },
        "resources": {
            "gpu_seconds": gpu_active_times.iter().sum::<Duration>().as_secs_f64(),
            "lookup_plane_bytes": invocation.logical_lookup_plane_bytes(),
            "logical_branch_bytes": invocation.logical_branch_bytes(),
            "logical_weight_bytes": invocation.logical_weight_bytes(),
            "useful_multiplications": useful_multiplications
        },
        "workload": {
            "log_n": log_n,
            "rows": rows,
            "groups": GROUPS,
            "factors_per_group": FACTORS_PER_GROUP,
            "chunk_bits": 8,
            "threads_per_threadgroup": threads,
            "layout": "bit-reversed table-major with cycle inverse",
            "timed_boundary": "resident first-message dispatch and wait"
        },
        "fingerprint": {
            "device": context.device_info().name,
            "max_buffer_length": context.device_info().max_buffer_length,
            "cpu_threads": std::thread::available_parallelism()?.get()
        }
    });
    println!("{}", serde_json::to_string(&output)?);
    Ok(())
}
