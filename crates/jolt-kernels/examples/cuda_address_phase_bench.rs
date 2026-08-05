#![expect(
    clippy::expect_used,
    clippy::print_stdout,
    reason = "benchmark example: fails loudly and reports to stdout"
)]

use std::time::Instant;

use jolt_field::{Fr, FromPrimitiveInt, MulPow2};
use jolt_kernels::cuda::{init_raf_buckets, init_suffix_buckets, shared_context, DeviceRows};
use jolt_lookup_tables::lookup_bits::LookupBits;
use jolt_lookup_tables::tables::LookupTableKind;
use jolt_lookup_tables::XLEN as RISCV_XLEN;
use rayon::prelude::*;

const ADDRESS_BITS: usize = 128;
const CHUNK_LEN: usize = 8;
const CHUNK_SIZE: usize = 1 << CHUNK_LEN;

struct Rows {
    lookup_index: Vec<u128>,
    table_index: Vec<Option<usize>>,
    raf_flag: Vec<bool>,
    u_evals: Vec<Fr>,
}

fn rows(log_t: usize) -> Rows {
    let tables: Vec<LookupTableKind<RISCV_XLEN>> =
        <LookupTableKind<RISCV_XLEN> as strum::IntoEnumIterator>::iter().collect();
    let cycles = 1usize << log_t;
    let mut lookup_index = Vec::with_capacity(cycles);
    let mut table_index = Vec::with_capacity(cycles);
    let mut raf_flag = Vec::with_capacity(cycles);
    let mut u_evals = Vec::with_capacity(cycles);
    for j in 0..cycles {
        let mixed = (j as u64)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(7);
        lookup_index.push((u128::from(mixed) << 61) | u128::from(mixed.rotate_left(17)));
        table_index.push(if mixed.is_multiple_of(11) {
            None
        } else {
            Some(tables[(mixed % tables.len() as u64) as usize].index())
        });
        raf_flag.push(mixed.is_multiple_of(3));
        u_evals.push(Fr::from_u64(mixed % 1_000_003 + 1));
    }
    Rows {
        lookup_index,
        table_index,
        raf_flag,
        u_evals,
    }
}

fn legacy_raf(rows: &Rows, suffix_len: usize) -> Vec<Fr> {
    let mask = if suffix_len >= 128 {
        u128::MAX
    } else {
        (1u128 << suffix_len) - 1
    };
    let threads = rayon::current_num_threads();
    let chunk = rows.lookup_index.len().div_ceil(threads).max(1);
    let mut buckets = rows
        .lookup_index
        .par_chunks(chunk)
        .zip(rows.u_evals.par_chunks(chunk))
        .zip(rows.raf_flag.par_chunks(chunk))
        .fold(
            || vec![Fr::from_u64(0); 5 * CHUNK_SIZE],
            |mut acc, ((indices, u_evals), flags)| {
                for ((&index, &u), &flag) in indices.iter().zip(u_evals).zip(flags) {
                    let bucket = ((index >> suffix_len) as usize) & (CHUNK_SIZE - 1);
                    let suffix_bits = index & mask;
                    if flag {
                        acc[CHUNK_SIZE + bucket] += u;
                        if suffix_bits != 0 {
                            acc[4 * CHUNK_SIZE + bucket] += u * Fr::from_u128(suffix_bits);
                        }
                    } else {
                        acc[bucket] += u;
                        let (left, right) = LookupBits::new(suffix_bits, suffix_len).uninterleave();
                        let left = u64::from(left);
                        if left != 0 {
                            acc[2 * CHUNK_SIZE + bucket] += u * Fr::from_u64(left);
                        }
                        let right = u64::from(right);
                        if right != 0 {
                            acc[3 * CHUNK_SIZE + bucket] += u * Fr::from_u64(right);
                        }
                    }
                }
                acc
            },
        )
        .reduce(
            || vec![Fr::from_u64(0); 5 * CHUNK_SIZE],
            |mut a, b| {
                for (slot, value) in a.iter_mut().zip(&b) {
                    *slot += *value;
                }
                a
            },
        );
    for slot in 0..CHUNK_SIZE {
        buckets[slot] = buckets[slot].mul_pow_2(suffix_len / 2);
        buckets[CHUNK_SIZE + slot] = buckets[CHUNK_SIZE + slot].mul_pow_2(suffix_len);
    }
    buckets
}

fn legacy_suffix(rows: &Rows, present: &[LookupTableKind<RISCV_XLEN>], suffix_len: usize) -> usize {
    let mask = if suffix_len >= 128 {
        u128::MAX
    } else {
        (1u128 << suffix_len) - 1
    };
    let mut by_table: Vec<Vec<usize>> = vec![Vec::new(); LookupTableKind::<RISCV_XLEN>::COUNT];
    for (j, table) in rows.table_index.iter().enumerate() {
        if let Some(index) = table {
            by_table[*index].push(j);
        }
    }
    present
        .par_iter()
        .map(|table| {
            let suffixes = table.suffixes();
            let mut acc = vec![Fr::from_u64(0); suffixes.len() * CHUNK_SIZE];
            for &j in &by_table[table.index()] {
                let index = rows.lookup_index[j];
                let bucket = ((index >> suffix_len) as usize) & (CHUNK_SIZE - 1);
                let bits = LookupBits::new(index & mask, suffix_len);
                let u = rows.u_evals[j];
                for (slot, suffix) in suffixes.iter().enumerate() {
                    let value = suffix.suffix_mle(bits);
                    if value != 0 {
                        acc[slot * CHUNK_SIZE + bucket] += u * Fr::from_u64(value);
                    }
                }
            }
            acc.len()
        })
        .sum()
}

fn main() {
    let Some(context) = shared_context() else {
        println!("no CUDA device; skipping");
        return;
    };

    println!(
        "legacy = legacy's own algorithm (rayon, {} threads)",
        rayon::current_num_threads()
    );
    println!(
        "{:>6}  {:>12}  {:>12}  {:>9}  {:>12}  {:>12}  {:>9}",
        "log_T", "raf_legacy", "raf_cuda", "speedup", "sfx_legacy", "sfx_cuda", "speedup"
    );

    for log_t in [16usize, 18, 20, 22] {
        let rows_host = rows(log_t);
        let present: Vec<LookupTableKind<RISCV_XLEN>> = {
            let mut seen: Vec<LookupTableKind<RISCV_XLEN>> = Vec::new();
            for table in rows_host.table_index.iter().flatten() {
                let kind = <LookupTableKind<RISCV_XLEN> as strum::IntoEnumIterator>::iter()
                    .find(|candidate| candidate.index() == *table)
                    .expect("table index");
                if !seen.contains(&kind) {
                    seen.push(kind);
                }
            }
            seen
        };
        let suffix_len = ADDRESS_BITS - CHUNK_LEN;

        let start = Instant::now();
        let _ = legacy_raf(&rows_host, suffix_len);
        let raf_legacy = start.elapsed();

        let start = Instant::now();
        let _ = legacy_suffix(&rows_host, &present, suffix_len);
        let sfx_legacy = start.elapsed();

        let device = DeviceRows::new(
            context,
            &rows_host.lookup_index,
            &rows_host.table_index,
            &rows_host.raf_flag,
        )
        .expect("device rows");
        let u_evals = context.upload(&rows_host.u_evals).expect("upload");

        let _ = init_raf_buckets(context, &device, &u_evals, ADDRESS_BITS, 0).expect("warm");

        let start = Instant::now();
        let _ = init_raf_buckets(context, &device, &u_evals, ADDRESS_BITS, 0).expect("raf");
        let raf_cuda = start.elapsed();

        let start = Instant::now();
        let _ = init_suffix_buckets(context, &device, &u_evals, &present, ADDRESS_BITS, 0)
            .expect("suffix");
        let sfx_cuda = start.elapsed();

        println!(
            "{log_t:>6}  {:>12.3?}  {:>12.3?}  {:>8.2}x  {:>12.3?}  {:>12.3?}  {:>8.2}x",
            raf_legacy,
            raf_cuda,
            raf_legacy.as_secs_f64() / raf_cuda.as_secs_f64(),
            sfx_legacy,
            sfx_cuda,
            sfx_legacy.as_secs_f64() / sfx_cuda.as_secs_f64(),
        );
    }
}
