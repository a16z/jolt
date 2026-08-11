#![expect(
    clippy::expect_used,
    clippy::print_stdout,
    reason = "benchmark example: fails loudly and reports to stdout"
)]

use std::time::Instant;

use jolt_field::{Fr, FromPrimitiveInt};
use jolt_kernels::cuda::{shared_context, DeviceAddressPhase};
use jolt_lookup_tables::tables::LookupTableKind;
use jolt_lookup_tables::XLEN as RISCV_XLEN;

const ADDRESS_BITS: usize = 128;

struct Rows {
    lookup_index: Vec<u128>,
    table_index: Vec<Option<usize>>,
    raf_flag: Vec<bool>,
}

fn rows(log_t: usize) -> Rows {
    let tables: Vec<LookupTableKind<RISCV_XLEN>> = LookupTableKind::<RISCV_XLEN>::iter().collect();
    let cycles = 1usize << log_t;
    let mut lookup_index = Vec::with_capacity(cycles);
    let mut table_index = Vec::with_capacity(cycles);
    let mut raf_flag = Vec::with_capacity(cycles);
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
    }
    Rows {
        lookup_index,
        table_index,
        raf_flag,
    }
}

fn main() {
    let Some(context) = shared_context() else {
        println!("no CUDA device; skipping");
        return;
    };

    println!("device address phase: all {ADDRESS_BITS} rounds + 16 phase transitions");
    println!(
        "{:>6}  {:>12}  {:>12}  {:>12}  {:>12}",
        "log_T", "prepare", "message", "bind", "total"
    );

    for log_t in [16usize, 18, 20, 22] {
        let host = rows(log_t);
        let gamma = Fr::from_u64(31);
        let r_reduction: Vec<Fr> = (0..log_t)
            .map(|i| Fr::from_u64(31 + 7 * i as u64))
            .collect();

        let start = Instant::now();
        let mut phase = DeviceAddressPhase::new(
            context,
            &host.lookup_index,
            &host.table_index,
            &host.raf_flag,
            &r_reduction,
            ADDRESS_BITS,
        )
        .expect("device address phase");
        let prepare = start.elapsed();

        let mut message = std::time::Duration::ZERO;
        let mut bind = std::time::Duration::ZERO;
        for round in 0..ADDRESS_BITS {
            let start = Instant::now();
            let _ = phase
                .round_message_hinted(context, gamma, Fr::from_u64(0))
                .expect("message");
            message += start.elapsed();
            let start = Instant::now();
            phase
                .bind(context, Fr::from_u64(17 + round as u64))
                .expect("bind");
            bind += start.elapsed();
        }
        let rounds = message + bind;

        println!(
            "{log_t:>6}  {:>12.3?}  {:>12.3?}  {:>12.3?}  {:>12.3?}",
            prepare,
            message,
            bind,
            prepare + rounds
        );
    }
}
