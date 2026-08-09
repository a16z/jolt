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
    let tables: Vec<LookupTableKind<RISCV_XLEN>> =
        <LookupTableKind<RISCV_XLEN> as strum::IntoEnumIterator>::iter().collect();
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

    println!("address phase, split by cost center (128 rounds, 16 phases of 8)");
    println!(
        "{:>6}  {:>12}  {:>12}  {:>12}  {:>12}",
        "log_T", "phase_init", "round_msgs", "binds", "total"
    );

    for log_t in [16usize, 20, 22] {
        let host = rows(log_t);
        let r_reduction: Vec<Fr> = (0..log_t).map(|i| Fr::from_u64(i as u64 + 3)).collect();
        let gamma = Fr::from_u64(7);

        // `new` runs init_phase(0); later phase inits happen inside bind.
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
        let mut init = start.elapsed();

        let mut messages = std::time::Duration::ZERO;
        let mut binds = std::time::Duration::ZERO;
        for round in 0..ADDRESS_BITS {
            let start = Instant::now();
            let _ = phase
                .round_message_hinted(context, gamma, Fr::from_u64(0))
                .expect("round message");
            messages += start.elapsed();

            let challenge = Fr::from_u64(round as u64 + 71);
            let start = Instant::now();
            phase.bind(context, challenge).expect("bind");
            let elapsed = start.elapsed();
            // Every 8th bind also runs the next phase's init scans; attribute
            // that to phase_init, not to binding.
            if (round + 1).is_multiple_of(8) {
                init += elapsed;
            } else {
                binds += elapsed;
            }
        }

        println!(
            "{log_t:>6}  {:>12.3?}  {:>12.3?}  {:>12.3?}  {:>12.3?}",
            init,
            messages,
            binds,
            init + messages + binds,
        );
    }
}
