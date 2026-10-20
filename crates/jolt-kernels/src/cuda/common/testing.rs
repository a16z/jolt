#![expect(
    clippy::expect_used,
    reason = "test scaffolding: fixture errors fail loudly"
)]

use common::jolt_device::MemoryLayout;
use jolt_field::{Fr, Ring};
use jolt_riscv::{RamRead, RamWrite};
use proptest::prelude::*;
use tracer::instruction::lw::LW;
use tracer::instruction::sw::SW;
use tracer::instruction::Cycle;
use tracer::instruction::RISCVCycle;

pub fn fr(seed: u64) -> Fr {
    Fr::from_u64(seed.wrapping_mul(2_654_435_761) % 1_000_003 + 1)
}

pub fn arb_point(len: usize) -> impl Strategy<Value = Vec<Fr>> {
    proptest::collection::vec(any::<u64>().prop_map(fr), len)
}

pub fn ram_read_cycle(address: u64, value: u64) -> Cycle {
    Cycle::LW(RISCVCycle::<LW> {
        ram_access: RamRead { address, value },
        ..Default::default()
    })
}

pub fn ram_write_cycle(address: u64, pre_value: u64, post_value: u64) -> Cycle {
    Cycle::SW(RISCVCycle::<SW> {
        ram_access: RamWrite {
            address,
            pre_value,
            post_value,
        },
        ..Default::default()
    })
}

pub fn ram_trace(log_t: usize, ram_k: usize) -> Vec<Cycle> {
    (0..1usize << log_t)
        .map(|cycle| {
            let word = 1 + (cycle as u64 * 5) % (ram_k as u64 - 1);
            let address = 8 * word;
            match cycle % 4 {
                0 => Cycle::NoOp,
                1 => ram_read_cycle(address, 900 + cycle as u64),
                2 => ram_write_cycle(address, 100 + cycle as u64, 700 + cycle as u64),
                _ => ram_write_cycle(address, 400 + cycle as u64, 400 + cycle as u64),
            }
        })
        .collect()
}

pub struct RamCell {
    pub row: usize,
    pub col: usize,
    pub prev_val: u64,
    pub next_val: u64,
}

pub fn ram_matrix_cells(log_t: usize, ram_k: usize) -> Vec<RamCell> {
    use jolt_riscv::RamAccess;

    let layout = MemoryLayout::default();
    let cells: Vec<RamCell> = ram_trace(log_t, ram_k)
        .iter()
        .enumerate()
        .filter_map(|(row, cycle)| {
            let (address, prev_val, next_val) = match cycle.ram_access() {
                RamAccess::Read(read) => (read.address, read.value, read.value),
                RamAccess::Write(write) => (write.address, write.pre_value, write.post_value),
                RamAccess::NoOp => return None,
            };
            let col = layout
                .remap_word_address(address)
                .expect("fixture RAM addresses are mapped")?;
            Some(RamCell {
                row,
                col: col as usize,
                prev_val,
                next_val,
            })
        })
        .collect();
    assert!(
        !cells.is_empty(),
        "the RAM fixture produced no accesses to build a matrix from",
    );
    cells
}
