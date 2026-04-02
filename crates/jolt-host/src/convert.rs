//! Trace-to-witness conversion: [`CycleRow`] → [`CycleInput`].
//!
//! The single function [`cycle_to_input`] bridges the tracer's per-cycle data
//! to the witness polynomial builder. It extracts the five values that
//! [`Polynomials::push`](jolt_witness::Polynomials::push) needs:
//! register increment, RAM increment, lookup index, dense PC index, and
//! remapped RAM address.

use common::jolt_device::MemoryLayout;
use jolt_witness::CycleInput;

use crate::bytecode::BytecodePreprocessing;
use crate::CycleRow;

/// Convert a single execution cycle to a [`CycleInput`] for witness generation.
///
/// # Arguments
///
/// - `cycle` — one step of the RISC-V execution trace
/// - `bytecode` — preprocessed bytecode table (for PC → dense index mapping)
/// - `memory_layout` — memory address layout (for RAM address remapping)
pub fn cycle_to_input(
    cycle: &impl CycleRow,
    bytecode: &BytecodePreprocessing,
    memory_layout: &MemoryLayout,
) -> CycleInput {
    if cycle.is_noop() {
        return CycleInput::PADDING;
    }

    let rd_inc = match cycle.rd_write() {
        Some((_, pre, post)) => post as i128 - pre as i128,
        None => 0,
    };

    let ram_inc = match (cycle.ram_read_value(), cycle.ram_write_value()) {
        (Some(pre), Some(post)) => post as i128 - pre as i128,
        _ => 0,
    };

    let ram_address = cycle.ram_access_address().map(|addr| {
        let lowest = memory_layout.get_lowest_address();
        assert!(
            addr >= lowest,
            "unexpected RAM address {addr:#x} below lowest {lowest:#x}"
        );
        (addr - lowest) / 8
    });

    CycleInput {
        rd_inc,
        ram_inc,
        lookup_index: cycle.lookup_index(),
        pc_index: bytecode.get_pc(cycle) as u32,
        ram_address,
    }
}
