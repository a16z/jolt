//! Device extraction of trace-backed witness atoms for the CUDA kernels.
//!
//! The trace backend and optimized kernels derive typed bundles on the host.
//! This module instead packs native row values, uploads a cycle window, and
//! derives reusable integer columns on the caller's CUDA stream. Instruction
//! descriptors select device formulas; flags and lookup-table indices come
//! from the active instruction profile. Unsupported descriptors are rejected
//! by the upload path.
//!
//! This is an extraction interface, not a second `JoltWitnessPlane`: program
//! metadata and non-trace oracles remain with the host witness. Packing uses
//! a fixed CUDA ABI rather than Rust enum discriminants or field elements.

use cudarc::driver::CudaSlice;
use jolt_riscv::CircuitFlags;
use jolt_riscv::InstructionFlags;
#[cfg(not(feature = "field-inline"))]
pub(crate) use jolt_riscv::RV64IMAC_JOLT as CUDA_PROFILE;
#[cfg(feature = "field-inline")]
pub(crate) use jolt_riscv::RV64IMAC_JOLT_FIELD_INLINE as CUDA_PROFILE;
use tables::TABLE_INDEX_ABSENT as TablesTableIndexAbsent;

mod descriptors;
mod device;
mod packed;
mod tables;
#[cfg(test)]
mod tests;

pub use device::{DeviceTrace, COLD};
pub use packed::{
    PackedTrace, EXTRA_IMM_HI, EXTRA_IMM_LO, EXTRA_RAM_READ, EXTRA_RAM_WRITE, EXTRA_RD_POST,
    EXTRA_RD_PRE, EXTRA_REGISTERS, EXTRA_RS1, EXTRA_RS2, EXTRA_WORDS, NO_SEQUENCE, RAM_NO_ACCESS,
    REGISTER_ABSENT,
};

/// Native per-cycle values shared by CUDA relations.
///
/// Signed increments and the right instruction input use low/high `u64`
/// pairs in two's-complement form; product magnitudes use unsigned pairs.
/// Flags occupy the explicit `FLAG_BIT_*` positions. No column here is in
/// Montgomery form; the arithmetic kernels perform field promotion.
pub struct DeviceAtomColumns {
    pub flags: CudaSlice<u32>,
    pub table_index: CudaSlice<u32>,
    pub bytecode_pc: CudaSlice<u64>,
    pub rd_pre_value: CudaSlice<u64>,
    pub rs1_address: CudaSlice<u32>,
    pub rs2_address: CudaSlice<u32>,
    pub rd_address: CudaSlice<u32>,
    pub rd_inc: CudaSlice<u64>,
    pub ram_inc: CudaSlice<u64>,
    pub left_instruction_input: CudaSlice<u64>,
    pub right_instruction_input: CudaSlice<u64>,
    pub left_lookup_operand: CudaSlice<u64>,
    pub right_lookup_operand: CudaSlice<u64>,
    pub lookup_output: CudaSlice<u64>,
    pub product_magnitude: CudaSlice<u64>,
}

impl DeviceAtomColumns {
    pub fn device_bytes(&self) -> usize {
        let words = self.flags.len()
            + self.table_index.len()
            + self.rs1_address.len()
            + self.rs2_address.len()
            + self.rd_address.len();
        let wide = self.bytecode_pc.len()
            + self.rd_pre_value.len()
            + self.rd_inc.len()
            + self.ram_inc.len()
            + self.left_instruction_input.len()
            + self.right_instruction_input.len()
            + self.left_lookup_operand.len()
            + self.right_lookup_operand.len()
            + self.lookup_output.len()
            + self.product_magnitude.len();
        words * size_of::<u32>() + wide * size_of::<u64>()
    }
}

pub const FLAG_BIT_CIRCUIT_BASE: u32 = 0;

pub const FLAG_BIT_INSTRUCTION_BASE: u32 = 14;

pub const FLAG_BIT_RAF: u32 = 20;

pub const FLAG_BIT_NOOP_ROW: u32 = 21;

pub const FLAG_BIT_NEXT_IS_NOOP: u32 = 22;

pub const FLAG_BIT_RAM_HAMMING: u32 = 23;

pub const FLAG_BIT_SHOULD_BRANCH: u32 = 24;

pub const FLAG_BIT_SHOULD_JUMP: u32 = 25;

pub const FLAG_BIT_PRODUCT_NEGATIVE: u32 = 26;

pub const REGISTER_ADDRESS_ABSENT: u32 = u32::MAX;

pub const TABLE_INDEX_ABSENT: u32 = TablesTableIndexAbsent;

pub use tables::{PACK_CIRCUIT_ORDER, PACK_INSTRUCTION_ORDER};

/// CUDA packed-bit position, or `None` for flags outside `PACK_CIRCUIT_ORDER`
/// (including the field-inline circuit flags).
pub fn circuit_flag_bit(flag: CircuitFlags) -> Option<u32> {
    let slot = PACK_CIRCUIT_ORDER.iter().position(|&entry| entry == flag)?;
    Some(FLAG_BIT_CIRCUIT_BASE + slot as u32)
}

pub fn instruction_flag_bit(flag: InstructionFlags) -> Option<u32> {
    let slot = PACK_INSTRUCTION_ORDER
        .iter()
        .position(|&entry| entry == flag)?;
    Some(FLAG_BIT_INSTRUCTION_BASE + slot as u32)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DeviceTraceColumn {
    MappedPcWord,
    RemappedRamWord { addresses: usize },
}
