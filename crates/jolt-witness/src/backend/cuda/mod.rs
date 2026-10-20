//! Device extraction of trace-backed indices for CUDA kernels.
//!
//! The trace backend and optimized kernels derive typed bundles on the host.
//! CUDA packs native row values, uploads a cycle window, and extracts bytecode
//! PCs, remapped RAM addresses, and lookup indices on the caller's stream.
//! Instruction descriptors select device formulas from the active profile;
//! upload rejects unsupported descriptors.
//!
//! Program metadata and non-trace oracles remain with the host witness. Packing
//! uses a fixed CUDA ABI rather than Rust enum discriminants or field elements.

#[cfg(not(feature = "field-inline"))]
pub(crate) use jolt_riscv::RV64IMAC_JOLT as CUDA_PROFILE;
#[cfg(feature = "field-inline")]
pub(crate) use jolt_riscv::RV64IMAC_JOLT_FIELD_INLINE as CUDA_PROFILE;

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

pub const FLAG_BIT_CIRCUIT_BASE: u32 = 0;

pub const FLAG_BIT_INSTRUCTION_BASE: u32 = 14;

pub const FLAG_BIT_RAF: u32 = 20;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DeviceTraceColumn {
    MappedPcWord,
    RemappedRamWord { addresses: usize },
}
