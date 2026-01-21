//! Runtime advice system constants and utilities.
//!
//! The advice system allows guest programs to provide non-deterministic witness data
//! during execution that will be verified by the proof system.

/// ECALL number for writing advice data during emulation.
/// The advice tape stores data from the first emulation pass that can be read
/// during the second (proving) pass via advice instructions.
pub const JOLT_ADVICE_WRITE_ECALL_NUM: u32 = 0xADBABE;
