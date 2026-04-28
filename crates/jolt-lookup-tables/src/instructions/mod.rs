//! Per-instruction `InstructionLookupTable` and `LookupQuery` impls.
//!
//! Split into:
//! - [`riscv`]: RV64I/M base ISA + RV64 W-suffix instructions
//! - [`virt`]: virtual (synthesized) instructions used by the proving system

pub mod riscv;
pub mod virt;
