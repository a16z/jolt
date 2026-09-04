//! Per-instruction `InstructionLookupTable` and `LookupQuery` impls.
//!
//! Split into:
//! - [`riscv`]: RV64I/M base ISA + RV64 W-suffix instructions
//! - [`virt`]: virtual (synthesized) instructions used by the proving system

#[cfg(feature = "field-inline")]
pub mod field_inline;
pub mod riscv;
pub mod virt;

#[cfg(test)]
pub mod test;
