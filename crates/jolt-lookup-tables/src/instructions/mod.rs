//! Per-instruction `InstructionLookupTable` and `LookupQuery` impls.
//!
//! Split into:
//! - [`riscv`]: RV64I/M base ISA + RV64 W-suffix instructions
//! - [`virt`]: virtual (synthesized) instructions used by the proving system
//! - [`implicit_carry`]: Jolt custom carry-consuming arithmetic (`ADDC`, `MULC`)

#[cfg(feature = "implicit-carry")]
pub mod implicit_carry;
pub mod riscv;
pub mod virt;

#[cfg(test)]
pub mod test;
