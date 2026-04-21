//! BN254 Fr native-field coprocessor SDK.
//!
//! Thin wrapper crate that exposes the FieldOp / FMov{I2F,F2I} instructions
//! introduced in `tracer/src/instruction/field_op.rs` as a guest-friendly
//! `Fr` type with `add / sub / mul / inv` methods.
//!
//! Unlike the `register_inline`-based inlines (SHA256, Keccak256, etc.) the
//! BN254 Fr family is a **first-class instruction** in the refactor tracer
//! (opcode `0x0B`, funct7 `0x40`). This crate only needs to emit the right
//! RISC-V words via inline asm — no host-side registration.

#![cfg_attr(not(feature = "host"), no_std)]

// RISC-V encoding constants (kept in sync with `tracer::instruction::field_op`).
pub const INLINE_OPCODE: u32 = 0x0B;
pub const BN254_FR_FUNCT7: u32 = 0x40;

pub const FUNCT3_FMUL: u32 = 0x02;
pub const FUNCT3_FADD: u32 = 0x03;
pub const FUNCT3_FINV: u32 = 0x04;
pub const FUNCT3_FSUB: u32 = 0x05;
pub const FUNCT3_FMOV_I2F: u32 = 0x06;
pub const FUNCT3_FMOV_F2I: u32 = 0x07;

/// BN254 Fr modulus `p` in natural-form little-endian limbs.
/// `p = 21888242871839275222246405745257275088548364400416034343698204186575808495617`.
pub const BN254_FR_MODULUS: [u64; 4] = [
    0x43e1_f593_f000_0001,
    0x2833_e848_79b9_7091,
    0xb850_45b6_8181_585d,
    0x3064_4e72_e131_a029,
];

pub mod sdk;
pub use sdk::*;
