//! BN254 Fr native-field coprocessor SDK for Jolt zkVM.
//!
//! Exposes the 9 R-type instructions (custom-0, opcode 0x0B) that map to the
//! 16 × 256-bit field-register file inside the tracer's emulator:
//!
//! | funct7 | funct3 | Mnemonic        | Semantics                                        |
//! |:------:|:------:|:----------------|:-------------------------------------------------|
//! | 0x40   | 0x02   | FMUL            | `FReg[frd] = FReg[frs1] · FReg[frs2]` (Fr)       |
//! | 0x40   | 0x03   | FADD            | `FReg[frd] = FReg[frs1] + FReg[frs2]` (Fr)       |
//! | 0x40   | 0x04   | FINV            | `FReg[frd] = FReg[frs1]⁻¹` (Fr; 0 → 0)           |
//! | 0x40   | 0x05   | FSUB            | `FReg[frd] = FReg[frs1] − FReg[frs2]` (Fr)       |
//! | 0x40   | 0x06   | FieldAssertEq   | assert `FReg[frs1] == FReg[frs2]`                |
//! | 0x40   | 0x07   | FieldMov        | `FReg[frd] = [XReg[rs1], 0, 0, 0]`               |
//! | 0x41   | 0x00   | FieldSLL64      | `FReg[frd] = XReg[rs1] · 2^64`                   |
//! | 0x41   | 0x01   | FieldSLL128     | `FReg[frd] = XReg[rs1] · 2^128`                  |
//! | 0x41   | 0x02   | FieldSLL192     | `FReg[frd] = XReg[rs1] · 2^192`                  |
//!
//! The guest path emits R-type asm with the appropriate funct3/funct7 and
//! the tracer decodes + executes them against the FieldReg state. The host
//! path (enabled by `--features host`) is a thin facade over `ark-bn254::Fr`
//! and exists for unit tests, fixture generation, and host-side simulation.

#![cfg_attr(not(feature = "host"), no_std)]

pub const FIELD_OP_OPCODE: u32 = 0x0B;

pub const BN254_FR_FUNCT7: u32 = 0x40;
pub const BN254_FR_SLL_FUNCT7: u32 = 0x41;

pub const FUNCT3_FMUL: u32 = 0x02;
pub const FUNCT3_FADD: u32 = 0x03;
pub const FUNCT3_FINV: u32 = 0x04;
pub const FUNCT3_FSUB: u32 = 0x05;
pub const FUNCT3_FIELD_ASSERT_EQ: u32 = 0x06;
pub const FUNCT3_FIELD_MOV: u32 = 0x07;

pub const FUNCT3_FIELD_SLL64: u32 = 0x00;
pub const FUNCT3_FIELD_SLL128: u32 = 0x01;
pub const FUNCT3_FIELD_SLL192: u32 = 0x02;

/// Number of 256-bit field registers in the FR coprocessor's register file.
pub const FIELD_REG_COUNT: usize = 16;

pub mod sdk;
