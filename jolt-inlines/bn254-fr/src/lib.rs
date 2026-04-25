//! BN254 Fr native-field coprocessor SDK for the Jolt zkVM.
//!
//! Exposes:
//! - ISA constants (`FIELD_OP_OPCODE`, `BN254_FR_FUNCT7`,
//!   `BN254_FR_SLL_FUNCT7`, the 9 funct3 selectors).
//! - R-type instruction-word encoders (`encode_field_op`,
//!   `encode_field_mov`, `encode_field_sll64/128/192`,
//!   `encode_field_assert_eq`).
//! - Host-side Fr reference arithmetic + load/extract sequence builders
//!   (behind the `host` feature).
//!
//! Downstream: guest programs call the host-feature-gated `Fr::{add,sub,
//! mul,inv}` methods which emit the correct instruction-word sequence
//! via inline asm on a RISC-V target, or fall through to `ark_bn254::Fr`
//! on the host for tracing-free execution.

#![cfg_attr(not(feature = "host"), no_std)]

pub mod encode;

#[cfg(feature = "host")]
pub mod sdk;

pub use encode::*;
