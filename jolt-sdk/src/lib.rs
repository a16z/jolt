#![cfg_attr(not(feature = "std"), no_std)]

extern crate jolt_sdk_macros;

pub use jolt_sdk_macros::provable;
pub use postcard;

#[cfg(feature = "std")]
pub use ark_ec::CurveGroup;
#[cfg(feature = "std")]
pub use ark_ff::PrimeField;
#[cfg(feature = "std")]
pub use ark_bn254::{Fr as F, G1Projective as G};
#[cfg(feature = "std")]
pub use common::{constants::MEMORY_OPS_PER_INSTRUCTION, rv_trace::MemoryOp};
#[cfg(feature = "std")]
pub use liblasso::host;
#[cfg(feature = "std")]
pub use liblasso::jolt::instruction;
#[cfg(feature = "std")]
pub use liblasso::jolt::vm::{
    bytecode::BytecodeRow,
    rv32i_vm::{RV32IJoltProof, RV32IJoltVM, RV32I},
    Jolt, JoltCommitments, JoltPreprocessing, JoltProof,
};
#[cfg(feature = "std")]
pub use tracer;

#[cfg(feature = "std")]
pub struct Proof {
    pub proof: RV32IJoltProof<F, G>,
    pub commitments: JoltCommitments<G>,
}

