#![cfg_attr(not(feature = "std"), no_std)]

extern crate jolt_sdk_macros;

#[cfg(feature = "std")]
use std::fs::File;
#[cfg(feature = "std")]
use std::path::PathBuf;

#[cfg(feature = "std")]
use ark_serialize::{CanonicalSerialize, CanonicalDeserialize};
#[cfg(feature = "std")]
use eyre::Result;

pub use jolt_sdk_macros::provable;
pub use postcard;

#[cfg(feature = "std")]
pub use ark_ec::CurveGroup;
#[cfg(feature = "std")]
pub use ark_ff::PrimeField;
#[cfg(feature = "std")]
pub use ark_bn254::{Fr as F, G1Projective as G};
#[cfg(feature = "std")]
pub use common::{constants::MEMORY_OPS_PER_INSTRUCTION, rv_trace::{MemoryOp, RV32IM}};
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
#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct Proof {
    pub proof: RV32IJoltProof<F, G>,
    pub commitments: JoltCommitments<G>,
}

#[cfg(feature = "std")]
impl Proof {
    /// Gets the byte size of the full proof
    pub fn size(&self) -> Result<usize> {
        let mut buffer = Vec::new();
        self.serialize_compressed(&mut buffer)?;
        Ok(buffer.len())
    }

    /// Saves the proof to a file
    pub fn save_to_file<P: Into<PathBuf>>(&self, path: P) -> Result<()> {
        let file = File::create(path.into())?;
        self.serialize_compressed(file)?;
        Ok(())
    }

    /// Reads a proof from a file
    pub fn from_file<P: Into<PathBuf>>(path: P) -> Result<Self> {
        let file = File::open(path.into())?;
        Ok(Proof::deserialize_compressed(file)?)
    }
}

