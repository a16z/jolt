use std::fs::File;
use std::path::PathBuf;

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use eyre::Result;

pub use ark_bn254::{Fr as F, G1Projective as G};
pub use ark_ec::CurveGroup;
pub use jolt_core::field::JoltField;
use jolt_core::poly::commitment::hyrax::HyraxScheme;

pub use common::{
    constants::MEMORY_OPS_PER_INSTRUCTION,
    rv_trace::{MemoryOp, RV32IM},
};
pub use jolt_core::host;
pub use jolt_core::jolt::instruction;
pub use jolt_core::jolt::vm::{
    bytecode::BytecodeRow,
    rv32i_vm::{RV32IJoltProof, RV32IJoltVM, RV32I},
    Jolt, JoltCommitments, JoltPreprocessing, JoltProof,
};
pub use tracer;

pub type CommitmentScheme = HyraxScheme<G>;

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct Proof {
    pub proof: RV32IJoltProof<F, CommitmentScheme>,
    pub commitments: JoltCommitments<CommitmentScheme>,
}

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

    /// Serializes the proof to a byte vector
    pub fn serialize_to_bytes(&self) -> Result<Vec<u8>> {
        let mut buffer = Vec::new();
        self.serialize_compressed(&mut buffer)?;
        Ok(buffer)
    }

    /// Deserializes a proof from a byte vector
    pub fn deserialize_from_bytes(bytes: &[u8]) -> Result<Self> {
        let cursor = std::io::Cursor::new(bytes);
        Ok(Proof::deserialize_compressed(cursor)?)
    }
}
