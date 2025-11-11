//! Serialization implementations for Dory commitment scheme types

use crate::transcripts::{AppendToTranscript, Transcript};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, Valid};
use dory::DoryProof as FinalDoryProof;

use super::{ArkG1, ArkG2, ArkGT};

// Re-export the arkworks setup types from dory
pub use dory::backends::arkworks::{ArkworksProverSetup, ArkworksVerifierSetup};

/// Proof data for Dory opening
#[derive(Clone, Debug)]
pub struct DoryProofData {
    pub proof: FinalDoryProof<ArkG1, ArkG2, ArkGT>,
}

impl CanonicalSerialize for DoryProofData {
    fn serialize_with_mode<W: std::io::Write>(
        &self,
        _writer: W,
        _compress: ark_serialize::Compress,
    ) -> Result<(), ark_serialize::SerializationError> {
        Ok(())
    }

    fn serialized_size(&self, _compress: ark_serialize::Compress) -> usize {
        0
    }
}

impl Valid for DoryProofData {
    fn check(&self) -> Result<(), ark_serialize::SerializationError> {
        Ok(())
    }
}

impl CanonicalDeserialize for DoryProofData {
    fn deserialize_with_mode<R: std::io::Read>(
        _reader: R,
        _compress: ark_serialize::Compress,
        _validate: ark_serialize::Validate,
    ) -> Result<Self, ark_serialize::SerializationError> {
        Err(ark_serialize::SerializationError::NotEnoughSpace)
    }
}

impl AppendToTranscript for DoryProofData {
    fn append_to_transcript<S: Transcript>(&self, transcript: &mut S) {
        transcript.append_serializable(self);
    }
}

impl AppendToTranscript for &DoryProofData {
    fn append_to_transcript<S: Transcript>(&self, transcript: &mut S) {
        transcript.append_serializable(*self);
    }
}
