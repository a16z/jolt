//! Verifier preprocessing inputs.

use common::jolt_device::MemoryLayout;
use jolt_crypto::VectorCommitment;
use jolt_openings::CommitmentScheme;
use jolt_program::preprocess::{JoltProgramPreprocessing, ProgramMetadata};
use serde::{Deserialize, Serialize};

/// Committed-program verifier inputs: trusted bytecode-chunk and program-image
/// commitments plus the program metadata they bind to. Mirrors `jolt-prover-legacy`'s
/// `CommittedProgramPreprocessing`; the chunk count is implied by
/// `bytecode_chunk_commitments.len()`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(bound(
    serialize = "PCS::Output: Serialize",
    deserialize = "PCS::Output: serde::de::DeserializeOwned"
))]
pub struct CommittedProgramPreprocessing<PCS: CommitmentScheme> {
    pub meta: ProgramMetadata,
    pub memory_layout: MemoryLayout,
    pub max_padded_trace_length: usize,
    #[cfg(not(feature = "akita"))]
    pub bytecode_chunk_commitments: Vec<PCS::Output>,
    #[cfg(not(feature = "akita"))]
    pub program_image_commitment: PCS::Output,
    /// The one packed `W_prog` commitment covering every bytecode lane
    /// sub-column and the program image bytes (the per-chunk/image commitment
    /// pair does not exist on the packed path).
    #[cfg(feature = "akita")]
    pub w_prog_commitment: PCS::Output,
    #[cfg(feature = "akita")]
    pub bytecode_chunk_count: usize,
}

impl<PCS: CommitmentScheme> CommittedProgramPreprocessing<PCS> {
    pub fn bytecode_chunk_count(&self) -> usize {
        #[cfg(not(feature = "akita"))]
        {
            self.bytecode_chunk_commitments.len()
        }
        #[cfg(feature = "akita")]
        {
            self.bytecode_chunk_count
        }
    }
}

/// Program preprocessing in one of two modes, detected at runtime from the
/// deserialized preprocessing exactly like `jolt-prover-legacy`'s
/// `ProgramPreprocessing`: `Full` carries the bytecode table and initial RAM
/// image, `Committed` replaces them with trusted commitments plus metadata.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(bound(
    serialize = "PCS::Output: Serialize",
    deserialize = "PCS::Output: serde::de::DeserializeOwned"
))]
pub enum ProgramPreprocessing<PCS: CommitmentScheme> {
    Full(JoltProgramPreprocessing),
    Committed(CommittedProgramPreprocessing<PCS>),
}

impl<PCS: CommitmentScheme> ProgramPreprocessing<PCS> {
    pub fn as_full(&self) -> Option<&JoltProgramPreprocessing> {
        match self {
            Self::Full(full) => Some(full),
            Self::Committed(_) => None,
        }
    }

    pub fn committed(&self) -> Option<&CommittedProgramPreprocessing<PCS>> {
        match self {
            Self::Full(_) => None,
            Self::Committed(committed) => Some(committed),
        }
    }

    pub fn memory_layout(&self) -> &MemoryLayout {
        match self {
            Self::Full(full) => &full.memory_layout,
            Self::Committed(committed) => &committed.memory_layout,
        }
    }

    pub fn max_padded_trace_length(&self) -> usize {
        match self {
            Self::Full(full) => full.max_padded_trace_length,
            Self::Committed(committed) => committed.max_padded_trace_length,
        }
    }

    pub fn entry_address(&self) -> u64 {
        match self {
            Self::Full(full) => full.bytecode.entry_address,
            Self::Committed(committed) => committed.meta.entry_address,
        }
    }

    pub fn entry_bytecode_index(&self) -> Option<usize> {
        match self {
            Self::Full(full) => full.bytecode.entry_bytecode_index(),
            Self::Committed(committed) => Some(committed.meta.entry_bytecode_index),
        }
    }

    pub fn bytecode_len(&self) -> usize {
        match self {
            Self::Full(full) => full.bytecode.code_size,
            Self::Committed(committed) => committed.meta.bytecode_len,
        }
    }

    pub fn min_bytecode_address(&self) -> u64 {
        match self {
            Self::Full(full) => full.ram.min_bytecode_address,
            Self::Committed(committed) => committed.meta.min_bytecode_address,
        }
    }

    pub fn program_image_len_words(&self) -> usize {
        match self {
            Self::Full(full) => full.ram.bytecode_words.len(),
            Self::Committed(committed) => committed.meta.program_image_len_words,
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "ProgramPreprocessing<PCS>: Serialize, PCS::VerifierSetup: Serialize, VC::Setup: Serialize",
    deserialize = "ProgramPreprocessing<PCS>: serde::de::DeserializeOwned, PCS::VerifierSetup: serde::de::DeserializeOwned, VC::Setup: serde::de::DeserializeOwned"
))]
pub struct JoltVerifierPreprocessing<PCS, VC>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    pub program: ProgramPreprocessing<PCS>,
    pub preprocessing_digest: [u8; 32],
    /// The main PCS setup: every per-polynomial opening on the homomorphic
    /// build, the `W_jolt` object on the `akita` build (whose remaining
    /// objects carry their own shape-exact setups below).
    pub pcs_setup: PCS::VerifierSetup,
    pub vc_setup: Option<VC::Setup>,
    #[cfg(feature = "akita")]
    pub untrusted_advice_setup: Option<PCS::VerifierSetup>,
    #[cfg(feature = "akita")]
    pub trusted_advice_setup: Option<PCS::VerifierSetup>,
    /// Committed-program mode: the `W_prog` object setup.
    #[cfg(feature = "akita")]
    pub w_prog_setup: Option<PCS::VerifierSetup>,
}

impl<PCS, VC> JoltVerifierPreprocessing<PCS, VC>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    pub fn new(
        program: ProgramPreprocessing<PCS>,
        preprocessing_digest: [u8; 32],
        pcs_setup: PCS::VerifierSetup,
        vc_setup: Option<VC::Setup>,
    ) -> Self {
        Self {
            program,
            preprocessing_digest,
            pcs_setup,
            vc_setup,
            #[cfg(feature = "akita")]
            untrusted_advice_setup: None,
            #[cfg(feature = "akita")]
            trusted_advice_setup: None,
            #[cfg(feature = "akita")]
            w_prog_setup: None,
        }
    }
}
