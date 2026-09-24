//! Verifier preprocessing inputs.

use blake2::{digest::consts::U32, Blake2b, Digest};
use common::jolt_device::MemoryLayout;
use jolt_claims::protocols::jolt::JoltRelationId;
#[cfg(feature = "akita")]
use jolt_claims::protocols::jolt::TracePolynomialOrder;
use jolt_crypto::VectorCommitment;
use jolt_openings::CommitmentScheme;
use jolt_program::preprocess::{JoltProgramPreprocessing, ProgramMetadata};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::VerifierError;

/// Committed-program verifier inputs: trusted bytecode-chunk and program-image
/// commitments plus the program metadata they bind to.
/// For Dory, the chunk count is implied by `bytecode_chunk_commitments.len()`.
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
    /// Direct bounded-dense program objects in canonical order: indexed
    /// bytecode chunks, then the program image.
    #[cfg(feature = "akita")]
    pub direct_program_commitments: Vec<PCS::Output>,
    #[cfg(feature = "akita")]
    pub bytecode_chunk_count: usize,
    #[cfg(feature = "akita")]
    pub trace_order: TracePolynomialOrder,
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

/// Program preprocessing in one of two modes. `Full` carries the bytecode
/// table and initial RAM image; `Committed` replaces them with trusted
/// commitments plus metadata.
///
/// The serde layout of everything reachable from this enum is hashed into
/// the Fiat-Shamir preamble by [`digest`](Self::digest) and pinned by the
/// golden tests below, so a field or variant reorder anywhere in that
/// closure is a protocol change: bump `PROGRAM_PREPROCESSING_DIGEST_DOMAIN`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(bound(
    serialize = "PCS::Output: Serialize",
    deserialize = "PCS::Output: serde::de::DeserializeOwned"
))]
#[expect(
    clippy::large_enum_variant,
    reason = "constructed once per preprocessing; boxing Committed buys nothing"
)]
pub enum ProgramPreprocessing<PCS: CommitmentScheme> {
    /// `Arc` so witness backends take an owning handle without deep-cloning
    /// the program-sized tables (serde `rc`: serializes as the contents).
    Full(Arc<JoltProgramPreprocessing>),
    Committed(CommittedProgramPreprocessing<PCS>),
}

impl<PCS: CommitmentScheme> ProgramPreprocessing<PCS> {
    pub fn as_full(&self) -> Option<&JoltProgramPreprocessing> {
        match self {
            Self::Full(full) => Some(full),
            Self::Committed(_) => None,
        }
    }

    /// The owning counterpart of [`as_full`](Self::as_full) — a refcount
    /// bump, never a copy.
    pub fn as_full_arc(&self) -> Option<Arc<JoltProgramPreprocessing>> {
        match self {
            Self::Full(full) => Some(Arc::clone(full)),
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

    /// [`entry_bytecode_index`](Self::entry_bytecode_index), attributing an
    /// entry address absent from the bytecode to the consuming `stage`.
    pub fn entry_bytecode_index_checked(
        &self,
        stage: JoltRelationId,
    ) -> Result<usize, VerifierError> {
        self.entry_bytecode_index()
            .ok_or_else(|| VerifierError::StageClaimPublicInputFailed {
                stage,
                reason: "entry address was not found in bytecode preprocessing".to_string(),
            })
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

/// Domain separator for [`ProgramPreprocessing::digest`]. Bump the version
/// whenever the digest input changes: it is the only compatibility switch a
/// deployed verifier sees.
#[cfg(not(feature = "field-inline"))]
const PROGRAM_PREPROCESSING_DIGEST_DOMAIN: &[u8] = b"jolt/program-preprocessing/v2";
// Accumulating ingress changes the field-inline instruction and proof schemas.
#[cfg(feature = "field-inline")]
const PROGRAM_PREPROCESSING_DIGEST_DOMAIN: &[u8] = b"jolt/program-preprocessing/v3";

impl<PCS: CommitmentScheme> ProgramPreprocessing<PCS> {
    /// The 32-byte program binding absorbed first into the Fiat-Shamir
    /// preamble: Blake2b-256 over the domain tag and this value's bincode
    /// encoding. Hashing the whole type binds every field the verifier trusts
    /// (mode, bytecode or its commitments, RAM image, memory layout, trace
    /// bound) without a hand-maintained field list, so a field added to any
    /// preprocessing type enters the digest by construction. The flip side is
    /// that cfg-gated fields enter it too: a prover and a separately built
    /// verifier must agree on `jolt-program/field-inline` (adds a `Full`
    /// field) and on the PCS (`Committed` carries PCS-specific fields).
    pub(crate) fn digest(&self) -> Result<[u8; 32], VerifierError> {
        let encoded =
            bincode::serde::encode_to_vec(self, bincode::config::standard()).map_err(|error| {
                VerifierError::PreprocessingDigestFailed {
                    reason: error.to_string(),
                }
            })?;
        Ok(
            Blake2b::<U32>::new_with_prefix(PROGRAM_PREPROCESSING_DIGEST_DOMAIN)
                .chain_update(encoded)
                .finalize()
                .into(),
        )
    }
}

/// Verifier inputs for one program: its preprocessing view, the digest that
/// binds it into the Fiat-Shamir preamble, and the PCS and BlindFold setups.
///
/// `preprocessing_digest` is derived from `program` by [`new`](Self::new) and
/// is never on the wire: `Serialize` omits it and `Deserialize` rebuilds
/// through `new`, so a persisted verifier preprocessing cannot carry a stale
/// digest. In memory the field is public and `verify` absorbs it as stored;
/// the Fiat-Shamir attack tests rely on flipping it.
#[derive(Clone, Serialize, Deserialize)]
#[serde(
    into = "VerifierPreprocessingWire<PCS, VC>",
    try_from = "VerifierPreprocessingWire<PCS, VC>",
    bound(
        serialize = "PCS: Clone, VC: Clone, VC::Setup: Serialize",
        deserialize = "VC::Setup: DeserializeOwned"
    )
)]
pub struct JoltVerifierPreprocessing<PCS, VC>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    pub program: ProgramPreprocessing<PCS>,
    pub preprocessing_digest: [u8; 32],
    /// The main PCS setup: every per-polynomial opening on the homomorphic
    /// build, or the complete grouped opening on the `akita` build.
    pub pcs_setup: PCS::VerifierSetup,
    pub vc_setup: Option<VC::Setup>,
}

impl<PCS, VC> JoltVerifierPreprocessing<PCS, VC>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    pub fn new(
        program: ProgramPreprocessing<PCS>,
        pcs_setup: PCS::VerifierSetup,
        vc_setup: Option<VC::Setup>,
    ) -> Result<Self, VerifierError> {
        let preprocessing_digest = program.digest()?;
        Ok(Self {
            program,
            preprocessing_digest,
            pcs_setup,
            vc_setup,
        })
    }
}

/// Wire form of [`JoltVerifierPreprocessing`]: everything except the derived
/// digest.
#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "VC::Setup: Serialize",
    deserialize = "VC::Setup: DeserializeOwned"
))]
struct VerifierPreprocessingWire<PCS, VC>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    program: ProgramPreprocessing<PCS>,
    pcs_setup: PCS::VerifierSetup,
    vc_setup: Option<VC::Setup>,
}

impl<PCS, VC> TryFrom<VerifierPreprocessingWire<PCS, VC>> for JoltVerifierPreprocessing<PCS, VC>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    type Error = VerifierError;

    fn try_from(wire: VerifierPreprocessingWire<PCS, VC>) -> Result<Self, Self::Error> {
        Self::new(wire.program, wire.pcs_setup, wire.vc_setup)
    }
}

impl<PCS, VC> From<JoltVerifierPreprocessing<PCS, VC>> for VerifierPreprocessingWire<PCS, VC>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    fn from(preprocessing: JoltVerifierPreprocessing<PCS, VC>) -> Self {
        Self {
            program: preprocessing.program,
            pcs_setup: preprocessing.pcs_setup,
            vc_setup: preprocessing.vc_setup,
        }
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "digest fixtures fail loudly")]
mod tests {
    use std::sync::Arc;

    use common::jolt_device::{MemoryConfig, MemoryLayout};
    #[cfg(feature = "akita")]
    use jolt_akita::{AkitaCommitment as Commitment, AkitaScheme as Pcs};
    #[cfg(feature = "akita")]
    use jolt_claims::protocols::jolt::TracePolynomialOrder;
    #[cfg(not(feature = "akita"))]
    use jolt_dory::{DoryCommitment as Commitment, DoryScheme as Pcs};
    use jolt_program::preprocess::JoltProgramPreprocessing;
    use jolt_riscv::RV64IMAC_JOLT;

    use super::{CommittedProgramPreprocessing, ProgramPreprocessing};

    /// Golden digests. A change here is a Fiat-Shamir break for every
    /// deployed verifier: bump `PROGRAM_PREPROCESSING_DIGEST_DOMAIN` and say so
    /// in the PR. Pinned separately with and without `jolt-program/field-inline`:
    /// the base-profile fixture has no side table, but enabling the feature adds
    /// its serialized `None` tag to the `Full` bytecode encoding and selects
    /// the field-inline digest domain.
    #[cfg(not(feature = "field-inline"))]
    const FULL_PROGRAM_DIGEST: [u8; 32] = [
        42, 63, 50, 98, 242, 124, 42, 171, 43, 223, 155, 146, 108, 130, 235, 136, 177, 93, 248,
        227, 104, 23, 145, 35, 121, 150, 138, 9, 19, 215, 204, 12,
    ];
    #[cfg(feature = "field-inline")]
    const FULL_PROGRAM_DIGEST: [u8; 32] = [
        84, 207, 166, 21, 38, 96, 179, 244, 64, 246, 59, 246, 249, 194, 204, 57, 196, 148, 76, 225,
        76, 151, 215, 73, 146, 189, 106, 44, 150, 199, 109, 113,
    ];
    #[cfg(all(not(feature = "akita"), not(feature = "field-inline")))]
    const COMMITTED_PROGRAM_DIGEST: [u8; 32] = [
        76, 161, 182, 52, 209, 226, 192, 126, 13, 13, 181, 24, 203, 128, 171, 65, 168, 64, 127,
        107, 153, 86, 181, 56, 83, 191, 66, 19, 164, 158, 146, 116,
    ];
    #[cfg(all(feature = "akita", not(feature = "field-inline")))]
    const COMMITTED_PROGRAM_DIGEST: [u8; 32] = [
        251, 63, 111, 254, 167, 21, 41, 185, 193, 117, 188, 112, 255, 206, 156, 249, 230, 201, 4,
        155, 92, 191, 65, 14, 2, 241, 131, 79, 154, 216, 42, 71,
    ];
    #[cfg(all(not(feature = "akita"), feature = "field-inline"))]
    const COMMITTED_PROGRAM_DIGEST: [u8; 32] = [
        162, 146, 83, 205, 187, 102, 142, 240, 60, 254, 5, 208, 0, 64, 41, 117, 93, 148, 221, 201,
        152, 57, 253, 195, 224, 8, 77, 137, 82, 212, 231, 26,
    ];
    #[cfg(all(feature = "akita", feature = "field-inline"))]
    const COMMITTED_PROGRAM_DIGEST: [u8; 32] = [
        3, 184, 27, 109, 46, 217, 18, 137, 189, 228, 170, 104, 254, 191, 39, 158, 19, 108, 240,
        246, 36, 221, 177, 255, 57, 114, 255, 45, 163, 132, 24, 163,
    ];

    /// An empty program over a real (non-zero) memory layout, so every layout
    /// field participates in the pinned bytes.
    fn program() -> JoltProgramPreprocessing {
        let memory_layout = MemoryLayout::new(&MemoryConfig {
            max_untrusted_advice_size: 4096,
            max_trusted_advice_size: 4096,
            max_input_size: 4096,
            max_output_size: 4096,
            stack_size: 4096,
            heap_size: 65536,
            program_size: Some(1 << 20),
        });
        JoltProgramPreprocessing::new(
            Vec::new(),
            Vec::new(),
            memory_layout,
            0,
            1 << 16,
            RV64IMAC_JOLT,
        )
        .unwrap()
    }

    fn committed() -> CommittedProgramPreprocessing<Pcs> {
        let full = program();
        CommittedProgramPreprocessing {
            meta: full.metadata().unwrap(),
            memory_layout: full.memory_layout,
            max_padded_trace_length: full.max_padded_trace_length,
            #[cfg(not(feature = "akita"))]
            bytecode_chunk_commitments: vec![Commitment::default(), Commitment::default()],
            #[cfg(not(feature = "akita"))]
            program_image_commitment: Commitment::default(),
            #[cfg(feature = "akita")]
            direct_program_commitments: vec![Commitment::default(), Commitment::default()],
            #[cfg(feature = "akita")]
            bytecode_chunk_count: 1,
            #[cfg(feature = "akita")]
            trace_order: TracePolynomialOrder::CycleMajor,
        }
    }

    #[test]
    fn full_program_digest_is_stable() {
        let digest = ProgramPreprocessing::<Pcs>::Full(Arc::new(program()))
            .digest()
            .unwrap();
        assert_eq!(digest, FULL_PROGRAM_DIGEST);
    }

    #[test]
    fn committed_program_digest_is_stable() {
        let digest = ProgramPreprocessing::<Pcs>::Committed(committed())
            .digest()
            .unwrap();
        assert_eq!(digest, COMMITTED_PROGRAM_DIGEST);
    }
}
