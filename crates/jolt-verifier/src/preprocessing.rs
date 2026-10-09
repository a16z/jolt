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

/// Committed-program verifier inputs: trusted bytecode and program-image
/// commitments plus the program metadata they bind to.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(bound(
    serialize = "PCS::Output: Serialize",
    deserialize = "PCS::Output: serde::de::DeserializeOwned"
))]
pub struct CommittedProgramPreprocessing<PCS: CommitmentScheme> {
    pub meta: ProgramMetadata,
    pub memory_layout: MemoryLayout,
    pub max_padded_trace_length: usize,
    pub program_image_commitment: PCS::Output,
    pub bytecode_commitment: PCS::Output,
    #[cfg(feature = "akita")]
    pub trace_order: TracePolynomialOrder,
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
#[cfg_attr(
    feature = "akita",
    expect(
        clippy::large_enum_variant,
        reason = "constructed once per preprocessing; boxing Committed buys nothing"
    )
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
#[cfg(all(not(feature = "akita"), not(feature = "field-inline")))]
const PROGRAM_PREPROCESSING_DIGEST_DOMAIN: &[u8] =
    b"jolt/program-preprocessing/dory-whole-bytecode/v1";
// Field flags use the common circuit columns; preprocessing has no side table.
#[cfg(all(not(feature = "akita"), feature = "field-inline"))]
const PROGRAM_PREPROCESSING_DIGEST_DOMAIN: &[u8] =
    b"jolt/program-preprocessing/dory-whole-bytecode/field-inline/v1";

#[cfg(all(feature = "akita", not(feature = "field-inline")))]
const PROGRAM_PREPROCESSING_DIGEST_DOMAIN: &[u8] =
    b"jolt/program-preprocessing/akita-whole-bytecode/v1";
#[cfg(all(feature = "akita", feature = "field-inline"))]
const PROGRAM_PREPROCESSING_DIGEST_DOMAIN: &[u8] =
    b"jolt/program-preprocessing/akita-whole-bytecode/field-inline/v1";

impl<PCS: CommitmentScheme> ProgramPreprocessing<PCS> {
    /// The 32-byte program binding absorbed first into the Fiat-Shamir
    /// preamble: Blake2b-256 over the domain tag and this value's bincode
    /// encoding. Hashing the whole type binds every field the verifier trusts
    /// (mode, bytecode or its commitments, RAM image, memory layout, trace
    /// bound) without a hand-maintained field list, so a field added to any
    /// preprocessing type enters the digest by construction. A prover and a
    /// separately built verifier must agree on `field-inline` (selects the
    /// domain tag) and on the PCS (`Committed` carries PCS-specific fields).
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
    /// the bytecode row schema is shared, while field-inline selects its own
    /// digest domain for the extended instruction and proof profile.
    #[cfg(all(not(feature = "akita"), not(feature = "field-inline")))]
    const FULL_PROGRAM_DIGEST: [u8; 32] = [
        244, 6, 173, 180, 102, 227, 123, 89, 175, 147, 200, 109, 100, 66, 232, 11, 109, 231, 53,
        24, 12, 157, 107, 201, 228, 48, 74, 158, 19, 246, 64, 42,
    ];
    #[cfg(all(not(feature = "akita"), feature = "field-inline"))]
    const FULL_PROGRAM_DIGEST: [u8; 32] = [
        181, 23, 45, 136, 228, 24, 238, 134, 235, 114, 52, 43, 248, 33, 38, 232, 19, 33, 167, 38,
        238, 71, 235, 184, 143, 205, 31, 35, 121, 138, 3, 119,
    ];
    #[cfg(all(feature = "akita", not(feature = "field-inline")))]
    const FULL_PROGRAM_DIGEST: [u8; 32] = [
        10, 241, 10, 25, 152, 236, 113, 66, 234, 148, 180, 8, 19, 181, 13, 175, 248, 211, 234, 243,
        37, 208, 9, 65, 14, 159, 55, 184, 173, 177, 237, 213,
    ];
    #[cfg(all(feature = "akita", feature = "field-inline"))]
    const FULL_PROGRAM_DIGEST: [u8; 32] = [
        15, 187, 25, 179, 115, 141, 121, 127, 18, 224, 184, 85, 46, 153, 30, 31, 49, 77, 174, 174,
        246, 42, 254, 70, 82, 144, 107, 4, 129, 86, 83, 51,
    ];
    #[cfg(all(not(feature = "akita"), not(feature = "field-inline")))]
    const COMMITTED_PROGRAM_DIGEST: [u8; 32] = [
        70, 90, 230, 155, 241, 24, 184, 115, 134, 123, 194, 81, 78, 30, 203, 47, 124, 122, 155, 66,
        29, 231, 235, 138, 9, 161, 39, 8, 44, 195, 136, 136,
    ];
    #[cfg(all(feature = "akita", not(feature = "field-inline")))]
    const COMMITTED_PROGRAM_DIGEST: [u8; 32] = [
        46, 218, 61, 222, 109, 82, 229, 196, 36, 71, 22, 191, 255, 174, 216, 201, 97, 149, 220, 63,
        57, 1, 117, 198, 80, 239, 126, 253, 235, 189, 178, 98,
    ];
    #[cfg(all(not(feature = "akita"), feature = "field-inline"))]
    const COMMITTED_PROGRAM_DIGEST: [u8; 32] = [
        73, 161, 154, 115, 7, 159, 203, 43, 179, 51, 184, 201, 119, 78, 146, 243, 222, 198, 86,
        130, 115, 122, 141, 30, 154, 5, 176, 194, 247, 206, 114, 167,
    ];
    #[cfg(all(feature = "akita", feature = "field-inline"))]
    const COMMITTED_PROGRAM_DIGEST: [u8; 32] = [
        137, 52, 205, 208, 140, 177, 124, 190, 165, 69, 252, 53, 193, 68, 174, 237, 91, 43, 197,
        129, 25, 27, 4, 20, 124, 93, 228, 56, 33, 25, 6, 75,
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
            program_image_commitment: Commitment::default(),
            bytecode_commitment: Commitment::default(),
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
