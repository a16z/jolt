use common::jolt_device::MemoryLayout;
use jolt_claims::protocols::jolt::{JoltRelationId, TracePolynomialOrder};
use jolt_crypto::VectorCommitment;
use jolt_openings::CommitmentScheme;
use jolt_program::preprocess::{JoltProgramPreprocessing, ProgramMetadata};
use jolt_transcript::DigestHasher;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use sha3::Digest;
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
/// whenever the digest input or instruction/lookup protocol changes: it is
/// the compatibility switch a deployed verifier sees.
#[cfg(all(not(feature = "akita"), not(feature = "field-inline")))]
const PROGRAM_PREPROCESSING_DIGEST_DOMAIN: &[u8] =
    b"jolt/program-preprocessing/dory-whole-bytecode/v3";
// Field flags use the common circuit columns; preprocessing has no side table.
#[cfg(all(not(feature = "akita"), feature = "field-inline"))]
const PROGRAM_PREPROCESSING_DIGEST_DOMAIN: &[u8] =
    b"jolt/program-preprocessing/dory-whole-bytecode/field-inline/v3";

#[cfg(all(feature = "akita", not(feature = "field-inline")))]
const PROGRAM_PREPROCESSING_DIGEST_DOMAIN: &[u8] =
    b"jolt/program-preprocessing/akita-whole-bytecode/v2";
#[cfg(all(feature = "akita", feature = "field-inline"))]
const PROGRAM_PREPROCESSING_DIGEST_DOMAIN: &[u8] =
    b"jolt/program-preprocessing/akita-whole-bytecode/field-inline/v2";

impl<PCS: CommitmentScheme> ProgramPreprocessing<PCS> {
    /// The 32-byte program binding absorbed first into the Fiat-Shamir
    /// preamble: the selected digest hash ([`jolt_transcript::DIGEST_HASH`],
    /// SHA3-256 by default) over the domain tag and this value's bincode
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
            DigestHasher::new_with_prefix(PROGRAM_PREPROCESSING_DIGEST_DOMAIN)
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
///
/// Decoding a [`JoltVerifierPreprocessing`] recomputes its digest from the
/// program. A verifier whose trust root fixes the digest together with the
/// program, such as a recursion guest whose image embeds both, may decode this
/// form instead and assemble the preprocessing from its fields and that digest.
#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "VC::Setup: Serialize",
    deserialize = "VC::Setup: DeserializeOwned"
))]
pub struct VerifierPreprocessingWire<PCS, VC>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    pub program: ProgramPreprocessing<PCS>,
    pub pcs_setup: PCS::VerifierSetup,
    pub vc_setup: Option<VC::Setup>,
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
        23, 122, 87, 153, 110, 232, 141, 118, 12, 218, 209, 189, 206, 198, 99, 177, 87, 60, 161,
        30, 70, 226, 249, 122, 184, 166, 153, 230, 177, 3, 18, 249,
    ];
    #[cfg(all(not(feature = "akita"), feature = "field-inline"))]
    const FULL_PROGRAM_DIGEST: [u8; 32] = [
        251, 31, 23, 216, 138, 216, 194, 197, 138, 63, 68, 72, 177, 150, 8, 96, 155, 175, 220, 235,
        199, 3, 85, 26, 85, 117, 40, 26, 71, 65, 222, 135,
    ];
    #[cfg(all(feature = "akita", not(feature = "field-inline")))]
    const FULL_PROGRAM_DIGEST: [u8; 32] = [
        28, 231, 178, 202, 141, 29, 196, 251, 78, 7, 162, 60, 46, 243, 226, 160, 50, 84, 1, 192,
        143, 8, 12, 118, 7, 205, 36, 177, 52, 85, 138, 8,
    ];
    #[cfg(all(feature = "akita", feature = "field-inline"))]
    const FULL_PROGRAM_DIGEST: [u8; 32] = [
        8, 185, 214, 73, 203, 201, 101, 109, 121, 105, 231, 113, 161, 93, 74, 171, 164, 36, 61,
        233, 121, 23, 81, 23, 133, 204, 39, 125, 20, 157, 76, 42,
    ];
    #[cfg(all(not(feature = "akita"), not(feature = "field-inline")))]
    const COMMITTED_PROGRAM_DIGEST: [u8; 32] = [
        79, 75, 180, 174, 130, 233, 185, 79, 182, 42, 65, 141, 227, 27, 135, 89, 140, 122, 243,
        165, 84, 234, 206, 248, 236, 84, 202, 159, 10, 237, 40, 125,
    ];
    #[cfg(all(feature = "akita", not(feature = "field-inline")))]
    const COMMITTED_PROGRAM_DIGEST: [u8; 32] = [
        215, 127, 152, 112, 39, 107, 5, 1, 132, 113, 171, 133, 192, 178, 41, 123, 43, 74, 163, 1,
        197, 208, 184, 128, 142, 227, 10, 217, 77, 175, 23, 52,
    ];
    #[cfg(all(not(feature = "akita"), feature = "field-inline"))]
    const COMMITTED_PROGRAM_DIGEST: [u8; 32] = [
        16, 30, 78, 36, 240, 215, 255, 160, 237, 23, 44, 87, 67, 108, 161, 128, 221, 184, 99, 229,
        202, 168, 190, 252, 215, 72, 183, 183, 1, 198, 225, 98,
    ];
    #[cfg(all(feature = "akita", feature = "field-inline"))]
    const COMMITTED_PROGRAM_DIGEST: [u8; 32] = [
        102, 69, 211, 57, 187, 114, 186, 63, 245, 61, 48, 221, 105, 189, 176, 17, 233, 138, 227,
        128, 40, 98, 60, 167, 24, 55, 210, 90, 200, 82, 168, 243,
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
