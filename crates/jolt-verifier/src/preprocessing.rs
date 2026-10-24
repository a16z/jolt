use blake2::{digest::consts::U32, Blake2b, Digest};
use common::jolt_device::MemoryLayout;
use jolt_claims::protocols::jolt::geometry::claim_reductions::bytecode;
use jolt_claims::protocols::jolt::{JoltRelationId, TracePolynomialOrder};
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
    /// Lanes in each committed bytecode row: `bytecode::total_lanes()` of the
    /// build that committed it. The commitment pads them to
    /// `COMMITTED_BYTECODE_LANE_CAPACITY`, so without this a build with another
    /// `zicclsm` setting, whose rows carry two more or fewer lookup-table
    /// lanes, would commit an artifact of the same shape.
    pub bytecode_lanes: usize,
    pub trace_order: TracePolynomialOrder,
}

impl<PCS: CommitmentScheme> CommittedProgramPreprocessing<PCS> {
    /// Rejects an artifact whose recorded lane count differs from this
    /// build's. Every `field-inline` × `zicclsm` combination has its own count;
    /// a layout change that kept the count would pass.
    pub(crate) fn check_bytecode_lanes(&self) -> Result<(), VerifierError> {
        if self.bytecode_lanes != bytecode::total_lanes() {
            return Err(VerifierError::InvalidCommittedProgram {
                reason: format!(
                    "committed bytecode rows have {} lanes; this build reads {}",
                    self.bytecode_lanes,
                    bytecode::total_lanes()
                ),
            });
        }
        Ok(())
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
/// deployed verifier sees. The extended instruction profiles (`field-inline`,
/// `zicclsm`) each add a tag.
#[cfg(all(
    not(feature = "akita"),
    not(feature = "field-inline"),
    not(feature = "zicclsm")
))]
const PROGRAM_PREPROCESSING_DIGEST_DOMAIN: &[u8] =
    b"jolt/program-preprocessing/dory-whole-bytecode/v3";
#[cfg(all(
    not(feature = "akita"),
    not(feature = "field-inline"),
    feature = "zicclsm"
))]
const PROGRAM_PREPROCESSING_DIGEST_DOMAIN: &[u8] =
    b"jolt/program-preprocessing/dory-whole-bytecode/zicclsm/v3";
// Field flags use the common circuit columns; preprocessing has no side table.
#[cfg(all(
    not(feature = "akita"),
    feature = "field-inline",
    not(feature = "zicclsm")
))]
const PROGRAM_PREPROCESSING_DIGEST_DOMAIN: &[u8] =
    b"jolt/program-preprocessing/dory-whole-bytecode/field-inline/v3";
#[cfg(all(not(feature = "akita"), feature = "field-inline", feature = "zicclsm"))]
const PROGRAM_PREPROCESSING_DIGEST_DOMAIN: &[u8] =
    b"jolt/program-preprocessing/dory-whole-bytecode/field-inline/zicclsm/v3";

#[cfg(all(
    feature = "akita",
    not(feature = "field-inline"),
    not(feature = "zicclsm")
))]
const PROGRAM_PREPROCESSING_DIGEST_DOMAIN: &[u8] =
    b"jolt/program-preprocessing/akita-whole-bytecode/v2";
#[cfg(all(feature = "akita", not(feature = "field-inline"), feature = "zicclsm"))]
const PROGRAM_PREPROCESSING_DIGEST_DOMAIN: &[u8] =
    b"jolt/program-preprocessing/akita-whole-bytecode/zicclsm/v2";
#[cfg(all(feature = "akita", feature = "field-inline", not(feature = "zicclsm")))]
const PROGRAM_PREPROCESSING_DIGEST_DOMAIN: &[u8] =
    b"jolt/program-preprocessing/akita-whole-bytecode/field-inline/v2";
#[cfg(all(feature = "akita", feature = "field-inline", feature = "zicclsm"))]
const PROGRAM_PREPROCESSING_DIGEST_DOMAIN: &[u8] =
    b"jolt/program-preprocessing/akita-whole-bytecode/field-inline/zicclsm/v2";

impl<PCS: CommitmentScheme> ProgramPreprocessing<PCS> {
    /// The 32-byte program binding absorbed first into the Fiat-Shamir
    /// preamble: Blake2b-256 over the domain tag and this value's bincode
    /// encoding. Hashing the whole type binds every field the verifier trusts
    /// (mode, bytecode or its commitments, RAM image, memory layout, trace
    /// bound) without a hand-maintained field list, so a field added to any
    /// preprocessing type enters the digest by construction. A prover and a
    /// separately built verifier must agree on `field-inline` and `zicclsm`
    /// (they select the domain tag) and on the PCS (`Committed` carries
    /// PCS-specific fields).
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
    use jolt_claims::protocols::jolt::TracePolynomialOrder;
    #[cfg(not(feature = "akita"))]
    use jolt_dory::{DoryCommitment as Commitment, DoryScheme as Pcs};
    use jolt_program::preprocess::JoltProgramPreprocessing;
    use jolt_riscv::RV64IMAC_JOLT;

    use super::{bytecode, CommittedProgramPreprocessing, ProgramPreprocessing};

    /// Golden digests. A change here is a Fiat-Shamir break for every
    /// deployed verifier: bump `PROGRAM_PREPROCESSING_DIGEST_DOMAIN` and say so
    /// in the PR. Pinned separately per `field-inline` and `zicclsm`: the
    /// bytecode row schema is shared, while each selects its own digest domain
    /// for the extended instruction and proof profile.
    #[cfg(all(
        not(feature = "akita"),
        not(feature = "field-inline"),
        not(feature = "zicclsm")
    ))]
    const FULL_PROGRAM_DIGEST: [u8; 32] = [
        210, 191, 0, 84, 62, 128, 103, 70, 77, 151, 55, 117, 243, 19, 62, 125, 251, 197, 144, 253,
        15, 247, 188, 75, 75, 195, 220, 212, 192, 29, 83, 195,
    ];
    #[cfg(all(
        not(feature = "akita"),
        not(feature = "field-inline"),
        feature = "zicclsm"
    ))]
    const FULL_PROGRAM_DIGEST: [u8; 32] = [
        82, 219, 149, 37, 129, 182, 240, 85, 146, 153, 255, 58, 123, 143, 184, 0, 142, 190, 240,
        83, 146, 111, 22, 152, 67, 114, 83, 44, 48, 177, 235, 68,
    ];
    #[cfg(all(
        not(feature = "akita"),
        feature = "field-inline",
        not(feature = "zicclsm")
    ))]
    const FULL_PROGRAM_DIGEST: [u8; 32] = [
        199, 62, 95, 115, 80, 16, 91, 120, 9, 45, 132, 232, 22, 36, 35, 193, 106, 40, 132, 174, 34,
        116, 228, 50, 138, 147, 85, 45, 223, 179, 78, 189,
    ];
    #[cfg(all(not(feature = "akita"), feature = "field-inline", feature = "zicclsm"))]
    const FULL_PROGRAM_DIGEST: [u8; 32] = [
        210, 157, 226, 127, 150, 35, 197, 82, 117, 254, 36, 172, 154, 152, 131, 104, 122, 83, 97,
        72, 136, 180, 235, 119, 137, 152, 77, 212, 90, 175, 183, 122,
    ];
    #[cfg(all(
        feature = "akita",
        not(feature = "field-inline"),
        not(feature = "zicclsm")
    ))]
    const FULL_PROGRAM_DIGEST: [u8; 32] = [
        105, 5, 115, 202, 3, 231, 207, 114, 177, 20, 72, 32, 241, 244, 126, 223, 135, 229, 84, 99,
        12, 39, 4, 126, 119, 56, 125, 112, 154, 151, 99, 101,
    ];
    #[cfg(all(feature = "akita", not(feature = "field-inline"), feature = "zicclsm"))]
    const FULL_PROGRAM_DIGEST: [u8; 32] = [
        161, 203, 129, 76, 175, 88, 0, 55, 46, 102, 51, 132, 96, 129, 141, 96, 164, 162, 117, 155,
        82, 253, 81, 65, 166, 160, 178, 6, 171, 120, 181, 163,
    ];
    #[cfg(all(feature = "akita", feature = "field-inline", not(feature = "zicclsm")))]
    const FULL_PROGRAM_DIGEST: [u8; 32] = [
        90, 176, 88, 183, 239, 99, 144, 139, 34, 165, 161, 59, 174, 18, 12, 249, 211, 247, 109,
        224, 96, 181, 147, 97, 109, 204, 137, 72, 21, 53, 110, 119,
    ];
    #[cfg(all(feature = "akita", feature = "field-inline", feature = "zicclsm"))]
    const FULL_PROGRAM_DIGEST: [u8; 32] = [
        201, 139, 113, 170, 145, 27, 212, 117, 144, 104, 83, 34, 139, 232, 46, 229, 47, 63, 18, 78,
        111, 23, 91, 37, 151, 82, 132, 95, 204, 127, 103, 244,
    ];
    #[cfg(all(
        not(feature = "akita"),
        not(feature = "field-inline"),
        not(feature = "zicclsm")
    ))]
    const COMMITTED_PROGRAM_DIGEST: [u8; 32] = [
        122, 93, 232, 246, 35, 255, 146, 122, 161, 45, 188, 171, 87, 148, 43, 176, 86, 252, 16,
        206, 86, 55, 152, 139, 31, 169, 131, 111, 199, 223, 4, 171,
    ];
    #[cfg(all(
        not(feature = "akita"),
        not(feature = "field-inline"),
        feature = "zicclsm"
    ))]
    const COMMITTED_PROGRAM_DIGEST: [u8; 32] = [
        104, 180, 8, 135, 251, 193, 145, 120, 116, 209, 63, 44, 30, 127, 148, 0, 116, 156, 45, 12,
        39, 183, 129, 81, 4, 108, 156, 144, 33, 181, 121, 0,
    ];
    #[cfg(all(
        not(feature = "akita"),
        feature = "field-inline",
        not(feature = "zicclsm")
    ))]
    const COMMITTED_PROGRAM_DIGEST: [u8; 32] = [
        154, 239, 41, 59, 3, 219, 83, 222, 150, 175, 249, 18, 121, 158, 82, 140, 177, 217, 245,
        119, 38, 98, 119, 26, 93, 18, 23, 69, 206, 139, 85, 92,
    ];
    #[cfg(all(not(feature = "akita"), feature = "field-inline", feature = "zicclsm"))]
    const COMMITTED_PROGRAM_DIGEST: [u8; 32] = [
        201, 109, 244, 43, 132, 197, 210, 87, 69, 141, 18, 22, 39, 91, 190, 150, 234, 191, 204, 4,
        165, 188, 120, 236, 167, 180, 24, 74, 61, 168, 54, 157,
    ];
    #[cfg(all(
        feature = "akita",
        not(feature = "field-inline"),
        not(feature = "zicclsm")
    ))]
    const COMMITTED_PROGRAM_DIGEST: [u8; 32] = [
        75, 123, 228, 60, 22, 206, 3, 58, 37, 98, 160, 147, 118, 236, 167, 26, 31, 254, 111, 29,
        215, 153, 192, 91, 215, 135, 1, 197, 200, 217, 164, 237,
    ];
    #[cfg(all(feature = "akita", not(feature = "field-inline"), feature = "zicclsm"))]
    const COMMITTED_PROGRAM_DIGEST: [u8; 32] = [
        250, 167, 133, 200, 159, 0, 156, 168, 84, 181, 110, 54, 48, 65, 156, 86, 169, 187, 5, 72,
        126, 114, 206, 63, 106, 255, 94, 90, 216, 177, 77, 190,
    ];
    #[cfg(all(feature = "akita", feature = "field-inline", not(feature = "zicclsm")))]
    const COMMITTED_PROGRAM_DIGEST: [u8; 32] = [
        162, 230, 220, 176, 188, 98, 198, 82, 157, 54, 145, 50, 21, 172, 157, 132, 250, 224, 182,
        53, 254, 129, 201, 181, 44, 252, 79, 105, 186, 60, 174, 47,
    ];
    #[cfg(all(feature = "akita", feature = "field-inline", feature = "zicclsm"))]
    const COMMITTED_PROGRAM_DIGEST: [u8; 32] = [
        74, 49, 122, 235, 29, 23, 89, 114, 174, 235, 13, 153, 158, 54, 235, 191, 121, 246, 213,
        121, 111, 222, 67, 232, 199, 95, 252, 60, 31, 136, 64, 48,
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
            bytecode_lanes: bytecode::total_lanes(),
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
