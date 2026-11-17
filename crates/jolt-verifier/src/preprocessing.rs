use common::jolt_device::MemoryLayout;
use jolt_claims::protocols::jolt::{JoltRelationId, TracePolynomialOrder};
use jolt_crypto::VectorCommitment;
use jolt_openings::CommitmentScheme;
use jolt_program::preprocess::{compute_max_ram_k, JoltProgramPreprocessing, ProgramMetadata};
use jolt_transcript::DigestHasher;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use sha3::Digest;
use std::{ops::Range, sync::Arc};

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

    /// Require the entry to name a real instruction row. Bytecode read-RAF's
    /// entry term pins cycle 0 to this row; were it a no-op, an all-padding
    /// trace would satisfy every constraint under either panic claim. A
    /// committed key carries no rows, so here this rejects only the padding
    /// row at index 0; that its entry index names a real row is trusted
    /// preprocessing, like the committed rows themselves.
    pub fn validate_entry_row(&self) -> Result<(), VerifierError> {
        let names_instruction = match self {
            Self::Full(full) => full.bytecode.entry_instruction().is_some(),
            Self::Committed(committed) => committed.meta.entry_bytecode_index != 0,
        };
        if names_instruction {
            Ok(())
        } else {
            Err(VerifierError::EntryNotInstruction {
                entry_address: self.entry_address(),
            })
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

    /// The code region: derived from the bytecode in full mode, and taken from
    /// the committed key's metadata, after its admission check, otherwise.
    pub fn code_region(&self) -> Result<Range<usize>, VerifierError> {
        match self {
            Self::Full(full) => full.code_region(),
            Self::Committed(committed) => committed
                .meta
                .validate_code_region(&committed.memory_layout)
                .map(|()| committed.meta.code_region.clone()),
        }
        .map_err(|error| VerifierError::InvalidMemoryLayout {
            reason: error.to_string(),
        })
    }

    pub fn validate_code_region(&self) -> Result<(), VerifierError> {
        self.code_region().map(|_| ())
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
    b"jolt/program-preprocessing/dory-whole-bytecode/v4";
// Field flags use the common circuit columns; preprocessing has no side table.
#[cfg(all(not(feature = "akita"), feature = "field-inline"))]
const PROGRAM_PREPROCESSING_DIGEST_DOMAIN: &[u8] =
    b"jolt/program-preprocessing/dory-whole-bytecode/field-inline/v4";

#[cfg(all(feature = "akita", not(feature = "field-inline")))]
const PROGRAM_PREPROCESSING_DIGEST_DOMAIN: &[u8] =
    b"jolt/program-preprocessing/akita-whole-bytecode/v3";
#[cfg(all(feature = "akita", feature = "field-inline"))]
const PROGRAM_PREPROCESSING_DIGEST_DOMAIN: &[u8] =
    b"jolt/program-preprocessing/akita-whole-bytecode/field-inline/v3";

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

fn validate_memory_layout_admission(memory_layout: &MemoryLayout) -> Result<(), VerifierError> {
    compute_max_ram_k(memory_layout)
        .map(|_| ())
        .map_err(|error| VerifierError::InvalidMemoryLayout {
            reason: error.to_string(),
        })
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
        validate_memory_layout_admission(program.memory_layout())?;
        program.validate_code_region()?;
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

    use super::{
        CommittedProgramPreprocessing, JoltVerifierPreprocessing, ProgramPreprocessing,
        VerifierPreprocessingWire,
    };
    use crate::VerifierError;
    use jolt_crypto::{Bn254G1, Pedersen};
    use jolt_dory::{DoryCommitment, DoryScheme};
    use jolt_openings::CommitmentScheme;

    /// Golden digests. A change here is a Fiat-Shamir break for every
    /// deployed verifier: bump `PROGRAM_PREPROCESSING_DIGEST_DOMAIN` and say so
    /// in the PR. Pinned separately with and without `jolt-program/field-inline`:
    /// the bytecode row schema is shared, while field-inline selects its own
    /// digest domain for the extended instruction and proof profile.
    #[cfg(all(not(feature = "akita"), not(feature = "field-inline")))]
    const FULL_PROGRAM_DIGEST: [u8; 32] = [
        245, 210, 146, 17, 174, 143, 17, 189, 163, 125, 103, 73, 93, 105, 81, 129, 84, 129, 32,
        238, 22, 151, 123, 185, 119, 157, 85, 67, 254, 97, 224, 197,
    ];
    #[cfg(all(not(feature = "akita"), feature = "field-inline"))]
    const FULL_PROGRAM_DIGEST: [u8; 32] = [
        4, 191, 79, 187, 254, 201, 236, 63, 73, 224, 17, 12, 249, 216, 121, 160, 112, 11, 59, 101,
        36, 197, 48, 191, 180, 171, 69, 163, 120, 4, 143, 221,
    ];
    #[cfg(all(feature = "akita", not(feature = "field-inline")))]
    const FULL_PROGRAM_DIGEST: [u8; 32] = [
        33, 19, 4, 91, 109, 182, 112, 145, 156, 167, 132, 177, 188, 48, 249, 211, 88, 248, 42, 83,
        243, 239, 218, 93, 173, 80, 149, 45, 232, 211, 73, 154,
    ];
    #[cfg(all(feature = "akita", feature = "field-inline"))]
    const FULL_PROGRAM_DIGEST: [u8; 32] = [
        34, 166, 244, 254, 92, 130, 120, 139, 214, 165, 211, 190, 161, 95, 202, 64, 248, 34, 108,
        159, 136, 121, 55, 146, 235, 114, 227, 128, 252, 0, 161, 72,
    ];
    #[cfg(all(not(feature = "akita"), not(feature = "field-inline")))]
    const COMMITTED_PROGRAM_DIGEST: [u8; 32] = [
        154, 87, 137, 235, 121, 218, 34, 215, 186, 177, 56, 39, 73, 64, 42, 152, 207, 30, 145, 97,
        48, 130, 215, 45, 159, 89, 103, 112, 51, 46, 113, 85,
    ];
    #[cfg(all(feature = "akita", not(feature = "field-inline")))]
    const COMMITTED_PROGRAM_DIGEST: [u8; 32] = [
        46, 141, 48, 70, 223, 15, 119, 143, 106, 236, 197, 204, 209, 22, 24, 183, 105, 153, 184,
        249, 215, 124, 225, 168, 75, 6, 224, 86, 184, 131, 212, 20,
    ];
    #[cfg(all(not(feature = "akita"), feature = "field-inline"))]
    const COMMITTED_PROGRAM_DIGEST: [u8; 32] = [
        202, 93, 152, 26, 214, 171, 217, 61, 112, 189, 249, 157, 205, 171, 254, 89, 154, 119, 106,
        93, 154, 233, 6, 69, 144, 107, 143, 177, 242, 10, 205, 231,
    ];
    #[cfg(all(feature = "akita", feature = "field-inline"))]
    const COMMITTED_PROGRAM_DIGEST: [u8; 32] = [
        6, 49, 50, 184, 126, 170, 253, 87, 126, 7, 82, 39, 119, 93, 42, 170, 9, 7, 74, 246, 169,
        88, 162, 133, 45, 230, 71, 229, 165, 28, 43, 115,
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
    fn verifier_key_admission_rejects_invalid_full_and_committed_layouts() {
        type Key = JoltVerifierPreprocessing<DoryScheme, Pedersen<Bn254G1>>;
        let (_, pcs_setup) = DoryScheme::setup(1).unwrap();
        let full = program();
        let committed = CommittedProgramPreprocessing {
            meta: full.metadata().unwrap(),
            memory_layout: full.memory_layout.clone(),
            max_padded_trace_length: full.max_padded_trace_length,
            program_image_commitment: DoryCommitment::default(),
            bytecode_commitment: DoryCommitment::default(),
            trace_order: TracePolynomialOrder::CycleMajor,
        };
        for valid in [
            ProgramPreprocessing::Full(Arc::new(full)),
            ProgramPreprocessing::Committed(committed),
        ] {
            let key = Key::new(valid.clone(), pcs_setup.clone(), None).unwrap();
            let encoded = bincode::serde::encode_to_vec(&key, bincode::config::standard()).unwrap();
            assert!(bincode::serde::decode_from_slice::<Key, _>(
                &encoded,
                bincode::config::standard()
            )
            .is_ok());

            for invalid_heap_end in [
                u64::MAX - 6,
                (1u64 << 63) + valid.memory_layout().get_lowest_address() + 8,
                valid.memory_layout().heap_end + 4,
            ] {
                let mut invalid = valid.clone();
                let layout = match &mut invalid {
                    ProgramPreprocessing::Full(full) => &mut Arc::make_mut(full).memory_layout,
                    ProgramPreprocessing::Committed(committed) => &mut committed.memory_layout,
                };
                layout.heap_end = invalid_heap_end;
                assert!(matches!(
                    Key::new(invalid.clone(), pcs_setup.clone(), None),
                    Err(VerifierError::InvalidMemoryLayout { .. })
                ));
                let wire = VerifierPreprocessingWire::<DoryScheme, Pedersen<Bn254G1>> {
                    program: invalid,
                    pcs_setup: pcs_setup.clone(),
                    vc_setup: None,
                };
                let encoded =
                    bincode::serde::encode_to_vec(&wire, bincode::config::standard()).unwrap();
                assert!(bincode::serde::decode_from_slice::<Key, _>(
                    &encoded,
                    bincode::config::standard()
                )
                .is_err());
            }
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
    fn digest_and_protection_mask_bind_layout_in_both_program_modes() {
        use jolt_program::preprocess::ram::write_protected_word_intervals;

        for original in [
            ProgramPreprocessing::<Pcs>::Full(Arc::new(program())),
            ProgramPreprocessing::<Pcs>::Committed(committed()),
        ] {
            let mut changed = original.clone();
            let layout = match &mut changed {
                ProgramPreprocessing::Full(full) => &mut Arc::make_mut(full).memory_layout,
                ProgramPreprocessing::Committed(committed) => &mut committed.memory_layout,
            };
            layout.stack_end += 16;
            let mask = |program: &ProgramPreprocessing<Pcs>| {
                write_protected_word_intervals(
                    program.code_region().unwrap(),
                    program.memory_layout(),
                    program.memory_layout().max_ram_k().unwrap(),
                )
                .unwrap()
            };
            assert_ne!(mask(&original), mask(&changed));
            assert_ne!(original.digest().unwrap(), changed.digest().unwrap());
        }
    }

    #[test]
    fn committed_program_digest_is_stable() {
        let digest = ProgramPreprocessing::<Pcs>::Committed(committed())
            .digest()
            .unwrap();
        assert_eq!(digest, COMMITTED_PROGRAM_DIGEST);
    }
}
