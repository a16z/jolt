use ark_serialize::{CanonicalSerialize, SerializationError};
use common::jolt_device::MemoryLayout;
#[cfg(not(feature = "akita"))]
use jolt_claims::protocols::jolt::TracePolynomialOrder;
use jolt_crypto::VectorCommitment;
use jolt_openings::CommitmentScheme;
use jolt_program::preprocess::{JoltProgramPreprocessing, ProgramMetadata};
use jolt_verifier::JoltVerifierPreprocessing;
use std::sync::Arc;

use crate::PreprocessingError;

#[derive(Clone)]
pub struct JoltSharedPreprocessing {
    pub program: Arc<JoltProgramPreprocessing>,
    pub preprocessing_digest: [u8; 32],
}

impl JoltSharedPreprocessing {
    pub fn new(program: JoltProgramPreprocessing) -> Result<Self, PreprocessingError> {
        let preprocessing_digest = full_preprocessing_digest(&program)?;
        Ok(Self {
            program: Arc::new(program),
            preprocessing_digest,
        })
    }
}

/// Reproduces the canonical encoding used by `JoltSharedPreprocessing` before
/// the host pipeline moved out of the legacy prover. This keeps the default
/// program-bound Fiat-Shamir preamble unchanged across the migration.
pub(crate) fn full_preprocessing_digest(
    program: &JoltProgramPreprocessing,
) -> Result<[u8; 32], PreprocessingError> {
    const DEFAULT_BYTECODE_CHUNK_COUNT: usize = 1;

    let metadata =
        program
            .metadata()
            .ok_or_else(|| PreprocessingError::InvalidCommittedProgram {
                reason: "entry address is absent from bytecode preprocessing".to_owned(),
            })?;
    canonical_preprocessing_digest(|encoded| {
        FULL_PROGRAM_TAG.serialize_compressed(&mut *encoded)?;
        program.bytecode.serialize_compressed(&mut *encoded)?;
        program.ram.serialize_compressed(&mut *encoded)?;
        encode_shared_preprocessing_tail(
            &metadata,
            &program.memory_layout,
            program.max_padded_trace_length,
            DEFAULT_BYTECODE_CHUNK_COUNT,
            encoded,
        )
    })
}

pub(crate) const FULL_PROGRAM_TAG: u8 = 0;
pub(crate) const COMMITTED_PROGRAM_TAG: u8 = 1;

pub(crate) fn canonical_preprocessing_digest(
    encode: impl FnOnce(&mut Vec<u8>) -> Result<(), SerializationError>,
) -> Result<[u8; 32], PreprocessingError> {
    let mut encoded = Vec::new();
    encode(&mut encoded).map_err(|error| PreprocessingError::Encoding {
        reason: error.to_string(),
    })?;
    Ok(blake2b_256(&encoded))
}

pub(crate) fn encode_program_metadata(
    metadata: &ProgramMetadata,
    encoded: &mut Vec<u8>,
) -> Result<(), SerializationError> {
    metadata.entry_address.serialize_compressed(&mut *encoded)?;
    metadata
        .min_bytecode_address
        .serialize_compressed(&mut *encoded)?;
    metadata
        .entry_bytecode_index
        .serialize_compressed(&mut *encoded)?;
    metadata
        .program_image_len_words
        .serialize_compressed(&mut *encoded)?;
    metadata.bytecode_len.serialize_compressed(&mut *encoded)
}

pub(crate) fn encode_shared_preprocessing_tail(
    metadata: &ProgramMetadata,
    memory_layout: &MemoryLayout,
    max_padded_trace_length: usize,
    bytecode_chunk_count: usize,
    encoded: &mut Vec<u8>,
) -> Result<(), SerializationError> {
    encode_program_metadata(metadata, encoded)?;
    memory_layout.serialize_compressed(&mut *encoded)?;
    max_padded_trace_length.serialize_compressed(&mut *encoded)?;
    bytecode_chunk_count.serialize_compressed(&mut *encoded)
}

fn blake2b_256(encoded: &[u8]) -> [u8; 32] {
    use blake2::{digest::consts::U32, Blake2b, Digest};

    Blake2b::<U32>::digest(encoded).into()
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use common::jolt_device::MemoryLayout;
    use jolt_program::preprocess::JoltProgramPreprocessing;
    use jolt_riscv::RV64IMAC_JOLT;

    use super::full_preprocessing_digest;

    #[test]
    fn full_preprocessing_digest_is_stable() {
        let program = JoltProgramPreprocessing::new(
            Vec::new(),
            Vec::new(),
            MemoryLayout::default(),
            0,
            1 << 12,
            RV64IMAC_JOLT,
        )
        .unwrap();

        assert_eq!(
            full_preprocessing_digest(&program).unwrap(),
            [
                145, 121, 88, 160, 204, 106, 98, 59, 169, 97, 216, 209, 50, 77, 116, 101, 132, 24,
                249, 19, 196, 162, 146, 128, 195, 60, 253, 174, 10, 250, 92, 68,
            ]
        );
    }
}

/// The prover-retained committed-program data: the verifier's preprocessing
/// carries only the program COMMITMENTS in committed mode, but the prover
/// still needs the full program (witness generation, the bytecode stage-value
/// folds, the reduction chunk grids, the stage-8 materialization) and the
/// commitments' opening material (the stage-8 openings). Mirrors legacy's
/// `CommittedProgramProverData`.
///
/// On the packed (`akita`) build the per-chunk/image hints are replaced by
/// the precommitted `ProgramOneHot` objects themselves — witnesses, plans,
/// setups, and hints — built once at preprocessing time
/// ([`crate::akita::witness::commit_program_one_hot`]) so proving consumes
/// them directly instead of re-deriving them per proof.
#[derive(Clone)]
pub struct CommittedProgramProverData<PCS: CommitmentScheme> {
    pub full: Arc<JoltProgramPreprocessing>,
    /// One opening hint per committed bytecode chunk, in chunk order.
    #[cfg(not(feature = "akita"))]
    pub bytecode_chunk_hints: Vec<PCS::OpeningHint>,
    #[cfg(not(feature = "akita"))]
    pub program_image_hint: PCS::OpeningHint,
    /// The precommitted `ProgramOneHot` objects in canonical order (bytecode,
    /// then program image); their commitments must match the verifier
    /// preprocessing's `program_one_hot_commitments` (stage 0 checks
    /// fail-closed).
    #[cfg(feature = "akita")]
    pub program_one_hot: crate::akita::witness::ProgramOneHot<PCS>,
    /// The trace order the chunk commitments' coefficient grids were built
    /// under at preprocessing time (legacy couples the two through one
    /// process-global layout). Stage 0 rejects a proof config whose order
    /// disagrees — the chunk tables stages 6b/8 rebuild would transpose
    /// against the absorbed commitments and fail only at verification.
    #[cfg(not(feature = "akita"))]
    pub trace_order: TracePolynomialOrder,
}

/// The prover's preprocessing is a strict superset of the verifier's: the
/// embedded [`JoltVerifierPreprocessing`] carries the program view, the
/// preprocessing digest (an opaque input — its computation is a
/// preprocessing-time policy, never recomputed here), the PCS verifier setup,
/// and the ZK vector-commitment setup; the prover adds its PCS prover setup
/// and, in committed-program mode, the retained full program and opening
/// hints. Witness generation reads the full program through
/// [`program`](Self::program).
#[derive(Clone)]
pub struct JoltProverPreprocessing<PCS, VC>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    pub verifier: JoltVerifierPreprocessing<PCS, VC>,
    pub pcs_setup: PCS::ProverSetup,
    /// Present exactly when the verifier preprocessing is committed-program.
    pub committed_program: Option<CommittedProgramProverData<PCS>>,
}

impl<PCS, VC> JoltProverPreprocessing<PCS, VC>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    /// The full program preprocessing witness generation and the bytecode
    /// folds consume: the verifier's own full view, or the prover-retained
    /// copy in committed-program mode.
    pub fn program(&self) -> Option<&JoltProgramPreprocessing> {
        self.verifier.program.as_full().or_else(|| {
            self.committed_program
                .as_ref()
                .map(|data| data.full.as_ref())
        })
    }

    pub fn program_arc(&self) -> Option<Arc<JoltProgramPreprocessing>> {
        self.verifier.program.as_full_arc().or_else(|| {
            self.committed_program
                .as_ref()
                .map(|data| Arc::clone(&data.full))
        })
    }
}
