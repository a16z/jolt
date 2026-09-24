use jolt_claims::protocols::jolt::TracePolynomialOrder;
use jolt_crypto::VectorCommitment;
use jolt_openings::CommitmentScheme;
use jolt_program::preprocess::JoltProgramPreprocessing;
use jolt_verifier::JoltVerifierPreprocessing;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::PreprocessingError;

/// The full program preprocessing shared by prover and verifier construction.
/// Serializes as the program itself; the Fiat-Shamir digest is derived by
/// [`JoltVerifierPreprocessing`] from whichever program view it holds.
#[derive(Clone, Serialize, Deserialize)]
#[serde(
    into = "Arc<JoltProgramPreprocessing>",
    try_from = "Arc<JoltProgramPreprocessing>"
)]
pub struct JoltSharedPreprocessing {
    pub program: Arc<JoltProgramPreprocessing>,
}

impl JoltSharedPreprocessing {
    pub fn new(program: JoltProgramPreprocessing) -> Result<Self, PreprocessingError> {
        Self::try_from(Arc::new(program))
    }
}

impl TryFrom<Arc<JoltProgramPreprocessing>> for JoltSharedPreprocessing {
    type Error = PreprocessingError;

    // Fails the SDK's fallible `preprocess_shared_*` entry point early; the
    // verifier re-checks at stage 6 via `entry_bytecode_index_checked`.
    fn try_from(program: Arc<JoltProgramPreprocessing>) -> Result<Self, Self::Error> {
        if program.metadata().is_none() {
            return Err(PreprocessingError::InvalidProgram {
                reason: "entry address is absent from bytecode preprocessing".to_owned(),
            });
        }
        Ok(Self { program })
    }
}

impl From<JoltSharedPreprocessing> for Arc<JoltProgramPreprocessing> {
    fn from(shared: JoltSharedPreprocessing) -> Self {
        shared.program
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use common::jolt_device::MemoryLayout;
    use jolt_program::preprocess::JoltProgramPreprocessing;
    use jolt_riscv::RV64IMAC_JOLT;

    use super::JoltSharedPreprocessing;
    use crate::PreprocessingError;

    #[test]
    fn shared_preprocessing_serializes_as_its_program() {
        let program = JoltProgramPreprocessing::new(
            Vec::new(),
            Vec::new(),
            MemoryLayout::default(),
            0,
            1 << 12,
            RV64IMAC_JOLT,
        )
        .unwrap();
        let shared = JoltSharedPreprocessing::new(program.clone()).unwrap();

        let encoded = bincode::serde::encode_to_vec(&shared, bincode::config::standard()).unwrap();
        assert_eq!(
            encoded,
            bincode::serde::encode_to_vec(&program, bincode::config::standard()).unwrap()
        );
        let (decoded, consumed): (JoltSharedPreprocessing, usize) =
            bincode::serde::decode_from_slice(&encoded, bincode::config::standard()).unwrap();
        assert_eq!(consumed, encoded.len());
        assert_eq!(*decoded.program, program);
    }

    #[test]
    fn shared_preprocessing_rejects_an_entry_address_without_a_bytecode_row() {
        let program = JoltProgramPreprocessing::new(
            Vec::new(),
            Vec::new(),
            MemoryLayout::default(),
            0x8000_0000,
            1 << 12,
            RV64IMAC_JOLT,
        )
        .unwrap();
        assert!(matches!(
            JoltSharedPreprocessing::new(program.clone()),
            Err(PreprocessingError::InvalidProgram { .. })
        ));
        let encoded = bincode::serde::encode_to_vec(&program, bincode::config::standard()).unwrap();
        assert!(
            bincode::serde::decode_from_slice::<JoltSharedPreprocessing, _>(
                &encoded,
                bincode::config::standard()
            )
            .is_err()
        );
    }
}

#[cfg(feature = "akita")]
use crate::akita::witness::DirectProgramObjects;

/// The prover-retained committed-program data: the verifier's preprocessing
/// carries only the program COMMITMENTS in committed mode, but the prover
/// still needs the full program (witness generation, the bytecode stage-value
/// folds, the reduction chunk grids, the stage-8 materialization) and the
/// commitments' opening material (the stage-8 openings). Mirrors legacy's
/// `CommittedProgramProverData`.
///
/// On the packed (`akita`) build the per-chunk/image plans and hints are
/// retained in direct bounded-dense program objects built at preprocessing
/// time, so proving consumes them directly instead of re-deriving them.
#[derive(Clone)]
#[cfg_attr(
    not(feature = "akita"),
    derive(Serialize, Deserialize),
    serde(bound(
        serialize = "PCS::OpeningHint: Serialize",
        deserialize = "PCS::OpeningHint: serde::de::DeserializeOwned"
    ))
)]
pub struct CommittedProgramProverData<PCS: CommitmentScheme> {
    pub full: Arc<JoltProgramPreprocessing>,
    /// One opening hint per committed bytecode chunk, in chunk order.
    #[cfg(not(feature = "akita"))]
    pub bytecode_chunk_hints: Vec<PCS::OpeningHint>,
    #[cfg(not(feature = "akita"))]
    pub program_image_hint: PCS::OpeningHint,
    /// Direct program objects in canonical order (bytecode chunks, then
    /// program image); their commitments must match the verifier
    /// preprocessing's `direct_program_commitments` (stage 0 checks
    /// fail-closed).
    #[cfg(feature = "akita")]
    pub direct_program: DirectProgramObjects<PCS>,
    /// The trace order the chunk commitments' coefficient grids were built
    /// under at preprocessing time. Stage 0 rejects a proof config whose
    /// order disagrees because the reduction point would address the
    /// committed grid in the wrong order.
    pub trace_order: TracePolynomialOrder,
}

/// The prover's preprocessing is a strict superset of the verifier's: the
/// embedded [`JoltVerifierPreprocessing`] carries the program view, the
/// Fiat-Shamir digest it derives from that view (recomputed whenever it is
/// deserialized), the PCS verifier setup, and the ZK vector-commitment
/// setup; the prover adds its PCS prover setup
/// and, in committed-program mode, the retained full program and opening
/// hints. Witness generation reads the full program through
/// [`program`](Self::program).
///
/// Dory preprocessing is serializable. Akita preprocessing retains backend
/// setup and witness objects and must be regenerated in the proving process.
#[derive(Clone)]
#[cfg_attr(
    not(feature = "akita"),
    derive(Serialize, Deserialize),
    serde(bound(
        serialize = "JoltVerifierPreprocessing<PCS, VC>: Serialize, PCS::ProverSetup: Serialize, CommittedProgramProverData<PCS>: Serialize",
        deserialize = "JoltVerifierPreprocessing<PCS, VC>: serde::de::DeserializeOwned, PCS::ProverSetup: serde::de::DeserializeOwned, CommittedProgramProverData<PCS>: serde::de::DeserializeOwned"
    ))
)]
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
