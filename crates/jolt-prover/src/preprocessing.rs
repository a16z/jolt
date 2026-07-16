use jolt_claims::protocols::jolt::TracePolynomialOrder;
use jolt_crypto::VectorCommitment;
use jolt_openings::CommitmentScheme;
use jolt_program::preprocess::JoltProgramPreprocessing;
use jolt_verifier::JoltVerifierPreprocessing;

/// The prover-retained committed-program data: the verifier's preprocessing
/// carries only the chunk/image COMMITMENTS in committed mode, but the prover
/// still needs the full program (witness generation, the bytecode stage-value
/// folds, the reduction chunk grids, the stage-8 materialization) and the
/// commitments' opening hints (the stage-8 joint opening). Mirrors legacy's
/// `CommittedProgramProverData`.
#[derive(Clone)]
pub struct CommittedProgramProverData<PCS: CommitmentScheme> {
    pub full: JoltProgramPreprocessing,
    /// One opening hint per committed bytecode chunk, in chunk order.
    pub bytecode_chunk_hints: Vec<PCS::OpeningHint>,
    pub program_image_hint: PCS::OpeningHint,
    /// The trace order the chunk commitments' coefficient grids were built
    /// under at preprocessing time (legacy couples the two through one
    /// process-global layout). Stage 0 rejects a proof config whose order
    /// disagrees — the chunk tables stages 6b/8 rebuild would transpose
    /// against the absorbed commitments and fail only at verification.
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
        self.verifier
            .program
            .as_full()
            .or_else(|| self.committed_program.as_ref().map(|data| &data.full))
    }
}
