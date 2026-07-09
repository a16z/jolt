use jolt_crypto::VectorCommitment;
use jolt_openings::CommitmentScheme;
use jolt_program::preprocess::JoltProgramPreprocessing;
use jolt_verifier::JoltVerifierPreprocessing;

/// The prover's preprocessing is a strict superset of the verifier's: the
/// embedded [`JoltVerifierPreprocessing`] carries the program view, the
/// preprocessing digest (an opaque input — its computation is a
/// preprocessing-time policy, never recomputed here), the PCS verifier setup,
/// and the ZK vector-commitment setup; the prover adds only its PCS prover
/// setup. Witness generation reads the full program through
/// [`program`](Self::program).
#[derive(Clone)]
pub struct JoltProverPreprocessing<PCS, VC>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    pub verifier: JoltVerifierPreprocessing<PCS, VC>,
    pub pcs_setup: PCS::ProverSetup,
}

impl<PCS, VC> JoltProverPreprocessing<PCS, VC>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    /// The full program preprocessing witness generation consumes. `None` in
    /// committed-program mode, which carries only commitments on the verifier
    /// side and is not yet supported by the modular prover.
    pub fn program(&self) -> Option<&JoltProgramPreprocessing> {
        self.verifier.program.as_full()
    }
}
