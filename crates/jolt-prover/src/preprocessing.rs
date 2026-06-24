use jolt_crypto::VectorCommitment;
use jolt_openings::CommitmentScheme;
use jolt_verifier::JoltVerifierPreprocessing;

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
    pub fn new(verifier: JoltVerifierPreprocessing<PCS, VC>, pcs_setup: PCS::ProverSetup) -> Self {
        Self {
            verifier,
            pcs_setup,
        }
    }
}
