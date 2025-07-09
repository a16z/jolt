use crate::dag::stage::{StageResult, SumcheckStages};
use crate::dag::state_manager::StateManager;
use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::r1cs::spartan::UniformSpartanProof;
use crate::utils::transcript::Transcript;

pub struct JoltDAG<
    'a,
    F: JoltField,
    ProofTranscript: Transcript,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
> {
    state_manager: StateManager<'a, F, ProofTranscript, PCS>,
}

impl<
        'a,
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<ProofTranscript, Field = F>,
    > JoltDAG<'a, F, ProofTranscript, PCS>
{
    pub fn new(
        state_manager: StateManager<'a, F, ProofTranscript, PCS>,
        _transcript: ProofTranscript,
    ) -> Self {
        Self { state_manager }
    }

    pub fn prove(&mut self) -> Result<(), String> {
        // Stage 1: Run Spartan's outer sumcheck proof
        let spartan_proof = UniformSpartanProof::<F, ProofTranscript>::default();
        match spartan_proof.stage1_prove(&mut self.state_manager) {
            StageResult::Success => Ok(()),
            StageResult::Failed(reason) => Err(format!("Stage 1 prove failed: {reason}")),
        }
    }

    pub fn verify(&mut self) -> Result<(), String> {
        // Stage 1: Verify Spartan's outer sumcheck proof
        let spartan_proof = UniformSpartanProof::<F, ProofTranscript>::default();
        match spartan_proof.stage1_verify(&mut self.state_manager) {
            StageResult::Success => Ok(()),
            StageResult::Failed(reason) => Err(format!("Stage 1 verify failed: {reason}")),
        }
    }
}
