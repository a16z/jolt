use crate::dag::state_manager::StateManager;
// use crate::dag::stage::SumcheckStages; // TODO: Use when implementing prove/verify
use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::utils::transcript::Transcript;

pub struct JoltDAG<'a, F: JoltField, ProofTranscript: Transcript, PCS: CommitmentScheme<ProofTranscript, Field = F>> {
    state_manager: StateManager<'a, F, ProofTranscript, PCS>,
}

impl<'a, F: JoltField, ProofTranscript: Transcript, PCS: CommitmentScheme<ProofTranscript, Field = F>> JoltDAG<'a, F, ProofTranscript, PCS> {
    pub fn new(state_manager: StateManager<'a, F, ProofTranscript, PCS>, _transcript: ProofTranscript) -> Self {
        Self {
            state_manager
        }
    }


    pub fn prove(&mut self) {
        todo!()
    }

    pub fn verify(&mut self) -> Result<(), crate::utils::errors::ProofVerifyError> {
        todo!()
    }
}