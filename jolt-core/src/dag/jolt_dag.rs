use crate::dag::stage::SumcheckStages;
use crate::dag::state_manager::StateManager;
use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::utils::transcript::Transcript;
use crate::r1cs::spartan::SpartanDag;
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
    pub fn new(state_manager: StateManager<'a, F, ProofTranscript, PCS>) -> Self {
        Self { state_manager }
    }

    pub fn prove(&mut self) -> Result<(), anyhow::Error> {
        let spartan_dag = SpartanDag::new();
        spartan_dag.stage1_prove(&mut self.state_manager)?;
        Ok(())
    }

    pub fn verify(&mut self) -> Result<(), anyhow::Error> {
        let spartan_dag = SpartanDag::new();
        spartan_dag.stage1_verify(&mut self.state_manager)?;
        Ok(())
    }
}
