use crate::dag::stage::SumcheckStages;
use crate::dag::state_manager::StateManager;
use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::r1cs::spartan::SpartanDag;
use crate::utils::transcript::Transcript;
pub struct JoltDAG<
    'a,
    F: JoltField,
    ProofTranscript: Transcript,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
> {
    prover_state_manager: StateManager<'a, F, ProofTranscript, PCS>,
    verifier_state_manager: StateManager<'a, F, ProofTranscript, PCS>,
}

impl<
        'a,
        F: JoltField,
        ProofTranscript: Transcript,
        PCS: CommitmentScheme<ProofTranscript, Field = F>,
    > JoltDAG<'a, F, ProofTranscript, PCS>
{
    pub fn new(
        prover_state_manager: StateManager<'a, F, ProofTranscript, PCS>,
        verifier_state_manager: StateManager<'a, F, ProofTranscript, PCS>,
    ) -> Self {
        Self {
            prover_state_manager,
            verifier_state_manager,
        }
    }

    pub fn prove(&mut self) -> Result<(), anyhow::Error> {
        let spartan_dag = SpartanDag::default();
        spartan_dag.stage1_prove(&mut self.prover_state_manager)?;
        Ok(())
    }

    pub fn verify(&mut self) -> Result<(), anyhow::Error> {
        let spartan_dag = SpartanDag::default();
        spartan_dag.stage1_verify(&mut self.verifier_state_manager)?;
        Ok(())
    }
}
