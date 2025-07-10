use crate::dag::state_manager::StateManager;
use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::subprotocols::sumcheck::BatchableSumcheckInstance;
use crate::utils::transcript::Transcript;

pub trait SumcheckStages<
    F: JoltField,
    ProofTranscript: Transcript,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
>: Send + Sync
{
    // Stage 1 is special case of outer sumchec from spartan
    fn stage1_prove(
        &self,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> anyhow::Result<()> {
        let _ = state_manager;
        Ok(())
    }

    fn stage1_verify(
        &self,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> anyhow::Result<()> {
        let _ = state_manager;
        Ok(())
    }

    // Stages 2-5 return sumcheck instances that will be batched together
    fn stage2_prover_instances(
        &self,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn BatchableSumcheckInstance<F, ProofTranscript>>> {
        let _ = state_manager;
        vec![]
    }

    fn stage2_verifier_instances(
        &self,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn BatchableSumcheckInstance<F, ProofTranscript>>> {
        let _ = state_manager;
        vec![]
    }

    fn stage3_prover_instances(
        &self,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn BatchableSumcheckInstance<F, ProofTranscript>>> {
        let _ = state_manager;
        vec![]
    }

    fn stage3_verifier_instances(
        &self,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn BatchableSumcheckInstance<F, ProofTranscript>>> {
        let _ = state_manager;
        vec![]
    }

    fn stage4_prover_instances(
        &self,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn BatchableSumcheckInstance<F, ProofTranscript>>> {
        let _ = state_manager;
        vec![]
    }

    fn stage4_verifier_instances(
        &self,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn BatchableSumcheckInstance<F, ProofTranscript>>> {
        let _ = state_manager;
        vec![]
    }

    fn stage5_prover_instances(
        &self,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn BatchableSumcheckInstance<F, ProofTranscript>>> {
        let _ = state_manager;
        vec![]
    }

    fn stage5_verifier_instances(
        &self,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn BatchableSumcheckInstance<F, ProofTranscript>>> {
        let _ = state_manager;
        vec![]
    }
}
