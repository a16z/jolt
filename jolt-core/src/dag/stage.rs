use crate::dag::state_manager::StateManager;
use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::subprotocols::sumcheck::{BatchableSumcheckInstance, CacheSumcheckOpenings};
use crate::utils::transcript::Transcript;

pub trait StagedSumcheck<F, ProofTranscript, PCS>:
    BatchableSumcheckInstance<F, ProofTranscript> + CacheSumcheckOpenings<F, ProofTranscript, PCS>
where
    F: JoltField,
    ProofTranscript: Transcript,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
{
}

pub trait SumcheckStages<
    F: JoltField,
    ProofTranscript: Transcript,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
>: Send + Sync
{
    // Stage 1 is special case of outer sumcheck from spartan
    fn stage1_prove(
        &self,
        _state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Result<(), anyhow::Error> {
        Ok(())
    }

    fn stage1_verify(
        &self,
        _state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Result<(), anyhow::Error> {
        Ok(())
    }

    // Stages 2-5 return sumcheck instances that will be batched together
    fn stage2_prover_instances(
        &self,
        _state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn StagedSumcheck<F, ProofTranscript, PCS>>> {
        vec![]
    }

    fn stage2_verifier_instances(
        &self,
        _state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn StagedSumcheck<F, ProofTranscript, PCS>>> {
        vec![]
    }

    fn stage3_prover_instances(
        &self,
        _state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn StagedSumcheck<F, ProofTranscript, PCS>>> {
        vec![]
    }

    fn stage3_verifier_instances(
        &self,
        _state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn StagedSumcheck<F, ProofTranscript, PCS>>> {
        vec![]
    }

    fn stage4_prover_instances(
        &self,
        _state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn StagedSumcheck<F, ProofTranscript, PCS>>> {
        vec![]
    }

    fn stage4_verifier_instances(
        &self,
        _state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn StagedSumcheck<F, ProofTranscript, PCS>>> {
        vec![]
    }

    fn stage5_prover_instances(
        &self,
        _state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn StagedSumcheck<F, ProofTranscript, PCS>>> {
        vec![]
    }

    fn stage5_verifier_instances(
        &self,
        _state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn StagedSumcheck<F, ProofTranscript, PCS>>> {
        vec![]
    }
}
