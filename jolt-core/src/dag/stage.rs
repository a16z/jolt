use crate::dag::state_manager::StateManager;
use crate::field::JoltField;
use crate::subprotocols::sumcheck::{BatchableSumcheckInstance, SumcheckInstanceProof};
use crate::utils::errors::ProofVerifyError;
use crate::utils::transcript::Transcript;

/// Trait for components that contribute to specific sumcheck stages
/// Each implementor handles its own sumcheck instances for a particular stage
pub trait SumcheckStages<F: JoltField, ProofTranscript: Transcript>: Send + Sync {
    // Stage 1 is special - it returns proofs directly
    fn stage1_prove(
        &self,
        state_manager: &mut StateManager<F, ProofTranscript>,
        transcript: &mut ProofTranscript,
    ) -> Vec<(SumcheckInstanceProof<F, ProofTranscript>, Vec<F>, [F; 3])> {
        let (_, _) = (transcript, state_manager);
        vec![]
    }

    fn stage1_verify(
        &self,
        proofs: &[(SumcheckInstanceProof<F, ProofTranscript>, Vec<F>, [F; 3])],
        state_manager: &mut StateManager<F, ProofTranscript>,
        transcript: &mut ProofTranscript,
    ) -> Result<Vec<(Vec<F>, (F, F, F))>, ProofVerifyError> {
        let (_, _) = (transcript, state_manager);
        Ok(vec![])
    }

    // Stages 2-5 return sumcheck instances that will be batched together
    fn stage2_prover_instances(
        &self,
        state_manager: &mut StateManager<F, ProofTranscript>,
    ) -> Vec<Box<dyn BatchableSumcheckInstance<F, ProofTranscript>>> {
        let _ = state_manager;
        vec![]
    }

    fn stage2_verifier_instances(
        &self,
        state_manager: &mut StateManager<F, ProofTranscript>,
    ) -> Vec<Box<dyn BatchableSumcheckInstance<F, ProofTranscript>>> {
        let _ = state_manager;
        vec![]
    }

    fn stage3_prover_instances(
        &self,
        state_manager: &mut StateManager<F, ProofTranscript>,
    ) -> Vec<Box<dyn BatchableSumcheckInstance<F, ProofTranscript>>> {
        let _ = state_manager;
        vec![]
    }

    fn stage3_verifier_instances(
        &self,
        state_manager: &mut StateManager<F, ProofTranscript>,
    ) -> Vec<Box<dyn BatchableSumcheckInstance<F, ProofTranscript>>> {
        let _ = state_manager;
        vec![]
    }

    fn stage4_prover_instances(
        &self,
        state_manager: &mut StateManager<F, ProofTranscript>,
    ) -> Vec<Box<dyn BatchableSumcheckInstance<F, ProofTranscript>>> {
        let _ = state_manager;
        vec![]
    }

    fn stage4_verifier_instances(
        &self,
        state_manager: &mut StateManager<F, ProofTranscript>,
    ) -> Vec<Box<dyn BatchableSumcheckInstance<F, ProofTranscript>>> {
        let _ = state_manager;
        vec![]
    }

    fn stage5_prover_instances(
        &self,
        state_manager: &mut StateManager<F, ProofTranscript>,
    ) -> Vec<Box<dyn BatchableSumcheckInstance<F, ProofTranscript>>> {
        let _ = state_manager;
        vec![]
    }

    fn stage5_verifier_instances(
        &self,
        state_manager: &mut StateManager<F, ProofTranscript>,
    ) -> Vec<Box<dyn BatchableSumcheckInstance<F, ProofTranscript>>> {
        let _ = state_manager;
        vec![]
    }
}