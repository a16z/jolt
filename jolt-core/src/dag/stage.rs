use crate::dag::state_manager::StateManager;
use crate::field::JoltField;
use crate::subprotocols::sumcheck::BatchableSumcheckInstance;
use crate::utils::transcript::Transcript;

pub trait SumcheckStages<F: JoltField, ProofTranscript: Transcript> {
    fn stage1_prover_instances(
        &self,
        state_manager: &mut StateManager<F, ProofTranscript>,
    ) -> Vec<Box<dyn BatchableSumcheckInstance<F, ProofTranscript>>> {
        vec![]
    }

    fn stage1_verifier_instances(
        &self,
        state_manager: &StateManager<F, ProofTranscript>,
    ) -> Vec<Box<dyn BatchableSumcheckInstance<F, ProofTranscript>>> {
        vec![]
    }

    fn stage2_prover_instances(
        &self,
        state_manager: &StateManager<F, ProofTranscript>,
    ) -> Vec<Box<dyn BatchableSumcheckInstance<F, ProofTranscript>>> {
        vec![]
    }

    fn stage2_verifier_instances(
        &self,
        state_manager: &StateManager<F, ProofTranscript>,
    ) -> Vec<Box<dyn BatchableSumcheckInstance<F, ProofTranscript>>> {
        vec![]
    }

    fn stage3_prover_instances(
        &self,
        state_manager: &StateManager<F, ProofTranscript>,
    ) -> Vec<Box<dyn BatchableSumcheckInstance<F, ProofTranscript>>> {
        vec![]
    }

    fn stage3_verifier_instances(
        &self,
        state_manager: &StateManager<F, ProofTranscript>,
    ) -> Vec<Box<dyn BatchableSumcheckInstance<F, ProofTranscript>>> {
        vec![]
    }

    fn stage4_prover_instances(
        &self,
        state_manager: &StateManager<F, ProofTranscript>,
    ) -> Vec<Box<dyn BatchableSumcheckInstance<F, ProofTranscript>>> {
        vec![]
    }

    fn stage4_verifier_instances(
        &self,
        state_manager: &StateManager<F, ProofTranscript>,
    ) -> Vec<Box<dyn BatchableSumcheckInstance<F, ProofTranscript>>> {
        vec![]
    }

    fn stage5_prover_instances(
        &self,
        state_manager: &StateManager<F, ProofTranscript>,
    ) -> Vec<Box<dyn BatchableSumcheckInstance<F, ProofTranscript>>> {
        vec![]
    }

    fn stage5_verifier_instances(
        &self,
        state_manager: &StateManager<F, ProofTranscript>,
    ) -> Vec<Box<dyn BatchableSumcheckInstance<F, ProofTranscript>>> {
        vec![]
    }
}
