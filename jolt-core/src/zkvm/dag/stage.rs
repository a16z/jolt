use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::opening_proof::{ProverOpeningAccumulator, VerifierOpeningAccumulator};
use crate::subprotocols::sumcheck_prover::SumcheckInstanceProver;
use crate::subprotocols::sumcheck_verifier::SumcheckInstanceVerifier;
use crate::transcripts::Transcript;
use crate::zkvm::dag::state_manager::StateManager;

pub trait SumcheckStagesProver<
    F: JoltField,
    ProofTranscript: Transcript,
    PCS: CommitmentScheme<Field = F>,
>: Send + Sync
{
    /// Stage 1b: Other sumchecks (outer-remaining + extras) as batchable instances
    fn stage1_instances(
        &mut self,
        _state_manager: &mut StateManager<'_, F, PCS>,
        _opening_accumulator: &mut ProverOpeningAccumulator<F>,
        _transcript: &mut ProofTranscript,
    ) -> Vec<Box<dyn SumcheckInstanceProver<F, ProofTranscript>>> {
        vec![]
    }

    // Stages 2-5 return sumcheck instances that will be batched together
    fn stage2_instances(
        &mut self,
        _state_manager: &mut StateManager<'_, F, PCS>,
        _opening_accumulator: &mut ProverOpeningAccumulator<F>,
        _transcript: &mut ProofTranscript,
    ) -> Vec<Box<dyn SumcheckInstanceProver<F, ProofTranscript>>> {
        vec![]
    }

    fn stage3_instances(
        &mut self,
        _state_manager: &mut StateManager<'_, F, PCS>,
        _opening_accumulator: &mut ProverOpeningAccumulator<F>,
        _transcript: &mut ProofTranscript,
    ) -> Vec<Box<dyn SumcheckInstanceProver<F, ProofTranscript>>> {
        vec![]
    }

    fn stage4_instances(
        &mut self,
        _state_manager: &mut StateManager<'_, F, PCS>,
        _opening_accumulator: &mut ProverOpeningAccumulator<F>,
        _transcript: &mut ProofTranscript,
    ) -> Vec<Box<dyn SumcheckInstanceProver<F, ProofTranscript>>> {
        vec![]
    }

    fn stage5_instances(
        &mut self,
        _state_manager: &mut StateManager<'_, F, PCS>,
        _opening_accumulator: &mut ProverOpeningAccumulator<F>,
        _transcript: &mut ProofTranscript,
    ) -> Vec<Box<dyn SumcheckInstanceProver<F, ProofTranscript>>> {
        vec![]
    }

    fn stage6_instances(
        &mut self,
        _state_manager: &mut StateManager<'_, F, PCS>,
        _opening_accumulator: &mut ProverOpeningAccumulator<F>,
        _transcript: &mut ProofTranscript,
    ) -> Vec<Box<dyn SumcheckInstanceProver<F, ProofTranscript>>> {
        vec![]
    }

    fn stage7_instances(
        &mut self,
        _state_manager: &mut StateManager<'_, F, PCS>,
        _opening_accumulator: &mut ProverOpeningAccumulator<F>,
        _transcript: &mut ProofTranscript,
    ) -> Vec<Box<dyn SumcheckInstanceProver<F, ProofTranscript>>> {
        vec![]
    }
}

pub trait SumcheckStagesVerifier<
    F: JoltField,
    ProofTranscript: Transcript,
    PCS: CommitmentScheme<Field = F>,
>: Send + Sync
{
    /// Stage 1b: Other sumchecks (outer-remaining + extras) as batchable instances
    fn stage1_instances(
        &mut self,
        _state_manager: &mut StateManager<'_, F, PCS>,
        _opening_accumulator: &mut VerifierOpeningAccumulator<F>,
        _transcript: &mut ProofTranscript,
    ) -> Vec<Box<dyn SumcheckInstanceVerifier<F, ProofTranscript>>> {
        vec![]
    }

    fn stage2_instances(
        &mut self,
        _state_manager: &mut StateManager<'_, F, PCS>,
        _opening_accumulator: &mut VerifierOpeningAccumulator<F>,
        _transcript: &mut ProofTranscript,
    ) -> Vec<Box<dyn SumcheckInstanceVerifier<F, ProofTranscript>>> {
        vec![]
    }

    fn stage3_instances(
        &mut self,
        _state_manager: &mut StateManager<'_, F, PCS>,
        _opening_accumulator: &mut VerifierOpeningAccumulator<F>,
        _transcript: &mut ProofTranscript,
    ) -> Vec<Box<dyn SumcheckInstanceVerifier<F, ProofTranscript>>> {
        vec![]
    }

    fn stage4_instances(
        &mut self,
        _state_manager: &mut StateManager<'_, F, PCS>,
        _opening_accumulator: &mut VerifierOpeningAccumulator<F>,
        _transcript: &mut ProofTranscript,
    ) -> Vec<Box<dyn SumcheckInstanceVerifier<F, ProofTranscript>>> {
        vec![]
    }

    fn stage5_instances(
        &mut self,
        _state_manager: &mut StateManager<'_, F, PCS>,
        _opening_accumulator: &mut VerifierOpeningAccumulator<F>,
        _transcript: &mut ProofTranscript,
    ) -> Vec<Box<dyn SumcheckInstanceVerifier<F, ProofTranscript>>> {
        vec![]
    }

    fn stage6_instances(
        &mut self,
        _state_manager: &mut StateManager<'_, F, PCS>,
        _opening_accumulator: &mut VerifierOpeningAccumulator<F>,
        _transcript: &mut ProofTranscript,
    ) -> Vec<Box<dyn SumcheckInstanceVerifier<F, ProofTranscript>>> {
        vec![]
    }

    fn stage7_instances(
        &mut self,
        _state_manager: &mut StateManager<'_, F, PCS>,
        _opening_accumulator: &mut VerifierOpeningAccumulator<F>,
        _transcript: &mut ProofTranscript,
    ) -> Vec<Box<dyn SumcheckInstanceVerifier<F, ProofTranscript>>> {
        vec![]
    }
}
