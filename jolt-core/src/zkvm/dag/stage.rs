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
    /// Stage 1a: Prove first round of Spartan outer sum-check with univariate skip
    fn stage1_uni_skip(
        &mut self,
        _state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
        _opening_accumulator: &mut ProverOpeningAccumulator<F>,
    ) -> Result<(), anyhow::Error> {
        Ok(())
    }

    /// Stage 1b: Other sumchecks (outer-remaining + extras) as batchable instances
    fn stage1_instances(
        &mut self,
        _state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
        _opening_accumulator: &mut ProverOpeningAccumulator<F>,
    ) -> Vec<Box<dyn SumcheckInstanceProver<F, ProofTranscript>>> {
        vec![]
    }

    /// Stage 2a: Prove first round of product virtualization sum-check with univariate skip
    fn stage2_uni_skip(
        &mut self,
        _state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
        _opening_accumulator: &mut ProverOpeningAccumulator<F>,
    ) -> Result<(), anyhow::Error> {
        Ok(())
    }

    // Stages 2-5 return sumcheck instances that will be batched together
    fn stage2_instances(
        &mut self,
        _state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
        _opening_accumulator: &mut ProverOpeningAccumulator<F>,
    ) -> Vec<Box<dyn SumcheckInstanceProver<F, ProofTranscript>>> {
        vec![]
    }

    fn stage3_instances(
        &mut self,
        _state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
        _opening_accumulator: &mut ProverOpeningAccumulator<F>,
    ) -> Vec<Box<dyn SumcheckInstanceProver<F, ProofTranscript>>> {
        vec![]
    }

    fn stage4_instances(
        &mut self,
        _state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
        _opening_accumulator: &mut ProverOpeningAccumulator<F>,
    ) -> Vec<Box<dyn SumcheckInstanceProver<F, ProofTranscript>>> {
        vec![]
    }

    fn stage5_instances(
        &mut self,
        _state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
        _opening_accumulator: &mut ProverOpeningAccumulator<F>,
    ) -> Vec<Box<dyn SumcheckInstanceProver<F, ProofTranscript>>> {
        vec![]
    }

    fn stage6_instances(
        &mut self,
        _state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
        _opening_accumulator: &mut ProverOpeningAccumulator<F>,
    ) -> Vec<Box<dyn SumcheckInstanceProver<F, ProofTranscript>>> {
        vec![]
    }

    fn stage7_instances(
        &mut self,
        _state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
        _opening_accumulator: &mut ProverOpeningAccumulator<F>,
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
    /// Stage 1a: Verify first round of Spartan outer sum-check with univariate skip
    fn stage1_uni_skip(
        &mut self,
        _state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
        _opening_accumulator: &mut VerifierOpeningAccumulator<F>,
    ) -> Result<(), anyhow::Error> {
        Ok(())
    }

    /// Stage 1b: Other sumchecks (outer-remaining + extras) as batchable instances
    fn stage1_instances(
        &mut self,
        _state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
        _opening_accumulator: &mut VerifierOpeningAccumulator<F>,
    ) -> Vec<Box<dyn SumcheckInstanceVerifier<F, ProofTranscript>>> {
        vec![]
    }

    /// Stage 2a: Verify first round of product virtualization sum-check with univariate skip
    fn stage2_uni_skip(
        &mut self,
        _state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
        _opening_accumulator: &mut VerifierOpeningAccumulator<F>,
    ) -> Result<(), anyhow::Error> {
        Ok(())
    }

    fn stage2_instances(
        &mut self,
        _state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
        _opening_accumulator: &mut VerifierOpeningAccumulator<F>,
    ) -> Vec<Box<dyn SumcheckInstanceVerifier<F, ProofTranscript>>> {
        vec![]
    }

    fn stage3_instances(
        &mut self,
        _state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
        _opening_accumulator: &mut VerifierOpeningAccumulator<F>,
    ) -> Vec<Box<dyn SumcheckInstanceVerifier<F, ProofTranscript>>> {
        vec![]
    }

    fn stage4_instances(
        &mut self,
        _state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
        _opening_accumulator: &mut VerifierOpeningAccumulator<F>,
    ) -> Vec<Box<dyn SumcheckInstanceVerifier<F, ProofTranscript>>> {
        vec![]
    }

    fn stage5_instances(
        &mut self,
        _state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
        _opening_accumulator: &mut VerifierOpeningAccumulator<F>,
    ) -> Vec<Box<dyn SumcheckInstanceVerifier<F, ProofTranscript>>> {
        vec![]
    }

    fn stage6_instances(
        &mut self,
        _state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
        _opening_accumulator: &mut VerifierOpeningAccumulator<F>,
    ) -> Vec<Box<dyn SumcheckInstanceVerifier<F, ProofTranscript>>> {
        vec![]
    }

    fn stage7_instances(
        &mut self,
        _state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
        _opening_accumulator: &mut VerifierOpeningAccumulator<F>,
    ) -> Vec<Box<dyn SumcheckInstanceVerifier<F, ProofTranscript>>> {
        vec![]
    }
}
