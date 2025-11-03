#[cfg(feature = "allocative")]
use crate::utils::profiling::print_data_structure_heap_usage;
use crate::{
    field::JoltField,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        opening_proof::{ProverOpeningAccumulator, VerifierOpeningAccumulator},
    },
    subprotocols::{
        sumcheck_prover::SumcheckInstanceProver, sumcheck_verifier::SumcheckInstanceVerifier,
    },
    transcripts::Transcript,
    zkvm::{
        dag::{
            stage::{SumcheckStagesProver, SumcheckStagesVerifier},
            state_manager::StateManager,
        },
        registers::{
            read_write_checking::{
                RegistersReadWriteCheckingProver, RegistersReadWriteCheckingVerifier,
            },
            val_evaluation::{ValEvaluationSumcheckProver, ValEvaluationSumcheckVerifier},
        },
    },
};

pub mod read_write_checking;
pub mod val_evaluation;

pub struct RegistersDagProver;

impl<F: JoltField, ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>
    SumcheckStagesProver<F, ProofTranscript, PCS> for RegistersDagProver
{
    fn stage4_instances(
        &mut self,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
        opening_accumulator: &mut ProverOpeningAccumulator<F>,
    ) -> Vec<Box<dyn SumcheckInstanceProver<F, ProofTranscript>>> {
        let read_write_checking =
            RegistersReadWriteCheckingProver::gen(state_manager, opening_accumulator);
        #[cfg(feature = "allocative")]
        print_data_structure_heap_usage(
            "registers RegistersReadWriteChecking",
            &read_write_checking,
        );
        vec![Box::new(read_write_checking)]
    }

    fn stage5_instances(
        &mut self,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
        opening_accumulator: &mut ProverOpeningAccumulator<F>,
    ) -> Vec<Box<dyn SumcheckInstanceProver<F, ProofTranscript>>> {
        let val_evaluation = ValEvaluationSumcheckProver::gen(state_manager, opening_accumulator);
        #[cfg(feature = "allocative")]
        print_data_structure_heap_usage("registers ValEvaluationSumcheck", &val_evaluation);
        vec![Box::new(val_evaluation)]
    }
}

pub struct RegistersDagVerifier;

impl<F: JoltField, ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>
    SumcheckStagesVerifier<F, ProofTranscript, PCS> for RegistersDagVerifier
{
    fn stage4_instances(
        &mut self,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
        opening_accumulator: &mut VerifierOpeningAccumulator<F>,
    ) -> Vec<Box<dyn SumcheckInstanceVerifier<F, ProofTranscript>>> {
        let read_write_checking =
            RegistersReadWriteCheckingVerifier::new(state_manager, opening_accumulator);
        vec![Box::new(read_write_checking)]
    }

    fn stage5_instances(
        &mut self,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
        _opening_accumulator: &mut VerifierOpeningAccumulator<F>,
    ) -> Vec<Box<dyn SumcheckInstanceVerifier<F, ProofTranscript>>> {
        let val_evaluation = ValEvaluationSumcheckVerifier::new(state_manager);
        vec![Box::new(val_evaluation)]
    }
}
