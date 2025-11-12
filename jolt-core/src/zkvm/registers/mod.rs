#[cfg(feature = "allocative")]
use crate::utils::profiling::print_data_structure_heap_usage;
use crate::{
    field::JoltField,
    poly::{
        commitment::commitment_scheme::CommitmentScheme, opening_proof::ProverOpeningAccumulator,
    },
    subprotocols::sumcheck_prover::SumcheckInstanceProver,
    transcripts::Transcript,
    zkvm::{
        dag::{stage::SumcheckStagesProver, state_manager::StateManager},
        registers::{
            read_write_checking::RegistersReadWriteCheckingProver,
            val_evaluation::ValEvaluationSumcheckProver,
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
        state_manager: &mut StateManager<'_, F, PCS>,
        opening_accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut ProofTranscript,
    ) -> Vec<Box<dyn SumcheckInstanceProver<F, ProofTranscript>>> {
        let read_write_checking =
            RegistersReadWriteCheckingProver::gen(state_manager, opening_accumulator, transcript);
        #[cfg(feature = "allocative")]
        print_data_structure_heap_usage(
            "registers RegistersReadWriteChecking",
            &read_write_checking,
        );
        vec![Box::new(read_write_checking)]
    }

    fn stage5_instances(
        &mut self,
        state_manager: &mut StateManager<'_, F, PCS>,
        opening_accumulator: &mut ProverOpeningAccumulator<F>,
        _transcript: &mut ProofTranscript,
    ) -> Vec<Box<dyn SumcheckInstanceProver<F, ProofTranscript>>> {
        let val_evaluation = ValEvaluationSumcheckProver::gen(state_manager, opening_accumulator);
        #[cfg(feature = "allocative")]
        print_data_structure_heap_usage("registers ValEvaluationSumcheck", &val_evaluation);
        vec![Box::new(val_evaluation)]
    }
}
