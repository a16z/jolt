#[cfg(feature = "allocative")]
use crate::utils::profiling::print_data_structure_heap_usage;
use crate::{
    field::JoltField,
    poly::commitment::commitment_scheme::CommitmentScheme,
    subprotocols::sumcheck::SumcheckInstance,
    transcripts::Transcript,
    zkvm::dag::{stage::SumcheckStages, state_manager::StateManager},
    zkvm::registers::{
        read_write_checking::RegistersReadWriteChecking, val_evaluation::ValEvaluationSumcheck,
    },
};

pub mod read_write_checking;
pub mod val_evaluation;

#[derive(Default)]
pub struct RegistersDag {}

impl<F: JoltField, ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>
    SumcheckStages<F, ProofTranscript, PCS> for RegistersDag
{
    fn stage2_prover_instances(
        &mut self,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
        ram_d: usize,
    ) -> Vec<Box<dyn SumcheckInstance<F>>> {
        let read_write_checking = RegistersReadWriteChecking::new_prover(state_manager, ram_d);
        #[cfg(feature = "allocative")]
        print_data_structure_heap_usage(
            "registers RegistersReadWriteChecking",
            &read_write_checking,
        );
        vec![Box::new(read_write_checking)]
    }

    fn stage2_verifier_instances(
        &mut self,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn SumcheckInstance<F>>> {
        let read_write_checking = RegistersReadWriteChecking::new_verifier(state_manager);
        vec![Box::new(read_write_checking)]
    }

    fn stage3_prover_instances(
        &mut self,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
        ram_d: usize,
    ) -> Vec<Box<dyn SumcheckInstance<F>>> {
        let val_evaluation = ValEvaluationSumcheck::new_prover(state_manager, ram_d);
        #[cfg(feature = "allocative")]
        print_data_structure_heap_usage("registers ValEvaluationSumcheck", &val_evaluation);
        vec![Box::new(val_evaluation)]
    }

    fn stage3_verifier_instances(
        &mut self,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Vec<Box<dyn SumcheckInstance<F>>> {
        let val_evaluation = ValEvaluationSumcheck::new_verifier(state_manager);
        vec![Box::new(val_evaluation)]
    }
}
