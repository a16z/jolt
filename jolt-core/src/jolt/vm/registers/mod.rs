use crate::{
    dag::{stage::SumcheckStages, state_manager::StateManager},
    field::JoltField,
    jolt::vm::registers::{
        read_write_checking::RegistersReadWriteChecking, val_evaluation::ValEvaluationSumcheck,
    },
    poly::commitment::commitment_scheme::CommitmentScheme,
    subprotocols::sumcheck::SumcheckInstance,
    utils::transcript::Transcript,
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
    ) -> Vec<Box<dyn SumcheckInstance<F>>> {
        let read_write_checking = RegistersReadWriteChecking::new_prover(state_manager);
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
    ) -> Vec<Box<dyn SumcheckInstance<F>>> {
        let val_evaluation = ValEvaluationSumcheck::new_prover(state_manager);
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
