pub mod jolt_dag;
pub mod stage;
pub mod state_manager;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host;
    use crate::jolt::vm::{rv32i_vm::RV32IJoltVM, Jolt, JoltProverPreprocessing};
    use crate::poly::commitment::dory::DoryCommitmentScheme;
    use crate::poly::opening_proof::{ProverOpeningAccumulator, VerifierOpeningAccumulator};
    use crate::utils::transcript::{KeccakTranscript, Transcript};
    use ark_bn254::Fr;
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::rc::Rc;
    use tracer;
    use tracer::instruction::RV32IMCycle;

    #[test]
    fn test_dag_fib_e2e() {
        let mut program = host::Program::new("fibonacci-guest");
        let inputs = postcard::to_stdvec(&100u32).unwrap();
        let (bytecode, init_memory_state) = program.decode();
        let (mut trace, final_memory_state, mut io_device) = program.trace(&inputs);

        // Preprocessing
        let preprocessing: JoltProverPreprocessing<Fr, DoryCommitmentScheme> =
            RV32IJoltVM::prover_preprocess(
                bytecode.clone(),
                io_device.memory_layout.clone(),
                init_memory_state,
                1 << 16,
                1 << 16,
                1 << 16,
            );

        // Setup trace length and padding
        let padded_trace_length = (trace.len() + 1).next_power_of_two();
        trace.resize(padded_trace_length, RV32IMCycle::NoOp);

        // truncate trailing zeros on device outputs
        io_device.outputs.truncate(
            io_device
                .outputs
                .iter()
                .rposition(|&b| b != 0)
                .map_or(0, |pos| pos + 1),
        );

        // State manager components
        let prover_accumulator_pre_wrap = ProverOpeningAccumulator::<Fr>::new();
        let verifier_accumulator_pre_wrap = VerifierOpeningAccumulator::<Fr>::new();

        let prover_accumulator = Rc::new(RefCell::new(prover_accumulator_pre_wrap));
        let verifier_accumulator = Rc::new(RefCell::new(verifier_accumulator_pre_wrap));
        let prover_transcript = Rc::new(RefCell::new(KeccakTranscript::new(b"Jolt")));
        let verifier_transcript = Rc::new(RefCell::new(KeccakTranscript::new(b"Jolt")));
        let proofs = Rc::new(RefCell::new(HashMap::new()));
        let commitments = Rc::new(RefCell::new(None));

        // Create state managers
        let mut prover_state_manager = state_manager::StateManager::new_prover(
            prover_accumulator,
            prover_transcript.clone(),
            proofs.clone(),
            commitments.clone(),
        );
        prover_state_manager.set_prover_data(
            &preprocessing,
            trace.clone(),
            io_device.clone(),
            final_memory_state.clone(),
        );

        let mut verifier_state_manager = state_manager::StateManager::new_verifier(
            verifier_accumulator,
            verifier_transcript.clone(),
            proofs,
            commitments,
        );

        let verifier_preprocessing =
            crate::jolt::vm::JoltVerifierPreprocessing::from(&preprocessing);
        verifier_state_manager.set_verifier_data(&verifier_preprocessing, io_device, trace.len());

        let mut dag = jolt_dag::JoltDAG::new(prover_state_manager, verifier_state_manager);

        if let Err(e) = dag.prove() {
            panic!("DAG prove failed: {e}");
        }

        if let Err(e) = dag.verify() {
            panic!("DAG verify failed: {e}");
        }

        println!("DAG fib_e2e OK!");
    }
}
