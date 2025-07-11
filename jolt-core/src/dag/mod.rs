pub mod jolt_dag;
pub mod stage;
pub mod state_manager;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host;
    use crate::jolt::vm::{rv32i_vm::RV32IJoltVM, Jolt, JoltProverPreprocessing};
    use crate::poly::commitment::mock::MockCommitScheme;
    use crate::utils::transcript::{KeccakTranscript, Transcript};
    use ark_bn254::Fr;
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::rc::Rc;
    use tracer;

    #[test]
    fn test_dag_fib_e2e() {
        let mut program = host::Program::new("fibonacci-guest");
        let inputs = postcard::to_stdvec(&9u32).unwrap();
        let (bytecode, init_memory_state) = program.decode();
        let (mut trace, final_memory_state, mut io_device) = program.trace(&inputs);

        // Preprocessing
        let preprocessing: JoltProverPreprocessing<Fr, MockCommitScheme<Fr>> =
            RV32IJoltVM::prover_preprocess(
                bytecode.clone(),
                io_device.memory_layout.clone(),
                init_memory_state,
                1 << 16,
                1 << 16,
                1 << 16,
            );

        // Setup trace length and padding
        let trace_length = trace.len();
        let padded_trace_length = trace_length.next_power_of_two();
        let padding = padded_trace_length - trace_length;

        let last_address = trace.last().unwrap().instruction().normalize().address;
        if padding != 0 {
            trace.extend(
                (0..padding - 1)
                    .map(|i| tracer::instruction::RV32IMCycle::NoOp(last_address + 4 * i)),
            );
            trace.push(tracer::instruction::RV32IMCycle::last_jalr(
                last_address + 4 * (padding - 1),
            ));
        } else {
            assert!(matches!(
                trace.last().unwrap(),
                tracer::instruction::RV32IMCycle::JAL(_)
            ));
            *trace.last_mut().unwrap() = tracer::instruction::RV32IMCycle::last_jalr(last_address);
        }

        // truncate trailing zeros on device outputs
        io_device.outputs.truncate(
            io_device
                .outputs
                .iter()
                .rposition(|&b| b != 0)
                .map_or(0, |pos| pos + 1),
        );

        // State manager components
        let openings = Rc::new(RefCell::new(HashMap::new()));
        let prover_accumulator_pre_wrap =
            crate::poly::opening_proof::ProverOpeningAccumulator::<Fr, MockCommitScheme<Fr>>::new();
        let verifier_accumulator_pre_wrap = crate::poly::opening_proof::VerifierOpeningAccumulator::<
            Fr,
            MockCommitScheme<Fr>,
        >::new();

        let prover_accumulator = Rc::new(RefCell::new(prover_accumulator_pre_wrap));
        let verifier_accumulator = Rc::new(RefCell::new(verifier_accumulator_pre_wrap));
        let mut prover_transcript = KeccakTranscript::new(b"Jolt");
        let mut verifier_transcript = KeccakTranscript::new(b"Jolt");
        let proofs = Rc::new(RefCell::new(HashMap::new()));

        // Create prover state manager
        let mut prover_state_manager = state_manager::StateManager::new_prover(
            openings.clone(),
            prover_accumulator,
            &mut prover_transcript,
            proofs.clone(),
        );
        prover_state_manager.set_prover_data(
            &preprocessing,
            trace.clone(),
            io_device.clone(),
            final_memory_state.clone(),
        );

        // Create verifier state manager
        let mut verifier_state_manager = state_manager::StateManager::new_verifier(
            openings,
            verifier_accumulator,
            &mut verifier_transcript,
            proofs,
        );

        let verifier_preprocessing =
            crate::jolt::vm::JoltVerifierPreprocessing::from(&preprocessing);
        verifier_state_manager.set_verifier_data(&verifier_preprocessing, io_device, trace.len());

        // JoltDAG
        let mut dag = jolt_dag::JoltDAG::new(prover_state_manager, verifier_state_manager);

        // Run prove
        if let Err(e) = dag.prove() {
            panic!("DAG prove failed: {e}");
        }

        // Now verify the proof
        if let Err(e) = dag.verify() {
            panic!("DAG verify failed: {e}");
        }

        println!("DAG fib_e2e OK!");
    }
}
