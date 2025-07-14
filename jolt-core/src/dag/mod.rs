pub mod jolt_dag;
pub mod stage;
pub mod state_manager;
pub mod proof_serialization;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dag::state_manager::Proofs;
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
        let prover_accumulator_pre_wrap =
            crate::poly::opening_proof::ProverOpeningAccumulator::<Fr, MockCommitScheme<Fr>>::new();
        let verifier_accumulator_pre_wrap = crate::poly::opening_proof::VerifierOpeningAccumulator::<
            Fr,
            MockCommitScheme<Fr>,
        >::new();

        let prover_accumulator = Rc::new(RefCell::new(prover_accumulator_pre_wrap));
        let verifier_accumulator = Rc::new(RefCell::new(verifier_accumulator_pre_wrap));
        let prover_transcript = Rc::new(RefCell::new(KeccakTranscript::new(b"Jolt")));
        let verifier_transcript = Rc::new(RefCell::new(KeccakTranscript::new(b"Jolt")));
        let proofs = Rc::new(RefCell::new(Proofs::default()));
        let commitments = Rc::new(RefCell::new(None));

        // Create state manager for prover
        let prover_state_manager = state_manager::StateManager::new_prover(
            &preprocessing,
            trace.clone(),
            io_device.clone(),
            final_memory_state.clone(),
            prover_accumulator,
            prover_transcript.clone(),
            proofs.clone(),
            commitments.clone(),
        );

        // Create DAG with prover state
        let mut prover_dag = jolt_dag::JoltDAG::new_prover(prover_state_manager);

        // Prove and get the proof
        let proof = match prover_dag.prove::<32>() {
            Ok(proof) => proof,
            Err(e) => panic!("DAG prove failed: {e}"),
        };

        // Create verifier preprocessing
        let verifier_preprocessing =
            crate::jolt::vm::JoltVerifierPreprocessing::from(&preprocessing);
        
        // Create verifier state manager with minimal data
        let verifier_accumulator = Rc::new(RefCell::new(
            crate::poly::opening_proof::VerifierOpeningAccumulator::<Fr, MockCommitScheme<Fr>>::new()
        ));
        let verifier_transcript = Rc::new(RefCell::new(KeccakTranscript::new(b"Jolt")));
        let verifier_proofs = Rc::new(RefCell::new(Proofs::default()));
        let verifier_commitments = Rc::new(RefCell::new(None));
        
        let verifier_state_manager = state_manager::StateManager::new_verifier(
            &verifier_preprocessing,
            io_device,
            trace_length,
            verifier_accumulator,
            verifier_transcript,
            verifier_proofs,
            verifier_commitments,
        );

        // Create DAG with verifier state
        let mut verifier_dag = jolt_dag::JoltDAG::new_verifier(verifier_state_manager);

        // Verify with the proof
        if let Err(e) = verifier_dag.verify(proof) {
            panic!("DAG verify failed: {e}");
        }

        println!("DAG fib_e2e OK!");
    }
}
