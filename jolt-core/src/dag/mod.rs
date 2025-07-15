pub mod jolt_dag;
pub mod proof_serialization;
pub mod stage;
pub mod state_manager;

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
    use std::rc::Rc;
    use tracer;

    #[test]
    fn test_dag_fib_e2e() {
        let mut program = host::Program::new("fibonacci-guest");
        let inputs = postcard::to_stdvec(&9u32).unwrap();
        let (bytecode, init_memory_state) = program.decode();
        let (mut trace, final_memory_state, mut io_device) = program.trace(&inputs);

        let preprocessing: JoltProverPreprocessing<Fr, MockCommitScheme<Fr>> =
            RV32IJoltVM::prover_preprocess(
                bytecode.clone(),
                io_device.memory_layout.clone(),
                init_memory_state,
                1 << 16,
                1 << 16,
                1 << 16,
            );

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

        let prover_accumulator = Rc::new(RefCell::new(prover_accumulator_pre_wrap));
        let prover_transcript = Rc::new(RefCell::new(KeccakTranscript::new(b"Jolt")));
        let proofs = Rc::new(RefCell::new(Proofs::default()));
        let commitments = Rc::new(RefCell::new(None));

        let program_data = state_manager::ProgramData {
            preprocessing: &preprocessing,
            trace: trace.clone(),
            program_io: io_device.clone(),
            final_memory_state: final_memory_state.clone(),
        };

        let mut prover_state_manager = state_manager::StateManager::new_prover(
            program_data,
            prover_accumulator,
            prover_transcript.clone(),
            proofs.clone(),
            commitments.clone(),
        );

        let mut dag = jolt_dag::JoltDAG::default();

        let proof = match dag
            .prove::<32, Fr, KeccakTranscript, MockCommitScheme<Fr>>(&mut prover_state_manager)
        {
            Ok(proof) => proof,
            Err(e) => panic!("DAG prove failed: {e}"),
        };

        let prover_verification_data = {
            let prover_accumulator = prover_state_manager
                .get_prover_accumulator()
                .borrow()
                .clone();
            let prover_transcript = prover_state_manager.get_transcript().borrow().clone();
            let (preprocessing, _, _, _) = prover_state_manager.get_prover_data();
            jolt_dag::ProverVerificationData {
                transcript: prover_transcript,
                accumulator: prover_accumulator,
                generators: preprocessing.generators.clone(),
            }
        };

        if let Err(e) = dag.verify::<32, Fr, KeccakTranscript, MockCommitScheme<Fr>>(
            proof,
            prover_verification_data,
        ) {
            panic!("DAG verify failed: {e}");
        }

        println!("DAG fib_e2e OK!");
    }
}
