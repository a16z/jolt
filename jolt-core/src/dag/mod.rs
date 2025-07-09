pub mod jolt_dag;
pub mod stage;
pub mod state_manager;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host;
    use crate::jolt::vm::{rv32i_vm::RV32IJoltVM, Jolt, JoltProverPreprocessing};
    use crate::poly::commitment::mock::MockCommitScheme;
    use crate::r1cs::constraints::{JoltRV32IMConstraints, R1CSConstraints};
    use crate::r1cs::inputs::JoltR1CSInputs;
    use crate::r1cs::spartan::UniformSpartanProof;
    use crate::utils::transcript::{KeccakTranscript, Transcript};
    use ark_bn254::Fr;
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};
    use tracer;

    #[test]
    fn test_dag_fib_e2e() {
        let mut program = host::Program::new("fibonacci-guest");
        let inputs = postcard::to_stdvec(&9u32).unwrap();
        let (bytecode, init_memory_state) = program.decode();
        let (mut trace, _final_memory_state, mut io_device) = program.trace(&inputs);

        // Preprocessing
        let preprocessing: JoltProverPreprocessing<
            Fr,
            MockCommitScheme<Fr, KeccakTranscript>,
            KeccakTranscript,
        > = RV32IJoltVM::prover_preprocess(
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

        // Spartan stuff
        let constraint_builder = JoltRV32IMConstraints::construct_constraints(padded_trace_length);

        let spartan_key = UniformSpartanProof::<Fr, KeccakTranscript>::setup(
            &constraint_builder,
            padded_trace_length,
        );

        // Create input polynomials from trace
        let mut input_polys = Vec::new();
        for i in 0..JoltR1CSInputs::num_inputs() {
            let input = JoltR1CSInputs::from_index(i);
            let poly = input.generate_witness(&trace, &preprocessing);
            input_polys.push(poly);
        }

        // State manager components
        let openings = Arc::new(Mutex::new(HashMap::new()));
        let mut prover_accumulator = crate::poly::opening_proof::ProverOpeningAccumulator::<
            Fr,
            MockCommitScheme<Fr, KeccakTranscript>,
            KeccakTranscript,
        >::new();
        let mut verifier_accumulator = crate::poly::opening_proof::VerifierOpeningAccumulator::<
            Fr,
            MockCommitScheme<Fr, KeccakTranscript>,
            KeccakTranscript,
        >::new();
        let mut prover_transcript = KeccakTranscript::new(b"Jolt");
        let mut verifier_transcript = KeccakTranscript::new(b"Jolt");
        let proofs = Arc::new(Mutex::new(HashMap::new()));

        let spartan_state = state_manager::SpartanState {
            spartan_key: None,
            constraint_builder: None,
            input_polys: None,
        };

        // Create state manager
        let mut state_manager = state_manager::StateManager::new(
            openings,
            &mut prover_accumulator,
            &mut verifier_accumulator,
            &mut prover_transcript,
            &mut verifier_transcript,
            proofs,
            spartan_state,
        );
        state_manager.set_spartan_data(&spartan_key, &constraint_builder, input_polys);

        // JoltDAG
        let mut dag = jolt_dag::JoltDAG::new(state_manager, KeccakTranscript::new(b"Jolt"));

        // Run prove
        if let Err(e) = dag.prove() {
            panic!("DAG prove failed: {}", e);
        }

        // Now verify the proof
        if let Err(e) = dag.verify() {
            panic!("DAG verify failed: {}", e);
        }

        println!("DAG fib_e2e OK!");
    }
}
