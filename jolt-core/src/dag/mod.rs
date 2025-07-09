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
    use crate::utils::math::Math;
    use ark_bn254::Fr;
    use std::sync::{Arc, Mutex};
    use tracer;

    #[test]
    fn test_dag_fib_e2e() {
        // Mimic the actual fib_e2e test setup
        let mut program = host::Program::new("fibonacci-guest");
        let inputs = postcard::to_stdvec(&9u32).unwrap();
        let (bytecode, init_memory_state) = program.decode();
        let (mut trace, _final_memory_state, mut io_device) = program.trace(&inputs);

        // Preprocessing
        let preprocessing: JoltProverPreprocessing<Fr, MockCommitScheme<Fr, KeccakTranscript>, KeccakTranscript> = RV32IJoltVM::prover_preprocess(
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
            trace.extend((0..padding - 1).map(|i| tracer::instruction::RV32IMCycle::NoOp(last_address + 4 * i)));
            trace.push(tracer::instruction::RV32IMCycle::last_jalr(last_address + 4 * (padding - 1)));
        } else {
            assert!(matches!(trace.last().unwrap(), tracer::instruction::RV32IMCycle::JAL(_)));
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

        // Create constraint builder
        let constraint_builder = JoltRV32IMConstraints::construct_constraints(padded_trace_length);
        
        // Create Spartan key
        let spartan_key = UniformSpartanProof::<Fr, KeccakTranscript>::setup(
            &constraint_builder,
            padded_trace_length,
        );

        // Create transcript
        let transcript = KeccakTranscript::new(b"Jolt");

        // Create input polynomials from trace
        let mut input_polys = Vec::new();
        for i in 0..JoltR1CSInputs::num_inputs() {
            let input = JoltR1CSInputs::from_index(i);
            let poly = input.generate_witness(&trace, &preprocessing);
            input_polys.push(poly);
        }

        // Create state manager
        let state_manager = state_manager::StateManager::new(
            padded_trace_length,
            padded_trace_length.log_2(),
            state_manager::Challenges {
                instruction_booleanity: Fr::from(1),
                instruction_hamming: Fr::from(2),
                instruction_read_raf: Fr::from(3),
            },
            None, // prover_state
            None, // verifier_state
            Arc::new(Mutex::new(std::collections::HashMap::new())),
            Some(&spartan_key),
            Some(constraint_builder.uniform_builder.get_constraints()),
            Some(input_polys),
            None,  // tau - Will be populated by stage1_prove
            None,  // outer_sumcheck_claims
        );

        // Create JoltDAG
        let mut dag = jolt_dag::JoltDAG::new(state_manager, transcript);

        // Register Spartan implementation
        let spartan_proof = UniformSpartanProof::<Fr, KeccakTranscript>::default();
        dag.register(Box::new(spartan_proof));

        // Run prove
        dag.prove();

        println!("DAG prove with fibonacci e2e setup completed successfully");
        
        // Now verify the proof
        let verification_result = dag.verify();
        assert!(verification_result.is_ok(), "Verification failed: {:?}", verification_result);
        
        println!("DAG verify with fibonacci e2e setup completed successfully");
    }
}
