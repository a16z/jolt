pub mod jolt_dag;
pub mod proof_serialization;
pub mod stage;
pub mod state_manager;

#[cfg(test)]
mod tests {
    use crate::host;
    use crate::poly::commitment::dory::DoryCommitmentScheme;
    use crate::zkvm::dag::jolt_dag::JoltDAG;
    use crate::zkvm::dag::state_manager::StateManager;
    use crate::zkvm::{Jolt, JoltRV64IMAC, JoltVerifierPreprocessing};
    use ark_bn254::Fr;
    use serial_test::serial;
    use tracer;
    use tracer::instruction::Cycle;

    #[test]
    #[serial]
    #[should_panic]
    fn truncated_trace() {
        let mut program = host::Program::new("fibonacci-guest");
        let (bytecode, init_memory_state, _) = program.decode();
        let inputs = postcard::to_stdvec(&9u8).unwrap();
        let (_, mut trace, final_memory_state, mut program_io) = program.trace(&inputs);
        trace.truncate(100);
        program_io.outputs[0] = 0; // change the output to 0

        let preprocessing = JoltRV64IMAC::prover_preprocess(
            bytecode.clone(),
            program_io.memory_layout.clone(),
            init_memory_state,
            1 << 16,
        );

        // Setup trace length and padding
        let padded_trace_length = (trace.len() + 1).next_power_of_two();
        trace.resize(padded_trace_length, Cycle::NoOp);

        // truncate trailing zeros on device outputs
        program_io.outputs.truncate(
            program_io
                .outputs
                .iter()
                .rposition(|&b| b != 0)
                .map_or(0, |pos| pos + 1),
        );

        let state_manager = StateManager::new_prover(
            &preprocessing,
            trace,
            program_io.clone(),
            final_memory_state,
        );
        let (proof, _) = JoltDAG::prove(state_manager).ok().unwrap();

        let verifier_preprocessing =
            JoltVerifierPreprocessing::<Fr, DoryCommitmentScheme>::from(&preprocessing);
        let _verification_result =
            JoltRV64IMAC::verify(&verifier_preprocessing, proof, program_io, None);
    }

    #[test]
    #[serial]
    #[should_panic]
    fn malicious_trace() {
        let mut program = host::Program::new("fibonacci-guest");
        let inputs = postcard::to_stdvec(&1u8).unwrap();
        let (bytecode, init_memory_state, _) = program.decode();
        let (mut trace, final_memory_state, mut program_io) = program.trace(&inputs);

        // Since the preprocessing is done with the original memory layout, the verifier should fail
        let preprocessing = JoltRV64IMAC::prover_preprocess(
            bytecode.clone(),
            program_io.memory_layout.clone(),
            init_memory_state,
            1 << 16,
        );

        // change memory address of output & termination bit to the same address as input
        // changes here should not be able to spoof the verifier result
        program_io.memory_layout.output_start = program_io.memory_layout.input_start;
        program_io.memory_layout.output_end = program_io.memory_layout.input_end;
        program_io.memory_layout.termination = program_io.memory_layout.input_start;

        // Setup trace length and padding
        let padded_trace_length = (trace.len() + 1).next_power_of_two();
        trace.resize(padded_trace_length, Cycle::NoOp);

        // truncate trailing zeros on device outputs
        program_io.outputs.truncate(
            program_io
                .outputs
                .iter()
                .rposition(|&b| b != 0)
                .map_or(0, |pos| pos + 1),
        );

        let state_manager = StateManager::new_prover(
            &preprocessing,
            trace,
            program_io.clone(),
            final_memory_state,
        );
        let (proof, _) = JoltDAG::prove(state_manager).ok().unwrap();

        let verifier_preprocessing =
            JoltVerifierPreprocessing::<Fr, DoryCommitmentScheme>::from(&preprocessing);
        let _verification_result =
            JoltRV64IMAC::verify(&verifier_preprocessing, proof, program_io, None);
    }
}
