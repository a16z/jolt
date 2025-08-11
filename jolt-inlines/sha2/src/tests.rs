use crate::test_constants::TestVectors;
use crate::test_utils::{sverify, Sha256CpuHarness};
use tracer::instruction::RISCVInstruction;

#[test]
fn test_sha256_direct_execution() {
    // Test against multiple canonical NIST test vectors
    for (desc, block, initial_state, expected) in TestVectors::get_standard_test_vectors() {
        let mut harness = Sha256CpuHarness::new();
        harness.load_block(&block);
        harness.load_state(&initial_state);
        Sha256CpuHarness::instruction_sha256().execute(&mut harness.harness.cpu, &mut ());
        let result = harness.read_state();

        sverify::assert_states_equal(
            &expected,
            &result,
            &format!("SHA256 direct execution: {desc}"),
        );
    }
}

#[test]
fn test_sha256_exec_trace_equal() {
    // Test exec vs trace equivalence with canonical test vectors
    for (desc, block, initial_state, _expected) in TestVectors::get_standard_test_vectors() {
        sverify::assert_exec_trace_equiv_custom(
            &block,
            &initial_state,
            &format!("SHA256 exec vs trace: {desc}"),
        );
    }
}

#[test]
fn measure_sha256_length() {
    use tracer::emulator::cpu::Xlen;
    use tracer::instruction::RISCVTrace;
    let instr = Sha256CpuHarness::instruction_sha256();
    let sequence = instr.inline_sequence(Xlen::Bit32);
    let bytecode_len = sequence.len();
    println!(
        "SHA256 compression: bytecode length {}, {:.2} instructions per byte",
        bytecode_len,
        bytecode_len as f64 / 64.0,
    );
}

#[test]
fn test_sha256init_direct_execution() {
    // Test against canonical NIST test vectors with initial IV
    for (desc, block, _initial_state, expected) in TestVectors::get_standard_test_vectors() {
        let mut harness = Sha256CpuHarness::new();
        harness.load_block(&block);
        // Note: SHA256INIT doesn't need to load an initial state (uses default IV from BLOCK constants)
        // but it still needs RS2 set to a valid output address
        harness.setup_output_only();
        Sha256CpuHarness::instruction_sha256init().execute(&mut harness.harness.cpu, &mut ());
        let result = harness.read_state();

        sverify::assert_states_equal(
            &expected,
            &result,
            &format!("SHA256INIT direct execution: {desc}"),
        );
    }
}

#[test]
fn test_sha256init_exec_trace_equal() {
    // Test exec vs trace equivalence with canonical test vectors
    for (desc, block, _initial_state, _expected) in TestVectors::get_standard_test_vectors() {
        sverify::assert_exec_trace_equiv_initial(
            &block,
            &format!("SHA256INIT exec vs trace: {desc}"),
        );
    }
}
