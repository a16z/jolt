mod exec {
    use crate::test_constants::TestVectors;
    use crate::test_utils::{sverify, Sha256CpuHarness};
    use tracer::emulator::cpu::Xlen;
    use tracer::instruction::RISCVTrace;

    #[test]
    fn test_sha256_direct_execution() {
        for (desc, block, initial_state, expected) in TestVectors::get_standard_test_vectors() {
            for xlen in [Xlen::Bit32, Xlen::Bit64] {
                let mut harness = Sha256CpuHarness::new(xlen);
                harness.load_block(&block);
                harness.load_state(&initial_state);
                Sha256CpuHarness::instruction_sha256().trace(&mut harness.harness.cpu, None);
                let result = harness.read_state();

                sverify::assert_states_equal(
                    &expected,
                    &result,
                    &format!("SHA256 direct execution for {xlen:?}: {desc}"),
                );
            }
        }
    }

    #[test]
    fn test_sha256init_direct_execution() {
        for (desc, block, _initial_state, expected) in TestVectors::get_standard_test_vectors() {
            for xlen in [Xlen::Bit32, Xlen::Bit64] {
                let mut harness = Sha256CpuHarness::new(xlen);
                harness.load_block(&block);
                harness.setup_output_only();
                Sha256CpuHarness::instruction_sha256init().trace(&mut harness.harness.cpu, None);
                let result = harness.read_state();

                sverify::assert_states_equal(
                    &expected,
                    &result,
                    &format!("SHA256INIT direct execution for {xlen:?}: {desc}"),
                );
            }
        }
    }
}

mod cpb {
    use crate::test_utils::Sha256CpuHarness;
    use tracer::emulator::cpu::Xlen;

    #[test]
    fn measure_sha256_length() {
        use tracer::instruction::RISCVTrace;
        let instr = Sha256CpuHarness::instruction_sha256();
        for xlen in [Xlen::Bit32, Xlen::Bit64] {
            let sequence = instr.inline_sequence(xlen);
            let bytecode_len = sequence.len();
            println!(
                "SHA256 compression: xlen={:?}, bytecode length {}, {:.2} instructions per byte",
                xlen,
                bytecode_len,
                bytecode_len as f64 / 64.0
            );
        }
    }
}
