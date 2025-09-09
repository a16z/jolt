mod exec {
    use crate::test_constants::TestVectors;
    use crate::test_utils::{
        create_sha256_harness, instruction_sha256, instruction_sha256init, sverify,
    };
    use tracer::emulator::cpu::Xlen;

    #[test]
    fn test_sha256_direct_execution() {
        for (desc, block, initial_state, expected) in TestVectors::get_standard_test_vectors() {
            for xlen in [Xlen::Bit32, Xlen::Bit64] {
                let mut harness = create_sha256_harness(xlen);
                harness.setup_registers();
                harness.load_input32(&block);
                harness.load_state32(&initial_state);
                harness.execute_inline(instruction_sha256());

                let result_vec = harness.read_output32(8);
                let mut result = [0u32; 8];
                result.copy_from_slice(&result_vec);

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
                let mut harness = create_sha256_harness(xlen);
                harness.setup_registers();
                harness.load_input32(&block);
                harness.execute_inline(instruction_sha256init());

                let result_vec = harness.read_output32(8);
                let mut result = [0u32; 8];
                result.copy_from_slice(&result_vec);

                sverify::assert_states_equal(
                    &expected,
                    &result,
                    &format!("SHA256INIT direct execution for {xlen:?}: {desc}"),
                );
            }
        }
    }
}

mod cycles_per_byte {
    use crate::test_utils::instruction_sha256;
    use tracer::emulator::cpu::Xlen;

    #[test]
    fn measure_sha256_length() {
        use tracer::instruction::RISCVTrace;
        let instr = instruction_sha256();
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
