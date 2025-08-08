use common::constants::virtual_register_index;
use serde::{Deserialize, Serialize};

use crate::declare_riscv_instr;
use crate::emulator::cpu::Cpu;
use crate::emulator::cpu::Xlen;
use crate::instruction::format::format_r::FormatR;
use crate::instruction::format::InstructionFormat;
use crate::instruction::inline_keccak256::{
    execute_keccak_f, Keccak256SequenceBuilder, NEEDED_REGISTERS,
};
use crate::instruction::{
    RISCVInstruction, RISCVTrace, RV32IMCycle, RV32IMInstruction,
};

declare_riscv_instr!(
    name   = KECCAK256,
    mask   = 0xfe00707f,  // Mask for funct7 + funct3 + opcode
    match  = 0x0200000b,  // funct7=0x01, funct3=0x0, opcode=0x0B (custom-0)
    format = FormatR,
    ram    = ()
);

impl KECCAK256 {
    fn exec(&self, cpu: &mut Cpu, _ram_access: &mut <KECCAK256 as RISCVInstruction>::RAMAccess) {
        // This is the "fast path" for emulation without tracing.
        // It performs the Keccak permutation using a native Rust implementation.

        // 1. Read the 25-lane (200-byte) state from memory pointed to by rs1.
        let mut state = [0u64; 25];
        let base_addr = cpu.x[self.operands.rs1 as usize] as u64;
        for (i, lane) in state.iter_mut().enumerate() {
            *lane = cpu
                .mmu
                .load_doubleword(base_addr.wrapping_add((i * 8) as u64))
                .expect("KECCAK256: Failed to load state from memory")
                .0;
        }

        // 2. Execute the Keccak-f permutation on the state.
        execute_keccak_f(&mut state);

        // 3. Write the permuted state back to memory.
        for (i, &lane) in state.iter().enumerate() {
            cpu.mmu
                .store_doubleword(base_addr.wrapping_add((i * 8) as u64), lane)
                .expect("KECCAK256: Failed to store state to memory");
        }
    }
}

impl RISCVTrace for KECCAK256 {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<RV32IMCycle>>) {
        let virtual_sequence = self.virtual_sequence(cpu.xlen);

        let mut trace = trace;
        for instr in virtual_sequence {
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    fn virtual_sequence(&self, xlen: Xlen) -> Vec<RV32IMInstruction> {
        // Virtual registers used as a scratch space
        let mut vr = [0; NEEDED_REGISTERS];
        (0..NEEDED_REGISTERS).for_each(|i| {
            vr[i] = virtual_register_index(i as u8);
        });
        let builder = Keccak256SequenceBuilder::new(
            self.address,
            self.is_compressed,
            xlen,
            vr,
            self.operands.rs1,
            self.operands.rs2,
        );
        builder.build()
    }
}

#[cfg(test)]
mod tests {
    use crate::emulator::cpu::Xlen;
    use crate::instruction::inline_keccak256::test_constants::*;
    use crate::instruction::inline_keccak256::test_utils::*;

    // Test that the virtual sequence and direct execution paths produce the same end result
    #[test]
    fn test_keccak_exec_trace_equal() {
        for (desc, state) in TestVectors::get_standard_test_vectors() {
            kverify::assert_exec_trace_equiv(&state, desc);
        }
    }

    // Test that the direct execution (exec method) works correctly against known test vectors
    #[test]
    fn test_keccak256_direct_execution() {
        // Test that the direct execution (exec method) works correctly against known test vectors
        // This validates the instruction logic before testing virtual sequences
        for (i, test_case) in keccak_test_vectors().iter().enumerate() {
            let mut setup_exec = KeccakCpuHarness::new();
            setup_exec.load_state(&test_case.input);

            // Create and execute instruction
            setup_exec.execute_keccak_instruction();

            let result = setup_exec.read_state();
            assert_eq!(
                result, test_case.expected,
                "Keccak256 direct execution test case {} failed: {}\nInput: {:016x?}\nExpected: {:016x?}\nActual: {:016x?}",
                i + 1, test_case.description, test_case.input, test_case.expected, result
            );
        }
    }

    // Identifies exact point at which direct execution and virtual sequence diverge
    // by that virtual registers match (in contrast to test_keccak_state_equivalence which only tests the final state).
    // This test is SUPER useful and has caught the most bugs and regressions.
    #[test]
    fn test_keccak_exec_trace_intermediate_vr_equal() {
        for (description, initial_state) in TestVectors::get_standard_test_vectors() {
            // Test each round and step
            for round in 0..24 {
                for step in &["theta", "rho_and_pi", "chi", "iota"] {
                    // For debugging, you can uncomment this to see the progress.
                    // println!("\n--- Testing Round {} after {} ---", round, step);

                    // 1. Set up a fresh CPU for this specific test case.
                    let mut setup = KeccakCpuHarness::new();
                    setup.load_state(&initial_state);

                    // 2. Generate and execute the virtual sequence up to the current step.
                    let builder = super::Keccak256SequenceBuilder::new(
                        0x1000,
                        false,
                        Xlen::Bit64,
                        setup.vr,
                        10,
                        11,
                    );
                    let sequence = builder.build_up_to_step(round, step);
                    setup.execute_virtual_sequence(&sequence);
                    let vr_state = if *step == "rho_and_pi" {
                        // This is needed because there is an optimization where the instructions in chi read
                        // directly from the rho_and_pi scratch regisers to avoid a copy back.
                        setup.read_vr_at_offset(25)
                    } else {
                        setup.read_vr()
                    };

                    // 3. Compute the correct, expected state using the reference implementation helper.
                    let expected_vr_state =
                        execute_reference_up_to_step(&initial_state, round as usize, step);

                    // 4. Assert that the states are equal. This will panic with a detailed
                    //    error message on the first divergence, stopping the test.
                    kverify::assert_states_equal(
                        &expected_vr_state,
                        &vr_state,
                        &format!(
                            "test_keccak_exec_trace_intermediate_vr_equal(case={description}, round={round}, step={step})"
                        ),
                    );
                }
            }
        }
    }

    #[test]
    fn test_keccak_against_reference() {
        let initial_state = [0u64; 25];
        let expected_final_state = xkcp_vectors::AFTER_ONE_PERMUTATION;

        // EXEC PATH
        let mut setup_exec = KeccakCpuHarness::new();
        setup_exec.load_state(&initial_state);
        setup_exec.execute_keccak_instruction();
        let exec_result = setup_exec.read_state();

        // TRACE PATH
        let mut setup_trace = KeccakCpuHarness::new();
        setup_trace.load_state(&initial_state);
        setup_trace.trace_keccak_instruction();
        let trace_result = setup_trace.read_state();

        for i in 0..25 {
            assert_eq!(
                exec_result[i], trace_result[i],
                "Lane {i:02}: Exec vs Trace mismatch"
            );
            assert_eq!(
                exec_result[i], expected_final_state[i],
                "Lane {i:02}: Exec vs Expected mismatch"
            );
        }
    }

    #[test]
    fn measure_keccak_length() {
        let mut h = KeccakCpuHarness::new();
        h.load_state(&xkcp_vectors::AFTER_ONE_PERMUTATION);
        let bytecode_len = h.trace_keccak_instruction().len();

        println!(
            "Keccak1600: bytecode length {}, {:.2} instructions per byte",
            bytecode_len,
            bytecode_len as f64 / 136.0,
        );
    }
}
