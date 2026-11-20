//! Sequential execution that reaches a guest address with no bytecode row.
//! The interpreter decodes the instruction there from memory, then fails to
//! map it to a bytecode PC. The native backend has no code for the address,
//! so it must stop there with the same error instead of continuing in the
//! next compiled group.

#![cfg(all(target_arch = "x86_64", target_os = "linux"))]
#![expect(clippy::expect_used, clippy::panic)]

use common::constants::RAM_START_ADDRESS;
use common::jolt_device::MemoryConfig;
use jolt_program::execution::{ExecutionBackend, TraceError, TraceInputs, TraceOutput};
use jolt_riscv::JOLT_SUCCESS_WORD;
use jolt_tracer_x86::X86TracerBackend;
use tracer::TracerBackend;

mod text_elf;
use text_elf::text_elf;

/// `li a0, 0x100`; `mulh a0, a0, a0`; `jolt.success`. `mulh` expands to
/// several bytecode rows, the other two to one each.
const TEXT: [u32; 3] = [0x1000_0513, 0x02a5_1533, JOLT_SUCCESS_WORD];

fn missing_bytecode_pc(result: Result<TraceOutput, TraceError>) -> (u64, Option<u16>) {
    match result {
        Err(TraceError::MissingBytecodePc {
            address,
            virtual_sequence_remaining,
        }) => (address, virtual_sequence_remaining),
        Err(error) => panic!("expected MissingBytecodePc, got {error:?}"),
        Ok(output) => panic!(
            "expected MissingBytecodePc, traced {} rows",
            output.trace.rows().len()
        ),
    }
}

/// The rowless address is first followed by a compiled group, then is the
/// address after the last bytecode row.
#[test]
fn fallthrough_into_a_rowless_address_matches_the_interpreter() {
    let complete = jolt_program::build_jolt_program(&text_elf(&TEXT)).expect("build program");
    for rowless in [RAM_START_ADDRESS + 4, RAM_START_ADDRESS + 8] {
        let mut program = complete.clone();
        program
            .expanded_bytecode
            .retain(|row| row.address as u64 != rowless);
        let memory_config = MemoryConfig {
            program_size: Some(program.program_end - RAM_START_ADDRESS),
            ..Default::default()
        };
        let inputs = TraceInputs::new(Vec::new(), Vec::new(), Vec::new(), memory_config);

        let reference = missing_bytecode_pc(TracerBackend::new().trace(&program, inputs.clone()));
        assert_eq!(reference.0, rowless);
        assert_eq!(
            missing_bytecode_pc(X86TracerBackend::new().trace(&program, inputs)),
            reference
        );
    }
}
