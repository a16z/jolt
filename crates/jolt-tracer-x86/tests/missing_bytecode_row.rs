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

/// `li a0, 0x100`; `mulh a0, a0, a0`; `jolt.success`. `mulh` expands to
/// several bytecode rows, the other two to one each.
const TEXT: [u32; 3] = [0x1000_0513, 0x02a5_1533, JOLT_SUCCESS_WORD];

/// An ELF64 image (System V gABI layout) whose only loaded section is `TEXT`
/// at `RAM_START_ADDRESS`.
fn text_elf() -> Vec<u8> {
    const SHSTRTAB: &[u8] = b"\0.text\0.shstrtab\0";
    const SECTION_HEADERS: u64 = 64;
    const TEXT_OFFSET: u64 = SECTION_HEADERS + 3 * 64;
    let text: Vec<u8> = TEXT.iter().flat_map(|word| word.to_le_bytes()).collect();
    let text_size = text.len() as u64;

    let mut elf = b"\x7fELF\x02\x01\x01\0\0\0\0\0\0\0\0\0".to_vec();
    elf.extend_from_slice(&2u16.to_le_bytes()); // e_type = ET_EXEC
    elf.extend_from_slice(&243u16.to_le_bytes()); // e_machine = EM_RISCV
    elf.extend_from_slice(&1u32.to_le_bytes()); // e_version
    elf.extend_from_slice(&RAM_START_ADDRESS.to_le_bytes()); // e_entry
    elf.extend_from_slice(&0u64.to_le_bytes()); // e_phoff
    elf.extend_from_slice(&SECTION_HEADERS.to_le_bytes()); // e_shoff
    elf.extend_from_slice(&0u32.to_le_bytes()); // e_flags
    elf.extend_from_slice(&64u16.to_le_bytes()); // e_ehsize
    elf.extend_from_slice(&56u16.to_le_bytes()); // e_phentsize
    elf.extend_from_slice(&0u16.to_le_bytes()); // e_phnum
    elf.extend_from_slice(&64u16.to_le_bytes()); // e_shentsize
    elf.extend_from_slice(&3u16.to_le_bytes()); // e_shnum
    elf.extend_from_slice(&2u16.to_le_bytes()); // e_shstrndx

    // sh_name, sh_type, sh_flags, sh_addr, sh_offset, sh_size
    let sections = [
        (0u32, 0u32, 0u64, 0u64, 0u64, 0u64),
        // .text: SHT_PROGBITS, SHF_ALLOC | SHF_EXECINSTR
        (1, 1, 6, RAM_START_ADDRESS, TEXT_OFFSET, text_size),
        // .shstrtab: SHT_STRTAB
        (7, 3, 0, 0, TEXT_OFFSET + text_size, SHSTRTAB.len() as u64),
    ];
    for (name, kind, flags, address, offset, size) in sections {
        elf.extend_from_slice(&name.to_le_bytes());
        elf.extend_from_slice(&kind.to_le_bytes());
        // The trailing zeros are sh_link with sh_info, sh_addralign and sh_entsize.
        for field in [flags, address, offset, size, 0, 0, 0] {
            elf.extend_from_slice(&field.to_le_bytes());
        }
    }
    elf.extend_from_slice(&text);
    elf.extend_from_slice(SHSTRTAB);
    elf
}

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
    let complete = jolt_program::build_jolt_program(&text_elf()).expect("build program");
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
