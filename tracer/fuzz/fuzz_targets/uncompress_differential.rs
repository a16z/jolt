#![no_main]

//! Differential expansion of a 16-bit compressed RISC-V instruction by the two
//! independently maintained decompressors: `tracer::uncompress_instruction`
//! and `jolt_riscv::uncompress_rv64_instruction`.
//!
//! The expanded 32-bit words must be identical — a mismatch means the emulator
//! and bytecode preprocessing expand the same compressed instruction to
//! different base instructions. As a second leg, the expanded word is decoded
//! by both full decoders and required to agree, tying the C-extension path
//! into the same soundness contract as `decode_differential`.

use jolt_program::image::decode::decode_instruction;
use jolt_riscv::{uncompress_rv64_instruction, SourceInstructionKind, RV64IMAC_JOLT};
use libfuzzer_sys::fuzz_target;
use tracer::instruction::Instruction;

fuzz_target!(|data: &[u8]| {
    if data.len() < 2 {
        return;
    }
    let halfword = u16::from_le_bytes([data[0], data[1]]);

    let tracer_word = tracer::instruction::uncompress_instruction(u32::from(halfword));
    let program_word = uncompress_rv64_instruction(halfword);
    assert_eq!(
        tracer_word, program_word,
        "decompressors disagree for halfword {halfword:#06x}: \
         tracer {tracer_word:#010x} vs jolt-riscv {program_word:#010x}"
    );

    // An expansion of 0 marks an illegal/reserved compressed encoding in both
    // implementations; there is nothing to decode.
    if program_word == 0 {
        return;
    }

    let address: u64 = 0x8000_0000;
    let tracer_result = Instruction::decode(program_word, address, true);
    let program_result = decode_instruction(program_word, address, true, RV64IMAC_JOLT);

    if let (Ok(tracer_instruction), Ok(program_instruction)) = (tracer_result, program_result) {
        let tracer_source = tracer_instruction.source_instruction();
        if tracer_source.kind() == SourceInstructionKind::Inline
            || program_instruction.kind() == SourceInstructionKind::Inline
        {
            return;
        }
        assert_eq!(
            tracer_source.kind(),
            program_instruction.kind(),
            "decoders disagree on kind for expanded halfword {halfword:#06x} (word {program_word:#010x})"
        );
        assert_eq!(
            tracer_source.row().operands,
            program_instruction.row().operands,
            "decoders disagree on operands for expanded halfword {halfword:#06x}"
        );
    }
});
