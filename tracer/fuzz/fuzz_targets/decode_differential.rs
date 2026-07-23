#![no_main]

//! Differential decode of a 32-bit RISC-V word by the two independently
//! maintained RV64IMAC decoders: `tracer::Instruction::decode` (the emulator)
//! and `jolt_program::decode_instruction` (bytecode preprocessing).
//!
//! The soundness-critical property is *agreement when both accept*: if both
//! decoders decode a word, they must produce the same instruction kind and
//! the same normalized operands. A disagreement there means the emulator
//! executes one instruction while the prover commits to another.
//!
//! Accept/reject *parity* is deliberately NOT asserted: the two decoders
//! validate at different depths. tracer's `decode` fully validates because it
//! is about to execute (e.g. it rejects an unsupported CSR operand), while
//! jolt-program's `decode_instruction` is a structural first pass that defers
//! operand validation to expansion. A word that one accepts and the other
//! rejects never reaches both a trace and a proof, so it is not a soundness
//! break; only a both-accept disagreement is.
//!
//! Both decoders can produce a `SourceInstruction`, so the comparison is over
//! `kind()` and `row().operands` on a common type. Inline/field-inline
//! opcodes are decoder-configuration-dependent rather than a shared ISA
//! contract, so words that either side classifies as inline are skipped.

use jolt_program::image::decode::decode_instruction;
use jolt_riscv::{SourceInstructionKind, RV64IMAC_JOLT};
use libfuzzer_sys::fuzz_target;
use tracer::instruction::Instruction;

fuzz_target!(|data: &[u8]| {
    if data.len() < 4 {
        return;
    }
    let word = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
    // A fixed word-aligned address keeps PC-relative operands comparable
    // without spending fuzzer entropy on the address.
    let address: u64 = 0x8000_0000;

    let tracer_result = Instruction::decode(word, address, false);
    let program_result = decode_instruction(word, address, false, RV64IMAC_JOLT);

    let (Ok(tracer_instruction), Ok(program_instruction)) = (tracer_result, program_result) else {
        return;
    };
    let tracer_source = tracer_instruction.source_instruction();
    // Inline classification is a decoder-profile choice, not a shared ISA
    // decode; skip those rather than flag a benign difference.
    if tracer_source.kind() == SourceInstructionKind::Inline
        || program_instruction.kind() == SourceInstructionKind::Inline
    {
        return;
    }
    assert_eq!(
        tracer_source.kind(),
        program_instruction.kind(),
        "decoders disagree on instruction kind for word {word:#010x}"
    );
    assert_eq!(
        tracer_source.row().operands,
        program_instruction.row().operands,
        "decoders disagree on operands for {:?} word {word:#010x}",
        program_instruction.kind()
    );
});
