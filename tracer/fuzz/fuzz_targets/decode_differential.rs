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
use jolt_riscv::{SourceInstruction, SourceInstructionKind, RV64IMAC_JOLT};
use libfuzzer_sys::fuzz_target;
use tracer::instruction::{
    add::ADD, addi::ADDI, addiw::ADDIW, addw::ADDW, advice_lb::AdviceLB, advice_ld::AdviceLD,
    advice_lh::AdviceLH, advice_lw::AdviceLW, amoaddd::AMOADDD, amoaddw::AMOADDW, amoandd::AMOANDD,
    amoandw::AMOANDW, amomaxd::AMOMAXD, amomaxud::AMOMAXUD, amomaxuw::AMOMAXUW, amomaxw::AMOMAXW,
    amomind::AMOMIND, amominud::AMOMINUD, amominuw::AMOMINUW, amominw::AMOMINW, amoord::AMOORD,
    amoorw::AMOORW, amoswapd::AMOSWAPD, amoswapw::AMOSWAPW, amoxord::AMOXORD, amoxorw::AMOXORW,
    and::AND, andi::ANDI, andn::ANDN, auipc::AUIPC, beq::BEQ, bge::BGE, bgeu::BGEU, blt::BLT,
    bltu::BLTU, bne::BNE, csrrs::CSRRS, csrrw::CSRRW, div::DIV, divu::DIVU, divuw::DIVUW,
    divw::DIVW, ebreak::EBREAK, ecall::ECALL, fence::FENCE, jal::JAL, jalr::JALR, lb::LB, lbu::LBU,
    ld::LD, lh::LH, lhu::LHU, lrd::LRD, lrw::LRW, lui::LUI, lw::LW, lwu::LWU, mret::MRET, mul::MUL,
    mulh::MULH, mulhsu::MULHSU, mulhu::MULHU, mulw::MULW, or::OR, ori::ORI, rem::REM, remu::REMU,
    remuw::REMUW, remw::REMW, sb::SB, scd::SCD, scw::SCW, sd::SD, sh::SH, sll::SLL, slli::SLLI,
    slliw::SLLIW, sllw::SLLW, slt::SLT, slti::SLTI, sltiu::SLTIU, sltu::SLTU, sra::SRA, srai::SRAI,
    sraiw::SRAIW, sraw::SRAW, srl::SRL, srli::SRLI, srliw::SRLIW, srlw::SRLW, sub::SUB, subw::SUBW,
    sw::SW, virtual_advice_len::VirtualAdviceLen, virtual_assert_eq::VirtualAssertEQ,
    virtual_host_io::VirtualHostIO, virtual_rev8w::VirtualRev8W, xor::XOR, xori::XORI, Instruction,
    RISCVInstruction,
};

fn matches_tracer_mask<T: RISCVInstruction>(word: u32) -> bool {
    word & T::MASK == T::MATCH
}

fn matches_program_kind_tracer_mask(kind: SourceInstructionKind, word: u32) -> bool {
    match kind {
        SourceInstructionKind::ADD => matches_tracer_mask::<ADD>(word),
        SourceInstructionKind::ADDI => matches_tracer_mask::<ADDI>(word),
        SourceInstructionKind::ADDIW => matches_tracer_mask::<ADDIW>(word),
        SourceInstructionKind::ADDW => matches_tracer_mask::<ADDW>(word),
        SourceInstructionKind::AdviceLB => matches_tracer_mask::<AdviceLB>(word),
        SourceInstructionKind::AdviceLD => matches_tracer_mask::<AdviceLD>(word),
        SourceInstructionKind::AdviceLH => matches_tracer_mask::<AdviceLH>(word),
        SourceInstructionKind::AdviceLW => matches_tracer_mask::<AdviceLW>(word),
        SourceInstructionKind::AMOADDD => matches_tracer_mask::<AMOADDD>(word),
        SourceInstructionKind::AMOADDW => matches_tracer_mask::<AMOADDW>(word),
        SourceInstructionKind::AMOANDD => matches_tracer_mask::<AMOANDD>(word),
        SourceInstructionKind::AMOANDW => matches_tracer_mask::<AMOANDW>(word),
        SourceInstructionKind::AMOMAXD => matches_tracer_mask::<AMOMAXD>(word),
        SourceInstructionKind::AMOMAXUD => matches_tracer_mask::<AMOMAXUD>(word),
        SourceInstructionKind::AMOMAXUW => matches_tracer_mask::<AMOMAXUW>(word),
        SourceInstructionKind::AMOMAXW => matches_tracer_mask::<AMOMAXW>(word),
        SourceInstructionKind::AMOMIND => matches_tracer_mask::<AMOMIND>(word),
        SourceInstructionKind::AMOMINUD => matches_tracer_mask::<AMOMINUD>(word),
        SourceInstructionKind::AMOMINUW => matches_tracer_mask::<AMOMINUW>(word),
        SourceInstructionKind::AMOMINW => matches_tracer_mask::<AMOMINW>(word),
        SourceInstructionKind::AMOORD => matches_tracer_mask::<AMOORD>(word),
        SourceInstructionKind::AMOORW => matches_tracer_mask::<AMOORW>(word),
        SourceInstructionKind::AMOSWAPD => matches_tracer_mask::<AMOSWAPD>(word),
        SourceInstructionKind::AMOSWAPW => matches_tracer_mask::<AMOSWAPW>(word),
        SourceInstructionKind::AMOXORD => matches_tracer_mask::<AMOXORD>(word),
        SourceInstructionKind::AMOXORW => matches_tracer_mask::<AMOXORW>(word),
        SourceInstructionKind::AND => matches_tracer_mask::<AND>(word),
        SourceInstructionKind::ANDI => matches_tracer_mask::<ANDI>(word),
        SourceInstructionKind::ANDN => matches_tracer_mask::<ANDN>(word),
        SourceInstructionKind::AUIPC => matches_tracer_mask::<AUIPC>(word),
        SourceInstructionKind::BEQ => matches_tracer_mask::<BEQ>(word),
        SourceInstructionKind::BGE => matches_tracer_mask::<BGE>(word),
        SourceInstructionKind::BGEU => matches_tracer_mask::<BGEU>(word),
        SourceInstructionKind::BLT => matches_tracer_mask::<BLT>(word),
        SourceInstructionKind::BLTU => matches_tracer_mask::<BLTU>(word),
        SourceInstructionKind::BNE => matches_tracer_mask::<BNE>(word),
        SourceInstructionKind::CSRRS => matches_tracer_mask::<CSRRS>(word),
        SourceInstructionKind::CSRRW => matches_tracer_mask::<CSRRW>(word),
        SourceInstructionKind::DIV => matches_tracer_mask::<DIV>(word),
        SourceInstructionKind::DIVU => matches_tracer_mask::<DIVU>(word),
        SourceInstructionKind::DIVUW => matches_tracer_mask::<DIVUW>(word),
        SourceInstructionKind::DIVW => matches_tracer_mask::<DIVW>(word),
        SourceInstructionKind::EBREAK => matches_tracer_mask::<EBREAK>(word),
        SourceInstructionKind::ECALL => matches_tracer_mask::<ECALL>(word),
        SourceInstructionKind::FENCE => matches_tracer_mask::<FENCE>(word),
        SourceInstructionKind::JAL => matches_tracer_mask::<JAL>(word),
        SourceInstructionKind::JALR => matches_tracer_mask::<JALR>(word),
        SourceInstructionKind::LB => matches_tracer_mask::<LB>(word),
        SourceInstructionKind::LBU => matches_tracer_mask::<LBU>(word),
        SourceInstructionKind::LD => matches_tracer_mask::<LD>(word),
        SourceInstructionKind::LH => matches_tracer_mask::<LH>(word),
        SourceInstructionKind::LHU => matches_tracer_mask::<LHU>(word),
        SourceInstructionKind::LRD => matches_tracer_mask::<LRD>(word),
        SourceInstructionKind::LRW => matches_tracer_mask::<LRW>(word),
        SourceInstructionKind::LUI => matches_tracer_mask::<LUI>(word),
        SourceInstructionKind::LW => matches_tracer_mask::<LW>(word),
        SourceInstructionKind::LWU => matches_tracer_mask::<LWU>(word),
        SourceInstructionKind::MRET => matches_tracer_mask::<MRET>(word),
        SourceInstructionKind::MUL => matches_tracer_mask::<MUL>(word),
        SourceInstructionKind::MULH => matches_tracer_mask::<MULH>(word),
        SourceInstructionKind::MULHSU => matches_tracer_mask::<MULHSU>(word),
        SourceInstructionKind::MULHU => matches_tracer_mask::<MULHU>(word),
        SourceInstructionKind::MULW => matches_tracer_mask::<MULW>(word),
        SourceInstructionKind::OR => matches_tracer_mask::<OR>(word),
        SourceInstructionKind::ORI => matches_tracer_mask::<ORI>(word),
        SourceInstructionKind::REM => matches_tracer_mask::<REM>(word),
        SourceInstructionKind::REMU => matches_tracer_mask::<REMU>(word),
        SourceInstructionKind::REMUW => matches_tracer_mask::<REMUW>(word),
        SourceInstructionKind::REMW => matches_tracer_mask::<REMW>(word),
        SourceInstructionKind::SB => matches_tracer_mask::<SB>(word),
        SourceInstructionKind::SCD => matches_tracer_mask::<SCD>(word),
        SourceInstructionKind::SCW => matches_tracer_mask::<SCW>(word),
        SourceInstructionKind::SD => matches_tracer_mask::<SD>(word),
        SourceInstructionKind::SH => matches_tracer_mask::<SH>(word),
        SourceInstructionKind::SLL => matches_tracer_mask::<SLL>(word),
        SourceInstructionKind::SLLI => matches_tracer_mask::<SLLI>(word),
        SourceInstructionKind::SLLIW => matches_tracer_mask::<SLLIW>(word),
        SourceInstructionKind::SLLW => matches_tracer_mask::<SLLW>(word),
        SourceInstructionKind::SLT => matches_tracer_mask::<SLT>(word),
        SourceInstructionKind::SLTI => matches_tracer_mask::<SLTI>(word),
        SourceInstructionKind::SLTIU => matches_tracer_mask::<SLTIU>(word),
        SourceInstructionKind::SLTU => matches_tracer_mask::<SLTU>(word),
        SourceInstructionKind::SRA => matches_tracer_mask::<SRA>(word),
        SourceInstructionKind::SRAI => matches_tracer_mask::<SRAI>(word),
        SourceInstructionKind::SRAIW => matches_tracer_mask::<SRAIW>(word),
        SourceInstructionKind::SRAW => matches_tracer_mask::<SRAW>(word),
        SourceInstructionKind::SRL => matches_tracer_mask::<SRL>(word),
        SourceInstructionKind::SRLI => matches_tracer_mask::<SRLI>(word),
        SourceInstructionKind::SRLIW => matches_tracer_mask::<SRLIW>(word),
        SourceInstructionKind::SRLW => matches_tracer_mask::<SRLW>(word),
        SourceInstructionKind::SUB => matches_tracer_mask::<SUB>(word),
        SourceInstructionKind::SUBW => matches_tracer_mask::<SUBW>(word),
        SourceInstructionKind::SW => matches_tracer_mask::<SW>(word),
        SourceInstruction::VirtualAdviceLen(_) => matches_tracer_mask::<VirtualAdviceLen>(word),
        SourceInstructionKind::VirtualAssertEQ => matches_tracer_mask::<VirtualAssertEQ>(word),
        SourceInstruction::VirtualHostIO(_) => matches_tracer_mask::<VirtualHostIO>(word),
        SourceInstruction::VirtualRev8W(_) => matches_tracer_mask::<VirtualRev8W>(word),
        SourceInstructionKind::XOR => matches_tracer_mask::<XOR>(word),
        SourceInstructionKind::XORI => matches_tracer_mask::<XORI>(word),
        // Every word-decodable kind in this build (no `field-inline`, so no
        // FIELD_* arm) is mapped above; `InlineDispatch` is skipped by the
        // caller before the lookup, and the remaining Virtual* kinds are
        // sequence-only and never come out of `decode_instruction`. A silent
        // `true` here would wave a future word-decodable kind through with a
        // wrong mask assumption, so fail loudly instead.
        unmapped => panic!(
            "decode_instruction produced {unmapped:?}, which has no tracer mask entry; \
             add it to matches_program_kind_tracer_mask"
        ),
    }
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 4 {
        return;
    }
    let word = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
    // A fixed word-aligned address keeps PC-relative operands comparable
    // without spending fuzzer entropy on the address.
    let address: u64 = 0x8000_0000;

    let program_result = decode_instruction(word, address, false, RV64IMAC_JOLT);

    let Ok(program_instruction) = program_result else {
        return;
    };
    // Inline classification is a decoder-profile choice, not a shared ISA
    // decode; skip it before the mask lookup, which has no tracer entry for
    // `InlineDispatch` (post-#1717 `decode_instruction` classifies inline
    // opcodes itself instead of leaving them to expansion).
    if program_instruction.kind() == SourceInstructionKind::Inline {
        return;
    }
    // `decode_instruction` is allowed to classify structurally-valid words
    // whose reserved operand bits fail the tracer constructor's exact mask.
    if !matches_program_kind_tracer_mask(program_instruction.kind(), word) {
        return;
    }
    let Ok(tracer_instruction) = Instruction::decode(word, address, false) else {
        return;
    };
    let tracer_source = tracer_instruction.source_instruction();
    // Same skip for the tracer side: its decoder may classify a word as
    // inline that jolt-program decodes structurally.
    if tracer_source.kind() == SourceInstructionKind::Inline {
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
