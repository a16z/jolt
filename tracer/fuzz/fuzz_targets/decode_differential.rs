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
use tracer::instruction::{Instruction, RISCVInstruction};

fn matches_tracer_mask<T: RISCVInstruction>(word: u32) -> bool {
    word & T::MASK == T::MATCH
}

fn matches_program_kind_tracer_mask(kind: SourceInstructionKind, word: u32) -> bool {
    match kind {
        SourceInstructionKind::ADD => matches_tracer_mask::<tracer::instruction::add::ADD>(word),
        SourceInstructionKind::ADDI => {
            matches_tracer_mask::<tracer::instruction::addi::ADDI>(word)
        }
        SourceInstructionKind::ADDIW => {
            matches_tracer_mask::<tracer::instruction::addiw::ADDIW>(word)
        }
        SourceInstructionKind::ADDW => {
            matches_tracer_mask::<tracer::instruction::addw::ADDW>(word)
        }
        SourceInstructionKind::AdviceLB => {
            matches_tracer_mask::<tracer::instruction::advice_lb::AdviceLB>(word)
        }
        SourceInstructionKind::AdviceLD => {
            matches_tracer_mask::<tracer::instruction::advice_ld::AdviceLD>(word)
        }
        SourceInstructionKind::AdviceLH => {
            matches_tracer_mask::<tracer::instruction::advice_lh::AdviceLH>(word)
        }
        SourceInstructionKind::AdviceLW => {
            matches_tracer_mask::<tracer::instruction::advice_lw::AdviceLW>(word)
        }
        SourceInstructionKind::AMOADDD => {
            matches_tracer_mask::<tracer::instruction::amoaddd::AMOADDD>(word)
        }
        SourceInstructionKind::AMOADDW => {
            matches_tracer_mask::<tracer::instruction::amoaddw::AMOADDW>(word)
        }
        SourceInstructionKind::AMOANDD => {
            matches_tracer_mask::<tracer::instruction::amoandd::AMOANDD>(word)
        }
        SourceInstructionKind::AMOANDW => {
            matches_tracer_mask::<tracer::instruction::amoandw::AMOANDW>(word)
        }
        SourceInstructionKind::AMOMAXD => {
            matches_tracer_mask::<tracer::instruction::amomaxd::AMOMAXD>(word)
        }
        SourceInstructionKind::AMOMAXUD => {
            matches_tracer_mask::<tracer::instruction::amomaxud::AMOMAXUD>(word)
        }
        SourceInstructionKind::AMOMAXUW => {
            matches_tracer_mask::<tracer::instruction::amomaxuw::AMOMAXUW>(word)
        }
        SourceInstructionKind::AMOMAXW => {
            matches_tracer_mask::<tracer::instruction::amomaxw::AMOMAXW>(word)
        }
        SourceInstructionKind::AMOMIND => {
            matches_tracer_mask::<tracer::instruction::amomind::AMOMIND>(word)
        }
        SourceInstructionKind::AMOMINUD => {
            matches_tracer_mask::<tracer::instruction::amominud::AMOMINUD>(word)
        }
        SourceInstructionKind::AMOMINUW => {
            matches_tracer_mask::<tracer::instruction::amominuw::AMOMINUW>(word)
        }
        SourceInstructionKind::AMOMINW => {
            matches_tracer_mask::<tracer::instruction::amominw::AMOMINW>(word)
        }
        SourceInstructionKind::AMOORD => {
            matches_tracer_mask::<tracer::instruction::amoord::AMOORD>(word)
        }
        SourceInstructionKind::AMOORW => {
            matches_tracer_mask::<tracer::instruction::amoorw::AMOORW>(word)
        }
        SourceInstructionKind::AMOSWAPD => {
            matches_tracer_mask::<tracer::instruction::amoswapd::AMOSWAPD>(word)
        }
        SourceInstructionKind::AMOSWAPW => {
            matches_tracer_mask::<tracer::instruction::amoswapw::AMOSWAPW>(word)
        }
        SourceInstructionKind::AMOXORD => {
            matches_tracer_mask::<tracer::instruction::amoxord::AMOXORD>(word)
        }
        SourceInstructionKind::AMOXORW => {
            matches_tracer_mask::<tracer::instruction::amoxorw::AMOXORW>(word)
        }
        SourceInstructionKind::AND => matches_tracer_mask::<tracer::instruction::and::AND>(word),
        SourceInstructionKind::ANDI => {
            matches_tracer_mask::<tracer::instruction::andi::ANDI>(word)
        }
        SourceInstructionKind::ANDN => {
            matches_tracer_mask::<tracer::instruction::andn::ANDN>(word)
        }
        SourceInstructionKind::AUIPC => {
            matches_tracer_mask::<tracer::instruction::auipc::AUIPC>(word)
        }
        SourceInstructionKind::BEQ => matches_tracer_mask::<tracer::instruction::beq::BEQ>(word),
        SourceInstructionKind::BGE => matches_tracer_mask::<tracer::instruction::bge::BGE>(word),
        SourceInstructionKind::BGEU => {
            matches_tracer_mask::<tracer::instruction::bgeu::BGEU>(word)
        }
        SourceInstructionKind::BLT => matches_tracer_mask::<tracer::instruction::blt::BLT>(word),
        SourceInstructionKind::BLTU => {
            matches_tracer_mask::<tracer::instruction::bltu::BLTU>(word)
        }
        SourceInstructionKind::BNE => matches_tracer_mask::<tracer::instruction::bne::BNE>(word),
        SourceInstructionKind::CSRRS => {
            matches_tracer_mask::<tracer::instruction::csrrs::CSRRS>(word)
        }
        SourceInstructionKind::CSRRW => {
            matches_tracer_mask::<tracer::instruction::csrrw::CSRRW>(word)
        }
        SourceInstructionKind::DIV => matches_tracer_mask::<tracer::instruction::div::DIV>(word),
        SourceInstructionKind::DIVU => {
            matches_tracer_mask::<tracer::instruction::divu::DIVU>(word)
        }
        SourceInstructionKind::DIVUW => {
            matches_tracer_mask::<tracer::instruction::divuw::DIVUW>(word)
        }
        SourceInstructionKind::DIVW => {
            matches_tracer_mask::<tracer::instruction::divw::DIVW>(word)
        }
        SourceInstructionKind::EBREAK => {
            matches_tracer_mask::<tracer::instruction::ebreak::EBREAK>(word)
        }
        SourceInstructionKind::ECALL => {
            matches_tracer_mask::<tracer::instruction::ecall::ECALL>(word)
        }
        SourceInstructionKind::FENCE => {
            matches_tracer_mask::<tracer::instruction::fence::FENCE>(word)
        }
        SourceInstructionKind::JAL => matches_tracer_mask::<tracer::instruction::jal::JAL>(word),
        SourceInstructionKind::JALR => {
            matches_tracer_mask::<tracer::instruction::jalr::JALR>(word)
        }
        SourceInstructionKind::LB => matches_tracer_mask::<tracer::instruction::lb::LB>(word),
        SourceInstructionKind::LBU => matches_tracer_mask::<tracer::instruction::lbu::LBU>(word),
        SourceInstructionKind::LD => matches_tracer_mask::<tracer::instruction::ld::LD>(word),
        SourceInstructionKind::LH => matches_tracer_mask::<tracer::instruction::lh::LH>(word),
        SourceInstructionKind::LHU => matches_tracer_mask::<tracer::instruction::lhu::LHU>(word),
        SourceInstructionKind::LRD => matches_tracer_mask::<tracer::instruction::lrd::LRD>(word),
        SourceInstructionKind::LRW => matches_tracer_mask::<tracer::instruction::lrw::LRW>(word),
        SourceInstructionKind::LUI => matches_tracer_mask::<tracer::instruction::lui::LUI>(word),
        SourceInstructionKind::LW => matches_tracer_mask::<tracer::instruction::lw::LW>(word),
        SourceInstructionKind::LWU => matches_tracer_mask::<tracer::instruction::lwu::LWU>(word),
        SourceInstructionKind::MRET => {
            matches_tracer_mask::<tracer::instruction::mret::MRET>(word)
        }
        SourceInstructionKind::MUL => matches_tracer_mask::<tracer::instruction::mul::MUL>(word),
        SourceInstructionKind::MULH => {
            matches_tracer_mask::<tracer::instruction::mulh::MULH>(word)
        }
        SourceInstructionKind::MULHSU => {
            matches_tracer_mask::<tracer::instruction::mulhsu::MULHSU>(word)
        }
        SourceInstructionKind::MULHU => {
            matches_tracer_mask::<tracer::instruction::mulhu::MULHU>(word)
        }
        SourceInstructionKind::MULW => {
            matches_tracer_mask::<tracer::instruction::mulw::MULW>(word)
        }
        SourceInstructionKind::OR => matches_tracer_mask::<tracer::instruction::or::OR>(word),
        SourceInstructionKind::ORI => matches_tracer_mask::<tracer::instruction::ori::ORI>(word),
        SourceInstructionKind::REM => matches_tracer_mask::<tracer::instruction::rem::REM>(word),
        SourceInstructionKind::REMU => {
            matches_tracer_mask::<tracer::instruction::remu::REMU>(word)
        }
        SourceInstructionKind::REMUW => {
            matches_tracer_mask::<tracer::instruction::remuw::REMUW>(word)
        }
        SourceInstructionKind::REMW => {
            matches_tracer_mask::<tracer::instruction::remw::REMW>(word)
        }
        SourceInstructionKind::SB => matches_tracer_mask::<tracer::instruction::sb::SB>(word),
        SourceInstructionKind::SCD => matches_tracer_mask::<tracer::instruction::scd::SCD>(word),
        SourceInstructionKind::SCW => matches_tracer_mask::<tracer::instruction::scw::SCW>(word),
        SourceInstructionKind::SD => matches_tracer_mask::<tracer::instruction::sd::SD>(word),
        SourceInstructionKind::SH => matches_tracer_mask::<tracer::instruction::sh::SH>(word),
        SourceInstructionKind::SLL => matches_tracer_mask::<tracer::instruction::sll::SLL>(word),
        SourceInstructionKind::SLLI => {
            matches_tracer_mask::<tracer::instruction::slli::SLLI>(word)
        }
        SourceInstructionKind::SLLIW => {
            matches_tracer_mask::<tracer::instruction::slliw::SLLIW>(word)
        }
        SourceInstructionKind::SLLW => {
            matches_tracer_mask::<tracer::instruction::sllw::SLLW>(word)
        }
        SourceInstructionKind::SLT => matches_tracer_mask::<tracer::instruction::slt::SLT>(word),
        SourceInstructionKind::SLTI => {
            matches_tracer_mask::<tracer::instruction::slti::SLTI>(word)
        }
        SourceInstructionKind::SLTIU => {
            matches_tracer_mask::<tracer::instruction::sltiu::SLTIU>(word)
        }
        SourceInstructionKind::SLTU => {
            matches_tracer_mask::<tracer::instruction::sltu::SLTU>(word)
        }
        SourceInstructionKind::SRA => matches_tracer_mask::<tracer::instruction::sra::SRA>(word),
        SourceInstructionKind::SRAI => {
            matches_tracer_mask::<tracer::instruction::srai::SRAI>(word)
        }
        SourceInstructionKind::SRAIW => {
            matches_tracer_mask::<tracer::instruction::sraiw::SRAIW>(word)
        }
        SourceInstructionKind::SRAW => {
            matches_tracer_mask::<tracer::instruction::sraw::SRAW>(word)
        }
        SourceInstructionKind::SRL => matches_tracer_mask::<tracer::instruction::srl::SRL>(word),
        SourceInstructionKind::SRLI => {
            matches_tracer_mask::<tracer::instruction::srli::SRLI>(word)
        }
        SourceInstructionKind::SRLIW => {
            matches_tracer_mask::<tracer::instruction::srliw::SRLIW>(word)
        }
        SourceInstructionKind::SRLW => {
            matches_tracer_mask::<tracer::instruction::srlw::SRLW>(word)
        }
        SourceInstructionKind::SUB => matches_tracer_mask::<tracer::instruction::sub::SUB>(word),
        SourceInstructionKind::SUBW => {
            matches_tracer_mask::<tracer::instruction::subw::SUBW>(word)
        }
        SourceInstructionKind::SW => matches_tracer_mask::<tracer::instruction::sw::SW>(word),
        jolt_riscv::SourceInstruction::VirtualAdviceLen(_) => {
            matches_tracer_mask::<tracer::instruction::virtual_advice_len::VirtualAdviceLen>(word)
        }
        SourceInstructionKind::VirtualAssertEQ => {
            matches_tracer_mask::<tracer::instruction::virtual_assert_eq::VirtualAssertEQ>(word)
        }
        jolt_riscv::SourceInstruction::VirtualHostIO(_) => {
            matches_tracer_mask::<tracer::instruction::virtual_host_io::VirtualHostIO>(word)
        }
        jolt_riscv::SourceInstruction::VirtualRev8W(_) => {
            matches_tracer_mask::<tracer::instruction::virtual_rev8w::VirtualRev8W>(word)
        }
        SourceInstructionKind::XOR => matches_tracer_mask::<tracer::instruction::xor::XOR>(word),
        SourceInstructionKind::XORI => {
            matches_tracer_mask::<tracer::instruction::xori::XORI>(word)
        }
        _ => true,
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
    // `decode_instruction` is allowed to classify structurally-valid words
    // whose reserved operand bits fail the tracer constructor's exact mask.
    if !matches_program_kind_tracer_mask(program_instruction.kind(), word) {
        return;
    }
    let Ok(tracer_instruction) = Instruction::decode(word, address, false) else {
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
