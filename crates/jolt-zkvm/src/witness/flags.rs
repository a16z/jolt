//! Flag dispatch: maps tracer instruction types to jolt-instructions flag arrays.
//!
//! Each tracer [`Instruction`] variant maps to a jolt-instructions unit struct
//! whose [`Flags`] implementation provides the static circuit and instruction
//! flags. Dynamic flags (`VirtualInstruction`, `IsCompressed`, etc.) are
//! overlaid from the [`NormalizedInstruction`] context.

use jolt_instructions::flags::{
    CircuitFlags, Flags, InstructionFlags, NUM_CIRCUIT_FLAGS, NUM_INSTRUCTION_FLAGS,
};
use tracer::instruction::{Instruction, NormalizedInstruction};

/// Returns the complete circuit flags for an instruction, including dynamic overlays.
///
/// Static flags come from the jolt-instructions [`Flags`] trait implementation.
/// Dynamic flags are derived from the instruction's [`NormalizedInstruction`] context:
/// - `VirtualInstruction`: set when `virtual_sequence_remaining.is_some()`
/// - `DoNotUpdateUnexpandedPC`: set when in the middle of a virtual sequence
/// - `IsFirstInSequence`: from `NormalizedInstruction::is_first_in_sequence`
/// - `IsCompressed`: from `NormalizedInstruction::is_compressed`
/// - `IsLastInSequence`: set when `virtual_sequence_remaining == Some(0)`
pub fn circuit_flags(instr: &Instruction) -> [bool; NUM_CIRCUIT_FLAGS] {
    let mut flags = static_circuit_flags(instr);
    let norm = instr.normalize();
    apply_dynamic_circuit_flags(&mut flags, &norm);
    flags
}

/// Returns the complete instruction flags, including the dynamic `IsRdNotZero`.
pub fn instruction_flags(instr: &Instruction) -> [bool; NUM_INSTRUCTION_FLAGS] {
    let mut flags = static_instruction_flags(instr);
    let norm = instr.normalize();
    flags[InstructionFlags::IsRdNotZero as usize] = matches!(norm.operands.rd, Some(rd) if rd != 0);
    flags
}

/// Static circuit flags from jolt-instructions, without dynamic overlays.
fn static_circuit_flags(instr: &Instruction) -> [bool; NUM_CIRCUIT_FLAGS] {
    use jolt_instructions::rv::{arithmetic, branch, compare, jump, load, logic, store, system};
    use jolt_instructions::virtual_::{
        advice, arithmetic as varith, assert as vassert, bitwise, byte, division, extension, shift,
        xor_rotate,
    };

    match instr {
        // RV64I arithmetic
        Instruction::ADD(_) => Flags::circuit_flags(&arithmetic::Add),
        Instruction::ADDI(_) => Flags::circuit_flags(&arithmetic::Addi),
        Instruction::SUB(_) => Flags::circuit_flags(&arithmetic::Sub),
        Instruction::LUI(_) => Flags::circuit_flags(&arithmetic::Lui),
        Instruction::AUIPC(_) => Flags::circuit_flags(&arithmetic::Auipc),
        Instruction::MUL(_) => Flags::circuit_flags(&arithmetic::Mul),
        Instruction::MULHU(_) => Flags::circuit_flags(&arithmetic::MulHU),

        // RV64I logic
        Instruction::AND(_) => Flags::circuit_flags(&logic::And),
        Instruction::ANDI(_) => Flags::circuit_flags(&logic::AndI),
        Instruction::ANDN(_) => Flags::circuit_flags(&logic::Andn),
        Instruction::OR(_) => Flags::circuit_flags(&logic::Or),
        Instruction::ORI(_) => Flags::circuit_flags(&logic::OrI),
        Instruction::XOR(_) => Flags::circuit_flags(&logic::Xor),
        Instruction::XORI(_) => Flags::circuit_flags(&logic::XorI),

        // RV64I compare
        Instruction::SLT(_) => Flags::circuit_flags(&compare::Slt),
        Instruction::SLTI(_) => Flags::circuit_flags(&compare::SltI),
        Instruction::SLTIU(_) => Flags::circuit_flags(&compare::SltIU),
        Instruction::SLTU(_) => Flags::circuit_flags(&compare::SltU),

        // RV64I branch
        Instruction::BEQ(_) => Flags::circuit_flags(&branch::Beq),
        Instruction::BGE(_) => Flags::circuit_flags(&branch::Bge),
        Instruction::BGEU(_) => Flags::circuit_flags(&branch::BgeU),
        Instruction::BLT(_) => Flags::circuit_flags(&branch::Blt),
        Instruction::BLTU(_) => Flags::circuit_flags(&branch::BltU),
        Instruction::BNE(_) => Flags::circuit_flags(&branch::Bne),

        // RV64I jump
        Instruction::JAL(_) => Flags::circuit_flags(&jump::Jal),
        Instruction::JALR(_) => Flags::circuit_flags(&jump::Jalr),

        // RV64I load/store
        Instruction::LD(_) => Flags::circuit_flags(&load::Ld),
        Instruction::SD(_) => Flags::circuit_flags(&store::Sd),

        // RV64I system
        Instruction::EBREAK(_) => Flags::circuit_flags(&system::Ebreak),
        Instruction::ECALL(_) => Flags::circuit_flags(&system::Ecall),
        Instruction::FENCE(_) => Flags::circuit_flags(&system::Fence),

        // Virtual: advice
        Instruction::VirtualAdvice(_) => Flags::circuit_flags(&advice::VirtualAdvice),
        Instruction::VirtualAdviceLen(_) => Flags::circuit_flags(&advice::VirtualAdviceLen),
        Instruction::VirtualAdviceLoad(_) => Flags::circuit_flags(&advice::VirtualAdviceLoad),
        Instruction::VirtualHostIO(_) => Flags::circuit_flags(&advice::VirtualHostIO),

        // Virtual: arithmetic
        Instruction::VirtualMULI(_) => Flags::circuit_flags(&varith::MulI),
        Instruction::VirtualPow2(_) => Flags::circuit_flags(&varith::Pow2),
        Instruction::VirtualPow2I(_) => Flags::circuit_flags(&varith::Pow2I),
        Instruction::VirtualPow2W(_) => Flags::circuit_flags(&varith::Pow2W),
        Instruction::VirtualPow2IW(_) => Flags::circuit_flags(&varith::Pow2IW),

        // Virtual: assert
        Instruction::VirtualAssertEQ(_) => Flags::circuit_flags(&vassert::AssertEq),
        Instruction::VirtualAssertLTE(_) => Flags::circuit_flags(&vassert::AssertLte),
        Instruction::VirtualAssertValidDiv0(_) => Flags::circuit_flags(&vassert::AssertValidDiv0),
        Instruction::VirtualAssertValidUnsignedRemainder(_) => {
            Flags::circuit_flags(&vassert::AssertValidUnsignedRemainder)
        }
        Instruction::VirtualAssertMulUNoOverflow(_) => {
            Flags::circuit_flags(&vassert::AssertMulUNoOverflow)
        }
        Instruction::VirtualAssertWordAlignment(_) => {
            Flags::circuit_flags(&vassert::AssertWordAlignment)
        }
        Instruction::VirtualAssertHalfwordAlignment(_) => {
            Flags::circuit_flags(&vassert::AssertHalfwordAlignment)
        }

        // Virtual: bitwise
        Instruction::VirtualMovsign(_) => Flags::circuit_flags(&bitwise::MovSign),

        // Virtual: byte
        Instruction::VirtualRev8W(_) => Flags::circuit_flags(&byte::VirtualRev8W),

        // Virtual: division
        Instruction::VirtualChangeDivisor(_) => {
            Flags::circuit_flags(&division::VirtualChangeDivisor)
        }
        Instruction::VirtualChangeDivisorW(_) => {
            Flags::circuit_flags(&division::VirtualChangeDivisorW)
        }

        // Virtual: extension
        Instruction::VirtualZeroExtendWord(_) => {
            Flags::circuit_flags(&extension::VirtualZeroExtendWord)
        }
        Instruction::VirtualSignExtendWord(_) => {
            Flags::circuit_flags(&extension::VirtualSignExtendWord)
        }

        // Virtual: shift
        Instruction::VirtualSRL(_) => Flags::circuit_flags(&shift::VirtualSrl),
        Instruction::VirtualSRLI(_) => Flags::circuit_flags(&shift::VirtualSrli),
        Instruction::VirtualSRA(_) => Flags::circuit_flags(&shift::VirtualSra),
        Instruction::VirtualSRAI(_) => Flags::circuit_flags(&shift::VirtualSrai),
        Instruction::VirtualShiftRightBitmask(_) => {
            Flags::circuit_flags(&shift::VirtualShiftRightBitmask)
        }
        Instruction::VirtualShiftRightBitmaskI(_) => {
            Flags::circuit_flags(&shift::VirtualShiftRightBitmaski)
        }
        Instruction::VirtualROTRI(_) => Flags::circuit_flags(&shift::VirtualRotri),
        Instruction::VirtualROTRIW(_) => Flags::circuit_flags(&shift::VirtualRotriw),

        // Virtual: xor-rotate
        Instruction::VirtualXORROT32(_) => Flags::circuit_flags(&xor_rotate::VirtualXorRot32),
        Instruction::VirtualXORROT24(_) => Flags::circuit_flags(&xor_rotate::VirtualXorRot24),
        Instruction::VirtualXORROT16(_) => Flags::circuit_flags(&xor_rotate::VirtualXorRot16),
        Instruction::VirtualXORROT63(_) => Flags::circuit_flags(&xor_rotate::VirtualXorRot63),
        Instruction::VirtualXORROTW16(_) => Flags::circuit_flags(&xor_rotate::VirtualXorRotW16),
        Instruction::VirtualXORROTW12(_) => Flags::circuit_flags(&xor_rotate::VirtualXorRotW12),
        Instruction::VirtualXORROTW8(_) => Flags::circuit_flags(&xor_rotate::VirtualXorRotW8),
        Instruction::VirtualXORROTW7(_) => Flags::circuit_flags(&xor_rotate::VirtualXorRotW7),

        // NoOp: DoNotUpdateUnexpandedPC is the only static flag
        Instruction::NoOp => {
            let mut flags = [false; NUM_CIRCUIT_FLAGS];
            flags[CircuitFlags::DoNotUpdateUnexpandedPC as usize] = true;
            flags
        }

        // INLINE instructions are expanded to virtual sequences by the tracer
        // before reaching witness generation. Panic if one leaks through.
        Instruction::INLINE(i) => panic!(
            "INLINE instruction reached witness gen unexpanded: opcode={}, funct3={}, funct7={}",
            i.opcode, i.funct3, i.funct7
        ),

        _ => panic!("unsupported instruction: {instr:?}"),
    }
}

/// Static instruction flags from jolt-instructions.
fn static_instruction_flags(instr: &Instruction) -> [bool; NUM_INSTRUCTION_FLAGS] {
    use jolt_instructions::rv::{arithmetic, branch, compare, jump, load, logic, store, system};
    use jolt_instructions::virtual_::{
        advice, arithmetic as varith, assert as vassert, bitwise, byte, division, extension, shift,
        xor_rotate,
    };

    match instr {
        Instruction::ADD(_) => Flags::instruction_flags(&arithmetic::Add),
        Instruction::ADDI(_) => Flags::instruction_flags(&arithmetic::Addi),
        Instruction::SUB(_) => Flags::instruction_flags(&arithmetic::Sub),
        Instruction::LUI(_) => Flags::instruction_flags(&arithmetic::Lui),
        Instruction::AUIPC(_) => Flags::instruction_flags(&arithmetic::Auipc),
        Instruction::MUL(_) => Flags::instruction_flags(&arithmetic::Mul),
        Instruction::MULHU(_) => Flags::instruction_flags(&arithmetic::MulHU),

        Instruction::AND(_) => Flags::instruction_flags(&logic::And),
        Instruction::ANDI(_) => Flags::instruction_flags(&logic::AndI),
        Instruction::ANDN(_) => Flags::instruction_flags(&logic::Andn),
        Instruction::OR(_) => Flags::instruction_flags(&logic::Or),
        Instruction::ORI(_) => Flags::instruction_flags(&logic::OrI),
        Instruction::XOR(_) => Flags::instruction_flags(&logic::Xor),
        Instruction::XORI(_) => Flags::instruction_flags(&logic::XorI),

        Instruction::SLT(_) => Flags::instruction_flags(&compare::Slt),
        Instruction::SLTI(_) => Flags::instruction_flags(&compare::SltI),
        Instruction::SLTIU(_) => Flags::instruction_flags(&compare::SltIU),
        Instruction::SLTU(_) => Flags::instruction_flags(&compare::SltU),

        Instruction::BEQ(_) => Flags::instruction_flags(&branch::Beq),
        Instruction::BGE(_) => Flags::instruction_flags(&branch::Bge),
        Instruction::BGEU(_) => Flags::instruction_flags(&branch::BgeU),
        Instruction::BLT(_) => Flags::instruction_flags(&branch::Blt),
        Instruction::BLTU(_) => Flags::instruction_flags(&branch::BltU),
        Instruction::BNE(_) => Flags::instruction_flags(&branch::Bne),

        Instruction::JAL(_) => Flags::instruction_flags(&jump::Jal),
        Instruction::JALR(_) => Flags::instruction_flags(&jump::Jalr),

        Instruction::LD(_) => Flags::instruction_flags(&load::Ld),
        Instruction::SD(_) => Flags::instruction_flags(&store::Sd),

        Instruction::EBREAK(_) => Flags::instruction_flags(&system::Ebreak),
        Instruction::ECALL(_) => Flags::instruction_flags(&system::Ecall),
        Instruction::FENCE(_) => Flags::instruction_flags(&system::Fence),

        Instruction::VirtualAdvice(_) => Flags::instruction_flags(&advice::VirtualAdvice),
        Instruction::VirtualAdviceLen(_) => Flags::instruction_flags(&advice::VirtualAdviceLen),
        Instruction::VirtualAdviceLoad(_) => Flags::instruction_flags(&advice::VirtualAdviceLoad),
        Instruction::VirtualHostIO(_) => Flags::instruction_flags(&advice::VirtualHostIO),

        Instruction::VirtualMULI(_) => Flags::instruction_flags(&varith::MulI),
        Instruction::VirtualPow2(_) => Flags::instruction_flags(&varith::Pow2),
        Instruction::VirtualPow2I(_) => Flags::instruction_flags(&varith::Pow2I),
        Instruction::VirtualPow2W(_) => Flags::instruction_flags(&varith::Pow2W),
        Instruction::VirtualPow2IW(_) => Flags::instruction_flags(&varith::Pow2IW),

        Instruction::VirtualAssertEQ(_) => Flags::instruction_flags(&vassert::AssertEq),
        Instruction::VirtualAssertLTE(_) => Flags::instruction_flags(&vassert::AssertLte),
        Instruction::VirtualAssertValidDiv0(_) => {
            Flags::instruction_flags(&vassert::AssertValidDiv0)
        }
        Instruction::VirtualAssertValidUnsignedRemainder(_) => {
            Flags::instruction_flags(&vassert::AssertValidUnsignedRemainder)
        }
        Instruction::VirtualAssertMulUNoOverflow(_) => {
            Flags::instruction_flags(&vassert::AssertMulUNoOverflow)
        }
        Instruction::VirtualAssertWordAlignment(_) => {
            Flags::instruction_flags(&vassert::AssertWordAlignment)
        }
        Instruction::VirtualAssertHalfwordAlignment(_) => {
            Flags::instruction_flags(&vassert::AssertHalfwordAlignment)
        }

        Instruction::VirtualMovsign(_) => Flags::instruction_flags(&bitwise::MovSign),
        Instruction::VirtualRev8W(_) => Flags::instruction_flags(&byte::VirtualRev8W),
        Instruction::VirtualChangeDivisor(_) => {
            Flags::instruction_flags(&division::VirtualChangeDivisor)
        }
        Instruction::VirtualChangeDivisorW(_) => {
            Flags::instruction_flags(&division::VirtualChangeDivisorW)
        }
        Instruction::VirtualZeroExtendWord(_) => {
            Flags::instruction_flags(&extension::VirtualZeroExtendWord)
        }
        Instruction::VirtualSignExtendWord(_) => {
            Flags::instruction_flags(&extension::VirtualSignExtendWord)
        }

        Instruction::VirtualSRL(_) => Flags::instruction_flags(&shift::VirtualSrl),
        Instruction::VirtualSRLI(_) => Flags::instruction_flags(&shift::VirtualSrli),
        Instruction::VirtualSRA(_) => Flags::instruction_flags(&shift::VirtualSra),
        Instruction::VirtualSRAI(_) => Flags::instruction_flags(&shift::VirtualSrai),
        Instruction::VirtualShiftRightBitmask(_) => {
            Flags::instruction_flags(&shift::VirtualShiftRightBitmask)
        }
        Instruction::VirtualShiftRightBitmaskI(_) => {
            Flags::instruction_flags(&shift::VirtualShiftRightBitmaski)
        }
        Instruction::VirtualROTRI(_) => Flags::instruction_flags(&shift::VirtualRotri),
        Instruction::VirtualROTRIW(_) => Flags::instruction_flags(&shift::VirtualRotriw),

        Instruction::VirtualXORROT32(_) => Flags::instruction_flags(&xor_rotate::VirtualXorRot32),
        Instruction::VirtualXORROT24(_) => Flags::instruction_flags(&xor_rotate::VirtualXorRot24),
        Instruction::VirtualXORROT16(_) => Flags::instruction_flags(&xor_rotate::VirtualXorRot16),
        Instruction::VirtualXORROT63(_) => Flags::instruction_flags(&xor_rotate::VirtualXorRot63),
        Instruction::VirtualXORROTW16(_) => Flags::instruction_flags(&xor_rotate::VirtualXorRotW16),
        Instruction::VirtualXORROTW12(_) => Flags::instruction_flags(&xor_rotate::VirtualXorRotW12),
        Instruction::VirtualXORROTW8(_) => Flags::instruction_flags(&xor_rotate::VirtualXorRotW8),
        Instruction::VirtualXORROTW7(_) => Flags::instruction_flags(&xor_rotate::VirtualXorRotW7),

        Instruction::NoOp => {
            let mut flags = [false; NUM_INSTRUCTION_FLAGS];
            flags[InstructionFlags::IsNoop as usize] = true;
            flags
        }

        Instruction::INLINE(i) => panic!(
            "INLINE instruction reached witness gen unexpanded: opcode={}, funct3={}, funct7={}",
            i.opcode, i.funct3, i.funct7
        ),

        _ => panic!("unsupported instruction: {instr:?}"),
    }
}

/// Applies dynamic circuit flags from the instruction's runtime context.
fn apply_dynamic_circuit_flags(
    flags: &mut [bool; NUM_CIRCUIT_FLAGS],
    norm: &NormalizedInstruction,
) {
    if norm.virtual_sequence_remaining.is_some() {
        flags[CircuitFlags::VirtualInstruction as usize] = true;
    }
    if norm.virtual_sequence_remaining.unwrap_or(0) != 0 {
        flags[CircuitFlags::DoNotUpdateUnexpandedPC as usize] = true;
    }
    if norm.is_first_in_sequence {
        flags[CircuitFlags::IsFirstInSequence as usize] = true;
    }
    if norm.is_compressed {
        flags[CircuitFlags::IsCompressed as usize] = true;
    }
    // IsLastInSequence: virtual_sequence_remaining == Some(0)
    if norm.virtual_sequence_remaining == Some(0) {
        flags[CircuitFlags::IsLastInSequence as usize] = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_instructions::flags::{CircuitFlags, InstructionFlags};

    #[test]
    fn noop_flags() {
        let cflags = circuit_flags(&Instruction::NoOp);
        assert!(cflags[CircuitFlags::DoNotUpdateUnexpandedPC as usize]);
        // All other circuit flags should be false
        assert!(!cflags[CircuitFlags::AddOperands as usize]);

        let iflags = instruction_flags(&Instruction::NoOp);
        assert!(iflags[InstructionFlags::IsNoop as usize]);
    }
}
