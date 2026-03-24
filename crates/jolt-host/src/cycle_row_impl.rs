//! `CycleRow` implementation for `tracer::Cycle`.
//!
//! Absorbs the ISA dispatch table from jolt-zkvm's `flags.rs`, mapping
//! every `Instruction` variant to its circuit and instruction flags via
//! the jolt-instructions `Flags` trait.

use jolt_instructions::flags::{
    CircuitFlags, Flags, InstructionFlags, NUM_CIRCUIT_FLAGS, NUM_INSTRUCTION_FLAGS,
};
use tracer::instruction::{Cycle, Instruction, NormalizedInstruction, RAMAccess};

use crate::CycleRow;

impl CycleRow for Cycle {
    fn noop() -> Self {
        Cycle::NoOp
    }

    fn is_noop(&self) -> bool {
        matches!(self, Cycle::NoOp)
    }

    fn unexpanded_pc(&self) -> u64 {
        match self {
            Cycle::NoOp => 0,
            _ => self.instruction().normalize().address as u64,
        }
    }

    fn virtual_sequence_remaining(&self) -> Option<u16> {
        match self {
            Cycle::NoOp => None,
            _ => self.instruction().normalize().virtual_sequence_remaining,
        }
    }

    fn is_first_in_sequence(&self) -> bool {
        match self {
            Cycle::NoOp => false,
            _ => self.instruction().normalize().is_first_in_sequence,
        }
    }

    fn is_virtual(&self) -> bool {
        self.virtual_sequence_remaining().is_some()
    }

    fn rs1_read(&self) -> Option<(u8, u64)> {
        self.rs1_read()
    }

    fn rs2_read(&self) -> Option<(u8, u64)> {
        self.rs2_read()
    }

    fn rd_write(&self) -> Option<(u8, u64, u64)> {
        self.rd_write()
    }

    fn rd_operand(&self) -> Option<u8> {
        match self {
            Cycle::NoOp => None,
            _ => self.instruction().normalize().operands.rd,
        }
    }

    fn ram_access_address(&self) -> Option<u64> {
        match self.ram_access() {
            RAMAccess::Read(r) => Some(r.address),
            RAMAccess::Write(w) => Some(w.address),
            RAMAccess::NoOp => None,
        }
    }

    fn ram_read_value(&self) -> Option<u64> {
        match self.ram_access() {
            RAMAccess::Read(r) => Some(r.value),
            RAMAccess::Write(w) => Some(w.pre_value),
            RAMAccess::NoOp => None,
        }
    }

    fn ram_write_value(&self) -> Option<u64> {
        match self.ram_access() {
            RAMAccess::Read(r) => Some(r.value),
            RAMAccess::Write(w) => Some(w.post_value),
            RAMAccess::NoOp => None,
        }
    }

    fn imm(&self) -> i128 {
        match self {
            Cycle::NoOp => 0,
            _ => self.instruction().normalize().operands.imm,
        }
    }

    fn circuit_flags(&self) -> [bool; NUM_CIRCUIT_FLAGS] {
        let instr = self.instruction();
        let mut flags = static_circuit_flags(&instr);
        let norm = instr.normalize();
        apply_dynamic_circuit_flags(&mut flags, &norm);
        flags
    }

    fn instruction_flags(&self) -> [bool; NUM_INSTRUCTION_FLAGS] {
        let instr = self.instruction();
        let mut flags = static_instruction_flags(&instr);
        let norm = instr.normalize();
        flags[InstructionFlags::IsRdNotZero as usize] =
            matches!(norm.operands.rd, Some(rd) if rd != 0);
        flags
    }

    fn lookup_index(&self) -> u128 {
        let cflags = self.circuit_flags();
        let iflags = self.instruction_flags();

        let (left, right) = instruction_inputs(self, &iflags);

        if cflags[CircuitFlags::AddOperands] {
            (left as u128).wrapping_add(right)
        } else if cflags[CircuitFlags::SubtractOperands] {
            let right_twos = (1u128 << 64).wrapping_sub(right);
            (left as u128).wrapping_add(right_twos)
        } else if cflags[CircuitFlags::MultiplyOperands] {
            (left as u128).wrapping_mul(right)
        } else if cflags[CircuitFlags::Advice] || self.is_noop() {
            0
        } else {
            jolt_instructions::interleave_bits(left, right as u64)
        }
    }
}

/// Compute the instruction operand inputs from CycleRow data.
fn instruction_inputs(cycle: &impl CycleRow, iflags: &[bool; NUM_INSTRUCTION_FLAGS]) -> (u64, u128) {
    let left = if iflags[InstructionFlags::LeftOperandIsPC] {
        cycle.unexpanded_pc()
    } else if iflags[InstructionFlags::LeftOperandIsRs1Value] {
        match cycle.rs1_read() {
            Some((_, v)) => v,
            None => 0,
        }
    } else {
        0
    };

    let right: i128 = if iflags[InstructionFlags::RightOperandIsImm] {
        cycle.imm()
    } else if iflags[InstructionFlags::RightOperandIsRs2Value] {
        match cycle.rs2_read() {
            Some((_, v)) => v as i128,
            None => 0,
        }
    } else {
        0
    };

    (left, right as u128)
}

// ISA dispatch tables (absorbed from jolt-zkvm/src/witness/flags.rs)

fn static_circuit_flags(instr: &Instruction) -> [bool; NUM_CIRCUIT_FLAGS] {
    use jolt_instructions::rv::{arithmetic, branch, compare, jump, load, logic, store, system};
    use jolt_instructions::virtual_::{
        advice, arithmetic as varith, assert as vassert, bitwise, byte, division, extension, shift,
        xor_rotate,
    };

    match instr {
        Instruction::ADD(_) => Flags::circuit_flags(&arithmetic::Add),
        Instruction::ADDI(_) => Flags::circuit_flags(&arithmetic::Addi),
        Instruction::SUB(_) => Flags::circuit_flags(&arithmetic::Sub),
        Instruction::LUI(_) => Flags::circuit_flags(&arithmetic::Lui),
        Instruction::AUIPC(_) => Flags::circuit_flags(&arithmetic::Auipc),
        Instruction::MUL(_) => Flags::circuit_flags(&arithmetic::Mul),
        Instruction::MULHU(_) => Flags::circuit_flags(&arithmetic::MulHU),
        Instruction::AND(_) => Flags::circuit_flags(&logic::And),
        Instruction::ANDI(_) => Flags::circuit_flags(&logic::AndI),
        Instruction::ANDN(_) => Flags::circuit_flags(&logic::Andn),
        Instruction::OR(_) => Flags::circuit_flags(&logic::Or),
        Instruction::ORI(_) => Flags::circuit_flags(&logic::OrI),
        Instruction::XOR(_) => Flags::circuit_flags(&logic::Xor),
        Instruction::XORI(_) => Flags::circuit_flags(&logic::XorI),
        Instruction::SLT(_) => Flags::circuit_flags(&compare::Slt),
        Instruction::SLTI(_) => Flags::circuit_flags(&compare::SltI),
        Instruction::SLTIU(_) => Flags::circuit_flags(&compare::SltIU),
        Instruction::SLTU(_) => Flags::circuit_flags(&compare::SltU),
        Instruction::BEQ(_) => Flags::circuit_flags(&branch::Beq),
        Instruction::BGE(_) => Flags::circuit_flags(&branch::Bge),
        Instruction::BGEU(_) => Flags::circuit_flags(&branch::BgeU),
        Instruction::BLT(_) => Flags::circuit_flags(&branch::Blt),
        Instruction::BLTU(_) => Flags::circuit_flags(&branch::BltU),
        Instruction::BNE(_) => Flags::circuit_flags(&branch::Bne),
        Instruction::JAL(_) => Flags::circuit_flags(&jump::Jal),
        Instruction::JALR(_) => Flags::circuit_flags(&jump::Jalr),
        Instruction::LD(_) => Flags::circuit_flags(&load::Ld),
        Instruction::SD(_) => Flags::circuit_flags(&store::Sd),
        Instruction::EBREAK(_) => Flags::circuit_flags(&system::Ebreak),
        Instruction::ECALL(_) => Flags::circuit_flags(&system::Ecall),
        Instruction::FENCE(_) => Flags::circuit_flags(&system::Fence),
        Instruction::VirtualAdvice(_) => Flags::circuit_flags(&advice::VirtualAdvice),
        Instruction::VirtualAdviceLen(_) => Flags::circuit_flags(&advice::VirtualAdviceLen),
        Instruction::VirtualAdviceLoad(_) => Flags::circuit_flags(&advice::VirtualAdviceLoad),
        Instruction::VirtualHostIO(_) => Flags::circuit_flags(&advice::VirtualHostIO),
        Instruction::VirtualMULI(_) => Flags::circuit_flags(&varith::MulI),
        Instruction::VirtualPow2(_) => Flags::circuit_flags(&varith::Pow2),
        Instruction::VirtualPow2I(_) => Flags::circuit_flags(&varith::Pow2I),
        Instruction::VirtualPow2W(_) => Flags::circuit_flags(&varith::Pow2W),
        Instruction::VirtualPow2IW(_) => Flags::circuit_flags(&varith::Pow2IW),
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
        Instruction::VirtualMovsign(_) => Flags::circuit_flags(&bitwise::MovSign),
        Instruction::VirtualRev8W(_) => Flags::circuit_flags(&byte::VirtualRev8W),
        Instruction::VirtualChangeDivisor(_) => {
            Flags::circuit_flags(&division::VirtualChangeDivisor)
        }
        Instruction::VirtualChangeDivisorW(_) => {
            Flags::circuit_flags(&division::VirtualChangeDivisorW)
        }
        Instruction::VirtualZeroExtendWord(_) => {
            Flags::circuit_flags(&extension::VirtualZeroExtendWord)
        }
        Instruction::VirtualSignExtendWord(_) => {
            Flags::circuit_flags(&extension::VirtualSignExtendWord)
        }
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
        Instruction::VirtualXORROT32(_) => Flags::circuit_flags(&xor_rotate::VirtualXorRot32),
        Instruction::VirtualXORROT24(_) => Flags::circuit_flags(&xor_rotate::VirtualXorRot24),
        Instruction::VirtualXORROT16(_) => Flags::circuit_flags(&xor_rotate::VirtualXorRot16),
        Instruction::VirtualXORROT63(_) => Flags::circuit_flags(&xor_rotate::VirtualXorRot63),
        Instruction::VirtualXORROTW16(_) => Flags::circuit_flags(&xor_rotate::VirtualXorRotW16),
        Instruction::VirtualXORROTW12(_) => Flags::circuit_flags(&xor_rotate::VirtualXorRotW12),
        Instruction::VirtualXORROTW8(_) => Flags::circuit_flags(&xor_rotate::VirtualXorRotW8),
        Instruction::VirtualXORROTW7(_) => Flags::circuit_flags(&xor_rotate::VirtualXorRotW7),
        Instruction::NoOp => {
            let mut flags = [false; NUM_CIRCUIT_FLAGS];
            flags[CircuitFlags::DoNotUpdateUnexpandedPC as usize] = true;
            flags
        }
        Instruction::INLINE(i) => panic!(
            "INLINE instruction reached CycleRow unexpanded: opcode={}, funct3={}, funct7={}",
            i.opcode, i.funct3, i.funct7
        ),
        _ => panic!("unsupported instruction: {instr:?}"),
    }
}

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
        Instruction::VirtualXORROTW16(_) => {
            Flags::instruction_flags(&xor_rotate::VirtualXorRotW16)
        }
        Instruction::VirtualXORROTW12(_) => {
            Flags::instruction_flags(&xor_rotate::VirtualXorRotW12)
        }
        Instruction::VirtualXORROTW8(_) => Flags::instruction_flags(&xor_rotate::VirtualXorRotW8),
        Instruction::VirtualXORROTW7(_) => Flags::instruction_flags(&xor_rotate::VirtualXorRotW7),
        Instruction::NoOp => {
            let mut flags = [false; NUM_INSTRUCTION_FLAGS];
            flags[InstructionFlags::IsNoop as usize] = true;
            flags
        }
        Instruction::INLINE(i) => panic!(
            "INLINE instruction reached CycleRow unexpanded: opcode={}, funct3={}, funct7={}",
            i.opcode, i.funct3, i.funct7
        ),
        _ => panic!("unsupported instruction: {instr:?}"),
    }
}

fn apply_dynamic_circuit_flags(flags: &mut [bool; NUM_CIRCUIT_FLAGS], norm: &NormalizedInstruction) {
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
    if norm.virtual_sequence_remaining == Some(0) {
        flags[CircuitFlags::IsLastInSequence as usize] = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noop_trait_methods() {
        let noop = Cycle::noop();
        assert!(noop.is_noop());
        assert_eq!(noop.unexpanded_pc(), 0);
        assert!(noop.ram_access_address().is_none());
        assert!(noop.rs1_read().is_none());
        assert!(noop.rd_write().is_none());

        let cflags = CycleRow::circuit_flags(&noop);
        assert!(cflags[CircuitFlags::DoNotUpdateUnexpandedPC as usize]);

        let iflags = CycleRow::instruction_flags(&noop);
        assert!(iflags[InstructionFlags::IsNoop as usize]);
    }

    #[test]
    fn noop_lookup_index_is_zero() {
        let noop = Cycle::noop();
        assert_eq!(noop.lookup_index(), 0);
    }
}
