//! `JoltInstruction`: the static, encoding-time view of a RISC-V instruction
//! used by Jolt's lookup-query layer.
//!
//! The trait is implemented blanket-style for every `tracer::RISCVInstruction`
//! by delegating through `NormalizedInstruction`, so any concrete instruction
//! type defined in `tracer` automatically satisfies it.

use tracer::instruction::{Instruction, NormalizedInstruction, RISCVInstruction};

/// Static instruction view: encoding-time data.
pub trait JoltInstruction {
    /// True if this cycle is a no-op (padding).
    fn is_noop(&self) -> bool;

    /// Program-counter address where this instruction lives.
    fn address(&self) -> u64;

    /// Sign-extended immediate, or `0` if the instruction has none.
    fn imm(&self) -> i128;

    /// rs1 register index, or `None` if unused.
    fn rs1(&self) -> Option<u8>;

    /// rs2 register index, or `None` if unused.
    fn rs2(&self) -> Option<u8>;

    /// rd register index, or `None` if unused.
    fn rd(&self) -> Option<u8>;

    /// Remaining steps in a virtual instruction sequence, or `None` if
    /// this is a real (non-virtual) instruction.
    fn virtual_sequence_remaining(&self) -> Option<u16>;

    /// True if this is the first instruction in a virtual sequence.
    fn is_first_in_sequence(&self) -> bool;

    /// True if this is a virtual (expanded) instruction.
    fn is_virtual(&self) -> bool;

    /// True if this is a compressed (C-extension) instruction.
    fn is_compressed(&self) -> bool;
}

impl<T: RISCVInstruction> JoltInstruction for T {
    fn is_noop(&self) -> bool {
        matches!((*self).into(), Instruction::NoOp)
    }

    fn address(&self) -> u64 {
        let normalized: NormalizedInstruction = (*self).into();
        normalized.address as u64
    }

    fn imm(&self) -> i128 {
        let normalized: NormalizedInstruction = (*self).into();
        normalized.operands.imm
    }

    fn rs1(&self) -> Option<u8> {
        let normalized: NormalizedInstruction = (*self).into();
        normalized.operands.rs1
    }

    fn rs2(&self) -> Option<u8> {
        let normalized: NormalizedInstruction = (*self).into();
        normalized.operands.rs2
    }

    fn rd(&self) -> Option<u8> {
        let normalized: NormalizedInstruction = (*self).into();
        normalized.operands.rd
    }

    fn virtual_sequence_remaining(&self) -> Option<u16> {
        let normalized: NormalizedInstruction = (*self).into();
        normalized.virtual_sequence_remaining
    }

    fn is_first_in_sequence(&self) -> bool {
        let normalized: NormalizedInstruction = (*self).into();
        normalized.is_first_in_sequence
    }

    fn is_virtual(&self) -> bool {
        JoltInstruction::virtual_sequence_remaining(self).is_some()
    }

    fn is_compressed(&self) -> bool {
        let normalized: NormalizedInstruction = (*self).into();
        normalized.is_compressed
    }
}
