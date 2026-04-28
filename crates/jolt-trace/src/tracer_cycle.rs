//! `JoltInstruction` and `JoltCycle` adapters for tracer's `Instruction` and
//! `Cycle` enums. The blanket impl in `traits.rs` then provides
//! `JoltInstruction` for `Cycle` automatically by delegating through
//! `Cycle::instruction()`.

use tracer::instruction::{Cycle, Instruction, RAMAccess};

use crate::traits::{JoltCycle, JoltInstruction};

impl JoltInstruction for Instruction {
    fn is_noop(&self) -> bool {
        matches!(self, Instruction::NoOp)
    }

    fn address(&self) -> u64 {
        match self {
            Instruction::NoOp => 0,
            _ => self.normalize().address as u64,
        }
    }

    fn imm(&self) -> i128 {
        match self {
            Instruction::NoOp => 0,
            _ => self.normalize().operands.imm,
        }
    }

    fn rs1(&self) -> Option<u8> {
        match self {
            Instruction::NoOp => None,
            _ => self.normalize().operands.rs1,
        }
    }

    fn rs2(&self) -> Option<u8> {
        match self {
            Instruction::NoOp => None,
            _ => self.normalize().operands.rs2,
        }
    }

    fn rd(&self) -> Option<u8> {
        match self {
            Instruction::NoOp => None,
            _ => self.normalize().operands.rd,
        }
    }

    fn virtual_sequence_remaining(&self) -> Option<u16> {
        match self {
            Instruction::NoOp => None,
            _ => self.normalize().virtual_sequence_remaining,
        }
    }

    fn is_first_in_sequence(&self) -> bool {
        match self {
            Instruction::NoOp => false,
            _ => self.normalize().is_first_in_sequence,
        }
    }

    fn is_virtual(&self) -> bool {
        JoltInstruction::virtual_sequence_remaining(self).is_some()
    }
}

impl JoltCycle for Cycle {
    type Instruction = Instruction;

    fn instruction(&self) -> Instruction {
        Cycle::instruction(self)
    }

    fn rs1_val(&self) -> Option<u64> {
        Cycle::rs1_read(self).map(|(_, v)| v)
    }

    fn rs2_val(&self) -> Option<u64> {
        Cycle::rs2_read(self).map(|(_, v)| v)
    }

    fn rd_vals(&self) -> Option<(u64, u64)> {
        Cycle::rd_write(self).map(|(_, pre, post)| (pre, post))
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noop_cycle_traits() {
        let noop = Cycle::NoOp;
        // `JoltInstruction` for `Cycle` comes from the blanket impl in `traits.rs`.
        assert!(JoltInstruction::is_noop(&noop));
        assert_eq!(JoltInstruction::address(&noop), 0);
        assert!(noop.ram_access_address().is_none());
        assert!(noop.rs1_val().is_none());
        assert!(noop.rd_vals().is_none());
    }
}
