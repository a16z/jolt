//! `CycleRow` implementation for `tracer::Cycle`.

use tracer::instruction::{Cycle, RAMAccess};

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
        Cycle::rs1_read(self)
    }

    fn rs2_read(&self) -> Option<(u8, u64)> {
        Cycle::rs2_read(self)
    }

    fn rd_write(&self) -> Option<(u8, u64, u64)> {
        Cycle::rd_write(self)
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noop_trait_methods() {
        let noop = <Cycle as CycleRow>::noop();
        assert!(noop.is_noop());
        assert_eq!(noop.unexpanded_pc(), 0);
        assert!(noop.ram_access_address().is_none());
        assert!(noop.rs1_read().is_none());
        assert!(noop.rd_write().is_none());
    }
}
