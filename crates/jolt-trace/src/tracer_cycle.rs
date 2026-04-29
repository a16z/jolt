//! `JoltInstruction` and `JoltCycle` adapters for tracer's `Instruction` and
//! `Cycle` enums. The blanket impl in `traits.rs` then provides
//! `JoltInstruction` for `Cycle` automatically by delegating through
//! `Cycle::instruction()`.

use tracer::instruction::{
    format::InstructionRegisterState, NormalizedInstruction, RAMAccess, RISCVCycle,
    RISCVInstruction,
};

use crate::traits::{JoltCycle, JoltInstruction};

impl<T: RISCVInstruction> JoltInstruction for T {
    fn is_noop(&self) -> bool {
        todo!()
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
}

impl<T: RISCVInstruction> JoltCycle for RISCVCycle<T> {
    type Instruction = T;

    fn instruction(&self) -> T {
        self.instruction
    }

    fn rs1_val(&self) -> Option<u64> {
        self.register_state.rs1_value()
    }

    fn rs2_val(&self) -> Option<u64> {
        self.register_state.rs2_value()
    }

    fn rd_vals(&self) -> Option<(u64, u64)> {
        self.register_state.rd_values()
    }

    fn ram_access_address(&self) -> Option<u64> {
        let ram_access: RAMAccess = self.ram_access.into();
        match ram_access {
            RAMAccess::Read(r) => Some(r.address),
            RAMAccess::Write(w) => Some(w.address),
            RAMAccess::NoOp => None,
        }
    }

    fn ram_read_value(&self) -> Option<u64> {
        let ram_access: RAMAccess = self.ram_access.into();
        match ram_access {
            RAMAccess::Read(r) => Some(r.value),
            RAMAccess::Write(w) => Some(w.pre_value),
            RAMAccess::NoOp => None,
        }
    }

    fn ram_write_value(&self) -> Option<u64> {
        let ram_access: RAMAccess = self.ram_access.into();
        match ram_access {
            RAMAccess::Read(r) => Some(r.value),
            RAMAccess::Write(w) => Some(w.post_value),
            RAMAccess::NoOp => None,
        }
    }
}
