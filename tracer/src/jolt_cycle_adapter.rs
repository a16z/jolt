use jolt_riscv::{JoltCycle, JoltInstructionRowData, RamAccess};

use crate::instruction::{registers::InstructionRegisterState, RISCVCycle, RISCVInstruction};

impl<T: RISCVInstruction + JoltInstructionRowData> JoltCycle for RISCVCycle<T> {
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
        let ram_access: RamAccess = self.ram_access.into();
        match ram_access {
            RamAccess::Read(r) => Some(r.address),
            RamAccess::Write(w) => Some(w.address),
            RamAccess::NoOp => None,
        }
    }

    fn ram_read_value(&self) -> Option<u64> {
        let ram_access: RamAccess = self.ram_access.into();
        match ram_access {
            RamAccess::Read(r) => Some(r.value),
            RamAccess::Write(w) => Some(w.pre_value),
            RamAccess::NoOp => None,
        }
    }

    fn ram_write_value(&self) -> Option<u64> {
        let ram_access: RamAccess = self.ram_access.into();
        match ram_access {
            RamAccess::Read(r) => Some(r.value),
            RamAccess::Write(w) => Some(w.post_value),
            RamAccess::NoOp => None,
        }
    }
}
