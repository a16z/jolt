use jolt_riscv::JoltInstruction;
use jolt_trace::JoltCycle;

#[cfg(any(feature = "test-utils", test))]
use crate::instruction::format::InstructionFormat;

#[cfg(any(feature = "test-utils", test))]
use crate::instruction::NormalizedInstruction;

use crate::instruction::{
    format::InstructionRegisterState, RAMAccess, RISCVCycle, RISCVInstruction,
};

impl<T: RISCVInstruction + JoltInstruction> JoltCycle for RISCVCycle<T> {
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

    #[cfg(any(feature = "test-utils", test))]
    fn random(rng: &mut rand::rngs::StdRng) -> Self {
        let instruction = T::random(rng);
        let normalized: NormalizedInstruction = instruction.into();
        let register_state =
            <<T::Format as InstructionFormat>::RegisterState as InstructionRegisterState>::random(
                rng,
                &normalized.operands,
            );
        // RAM access is left at the default (no-op) state. Coverage gap:
        // any lookup logic that depends on RAM values needs a richer
        // generator in tracer.
        Self {
            instruction,
            register_state,
            ram_access: T::RAMAccess::default(),
        }
    }
}
