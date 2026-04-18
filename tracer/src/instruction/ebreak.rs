//! EBREAK — Breakpoint / program termination.
//!
//! Encoding: 0x00100073 (SYSTEM opcode, funct3=000, imm=1)
//!
//! In a zkVM context without a debugger, EBREAK serves as a termination point.
//! The emulator detects termination when PC doesn't change (prev_pc == pc).

use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
    utils::inline_helpers::InstrAssembler,
    utils::virtual_registers::VirtualRegisterAllocator,
};

use super::{
    format::format_i::FormatI, jal::JAL, Cycle, Instruction, RISCVInstruction, RISCVTrace,
};

declare_riscv_instr!(
    name   = EBREAK,
    mask   = 0xffffffff,  // Exact match
    match  = 0x00100073,  // EBREAK encoding
    format = FormatI,
    ram    = (),
    side_effects = true
);

impl EBREAK {
    fn exec(&self, cpu: &mut Cpu, _: &mut <EBREAK as RISCVInstruction>::RAMAccess) {
        // Don't advance PC - emulator will detect prev_pc == pc and terminate
        cpu.pc = self.address;
    }
}

impl RISCVTrace for EBREAK {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        let inline_sequence = self.inline_sequence(&cpu.vr_allocator, cpu.xlen);
        let mut trace = trace;
        for instr in inline_sequence {
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    /// Expand EBREAK into a JAL-to-self (`j .`), which stalls the PC and
    /// terminates execution. Using JAL gives the cycle a Jump flag, which
    /// disables the NextUnexpPCUpdateOtherwise constraint at the NoOp
    /// padding boundary.
    fn inline_sequence(
        &self,
        allocator: &VirtualRegisterAllocator,
        xlen: Xlen,
    ) -> Vec<Instruction> {
        let mut asm = InstrAssembler::new(self.address, self.is_compressed, xlen, allocator);
        let vr = allocator.allocate();
        asm.emit_j::<JAL>(*vr, 0);
        drop(vr);
        asm.finalize()
    }
}
