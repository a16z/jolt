use crate::utils::inline_helpers::InstrAssembler;
use crate::utils::virtual_registers::VirtualRegisterAllocator;
use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};

use super::{
    format::format_r::FormatR, virtual_shift_right_bitmask::VirtualShiftRightBitmask,
    virtual_srl::VirtualSRL, RISCVInstruction, RISCVTrace, RV32IMCycle, RV32IMInstruction,
};

declare_riscv_instr!(
    name   = SRL,
    mask   = 0xfe00707f,
    match  = 0x00005033,
    format = FormatR,
    ram    = ()
);

impl SRL {
    fn exec(&self, cpu: &mut Cpu, _: &mut <SRL as RISCVInstruction>::RAMAccess) {
        let mask = match cpu.xlen {
            Xlen::Bit32 => 0x1f,
            Xlen::Bit64 => 0x3f,
        };
        cpu.x[self.operands.rd as usize] = cpu.sign_extend(
            cpu.unsigned_data(cpu.x[self.operands.rs1 as usize])
                .wrapping_shr(cpu.x[self.operands.rs2 as usize] as u32 & mask) as i64,
        );
    }
}

impl RISCVTrace for SRL {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<RV32IMCycle>>) {
        let inline_sequence = self.inline_sequence(&cpu.vr_allocator, cpu.xlen);
        let mut trace = trace;
        for instr in inline_sequence {
            // In each iteration, create a new Option containing a re-borrowed reference
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    fn inline_sequence(
        &self,
        allocator: &VirtualRegisterAllocator,
        xlen: Xlen,
    ) -> Vec<RV32IMInstruction> {
        let v_bitmask = allocator.allocate();

        let mut asm = InstrAssembler::new(self.address, self.is_compressed, xlen, allocator);
        asm.emit_i::<VirtualShiftRightBitmask>(*v_bitmask, self.operands.rs2, 0);
        asm.emit_vshift_r::<VirtualSRL>(self.operands.rd, self.operands.rs1, *v_bitmask);
        asm.finalize()
    }
}
