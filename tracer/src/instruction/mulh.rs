use crate::utils::inline_helpers::InstrAssembler;
use crate::utils::virtual_registers::VirtualRegisterAllocator;
use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};

use super::{
    add::ADD, format::format_r::FormatR, mul::MUL, mulhu::MULHU, virtual_movsign::VirtualMovsign,
    Cycle, Instruction, RISCVInstruction, RISCVTrace,
};

declare_riscv_instr!(
    name   = MULH,
    mask   = 0xfe00707f,
    match  = 0x02001033,
    format = FormatR,
    ram    = ()
);

impl MULH {
    fn exec(&self, cpu: &mut Cpu, _: &mut <MULH as RISCVInstruction>::RAMAccess) {
        cpu.x[self.operands.rd as usize] = match cpu.xlen {
            Xlen::Bit32 => cpu.sign_extend(
                (cpu.x[self.operands.rs1 as usize] * cpu.x[self.operands.rs2 as usize]) >> 32,
            ),
            Xlen::Bit64 => {
                (((cpu.x[self.operands.rs1 as usize] as i128)
                    * (cpu.x[self.operands.rs2 as usize] as i128))
                    >> 64) as i64
            }
        };
    }
}

impl RISCVTrace for MULH {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        let inline_sequence = self.inline_sequence(&cpu.vr_allocator, cpu.xlen);
        let mut trace = trace;
        for instr in inline_sequence {
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    /// MULH computes the high 64 bits of signed multiplication using identity:
    /// (x × y)_high = (x × y)_high_unsigned + sign(x) × y + sign(y) × x
    fn inline_sequence(
        &self,
        allocator: &VirtualRegisterAllocator,
        xlen: Xlen,
    ) -> Vec<Instruction> {
        let v_sx = allocator.allocate();
        let v_sy = allocator.allocate();
        let v_0 = allocator.allocate();

        let mut asm = InstrAssembler::new(self.address, self.is_compressed, xlen, allocator);

        asm.emit_i::<VirtualMovsign>(*v_sx, self.operands.rs1, 0);
        asm.emit_i::<VirtualMovsign>(*v_sy, self.operands.rs2, 0);
        asm.emit_r::<MULHU>(*v_0, self.operands.rs1, self.operands.rs2);
        asm.emit_r::<MUL>(*v_sx, *v_sx, self.operands.rs2);
        asm.emit_r::<MUL>(*v_sy, *v_sy, self.operands.rs1);
        asm.emit_r::<ADD>(*v_0, *v_0, *v_sx);
        asm.emit_r::<ADD>(self.operands.rd, *v_0, *v_sy);

        asm.finalize()
    }
}
