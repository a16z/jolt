use crate::utils::inline_helpers::InstrAssembler;
use crate::utils::virtual_registers::allocate_virtual_register;
use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};

use super::{
    add::ADD, format::format_r::FormatR, mul::MUL, mulhu::MULHU, virtual_movsign::VirtualMovsign,
    RISCVInstruction, RISCVTrace, RV32IMCycle, RV32IMInstruction,
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
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<RV32IMCycle>>) {
        let inline_sequence = self.inline_sequence(cpu.xlen);
        let mut trace = trace;
        for instr in inline_sequence {
            // In each iteration, create a new Option containing a re-borrowed reference
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    fn inline_sequence(&self, xlen: Xlen) -> Vec<RV32IMInstruction> {
        let v_sx = allocate_virtual_register();
        let v_sy = allocate_virtual_register();
        let v_0 = allocate_virtual_register();
        let v_1 = allocate_virtual_register();
        let v_2 = allocate_virtual_register();
        let v_3 = allocate_virtual_register();

        let mut asm = InstrAssembler::new(self.address, self.is_compressed, xlen);
        asm.emit_i::<VirtualMovsign>(*v_sx, self.operands.rs1, 0);
        asm.emit_i::<VirtualMovsign>(*v_sy, self.operands.rs2, 0);
        asm.emit_r::<MULHU>(*v_0, self.operands.rs1, self.operands.rs2);
        asm.emit_r::<MUL>(*v_1, *v_sx, self.operands.rs2);
        asm.emit_r::<MUL>(*v_2, *v_sy, self.operands.rs1);
        asm.emit_r::<ADD>(*v_3, *v_0, *v_1);
        asm.emit_r::<ADD>(self.operands.rd, *v_3, *v_2);
        asm.finalize()
    }
}
