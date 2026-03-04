use crate::utils::inline_helpers::InstrAssembler;
use crate::utils::virtual_registers::VirtualRegisterAllocator;
use serde::{Deserialize, Serialize};

use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};

use super::{
    add::ADD, andi::ANDI, format::format_r::FormatR, mul::MUL, mulhu::MULHU, sltu::SLTU,
    virtual_movsign::VirtualMovsign, xor::XOR, Cycle, Instruction, RISCVInstruction, RISCVTrace,
};

declare_riscv_instr!(
    name   = MULHSU,
    mask   = 0xfe00707f,
    match  = 0x02002033,
    format = FormatR,
    ram    = ()
);

impl MULHSU {
    fn exec(&self, cpu: &mut Cpu, _: &mut <MULHSU as RISCVInstruction>::RAMAccess) {
        cpu.write_register(
            self.operands.rd as usize,
            match cpu.xlen {
                Xlen::Bit32 => cpu.sign_extend(
                    cpu.x[self.operands.rs1 as usize]
                        .wrapping_mul(cpu.x[self.operands.rs2 as usize] as u32 as i64)
                        >> 32,
                ),
                Xlen::Bit64 => {
                    ((cpu.x[self.operands.rs1 as usize] as u128)
                        .wrapping_mul(cpu.x[self.operands.rs2 as usize] as u64 as u128)
                        >> 64) as i64
                }
            },
        );
    }
}

impl RISCVTrace for MULHSU {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        let inline_sequence = self.inline_sequence(&cpu.vr_allocator, cpu.xlen);
        let mut trace = trace;
        for instr in inline_sequence {
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    fn inline_sequence(
        &self,
        allocator: &VirtualRegisterAllocator,
        xlen: Xlen,
    ) -> Vec<Instruction> {
        let v0 = allocator.allocate();
        let v1 = allocator.allocate();
        let v2 = allocator.allocate();
        let v3 = allocator.allocate();

        let mut asm = InstrAssembler::new(self.address, self.is_compressed, xlen, allocator);

        asm.emit_i::<VirtualMovsign>(*v0, self.operands.rs1, 0);
        asm.emit_i::<ANDI>(*v1, *v0, 1);
        asm.emit_r::<XOR>(*v2, self.operands.rs1, *v0);
        asm.emit_r::<ADD>(*v2, *v2, *v1);
        asm.emit_r::<MULHU>(*v3, *v2, self.operands.rs2);
        asm.emit_r::<MUL>(*v2, *v2, self.operands.rs2);
        asm.emit_r::<XOR>(*v3, *v3, *v0);
        asm.emit_r::<XOR>(*v2, *v2, *v0);
        asm.emit_r::<ADD>(*v0, *v2, *v1);
        asm.emit_r::<SLTU>(*v0, *v0, *v2);
        asm.emit_r::<ADD>(self.operands.rd, *v3, *v0);

        asm.finalize()
    }
}
