use crate::utils::inline_helpers::InstrAssembler;
use crate::utils::virtual_registers::allocate_virtual_register;
use serde::{Deserialize, Serialize};

use super::add::ADD;
use super::ld::LD;
use super::mul::MUL;
use super::sd::SD;
use super::slt::SLT;
use super::virtual_move::VirtualMove;
use super::xori::XORI;
use super::RV32IMInstruction;
use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};

use super::{format::format_r::FormatR, RISCVInstruction, RISCVTrace, RV32IMCycle};

declare_riscv_instr!(
    name   = AMOMAXD,
    mask   = 0xf800707f,
    match  = 0xa000302f,
    format = FormatR,
    ram    = ()
);

impl AMOMAXD {
    fn exec(&self, cpu: &mut Cpu, _: &mut <AMOMAXD as RISCVInstruction>::RAMAccess) {
        let address = cpu.x[self.operands.rs1 as usize] as u64;
        let compare_value = cpu.x[self.operands.rs2 as usize];

        // Load the original doubleword from memory
        let load_result = cpu.mmu.load_doubleword(address);
        let original_value = match load_result {
            Ok((doubleword, _)) => doubleword as i64,
            Err(_) => panic!("MMU load error"),
        };

        // Find the maximum and store back to memory
        let new_value = if original_value >= compare_value {
            original_value
        } else {
            compare_value
        };
        cpu.mmu
            .store_doubleword(address, new_value as u64)
            .expect("MMU store error");

        // Return the original value
        cpu.x[self.operands.rd as usize] = original_value;
    }
}

impl RISCVTrace for AMOMAXD {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<RV32IMCycle>>) {
        let inline_sequence = self.inline_sequence(cpu.xlen);
        let mut trace = trace;
        for instr in inline_sequence {
            // In each iteration, create a new Option containing a re-borrowed reference
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    fn inline_sequence(&self, xlen: Xlen) -> Vec<RV32IMInstruction> {
        let v_rs2 = allocate_virtual_register();
        let v_rd = allocate_virtual_register();
        let v_sel_rs2 = allocate_virtual_register();
        let v_sel_rd = allocate_virtual_register();
        let v_tmp = allocate_virtual_register();

        let mut asm = InstrAssembler::new(self.address, self.is_compressed, xlen, false);
        asm.emit_ld::<LD>(*v_rd, self.operands.rs1, 0);
        asm.emit_r::<SLT>(*v_sel_rs2, *v_rd, self.operands.rs2);
        asm.emit_i::<XORI>(*v_sel_rd, *v_sel_rs2, 1);
        asm.emit_r::<MUL>(*v_rs2, *v_sel_rs2, self.operands.rs2);
        asm.emit_r::<MUL>(*v_tmp, *v_sel_rd, *v_rd);
        asm.emit_r::<ADD>(*v_rs2, *v_tmp, *v_rs2);
        asm.emit_s::<SD>(self.operands.rs1, *v_rs2, 0);
        asm.emit_i::<VirtualMove>(self.operands.rd, *v_rd, 0);
        asm.finalize()
    }
}
