use serde::{Deserialize, Serialize};

use super::add::ADD;
use super::amo::{amo_post32, amo_post64, amo_pre32, amo_pre64};
use super::mul::MUL;
use super::sltu::SLTU;
use super::virtual_extend::VirtualExtend;
use super::virtual_move::VirtualMove;
use super::xori::XORI;
use super::RV32IMInstruction;
use crate::utils::inline_helpers::InstrAssembler;
use crate::utils::virtual_registers::allocate_virtual_register;
use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};

use super::{format::format_r::FormatR, RISCVInstruction, RISCVTrace, RV32IMCycle};

declare_riscv_instr!(
    name   = AMOMAXUW,
    mask   = 0xf800707f,
    match  = 0xe000202f,
    format = FormatR,
    ram    = ()
);

impl AMOMAXUW {
    fn exec(&self, cpu: &mut Cpu, _: &mut <AMOMAXUW as RISCVInstruction>::RAMAccess) {
        let address = cpu.x[self.operands.rs1 as usize] as u64;
        let compare_value = cpu.x[self.operands.rs2 as usize] as u32;

        // Load the original word from memory
        let load_result = cpu.mmu.load_word(address);
        let original_value = match load_result {
            Ok((word, _)) => word as i32 as i64,
            Err(_) => panic!("MMU load error"),
        };

        // Find the maximum (unsigned comparison) and store back to memory
        let new_value = if (original_value as u32) >= compare_value {
            original_value as u32
        } else {
            compare_value
        };
        cpu.mmu
            .store_word(address, new_value)
            .expect("MMU store error");

        // Return the original value (sign extended)
        cpu.x[self.operands.rd as usize] = original_value;
    }
}

impl RISCVTrace for AMOMAXUW {
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<RV32IMCycle>>) {
        let inline_sequence = self.inline_sequence(cpu.xlen);
        let mut trace = trace;
        for instr in inline_sequence {
            // In each iteration, create a new Option containing a re-borrowed reference
            instr.trace(cpu, trace.as_deref_mut());
        }
    }

    fn inline_sequence(&self, xlen: Xlen) -> Vec<RV32IMInstruction> {
        let v_rd = allocate_virtual_register();
        let v_rs2 = allocate_virtual_register();
        let v_sel_rs2 = allocate_virtual_register();
        let v_sel_rd = allocate_virtual_register();
        let v_tmp = allocate_virtual_register();

        let mut asm = InstrAssembler::new(self.address, self.is_compressed, xlen);

        match xlen {
            Xlen::Bit32 => {
                amo_pre32(&mut asm, self.operands.rs1, *v_rd);
                asm.emit_i::<VirtualMove>(*v_rs2, self.operands.rs2, 0);
                asm.emit_i::<VirtualMove>(*v_tmp, *v_rd, 0);
                asm.emit_r::<SLTU>(*v_sel_rs2, *v_tmp, *v_rs2);
                asm.emit_i::<XORI>(*v_sel_rd, *v_sel_rs2, 1);
                asm.emit_r::<MUL>(*v_rs2, *v_sel_rs2, self.operands.rs2);
                asm.emit_r::<MUL>(*v_tmp, *v_sel_rd, *v_rd);
                asm.emit_r::<ADD>(*v_rs2, *v_tmp, *v_rs2);
                amo_post32(&mut asm, *v_rs2, self.operands.rs1, self.operands.rd, *v_rd);
            }
            Xlen::Bit64 => {
                let v_mask = allocate_virtual_register();
                let v_dword_address = allocate_virtual_register();
                let v_dword = allocate_virtual_register();
                let v_word = allocate_virtual_register();
                let v_shift = allocate_virtual_register();

                amo_pre64(
                    &mut asm,
                    self.operands.rs1,
                    *v_rd,
                    *v_dword_address,
                    *v_dword,
                    *v_shift,
                );
                asm.emit_i::<VirtualExtend>(*v_rs2, self.operands.rs2, 0);
                asm.emit_i::<VirtualExtend>(*v_tmp, *v_rd, 0);
                asm.emit_r::<SLTU>(*v_sel_rs2, *v_tmp, *v_rs2);
                asm.emit_i::<XORI>(*v_sel_rd, *v_sel_rs2, 1);
                asm.emit_r::<MUL>(*v_rs2, *v_sel_rs2, self.operands.rs2);
                asm.emit_r::<MUL>(*v_tmp, *v_sel_rd, *v_rd);
                asm.emit_r::<ADD>(*v_rs2, *v_tmp, *v_rs2);
                amo_post64(
                    &mut asm,
                    *v_rs2,
                    *v_dword_address,
                    *v_dword,
                    *v_shift,
                    *v_mask,
                    *v_word,
                    self.operands.rd,
                    *v_rd,
                );
            }
        }

        asm.finalize()
    }
}
