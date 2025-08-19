use serde::{Deserialize, Serialize};

use super::lw::LW;
use super::sw::SW;
use super::virtual_move::VirtualMove;
use super::xor::XOR;
use super::RV32IMInstruction;
use crate::utils::inline_helpers::InstrAssembler;
use crate::utils::virtual_registers::allocate_virtual_register;
use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};

use super::{format::format_r::FormatR, RISCVInstruction, RISCVTrace, RV32IMCycle};

declare_riscv_instr!(
    name   = AMOXORW,
    mask   = 0xf800707f,
    match  = 0x2000202f,
    format = FormatR,
    ram    = ()
);

impl AMOXORW {
    fn exec(&self, cpu: &mut Cpu, _: &mut <AMOXORW as RISCVInstruction>::RAMAccess) {
        let address = cpu.x[self.operands.rs1 as usize] as u64;
        let xor_value = cpu.x[self.operands.rs2 as usize] as u32;

        // Load the original word from memory
        let load_result = cpu.mmu.load_word(address);
        let original_value = match load_result {
            Ok((word, _)) => word as i32 as i64,
            Err(_) => panic!("MMU load error"),
        };

        // XOR the values and store back to memory
        let new_value = (original_value as u32) ^ xor_value;
        cpu.mmu
            .store_word(address, new_value)
            .expect("MMU store error");

        // Return the original value
        cpu.x[self.operands.rd as usize] = original_value;
    }
}

impl RISCVTrace for AMOXORW {
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

        let mut asm = InstrAssembler::new(self.address, self.is_compressed, xlen);
        asm.emit_ld::<LW>(*v_rd, self.operands.rs1, 0);
        asm.emit_r::<XOR>(*v_rs2, *v_rd, self.operands.rs2);
        asm.emit_s::<SW>(self.operands.rs1, *v_rs2, 0);
        asm.emit_i::<VirtualMove>(self.operands.rd, *v_rd, 0);
        asm.finalize()
    }
}
