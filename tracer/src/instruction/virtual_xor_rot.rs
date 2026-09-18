use serde::{Deserialize, Serialize};

use super::{RISCVInstruction, RISCVTrace};
use crate::instruction::format::format_virtual_xor_rot::FormatVirtualXorRot;
use crate::{declare_riscv_instr, emulator::cpu::Cpu};

declare_riscv_instr!(
    name = VirtualXORROT,
    mask = 0,
    match = 0,
    format = FormatVirtualXorRot,
    ram = ()
);

impl VirtualXORROT {
    fn exec(&self, cpu: &mut Cpu, _: &mut <VirtualXORROT as RISCVInstruction>::RAMAccess) {
        let xor_result = cpu.x[self.operands.rs1 as usize] ^ cpu.x[self.operands.rs2 as usize];
        let rotated = xor_result.rotate_right(self.operands.rotation);
        cpu.write_register(self.operands.rd as usize, rotated);
    }
}

impl RISCVTrace for VirtualXORROT {}
