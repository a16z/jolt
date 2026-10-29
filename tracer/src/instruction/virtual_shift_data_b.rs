use crate::instruction::registers::shift_data::RegisterStateShiftData;
use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{format::format_shift_data::FormatShiftData, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name = VirtualShiftDataB,
    mask = 0,
    match = 0,
    format = FormatShiftData,
    registers = RegisterStateShiftData,
    ram = ()
);

impl VirtualShiftDataB {
    fn exec(&self, cpu: &mut Cpu, _: &mut <VirtualShiftDataB as RISCVInstruction>::RamAccess) {
        // Store byte moved into its lane within the containing doubleword:
        // rs1 holds the store value, the effective address is rs2 + imm.
        let x = cpu.x[self.operands.rs1 as usize] as u64;
        let ea = (cpu.x[self.operands.rs2 as usize] as u64).wrapping_add(self.operands.imm as u64);
        let v = (x & 0xFF) << (8 * (ea & 7));
        cpu.write_register(self.operands.rd as usize, v as i64);
    }
}

impl RISCVTrace for VirtualShiftDataB {}
