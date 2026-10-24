use crate::instruction::registers::r::RegisterStateR;
use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{format::format_r::FormatR, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name = VirtualFunnelHigh,
    mask = 0,
    match = 0,
    format = FormatR,
    registers = RegisterStateR,
    ram = ()
);

impl VirtualFunnelHigh {
    fn exec(&self, cpu: &mut Cpu, _: &mut <VirtualFunnelHigh as RISCVInstruction>::RamAccess) {
        let x = cpu.x[self.operands.rs1 as usize] as u64;
        let y = cpu.x[self.operands.rs2 as usize] as u64;
        cpu.write_register(self.operands.rd as usize, funnel_high(x, y) as i64);
    }
}

impl RISCVTrace for VirtualFunnelHigh {}

/// Moves each bit of `x` at a zero of `y` up by the number of ones of `y`
/// above it; for `y = !0 << s` this is `(x mod 2^s) << (64 − s)`.
fn funnel_high(x: u64, y: u64) -> u64 {
    let mut ones_above = 0;
    let mut result = 0;
    for i in (0..64).rev() {
        if (y >> i) & 1 == 1 {
            ones_above += 1;
        } else {
            result |= ((x >> i) & 1) << (i + ones_above);
        }
    }
    result
}
