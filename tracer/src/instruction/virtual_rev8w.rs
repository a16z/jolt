use serde::{Deserialize, Serialize};

use super::{format::format_i::FormatI, RISCVInstruction, RISCVTrace};
use crate::{
    declare_riscv_instr,
    emulator::cpu::{Cpu, Xlen},
};

declare_riscv_instr!(
    name = VirtualRev8W,
    mask = 0,
    match = 0,
    format = FormatI,
    ram = ()
);

impl VirtualRev8W {
    fn exec(&self, cpu: &mut Cpu, _: &mut <VirtualRev8W as RISCVInstruction>::RAMAccess) {
        match cpu.xlen {
            Xlen::Bit64 => {
                let v = cpu.x[self.operands.rs1 as usize] as u64;
                cpu.x[self.operands.rd as usize] = rev8w(v) as i64;
            }
            Xlen::Bit32 => unimplemented!(),
        }
    }
}

impl RISCVTrace for VirtualRev8W {}

/// Reverses the bytes in each 32-bit word.
#[inline]
pub fn rev8w(v: u64) -> u64 {
    let lo = (v as u32).swap_bytes();
    let hi = ((v >> 32) as u32).swap_bytes();
    lo as u64 + ((hi as u64) << 32)
}
