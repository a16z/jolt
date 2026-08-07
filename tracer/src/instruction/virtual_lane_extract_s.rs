use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{format::format_r::FormatR, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name = VirtualLaneExtractS,
    mask = 0,
    match = 0,
    format = FormatR,
    ram = ()
);

pub(crate) fn signed_extract(data: u64, mask: u64) -> u64 {
    let mut packed = 0u128;
    let mut top_count = 0u128;
    let mut signed_weight = 0u128;
    let mut previous_mask = 0u128;
    for bit in (0..64).rev() {
        let x_i = u128::from((data >> bit) & 1);
        let y_i = u128::from((mask >> bit) & 1);
        packed = packed * (1 + y_i) + x_i * y_i;
        let top = x_i * y_i * (1 - previous_mask);
        top_count += top;
        signed_weight = signed_weight * (1 + y_i) + top;
        previous_mask = y_i;
    }
    (packed + (1u128 << 64) * top_count - 2 * signed_weight) as u64
}

impl VirtualLaneExtractS {
    fn exec(&self, cpu: &mut Cpu, _: &mut <Self as RISCVInstruction>::RAMAccess) {
        let data = cpu.x[self.operands.rs1 as usize] as u64;
        let mask = cpu.x[self.operands.rs2 as usize] as u64;
        cpu.write_register(self.operands.rd as usize, signed_extract(data, mask) as i64);
    }
}

impl RISCVTrace for VirtualLaneExtractS {}
