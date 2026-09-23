use serde::{Deserialize, Serialize};

use crate::{declare_riscv_instr, emulator::cpu::Cpu};

use super::{format::format_r::FormatR, RISCVInstruction, RISCVTrace};

declare_riscv_instr!(
    name = VirtualPextSigned,
    mask = 0,
    match = 0,
    format = FormatR,
    ram = ()
);

impl VirtualPextSigned {
    fn exec(&self, cpu: &mut Cpu, _: &mut <VirtualPextSigned as RISCVInstruction>::RAMAccess) {
        let x = cpu.x[self.operands.rs1 as usize] as u64;
        let y = cpu.x[self.operands.rs2 as usize] as u64;
        cpu.write_register(self.operands.rd as usize, pext_signed(x, y) as i64);
    }
}

/// `pext(x, y)`: packs `x`'s bits at `y`'s set positions toward bit 0,
/// preserving order (the window's top bit lands at `popcount(y) − 1`).
#[inline]
pub(crate) fn pext(x: u64, y: u64) -> u64 {
    if y == 0 {
        return 0;
    }
    let tz = y.trailing_zeros();
    let normalized = y >> tz;
    if normalized & normalized.wrapping_add(1) == 0 {
        // Contiguous mask (every mask the window-mask instructions produce):
        // extract is a shift plus truncate.
        return (x >> tz) & normalized;
    }
    // General mask: gather one bit per set position, lowest first.
    let mut bits = y;
    let mut out = 0u64;
    let mut k = 0;
    while bits != 0 {
        out |= ((x >> bits.trailing_zeros()) & 1) << k;
        k += 1;
        bits &= bits - 1;
    }
    out
}

/// Packs `x`'s bits at `y`'s set positions (MSB-first), then sign-extends by
/// the extracted window's top bit: `pext(x, y) + σ·(2^64 − 2^popcount(y))`
/// where `σ` is `x`'s bit at `y`'s most significant set bit. Same formulation
/// as the lookup tables' `pext_signed` at `XLEN = 64`.
pub(crate) fn pext_signed(x: u64, y: u64) -> u64 {
    let pc = y.count_ones();
    if pc == 0 {
        return 0;
    }
    let pext = pext(x, y);
    // σ: the window sign, x's bit at y's most significant set bit.
    let sign = (x >> y.ilog2()) & 1;
    // pext < 2^pc, so the sum never overflows 64 bits.
    let ext = if sign == 1 {
        ((1u128 << 64) - (1u128 << pc)) as u64
    } else {
        0
    };
    pext + ext
}

impl RISCVTrace for VirtualPextSigned {}
