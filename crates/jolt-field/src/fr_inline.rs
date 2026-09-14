//! Field-inline (FR) guest arithmetic.
//!
//! On a RISC-V guest built with the `field-inline-guest` feature, the ring
//! operations of the FR-capable fields ([`crate::Fr`], [`Fp128`]) execute as
//! field-inline instructions instead of software limb arithmetic: operands
//! enter the FR register file straight from memory (`FIELD_LOAD_WORD` for the
//! top limb, `FIELD_LOAD_WORD_HI` folding each lower limb in radix 2^64), the
//! operation runs as one instruction, and the result leaves through
//! `FIELD_ADVICE_LIMB` (one range-bound low limb per row, the quotient staying
//! in the register file) closed by `FIELD_STORE_TO_X`. These relations bind
//! the result modulo the field characteristic. The field wrappers check
//! that the returned integer is below the modulus before using it.
//!
//! `Fr` keeps its Montgomery representation: a raw limb vector `aR` loaded
//! as a field element differs from `a` by the constant `R`, which
//! multiplication and inversion correct with one extra FR multiplication
//! (`abR² · R⁻¹`, `(aR)⁻¹ · R²`); addition and subtraction are
//! representation-transparent. `Fp128` is stored canonically and needs no
//! correction.

#[cfg(target_arch = "riscv64")]
use core::sync::atomic::{AtomicBool, Ordering};

/// The custom-0 opcode the tracer dispatches field-inline words on.
pub const OPCODE: u32 = 0x7b;
pub const FUNCT3_ADD: u32 = 0;
pub const FUNCT3_SUB: u32 = 1;
pub const FUNCT3_MUL: u32 = 2;
pub const FUNCT3_INV: u32 = 3;
pub const FUNCT3_ASSERT_EQ: u32 = 4;
pub const FUNCT3_LOAD_FROM_X: u32 = 5;
pub const FUNCT3_STORE_TO_X: u32 = 6;
pub const FUNCT3_LOAD_IMM: u32 = 7;
/// funct7 of `FIELD_ADVICE_LIMB` (shares `FIELD_STORE_TO_X`'s funct3).
pub const FUNCT7_ADVICE_LIMB: u32 = 1;
/// funct7 of the memory-sourced loads: the family bit, the high-word bit, and
/// the word offset (see `jolt_riscv::field_inline_load_word_funct7`).
pub const FUNCT7_LOAD_WORD_FAMILY: u32 = 0x40;
pub const FUNCT7_LOAD_WORD_HIGH: u32 = 0x20;

#[cfg(target_arch = "riscv64")]
const fn r_word(funct3: u32, rd: u32, rs1: u32, rs2: u32) -> u32 {
    OPCODE | (rd << 7) | (funct3 << 12) | (rs1 << 15) | (rs2 << 20)
}

#[cfg(target_arch = "riscv64")]
const fn i_word(funct3: u32, rd: u32, imm: u32) -> u32 {
    OPCODE | (rd << 7) | (funct3 << 12) | (imm << 20)
}

// ---------------------------------------------------------------------------
// FR register map for the hinted ops. Constants live in low registers for
// the whole run; each operation uses the scratch registers above them.
// ---------------------------------------------------------------------------
#[cfg(target_arch = "riscv64")]
const REG_RINV: u32 = 1; // BN254 R^-1 (Montgomery product correction)
#[cfg(target_arch = "riscv64")]
const REG_R2: u32 = 2; // BN254 R^2 (Montgomery inverse correction)
#[cfg(target_arch = "riscv64")]
const REG_ZERO: u32 = 3;
#[cfg(target_arch = "riscv64")]
const REG_A: u32 = 4;
#[cfg(target_arch = "riscv64")]
const REG_B: u32 = 5;
/// The two registers the limb readout alternates its quotients through.
#[cfg(target_arch = "riscv64")]
const REG_SCRATCH_A: u32 = 6;
#[cfg(target_arch = "riscv64")]
const REG_OUT: u32 = 7;
#[cfg(target_arch = "riscv64")]
const REG_SCRATCH_B: u32 = 8;
/// Running sum of a register-resident dot product.
#[cfg(target_arch = "riscv64")]
const REG_ACC: u32 = 9;
/// The weighted-rows kernel keeps one accumulator per row of a block in
/// registers 9..=13 and the weighted total in 14.
#[cfg(target_arch = "riscv64")]
const REG_SUM: u32 = 14;
/// Rows per block of the weighted-rows kernel (registers 9..=13).
pub const WEIGHTED_ROWS_BLOCK: usize = 5;

/// BN254 scalar-field Montgomery constants as canonical little-endian limbs.
#[cfg(target_arch = "riscv64")]
const BN254_RINV: [u64; 4] = [
    0xdc5b_a005_6db1_194e,
    0x090e_f5a9_e111_ec87,
    0xc826_0de4_aeb8_5d5d,
    0x15eb_f951_82c5_551c,
];
#[cfg(target_arch = "riscv64")]
const BN254_R2: [u64; 4] = [
    0x1bb8_e645_ae21_6da7,
    0x53fe_3ab1_e35c_59e3,
    0x8c49_833d_53bb_8085,
    0x0216_d0b1_7f4e_44a5,
];

// ---------------------------------------------------------------------------
// Guest instruction emitters.
// ---------------------------------------------------------------------------
#[cfg(target_arch = "riscv64")]
mod emit {
    use super::*;

    macro_rules! fixed {
        ($w:expr) => {
            // SAFETY: one fixed field-inline word; no Rust memory is touched.
            unsafe {
                core::arch::asm!(".word {w}", w = const $w, options(nostack));
            }
        };
    }

    /// One memory-sourced load: `fr[dst] = [fr[dst]·2^64 +] mem[base + 8·offset]`.
    /// The word is built from run-time register numbers and offsets, so it
    /// goes through a small match on the destination and limb position.
    #[inline(always)]
    pub fn load_word(dst: u32, high: bool, offset: usize, base: *const u64) {
        macro_rules! word {
            ($rd:expr, $high:expr, $offset:expr) => {
                // SAFETY: one field-inline load from the live limb pointer.
                // The integer destination is scratch; the field destination
                // remains fixed while LLVM allocates the address register.
                unsafe {
                    core::arch::asm!(
                        ".insn r {opcode}, {funct3}, {funct7}, {scratch}, {base}, x{field}",
                        opcode = const OPCODE,
                        funct3 = const FUNCT3_LOAD_FROM_X,
                        funct7 = const FUNCT7_LOAD_WORD_FAMILY
                            | if $high { FUNCT7_LOAD_WORD_HIGH } else { 0 } | $offset,
                        field = const $rd,
                        base = in(reg) base,
                        scratch = lateout(reg) _,
                        options(nostack, readonly),
                    );
                }
            };
        }
        macro_rules! limbs {
            ($rd:expr) => {
                match (high, offset) {
                    (false, 0) => word!($rd, false, 0),
                    (false, 1) => word!($rd, false, 1),
                    (false, 2) => word!($rd, false, 2),
                    (false, _) => word!($rd, false, 3),
                    (true, 0) => word!($rd, true, 0),
                    (true, 1) => word!($rd, true, 1),
                    (true, 2) => word!($rd, true, 2),
                    (true, _) => word!($rd, true, 3),
                }
            };
        }
        match dst {
            REG_RINV => limbs!(REG_RINV),
            REG_R2 => limbs!(REG_R2),
            REG_A => limbs!(REG_A),
            _ => limbs!(REG_B),
        }
    }

    /// Supply a 64-bit limb with `fr[quotient] = (fr[src] − limb) / 2^64`.
    /// The honest tracer chooses the canonical low limb. This row alone
    /// does not enforce that choice.
    #[inline(always)]
    pub fn advice_limb(src: u32, quotient: u32) -> u64 {
        macro_rules! word {
            ($src:expr, $quotient:expr) => {{
                let low: u64;
                // SAFETY: one field-inline word, with a compiler-allocated integer output.
                unsafe {
                    core::arch::asm!(
                        ".insn r {opcode}, {funct3}, {funct7}, {low}, x{src}, x{quotient}",
                        opcode = const OPCODE, funct3 = const FUNCT3_STORE_TO_X,
                        funct7 = const FUNCT7_ADVICE_LIMB,
                        src = const $src, quotient = const $quotient,
                        low = lateout(reg) low, options(nostack, nomem),
                    );
                }
                low
            }};
        }
        macro_rules! sources {
            ($quotient:expr) => {
                match src {
                    REG_OUT => word!(REG_OUT, $quotient),
                    REG_ACC => word!(REG_ACC, $quotient),
                    REG_SUM => word!(REG_SUM, $quotient),
                    REG_SCRATCH_A => word!(REG_SCRATCH_A, $quotient),
                    _ => word!(REG_SCRATCH_B, $quotient),
                }
            };
        }
        match quotient {
            REG_SCRATCH_A => sources!(REG_SCRATCH_A),
            _ => sources!(REG_SCRATCH_B),
        }
    }

    /// Read `fr[src]` below 2^64 (the last quotient of a readout).
    #[inline(always)]
    pub fn store_to_x(src: u32) -> u64 {
        macro_rules! word {
            ($src:expr) => {{
                let value: u64;
                // SAFETY: one field-inline word, with a compiler-allocated integer output.
                unsafe {
                    core::arch::asm!(
                        ".insn r {opcode}, {funct3}, 0, {value}, x{src}, x0",
                        opcode = const OPCODE, funct3 = const FUNCT3_STORE_TO_X,
                        src = const $src, value = lateout(reg) value,
                        options(nostack, nomem),
                    );
                }
                value
            }};
        }
        match src {
            REG_OUT => word!(REG_OUT),
            REG_ACC => word!(REG_ACC),
            REG_SUM => word!(REG_SUM),
            REG_SCRATCH_A => word!(REG_SCRATCH_A),
            _ => word!(REG_SCRATCH_B),
        }
    }

    #[inline(always)]
    pub fn load_imm_zero() {
        fixed!(i_word(FUNCT3_LOAD_IMM, REG_ZERO, 0));
    }
    #[inline(always)]
    pub fn add_out() {
        fixed!(r_word(FUNCT3_ADD, REG_OUT, REG_A, REG_B));
    }
    #[inline(always)]
    pub fn sub_out() {
        fixed!(r_word(FUNCT3_SUB, REG_OUT, REG_A, REG_B));
    }
    #[inline(always)]
    pub fn neg_out() {
        fixed!(r_word(FUNCT3_SUB, REG_OUT, REG_ZERO, REG_A));
    }
    #[inline(always)]
    pub fn mul_out() {
        fixed!(r_word(FUNCT3_MUL, REG_OUT, REG_A, REG_B));
    }
    #[inline(always)]
    pub fn mul_out_rinv() {
        fixed!(r_word(FUNCT3_MUL, REG_OUT, REG_OUT, REG_RINV));
    }
    #[inline(always)]
    pub fn inv_out() {
        fixed!(r_word(FUNCT3_INV, REG_OUT, REG_A, 0));
    }
    #[inline(always)]
    pub fn mul_out_r2() {
        fixed!(r_word(FUNCT3_MUL, REG_OUT, REG_OUT, REG_R2));
    }
    #[inline(always)]
    pub fn acc_zero() {
        fixed!(i_word(FUNCT3_LOAD_IMM, REG_ACC, 0));
    }
    #[inline(always)]
    pub fn acc_add_out() {
        fixed!(r_word(FUNCT3_ADD, REG_ACC, REG_ACC, REG_OUT));
    }
    /// Zero row accumulator `k` (registers 9..=13).
    #[inline(always)]
    pub fn row_acc_zero(k: usize) {
        match k {
            0 => fixed!(i_word(FUNCT3_LOAD_IMM, 9, 0)),
            1 => fixed!(i_word(FUNCT3_LOAD_IMM, 10, 0)),
            2 => fixed!(i_word(FUNCT3_LOAD_IMM, 11, 0)),
            3 => fixed!(i_word(FUNCT3_LOAD_IMM, 12, 0)),
            _ => fixed!(i_word(FUNCT3_LOAD_IMM, 13, 0)),
        }
    }
    /// `acc_k += OUT`.
    #[inline(always)]
    pub fn row_acc_add_out(k: usize) {
        match k {
            0 => fixed!(r_word(FUNCT3_ADD, 9, 9, REG_OUT)),
            1 => fixed!(r_word(FUNCT3_ADD, 10, 10, REG_OUT)),
            2 => fixed!(r_word(FUNCT3_ADD, 11, 11, REG_OUT)),
            3 => fixed!(r_word(FUNCT3_ADD, 12, 12, REG_OUT)),
            _ => fixed!(r_word(FUNCT3_ADD, 13, 13, REG_OUT)),
        }
    }
    /// `OUT = acc_k * B`.
    #[inline(always)]
    pub fn mul_out_row_acc_b(k: usize) {
        match k {
            0 => fixed!(r_word(FUNCT3_MUL, REG_OUT, 9, REG_B)),
            1 => fixed!(r_word(FUNCT3_MUL, REG_OUT, 10, REG_B)),
            2 => fixed!(r_word(FUNCT3_MUL, REG_OUT, 11, REG_B)),
            3 => fixed!(r_word(FUNCT3_MUL, REG_OUT, 12, REG_B)),
            _ => fixed!(r_word(FUNCT3_MUL, REG_OUT, 13, REG_B)),
        }
    }
    #[inline(always)]
    pub fn sum_zero() {
        fixed!(i_word(FUNCT3_LOAD_IMM, REG_SUM, 0));
    }
    #[inline(always)]
    pub fn sum_add_out() {
        fixed!(r_word(FUNCT3_ADD, REG_SUM, REG_SUM, REG_OUT));
    }
}

#[cfg(target_arch = "riscv64")]
mod guest {
    use super::*;

    static READY: AtomicBool = AtomicBool::new(false);

    /// Load `limbs` (little-endian radix 2^64) into `dst` straight from
    /// memory: the top limb by `FIELD_LOAD_WORD`, each lower limb folded in by
    /// `FIELD_LOAD_WORD_HI` (a load and the Horner step in one row). One row
    /// per limb instead of the bridge's three.
    #[inline(always)]
    fn load<const N: usize>(dst: u32, limbs: &[u64; N]) {
        let base = limbs.as_ptr();
        emit::load_word(dst, false, N - 1, base);
        for i in (0..N - 1).rev() {
            emit::load_word(dst, true, i, base);
        }
    }

    #[inline(always)]
    fn ensure_montgomery_constants() {
        if READY.load(Ordering::Relaxed) {
            return;
        }
        load(REG_RINV, &BN254_RINV);
        load(REG_R2, &BN254_R2);
        READY.store(true, Ordering::Relaxed);
    }

    /// Read the `N`-limb result out of `src`: `N − 1` splits peel the low
    /// limbs (quotients alternating through the two scratch registers), and
    /// the last quotient, below 2^64, leaves through the store bridge.
    /// The caller must reject integers at or above the active field modulus.
    #[inline(always)]
    fn read_out<const N: usize>(src: u32) -> [u64; N] {
        let mut limbs = [0u64; N];
        let mut current = src;
        for limb in limbs.iter_mut().take(N - 1) {
            let quotient = if current == REG_SCRATCH_A {
                REG_SCRATCH_B
            } else {
                REG_SCRATCH_A
            };
            *limb = emit::advice_limb(current, quotient);
            current = quotient;
        }
        limbs[N - 1] = emit::store_to_x(current);
        limbs
    }

    #[inline(always)]
    fn finish<const N: usize>() -> [u64; N] {
        read_out(REG_OUT)
    }

    #[inline(always)]
    pub fn add<const N: usize>(a: &[u64; N], b: &[u64; N]) -> [u64; N] {
        load(REG_A, a);
        load(REG_B, b);
        emit::add_out();
        finish()
    }
    #[inline(always)]
    pub fn sub<const N: usize>(a: &[u64; N], b: &[u64; N]) -> [u64; N] {
        load(REG_A, a);
        load(REG_B, b);
        emit::sub_out();
        finish()
    }
    #[inline(always)]
    pub fn neg<const N: usize>(a: &[u64; N]) -> [u64; N] {
        emit::load_imm_zero();
        load(REG_A, a);
        emit::neg_out();
        finish()
    }
    /// `montgomery`: correct the product of two Montgomery representatives by R⁻¹.
    #[inline(always)]
    pub fn mul<const N: usize>(a: &[u64; N], b: &[u64; N], montgomery: bool) -> [u64; N] {
        load(REG_A, a);
        load(REG_B, b);
        emit::mul_out();
        if montgomery {
            ensure_montgomery_constants();
            emit::mul_out_rinv();
        }
        finish()
    }
    /// Caller guarantees `a != 0` (FIELD_INV traps on zero).
    /// `Σ a[i]·b[i]` with the running sum resident in the field register file:
    /// operands are loaded once each and only the final sum is read out, so a
    /// length-`k` dot product costs `k` multiplies and one readout instead of
    /// `k` multiplies and `k − 1` additions each read out.
    /// Canonical (non-Montgomery) limbs only.
    #[inline(always)]
    pub fn dot<const N: usize>(a: &[[u64; N]], b: &[[u64; N]]) -> [u64; N] {
        emit::acc_zero();
        for (x, y) in a.iter().zip(b) {
            load(REG_A, x);
            load(REG_B, y);
            emit::mul_out();
            emit::acc_add_out();
        }
        read_out(REG_ACC)
    }

    /// `Σ_i weights[i] · Σ_j rows[i][j]·pows[j]` with the row sums and the
    /// weighted total register-resident and one readout for the result. Each
    /// power is loaded once per block of [`WEIGHTED_ROWS_BLOCK`] rows, which
    /// is what makes this cheaper than one [`dot`] per row: operand ingress,
    /// not arithmetic, is the cost of a field-inline multiply-accumulate.
    /// Canonical limbs; every row has `pows.len()` elements.
    #[inline(always)]
    pub fn weighted_dot_rows<const N: usize>(
        rows: &[&[[u64; N]]],
        weights: &[[u64; N]],
        pows: &[[u64; N]],
    ) -> [u64; N] {
        let len = pows.len();
        assert_eq!(rows.len(), weights.len(), "one weight per row");
        assert!(
            rows.iter().all(|row| row.len() == len),
            "every row has one element per power"
        );
        // One block's row pointers stay in scalar registers and the element
        // index is the only loop state: the scalar bookkeeping around each
        // field-inline word costs as much as the word itself, so the inner
        // loop is unrolled per block width and indexes without bounds checks.
        macro_rules! accumulate_block {
            ($($row:ident => $k:literal),+) => {
                for j in 0..len {
                    // SAFETY: `j < len`, and every row has `len` elements
                    // (asserted above).
                    unsafe {
                        load(REG_A, pows.get_unchecked(j));
                        $(
                            load(REG_B, $row.get_unchecked(j));
                            emit::mul_out();
                            emit::row_acc_add_out($k);
                        )+
                    }
                }
            };
        }
        const _: () = assert!(
            WEIGHTED_ROWS_BLOCK == 5,
            "the block arms below are five wide"
        );
        emit::sum_zero();
        for (block_rows, block_weights) in rows
            .chunks(WEIGHTED_ROWS_BLOCK)
            .zip(weights.chunks(WEIGHTED_ROWS_BLOCK))
        {
            for k in 0..block_rows.len() {
                emit::row_acc_zero(k);
            }
            match block_rows {
                [r0, r1, r2, r3, r4] => {
                    accumulate_block!(r0 => 0, r1 => 1, r2 => 2, r3 => 3, r4 => 4)
                }
                [r0, r1, r2, r3] => accumulate_block!(r0 => 0, r1 => 1, r2 => 2, r3 => 3),
                [r0, r1, r2] => accumulate_block!(r0 => 0, r1 => 1, r2 => 2),
                [r0, r1] => accumulate_block!(r0 => 0, r1 => 1),
                [r0] => accumulate_block!(r0 => 0),
                // `chunks(WEIGHTED_ROWS_BLOCK)` yields one to five rows.
                _ => {}
            }
            for (k, weight) in block_weights.iter().enumerate() {
                load(REG_B, weight);
                emit::mul_out_row_acc_b(k);
                emit::sum_add_out();
            }
        }
        read_out(REG_SUM)
    }

    #[inline(always)]
    pub fn inv<const N: usize>(a: &[u64; N], montgomery: bool) -> [u64; N] {
        load(REG_A, a);
        emit::inv_out();
        if montgomery {
            ensure_montgomery_constants();
            emit::mul_out_r2();
        }
        finish()
    }
}

#[cfg(target_arch = "riscv64")]
pub use guest::{add, dot, inv, mul, neg, sub, weighted_dot_rows};
