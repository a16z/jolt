//! Field-inline (FR) guest arithmetic with prover-supplied result hints.
//!
//! On a RISC-V guest built with the `field-inline-guest` feature, the ring
//! operations of the FR-capable fields ([`crate::Fr`], [`Fp128`]) execute as
//! field-inline instructions instead of software limb arithmetic: both
//! operands are recomposed into the FR register file (Horner in radix
//! 2^64), the operation runs as one instruction, and the result is compared
//! by `FIELD_ASSERT_EQ` against a hint the prover recorded on the host. A
//! wrong or missing hint traps the trace, so a hinted guest can never accept
//! a computation the FR constraints would reject; the only thing a malicious
//! hint stream can do is make the guest trap or reject.
//!
//! Hints are the raw limb representation of each operation's result in the
//! same order the operations execute; the host records them by running the
//! identical code with [`start_recording`] / [`take_recording`], and the
//! guest installs the tape with [`install`] before the hinted computation.
//!
//! `Fr` keeps its Montgomery representation: a raw limb vector `aR` loaded
//! as a field element differs from `a` by the constant `R`, which
//! multiplication and inversion correct with one extra FR multiplication
//! (`abR² · R⁻¹`, `(aR)⁻¹ · R²`); addition and subtraction are
//! representation-transparent. `Fp128` is stored canonically and needs no
//! correction.

#[cfg(target_arch = "riscv64")]
use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// The custom-0 opcode the tracer dispatches field-inline words on.
pub const OPCODE: u32 = 0x7b;
pub const FUNCT3_ADD: u32 = 0;
pub const FUNCT3_SUB: u32 = 1;
pub const FUNCT3_MUL: u32 = 2;
pub const FUNCT3_INV: u32 = 3;
pub const FUNCT3_ASSERT_EQ: u32 = 4;
pub const FUNCT3_LOAD_FROM_X: u32 = 5;
pub const FUNCT3_LOAD_IMM: u32 = 7;
/// The x-register the bridge instructions read (`a0`).
pub const BRIDGE_X_REGISTER: u32 = 10;

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
const REG_RADIX: u32 = 0; // 2^64
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
#[cfg(target_arch = "riscv64")]
const REG_LIMB: u32 = 6;
#[cfg(target_arch = "riscv64")]
const REG_OUT: u32 = 7;
#[cfg(target_arch = "riscv64")]
const REG_HINT: u32 = 8;
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
// Hint tape (guest) and recorder (host).
// ---------------------------------------------------------------------------

// The tape is process-global, not thread-local: the guest is single-core and
// verifier code may run on more than one thread-local block, which must all
// see the same cursor.
#[cfg(target_arch = "riscv64")]
static TAPE_PTR: AtomicUsize = AtomicUsize::new(0);
#[cfg(target_arch = "riscv64")]
static TAPE_LEN: AtomicUsize = AtomicUsize::new(0);
#[cfg(target_arch = "riscv64")]
static TAPE_POS: AtomicUsize = AtomicUsize::new(0);

/// Install the hint tape the hinted operations consume, in execution order:
/// each hint is its result's limbs as little-endian `u64`s, so the tape is
/// consumed straight from the byte buffer it arrives in. The slice must
/// outlive every hinted operation.
#[cfg(target_arch = "riscv64")]
pub fn install(hints: &[u8]) {
    TAPE_PTR.store(hints.as_ptr() as usize, Ordering::Relaxed);
    TAPE_LEN.store(hints.len(), Ordering::Relaxed);
    TAPE_POS.store(0, Ordering::Relaxed);
}

/// Tape bytes consumed so far (guest) — lets a driver check the tape was
/// used exactly.
#[cfg(target_arch = "riscv64")]
pub fn consumed() -> usize {
    TAPE_POS.load(Ordering::Relaxed)
}

#[cfg(target_arch = "riscv64")]
#[cold]
fn tape_exhausted() -> ! {
    panic!("field-inline hint tape exhausted: the host recording ran fewer field operations than the guest")
}

/// Pops the next `N`-limb hint. Plain unaligned word loads: this sits on
/// every field operation, so it must not turn into `memcpy` calls.
#[cfg(target_arch = "riscv64")]
#[inline(always)]
fn next_hint<const N: usize>() -> [u64; N] {
    let pos = TAPE_POS.load(Ordering::Relaxed);
    if pos + 8 * N > TAPE_LEN.load(Ordering::Relaxed) {
        tape_exhausted();
    }
    let ptr = TAPE_PTR.load(Ordering::Relaxed) as *const u8;
    let mut out = [0u64; N];
    for (i, slot) in out.iter_mut().enumerate() {
        // SAFETY: bounds checked above against the installed slice length;
        // the tape is little-endian, as is the guest.
        *slot = unsafe { ptr.add(pos + 8 * i).cast::<u64>().read_unaligned() };
    }
    TAPE_POS.store(pos + 8 * N, Ordering::Relaxed);
    out
}

#[cfg(not(target_arch = "riscv64"))]
mod recorder {
    use std::sync::{Mutex, MutexGuard, PoisonError};
    static RECORD: Mutex<Option<Vec<u8>>> = Mutex::new(None);
    fn tape() -> MutexGuard<'static, Option<Vec<u8>>> {
        // A poisoned recorder holds a well-formed prefix; keep recording.
        RECORD.lock().unwrap_or_else(PoisonError::into_inner)
    }
    pub fn start() {
        *tape() = Some(Vec::new());
    }
    pub fn take() -> Option<Vec<u8>> {
        tape().take()
    }
    pub fn record(limbs: &[u64]) {
        if let Some(tape) = tape().as_mut() {
            for limb in limbs {
                tape.extend_from_slice(&limb.to_le_bytes());
            }
        }
    }
}

/// Start recording result hints on the host (no-op on the guest).
pub fn start_recording() {
    #[cfg(not(target_arch = "riscv64"))]
    recorder::start();
}

/// Stop recording and return the tape bytes (host); `None` on the guest.
pub fn take_recording() -> Option<Vec<u8>> {
    #[cfg(not(target_arch = "riscv64"))]
    return recorder::take();
    #[cfg(target_arch = "riscv64")]
    None
}

/// Record one result on the host recorder; no-op on the guest.
#[inline]
pub fn record(limbs: &[u64]) {
    #[cfg(not(target_arch = "riscv64"))]
    recorder::record(limbs);
    #[cfg(target_arch = "riscv64")]
    let _ = limbs;
}

// ---------------------------------------------------------------------------
// Guest instruction emitters.
// ---------------------------------------------------------------------------
#[cfg(target_arch = "riscv64")]
mod emit {
    use super::*;

    #[inline(always)]
    pub fn load_from_x(rd: u32, value: u64) {
        // The word is built from run-time register numbers, so it goes through
        // a register-indexed `.word` via a small match on the destination.
        macro_rules! word {
            ($rd:expr) => {
                // SAFETY: one fixed field-inline word; a0 carries the operand.
                unsafe {
                    core::arch::asm!(".word {w}", w = const r_word(FUNCT3_LOAD_FROM_X, $rd, BRIDGE_X_REGISTER, 0), in("x10") value, options(nostack));
                }
            };
        }
        match rd {
            0 => word!(0),
            1 => word!(1),
            2 => word!(2),
            3 => word!(3),
            4 => word!(4),
            5 => word!(5),
            6 => word!(6),
            7 => word!(7),
            8 => word!(8),
            _ => word!(9),
        }
    }

    macro_rules! fixed {
        ($w:expr) => {
            // SAFETY: one fixed field-inline word; no Rust memory is touched.
            unsafe {
                core::arch::asm!(".word {w}", w = const $w, options(nostack));
            }
        };
    }

    #[inline(always)]
    pub fn load_imm_radix_2() {
        fixed!(i_word(FUNCT3_LOAD_IMM, REG_RADIX, 2));
    }
    #[inline(always)]
    pub fn load_imm_zero() {
        fixed!(i_word(FUNCT3_LOAD_IMM, REG_ZERO, 0));
    }
    #[inline(always)]
    pub fn square_radix() {
        fixed!(r_word(FUNCT3_MUL, REG_RADIX, REG_RADIX, REG_RADIX));
    }
    /// `dst = dst * radix + limb` — the Horner step; `dst` is A, B, RINV, R2 or HINT.
    #[inline(always)]
    pub fn horner_step(dst: u32) {
        match dst {
            REG_A => {
                fixed!(r_word(FUNCT3_MUL, REG_A, REG_A, REG_RADIX));
                fixed!(r_word(FUNCT3_ADD, REG_A, REG_A, REG_LIMB));
            }
            REG_B => {
                fixed!(r_word(FUNCT3_MUL, REG_B, REG_B, REG_RADIX));
                fixed!(r_word(FUNCT3_ADD, REG_B, REG_B, REG_LIMB));
            }
            REG_RINV => {
                fixed!(r_word(FUNCT3_MUL, REG_RINV, REG_RINV, REG_RADIX));
                fixed!(r_word(FUNCT3_ADD, REG_RINV, REG_RINV, REG_LIMB));
            }
            REG_R2 => {
                fixed!(r_word(FUNCT3_MUL, REG_R2, REG_R2, REG_RADIX));
                fixed!(r_word(FUNCT3_ADD, REG_R2, REG_R2, REG_LIMB));
            }
            _ => {
                fixed!(r_word(FUNCT3_MUL, REG_HINT, REG_HINT, REG_RADIX));
                fixed!(r_word(FUNCT3_ADD, REG_HINT, REG_HINT, REG_LIMB));
            }
        }
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
    pub fn assert_out_eq_hint() {
        fixed!(r_word(FUNCT3_ASSERT_EQ, 0, REG_OUT, REG_HINT));
    }
    #[inline(always)]
    pub fn acc_zero() {
        fixed!(i_word(FUNCT3_LOAD_IMM, REG_ACC, 0));
    }
    #[inline(always)]
    pub fn acc_add_out() {
        fixed!(r_word(FUNCT3_ADD, REG_ACC, REG_ACC, REG_OUT));
    }
    #[inline(always)]
    pub fn assert_acc_eq_hint() {
        fixed!(r_word(FUNCT3_ASSERT_EQ, 0, REG_ACC, REG_HINT));
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
    #[inline(always)]
    pub fn assert_sum_eq_hint() {
        fixed!(r_word(FUNCT3_ASSERT_EQ, 0, REG_SUM, REG_HINT));
    }
}

#[cfg(target_arch = "riscv64")]
mod guest {
    use super::*;

    static READY: AtomicBool = AtomicBool::new(false);

    /// Load `limbs` (little-endian radix 2^64) into `dst` by Horner.
    #[inline(always)]
    fn load<const N: usize>(dst: u32, limbs: &[u64; N]) {
        emit::load_from_x(dst, limbs[N - 1]);
        for i in (0..N - 1).rev() {
            emit::load_from_x(REG_LIMB, limbs[i]);
            emit::horner_step(dst);
        }
    }

    #[inline(always)]
    fn ensure_constants() {
        if READY.load(Ordering::Relaxed) {
            return;
        }
        emit::load_imm_radix_2();
        for _ in 0..6 {
            emit::square_radix(); // 2 -> 2^64
        }
        emit::load_imm_zero();
        load(REG_RINV, &BN254_RINV);
        load(REG_R2, &BN254_R2);
        READY.store(true, Ordering::Relaxed);
    }

    #[inline(always)]
    fn finish<const N: usize>() -> [u64; N] {
        let hint = next_hint::<N>();
        load(REG_HINT, &hint);
        emit::assert_out_eq_hint();
        hint
    }

    #[inline(always)]
    pub fn add<const N: usize>(a: &[u64; N], b: &[u64; N]) -> [u64; N] {
        ensure_constants();
        load(REG_A, a);
        load(REG_B, b);
        emit::add_out();
        finish()
    }
    #[inline(always)]
    pub fn sub<const N: usize>(a: &[u64; N], b: &[u64; N]) -> [u64; N] {
        ensure_constants();
        load(REG_A, a);
        load(REG_B, b);
        emit::sub_out();
        finish()
    }
    #[inline(always)]
    pub fn neg<const N: usize>(a: &[u64; N]) -> [u64; N] {
        ensure_constants();
        load(REG_A, a);
        emit::neg_out();
        finish()
    }
    /// `montgomery`: correct the product of two Montgomery representatives by R⁻¹.
    #[inline(always)]
    pub fn mul<const N: usize>(a: &[u64; N], b: &[u64; N], montgomery: bool) -> [u64; N] {
        ensure_constants();
        load(REG_A, a);
        load(REG_B, b);
        emit::mul_out();
        if montgomery {
            emit::mul_out_rinv();
        }
        finish()
    }
    /// Caller guarantees `a != 0` (FIELD_INV traps on zero).
    /// `Σ a[i]·b[i]` with the running sum resident in the field register file:
    /// operands are loaded once each and only the final sum is hinted, so a
    /// length-`k` dot product costs `k` multiplies with one hint round trip
    /// instead of `k` hinted multiplies and `k − 1` hinted additions.
    /// Canonical (non-Montgomery) limbs only.
    #[inline(always)]
    pub fn dot<const N: usize>(a: &[[u64; N]], b: &[[u64; N]]) -> [u64; N] {
        ensure_constants();
        emit::acc_zero();
        for (x, y) in a.iter().zip(b) {
            load(REG_A, x);
            load(REG_B, y);
            emit::mul_out();
            emit::acc_add_out();
        }
        let hint = next_hint::<N>();
        load(REG_HINT, &hint);
        emit::assert_acc_eq_hint();
        hint
    }

    /// `Σ_i weights[i] · Σ_j rows[i][j]·pows[j]` with the row sums and the
    /// weighted total register-resident and one hint for the result. Each
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
        ensure_constants();
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
        let hint = next_hint::<N>();
        load(REG_HINT, &hint);
        emit::assert_sum_eq_hint();
        hint
    }

    #[inline(always)]
    pub fn inv<const N: usize>(a: &[u64; N], montgomery: bool) -> [u64; N] {
        ensure_constants();
        load(REG_A, a);
        emit::inv_out();
        if montgomery {
            emit::mul_out_r2();
        }
        finish()
    }
}

#[cfg(target_arch = "riscv64")]
pub use guest::{add, dot, inv, mul, neg, sub, weighted_dot_rows};
