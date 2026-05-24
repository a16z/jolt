// SPDX-License-Identifier: Apache-2.0

//! Sequence builder for the Goldilocks Poseidon2 inline.
//!
//! Emits a flat sequence of virtual RISC-V instructions that permutes
//! an 8-element Goldilocks state in place. Operates over ~35 virtual
//! registers so the entire permutation runs without spilling state to
//! memory between rounds.
//!
//! Memory layout:
//! - `rs1`: pointer to the 8-element state (64 bytes), permuted in place
//! - `rs2`: pointer to the 86-element round-constant table (688 bytes)
//!
use core::array;

use jolt_inlines_sdk::host::{
    instruction::{
        add::ADD, addi::ADDI, and::AND, ld::LD, mul::MUL, mulhu::MULHU, sd::SD, slli::SLLI,
        sltu::SLTU, srli::SRLI, sub::SUB,
    },
    FormatInline, InlineOp, InstrAssembler, Instruction, VirtualRegisterGuard,
};

use crate::exec::POSEIDON2_INTERNAL_DIAG;
use crate::STATE_WIDTH;

/// Virtual-register count.
///
/// Layout:
/// - `vr[0..8]`   — state `S[0..7]` (live across all rounds)
/// - `vr[8..16]`  — temp state `T[0..7]` for MDS reorganization
/// - `vr[16]`     — P (Goldilocks modulus) loaded once
/// - `vr[17..24]` — mul_mod scratch (7 regs: lo, hi, hi_lo, hi_hi,
///                   shifted, add_term, sub_term)
/// - `vr[24..28]` — add_mod / final-reduction scratch (4 regs)
/// - `vr[28..32]` — generic scratch (round constants, diff sums, etc.)
/// - `vr[32]`     — internal-diffusion row-sum accumulator
/// - `vr[33..35]` — extra scratch for shifts and intermediates
/// - `vr[35]`     — mask_low_32 (constant 2^32 - 1, loaded once)
/// - `vr[36]`     — mm_add_ovf (mul_mod add_term overflow flag)
pub const NEEDED_REGISTERS: u8 = 37;

const P_REG: usize = 16;

const STATE_LEN: usize = STATE_WIDTH;

/// Inline operation tag, registered with the Jolt prover via
/// [`crate::host`].
pub struct Poseidon2GoldilocksPermutation;

impl InlineOp for Poseidon2GoldilocksPermutation {
    const OPCODE: u32 = crate::INLINE_OPCODE;
    const FUNCT3: u32 = crate::POSEIDON2_GOLDILOCKS_FUNCT3;
    const FUNCT7: u32 = crate::POSEIDON2_GOLDILOCKS_FUNCT7;
    const NAME: &'static str = crate::POSEIDON2_GOLDILOCKS_NAME;

    fn build_sequence(asm: InstrAssembler, operands: FormatInline) -> Vec<Instruction> {
        Poseidon2GoldilocksSequenceBuilder::new(asm, operands).build()
    }
}

pub(crate) struct Poseidon2GoldilocksSequenceBuilder {
    asm: InstrAssembler,
    vr: [VirtualRegisterGuard; NEEDED_REGISTERS as usize],
    operands: FormatInline,
}

impl Poseidon2GoldilocksSequenceBuilder {
    fn new(asm: InstrAssembler, operands: FormatInline) -> Self {
        let vr = array::from_fn(|_| asm.allocator.allocate_for_inline());
        Poseidon2GoldilocksSequenceBuilder { asm, vr, operands }
    }

    // ── Register accessors ────────────────────────────────────────────

    fn s(&self, i: usize) -> u8 {
        *self.vr[i]
    }
    fn t(&self, i: usize) -> u8 {
        *self.vr[STATE_LEN + i]
    }
    fn p_reg(&self) -> u8 {
        *self.vr[P_REG]
    }
    // Named scratch registers for mul_mod
    fn mm_lo(&self) -> u8 {
        *self.vr[17]
    }
    fn mm_hi(&self) -> u8 {
        *self.vr[18]
    }
    fn mm_hi_lo(&self) -> u8 {
        *self.vr[19]
    }
    fn mm_hi_hi(&self) -> u8 {
        *self.vr[20]
    }
    fn mm_shifted(&self) -> u8 {
        *self.vr[21]
    }
    fn mm_add_term(&self) -> u8 {
        *self.vr[22]
    }
    fn mm_sub_term(&self) -> u8 {
        *self.vr[23]
    }
    // add_mod / final-reduction scratch
    fn am_ovf(&self) -> u8 {
        *self.vr[24]
    }
    fn am_corr(&self) -> u8 {
        *self.vr[25]
    }
    fn am_less(&self) -> u8 {
        *self.vr[26]
    }
    fn am_tmp(&self) -> u8 {
        *self.vr[27]
    }
    // Generic scratch
    fn sc_rc(&self) -> u8 {
        *self.vr[28]
    }
    fn sc_diag(&self) -> u8 {
        *self.vr[29]
    }
    fn sc_a(&self) -> u8 {
        *self.vr[30]
    }
    fn sc_b(&self) -> u8 {
        *self.vr[31]
    }
    fn sum_reg(&self) -> u8 {
        *self.vr[32]
    }
    fn sc_c(&self) -> u8 {
        *self.vr[33]
    }
    fn sc_d(&self) -> u8 {
        *self.vr[34]
    }
    fn mask_low32(&self) -> u8 {
        *self.vr[35]
    }
    fn mm_add_ovf(&self) -> u8 {
        *self.vr[36]
    }

    // ── Top-level build ───────────────────────────────────────────────

    fn build(mut self) -> Vec<Instruction> {
        // 1. Load Goldilocks modulus P into p_reg (3 instructions).
        self.load_p();

        // 2. Load state from memory into vr[0..8].
        self.load_state();

        // 3. Initial external MDS.
        self.external_mds();

        // 4. 4 external initial rounds.
        let mut rc_idx: usize = 0;
        for _ in 0..4 {
            self.add_round_constants_full(rc_idx);
            rc_idx += STATE_LEN;
            self.sbox_full();
            self.external_mds();
        }

        // 5. 22 internal rounds.
        for _ in 0..22 {
            self.add_round_constant_partial(rc_idx);
            rc_idx += 1;
            self.sbox_state_0();
            self.internal_diffusion();
        }

        // 6. 4 external final rounds.
        for _ in 0..4 {
            self.add_round_constants_full(rc_idx);
            rc_idx += STATE_LEN;
            self.sbox_full();
            self.external_mds();
        }
        debug_assert_eq!(rc_idx, 86, "round-constant index off");

        // 7. Store state back to memory.
        self.store_state();

        drop(self.vr);
        self.asm.finalize_inline()
    }

    // ── Constant loading ──────────────────────────────────────────────

    /// Load P = 2^64 - 2^32 + 1 into p_reg, and (2^32 - 1) into
    /// mask_low32. Both are constants used throughout the permutation.
    ///
    /// The Jolt inline assembler's `emit_i::<ADDI>` accepts a full
    /// u64 immediate (it stores it raw and the emulator does
    /// `x[rs1].wrapping_add(imm as i64)`). So a single ADDI with rs1=x0
    /// loads any u64 value.
    fn load_p(&mut self) {
        let p = self.p_reg();
        let mask = self.mask_low32();
        self.asm.emit_i::<ADDI>(p, 0, crate::GOLDILOCKS_MODULUS);
        self.asm.emit_i::<ADDI>(mask, 0, 0xFFFF_FFFF);
    }

    // ── State load / store ────────────────────────────────────────────

    fn load_state(&mut self) {
        for i in 0..STATE_LEN {
            self.asm
                .emit_ld::<LD>(self.s(i), self.operands.rs1, (i * 8) as i64);
        }
    }

    fn store_state(&mut self) {
        for i in 0..STATE_LEN {
            self.asm
                .emit_s::<SD>(self.operands.rs1, self.s(i), (i * 8) as i64);
        }
    }

    // ── Field arithmetic primitives ───────────────────────────────────

    /// dst = (a + b) mod P.
    ///
    /// 11 instructions. Note: `dst` may alias `a` or `b` — we snapshot
    /// `a` first to avoid clobbering it before the overflow check.
    fn add_mod_into(&mut self, dst: u8, a: u8, b: u8) {
        let p = self.p_reg();
        let ovf = self.am_ovf();
        let corr = self.am_corr();
        let less = self.am_less();
        let tmp = self.am_tmp();

        // 0. Snapshot `a` into tmp to survive the dst-write aliasing.
        //    Common call shape is `add_mod_into(s, s, rc)` where dst == a.
        self.asm.emit_r::<ADD>(tmp, a, 0);
        // 1. dst = a + b (wrapping)
        self.asm.emit_r::<ADD>(dst, a, b);
        // 2. ovf = (dst < tmp) ? 1 : 0  -- overflow detection using snapshot
        self.asm.emit_r::<SLTU>(ovf, dst, tmp);
        // 3-4. corr = ovf * (2^32 - 1) = (ovf << 32) - ovf
        self.asm.emit_i::<SLLI>(corr, ovf, 32);
        self.asm.emit_r::<SUB>(corr, corr, ovf);
        // 5. dst = dst + corr  -- if no overflow, corr = 0
        self.asm.emit_r::<ADD>(dst, dst, corr);
        // 6-10. Final reduction: if dst >= P, dst -= P.
        self.asm.emit_r::<SLTU>(less, dst, p);
        self.asm.emit_r::<SUB>(tmp, dst, p);
        self.asm.emit_r::<SUB>(corr, 0, less); // corr = -less = 0 or all-ones
        self.asm.emit_r::<AND>(corr, corr, p); // corr = P if less else 0
        self.asm.emit_r::<ADD>(dst, tmp, corr);
    }

    /// dst = (a * b) mod P using the Goldilocks reduction trick.
    ///
    /// Mirrors the corrected `exec::mul_mod`. ~25 instructions.
    ///
    /// Critical detail: when `lo + (hi_lo << 32)` overflows 2^64
    /// during the `add_term` step, naive wrapping loses 2^64 worth of
    /// magnitude. Since 2^64 ≡ (2^32 - 1) mod P, the result is short
    /// by (2^32 - 1) mod P. We detect this overflow and add the
    /// correction, handling the double-wrap case (where the
    /// correction itself overflows).
    fn mul_mod_into(&mut self, dst: u8, a: u8, b: u8) {
        let p = self.p_reg();
        let mask = self.mask_low32();
        let lo = self.mm_lo();
        let hi = self.mm_hi();
        let hi_lo = self.mm_hi_lo();
        let hi_hi = self.mm_hi_hi();
        let shifted = self.mm_shifted();
        let add_term = self.mm_add_term();
        let sub_term = self.mm_sub_term();
        let add_ovf = self.mm_add_ovf();
        let sub_ovf = self.am_ovf();
        let corr = self.am_corr();
        let less = self.am_less();
        let tmp = self.am_tmp();

        // 1. lo = low 64 bits of a*b
        self.asm.emit_r::<MUL>(lo, a, b);
        // 2. hi = high 64 bits of a*b
        self.asm.emit_r::<MULHU>(hi, a, b);
        // 3. hi_hi = hi >> 32
        self.asm.emit_i::<SRLI>(hi_hi, hi, 32);
        // 4-5. hi_lo = (hi << 32) >> 32  -- zero-extends the low 32 bits
        self.asm.emit_i::<SLLI>(hi_lo, hi, 32);
        self.asm.emit_i::<SRLI>(hi_lo, hi_lo, 32);
        // 6. shifted = hi_lo << 32
        self.asm.emit_i::<SLLI>(shifted, hi_lo, 32);
        // 7. add_term = lo + shifted (wrapping)
        self.asm.emit_r::<ADD>(add_term, lo, shifted);
        // 8. add_ovf = (add_term < lo) ? 1 : 0  -- detect 2^64 overflow
        self.asm.emit_r::<SLTU>(add_ovf, add_term, lo);
        // 9. sub_term = hi_lo + hi_hi
        self.asm.emit_r::<ADD>(sub_term, hi_lo, hi_hi);
        // 10. r = add_term - sub_term (wrapping)
        self.asm.emit_r::<SUB>(dst, add_term, sub_term);
        // 11. sub_ovf = (add_term < sub_term) ? 1 : 0
        self.asm.emit_r::<SLTU>(sub_ovf, add_term, sub_term);
        // 12-14. If underflow, add P. Compute corr = (0 - sub_ovf) AND P; r += corr.
        self.asm.emit_r::<SUB>(corr, 0, sub_ovf);
        self.asm.emit_r::<AND>(corr, corr, p);
        self.asm.emit_r::<ADD>(dst, dst, corr);
        // 15-21. add_term-overflow correction: if add_ovf, add
        //   (2^32 - 1). Snapshot dst, do the conditional add, detect
        //   wrap, conditional second add of (2^32 - 1).
        self.asm.emit_r::<SUB>(corr, 0, add_ovf);
        self.asm.emit_r::<AND>(corr, corr, mask); // corr = (2^32-1) if add_ovf else 0
        self.asm.emit_r::<ADD>(tmp, dst, 0); // snapshot dst for wrap detection
        self.asm.emit_r::<ADD>(dst, dst, corr); // dst += corr
        self.asm.emit_r::<SLTU>(less, dst, tmp); // less = 1 if wrap (dst < snapshot)
        self.asm.emit_r::<SUB>(corr, 0, less);
        self.asm.emit_r::<AND>(corr, corr, mask);
        self.asm.emit_r::<ADD>(dst, dst, corr); // if wrap, add another (2^32-1)
                                                // 22-26. Final reduction: if dst >= P, dst -= P.
                                                //   After all corrections, dst < 2^64 < 2P, so one sub
                                                //   suffices.
        self.asm.emit_r::<SLTU>(less, dst, p);
        self.asm.emit_r::<SUB>(tmp, dst, p);
        self.asm.emit_r::<SUB>(corr, 0, less);
        self.asm.emit_r::<AND>(corr, corr, p);
        self.asm.emit_r::<ADD>(dst, tmp, corr);
    }

    /// dst = x^7 over Goldilocks. 4 mul_mod calls.
    ///
    /// Now that `mul_mod_into` correctly handles `add_term` overflow,
    /// the natural `x^7 = x^6 * x` decomposition works correctly.
    /// (Previously this triggered a bug in `mul_mod_into`; the
    /// workaround was the `x^7 = x^4 * x^3` decomposition. Fixed in
    /// the v1 mul_mod correction.)
    fn sbox_into(&mut self, dst: u8, x: u8) {
        let x2 = self.sc_a();
        let x4 = self.sc_b();
        let x6 = self.sc_c();

        // x2 = x * x
        self.mul_mod_into(x2, x, x);
        // x4 = x2 * x2
        self.mul_mod_into(x4, x2, x2);
        // x6 = x4 * x2
        self.mul_mod_into(x6, x4, x2);
        // dst = x6 * x = x^7
        self.mul_mod_into(dst, x6, x);
    }

    // ── Round-constant loading ────────────────────────────────────────

    /// Load round constant at index `idx` into sc_rc from rs2 base.
    /// 1 instruction.
    fn load_rc(&mut self, idx: usize) {
        let dst = self.sc_rc();
        self.asm
            .emit_ld::<LD>(dst, self.operands.rs2, (idx * 8) as i64);
    }

    /// Load the i'th diagonal constant into sc_diag.
    /// Uses inline u64 immediate construction since DIAG only has 8 entries.
    fn load_diag(&mut self, i: usize) {
        let value = POSEIDON2_INTERNAL_DIAG[i];
        let dst = self.sc_diag();
        self.load_u64_immediate(dst, value);
    }

    /// Load a 64-bit constant into `dst`. Single instruction: the Jolt
    /// inline `emit_i::<ADDI>` accepts a full u64 immediate.
    fn load_u64_immediate(&mut self, dst: u8, value: u64) {
        self.asm.emit_i::<ADDI>(dst, 0, value);
    }

    /// For each i in 0..8: S[i] = (S[i] + RC[rc_base + i]) mod P.
    fn add_round_constants_full(&mut self, rc_base: usize) {
        for i in 0..STATE_LEN {
            self.load_rc(rc_base + i);
            let s_i = self.s(i);
            let rc = self.sc_rc();
            self.add_mod_into(s_i, s_i, rc);
        }
    }

    /// S[0] = (S[0] + RC[idx]) mod P. Used in internal rounds.
    fn add_round_constant_partial(&mut self, idx: usize) {
        self.load_rc(idx);
        let s0 = self.s(0);
        let rc = self.sc_rc();
        self.add_mod_into(s0, s0, rc);
    }

    /// For each i in 0..8: S[i] = sbox(S[i]).
    fn sbox_full(&mut self) {
        for i in 0..STATE_LEN {
            let s_i = self.s(i);
            self.sbox_into(s_i, s_i);
        }
    }

    fn sbox_state_0(&mut self) {
        let s0 = self.s(0);
        self.sbox_into(s0, s0);
    }

    // ── External MDS ──────────────────────────────────────────────────

    /// Apply the m4 sub-block to four registers in place.
    ///
    /// Reference (from exec.rs::external_mds):
    /// ```text
    /// let (a, b, c, d) = (s[0], s[1], s[2], s[3]);
    /// let sum = add_mod(add_mod(a, b), add_mod(c, d));
    /// s[0] = add_mod(sum, add_mod(a, add_mod(b, b)));
    /// s[1] = add_mod(sum, add_mod(b, add_mod(c, c)));
    /// s[2] = add_mod(sum, add_mod(c, add_mod(d, d)));
    /// s[3] = add_mod(sum, add_mod(d, add_mod(a, a)));
    /// ```
    ///
    /// We snapshot a,b,c,d into named scratch (sc_a/b/c/d) so we can
    /// freely overwrite s[].
    fn m4_apply(&mut self, s: [u8; 4]) {
        // Snapshot inputs.
        let a = self.sc_a();
        let b = self.sc_b();
        let c = self.sc_c();
        let d = self.sc_d();
        self.asm.emit_r::<ADD>(a, s[0], 0);
        self.asm.emit_r::<ADD>(b, s[1], 0);
        self.asm.emit_r::<ADD>(c, s[2], 0);
        self.asm.emit_r::<ADD>(d, s[3], 0);

        // sum = (a+b) + (c+d), using a temp register (mm_lo is free here).
        let ab = self.mm_lo();
        let cd = self.mm_hi();
        let sum = self.sum_reg();
        self.add_mod_into(ab, a, b);
        self.add_mod_into(cd, c, d);
        self.add_mod_into(sum, ab, cd);

        // s[0] = sum + a + 2b
        let bb = self.mm_hi_lo();
        let a_plus_bb = self.mm_hi_hi();
        self.add_mod_into(bb, b, b);
        self.add_mod_into(a_plus_bb, a, bb);
        self.add_mod_into(s[0], sum, a_plus_bb);

        // s[1] = sum + b + 2c
        let cc = self.mm_hi_lo();
        let b_plus_cc = self.mm_hi_hi();
        self.add_mod_into(cc, c, c);
        self.add_mod_into(b_plus_cc, b, cc);
        self.add_mod_into(s[1], sum, b_plus_cc);

        // s[2] = sum + c + 2d
        let dd = self.mm_hi_lo();
        let c_plus_dd = self.mm_hi_hi();
        self.add_mod_into(dd, d, d);
        self.add_mod_into(c_plus_dd, c, dd);
        self.add_mod_into(s[2], sum, c_plus_dd);

        // s[3] = sum + d + 2a
        let aa = self.mm_hi_lo();
        let d_plus_aa = self.mm_hi_hi();
        self.add_mod_into(aa, a, a);
        self.add_mod_into(d_plus_aa, d, aa);
        self.add_mod_into(s[3], sum, d_plus_aa);
    }

    /// External MDS: two m4 sub-blocks (on left and right halves of the
    /// state) then the cross-mix described in exec.rs::external_mds.
    fn external_mds(&mut self) {
        let left: [u8; 4] = [self.s(0), self.s(1), self.s(2), self.s(3)];
        let right: [u8; 4] = [self.s(4), self.s(5), self.s(6), self.s(7)];

        self.m4_apply(left);
        self.m4_apply(right);

        // After m4, left = transformed(S[0..4]), right = transformed(S[4..8]).
        // Reference:
        //   for i in 0..4 { state[i] = left[i] + right[i]; state[i+4] = left[i] + right[i]; }
        //   for i in 0..4 { state[i] = state[i] + left[i]; state[i+4] = state[i+4] + right[i]; }
        //
        // After these two passes:
        //   state[i]   = (left[i] + right[i]) + left[i]  = 2*left[i] + right[i]
        //   state[i+4] = (left[i] + right[i]) + right[i] = left[i] + 2*right[i]
        //
        // Implement directly: snapshot left[i] and right[i] before
        // overwriting them.
        for i in 0..4 {
            // We can't freely snapshot here because left/right ARE the
            // state registers. Use t[i] and t[i+4] as scratch.
            let l = left[i]; // = S[i]
            let r = right[i]; // = S[i+4]
            let t_l = self.t(i);
            let t_r = self.t(i + 4);

            // t_l = l, t_r = r
            self.asm.emit_r::<ADD>(t_l, l, 0);
            self.asm.emit_r::<ADD>(t_r, r, 0);

            // l = 2*t_l + t_r  =  (t_l + t_r) + t_l
            let sum_lr = self.sc_a();
            self.add_mod_into(sum_lr, t_l, t_r);
            self.add_mod_into(l, sum_lr, t_l);

            // r = t_l + 2*t_r  =  (t_l + t_r) + t_r
            self.add_mod_into(r, sum_lr, t_r);
        }
    }

    // ── Internal diffusion ────────────────────────────────────────────

    /// Compute row-sum into sum_reg, then state[i] = diag[i]*state[i] + sum.
    fn internal_diffusion(&mut self) {
        // 1. sum = S[0] + S[1] + ... + S[7]
        let sum = self.sum_reg();
        // Start with sum = S[0]+S[1].
        let s0 = self.s(0);
        let s1 = self.s(1);
        self.add_mod_into(sum, s0, s1);
        for i in 2..STATE_LEN {
            let s_i = self.s(i);
            self.add_mod_into(sum, sum, s_i);
        }

        // 2. For each i in 0..8: S[i] = (diag[i] * S[i]) + sum.
        for i in 0..STATE_LEN {
            self.load_diag(i);
            let diag = self.sc_diag();
            let s_i = self.s(i);
            // S[i] = diag * S[i]
            self.mul_mod_into(s_i, diag, s_i);
            // S[i] = S[i] + sum
            self.add_mod_into(s_i, s_i, sum);
        }
    }
}

// Compile-time sanity: the constants we pull in MUST be the same shape
// the reference implementation uses. If these `_` bindings fail to
// type-check, the implementer has the wrong constants.
const _: [u64; STATE_WIDTH] = POSEIDON2_INTERNAL_DIAG;

// ── Test-only helpers for sub-operation isolation ────────────────────

#[cfg(test)]
#[allow(dead_code)]
impl Poseidon2GoldilocksSequenceBuilder {
    pub fn new_for_test(asm: InstrAssembler, operands: FormatInline) -> Self {
        Self::new(asm, operands)
    }

    pub fn test_load_p_and_state_and_add_rc_full(&mut self, rc_base: usize) {
        self.load_p();
        self.load_state();
        self.add_round_constants_full(rc_base);
    }

    pub fn test_load_p_state_addrc_sbox(&mut self, rc_base: usize) {
        self.load_p();
        self.load_state();
        self.add_round_constants_full(rc_base);
        self.sbox_full();
    }

    pub fn test_load_p_state_addrc_sbox_mds(&mut self, rc_base: usize) {
        self.load_p();
        self.load_state();
        self.add_round_constants_full(rc_base);
        self.sbox_full();
        self.external_mds();
    }

    pub fn test_load_p_state_mds_only(&mut self) {
        self.load_p();
        self.load_state();
        self.external_mds();
    }

    pub fn test_load_p_state_intdiff_only(&mut self) {
        self.load_p();
        self.load_state();
        self.internal_diffusion();
    }

    /// For each i in 0..8: S[i] = S[i] * S[i] mod P (single squaring).
    pub fn test_load_p_state_square_only(&mut self) {
        self.load_p();
        self.load_state();
        for i in 0..STATE_LEN {
            let s_i = self.s(i);
            self.mul_mod_into(s_i, s_i, s_i);
        }
    }

    /// For each i in 0..8: S[i] = sbox(S[i]) (without the surrounding RC add).
    pub fn test_load_p_state_sbox_only(&mut self) {
        self.load_p();
        self.load_state();
        self.sbox_full();
    }

    /// For each i: S[i] = x^4 mod P (just two squares).
    pub fn test_load_p_state_x4_only(&mut self) {
        self.load_p();
        self.load_state();
        for i in 0..STATE_LEN {
            let s_i = self.s(i);
            let scratch = self.sc_a();
            self.mul_mod_into(scratch, s_i, s_i); // x^2
            self.mul_mod_into(s_i, scratch, scratch); // x^4
        }
    }

    /// For each i: S[i] = x^3 = x^2 * x. Tests asymmetric mul.
    pub fn test_load_p_state_x3(&mut self) {
        self.load_p();
        self.load_state();
        for i in 0..STATE_LEN {
            let s_i = self.s(i);
            let scratch = self.sc_a();
            self.mul_mod_into(scratch, s_i, s_i); // x^2
            self.mul_mod_into(s_i, scratch, s_i); // x^3 = x^2 * x
        }
    }

    /// For each i: S[i] = x^6 = x^4 * x^2.
    pub fn test_load_p_state_x6(&mut self) {
        self.load_p();
        self.load_state();
        for i in 0..STATE_LEN {
            let s_i = self.s(i);
            let x2 = self.sc_a();
            let x4 = self.sc_b();
            self.mul_mod_into(x2, s_i, s_i); // x^2
            self.mul_mod_into(x4, x2, x2); // x^4
            self.mul_mod_into(s_i, x4, x2); // x^6 = x^4 * x^2
        }
    }

    /// For each i: S[i] = x * x (square), but written as
    /// scratch = x; mul_mod(s_i, scratch, s_i). Tests dst-aliases-b
    /// when a is a non-aliasing reg.
    /// For each i: S[i] = S[i] * S[i+8] mod P. Used to stress
    /// mul_mod_into with arbitrary (a, b) pairs supplied by the test
    /// harness. The state's first 8 elements are a's, next 8 are b's.
    /// Output is written back to the first 8 (overwriting a's).
    ///
    /// Note: this requires a 16-element state buffer at rs1. The
    /// emulator test allocates 128 bytes for this.
    pub fn test_load_p_state_mul_pairs(&mut self) {
        self.load_p();
        // Load 16 u64s from rs1.
        for i in 0..8 {
            let s_i = self.s(i);
            self.asm
                .emit_ld::<LD>(s_i, self.operands.rs1, (i * 8) as i64);
        }
        // Load the b's into T[0..8] (vr[8..16]).
        for i in 0..8 {
            let t_i = self.t(i);
            self.asm
                .emit_ld::<LD>(t_i, self.operands.rs1, ((i + 8) * 8) as i64);
        }
        // For each i: S[i] = S[i] * T[i] mod P
        for i in 0..8 {
            let s_i = self.s(i);
            let t_i = self.t(i);
            self.mul_mod_into(s_i, s_i, t_i);
        }
        // Store the 8 results back to the first 8 slots.
        for i in 0..8 {
            let s_i = self.s(i);
            self.asm
                .emit_s::<SD>(self.operands.rs1, s_i, (i * 8) as i64);
        }
    }

    pub fn test_load_p_state_mul_dst_aliases_b(&mut self) {
        self.load_p();
        self.load_state();
        for i in 0..STATE_LEN {
            let s_i = self.s(i);
            let scratch = self.sc_a();
            // scratch = s_i (just a copy via ADD scratch, s_i, 0)
            self.asm.emit_r::<ADD>(scratch, s_i, 0);
            // s_i = scratch * s_i (dst aliases b)
            self.mul_mod_into(s_i, scratch, s_i);
        }
    }

    /// For each i: S[i] = x^7 via INLINED sbox using the natural
    /// `x^7 = x^6 * x` decomposition. Previously diagnosed as
    /// triggering a `mul_mod_into` bug; now fixed by the add_term-
    /// overflow correction. This test stays in place as a regression
    /// guard.
    pub fn test_load_p_state_sbox_inlined(&mut self) {
        self.load_p();
        self.load_state();
        for i in 0..STATE_LEN {
            let s_i = self.s(i);
            let a = self.sc_a();
            let b = self.sc_b();
            let c = self.sc_c();
            self.mul_mod_into(a, s_i, s_i); // a = x^2
            self.mul_mod_into(b, a, a); // b = x^4
            self.mul_mod_into(c, b, a); // c = x^6 = x^4 * x^2
            self.mul_mod_into(s_i, c, s_i); // s_i = x^7 = x^6 * x
        }
    }

    pub fn test_store_and_finalize(mut self) -> Vec<Instruction> {
        self.store_state();
        drop(self.vr);
        self.asm.finalize_inline()
    }
}
