//! Lower [`PrefixMleFormula`] variants into [`ScalarExpr`] expressions.
//!
//! Each (prefix, round, c-side) combination is lowered to a monomial sum over
//! challenges, checkpoints, and precomputed per-`b` mask buffers. Evaluating
//! the expression at index `b ∈ [0, 2^b_len)` with `eval_scalar_expr` yields
//! the same field value as the interpretive `eval_prefix_mle`.
//!
//! Round geometry (j, b_len, c) is known at compile time, so the lowering
//! folds constants, bakes in pow2 coefficients, and selects between even/odd
//! branches. Runtime evaluation is then a flat walk over a ScalarExpr.

use std::collections::HashMap;

use crate::module::{
    BilinearExpr, ChallengeIdx, Comparison, DefaultVal, IntBitOp, Monomial, PrefixMleFormula,
    PrefixMleRule, RemainingTest, RoundGuard, ScalarExpr, ValueSource, WeightFn,
};
use crate::PolynomialId;

// ── Bit-manipulation helpers (mirror checkpoint_eval.rs) ────────────────────

fn uninterleave_bits(val: u128) -> (u64, u64) {
    let mut x = (val >> 1) & 0x5555_5555_5555_5555_5555_5555_5555_5555;
    let mut y = val & 0x5555_5555_5555_5555_5555_5555_5555_5555;
    x = (x | (x >> 1)) & 0x3333_3333_3333_3333_3333_3333_3333_3333;
    x = (x | (x >> 2)) & 0x0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F;
    x = (x | (x >> 4)) & 0x00FF_00FF_00FF_00FF_00FF_00FF_00FF_00FF;
    x = (x | (x >> 8)) & 0x0000_FFFF_0000_FFFF_0000_FFFF_0000_FFFF;
    x = (x | (x >> 16)) & 0x0000_0000_FFFF_FFFF_0000_0000_FFFF_FFFF;
    x = (x | (x >> 32)) & 0x0000_0000_0000_0000_FFFF_FFFF_FFFF_FFFF;
    y = (y | (y >> 1)) & 0x3333_3333_3333_3333_3333_3333_3333_3333;
    y = (y | (y >> 2)) & 0x0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F;
    y = (y | (y >> 4)) & 0x00FF_00FF_00FF_00FF_00FF_00FF_00FF_00FF;
    y = (y | (y >> 8)) & 0x0000_FFFF_0000_FFFF_0000_FFFF_0000_FFFF;
    y = (y | (y >> 16)) & 0x0000_0000_FFFF_FFFF_0000_0000_FFFF_FFFF;
    y = (y | (y >> 32)) & 0x0000_0000_0000_0000_FFFF_FFFF_FFFF_FFFF;
    (x as u64, y as u64)
}

fn uninterleave_with_lens(val: u128, len: usize) -> ((u64, usize), (u64, usize)) {
    let (x, y) = uninterleave_bits(val);
    let x_len = len / 2;
    let y_len = len - x_len;
    let x_masked = if x_len < 64 {
        x & ((1u64 << x_len) - 1)
    } else {
        x
    };
    let y_masked = if y_len < 64 {
        y & ((1u64 << y_len) - 1)
    } else {
        y
    };
    ((x_masked, x_len), (y_masked, y_len))
}

fn eval_remaining_test(test: RemainingTest, x: u64, y: u64, y_len: usize) -> bool {
    match test {
        RemainingTest::Equality => x == y,
        RemainingTest::LeftZero => x == 0,
        RemainingTest::RightZero => y == 0,
        RemainingTest::LeftZeroRightAllOnes => x == 0 && y == (1u64 << y_len) - 1,
        RemainingTest::Always => true,
    }
}

// ── Mask roles ─────────────────────────────────────────────────────────────

/// A mask role names a per-`b` integer function computable from `(b, b_len)`.
///
/// Each `(role, b_len)` pair materializes as a distinct preprocessed buffer
/// (`PolynomialId::PrefixMask(id)`) with `2^b_len` entries.
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum MaskRole {
    /// `pop_msb(b)` — the MSB bit of `b` as a length-`b_len` value.
    /// Returns 0 if `b_len == 0`.
    YMsb,
    /// `I(remaining_test(...))`. If `pop_msb` is true, pops the MSB of `b`
    /// before uninterleaving (even-round semantics).
    RemainingIndicator { test: RemainingTest, pop_msb: bool },
    /// `I(((rest << suffix_len) >> word_bits) == 0)` where `rest` is `b` with
    /// its MSB optionally popped off. Matches `OverflowBitsZero` semantics.
    OverflowRestOk {
        word_bits: usize,
        suffix_len: usize,
        pop_msb: bool,
    },
    /// `(b >> bit) & 1` — extract a specific bit position of `b`.
    BitExtract { bit: u32 },
    /// `I(b.trailing_zeros() >= k)` — trailing-zero-count threshold indicator.
    TrailingZerosGe { k: u32 },
    /// Shifted bit-op on uninterleaved remaining halves. Matches the additive
    /// constant term produced by `BitwiseAdditive`:
    ///
    /// `rotation == 0`: `eval_bit_op(op, rem_x, rem_y) << (suffix_len / 2)`
    /// `rotation != 0`: `eval_bit_op(op, rem_x, rem_y).rotate_left(shift)`
    BitOpShifted {
        op: IntBitOp,
        rotation: u32,
        suffix_len: usize,
        word_bits: usize,
        pop_msb: bool,
    },
    /// Remaining-bits-in-word term from `OperandExtract`:
    ///
    /// `is_upper == true`: positional scale depending on `suffix_len` vs `word_bits`.
    /// `is_upper == false`: `b_eff << suffix_len` (128-bit).
    RemainingInWord {
        suffix_len: usize,
        word_bits: usize,
        is_upper: bool,
        pop_msb: bool,
    },
    /// `RightOperandExtract` additive term: `(y_masked << (suffix_len / 2))`
    /// where `y_masked` is the y-half of `uninterleave(b_eff, b_len_eff)`.
    YMaskedShifted { suffix_len: usize, pop_msb: bool },
    /// `Pow2` mask: `2^(b & word_mask)`.
    Pow2Masked { word_mask: usize },
    /// `Rev8W` per-b term: `rev8w((b as u64) << suffix_len) as u128`.
    Rev8wShifted { suffix_len: usize },
    /// Comparison indicator: `I(cmp_test(rem_x, rem_y))`.
    CmpIndicator { cmp: Comparison, pop_msb: bool },
    /// Leading-ones power: `2^leading_ones(y_rem, y_len)` where
    /// `(_, y_rem, y_len) = uninterleave(b_eff, len_eff)` and `b_eff` is `b`
    /// with the MSB optionally popped.
    LeadingOnesPow2 { pop_msb: bool },
    /// RightShift per-b term: `x_rem >> trailing_zeros_val(y_rem, y_len)` as u32.
    XRemShiftedByTZ { pop_msb: bool },
    /// LeftShift per-b term: `(x_rem & !y_rem).unbounded_shl(lo + wb-1-j/2 - y_len)`.
    XMaskedShifted {
        word_bits: usize,
        j: usize,
        pop_msb: bool,
    },
    /// `SignExtension` per-b mask: sum over zero bits of y of `2^(base_index+1+i)`
    /// where `i=0` is MSB.
    YComplementSum { base_index: usize, pop_msb: bool },
    /// Like `RemainingIndicator` but always evaluates with `y_len=0`.
    /// Used by `SignGatedMultiplicative` at jr=0,1 where the remaining test
    /// runs on freshly-uninterleaved bits before jr>=2 geometry applies.
    RemainingIndicatorYZero { test: RemainingTest, pop_msb: bool },
}

/// Compute the raw integer value of a mask at index `b` for `b_len` bits.
pub fn compute_mask_value(role: MaskRole, b: u128, b_len: usize) -> u128 {
    match role {
        MaskRole::YMsb => {
            if b_len == 0 {
                0
            } else {
                (b >> (b_len - 1)) & 1
            }
        }
        MaskRole::RemainingIndicator { test, pop_msb } => {
            let (b_eff, len_eff) = if pop_msb {
                if b_len == 0 {
                    (b, 0)
                } else {
                    let rest = b & ((1u128 << (b_len - 1)) - 1);
                    (rest, b_len - 1)
                }
            } else {
                (b, b_len)
            };
            let ((rem_x, _), (rem_y, y_len)) = uninterleave_with_lens(b_eff, len_eff);
            u128::from(eval_remaining_test(test, rem_x, rem_y, y_len))
        }
        MaskRole::OverflowRestOk {
            word_bits,
            suffix_len,
            pop_msb,
        } => {
            let rest = if pop_msb && b_len > 0 {
                b & ((1u128 << (b_len - 1)) - 1)
            } else {
                b
            };
            let shifted = rest << suffix_len;
            u128::from((shifted >> word_bits) == 0)
        }
        MaskRole::BitExtract { bit } => (b >> bit) & 1,
        MaskRole::TrailingZerosGe { k } => u128::from(b.trailing_zeros() >= k),
        MaskRole::BitOpShifted {
            op,
            rotation,
            suffix_len,
            word_bits,
            pop_msb,
        } => {
            let (b_eff, len_eff) = if pop_msb && b_len > 0 {
                (b & ((1u128 << (b_len - 1)) - 1), b_len - 1)
            } else {
                (b, b_len)
            };
            let ((rem_x, _), (rem_y, _)) = uninterleave_with_lens(b_eff, len_eff);
            let op_val = eval_bit_op(op, rem_x, rem_y);
            let out = if rotation == 0 {
                op_val << (suffix_len / 2)
            } else {
                let half_suffix = suffix_len as i32 / 2;
                let rot = rotation as i32;
                let wb = word_bits as i32;
                let shift = if half_suffix - rot >= 0 {
                    (half_suffix - rot) as u32
                } else {
                    (wb + half_suffix - rot) as u32
                };
                op_val.rotate_left(shift)
            };
            u128::from(out)
        }
        MaskRole::Pow2Masked { word_mask } => {
            let b_masked = (b as usize) & word_mask;
            1u128 << b_masked
        }
        MaskRole::Rev8wShifted { suffix_len } => {
            let shifted = (b as u64).wrapping_shl(suffix_len as u32);
            u128::from(crate::module::rev8w(shifted))
        }
        MaskRole::CmpIndicator { cmp, pop_msb } => {
            let (b_eff, len_eff) = if pop_msb && b_len > 0 {
                (b & ((1u128 << (b_len - 1)) - 1), b_len - 1)
            } else {
                (b, b_len)
            };
            let ((rem_x, _), (rem_y, _)) = uninterleave_with_lens(b_eff, len_eff);
            u128::from(cmp_test(cmp, rem_x, rem_y))
        }
        MaskRole::LeadingOnesPow2 { pop_msb } => {
            let (b_eff, len_eff) = if pop_msb && b_len > 0 {
                (b & ((1u128 << (b_len - 1)) - 1), b_len - 1)
            } else {
                (b, b_len)
            };
            let (_, (y_rem, y_len)) = uninterleave_with_lens(b_eff, len_eff);
            1u128 << leading_ones_val(y_rem, y_len)
        }
        MaskRole::XRemShiftedByTZ { pop_msb } => {
            let (b_eff, len_eff) = if pop_msb && b_len > 0 {
                (b & ((1u128 << (b_len - 1)) - 1), b_len - 1)
            } else {
                (b, b_len)
            };
            let ((x_rem, _), (y_rem, y_len)) = uninterleave_with_lens(b_eff, len_eff);
            u128::from((x_rem as u32) >> trailing_zeros_val(y_rem, y_len))
        }
        MaskRole::XMaskedShifted {
            word_bits,
            j,
            pop_msb,
        } => {
            let (b_eff, len_eff) = if pop_msb && b_len > 0 {
                (b & ((1u128 << (b_len - 1)) - 1), b_len - 1)
            } else {
                (b, b_len)
            };
            let ((x_rem, _), (y_rem, y_len)) = uninterleave_with_lens(b_eff, len_eff);
            let x_masked = x_rem & !y_rem;
            let lo = leading_ones_val(y_rem, y_len) as usize;
            let shift = (lo + word_bits - 1 - j / 2 - y_len) as u32;
            u128::from(x_masked.unbounded_shl(shift))
        }
        MaskRole::YComplementSum {
            base_index,
            pop_msb,
        } => {
            let (b_eff, len_eff) = if pop_msb && b_len > 0 {
                (b & ((1u128 << (b_len - 1)) - 1), b_len - 1)
            } else {
                (b, b_len)
            };
            let (_, (y_val, y_len)) = uninterleave_with_lens(b_eff, len_eff);
            let mut result: u128 = 0;
            let mut index = base_index;
            for i in 0..y_len {
                index += 1;
                let y_i = (y_val >> (y_len - 1 - i)) & 1;
                if y_i == 0 {
                    result += 1u128 << index;
                }
            }
            result
        }
        MaskRole::RemainingIndicatorYZero { test, pop_msb } => {
            let (b_eff, len_eff) = if pop_msb && b_len > 0 {
                (b & ((1u128 << (b_len - 1)) - 1), b_len - 1)
            } else {
                (b, b_len)
            };
            let ((rem_x, _), (rem_y, _)) = uninterleave_with_lens(b_eff, len_eff);
            u128::from(eval_remaining_test(test, rem_x, rem_y, 0))
        }
        MaskRole::YMaskedShifted {
            suffix_len,
            pop_msb,
        } => {
            let (b_eff, len_eff) = if pop_msb && b_len > 0 {
                (b & ((1u128 << (b_len - 1)) - 1), b_len - 1)
            } else {
                (b, b_len)
            };
            let (_, (y_masked, _)) = uninterleave_with_lens(b_eff, len_eff);
            (y_masked as u128) << (suffix_len / 2)
        }
        MaskRole::RemainingInWord {
            suffix_len,
            word_bits,
            is_upper,
            pop_msb,
        } => {
            let b_eff = if pop_msb && b_len > 0 {
                b & ((1u128 << (b_len - 1)) - 1)
            } else {
                b
            };
            if is_upper {
                match suffix_len.cmp(&word_bits) {
                    std::cmp::Ordering::Greater => {
                        u128::from((b_eff << (suffix_len - word_bits)) as u64)
                    }
                    std::cmp::Ordering::Less => {
                        u128::from((b_eff >> (word_bits - suffix_len)) as u64)
                    }
                    std::cmp::Ordering::Equal => u128::from(b_eff as u64),
                }
            } else {
                b_eff << suffix_len
            }
        }
    }
}

fn eval_bit_op(op: IntBitOp, x: u64, y: u64) -> u64 {
    match op {
        IntBitOp::And => x & y,
        IntBitOp::AndNot => x & !y,
        IntBitOp::Or => x | y,
        IntBitOp::Xor => x ^ y,
    }
}

fn cmp_test(cmp: Comparison, a: u64, b: u64) -> bool {
    match cmp {
        Comparison::LessThan => a < b,
        Comparison::GreaterThan => a > b,
    }
}

fn anti_pair_for_cmp(cmp: Comparison) -> BilinearExpr {
    match cmp {
        Comparison::LessThan => BilinearExpr::AntiXY,
        Comparison::GreaterThan => BilinearExpr::AntiYX,
    }
}

fn leading_ones_val(val: u64, len: usize) -> u32 {
    if len == 0 {
        return 0;
    }
    (val.wrapping_shl(64 - len as u32)).leading_ones()
}

fn trailing_zeros_val(val: u64, len: usize) -> u32 {
    if len == 0 || val == 0 {
        return len as u32;
    }
    std::cmp::min(val.trailing_zeros(), len as u32)
}

fn weight_eval(w: WeightFn, j: usize) -> i128 {
    match w {
        WeightFn::Positional {
            rotation,
            word_bits,
            j_offset,
        } => {
            let pos = (j - j_offset) / 2;
            let rotated = (pos + rotation as usize) % word_bits as usize;
            let shift = word_bits as usize - 1 - rotated;
            1i128 << shift
        }
        WeightFn::LinearJ { base } => 1i128 << (base - j),
        WeightFn::LinearJMinusOne { base } => 1i128 << (base - j - 1),
        WeightFn::HalfJ => 1i128 << (j / 2),
    }
}

/// Fill `out` with mask values for `b in 0..2^b_len`.
pub fn fill_mask_buffer(role: MaskRole, b_len: usize, out: &mut Vec<u128>) {
    let n = 1usize << b_len;
    out.clear();
    out.reserve(n);
    for b in 0..n {
        out.push(compute_mask_value(role, b as u128, b_len));
    }
}

// ── Bilinear expansion with mixed constant/factor operands ─────────────────

/// An operand in a bilinear expression: either a compile-time 0/1 constant
/// or a runtime value source (challenge, indexed mask, etc.).
#[derive(Clone, Debug)]
enum Operand {
    /// Constant 0 or 1.
    Const01(u32),
    /// Runtime value source.
    Factor(ValueSource),
}

fn expr_of(op: &Operand) -> ScalarExpr {
    match op {
        Operand::Const01(0) => vec![],
        Operand::Const01(c) => vec![Monomial {
            coeff: *c as i128,
            factors: vec![],
        }],
        Operand::Factor(f) => vec![Monomial {
            coeff: 1,
            factors: vec![f.clone()],
        }],
    }
}

/// ScalarExpr equal to `1 − op`.
fn one_minus_expr(op: &Operand) -> ScalarExpr {
    match op {
        Operand::Const01(0) => one_expr(),
        Operand::Const01(1) => vec![],
        Operand::Const01(c) => vec![
            Monomial {
                coeff: 1,
                factors: vec![],
            },
            Monomial {
                coeff: -(*c as i128),
                factors: vec![],
            },
        ],
        // Challenge ↔ OneMinusChallenge is a single-factor form.
        Operand::Factor(ValueSource::Challenge(ci)) => vec![Monomial {
            coeff: 1,
            factors: vec![ValueSource::OneMinusChallenge(*ci)],
        }],
        Operand::Factor(ValueSource::OneMinusChallenge(ci)) => vec![Monomial {
            coeff: 1,
            factors: vec![ValueSource::Challenge(*ci)],
        }],
        Operand::Factor(f) => vec![
            Monomial {
                coeff: 1,
                factors: vec![],
            },
            Monomial {
                coeff: -1,
                factors: vec![f.clone()],
            },
        ],
    }
}

fn one_expr() -> ScalarExpr {
    vec![Monomial {
        coeff: 1,
        factors: vec![],
    }]
}

/// Pointwise multiply two ScalarExprs (expand as sum of monomial products).
fn mul_exprs(a: &ScalarExpr, b: &ScalarExpr) -> ScalarExpr {
    let mut out = Vec::with_capacity(a.len() * b.len());
    for ma in a {
        for mb in b {
            let coeff = ma.coeff.wrapping_mul(mb.coeff);
            if coeff == 0 {
                continue;
            }
            let mut factors = Vec::with_capacity(ma.factors.len() + mb.factors.len());
            factors.extend_from_slice(&ma.factors);
            factors.extend_from_slice(&mb.factors);
            out.push(Monomial { coeff, factors });
        }
    }
    out
}

fn add_exprs(mut a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    a.extend(b);
    a
}

fn negate_expr(mut e: ScalarExpr) -> ScalarExpr {
    for m in &mut e {
        m.coeff = m.coeff.wrapping_neg();
    }
    e
}

/// Expand `expr(x, y)` where x, y are mixed const/factor operands.
fn bilinear_expand(expr: BilinearExpr, x: &Operand, y: &Operand) -> ScalarExpr {
    let x_e = expr_of(x);
    let y_e = expr_of(y);
    let omx = one_minus_expr(x);
    let omy = one_minus_expr(y);
    match expr {
        BilinearExpr::Product => mul_exprs(&x_e, &y_e),
        BilinearExpr::AntiXY => mul_exprs(&omx, &y_e),
        BilinearExpr::AntiYX => mul_exprs(&x_e, &omy),
        BilinearExpr::NorBit => mul_exprs(&omx, &omy),
        BilinearExpr::EqBit => add_exprs(mul_exprs(&x_e, &y_e), mul_exprs(&omx, &omy)),
        BilinearExpr::XorBit => add_exprs(mul_exprs(&omx, &y_e), mul_exprs(&x_e, &omy)),
        BilinearExpr::OrBit => add_exprs(
            add_exprs(x_e.clone(), y_e.clone()),
            negate_expr(mul_exprs(&x_e, &y_e)),
        ),
        BilinearExpr::OneMinusX => omx,
        BilinearExpr::OneMinusY => omy,
        BilinearExpr::OnePlusY => add_exprs(one_expr(), y_e),
        BilinearExpr::X => x_e,
        BilinearExpr::Y => y_e,
    }
}

fn prepend_factor(mut e: ScalarExpr, f: ValueSource) -> ScalarExpr {
    for m in &mut e {
        m.factors.insert(0, f.clone());
    }
    e
}

// ── Public: lowering dispatcher ────────────────────────────────────────────

/// Inputs that define a single (prefix, round, c-side) lowering task.
#[derive(Clone, Copy, Debug)]
pub struct LoweringCtx {
    /// Current sumcheck round within the batched instance.
    pub j: usize,
    /// Bits of `b` at this round (= log₂(current_len) − 1).
    pub b_len: usize,
    /// Total instance bits (phases × chunk_bits).
    pub total_bits: usize,
    /// Bound x-challenge when `j` is odd, else `None`.
    pub r_x: Option<ChallengeIdx>,
    /// Evaluation side: 0 (c=0 → left half) or 1 (c=1 → right half).
    pub c: u32,
}

impl LoweringCtx {
    pub fn suffix_len(&self) -> usize {
        self.total_bits - self.j - self.b_len - 1
    }
}

/// Lower one `PrefixMleRule` at the given context.
///
/// `masks(role)` must return a unique `PolynomialId` for each role used
/// (caller's responsibility to allocate & preprocess the buffer).
pub fn lower_prefix_mle<M>(rule: &PrefixMleRule, ctx: LoweringCtx, mut masks: M) -> ScalarExpr
where
    M: FnMut(MaskRole) -> PolynomialId,
{
    match &rule.formula {
        PrefixMleFormula::Multiplicative { pair, remaining } => {
            lower_multiplicative_inner(rule, *pair, *remaining, ctx, &mut masks)
        }
        PrefixMleFormula::OverflowBitsZero {
            pair,
            word_bits,
            total_bits: _,
        } => lower_overflow_bits_zero(rule, *pair, *word_bits, ctx, &mut masks),
        PrefixMleFormula::Lsb { total_bits } => lower_lsb(*total_bits, ctx, &mut masks),
        PrefixMleFormula::TwoLsb { total_bits } => lower_two_lsb(*total_bits, ctx, &mut masks),
        PrefixMleFormula::BitwiseAdditive {
            pair,
            weight,
            op,
            total_bits: _,
            word_bits,
            rotation,
        } => lower_bitwise_additive(
            rule, *pair, *weight, *op, *word_bits, *rotation, ctx, &mut masks,
        ),
        PrefixMleFormula::OperandExtract {
            base_shift,
            has_x,
            has_y,
            word_bits,
            total_bits: _,
            active_when,
            passthrough_when_inactive,
            is_upper,
        } => lower_operand_extract(
            rule,
            *base_shift,
            *has_x,
            *has_y,
            *word_bits,
            *active_when,
            *passthrough_when_inactive,
            *is_upper,
            ctx,
            &mut masks,
        ),
        PrefixMleFormula::Msb {
            msb_idx,
            start_round,
            is_left,
        } => lower_msb(*msb_idx, *start_round, *is_left, ctx, &mut masks),
        PrefixMleFormula::SignExtUpperHalf { word_bits } => {
            lower_sign_ext_upper_half(rule, *word_bits, ctx)
        }
        PrefixMleFormula::SignExtRightOp { word_bits } => {
            lower_sign_ext_right_op(rule, *word_bits, ctx, &mut masks)
        }
        PrefixMleFormula::RightOperandExtract {
            word_bits,
            start_round,
            total_bits: _,
            suffix_guard,
        } => lower_right_operand_extract(
            rule,
            *word_bits,
            *start_round,
            *suffix_guard,
            ctx,
            &mut masks,
        ),
        PrefixMleFormula::Pow2 {
            word_mask,
            log_word_bits,
            total_bits: _,
        } => lower_pow2(rule, *word_mask, *log_word_bits, ctx, &mut masks),
        PrefixMleFormula::Rev8W { xlen: _ } => lower_rev8w(rule, ctx, &mut masks),
        PrefixMleFormula::NegDivZeroRem {
            word_bits: _,
            start_round,
        } => lower_neg_div_zero_rem(rule, *start_round, ctx, &mut masks),
        PrefixMleFormula::DependentComparison {
            eq_idx,
            eq_default,
            cmp,
        } => lower_dependent_comparison(rule, *eq_idx, *eq_default, *cmp, ctx, &mut masks),
        PrefixMleFormula::LeftShiftHelper {
            word_bits: _,
            start_round,
        } => lower_left_shift_helper(rule, *start_round, ctx, &mut masks),
        PrefixMleFormula::RightShift {
            word_bits: _,
            start_round,
        } => lower_right_shift(rule, *start_round, ctx, &mut masks),
        PrefixMleFormula::LeftShift {
            helper_idx,
            helper_default,
            word_bits,
            start_round,
        } => lower_left_shift(
            rule,
            *helper_idx,
            *helper_default,
            *word_bits,
            *start_round,
            ctx,
            &mut masks,
        ),
        PrefixMleFormula::SignExtension {
            msb_idx,
            word_bits: _,
        } => lower_sign_extension(rule, *msb_idx, ctx, &mut masks),
        PrefixMleFormula::SignGatedMultiplicative {
            sign_pair,
            base_pair,
            remaining,
            start_round,
        } => lower_sign_gated_multiplicative(
            rule,
            *sign_pair,
            *base_pair,
            *remaining,
            *start_round,
            ctx,
            &mut masks,
        ),
        PrefixMleFormula::ChangeDivisor {
            word_bits: _,
            start_round,
        } => lower_change_divisor(rule, *start_round, ctx, &mut masks),
        PrefixMleFormula::ComplexComparison {
            eq_idx,
            cmp,
            sign_pair_j0,
            init_cmp,
            mul_anti,
            start_round,
        } => lower_complex_comparison(
            rule,
            *eq_idx,
            *cmp,
            *sign_pair_j0,
            *init_cmp,
            *mul_anti,
            *start_round,
            ctx,
            &mut masks,
        ),
    }
}

/// Collect all mask roles used by `lower_prefix_mle` for the given rule &
/// context, returning `(role, PolynomialId)` pairs in allocation order.
pub fn mask_roles_for(rule: &PrefixMleRule, ctx: LoweringCtx) -> Vec<(MaskRole, PolynomialId)> {
    let mut alloc = MaskAllocator::default();
    let _ = lower_prefix_mle(rule, ctx, |role| alloc.get_or_alloc(role));
    alloc.into_pairs()
}

// ── Per-formula lowerings ──────────────────────────────────────────────────

fn cp_factor(rule: &PrefixMleRule) -> ValueSource {
    ValueSource::Checkpoint {
        idx: rule.checkpoint_idx,
        default: rule.default,
    }
}

fn cp_expr(rule: &PrefixMleRule) -> ScalarExpr {
    vec![Monomial {
        coeff: 1,
        factors: vec![cp_factor(rule)],
    }]
}

/// Operands (x_f, y_f) for the pair-extraction convention used in
/// `Multiplicative`-style variants (Multiplicative, OverflowBitsZero, etc.):
/// - Odd round: x = r_x (challenge), y = c (constant).
/// - Even round: x = c (constant), y = pop_msb(b) (mask).
fn pair_operands<M>(ctx: LoweringCtx, masks: &mut M) -> (Operand, Operand)
where
    M: FnMut(MaskRole) -> PolynomialId,
{
    match ctx.r_x {
        Some(rx) => (
            Operand::Factor(ValueSource::Challenge(rx)),
            Operand::Const01(ctx.c),
        ),
        None => (
            Operand::Const01(ctx.c),
            Operand::Factor(ValueSource::IndexedPoly(masks(MaskRole::YMsb))),
        ),
    }
}

fn lower_multiplicative_inner<M>(
    rule: &PrefixMleRule,
    pair: BilinearExpr,
    remaining: RemainingTest,
    ctx: LoweringCtx,
    masks: &mut M,
) -> ScalarExpr
where
    M: FnMut(MaskRole) -> PolynomialId,
{
    let (x_op, y_op) = pair_operands(ctx, masks);
    let pair_expr = bilinear_expand(pair, &x_op, &y_op);
    if pair_expr.is_empty() {
        return vec![];
    }
    let with_remaining = if matches!(remaining, RemainingTest::Always) {
        pair_expr
    } else {
        let pop_msb = ctx.r_x.is_none();
        let rem_id = masks(MaskRole::RemainingIndicator {
            test: remaining,
            pop_msb,
        });
        prepend_factor(pair_expr, ValueSource::IndexedPoly(rem_id))
    };
    prepend_factor(with_remaining, cp_factor(rule))
}

/// `OverflowBitsZero` = Multiplicative with a "rest shifts to zero" indicator.
///
/// If `j >= 128 - word_bits`, returns `cp` (no pair factor, no indicator).
/// Else, returns `cp × pair(x,y) × I(rest_ok(b))`.
fn lower_overflow_bits_zero<M>(
    rule: &PrefixMleRule,
    pair: BilinearExpr,
    word_bits: usize,
    ctx: LoweringCtx,
    masks: &mut M,
) -> ScalarExpr
where
    M: FnMut(MaskRole) -> PolynomialId,
{
    if ctx.j >= 128 - word_bits {
        return cp_expr(rule);
    }
    let (x_op, y_op) = pair_operands(ctx, masks);
    let pair_expr = bilinear_expand(pair, &x_op, &y_op);
    if pair_expr.is_empty() {
        return vec![];
    }
    let rest_id = masks(MaskRole::OverflowRestOk {
        word_bits,
        suffix_len: ctx.suffix_len(),
        pop_msb: ctx.r_x.is_none(),
    });
    let with_rest = prepend_factor(pair_expr, ValueSource::IndexedPoly(rest_id));
    prepend_factor(with_rest, cp_factor(rule))
}

/// `Lsb { total_bits }`:
///   - `j == total_bits - 1`: c (constant)
///   - `suffix_len == 0`: b & 1 (mask)
///   - else: 1
fn lower_lsb<M>(total_bits: usize, ctx: LoweringCtx, masks: &mut M) -> ScalarExpr
where
    M: FnMut(MaskRole) -> PolynomialId,
{
    if ctx.j == total_bits.saturating_sub(1) {
        return match ctx.c {
            0 => vec![],
            1 => one_expr(),
            _ => unreachable!(),
        };
    }
    if ctx.suffix_len() == 0 {
        let bit0 = masks(MaskRole::BitExtract { bit: 0 });
        return vec![Monomial {
            coeff: 1,
            factors: vec![ValueSource::IndexedPoly(bit0)],
        }];
    }
    one_expr()
}

/// `TwoLsb { total_bits }`:
///   - `j == total_bits - 1`: (1-c)(1-r_x)   (odd round)
///   - `j == total_bits - 2`: (1-bit0)(1-c)  (even round)
///   - `suffix_len == 0`: I(trailing_zeros(b) >= 2)
///   - else: 1
fn lower_two_lsb<M>(total_bits: usize, ctx: LoweringCtx, masks: &mut M) -> ScalarExpr
where
    M: FnMut(MaskRole) -> PolynomialId,
{
    let tb = total_bits;
    if ctx.j + 1 == tb {
        // Odd round: r_x is Some.
        let rx = ctx.r_x.expect("TwoLsb at j=tb-1 requires r_x");
        // (1 - c) * (1 - r_x)
        let c_part: ScalarExpr = match ctx.c {
            0 => one_expr(),
            1 => vec![],
            _ => unreachable!(),
        };
        let omr: ScalarExpr = vec![Monomial {
            coeff: 1,
            factors: vec![ValueSource::OneMinusChallenge(rx)],
        }];
        return mul_exprs(&c_part, &omr);
    }
    if ctx.j + 2 == tb {
        // Even round: bit0 mask × (1 - c).
        let one_minus_bit0: ScalarExpr = vec![
            Monomial {
                coeff: 1,
                factors: vec![],
            },
            Monomial {
                coeff: -1,
                factors: vec![ValueSource::IndexedPoly(masks(MaskRole::BitExtract {
                    bit: 0,
                }))],
            },
        ];
        let c_part: ScalarExpr = match ctx.c {
            0 => one_expr(),
            1 => vec![],
            _ => unreachable!(),
        };
        return mul_exprs(&one_minus_bit0, &c_part);
    }
    if ctx.suffix_len() == 0 {
        let mask = masks(MaskRole::TrailingZerosGe { k: 2 });
        return vec![Monomial {
            coeff: 1,
            factors: vec![ValueSource::IndexedPoly(mask)],
        }];
    }
    one_expr()
}

/// `BitwiseAdditive`: `cp + weight(j) × pair(x,y) + shifted_bitop(b)`.
///
/// When `weight` is `Positional` with `j_offset > j`, returns `0` (prefix
/// inactive). Otherwise returns the three-term sum above, with the bit-op
/// term materialized as a mask buffer.
#[allow(clippy::too_many_arguments)]
fn lower_bitwise_additive<M>(
    rule: &PrefixMleRule,
    pair: BilinearExpr,
    weight: WeightFn,
    op: IntBitOp,
    word_bits: usize,
    rotation: usize,
    ctx: LoweringCtx,
    masks: &mut M,
) -> ScalarExpr
where
    M: FnMut(MaskRole) -> PolynomialId,
{
    if let WeightFn::Positional { j_offset, .. } = weight {
        if ctx.j < j_offset {
            return vec![];
        }
    }
    let (x_op, y_op) = pair_operands(ctx, masks);
    let w = weight_eval(weight, ctx.j);

    // cp term
    let mut result: ScalarExpr = cp_expr(rule);

    // + weight × pair(x, y)
    let pair_expr = bilinear_expand(pair, &x_op, &y_op);
    for mut m in pair_expr {
        m.coeff = m.coeff.wrapping_mul(w);
        if m.coeff != 0 {
            result.push(m);
        }
    }

    // + shifted bitop mask
    let bitop_id = masks(MaskRole::BitOpShifted {
        op,
        rotation: rotation as u32,
        suffix_len: ctx.suffix_len(),
        word_bits,
        pop_msb: ctx.r_x.is_none(),
    });
    result.push(Monomial {
        coeff: 1,
        factors: vec![ValueSource::IndexedPoly(bitop_id)],
    });

    result
}

/// `Msb`: returns 1 before `start_round`, pair bit at `jr=0,1`, checkpoint otherwise.
fn lower_msb<M>(
    msb_idx: usize,
    start_round: usize,
    is_left: bool,
    ctx: LoweringCtx,
    masks: &mut M,
) -> ScalarExpr
where
    M: FnMut(MaskRole) -> PolynomialId,
{
    if ctx.j < start_round {
        return one_expr();
    }
    let jr = ctx.j - start_round;
    if jr == 0 {
        return if is_left {
            match ctx.c {
                0 => vec![],
                1 => one_expr(),
                _ => unreachable!(),
            }
        } else {
            vec![Monomial {
                coeff: 1,
                factors: vec![ValueSource::IndexedPoly(masks(MaskRole::YMsb))],
            }]
        };
    }
    if jr == 1 {
        return if is_left {
            let rx = ctx.r_x.expect("Msb jr=1 with is_left=true requires r_x");
            vec![Monomial {
                coeff: 1,
                factors: vec![ValueSource::Challenge(rx)],
            }]
        } else {
            match ctx.c {
                0 => vec![],
                1 => one_expr(),
                _ => unreachable!(),
            }
        };
    }
    // jr >= 2: return checkpoint value at msb_idx.
    vec![Monomial {
        coeff: 1,
        factors: vec![ValueSource::Checkpoint {
            idx: msb_idx,
            default: crate::module::DefaultVal::Zero,
        }],
    }]
}

/// Build a ScalarExpr evaluating to `((1u128 << hi) - (1u128 << lo))` — the
/// sum `sum_{k=lo..hi} 2^k`. Returns `vec![]` if `lo >= hi`.
fn mask_range_expr(lo: usize, hi: usize) -> ScalarExpr {
    (lo..hi)
        .map(|k| Monomial {
            coeff: 1,
            factors: vec![ValueSource::Pow2(k as u32)],
        })
        .collect()
}

/// Multiply every monomial's factors by `factor` (prepend).
fn scale_by_factor(mut e: ScalarExpr, factor: ValueSource) -> ScalarExpr {
    for m in &mut e {
        m.factors.insert(0, factor.clone());
    }
    e
}

/// `SignExtUpperHalf { word_bits }`.
fn lower_sign_ext_upper_half(
    rule: &PrefixMleRule,
    word_bits: usize,
    ctx: LoweringCtx,
) -> ScalarExpr {
    let half = word_bits / 2;
    if ctx.suffix_len() >= half {
        return one_expr();
    }
    let mask_expr = mask_range_expr(half, word_bits);
    if ctx.j == word_bits + half {
        return match ctx.c {
            0 => vec![],
            1 => mask_expr,
            _ => unreachable!(),
        };
    }
    if ctx.j == word_bits + half + 1 {
        let rx = ctx
            .r_x
            .expect("SignExtUpperHalf at j=word_bits+half+1 requires r_x");
        return scale_by_factor(mask_expr, ValueSource::Challenge(rx));
    }
    vec![Monomial {
        coeff: 1,
        factors: vec![ValueSource::Checkpoint {
            idx: rule.checkpoint_idx,
            default: crate::module::DefaultVal::Zero,
        }],
    }]
}

/// `Pow2`: mask buffer `2^(b & word_mask)` with conditional c/rx/cp multipliers.
fn lower_pow2<M>(
    rule: &PrefixMleRule,
    word_mask: usize,
    log_word_bits: u32,
    ctx: LoweringCtx,
    masks: &mut M,
) -> ScalarExpr
where
    M: FnMut(MaskRole) -> PolynomialId,
{
    if ctx.suffix_len() != 0 {
        return one_expr();
    }
    let b_len = ctx.b_len;
    let log_wb = log_word_bits as usize;

    let pow2_mask = masks(MaskRole::Pow2Masked { word_mask });
    let pow2_expr: ScalarExpr = vec![Monomial {
        coeff: 1,
        factors: vec![ValueSource::IndexedPoly(pow2_mask)],
    }];

    if b_len >= log_wb {
        return pow2_expr;
    }

    // (1 + K_c × c) — compile-time coefficient folded into monomials.
    let k_c: u128 = 1u128 << (1u32 << b_len);
    let c_factor: u128 = 1u128 + (k_c - 1) * (ctx.c as u128);
    let mut base = pow2_expr;
    for m in &mut base {
        m.coeff = m.coeff.wrapping_mul(c_factor as i128);
    }

    if b_len == log_wb - 1 {
        return base;
    }

    // × (1 + K_rx × rx) if r_x is Some.
    let num_bits = b_len + 1;
    // Match interpretive's wrapping semantics: 1u64.wrapping_shl(1u32.wrapping_shl(num_bits)).
    let shift2 = 1u64.wrapping_shl(1u32.wrapping_shl(num_bits as u32));
    let k_rx = shift2.wrapping_sub(1) as i128;

    let mut with_rx = base.clone();
    if let Some(rx) = ctx.r_x {
        let mut rx_term = base;
        for m in &mut rx_term {
            m.coeff = m.coeff.wrapping_mul(k_rx);
            m.factors.push(ValueSource::Challenge(rx));
        }
        with_rx.extend(rx_term);
    }

    // × cp
    scale_by_factor(with_rx, cp_factor(rule))
}

/// Build `default_to_expr(default)` as a `ScalarExpr`.
fn default_expr(default: DefaultVal) -> ScalarExpr {
    match default {
        DefaultVal::Zero => vec![],
        DefaultVal::One => one_expr(),
        DefaultVal::Custom(v) => vec![Monomial {
            coeff: v,
            factors: vec![],
        }],
    }
}

/// `ComplexComparison`: 4-phase formula gated by `j >= start_round`.
///
/// jr=0 (even): `I(init_cmp, pop=true) × sign_pair_j0(c, YMsb)`
/// jr=1 (odd):  `I(init_cmp, pop=false) × sign_pair_j0(r_x, c)`
/// jr∈{2,3}:    `cp × mul_anti(x,y) + I(cmp) × eq × EqBit(x,y)`
/// jr≥4:        `cp + eq × anti_pair(cmp, x, y) + I(cmp) × eq × EqBit(x, y)`
#[allow(clippy::too_many_arguments)]
fn lower_complex_comparison<M>(
    rule: &PrefixMleRule,
    eq_idx: usize,
    cmp: Comparison,
    sign_pair_j0: BilinearExpr,
    init_cmp: Comparison,
    mul_anti: BilinearExpr,
    start_round: usize,
    ctx: LoweringCtx,
    masks: &mut M,
) -> ScalarExpr
where
    M: FnMut(MaskRole) -> PolynomialId,
{
    if ctx.j < start_round {
        return vec![];
    }
    let jr = ctx.j - start_round;

    if jr == 0 {
        let x = Operand::Const01(ctx.c);
        let y = Operand::Factor(ValueSource::IndexedPoly(masks(MaskRole::YMsb)));
        let pair_expr = bilinear_expand(sign_pair_j0, &x, &y);
        if pair_expr.is_empty() {
            return vec![];
        }
        let init_mask = masks(MaskRole::CmpIndicator {
            cmp: init_cmp,
            pop_msb: true,
        });
        return prepend_factor(pair_expr, ValueSource::IndexedPoly(init_mask));
    }

    if jr == 1 {
        let rx = ctx.r_x.expect("ComplexComparison jr=1 requires r_x");
        let x = Operand::Factor(ValueSource::Challenge(rx));
        let y = Operand::Const01(ctx.c);
        let pair_expr = bilinear_expand(sign_pair_j0, &x, &y);
        if pair_expr.is_empty() {
            return vec![];
        }
        let init_mask = masks(MaskRole::CmpIndicator {
            cmp: init_cmp,
            pop_msb: false,
        });
        return prepend_factor(pair_expr, ValueSource::IndexedPoly(init_mask));
    }

    // jr >= 2: normal multiplicative + equality indicator.
    let (x_op, y_op) = pair_operands(ctx, masks);
    let eq_expand = bilinear_expand(BilinearExpr::EqBit, &x_op, &y_op);
    // eq_idx checkpoint (hardcoded `.unwrap()` → default Zero on lowering side).
    let eq_cp = ValueSource::Checkpoint {
        idx: eq_idx,
        default: DefaultVal::Zero,
    };
    let pop_msb = ctx.r_x.is_none();
    let cmp_mask = masks(MaskRole::CmpIndicator { cmp, pop_msb });

    let mut result: ScalarExpr;
    if jr == 2 || jr == 3 {
        // cp × mul_anti(x, y)
        let mul = bilinear_expand(mul_anti, &x_op, &y_op);
        result = prepend_factor(mul, cp_factor(rule));
    } else {
        // cp + eq × anti_pair(cmp, x, y)
        let anti = bilinear_expand(anti_pair_for_cmp(cmp), &x_op, &y_op);
        result = cp_expr(rule);
        result.extend(prepend_factor(anti, eq_cp.clone()));
    }
    // + I(cmp_test) × eq × EqBit(x, y)
    let with_eq = prepend_factor(eq_expand, eq_cp);
    result.extend(prepend_factor(with_eq, ValueSource::IndexedPoly(cmp_mask)));
    result
}

/// `ChangeDivisor`: 4-case formula gated by `j >= start_round`.
///
/// jr=0 (even, r_x=None):  `cp × YMsb × I(LZRAO, pop=true) × c`
/// jr=1 (odd):             `cp × rx × c × I(LZRAO, pop=false)`
/// jr≥2 odd:               `cp × (1-rx) × c × I(LZRAO, pop=false)`
/// jr≥2 even (b_len>0):    `cp × (1-c) × I(LZRAO, pop=false)`
///
/// Where `LZRAO = LeftZeroRightAllOnes`.
fn lower_change_divisor<M>(
    rule: &PrefixMleRule,
    start_round: usize,
    ctx: LoweringCtx,
    masks: &mut M,
) -> ScalarExpr
where
    M: FnMut(MaskRole) -> PolynomialId,
{
    if ctx.j < start_round {
        return vec![];
    }
    let jr = ctx.j - start_round;

    if jr == 0 {
        // Even, pop_msb=true. Output = cp × YMsb × I(LZRAO pop=true) × c.
        if ctx.c == 0 {
            return vec![];
        }
        let y_msb = masks(MaskRole::YMsb);
        let rem = masks(MaskRole::RemainingIndicator {
            test: RemainingTest::LeftZeroRightAllOnes,
            pop_msb: true,
        });
        return vec![Monomial {
            coeff: 1,
            factors: vec![
                cp_factor(rule),
                ValueSource::IndexedPoly(y_msb),
                ValueSource::IndexedPoly(rem),
            ],
        }];
    }

    if jr == 1 {
        if ctx.c == 0 {
            return vec![];
        }
        let rx = ctx.r_x.expect("ChangeDivisor jr=1 requires r_x");
        let rem = masks(MaskRole::RemainingIndicator {
            test: RemainingTest::LeftZeroRightAllOnes,
            pop_msb: false,
        });
        return vec![Monomial {
            coeff: 1,
            factors: vec![
                cp_factor(rule),
                ValueSource::Challenge(rx),
                ValueSource::IndexedPoly(rem),
            ],
        }];
    }

    if let Some(rx) = ctx.r_x {
        // cp × (1 - rx) × c × I(LZRAO, pop=false).
        if ctx.c == 0 {
            return vec![];
        }
        let rem = masks(MaskRole::RemainingIndicator {
            test: RemainingTest::LeftZeroRightAllOnes,
            pop_msb: false,
        });
        vec![Monomial {
            coeff: 1,
            factors: vec![
                cp_factor(rule),
                ValueSource::OneMinusChallenge(rx),
                ValueSource::IndexedPoly(rem),
            ],
        }]
    } else {
        // cp × (1 - c) × [I(LZRAO, pop=false) if b_len>0 else 1].
        if ctx.c == 1 {
            return vec![];
        }
        if ctx.b_len > 0 {
            let rem = masks(MaskRole::RemainingIndicator {
                test: RemainingTest::LeftZeroRightAllOnes,
                pop_msb: false,
            });
            return vec![Monomial {
                coeff: 1,
                factors: vec![cp_factor(rule), ValueSource::IndexedPoly(rem)],
            }];
        }
        vec![Monomial {
            coeff: 1,
            factors: vec![cp_factor(rule)],
        }]
    }
}

/// `SignGatedMultiplicative`: sign-gate at jr=0,1 (pair of sign bits), then
/// normal multiplicative with equality indicator for jr>=2.
///
/// jr=0 (even, r_x=None): `sign_pair(c, YMsb) × I_y0(remaining, pop_msb=true)`
/// jr=1 (odd,  r_x=Some): `sign_pair(r_x, c) × I_y0(remaining, pop_msb=false)`
/// jr≥2: `cp × base_pair(x, y) × I(remaining, standard y_len)`
#[allow(clippy::too_many_arguments)]
fn lower_sign_gated_multiplicative<M>(
    rule: &PrefixMleRule,
    sign_pair: BilinearExpr,
    base_pair: BilinearExpr,
    remaining: RemainingTest,
    start_round: usize,
    ctx: LoweringCtx,
    masks: &mut M,
) -> ScalarExpr
where
    M: FnMut(MaskRole) -> PolynomialId,
{
    if ctx.j < start_round {
        return vec![];
    }
    let jr = ctx.j - start_round;

    if jr == 0 {
        let x = Operand::Const01(ctx.c);
        let y = Operand::Factor(ValueSource::IndexedPoly(masks(MaskRole::YMsb)));
        let pair_expr = bilinear_expand(sign_pair, &x, &y);
        if pair_expr.is_empty() {
            return vec![];
        }
        if matches!(remaining, RemainingTest::Always) {
            return pair_expr;
        }
        let rem = masks(MaskRole::RemainingIndicatorYZero {
            test: remaining,
            pop_msb: true,
        });
        return prepend_factor(pair_expr, ValueSource::IndexedPoly(rem));
    }
    if jr == 1 {
        let rx = ctx.r_x.expect("SignGatedMultiplicative jr=1 requires r_x");
        let x = Operand::Factor(ValueSource::Challenge(rx));
        let y = Operand::Const01(ctx.c);
        let pair_expr = bilinear_expand(sign_pair, &x, &y);
        if pair_expr.is_empty() {
            return vec![];
        }
        if matches!(remaining, RemainingTest::Always) {
            return pair_expr;
        }
        let rem = masks(MaskRole::RemainingIndicatorYZero {
            test: remaining,
            pop_msb: false,
        });
        return prepend_factor(pair_expr, ValueSource::IndexedPoly(rem));
    }

    // jr >= 2: standard multiplicative.
    let (x_op, y_op) = pair_operands(ctx, masks);
    let pair_expr = bilinear_expand(base_pair, &x_op, &y_op);
    if pair_expr.is_empty() {
        return vec![];
    }
    let with_remaining = if matches!(remaining, RemainingTest::Always) {
        pair_expr
    } else {
        let rem = masks(MaskRole::RemainingIndicator {
            test: remaining,
            pop_msb: ctx.r_x.is_none(),
        });
        prepend_factor(pair_expr, ValueSource::IndexedPoly(rem))
    };
    prepend_factor(with_remaining, cp_factor(rule))
}

/// `SignExtension`: sign-extends the upper bits per the sign-bit checkpoint.
///
/// j=0 (even): `YComplementSum(0) × c` (sign_bit = c; empty if c=0).
/// j=1 (odd):  `YComplementSum(0) × rx`.
/// j≥2:        `[cp + delta] × sign_bit + YComplementSum(j/2) × sign_bit`
///             where sign_bit = Checkpoint[msb_idx], delta depends on round parity.
fn lower_sign_extension<M>(
    rule: &PrefixMleRule,
    msb_idx: usize,
    ctx: LoweringCtx,
    masks: &mut M,
) -> ScalarExpr
where
    M: FnMut(MaskRole) -> PolynomialId,
{
    if ctx.j == 0 {
        // Even round, pop_msb=true.
        if ctx.c == 0 {
            return vec![];
        }
        let comp = masks(MaskRole::YComplementSum {
            base_index: 0,
            pop_msb: true,
        });
        return vec![Monomial {
            coeff: 1,
            factors: vec![ValueSource::IndexedPoly(comp)],
        }];
    }
    if ctx.j == 1 {
        let rx = ctx.r_x.expect("SignExtension j=1 requires r_x");
        let comp = masks(MaskRole::YComplementSum {
            base_index: 0,
            pop_msb: false,
        });
        return vec![Monomial {
            coeff: 1,
            factors: vec![ValueSource::IndexedPoly(comp), ValueSource::Challenge(rx)],
        }];
    }
    // j >= 2
    let sign_bit = ValueSource::Checkpoint {
        idx: msb_idx,
        default: DefaultVal::Zero,
    };
    let pow_const = 1i128 << (ctx.j / 2);
    let pop_msb = ctx.r_x.is_none();
    let comp = masks(MaskRole::YComplementSum {
        base_index: ctx.j / 2,
        pop_msb,
    });

    let mut result: ScalarExpr = vec![Monomial {
        coeff: 1,
        factors: vec![cp_factor(rule), sign_bit.clone()],
    }];
    if ctx.r_x.is_some() {
        // + (1 - c) × 2^(j/2) × sign_bit
        if ctx.c == 0 {
            result.push(Monomial {
                coeff: pow_const,
                factors: vec![sign_bit.clone()],
            });
        }
    } else {
        // + 2^(j/2) × sign_bit - YMsb × 2^(j/2) × sign_bit
        let y_msb = ValueSource::IndexedPoly(masks(MaskRole::YMsb));
        result.push(Monomial {
            coeff: pow_const,
            factors: vec![sign_bit.clone()],
        });
        result.push(Monomial {
            coeff: -pow_const,
            factors: vec![y_msb, sign_bit.clone()],
        });
    }
    // + YComplementSum × sign_bit
    result.push(Monomial {
        coeff: 1,
        factors: vec![ValueSource::IndexedPoly(comp), sign_bit],
    });
    result
}

/// `LeftShift`: before start: default. After:
///   odd:  `cp + rx × (1-c) × cp_helper × 2^(wb-1-j/2) + XMaskedShifted × cp_helper × (1+c)`
///   even: `cp + c × (1-YMsb) × cp_helper × 2^(wb-1-j/2) + XMaskedShifted × cp_helper × (1+YMsb)`
#[allow(clippy::too_many_arguments)]
fn lower_left_shift<M>(
    rule: &PrefixMleRule,
    helper_idx: usize,
    helper_default: DefaultVal,
    word_bits: usize,
    start_round: usize,
    ctx: LoweringCtx,
    masks: &mut M,
) -> ScalarExpr
where
    M: FnMut(MaskRole) -> PolynomialId,
{
    if ctx.j < start_round {
        return default_expr(rule.default);
    }
    let shift_const = 1i128 << (word_bits - 1 - ctx.j / 2);
    let prod = ValueSource::Checkpoint {
        idx: helper_idx,
        default: helper_default,
    };
    let pop_msb = ctx.r_x.is_none();
    let x_masked_mask = ValueSource::IndexedPoly(masks(MaskRole::XMaskedShifted {
        word_bits,
        j: ctx.j,
        pop_msb,
    }));

    let mut result: ScalarExpr = cp_expr(rule);
    if let Some(rx) = ctx.r_x {
        // rx × (1 - c) × prod × shift_const
        if ctx.c == 0 {
            result.push(Monomial {
                coeff: shift_const,
                factors: vec![ValueSource::Challenge(rx), prod.clone()],
            });
        }
        // XMaskedShifted × prod × (1 + c)
        result.push(Monomial {
            coeff: 1 + ctx.c as i128,
            factors: vec![x_masked_mask, prod],
        });
    } else {
        let y_msb = ValueSource::IndexedPoly(masks(MaskRole::YMsb));
        // c × (1 - YMsb) × prod × shift_const  (only if c=1)
        if ctx.c == 1 {
            result.push(Monomial {
                coeff: shift_const,
                factors: vec![prod.clone()],
            });
            result.push(Monomial {
                coeff: -shift_const,
                factors: vec![y_msb.clone(), prod.clone()],
            });
        }
        // XMaskedShifted × prod × (1 + YMsb) = XM × prod + XM × YMsb × prod
        result.push(Monomial {
            coeff: 1,
            factors: vec![x_masked_mask.clone(), prod.clone()],
        });
        result.push(Monomial {
            coeff: 1,
            factors: vec![x_masked_mask, y_msb, prod],
        });
    }
    result
}

/// `RightShift`: before start: default. After:
///   odd:  `[cp × (1+c) + c × rx] × 2^LO + XRemShiftedByTZ`
///   even: `[cp × (1+YMsb) + c × YMsb] × 2^LO + XRemShiftedByTZ`
fn lower_right_shift<M>(
    rule: &PrefixMleRule,
    start_round: usize,
    ctx: LoweringCtx,
    masks: &mut M,
) -> ScalarExpr
where
    M: FnMut(MaskRole) -> PolynomialId,
{
    if ctx.j < start_round {
        return default_expr(rule.default);
    }
    let pop_msb = ctx.r_x.is_none();
    let lo_mask = masks(MaskRole::LeadingOnesPow2 { pop_msb });
    let lo_factor = ValueSource::IndexedPoly(lo_mask);

    let mut with_lo: ScalarExpr = if let Some(rx) = ctx.r_x {
        // cp × (1 + c) × 2^LO  + rx × c × 2^LO
        let mut terms: ScalarExpr = vec![Monomial {
            coeff: 1 + ctx.c as i128,
            factors: vec![cp_factor(rule), lo_factor.clone()],
        }];
        if ctx.c == 1 {
            terms.push(Monomial {
                coeff: 1,
                factors: vec![ValueSource::Challenge(rx), lo_factor.clone()],
            });
        }
        terms
    } else {
        let y_msb = ValueSource::IndexedPoly(masks(MaskRole::YMsb));
        // cp × 2^LO + cp × YMsb × 2^LO + c × YMsb × 2^LO
        let mut terms: ScalarExpr = vec![
            Monomial {
                coeff: 1,
                factors: vec![cp_factor(rule), lo_factor.clone()],
            },
            Monomial {
                coeff: 1,
                factors: vec![cp_factor(rule), y_msb.clone(), lo_factor.clone()],
            },
        ];
        if ctx.c == 1 {
            terms.push(Monomial {
                coeff: 1,
                factors: vec![y_msb, lo_factor.clone()],
            });
        }
        terms
    };
    let xrem_mask = masks(MaskRole::XRemShiftedByTZ { pop_msb });
    with_lo.push(Monomial {
        coeff: 1,
        factors: vec![ValueSource::IndexedPoly(xrem_mask)],
    });
    with_lo
}

/// `LeftShiftHelper`: before `start_round` returns default; otherwise
/// `cp × (1 + y) × 2^leading_ones(y_rem)` where `y` is `c` (odd) or `YMsb` (even).
fn lower_left_shift_helper<M>(
    rule: &PrefixMleRule,
    start_round: usize,
    ctx: LoweringCtx,
    masks: &mut M,
) -> ScalarExpr
where
    M: FnMut(MaskRole) -> PolynomialId,
{
    if ctx.j < start_round {
        return default_expr(rule.default);
    }
    let pow2_mask = masks(MaskRole::LeadingOnesPow2 {
        pop_msb: ctx.r_x.is_none(),
    });
    let mask_factor = ValueSource::IndexedPoly(pow2_mask);

    let multiplier: ScalarExpr = match ctx.r_x {
        // Odd: (1 + c) is a compile-time integer in {1, 2}.
        Some(_) => vec![Monomial {
            coeff: 1 + ctx.c as i128,
            factors: vec![],
        }],
        // Even: 1 + YMsb — polynomial over a mask.
        None => vec![
            Monomial {
                coeff: 1,
                factors: vec![],
            },
            Monomial {
                coeff: 1,
                factors: vec![ValueSource::IndexedPoly(masks(MaskRole::YMsb))],
            },
        ],
    };
    // cp × multiplier × pow2
    let scaled = scale_by_factor(multiplier, cp_factor(rule));
    prepend_factor(scaled, mask_factor)
}

/// `DependentComparison`: `cp + eq × anti_pair(x,y) + I(cmp_test) × eq × EqBit(x,y)`.
///
/// - `cp` is `checkpoints[rule.checkpoint_idx]` (lt-running total).
/// - `eq` is `checkpoints[eq_idx]` (equality-running total).
/// - `anti_pair(x,y)` is `AntiXY` for LessThan, `AntiYX` for GreaterThan.
/// - `(x, y)` from `pair_operands` (pop_msb on even rounds).
/// - `I(cmp_test)` is `I(rem_x < rem_y)` or `I(rem_x > rem_y)` per `cmp`.
fn lower_dependent_comparison<M>(
    rule: &PrefixMleRule,
    eq_idx: usize,
    eq_default: DefaultVal,
    cmp: Comparison,
    ctx: LoweringCtx,
    masks: &mut M,
) -> ScalarExpr
where
    M: FnMut(MaskRole) -> PolynomialId,
{
    let (x_op, y_op) = pair_operands(ctx, masks);
    let anti = bilinear_expand(anti_pair_for_cmp(cmp), &x_op, &y_op);
    let eq_expand = bilinear_expand(BilinearExpr::EqBit, &x_op, &y_op);
    let eq_cp = ValueSource::Checkpoint {
        idx: eq_idx,
        default: eq_default,
    };
    let pop_msb = ctx.r_x.is_none();
    let cmp_mask = masks(MaskRole::CmpIndicator { cmp, pop_msb });

    let mut result: ScalarExpr = cp_expr(rule);
    // + eq × anti(x, y)
    result.extend(prepend_factor(anti, eq_cp.clone()));
    // + I(cmp_test) × eq × EqBit(x, y)
    let with_eq = prepend_factor(eq_expand, eq_cp);
    result.extend(prepend_factor(with_eq, ValueSource::IndexedPoly(cmp_mask)));
    result
}

/// `NegDivZeroRem`: predicate that divisor was negative with left-half all-zero remainder.
///
/// Active only when `j >= start_round`. Four cases based on `jr = j - start_round`:
///   jr=0 (even, r_x=None): `(1-c) × YMsb(b) × I(rem_x(pop_msb(b)) == 0)`
///   jr=1 (odd, r_x=Some):  `c × (1 - r_x) × I(rem_x(b) == 0)`
///   jr≥2 odd:              `cp × (1 - r_x) × I(rem_x(b) == 0)`
///   jr≥2 even:             `cp × (1 - c) × I(rem_x(pop_msb(b)) == 0)`
fn lower_neg_div_zero_rem<M>(
    rule: &PrefixMleRule,
    start_round: usize,
    ctx: LoweringCtx,
    masks: &mut M,
) -> ScalarExpr
where
    M: FnMut(MaskRole) -> PolynomialId,
{
    if ctx.j < start_round {
        return vec![];
    }
    let jr = ctx.j - start_round;
    if jr == 0 {
        if ctx.c == 1 {
            return vec![];
        }
        let y_msb = masks(MaskRole::YMsb);
        let rem = masks(MaskRole::RemainingIndicator {
            test: RemainingTest::LeftZero,
            pop_msb: true,
        });
        return vec![Monomial {
            coeff: 1,
            factors: vec![
                ValueSource::IndexedPoly(rem),
                ValueSource::IndexedPoly(y_msb),
            ],
        }];
    }
    if jr == 1 {
        if ctx.c == 0 {
            return vec![];
        }
        let rx = ctx.r_x.expect("NegDivZeroRem jr=1 requires r_x");
        let rem = masks(MaskRole::RemainingIndicator {
            test: RemainingTest::LeftZero,
            pop_msb: false,
        });
        return vec![Monomial {
            coeff: 1,
            factors: vec![
                ValueSource::IndexedPoly(rem),
                ValueSource::OneMinusChallenge(rx),
            ],
        }];
    }
    if let Some(rx) = ctx.r_x {
        let rem = masks(MaskRole::RemainingIndicator {
            test: RemainingTest::LeftZero,
            pop_msb: false,
        });
        vec![Monomial {
            coeff: 1,
            factors: vec![
                cp_factor(rule),
                ValueSource::OneMinusChallenge(rx),
                ValueSource::IndexedPoly(rem),
            ],
        }]
    } else {
        if ctx.c == 1 {
            return vec![];
        }
        let rem = masks(MaskRole::RemainingIndicator {
            test: RemainingTest::LeftZero,
            pop_msb: true,
        });
        vec![Monomial {
            coeff: 1,
            factors: vec![cp_factor(rule), ValueSource::IndexedPoly(rem)],
        }]
    }
}

/// `Rev8W`: `cp + (c × 2^shift) + rx × rev8w(1<<rx_bit) + rev8w(b << suffix_n)`.
///
/// `shift = trailing_zeros(rev8w(1 << c_bit_index))`, all compile-time constants.
fn lower_rev8w<M>(rule: &PrefixMleRule, ctx: LoweringCtx, masks: &mut M) -> ScalarExpr
where
    M: FnMut(MaskRole) -> PolynomialId,
{
    let suffix_n = ctx.suffix_len();
    if suffix_n >= 64 {
        return vec![];
    }
    let mut result: ScalarExpr = cp_expr(rule);
    let c_bit_index = suffix_n + ctx.b_len;
    if c_bit_index < 64 && ctx.c == 1 {
        let shift = crate::module::rev8w(1u64 << c_bit_index).trailing_zeros();
        result.push(Monomial {
            coeff: 1i128 << shift,
            factors: vec![],
        });
    }
    let rx_bit_index = c_bit_index + 1;
    if rx_bit_index < 64 {
        if let Some(rx) = ctx.r_x {
            let rev_pow2 = crate::module::rev8w(1u64 << rx_bit_index);
            result.push(Monomial {
                coeff: rev_pow2 as i128,
                factors: vec![ValueSource::Challenge(rx)],
            });
        }
    }
    let mask_id = masks(MaskRole::Rev8wShifted {
        suffix_len: suffix_n,
    });
    result.push(Monomial {
        coeff: 1,
        factors: vec![ValueSource::IndexedPoly(mask_id)],
    });
    result
}

/// `RightOperandExtract`: cp + (c-weighted shift on odd rounds > start) + y_masked<<(suffix_len/2).
fn lower_right_operand_extract<M>(
    rule: &PrefixMleRule,
    word_bits: usize,
    start_round: usize,
    suffix_guard: usize,
    ctx: LoweringCtx,
    masks: &mut M,
) -> ScalarExpr
where
    M: FnMut(MaskRole) -> PolynomialId,
{
    let mut result: ScalarExpr = cp_expr(rule);
    if ctx.r_x.is_some() && ctx.j % 2 == 1 && ctx.j > start_round {
        let shift = word_bits - 1 - ctx.j / 2;
        if ctx.c == 1 {
            result.push(Monomial {
                coeff: 1,
                factors: vec![ValueSource::Pow2(shift as u32)],
            });
        }
    }
    if ctx.suffix_len() < suffix_guard {
        let mask = masks(MaskRole::YMaskedShifted {
            suffix_len: ctx.suffix_len(),
            pop_msb: false,
        });
        result.push(Monomial {
            coeff: 1,
            factors: vec![ValueSource::IndexedPoly(mask)],
        });
    }
    result
}

/// `SignExtRightOp { word_bits }`.
fn lower_sign_ext_right_op<M>(
    rule: &PrefixMleRule,
    word_bits: usize,
    ctx: LoweringCtx,
    masks: &mut M,
) -> ScalarExpr
where
    M: FnMut(MaskRole) -> PolynomialId,
{
    let half = word_bits / 2;
    if ctx.suffix_len() >= word_bits {
        return one_expr();
    }
    let mask_expr = mask_range_expr(half, word_bits);
    if ctx.j == word_bits {
        // Even round: use pop_msb(b) as sign bit.
        let y_msb = masks(MaskRole::YMsb);
        return scale_by_factor(mask_expr, ValueSource::IndexedPoly(y_msb));
    }
    if ctx.j == word_bits + 1 {
        // Odd round (r_x is the bound sign bit already handled by
        // the round before; interpretive code uses c here).
        return match ctx.c {
            0 => vec![],
            1 => mask_expr,
            _ => unreachable!(),
        };
    }
    vec![Monomial {
        coeff: 1,
        factors: vec![ValueSource::Checkpoint {
            idx: rule.checkpoint_idx,
            default: crate::module::DefaultVal::Zero,
        }],
    }]
}

fn guard_matches(g: RoundGuard, j: usize, suffix_len: usize) -> bool {
    match g {
        RoundGuard::JEq(v) => j == v,
        RoundGuard::JLt(v) => j < v,
        RoundGuard::JGe(v) => j >= v,
        RoundGuard::JGt(v) => j > v,
        RoundGuard::SuffixLenNonZero => suffix_len != 0,
    }
}

/// `OperandExtract`: extracts the operand at a specific position with
/// an optional round guard that may passthrough cp when inactive.
#[allow(clippy::too_many_arguments)]
fn lower_operand_extract<M>(
    rule: &PrefixMleRule,
    base_shift: usize,
    has_x: bool,
    has_y: bool,
    word_bits: usize,
    active_when: Option<RoundGuard>,
    passthrough_when_inactive: bool,
    is_upper: bool,
    ctx: LoweringCtx,
    masks: &mut M,
) -> ScalarExpr
where
    M: FnMut(MaskRole) -> PolynomialId,
{
    if let Some(guard) = active_when {
        if !guard_matches(guard, ctx.j, ctx.suffix_len()) {
            return if passthrough_when_inactive {
                cp_expr(rule)
            } else {
                vec![]
            };
        }
    }
    let mut result: ScalarExpr = cp_expr(rule);

    if let Some(rx) = ctx.r_x {
        if has_x {
            result.push(Monomial {
                coeff: 1i128 << (base_shift - ctx.j),
                factors: vec![ValueSource::Challenge(rx)],
            });
        }
        if has_y {
            let coeff = 1i128 << (base_shift - ctx.j - 1);
            let c_val = ctx.c as i128;
            if c_val != 0 {
                result.push(Monomial {
                    coeff: coeff.wrapping_mul(c_val),
                    factors: vec![],
                });
            }
        }
    } else {
        if has_x {
            let coeff = 1i128 << (base_shift - ctx.j - 1);
            let c_val = ctx.c as i128;
            if c_val != 0 {
                result.push(Monomial {
                    coeff: coeff.wrapping_mul(c_val),
                    factors: vec![],
                });
            }
        }
        if has_y {
            let coeff = 1i128 << (base_shift - ctx.j - 2);
            let y_msb = masks(MaskRole::YMsb);
            result.push(Monomial {
                coeff,
                factors: vec![ValueSource::IndexedPoly(y_msb)],
            });
        }
    }

    let rem_id = masks(MaskRole::RemainingInWord {
        suffix_len: ctx.suffix_len(),
        word_bits,
        is_upper,
        pop_msb: ctx.r_x.is_none(),
    });
    result.push(Monomial {
        coeff: 1,
        factors: vec![ValueSource::IndexedPoly(rem_id)],
    });

    result
}

/// Simple sequential allocator: assigns `PolynomialId::PrefixMask(0, 1, 2, ...)`
/// in the order roles are first requested.
#[derive(Default, Debug)]
pub struct MaskAllocator {
    map: HashMap<MaskRole, PolynomialId>,
    order: Vec<(MaskRole, PolynomialId)>,
    next_id: usize,
}

impl MaskAllocator {
    pub fn get_or_alloc(&mut self, role: MaskRole) -> PolynomialId {
        if let Some(id) = self.map.get(&role) {
            return *id;
        }
        let id = PolynomialId::PrefixMask(self.next_id);
        self.next_id += 1;
        let _ = self.map.insert(role, id);
        self.order.push((role, id));
        id
    }

    pub fn into_pairs(self) -> Vec<(MaskRole, PolynomialId)> {
        self.order
    }
}

/// Pre-lower every `(round, rule, c-side)` triple into the ScalarExpr form the
/// runtime expects. The runtime stores the resulting bundles in
/// `InstanceConfig::prefix_lowered`, so `lower_prefix_mle` / `mask_roles_for`
/// never need to run during proving — the runtime only evaluates the baked
/// expressions and mask-role buffers.
///
/// Layout invariants (mirrored by the runtime):
/// - Round `j` uses a fresh `MaskAllocator` per `(rule, c_side)`: buffer maps
///   are rebuilt per rule at runtime, so ID collisions across rules are fine.
/// - `r_x` is represented as the proxy `ChallengeIdx(0)`; the runtime passes a
///   1-element `challenges` slice holding the bound x-challenge at index 0.
/// - `b_len(j) = chunk_bits − (j mod chunk_bits) − 1`; `total_bits = num_phases
///   × chunk_bits` matches the `InstanceConfig::total_address_bits` convention.
pub fn build_prefix_lowered_rounds(
    prefix_mle_rules: &[PrefixMleRule],
    chunk_bits: usize,
    num_phases: usize,
) -> Vec<crate::module::PrefixLoweredRound> {
    let total_bits = chunk_bits * num_phases;
    let mut rounds = Vec::with_capacity(total_bits);
    for j in 0..total_bits {
        let round_in_sub = j % chunk_bits;
        let b_len = chunk_bits - round_in_sub - 1;
        let r_x = if j % 2 == 1 {
            Some(ChallengeIdx(0))
        } else {
            None
        };
        let mut c0 = Vec::with_capacity(prefix_mle_rules.len());
        let mut c1 = Vec::with_capacity(prefix_mle_rules.len());
        let mut masks_c0 = Vec::with_capacity(prefix_mle_rules.len());
        let mut masks_c1 = Vec::with_capacity(prefix_mle_rules.len());
        for rule in prefix_mle_rules {
            for c_side in 0..2u32 {
                let ctx = LoweringCtx {
                    j,
                    b_len,
                    total_bits,
                    r_x,
                    c: c_side,
                };
                let mut alloc = MaskAllocator::default();
                let expr = lower_prefix_mle(rule, ctx, |role| alloc.get_or_alloc(role));
                let bindings: Vec<(PolynomialId, MaskRole)> = alloc
                    .into_pairs()
                    .into_iter()
                    .map(|(role, id)| (id, role))
                    .collect();
                if c_side == 0 {
                    c0.push(expr);
                    masks_c0.push(bindings);
                } else {
                    c1.push(expr);
                    masks_c1.push(bindings);
                }
            }
        }
        rounds.push(crate::module::PrefixLoweredRound {
            b_len,
            c0,
            c1,
            masks_c0,
            masks_c1,
            r_x,
        });
    }
    rounds
}
