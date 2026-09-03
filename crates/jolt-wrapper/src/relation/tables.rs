//! The 54 lookup-table MLEs over the 128 interleaved instruction-address
//! wires (`x_i = r[2i]`, `y_i = r[2i+1]`, most significant first). Each
//! gadget transcribes `LookupTableKind::evaluate_mle`; the shared chains
//! (`x_i·y_i`, the equality prefix, the shift recurrence, …) are built once
//! per point and reused across tables.

use jolt_field::{Fr, One, Ring};
use jolt_lookup_tables::LookupTableKind;

use super::ctx::{lc_const, Accum, Ctx, Lc};
use super::gadgets::one_minus;

const XLEN: usize = 64;
const HALF: usize = XLEN / 2;
const EIGHTH: usize = XLEN / 8;

fn pow2(exponent: usize) -> Fr {
    Fr::from_u128(1u128 << exponent)
}

fn mask(bits: usize) -> Fr {
    Fr::from_u128((1u128 << bits) - 1)
}

fn weighted(terms: impl Iterator<Item = (Lc, Fr)>) -> Lc {
    let mut acc = Accum::default();
    for (lc, weight) in terms {
        acc.add(&lc, weight);
    }
    acc.finish()
}

/// Shared per-point products, materialized on first use.
struct TablePoint<'a> {
    r: &'a [Lc],
    xy: Vec<Lc>,
    eq_prefix: Option<Vec<Lc>>,
    lt: Option<Lc>,
    none_y_prefix: Option<Vec<Lc>>,
    shift_chain: Option<Lc>,
    shift_chain_half: Option<Lc>,
    pow2_w: Option<Lc>,
}

impl<'a> TablePoint<'a> {
    fn new(ctx: &mut Ctx, r: &'a [Lc]) -> Self {
        assert_eq!(r.len(), 2 * XLEN, "table point arity");
        let xy = (0..XLEN)
            .map(|i| ctx.mul(&r[2 * i], &r[2 * i + 1]))
            .collect();
        Self {
            r,
            xy,
            eq_prefix: None,
            lt: None,
            none_y_prefix: None,
            shift_chain: None,
            shift_chain_half: None,
            pow2_w: None,
        }
    }

    fn x(&self, i: usize) -> &Lc {
        &self.r[2 * i]
    }

    fn y(&self, i: usize) -> &Lc {
        &self.r[2 * i + 1]
    }

    /// `x_i·y_i + (1−x_i)(1−y_i)`.
    fn eq_bit(&self, i: usize) -> Lc {
        Lc::one() - self.x(i).clone() - self.y(i).clone()
            + self.xy[i].clone().scale(Fr::from_u64(2))
    }

    /// `(1−x_i)·y_i`.
    fn lt_bit(&self, i: usize) -> Lc {
        self.y(i).clone() - self.xy[i].clone()
    }

    /// `x_i ⊕ y_i = x_i + y_i − 2·x_i·y_i`.
    fn xor_bit(&self, i: usize) -> Lc {
        self.x(i).clone() + self.y(i).clone() - self.xy[i].clone().scale(Fr::from_u64(2))
    }

    /// `eq_prefix[k] = Π_{i<k} eq_bit(i)`, `k ∈ 0..=XLEN`.
    fn eq_prefix(&mut self, ctx: &mut Ctx) -> &[Lc] {
        if self.eq_prefix.is_none() {
            let mut prefix = vec![Lc::one()];
            for i in 0..XLEN {
                let factor = self.eq_bit(i);
                let next = ctx.mul(&prefix[i], &factor);
                prefix.push(next);
            }
            self.eq_prefix = Some(prefix);
        }
        self.eq_prefix.as_deref().unwrap_or(&[])
    }

    /// Unsigned `x < y`: `Σ_i (1−x_i)·y_i·eq_prefix[i]`.
    fn lt(&mut self, ctx: &mut Ctx) -> Lc {
        if let Some(lt) = &self.lt {
            return lt.clone();
        }
        let prefix = self.eq_prefix(ctx).to_vec();
        let mut acc = Accum::default();
        for (i, eq_prefix) in prefix.iter().enumerate().take(XLEN) {
            let term = ctx.mul(&self.lt_bit(i), eq_prefix);
            acc.add(&term, Fr::one());
        }
        let lt = acc.finish();
        self.lt = Some(lt.clone());
        lt
    }

    /// `none_y[k] = Π_{i<k} (1−y_i)`, `k ∈ 0..=XLEN`.
    fn none_y_prefix(&mut self, ctx: &mut Ctx) -> &[Lc] {
        if self.none_y_prefix.is_none() {
            let mut prefix = vec![Lc::one()];
            for i in 0..XLEN {
                let next = ctx.mul(&prefix[i], &one_minus(self.y(i)));
                prefix.push(next);
            }
            self.none_y_prefix = Some(prefix);
        }
        self.none_y_prefix.as_deref().unwrap_or(&[])
    }

    /// `acc ← acc·(1+y_i) + x_i·y_i` over `range` (the SRL/PEXT recurrence).
    fn shift_recurrence(&self, ctx: &mut Ctx, range: std::ops::Range<usize>) -> Lc {
        let mut acc = Lc::zero();
        for i in range {
            let scaled = ctx.mul(&acc, self.y(i));
            acc = acc + scaled + self.xy[i].clone();
        }
        acc
    }

    fn shift_chain(&mut self, ctx: &mut Ctx) -> Lc {
        if self.shift_chain.is_none() {
            self.shift_chain = Some(self.shift_recurrence(ctx, 0..XLEN));
        }
        self.shift_chain.clone().unwrap_or_else(Lc::zero)
    }

    fn shift_chain_half(&mut self, ctx: &mut Ctx) -> Lc {
        if self.shift_chain_half.is_none() {
            self.shift_chain_half = Some(self.shift_recurrence(ctx, HALF..XLEN));
        }
        self.shift_chain_half.clone().unwrap_or_else(Lc::zero)
    }

    /// `Π_{i<log} (1 + (2^(2^i) − 1)·r[len−1−i])`.
    fn pow2_tail(&self, ctx: &mut Ctx, log: usize) -> Lc {
        let mut acc = Lc::one();
        for i in 0..log {
            let factor = Lc::one() + self.r[2 * XLEN - 1 - i].clone().scale(mask(1 << i));
            acc = ctx.mul(&acc, &factor);
        }
        acc
    }

    fn pow2_w(&mut self, ctx: &mut Ctx) -> Lc {
        if self.pow2_w.is_none() {
            self.pow2_w = Some(self.pow2_tail(ctx, HALF.trailing_zeros() as usize));
        }
        self.pow2_w.clone().unwrap_or_else(Lc::zero)
    }

    /// `Σ_{i<XLEN} 2^(XLEN−1−i) · r[XLEN+i]` restricted to `i < limit`.
    fn right_operand(&self, limit: usize) -> Lc {
        weighted((0..limit).map(|i| (self.r[XLEN + i].clone(), pow2(XLEN - 1 - i))))
    }

    /// Rotate-right variants: `first_sum` is the shift recurrence, `second_sum`
    /// re-inserts the shifted-out bits at the top.
    fn rotr(&self, ctx: &mut Ctx, range: std::ops::Range<usize>, first_sum: &Lc) -> Lc {
        let mut prod_one_plus_y = Lc::one();
        let mut second = Accum::default();
        for i in range {
            let carried = ctx.mul(&(self.x(i).clone() - self.xy[i].clone()), &prod_one_plus_y);
            second.add(&carried, pow2(XLEN - 1 - i));
            prod_one_plus_y = ctx.mul(&prod_one_plus_y, &(Lc::one() + self.y(i).clone()));
        }
        first_sum.clone() + second.finish()
    }

    /// `lane · Π_{i ∈ steps} (1 + (2^(EIGHTH·2^i) − 1) · bit_i)`.
    fn window(
        ctx: &mut Ctx,
        lane: Lc,
        steps: std::ops::Range<usize>,
        bit: impl Fn(usize) -> Lc,
    ) -> Lc {
        let mut acc = lane;
        for i in steps {
            let factor = Lc::one() + bit(i).scale(mask(EIGHTH << i));
            acc = ctx.mul(&acc, &factor);
        }
        acc
    }

    fn shift_data(&self, ctx: &mut Ctx, lane_bits: usize, steps: std::ops::Range<usize>) -> Lc {
        let lane = weighted((0..lane_bits).map(|k| (self.x(XLEN - 1 - k).clone(), pow2(k))));
        Self::window(ctx, lane, steps, |i| self.y(XLEN - 1 - i).clone())
    }

    fn evaluate(&mut self, ctx: &mut Ctx, kind: &LookupTableKind<XLEN>) -> Lc {
        use LookupTableKind as K;
        match kind {
            K::RangeCheck(_) => self.right_operand(XLEN),
            K::RangeCheckAligned(_) => self.right_operand(XLEN - 1),
            K::And(_) => weighted((0..XLEN).map(|i| (self.xy[i].clone(), pow2(XLEN - 1 - i)))),
            K::Andn(_) => weighted(
                (0..XLEN).map(|i| (self.x(i).clone() - self.xy[i].clone(), pow2(XLEN - 1 - i))),
            ),
            K::Or(_) => weighted((0..XLEN).map(|i| {
                (
                    self.x(i).clone() + self.y(i).clone() - self.xy[i].clone(),
                    pow2(XLEN - 1 - i),
                )
            })),
            K::Xor(_) => weighted((0..XLEN).map(|i| (self.xor_bit(i), pow2(XLEN - 1 - i)))),
            K::Equal(_) => self.eq_prefix(ctx)[XLEN].clone(),
            K::NotEqual(_) => Lc::one() - self.eq_prefix(ctx)[XLEN].clone(),
            K::SignedLessThan(_) => self.x(0).clone() - self.y(0).clone() + self.lt(ctx),
            K::SignedGreaterThanEqual(_) => {
                Lc::one() - self.x(0).clone() + self.y(0).clone() - self.lt(ctx)
            }
            K::UnsignedLessThan(_) => self.lt(ctx),
            K::UnsignedGreaterThanEqual(_) => Lc::one() - self.lt(ctx),
            K::UnsignedLessThanEqual(_) => self.lt(ctx) + self.eq_prefix(ctx)[XLEN].clone(),
            K::SignMask(_) => self.r[0].clone().scale(mask(XLEN)),
            K::UpperWord(_) => weighted((0..XLEN).map(|i| (self.r[i].clone(), pow2(XLEN - 1 - i)))),
            K::ValidUnsignedRemainder(_) => self.lt(ctx) + self.none_y_prefix(ctx)[XLEN].clone(),
            K::ValidDiv0(_) => {
                let mut divisor_is_zero = Lc::one();
                let mut valid = Lc::one();
                for i in 0..XLEN {
                    divisor_is_zero = ctx.mul(&divisor_is_zero, &one_minus(self.x(i)));
                    valid = ctx.mul(&valid, &self.lt_bit(i));
                }
                Lc::one() - divisor_is_zero + valid
            }
            K::HalfwordAlignment(_) => one_minus(&self.r[2 * XLEN - 1]),
            K::WordAlignment(_) => ctx.mul(
                &one_minus(&self.r[2 * XLEN - 1]),
                &one_minus(&self.r[2 * XLEN - 2]),
            ),
            K::LowerHalfWord(_) => {
                weighted((0..HALF).map(|i| (self.r[XLEN + HALF + i].clone(), pow2(HALF - 1 - i))))
            }
            K::SignExtendWord(_) => {
                let lower = weighted(
                    (0..HALF).map(|i| (self.r[XLEN + HALF + i].clone(), pow2(HALF - 1 - i))),
                );
                lower + self.r[XLEN + HALF].clone().scale(mask(HALF) * pow2(HALF))
            }
            K::Pow2(_) => self.pow2_tail(ctx, XLEN.trailing_zeros() as usize),
            K::Pow2W(_) => self.pow2_w(ctx),
            K::ShiftRightBitmask(_) => {
                let log_w = XLEN.trailing_zeros() as usize;
                let tail = &self.r[2 * XLEN - log_w..];
                // eq tensor over the tail, bit i of the shift paired with tail[log_w − 1 − i].
                // Bit i of the shift amount pairs with `tail[log_w − 1 − i]`; the
                // tensor is built most-significant bit first so entry `s` is
                // `eq(bits(s), tail)`.
                let mut tensor = vec![Lc::one()];
                for i in (0..log_w).rev() {
                    let bit = &tail[log_w - 1 - i];
                    let mut next = Vec::with_capacity(tensor.len() * 2);
                    for entry in &tensor {
                        let on = ctx.mul(entry, bit);
                        next.push(entry.clone() - on.clone());
                        next.push(on);
                    }
                    tensor = next;
                }
                weighted((0..XLEN).map(|s| {
                    (
                        tensor[s].clone(),
                        Fr::from_u128(((1u128 << (XLEN - s)) - 1) << s),
                    )
                }))
            }
            K::ShiftRightBitmaskW(_) => lc_const(pow2(HALF)) - self.pow2_w(ctx),
            K::VirtualRev8W(_) => {
                let byte = |j: usize| {
                    weighted((0..8).map(|i| (self.r[2 * XLEN - 1 - 8 * j - i].clone(), pow2(i))))
                };
                let bytes = [
                    byte(3),
                    byte(2),
                    byte(1),
                    byte(0),
                    byte(7),
                    byte(6),
                    byte(5),
                    byte(4),
                ];
                weighted(bytes.into_iter().enumerate().map(|(k, b)| (b, pow2(8 * k))))
            }
            K::VirtualSRL(_) | K::Pext(_) => self.shift_chain(ctx),
            K::VirtualSRA(_) => {
                let chain = self.shift_chain(ctx);
                let sign_extension = weighted((1..XLEN).map(|i| (one_minus(self.y(i)), pow2(i))));
                chain + ctx.mul(&self.r[0], &sign_extension)
            }
            K::VirtualSRLW(_) => {
                let chain = self.shift_chain_half(ctx);
                let top = ctx.mul(&self.r[XLEN], &self.r[2 * XLEN - 1]);
                chain + top.scale(pow2(XLEN) - pow2(HALF))
            }
            K::VirtualSRAW(_) => {
                let chain = self.shift_chain_half(ctx);
                let sign_extension = lc_const(pow2(XLEN) - pow2(HALF))
                    + weighted((1..HALF).map(|i| (one_minus(self.y(HALF + i)), pow2(i))));
                chain + ctx.mul(&self.r[XLEN], &sign_extension)
            }
            K::VirtualROTR(_) => {
                let first = self.shift_chain(ctx);
                self.rotr(ctx, 0..XLEN, &first)
            }
            K::VirtualROTRW(_) => {
                let first = self.shift_chain_half(ctx);
                self.rotr(ctx, HALF..XLEN, &first)
            }
            K::VirtualNegateIf(_) => {
                let sign = &self.r[0];
                let value = weighted((0..XLEN).map(|i| (self.y(i).clone(), pow2(XLEN - 1 - i))));
                let value_is_zero = self.none_y_prefix(ctx)[XLEN].clone();
                let sign_value = ctx.mul(sign, &value);
                let sign_zero = ctx.mul(sign, &value_is_zero);
                value - sign_value.scale(Fr::from_u64(2)) + sign.clone().scale(pow2(XLEN))
                    - sign_zero.scale(pow2(XLEN))
            }
            K::MulUNoOverflow(_) => {
                let mut acc = Lc::one();
                for i in 0..XLEN {
                    acc = ctx.mul(&acc, &one_minus(&self.r[i]));
                }
                acc
            }
            K::VirtualXORROT32(_) => self.xor_rot(32),
            K::VirtualXORROT24(_) => self.xor_rot(24),
            K::VirtualXORROT16(_) => self.xor_rot(16),
            K::VirtualXORROT63(_) => self.xor_rot(63),
            K::VirtualXORROTW16(_) => self.xor_rotw(16),
            K::VirtualXORROTW12(_) => self.xor_rotw(12),
            K::VirtualXORROTW8(_) => self.xor_rotw(8),
            K::VirtualXORROTW7(_) => self.xor_rotw(7),
            K::VirtualXORROTW22(_) => self.xor_rotw(22),
            K::VirtualXORROTW19(_) => self.xor_rotw(19),
            K::VirtualXORROTW6(_) => self.xor_rotw(6),
            K::WindowMaskW(_) => {
                lc_const(mask(HALF)) + self.r[2 * XLEN - 3].clone().scale(mask(HALF) * mask(HALF))
            }
            K::PextSigned(_) => {
                let pext = self.shift_chain(ctx);
                let none = self.none_y_prefix(ctx).to_vec();
                let mut sigma = Accum::default();
                let mut sig2pc = Lc::zero();
                for (i, none_before) in none.iter().enumerate().take(XLEN) {
                    let gated = ctx.mul(none_before, &self.xy[i]);
                    let carried = ctx.mul(&sig2pc, self.y(i));
                    sig2pc = sig2pc + carried + gated.clone().scale(Fr::from_u64(2));
                    sigma.add(&gated, Fr::one());
                }
                pext + sigma.finish().scale(pow2(XLEN)) - sig2pc
            }
            K::WindowMaskB(_) => Self::window(ctx, lc_const(mask(EIGHTH)), 0..3, |i| {
                self.r[2 * XLEN - 1 - i].clone()
            }),
            K::WindowMaskH(_) => Self::window(ctx, lc_const(mask(2 * EIGHTH)), 1..3, |i| {
                self.r[2 * XLEN - 1 - i].clone()
            }),
            K::AlignAddr(_) => self.right_operand(XLEN - 3),
            K::ShiftDataB(_) => self.shift_data(ctx, EIGHTH, 0..3),
            K::ShiftDataH(_) => self.shift_data(ctx, 2 * EIGHTH, 1..3),
            K::ShiftDataW(_) => self.shift_data(ctx, 4 * EIGHTH, 2..3),
        }
    }

    fn xor_rot(&self, rotation: usize) -> Lc {
        weighted((0..XLEN).map(|i| (self.xor_bit(i), pow2(XLEN - 1 - (i + rotation) % XLEN))))
    }

    fn xor_rotw(&self, rotation: usize) -> Lc {
        weighted((HALF..XLEN).map(|idx| {
            let rotated = (idx - HALF + rotation) % HALF;
            (self.xor_bit(idx), pow2(HALF - 1 - rotated))
        }))
    }
}

/// Every table MLE at `r`, in `LookupTableKind::iter()` order.
pub(crate) fn table_mles(ctx: &mut Ctx, r: &[Lc]) -> Vec<Lc> {
    let mut point = TablePoint::new(ctx, r);
    LookupTableKind::<XLEN>::iter()
        .map(|kind| point.evaluate(ctx, &kind))
        .collect()
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use jolt_field::Field;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    use super::*;
    use crate::relation::ctx::lc_var;

    /// Every table gadget agrees with `LookupTableKind::evaluate_mle` on
    /// random points, and the rows it emits are satisfied by its own hints.
    #[test]
    fn table_gadgets_match_native_mles() {
        let mut rng = StdRng::seed_from_u64(0x7ab1e);
        for _ in 0..200 {
            let point: Vec<Fr> = (0..2 * XLEN).map(|_| Fr::random(&mut rng)).collect();
            let mut ctx = Ctx::new(None);
            let wires: Vec<Lc> = point
                .iter()
                .map(|value| lc_var(ctx.alloc(Some(*value))))
                .collect();
            let gadgets = table_mles(&mut ctx, &wires);
            let mismatches: Vec<String> = LookupTableKind::<XLEN>::iter()
                .zip(&gadgets)
                .filter(|(kind, gadget)| {
                    ctx.value(gadget).unwrap() != kind.evaluate_mle::<Fr, Fr>(&point)
                })
                .map(|(kind, _)| format!("{kind:?}"))
                .collect();
            assert!(mismatches.is_empty(), "{mismatches:?}");
            let (matrices, values, _, _) = ctx.finish();
            let witness: Vec<Fr> = values.into_iter().map(Option::unwrap).collect();
            matrices.check_witness(&witness).unwrap();
        }
    }

    #[test]
    fn table_row_budget() {
        let mut ctx = Ctx::new(None);
        let wires: Vec<Lc> = (0..2 * XLEN).map(|_| lc_var(ctx.alloc(None))).collect();
        let _ = table_mles(&mut ctx, &wires);
        let (matrices, _, _, _) = ctx.finish();
        assert!(
            matrices.num_constraints <= 1_300,
            "54 tables cost {} rows",
            matrices.num_constraints
        );
    }
}
