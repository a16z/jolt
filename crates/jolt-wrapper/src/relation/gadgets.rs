//! Arithmetic gadgets over linear combinations. Each mirrors one `jolt-poly`
//! formula the verifier evaluates natively; the native function is the
//! witness oracle in the parity tests.

use jolt_claims::protocols::jolt::geometry::dimensions::ReadWriteDimensions;
use jolt_field::{Field, Fr, One, Ring};
use jolt_poly::lagrange::centered_domain_start;

use super::ctx::{lc_const, Accum, Ctx, Lc};

/// `Σ_k c_k · x^k` by Horner: `degree` multiplications.
pub(crate) fn horner(ctx: &mut Ctx, coefficients: &[Lc], x: &Lc) -> Lc {
    let mut iter = coefficients.iter().rev();
    let Some(top) = iter.next() else {
        return Lc::zero();
    };
    let mut acc = top.clone();
    for coefficient in iter {
        acc = ctx.mul(&acc, x) + coefficient.clone();
    }
    acc
}

/// `[1, x, x², …, x^(len−1)]`.
pub(crate) fn powers(ctx: &mut Ctx, x: &Lc, len: usize) -> Vec<Lc> {
    let mut out = Vec::with_capacity(len);
    if len == 0 {
        return out;
    }
    out.push(Lc::one());
    if len > 1 {
        out.push(x.clone());
    }
    for _ in 2..len {
        let next = ctx.mul(&out[out.len() - 1], x);
        out.push(next);
    }
    out
}

pub(crate) fn product(ctx: &mut Ctx, factors: &[Lc]) -> Lc {
    let mut iter = factors.iter();
    let Some(first) = iter.next() else {
        return Lc::one();
    };
    let mut acc = first.clone();
    for factor in iter {
        acc = ctx.mul(&acc, factor);
    }
    acc
}

pub(crate) fn one_minus(lc: &Lc) -> Lc {
    Lc::one() - lc.clone()
}

/// `x·y + (1−x)(1−y) = 1 − x − y + 2xy`.
fn eq_factor(ctx: &mut Ctx, x: &Lc, y: &Lc) -> Lc {
    let xy = ctx.mul(x, y);
    let mut acc = Accum::default();
    acc.add(&Lc::one(), Fr::one());
    acc.add(x, -Fr::one());
    acc.add(y, -Fr::one());
    acc.add(&xy, Fr::from_u64(2));
    acc.finish()
}

/// `EqPolynomial::mle(x, y)`; constant coordinates on either side cost
/// nothing beyond the chain product.
pub(crate) fn eq(ctx: &mut Ctx, x: &[Lc], y: &[Lc]) -> Lc {
    assert_eq!(x.len(), y.len(), "eq gadget arity mismatch");
    let factors: Vec<Lc> = x
        .iter()
        .zip(y)
        .map(|(x_i, y_i)| eq_factor(ctx, x_i, y_i))
        .collect();
    product(ctx, &factors)
}

/// `LtPolynomial::evaluate(x, r)`: `Σ_i (1−x_i)·r_i·eq(x[..i], r[..i])`,
/// most-significant coordinate first.
pub(crate) fn lt(ctx: &mut Ctx, x: &[Lc], r: &[Lc]) -> Lc {
    assert_eq!(x.len(), r.len(), "lt gadget arity mismatch");
    let mut lt = Accum::default();
    let mut eq_prefix = Lc::one();
    for (x_i, r_i) in x.iter().zip(r) {
        let flip = ctx.mul(&one_minus(x_i), r_i);
        let term = ctx.mul(&flip, &eq_prefix);
        lt.add(&term, Fr::one());
        let factor = eq_factor(ctx, x_i, r_i);
        eq_prefix = ctx.mul(&eq_prefix, &factor);
    }
    lt.finish()
}

/// `EqPlusOnePolynomial::new(x).evaluate(y)`: the suffix-length
/// decomposition, with the per-position `x_p·y_p` shared between the
/// `x(1−y)`, `(1−x)y` and `eq` factors.
pub(crate) fn eq_plus_one(ctx: &mut Ctx, x: &[Lc], y: &[Lc]) -> Lc {
    let l = x.len();
    assert_eq!(l, y.len(), "eq+1 gadget arity mismatch");
    // Position p = l − 1 − i: index i runs least-significant first.
    let mut lower_factor = Vec::with_capacity(l);
    let mut flip = Vec::with_capacity(l);
    let mut higher_factor = Vec::with_capacity(l);
    for i in 0..l {
        let p = l - 1 - i;
        let xy = ctx.mul(&x[p], &y[p]);
        lower_factor.push(x[p].clone() - xy.clone());
        flip.push(y[p].clone() - xy.clone());
        let mut eq = Accum::default();
        eq.add(&Lc::one(), Fr::one());
        eq.add(&x[p], -Fr::one());
        eq.add(&y[p], -Fr::one());
        eq.add(&xy, Fr::from_u64(2));
        higher_factor.push(eq.finish());
    }
    // lower[k] = Π_{i<k} lower_factor[i]; higher[k] = Π_{i>k} higher_factor[i].
    let mut lower = vec![Lc::one()];
    for k in 1..l {
        let next = ctx.mul(&lower[k - 1], &lower_factor[k - 1]);
        lower.push(next);
    }
    let mut higher = vec![Lc::one(); l];
    for k in (0..l.saturating_sub(1)).rev() {
        higher[k] = ctx.mul(&higher[k + 1], &higher_factor[k + 1]);
    }
    let mut sum = Accum::default();
    for k in 0..l {
        let term = ctx.mul(&lower[k], &flip[k]);
        let term = ctx.mul(&term, &higher[k]);
        sum.add(&term, Fr::one());
    }
    sum.finish()
}

/// Lagrange basis over the centered integer domain, `L_k(r) = w_k · Π_{j<k}(r−x_j) · Π_{j>k}(r−x_j)`
/// with prefix/suffix products (no inversion hint, agrees with
/// `centered_lagrange_evals` everywhere including the nodes).
#[expect(
    clippy::expect_used,
    reason = "domain sizes are the protocol constants 3 and 10; nodes are distinct integers"
)]
pub(crate) fn centered_lagrange(ctx: &mut Ctx, domain_size: usize, r: &Lc) -> Vec<Lc> {
    let start = centered_domain_start(domain_size).expect("non-empty centered domain");
    let nodes: Vec<i64> = (0..domain_size).map(|k| start + k as i64).collect();
    let diffs: Vec<Lc> = nodes
        .iter()
        .map(|node| r.clone() - lc_const(Fr::from_i64(*node)))
        .collect();
    let mut prefix = vec![Lc::one()];
    for k in 1..domain_size {
        let next = ctx.mul(&prefix[k - 1], &diffs[k - 1]);
        prefix.push(next);
    }
    let mut suffix = vec![Lc::one(); domain_size];
    for k in (0..domain_size.saturating_sub(1)).rev() {
        suffix[k] = ctx.mul(&suffix[k + 1], &diffs[k + 1]);
    }
    (0..domain_size)
        .map(|k| {
            let mut weight = Fr::one();
            for (j, node) in nodes.iter().enumerate() {
                if j != k {
                    weight *= Fr::from_i64(nodes[k] - node);
                }
            }
            let weight = weight.inverse().expect("distinct nodes");
            ctx.mul(&prefix[k], &suffix[k]).scale(weight)
        })
        .collect()
}

/// `centered_lagrange_kernel(domain_size, x, y) = Σ_k L_k(x)·L_k(y)`.
pub(crate) fn centered_lagrange_kernel(ctx: &mut Ctx, domain_size: usize, x: &Lc, y: &Lc) -> Lc {
    let lx = centered_lagrange(ctx, domain_size, x);
    let ly = centered_lagrange(ctx, domain_size, y);
    let mut sum = Accum::default();
    for (a, b) in lx.iter().zip(&ly) {
        let term = ctx.mul(a, b);
        sum.add(&term, Fr::one());
    }
    sum.finish()
}

/// `IdentityPolynomial::evaluate`: `Σ_i r_i · 2^(n−1−i)`.
pub(crate) fn identity_msb(point: &[Lc]) -> Lc {
    let n = point.len();
    let mut acc = Accum::default();
    for (i, r_i) in point.iter().enumerate() {
        acc.add(r_i, Fr::pow2(n - 1 - i));
    }
    acc.finish()
}

/// `OperandPolynomial::evaluate` over an interleaved point: `offset` 0 (left)
/// or 1 (right).
pub(crate) fn operand(point: &[Lc], offset: usize) -> Lc {
    let bits = point.len() / 2;
    let mut acc = Accum::default();
    for bit_index in 0..bits {
        acc.add(
            &point[2 * bit_index + offset],
            Fr::pow2(bits - 1 - bit_index),
        );
    }
    acc.finish()
}

/// `eq_index_msb(point, index) = Π_i (bit_i ? r_i : 1 − r_i)`.
pub(crate) fn eq_index_msb(ctx: &mut Ctx, point: &[Lc], index: u128) -> Lc {
    let n = point.len();
    let factors: Vec<Lc> = point
        .iter()
        .enumerate()
        .map(|(position, r)| {
            let shift = n - 1 - position;
            let bit = if shift < 128 { (index >> shift) & 1 } else { 0 };
            if bit == 1 {
                r.clone()
            } else {
                one_minus(r)
            }
        })
        .collect();
    product(ctx, &factors)
}

fn less_than_mle_msb(ctx: &mut Ctx, bound: u128, point: &[Lc]) -> Lc {
    let n = point.len();
    if n < 128 && bound == 1u128 << n {
        return Lc::one();
    }
    let mut lt = Accum::default();
    let mut eq_bound = Lc::one();
    for (index, r) in point.iter().enumerate() {
        let shift = n - 1 - index;
        let bit = if shift < 128 { (bound >> shift) & 1 } else { 0 };
        if bit == 1 {
            let term = ctx.mul(&eq_bound, &one_minus(r));
            lt.add(&term, Fr::one());
            eq_bound = ctx.mul(&eq_bound, r);
        } else {
            eq_bound = ctx.mul(&eq_bound, &one_minus(r));
        }
    }
    lt.finish()
}

/// `range_mask_mle_msb(start, end, point)`.
pub(crate) fn range_mask_msb(ctx: &mut Ctx, start: u128, end: u128, point: &[Lc]) -> Lc {
    let upper = less_than_mle_msb(ctx, end, point);
    let lower = less_than_mle_msb(ctx, start, point);
    upper - lower
}

/// `Π_i (1 − r_i)`: `EqPolynomial::zero_selector`.
pub(crate) fn zero_selector(ctx: &mut Ctx, point: &[Lc]) -> Lc {
    let factors: Vec<Lc> = point.iter().map(one_minus).collect();
    product(ctx, &factors)
}

pub(crate) fn reversed(point: &[Lc]) -> Vec<Lc> {
    point.iter().rev().cloned().collect()
}

/// `ReadWriteDimensions::read_write_opening_point`: the three-phase batch
/// point regrouped as `(r_address, r_cycle)`.
pub(crate) fn read_write_opening_point(
    dimensions: ReadWriteDimensions,
    point: &[Lc],
) -> (Vec<Lc>, Vec<Lc>) {
    let p1 = dimensions.phase1_num_rounds();
    let p2 = dimensions.phase2_num_rounds();
    let cycle_rounds = dimensions.phase3_cycle_rounds();
    let (phase1, rest) = point.split_at(p1);
    let (phase2, rest) = rest.split_at(p2);
    let (phase3_cycle, phase3_address) = rest.split_at(cycle_rounds);
    let mut r_cycle = reversed(phase3_cycle);
    r_cycle.extend(reversed(phase1));
    let mut r_address = reversed(phase3_address);
    r_address.extend(reversed(phase2));
    (r_address, r_cycle)
}

/// `ReadWriteDimensions::address_opening_point` over a phase-1-offset
/// instance point: phase-2 coordinates, then the phase-3 address coordinates,
/// reversed.
pub(crate) fn address_opening_point(dimensions: ReadWriteDimensions, instance: &[Lc]) -> Vec<Lc> {
    let p2 = dimensions.phase2_num_rounds();
    let cycle_gap = dimensions.phase3_cycle_rounds();
    let mut address = instance[..p2].to_vec();
    address.extend_from_slice(&instance[p2 + cycle_gap..]);
    reversed(&address)
}

/// `committed_address_chunks`: the address zero-padded at the front to a
/// multiple of `chunk_bits`, split into chunks.
pub(crate) fn address_chunks(point: &[Lc], chunk_bits: usize) -> Vec<Vec<Lc>> {
    let padding = point.len().next_multiple_of(chunk_bits) - point.len();
    let mut padded = vec![Lc::zero(); padding];
    padded.extend_from_slice(point);
    padded.chunks(chunk_bits).map(<[Lc]>::to_vec).collect()
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use jolt_claims::protocols::jolt::geometry::dimensions::committed_address_chunks;
    use jolt_field::Field;
    use jolt_poly::lagrange::{centered_lagrange_evals, centered_lagrange_kernel as native_kernel};
    use jolt_poly::{
        eq_index_msb as native_eq_index, range_mask_mle_msb, EqPlusOnePolynomial, EqPolynomial,
        IdentityPolynomial, LtPolynomial, MultilinearEvaluation, OperandPolynomial, OperandSide,
    };
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    use super::*;
    use crate::relation::ctx::lc_var;

    struct Probe {
        ctx: Ctx,
    }

    impl Probe {
        fn new() -> Self {
            Self {
                ctx: Ctx::new(None),
            }
        }

        fn point(&mut self, values: &[Fr]) -> Vec<Lc> {
            values
                .iter()
                .map(|value| lc_var(self.ctx.alloc(Some(*value))))
                .collect()
        }

        fn scalar(&mut self, value: Fr) -> Lc {
            lc_var(self.ctx.alloc(Some(value)))
        }

        /// The gadget's hinted value, after checking every emitted row.
        fn finish(self, lc: &Lc) -> Fr {
            let value = self.ctx.value(lc).unwrap();
            let (matrices, values, _, _) = self.ctx.finish();
            let witness: Vec<Fr> = values.into_iter().map(Option::unwrap).collect();
            matrices.check_witness(&witness).unwrap();
            value
        }
    }

    fn random_point(rng: &mut StdRng, len: usize) -> Vec<Fr> {
        (0..len).map(|_| Fr::random(rng)).collect()
    }

    #[test]
    fn eq_lt_eq_plus_one_match_jolt_poly() {
        let mut rng = StdRng::seed_from_u64(1);
        for len in [1usize, 4, 7, 18] {
            let x = random_point(&mut rng, len);
            let y = random_point(&mut rng, len);
            let mut probe = Probe::new();
            let (wx, wy) = (probe.point(&x), probe.point(&y));
            let out = eq(&mut probe.ctx, &wx, &wy);
            assert_eq!(probe.finish(&out), EqPolynomial::<Fr>::mle(&x, &y));

            let mut probe = Probe::new();
            let (wx, wy) = (probe.point(&x), probe.point(&y));
            let out = lt(&mut probe.ctx, &wx, &wy);
            assert_eq!(probe.finish(&out), LtPolynomial::<Fr>::evaluate(&x, &y));

            let mut probe = Probe::new();
            let (wx, wy) = (probe.point(&x), probe.point(&y));
            let out = eq_plus_one(&mut probe.ctx, &wx, &wy);
            assert_eq!(
                probe.finish(&out),
                EqPlusOnePolynomial::new(x.clone()).evaluate(&y)
            );
        }
    }

    #[test]
    fn lagrange_gadgets_match_jolt_poly() {
        let mut rng = StdRng::seed_from_u64(2);
        for domain_size in [3usize, 10] {
            let r = Fr::random(&mut rng);
            let s = Fr::random(&mut rng);
            let mut probe = Probe::new();
            let wr = probe.scalar(r);
            let basis = centered_lagrange(&mut probe.ctx, domain_size, &wr);
            let native = centered_lagrange_evals::<Fr>(domain_size, r).unwrap();
            let values: Vec<Fr> = basis
                .iter()
                .map(|lc| probe.ctx.value(lc).unwrap())
                .collect();
            assert_eq!(values, native);
            let _ = probe.finish(&basis[0]);

            let mut probe = Probe::new();
            let (wr, ws) = (probe.scalar(r), probe.scalar(s));
            let out = centered_lagrange_kernel(&mut probe.ctx, domain_size, &wr, &ws);
            assert_eq!(
                probe.finish(&out),
                native_kernel::<Fr>(domain_size, r, s).unwrap()
            );
        }
    }

    #[test]
    fn address_gadgets_match_jolt_poly() {
        let mut rng = StdRng::seed_from_u64(3);
        let point = random_point(&mut rng, 16);
        let mut probe = Probe::new();
        let wires = probe.point(&point);
        let identity = identity_msb(&wires);
        assert_eq!(
            probe.ctx.value(&identity).unwrap(),
            IdentityPolynomial::new(16).evaluate(&point)
        );
        let left = operand(&wires, 0);
        let right = operand(&wires, 1);
        assert_eq!(
            probe.ctx.value(&left).unwrap(),
            OperandPolynomial::new(16, OperandSide::Left).evaluate(&point)
        );
        assert_eq!(
            probe.ctx.value(&right).unwrap(),
            OperandPolynomial::new(16, OperandSide::Right).evaluate(&point)
        );
        let index = rng.gen_range(0..1u128 << 16);
        let eq_index = eq_index_msb(&mut probe.ctx, &wires, index);
        assert_eq!(
            probe.ctx.value(&eq_index).unwrap(),
            native_eq_index(&point, index)
        );
        let start = rng.gen_range(0..1u128 << 15);
        let end = rng.gen_range(start..1u128 << 16);
        let mask = range_mask_msb(&mut probe.ctx, start, end, &wires);
        assert_eq!(
            probe.ctx.value(&mask).unwrap(),
            range_mask_mle_msb(start, end, &point).unwrap()
        );
        let zero = zero_selector(&mut probe.ctx, &wires);
        assert_eq!(probe.ctx.value(&zero).unwrap(), native_eq_index(&point, 0));
        let _ = probe.finish(&mask);
    }

    #[test]
    fn point_regrouping_matches_geometry() {
        let mut rng = StdRng::seed_from_u64(4);
        let dimensions = ReadWriteDimensions::new(18, 16, 6, 5);
        let point = random_point(&mut rng, dimensions.read_write_rounds());
        let mut probe = Probe::new();
        let wires = probe.point(&point);
        let (address, cycle) = read_write_opening_point(dimensions, &wires);
        let native = dimensions.read_write_opening_point(&point).unwrap();
        let values =
            |lcs: &[Lc]| -> Vec<Fr> { lcs.iter().map(|lc| probe.ctx.value(lc).unwrap()).collect() };
        assert_eq!(values(&address), native.r_address);
        assert_eq!(values(&cycle), native.r_cycle);
        let mut regrouped = values(&address);
        regrouped.extend(values(&cycle));
        assert_eq!(regrouped, native.opening_point);
        let instance = &wires[dimensions.phase1_num_rounds()..];
        let address_only = address_opening_point(dimensions, instance);
        assert_eq!(
            values(&address_only),
            dimensions
                .address_opening_point(&point[dimensions.phase1_num_rounds()..])
                .unwrap()
        );
        let chunks = address_chunks(&wires[..13], 4);
        let native_chunks = committed_address_chunks(&point[..13], 4);
        assert_eq!(chunks.len(), native_chunks.len());
        for (chunk, native) in chunks.iter().zip(&native_chunks) {
            assert_eq!(values(chunk), *native);
        }
    }
}
