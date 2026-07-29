use std::sync::{Arc, RwLock};

use allocative::Allocative;
use num::Integer;

use crate::field::{ChallengeFieldOps, FieldChallengeOps, JoltField};
use crate::poly::multilinear_polynomial::MultilinearPolynomial;
use crate::poly::prefix_suffix::{
    CachedPolynomial, Prefix, PrefixCheckpoints, PrefixPolynomial, PrefixRegistry,
    PrefixSuffixPolynomial, SuffixPolynomial,
};
use crate::utils::lookup_bits::LookupBits;
use crate::utils::math::Math;
use crate::utils::uninterleave_bits;

use super::multilinear_polynomial::{BindingOrder, PolynomialBinding, PolynomialEvaluation};

#[derive(Clone, Debug, Allocative)]
pub struct IdentityPolynomial<F: JoltField> {
    num_vars: usize,
    num_bound_vars: usize,
    bound_value: F,
}

impl<F: JoltField> IdentityPolynomial<F> {
    pub fn new(num_vars: usize) -> Self {
        IdentityPolynomial {
            num_vars,
            num_bound_vars: 0,
            bound_value: F::zero(),
        }
    }
}

impl<F: JoltField> PolynomialBinding<F> for IdentityPolynomial<F> {
    fn is_bound(&self) -> bool {
        self.num_bound_vars != 0
    }

    fn bind(&mut self, r: F::Challenge, order: BindingOrder) {
        debug_assert!(self.num_bound_vars < self.num_vars);

        match order {
            BindingOrder::LowToHigh => {
                self.bound_value += F::from_u128(1 << self.num_bound_vars) * r;
            }
            BindingOrder::HighToLow => {
                self.bound_value += self.bound_value;
                self.bound_value = self.bound_value + r;
            }
        }
        self.num_bound_vars += 1;
    }

    fn bind_parallel(&mut self, r: F::Challenge, order: BindingOrder) {
        // Binding is constant time, no parallelism necessary
        self.bind(r, order);
    }

    fn final_sumcheck_claim(&self) -> F {
        debug_assert_eq!(self.num_vars, self.num_bound_vars);
        self.bound_value
    }
}

impl<F: JoltField> PolynomialEvaluation<F> for IdentityPolynomial<F> {
    fn evaluate<C>(&self, r: &[C]) -> F
    where
        C: Copy + Send + Sync + Into<F>,
        F: std::ops::Mul<C, Output = F> + std::ops::SubAssign<F>,
    {
        let len = r.len();
        debug_assert_eq!(len, self.num_vars);
        (0..len)
            .map(|i| r[i].into().mul_u128(1 << (len - 1 - i)))
            .sum()
    }

    fn batch_evaluate<C>(_polys: &[&Self], _r: &[C]) -> Vec<F>
    where
        C: Copy + Send + Sync + Into<F> + ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>,
    {
        unimplemented!("Currently unused")
    }

    fn sumcheck_evals(&self, index: usize, degree: usize, order: BindingOrder) -> Vec<F> {
        let mut evals = vec![F::zero(); degree];
        let m = match order {
            BindingOrder::LowToHigh => {
                let m = F::from_u128(1 << self.num_bound_vars);
                evals[0] = self.bound_value + (m + m).mul_u64(index as u64);
                m
            }
            BindingOrder::HighToLow => {
                let m = F::from_u128(1 << (self.num_vars - 1 - self.num_bound_vars));
                evals[0] = self.bound_value * (m + m) + F::from_u64(index as u64);
                m
            }
        };

        let mut eval = evals[0] + m;
        for i in 1..degree {
            eval += m;
            evals[i] = eval;
        }
        evals
    }
}

impl<F: JoltField> PrefixSuffixPolynomial<F, 2> for IdentityPolynomial<F> {
    fn suffixes(&self) -> [Box<dyn SuffixPolynomial<F> + Sync>; 2] {
        [Box::new(ShiftSuffixPolynomial), Box::new(self.clone())]
    }

    fn prefixes(
        &self,
        chunk_len: usize,
        phase: usize,
        registry: &mut PrefixRegistry<F>,
    ) -> [Option<Arc<RwLock<CachedPolynomial<F>>>>; 2] {
        if registry[Prefix::Identity].is_none() {
            registry[Prefix::Identity] = Some(Arc::new(RwLock::new(self.prefix_polynomial(
                &registry.checkpoints,
                chunk_len,
                phase,
            ))));
        }
        [registry[Prefix::Identity].clone(), None]
    }
}

impl<F: JoltField> PrefixPolynomial<F> for IdentityPolynomial<F> {
    fn prefix_polynomial(
        &self,
        checkpoints: &PrefixCheckpoints<F>,
        chunk_len: usize,
        _phase: usize,
    ) -> CachedPolynomial<F> {
        debug_assert!(chunk_len.is_even());
        let bound_value = checkpoints[Prefix::Identity].unwrap_or(F::zero());

        let evals: Vec<F> = (0..chunk_len.pow2())
            .map(|i| bound_value.mul_u128(1 << chunk_len) + F::from_u64(i as u64))
            .collect();

        CachedPolynomial::new(MultilinearPolynomial::from(evals), (chunk_len - 1).pow2())
    }
}

impl<F: JoltField> SuffixPolynomial<F> for IdentityPolynomial<F> {
    fn suffix_mle(&self, b: LookupBits) -> u128 {
        debug_assert!(b.len().is_even());
        u128::from(b)
    }
}

pub struct ShiftSuffixPolynomial;
impl<F: JoltField> SuffixPolynomial<F> for ShiftSuffixPolynomial {
    fn suffix_mle(&self, b: LookupBits) -> u128 {
        debug_assert!(b.len().is_even());
        1u128 << b.len()
    }
}

pub struct ShiftHalfSuffixPolynomial;
impl<F: JoltField> SuffixPolynomial<F> for ShiftHalfSuffixPolynomial {
    fn suffix_mle(&self, b: LookupBits) -> u128 {
        debug_assert!(b.len().is_even());
        1u128 << (b.len() / 2)
    }
}

/// `U(k) = ∏_{i < num_vars/2} k_i` — the indicator that the upper half of a
/// lookup index is all ones. Already multilinear, so this *is* its own MLE.
///
/// Instruction read-RAF folds `U` in to force identity-RAF addresses to be
/// canonical. Over a field with `p < 2^num_vars` the identity leg pins the
/// committed address only modulo `p`, so `k` and `k + p` are interchangeable;
/// every such alias has an all-ones upper half, and conversely `U(k) = 0`
/// implies `k < 2^num_vars - 2^(num_vars/2) < p`, which is canonical. See
/// `jolt_claims::protocols::jolt::geometry::instruction::CANONICAL_INSTRUCTION_ADDRESS`.
///
/// WARNING: the upper half is the *leading* half of the coordinate vector —
/// `r[0]` is the most significant bit, matching [`IdentityPolynomial`]. Taking
/// the trailing half would both miss every alias and reject honest cycles:
/// `SUB(2^64-1, 0)` has index `0x1_FFFF_FFFF_FFFF_FFFF`, whose *low* limb is
/// all ones.
#[derive(Clone, Copy, Debug, Allocative)]
pub struct UpperHalfAllOnesPolynomial {
    num_vars: usize,
}

impl UpperHalfAllOnesPolynomial {
    pub fn new(num_vars: usize) -> Self {
        debug_assert!(num_vars.is_even());
        Self { num_vars }
    }

    /// How many of a suffix's coordinates fall in the upper half. Zero once the
    /// bound prefix has passed the midpoint, after which `U` is determined.
    fn suffix_upper_bits(&self, suffix_len: usize) -> usize {
        upper_half_suffix_bits(suffix_len, self.num_vars)
    }
}

/// How many of a suffix's coordinates fall in the address's upper half.
///
/// Shared with `PrefixSuffixDecomposition::init_Q_raf`, which accumulates this
/// polynomial's `Q` in its fused trace scan rather than through
/// [`SuffixPolynomial::suffix_mle`]. Keep the two on this one definition: the
/// generic `init_Q` path is only reached from tests, so a divergence here would
/// leave the brute-force decomposition test green while the prover computed
/// something else.
#[inline]
pub(crate) fn upper_half_suffix_bits(suffix_len: usize, num_vars: usize) -> usize {
    suffix_len.saturating_sub(num_vars / 2)
}

/// Whether a suffix leaves the upper half's still-unbound coordinates all ones.
///
/// Vacuously true once the suffix no longer reaches into the upper half, where
/// the bound prefix carries the whole value. See [`upper_half_suffix_bits`] for
/// why this is shared rather than written out at each call site.
#[inline]
pub(crate) fn upper_half_suffix_all_ones(
    suffix_bits: u128,
    suffix_len: usize,
    upper_bits: usize,
) -> bool {
    upper_bits == 0 || (suffix_bits >> (suffix_len - upper_bits)) == (1u128 << upper_bits) - 1
}

impl<F: JoltField> PolynomialEvaluation<F> for UpperHalfAllOnesPolynomial {
    fn evaluate<C>(&self, r: &[C]) -> F
    where
        C: Copy + Send + Sync + Into<F> + ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>,
    {
        debug_assert_eq!(r.len(), self.num_vars);
        r[..self.num_vars / 2]
            .iter()
            .fold(F::one(), |acc, coordinate| acc * (*coordinate).into())
    }

    fn batch_evaluate<C>(_polys: &[&Self], _r: &[C]) -> Vec<F>
    where
        C: Copy + Send + Sync + Into<F> + ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>,
    {
        unimplemented!("Currently unused")
    }

    fn sumcheck_evals(&self, _index: usize, _degree: usize, _order: BindingOrder) -> Vec<F> {
        unimplemented!("Only reached through PrefixSuffixDecomposition")
    }
}

impl<F: JoltField> SuffixPolynomial<F> for UpperHalfAllOnesPolynomial {
    fn suffix_mle(&self, b: LookupBits) -> u128 {
        // `upper_bits == 0` contributes 1: `U` no longer depends on the unbound
        // variables, and the checkpoint carries the whole value.
        let upper_bits = self.suffix_upper_bits(b.len());
        u128::from(upper_half_suffix_all_ones(
            u128::from(b),
            b.len(),
            upper_bits,
        ))
    }
}

impl<F: JoltField> PrefixPolynomial<F> for UpperHalfAllOnesPolynomial {
    fn prefix_polynomial(
        &self,
        checkpoints: &PrefixCheckpoints<F>,
        chunk_len: usize,
        phase: usize,
    ) -> CachedPolynomial<F> {
        // WARNING: the empty product is ONE. Unlike the arithmetic prefixes, an
        // unset checkpoint here means "no upper-half bits bound yet", not zero —
        // defaulting to zero would make `U` vanish identically and silently turn
        // the canonical-address check into a no-op.
        let bound_value = checkpoints[Prefix::UpperHalfAllOnes].unwrap_or(F::one());
        let bound_bits = phase * chunk_len;
        let chunk_upper_bits = (self.num_vars / 2)
            .saturating_sub(bound_bits)
            .min(chunk_len);

        let evals: Vec<F> = (0..chunk_len.pow2())
            .map(|i| {
                // Vacuously true once binding has passed the midpoint, where the
                // chunk carries no upper-half coordinates and `U` is already fixed.
                let chunk_all_ones = chunk_upper_bits == 0
                    || (i >> (chunk_len - chunk_upper_bits)) == (1 << chunk_upper_bits) - 1;
                if chunk_all_ones {
                    bound_value
                } else {
                    F::zero()
                }
            })
            .collect();

        CachedPolynomial::new(MultilinearPolynomial::from(evals), (chunk_len - 1).pow2())
    }
}

impl<F: JoltField> PrefixSuffixPolynomial<F, 1> for UpperHalfAllOnesPolynomial {
    fn suffixes(&self) -> [Box<dyn SuffixPolynomial<F> + Sync>; 1] {
        [Box::new(*self)]
    }

    fn prefixes(
        &self,
        chunk_len: usize,
        phase: usize,
        registry: &mut PrefixRegistry<F>,
    ) -> [Option<Arc<RwLock<CachedPolynomial<F>>>>; 1] {
        if registry[Prefix::UpperHalfAllOnes].is_none() {
            registry[Prefix::UpperHalfAllOnes] = Some(Arc::new(RwLock::new(
                self.prefix_polynomial(&registry.checkpoints, chunk_len, phase),
            )));
        }
        [registry[Prefix::UpperHalfAllOnes].clone()]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OperandSide {
    Left,
    Right,
}

/// OperandPolynomial evaluates to either the right or left operand value from uninterleaved bits
/// For Right: sum_{i=0}^{num_vars/2-1} r[2i] * 2^(num_vars/2 - 1 - i)
/// For Left: sum_{i=0}^{num_vars/2-1} r[2i + 1] * 2^(num_vars/2 - 1 - i)
#[derive(Clone)]
pub struct OperandPolynomial<F: JoltField> {
    num_vars: usize,
    num_bound_vars: usize,
    bound_value: F,
    side: OperandSide,
}

impl<F: JoltField> OperandPolynomial<F> {
    pub fn new(num_vars: usize, side: OperandSide) -> Self {
        debug_assert!(num_vars.is_even(), "num_vars must be divisible by 2");
        OperandPolynomial {
            num_vars,
            num_bound_vars: 0,
            bound_value: F::zero(),
            side,
        }
    }
}

impl<F: JoltField> PolynomialBinding<F> for OperandPolynomial<F> {
    fn is_bound(&self) -> bool {
        self.num_bound_vars != 0
    }

    fn bind(&mut self, r: F::Challenge, order: BindingOrder) {
        debug_assert!(self.num_bound_vars < self.num_vars);
        debug_assert_eq!(
            order,
            BindingOrder::HighToLow,
            "OperandPolynomial only supports high-to-low binding"
        );

        if (self.num_bound_vars.is_even() && self.side == OperandSide::Left)
            || (self.num_bound_vars.is_odd() && self.side == OperandSide::Right)
        {
            self.bound_value += self.bound_value;
            self.bound_value += r.into();
        }
        self.num_bound_vars += 1;
    }

    fn bind_parallel(&mut self, r: F::Challenge, order: BindingOrder) {
        // Binding is constant time, no parallelism necessary
        self.bind(r, order);
    }

    fn final_sumcheck_claim(&self) -> F {
        debug_assert_eq!(self.num_vars, self.num_bound_vars);
        self.bound_value
    }
}

impl<F: JoltField> PolynomialEvaluation<F> for OperandPolynomial<F> {
    fn evaluate<C>(&self, r: &[C]) -> F
    where
        C: Copy + Send + Sync + Into<F>,
        F: std::ops::Mul<C, Output = F> + std::ops::SubAssign<F>,
    {
        let len = r.len();
        debug_assert_eq!(len, self.num_vars);
        debug_assert!(len.is_even());

        match self.side {
            OperandSide::Left => (0..len / 2)
                .map(|i| r[2 * i].into().mul_u128(1 << (self.num_vars / 2 - 1 - i)))
                .sum(),
            OperandSide::Right => (0..len / 2)
                .map(|i| {
                    r[2 * i + 1]
                        .into()
                        .mul_u128(1 << (self.num_vars / 2 - 1 - i))
                })
                .sum(),
        }
    }

    fn batch_evaluate<C>(_polys: &[&Self], _r: &[C]) -> Vec<F>
    where
        C: Copy + Send + Sync + Into<F> + ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>,
    {
        unimplemented!("Currently unused")
    }

    fn sumcheck_evals(&self, index: usize, degree: usize, order: BindingOrder) -> Vec<F> {
        debug_assert_eq!(
            order,
            BindingOrder::HighToLow,
            "OperandPolynomial only supports high-to-low binding"
        );

        let mut evals = vec![F::zero(); degree];
        let (left, right) = uninterleave_bits(index as u128);

        let index = match self.side {
            OperandSide::Left => F::from_u64(left),
            OperandSide::Right => F::from_u64(right),
        };

        if self.num_bound_vars.is_even() {
            let unbound_pairs = (self.num_vars - self.num_bound_vars) / 2;
            debug_assert!(unbound_pairs > 0);
            evals[0] = self.bound_value.mul_u128(1 << unbound_pairs) + index;
            if self.side == OperandSide::Left {
                let m = F::from_u128(1 << (unbound_pairs - 1));
                let mut eval = evals[0] + m;
                for i in 1..degree {
                    eval += m;
                    evals[i] = eval;
                }
            } else {
                for i in 1..degree {
                    evals[i] = evals[0];
                }
            }
        } else if self.side == OperandSide::Right {
            // We are currently bindinng the right operand variable
            let unbound_pairs = (self.num_vars - self.num_bound_vars).div_ceil(2);
            evals[0] = self.bound_value.mul_u128(1 << unbound_pairs) + index;
            let m = F::from_u128(1 << (unbound_pairs - 1));
            let mut eval = evals[0] + m;
            for i in 1..degree {
                eval += m;
                evals[i] = eval;
            }
        } else {
            // We are currently binding right operand variable, but polynoimal is LeftOperand
            let unbound_pairs = (self.num_vars - self.num_bound_vars) / 2;
            evals[0] = self.bound_value.mul_u128(1 << unbound_pairs) + index;
            for i in 1..degree {
                evals[i] = evals[0];
            }
        }

        evals
    }
}

impl<F: JoltField> PrefixSuffixPolynomial<F, 2> for OperandPolynomial<F> {
    fn suffixes(&self) -> [Box<dyn SuffixPolynomial<F> + Sync>; 2] {
        [Box::new(ShiftHalfSuffixPolynomial), Box::new(self.clone())]
    }

    fn prefixes(
        &self,
        chunk_len: usize,
        phase: usize,
        prefix_registry: &mut PrefixRegistry<F>,
    ) -> [Option<Arc<RwLock<CachedPolynomial<F>>>>; 2] {
        match self.side {
            OperandSide::Left => {
                if prefix_registry[Prefix::LeftOperand].is_none() {
                    let lo_poly = OperandPolynomial::new(self.num_vars, OperandSide::Left);
                    prefix_registry[Prefix::LeftOperand] = Some(Arc::new(RwLock::new(
                        lo_poly.prefix_polynomial(&prefix_registry.checkpoints, chunk_len, phase),
                    )));
                }
                [prefix_registry[Prefix::LeftOperand].clone(), None]
            }
            OperandSide::Right => {
                if prefix_registry[Prefix::RightOperand].is_none() {
                    let ro_poly = OperandPolynomial::new(self.num_vars, OperandSide::Right);
                    prefix_registry[Prefix::RightOperand] = Some(Arc::new(RwLock::new(
                        ro_poly.prefix_polynomial(&prefix_registry.checkpoints, chunk_len, phase),
                    )));
                }
                [prefix_registry[Prefix::RightOperand].clone(), None]
            }
        }
    }
}

impl<F: JoltField> SuffixPolynomial<F> for OperandPolynomial<F> {
    fn suffix_mle(&self, b: LookupBits) -> u128 {
        debug_assert!(b.len().is_even());
        debug_assert!(self.num_bound_vars.is_even());
        let (left, right) = b.uninterleave();
        match self.side {
            OperandSide::Left => u128::from(left),
            OperandSide::Right => u128::from(right),
        }
    }
}

impl<F: JoltField> PrefixPolynomial<F> for OperandPolynomial<F> {
    fn prefix_polynomial(
        &self,
        checkpoints: &PrefixCheckpoints<F>,
        chunk_len: usize,
        _phase: usize,
    ) -> CachedPolynomial<F> {
        debug_assert!(chunk_len.is_even());
        debug_assert!(self.num_bound_vars.is_even());

        let bound_value = match self.side {
            OperandSide::Left => checkpoints[Prefix::LeftOperand].unwrap_or(F::zero()),
            OperandSide::Right => checkpoints[Prefix::RightOperand].unwrap_or(F::zero()),
        };

        let evals: Vec<F> = (0..chunk_len.pow2())
            .map(|i| {
                let bits = LookupBits::new(i as u128, chunk_len);
                let (left, right) = bits.uninterleave();
                let operand_value = match self.side {
                    OperandSide::Left => F::from_u64(u64::from(left)),
                    OperandSide::Right => F::from_u64(u64::from(right)),
                };
                bound_value.mul_u128(1 << (chunk_len / 2)) + operand_value
            })
            .collect();

        CachedPolynomial::new(MultilinearPolynomial::from(evals), (chunk_len - 1).pow2())
    }
}

/// Polynomial that unmaps RAM addresses: k -> k*8 + start_address
#[derive(Allocative)]
pub struct UnmapRamAddressPolynomial<F: JoltField> {
    pub start_address: u64,
    int_poly: IdentityPolynomial<F>,
}

impl<F: JoltField> UnmapRamAddressPolynomial<F> {
    pub fn new(num_vars: usize, start_address: u64) -> Self {
        assert!(start_address > 8);
        UnmapRamAddressPolynomial {
            start_address,
            int_poly: IdentityPolynomial::new(num_vars),
        }
    }
}

impl<F: JoltField> PolynomialBinding<F> for UnmapRamAddressPolynomial<F> {
    fn is_bound(&self) -> bool {
        self.int_poly.is_bound()
    }

    fn bind(&mut self, r: F::Challenge, order: BindingOrder) {
        self.int_poly.bind(r, order);
    }

    fn bind_parallel(&mut self, r: F::Challenge, order: BindingOrder) {
        // Binding is constant time, no parallelism necessary
        self.bind(r, order);
    }

    fn final_sumcheck_claim(&self) -> F {
        self.int_poly.final_sumcheck_claim().mul_u64(8) + F::from_u64(self.start_address)
    }
}

impl<F: JoltField> PolynomialEvaluation<F> for UnmapRamAddressPolynomial<F> {
    fn evaluate<C>(&self, r: &[C]) -> F
    where
        C: Copy + Send + Sync + Into<F> + ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>,
    {
        self.int_poly.evaluate(r).mul_u64(8) + F::from_u64(self.start_address)
    }

    fn batch_evaluate<C>(_polys: &[&Self], _r: &[C]) -> Vec<F>
    where
        C: Copy + Send + Sync + Into<F> + ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>,
    {
        unimplemented!("Unused")
    }

    fn sumcheck_evals(&self, index: usize, degree: usize, order: BindingOrder) -> Vec<F> {
        let evals = self.int_poly.sumcheck_evals(index, degree, order);
        evals
            .into_iter()
            .map(|l| l.mul_u64(8) + F::from_u64(self.start_address))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use crate::poly::multilinear_polynomial::MultilinearPolynomial;
    use crate::poly::prefix_suffix::tests::prefix_suffix_decomposition_test;

    use super::*;
    use ark_bn254::Fr;
    use ark_ec::AdditiveGroup;
    use ark_ff::Field;
    use ark_std::test_rng;

    #[test]
    fn identity_poly() {
        const NUM_VARS: usize = 10;
        const DEGREE: usize = 3;

        let mut rng = test_rng();
        let mut identity_poly: IdentityPolynomial<Fr> = IdentityPolynomial::new(NUM_VARS);
        let mut reference_poly: MultilinearPolynomial<Fr> =
            MultilinearPolynomial::from((0..(1 << NUM_VARS)).map(|i| i as u32).collect::<Vec<_>>());

        for j in 0..reference_poly.len() / 2 {
            let identity_poly_evals =
                identity_poly.sumcheck_evals(j, DEGREE, BindingOrder::LowToHigh);
            let reference_poly_evals =
                reference_poly.sumcheck_evals(j, DEGREE, BindingOrder::LowToHigh);
            assert_eq!(identity_poly_evals, reference_poly_evals);
        }

        for _ in 0..NUM_VARS {
            let r = <Fr as JoltField>::Challenge::random(&mut rng);
            identity_poly.bind(r, BindingOrder::LowToHigh);
            reference_poly.bind(r, BindingOrder::LowToHigh);
            for j in 0..reference_poly.len() / 2 {
                let identity_poly_evals =
                    identity_poly.sumcheck_evals(j, DEGREE, BindingOrder::LowToHigh);
                let reference_poly_evals =
                    reference_poly.sumcheck_evals(j, DEGREE, BindingOrder::LowToHigh);
                assert_eq!(identity_poly_evals, reference_poly_evals);
            }
        }

        assert_eq!(
            identity_poly.final_sumcheck_claim(),
            reference_poly.final_sumcheck_claim()
        );
    }

    #[test]
    fn identity_poly_prefix_suffix_decomposition() {
        prefix_suffix_decomposition_test::<8, 2, 2, _>(
            IdentityPolynomial::new(8),
            Prefix::Identity,
        );
    }

    #[test]
    fn upper_half_all_ones_matches_predicate_on_hypercube() {
        const NUM_VARS: usize = 8;
        let poly = UpperHalfAllOnesPolynomial::new(NUM_VARS);

        for i in 0u128..(1 << NUM_VARS) {
            // `evaluate` reads its point most-significant-coordinate first.
            let point: Vec<Fr> = (0..NUM_VARS)
                .map(|j| {
                    if (i >> (NUM_VARS - 1 - j)) & 1 == 1 {
                        Fr::ONE
                    } else {
                        Fr::ZERO
                    }
                })
                .collect();
            let half = NUM_VARS / 2;
            let expected = if (i >> half) == (1 << half) - 1 {
                Fr::ONE
            } else {
                Fr::ZERO
            };
            assert_eq!(
                PolynomialEvaluation::<Fr>::evaluate(&poly, &point),
                expected,
                "index {i:#x}"
            );
        }
    }

    /// The predicate must read the *leading* half. `SUB(2^64-1, 0)` produces
    /// lookup index `0x1_FFFF_FFFF_FFFF_FFFF`, whose low limb is all ones and
    /// whose high limb is 1 — reading the trailing half would reject it.
    #[test]
    fn upper_half_all_ones_ignores_an_all_ones_low_limb() {
        const NUM_VARS: usize = 128;
        let poly = UpperHalfAllOnesPolynomial::new(NUM_VARS);
        let bit_point = |k: u128| -> Vec<Fr> {
            (0..NUM_VARS)
                .map(|j| {
                    if (k >> (NUM_VARS - 1 - j)) & 1 == 1 {
                        Fr::ONE
                    } else {
                        Fr::ZERO
                    }
                })
                .collect()
        };

        // SUB(x = 2^64-1, y = 0) => x + 2^64 - y.
        let sub_max = (u64::MAX as u128) + (1u128 << 64);
        assert_eq!(sub_max, 0x1_FFFF_FFFF_FFFF_FFFF);
        assert_eq!(
            PolynomialEvaluation::<Fr>::evaluate(&poly, &bit_point(sub_max)),
            Fr::ZERO,
            "honest SUB boundary must not be flagged"
        );

        // MUL(2^64-1, 2^64-1) — the largest honest index; high limb is 2^64-2.
        let mul_max = (u64::MAX as u128) * (u64::MAX as u128);
        assert_eq!(mul_max >> 64, (u64::MAX - 1) as u128);
        assert_eq!(
            PolynomialEvaluation::<Fr>::evaluate(&poly, &bit_point(mul_max)),
            Fr::ZERO,
            "honest MUL boundary must not be flagged"
        );

        // The fp128 aliases `k = r + p` for r in {0, 1, c-1}, c = 2^32 - 22537.
        let p = (1u128 << 127) + ((1u128 << 127) - (1u128 << 32) + 22537);
        let c = (1u128 << 32) - 22537;
        for r in [0u128, 1, c - 1] {
            let alias = p.wrapping_add(r);
            assert_eq!(alias >> 64, u64::MAX as u128, "alias r={r} shape");
            assert_eq!(
                PolynomialEvaluation::<Fr>::evaluate(&poly, &bit_point(alias)),
                Fr::ONE,
                "alias r={r} must be flagged"
            );
        }
    }

    #[test]
    fn upper_half_all_ones_prefix_suffix_decomposition() {
        // Chunk strictly inside each half.
        prefix_suffix_decomposition_test::<8, 2, 1, _>(
            UpperHalfAllOnesPolynomial::new(8),
            Prefix::UpperHalfAllOnes,
        );
        // Chunk boundary exactly on the midpoint — the production shape, where
        // `log_m` (8 or 16) divides the 64-bit half.
        prefix_suffix_decomposition_test::<8, 4, 1, _>(
            UpperHalfAllOnesPolynomial::new(8),
            Prefix::UpperHalfAllOnes,
        );
        // A single chunk straddling the midpoint, so `chunk_upper_bits` is a
        // proper fraction of the chunk.
        prefix_suffix_decomposition_test::<8, 8, 1, _>(
            UpperHalfAllOnesPolynomial::new(8),
            Prefix::UpperHalfAllOnes,
        );
    }

    #[test]
    fn operand_poly_prefix_suffix_decomposition() {
        prefix_suffix_decomposition_test::<8, 2, 2, _>(
            OperandPolynomial::new(8, OperandSide::Left),
            Prefix::LeftOperand,
        );
        prefix_suffix_decomposition_test::<8, 2, 2, _>(
            OperandPolynomial::new(8, OperandSide::Right),
            Prefix::RightOperand,
        );
    }

    #[test]
    fn unmap_address_poly_evaluate() {
        const NUM_VARS: usize = 10;
        const START_ADDRESS: u64 = 0x80000000;

        // Test that UnmapAddressPolynomial evaluates correctly on boolean hypercube
        let unmap_poly = UnmapRamAddressPolynomial::<Fr>::new(NUM_VARS, START_ADDRESS);

        // Test a few specific points
        // k=0 should map to start_address
        let point_0 = vec![Fr::ZERO; NUM_VARS];
        assert_eq!(unmap_poly.evaluate(&point_0), Fr::from(START_ADDRESS));

        // k=1 should map to start_address + 8
        let mut point_1 = vec![Fr::ZERO; NUM_VARS];
        point_1[NUM_VARS - 1] = Fr::ONE;
        assert_eq!(unmap_poly.evaluate(&point_1), Fr::from(START_ADDRESS + 8));

        // k=2 should map to start_address + 16
        let mut point_2 = vec![Fr::ZERO; NUM_VARS];
        point_2[NUM_VARS - 2] = Fr::ONE;
        assert_eq!(unmap_poly.evaluate(&point_2), Fr::from(START_ADDRESS + 16));

        // k=3 should map to start_address + 24
        let mut point_3 = vec![Fr::ZERO; NUM_VARS];
        point_3[NUM_VARS - 1] = Fr::ONE;
        point_3[NUM_VARS - 2] = Fr::ONE;
        assert_eq!(unmap_poly.evaluate(&point_3), Fr::from(START_ADDRESS + 24));
    }

    #[test]
    fn operand_poly_boolean_hypercube() {
        const NUM_VARS: usize = 8;

        let ro_poly = OperandPolynomial::<Fr>::new(NUM_VARS, OperandSide::Right);
        let lo_poly = OperandPolynomial::<Fr>::new(NUM_VARS, OperandSide::Left);

        // Test evaluation on boolean hypercube
        for i in 0..(1 << NUM_VARS) {
            let mut eval_point = vec![Fr::ZERO; NUM_VARS];
            for j in 0..NUM_VARS {
                if (i >> j) & 1 == 1 {
                    eval_point[j] = Fr::ONE;
                }
            }
            eval_point.reverse();

            // Use uninterleave_bits to match the polynomial's implementation
            let (left, right) = uninterleave_bits(i as u128);
            let expected_r = Fr::from_u64(right);
            let expected_l = Fr::from_u64(left);

            assert_eq!(
                ro_poly.evaluate(&eval_point),
                expected_r,
                "Boolean hypercube RIGHT OPERAND evaluation failed at index {i}"
            );
            assert_eq!(
                lo_poly.evaluate(&eval_point),
                expected_l,
                "Boolean hypercube LEFT OPERAND evaluation failed at index {i}"
            );
        }
    }

    #[test]
    fn operand_poly() {
        const NUM_VARS: usize = 8;

        let mut rng = test_rng();
        let mut ro_poly = OperandPolynomial::<Fr>::new(NUM_VARS, OperandSide::Right);
        let mut lo_poly = OperandPolynomial::<Fr>::new(NUM_VARS, OperandSide::Left);

        // Create reference polynomial with evaluations
        let (reference_evals_l, reference_evals_r): (Vec<Fr>, Vec<Fr>) = (0u128..(1 << NUM_VARS))
            .map(|i| {
                let (left, right) = uninterleave_bits(i);
                (Fr::from_u64(left), Fr::from_u64(right))
            })
            .collect();
        let mut reference_poly_r = MultilinearPolynomial::from(reference_evals_r);
        let mut reference_poly_l = MultilinearPolynomial::from(reference_evals_l);

        // Verify that both polynomials agree on the entire boolean hypercube
        for i in 0..(1 << NUM_VARS) {
            let mut eval_point = vec![Fr::ZERO; NUM_VARS];
            for j in 0..NUM_VARS {
                if (i >> j) & 1 == 1 {
                    eval_point[j] = Fr::ONE;
                }
            }
            let ro_poly = ro_poly.evaluate(&eval_point);
            let lo_poly = lo_poly.evaluate(&eval_point);
            let reference_r = reference_poly_r.evaluate(&eval_point);
            let reference_l = reference_poly_l.evaluate(&eval_point);
            assert_eq!(
                (ro_poly, lo_poly), (reference_r, reference_l),
                "Evaluation mismatch at index {i}:, operand_poly={ro_poly}, {lo_poly}, reference={reference_r}, {reference_l}"
            );
        }

        for round in 0..NUM_VARS {
            let num_evals = 1 << (NUM_VARS - round - 1);

            for i in 0..num_evals {
                let ro_poly = ro_poly.sumcheck_evals(i, 3, BindingOrder::HighToLow);
                let lo_poly = lo_poly.sumcheck_evals(i, 3, BindingOrder::HighToLow);
                let reference_r = reference_poly_r.sumcheck_evals(i, 3, BindingOrder::HighToLow);
                let reference_l = reference_poly_l.sumcheck_evals(i, 3, BindingOrder::HighToLow);

                assert_eq!(
                    (ro_poly, lo_poly),
                    (reference_r, reference_l),
                    "Round {round}, index {i}: sumcheck_evals mismatch"
                );
            }

            let r = <Fr as JoltField>::Challenge::random(&mut rng);
            ro_poly.bind(r, BindingOrder::HighToLow);
            lo_poly.bind(r, BindingOrder::HighToLow);
            reference_poly_r.bind(r, BindingOrder::HighToLow);
            reference_poly_l.bind(r, BindingOrder::HighToLow);
        }

        assert_eq!(
            ro_poly.final_sumcheck_claim(),
            reference_poly_r.final_sumcheck_claim(),
            "Final sumcheck claims don't match"
        );
        assert_eq!(
            lo_poly.final_sumcheck_claim(),
            reference_poly_l.final_sumcheck_claim(),
            "Final sumcheck claims don't match"
        );
    }

    #[test]
    fn unmap_address_poly() {
        const NUM_VARS: usize = 4;
        const START_ADDRESS: u64 = 0x80000000;

        let mut unmap_poly = UnmapRamAddressPolynomial::<Fr>::new(NUM_VARS, START_ADDRESS);

        let K = 1 << NUM_VARS;
        let unmap_evals: Vec<Fr> = (0..K)
            .map(|k| Fr::from((k as u64).wrapping_mul(8).wrapping_add(START_ADDRESS)))
            .collect();
        let mut reference_poly = MultilinearPolynomial::from(unmap_evals.clone());

        for round in 0..NUM_VARS {
            let num_evals = 1 << (NUM_VARS - round - 1);

            for i in 0..num_evals {
                let unmap_evals = unmap_poly.sumcheck_evals(i, 3, BindingOrder::LowToHigh);
                let reference_evals = reference_poly.sumcheck_evals(i, 3, BindingOrder::LowToHigh);

                assert_eq!(
                    unmap_evals, reference_evals,
                    "Round {round}, index {i}: sumcheck_evals mismatch",
                );
            }

            let r = <Fr as JoltField>::Challenge::from(0x12345678 + round as u128);
            unmap_poly.bind(r, BindingOrder::LowToHigh);
            reference_poly.bind(r, BindingOrder::LowToHigh);
        }

        assert_eq!(
            unmap_poly.final_sumcheck_claim(),
            reference_poly.final_sumcheck_claim(),
            "Final sumcheck claims don't match"
        );
    }
}
