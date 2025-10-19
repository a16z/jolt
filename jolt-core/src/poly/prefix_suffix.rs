#![allow(clippy::type_complexity)]

use std::ops::{Index, IndexMut};
use std::sync::{Arc, RwLock};

use allocative::Allocative;
use num_traits::Zero;
use once_cell::sync::OnceCell;
use rayon::prelude::*;
use strum::{EnumCount, IntoEnumIterator};
use strum_macros::{EnumCount as EnumCountMacro, EnumIter as EnumIterMacro};

use crate::field::{ChallengeFieldOps, FieldChallengeOps, JoltField};
use crate::poly::dense_mlpoly::DensePolynomial;
use crate::poly::multilinear_polynomial::{
    BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
};
use crate::utils::lookup_bits::LookupBits;
use crate::utils::math::Math;
use crate::utils::thread::unsafe_allocate_zero_vec;

#[repr(u8)]
#[derive(Clone, Copy, EnumIterMacro, EnumCountMacro)]
pub enum Prefix {
    RightOperand,
    LeftOperand,
    Identity,
}

pub type PrefixCheckpoints<F> = [Option<F>; Prefix::COUNT];

#[derive(Default, Allocative)]
pub struct PrefixRegistry<F: JoltField> {
    pub checkpoints: PrefixCheckpoints<F>,
    pub polys: [Option<Arc<RwLock<CachedPolynomial<F>>>>; Prefix::COUNT],
}

impl<F: JoltField> PrefixRegistry<F> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn update_checkpoints(&mut self) {
        Prefix::iter().for_each(|p| {
            self.checkpoints[p] = self[p]
                .as_ref()
                .map(|p| p.read().unwrap().final_sumcheck_claim());
            self[p] = None;
        });
    }
}

impl<F: JoltField> Index<Prefix> for PrefixRegistry<F> {
    type Output = Option<Arc<RwLock<CachedPolynomial<F>>>>;

    fn index(&self, index: Prefix) -> &Self::Output {
        &self.polys[index]
    }
}

impl<F: JoltField> IndexMut<Prefix> for PrefixRegistry<F> {
    fn index_mut(&mut self, index: Prefix) -> &mut Self::Output {
        &mut self.polys[index]
    }
}

impl<T> Index<Prefix> for [T; Prefix::COUNT] {
    type Output = T;

    fn index(&self, index: Prefix) -> &Self::Output {
        &self[index as usize]
    }
}

impl<T> IndexMut<Prefix> for [T; Prefix::COUNT] {
    fn index_mut(&mut self, index: Prefix) -> &mut Self::Output {
        &mut self[index as usize]
    }
}

#[derive(Allocative)]
pub struct CachedPolynomial<F: JoltField> {
    pub inner: MultilinearPolynomial<F>,
    #[allocative(skip)]
    pub sumcheck_evals_cache: Vec<OnceCell<(F, F)>>,
    pub bound_this_round: bool,
}

impl<F: JoltField> PolynomialEvaluation<F> for CachedPolynomial<F> {
    fn evaluate<C>(&self, x: &[C]) -> F
    where
        C: Copy + Send + Sync + Into<F> + ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>,
    {
        self.inner.evaluate(x)
    }

    fn batch_evaluate<C>(_polys: &[&Self], _r: &[C]) -> Vec<F>
    where
        C: Copy + Send + Sync + Into<F>,
        F: std::ops::Mul<C, Output = F> + std::ops::SubAssign<F>,
    {
        unimplemented!("Currently unused")
    }

    fn sumcheck_evals(&self, index: usize, degree: usize, order: BindingOrder) -> Vec<F> {
        self.inner.sumcheck_evals(index, degree, order)
    }
}

impl<F: JoltField> PolynomialBinding<F> for CachedPolynomial<F> {
    fn bind(&mut self, r: F::Challenge, order: BindingOrder) {
        if !self.bound_this_round {
            self.inner.bind(r, order);
            self.bound_this_round = true;
        }
    }

    fn bind_parallel(&mut self, r: F::Challenge, order: BindingOrder) {
        if !self.bound_this_round {
            self.inner.bind_parallel(r, order);
            self.bound_this_round = true;
        }
    }

    fn final_sumcheck_claim(&self) -> F {
        self.inner.final_sumcheck_claim()
    }

    fn is_bound(&self) -> bool {
        self.inner.is_bound()
    }
}

impl<F: JoltField> CachedPolynomial<F> {
    pub fn new(inner: MultilinearPolynomial<F>, cache_capacity: usize) -> Self {
        Self {
            inner,
            sumcheck_evals_cache: vec![OnceCell::new(); cache_capacity],
            bound_this_round: false,
        }
    }

    /// Returns evaluation at 0 and 2
    pub fn cached_sumcheck_evals(
        &self,
        index: usize,
        degree: usize,
        order: BindingOrder,
        use_cache: bool,
    ) -> (F, F) {
        assert!(degree == 2);
        if use_cache {
            *self.sumcheck_evals_cache[index].get_or_init(|| {
                let evals = self.sumcheck_evals(index, degree, order);
                (evals[0], evals[1])
            })
        } else {
            let evals = self.sumcheck_evals(index, degree, order);
            (evals[0], evals[1])
        }
    }

    pub fn clear_cache(&mut self, use_cache: bool) {
        if use_cache {
            self.sumcheck_evals_cache.iter_mut().for_each(|c| {
                c.take();
            });
        }
        self.bound_this_round = false;
    }
}

pub trait PrefixPolynomial<F: JoltField> {
    /// Computes P[i] polynomial
    /// Assumes binding_order to be HighToLow
    fn prefix_polynomial(
        &self,
        checkpoints: &PrefixCheckpoints<F>,
        chunk_len: usize,
        phase: usize,
    ) -> CachedPolynomial<F>;
}

pub trait SuffixPolynomial<F: JoltField> {
    fn suffix_mle(&self, b: LookupBits) -> u128;
}

pub trait PrefixSuffixPolynomial<F: JoltField, const ORDER: usize> {
    fn prefixes(
        &self,
        chunk_len: usize,
        phase: usize,
        prefix_registry: &mut PrefixRegistry<F>,
    ) -> [Option<Arc<RwLock<CachedPolynomial<F>>>>; ORDER];
    fn suffixes(&self) -> [Box<dyn SuffixPolynomial<F> + Sync>; ORDER];
}

// =========================
// Dynamic, field-valued variant (no 0/1 restriction)
// =========================

/// Registry for the dynamic, field-valued prefix–suffix decomposition.
///
/// Stores per-term prefix polynomials for the current phase (P) and the
/// per-term checkpoints (final prefix claims) to carry multiplicative
/// contributions across phases.
///
/// Notes:
/// - Two-phase only: used with `PrefixSuffixDecompositionFieldDyn`, which
///   splits variables into exactly two chunks (phase 0 and phase 1).
/// - No deduplication: identical per-term prefixes/suffixes (e.g., in EqPlusOne
///   for terms sharing the same side of the cutoff) are not deduplicated; each
///   term has its own entry.
#[derive(Default, Allocative)]
pub struct DynamicPrefixRegistry<F: JoltField> {
    pub checkpoints: Vec<Option<F>>, // per-term checkpoints
    pub polys: Vec<Option<Arc<RwLock<CachedPolynomial<F>>>>>,
}

impl<F: JoltField> DynamicPrefixRegistry<F> {
    /// Creates a new registry for `order` terms (one entry per term).
    pub fn new(order: usize) -> Self {
        Self {
            checkpoints: vec![None; order],
            polys: vec![None; order],
        }
    }

    /// Stores each term's final prefix claim into `checkpoints` and clears
    /// the cached prefix polynomials for the next phase.
    pub fn update_checkpoints(&mut self) {
        for (chkpt, poly) in self.checkpoints.iter_mut().zip(self.polys.iter_mut()) {
            if let Some(p) = poly.as_ref() {
                *chkpt = Some(p.read().unwrap().final_sumcheck_claim());
            }
            *poly = None;
        }
    }
}

/// Dynamic, field-valued prefix–suffix interface.
///
/// Implementors provide per-term prefix polynomials for the current phase and a
/// method to evaluate the per-term suffix factor on given suffix bits.
///
/// Conventions: two-phase only (phase ∈ {0,1}), with HighToLow (MSB→LSB) binding.
/// No term deduplication is performed by the framework.
pub trait PrefixSuffixPolynomialFieldDyn<F: JoltField>: Send + Sync {
    /// Number of terms k in the decomposition.
    fn order(&self) -> usize;
    /// Builds the per-term prefix polynomials for the current phase.
    /// - `chunk_len`: number of variables in this phase
    /// - `phase`: 0 for the first chunk, 1 for the second
    /// - `prefix_registry`: contains per-term checkpoints to scale prefixes
    fn prefixes(
        &self,
        chunk_len: usize,
        phase: usize,
        prefix_registry: &mut DynamicPrefixRegistry<F>,
    ) -> Vec<Option<Arc<RwLock<CachedPolynomial<F>>>>>;

    /// Evaluates the suffix factor for term `k` on given suffix bits.
    fn suffix_eval(&self, k: usize, suffix: LookupBits) -> F;
}

/// Two-phase dynamic prefix–suffix decomposition driver.
///
/// Orchestrates sumcheck over a function of the form
///   g(x) = ∑_i P_i(x_prefix) · Q_i(x_prefix),
/// where Q_i pre-aggregates the witness against the suffix factors.
///
/// Notes:
/// - Two-phase only: variables are split into exactly two chunks
///   (`first_chunk_len`, `second_chunk_len`).
/// - No deduplication of identical per-term prefixes/suffixes.
#[derive(Allocative)]
pub struct PrefixSuffixDecompositionFieldDyn<F: JoltField> {
    #[allocative(skip)]
    poly: Box<dyn PrefixSuffixPolynomialFieldDyn<F> + Send + Sync>,
    #[allocative(skip)]
    P: Vec<Option<Arc<RwLock<CachedPolynomial<F>>>>>, // per-term, current-phase prefixes
    Q: Vec<DensePolynomial<F>>, // per-term, current-phase Q polynomials
    // Two-phase control
    first_chunk_len: usize,
    second_chunk_len: usize,
    total_len: usize,
    phase: usize, // 0 or 1
    rounds_done: usize,
}

impl<F: JoltField> PrefixSuffixDecompositionFieldDyn<F> {
    /// Constructs a two-phase dynamic prefix–suffix decomposition.
    /// - `poly`: per-term prefix/suffix provider
    /// - `cutoff`: number of variables in phase 0 (MSB side)
    /// - `total_len`: total number of variables
    pub fn new(
        poly: Box<dyn PrefixSuffixPolynomialFieldDyn<F> + Send + Sync>,
        cutoff: usize,
        total_len: usize,
    ) -> Self {
        assert!(cutoff > 0 && cutoff < total_len);
        let order = poly.order();
        Self {
            poly,
            P: std::iter::repeat_with(|| None).take(order).collect(),
            Q: std::iter::repeat_with(DensePolynomial::default)
                .take(order)
                .collect(),
            first_chunk_len: cutoff,
            second_chunk_len: total_len - cutoff,
            total_len,
            phase: 0,
            rounds_done: 0,
        }
    }

    #[inline(always)]
    fn current_chunk_len(&self) -> usize {
        if self.phase == 0 {
            self.first_chunk_len
        } else {
            self.second_chunk_len
        }
    }

    /// Returns the number of variables remaining after the current phase's prefix.
    #[inline(always)]
    pub fn suffix_len(&self) -> usize {
        // Variables after the current chunk start
        let prefix_len = if self.phase == 0 {
            self.first_chunk_len
        } else {
            self.total_len
        };
        self.total_len - prefix_len
    }

    /// Builds per-term prefix polynomials P for the current phase.
    /// If checkpoints from a previous phase exist, prefixes are scaled accordingly.
    pub fn init_P(&mut self, prefix_registry: &mut DynamicPrefixRegistry<F>) {
        self.P = self
            .poly
            .prefixes(self.current_chunk_len(), self.phase, prefix_registry);
    }

    /// Builds per-term Q polynomials by streaming over the full domain once and
    /// accumulating u(prefix||suffix)·suffix_i(suffix) into Q_i[prefix].
    ///
    /// `u_evals` supplies u over the same `indices` domain; `lookup_bits` carries
    /// the bit decomposition to split into (prefix, suffix).
    #[tracing::instrument(skip_all, name = "PrefixSuffixFieldDyn::init_Q")]
    pub fn init_Q(&mut self, u_evals: &[F], indices: &[usize], lookup_bits: &[LookupBits]) {
        let poly_len = self.current_chunk_len().pow2();
        let order = self.poly.order();
        let suffix_len = self.suffix_len();

        let num_chunks = rayon::current_num_threads().next_power_of_two();
        let chunk_size = (indices.len() / num_chunks).max(1);

        let new_Q: Vec<Vec<F::Unreduced<9>>> = indices
            .par_chunks(chunk_size)
            .map(|chunk| {
                let mut chunk_result: Vec<Vec<F::Unreduced<9>>> = (0..order)
                    .map(|_| unsafe_allocate_zero_vec(poly_len))
                    .collect();

                for j in chunk {
                    let k = lookup_bits[*j];
                    let (prefix_bits, suffix_bits) = k.split(suffix_len);
                    for term_k in 0..order {
                        let t_val = self.poly.suffix_eval(term_k, suffix_bits);
                        if !t_val.is_zero() {
                            if let Some(u) = u_evals.get(*j) {
                                chunk_result[term_k][prefix_bits % poly_len] +=
                                    u.mul_unreduced::<9>(t_val);
                            }
                        }
                    }
                }

                chunk_result
            })
            .reduce(
                || {
                    (0..order)
                        .map(|_| unsafe_allocate_zero_vec(poly_len))
                        .collect()
                },
                |mut acc, new| {
                    for (acc_i, new_i) in acc.iter_mut().zip(new.iter()) {
                        for (acc_coeff, new_coeff) in acc_i.iter_mut().zip(new_i.iter()) {
                            *acc_coeff += new_coeff;
                        }
                    }
                    acc
                },
            );

        // Reduce to field and store
        self.Q = new_Q
            .into_iter()
            .map(|coeffs| {
                let mut reduced: Vec<F> = unsafe_allocate_zero_vec(poly_len);
                for (src, dst) in coeffs.into_iter().zip(reduced.iter_mut()) {
                    *dst = F::from_montgomery_reduce(src);
                }
                DensePolynomial::new(reduced)
            })
            .collect();
    }

    /// Fused initialization of Q for two decompositions in a single pass over `indices`.
    #[tracing::instrument(skip_all)]
    pub fn init_Q_dual(
        left: &mut PrefixSuffixDecompositionFieldDyn<F>,
        right: &mut PrefixSuffixDecompositionFieldDyn<F>,
        u_left: &[F],
        u_right: &[F],
        indices: &[usize],
        lookup_bits: &[LookupBits],
    ) {
        debug_assert_eq!(left.current_chunk_len(), right.current_chunk_len());
        debug_assert_eq!(left.total_len, right.total_len);
        debug_assert_eq!(left.phase, right.phase);

        let poly_len = left.current_chunk_len().pow2();
        let order_left = left.poly.order();
        let order_right = right.poly.order();
        let suffix_len = left.suffix_len();

        let num_chunks = rayon::current_num_threads().next_power_of_two();
        let chunk_size = (indices.len() / num_chunks).max(1);

        let (new_left, new_right): (Vec<Vec<F::Unreduced<9>>>, Vec<Vec<F::Unreduced<9>>>) = indices
            .par_chunks(chunk_size)
            .map(|chunk| {
                let mut chunk_left: Vec<Vec<F::Unreduced<9>>> = (0..order_left)
                    .map(|_| unsafe_allocate_zero_vec(poly_len))
                    .collect();
                let mut chunk_right: Vec<Vec<F::Unreduced<9>>> = (0..order_right)
                    .map(|_| unsafe_allocate_zero_vec(poly_len))
                    .collect();

                for j in chunk {
                    let k = lookup_bits[*j];
                    let (prefix_bits, suffix_bits) = k.split(suffix_len);

                    // Left terms
                    for term_k in 0..order_left {
                        let t_val = left.poly.suffix_eval(term_k, suffix_bits);
                        if !t_val.is_zero() {
                            if let Some(u) = u_left.get(*j) {
                                chunk_left[term_k][prefix_bits % poly_len] +=
                                    u.mul_unreduced::<9>(t_val);
                            }
                        }
                    }

                    // Right terms
                    for term_k in 0..order_right {
                        let t_val = right.poly.suffix_eval(term_k, suffix_bits);
                        if !t_val.is_zero() {
                            if let Some(u) = u_right.get(*j) {
                                chunk_right[term_k][prefix_bits % poly_len] +=
                                    u.mul_unreduced::<9>(t_val);
                            }
                        }
                    }
                }

                (chunk_left, chunk_right)
            })
            .reduce(
                || {
                    (
                        (0..order_left)
                            .map(|_| unsafe_allocate_zero_vec(poly_len))
                            .collect(),
                        (0..order_right)
                            .map(|_| unsafe_allocate_zero_vec(poly_len))
                            .collect(),
                    )
                },
                |(mut acc_l, mut acc_r), (new_l, new_r)| {
                    for (acc_i, new_i) in acc_l.iter_mut().zip(new_l.iter()) {
                        for (acc_coeff, new_coeff) in acc_i.iter_mut().zip(new_i.iter()) {
                            *acc_coeff += new_coeff;
                        }
                    }
                    for (acc_i, new_i) in acc_r.iter_mut().zip(new_r.iter()) {
                        for (acc_coeff, new_coeff) in acc_i.iter_mut().zip(new_i.iter()) {
                            *acc_coeff += new_coeff;
                        }
                    }
                    (acc_l, acc_r)
                },
            );

        // Reduce to field for left
        left.Q = new_left
            .into_iter()
            .map(|coeffs| {
                let mut reduced: Vec<F> = unsafe_allocate_zero_vec(poly_len);
                for (src, dst) in coeffs.into_iter().zip(reduced.iter_mut()) {
                    *dst = F::from_montgomery_reduce(src);
                }
                DensePolynomial::new(reduced)
            })
            .collect();
        // Reduce to field for right
        right.Q = new_right
            .into_iter()
            .map(|coeffs| {
                let mut reduced: Vec<F> = unsafe_allocate_zero_vec(poly_len);
                for (src, dst) in coeffs.into_iter().zip(reduced.iter_mut()) {
                    *dst = F::from_montgomery_reduce(src);
                }
                DensePolynomial::new(reduced)
            })
            .collect();
    }

    /// Returns the prover's univariate evaluations at t=0 and t=2 for `index`.
    ///
    /// For degree-2 products g(t) = P(t)·Q(t) with linear P, Q, we return
    /// g(0) = P(0)·Q(0) and g(2) = P(2)·(2·Q(1) − Q(0)).
    pub fn sumcheck_evals(&self, index: usize) -> (F, F) {
        let len = self.Q.first().map(|q| q.len()).unwrap_or(0);
        let (eval_0, eval_2_left, eval_2_right) = self
            .P
            .par_iter()
            .zip(self.Q.par_iter())
            .map(|(p, q)| {
                let p_evals = if let Some(p) = p {
                    // one for registry and one for self
                    let use_cache = Arc::strong_count(p) > 2;
                    let p = p.read().unwrap();
                    let p_evals =
                        p.cached_sumcheck_evals(index, 2, BindingOrder::HighToLow, use_cache);
                    drop(p);
                    p_evals
                } else {
                    // Prefixes are just constant 1, 1 if it's none
                    (F::one(), F::one())
                };
                let q_left = q[index];
                let q_right = q[index + len / 2];
                // We need g(0) = P(0)*Q(0). For t=2, use g(2) = P(2) * (2*Q(1) - Q(0))
                // Accumulate: eval_0_sum = P(0)*Q(0), left_2_sum = P(2)*Q(0), right_2_sum = P(2)*Q(1)
                (
                    p_evals.0.mul_unreduced::<9>(q_left),  // eval_0_sum = P(0) * Q(0)
                    p_evals.1.mul_unreduced::<9>(q_left),  // left_2_sum = P(2) * Q(0)
                    p_evals.1.mul_unreduced::<9>(q_right), // right_2_sum = P(2) * Q(1)
                )
            })
            .reduce(
                || {
                    (
                        F::Unreduced::<9>::zero(),
                        F::Unreduced::<9>::zero(),
                        F::Unreduced::<9>::zero(),
                    )
                },
                |running, new| (running.0 + new.0, running.1 + new.1, running.2 + new.2),
            );
        let eval_0 = F::from_montgomery_reduce(eval_0);
        let eval_2_right = F::from_montgomery_reduce(eval_2_right);
        let eval_2_left = F::from_montgomery_reduce(eval_2_left);
        // g(2) = P(2) * (2*Q(1) - Q(0))
        (eval_0, eval_2_right + eval_2_right - eval_2_left)
    }

    /// Binds the current variable (MSB-first) with challenge `r`, reducing the
    /// dimension of both P and Q. After `first_chunk_len` binds, advances to phase 1.
    pub fn bind(&mut self, r: F::Challenge) {
        self.P.par_iter().for_each(|p| {
            if let Some(p) = p {
                // one for registry and one for self
                let use_cache = Arc::strong_count(p) > 2;
                let mut p = p.write().unwrap();
                p.bind_parallel(r, BindingOrder::HighToLow);
                p.clear_cache(use_cache);
            }
        });
        self.Q.par_iter_mut().for_each(|poly| {
            poly.bind_parallel(r, BindingOrder::HighToLow);
        });
        self.rounds_done += 1;
        // Transition to phase 1 after binding first chunk
        if self.phase == 0 && self.rounds_done == self.first_chunk_len {
            self.phase = 1;
        }
    }

    /// Sums per-term final prefix claims; used for carrying state or verification
    /// outside this driver. Does not include Q.
    pub fn final_sumcheck_claim(&self) -> F {
        self.P
            .par_iter()
            .map(|p| {
                if let Some(p) = p {
                    let p = p.read().unwrap();
                    p.final_sumcheck_claim()
                } else {
                    F::one()
                }
            })
            .sum()
    }

    /// Current length of Q arrays (useful for selecting indices per round).
    pub fn Q_len(&self) -> usize {
        self.Q.first().map(|q| q.len()).unwrap_or(0)
    }
}

#[derive(Allocative)]
pub struct PrefixSuffixDecomposition<F: JoltField, const ORDER: usize> {
    #[allocative(skip)]
    poly: Box<dyn PrefixSuffixPolynomial<F, ORDER> + Send + Sync>,
    #[allocative(skip)]
    P: [Option<Arc<RwLock<CachedPolynomial<F>>>>; ORDER],
    Q: [DensePolynomial<F>; ORDER],
    chunk_len: usize,
    total_len: usize,
    phase: usize,
    round: usize,
}

impl<F: JoltField, const ORDER: usize> PrefixSuffixDecomposition<F, ORDER> {
    pub fn new(
        poly: Box<dyn PrefixSuffixPolynomial<F, ORDER> + Send + Sync>,
        chunk_len: usize,
        total_len: usize,
    ) -> Self {
        assert!(
            total_len.is_multiple_of(chunk_len),
            "total_len must be a multiple of chunk_len"
        );
        Self {
            poly,
            P: std::array::from_fn(|_| None),
            Q: Self::alloc_Q(chunk_len.pow2()),
            chunk_len,
            total_len,
            phase: 0,
            round: 0,
        }
    }

    #[inline(always)]
    pub fn suffix_len(&self) -> usize {
        let total_chunks = self.total_len / self.chunk_len;
        let suffix_chunks = total_chunks - self.phase - 1;
        suffix_chunks * self.chunk_len
    }

    pub fn prefix_len(&self) -> usize {
        (self.phase + 1) * self.chunk_len
    }

    fn alloc_Q(m: usize) -> [DensePolynomial<F>; ORDER] {
        rayon::iter::repeat_n(0, ORDER)
            .map(|_| DensePolynomial::new(unsafe_allocate_zero_vec(m)))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    }

    /// P array is defined as P[x] = prefix(x)
    /// Read more about prefix-suffix argument in Appendix A of the paper
    /// https://eprint.iacr.org/2025/611.pdf
    #[tracing::instrument(skip_all, name = "PrefixSuffix::init_P")]
    pub fn init_P(&mut self, prefix_registry: &mut PrefixRegistry<F>) {
        self.P = self
            .poly
            .prefixes(self.chunk_len, self.phase, prefix_registry);
    }

    /// Q array is defined as Q[x] = \sum_{y \in {0, 1}^m} u(x || y) * suffix(y)
    /// Read more about prefix-suffix argument in Appendix A of the paper
    /// https://eprint.iacr.org/2025/611.pdf
    #[tracing::instrument(skip_all, name = "PrefixSuffix::init_Q")]
    pub fn init_Q(&mut self, u_evals: &[F], indices: &[usize], lookup_bits: &[LookupBits]) {
        let poly_len = self.chunk_len.pow2();
        let suffix_len = self.suffix_len();
        let suffixes = self.poly.suffixes();

        let num_chunks = rayon::current_num_threads().next_power_of_two();
        let chunk_size = (indices.len() / num_chunks).max(1);

        let new_Q = indices
            .par_chunks(chunk_size)
            .map(|chunk| {
                let mut chunk_result: [Vec<F::Unreduced<7>>; ORDER] =
                    std::array::from_fn(|_| unsafe_allocate_zero_vec(poly_len));

                for j in chunk {
                    let k = lookup_bits[*j];
                    let (prefix_bits, suffix_bits) = k.split(suffix_len);
                    for (suffix, result) in suffixes.iter().zip(chunk_result.iter_mut()) {
                        let t = suffix.suffix_mle(suffix_bits);
                        if t != 0 {
                            if let Some(u) = u_evals.get(*j) {
                                result[prefix_bits % poly_len] += u.mul_u128_unreduced(t);
                            }
                        }
                    }
                }

                chunk_result
            })
            .reduce(
                || std::array::from_fn(|_| unsafe_allocate_zero_vec(poly_len)),
                |mut acc, new| {
                    for (acc_i, new_i) in acc.iter_mut().zip(new.iter()) {
                        for (acc_coeff, new_coeff) in acc_i.iter_mut().zip(new_i.iter()) {
                            *acc_coeff += new_coeff;
                        }
                    }
                    acc
                },
            );

        let mut reduced_Q: [Vec<F>; ORDER] =
            std::array::from_fn(|_| unsafe_allocate_zero_vec(poly_len));
        for (q, reduced_q) in new_Q.iter().zip(reduced_Q.iter_mut()) {
            for (q_coeff, reduced_q_coeff) in q.iter().zip(reduced_q.iter_mut()) {
                *reduced_q_coeff = F::from_barrett_reduce(*q_coeff);
            }
        }

        self.Q = std::array::from_fn(|i| DensePolynomial::new(std::mem::take(&mut reduced_Q[i])));
    }

    /// Initialize Q for two PrefixSuffixDecomposition instances in a single pass over indices
    /// Q array is defined as Q[x] = \sum_{y \in {0, 1}^m} u(x || y) * suffix(y)
    /// Read more about prefix-suffix argument in Appendix A of the paper
    /// https://eprint.iacr.org/2025/611.pdf
    #[tracing::instrument(skip_all)]
    pub fn init_Q_dual(
        left: &mut PrefixSuffixDecomposition<F, ORDER>,
        right: &mut PrefixSuffixDecomposition<F, ORDER>,
        u_evals: &[F],
        indices: &[usize],
        lookup_bits: &[LookupBits],
    ) {
        debug_assert_eq!(left.chunk_len, right.chunk_len);
        debug_assert_eq!(left.total_len, right.total_len);
        debug_assert_eq!(left.phase, right.phase);

        let poly_len = left.chunk_len.pow2();
        let suffix_len = left.suffix_len();
        let suffixes_left = left.poly.suffixes();
        let suffixes_right = right.poly.suffixes();

        let num_chunks = rayon::current_num_threads().next_power_of_two();
        let chunk_size = (indices.len() / num_chunks).max(1);

        let (new_left, new_right) = indices
            .par_chunks(chunk_size)
            .map(|chunk| {
                let mut chunk_left: [Vec<F::Unreduced<7>>; ORDER] =
                    std::array::from_fn(|_| unsafe_allocate_zero_vec(poly_len));
                let mut chunk_right: [Vec<F::Unreduced<7>>; ORDER] =
                    std::array::from_fn(|_| unsafe_allocate_zero_vec(poly_len));

                for j in chunk {
                    let k = lookup_bits[*j];
                    let (prefix_bits, suffix_bits) = k.split(suffix_len);

                    // Left accumulators
                    for (suffix, result) in suffixes_left.iter().zip(chunk_left.iter_mut()) {
                        let t = suffix.suffix_mle(suffix_bits);
                        if t != 0 {
                            if let Some(u) = u_evals.get(*j) {
                                result[prefix_bits % poly_len] += u.mul_u128_unreduced(t);
                            }
                        }
                    }

                    // Right accumulators
                    for (suffix, result) in suffixes_right.iter().zip(chunk_right.iter_mut()) {
                        let t = suffix.suffix_mle(suffix_bits);
                        if t != 0 {
                            if let Some(u) = u_evals.get(*j) {
                                result[prefix_bits % poly_len] += u.mul_u128_unreduced(t);
                            }
                        }
                    }
                }

                (chunk_left, chunk_right)
            })
            .reduce(
                || {
                    (
                        std::array::from_fn(|_| unsafe_allocate_zero_vec(poly_len)),
                        std::array::from_fn(|_| unsafe_allocate_zero_vec(poly_len)),
                    )
                },
                |(mut acc_l, mut acc_r), (new_l, new_r)| {
                    for (acc_i, new_i) in acc_l.iter_mut().zip(new_l.iter()) {
                        for (acc_coeff, new_coeff) in acc_i.iter_mut().zip(new_i.iter()) {
                            *acc_coeff += new_coeff;
                        }
                    }
                    for (acc_i, new_i) in acc_r.iter_mut().zip(new_r.iter()) {
                        for (acc_coeff, new_coeff) in acc_i.iter_mut().zip(new_i.iter()) {
                            *acc_coeff += new_coeff;
                        }
                    }
                    (acc_l, acc_r)
                },
            );

        // Reduce to field for left
        let mut reduced_left: [Vec<F>; ORDER] =
            std::array::from_fn(|_| unsafe_allocate_zero_vec(poly_len));
        for (q, reduced_q) in new_left.iter().zip(reduced_left.iter_mut()) {
            for (q_coeff, reduced_q_coeff) in q.iter().zip(reduced_q.iter_mut()) {
                *reduced_q_coeff = F::from_barrett_reduce(*q_coeff);
            }
        }

        // Reduce to field for right
        let mut reduced_right: [Vec<F>; ORDER] =
            std::array::from_fn(|_| unsafe_allocate_zero_vec(poly_len));
        for (q, reduced_q) in new_right.iter().zip(reduced_right.iter_mut()) {
            for (q_coeff, reduced_q_coeff) in q.iter().zip(reduced_q.iter_mut()) {
                *reduced_q_coeff = F::from_barrett_reduce(*q_coeff);
            }
        }

        left.Q =
            std::array::from_fn(|i| DensePolynomial::new(std::mem::take(&mut reduced_left[i])));
        right.Q =
            std::array::from_fn(|i| DensePolynomial::new(std::mem::take(&mut reduced_right[i])));
    }

    /// Returns evaluation at 0 and at 2 at index
    pub fn sumcheck_evals(&self, index: usize) -> (F, F) {
        let len = self.Q[0].len();
        let (eval_0, eval_2_left, eval_2_right) = self
            .P
            .par_iter()
            .zip(self.Q.par_iter())
            .map(|(p, q)| {
                let p_evals = if let Some(p) = p {
                    // one for registry and one for self
                    let use_cache = Arc::strong_count(p) > 2;
                    let p = p.read().unwrap();
                    let p_evals =
                        p.cached_sumcheck_evals(index, 2, BindingOrder::HighToLow, use_cache);
                    drop(p);
                    p_evals
                } else {
                    // Prefixes are just constant 1, 1 if it's none
                    (F::one(), F::one())
                };
                let q_left = q[index];
                let q_right = q[index + len / 2];
                (
                    p_evals.0.mul_unreduced::<9>(q_left),  // prefix(0) * suffix(0)
                    p_evals.1.mul_unreduced::<9>(q_left),  // prefix(2) * suffix(0)
                    p_evals.1.mul_unreduced::<9>(q_right), // prefix(2) * suffix(1)
                )
            })
            .reduce(
                || {
                    (
                        F::Unreduced::<9>::zero(),
                        F::Unreduced::<9>::zero(),
                        F::Unreduced::<9>::zero(),
                    )
                },
                |running, new| (running.0 + new.0, running.1 + new.1, running.2 + new.2),
            );
        let eval_0 = F::from_montgomery_reduce(eval_0);
        let eval_2_right = F::from_montgomery_reduce(eval_2_right);
        let eval_2_left = F::from_montgomery_reduce(eval_2_left);
        (eval_0, eval_2_right + eval_2_right - eval_2_left)
    }

    pub fn bind(&mut self, r: F::Challenge) {
        self.P.par_iter().for_each(|p| {
            if let Some(p) = p {
                let mut p = p.write().unwrap();
                p.bind_parallel(r, BindingOrder::HighToLow);
            }
        });
        self.Q.par_iter_mut().for_each(|poly| {
            poly.bind_parallel(r, BindingOrder::HighToLow);
        });
        self.next_round();
    }

    pub fn final_sumcheck_claim(&self) -> F {
        self.P
            .par_iter()
            .zip(self.poly.suffixes().par_iter())
            .map(|(p, suffix)| {
                let suff = suffix.suffix_mle(LookupBits::new(0, 0));
                if suff == 0 {
                    return F::zero();
                }
                if let Some(p) = p {
                    let p = p.read().unwrap();
                    p.final_sumcheck_claim().mul_u128(suff)
                } else {
                    F::from_u128(suff)
                }
            })
            .sum()
    }

    fn next_round(&mut self) {
        self.P.par_iter().for_each(|p| {
            if let Some(p) = p {
                // one for registry and one for self
                let use_cache = Arc::strong_count(p) > 2;
                let mut p = p.write().unwrap();
                p.clear_cache(use_cache);
            }
        });
        self.round += 1;
        if self.round.is_multiple_of(self.chunk_len) {
            self.phase += 1;
        }
    }

    pub fn Q_len(&self) -> usize {
        self.Q[0].len()
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use ark_bn254::Fr;
    use ark_ff::{AdditiveGroup, Field};
    use ark_std::test_rng;

    pub fn prefix_suffix_decomposition_test<
        const NUM_VARS: usize,
        const PREFIX_LEN: usize,
        const ORDER: usize,
        P: PolynomialEvaluation<Fr>
            + PrefixSuffixPolynomial<Fr, ORDER>
            + Clone
            + Send
            + Sync
            + 'static,
    >(
        poly: P,
        prefix_registry_index: Prefix,
    ) {
        let SUFFIX_LEN: usize = NUM_VARS - PREFIX_LEN;

        let mut rng = test_rng();
        let mut prefix_registry = PrefixRegistry::new();
        let mut ps = PrefixSuffixDecomposition::new(Box::new(poly.clone()), PREFIX_LEN, NUM_VARS);

        let indices: Vec<_> = (0..(1 << NUM_VARS)).collect();
        let lookup_bits = (0..(1 << NUM_VARS))
            .map(|i| LookupBits::new(i, NUM_VARS))
            .collect::<Vec<_>>();

        let mut rr = vec![];
        for phase in 0..(NUM_VARS / PREFIX_LEN) {
            ps.init_P(&mut prefix_registry);
            ps.init_Q(
                &(0..(1 << (NUM_VARS - PREFIX_LEN * phase)))
                    .map(|_| Fr::ONE)
                    .collect::<Vec<_>>(),
                &indices,
                &lookup_bits,
            );

            for round in (0..PREFIX_LEN).rev() {
                for b in 0..round.pow2() {
                    let eval = ps.sumcheck_evals(b);

                    let eval_point = rr
                        .iter()
                        .cloned()
                        .chain(std::iter::once(Fr::ZERO))
                        .chain(
                            std::iter::repeat_n(b, round)
                                .enumerate()
                                .rev()
                                .map(|(i, b)| if (b >> i) & 1 == 1 { Fr::ONE } else { Fr::ZERO }),
                        )
                        .collect::<Vec<Fr>>();
                    let suffix_len = SUFFIX_LEN - phase * PREFIX_LEN;
                    let direct_eval: Fr = (0..(1 << suffix_len))
                        .map(|i| {
                            let mut eval_point = eval_point.clone();
                            for j in (0..suffix_len).rev() {
                                if (i >> j) & 1 == 1 {
                                    eval_point.push(Fr::ONE);
                                } else {
                                    eval_point.push(Fr::ZERO);
                                }
                            }
                            poly.evaluate(&eval_point)
                        })
                        .sum();

                    assert_eq!(direct_eval, eval.0);

                    let eval_point = rr
                        .iter()
                        .cloned()
                        .chain(std::iter::once(Fr::ONE + Fr::ONE))
                        .chain(
                            std::iter::repeat_n(b, round)
                                .enumerate()
                                .rev()
                                .map(|(i, b)| if (b >> i) & 1 == 1 { Fr::ONE } else { Fr::ZERO }),
                        )
                        .collect::<Vec<Fr>>();
                    let direct_eval: Fr = (0..(1 << suffix_len))
                        .map(|i| {
                            let mut eval_point = eval_point.clone();
                            for j in (0..suffix_len).rev() {
                                if (i >> j) & 1 == 1 {
                                    eval_point.push(Fr::ONE);
                                } else {
                                    eval_point.push(Fr::ZERO);
                                }
                            }
                            poly.evaluate(&eval_point)
                        })
                        .sum();

                    assert_eq!(direct_eval, eval.1);
                }
                let r = <Fr as JoltField>::Challenge::random(&mut rng);
                rr.push(r.into());
                ps.bind(r);
            }

            prefix_registry.update_checkpoints();
        }
        assert_eq!(
            prefix_registry.checkpoints[prefix_registry_index],
            Some(poly.evaluate(&rr))
        )
    }
}
