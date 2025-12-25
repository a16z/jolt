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

/// Array storing prefix polynomial evaluations, indexed by Prefix enum variants.
pub type PrefixCheckpoints<F> = [Option<F>; Prefix::COUNT];

#[derive(Default, Allocative)]
/// Registry storing prefix polynomial evaluations at r_prefix for all prefix types.
pub struct PrefixRegistry<F: JoltField> {
    /// checkpoints[i] = P_i(r_prefix) where P_i is the i-th prefix polynomial evaluation.
    pub checkpoints: PrefixCheckpoints<F>,
    /// Cached polynomial representations for potential reuse across decompositions.
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

#[derive(Allocative)]
/// Decomposes a polynomial f(x) = Σ_i P_i(x_prefix)·Q_i(x_suffix) for efficient sumcheck evaluation.
pub struct PrefixSuffixDecomposition<F: JoltField, const ORDER: usize> {
    #[allocative(skip)]
    /// Original polynomial to decompose (e.g., OperandPolynomial, IdentityPolynomial).
    poly: Box<dyn PrefixSuffixPolynomial<F, ORDER> + Send + Sync>,
    #[allocative(skip)]
    /// P[i] = prefix polynomial i, computed from registry and cached for reuse.
    P: [Option<Arc<RwLock<CachedPolynomial<F>>>>; ORDER],
    /// Q[i] = suffix polynomial i, recomputed each phase from trace indices.
    Q: [DensePolynomial<F>; ORDER],
    /// Number of variables per chunk (typically LOG_M).
    chunk_len: usize,
    /// Total number of variables (typically LOG_K).
    total_len: usize,
    /// Current phase in multi-phase decomposition.
    phase: usize,
    /// Current round within phase.
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

        // Accumulate in row-major for write locality: rows are r_index in [0, poly_len)
        let new_Q_rows: Vec<[F::Unreduced<7>; ORDER]> = indices
            .par_chunks(chunk_size)
            .fold(
                || vec![[F::Unreduced::<7>::zero(); ORDER]; poly_len],
                |mut acc, chunk| {
                    for j in chunk {
                        let k = lookup_bits[*j];
                        let (prefix_bits, suffix_bits) = k.split(suffix_len);
                        let r_index: usize = (u128::from(&prefix_bits) as usize) & (poly_len - 1);
                        if let Some(u) = u_evals.get(*j) {
                            for (s_idx, suffix) in suffixes.iter().enumerate() {
                                let t = suffix.suffix_mle(suffix_bits);
                                if t != 0 {
                                    acc[r_index][s_idx] += u.mul_u128_unreduced(t);
                                }
                            }
                        }
                    }
                    acc
                },
            )
            .reduce(
                || vec![[F::Unreduced::<7>::zero(); ORDER]; poly_len],
                |mut acc, new| {
                    for (acc_row, new_row) in acc.iter_mut().zip(new.iter()) {
                        for s in 0..ORDER {
                            acc_row[s] += new_row[s];
                        }
                    }
                    acc
                },
            );

        // Transpose rows back to suffix-major and reduce to field
        let mut reduced_Q: [Vec<F>; ORDER] =
            std::array::from_fn(|_| unsafe_allocate_zero_vec(poly_len));
        for r_idx in 0..poly_len {
            for s in 0..ORDER {
                reduced_Q[s][r_idx] = F::from_barrett_reduce(new_Q_rows[r_idx][s]);
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

        #[allow(clippy::type_complexity)]
        let (new_left_rows, new_right_rows): (
            Vec<[F::Unreduced<7>; ORDER]>,
            Vec<[F::Unreduced<7>; ORDER]>,
        ) = indices
            .par_chunks(chunk_size)
            .fold(
                || {
                    (
                        vec![[F::Unreduced::<7>::zero(); ORDER]; poly_len],
                        vec![[F::Unreduced::<7>::zero(); ORDER]; poly_len],
                    )
                },
                |(mut acc_l, mut acc_r), chunk| {
                    for j in chunk {
                        let k = lookup_bits[*j];
                        let (prefix_bits, suffix_bits) = k.split(suffix_len);
                        let r_index: usize = (u128::from(&prefix_bits) as usize) & (poly_len - 1);
                        if let Some(u) = u_evals.get(*j) {
                            // Left
                            for (s_idx, suffix) in suffixes_left.iter().enumerate() {
                                let t = suffix.suffix_mle(suffix_bits);
                                if t != 0 {
                                    acc_l[r_index][s_idx] += u.mul_u128_unreduced(t);
                                }
                            }
                            // Right
                            for (s_idx, suffix) in suffixes_right.iter().enumerate() {
                                let t = suffix.suffix_mle(suffix_bits);
                                if t != 0 {
                                    acc_r[r_index][s_idx] += u.mul_u128_unreduced(t);
                                }
                            }
                        }
                    }
                    (acc_l, acc_r)
                },
            )
            .reduce(
                || {
                    (
                        vec![[F::Unreduced::<7>::zero(); ORDER]; poly_len],
                        vec![[F::Unreduced::<7>::zero(); ORDER]; poly_len],
                    )
                },
                |(mut acc_l, mut acc_r), (new_l, new_r)| {
                    for (acc_row, new_row) in acc_l.iter_mut().zip(new_l.iter()) {
                        for s in 0..ORDER {
                            acc_row[s] += new_row[s];
                        }
                    }
                    for (acc_row, new_row) in acc_r.iter_mut().zip(new_r.iter()) {
                        for s in 0..ORDER {
                            acc_row[s] += new_row[s];
                        }
                    }
                    (acc_l, acc_r)
                },
            );

        // Reduce to field for left and right (transpose rows to suffix-major)
        let mut reduced_left: [Vec<F>; ORDER] =
            std::array::from_fn(|_| unsafe_allocate_zero_vec(poly_len));
        for r_idx in 0..poly_len {
            for s in 0..ORDER {
                reduced_left[s][r_idx] = F::from_barrett_reduce(new_left_rows[r_idx][s]);
            }
        }
        let mut reduced_right: [Vec<F>; ORDER] =
            std::array::from_fn(|_| unsafe_allocate_zero_vec(poly_len));
        for r_idx in 0..poly_len {
            for s in 0..ORDER {
                reduced_right[s][r_idx] = F::from_barrett_reduce(new_right_rows[r_idx][s]);
            }
        }

        left.Q =
            std::array::from_fn(|i| DensePolynomial::new(std::mem::take(&mut reduced_left[i])));
        right.Q =
            std::array::from_fn(|i| DensePolynomial::new(std::mem::take(&mut reduced_right[i])));
    }

    // NOTE: RAF-specific fused Q initialization lives in a specialization below:
    // `impl<F: JoltField> PrefixSuffixDecomposition<F, 2>`.

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

impl<F: JoltField> PrefixSuffixDecomposition<F, 2> {
    /// Specialized fused initializer for the RAF sumcheck Q polynomials.
    ///
    /// This replaces `init_Q_dual(left_operand, right_operand, ...)` plus
    /// `identity.init_Q(...)` with a single scan over cycles.
    ///
    /// Assumptions (true for `ReadRafSumcheckProver`):
    /// - `left` and `right` are `OperandPolynomial` prefix-suffix decompositions.
    ///   Their suffixes are `[ShiftHalfSuffixPolynomial, OperandPolynomial(side)]`.
    /// - `identity` is an `IdentityPolynomial` prefix-suffix decomposition.
    ///   Its suffixes are `[ShiftSuffixPolynomial, IdentityPolynomial]`.
    ///
    /// Under these assumptions we can avoid dynamic dispatch and compute:
    /// - Shift terms once per phase (they are constants depending only on `suffix_len`)
    /// - Operand `uninterleave()` once per cycle and reuse for both left/right
    #[tracing::instrument(skip_all, name = "PrefixSuffix::init_Q_raf")]
    pub fn init_Q_raf(
        left: &mut PrefixSuffixDecomposition<F, 2>,
        right: &mut PrefixSuffixDecomposition<F, 2>,
        identity: &mut PrefixSuffixDecomposition<F, 2>,
        u_evals: &[F],
        lookup_bits: &[LookupBits],
        is_interleaved_operands: &[bool],
    ) {
        debug_assert_eq!(left.chunk_len, right.chunk_len);
        debug_assert_eq!(left.total_len, right.total_len);
        debug_assert_eq!(left.phase, right.phase);
        debug_assert_eq!(left.chunk_len, identity.chunk_len);
        debug_assert_eq!(left.total_len, identity.total_len);
        debug_assert_eq!(left.phase, identity.phase);
        debug_assert_eq!(lookup_bits.len(), u_evals.len());
        debug_assert_eq!(lookup_bits.len(), is_interleaved_operands.len());

        let poly_len = left.chunk_len.pow2();
        let suffix_len = left.suffix_len();
        debug_assert!(suffix_len % 2 == 0);

        // Constants for this phase:
        // - Operand path: ShiftHalfSuffixPolynomial(b) = 2^{|b|/2}
        // - Identity path: ShiftSuffixPolynomial(b) = 2^{|b|}
        let shift_half: u128 = 1u128 << (suffix_len / 2);
        let shift_full: u128 = 1u128 << suffix_len;

        let num_chunks = rayon::current_num_threads().next_power_of_two();
        let chunk_size = (lookup_bits.len() / num_chunks).max(1);

        type U7<F> = <F as JoltField>::Unreduced<7>;

        #[allow(clippy::type_complexity)]
        let (rows_shift_half, rows_left, rows_right, rows_shift_full, rows_identity): (
            Vec<U7<F>>,
            Vec<U7<F>>,
            Vec<U7<F>>,
            Vec<U7<F>>,
            Vec<U7<F>>,
        ) = lookup_bits
            .par_chunks(chunk_size)
            .zip(u_evals.par_chunks(chunk_size))
            .zip(is_interleaved_operands.par_chunks(chunk_size))
            .fold(
                || {
                    (
                        vec![U7::<F>::zero(); poly_len], // operand shift-half (shared by left/right)
                        vec![U7::<F>::zero(); poly_len], // operand left value
                        vec![U7::<F>::zero(); poly_len], // operand right value
                        vec![U7::<F>::zero(); poly_len], // identity shift-full
                        vec![U7::<F>::zero(); poly_len], // identity value
                    )
                },
                |(mut acc_sh, mut acc_l, mut acc_r, mut acc_sf, mut acc_id),
                 ((k_chunk, u_chunk), inter_chunk)| {
                    debug_assert_eq!(k_chunk.len(), u_chunk.len());
                    debug_assert_eq!(k_chunk.len(), inter_chunk.len());

                    for ((k, u), is_interleaved) in
                        k_chunk.iter().zip(u_chunk.iter()).zip(inter_chunk.iter())
                    {
                        let (prefix_bits, suffix_bits) = k.split(suffix_len);
                        let r_index: usize = prefix_bits & (poly_len - 1);

                        if *is_interleaved {
                            // Operand path (left/right)
                            acc_sh[r_index] += u.mul_u128_unreduced(shift_half);

                            let (lo_bits, ro_bits) = suffix_bits.uninterleave();
                            let lo: u128 = lo_bits.into();
                            if lo != 0 {
                                acc_l[r_index] += u.mul_u128_unreduced(lo);
                            }
                            let ro: u128 = ro_bits.into();
                            if ro != 0 {
                                acc_r[r_index] += u.mul_u128_unreduced(ro);
                            }
                        } else {
                            // Identity path
                            acc_sf[r_index] += u.mul_u128_unreduced(shift_full);
                            let id: u128 = suffix_bits.into();
                            if id != 0 {
                                acc_id[r_index] += u.mul_u128_unreduced(id);
                            }
                        }
                    }

                    (acc_sh, acc_l, acc_r, acc_sf, acc_id)
                },
            )
            .reduce(
                || {
                    (
                        vec![U7::<F>::zero(); poly_len],
                        vec![U7::<F>::zero(); poly_len],
                        vec![U7::<F>::zero(); poly_len],
                        vec![U7::<F>::zero(); poly_len],
                        vec![U7::<F>::zero(); poly_len],
                    )
                },
                |(mut a_sh, mut a_l, mut a_r, mut a_sf, mut a_id),
                 (b_sh, b_l, b_r, b_sf, b_id)| {
                    for (a, b) in a_sh.iter_mut().zip(b_sh.iter()) {
                        *a += b;
                    }
                    for (a, b) in a_l.iter_mut().zip(b_l.iter()) {
                        *a += b;
                    }
                    for (a, b) in a_r.iter_mut().zip(b_r.iter()) {
                        *a += b;
                    }
                    for (a, b) in a_sf.iter_mut().zip(b_sf.iter()) {
                        *a += b;
                    }
                    for (a, b) in a_id.iter_mut().zip(b_id.iter()) {
                        *a += b;
                    }
                    (a_sh, a_l, a_r, a_sf, a_id)
                },
            );

        // Reduce to field elements.
        let mut q_shift_half: Vec<F> = unsafe_allocate_zero_vec(poly_len);
        let mut q_left: Vec<F> = unsafe_allocate_zero_vec(poly_len);
        let mut q_right: Vec<F> = unsafe_allocate_zero_vec(poly_len);
        let mut q_shift_full: Vec<F> = unsafe_allocate_zero_vec(poly_len);
        let mut q_identity: Vec<F> = unsafe_allocate_zero_vec(poly_len);

        for i in 0..poly_len {
            q_shift_half[i] = F::from_barrett_reduce(rows_shift_half[i]);
            q_left[i] = F::from_barrett_reduce(rows_left[i]);
            q_right[i] = F::from_barrett_reduce(rows_right[i]);
            q_shift_full[i] = F::from_barrett_reduce(rows_shift_full[i]);
            q_identity[i] = F::from_barrett_reduce(rows_identity[i]);
        }

        // Operand Q0 is shared (ShiftHalfSuffixPolynomial).
        left.Q = [
            DensePolynomial::new(q_shift_half.clone()),
            DensePolynomial::new(q_left),
        ];
        right.Q = [DensePolynomial::new(q_shift_half), DensePolynomial::new(q_right)];

        identity.Q = [
            DensePolynomial::new(q_shift_full),
            DensePolynomial::new(q_identity),
        ];
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use ark_bn254::Fr;
    use ark_ff::{AdditiveGroup, Field};
    use ark_std::{test_rng, UniformRand};
    use rand_core::RngCore;
    use crate::poly::identity_poly::{IdentityPolynomial, OperandPolynomial, OperandSide};

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

    #[test]
    fn init_q_raf_matches_legacy_init_q() {
        // Compare the fused RAF initializer against the legacy path:
        //   init_Q_dual(left_operand, right_operand) + init_Q(identity)
        //
        // We exercise multiple phases by binding `chunk_len` challenges between inits.
        const TOTAL_LEN: usize = 8;
        const CHUNK_LEN: usize = 2;
        const N_CYCLES: usize = 64;

        let mut rng = test_rng();

        // Random per-cycle lookup bits (address index) and weights.
        let lookup_bits: Vec<LookupBits> = (0..N_CYCLES)
            .map(|_| {
                let lo = rng.next_u64() as u128;
                let hi = rng.next_u64() as u128;
                LookupBits::new((hi << 64) | lo, TOTAL_LEN)
            })
            .collect();

        let u_evals: Vec<Fr> = (0..N_CYCLES).map(|_| Fr::rand(&mut rng)).collect();

        let is_interleaved: Vec<bool> = (0..N_CYCLES).map(|_| (rng.next_u64() & 1) == 1).collect();
        let indices_interleaved: Vec<usize> = is_interleaved
            .iter()
            .enumerate()
            .filter_map(|(i, b)| if *b { Some(i) } else { None })
            .collect();
        let indices_identity: Vec<usize> = is_interleaved
            .iter()
            .enumerate()
            .filter_map(|(i, b)| if !*b { Some(i) } else { None })
            .collect();

        // Legacy instances
        let mut left_old = PrefixSuffixDecomposition::new(
            Box::new(OperandPolynomial::<Fr>::new(TOTAL_LEN, OperandSide::Left)),
            CHUNK_LEN,
            TOTAL_LEN,
        );
        let mut right_old = PrefixSuffixDecomposition::new(
            Box::new(OperandPolynomial::<Fr>::new(TOTAL_LEN, OperandSide::Right)),
            CHUNK_LEN,
            TOTAL_LEN,
        );
        let mut identity_old = PrefixSuffixDecomposition::new(
            Box::new(IdentityPolynomial::<Fr>::new(TOTAL_LEN)),
            CHUNK_LEN,
            TOTAL_LEN,
        );

        // Fused instances
        let mut left_new = PrefixSuffixDecomposition::new(
            Box::new(OperandPolynomial::<Fr>::new(TOTAL_LEN, OperandSide::Left)),
            CHUNK_LEN,
            TOTAL_LEN,
        );
        let mut right_new = PrefixSuffixDecomposition::new(
            Box::new(OperandPolynomial::<Fr>::new(TOTAL_LEN, OperandSide::Right)),
            CHUNK_LEN,
            TOTAL_LEN,
        );
        let mut identity_new = PrefixSuffixDecomposition::new(
            Box::new(IdentityPolynomial::<Fr>::new(TOTAL_LEN)),
            CHUNK_LEN,
            TOTAL_LEN,
        );

        let total_phases = TOTAL_LEN / CHUNK_LEN;
        for _phase in 0..total_phases {
            // Legacy
            PrefixSuffixDecomposition::init_Q_dual(
                &mut left_old,
                &mut right_old,
                &u_evals,
                &indices_interleaved,
                &lookup_bits,
            );
            identity_old.init_Q(&u_evals, &indices_identity, &lookup_bits);

            // Fused
            PrefixSuffixDecomposition::init_Q_raf(
                &mut left_new,
                &mut right_new,
                &mut identity_new,
                &u_evals,
                &lookup_bits,
                &is_interleaved,
            );

            assert_eq!(left_old.Q, left_new.Q);
            assert_eq!(right_old.Q, right_new.Q);
            assert_eq!(identity_old.Q, identity_new.Q);

            // Advance to next phase by binding `chunk_len` variables.
            for _ in 0..CHUNK_LEN {
                let r = <Fr as JoltField>::Challenge::random(&mut rng);
                left_old.bind(r);
                right_old.bind(r);
                identity_old.bind(r);

                left_new.bind(r);
                right_new.bind(r);
                identity_new.bind(r);
            }
        }
    }
}
