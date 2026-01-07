use std::ops::{Index, IndexMut};
use std::sync::{Arc, OnceLock, RwLock};

use allocative::Allocative;
use num_traits::Zero;
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
    pub sumcheck_evals_cache: Vec<OnceLock<(F, F)>>,
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
            sumcheck_evals_cache: vec![OnceLock::new(); cache_capacity],
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
    /// Fused initializer for RAF sumcheck Q polynomials.
    ///
    /// Initializes Q polynomials for left operand, right operand, and identity
    /// decompositions in a single scan over all cycles. This exploits the known
    /// structure of RAF suffixes to avoid dynamic dispatch and redundant computation.
    ///
    /// # Assumptions
    ///
    /// These assumptions hold for `ReadRafSumcheckProver`:
    ///
    /// - **`left` and `right`**: `OperandPolynomial` prefix-suffix decompositions
    ///   with suffixes `[ShiftHalfSuffixPolynomial, OperandPolynomial(side)]`.
    /// - **`identity`**: `IdentityPolynomial` prefix-suffix decomposition
    ///   with suffixes `[ShiftSuffixPolynomial, IdentityPolynomial]`.
    ///
    /// # Optimizations
    ///
    /// - Shift terms (`2^{suffix_len}` or `2^{suffix_len/2}`) are constants per phase,
    ///   so we accumulate raw `u_evals` and apply the shift multiplier once after reduction.
    /// - `uninterleave()` is computed once per cycle and reused for both left/right operands.
    ///
    /// # Arguments
    ///
    /// * `left`, `right`, `identity` - Decompositions to initialize (mutated in place).
    /// * `u_evals` - Per-cycle equality polynomial evaluations `eq(r_reduction, j)`.
    /// * `lookup_bits` - Per-cycle lookup indices (bit-packed).
    /// * `is_interleaved_operands` - Per-cycle flag: true for operand path, false for identity.
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
        let id_fits_u64 = suffix_len <= 64;
        let shift_half_f = F::from_u128(shift_half);
        let shift_full_f = F::from_u128(shift_full);

        let num_threads = rayon::current_num_threads();
        let chunk_size = lookup_bits.len().div_ceil(num_threads).max(1);

        type U7<F> = <F as JoltField>::Unreduced<7>;
        let total_len = 5 * poly_len;

        // Single allocation for all 5 accumulators (layout: [sh | l | r | sf | id]).
        // This reduces allocator overhead and helps explain any delay before per-chunk spans fire.
        let rows: Vec<U7<F>> = lookup_bits
            .par_chunks(chunk_size)
            .zip(u_evals.par_chunks(chunk_size))
            .zip(is_interleaved_operands.par_chunks(chunk_size))
            .fold(
                || unsafe_allocate_zero_vec(total_len),
                |mut acc, ((k_chunk, u_chunk), inter_chunk)| {
                    debug_assert_eq!(k_chunk.len(), u_chunk.len());
                    debug_assert_eq!(k_chunk.len(), inter_chunk.len());

                    let sh_off = 0;
                    let l_off = poly_len;
                    let r_off = 2 * poly_len;
                    let sf_off = 3 * poly_len;
                    let id_off = 4 * poly_len;

                    for ((k, u), is_interleaved) in
                        k_chunk.iter().zip(u_chunk.iter()).zip(inter_chunk.iter())
                    {
                        let (prefix_bits, suffix_bits) = k.split(suffix_len);
                        let r_index: usize = prefix_bits & (poly_len - 1);
                        let sh_idx = sh_off + r_index;
                        let l_idx = l_off + r_index;
                        let r_idx = r_off + r_index;
                        let sf_idx = sf_off + r_index;
                        let id_idx = id_off + r_index;

                        if *is_interleaved {
                            // Operand path (left/right)
                            // ShiftHalfSuffixPolynomial is constant for a fixed `suffix_len`, so
                            // we just accumulate `u` here and apply the 2^{suffix_len/2} scaling
                            // once per bucket after reduction.
                            acc[sh_idx] += *u.as_unreduced_ref();

                            let (lo_bits, ro_bits) = suffix_bits.uninterleave();
                            let lo: u64 = lo_bits.into();
                            if lo != 0 {
                                acc[l_idx] += u.mul_u64_unreduced(lo);
                            }
                            let ro: u64 = ro_bits.into();
                            if ro != 0 {
                                acc[r_idx] += u.mul_u64_unreduced(ro);
                            }
                        } else {
                            // Identity path
                            // ShiftSuffixPolynomial is constant for a fixed `suffix_len`, so
                            // we just accumulate `u` here and apply the 2^{suffix_len} scaling
                            // once per bucket after reduction.
                            acc[sf_idx] += *u.as_unreduced_ref();

                            if id_fits_u64 {
                                let id: u64 = suffix_bits.into();
                                if id != 0 {
                                    acc[id_idx] += u.mul_u64_unreduced(id);
                                }
                            } else {
                                let id: u128 = suffix_bits.into();
                                if id != 0 {
                                    acc[id_idx] += u.mul_u128_unreduced(id);
                                }
                            }
                        }
                    }

                    acc
                },
            )
            .reduce(
                || unsafe_allocate_zero_vec(total_len),
                |mut a, b| {
                    a.par_iter_mut()
                        .zip(b.par_iter())
                        .for_each(|(x, y)| *x += y);
                    a
                },
            );

        // Reduce unreduced accumulators to field elements
        let rows_shift_half: &[U7<F>] = &rows[..poly_len];
        let rows_left: &[U7<F>] = &rows[poly_len..2 * poly_len];
        let rows_right: &[U7<F>] = &rows[2 * poly_len..3 * poly_len];
        let rows_shift_full: &[U7<F>] = &rows[3 * poly_len..4 * poly_len];
        let rows_identity: &[U7<F>] = &rows[4 * poly_len..5 * poly_len];

        // Simple reductions (no scaling): left, right, identity
        let ([q_left, q_right, q_identity], (q_shift_half, q_shift_full)) = rayon::join(
            || {
                let (q_left, (q_right, q_identity)) = rayon::join(
                    || {
                        rows_left
                            .par_iter()
                            .copied()
                            .map(F::from_barrett_reduce)
                            .collect::<Vec<F>>()
                    },
                    || {
                        rayon::join(
                            || {
                                rows_right
                                    .par_iter()
                                    .copied()
                                    .map(F::from_barrett_reduce)
                                    .collect::<Vec<F>>()
                            },
                            || {
                                rows_identity
                                    .par_iter()
                                    .copied()
                                    .map(F::from_barrett_reduce)
                                    .collect::<Vec<F>>()
                            },
                        )
                    },
                );
                [q_left, q_right, q_identity]
            },
            // Scaled reductions: shift_half and shift_full need post-reduction scaling
            || {
                rayon::join(
                    || {
                        rows_shift_half
                            .par_iter()
                            .copied()
                            .map(|v| {
                                let reduced = F::from_barrett_reduce(v);
                                if shift_half == 1 {
                                    reduced
                                } else {
                                    reduced * shift_half_f
                                }
                            })
                            .collect::<Vec<F>>()
                    },
                    || {
                        rows_shift_full
                            .par_iter()
                            .copied()
                            .map(|v| {
                                let reduced = F::from_barrett_reduce(v);
                                if shift_full == 1 {
                                    reduced
                                } else {
                                    reduced * shift_full_f
                                }
                            })
                            .collect::<Vec<F>>()
                    },
                )
            },
        );

        // Operand Q0 is shared (ShiftHalfSuffixPolynomial).
        left.Q = [
            DensePolynomial::new(q_shift_half.clone()),
            DensePolynomial::new(q_left),
        ];
        right.Q = [
            DensePolynomial::new(q_shift_half),
            DensePolynomial::new(q_right),
        ];

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
