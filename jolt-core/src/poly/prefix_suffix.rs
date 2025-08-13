use std::ops::{Index, IndexMut};
use std::sync::{Arc, RwLock};

use once_cell::sync::OnceCell;
use rayon::prelude::*;
use strum::{EnumCount, IntoEnumIterator};
use strum_macros::{EnumCount as EnumCountMacro, EnumIter as EnumIterMacro};

use crate::field::JoltField;
use crate::poly::dense_mlpoly::DensePolynomial;
use crate::poly::multilinear_polynomial::{BindingOrder, PolynomialBinding, PolynomialEvaluation};
use crate::utils::lookup_bits::LookupBits;
use crate::utils::math::Math;
use crate::utils::thread::{unsafe_allocate_zero_vec, unsafe_zero_slice};

#[repr(u8)]
#[derive(Clone, Copy, EnumIterMacro, EnumCountMacro)]
pub enum Prefix {
    RightOperand,
    LeftOperand,
    Identity,
}

pub type PrefixCheckpoints<F> = [Option<F>; Prefix::COUNT];

#[derive(Default)]
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

pub trait CacheablePolynomial<F: JoltField>:
    PolynomialEvaluation<F> + PolynomialBinding<F> + Send + Sync
{
}

pub struct CachedPolynomial<F: JoltField> {
    pub inner: Box<dyn CacheablePolynomial<F>>,
    pub sumcheck_evals_cache: Vec<OnceCell<(F, F)>>,
    pub bound_this_round: bool,
}

impl<F: JoltField> PolynomialEvaluation<F> for CachedPolynomial<F> {
    fn evaluate(&self, x: &[F]) -> F {
        self.inner.evaluate(x)
    }

    fn batch_evaluate(_polys: &[&Self], _r: &[F]) -> Vec<F> {
        unimplemented!("Currently unused")
    }

    fn sumcheck_evals(&self, index: usize, degree: usize, order: BindingOrder) -> Vec<F> {
        self.inner.sumcheck_evals(index, degree, order)
    }
}

impl<F: JoltField> PolynomialBinding<F> for CachedPolynomial<F> {
    fn bind(&mut self, r: F, order: BindingOrder) {
        if !self.bound_this_round {
            self.inner.bind(r, order);
            self.bound_this_round = true;
        }
    }

    fn bind_parallel(&mut self, r: F, order: BindingOrder) {
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
    pub fn new(inner: Box<dyn CacheablePolynomial<F>>, cache_capacity: usize) -> Self {
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
    fn suffix_mle(&self, index: u64, suffix_len: usize) -> u64;
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

pub struct PrefixSuffixDecomposition<F: JoltField, const ORDER: usize> {
    poly: Box<dyn PrefixSuffixPolynomial<F, ORDER> + Send + Sync>,
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
        rayon::iter::repeatn(0, ORDER)
            .map(|_| DensePolynomial::new(unsafe_allocate_zero_vec(m)))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    }

    pub fn reset_Q(&mut self) {
        self.Q.iter_mut().for_each(|poly| {
            poly.len = self.chunk_len.pow2();
            poly.num_vars = self.chunk_len;
            unsafe_zero_slice(&mut poly.Z);
        });
    }

    pub fn init_P(&mut self, prefix_registry: &mut PrefixRegistry<F>) {
        self.P = self
            .poly
            .prefixes(self.chunk_len, self.phase, prefix_registry);
    }

    #[tracing::instrument(skip_all)]
    pub fn init_Q<'a, I: IntoIterator<Item = &'a (usize, LookupBits)> + Clone + Send + Sync>(
        &mut self,
        u_evals: &[F],
        indices: I,
    ) {
        if self.phase != 0 {
            self.reset_Q();
        }
        let suffix_len = self.suffix_len();
        let suffixes = self.poly.suffixes();
        self.Q
            .par_iter_mut()
            .zip(suffixes.par_iter())
            .for_each(|(poly, suffix)| {
                for (j, k) in indices.clone() {
                    let (prefix_bits, suffix_bits) = k.split(suffix_len);
                    let t = suffix.suffix_mle(suffix_bits.into(), suffix_len);
                    if t != 0 {
                        if let Some(u) = u_evals.get(*j) {
                            poly.Z[prefix_bits % self.chunk_len.pow2()] += (*u).mul_u64(t);
                        }
                    }
                }
            });
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
                    p_evals.0 * q_left,  // prefix(0) * suffix(0)
                    p_evals.1 * q_left,  // prefix(2) * suffix(0)
                    p_evals.1 * q_right, // prefix(2) * suffix(1)
                )
            })
            .reduce(
                || (F::zero(), F::zero(), F::zero()),
                |running, new| (running.0 + new.0, running.1 + new.1, running.2 + new.2),
            );
        (eval_0, eval_2_right + eval_2_right - eval_2_left)
    }

    pub fn bind(&mut self, r: F) {
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
                let suff = suffix.suffix_mle(0, 0);
                if suff == 0 {
                    return F::zero();
                }
                if let Some(p) = p {
                    let p = p.read().unwrap();
                    p.final_sumcheck_claim().mul_u64(suff)
                } else {
                    F::from_u64(suff)
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
    use ark_bn254::Fr;
    use ark_ff::{AdditiveGroup, Field};
    use ark_std::test_rng;

    use super::*;

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

        let indices = (0..(1 << NUM_VARS))
            .map(|i| LookupBits::new(i, NUM_VARS))
            .enumerate()
            .collect::<Vec<_>>();

        let mut rr = vec![];
        for phase in 0..(NUM_VARS / PREFIX_LEN) {
            ps.init_P(&mut prefix_registry);
            ps.init_Q(
                &(0..(1 << (NUM_VARS - PREFIX_LEN * phase)))
                    .map(|_| Fr::ONE)
                    .collect::<Vec<_>>(),
                indices.iter(),
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
                let r = Fr::random(&mut rng);
                rr.push(r);
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
