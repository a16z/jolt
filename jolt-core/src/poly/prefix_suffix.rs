use std::ops::{Index, IndexMut};
use std::sync::{Arc, Mutex};

use rayon::prelude::*;
use strum::{EnumCount, IntoEnumIterator};
use strum_macros::{EnumCount as EnumCountMacro, EnumIter as EnumIterMacro};

use crate::field::JoltField;
use crate::poly::dense_mlpoly::DensePolynomial;
use crate::poly::multilinear_polynomial::{
    BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
};
use crate::subprotocols::sparse_dense_shout::LookupBits;
use crate::utils::math::Math;
use crate::utils::thread::{unsafe_allocate_zero_vec, unsafe_zero_slice};

#[repr(u8)]
#[derive(Clone, Copy, EnumIterMacro, EnumCountMacro)]
pub enum Prefix {
    BatchedUninterleaved,
    Identity,
}

pub type PrefixCheckpoints<F> = [Option<F>; Prefix::COUNT];

#[derive(Default)]
pub struct PrefixRegistry<F: JoltField> {
    pub checkpoints: PrefixCheckpoints<F>,
    pub polys: [Option<Arc<Mutex<CachedMultilinearPolynomial<F>>>>; Prefix::COUNT],
}

impl<F: JoltField> PrefixRegistry<F> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn next_phase(&mut self) {
        Prefix::iter().for_each(|p| {
            self.checkpoints[p] = self[p]
                .as_ref()
                .map(|p| p.lock().unwrap().final_sumcheck_claim());
            self[p] = None;
        });
    }
}

impl<F: JoltField> Index<Prefix> for PrefixRegistry<F> {
    type Output = Option<Arc<Mutex<CachedMultilinearPolynomial<F>>>>;

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

pub struct CachedMultilinearPolynomial<F: JoltField> {
    pub inner: MultilinearPolynomial<F>,
    pub sumcheck_evals_cache: Vec<Option<Vec<F>>>,
    pub bound_this_round: bool,
}

impl<F: JoltField> PolynomialEvaluation<F> for CachedMultilinearPolynomial<F> {
    fn evaluate(&self, x: &[F]) -> F {
        self.inner.evaluate(x)
    }

    fn batch_evaluate(polys: &[&Self], r: &[F]) -> (Vec<F>, Vec<F>) {
        MultilinearPolynomial::batch_evaluate(
            &polys.iter().map(|p| &p.inner).collect::<Vec<_>>(),
            r,
        )
    }

    fn sumcheck_evals(&self, index: usize, degree: usize, order: BindingOrder) -> Vec<F> {
        self.inner.sumcheck_evals(index, degree, order)
    }
}

impl<F: JoltField> PolynomialBinding<F> for CachedMultilinearPolynomial<F> {
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

impl<F: JoltField> CachedMultilinearPolynomial<F> {
    pub fn new(inner: MultilinearPolynomial<F>) -> Self {
        Self {
            inner,
            sumcheck_evals_cache: Vec::with_capacity(1 << 16),
            bound_this_round: false,
        }
    }

    pub fn cached_sumcheck_evals(
        &mut self,
        index: usize,
        degree: usize,
        order: BindingOrder,
    ) -> Vec<F> {
        // grow to the next power of two if needed
        if index >= self.sumcheck_evals_cache.len() {
            let new_len = (index + 1).next_power_of_two();
            self.sumcheck_evals_cache.resize(new_len, None);
        }
        if self.sumcheck_evals_cache[index].is_none() {
            self.sumcheck_evals_cache[index] = Some(self.sumcheck_evals(index, degree, order));
        }
        self.sumcheck_evals_cache[index].as_ref().unwrap().clone()
    }

    pub fn clear_cache(&mut self) {
        self.sumcheck_evals_cache.clear();
        self.bound_this_round = false;
    }
}

pub trait PrefixPolynomial<F: JoltField> {
    /// Computes P[i] polynomial
    /// Assumes binding_order to be HighToLow
    fn prefix_polynomial(
        &self,
        checkpoints: &PrefixCheckpoints<F>,
        prefix_len: usize,
    ) -> CachedMultilinearPolynomial<F>;
}

pub trait SuffixPolynomial<F: JoltField> {
    fn suffix_mle(&self, index: u64, suffix_len: usize) -> F;
}

pub trait PrefixSuffixPolynomial<F: JoltField, const ORDER: usize> {
    fn prefixes(
        &self,
        chunk_len: usize,
        prefix_registry: &mut PrefixRegistry<F>,
    ) -> [Option<Arc<Mutex<CachedMultilinearPolynomial<F>>>>; ORDER];
    fn suffixes(&self) -> [Box<dyn SuffixPolynomial<F> + Sync>; ORDER];
}

pub struct PrefixSuffixDecomposition<F: JoltField, const ORDER: usize> {
    poly: Box<dyn PrefixSuffixPolynomial<F, ORDER> + Send + Sync>,
    P: Vec<Option<Arc<Mutex<CachedMultilinearPolynomial<F>>>>>,
    Q: Vec<DensePolynomial<F>>,
    m: usize,
    chunk_len: usize,
    total_len: usize,
    phase: usize,
}

impl<F: JoltField, const ORDER: usize> PrefixSuffixDecomposition<F, ORDER> {
    pub fn new(
        poly: Box<dyn PrefixSuffixPolynomial<F, ORDER> + Send + Sync>,
        m: usize,
        total_len: usize,
    ) -> Self {
        assert!(
            total_len % m.log_2() == 0,
            "total_len must be a multiple of log_2(m)"
        );
        Self {
            poly,
            P: vec![],
            Q: Self::alloc_Q(m),
            m,
            chunk_len: m.log_2(),
            total_len,
            phase: 0,
        }
    }

    #[inline(always)]
    pub fn suffix_len(&self) -> usize {
        let total_chunks = self.total_len / self.chunk_len;
        let suffix_chunks = total_chunks - self.phase - 1;
        suffix_chunks * self.chunk_len
    }

    fn alloc_Q(m: usize) -> Vec<DensePolynomial<F>> {
        rayon::iter::repeatn(0, ORDER)
            .map(|_| DensePolynomial::new(unsafe_allocate_zero_vec(m)))
            .collect()
    }

    pub fn reset_Q(&mut self) {
        self.Q.iter_mut().for_each(|poly| {
            poly.len = self.m;
            poly.num_vars = poly.len.log_2();
            unsafe_zero_slice(&mut poly.Z);
        });
    }

    pub fn init_P(&mut self, prefix_registry: &mut PrefixRegistry<F>) {
        // TODO: Don't init this thing if our Q is ewerywhere zero
        self.P = self.poly.prefixes(self.chunk_len, prefix_registry).into();
    }

    pub fn init_Q<'a, I: IntoIterator<Item = &'a (usize, &'a LookupBits)> + Clone + Send + Sync>(
        &mut self,
        u_evals: &[F],
        indices: I,
    ) {
        if self.phase == 0 {
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
                    if t != F::zero() {
                        let u = u_evals[*j];
                        poly.Z[prefix_bits % self.m] += u * t;
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
                    let mut p = p.lock().unwrap();
                    let p_evals = p.cached_sumcheck_evals(index, 2, BindingOrder::HighToLow);
                    drop(p);
                    p_evals
                } else {
                    // Prefixes are just constant 1, 1 if it's none
                    vec![F::one(), F::one()]
                };
                let q_left = q[index];
                let q_right = q[index + len / 2];
                (
                    p_evals[0] * q_left,  // prefix(0) * suffix(0)
                    p_evals[1] * q_left,  // prefix(2) * suffix(0)
                    p_evals[1] * q_right, // prefix(2) * suffix(1)
                )
            })
            .reduce(
                || (F::zero(), F::zero(), F::zero()),
                |running, new| (running.0 + new.0, running.1 + new.1, running.2 + new.2),
            );
        (eval_0, eval_2_left + eval_2_left + eval_2_right)
    }

    pub fn bind(&mut self, r: F, order: BindingOrder) {
        assert_eq!(
            BindingOrder::HighToLow,
            order,
            "PrefixSuffixDecomposition only supports high-to-low binding"
        );
        self.P.par_iter().for_each(|p| {
            if let Some(p) = p {
                let mut p = p.lock().unwrap();
                p.bind_parallel(r, order);
            }
        });
        self.Q.par_iter_mut().for_each(|poly| {
            poly.bind_parallel(r, order);
        });
    }

    pub fn final_sumcheck_claim(&self) -> F {
        self.P
            .par_iter()
            .zip(self.poly.suffixes().par_iter())
            .map(|(p, suffix)| {
                let suff = suffix.suffix_mle(0, 0);
                if suff == F::zero() {
                    return F::zero();
                }
                if let Some(p) = p {
                    let p = p.lock().unwrap();
                    p.final_sumcheck_claim() * suff
                } else {
                    suff
                }
            })
            .sum()
    }

    pub fn next_round(&self) {
        self.P.par_iter().for_each(|p| {
            if let Some(p) = p {
                p.lock().unwrap().clear_cache();
            }
        });
    }

    pub fn next_phase(&mut self) {
        self.phase += 1;
    }
}
