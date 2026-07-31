use rayon::prelude::*;
use std::{iter::zip, mem, sync::Arc};

use allocative::Allocative;

use crate::{
    field::{ChallengeFieldOps, FieldChallengeOps, JoltField},
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
    },
    utils::thread::{drop_in_background_thread, unsafe_allocate_zero_vec},
};

#[derive(Allocative, Clone, Debug, PartialEq)]
enum RaLookupIndices<I: Copy + Send + Sync + 'static> {
    Sparse(Arc<Vec<Option<I>>>),
    SparseSentinel {
        values: Arc<Vec<I>>,
        sentinel: usize,
    },
    Dense(Arc<Vec<I>>),
    DenseStrided {
        values: Arc<Vec<I>>,
        len: usize,
        stride: usize,
        offset: usize,
    },
}

impl<I: Copy + Send + Sync + 'static> Default for RaLookupIndices<I> {
    fn default() -> Self {
        Self::Sparse(Arc::default())
    }
}

impl<I: Copy + Send + Sync + 'static> RaLookupIndices<I> {
    #[inline]
    fn len(&self) -> usize {
        match self {
            Self::Sparse(indices) => indices.len(),
            Self::SparseSentinel { values, .. } => values.len(),
            Self::Dense(indices) => indices.len(),
            Self::DenseStrided { len, .. } => *len,
        }
    }

    #[inline]
    fn lookup(&self, index: usize) -> Option<I>
    where
        I: Into<usize>,
    {
        match self {
            Self::Sparse(indices) => indices[index],
            Self::SparseSentinel { values, sentinel } => {
                let value = values[index];
                (value.into() != *sentinel).then_some(value)
            }
            Self::Dense(indices) => Some(indices[index]),
            Self::DenseStrided {
                values,
                stride,
                offset,
                ..
            } => Some(values[index * stride + offset]),
        }
    }

    #[inline]
    fn lookup_eval<F: JoltField>(&self, index: usize, table: &[F]) -> F
    where
        I: Into<usize>,
    {
        self.lookup(index).map_or(F::zero(), |i| table[i.into()])
    }
}

/// Represents the state of an `ra_i` polynomial during the last log(T) sumcheck rounds.
///
/// The first two rounds are specialized to reduce the amount of allocated memory.
#[derive(Allocative, Clone, Debug, PartialEq)]
pub enum RaPolynomial<I: Into<usize> + Copy + Default + Send + Sync + 'static, F: JoltField> {
    None,
    Round1(RaPolynomialRound1<I, F>),
    Round2(RaPolynomialRound2<I, F>),
    Round3(RaPolynomialRound3<I, F>),
    RoundN(MultilinearPolynomial<F>),
}

/// Read access used by the RA product-sum kernels.
pub trait RaPolynomialAccess<F: JoltField>: Sync {
    /// Returns the coefficient at `j` in the polynomial's current binding state.
    fn get_bound_coeff(&self, j: usize) -> F;
}

impl<I: Into<usize> + Copy + Default + Send + Sync + 'static, F: JoltField> RaPolynomial<I, F> {
    pub fn new(lookup_indices: Arc<Vec<Option<I>>>, eq_evals: Vec<F>) -> Self {
        Self::Round1(RaPolynomialRound1 {
            F: eq_evals,
            lookup_indices: RaLookupIndices::Sparse(lookup_indices),
        })
    }

    pub fn new_sparse_sentinel(
        lookup_indices: Arc<Vec<I>>,
        sentinel: usize,
        eq_evals: Vec<F>,
    ) -> Self {
        Self::Round1(RaPolynomialRound1 {
            F: eq_evals,
            lookup_indices: RaLookupIndices::SparseSentinel {
                values: lookup_indices,
                sentinel,
            },
        })
    }

    /// Constructs an RA polynomial whose lookup index exists at every point.
    pub fn new_dense(lookup_indices: Arc<Vec<I>>, eq_evals: Vec<F>) -> Self {
        Self::Round1(RaPolynomialRound1 {
            F: eq_evals,
            lookup_indices: RaLookupIndices::Dense(lookup_indices),
        })
    }

    pub fn new_dense_strided(
        lookup_indices: Arc<Vec<I>>,
        len: usize,
        stride: usize,
        offset: usize,
        eq_evals: Vec<F>,
    ) -> Self {
        assert!(stride > 0);
        assert!(offset < stride);
        assert!(
            len == 0 || (len - 1) * stride + offset < lookup_indices.len(),
            "strided RA column exceeds its backing storage"
        );
        Self::Round1(RaPolynomialRound1 {
            F: eq_evals,
            lookup_indices: RaLookupIndices::DenseStrided {
                values: lookup_indices,
                len,
                stride,
                offset,
            },
        })
    }

    #[inline]
    pub fn get_bound_coeff(&self, j: usize) -> F {
        match self {
            Self::None => panic!("RaPolynomial::get_bound_coeff called on None"),
            Self::Round1(mle) => mle.get_bound_coeff(j),
            Self::Round2(mle) => mle.get_bound_coeff(j),
            Self::Round3(mle) => mle.get_bound_coeff(j),
            Self::RoundN(mle) => mle.get_bound_coeff(j),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::None => panic!("RaPolynomial::len called on None"),
            Self::Round1(mle) => mle.len(),
            Self::Round2(mle) => mle.len(),
            Self::Round3(mle) => mle.len(),
            Self::RoundN(mle) => mle.len(),
        }
    }
}

impl<I: Into<usize> + Copy + Default + Send + Sync + 'static, F: JoltField> RaPolynomialAccess<F>
    for RaPolynomial<I, F>
{
    #[inline]
    fn get_bound_coeff(&self, j: usize) -> F {
        RaPolynomial::get_bound_coeff(self, j)
    }
}

impl<I: Into<usize> + Copy + Default + Send + Sync + 'static, F: JoltField> PolynomialBinding<F>
    for RaPolynomial<I, F>
{
    fn is_bound(&self) -> bool {
        !matches!(self, Self::Round1(_))
    }

    fn bind(&mut self, _r: F::Challenge, _order: BindingOrder) {
        unimplemented!()
    }

    fn bind_parallel(&mut self, r: F::Challenge, order: BindingOrder) {
        match self {
            Self::None => panic!("RaPolynomial::bind called on None"),
            Self::Round1(mle) => *self = Self::Round2(mem::take(mle).bind(r, order)),
            Self::Round2(mle) => *self = Self::Round3(mem::take(mle).bind(r, order)),
            Self::Round3(mle) => *self = Self::RoundN(mem::take(mle).bind(r, order)),
            Self::RoundN(mle) => mle.bind_parallel(r, order),
        };
    }

    fn final_sumcheck_claim(&self) -> F {
        match self {
            Self::RoundN(mle) => mle.final_sumcheck_claim(),
            _ => panic!(),
        }
    }
}

impl<I: Into<usize> + Copy + Default + Send + Sync + 'static, F: JoltField> PolynomialEvaluation<F>
    for RaPolynomial<I, F>
{
    fn evaluate<C>(&self, _r: &[C]) -> F
    where
        C: Copy + Send + Sync + Into<F> + ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>,
    {
        unimplemented!()
    }

    fn batch_evaluate<C>(_polys: &[&Self], _r: &[C]) -> Vec<F>
    where
        Self: Sized,
        C: Copy + Send + Sync + Into<F> + ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>,
    {
        unimplemented!()
    }

    #[inline]
    fn sumcheck_evals(&self, index: usize, degree: usize, order: BindingOrder) -> Vec<F> {
        debug_assert!(degree > 0);
        debug_assert!(index < self.len() / 2);

        let mut evals = vec![F::zero(); degree];
        match order {
            BindingOrder::HighToLow => {
                evals[0] = self.get_bound_coeff(index);
                if degree == 1 {
                    return evals;
                }
                let mut eval = self.get_bound_coeff(index + self.len() / 2);
                let m = eval - evals[0];
                for i in 1..degree {
                    eval += m;
                    evals[i] = eval;
                }
            }
            BindingOrder::LowToHigh => {
                evals[0] = self.get_bound_coeff(2 * index);
                if degree == 1 {
                    return evals;
                }
                let mut eval = self.get_bound_coeff(2 * index + 1);
                let m = eval - evals[0];
                for i in 1..degree {
                    eval += m;
                    evals[i] = eval;
                }
            }
        };
        evals
    }
}

/// Represents MLE `ra_i` during the 1st round of the last log(T) sumcheck rounds.
#[derive(Allocative, Default, Clone, Debug, PartialEq)]
pub struct RaPolynomialRound1<I: Into<usize> + Copy + Default + Send + Sync + 'static, F: JoltField>
{
    // Index `x` stores `eq(x, r)`.
    F: Vec<F>,
    lookup_indices: RaLookupIndices<I>,
}

impl<I: Into<usize> + Copy + Default + Send + Sync + 'static, F: JoltField>
    RaPolynomialRound1<I, F>
{
    fn len(&self) -> usize {
        self.lookup_indices.len()
    }

    #[tracing::instrument(skip_all, name = "RaPolynomialRound1::bind")]
    fn bind(self, r0: F::Challenge, binding_order: BindingOrder) -> RaPolynomialRound2<I, F> {
        // Construct lookup tables.
        let eq_0_r0 = EqPolynomial::mle(&[F::zero()], &[r0]);
        let eq_1_r0 = EqPolynomial::mle(&[F::one()], &[r0]);
        let F_0 = self.F.iter().map(|v| eq_0_r0 * v).collect();
        let F_1 = self.F.iter().map(|v| eq_1_r0 * v).collect();
        drop_in_background_thread(self.F);
        RaPolynomialRound2 {
            F_0,
            F_1,
            lookup_indices: self.lookup_indices,
            r0,
            binding_order,
        }
    }

    #[inline]
    fn get_bound_coeff(&self, j: usize) -> F {
        // Lookup ra_i(r, j).
        self.lookup_indices
            .lookup(j)
            .map_or(F::zero(), |i| self.F[i.into()])
    }
}

/// Represents `ra_i` during the 2nd of the last log(T) sumcheck rounds.
///
/// i.e. represents MLE `ra_i(r, r0, x)`
#[derive(Allocative, Default, Clone, Debug, PartialEq)]
pub struct RaPolynomialRound2<I: Into<usize> + Copy + Default + Send + Sync + 'static, F: JoltField>
{
    // Index `x` stores `eq(x, r_address_chunk_i) * eq(0, r0)`.
    F_0: Vec<F>,
    // Index `x` stores `eq(x, r_address_chunk_i) * eq(1, r0)`.
    F_1: Vec<F>,
    lookup_indices: RaLookupIndices<I>,
    r0: F::Challenge,
    binding_order: BindingOrder,
}

impl<I: Into<usize> + Copy + Default + Send + Sync + 'static, F: JoltField>
    RaPolynomialRound2<I, F>
{
    fn len(&self) -> usize {
        self.lookup_indices.len() / 2
    }

    #[tracing::instrument(skip_all, name = "RaPolynomialRound2::bind")]
    fn bind(self, r1: F::Challenge, binding_order: BindingOrder) -> RaPolynomialRound3<I, F> {
        assert_eq!(binding_order, self.binding_order);
        // Construct lookup tables.
        let eq_0_r1 = EqPolynomial::mle(&[F::zero()], &[r1]);
        let eq_1_r1 = EqPolynomial::mle(&[F::one()], &[r1]);
        let mut F_00: Vec<F> = self.F_0.clone();
        let mut F_01: Vec<F> = self.F_0;
        let mut F_10: Vec<F> = self.F_1.clone();
        let mut F_11: Vec<F> = self.F_1;

        F_00.par_iter_mut().for_each(|f| *f *= eq_0_r1);
        F_01.par_iter_mut().for_each(|f| *f *= eq_1_r1);
        F_10.par_iter_mut().for_each(|f| *f *= eq_0_r1);
        F_11.par_iter_mut().for_each(|f| *f *= eq_1_r1);

        RaPolynomialRound3 {
            F_00,
            F_01,
            F_10,
            F_11,
            lookup_indices: self.lookup_indices,
            r1,
            binding_order: self.binding_order,
        }
    }

    #[inline]
    fn get_bound_coeff(&self, j: usize) -> F {
        let mid = self.lookup_indices.len() / 2;
        match self.binding_order {
            BindingOrder::HighToLow => {
                let H_0 = self
                    .lookup_indices
                    .lookup(j)
                    .map_or(F::zero(), |i| self.F_0[i.into()]);
                let H_1 = self
                    .lookup_indices
                    .lookup(mid + j)
                    .map_or(F::zero(), |i| self.F_1[i.into()]);
                // Compute ra_i(r, r0, j) = eq(0, r0) * ra_i(r, 0, j) +
                //                          eq(1, r0) * ra_i(r, 1, j)
                H_0 + H_1
            }
            BindingOrder::LowToHigh => {
                let H_0 = self
                    .lookup_indices
                    .lookup(2 * j)
                    .map_or(F::zero(), |i| self.F_0[i.into()]);
                let H_1 = self
                    .lookup_indices
                    .lookup(2 * j + 1)
                    .map_or(F::zero(), |i| self.F_1[i.into()]);
                // Compute ra_i(r, r0, j) = eq(0, r0) * ra_i(r, 0, j) +
                //                          eq(1, r0) * ra_i(r, 1, j)
                H_0 + H_1
            }
        }
    }
}

/// Represents `ra_i` during the 3nd of the last log(T) sumcheck rounds.
///
/// i.e. represents MLE `ra_i(r, r0, x)`
#[derive(Allocative, Default, Clone, Debug, PartialEq)]
pub struct RaPolynomialRound3<I: Into<usize> + Copy + Default + Send + Sync + 'static, F: JoltField>
{
    // Index `x` stores `eq(x, r_address_chunk_i) * eq(00, r0 r1)`.
    F_00: Vec<F>,
    // Index `x` stores `eq(x, r_address_chunk_i) * eq(01, r0 r1)`.
    F_01: Vec<F>,
    // Index `x` stores `eq(x, r_address_chunk_i) * eq(10, r0 r1)`.
    F_10: Vec<F>,
    // Index `x` stores `eq(x, r_address_chunk_i) * eq(11, r0 r1)`.
    F_11: Vec<F>,
    lookup_indices: RaLookupIndices<I>,
    r1: F::Challenge,
    binding_order: BindingOrder,
}

impl<I: Into<usize> + Copy + Default + Send + Sync + 'static, F: JoltField>
    RaPolynomialRound3<I, F>
{
    fn len(&self) -> usize {
        self.lookup_indices.len() / 4
    }

    #[tracing::instrument(skip_all, name = "RaPolynomialRound3::bind")]
    fn bind(self, r2: F::Challenge, _binding_order: BindingOrder) -> MultilinearPolynomial<F> {
        // Construct lookup tables.
        let eq_0_r2 = EqPolynomial::mle(&[F::zero()], &[r2]);
        let eq_1_r2 = EqPolynomial::mle(&[F::one()], &[r2]);
        let mut F_000: Vec<F> = self.F_00.clone();
        let mut F_001: Vec<F> = self.F_00;
        let mut F_010: Vec<F> = self.F_01.clone();
        let mut F_011: Vec<F> = self.F_01;
        let mut F_100: Vec<F> = self.F_10.clone();
        let mut F_101: Vec<F> = self.F_10;
        let mut F_110: Vec<F> = self.F_11.clone();
        let mut F_111: Vec<F> = self.F_11;

        F_000.par_iter_mut().for_each(|f| *f *= eq_0_r2);
        F_010.par_iter_mut().for_each(|f| *f *= eq_0_r2);
        F_100.par_iter_mut().for_each(|f| *f *= eq_0_r2);
        F_110.par_iter_mut().for_each(|f| *f *= eq_0_r2);
        F_001.par_iter_mut().for_each(|f| *f *= eq_1_r2);
        F_011.par_iter_mut().for_each(|f| *f *= eq_1_r2);
        F_101.par_iter_mut().for_each(|f| *f *= eq_1_r2);
        F_111.par_iter_mut().for_each(|f| *f *= eq_1_r2);

        let lookup_indices = &self.lookup_indices;
        let n = lookup_indices.len() / 8;
        let mut res = unsafe_allocate_zero_vec(n);

        let chunk_size = 1 << 16;

        // Eval ra_i(r, r0, r1, j) for all j in the hypercube.
        match self.binding_order {
            BindingOrder::HighToLow => {
                res.par_chunks_mut(chunk_size).enumerate().for_each(
                    |(chunk_index, evals_chunk)| {
                        for (j, eval) in zip(chunk_index * chunk_size.., evals_chunk) {
                            let H_000 = lookup_indices.lookup_eval(j, &F_000);
                            let H_001 = lookup_indices.lookup_eval(j + n, &F_001);
                            let H_010 = lookup_indices.lookup_eval(j + n * 2, &F_010);
                            let H_011 = lookup_indices.lookup_eval(j + n * 3, &F_011);
                            let H_100 = lookup_indices.lookup_eval(j + n * 4, &F_100);
                            let H_101 = lookup_indices.lookup_eval(j + n * 5, &F_101);
                            let H_110 = lookup_indices.lookup_eval(j + n * 6, &F_110);
                            let H_111 = lookup_indices.lookup_eval(j + n * 7, &F_111);
                            *eval = H_000 + H_010 + H_100 + H_110 + H_001 + H_011 + H_101 + H_111;
                        }
                    },
                );
            }
            BindingOrder::LowToHigh => {
                res.par_chunks_mut(chunk_size).enumerate().for_each(
                    |(chunk_index, evals_chunk)| {
                        for (j, eval) in zip(chunk_index * chunk_size.., evals_chunk) {
                            let H_000 = lookup_indices.lookup_eval(8 * j, &F_000);
                            let H_100 = lookup_indices.lookup_eval(8 * j + 1, &F_100);
                            let H_010 = lookup_indices.lookup_eval(8 * j + 2, &F_010);
                            let H_110 = lookup_indices.lookup_eval(8 * j + 3, &F_110);
                            let H_001 = lookup_indices.lookup_eval(8 * j + 4, &F_001);
                            let H_101 = lookup_indices.lookup_eval(8 * j + 5, &F_101);
                            let H_011 = lookup_indices.lookup_eval(8 * j + 6, &F_011);
                            let H_111 = lookup_indices.lookup_eval(8 * j + 7, &F_111);
                            *eval = H_000 + H_010 + H_100 + H_110 + H_001 + H_011 + H_101 + H_111;
                        }
                    },
                );
            }
        }

        drop_in_background_thread(self.lookup_indices);
        drop_in_background_thread(F_000);
        drop_in_background_thread(F_100);
        drop_in_background_thread(F_010);
        drop_in_background_thread(F_110);
        drop_in_background_thread(F_001);
        drop_in_background_thread(F_101);
        drop_in_background_thread(F_011);
        drop_in_background_thread(F_111);

        res.into()
    }

    #[inline]
    fn get_bound_coeff(&self, j: usize) -> F {
        match self.binding_order {
            BindingOrder::HighToLow => {
                let n = self.lookup_indices.len() / 4;
                let H_00 = self.lookup_indices.lookup_eval(j, &self.F_00);
                let H_01 = self.lookup_indices.lookup_eval(j + n, &self.F_01);
                let H_10 = self.lookup_indices.lookup_eval(j + n * 2, &self.F_10);
                let H_11 = self.lookup_indices.lookup_eval(j + n * 3, &self.F_11);
                H_00 + H_10 + H_01 + H_11
            }
            BindingOrder::LowToHigh => {
                let H_00 = self.lookup_indices.lookup_eval(4 * j, &self.F_00);
                let H_10 = self.lookup_indices.lookup_eval(4 * j + 1, &self.F_10);
                let H_01 = self.lookup_indices.lookup_eval(4 * j + 2, &self.F_01);
                let H_11 = self.lookup_indices.lookup_eval(4 * j + 3, &self.F_11);
                H_00 + H_10 + H_01 + H_11
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use ark_bn254::Fr;
    use ark_std::{test_rng, UniformRand};
    use std::sync::Arc;

    use super::RaPolynomial;
    use crate::{
        field::JoltField,
        poly::multilinear_polynomial::{BindingOrder, PolynomialBinding},
    };

    fn assert_dense_ra_matches_sparse(binding_order: BindingOrder) {
        let lookup_indices = vec![
            0_u8, 255, 1, 254, 2, 253, 3, 252, 4, 251, 5, 250, 6, 249, 7, 248,
        ];
        let eq_evals: Vec<Fr> = (0..=u8::MAX).map(|value| Fr::from(value as u64)).collect();
        let sparse_indices = lookup_indices.iter().copied().map(Some).collect();
        let mut sparse = RaPolynomial::new(Arc::new(sparse_indices), eq_evals.clone());
        let mut dense: RaPolynomial<u8, Fr> =
            RaPolynomial::new_dense(Arc::new(lookup_indices), eq_evals);
        let mut rng = test_rng();

        while sparse.len() > 1 {
            assert_eq!(dense.len(), sparse.len());
            for j in 0..sparse.len() {
                assert_eq!(dense.get_bound_coeff(j), sparse.get_bound_coeff(j));
            }

            let challenge = <Fr as JoltField>::Challenge::rand(&mut rng);
            sparse.bind_parallel(challenge, binding_order);
            dense.bind_parallel(challenge, binding_order);
        }

        assert_eq!(dense.final_sumcheck_claim(), sparse.final_sumcheck_claim());
    }

    fn assert_strided_ra_matches_dense(binding_order: BindingOrder) {
        const STRIDE: usize = 5;
        const OFFSET: usize = 2;
        let lookup_indices = vec![
            0_u8, 255, 1, 254, 2, 253, 3, 252, 4, 251, 5, 250, 6, 249, 7, 248,
        ];
        let mut rows = vec![123_u8; lookup_indices.len() * STRIDE];
        for (row, index) in lookup_indices.iter().copied().enumerate() {
            rows[row * STRIDE + OFFSET] = index;
        }
        let eq_evals: Vec<Fr> = (0..=u8::MAX).map(|value| Fr::from(value as u64)).collect();
        let mut dense: RaPolynomial<u8, Fr> =
            RaPolynomial::new_dense(Arc::new(lookup_indices.clone()), eq_evals.clone());
        let mut strided: RaPolynomial<u8, Fr> = RaPolynomial::new_dense_strided(
            Arc::new(rows),
            lookup_indices.len(),
            STRIDE,
            OFFSET,
            eq_evals,
        );
        let mut rng = test_rng();

        while dense.len() > 1 {
            assert_eq!(strided.len(), dense.len());
            for j in 0..dense.len() {
                assert_eq!(strided.get_bound_coeff(j), dense.get_bound_coeff(j));
            }

            let challenge = <Fr as JoltField>::Challenge::rand(&mut rng);
            dense.bind_parallel(challenge, binding_order);
            strided.bind_parallel(challenge, binding_order);
        }

        assert_eq!(strided.final_sumcheck_claim(), dense.final_sumcheck_claim());
    }

    fn assert_sentinel_ra_matches_sparse(binding_order: BindingOrder) {
        const SENTINEL: usize = usize::MAX;
        let lookup_indices = vec![
            Some(0_usize),
            None,
            Some(1),
            Some(254),
            None,
            Some(253),
            Some(3),
            None,
            Some(4),
            Some(251),
            Some(5),
            None,
            Some(6),
            Some(249),
            None,
            Some(248),
        ];
        let sentinel_indices = lookup_indices
            .iter()
            .map(|index| index.unwrap_or(SENTINEL))
            .collect();
        let eq_evals: Vec<Fr> = (0..=u8::MAX).map(|value| Fr::from(value as u64)).collect();
        let mut sparse = RaPolynomial::new(Arc::new(lookup_indices), eq_evals.clone());
        let mut sentinel: RaPolynomial<usize, Fr> =
            RaPolynomial::new_sparse_sentinel(Arc::new(sentinel_indices), SENTINEL, eq_evals);
        let mut rng = test_rng();

        while sparse.len() > 1 {
            assert_eq!(sentinel.len(), sparse.len());
            for j in 0..sparse.len() {
                assert_eq!(sentinel.get_bound_coeff(j), sparse.get_bound_coeff(j));
            }

            let challenge = <Fr as JoltField>::Challenge::rand(&mut rng);
            sparse.bind_parallel(challenge, binding_order);
            sentinel.bind_parallel(challenge, binding_order);
        }

        assert_eq!(
            sentinel.final_sumcheck_claim(),
            sparse.final_sumcheck_claim()
        );
    }

    #[test]
    fn dense_ra_polynomial_matches_sparse() {
        assert_dense_ra_matches_sparse(BindingOrder::LowToHigh);
        assert_dense_ra_matches_sparse(BindingOrder::HighToLow);
    }

    #[test]
    fn strided_ra_polynomial_matches_dense() {
        assert_strided_ra_matches_dense(BindingOrder::LowToHigh);
        assert_strided_ra_matches_dense(BindingOrder::HighToLow);
    }

    #[test]
    fn sentinel_ra_polynomial_matches_sparse() {
        assert_sentinel_ra_matches_sparse(BindingOrder::LowToHigh);
        assert_sentinel_ra_matches_sparse(BindingOrder::HighToLow);
    }
}
