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

impl<I: Into<usize> + Copy + Default + Send + Sync + 'static, F: JoltField> RaPolynomial<I, F> {
    pub fn new(lookup_indices: Arc<Vec<Option<I>>>, eq_evals: Vec<F>) -> Self {
        Self::Round1(RaPolynomialRound1 {
            F: eq_evals,
            lookup_indices,
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
    lookup_indices: Arc<Vec<Option<I>>>,
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
            .get(j)
            .expect("j out of bounds")
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
    lookup_indices: Arc<Vec<Option<I>>>,
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
                let H_0 = self.lookup_indices[j].map_or(F::zero(), |i| self.F_0[i.into()]);
                let H_1 = self.lookup_indices[mid + j].map_or(F::zero(), |i| self.F_1[i.into()]);
                // Compute ra_i(r, r0, j) = eq(0, r0) * ra_i(r, 0, j) +
                //                          eq(1, r0) * ra_i(r, 1, j)
                H_0 + H_1
            }
            BindingOrder::LowToHigh => {
                let H_0 = self.lookup_indices[2 * j].map_or(F::zero(), |i| self.F_0[i.into()]);
                let H_1 = self.lookup_indices[2 * j + 1].map_or(F::zero(), |i| self.F_1[i.into()]);
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
    lookup_indices: Arc<Vec<Option<I>>>,
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
                            let H_000 = lookup_indices[j].map_or(F::zero(), |i| F_000[i.into()]);
                            let H_001 =
                                lookup_indices[j + n].map_or(F::zero(), |i| F_001[i.into()]);
                            let H_010 =
                                lookup_indices[j + n * 2].map_or(F::zero(), |i| F_010[i.into()]);
                            let H_011 =
                                lookup_indices[j + n * 3].map_or(F::zero(), |i| F_011[i.into()]);
                            let H_100 =
                                lookup_indices[j + n * 4].map_or(F::zero(), |i| F_100[i.into()]);
                            let H_101 =
                                lookup_indices[j + n * 5].map_or(F::zero(), |i| F_101[i.into()]);
                            let H_110 =
                                lookup_indices[j + n * 6].map_or(F::zero(), |i| F_110[i.into()]);
                            let H_111 =
                                lookup_indices[j + n * 7].map_or(F::zero(), |i| F_111[i.into()]);
                            *eval = H_000 + H_010 + H_100 + H_110 + H_001 + H_011 + H_101 + H_111;
                        }
                    },
                );
            }
            BindingOrder::LowToHigh => {
                res.par_chunks_mut(chunk_size).enumerate().for_each(
                    |(chunk_index, evals_chunk)| {
                        for (j, eval) in zip(chunk_index * chunk_size.., evals_chunk) {
                            let H_000 =
                                lookup_indices[8 * j].map_or(F::zero(), |i| F_000[i.into()]);
                            let H_100 =
                                lookup_indices[8 * j + 1].map_or(F::zero(), |i| F_100[i.into()]);
                            let H_010 =
                                lookup_indices[8 * j + 2].map_or(F::zero(), |i| F_010[i.into()]);
                            let H_110 =
                                lookup_indices[8 * j + 3].map_or(F::zero(), |i| F_110[i.into()]);
                            let H_001 =
                                lookup_indices[8 * j + 4].map_or(F::zero(), |i| F_001[i.into()]);
                            let H_101 =
                                lookup_indices[8 * j + 5].map_or(F::zero(), |i| F_101[i.into()]);
                            let H_011 =
                                lookup_indices[8 * j + 6].map_or(F::zero(), |i| F_011[i.into()]);
                            let H_111 =
                                lookup_indices[8 * j + 7].map_or(F::zero(), |i| F_111[i.into()]);
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
                let H_00 = self.lookup_indices[j].map_or(F::zero(), |i| self.F_00[i.into()]);
                let H_01 = self.lookup_indices[j + n].map_or(F::zero(), |i| self.F_01[i.into()]);
                let H_10 =
                    self.lookup_indices[j + n * 2].map_or(F::zero(), |i| self.F_10[i.into()]);
                let H_11 =
                    self.lookup_indices[j + n * 3].map_or(F::zero(), |i| self.F_11[i.into()]);
                H_00 + H_10 + H_01 + H_11
            }
            BindingOrder::LowToHigh => {
                let H_00 = self.lookup_indices[4 * j].map_or(F::zero(), |i| self.F_00[i.into()]);
                let H_10 =
                    self.lookup_indices[4 * j + 1].map_or(F::zero(), |i| self.F_10[i.into()]);
                let H_01 =
                    self.lookup_indices[4 * j + 2].map_or(F::zero(), |i| self.F_01[i.into()]);
                let H_11 =
                    self.lookup_indices[4 * j + 3].map_or(F::zero(), |i| self.F_11[i.into()]);
                H_00 + H_10 + H_01 + H_11
            }
        }
    }
}
