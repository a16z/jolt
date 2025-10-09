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
pub enum RaPolynomial<F: JoltField> {
    None,
    Round1(RaPolynomialRound1<F>),
    Round2(RaPolynomialRound2<F>),
    RoundN(MultilinearPolynomial<F>),
}

impl<F: JoltField> RaPolynomial<F> {
    pub fn new(lookup_indices: Arc<Vec<Option<u8>>>, eq_evals: Vec<F>) -> Self {
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
            Self::RoundN(mle) => mle.get_bound_coeff(j),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::None => panic!("RaPolynomial::len called on None"),
            Self::Round1(mle) => mle.len(),
            Self::Round2(mle) => mle.len(),
            Self::RoundN(mle) => mle.len(),
        }
    }
}

impl<F: JoltField> PolynomialBinding<F> for RaPolynomial<F> {
    fn is_bound(&self) -> bool {
        !matches!(self, Self::Round1(_))
    }

    fn bind(&mut self, r: F::Challenge, order: BindingOrder) {
        match self {
            Self::None => panic!("RaPolynomial::bind called on None"),
            Self::Round1(mle) => *self = Self::Round2(mem::take(mle).bind(r, order)),
            Self::Round2(mle) => *self = Self::RoundN(mem::take(mle).bind(r, order)),
            Self::RoundN(mle) => mle.bind(r, order),
        };
    }

    fn bind_parallel(&mut self, r: F::Challenge, order: BindingOrder) {
        self.bind(r, order);
    }

    fn final_sumcheck_claim(&self) -> F {
        match self {
            Self::RoundN(mle) => mle.final_sumcheck_claim(),
            _ => panic!(),
        }
    }
}

impl<F: JoltField> PolynomialEvaluation<F> for RaPolynomial<F> {
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
pub struct RaPolynomialRound1<F: JoltField> {
    // Index `x` stores `eq(x, r)`.
    F: Vec<F>,
    lookup_indices: Arc<Vec<Option<u8>>>,
}

impl<F: JoltField> RaPolynomialRound1<F> {
    fn len(&self) -> usize {
        self.lookup_indices.len()
    }

    #[tracing::instrument(skip_all, name = "RaPolynomialRound1::bind")]
    fn bind(self, r0: F::Challenge, binding_order: BindingOrder) -> RaPolynomialRound2<F> {
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
            .map_or(F::zero(), |i| self.F[i as usize])
    }
}

/// Represents `ra_i` during the 2nd of the last log(T) sumcheck rounds.
///
/// i.e. represents MLE `ra_i(r, r0, x)`
#[derive(Allocative, Default, Clone, Debug, PartialEq)]
pub struct RaPolynomialRound2<F: JoltField> {
    // Index `x` stores `eq(x, r_address_chunk_i) * eq(0, r0)`.
    F_0: Vec<F>,
    // Index `x` stores `eq(x, r_address_chunk_i) * eq(1, r0)`.
    F_1: Vec<F>,
    lookup_indices: Arc<Vec<Option<u8>>>,
    r0: F::Challenge,
    binding_order: BindingOrder,
}

impl<F: JoltField> RaPolynomialRound2<F> {
    fn len(&self) -> usize {
        self.lookup_indices.len() / 2
    }

    #[tracing::instrument(skip_all, name = "RaPolynomialRound2::bind")]
    fn bind(self, r1: F::Challenge, binding_order: BindingOrder) -> MultilinearPolynomial<F> {
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
        let lookup_indices = &self.lookup_indices;
        let n = lookup_indices.len() / 4;
        let mut res = unsafe_allocate_zero_vec(n);

        let chunk_size = 1 << 16;

        // Eval ra_i(r, r0, r1, j) for all j in the hypercube.
        match self.binding_order {
            BindingOrder::HighToLow => {
                res.par_chunks_mut(chunk_size).enumerate().for_each(
                    |(chunk_index, evals_chunk)| {
                        for (j, eval) in zip(chunk_index * chunk_size.., evals_chunk) {
                            let H_00 = lookup_indices[j].map_or(F::zero(), |i| F_00[i as usize]);
                            let H_01 =
                                lookup_indices[j + n].map_or(F::zero(), |i| F_01[i as usize]);
                            let H_10 =
                                lookup_indices[j + n * 2].map_or(F::zero(), |i| F_10[i as usize]);
                            let H_11 =
                                lookup_indices[j + n * 3].map_or(F::zero(), |i| F_11[i as usize]);
                            // ra_i(r, r0, r1, j) = eq((0, 0), (r0, r1)) * ra_i(r, 0, 0, j) +
                            //                      eq((0, 1), (r0, r1)) * ra_i(r, 0, 1, j) +
                            //                      eq((1, 0), (r0, r1)) * ra_i(r, 1, 0, j) +
                            //                      eq((1, 1), (r0, r1)) * ra_i(r, 1, 1, j)
                            *eval = H_00 + H_01 + H_10 + H_11;
                        }
                    },
                );
            }
            BindingOrder::LowToHigh => {
                res.par_chunks_mut(chunk_size).enumerate().for_each(
                    |(chunk_index, evals_chunk)| {
                        for (j, eval) in zip(chunk_index * chunk_size.., evals_chunk) {
                            let H_00 =
                                lookup_indices[4 * j].map_or(F::zero(), |i| F_00[i as usize]);
                            let H_01 =
                                lookup_indices[4 * j + 2].map_or(F::zero(), |i| F_01[i as usize]);
                            let H_10 =
                                lookup_indices[4 * j + 1].map_or(F::zero(), |i| F_10[i as usize]);
                            let H_11 =
                                lookup_indices[4 * j + 3].map_or(F::zero(), |i| F_11[i as usize]);
                            // ra_i(r, r0, r1, j) = eq((0, 0), (r0, r1)) * ra_i(r, 0, 0, j) +
                            //                      eq((0, 1), (r0, r1)) * ra_i(r, 0, 1, j) +
                            //                      eq((1, 0), (r0, r1)) * ra_i(r, 1, 0, j) +
                            //                      eq((1, 1), (r0, r1)) * ra_i(r, 1, 1, j)
                            *eval = H_00 + H_01 + H_10 + H_11;
                        }
                    },
                );
            }
        }

        res.into()
    }

    #[inline]
    fn get_bound_coeff(&self, j: usize) -> F {
        let mid = self.lookup_indices.len() / 2;
        match self.binding_order {
            BindingOrder::HighToLow => {
                let H_0 = self.lookup_indices[j].map_or(F::zero(), |i| self.F_0[i as usize]);
                let H_1 = self.lookup_indices[mid + j].map_or(F::zero(), |i| self.F_1[i as usize]);
                // Compute ra_i(r, r0, j) = eq(0, r0) * ra_i(r, 0, j) +
                //                          eq(1, r0) * ra_i(r, 1, j)
                H_0 + H_1
            }
            BindingOrder::LowToHigh => {
                let H_0 = self.lookup_indices[2 * j].map_or(F::zero(), |i| self.F_0[i as usize]);
                let H_1 =
                    self.lookup_indices[2 * j + 1].map_or(F::zero(), |i| self.F_1[i as usize]);
                // Compute ra_i(r, r0, j) = eq(0, r0) * ra_i(r, 0, j) +
                //                          eq(1, r0) * ra_i(r, 1, j)
                H_0 + H_1
            }
        }
    }
}
