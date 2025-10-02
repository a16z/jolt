use rayon::prelude::*;
use std::{iter::zip, mem, sync::Arc};

use allocative::Allocative;

use crate::{
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
    },
    utils::{lookup_bits::LookupBits, thread::unsafe_allocate_zero_vec},
    zkvm::instruction_lookups::K_CHUNK,
};

/// Represents the state of an `ra_i` polynomial during the last log(T) sumcheck rounds.
///
/// The first two rounds are specialized to reduce the amount of allocated memory.
// TODO: Implement PolynomialEvaluation.
#[derive(Allocative, Clone)]
pub enum RaPolynomial<F: JoltField> {
    Round1(RaPolynomialRound1<F>),
    Round2(RaPolynomialRound2<F>),
    RoundN(MultilinearPolynomial<F>),
}

impl<F: JoltField> RaPolynomial<F> {
    pub fn new(
        lookup_indices: Arc<[LookupBits]>,
        d: usize,
        log_k_chunk: usize,
        r_address: &[F::Challenge],
        i: usize,
    ) -> Self {
        let r = &r_address[log_k_chunk * i..log_k_chunk * (i + 1)];
        let eq_evals = EqPolynomial::evals(r);
        let shift = log_k_chunk * (d - 1 - i);
        let mask = (1 << log_k_chunk) - 1;
        Self::Round1(RaPolynomialRound1 {
            F: eq_evals,
            lookup_indices,
            shift,
            mask,
        })
    }

    pub fn get_bound_coeff(&self, j: usize) -> F {
        match self {
            Self::Round1(mle) => mle.get_bound_coeff(j),
            Self::Round2(mle) => mle.get_bound_coeff(j),
            Self::RoundN(mle) => mle.get_bound_coeff(j),
        }
    }
}

impl<F: JoltField> PolynomialBinding<F> for RaPolynomial<F> {
    fn is_bound(&self) -> bool {
        !matches!(self, Self::Round1(_))
    }

    fn bind(&mut self, r: F::Challenge, order: BindingOrder) {
        assert!(order == BindingOrder::HighToLow);
        match self {
            Self::Round1(mle) => *self = Self::Round2(mem::take(mle).bind(r)),
            Self::Round2(mle) => *self = Self::RoundN(mem::take(mle).bind(r)),
            Self::RoundN(mle) => mle.bind(r, BindingOrder::HighToLow),
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

/// Represents MLE `ra_i` during the 1st round of the last log(T) sumcheck rounds.
#[derive(Allocative, Default, Clone)]
pub struct RaPolynomialRound1<F: JoltField> {
    // Index `x` stores `eq(x, r)`.
    F: Vec<F>,
    lookup_indices: Arc<[LookupBits]>,
    /// Equals `log_k_chunk * (d - 1 - i)`.
    shift: usize,
    /// Equals `2^log_k_chunk - 1`.
    mask: usize,
}

impl<F: JoltField> RaPolynomialRound1<F> {
    fn bind(self, r0: F::Challenge) -> RaPolynomialRound2<F> {
        // Construct lookup tables.
        let eq_0_r0 = EqPolynomial::mle(&[F::zero()], &[r0]);
        let eq_1_r0 = EqPolynomial::mle(&[F::one()], &[r0]);
        let F_0 = self.F.iter().map(|v| eq_0_r0 * v).collect();
        let F_1 = self.F.iter().map(|v| eq_1_r0 * v).collect();
        RaPolynomialRound2 {
            F_0,
            F_1,
            lookup_indices: self.lookup_indices,
            r0,
            shift: self.shift,
            mask: self.mask,
        }
    }

    fn get_bound_coeff(&self, j: usize) -> F {
        let key = ((u128::from(self.lookup_indices[j]) >> self.shift) as usize) & self.mask;
        // Lookup ra_i(r, j).
        self.F[key]
    }
}

/// Represents `ra_i` during the 2nd of the last log(T) sumcheck rounds.
///
/// i.e. represents MLE `ra_i(r, r0, x)`
#[derive(Allocative, Default, Clone)]
pub struct RaPolynomialRound2<F: JoltField> {
    // Index `x` stores `eq(x, r_address_chunk_i) * eq(0, r0)`.
    F_0: Vec<F>,
    // Index `x` stores `eq(x, r_address_chunk_i) * eq(1, r0)`.
    F_1: Vec<F>,
    lookup_indices: Arc<[LookupBits]>,
    r0: F::Challenge,
    /// Equals `log_k_chunk * (d - 1 - i)`.   
    shift: usize,
    /// Equals `2^log_k_chunk - 1`.    
    mask: usize,
}

impl<F: JoltField> RaPolynomialRound2<F> {
    fn bind(self, r1: F::Challenge) -> MultilinearPolynomial<F> {
        // Construct lookup tables.
        let eq_0_r1 = EqPolynomial::mle(&[F::zero()], &[r1]);
        let eq_1_r1 = EqPolynomial::mle(&[F::one()], &[r1]);
        let mut F_00 = Vec::with_capacity(K_CHUNK);
        let mut F_01 = Vec::with_capacity(K_CHUNK);
        let mut F_10 = Vec::with_capacity(K_CHUNK);
        let mut F_11 = Vec::with_capacity(K_CHUNK);
        for i in 0..K_CHUNK {
            F_00.push(self.F_0[i] * eq_0_r1);
            F_01.push(self.F_0[i] * eq_1_r1);
            F_10.push(self.F_1[i] * eq_0_r1);
            F_11.push(self.F_1[i] * eq_1_r1);
        }

        let lookup_indices = &self.lookup_indices;
        let shift = self.shift;
        let mask = self.mask;
        let n = lookup_indices.len() / 4;
        let mut res = unsafe_allocate_zero_vec(n);

        let chunk_size = 1 << 16;

        // Eval ra_i(r, r0, r1, j) for all j in the hypercube.
        res.par_chunks_mut(chunk_size)
            .enumerate()
            .for_each(|(chunk_index, evals_chunk)| {
                for (j, eval) in zip(chunk_index * chunk_size.., evals_chunk) {
                    let key0 = ((u128::from(lookup_indices[j]) >> shift) as usize) & mask;
                    let key1 = ((u128::from(lookup_indices[j + n]) >> shift) as usize) & mask;
                    let key2 = ((u128::from(lookup_indices[j + n * 2]) >> shift) as usize) & mask;
                    let key3 = ((u128::from(lookup_indices[j + n * 3]) >> shift) as usize) & mask;
                    let H_00 = F_00[key0];
                    let H_01 = F_01[key1];
                    let H_10 = F_10[key2];
                    let H_11 = F_11[key3];
                    // ra_i(r, r0, r1, j) = eq((0, 0), (r0, r1)) * ra_i(r, 0, 0, j) +
                    //                      eq((0, 1), (r0, r1)) * ra_i(r, 0, 1, j) +
                    //                      eq((1, 0), (r0, r1)) * ra_i(r, 1, 0, j) +
                    //                      eq((1, 1), (r0, r1)) * ra_i(r, 1, 1, j)
                    *eval = H_00 + H_01 + H_10 + H_11;
                }
            });

        res.into()
    }

    fn get_bound_coeff(&self, j: usize) -> F {
        let mid = self.lookup_indices.len() / 2;
        let key0 = ((u128::from(self.lookup_indices[j]) >> self.shift) as usize) & self.mask;
        let key1 = ((u128::from(self.lookup_indices[j + mid]) >> self.shift) as usize) & self.mask;
        let H_0 = self.F_0[key0];
        let H_1 = self.F_1[key1];
        // Compute ra_i(r, r0, j) = eq(0, r0) * ra_i(r, 0, j) +
        //                          eq(1, r0) * ra_i(r, 1, j)
        H_0 + H_1
    }
}
