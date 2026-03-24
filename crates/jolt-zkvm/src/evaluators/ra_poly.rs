//! RA (read-address) polynomial with lazy materialization.
//!
//! During the last `log(T)` rounds of the RA virtual sumcheck, each `ra_i`
//! polynomial progresses through a state machine:
//!
//! ```text
//! Round1 ──bind──▸ Round2 ──bind──▸ Round3 ──bind──▸ RoundN (dense)
//! ```
//!
//! The first three rounds use compact specialized storage to avoid
//! materializing the full multilinear polynomial. Only the Round3→RoundN
//! transition materializes a dense evaluation table.

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use std::{mem, sync::Arc};

use jolt_field::Field;
use jolt_poly::{
    thread::{drop_in_background_thread, unsafe_allocate_zero_vec},
    BindingOrder, Polynomial,
};

/// `eq(b, r) = b·r + (1-b)·(1-r)` for a single-bit argument.
#[inline(always)]
fn eq_single_bit<F: Field>(bit: F, r: F) -> F {
    bit * r + (F::one() - bit) * (F::one() - r)
}

/// State machine for an `ra_i` polynomial during the last `log(T)` sumcheck rounds.
///
/// Generic over:
/// - `I`: lookup index type (typically `u8`)
/// - `F`: field type
#[allow(non_snake_case)]
#[derive(Clone)]
pub enum RaPolynomial<I: Into<usize> + Copy + Default + Send + Sync + 'static, F: Field> {
    None,
    Round1(RaPolynomialRound1<I, F>),
    Round2(RaPolynomialRound2<I, F>),
    Round3(RaPolynomialRound3<I, F>),
    RoundN(Polynomial<F>),
}

#[allow(non_snake_case)]
impl<I: Into<usize> + Copy + Default + Send + Sync + 'static, F: Field> RaPolynomial<I, F> {
    pub fn new(lookup_indices: Arc<Vec<Option<I>>>, eq_evals: Vec<F>) -> Self {
        Self::Round1(RaPolynomialRound1 {
            F: eq_evals,
            lookup_indices,
        })
    }

    /// Returns the bound coefficient at index `j` for the current round state.
    ///
    /// In rounds 1-3, this uses lookup indices to avoid materializing the full
    /// polynomial. In RoundN, this is a direct table lookup.
    #[inline]
    pub fn get_bound_coeff(&self, j: usize) -> F {
        match self {
            Self::None => panic!("RaPolynomial::get_bound_coeff called on None"),
            Self::Round1(mle) => mle.get_bound_coeff(j),
            Self::Round2(mle) => mle.get_bound_coeff(j),
            Self::Round3(mle) => mle.get_bound_coeff(j),
            Self::RoundN(poly) => poly.evals()[j],
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::None => panic!("RaPolynomial::len called on None"),
            Self::Round1(mle) => mle.len(),
            Self::Round2(mle) => mle.len(),
            Self::Round3(mle) => mle.len(),
            Self::RoundN(poly) => poly.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn is_bound(&self) -> bool {
        !matches!(self, Self::Round1(_))
    }

    /// Bind the next variable, advancing the state machine.
    pub fn bind(&mut self, r: F, order: BindingOrder) {
        match self {
            Self::None => panic!("RaPolynomial::bind called on None"),
            Self::Round1(mle) => *self = Self::Round2(mem::take(mle).bind_round(r, order)),
            Self::Round2(mle) => *self = Self::Round3(mem::take(mle).bind_round(r, order)),
            Self::Round3(mle) => *self = Self::RoundN(mem::take(mle).bind_round(r, order)),
            Self::RoundN(poly) => {
                poly.bind_with_order(r, order);
            }
        }
    }

    /// Returns the final scalar value after all variables have been bound.
    ///
    /// Only valid in the `RoundN` state with a single evaluation remaining.
    pub fn final_sumcheck_claim(&self) -> F {
        match self {
            Self::RoundN(poly) => {
                debug_assert_eq!(poly.len(), 1);
                poly.evals()[0]
            }
            _ => panic!("final_sumcheck_claim requires RoundN state"),
        }
    }

    /// Returns the `(lo, hi)` pair for sumcheck round polynomial evaluation at index `j`.
    #[inline]
    pub fn sumcheck_eval_pair(&self, index: usize, order: BindingOrder) -> (F, F) {
        match order {
            BindingOrder::HighToLow => {
                let half = self.len() / 2;
                (
                    self.get_bound_coeff(index),
                    self.get_bound_coeff(index + half),
                )
            }
            BindingOrder::LowToHigh => (
                self.get_bound_coeff(2 * index),
                self.get_bound_coeff(2 * index + 1),
            ),
        }
    }
}

/// Round 1 state: stores eq evaluations and lookup indices.
///
/// `F[x]` stores `eq(x, r_address_chunk)`.
#[allow(non_snake_case)]
#[derive(Default, Clone)]
pub struct RaPolynomialRound1<I: Into<usize> + Copy + Default + Send + Sync + 'static, F: Field> {
    F: Vec<F>,
    lookup_indices: Arc<Vec<Option<I>>>,
}

#[allow(non_snake_case)]
impl<I: Into<usize> + Copy + Default + Send + Sync + 'static, F: Field> RaPolynomialRound1<I, F> {
    fn len(&self) -> usize {
        self.lookup_indices.len()
    }

    #[tracing::instrument(skip_all, name = "RaPolynomialRound1::bind")]
    fn bind_round(self, r0: F, binding_order: BindingOrder) -> RaPolynomialRound2<I, F> {
        let eq_0_r0 = eq_single_bit(F::zero(), r0);
        let eq_1_r0 = eq_single_bit(F::one(), r0);
        let F_0 = self.F.iter().map(|v| eq_0_r0 * *v).collect();
        let F_1 = self.F.iter().map(|v| eq_1_r0 * *v).collect();
        drop_in_background_thread(self.F);
        RaPolynomialRound2 {
            F_0,
            F_1,
            lookup_indices: self.lookup_indices,
            binding_order,
        }
    }

    #[inline]
    fn get_bound_coeff(&self, j: usize) -> F {
        self.lookup_indices
            .get(j)
            .expect("j out of bounds")
            .map_or(F::zero(), |i| self.F[i.into()])
    }
}

/// Round 2 state: stores `F_0` and `F_1` tables after binding `r0`.
///
/// `F_0[x]` stores `eq(x, r_addr) * eq(0, r0)`.
/// `F_1[x]` stores `eq(x, r_addr) * eq(1, r0)`.
#[allow(non_snake_case)]
#[derive(Clone)]
pub struct RaPolynomialRound2<I: Into<usize> + Copy + Default + Send + Sync + 'static, F: Field> {
    F_0: Vec<F>,
    F_1: Vec<F>,
    lookup_indices: Arc<Vec<Option<I>>>,
    binding_order: BindingOrder,
}

impl<I: Into<usize> + Copy + Default + Send + Sync + 'static, F: Field> Default
    for RaPolynomialRound2<I, F>
{
    fn default() -> Self {
        Self {
            F_0: Vec::new(),
            F_1: Vec::new(),
            lookup_indices: Arc::new(Vec::new()),
            binding_order: BindingOrder::default(),
        }
    }
}

#[allow(non_snake_case)]
impl<I: Into<usize> + Copy + Default + Send + Sync + 'static, F: Field> RaPolynomialRound2<I, F> {
    fn len(&self) -> usize {
        self.lookup_indices.len() / 2
    }

    #[tracing::instrument(skip_all, name = "RaPolynomialRound2::bind")]
    fn bind_round(self, r1: F, binding_order: BindingOrder) -> RaPolynomialRound3<I, F> {
        assert_eq!(binding_order, self.binding_order);
        let eq_0_r1 = eq_single_bit(F::zero(), r1);
        let eq_1_r1 = eq_single_bit(F::one(), r1);

        let mut F_00: Vec<F> = self.F_0.clone();
        let mut F_01: Vec<F> = self.F_0;
        let mut F_10: Vec<F> = self.F_1.clone();
        let mut F_11: Vec<F> = self.F_1;

        #[cfg(feature = "parallel")]
        {
            F_00.par_iter_mut().for_each(|f| *f *= eq_0_r1);
            F_01.par_iter_mut().for_each(|f| *f *= eq_1_r1);
            F_10.par_iter_mut().for_each(|f| *f *= eq_0_r1);
            F_11.par_iter_mut().for_each(|f| *f *= eq_1_r1);
        }

        #[cfg(not(feature = "parallel"))]
        {
            for f in &mut F_00 {
                *f *= eq_0_r1;
            }
            for f in &mut F_01 {
                *f *= eq_1_r1;
            }
            for f in &mut F_10 {
                *f *= eq_0_r1;
            }
            for f in &mut F_11 {
                *f *= eq_1_r1;
            }
        }

        RaPolynomialRound3 {
            F_00,
            F_01,
            F_10,
            F_11,
            lookup_indices: self.lookup_indices,
            binding_order: self.binding_order,
        }
    }

    #[inline]
    fn get_bound_coeff(&self, j: usize) -> F {
        let mid = self.lookup_indices.len() / 2;
        match self.binding_order {
            BindingOrder::HighToLow => {
                let h0 = self.lookup_indices[j].map_or(F::zero(), |i| self.F_0[i.into()]);
                let h1 = self.lookup_indices[mid + j].map_or(F::zero(), |i| self.F_1[i.into()]);
                h0 + h1
            }
            BindingOrder::LowToHigh => {
                let h0 = self.lookup_indices[2 * j].map_or(F::zero(), |i| self.F_0[i.into()]);
                let h1 = self.lookup_indices[2 * j + 1].map_or(F::zero(), |i| self.F_1[i.into()]);
                h0 + h1
            }
        }
    }
}

/// Round 3 state: stores four tables after binding `r0` and `r1`.
#[allow(non_snake_case)]
#[derive(Clone)]
pub struct RaPolynomialRound3<I: Into<usize> + Copy + Default + Send + Sync + 'static, F: Field> {
    F_00: Vec<F>,
    F_01: Vec<F>,
    F_10: Vec<F>,
    F_11: Vec<F>,
    lookup_indices: Arc<Vec<Option<I>>>,
    binding_order: BindingOrder,
}

impl<I: Into<usize> + Copy + Default + Send + Sync + 'static, F: Field> Default
    for RaPolynomialRound3<I, F>
{
    fn default() -> Self {
        Self {
            F_00: Vec::new(),
            F_01: Vec::new(),
            F_10: Vec::new(),
            F_11: Vec::new(),
            lookup_indices: Arc::new(Vec::new()),
            binding_order: BindingOrder::default(),
        }
    }
}

#[allow(non_snake_case)]
impl<I: Into<usize> + Copy + Default + Send + Sync + 'static, F: Field> RaPolynomialRound3<I, F> {
    fn len(&self) -> usize {
        self.lookup_indices.len() / 4
    }

    #[tracing::instrument(skip_all, name = "RaPolynomialRound3::bind")]
    fn bind_round(self, r2: F, _binding_order: BindingOrder) -> Polynomial<F> {
        let eq_0_r2 = eq_single_bit(F::zero(), r2);
        let eq_1_r2 = eq_single_bit(F::one(), r2);

        let mut f000: Vec<F> = self.F_00.clone();
        let mut f001: Vec<F> = self.F_00;
        let mut f010: Vec<F> = self.F_01.clone();
        let mut f011: Vec<F> = self.F_01;
        let mut f100: Vec<F> = self.F_10.clone();
        let mut f101: Vec<F> = self.F_10;
        let mut f110: Vec<F> = self.F_11.clone();
        let mut f111: Vec<F> = self.F_11;

        #[cfg(feature = "parallel")]
        {
            f000.par_iter_mut().for_each(|f| *f *= eq_0_r2);
            f010.par_iter_mut().for_each(|f| *f *= eq_0_r2);
            f100.par_iter_mut().for_each(|f| *f *= eq_0_r2);
            f110.par_iter_mut().for_each(|f| *f *= eq_0_r2);
            f001.par_iter_mut().for_each(|f| *f *= eq_1_r2);
            f011.par_iter_mut().for_each(|f| *f *= eq_1_r2);
            f101.par_iter_mut().for_each(|f| *f *= eq_1_r2);
            f111.par_iter_mut().for_each(|f| *f *= eq_1_r2);
        }

        #[cfg(not(feature = "parallel"))]
        {
            for f in &mut f000 {
                *f *= eq_0_r2;
            }
            for f in &mut f010 {
                *f *= eq_0_r2;
            }
            for f in &mut f100 {
                *f *= eq_0_r2;
            }
            for f in &mut f110 {
                *f *= eq_0_r2;
            }
            for f in &mut f001 {
                *f *= eq_1_r2;
            }
            for f in &mut f011 {
                *f *= eq_1_r2;
            }
            for f in &mut f101 {
                *f *= eq_1_r2;
            }
            for f in &mut f111 {
                *f *= eq_1_r2;
            }
        }

        let lookup_indices = &self.lookup_indices;
        let n = lookup_indices.len() / 8;
        let mut res = unsafe_allocate_zero_vec(n);

        let chunk_size = 1 << 16;

        match self.binding_order {
            BindingOrder::HighToLow => {
                #[cfg(feature = "parallel")]
                let iter = res.par_chunks_mut(chunk_size);
                #[cfg(not(feature = "parallel"))]
                let iter = res.chunks_mut(chunk_size);

                iter.enumerate().for_each(|(chunk_index, evals_chunk)| {
                    for (j, eval) in (chunk_index * chunk_size..).zip(evals_chunk.iter_mut()) {
                        let h000 = lookup_indices[j].map_or(F::zero(), |i| f000[i.into()]);
                        let h001 = lookup_indices[j + n].map_or(F::zero(), |i| f001[i.into()]);
                        let h010 = lookup_indices[j + n * 2].map_or(F::zero(), |i| f010[i.into()]);
                        let h011 = lookup_indices[j + n * 3].map_or(F::zero(), |i| f011[i.into()]);
                        let h100 = lookup_indices[j + n * 4].map_or(F::zero(), |i| f100[i.into()]);
                        let h101 = lookup_indices[j + n * 5].map_or(F::zero(), |i| f101[i.into()]);
                        let h110 = lookup_indices[j + n * 6].map_or(F::zero(), |i| f110[i.into()]);
                        let h111 = lookup_indices[j + n * 7].map_or(F::zero(), |i| f111[i.into()]);
                        *eval = h000 + h010 + h100 + h110 + h001 + h011 + h101 + h111;
                    }
                });
            }
            BindingOrder::LowToHigh => {
                #[cfg(feature = "parallel")]
                let iter = res.par_chunks_mut(chunk_size);
                #[cfg(not(feature = "parallel"))]
                let iter = res.chunks_mut(chunk_size);

                iter.enumerate().for_each(|(chunk_index, evals_chunk)| {
                    for (j, eval) in (chunk_index * chunk_size..).zip(evals_chunk.iter_mut()) {
                        let h000 = lookup_indices[8 * j].map_or(F::zero(), |i| f000[i.into()]);
                        let h100 = lookup_indices[8 * j + 1].map_or(F::zero(), |i| f100[i.into()]);
                        let h010 = lookup_indices[8 * j + 2].map_or(F::zero(), |i| f010[i.into()]);
                        let h110 = lookup_indices[8 * j + 3].map_or(F::zero(), |i| f110[i.into()]);
                        let h001 = lookup_indices[8 * j + 4].map_or(F::zero(), |i| f001[i.into()]);
                        let h101 = lookup_indices[8 * j + 5].map_or(F::zero(), |i| f101[i.into()]);
                        let h011 = lookup_indices[8 * j + 6].map_or(F::zero(), |i| f011[i.into()]);
                        let h111 = lookup_indices[8 * j + 7].map_or(F::zero(), |i| f111[i.into()]);
                        *eval = h000 + h010 + h100 + h110 + h001 + h011 + h101 + h111;
                    }
                });
            }
        }

        drop_in_background_thread(self.lookup_indices);
        drop_in_background_thread(f000);
        drop_in_background_thread(f100);
        drop_in_background_thread(f010);
        drop_in_background_thread(f110);
        drop_in_background_thread(f001);
        drop_in_background_thread(f101);
        drop_in_background_thread(f011);
        drop_in_background_thread(f111);

        Polynomial::new(res)
    }

    #[inline]
    fn get_bound_coeff(&self, j: usize) -> F {
        match self.binding_order {
            BindingOrder::HighToLow => {
                let n = self.lookup_indices.len() / 4;
                let h00 = self.lookup_indices[j].map_or(F::zero(), |i| self.F_00[i.into()]);
                let h01 = self.lookup_indices[j + n].map_or(F::zero(), |i| self.F_01[i.into()]);
                let h10 = self.lookup_indices[j + n * 2].map_or(F::zero(), |i| self.F_10[i.into()]);
                let h11 = self.lookup_indices[j + n * 3].map_or(F::zero(), |i| self.F_11[i.into()]);
                h00 + h10 + h01 + h11
            }
            BindingOrder::LowToHigh => {
                let h00 = self.lookup_indices[4 * j].map_or(F::zero(), |i| self.F_00[i.into()]);
                let h10 = self.lookup_indices[4 * j + 1].map_or(F::zero(), |i| self.F_10[i.into()]);
                let h01 = self.lookup_indices[4 * j + 2].map_or(F::zero(), |i| self.F_01[i.into()]);
                let h11 = self.lookup_indices[4 * j + 3].map_or(F::zero(), |i| self.F_11[i.into()]);
                h00 + h10 + h01 + h11
            }
        }
    }
}
