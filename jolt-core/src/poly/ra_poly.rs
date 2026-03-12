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

/// When the table round reaches this many groups, the next bind materializes to dense.
const MATERIALIZE_THRESHOLD: usize = 8;

/// RA polynomial with lazy materialization via a table-doubling state machine.
///
/// Starts with 1 eq table group. Each bind doubles the tables (splitting on a new
/// challenge bit). After reaching `MATERIALIZE_THRESHOLD` groups, the next bind
/// materializes to a dense `MultilinearPolynomial`.
#[derive(Allocative, Clone, Debug, PartialEq)]
pub enum RaPolynomial<I: Into<usize> + Copy + Default + Send + Sync + 'static, F: JoltField> {
    None,
    TableRound(RaPolynomialTableRound<I, F>),
    RoundN(MultilinearPolynomial<F>),
}

impl<I: Into<usize> + Copy + Default + Send + Sync + 'static, F: JoltField> RaPolynomial<I, F> {
    pub fn new(lookup_indices: Arc<Vec<Option<I>>>, eq_evals: Vec<F>) -> Self {
        Self::TableRound(RaPolynomialTableRound {
            tables: vec![eq_evals],
            lookup_indices,
            binding_order: BindingOrder::LowToHigh,
        })
    }

    #[inline]
    pub fn get_bound_coeff(&self, j: usize) -> F {
        match self {
            Self::None => panic!("RaPolynomial::get_bound_coeff called on None"),
            Self::TableRound(t) => t.get_bound_coeff(j),
            Self::RoundN(mle) => mle.get_bound_coeff(j),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::None => panic!("RaPolynomial::len called on None"),
            Self::TableRound(t) => t.len(),
            Self::RoundN(mle) => mle.len(),
        }
    }
}

impl<I: Into<usize> + Copy + Default + Send + Sync + 'static, F: JoltField> PolynomialBinding<F>
    for RaPolynomial<I, F>
{
    fn is_bound(&self) -> bool {
        match self {
            Self::TableRound(t) => t.n_groups() > 1,
            Self::RoundN(_) => true,
            Self::None => false,
        }
    }

    fn bind(&mut self, _r: F::Challenge, _order: BindingOrder) {
        unimplemented!()
    }

    fn bind_parallel(&mut self, r: F::Challenge, order: BindingOrder) {
        match self {
            Self::None => panic!("RaPolynomial::bind called on None"),
            Self::TableRound(t) => {
                if t.n_groups() >= MATERIALIZE_THRESHOLD {
                    *self = Self::RoundN(mem::take(t).materialize(r, order));
                } else {
                    *self = Self::TableRound(mem::take(t).bind(r, order));
                }
            }
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

/// Generic table round for RaPolynomial with `n_groups` eq table groups.
///
/// Tables are stored in LowToHigh interleaving order: after k binds, table at
/// index `i` corresponds to the bit pattern where bit_0 = r0_val, bit_1 = r1_val, etc.
/// (LSB-first encoding of the bound challenge values).
#[derive(Allocative, Default, Clone, Debug, PartialEq)]
pub struct RaPolynomialTableRound<
    I: Into<usize> + Copy + Default + Send + Sync + 'static,
    F: JoltField,
> {
    tables: Vec<Vec<F>>,
    lookup_indices: Arc<Vec<Option<I>>>,
    binding_order: BindingOrder,
}

impl<I: Into<usize> + Copy + Default + Send + Sync + 'static, F: JoltField>
    RaPolynomialTableRound<I, F>
{
    #[inline]
    fn n_groups(&self) -> usize {
        self.tables.len()
    }

    fn len(&self) -> usize {
        self.lookup_indices.len() / self.n_groups()
    }

    /// Double tables from N to 2N groups by splitting on a new challenge.
    /// First N groups get scaled by eq(0, r), second N by eq(1, r).
    fn double_tables(tables: Vec<Vec<F>>, r: F::Challenge) -> Vec<Vec<F>> {
        let eq_0 = EqPolynomial::mle(&[F::zero()], &[r]);
        let eq_1 = EqPolynomial::mle(&[F::one()], &[r]);
        let n = tables.len();
        let mut doubled: Vec<Vec<F>> = Vec::with_capacity(2 * n);
        for t in &tables {
            doubled.push(t.clone());
        }
        for t in tables {
            doubled.push(t);
        }
        let (lo, hi) = doubled.split_at_mut(n);
        rayon::join(
            || {
                lo.par_iter_mut()
                    .for_each(|t| t.par_iter_mut().for_each(|f| *f *= eq_0))
            },
            || {
                hi.par_iter_mut()
                    .for_each(|t| t.par_iter_mut().for_each(|f| *f *= eq_1))
            },
        );
        doubled
    }

    #[tracing::instrument(skip_all, name = "RaPolynomialTableRound::bind")]
    fn bind(self, r: F::Challenge, order: BindingOrder) -> Self {
        if self.n_groups() > 1 {
            assert_eq!(order, self.binding_order);
        }
        Self {
            tables: Self::double_tables(self.tables, r),
            lookup_indices: self.lookup_indices,
            binding_order: order,
        }
    }

    #[tracing::instrument(skip_all, name = "RaPolynomialTableRound::materialize")]
    fn materialize(self, r: F::Challenge, order: BindingOrder) -> MultilinearPolynomial<F> {
        let binding_order = if self.n_groups() > 1 {
            assert_eq!(order, self.binding_order);
            self.binding_order
        } else {
            order
        };
        let tables = Self::double_tables(self.tables, r);
        let n_groups = tables.len();
        let lookup_indices = &self.lookup_indices;
        let n = lookup_indices.len() / n_groups;
        let mut res: Vec<F> = unsafe_allocate_zero_vec(n);

        let chunk_size = 1 << 16;
        match binding_order {
            BindingOrder::HighToLow => {
                let n_bits = n_groups.trailing_zeros() as usize;
                res.par_chunks_mut(chunk_size).enumerate().for_each(
                    |(chunk_index, evals_chunk)| {
                        for (j, eval) in zip(chunk_index * chunk_size.., evals_chunk) {
                            *eval = (0..n_groups)
                                .map(|seg| {
                                    let table_idx = bit_reverse(seg, n_bits);
                                    lookup_indices[seg * n + j]
                                        .map_or(F::zero(), |i| tables[table_idx][i.into()])
                                })
                                .sum();
                        }
                    },
                );
            }
            BindingOrder::LowToHigh => {
                res.par_chunks_mut(chunk_size).enumerate().for_each(
                    |(chunk_index, evals_chunk)| {
                        for (j, eval) in zip(chunk_index * chunk_size.., evals_chunk) {
                            *eval = (0..n_groups)
                                .map(|offset| {
                                    lookup_indices[n_groups * j + offset]
                                        .map_or(F::zero(), |i| tables[offset][i.into()])
                                })
                                .sum();
                        }
                    },
                );
            }
        }

        drop_in_background_thread(self.lookup_indices);
        res.into()
    }

    #[inline]
    fn get_bound_coeff(&self, j: usize) -> F {
        let n_groups = self.n_groups();
        match self.binding_order {
            BindingOrder::HighToLow => {
                let segment = self.lookup_indices.len() / n_groups;
                let n_bits = n_groups.trailing_zeros() as usize;
                (0..n_groups)
                    .map(|seg| {
                        let table_idx = bit_reverse(seg, n_bits);
                        self.lookup_indices[seg * segment + j]
                            .map_or(F::zero(), |i| self.tables[table_idx][i.into()])
                    })
                    .sum()
            }
            BindingOrder::LowToHigh => (0..n_groups)
                .map(|offset| {
                    self.lookup_indices[n_groups * j + offset]
                        .map_or(F::zero(), |i| self.tables[offset][i.into()])
                })
                .sum(),
        }
    }
}

/// Reverse the lowest `bits` bits of `x`.
#[inline]
pub(crate) fn bit_reverse(x: usize, bits: usize) -> usize {
    if bits == 0 {
        return 0;
    }
    let mut result = 0;
    let mut x = x;
    for _ in 0..bits {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    result
}
