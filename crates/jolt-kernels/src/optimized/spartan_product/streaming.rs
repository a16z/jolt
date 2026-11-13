//! The streaming stage-2 product remainder: while the two dense left/right
//! tables exceed [`DENSE_TABLE_CAP`], round `j` recomputes its pairs from the
//! rows with the uni-skip weights scaled by `eq(r_1..r_j, ·)` (LowToHigh
//! binding gives `f^(j)(y) = Σ_b eq(r, b)·f(2^j·y + b)`). The first round
//! whose tables fit fills them in the pass that yields its endpoints, and the
//! default kernel's dense rounds bind them from there.

use std::ops::Range;

use jolt_field::JoltField;
use jolt_poly::{EqPolynomial, UnivariatePoly};
use jolt_sumcheck::SumcheckError;
use jolt_utils::unsafe_allocate_zero_vec;
use jolt_verifier::stages::stage2::product_remainder::ProductRemainder;
use jolt_witness::{JoltWitnessPlane, WitnessError};

#[cfg(feature = "field-inline")]
use super::FieldInlineRowCursor;
use super::{
    rows_unavailable, take_carry, PrefixColumn, ProductRemainderKernel, ProductTables,
    SpartanProductCarry, DOMAIN,
};
use crate::optimized::streaming::{
    dense_crossover, gruen_pair_pass, product_endpoint_terms, PairHalves, DENSE_TABLE_CAP,
};
use crate::optimized::support::{eq_round_error, GruenRoundMessage};
use crate::{KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel};

/// The streaming stage-2 remainder slot.
pub struct StreamingProductRemainder;

/// The round whose pass fills both tables over `2^log_t` cycles: the first
/// whose tables fit `cap` bytes, or the round count when none does. Sizes are
/// compared in cycles against the cycles `cap` holds, which is the same test
/// as bytes against `cap` and cannot overflow where the byte count of
/// `2^log_t` cycles would, for `log_t < usize::BITS`. That bound is the
/// carry's: its uni-skip kernel sized the row store by `1 << log_t`, a shift
/// no check guards.
fn crossover_round<F: JoltField>(log_t: usize, cap: usize) -> usize {
    let cap_cycles = cap / (2 * size_of::<F>());
    dense_crossover(1 << log_t, cap_cycles, log_t)
}

impl<F: JoltField> PrepareKernel<F, ProductRemainder<F>> for StreamingProductRemainder {
    fn prepare(
        &self,
        session: &mut ProofSession,
        _witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, ProductRemainder<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = ProductRemainder<F>>>, KernelError<F>> {
        Ok(Box::new(ProductRemainderKernel::prepare_streamed(
            take_carry(session)?,
            &inputs,
            DENSE_TABLE_CAP,
        )?))
    }
}

impl<F: JoltField> ProductRemainderKernel<F> {
    /// `dense_cap` bounds the bytes of the two dense tables; production passes
    /// [`DENSE_TABLE_CAP`].
    pub(super) fn prepare_streamed(
        carry: SpartanProductCarry<F>,
        inputs: &ProverInputs<'_, F, ProductRemainder<F>>,
        dense_cap: usize,
    ) -> Result<Self, KernelError<F>> {
        let mut kernel = Self::unfilled(carry, inputs)?;
        let dense_round = crossover_round::<F>(kernel.challenges.total(), dense_cap);
        kernel.tables = ProductTables::Streamed { dense_round };
        if dense_round == 0 {
            kernel.pending_endpoints = Some(kernel.fill_tables()?);
        }
        Ok(kernel)
    }

    /// The current round's pass, filling the dense tables: each of length
    /// `T / 2^round`, allocated once at its final capacity and written in
    /// disjoint task slices.
    pub(super) fn fill_tables(&mut self) -> Result<(F, F), WitnessError> {
        let len = (1usize << self.challenges.total()) >> self.challenges.bound();
        let mut left: Vec<F> = unsafe_allocate_zero_vec(len);
        let mut right: Vec<F> = unsafe_allocate_zero_vec(len);
        let [zero, infinity] = self.round_pass(
            Some([&mut left[..], &mut right[..]]),
            product_endpoint_terms,
        )?;
        self.tables = ProductTables::Dense {
            left: PrefixColumn {
                values: left,
                tail: [F::zero(); 2],
            },
            right: PrefixColumn {
                values: right,
                tail: [F::zero(); 2],
            },
        };
        Ok((zero, infinity))
    }

    /// A round before the crossover: its endpoints, and `q(1)` when the
    /// split-eq factor needs it, from passes over the rows. A failed `q(1)`
    /// pass is held and returned, so its placeholder zero never reaches a
    /// round message.
    pub(super) fn streamed_round(
        &self,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        let [q_zero, q_infinity] = self
            .round_pass(None, product_endpoint_terms)
            .map_err(rows_unavailable)?;
        let gruen_round = self.split_eq.round().ok_or_else(|| eq_round_error(round))?;
        let mut failure = None;
        let message = gruen_round.checked_cubic(q_zero, q_infinity, previous_claim, round, || {
            self.round_pass(None, |[left, right]| [left[1] * right[1]])
                .map_or_else(
                    |error| {
                        failure = Some(error);
                        F::zero()
                    },
                    |[at_one]| at_one,
                )
        });
        match failure {
            Some(error) => Err(rows_unavailable(error)),
            None => message,
        }
    }

    /// Lane weights of the current round `j`'s pass, one per cycle offset `c`
    /// within a pair half: the uni-skip Lagrange weights times
    /// `eq(r_1..r_j, c)`, offset bit `i` pairing `r_{i+1}`.
    fn round_weights(&self) -> Vec<[F; DOMAIN]> {
        let point: Vec<F> = self.challenges.as_slice().iter().rev().copied().collect();
        EqPolynomial::<F>::evals(&point, None)
            .into_iter()
            .map(|eq| std::array::from_fn(|lane| eq * self.lagrange_weights[lane]))
            .collect()
    }

    /// One pass of the current round `j` over its pairs of `2^(j+1)` cycles
    /// ([`gruen_pair_pass`]), filling `tables` when given.
    fn round_pass<const N: usize>(
        &self,
        tables: Option<[&mut [F]; 2]>,
        terms: impl Fn([[F; 2]; 2]) -> [F; N] + Sync,
    ) -> Result<[F; N], WitnessError> {
        let weights = self.round_weights();
        gruen_pair_pass(
            &self.split_eq,
            self.challenges.bound() + 1,
            tables,
            terms,
            |range, chunk, sink| {
                self.walk_pairs(range, chunk, &weights, |pair, halves| {
                    sink.emit(pair, halves);
                })
            },
        )
    }

    /// One task's serial walk of `range` in the current round `j`: each row's
    /// factor forms, weighted for its offset within its half, add to the low
    /// or high half of its `2^(j+1)`-cycle pair ([`PairHalves`]), and `emit`
    /// receives every completed pair's `[left, right]` halves in order.
    fn walk_pairs(
        &self,
        range: Range<usize>,
        chunk: usize,
        weights: &[[F; DOMAIN]],
        mut emit: impl FnMut(usize, [[F; 2]; 2]),
    ) -> Result<(), WitnessError> {
        #[cfg(feature = "field-inline")]
        let mut field_cursor = FieldInlineRowCursor::seek(&self.field_rows, range.start);
        let mut halves = PairHalves::new(self.challenges.bound());
        self.rows.visit(range, chunk, |start, bundles| {
            for (t, row) in (start..).zip(bundles) {
                let factors = row.factors(
                    #[cfg(feature = "field-inline")]
                    field_cursor.advance(t),
                    &weights[halves.offset(t)],
                );
                if let Some((pair, pair_halves)) = halves.add(t, factors) {
                    emit(pair, pair_halves);
                }
            }
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use jolt_field::Fr;

    use super::crossover_round;

    /// The crossover is exact at the cap's boundary, is the round count when
    /// the last round's tables exceed the cap, and does not wrap when the
    /// tables exceed `usize` bytes (the widest cycle count `usize` holds).
    #[test]
    fn crossover_round_is_exact_at_the_cap_and_past_usize_bytes() {
        let cycle_bytes = 2 * size_of::<Fr>();
        assert_eq!(crossover_round::<Fr>(20, cycle_bytes << 10), 10);
        assert_eq!(crossover_round::<Fr>(20, (cycle_bytes << 10) - 1), 11);
        assert_eq!(crossover_round::<Fr>(20, usize::MAX), 0);
        assert_eq!(crossover_round::<Fr>(20, 2 * cycle_bytes), 19);
        assert_eq!(crossover_round::<Fr>(20, 2 * cycle_bytes - 1), 20);
        let wide = usize::BITS as usize - 1;
        assert_eq!(crossover_round::<Fr>(wide, cycle_bytes << 20), wide - 20);
    }
}
