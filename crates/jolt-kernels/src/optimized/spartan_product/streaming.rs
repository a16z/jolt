//! The streaming stage-2 product remainder: while the two dense left/right
//! tables exceed the kernel's [`DenseCap`], round `j` recomputes its pairs
//! from the rows with the uni-skip weights scaled by `eq(r_1..r_j, ·)`
//! (LowToHigh binding gives `f^(j)(y) = Σ_b eq(r, b)·f(2^j·y + b)`). The
//! first round whose tables fit fills them in the pass that yields its
//! endpoints, and the default kernel's dense rounds bind them from there.
//!
//! The passes and the tables stop at the rows' constant suffix, as the
//! default kernel's tables do: a pass visits the pairs of the split-eq groups
//! that hold a cycle before the suffix and weighs the later pairs once, and
//! the tables keep the suffix as their tail ([`PrefixColumn`]).

use std::ops::Range;

use jolt_field::JoltField;
use jolt_poly::{EqPolynomial, UnivariatePoly};
use jolt_sumcheck::SumcheckError;
use jolt_verifier::stages::stage2::product_remainder::ProductRemainder;
use jolt_witness::{JoltWitnessPlane, WitnessError};

#[cfg(feature = "field-inline")]
use super::FieldInlineRowCursor;
use super::{
    rows_unavailable, take_carry, PrefixColumn, ProductRemainderKernel, ProductTables,
    SpartanProductCarry, DOMAIN,
};
use crate::optimized::streaming::{
    gruen_pair_pass, product_endpoint_terms, DenseCap, PairHalves, PairSuffix, DENSE_CAP,
};
use crate::optimized::support::{eq_round_error, GruenRoundMessage};
use crate::{KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel};

/// The streaming stage-2 remainder slot.
pub struct StreamingProductRemainder;

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
            DENSE_CAP,
        )?))
    }
}

impl<F: JoltField> ProductRemainderKernel<F> {
    /// `cap` bounds the two dense tables, which take two field elements per
    /// cycle before any bind: the first round at which they fit fills them,
    /// and no round does when none fits. Production passes [`DENSE_CAP`].
    pub(super) fn prepare_streamed(
        carry: SpartanProductCarry<F>,
        inputs: &ProverInputs<'_, F, ProductRemainder<F>>,
        cap: DenseCap,
    ) -> Result<Self, KernelError<F>> {
        let mut kernel = Self::unfilled(carry, inputs)?;
        let dense_round = cap.crossover(2 * size_of::<F>(), kernel.challenges.total());
        kernel.tables = ProductTables::Streamed { dense_round };
        if dense_round == 0 {
            kernel.pending_endpoints = Some(kernel.fill_tables()?);
        }
        Ok(kernel)
    }

    /// The current round's pass, filling the dense tables with the entries
    /// of the pairs it visits; the constant suffix past them is each table's
    /// tail.
    pub(super) fn fill_tables(&mut self) -> Result<(F, F), WitnessError> {
        let suffix = self.pair_suffix()?;
        let mut tables = [Vec::new(), Vec::new()];
        let [zero, infinity] =
            self.round_pass(&suffix, Some(&mut tables), product_endpoint_terms)?;
        let [left, right] = tables;
        let [left_tail, right_tail] = suffix.halves;
        self.tables = ProductTables::Dense {
            left: PrefixColumn {
                values: left,
                tail: left_tail,
            },
            right: PrefixColumn {
                values: right,
                tail: right_tail,
            },
        };
        Ok((zero, infinity))
    }

    /// The constant suffix of the current round's pairs: both halves of a
    /// pair past the rows' prefix hold the suffix row's factors in every
    /// round, the cycle weights of a half summing to one.
    fn pair_suffix(&self) -> Result<PairSuffix<F, 2>, WitnessError> {
        Ok(PairSuffix {
            start: self.prefix_cycles,
            halves: self.suffix_factors()?.map(|factor| [factor; 2]),
        })
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
        let suffix = self.pair_suffix().map_err(rows_unavailable)?;
        let [q_zero, q_infinity] = self
            .round_pass(&suffix, None, product_endpoint_terms)
            .map_err(rows_unavailable)?;
        let gruen_round = self.split_eq.round().ok_or_else(|| eq_round_error(round))?;
        let mut failure = None;
        let message = gruen_round.checked_cubic(q_zero, q_infinity, previous_claim, round, || {
            self.round_pass(&suffix, None, |[left, right]| [left[1] * right[1]])
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
        suffix: &PairSuffix<F, 2>,
        tables: Option<&mut [Vec<F>; 2]>,
        terms: impl Fn([[F; 2]; 2]) -> [F; N] + Sync,
    ) -> Result<[F; N], WitnessError> {
        let weights = self.round_weights();
        gruen_pair_pass(
            &self.split_eq,
            self.challenges.bound() + 1,
            suffix,
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
