//! The streaming stage-1 outer remainder: while the two `Az`/`Bz` tables over
//! the joint `(cycle ‖ stream)` domain exceed the kernel's [`DenseCap`], round
//! `j` recomputes its pairs from the rows, folding each row's group values
//! with `eq(r_1..r_j, ·)`-scaled Lagrange weights (LowToHigh binding gives
//! `f^(j)(y) = Σ_b eq(r, b)·f(2^j·y + b)`). The first round whose tables fit
//! fills them in the pass that yields its endpoints, and the default kernel's
//! dense rounds bind them from there.
//!
//! The passes and the tables stop at the rows' constant suffix, as the
//! default kernel's tables do: a pass visits the pairs of the split-eq groups
//! that hold a cycle before the suffix and weighs the later pairs once, and
//! the tables keep the suffix as their tail ([`PrefixColumn`]).

use std::ops::Range;

use jolt_field::JoltField;
use jolt_poly::{EqPolynomial, UnivariatePoly};
use jolt_sumcheck::SumcheckError;
use jolt_verifier::stages::stage1::outer_remainder::OuterRemainder;
use jolt_witness::{JoltWitnessPlane, WitnessError};

#[cfg(feature = "field-inline")]
use super::FieldInlineRowCursor;
use super::{
    rows_unavailable, take_carry, OuterRemainderKernel, OuterTables, PrefixColumn,
    SpartanOuterCarry, DOMAIN,
};
use crate::optimized::streaming::{
    gruen_pair_pass, product_endpoint_terms, DenseCap, PairHalves, PairSuffix, DENSE_CAP,
};
use crate::optimized::support::{eq_round_error, GruenRoundMessage};
use crate::{KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel};

/// The streaming stage-1 remainder slot.
pub struct StreamingOuterRemainder;

impl<F: JoltField> PrepareKernel<F, OuterRemainder<F>> for StreamingOuterRemainder {
    fn prepare(
        &self,
        session: &mut ProofSession,
        _witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, OuterRemainder<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = OuterRemainder<F>>>, KernelError<F>> {
        Ok(Box::new(OuterRemainderKernel::prepare_streamed(
            take_carry(session)?,
            &inputs,
            DENSE_CAP,
        )?))
    }
}

impl<F: JoltField> OuterRemainderKernel<F> {
    /// `cap` bounds the two dense tables, which take four field elements per
    /// cycle before any bind: the first round at which they fit fills them,
    /// and no round does when none fits. Production passes [`DENSE_CAP`].
    pub(super) fn prepare_streamed(
        carry: SpartanOuterCarry<F>,
        inputs: &ProverInputs<'_, F, OuterRemainder<F>>,
        cap: DenseCap,
    ) -> Result<Self, KernelError<F>> {
        let mut kernel = Self::unfilled(carry, inputs)?;
        let dense_round = cap.crossover(4 * size_of::<F>(), kernel.challenges.total());
        kernel.tables = OuterTables::Streamed { dense_round };
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
        let [az, bz] = tables;
        let [az_tail, bz_tail] = suffix.halves;
        self.tables = OuterTables::Dense {
            az: PrefixColumn {
                values: az,
                tail: az_tail,
            },
            bz: PrefixColumn {
                values: bz,
                tail: bz_tail,
            },
        };
        Ok((zero, infinity))
    }

    /// The constant suffix of the current round's pairs. A pair past the
    /// rows' prefix holds the suffix row's two streams in round 0. In a later
    /// round both its halves hold those streams bound at the first challenge:
    /// the cycle weights of a half sum to one.
    fn pair_suffix(&self) -> Result<PairSuffix<F, 2>, WitnessError> {
        let (az, bz) = self.suffix_groups()?;
        let halves = match self.challenges.as_slice().first() {
            None => [az, bz],
            Some(&stream) => [az, bz].map(|[first, second]| [first + stream * (second - first); 2]),
        };
        Ok(PairSuffix {
            start: self.prefix_cycles,
            halves,
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
            self.round_pass(&suffix, None, |[az, bz]| [az[1] * bz[1]])
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

    /// Row weights of the current round's pass, one `[first, second]` group
    /// pair per cycle offset `c` within a pair half: the Lagrange weights in
    /// round 0, whose halves are the two streams; in round `j ≥ 1`, those
    /// weights times `eq(r_2..r_j, c)·eq(r_1, s)` for stream `s`, offset bit
    /// `i` pairing `r_{i+2}`. Round `j` thus holds `2^(j−1)` pairs.
    fn round_weights(&self) -> Vec<[[F; DOMAIN]; 2]> {
        let round = self.challenges.bound();
        let Some((&stream, cycle_challenges)) = self.challenges.as_slice()[..round].split_first()
        else {
            return vec![[self.lagrange; 2]];
        };
        let point: Vec<F> = cycle_challenges.iter().rev().copied().collect();
        EqPolynomial::<F>::evals(&point, None)
            .into_iter()
            .map(|eq| {
                let high = eq * stream;
                let low = eq - high;
                [
                    self.lagrange.map(|weight| low * weight),
                    self.lagrange.map(|weight| high * weight),
                ]
            })
            .collect()
    }

    /// One pass of the current round `j` over its pairs of `2^j` cycles
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
            self.challenges.bound(),
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

    /// One task's serial walk of `range` in the current round: each row folds
    /// into its pair's low or high half — round 0's halves are one cycle's two
    /// streams, round `j ≥ 1`'s the first and last `2^(j−1)` cycles of a
    /// `2^j`-cycle pair ([`PairHalves`]) — and `emit` receives every completed
    /// pair's `[Az, Bz]` halves in order.
    fn walk_pairs(
        &self,
        range: Range<usize>,
        chunk: usize,
        weights: &[[[F; DOMAIN]; 2]],
        mut emit: impl FnMut(usize, [[F; 2]; 2]),
    ) -> Result<(), WitnessError> {
        let round = self.challenges.bound();
        #[cfg(feature = "field-inline")]
        let mut field_cursor = FieldInlineRowCursor::seek(&self.field_rows, range.start);
        let mut halves = PairHalves::new(round.saturating_sub(1));
        self.rows.visit(range, chunk, |start, bundles| {
            for (t, row) in (start..).zip(bundles) {
                let [first, second] = &weights[halves.offset(t)];
                let [(az_first, bz_first), (az_second, bz_second)] = row.fold_streams(
                    #[cfg(feature = "field-inline")]
                    field_cursor.advance(t),
                    first,
                    second,
                );
                if round == 0 {
                    emit(t, [[az_first, az_second], [bz_first, bz_second]]);
                } else if let Some((pair, pair_halves)) =
                    halves.add(t, [az_first + az_second, bz_first + bz_second])
                {
                    emit(pair, pair_halves);
                }
            }
            Ok(())
        })
    }
}
