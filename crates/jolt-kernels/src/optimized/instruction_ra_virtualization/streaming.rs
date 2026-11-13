//! Streaming instruction RA virtualization: the default kernel over selectors
//! gathered from the witness's resident rows ([`StreamedFoldedRa`]) instead of
//! from the packed row cache the default kernels share. Until the family's
//! materializing bind, every round evaluates the default kernel's summand on
//! each pair row of the padded cycle domain; from that bind on, the default
//! kernel's rounds run over the family's dense state.

use jolt_field::{Accumulator, JoltField};
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::SumcheckError;
use jolt_verifier::stages::stage6b::instruction_ra_virtualization::InstructionRaVirtualization;
use jolt_witness::witnesses::LookupIndex;
use jolt_witness::{JoltWitnessPlane, WitnessBundle, WitnessError};

use super::{OptimizedInstructionRaVirtualizationKernel, Selectors};
use crate::optimized::streaming::{RowIndexSource, StreamedFoldedRa, MATERIALIZE_BYTES};
use crate::optimized::support::{
    accumulate_product_grid, eq_round_error, product_grid_scratch_len, BundleStore,
    GruenRoundMessage,
};
use crate::{KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel};

/// The streaming stage-6b instruction RA virtualization slot.
pub struct StreamingInstructionRaVirtualization;

impl<F: JoltField> PrepareKernel<F, InstructionRaVirtualization<F>>
    for StreamingInstructionRaVirtualization
{
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, InstructionRaVirtualization<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = InstructionRaVirtualization<F>>>, KernelError<F>>
    {
        let relation = inputs.relation;
        let rows = BundleStore::resident(
            witness,
            relation.dimensions().log_t(),
            "instruction RA virtualization trace rows",
        )?;
        Ok(Box::new(
            OptimizedInstructionRaVirtualizationKernel::streamed(
                relation,
                inputs.challenges.gamma,
                rows,
                MATERIALIZE_BYTES,
            )?,
        ))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, WitnessBundle)]
pub(super) struct LookupIndexRow {
    pub(super) lookup_index: LookupIndex,
}

/// Column `i` is the `i`-th most significant `bits`-bit digit of a cycle's
/// lookup index, always hot.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
pub(super) struct LookupIndexDigits {
    columns: usize,
    /// At most 32: the kernel's constructor refuses wider chunks.
    bits: usize,
}

impl RowIndexSource for LookupIndexDigits {
    type Row = LookupIndexRow;

    #[inline]
    fn for_each_hot(&self, row: &LookupIndexRow, mut visit: impl FnMut(usize, usize)) {
        let mask = (1u128 << self.bits) - 1;
        let mut index = row.lookup_index.0;
        for column in (0..self.columns).rev() {
            visit(column, (index & mask) as usize);
            index >>= self.bits;
        }
    }
}

pub(super) fn rows_unavailable<F: JoltField>(_: WitnessError) -> SumcheckError<F> {
    SumcheckError::MissingEvaluationSource {
        kind: "instruction RA virtualization trace rows",
    }
}

impl<F: JoltField> OptimizedInstructionRaVirtualizationKernel<F> {
    /// The kernel over selectors gathered from `rows`, which span the
    /// relation's cycle domain; each selector materializes within
    /// `materialize_bytes`, and production passes [`MATERIALIZE_BYTES`].
    pub(super) fn streamed(
        relation: &InstructionRaVirtualization<F>,
        gamma: F,
        rows: BundleStore<LookupIndexRow>,
        materialize_bytes: usize,
    ) -> Result<Self, KernelError<F>> {
        Self::with_selectors(relation, gamma, |tables| {
            let source = LookupIndexDigits {
                columns: tables.len(),
                bits: relation.committed_chunk_bits(),
            };
            let selectors = StreamedFoldedRa::new(
                tables,
                source,
                rows,
                relation.dimensions().log_t(),
                materialize_bytes,
            );
            // A one-cycle domain has no bind to materialize at.
            Ok(match selectors.materialized()? {
                Some(dense) => Selectors::Lazy(dense),
                None => Selectors::Streamed(selectors),
            })
        })
    }

    /// A round before the selectors materialize: the default round's `q`
    /// samples, summed over every pair row from passes over the rows. A
    /// failed `q(0)` pass is held and returned, so its placeholder zero never
    /// reaches a round message.
    pub(super) fn streamed_message(
        &self,
        selectors: &StreamedFoldedRa<F, LookupIndexDigits>,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        let n = self.num_committed_per_virtual;
        let active = self.active_virtuals;
        let gruen_round = self.gruen.round().ok_or_else(|| eq_round_error(round))?;
        if n < 2 {
            let mut q_evals = selectors
                .sum_pair_terms(
                    &self.gruen,
                    3,
                    || (),
                    |(), lo, hi, terms| {
                        terms.fill(F::zero());
                        for (lo, hi) in lo.iter().zip(hi).take(active) {
                            terms[0] += *lo;
                            terms[1] += *hi;
                            terms[2] += *hi + *hi - *lo;
                        }
                    },
                )
                .map_err(rows_unavailable)?;
            return gruen_round.checked_round_poly(&mut q_evals, previous_claim, round);
        }

        struct Scratch<F: JoltField> {
            row_lanes: Vec<F::Accumulator>,
            evals: Vec<F>,
            steps: Vec<F>,
            grid: Vec<F>,
        }

        let q_evals = selectors
            .sum_pair_terms(
                &self.gruen,
                n,
                || Scratch::<F> {
                    row_lanes: vec![F::Accumulator::default(); n],
                    evals: vec![F::zero(); n],
                    steps: vec![F::zero(); n],
                    grid: vec![F::zero(); product_grid_scratch_len(n)],
                },
                |scratch, lo, hi, terms| {
                    scratch.row_lanes.fill(F::Accumulator::default());
                    for (lo, hi) in lo.chunks_exact(n).zip(hi.chunks_exact(n)).take(active) {
                        for (((eval, step), lo), hi) in scratch
                            .evals
                            .iter_mut()
                            .zip(&mut scratch.steps)
                            .zip(lo)
                            .zip(hi)
                        {
                            *eval = *hi;
                            *step = *hi - *lo;
                        }
                        accumulate_product_grid(
                            &scratch.evals,
                            &scratch.steps,
                            &mut scratch.row_lanes,
                            &mut scratch.grid,
                        );
                    }
                    for (term, lane) in terms.iter_mut().zip(&scratch.row_lanes) {
                        *term = lane.reduce();
                    }
                },
            )
            .map_err(rows_unavailable)?;
        let mut failure = None;
        let message = gruen_round.checked_toom(&q_evals, previous_claim, round, || {
            selectors
                .sum_pair_terms(
                    &self.gruen,
                    1,
                    || (),
                    |(), lo, _, terms| {
                        terms[0] = lo
                            .chunks_exact(n)
                            .take(active)
                            .map(|factors| factors.iter().copied().product::<F>())
                            .sum();
                    },
                )
                .map_or_else(
                    |error| {
                        failure = Some(error);
                        F::zero()
                    },
                    |terms| terms[0],
                )
        });
        match failure {
            Some(error) => Err(rows_unavailable(error)),
            None => message,
        }
    }
}
