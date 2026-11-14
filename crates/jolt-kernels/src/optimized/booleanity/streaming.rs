//! Streaming booleanity kernels: the default kernels over the witness's
//! resident rows instead of the packed row cache the default kernels share.
//! The address phase accumulates its reference-cycle masses from the rows.
//! The cycle phase gathers the checked polynomials' address-folded rows from
//! them ([`StreamedFoldedRa`]): until the family's materializing bind, every
//! round evaluates the default kernel's summand on each pair row of the
//! padded cycle domain; from that bind on, the default kernel's rounds run
//! over the family's dense state.

use jolt_field::{Accumulator, JoltField};
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::SumcheckError;
use jolt_verifier::stages::stage6a::booleanity::BooleanityAddressPhase;
use jolt_verifier::stages::stage6b::booleanity::Booleanity;
use jolt_witness::{JoltWitnessPlane, WitnessError};

use super::{
    ColumnSelector, CycleTables, OptimizedBooleanityAddressKernel, OptimizedBooleanityCycleKernel,
    SelectedValues,
};
use crate::optimized::one_hot_histograms::{pushforward_access, HistogramRow};
use crate::optimized::streaming::{RowIndexSource, StreamedFoldedRa, MATERIALIZE_BYTES};
use crate::optimized::support::{eq_round_error, BundleStore, GruenRoundMessage};
use crate::{KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel};

impl SelectedValues for HistogramRow {
    #[inline]
    fn lookup_index(&self) -> u128 {
        self.lookup_index.0
    }

    #[inline]
    fn bytecode_pc(&self) -> usize {
        self.bytecode_pc.0
    }

    #[inline]
    fn remapped_ram_address(&self) -> Option<u64> {
        self.remapped_ram_address.0
    }
}

/// The streaming stage-6a booleanity address slot.
pub struct StreamingBooleanityAddress;

impl<F: JoltField> PrepareKernel<F, BooleanityAddressPhase<F>> for StreamingBooleanityAddress {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, BooleanityAddressPhase<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = BooleanityAddressPhase<F>>>, KernelError<F>>
    {
        let log_t = inputs.relation.dimensions().log_t;
        Ok(Box::new(OptimizedBooleanityAddressKernel::prepare(
            witness,
            &inputs,
            |families, point, k_chunk| {
                let rows = BundleStore::<HistogramRow>::resident(
                    witness,
                    log_t,
                    "booleanity address trace rows",
                )?;
                let access = rows.access();
                // The rows span `2^log_t` cycles, so the shift cannot
                // overflow. The pushforward weighs the suffix from `prefix`
                // on by its row there, so `prefix` stays inside the domain.
                let last = (1usize << log_t) - 1;
                let prefix = access.constant_suffix_start(last + 1)?.min(last);
                Ok(pushforward_access(
                    |index| access.row(index),
                    prefix,
                    point,
                    families,
                    k_chunk,
                    true,
                )?)
            },
        )?))
    }
}

/// The streaming stage-6b booleanity cycle slot.
pub struct StreamingBooleanityCycle;

impl<F: JoltField> PrepareKernel<F, Booleanity<F>> for StreamingBooleanityCycle {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, Booleanity<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = Booleanity<F>>>, KernelError<F>> {
        let rows = BundleStore::resident(
            witness,
            inputs.relation.dimensions().log_t,
            "booleanity cycle trace rows",
        )?;
        Ok(Box::new(OptimizedBooleanityCycleKernel::streamed(
            witness,
            &inputs,
            rows,
            MATERIALIZE_BYTES,
        )?))
    }
}

/// The checked polynomials' chunk selectors over a cycle's resident row.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
pub(super) struct ResidentChunks {
    #[cfg_attr(feature = "allocative", allocative(visit = crate::backend::visit_heap_free_elements))]
    selectors: Vec<ColumnSelector>,
}

impl RowIndexSource for ResidentChunks {
    type Row = HistogramRow;

    #[inline]
    fn for_each_hot(&self, row: &HistogramRow, mut visit: impl FnMut(usize, usize)) {
        for (column, selector) in self.selectors.iter().enumerate() {
            if let Some(index) = selector.index(row) {
                visit(column, index);
            }
        }
    }
}

pub(super) fn rows_unavailable<F: JoltField>(_: WitnessError) -> SumcheckError<F> {
    SumcheckError::MissingEvaluationSource {
        kind: "booleanity cycle trace rows",
    }
}

impl<F: JoltField> OptimizedBooleanityCycleKernel<F> {
    /// The kernel over address-folded rows gathered from `rows`, which span
    /// the relation's cycle domain; each checked polynomial materializes
    /// within `materialize_bytes`, and production passes
    /// [`MATERIALIZE_BYTES`].
    pub(super) fn streamed(
        witness: &dyn JoltWitnessPlane<F>,
        inputs: &ProverInputs<'_, F, Booleanity<F>>,
        rows: BundleStore<HistogramRow>,
        materialize_bytes: usize,
    ) -> Result<Self, KernelError<F>> {
        let log_t = inputs.relation.dimensions().log_t;
        Self::prepare(witness, inputs, |selectors, tables| {
            let selectors = StreamedFoldedRa::new(
                tables,
                ResidentChunks { selectors },
                rows,
                log_t,
                materialize_bytes,
            );
            // A one-cycle domain has no bind to materialize at.
            Ok(match selectors.materialized()? {
                Some(dense) => CycleTables::Lazy(dense),
                None => CycleTables::Streamed(selectors),
            })
        })
    }

    /// A round before the rows materialize: the default round's constant and
    /// leading coefficients of `q`, summed over every pair row from one pass
    /// over the rows. A failed `q(1)` pass is held and returned, so its
    /// placeholder zero never reaches a round message.
    pub(super) fn streamed_round(
        &self,
        selectors: &StreamedFoldedRa<F, ResidentChunks>,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        let gamma_powers = &self.gamma_powers;
        let active = gamma_powers.iter().take_while(|rho| !rho.is_zero()).count();
        let q = selectors
            .sum_pair_terms(
                &self.eq,
                2,
                || (),
                |(), lo, hi, terms| {
                    let mut constant = F::Accumulator::default();
                    let mut leading = F::Accumulator::default();
                    for ((h_0, h_1), rho) in lo.iter().zip(hi).zip(gamma_powers).take(active) {
                        let delta = *h_1 - *h_0;
                        constant.fmadd(*h_0, *h_0 - *rho);
                        leading.fmadd(delta, delta);
                    }
                    terms[0] = constant.reduce();
                    terms[1] = leading.reduce();
                },
            )
            .map_err(rows_unavailable)?;
        let mut failure = None;
        let message = self
            .eq
            .round()
            .ok_or_else(|| eq_round_error(round))?
            .checked_cubic(q[0], q[1], previous_claim, round, || {
                selectors
                    .sum_pair_terms(
                        &self.eq,
                        1,
                        || (),
                        |(), _, hi, terms| {
                            let mut value = F::Accumulator::default();
                            for (h_1, rho) in hi.iter().zip(gamma_powers).take(active) {
                                value.fmadd(*h_1, *h_1 - *rho);
                            }
                            terms[0] = value.reduce();
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
