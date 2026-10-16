//! Optimized booleanity kernels (stage 6a address phase + stage 6b cycle
//! phase), porting the legacy `BooleanityAddressSumcheckProver` /
//! `BooleanityCycleSumcheckProver` techniques behind the modular seam.
//!
//! The summand (shared with the reference kernels, two-phase split):
//!
//! `0 = Σ_{k,j} eq(r_ref_addr, k) · eq(r_ref_cycle, j) · Σ_i γ^{2i} · (ra_i(k,j)² − ra_i(k,j))`
//!
//! Ported techniques, and why they are exact (byte parity with the
//! reference tier holds because field arithmetic is exact — any correct
//! regrouping of the same sums/products yields identical field elements,
//! and deferred reduction is exact mod `p`):
//!
//! - **Sparse one-hot access.** Each `ra_i(·, j)` is one-hot in the chunk
//!   domain, so the per-cycle hot index is a chunk of that cycle's lookup
//!   index / mapped PC / remapped RAM address. Both phases gather through
//!   [`RaChunkSelector`]s over packed rows decoded block by block from the
//!   witness's row handle (`CycleRows`) and never materialize the `K × T`
//!   grids the naive tier's `oracle_table` walks, nor per-polynomial index
//!   columns (legacy `RaIndices`).
//! - **Pushforward `G` tables (address phase).** `G_i[k] = Σ_j
//!   eq(r_ref_cycle, j) · ra_i(k, j)` collapses to a scatter of the
//!   tensor-factored eq weights into a `K`-sized accumulator per
//!   polynomial: per `E_out` block, the `E_in` weights accumulate
//!   *unreduced* into the hot buckets (no per-cycle multiply at all), and
//!   each block reduces and folds by its `e_out` once (legacy
//!   `compute_all_G`).
//! - **Shared expanded-eq gather (cycle phase).** The address-folded row
//!   `x_i(j) = eq(r_address)[hot_i(j)]` is a lookup into a `K`-sized table
//!   pre-scaled by `γ^i`, so `γ^{2i}(x² − x) = H(H − γ^i)` needs no
//!   batching multiply in the round loop (legacy `SharedRaPolynomials`
//!   pre-scaling), served by the shared `LazyFoldedRa` state machine:
//!   index-encoded until its byte-budgeted materialization bind, dense after.
//! - **Categorical startup products.** After address binding, values still
//!   come from the address table (or zero on cold RAM rows). In the first
//!   two cycle rounds, sufficiently small alphabets let us precompute the
//!   quadratic coefficients once per category pair instead of multiplying
//!   each cycle's gathered values. Large alphabets and small traces retain
//!   the original deferred-product loop.
//! - **Split-eq / Gruen round messages (cycle phase).** Only the constant
//!   and leading coefficients of the inner quadratic are accumulated, in
//!   deferred-reduction lanes at every level (per-row products, per-block
//!   `e_in` folds, cross-block `e_out` folds); the cubic is reconstructed
//!   from `s(0)+s(1) = previous_claim` via
//!   [`GruenSplitEqPolynomial::gruen_poly_deg_3`]. The stage-6a
//!   `eq(r_address, reference_address)` scalar rides in the split-eq
//!   scaling factor, mirroring the reference's `EqAddressCycle` derived
//!   table exactly.
//!
//! The address phase's round loop runs over the `2^log_k_chunk`-point chunk
//! domain (16–256 points) — negligible next to the `T`-scale table
//! construction — so it reuses the reference kernel's explicit four-point
//! sampling verbatim; only the table construction is replaced.

use std::collections::BTreeMap;
use std::ops::Range;

use jolt_claims::protocols::jolt::geometry::booleanity::BooleanityDimensions;
use jolt_claims::protocols::jolt::geometry::ra::JoltRaPolynomial;
#[cfg(feature = "akita")]
use jolt_claims::protocols::jolt::lattice::relations::booleanity::{
    lattice_booleanity_output_openings, LatticeBooleanityDimensions,
};
#[cfg(feature = "akita")]
use jolt_claims::protocols::jolt::lattice::BalancedIncChunking;
#[cfg(not(feature = "akita"))]
use jolt_claims::protocols::jolt::JoltRelationId;
use jolt_claims::protocols::jolt::{BooleanityPublic, JoltDerivedId, JoltOpeningId};
use jolt_claims::OutputClaims;
use jolt_field::{Accumulator, JoltField};
use jolt_poly::{
    try_eq_mle, BindingOrder, GruenSplitEqPolynomial, Polynomial, TensorEqTable, UnivariatePoly,
};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::{
    ConcreteSumcheck, SumcheckInputClaims, SumcheckInputPoints, SumcheckOutputClaims,
    SumcheckOutputPoints,
};
use jolt_verifier::stages::stage6a::booleanity::{
    BooleanityAddressPhase, BooleanityAddressPhaseChallenges, BooleanityAddressPhaseOutputClaims,
};
use jolt_verifier::stages::stage6b::booleanity::{Booleanity, BooleanityCyclePhaseChallenges};
#[cfg(feature = "akita")]
use jolt_witness::witnesses::BalancedIncColumn;
use jolt_witness::witnesses::RaChunkSelector;
use jolt_witness::{JoltWitnessPlane, WitnessError};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::instruction_read_raf::{CycleRows, InstructionCycleRow};
use super::lazy_ra::{
    try_fold_groups, visit_source_rows, ChunkIndexSource, LazyFoldedRa, SourceBlock,
    MATERIALIZE_BYTES,
};
use super::support::{
    gamma_powers, pin_derived_term_if_derived, task_ranges, GruenRoundMessage, RoundProgress,
};
use crate::reference::views::eq_table;
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

/// One checked polynomial's chunk selector over the packed per-cycle rows
/// (canonical layout order: instruction, bytecode, ram).
enum ColumnSelector {
    Instruction(RaChunkSelector),
    Bytecode(RaChunkSelector),
    Ram(RaChunkSelector),
    #[cfg(feature = "akita")]
    UnsignedInc(BalancedIncColumn),
}

impl ColumnSelector {
    /// The selected row at `row`; `None` is a cold cycle. Mirrors the
    /// trace oracle's grid materializers (`materialize_one_hot`), so
    /// gathered indices and the reference tier's dense grids describe the
    /// same one-hot polynomials.
    #[inline]
    fn index(&self, row: &InstructionCycleRow) -> Option<usize> {
        match self {
            Self::Instruction(selector) => Some(selector.chunk_u128(row.lookup_index())),
            Self::Bytecode(selector) => Some(selector.chunk_usize(row.bytecode_pc())),
            Self::Ram(selector) => row
                .remapped_ram_address()
                .map(|address| selector.chunk_usize(address as usize)),
            #[cfg(feature = "akita")]
            Self::UnsignedInc(column) => Some(row.fused_inc_row(*column)),
        }
    }
}

struct BooleanityColumns {
    openings: Vec<JoltOpeningId>,
    selectors: Vec<ColumnSelector>,
}

impl BooleanityColumns {
    fn openings<F: JoltField>(
        dimensions: BooleanityDimensions,
    ) -> Result<Vec<JoltOpeningId>, KernelError<F>> {
        #[cfg(not(feature = "akita"))]
        {
            Ok(dimensions
                .layout
                .openings(JoltRelationId::Booleanity)
                .collect())
        }
        #[cfg(feature = "akita")]
        {
            let lattice_dimensions =
                LatticeBooleanityDimensions::new(dimensions).map_err(|_| {
                    KernelError::InvariantViolation {
                        reason: "the packed shape requires a lattice-compatible chunk width",
                    }
                })?;
            Ok(lattice_booleanity_output_openings(lattice_dimensions))
        }
    }

    fn new<F: JoltField>(
        witness: &dyn JoltWitnessPlane<F>,
        dimensions: BooleanityDimensions,
    ) -> Result<Self, KernelError<F>> {
        let log_t = dimensions.log_t;
        let log_k_chunk = dimensions.log_k_chunk;
        let layout = dimensions.layout;
        let openings = Self::openings(dimensions)?;
        for opening in &openings {
            let shape = witness.shape(opening.polynomial_id())?;
            if shape.log_rows != log_k_chunk + log_t {
                return Err(KernelError::TableSizeMismatch {
                    table: format!("{opening:?}"),
                    expected: 1usize << (log_k_chunk + log_t),
                    got: shape.rows(),
                });
            }
        }
        let selectors = layout
            .polynomials()
            .map(|polynomial| {
                Ok(match polynomial {
                    JoltRaPolynomial::Instruction(index) => ColumnSelector::Instruction(
                        RaChunkSelector::new(index, layout.instruction(), log_k_chunk)?,
                    ),
                    JoltRaPolynomial::Bytecode(index) => ColumnSelector::Bytecode(
                        RaChunkSelector::new(index, layout.bytecode(), log_k_chunk)?,
                    ),
                    JoltRaPolynomial::Ram(index) => {
                        ColumnSelector::Ram(RaChunkSelector::new(index, layout.ram(), log_k_chunk)?)
                    }
                })
            })
            .collect::<Result<Vec<_>, KernelError<F>>>()?;
        #[cfg(feature = "akita")]
        let mut selectors = selectors;
        #[cfg(feature = "akita")]
        {
            let chunking = BalancedIncChunking::new(log_k_chunk).map_err(|_| {
                KernelError::InvariantViolation {
                    reason: "the packed shape requires a lattice-compatible chunk width",
                }
            })?;
            selectors.extend((0..chunking.chunk_count()).map(|index| {
                ColumnSelector::UnsignedInc(BalancedIncColumn::Digit {
                    width: log_k_chunk,
                    index,
                })
            }));
            selectors.push(ColumnSelector::UnsignedInc(BalancedIncColumn::Carry {
                width: log_k_chunk,
            }));
        }
        debug_assert_eq!(openings.len(), selectors.len());
        Ok(Self {
            openings,
            selectors,
        })
    }
}

/// The one-hot pushforward `G_i[k] = Σ_{j : hot_i(j) = k} eq(point, j)`
/// (legacy `compute_all_G` / `one_hot_pushforwards`): per `E_out` block,
/// the `E_in` weights scatter into per-polynomial `K`-sized
/// deferred-reduction buckets (an unreduced add per hot polynomial, no
/// per-cycle multiply); each block then reduces its touched buckets and
/// folds them by `e_out` into the running partials. Equals the reference's
/// per-chunk cycle masses exactly — same terms, regrouped through the
/// `eq = E_out ⊗ E_in` factorization. At most one task per pool thread
/// visits whole `E_out` blocks ([`task_ranges`]) once, serially, one block
/// per callback.
fn cycle_pushforward<F: JoltField>(
    rows: &CycleRows,
    selectors: &[ColumnSelector],
    k_chunk: usize,
    point: &[F],
) -> Result<Vec<Vec<F>>, WitnessError> {
    let eq = TensorEqTable::new(point);
    let e_out = eq.e_out();
    let e_in = eq.e_in();
    let in_len = e_in.len();

    struct State<F: JoltField> {
        partial: Vec<Vec<F::Accumulator>>,
        block: Vec<Vec<F::Accumulator>>,
    }
    let zero = || State::<F> {
        partial: vec![vec![F::Accumulator::default(); k_chunk]; selectors.len()],
        block: vec![vec![F::Accumulator::default(); k_chunk]; selectors.len()],
    };
    let scatter = |state: &mut State<F>, x_out: usize, block_rows: &[InstructionCycleRow]| {
        for (row, e) in block_rows.iter().zip(e_in) {
            for (selector, block) in selectors.iter().zip(state.block.iter_mut()) {
                if let Some(k) = selector.index(row) {
                    block[k].add(*e);
                }
            }
        }
        let e_out = e_out[x_out];
        for (partial, block) in state.partial.iter_mut().zip(state.block.iter_mut()) {
            for (partial, block) in partial.iter_mut().zip(block.iter_mut()) {
                let value = std::mem::take(block).reduce();
                if value != F::zero() {
                    partial.fmadd(e_out, value);
                }
            }
        }
    };
    let finish = |state: State<F>| -> Vec<Vec<F>> {
        state
            .partial
            .into_iter()
            .map(|buckets| buckets.into_iter().map(|bucket| bucket.reduce()).collect())
            .collect()
    };
    let merge = |mut left: State<F>, right: State<F>| {
        for (left, right) in left.partial.iter_mut().zip(right.partial) {
            for (left, right) in left.iter_mut().zip(right) {
                left.merge(right);
            }
        }
        left
    };
    let task = |range: &Range<usize>| -> Result<State<F>, WitnessError> {
        let mut state = zero();
        rows.visit_blocks(range.clone(), in_len, |first, block_rows| {
            scatter(&mut state, first / in_len, block_rows);
            Ok(())
        })?;
        Ok(state)
    };

    let ranges = task_ranges(eq.len(), rows.segment_len(), in_len);
    #[cfg(feature = "parallel")]
    let state = ranges
        .par_iter()
        .map(task)
        .try_reduce(zero, |left, right| Ok(merge(left, right)))?;
    #[cfg(not(feature = "parallel"))]
    let state = ranges.iter().map(task).try_fold(zero(), |left, right| {
        Ok::<_, WitnessError>(merge(left, right?))
    })?;
    Ok(finish(state))
}

pub struct OptimizedBooleanityAddress;

impl<F: JoltField> PrepareKernel<F, BooleanityAddressPhase<F>> for OptimizedBooleanityAddress {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, BooleanityAddressPhase<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = BooleanityAddressPhase<F>>>, KernelError<F>>
    {
        let relation = inputs.relation;
        let dimensions = relation.dimensions();
        let BooleanityAddressPhaseChallenges {
            reference_address,
            gamma,
        } = inputs.challenges;
        let reference_cycle = relation.reference_cycle();
        if reference_address.len() != dimensions.log_k_chunk
            || reference_cycle.len() != dimensions.log_t
        {
            return Err(KernelError::InvariantViolation {
                reason: "booleanity reference point lengths disagree with the dimensions",
            });
        }

        let columns = BooleanityColumns::new(witness, dimensions)?;
        let rows = witness.rows().ok_or(WitnessError::UnavailableView {
            label: "stage-6a booleanity rows",
        })?;
        let masses = cycle_pushforward(
            &CycleRows::Trace(rows),
            &columns.selectors,
            1usize << dimensions.log_k_chunk,
            &reference_cycle,
        )?;

        Ok(Box::new(OptimizedBooleanityAddressKernel::new(
            relation.rounds(),
            *gamma,
            reference_address,
            masses,
        )))
    }
}

/// The address-phase kernel: the reference kernel's round machinery over
/// pushforward-built tables. The linear term binds `A_i[k] = G_i[k]` as a
/// plain multilinear; the squared term binds `B_i[k]` (same initial masses)
/// with squared weights, because binding squares the one-hot's accumulated
/// eq factor. The initial `A = B` makes the input claim exactly zero.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
struct OptimizedBooleanityAddressKernel<F: JoltField> {
    progress: RoundProgress,
    /// Per checked polynomial, its `γ^{2i}` batching weight, in the layout's
    /// canonical order.
    gamma_weights: Vec<F>,
    linear: Vec<Polynomial<F>>,
    /// Raw vectors because the squared-weight bind is not a multilinear bind.
    squared: Vec<Vec<F>>,
    eq_address: Polynomial<F>,
}

impl<F: JoltField> OptimizedBooleanityAddressKernel<F> {
    fn new(rounds: usize, gamma: F, reference_address: &[F], masses: Vec<Vec<F>>) -> Self {
        let linear: Vec<Polynomial<F>> = masses.into_iter().map(Polynomial::new).collect();
        let squared: Vec<Vec<F>> = linear.iter().map(|table| table.evals().to_vec()).collect();
        let mut gamma_weights = Vec::with_capacity(linear.len());
        let mut weight = F::one();
        let gamma_sqr = gamma * gamma;
        for _ in 0..linear.len() {
            gamma_weights.push(weight);
            weight *= gamma_sqr;
        }
        Self {
            progress: RoundProgress::new(rounds),
            gamma_weights,
            linear,
            squared,
            eq_address: Polynomial::new(eq_table(reference_address)),
        }
    }

    fn bind(&mut self, challenge: F) {
        let one_minus_sqr = (F::one() - challenge) * (F::one() - challenge);
        let challenge_sqr = challenge * challenge;
        for table in &mut self.linear {
            table.bind_with_order(challenge, BindingOrder::LowToHigh);
        }
        for table in &mut self.squared {
            let half = table.len() / 2;
            for k in 0..half {
                table[k] = one_minus_sqr * table[2 * k] + challenge_sqr * table[2 * k + 1];
            }
            table.truncate(half);
        }
        self.eq_address
            .bind_with_order(challenge, BindingOrder::LowToHigh);
        self.progress.advance();
    }
}

impl<F: JoltField> ProveRounds<F> for OptimizedBooleanityAddressKernel<F> {
    fn num_rounds(&self) -> usize {
        self.progress.total()
    }

    fn prove_round(
        &mut self,
        bind: Option<F>,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if let Some(challenge) = bind {
            self.bind(challenge);
        }
        let half = self.eq_address.evals().len() / 2;
        let mut evals = [F::zero(); 4];
        for (c, eval) in evals.iter_mut().enumerate() {
            let point = F::from_u64(c as u64);
            let point_sqr = point * point;
            let one_minus_sqr = (F::one() - point) * (F::one() - point);
            let mut sum = F::zero();
            for y in 0..half {
                let mut inner = F::zero();
                for ((weight, linear), squared) in self
                    .gamma_weights
                    .iter()
                    .zip(&self.linear)
                    .zip(&self.squared)
                {
                    let squared_ext =
                        one_minus_sqr * squared[2 * y] + point_sqr * squared[2 * y + 1];
                    let linear_ext =
                        linear.sumcheck_round_eval_with_order(y, point, BindingOrder::LowToHigh);
                    inner += *weight * (squared_ext - linear_ext);
                }
                sum += self.eq_address.sumcheck_round_eval_with_order(
                    y,
                    point,
                    BindingOrder::LowToHigh,
                ) * inner;
            }
            *eval = sum;
        }

        let round_sum = evals[0] + evals[1];
        if round_sum != previous_claim {
            return Err(SumcheckError::RoundCheckFailed {
                round,
                expected: previous_claim,
                actual: round_sum,
            });
        }
        Ok(UnivariatePoly::from_evals(&evals))
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind);
        Ok(())
    }
}

impl<F: JoltField> SumcheckKernel<F> for OptimizedBooleanityAddressKernel<F> {
    type Relation = BooleanityAddressPhase<F>;

    fn output_claims(
        &mut self,
        _inputs: &SumcheckInputClaims<F, Self::Relation>,
    ) -> Result<BooleanityAddressPhaseOutputClaims<F>, SumcheckKernelError<F>> {
        self.progress.require_complete()?;
        let mut inner = F::zero();
        for ((weight, linear), squared) in self
            .gamma_weights
            .iter()
            .zip(&self.linear)
            .zip(&self.squared)
        {
            inner += *weight * (squared[0] - linear.evals()[0]);
        }
        Ok(BooleanityAddressPhaseOutputClaims {
            intermediate: self.eq_address.evals()[0] * inner,
        })
    }
}

pub struct OptimizedBooleanityCycle;

impl<F: JoltField> PrepareKernel<F, Booleanity<F>> for OptimizedBooleanityCycle {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, Booleanity<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = Booleanity<F>>>, KernelError<F>> {
        Ok(Box::new(OptimizedBooleanityCycleKernel::prepare(
            witness,
            &inputs,
            MATERIALIZE_BYTES,
        )?))
    }
}

impl<F: JoltField> OptimizedBooleanityCycleKernel<F> {
    /// `materialize_bytes` bounds each dense table ([`LazyFoldedRa::new`]);
    /// production passes [`MATERIALIZE_BYTES`].
    fn prepare(
        witness: &dyn JoltWitnessPlane<F>,
        inputs: &ProverInputs<'_, F, Booleanity<F>>,
        materialize_bytes: usize,
    ) -> Result<Self, KernelError<F>> {
        let relation = inputs.relation;
        let dimensions = relation.dimensions();
        let r_address = relation.r_address();
        let reference_address = relation.reference_address();
        let reference_cycle = relation.reference_cycle();
        if r_address.len() != dimensions.log_k_chunk || reference_cycle.len() != dimensions.log_t {
            return Err(KernelError::InvariantViolation {
                reason: "booleanity cycle-phase point lengths disagree with the dimensions",
            });
        }
        let columns = BooleanityColumns::new(witness, dimensions)?;
        let rows = witness.rows().ok_or(WitnessError::UnavailableView {
            label: "stage-6b booleanity rows",
        })?;
        if rows.cycles() != 1 << dimensions.log_t {
            return Err(KernelError::TableSizeMismatch {
                table: "stage-6b booleanity rows".to_owned(),
                expected: 1 << dimensions.log_t,
                got: rows.cycles(),
            });
        }

        // The fixed address eq factor of the `EqAddressCycle` public; rides
        // in the split-eq scaling so round messages and the bound scalar
        // carry it exactly like the reference's derived table.
        let address_scalar = try_eq_mle(r_address, reference_address).map_err(|_| {
            KernelError::InvariantViolation {
                reason: "booleanity address point and reference length mismatch",
            }
        })?;
        let eq_address = eq_table(r_address);
        let count = columns.selectors.len();
        let opening_unscale = inputs.challenges.gamma.inverse().map_or_else(
            || vec![F::one(); count],
            |gamma_inv| gamma_powers(gamma_inv, count),
        );
        let gamma_powers = gamma_powers(inputs.challenges.gamma, count);
        // Zero gamma disables all but the first summand, but every RA opening is
        // still required. Keep disabled columns unscaled rather than losing them.
        let tables: Vec<Vec<F>> = gamma_powers
            .iter()
            .map(|rho| {
                if rho.is_zero() {
                    eq_address.clone()
                } else {
                    eq_address.iter().map(|eq| *rho * *eq).collect()
                }
            })
            .collect();

        Ok(Self {
            progress: RoundProgress::new(relation.rounds()),
            eq: GruenSplitEqPolynomial::new_with_scaling(
                reference_cycle,
                BindingOrder::LowToHigh,
                Some(address_scalar),
            ),
            tables: LazyFoldedRa::new(
                tables,
                BooleanityChunks {
                    rows: CycleRows::Trace(rows),
                    selectors: columns.selectors,
                },
                materialize_bytes,
            ),
            gamma_powers,
            opening_unscale,
            openings: columns.openings,
        })
    }
}

#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
struct BooleanityChunks {
    rows: CycleRows,
    #[cfg_attr(feature = "allocative", allocative(visit = crate::backend::visit_heap_free_elements))]
    selectors: Vec<ColumnSelector>,
}

impl ChunkIndexSource for BooleanityChunks {
    type Row = InstructionCycleRow;

    fn num_columns(&self) -> usize {
        self.selectors.len()
    }

    fn cycles(&self) -> usize {
        self.rows.cycles()
    }

    #[inline]
    fn index(&self, i: usize, row: &InstructionCycleRow) -> Option<usize> {
        self.selectors[i].index(row)
    }

    fn visit_blocks(
        &self,
        range: Range<usize>,
        block: usize,
        visit: impl FnMut(usize, &[Self::Row]) -> Result<(), WitnessError>,
    ) -> Result<(), WitnessError> {
        self.rows.visit_blocks(range, block, visit)
    }
}

#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
struct OptimizedBooleanityCycleKernel<F: JoltField> {
    progress: RoundProgress,
    /// Split-eq over the reference cycle, scaled by
    /// `eq(r_address, reference_address)` — together the reference's
    /// `EqAddressCycle` derived table.
    eq: GruenSplitEqPolynomial<F>,
    tables: LazyFoldedRa<F, BooleanityChunks>,
    gamma_powers: Vec<F>,
    opening_unscale: Vec<F>,
    #[cfg_attr(feature = "allocative", allocative(visit = crate::backend::visit_heap_free_elements))]
    openings: Vec<JoltOpeningId>,
}

impl<F: JoltField> OptimizedBooleanityCycleKernel<F> {
    fn bind(&mut self, challenge: F) -> Result<(), SumcheckError<F>> {
        self.eq.bind(challenge);
        self.tables.bind(challenge)?;
        self.progress.advance();
        Ok(())
    }
}

/// After address binding, every unbound value comes from a small alphabet.
/// Cache the two nonlinear coefficients over that alphabet, retaining the
/// existing exact split-equality fold. The extra category is a cold RAM row;
/// it must remain distinct from hot address zero, including after a bind.
struct CategoricalProducts<'a, F: JoltField, S> {
    source: &'a S,
    width: usize,
    radix: usize,
    states: usize,
    products: Vec<(Vec<F>, Vec<F>)>,
}

impl<'a, F: JoltField, S: ChunkIndexSource> CategoricalProducts<'a, F, S> {
    fn fits_budget(width: usize, addresses: usize, dense_len: usize) -> bool {
        if width != 1 && width != 2 {
            return false;
        }
        let Some(states) = addresses
            .checked_add(1)
            .and_then(|radix| radix.checked_pow(width as u32))
            .filter(|states| *states <= 289)
        else {
            return false;
        };
        // Cached products, construction temporary and lazy branches fit
        // within the family's dense allocation, `dense_len` entries per table.
        states * (states + 2) + width * addresses <= dense_len
    }

    fn new(tables: &'a [Vec<F>], width: usize, source: &'a S, rho: &[F]) -> Self {
        let addresses = tables[0].len() / width;
        let radix = addresses + 1;
        let states = radix.pow(width as u32);
        let products = tables
            .iter()
            .zip(rho)
            .take_while(|(_, rho)| !rho.is_zero())
            .map(|(table, rho)| {
                let values: Vec<F> = (0..states)
                    .map(|mut state| {
                        let mut value = F::zero();
                        for branch in 0..width {
                            let address = state % radix;
                            state /= radix;
                            if address < addresses {
                                value += table[branch * addresses + address];
                            }
                        }
                        value
                    })
                    .collect();
                let constants = values.iter().map(|v| *v * (*v - *rho)).collect();
                let mut squares = Vec::with_capacity(states * states);
                for lo in &values {
                    squares.extend(values.iter().map(|hi| (*hi - *lo).square()));
                }
                (constants, squares)
            })
            .collect();
        Self {
            source,
            width,
            radix,
            states,
            products,
        }
    }

    #[inline]
    fn state(&self, rows: &SourceBlock<'_, S>, family: usize, row: usize) -> usize {
        let first = rows
            .index(family, row * self.width)
            .unwrap_or(self.radix - 1);
        if self.width == 1 {
            first
        } else {
            first
                + self.radix
                    * rows
                        .index(family, row * self.width + 1)
                        .unwrap_or(self.radix - 1)
        }
    }

    /// [`try_fold_groups`] over the source rows of `eq`'s current pairs.
    fn try_fold<Inner, Outer: Send>(
        &self,
        eq: &GruenSplitEqPolynomial<F>,
        make_inner: impl Fn() -> Inner + Sync,
        step: impl Fn(&mut Inner, &SourceBlock<'_, S>, usize, F) + Sync,
        outer_step: impl Fn(usize, F, Inner) -> Outer + Sync,
        merge: impl Fn(Outer, Outer) -> Outer + Sync,
    ) -> Result<Outer, SumcheckError<F>> {
        try_fold_groups(
            eq,
            make_inner,
            |rows, groups| {
                visit_source_rows(self.source, self.width, rows, |rows, block| {
                    for row in rows {
                        groups.row(row, |inner, _, weight| step(inner, block, row, weight));
                    }
                    Ok(())
                })
            },
            outer_step,
            merge,
        )
    }

    fn lookup(
        &self,
        eq: &GruenSplitEqPolynomial<F>,
        round: usize,
        claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        let lanes = self.try_fold(
            eq,
            || [F::Accumulator::default(); 2],
            |lanes, rows, row, weight| {
                let mut constant = F::zero();
                let mut leading = F::zero();
                for (family, (constants, squares)) in self.products.iter().enumerate() {
                    let lo = self.state(rows, family, 2 * row);
                    let hi = self.state(rows, family, 2 * row + 1);
                    constant += constants[lo];
                    leading += squares[lo * self.states + hi];
                }
                lanes[0].fmadd(weight, constant);
                lanes[1].fmadd(weight, leading);
            },
            |_, weight, lanes| {
                let mut out = [F::Accumulator::default(); 2];
                out[0].fmadd(weight, lanes[0].reduce());
                out[1].fmadd(weight, lanes[1].reduce());
                out
            },
            |mut a, b| {
                a[0].merge(b[0]);
                a[1].merge(b[1]);
                a
            },
        )?;
        // A failed fallback pass is returned; its placeholder never reaches a message.
        let mut failure = None;
        let message = eq.checked_cubic(lanes[0].reduce(), lanes[1].reduce(), claim, round, || {
            let q_at_one = self.try_fold(
                eq,
                F::zero,
                |sum, rows, row, weight| {
                    let value = self.products.iter().enumerate().fold(
                        F::zero(),
                        |value, (family, (constants, _))| {
                            value + constants[self.state(rows, family, 2 * row + 1)]
                        },
                    );
                    *sum += weight * value;
                },
                |_, weight, sum| weight * sum,
                |a, b| a + b,
            );
            q_at_one.unwrap_or_else(|error| {
                failure = Some(error);
                F::zero()
            })
        });
        failure.map_or(message, Err)
    }
}

impl<F: JoltField> ProveRounds<F> for OptimizedBooleanityCycleKernel<F> {
    fn num_rounds(&self) -> usize {
        self.progress.total()
    }

    fn prove_round(
        &mut self,
        bind: Option<F>,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if let Some(challenge) = bind {
            self.bind(challenge)?;
        }
        if let LazyFoldedRa::Lazy {
            tables,
            width,
            materialize_branches,
            factors: 1,
            source,
        } = &self.tables
        {
            if !tables.is_empty()
                && CategoricalProducts::<F, BooleanityChunks>::fits_budget(
                    *width,
                    tables[0].len() / width,
                    source.cycles() / *materialize_branches,
                )
            {
                let products = CategoricalProducts::new(tables, *width, source, &self.gamma_powers);
                return products.lookup(&self.eq, round, previous_claim);
            }
        }
        let tables = &self.tables;
        let gamma_powers = &self.gamma_powers;
        let num_polys = gamma_powers.len();
        let active_polys = gamma_powers.iter().take_while(|rho| !rho.is_zero()).count();

        struct Scratch<F: JoltField> {
            lanes: [F::Accumulator; 2],
            pairs: Vec<(F, F)>,
        }

        // Inner quadratic `q(X) = Σ_j eq_rest(j) · Σ_i (H_i(X)² − γ^i·H_i(X))`:
        // constant coefficient from `H` at 0, leading coefficient from the
        // pair delta — the pre-scaling makes `γ^{2i}(x² − x) = H(H − γ^i)`.
        // Per-row products accumulate unreduced, reduce once, and fold into
        // the block lanes by `e_in`; blocks fold by `e_out` (legacy
        // `par_fold_out_in_unreduced`).
        let block_lanes = tables.try_fold_out_in(
            &self.eq,
            || Scratch {
                lanes: [F::Accumulator::default(); 2],
                pairs: vec![(F::zero(), F::zero()); num_polys],
            },
            |scratch, block, row, _x_in, e_in| {
                block.lo_hi_all(row, &mut scratch.pairs);
                let mut constant = F::Accumulator::default();
                let mut leading = F::Accumulator::default();
                for ((h_0, h_1), rho) in scratch.pairs.iter().zip(gamma_powers).take(active_polys) {
                    let delta = *h_1 - *h_0;
                    constant.fmadd(*h_0, *h_0 - *rho);
                    leading.fmadd(delta, delta);
                }
                scratch.lanes[0].fmadd(e_in, constant.reduce());
                scratch.lanes[1].fmadd(e_in, leading.reduce());
            },
            |_x_out, e_out, scratch| {
                let mut out = [F::Accumulator::default(); 2];
                out[0].fmadd(e_out, scratch.lanes[0].reduce());
                out[1].fmadd(e_out, scratch.lanes[1].reduce());
                out
            },
            |mut a, b| {
                a[0].merge(b[0]);
                a[1].merge(b[1]);
                a
            },
        )?;
        // A failed fallback pass is returned; its placeholder never reaches a message.
        let mut failure = None;
        let message = self.eq.checked_cubic(
            block_lanes[0].reduce(),
            block_lanes[1].reduce(),
            previous_claim,
            round,
            || {
                let q_at_one = tables.try_fold_out_in(
                    &self.eq,
                    || {
                        (
                            vec![(F::zero(), F::zero()); num_polys],
                            F::Accumulator::default(),
                        )
                    },
                    |(pairs, sum), block, row, _, weight| {
                        block.lo_hi_all(row, pairs);
                        let mut value = F::Accumulator::default();
                        for ((_, hi), rho) in pairs.iter().zip(gamma_powers).take(active_polys) {
                            value.fmadd(*hi, *hi - *rho);
                        }
                        sum.fmadd(weight, value.reduce());
                    },
                    |_, weight, (_, sum)| weight * sum.reduce(),
                    |a, b| a + b,
                );
                q_at_one.unwrap_or_else(|error| {
                    failure = Some(error);
                    F::zero()
                })
            },
        );
        failure.map_or(message, Err)
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind)
    }
}

impl<F: JoltField> SumcheckKernel<F> for OptimizedBooleanityCycleKernel<F> {
    type Relation = Booleanity<F>;

    fn output_claims(
        &mut self,
        _inputs: &SumcheckInputClaims<F, Self::Relation>,
    ) -> Result<SumcheckOutputClaims<F, Self::Relation>, SumcheckKernelError<F>> {
        self.progress.require_complete()?;
        let finals = self.tables.final_values()?;
        // Disabled columns were retained unscaled; the unit factor preserves
        // their opening values when gamma is zero.
        let values: BTreeMap<JoltOpeningId, F> = self
            .openings
            .iter()
            .copied()
            .enumerate()
            .map(|(i, id)| (id, finals[i] * self.opening_unscale[i]))
            .collect();
        SumcheckOutputClaims::<F, Self::Relation>::from_opening_values(|id| values.get(id).copied())
            .map_err(SumcheckKernelError::from)
    }

    /// The split-eq scalar (fully bound `EqAddressCycle`) against the
    /// verifier's `derive_output_term` — the same drift detector the naive
    /// tier runs on its hand-materialized derived table.
    fn validate_derived_tables(
        &self,
        relation: &Self::Relation,
        input_points: &SumcheckInputPoints<F, Self::Relation>,
        output_points: &SumcheckOutputPoints<F, Self::Relation>,
        challenges: &BooleanityCyclePhaseChallenges<F>,
    ) -> Result<(), SumcheckKernelError<F>> {
        self.progress.require_complete()?;
        pin_derived_term_if_derived(
            relation,
            JoltDerivedId::from(BooleanityPublic::EqAddressCycle),
            input_points,
            output_points,
            challenges,
            self.eq.current_scalar(),
        )
    }
}

/// Trace-backed test fixtures shared by the optimized-kernel parity tests
/// (this module and `optimized::ram_hamming_booleanity`): a tiny consistent
/// trace behind a full witness plane, so the reference kernels' dense
/// `oracle_table` grids and the optimized kernels' typed bundle rows
/// describe the same witness by construction.
#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test fixture construction")]
pub(crate) mod testing {
    use jolt_claims::protocols::jolt::geometry::booleanity::BooleanityDimensions;
    use jolt_claims::protocols::jolt::geometry::ra::JoltRaPolynomialLayout;
    use jolt_claims::protocols::jolt::{
        JoltCommittedPolynomial, JoltOneHotConfig, JoltPolynomialId,
    };
    use jolt_field::Fr;
    use jolt_program::execution::{JoltProgram, TraceData, TraceOutput};
    use jolt_program::preprocess::{
        BytecodePreprocessing, JoltProgramPreprocessing, RAMPreprocessing,
    };
    use jolt_riscv::{JoltInstructionKind, JoltInstructionRow, NormalizedOperands, RV64IMAC_JOLT};
    use jolt_riscv::{
        JoltTraceRow as TraceRow, RamAccess, RamRead, RamWrite, RegisterRead, RegisterState,
        RegisterWrite,
    };
    use jolt_witness::{JoltVmWitnessConfig, JoltVmWitnessInputs, JoltWitnessOracle, TraceBackend};

    const LOAD: JoltInstructionRow = JoltInstructionRow {
        instruction_kind: JoltInstructionKind::LD,
        address: 0x8000_0000,
        operands: NormalizedOperands {
            rd: Some(1),
            rs1: Some(2),
            rs2: None,
            imm: 3,
        },
        virtual_sequence_remaining: None,
        is_first_in_sequence: false,
        is_compressed: false,
    };
    const STORE: JoltInstructionRow = JoltInstructionRow {
        instruction_kind: JoltInstructionKind::SD,
        address: 0x8000_0004,
        operands: NormalizedOperands {
            rd: None,
            rs1: Some(1),
            rs2: Some(3),
            imm: 8,
        },
        ..LOAD
    };
    const ALU: JoltInstructionRow = JoltInstructionRow {
        instruction_kind: JoltInstructionKind::ADDI,
        address: 0x8000_0008,
        operands: NormalizedOperands {
            rd: Some(1),
            rs1: Some(2),
            rs2: None,
            imm: 3,
        },
        ..LOAD
    };

    pub(crate) fn load_row(address: u64) -> TraceRow {
        TraceRow::new(
            LOAD,
            RegisterState {
                rs1: Some(RegisterRead {
                    register: 2,
                    value: 5,
                }),
                rd: Some(RegisterWrite {
                    register: 1,
                    pre_value: 0,
                    post_value: 8,
                }),
                ..Default::default()
            },
            RamAccess::Read(RamRead { address, value: 8 }),
            1,
        )
        .unwrap()
    }

    pub(crate) fn store_row(address: u64) -> TraceRow {
        TraceRow::new(
            STORE,
            RegisterState {
                rs1: Some(RegisterRead {
                    register: 1,
                    value: 8,
                }),
                rs2: Some(RegisterRead {
                    register: 3,
                    value: 11,
                }),
                ..Default::default()
            },
            RamAccess::Write(RamWrite {
                address,
                pre_value: 7,
                post_value: 11,
            }),
            2,
        )
        .unwrap()
    }

    pub(crate) fn no_op_row() -> TraceRow {
        TraceRow::new(
            JoltInstructionRow::default(),
            RegisterState::default(),
            RamAccess::NoOp,
            0,
        )
        .unwrap()
    }

    pub(crate) fn with_booleanity_backend<R>(
        log_t: usize,
        log_k_chunk: u8,
        f: impl FnOnce(&TraceBackend, BooleanityDimensions) -> R,
    ) -> R {
        let alu = TraceRow::new(
            ALU,
            RegisterState {
                rs1: Some(RegisterRead {
                    register: 2,
                    value: 5,
                }),
                rd: Some(RegisterWrite {
                    register: 1,
                    pre_value: 8,
                    post_value: 11,
                }),
                ..Default::default()
            },
            RamAccess::NoOp,
            3,
        )
        .unwrap();
        let mut rows = vec![
            load_row(0x8000_1000),
            store_row(0x8000_1008),
            no_op_row(),
            alu,
        ];
        rows.truncate(1 << log_t);
        with_trace_backend(log_t, log_k_chunk, rows, f)
    }

    /// Runs `f` against a trace backend over `rows` of the fixture program
    /// (padded by the backend up to `2^log_t`). Booleanity dimensions are
    /// probed off the backend's own servable set so test and backend
    /// geometry cannot drift.
    pub(crate) fn with_trace_backend<R>(
        log_t: usize,
        log_k_chunk: u8,
        rows: Vec<TraceRow>,
        f: impl FnOnce(&TraceBackend, BooleanityDimensions) -> R,
    ) -> R {
        use std::sync::Arc;
        let preprocessing = Arc::new(JoltProgramPreprocessing {
            bytecode: BytecodePreprocessing::preprocess(
                vec![LOAD, STORE, ALU],
                LOAD.address as u64,
                RV64IMAC_JOLT,
            )
            .unwrap(),
            ram: RAMPreprocessing::default(),
            memory_layout: Default::default(),
            max_padded_trace_length: 4.max(1 << log_t),
        });
        let program = Arc::new(JoltProgram::default());

        let config = JoltVmWitnessConfig::new(
            log_t,
            64,
            JoltOneHotConfig {
                log_k_chunk,
                lookups_ra_virtual_log_k_chunk: 16,
            },
        );
        let inputs = JoltVmWitnessInputs::new(
            &program,
            &preprocessing,
            TraceOutput::new(TraceData::new(rows), Default::default(), None, None),
        );
        let backend = TraceBackend::new(config, inputs);

        let probe = |family: fn(usize) -> JoltCommittedPolynomial| {
            (0..64)
                .take_while(|index| {
                    JoltWitnessOracle::<Fr>::shape(
                        &backend,
                        JoltPolynomialId::Committed(family(*index)),
                    )
                    .is_ok()
                })
                .count()
        };
        let layout = JoltRaPolynomialLayout::new(
            probe(JoltCommittedPolynomial::InstructionRa),
            probe(JoltCommittedPolynomial::BytecodeRa),
            probe(JoltCommittedPolynomial::RamRa),
        )
        .unwrap();
        let dimensions = BooleanityDimensions::new(layout, log_t, log_k_chunk as usize);
        f(&backend, dimensions)
    }

    pub(crate) fn test_challenge(round: usize) -> Fr {
        use jolt_field::Ring;
        Fr::from_u64(0x1234_5678 + 1000 * round as u64 + 7)
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use crate::optimized::parity::ExceptionalEq;
    use jolt_claims::protocols::jolt::JoltChallengeId;
    use jolt_claims::{InputClaims, OutputClaims, SumcheckChallenges};
    use jolt_field::{Fr, Ring, Zero};
    use jolt_poly::EqPolynomial;
    use jolt_verifier::stages::relations::ConcreteSumcheckChallenges;
    use jolt_verifier::stages::stage6b::booleanity::BooleanityInputClaims;
    use jolt_witness::JoltWitnessOracle;

    use super::testing::{test_challenge, with_booleanity_backend};
    use super::*;
    use crate::optimized::testing::replayed;
    use crate::ReferenceBackend;

    #[expect(clippy::type_complexity)]
    fn drive_lockstep<R>(
        mut reference: Box<dyn SumcheckKernel<Fr, Relation = R>>,
        mut optimized: Box<dyn SumcheckKernel<Fr, Relation = R>>,
        input_claim: Fr,
    ) -> (
        Box<dyn SumcheckKernel<Fr, Relation = R>>,
        Box<dyn SumcheckKernel<Fr, Relation = R>>,
        Vec<Fr>,
    )
    where
        R: ConcreteSumcheck<Fr>,
        SumcheckInputClaims<Fr, R>: InputClaims<Fr>,
        SumcheckOutputClaims<Fr, R>: OutputClaims<Fr>,
        ConcreteSumcheckChallenges<Fr, R>: SumcheckChallenges<Fr, JoltChallengeId>,
    {
        let rounds = reference.num_rounds();
        assert_eq!(rounds, optimized.num_rounds());
        let mut claim = input_claim;
        let mut bind = None;
        let mut challenges = Vec::new();
        for round in 0..rounds {
            let expected = reference.prove_round(bind, round, claim).unwrap();
            let actual = optimized.prove_round(bind, round, claim).unwrap();
            assert_eq!(expected, actual, "round {round} polynomial mismatch");
            let challenge = test_challenge(round);
            claim = expected.evaluate(challenge);
            challenges.push(challenge);
            bind = Some(challenge);
        }
        if let Some(last) = challenges.last() {
            reference.finish_rounds(*last).unwrap();
            optimized.finish_rounds(*last).unwrap();
        }
        (reference, optimized, challenges)
    }

    fn point(seed: u64, len: usize) -> Vec<Fr> {
        (0..len as u64)
            .map(|index| Fr::from_u64(seed + 37 * index + 5))
            .collect()
    }

    fn cycle_relation(
        dimensions: BooleanityDimensions,
        r_address: Vec<Fr>,
        reference_address: Vec<Fr>,
        reference_cycle: Vec<Fr>,
    ) -> Booleanity<Fr> {
        #[cfg(feature = "akita")]
        let dimensions = LatticeBooleanityDimensions::new(dimensions).unwrap();
        Booleanity::new(dimensions, r_address, reference_address, reference_cycle)
    }

    /// Brute-forces the cycle-phase input claim from the dense one-hot
    /// grids: `Σ_j eq_rr · eq_cycle(j) · Σ_i γ^{2i} (x_i(j)² − x_i(j))` with
    /// `x_i` the address-folded rows — independent of both kernels.
    fn brute_force_cycle_claim(
        backend: &dyn JoltWitnessOracle<Fr>,
        dimensions: BooleanityDimensions,
        r_address: &[Fr],
        reference_address: &[Fr],
        reference_cycle: &[Fr],
        gamma: Fr,
    ) -> Fr {
        let eq_address = EqPolynomial::new(r_address.to_vec()).evaluations();
        let address_scalar = try_eq_mle(r_address, reference_address).unwrap();
        let eq_cycle = EqPolynomial::new(reference_cycle.to_vec()).evaluations();
        let log_t = dimensions.log_t;
        let gamma_sqr = gamma * gamma;
        let mut total = Fr::from_u64(0);
        let mut weight = Fr::from_u64(1);
        for opening in BooleanityColumns::openings::<Fr>(dimensions).unwrap() {
            let grid: Vec<Fr> = backend.oracle_table(opening.polynomial_id()).unwrap();
            for (j, eq_cycle) in eq_cycle.iter().enumerate() {
                let x: Fr = eq_address
                    .iter()
                    .enumerate()
                    .map(|(k, eq)| *eq * grid[(k << log_t) | j])
                    .sum();
                total += address_scalar * *eq_cycle * weight * (x * x - x);
            }
            weight *= gamma_sqr;
        }
        total
    }

    /// `replayed_rows`: the optimized kernel reads the rows replayed from
    /// two-row checkpoints; the reference reads the resident rows.
    fn address_parity(log_t: usize, log_k_chunk: u8, replayed_rows: bool) {
        with_booleanity_backend(log_t, log_k_chunk, |backend, dimensions| {
            let replay = replayed_rows.then(|| replayed(backend, 2, None));
            let witness = replay.as_ref().unwrap_or(backend);
            let relation = BooleanityAddressPhase::<Fr>::new(
                dimensions,
                point(900, dimensions.log_k_chunk),
                point(300, log_t),
            );
            let claims = Default::default();
            let points = Default::default();
            let challenges = BooleanityAddressPhaseChallenges {
                reference_address: point(700, dimensions.log_k_chunk),
                gamma: Fr::from_u64(31),
            };
            let reference = ReferenceBackend
                .prepare(
                    &mut ProofSession::default(),
                    backend,
                    ProverInputs {
                        relation: &relation,
                        claims: &claims,
                        points: &points,
                        challenges: &challenges,
                    },
                )
                .unwrap();
            let optimized = OptimizedBooleanityAddress
                .prepare(
                    &mut ProofSession::default(),
                    witness,
                    ProverInputs {
                        relation: &relation,
                        claims: &claims,
                        points: &points,
                        challenges: &challenges,
                    },
                )
                .unwrap();
            let (mut reference, mut optimized, _) =
                drive_lockstep(reference, optimized, Fr::from_u64(0));
            assert_eq!(
                reference.output_claims(&claims).unwrap(),
                optimized.output_claims(&claims).unwrap(),
            );
        });
    }

    #[test]
    fn address_kernel_matches_reference() {
        address_parity(2, 4, false);
    }

    #[test]
    fn address_kernel_matches_reference_single_cycle_round() {
        address_parity(1, 4, false);
    }

    /// Two scan tasks cross the replay checkpoints and run into the padding.
    #[test]
    fn address_kernel_matches_reference_on_replayed_rows() {
        address_parity(3, 4, true);
        address_parity(2, 8, true);
    }

    #[test]
    fn address_kernel_matches_reference_wide_chunk() {
        address_parity(2, 8, false);
    }

    /// A replay failing inside the pushforward pass fails `prepare` typed.
    #[test]
    fn address_replay_failure_is_typed() {
        with_booleanity_backend(3, 4, |backend, dimensions| {
            let relation =
                BooleanityAddressPhase::<Fr>::new(dimensions, point(900, 4), point(300, 3));
            let challenges = BooleanityAddressPhaseChallenges {
                reference_address: point(700, 4),
                gamma: Fr::from_u64(31),
            };
            let prepared = OptimizedBooleanityAddress.prepare(
                &mut ProofSession::default(),
                &replayed(backend, 2, Some(1)),
                ProverInputs {
                    relation: &relation,
                    claims: &Default::default(),
                    points: &Default::default(),
                    challenges: &challenges,
                },
            );
            assert!(matches!(
                prepared,
                Err(KernelError::Witness(WitnessError::Replay { .. }))
            ));
        });
    }

    fn cycle_parity(log_t: usize, log_k_chunk: u8, replayed_rows: bool) {
        cycle_parity_with_gamma(log_t, log_k_chunk, replayed_rows, Fr::from_u64(31));
    }

    fn cycle_parity_with_gamma(log_t: usize, log_k_chunk: u8, replayed_rows: bool, gamma: Fr) {
        cycle_parity_case(
            log_t,
            log_k_chunk,
            replayed_rows,
            gamma,
            None,
            MATERIALIZE_BYTES,
        );
    }

    /// `replayed_rows`: the optimized kernel reads the rows replayed from
    /// two-row checkpoints; the reference reads the resident rows.
    fn cycle_parity_case(
        log_t: usize,
        log_k_chunk: u8,
        replayed_rows: bool,
        gamma: Fr,
        exceptional: Option<ExceptionalEq>,
        materialize_bytes: usize,
    ) {
        with_booleanity_backend(log_t, log_k_chunk, |backend, dimensions| {
            let replay = replayed_rows.then(|| replayed(backend, 2, None));
            let witness = replay.as_ref().unwrap_or(backend);
            let r_address = point(110, dimensions.log_k_chunk);
            let reference_address = point(700, dimensions.log_k_chunk);
            let reference_cycle = exceptional.map_or_else(
                || point(400, log_t),
                |case| case.point(log_t, test_challenge(0)),
            );
            let relation = cycle_relation(
                dimensions,
                r_address.clone(),
                reference_address.clone(),
                reference_cycle.clone(),
            );
            let input_claim = brute_force_cycle_claim(
                backend,
                dimensions,
                &r_address,
                &reference_address,
                &reference_cycle,
                gamma,
            );
            let claims = BooleanityInputClaims {
                address_phase: input_claim,
            };
            let points = BooleanityInputClaims {
                address_phase: point(50, dimensions.log_k_chunk),
            };
            let challenges = BooleanityCyclePhaseChallenges { gamma };

            let reference = ReferenceBackend
                .prepare(
                    &mut ProofSession::default(),
                    backend,
                    ProverInputs {
                        relation: &relation,
                        claims: &claims,
                        points: &points,
                        challenges: &challenges,
                    },
                )
                .unwrap();
            let optimized: Box<dyn SumcheckKernel<Fr, Relation = Booleanity<Fr>>> = Box::new(
                OptimizedBooleanityCycleKernel::prepare(
                    witness,
                    &ProverInputs {
                        relation: &relation,
                        claims: &claims,
                        points: &points,
                        challenges: &challenges,
                    },
                    materialize_bytes,
                )
                .unwrap(),
            );

            let (mut reference, mut optimized, challenges_drawn) =
                drive_lockstep(reference, optimized, input_claim);
            let reference_outputs = reference.output_claims(&claims).unwrap();
            let optimized_outputs = optimized.output_claims(&claims).unwrap();
            assert_eq!(reference_outputs, optimized_outputs);
            if gamma.is_zero() {
                assert!(reference_outputs
                    .opening_values()
                    .iter()
                    .skip(1)
                    .any(|value| !value.is_zero()));
            }

            let output_points = relation
                .derive_opening_points(&challenges_drawn, &points)
                .unwrap();
            reference
                .validate_derived_tables(&relation, &points, &output_points, &challenges)
                .unwrap();
            optimized
                .validate_derived_tables(&relation, &points, &output_points, &challenges)
                .unwrap();
        });
    }

    #[test]
    fn cycle_kernel_matches_reference_at_exceptional_eq_points() {
        for bits in [4u8, 8] {
            for gamma in [Fr::from_u64(0), Fr::from_u64(31)] {
                for case in ExceptionalEq::ALL {
                    cycle_parity_case(6, bits, true, gamma, Some(case), MATERIALIZE_BYTES);
                }
            }
        }
    }

    #[test]
    fn cycle_kernel_matches_reference_with_zero_batching_gamma() {
        cycle_parity_with_gamma(5, 4, false, Fr::from_u64(0));
    }

    #[test]
    fn cycle_kernel_matches_reference_single_round() {
        cycle_parity(1, 4, false);
    }

    /// `log_t = 5` drives the shared-table state machine through
    /// materialization (four staged binds, dense at `T/16`) into a dense
    /// round message. At `log_t = 4`, materialization happens only during
    /// `finish_rounds`.
    #[test]
    fn cycle_kernel_matches_reference_through_dense_rounds() {
        cycle_parity(5, 4, false);
    }

    #[test]
    fn cycle_kernel_matches_reference_materializing_at_finish() {
        cycle_parity(4, 4, false);
    }

    #[test]
    fn cycle_kernel_matches_reference_wide_chunk() {
        cycle_parity(2, 8, false);
    }

    /// Lazy rounds, the materializing bind and dense rounds over replayed
    /// visits that cross checkpoints and run into the padding.
    #[test]
    fn cycle_kernel_matches_reference_on_replayed_rows() {
        cycle_parity(5, 4, true);
    }

    /// A budget of `T/64` entries per dense table keeps the tables lazy
    /// through 32 branches and materializes them at the sixth bind. The
    /// all-zero cycle point (`ℓ(1) = 0`, nonzero scalar) takes the `q(1)`
    /// fallback at every width, including 16 and 32.
    #[test]
    fn cycle_kernel_matches_reference_materializing_at_64_branches() {
        for (log_t, exceptional) in [(8, None), (10, None), (8, Some(ExceptionalEq::Zero))] {
            let materialize_bytes = (size_of::<Fr>() << log_t) / 64;
            cycle_parity_case(
                log_t,
                4,
                false,
                Fr::from_u64(31),
                exceptional,
                materialize_bytes,
            );
        }
    }

    /// The production 6a→6b sequencing: the address phase (both kernels in
    /// lockstep) stages the intermediate claim; the cycle phase consumes it
    /// with `r_address` = the reversed address challenges, through one shared
    /// session. The reference kernel's round-zero check pins the intermediate
    /// claim to the cycle-phase sum, so a cross-phase orientation drift fails
    /// loudly.
    #[test]
    fn two_phase_flow_matches_reference_end_to_end() {
        let log_t = 2;
        with_booleanity_backend(log_t, 4, |backend, dimensions| {
            let reference_address = point(700, dimensions.log_k_chunk);
            let reference_cycle = point(400, log_t);
            let gamma = Fr::from_u64(31);
            // The stage-6a relation derives its reference cycle from the
            // stage-5 instruction cycle by reversal; feed the reversed
            // vector so `reference_cycle()` equals the 6b relation's.
            let instruction_r_cycle: Vec<Fr> = reference_cycle.iter().rev().copied().collect();
            let address_relation = BooleanityAddressPhase::<Fr>::new(
                dimensions,
                point(900, dimensions.log_k_chunk),
                instruction_r_cycle,
            );
            let address_claims = Default::default();
            let address_points = Default::default();
            let address_challenges = BooleanityAddressPhaseChallenges {
                reference_address: reference_address.clone(),
                gamma,
            };

            let mut session = ProofSession::default();
            let reference = ReferenceBackend
                .prepare(
                    &mut ProofSession::default(),
                    backend,
                    ProverInputs {
                        relation: &address_relation,
                        claims: &address_claims,
                        points: &address_points,
                        challenges: &address_challenges,
                    },
                )
                .unwrap();
            let optimized = OptimizedBooleanityAddress
                .prepare(
                    &mut session,
                    backend,
                    ProverInputs {
                        relation: &address_relation,
                        claims: &address_claims,
                        points: &address_points,
                        challenges: &address_challenges,
                    },
                )
                .unwrap();
            let (mut reference, mut optimized, address_challenges_drawn) =
                drive_lockstep(reference, optimized, Fr::from_u64(0));
            let intermediate = reference.output_claims(&address_claims).unwrap();
            assert_eq!(
                intermediate,
                optimized.output_claims(&address_claims).unwrap()
            );

            let r_address: Vec<Fr> = address_challenges_drawn.iter().rev().copied().collect();
            let cycle_relation = cycle_relation(
                dimensions,
                r_address.clone(),
                reference_address.clone(),
                reference_cycle.clone(),
            );
            let input_claim = brute_force_cycle_claim(
                backend,
                dimensions,
                &r_address,
                &reference_address,
                &reference_cycle,
                gamma,
            );
            assert_eq!(intermediate.intermediate, input_claim);

            let cycle_claims = BooleanityInputClaims {
                address_phase: input_claim,
            };
            let cycle_points = BooleanityInputClaims {
                address_phase: r_address.clone(),
            };
            let cycle_challenges = BooleanityCyclePhaseChallenges { gamma };
            let reference = ReferenceBackend
                .prepare(
                    &mut ProofSession::default(),
                    backend,
                    ProverInputs {
                        relation: &cycle_relation,
                        claims: &cycle_claims,
                        points: &cycle_points,
                        challenges: &cycle_challenges,
                    },
                )
                .unwrap();
            let optimized = OptimizedBooleanityCycle
                .prepare(
                    &mut session,
                    backend,
                    ProverInputs {
                        relation: &cycle_relation,
                        claims: &cycle_claims,
                        points: &cycle_points,
                        challenges: &cycle_challenges,
                    },
                )
                .unwrap();
            let (mut reference, mut optimized, cycle_challenges_drawn) =
                drive_lockstep(reference, optimized, input_claim);
            assert_eq!(
                reference.output_claims(&cycle_claims).unwrap(),
                optimized.output_claims(&cycle_claims).unwrap()
            );
            let output_points = cycle_relation
                .derive_opening_points(&cycle_challenges_drawn, &cycle_points)
                .unwrap();
            optimized
                .validate_derived_tables(
                    &cycle_relation,
                    &cycle_points,
                    &output_points,
                    &cycle_challenges,
                )
                .unwrap();
        });
    }
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "test module asserts successful categorical round reconstruction"
)]
mod categorical_tests {
    use crate::optimized::parity::ExceptionalEq;
    use std::ops::Range;

    #[cfg(feature = "akita")]
    use jolt_field::Prime128OffsetA7F7;
    use jolt_field::{Fr, JoltField, Ring, Zero};
    use jolt_poly::{BindingOrder, GruenSplitEqPolynomial};
    use jolt_witness::WitnessError;

    use super::super::lazy_ra::visit_resident;
    use super::{CategoricalProducts, ChunkIndexSource, LazyFoldedRa, MATERIALIZE_BYTES};
    use crate::reference::views::eq_table;

    /// Column-major hot indices; a cycle's row is its index.
    struct Source(Vec<Vec<Option<usize>>>);

    impl ChunkIndexSource for Source {
        type Row = usize;

        fn num_columns(&self) -> usize {
            self.0.len()
        }
        fn cycles(&self) -> usize {
            self.0[0].len()
        }
        fn index(&self, i: usize, cycle: &usize) -> Option<usize> {
            self.0[i][*cycle]
        }
        fn visit_blocks(
            &self,
            range: Range<usize>,
            block: usize,
            visit: impl FnMut(usize, &[Self::Row]) -> Result<(), WitnessError>,
        ) -> Result<(), WitnessError> {
            let cycles: Vec<usize> = (0..self.cycles()).collect();
            visit_resident(&cycles, range, block, visit)
        }
    }

    fn check_categorical_cubics<F: JoltField>(gamma: F) {
        check_categorical_case(gamma, None);
        for case in ExceptionalEq::ALL {
            check_categorical_case(gamma, Some(case));
        }
    }

    fn check_categorical_case<F: JoltField>(gamma: F, exceptional: Option<ExceptionalEq>) {
        for addresses in [2, 16, 256] {
            for bind in [F::zero(), F::one(), F::from_u64(13)] {
                // All-cold, alternating hot/cold, and fully hot families.
                // In particular None is not address zero, and a folded pair
                // can have only its low or only its high branch cold.
                let source = Source(
                    (0..3)
                        .map(|family| {
                            (0..32)
                                .map(|row| {
                                    if family == 0 || (family == 1 && row % 3 == 0) {
                                        None
                                    } else {
                                        Some((row * row + 7 * row + family) % addresses)
                                    }
                                })
                                .collect()
                        })
                        .collect(),
                );
                let rho: Vec<F> = [F::one(), gamma, gamma * gamma].to_vec();
                let unscaled: Vec<Vec<F>> = (0..3)
                    .map(|_| {
                        (0..addresses)
                            .map(|i| F::from_u64((i * i + 3 * i + 11) as u64))
                            .collect()
                    })
                    .collect();
                let tables: Vec<Vec<F>> = unscaled
                    .iter()
                    .zip(&rho)
                    .map(|(table, rho)| {
                        let scale = if rho.is_zero() { F::one() } else { *rho };
                        table.iter().map(|value| *value * scale).collect()
                    })
                    .collect();
                let mut dense: Vec<Vec<F>> = source
                    .0
                    .iter()
                    .zip(&unscaled)
                    .map(|(column, table)| {
                        column
                            .iter()
                            .map(|i| i.map_or(F::zero(), |i| table[i]))
                            .collect()
                    })
                    .collect();
                let mut lazy = LazyFoldedRa::new(tables, source, MATERIALIZE_BYTES);
                let reference: Vec<F> = exceptional.map_or_else(
                    || (0..5).map(|i| F::from_u64(7 + 3 * i)).collect(),
                    |case| case.point(5, bind),
                );
                let mut prefix = F::from_u64(19);
                let mut eq = GruenSplitEqPolynomial::new_with_scaling(
                    &reference,
                    BindingOrder::LowToHigh,
                    Some(prefix),
                );
                for round in 0..if addresses <= 16 { 2 } else { 1 } {
                    let suffix = eq_table(&reference[..4 - round]);
                    let bit = reference[4 - round];
                    // Direct evaluations of the defining weighted cubic;
                    // neither Gruen reconstruction nor coefficient caching.
                    let direct: Vec<F> = (0..4)
                        .map(|x| {
                            let x = F::from_u64(x);
                            let mut sum = F::zero();
                            for (row, weight) in suffix.iter().enumerate() {
                                for (column, r) in dense.iter().zip(&rho) {
                                    let v = column[2 * row]
                                        + x * (column[2 * row + 1] - column[2 * row]);
                                    sum += *weight * *r * *r * v * (v - F::one());
                                }
                            }
                            prefix * (bit * x + (F::one() - bit) * (F::one() - x)) * sum
                        })
                        .collect();
                    let LazyFoldedRa::Lazy {
                        tables,
                        width,
                        factors: 1,
                        source,
                        ..
                    } = &lazy
                    else {
                        unreachable!()
                    };
                    let products = CategoricalProducts::new(tables, *width, source, &rho);
                    let polynomial = products.lookup(&eq, round, direct[0] + direct[1]).unwrap();
                    for (x, value) in direct.iter().enumerate() {
                        assert_eq!(polynomial.evaluate(F::from_u64(x as u64)), *value);
                    }
                    for column in &mut dense {
                        for row in 0..column.len() / 2 {
                            column[row] =
                                column[2 * row] + bind * (column[2 * row + 1] - column[2 * row]);
                        }
                        column.truncate(column.len() / 2);
                    }
                    lazy.bind(bind).unwrap();
                    eq.bind(bind);
                    prefix *= bit * bind + (F::one() - bit) * (F::one() - bind);
                }
            }
        }
    }

    #[test]
    fn categorical_booleanity_matches_direct_cubic_dory() {
        check_categorical_cubics::<Fr>(Fr::zero());
        check_categorical_cubics::<Fr>(Fr::from_u64(3));
    }

    #[test]
    #[cfg(feature = "akita")]
    fn categorical_booleanity_matches_direct_cubic_akita() {
        check_categorical_cubics::<Prime128OffsetA7F7>(Prime128OffsetA7F7::zero());
        check_categorical_cubics::<Prime128OffsetA7F7>(Prime128OffsetA7F7::from_u64(3));
    }
}
