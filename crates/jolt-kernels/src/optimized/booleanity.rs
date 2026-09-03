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
//!   [`RaChunkSelector`]s over the packed stage-5 rows
//!   ([`SharedInstructionRows`], reclaimed from the [`ProofSession`] or
//!   collected in one streaming pass) and never materialize the `K × T`
//!   grids the naive tier's `oracle_table` walks, nor per-polynomial index
//!   columns (legacy `RaIndices`).
//! - **Pushforward `G` tables (address phase).** `G_i[k] = Σ_j
//!   eq(r_ref_cycle, j) · ra_i(k, j)` collapses to a scatter of the
//!   tensor-factored eq weights into a `K`-sized accumulator per
//!   polynomial: per `E_out` block, the `E_in` weights accumulate
//!   *unreduced* into the hot buckets (no per-cycle multiply at all), and
//!   each block reduces and folds by its `e_out` once (legacy
//!   `compute_all_G`). [`spawn_booleanity_address_masses`] runs the same
//!   build on a capped background pool once the anchor point exists; the
//!   6a prepare reclaims the tables (or rebuilds inline on any mismatch —
//!   identical values either way).
//! - **Shared expanded-eq gather (cycle phase).** The address-folded row
//!   `x_i(j) = eq(r_address)[hot_i(j)]` is a lookup into a `K`-sized table
//!   pre-scaled by `γ^i`, so `γ^{2i}(x² − x) = H(H − γ^i)` needs no
//!   batching multiply in the round loop (legacy `SharedRaPolynomials`
//!   pre-scaling), served by the shared [`LazyFoldedRa`] state machine —
//!   index-encoded for the first four binds, dense at `T/16` after. The
//!   Metal member drives the same state machine through the
//!   [`prepare_booleanity_cycle`] driver factory (two-phase
//!   `begin_round`/`collect_round` launches, CPU recompute on decline).
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
//!
//! Cross-stage carry: both prepares reclaim the packed stage-5 rows with
//! [`ProofSession::take`], clone the [`Arc`], and park the carry back for
//! the later consumers (mixed-backend registries fall back to a fresh
//! streaming pass when the carry is absent or geometry-stale).

use std::collections::BTreeMap;
use std::sync::Arc;

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
use jolt_witness::JoltWitnessPlane;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::instruction_read_raf::{shared_instruction_rows, InstructionCycleRow, InstructionRows};
use super::lazy_ra::{ChunkIndexSource, LazyFoldedRa, LazyRaDevice};
use super::support::{gamma_power_pairs, pin_derived_term_if_derived, RoundProgress};
use crate::reference::views::eq_table;
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

/// What a booleanity-cycle device driver captures at prepare (borrowed —
/// the factory clones what it keeps).
#[cfg_attr(
    not(all(feature = "metal", target_os = "macos")),
    expect(dead_code, reason = "read only by the Metal driver factory")
)]
pub(crate) struct BooleanityDeviceInputs<'a, F> {
    pub(crate) rows: &'a Arc<InstructionRows>,
    /// Per polynomial (layout order): [`ColumnSelector::device_meta`].
    pub(crate) poly_meta: Vec<(u32, u32)>,
    pub(crate) log_k_chunk: usize,
    /// γ^i per polynomial — the summand's `H(H − γ^i)` needs the unscaled
    /// power next to the pre-scaled tables.
    pub(crate) gamma_powers: &'a [F],
}

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
            Self::Instruction(selector) => Some(selector.chunk_u128(row.lookup_index)),
            Self::Bytecode(selector) => {
                Some(selector.chunk_usize(row.mapped_pc().unwrap_or_default()))
            }
            Self::Ram(selector) => row
                .remapped_ram_address()
                .map(|address| selector.chunk_usize(address as usize)),
            #[cfg(feature = "akita")]
            Self::UnsignedInc(column) => Some(row.fused_inc_row(*column)),
        }
    }

    /// The device shader's `(family, shift)` selector encoding. The packed
    /// protocol's extra columns have no device encoding, so the driver
    /// factory is not offered under `akita`.
    #[cfg(not(feature = "akita"))]
    fn device_meta(&self) -> (u32, u32) {
        match self {
            Self::Instruction(selector) => (0, selector.shift() as u32),
            Self::Bytecode(selector) => (1, selector.shift() as u32),
            Self::Ram(selector) => (2, selector.shift() as u32),
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

    /// The layout's chunk selectors, in canonical polynomial order, with the
    /// witness shapes validated up front.
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
/// `eq = E_out ⊗ E_in` factorization.
fn cycle_pushforward<F: JoltField>(
    rows: &[InstructionCycleRow],
    selectors: &[ColumnSelector],
    k_chunk: usize,
    point: &[F],
) -> Vec<Vec<F>> {
    let eq = TensorEqTable::new(point);
    debug_assert_eq!(eq.len(), rows.len());
    let e_out = eq.e_out();
    let e_in = eq.e_in();
    let in_len = e_in.len();

    struct State<F: JoltField> {
        /// Cross-block `Σ e_out · reduce(block)` lanes, still deferred.
        partial: Vec<Vec<F::Accumulator>>,
        /// Within-block unreduced `Σ e_in` buckets, cleared per block.
        block: Vec<Vec<F::Accumulator>>,
    }
    let zero = || State::<F> {
        partial: vec![vec![F::Accumulator::default(); k_chunk]; selectors.len()],
        block: vec![vec![F::Accumulator::default(); k_chunk]; selectors.len()],
    };
    let scatter = |mut state: State<F>, x_out: usize| {
        let base = x_out * in_len;
        for (x_in, e) in e_in.iter().enumerate() {
            let row = &rows[base + x_in];
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
        state
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

    #[cfg(feature = "parallel")]
    {
        finish(
            (0..e_out.len())
                .into_par_iter()
                .fold(zero, scatter)
                .reduce(zero, merge),
        )
    }
    #[cfg(not(feature = "parallel"))]
    {
        let _ = merge;
        finish((0..e_out.len()).fold(zero(), scatter))
    }
}

// ---------------------------------------------------------------------------
// Stage 6a: address phase
// ---------------------------------------------------------------------------

/// Session carry: the address-phase pushforward masses, built ahead of stage
/// 6a on a dedicated thread (the prover spawns the build before stage 5 when
/// the booleanity anchor is transcript-prior to it). The reference cycle is
/// stored for validation — a point mismatch (or a panicked worker) falls back
/// to the inline build, which produces bit-identical values.
#[cfg_attr(
    feature = "allocative",
    derive(allocative::Allocative),
    allocative(bound = "F: JoltField")
)]
struct PrebuiltBooleanityMasses<F> {
    #[cfg_attr(feature = "allocative", allocative(visit = jolt_poly::visit_scalars))]
    reference_cycle: Vec<F>,
    #[cfg_attr(feature = "allocative", allocative(skip))]
    handle: std::thread::JoinHandle<Vec<Vec<F>>>,
}

/// The background pool width for [`spawn_booleanity_address_masses`]: wide
/// enough to finish well inside stage 5's window at every scale, narrow
/// enough not to contend with the stage's own host work.
const BOOLEANITY_BACKGROUND_THREADS: usize = 4;

/// Spawn the stage-6a booleanity pushforward build (`cycle_pushforward` at
/// the little-endian `reference_cycle`) on a dedicated capped thread pool and
/// park the join handle in the session. Call after the anchor point exists;
/// the address-phase prepare reclaims the result (or rebuilds inline on any
/// mismatch). Row collection cost is unchanged: this reuses (or first parks)
/// the shared stage rows the later prepares would collect anyway.
pub fn spawn_booleanity_address_masses<F: JoltField>(
    session: &mut ProofSession,
    witness: &dyn JoltWitnessPlane<F>,
    dimensions: BooleanityDimensions,
    reference_cycle: Vec<F>,
) -> Result<(), KernelError<F>> {
    if reference_cycle.len() != dimensions.log_t {
        return Err(KernelError::InvariantViolation {
            reason: "booleanity anchor point length disagrees with the dimensions",
        });
    }
    let selectors = BooleanityColumns::new(witness, dimensions)?.selectors;
    let rows = shared_instruction_rows(session, witness, 1usize << dimensions.log_t)?;
    let k_chunk = 1usize << dimensions.log_k_chunk;
    let point = reference_cycle.clone();
    let handle = std::thread::spawn(move || {
        let _token = super::BACKGROUND_BUILD_TOKEN
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let build = || cycle_pushforward::<F>(&rows, &selectors, k_chunk, &point);
        #[cfg(feature = "parallel")]
        if let Ok(pool) = rayon::ThreadPoolBuilder::new()
            .num_threads(BOOLEANITY_BACKGROUND_THREADS)
            .build()
        {
            return pool.install(build);
        }
        build()
    });
    session.park(PrebuiltBooleanityMasses {
        reference_cycle,
        handle,
    });
    Ok(())
}

/// Slot front for the stage-6a booleanity address phase.
pub struct OptimizedBooleanityAddress;

impl<F: JoltField> PrepareKernel<F, BooleanityAddressPhase<F>> for OptimizedBooleanityAddress {
    fn prepare(
        &self,
        session: &mut ProofSession,
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

        // Reclaim the background-built masses when they match this relation's
        // point; otherwise (no spawn, stale anchor, panicked worker) build
        // inline — the same sums in either case, so the values are identical.
        let prebuilt = session
            .take::<PrebuiltBooleanityMasses<F>>()
            .filter(|carry| carry.reference_cycle == reference_cycle)
            .and_then(|carry| carry.handle.join().ok());
        let masses = if let Some(masses) = prebuilt {
            masses
        } else {
            let columns = BooleanityColumns::new(witness, dimensions)?;
            let rows = shared_instruction_rows(session, witness, 1usize << dimensions.log_t)?;
            cycle_pushforward(
                &rows,
                &columns.selectors,
                1usize << dimensions.log_k_chunk,
                &reference_cycle,
            )
        };

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
#[cfg_attr(
    feature = "allocative",
    derive(allocative::Allocative),
    allocative(bound = "F: JoltField")
)]
struct OptimizedBooleanityAddressKernel<F: JoltField> {
    progress: RoundProgress,
    /// Per checked polynomial, its `γ^{2i}` batching weight, in the layout's
    /// canonical order.
    #[cfg_attr(feature = "allocative", allocative(visit = jolt_poly::visit_scalars))]
    gamma_weights: Vec<F>,
    linear: Vec<Polynomial<F>>,
    /// Raw vectors because the squared-weight bind is not a multilinear bind.
    #[cfg_attr(feature = "allocative", allocative(visit = jolt_poly::visit_scalar_rows))]
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

// ---------------------------------------------------------------------------
// Stage 6b: cycle phase
// ---------------------------------------------------------------------------

/// Slot front for the stage-6b booleanity cycle phase.
pub struct OptimizedBooleanityCycle;

impl<F: JoltField> PrepareKernel<F, Booleanity<F>> for OptimizedBooleanityCycle {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, Booleanity<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = Booleanity<F>>>, KernelError<F>> {
        prepare_booleanity_cycle(session, witness, inputs, |_| None)
    }
}

/// The cycle-phase prepare with a [`LazyRaDevice`] factory (the optimized
/// slot passes `|_| None`; the Metal slot builds its driver from the
/// [`BooleanityDeviceInputs`]). Lane contract for installed drivers:
/// `[q_constant, q_leading]` — the two inner-quadratic coefficients
/// [`GruenSplitEqPolynomial::gruen_poly_deg_3`] consumes.
pub(crate) fn prepare_booleanity_cycle<F: JoltField>(
    session: &mut ProofSession,
    witness: &dyn JoltWitnessPlane<F>,
    inputs: ProverInputs<'_, F, Booleanity<F>>,
    driver: impl FnOnce(BooleanityDeviceInputs<'_, F>) -> Option<Box<dyn LazyRaDevice<F>>>,
) -> Result<Box<dyn SumcheckKernel<F, Relation = Booleanity<F>>>, KernelError<F>> {
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
    let rows = shared_instruction_rows(session, witness, 1usize << dimensions.log_t)?;

    // The fixed address eq factor of the `EqAddressCycle` public; rides
    // in the split-eq scaling so round messages and the bound scalar
    // carry it exactly like the reference's derived table.
    let address_scalar =
        try_eq_mle(r_address, reference_address).map_err(|_| KernelError::InvariantViolation {
            reason: "booleanity address point and reference length mismatch",
        })?;
    let eq_address = eq_table(r_address);
    let (gamma_powers, gamma_powers_inv) = gamma_power_pairs(
        inputs.challenges.gamma,
        columns.selectors.len(),
        "booleanity batching gamma must be invertible",
    )?;
    let tables: Vec<Vec<F>> = gamma_powers
        .iter()
        .map(|rho| eq_address.iter().map(|eq| *rho * *eq).collect())
        .collect();

    // The packed protocol's extra columns have no device selector encoding,
    // so the device tier is not offered under `akita`.
    #[cfg(not(feature = "akita"))]
    let driver = driver(BooleanityDeviceInputs {
        rows: &rows,
        poly_meta: columns
            .selectors
            .iter()
            .map(ColumnSelector::device_meta)
            .collect(),
        log_k_chunk: dimensions.log_k_chunk,
        gamma_powers: &gamma_powers,
    });
    #[cfg(feature = "akita")]
    let driver: Option<Box<dyn LazyRaDevice<F>>> = {
        let _ = driver;
        None
    };
    Ok(Box::new(OptimizedBooleanityCycleKernel {
        progress: RoundProgress::new(relation.rounds()),
        eq: GruenSplitEqPolynomial::new_with_scaling(
            reference_cycle,
            BindingOrder::LowToHigh,
            Some(address_scalar),
        ),
        tables: LazyFoldedRa::new_with_driver(
            tables,
            BooleanityChunks {
                rows,
                selectors: columns.selectors,
            },
            driver,
        ),
        gamma_powers,
        gamma_powers_inv,
        openings: columns.openings,
        launched: false,
    }))
}

/// Lazy-RA index source over the packed stage-5 rows: polynomial `i`'s hot
/// chunk at cycle `j`, through the layout's selectors.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
struct BooleanityChunks {
    #[cfg_attr(feature = "allocative", allocative(skip))]
    rows: Arc<InstructionRows>,
    #[cfg_attr(feature = "allocative", allocative(visit = jolt_poly::visit_scalars))]
    selectors: Vec<ColumnSelector>,
}

impl ChunkIndexSource for BooleanityChunks {
    fn num_polys(&self) -> usize {
        self.selectors.len()
    }

    fn cycles(&self) -> usize {
        self.rows.len()
    }

    #[inline]
    fn index(&self, i: usize, j: usize) -> Option<usize> {
        self.selectors[i].index(&self.rows[j])
    }
}

#[cfg_attr(
    feature = "allocative",
    derive(allocative::Allocative),
    allocative(bound = "F: JoltField")
)]
struct OptimizedBooleanityCycleKernel<F: JoltField> {
    progress: RoundProgress,
    /// Split-eq over the reference cycle, scaled by
    /// `eq(r_address, reference_address)` — together the reference's
    /// `EqAddressCycle` derived table.
    eq: GruenSplitEqPolynomial<F>,
    /// Pre-scaled (`γ^i`) shared address-folded tables, index-encoded for
    /// the first four binds (dense at `T/16` after).
    tables: LazyFoldedRa<F, BooleanityChunks>,
    #[cfg_attr(feature = "allocative", allocative(visit = jolt_poly::visit_scalars))]
    gamma_powers: Vec<F>,
    #[cfg_attr(feature = "allocative", allocative(visit = jolt_poly::visit_scalars))]
    gamma_powers_inv: Vec<F>,
    #[cfg_attr(feature = "allocative", allocative(visit = jolt_poly::visit_scalars))]
    openings: Vec<JoltOpeningId>,
    /// A `begin_round` device launch is in flight (its `collect_round`
    /// pending).
    launched: bool,
}

impl<F: JoltField> OptimizedBooleanityCycleKernel<F> {
    fn bind(&mut self, challenge: F) {
        self.eq.bind(challenge);
        self.tables.bind(challenge);
        self.progress.advance();
    }
}

impl<F: JoltField> ProveRounds<F> for OptimizedBooleanityCycleKernel<F> {
    fn num_rounds(&self) -> usize {
        self.progress.total()
    }

    fn prove_round(
        &mut self,
        bind: Option<F>,
        _round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if let Some(challenge) = bind {
            self.bind(challenge);
        }
        // Device tier first (lazy scan or fused dense round); a decline has
        // already normalized the state for the CPU fold below.
        if let Some(lanes) = self
            .tables
            .device_lanes(self.eq.e_in_current(), self.eq.e_out_current())
        {
            debug_assert_eq!(lanes.len(), 2);
            return Ok(self.eq.gruen_poly_deg_3(lanes[0], lanes[1], previous_claim));
        }
        let tables = &self.tables;
        let gamma_powers = &self.gamma_powers;
        let num_polys = gamma_powers.len();

        struct Scratch<F: JoltField> {
            /// Within-block `Σ e_in · (constant, leading)` lanes, deferred.
            lanes: [F::Accumulator; 2],
            pairs: Vec<(F, F)>,
        }

        // Inner quadratic `q(X) = Σ_j eq_rest(j) · Σ_i (H_i(X)² − γ^i·H_i(X))`:
        // constant coefficient from `H` at 0, leading coefficient from the
        // pair delta — the pre-scaling makes `γ^{2i}(x² − x) = H(H − γ^i)`.
        // Per-row products accumulate unreduced, reduce once, and fold into
        // the block lanes by `e_in`; blocks fold by `e_out` (legacy
        // `par_fold_out_in_unreduced`).
        let block_lanes = self.eq.par_fold_out_in(
            || Scratch {
                lanes: [F::Accumulator::default(); 2],
                pairs: vec![(F::zero(), F::zero()); num_polys],
            },
            |scratch, row, _x_in, e_in| {
                tables.lo_hi_all(row, &mut scratch.pairs);
                let mut constant = F::Accumulator::default();
                let mut leading = F::Accumulator::default();
                for ((h_0, h_1), rho) in scratch.pairs.iter().zip(gamma_powers) {
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
        );
        Ok(self.eq.gruen_poly_deg_3(
            block_lanes[0].reduce(),
            block_lanes[1].reduce(),
            previous_claim,
        ))
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind);
        Ok(())
    }

    fn begin_round(
        &mut self,
        bind: Option<F>,
        _round: usize,
        _previous_claim: F,
    ) -> Result<bool, SumcheckError<F>> {
        // A prepare-time prelaunched round 0 (only round 0 arrives bindless)
        // is already in flight: report launched without re-launching.
        if bind.is_none() && self.launched {
            return Ok(true);
        }
        if let Some(challenge) = bind {
            self.bind(challenge);
        }
        self.launched = self
            .tables
            .launch_device_lanes(self.eq.e_in_current(), self.eq.e_out_current());
        Ok(self.launched)
    }

    fn collect_round(
        &mut self,
        _bind: Option<F>,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if std::mem::take(&mut self.launched) {
            if let Some(lanes) = self.tables.collect_device_lanes() {
                debug_assert_eq!(lanes.len(), 2);
                return Ok(self.eq.gruen_poly_deg_3(lanes[0], lanes[1], previous_claim));
            }
            // Wait failure: the driver latched off and normalized state —
            // fall through to the synchronous recompute of the SAME round.
        }
        // `begin_round` already bound, so recompute with no bind. The
        // device tier inside declines (latched off or already reclaimed).
        self.prove_round(None, round, previous_claim)
    }
}

impl<F: JoltField> SumcheckKernel<F> for OptimizedBooleanityCycleKernel<F> {
    type Relation = Booleanity<F>;

    fn output_claims(
        &mut self,
        _inputs: &SumcheckInputClaims<F, Self::Relation>,
    ) -> Result<SumcheckOutputClaims<F, Self::Relation>, SumcheckKernelError<F>> {
        self.progress.require_complete()?;
        // Unscale the pre-scaled tables back to the committed polynomials'
        // claims; resolve by id so the output struct shape stays the
        // relation's business.
        self.tables.ensure_host();
        let values: BTreeMap<JoltOpeningId, F> = self
            .openings
            .iter()
            .copied()
            .enumerate()
            .map(|(i, id)| (id, self.tables.value(i, 0) * self.gamma_powers_inv[i]))
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
    use jolt_program::execution::{
        JoltProgram, OwnedTrace, RamAccess, RamRead, RamWrite, RegisterRead, RegisterState,
        RegisterWrite, TraceOutput, TraceRow,
    };
    use jolt_program::preprocess::{
        BytecodePreprocessing, JoltProgramPreprocessing, RAMPreprocessing,
    };
    use jolt_riscv::{JoltInstructionKind, JoltInstructionRow, NormalizedOperands, RV64IMAC_JOLT};
    use jolt_witness::{JoltVmWitnessConfig, JoltVmWitnessInputs, JoltWitnessOracle, TraceBackend};

    /// Runs `f` against a trace backend whose rows exercise the one-hot
    /// sparsity structure: hot/cold bytecode cycles, hot/cold RAM cycles,
    /// varied lookup indices, plus backend-synthesized padding when
    /// `log_t > 2`. Booleanity dimensions are probed off the backend's own
    /// servable set so test and backend geometry cannot drift.
    pub(crate) fn with_booleanity_backend<R>(
        log_t: usize,
        log_k_chunk: u8,
        f: impl FnOnce(&TraceBackend<OwnedTrace>, BooleanityDimensions) -> R,
    ) -> R {
        let instruction_a = JoltInstructionRow {
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
        let instruction_b = JoltInstructionRow {
            instruction_kind: JoltInstructionKind::SD,
            address: 0x8000_0004,
            operands: NormalizedOperands {
                rd: None,
                rs1: Some(1),
                rs2: Some(3),
                imm: 8,
            },
            ..instruction_a
        };
        let instruction_c = JoltInstructionRow {
            instruction_kind: JoltInstructionKind::ADDI,
            address: 0x8000_0008,
            operands: NormalizedOperands {
                rd: Some(1),
                rs1: Some(2),
                rs2: None,
                imm: 3,
            },
            ..instruction_a
        };
        use std::sync::Arc;
        let preprocessing = Arc::new(JoltProgramPreprocessing {
            bytecode: BytecodePreprocessing::preprocess(
                vec![instruction_a, instruction_b, instruction_c],
                instruction_a.address as u64,
                RV64IMAC_JOLT,
            )
            .unwrap(),
            ram: RAMPreprocessing::default(),
            memory_layout: Default::default(),
            max_padded_trace_length: 4.max(1 << log_t),
        });
        let program = Arc::new(JoltProgram::default());
        // Field mutation instead of struct literals: `TraceRow` grows a
        // cfg-gated field under the `field-inline` feature, which a literal
        // cannot spell portably from this crate.
        let row = |instruction: Option<JoltInstructionRow>,
                   registers: RegisterState,
                   ram_access: RamAccess| {
            let mut row = TraceRow::default();
            if let Some(instruction) = instruction {
                row.instruction = instruction;
            }
            row.registers = registers;
            row.ram_access = ram_access;
            row
        };
        let mut rows = vec![
            // Hot bytecode, hot RAM, register activity.
            row(
                Some(instruction_a),
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
                RamAccess::Read(RamRead {
                    address: 0x8000_1000,
                    value: 8,
                }),
            ),
            // Hot bytecode (different PC / lookup index), hot RAM (write).
            row(
                Some(instruction_b),
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
                    address: 0x8000_1008,
                    pre_value: 7,
                    post_value: 11,
                }),
            ),
            // Cold bytecode and RAM.
            row(None, RegisterState::default(), RamAccess::NoOp),
            // Hot bytecode, cold RAM.
            row(
                Some(instruction_c),
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
            ),
        ];
        rows.truncate(1 << log_t);

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
            TraceOutput::new(OwnedTrace::new(rows), Default::default(), None, None),
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

    /// Deterministic nonzero challenge sequence for lockstep test drives.
    pub(crate) fn test_challenge(round: usize) -> Fr {
        use jolt_field::Ring;
        Fr::from_u64(0x1234_5678 + 1000 * round as u64 + 7)
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use jolt_claims::protocols::jolt::JoltChallengeId;
    use jolt_claims::{InputClaims, OutputClaims, SumcheckChallenges};
    use jolt_field::{Fr, Ring};
    use jolt_poly::EqPolynomial;
    use jolt_verifier::stages::relations::ConcreteSumcheckChallenges;
    use jolt_verifier::stages::stage6b::booleanity::BooleanityInputClaims;
    use jolt_witness::JoltWitnessOracle;

    use super::super::instruction_read_raf::{
        collect_instruction_cycle_rows, SharedInstructionRows,
    };
    use super::testing::{test_challenge, with_booleanity_backend};
    use super::*;
    use crate::ReferenceBackend;

    /// Drives both kernels through the full round loop with identical
    /// challenges, asserting byte-identical round polynomials, and delivers
    /// the terminal bind. Returns the fully bound kernels and the challenge
    /// sequence.
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

    fn address_parity(log_t: usize, log_k_chunk: u8) {
        with_booleanity_backend(log_t, log_k_chunk, |backend, dimensions| {
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
            let mut session = ProofSession::default();
            let optimized = OptimizedBooleanityAddress
                .prepare(
                    &mut session,
                    backend,
                    ProverInputs {
                        relation: &relation,
                        claims: &claims,
                        points: &points,
                        challenges: &challenges,
                    },
                )
                .unwrap();
            // The two-table split makes the input claim exactly zero.
            let (mut reference, mut optimized, _) =
                drive_lockstep(reference, optimized, Fr::from_u64(0));
            assert_eq!(
                reference.output_claims(&claims).unwrap(),
                optimized.output_claims(&claims).unwrap(),
            );
            // The 6a prepare parks a shared-rows carry for the 6b consumers.
            assert!(session.state::<SharedInstructionRows>().is_some());
        });
    }

    #[test]
    fn address_kernel_matches_reference() {
        address_parity(2, 4);
    }

    #[test]
    fn address_kernel_matches_reference_single_cycle_round() {
        address_parity(1, 4);
    }

    #[test]
    fn address_kernel_matches_reference_wide_chunk() {
        address_parity(2, 8);
    }

    /// The background spawn and the inline build feed byte-identical kernels:
    /// same round polynomials, same output claims. Also pins the carry
    /// contract — the prepare consumes a matching carry and leaves a
    /// shared-rows carry parked (the spawn parks it) for the 6b consumers.
    #[test]
    fn background_masses_match_inline_build() {
        with_booleanity_backend(3, 4, |backend, dimensions| {
            let relation = BooleanityAddressPhase::<Fr>::new(
                dimensions,
                point(900, dimensions.log_k_chunk),
                point(300, 3),
            );
            let claims = Default::default();
            let points = Default::default();
            let challenges = BooleanityAddressPhaseChallenges {
                reference_address: point(700, dimensions.log_k_chunk),
                gamma: Fr::from_u64(31),
            };
            let mut spawned_session = ProofSession::default();
            spawn_booleanity_address_masses(
                &mut spawned_session,
                backend,
                dimensions,
                relation.reference_cycle(),
            )
            .unwrap();
            let spawned = OptimizedBooleanityAddress
                .prepare(
                    &mut spawned_session,
                    backend,
                    ProverInputs {
                        relation: &relation,
                        claims: &claims,
                        points: &points,
                        challenges: &challenges,
                    },
                )
                .unwrap();
            assert!(
                spawned_session
                    .state::<PrebuiltBooleanityMasses<Fr>>()
                    .is_none(),
                "the prepare must consume the background carry"
            );
            assert!(spawned_session.state::<SharedInstructionRows>().is_some());
            let inline = OptimizedBooleanityAddress
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
            let (mut spawned, mut inline, _) = drive_lockstep(spawned, inline, Fr::from_u64(0));
            assert_eq!(
                spawned.output_claims(&claims).unwrap(),
                inline.output_claims(&claims).unwrap(),
            );
        });
    }

    /// A carry built at a different anchor point is discarded and the prepare
    /// rebuilds inline — the kernel still matches the clean inline build.
    #[test]
    fn stale_background_carry_falls_back_inline() {
        with_booleanity_backend(3, 4, |backend, dimensions| {
            let relation = BooleanityAddressPhase::<Fr>::new(
                dimensions,
                point(900, dimensions.log_k_chunk),
                point(300, 3),
            );
            let claims = Default::default();
            let points = Default::default();
            let challenges = BooleanityAddressPhaseChallenges {
                reference_address: point(700, dimensions.log_k_chunk),
                gamma: Fr::from_u64(31),
            };
            let mut stale_session = ProofSession::default();
            spawn_booleanity_address_masses(
                &mut stale_session,
                backend,
                dimensions,
                point(555, 3), // not the relation's reference cycle
            )
            .unwrap();
            let stale = OptimizedBooleanityAddress
                .prepare(
                    &mut stale_session,
                    backend,
                    ProverInputs {
                        relation: &relation,
                        claims: &claims,
                        points: &points,
                        challenges: &challenges,
                    },
                )
                .unwrap();
            let inline = OptimizedBooleanityAddress
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
            let (mut stale, mut inline, _) = drive_lockstep(stale, inline, Fr::from_u64(0));
            assert_eq!(
                stale.output_claims(&claims).unwrap(),
                inline.output_claims(&claims).unwrap(),
            );
        });
    }

    fn cycle_parity(log_t: usize, log_k_chunk: u8, carried_indices: bool) {
        with_booleanity_backend(log_t, log_k_chunk, |backend, dimensions| {
            let r_address = point(110, dimensions.log_k_chunk);
            let reference_address = point(700, dimensions.log_k_chunk);
            let reference_cycle = point(400, log_t);
            let gamma = Fr::from_u64(31);
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
            let mut session = ProofSession::default();
            if carried_indices {
                let rows = collect_instruction_cycle_rows::<Fr>(backend, 1usize << log_t).unwrap();
                session.park(SharedInstructionRows(Arc::new(InstructionRows::new(rows))));
            }
            let optimized = OptimizedBooleanityCycle
                .prepare(
                    &mut session,
                    backend,
                    ProverInputs {
                        relation: &relation,
                        claims: &claims,
                        points: &points,
                        challenges: &challenges,
                    },
                )
                .unwrap();
            if carried_indices {
                assert!(
                    session.state::<SharedInstructionRows>().is_some(),
                    "cycle prepare must park a shared-rows carry back for later consumers"
                );
            }

            let (mut reference, mut optimized, challenges_drawn) =
                drive_lockstep(reference, optimized, input_claim);
            let reference_outputs = reference.output_claims(&claims).unwrap();
            let optimized_outputs = optimized.output_claims(&claims).unwrap();
            assert_eq!(reference_outputs, optimized_outputs);

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
    fn cycle_kernel_matches_reference() {
        cycle_parity(2, 4, false);
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

    #[test]
    fn cycle_kernel_matches_reference_with_carried_indices() {
        cycle_parity(2, 4, true);
    }

    /// The production 6a→6b sequencing: the address phase (both kernels in
    /// lockstep) stages the intermediate claim; the cycle phase consumes it
    /// with `r_address` = the reversed address challenges, reclaiming the
    /// parked index columns from the shared session. The reference kernel's
    /// round-zero check pins the intermediate claim to the cycle-phase sum,
    /// so a cross-phase orientation drift fails loudly.
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

            // 6b: the address opening prefix is the reversed 6a point.
            let r_address: Vec<Fr> = address_challenges_drawn.iter().rev().copied().collect();
            let cycle_relation = cycle_relation(
                dimensions,
                r_address.clone(),
                reference_address.clone(),
                reference_cycle.clone(),
            );
            // Cross-phase consistency: the staged intermediate equals the
            // cycle phase's sum at the bound address point.
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
            assert!(
                session.state::<SharedInstructionRows>().is_some(),
                "cycle prepare must park a shared-rows carry back"
            );
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
