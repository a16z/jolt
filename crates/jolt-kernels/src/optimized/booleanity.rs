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
//!   the canonical address chunks of the packed stage-5 rows
//!   (`SharedInstructionRows`, reclaimed from the [`ProofSession`] or
//!   collected in one streaming pass) and never materialize the `K × T`
//!   grids the naive tier's `oracle_table` walks, nor per-polynomial index
//!   columns (legacy `RaIndices`).
//! - **Pushforward `G` tables (address phase).** `G_i[k] = Σ_j
//!   eq(r_ref_cycle, j) · ra_i(k, j)` collapses to a scatter of the
//!   tensor-factored eq weights into a `K`-sized accumulator per
//!   polynomial: per `E_out` block, the `E_in` weights accumulate
//!   *unreduced* into nonzero instruction/bytecode buckets and all hot RAM
//!   buckets. Each block reduces and folds by its `e_out` once. The omitted
//!   instruction/bytecode zero buckets are recovered from total mass one.
//! - **Shared expanded-eq gather (cycle phase).** The address-folded row
//!   `x_i(j) = eq(r_address)[hot_i(j)]` is a lookup into a `K`-sized table
//!   pre-scaled by `γ^i`, so `γ^{2i}(x² − x) = H(H − γ^i)` needs no
//!   batching multiply in the round loop (legacy `SharedRaPolynomials`
//!   pre-scaling), served by the shared `LazyFoldedRa` state machine:
//!   index-encoded for the first four binds, dense at `T/16` after.
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
//!   [`jolt_poly::GruenRound::gruen_poly_deg_3`]. The stage-6a
//!   `eq(r_address, reference_address)` scalar rides in the split-eq
//!   scaling factor, mirroring the reference's `EqAddressCycle` derived
//!   table exactly.
//!
//! The address phase's round loop runs over the `2^log_k_chunk`-point chunk
//! domain (16–256 points) — negligible next to the `T`-scale table
//! construction — so it reuses the reference kernel's explicit four-point
//! sampling verbatim; only the table construction is replaced.
//!
//! Cross-stage carry: both prepares share the packed stage-5 rows through
//! `SharedInstructionRows`, whose docs own the carry protocol.

use crate::optimized::support::eq_round_error;
use std::collections::BTreeMap;
use std::sync::Arc;

use jolt_claims::protocols::jolt::geometry::booleanity::BooleanityDimensions;
use jolt_claims::protocols::jolt::geometry::ra::{JoltRaPolynomial, JoltRaPolynomialLayout};
use jolt_claims::protocols::jolt::JoltRelationId;
use jolt_claims::protocols::jolt::{BooleanityPublic, JoltDerivedId, JoltOpeningId};
use jolt_claims::OutputClaims;
use jolt_field::{Accumulator, JoltField};
use jolt_poly::{try_eq_mle, BindingOrder, GruenSplitEqPolynomial, Polynomial, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::{
    ConcreteSumcheck, SumcheckInputClaims, SumcheckInputPoints, SumcheckOutputClaims,
    SumcheckOutputPoints,
};
use jolt_verifier::stages::stage6a::booleanity::{
    BooleanityAddressPhase, BooleanityAddressPhaseChallenges, BooleanityAddressPhaseOutputClaims,
};
use jolt_verifier::stages::stage6b::booleanity::{Booleanity, BooleanityCyclePhaseChallenges};
use jolt_witness::witnesses::RaChunkSelector;
use jolt_witness::JoltWitnessPlane;

use super::constant_suffix::ConstantSuffix;
use super::instruction_read_raf::InstructionCycleRow;
use super::lazy_ra::{ChunkIndexSource, LazyFoldedRa};
use super::one_hot_histograms::{pushforwards, FamilySelectors};
use super::support::{gamma_powers, pin_derived_term_if_derived, GruenRoundMessage, RoundProgress};
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
        }
    }
}

struct BooleanityColumns {
    openings: Vec<JoltOpeningId>,
    selectors: Vec<ColumnSelector>,
}

impl BooleanityColumns {
    fn new<F: JoltField>(
        witness: &dyn JoltWitnessPlane<F>,
        dimensions: BooleanityDimensions,
    ) -> Result<Self, KernelError<F>> {
        let log_t = dimensions.log_t;
        let log_k_chunk = dimensions.log_k_chunk;
        let layout = dimensions.layout;
        let openings = layout
            .openings(JoltRelationId::Booleanity)
            .collect::<Vec<_>>();
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
        debug_assert_eq!(openings.len(), selectors.len());
        Ok(Self {
            openings,
            selectors,
        })
    }
}

/// Canonical-layout masses `G_i[k] = Σ_{j : hot_i(j) = k} eq(point, j)`.
/// Instruction and bytecode digit-zero masses follow from total weight one;
/// RAM may be cold, so every RAM bucket is accumulated directly.
fn cycle_pushforward<F: JoltField>(
    rows: &ConstantSuffix<InstructionCycleRow>,
    layout: JoltRaPolynomialLayout,
    k_chunk: usize,
    point: &[F],
) -> Result<Vec<Vec<F>>, KernelError<F>> {
    let families = FamilySelectors::new::<F>(
        (layout.instruction(), layout.bytecode(), layout.ram()),
        k_chunk.trailing_zeros() as usize,
    )?;
    let mut masses = pushforwards(rows, point, &families, k_chunk, true)?;
    for mass in &mut masses[..layout.instruction() + layout.bytecode()] {
        mass[0] = F::one() - mass[1..].iter().copied().sum::<F>();
    }
    Ok(masses)
}

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

        drop(BooleanityColumns::new(witness, dimensions)?);
        let rows = InstructionCycleRow::shared_for_next_consumer(
            session,
            witness,
            1usize << dimensions.log_t,
        )?;
        let masses = cycle_pushforward(
            &rows,
            dimensions.layout,
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
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, Booleanity<F>>,
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
        let rows = InstructionCycleRow::shared(session, witness, 1usize << dimensions.log_t)?;

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

        Ok(Box::new(OptimizedBooleanityCycleKernel {
            progress: RoundProgress::new(relation.rounds()),
            eq: GruenSplitEqPolynomial::new_with_scaling(
                reference_cycle,
                BindingOrder::LowToHigh,
                Some(address_scalar),
            ),
            tables: LazyFoldedRa::new(
                tables,
                BooleanityChunks {
                    prefix_len: rows.prefix_len(),
                    rows,
                    selectors: columns.selectors,
                    instruction_chunks: dimensions.layout.instruction(),
                    chunk_bits: dimensions.log_k_chunk,
                },
            ),
            gamma_powers,
            opening_unscale,
            openings: columns.openings,
        }))
    }
}

#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
struct BooleanityChunks {
    rows: Arc<ConstantSuffix<InstructionCycleRow>>,
    prefix_len: usize,
    #[cfg_attr(feature = "allocative", allocative(visit = crate::backend::visit_heap_free_elements))]
    selectors: Vec<ColumnSelector>,
    instruction_chunks: usize,
    chunk_bits: usize,
}

mod bulk;

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
        let round = self.progress.bound();
        self.eq
            .round_mut()
            .ok_or_else(|| eq_round_error(round))?
            .bind(challenge);
        self.tables.bind(challenge);
        self.progress.advance();
        Ok(())
    }
}

/// After address binding, every unbound value comes from a small alphabet.
/// Cache the two nonlinear coefficients over that alphabet, retaining the
/// existing exact split-equality fold. The extra category is a cold RAM row;
/// it must remain distinct from hot address zero, including after a bind.
struct CategoricalProducts<'a, F: JoltField, S, const TRIANGULAR: bool> {
    source: &'a S,
    width: usize,
    radix: usize,
    states: usize,
    products: Vec<(Vec<F>, Vec<F>)>,
}

impl<'a, F: JoltField, S: ChunkIndexSource, const TRIANGULAR: bool>
    CategoricalProducts<'a, F, S, TRIANGULAR>
{
    fn square_entries(states: usize) -> usize {
        if TRIANGULAR {
            states * (states + 1) / 2
        } else {
            states * states
        }
    }

    #[inline]
    fn square_index(&self, lo: usize, hi: usize) -> usize {
        if TRIANGULAR {
            Self::square_entries(lo.max(hi)) + lo.min(hi)
        } else {
            lo * self.states + hi
        }
    }

    fn fits_budget(width: usize, addresses: usize, cycles: usize) -> bool {
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
        // within the existing T/16 dense allocation budget per family.
        Self::square_entries(states) + 2 * states + width * addresses <= cycles / 16
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
                let mut squares = Vec::with_capacity(Self::square_entries(states));
                for (row, lo) in values.iter().enumerate() {
                    // (hi - lo)^2 = (lo - hi)^2, including cold RAM categories.
                    let columns = if TRIANGULAR { row + 1 } else { states };
                    squares.extend(values[..columns].iter().map(|hi| (*hi - *lo).square()));
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
    fn state(&self, family: usize, row: usize) -> usize {
        let first = self
            .source
            .index(family, row * self.width)
            .unwrap_or(self.radix - 1);
        if self.width == 1 {
            first
        } else {
            first
                + self.radix
                    * self
                        .source
                        .index(family, row * self.width + 1)
                        .unwrap_or(self.radix - 1)
        }
    }

    fn lookup(
        &self,
        eq: &GruenSplitEqPolynomial<F>,
        round: usize,
        claim: F,
        prefix_pairs: usize,
        tail_mass: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        let lanes = eq.par_fold_out_in_prefix(
            prefix_pairs,
            || [F::Accumulator::default(); 2],
            |lanes, row, _, weight| {
                // These are sums of field elements, so defer reduction across
                // the columns just as the outer lanes defer it across rows.
                let mut constant = F::Accumulator::default();
                let mut leading = F::Accumulator::default();
                self.source.for_each_category_pair(
                    self.width,
                    row,
                    self.radix,
                    self.products.len(),
                    |family, lo, hi| {
                        let (constants, squares) = &self.products[family];
                        constant.add(constants[lo]);
                        leading.add(squares[self.square_index(lo, hi)]);
                    },
                );
                lanes[0].fmadd(weight, constant.reduce());
                lanes[1].fmadd(weight, leading.reduce());
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
        );
        eq.round()
            .ok_or_else(|| eq_round_error(round))?
            .checked_cubic(
                lanes[0].reduce() + tail_mass,
                lanes[1].reduce(),
                claim,
                round,
                || {
                    tail_mass
                        + eq.par_fold_out_in_prefix(
                            prefix_pairs,
                            F::zero,
                            |sum, row, _, weight| {
                                let value = self.products.iter().enumerate().fold(
                                    F::zero(),
                                    |value, (family, (constants, _))| {
                                        value + constants[self.state(family, 2 * row + 1)]
                                    },
                                );
                                *sum += weight * value;
                            },
                            |_, weight, sum| weight * sum,
                            |a, b| a + b,
                        )
                },
            )
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
        let prefix_pairs = self.tables.prefix_len().div_ceil(2);
        let tail = self
            .tables
            .tail_values()
            .iter()
            .zip(&self.gamma_powers)
            .take_while(|(_, rho)| !rho.is_zero())
            .map(|(value, rho)| *value * (*value - *rho))
            .sum::<F>();
        let tail_mass = self.eq.suffix_weight(prefix_pairs) * tail;
        if let LazyFoldedRa::Lazy {
            tables,
            width,
            factors: 1,
            source,
            ..
        } = &self.tables
        {
            if !tables.is_empty() {
                let addresses = tables[0].len() / width;
                let cycles = source.cycles();
                if CategoricalProducts::<F, BooleanityChunks, false>::fits_budget(
                    *width, addresses, cycles,
                ) {
                    return CategoricalProducts::<F, _, false>::new(
                        tables,
                        *width,
                        source,
                        &self.gamma_powers,
                    )
                    .lookup(
                        &self.eq,
                        round,
                        previous_claim,
                        prefix_pairs,
                        tail_mass,
                    );
                }
                // Retain direct indexing whenever the full table fits. The
                // compact table enables caching without increasing its budget.
                if CategoricalProducts::<F, BooleanityChunks, true>::fits_budget(
                    *width, addresses, cycles,
                ) {
                    return CategoricalProducts::<F, _, true>::new(
                        tables,
                        *width,
                        source,
                        &self.gamma_powers,
                    )
                    .lookup(
                        &self.eq,
                        round,
                        previous_claim,
                        prefix_pairs,
                        tail_mass,
                    );
                }
            }
        }
        let tables = &self.tables;
        let gamma_powers = &self.gamma_powers;
        let num_polys = gamma_powers.len();
        let active_polys = gamma_powers.iter().take_while(|rho| !rho.is_zero()).count();
        let paired = tables.paired_lo_hi(active_polys);

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
        let block_lanes = self.eq.par_fold_out_in_prefix(
            prefix_pairs,
            || Scratch {
                lanes: [F::Accumulator::default(); 2],
                pairs: vec![(F::zero(), F::zero()); num_polys],
            },
            |scratch, row, _x_in, e_in| {
                if let Some(paired) = &paired {
                    paired.gather(row, &mut scratch.pairs);
                } else {
                    tables.lo_hi_all(row, &mut scratch.pairs);
                }
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
        );
        self.eq
            .round()
            .ok_or_else(|| eq_round_error(round))?
            .checked_cubic(
                block_lanes[0].reduce() + tail_mass,
                block_lanes[1].reduce(),
                previous_claim,
                round,
                || {
                    tail_mass
                        + self.eq.par_fold_out_in_prefix(
                            prefix_pairs,
                            || {
                                (
                                    vec![(F::zero(), F::zero()); num_polys],
                                    F::Accumulator::default(),
                                )
                            },
                            |(pairs, sum), row, _, weight| {
                                if let Some(paired) = &paired {
                                    paired.gather(row, pairs);
                                } else {
                                    tables.lo_hi_all(row, pairs);
                                }
                                let mut value = F::Accumulator::default();
                                for ((_, hi), rho) in
                                    pairs.iter().zip(gamma_powers).take(active_polys)
                                {
                                    value.fmadd(*hi, *hi - *rho);
                                }
                                sum.fmadd(weight, value.reduce());
                            },
                            |_, weight, (_, sum)| weight * sum.reduce(),
                            |a, b| a + b,
                        )
                },
            )
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
        // Disabled columns were retained unscaled; the unit factor preserves
        // their opening values when gamma is zero.
        let values: BTreeMap<JoltOpeningId, F> = self
            .openings
            .iter()
            .copied()
            .enumerate()
            .map(|(i, id)| (id, self.tables.value(i, 0) * self.opening_unscale[i]))
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
pub(crate) mod testing;

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests;

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "test module asserts successful categorical round reconstruction"
)]
mod categorical_tests;
