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
//! regrouping of the same sums/products yields identical field elements):
//!
//! - **Sparse one-hot access.** Each `ra_i(·, j)` is one-hot in the chunk
//!   domain, so the per-cycle hot index (a `u8`) is the whole polynomial.
//!   The kernels collect one typed bundle pass over the trace
//!   ([`RaAddressRow`]) and never materialize the `K × T` grids the naive
//!   tier's `oracle_table` walks (legacy `RaIndices`).
//! - **Pushforward `G` tables (address phase).** `G_i[k] = Σ_j eq(r_ref_cycle, j) ·
//!   ra_i(k, j)` collapses to a scatter of tensor-factored eq weights into a
//!   `K`-sized accumulator per polynomial — `O(N·T)` adds instead of the
//!   reference's `O(N·K·T)` multiplies over dense grids (legacy
//!   `compute_all_G`).
//! - **Shared expanded-eq gather (cycle phase).** The address-folded row
//!   `x_i(j) = eq(r_address)[hot_i(j)]` is a lookup into one shared
//!   `K`-sized table; per-polynomial tables are that table pre-scaled by
//!   `γ^i`, so `γ^{2i}(x² − x) = H(H − γ^i)` needs no batching multiply in
//!   the round loop (legacy `SharedRaPolynomials` pre-scaling).
//! - **Phase-staged materialization.** The cycle tables stay index-encoded
//!   for the first three binds (K-sized scale tables per branch), then
//!   materialize dense at `T/8` length — never `N·T` field elements up
//!   front (legacy `SharedRaRound1→2→3→N` state machine).
//! - **Split-eq / Gruen round messages (cycle phase).** Only the constant
//!   and leading coefficients of the inner quadratic are accumulated
//!   (two point-evaluations per pair instead of four full-summand
//!   evaluations); the cubic is reconstructed from `s(0)+s(1) =
//!   previous_claim` via [`GruenSplitEqPolynomial::gruen_poly_deg_3`]. The
//!   stage-6a `eq(r_address, reference_address)` scalar rides in the
//!   split-eq scaling factor, mirroring the reference's `EqAddressCycle`
//!   derived table exactly.
//! - **Rayon-parallel walks** over cycle chunks (pushforward) and
//!   polynomials (materialization), gated on the crate's `parallel`
//!   feature.
//!
//! The address phase's round loop runs over the `2^log_k_chunk`-point chunk
//! domain (16–256 points) — negligible next to the `T`-scale table
//! construction — so it reuses the reference kernel's explicit four-point
//! sampling verbatim; only the table construction is replaced.
//!
//! Cross-stage carry: the 6a `prepare` parks the per-cycle index columns in
//! the [`ProofSession`] under a module-private key; the 6b `prepare`
//! reclaims them (falling back to a fresh bundle pass when absent or
//! geometry-stale), so mixing this backend with the reference tier per slot
//! stays correct.

use std::collections::BTreeMap;

use jolt_claims::protocols::jolt::geometry::booleanity::BooleanityDimensions;
use jolt_claims::protocols::jolt::geometry::ra::{JoltRaPolynomial, JoltRaPolynomialLayout};
use jolt_claims::protocols::jolt::{
    BooleanityPublic, JoltDerivedId, JoltOpeningId, JoltRelationId,
};
use jolt_claims::{OutputClaims, Source, SymbolicSumcheck};
use jolt_field::Field;
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
use jolt_verifier::VerifierError;
use jolt_witness::witnesses::{LookupIndex, MappedPc, RaChunkSelector, RemappedRamAddress};
use jolt_witness::{collect_bundles, JoltWitnessPlane, WitnessBundle};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::reference::views::eq_table;
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

/// The per-cycle facts every booleanity chunk index derives from: the
/// instruction lookup index, the mapped bytecode PC (`None` = cold cycle),
/// and the remapped RAM word address (`None` = cold cycle). One bundle pass
/// serves all `N` chunk columns.
#[derive(Clone, Copy, Debug, WitnessBundle)]
struct RaAddressRow {
    lookup_index: LookupIndex,
    mapped_pc: MappedPc,
    remapped_ram_address: RemappedRamAddress,
}

/// Per checked polynomial (canonical layout order: instruction, bytecode,
/// ram), the hot chunk index per cycle; `None` is a cold cycle. The geometry
/// stamp guards the 6a→6b session carry against a stale or mismatched
/// reclaim.
struct RaIndexColumns {
    columns: Vec<Vec<Option<u8>>>,
    layout: JoltRaPolynomialLayout,
    log_t: usize,
    log_k_chunk: usize,
}

/// Module-private [`ProofSession`] key for the 6a→6b index-column carry.
struct BooleanityIndexCarry(RaIndexColumns);

/// One typed bundle pass over the cycle domain, then chunk-selector gathers
/// per checked polynomial — the sparse replacement for the naive tier's
/// `K × T` `oracle_table` grids. Shapes are validated against the relation's
/// dimensions up front, mirroring the reference kernel's size checks.
fn collect_index_columns<F: Field>(
    witness: &dyn JoltWitnessPlane<F>,
    dimensions: BooleanityDimensions,
) -> Result<RaIndexColumns, KernelError<F>> {
    let log_t = dimensions.log_t;
    let log_k_chunk = dimensions.log_k_chunk;
    let layout = dimensions.layout;
    if log_k_chunk > 8 {
        return Err(KernelError::Unsupported {
            reason: "optimized booleanity stores chunk indices as u8 (log_k_chunk > 8)",
        });
    }
    for opening in layout.openings(JoltRelationId::Booleanity) {
        let shape = witness.shape(opening.polynomial_id())?;
        if shape.log_rows != log_k_chunk + log_t {
            return Err(KernelError::TableSizeMismatch {
                table: format!("{opening:?}"),
                expected: 1usize << (log_k_chunk + log_t),
                got: shape.rows(),
            });
        }
    }

    let rows: Vec<RaAddressRow> = collect_bundles(witness, 1usize << log_t)?;
    let polynomials: Vec<JoltRaPolynomial> = layout.polynomials().collect();
    let column_for = |polynomial: &JoltRaPolynomial| -> Result<Vec<Option<u8>>, KernelError<F>> {
        // The selectors mirror the trace oracle's grid materializers
        // (`materialize_one_hot`), so gathered indices and the reference
        // tier's dense grids describe the same one-hot polynomials.
        Ok(match *polynomial {
            JoltRaPolynomial::Instruction(index) => {
                let selector = RaChunkSelector::new(index, layout.instruction(), log_k_chunk)?;
                rows.iter()
                    .map(|row| Some(selector.chunk_u128(row.lookup_index.0) as u8))
                    .collect()
            }
            JoltRaPolynomial::Bytecode(index) => {
                let selector = RaChunkSelector::new(index, layout.bytecode(), log_k_chunk)?;
                rows.iter()
                    .map(|row| row.mapped_pc.0.map(|pc| selector.chunk_usize(pc) as u8))
                    .collect()
            }
            JoltRaPolynomial::Ram(index) => {
                let selector = RaChunkSelector::new(index, layout.ram(), log_k_chunk)?;
                rows.iter()
                    .map(|row| {
                        row.remapped_ram_address
                            .0
                            .map(|address| selector.chunk_usize(address as usize) as u8)
                    })
                    .collect()
            }
        })
    };

    #[cfg(feature = "parallel")]
    let columns = polynomials
        .par_iter()
        .map(column_for)
        .collect::<Result<Vec<_>, _>>()?;
    #[cfg(not(feature = "parallel"))]
    let columns = polynomials
        .iter()
        .map(column_for)
        .collect::<Result<Vec<_>, _>>()?;

    Ok(RaIndexColumns {
        columns,
        layout,
        log_t,
        log_k_chunk,
    })
}

/// Cycle-chunk granularity of the parallel pushforward scatter.
const PUSHFORWARD_CHUNK: usize = 1 << 14;

/// The one-hot pushforward `G_i[k] = Σ_{j : hot_i(j) = k} eq(point, j)`:
/// per cycle, one tensor-factored eq product scattered into every column's
/// `K`-sized accumulator (legacy `compute_all_G` / `one_hot_pushforwards`).
/// Equals the reference's per-chunk cycle masses exactly — same terms,
/// regrouped.
fn cycle_pushforward<F: Field>(columns: &RaIndexColumns, point: &[F]) -> Vec<Vec<F>> {
    let cycles = 1usize << columns.log_t;
    let k_chunk = 1usize << columns.log_k_chunk;
    let eq = TensorEqTable::new(point);
    debug_assert_eq!(eq.len(), cycles);
    let zero = || vec![vec![F::zero(); k_chunk]; columns.columns.len()];
    let scatter = |mut accumulator: Vec<Vec<F>>, chunk_index: usize| {
        let start = chunk_index * PUSHFORWARD_CHUNK;
        let end = (start + PUSHFORWARD_CHUNK).min(cycles);
        for j in start..end {
            let eq_eval = eq.evaluate_index(j);
            for (column, table) in columns.columns.iter().zip(accumulator.iter_mut()) {
                if let Some(k) = column[j] {
                    table[k as usize] += eq_eval;
                }
            }
        }
        accumulator
    };
    let merge = |mut left: Vec<Vec<F>>, right: Vec<Vec<F>>| {
        for (left, right) in left.iter_mut().zip(right) {
            for (left, right) in left.iter_mut().zip(right) {
                *left += right;
            }
        }
        left
    };

    #[cfg(feature = "parallel")]
    {
        (0..cycles.div_ceil(PUSHFORWARD_CHUNK))
            .into_par_iter()
            .fold(zero, scatter)
            .reduce(zero, merge)
    }
    #[cfg(not(feature = "parallel"))]
    {
        let _ = merge;
        (0..cycles.div_ceil(PUSHFORWARD_CHUNK)).fold(zero(), scatter)
    }
}

// ---------------------------------------------------------------------------
// Stage 6a: address phase
// ---------------------------------------------------------------------------

/// Slot front for the stage-6a booleanity address phase.
pub struct OptimizedBooleanityAddress;

impl<F: Field> PrepareKernel<F, BooleanityAddressPhase<F>> for OptimizedBooleanityAddress {
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

        let columns = collect_index_columns(witness, dimensions)?;
        let masses = cycle_pushforward(&columns, &reference_cycle);
        // The index columns are pure witness data (challenge-free), so the
        // carry cannot go stale; the 6b cycle phase reclaims them and skips
        // its own trace pass.
        session.park(BooleanityIndexCarry(columns));

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
struct OptimizedBooleanityAddressKernel<F: Field> {
    rounds: usize,
    /// Per checked polynomial, its `γ^{2i}` batching weight, in the layout's
    /// canonical order.
    gamma_weights: Vec<F>,
    linear: Vec<Polynomial<F>>,
    /// Raw vectors because the squared-weight bind is not a multilinear bind.
    squared: Vec<Vec<F>>,
    eq_address: Polynomial<F>,
    rounds_bound: usize,
}

impl<F: Field> OptimizedBooleanityAddressKernel<F> {
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
            rounds,
            gamma_weights,
            linear,
            squared,
            eq_address: Polynomial::new(eq_table(reference_address)),
            rounds_bound: 0,
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
        self.rounds_bound += 1;
    }
}

impl<F: Field> ProveRounds<F> for OptimizedBooleanityAddressKernel<F> {
    fn num_rounds(&self) -> usize {
        self.rounds
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

impl<F: Field> SumcheckKernel<F> for OptimizedBooleanityAddressKernel<F> {
    type Relation = BooleanityAddressPhase<F>;

    fn output_claims(
        &mut self,
        _inputs: &SumcheckInputClaims<F, Self::Relation>,
    ) -> Result<BooleanityAddressPhaseOutputClaims<F>, SumcheckKernelError<F>> {
        if self.rounds_bound != self.rounds {
            return Err(SumcheckKernelError::NotFullyBound {
                remaining: self.rounds - self.rounds_bound,
            });
        }
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

impl<F: Field> PrepareKernel<F, Booleanity<F>> for OptimizedBooleanityCycle {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, Booleanity<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = Booleanity<F>>>, KernelError<F>> {
        let relation = inputs.relation;
        let dimensions = relation.dimensions();
        let layout = dimensions.layout;
        let r_address = relation.r_address();
        let reference_address = relation.reference_address();
        let reference_cycle = relation.reference_cycle();
        if r_address.len() != dimensions.log_k_chunk || reference_cycle.len() != dimensions.log_t {
            return Err(KernelError::InvariantViolation {
                reason: "booleanity cycle-phase point lengths disagree with the dimensions",
            });
        }
        // Fail closed on relation variants whose summand checks more
        // openings than the base RA layout (the akita lattice cycle phase):
        // this kernel serves exactly the layout's members.
        let expression = relation.symbolic().output_expression::<F>();
        let mut leaf_openings: Vec<JoltOpeningId> = expression
            .terms
            .iter()
            .flat_map(|term| &term.factors)
            .filter_map(|factor| match factor {
                Source::Opening(id) => Some(*id),
                _ => None,
            })
            .collect();
        leaf_openings.sort_unstable();
        leaf_openings.dedup();
        let mut layout_openings: Vec<JoltOpeningId> =
            layout.openings(JoltRelationId::Booleanity).collect();
        layout_openings.sort_unstable();
        if leaf_openings != layout_openings {
            return Err(KernelError::Unsupported {
                reason: "optimized booleanity cycle kernel serves the base RA layout only",
            });
        }

        let columns = match session.take::<BooleanityIndexCarry>() {
            Some(BooleanityIndexCarry(columns))
                if columns.layout == layout
                    && columns.log_t == dimensions.log_t
                    && columns.log_k_chunk == dimensions.log_k_chunk =>
            {
                columns
            }
            _ => collect_index_columns(witness, dimensions)?,
        };

        // The fixed address eq factor of the `EqAddressCycle` public; rides
        // in the split-eq scaling so round messages and the bound scalar
        // carry it exactly like the reference's derived table.
        let address_scalar = try_eq_mle(r_address, reference_address).map_err(|_| {
            KernelError::InvariantViolation {
                reason: "booleanity address point and reference length mismatch",
            }
        })?;
        let eq_address = eq_table(r_address);
        let (gamma_powers, gamma_powers_inv) =
            gamma_power_pairs(inputs.challenges.gamma, layout.total())?;
        let tables: Vec<Vec<F>> = gamma_powers
            .iter()
            .map(|rho| eq_address.iter().map(|eq| *rho * *eq).collect())
            .collect();

        Ok(Box::new(OptimizedBooleanityCycleKernel {
            rounds: relation.rounds(),
            eq: GruenSplitEqPolynomial::new_with_scaling(
                reference_cycle,
                BindingOrder::LowToHigh,
                Some(address_scalar),
            ),
            tables: SharedRaTables::new(tables, columns),
            gamma_powers,
            gamma_powers_inv,
            layout,
            rounds_bound: 0,
        }))
    }
}

/// `(γ^i, γ^{-i})` pairs for the pre-scaled shared tables. The inverse
/// powers unscale the final claims back to the committed polynomials'
/// values; `γ^i · γ^{-i} = 1` exactly, so unscaling is byte-exact.
fn gamma_power_pairs<F: Field>(gamma: F, count: usize) -> Result<(Vec<F>, Vec<F>), KernelError<F>> {
    let gamma_inv = gamma.inverse().ok_or(KernelError::InvariantViolation {
        reason: "booleanity batching gamma must be invertible",
    })?;
    let mut powers = Vec::with_capacity(count);
    let mut powers_inv = Vec::with_capacity(count);
    let mut power = F::one();
    let mut power_inv = F::one();
    for _ in 0..count {
        powers.push(power);
        powers_inv.push(power_inv);
        power *= gamma;
        power_inv *= gamma_inv;
    }
    Ok((powers, powers_inv))
}

/// The legacy `SharedRaPolynomials` state machine: the address-folded cycle
/// rows of all `N` one-hot polynomials, served as gathers into shared
/// `K`-sized scale tables for the first three binds, then materialized
/// dense at `T/8` length. `LowToHigh` binding only (the booleanity
/// convention).
enum SharedRaTables<F: Field> {
    /// No binds yet: `value(i, j) = tables[i][hot_i(j)]` (0 when cold).
    Fresh {
        tables: Vec<Vec<F>>,
        columns: RaIndexColumns,
    },
    /// One bind: branch tables scaled by `(1−r)` / `r`.
    Bound1 {
        branch: [Vec<Vec<F>>; 2],
        columns: RaIndexColumns,
    },
    /// Two binds: branch tables indexed by the two bound bits `(b0, b1)`
    /// packed as `b0 + 2·b1` — the original index's low bits.
    Bound2 {
        branch: [Vec<Vec<F>>; 4],
        columns: RaIndexColumns,
    },
    /// Three or more binds: plain dense multilinears.
    Dense(Vec<Polynomial<F>>),
}

impl<F: Field> SharedRaTables<F> {
    fn new(tables: Vec<Vec<F>>, columns: RaIndexColumns) -> Self {
        Self::Fresh { tables, columns }
    }

    /// The current (bound) evaluation of polynomial `i` at index `j`: the
    /// eq-weighted sum of the surviving original indices, exactly the value
    /// the reference's dense table holds after the same binds.
    #[inline]
    fn value(&self, i: usize, j: usize) -> F {
        let gather = |tables: &[Vec<Vec<F>>], columns: &RaIndexColumns, j: usize| -> F {
            let width = tables.len();
            let column = &columns.columns[i];
            let mut sum = F::zero();
            for (offset, table) in tables.iter().enumerate() {
                if let Some(k) = column[j * width + offset] {
                    sum += table[i][k as usize];
                }
            }
            sum
        };
        match self {
            Self::Fresh { tables, columns } => {
                columns.columns[i][j].map_or_else(F::zero, |k| tables[i][k as usize])
            }
            Self::Bound1 { branch, columns } => gather(branch, columns, j),
            Self::Bound2 { branch, columns } => gather(branch, columns, j),
            Self::Dense(polys) => polys[i].evals()[j],
        }
    }

    /// The fully bound claim of polynomial `i` (any state, so short
    /// geometries with fewer than three rounds extract correctly).
    fn final_value(&self, i: usize) -> F {
        self.value(i, 0)
    }

    fn bind(&mut self, challenge: F) {
        let scale = |tables: &[Vec<F>], factor: F| -> Vec<Vec<F>> {
            tables
                .iter()
                .map(|table| table.iter().map(|value| factor * *value).collect())
                .collect()
        };
        let one_minus = F::one() - challenge;
        *self = match std::mem::replace(self, Self::Dense(Vec::new())) {
            Self::Fresh { tables, columns } => Self::Bound1 {
                branch: [scale(&tables, one_minus), scale(&tables, challenge)],
                columns,
            },
            Self::Bound1 {
                branch: [zero, one],
                columns,
            } => Self::Bound2 {
                // Packing: index = b0 + 2·b1 (b0 = first bound bit).
                branch: [
                    scale(&zero, one_minus),
                    scale(&one, one_minus),
                    scale(&zero, challenge),
                    scale(&one, challenge),
                ],
                columns,
            },
            Self::Bound2 { branch, columns } => {
                Self::Dense(materialize(&branch, &columns, challenge))
            }
            Self::Dense(mut polys) => {
                for poly in &mut polys {
                    poly.bind_with_order(challenge, BindingOrder::LowToHigh);
                }
                Self::Dense(polys)
            }
        };
    }
}

/// The third bind's materialization: eight branch scale tables (the three
/// bound low bits), gathered per polynomial into a dense `T/8`-length
/// vector (legacy `SharedRaRound3::bind`).
fn materialize<F: Field>(
    branch: &[Vec<Vec<F>>; 4],
    columns: &RaIndexColumns,
    challenge: F,
) -> Vec<Polynomial<F>> {
    let one_minus = F::one() - challenge;
    let scale = |tables: &[Vec<F>], factor: F| -> Vec<Vec<F>> {
        tables
            .iter()
            .map(|table| table.iter().map(|value| factor * *value).collect())
            .collect()
    };
    // Packing: index = b0 + 2·b1 + 4·b2.
    let tables: Vec<Vec<Vec<F>>> = (0..8)
        .map(|offset| {
            let factor = if offset >= 4 { challenge } else { one_minus };
            scale(&branch[offset % 4], factor)
        })
        .collect();

    let materialize_poly = |column: &[Option<u8>], i: usize| -> Polynomial<F> {
        let new_len = column.len() / 8;
        let evals: Vec<F> = (0..new_len)
            .map(|j| {
                let mut sum = F::zero();
                for (offset, table) in tables.iter().enumerate() {
                    if let Some(k) = column[8 * j + offset] {
                        sum += table[i][k as usize];
                    }
                }
                sum
            })
            .collect();
        Polynomial::new(evals)
    };

    #[cfg(feature = "parallel")]
    {
        columns
            .columns
            .par_iter()
            .enumerate()
            .map(|(i, column)| materialize_poly(column, i))
            .collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        columns
            .columns
            .iter()
            .enumerate()
            .map(|(i, column)| materialize_poly(column, i))
            .collect()
    }
}

struct OptimizedBooleanityCycleKernel<F: Field> {
    rounds: usize,
    /// Split-eq over the reference cycle, scaled by
    /// `eq(r_address, reference_address)` — together the reference's
    /// `EqAddressCycle` derived table.
    eq: GruenSplitEqPolynomial<F>,
    /// Pre-scaled (`γ^i`) shared address-folded tables.
    tables: SharedRaTables<F>,
    gamma_powers: Vec<F>,
    gamma_powers_inv: Vec<F>,
    layout: JoltRaPolynomialLayout,
    rounds_bound: usize,
}

impl<F: Field> OptimizedBooleanityCycleKernel<F> {
    fn bind(&mut self, challenge: F) {
        self.eq.bind(challenge);
        self.tables.bind(challenge);
        self.rounds_bound += 1;
    }
}

impl<F: Field> ProveRounds<F> for OptimizedBooleanityCycleKernel<F> {
    fn num_rounds(&self) -> usize {
        self.rounds
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
        let tables = &self.tables;
        let gamma_powers = &self.gamma_powers;
        // Inner quadratic `q(X) = Σ_j eq_rest(j) · Σ_i (H_i(X)² − γ^i·H_i(X))`:
        // constant coefficient from `H` at 0, leading coefficient from the
        // pair delta — the pre-scaling makes `γ^{2i}(x² − x) = H(H − γ^i)`.
        let [constant, leading] = self.eq.par_fold_out_in(
            || [F::zero(); 2],
            |accumulator, row, _x_in, e_in| {
                let mut pair_constant = F::zero();
                let mut pair_leading = F::zero();
                for (i, rho) in gamma_powers.iter().enumerate() {
                    let h_0 = tables.value(i, 2 * row);
                    let h_1 = tables.value(i, 2 * row + 1);
                    let delta = h_1 - h_0;
                    pair_constant += h_0 * (h_0 - *rho);
                    pair_leading += delta * delta;
                }
                accumulator[0] += e_in * pair_constant;
                accumulator[1] += e_in * pair_leading;
            },
            |_x_out, e_out, inner| [e_out * inner[0], e_out * inner[1]],
            |left, right| [left[0] + right[0], left[1] + right[1]],
        );
        Ok(self.eq.gruen_poly_deg_3(constant, leading, previous_claim))
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind);
        Ok(())
    }
}

impl<F: Field> SumcheckKernel<F> for OptimizedBooleanityCycleKernel<F> {
    type Relation = Booleanity<F>;

    fn output_claims(
        &mut self,
        _inputs: &SumcheckInputClaims<F, Self::Relation>,
    ) -> Result<SumcheckOutputClaims<F, Self::Relation>, SumcheckKernelError<F>> {
        if self.rounds_bound != self.rounds {
            return Err(SumcheckKernelError::NotFullyBound {
                remaining: self.rounds - self.rounds_bound,
            });
        }
        // Unscale the pre-scaled tables back to the committed polynomials'
        // claims; resolve by id so the output struct shape stays the
        // relation's business.
        let values: BTreeMap<JoltOpeningId, F> = self
            .layout
            .openings(JoltRelationId::Booleanity)
            .enumerate()
            .map(|(i, id)| (id, self.tables.final_value(i) * self.gamma_powers_inv[i]))
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
        if self.rounds_bound != self.rounds {
            return Err(SumcheckKernelError::NotFullyBound {
                remaining: self.rounds - self.rounds_bound,
            });
        }
        let id = JoltDerivedId::from(BooleanityPublic::EqAddressCycle);
        let expected =
            match relation.derive_output_term(&id, input_points, output_points, challenges) {
                Ok(value) => value,
                Err(VerifierError::MissingStageClaimDerived { .. }) => return Ok(()),
                Err(error) => return Err(error.into()),
            };
        let got = self.eq.current_scalar();
        if got != expected {
            return Err(SumcheckKernelError::DerivedTableDrift { id, expected, got });
        }
        Ok(())
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
        f: impl FnOnce(&TraceBackend<'_, OwnedTrace>, BooleanityDimensions) -> R,
    ) -> R {
        let instruction_a = JoltInstructionRow {
            instruction_kind: JoltInstructionKind::ADDI,
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
            address: 0x8000_0004,
            operands: NormalizedOperands {
                rd: Some(3),
                rs1: Some(1),
                rs2: None,
                imm: 113,
            },
            ..instruction_a
        };
        let preprocessing = JoltProgramPreprocessing {
            bytecode: BytecodePreprocessing::preprocess(
                vec![instruction_a, instruction_b],
                instruction_a.address as u64,
                RV64IMAC_JOLT,
            )
            .unwrap(),
            ram: RAMPreprocessing::default(),
            memory_layout: Default::default(),
            max_padded_trace_length: 4.max(1 << log_t),
        };
        let program = JoltProgram::default();
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
                    value: 7,
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
                    rd: Some(RegisterWrite {
                        register: 3,
                        pre_value: 0,
                        post_value: 121,
                    }),
                    ..Default::default()
                },
                RamAccess::Write(RamWrite {
                    address: 0x8000_1008,
                    pre_value: 7,
                    post_value: 11,
                }),
            ),
            // Cold bytecode, hot RAM.
            row(
                None,
                RegisterState::default(),
                RamAccess::Write(RamWrite {
                    address: 0x8000_1010,
                    pre_value: 0,
                    post_value: 5,
                }),
            ),
            // Hot bytecode, cold RAM.
            row(
                Some(instruction_a),
                RegisterState {
                    rs1: Some(RegisterRead {
                        register: 2,
                        value: 8,
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
            TraceOutput::new(OwnedTrace::new(rows), Default::default(), None),
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
        use jolt_field::FromPrimitiveInt;
        Fr::from_u64(0x1234_5678 + 1000 * round as u64 + 7)
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use jolt_claims::protocols::jolt::JoltChallengeId;
    use jolt_claims::{InputClaims, OutputClaims, SumcheckChallenges};
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_poly::EqPolynomial;
    use jolt_verifier::stages::relations::ConcreteSumcheckChallenges;
    use jolt_verifier::stages::stage6b::booleanity::BooleanityInputClaims;
    use jolt_witness::JoltWitnessOracle;

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
        for opening in dimensions.layout.openings(JoltRelationId::Booleanity) {
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
            // The 6a prepare parks the index columns for the 6b cycle phase.
            assert!(session.state::<BooleanityIndexCarry>().is_some());
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

    fn cycle_parity(log_t: usize, log_k_chunk: u8, carried_indices: bool) {
        with_booleanity_backend(log_t, log_k_chunk, |backend, dimensions| {
            let r_address = point(110, dimensions.log_k_chunk);
            let reference_address = point(700, dimensions.log_k_chunk);
            let reference_cycle = point(400, log_t);
            let gamma = Fr::from_u64(31);
            let relation = Booleanity::new(
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
                session.park(BooleanityIndexCarry(
                    collect_index_columns::<Fr>(backend, dimensions).unwrap(),
                ));
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
                    session.state::<BooleanityIndexCarry>().is_none(),
                    "cycle prepare must consume the parked index columns"
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

    /// `log_t = 4` drives the shared-table state machine through
    /// materialization (three staged binds) into dense rounds.
    #[test]
    fn cycle_kernel_matches_reference_through_dense_rounds() {
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
            let cycle_relation = Booleanity::new(
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
                session.state::<BooleanityIndexCarry>().is_none(),
                "cycle prepare must consume the 6a-parked index columns"
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
