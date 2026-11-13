//! Optimized instruction RA virtualization (stage 6b) kernel.
//!
//! The summand is `eq(r_cycle, j) · Σ_v γ^v · Π_{i<N} ra_{N·v+i}(chunk_i, j)`
//! over the cycle domain. The reference kernel materializes every committed
//! one-hot `(K × T)` grid through `oracle_table` and address-folds it (`K·T`
//! multiply-adds per committed chunk), then interprets the relation
//! expression through the naive prover. This kernel exploits the one-hot
//! structure instead: each committed `InstructionRa` chunk is a point mass at
//! that chunk of the cycle's 128-bit lookup index, so the address-folded
//! value is a single eq-table lookup,
//!
//! `ra_i(r_chunk_i, j) = eq(r_chunk_i, chunk_i(k_j)) = eq_table_i[chunk_i(k_j)]`
//!
//! — `T` lookups per committed chunk instead of a `K × T` grid walk, and no
//! grid materialization at all. The per-cycle lookup indices are reclaimed
//! from the [`ProofSession`] through `SharedInstructionRows` when the stage-5
//! optimized kernel already collected them, or collected fresh otherwise.
//!
//! Round messages use the Gruen split-eq factorization: `eq(r_cycle, ·)` is
//! never materialized or bound as a `T`-sized table; each round computes the
//! inner factor `q(t) = Σ_y E_out·E_in · Σ_v γ^v Π_i ra(t, y)` on the grid
//! `[1, …, N−1, ∞]` with deferred-reduction accumulation (per-batch `γ^v`
//! pre-scaled into the batch's first table, so the row loop is pure
//! products), recovers `q(0)` from `s(0) + s(1) = previous_claim`, and
//! recomposes `s = ℓ · q` (legacy `compute_mles_product_sum` /
//! `finish_mles_product_sum_from_evals`). The emitted coefficient vector is
//! the unique degree-`(N+1)` polynomial through the same values the naive
//! prover interpolates, so round polynomials and output claims are
//! byte-identical (field arithmetic is exact under any regrouping, and
//! deferred reduction is exact mod `p`).
//!
//! The bound eq factor is pinned to the verifier's scalar path by
//! [`validate_derived_tables`](crate::SumcheckKernel::validate_derived_tables):
//! the fully-bound Gruen scalar must equal `derive_output_term(EqCycle)`,
//! exactly as the naive tier's bound `EqCycle` table is checked.

use crate::optimized::support::eq_round_error;
mod categorical;
mod streaming;

use std::sync::Arc;

pub use streaming::StreamingInstructionRaVirtualization;
use streaming::{rows_unavailable, LookupIndexDigits};

use super::constant_suffix::ConstantSuffix;
use super::instruction_read_raf::{InstructionCycleRow, LookupIndexChunks};
use super::lazy_ra::LazyFoldedRa;
use super::streaming::StreamedFoldedRa;
use super::support::{
    accumulate_product_grid, map_indices, pin_derived_term, product_grid_scratch_len,
    GruenRoundMessage, RoundProgress, MAX_GRID_FACTORS,
};
use crate::reference::views::eq_table;
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};
use jolt_claims::protocols::jolt::geometry::dimensions::committed_address_chunks;
use jolt_claims::protocols::jolt::relations::instruction::InstructionRaVirtualizationOutputClaims;
use jolt_claims::protocols::jolt::{InstructionRaVirtualizationPublic, JoltDerivedId};
use jolt_field::{Accumulator, JoltField};
use jolt_poly::{BindingOrder, GruenSplitEqPolynomial, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::{
    ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckInputPoints, SumcheckOutputPoints,
};
use jolt_verifier::stages::stage6b::instruction_ra_virtualization::InstructionRaVirtualization;
use jolt_witness::JoltWitnessPlane;

pub struct OptimizedInstructionRaVirtualization;

impl<F: JoltField> PrepareKernel<F, InstructionRaVirtualization<F>>
    for OptimizedInstructionRaVirtualization
{
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, InstructionRaVirtualization<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = InstructionRaVirtualization<F>>>, KernelError<F>>
    {
        let relation = inputs.relation;
        let cycles = 1usize << relation.dimensions().log_t();
        let rows = InstructionCycleRow::shared(session, witness, cycles)?;
        Ok(Box::new(OptimizedInstructionRaVirtualizationKernel::new(
            relation,
            rows,
            inputs.challenges.gamma,
        )?))
    }
}

/// The committed RA selectors: the default family, or the streaming family
/// until the bind that gathers it dense.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
enum Selectors<F: JoltField> {
    Lazy(LazyFoldedRa<F, LookupIndexChunks>),
    Streamed(StreamedFoldedRa<F, LookupIndexDigits>),
}

#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
pub struct OptimizedInstructionRaVirtualizationKernel<F: JoltField> {
    progress: RoundProgress,
    num_committed_per_virtual: usize,
    /// Inverse batch weights for active columns, and one for disabled columns
    /// retained unscaled so their final opening values remain available.
    opening_unscale: Vec<F>,
    active_virtuals: usize,
    /// Address-folded committed RA selectors, one per committed chunk:
    /// `folded[i][j] = eq(r_chunk_i, chunk_i(k_j))` — with each virtual
    /// batch's first table pre-scaled by `γ^v` so the round loop needs no
    /// batching multiplies — served lazily off the shared rows for the
    /// first four binds instead of `N × T` dense.
    folded_ra: Selectors<F>,
    gruen: GruenSplitEqPolynomial<F>,
}

impl<F: JoltField> OptimizedInstructionRaVirtualizationKernel<F> {
    pub(crate) fn new(
        relation: &InstructionRaVirtualization<F>,
        rows: Arc<ConstantSuffix<InstructionCycleRow>>,
        gamma: F,
    ) -> Result<Self, KernelError<F>> {
        let log_t = relation.dimensions().log_t();
        if rows.len() != 1 << log_t {
            return Err(KernelError::TableSizeMismatch {
                table: "stage-6b instruction rows".to_owned(),
                expected: 1 << log_t,
                got: rows.len(),
            });
        }
        Self::with_selectors(relation, gamma, |tables| {
            let source =
                LookupIndexChunks::new(rows, tables.len(), relation.committed_chunk_bits());
            Ok(Selectors::Lazy(LazyFoldedRa::new(tables, source)))
        })
    }

    /// The kernel over the selector family `selectors` builds from the
    /// committed chunks' scale tables, in chunk order.
    fn with_selectors(
        relation: &InstructionRaVirtualization<F>,
        gamma: F,
        selectors: impl FnOnce(Vec<Vec<F>>) -> Result<Selectors<F>, KernelError<F>>,
    ) -> Result<Self, KernelError<F>> {
        let log_t = relation.dimensions().log_t();
        let num_virtual = relation.dimensions().num_virtual_ra_polys();
        let num_committed_per_virtual = relation.dimensions().num_committed_per_virtual();
        let instruction_address = relation.instruction_address();
        let instruction_read_raf_cycle = relation.instruction_read_raf_cycle();
        let committed_chunk_bits = relation.committed_chunk_bits();
        let num_committed = num_virtual * num_committed_per_virtual;
        let chunks = committed_address_chunks(instruction_address, committed_chunk_bits);
        if chunks.len() != num_committed
            || instruction_address.len() != num_committed * committed_chunk_bits
        {
            return Err(KernelError::InvariantViolation {
                reason: "instruction address chunk count disagrees with the committed RA count",
            });
        }
        if committed_chunk_bits == 0 || committed_chunk_bits > 32 {
            return Err(KernelError::Unsupported {
                reason: "committed RA chunk width outside the supported one-hot range",
            });
        }
        if num_committed_per_virtual > MAX_GRID_FACTORS {
            return Err(KernelError::Unsupported {
                reason: "more committed RA chunks per virtual polynomial than the product grid \
                         supports",
            });
        }
        if instruction_read_raf_cycle.len() != log_t {
            return Err(KernelError::TableSizeMismatch {
                table: "instruction read-RAF cycle point".to_owned(),
                expected: log_t,
                got: instruction_read_raf_cycle.len(),
            });
        }

        let gamma_inv = gamma.inverse().unwrap_or_else(F::one);
        let mut gamma_powers = Vec::with_capacity(num_virtual);
        let mut opening_unscale = Vec::with_capacity(num_virtual);
        let mut power = F::one();
        let mut power_inv = F::one();
        for _ in 0..num_virtual {
            gamma_powers.push(power);
            opening_unscale.push(power_inv);
            power *= gamma;
            power_inv *= gamma_inv;
        }

        // One eq table per committed chunk point (each `2^w` entries); the
        // point-mass fold stays lazy — one table lookup per gathered cycle —
        // instead of materializing `N × T` dense selectors up front. Each
        // virtual batch's `γ^v` weight rides in the batch's first table.
        let chunk_tables: Vec<Vec<F>> = map_indices(chunks.len(), |i| {
            let mut table = eq_table(&chunks[i]);
            if i % num_committed_per_virtual == 0 {
                let weight = gamma_powers[i / num_committed_per_virtual];
                if !weight.is_zero() && weight != F::one() {
                    for value in &mut table {
                        *value *= weight;
                    }
                }
            }
            table
        });
        let folded_ra = selectors(chunk_tables)?;

        Ok(Self {
            progress: RoundProgress::new(log_t),
            num_committed_per_virtual,
            opening_unscale,
            active_virtuals: gamma_powers
                .iter()
                .take_while(|power| !power.is_zero())
                .count(),
            folded_ra,
            gruen: GruenSplitEqPolynomial::new(instruction_read_raf_cycle, BindingOrder::LowToHigh),
        })
    }

    /// `s(t) = ℓ(t) · q(t)` with
    /// `q(t) = Σ_y E(y) · Σ_v γ^v Π_{i<N} ra_{N·v+i}(t, y)` (the `γ^v` live
    /// in the pre-scaled tables): `q` is evaluated on the grid
    /// `[1, …, N−1, ∞]` with deferred-reduction accumulation at every level
    /// (per-row product lanes, per-block `e_in` folds, cross-block `e_out`
    /// folds), `q(0)` is recovered from `s(0) + s(1) = previous_claim`, and
    /// [`jolt_poly::GruenRound::gruen_poly_from_evals`] recomposes the
    /// unique degree-`(N+1)` coefficient vector.
    fn message(
        &self,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        let folded_ra = match &self.folded_ra {
            Selectors::Lazy(folded_ra) => folded_ra,
            Selectors::Streamed(selectors) => {
                return self.streamed_message(selectors, round, previous_claim);
            }
        };
        let n = self.num_committed_per_virtual;
        if n < 2 {
            return self.message_single_factor(folded_ra, round, previous_claim);
        }
        let num_committed = folded_ra.num_polys();
        let paired = folded_ra.paired_lo_hi(num_committed);
        let gather = |row, pairs: &mut [(F, F)]| {
            if let Some(paired) = &paired {
                paired.gather(row, pairs);
            } else {
                folded_ra.lo_hi_all(row, pairs);
            }
        };
        let prefix_pairs = folded_ra.prefix_len().div_ceil(2);
        let constant_groups: Vec<_> = folded_ra
            .tail_values()
            .chunks_exact(n)
            .take(self.active_virtuals)
            .map(|values| (values, values.iter().copied().product::<F>()))
            .collect();
        let tail = constant_groups
            .iter()
            .map(|(_, product)| *product)
            .sum::<F>();
        let tail_mass = self.gruen.suffix_weight(prefix_pairs) * tail;

        struct Scratch<F: JoltField> {
            lanes: Vec<F::Accumulator>,
            row_lanes: Vec<F::Accumulator>,
            pairs: Vec<(F, F)>,
            evals: Vec<F>,
            steps: Vec<F>,
            grid: Vec<F>,
        }

        let block_lanes = if let Some(lanes) = self.categorical_lanes(prefix_pairs) {
            lanes
        } else {
            self.gruen.par_fold_out_in_prefix(
                prefix_pairs,
                || Scratch {
                    lanes: vec![F::Accumulator::default(); n],
                    row_lanes: vec![F::Accumulator::default(); n],
                    pairs: vec![(F::zero(), F::zero()); num_committed],
                    evals: vec![F::zero(); n],
                    steps: vec![F::zero(); n],
                    grid: vec![F::zero(); product_grid_scratch_len(n)],
                },
                |scratch, row, _x_in, e_in| {
                    gather(row, &mut scratch.pairs);
                    for lane in &mut scratch.row_lanes {
                        *lane = F::Accumulator::default();
                    }
                    for (group, pairs) in scratch
                        .pairs
                        .chunks_exact(n)
                        .take(self.active_virtuals)
                        .enumerate()
                    {
                        if let Some((values, product)) = constant_groups.get(group) {
                            if pairs
                                .iter()
                                .zip(*values)
                                .all(|((lo, hi), value)| lo == value && hi == value)
                            {
                                // Equal endpoints make this product constant; its
                                // degree-n coefficient (the last lane) is zero.
                                for lane in &mut scratch.row_lanes[..n - 1] {
                                    lane.add(*product);
                                }
                                continue;
                            }
                        }
                        for ((pair, eval), step) in pairs
                            .iter()
                            .zip(scratch.evals.iter_mut())
                            .zip(scratch.steps.iter_mut())
                        {
                            *eval = pair.1;
                            *step = pair.1 - pair.0;
                        }
                        accumulate_product_grid(
                            &scratch.evals,
                            &scratch.steps,
                            &mut scratch.row_lanes,
                            &mut scratch.grid,
                        );
                    }
                    for (lane, row_lane) in scratch.lanes.iter_mut().zip(&scratch.row_lanes) {
                        lane.fmadd(e_in, row_lane.reduce());
                    }
                },
                |_x_out, e_out, scratch| {
                    let mut out = vec![F::Accumulator::default(); n];
                    for (out, lane) in out.iter_mut().zip(scratch.lanes) {
                        out.fmadd(e_out, lane.reduce());
                    }
                    out
                },
                |mut a, b| {
                    for (a, b) in a.iter_mut().zip(b) {
                        a.merge(b);
                    }
                    a
                },
            )
        };

        let mut q_evals: Vec<F> = block_lanes.into_iter().map(|lane| lane.reduce()).collect();
        // Finite grid values receive the constant suffix; its leading
        // coefficient (the infinity evaluation) is zero.
        for value in q_evals.iter_mut().take(n - 1) {
            *value += tail_mass;
        }
        self.gruen
            .round()
            .ok_or_else(|| eq_round_error(round))?
            .checked_toom(&q_evals, previous_claim, round, || {
                tail_mass
                    + self.gruen.par_fold_out_in_prefix(
                        prefix_pairs,
                        || {
                            (
                                vec![(F::zero(), F::zero()); num_committed],
                                F::Accumulator::default(),
                            )
                        },
                        |(pairs, sum), row, _, weight| {
                            gather(row, pairs);
                            let value = pairs.chunks_exact(n).take(self.active_virtuals).fold(
                                F::zero(),
                                |sum, factors| {
                                    sum + factors
                                        .iter()
                                        .fold(F::one(), |product, pair| product * pair.0)
                                },
                            );
                            sum.fmadd(weight, value);
                        },
                        |_, weight, (_, sum)| weight * sum.reduce(),
                        |a, b| a + b,
                    )
            })
    }

    /// Degenerate `N = 1` geometry (virtual = committed): the grid recovery
    /// assumes `q(1)` is sampled, so fall back to explicit `t = 0..=2`
    /// sampling of the quadratic summand.
    fn message_single_factor(
        &self,
        folded_ra: &LazyFoldedRa<F, LookupIndexChunks>,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        let num_committed = folded_ra.num_polys();
        let prefix_pairs = folded_ra.prefix_len().div_ceil(2);
        let tail = folded_ra
            .tail_values()
            .iter()
            .take(self.active_virtuals)
            .copied()
            .sum::<F>();
        let tail_mass = self.gruen.suffix_weight(prefix_pairs) * tail;
        let mut q_evals = self.gruen.par_fold_out_in_prefix(
            prefix_pairs,
            || ([F::zero(); 3], vec![(F::zero(), F::zero()); num_committed]),
            |(acc, pairs), row, _x_in, e_in| {
                folded_ra.lo_hi_all(row, pairs);
                let mut at_0 = F::zero();
                let mut at_1 = F::zero();
                let mut at_2 = F::zero();
                for (lo, hi) in pairs.iter().take(self.active_virtuals) {
                    at_0 += *lo;
                    at_1 += *hi;
                    at_2 += *hi + *hi - *lo;
                }
                acc[0] += e_in * at_0;
                acc[1] += e_in * at_1;
                acc[2] += e_in * at_2;
            },
            |_x_out, e_out, (mut acc, _)| {
                for value in &mut acc {
                    *value *= e_out;
                }
                acc
            },
            |mut a, b| {
                for (a, b) in a.iter_mut().zip(&b) {
                    *a += *b;
                }
                a
            },
        );
        for value in &mut q_evals {
            *value += tail_mass;
        }

        self.gruen
            .round()
            .ok_or_else(|| eq_round_error(round))?
            .checked_round_poly(&mut q_evals, previous_claim, round)
    }

    fn bind(&mut self, challenge: F) -> Result<(), SumcheckError<F>> {
        let round = self.progress.bound();
        self.gruen
            .round_mut()
            .ok_or_else(|| eq_round_error(round))?
            .bind(challenge);
        match &mut self.folded_ra {
            Selectors::Lazy(folded_ra) => folded_ra.bind(challenge),
            Selectors::Streamed(selectors) => {
                selectors.bind(challenge);
                if let Some(dense) = selectors.materialized().map_err(rows_unavailable)? {
                    self.folded_ra = Selectors::Lazy(dense);
                }
            }
        }
        self.progress.advance();
        Ok(())
    }
}

impl<F: JoltField> ProveRounds<F> for OptimizedInstructionRaVirtualizationKernel<F> {
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
        self.message(round, previous_claim)
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind)
    }
}

impl<F: JoltField> SumcheckKernel<F> for OptimizedInstructionRaVirtualizationKernel<F> {
    type Relation = InstructionRaVirtualization<F>;

    fn output_claims(
        &mut self,
        _inputs: &SumcheckInputClaims<F, Self::Relation>,
    ) -> Result<InstructionRaVirtualizationOutputClaims<F>, SumcheckKernelError<F>> {
        self.progress.require_complete()?;
        let Selectors::Lazy(folded_ra) = &self.folded_ra else {
            return Err(SumcheckKernelError::InvariantViolation {
                reason: "streamed instruction RA selectors outlived their last bind",
            });
        };
        // Unscale the batch-first tables' γ^v pre-scaling back to the
        // committed polynomials' claims.
        let mut committed_instruction_ra = folded_ra.final_values();
        for (index, value) in committed_instruction_ra.iter_mut().enumerate() {
            if index % self.num_committed_per_virtual == 0 {
                *value *= self.opening_unscale[index / self.num_committed_per_virtual];
            }
        }
        Ok(InstructionRaVirtualizationOutputClaims {
            committed_instruction_ra,
        })
    }

    /// The Gruen scalar after full binding is the bound `EqCycle` value; pin
    /// it to the verifier's `derive_output_term`, exactly as the naive tier's
    /// materialized eq table is pinned.
    fn validate_derived_tables(
        &self,
        relation: &Self::Relation,
        input_points: &SumcheckInputPoints<F, Self::Relation>,
        output_points: &SumcheckOutputPoints<F, Self::Relation>,
        challenges: &ConcreteSumcheckChallenges<F, Self::Relation>,
    ) -> Result<(), SumcheckKernelError<F>> {
        self.progress.require_complete()?;
        let id = JoltDerivedId::from(InstructionRaVirtualizationPublic::EqCycle);
        pin_derived_term(
            relation,
            id,
            input_points,
            output_points,
            challenges,
            self.gruen.current_scalar(),
        )
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests;
