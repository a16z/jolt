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
//! from the [`ProofSession`] when a stage-5/6a/6b co-consumer already
//! collected the column (see
//! [`SharedLookupIndices`](super::instruction_read_raf::SharedLookupIndices)),
//! or collected fresh otherwise.
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

use std::sync::Arc;

use jolt_claims::protocols::jolt::geometry::dimensions::committed_address_chunks;
use jolt_claims::protocols::jolt::relations::instruction::InstructionRaVirtualizationOutputClaims;
use jolt_claims::protocols::jolt::{InstructionRaVirtualizationPublic, JoltDerivedId};
use jolt_field::{AdditiveAccumulator, Field, RingAccumulator};
use jolt_poly::{BindingOrder, GruenSplitEqPolynomial, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::{
    ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckInputPoints, SumcheckOutputPoints,
};
use jolt_verifier::stages::stage6b::instruction_ra_virtualization::InstructionRaVirtualization;
use jolt_witness::JoltWitnessPlane;

use super::instruction_read_raf::{shared_lookup_indices, SharedLookupIndices};
use super::lazy_ra::{ChunkIndexSource, LazyFoldedRa};
use super::support::{
    accumulate_product_grid, gamma_power_pairs, map_indices, pin_derived_term, GruenRoundMessage,
    RoundProgress,
};
use crate::reference::views::eq_table;
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

/// Optimized [`PrepareKernel`] implementor for the
/// `instruction_ra_virtualization` slot.
pub struct OptimizedInstructionRaVirtualization;

impl<F: Field> PrepareKernel<F, InstructionRaVirtualization<F>>
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
        let indices = shared_lookup_indices(session, witness, cycles)?;
        // This is the lookup-index column's last consumer (booleanity, the
        // only other stage-6b reader, prepares earlier in the batch and holds
        // its own Arc): remove any strong session carry so the column frees
        // at the lazy fold's materialization instead of living to the end of
        // the proof.
        let _ = session.take::<SharedLookupIndices>();
        Ok(Box::new(OptimizedInstructionRaVirtualizationKernel::new(
            relation.dimensions().log_t(),
            relation.dimensions().num_virtual_ra_polys(),
            relation.dimensions().num_committed_per_virtual(),
            relation.instruction_address(),
            relation.instruction_read_raf_cycle(),
            relation.committed_chunk_bits(),
            indices,
            inputs.challenges.gamma,
        )?))
    }
}

/// Lazy-RA index source: chunk `i` of the per-cycle lookup index (always
/// hot), off the shared lookup-index column.
struct LookupIndexChunks {
    indices: Arc<Vec<u128>>,
    num_committed: usize,
    committed_chunk_bits: usize,
}

impl ChunkIndexSource for LookupIndexChunks {
    fn num_polys(&self) -> usize {
        self.num_committed
    }

    fn cycles(&self) -> usize {
        self.indices.len()
    }

    #[inline]
    fn index(&self, i: usize, j: usize) -> Option<usize> {
        let shift = (self.num_committed - 1 - i) * self.committed_chunk_bits;
        let mask = (1u128 << self.committed_chunk_bits) - 1;
        Some(((self.indices[j] >> shift) & mask) as usize)
    }
}

pub struct OptimizedInstructionRaVirtualizationKernel<F: Field> {
    progress: RoundProgress,
    num_committed_per_virtual: usize,
    /// `γ^{-v}` per virtual batch — unscales the batch-first final claims
    /// back to the committed polynomials' values (`γ^v · γ^{-v} = 1`
    /// exactly, so unscaling is byte-exact).
    gamma_powers_inv: Vec<F>,
    /// Address-folded committed RA selectors, one per committed chunk:
    /// `folded[i][j] = eq(r_chunk_i, chunk_i(k_j))` — with each virtual
    /// batch's first table pre-scaled by `γ^v` so the round loop needs no
    /// batching multiplies — served lazily off the shared rows for the
    /// first four binds instead of `N × T` dense.
    folded_ra: LazyFoldedRa<F, LookupIndexChunks>,
    gruen: GruenSplitEqPolynomial<F>,
}

#[cfg(feature = "allocative")]
crate::optimized::impl_field_allocative!(OptimizedInstructionRaVirtualizationKernel, |kernel| {
    use crate::backend::{arc_vec_heap_bytes, vec_heap_bytes};
    vec_heap_bytes(&kernel.gamma_powers_inv)
        + kernel
            .folded_ra
            .heap_bytes(|source| arc_vec_heap_bytes(&source.indices))
        + kernel.gruen.heap_bytes()
});

impl<F: Field> OptimizedInstructionRaVirtualizationKernel<F> {
    #[expect(clippy::too_many_arguments, reason = "mirrors the relation accessors")]
    pub(crate) fn new(
        log_t: usize,
        num_virtual: usize,
        num_committed_per_virtual: usize,
        instruction_address: &[F],
        instruction_read_raf_cycle: &[F],
        committed_chunk_bits: usize,
        indices: Arc<Vec<u128>>,
        gamma: F,
    ) -> Result<Self, KernelError<F>> {
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
        if indices.len() != 1 << log_t {
            return Err(KernelError::TableSizeMismatch {
                table: "stage-6b instruction lookup indices".to_owned(),
                expected: 1 << log_t,
                got: indices.len(),
            });
        }
        if instruction_read_raf_cycle.len() != log_t {
            return Err(KernelError::TableSizeMismatch {
                table: "instruction read-RAF cycle point".to_owned(),
                expected: log_t,
                got: instruction_read_raf_cycle.len(),
            });
        }

        let (gamma_powers, gamma_powers_inv) = gamma_power_pairs(
            gamma,
            num_virtual,
            "instruction RA batching gamma must be invertible",
        )?;

        // One eq table per committed chunk point (each `2^w` entries); the
        // point-mass fold stays lazy — one table lookup per gathered cycle —
        // instead of materializing `N × T` dense selectors up front. Each
        // virtual batch's `γ^v` weight rides in the batch's first table.
        let chunk_tables: Vec<Vec<F>> = map_indices(chunks.len(), |i| {
            let mut table = eq_table(&chunks[i]);
            if i % num_committed_per_virtual == 0 {
                let weight = gamma_powers[i / num_committed_per_virtual];
                if weight != F::one() {
                    for value in &mut table {
                        *value *= weight;
                    }
                }
            }
            table
        });
        let folded_ra = LazyFoldedRa::new(
            chunk_tables,
            LookupIndexChunks {
                indices,
                num_committed,
                committed_chunk_bits,
            },
        );

        Ok(Self {
            progress: RoundProgress::new(log_t),
            num_committed_per_virtual,
            gamma_powers_inv,
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
    /// [`GruenSplitEqPolynomial::gruen_poly_from_evals`] recomposes the
    /// unique degree-`(N+1)` coefficient vector.
    fn message(
        &self,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        let n = self.num_committed_per_virtual;
        if n < 2 {
            return self.message_single_factor(round, previous_claim);
        }
        let num_committed = self.folded_ra.num_polys();
        let folded_ra = &self.folded_ra;

        struct Scratch<F: Field> {
            /// Cross-row lanes for `q(1), …, q(N−1), q(∞)`.
            lanes: Vec<F::Accumulator>,
            /// Per-row product lanes (reduced and folded by `e_in` each row).
            row_lanes: Vec<F::Accumulator>,
            pairs: Vec<(F, F)>,
            evals: Vec<F>,
            steps: Vec<F>,
        }

        let block_lanes = self.gruen.par_fold_out_in(
            || Scratch {
                lanes: vec![F::Accumulator::default(); n],
                row_lanes: vec![F::Accumulator::default(); n],
                pairs: vec![(F::zero(), F::zero()); num_committed],
                evals: vec![F::zero(); n],
                steps: vec![F::zero(); n],
            },
            |scratch, row, _x_in, e_in| {
                folded_ra.lo_hi_all(row, &mut scratch.pairs);
                for lane in &mut scratch.row_lanes {
                    *lane = F::Accumulator::default();
                }
                for pairs in scratch.pairs.chunks_exact(n) {
                    for ((pair, eval), step) in pairs
                        .iter()
                        .zip(scratch.evals.iter_mut())
                        .zip(scratch.steps.iter_mut())
                    {
                        *eval = pair.1;
                        *step = pair.1 - pair.0;
                    }
                    accumulate_product_grid(
                        &mut scratch.evals,
                        &scratch.steps,
                        &mut scratch.row_lanes,
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
        );

        let q_evals: Vec<F> = block_lanes.into_iter().map(|lane| lane.reduce()).collect();
        Ok(self.gruen.gruen_poly_from_evals(&q_evals, previous_claim))
    }

    /// Degenerate `N = 1` geometry (virtual = committed): the grid recovery
    /// assumes `q(1)` is sampled, so fall back to explicit `t = 0..=2`
    /// sampling of the quadratic summand.
    fn message_single_factor(
        &self,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        let num_committed = self.folded_ra.num_polys();
        let folded_ra = &self.folded_ra;
        let mut q_evals = self.gruen.par_fold_out_in(
            || ([F::zero(); 3], vec![(F::zero(), F::zero()); num_committed]),
            |(acc, pairs), row, _x_in, e_in| {
                folded_ra.lo_hi_all(row, pairs);
                let mut at_0 = F::zero();
                let mut at_1 = F::zero();
                let mut at_2 = F::zero();
                for (lo, hi) in pairs.iter() {
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

        self.gruen
            .checked_round_poly(&mut q_evals, previous_claim, round)
    }

    fn bind(&mut self, challenge: F) {
        self.gruen.bind(challenge);
        self.folded_ra.bind(challenge);
        self.progress.advance();
        // Mid-batch retention purges (allocator-only), placed on this
        // always-present head-aligned member on behalf of the whole
        // stage-6b batch: by round 3 the co-members' first binds have
        // stranded their full-length staging (the dense Hamming table's
        // and increment tables' first halvings); by round 6 the lazy
        // folds' `T/16` dense materializations have been halved twice,
        // stranding multi-GiB generations. Without these the dead pages
        // sit in the allocator's freed-large-block cache until the stage
        // boundary — the process-global resident peak.
        if self.progress.bound() == 3 || self.progress.bound() == 6 {
            crate::mem::purge_staging(self.progress.total());
        }
    }
}

impl<F: Field> ProveRounds<F> for OptimizedInstructionRaVirtualizationKernel<F> {
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
        self.message(round, previous_claim)
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind);
        Ok(())
    }
}

impl<F: Field> SumcheckKernel<F> for OptimizedInstructionRaVirtualizationKernel<F> {
    type Relation = InstructionRaVirtualization<F>;

    fn output_claims(
        &mut self,
        _inputs: &SumcheckInputClaims<F, Self::Relation>,
    ) -> Result<InstructionRaVirtualizationOutputClaims<F>, SumcheckKernelError<F>> {
        self.progress.require_complete()?;
        // Unscale the batch-first tables' γ^v pre-scaling back to the
        // committed polynomials' claims.
        let mut committed_instruction_ra = self.folded_ra.final_values();
        for (index, value) in committed_instruction_ra.iter_mut().enumerate() {
            if index % self.num_committed_per_virtual == 0 {
                *value *= self.gamma_powers_inv[index / self.num_committed_per_virtual];
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
        pin_derived_term(
            relation,
            JoltDerivedId::from(InstructionRaVirtualizationPublic::EqCycle),
            input_points,
            output_points,
            challenges,
            self.gruen.current_scalar(),
        )
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use std::sync::Arc;

    use std::collections::BTreeMap;
    use std::num::NonZeroUsize;

    use jolt_claims::protocols::jolt::geometry::dimensions::committed_address_chunks;
    use jolt_claims::protocols::jolt::geometry::instruction::{
        committed_instruction_ra, InstructionRaVirtualizationDimensions,
    };
    use jolt_claims::protocols::jolt::relations::instruction::{
        InstructionRaVirtualizationChallenges, InstructionRaVirtualizationInputClaims,
    };
    use jolt_claims::protocols::jolt::{InstructionRaVirtualizationPublic, JoltDerivedId};
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_poly::{BindingOrder, Polynomial};
    use jolt_sumcheck::ProveRounds;
    use jolt_verifier::stages::relations::ConcreteSumcheck;
    use jolt_verifier::stages::stage6b::instruction_ra_virtualization::InstructionRaVirtualization;
    use jolt_witness::witnesses::{InstructionRafFlag, LookupIndex, TableIndex};
    use jolt_witness::{FixedBackend, PolynomialEncoding, Shape};

    use crate::reference::instruction_read_raf::InstructionReadRafWitness;
    use crate::reference::views::{address_fold, eq_table};
    use crate::{NaiveSumcheckProver, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel};

    use super::super::instruction_read_raf::SharedLookupIndices;
    use super::super::testing::{with_ram_fixture, FixtureShape};
    use super::{OptimizedInstructionRaVirtualization, OptimizedInstructionRaVirtualizationKernel};

    /// Packs reference-typed fixture rows into the optimized kernels' shared
    /// column form (this kernel reads only the lookup index).
    fn pack(rows: &[InstructionReadRafWitness]) -> Vec<u128> {
        rows.iter().map(|row| row.lookup_index.0).collect()
    }

    fn fr(value: u64) -> Fr {
        Fr::from_u64(value)
    }

    fn challenge(round: usize) -> Fr {
        fr(0xD1B5_4A32_D192_ED03 ^ (round as u64).wrapping_mul(0x94D0_49BB_1331_11EB) ^ 3)
    }

    fn splitmix(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn fixture_rows(log_t: usize, seed: u64) -> Vec<InstructionReadRafWitness> {
        let mut state = seed;
        (0..1usize << log_t)
            .map(|j| {
                let lookup_index = match j {
                    0 => 0u128,
                    1 => u128::MAX,
                    _ => ((splitmix(&mut state) as u128) << 64) | splitmix(&mut state) as u128,
                };
                InstructionReadRafWitness {
                    lookup_index: LookupIndex(lookup_index),
                    table_index: TableIndex(None),
                    raf_flag: InstructionRafFlag(false),
                }
            })
            .collect()
    }

    /// Builds the committed one-hot `(K × T)` grid for chunk `i` exactly as
    /// the trace backend serves it: address-major, hot at that chunk of the
    /// cycle's lookup index, every cycle hot.
    fn one_hot_grid(
        rows: &[InstructionReadRafWitness],
        chunk_index: usize,
        num_committed: usize,
        chunk_bits: usize,
    ) -> Vec<Fr> {
        let k = 1usize << chunk_bits;
        let t = rows.len();
        let shift = (num_committed - 1 - chunk_index) * chunk_bits;
        let mask = (1u128 << chunk_bits) - 1;
        let mut grid = vec![fr(0); k * t];
        for (j, row) in rows.iter().enumerate() {
            let hot = ((row.lookup_index.0 >> shift) & mask) as usize;
            grid[(hot * t) | j] = fr(1);
        }
        grid
    }

    /// Reference (naive prover over address-folded oracle grids, exactly as
    /// the reference `prepare` assembles it) vs the optimized kernel, same
    /// challenges: byte-equal round polynomials and output claims, and the
    /// optimized eq-scalar passes the derived-table cross-check.
    ///
    /// `with_session` builds the optimized kernel through the `prepare` slot
    /// with a pre-parked lookup-index column instead of direct construction,
    /// exercising the session carry and its final-consumer take.
    fn assert_parity(
        log_t: usize,
        num_virtual: usize,
        per_virtual: usize,
        chunk_bits: usize,
        seed: u64,
        with_session: bool,
    ) {
        let num_committed = num_virtual * per_virtual;
        let dimensions = InstructionRaVirtualizationDimensions::new(
            log_t,
            NonZeroUsize::new(num_virtual).unwrap(),
            NonZeroUsize::new(per_virtual).unwrap(),
        )
        .unwrap();
        let rows = fixture_rows(log_t, seed);
        let instruction_address: Vec<Fr> = (0..num_committed * chunk_bits)
            .map(|i| fr(300 + 13 * i as u64))
            .collect();
        let r_cycle: Vec<Fr> = (0..log_t).map(|i| fr(7000 + 29 * i as u64)).collect();
        let gamma = fr(0xFEED_5EED);
        let relation = InstructionRaVirtualization::<Fr>::new(
            dimensions,
            instruction_address.clone(),
            r_cycle.clone(),
            chunk_bits,
        );

        // The reference tier, assembled exactly as its `prepare` does: one-hot
        // grids behind a fixed oracle, address-folded per committed chunk.
        let mut backend = FixedBackend::new();
        for index in 0..num_committed {
            let grid = one_hot_grid(&rows, index, num_committed, chunk_bits);
            backend
                .insert(
                    committed_instruction_ra(index).polynomial_id(),
                    Shape::new(chunk_bits + log_t, PolynomialEncoding::Dense),
                    grid,
                )
                .unwrap();
        }
        let chunks = committed_address_chunks(&instruction_address, chunk_bits);
        let mut opening_tables = BTreeMap::new();
        for (index, chunk) in chunks.iter().enumerate() {
            let folded =
                address_fold::<Fr>(&backend, committed_instruction_ra(index), log_t, chunk)
                    .unwrap();
            let _ = opening_tables.insert(committed_instruction_ra(index), Polynomial::new(folded));
        }
        let derived_tables = BTreeMap::from([(
            JoltDerivedId::from(InstructionRaVirtualizationPublic::EqCycle),
            Polynomial::new(eq_table(&r_cycle)),
        )]);

        let input_claims = InstructionRaVirtualizationInputClaims {
            instruction_ra: vec![fr(0); num_virtual],
        };
        let input_points = InstructionRaVirtualizationInputClaims {
            instruction_ra: vec![Vec::new(); num_virtual],
        };
        let challenges = InstructionRaVirtualizationChallenges { gamma };
        let inputs = ProverInputs {
            relation: &relation,
            claims: &input_claims,
            points: &input_points,
            challenges: &challenges,
        };
        let mut reference = NaiveSumcheckProver::new(
            &inputs,
            opening_tables,
            derived_tables,
            BindingOrder::LowToHigh,
        )
        .unwrap();

        let mut optimized: Box<dyn SumcheckKernel<Fr, Relation = InstructionRaVirtualization<Fr>>> =
            if with_session {
                // The witness plane comes from an unrelated trace fixture: a
                // missed take of the parked rows would stream that trace's
                // lookup indices instead and fail the parity loop loudly.
                let shape = FixtureShape { log_t, ram_k: 16 };
                with_ram_fixture(shape, Vec::new(), |witness| {
                    let mut session = ProofSession::default();
                    session.park(SharedLookupIndices(Arc::new(pack(&rows))));
                    let kernel = OptimizedInstructionRaVirtualization
                        .prepare(
                            &mut session,
                            witness,
                            ProverInputs {
                                relation: &relation,
                                claims: &input_claims,
                                points: &input_points,
                                challenges: &challenges,
                            },
                        )
                        .unwrap();
                    // This kernel is the column's LAST consumer: the strong
                    // carry must be consumed (a weak handle may remain, alive
                    // only through the kernel's own Arc).
                    assert!(
                        session.state::<SharedLookupIndices>().is_none(),
                        "prepare must consume the strong lookup-index carry"
                    );
                    kernel
                })
            } else {
                Box::new(
                    OptimizedInstructionRaVirtualizationKernel::new(
                        log_t,
                        num_virtual,
                        per_virtual,
                        &instruction_address,
                        &r_cycle,
                        chunk_bits,
                        Arc::new(pack(&rows)),
                        gamma,
                    )
                    .unwrap(),
                )
            };

        // True input claim: the full hypercube sum of the output summand.
        let eq_cycle = eq_table(&r_cycle);
        let mut claim = fr(0);
        for j in 0..rows.len() {
            let mut sum = fr(0);
            let mut gamma_power = fr(1);
            for v in 0..num_virtual {
                let mut product = fr(1);
                for i in 0..per_virtual {
                    let index = v * per_virtual + i;
                    let shift = (num_committed - 1 - index) * chunk_bits;
                    let hot =
                        ((rows[j].lookup_index.0 >> shift) & ((1u128 << chunk_bits) - 1)) as usize;
                    product *= eq_table(&chunks[index])[hot];
                }
                sum += gamma_power * product;
                gamma_power *= gamma;
            }
            claim += eq_cycle[j] * sum;
        }

        let rounds = reference.num_rounds();
        assert_eq!(rounds, optimized.num_rounds());
        assert_eq!(rounds, log_t);
        for round in 0..rounds {
            let bind = round.checked_sub(1).map(challenge);
            let reference_poly = reference.prove_round(bind, round, claim).unwrap();
            let optimized_poly = optimized.prove_round(bind, round, claim).unwrap();
            assert_eq!(
                reference_poly.coefficients(),
                optimized_poly.coefficients(),
                "round {round} polynomial mismatch (log_t={log_t}, V={num_virtual}, N={per_virtual})"
            );
            claim = reference_poly.evaluate(challenge(round));
        }
        reference.finish_rounds(challenge(rounds - 1)).unwrap();
        optimized.finish_rounds(challenge(rounds - 1)).unwrap();

        let reference_outputs = reference.output_claims(&input_claims).unwrap();
        let optimized_outputs = optimized.output_claims(&input_claims).unwrap();
        assert_eq!(
            reference_outputs.committed_instruction_ra,
            optimized_outputs.committed_instruction_ra
        );

        // The optimized eq scalar passes the same derived-table cross-check
        // the naive tier's materialized table does.
        let sumcheck_point: Vec<Fr> = (0..rounds).map(challenge).collect();
        let output_points = relation
            .derive_opening_points(&sumcheck_point, &input_points)
            .unwrap();
        optimized
            .validate_derived_tables(&relation, &input_points, &output_points, &challenges)
            .unwrap();
        reference
            .validate_derived_tables(&relation, &input_points, &output_points, &challenges)
            .unwrap();
    }

    /// Production shape: 8 virtuals × 4 committed each, 4-bit chunks (the
    /// 128-bit instruction address).
    #[test]
    fn parity_production_geometry() {
        assert_parity(4, 8, 4, 4, 42, false);
    }

    /// Odd geometry: 3 virtuals × 2 committed, 2-bit chunks, odd log_t.
    #[test]
    fn parity_small_odd_geometry() {
        assert_parity(3, 3, 2, 2, 1337, false);
    }

    #[test]
    fn parity_past_lazy_materialization() {
        // log_t = 6: three lazy binds, the dense materialization at the
        // fourth (`T/16` = 4 entries), and two plain multilinear binds
        // after it.
        assert_parity(6, 2, 2, 4, 7, false);
    }

    /// Through the `prepare` slot with a pre-parked lookup-index column: the
    /// session carry serves this kernel the same indices, and the strong
    /// carry is consumed (this kernel is the column's last consumer).
    #[test]
    fn parity_with_carried_session_indices() {
        assert_parity(4, 8, 4, 4, 42, true);
    }
}
