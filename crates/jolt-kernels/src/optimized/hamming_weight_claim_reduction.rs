//! Optimized Hamming-weight claim-reduction (stage 7) kernel, byte-parity
//! twin of [`crate::reference::hamming_weight_claim_reduction`].
//!
//! Ported legacy techniques
//! (`jolt-prover-legacy/src/zkvm/claim_reductions/hamming_weight.rs` +
//! `poly/shared_ra_polys.rs::compute_all_G`):
//!
//! - **Shared-eq pushforwards over every RA family in one trace pass**: each
//!   `G_i(k) = Σ_j eq(r_cycle, j) · ra_i(k, j)` collapses, for a one-hot
//!   `ra_i`, to `G_i[chunk_i(j)] += eq_cycle[j]` — one `O(T)` bundle walk
//!   accumulates ALL `N = instruction_d + bytecode_d + ram_d` pushforwards
//!   against ONE shared `eq(r_cycle)` table (every stage-6b claim family
//!   lives at the same cycle point), replacing the reference tier's `N`
//!   independent `O(K_chunk·T)` folds over materialized one-hot grids. The
//!   walk reuses booleanity's split-eq deferred buckets: inner weights enter
//!   by addition, and the outer eq multiplication occurs once per bucket.
//! - **One-hot weight fusion**: the three per-polynomial claim weights
//!   `γ^{3i} + γ^{3i+1}·eq_bool(k) + γ^{3i+2}·eq_virt_i(k)` are one combined
//!   multilinear `W_i(k)` (the Hamming-weight leg's constant-1 rides the
//!   constant term), so the round summand is `Σ_i G_i·W_i` — `2N` bound
//!   tables instead of `2N + 1` and one fused multiply per pair per point.
//! - **Eval-at-1 recovery** and **rayon walks** (module docs on
//!   [`crate::optimized`]).

use jolt_claims::protocols::jolt::geometry::ra::JoltRaPolynomialLayout;
use jolt_claims::protocols::jolt::{JoltOpeningId, JoltRelationId};
use jolt_claims::OutputClaims;
use jolt_field::Field;
use jolt_poly::{Polynomial, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::{
    ConcreteSumcheck, SumcheckInputClaims, SumcheckOutputClaims,
};
use jolt_verifier::stages::stage7::hamming_weight_claim_reduction::HammingWeightClaimReduction;
use jolt_witness::JoltWitnessPlane;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::instruction_read_raf::{shared_instruction_rows, InstructionCycleRow};
use super::one_hot_pushforward::{parallel_outer_bits, split_eq_pushforwards, ColumnSelector};
#[cfg(feature = "parallel")]
use super::support::merge_evals;
use super::support::{bind_all, eq_table, pair, round_poly_from_skipped_evals};
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

/// All `N` pushforwards from one pass over the shared stage-5 rows, in
/// canonical (instruction, bytecode, RAM) order. The cycle-eq weight is
/// computed from split tables `eq(r, j) = eq(r[..hi], j_hi) ·
/// eq(r[hi..], j_lo)`. Outer blocks stay at four per Rayon worker so the
/// deferred inner-bucket accumulation removes per-row field multiplication
/// without starving parallelism.
fn pushforwards<F: Field>(
    rows: &[InstructionCycleRow],
    r_cycle: &[F],
    layout: JoltRaPolynomialLayout,
    chunk_bits: usize,
) -> Result<Vec<Vec<F>>, KernelError<F>> {
    let selectors = ColumnSelector::for_layout(layout, chunk_bits)?;
    Ok(split_eq_pushforwards(
        rows,
        &selectors,
        1usize << chunk_bits,
        r_cycle,
        parallel_outer_bits(r_cycle.len()),
    ))
}

#[cfg(feature = "bench-utils")]
pub mod bench {
    use std::sync::Arc;

    use jolt_field::{Fr, FromPrimitiveInt};

    use jolt_claims::protocols::jolt::geometry::ra::JoltRaPolynomialLayout;

    use super::pushforwards;
    use crate::optimized::instruction_read_raf::InstructionCycleRow;
    use crate::reference::views::eq_table;

    const LOG_T: usize = 22;
    const CHUNK_BITS: usize = 8;
    const FAMILY_COUNTS: (usize, usize, usize) = (16, 2, 3);

    pub struct HammingWeightPushforwardFixture {
        rows: Arc<[InstructionCycleRow]>,
        r_cycle: Vec<Fr>,
        layout: JoltRaPolynomialLayout,
        expected: Vec<Vec<Fr>>,
    }

    impl HammingWeightPushforwardFixture {
        pub fn production_geometry() -> Self {
            let rows: Arc<[InstructionCycleRow]> = (0..1usize << LOG_T)
                .map(|cycle| {
                    let x = cycle as u128;
                    let lookup_index = x
                        .wrapping_mul(0x9e37_79b9_7f4a_7c15_6a09_e667_f3bc_c909)
                        .rotate_left((cycle & 127) as u32);
                    let mapped_pc = (cycle % 17 != 0)
                        .then_some((cycle.wrapping_mul(13) ^ (cycle >> 3)) & 0xffff);
                    let ram_address = (cycle % 5 != 0)
                        .then_some(((cycle.wrapping_mul(29) ^ (cycle >> 5)) & 0x00ff_ffff) as u64);
                    InstructionCycleRow::new(lookup_index, None, false, mapped_pc, ram_address)
                })
                .collect::<Vec<_>>()
                .into();
            let r_cycle = (0..LOG_T)
                .map(|i| Fr::from_u64(0x100 + 17 * i as u64))
                .collect::<Vec<_>>();
            let Ok(layout) =
                JoltRaPolynomialLayout::new(FAMILY_COUNTS.0, FAMILY_COUNTS.1, FAMILY_COUNTS.2)
            else {
                std::process::abort();
            };
            let expected = reference_pushforwards(&rows, &r_cycle, layout);
            let fixture = Self {
                rows,
                r_cycle,
                layout,
                expected,
            };
            assert_eq!(fixture.compute(), fixture.expected);
            fixture
        }

        pub fn compute(&self) -> Vec<Vec<Fr>> {
            let Ok(output) = pushforwards(&self.rows, &self.r_cycle, self.layout, CHUNK_BITS)
            else {
                std::process::abort();
            };
            output
        }
    }

    fn reference_pushforwards(
        rows: &[InstructionCycleRow],
        r_cycle: &[Fr],
        layout: JoltRaPolynomialLayout,
    ) -> Vec<Vec<Fr>> {
        let k_chunk = 1 << CHUNK_BITS;
        let total = layout.total();
        let mut out = (0..total)
            .map(|_| vec![Fr::from_u64(0); k_chunk])
            .collect::<Vec<_>>();
        for (row, weight) in rows.iter().zip(eq_table(r_cycle)) {
            let mut slot = 0;
            for index in 0..layout.instruction() {
                let shift = (layout.instruction() - 1 - index) * CHUNK_BITS;
                out[slot][((row.lookup_index >> shift) & 0xff) as usize] += weight;
                slot += 1;
            }
            for index in 0..layout.bytecode() {
                if let Some(pc) = row.mapped_pc() {
                    let shift = (layout.bytecode() - 1 - index) * CHUNK_BITS;
                    out[slot][(pc >> shift) & 0xff] += weight;
                }
                slot += 1;
            }
            for index in 0..layout.ram() {
                if let Some(address) = row.remapped_ram_address() {
                    let shift = (layout.ram() - 1 - index) * CHUNK_BITS;
                    out[slot][(address as usize >> shift) & 0xff] += weight;
                }
                slot += 1;
            }
        }
        out
    }
}

/// Stage-7 Hamming-weight claim reduction: `PrepareKernel` front of the
/// optimized kernel.
pub struct OptimizedHammingWeightClaimReduction;

/// The `N` pushforward/weight table pairs of the fused summand
/// `Σ_i G_i·W_i`, as built by [`build_hamming_weight_tables`] — shared
/// between the optimized kernel and its Metal twin (which concatenates the
/// pairs into two flat device tables).
pub(crate) struct HammingWeightTables<F> {
    pub(crate) rounds: usize,
    /// Pushforwards `G_i`, canonical layout order.
    pub(crate) g_tables: Vec<Vec<F>>,
    /// Combined claim weights `W_i`, index-aligned with `g_tables`.
    pub(crate) weight_tables: Vec<Vec<F>>,
    pub(crate) output_openings: Vec<JoltOpeningId>,
}

pub(crate) fn build_hamming_weight_tables<F: Field>(
    session: &mut ProofSession,
    witness: &dyn JoltWitnessPlane<F>,
    inputs: &ProverInputs<'_, F, HammingWeightClaimReduction<F>>,
) -> Result<HammingWeightTables<F>, KernelError<F>> {
    let relation = inputs.relation;
    let dimensions = relation.dimensions();
    let layout = dimensions.layout;
    let r_cycle = relation.r_cycle();
    let r_address = relation.r_address();
    let virtualization_points = relation.virtualization_points();
    if r_address.len() != dimensions.log_k_chunk || virtualization_points.len() != layout.total() {
        return Err(KernelError::InvariantViolation {
            reason: "hamming reduction reference point shapes disagree with the layout",
        });
    }
    let k_chunk = 1usize << dimensions.log_k_chunk;
    let cycles = 1usize << r_cycle.len();

    // The three hot-address lanes (lookup index, mapped PC, RAM address) are
    // exactly the shared stage-5 rows the record walk co-produced — reclaim
    // them instead of re-walking the trace.
    let rows = shared_instruction_rows(session, witness, cycles)?;
    let g_tables = pushforwards(&rows, r_cycle, layout, dimensions.log_k_chunk)?;

    // W_i(k) = γ^{3i} + γ^{3i+1}·eq_bool(k) + γ^{3i+2}·eq_virt_i(k).
    let gamma = inputs.challenges.gamma;
    let mut gamma_powers = vec![F::one(); 3 * layout.total()];
    for i in 1..gamma_powers.len() {
        gamma_powers[i] = gamma_powers[i - 1] * gamma;
    }
    let eq_bool = eq_table(r_address);
    let weight_tables: Vec<Vec<F>> = virtualization_points
        .iter()
        .enumerate()
        .map(|(i, point)| {
            if point.len() != dimensions.log_k_chunk {
                return Err(KernelError::InvariantViolation {
                    reason: "hamming virtualization point has the wrong variable count",
                });
            }
            let eq_virt = eq_table(point);
            Ok((0..k_chunk)
                .map(|k| {
                    gamma_powers[3 * i]
                        + gamma_powers[3 * i + 1] * eq_bool[k]
                        + gamma_powers[3 * i + 2] * eq_virt[k]
                })
                .collect())
        })
        .collect::<Result<_, _>>()?;

    let output_openings: Vec<JoltOpeningId> = layout
        .openings(JoltRelationId::HammingWeightClaimReduction)
        .collect();

    Ok(HammingWeightTables {
        rounds: relation.rounds(),
        g_tables,
        weight_tables,
        output_openings,
    })
}

impl<F: Field> PrepareKernel<F, HammingWeightClaimReduction<F>>
    for OptimizedHammingWeightClaimReduction
{
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, HammingWeightClaimReduction<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = HammingWeightClaimReduction<F>>>, KernelError<F>>
    {
        let tables = build_hamming_weight_tables(session, witness, &inputs)?;
        Ok(Box::new(HammingWeightKernel {
            rounds: tables.rounds,
            g_tables: tables.g_tables.into_iter().map(Polynomial::new).collect(),
            weight_tables: tables
                .weight_tables
                .into_iter()
                .map(Polynomial::new)
                .collect(),
            output_openings: tables.output_openings,
            rounds_bound: 0,
        }))
    }
}

struct HammingWeightKernel<F: Field> {
    rounds: usize,
    /// Pushforwards `G_i`, canonical layout order.
    g_tables: Vec<Polynomial<F>>,
    /// Combined claim weights `W_i`, index-aligned with `g_tables`.
    weight_tables: Vec<Polynomial<F>>,
    output_openings: Vec<JoltOpeningId>,
    rounds_bound: usize,
}

impl<F: Field> HammingWeightKernel<F> {
    fn bind(&mut self, challenge: F) {
        bind_all(
            self.g_tables
                .iter_mut()
                .chain(self.weight_tables.iter_mut()),
            challenge,
        );
        self.rounds_bound += 1;
    }

    /// The summand's evaluations at `t ∈ {0, 2}` summed over group `y`.
    #[inline]
    fn group_evals(&self, y: usize) -> [F; 2] {
        let mut out = [F::zero(); 2];
        for (g, w) in self.g_tables.iter().zip(&self.weight_tables) {
            let (g_lo, g_hi) = pair(g, y);
            let (w_lo, w_hi) = pair(w, y);
            out[0] += g_lo * w_lo;
            out[1] += (g_hi + g_hi - g_lo) * (w_hi + w_hi - w_lo);
        }
        out
    }
}

impl<F: Field> ProveRounds<F> for HammingWeightKernel<F> {
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
        let half = self.weight_tables[0].len() / 2;

        #[cfg(feature = "parallel")]
        let evals = (0..half)
            .into_par_iter()
            .fold(
                || vec![F::zero(); 2],
                |mut acc, y| {
                    let group = self.group_evals(y);
                    acc[0] += group[0];
                    acc[1] += group[1];
                    acc
                },
            )
            .reduce(|| vec![F::zero(); 2], merge_evals);
        #[cfg(not(feature = "parallel"))]
        let evals = (0..half).fold(vec![F::zero(); 2], |mut acc, y| {
            let group = self.group_evals(y);
            acc[0] += group[0];
            acc[1] += group[1];
            acc
        });

        Ok(round_poly_from_skipped_evals(&evals, previous_claim))
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind);
        Ok(())
    }
}

impl<F: Field> SumcheckKernel<F> for HammingWeightKernel<F> {
    type Relation = HammingWeightClaimReduction<F>;

    fn output_claims(
        &mut self,
        _inputs: &SumcheckInputClaims<F, Self::Relation>,
    ) -> Result<SumcheckOutputClaims<F, Self::Relation>, SumcheckKernelError<F>> {
        if self.rounds_bound != self.rounds {
            return Err(SumcheckKernelError::NotFullyBound {
                remaining: self.rounds - self.rounds_bound,
            });
        }
        let g_tables = &self.g_tables;
        let output_openings = &self.output_openings;
        SumcheckOutputClaims::<F, Self::Relation>::from_opening_values(|id: &JoltOpeningId| {
            output_openings
                .iter()
                .position(|opening| opening == id)
                .map(|index| g_tables[index].evals()[0])
        })
        .map_err(SumcheckKernelError::from)
    }
}

/// Byte parity against the reference kernel over the sample backend. All
/// three families are live at fixture scale (the instruction family alone
/// contributes 128/log_k_chunk polynomials), so the multi-family pushforward
/// walk, cold RAM/bytecode cycles, and per-polynomial weight fusion are all
/// exercised.
#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use jolt_claims::protocols::jolt::geometry::claim_reductions::hamming_weight::HammingWeightClaimReductionDimensions;
    use jolt_claims::protocols::jolt::geometry::ra::JoltRaPolynomialLayout;
    use jolt_claims::protocols::jolt::{JoltCommittedPolynomial, JoltPolynomialId};
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_verifier::stages::stage7::hamming_weight_claim_reduction::{
        HammingWeightClaimReductionChallenges, HammingWeightClaimReductionInputClaims,
    };
    use jolt_witness::testing::with_sample_backend;
    use jolt_witness::JoltWitnessOracle;

    use super::*;
    use crate::optimized::harness::{
        probe_input_claim, probe_one_hot_family, run_lockstep, synthetic_point,
    };
    use crate::{ProofSession, ReferenceBackend};

    #[test]
    fn hamming_weight_reduction_matches_reference() {
        with_sample_backend(|backend| {
            let log_t = JoltWitnessOracle::<Fr>::shape(
                backend,
                JoltPolynomialId::Committed(JoltCommittedPolynomial::RdInc),
            )
            .unwrap()
            .rows()
            .ilog2() as usize;
            let (instruction_d, log_k_chunk) =
                probe_one_hot_family(backend, JoltCommittedPolynomial::InstructionRa, log_t);
            let (bytecode_d, _) =
                probe_one_hot_family(backend, JoltCommittedPolynomial::BytecodeRa, log_t);
            let (ram_d, _) = probe_one_hot_family(backend, JoltCommittedPolynomial::RamRa, log_t);
            assert!(
                instruction_d > 0 && bytecode_d > 0 && ram_d > 0,
                "fixture must serve all three RA families \
                 (instruction {instruction_d}, bytecode {bytecode_d}, ram {ram_d})"
            );
            let layout = JoltRaPolynomialLayout::new(instruction_d, bytecode_d, ram_d).unwrap();
            let dimensions = HammingWeightClaimReductionDimensions::new(layout, log_k_chunk);

            let relation = HammingWeightClaimReduction::new(
                dimensions,
                synthetic_point(log_t, 3),
                synthetic_point(log_k_chunk, 5),
                (0..layout.total())
                    .map(|index| synthetic_point(log_k_chunk, 7 + index as u64))
                    .collect(),
            );
            let challenges = HammingWeightClaimReductionChallenges {
                gamma: Fr::from_u64(23),
            };
            let claims = HammingWeightClaimReductionInputClaims::<Fr>::default();
            let input_points = HammingWeightClaimReductionInputClaims::<Vec<Fr>>::default();

            let mut session = ProofSession::default();
            let mut reference =
                <ReferenceBackend as PrepareKernel<Fr, HammingWeightClaimReduction<Fr>>>::prepare(
                    &ReferenceBackend,
                    &mut session,
                    backend,
                    ProverInputs {
                        relation: &relation,
                        claims: &claims,
                        points: &input_points,
                        challenges: &challenges,
                    },
                )
                .unwrap();
            let mut optimized = OptimizedHammingWeightClaimReduction
                .prepare(
                    &mut session,
                    backend,
                    ProverInputs {
                        relation: &relation,
                        claims: &claims,
                        points: &input_points,
                        challenges: &challenges,
                    },
                )
                .unwrap();

            let claim = probe_input_claim(reference.as_mut());
            let sumcheck_challenges = synthetic_point(log_k_chunk, 301);
            run_lockstep(
                reference.as_mut(),
                optimized.as_mut(),
                claim,
                &sumcheck_challenges,
            );
            assert_eq!(
                reference.output_claims(&claims).unwrap(),
                optimized.output_claims(&claims).unwrap()
            );
        });
    }
}
