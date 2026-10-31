//! Optimized Hamming-weight claim-reduction (stage 7) kernel, byte-parity
//! twin of [`crate::reference::hamming_weight_claim_reduction`].
//!
//! Carries forward the former prover's shared-RA `compute_all_G` technique:
//!
//! - **Shared-eq pushforwards over every RA family in one trace pass**: each
//!   `G_i(k) = Σ_j eq(r_cycle, j) · ra_i(k, j)` collapses, for a one-hot
//!   `ra_i`, to `G_i[chunk_i(j)] += eq_cycle[j]` — one `O(T)` bundle walk
//!   accumulates ALL `N = instruction_d + bytecode_d + ram_d` pushforwards
//!   against ONE tensor-factored `eq(r_cycle)` (every stage-6b claim family
//!   lives at the same cycle point), replacing the reference tier's `N`
//!   independent `O(K_chunk·T)` folds over materialized one-hot grids.
//! - **One-hot weight fusion**: the base protocol's three per-polynomial claim
//!   weights `γ^{3i} + γ^{3i+1}·eq_bool(k) + γ^{3i+2}·eq_virt_i(k)` are one
//!   combined multilinear `W_i(k)` (the Hamming-weight leg's constant-1 rides
//!   the constant term), so the round summand is `Σ_i G_i·W_i` — `2N` bound
//!   tables instead of `2N + 1` and one fused multiply per pair per point.
//! - **Eval-at-1 recovery** and **rayon walks** (module docs on
//!   [`crate::optimized`]).

#[cfg(feature = "akita")]
use jolt_claims::protocols::jolt::geometry::ra::JoltRaPolynomial;
use jolt_claims::protocols::jolt::{JoltOpeningId, JoltRelationId};
use jolt_claims::OutputClaims;
use jolt_field::{Accumulator, JoltField};
use jolt_poly::{EqPolynomial, Polynomial, TensorEqTable, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::{
    ConcreteSumcheck, SumcheckInputClaims, SumcheckOutputClaims,
};
use jolt_verifier::stages::stage7::hamming_weight_claim_reduction::HammingWeightClaimReduction;
use jolt_witness::witnesses::RaChunkSelector;
use jolt_witness::JoltWitnessPlane;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::constant_suffix::ConstantSuffix;
use super::instruction_read_raf::InstructionCycleRow;
use super::support::{
    bind_all, eq_table, gamma_powers, pair, par_sum_pair_groups, round_poly_from_skipped_evals,
    RoundProgress,
};
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

/// Per-family chunk selectors in canonical layout order.
struct FamilySelectors {
    instruction: Vec<RaChunkSelector>,
    bytecode: Vec<RaChunkSelector>,
    ram: Vec<RaChunkSelector>,
}

impl FamilySelectors {
    fn new<F: JoltField>(
        counts: (usize, usize, usize),
        chunk_bits: usize,
    ) -> Result<Self, KernelError<F>> {
        let family = |count: usize| -> Result<Vec<RaChunkSelector>, KernelError<F>> {
            (0..count)
                .map(|index| {
                    RaChunkSelector::new(index, count, chunk_bits).map_err(KernelError::from)
                })
                .collect()
        };
        Ok(Self {
            instruction: family(counts.0)?,
            bytecode: family(counts.1)?,
            ram: family(counts.2)?,
        })
    }
}

/// Fold equality weights into small selector histograms without a trace-sized
/// equality table. The outer tensor weight is applied once per histogram
/// bucket, after summing its inner weights with deferred reduction.
fn pushforwards<F: JoltField>(
    rows: &ConstantSuffix<InstructionCycleRow>,
    point: &[F],
    selectors: &FamilySelectors,
    k_chunk: usize,
    omit_digit_zero: bool,
) -> Vec<Vec<F>> {
    let instruction = selectors.instruction.len();
    let bytecode = selectors.bytecode.len();
    let ram = selectors.ram.len();
    let total = instruction + bytecode + ram;
    let eq = TensorEqTable::new(point);
    debug_assert_eq!(eq.len(), rows.len());
    let in_len = eq.e_in().len();
    let prefix = rows.prefix_len();
    let bits = k_chunk.trailing_zeros();
    let mask = (k_chunk - 1) as u128;
    let native_decode = bits.is_power_of_two() && bits <= 32 && instruction <= 128 / bits as usize;
    struct State<F: JoltField> {
        block: Vec<Vec<F::Accumulator>>,
        sums: Vec<Vec<F::Accumulator>>,
    }
    let zero = || State::<F> {
        block: vec![vec![F::Accumulator::default(); k_chunk]; total],
        sums: vec![vec![F::Accumulator::default(); k_chunk]; total],
    };
    let add_row = |buckets: &mut [Vec<F::Accumulator>], row: &InstructionCycleRow, weight: F| {
        let (instruction_buckets, other) = buckets.split_at_mut(instruction);
        let (bytecode_buckets, ram_buckets) = other.split_at_mut(bytecode);
        let index = row.lookup_index();
        if native_decode {
            for (buckets, mut limb) in instruction_buckets
                .rchunks_mut(64 / bits as usize)
                .zip([index as u64, (index >> 64) as u64])
            {
                if omit_digit_zero && limb == 0 {
                    continue;
                }
                for bucket in buckets.iter_mut().rev() {
                    let digit = limb as usize & (k_chunk - 1);
                    if !omit_digit_zero || digit != 0 {
                        bucket[digit].add(weight);
                    }
                    limb >>= bits;
                }
            }
        } else {
            let mut index = index;
            for bucket in instruction_buckets.iter_mut().rev() {
                let digit = (index & mask) as usize;
                if !omit_digit_zero || digit != 0 {
                    bucket[digit].add(weight);
                }
                index >>= bits;
            }
        }
        let mut pc = row.bytecode_pc();
        for bucket in bytecode_buckets.iter_mut().rev() {
            let digit = pc & (k_chunk - 1);
            if !omit_digit_zero || digit != 0 {
                bucket[digit].add(weight);
            }
            pc >>= bits;
        }
        if let Some(mut address) = row.remapped_ram_address() {
            for bucket in ram_buckets.iter_mut().rev() {
                bucket[address as usize & (k_chunk - 1)].add(weight);
                address >>= bits;
            }
        }
    };
    let scatter = |mut state: State<F>, outer: usize| {
        let base = outer * in_len;
        for (inner, &weight) in eq.e_in().iter().take(prefix - base).enumerate() {
            add_row(&mut state.block, &rows[base + inner], weight);
        }
        let outer_weight = eq.e_out()[outer];
        for (sums, block) in state.sums.iter_mut().zip(&mut state.block) {
            for (sum, bucket) in sums.iter_mut().zip(block) {
                let value = std::mem::take(bucket).reduce();
                sum.fmadd(outer_weight, value);
            }
        }
        state
    };
    let merge = |mut left: State<F>, right: State<F>| {
        for (left, right) in left.sums.iter_mut().zip(right.sums) {
            for (left, right) in left.iter_mut().zip(right) {
                left.merge(right);
            }
        }
        left
    };
    let blocks = prefix.div_ceil(in_len);
    #[cfg(feature = "parallel")]
    let mut state = (0..blocks)
        .into_par_iter()
        .fold(zero, scatter)
        .reduce(zero, merge);
    #[cfg(not(feature = "parallel"))]
    let mut state = {
        let _ = merge;
        (0..blocks).fold(zero(), scatter)
    };
    add_row(
        &mut state.sums,
        &rows[prefix],
        EqPolynomial::suffix_sum(point, prefix),
    );
    state
        .sums
        .into_iter()
        .map(|buckets| buckets.into_iter().map(|bucket| bucket.reduce()).collect())
        .collect()
}

/// Stage-7 Hamming-weight claim reduction: `PrepareKernel` front of the
/// optimized kernel.
pub struct OptimizedHammingWeightClaimReduction;

impl<F: JoltField> PrepareKernel<F, HammingWeightClaimReduction<F>>
    for OptimizedHammingWeightClaimReduction
{
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, HammingWeightClaimReduction<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = HammingWeightClaimReduction<F>>>, KernelError<F>>
    {
        let relation = inputs.relation;
        let dimensions = relation.dimensions();
        let layout = dimensions.layout;
        let r_cycle = relation.r_cycle();
        let r_address = relation.r_address();
        let virtualization_points = relation.virtualization_points();
        if r_address.len() != dimensions.log_k_chunk
            || virtualization_points.len() != layout.total()
        {
            return Err(KernelError::InvariantViolation {
                reason: "hamming reduction reference point shapes disagree with the layout",
            });
        }
        let k_chunk = 1usize << dimensions.log_k_chunk;
        let cycles = 1usize << r_cycle.len();

        let rows = InstructionCycleRow::shared(session, witness, cycles)?;
        let selectors = FamilySelectors::new(
            (layout.instruction(), layout.bytecode(), layout.ram()),
            dimensions.log_k_chunk,
        )?;
        let g_evals = pushforwards(&rows, r_cycle, &selectors, k_chunk, cfg!(feature = "akita"));
        let g_tables = g_evals.into_iter().map(Polynomial::new).collect();

        #[cfg(not(feature = "akita"))]
        let gamma_powers = gamma_powers(inputs.challenges.gamma, 3 * layout.total());
        let eq_bool = eq_table(r_address);
        #[cfg(not(feature = "akita"))]
        let weight_tables: Vec<Polynomial<F>> = virtualization_points
            .iter()
            .enumerate()
            .map(|(i, point)| {
                if point.len() != dimensions.log_k_chunk {
                    return Err(KernelError::InvariantViolation {
                        reason: "hamming virtualization point has the wrong variable count",
                    });
                }
                let eq_virt = eq_table(point);
                Ok(Polynomial::new(
                    (0..k_chunk)
                        .map(|k| {
                            gamma_powers[3 * i]
                                + gamma_powers[3 * i + 1] * eq_bool[k]
                                + gamma_powers[3 * i + 2] * eq_virt[k]
                        })
                        .collect(),
                ))
            })
            .collect::<Result<_, _>>()?;

        // Instruction and bytecode use the paper's digit-zero recentering;
        // RAM keeps the base three-leg reduction.
        #[cfg(feature = "akita")]
        let weight_tables: Vec<Polynomial<F>> = {
            let ra_terms = layout
                .polynomials()
                .map(|polynomial| match polynomial {
                    JoltRaPolynomial::Instruction(_) | JoltRaPolynomial::Bytecode(_) => 2,
                    JoltRaPolynomial::Ram(_) => 3,
                })
                .sum::<usize>();
            let gamma_powers = gamma_powers(inputs.challenges.gamma, ra_terms);
            let at_digit_zero = |point: &[F]| {
                point
                    .iter()
                    .fold(F::one(), |acc, coordinate| acc * (F::one() - *coordinate))
            };
            let eq_bool_digit_zero = at_digit_zero(r_address);
            let mut weights = Vec::with_capacity(layout.total());
            let mut power = 0;
            for (i, polynomial) in layout.polynomials().enumerate() {
                let point = &virtualization_points[i];
                if point.len() != dimensions.log_k_chunk {
                    return Err(KernelError::InvariantViolation {
                        reason: "hamming virtualization point has the wrong variable count",
                    });
                }
                let eq_virt = eq_table(point);
                match polynomial {
                    JoltRaPolynomial::Instruction(_) | JoltRaPolynomial::Bytecode(_) => {
                        let eq_virt_digit_zero = at_digit_zero(point);
                        weights.push(Polynomial::new(
                            (0..k_chunk)
                                .map(|k| {
                                    gamma_powers[power] * (eq_bool[k] - eq_bool_digit_zero)
                                        + gamma_powers[power + 1]
                                            * (eq_virt[k] - eq_virt_digit_zero)
                                })
                                .collect(),
                        ));
                        power += 2;
                    }
                    JoltRaPolynomial::Ram(_) => {
                        weights.push(Polynomial::new(
                            (0..k_chunk)
                                .map(|k| {
                                    gamma_powers[power]
                                        + gamma_powers[power + 1] * eq_bool[k]
                                        + gamma_powers[power + 2] * eq_virt[k]
                                })
                                .collect(),
                        ));
                        power += 3;
                    }
                }
            }
            debug_assert_eq!(power, ra_terms);
            weights
        };

        let output_openings: Vec<JoltOpeningId> = layout
            .openings(JoltRelationId::HammingWeightClaimReduction)
            .collect();
        Ok(Box::new(HammingWeightKernel {
            progress: RoundProgress::new(relation.rounds()),
            g_tables,
            weight_tables,
            output_openings,
        }))
    }
}

#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
struct HammingWeightKernel<F: JoltField> {
    progress: RoundProgress,
    /// Pushforwards `G_i`, canonical layout order.
    g_tables: Vec<Polynomial<F>>,
    /// Combined claim weights `W_i`, index-aligned with `g_tables`.
    weight_tables: Vec<Polynomial<F>>,
    #[cfg_attr(feature = "allocative", allocative(visit = crate::backend::visit_heap_free_elements))]
    output_openings: Vec<JoltOpeningId>,
}
impl<F: JoltField> HammingWeightKernel<F> {
    fn bind(&mut self, challenge: F) {
        bind_all(
            self.g_tables
                .iter_mut()
                .chain(self.weight_tables.iter_mut()),
            challenge,
        );
        self.progress.advance();
    }

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

impl<F: JoltField> ProveRounds<F> for HammingWeightKernel<F> {
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
        let half = self.weight_tables[0].len() / 2;

        let evals = par_sum_pair_groups(half, 2, |acc, y| {
            let group = self.group_evals(y);
            acc[0] += group[0];
            acc[1] += group[1];
        });

        Ok(round_poly_from_skipped_evals(&evals, previous_claim))
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind);
        Ok(())
    }
}

impl<F: JoltField> SumcheckKernel<F> for HammingWeightKernel<F> {
    type Relation = HammingWeightClaimReduction<F>;

    fn output_claims(
        &mut self,
        _inputs: &SumcheckInputClaims<F, Self::Relation>,
    ) -> Result<SumcheckOutputClaims<F, Self::Relation>, SumcheckKernelError<F>> {
        self.progress.require_complete()?;
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

#[cfg(all(test, not(feature = "akita")))]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use jolt_claims::protocols::jolt::geometry::claim_reductions::hamming_weight::HammingWeightClaimReductionDimensions;
    use jolt_claims::protocols::jolt::geometry::ra::JoltRaPolynomialLayout;
    use jolt_claims::protocols::jolt::{JoltCommittedPolynomial, JoltPolynomialId};
    use jolt_field::{Fr, Ring};
    use jolt_verifier::stages::stage7::hamming_weight_claim_reduction::{
        HammingWeightClaimReductionChallenges, HammingWeightClaimReductionInputClaims,
    };
    use jolt_witness::testing::with_sample_backend;
    use jolt_witness::JoltWitnessOracle;

    use super::*;
    use crate::optimized::parity::{
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

#[cfg(all(test, feature = "akita"))]
#[expect(clippy::unwrap_used, reason = "test module")]
mod akita_tests {
    use jolt_claims::protocols::jolt::geometry::claim_reductions::hamming_weight::HammingWeightClaimReductionDimensions;
    use jolt_field::{Fr, Ring};
    use jolt_verifier::stages::stage7::hamming_weight_claim_reduction::{
        HammingWeightClaimReductionChallenges, HammingWeightClaimReductionInputClaims,
    };

    use super::*;
    use crate::optimized::booleanity::testing::with_booleanity_backend;
    use crate::optimized::parity::{probe_input_claim, run_lockstep, synthetic_point};
    use crate::{ProofSession, ReferenceBackend};

    fn hamming_weight_parity(log_t: usize, log_k_chunk: u8) {
        with_booleanity_backend(log_t, log_k_chunk, |backend, base_dimensions| {
            let dimensions = HammingWeightClaimReductionDimensions::new(
                base_dimensions.layout,
                base_dimensions.log_k_chunk,
            );
            let relation = HammingWeightClaimReduction::new(
                dimensions,
                synthetic_point(log_t, 3),
                synthetic_point(dimensions.log_k_chunk, 5),
                (0..dimensions.layout.total())
                    .map(|index| synthetic_point(dimensions.log_k_chunk, 7 + index as u64))
                    .collect(),
            );
            let challenges = HammingWeightClaimReductionChallenges {
                gamma: Fr::from_u64(23),
            };
            let claims = HammingWeightClaimReductionInputClaims::<Fr>::default();
            let input_points = HammingWeightClaimReductionInputClaims::<Vec<Fr>>::default();

            let mut session = ProofSession::default();
            let mut reference = ReferenceBackend
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
            run_lockstep(
                reference.as_mut(),
                optimized.as_mut(),
                claim,
                &synthetic_point(dimensions.log_k_chunk, 301),
            );
            assert_eq!(
                reference.output_claims(&claims).unwrap(),
                optimized.output_claims(&claims).unwrap()
            );
        });
    }

    #[test]
    fn hamming_weight_reduction_matches_reference_k16() {
        hamming_weight_parity(2, 4);
    }

    #[test]
    fn hamming_weight_reduction_matches_reference_k256() {
        hamming_weight_parity(3, 8);
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod pushforward_tests {
    use super::{pushforwards, FamilySelectors, InstructionCycleRow};
    use crate::optimized::constant_suffix::ConstantSuffix;
    #[cfg(feature = "akita")]
    use jolt_field::Prime128OffsetA7F7;
    use jolt_field::{Fr, JoltField};
    use jolt_poly::EqPolynomial;
    #[cfg(feature = "akita")]
    use jolt_witness::witnesses::FusedInc;

    #[cfg(feature = "akita")]
    fn inc(index: u128) -> i128 {
        let magnitude = (index % (1u128 << 64)) as i128;
        if index.is_multiple_of(2) {
            magnitude
        } else {
            -magnitude
        }
    }

    fn row(index: u128, pc: usize, address: Option<u64>) -> InstructionCycleRow {
        InstructionCycleRow::new(
            index,
            None,
            false,
            pc,
            address,
            #[cfg(feature = "akita")]
            FusedInc(inc(index)),
        )
    }

    fn check<F: JoltField>() {
        for bits in [1, 2, 3, 4, 8] {
            for counts in [(128 / bits, 3, 4), (2, 2, 2), (0, 2, 1), (2, 0, 0)] {
                let selectors = FamilySelectors::new::<F>(counts, bits).unwrap();
                for prefix in [0, 1, 15, 16, 17, 33, 63] {
                    for hot_tail in [false, true] {
                        let tail = if hot_tail {
                            row(u128::MAX, 37, Some(0))
                        } else {
                            row(0, 0, None)
                        };
                        let mut rows = vec![tail; 64];
                        for (j, item) in rows[..prefix].iter_mut().enumerate() {
                            let value = (j as u128 + 1) * 0xdead_beef_1234_5678;
                            let value = (value << 64) | value;
                            *item = row(
                                value,
                                j % 23,
                                if j % 3 == 0 {
                                    None
                                } else {
                                    Some((j % 19) as u64)
                                },
                            );
                        }
                        let compact: ConstantSuffix<_> = rows.clone().into();
                        for boolean_point in [false, true] {
                            let point: Vec<_> = (0..6)
                                .map(|i| {
                                    F::from_u64(if boolean_point { i % 2 } else { 17 + i * 3 })
                                })
                                .collect();
                            let weights = EqPolynomial::<F>::evals(&point, None);
                            for omit_zero in [false, true] {
                                let mut expected = vec![
                                    vec![F::zero(); 1 << bits];
                                    counts.0 + counts.1 + counts.2
                                ];
                                for (row, &weight) in rows.iter().zip(&weights) {
                                    let mut column = 0;
                                    for selector in &selectors.instruction {
                                        expected[column]
                                            [selector.chunk_u128(row.lookup_index())] += weight;
                                        column += 1;
                                    }
                                    for selector in &selectors.bytecode {
                                        expected[column]
                                            [selector.chunk_usize(row.bytecode_pc())] += weight;
                                        column += 1;
                                    }
                                    for selector in &selectors.ram {
                                        if let Some(address) = row.remapped_ram_address() {
                                            expected[column]
                                                [selector.chunk_usize(address as usize)] += weight;
                                        }
                                        column += 1;
                                    }
                                }
                                if omit_zero {
                                    let ram = counts.0 + counts.1..counts.0 + counts.1 + counts.2;
                                    for (index, bucket) in expected.iter_mut().enumerate() {
                                        if !ram.contains(&index) {
                                            bucket[0] = F::zero();
                                        }
                                    }
                                }
                                assert_eq!(
                                    pushforwards(
                                        &compact,
                                        &point,
                                        &selectors,
                                        1 << bits,
                                        omit_zero
                                    ),
                                    expected
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn tensor_histograms_match_dense_equality() {
        check::<Fr>();
    }
    #[cfg(feature = "akita")]
    #[test]
    fn tensor_histograms_match_dense_equality_akita() {
        check::<Prime128OffsetA7F7>();
    }
}
