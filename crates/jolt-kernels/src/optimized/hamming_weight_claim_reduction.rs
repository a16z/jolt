//! Optimized Hamming-weight claim-reduction (stage 7) kernel, byte-parity
//! twin of [`crate::reference::hamming_weight_claim_reduction`].
//!
//! Carries forward the former prover's shared-RA `compute_all_G` technique:
//!
//! - **Shared-eq pushforwards over every RA family in one trace pass**: each
//!   `G_i(k) = Σ_j eq(r_cycle, j) · ra_i(k, j)` collapses, for a one-hot
//!   `ra_i`, to `G_i[chunk_i(j)] += eq_cycle[j]` — one `O(T)` bundle walk
//!   accumulates ALL `N = instruction_d + bytecode_d + ram_d` pushforwards
//!   against ONE shared `eq(r_cycle)` table (every stage-6b claim family
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
#[cfg(feature = "akita")]
use jolt_claims::protocols::jolt::lattice::geometry::balanced_inc_value;
#[cfg(feature = "akita")]
use jolt_claims::protocols::jolt::lattice::relations::digit_zero::{
    reduced_balanced_inc_carry_opening, reduced_balanced_inc_digit_opening,
};
use jolt_claims::protocols::jolt::{JoltOpeningId, JoltRelationId};
use jolt_claims::OutputClaims;
use jolt_field::JoltField;
#[cfg(feature = "akita")]
use jolt_poly::boolean_point_msb;
use jolt_poly::{Polynomial, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::{
    ConcreteSumcheck, SumcheckInputClaims, SumcheckOutputClaims,
};
use jolt_verifier::stages::stage7::hamming_weight_claim_reduction::HammingWeightClaimReduction;
#[cfg(feature = "akita")]
use jolt_witness::witnesses::BalancedIncColumn;
use jolt_witness::witnesses::RaChunkSelector;
use jolt_witness::JoltWitnessPlane;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

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
    #[cfg(feature = "akita")]
    balanced_inc: Vec<BalancedIncColumn>,
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
            #[cfg(feature = "akita")]
            balanced_inc: Vec::new(),
        })
    }
}

/// All `N` pushforwards from one bundle walk against the shared cycle-eq
/// table, in canonical (instruction, bytecode, RAM) order.
fn pushforwards<F: JoltField>(
    rows: &[InstructionCycleRow],
    eq_cycle: &[F],
    selectors: &FamilySelectors,
    k_chunk: usize,
) -> Vec<Vec<F>> {
    let total = selectors.instruction.len() + selectors.bytecode.len() + selectors.ram.len() + {
        #[cfg(feature = "akita")]
        {
            selectors.balanced_inc.len()
        }
        #[cfg(not(feature = "akita"))]
        {
            0
        }
    };
    let accumulate = |range: std::ops::Range<usize>| -> Vec<Vec<F>> {
        let mut partial: Vec<Vec<F>> = (0..total).map(|_| vec![F::zero(); k_chunk]).collect();
        for j in range {
            let row = &rows[j];
            let eq = eq_cycle[j];
            let mut slot = 0;
            for selector in &selectors.instruction {
                partial[slot][selector.chunk_u128(row.lookup_index())] += eq;
                slot += 1;
            }
            for selector in &selectors.bytecode {
                partial[slot][selector.chunk_usize(row.bytecode_pc())] += eq;
                slot += 1;
            }
            for selector in &selectors.ram {
                if let Some(address) = row.remapped_ram_address() {
                    partial[slot][selector.chunk_usize(address as usize)] += eq;
                }
                slot += 1;
            }
            #[cfg(feature = "akita")]
            for column in &selectors.balanced_inc {
                partial[slot][row.fused_inc_row(*column)] += eq;
                slot += 1;
            }
        }
        partial
    };

    #[cfg(feature = "parallel")]
    {
        let num_threads = rayon::current_num_threads();
        let chunk = rows.len().div_ceil(num_threads).max(1);
        (0..rows.len())
            .into_par_iter()
            .step_by(chunk)
            .map(|start| accumulate(start..(start + chunk).min(rows.len())))
            .reduce(
                || (0..total).map(|_| vec![F::zero(); k_chunk]).collect(),
                |mut left, right| {
                    for (left, right) in left.iter_mut().zip(right) {
                        for (left, right) in left.iter_mut().zip(right) {
                            *left += right;
                        }
                    }
                    left
                },
            )
    }
    #[cfg(not(feature = "parallel"))]
    {
        accumulate(0..rows.len())
    }
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
        let eq_cycle = eq_table(r_cycle);
        let selectors = FamilySelectors::new(
            (layout.instruction(), layout.bytecode(), layout.ram()),
            dimensions.log_k_chunk,
        )?;
        #[cfg(feature = "akita")]
        let mut selectors = selectors;
        #[cfg(feature = "akita")]
        {
            selectors
                .balanced_inc
                .extend((0..dimensions.chunking().chunk_count()).map(|index| {
                    BalancedIncColumn::Digit {
                        width: dimensions.log_k_chunk,
                        index,
                    }
                }));
            selectors.balanced_inc.push(BalancedIncColumn::Carry {
                width: dimensions.log_k_chunk,
            });
        }
        let g_evals = pushforwards(&rows, &eq_cycle, &selectors, k_chunk);
        #[cfg(feature = "akita")]
        let mut g_evals = g_evals;
        #[cfg(feature = "akita")]
        {
            let (ra_evals, increment_evals) = g_evals.split_at_mut(layout.total());
            for table in ra_evals
                .iter_mut()
                .take(layout.instruction() + layout.bytecode())
                .chain(increment_evals)
            {
                table[0] = F::zero();
            }
        }
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
        // RAM keeps the base three-leg reduction. Increment columns are
        // recentered and followed by the fused decode power.
        #[cfg(feature = "akita")]
        let weight_tables: Vec<Polynomial<F>> = {
            let chunk_count = dimensions.chunking().chunk_count();
            let ra_terms = layout
                .polynomials()
                .map(|polynomial| match polynomial {
                    JoltRaPolynomial::Instruction(_) | JoltRaPolynomial::Bytecode(_) => 2,
                    JoltRaPolynomial::Ram(_) => 3,
                })
                .sum::<usize>();
            let decode_power = ra_terms + chunk_count + 1;
            let gamma_powers = gamma_powers(inputs.challenges.gamma, decode_power + 1);
            let at_digit_zero = |point: &[F]| {
                point
                    .iter()
                    .fold(F::one(), |acc, coordinate| acc * (F::one() - *coordinate))
            };
            let eq_bool_digit_zero = at_digit_zero(r_address);
            let mut weights = Vec::with_capacity(layout.total() + chunk_count + 1);
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
            let balanced_values = (0..k_chunk)
                .map(|row| balanced_inc_value(&boolean_point_msb::<F>(dimensions.log_k_chunk, row)))
                .collect::<Vec<_>>();
            for index in 0..chunk_count {
                let offset = ra_terms + index;
                let decode_scale =
                    gamma_powers[decode_power] * dimensions.chunking().place_value::<F>(index);
                weights.push(Polynomial::new(
                    (0..k_chunk)
                        .map(|k| {
                            gamma_powers[offset] * (eq_bool[k] - eq_bool_digit_zero)
                                + decode_scale * balanced_values[k]
                        })
                        .collect(),
                ));
            }
            let carry_offset = ra_terms + chunk_count;
            let decode_scale = gamma_powers[decode_power] * F::pow2(64);
            weights.push(Polynomial::new(
                (0..k_chunk)
                    .map(|k| {
                        gamma_powers[carry_offset] * (eq_bool[k] - eq_bool_digit_zero)
                            + decode_scale * balanced_values[k]
                    })
                    .collect(),
            ));
            weights
        };

        let output_openings: Vec<JoltOpeningId> = layout
            .openings(JoltRelationId::HammingWeightClaimReduction)
            .collect();
        #[cfg(feature = "akita")]
        let mut output_openings = output_openings;
        #[cfg(feature = "akita")]
        {
            output_openings.extend(
                (0..dimensions.chunking().chunk_count()).map(reduced_balanced_inc_digit_opening),
            );
            output_openings.push(reduced_balanced_inc_carry_opening());
        }

        Ok(Box::new(HammingWeightKernel {
            progress: RoundProgress::new(relation.rounds()),
            g_tables,
            weight_tables,
            output_openings,
        }))
    }
}

#[cfg_attr(
    feature = "allocative",
    derive(allocative::Allocative),
    allocative(bound = "F: JoltField")
)]
struct HammingWeightKernel<F: JoltField> {
    progress: RoundProgress,
    /// Pushforwards `G_i`, canonical layout order.
    g_tables: Vec<Polynomial<F>>,
    /// Combined claim weights `W_i`, index-aligned with `g_tables`.
    weight_tables: Vec<Polynomial<F>>,
    #[cfg_attr(feature = "allocative", allocative(visit = jolt_poly::visit_scalars))]
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

/// Byte parity against the reference kernel over the sample backend. All
/// three families are live at fixture scale (the instruction family alone
/// contributes 128/log_k_chunk polynomials), so the multi-family pushforward
/// walk, cold RAM/bytecode cycles, and per-polynomial weight fusion are all
/// exercised.
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
    use jolt_claims::protocols::jolt::lattice::relations::digit_zero::LatticeDigitZeroClaimReductionDimensions;
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
            let dimensions = LatticeDigitZeroClaimReductionDimensions::new(
                base_dimensions.layout,
                base_dimensions.log_k_chunk,
            )
            .unwrap();
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
