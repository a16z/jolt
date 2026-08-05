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
//!   independent `O(K_chunk·T)` folds over materialized one-hot grids.
//! - **One-hot weight fusion**: the three per-polynomial claim weights
//!   `γ^{3i} + γ^{3i+1}·eq_bool(k) + γ^{3i+2}·eq_virt_i(k)` are one combined
//!   multilinear `W_i(k)` (the Hamming-weight leg's constant-1 rides the
//!   constant term), so the round summand is `Σ_i G_i·W_i` — `2N` bound
//!   tables instead of `2N + 1` and one fused multiply per pair per point.
//! - **Eval-at-1 recovery** and **rayon walks** (module docs on
//!   [`crate::optimized`]).

#[cfg(feature = "akita")]
use jolt_claims::protocols::jolt::geometry::ra::JoltRaPolynomial;
#[cfg(feature = "akita")]
use jolt_claims::protocols::jolt::lattice::geometry::balanced_inc_value;
use jolt_claims::protocols::jolt::{JoltOpeningId, JoltRelationId};
use jolt_claims::OutputClaims;
use jolt_field::Field;
#[cfg(feature = "akita")]
use jolt_poly::boolean_point_msb;
use jolt_poly::{Polynomial, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::{
    ConcreteSumcheck, SumcheckInputClaims, SumcheckOutputClaims,
};
use jolt_verifier::stages::stage7::hamming_weight_claim_reduction::{
    HammingWeightClaimReduction, HammingWeightClaimReductionChallenges,
};
use jolt_witness::witnesses::RaChunkSelector;
#[cfg(feature = "akita")]
use jolt_witness::witnesses::UnsignedIncLane;
use jolt_witness::JoltWitnessPlane;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::instruction_read_raf::{shared_instruction_rows, InstructionCycleRow};
#[cfg(feature = "parallel")]
use super::support::merge_evals;
use super::support::{bind_all, eq_table, pair, round_poly_from_skipped_evals};
#[cfg(all(feature = "akita", feature = "metal", target_os = "macos"))]
use crate::metal::solinas::BooleanitySelector;
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

/// Per-family chunk selectors in canonical layout order.
struct FamilySelectors {
    instruction: Vec<RaChunkSelector>,
    bytecode: Vec<RaChunkSelector>,
    ram: Vec<RaChunkSelector>,
    #[cfg(feature = "akita")]
    unsigned_inc: Vec<UnsignedIncLane>,
}

impl FamilySelectors {
    fn new<F: Field>(
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
            unsigned_inc: Vec::new(),
        })
    }

    fn len(&self) -> usize {
        self.instruction.len() + self.bytecode.len() + self.ram.len() + {
            #[cfg(feature = "akita")]
            {
                self.unsigned_inc.len()
            }
            #[cfg(not(feature = "akita"))]
            {
                0
            }
        }
    }

    #[cfg(all(feature = "akita", feature = "metal", target_os = "macos"))]
    fn metal_selectors(&self) -> Vec<BooleanitySelector> {
        self.instruction
            .iter()
            .map(|selector| BooleanitySelector::Lookup {
                shift: selector.shift() as u32,
            })
            .chain(
                self.bytecode
                    .iter()
                    .map(|selector| BooleanitySelector::Bytecode {
                        shift: selector.shift() as u32,
                    }),
            )
            .chain(self.ram.iter().map(|selector| BooleanitySelector::Ram {
                shift: selector.shift() as u32,
            }))
            .chain(self.unsigned_inc.iter().map(|lane| match lane {
                UnsignedIncLane::Chunk { width, index } => BooleanitySelector::FusedInc {
                    shift: (width * index) as u32,
                },
                UnsignedIncLane::Msb { .. } => BooleanitySelector::FusedIncMsb,
            }))
            .collect()
    }
}

/// All `N` pushforwards from one bundle walk against the shared cycle-eq
/// table, in canonical (instruction, bytecode, RAM) order.
fn pushforwards<F: Field>(
    rows: &[InstructionCycleRow],
    eq_cycle: &[F],
    selectors: &FamilySelectors,
    k_chunk: usize,
) -> Vec<Vec<F>> {
    let total = selectors.len();
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
                if let Some(pc) = row.mapped_pc() {
                    partial[slot][selector.chunk_usize(pc)] += eq;
                }
                slot += 1;
            }
            for selector in &selectors.ram {
                if let Some(address) = row.remapped_ram_address() {
                    partial[slot][selector.chunk_usize(address as usize)] += eq;
                }
                slot += 1;
            }
            #[cfg(feature = "akita")]
            for lane in &selectors.unsigned_inc {
                partial[slot][row.fused_inc_hot_lane(*lane)] += eq;
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

pub(crate) struct HammingWeightPreparePlan<F: Field> {
    rounds: usize,
    reference_cycle: Vec<F>,
    selectors: FamilySelectors,
    k_chunk: usize,
    weight_tables: Vec<Polynomial<F>>,
    #[cfg(feature = "akita")]
    baseline_table: Polynomial<F>,
    output_openings: Vec<JoltOpeningId>,
}

impl<F: Field> HammingWeightPreparePlan<F> {
    pub(crate) fn new(
        relation: &HammingWeightClaimReduction<F>,
        challenges: &HammingWeightClaimReductionChallenges<F>,
    ) -> Result<Self, KernelError<F>> {
        let dimensions = relation.dimensions();
        let layout = dimensions.layout;
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
        let selectors = FamilySelectors::new(
            (layout.instruction(), layout.bytecode(), layout.ram()),
            dimensions.log_k_chunk,
        )?;
        #[cfg(feature = "akita")]
        let mut selectors = selectors;
        #[cfg(feature = "akita")]
        {
            selectors
                .unsigned_inc
                .extend((0..dimensions.chunking().chunk_count()).map(|index| {
                    UnsignedIncLane::Chunk {
                        width: dimensions.log_k_chunk,
                        index,
                    }
                }));
            selectors.unsigned_inc.push(UnsignedIncLane::Msb {
                width: dimensions.log_k_chunk,
            });
        }

        let gamma = challenges.gamma;
        #[cfg(not(feature = "akita"))]
        let mut gamma_powers = vec![F::one(); 3 * layout.total()];
        #[cfg(not(feature = "akita"))]
        for i in 1..gamma_powers.len() {
            gamma_powers[i] = gamma_powers[i - 1] * gamma;
        }
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

        #[cfg(feature = "akita")]
        let (weight_tables, baseline_table) = {
            let chunk_count = dimensions.chunking().chunk_count();
            let ra_terms = 3 * layout.total();
            let decode_power = ra_terms + 2 * (chunk_count + 1);
            let mut gamma_powers = vec![F::one(); decode_power + 1];
            for i in 1..gamma_powers.len() {
                gamma_powers[i] = gamma_powers[i - 1] * gamma;
            }
            let at_default = |point: &[F]| {
                point
                    .iter()
                    .fold(F::one(), |acc, coordinate| acc * (F::one() - *coordinate))
            };
            let eq_bool_default = at_default(r_address);
            let ram_hamming_weight =
                relation
                    .ram_hamming_weight()
                    .ok_or(KernelError::InvariantViolation {
                        reason: "Akita hamming reduction is missing the RAM activation",
                    })?;
            let mut baseline = F::zero();
            let mut weights = Vec::with_capacity(layout.total() + chunk_count + 1);
            for (i, polynomial) in layout.polynomials().enumerate() {
                let point = &virtualization_points[i];
                if point.len() != dimensions.log_k_chunk {
                    return Err(KernelError::InvariantViolation {
                        reason: "hamming virtualization point has the wrong variable count",
                    });
                }
                let eq_virt = eq_table(point);
                let eq_virt_default = at_default(point);
                let hamming_weight = match polynomial {
                    JoltRaPolynomial::Instruction(_) | JoltRaPolynomial::Bytecode(_) => F::one(),
                    JoltRaPolynomial::Ram(_) => ram_hamming_weight,
                };
                baseline += hamming_weight
                    * (gamma_powers[3 * i]
                        + gamma_powers[3 * i + 1] * eq_bool_default
                        + gamma_powers[3 * i + 2] * eq_virt_default);
                weights.push(Polynomial::new(
                    (0..k_chunk)
                        .map(|k| {
                            gamma_powers[3 * i + 1] * (eq_bool[k] - eq_bool_default)
                                + gamma_powers[3 * i + 2] * (eq_virt[k] - eq_virt_default)
                        })
                        .collect(),
                ));
            }
            let balanced_values = (0..k_chunk)
                .map(|lane| {
                    balanced_inc_value(&boolean_point_msb::<F>(dimensions.log_k_chunk, lane))
                })
                .collect::<Vec<_>>();
            for index in 0..chunk_count {
                let offset = ra_terms + 2 * index;
                baseline += gamma_powers[offset] + gamma_powers[offset + 1] * eq_bool_default;
                let decode_scale =
                    gamma_powers[decode_power] * dimensions.chunking().place_value::<F>(index);
                weights.push(Polynomial::new(
                    (0..k_chunk)
                        .map(|k| {
                            gamma_powers[offset + 1] * (eq_bool[k] - eq_bool_default)
                                + decode_scale * balanced_values[k]
                        })
                        .collect(),
                ));
            }
            let msb_offset = ra_terms + 2 * chunk_count;
            baseline += gamma_powers[msb_offset] + gamma_powers[msb_offset + 1] * eq_bool_default;
            let decode_scale = gamma_powers[decode_power] * F::pow2(64);
            weights.push(Polynomial::new(
                (0..k_chunk)
                    .map(|k| {
                        gamma_powers[msb_offset + 1] * (eq_bool[k] - eq_bool_default)
                            + decode_scale * balanced_values[k]
                    })
                    .collect(),
            ));
            let mut baseline_table = vec![F::zero(); k_chunk];
            baseline_table[0] = baseline;
            (weights, Polynomial::new(baseline_table))
        };

        let output_openings: Vec<JoltOpeningId> = layout
            .openings(JoltRelationId::HammingWeightClaimReduction)
            .collect();
        #[cfg(feature = "akita")]
        let mut output_openings = output_openings;
        #[cfg(feature = "akita")]
        {
            output_openings.extend((0..dimensions.chunking().chunk_count()).map(
                jolt_claims::protocols::jolt::lattice::relations::hamming_weight::reduced_unsigned_inc_chunk_opening,
            ));
            output_openings.push(
                jolt_claims::protocols::jolt::lattice::relations::hamming_weight::reduced_unsigned_inc_msb_opening(),
            );
        }
        debug_assert_eq!(selectors.len(), weight_tables.len());
        debug_assert_eq!(selectors.len(), output_openings.len());

        Ok(Self {
            rounds: relation.rounds(),
            reference_cycle: relation.r_cycle().to_vec(),
            selectors,
            k_chunk,
            weight_tables,
            #[cfg(feature = "akita")]
            baseline_table,
            output_openings,
        })
    }

    pub(crate) fn cycles(&self) -> usize {
        1usize << self.reference_cycle.len()
    }

    pub(crate) fn k_chunk(&self) -> usize {
        self.k_chunk
    }

    pub(crate) fn reference_cycle(&self) -> &[F] {
        &self.reference_cycle
    }

    fn selectors(&self) -> &FamilySelectors {
        &self.selectors
    }

    #[cfg(all(feature = "akita", feature = "metal", target_os = "macos"))]
    pub(crate) fn metal_selectors(&self) -> Vec<BooleanitySelector> {
        self.selectors.metal_selectors()
    }

    #[cfg(all(feature = "akita", feature = "metal", target_os = "macos"))]
    pub(crate) fn finish_flat(
        self,
        flat_g_evals: Vec<F>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = HammingWeightClaimReduction<F>>>, KernelError<F>>
    {
        let expected = self
            .selectors
            .len()
            .checked_mul(self.k_chunk)
            .ok_or_else(|| KernelError::InvalidGeometry {
                reason: "Hamming-weight pushforward mass count overflows usize".to_owned(),
            })?;
        if flat_g_evals.len() != expected {
            return Err(KernelError::TableSizeMismatch {
                table: "Metal Hamming-weight pushforward masses".to_owned(),
                expected,
                got: flat_g_evals.len(),
            });
        }
        let g_evals = flat_g_evals
            .chunks_exact(self.k_chunk)
            .map(<[F]>::to_vec)
            .collect();
        self.finish(g_evals)
    }

    pub(crate) fn finish(
        self,
        g_evals: Vec<Vec<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = HammingWeightClaimReduction<F>>>, KernelError<F>>
    {
        if g_evals.len() != self.selectors.len() {
            return Err(KernelError::TableSizeMismatch {
                table: "Hamming-weight pushforward table count".to_owned(),
                expected: self.selectors.len(),
                got: g_evals.len(),
            });
        }
        for (index, table) in g_evals.iter().enumerate() {
            if table.len() != self.k_chunk {
                return Err(KernelError::TableSizeMismatch {
                    table: format!("Hamming-weight pushforward table {index}"),
                    expected: self.k_chunk,
                    got: table.len(),
                });
            }
        }
        #[cfg(feature = "akita")]
        let mut g_evals = g_evals;
        #[cfg(feature = "akita")]
        for table in &mut g_evals {
            table[0] = F::zero();
        }
        let g_tables = g_evals.into_iter().map(Polynomial::new).collect();
        Ok(Box::new(HammingWeightKernel {
            rounds: self.rounds,
            g_tables,
            weight_tables: self.weight_tables,
            #[cfg(feature = "akita")]
            baseline_table: self.baseline_table,
            output_openings: self.output_openings,
            rounds_bound: 0,
        }))
    }
}

/// Stage-7 Hamming-weight claim reduction: `PrepareKernel` front of the
/// optimized kernel.
pub struct OptimizedHammingWeightClaimReduction;

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
        let plan = HammingWeightPreparePlan::new(inputs.relation, inputs.challenges)?;
        let rows = {
            let _span =
                tracing::info_span!("OptimizedHammingWeightClaimReduction::row_source").entered();
            shared_instruction_rows(session, witness, plan.cycles())?
        };
        let eq_cycle = eq_table(plan.reference_cycle());
        let g_evals = pushforwards(&rows, &eq_cycle, plan.selectors(), plan.k_chunk());
        plan.finish(g_evals)
    }
}

struct HammingWeightKernel<F: Field> {
    rounds: usize,
    /// Pushforwards `G_i`, canonical layout order.
    g_tables: Vec<Polynomial<F>>,
    /// Combined claim weights `W_i`, index-aligned with `g_tables`.
    weight_tables: Vec<Polynomial<F>>,
    #[cfg(feature = "akita")]
    /// The logical default lane represented as a single delta-at-zero table.
    baseline_table: Polynomial<F>,
    output_openings: Vec<JoltOpeningId>,
    rounds_bound: usize,
}

#[cfg(feature = "allocative")]
impl<F: Field> allocative::Allocative for HammingWeightKernel<F> {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        #[cfg(feature = "akita")]
        use crate::backend::poly_heap_bytes;
        use crate::backend::{polys_heap_bytes, vec_heap_bytes};
        let mut visitor = visitor.enter_self_sized::<Self>();
        visitor.visit_simple(
            allocative::Key::new("g_tables"),
            polys_heap_bytes(&self.g_tables),
        );
        visitor.visit_simple(
            allocative::Key::new("weight_tables"),
            polys_heap_bytes(&self.weight_tables),
        );
        #[cfg(feature = "akita")]
        visitor.visit_simple(
            allocative::Key::new("baseline_table"),
            poly_heap_bytes(&self.baseline_table),
        );
        visitor.visit_simple(
            allocative::Key::new("output_openings"),
            vec_heap_bytes(&self.output_openings),
        );
        visitor.exit();
    }
}

impl<F: Field> HammingWeightKernel<F> {
    fn bind(&mut self, challenge: F) {
        bind_all(
            self.g_tables
                .iter_mut()
                .chain(self.weight_tables.iter_mut()),
            challenge,
        );
        #[cfg(feature = "akita")]
        bind_all(std::iter::once(&mut self.baseline_table), challenge);
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
        #[cfg(feature = "akita")]
        {
            let (baseline_lo, baseline_hi) = pair(&self.baseline_table, y);
            out[0] += baseline_lo;
            out[1] += baseline_hi + baseline_hi - baseline_lo;
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
#[cfg(all(test, not(feature = "akita")))]
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
                None,
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
    use jolt_claims::protocols::jolt::lattice::relations::hamming_weight::LatticeHammingWeightClaimReductionDimensions;
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_verifier::stages::stage7::hamming_weight_claim_reduction::{
        HammingWeightClaimReductionChallenges, HammingWeightClaimReductionInputClaims,
    };

    use super::*;
    use crate::optimized::booleanity::testing::with_booleanity_backend;
    use crate::optimized::harness::{probe_input_claim, run_lockstep, synthetic_point};
    use crate::{ProofSession, ReferenceBackend};

    fn hamming_weight_parity(log_t: usize, log_k_chunk: u8) {
        with_booleanity_backend(log_t, log_k_chunk, |backend, base_dimensions| {
            let dimensions = LatticeHammingWeightClaimReductionDimensions::new(
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
                Some(Fr::from_u64(17)),
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
