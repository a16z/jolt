//! Optimized instruction claim reduction (stage 2).
//!
//! The summand `eq(τ, j)·C(j)`, `C = Σ_i γ^i·o_i` over five operands, is
//! linear in the witness, and `eq` factors across the halves of
//! `j = x_H·2^L + x_L`: `eq(τ, j) = eq(τ_H, x_H)·eq(τ_L, x_L)`, where the
//! `L = ⌈log_t/2⌉` low variables bind first (`LowToHigh`). No table spans the
//! cycle domain:
//!
//! - Rounds `0..L` run the Gruen message on `(τ_L, P)`, where one row pass
//!   builds `P(x_L) = Σ_{x_H} eq(τ_H, x_H)·C(x_L, x_H)`. Summing out `x_H`
//!   commutes with binding `x_L`, so every message equals the dense
//!   summand's.
//! - Binding the last low variable runs a second pass for
//!   `Q_i(x_H) = Σ_{x_L} eq(r_L, x_L)·o_i(x_L, x_H)` and `Q = Σ_i γ^i·Q_i`;
//!   rounds `L..log_t` run the Gruen message on `(τ_H, Q)` scaled by the low
//!   half's bound `eq(τ_L, r_L)`.
//! - The output claims are `o_i(r) = Σ_{x_H} eq(r_H, x_H)·Q_i(x_H)`.

use std::ops::Range;

use jolt_claims::protocols::jolt::relations::claim_reductions::instruction::InstructionClaimReductionOutputClaims;
use jolt_claims::protocols::jolt::{InstructionClaimReductionPublic, JoltDerivedId};
use jolt_field::{Accumulator, JoltField};
use jolt_poly::{BindingOrder, EqPolynomial, GruenSplitEqPolynomial, Polynomial, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_utils::unsafe_allocate_zero_vec;
use jolt_verifier::stages::relations::{
    ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckInputPoints, SumcheckOutputPoints,
};
use jolt_verifier::stages::stage2::instruction_claim_reduction::InstructionClaimReduction;
use jolt_witness::witnesses::{
    LeftInstructionInput, LeftLookupOperand, LookupOutput, RightInstructionInput,
    RightLookupOperand,
};
use jolt_witness::{JoltWitnessPlane, WitnessBundle, WitnessError};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::support::{
    pin_derived_term, split_at_tasks, task_ranges, GruenRoundMessage, KernelRows, RoundChallenges,
    VISIT_ROWS,
};
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

/// The five reduced tables, in output-claim declaration order.
const NUM_TABLES: usize = 5;

/// One cycle's five reduced instruction operands as native scalars — the
/// rows both passes fold.
#[derive(Clone, Copy, Debug, WitnessBundle)]
pub struct InstructionOperandRow {
    pub lookup_output: LookupOutput,
    pub left_lookup_operand: LeftLookupOperand,
    pub right_lookup_operand: RightLookupOperand,
    pub left_instruction_input: LeftInstructionInput,
    pub right_instruction_input: RightInstructionInput,
}

impl InstructionOperandRow {
    /// `sums[i] += weight·o_i` for the five operands, except that a negative
    /// right instruction input adds its magnitude to `sums[NUM_TABLES]`;
    /// `weight_hi = weight·2^64` scales a 128-bit operand's high limb.
    #[inline]
    fn fmadd_operands<F: JoltField>(
        &self,
        sums: &mut [F::Accumulator; NUM_TABLES + 1],
        [weight, weight_hi]: [F; 2],
    ) {
        sums[0].fmadd_u64(weight, self.lookup_output.0);
        sums[1].fmadd_u64(weight, self.left_lookup_operand.0);
        let right_lookup = self.right_lookup_operand.0;
        sums[2].fmadd_u64(weight, right_lookup as u64);
        sums[2].fmadd_u64(weight_hi, (right_lookup >> 64) as u64);
        sums[3].fmadd_u64(weight, self.left_instruction_input.0);
        let right_input = self.right_instruction_input.0;
        let magnitude = right_input.unsigned_abs();
        let sum = &mut sums[if right_input < 0 { NUM_TABLES } else { 4 }];
        sum.fmadd_u64(weight, magnitude as u64);
        sum.fmadd_u64(weight_hi, (magnitude >> 64) as u64);
    }
}

pub struct OptimizedInstructionClaimReduction;

impl<F: JoltField> PrepareKernel<F, InstructionClaimReduction<F>>
    for OptimizedInstructionClaimReduction
{
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, InstructionClaimReduction<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = InstructionClaimReduction<F>>>, KernelError<F>>
    {
        let tau = inputs.relation.tau_low();
        let rows =
            KernelRows::from_witness(witness, tau.len(), "instruction claim-reduction trace rows")?;
        Ok(Box::new(OptimizedInstructionClaimReductionKernel::new(
            tau,
            rows,
            inputs.challenges.gamma,
        )?))
    }
}

/// The coefficients of `C = Σ_i γ^i·o_i` on a row's seven `u64` limbs.
struct CombineCoefficients<F> {
    gamma_powers: [F; NUM_TABLES],
    right_lookup_hi: F,
    /// `(lo, hi)` coefficient pairs for the signed lane: `.0` negative, `.1`
    /// positive.
    right_input_coeffs: ((F, F), (F, F)),
}

impl<F: JoltField> CombineCoefficients<F> {
    fn new(gamma: F) -> Self {
        let gamma_sqr = gamma * gamma;
        let gamma_powers = [
            F::one(),
            gamma,
            gamma_sqr,
            gamma_sqr * gamma,
            gamma_sqr * gamma_sqr,
        ];
        let shift_64 = F::from_u128(1u128 << 64);
        let right_lookup_hi = gamma_powers[2] * shift_64;
        let right_input_coeffs = {
            let positive = (gamma_powers[4], gamma_powers[4] * shift_64);
            ((-positive.0, -positive.1), positive)
        };
        Self {
            gamma_powers,
            right_lookup_hi,
            right_input_coeffs,
        }
    }

    /// Every coefficient times `scale`.
    fn scaled(&self, scale: F) -> Self {
        let (lo, hi) = self.right_input_coeffs.1;
        let positive = (scale * lo, scale * hi);
        Self {
            gamma_powers: self.gamma_powers.map(|power| scale * power),
            right_lookup_hi: scale * self.right_lookup_hi,
            right_input_coeffs: ((-positive.0, -positive.1), positive),
        }
    }

    /// `acc += C(row)` under these coefficients, as seven unreduced `u64`
    /// products.
    #[inline]
    fn fmadd(&self, acc: &mut F::Accumulator, row: &InstructionOperandRow) {
        acc.fmadd_u64(self.gamma_powers[0], row.lookup_output.0);
        acc.fmadd_u64(self.gamma_powers[1], row.left_lookup_operand.0);
        let right_lookup = row.right_lookup_operand.0;
        acc.fmadd_u64(self.gamma_powers[2], right_lookup as u64);
        acc.fmadd_u64(self.right_lookup_hi, (right_lookup >> 64) as u64);
        acc.fmadd_u64(self.gamma_powers[3], row.left_instruction_input.0);
        let right_input = row.right_instruction_input.0;
        let magnitude = right_input.unsigned_abs();
        let (lo, hi) = if right_input < 0 {
            self.right_input_coeffs.0
        } else {
            self.right_input_coeffs.1
        };
        acc.fmadd_u64(lo, magnitude as u64);
        acc.fmadd_u64(hi, (magnitude >> 64) as u64);
    }
}

/// `(τ_H, τ_L)`: the trailing `⌈log_t/2⌉` coordinates `τ_L` pair the cycle
/// index's low bits, which `LowToHigh` binds first.
fn tau_halves<F>(tau: &[F]) -> (&[F], &[F]) {
    tau.split_at(tau.len() / 2)
}

/// The cycle-variable half the rounds bind: its Gruen split eq and its
/// `γ`-combined table, bound in place.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
enum Half<F: JoltField> {
    /// Rounds `0..L`: `P(x_L)` over `τ_L`.
    Low {
        split_eq: GruenSplitEqPolynomial<F>,
        combined: Polynomial<F>,
    },
    /// Rounds `L..log_t` and the output claims: `Q(x_H)` over `τ_H` scaled by
    /// `eq(τ_L, r_L)`, and the five unbound operand tables `Q_i(x_H)`.
    High {
        split_eq: GruenSplitEqPolynomial<F>,
        combined: Polynomial<F>,
        operands: [Vec<F>; NUM_TABLES],
    },
}

impl<F: JoltField> Half<F> {
    fn round_tables(&self) -> (&GruenSplitEqPolynomial<F>, &Polynomial<F>) {
        match self {
            Self::Low { split_eq, combined }
            | Self::High {
                split_eq, combined, ..
            } => (split_eq, combined),
        }
    }

    fn round_tables_mut(&mut self) -> (&mut GruenSplitEqPolynomial<F>, &mut Polynomial<F>) {
        match self {
            Self::Low { split_eq, combined }
            | Self::High {
                split_eq, combined, ..
            } => (split_eq, combined),
        }
    }
}

#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
pub struct OptimizedInstructionClaimReductionKernel<F: JoltField> {
    tau: Vec<F>,
    #[cfg_attr(feature = "allocative", allocative(skip))]
    coefficients: CombineCoefficients<F>,
    /// The witness plane owns these rows; it reports them itself.
    #[cfg_attr(feature = "allocative", allocative(skip))]
    rows: KernelRows<InstructionOperandRow>,
    challenges: RoundChallenges<F>,
    half: Half<F>,
}

impl<F: JoltField> OptimizedInstructionClaimReductionKernel<F> {
    fn new(
        tau: &[F],
        rows: KernelRows<InstructionOperandRow>,
        gamma: F,
    ) -> Result<Self, KernelError<F>> {
        if tau.is_empty() {
            return Err(KernelError::Unsupported {
                reason: "the instruction claim reduction needs at least one cycle variable",
            });
        }
        let coefficients = CombineCoefficients::new(gamma);
        let (tau_hi, tau_lo) = tau_halves(tau);
        let combined = Self::sum_high_half(&rows, &coefficients, tau_hi, tau_lo.len())?;
        Ok(Self {
            tau: tau.to_vec(),
            coefficients,
            rows,
            challenges: RoundChallenges::new(tau.len()),
            half: Half::Low {
                split_eq: GruenSplitEqPolynomial::new(tau_lo, BindingOrder::LowToHigh),
                combined: Polynomial::new(combined),
            },
        })
    }

    /// `P(x_L) = Σ_{x_H} eq(τ_H, x_H)·C(x_L, x_H)` from one pass. Each task
    /// covers whole `x_H` groups of `2^L` consecutive cycles and folds them
    /// into its own slot of `2^L` accumulators, one per `x_L`, allocated
    /// before the pass; each `x_L`'s entries are then merged across the slots
    /// and reduced once. A merged accumulator takes seven `u64` products per
    /// `x_H`, `7·2^14` at the `2^28` target, far inside the `2^63`-term
    /// headroom of the fp128 and BN254 product accumulators.
    fn sum_high_half(
        rows: &KernelRows<InstructionOperandRow>,
        coefficients: &CombineCoefficients<F>,
        tau_hi: &[F],
        log_low: usize,
    ) -> Result<Vec<F>, WitnessError> {
        let group = 1usize << log_low;
        let e_hi = EqPolynomial::<F>::evals(tau_hi, None);
        let ranges = task_ranges(group * e_hi.len(), rows.segment_len(), group);
        let mut slots: Vec<Vec<F::Accumulator>> = ranges
            .iter()
            .map(|_| vec![F::Accumulator::default(); group])
            .collect();
        let task = |(slot, range): (&mut Vec<F::Accumulator>, &Range<usize>)| {
            // A callback of at most `group` aligned rows lies in one `x_H` group.
            rows.visit(range.clone(), VISIT_ROWS.min(group), |start, bundles| {
                let weights = coefficients.scaled(e_hi[start >> log_low]);
                for (sum, row) in slot[start & (group - 1)..].iter_mut().zip(bundles) {
                    weights.fmadd(sum, row);
                }
                Ok(())
            })
        };
        #[cfg(feature = "parallel")]
        slots.par_iter_mut().zip(&ranges).try_for_each(task)?;
        #[cfg(not(feature = "parallel"))]
        slots.iter_mut().zip(&ranges).try_for_each(task)?;

        let merge = |x_lo: usize| {
            let mut sum = F::Accumulator::default();
            for slot in &slots {
                sum.merge(slot[x_lo]);
            }
            sum.reduce()
        };
        #[cfg(feature = "parallel")]
        let combined: Vec<F> = (0..group).into_par_iter().map(merge).collect();
        #[cfg(not(feature = "parallel"))]
        let combined: Vec<F> = (0..group).map(merge).collect();
        Ok(combined)
    }

    /// The high half at `x_L = r_L`, from one pass:
    /// `Q_i(x_H) = Σ_{x_L} eq(r_L, x_L)·o_i(x_L, x_H)` and `Q = Σ_i γ^i·Q_i`.
    /// Each task covers whole `x_H` groups and writes their entries into its
    /// own slices of the final tables; a group's sums take at most two `u64`
    /// products per cycle. The high split eq starts from the low half's fully
    /// bound scalar `eq(τ_L, r_L)`.
    fn fix_low_half(&self) -> Result<Half<F>, WitnessError> {
        let (tau_hi, tau_lo) = tau_halves(&self.tau);
        let log_low = tau_lo.len();
        let group = 1usize << log_low;
        let len = 1usize << tau_hi.len();
        let r_lo: Vec<F> = self.challenges.as_slice().iter().rev().copied().collect();
        let shift_64 = F::from_u128(1u128 << 64);
        let weights: Vec<[F; 2]> = EqPolynomial::<F>::evals(&r_lo, None)
            .into_iter()
            .map(|weight| [weight, weight * shift_64])
            .collect();
        let gamma_powers = &self.coefficients.gamma_powers;
        let mut tables: [Vec<F>; NUM_TABLES + 1] =
            std::array::from_fn(|_| unsafe_allocate_zero_vec(len));
        let ranges = task_ranges(group * len, self.rows.segment_len(), group);
        let outputs = split_at_tasks(tables.each_mut().map(Vec::as_mut_slice), &ranges, |cycle| {
            cycle >> log_low
        });
        let task = |(range, mut outputs): (&Range<usize>, [&mut [F]; NUM_TABLES + 1])| {
            let first = range.start >> log_low;
            // Sums run across a group's callbacks; a callback of at most
            // `group` aligned rows lies in one `x_H` group.
            let mut sums = [F::Accumulator::default(); NUM_TABLES + 1];
            self.rows
                .visit(range.clone(), VISIT_ROWS.min(group), |start, bundles| {
                    for (cycle, row) in (start..).zip(bundles) {
                        let x_lo = cycle & (group - 1);
                        row.fmadd_operands::<F>(&mut sums, weights[x_lo]);
                        if x_lo + 1 == group {
                            let x_hi = (cycle >> log_low) - first;
                            let [s0, s1, s2, s3, positive, negative] = std::mem::take(&mut sums);
                            let values = [
                                s0.reduce(),
                                s1.reduce(),
                                s2.reduce(),
                                s3.reduce(),
                                positive.reduce() - negative.reduce(),
                            ];
                            let mut q = F::zero();
                            for ((table, value), power) in
                                outputs.iter_mut().zip(values).zip(gamma_powers)
                            {
                                table[x_hi] = value;
                                q += *power * value;
                            }
                            outputs[NUM_TABLES][x_hi] = q;
                        }
                    }
                    Ok(())
                })
        };
        #[cfg(feature = "parallel")]
        ranges.par_iter().zip(outputs).try_for_each(task)?;
        #[cfg(not(feature = "parallel"))]
        ranges.iter().zip(outputs).try_for_each(task)?;

        let scale = self.half.round_tables().0.current_scalar();
        let [operands @ .., combined] = tables;
        Ok(Half::High {
            split_eq: GruenSplitEqPolynomial::new_with_scaling(
                tau_hi,
                BindingOrder::LowToHigh,
                Some(scale),
            ),
            combined: Polynomial::new(combined),
            operands,
        })
    }

    /// `o_i(r) = Σ_{x_H} eq(r_H, x_H)·Q_i(x_H)`, with the high half's
    /// challenges reversed into big-endian order.
    fn operand_claims(&self) -> Result<[F; NUM_TABLES], SumcheckKernelError<F>> {
        self.challenges.require_complete()?;
        let Half::High { operands, .. } = &self.half else {
            return Err(SumcheckKernelError::InvariantViolation {
                reason: "instruction claim reduction completed before fixing its low half",
            });
        };
        let log_low = tau_halves(&self.tau).1.len();
        let r_hi: Vec<F> = self.challenges.as_slice()[log_low..]
            .iter()
            .rev()
            .copied()
            .collect();
        let weights = EqPolynomial::<F>::evals(&r_hi, None);
        Ok(operands.each_ref().map(|table| {
            let mut claim = F::Accumulator::default();
            for (&weight, &value) in weights.iter().zip(table) {
                claim.fmadd(weight, value);
            }
            claim.reduce()
        }))
    }

    /// `s(t) = ℓ(t) · Σ_y E(y) · combo(t, y)` at `t ∈ {0, 1, 2}` over the
    /// current half, with the γ-combination folded before the point
    /// interpolation (exact by linearity of binding).
    fn message(
        &self,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        const POINTS: usize = 3;
        let (split_eq, combined) = self.half.round_tables();
        let mut q_evals = split_eq.par_fold_out_in(
            || [F::zero(); POINTS],
            |acc, row, _x_in, e_in| {
                let evals = combined.evals();
                let lo = evals[2 * row];
                let step = evals[2 * row + 1] - lo;
                let mut eval = lo;
                for value in acc.iter_mut() {
                    *value += e_in * eval;
                    eval += step;
                }
            },
            |_x_out, e_out, mut acc| {
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

        split_eq.checked_round_poly(&mut q_evals, previous_claim, round)
    }

    /// Binds the current half; binding the low half's last variable fixes
    /// `x_L = r_L` for the remaining rounds.
    fn bind(&mut self, challenge: F) -> Result<(), WitnessError> {
        let (split_eq, combined) = self.half.round_tables_mut();
        split_eq.bind(challenge);
        // Avoid a fresh half-size table each round.
        let _ = combined.bind_low_to_high_in_place(challenge);
        self.challenges.push(challenge);
        if self.challenges.bound() == tau_halves(&self.tau).1.len() {
            self.half = self.fix_low_half()?;
        }
        Ok(())
    }
}

/// A row failure in the high half's pass, after `prepare` visited the rows.
fn rows_unavailable<F: JoltField>(_: WitnessError) -> SumcheckError<F> {
    SumcheckError::MissingEvaluationSource {
        kind: "instruction claim-reduction operand rows",
    }
}

impl<F: JoltField> ProveRounds<F> for OptimizedInstructionClaimReductionKernel<F> {
    fn num_rounds(&self) -> usize {
        self.challenges.total()
    }

    fn prove_round(
        &mut self,
        bind: Option<F>,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if let Some(challenge) = bind {
            self.bind(challenge).map_err(rows_unavailable)?;
        }
        self.message(round, previous_claim)
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind).map_err(rows_unavailable)
    }
}

impl<F: JoltField> SumcheckKernel<F> for OptimizedInstructionClaimReductionKernel<F> {
    type Relation = InstructionClaimReduction<F>;

    fn output_claims(
        &mut self,
        _inputs: &SumcheckInputClaims<F, Self::Relation>,
    ) -> Result<InstructionClaimReductionOutputClaims<F>, SumcheckKernelError<F>> {
        let [lookup_output, left_lookup_operand, right_lookup_operand, left_instruction_input, right_instruction_input] =
            self.operand_claims()?;
        Ok(InstructionClaimReductionOutputClaims {
            lookup_output,
            left_lookup_operand,
            right_lookup_operand,
            left_instruction_input,
            right_instruction_input,
        })
    }

    /// Pin the fully-bound Gruen scalar `eq(τ_L, r_L)·eq(τ_H, r_H)` to the
    /// verifier's `derive_output_term(EqSpartan)`, exactly as the naive
    /// tier's materialized eq table is pinned.
    fn validate_derived_tables(
        &self,
        relation: &Self::Relation,
        input_points: &SumcheckInputPoints<F, Self::Relation>,
        output_points: &SumcheckOutputPoints<F, Self::Relation>,
        challenges: &ConcreteSumcheckChallenges<F, Self::Relation>,
    ) -> Result<(), SumcheckKernelError<F>> {
        self.challenges.require_complete()?;
        let id = JoltDerivedId::from(InstructionClaimReductionPublic::EqSpartan);
        pin_derived_term(
            relation,
            id,
            input_points,
            output_points,
            challenges,
            self.half.round_tables().0.current_scalar(),
        )
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use std::collections::BTreeMap;

    use jolt_claims::protocols::jolt::geometry::claim_reductions::instruction::{
        left_instruction_input_reduced, left_lookup_operand_reduced, lookup_output_reduced,
        right_instruction_input_reduced, right_lookup_operand_reduced,
    };
    use jolt_claims::protocols::jolt::relations::claim_reductions::instruction::{
        InstructionClaimReductionChallenges, InstructionClaimReductionInputClaims,
    };
    use jolt_claims::protocols::jolt::{JoltOpeningId, TraceDimensions};
    #[cfg(feature = "akita")]
    use jolt_field::Prime128OffsetA7F7;
    use jolt_field::{Fr, Ring, Zero};
    use jolt_verifier::stages::relations::ConcreteSumcheck;
    use jolt_witness::witnesses::ToField;
    use jolt_witness::TraceBackend;

    use super::*;
    use crate::optimized::parity::{replayed, ExceptionalEq};
    use crate::optimized::registers_read_write::test_support::structured_fixture;
    use crate::reference::views::{dense_view, eq_table};
    use crate::{NaiveSumcheckProver, ReferenceBackend};

    fn opening_ids() -> [JoltOpeningId; NUM_TABLES] {
        [
            lookup_output_reduced(),
            left_lookup_operand_reduced(),
            right_lookup_operand_reduced(),
            left_instruction_input_reduced(),
            right_instruction_input_reduced(),
        ]
    }

    fn splitmix(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Rows no trace produces: uniform full-width limbs, right instruction
    /// inputs of both signs, `i128::MIN` in row 0 and every maximum in row 1.
    fn synthetic_rows(log_t: usize, seed: u64) -> Vec<InstructionOperandRow> {
        let mut state = seed;
        let mut word = move || splitmix(&mut state);
        (0..1usize << log_t)
            .map(|cycle| {
                let mut wide = || (u128::from(word()) << 64) | u128::from(word());
                let (right_lookup, right_input) = (wide(), wide() as i128);
                let row = InstructionOperandRow {
                    lookup_output: LookupOutput(word()),
                    left_lookup_operand: LeftLookupOperand(word()),
                    right_lookup_operand: RightLookupOperand(right_lookup),
                    left_instruction_input: LeftInstructionInput(word()),
                    right_instruction_input: RightInstructionInput(right_input),
                };
                match cycle {
                    0 => InstructionOperandRow {
                        right_instruction_input: RightInstructionInput(i128::MIN),
                        ..row
                    },
                    1 => InstructionOperandRow {
                        lookup_output: LookupOutput(u64::MAX),
                        left_lookup_operand: LeftLookupOperand(u64::MAX),
                        right_lookup_operand: RightLookupOperand(u128::MAX),
                        left_instruction_input: LeftInstructionInput(u64::MAX),
                        right_instruction_input: RightInstructionInput(i128::MAX),
                    },
                    _ => row,
                }
            })
            .collect()
    }

    /// The five reference tables of `rows`, through the witness types' own
    /// field conversions.
    fn operand_tables<F: JoltField>(rows: &[InstructionOperandRow]) -> [Vec<F>; NUM_TABLES] {
        [
            rows.iter()
                .map(|row| row.lookup_output.to_field())
                .collect(),
            rows.iter()
                .map(|row| row.left_lookup_operand.to_field())
                .collect(),
            rows.iter()
                .map(|row| row.right_lookup_operand.to_field())
                .collect(),
            rows.iter()
                .map(|row| row.left_instruction_input.to_field())
                .collect(),
            rows.iter()
                .map(|row| row.right_instruction_input.to_field())
                .collect(),
        ]
    }

    /// `Σ_j eq(τ, j)·Σ_i γ^i·o_i(j)` over reference tables: the true input
    /// claim both kernels start from.
    fn input_claim<F: JoltField>(tables: &[Vec<F>; NUM_TABLES], tau: &[F], gamma: F) -> F {
        let eq = eq_table(tau);
        let mut power = F::one();
        let mut claim = F::zero();
        for table in tables {
            for (weight, value) in eq.iter().zip(table) {
                claim += power * *weight * *value;
            }
            power *= gamma;
        }
        claim
    }

    fn generic_tau<F: JoltField>(log_t: usize, seed: u64) -> Vec<F> {
        (0..log_t as u64)
            .map(|i| F::from_u64(500 + 41 * i + seed))
            .collect()
    }

    fn round_challenges<F: JoltField>(rounds: usize, seed: u64) -> Vec<F> {
        (0..rounds as u64)
            .map(|round| {
                F::from_u64(0xA5A5_1234_DEAD_BEEF ^ round.wrapping_mul(0x2545_F491) ^ seed)
            })
            .collect()
    }

    fn input_claims<F: JoltField>() -> InstructionClaimReductionInputClaims<F> {
        InstructionClaimReductionInputClaims {
            lookup_output: F::zero(),
            left_lookup_operand: F::zero(),
            right_lookup_operand: F::zero(),
            left_instruction_input: F::zero(),
            right_instruction_input: F::zero(),
        }
    }

    /// Drives both kernels through every round on the reference's messages,
    /// asserting equal round polynomials, output claims and derived-table
    /// pins.
    fn assert_lockstep<F: JoltField>(
        reference: &mut dyn SumcheckKernel<F, Relation = InstructionClaimReduction<F>>,
        optimized: &mut dyn SumcheckKernel<F, Relation = InstructionClaimReduction<F>>,
        relation: &InstructionClaimReduction<F>,
        gamma: F,
        input_claim: F,
        challenges: &[F],
    ) {
        let rounds = challenges.len();
        assert_eq!(
            (reference.num_rounds(), optimized.num_rounds()),
            (rounds, rounds)
        );
        let mut previous = input_claim;
        let mut bind = None;
        for (round, &challenge) in challenges.iter().enumerate() {
            let expected = reference.prove_round(bind, round, previous).unwrap();
            let got = optimized.prove_round(bind, round, previous).unwrap();
            assert_eq!(got, expected, "round {round} of {rounds}");
            previous = expected.evaluate(challenge);
            bind = Some(challenge);
        }
        let last = bind.unwrap();
        reference.finish_rounds(last).unwrap();
        optimized.finish_rounds(last).unwrap();
        let claims = input_claims();
        assert_eq!(
            optimized.output_claims(&claims).unwrap(),
            reference.output_claims(&claims).unwrap(),
            "output claims, {rounds} rounds"
        );
        let points = InstructionClaimReductionInputClaims::default();
        let output_points = relation.derive_opening_points(challenges, &points).unwrap();
        let challenges = InstructionClaimReductionChallenges { gamma };
        reference
            .validate_derived_tables(relation, &points, &output_points, &challenges)
            .unwrap();
        optimized
            .validate_derived_tables(relation, &points, &output_points, &challenges)
            .unwrap();
    }

    /// Synthetic rows through the optimized kernel against the naive tier
    /// over their reference tables.
    fn synthetic_case<F: JoltField>(log_t: usize, seed: u64, tau: &[F]) {
        let rows = synthetic_rows(log_t, seed);
        let tables = operand_tables::<F>(&rows);
        let gamma = F::from_u64(0xC0FF_EE11 + seed);
        let relation = InstructionClaimReduction::new(TraceDimensions::new(log_t), tau.to_vec());
        let claims = input_claims();
        let points = InstructionClaimReductionInputClaims::default();
        let challenges = InstructionClaimReductionChallenges { gamma };
        let inputs = ProverInputs {
            relation: &relation,
            claims: &claims,
            points: &points,
            challenges: &challenges,
        };
        let opening_tables = opening_ids()
            .into_iter()
            .zip(&tables)
            .map(|(id, table)| (id, Polynomial::new(table.clone())))
            .collect();
        let derived_tables = BTreeMap::from([(
            JoltDerivedId::from(InstructionClaimReductionPublic::EqSpartan),
            Polynomial::new(eq_table(tau)),
        )]);
        let mut reference = NaiveSumcheckProver::new(
            &inputs,
            opening_tables,
            derived_tables,
            BindingOrder::LowToHigh,
        )
        .unwrap();
        let mut optimized =
            OptimizedInstructionClaimReductionKernel::new(tau, KernelRows::Synthetic(rows), gamma)
                .unwrap();
        assert_lockstep(
            &mut reference,
            &mut optimized,
            &relation,
            gamma,
            input_claim(&tables, tau, gamma),
            &round_challenges::<F>(log_t, seed),
        );
    }

    /// Every low/high split from `log_t = 1` (no high-half round) to 6, with
    /// full-width limbs and both signs, in each field the backends prove over.
    #[test]
    fn synthetic_rows_match_the_naive_tier() {
        for log_t in [1usize, 2, 3, 5, 6] {
            let seed = 7 * log_t as u64;
            synthetic_case::<Fr>(log_t, seed, &generic_tau::<Fr>(log_t, seed));
            #[cfg(feature = "akita")]
            synthetic_case::<Prime128OffsetA7F7>(
                log_t,
                seed,
                &generic_tau::<Prime128OffsetA7F7>(log_t, seed),
            );
        }
    }

    /// Zero and one `τ` coordinates in both halves, single zeros at either
    /// end, and an eq factor that vanishes at the first bind, so the high
    /// half starts from a zero scale.
    #[test]
    fn exceptional_tau_matches_the_naive_tier() {
        let (log_t, seed) = (5, 3);
        let first_bind = round_challenges::<Fr>(log_t, seed)[0];
        for case in ExceptionalEq::ALL {
            synthetic_case::<Fr>(log_t, seed, &case.point(log_t, first_bind));
        }
        let mut tau = generic_tau::<Fr>(log_t, seed);
        tau[0] = Fr::zero();
        tau[log_t - 1] = Fr::zero();
        synthetic_case::<Fr>(log_t, seed, &tau);
    }

    /// The optimized slot over `rows` against the reference slot over
    /// `resident`, from the input claim of `resident`'s oracle tables.
    fn trait_path_case<F: JoltField>(resident: &TraceBackend, rows: &TraceBackend, log_t: usize) {
        let tau = generic_tau::<F>(log_t, 11);
        let gamma = F::from_u64(7177);
        let relation = InstructionClaimReduction::new(TraceDimensions::new(log_t), tau.clone());
        let claims = input_claims();
        let points = InstructionClaimReductionInputClaims::default();
        let challenges = InstructionClaimReductionChallenges { gamma };
        let inputs = || ProverInputs {
            relation: &relation,
            claims: &claims,
            points: &points,
            challenges: &challenges,
        };
        let mut reference =
            <ReferenceBackend as PrepareKernel<F, InstructionClaimReduction<F>>>::prepare(
                &ReferenceBackend,
                &mut ProofSession::default(),
                resident,
                inputs(),
            )
            .unwrap();
        let mut optimized = OptimizedInstructionClaimReduction
            .prepare(&mut ProofSession::default(), rows, inputs())
            .unwrap();
        let tables = opening_ids().map(|id| dense_view::<F>(resident, id).unwrap());
        assert_lockstep(
            reference.as_mut(),
            optimized.as_mut(),
            &relation,
            gamma,
            input_claim(&tables, &tau, gamma),
            &round_challenges::<F>(log_t, 11),
        );
    }

    /// Rows extracted through the trace handle, resident and replayed from
    /// two-row checkpoints: several tasks per pass, a padded suffix of one
    /// row or of whole tasks past the proof prefix, in both fields.
    #[test]
    fn trace_rows_match_the_reference_slot() {
        for (log_t, proof_rows) in [(1usize, 1usize), (3, 7), (6, 63), (6, 37)] {
            structured_fixture(proof_rows).with_plane(log_t, |resident| {
                let replay = replayed(resident, 2, None);
                for rows in [resident, &replay] {
                    trait_path_case::<Fr>(resident, rows, log_t);
                    #[cfg(feature = "akita")]
                    trait_path_case::<Prime128OffsetA7F7>(resident, rows, log_t);
                }
            });
        }
    }

    /// A plane without a handle for the stage's domain fails typed instead
    /// of collecting rows, as do a domain with no cycle variable and replay
    /// failures in the low half's pass (`prepare`) and the high half's pass
    /// (a round, or `finish_rounds` when no high-half round follows).
    #[test]
    fn missing_rows_and_replay_failures_are_typed() {
        let gamma = Fr::from_u64(5);
        assert!(matches!(
            OptimizedInstructionClaimReductionKernel::new(
                &[],
                KernelRows::Synthetic(synthetic_rows(0, 1)),
                gamma
            ),
            Err(KernelError::Unsupported { .. })
        ));
        for log_t in [1usize, 3] {
            structured_fixture((1 << log_t) - 1).with_plane(log_t, |resident| {
                let tau = generic_tau::<Fr>(log_t, 5);
                let relation =
                    InstructionClaimReduction::new(TraceDimensions::new(log_t), tau.clone());
                let claims = input_claims();
                let points = InstructionClaimReductionInputClaims::default();
                let challenges = InstructionClaimReductionChallenges { gamma };
                let prepare = |witness: &TraceBackend| {
                    OptimizedInstructionClaimReduction.prepare(
                        &mut ProofSession::default(),
                        witness,
                        ProverInputs {
                            relation: &relation,
                            claims: &claims,
                            points: &points,
                            challenges: &challenges,
                        },
                    )
                };
                if log_t == 3 {
                    let mut other_domain = replayed(resident, 2, None);
                    other_domain.config.log_t = 2;
                    assert!(matches!(
                        prepare(&other_domain),
                        Err(KernelError::Witness(WitnessError::UnavailableView { .. }))
                    ));
                    other_domain.config.log_t = 4;
                    assert!(matches!(
                        prepare(&other_domain),
                        Err(KernelError::Witness(WitnessError::InvalidDimensions { .. }))
                    ));
                    assert!(matches!(
                        prepare(&replayed(resident, 2, Some(5))),
                        Err(KernelError::Witness(WitnessError::Replay { .. }))
                    ));
                }

                let mut kernel = OptimizedInstructionClaimReductionKernel::new(
                    &tau,
                    KernelRows::from_witness::<Fr>(resident, log_t, "resident rows").unwrap(),
                    gamma,
                )
                .unwrap();
                let failing = replayed(resident, 2, Some(0));
                kernel.rows =
                    KernelRows::from_witness::<Fr>(&failing, log_t, "replayed rows").unwrap();
                let tables = opening_ids().map(|id| dense_view::<Fr>(resident, id).unwrap());
                let mut previous = input_claim(&tables, &tau, gamma);
                let mut bind = None;
                let log_low = tau_halves(&tau).1.len();
                for (round, challenge) in round_challenges::<Fr>(log_low, 5).into_iter().enumerate()
                {
                    previous = kernel
                        .prove_round(bind, round, previous)
                        .unwrap()
                        .evaluate(challenge);
                    bind = Some(challenge);
                }
                let high_half = if log_low < log_t {
                    kernel.prove_round(bind, log_low, previous).map(|_| ())
                } else {
                    kernel.finish_rounds(bind.unwrap())
                };
                assert!(matches!(
                    high_half,
                    Err(SumcheckError::MissingEvaluationSource { .. })
                ));
            });
        }
    }
}
