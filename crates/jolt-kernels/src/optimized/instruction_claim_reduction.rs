//! Optimized instruction claim-reduction (stage 2) kernel.
//!
//! The summand is `eq(τ_low, j) · (o₁ + γ·o₂ + γ²·o₃ + γ³·o₄ + γ⁴·o₅)(j)`
//! over the five instruction-lookup operand tables. The reference tier
//! interprets the relation expression through the naive prover (per-point
//! `BTreeMap` leaf resolution, five dense bound tables, a materialized and
//! bound `T`-sized eq table). This kernel:
//!
//! - binds a single γ-combined table `C(j) = Σ_i γ^i·o_i(j)` — the operands
//!   enter the summand linearly, so every round message over `C` equals the
//!   five-table computation exactly (distributivity; binding is linear), at
//!   a fifth of the table memory;
//! - keeps the per-cycle operands as native scalars (56 bytes per cycle) and
//!   recovers the five individual output claims post hoc: each bound value
//!   is the operand's multilinear evaluation at the bound point, one
//!   split-eq-weighted walk over the native rows (the spartan-outer
//!   `claimed_inputs` technique) — never five bound field tables;
//! - factors the eq weight out via `GruenSplitEqPolynomial` — the round
//!   message is `s(t) = ℓ(t) · Σ_y E_out·E_in · combo(t, y)` at the naive
//!   prover's `t ∈ {0, 1, 2}` sample points, and no eq table is ever
//!   materialized or bound.
//!
//! The bound eq factor is pinned to the verifier's scalar path by
//! [`validate_derived_tables`](crate::SumcheckKernel::validate_derived_tables)
//! (Gruen scalar vs `derive_output_term(EqSpartan)`), exactly as the naive
//! tier's materialized table is.

use jolt_claims::protocols::jolt::relations::claim_reductions::instruction::InstructionClaimReductionOutputClaims;
use jolt_claims::protocols::jolt::{InstructionClaimReductionPublic, JoltDerivedId};
use jolt_field::signed::S256;
use jolt_field::{Accumulator, JoltField, WithAccumulator};
use jolt_poly::{BindingOrder, EqPolynomial, GruenSplitEqPolynomial, Polynomial, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::{
    ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckInputPoints, SumcheckOutputPoints,
};
use jolt_verifier::stages::stage2::instruction_claim_reduction::InstructionClaimReduction;
use jolt_witness::witnesses::{
    LeftInstructionInput, LeftLookupOperand, LookupOutput, RightInstructionInput,
    RightLookupOperand,
};
use jolt_witness::{JoltWitnessPlane, WitnessBundle};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::support::{pin_derived_term, GruenRoundMessage, RoundProgress};
use super::trace_record::{RecordRows, TraceRecord};
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

/// The five reduced tables, in output-claim declaration order.
const NUM_TABLES: usize = 5;

/// One cycle's five reduced instruction operands as native scalars — the
/// compact backing of the γ-combined table build and the post-hoc
/// output-claim walk.
#[derive(Clone, Copy, Debug, PartialEq, Eq, WitnessBundle)]
pub struct InstructionOperandRow {
    pub lookup_output: LookupOutput,
    pub left_lookup_operand: LeftLookupOperand,
    pub right_lookup_operand: RightLookupOperand,
    pub left_instruction_input: LeftInstructionInput,
    pub right_instruction_input: RightInstructionInput,
}

impl InstructionOperandRow {
    /// The row's five operand values as field elements, in output-claim
    /// declaration order — the exact entries the dense reduced tables hold.
    /// Production paths accumulate the native scalars unreduced instead;
    /// the parity tests build their reference tables through this.
    #[cfg(test)]
    #[inline]
    fn field_values<F: JoltField>(&self) -> [F; NUM_TABLES] {
        [
            F::from_u64(self.lookup_output.0),
            F::from_u64(self.left_lookup_operand.0),
            F::from_u128(self.right_lookup_operand.0),
            F::from_u64(self.left_instruction_input.0),
            F::from_i128(self.right_instruction_input.0),
        ]
    }
}

/// Optimized [`PrepareKernel`] implementor for the
/// `instruction_claim_reduction` slot.
pub struct OptimizedInstructionClaimReduction;

impl<F: JoltField> PrepareKernel<F, InstructionClaimReduction<F>>
    for OptimizedInstructionClaimReduction
{
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, InstructionClaimReduction<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = InstructionClaimReduction<F>>>, KernelError<F>>
    {
        let record = TraceRecord::shared(session, witness, inputs.relation.tau_low().len())?;
        Ok(Box::new(OptimizedInstructionClaimReductionKernel::new(
            inputs.relation.tau_low(),
            RecordRows::Record(record),
            inputs.challenges.gamma,
        )?))
    }
}

#[cfg_attr(
    feature = "allocative",
    derive(allocative::Allocative),
    allocative(bound = "F: JoltField")
)]
pub struct OptimizedInstructionClaimReductionKernel<F: JoltField> {
    progress: RoundProgress,
    /// The γ-combined operand table `C(j) = Σ_i γ^i·o_i(j)` — the only bound
    /// table (the summand is linear in the five operands).
    combined: Polynomial<F>,
    /// Native per-cycle operand values, kept for the post-hoc output-claim
    /// walk.
    #[cfg_attr(feature = "allocative", allocative(skip))]
    rows: RecordRows<InstructionOperandRow>,
    gruen: GruenSplitEqPolynomial<F>,
    #[cfg_attr(feature = "allocative", allocative(visit = jolt_poly::visit_scalars))]
    bound_challenges: Vec<F>,
}

impl<F: JoltField> OptimizedInstructionClaimReductionKernel<F> {
    pub(crate) fn new(
        tau_low: &[F],
        rows: RecordRows<InstructionOperandRow>,
        gamma: F,
    ) -> Result<Self, KernelError<F>> {
        let log_t = tau_low.len();
        if rows.len() != 1 << log_t {
            return Err(KernelError::TableSizeMismatch {
                table: "instruction claim-reduction operand rows".to_owned(),
                expected: 1 << log_t,
                got: rows.len(),
            });
        }
        let gamma_sqr = gamma * gamma;
        let gamma_powers = [
            F::one(),
            gamma,
            gamma_sqr,
            gamma_sqr * gamma,
            gamma_sqr * gamma_sqr,
        ];
        // The combine runs on the native scalars through the wide accumulator
        // — one 4×1 fused multiply per u64 limb and a single reduction, no
        // per-operand Montgomery conversion. The wide lanes split limb-wise
        // against `2^64`-shifted coefficient copies, the signed lane folds its
        // sign into the coefficient: exactly `Σ_i γ^i·F(o_i)` by
        // distributivity.
        let shift_64 = F::from_u128(1u128 << 64);
        let right_lookup_hi = gamma_powers[2] * shift_64;
        let right_input_coeffs = {
            let positive = (gamma_powers[4], gamma_powers[4] * shift_64);
            ((-positive.0, -positive.1), positive)
        };
        let combine = |row: &InstructionOperandRow| -> F {
            let mut acc = F::Accumulator::default();
            acc.fmadd_u64(gamma_powers[0], row.lookup_output.0);
            acc.fmadd_u64(gamma_powers[1], row.left_lookup_operand.0);
            let right_lookup = row.right_lookup_operand.0;
            acc.fmadd_u64(gamma_powers[2], right_lookup as u64);
            acc.fmadd_u64(right_lookup_hi, (right_lookup >> 64) as u64);
            acc.fmadd_u64(gamma_powers[3], row.left_instruction_input.0);
            let right_input = row.right_instruction_input.0;
            let magnitude = right_input.unsigned_abs();
            let (lo, hi) = if right_input < 0 {
                right_input_coeffs.0
            } else {
                right_input_coeffs.1
            };
            acc.fmadd_u64(lo, magnitude as u64);
            acc.fmadd_u64(hi, (magnitude >> 64) as u64);
            acc.reduce()
        };
        #[cfg(feature = "parallel")]
        let combined: Vec<F> = (0..rows.len())
            .into_par_iter()
            .map(|t| combine(&rows.row(t)))
            .collect();
        #[cfg(not(feature = "parallel"))]
        let combined: Vec<F> = (0..rows.len()).map(|t| combine(&rows.row(t))).collect();

        Ok(Self {
            progress: RoundProgress::new(log_t),
            combined: Polynomial::new(combined),
            rows,
            gruen: GruenSplitEqPolynomial::new(tau_low, BindingOrder::LowToHigh),
            bound_challenges: Vec::with_capacity(log_t),
        })
    }

    /// The five individual bound operand values: multilinear evaluations of
    /// the native rows at the bound point, one split-eq-weighted walk. The
    /// operands accumulate unreduced (`fmadd_s256`, one Barrett reduce per
    /// `e_hi` block ≡ the same sum mod p) instead of one Montgomery
    /// conversion + field multiply per lane per cycle.
    fn operand_claims(&self) -> [F; NUM_TABLES] {
        let reversed: Vec<F> = self.bound_challenges.iter().rev().copied().collect();
        let split = reversed.len() / 2;
        let (r_hi, r_lo) = reversed.split_at(split);
        let e_hi = EqPolynomial::<F>::evals(r_hi, None);
        let e_lo = EqPolynomial::<F>::evals(r_lo, None);
        let lo_len = e_lo.len();

        let block = |idx_hi: usize| -> [F; NUM_TABLES] {
            let mut sums: [<F as WithAccumulator>::SignedProductAccumulator; NUM_TABLES] =
                Default::default();
            let start = idx_hi * lo_len;
            for (t, &weight) in (start..start + lo_len).zip(&e_lo) {
                let row = self.rows.row(t);
                sums[0].fmadd_s256(weight, &S256::from_u64(row.lookup_output.0));
                sums[1].fmadd_s256(weight, &S256::from_u64(row.left_lookup_operand.0));
                sums[2].fmadd_s256(weight, &S256::from_u128(row.right_lookup_operand.0));
                sums[3].fmadd_s256(weight, &S256::from_u64(row.left_instruction_input.0));
                sums[4].fmadd_s256(weight, &S256::from_i128(row.right_instruction_input.0));
            }
            let e_hi_eval = e_hi[idx_hi];
            sums.map(|sum| e_hi_eval * sum.reduce())
        };
        let merge = |mut left: [F; NUM_TABLES], right: [F; NUM_TABLES]| {
            for (left, right) in left.iter_mut().zip(right) {
                *left += right;
            }
            left
        };

        #[cfg(feature = "parallel")]
        {
            (0..e_hi.len())
                .into_par_iter()
                .map(block)
                .reduce(|| [F::zero(); NUM_TABLES], merge)
        }
        #[cfg(not(feature = "parallel"))]
        {
            (0..e_hi.len())
                .map(block)
                .fold([F::zero(); NUM_TABLES], merge)
        }
    }

    /// `s(t) = ℓ(t) · Σ_y E(y) · combo(t, y)` at `t ∈ {0, 1, 2}`, with the
    /// γ-combination folded before the point interpolation (exact by
    /// linearity of binding).
    fn message(
        &self,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        const POINTS: usize = 3;
        let mut q_evals = self.gruen.par_fold_out_in(
            || [F::zero(); POINTS],
            |acc, row, _x_in, e_in| {
                let evals = self.combined.evals();
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

        self.gruen
            .checked_round_poly(&mut q_evals, previous_claim, round)
    }

    fn bind(&mut self, challenge: F) {
        self.gruen.bind(challenge);
        self.combined
            .bind_with_order(challenge, BindingOrder::LowToHigh);
        self.bound_challenges.push(challenge);
        self.progress.advance();
    }
}

impl<F: JoltField> ProveRounds<F> for OptimizedInstructionClaimReductionKernel<F> {
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

impl<F: JoltField> SumcheckKernel<F> for OptimizedInstructionClaimReductionKernel<F> {
    type Relation = InstructionClaimReduction<F>;

    fn output_claims(
        &mut self,
        _inputs: &SumcheckInputClaims<F, Self::Relation>,
    ) -> Result<InstructionClaimReductionOutputClaims<F>, SumcheckKernelError<F>> {
        self.progress.require_complete()?;
        let [lookup_output, left_lookup_operand, right_lookup_operand, left_instruction_input, right_instruction_input] =
            self.operand_claims();
        Ok(InstructionClaimReductionOutputClaims {
            lookup_output,
            left_lookup_operand,
            right_lookup_operand,
            left_instruction_input,
            right_instruction_input,
        })
    }

    /// Pin the fully-bound Gruen scalar to the verifier's
    /// `derive_output_term(EqSpartan)`, exactly as the naive tier's
    /// materialized eq table is pinned.
    fn validate_derived_tables(
        &self,
        relation: &Self::Relation,
        input_points: &SumcheckInputPoints<F, Self::Relation>,
        output_points: &SumcheckOutputPoints<F, Self::Relation>,
        challenges: &ConcreteSumcheckChallenges<F, Self::Relation>,
    ) -> Result<(), SumcheckKernelError<F>> {
        self.progress.require_complete()?;
        let id = JoltDerivedId::from(InstructionClaimReductionPublic::EqSpartan);
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
mod tests {
    use std::collections::BTreeMap;

    use jolt_claims::protocols::jolt::geometry::claim_reductions::instruction::{
        left_instruction_input_reduced, left_lookup_operand_reduced, lookup_output_reduced,
        right_instruction_input_reduced, right_lookup_operand_reduced,
    };
    use jolt_claims::protocols::jolt::relations::claim_reductions::instruction::{
        InstructionClaimReductionChallenges, InstructionClaimReductionInputClaims,
    };
    use jolt_claims::protocols::jolt::{
        InstructionClaimReductionPublic, JoltDerivedId, TraceDimensions,
    };
    use jolt_field::{Fr, Ring};
    use jolt_poly::{BindingOrder, Polynomial};
    use jolt_sumcheck::ProveRounds;
    use jolt_verifier::stages::relations::ConcreteSumcheck;
    use jolt_verifier::stages::stage2::instruction_claim_reduction::InstructionClaimReduction;

    use jolt_witness::witnesses::{
        LeftInstructionInput, LeftLookupOperand, LookupOutput, RightInstructionInput,
        RightLookupOperand,
    };

    use crate::reference::views::eq_table;
    use crate::{NaiveSumcheckProver, ProverInputs, SumcheckKernel};

    use super::{InstructionOperandRow, OptimizedInstructionClaimReductionKernel};

    fn fr(value: u64) -> Fr {
        Fr::from_u64(value)
    }

    fn challenge(round: usize) -> Fr {
        fr(0xA5A5_1234_DEAD_BEEF ^ (round as u64).wrapping_mul(0x2545_F491_4F6C_DD1D) ^ 0x33)
    }

    fn splitmix(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn assert_parity(log_t: usize, seed: u64) {
        let mut state = seed;
        // Native operand rows (the production shape), including negative
        // right instruction inputs; the reference tables are their exact
        // field images.
        let rows: Vec<InstructionOperandRow> = (0..1usize << log_t)
            .map(|_| InstructionOperandRow {
                lookup_output: LookupOutput(splitmix(&mut state)),
                left_lookup_operand: LeftLookupOperand(splitmix(&mut state)),
                right_lookup_operand: RightLookupOperand(
                    (u128::from(splitmix(&mut state)) << 64) | u128::from(splitmix(&mut state)),
                ),
                left_instruction_input: LeftInstructionInput(splitmix(&mut state)),
                right_instruction_input: RightInstructionInput(
                    i128::from(splitmix(&mut state) as i64) - (1i128 << 64),
                ),
            })
            .collect();
        let tables: Vec<Vec<Fr>> = (0..5)
            .map(|i| rows.iter().map(|row| row.field_values::<Fr>()[i]).collect())
            .collect();
        let tau_low: Vec<Fr> = (0..log_t).map(|i| fr(500 + 41 * i as u64)).collect();
        let gamma = fr(0xC0FF_EE11);
        let relation =
            InstructionClaimReduction::<Fr>::new(TraceDimensions::new(log_t), tau_low.clone());

        let ids = [
            lookup_output_reduced(),
            left_lookup_operand_reduced(),
            right_lookup_operand_reduced(),
            left_instruction_input_reduced(),
            right_instruction_input_reduced(),
        ];
        let opening_tables: BTreeMap<_, _> = ids
            .into_iter()
            .zip(&tables)
            .map(|(id, values)| (id, Polynomial::new(values.clone())))
            .collect();
        let derived_tables = BTreeMap::from([(
            JoltDerivedId::from(InstructionClaimReductionPublic::EqSpartan),
            Polynomial::new(eq_table(&tau_low)),
        )]);
        let input_claims = InstructionClaimReductionInputClaims {
            lookup_output: fr(0),
            left_lookup_operand: fr(0),
            right_lookup_operand: fr(0),
            left_instruction_input: fr(0),
            right_instruction_input: fr(0),
        };
        let input_points = InstructionClaimReductionInputClaims {
            lookup_output: Vec::new(),
            left_lookup_operand: Vec::new(),
            right_lookup_operand: Vec::new(),
            left_instruction_input: Vec::new(),
            right_instruction_input: Vec::new(),
        };
        let challenges = InstructionClaimReductionChallenges { gamma };
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

        let mut optimized = OptimizedInstructionClaimReductionKernel::new(
            &tau_low,
            super::super::trace_record::RecordRows::Collected(rows),
            gamma,
        )
        .unwrap();

        // True input claim: the full hypercube sum of the summand.
        let eq = eq_table(&tau_low);
        let gamma_powers = [
            fr(1),
            gamma,
            gamma * gamma,
            gamma * gamma * gamma,
            gamma * gamma * gamma * gamma,
        ];
        let mut claim = fr(0);
        for j in 0..1usize << log_t {
            let mut combo = fr(0);
            for (table, power) in tables.iter().zip(&gamma_powers) {
                combo += *power * table[j];
            }
            claim += eq[j] * combo;
        }

        let rounds = reference.num_rounds();
        assert_eq!(rounds, optimized.num_rounds());
        for round in 0..rounds {
            let bind = round.checked_sub(1).map(challenge);
            let reference_poly = reference.prove_round(bind, round, claim).unwrap();
            let optimized_poly = optimized.prove_round(bind, round, claim).unwrap();
            assert_eq!(
                reference_poly.coefficients(),
                optimized_poly.coefficients(),
                "round {round} polynomial mismatch (log_t={log_t})"
            );
            claim = reference_poly.evaluate(challenge(round));
        }
        reference.finish_rounds(challenge(rounds - 1)).unwrap();
        optimized.finish_rounds(challenge(rounds - 1)).unwrap();

        let reference_outputs = reference.output_claims(&input_claims).unwrap();
        let optimized_outputs = optimized.output_claims(&input_claims).unwrap();
        assert_eq!(reference_outputs, optimized_outputs);

        let sumcheck_point: Vec<Fr> = (0..rounds).map(challenge).collect();
        let output_points = relation
            .derive_opening_points(&sumcheck_point, &input_points)
            .unwrap();
        optimized
            .validate_derived_tables(&relation, &input_points, &output_points, &challenges)
            .unwrap();
    }

    #[test]
    fn parity_even_log_t() {
        assert_parity(4, 7);
    }

    #[test]
    fn parity_odd_log_t() {
        assert_parity(3, 99);
    }
}
