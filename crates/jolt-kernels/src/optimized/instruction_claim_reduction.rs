//! Optimized instruction claim reduction (stage 2).
//!
//! Combines five operands into `C(j) = Σ_i γ^i·o_i(j)` and binds only `C`.
//! Gruen factoring avoids a dense eq table. One split-eq row walk recovers
//! the five output claims at the bound point.

use jolt_claims::protocols::jolt::relations::claim_reductions::instruction::InstructionClaimReductionOutputClaims;
use jolt_claims::protocols::jolt::{InstructionClaimReductionPublic, JoltDerivedId};
use jolt_field::{Accumulator, JoltField};
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
use jolt_witness::{JoltWitnessPlane, WitnessBundle, WitnessError};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::support::{pin_derived_term, BundleStore, GruenRoundMessage, RoundProgress};
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

/// The five reduced tables, in output-claim declaration order.
const NUM_TABLES: usize = 5;

/// One cycle's five reduced instruction operands as native scalars — the
/// compact backing of the γ-combined table build and the post-hoc
/// output-claim walk.
#[derive(Clone, Copy, Debug, WitnessBundle)]
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
        _session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, InstructionClaimReduction<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = InstructionClaimReduction<F>>>, KernelError<F>>
    {
        let rows = BundleStore::resolve(witness, 1usize << inputs.relation.tau_low().len())?;
        Ok(Box::new(OptimizedInstructionClaimReductionKernel::new(
            inputs.relation.tau_low(),
            rows,
            inputs.challenges.gamma,
        )?))
    }
}

/// Coefficients for combining native scalar limbs in one wide accumulation.
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

    #[inline]
    fn combine(&self, row: &InstructionOperandRow) -> F {
        let mut acc = F::Accumulator::default();
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
        acc.reduce()
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
    /// Native rows used to recover individual output claims.
    rows: BundleStore<InstructionOperandRow>,
    gruen: GruenSplitEqPolynomial<F>,
    #[cfg_attr(feature = "allocative", allocative(visit = jolt_poly::visit_scalars))]
    bound_challenges: Vec<F>,
}

impl<F: JoltField> OptimizedInstructionClaimReductionKernel<F> {
    pub(crate) fn new(
        tau_low: &[F],
        rows: BundleStore<InstructionOperandRow>,
        gamma: F,
    ) -> Result<Self, KernelError<F>> {
        let log_t = tau_low.len();
        if let BundleStore::Retained(rows) = &rows {
            if rows.len() != 1 << log_t {
                return Err(KernelError::TableSizeMismatch {
                    table: "instruction claim-reduction operand rows".to_owned(),
                    expected: 1 << log_t,
                    got: rows.len(),
                });
            }
        }
        let coefficients = CombineCoefficients::new(gamma);
        // Build once; rounds only bind this table.
        let combined: Vec<F> = {
            let access = rows.access();
            let coefficients = &coefficients;
            let cell =
                |j: usize| -> Result<F, WitnessError> { Ok(coefficients.combine(&access.row(j)?)) };
            #[cfg(feature = "parallel")]
            {
                (0..1usize << log_t)
                    .into_par_iter()
                    .map(cell)
                    .collect::<Result<_, _>>()?
            }
            #[cfg(not(feature = "parallel"))]
            {
                (0..1usize << log_t).map(cell).collect::<Result<_, _>>()?
            }
        };
        Ok(Self {
            progress: RoundProgress::new(log_t),
            combined: Polynomial::new(combined),
            rows,
            gruen: GruenSplitEqPolynomial::new(tau_low, BindingOrder::LowToHigh),
            bound_challenges: Vec::with_capacity(log_t),
        })
    }

    /// The five individual bound operand values: multilinear evaluations of
    /// the native rows at the bound point, one split-eq-weighted walk.
    fn operand_claims(&self) -> Result<[F; NUM_TABLES], WitnessError> {
        let reversed: Vec<F> = self.bound_challenges.iter().rev().copied().collect();
        let split = reversed.len() / 2;
        let (r_hi, r_lo) = reversed.split_at(split);
        let e_hi = EqPolynomial::<F>::evals(r_hi, None);
        let e_lo = EqPolynomial::<F>::evals(r_lo, None);
        let lo_len = e_lo.len();
        let access = self.rows.access();

        let block = |idx_hi: usize| -> Result<[F; NUM_TABLES], WitnessError> {
            let mut sums = [F::zero(); NUM_TABLES];
            let start = idx_hi * lo_len;
            for (offset, &weight) in e_lo.iter().enumerate() {
                let row = access.row(start + offset)?;
                let values = row.field_values::<F>();
                for (sum, value) in sums.iter_mut().zip(values) {
                    *sum += weight * value;
                }
            }
            let e_hi_eval = e_hi[idx_hi];
            Ok(sums.map(|sum| e_hi_eval * sum))
        };
        let merge = |mut left: [F; NUM_TABLES], right: [F; NUM_TABLES]| {
            for (left, right) in left.iter_mut().zip(right) {
                *left += right;
            }
            left
        };

        #[cfg(feature = "parallel")]
        {
            (0..e_hi.len()).into_par_iter().map(block).try_reduce(
                || [F::zero(); NUM_TABLES],
                |left, right| Ok(merge(left, right)),
            )
        }
        #[cfg(not(feature = "parallel"))]
        {
            let mut folded = [F::zero(); NUM_TABLES];
            for idx_hi in 0..e_hi.len() {
                folded = merge(folded, block(idx_hi)?);
            }
            Ok(folded)
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
        let combined = &self.combined;
        let mut q_evals = self.gruen.par_fold_out_in(
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

        self.gruen
            .checked_round_poly(&mut q_evals, previous_claim, round)
    }

    fn bind(&mut self, challenge: F) {
        self.gruen.bind(challenge);
        // Avoid a fresh half-size table each round.
        self.combined.bind_low_to_high_in_place(challenge);
        if self.combined.capacity() >= 8 * self.combined.len().max(1) {
            self.combined.shrink_to_fit();
        }
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
            self.operand_claims()
                .map_err(|_| SumcheckKernelError::InvariantViolation {
                    reason: "instruction operand walk re-extraction failed after the rounds",
                })?;
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
            crate::optimized::support::BundleStore::Retained(rows),
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
