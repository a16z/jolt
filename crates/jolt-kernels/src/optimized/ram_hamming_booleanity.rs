//! Optimized RAM Hamming-weight booleanity (stage 6b) kernel, porting the
//! legacy `HammingBooleanitySumcheckProver`.
//!
//! The summand is `eq(r_cycle, j) · (H(j)² − H(j))` with `H` the RAM
//! Hamming-weight indicator. Ported techniques:
//!
//! - **Split-eq / Gruen round messages.** The eq factor lives in a
//!   [`GruenSplitEqPolynomial`] (never a dense bound table); per round only
//!   the inner quadratic's constant (`h₀² − h₀`) and leading (`(h₁ − h₀)²`)
//!   coefficients are accumulated and the cubic is reconstructed from
//!   `s(0)+s(1) = previous_claim` — two point-evaluations per pair instead
//!   of the naive tier's four full-summand evaluations plus an eq-table
//!   bind.
//! - **In-place parallel binding** of the single dense `H` table
//!   (`Polynomial::bind_with_order`, rayon inside).
//!
//! Byte parity with the reference kernel holds because field arithmetic is
//! exact: the Gruen-reconstructed evaluations equal the true round
//! polynomial's, and both sides interpolate the same four points.

use jolt_claims::protocols::jolt::geometry::ram::ram_hamming_weight;
use jolt_claims::protocols::jolt::{JoltDerivedId, RamHammingBooleanityPublic};
use jolt_claims::NoChallenges;
use jolt_field::JoltField;
use jolt_poly::{BindingOrder, GruenSplitEqPolynomial, Polynomial, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::ConcreteSumcheck;
use jolt_verifier::stages::relations::{
    SumcheckInputClaims, SumcheckInputPoints, SumcheckOutputPoints,
};
use jolt_verifier::stages::stage6b::ram_hamming_booleanity::{
    RamHammingBooleanity, RamHammingBooleanityOutputClaims,
};
use jolt_witness::JoltWitnessPlane;

use super::support::{pin_derived_term_if_derived, RoundProgress};
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

/// Slot front for the stage-6b RAM Hamming-weight booleanity member.
pub struct OptimizedRamHammingBooleanity;

impl<F: JoltField> PrepareKernel<F, RamHammingBooleanity<F>> for OptimizedRamHammingBooleanity {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, RamHammingBooleanity<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = RamHammingBooleanity<F>>>, KernelError<F>>
    {
        let relation = inputs.relation;
        let trace_dimensions = relation.trace_dimensions();
        let stage1_cycle_binding = relation.stage1_cycle_binding();
        if stage1_cycle_binding.len() != trace_dimensions.log_t() {
            return Err(KernelError::InvariantViolation {
                reason: "stage-1 cycle binding has the wrong variable count",
            });
        }
        // The Hamming indicator is cycle-indexed and dense — `oracle_table`
        // is the right access here (no one-hot grid behind it).
        let opening = ram_hamming_weight();
        let hamming = witness.oracle_table(opening.polynomial_id())?;
        let cycles = 1usize << trace_dimensions.log_t();
        if hamming.len() != cycles {
            return Err(KernelError::TableSizeMismatch {
                table: format!("{opening:?}"),
                expected: cycles,
                got: hamming.len(),
            });
        }
        // The verifier's `derive_output_term` pairs the raw sumcheck point
        // against the stage-1 binding positionally, so the eq table's
        // big-endian point is the binding reversed — same orientation as the
        // reference's derived table.
        let eq_point: Vec<F> = stage1_cycle_binding.iter().rev().copied().collect();

        Ok(Box::new(OptimizedRamHammingBooleanityKernel {
            progress: RoundProgress::new(relation.rounds()),
            eq: GruenSplitEqPolynomial::new(&eq_point, BindingOrder::LowToHigh),
            hamming: Polynomial::new(hamming),
        }))
    }
}

#[cfg_attr(
    feature = "allocative",
    derive(allocative::Allocative),
    allocative(bound = "F: JoltField")
)]
struct OptimizedRamHammingBooleanityKernel<F: JoltField> {
    progress: RoundProgress,
    eq: GruenSplitEqPolynomial<F>,
    hamming: Polynomial<F>,
}
impl<F: JoltField> OptimizedRamHammingBooleanityKernel<F> {
    fn bind(&mut self, challenge: F) {
        self.eq.bind(challenge);
        self.hamming
            .bind_with_order(challenge, BindingOrder::LowToHigh);
        self.progress.advance();
    }
}

impl<F: JoltField> ProveRounds<F> for OptimizedRamHammingBooleanityKernel<F> {
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
        let hamming = &self.hamming;
        let [constant, leading] = self.eq.par_fold_out_in(
            || [F::zero(); 2],
            |accumulator, row, _x_in, e_in| {
                let (h_0, h_1) = hamming.sumcheck_eval_pair(row, BindingOrder::LowToHigh);
                let delta = h_1 - h_0;
                accumulator[0] += e_in * (h_0 * h_0 - h_0);
                accumulator[1] += e_in * (delta * delta);
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

impl<F: JoltField> SumcheckKernel<F> for OptimizedRamHammingBooleanityKernel<F> {
    type Relation = RamHammingBooleanity<F>;

    fn output_claims(
        &mut self,
        _inputs: &SumcheckInputClaims<F, Self::Relation>,
    ) -> Result<RamHammingBooleanityOutputClaims<F>, SumcheckKernelError<F>> {
        self.progress.require_complete()?;
        Ok(RamHammingBooleanityOutputClaims {
            ram_hamming_weight: self.hamming.evals()[0],
        })
    }

    /// The split-eq scalar (fully bound `EqCycle`) against the verifier's
    /// `derive_output_term` — the same drift detector the naive tier runs on
    /// its hand-materialized derived table.
    fn validate_derived_tables(
        &self,
        relation: &Self::Relation,
        input_points: &SumcheckInputPoints<F, Self::Relation>,
        output_points: &SumcheckOutputPoints<F, Self::Relation>,
        challenges: &NoChallenges<F>,
    ) -> Result<(), SumcheckKernelError<F>> {
        self.progress.require_complete()?;
        pin_derived_term_if_derived(
            relation,
            JoltDerivedId::from(RamHammingBooleanityPublic::EqCycle),
            input_points,
            output_points,
            challenges,
            self.eq.current_scalar(),
        )
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use jolt_claims::protocols::jolt::geometry::dimensions::TraceDimensions;
    use jolt_field::{Fr, Ring};

    use super::*;
    use crate::optimized::booleanity::testing::{test_challenge, with_booleanity_backend};
    use crate::ReferenceBackend;
    use jolt_verifier::stages::stage6b::ram_hamming_booleanity::RamHammingBooleanityInputClaims;

    /// Lockstep parity drive against the reference kernel: identical round
    /// polynomials every round, identical output claims, and the split-eq
    /// scalar passing the verifier's derived-term cross-check. The fixture's
    /// rows mix RAM and non-RAM cycles, so the Hamming column is a genuine
    /// 0/1 mix (input claim exactly zero).
    fn parity(log_t: usize) {
        with_booleanity_backend(log_t, 4, |backend, _| {
            let stage1_cycle_binding: Vec<Fr> = (0..log_t as u64)
                .map(|index| Fr::from_u64(600 + 41 * index))
                .collect();
            let relation =
                RamHammingBooleanity::new(TraceDimensions::new(log_t), stage1_cycle_binding);
            let claims = RamHammingBooleanityInputClaims::default();
            let points = RamHammingBooleanityInputClaims::default();
            let challenges = NoChallenges::default();

            let mut reference = ReferenceBackend
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
            let mut optimized = OptimizedRamHammingBooleanity
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

            // The Hamming indicator is boolean, so the input claim is zero.
            let mut claim = Fr::from_u64(0);
            let mut bind = None;
            let mut drawn = Vec::new();
            for round in 0..reference.num_rounds() {
                let expected = reference.prove_round(bind, round, claim).unwrap();
                let actual = optimized.prove_round(bind, round, claim).unwrap();
                assert_eq!(expected, actual, "round {round} polynomial mismatch");
                let challenge = test_challenge(round);
                claim = expected.evaluate(challenge);
                drawn.push(challenge);
                bind = Some(challenge);
            }
            if let Some(last) = drawn.last() {
                reference.finish_rounds(*last).unwrap();
                optimized.finish_rounds(*last).unwrap();
            }

            assert_eq!(
                reference.output_claims(&claims).unwrap(),
                optimized.output_claims(&claims).unwrap()
            );
            let output_points = relation.derive_opening_points(&drawn, &points).unwrap();
            reference
                .validate_derived_tables(&relation, &points, &output_points, &challenges)
                .unwrap();
            optimized
                .validate_derived_tables(&relation, &points, &output_points, &challenges)
                .unwrap();
        });
    }

    #[test]
    fn matches_reference() {
        parity(2);
    }

    #[test]
    fn matches_reference_single_round() {
        parity(1);
    }

    #[test]
    fn matches_reference_with_padding_rows() {
        parity(3);
    }
}
