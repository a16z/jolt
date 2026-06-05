//! Two-phase advice claim reduction (Stage 6 cycle -> Stage 7 address).

use crate::field::JoltField;
use crate::poly::commitment::dory::DoryGlobals;
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::multilinear_polynomial::MultilinearPolynomial;
#[cfg(feature = "zk")]
use crate::poly::opening_proof::OpeningId;
use crate::poly::opening_proof::{
    AbstractVerifierOpeningAccumulator, OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator,
    SumcheckId, BIG_ENDIAN, LITTLE_ENDIAN,
};
use crate::poly::unipoly::UniPoly;
#[cfg(feature = "zk")]
use crate::subprotocols::blindfold::{InputClaimConstraint, OutputClaimConstraint, ValueSource};
use crate::subprotocols::sumcheck_prover::SumcheckInstanceProver;
use crate::subprotocols::sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier};
use crate::zkvm::claim_reductions::{
    permute_precommitted_polys, precommitted_eq_evals_with_scaling, precommitted_skip_round_scale,
    PrecommittedClaimReduction, PrecommittedPhase, PrecommittedSchedulingReference,
    TWO_PHASE_DEGREE_BOUND,
};
use allocative::Allocative;

use super::precommitted::{PrecommittedParams, PrecommittedProver};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Allocative)]
pub enum AdviceKind {
    Trusted,
    Untrusted,
}

#[derive(Clone, Allocative)]
pub struct AdviceClaimReductionParams<F: JoltField> {
    pub kind: AdviceKind,
    pub precommitted: PrecommittedClaimReduction<F>,
    pub advice_col_vars: usize,
    pub advice_row_vars: usize,
    pub r_val: OpeningPoint<BIG_ENDIAN, F>,
}

impl<F: JoltField> AdviceClaimReductionParams<F> {
    pub fn new(
        kind: AdviceKind,
        advice_size_bytes: usize,
        scheduling_reference: PrecommittedSchedulingReference,
        accumulator: &dyn OpeningAccumulator<F>,
    ) -> Self {
        let r_val = accumulator
            .get_advice_opening(kind, SumcheckId::RamValCheck)
            .map(|(p, _)| p)
            .unwrap();

        let (advice_col_vars, advice_row_vars) =
            DoryGlobals::advice_sigma_nu_from_max_bytes(advice_size_bytes);
        let precommitted =
            PrecommittedClaimReduction::new(advice_row_vars, advice_col_vars, scheduling_reference);

        Self {
            kind,
            precommitted,
            advice_col_vars,
            advice_row_vars,
            r_val,
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for AdviceClaimReductionParams<F> {
    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        match self.precommitted.phase {
            PrecommittedPhase::CycleVariables => {
                accumulator
                    .get_advice_opening(self.kind, SumcheckId::RamValCheck)
                    .expect("RamValCheck advice opening missing")
                    .1
            }
            PrecommittedPhase::AddressVariables => {
                accumulator
                    .get_advice_opening(self.kind, SumcheckId::AdviceClaimReductionCyclePhase)
                    .expect("Cycle phase intermediate claim not found")
                    .1
            }
        }
    }

    fn degree(&self) -> usize {
        TWO_PHASE_DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        self.precommitted.num_rounds_for_current_phase()
    }

    fn normalize_opening_point(&self, challenges: &[F::Challenge]) -> OpeningPoint<BIG_ENDIAN, F> {
        self.precommitted.normalize_opening_point(challenges)
    }

    #[cfg(feature = "zk")]
    fn input_claim_constraint(&self) -> InputClaimConstraint {
        let opening = match self.precommitted.phase {
            PrecommittedPhase::CycleVariables => match self.kind {
                AdviceKind::Trusted => OpeningId::TrustedAdvice(SumcheckId::RamValCheck),
                AdviceKind::Untrusted => OpeningId::UntrustedAdvice(SumcheckId::RamValCheck),
            },
            PrecommittedPhase::AddressVariables => match self.kind {
                AdviceKind::Trusted => {
                    OpeningId::TrustedAdvice(SumcheckId::AdviceClaimReductionCyclePhase)
                }
                AdviceKind::Untrusted => {
                    OpeningId::UntrustedAdvice(SumcheckId::AdviceClaimReductionCyclePhase)
                }
            },
        };
        InputClaimConstraint::direct(opening)
    }

    #[cfg(feature = "zk")]
    fn input_constraint_challenge_values(&self, _: &dyn OpeningAccumulator<F>) -> Vec<F> {
        Vec::new()
    }

    #[cfg(feature = "zk")]
    fn output_claim_constraint(&self) -> Option<OutputClaimConstraint> {
        match self.precommitted.phase {
            PrecommittedPhase::CycleVariables => {
                if self.precommitted.num_address_phase_rounds() > 0 {
                    let advice_opening = match self.kind {
                        AdviceKind::Trusted => {
                            OpeningId::TrustedAdvice(SumcheckId::AdviceClaimReductionCyclePhase)
                        }
                        AdviceKind::Untrusted => {
                            OpeningId::UntrustedAdvice(SumcheckId::AdviceClaimReductionCyclePhase)
                        }
                    };
                    return Some(OutputClaimConstraint::direct(advice_opening));
                }
                self.final_advice_output_claim_constraint()
            }
            PrecommittedPhase::AddressVariables => self.final_advice_output_claim_constraint(),
        }
    }

    #[cfg(feature = "zk")]
    fn output_constraint_challenge_values(&self, sumcheck_challenges: &[F::Challenge]) -> Vec<F> {
        match self.precommitted.phase {
            PrecommittedPhase::CycleVariables
                if self.precommitted.num_address_phase_rounds() > 0 =>
            {
                vec![]
            }
            PrecommittedPhase::CycleVariables | PrecommittedPhase::AddressVariables => {
                vec![self.final_advice_output_scale(sumcheck_challenges)]
            }
        }
    }
}

impl<F: JoltField> AdviceClaimReductionParams<F> {
    #[cfg(feature = "zk")]
    fn final_advice_output_claim_constraint(&self) -> Option<OutputClaimConstraint> {
        let advice_opening = match self.kind {
            AdviceKind::Trusted => OpeningId::TrustedAdvice(SumcheckId::AdviceClaimReduction),
            AdviceKind::Untrusted => OpeningId::UntrustedAdvice(SumcheckId::AdviceClaimReduction),
        };
        Some(OutputClaimConstraint::linear(vec![(
            ValueSource::Challenge(0),
            ValueSource::Opening(advice_opening),
        )]))
    }

    fn final_advice_output_scale(&self, sumcheck_challenges: &[F::Challenge]) -> F {
        let eq_eval = self.final_advice_eq_eval(sumcheck_challenges);
        let scale = match self.precommitted.phase {
            PrecommittedPhase::CycleVariables => self.precommitted.cycle_phase_skip_scale(),
            PrecommittedPhase::AddressVariables => {
                precommitted_skip_round_scale(&self.precommitted)
            }
        };
        eq_eval * scale
    }

    fn final_advice_eq_eval(&self, sumcheck_challenges: &[F::Challenge]) -> F {
        let opening_point = OpeningPoint::<BIG_ENDIAN, F>::new(
            self.precommitted
                .poly_opening_round_permutation_be()
                .iter()
                .map(|&global_round| {
                    self.challenge_for_global_round(global_round, sumcheck_challenges)
                })
                .collect(),
        );
        EqPolynomial::mle(&opening_point.r, &self.r_val.r)
    }

    fn challenge_for_global_round(
        &self,
        global_round: usize,
        sumcheck_challenges: &[F::Challenge],
    ) -> F::Challenge {
        let cycle_rounds = self.precommitted.cycle_alignment_rounds();
        if global_round < cycle_rounds {
            return match self.precommitted.phase {
                PrecommittedPhase::CycleVariables => sumcheck_challenges[global_round],
                PrecommittedPhase::AddressVariables => {
                    let idx = self
                        .precommitted
                        .cycle_phase_rounds()
                        .binary_search(&global_round)
                        .expect("cycle round should be active for advice polynomial");
                    self.precommitted.cycle_var_challenges[idx]
                }
            };
        }

        assert_eq!(
            self.precommitted.phase,
            PrecommittedPhase::AddressVariables,
            "cycle-phase final advice scale should not contain address rounds"
        );
        sumcheck_challenges[global_round - cycle_rounds]
    }
}

impl<F: JoltField> PrecommittedParams<F> for AdviceClaimReductionParams<F> {
    fn precommitted(&self) -> &PrecommittedClaimReduction<F> {
        &self.precommitted
    }

    fn precommitted_mut(&mut self) -> &mut PrecommittedClaimReduction<F> {
        &mut self.precommitted
    }

    fn get_cycle_challenges<A: AbstractVerifierOpeningAccumulator<F>>(
        &self,
        accumulator: &A,
    ) -> Vec<F::Challenge> {
        let (cycle_opening_point, _) = accumulator
            .get_advice_opening(self.kind, SumcheckId::AdviceClaimReductionCyclePhase)
            .expect("Cycle phase intermediate claim not found");
        let opening_point_le: OpeningPoint<LITTLE_ENDIAN, F> =
            cycle_opening_point.match_endianness();
        opening_point_le.r
    }
}

#[derive(Allocative)]
pub struct AdviceClaimReductionProver<F: JoltField> {
    core: PrecommittedProver<F, AdviceClaimReductionParams<F>>,
}

impl<F: JoltField> AdviceClaimReductionProver<F> {
    pub fn params(&self) -> &AdviceClaimReductionParams<F> {
        self.core.params()
    }

    pub fn transition_to_address_phase(&mut self) {
        self.core.transition_to_address_phase();
    }

    pub fn initialize(
        params: AdviceClaimReductionParams<F>,
        advice_poly: MultilinearPolynomial<F>,
    ) -> Self {
        let eq_evals =
            precommitted_eq_evals_with_scaling(&params.r_val.r, None, &params.precommitted);
        let (advice_poly, eq_poly): (MultilinearPolynomial<F>, MultilinearPolynomial<F>) = {
            let MultilinearPolynomial::U64Scalars(poly) = advice_poly else {
                panic!("Advice should have u64 coefficients");
            };
            let mut permuted =
                permute_precommitted_polys(vec![poly.coeffs], &params.precommitted).into_iter();
            let advice_poly = permuted
                .next()
                .expect("expected one permuted advice polynomial");
            let eq_poly = eq_evals.into();
            (advice_poly, eq_poly)
        };

        Self {
            core: PrecommittedProver::new(params, advice_poly, eq_poly, None),
        }
    }

    pub fn from_bound_state(
        params: AdviceClaimReductionParams<F>,
        advice_coeffs: Vec<F>,
        eq_coeffs: Vec<F>,
    ) -> Self {
        Self::from_bound_state_with_scale(params, advice_coeffs, eq_coeffs, F::one())
    }

    pub fn from_bound_state_with_scale(
        params: AdviceClaimReductionParams<F>,
        advice_coeffs: Vec<F>,
        eq_coeffs: Vec<F>,
        scale: F,
    ) -> Self {
        let mut prover = Self {
            core: PrecommittedProver::new(
                params,
                MultilinearPolynomial::from(advice_coeffs),
                MultilinearPolynomial::from(eq_coeffs),
                None,
            ),
        };
        prover.core.set_scale(scale);
        prover
    }
}

impl<F: JoltField> SumcheckInstanceProver<F> for AdviceClaimReductionProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        self.core.params()
    }

    fn round_offset(&self, _max_num_rounds: usize) -> usize {
        0
    }

    fn compute_message(&mut self, round: usize, previous_claim: F) -> UniPoly<F> {
        self.core.compute_message(round, previous_claim)
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, round: usize) {
        self.core.ingest_challenge(r_j, round);
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let params = self.core.params();
        let opening_point = params.normalize_opening_point(sumcheck_challenges);
        if params.is_cycle_phase() && params.precommitted.num_address_phase_rounds() > 0 {
            let c_mid = self.core.cycle_intermediate_claim();
            match params.kind {
                AdviceKind::Trusted => accumulator.append_trusted_advice(
                    SumcheckId::AdviceClaimReductionCyclePhase,
                    opening_point.clone(),
                    c_mid,
                ),
                AdviceKind::Untrusted => accumulator.append_untrusted_advice(
                    SumcheckId::AdviceClaimReductionCyclePhase,
                    opening_point.clone(),
                    c_mid,
                ),
            }
        }

        if let Some(advice_claim) = self.core.final_claim_if_ready() {
            match params.kind {
                AdviceKind::Trusted => accumulator.append_trusted_advice(
                    SumcheckId::AdviceClaimReduction,
                    opening_point,
                    advice_claim,
                ),
                AdviceKind::Untrusted => accumulator.append_untrusted_advice(
                    SumcheckId::AdviceClaimReduction,
                    opening_point,
                    advice_claim,
                ),
            }
        }
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut allocative::FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

pub struct AdviceClaimReductionVerifier<F: JoltField> {
    pub params: AdviceClaimReductionParams<F>,
}

impl<F: JoltField> AdviceClaimReductionVerifier<F> {
    pub fn new(
        kind: AdviceKind,
        advice_size_bytes: usize,
        scheduling_reference: PrecommittedSchedulingReference,
        accumulator: &dyn OpeningAccumulator<F>,
    ) -> Self {
        let params = AdviceClaimReductionParams::new(
            kind,
            advice_size_bytes,
            scheduling_reference,
            accumulator,
        );

        Self { params }
    }
}

impl<F: JoltField, A: AbstractVerifierOpeningAccumulator<F>>
    SumcheckInstanceVerifier<F, A> for AdviceClaimReductionVerifier<F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(&self, accumulator: &A, sumcheck_challenges: &[F::Challenge]) -> F {
        let params = &self.params;
        match params.precommitted.phase {
            PrecommittedPhase::CycleVariables
                if params.precommitted.num_address_phase_rounds() > 0 =>
            {
                accumulator
                    .get_advice_opening(params.kind, SumcheckId::AdviceClaimReductionCyclePhase)
                    .unwrap_or_else(|| panic!("Cycle phase intermediate claim not found"))
                    .1
            }
            PrecommittedPhase::CycleVariables | PrecommittedPhase::AddressVariables => {
                let advice_claim = accumulator
                    .get_advice_opening(params.kind, SumcheckId::AdviceClaimReduction)
                    .expect("Final advice claim not found")
                    .1;

                // Account for Phase 1's internal dummy-gap traversal via constant scaling.
                advice_claim * params.final_advice_output_scale(sumcheck_challenges)
            }
        }
    }

    fn cache_openings(&self, accumulator: &mut A, sumcheck_challenges: &[F::Challenge]) {
        let params = &self.params;
        if params.is_cycle_phase() && params.precommitted.num_address_phase_rounds() > 0 {
            let opening_point = params.normalize_opening_point(sumcheck_challenges);
            match params.kind {
                AdviceKind::Trusted => accumulator.append_trusted_advice(
                    SumcheckId::AdviceClaimReductionCyclePhase,
                    opening_point.clone(),
                ),
                AdviceKind::Untrusted => accumulator.append_untrusted_advice(
                    SumcheckId::AdviceClaimReductionCyclePhase,
                    opening_point,
                ),
            }
        }

        if params.precommitted.num_address_phase_rounds() == 0 || !params.is_cycle_phase() {
            let opening_point = params.normalize_opening_point(sumcheck_challenges);
            match params.kind {
                AdviceKind::Trusted => accumulator
                    .append_trusted_advice(SumcheckId::AdviceClaimReduction, opening_point),
                AdviceKind::Untrusted => accumulator
                    .append_untrusted_advice(SumcheckId::AdviceClaimReduction, opening_point),
            }
        }
    }

    fn round_offset(&self, _max_num_rounds: usize) -> usize {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;

    type Challenge = <Fr as JoltField>::Challenge;

    #[test]
    fn final_advice_output_scale_counts_leading_dummy_rounds_when_col_range_is_empty() {
        let challenges = [0, 12, 13]
            .map(|value| Challenge::from(value as u128))
            .to_vec();
        let active_opening_point =
            OpeningPoint::<LITTLE_ENDIAN, Fr>::new(challenges[0..1].to_vec()).match_endianness();
        let scheduling_reference = PrecommittedSchedulingReference {
            main_total_vars: 3,
            reference_total_vars: 3,
            cycle_alignment_rounds: 3,
            address_rounds: 0,
            joint_col_vars: 0,
        };
        let params = AdviceClaimReductionParams {
            kind: AdviceKind::Trusted,
            precommitted: PrecommittedClaimReduction::new(1, 0, scheduling_reference),
            advice_col_vars: 0,
            advice_row_vars: 1,
            r_val: active_opening_point,
        };

        let two_inv = Fr::from_u64(2).inverse().unwrap();

        assert_eq!(
            params.final_advice_output_scale(&challenges),
            two_inv * two_inv
        );
    }

    #[test]
    fn final_advice_output_scale_ignores_unrun_address_dummy_rounds() {
        let challenges = [11, 12]
            .map(|value| Challenge::from(value as u128))
            .to_vec();
        let r_val = OpeningPoint::<BIG_ENDIAN, Fr>::new(vec![Challenge::from(7)]);
        let scheduling_reference = PrecommittedSchedulingReference {
            main_total_vars: 4,
            reference_total_vars: 4,
            cycle_alignment_rounds: 2,
            address_rounds: 2,
            joint_col_vars: 0,
        };
        let params = AdviceClaimReductionParams {
            kind: AdviceKind::Trusted,
            precommitted: PrecommittedClaimReduction::new(1, 0, scheduling_reference),
            advice_col_vars: 0,
            advice_row_vars: 1,
            r_val,
        };

        let two_inv = Fr::from_u64(2).inverse().unwrap();
        let expected_eq = EqPolynomial::mle(&[challenges[0]], &params.r_val.r);

        assert_eq!(
            params.final_advice_output_scale(&challenges),
            expected_eq * two_inv
        );
    }
}
