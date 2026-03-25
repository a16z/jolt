//! Two-phase advice claim reduction (Stage 6 cycle -> Stage 7 address).

use std::cell::RefCell;

use crate::field::JoltField;
use crate::poly::commitment::dory::DoryGlobals;
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::multilinear_polynomial::MultilinearPolynomial;
#[cfg(feature = "zk")]
use crate::poly::opening_proof::OpeningId;
use crate::poly::opening_proof::{
    OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
    VerifierOpeningAccumulator, BIG_ENDIAN, LITTLE_ENDIAN,
};
use crate::poly::unipoly::UniPoly;
#[cfg(feature = "zk")]
use crate::subprotocols::blindfold::{InputClaimConstraint, OutputClaimConstraint, ValueSource};
use crate::subprotocols::sumcheck_prover::SumcheckInstanceProver;
use crate::subprotocols::sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier};
use crate::transcripts::Transcript;
use crate::zkvm::claim_reductions::{
    permute_precommitted_polys, precommitted_eq_evals_with_scaling, precommitted_skip_round_scale,
    PrecomittedParams, PrecomittedProver, PrecommittedClaimReduction, PrecommittedPhase,
    PrecommittedSchedulingReference, TWO_PHASE_DEGREE_BOUND,
};
use allocative::Allocative;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Allocative)]
pub enum AdviceKind {
    Trusted,
    Untrusted,
}

#[derive(Clone, Allocative)]
pub struct AdviceClaimReductionParams<F: JoltField> {
    pub kind: AdviceKind,
    pub phase: PrecommittedPhase,
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
            phase: PrecommittedPhase::CycleVariables,
            precommitted,
            advice_col_vars,
            advice_row_vars,
            r_val,
        }
    }

    pub fn num_address_phase_rounds(&self) -> usize {
        self.precommitted.num_address_phase_rounds()
    }

    pub fn transition_to_address_phase(&mut self) {
        self.phase = PrecommittedPhase::AddressVariables;
    }

    pub fn round_offset(&self, max_num_rounds: usize) -> usize {
        self.precommitted.round_offset(
            self.phase == PrecommittedPhase::CycleVariables,
            max_num_rounds,
        )
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for AdviceClaimReductionParams<F> {
    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        match self.phase {
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
        self.precommitted
            .num_rounds_for_phase(self.phase == PrecommittedPhase::CycleVariables)
    }

    fn normalize_opening_point(&self, challenges: &[F::Challenge]) -> OpeningPoint<BIG_ENDIAN, F> {
        self.precommitted
            .normalize_opening_point(self.phase == PrecommittedPhase::CycleVariables, challenges)
    }

    #[cfg(feature = "zk")]
    fn input_claim_constraint(&self) -> InputClaimConstraint {
        let opening = match self.phase {
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
        match self.phase {
            PrecommittedPhase::CycleVariables => {
                let opening = match self.kind {
                    AdviceKind::Trusted => {
                        OpeningId::TrustedAdvice(SumcheckId::AdviceClaimReductionCyclePhase)
                    }
                    AdviceKind::Untrusted => {
                        OpeningId::UntrustedAdvice(SumcheckId::AdviceClaimReductionCyclePhase)
                    }
                };
                Some(OutputClaimConstraint::direct(opening))
            }
            PrecommittedPhase::AddressVariables => {
                let opening = match self.kind {
                    AdviceKind::Trusted => {
                        OpeningId::TrustedAdvice(SumcheckId::AdviceClaimReduction)
                    }
                    AdviceKind::Untrusted => {
                        OpeningId::UntrustedAdvice(SumcheckId::AdviceClaimReduction)
                    }
                };
                Some(OutputClaimConstraint::linear(vec![(
                    ValueSource::Challenge(0),
                    ValueSource::Opening(opening),
                )]))
            }
        }
    }

    #[cfg(feature = "zk")]
    fn output_constraint_challenge_values(&self, sumcheck_challenges: &[F::Challenge]) -> Vec<F> {
        match self.phase {
            PrecommittedPhase::CycleVariables => vec![],
            PrecommittedPhase::AddressVariables => {
                let opening_point = self.normalize_opening_point(sumcheck_challenges);
                let eq_eval = EqPolynomial::mle(&opening_point.r, &self.r_val.r);
                let scale: F = precommitted_skip_round_scale(&self.precommitted);
                vec![eq_eval * scale]
            }
        }
    }
}

impl<F: JoltField> PrecomittedParams<F> for AdviceClaimReductionParams<F> {
    fn is_cycle_phase(&self) -> bool {
        self.phase == PrecommittedPhase::CycleVariables
    }

    fn is_cycle_phase_round(&self, round: usize) -> bool {
        self.precommitted.is_cycle_phase_round(round)
    }

    fn is_address_phase_round(&self, round: usize) -> bool {
        self.precommitted.is_address_phase_round(round)
    }

    fn cycle_alignment_rounds(&self) -> usize {
        self.precommitted.cycle_alignment_rounds()
    }

    fn address_alignment_rounds(&self) -> usize {
        self.precommitted.address_alignment_rounds()
    }

    fn record_cycle_challenge(&mut self, challenge: F::Challenge) {
        self.precommitted.record_cycle_challenge(challenge);
    }
}

#[derive(Allocative)]
pub struct AdviceClaimReductionProver<F: JoltField> {
    core: PrecomittedProver<F, AdviceClaimReductionParams<F>>,
}

impl<F: JoltField> AdviceClaimReductionProver<F> {
    pub fn params(&self) -> &AdviceClaimReductionParams<F> {
        self.core.params()
    }

    pub fn transition_to_address_phase(&mut self) {
        self.core.params_mut().transition_to_address_phase();
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
            core: PrecomittedProver::new(params, advice_poly, eq_poly),
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for AdviceClaimReductionProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        self.core.params()
    }

    fn round_offset(&self, max_num_rounds: usize) -> usize {
        self.core.params().round_offset(max_num_rounds)
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
        if params.phase == PrecommittedPhase::CycleVariables {
            let c_mid = self.core.cycle_intermediate_claim();

            match params.kind {
                AdviceKind::Trusted => accumulator.append_trusted_advice(
                    SumcheckId::AdviceClaimReductionCyclePhase,
                    OpeningPoint::<BIG_ENDIAN, F>::new(vec![]),
                    c_mid,
                ),
                AdviceKind::Untrusted => accumulator.append_untrusted_advice(
                    SumcheckId::AdviceClaimReductionCyclePhase,
                    OpeningPoint::<BIG_ENDIAN, F>::new(vec![]),
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
    pub params: RefCell<AdviceClaimReductionParams<F>>,
}

impl<F: JoltField> AdviceClaimReductionVerifier<F> {
    pub fn new(
        kind: AdviceKind,
        advice_size_bytes: usize,
        scheduling_reference: PrecommittedSchedulingReference,
        accumulator: &VerifierOpeningAccumulator<F>,
    ) -> Self {
        let params = AdviceClaimReductionParams::new(
            kind,
            advice_size_bytes,
            scheduling_reference,
            accumulator,
        );

        Self {
            params: RefCell::new(params),
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T>
    for AdviceClaimReductionVerifier<F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        unsafe { &*self.params.as_ptr() }
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let params = self.params.borrow();
        match params.phase {
            PrecommittedPhase::CycleVariables => {
                accumulator
                    .get_advice_opening(params.kind, SumcheckId::AdviceClaimReductionCyclePhase)
                    .unwrap_or_else(|| panic!("Cycle phase intermediate claim not found"))
                    .1
            }
            PrecommittedPhase::AddressVariables => {
                let opening_point = params.normalize_opening_point(sumcheck_challenges);
                let advice_claim = accumulator
                    .get_advice_opening(params.kind, SumcheckId::AdviceClaimReduction)
                    .expect("Final advice claim not found")
                    .1;
                let eq_eval = EqPolynomial::mle(&opening_point.r, &params.r_val.r);
                let scale: F = precommitted_skip_round_scale(&params.precommitted);
                advice_claim * eq_eval * scale
            }
        }
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let mut params = self.params.borrow_mut();
        if params.phase == PrecommittedPhase::CycleVariables {
            let opening_point = params.normalize_opening_point(sumcheck_challenges);
            match params.kind {
                AdviceKind::Trusted => accumulator.append_trusted_advice(
                    SumcheckId::AdviceClaimReductionCyclePhase,
                    OpeningPoint::<BIG_ENDIAN, F>::new(vec![]),
                ),
                AdviceKind::Untrusted => accumulator.append_untrusted_advice(
                    SumcheckId::AdviceClaimReductionCyclePhase,
                    OpeningPoint::<BIG_ENDIAN, F>::new(vec![]),
                ),
            }
            let opening_point_le: OpeningPoint<LITTLE_ENDIAN, F> = opening_point.match_endianness();
            params
                .precommitted
                .set_cycle_var_challenges(opening_point_le.r);
        }

        if params.num_address_phase_rounds() == 0
            || params.phase == PrecommittedPhase::AddressVariables
        {
            let opening_point = params.normalize_opening_point(sumcheck_challenges);
            match params.kind {
                AdviceKind::Trusted => accumulator
                    .append_trusted_advice(SumcheckId::AdviceClaimReduction, opening_point),
                AdviceKind::Untrusted => accumulator
                    .append_untrusted_advice(SumcheckId::AdviceClaimReduction, opening_point),
            }
        }
    }

    fn round_offset(&self, max_num_rounds: usize) -> usize {
        let params = self.params.borrow();
        params.round_offset(max_num_rounds)
    }
}
