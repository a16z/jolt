//! Reduces `FusedInc` from the standalone virtualization point to the cycle
//! point shared by stage 6b.

use allocative::Allocative;

use crate::field::JoltField;
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding};
use crate::poly::opening_proof::{
    OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId, BIG_ENDIAN,
    LITTLE_ENDIAN,
};
use crate::poly::unipoly::UniPoly;
use crate::subprotocols::sumcheck_prover::SumcheckInstanceProver;
use crate::subprotocols::sumcheck_verifier::SumcheckInstanceParams;
use crate::transcripts::Transcript;
use crate::zkvm::witness::VirtualPolynomial;

const DEGREE_BOUND: usize = 2;

#[derive(Allocative, Clone)]
pub struct FusedIncClaimReductionParams<F: JoltField> {
    pub log_t: usize,
    pub input_point: OpeningPoint<BIG_ENDIAN, F>,
}

impl<F: JoltField> FusedIncClaimReductionParams<F> {
    pub fn new(trace_len: usize, accumulator: &dyn OpeningAccumulator<F>) -> Self {
        let (input_point, _) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::FusedInc,
            SumcheckId::IncVirtualization,
        );
        Self {
            log_t: trace_len.ilog2() as usize,
            input_point,
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for FusedIncClaimReductionParams<F> {
    fn input_claim(&self, accumulator: &dyn OpeningAccumulator<F>) -> F {
        accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::FusedInc,
                SumcheckId::IncVirtualization,
            )
            .1
    }

    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        self.log_t
    }

    fn normalize_opening_point(&self, challenges: &[F::Challenge]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }

    #[cfg(feature = "zk")]
    fn input_claim_constraint(&self) -> crate::subprotocols::blindfold::InputClaimConstraint {
        unimplemented!("zk x lattice is rejected fail-closed")
    }

    #[cfg(feature = "zk")]
    fn input_constraint_challenge_values(&self, _: &dyn OpeningAccumulator<F>) -> Vec<F> {
        unimplemented!("zk x lattice is rejected fail-closed")
    }

    #[cfg(feature = "zk")]
    fn output_claim_constraint(
        &self,
    ) -> Option<crate::subprotocols::blindfold::OutputClaimConstraint> {
        unimplemented!("zk x lattice is rejected fail-closed")
    }

    #[cfg(feature = "zk")]
    fn output_constraint_challenge_values(&self, _: &[F::Challenge]) -> Vec<F> {
        unimplemented!("zk x lattice is rejected fail-closed")
    }
}

#[derive(Allocative)]
pub struct FusedIncClaimReductionProver<F: JoltField> {
    eq: MultilinearPolynomial<F>,
    fused_inc: MultilinearPolynomial<F>,
    pub params: FusedIncClaimReductionParams<F>,
}

impl<F: JoltField> FusedIncClaimReductionProver<F> {
    pub fn initialize(params: FusedIncClaimReductionParams<F>, fused_inc: Vec<i128>) -> Self {
        let eq = MultilinearPolynomial::from(EqPolynomial::evals(&params.input_point.r));
        Self {
            eq,
            fused_inc: fused_inc.into(),
            params,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for FusedIncClaimReductionProver<F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let half = self.eq.len() / 2;
        let mut evals = [F::zero(); DEGREE_BOUND];
        for index in 0..half {
            let eq = self
                .eq
                .sumcheck_evals_array::<DEGREE_BOUND>(index, BindingOrder::LowToHigh);
            let fused = self
                .fused_inc
                .sumcheck_evals_array::<DEGREE_BOUND>(index, BindingOrder::LowToHigh);
            for evaluation in 0..DEGREE_BOUND {
                evals[evaluation] += eq[evaluation] * fused[evaluation];
            }
        }
        UniPoly::from_evals_and_hint(previous_claim, &evals)
    }

    fn ingest_challenge(&mut self, challenge: F::Challenge, _round: usize) {
        self.eq.bind_parallel(challenge, BindingOrder::LowToHigh);
        self.fused_inc
            .bind_parallel(challenge, BindingOrder::LowToHigh);
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) {
        accumulator.append_virtual(
            VirtualPolynomial::FusedInc,
            SumcheckId::FusedIncClaimReduction,
            self.params.normalize_opening_point(sumcheck_challenges),
            self.fused_inc.final_sumcheck_claim(),
        );
    }
}
