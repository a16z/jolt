//! Stage 2: PackedGtExp claim reduction sumcheck.
//!
//! Reduces PackedGtExp rho/quotient claims from the Stage 1 point r1 to a single
//! opening point r2 using an eq-weighted sumcheck.

use allocative::Allocative;
use rayon::prelude::*;

use crate::field::JoltField;
use crate::poly::{
    multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
    opening_proof::{
        OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
        VerifierOpeningAccumulator, BIG_ENDIAN, LITTLE_ENDIAN,
    },
    unipoly::UniPoly,
};
use crate::subprotocols::{
    sumcheck_prover::SumcheckInstanceProver,
    sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier},
};
use crate::transcripts::Transcript;
use crate::zkvm::witness::VirtualPolynomial;
use crate::{virtual_claims, zkvm::recursion::utils::virtual_polynomial_utils::*};

use super::super::stage1::shift_rho::{eq_lsb_evals, eq_lsb_mle};

#[derive(Allocative, Clone)]
pub struct PackedGtExpClaimReductionParams {
    pub num_vars: usize,
    pub num_claims: usize,
    pub sumcheck_id: SumcheckId,
}

impl PackedGtExpClaimReductionParams {
    pub fn new(num_claims: usize) -> Self {
        Self {
            num_vars: 11,
            num_claims,
            sumcheck_id: SumcheckId::PackedGtExpClaimReduction,
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for PackedGtExpClaimReductionParams {
    fn input_claim(&self, _accumulator: &dyn OpeningAccumulator<F>) -> F {
        F::zero()
    }

    fn degree(&self) -> usize {
        2
    }

    fn num_rounds(&self) -> usize {
        self.num_vars
    }

    fn normalize_opening_point(&self, challenges: &[F::Challenge]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::<LITTLE_ENDIAN, F>::new(challenges.to_vec()).match_endianness()
    }
}

#[derive(Allocative)]
pub struct PackedGtExpClaimReductionProver<F: JoltField, T: Transcript> {
    pub params: PackedGtExpClaimReductionParams,
    pub gamma: F,
    pub round: usize,
    pub eq_poly: MultilinearPolynomial<F>,
    pub polys: Vec<MultilinearPolynomial<F>>,
    pub claimed_values: Vec<F>,
    _marker: std::marker::PhantomData<T>,
}

impl<F: JoltField, T: Transcript> PackedGtExpClaimReductionProver<F, T> {
    pub fn new(
        params: PackedGtExpClaimReductionParams,
        claim_indices: &[usize],
        rho_polys: Vec<MultilinearPolynomial<F>>,
        quotient_polys: Vec<MultilinearPolynomial<F>>,
        accumulator: &ProverOpeningAccumulator<F>,
        transcript: &mut T,
    ) -> Self {
        let gamma: F = transcript.challenge_scalar_optimized::<F>().into();

        let (rho_point, _) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::PackedGtExpRho(claim_indices[0]),
            SumcheckId::PackedGtExp,
        );
        let r1 = rho_point.r;
        let eq_evals = eq_lsb_evals::<F>(&r1);
        let eq_poly = MultilinearPolynomial::from(eq_evals);

        let mut claimed_values = Vec::with_capacity(params.num_claims);
        for idx in claim_indices {
            let (_, rho_claim) = accumulator.get_virtual_polynomial_opening(
                VirtualPolynomial::PackedGtExpRho(*idx),
                SumcheckId::PackedGtExp,
            );
            claimed_values.push(rho_claim);
        }
        for idx in claim_indices {
            let (_, quotient_claim) = accumulator.get_virtual_polynomial_opening(
                VirtualPolynomial::PackedGtExpQuotient(*idx),
                SumcheckId::PackedGtExp,
            );
            claimed_values.push(quotient_claim);
        }

        let mut polys = Vec::with_capacity(params.num_claims);
        polys.extend(rho_polys);
        polys.extend(quotient_polys);

        Self {
            params,
            gamma,
            round: 0,
            eq_poly,
            polys,
            claimed_values,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T>
    for PackedGtExpClaimReductionProver<F, T>
{
    fn degree(&self) -> usize {
        2 // eq * poly
    }

    fn num_rounds(&self) -> usize {
        self.params.num_vars
    }

    fn input_claim(&self, _accumulator: &ProverOpeningAccumulator<F>) -> F {
        let mut sum = F::zero();
        let mut gamma_power = F::one();
        for claimed in &self.claimed_values {
            sum += gamma_power * claimed;
            gamma_power *= self.gamma;
        }
        sum
    }

    #[tracing::instrument(skip_all, name = "PackedGtExpClaimReduction::compute_message")]
    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        const DEGREE: usize = 2;
        let half = if !self.polys.is_empty() {
            self.polys[0].len() / 2
        } else {
            return UniPoly::from_evals_and_hint(previous_claim, &[F::zero(); DEGREE]);
        };

        let gamma = self.gamma;

        let evals = (0..half)
            .into_par_iter()
            .map(|i| {
                let eq_evals = self
                    .eq_poly
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);

                let mut term_evals = [F::zero(); DEGREE];
                let mut gamma_power = F::one();
                for poly in &self.polys {
                    let poly_evals =
                        poly.sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);
                    for t in 0..DEGREE {
                        term_evals[t] += eq_evals[t] * gamma_power * poly_evals[t];
                    }
                    gamma_power *= gamma;
                }

                term_evals
            })
            .reduce(
                || [F::zero(); DEGREE],
                |mut acc, evals| {
                    for (a, e) in acc.iter_mut().zip(evals.iter()) {
                        *a += *e;
                    }
                    acc
                },
            );

        UniPoly::from_evals_and_hint(previous_claim, &evals)
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, round: usize) {
        for poly in &mut self.polys {
            poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
        self.eq_poly.bind_parallel(r_j, BindingOrder::LowToHigh);
        self.round = round + 1;
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = OpeningPoint::<BIG_ENDIAN, F>::new(sumcheck_challenges.to_vec());
        let num_witnesses = self.params.num_claims / 2;

        for w in 0..num_witnesses {
            let rho_eval = self.polys[w].get_bound_coeff(0);
            let quotient_eval = self.polys[w + num_witnesses].get_bound_coeff(0);

            let claims = virtual_claims![
                VirtualPolynomial::PackedGtExpRho(w) => rho_eval,
                VirtualPolynomial::PackedGtExpQuotient(w) => quotient_eval,
            ];
            append_virtual_claims(
                accumulator,
                transcript,
                self.params.sumcheck_id,
                &opening_point,
                &claims,
            );
        }
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut allocative::FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

#[derive(Allocative)]
pub struct PackedGtExpClaimReductionVerifier<F: JoltField> {
    pub params: PackedGtExpClaimReductionParams,
    pub claim_indices: Vec<usize>,
    pub gamma: F,
}

impl<F: JoltField> PackedGtExpClaimReductionVerifier<F> {
    pub fn new<T: Transcript>(
        params: PackedGtExpClaimReductionParams,
        claim_indices: Vec<usize>,
        transcript: &mut T,
    ) -> Self {
        let gamma: F = transcript.challenge_scalar_optimized::<F>().into();
        Self {
            params,
            claim_indices,
            gamma,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T>
    for PackedGtExpClaimReductionVerifier<F>
{
    fn degree(&self) -> usize {
        2
    }

    fn num_rounds(&self) -> usize {
        self.params.num_vars
    }

    fn input_claim(&self, accumulator: &VerifierOpeningAccumulator<F>) -> F {
        let mut sum = F::zero();
        let mut gamma_power = F::one();
        for idx in &self.claim_indices {
            let (_, rho_claim) = accumulator.get_virtual_polynomial_opening(
                VirtualPolynomial::PackedGtExpRho(*idx),
                SumcheckId::PackedGtExp,
            );
            sum += gamma_power * rho_claim;
            gamma_power *= self.gamma;
        }
        for idx in &self.claim_indices {
            let (_, quotient_claim) = accumulator.get_virtual_polynomial_opening(
                VirtualPolynomial::PackedGtExpQuotient(*idx),
                SumcheckId::PackedGtExp,
            );
            sum += gamma_power * quotient_claim;
            gamma_power *= self.gamma;
        }
        sum
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let (rho_point, _) = accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::PackedGtExpRho(self.claim_indices[0]),
            SumcheckId::PackedGtExp,
        );
        let r1 = rho_point.r;
        let eq_val = eq_lsb_mle::<F>(&r1, sumcheck_challenges);

        let mut rho_sum = F::zero();
        let mut gamma_power = F::one();
        for idx in &self.claim_indices {
            let (_, rho_eval) = accumulator.get_virtual_polynomial_opening(
                VirtualPolynomial::PackedGtExpRho(*idx),
                self.params.sumcheck_id,
            );
            rho_sum += gamma_power * rho_eval;
            gamma_power *= self.gamma;
        }
        for idx in &self.claim_indices {
            let (_, quotient_eval) = accumulator.get_virtual_polynomial_opening(
                VirtualPolynomial::PackedGtExpQuotient(*idx),
                self.params.sumcheck_id,
            );
            rho_sum += gamma_power * quotient_eval;
            gamma_power *= self.gamma;
        }

        eq_val * rho_sum
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        let opening_point = OpeningPoint::<BIG_ENDIAN, F>::new(sumcheck_challenges.to_vec());
        for idx in &self.claim_indices {
            let polynomials = vec![
                VirtualPolynomial::PackedGtExpRho(*idx),
                VirtualPolynomial::PackedGtExpQuotient(*idx),
            ];
            append_virtual_openings(
                accumulator,
                transcript,
                self.params.sumcheck_id,
                &opening_point,
                &polynomials,
            );
        }
    }
}
