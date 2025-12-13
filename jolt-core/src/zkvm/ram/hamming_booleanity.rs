use crate::field::JoltField;
use crate::poly::multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding};
use crate::poly::opening_proof::{
    OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
    VerifierOpeningAccumulator, BIG_ENDIAN, LITTLE_ENDIAN,
};
use crate::poly::split_eq_poly::GruenSplitEqPolynomial;
use crate::poly::unipoly::UniPoly;
use crate::subprotocols::sumcheck_claim::{Claim, ClaimExpr, InputOutputClaims, SumcheckFrontend};
use crate::subprotocols::sumcheck_prover::SumcheckInstanceProver;
use crate::subprotocols::sumcheck_verifier::SumcheckInstanceVerifier;
use crate::transcripts::Transcript;
use crate::utils::math::Math;
use crate::zkvm::witness::VirtualPolynomial;
use allocative::Allocative;
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use rayon::prelude::*;
use std::marker::PhantomData;
use tracer::instruction::Cycle;

// RAM Hamming booleanity sumcheck
//
// Proves a zero-check of the form
//   0 = Σ_j eq(r_cycle, j) · (H(j)^2 − H(j))
// where:
// - r_cycle are the time/cycle variables bound in this sumcheck
// - H(j) is an indicator of whether a RAM access occurred at cycle j (1 if address != 0, 0 otherwise)

/// Degree bound of the sumcheck round polynomials in [`HammingBooleanitySumcheckVerifier`].
const DEGREE_BOUND: usize = 3;

#[derive(Allocative)]
pub struct HammingBooleanitySumcheckProver<F: JoltField> {
    eq_r_cycle: GruenSplitEqPolynomial<F>,
    H: MultilinearPolynomial<F>,
    log_T: usize,
}

impl<F: JoltField> HammingBooleanitySumcheckProver<F> {
    #[tracing::instrument(skip_all, name = "RamHammingBooleanitySumcheckProver::gen")]
    pub fn gen(trace: &[Cycle], opening_accumulator: &ProverOpeningAccumulator<F>) -> Self {
        let T = trace.len();
        let log_T = T.log_2();

        let H = trace
            .par_iter()
            .map(|cycle| cycle.ram_access().address() != 0)
            .collect::<Vec<bool>>();
        let H = MultilinearPolynomial::from(H);

        let (r_cycle, _) = opening_accumulator.get_virtual_polynomial_opening(
            VirtualPolynomial::LookupOutput,
            SumcheckId::SpartanOuter,
        );

        let eq_r_cycle = GruenSplitEqPolynomial::new(&r_cycle.r, BindingOrder::LowToHigh);

        Self {
            eq_r_cycle,
            H,
            log_T,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T>
    for HammingBooleanitySumcheckProver<F>
{
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        self.log_T
    }

    fn input_claim(&self, _accumulator: &ProverOpeningAccumulator<F>) -> F {
        F::zero()
    }

    #[tracing::instrument(skip_all, name = "RamHammingBooleanitySumcheckProver::compute_message")]
    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        let eq = &self.eq_r_cycle;
        let H = &self.H;

        // Accumulate constant (c0) and quadratic (e) coefficients via generic split-eq fold.
        let [c0, e] = eq.par_fold_out_in_unreduced::<9, 2>(&|g| {
            let h0 = H.get_bound_coeff(2 * g);
            let h1 = H.get_bound_coeff(2 * g + 1);
            let delta = h1 - h0;
            [h0.square() - h0, delta.square()]
        });
        eq.gruen_poly_deg_3(c0, e, previous_claim)
    }

    #[tracing::instrument(
        skip_all,
        name = "RamHammingBooleanitySumcheckProver::ingest_challenge"
    )]
    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        self.eq_r_cycle.bind(r_j);
        self.H.bind_parallel(r_j, BindingOrder::LowToHigh);
    }

    fn cache_openings(
        &self,
        accumulator: &mut ProverOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::RamHammingWeight,
            SumcheckId::RamHammingBooleanity,
            get_opening_point(sumcheck_challenges),
            self.H.final_sumcheck_claim(),
        );
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

pub struct HammingBooleanitySumcheckVerifier<F: JoltField> {
    log_T: usize,
    _phantom: PhantomData<F>,
}

impl<F: JoltField> HammingBooleanitySumcheckVerifier<F> {
    pub fn new(n_cycle_vars: usize) -> Self {
        Self {
            // TODO: Make the name for this consistent across the codebase.
            log_T: n_cycle_vars,
            _phantom: PhantomData,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T>
    for HammingBooleanitySumcheckVerifier<F>
{
    fn degree(&self) -> usize {
        DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        self.log_T
    }

    fn input_claim(&self, accumulator: &VerifierOpeningAccumulator<F>) -> F {
        let result = Self::input_output_claims().input_claim(&[F::one()], accumulator);

        #[cfg(test)]
        {
            let reference_result = F::zero();
            assert_eq!(result, reference_result);
        }

        result
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
        let r = OpeningPoint::new(
            sumcheck_challenges
                .iter()
                .cloned()
                .rev()
                .collect::<Vec<F::Challenge>>(),
        );
        let result =
            Self::input_output_claims().expected_output_claim(&r, &[F::one()], accumulator);

        #[cfg(test)]
        {
            use crate::poly::eq_poly::EqPolynomial;

            let H_claim = accumulator
                .get_virtual_polynomial_opening(
                    VirtualPolynomial::RamHammingWeight,
                    SumcheckId::RamHammingBooleanity,
                )
                .1;

            let (r_cycle, _) = accumulator.get_virtual_polynomial_opening(
                VirtualPolynomial::LookupOutput,
                SumcheckId::SpartanOuter,
            );

            let eq = EqPolynomial::<F>::mle(
                sumcheck_challenges,
                &r_cycle
                    .r
                    .iter()
                    .cloned()
                    .rev()
                    .collect::<Vec<F::Challenge>>(),
            );

            let reference_result = (H_claim.square() - H_claim) * eq;
            assert_eq!(result, reference_result);
        }

        result
    }

    fn cache_openings(
        &self,
        accumulator: &mut VerifierOpeningAccumulator<F>,
        transcript: &mut T,
        sumcheck_challenges: &[F::Challenge],
    ) {
        accumulator.append_virtual(
            transcript,
            VirtualPolynomial::RamHammingWeight,
            SumcheckId::RamHammingBooleanity,
            get_opening_point(sumcheck_challenges),
        );
    }
}

impl<F: JoltField> SumcheckFrontend<F> for HammingBooleanitySumcheckVerifier<F> {
    fn input_output_claims() -> InputOutputClaims<F> {
        let lookup_output: ClaimExpr<F> = VirtualPolynomial::LookupOutput.into();
        let ram_hamming_weight: ClaimExpr<F> = VirtualPolynomial::RamHammingWeight.into();
        let ram_hamming_weight_squared = ram_hamming_weight.clone() * ram_hamming_weight.clone();

        InputOutputClaims {
            claims: vec![Claim {
                // NOTE: In this case, the input claim is 0, so this is just the sumcheck to
                // take r_cycle from.
                input_sumcheck_id: SumcheckId::SpartanOuter,
                // FIXME: This is a kludge. Should just be 0, but then how do we know which
                // virtual polynomial to use to get the opening from the accumulator?
                input_claim_expr: ClaimExpr::Val(F::zero()) * lookup_output,
                expected_output_claim_expr: ram_hamming_weight_squared - ram_hamming_weight,
                is_offset: false,
            }],
            output_sumcheck_id: SumcheckId::RamHammingBooleanity,
        }
    }
}

fn get_opening_point<F: JoltField>(
    sumcheck_challenges: &[F::Challenge],
) -> OpeningPoint<BIG_ENDIAN, F> {
    OpeningPoint::<LITTLE_ENDIAN, F>::new(sumcheck_challenges.to_vec()).match_endianness()
}
