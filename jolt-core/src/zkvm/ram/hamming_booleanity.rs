use std::marker::PhantomData;

use num_traits::Zero;

use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding};
use crate::poly::opening_proof::{
    OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, SumcheckId,
    VerifierOpeningAccumulator, BIG_ENDIAN, LITTLE_ENDIAN,
};
use crate::poly::split_eq_poly::GruenSplitEqPolynomial;
use crate::subprotocols::sumcheck_prover::SumcheckInstanceProver;
use crate::subprotocols::sumcheck_verifier::SumcheckInstanceVerifier;
use crate::transcripts::Transcript;
use crate::utils::math::Math;
use crate::zkvm::dag::state_manager::StateManager;
use crate::zkvm::witness::VirtualPolynomial;
use allocative::Allocative;
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use rayon::prelude::*;

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
    pub fn gen(
        state_manager: &mut StateManager<F, impl CommitmentScheme<Field = F>>,
        opening_accumulator: &ProverOpeningAccumulator<F>,
    ) -> Self {
        let (_, _, trace, _, _) = state_manager.get_prover_data();

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

    #[tracing::instrument(
        skip_all,
        name = "RamHammingBooleanitySumcheckProver::compute_prover_message"
    )]
    fn compute_prover_message(&mut self, _round: usize, previous_claim: F) -> Vec<F> {
        let eq = &self.eq_r_cycle;
        let H = &self.H;

        // Accumulate constant (c0) and quadratic (e) coefficients in unreduced form
        let coeffs_unr: [F::Unreduced<9>; 2] = if eq.E_in_current_len() == 1 {
            (0..eq.len() / 2)
                .into_par_iter()
                .map(|j| {
                    let eq_eval = eq.E_out_current()[j];
                    let h0 = H.get_bound_coeff(2 * j);
                    let h1 = H.get_bound_coeff(2 * j + 1);
                    let delta = h1 - h0;
                    let c0 = h0.square() - h0;
                    let e = delta.square();
                    [
                        eq_eval.mul_unreduced::<9>(c0),
                        eq_eval.mul_unreduced::<9>(e),
                    ]
                })
                .reduce(
                    || [<F as JoltField>::Unreduced::<9>::zero(); 2],
                    |a, b| [a[0] + b[0], a[1] + b[1]],
                )
        } else {
            let num_x_in_bits = eq.E_in_current_len().log_2();
            let chunk_size = 1 << num_x_in_bits;
            let x_bitmask = chunk_size - 1;
            (0..eq.len() / 2)
                .collect::<Vec<_>>()
                .par_chunks(chunk_size)
                .enumerate()
                .map(|(x_out, chunk)| {
                    let E_out_eval = eq.E_out_current()[x_out];
                    let inner_unr: [F::Unreduced<9>; 2] = chunk
                        .par_iter()
                        .map(|j| {
                            let j = *j;
                            let x_in = j & x_bitmask;
                            let E_in_eval = eq.E_in_current()[x_in];
                            let h0 = H.get_bound_coeff(2 * j);
                            let h1 = H.get_bound_coeff(2 * j + 1);
                            let delta = h1 - h0;
                            let c0 = h0.square() - h0;
                            let e = delta.square();
                            [
                                E_in_eval.mul_unreduced::<9>(c0),
                                E_in_eval.mul_unreduced::<9>(e),
                            ]
                        })
                        .reduce(
                            || [<F as JoltField>::Unreduced::<9>::zero(); 2],
                            |a, b| [a[0] + b[0], a[1] + b[1]],
                        );

                    // Reduce inner then scale by E_out in unreduced domain
                    let inner_c0 = F::from_montgomery_reduce(inner_unr[0]);
                    let inner_e = F::from_montgomery_reduce(inner_unr[1]);
                    [
                        E_out_eval.mul_unreduced::<9>(inner_c0),
                        E_out_eval.mul_unreduced::<9>(inner_e),
                    ]
                })
                .reduce(
                    || [<F as JoltField>::Unreduced::<9>::zero(); 2],
                    |a, b| [a[0] + b[0], a[1] + b[1]],
                )
        };

        let c0 = F::from_montgomery_reduce(coeffs_unr[0]);
        let e = F::from_montgomery_reduce(coeffs_unr[1]);
        eq.gruen_evals_deg_3(c0, e, previous_claim).to_vec()
    }

    #[tracing::instrument(skip_all, name = "RamHammingBooleanitySumcheckProver::bind")]
    fn bind(&mut self, r_j: F::Challenge, _round: usize) {
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

    fn input_claim(&self, _accumulator: &VerifierOpeningAccumulator<F>) -> F {
        F::zero()
    }

    fn expected_output_claim(
        &self,
        accumulator: &VerifierOpeningAccumulator<F>,
        sumcheck_challenges: &[F::Challenge],
    ) -> F {
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

        (H_claim.square() - H_claim) * eq
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

fn get_opening_point<F: JoltField>(
    sumcheck_challenges: &[F::Challenge],
) -> OpeningPoint<BIG_ENDIAN, F> {
    OpeningPoint::<LITTLE_ENDIAN, F>::new(sumcheck_challenges.to_vec()).match_endianness()
}
