use std::{cell::RefCell, rc::Rc};

use allocative::Allocative;
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use num_traits::Zero;
use rayon::prelude::*;

use crate::{
    field::JoltField,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_proof::{
            OpeningAccumulator,
            OpeningPoint,
            ProverOpeningAccumulator,
            SumcheckId,
            VerifierOpeningAccumulator,
            BIG_ENDIAN,
        },
        split_eq_poly::GruenSplitEqPolynomial,
    },
    subprotocols::sumcheck::SumcheckInstance,
    transcripts::Transcript,
    utils::math::Math,
    zkvm::{dag::state_manager::StateManager, witness::VirtualPolynomial},
};

// RAM Hamming booleanity sumcheck
//
// Proves a zero-check of the form
//   0 = Σ_j eq(r_cycle, j) · (H(j)^2 − H(j))
// where:
// - r_cycle are the time/cycle variables bound in this sumcheck
// - H(j) is an indicator of whether a RAM access occurred at cycle j (1 if address != 0, 0 otherwise)

const DEGREE: usize = 3;

#[derive(Allocative)]
struct HammingBooleanityProverState<F: JoltField> {
    eq_r_cycle: GruenSplitEqPolynomial<F>,
    H: MultilinearPolynomial<F>,
}

#[derive(Allocative)]
pub struct HammingBooleanitySumcheck<F: JoltField> {
    prover_state: Option<HammingBooleanityProverState<F>>,
    log_T: usize,
}

impl<F: JoltField> HammingBooleanitySumcheck<F> {
    #[tracing::instrument(skip_all, name = "RamHammingBooleanitySumcheck::new_prover")]
    pub fn new_prover(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
    ) -> Self {
        let (_, trace, _, _) = sm.get_prover_data();

        let T = trace.len();
        let log_T = T.log_2();

        let H = trace
            .par_iter()
            .map(|cycle| cycle.ram_access().address() != 0)
            .collect::<Vec<bool>>();
        let H = MultilinearPolynomial::from(H);

        let (r_cycle, _) = sm.get_virtual_polynomial_opening(
            VirtualPolynomial::LookupOutput,
            SumcheckId::SpartanOuter,
        );

        let eq_r_cycle = GruenSplitEqPolynomial::new(&r_cycle.r, BindingOrder::LowToHigh);

        Self {
            prover_state: Some(HammingBooleanityProverState { eq_r_cycle, H }),
            log_T,
        }
    }

    pub fn new_verifier(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
    ) -> Self {
        let (_, _, T) = sm.get_verifier_data();
        let log_T = T.log_2();
        Self {
            prover_state: None,
            log_T,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstance<F, T> for HammingBooleanitySumcheck<F> {
    fn degree(&self) -> usize {
        DEGREE
    }

    fn num_rounds(&self) -> usize {
        self.log_T
    }

    fn input_claim(&self, _acc: Option<&RefCell<dyn OpeningAccumulator<F>>>) -> F {
        F::zero()
    }

    #[tracing::instrument(
        skip_all,
        name = "RamHammingBooleanitySumcheck::compute_prover_message"
    )]
    fn compute_prover_message(&mut self, _round: usize, previous_claim: F) -> Vec<F> {
        let p = self.prover_state.as_ref().unwrap();
        let eq = &p.eq_r_cycle;
        let H = &p.H;

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

    #[tracing::instrument(skip_all, name = "RamHammingBooleanitySumcheck::bind")]
    fn bind(&mut self, r_j: F::Challenge, _round: usize) {
        let ps = self.prover_state.as_mut().unwrap();
        ps.eq_r_cycle.bind(r_j);
        ps.H.bind_parallel(r_j, BindingOrder::LowToHigh);
    }

    fn expected_output_claim(
        &self,
        accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
        r: &[F::Challenge],
    ) -> F {
        let accumulator = accumulator.as_ref().unwrap();
        let H_claim = accumulator
            .borrow()
            .get_virtual_polynomial_opening(
                VirtualPolynomial::RamHammingWeight,
                SumcheckId::RamHammingBooleanity,
            )
            .1;

        let (r_cycle, _) = accumulator.borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::LookupOutput,
            SumcheckId::SpartanOuter,
        );

        let eq = EqPolynomial::<F>::mle(
            r,
            &r_cycle
                .r
                .iter()
                .cloned()
                .rev()
                .collect::<Vec<F::Challenge>>(),
        );

        (H_claim.square() - H_claim) * eq
    }

    fn normalize_opening_point(
        &self,
        opening_point: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        let mut opening_point = opening_point.to_vec();
        opening_point.reverse();
        opening_point.into()
    }

    fn cache_openings_prover(
        &self,
        accumulator: Rc<RefCell<ProverOpeningAccumulator<F>>>,
        transcript: &mut T,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let ps = self.prover_state.as_ref().unwrap();

        let claim = ps.H.final_sumcheck_claim();

        accumulator.borrow_mut().append_virtual(
            transcript,
            VirtualPolynomial::RamHammingWeight,
            SumcheckId::RamHammingBooleanity,
            opening_point.clone(),
            claim,
        );
    }

    fn cache_openings_verifier(
        &self,
        accumulator: Rc<RefCell<VerifierOpeningAccumulator<F>>>,
        transcript: &mut T,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        accumulator.borrow_mut().append_virtual(
            transcript,
            VirtualPolynomial::RamHammingWeight,
            SumcheckId::RamHammingBooleanity,
            opening_point,
        );
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}
