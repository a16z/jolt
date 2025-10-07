use allocative::Allocative;
use std::cell::RefCell;
use std::rc::Rc;

use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding};
use crate::poly::opening_proof::{
    OpeningPoint, ProverOpeningAccumulator, SumcheckId, VerifierOpeningAccumulator, BIG_ENDIAN,
};
use crate::poly::split_eq_poly::GruenSplitEqPolynomial;
use crate::subprotocols::sumcheck::SumcheckInstance;
use crate::transcripts::Transcript;
use crate::utils::math::Math;
use crate::zkvm::dag::state_manager::StateManager;
use crate::zkvm::r1cs::inputs::generate_product_virtualization_witnesses;
use crate::zkvm::witness::{CommittedPolynomial, VirtualPolynomial};
use rayon::prelude::*;

#[derive(Allocative)]
struct ProductVirtualizationSumcheckProverState<F: JoltField> {
    left_input_poly: MultilinearPolynomial<F>,
    right_input_poly: MultilinearPolynomial<F>,
    eq_r_cycle: GruenSplitEqPolynomial<F>,
}

#[derive(Allocative)]
pub struct ProductVirtualizationSumcheck<F: JoltField> {
    input_claim: F,
    log_T: usize,
    prover_state: Option<ProductVirtualizationSumcheckProverState<F>>,
}

impl<F: JoltField> ProductVirtualizationSumcheck<F> {
    #[tracing::instrument(skip_all, name = "ProductVirtualizationSumcheck::new_prover")]
    pub fn new_prover<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Self {
        let (_, trace, _, _) = state_manager.get_prover_data();

        // Stream once to generate LeftInstructionInput and RightInstructionInput witnesses
        let (left_input_poly, right_input_poly) = generate_product_virtualization_witnesses(trace);

        // Get opening_point and claim from accumulator
        let accumulator = state_manager.get_prover_accumulator();
        let (outer_sumcheck_r, input_claim) = accumulator
            .borrow()
            .get_virtual_polynomial_opening(VirtualPolynomial::Product, SumcheckId::SpartanOuter);

        let (r_cycle, _rx_var) = outer_sumcheck_r.split_at(trace.len().log_2());

        Self {
            input_claim,
            log_T: r_cycle.len(),
            prover_state: Some(ProductVirtualizationSumcheckProverState {
                left_input_poly,
                right_input_poly,
                eq_r_cycle: GruenSplitEqPolynomial::new(&r_cycle.r, BindingOrder::LowToHigh),
            }),
        }
    }

    pub fn new_verifier<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Self {
        let accumulator = state_manager.get_verifier_accumulator();
        // Get the Product claim from the accumulator
        let (_, input_claim) = accumulator
            .borrow()
            .get_virtual_polynomial_opening(VirtualPolynomial::Product, SumcheckId::SpartanOuter);
        let (_, _, T) = state_manager.get_verifier_data();

        Self {
            input_claim,
            prover_state: None,
            log_T: T.log_2(),
        }
    }
}

impl<F: JoltField> SumcheckInstance<F> for ProductVirtualizationSumcheck<F> {
    fn degree(&self) -> usize {
        3
    }

    fn num_rounds(&self) -> usize {
        self.log_T
    }

    fn input_claim(&self) -> F {
        self.input_claim
    }

    #[tracing::instrument(skip_all, name = "PCSumcheck::compute_prover_message")]
    fn compute_prover_message(&mut self, _round: usize, previous_claim: F) -> Vec<F> {
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");
        const DEGREE: usize = 3;

        let eq = &prover_state.eq_r_cycle;
        let left_input_poly = &prover_state.left_input_poly;
        let right_input_poly = &prover_state.right_input_poly;

        let quadratic_coeffs: [F; DEGREE - 1] = if eq.E_in_current_len() == 1 {
            // E_in is fully bound
            (0..eq.len() / 2)
                .into_par_iter()
                .map(|j| {
                    let eq_eval = eq.E_out_current()[j];
                    let left_evals = [
                        left_input_poly.get_bound_coeff(2 * j),
                        left_input_poly.get_bound_coeff(2 * j + 1),
                    ];
                    let right_evals = [
                        right_input_poly.get_bound_coeff(2 * j),
                        right_input_poly.get_bound_coeff(2 * j + 1),
                    ];

                    [
                        eq_eval * left_evals[0] * right_evals[0],
                        eq_eval
                            * (left_evals[1] - left_evals[0])
                            * (right_evals[1] - right_evals[0]),
                    ]
                })
                .reduce(
                    || [F::zero(); DEGREE - 1],
                    |running, new| [running[0] + new[0], running[1] + new[1]],
                )
        } else {
            // E_in has not been fully bound - use nested structure
            let num_x_in_bits = eq.E_in_current_len().log_2();
            let x_bitmask = (1 << num_x_in_bits) - 1;
            let chunk_size = 1 << num_x_in_bits;

            (0..eq.len() / 2)
                .collect::<Vec<_>>()
                .par_chunks(chunk_size)
                .enumerate()
                .map(|(x_out, chunk)| {
                    let E_out_eval = eq.E_out_current()[x_out];

                    let chunk_evals = chunk
                        .par_iter()
                        .map(|j| {
                            let x_in = j & x_bitmask;
                            let E_in_eval = eq.E_in_current()[x_in];
                            let left_evals = [
                                left_input_poly.get_bound_coeff(2 * j),
                                left_input_poly.get_bound_coeff(2 * j + 1),
                            ];
                            let right_evals = [
                                right_input_poly.get_bound_coeff(2 * j),
                                right_input_poly.get_bound_coeff(2 * j + 1),
                            ];

                            // Inner eq contribution
                            [
                                E_in_eval * left_evals[0] * right_evals[0],
                                E_in_eval
                                    * (left_evals[1] - left_evals[0])
                                    * (right_evals[1] - right_evals[0]),
                            ]
                        })
                        .reduce(
                            || [F::zero(); DEGREE - 1],
                            |running, new| [running[0] + new[0], running[1] + new[1]],
                        );

                    // Outer eq contribution
                    [E_out_eval * chunk_evals[0], E_out_eval * chunk_evals[1]]
                })
                .reduce(
                    || [F::zero(); DEGREE - 1],
                    |running, new| [running[0] + new[0], running[1] + new[1]],
                )
        };

        eq.gruen_evals_deg_3(quadratic_coeffs[0], quadratic_coeffs[1], previous_claim)
            .to_vec()
    }

    #[tracing::instrument(skip_all, name = "PCSumcheck::bind")]
    fn bind(&mut self, r_j: F, _round: usize) {
        let prover_state = self
            .prover_state
            .as_mut()
            .expect("Prover state not initialized");

        prover_state.eq_r_cycle.bind(r_j);
        rayon::join(
            || {
                prover_state
                    .left_input_poly
                    .bind_parallel(r_j, BindingOrder::LowToHigh)
            },
            || {
                prover_state
                    .right_input_poly
                    .bind_parallel(r_j, BindingOrder::LowToHigh)
            },
        );
    }

    fn expected_output_claim(
        &self,
        accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
        r: &[F],
    ) -> F {
        let accumulator = accumulator.as_ref().unwrap().borrow();

        // Get r_cycle from the SpartanOuter sumcheck opening point
        let (outer_sumcheck_opening, _) = accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::Product, SumcheckId::SpartanOuter);
        let outer_sumcheck_r = &outer_sumcheck_opening.r;
        let (r_cycle, _) = outer_sumcheck_r.split_at(self.log_T);

        // Get the instruction input evaluations from the accumulator
        let (_, left_input_eval) = accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::LeftInstructionInput,
            SumcheckId::ProductVirtualization,
        );
        let (_, right_input_eval) = accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::RightInstructionInput,
            SumcheckId::ProductVirtualization,
        );

        let eq_eval = EqPolynomial::mle(&self.normalize_opening_point(r).r, r_cycle);
        eq_eval * left_input_eval * right_input_eval
    }

    fn normalize_opening_point(&self, opening_point: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::new(opening_point.iter().rev().copied().collect())
    }

    fn cache_openings_prover(
        &self,
        accumulator: Rc<RefCell<ProverOpeningAccumulator<F>>>,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");

        let left_input_eval = prover_state.left_input_poly.final_sumcheck_claim();
        let right_input_eval = prover_state.right_input_poly.final_sumcheck_claim();

        accumulator.borrow_mut().append_dense(
            vec![
                CommittedPolynomial::LeftInstructionInput,
                CommittedPolynomial::RightInstructionInput,
            ],
            SumcheckId::ProductVirtualization,
            opening_point.r,
            &[left_input_eval, right_input_eval],
        );
    }

    fn cache_openings_verifier(
        &self,
        accumulator: Rc<RefCell<VerifierOpeningAccumulator<F>>>,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        accumulator.borrow_mut().append_dense(
            vec![
                CommittedPolynomial::LeftInstructionInput,
                CommittedPolynomial::RightInstructionInput,
            ],
            SumcheckId::ProductVirtualization,
            opening_point.r,
        );
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut allocative::FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}
