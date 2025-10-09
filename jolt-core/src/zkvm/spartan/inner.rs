use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use allocative::Allocative;
use tracing::{span, Level};

use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::dense_mlpoly::DensePolynomial;
use crate::poly::multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding};
use crate::poly::opening_proof::{
    OpeningId, OpeningPoint, ProverOpeningAccumulator, SumcheckId, VerifierOpeningAccumulator,
    BIG_ENDIAN,
};
use crate::subprotocols::sumcheck::SumcheckInstance;
use crate::transcripts::Transcript;
use crate::utils::math::Math;
use crate::zkvm::dag::state_manager::StateManager;
use crate::zkvm::r1cs::inputs::{JoltR1CSInputs, ALL_R1CS_INPUTS};
use crate::zkvm::r1cs::key::UniformSpartanKey;
use crate::zkvm::witness::VirtualPolynomial;

use rayon::prelude::*;

#[derive(Allocative)]
struct InnerSumcheckProverState<F: JoltField> {
    poly_abc_small: MultilinearPolynomial<F>,
    poly_z: MultilinearPolynomial<F>,
}

#[derive(Allocative)]
pub struct InnerSumcheck<F: JoltField> {
    input_claim: F,
    prover_state: Option<InnerSumcheckProverState<F>>,
    #[allocative(skip)]
    key: Option<Arc<UniformSpartanKey<F>>>,
    gamma: F,
}

impl<F: JoltField> InnerSumcheck<F> {
    #[tracing::instrument(skip_all, name = "InnerSumcheck::new_prover")]
    pub fn new_prover<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
        key: Arc<UniformSpartanKey<F>>,
    ) -> Self {
        let num_vars_uniform = key.num_vars_uniform_padded();
        let num_cycles = key.num_steps;
        let num_cycles_bits = num_cycles.ilog2() as usize;

        // Get gamma challenge for batching
        let gamma: F = state_manager.transcript.borrow_mut().challenge_scalar();

        // Get opening_point and claims from accumulator (Az, Bz, Cz all have the same point)
        let (outer_sumcheck_r, claim_Az) = state_manager
            .get_virtual_polynomial_opening(VirtualPolynomial::SpartanAz, SumcheckId::SpartanOuter);
        let (_, claim_Bz) = state_manager
            .get_virtual_polynomial_opening(VirtualPolynomial::SpartanBz, SumcheckId::SpartanOuter);
        let (_, claim_Cz) = state_manager
            .get_virtual_polynomial_opening(VirtualPolynomial::SpartanCz, SumcheckId::SpartanOuter);

        let (_r_cycle, rx_var) = outer_sumcheck_r.r.split_at(num_cycles_bits);

        let claim_inner_joint = claim_Az + gamma * claim_Bz + gamma.square() * claim_Cz;

        // Evaluate A_small, B_small, C_small combined with RLC at point rx_var
        let poly_abc_small = DensePolynomial::new(key.evaluate_small_matrix_rlc(rx_var, gamma));

        let span = span!(Level::INFO, "binding_z_second_sumcheck");
        let _guard = span.enter();

        // Bind witness polynomials z at point r_cycle
        let mut bind_z = vec![F::zero(); num_vars_uniform];

        ALL_R1CS_INPUTS
            .into_iter()
            .zip(bind_z.iter_mut())
            .for_each(|(r1cs_input, dest)| {
                let accumulator = state_manager.get_prover_accumulator();
                let accumulator = accumulator.borrow();
                let key = OpeningId::from(&r1cs_input);
                let (_, claim) = accumulator
                    .openings
                    .get(&key)
                    .expect("Missing opening claim for expected OpeningId in bind_z");
                *dest = *claim;
            });

        // Set the constant value at the appropriate position
        let const_col = JoltR1CSInputs::num_inputs();
        if const_col < num_vars_uniform {
            bind_z[const_col] = F::one();
        }

        drop(_guard);
        drop(span);

        let poly_z = DensePolynomial::new(bind_z);
        assert_eq!(poly_z.len(), poly_abc_small.len());

        Self {
            input_claim: claim_inner_joint,
            prover_state: Some(InnerSumcheckProverState {
                poly_abc_small: MultilinearPolynomial::LargeScalars(poly_abc_small),
                poly_z: MultilinearPolynomial::LargeScalars(poly_z),
            }),
            key: None,
            gamma,
        }
    }

    pub fn new_verifier<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
        key: Arc<UniformSpartanKey<F>>,
    ) -> Self {
        // Pull the outer sumcheck claims from the accumulator
        let accumulator = state_manager.get_verifier_accumulator();
        let accumulator_ref = accumulator.borrow();
        let (_, claim_Az) = accumulator_ref
            .get_virtual_polynomial_opening(VirtualPolynomial::SpartanAz, SumcheckId::SpartanOuter);
        let (_, claim_Bz) = accumulator_ref
            .get_virtual_polynomial_opening(VirtualPolynomial::SpartanBz, SumcheckId::SpartanOuter);
        let (_, claim_Cz) = accumulator_ref
            .get_virtual_polynomial_opening(VirtualPolynomial::SpartanCz, SumcheckId::SpartanOuter);
        drop(accumulator_ref);

        // Get gamma challenge for batching
        let gamma: F = state_manager.transcript.borrow_mut().challenge_scalar();

        // Compute joint claim
        let input_claim = claim_Az + gamma * claim_Bz + gamma.square() * claim_Cz;

        Self {
            input_claim,
            prover_state: None,
            key: Some(key),
            gamma,
        }
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstance<F, T> for InnerSumcheck<F> {
    fn degree(&self) -> usize {
        2
    }

    fn num_rounds(&self) -> usize {
        if let Some(prover_state) = &self.prover_state {
            prover_state.poly_abc_small.original_len().log_2()
        } else if let Some(key) = &self.key {
            key.num_vars_uniform_padded().log_2()
        } else {
            panic!("Neither prover state nor key is initialized");
        }
    }

    fn input_claim(&self) -> F {
        self.input_claim
    }

    #[tracing::instrument(skip_all, name = "InnerSumcheck::compute_prover_message")]
    fn compute_prover_message(&mut self, _round: usize, _previous_claim: F) -> Vec<F> {
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");
        const DEGREE: usize = 2;

        let univariate_poly_evals: [F; DEGREE] = (0..prover_state.poly_abc_small.len() / 2)
            .into_par_iter()
            .map(|i| {
                let abc_evals = prover_state
                    .poly_abc_small
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::HighToLow);
                let z_evals = prover_state
                    .poly_z
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::HighToLow);

                [
                    abc_evals[0] * z_evals[0], // eval at 0
                    abc_evals[1] * z_evals[1], // eval at 2
                ]
            })
            .reduce(
                || [F::zero(); DEGREE],
                |mut running, new| {
                    for i in 0..DEGREE {
                        running[i] += new[i];
                    }
                    running
                },
            );

        univariate_poly_evals.into()
    }

    #[tracing::instrument(skip_all, name = "InnerSumcheck::bind")]
    fn bind(&mut self, r_j: F::Challenge, _round: usize) {
        let prover_state = self
            .prover_state
            .as_mut()
            .expect("Prover state not initialized");

        // Bind both polynomials in parallel
        rayon::join(
            || {
                prover_state
                    .poly_abc_small
                    .bind_parallel(r_j, BindingOrder::HighToLow)
            },
            || {
                prover_state
                    .poly_z
                    .bind_parallel(r_j, BindingOrder::HighToLow)
            },
        );
    }

    fn expected_output_claim(
        &self,
        accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
        r: &[F::Challenge],
    ) -> F {
        let key = self.key.as_ref().expect("Key not initialized");

        let accumulator = accumulator.as_ref().unwrap().borrow();

        // Get rx_var from the outer sumcheck opening point in accumulator
        let (outer_sumcheck_opening, _) = accumulator
            .get_virtual_polynomial_opening(VirtualPolynomial::SpartanAz, SumcheckId::SpartanOuter);
        let num_cycles_bits = key.num_steps.log_2();
        let (_r_cycle, rx_var) = outer_sumcheck_opening.r.split_at(num_cycles_bits);

        // Pull claimed witness evaluations from the accumulator
        let claimed_witness_evals: Vec<F> = ALL_R1CS_INPUTS
            .into_iter()
            .map(|r1cs_input| {
                let (_, claim) = accumulator
                    .openings
                    .get(&OpeningId::from(&r1cs_input))
                    .unwrap();
                *claim
            })
            .collect();

        // The verifier needs to compute:
        // (A_small(rx, ry) + r * B_small(rx, ry) + r^2 * C_small(rx, ry)) * z(ry)

        // Evaluate uniform matrices A_small, B_small, C_small at point (rx_var, ry_var)
        let eval_a = key.evaluate_uniform_a_at_point(rx_var, r);
        let eval_b = key.evaluate_uniform_b_at_point(rx_var, r);
        let eval_c = key.evaluate_uniform_c_at_point(rx_var, r);

        let left_expected = eval_a + self.gamma * eval_b + self.gamma.square() * eval_c;

        // Evaluate z(ry)
        let eval_z = key.evaluate_z_mle_with_segment_evals(&claimed_witness_evals, r, true);
        left_expected * eval_z
    }

    fn normalize_opening_point(
        &self,
        opening_point: &[F::Challenge],
    ) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::new(opening_point.to_vec())
    }

    fn cache_openings_prover(
        &self,
        _accumulator: Rc<RefCell<ProverOpeningAccumulator<F>>>,
        _transcript: &mut T,
        _opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        // Nothing to cache
    }

    fn cache_openings_verifier(
        &self,
        _accumulator: Rc<RefCell<VerifierOpeningAccumulator<F>>>,
        _transcript: &mut T,
        _opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        // Nothing to cache
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut allocative::FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}
