use std::{cell::RefCell, rc::Rc};

use crate::{
    field::JoltField,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        eq_poly::EqPolynomial,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        opening_proof::{
            OpeningPoint, ProverOpeningAccumulator, SumcheckId, VerifierOpeningAccumulator,
            BIG_ENDIAN,
        },
    },
    subprotocols::sumcheck::SumcheckInstance,
    transcripts::Transcript,
    utils::{math::Math, thread::unsafe_allocate_zero_vec},
    zkvm::{
        dag::state_manager::StateManager,
        ram::remap_address,
        witness::{CommittedPolynomial, VirtualPolynomial},
    },
};
use allocative::Allocative;
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use rayon::prelude::*;

#[derive(Allocative)]
pub struct ValEvaluationProverState<F: JoltField> {
    inc: MultilinearPolynomial<F>,
    wa: MultilinearPolynomial<F>,
    lt: MultilinearPolynomial<F>,
}

/// Val-evaluation sumcheck for RAM
#[derive(Allocative)]
pub struct ValEvaluationSumcheck<F: JoltField> {
    /// Initial claim value
    claimed_evaluation: F,
    /// Initial evaluation to subtract (for RAM)
    init_eval: F,
    /// log T
    num_rounds: usize,
    /// used to compute LT evaluation
    prover_state: Option<ValEvaluationProverState<F>>,
    /// ram K parameter
    K: usize,
}

impl<F: JoltField> ValEvaluationSumcheck<F> {
    #[tracing::instrument(skip_all, name = "RamValEvaluationSumcheck::new_prover")]
    pub fn new_prover<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        initial_ram_state: &[u64],
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Self {
        let K = state_manager.ram_K;

        let (r, claimed_evaluation) = state_manager.get_virtual_polynomial_opening(
            VirtualPolynomial::RamVal,
            SumcheckId::RamReadWriteChecking,
        );
        println!("Prover's private_input_len: {}", state_manager.get_prover_data().2.private_inputs.len());
        let (r_address, r_cycle) = r.split_at(K.log_2());
        
        // Compute the evaluation of private input polynomial at r_address
        let private_input_eval = if let Some(ref prover_state) = state_manager.prover_state {
            if let Some(ref private_input_poly) = prover_state.private_input_polynomial {
                tracing::info!(
                    "Evaluating private input polynomial: poly has {} vars, r_address has {} vars",
                    private_input_poly.get_num_vars(),
                    r_address.r.len()
                );
                let eval = private_input_poly.evaluate(&r_address.r);
                tracing::info!("Private input evaluation at r_address: {:?}", eval);
                Some(eval)
            } else {
                None
            }
        } else {
            None
        };
        
        if let Some(eval) = private_input_eval {
            state_manager.private_input_evaluation = Some(eval);
            
            // Generate the opening proof for private input
            if let Some(ref prover_state) = state_manager.prover_state {
                if let Some(ref private_input_poly) = prover_state.private_input_polynomial {
                    if let Some(ref hint) = prover_state.private_input_hint {
                        let (preprocessing, _, _, _) = state_manager.get_prover_data();
                        let mut transcript_clone = state_manager.get_transcript().borrow().clone();
                        
                        // Use the existing generators that are already properly configured for Dory
                        let proof = PCS::prove(
                            &preprocessing.generators,
                            private_input_poly,
                            &r_address.r,
                            hint.clone(),
                            &mut transcript_clone,
                        );
                        
                        state_manager.private_input_proof = Some(proof);
                        tracing::info!("Generated private input opening proof");
                    }
                }
            }
        }
        
        let (preprocessing, trace, program_io, _) = state_manager.get_prover_data();
        let memory_layout = &program_io.memory_layout;
        let T = trace.len();

        let val_init: MultilinearPolynomial<F> =
            MultilinearPolynomial::from(initial_ram_state.to_vec());
        // println!("initial_ram_state prover: {:?}", initial_ram_state.to_vec());
        let init_eval = val_init.evaluate(&r_address.r);
        println!("Val init is: {}", init_eval);

        // Compute the size-K table storing all eq(r_address, k) evaluations for
        // k \in {0, 1}^log(K)
        let eq_r_address = EqPolynomial::evals(&r_address.r);

        let span = tracing::span!(tracing::Level::INFO, "compute wa(r_address, j)");
        let _guard = span.enter();

        // Compute the wa polynomial using the above table
        let wa: Vec<F> = trace
            .par_iter()
            .map(|cycle| {
                remap_address(cycle.ram_access().address() as u64, memory_layout)
                    .map_or(F::zero(), |k| eq_r_address[k as usize])
            })
            .collect();
        let wa = MultilinearPolynomial::from(wa);

        drop(_guard);
        drop(span);

        let inc = CommittedPolynomial::RamInc.generate_witness(preprocessing, trace);

        let span = tracing::span!(tracing::Level::INFO, "compute LT(j, r_cycle)");
        let _guard = span.enter();

        let mut lt: Vec<F> = unsafe_allocate_zero_vec(T);
        for (i, r) in r_cycle.r.iter().rev().enumerate() {
            let (evals_left, evals_right) = lt.split_at_mut(1 << i);
            evals_left
                .par_iter_mut()
                .zip(evals_right.par_iter_mut())
                .for_each(|(x, y)| {
                    *y = *x * r;
                    *x += *r - *y;
                });
        }
        let lt = MultilinearPolynomial::from(lt);

        drop(_guard);
        drop(span);

        ValEvaluationSumcheck {
            claimed_evaluation,
            init_eval,
            num_rounds: T.log_2(),
            prover_state: Some(ValEvaluationProverState { inc, wa, lt }),
            K,
        }
    }

    pub fn new_verifier<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        initial_ram_state: &[u64],
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
        private_input_evaluation: F,
    ) -> Self {
        let (_, _, T) = state_manager.get_verifier_data();
        let K = state_manager.ram_K;

        let (r, claimed_evaluation) = state_manager.get_virtual_polynomial_opening(
            VirtualPolynomial::RamVal,
            SumcheckId::RamReadWriteChecking,
        );
        let (r_address, _) = r.split_at(K.log_2());
        
        // Store the private input evaluation for verification later
        state_manager.private_input_evaluation = Some(private_input_evaluation);
        tracing::info!("Verifier received private input evaluation: {:?}", private_input_evaluation);
        
        // Verify the private input commitment
        if let Some(ref private_input_commitment) = state_manager.private_input_commitment {
            if let Some(ref private_input_proof) = state_manager.private_input_proof {
                let (preprocessing, _, _) = state_manager.get_verifier_data();
                let mut transcript_clone = state_manager.get_transcript().borrow().clone();
                
                // Use the existing verifier setup that's already properly configured for Dory
                let verifier_setup = &preprocessing.generators;
                
                // Verify the opening proof
                match PCS::verify(
                    private_input_proof,
                    verifier_setup,
                    &mut transcript_clone,
                    &r_address.r,
                    &private_input_evaluation,
                    private_input_commitment,
                ) {
                    Ok(()) => {
                        tracing::info!("Private input opening proof verified successfully");
                    },
                    Err(e) => {
                        tracing::error!("Private input opening proof verification failed: {:?}", e);
                        panic!("Private input opening proof verification failed: {:?}", e);
                    }
                }
            } else {
                tracing::warn!("Private input proof not found in state manager");
            }
        } else {
            tracing::warn!("Private input commitment not found in state manager");
        }

        // let val_init: MultilinearPolynomial<F> =
        //     MultilinearPolynomial::from(initial_ram_state.to_vec());
        // // println!("initial_ram_state verifier: {:?}", initial_ram_state.to_vec());
        // let init_eval = val_init.evaluate(&r_address.r);
        // println!("Val init is verifier: {}", init_eval);

        
        // Create multilinear polynomials from the split parts
        let val_right: MultilinearPolynomial<F> = MultilinearPolynomial::from(initial_ram_state.to_vec());
        let right_eval = val_right.evaluate(&r_address.r);

        ValEvaluationSumcheck {
            claimed_evaluation,
            // init_eval,
            init_eval: state_manager.private_input_evaluation.unwrap() + right_eval,
            num_rounds: T.log_2(),
            prover_state: None,
            K,
        }
    }
}

impl<F: JoltField> SumcheckInstance<F> for ValEvaluationSumcheck<F> {
    fn degree(&self) -> usize {
        3
    }

    fn num_rounds(&self) -> usize {
        self.num_rounds
    }

    fn input_claim(&self) -> F {
        self.claimed_evaluation - self.init_eval
    }

    #[tracing::instrument(skip_all, name = "RamValEvaluationSumcheck::compute_prover_message")]
    fn compute_prover_message(&mut self, _round: usize, _previous_claim: F) -> Vec<F> {
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");

        const DEGREE: usize = 3;
        let univariate_poly_evals: [F; 3] = (0..prover_state.inc.len() / 2)
            .into_par_iter()
            .map(|i| {
                let inc_evals = prover_state
                    .inc
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::HighToLow);
                let wa_evals = prover_state
                    .wa
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::HighToLow);
                let lt_evals = prover_state
                    .lt
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::HighToLow);

                [
                    inc_evals[0] * wa_evals[0] * lt_evals[0],
                    inc_evals[1] * wa_evals[1] * lt_evals[1],
                    inc_evals[2] * wa_evals[2] * lt_evals[2],
                ]
            })
            .reduce(
                || [F::zero(); 3],
                |running, new| {
                    [
                        running[0] + new[0],
                        running[1] + new[1],
                        running[2] + new[2],
                    ]
                },
            );

        univariate_poly_evals.to_vec()
    }

    #[tracing::instrument(skip_all, name = "RamValEvaluationSumcheck::bind")]
    fn bind(&mut self, r_j: F, _round: usize) {
        if let Some(prover_state) = &mut self.prover_state {
            [
                &mut prover_state.inc,
                &mut prover_state.wa,
                &mut prover_state.lt,
            ]
            .par_iter_mut()
            .for_each(|poly| poly.bind_parallel(r_j, BindingOrder::HighToLow));
        }
    }

    fn expected_output_claim(
        &self,
        accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
        r: &[F],
    ) -> F {
        let accumulator = accumulator.as_ref().unwrap();
        let (r_val, _) = accumulator.borrow().get_virtual_polynomial_opening(
            VirtualPolynomial::RamVal,
            SumcheckId::RamReadWriteChecking,
        );
        let (_, r_cycle) = r_val.split_at(self.K.log_2());
        // Compute LT(r_cycle', r_cycle)
        let mut lt_eval = F::zero();
        let mut eq_term = F::one();
        for (x, y) in r.iter().zip(r_cycle.r.iter()) {
            lt_eval += (F::one() - x) * y * eq_term;
            eq_term *= F::one() - x - y + *x * y + *x * y;
        }

        let (_, inc_claim) = accumulator.borrow().get_committed_polynomial_opening(
            CommittedPolynomial::RamInc,
            SumcheckId::RamValEvaluation,
        );
        let (_, wa_claim) = accumulator
            .borrow()
            .get_virtual_polynomial_opening(VirtualPolynomial::RamRa, SumcheckId::RamValEvaluation);

        // Return inc_claim * wa_claim * lt_eval
        inc_claim * wa_claim * lt_eval
    }

    fn normalize_opening_point(&self, opening_point: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::new(opening_point.to_vec())
    }

    fn cache_openings_prover(
        &self,
        accumulator: Rc<RefCell<ProverOpeningAccumulator<F>>>,
        r_cycle_prime: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        if let Some(prover_state) = &self.prover_state {
            let r = accumulator
                .borrow()
                .get_virtual_polynomial_opening(
                    VirtualPolynomial::RamVal,
                    SumcheckId::RamReadWriteChecking,
                )
                .0;
            let (r_address, _) = r.split_at(r.len() - r_cycle_prime.len());
            let wa_opening_point =
                OpeningPoint::new([r_address.r.as_slice(), r_cycle_prime.r.as_slice()].concat());

            accumulator.borrow_mut().append_virtual(
                VirtualPolynomial::RamRa,
                SumcheckId::RamValEvaluation,
                wa_opening_point,
                prover_state.wa.final_sumcheck_claim(),
            );

            accumulator.borrow_mut().append_dense(
                vec![CommittedPolynomial::RamInc],
                SumcheckId::RamValEvaluation,
                r_cycle_prime.r,
                &[prover_state.inc.final_sumcheck_claim()],
            );
        }
    }

    fn cache_openings_verifier(
        &self,
        accumulator: Rc<RefCell<VerifierOpeningAccumulator<F>>>,
        r_cycle_prime: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let r = accumulator
            .borrow()
            .get_virtual_polynomial_opening(
                VirtualPolynomial::RamVal,
                SumcheckId::RamReadWriteChecking,
            )
            .0;
        let (r_address, _) = r.split_at(r.len() - r_cycle_prime.len());
        let wa_opening_point =
            OpeningPoint::new([r_address.r.as_slice(), r_cycle_prime.r.as_slice()].concat());

        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::RamRa,
            SumcheckId::RamValEvaluation,
            wa_opening_point,
        );

        accumulator.borrow_mut().append_dense(
            vec![CommittedPolynomial::RamInc],
            SumcheckId::RamValEvaluation,
            r_cycle_prime.r,
        );
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}
