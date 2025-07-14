use std::{cell::RefCell, rc::Rc};

use crate::{
    dag::{stage::StagedSumcheck, state_manager::StateManager},
    field::JoltField,
    jolt::{vm::ram::remap_address, witness::CommittedPolynomials},
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        eq_poly::EqPolynomial,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        opening_proof::{
            OpeningPoint, OpeningsKeys, ProverOpeningAccumulator, VerifierOpeningAccumulator,
            BIG_ENDIAN,
        },
    },
    subprotocols::sumcheck::{
        BatchableSumcheckInstance, CacheSumcheckOpenings, SumcheckInstanceProof,
    },
    utils::{math::Math, thread::unsafe_allocate_zero_vec, transcript::Transcript},
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use rayon::prelude::*;

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone)]
pub struct ValEvaluationProof<F: JoltField, ProofTranscript: Transcript> {
    /// Sumcheck proof for the Val-evaluation sumcheck (steps 6 of Figure 9).
    sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    /// Inc(r_cycle')
    inc_claim: F,
    /// wa(r_address, r_cycle')
    wa_claim: F,
}

pub struct ValEvaluationProverState<F: JoltField> {
    /// Inc polynomial
    inc: MultilinearPolynomial<F>,
    /// wa polynomial
    wa: MultilinearPolynomial<F>,
    /// LT polynomial
    lt: MultilinearPolynomial<F>,
}

pub struct ValEvaluationVerifierState<F: JoltField> {
    /// log T
    num_rounds: usize,
    /// used to compute LT evaluation
    r_address: Vec<F>,
    /// used to compute LT evaluation
    r_cycle: Vec<F>,
}

#[derive(Clone)]
pub struct ValEvaluationSumcheckClaims<F: JoltField> {
    /// Inc(r_cycle')
    inc_claim: F,
    /// wa(r_address, r_cycle')
    wa_claim: F,
}

/// Val-evaluation sumcheck for RAM
pub struct ValEvaluationSumcheck<F: JoltField> {
    /// Initial claim value
    claimed_evaluation: F,
    /// Initial evaluation to subtract (for RAM)
    init_eval: F,
    /// Prover state
    prover_state: Option<ValEvaluationProverState<F>>,
    /// Verifier state
    verifier_state: Option<ValEvaluationVerifierState<F>>,
    /// Claims
    claims: Option<ValEvaluationSumcheckClaims<F>>,
}

impl<F: JoltField> ValEvaluationSumcheck<F> {
    pub fn new_prover<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        K: usize,
        initial_ram_state: &[u32],
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Self {
        let (preprocessing, trace, program_io, _) = state_manager.get_prover_data();
        let memory_layout = &program_io.memory_layout;
        let T = trace.len();

        let mut r_address = state_manager
            .get_opening_point(OpeningsKeys::RamReadWriteCheckingVal)
            .unwrap()
            .r;
        let r_cycle = r_address.split_off(K.log_2());

        let val_init: MultilinearPolynomial<F> =
            MultilinearPolynomial::from(initial_ram_state.to_vec());
        let init_eval = val_init.evaluate(&r_address);

        // Compute the size-K table storing all eq(r_address, k) evaluations for
        // k \in {0, 1}^log(K)
        let eq_r_address = EqPolynomial::evals(&r_address);

        let span = tracing::span!(tracing::Level::INFO, "compute wa(r_address, j)");
        let _guard = span.enter();

        // Compute the wa polynomial using the above table
        let wa: Vec<F> = trace
            .par_iter()
            .map(|cycle| {
                let k = remap_address(cycle.ram_access().address() as u64, memory_layout) as usize;
                eq_r_address[k]
            })
            .collect();
        let wa = MultilinearPolynomial::from(wa);

        drop(_guard);
        drop(span);

        let inc = CommittedPolynomials::RamInc.generate_witness(preprocessing, trace);

        let span = tracing::span!(tracing::Level::INFO, "compute LT(j, r_cycle)");
        let _guard = span.enter();

        let mut lt: Vec<F> = unsafe_allocate_zero_vec(T);
        for (i, r) in r_cycle.iter().rev().enumerate() {
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

        // Create the sumcheck instance
        ValEvaluationSumcheck {
            claimed_evaluation: state_manager.get_opening(OpeningsKeys::RamReadWriteCheckingVal),
            init_eval,
            prover_state: Some(ValEvaluationProverState { inc, wa, lt }),
            verifier_state: None,
            claims: None,
        }
    }

    pub fn new_verifier<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        K: usize,
        initial_ram_state: &[u32],
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Self {
        let (_, _, T) = state_manager.get_verifier_data();

        let mut r_address = state_manager
            .get_opening_point(OpeningsKeys::RamReadWriteCheckingVal)
            .unwrap()
            .r;
        let r_cycle = r_address.split_off(K.log_2());

        let val_init: MultilinearPolynomial<F> =
            MultilinearPolynomial::from(initial_ram_state.to_vec());
        let init_eval = val_init.evaluate(&r_address);
        let verifier_state = ValEvaluationVerifierState {
            num_rounds: T.log_2(),
            r_cycle,
            r_address,
        };
        let claims = ValEvaluationSumcheckClaims {
            inc_claim: state_manager.get_opening(OpeningsKeys::RamValEvaluationInc),
            wa_claim: state_manager.get_opening(OpeningsKeys::RamValEvaluationWa),
        };

        // Create the sumcheck instance
        ValEvaluationSumcheck {
            claimed_evaluation: state_manager.get_opening(OpeningsKeys::RamReadWriteCheckingVal),
            init_eval,
            prover_state: None,
            verifier_state: Some(verifier_state),
            claims: Some(claims),
        }
    }
}

impl<F: JoltField> BatchableSumcheckInstance<F> for ValEvaluationSumcheck<F> {
    fn degree(&self) -> usize {
        3
    }

    fn num_rounds(&self) -> usize {
        if let Some(prover_state) = &self.prover_state {
            prover_state.inc.len().log_2()
        } else if let Some(verifier_state) = &self.verifier_state {
            verifier_state.num_rounds
        } else {
            panic!("Neither prover state nor verifier state is initialized");
        }
    }

    fn input_claim(&self) -> F {
        self.claimed_evaluation - self.init_eval
    }

    fn compute_prover_message(&mut self, _round: usize) -> Vec<F> {
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
                    .sumcheck_evals(i, DEGREE, BindingOrder::LowToHigh);
                let wa_evals = prover_state
                    .wa
                    .sumcheck_evals(i, DEGREE, BindingOrder::LowToHigh);
                let lt_evals = prover_state
                    .lt
                    .sumcheck_evals(i, DEGREE, BindingOrder::LowToHigh);

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

    fn bind(&mut self, r_j: F, _round: usize) {
        if let Some(prover_state) = &mut self.prover_state {
            [
                &mut prover_state.inc,
                &mut prover_state.wa,
                &mut prover_state.lt,
            ]
            .par_iter_mut()
            .for_each(|poly| poly.bind_parallel(r_j, BindingOrder::LowToHigh));
        }
    }

    fn expected_output_claim(&self, r: &[F]) -> F {
        let verifier_state = self
            .verifier_state
            .as_ref()
            .expect("Verifier state not initialized");
        let claims = self.claims.as_ref().expect("Claims not cached");

        // r contains r_cycle_prime in low-to-high order
        let r_cycle_prime: Vec<F> = r.iter().rev().copied().collect();

        // Compute LT(r_cycle', r_cycle)
        let mut lt_eval = F::zero();
        let mut eq_term = F::one();
        for (x, y) in r_cycle_prime.iter().zip(verifier_state.r_cycle.iter()) {
            lt_eval += (F::one() - x) * y * eq_term;
            eq_term *= F::one() - x - y + *x * y + *x * y;
        }

        // Return inc_claim * wa_claim * lt_eval
        claims.inc_claim * claims.wa_claim * lt_eval
    }
}

impl<F, PCS> CacheSumcheckOpenings<F, PCS> for ValEvaluationSumcheck<F>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    fn normalize_opening_point(&self, opening_point: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::new(opening_point.iter().copied().rev().collect())
    }

    fn cache_openings_prover(
        &mut self,
        accumulator: Option<Rc<RefCell<ProverOpeningAccumulator<F, PCS>>>>,
        r_cycle_prime: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        if let Some(prover_state) = &self.prover_state {
            self.claims = Some(ValEvaluationSumcheckClaims {
                inc_claim: prover_state.inc.final_sumcheck_claim(),
                wa_claim: prover_state.wa.final_sumcheck_claim(),
            });

            let accumulator = accumulator.expect("accumulator is needed");
            let mut r_address = accumulator
                .borrow()
                .get_opening_point(OpeningsKeys::RamReadWriteCheckingVal)
                .unwrap();
            let _r_cycle = r_address.split_off(r_address.len() - r_cycle_prime.len());
            let wa_opening_point =
                OpeningPoint::new([r_address.r.as_slice(), r_cycle_prime.r.as_slice()].concat());

            // TODO(moodlezoup): append_dense
            accumulator.borrow_mut().append_virtual(
                OpeningsKeys::RamValEvaluationWa,
                wa_opening_point,
                prover_state.wa.final_sumcheck_claim(),
            );

            // TODO(moodlezoup): append_dense
            accumulator.borrow_mut().append_virtual(
                OpeningsKeys::RamValEvaluationInc,
                r_cycle_prime,
                prover_state.inc.final_sumcheck_claim(),
            );
        }
    }

    fn cache_openings_verifier(
        &mut self,
        accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F, PCS>>>>,
        r_cycle_prime: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let accumulator = accumulator.expect("accumulator is needed");
        let mut r_address = accumulator
            .borrow()
            .get_opening_point(OpeningsKeys::RamReadWriteCheckingVal)
            .unwrap();
        let _r_cycle = r_address.split_off(r_address.len() - r_cycle_prime.len());
        let wa_opening_point =
            OpeningPoint::new([r_address.r.as_slice(), r_cycle_prime.r.as_slice()].concat());

        accumulator
            .borrow_mut()
            .populate_claim_opening(OpeningsKeys::RamValEvaluationInc, r_cycle_prime);
        accumulator
            .borrow_mut()
            .populate_claim_opening(OpeningsKeys::RamValEvaluationWa, wa_opening_point);
    }
}

impl<F, PCS> StagedSumcheck<F, PCS> for ValEvaluationSumcheck<F>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
}
