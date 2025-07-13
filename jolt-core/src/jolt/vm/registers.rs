use std::{cell::RefCell, rc::Rc};

use crate::{
    dag::stage::StagedSumcheck,
    field::JoltField,
    jolt::{
        vm::{
            registers_read_write_checking::{
                RegistersReadWriteChecking, RegistersReadWriteCheckingProof,
            },
            JoltCommitments, JoltProverPreprocessing,
        },
        witness::CommittedPolynomials,
    },
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        eq_poly::EqPolynomial,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        opening_proof::{OpeningsKeys, ProverOpeningAccumulator, VerifierOpeningAccumulator},
    },
    subprotocols::sumcheck::{
        BatchableSumcheckInstance, CacheSumcheckOpenings, SumcheckInstanceProof,
    },
    utils::{
        errors::ProofVerifyError,
        math::Math,
        thread::{drop_in_background_thread, unsafe_allocate_zero_vec},
        transcript::Transcript,
    },
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use rayon::prelude::*;
use tracer::instruction::RV32IMCycle;

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone)]
pub struct RegistersTwistProof<F: JoltField, ProofTranscript: Transcript> {
    /// Proof for the read-checking and write-checking sumchecks
    /// (steps 3 and 4 of Figure 9).
    read_write_checking_proof: RegistersReadWriteCheckingProof<F, ProofTranscript>,
    /// Proof of the Val-evaluation sumcheck (step 6 of Figure 9).
    val_evaluation_proof: ValEvaluationProof<F, ProofTranscript>,
}

pub struct RegistersDAG {}

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone)]
pub struct ValEvaluationProof<F: JoltField, ProofTranscript: Transcript> {
    /// Sumcheck proof for the Val-evaluation sumcheck (steps 6 of Figure 9).
    sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    /// Inc(r_cycle')
    inc_claim: F,
    /// wa(r_address, r_cycle')
    wa_claim: F,
}

pub(crate) struct ValEvaluationProverState<F: JoltField> {
    /// Inc polynomial
    pub inc: MultilinearPolynomial<F>,
    /// wa polynomial
    pub wa: MultilinearPolynomial<F>,
    /// LT polynomial
    pub lt: MultilinearPolynomial<F>,
    /// Track the sumcheck rounds
    pub r_sumcheck: Vec<F>,
}

/// Verifier state for the Val-evaluation sumcheck
pub(crate) struct ValEvaluationVerifierState<F: JoltField> {
    /// The number of rounds (log T)
    pub num_rounds: usize,
    /// r_address used to compute LT evaluation
    pub r_address: Vec<F>,
    /// r_cycle used to compute LT evaluation
    pub r_cycle: Vec<F>,
}

/// Claims output by the Val-evaluation sumcheck
#[derive(Clone)]
pub(crate) struct ValEvaluationSumcheckClaims<F: JoltField> {
    /// Inc(r_cycle')
    pub inc_claim: F,
    /// wa(r_address, r_cycle')
    pub wa_claim: F,
}

/// Val-evaluation sumcheck instance implementing BatchableSumcheckInstance
pub(crate) struct ValEvaluationSumcheck<F: JoltField> {
    /// Initial claim value
    pub claimed_evaluation: F,
    /// Prover state
    pub prover_state: Option<ValEvaluationProverState<F>>,
    /// Verifier state
    pub verifier_state: Option<ValEvaluationVerifierState<F>>,
    /// Claims
    pub claims: Option<ValEvaluationSumcheckClaims<F>>,
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
        self.claimed_evaluation
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
            // Track the sumcheck rounds for prover
            prover_state.r_sumcheck.push(r_j);
            
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
    fn cache_openings_prover(
        &mut self,
        accumulator: Option<Rc<RefCell<ProverOpeningAccumulator<F, PCS>>>>,
    ) {
        debug_assert!(self.claims.is_none());
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");
        
        let inc_claim = prover_state.inc.final_sumcheck_claim();
        let wa_claim = prover_state.wa.final_sumcheck_claim();
        
        self.claims = Some(ValEvaluationSumcheckClaims {
            inc_claim,
            wa_claim,
        });
        
        // Append claims to accumulator if provided
        if let Some(accumulator) = accumulator {
            // Get the actual sumcheck opening point (r_cycle_prime)
            // The sumcheck binds in low-to-high order, so we reverse to get r_cycle_prime
            let mut r_cycle_prime = prover_state.r_sumcheck.clone();
            r_cycle_prime.reverse();
            
            accumulator.borrow_mut().append_virtual(
                OpeningsKeys::RegistersValEvaluationInc,
                r_cycle_prime.clone(),
                inc_claim,
            );
            accumulator.borrow_mut().append_virtual(
                OpeningsKeys::RegistersValEvaluationWa,
                r_cycle_prime,
                wa_claim,
            );
        }
    }
}

impl<F: JoltField, PCS: CommitmentScheme<Field = F>> StagedSumcheck<F, PCS>
    for ValEvaluationSumcheck<F>
{
}

impl<F: JoltField, ProofTranscript: Transcript> RegistersTwistProof<F, ProofTranscript> {
    #[tracing::instrument(skip_all, name = "RegistersTwistProof::prove")]
    pub fn prove<PCS: CommitmentScheme<Field = F>>(
        preprocessing: &JoltProverPreprocessing<F, PCS>,
        trace: &[RV32IMCycle],
        opening_accumulator: &mut ProverOpeningAccumulator<F, PCS>,
        transcript: &mut ProofTranscript,
    ) -> RegistersTwistProof<F, ProofTranscript> {
        let log_T = trace.len().log_2();
        let r_prime: Vec<F> = transcript.challenge_vector(log_T);

        let (read_write_checking_proof, r_address, r_cycle) =
            RegistersReadWriteChecking::prove(preprocessing, trace, &r_prime, transcript);

        let (val_evaluation_proof, mut r_cycle_prime) = prove_val_evaluation(
            preprocessing,
            trace,
            r_address.clone(),
            r_cycle,
            read_write_checking_proof.claims.val_claim,
            transcript,
        );
        // Cycle variables are bound from low to high
        r_cycle_prime.reverse();

        let rd_inc_poly = CommittedPolynomials::RdInc.generate_witness(preprocessing, trace);
        opening_accumulator.append_dense(
            &[&rd_inc_poly],
            EqPolynomial::evals(&r_cycle_prime),
            r_cycle_prime,
            &[val_evaluation_proof.inc_claim],
            transcript,
            None, // No openings keys needed
        );

        RegistersTwistProof {
            read_write_checking_proof,
            val_evaluation_proof,
        }
    }

    pub fn verify<PCS: CommitmentScheme<Field = F>>(
        &self,
        commitments: &JoltCommitments<F, PCS>,
        T: usize,
        opening_accumulator: &mut VerifierOpeningAccumulator<F, PCS>,
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        let log_T = T.log_2();
        let r_prime: Vec<F> = transcript.challenge_vector(log_T);

        let (r_address, r_cycle) = RegistersReadWriteChecking::verify(
            &self.read_write_checking_proof,
            &r_prime,
            transcript,
        )?;

        let sumcheck_instance = ValEvaluationSumcheck {
            claimed_evaluation: self.read_write_checking_proof.claims.val_claim,
            prover_state: None,
            verifier_state: Some(ValEvaluationVerifierState {
                num_rounds: log_T,
                r_address,
                r_cycle,
            }),
            claims: Some(ValEvaluationSumcheckClaims {
                inc_claim: self.val_evaluation_proof.inc_claim,
                wa_claim: self.val_evaluation_proof.wa_claim,
            }),
        };

        let mut r_cycle_prime =
            <ValEvaluationSumcheck<F> as BatchableSumcheckInstance<F>>::verify_single(
                &sumcheck_instance,
                &self.val_evaluation_proof.sumcheck_proof,
                transcript,
            )?;

        // Cycle variables are bound from low to high
        r_cycle_prime.reverse();

        let inc_commitment = &commitments.commitments[CommittedPolynomials::RdInc.to_index()];
        opening_accumulator.append(
            &[inc_commitment],
            r_cycle_prime,
            &[self.val_evaluation_proof.inc_claim],
            transcript,
        );

        // TODO: Append Inc claim to opening proof accumulator

        Ok(())
    }
}

/// Implements the sumcheck prover for the Val-evaluation sumcheck described in
/// Section 8.1 and Appendix B of the Twist+Shout paper
/// TODO(moodlezoup): incorporate optimization from Appendix B.2
#[tracing::instrument(skip_all)]
pub fn prove_val_evaluation<
    F: JoltField,
    ProofTranscript: Transcript,
    PCS: CommitmentScheme<Field = F>,
>(
    preprocessing: &JoltProverPreprocessing<F, PCS>,
    trace: &[RV32IMCycle],
    r_address: Vec<F>,
    r_cycle: Vec<F>,
    claimed_evaluation: F,
    transcript: &mut ProofTranscript,
) -> (ValEvaluationProof<F, ProofTranscript>, Vec<F>) {
    let T = r_cycle.len().pow2();

    // Compute the size-K table storing all eq(r_address, k) evaluations for
    // k \in {0, 1}^log(K)
    let eq_r_address = EqPolynomial::evals(&r_address);

    let span = tracing::span!(tracing::Level::INFO, "compute wa(r_address, j)");
    let _guard = span.enter();

    // Compute the wa polynomial using the above table
    let wa: Vec<F> = trace
        .par_iter()
        .map(|cycle| {
            let instr = cycle.instruction().normalize();
            eq_r_address[instr.operands.rd]
        })
        .collect();
    let wa = MultilinearPolynomial::from(wa);

    drop(_guard);
    drop(span);

    let inc = CommittedPolynomials::RdInc.generate_witness(preprocessing, trace);

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

    let mut sumcheck_instance: ValEvaluationSumcheck<F> = ValEvaluationSumcheck {
        claimed_evaluation,
        prover_state: Some(ValEvaluationProverState { 
            inc, 
            wa, 
            lt,
            r_sumcheck: Vec::new(),
        }),
        verifier_state: None,
        claims: None,
    };

    let span = tracing::span!(tracing::Level::INFO, "Val-evaluation sumcheck");
    let _guard = span.enter();

    let (sumcheck_proof, r_cycle_prime) = <ValEvaluationSumcheck<F> as BatchableSumcheckInstance<
        F,
    >>::prove_single(&mut sumcheck_instance, transcript);

    drop(_guard);
    drop(span);

    let claims = sumcheck_instance.claims.expect("Claims should be set");

    let proof = ValEvaluationProof {
        sumcheck_proof,
        inc_claim: claims.inc_claim,
        wa_claim: claims.wa_claim,
    };

    // Clean up
    if let Some(prover_state) = sumcheck_instance.prover_state {
        drop_in_background_thread((
            prover_state.inc,
            prover_state.wa,
            eq_r_address,
            prover_state.lt,
        ));
    }

    (proof, r_cycle_prime)
}
