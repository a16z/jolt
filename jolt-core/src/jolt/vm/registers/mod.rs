use std::{cell::RefCell, rc::Rc};

use crate::{
    field::JoltField,
    jolt::{
        vm::{
            registers::read_write_checking::RegistersReadWriteCheckingProof, JoltCommitments,
            JoltProverPreprocessing,
        },
        witness::{CommittedPolynomial, VirtualPolynomial},
    },
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_proof::{
            OpeningPoint, ProverOpeningAccumulator, SumcheckId, VerifierOpeningAccumulator,
            BIG_ENDIAN,
        },
    },
    subprotocols::sumcheck::{SumcheckInstance, SumcheckInstanceProof},
    utils::{errors::ProofVerifyError, math::Math, transcript::Transcript},
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use rayon::prelude::*;
use tracer::instruction::RV32IMCycle;

pub mod read_write_checking;

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone)]
pub struct RegistersTwistProof<F: JoltField, ProofTranscript: Transcript> {
    /// Proof for the read-checking and write-checking sumchecks
    /// (steps 3 and 4 of Figure 9).
    read_write_checking_proof: RegistersReadWriteCheckingProof<F, ProofTranscript>,
    /// Proof of the Val-evaluation sumcheck (step 6 of Figure 9).
    val_evaluation_proof: ValEvaluationProof<F, ProofTranscript>,
}

#[derive(Default)]
pub struct RegistersDag {}

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
    pub inc: MultilinearPolynomial<F>,
    /// wa polynomial
    pub wa: MultilinearPolynomial<F>,
    /// LT polynomial
    pub lt: MultilinearPolynomial<F>,
}

/// Verifier state for the Val-evaluation sumcheck
pub(crate) struct ValEvaluationVerifierState<F: JoltField> {
    /// The number of rounds (log T)
    pub num_rounds: usize,
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

/// Val-evaluation sumcheck instance implementing SumcheckInstance
pub(crate) struct ValEvaluationSumcheck<F: JoltField> {
    // The `r_address` at in the claimed `Val(r_address, r_cycle)`
    r_address: Vec<F>,
    /// Initial claim value
    pub claimed_evaluation: F,
    /// Prover state
    pub prover_state: Option<ValEvaluationProverState<F>>,
    /// Verifier state
    pub verifier_state: Option<ValEvaluationVerifierState<F>>,
    /// Claims
    pub claims: Option<ValEvaluationSumcheckClaims<F>>,
}

impl<F: JoltField> SumcheckInstance<F> for ValEvaluationSumcheck<F> {
    fn degree(&self) -> usize {
        3
    }

    fn num_rounds(&self) -> usize {
        if let Some(prover_state) = &self.prover_state {
            prover_state.inc.original_len().log_2()
        } else if let Some(verifier_state) = &self.verifier_state {
            verifier_state.num_rounds
        } else {
            panic!("Neither prover state nor verifier state is initialized");
        }
    }

    fn input_claim(&self) -> F {
        self.claimed_evaluation
    }

    #[tracing::instrument(
        skip_all,
        name = "RegistersValEvaluationSumcheck::compute_prover_message"
    )]
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

    #[tracing::instrument(skip_all, name = "RegistersValEvaluationSumcheck::bind")]
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
        _accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
        r: &[F],
    ) -> F {
        let verifier_state = self
            .verifier_state
            .as_ref()
            .expect("Verifier state not initialized");
        let claims = self.claims.as_ref().expect("Claims not cached");

        // Compute LT(r_cycle', r_cycle)
        let mut lt_eval = F::zero();
        let mut eq_term = F::one();
        for (x, y) in r.iter().zip(verifier_state.r_cycle.iter()) {
            lt_eval += (F::one() - x) * y * eq_term;
            eq_term *= F::one() - x - y + *x * y + *x * y;
        }

        // Return inc_claim * wa_claim * lt_eval
        claims.inc_claim * claims.wa_claim * lt_eval
    }

    fn normalize_opening_point(&self, opening_point: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::new(opening_point.to_vec())
    }

    fn cache_openings_prover(
        &self,
        accumulator: Rc<RefCell<ProverOpeningAccumulator<F>>>,
        r_cycle: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");

        let inc_claim = prover_state.inc.final_sumcheck_claim();
        let wa_claim = prover_state.wa.final_sumcheck_claim();

        // Append claims to accumulator
        accumulator.borrow_mut().append_dense(
            vec![CommittedPolynomial::RdInc],
            SumcheckId::RegistersValEvaluation,
            r_cycle.r.clone(),
            &[inc_claim],
        );

        let r = [self.r_address.as_slice(), r_cycle.r.as_slice()].concat();
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::RdWa,
            SumcheckId::RegistersValEvaluation,
            OpeningPoint::new(r),
            wa_claim,
        );
    }

    fn cache_openings_verifier(
        &self,
        accumulator: Rc<RefCell<VerifierOpeningAccumulator<F>>>,
        r_cycle: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        // Append claims to accumulator
        accumulator.borrow_mut().append_dense(
            vec![CommittedPolynomial::RdInc],
            SumcheckId::RegistersValEvaluation,
            r_cycle.r.clone(),
        );

        let r = [self.r_address.as_slice(), r_cycle.r.as_slice()].concat();
        accumulator.borrow_mut().append_virtual(
            VirtualPolynomial::RdWa,
            SumcheckId::RegistersValEvaluation,
            OpeningPoint::new(r),
        );
    }
}

impl<F: JoltField, ProofTranscript: Transcript> RegistersTwistProof<F, ProofTranscript> {
    #[tracing::instrument(skip_all, name = "RegistersTwistProof::prove")]
    pub fn prove<PCS: CommitmentScheme<Field = F>>(
        _preprocessing: &JoltProverPreprocessing<F, PCS>,
        _trace: &[RV32IMCycle],
        _opening_accumulator: &mut ProverOpeningAccumulator<F>,
        _transcript: &mut ProofTranscript,
    ) -> RegistersTwistProof<F, ProofTranscript> {
        todo!()
    }

    pub fn verify<PCS: CommitmentScheme<Field = F>>(
        &self,
        _commitments: &JoltCommitments<F, PCS>,
        _T: usize,
        _opening_accumulator: &mut VerifierOpeningAccumulator<F>,
        _transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        todo!()
    }
}
