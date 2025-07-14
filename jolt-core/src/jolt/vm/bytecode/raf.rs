use std::cell::RefCell;
use std::rc::Rc;

use crate::jolt::vm::bytecode::BytecodePreprocessing;
use crate::poly::identity_poly::IdentityPolynomial;
use crate::subprotocols::sumcheck::CacheSumcheckOpenings;
use crate::utils::errors::ProofVerifyError;
use crate::{
    field::JoltField,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        opening_proof::ProverOpeningAccumulator,
    },
    subprotocols::sumcheck::{BatchableSumcheckInstance, SumcheckInstanceProof},
    utils::{math::Math, transcript::Transcript},
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use rayon::prelude::*;
use tracer::instruction::RV32IMCycle;

struct RafBytecodeProverState<F: JoltField> {
    ra_poly: MultilinearPolynomial<F>,
    ra_poly_shift: MultilinearPolynomial<F>,
    int_poly: IdentityPolynomial<F>,
}

pub struct RafBytecode<F: JoltField> {
    /// Input claim: raf_claim + challenge * raf_claim_shift
    input_claim: F,
    /// Challenge value shared by prover and verifier
    challenge: F,
    /// K value shared by prover and verifier
    K: usize,
    /// Prover state
    prover_state: Option<RafBytecodeProverState<F>>,
    /// Cached ra claims after sumcheck completes
    ra_claims: Option<(F, F)>,
}

impl<F: JoltField> RafBytecode<F> {
    pub fn new(
        input_claim: F,
        ra_poly: MultilinearPolynomial<F>,
        ra_poly_shift: MultilinearPolynomial<F>,
        int_poly: IdentityPolynomial<F>,
        challenge: F,
        K: usize,
    ) -> Self {
        Self {
            input_claim,
            challenge,
            K,
            prover_state: Some(RafBytecodeProverState {
                ra_poly,
                ra_poly_shift,
                int_poly,
            }),
            ra_claims: None,
        }
    }

    pub fn new_verifier(input_claim: F, challenge: F, K: usize) -> Self {
        Self {
            input_claim,
            challenge,
            K,
            prover_state: None,
            ra_claims: None,
        }
    }
}

impl<F: JoltField> BatchableSumcheckInstance<F> for RafBytecode<F> {
    fn degree(&self) -> usize {
        2
    }

    fn num_rounds(&self) -> usize {
        self.K.log_2()
    }

    fn input_claim(&self) -> F {
        self.input_claim
    }

    fn compute_prover_message(&mut self, _round: usize) -> Vec<F> {
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");
        const DEGREE: usize = 2;

        let univariate_poly_evals: [F; 2] = (0..prover_state.ra_poly.len() / 2)
            .into_par_iter()
            .map(|i| {
                let ra_evals = prover_state
                    .ra_poly
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);
                let ra_evals_shift = prover_state
                    .ra_poly_shift
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);
                let int_evals =
                    prover_state
                        .int_poly
                        .sumcheck_evals(i, DEGREE, BindingOrder::LowToHigh);

                [
                    (ra_evals[0] + self.challenge * ra_evals_shift[0]) * int_evals[0],
                    (ra_evals[1] + self.challenge * ra_evals_shift[1]) * int_evals[1],
                ]
            })
            .reduce(
                || [F::zero(); 2],
                |mut running, new| {
                    for i in 0..2 {
                        running[i] += new[i];
                    }
                    running
                },
            );

        univariate_poly_evals.to_vec()
    }

    fn bind(&mut self, r_j: F, _round: usize) {
        let prover_state = self
            .prover_state
            .as_mut()
            .expect("Prover state not initialized");

        rayon::join(
            || {
                prover_state
                    .ra_poly
                    .bind_parallel(r_j, BindingOrder::LowToHigh)
            },
            || {
                rayon::join(
                    || {
                        prover_state
                            .ra_poly_shift
                            .bind_parallel(r_j, BindingOrder::LowToHigh)
                    },
                    || {
                        prover_state
                            .int_poly
                            .bind_parallel(r_j, BindingOrder::LowToHigh)
                    },
                )
            },
        );
    }

    fn expected_output_claim(&self, r: &[F]) -> F {
        let (ra_claim, ra_claim_shift) = self.ra_claims.as_ref().expect("ra_claims not set");

        let int_eval = IdentityPolynomial::new(self.K.log_2()).evaluate(r);

        // Verify sumcheck_claim = int(r) * (ra_claim + challenge * ra_claim_shift)
        int_eval * (*ra_claim + self.challenge * *ra_claim_shift)
    }
}

impl<F, PCS> CacheSumcheckOpenings<F, PCS> for RafBytecode<F>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    fn cache_openings_prover(
        &mut self,
        _accumulator: Option<Rc<RefCell<ProverOpeningAccumulator<F, PCS>>>>,
    ) {
        debug_assert!(self.ra_claims.is_none());
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");

        let ra_claim = prover_state.ra_poly.final_sumcheck_claim();
        let ra_claim_shift = prover_state.ra_poly_shift.final_sumcheck_claim();

        self.ra_claims = Some((ra_claim, ra_claim_shift));
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone)]
pub struct RafEvaluationProof<F: JoltField, ProofTranscript: Transcript> {
    sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    ra_claim: F,
    ra_claim_shift: F,
    raf_claim: F,
    raf_claim_shift: F,
}

impl<F: JoltField, ProofTranscript: Transcript> RafEvaluationProof<F, ProofTranscript> {
    #[allow(clippy::too_many_arguments)]
    #[tracing::instrument(skip_all, name = "RafEvaluationProof::prove")]
    pub fn prove(
        preprocessing: &BytecodePreprocessing,
        trace: &[RV32IMCycle],
        ra_poly: MultilinearPolynomial<F>,
        ra_poly_shift: MultilinearPolynomial<F>,
        r_cycle: &[F],
        r_shift: &[F],
        challenge: F,
        transcript: &mut ProofTranscript,
    ) -> Self {
        let K = preprocessing.bytecode.len().next_power_of_two();
        let int_poly = IdentityPolynomial::new(K.log_2());

        // TODO: Propagate raf claim from Spartan
        let raf_evals = preprocessing.map_trace_to_pc(trace).collect::<Vec<u64>>();
        let raf_poly = MultilinearPolynomial::from(raf_evals);
        let raf_claim = raf_poly.evaluate(r_cycle);
        let raf_claim_shift = raf_poly.evaluate(r_shift);
        let input_claim = raf_claim + challenge * raf_claim_shift;

        let mut raf_sumcheck =
            RafBytecode::new(input_claim, ra_poly, ra_poly_shift, int_poly, challenge, K);

        let (sumcheck_proof, _r_address) = raf_sumcheck.prove_single(transcript);

        let (ra_claim, ra_claim_shift) = raf_sumcheck
            .ra_claims
            .expect("ra_claims should be set after prove_single");

        Self {
            sumcheck_proof,
            ra_claim,
            ra_claim_shift,
            raf_claim,
            raf_claim_shift,
        }
    }

    pub fn verify(
        &self,
        K: usize,
        challenge: F,
        transcript: &mut ProofTranscript,
    ) -> Result<Vec<F>, ProofVerifyError> {
        let input_claim = self.raf_claim + challenge * self.raf_claim_shift;

        let mut raf_sumcheck = RafBytecode::new_verifier(input_claim, challenge, K);

        raf_sumcheck.ra_claims = Some((self.ra_claim, self.ra_claim_shift));

        let r_raf_sumcheck = raf_sumcheck.verify_single(&self.sumcheck_proof, transcript)?;

        Ok(r_raf_sumcheck)
    }
}
