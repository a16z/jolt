use std::cell::RefCell;
use std::rc::Rc;

use crate::dag::state_manager::StateManager;
use crate::jolt::vm::bytecode::BytecodePreprocessing;
use crate::jolt::witness::{CommittedPolynomial, VirtualPolynomial};
use crate::poly::identity_poly::IdentityPolynomial;
use crate::poly::opening_proof::{
    OpeningPoint, SumcheckId, VerifierOpeningAccumulator, BIG_ENDIAN,
};
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
    subprotocols::sumcheck::{SumcheckInstance, SumcheckInstanceProof},
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
    raf_claim: F,
    raf_shift_claim: F,
    /// Batching challenge
    gamma: F,
    K: usize,
    /// Prover state
    prover_state: Option<RafBytecodeProverState<F>>,
}

impl<F: JoltField> RafBytecode<F> {
    pub fn new_prover(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
        ra_poly: MultilinearPolynomial<F>,
        ra_poly_shift: MultilinearPolynomial<F>,
    ) -> Self {
        let K = sm.get_prover_data().0.shared.bytecode.bytecode.len();
        let int_poly = IdentityPolynomial::new(K.log_2());
        let gamma: F = sm.get_transcript().borrow_mut().challenge_scalar();
        let (_, raf_claim) =
            sm.get_virtual_polynomial_opening(VirtualPolynomial::PC, SumcheckId::SpartanOuter);
        let (_, raf_shift_claim) =
            sm.get_virtual_polynomial_opening(VirtualPolynomial::PC, SumcheckId::SpartanShift);
        Self {
            raf_claim,
            raf_shift_claim,
            gamma,
            K,
            prover_state: Some(RafBytecodeProverState {
                ra_poly,
                ra_poly_shift,
                int_poly,
            }),
        }
    }

    pub fn new_verifier(
        sm: &mut StateManager<F, impl Transcript, impl CommitmentScheme<Field = F>>,
    ) -> Self {
        let (_, raf_claim) =
            sm.get_virtual_polynomial_opening(VirtualPolynomial::PC, SumcheckId::SpartanOuter);
        let (_, raf_shift_claim) =
            sm.get_virtual_polynomial_opening(VirtualPolynomial::PC, SumcheckId::SpartanShift);
        let gamma = sm.get_transcript().borrow_mut().challenge_scalar();
        let K = sm.get_verifier_data().0.shared.bytecode.bytecode.len();
        Self {
            raf_claim,
            raf_shift_claim,
            gamma,
            K,
            prover_state: None,
        }
    }
}

impl<F: JoltField> SumcheckInstance<F> for RafBytecode<F> {
    fn degree(&self) -> usize {
        2
    }

    fn num_rounds(&self) -> usize {
        self.K.log_2()
    }

    fn input_claim(&self) -> F {
        self.raf_claim + self.gamma * self.raf_shift_claim
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
                    (ra_evals[0] + self.gamma * ra_evals_shift[0]) * int_evals[0],
                    (ra_evals[1] + self.gamma * ra_evals_shift[1]) * int_evals[1],
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

    fn expected_output_claim(
        &self,
        accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
        r: &[F],
    ) -> F {
        let accumulator = accumulator.as_ref().unwrap();
        let (_, ra_claim) = accumulator.borrow().get_committed_polynomial_opening(
            CommittedPolynomial::BytecodeRa,
            SumcheckId::BytecodeReadChecking(0),
        );
        let (_, ra_claim_shift) = accumulator.borrow().get_committed_polynomial_opening(
            CommittedPolynomial::BytecodeRa,
            SumcheckId::BytecodeReadChecking(2),
        );

        let int_eval = IdentityPolynomial::new(self.K.log_2()).evaluate(r);

        int_eval * (ra_claim + self.gamma * ra_claim_shift)
    }

    fn normalize_opening_point(&self, opening_point: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::new(opening_point.iter().copied().rev().collect())
    }

    fn cache_openings_prover(
        &self,
        _accumulator: Rc<RefCell<ProverOpeningAccumulator<F>>>,
        _opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        // We don't need to cache claims since the read-checking sumcheck
        // already cached them, and since we are in the same batch,
        // claims are exactly the same
        // TODO: Add asserts
    }

    fn cache_openings_verifier(
        &self,
        _accumulator: Rc<RefCell<VerifierOpeningAccumulator<F>>>,
        _opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        // We don't need to cache claims since the read-checking sumcheck
        // already cached them, and since we are in the same batch,
        // claims are exactly the same
        // TODO: Add asserts
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
        _preprocessing: &BytecodePreprocessing,
        _trace: &[RV32IMCycle],
        _ra_poly: MultilinearPolynomial<F>,
        _ra_poly_shift: MultilinearPolynomial<F>,
        _r_cycle: &[F],
        _r_shift: &[F],
        _challenge: F,
        _transcript: &mut ProofTranscript,
    ) -> Self {
        todo!()
    }

    pub fn verify(
        &self,
        _K: usize,
        _challenge: F,
        _transcript: &mut ProofTranscript,
    ) -> Result<Vec<F>, ProofVerifyError> {
        todo!()
    }
}
