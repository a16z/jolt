use std::{cell::RefCell, rc::Rc};

use crate::{
    field::JoltField,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_proof::{OpeningPoint, ProverOpeningAccumulator, BIG_ENDIAN},
    },
    subprotocols::sumcheck::{
        BatchableSumcheckInstance, CacheSumcheckOpenings, SumcheckInstanceProof,
    },
    utils::{errors::ProofVerifyError, math::Math, transcript::Transcript},
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use rayon::prelude::*;

pub struct HammingWeightProverState<F: JoltField> {
    ra_poly: MultilinearPolynomial<F>,
}

pub struct HammingWeightSumcheck<F: JoltField> {
    /// K value shared by prover and verifier
    K: usize,
    /// Prover state
    prover_state: Option<HammingWeightProverState<F>>,
    /// Cached ra claim after sumcheck completes
    ra_claim: Option<F>,
}

impl<F: JoltField> HammingWeightSumcheck<F> {
    pub fn new(ra_poly: MultilinearPolynomial<F>, K: usize) -> Self {
        Self {
            K,
            prover_state: Some(HammingWeightProverState { ra_poly }),
            ra_claim: None,
        }
    }

    pub fn new_verifier(K: usize) -> Self {
        Self {
            K,
            prover_state: None,
            ra_claim: None,
        }
    }
}

impl<F: JoltField> BatchableSumcheckInstance<F> for HammingWeightSumcheck<F> {
    fn degree(&self) -> usize {
        1
    }

    fn num_rounds(&self) -> usize {
        self.K.log_2()
    }

    fn input_claim(&self) -> F {
        F::one()
    }

    fn compute_prover_message(&mut self, _round: usize) -> Vec<F> {
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");

        let univariate_poly_eval: F = (0..prover_state.ra_poly.len() / 2)
            .into_par_iter()
            .map(|i| prover_state.ra_poly.get_bound_coeff(2 * i))
            .sum();

        vec![univariate_poly_eval]
    }

    fn bind(&mut self, r_j: F, _round: usize) {
        let prover_state = self
            .prover_state
            .as_mut()
            .expect("Prover state not initialized");

        prover_state
            .ra_poly
            .bind_parallel(r_j, BindingOrder::LowToHigh)
    }

    fn expected_output_claim(&self, _: &[F]) -> F {
        self.ra_claim.expect("ra_claim not set")
    }
}

impl<F, PCS> CacheSumcheckOpenings<F, PCS> for HammingWeightSumcheck<F>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    fn cache_openings_prover(
        &mut self,
        _accumulator: Option<Rc<RefCell<ProverOpeningAccumulator<F, PCS>>>>,
        _opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        debug_assert!(self.ra_claim.is_none());
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");

        self.ra_claim = Some(prover_state.ra_poly.final_sumcheck_claim());
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone)]
pub struct HammingWeightProof<F: JoltField, ProofTranscript: Transcript> {
    sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    ra_claim: F,
}

impl<F: JoltField, ProofTranscript: Transcript> HammingWeightProof<F, ProofTranscript> {
    #[tracing::instrument(skip_all, name = "HammingWeightProof::prove")]
    pub fn prove(F: Vec<F>, K: usize, transcript: &mut ProofTranscript) -> (Self, Vec<F>) {
        let ra_poly = MultilinearPolynomial::from(F);
        let mut core_piop_sumcheck = HammingWeightSumcheck::new(ra_poly, K);

        let (sumcheck_proof, r_address) = core_piop_sumcheck.prove_single(transcript);
        // BatchedSumcheck::cache_openings(vec![&mut core_piop_sumcheck], openings, accumulator);

        let ra_claim = core_piop_sumcheck
            .ra_claim
            .expect("ra_claim should be set after prove_single");

        let proof = Self {
            sumcheck_proof,
            ra_claim,
        };

        (proof, r_address)
    }

    pub fn verify(
        &self,
        K: usize,
        transcript: &mut ProofTranscript,
    ) -> Result<Vec<F>, ProofVerifyError> {
        let mut core_piop_sumcheck = HammingWeightSumcheck::new_verifier(K);
        core_piop_sumcheck.ra_claim = Some(self.ra_claim);

        let r_address = core_piop_sumcheck.verify_single(&self.sumcheck_proof, transcript)?;

        Ok(r_address)
    }
}
