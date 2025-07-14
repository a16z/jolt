use std::{cell::RefCell, rc::Rc};

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use rayon::prelude::*;

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
    utils::transcript::Transcript,
};

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone)]
pub struct HammingWeightProof<F, ProofTranscript>
where
    F: JoltField,
    ProofTranscript: Transcript,
{
    sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    ra_claims: Vec<F>,
}

pub struct HammingWeightProverState<F: JoltField> {
    /// The ra polynomials - one for each decomposed part
    ra: Vec<MultilinearPolynomial<F>>,
    /// z powers for batching
    z_powers: Vec<F>,
    /// D parameter as in Twist and Shout paper
    d: usize,
}

pub struct HammingWeightVerifierState<F: JoltField> {
    /// log K (number of rounds)
    log_K: usize,
    /// D parameter as in Twist and Shout paper
    d: usize,
    /// z powers for verification
    z_powers: Vec<F>,
}

pub struct HammingWeightSumcheck<F: JoltField> {
    /// The initial claim (sum of z powers for hamming weight)
    input_claim: F,
    /// Prover state
    prover_state: Option<HammingWeightProverState<F>>,
    /// Verifier state
    verifier_state: Option<HammingWeightVerifierState<F>>,
    /// Cached claims for all d polynomials
    cached_claims: Option<Vec<F>>,
    /// D parameter
    d: usize,
}

impl<F: JoltField> HammingWeightSumcheck<F> {
    fn new_prover(ra: Vec<MultilinearPolynomial<F>>, z_powers: Vec<F>, d: usize) -> Self {
        // Compute input claim as sum of z powers
        let input_claim = z_powers.iter().sum();

        Self {
            input_claim,
            prover_state: Some(HammingWeightProverState { ra, z_powers, d }),
            verifier_state: None,
            cached_claims: None,
            d,
        }
    }

    fn new_verifier(log_K: usize, ra_claims: Vec<F>, z_powers: Vec<F>, d: usize) -> Self {
        // Compute input claim as sum of z powers
        let input_claim = z_powers.iter().sum();

        Self {
            input_claim,
            prover_state: None,
            verifier_state: Some(HammingWeightVerifierState { log_K, d, z_powers }),
            cached_claims: Some(ra_claims),
            d,
        }
    }
}

impl<F: JoltField> BatchableSumcheckInstance<F> for HammingWeightSumcheck<F> {
    fn degree(&self) -> usize {
        1
    }

    fn num_rounds(&self) -> usize {
        if let Some(prover_state) = &self.prover_state {
            prover_state.ra[0].get_num_vars()
        } else if let Some(verifier_state) = &self.verifier_state {
            verifier_state.log_K
        } else {
            panic!("Neither prover state nor verifier state is initialized")
        }
    }

    fn input_claim(&self) -> F {
        self.input_claim
    }

    fn compute_prover_message(&mut self, _round: usize) -> Vec<F> {
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");

        let univariate_poly_eval: F = prover_state
            .ra
            .par_iter()
            .zip(prover_state.z_powers.par_iter())
            .map(|(ra_poly, z_power)| {
                let sum: F = (0..ra_poly.len() / 2)
                    .into_par_iter()
                    .map(|i| ra_poly.get_bound_coeff(2 * i))
                    .sum();
                sum * z_power
            })
            .sum();

        vec![univariate_poly_eval]
    }

    fn bind(&mut self, r_j: F, _round: usize) {
        if let Some(prover_state) = &mut self.prover_state {
            prover_state
                .ra
                .par_iter_mut()
                .for_each(|ra_poly| ra_poly.bind_parallel(r_j, BindingOrder::LowToHigh));
        }
    }

    fn expected_output_claim(&self, _r: &[F]) -> F {
        let verifier_state = self
            .verifier_state
            .as_ref()
            .expect("Verifier state not initialized");
        let ra_claims = self.cached_claims.as_ref().expect("RA claims not cached");

        // Compute batched claim: sum_{i=0}^{d-1} z^i * ra_i
        ra_claims
            .iter()
            .zip(verifier_state.z_powers.iter())
            .map(|(ra_claim, z_power)| *ra_claim * z_power)
            .sum()
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
        debug_assert!(self.cached_claims.is_none());
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");

        let claims: Vec<F> = prover_state
            .ra
            .iter()
            .map(|ra_poly| ra_poly.final_sumcheck_claim())
            .collect();
        self.cached_claims = Some(claims);
    }
}
