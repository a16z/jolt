use crate::{
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
    },
    subprotocols::sumcheck::{BatchableSumcheckInstance, SumcheckInstanceProof},
    utils::{math::Math, transcript::Transcript},
};

use crate::poly::multilinear_polynomial::PolynomialEvaluation;
pub struct RAProof<F: JoltField, ProofTranscript: Transcript> {
    pub ra_i_claims: Vec<F>,
    pub sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
}

pub struct RAProverState<F: JoltField, const D: usize> {
    /// ra polynomials for each chunk
    ra_i_polys: [MultilinearPolynomial<F>; D],
    /// Eq polynomial as a multilinear polynomial
    eq_poly: MultilinearPolynomial<F>,
    /// Length of the trace
    T: usize,
}

pub struct RAVerifierState<F: JoltField, const D: usize> {
    /// Random point r_cycle
    r_cycle: Vec<F>,
    /// Random points r_address^(i) for each chunk
    r_address_chunks: [Vec<F>; D],
    /// Length of the trace
    T: usize,
}

pub struct RASumcheck<F: JoltField, const D: usize> {
    /// ra(r_cycle, r_address)
    ra_claim: F,
    /// Prover state
    prover_state: Option<RAProverState<F, D>>,
    /// Verifier state
    verifier_state: Option<RAVerifierState<F, D>>,
    /// ra_i_ claims to be proved via evaluation proof
    ra_i_claims: Option<[F; D]>,
}

impl<F: JoltField, const D: usize> RASumcheck<F, D> {
    pub fn new(
        ra_claim: F,
        mut ra_i_polys: [MultilinearPolynomial<F>; D],
        r_cycle: Vec<F>,
        r_address: Vec<F>,
        T: usize,
    ) -> Self {
        assert_eq!(
            r_address.len() % D,
            0,
            "r_address length must be divisible by D"
        );

        let chunk_size = r_address.len() / D;
        let mut r_address_chunks_vec = Vec::with_capacity(D);

        for i in 0..D {
            let start = i * chunk_size;
            let end = (i + 1) * chunk_size;
            r_address_chunks_vec.push(r_address[start..end].to_vec());
        }

        let r_address_chunks: [Vec<F>; D] = r_address_chunks_vec
            .try_into()
            .expect("Failed to convert Vec to array");

        let eq_evals = EqPolynomial::evals(&r_cycle);

        let eq_poly = MultilinearPolynomial::from(eq_evals);

        // We do pre-binding. The sumcheck is in variable j hence we can pre-bind the r_address chunks:
        for (poly, chunk) in ra_i_polys.iter_mut().zip(r_address_chunks.iter()) {
            for &r_value in chunk.iter() {
                poly.bind_parallel(r_value, BindingOrder::LowToHigh);
            }
        }

        let prover_state = RAProverState {
            ra_i_polys,
            eq_poly,
            T,
        };

        Self {
            ra_claim,
            prover_state: Some(prover_state),
            verifier_state: None,
            ra_i_claims: None,
        }
    }

    pub fn new_verifier(
        ra_claim: F,
        r_cycle: Vec<F>,
        r_address_chunks: [Vec<F>; D],
        T: usize,
    ) -> Self {
        let verifier_state = RAVerifierState {
            r_cycle,
            r_address_chunks,
            T,
        };

        Self {
            ra_claim,
            prover_state: None,
            verifier_state: Some(verifier_state),
            ra_i_claims: None,
        }
    }

    pub fn prove<ProofTranscript: Transcript>(
        mut self,
        transcript: &mut ProofTranscript,
    ) -> (RAProof<F, ProofTranscript>, Vec<F>) {
        let (sumcheck_proof, r_cycle_bound) =
            crate::subprotocols::sumcheck::BatchedSumcheck::prove(vec![&mut self], transcript);

        let ra_i_claims = self
            .ra_i_claims
            .expect("ra_i_claims were not set after prove")
            .to_vec();

        let proof = RAProof {
            sumcheck_proof,
            ra_i_claims,
        };

        (proof, r_cycle_bound)
    }

    pub fn verify<ProofTranscript: Transcript>(
        ra_claim: F,
        ra_i_claims: Vec<F>,
        r_cycle: Vec<F>,
        r_address_chunks: [Vec<F>; D],
        T: usize,
        sumcheck_proof: &SumcheckInstanceProof<F, ProofTranscript>,
        transcript: &mut ProofTranscript,
    ) -> Result<Vec<F>, crate::utils::errors::ProofVerifyError> {
        let mut verifier_sumcheck = Self::new_verifier(ra_claim, r_cycle, r_address_chunks, T);
      
        let ra_i_claims_array: [F; D] = ra_i_claims
            .try_into()
            .map_err(|_| crate::utils::errors::ProofVerifyError::InternalError)?;
        verifier_sumcheck.ra_i_claims = Some(ra_i_claims_array);

        let r_cycle_bound = crate::subprotocols::sumcheck::BatchedSumcheck::verify(
            sumcheck_proof,
            vec![&verifier_sumcheck],
            transcript,
        )?;

        Ok(r_cycle_bound)
    }
}

impl<F: JoltField, ProofTranscript: Transcript, const D: usize>
    BatchableSumcheckInstance<F, ProofTranscript> for RASumcheck<F, D>
{
    fn degree(&self) -> usize {
        D + 1
    }

    fn num_rounds(&self) -> usize {
        if self.prover_state.is_some() {
            self.prover_state.as_ref().unwrap().T.log_2()
        } else if self.verifier_state.is_some() {
            self.verifier_state.as_ref().unwrap().T.log_2()
        } else {
            panic!("Neither prover state nor verifier state is initialized");
        }
    }

    fn cache_openings(&mut self) {
        debug_assert!(self.ra_i_claims.is_none());
        let prover_state = self.prover_state.as_ref().expect("Prover state not initialized");
        
        let mut openings = [F::zero(); D];
        for i in 0..D {
            openings[i] = prover_state.ra_i_polys[i].final_sumcheck_claim();
        }

        self.ra_i_claims = Some(openings);
    }

    fn bind(&mut self, r_j: F, _: usize) {
        let prover_state = self.prover_state.as_mut().expect("Prover state not initialized");
        
        for ra_i in prover_state.ra_i_polys.iter_mut() {
            ra_i.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
        prover_state.eq_poly.bind_parallel(r_j, BindingOrder::LowToHigh);
    }

    fn input_claim(&self) -> F {
        self.ra_claim.clone()
    }

    fn expected_output_claim(&self, r: &[F]) -> F {
        // This is called on the verifier side
        let verifier_state = self.verifier_state.as_ref().expect("Verifier state not initialized");
        let ra_i_claims = self.ra_i_claims.as_ref().expect("ra_i_claims not set");
        
        // Compute eq(r_cycle, r_cycle_bound)
        let eq_eval = EqPolynomial::new(verifier_state.r_cycle.clone()).evaluate(r);

        // Compute the product of all ra_i evaluations
        let mut product = F::one();
        for ra_i_claim in ra_i_claims.iter() {
            product *= *ra_i_claim;
        }

        eq_eval * product
    }

    fn compute_prover_message(&self, round: usize) -> Vec<F> {
        let prover_state = self.prover_state.as_ref().expect("Prover state not initialized");
        let degree = <Self as BatchableSumcheckInstance<F, ProofTranscript>>::degree(self);
        let ra_i_polys = &prover_state.ra_i_polys;
        let eq_poly = &prover_state.eq_poly;

        // We need to compute evaluations at 0, 2, 3, ..., degree
        // = eq(r_cycle, j) * ∏_{i=0}^{D-1} ra_i(j)

        let eval_points: Vec<usize> = (0..=degree).filter(|&i| i != 1).collect();

        let mut evals = vec![F::zero(); eval_points.len()];

        let remaining_vars =
            <Self as BatchableSumcheckInstance<F, ProofTranscript>>::num_rounds(&self) - round - 1;

        for (eval_idx, &point) in eval_points.iter().enumerate() {
            for j in 0..(1 << remaining_vars) {
                let eq_evals = eq_poly.sumcheck_evals(j, degree, BindingOrder::LowToHigh);
                // Extract the evaluation at the current point
                let eq_eval = if point == 0 {
                    eq_evals[0]
                } else {
                    eq_evals[point - 1]
                };

                // Compute ∏_{i=0}^{D-1} ra_i(j) evaluated at point
                let mut ra_product = F::one();
                for ra_i_poly in ra_i_polys.iter() {
                    // Get sumcheck evaluations for this polynomial at index j
                    let ra_i_evals = ra_i_poly.sumcheck_evals(j, degree, BindingOrder::LowToHigh);
                    // The evaluation at point k is at index k (0-indexed for points 0, 2, 3, ...)
                    let ra_i_eval = if point == 0 {
                        ra_i_evals[0]
                    } else {
                        ra_i_evals[point - 1]
                    };
                    ra_product *= ra_i_eval;
                }

                // Add to the sum: eq(r_cycle, j) * ∏_{i=0}^{D-1} ra_i(j)
                evals[eval_idx] += eq_eval * ra_product;
            }
        }

        evals
    }
}
