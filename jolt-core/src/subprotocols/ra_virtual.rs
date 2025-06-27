use crate::poly::multilinear_polynomial::PolynomialEvaluation;
use crate::{
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
    },
    subprotocols::sumcheck::{BatchableSumcheckInstance, SumcheckInstanceProof},
    utils::{math::Math, transcript::Transcript},
};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

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
        addresses: Vec<usize>,
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

        // Compute K^(1/D)
        let k_one_over_d = 1 << chunk_size;

        let mut eq_tables_vec = Vec::with_capacity(D);

        for chunk in r_address_chunks.iter() {
            let eq_table = EqPolynomial::evals(chunk);
            eq_tables_vec.push(eq_table);
        }

        let eq_tables: [Vec<F>; D] = eq_tables_vec
            .try_into()
            .expect("Failed to convert Vec to array for eq tables");

        // We construct ra_i directly from a list of addresses.
        // This way we avoid |ra_i| = k^(1/d) * T size, but rather just |ra_i| = T.

        let mut ra_i_vecs: Vec<Vec<F>> = (0..D).map(|_| vec![F::zero(); T]).collect();

        // For each address, decompose it and add the corresponding EQ evaluations
        for (cycle_idx, &address) in addresses.iter().enumerate() {
            // this is LSB to MSB!! (Doesn't work the other way)
            let mut remaining_address = address;
            for i in 0..D {
                let chunk_value = remaining_address % k_one_over_d;
                remaining_address /= k_one_over_d;

                ra_i_vecs[D - 1 - i][cycle_idx] += eq_tables[D - 1 - i][chunk_value];
            }
        }

        let mut ra_i_polys: Vec<MultilinearPolynomial<F>> = ra_i_vecs
            .into_iter()
            .map(|vec| MultilinearPolynomial::from(vec))
            .collect();

        let mut ra_i_polys: Vec<MultilinearPolynomial<F>> = ra_i_polys
            .into_iter()
            .map(|vec| MultilinearPolynomial::from(vec))
            .collect();

        let mut ra_i_polys: [MultilinearPolynomial<F>; D] = ra_i_polys
            .try_into()
            .expect("Failed to convert Vec to array");

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
            crate::subprotocols::sumcheck::BatchedSumcheck::prove_single(&mut self, transcript);

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

        let r_cycle_bound = crate::subprotocols::sumcheck::BatchedSumcheck::verify_single(
            sumcheck_proof,
            &verifier_sumcheck,
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
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");

        let mut openings = [F::zero(); D];
        for i in 0..D {
            openings[i] = prover_state.ra_i_polys[i].final_sumcheck_claim();
        }

        self.ra_i_claims = Some(openings);
    }

    fn bind(&mut self, r_j: F, _: usize) {
        let prover_state = self
            .prover_state
            .as_mut()
            .expect("Prover state not initialized");

        for ra_i in prover_state.ra_i_polys.iter_mut() {
            ra_i.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
        prover_state
            .eq_poly
            .bind_parallel(r_j, BindingOrder::LowToHigh);
    }

    fn input_claim(&self) -> F {
        self.ra_claim.clone()
    }

    fn expected_output_claim(&self, r: &[F]) -> F {
        let verifier_state = self
            .verifier_state
            .as_ref()
            .expect("Verifier state not initialized");
        let ra_i_claims = self.ra_i_claims.as_ref().expect("ra_i_claims not set");

        // eq(r_cycle, r_cycle_bound)
        let eq_eval = EqPolynomial::new(verifier_state.r_cycle.clone()).evaluate(r);

        // Compute the product of all ra_i evaluations
        let mut product = F::one();
        for ra_i_claim in ra_i_claims.iter() {
            product *= *ra_i_claim;
        }

        eq_eval * product
    }

    fn compute_prover_message(&self, _round: usize) -> Vec<F> {
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");
        let degree = <Self as BatchableSumcheckInstance<F, ProofTranscript>>::degree(self);
        let ra_i_polys = &prover_state.ra_i_polys;
        let eq_poly = &prover_state.eq_poly;

        // We need to compute evaluations at 0, 2, 3, ..., degree
        // = eq(r_cycle, j) * ∏_{i=0}^{D-1} ra_i(j)

        let univariate_poly_evals: Vec<F> = (0..ra_i_polys[0].len() / 2)
            .into_par_iter()
            .map(|i| {
                let eq_evals = eq_poly.sumcheck_evals(i, degree, BindingOrder::LowToHigh);

                let mut evals = vec![F::zero(); degree];

                for eval_point in 0..degree {
                    if eval_point == 1 {
                        continue;
                    }

                    let mut result = eq_evals[eval_point];

                    for ra_i_poly in ra_i_polys.iter() {
                        let ra_i_evals =
                            ra_i_poly.sumcheck_evals(i, degree, BindingOrder::LowToHigh);
                        result *= ra_i_evals[eval_point];
                    }

                    evals[eval_point] = result;
                }

                evals
            })
            .reduce(
                || vec![F::zero(); degree],
                |mut running, new| {
                    for i in 0..degree {
                        running[i] += new[i];
                    }
                    running
                },
            );

        univariate_poly_evals
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::transcript::KeccakTranscript;
    use ark_bn254::Fr;
    use ark_std::{rand::Rng, test_rng, One, Zero};
    use rand::thread_rng;

    // #[test]
    // fn test_ra_sumcheck_with_specific_memory() {
    //     // k = 8
    //     // d = 3
    //     // T = 1
    //     // RA polynomial: multilinear extension of 8-length vector with 3rd index (0-indexed) = 1
    //     // RA_i vectors: [0, 1], [1, 0], [1, 0]

    //     let mut rng = test_rng();
    //     const D: usize = 3;
    //     let T = 1;
    //     let k = 8;

    //     let mut ra_values = vec![Fr::zero(); k];
    //     ra_values[3] = Fr::one(); // [0, 0, 0, 1, 0, 0, 0, 0]
    //     let ra_poly = MultilinearPolynomial::from(ra_values);
    //     println!("ra poly: {:?}", ra_poly);

    //     // Create addresses vector - only address 3 has value 1
    //     let addresses = vec![3];

    //     let r_cycle: Vec<Fr> = (0..T.log_2()).map(|_| Fr::zero()).collect();
    //     let r_address: Vec<Fr> = (0..D)
    //         .map(|_| {
    //             let addr = rng.gen::<u64>() % (k as u64);
    //             Fr::from(addr)
    //         })
    //         .collect();

    //     // let r_address = vec![Fr::from(4), Fr::from(4), Fr::from(1)];

    //     let mut eval_point = r_cycle.clone();
    //     eval_point.extend_from_slice(&r_address);
    //     println!("eval point: {:?}", eval_point);
    //     let ra_claim = ra_poly.evaluate(&eval_point);
    //     println!("ra claim: {:?}", ra_claim);

    //     let prover_sumcheck = RASumcheck::<Fr, D>::new(
    //         ra_claim,
    //         addresses.clone(),
    //         r_cycle.clone(),
    //         r_address.clone(),
    //         T,
    //     );

    //     let mut prover_transcript = KeccakTranscript::new(b"test_ra_sumcheck");
    //     let (proof, r_cycle_bound) = prover_sumcheck.prove(&mut prover_transcript);

    //     let mut verifier_transcript = KeccakTranscript::new(b"test_ra_sumcheck");

    //     let chunk_size = r_address.len() / D;
    //     let mut r_address_chunks_vec = Vec::with_capacity(D);
    //     for i in 0..D {
    //         let start = i * chunk_size;
    //         let end = (i + 1) * chunk_size;
    //         r_address_chunks_vec.push(r_address[start..end].to_vec());
    //     }
    //     let r_address_chunks: [Vec<Fr>; D] = r_address_chunks_vec
    //         .try_into()
    //         .expect("Failed to convert Vec to array");

    //     let verify_result = RASumcheck::<Fr, D>::verify(
    //         ra_claim,
    //         proof.ra_i_claims,
    //         r_cycle,
    //         r_address_chunks,
    //         T,
    //         &proof.sumcheck_proof,
    //         &mut verifier_transcript,
    //     );

    //     assert!(verify_result.is_ok(), "Verification failed");
    //     let verified_r_cycle_bound = verify_result.unwrap();
    //     assert_eq!(
    //         r_cycle_bound, verified_r_cycle_bound,
    //         "r_cycle_bound mismatch"
    //     );
    // }

    #[test]
    fn test_ra_sumcheck_with_correct_tensor_decomposition() {
        use rand::Rng;
        // Test with random one-hot RA and correct tensor decomposition
        let mut rng = thread_rng();
        const D: usize = 3;
        let T = 1;
        let k = 8; // 2^3 = 8

        // Generate random index for the one-hot position
        let one_hot_index = rng.gen::<usize>() % k;

        // Create one-hot RA vector
        let mut ra_values = vec![Fr::zero(); k];
        ra_values[one_hot_index] = Fr::one();
        let ra_poly = MultilinearPolynomial::from(ra_values);

        // Create addresses vector - only one_hot_index has value 1
        let addresses = vec![one_hot_index];

        // Decompose the index: index = c0*2^0 + c1*2^1 + c2*2^2
        // For index 3: 3 = 1*2^0 + 1*2^1 + 0*2^2 = 1 + 2 + 0
        let c0 = one_hot_index & 1; // LSB
        let c1 = (one_hot_index >> 1) & 1; // Middle bit
        let c2 = (one_hot_index >> 2) & 1; // MSB

        println!("One-hot index: {}", one_hot_index);
        println!("Decomposition: c0={}, c1={}, c2={}", c0, c1, c2);
        println!(
            "Verification: {} = {}*1 + {}*2 + {}*4",
            one_hot_index, c0, c1, c2
        );

        // Generate random evaluation points
        let r_cycle: Vec<Fr> = (0..T.log_2()).map(|_| Fr::from(rng.gen::<u64>())).collect();
        let r_address: Vec<Fr> = (0..D).map(|_| Fr::from(rng.gen::<u64>())).collect();

        // Evaluate RA at the random point
        let mut eval_point = r_cycle.clone();
        eval_point.extend_from_slice(&r_address);
        let ra_claim = ra_poly.evaluate(&eval_point);

        // Create and run prover
        let prover_sumcheck =
            RASumcheck::<Fr, D>::new(ra_claim, addresses, r_cycle.clone(), r_address.clone(), T);

        let mut prover_transcript = KeccakTranscript::new(b"test_correct_tensor");
        let (proof, r_cycle_bound) = prover_sumcheck.prove(&mut prover_transcript);

        // Verify the proof
        let mut verifier_transcript = KeccakTranscript::new(b"test_correct_tensor");

        let chunk_size = r_address.len() / D;
        let mut r_address_chunks_vec = Vec::with_capacity(D);
        for i in 0..D {
            let start = i * chunk_size;
            let end = (i + 1) * chunk_size;
            r_address_chunks_vec.push(r_address[start..end].to_vec());
        }
        let r_address_chunks: [Vec<Fr>; D] = r_address_chunks_vec
            .try_into()
            .expect("Failed to convert Vec to array");

        let verify_result = RASumcheck::<Fr, D>::verify(
            ra_claim,
            proof.ra_i_claims,
            r_cycle,
            r_address_chunks,
            T,
            &proof.sumcheck_proof,
            &mut verifier_transcript,
        );

        assert!(verify_result.is_ok(), "Verification failed");
        let verified_r_cycle_bound = verify_result.unwrap();
        assert_eq!(
            r_cycle_bound, verified_r_cycle_bound,
            "r_cycle_bound mismatch"
        );
    }
}
