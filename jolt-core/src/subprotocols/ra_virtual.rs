use crate::{
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        unipoly::UniPoly,
    },
    subprotocols::sumcheck::{BatchableSumcheckInstance, SumcheckInstanceProof},
    utils::{errors::ProofVerifyError, transcript::Transcript},
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use rayon::prelude::*;

/// Represents the virtual RA sumcheck for d-way chunked addresses
///
/// Where:
/// - T is the trace length
/// - d is the number of address chunks (const generic)
/// - r_address^(i) is the i-th chunk of r_address
/// - ra_i is the read-address polynomial for chunk i
pub struct VirtualRASumcheck<F: JoltField, const D: usize> {
    /// The individual RA polynomials for each chunk
    /// Each ra_i is a polynomial in (cycle, address_chunk) variables
    ra_i_polys: [MultilinearPolynomial<F>; D],

    /// The precomputed evaluations of eq(r_cycle, j)
    eq_evals: Vec<F>,

    /// The random point r_cycle at which we're evaluating
    r_cycle: Vec<F>,

    /// The random points r_address^(i) for each chunk
    r_address_chunks: [Vec<F>; D],

    /// Current round index during sumcheck
    current_round: usize,

    /// Number of cycle variables (log T)
    num_cycle_vars: usize,

    /// Cached openings to be proven later
    cached_openings: Option<[F; D]>, //TODO(markosg04): fix this

    /// Current binding of the eq polynomial evaluations
    current_eq_evals: Vec<F>,
}

impl<F: JoltField, const D: usize> VirtualRASumcheck<F, D> {
    pub fn new(
        ra_i_polys: [MultilinearPolynomial<F>; D],
        r_cycle: Vec<F>,
        r_address_chunks: [Vec<F>; D],
    ) -> Self {
        let num_cycle_vars = r_cycle.len();
        // Precompute eq(r_cycle, *) evaluations
        let eq_evals = EqPolynomial::evals(&r_cycle);
        let current_eq_evals = eq_evals.clone();

        Self {
            ra_i_polys,
            eq_evals,
            r_cycle,
            r_address_chunks,
            current_round: 0,
            num_cycle_vars,
            cached_openings: None,
            current_eq_evals,
        }
    }

    /// Compute the univariate polynomial for the current round of sumcheck
    fn compute_round_polynomial(&self) -> UniPoly<F> {
        let half_len = 1 << (self.num_cycle_vars - self.current_round - 1);

        // We need to compute the sum over the remaining variables of:
        // eq(r_cycle[..current_round] || X || remaining_vars, j) * _i ra_i(j, r_address^(i))

        // The degree is D+1 (D from the product of ra_i's, plus 1 from eq)
        let degree = D + 1;

        // Evaluate at points 0, 1, 2, ..., degree
        let evals: Vec<F> = (0..=degree)
            .into_par_iter()
            .map(|eval_point| {
                let point = F::from_u64(eval_point as u64);
                let mut sum = F::zero();

                // Sum over all possible values of the remaining variables
                for k in 0..half_len {
                    // Compute eq polynomial contribution using current bindings
                    // The eq polynomial is bound as we go through sumcheck rounds
                    let eq_contrib = if eval_point == 0 {
                        self.current_eq_evals[k]
                    } else if eval_point == 1 {
                        self.current_eq_evals[k + half_len]
                    } else {
                        // For higher degree evaluations, interpolate
                        let eq_0 = self.current_eq_evals[k];
                        let eq_1 = self.current_eq_evals[k + half_len];
                        eq_0 + point * (eq_1 - eq_0)
                    };

                    // Compute product of ra_i evaluations
                    let mut ra_product = F::one();
                    for i in 0..D {
                        // For the current round, we need to evaluate ra_i at:
                        // - Previously bound cycle variables (from earlier rounds)
                        // - Current variable bound to `point`
                        // - Remaining cycle variables fixed by k
                        // - Address variables fixed to r_address_chunks[i]

                        // The ra_i polynomials have already been bound for previous rounds
                        // Now we evaluate at the current point
                        let ra_eval = if eval_point == 0 {
                            // Use the coefficient at index k (low half)
                            self.ra_i_polys[i].get_bound_coeff(k)
                        } else if eval_point == 1 {
                            // Use the coefficient at index k + half_len (high half)
                            self.ra_i_polys[i].get_bound_coeff(k + half_len)
                        } else {
                            // Interpolate for higher degree points
                            let low = self.ra_i_polys[i].get_bound_coeff(k);
                            let high = self.ra_i_polys[i].get_bound_coeff(k + half_len);
                            low + point * (high - low)
                        };
                        ra_product *= ra_eval;
                    }

                    sum += eq_contrib * ra_product;
                }

                sum
            })
            .collect();

        UniPoly::from_evals(&evals)
    }
}

impl<F: JoltField, ProofTranscript: Transcript, const D: usize>
    BatchableSumcheckInstance<F, ProofTranscript> for VirtualRASumcheck<F, D>
{
    fn degree(&self) -> usize {
        D + 1 // D from product of ra_i's, plus 1 from eq polynomial
    }

    fn num_rounds(&self) -> usize {
        self.num_cycle_vars
    }

    fn input_claim(&self) -> F {
        // The claim is the evaluation of the full sum
        // Since we're proving the sumcheck for the cycle variables,
        // the claim is: _j eq(r_cycle, j) * _i ra_i(j, r_address^(i))

        let mut sum = F::zero();
        let num_points = 1 << self.num_cycle_vars;

        for j in 0..num_points {
            // Use precomputed eq evaluations
            let eq_eval = self.eq_evals[j];

            let mut ra_product = F::one();
            for i in 0..D {
                // The ra_i polynomial is over (cycle, address) variables
                // We need to evaluate it at cycle index j with address already bound to r_address_chunks[i]
                // The address variables have been pre-bound, so we use get_bound_coeff
                let ra_eval = self.ra_i_polys[i].get_bound_coeff(j);
                ra_product *= ra_eval;
            }

            sum += eq_eval * ra_product;
        }

        sum
    }

    fn compute_prover_message(&self, _round: usize) -> Vec<F> {
        // Update current round (this is a bit awkward due to the trait design)
        // In practice, you might want to use interior mutability or a different approach
        let poly = self.compute_round_polynomial();

        // Return evaluations at 0, 2, 3, ..., degree
        // (evaluation at 1 is computed from the claim)
        let degree = D + 1; // We know the degree is D+1
        let mut evals = vec![];
        evals.push(poly.evaluate(&F::zero()));
        for i in 2..=degree {
            evals.push(poly.evaluate(&F::from_u64(i as u64)));
        }

        evals
    }

    fn bind(&mut self, r_j: F, round: usize) {
        assert_eq!(round, self.current_round);

        // Update the eq polynomial evaluations
        // Similar to how ExpandingTable works in sparse_dense_shout.rs
        let len = self.current_eq_evals.len() / 2;
        let mut new_evals = vec![F::zero(); len];

        for i in 0..len {
            let eq_0 = self.current_eq_evals[i];
            let eq_1 = self.current_eq_evals[i + len];
            new_evals[i] = eq_0 + r_j * (eq_1 - eq_0);
        }

        self.current_eq_evals = new_evals;

        // Bind each ra polynomial's cycle variable
        for i in 0..D {
            self.ra_i_polys[i].bind(r_j, BindingOrder::HighToLow);
        }

        self.current_round += 1;
    }

    fn cache_openings(&mut self) {
        // After all cycle variables are bound, we need to cache the openings
        // of each ra_i at the bound point
        let mut openings = [F::zero(); D];

        for i in 0..D {
            // The ra_i polynomials were pre-bound to address variables and then
            // all cycle variables were bound during sumcheck. 
            // Now they should be constant polynomials.
            if self.ra_i_polys[i].get_num_vars() == 0 {
                // Polynomial is fully bound, get the constant value
                openings[i] = self.ra_i_polys[i].get_bound_coeff(0);
            } else {
                // This shouldn't happen in our case, but handle it anyway
                openings[i] = self.ra_i_polys[i].evaluate(&[]);
            }
        }

        self.cached_openings = Some(openings);
    }

    fn expected_output_claim(&self, r: &[F]) -> F {
        // After binding all cycle variables with r, the claim should be:
        // eq(r_cycle, r) * _i ra_i(r, r_address^(i))

        let eq_poly = EqPolynomial::new(self.r_cycle.clone());
        let eq_eval = eq_poly.evaluate(r);

        let mut ra_product = F::one();
        for i in 0..D {
            // The ra_i polynomials were pre-bound to address variables and then
            // all cycle variables were bound during sumcheck to r.
            // Now they should be constant polynomials.
            let ra_eval = if self.ra_i_polys[i].get_num_vars() == 0 {
                self.ra_i_polys[i].get_bound_coeff(0)
            } else {
                // This shouldn't happen in our case
                self.ra_i_polys[i].evaluate(&[])
            };
            ra_product *= ra_eval;
        }

        eq_eval * ra_product
    }
}

/// Proof structure for the virtual RA sumcheck
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct VirtualRAProof<F: JoltField, ProofTranscript: Transcript> {
    /// The sumcheck proof for the cycle variables
    pub sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,

    /// The claimed evaluations of each ra_i polynomial
    pub ra_claims: Vec<F>,
}

impl<F: JoltField, const D: usize> VirtualRASumcheck<F, D> {
    /// Prove the virtual RA sumcheck
    pub fn prove<ProofTranscript: Transcript>(
        mut self,
        transcript: &mut ProofTranscript,
    ) -> (VirtualRAProof<F, ProofTranscript>, Vec<F>) {
        // Use the batched sumcheck framework
        let (sumcheck_proof, r_cycle_bound) = crate::subprotocols::sumcheck::BatchedSumcheck::prove(
            vec![&mut self],
            transcript,
        );

        // Extract the cached openings
        let ra_claims = if let Some(openings) = self.cached_openings {
            openings.to_vec()
        } else {
            panic!("Openings not cached");
        };

        let proof = VirtualRAProof {
            sumcheck_proof,
            ra_claims,
        };

        (proof, r_cycle_bound)
    }

    /// Verify the virtual RA sumcheck
    pub fn verify<ProofTranscript: Transcript>(
        proof: &VirtualRAProof<F, ProofTranscript>,
        claim: F,
        r_cycle: &[F],
        r_address_chunks: &[Vec<F>; D],
        transcript: &mut ProofTranscript,
    ) -> Result<Vec<F>, ProofVerifyError> {
        // Create a verifier instance (simplified version without full polynomials)
        let verifier_instance = VirtualRASumcheckVerifier::<F, D> {
            r_cycle: r_cycle.to_vec(),
            _r_address_chunks: r_address_chunks.clone(),
            ra_claims: proof.ra_claims.clone(),
            claim,
        };

        let instances: Vec<&dyn BatchableSumcheckInstance<F, ProofTranscript>> =
            vec![&verifier_instance];

        crate::subprotocols::sumcheck::BatchedSumcheck::verify(
            &proof.sumcheck_proof,
            instances,
            transcript,
        )
    }
}

/// Verifier-side instance for the virtual RA sumcheck
struct VirtualRASumcheckVerifier<F: JoltField, const D: usize> {
    r_cycle: Vec<F>,
    _r_address_chunks: [Vec<F>; D],
    ra_claims: Vec<F>,
    claim: F,
}

impl<F: JoltField, ProofTranscript: Transcript, const D: usize>
    BatchableSumcheckInstance<F, ProofTranscript> for VirtualRASumcheckVerifier<F, D>
{
    fn degree(&self) -> usize {
        D + 1
    }

    fn num_rounds(&self) -> usize {
        self.r_cycle.len()
    }

    fn input_claim(&self) -> F {
        // Return the claim provided by the prover
        self.claim
    }

    fn compute_prover_message(&self, _round: usize) -> Vec<F> {
        unimplemented!("Verifier doesn't compute prover messages")
    }

    fn bind(&mut self, _r_j: F, _round: usize) {
        // Verifier doesn't need to bind
    }

    fn cache_openings(&mut self) {
        // Verifier doesn't cache openings
    }

    fn expected_output_claim(&self, r: &[F]) -> F {
        // Compute eq(r_cycle, r) * _i ra_claims[i]
        let eq_poly = EqPolynomial::new(self.r_cycle.clone());
        let eq_eval = eq_poly.evaluate(r);

        let mut ra_product = F::one();
        for i in 0..D {
            ra_product *= self.ra_claims[i];
        }

        eq_eval * ra_product
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{poly::dense_mlpoly::DensePolynomial, utils::transcript::KeccakTranscript};
    use ark_bn254::Fr;
    use ark_std::{One, Zero, test_rng};

    type F = Fr;
    type ProofTranscript = KeccakTranscript;

    /// Helper function to create one-hot ra_i polynomials  
    /// Given an address decomposed into D chunks, each of B bits,
    /// creates D polynomials where ra_i represents the one-hot encoding
    /// of the i-th chunk of the address at the given cycle.
    /// 
    /// Each ra_i polynomial is initially over all (cycle, address) variables,
    /// but we need to pre-bind it to the specific address chunk evaluation point.
    fn create_one_hot_ra_polys<const D: usize>(
        num_cycle_vars: usize,
        chunk_bits: usize,
        cycle: usize,
        address_chunks: &[usize; D],
        r_address_chunks: &[Vec<F>; D],
    ) -> [MultilinearPolynomial<F>; D] {
        let num_cycles = 1 << num_cycle_vars;
        let chunk_size = 1 << chunk_bits;
        
        // Create D polynomials
        let mut ra_polys = Vec::with_capacity(D);
        
        for i in 0..D {
            // For ra_i, we need num_cycles * chunk_size values
            // The polynomial is over (cycle, address_chunk) variables
            let mut values = vec![F::zero(); num_cycles * chunk_size];
            
            // Set the one-hot value at the specified cycle and address chunk
            // The index is: cycle * chunk_size + address_chunk
            let index = cycle * chunk_size + address_chunks[i];
            values[index] = F::one();
            
            // Create the polynomial
            let mut poly = MultilinearPolynomial::LargeScalars(DensePolynomial::new(values));
            
            // Pre-bind the address variables to r_address_chunks[i]
            // The polynomial has num_cycle_vars + chunk_bits variables total
            // We bind the last chunk_bits variables (the address part)
            for j in 0..chunk_bits {
                poly.bind(r_address_chunks[i][j], BindingOrder::HighToLow);
            }
            
            ra_polys.push(poly);
        }
        
        // Convert Vec to array
        ra_polys.try_into().unwrap()
    }

    #[test]
    fn test_virtual_ra_sumcheck_prove_verify() {
        // Test with D=2 (2-way chunked addresses)
        const D: usize = 2;
        let mut rng = test_rng();
        
        // Setup parameters
        let num_cycle_vars = 3; // 8 cycles
        let chunk_bits = 2; // 4 possible values per chunk (2 bits)
        
        // Simulate a read at cycle 5 to address 10 (binary: 1010)
        // With 2-bit chunks: chunk_0 = 10 (decimal 2), chunk_1 = 10 (decimal 2)
        let read_cycle = 5;
        let address_chunks = [2, 2]; // Both chunks are 2
        
        // Choose random evaluation points
        let r_cycle: Vec<F> = (0..num_cycle_vars).map(|_| F::random(&mut rng)).collect();
        let r_address_chunks: [Vec<F>; D] = [
            (0..chunk_bits).map(|_| F::random(&mut rng)).collect(),
            (0..chunk_bits).map(|_| F::random(&mut rng)).collect(),
        ];
        
        // Create one-hot ra_i polynomials
        let ra_polys = create_one_hot_ra_polys::<D>(
            num_cycle_vars,
            chunk_bits,
            read_cycle,
            &address_chunks,
            &r_address_chunks,
        );
        
        // Create the sumcheck instance
        let sumcheck = VirtualRASumcheck::<F, D>::new(
            ra_polys,
            r_cycle.clone(),
            r_address_chunks.clone(),
        );
        
        // Compute the expected claim
        let claim = <VirtualRASumcheck<F, D> as BatchableSumcheckInstance<F, ProofTranscript>>::input_claim(&sumcheck);
        
        // Create transcripts for prover and verifier
        let mut prover_transcript = ProofTranscript::new(b"test_virtual_ra_prove_verify");
        let mut verifier_transcript = ProofTranscript::new(b"test_virtual_ra_prove_verify");
        
        // Prove
        let (proof, r_cycle_bound) = sumcheck.prove(&mut prover_transcript);
        
        // Verify
        let result = VirtualRASumcheck::<F, D>::verify(
            &proof,
            claim,
            &r_cycle,
            &r_address_chunks,
            &mut verifier_transcript,
        );
        
        // Check that verification succeeded
        assert!(result.is_ok());
        let verifier_r_cycle_bound = result.unwrap();
        
        // Check that prover and verifier agree on the bound cycle point
        assert_eq!(r_cycle_bound, verifier_r_cycle_bound);
        
        // Verify the ra_claims match what we expect
        // Since we have one-hot polynomials, the evaluation at the random points
        // should be the multilinear extension of the one-hot encoding
        assert_eq!(proof.ra_claims.len(), D);
    }

    #[test]
    fn test_virtual_ra_sumcheck_multiple_reads() {
        // Test with D=3 (3-way chunked addresses) and multiple non-zero reads
        const D: usize = 3;
        let mut rng = test_rng();
        
        // Setup parameters
        let num_cycle_vars = 2; // 4 cycles
        let chunk_bits = 2; // 4 possible values per chunk
        
        // Create ra_i polynomials with multiple reads:
        // Cycle 0: read from address [1, 2, 3]
        // Cycle 1: read from address [3, 1, 0]
        // Cycle 2: read from address [2, 2, 1]
        // Cycle 3: read from address [0, 3, 2]
        let reads = [
            (0, [1, 2, 3]),
            (1, [3, 1, 0]),
            (2, [2, 2, 1]),
            (3, [0, 3, 2]),
        ];
        
        // Choose random evaluation points
        let r_cycle: Vec<F> = (0..num_cycle_vars).map(|_| F::random(&mut rng)).collect();
        let r_address_chunks: [Vec<F>; D] = [
            (0..chunk_bits).map(|_| F::random(&mut rng)).collect(),
            (0..chunk_bits).map(|_| F::random(&mut rng)).collect(),
            (0..chunk_bits).map(|_| F::random(&mut rng)).collect(),
        ];
        
        // Create ra_i polynomials by summing one-hot encodings
        let num_cycles = 1 << num_cycle_vars;
        let chunk_size = 1 << chunk_bits;
        
        let mut ra_values: [Vec<F>; D] = [
            vec![F::zero(); num_cycles * chunk_size],
            vec![F::zero(); num_cycles * chunk_size],
            vec![F::zero(); num_cycles * chunk_size],
        ];
        
        // Add each read to the polynomials
        for (cycle, chunks) in reads.iter() {
            for i in 0..D {
                let index = cycle * chunk_size + chunks[i];
                ra_values[i][index] = F::one();
            }
        }
        
        // Create and pre-bind the polynomials
        let mut ra_polys = Vec::with_capacity(D);
        for i in 0..D {
            let mut poly = MultilinearPolynomial::LargeScalars(DensePolynomial::new(ra_values[i].clone()));
            // Pre-bind the address variables
            for j in 0..chunk_bits {
                poly.bind(r_address_chunks[i][j], BindingOrder::HighToLow);
            }
            ra_polys.push(poly);
        }
        let ra_polys: [MultilinearPolynomial<F>; D] = ra_polys.try_into().unwrap();
        
        // Create the sumcheck instance
        let sumcheck = VirtualRASumcheck::<F, D>::new(
            ra_polys,
            r_cycle.clone(),
            r_address_chunks.clone(),
        );
        
        // Compute the expected claim
        let claim = <VirtualRASumcheck<F, D> as BatchableSumcheckInstance<F, ProofTranscript>>::input_claim(&sumcheck);
        
        // Create transcripts
        let mut prover_transcript = ProofTranscript::new(b"test_virtual_ra_multiple");
        let mut verifier_transcript = ProofTranscript::new(b"test_virtual_ra_multiple");
        
        // Prove
        let (proof, _) = sumcheck.prove(&mut prover_transcript);
        
        // Verify
        let result = VirtualRASumcheck::<F, D>::verify(
            &proof,
            claim,
            &r_cycle,
            &r_address_chunks,
            &mut verifier_transcript,
        );
        
        // Check that verification succeeded
        assert!(result.is_ok());
    }
}
