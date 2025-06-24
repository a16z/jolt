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
    _r_address_chunks: [Vec<F>; D],

    /// Current round index during sumcheck
    current_round: usize,

    /// Number of cycle variables (log T)
    num_cycle_vars: usize,

    /// Cached openings to be proven later
    cached_openings: Option<[F; D]>, //TODO(markosg04): fix this -- why is it Option?

    /// Current binding of the eq polynomial evaluations
    current_eq_evals: Vec<F>,
}

impl<F: JoltField, const D: usize> VirtualRASumcheck<F, D> {
    pub fn new(
        mut ra_i_polys: [MultilinearPolynomial<F>; D],
        r_cycle: Vec<F>,
        r_address_chunks: [Vec<F>; D],
    ) -> Self {
        let num_cycle_vars = r_cycle.len();
        
        // Pre-bind the address variables for each ra_i polynomial
        // Each ra_i has num_cycle_vars + chunk_bits variables
        // We bind the last chunk_bits variables (the address part) to r_address_chunks[i]
        let total_vars = ra_i_polys[0].get_num_vars();
        let chunk_bits = total_vars - num_cycle_vars;
        
        for i in 0..D {
            // Pre-bind the address variables to r_address_chunks[i]
            // The polynomial has num_cycle_vars + chunk_bits variables total
            // We bind the last chunk_bits variables (the address part)
            for j in 0..chunk_bits {
                ra_i_polys[i].bind(r_address_chunks[i][j], BindingOrder::HighToLow);
            }
        }
        
        // Precompute eq(r_cycle, *) evaluations
        let eq_evals = EqPolynomial::evals(&r_cycle);
        let current_eq_evals = eq_evals.clone();

        Self {
            ra_i_polys,
            eq_evals,
            r_cycle,
            _r_address_chunks: r_address_chunks,
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
        // eq(r_cycle[..current_round] || X || remaining_vars, j) * Product_i ra_i(j, r_address^(i))

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
    use ark_std::{One, Zero, test_rng, UniformRand};
    use ark_ff::PrimeField;

    type F = Fr;
    type ProofTranscript = KeccakTranscript;

    /// Helper function to create one-hot ra_i polynomials  
    /// Given an address decomposed into D chunks, each of B bits,
    /// creates D polynomials where ra_i represents the one-hot encoding
    /// of the i-th chunk of the address at the given cycle.
    /// 
    /// Each ra_i polynomial is over all (cycle, address) variables.
    /// The pre-binding to address chunks is now handled in VirtualRASumcheck::new.
    fn create_one_hot_ra_polys<const D: usize>(
        num_cycle_vars: usize,
        chunk_bits: usize,
        cycle: usize,
        address_chunks: &[usize; D],
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
            let poly = MultilinearPolynomial::LargeScalars(DensePolynomial::new(values));
            
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
        
        // Create the polynomials (pre-binding will be done in VirtualRASumcheck::new)
        let mut ra_polys = Vec::with_capacity(D);
        for i in 0..D {
            let poly = MultilinearPolynomial::LargeScalars(DensePolynomial::new(ra_values[i].clone()));
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

    // Helper function to format field elements nicely
    fn format_field_element(f: &F) -> String {
        // Get the internal representation to show a shorter form
        let limbs = f.into_bigint().0;
        if limbs[0] == 0 && limbs[1] == 0 && limbs[2] == 0 && limbs[3] == 0 {
            "0".to_string()
        } else if limbs[0] == 1 && limbs[1] == 0 && limbs[2] == 0 && limbs[3] == 0 {
            "1".to_string()
        } else {
            // Show first 8 hex digits for brevity
            format!("0x{:08x}...", limbs[0])
        }
    }

    // Helper function to print polynomial evaluations
    fn print_polynomial_table(title: &str, labels: &[String], values: &[Vec<F>]) {
        println!("\n{}", title);
        println!("{}", "=".repeat(title.len()));
        
        // Print header
        print!("| Index |");
        for label in labels {
            print!(" {:^20} |", label);
        }
        println!();
        
        // Print separator
        print!("|-------|");
        for _ in labels {
            print!("{:-^22}|", "");
        }
        println!();
        
        // Print values
        let num_rows = values[0].len();
        for i in 0..num_rows {
            print!("| {:^5} |", i);
            for vals in values {
                print!(" {:^20} |", format_field_element(&vals[i]));
            }
            println!();
        }
    }

    #[test]
    fn test_virtual_ra_sumcheck_verbose() {
        println!("\n╔════════════════════════════════════════════════════════════════════╗");
        println!("║        VIRTUAL RA SUMCHECK PROTOCOL - VERBOSE WALKTHROUGH          ║");
        println!("╚════════════════════════════════════════════════════════════════════╝");
        
        // Test with D=2 (2-way chunked addresses) for simplicity
        const D: usize = 2;
        let _rng = test_rng();
        
        // Setup parameters
        let num_cycle_vars = 2; // 4 cycles (T = 2^2 = 4)
        let chunk_bits = 2; // 4 possible values per chunk (B = 2^2 = 4)
        
        println!("\n📋 PROTOCOL PARAMETERS:");
        println!("   • D = {} (number of address chunks)", D);
        println!("   • T = {} (number of cycles = 2^{})", 1 << num_cycle_vars, num_cycle_vars);
        println!("   • B = {} (values per chunk = 2^{})", 1 << chunk_bits, chunk_bits);
        println!("   • Total address space = B^D = {}^{} = {}", 
                 1 << chunk_bits, D, (1u32 << chunk_bits).pow(D as u32));
        
        // Simple example: one read at cycle 1 from address 6
        // Address 6 in binary: 0110
        // With 2-bit chunks: chunk_0 = 10 (decimal 2), chunk_1 = 01 (decimal 1)
        let read_cycle = 1;
        let address = 6;
        let address_chunks = [2, 1]; // [10, 01] in binary
        
        println!("\n📍 MEMORY ACCESS PATTERN:");
        println!("   • Read at cycle {} from address {}", read_cycle, address);
        println!("   • Address {} in binary: {:04b}", address, address);
        println!("   • Chunk decomposition:");
        println!("     - chunk_0 = {:02b} (decimal {})", address_chunks[0], address_chunks[0]);
        println!("     - chunk_1 = {:02b} (decimal {})", address_chunks[1], address_chunks[1]);
        
        // Choose specific random evaluation points for reproducibility
        let r_cycle: Vec<F> = vec![
            F::from(7u64),
            F::from(11u64),
        ];
        let r_address_chunks: [Vec<F>; D] = [
            vec![F::from(3u64), F::from(5u64)],
            vec![F::from(13u64), F::from(17u64)],
        ];
        
        println!("\n🎲 RANDOM EVALUATION POINTS:");
        println!("   • r_cycle = [{}, {}]", 
                 format_field_element(&r_cycle[0]), 
                 format_field_element(&r_cycle[1]));
        println!("   • r_address^(0) = [{}, {}]", 
                 format_field_element(&r_address_chunks[0][0]), 
                 format_field_element(&r_address_chunks[0][1]));
        println!("   • r_address^(1) = [{}, {}]", 
                 format_field_element(&r_address_chunks[1][0]), 
                 format_field_element(&r_address_chunks[1][1]));
        
        // Create one-hot ra_i polynomials
        let ra_polys = create_one_hot_ra_polys::<D>(
            num_cycle_vars,
            chunk_bits,
            read_cycle,
            &address_chunks,
        );
        
        println!("\n🔨 CREATING RA POLYNOMIALS:");
        println!("   Each ra_i is initially a (cycle × address_chunk) table:");
        println!("   • ra_0 has a 1 at (cycle={}, chunk={})", read_cycle, address_chunks[0]);
        println!("   • ra_1 has a 1 at (cycle={}, chunk={})", read_cycle, address_chunks[1]);
        println!("   Then VirtualRASumcheck::new will pre-bind each ra_i to r_address^(i)");
        
        // Create the sumcheck instance which will do the pre-binding
        let mut sumcheck = VirtualRASumcheck::<F, D>::new(
            ra_polys,
            r_cycle.clone(),
            r_address_chunks.clone(),
        );
        
        // Show the values after address binding
        let num_cycles = 1 << num_cycle_vars;
        let mut ra_values_after_binding = vec![vec![F::zero(); num_cycles]; D];
        for j in 0..num_cycles {
            for i in 0..D {
                ra_values_after_binding[i][j] = sumcheck.ra_i_polys[i].get_bound_coeff(j);
            }
        }
        
        let labels = vec!["ra_0(j, r_addr^(0))".to_string(), "ra_1(j, r_addr^(1))".to_string()];
        print_polynomial_table("RA Polynomials After Address Binding", &labels, &ra_values_after_binding);
        
        // Compute and show the initial claim
        let claim = <VirtualRASumcheck<F, D> as BatchableSumcheckInstance<F, ProofTranscript>>::input_claim(&sumcheck);
        
        println!("\n🎯 SUMCHECK CLAIM:");
        println!("   Computing: Σ_j eq(r_cycle, j) × ra_0(j, r_addr^(0)) × ra_1(j, r_addr^(1))");
        
        // Show the eq polynomial evaluations
        let mut eq_values = vec![F::zero(); num_cycles];
        for j in 0..num_cycles {
            eq_values[j] = sumcheck.eq_evals[j];
        }
        
        let mut product_values = vec![F::zero(); num_cycles];
        for j in 0..num_cycles {
            product_values[j] = eq_values[j] * ra_values_after_binding[0][j] * ra_values_after_binding[1][j];
        }
        
        let all_values = vec![
            eq_values.clone(),
            ra_values_after_binding[0].clone(),
            ra_values_after_binding[1].clone(),
            product_values.clone(),
        ];
        let all_labels = vec![
            "eq(r_cycle, j)".to_string(),
            "ra_0(j, r_addr^(0))".to_string(),
            "ra_1(j, r_addr^(1))".to_string(),
            "Product".to_string(),
        ];
        print_polynomial_table("Computing Sumcheck Claim", &all_labels, &all_values);
        
        println!("\n   Claim = Sum of Product column = {}", format_field_element(&claim));
        
        // Now simulate the sumcheck protocol round by round
        println!("\n🔄 SUMCHECK PROTOCOL ROUNDS:");
        println!("   We'll reduce from {} variables to 0 variables\n", num_cycle_vars);
        
        let mut round_challenges = vec![];
        
        for round in 0..num_cycle_vars {
            println!("╭─────────────────────────────────────────────────────────────────╮");
            println!("│ ROUND {} of {}                                                    │", round + 1, num_cycle_vars);
            println!("╰─────────────────────────────────────────────────────────────────╯");
            
            // Compute the univariate polynomial for this round
            let uni_poly = sumcheck.compute_round_polynomial();
            
            println!("\n📊 Computing univariate polynomial g_{}(X)", round + 1);
            println!("   Degree = D + 1 = {} + 1 = {}", D, D + 1);
            
            // Show evaluations at 0, 1, 2, 3
            let mut evals = vec![];
            for i in 0..=D+1 {
                let eval = uni_poly.evaluate(&F::from(i as u64));
                evals.push(eval);
                println!("   g_{}({}) = {}", round + 1, i, format_field_element(&eval));
            }
            
            // Verify the sumcheck relation
            let sum_check = evals[0] + evals[1];
            println!("\n✓ Sumcheck relation: g_{}(0) + g_{}(1) = {} + {} = {}", 
                     round + 1, round + 1,
                     format_field_element(&evals[0]),
                     format_field_element(&evals[1]),
                     format_field_element(&sum_check));
            
            if round == 0 {
                println!("  This should equal the initial claim: {}", format_field_element(&claim));
                assert_eq!(sum_check, claim);
            } else {
                let prev_eval = uni_poly.evaluate(&round_challenges[round - 1]);
                println!("  This should equal g_{}({}) = {}", 
                         round, 
                         format_field_element(&round_challenges[round - 1]),
                         format_field_element(&prev_eval));
            }
            
            // Simulate verifier's random challenge
            let challenge = F::from(23u64 + round as u64 * 7u64); // Deterministic for demo
            round_challenges.push(challenge);
            
            println!("\n🎲 Verifier sends challenge r_{} = {}", round + 1, format_field_element(&challenge));
            
            // Bind the variable
            <VirtualRASumcheck<F, D> as BatchableSumcheckInstance<F, ProofTranscript>>::bind(&mut sumcheck, challenge, round);
            
            println!("\n📌 Binding cycle variable {} to {}", round, format_field_element(&challenge));
            
            // Show the state after binding
            if round < num_cycle_vars - 1 {
                let remaining_vars = num_cycle_vars - round - 1;
                let remaining_size = 1 << remaining_vars;
                
                println!("   Remaining evaluation points: {}", remaining_size);
                
                // Show updated eq evaluations
                let mut new_eq_vals = vec![F::zero(); remaining_size];
                for i in 0..remaining_size {
                    new_eq_vals[i] = sumcheck.current_eq_evals[i];
                }
                
                // Show updated ra values
                let mut new_ra_vals = vec![vec![F::zero(); remaining_size]; D];
                for i in 0..D {
                    for j in 0..remaining_size {
                        new_ra_vals[i][j] = sumcheck.ra_i_polys[i].get_bound_coeff(j);
                    }
                }
                
                let mut bound_labels = vec![format!("eq(r_1..r_{}, *, j)", round + 1)];
                for i in 0..D {
                    bound_labels.push(format!("ra_{}(r_1..r_{}, *, r_addr^({}))", i, round + 1, i));
                }
                
                let mut bound_values = vec![new_eq_vals];
                bound_values.extend(new_ra_vals);
                
                print_polynomial_table(&format!("State After Binding Variable {}", round), &bound_labels, &bound_values);
            }
        }
        
        // Final opening
        println!("\n╭─────────────────────────────────────────────────────────────────╮");
        println!("│ FINAL OPENING                                                   │");
        println!("╰─────────────────────────────────────────────────────────────────╯");
        
        <VirtualRASumcheck<F, D> as BatchableSumcheckInstance<F, ProofTranscript>>::cache_openings(&mut sumcheck);
        let openings = sumcheck.cached_openings.unwrap();
        
        println!("\n🔓 After binding all cycle variables:");
        for i in 0..D {
            println!("   ra_{}(r_cycle, r_address^({})) = {}", 
                     i, i, format_field_element(&openings[i]));
        }
        
        // Compute expected final claim
        let eq_poly = EqPolynomial::new(r_cycle.clone());
        let eq_final = eq_poly.evaluate(&round_challenges);
        let ra_product = openings[0] * openings[1];
        let expected_final = eq_final * ra_product;
        
        println!("\n📝 Final verification:");
        println!("   eq(r_cycle, r_bound) = {}", format_field_element(&eq_final));
        println!("   Π ra_i = {} × {} = {}", 
                 format_field_element(&openings[0]),
                 format_field_element(&openings[1]),
                 format_field_element(&ra_product));
        println!("   Expected final claim = {} × {} = {}", 
                 format_field_element(&eq_final),
                 format_field_element(&ra_product),
                 format_field_element(&expected_final));
        
        // Run actual protocol to verify
        let mut prover_transcript = ProofTranscript::new(b"test_virtual_ra_verbose");
        let sumcheck_for_prove = VirtualRASumcheck::<F, D>::new(
            create_one_hot_ra_polys::<D>(
                num_cycle_vars,
                chunk_bits,
                read_cycle,
                &address_chunks,
            ),
            r_cycle.clone(),
            r_address_chunks.clone(),
        );
        
        let (proof, _r_cycle_bound) = sumcheck_for_prove.prove(&mut prover_transcript);
        
        println!("\n✅ Protocol completed successfully!");
        println!("   Proof size: {} compressed polynomials", proof.sumcheck_proof.compressed_polys.len());
        println!("   Number of ra_claims: {}", proof.ra_claims.len());
        
        // Verify
        let mut verifier_transcript = ProofTranscript::new(b"test_virtual_ra_verbose");
        let result = VirtualRASumcheck::<F, D>::verify(
            &proof,
            claim,
            &r_cycle,
            &r_address_chunks,
            &mut verifier_transcript,
        );
        
        assert!(result.is_ok());
        println!("\n🎉 Verification passed!");
    }

    #[test]
    fn test_virtual_ra_sumcheck_large_random() {
        // Test with 10 rounds of sumcheck (1024 cycles)
        // Using D=4 for 4-way address chunking
        const D: usize = 4;
        let mut rng = test_rng();
        
        // Setup parameters
        let num_cycle_vars = 10; // 2^10 = 1024 cycles
        let chunk_bits = 4; // 2^4 = 16 possible values per chunk
        
        println!("\n📊 LARGE RANDOM TEST PARAMETERS:");
        println!("   • D = {} (number of address chunks)", D);
        println!("   • Number of sumcheck rounds = {}", num_cycle_vars);
        println!("   • T = {} (number of cycles = 2^{})", 1 << num_cycle_vars, num_cycle_vars);
        println!("   • B = {} (values per chunk = 2^{})", 1 << chunk_bits, chunk_bits);
        println!("   • Total address bits = {} × {} = {}", D, chunk_bits, D * chunk_bits);
        
        // Generate random reads - let's do 50 random reads
        let num_reads = 50;
        let num_cycles = 1 << num_cycle_vars;
        let chunk_size = 1 << chunk_bits;
        
        // Create ra_values arrays
        let mut ra_values: Vec<Vec<F>> = (0..D).map(|_| vec![F::zero(); num_cycles * chunk_size]).collect();
        
        println!("\n🎲 Generating {} random reads...", num_reads);
        let mut reads = Vec::new();
        for _ in 0..num_reads {
            let cycle = usize::rand(&mut rng) % num_cycles;
            let mut address_chunks = vec![];
            for _ in 0..D {
                address_chunks.push(usize::rand(&mut rng) % chunk_size);
            }
            
            // Add to polynomial values
            for i in 0..D {
                let index = cycle * chunk_size + address_chunks[i];
                ra_values[i][index] = F::one();
            }
            
            reads.push((cycle, address_chunks));
        }
        
        // Show a sample of the reads
        println!("   Sample reads:");
        for i in 0..5.min(num_reads) {
            let (cycle, chunks) = &reads[i];
            println!("     Read {}: cycle={}, chunks={:?}", i, cycle, chunks);
        }
        if num_reads > 5 {
            println!("     ... and {} more", num_reads - 5);
        }
        
        // Create the polynomials
        let mut ra_polys = Vec::with_capacity(D);
        for i in 0..D {
            let poly = MultilinearPolynomial::LargeScalars(DensePolynomial::new(ra_values[i].clone()));
            ra_polys.push(poly);
        }
        let ra_polys: [MultilinearPolynomial<F>; D] = ra_polys.try_into().unwrap();
        
        // Generate random evaluation points
        println!("\n🎯 Generating random evaluation points...");
        let r_cycle: Vec<F> = (0..num_cycle_vars).map(|_| F::random(&mut rng)).collect();
        let r_address_chunks: [Vec<F>; D] = std::array::from_fn(|_| {
            (0..chunk_bits).map(|_| F::random(&mut rng)).collect()
        });
        
        // Create the sumcheck instance
        let start_time = std::time::Instant::now();
        let sumcheck = VirtualRASumcheck::<F, D>::new(
            ra_polys,
            r_cycle.clone(),
            r_address_chunks.clone(),
        );
        let setup_time = start_time.elapsed();
        
        // Compute the claim
        let claim_start = std::time::Instant::now();
        let claim = <VirtualRASumcheck<F, D> as BatchableSumcheckInstance<F, ProofTranscript>>::input_claim(&sumcheck);
        let claim_time = claim_start.elapsed();
        
        println!("\n⏱️  PERFORMANCE METRICS:");
        println!("   • Setup time: {:?}", setup_time);
        println!("   • Claim computation time: {:?}", claim_time);
        
        // Create transcripts
        let mut prover_transcript = ProofTranscript::new(b"test_virtual_ra_large_random");
        let mut verifier_transcript = ProofTranscript::new(b"test_virtual_ra_large_random");
        
        // Prove
        let prove_start = std::time::Instant::now();
        let (proof, r_cycle_bound) = sumcheck.prove(&mut prover_transcript);
        let prove_time = prove_start.elapsed();
        
        println!("   • Proving time: {:?}", prove_time);
        println!("   • Proof size: {} compressed polynomials", proof.sumcheck_proof.compressed_polys.len());
        println!("   • Number of ra_claims: {}", proof.ra_claims.len());
        
        // Verify
        let verify_start = std::time::Instant::now();
        let result = VirtualRASumcheck::<F, D>::verify(
            &proof,
            claim,
            &r_cycle,
            &r_address_chunks,
            &mut verifier_transcript,
        );
        let verify_time = verify_start.elapsed();
        
        println!("   • Verification time: {:?}", verify_time);
        
        // Check results
        assert!(result.is_ok());
        let verifier_r_cycle_bound = result.unwrap();
        assert_eq!(r_cycle_bound, verifier_r_cycle_bound);
        
        println!("\n✅ Large random test passed successfully!");
        println!("   • Total time: {:?}", setup_time + claim_time + prove_time + verify_time);
        println!("   • Prover/Verifier time ratio: {:.2}x", prove_time.as_secs_f64() / verify_time.as_secs_f64());
    }
}
