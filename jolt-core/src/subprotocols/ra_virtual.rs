use crate::{
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
    },
    subprotocols::sumcheck::{BatchableSumcheckInstance, SumcheckInstanceProof},
    utils::{errors::ProofVerifyError, transcript::Transcript},
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

/// Proof for the virtual RA sumcheck
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct VirtualRAProof<F: JoltField, ProofTranscript: Transcript> {
    pub sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    pub ra_i_claims: Vec<F>,
}

/// Virtual RA sumcheck for d-way chunked addresses
pub struct VirtualRASumcheck<F: JoltField, const D: usize> {
    /// RA polynomials for each chunk
    ra_i_polys: [MultilinearPolynomial<F>; D],
    /// Eq polynomial for r_cycle
    eq_poly: EqPolynomial<F>,
    /// Current partial evaluation point for eq polynomial
    eq_partial_point: Vec<F>,
    /// Random point r_cycle
    r_cycle: Vec<F>,
    /// Random points r_address^(i) for each chunk
    r_address_chunks: [Vec<F>; D],
    /// Current sumcheck round
    current_round: usize,
    /// Number of cycle variables
    num_cycle_vars: usize,
    /// Cached openings after sumcheck
    cached_openings: Option<[F; D]>,
    /// RA claims from proof (used during verification)
    verifier_ra_claims: Option<Vec<F>>,
}

impl<F: JoltField, const D: usize> VirtualRASumcheck<F, D> {
    /// Creates a new virtual RA sumcheck instance
    pub fn new(
        mut ra_polys: [MultilinearPolynomial<F>; D],
        r_cycle: Vec<F>,
        r_address_chunks: [Vec<F>; D],
    ) -> Self {
        let num_cycle_vars = r_cycle.len();

        // Pre-bind address variables for each ra polynomial
        let total_vars = ra_polys[0].get_num_vars();
        let addr_vars = total_vars - num_cycle_vars;

        for i in 0..D {
            for j in 0..addr_vars {
                ra_polys[i].bind(r_address_chunks[i][j], BindingOrder::LowToHigh);
            }
        }

        let eq_poly = EqPolynomial::new(r_cycle.clone());
        let eq_partial_point = Vec::with_capacity(num_cycle_vars);

        Self {
            ra_i_polys: ra_polys,
            eq_poly,
            eq_partial_point,
            r_cycle,
            r_address_chunks,
            current_round: 0,
            num_cycle_vars,
            cached_openings: None,
            verifier_ra_claims: None,
        }
    }

    /// Proves the virtual RA sumcheck
    pub fn prove<ProofTranscript: Transcript>(
        mut self,
        transcript: &mut ProofTranscript,
    ) -> (VirtualRAProof<F, ProofTranscript>, Vec<F>) {
        let (sumcheck_proof, r_cycle_bound) =
            crate::subprotocols::sumcheck::BatchedSumcheck::prove(vec![&mut self], transcript);

        let ra_i_claims = self.cached_openings.unwrap().to_vec();

        let proof = VirtualRAProof {
            sumcheck_proof,
            ra_i_claims,
        };

        (proof, r_cycle_bound)
    }

    /// Verifies the virtual RA sumcheck
    pub fn verify<ProofTranscript: Transcript>(
        proof: &VirtualRAProof<F, ProofTranscript>,
        claim: F,
        ra_polys: [MultilinearPolynomial<F>; D],
        r_cycle: Vec<F>,
        r_address_chunks: [Vec<F>; D],
        transcript: &mut ProofTranscript,
    ) -> Result<Vec<F>, ProofVerifyError> {
        // Create a new verifier instance
        let mut verifier_instance = Self::new(ra_polys, r_cycle, r_address_chunks);

        // Verify that the claim matches the expected input claim
        let expected_claim = <Self as BatchableSumcheckInstance<F, ProofTranscript>>::input_claim(
            &verifier_instance,
        );
        if claim != expected_claim {
            return Err(ProofVerifyError::InternalError);
        }

        // Set the verifier claims so expected_output_claim can use them
        verifier_instance.verifier_ra_claims = Some(proof.ra_i_claims.clone());

        let instances: Vec<&dyn BatchableSumcheckInstance<F, ProofTranscript>> =
            vec![&verifier_instance];

        crate::subprotocols::sumcheck::BatchedSumcheck::verify(
            &proof.sumcheck_proof,
            instances,
            transcript,
        )
    }
}

impl<F: JoltField, ProofTranscript: Transcript, const D: usize>
    BatchableSumcheckInstance<F, ProofTranscript> for VirtualRASumcheck<F, D>
{
    fn degree(&self) -> usize {
        D + 1
    }

    fn num_rounds(&self) -> usize {
        self.num_cycle_vars
    }

    fn input_claim(&self) -> F {
        let mut sum = F::zero();
        let num_points = 1 << self.num_cycle_vars;

        for j in 0..num_points {
            // Convert index j to binary representation for evaluation point
            let mut eval_point = vec![F::zero(); self.num_cycle_vars];
            for k in 0..self.num_cycle_vars {
                if (j >> k) & 1 == 1 {
                    eval_point[k] = F::one();
                }
            }

            // Compute eq(r_cycle, eval_point)
            let eq_eval = self.eq_poly.evaluate(&eval_point);

            let mut ra_product = F::one();
            for i in 0..D {
                let ra_eval = self.ra_i_polys[i].get_bound_coeff(j);
                ra_product *= ra_eval;
            }

            sum += eq_eval * ra_product;
        }

        sum
    }

    fn compute_prover_message(&self, _round: usize) -> Vec<F> {
        let half_len = 1 << (self.num_cycle_vars - self.current_round - 1);
        let degree = D + 1;

        // Initialize evaluations for degree D+1
        let mut evals = vec![F::zero(); degree];

        // For each index k, compute contributions
        for k in 0..half_len {
            // Get RA polynomial evaluations using sumcheck_evals
            let mut ra_evals_at_points = Vec::with_capacity(D);
            for i in 0..D {
                let ra_evals =
                    self.ra_i_polys[i].sumcheck_evals(k, degree, BindingOrder::LowToHigh);
                ra_evals_at_points.push(ra_evals);
            }

            // Build evaluation points for eq polynomial
            let mut eval_point_0 = self.eq_partial_point.clone();
            let mut eval_point_1 = self.eq_partial_point.clone();
            
            // Complete the evaluation points with the remaining variables
            let remaining_vars = self.num_cycle_vars - self.current_round;
            for j in 1..remaining_vars {
                let bit = (k >> (j - 1)) & 1;
                if bit == 1 {
                    eval_point_0.push(F::one());
                    eval_point_1.push(F::one());
                } else {
                    eval_point_0.push(F::zero());
                    eval_point_1.push(F::zero());
                }
            }
            
            // The last variable is what we're evaluating over
            eval_point_0.push(F::zero());
            eval_point_1.push(F::one());

            // Compute eq evaluations at 0 and 1
            let eq_k_0 = self.eq_poly.evaluate(&eval_point_0);
            let eq_k_1 = self.eq_poly.evaluate(&eval_point_1);

            // Compute contributions for each evaluation point
            for point in 0..degree {
                // For degree D+1 polynomial, we evaluate at points 0, 1, 2, ..., D
                let t = F::from_u64(point as u64);
                let eq_eval = eq_k_0 + t * (eq_k_1 - eq_k_0);

                // Compute product of RA evaluations
                let mut ra_product = F::one();
                for i in 0..D {
                    ra_product *= ra_evals_at_points[i][point];
                }

                evals[point] += eq_eval * ra_product;
            }
        }

        // Extract evaluations at 0, 2, 3, ..., degree (skipping 1)
        let mut result = vec![evals[0]];
        for i in 2..degree {
            result.push(evals[i]);
        }

        result
    }

    fn bind(&mut self, r_j: F, round: usize) {
        assert_eq!(round, self.current_round);

        // Update the partial evaluation point for eq polynomial
        self.eq_partial_point.push(r_j);

        // Bind each ra polynomial
        for i in 0..D {
            self.ra_i_polys[i].bind(r_j, BindingOrder::LowToHigh);
        }

        self.current_round += 1;
    }

    fn cache_openings(&mut self) {
        let mut openings = [F::zero(); D];

        for i in 0..D {
            openings[i] = self.ra_i_polys[i].final_sumcheck_claim();
        }

        self.cached_openings = Some(openings);
    }

    fn expected_output_claim(&self, _r: &[F]) -> F {
        // After all bindings, we have the complete evaluation point
        // The eq_partial_point should have all num_cycle_vars elements
        assert_eq!(self.eq_partial_point.len(), self.num_cycle_vars);
        
        // Evaluate eq(r_cycle, eq_partial_point)
        let eq_eval = self.eq_poly.evaluate(&self.eq_partial_point);

        let mut ra_product = F::one();
        
        // If we have verifier claims (during verification), use those
        // Otherwise use the computed claims from the polynomials (during proving)
        if let Some(ref ra_claims) = self.verifier_ra_claims {
            for i in 0..D {
                ra_product *= ra_claims[i];
            }
        } else {
            for i in 0..D {
                ra_product *= self.ra_i_polys[i].final_sumcheck_claim();
            }
        }

        eq_eval * ra_product
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        poly::dense_mlpoly::DensePolynomial,
        utils::{thread::unsafe_allocate_zero_vec, transcript::KeccakTranscript},
    };
    use ark_bn254::Fr;
    use ark_std::{test_rng, One, UniformRand};

    type F = Fr;
    type ProofTranscript = KeccakTranscript;

    fn create_one_hot_ra_polys<const D: usize>(
        num_cycle_vars: usize,
        chunk_bits: usize,
        cycle: usize,
        address_chunks: &[usize; D],
    ) -> [MultilinearPolynomial<F>; D] {
        let num_cycles = 1 << num_cycle_vars;
        let chunk_size = 1 << chunk_bits;

        let mut ra_polys = Vec::with_capacity(D);

        for i in 0..D {
            let mut values = unsafe_allocate_zero_vec(num_cycles * chunk_size);
            let index = cycle * chunk_size + address_chunks[i];
            values[index] = F::one();

            let poly = MultilinearPolynomial::LargeScalars(DensePolynomial::new(values));
            ra_polys.push(poly);
        }

        ra_polys.try_into().unwrap()
    }

    #[test]
    fn test_virtual_ra_sumcheck_prove_verify() {
        const D: usize = 2;
        let mut rng = test_rng();

        let num_cycle_vars = 3;
        let chunk_bits = 2;
        let read_cycle = 5;
        let address_chunks = [2, 2];

        let r_cycle: Vec<F> = (0..num_cycle_vars).map(|_| F::random(&mut rng)).collect();
        let r_address_chunks: [Vec<F>; D] = [
            (0..chunk_bits).map(|_| F::random(&mut rng)).collect(),
            (0..chunk_bits).map(|_| F::random(&mut rng)).collect(),
        ];

        let ra_polys =
            create_one_hot_ra_polys::<D>(num_cycle_vars, chunk_bits, read_cycle, &address_chunks);

        // Clone polynomials for the prover - they will be mutated during proving
        let prover_ra_polys = ra_polys.clone();

        let sumcheck = VirtualRASumcheck::<F, D>::new(
            prover_ra_polys,
            r_cycle.clone(),
            r_address_chunks.clone(),
        );

        let claim =
            <VirtualRASumcheck<F, D> as BatchableSumcheckInstance<F, ProofTranscript>>::input_claim(
                &sumcheck,
            );

        let mut prover_transcript = ProofTranscript::new(b"test_virtual_ra");
        let mut verifier_transcript = ProofTranscript::new(b"test_virtual_ra");

        let (proof, r_cycle_bound) = sumcheck.prove(&mut prover_transcript);

        // Use fresh polynomials for verification
        let result = VirtualRASumcheck::<F, D>::verify(
            &proof,
            claim,
            ra_polys,
            r_cycle.clone(),
            r_address_chunks.clone(),
            &mut verifier_transcript,
        );

        assert!(result.is_ok());
        assert_eq!(r_cycle_bound, result.unwrap());
    }

    #[test]
    fn test_virtual_ra_sumcheck_multiple_reads() {
        const D: usize = 3;
        let mut rng = test_rng();

        let num_cycle_vars = 2;
        let chunk_bits = 2;

        let reads = [
            (0, [1, 2, 3]),
            (1, [3, 1, 0]),
            (2, [2, 2, 1]),
            (3, [0, 3, 2]),
        ];

        let r_cycle: Vec<F> = (0..num_cycle_vars).map(|_| F::random(&mut rng)).collect();
        let r_address_chunks: [Vec<F>; D] = [
            (0..chunk_bits).map(|_| F::random(&mut rng)).collect(),
            (0..chunk_bits).map(|_| F::random(&mut rng)).collect(),
            (0..chunk_bits).map(|_| F::random(&mut rng)).collect(),
        ];

        let num_cycles = 1 << num_cycle_vars;
        let chunk_size = 1 << chunk_bits;

        let mut ra_values: [Vec<F>; D] = [
            unsafe_allocate_zero_vec(num_cycles * chunk_size),
            unsafe_allocate_zero_vec(num_cycles * chunk_size),
            unsafe_allocate_zero_vec(num_cycles * chunk_size),
        ];

        for (cycle, chunks) in reads.iter() {
            for i in 0..D {
                let index = cycle * chunk_size + chunks[i];
                ra_values[i][index] = F::one();
            }
        }

        let mut ra_polys = Vec::with_capacity(D);
        for i in 0..D {
            let poly =
                MultilinearPolynomial::LargeScalars(DensePolynomial::new(ra_values[i].clone()));
            ra_polys.push(poly);
        }
        let ra_polys: [MultilinearPolynomial<F>; D] = ra_polys.try_into().unwrap();

        // Clone polynomials for the prover - they will be mutated during proving
        let prover_ra_polys = ra_polys.clone();

        let sumcheck = VirtualRASumcheck::<F, D>::new(
            prover_ra_polys,
            r_cycle.clone(),
            r_address_chunks.clone(),
        );

        let claim =
            <VirtualRASumcheck<F, D> as BatchableSumcheckInstance<F, ProofTranscript>>::input_claim(
                &sumcheck,
            );

        let mut prover_transcript = ProofTranscript::new(b"test_multiple");
        let mut verifier_transcript = ProofTranscript::new(b"test_multiple");

        let (proof, _) = sumcheck.prove(&mut prover_transcript);

        // Use fresh polynomials for verification
        let result = VirtualRASumcheck::<F, D>::verify(
            &proof,
            claim,
            ra_polys,
            r_cycle.clone(),
            r_address_chunks.clone(),
            &mut verifier_transcript,
        );

        assert!(result.is_ok());
    }

    #[test]
    fn test_virtual_ra_sumcheck_large_random() {
        const D: usize = 4;
        let mut rng = test_rng();

        let num_cycle_vars = 15;
        let chunk_bits = 4;
        let num_reads = 50;
        let num_cycles = 1 << num_cycle_vars;
        let chunk_size = 1 << chunk_bits;

        let mut ra_values: Vec<Vec<F>> = (0..D)
            .map(|_| unsafe_allocate_zero_vec(num_cycles * chunk_size))
            .collect();

        for _ in 0..num_reads {
            let cycle = usize::rand(&mut rng) % num_cycles;
            for i in 0..D {
                let chunk = usize::rand(&mut rng) % chunk_size;
                let index = cycle * chunk_size + chunk;
                ra_values[i][index] = F::one();
            }
        }

        let mut ra_polys = Vec::with_capacity(D);
        for i in 0..D {
            let poly =
                MultilinearPolynomial::LargeScalars(DensePolynomial::new(ra_values[i].clone()));
            ra_polys.push(poly);
        }
        let ra_polys: [MultilinearPolynomial<F>; D] = ra_polys.try_into().unwrap();

        let r_cycle: Vec<F> = (0..num_cycle_vars).map(|_| F::random(&mut rng)).collect();
        let r_address_chunks: [Vec<F>; D] =
            std::array::from_fn(|_| (0..chunk_bits).map(|_| F::random(&mut rng)).collect());

        // Clone polynomials for the prover - they will be mutated during proving
        let prover_ra_polys = ra_polys.clone();

        let sumcheck = VirtualRASumcheck::<F, D>::new(
            prover_ra_polys,
            r_cycle.clone(),
            r_address_chunks.clone(),
        );

        let claim =
            <VirtualRASumcheck<F, D> as BatchableSumcheckInstance<F, ProofTranscript>>::input_claim(
                &sumcheck,
            );

        let mut prover_transcript = ProofTranscript::new(b"test_large");
        let mut verifier_transcript = ProofTranscript::new(b"test_large");

        let (proof, r_cycle_bound) = sumcheck.prove(&mut prover_transcript);

        // Use fresh polynomials for verification
        let result = VirtualRASumcheck::<F, D>::verify(
            &proof,
            claim,
            ra_polys,
            r_cycle.clone(),
            r_address_chunks.clone(),
            &mut verifier_transcript,
        );

        assert!(result.is_ok());
        assert_eq!(r_cycle_bound, result.unwrap());
    }
}
