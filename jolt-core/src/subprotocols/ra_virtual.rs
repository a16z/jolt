use crate::{
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
    },
    subprotocols::sumcheck::{BatchableSumcheckInstance, SumcheckInstanceProof},
    utils::{errors::ProofVerifyError, thread::unsafe_allocate_zero_vec, transcript::Transcript},
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use rayon::prelude::*;

/// Proof for the virtual RA sumcheck
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct VirtualRAProof<F: JoltField, ProofTranscript: Transcript> {
    pub sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    pub ra_claims: Vec<F>,
}

/// Virtual RA sumcheck for d-way chunked addresses
pub struct VirtualRASumcheck<F: JoltField, const D: usize> {
    /// RA polynomials for each chunk
    ra_polys: [MultilinearPolynomial<F>; D],
    /// Precomputed evaluations of eq(r_cycle, *)
    eq_evaluations: Vec<F>,
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
    /// Current eq polynomial evaluations
    current_eq_evaluations: Vec<F>,
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
        let chunk_bits = total_vars - num_cycle_vars;

        for i in 0..D {
            for j in 0..chunk_bits {
                ra_polys[i].bind(r_address_chunks[i][j], BindingOrder::HighToLow);
            }
        }

        let eq_evaluations = EqPolynomial::evals(&r_cycle);
        let current_eq_evaluations = eq_evaluations.clone();

        Self {
            ra_polys,
            eq_evaluations,
            r_cycle,
            r_address_chunks,
            current_round: 0,
            num_cycle_vars,
            cached_openings: None,
            current_eq_evaluations,
        }
    }

    /// Computes the evaluations for the current sumcheck round
    fn compute_round_evaluations(&self) -> Vec<F> {
        let half_len = 1 << (self.num_cycle_vars - self.current_round - 1);
        let degree = D + 1;

        let evals: Vec<F> = (0..=degree)
            .into_par_iter()
            .map(|eval_point| {
                let point = F::from_u64(eval_point as u64);
                let mut sum = F::zero();

                for k in 0..half_len {
                    // Compute eq polynomial contribution
                    let eq_contrib = if eval_point == 0 {
                        self.current_eq_evaluations[k]
                    } else if eval_point == 1 {
                        self.current_eq_evaluations[k + half_len]
                    } else {
                        let eq_0 = self.current_eq_evaluations[k];
                        let eq_1 = self.current_eq_evaluations[k + half_len];
                        eq_0 + point * (eq_1 - eq_0)
                    };

                    // Compute product of ra evaluations
                    let mut ra_product = F::one();
                    for i in 0..D {
                        let ra_eval = if eval_point == 0 {
                            self.ra_polys[i].get_bound_coeff(k)
                        } else if eval_point == 1 {
                            self.ra_polys[i].get_bound_coeff(k + half_len)
                        } else {
                            let low = self.ra_polys[i].get_bound_coeff(k);
                            let high = self.ra_polys[i].get_bound_coeff(k + half_len);
                            low + point * (high - low)
                        };
                        ra_product *= ra_eval;
                    }

                    sum += eq_contrib * ra_product;
                }

                sum
            })
            .collect();

        evals
    }

    /// Proves the virtual RA sumcheck
    pub fn prove<ProofTranscript: Transcript>(
        mut self,
        transcript: &mut ProofTranscript,
    ) -> (VirtualRAProof<F, ProofTranscript>, Vec<F>) {
        let (sumcheck_proof, r_cycle_bound) =
            crate::subprotocols::sumcheck::BatchedSumcheck::prove(vec![&mut self], transcript);

        let ra_claims = self.cached_openings.unwrap().to_vec();

        let proof = VirtualRAProof {
            sumcheck_proof,
            ra_claims,
        };

        (proof, r_cycle_bound)
    }

    /// Verifies the virtual RA sumcheck
    pub fn verify<ProofTranscript: Transcript>(
        proof: &VirtualRAProof<F, ProofTranscript>,
        claim: F,
        r_cycle: &[F],
        r_address_chunks: &[Vec<F>; D],
        transcript: &mut ProofTranscript,
    ) -> Result<Vec<F>, ProofVerifyError> {
        let verifier = VirtualRASumcheckVerifier::<F, D> {
            r_cycle: r_cycle.to_vec(),
            r_address_chunks: r_address_chunks.clone(),
            ra_claims: proof.ra_claims.clone(),
            claim,
        };

        let instances: Vec<&dyn BatchableSumcheckInstance<F, ProofTranscript>> = vec![&verifier];

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
            let eq_eval = self.eq_evaluations[j];

            let mut ra_product = F::one();
            for i in 0..D {
                let ra_eval = self.ra_polys[i].get_bound_coeff(j);
                ra_product *= ra_eval;
            }

            sum += eq_eval * ra_product;
        }

        sum
    }

    fn compute_prover_message(&self, _round: usize) -> Vec<F> {
        let all_evals = self.compute_round_evaluations();
        
        // Extract evaluations at 0, 2, 3, ..., degree (skipping 1)
        let mut evals = vec![all_evals[0]];
        for i in 2..all_evals.len() {
            evals.push(all_evals[i]);
        }

        evals
    }

    fn bind(&mut self, r_j: F, round: usize) {
        assert_eq!(round, self.current_round);

        // Update eq polynomial evaluations
        let len = self.current_eq_evaluations.len() / 2;
        let mut new_evals = unsafe_allocate_zero_vec(len);

        for i in 0..len {
            let eq_0 = self.current_eq_evaluations[i];
            let eq_1 = self.current_eq_evaluations[i + len];
            new_evals[i] = eq_0 + r_j * (eq_1 - eq_0);
        }

        self.current_eq_evaluations = new_evals;

        // Bind each ra polynomial
        for i in 0..D {
            self.ra_polys[i].bind(r_j, BindingOrder::HighToLow);
        }

        self.current_round += 1;
    }

    fn cache_openings(&mut self) {
        let mut openings = [F::zero(); D];

        for i in 0..D {
            openings[i] = if self.ra_polys[i].get_num_vars() == 0 {
                self.ra_polys[i].get_bound_coeff(0)
            } else {
                self.ra_polys[i].evaluate(&[])
            };
        }

        self.cached_openings = Some(openings);
    }

    fn expected_output_claim(&self, r: &[F]) -> F {
        let eq_poly = EqPolynomial::new(self.r_cycle.clone());
        let eq_eval = eq_poly.evaluate(r);

        let mut ra_product = F::one();
        for i in 0..D {
            let ra_eval = if self.ra_polys[i].get_num_vars() == 0 {
                self.ra_polys[i].get_bound_coeff(0)
            } else {
                self.ra_polys[i].evaluate(&[])
            };
            ra_product *= ra_eval;
        }

        eq_eval * ra_product
    }
}

/// Verifier for the virtual RA sumcheck
struct VirtualRASumcheckVerifier<F: JoltField, const D: usize> {
    r_cycle: Vec<F>,
    r_address_chunks: [Vec<F>; D],
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

        let sumcheck =
            VirtualRASumcheck::<F, D>::new(ra_polys, r_cycle.clone(), r_address_chunks.clone());

        let claim =
            <VirtualRASumcheck<F, D> as BatchableSumcheckInstance<F, ProofTranscript>>::input_claim(
                &sumcheck,
            );

        let mut prover_transcript = ProofTranscript::new(b"test_virtual_ra");
        let mut verifier_transcript = ProofTranscript::new(b"test_virtual_ra");

        let (proof, r_cycle_bound) = sumcheck.prove(&mut prover_transcript);

        let result = VirtualRASumcheck::<F, D>::verify(
            &proof,
            claim,
            &r_cycle,
            &r_address_chunks,
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

        let sumcheck =
            VirtualRASumcheck::<F, D>::new(ra_polys, r_cycle.clone(), r_address_chunks.clone());

        let claim =
            <VirtualRASumcheck<F, D> as BatchableSumcheckInstance<F, ProofTranscript>>::input_claim(
                &sumcheck,
            );

        let mut prover_transcript = ProofTranscript::new(b"test_multiple");
        let mut verifier_transcript = ProofTranscript::new(b"test_multiple");

        let (proof, _) = sumcheck.prove(&mut prover_transcript);

        let result = VirtualRASumcheck::<F, D>::verify(
            &proof,
            claim,
            &r_cycle,
            &r_address_chunks,
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

        let sumcheck =
            VirtualRASumcheck::<F, D>::new(ra_polys, r_cycle.clone(), r_address_chunks.clone());

        let claim =
            <VirtualRASumcheck<F, D> as BatchableSumcheckInstance<F, ProofTranscript>>::input_claim(
                &sumcheck,
            );

        let mut prover_transcript = ProofTranscript::new(b"test_large");
        let mut verifier_transcript = ProofTranscript::new(b"test_large");

        let (proof, r_cycle_bound) = sumcheck.prove(&mut prover_transcript);

        let result = VirtualRASumcheck::<F, D>::verify(
            &proof,
            claim,
            &r_cycle,
            &r_address_chunks,
            &mut verifier_transcript,
        );

        assert!(result.is_ok());
        assert_eq!(r_cycle_bound, result.unwrap());
    }
}
