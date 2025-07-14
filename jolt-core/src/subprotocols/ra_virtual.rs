use std::cell::RefCell;
use std::rc::Rc;

use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::multilinear_polynomial::PolynomialEvaluation;
use crate::poly::opening_proof::{OpeningPoint, ProverOpeningAccumulator, BIG_ENDIAN};
use crate::subprotocols::sumcheck::CacheSumcheckOpenings;
use crate::{
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
    },
    subprotocols::sumcheck::{BatchableSumcheckInstance, SumcheckInstanceProof},
    utils::{math::Math, transcript::Transcript},
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct RAProof<F: JoltField, ProofTranscript: Transcript> {
    pub ra_i_claims: Vec<F>,
    pub sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
}

pub struct RAProverState<F: JoltField> {
    /// `ra` polys to be constructed based addresses
    ra_i_polys: Vec<MultilinearPolynomial<F>>,
    /// eq poly
    eq_poly: MultilinearPolynomial<F>,
    /// Length of the trace
    T: usize,
}

pub struct RAVerifierState<F: JoltField> {
    /// Random challenge r_cycle
    r_cycle: Vec<F>,
    /// Length of the trace
    T: usize,
}

pub struct RASumcheck<F: JoltField> {
    /// ra(r_cycle, r_address)
    ra_claim: F,
    /// Number of decomposition parts
    d: usize,
    /// Prover state
    prover_state: Option<RAProverState<F>>,
    /// Verifier state
    verifier_state: Option<RAVerifierState<F>>,
    /// ra_i_ claims to be later queried by verifier
    ra_i_claims: Option<Vec<F>>,
}

impl<F: JoltField> RASumcheck<F> {
    pub fn new(
        ra_claim: F,
        addresses: Vec<usize>,
        r_cycle: Vec<F>,
        r_address: Vec<F>,
        T: usize,
        d: usize,
    ) -> Self {
        let base_chunk_size = r_address.len() / d;
        let remainder = r_address.len() % d;

        // First `remainder`` chunks get size base_chunk_size + 1,
        // remaining chunks get size base_chunk_size
        let chunk_sizes: Vec<usize> = (0..d)
            .map(|i| {
                if i < remainder {
                    base_chunk_size + 1
                } else {
                    base_chunk_size
                }
            })
            .collect();

        // Split r_address into d chunks of variable sizes
        let mut r_address_chunks: Vec<Vec<F>> = Vec::with_capacity(d);
        let mut offset = 0;
        for &size in &chunk_sizes {
            r_address_chunks.push(r_address[offset..offset + size].to_vec());
            offset += size;
        }

        let eq_poly = MultilinearPolynomial::from(EqPolynomial::evals(&r_cycle));

        // Precompute EQ tables for each chunk
        let eq_tables: Vec<Vec<F>> = r_address_chunks
            .iter()
            .map(|chunk| EqPolynomial::evals(chunk))
            .collect();

        // We construct ra_i directly from a list of addresses.
        // This way we avoid |ra_i| = k^(1/d) * T size, but rather just |ra_i| = T.

        let mut ra_i_vecs: Vec<Vec<F>> = (0..d).map(|_| vec![F::zero(); T]).collect();

        // For each address, decompose it and add the corresponding EQ evaluations
        for (cycle_idx, &address) in addresses.iter().enumerate() {
            // this is LSB to MSB!! (Doesn't work the other way)
            let mut remaining_address = address;
            for i in 0..d {
                // Each chunk has its own modulo value based on its size
                let chunk_modulo = 1 << chunk_sizes[d - 1 - i];
                let chunk_value = remaining_address % chunk_modulo;
                remaining_address /= chunk_modulo;

                ra_i_vecs[d - 1 - i][cycle_idx] += eq_tables[d - 1 - i][chunk_value];
            }
        }

        let ra_i_polys: Vec<MultilinearPolynomial<F>> = ra_i_vecs
            .into_iter()
            .map(MultilinearPolynomial::from)
            .collect();

        Self {
            ra_claim,
            d,
            prover_state: Some(RAProverState {
                ra_i_polys,
                eq_poly,
                T,
            }),
            verifier_state: None,
            ra_i_claims: None,
        }
    }

    pub fn new_verifier(ra_claim: F, r_cycle: Vec<F>, T: usize, d: usize) -> Self {
        let verifier_state = RAVerifierState { r_cycle, T };

        Self {
            ra_claim,
            d,
            prover_state: None,
            verifier_state: Some(verifier_state),
            ra_i_claims: None,
        }
    }

    #[tracing::instrument(skip_all, name = "ra virtualization")]
    pub fn prove<ProofTranscript: Transcript>(
        mut self,
        transcript: &mut ProofTranscript,
    ) -> (RAProof<F, ProofTranscript>, Vec<F>) {
        let (sumcheck_proof, r_cycle_bound) = self.prove_single(transcript);

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
        T: usize,
        d: usize,
        sumcheck_proof: &SumcheckInstanceProof<F, ProofTranscript>,
        transcript: &mut ProofTranscript,
    ) -> Result<Vec<F>, crate::utils::errors::ProofVerifyError> {
        let mut verifier_sumcheck = Self::new_verifier(ra_claim, r_cycle, T, d);

        verifier_sumcheck.ra_i_claims = Some(ra_i_claims);

        let r_cycle_bound = verifier_sumcheck.verify_single(sumcheck_proof, transcript)?;

        Ok(r_cycle_bound)
    }
}

impl<F: JoltField> BatchableSumcheckInstance<F> for RASumcheck<F> {
    fn degree(&self) -> usize {
        self.d + 1
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
        self.ra_claim
    }

    fn expected_output_claim(&self, r: &[F]) -> F {
        let verifier_state = self
            .verifier_state
            .as_ref()
            .expect("Verifier state not initialized");
        let ra_i_claims = self.ra_i_claims.as_ref().expect("ra_i_claims not set");

        // we need opposite endian-ness here
        let r_rev: Vec<_> = r.iter().cloned().rev().collect();
        let eq_eval = EqPolynomial::mle(&verifier_state.r_cycle, &r_rev);

        // Compute the product of all ra_i evaluations
        let mut product = F::one();
        for ra_i_claim in ra_i_claims.iter() {
            product *= *ra_i_claim;
        }

        eq_eval * product
    }

    fn compute_prover_message(&mut self, _round: usize) -> Vec<F> {
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");
        let degree = <Self as BatchableSumcheckInstance<F>>::degree(self);
        let ra_i_polys = &prover_state.ra_i_polys;
        let eq_poly = &prover_state.eq_poly;

        // We need to compute evaluations at 0, 2, 3, ..., degree
        // = eq(r_cycle, j) * ∏_{i=0}^{D-1} ra_i(j)
        let univariate_poly_evals: Vec<F> = (0..ra_i_polys[0].len() / 2)
            .into_par_iter()
            .map(|i| {
                let eq_evals = eq_poly.sumcheck_evals(i, degree, BindingOrder::LowToHigh);

                let mut evals = vec![F::zero(); degree];

                // Firstly compute all ra_i_evals
                let all_ra_i_evals: Vec<Vec<F>> = ra_i_polys
                    .iter()
                    .map(|ra_i_poly| ra_i_poly.sumcheck_evals(i, degree, BindingOrder::LowToHigh))
                    .collect();

                for eval_point in 0..degree {
                    let mut result = eq_evals[eval_point];

                    for ra_i_evals in all_ra_i_evals.iter() {
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

impl<F, PCS> CacheSumcheckOpenings<F, PCS> for RASumcheck<F>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    fn cache_openings_prover(
        &mut self,
        _accumulator: Option<Rc<RefCell<ProverOpeningAccumulator<F, PCS>>>>,
        _opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        debug_assert!(self.ra_i_claims.is_none());
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");

        let mut openings = vec![F::zero(); self.d];
        for i in 0..self.d {
            openings[i] = prover_state.ra_i_polys[i].final_sumcheck_claim();
        }

        self.ra_i_claims = Some(openings);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::transcript::KeccakTranscript;
    use ark_bn254::Fr;
    use ark_std::{One, Zero};
    use rand::thread_rng;

    // Test with just T = 1 (one cycle) for debugging:
    #[test]
    fn test_ra_sumcheck_tensor_decomp() {
        use rand::Rng;
        let mut rng = thread_rng();
        let d = 4;
        let T = 1;
        let k = 1 << 16;

        let one_hot_index = rng.gen::<usize>() % k;

        let mut ra_values = vec![Fr::zero(); k];
        ra_values[one_hot_index] = Fr::one();
        let ra_poly = MultilinearPolynomial::from(ra_values);

        let addresses = vec![one_hot_index];

        let r_cycle: Vec<Fr> = (0..T.log_2()).map(|_| Fr::from(rng.gen::<u64>())).collect();
        let r_address: Vec<Fr> = (0..k.log_2()).map(|_| Fr::from(rng.gen::<u64>())).collect();

        let mut eval_point = r_cycle.clone();
        eval_point.extend_from_slice(&r_address);
        let ra_claim = ra_poly.evaluate(&eval_point);

        let prover_sumcheck = RASumcheck::<Fr>::new(
            ra_claim,
            addresses,
            r_cycle.clone(),
            r_address.clone(),
            T,
            d,
        );

        let mut prover_transcript = KeccakTranscript::new(b"test_one_cycle");
        let (proof, r_cycle_bound) = prover_sumcheck.prove(&mut prover_transcript);

        let mut verifier_transcript = KeccakTranscript::new(b"test_one_cycle");

        let verify_result = RASumcheck::<Fr>::verify(
            ra_claim,
            proof.ra_i_claims,
            r_cycle,
            T,
            d,
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

    #[test]
    fn test_ra_sumcheck_large_t() {
        use rand::Rng;
        let mut rng = thread_rng();
        // pick d = 3, k = 11 so that d doesn't divide r_address.len()
        let d = 3;
        let T = 1 << 10;
        let k = 1 << 11;

        let addresses: Vec<_> = (0..T).map(|_| rng.gen::<usize>() % k).collect();
        let mut ra_values = vec![Fr::zero(); k * T];
        ra_values
            .chunks_mut(k)
            .zip(addresses.iter())
            .for_each(|(ra_chunk, k)| ra_chunk[*k] = Fr::one());

        let mut ra_poly = MultilinearPolynomial::from(ra_values);

        let r_cycle: Vec<Fr> = (0..T.log_2()).map(|_| Fr::from(rng.gen::<u64>())).collect();
        let r_address: Vec<Fr> = (0..k.log_2()).map(|_| Fr::from(rng.gen::<u64>())).collect();

        let mut eval_point = r_cycle.clone();
        eval_point.extend_from_slice(&r_address);
        let ra_claim = ra_poly.evaluate(&eval_point);

        for r in r_address.iter().rev() {
            ra_poly.bind_parallel(*r, BindingOrder::LowToHigh);
        }

        for r in r_cycle.iter().rev() {
            ra_poly.bind_parallel(*r, BindingOrder::LowToHigh);
        }
        assert_eq!(ra_poly.final_sumcheck_claim(), ra_claim);

        let prover_sumcheck = RASumcheck::<Fr>::new(
            ra_claim,
            addresses,
            r_cycle.clone(),
            r_address.clone(),
            T,
            d,
        );

        let mut prover_transcript = KeccakTranscript::new(b"test_t_large");
        let (proof, r_cycle_bound) = prover_sumcheck.prove(&mut prover_transcript);

        let mut verifier_transcript = KeccakTranscript::new(b"test_t_large");
        verifier_transcript.compare_to(prover_transcript);

        let verify_result = RASumcheck::<Fr>::verify(
            ra_claim,
            proof.ra_i_claims,
            r_cycle,
            T,
            d,
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
