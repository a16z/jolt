use std::cell::RefCell;
use std::rc::Rc;

use crate::dag::state_manager::StateManager;
use crate::jolt::vm::ram::remap_address;
use crate::jolt::witness::{CommittedPolynomial, VirtualPolynomial};
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::multilinear_polynomial::PolynomialEvaluation;
use crate::poly::opening_proof::{
    OpeningPoint, ProverOpeningAccumulator, SumcheckId, VerifierOpeningAccumulator, BIG_ENDIAN,
};
use crate::{
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
    },
    subprotocols::sumcheck::{SumcheckInstance, SumcheckInstanceProof},
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
}

pub struct RASumcheck<F: JoltField> {
    /// Random challenge r_cycle
    r_cycle: Vec<F>,
    /// ra(r_cycle, r_address)
    ra_claim: F,
    /// Number of decomposition parts
    d: usize,
    /// Length of the trace
    T: usize,
    /// Prover state
    prover_state: Option<RAProverState<F>>,
}

impl<F: JoltField> RASumcheck<F> {
    #[tracing::instrument(skip_all, name = "RaVirtualization::new_prover")]
    pub fn new_prover<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        K: usize,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Self {
        // Calculate D dynamically such that 2^8 = K^(1/D)
        let log_K = K.log_2();
        let d = (log_K / 8).max(1);

        let (preprocessing, trace, _, _) = state_manager.get_prover_data();
        let T = trace.len();

        let (r, ra_claim) = state_manager.get_virtual_polynomial_opening(
            VirtualPolynomial::RamRa,
            SumcheckId::RamValFinalEvaluation,
        );
        let (r_address, r_cycle) = r.split_at_r(log_K);

        let base_chunk_size = log_K / d;
        let remainder = log_K % d;

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

        let eq_poly = MultilinearPolynomial::from(EqPolynomial::evals(r_cycle));

        // Precompute EQ tables for each chunk
        let eq_tables: Vec<Vec<F>> = r_address_chunks
            .iter()
            .map(|chunk| EqPolynomial::evals(chunk))
            .collect();

        // We construct ra_i directly from a list of addresses.
        // This way we avoid |ra_i| = k^(1/d) * T size, but rather just |ra_i| = T.

        let mut ra_i_vecs: Vec<Vec<F>> = (0..d).map(|_| vec![F::zero(); T]).collect();

        // For each address, decompose it and add the corresponding EQ evaluations
        for (cycle_idx, &cycle) in trace.iter().enumerate() {
            let address = remap_address(
                cycle.ram_access().address() as u64,
                &preprocessing.shared.memory_layout,
            ) as usize;

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
            }),
            T,
            r_cycle: r_cycle.to_vec(),
        }
    }

    pub fn new_verifier<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        K: usize,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Self {
        // Calculate D dynamically such that 2^8 = K^(1/D)
        let log_K = K.log_2();
        let d = (log_K / 8).max(1);

        let (_, _, T) = state_manager.get_verifier_data();

        assert_eq!(
            state_manager.get_virtual_polynomial_opening(
                VirtualPolynomial::RamRa,
                SumcheckId::RamValFinalEvaluation,
            ),
            state_manager.get_virtual_polynomial_opening(
                VirtualPolynomial::RamRa,
                SumcheckId::RamValEvaluation,
            )
        );

        let (r, ra_claim) = state_manager.get_virtual_polynomial_opening(
            VirtualPolynomial::RamRa,
            SumcheckId::RamValFinalEvaluation,
        );
        let (_r_address, r_cycle) = r.split_at(log_K);

        Self {
            ra_claim,
            d,
            r_cycle: r_cycle.into(),
            T,
            prover_state: None,
        }
    }
}

impl<F: JoltField> SumcheckInstance<F> for RASumcheck<F> {
    fn degree(&self) -> usize {
        self.d + 1
    }

    fn num_rounds(&self) -> usize {
        self.T.log_2()
    }

    #[tracing::instrument(skip_all, name = "RaVirtualization::bind")]
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

    fn expected_output_claim(
        &self,
        accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
        r: &[F],
    ) -> F {
        // we need opposite endian-ness here
        let r_rev: Vec<_> = r.iter().cloned().rev().collect();
        let eq_eval = EqPolynomial::mle(&self.r_cycle, &r_rev);

        // Compute the product of all ra_i evaluations
        let mut product = F::one();
        for i in 0..self.d {
            let accumulator = accumulator.as_ref().unwrap();
            let accumulator = accumulator.borrow();
            let (_, ra_i_claim) = accumulator.get_committed_polynomial_opening(
                CommittedPolynomial::RamRa(i),
                SumcheckId::RamRaVirtualization(1),
            );
            product *= ra_i_claim;
        }
        eq_eval * product
    }

    #[tracing::instrument(skip_all, name = "RaVirtualization::compute_prover_message")]
    fn compute_prover_message(&mut self, _round: usize) -> Vec<F> {
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");
        let degree = <Self as SumcheckInstance<F>>::degree(self);
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

    fn normalize_opening_point(&self, opening_point: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::new(opening_point.iter().copied().rev().collect())
    }

    fn cache_openings_prover(
        &self,
        accumulator: Rc<RefCell<ProverOpeningAccumulator<F>>>,
        r_cycle: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");

        for i in 0..self.d {
            // TODO()
            let claim = prover_state.ra_i_polys[i].final_sumcheck_claim();
            // accumulator.borrow_mut().append_sparse(
            //     vec![CommittedPolynomial::RamRa(i)],
            //     SumcheckId::RamRaVirtualization,
            //     todo!(),
            //     r_cycle.r,
            //     vec![claim],
            // );
        }
    }

    fn cache_openings_verifier(
        &self,
        accumulator: Rc<RefCell<VerifierOpeningAccumulator<F>>>,
        r_cycle_prime: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        todo!()
        // let mut r_address = accumulator
        //     .borrow()
        //     .get_opening_point(OpeningId::ValFinalWa)
        //     .unwrap();
        // let _r_cycle = r_address.split_off(r_address.len() - r_cycle_prime.len());

        // let ra_opening_point =
        //     OpeningPoint::new([r_address.r.as_slice(), r_cycle_prime.r.as_slice()].concat());

        // for i in 0..self.d {
        //     accumulator
        //         .borrow_mut()
        //         .append_virtual(OpeningId::RamRaVirtualization(i), ra_opening_point.clone());
        // }
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use crate::utils::transcript::KeccakTranscript;
//     use ark_bn254::Fr;
//     use ark_std::{One, Zero};
//     use rand::thread_rng;

//     // Test with just T = 1 (one cycle) for debugging:
//     #[test]
//     fn test_ra_sumcheck_tensor_decomp() {
//         use rand::Rng;
//         let mut rng = thread_rng();
//         let d = 4;
//         let T = 1;
//         let k = 1 << 16;

//         let one_hot_index = rng.gen::<usize>() % k;

//         let mut ra_values = vec![Fr::zero(); k];
//         ra_values[one_hot_index] = Fr::one();
//         let ra_poly = MultilinearPolynomial::from(ra_values);

//         let addresses = vec![one_hot_index];

//         let r_cycle: Vec<Fr> = (0..T.log_2()).map(|_| Fr::from(rng.gen::<u64>())).collect();
//         let r_address: Vec<Fr> = (0..k.log_2()).map(|_| Fr::from(rng.gen::<u64>())).collect();

//         let mut eval_point = r_cycle.clone();
//         eval_point.extend_from_slice(&r_address);
//         let ra_claim = ra_poly.evaluate(&eval_point);

//         let prover_sumcheck = RASumcheck::<Fr>::new_prover(
//             ra_claim,
//             addresses,
//             r_cycle.clone(),
//             r_address.clone(),
//             T,
//             d,
//         );

//         let mut prover_transcript = KeccakTranscript::new(b"test_one_cycle");
//         let (proof, r_cycle_bound) = prover_sumcheck.prove(&mut prover_transcript);

//         let mut verifier_transcript = KeccakTranscript::new(b"test_one_cycle");

//         let verify_result = RASumcheck::<Fr>::verify(
//             ra_claim,
//             proof.ra_i_claims,
//             r_cycle,
//             T,
//             d,
//             &proof.sumcheck_proof,
//             &mut verifier_transcript,
//         );

//         assert!(verify_result.is_ok(), "Verification failed");
//         let verified_r_cycle_bound = verify_result.unwrap();
//         assert_eq!(
//             r_cycle_bound, verified_r_cycle_bound,
//             "r_cycle_bound mismatch"
//         );
//     }

//     #[test]
//     fn test_ra_sumcheck_large_t() {
//         use rand::Rng;
//         let mut rng = thread_rng();
//         // pick d = 3, k = 11 so that d doesn't divide r_address.len()
//         let d = 3;
//         let T = 1 << 10;
//         let k = 1 << 11;

//         let addresses: Vec<_> = (0..T).map(|_| rng.gen::<usize>() % k).collect();
//         let mut ra_values = vec![Fr::zero(); k * T];
//         ra_values
//             .chunks_mut(k)
//             .zip(addresses.iter())
//             .for_each(|(ra_chunk, k)| ra_chunk[*k] = Fr::one());

//         let mut ra_poly = MultilinearPolynomial::from(ra_values);

//         let r_cycle: Vec<Fr> = (0..T.log_2()).map(|_| Fr::from(rng.gen::<u64>())).collect();
//         let r_address: Vec<Fr> = (0..k.log_2()).map(|_| Fr::from(rng.gen::<u64>())).collect();

//         let mut eval_point = r_cycle.clone();
//         eval_point.extend_from_slice(&r_address);
//         let ra_claim = ra_poly.evaluate(&eval_point);

//         for r in r_address.iter().rev() {
//             ra_poly.bind_parallel(*r, BindingOrder::LowToHigh);
//         }

//         for r in r_cycle.iter().rev() {
//             ra_poly.bind_parallel(*r, BindingOrder::LowToHigh);
//         }
//         assert_eq!(ra_poly.final_sumcheck_claim(), ra_claim);

//         let prover_sumcheck = RASumcheck::<Fr>::new_prover(
//             ra_claim,
//             addresses,
//             r_cycle.clone(),
//             r_address.clone(),
//             T,
//             d,
//         );

//         let mut prover_transcript = KeccakTranscript::new(b"test_t_large");
//         let (proof, r_cycle_bound) = prover_sumcheck.prove(&mut prover_transcript);

//         let mut verifier_transcript = KeccakTranscript::new(b"test_t_large");
//         verifier_transcript.compare_to(prover_transcript);

//         let verify_result = RASumcheck::<Fr>::verify(
//             ra_claim,
//             proof.ra_i_claims,
//             r_cycle,
//             T,
//             d,
//             &proof.sumcheck_proof,
//             &mut verifier_transcript,
//         );

//         assert!(verify_result.is_ok(), "Verification failed");
//         let verified_r_cycle_bound = verify_result.unwrap();
//         assert_eq!(
//             r_cycle_bound, verified_r_cycle_bound,
//             "r_cycle_bound mismatch"
//         );
//     }
// }
