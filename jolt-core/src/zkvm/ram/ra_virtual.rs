use std::cell::RefCell;
use std::rc::Rc;

use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::multilinear_polynomial::PolynomialEvaluation;
use crate::poly::opening_proof::{
    OpeningPoint, ProverOpeningAccumulator, SumcheckId, VerifierOpeningAccumulator, BIG_ENDIAN,
};
use crate::zkvm::dag::state_manager::StateManager;
use crate::zkvm::ram::remap_address;
use crate::zkvm::witness::{
    compute_d_parameter, CommittedPolynomial, VirtualPolynomial, DTH_ROOT_OF_K,
};
use crate::{
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
    },
    subprotocols::sumcheck::{SumcheckInstance, SumcheckInstanceProof},
    transcript::Transcript,
    utils::math::Math,
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use rayon::prelude::*;

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
    rlc_coeffs: [F; 3],
    /// Random challenge r_cycle
    r_cycle: [Vec<F>; 3],
    r_address_chunks: Vec<Vec<F>>,
    /// [ra(r_address, r_cycle_val), ra(r_address, r_cycle_rw), ra(r_address, r_cycle_raf)]
    ra_claim: F,
    /// Number of decomposition parts
    d: usize,
    /// Length of the trace
    T: usize,
    prover_state: Option<RAProverState<F>>,
}

impl<F: JoltField> RASumcheck<F> {
    #[tracing::instrument(skip_all, name = "RaVirtualization::new_prover")]
    pub fn new_prover<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        K: usize,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Self {
        // Calculate d dynamically such that 2^8 = K^(1/D)
        let d = compute_d_parameter(K);
        let log_K = K.log_2();

        let (preprocessing, trace, _, _) = state_manager.get_prover_data();
        let T = trace.len();

        // These two sumchecks have the same binding order and number of rounds,
        // and they're run in parallel, so the openings are the same.
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

        let (r, ra_claim_val) = state_manager.get_virtual_polynomial_opening(
            VirtualPolynomial::RamRa,
            SumcheckId::RamValFinalEvaluation,
        );
        let (r_address, r_cycle_val) = r.split_at_r(log_K);

        let (r, ra_claim_rw) = state_manager.get_virtual_polynomial_opening(
            VirtualPolynomial::RamRa,
            SumcheckId::RamReadWriteChecking,
        );
        let (r_address_rw, r_cycle_rw) = r.split_at_r(log_K);
        assert_eq!(r_address, r_address_rw);

        let (r, ra_claim_raf) = state_manager
            .get_virtual_polynomial_opening(VirtualPolynomial::RamRa, SumcheckId::RamRafEvaluation);
        let (r_address_raf, r_cycle_raf) = r.split_at_r(log_K);
        assert_eq!(r_address, r_address_raf);

        let r_address = if r_address.len().is_multiple_of(DTH_ROOT_OF_K.log_2()) {
            r_address.to_vec()
        } else {
            // Pad with zeros
            [
                &vec![F::zero(); DTH_ROOT_OF_K.log_2() - (r_address.len() % DTH_ROOT_OF_K.log_2())],
                r_address,
            ]
            .concat()
        };
        // Split r_address into d chunks of variable sizes
        let r_address_chunks: Vec<Vec<F>> = r_address
            .chunks(DTH_ROOT_OF_K.log_2())
            .map(|chunk| chunk.to_vec())
            .collect();
        debug_assert_eq!(r_address_chunks.len(), d);

        // Precompute EQ tables for each chunk
        let eq_tables: Vec<Vec<F>> = r_address_chunks
            .iter()
            .map(|chunk| EqPolynomial::evals(chunk))
            .collect();

        let gamma: F = state_manager
            .get_transcript()
            .borrow_mut()
            .challenge_scalar();
        let rlc_coeffs = [F::one(), gamma, gamma.square()];

        let eq_poly = MultilinearPolynomial::linear_combination(
            &[
                &EqPolynomial::evals(r_cycle_val).into(),
                &EqPolynomial::evals(r_cycle_rw).into(),
                &EqPolynomial::evals(r_cycle_raf).into(),
            ],
            &rlc_coeffs,
        );
        let combined_ra_claim = rlc_coeffs[0] * ra_claim_val
            + rlc_coeffs[1] * ra_claim_rw
            + rlc_coeffs[2] * ra_claim_raf;

        let ra_i_polys: Vec<MultilinearPolynomial<F>> = (0..d)
            .into_par_iter()
            .map(|i| {
                let ra_i: Vec<F> = trace
                    .par_iter()
                    .map(|cycle| {
                        remap_address(
                            cycle.ram_access().address() as u64,
                            &preprocessing.shared.memory_layout,
                        )
                        .map_or(F::zero(), |address| {
                            // For each address, add eq_r_cycle[j] to each corresponding chunk
                            // This maintains the property that sum of all ra values for an address equals 1
                            let address_i = (address >> (DTH_ROOT_OF_K.log_2() * (d - 1 - i)))
                                % DTH_ROOT_OF_K as u64;

                            eq_tables[i][address_i as usize]
                        })
                    })
                    .collect();
                ra_i.into()
            })
            .collect();

        Self {
            rlc_coeffs,
            ra_claim: combined_ra_claim,
            d,
            prover_state: Some(RAProverState {
                ra_i_polys,
                eq_poly,
            }),
            T,
            r_cycle: [
                r_cycle_val.to_vec(),
                r_cycle_rw.to_vec(),
                r_cycle_raf.to_vec(),
            ],
            r_address_chunks,
        }
    }

    pub fn new_verifier<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        K: usize,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Self {
        // Calculate D dynamically such that 2^8 = K^(1/D)
        let d = compute_d_parameter(K);
        let log_K = K.log_2();

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

        let (r, ra_claim_val) = state_manager.get_virtual_polynomial_opening(
            VirtualPolynomial::RamRa,
            SumcheckId::RamValFinalEvaluation,
        );
        let (r_address, r_cycle_val) = r.split_at_r(log_K);

        let (r, ra_claim_rw) = state_manager.get_virtual_polynomial_opening(
            VirtualPolynomial::RamRa,
            SumcheckId::RamReadWriteChecking,
        );
        let (r_address_rw, r_cycle_rw) = r.split_at_r(log_K);
        assert_eq!(r_address, r_address_rw);

        let (r, ra_claim_raf) = state_manager
            .get_virtual_polynomial_opening(VirtualPolynomial::RamRa, SumcheckId::RamRafEvaluation);
        let (r_address_raf, r_cycle_raf) = r.split_at_r(log_K);
        assert_eq!(r_address, r_address_raf);

        let r_address = if r_address.len().is_multiple_of(DTH_ROOT_OF_K.log_2()) {
            r_address.to_vec()
        } else {
            // Pad with zeros
            [
                &vec![F::zero(); DTH_ROOT_OF_K.log_2() - (r_address.len() % DTH_ROOT_OF_K.log_2())],
                r_address,
            ]
            .concat()
        };
        // Split r_address into d chunks of variable sizes
        let r_address_chunks: Vec<Vec<F>> = r_address
            .chunks(DTH_ROOT_OF_K.log_2())
            .map(|chunk| chunk.to_vec())
            .collect();
        debug_assert_eq!(r_address_chunks.len(), d);

        let gamma: F = state_manager
            .get_transcript()
            .borrow_mut()
            .challenge_scalar();
        let rlc_coeffs = [F::one(), gamma, gamma.square()];

        let combined_ra_claim = rlc_coeffs[0] * ra_claim_val
            + rlc_coeffs[1] * ra_claim_rw
            + rlc_coeffs[2] * ra_claim_raf;

        Self {
            rlc_coeffs,
            ra_claim: combined_ra_claim,
            d,
            r_cycle: [
                r_cycle_val.to_vec(),
                r_cycle_rw.to_vec(),
                r_cycle_raf.to_vec(),
            ],
            r_address_chunks,
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
        let eq_eval = self.rlc_coeffs[0] * EqPolynomial::mle(&self.r_cycle[0], &r_rev)
            + self.rlc_coeffs[1] * EqPolynomial::mle(&self.r_cycle[1], &r_rev)
            + self.rlc_coeffs[2] * EqPolynomial::mle(&self.r_cycle[2], &r_rev);

        // Compute the product of all ra_i evaluations
        let mut product = F::one();
        for i in 0..self.d {
            let accumulator = accumulator.as_ref().unwrap();
            let accumulator = accumulator.borrow();
            let (_, ra_i_claim) = accumulator.get_committed_polynomial_opening(
                CommittedPolynomial::RamRa(i),
                SumcheckId::RamRaVirtualization,
            );
            product *= ra_i_claim;
        }
        eq_eval * product
    }

    #[tracing::instrument(skip_all, name = "RaVirtualization::compute_prover_message")]
    fn compute_prover_message(&mut self, _round: usize, _previous_claim: F) -> Vec<F> {
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
            let claim = prover_state.ra_i_polys[i].final_sumcheck_claim();
            accumulator.borrow_mut().append_sparse(
                vec![CommittedPolynomial::RamRa(i)],
                SumcheckId::RamRaVirtualization,
                self.r_address_chunks[i].clone(),
                r_cycle.r.clone(),
                vec![claim],
            );
        }
    }

    fn cache_openings_verifier(
        &self,
        accumulator: Rc<RefCell<VerifierOpeningAccumulator<F>>>,
        r_cycle: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        for i in 0..self.d {
            let opening_point =
                [self.r_address_chunks[i].as_slice(), r_cycle.r.as_slice()].concat();
            accumulator.borrow_mut().append_sparse(
                vec![CommittedPolynomial::RamRa(i)],
                SumcheckId::RamRaVirtualization,
                opening_point,
            );
        }
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use crate::transcript::KeccakTranscript;
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
