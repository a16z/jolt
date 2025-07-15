use std::{cell::RefCell, rc::Rc};

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use rayon::prelude::*;

use crate::{
    dag::state_manager::StateManager,
    field::JoltField,
    jolt::vm::ram::remap_address,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_proof::{
            OpeningPoint, OpeningsKeys, ProverOpeningAccumulator, VerifierOpeningAccumulator,
            BIG_ENDIAN,
        },
    },
    subprotocols::sumcheck::{
        BatchableSumcheckInstance, CacheSumcheckOpenings, SumcheckInstanceProof,
    },
    utils::{math::Math, thread::unsafe_allocate_zero_vec, transcript::Transcript},
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
    r_cycle: Vec<F>,
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
    #[tracing::instrument(skip_all, name = "RamHammingWeightSumcheck::new_prover")]
    pub fn new_prover<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        K: usize,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Self {
        // Calculate D dynamically such that 2^8 = K^(1/D)
        let log_K = K.log_2();
        let d = (log_K / 8).max(1);

        let (_, trace, program_io, _) = state_manager.get_prover_data();
        let memory_layout = &program_io.memory_layout;

        let T = trace.len();
        let num_chunks = rayon::current_num_threads().next_power_of_two().min(T);
        let chunk_size = (T / num_chunks).max(1);

        // Get z challenges for batching
        let z_challenges: Vec<F> = state_manager.transcript.borrow_mut().challenge_vector(d);
        let mut z_powers = vec![F::one(); d];
        for i in 1..d {
            z_powers[i] = z_powers[i - 1] * z_challenges[0];
        }

        let r_cycle: Vec<F> = state_manager
            .transcript
            .borrow_mut()
            .challenge_vector(T.log_2());
        let eq_r_cycle = EqPolynomial::evals(&r_cycle);

        // Calculate variable chunk sizes for address decomposition
        let log_k = K.log_2();
        let base_chunk_size = log_k / d;
        let remainder = log_k % d;
        let chunk_sizes: Vec<usize> = (0..d)
            .map(|i| {
                if i < remainder {
                    base_chunk_size + 1
                } else {
                    base_chunk_size
                }
            })
            .collect();

        // Compute F arrays for each decomposed part
        let F_arrays: Vec<Vec<F>> = trace
            .par_chunks(chunk_size)
            .enumerate()
            .map(|(chunk_index, trace_chunk)| {
                // TODO(moodlezoup): Can be K^(1/d)-sized vectors
                let mut local_arrays: Vec<Vec<F>> = vec![unsafe_allocate_zero_vec(K); d];
                let mut j = chunk_index * chunk_size;
                for cycle in trace_chunk {
                    let address =
                        remap_address(cycle.ram_access().address() as u64, memory_layout) as usize;

                    // For each address, add eq_r_cycle[j] to each corresponding chunk
                    // This maintains the property that sum of all ra values for an address equals 1
                    let mut remaining_address = address;
                    let mut chunk_values = Vec::with_capacity(d);

                    // Decompose address into chunks
                    for i in 0..d {
                        let chunk_size = chunk_sizes[d - 1 - i];
                        let chunk_modulo = 1 << chunk_size;
                        let chunk_value = remaining_address % chunk_modulo;
                        chunk_values.push(chunk_value);
                        remaining_address /= chunk_modulo;
                    }

                    // Add eq_r_cycle contribution to each ra polynomial
                    for (i, &chunk_value) in chunk_values.iter().enumerate() {
                        local_arrays[d - 1 - i][chunk_value] += eq_r_cycle[j];
                    }
                    j += 1;
                }
                local_arrays
            })
            .reduce(
                || vec![unsafe_allocate_zero_vec(K); d],
                |mut running, new| {
                    running.par_iter_mut().zip(new.into_par_iter()).for_each(
                        |(running_arr, new_arr)| {
                            running_arr
                                .par_iter_mut()
                                .zip(new_arr.into_par_iter())
                                .for_each(|(x, y)| *x += y);
                        },
                    );
                    running
                },
            );

        // Create MultilinearPolynomials from F arrays
        let ra: Vec<MultilinearPolynomial<F>> = F_arrays
            .into_iter()
            .map(MultilinearPolynomial::from)
            .collect();

        // Compute input claim as sum of z powers
        let input_claim = z_powers.iter().sum();

        Self {
            input_claim,
            prover_state: Some(HammingWeightProverState { ra, z_powers, d }),
            verifier_state: None,
            cached_claims: None,
            r_cycle,
            d,
        }
    }

    pub fn new_verifier<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        K: usize,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Self {
        let (_, _, T) = state_manager.get_verifier_data();

        // Calculate D dynamically such that 2^8 = K^(1/D)
        let log_K = K.log_2();
        let d = (log_K / 8).max(1);

        // Get z challenges for batching
        let z_challenges: Vec<F> = state_manager.transcript.borrow_mut().challenge_vector(d);
        let mut z_powers = vec![F::one(); d];
        for i in 1..d {
            z_powers[i] = z_powers[i - 1] * z_challenges[0];
        }

        let r_cycle: Vec<F> = state_manager
            .transcript
            .borrow_mut()
            .challenge_vector(T.log_2());

        // Compute input claim as sum of z powers
        let input_claim = z_powers.iter().sum();

        let ra_claims = (0..d)
            .map(|i| state_manager.get_opening(OpeningsKeys::RamHammingRa(i)))
            .collect();

        Self {
            input_claim,
            prover_state: None,
            verifier_state: Some(HammingWeightVerifierState { log_K, d, z_powers }),
            cached_claims: Some(ra_claims),
            r_cycle,
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
            verifier_state.log_K / self.d
        } else {
            panic!("Neither prover state nor verifier state is initialized")
        }
    }

    fn input_claim(&self) -> F {
        self.input_claim
    }

    #[tracing::instrument(skip_all, name = "RamHammingWeightSumcheck::compute_prover_message")]
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

    #[tracing::instrument(skip_all, name = "RamHammingWeightSumcheck::bind")]
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
    fn normalize_opening_point(&self, opening_point: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::new(opening_point.iter().copied().rev().collect())
    }

    fn cache_openings_prover(
        &mut self,
        accumulator: Option<Rc<RefCell<ProverOpeningAccumulator<F, PCS>>>>,
        r_address: OpeningPoint<BIG_ENDIAN, F>,
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

        let opening_point =
            OpeningPoint::new([r_address.r.as_slice(), self.r_cycle.as_slice()].concat());

        let accumulator = accumulator.expect("accumulator is needed");
        prover_state.ra.iter().enumerate().for_each(|(i, ra_i)| {
            accumulator.borrow_mut().append_virtual(
                OpeningsKeys::RamHammingRa(i),
                opening_point.clone(),
                ra_i.final_sumcheck_claim(),
            );
        });
    }

    fn cache_openings_verifier(
        &mut self,
        accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F, PCS>>>>,
        r_address: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let opening_point =
            OpeningPoint::new([r_address.r.as_slice(), self.r_cycle.as_slice()].concat());

        let accumulator = accumulator.expect("accumulator is needed");
        (0..self.d).for_each(|i| {
            accumulator
                .borrow_mut()
                .populate_claim_opening(OpeningsKeys::RamHammingRa(i), opening_point.clone())
        });
    }
}
