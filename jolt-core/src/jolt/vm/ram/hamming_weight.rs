use std::{cell::RefCell, rc::Rc};

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use rayon::prelude::*;

use crate::{
    dag::state_manager::StateManager,
    field::JoltField,
    jolt::{
        vm::ram::{compute_d_parameter, remap_address, NUM_RA_I_VARS},
        witness::{CommittedPolynomial, VirtualPolynomial},
    },
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_proof::{
            OpeningPoint, ProverOpeningAccumulator, SumcheckId, VerifierOpeningAccumulator,
            BIG_ENDIAN,
        },
    },
    subprotocols::sumcheck::{SumcheckInstance, SumcheckInstanceProof},
    utils::{thread::unsafe_allocate_zero_vec, transcript::Transcript},
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
    /// gamma powers for batching
    gamma_powers: Vec<F>,
    /// D parameter as in Twist and Shout paper
    d: usize,
}

pub struct HammingWeightVerifierState<F: JoltField> {
    /// D parameter as in Twist and Shout paper
    d: usize,
    /// gamma powers for verification
    gamma_powers: Vec<F>,
}

pub struct HammingWeightSumcheck<F: JoltField> {
    /// The initial claim (sum of gamma powers for hamming weight)
    input_claim: F,
    r_cycle: Vec<F>,
    /// Prover state
    prover_state: Option<HammingWeightProverState<F>>,
    /// Verifier state
    verifier_state: Option<HammingWeightVerifierState<F>>,
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
        let d = compute_d_parameter(K);

        let (_, trace, program_io, _) = state_manager.get_prover_data();
        let memory_layout = &program_io.memory_layout;

        let T = trace.len();
        let num_chunks = rayon::current_num_threads().next_power_of_two().min(T);
        let chunk_size = (T / num_chunks).max(1);

        let gamma: F = state_manager.transcript.borrow_mut().challenge_scalar();
        let mut gamma_powers = vec![F::one(); d];
        for i in 1..d {
            gamma_powers[i] = gamma_powers[i - 1] * gamma;
        }

        let (r_cycle, hamming_booleanity_claim) = state_manager.get_virtual_polynomial_opening(
            VirtualPolynomial::RamHammingWeight,
            SumcheckId::RamHammingBooleanity,
        );

        let eq_r_cycle = EqPolynomial::evals(&r_cycle.r);

        let mut F_arrays = Vec::with_capacity(d);
        for i in 0..d {
            let F: Vec<F> = trace
                .par_chunks(chunk_size)
                .enumerate()
                .map(|(chunk_index, trace_chunk)| {
                    let mut local_array = unsafe_allocate_zero_vec(1 << NUM_RA_I_VARS);
                    let mut j = chunk_index * chunk_size;
                    for cycle in trace_chunk {
                        if let Some(address) =
                            remap_address(cycle.ram_access().address() as u64, memory_layout)
                        {
                            // For each address, add eq_r_cycle[j] to each corresponding chunk
                            // This maintains the property that sum of all ra values for an address equals 1
                            let address_i =
                                (address >> (NUM_RA_I_VARS * (d - 1 - i))) % (1 << NUM_RA_I_VARS);
                            local_array[address_i as usize] += eq_r_cycle[j];
                        }
                        j += 1;
                    }
                    local_array
                })
                .reduce(
                    || unsafe_allocate_zero_vec(1 << NUM_RA_I_VARS),
                    |mut running, new| {
                        running
                            .par_iter_mut()
                            .zip(new.into_par_iter())
                            .for_each(|(x, y)| *x += y);
                        running
                    },
                );
            F_arrays.push(F);
        }

        // Create MultilinearPolynomials from F arrays
        let ra: Vec<MultilinearPolynomial<F>> = F_arrays
            .into_iter()
            .map(MultilinearPolynomial::from)
            .collect();

        let input_claim = hamming_booleanity_claim * gamma_powers.iter().sum::<F>();

        Self {
            input_claim,
            prover_state: Some(HammingWeightProverState {
                ra,
                gamma_powers,
                d,
            }),
            verifier_state: None,
            r_cycle: r_cycle.r.clone(),
            d,
        }
    }

    pub fn new_verifier<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        K: usize,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Self {
        // Calculate D dynamically such that 2^8 = K^(1/D)
        let d = compute_d_parameter(K);

        let gamma: F = state_manager.transcript.borrow_mut().challenge_scalar();
        let mut gamma_powers = vec![F::one(); d];
        for i in 1..d {
            gamma_powers[i] = gamma_powers[i - 1] * gamma;
        }

        let (r_cycle, hamming_booleanity_claim) = state_manager.get_virtual_polynomial_opening(
            VirtualPolynomial::RamHammingWeight,
            SumcheckId::RamHammingBooleanity,
        );

        let input_claim = hamming_booleanity_claim * gamma_powers.iter().sum::<F>();

        Self {
            input_claim,
            prover_state: None,
            verifier_state: Some(HammingWeightVerifierState { d, gamma_powers }),
            r_cycle: r_cycle.r.clone(),
            d,
        }
    }
}

impl<F: JoltField> SumcheckInstance<F> for HammingWeightSumcheck<F> {
    fn degree(&self) -> usize {
        1
    }

    fn num_rounds(&self) -> usize {
        NUM_RA_I_VARS
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
            .zip(prover_state.gamma_powers.par_iter())
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

    fn expected_output_claim(
        &self,
        accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
        _r: &[F],
    ) -> F {
        let verifier_state = self
            .verifier_state
            .as_ref()
            .expect("Verifier state not initialized");

        let ra_claims: Vec<_> = (0..self.d)
            .map(|i| {
                accumulator
                    .as_ref()
                    .unwrap()
                    .borrow()
                    .get_committed_polynomial_opening(
                        CommittedPolynomial::RamRa(i),
                        SumcheckId::RamHammingWeight,
                    )
                    .1
            })
            .collect();

        // Compute batched claim: sum_{i=0}^{d-1} z^i * ra_i
        ra_claims
            .iter()
            .zip(verifier_state.gamma_powers.iter())
            .map(|(ra_claim, z_power)| *ra_claim * z_power)
            .sum()
    }

    fn normalize_opening_point(&self, opening_point: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::new(opening_point.iter().copied().rev().collect())
    }

    fn cache_openings_prover(
        &self,
        accumulator: Rc<RefCell<ProverOpeningAccumulator<F>>>,
        r_address: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");

        let claims: Vec<F> = prover_state
            .ra
            .iter()
            .map(|ra_i| ra_i.final_sumcheck_claim())
            .collect();

        accumulator.borrow_mut().append_sparse(
            (0..self.d).map(CommittedPolynomial::RamRa).collect(),
            SumcheckId::RamHammingWeight,
            r_address.r,
            self.r_cycle.clone(),
            claims,
        );
    }

    fn cache_openings_verifier(
        &self,
        accumulator: Rc<RefCell<VerifierOpeningAccumulator<F>>>,
        r_address: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let opening_point: OpeningPoint<BIG_ENDIAN, F> =
            OpeningPoint::new([r_address.r.as_slice(), self.r_cycle.as_slice()].concat());

        accumulator.borrow_mut().append_sparse(
            (0..self.d).map(CommittedPolynomial::RamRa).collect(),
            SumcheckId::RamHammingWeight,
            opening_point.r,
        );
    }
}
