use std::cell::RefCell;

use allocative::Allocative;
use rayon::prelude::*;

use crate::{
    field::JoltField,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        eq_poly::EqPolynomial,
        multilinear_polynomial::MultilinearPolynomial,
        opening_proof::{OpeningAccumulator, SumcheckId},
    },
    subprotocols::hamming_weight::{
        HammingWeightConfig, HammingWeightProverState, HammingWeightSumcheck,
    },
    transcripts::Transcript,
    utils::{math::Math, thread::unsafe_allocate_zero_vec},
    zkvm::{
        dag::state_manager::StateManager,
        ram::remap_address,
        witness::{compute_d_parameter, CommittedPolynomial, VirtualPolynomial, DTH_ROOT_OF_K},
    },
};

#[derive(Allocative)]
pub struct RamHammingWeightSumcheck<F: JoltField> {
    d: usize,
    gamma_powers: Vec<F>,
    prover_state: Option<HammingWeightProverState<F>>,
}

impl<F: JoltField> RamHammingWeightSumcheck<F> {
    #[tracing::instrument(skip_all, name = "RamHammingWeightSumcheck::new_prover")]
    pub fn new_prover<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Self {
        // Calculate D dynamically such that 2^8 = K^(1/D)
        let d = compute_d_parameter(state_manager.ram_K);

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

        let (r_cycle, _) = state_manager.get_virtual_polynomial_opening(
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
                    let mut local_array = unsafe_allocate_zero_vec(DTH_ROOT_OF_K);
                    let mut j = chunk_index * chunk_size;
                    for cycle in trace_chunk {
                        if let Some(address) =
                            remap_address(cycle.ram_access().address() as u64, memory_layout)
                        {
                            // For each address, add eq_r_cycle[j] to each corresponding chunk
                            // This maintains the property that sum of all ra values for an address equals 1
                            let address_i = (address >> (DTH_ROOT_OF_K.log_2() * (d - 1 - i)))
                                % DTH_ROOT_OF_K as u64;
                            local_array[address_i as usize] += eq_r_cycle[j];
                        }
                        j += 1;
                    }
                    local_array
                })
                .reduce(
                    || unsafe_allocate_zero_vec(DTH_ROOT_OF_K),
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

        Self {
            d,
            gamma_powers,
            prover_state: Some(HammingWeightProverState { ra }),
        }
    }

    pub fn new_verifier<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Self {
        // Calculate D dynamically such that 2^8 = K^(1/D)
        let d = compute_d_parameter(state_manager.ram_K);

        let gamma: F = state_manager.transcript.borrow_mut().challenge_scalar();
        let mut gamma_powers = vec![F::one(); d];
        for i in 1..d {
            gamma_powers[i] = gamma_powers[i - 1] * gamma;
        }

        Self {
            d,
            gamma_powers,
            prover_state: None,
        }
    }
}

impl<F: JoltField> HammingWeightConfig for RamHammingWeightSumcheck<F> {
    fn d(&self) -> usize {
        self.d
    }

    fn num_rounds(&self) -> usize {
        DTH_ROOT_OF_K.log_2()
    }

    fn polynomial_type(i: usize) -> CommittedPolynomial {
        CommittedPolynomial::RamRa(i)
    }

    fn sumcheck_id() -> SumcheckId {
        SumcheckId::RamHammingWeight
    }
}

impl<F: JoltField, T: Transcript> HammingWeightSumcheck<F, T> for RamHammingWeightSumcheck<F> {
    fn gamma(&self) -> &[F] {
        &self.gamma_powers
    }

    fn prover_state(&self) -> Option<&HammingWeightProverState<F>> {
        self.prover_state.as_ref()
    }

    fn prover_state_mut(&mut self) -> Option<&mut HammingWeightProverState<F>> {
        self.prover_state.as_mut()
    }

    fn get_r_cycle(&self, accumulator: &dyn OpeningAccumulator<F>) -> Vec<F::Challenge> {
        accumulator
            .get_virtual_polynomial_opening(
                VirtualPolynomial::RamHammingWeight,
                SumcheckId::RamHammingBooleanity,
            )
            .0
            .r
    }

    // Override input claim for RAM-specific behavior
    fn hamming_weight_input_claim(&self, acc: Option<&RefCell<dyn OpeningAccumulator<F>>>) -> F {
        let acc = acc.unwrap().borrow();
        let (_, hamming_booleanity_claim) = acc.get_virtual_polynomial_opening(
            VirtualPolynomial::RamHammingWeight,
            SumcheckId::RamHammingBooleanity,
        );
        hamming_booleanity_claim * self.gamma_powers.iter().sum::<F>()
    }
}
