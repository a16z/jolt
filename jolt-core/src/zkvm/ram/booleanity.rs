use allocative::Allocative;
use rayon::prelude::*;

use crate::{
    field::JoltField,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, PolynomialBinding},
        opening_proof::{OpeningAccumulator, SumcheckId},
        ra_poly::RaPolynomial,
        split_eq_poly::GruenSplitEqPolynomial,
    },
    subprotocols::booleanity::{BooleanityConfig, BooleanityProverState, BooleanitySumcheck},
    transcripts::Transcript,
    utils::{math::Math, thread::unsafe_allocate_zero_vec},
    zkvm::{
        dag::state_manager::StateManager,
        ram::remap_address,
        witness::{compute_d_parameter, CommittedPolynomial, DTH_ROOT_OF_K},
    },
};

#[derive(Allocative)]
pub struct RamBooleanitySumcheck<F: JoltField> {
    T: usize,
    d: usize,
    r_address: Vec<F::Challenge>,
    r_cycle: Vec<F::Challenge>,
    gamma: Vec<F::Challenge>,
    prover_state: Option<BooleanityProverState<F>>,
    addresses: Vec<Option<u64>>,
}

impl<F: JoltField> RamBooleanitySumcheck<F> {
    #[tracing::instrument(skip_all, name = "RamBooleanitySumcheck::new_prover")]
    pub fn new_prover<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Self {
        let K = state_manager.ram_K;
        // Calculate D dynamically such that 2^8 = K^(1/D)
        let d = compute_d_parameter(K);

        let (_, trace, program_io, _) = state_manager.get_prover_data();
        let memory_layout = &program_io.memory_layout;

        let T = trace.len();
        let num_chunks = rayon::current_num_threads().next_power_of_two().min(T);
        let chunk_size = (T / num_chunks).max(1);

        let r_cycle: Vec<F::Challenge> = state_manager
            .transcript
            .borrow_mut()
            .challenge_vector_optimized::<F>(T.log_2());

        let r_address: Vec<F::Challenge> = state_manager
            .transcript
            .borrow_mut()
            .challenge_vector_optimized::<F>(DTH_ROOT_OF_K.log_2());

        let eq_r_cycle = EqPolynomial::<F>::evals(&r_cycle);

        // Get gamma challenges for batching (optimized)
        let gamma = state_manager
            .transcript
            .borrow_mut()
            .challenge_vector_optimized::<F>(d);

        // Compute G arrays
        let span = tracing::span!(tracing::Level::INFO, "compute G arrays");
        let _guard = span.enter();

        let addresses: Vec<Option<u64>> = trace
            .par_iter()
            .map(|cycle| remap_address(cycle.ram_access().address() as u64, memory_layout))
            .collect();

        let mut G_arrays = Vec::with_capacity(d);
        for i in 0..d {
            let G: Vec<F> = addresses
                .par_chunks(chunk_size)
                .enumerate()
                .map(|(chunk_index, address_chunk)| {
                    let mut local_array = unsafe_allocate_zero_vec(DTH_ROOT_OF_K);
                    let mut j = chunk_index * chunk_size;
                    for address_opt in address_chunk {
                        if let Some(address) = address_opt {
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
            G_arrays.push(G);
        }

        drop(_guard);
        drop(span);

        // Build H_indices
        let H_indices: Vec<Vec<Option<u8>>> = (0..d)
            .map(|i| {
                addresses
                    .par_iter()
                    .map(|address_opt| {
                        address_opt.map(|address| {
                            let address_i = (address >> (DTH_ROOT_OF_K.log_2() * (d - 1 - i)))
                                % DTH_ROOT_OF_K as u64;
                            address_i as u8
                        })
                    })
                    .collect()
            })
            .collect();

        // Create prover state
        let B = GruenSplitEqPolynomial::new(&r_address, BindingOrder::LowToHigh);
        let D = GruenSplitEqPolynomial::new(&r_cycle, BindingOrder::LowToHigh);

        let mut F: Vec<F> = unsafe_allocate_zero_vec(K);
        F[0] = F::one();

        let prover_state = BooleanityProverState {
            B,
            F,
            G: G_arrays,
            D,
            H: vec![],
            eq_r_r: F::zero(),
            H_indices,
        };

        Self {
            T,
            d,
            r_address,
            r_cycle,
            gamma,
            prover_state: Some(prover_state),
            addresses,
        }
    }

    pub fn new_verifier<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Self {
        let (_, _, T) = state_manager.get_verifier_data();

        // Calculate D dynamically such that 2^8 = K^(1/D)
        let d = compute_d_parameter(state_manager.ram_K);

        let r_cycle: Vec<F::Challenge> = state_manager
            .transcript
            .borrow_mut()
            .challenge_vector_optimized::<F>(T.log_2());

        let r_address: Vec<F::Challenge> = state_manager
            .transcript
            .borrow_mut()
            .challenge_vector_optimized::<F>(DTH_ROOT_OF_K.log_2());

        // Get gamma challenges for batching (optimized)
        let gamma = state_manager
            .transcript
            .borrow_mut()
            .challenge_vector_optimized::<F>(d);

        Self {
            T,
            d,
            r_address,
            r_cycle,
            gamma,
            prover_state: None,
            addresses: vec![],
        }
    }
}

// Implement BooleanityConfig trait
impl<F: JoltField> BooleanityConfig for RamBooleanitySumcheck<F> {
    fn d(&self) -> usize {
        self.d
    }

    fn log_k_chunk(&self) -> usize {
        DTH_ROOT_OF_K.log_2()
    }

    fn log_t(&self) -> usize {
        self.T.log_2()
    }

    fn polynomial_type(i: usize) -> CommittedPolynomial {
        CommittedPolynomial::RamRa(i)
    }

    fn sumcheck_id() -> SumcheckId {
        SumcheckId::RamBooleanity
    }
}

// Implement BooleanitySumcheck trait
impl<F: JoltField, T: Transcript> BooleanitySumcheck<F, T> for RamBooleanitySumcheck<F> {
    fn gamma(&self) -> &[F::Challenge] {
        &self.gamma
    }

    fn r_address(&self) -> &[F::Challenge] {
        &self.r_address
    }

    fn prover_state(&self) -> Option<&BooleanityProverState<F>> {
        self.prover_state.as_ref()
    }

    fn prover_state_mut(&mut self) -> Option<&mut BooleanityProverState<F>> {
        self.prover_state.as_mut()
    }

    fn get_r_cycle(&self, _accumulator: &dyn OpeningAccumulator<F>) -> Vec<F::Challenge> {
        self.r_cycle.clone()
    }

    // Override bind to handle addresses cleanup
    fn booleanity_bind(&mut self, r_j: F::Challenge, round: usize) {
        let log_k_chunk = <Self as BooleanityConfig>::log_k_chunk(self);
        let ps = self
            .prover_state
            .as_mut()
            .expect("Prover state not initialized");

        if round < log_k_chunk {
            // Phase 1: Bind B and update F
            ps.B.bind(r_j);

            // Update F for this round (see Equation 55)
            let (F_left, F_right) = ps.F.split_at_mut(1 << round);
            F_left
                .par_iter_mut()
                .zip(F_right.par_iter_mut())
                .for_each(|(x, y)| {
                    *y = *x * r_j;
                    *x -= *y;
                });

            // If transitioning to phase 2, prepare H polynomials
            if round == log_k_chunk - 1 {
                ps.eq_r_r = ps.B.get_current_scalar();

                // Initialize H polynomials using RaPolynomial
                let F = std::mem::take(&mut ps.F);
                let H_indices = std::mem::take(&mut ps.H_indices);
                ps.H = H_indices
                    .into_iter()
                    .map(|indices| RaPolynomial::new(std::sync::Arc::new(indices), F.clone()))
                    .collect();

                // Drop G arrays as they're no longer needed
                let g = std::mem::take(&mut ps.G);
                crate::utils::thread::drop_in_background_thread(g);

                // Drop addresses as it's no longer needed in phase 2
                let addresses = std::mem::take(&mut self.addresses);
                crate::utils::thread::drop_in_background_thread(addresses);
            }
        } else {
            // Phase 2: Bind D and H
            ps.D.bind(r_j);
            ps.H.par_iter_mut()
                .for_each(|poly| poly.bind_parallel(r_j, BindingOrder::LowToHigh));
        }
    }
}
