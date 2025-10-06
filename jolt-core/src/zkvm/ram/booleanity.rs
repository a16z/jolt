use allocative::Allocative;
#[cfg(feature = "allocative")]
use allocative::FlameGraphBuilder;
use jolt_field::JoltField;
use rayon::prelude::*;
use std::{cell::RefCell, rc::Rc};

use crate::{
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_proof::{
            OpeningPoint, ProverOpeningAccumulator, SumcheckId, VerifierOpeningAccumulator,
            BIG_ENDIAN,
        },
        split_eq_poly::GruenSplitEqPolynomial,
    },
    subprotocols::sumcheck::SumcheckInstance,
    transcripts::Transcript,
    utils::{
        math::Math,
        thread::{drop_in_background_thread, unsafe_allocate_zero_vec},
    },
    zkvm::{
        dag::state_manager::StateManager,
        ram::remap_address,
        witness::{compute_d_parameter, CommittedPolynomial, DTH_ROOT_OF_K},
    },
};

#[derive(Allocative)]
struct BooleanityProverState<F: JoltField> {
    /// B polynomial (GruenSplitEqPolynomial)
    B: GruenSplitEqPolynomial<F>,
    /// F array for phase 1
    F: Vec<F>,
    /// ra(k, r_cycle)
    G: Vec<Vec<F>>,
    /// eq(r_cycle, j) - using Gruen optimization
    D: GruenSplitEqPolynomial<F>,
    /// ra(r'_address, j)
    H: Vec<MultilinearPolynomial<F>>,
    /// eq(r_address, r'_address)
    eq_r_r: F,
}

#[derive(Allocative)]
pub struct BooleanitySumcheck<F: JoltField> {
    T: usize,
    d: usize,
    r_address: Vec<F>,
    r_cycle: Vec<F>,
    gamma_powers: Vec<F>,
    prover_state: Option<BooleanityProverState<F>>,
    current_round: usize,
    addresses: Vec<Option<u64>>,
}

impl<F: JoltField> BooleanitySumcheck<F> {
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

        let r_cycle: Vec<F> = state_manager
            .transcript
            .borrow_mut()
            .challenge_vector(T.log_2());

        let r_address: Vec<F> = state_manager
            .transcript
            .borrow_mut()
            .challenge_vector(DTH_ROOT_OF_K.log_2());

        let eq_r_cycle = EqPolynomial::evals(&r_cycle);

        // Get gamma challenge for batching
        let gamma: F = state_manager.transcript.borrow_mut().challenge_scalar();
        let mut gamma_powers = vec![F::one(); d];
        for i in 1..d {
            gamma_powers[i] = gamma_powers[i - 1] * gamma;
        }

        let span = tracing::span!(tracing::Level::INFO, "compute G arrays");
        let _guard = span.enter();

        // TODO(moodlezoup): `G_arrays` is identical to `F_arrays` in `hamming_weight.rs`
        let mut G_arrays = Vec::with_capacity(d);

        let addresses: Vec<Option<u64>> = trace
            .par_iter()
            .map(|cycle| remap_address(cycle.ram_access().address() as u64, memory_layout))
            .collect();

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
        };

        BooleanitySumcheck {
            T,
            d,
            r_address,
            r_cycle,
            gamma_powers,
            prover_state: Some(prover_state),
            current_round: 0,
            addresses,
        }
    }

    pub fn new_verifier<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Self {
        let (_, _, T) = state_manager.get_verifier_data();

        // Calculate D dynamically such that 2^8 = K^(1/D)
        let d = compute_d_parameter(state_manager.ram_K);

        let r_cycle: Vec<F> = state_manager
            .transcript
            .borrow_mut()
            .challenge_vector(T.log_2());

        let r_address: Vec<F> = state_manager
            .transcript
            .borrow_mut()
            .challenge_vector(DTH_ROOT_OF_K.log_2());

        // Get gamma challenge for batching
        let gamma: F = state_manager.transcript.borrow_mut().challenge_scalar();
        let mut gamma_powers = vec![F::one(); d];
        for i in 1..d {
            gamma_powers[i] = gamma_powers[i - 1] * gamma;
        }

        BooleanitySumcheck {
            T,
            d,
            r_address,
            r_cycle,
            gamma_powers,
            prover_state: None,
            current_round: 0,
            addresses: vec![],
        }
    }
}

impl<F: JoltField> SumcheckInstance<F> for BooleanitySumcheck<F> {
    fn degree(&self) -> usize {
        3
    }

    fn num_rounds(&self) -> usize {
        DTH_ROOT_OF_K.log_2() + self.T.log_2()
    }

    fn input_claim(&self) -> F {
        F::zero() // Always zero for booleanity
    }

    #[tracing::instrument(skip_all, name = "RamBooleanitySumcheck::compute_prover_message")]
    fn compute_prover_message(&mut self, round: usize, previous_claim: F) -> Vec<F> {
        if round < DTH_ROOT_OF_K.log_2() {
            // Phase 1: First log(K^(1/d)) rounds
            self.compute_phase1_message(round, previous_claim)
        } else {
            // Phase 2: Last log(T) rounds
            self.compute_phase2_message(round - DTH_ROOT_OF_K.log_2(), previous_claim)
        }
    }

    #[tracing::instrument(skip_all, name = "RamBooleanitySumcheck::bind")]
    fn bind(&mut self, r_j: F, round: usize) {
        let prover_state = self
            .prover_state
            .as_mut()
            .expect("Prover state not initialized");

        if round < DTH_ROOT_OF_K.log_2() {
            // Phase 1: Bind B and update F
            prover_state.B.bind(r_j);

            // Update F for this round (see Equation 55)
            let (F_left, F_right) = prover_state.F.split_at_mut(1 << round);
            F_left
                .par_iter_mut()
                .zip(F_right.par_iter_mut())
                .for_each(|(x, y)| {
                    *y = *x * r_j;
                    *x -= *y;
                });

            // If transitioning to phase 2, prepare H polynomials
            if round == DTH_ROOT_OF_K.log_2() - 1 {
                prover_state.eq_r_r = prover_state.B.current_scalar;

                // Compute H polynomials for each decomposed part
                let addresses = &self.addresses;

                let mut H_polys = Vec::with_capacity(self.d);

                for i in 0..self.d {
                    let H_vec: Vec<F> = addresses
                        .par_iter()
                        .map(|address_opt| {
                            address_opt.map_or(F::zero(), |address| {
                                // Get i-th address chunk
                                let address_i = (address
                                    >> (DTH_ROOT_OF_K.log_2() * (self.d - 1 - i)))
                                    % DTH_ROOT_OF_K as u64;
                                prover_state.F[address_i as usize]
                            })
                        })
                        .collect();
                    H_polys.push(MultilinearPolynomial::from(H_vec));
                }

                prover_state.H = H_polys;

                // Drop G arrays and F array as they're no longer needed in phase 2
                let g = std::mem::take(&mut prover_state.G);
                drop_in_background_thread(g);

                let f = std::mem::take(&mut prover_state.F);
                drop_in_background_thread(f);

                // Drop addresses as it's no longer needed in phase 2
                let addresses = std::mem::take(&mut self.addresses);
                drop_in_background_thread(addresses);
            }
        } else {
            // Phase 2: Bind D and all H polynomials

            // Bind D and all H polynomials
            prover_state.D.bind(r_j);
            prover_state
                .H
                .par_iter_mut()
                .for_each(|h_poly| h_poly.bind_parallel(r_j, BindingOrder::LowToHigh));
        }

        self.current_round += 1;
    }

    fn expected_output_claim(
        &self,
        accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
        r: &[F],
    ) -> F {
        let ra_claims: Vec<_> = (0..self.d)
            .map(|i| {
                accumulator
                    .as_ref()
                    .unwrap()
                    .borrow()
                    .get_committed_polynomial_opening(
                        CommittedPolynomial::RamRa(i),
                        SumcheckId::RamBooleanity,
                    )
                    .1
            })
            .collect();

        let (r_address_prime, r_cycle_prime) = r.split_at(DTH_ROOT_OF_K.log_2());

        let r_address_prime: Vec<_> = r_address_prime.iter().copied().rev().collect();
        let eq_eval_address = EqPolynomial::mle(&self.r_address, &r_address_prime);

        let r_cycle_prime: Vec<_> = r_cycle_prime.iter().copied().rev().collect();
        let eq_eval_cycle = EqPolynomial::mle(&self.r_cycle, &r_cycle_prime);

        // Compute batched booleanity check: sum_{i=0}^{d-1} gamma^i * (ra_i^2 - ra_i)
        let mut result = F::zero();
        for (i, ra_claim) in ra_claims.iter().enumerate() {
            result += self.gamma_powers[i] * (ra_claim.square() - *ra_claim);
        }

        eq_eval_address * eq_eval_cycle * result
    }

    fn normalize_opening_point(&self, opening_point: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        let (r_address, r_cycle) = opening_point.split_at(DTH_ROOT_OF_K.log_2());
        let mut r_big_endian: Vec<F> = r_address.iter().rev().copied().collect();
        r_big_endian.extend(r_cycle.iter().copied().rev());
        OpeningPoint::new(r_big_endian)
    }

    fn cache_openings_prover(
        &self,
        accumulator: Rc<RefCell<ProverOpeningAccumulator<F>>>,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");

        let claims: Vec<F> = prover_state
            .H
            .iter()
            .map(|h_poly| h_poly.final_sumcheck_claim())
            .collect();

        let (r_address, r_cycle) = opening_point.split_at(DTH_ROOT_OF_K.log_2());
        accumulator.borrow_mut().append_sparse(
            (0..self.d).map(CommittedPolynomial::RamRa).collect(),
            SumcheckId::RamBooleanity,
            r_address.r,
            r_cycle.r,
            claims,
        );
    }

    fn cache_openings_verifier(
        &self,
        accumulator: Rc<RefCell<VerifierOpeningAccumulator<F>>>,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        accumulator.borrow_mut().append_sparse(
            (0..self.d).map(CommittedPolynomial::RamRa).collect(),
            SumcheckId::RamBooleanity,
            opening_point.r,
        );
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, flamegraph: &mut FlameGraphBuilder) {
        flamegraph.visit_root(self);
    }
}

impl<F: JoltField> BooleanitySumcheck<F> {
    /// Compute prover message for first log k rounds
    fn compute_phase1_message(&self, round: usize, previous_claim: F) -> Vec<F> {
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");

        const DEGREE: usize = 3;
        let m = round + 1;
        let B = &prover_state.B;

        // Compute quadratic coefficients first
        let quadratic_coeffs: [F; DEGREE - 1] = if B.E_in_current_len() == 1 {
            // E_in is fully bound
            (0..B.len() / 2)
                .into_par_iter()
                .map(|k_prime| {
                    let B_eval = B.E_out_current()[k_prime];
                    let mut coeffs = [F::zero(); DEGREE - 1];

                    for i in 0..self.d {
                        let G_i = &prover_state.G[i];
                        let inner_sum = G_i[k_prime << m..(k_prime + 1) << m]
                            .par_iter()
                            .enumerate()
                            .map(|(k, &G_k)| {
                                let k_m = k >> (m - 1);
                                let F_k = prover_state.F[k % (1 << (m - 1))];
                                let G_times_F = G_k * F_k;

                                // For c \in {0, infty} compute:
                                // G[k] * (F[k_1, ...., k_{m-1}, c]^2 - F[k_1, ...., k_{m-1}, c])
                                // = G_times_F * (eq(k_m, c)^2 * F[k_1, ...., k_{m-1}] - eq(k_m, c))
                                let eval_infty = G_times_F * F_k;
                                let eval_0 = if k_m == 0 {
                                    eval_infty - G_times_F
                                } else {
                                    F::zero()
                                };

                                [eval_0, eval_infty]
                            })
                            .reduce(
                                || [F::zero(); DEGREE - 1],
                                |running, new| [running[0] + new[0], running[1] + new[1]],
                            );

                        coeffs[0] += self.gamma_powers[i] * inner_sum[0];
                        coeffs[1] += self.gamma_powers[i] * inner_sum[1];
                    }

                    [B_eval * coeffs[0], B_eval * coeffs[1]]
                })
                .reduce(
                    || [F::zero(); DEGREE - 1],
                    |running, new| [running[0] + new[0], running[1] + new[1]],
                )
        } else {
            // E_in has not been fully bound
            let num_x_in_bits = B.E_in_current_len().log_2();
            let x_bitmask = (1 << num_x_in_bits) - 1;
            let chunk_size = 1 << num_x_in_bits;

            (0..B.len() / 2)
                .collect::<Vec<_>>()
                .par_chunks(chunk_size)
                .enumerate()
                .map(|(x_out, chunk)| {
                    let B_E_out_eval = B.E_out_current()[x_out];

                    let chunk_evals = chunk
                        .par_iter()
                        .map(|k_prime| {
                            let x_in = k_prime & x_bitmask;
                            let B_E_in_eval = B.E_in_current()[x_in];
                            let mut coeffs = [F::zero(); DEGREE - 1];

                            for i in 0..self.d {
                                let G_i = &prover_state.G[i];
                                let inner_sum = G_i[k_prime << m..(k_prime + 1) << m]
                                    .par_iter()
                                    .enumerate()
                                    .map(|(k, &G_k)| {
                                        let k_m = k >> (m - 1);
                                        let F_k = prover_state.F[k % (1 << (m - 1))];
                                        let G_times_F = G_k * F_k;

                                        let eval_infty = G_times_F * F_k;
                                        let eval_0 = if k_m == 0 {
                                            eval_infty - G_times_F
                                        } else {
                                            F::zero()
                                        };
                                        [eval_0, eval_infty]
                                    })
                                    .reduce(
                                        || [F::zero(); DEGREE - 1],
                                        |running, new| [running[0] + new[0], running[1] + new[1]],
                                    );

                                coeffs[0] += self.gamma_powers[i] * inner_sum[0];
                                coeffs[1] += self.gamma_powers[i] * inner_sum[1];
                            }

                            [B_E_in_eval * coeffs[0], B_E_in_eval * coeffs[1]]
                        })
                        .reduce(
                            || [F::zero(); DEGREE - 1],
                            |running, new| [running[0] + new[0], running[1] + new[1]],
                        );

                    [B_E_out_eval * chunk_evals[0], B_E_out_eval * chunk_evals[1]]
                })
                .reduce(
                    || [F::zero(); DEGREE - 1],
                    |running, new| [running[0] + new[0], running[1] + new[1]],
                )
        };

        // Use Gruen optimization to get cubic evaluations from quadratic coefficients
        B.gruen_evals_deg_3(quadratic_coeffs[0], quadratic_coeffs[1], previous_claim)
            .to_vec()
    }

    /// Compute prover message for phase 2 (last log(T) rounds)
    fn compute_phase2_message(&self, _round: usize, previous_claim: F) -> Vec<F> {
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");
        const DEGREE: usize = 3;

        let D = &prover_state.D;

        // Compute quadratic coefficients
        let quadratic_coeffs: [F; DEGREE - 1] = if D.E_in_current_len() == 1 {
            // E_in is fully bound
            (0..D.len() / 2)
                .into_par_iter()
                .map(|j_prime| {
                    let D_eval = D.E_out_current()[j_prime];
                    let mut coeffs = [F::zero(); DEGREE - 1];

                    for i in 0..self.d {
                        let h_poly = &prover_state.H[i];

                        let h_0 = h_poly.get_bound_coeff(2 * j_prime); // h(0)
                        let h_1 = h_poly.get_bound_coeff(2 * j_prime + 1); // h(1)

                        // For c = 0: h(0)^2 - h(0)
                        coeffs[0] += self.gamma_powers[i] * (h_0.square() - h_0);

                        // For quadratic coefficient: b^2 where b = h(1) - h(0) is the linear coefficient
                        let b = h_1 - h_0; // Linear coefficient of h
                        coeffs[1] += self.gamma_powers[i] * b.square(); // Quadratic coefficient of h^2 - h
                    }

                    [D_eval * coeffs[0], D_eval * coeffs[1]]
                })
                .reduce(
                    || [F::zero(); DEGREE - 1],
                    |running, new| [running[0] + new[0], running[1] + new[1]],
                )
        } else {
            // E_in has not been fully bound - use nested structure
            let num_x_in_bits = D.E_in_current_len().log_2();
            let x_bitmask = (1 << num_x_in_bits) - 1;
            let chunk_size = 1 << num_x_in_bits;

            (0..D.len() / 2)
                .collect::<Vec<_>>()
                .par_chunks(chunk_size)
                .enumerate()
                .map(|(x_out, chunk)| {
                    let D_E_out_eval = D.E_out_current()[x_out];

                    let chunk_evals = chunk
                        .par_iter()
                        .map(|j_prime| {
                            let x_in = j_prime & x_bitmask;
                            let D_E_in_eval = D.E_in_current()[x_in];
                            let mut coeffs = [F::zero(); DEGREE - 1];

                            for i in 0..self.d {
                                let h_poly = &prover_state.H[i];

                                let h_0 = h_poly.get_bound_coeff(2 * j_prime); // h(0)
                                let h_1 = h_poly.get_bound_coeff(2 * j_prime + 1); // h(1)

                                // For c = 0: h(0)^2 - h(0)
                                coeffs[0] += self.gamma_powers[i] * (h_0.square() - h_0);

                                // For quadratic coefficient: b^2 where b = h(1) - h(0) is the linear coefficient
                                let b = h_1 - h_0; // Linear coefficient of h
                                coeffs[1] += self.gamma_powers[i] * b.square(); // Quadratic coefficient of h^2 - h
                            }

                            // Inner D contribution
                            [D_E_in_eval * coeffs[0], D_E_in_eval * coeffs[1]]
                        })
                        .reduce(
                            || [F::zero(); DEGREE - 1],
                            |running, new| [running[0] + new[0], running[1] + new[1]],
                        );

                    // Outer D contribution
                    [D_E_out_eval * chunk_evals[0], D_E_out_eval * chunk_evals[1]]
                })
                .reduce(
                    || [F::zero(); DEGREE - 1],
                    |running, new| [running[0] + new[0], running[1] + new[1]],
                )
        };

        // Adjust the previous claim by dividing out eq_r_r
        let adjusted_claim = previous_claim / prover_state.eq_r_r;

        let gruen_evals =
            D.gruen_evals_deg_3(quadratic_coeffs[0], quadratic_coeffs[1], adjusted_claim);

        vec![
            prover_state.eq_r_r * gruen_evals[0],
            prover_state.eq_r_r * gruen_evals[1],
            prover_state.eq_r_r * gruen_evals[2],
        ]
    }
}
