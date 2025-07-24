use std::{cell::RefCell, rc::Rc};

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use common::jolt_device::MemoryLayout;
use rayon::prelude::*;
use tracer::instruction::RV32IMCycle;

use crate::{
    dag::state_manager::StateManager,
    field::JoltField,
    jolt::{
        vm::ram::{compute_d_parameter, remap_address, NUM_RA_I_VARS},
        witness::CommittedPolynomial,
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
    utils::{math::Math, thread::unsafe_allocate_zero_vec, transcript::Transcript},
};

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone)]
pub struct BooleanityProof<F, ProofTranscript>
where
    F: JoltField,
    ProofTranscript: Transcript,
{
    sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    ra_claims: Vec<F>,
}

struct BooleanityProverState<F: JoltField> {
    /// B polynomial (EqPolynomial)
    B: MultilinearPolynomial<F>,
    /// F array for phase 1
    F: Vec<F>,
    /// G arrays (precomputed) - one for each decomposed part
    G: Vec<Vec<F>>,
    /// D polynomial for phase 2
    D: MultilinearPolynomial<F>,
    /// H polynomials for phase 2 - one for each decomposed part
    H: Option<Vec<MultilinearPolynomial<F>>>,
    /// eq(r, r) value computed at end of phase 1
    eq_r_r: F,
    /// z powers
    z_powers: Vec<F>,
    /// D parameter as in Twist and Shout paper
    d: usize,
}

struct BooleanityVerifierState<F: JoltField> {
    /// z powers
    z_powers: Vec<F>,
}

pub struct BooleanitySumcheck<F: JoltField> {
    /// Number of trace steps
    T: usize,
    /// D parameter as in Twist and Shout paper
    d: usize,
    /// r_address challenge
    r_address: Vec<F>,
    /// r_cycle challenge
    r_cycle: Vec<F>,
    /// Prover state (if prover)
    prover_state: Option<BooleanityProverState<F>>,
    /// Verifier state (if verifier)
    verifier_state: Option<BooleanityVerifierState<F>>,
    /// Current round
    current_round: usize,
    /// Store trace and memory layout for phase transition
    trace: Option<Vec<RV32IMCycle>>,
    memory_layout: Option<MemoryLayout>,
}

impl<F: JoltField> BooleanitySumcheck<F> {
    #[tracing::instrument(skip_all, name = "RamBooleanitySumcheck::new_prover")]
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

        let r_cycle: Vec<F> = state_manager
            .transcript
            .borrow_mut()
            .challenge_vector(T.log_2());

        let r_address: Vec<F> = state_manager
            .transcript
            .borrow_mut()
            .challenge_vector(NUM_RA_I_VARS);

        let eq_r_cycle = EqPolynomial::evals(&r_cycle);

        // Get z challenges for batching
        let z_challenges: Vec<F> = state_manager.transcript.borrow_mut().challenge_vector(d);
        let mut z_powers = vec![F::one(); d];
        for i in 1..d {
            z_powers[i] = z_powers[i - 1] * z_challenges[0];
        }

        let span = tracing::span!(tracing::Level::INFO, "compute G arrays");
        let _guard = span.enter();

        // TODO(moodlezoup): `G_arrays` is identical to `F_arrays` in `hamming_weight.rs`
        let mut G_arrays = Vec::with_capacity(d);
        for i in 0..d {
            let G: Vec<F> = trace
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
            G_arrays.push(G);
        }

        drop(_guard);
        drop(span);

        let B = MultilinearPolynomial::from(EqPolynomial::evals(&r_address));
        let D = MultilinearPolynomial::from(eq_r_cycle.to_vec());

        let mut F: Vec<F> = unsafe_allocate_zero_vec(K);
        F[0] = F::one();

        let prover_state = BooleanityProverState {
            B,
            F,
            G: G_arrays,
            D,
            H: None,
            eq_r_r: F::zero(),
            z_powers,
            d,
        };

        // Create the sumcheck instance
        BooleanitySumcheck {
            T,
            d,
            r_address,
            r_cycle,
            prover_state: Some(prover_state),
            verifier_state: None,
            current_round: 0,
            trace: Some(trace.to_vec()),
            memory_layout: Some(memory_layout.clone()),
        }
    }

    pub fn new_verifier<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        K: usize,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Self {
        let (_, program_io, T) = state_manager.get_verifier_data();
        let memory_layout = &program_io.memory_layout;

        // Calculate D dynamically such that 2^8 = K^(1/D)
        let d = compute_d_parameter(K);

        let r_cycle: Vec<F> = state_manager
            .transcript
            .borrow_mut()
            .challenge_vector(T.log_2());

        let r_address: Vec<F> = state_manager
            .transcript
            .borrow_mut()
            .challenge_vector(NUM_RA_I_VARS);

        // Get z challenges for batching
        let z_challenges: Vec<F> = state_manager.transcript.borrow_mut().challenge_vector(d);
        let mut z_powers = vec![F::one(); d];
        for i in 1..d {
            z_powers[i] = z_powers[i - 1] * z_challenges[0];
        }

        BooleanitySumcheck {
            T,
            d,
            r_address,
            r_cycle,
            prover_state: None,
            verifier_state: Some(BooleanityVerifierState { z_powers }),
            current_round: 0,
            trace: None,
            memory_layout: Some(memory_layout.clone()),
        }
    }
}

impl<F: JoltField> SumcheckInstance<F> for BooleanitySumcheck<F> {
    fn degree(&self) -> usize {
        3
    }

    fn num_rounds(&self) -> usize {
        NUM_RA_I_VARS + self.T.log_2()
    }

    fn input_claim(&self) -> F {
        F::zero() // Always zero for booleanity
    }

    #[tracing::instrument(skip_all, name = "RamBooleanitySumcheck::compute_prover_message")]
    fn compute_prover_message(&mut self, round: usize) -> Vec<F> {
        if round < NUM_RA_I_VARS {
            // Phase 1: First log(K^(1/d)) rounds
            self.compute_phase1_message(round)
        } else {
            // Phase 2: Last log(T) rounds
            self.compute_phase2_message(round - NUM_RA_I_VARS)
        }
    }

    #[tracing::instrument(skip_all, name = "RamBooleanitySumcheck::bind")]
    fn bind(&mut self, r_j: F, round: usize) {
        let prover_state = self
            .prover_state
            .as_mut()
            .expect("Prover state not initialized");

        if round < NUM_RA_I_VARS {
            // Phase 1: Bind B and update F
            prover_state.B.bind_parallel(r_j, BindingOrder::LowToHigh);

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
            if round == NUM_RA_I_VARS - 1 {
                prover_state.eq_r_r = prover_state.B.final_sumcheck_claim();

                // Compute H polynomials for each decomposed part
                let trace = self.trace.as_ref().expect("Trace not set");
                let memory_layout = self.memory_layout.as_ref().expect("Memory layout not set");
                let d = prover_state.d;

                let mut H_polys = Vec::with_capacity(d);

                for i in 0..d {
                    let H_vec: Vec<F> = trace
                        .par_iter()
                        .map(|cycle| {
                            if let Some(address) =
                                remap_address(cycle.ram_access().address() as u64, memory_layout)
                            {
                                // Get i-th address chunk
                                let address_i = (address >> (NUM_RA_I_VARS * (d - 1 - i)))
                                    % (1 << NUM_RA_I_VARS);
                                prover_state.F[address_i as usize]
                            } else {
                                F::zero()
                            }
                        })
                        .collect();
                    H_polys.push(MultilinearPolynomial::from(H_vec));
                }

                prover_state.H = Some(H_polys);
            }
        } else {
            // Phase 2: Bind D and all H polynomials
            let h_polys = prover_state
                .H
                .as_mut()
                .expect("H polynomials not initialized");

            // Bind D and all H polynomials in parallel
            rayon::join(
                || prover_state.D.bind_parallel(r_j, BindingOrder::LowToHigh),
                || {
                    h_polys
                        .par_iter_mut()
                        .for_each(|h_poly| h_poly.bind_parallel(r_j, BindingOrder::LowToHigh))
                },
            );
        }

        self.current_round += 1;
    }

    fn expected_output_claim(
        &self,
        accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F>>>>,
        r: &[F],
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
                        SumcheckId::RamBooleanity,
                    )
                    .1
            })
            .collect();

        let (r_address_prime, r_cycle_prime) = r.split_at(NUM_RA_I_VARS);

        let r_address_prime: Vec<_> = r_address_prime.iter().copied().rev().collect();
        let eq_eval_address = EqPolynomial::mle(&self.r_address, &r_address_prime);

        let r_cycle_prime: Vec<_> = r_cycle_prime.iter().copied().rev().collect();
        let eq_eval_cycle = EqPolynomial::mle(&self.r_cycle, &r_cycle_prime);

        // Compute batched booleanity check: sum_{i=0}^{d-1} z^i * (ra_i^2 - ra_i)
        let mut result = F::zero();
        for (i, ra_claim) in ra_claims.iter().enumerate() {
            result += verifier_state.z_powers[i] * (ra_claim.square() - *ra_claim);
        }

        eq_eval_address * eq_eval_cycle * result
    }

    fn normalize_opening_point(&self, opening_point: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        let (r_address, r_cycle) = opening_point.split_at(NUM_RA_I_VARS);
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

        let h_polys = prover_state.H.as_ref().expect("H polys not initialized");
        let claims: Vec<F> = h_polys
            .iter()
            .map(|h_poly| h_poly.final_sumcheck_claim())
            .collect();

        let (r_address, r_cycle) = opening_point.split_at(NUM_RA_I_VARS);
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
}

impl<F: JoltField> BooleanitySumcheck<F> {
    /// Compute prover message for first log k rounds
    fn compute_phase1_message(&self, round: usize) -> Vec<F> {
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");

        const DEGREE: usize = 3;
        let d = prover_state.d;
        let m = round + 1;

        let mut univariate_poly_evals = [F::zero(); DEGREE];

        (0..prover_state.B.len() / 2)
            .into_par_iter()
            .map(|k_prime| {
                let B_evals = prover_state
                    .B
                    .sumcheck_evals_array::<DEGREE>(k_prime, BindingOrder::LowToHigh);

                let mut evals = [F::zero(); DEGREE];

                for i in 0..d {
                    let G_i = &prover_state.G[i];

                    // Compute contribution from this part
                    let inner_sum = G_i[k_prime << m..(k_prime + 1) << m]
                        .par_iter()
                        .enumerate()
                        .map(|(k, &G_k)| {
                            let k_m = k >> (m - 1);
                            let F_k = prover_state.F[k % (1 << (m - 1))];
                            let G_times_F = G_k * F_k;

                            let mut local_evals = [F::zero(); DEGREE];

                            let eq_0 = if k_m == 0 { F::one() } else { F::zero() };
                            let eq_2 = if k_m == 0 {
                                F::from_i64(-1)
                            } else {
                                F::from_u8(2)
                            };
                            let eq_3 = if k_m == 0 {
                                F::from_i64(-2)
                            } else {
                                F::from_u8(3)
                            };

                            local_evals[0] = G_times_F * (eq_0 * eq_0 * F_k - eq_0);
                            local_evals[1] = G_times_F * (eq_2 * eq_2 * F_k - eq_2);
                            local_evals[2] = G_times_F * (eq_3 * eq_3 * F_k - eq_3);

                            local_evals
                        })
                        .reduce(
                            || [F::zero(); DEGREE],
                            |mut running, new| {
                                for j in 0..DEGREE {
                                    running[j] += new[j];
                                }
                                running
                            },
                        );

                    // Add contribution weighted by z^i
                    for j in 0..DEGREE {
                        evals[j] += prover_state.z_powers[i] * inner_sum[j];
                    }
                }

                // Multiply by B evaluations
                for j in 0..DEGREE {
                    evals[j] *= B_evals[j];
                }
                evals
            })
            .reduce(
                || [F::zero(); DEGREE],
                |mut running, new| {
                    for j in 0..DEGREE {
                        running[j] += new[j];
                    }
                    running
                },
            )
            .into_iter()
            .enumerate()
            .for_each(|(i, val)| univariate_poly_evals[i] = val);

        univariate_poly_evals.to_vec()
    }

    /// Compute prover message for phase 2 (last log(T) rounds)
    fn compute_phase2_message(&self, _round: usize) -> Vec<F> {
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");
        let h_polys = prover_state
            .H
            .as_ref()
            .expect("H polynomials not initialized");
        const DEGREE: usize = 3;
        let d = prover_state.d;

        let mut univariate_poly_evals = [F::zero(); DEGREE];

        (0..prover_state.D.len() / 2)
            .into_par_iter()
            .map(|i| {
                let D_evals = prover_state
                    .D
                    .sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);

                let mut evals = [F::zero(); DEGREE];

                // For each polynomial in the batch
                for j in 0..d {
                    let H_j_evals =
                        h_polys[j].sumcheck_evals_array::<DEGREE>(i, BindingOrder::LowToHigh);

                    // For each evaluation point
                    for k in 0..DEGREE {
                        // Add z^j * (H_j^2 - H_j) * D
                        evals[k] += prover_state.z_powers[j]
                            * D_evals[k]
                            * (H_j_evals[k].square() - H_j_evals[k]);
                    }
                }

                evals
            })
            .reduce(
                || [F::zero(); DEGREE],
                |mut running, new| {
                    for i in 0..DEGREE {
                        running[i] += new[i];
                    }
                    running
                },
            )
            .into_iter()
            .enumerate()
            .for_each(|(i, val)| univariate_poly_evals[i] = prover_state.eq_r_r * val);

        univariate_poly_evals.to_vec()
    }
}
