use std::{cell::RefCell, rc::Rc};

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use common::jolt_device::MemoryLayout;
use rayon::prelude::*;
use tracer::instruction::RV32IMCycle;

use crate::{
    dag::state_manager::StateManager,
    field::JoltField,
    jolt::vm::ram::remap_address,
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        eq_poly::EqPolynomial,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
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
    /// Chunk sizes for variable-sized d-way decomposition
    chunk_sizes: Vec<usize>,
}

struct BooleanityVerifierState<F: JoltField> {
    /// z powers
    z_powers: Vec<F>,
}

pub struct BooleanitySumcheck<F: JoltField> {
    /// Size of address space
    K: usize,
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
    /// Cached ra claims
    ra_claims: Option<Vec<F>>,
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
        let log_K = K.log_2();
        let d = (log_K / 8).max(1);

        let (_, trace, program_io, _) = state_manager.get_prover_data();
        let memory_layout = &program_io.memory_layout;

        let T = trace.len();
        let num_chunks = rayon::current_num_threads().next_power_of_two().min(T);
        let _chunk_size = (T / num_chunks).max(1);

        let r_cycle: Vec<F> = state_manager
            .transcript
            .borrow_mut()
            .challenge_vector(T.log_2());

        let r_address: Vec<F> = state_manager
            .transcript
            .borrow_mut()
            .challenge_vector(K.log_2());

        let eq_r_cycle = EqPolynomial::evals(&r_cycle);

        // Get z challenges for batching
        let z_challenges: Vec<F> = state_manager.transcript.borrow_mut().challenge_vector(d);
        let mut z_powers = vec![F::one(); d];
        for i in 1..d {
            z_powers[i] = z_powers[i - 1] * z_challenges[0];
        }

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

        let span = tracing::span!(tracing::Level::INFO, "compute G arrays");
        let _guard = span.enter();

        // Compute G arrays for each decomposed part
        let mut G_arrays: Vec<Vec<F>> = vec![unsafe_allocate_zero_vec(K); d];

        for (cycle_idx, cycle) in trace.iter().enumerate() {
            let address =
                remap_address(cycle.ram_access().address() as u64, memory_layout) as usize;

            // Decompose the address according to chunk sizes
            let mut remaining_address = address;
            for i in 0..d {
                let chunk_modulo = 1 << chunk_sizes[d - 1 - i];
                let chunk_value = remaining_address % chunk_modulo;
                remaining_address /= chunk_modulo;

                // Add to the corresponding G array
                G_arrays[d - 1 - i][chunk_value] += eq_r_cycle[cycle_idx];
            }
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
            chunk_sizes,
        };

        // Create the sumcheck instance
        BooleanitySumcheck {
            K,
            T,
            d,
            r_address,
            r_cycle,
            prover_state: Some(prover_state),
            verifier_state: None,
            ra_claims: None,
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
        let log_K = K.log_2();
        let d = (log_K / 8).max(1);

        let r_cycle: Vec<F> = state_manager
            .transcript
            .borrow_mut()
            .challenge_vector(T.log_2());

        let r_address: Vec<F> = state_manager
            .transcript
            .borrow_mut()
            .challenge_vector(K.log_2());

        // Get z challenges for batching
        let z_challenges: Vec<F> = state_manager.transcript.borrow_mut().challenge_vector(d);
        let mut z_powers = vec![F::one(); d];
        for i in 1..d {
            z_powers[i] = z_powers[i - 1] * z_challenges[0];
        }

        let ra_claims = (0..d)
            .map(|i| state_manager.get_opening(OpeningsKeys::RamBooleanityRa(i)))
            .collect();

        BooleanitySumcheck {
            K,
            T,
            d,
            r_address,
            r_cycle,
            prover_state: None,
            verifier_state: Some(BooleanityVerifierState { z_powers }),
            ra_claims: Some(ra_claims),
            current_round: 0,
            trace: None,
            memory_layout: Some(memory_layout.clone()),
        }
    }
}

impl<F: JoltField> BatchableSumcheckInstance<F> for BooleanitySumcheck<F> {
    fn degree(&self) -> usize {
        3
    }

    fn num_rounds(&self) -> usize {
        self.K.log_2() + self.T.log_2()
    }

    fn input_claim(&self) -> F {
        F::zero() // Always zero for booleanity
    }

    #[tracing::instrument(skip_all, name = "RamBooleanitySumcheck::compute_prover_message")]
    fn compute_prover_message(&mut self, round: usize) -> Vec<F> {
        let log_K = self.K.log_2();

        if round < log_K {
            // Phase 1: First log(K) rounds
            self.compute_phase1_message(round)
        } else {
            // Phase 2: Last log(T) rounds
            self.compute_phase2_message(round - log_K)
        }
    }

    #[tracing::instrument(skip_all, name = "RamBooleanitySumcheck::bind")]
    fn bind(&mut self, r_j: F, round: usize) {
        let prover_state = self
            .prover_state
            .as_mut()
            .expect("Prover state not initialized");
        let log_K = self.K.log_2();

        if round < log_K {
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
            if round == log_K - 1 {
                prover_state.eq_r_r = prover_state.B.final_sumcheck_claim();

                // Compute H polynomials for each decomposed part
                let trace = self.trace.as_ref().expect("Trace not set");
                let memory_layout = self.memory_layout.as_ref().expect("Memory layout not set");
                let d = prover_state.d;
                let chunk_sizes = &prover_state.chunk_sizes;

                let mut H_polys = Vec::with_capacity(d);

                for i in 0..d {
                    let H_vec: Vec<F> = trace
                        .par_iter()
                        .map(|cycle| {
                            let address =
                                remap_address(cycle.ram_access().address() as u64, memory_layout)
                                    as usize;

                            // Decompose address to get the i-th chunk
                            let (left, right) = chunk_sizes.split_at(d - i);
                            let shift: usize = right.iter().sum();
                            let chunk_size = left.last().unwrap();
                            let address_chunk = (address >> shift) % (1 << chunk_size);
                            prover_state.F[address_chunk]
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

    fn expected_output_claim(&self, r: &[F]) -> F {
        let verifier_state = self
            .verifier_state
            .as_ref()
            .expect("Verifier state not initialized");
        let ra_claims = self.ra_claims.as_ref().expect("RA claims not cached");

        let log_K = self.K.log_2();
        let (r_address_prime, r_cycle_prime) = r.split_at(log_K);

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
}

impl<F, PCS> CacheSumcheckOpenings<F, PCS> for BooleanitySumcheck<F>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    fn normalize_opening_point(&self, opening_point: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        let (r_address, r_cycle) = opening_point.split_at(self.K.log_2());
        let mut r_big_endian: Vec<F> = r_address.iter().rev().copied().collect();
        r_big_endian.extend(r_cycle.iter().copied().rev());
        OpeningPoint::new(r_big_endian)
    }

    fn cache_openings_prover(
        &mut self,
        accumulator: Option<Rc<RefCell<ProverOpeningAccumulator<F, PCS>>>>,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        debug_assert!(self.ra_claims.is_none());
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");

        let h_polys = prover_state.H.as_ref().expect("H polys not initialized");
        let claims: Vec<F> = h_polys
            .iter()
            .map(|h_poly| h_poly.final_sumcheck_claim())
            .collect();

        let accumulator = accumulator.expect("accumulator is needed");
        claims.iter().enumerate().for_each(|(i, ra_i)| {
            accumulator.borrow_mut().append_virtual(
                OpeningsKeys::RamBooleanityRa(i),
                opening_point.clone(),
                *ra_i,
            );
        });

        self.ra_claims = Some(claims);
    }

    fn cache_openings_verifier(
        &mut self,
        accumulator: Option<Rc<RefCell<VerifierOpeningAccumulator<F, PCS>>>>,
        opening_point: OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let accumulator = accumulator.expect("accumulator is needed");
        (0..self.d).for_each(|i| {
            accumulator
                .borrow_mut()
                .populate_claim_opening(OpeningsKeys::RamBooleanityRa(i), opening_point.clone());
        });
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
                let B_evals =
                    prover_state
                        .B
                        .sumcheck_evals(k_prime, DEGREE, BindingOrder::LowToHigh);

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
                    .sumcheck_evals(i, DEGREE, BindingOrder::LowToHigh);

                let mut evals = [F::zero(); DEGREE];

                // For each polynomial in the batch
                for j in 0..d {
                    let H_j_evals = h_polys[j].sumcheck_evals(i, DEGREE, BindingOrder::LowToHigh);

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
