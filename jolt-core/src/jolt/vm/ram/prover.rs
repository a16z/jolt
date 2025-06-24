#![allow(clippy::too_many_arguments)]

use crate::jolt::vm::ram::{
    remap_address, BooleanityProof, BooleanitySumcheck, HammingWeightProof,
    HammingWeightProverState, HammingWeightSumcheck, HammingWeightVerifierState, RAMPreprocessing,
    RAMTwistProof, RafEvaluationProof, RafEvaluationSumcheck, ReadWriteCheckingProof,
    ValEvaluationProof, ValEvaluationSumcheck,
};
use crate::{
    field::JoltField,
    jolt::{
        vm::{
            output_check::{OutputProof, OutputSumcheck},
            ram_read_write_checking::{RamReadWriteChecking, RamReadWriteCheckingProof},
            JoltCommitments, JoltProverPreprocessing,
        },
        witness::CommittedPolynomials,
    },
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        eq_poly::EqPolynomial,
        identity_poly::UnmapRamAddressPolynomial,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        opening_proof::{ProverOpeningAccumulator, VerifierOpeningAccumulator},
    },
    subprotocols::{
        ra_virtual::{RAProof, RASumcheck},
        sumcheck::{BatchableSumcheckInstance, SumcheckInstanceProof},
    },
    utils::{
        math::Math,
        thread::{drop_in_background_thread, unsafe_allocate_zero_vec},
        transcript::Transcript,
    },
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use common::{
    constants::{BYTES_PER_INSTRUCTION, RAM_START_ADDRESS},
    jolt_device::MemoryLayout,
};
use rayon::prelude::*;
use std::vec;
use tracer::{
    emulator::memory::Memory,
    instruction::{RAMAccess, RV32IMCycle},
    JoltDevice,
};

impl<F: JoltField, ProofTranscript: Transcript> BatchableSumcheckInstance<F, ProofTranscript>
    for ValEvaluationSumcheck<F>
{
    fn degree(&self) -> usize {
        3
    }

    fn num_rounds(&self) -> usize {
        if let Some(prover_state) = &self.prover_state {
            prover_state.inc.len().log_2()
        } else if let Some(verifier_state) = &self.verifier_state {
            verifier_state.num_rounds
        } else {
            panic!("Neither prover state nor verifier state is initialized");
        }
    }

    fn input_claim(&self) -> F {
        self.claimed_evaluation - self.init_eval
    }

    fn compute_prover_message(&mut self, _round: usize, _previous_claim: F) -> Vec<F> {
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");

        const DEGREE: usize = 3;
        let univariate_poly_evals: [F; 3] = (0..prover_state.inc.len() / 2)
            .into_par_iter()
            .map(|i| {
                let inc_evals = prover_state
                    .inc
                    .sumcheck_evals(i, DEGREE, BindingOrder::LowToHigh);
                let wa_evals = prover_state
                    .wa
                    .sumcheck_evals(i, DEGREE, BindingOrder::LowToHigh);
                let lt_evals = prover_state
                    .lt
                    .sumcheck_evals(i, DEGREE, BindingOrder::LowToHigh);

                [
                    inc_evals[0] * wa_evals[0] * lt_evals[0],
                    inc_evals[1] * wa_evals[1] * lt_evals[1],
                    inc_evals[2] * wa_evals[2] * lt_evals[2],
                ]
            })
            .reduce(
                || [F::zero(); 3],
                |running, new| {
                    [
                        running[0] + new[0],
                        running[1] + new[1],
                        running[2] + new[2],
                    ]
                },
            );

        univariate_poly_evals.to_vec()
    }

    fn bind(&mut self, r_j: F, _round: usize) {
        if let Some(prover_state) = &mut self.prover_state {
            [
                &mut prover_state.inc,
                &mut prover_state.wa,
                &mut prover_state.lt,
            ]
            .par_iter_mut()
            .for_each(|poly| poly.bind_parallel(r_j, BindingOrder::LowToHigh));
        }
    }

    fn cache_openings(&mut self) {
        if let Some(prover_state) = &self.prover_state {
            self.claims = Some(ValEvaluationSumcheckClaims {
                inc_claim: prover_state.inc.final_sumcheck_claim(),
                wa_claim: prover_state.wa.final_sumcheck_claim(),
            });
        }
    }

    fn expected_output_claim(&self, r: &[F]) -> F {
        let verifier_state = self
            .verifier_state
            .as_ref()
            .expect("Verifier state not initialized");
        let claims = self.claims.as_ref().expect("Claims not cached");

        // r contains r_cycle_prime in low-to-high order
        let r_cycle_prime: Vec<F> = r.iter().rev().copied().collect();

        // Compute LT(r_cycle', r_cycle)
        let mut lt_eval = F::zero();
        let mut eq_term = F::one();
        for (x, y) in r_cycle_prime.iter().zip(verifier_state.r_cycle.iter()) {
            lt_eval += (F::one() - x) * y * eq_term;
            eq_term *= F::one() - x - y + *x * y + *x * y;
        }

        // Return inc_claim * wa_claim * lt_eval
        claims.inc_claim * claims.wa_claim * lt_eval
    }
}

impl<F: JoltField, ProofTranscript: Transcript> BatchableSumcheckInstance<F, ProofTranscript>
    for BooleanitySumcheck<F>
{
    fn degree(&self) -> usize {
        3
    }

    fn num_rounds(&self) -> usize {
        self.K.log_2() + self.T.log_2()
    }

    fn input_claim(&self) -> F {
        F::zero() // Always zero for booleanity
    }

    fn compute_prover_message(&mut self, round: usize, _previous_claim: F) -> Vec<F> {
        let K_log = self.K.log_2();

        if round < K_log {
            // Phase 1: First log(K) rounds
            self.compute_phase1_message(round)
        } else {
            // Phase 2: Last log(T) rounds
            self.compute_phase2_message(round - K_log)
        }
    }

    fn bind(&mut self, r_j: F, round: usize) {
        let prover_state = self
            .prover_state
            .as_mut()
            .expect("Prover state not initialized");
        let K_log = self.K.log_2();

        if round < K_log {
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
            if round == K_log - 1 {
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

    fn cache_openings(&mut self) {
        if let Some(prover_state) = &self.prover_state {
            if let Some(h_polys) = &prover_state.H {
                let claims: Vec<F> = h_polys
                    .iter()
                    .map(|h_poly| h_poly.final_sumcheck_claim())
                    .collect();
                self.ra_claims = Some(claims);
            }
        }
    }

    fn expected_output_claim(&self, r: &[F]) -> F {
        let verifier_state = self
            .verifier_state
            .as_ref()
            .expect("Verifier state not initialized");
        let ra_claims = self.ra_claims.as_ref().expect("RA claims not cached");

        let K_log = self.K.log_2();
        let (r_address_prime, r_cycle_prime) = r.split_at(K_log);

        let eq_eval_address = EqPolynomial::mle(&verifier_state.r_address, r_address_prime);
        let r_cycle_prime: Vec<_> = r_cycle_prime.iter().copied().rev().collect();
        let eq_eval_cycle = EqPolynomial::mle(&verifier_state.r_prime, &r_cycle_prime);

        // Compute batched booleanity check: sum_{i=0}^{d-1} z^i * (ra_i^2 - ra_i)
        let mut result = F::zero();
        for (i, ra_claim) in ra_claims.iter().enumerate() {
            result += verifier_state.z_powers[i] * (ra_claim.square() - *ra_claim);
        }

        eq_eval_address * eq_eval_cycle * result
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

impl<F: JoltField> HammingWeightSumcheck<F> {
    fn new_prover(ra: Vec<MultilinearPolynomial<F>>, z_powers: Vec<F>, d: usize) -> Self {
        // Compute input claim as sum of z powers
        let input_claim = z_powers.iter().sum();

        Self {
            input_claim,
            prover_state: Some(HammingWeightProverState { ra, z_powers, d }),
            verifier_state: None,
            cached_claims: None,
            d,
        }
    }
}

impl<F: JoltField, ProofTranscript: Transcript> BatchableSumcheckInstance<F, ProofTranscript>
    for HammingWeightSumcheck<F>
{
    fn degree(&self) -> usize {
        1
    }

    fn num_rounds(&self) -> usize {
        if let Some(prover_state) = &self.prover_state {
            prover_state.ra[0].get_num_vars()
        } else if let Some(verifier_state) = &self.verifier_state {
            verifier_state.log_K
        } else {
            panic!("Neither prover state nor verifier state is initialized")
        }
    }

    fn input_claim(&self) -> F {
        self.input_claim
    }

    fn compute_prover_message(&mut self, _round: usize, _previous_claim: F) -> Vec<F> {
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

    fn bind(&mut self, r_j: F, _round: usize) {
        if let Some(prover_state) = &mut self.prover_state {
            prover_state
                .ra
                .par_iter_mut()
                .for_each(|ra_poly| ra_poly.bind_parallel(r_j, BindingOrder::LowToHigh));
        }
    }

    fn cache_openings(&mut self) {
        if let Some(prover_state) = &self.prover_state {
            let claims: Vec<F> = prover_state
                .ra
                .iter()
                .map(|ra_poly| ra_poly.final_sumcheck_claim())
                .collect();
            self.cached_claims = Some(claims);
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

impl<F: JoltField> RafEvaluationSumcheck<F> {
    /// Create a new prover instance
    fn new_prover(
        ra: MultilinearPolynomial<F>,
        unmap: UnmapRamAddressPolynomial<F>,
        raf_claim: F,
    ) -> Self {
        Self {
            input_claim: raf_claim,
            prover_state: Some(RafEvaluationProverState { ra, unmap }),
            verifier_state: None,
            cached_claim: None,
        }
    }

    /// Create a new verifier instance
    fn new_verifier(raf_claim: F, log_K: usize, start_address: u64, ra_claim: F) -> Self {
        Self {
            input_claim: raf_claim,
            prover_state: None,
            verifier_state: Some(RafEvaluationVerifierState {
                log_K,
                start_address,
            }),
            cached_claim: Some(ra_claim),
        }
    }
}

impl<F: JoltField, ProofTranscript: Transcript> BatchableSumcheckInstance<F, ProofTranscript>
    for RafEvaluationSumcheck<F>
{
    fn degree(&self) -> usize {
        2
    }

    fn num_rounds(&self) -> usize {
        if let Some(prover_state) = &self.prover_state {
            prover_state.ra.get_num_vars()
        } else if let Some(verifier_state) = &self.verifier_state {
            verifier_state.log_K
        } else {
            panic!("Neither prover state nor verifier state is initialized")
        }
    }

    fn input_claim(&self) -> F {
        self.input_claim
    }

    fn compute_prover_message(&mut self, _round: usize, _previous_claim: F) -> Vec<F> {
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");
        const DEGREE: usize = 2;

        let univariate_poly_evals: [F; 2] = (0..prover_state.ra.len() / 2)
            .into_par_iter()
            .map(|i| {
                let ra_evals = prover_state
                    .ra
                    .sumcheck_evals(i, DEGREE, BindingOrder::LowToHigh);
                let unmap_evals =
                    prover_state
                        .unmap
                        .sumcheck_evals(i, DEGREE, BindingOrder::LowToHigh);

                // Compute the product evaluations
                [ra_evals[0] * unmap_evals[0], ra_evals[1] * unmap_evals[1]]
            })
            .reduce(
                || [F::zero(); 2],
                |running, new| [running[0] + new[0], running[1] + new[1]],
            );

        univariate_poly_evals.to_vec()
    }

    fn bind(&mut self, r_j: F, _round: usize) {
        if let Some(prover_state) = &mut self.prover_state {
            rayon::join(
                || prover_state.ra.bind_parallel(r_j, BindingOrder::LowToHigh),
                || {
                    prover_state
                        .unmap
                        .bind_parallel(r_j, BindingOrder::LowToHigh)
                },
            );
        }
    }

    fn cache_openings(&mut self) {
        if let Some(prover_state) = &self.prover_state {
            self.cached_claim = Some(prover_state.ra.final_sumcheck_claim());
        }
    }

    fn expected_output_claim(&self, r: &[F]) -> F {
        let verifier_state = self
            .verifier_state
            .as_ref()
            .expect("Verifier state not initialized");

        // Compute unmap evaluation at r
        let unmap_eval =
            UnmapRamAddressPolynomial::new(verifier_state.log_K, verifier_state.start_address)
                .evaluate(r);

        // Return unmap(r) * ra(r)
        let ra_claim = self.cached_claim.expect("ra_claim not cached");
        unmap_eval * ra_claim
    }
}

impl<F: JoltField, ProofTranscript: Transcript> RafEvaluationProof<F, ProofTranscript> {
    #[tracing::instrument(skip_all, name = "RafEvaluationProof::prove")]
    pub fn prove(
        trace: &[RV32IMCycle],
        memory_layout: &MemoryLayout,
        r_cycle: Vec<F>,
        K: usize,
        transcript: &mut ProofTranscript,
    ) -> Self {
        let T = trace.len();
        debug_assert_eq!(T.log_2(), r_cycle.len());

        let eq_r_cycle: Vec<F> = EqPolynomial::evals(&r_cycle);

        let num_chunks = rayon::current_num_threads().next_power_of_two().min(T);
        let chunk_size = (T / num_chunks).max(1);

        // TODO: Propagate ra claim from Spartan
        let ra_evals: Vec<F> = trace
            .par_chunks(chunk_size)
            .enumerate()
            .map(|(chunk_index, trace_chunk)| {
                let mut result = unsafe_allocate_zero_vec(K);
                let mut j = chunk_index * chunk_size;
                for cycle in trace_chunk {
                    let k =
                        remap_address(cycle.ram_access().address() as u64, memory_layout) as usize;
                    result[k] += eq_r_cycle[j];
                    j += 1;
                }
                result
            })
            .reduce(
                || unsafe_allocate_zero_vec(K),
                |mut running, new| {
                    running
                        .par_iter_mut()
                        .zip(new.into_par_iter())
                        .for_each(|(x, y)| *x += y);
                    running
                },
            );

        let ra_poly = MultilinearPolynomial::from(ra_evals);
        let unmap_poly = UnmapRamAddressPolynomial::new(K.log_2(), memory_layout.input_start);

        // TODO: Propagate raf claim from Spartan
        let raf_evals = trace
            .par_iter()
            .map(|t| t.ram_access().address() as u64)
            .collect::<Vec<u64>>();
        let raf_poly = MultilinearPolynomial::from(raf_evals);
        let raf_claim = raf_poly.evaluate(&r_cycle);
        let mut sumcheck_instance =
            RafEvaluationSumcheck::new_prover(ra_poly, unmap_poly, raf_claim);

        let (sumcheck_proof, _r_address) = sumcheck_instance.prove_single(transcript);

        let ra_claim = sumcheck_instance
            .cached_claim
            .expect("ra_claim should be cached after proving");

        Self {
            sumcheck_proof,
            ra_claim,
            raf_claim,
        }
    }
}

impl<F: JoltField, ProofTranscript: Transcript> RAMTwistProof<F, ProofTranscript> {
    #[tracing::instrument(skip_all, name = "RAMTwistProof::prove")]
    pub fn prove<PCS: CommitmentScheme<ProofTranscript, Field = F>>(
        preprocessing: &JoltProverPreprocessing<F, PCS, ProofTranscript>,
        trace: &[RV32IMCycle],
        final_memory: Memory,
        program_io: &JoltDevice,
        K: usize,
        opening_accumulator: &mut ProverOpeningAccumulator<F, PCS, ProofTranscript>,
        transcript: &mut ProofTranscript,
    ) -> RAMTwistProof<F, ProofTranscript> {
        let ram_preprocessing = &preprocessing.shared.ram;
        let log_T = trace.len().log_2();

        let r_prime: Vec<F> = transcript.challenge_vector(log_T);
        // TODO(moodlezoup): Reuse from ReadWriteCheckingProof
        let eq_r_cycle = EqPolynomial::evals(&r_prime);

        let mut initial_memory_state = vec![0; K];
        // Copy bytecode
        let mut index = remap_address(
            ram_preprocessing.min_bytecode_address,
            &program_io.memory_layout,
        ) as usize;
        for word in ram_preprocessing.bytecode_words.iter() {
            initial_memory_state[index] = *word;
            index += 1;
        }

        let dram_start_index = remap_address(RAM_START_ADDRESS, &program_io.memory_layout) as usize;
        let mut final_memory_state = vec![0; K];
        // Note that `final_memory` only contains memory at addresses >= `RAM_START_ADDRESS`
        // so we will still need to populate `final_memory_state` with the contents of
        // `program_io`, which lives at addresses < `RAM_START_ADDRESS`
        final_memory_state[dram_start_index..]
            .par_iter_mut()
            .enumerate()
            .for_each(|(k, word)| {
                *word = final_memory.read_word(4 * k as u64);
            });

        index = remap_address(
            program_io.memory_layout.input_start,
            &program_io.memory_layout,
        ) as usize;
        // Convert input bytes into words and populate
        // `initial_memory_state` and `final_memory_state`
        for chunk in program_io.inputs.chunks(4) {
            let mut word = [0u8; 4];
            for (i, byte) in chunk.iter().enumerate() {
                word[i] = *byte;
            }
            let word = u32::from_le_bytes(word);
            initial_memory_state[index] = word;
            final_memory_state[index] = word;
            index += 1;
        }

        // Convert output bytes into words and populate
        // `final_memory_state`
        index = remap_address(
            program_io.memory_layout.output_start,
            &program_io.memory_layout,
        ) as usize;
        for chunk in program_io.outputs.chunks(4) {
            let mut word = [0u8; 4];
            for (i, byte) in chunk.iter().enumerate() {
                word[i] = *byte;
            }
            let word = u32::from_le_bytes(word);
            final_memory_state[index] = word;
            index += 1;
        }

        // Copy panic bit
        let panic_index =
            remap_address(program_io.memory_layout.panic, &program_io.memory_layout) as usize;
        final_memory_state[panic_index] = program_io.panic as u32;
        if !program_io.panic {
            // Set termination bit
            let termination_index = remap_address(
                program_io.memory_layout.termination,
                &program_io.memory_layout,
            ) as usize;
            final_memory_state[termination_index] = 1;
        }

        #[cfg(test)]
        {
            let mut expected_final_memory_state: Vec<_> = initial_memory_state
                .iter()
                .map(|word| *word as i64)
                .collect();
            let inc = CommittedPolynomials::RamInc.generate_witness(preprocessing, trace);
            for (j, cycle) in trace.iter().enumerate() {
                if let RAMAccess::Write(write) = cycle.ram_access() {
                    let k = remap_address(write.address, &program_io.memory_layout) as usize;
                    expected_final_memory_state[k] += inc.get_coeff_i64(j);
                }
            }
            let expected_final_memory_state: Vec<u32> = expected_final_memory_state
                .into_iter()
                .map(|word| word.try_into().unwrap())
                .collect();
            assert_eq!(expected_final_memory_state, final_memory_state);
        }

        let (read_write_checking_proof, r_address, r_cycle) = RamReadWriteChecking::prove(
            preprocessing,
            trace,
            &initial_memory_state,
            program_io,
            K,
            &r_prime,
            transcript,
        );

        let val_init: MultilinearPolynomial<F> =
            MultilinearPolynomial::from(initial_memory_state.clone()); // TODO(moodlezoup): avoid clone
        let init_eval = val_init.evaluate(&r_address);

        let (val_evaluation_proof, mut r_cycle_prime) = prove_val_evaluation(
            preprocessing,
            trace,
            &program_io.memory_layout,
            r_address.clone(),
            r_cycle.clone(),
            init_eval,
            read_write_checking_proof.claims.val_claim,
            transcript,
        );
        // Cycle variables are bound from low to high
        r_cycle_prime.reverse();

        let inc_poly = CommittedPolynomials::RamInc.generate_witness(preprocessing, trace);
        opening_accumulator.append_dense(
            &[&inc_poly],
            EqPolynomial::evals(&r_cycle_prime),
            r_cycle_prime,
            &[val_evaluation_proof.inc_claim],
            transcript,
        );

        // Calculate D dynamically such that 2^8 = K^(1/D)
        // let log_k = K.log_2();
        // let d = (log_k / 8).max(1);
        let d = 1; // @TODO(markosg04) keeping d = 1 for legacy prove
        let (booleanity_sumcheck, r_address_prime, r_cycle_prime, ra_claims) = prove_ra_booleanity(
            trace,
            &program_io.memory_layout,
            &eq_r_cycle,
            K,
            d,
            transcript,
        );
        let booleanity_proof = BooleanityProof {
            sumcheck_proof: booleanity_sumcheck,
            ra_claims: ra_claims.clone(),
        };

        let r_address_prime = r_address_prime.iter().copied().rev().collect::<Vec<_>>();
        let r_cycle_prime = r_cycle_prime.iter().rev().copied().collect::<Vec<_>>();

        // Prepare common data
        let addresses: Vec<usize> = trace
            .par_iter()
            .map(|cycle| {
                remap_address(
                    cycle.ram_access().address() as u64,
                    &preprocessing.shared.memory_layout,
                ) as usize
            })
            .collect();

        println!("picked D={:?} for RAM", d);

        let ra_claim = ra_claims[0]; // d = 1

        let ra_sumcheck_instance = RASumcheck::<F>::new(
            ra_claim,
            addresses,
            r_cycle_prime,
            r_address_prime.clone(),
            1 << log_T,
            d,
        );

        let (ra_proof, mut r_cycle_bound) = ra_sumcheck_instance.prove(transcript);

        let unbound_ra_poly = CommittedPolynomials::RamRa(0).generate_witness(preprocessing, trace);
        r_cycle_bound.reverse();

        opening_accumulator.append_sparse(
            vec![unbound_ra_poly],
            r_address_prime.clone(),
            r_cycle_bound,
            ra_proof.ra_i_claims.clone(),
        );

        let (hamming_weight_sumcheck, _, ra_claims) = prove_ra_hamming_weight(
            trace,
            &program_io.memory_layout,
            eq_r_cycle,
            K,
            d,
            transcript,
        );
        let hamming_weight_proof = HammingWeightProof {
            sumcheck_proof: hamming_weight_sumcheck,
            ra_claims,
        };

        let raf_evaluation_proof =
            RafEvaluationProof::prove(trace, &program_io.memory_layout, r_cycle, K, transcript);

        let output_proof = OutputSumcheck::prove(
            preprocessing,
            trace,
            initial_memory_state,
            final_memory_state,
            program_io,
            &r_address_prime,
            transcript,
        );

        // TODO: Append to opening proof accumulator

        RAMTwistProof {
            K,
            read_write_checking_proof,
            val_evaluation_proof,
            booleanity_proof,
            ra_proof,
            hamming_weight_proof,
            raf_evaluation_proof,
            output_proof,
        }
    }
}

impl<F: JoltField, ProofTranscript: Transcript> ReadWriteCheckingProof<F, ProofTranscript> {
    #[tracing::instrument(skip_all, name = "ReadWriteCheckingProof::prove")]
    pub fn prove(
        trace: &[RV32IMCycle],
        initial_memory_state: &[i64],
        memory_layout: &MemoryLayout,
        r: Vec<F>,
        r_prime: Vec<F>,
        transcript: &mut ProofTranscript,
    ) -> (ReadWriteCheckingProof<F, ProofTranscript>, Vec<F>, Vec<F>) {
        const DEGREE: usize = 3;
        let K = r.len().pow2();
        let T = r_prime.len().pow2();
        debug_assert_eq!(trace.len(), T);

        // Used to batch the read-checking and write-checking sumcheck
        // (see Section 4.2.1)
        let z: F = transcript.challenge_scalar();

        let num_rounds = K.log_2() + T.log_2();
        let mut r_cycle: Vec<F> = Vec::with_capacity(T.log_2());
        let mut r_address: Vec<F> = Vec::with_capacity(K.log_2());

        let num_chunks = rayon::current_num_threads().next_power_of_two().min(T);
        let chunk_size = T / num_chunks;

        let span = tracing::span!(tracing::Level::INFO, "compute deltas");
        let _guard = span.enter();

        let deltas: Vec<Vec<i64>> = trace[..T - chunk_size]
            .par_chunks_exact(chunk_size)
            .map(|trace_chunk| {
                let mut delta = vec![0i64; K];
                for cycle in trace_chunk.iter() {
                    let ram_op = cycle.ram_access();
                    if let RAMAccess::Write(write) = ram_op {
                        let increment = write.post_value as i64 - write.pre_value as i64;
                        let k = remap_address(write.address, memory_layout) as usize;
                        delta[k] += increment;
                    }
                }
                delta
            })
            .collect();

        drop(_guard);
        drop(span);

        let span = tracing::span!(tracing::Level::INFO, "compute checkpoints");
        let _guard = span.enter();

        // #[cfg(test)]
        // let mut val_test: MultilinearPolynomial<F> = {
        //     // Compute Val in cycle-major order, since we will be binding
        //     // from low-to-high starting with the cycle variables
        //     let mut val: Vec<u64> = vec![0; K * T];
        //     val.par_chunks_mut(T).enumerate().for_each(|(k, val_k)| {
        //         let mut current_val = initial_memory_state[k] as u64;
        //         for j in 0..T {
        //             val_k[j] = current_val;
        //             if let RAMAccess::Write(write) = trace[j].ram_access() {
        //                 if remap_address(write.address, memory_layout) == k as u64 {
        //                     current_val = write.post_value;
        //                 }
        //             }
        //         }
        //     });
        //     MultilinearPolynomial::from(val)
        // };
        // #[cfg(test)]
        // let mut ra_test = {
        //     // Compute ra in cycle-major order, since we will be binding
        //     // from low-to-high starting with the cycle variables
        //     let mut ra: Vec<F> = unsafe_allocate_zero_vec(K * T);
        //     ra.par_chunks_mut(T).enumerate().for_each(|(k, ra_k)| {
        //         for j in 0..T {
        //             if remap_address(trace[j].ram_access().address() as u64, memory_layout)
        //                 == k as u64
        //             {
        //                 ra_k[j] = F::one();
        //             }
        //         }
        //     });
        //     MultilinearPolynomial::from(ra)
        // };

        // Value in register k before the jth cycle, for j \in {0, chunk_size, 2 * chunk_size, ...}
        let mut checkpoints: Vec<Vec<i64>> = Vec::with_capacity(num_chunks);
        checkpoints.push(initial_memory_state.to_vec());

        for (chunk_index, delta) in deltas.into_iter().enumerate() {
            let next_checkpoint = checkpoints[chunk_index]
                .par_iter()
                .zip(delta.into_par_iter())
                .map(|(val_k, delta_k)| val_k + delta_k)
                .collect();
            checkpoints.push(next_checkpoint);
        }
        // TODO(moodlezoup): could potentially generate these checkpoints in the tracer
        // Generate checkpoints as a flat vector because it will be turned into the
        // materialized Val polynomial after the first half of sumcheck.
        let mut val_checkpoints: Vec<F> = unsafe_allocate_zero_vec(K * num_chunks);
        val_checkpoints
            .par_chunks_mut(K)
            .zip(checkpoints.into_par_iter())
            .for_each(|(val_checkpoint, checkpoint)| {
                val_checkpoint
                    .iter_mut()
                    .zip(checkpoint.iter())
                    .for_each(|(dest, src)| *dest = F::from_i64(*src))
            });

        drop(_guard);
        drop(span);

        // #[cfg(test)]
        // {
        //     // Check that checkpoints are correct
        //     for (chunk_index, checkpoint) in val_checkpoints.chunks(K).enumerate() {
        //         let j = chunk_index * chunk_size;
        //         for (k, V_k) in checkpoint.iter().enumerate() {
        //             assert_eq!(
        //                 *V_k,
        //                 val_test.get_bound_coeff(k * T + j),
        //                 "k = {k}, j = {j}"
        //             );
        //         }
        //     }
        // }

        // A table that, in round i of sumcheck, stores all evaluations
        //     EQ(x, r_i, ..., r_1)
        // as x ranges over {0, 1}^i.
        // (As described in "Computing other necessary arrays and worst-case
        // accounting", Section 8.2.2)
        let mut A: Vec<F> = unsafe_allocate_zero_vec(chunk_size);
        A[0] = F::one();

        let span = tracing::span!(
            tracing::Level::INFO,
            "compute I (increments data structure)"
        );
        let _guard = span.enter();

        // Data structure described in Equation (72)
        let mut I: Vec<Vec<(usize, usize, F, F)>> = trace
            .par_chunks(chunk_size)
            .enumerate()
            .map(|(chunk_index, trace_chunk)| {
                // Row index of the I matrix
                let mut j = chunk_index * chunk_size;
                let I_chunk = trace_chunk
                    .iter()
                    .map(|cycle| {
                        let ram_op = cycle.ram_access();
                        let inc = match ram_op {
                            RAMAccess::Read(read) => {
                                let k = remap_address(read.address, memory_layout) as usize;
                                (j, k, F::zero(), F::zero())
                            }
                            RAMAccess::Write(write) => {
                                let k = remap_address(write.address, memory_layout) as usize;
                                let increment = write.post_value as i64 - write.pre_value as i64;
                                (j, k, F::zero(), F::from_i64(increment))
                            }
                            RAMAccess::NoOp => (j, 0, F::zero(), F::zero()),
                        };
                        j += 1;
                        inc
                    })
                    .collect();
                I_chunk
            })
            .collect();

        drop(_guard);
        drop(span);

        let read_values: Vec<u64> = trace
            .par_iter()
            .map(|cycle| {
                let ram_op = cycle.ram_access();
                match ram_op {
                    RAMAccess::Read(read) => read.value,
                    RAMAccess::Write(write) => write.pre_value,
                    RAMAccess::NoOp => 0,
                }
            })
            .collect();
        let rv = MultilinearPolynomial::from(read_values);
        let write_values: Vec<u64> = trace
            .par_iter()
            .map(|cycle| {
                let ram_op = cycle.ram_access();
                match ram_op {
                    RAMAccess::Read(read) => read.value,
                    RAMAccess::Write(write) => write.post_value,
                    RAMAccess::NoOp => 0,
                }
            })
            .collect();
        let mut wv = MultilinearPolynomial::from(write_values);

        // z * eq(r, k)
        let mut z_eq_r = MultilinearPolynomial::from(EqPolynomial::evals_parallel(&r, Some(z)));
        // eq(r', j)
        let mut eq_r_prime = MultilinearPolynomial::from(EqPolynomial::evals(&r_prime));

        // rv(r')
        let rv_eval = rv.evaluate(&r_prime);

        let span = tracing::span!(tracing::Level::INFO, "compute Inc(r, r')");
        let _guard = span.enter();

        // z * Inc(r, r')
        let inc_eval: F = trace
            .par_iter()
            .enumerate()
            .map(|(j, cycle)| {
                let ram_op = cycle.ram_access();
                let (address, increment) = match ram_op {
                    RAMAccess::Read(read) => {
                        (remap_address(read.address, memory_layout), F::zero())
                    }
                    RAMAccess::Write(write) => (
                        remap_address(write.address, memory_layout),
                        F::from_i64(write.post_value as i64 - write.pre_value as i64),
                    ),
                    RAMAccess::NoOp => (0, F::zero()),
                };
                z_eq_r.get_coeff(address as usize) * eq_r_prime.get_coeff(j) * increment
            })
            .sum();

        drop(_guard);
        drop(span);

        // Linear combination of the read-checking claim (which is rv(r')) and the
        // write-checking claim (which is Inc(r, r'))
        let mut previous_claim = rv_eval + inc_eval;
        let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::with_capacity(num_rounds);

        let span = tracing::span!(
            tracing::Level::INFO,
            "First log(T / num_chunks) rounds of sumcheck"
        );
        let _guard = span.enter();

        /// A collection of vectors that are used in each of the first log(T / num_chunks)
        /// rounds of sumcheck. There is one `DataBuffers` struct per thread/chunk, reused
        /// across all log(T / num_chunks) rounds.
        struct DataBuffers<F: JoltField> {
            /// Contains
            ///     Val(k, j', 0, ..., 0)
            /// as we iterate over rows j' \in {0, 1}^(log(T) - i)
            val_j_0: Vec<F>,
            /// `val_j_r[0]` contains
            ///     Val(k, j'', 0, r_i, ..., r_1)
            /// `val_j_r[1]` contains
            ///     Val(k, j'', 1, r_i, ..., r_1)
            /// as we iterate over rows j' \in {0, 1}^(log(T) - i)
            val_j_r: [Vec<F>; 2],
            /// `ra[0]` contains
            ///     ra(k, j'', 0, r_i, ..., r_1)
            /// `ra[1]` contains
            ///     ra(k, j'', 1, r_i, ..., r_1)
            /// as we iterate over rows j' \in {0, 1}^(log(T) - i),
            ra: [Vec<F>; 2],
            dirty_indices: Vec<usize>,
        }
        let mut data_buffers: Vec<DataBuffers<F>> = (0..num_chunks)
            .into_par_iter()
            .map(|_| DataBuffers {
                val_j_0: Vec::with_capacity(K),
                val_j_r: [unsafe_allocate_zero_vec(K), unsafe_allocate_zero_vec(K)],
                ra: [unsafe_allocate_zero_vec(K), unsafe_allocate_zero_vec(K)],
                dirty_indices: Vec::with_capacity(K),
            })
            .collect();

        // First log(T / num_chunks) rounds of sumcheck
        for round in 0..chunk_size.log_2() {
            // #[cfg(test)]
            // {
            //     let mut expected_claim = F::zero();
            //     for j in 0..(T >> round) {
            //         let mut inner_sum = F::zero();
            //         for k in 0..K {
            //             let kj = k * (T >> round) + j;
            //             // read-checking sumcheck
            //             inner_sum += ra_test.get_bound_coeff(kj) * val_test.get_bound_coeff(kj);
            //             // write-checking sumcheck
            //             inner_sum += z_eq_r.get_bound_coeff(k)
            //                 * ra_test.get_bound_coeff(kj)
            //                 * (wv.get_bound_coeff(j) - val_test.get_bound_coeff(kj))
            //         }
            //         expected_claim += eq_r_prime.get_bound_coeff(j) * inner_sum;
            //     }
            //     assert_eq!(
            //         expected_claim, previous_claim,
            //         "Sumcheck sanity check failed in round {round}"
            //     );
            // }

            let inner_span = tracing::span!(tracing::Level::INFO, "Compute univariate poly");
            let _inner_guard = inner_span.enter();

            let univariate_poly_evals: [F; 3] = I
                .par_iter()
                .zip(data_buffers.par_iter_mut())
                .zip(val_checkpoints.par_chunks(K))
                .map(|((I_chunk, buffers), checkpoint)| {
                    let mut evals = [F::zero(), F::zero(), F::zero()];

                    let DataBuffers {
                        val_j_0,
                        val_j_r,
                        ra,
                        dirty_indices,
                    } = buffers;

                    *val_j_0 = checkpoint.to_vec();

                    // Iterate over I_chunk, two rows at a time.
                    I_chunk
                        .chunk_by(|a, b| a.0 / 2 == b.0 / 2)
                        .for_each(|inc_chunk| {
                            let j_prime = inc_chunk[0].0; // row index

                            for j in j_prime << round..(j_prime + 1) << round {
                                let j_bound = j % (1 << round);
                                let k = remap_address(
                                    trace[j].ram_access().address() as u64,
                                    memory_layout,
                                ) as usize;
                                if ra[0][k].is_zero() {
                                    dirty_indices.push(k);
                                }
                                ra[0][k] += A[j_bound];
                            }

                            for j in (j_prime + 1) << round..(j_prime + 2) << round {
                                let j_bound = j % (1 << round);
                                let k = remap_address(
                                    trace[j].ram_access().address() as u64,
                                    memory_layout,
                                ) as usize;
                                if ra[0][k].is_zero() && ra[1][k].is_zero() {
                                    dirty_indices.push(k);
                                }
                                ra[1][k] += A[j_bound];
                            }

                            for &k in dirty_indices.iter() {
                                val_j_r[0][k] = val_j_0[k];
                            }
                            let mut inc_iter = inc_chunk.iter().peekable();

                            // First of the two rows
                            loop {
                                let (row, col, inc_lt, inc) = inc_iter.next().unwrap();
                                debug_assert_eq!(*row, j_prime);
                                val_j_r[0][*col] += *inc_lt;
                                val_j_0[*col] += *inc;
                                if inc_iter.peek().unwrap().0 != j_prime {
                                    break;
                                }
                            }
                            for &k in dirty_indices.iter() {
                                val_j_r[1][k] = val_j_0[k];
                            }

                            // Second of the two rows
                            for inc in inc_iter {
                                let (row, col, inc_lt, inc) = *inc;
                                debug_assert_eq!(row, j_prime + 1);
                                val_j_r[1][col] += inc_lt;
                                val_j_0[col] += inc;
                            }

                            let eq_r_prime_evals = eq_r_prime.sumcheck_evals(
                                j_prime / 2,
                                DEGREE,
                                BindingOrder::LowToHigh,
                            );
                            let wv_evals =
                                wv.sumcheck_evals(j_prime / 2, DEGREE, BindingOrder::LowToHigh);

                            let mut inner_sum_evals = [F::zero(); 3];
                            for k in dirty_indices.drain(..) {
                                if !ra[0][k].is_zero() || !ra[1][k].is_zero() {
                                    let m_ra = ra[1][k] - ra[0][k];
                                    let ra_eval_2 = ra[1][k] + m_ra;
                                    let ra_eval_3 = ra_eval_2 + m_ra;

                                    let m_val = val_j_r[1][k] - val_j_r[0][k];
                                    let val_eval_2 = val_j_r[1][k] + m_val;
                                    let val_eval_3 = val_eval_2 + m_val;

                                    let z_eq_r_eval = z_eq_r.get_coeff(k);
                                    inner_sum_evals[0] += ra[0][k].mul_0_optimized(
                                        val_j_r[0][k] + z_eq_r_eval * (wv_evals[0] - val_j_r[0][k]),
                                    );
                                    inner_sum_evals[1] += ra_eval_2
                                        * (val_eval_2 + z_eq_r_eval * (wv_evals[1] - val_eval_2));
                                    inner_sum_evals[2] += ra_eval_3
                                        * (val_eval_3 + z_eq_r_eval * (wv_evals[2] - val_eval_3));

                                    ra[0][k] = F::zero();
                                    ra[1][k] = F::zero();
                                }

                                val_j_r[0][k] = F::zero();
                                val_j_r[1][k] = F::zero();
                            }

                            evals[0] += eq_r_prime_evals[0] * inner_sum_evals[0];
                            evals[1] += eq_r_prime_evals[1] * inner_sum_evals[1];
                            evals[2] += eq_r_prime_evals[2] * inner_sum_evals[2];
                        });

                    evals
                })
                .reduce(
                    || [F::zero(); DEGREE],
                    |running, new| {
                        [
                            running[0] + new[0],
                            running[1] + new[1],
                            running[2] + new[2],
                        ]
                    },
                );

            let univariate_poly = UniPoly::from_evals(&[
                univariate_poly_evals[0],
                previous_claim - univariate_poly_evals[0],
                univariate_poly_evals[1],
                univariate_poly_evals[2],
            ]);

            drop(_inner_guard);
            drop(inner_span);

            let compressed_poly = univariate_poly.compress();
            compressed_poly.append_to_transcript(transcript);
            compressed_polys.push(compressed_poly);

            let r_j = transcript.challenge_scalar::<F>();
            r_cycle.insert(0, r_j);

            previous_claim = univariate_poly.evaluate(&r_j);

            let inner_span = tracing::span!(tracing::Level::INFO, "Bind I");
            let _inner_guard = inner_span.enter();

            // Bind I
            I.par_iter_mut().for_each(|I_chunk| {
                // Note: A given row in an I_chunk may not be ordered by k after binding
                let mut next_bound_index = 0;
                let mut bound_indices: Vec<Option<usize>> = vec![None; K];

                for i in 0..I_chunk.len() {
                    let (j_prime, k, inc_lt, inc) = I_chunk[i];
                    if let Some(bound_index) = bound_indices[k] {
                        if I_chunk[bound_index].0 == j_prime / 2 {
                            // Neighbor was already processed
                            debug_assert!(j_prime % 2 == 1);
                            I_chunk[bound_index].2 += r_j * inc_lt;
                            I_chunk[bound_index].3 += inc;
                            continue;
                        }
                    }
                    // First time this k has been encountered
                    let bound_value = if j_prime % 2 == 0 {
                        // (1 - r_j) * inc_lt + r_j * inc
                        inc_lt + r_j * (inc - inc_lt)
                    } else {
                        r_j * inc_lt
                    };
                    I_chunk[next_bound_index] = (j_prime / 2, k, bound_value, inc);
                    bound_indices[k] = Some(next_bound_index);
                    next_bound_index += 1;
                }
                I_chunk.truncate(next_bound_index);
            });

            drop(_inner_guard);
            drop(inner_span);

            rayon::join(
                || wv.bind_parallel(r_j, BindingOrder::LowToHigh),
                || eq_r_prime.bind_parallel(r_j, BindingOrder::LowToHigh),
            );

            // #[cfg(test)]
            // {
            //     val_test.bind_parallel(r_j, BindingOrder::LowToHigh);
            //     ra_test.bind_parallel(r_j, BindingOrder::LowToHigh);

            //     // Check that row indices of I are non-decreasing
            //     let mut current_row = 0;
            //     for I_chunk in I.iter() {
            //         for (row, _, _, _) in I_chunk {
            //             if *row != current_row {
            //                 assert_eq!(*row, current_row + 1);
            //                 current_row = *row;
            //             }
            //         }
            //     }
            // }

            let inner_span = tracing::span!(tracing::Level::INFO, "Update A");
            let _inner_guard = inner_span.enter();

            // Update A for this round (see Equation 55)
            let (A_left, A_right) = A.split_at_mut(1 << round);
            A_left
                .par_iter_mut()
                .zip(A_right.par_iter_mut())
                .for_each(|(x, y)| {
                    *y = *x * r_j;
                    *x -= *y;
                });
        }

        drop(_guard);
        drop(span);

        // At this point I has been bound to a point where each chunk contains a single row,
        // so we might as well materialize the full `ra`, `wa`, and `Val` polynomials and perform
        // standard sumcheck directly using those polynomials.

        let span = tracing::span!(tracing::Level::INFO, "Materialize ra polynomial");
        let _guard = span.enter();

        let mut ra: Vec<F> = unsafe_allocate_zero_vec(K * num_chunks);
        ra.par_chunks_mut(K)
            .enumerate()
            .for_each(|(chunk_index, ra_chunk)| {
                for (j_bound, cycle) in trace
                    [chunk_index * chunk_size..(chunk_index + 1) * chunk_size]
                    .iter()
                    .enumerate()
                {
                    let ram_op = cycle.ram_access();
                    let k = remap_address(ram_op.address() as u64, memory_layout) as usize;
                    ra_chunk[k] += A[j_bound];
                }
            });
        let mut ra = MultilinearPolynomial::from(ra);

        drop(_guard);
        drop(span);

        let span = tracing::span!(tracing::Level::INFO, "Materialize Val polynomial");
        let _guard = span.enter();

        let mut val: Vec<F> = val_checkpoints;
        val.par_chunks_mut(K)
            .zip(I.into_par_iter())
            .enumerate()
            .for_each(|(chunk_index, (val_chunk, I_chunk))| {
                for (j, k, inc_lt, _inc) in I_chunk.into_iter() {
                    debug_assert_eq!(j, chunk_index);
                    val_chunk[k] += inc_lt;
                }
            });
        let mut val = MultilinearPolynomial::from(val);

        drop(_guard);
        drop(span);

        let span = tracing::span!(tracing::Level::INFO, "Remaining rounds of sumcheck");
        let _guard = span.enter();

        // Remaining rounds of sumcheck
        for round in 0..num_rounds - chunk_size.log_2() {
            let inner_span = tracing::span!(tracing::Level::INFO, "Compute univariate poly");
            let _inner_guard = inner_span.enter();

            let univariate_poly_evals: [F; 3] = if eq_r_prime.len() > 1 {
                // Not done binding cycle variables yet
                (0..eq_r_prime.len() / 2)
                    .into_par_iter()
                    .map(|j| {
                        let eq_r_prime_evals =
                            eq_r_prime.sumcheck_evals(j, DEGREE, BindingOrder::HighToLow);
                        let wv_evals = wv.sumcheck_evals(j, DEGREE, BindingOrder::HighToLow);

                        let inner_sum_evals: [F; 3] = (0..K)
                            .into_par_iter()
                            .map(|k| {
                                let index = j * K + k;
                                let ra_evals =
                                    ra.sumcheck_evals(index, DEGREE, BindingOrder::HighToLow);
                                let val_evals =
                                    val.sumcheck_evals(index, DEGREE, BindingOrder::HighToLow);
                                let z_eq_r_eval = z_eq_r.get_coeff(k);

                                [
                                    ra_evals[0].mul_0_optimized(val_evals[0])
                                        + z_eq_r_eval
                                            .mul_0_optimized(ra_evals[0])
                                            .mul_0_optimized(wv_evals[0] - val_evals[0]),
                                    ra_evals[1].mul_0_optimized(val_evals[1])
                                        + z_eq_r_eval
                                            .mul_0_optimized(ra_evals[1])
                                            .mul_0_optimized(wv_evals[1] - val_evals[1]),
                                    ra_evals[2].mul_0_optimized(val_evals[2])
                                        + z_eq_r_eval
                                            .mul_0_optimized(ra_evals[2])
                                            .mul_0_optimized(wv_evals[2] - val_evals[2]),
                                ]
                            })
                            .reduce(
                                || [F::zero(); 3],
                                |running, new| {
                                    [
                                        running[0] + new[0],
                                        running[1] + new[1],
                                        running[2] + new[2],
                                    ]
                                },
                            );

                        [
                            eq_r_prime_evals[0] * inner_sum_evals[0],
                            eq_r_prime_evals[1] * inner_sum_evals[1],
                            eq_r_prime_evals[2] * inner_sum_evals[2],
                        ]
                    })
                    .reduce(
                        || [F::zero(); 3],
                        |running, new| {
                            [
                                running[0] + new[0],
                                running[1] + new[1],
                                running[2] + new[2],
                            ]
                        },
                    )
            } else {
                // Cycle variables are fully bound, so:
                // eq(r', r_cycle) is a constant
                let eq_r_prime_eval = eq_r_prime.final_sumcheck_claim();
                // ...and wv(r_cycle) is a constant
                let wv_eval = wv.final_sumcheck_claim();

                let evals = (0..ra.len() / 2)
                    .into_par_iter()
                    .map(|k| {
                        let z_eq_r_evals =
                            z_eq_r.sumcheck_evals(k, DEGREE, BindingOrder::HighToLow);
                        let ra_evals = ra.sumcheck_evals(k, DEGREE, BindingOrder::HighToLow);
                        let val_evals = val.sumcheck_evals(k, DEGREE, BindingOrder::HighToLow);

                        [
                            ra_evals[0] * val_evals[0]
                                + z_eq_r_evals[0] * ra_evals[0] * (wv_eval - val_evals[0]),
                            ra_evals[1] * val_evals[1]
                                + z_eq_r_evals[1] * ra_evals[1] * (wv_eval - val_evals[1]),
                            ra_evals[2] * val_evals[2]
                                + z_eq_r_evals[2] * ra_evals[2] * (wv_eval - val_evals[2]),
                        ]
                    })
                    .reduce(
                        || [F::zero(); 3],
                        |running, new| {
                            [
                                running[0] + new[0],
                                running[1] + new[1],
                                running[2] + new[2],
                            ]
                        },
                    );
                [
                    eq_r_prime_eval * evals[0],
                    eq_r_prime_eval * evals[1],
                    eq_r_prime_eval * evals[2],
                ]
            };

            let univariate_poly = UniPoly::from_evals(&[
                univariate_poly_evals[0],
                previous_claim - univariate_poly_evals[0],
                univariate_poly_evals[1],
                univariate_poly_evals[2],
            ]);

            drop(_inner_guard);
            drop(inner_span);

            let compressed_poly = univariate_poly.compress();
            compressed_poly.append_to_transcript(transcript);
            compressed_polys.push(compressed_poly);

            let r_j = transcript.challenge_scalar::<F>();
            previous_claim = univariate_poly.evaluate(&r_j);

            // Bind polynomials
            if eq_r_prime.len() > 1 {
                // Bind a cycle variable j
                r_cycle.insert(round, r_j);
                // Note that `eq_r` is a polynomial over only the address variables,
                // so it is not bound here
                [&mut ra, &mut wv, &mut val, &mut eq_r_prime]
                    .into_par_iter()
                    .for_each(|poly| poly.bind_parallel(r_j, BindingOrder::HighToLow));
            } else {
                // Bind an address variable k
                r_address.push(r_j);
                // Note that `wv` and `eq_r_prime` are polynomials over only the cycle
                // variables, so they are not bound here
                [&mut ra, &mut val, &mut z_eq_r]
                    .into_par_iter()
                    .for_each(|poly| poly.bind_parallel(r_j, BindingOrder::HighToLow));
            }
        }

        let proof = ReadWriteCheckingProof {
            sumcheck_proof: SumcheckInstanceProof::new(compressed_polys),
            ra_claim: ra.final_sumcheck_claim(),
            rv_claim: rv_eval,
            wv_claim: wv.final_sumcheck_claim(),
            val_claim: val.final_sumcheck_claim(),
            inc_claim: inc_eval * z.inverse().unwrap(),
            sumcheck_switch_index: chunk_size.log_2(),
        };

        drop_in_background_thread((ra, wv, val, data_buffers, z_eq_r, eq_r_prime, A));

        (proof, r_address, r_cycle)
    }
}

/// Implements the sumcheck prover for the Val-evaluation sumcheck described in
/// Section 8.1 and Appendix B of the Twist+Shout paper
/// TODO(moodlezoup): incorporate optimization from Appendix B.2
#[tracing::instrument(skip_all)]
pub fn prove_val_evaluation<
    F: JoltField,
    ProofTranscript: Transcript,
    PCS: CommitmentScheme<ProofTranscript, Field = F>,
>(
    preprocessing: &JoltProverPreprocessing<F, PCS, ProofTranscript>,
    trace: &[RV32IMCycle],
    memory_layout: &MemoryLayout,
    r_address: Vec<F>,
    r_cycle: Vec<F>,
    init_eval: F,
    claimed_evaluation: F,
    transcript: &mut ProofTranscript,
) -> (ValEvaluationProof<F, ProofTranscript>, Vec<F>) {
    let T = r_cycle.len().pow2();

    // Compute the size-K table storing all eq(r_address, k) evaluations for
    // k \in {0, 1}^log(K)
    let eq_r_address = EqPolynomial::evals(&r_address);

    let span = tracing::span!(tracing::Level::INFO, "compute wa(r_address, j)");
    let _guard = span.enter();

    // Compute the wa polynomial using the above table
    let wa: Vec<F> = trace
        .par_iter()
        .map(|cycle| {
            let ram_op = cycle.ram_access();
            match ram_op {
                RAMAccess::Write(write) => {
                    let k = remap_address(write.address, memory_layout) as usize;
                    eq_r_address[k]
                }
                _ => F::zero(),
            }
        })
        .collect();
    let wa = MultilinearPolynomial::from(wa);

    drop(_guard);
    drop(span);

    let inc = CommittedPolynomials::RamInc.generate_witness(preprocessing, trace);

    let span = tracing::span!(tracing::Level::INFO, "compute LT(j, r_cycle)");
    let _guard = span.enter();

    let mut lt: Vec<F> = unsafe_allocate_zero_vec(T);
    for (i, r) in r_cycle.iter().rev().enumerate() {
        let (evals_left, evals_right) = lt.split_at_mut(1 << i);
        evals_left
            .par_iter_mut()
            .zip(evals_right.par_iter_mut())
            .for_each(|(x, y)| {
                *y = *x * r;
                *x += *r - *y;
            });
    }
    let lt = MultilinearPolynomial::from(lt);

    drop(_guard);
    drop(span);

    // Create the sumcheck instance
    let mut sumcheck_instance: ValEvaluationSumcheck<F> = ValEvaluationSumcheck {
        claimed_evaluation,
        init_eval,
        prover_state: Some(ValEvaluationProverState { inc, wa, lt }),
        verifier_state: None,
        claims: None,
    };

    let span = tracing::span!(tracing::Level::INFO, "Val-evaluation sumcheck");
    let _guard = span.enter();

    // Run the sumcheck protocol
    let (sumcheck_proof, r_cycle_prime) = <ValEvaluationSumcheck<F> as BatchableSumcheckInstance<
        F,
        ProofTranscript,
    >>::prove_single(&mut sumcheck_instance, transcript);

    drop(_guard);
    drop(span);

    let claims = sumcheck_instance.claims.expect("Claims should be set");

    let proof = ValEvaluationProof {
        sumcheck_proof,
        inc_claim: claims.inc_claim,
        wa_claim: claims.wa_claim,
    };

    // Clean up
    if let Some(prover_state) = sumcheck_instance.prover_state {
        drop_in_background_thread((
            prover_state.inc,
            prover_state.wa,
            eq_r_address,
            prover_state.lt,
        ));
    }

    (proof, r_cycle_prime)
}

#[tracing::instrument(skip_all)]
fn prove_ra_booleanity<F: JoltField, ProofTranscript: Transcript>(
    trace: &[RV32IMCycle],
    memory_layout: &MemoryLayout,
    eq_r_cycle: &[F],
    K: usize,
    d: usize,
    transcript: &mut ProofTranscript,
) -> (
    SumcheckInstanceProof<F, ProofTranscript>,
    Vec<F>,
    Vec<F>,
    Vec<F>,
) {
    let T = trace.len();
    let num_chunks = rayon::current_num_threads().next_power_of_two().min(T);
    let _chunk_size = (T / num_chunks).max(1);

    let r_address: Vec<F> = transcript.challenge_vector(K.log_2());

    // Get z challenges for batching
    let z_challenges: Vec<F> = transcript.challenge_vector(d);
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
        let address = remap_address(cycle.ram_access().address() as u64, memory_layout) as usize;

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

    // Create the sumcheck instance
    let mut sumcheck_instance = BooleanitySumcheck {
        K,
        T,
        d,
        prover_state: Some(BooleanityProverState {
            B,
            F,
            G: G_arrays,
            D,
            H: None,
            eq_r_r: F::zero(),
            z_powers: z_powers.clone(),
            d,
            chunk_sizes,
        }),
        verifier_state: None,
        ra_claims: None,
        current_round: 0,
        trace: Some(trace.to_vec()),
        memory_layout: Some(memory_layout.clone()),
    };

    let span = tracing::span!(tracing::Level::INFO, "Booleanity sumcheck");
    let _guard = span.enter();

    // Run the sumcheck protocol
    let (sumcheck_proof, r) = <BooleanitySumcheck<F> as BatchableSumcheckInstance<
        F,
        ProofTranscript,
    >>::prove_single(&mut sumcheck_instance, transcript);

    drop(_guard);
    drop(span);

    let ra_claims = sumcheck_instance
        .ra_claims
        .expect("RA claims should be set");

    // Extract r_address_prime and r_cycle_prime from r
    let K_log = K.log_2();
    let (r_address_prime, r_cycle_prime) = r.split_at(K_log);

    (
        sumcheck_proof,
        r_address_prime.to_vec(),
        r_cycle_prime.to_vec(),
        ra_claims,
    )
}

#[tracing::instrument(skip_all)]
fn prove_ra_hamming_weight<F: JoltField, ProofTranscript: Transcript>(
    trace: &[RV32IMCycle],
    memory_layout: &MemoryLayout,
    eq_r_cycle: Vec<F>,
    K: usize,
    d: usize,
    transcript: &mut ProofTranscript,
) -> (SumcheckInstanceProof<F, ProofTranscript>, Vec<F>, Vec<F>) {
    let T = trace.len();
    let num_chunks = rayon::current_num_threads().next_power_of_two().min(T);
    let chunk_size = (T / num_chunks).max(1);

    // Get z challenges for batching
    let z_challenges: Vec<F> = transcript.challenge_vector(d);
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

    // Compute F arrays for each decomposed part
    let F_arrays: Vec<Vec<F>> = trace
        .par_chunks(chunk_size)
        .enumerate()
        .map(|(chunk_index, trace_chunk)| {
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
    let ra_polys: Vec<MultilinearPolynomial<F>> = F_arrays
        .into_iter()
        .map(MultilinearPolynomial::from)
        .collect();

    // Create the sumcheck instance
    let mut sumcheck_instance = HammingWeightSumcheck::new_prover(ra_polys, z_powers, d);

    // Prove the sumcheck
    let (sumcheck_proof, r_address_double_prime) = sumcheck_instance.prove_single(transcript);

    // Get the cached ra_claims
    let ra_claims = sumcheck_instance
        .cached_claims
        .expect("ra_claims should be cached after proving");

    (sumcheck_proof, r_address_double_prime, ra_claims)
}
