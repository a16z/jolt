#![allow(clippy::too_many_arguments)]

use std::vec;

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
        errors::ProofVerifyError,
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
use tracer::{
    emulator::memory::Memory,
    instruction::{RAMAccess, RV32IMCycle},
    JoltDevice,
};

#[derive(Debug, Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct RAMPreprocessing {
    min_bytecode_address: u64,
    bytecode_words: Vec<u32>,
}

impl RAMPreprocessing {
    pub fn preprocess(memory_init: Vec<(u64, u8)>) -> Self {
        let min_bytecode_address = memory_init
            .iter()
            .map(|(address, _)| *address)
            .min()
            .unwrap_or(0);

        let max_bytecode_address = memory_init
            .iter()
            .map(|(address, _)| *address)
            .max()
            .unwrap_or(0)
            + (BYTES_PER_INSTRUCTION as u64 - 1); // For RV32IM, instructions occupy 4 bytes, so the max bytecode address is the max instruction address + 3

        let num_words = max_bytecode_address.next_multiple_of(4) / 4 - min_bytecode_address / 4 + 1;
        let mut bytecode_words = vec![0u32; num_words as usize];
        // Convert bytes into words and populate `bytecode_words`
        for chunk in
            memory_init.chunk_by(|(address_a, _), (address_b, _)| address_a / 4 == address_b / 4)
        {
            let mut word = [0u8; 4];
            for (address, byte) in chunk {
                word[(address % 4) as usize] = *byte;
            }
            let word = u32::from_le_bytes(word);
            let remapped_index = (chunk[0].0 / 4 - min_bytecode_address / 4) as usize;
            bytecode_words[remapped_index] = word;
        }

        Self {
            min_bytecode_address,
            bytecode_words,
        }
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone)]
pub struct RAMTwistProof<F: JoltField, ProofTranscript: Transcript> {
    pub(crate) K: usize,
    /// Proof for the read-checking and write-checking sumchecks
    /// (steps 3 and 4 of Figure 9).
    read_write_checking_proof: RamReadWriteCheckingProof<F, ProofTranscript>,
    /// Proof of the Val-evaluation sumcheck (step 6 of Figure 9).
    val_evaluation_proof: ValEvaluationProof<F, ProofTranscript>,

    booleanity_proof: BooleanityProof<F, ProofTranscript>,
    ra_proof: RAProof<F, ProofTranscript>,
    hamming_weight_proof: HammingWeightProof<F, ProofTranscript>,
    raf_evaluation_proof: RafEvaluationProof<F, ProofTranscript>,
    output_proof: OutputProof<F, ProofTranscript>,
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone)]
pub struct ValEvaluationProof<F: JoltField, ProofTranscript: Transcript> {
    /// Sumcheck proof for the Val-evaluation sumcheck (steps 6 of Figure 9).
    sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    /// Inc(r_cycle')
    inc_claim: F,
    /// wa(r_address, r_cycle')
    wa_claim: F,
}

struct ValEvaluationProverState<F: JoltField> {
    /// Inc polynomial
    inc: MultilinearPolynomial<F>,
    /// wa polynomial
    wa: MultilinearPolynomial<F>,
    /// LT polynomial
    lt: MultilinearPolynomial<F>,
}

struct ValEvaluationVerifierState<F: JoltField> {
    /// log T
    num_rounds: usize,
    /// used to compute LT evaluation
    r_address: Vec<F>,
    /// used to compute LT evaluation
    r_cycle: Vec<F>,
}

#[derive(Clone)]
struct ValEvaluationSumcheckClaims<F: JoltField> {
    /// Inc(r_cycle')
    inc_claim: F,
    /// wa(r_address, r_cycle')
    wa_claim: F,
}

/// Val-evaluation sumcheck for RAM
struct ValEvaluationSumcheck<F: JoltField> {
    /// Initial claim value
    claimed_evaluation: F,
    /// Initial evaluation to subtract (for RAM)
    init_eval: F,
    /// Prover state
    prover_state: Option<ValEvaluationProverState<F>>,
    /// Verifier state
    verifier_state: Option<ValEvaluationVerifierState<F>>,
    /// Claims
    claims: Option<ValEvaluationSumcheckClaims<F>>,
}

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

    fn compute_prover_message(&mut self, _round: usize) -> Vec<F> {
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
    /// Size of address space
    K: usize,
    /// Number of cycles
    T: usize,
    /// D parameter as in Twist and Shout paper
    d: usize,
    /// r_address challenge
    r_address: Vec<F>,
    /// r_prime (r_cycle) challenge
    r_prime: Vec<F>,
    /// z powers
    z_powers: Vec<F>,
}

struct BooleanitySumcheck<F: JoltField> {
    /// Size of address space
    K: usize,
    /// Number of trace steps
    T: usize,
    /// D parameter as in Twist and Shout paper
    d: usize,
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

    fn compute_prover_message(&mut self, round: usize) -> Vec<F> {
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

struct HammingWeightProverState<F: JoltField> {
    /// The ra polynomials - one for each decomposed part
    ra: Vec<MultilinearPolynomial<F>>,
    /// z powers for batching
    z_powers: Vec<F>,
    /// D parameter as in Twist and Shout paper
    d: usize,
}

struct HammingWeightVerifierState<F: JoltField> {
    /// log K (number of rounds)
    log_K: usize,
    /// D parameter as in Twist and Shout paper
    d: usize,
    /// z powers for verification
    z_powers: Vec<F>,
}

struct HammingWeightSumcheck<F: JoltField> {
    /// The initial claim (sum of z powers for hamming weight)
    input_claim: F,
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

    fn new_verifier(log_K: usize, ra_claims: Vec<F>, z_powers: Vec<F>, d: usize) -> Self {
        // Compute input claim as sum of z powers
        let input_claim = z_powers.iter().sum();

        Self {
            input_claim,
            prover_state: None,
            verifier_state: Some(HammingWeightVerifierState { log_K, d, z_powers }),
            cached_claims: Some(ra_claims),
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

struct RafEvaluationProverState<F: JoltField> {
    /// The ra polynomial
    ra: MultilinearPolynomial<F>,
    /// The unmap polynomial
    unmap: UnmapRamAddressPolynomial<F>,
}

struct RafEvaluationVerifierState {
    /// log K (number of rounds)
    log_K: usize,
    /// Start address for unmap polynomial
    start_address: u64,
}

struct RafEvaluationSumcheck<F: JoltField> {
    /// The initial claim (raf_claim)
    input_claim: F,
    /// Prover state (only present for prover)
    prover_state: Option<RafEvaluationProverState<F>>,
    /// Verifier state (only present for verifier)
    verifier_state: Option<RafEvaluationVerifierState>,
    /// Cached ra_claim after sumcheck completion
    cached_claim: Option<F>,
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

    fn compute_prover_message(&mut self, _round: usize) -> Vec<F> {
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

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone)]
pub struct HammingWeightProof<F, ProofTranscript>
where
    F: JoltField,
    ProofTranscript: Transcript,
{
    sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    ra_claims: Vec<F>,
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone)]
pub struct RafEvaluationProof<F: JoltField, ProofTranscript: Transcript> {
    sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    ra_claim: F,
    raf_claim: F,
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

    pub fn verify(
        &self,
        K: usize,
        transcript: &mut ProofTranscript,
        memory_layout: &MemoryLayout,
    ) -> Result<Vec<F>, ProofVerifyError> {
        let sumcheck_instance = RafEvaluationSumcheck::new_verifier(
            self.raf_claim,
            K.log_2(),
            memory_layout.input_start,
            self.ra_claim,
        );

        let r_raf_sumcheck = sumcheck_instance.verify_single(&self.sumcheck_proof, transcript)?;

        Ok(r_raf_sumcheck)
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

        println!("picked D={d:?} for RAM");

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

    pub fn verify<PCS: CommitmentScheme<ProofTranscript, Field = F>>(
        &self,
        T: usize,
        preprocessing: &RAMPreprocessing,
        commitments: &JoltCommitments<F, PCS, ProofTranscript>,
        program_io: &JoltDevice,
        transcript: &mut ProofTranscript,
        opening_accumulator: &mut VerifierOpeningAccumulator<F, PCS, ProofTranscript>,
    ) -> Result<(), ProofVerifyError> {
        let log_K = self.K.log_2();
        let log_T = T.log_2();
        let r_prime: Vec<F> = transcript.challenge_vector(log_T);

        let (r_address, r_cycle) = RamReadWriteChecking::verify(
            &self.read_write_checking_proof,
            program_io,
            self.K,
            &r_prime,
            transcript,
        )?;

        let mut initial_memory_state = vec![0; self.K];
        // Copy bytecode
        let mut index = remap_address(
            preprocessing.min_bytecode_address,
            &program_io.memory_layout,
        ) as usize;
        for word in preprocessing.bytecode_words.iter() {
            initial_memory_state[index] = *word as i64;
            index += 1;
        }
        // Copy input bytes
        index = remap_address(
            program_io.memory_layout.input_start,
            &program_io.memory_layout,
        ) as usize;
        // Convert input bytes into words and populate `v_init`
        for chunk in program_io.inputs.chunks(4) {
            let mut word = [0u8; 4];
            for (i, byte) in chunk.iter().enumerate() {
                word[i] = *byte;
            }
            let word = u32::from_le_bytes(word);
            initial_memory_state[index] = word as i64;
            index += 1;
        }

        // TODO: Verifier currently materializes and evaluates Val_init itself,
        // but this is not tractable for large K
        let val_init: MultilinearPolynomial<F> = MultilinearPolynomial::from(initial_memory_state);
        let init_eval = val_init.evaluate(&r_address);

        // Create the sumcheck instance for verification
        let sumcheck_instance = ValEvaluationSumcheck {
            claimed_evaluation: self.read_write_checking_proof.claims.val_claim,
            init_eval,
            prover_state: None,
            verifier_state: Some(ValEvaluationVerifierState {
                num_rounds: log_T,
                r_address: r_address.clone(),
                r_cycle,
            }),
            claims: Some(ValEvaluationSumcheckClaims {
                inc_claim: self.val_evaluation_proof.inc_claim,
                wa_claim: self.val_evaluation_proof.wa_claim,
            }),
        };

        let mut r_cycle_prime = <ValEvaluationSumcheck<F> as BatchableSumcheckInstance<
            F,
            ProofTranscript,
        >>::verify_single(
            &sumcheck_instance,
            &self.val_evaluation_proof.sumcheck_proof,
            transcript,
        )?;

        // Cycle variables are bound from low to high
        r_cycle_prime.reverse();

        let inc_commitment = &commitments.commitments[CommittedPolynomials::RamInc.to_index()];
        opening_accumulator.append(
            &[inc_commitment],
            r_cycle_prime,
            &[self.val_evaluation_proof.inc_claim],
            transcript,
        );

        // TODO: Append Inc claim to opening proof accumulator

        let mut r_address: Vec<F> = transcript.challenge_vector(log_K);
        r_address = r_address.into_iter().rev().collect();

        // Calculate D dynamically
        // let d = (log_K / 8).max(1);
        let d = 1; // @TODO(markosg04) keeping d = 1 for legacy prove

        // Get z challenges for batching
        let z: F = transcript.challenge_scalar();
        let mut z_powers = vec![F::one(); d];
        for i in 1..d {
            z_powers[i] = z_powers[i - 1] * z;
        }

        let sumcheck_instance = BooleanitySumcheck {
            K: self.K,
            T,
            d,
            prover_state: None,
            verifier_state: Some(BooleanityVerifierState {
                K: self.K,
                T,
                d,
                r_address: r_address.clone(),
                r_prime: r_prime.clone(),
                z_powers: z_powers.clone(),
            }),
            ra_claims: Some(self.booleanity_proof.ra_claims.clone()),
            current_round: 0,
            trace: None,
            memory_layout: None,
        };

        let r_booleanity = <BooleanitySumcheck<F> as BatchableSumcheckInstance<
            F,
            ProofTranscript,
        >>::verify_single(
            &sumcheck_instance,
            &self.booleanity_proof.sumcheck_proof,
            transcript,
        )?;

        let (r_address_prime, r_cycle_prime) = r_booleanity.split_at(log_K);

        let r_cycle_prime: Vec<_> = r_cycle_prime.iter().copied().rev().collect();
        let r_address_prime = r_address_prime.iter().copied().rev().collect::<Vec<_>>();
        let ra_commitment = &commitments.commitments[CommittedPolynomials::RamRa(0).to_index()];

        let ra_claim = self.booleanity_proof.ra_claims[0]; // d = 1

        let mut r_cycle_bound = RASumcheck::<F>::verify(
            ra_claim,
            self.ra_proof.ra_i_claims.clone(),
            r_cycle_prime,
            T,
            d,
            &self.ra_proof.sumcheck_proof,
            transcript,
        )?;

        r_cycle_bound.reverse();
        let r_concat = [r_address_prime.as_slice(), r_cycle_bound.as_slice()].concat();

        opening_accumulator.append(
            &[ra_commitment],
            r_concat,
            &self.ra_proof.ra_i_claims,
            transcript,
        );

        // Get z challenges for hamming weight batching
        let z_hw_challenge: F = transcript.challenge_scalar();
        let mut z_hw_powers = vec![F::one(); d];
        for i in 1..d {
            z_hw_powers[i] = z_hw_powers[i - 1] * z_hw_challenge;
        }

        let sumcheck_instance = HammingWeightSumcheck::new_verifier(
            log_K,
            self.hamming_weight_proof.ra_claims.clone(),
            z_hw_powers,
            d,
        );

        let _r_hamming_weight = sumcheck_instance
            .verify_single(&self.hamming_weight_proof.sumcheck_proof, transcript)?;

        let _r_address_raf =
            self.raf_evaluation_proof
                .verify(self.K, transcript, &program_io.memory_layout)?;

        OutputSumcheck::verify(
            program_io,
            val_init,
            &r_address_prime,
            T,
            &self.output_proof,
            transcript,
        )?;

        // TODO: Append to opening proof accumulator

        Ok(())
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

pub fn remap_address(address: u64, memory_layout: &MemoryLayout) -> u64 {
    if address == 0 {
        return 0; // [JOLT-135]: Better handling for no-ops
    }
    if address >= memory_layout.input_start {
        (address - memory_layout.input_start) / 4 + 1
    } else {
        panic!("Unexpected address {address}")
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::transcript::KeccakTranscript;
    use ark_bn254::Fr;

    #[test]
    fn test_raf_evaluation_no_ops() {
        const K: usize = 1 << 16;
        const T: usize = 1 << 8;

        let memory_layout = MemoryLayout {
            max_input_size: 256,
            max_output_size: 256,
            input_start: 0x80000000,
            input_end: 0x80000100,
            output_start: 0x80001000,
            output_end: 0x80001100,
            stack_size: 1024,
            stack_end: 0x7FFFFF00,
            memory_size: 0x10000,
            memory_end: 0x80010000,
            panic: 0x80002000,
            termination: 0x80002001,
            io_end: 0x80002002,
        };

        // Create trace with only no-ops (address = 0)
        let mut trace = Vec::new();
        for i in 0..T {
            trace.push(RV32IMCycle::NoOp(i));
        }

        let mut prover_transcript = KeccakTranscript::new(b"test_no_ops");
        let r_cycle: Vec<Fr> = prover_transcript.challenge_vector(T.log_2());

        // Prove
        let proof =
            RafEvaluationProof::prove(&trace, &memory_layout, r_cycle, K, &mut prover_transcript);

        // Verify
        let mut verifier_transcript = KeccakTranscript::new(b"test_no_ops");
        let _r_cycle: Vec<Fr> = verifier_transcript.challenge_vector(T.log_2());

        let r_address_result = proof.verify(K, &mut verifier_transcript, &memory_layout);

        assert!(
            r_address_result.is_ok(),
            "No-op RAF evaluation verification failed"
        );
    }
}
