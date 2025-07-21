use std::sync::Arc;

use crate::jolt::vm::bytecode::{
    bytecode_to_val, BooleanityProof, BooleanityProverState, BooleanitySumcheck,
    BytecodePreprocessing, BytecodeShoutProof, CorePIOPHammingProof, CorePIOPHammingProverState,
    CorePIOPHammingSumcheck, RafBytecode, RafBytecodeProverState, RafEvaluationProof,
};
use crate::subprotocols::sumcheck::BatchableSumcheckVerifierInstance;
use crate::{
    field::JoltField,
    jolt::{vm::JoltProverPreprocessing, witness::CommittedPolynomials},
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        eq_poly::EqPolynomial,
        identity_poly::IdentityPolynomial,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        opening_proof::ProverOpeningAccumulator,
    },
    subprotocols::sumcheck::BatchableSumcheckInstance,
    utils::{math::Math, thread::unsafe_allocate_zero_vec, transcript::Transcript},
};
use rayon::prelude::*;
use tracer::instruction::RV32IMCycle;

impl<F: JoltField, ProofTranscript: Transcript> BytecodeShoutProof<F, ProofTranscript> {
    #[tracing::instrument(skip_all, name = "BytecodeShoutProof::prove")]
    pub fn prove<PCS: CommitmentScheme<ProofTranscript, Field = F>>(
        preprocessing: &JoltProverPreprocessing<F, PCS, ProofTranscript>,
        trace: &[RV32IMCycle],
        opening_accumulator: &mut ProverOpeningAccumulator<F, PCS, ProofTranscript>,
        transcript: &mut ProofTranscript,
    ) -> Self {
        //// start of state gen (to be handled by state manager)
        let bytecode_preprocessing = &preprocessing.shared.bytecode;
        let K = bytecode_preprocessing.bytecode.len().next_power_of_two();
        let T = trace.len();
        // TODO: this should come from Spartan
        let r_cycle: Vec<F> = transcript.challenge_vector(T.log_2());
        let r_shift: Vec<F> = transcript.challenge_vector(T.log_2());
        // Used to batch the core PIOP sumcheck and Hamming weight sumcheck
        // (see Section 4.2.1)
        let z: F = transcript.challenge_scalar();

        let E: Vec<F> = EqPolynomial::evals(&r_cycle);
        let E_shift: Vec<F> = EqPolynomial::evals(&r_shift);

        let span = tracing::span!(tracing::Level::INFO, "compute F");
        let _guard = span.enter();

        let num_chunks = rayon::current_num_threads()
            .next_power_of_two()
            .min(trace.len());
        let chunk_size = (trace.len() / num_chunks).max(1);
        let (F, F_shift): (Vec<_>, Vec<_>) = trace
            .par_chunks(chunk_size)
            .enumerate()
            .map(|(chunk_index, trace_chunk)| {
                let mut result: Vec<F> = unsafe_allocate_zero_vec(K);
                let mut result_shift: Vec<F> = unsafe_allocate_zero_vec(K);
                let mut j = chunk_index * chunk_size;
                for cycle in trace_chunk {
                    let k = bytecode_preprocessing.get_pc(cycle, j == trace.len() - 1);
                    result[k] += E[j];
                    result_shift[k] += E_shift[j];
                    j += 1;
                }
                (result, result_shift)
            })
            .reduce(
                || (unsafe_allocate_zero_vec(K), unsafe_allocate_zero_vec(K)),
                |(mut running, mut running_shift), (new, new_shift)| {
                    running
                        .par_iter_mut()
                        .zip(new.into_par_iter())
                        .for_each(|(x, y)| *x += y);
                    running_shift
                        .par_iter_mut()
                        .zip(new_shift.into_par_iter())
                        .for_each(|(x, y)| *x += y);
                    (running, running_shift)
                },
            );
        drop(_guard);
        drop(span);

        // Used to combine the various fields in each instruction into a single
        // field element.
        let gamma: F = transcript.challenge_scalar();
        let val: Vec<F> = bytecode_to_val(&bytecode_preprocessing.bytecode, gamma);

        let rv_claim: F = F
            .par_iter()
            .zip(val.par_iter())
            .map(|(&ra, &val)| ra * val)
            .sum();

        //// End of state gen

        // Prove core PIOP and Hamming weight sumcheck (they're combined into one here)
        let (core_piop_hamming_proof, r_address, raf_ra) =
            CorePIOPHammingProof::prove(F.clone(), val, z, rv_claim, K, transcript);
        let ra_claim = core_piop_hamming_proof.ra_claim;

        let unbound_ra_poly =
            CommittedPolynomials::BytecodeRa.generate_witness(preprocessing, trace);

        let r_address_rev = r_address.iter().copied().rev().collect::<Vec<_>>();

        opening_accumulator.append_sparse(
            vec![unbound_ra_poly.clone()],
            r_address_rev,
            r_cycle.clone(),
            vec![ra_claim],
        );

        // Prove booleanity
        let (booleanity_proof, r_address_prime, r_cycle_prime) =
            BooleanityProof::prove(bytecode_preprocessing, trace, &r_address, E, F, transcript);
        let ra_claim_prime = booleanity_proof.ra_claim_prime;

        let r_address_prime = r_address_prime.iter().copied().rev().collect::<Vec<_>>();
        let r_cycle_prime = r_cycle_prime.iter().rev().copied().collect::<Vec<_>>();

        // TODO: Reduce 2 ra claims to 1 (Section 4.5.2 of Proofs, Arguments, and Zero-Knowledge)
        opening_accumulator.append_sparse(
            vec![unbound_ra_poly],
            r_address_prime,
            r_cycle_prime,
            vec![ra_claim_prime],
        );

        // Prove raf
        let challenge: F = transcript.challenge_scalar();
        let raf_ra_shift = MultilinearPolynomial::from(F_shift);
        let raf_sumcheck = RafEvaluationProof::prove(
            bytecode_preprocessing,
            trace,
            raf_ra,
            raf_ra_shift,
            &r_cycle,
            &r_shift,
            challenge,
            transcript,
        );

        Self {
            core_piop_hamming: core_piop_hamming_proof,
            booleanity: booleanity_proof,
            raf_sumcheck,
        }
    }
}

impl<F: JoltField> CorePIOPHammingSumcheck<F> {
    pub fn new(
        input_claim: F,
        ra_poly: MultilinearPolynomial<F>,
        val_poly: MultilinearPolynomial<F>,
        z: F,
        K: usize,
    ) -> Self {
        Self {
            input_claim,
            z,
            K,
            prover_state: Some(CorePIOPHammingProverState { ra_poly, val_poly }),
            ra_claim: None,
            val_eval: None,
        }
    }
}

impl<F: JoltField, ProofTranscript: Transcript> BatchableSumcheckInstance<F, ProofTranscript>
    for CorePIOPHammingSumcheck<F>
{
    fn compute_prover_message(&mut self, _round: usize, _previous_claim: F) -> Vec<F> {
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");
        let degree = <Self as BatchableSumcheckVerifierInstance<F, ProofTranscript>>::degree(self);

        let univariate_poly_evals: [F; 2] = (0..prover_state.ra_poly.len() / 2)
            .into_par_iter()
            .map(|i| {
                let ra_evals =
                    prover_state
                        .ra_poly
                        .sumcheck_evals(i, degree, BindingOrder::LowToHigh);
                let val_evals =
                    prover_state
                        .val_poly
                        .sumcheck_evals(i, degree, BindingOrder::LowToHigh);

                // Compute ra[i] * (z + val[i]) for points 0 and 2
                [
                    ra_evals[0] * (self.z + val_evals[0]),
                    ra_evals[1] * (self.z + val_evals[1]),
                ]
            })
            .reduce(
                || [F::zero(); 2],
                |mut running, new| {
                    for i in 0..2 {
                        running[i] += new[i];
                    }
                    running
                },
            );

        univariate_poly_evals.to_vec()
    }

    fn bind(&mut self, r_j: F, _round: usize) {
        let prover_state = self
            .prover_state
            .as_mut()
            .expect("Prover state not initialized");

        rayon::join(
            || {
                prover_state
                    .ra_poly
                    .bind_parallel(r_j, BindingOrder::LowToHigh)
            },
            || {
                prover_state
                    .val_poly
                    .bind_parallel(r_j, BindingOrder::LowToHigh)
            },
        );
    }

    fn cache_openings(&mut self) {
        debug_assert!(self.ra_claim.is_none());
        debug_assert!(self.val_eval.is_none());
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");

        self.ra_claim = Some(prover_state.ra_poly.final_sumcheck_claim());
        self.val_eval = Some(prover_state.val_poly.final_sumcheck_claim());
    }
}

impl<F: JoltField, ProofTranscript: Transcript> CorePIOPHammingProof<F, ProofTranscript> {
    #[tracing::instrument(skip_all, name = "CorePIOPHammingProof::prove")]
    pub fn prove(
        F: Vec<F>,
        val: Vec<F>,
        z: F,
        rv_claim: F,
        K: usize,
        transcript: &mut ProofTranscript,
    ) -> (Self, Vec<F>, MultilinearPolynomial<F>) {
        // Linear combination of the core PIOP claim and the Hamming weight claim (which is 1)
        let input_claim = rv_claim + z;

        let ra_poly = MultilinearPolynomial::from(F);
        let raf_ra = ra_poly.clone(); // Clone before binding for RAF sumcheck to return original
        let val_poly = MultilinearPolynomial::from(val);

        let mut core_piop_sumcheck =
            CorePIOPHammingSumcheck::new(input_claim, ra_poly, val_poly, z, K);

        let (sumcheck_proof, r_address) = core_piop_sumcheck.prove_single(transcript);

        let ra_claim = core_piop_sumcheck
            .ra_claim
            .expect("ra_claim should be set after prove_single");
        let val_eval = core_piop_sumcheck
            .val_eval
            .expect("val_eval should be set after prove_single");

        let proof = Self {
            sumcheck_proof,
            ra_claim,
            rv_claim,
            val_eval,
        };

        (proof, r_address, raf_ra)
    }
}

impl<F: JoltField> BooleanitySumcheck<F> {
    pub fn new(
        preprocessing: &BytecodePreprocessing,
        trace: &[RV32IMCycle],
        r: &[F],
        D: Vec<F>,
        G: Vec<F>,
        K: usize,
        T: usize,
    ) -> Self {
        let B = MultilinearPolynomial::from(EqPolynomial::evals(r));

        // Initialize F for the first phase
        let mut F_vec: Vec<F> = unsafe_allocate_zero_vec(K);
        F_vec[0] = F::one();

        // Compute H (will be used in phase 2)
        let H: Vec<F> = preprocessing
            .map_trace_to_pc(trace)
            .map(|_pc| F::zero()) // Will be computed during phase 1
            .collect();
        let H = MultilinearPolynomial::from(H);
        let D = MultilinearPolynomial::from(D);

        // Precompute EQ(k_m, c) for k_m \in {0, 1} and c \in {0, 2, 3}
        let eq_km_c: [[F; 3]; 2] = [
            [
                F::one(),        // eq(0, 0) = 0 * 0 + (1 - 0) * (1 - 0)
                F::from_i64(-1), // eq(0, 2) = 0 * 2 + (1 - 0) * (1 - 2)
                F::from_i64(-2), // eq(0, 3) = 0 * 3 + (1 - 0) * (1 - 3)
            ],
            [
                F::zero(),     // eq(1, 0) = 1 * 0 + (1 - 1) * (1 - 0)
                F::from_u8(2), // eq(1, 2) = 1 * 2 + (1 - 1) * (1 - 2)
                F::from_u8(3), // eq(1, 3) = 1 * 3 + (1 - 1) * (1 - 3)
            ],
        ];

        // Precompute EQ(k_m, c)^2 for k_m \in {0, 1} and c \in {0, 2, 3}
        let eq_km_c_squared: [[F; 3]; 2] = [
            [F::one(), F::one(), F::from_u8(4)],
            [F::zero(), F::from_u8(4), F::from_u8(9)],
        ];

        Self {
            input_claim: F::zero(),
            K,
            T,
            prover_state: Some(BooleanityProverState {
                B,
                D,
                H,
                G,
                F: F_vec,
                eq_r_r: F::zero(), // Will be set after phase 1
                eq_km_c,
                eq_km_c_squared,
            }),
            verifier_state: None,
            ra_claim_prime: None,
            current_round: 0,
            preprocessing: Some(Arc::new(preprocessing.clone())),
            trace: Some(Arc::from(trace)),
        }
    }
}

impl<F: JoltField, ProofTranscript: Transcript> BatchableSumcheckInstance<F, ProofTranscript>
    for BooleanitySumcheck<F>
{
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

            // If transitioning to phase 2, prepare H
            if round == K_log - 1 {
                prover_state.eq_r_r = prover_state.B.final_sumcheck_claim();
                // Compute H using the final F values
                let preprocessing = self.preprocessing.as_ref().unwrap();
                let trace = self.trace.as_ref().unwrap();
                let H_vec: Vec<F> = preprocessing
                    .map_trace_to_pc(trace)
                    .map(|pc| prover_state.F[pc as usize])
                    .collect();
                prover_state.H = MultilinearPolynomial::from(H_vec);
            }
        } else {
            // Phase 2: Bind D and H
            rayon::join(
                || prover_state.D.bind_parallel(r_j, BindingOrder::LowToHigh),
                || prover_state.H.bind_parallel(r_j, BindingOrder::LowToHigh),
            );
        }

        self.current_round += 1;
    }

    fn cache_openings(&mut self) {
        debug_assert!(self.ra_claim_prime.is_none());
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");

        self.ra_claim_prime = Some(prover_state.H.final_sumcheck_claim());
    }
}

impl<F: JoltField> BooleanitySumcheck<F> {
    fn compute_phase1_message(&self, round: usize) -> Vec<F> {
        let prover_state = self.prover_state.as_ref().unwrap();
        let m = round + 1;
        const DEGREE: usize = 3;

        let univariate_poly_evals: [F; 3] = (0..prover_state.B.len() / 2)
            .into_par_iter()
            .map(|k_prime| {
                // Get B evaluations at points 0, 2, 3
                let B_evals =
                    prover_state
                        .B
                        .sumcheck_evals(k_prime, DEGREE, BindingOrder::LowToHigh);

                let inner_sum = prover_state.G[k_prime << m..(k_prime + 1) << m]
                    .par_iter()
                    .enumerate()
                    .map(|(k, &G_k)| {
                        // Since we're binding variables from low to high, k_m is the high bit
                        let k_m = k >> (m - 1);
                        // We then index into F using (k_{m-1}, ..., k_1)
                        let F_k = prover_state.F[k % (1 << (m - 1))];
                        // G_times_F := G[k] * F[k_1, ...., k_{m-1}]
                        let G_times_F = G_k * F_k;

                        // For c \in {0, 2, 3} compute:
                        //    G[k] * (F[k_1, ...., k_{m-1}, c]^2 - F[k_1, ...., k_{m-1}, c])
                        //    = G_times_F * (eq(k_m, c)^2 * F[k_1, ...., k_{m-1}] - eq(k_m, c))
                        [
                            G_times_F
                                * (prover_state.eq_km_c_squared[k_m][0] * F_k
                                    - prover_state.eq_km_c[k_m][0]),
                            G_times_F
                                * (prover_state.eq_km_c_squared[k_m][1] * F_k
                                    - prover_state.eq_km_c[k_m][1]),
                            G_times_F
                                * (prover_state.eq_km_c_squared[k_m][2] * F_k
                                    - prover_state.eq_km_c[k_m][2]),
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
                    B_evals[0] * inner_sum[0],
                    B_evals[1] * inner_sum[1],
                    B_evals[2] * inner_sum[2],
                ]
            })
            .reduce(
                || [F::zero(); 3],
                |mut running, new| {
                    for i in 0..3 {
                        running[i] += new[i];
                    }
                    running
                },
            );

        univariate_poly_evals.to_vec()
    }

    fn compute_phase2_message(&self, _round: usize) -> Vec<F> {
        let prover_state = self.prover_state.as_ref().unwrap();
        const DEGREE: usize = 3;

        let mut univariate_poly_evals: [F; 3] = (0..prover_state.D.len() / 2)
            .into_par_iter()
            .map(|i| {
                // Get D and H evaluations at points 0, 2, 3
                let D_evals = prover_state
                    .D
                    .sumcheck_evals(i, DEGREE, BindingOrder::LowToHigh);
                let H_evals = prover_state
                    .H
                    .sumcheck_evals(i, DEGREE, BindingOrder::LowToHigh);

                [
                    D_evals[0] * (H_evals[0].square() - H_evals[0]),
                    D_evals[1] * (H_evals[1].square() - H_evals[1]),
                    D_evals[2] * (H_evals[2].square() - H_evals[2]),
                ]
            })
            .reduce(
                || [F::zero(); 3],
                |mut running, new| {
                    for i in 0..3 {
                        running[i] += new[i];
                    }
                    running
                },
            );

        // Multiply by eq_r_r
        for eval in &mut univariate_poly_evals {
            *eval *= prover_state.eq_r_r;
        }

        univariate_poly_evals.to_vec()
    }
}

impl<F: JoltField, ProofTranscript: Transcript> BooleanityProof<F, ProofTranscript> {
    #[tracing::instrument(skip_all, name = "BooleanityProof::prove")]
    pub fn prove(
        preprocessing: &BytecodePreprocessing,
        trace: &[RV32IMCycle],
        r: &[F],
        D: Vec<F>,
        G: Vec<F>,
        transcript: &mut ProofTranscript,
    ) -> (Self, Vec<F>, Vec<F>) {
        let K = r.len().pow2();
        let T = trace.len();

        let mut booleanity_sumcheck = BooleanitySumcheck::new(preprocessing, trace, r, D, G, K, T);

        let (sumcheck_proof, r_combined) = booleanity_sumcheck.prove_single(transcript);

        let (r_address_prime, r_cycle_prime) = r_combined.split_at(K.log_2());

        let ra_claim_prime = booleanity_sumcheck
            .ra_claim_prime
            .expect("ra_claim_prime should be set after prove_single");

        let proof = Self {
            sumcheck_proof,
            ra_claim_prime,
        };

        (proof, r_address_prime.to_vec(), r_cycle_prime.to_vec())
    }
}

impl<F: JoltField> RafBytecode<F> {
    pub fn new(
        input_claim: F,
        ra_poly: MultilinearPolynomial<F>,
        ra_poly_shift: MultilinearPolynomial<F>,
        int_poly: IdentityPolynomial<F>,
        challenge: F,
        K: usize,
    ) -> Self {
        Self {
            input_claim,
            challenge,
            K,
            prover_state: Some(RafBytecodeProverState {
                ra_poly,
                ra_poly_shift,
                int_poly,
            }),
            ra_claims: None,
        }
    }
}

impl<F: JoltField, ProofTranscript: Transcript> BatchableSumcheckInstance<F, ProofTranscript>
    for RafBytecode<F>
{
    fn compute_prover_message(&mut self, _round: usize, _previous_claim: F) -> Vec<F> {
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");
        let degree = <Self as BatchableSumcheckVerifierInstance<F, ProofTranscript>>::degree(self);

        let univariate_poly_evals: [F; 2] = (0..prover_state.ra_poly.len() / 2)
            .into_par_iter()
            .map(|i| {
                let ra_evals =
                    prover_state
                        .ra_poly
                        .sumcheck_evals(i, degree, BindingOrder::LowToHigh);
                let ra_evals_shift =
                    prover_state
                        .ra_poly_shift
                        .sumcheck_evals(i, degree, BindingOrder::LowToHigh);
                let int_evals =
                    prover_state
                        .int_poly
                        .sumcheck_evals(i, degree, BindingOrder::LowToHigh);

                [
                    (ra_evals[0] + self.challenge * ra_evals_shift[0]) * int_evals[0],
                    (ra_evals[1] + self.challenge * ra_evals_shift[1]) * int_evals[1],
                ]
            })
            .reduce(
                || [F::zero(); 2],
                |mut running, new| {
                    for i in 0..2 {
                        running[i] += new[i];
                    }
                    running
                },
            );

        univariate_poly_evals.to_vec()
    }

    fn bind(&mut self, r_j: F, _round: usize) {
        let prover_state = self
            .prover_state
            .as_mut()
            .expect("Prover state not initialized");

        rayon::join(
            || {
                prover_state
                    .ra_poly
                    .bind_parallel(r_j, BindingOrder::LowToHigh)
            },
            || {
                rayon::join(
                    || {
                        prover_state
                            .ra_poly_shift
                            .bind_parallel(r_j, BindingOrder::LowToHigh)
                    },
                    || {
                        prover_state
                            .int_poly
                            .bind_parallel(r_j, BindingOrder::LowToHigh)
                    },
                )
            },
        );
    }

    fn cache_openings(&mut self) {
        debug_assert!(self.ra_claims.is_none());
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");

        let ra_claim = prover_state.ra_poly.final_sumcheck_claim();
        let ra_claim_shift = prover_state.ra_poly_shift.final_sumcheck_claim();

        self.ra_claims = Some((ra_claim, ra_claim_shift));
    }
}

impl<F: JoltField, ProofTranscript: Transcript> RafEvaluationProof<F, ProofTranscript> {
    #[allow(clippy::too_many_arguments)]
    #[tracing::instrument(skip_all, name = "RafEvaluationProof::prove")]
    pub fn prove(
        preprocessing: &BytecodePreprocessing,
        trace: &[RV32IMCycle],
        ra_poly: MultilinearPolynomial<F>,
        ra_poly_shift: MultilinearPolynomial<F>,
        r_cycle: &[F],
        r_shift: &[F],
        challenge: F,
        transcript: &mut ProofTranscript,
    ) -> Self {
        let K = preprocessing.bytecode.len().next_power_of_two();
        let int_poly = IdentityPolynomial::new(K.log_2());

        // TODO: Propagate raf claim from Spartan
        let raf_evals = preprocessing.map_trace_to_pc(trace).collect::<Vec<u64>>();
        let raf_poly = MultilinearPolynomial::from(raf_evals);
        let raf_claim = raf_poly.evaluate(r_cycle);
        let raf_claim_shift = raf_poly.evaluate(r_shift);
        let input_claim = raf_claim + challenge * raf_claim_shift;

        let mut raf_sumcheck =
            RafBytecode::new(input_claim, ra_poly, ra_poly_shift, int_poly, challenge, K);

        let (sumcheck_proof, _r_address) = raf_sumcheck.prove_single(transcript);

        let (ra_claim, ra_claim_shift) = raf_sumcheck
            .ra_claims
            .expect("ra_claims should be set after prove_single");

        Self {
            sumcheck_proof,
            ra_claim,
            ra_claim_shift,
            raf_claim,
            raf_claim_shift,
        }
    }
}
