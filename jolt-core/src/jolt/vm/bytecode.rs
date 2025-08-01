use std::collections::BTreeMap;
use std::sync::Arc;

use crate::{
    field::JoltField,
    jolt::{
        vm::{JoltCommitments, JoltProverPreprocessing},
        witness::CommittedPolynomials,
    },
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        compact_polynomial::SmallScalar,
        eq_poly::EqPolynomial,
        identity_poly::IdentityPolynomial,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        opening_proof::{ProverOpeningAccumulator, VerifierOpeningAccumulator},
    },
    subprotocols::sumcheck::{BatchableSumcheckInstance, SumcheckInstanceProof},
    utils::{
        errors::ProofVerifyError, math::Math, thread::unsafe_allocate_zero_vec,
        transcript::Transcript,
    },
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use common::constants::{BYTES_PER_INSTRUCTION, RAM_START_ADDRESS};
use rayon::prelude::*;
use tracer::instruction::{NormalizedInstruction, RV32IMCycle, RV32IMInstruction};

#[derive(Debug, Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct BytecodePreprocessing {
    pub code_size: usize,
    bytecode: Vec<RV32IMInstruction>,
    /// Maps the memory address of each instruction in the bytecode to its "virtual" address.
    /// See Section 6.1 of the Jolt paper, "Reflecting the program counter". The virtual address
    /// is the one used to keep track of the next (potentially virtual) instruction to execute.
    /// Key: (ELF address, virtual sequence index or 0)
    pub virtual_address_map: BTreeMap<(usize, usize), usize>,
}

impl BytecodePreprocessing {
    #[tracing::instrument(skip_all, name = "BytecodePreprocessing::preprocess")]
    pub fn preprocess(mut bytecode: Vec<RV32IMInstruction>) -> Self {
        println!("{:?}", bytecode);
        let mut virtual_address_map = BTreeMap::new();
        let mut virtual_address = 1; // Account for no-op instruction prepended to bytecode
        for instruction in bytecode.iter() {
            if instruction.normalize().address == 0 {
                // ignore unimplemented instructions
                continue;
            }
            let instr = instruction.normalize();
            debug_assert!(instr.address >= RAM_START_ADDRESS as usize);
            debug_assert!(instr.address.is_multiple_of(BYTES_PER_INSTRUCTION));
            assert_eq!(
                virtual_address_map.insert(
                    (instr.address, instr.virtual_sequence_remaining.unwrap_or(0)),
                    virtual_address
                ),
                None
            );
            virtual_address += 1;
        }

        // Bytecode: Prepend a single no-op instruction
        bytecode.insert(0, RV32IMInstruction::NoOp(0));
        assert_eq!(virtual_address_map.insert((0, 0), 0), None);

        // Bytecode: Pad to nearest power of 2
        // Get last address
        let last_address = bytecode.last().unwrap().normalize().address;
        let code_size = bytecode.len().next_power_of_two();
        let padding = code_size - bytecode.len();
        bytecode.extend((0..padding).map(|i| RV32IMInstruction::NoOp(last_address + 4 * (i + 1))));

        Self {
            code_size,
            bytecode,
            virtual_address_map,
        }
    }

    pub fn get_pc(&self, cycle: &RV32IMCycle, is_last: bool) -> usize {
        let instr = cycle.instruction().normalize();
        if matches!(cycle, tracer::instruction::RV32IMCycle::NoOp(_)) || is_last {
            return 0;
        }
        *self
            .virtual_address_map
            .get(&(instr.address, instr.virtual_sequence_remaining.unwrap_or(0)))
            .unwrap()
    }

    pub fn map_trace_to_pc<'a, 'b>(
        &'b self,
        trace: &'a [RV32IMCycle],
    ) -> impl rayon::iter::ParallelIterator<Item = u64> + use<'a, 'b> {
        let (_, init) = trace.split_last().unwrap();
        init.par_iter()
            .map(|cycle| self.get_pc(cycle, false) as u64)
            .chain(rayon::iter::once(0))
    }
}

#[tracing::instrument(skip_all)]
fn bytecode_to_val<F: JoltField>(bytecode: &[RV32IMInstruction], gamma: F) -> Vec<F> {
    let mut gamma_powers = vec![F::one()];
    for _ in 0..5 {
        gamma_powers.push(gamma * gamma_powers.last().unwrap());
    }

    bytecode
        .par_iter()
        .map(|instruction| {
            let NormalizedInstruction {
                address,
                operands,
                virtual_sequence_remaining: _,
            } = instruction.normalize();
            let mut linear_combination = F::zero();
            linear_combination += (address as u64).field_mul(gamma_powers[0]);
            linear_combination += (operands.rd as u64).field_mul(gamma_powers[1]);
            linear_combination += (operands.rs1 as u64).field_mul(gamma_powers[2]);
            linear_combination += (operands.rs2 as u64).field_mul(gamma_powers[3]);
            linear_combination += operands.imm.field_mul(gamma_powers[4]);
            // TODO(moodlezoup): Circuit and lookup flags
            linear_combination
        })
        .collect()
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone)]
pub struct BytecodeShoutProof<F: JoltField, ProofTranscript: Transcript> {
    core_piop_hamming: CorePIOPHammingProof<F, ProofTranscript>,
    booleanity: BooleanityProof<F, ProofTranscript>,
    raf_sumcheck: RafEvaluationProof<F, ProofTranscript>,
}

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

    pub fn verify<PCS: CommitmentScheme<ProofTranscript, Field = F>>(
        &self,
        preprocessing: &BytecodePreprocessing,
        commitments: &JoltCommitments<F, PCS, ProofTranscript>,
        T: usize,
        transcript: &mut ProofTranscript,
        opening_accumulator: &mut VerifierOpeningAccumulator<F, PCS, ProofTranscript>,
    ) -> Result<(), ProofVerifyError> {
        let K = preprocessing.bytecode.len();
        // TODO: this should come from Spartan
        let r_cycle: Vec<F> = transcript.challenge_vector(T.log_2());
        let _r_shift: Vec<F> = transcript.challenge_vector(T.log_2());
        let z: F = transcript.challenge_scalar();
        let gamma: F = transcript.challenge_scalar();

        // Used to combine the various fields in each instruction into a single
        // field element.
        let val: Vec<F> = bytecode_to_val(&preprocessing.bytecode, gamma);

        // Verify core PIOP and Hamming weight sumcheck
        let r_address = self.core_piop_hamming.verify(&val, z, K, transcript)?;

        let r_address_rev: Vec<_> = r_address.iter().copied().rev().collect();
        let r_cycle_rev: Vec<_> = r_cycle.iter().copied().rev().collect();

        let r_concat = [r_address_rev.as_slice(), r_cycle.as_slice()].concat();
        let ra_commitment = &commitments.commitments[CommittedPolynomials::BytecodeRa.to_index()];
        opening_accumulator.append(
            &[ra_commitment],
            r_concat,
            &[self.core_piop_hamming.ra_claim],
            transcript,
        );

        // Verify booleanity sumcheck
        let (r_booleanity, ra_claim_prime) =
            self.booleanity
                .verify(&r_address_rev, &r_cycle_rev, K, T, transcript)?;

        let (r_address_prime, r_cycle_prime) = r_booleanity.split_at(K.log_2());
        let r_address_prime = r_address_prime.iter().copied().rev().collect::<Vec<_>>();
        let r_cycle_prime = r_cycle_prime.iter().rev().copied().collect::<Vec<_>>();
        let r_concat = [r_address_prime.as_slice(), r_cycle_prime.as_slice()].concat();

        opening_accumulator.append(&[ra_commitment], r_concat, &[ra_claim_prime], transcript);

        let challenge: F = transcript.challenge_scalar();
        let _ = self.raf_sumcheck.verify(K, challenge, transcript)?;

        Ok(())
    }
}

struct CorePIOPHammingProverState<F: JoltField> {
    ra_poly: MultilinearPolynomial<F>,
    val_poly: MultilinearPolynomial<F>,
}

pub struct CorePIOPHammingSumcheck<F: JoltField> {
    /// Input claim: rv_claim + z
    input_claim: F,
    /// z value shared by prover and verifier
    z: F,
    /// K value shared by prover and verifier
    K: usize,
    /// Prover state
    prover_state: Option<CorePIOPHammingProverState<F>>,
    /// Cached ra claim after sumcheck completes
    ra_claim: Option<F>,
    /// Cached val evaluation after sumcheck completes
    val_eval: Option<F>,
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

    pub fn new_verifier(input_claim: F, z: F, K: usize) -> Self {
        Self {
            input_claim,
            z,
            K,
            prover_state: None,
            ra_claim: None,
            val_eval: None,
        }
    }
}

impl<F: JoltField, ProofTranscript: Transcript> BatchableSumcheckInstance<F, ProofTranscript>
    for CorePIOPHammingSumcheck<F>
{
    fn degree(&self) -> usize {
        2
    }

    fn num_rounds(&self) -> usize {
        self.K.log_2()
    }

    fn input_claim(&self) -> F {
        self.input_claim
    }

    fn compute_prover_message(&mut self, _round: usize, _previous_claim: F) -> Vec<F> {
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");
        let degree = <Self as BatchableSumcheckInstance<F, ProofTranscript>>::degree(self);

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

    fn expected_output_claim(&self, _r: &[F]) -> F {
        let ra_claim = self.ra_claim.as_ref().expect("ra_claim not set");
        let val_eval = self.val_eval.as_ref().expect("val_eval not set");

        // Verify sumcheck_claim = ra_claim * (z + val_eval)
        *ra_claim * (self.z + *val_eval)
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone)]
pub struct CorePIOPHammingProof<F: JoltField, ProofTranscript: Transcript> {
    sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    ra_claim: F,
    rv_claim: F,
    val_eval: F,
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

    pub fn verify(
        &self,
        _val: &[F],
        z: F,
        K: usize,
        transcript: &mut ProofTranscript,
    ) -> Result<Vec<F>, ProofVerifyError> {
        let input_claim = self.rv_claim + z;
        let mut core_piop_sumcheck = CorePIOPHammingSumcheck::new_verifier(input_claim, z, K);

        core_piop_sumcheck.ra_claim = Some(self.ra_claim);
        core_piop_sumcheck.val_eval = Some(self.val_eval);

        let r_address = core_piop_sumcheck.verify_single(&self.sumcheck_proof, transcript)?;

        Ok(r_address)
    }
}

struct BooleanityProverState<F: JoltField> {
    B: MultilinearPolynomial<F>,
    D: MultilinearPolynomial<F>,
    H: MultilinearPolynomial<F>,
    G: Vec<F>,
    F: Vec<F>,
    eq_r_r: F,
    // Precomputed arrays for phase 1
    eq_km_c: [[F; 3]; 2],
    eq_km_c_squared: [[F; 3]; 2],
}

struct BooleanityVerifierState<F: JoltField> {
    r_address: Option<Vec<F>>,
    r_cycle: Option<Vec<F>>,
}

pub struct BooleanitySumcheck<F: JoltField> {
    /// Input claim: always F::zero() for booleanity
    input_claim: F,
    /// K value shared by prover and verifier
    K: usize,
    /// T value shared by prover and verifier
    T: usize,
    /// Prover state
    prover_state: Option<BooleanityProverState<F>>,
    /// Verifier state
    verifier_state: Option<BooleanityVerifierState<F>>,
    /// Cached ra claim after sumcheck completes
    ra_claim_prime: Option<F>,
    /// Current round
    current_round: usize,
    /// Store preprocessing and trace for phase transition
    preprocessing: Option<Arc<BytecodePreprocessing>>,
    trace: Option<Arc<[RV32IMCycle]>>,
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

    pub fn new_verifier(
        K: usize,
        T: usize,
        r_address: Vec<F>,
        r_cycle: Vec<F>,
        ra_claim_prime: F,
    ) -> Self {
        Self {
            input_claim: F::zero(),
            K,
            T,
            prover_state: None,
            verifier_state: Some(BooleanityVerifierState::<F> {
                r_address: Some(r_address),
                r_cycle: Some(r_cycle),
            }),
            ra_claim_prime: Some(ra_claim_prime),
            current_round: 0,
            preprocessing: None,
            trace: None,
        }
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
        self.input_claim
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

    fn expected_output_claim(&self, r: &[F]) -> F {
        let ra_claim_prime = self.ra_claim_prime.expect("ra_claim_prime not set");
        let verifier_state = self
            .verifier_state
            .as_ref()
            .expect("Verifier state not initialized");

        // Split r into r_address_prime and r_cycle_prime
        let (r_address_prime, r_cycle_prime) = r.split_at(self.K.log_2());

        let r_address = verifier_state
            .r_address
            .as_ref()
            .expect("r_address not set");
        let r_cycle = verifier_state.r_cycle.as_ref().expect("r_cycle not set");

        let eq_eval_address = EqPolynomial::mle(r_address, r_address_prime);
        let eq_eval_cycle = EqPolynomial::mle(r_cycle, r_cycle_prime);

        eq_eval_address * eq_eval_cycle * (ra_claim_prime.square() - ra_claim_prime)
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

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone)]
pub struct BooleanityProof<F: JoltField, ProofTranscript: Transcript> {
    sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    ra_claim_prime: F,
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

    pub fn verify(
        &self,
        r_address: &[F],
        r_cycle: &[F],
        K: usize,
        T: usize,
        transcript: &mut ProofTranscript,
    ) -> Result<(Vec<F>, F), ProofVerifyError> {
        let booleanity_sumcheck = BooleanitySumcheck::new_verifier(
            K,
            T,
            r_address.to_vec(),
            r_cycle.to_vec(),
            self.ra_claim_prime,
        );

        let r_combined = booleanity_sumcheck.verify_single(&self.sumcheck_proof, transcript)?;

        Ok((r_combined, self.ra_claim_prime))
    }
}

struct RafBytecodeProverState<F: JoltField> {
    ra_poly: MultilinearPolynomial<F>,
    ra_poly_shift: MultilinearPolynomial<F>,
    int_poly: IdentityPolynomial<F>,
}

pub struct RafBytecode<F: JoltField> {
    /// Input claim: raf_claim + challenge * raf_claim_shift
    input_claim: F,
    /// Challenge value shared by prover and verifier
    challenge: F,
    /// K value shared by prover and verifier
    K: usize,
    /// Prover state
    prover_state: Option<RafBytecodeProverState<F>>,
    /// Cached ra claims after sumcheck completes
    ra_claims: Option<(F, F)>,
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

    pub fn new_verifier(input_claim: F, challenge: F, K: usize) -> Self {
        Self {
            input_claim,
            challenge,
            K,
            prover_state: None,
            ra_claims: None,
        }
    }
}

impl<F: JoltField, ProofTranscript: Transcript> BatchableSumcheckInstance<F, ProofTranscript>
    for RafBytecode<F>
{
    fn degree(&self) -> usize {
        2
    }

    fn num_rounds(&self) -> usize {
        self.K.log_2()
    }

    fn input_claim(&self) -> F {
        self.input_claim
    }

    fn compute_prover_message(&mut self, _round: usize, _previous_claim: F) -> Vec<F> {
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");
        let degree = <Self as BatchableSumcheckInstance<F, ProofTranscript>>::degree(self);

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

    fn expected_output_claim(&self, r: &[F]) -> F {
        let (ra_claim, ra_claim_shift) = self.ra_claims.as_ref().expect("ra_claims not set");

        let int_eval = IdentityPolynomial::new(self.K.log_2()).evaluate(r);

        // Verify sumcheck_claim = int(r) * (ra_claim + challenge * ra_claim_shift)
        int_eval * (*ra_claim + self.challenge * *ra_claim_shift)
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize, Debug, Clone)]
pub struct RafEvaluationProof<F: JoltField, ProofTranscript: Transcript> {
    sumcheck_proof: SumcheckInstanceProof<F, ProofTranscript>,
    ra_claim: F,
    ra_claim_shift: F,
    raf_claim: F,
    raf_claim_shift: F,
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

    pub fn verify(
        &self,
        K: usize,
        challenge: F,
        transcript: &mut ProofTranscript,
    ) -> Result<Vec<F>, ProofVerifyError> {
        let input_claim = self.raf_claim + challenge * self.raf_claim_shift;

        let mut raf_sumcheck = RafBytecode::new_verifier(input_claim, challenge, K);

        raf_sumcheck.ra_claims = Some((self.ra_claim, self.ra_claim_shift));

        let r_raf_sumcheck = raf_sumcheck.verify_single(&self.sumcheck_proof, transcript)?;

        Ok(r_raf_sumcheck)
    }
}
