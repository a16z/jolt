use std::collections::BTreeMap;

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
        unipoly::{CompressedUniPoly, UniPoly},
    },
    subprotocols::sumcheck::{BatchableSumcheckInstance, SumcheckInstanceProof},
    utils::{
        errors::ProofVerifyError,
        math::Math,
        thread::unsafe_allocate_zero_vec,
        transcript::{AppendToTranscript, Transcript},
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
        let mut virtual_address_map = BTreeMap::new();
        let mut virtual_address = 1; // Account for no-op instruction prepended to bytecode
        for instruction in bytecode.iter() {
            if instruction.normalize().address == 0 {
                // ignore unimplemented instructions
                continue;
            }
            let instr = instruction.normalize();
            debug_assert!(instr.address >= RAM_START_ADDRESS as usize);
            debug_assert!(instr.address % BYTES_PER_INSTRUCTION == 0);
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
    core_piop_sumcheck: SumcheckInstanceProof<F, ProofTranscript>,
    booleanity_sumcheck: SumcheckInstanceProof<F, ProofTranscript>,
    raf_sumcheck: RafEvaluationProof<F, ProofTranscript>,
    ra_claim: F,
    ra_claim_prime: F,
    rv_claim: F,
}

impl<F: JoltField, ProofTranscript: Transcript> BytecodeShoutProof<F, ProofTranscript> {
    #[tracing::instrument(skip_all, name = "BytecodeShoutProof::prove")]
    pub fn prove<PCS: CommitmentScheme<ProofTranscript, Field = F>>(
        preprocessing: &JoltProverPreprocessing<F, PCS, ProofTranscript>,
        trace: &[RV32IMCycle],
        opening_accumulator: &mut ProverOpeningAccumulator<F, PCS, ProofTranscript>,
        transcript: &mut ProofTranscript,
    ) -> Self {

        //// start of state gen (to be hanled by state manager)
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

        // Call the extracted function for core PIOP and Hamming weight sumcheck
        let (core_piop_sumcheck_proof, r_address, ra_claim, raf_ra) = 
            prove_core_piop_hamming(F.clone(), val, z, rv_claim, K, transcript);

        let unbound_ra_poly =
            CommittedPolynomials::BytecodeRa.generate_witness(preprocessing, trace);

        let r_address_rev = r_address.iter().copied().rev().collect::<Vec<_>>();
        opening_accumulator.append_sparse(
            vec![unbound_ra_poly.clone()],
            r_address_rev,
            r_cycle.clone(),
            vec![ra_claim],
        );

        let (booleanity_sumcheck_proof, r_address_prime, r_cycle_prime, ra_claim_prime) =
            prove_booleanity(bytecode_preprocessing, trace, &r_address, E, F, transcript);

        let r_address_prime = r_address_prime.iter().copied().rev().collect::<Vec<_>>();
        let r_cycle_prime = r_cycle_prime.iter().rev().copied().collect::<Vec<_>>();

        // TODO: Reduce 2 ra claims to 1 (Section 4.5.2 of Proofs, Arguments, and Zero-Knowledge)
        opening_accumulator.append_sparse(
            vec![unbound_ra_poly],
            r_address_prime,
            r_cycle_prime,
            vec![ra_claim_prime],
        );

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
            core_piop_sumcheck: core_piop_sumcheck_proof,
            booleanity_sumcheck: booleanity_sumcheck_proof,
            ra_claim,
            ra_claim_prime,
            rv_claim,
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

        let (sumcheck_claim, r_address) =
            self.core_piop_sumcheck
                .verify(self.rv_claim + z, K.log_2(), 2, transcript)?;

        let r_address_rev: Vec<_> = r_address.iter().copied().rev().collect();
        let r_cycle_rev: Vec<_> = r_cycle.iter().copied().rev().collect();

        // Used to combine the various fields in each instruction into a single
        // field element.
        let val: Vec<F> = bytecode_to_val(&preprocessing.bytecode, gamma);
        let val = MultilinearPolynomial::from(val);

        assert_eq!(
            self.ra_claim * (z + val.evaluate(&r_address_rev)),
            sumcheck_claim,
            "Core PIOP + Hamming weight sumcheck failed"
        );

        let r_concat = [r_address_rev.as_slice(), r_cycle.as_slice()].concat();
        let ra_commitment = &commitments.commitments[CommittedPolynomials::BytecodeRa.to_index()];
        opening_accumulator.append(&[ra_commitment], r_concat, &[self.ra_claim], transcript);

        let (sumcheck_claim, r_booleanity) =
            self.booleanity_sumcheck
                .verify(F::zero(), K.log_2() + T.log_2(), 3, transcript)?;
        let (r_address_prime, r_cycle_prime) = r_booleanity.split_at(K.log_2());
        let eq_eval_address = EqPolynomial::mle(&r_address_rev, r_address_prime);
        let eq_eval_cycle = EqPolynomial::mle(&r_cycle_rev, r_cycle_prime);

        assert_eq!(
            eq_eval_address * eq_eval_cycle * (self.ra_claim_prime.square() - self.ra_claim_prime),
            sumcheck_claim,
            "Booleanity sumcheck failed"
        );

        let r_address_prime = r_address_prime.iter().copied().rev().collect::<Vec<_>>();
        let r_cycle_prime = r_cycle_prime.iter().rev().copied().collect::<Vec<_>>();
        let r_concat = [r_address_prime.as_slice(), r_cycle_prime.as_slice()].concat();
        opening_accumulator.append(
            &[ra_commitment],
            r_concat,
            &[self.ra_claim_prime],
            transcript,
        );

        let challenge: F = transcript.challenge_scalar();
        let _ = self.raf_sumcheck.verify(K, challenge, transcript)?;

        Ok(())
    }
}

/// Implements the sumcheck prover for the Booleanity check in step 3 of
/// Figure 6 in the Twist+Shout paper. The efficient implementation of this
/// sumcheck is described in Section 6.3.
#[tracing::instrument(skip_all, name = "Shout booleanity sumcheck")]
pub fn prove_booleanity<F: JoltField, ProofTranscript: Transcript>(
    preprocessing: &BytecodePreprocessing,
    trace: &[RV32IMCycle],
    r: &[F],
    D: Vec<F>,
    G: Vec<F>,
    transcript: &mut ProofTranscript,
) -> (SumcheckInstanceProof<F, ProofTranscript>, Vec<F>, Vec<F>, F) {
    const DEGREE: usize = 3;
    let K = r.len().pow2();
    let T = trace.len();
    debug_assert_eq!(D.len(), T);
    debug_assert_eq!(G.len(), K);

    let mut B = MultilinearPolynomial::from(EqPolynomial::evals(r)); // (53)

    // First log(K) rounds of sumcheck

    let mut F: Vec<F> = unsafe_allocate_zero_vec(K);
    F[0] = F::one();

    let num_rounds = K.log_2() + T.log_2();
    let mut r_address_prime: Vec<F> = Vec::with_capacity(K.log_2());
    let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::with_capacity(num_rounds);

    let mut previous_claim = F::zero();

    // EQ(k_m, c) for k_m \in {0, 1} and c \in {0, 2, 3}
    let eq_km_c: [[F; DEGREE]; 2] = [
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
    // EQ(k_m, c)^2 for k_m \in {0, 1} and c \in {0, 2, 3}
    let eq_km_c_squared: [[F; DEGREE]; 2] = [
        [F::one(), F::one(), F::from_u8(4)],
        [F::zero(), F::from_u8(4), F::from_u8(9)],
    ];

    // First log(K) rounds of sumcheck
    let span = tracing::span!(
        tracing::Level::INFO,
        "First log(K) rounds of Booleanity sumcheck"
    );
    let _guard = span.enter();

    for round in 0..K.log_2() {
        let m = round + 1;

        let inner_span = tracing::span!(tracing::Level::INFO, "Compute univariate poly");
        let _inner_guard = inner_span.enter();

        let univariate_poly_evals: [F; 3] = (0..B.len() / 2)
            .into_par_iter()
            .map(|k_prime| {
                let B_evals = B.sumcheck_evals(k_prime, DEGREE, BindingOrder::LowToHigh);
                let inner_sum = G[k_prime << m..(k_prime + 1) << m]
                    .par_iter()
                    .enumerate()
                    .map(|(k, &G_k)| {
                        // Since we're binding variables from low to high, k_m is the high bit
                        let k_m = k >> (m - 1);
                        // We then index into F using (k_{m-1}, ..., k_1)
                        let F_k = F[k % (1 << (m - 1))];
                        // G_times_F := G[k] * F[k_1, ...., k_{m-1}]
                        let G_times_F = G_k * F_k;
                        // For c \in {0, 2, 3} compute:
                        //    G[k] * (F[k_1, ...., k_{m-1}, c]^2 - F[k_1, ...., k_{m-1}, c])
                        //    = G_times_F * (eq(k_m, c)^2 * F[k_1, ...., k_{m-1}] - eq(k_m, c))
                        [
                            G_times_F * (eq_km_c_squared[k_m][0] * F_k - eq_km_c[k_m][0]),
                            G_times_F * (eq_km_c_squared[k_m][1] * F_k - eq_km_c[k_m][1]),
                            G_times_F * (eq_km_c_squared[k_m][2] * F_k - eq_km_c[k_m][2]),
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
        r_address_prime.push(r_j);

        previous_claim = univariate_poly.evaluate(&r_j);

        B.bind_parallel(r_j, BindingOrder::LowToHigh);

        let inner_span = tracing::span!(tracing::Level::INFO, "Update F");
        let _inner_guard = inner_span.enter();

        // Update F for this round (see Equation 55)
        let (F_left, F_right) = F.split_at_mut(1 << round);
        F_left
            .par_iter_mut()
            .zip(F_right.par_iter_mut())
            .for_each(|(x, y)| {
                *y = *x * r_j;
                *x -= *y;
            });
    }

    drop(_guard);
    drop(span);

    let span = tracing::span!(
        tracing::Level::INFO,
        "Last log(T) rounds of Booleanity sumcheck"
    );
    let _guard = span.enter();

    let eq_r_r = B.final_sumcheck_claim();
    let H: Vec<F> = preprocessing
        .map_trace_to_pc(trace)
        .map(|pc| F[pc as usize])
        .collect();
    let mut H = MultilinearPolynomial::from(H);
    let mut D = MultilinearPolynomial::from(D);
    let mut r_cycle_prime: Vec<F> = Vec::with_capacity(T.log_2());

    // TODO(moodlezoup): Implement optimization from Section 6.2.2 "An optimization leveraging small memory size"
    // Last log(T) rounds of sumcheck
    for _round in 0..T.log_2() {
        #[cfg(test)]
        {
            let expected: F = eq_r_r
                * (0..H.len())
                    .map(|j| {
                        let D_j = D.get_bound_coeff(j);
                        let H_j = H.get_bound_coeff(j);
                        D_j * (H_j.square() - H_j)
                    })
                    .sum::<F>();
            assert_eq!(
                expected, previous_claim,
                "Sumcheck sanity check failed in round {_round}"
            );
        }

        let inner_span = tracing::span!(tracing::Level::INFO, "Compute univariate poly");
        let _inner_guard = inner_span.enter();

        let mut univariate_poly_evals: [F; 3] = (0..D.len() / 2)
            .into_par_iter()
            .map(|i| {
                let D_evals = D.sumcheck_evals(i, DEGREE, BindingOrder::LowToHigh);
                let H_evals = H.sumcheck_evals(i, DEGREE, BindingOrder::LowToHigh);

                [
                    D_evals[0] * (H_evals[0].square() - H_evals[0]),
                    D_evals[1] * (H_evals[1].square() - H_evals[1]),
                    D_evals[2] * (H_evals[2].square() - H_evals[2]),
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

        univariate_poly_evals = [
            eq_r_r * univariate_poly_evals[0],
            eq_r_r * univariate_poly_evals[1],
            eq_r_r * univariate_poly_evals[2],
        ];

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
        r_cycle_prime.push(r_j);

        previous_claim = univariate_poly.evaluate(&r_j);

        // Bind polynomials
        rayon::join(
            || D.bind_parallel(r_j, BindingOrder::LowToHigh),
            || H.bind_parallel(r_j, BindingOrder::LowToHigh),
        );
    }

    let ra_claim = H.final_sumcheck_claim();
    (
        SumcheckInstanceProof::new(compressed_polys),
        r_address_prime,
        r_cycle_prime,
        ra_claim,
    )
}

/// Implements the sumcheck prover for the combined core PIOP and Hamming weight check.
/// This combines the core PIOP sumcheck (proving that the bytecode trace matches the program)
/// with the Hamming weight sumcheck (which always equals 1).
/// Returns the sumcheck proof, r_address vector, ra_claim, and the unbound ra polynomial.
#[tracing::instrument(skip_all, name = "Core PIOP + Hamming weight sumcheck")]
fn prove_core_piop_hamming<F: JoltField, ProofTranscript: Transcript>(
    F: Vec<F>,
    val: Vec<F>,
    z: F,
    rv_claim: F,
    K: usize,
    transcript: &mut ProofTranscript,
) -> (
    SumcheckInstanceProof<F, ProofTranscript>,
    Vec<F>,
    F,
    MultilinearPolynomial<F>,
) {
    const DEGREE: usize = 2;
    let num_rounds = K.log_2();
    let mut r_address: Vec<F> = Vec::with_capacity(num_rounds);
    
    // Linear combination of the core PIOP claim and the Hamming weight claim (which is 1)
    let mut previous_claim = rv_claim + z;
    
    let mut ra = MultilinearPolynomial::from(F);
    let raf_ra = ra.clone(); // Clone before binding for RAF sumcheck
    let mut val = MultilinearPolynomial::from(val);
    
    // Prove the core PIOP and Hamming weight sumchecks in parallel
    let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::with_capacity(num_rounds);
    for _ in 0..num_rounds {
        let inner_span = tracing::span!(tracing::Level::INFO, "Compute univariate poly");
        let _inner_guard = inner_span.enter();
        
        let univariate_poly_evals: [F; 2] = (0..ra.len() / 2)
            .into_par_iter()
            .map(|i| {
                let ra_evals = ra.sumcheck_evals(i, DEGREE, BindingOrder::LowToHigh);
                let val_evals = val.sumcheck_evals(i, DEGREE, BindingOrder::LowToHigh);
                
                [
                    ra_evals[0] * (z + val_evals[0]),
                    ra_evals[1] * (z + val_evals[1]),
                ]
            })
            .reduce(
                || [F::zero(); 2],
                |running, new| [running[0] + new[0], running[1] + new[1]],
            );
        
        let univariate_poly = UniPoly::from_evals(&[
            univariate_poly_evals[0],
            previous_claim - univariate_poly_evals[0],
            univariate_poly_evals[1],
        ]);
        
        drop(_inner_guard);
        drop(inner_span);
        
        let compressed_poly = univariate_poly.compress();
        compressed_poly.append_to_transcript(transcript);
        compressed_polys.push(compressed_poly);
        
        let r_j = transcript.challenge_scalar::<F>();
        r_address.push(r_j);
        
        previous_claim = univariate_poly.evaluate(&r_j);
        
        // Bind polynomials
        rayon::join(
            || ra.bind_parallel(r_j, BindingOrder::LowToHigh),
            || val.bind_parallel(r_j, BindingOrder::LowToHigh),
        );
    }
    
    let ra_claim = ra.final_sumcheck_claim();
    let sumcheck_proof = SumcheckInstanceProof::new(compressed_polys);
    
    (sumcheck_proof, r_address, ra_claim, raf_ra)
}

struct RafBytecodeProverState<F: JoltField> {
    ra_poly: MultilinearPolynomial<F>,
    ra_poly_shift: MultilinearPolynomial<F>,
    int_poly: IdentityPolynomial<F>,
    challenge: F,
    K: usize,
}

struct RafBytecodeVerifierState<F: JoltField> {
    challenge: F,
    K: usize,
}

pub struct RafBytecode<F: JoltField> {
    /// Input claim: raf_claim + challenge * raf_claim_shift
    input_claim: F,
    /// Prover state (optional for prover)
    prover_state: Option<RafBytecodeProverState<F>>,
    /// Verifier state (optional for verifier)
    verifier_state: Option<RafBytecodeVerifierState<F>>,
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
            prover_state: Some(RafBytecodeProverState {
                ra_poly,
                ra_poly_shift,
                int_poly,
                challenge,
                K,
            }),
            verifier_state: None,
            ra_claims: None,
        }
    }

    pub fn new_verifier(input_claim: F, challenge: F, K: usize) -> Self {
        Self {
            input_claim,
            prover_state: None,
            verifier_state: Some(RafBytecodeVerifierState { challenge, K }),
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
        if self.prover_state.is_some() {
            self.prover_state.as_ref().unwrap().K.log_2()
        } else if self.verifier_state.is_some() {
            self.verifier_state.as_ref().unwrap().K.log_2()
        } else {
            panic!("Neither prover state nor verifier state is initialized");
        }
    }

    fn input_claim(&self) -> F {
        self.input_claim
    }

    fn compute_prover_message(&self, _round: usize) -> Vec<F> {
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");
        let degree = <Self as BatchableSumcheckInstance<F, ProofTranscript>>::degree(self);
        
        // Compute univariate polynomial evaluations for degree-2 sumcheck
        let univariate_poly_evals: Vec<F> = (0..prover_state.ra_poly.len() / 2)
            .into_par_iter()
            .map(|i| {
                let ra_evals = prover_state
                    .ra_poly
                    .sumcheck_evals(i, degree, BindingOrder::LowToHigh);
                let ra_evals_shift = prover_state
                    .ra_poly_shift
                    .sumcheck_evals(i, degree, BindingOrder::LowToHigh);
                let int_evals = prover_state
                    .int_poly
                    .sumcheck_evals(i, degree, BindingOrder::LowToHigh);

                // Compute the product evaluations at 0 and 2
                vec![
                    (ra_evals[0] + prover_state.challenge * ra_evals_shift[0]) * int_evals[0],
                    (ra_evals[1] + prover_state.challenge * ra_evals_shift[1]) * int_evals[1],
                ]
            })
            .reduce(
                || vec![F::zero(); 2],
                |mut running, new| {
                    for i in 0..2 {
                        running[i] += new[i];
                    }
                    running
                },
            );

        univariate_poly_evals
    }

    fn bind(&mut self, r_j: F, _round: usize) {
        let prover_state = self
            .prover_state
            .as_mut()
            .expect("Prover state not initialized");

        rayon::join(
            || prover_state.ra_poly.bind_parallel(r_j, BindingOrder::LowToHigh),
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
        let verifier_state = self
            .verifier_state
            .as_ref()
            .expect("Verifier state not initialized");
        let (ra_claim, ra_claim_shift) = self.ra_claims.as_ref().expect("ra_claims not set");

        let int_eval = IdentityPolynomial::new(verifier_state.K.log_2()).evaluate(r);
        
        // Verify sumcheck_claim = int(r) * (ra_claim + challenge * ra_claim_shift)
        int_eval * (*ra_claim + verifier_state.challenge * *ra_claim_shift)
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

        // Create RafBytecode sumcheck instance
        let mut raf_sumcheck = RafBytecode::new(
            input_claim,
            ra_poly,
            ra_poly_shift,
            int_poly,
            challenge,
            K,
        );

        // Run the sumcheck protocol
        let (sumcheck_proof, _r_address) = raf_sumcheck.prove_single(transcript);

        // Get the cached claims
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
        
        // Create RafBytecode verifier instance
        let mut raf_sumcheck = RafBytecode::new_verifier(input_claim, challenge, K);
        
        // Set the cached claims for verification
        raf_sumcheck.ra_claims = Some((self.ra_claim, self.ra_claim_shift));
        
        // Verify the sumcheck proof
        let r_raf_sumcheck = raf_sumcheck.verify_single(&self.sumcheck_proof, transcript)?;
        
        Ok(r_raf_sumcheck)
    }
}
