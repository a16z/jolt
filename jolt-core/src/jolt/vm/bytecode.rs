use std::collections::BTreeMap;

use crate::{
    field::JoltField,
    poly::{
        compact_polynomial::SmallScalar,
        eq_poly::EqPolynomial,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        unipoly::{CompressedUniPoly, UniPoly},
    },
    subprotocols::sumcheck::SumcheckInstanceProof,
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

#[derive(Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct BytecodePreprocessing {
    bytecode: Vec<RV32IMInstruction>,
    /// Maps the memory address of each instruction in the bytecode to its "virtual" address.
    /// See Section 6.1 of the Jolt paper, "Reflecting the program counter". The virtual address
    /// is the one used to keep track of the next (potentially virtual) instruction to execute.
    /// Key: (ELF address, virtual sequence index or 0)
    virtual_address_map: BTreeMap<(usize, usize), usize>,
}

impl BytecodePreprocessing {
    #[tracing::instrument(skip_all, name = "BytecodePreprocessing::preprocess")]
    pub fn preprocess(mut bytecode: Vec<RV32IMInstruction>) -> Self {
        let mut virtual_address_map = BTreeMap::new();
        let mut virtual_address = 1; // Account for no-op instruction prepended to bytecode
        for instruction in bytecode.iter() {
            let instr = instruction.normalize();
            debug_assert!(instr.address >= (RAM_START_ADDRESS as usize));
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

    pub fn map_trace_to_pc_streaming<'a, 'b>(
        &'b self,
        trace: &'a [RV32IMCycle],
    ) -> impl rayon::iter::ParallelIterator<Item = u64> + use<'a, 'b> {
        trace
            .par_iter()
            .map(|cycle| self.get_pc(cycle, false) as u64)
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

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct BytecodeShoutProof<F: JoltField, ProofTranscript: Transcript> {
    core_piop_sumcheck: SumcheckInstanceProof<F, ProofTranscript>,
    booleanity_sumcheck: SumcheckInstanceProof<F, ProofTranscript>,
    ra_claim: F,
    ra_claim_prime: F,
    rv_claim: F,
}

impl<F: JoltField, ProofTranscript: Transcript> BytecodeShoutProof<F, ProofTranscript> {
    #[tracing::instrument(skip_all, name = "BytecodeShoutProof::prove")]
    pub fn prove(
        preprocessing: &BytecodePreprocessing,
        trace: &[RV32IMCycle],
        transcript: &mut ProofTranscript,
    ) -> Self {
        let K = preprocessing.bytecode.len().next_power_of_two();
        let T = trace.len();
        let r_cycle: Vec<F> = transcript.challenge_vector(T.log_2());
        // Used to batch the core PIOP sumcheck and Hamming weight sumcheck
        // (see Section 4.2.1)
        let z: F = transcript.challenge_scalar();

        let num_rounds = K.log_2();
        let mut r_address: Vec<F> = Vec::with_capacity(num_rounds);

        let E: Vec<F> = EqPolynomial::evals(&r_cycle);

        let span = tracing::span!(tracing::Level::INFO, "compute F");
        let _guard = span.enter();

        let num_chunks = rayon::current_num_threads()
            .next_power_of_two()
            .min(trace.len());
        let chunk_size = (trace.len() / num_chunks).max(1);
        let F: Vec<_> = trace
            .par_chunks(chunk_size)
            .enumerate()
            .map(|(chunk_index, trace_chunk)| {
                let mut result: Vec<F> = unsafe_allocate_zero_vec(K);
                let mut j = chunk_index * chunk_size;
                for cycle in trace_chunk {
                    let k = preprocessing.get_pc(cycle, j == trace.len() - 1);
                    result[k] += E[j];
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
                        .for_each(|(x, y)| {
                            *x += y;
                        });
                    running
                },
            );
        drop(_guard);
        drop(span);

        // Used to combine the various fields in each instruction into a single
        // field element.
        let gamma: F = transcript.challenge_scalar();
        let val: Vec<F> = bytecode_to_val(&preprocessing.bytecode, gamma);

        let rv_claim: F = F
            .par_iter()
            .zip(val.par_iter())
            .map(|(&ra, &val)| ra * val)
            .sum();
        // Linear combination of the core PIOP claim and the Hamming weight claim (which is 1)
        let mut previous_claim = rv_claim + z;

        let mut ra = MultilinearPolynomial::from(F.clone());
        let mut val = MultilinearPolynomial::from(val);

        const DEGREE: usize = 2;

        let span = tracing::span!(tracing::Level::INFO, "core PIOP + Hamming weight sumcheck");
        let _guard = span.enter();

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

        drop(_guard);
        drop(span);

        let ra_claim = ra.final_sumcheck_claim();

        let core_piop_sumcheck_proof = SumcheckInstanceProof::new(compressed_polys);

        let (booleanity_sumcheck_proof, _r_address_prime, _r_cycle_prime, ra_claim_prime) =
            prove_booleanity(preprocessing, trace, &r_address, E, F, transcript);

        // TODO: Reduce 2 ra claims to 1 (Section 4.5.2 of Proofs, Arguments, and Zero-Knowledge)
        // TODO: Append to opening proof accumulator

        Self {
            core_piop_sumcheck: core_piop_sumcheck_proof,
            booleanity_sumcheck: booleanity_sumcheck_proof,
            ra_claim,
            ra_claim_prime,
            rv_claim,
        }
    }

    pub fn verify(
        &self,
        preprocessing: &BytecodePreprocessing,
        T: usize,
        transcript: &mut ProofTranscript,
    ) -> Result<(), ProofVerifyError> {
        let K = preprocessing.bytecode.len();
        let r_cycle: Vec<F> = transcript.challenge_vector(T.log_2());
        let z: F = transcript.challenge_scalar();
        let gamma: F = transcript.challenge_scalar();

        let (sumcheck_claim, mut r_address) =
            self.core_piop_sumcheck
                .verify(self.rv_claim + z, K.log_2(), 2, transcript)?;
        r_address = r_address.into_iter().rev().collect();

        // Used to combine the various fields in each instruction into a single
        // field element.
        let val: Vec<F> = bytecode_to_val(&preprocessing.bytecode, gamma);
        let val = MultilinearPolynomial::from(val);

        assert_eq!(
            self.ra_claim * (z + val.evaluate(&r_address)),
            sumcheck_claim,
            "Core PIOP + Hamming weight sumcheck failed"
        );

        let (sumcheck_claim, r_booleanity) =
            self.booleanity_sumcheck
                .verify(F::zero(), K.log_2() + T.log_2(), 3, transcript)?;
        let (r_address_prime, r_cycle_prime) = r_booleanity.split_at(K.log_2());
        let eq_eval_address = EqPolynomial::new(r_address).evaluate(r_address_prime);
        let r_cycle: Vec<_> = r_cycle.iter().copied().rev().collect();
        let eq_eval_cycle = EqPolynomial::new(r_cycle).evaluate(r_cycle_prime);

        assert_eq!(
            eq_eval_address * eq_eval_cycle * (self.ra_claim_prime.square() - self.ra_claim_prime),
            sumcheck_claim,
            "Booleanity sumcheck failed"
        );

        // TODO: Reduce 2 ra claims to 1 (Section 4.5.2 of Proofs, Arguments, and Zero-Knowledge)
        // TODO: Append to opening proof accumulator

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
