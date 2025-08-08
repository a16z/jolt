//! Implements the fetch-decode portion of the zkVM.

use std::collections::BTreeMap;

use itertools::Itertools;
use jolt_core::{
    field::JoltField,
    poly::{
        compact_polynomial::SmallScalar,
        eq_poly::EqPolynomial,
        identity_poly::IdentityPolynomial,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        unipoly::{CompressedUniPoly, UniPoly},
    },
    subprotocols::sumcheck::SumcheckInstanceProof,
    utils::{
        errors::ProofVerifyError,
        math::Math,
        transcript::{AppendToTranscript, Transcript},
    },
};
use onnx_tracer::trace_types::ONNXInstr;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::jolt::execution_trace::JoltONNXCycle;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BytecodePreprocessing {
    pub code_size: usize,
    bytecode: Vec<ONNXInstr>,
    /// Maps the memory address of each instruction in the bytecode to its "virtual" address.
    /// See Section 6.1 of the Jolt paper, "Reflecting the program counter". The virtual address
    /// is the one used to keep track of the next (potentially virtual) instruction to execute.
    /// Key: (ELF address, virtual sequence index or 0)
    pub virtual_address_map: BTreeMap<(usize, usize), usize>,
}

impl BytecodePreprocessing {
    #[tracing::instrument(skip_all, name = "BytecodePreprocessing::preprocess")]
    pub fn preprocess(mut bytecode: Vec<ONNXInstr>) -> Self {
        let mut virtual_address_map = BTreeMap::new();
        let mut virtual_address = 1; // Account for no-op instruction prepended to bytecode
        for instruction in bytecode.iter_mut() {
            assert_eq!(
                virtual_address_map.insert(
                    (
                        instruction.address,
                        instruction.virtual_sequence_remaining.unwrap_or(0)
                    ),
                    virtual_address
                ),
                None
            );
            virtual_address += 1;
        }
        // Bytecode: Prepend a single no-op instruction
        bytecode.insert(0, ONNXInstr::no_op());

        // Bytecode: Pad to nearest power of 2
        let code_size = bytecode.len().next_power_of_two();
        bytecode.resize(code_size, ONNXInstr::no_op());
        Self {
            code_size,
            bytecode,
            virtual_address_map,
        }
    }
}

pub struct BytecodeProof<F, ProofTranscript>
where
    ProofTranscript: Transcript,
    F: JoltField,
{
    core_piop_sumcheck: SumcheckInstanceProof<F, ProofTranscript>,
    booleanity_sumcheck: SumcheckInstanceProof<F, ProofTranscript>,
    ra_claim: F,
    ra_claim_prime: F,
    rv_claim: F,
}

impl<F, ProofTranscript> BytecodeProof<F, ProofTranscript>
where
    ProofTranscript: Transcript,
    F: JoltField,
{
    pub fn prove(
        preprocessing: &BytecodePreprocessing,
        trace: &[JoltONNXCycle],
        transcript: &mut ProofTranscript,
    ) -> Self {
        let K = preprocessing.code_size;
        let T = trace.len();

        // --- Shout PIOP & Hamming weight check ---
        let r_cycle: Vec<F> = transcript.challenge_vector(T.log_2());
        // Used to batch the core PIOP sumcheck and Hamming weight sumcheck
        // (see Section 4.2.1)
        let z: F = transcript.challenge_scalar();
        let E = EqPolynomial::evals(&r_cycle);
        let mut F = vec![F::zero(); K];
        // Iterate through bytecode trace.
        for (j, cycle) in trace.iter().enumerate() {
            let k = cycle.instr.address;
            F[k] += E[j]
        }
        let gamma: F = transcript.challenge_scalar();
        let val = Self::bytecode_to_val(&preprocessing.bytecode, &gamma);
        // sum-check setup
        let rv_claim: F = F.iter().zip_eq(val.iter()).map(|(f, v)| *f * v).sum();
        // random linear combination of core piop claim and hamming weight claim
        let mut prev_claim = rv_claim + z/* .mul(1) */; // where 1 is hamming weight sumcheck claim
        let mut ra = MultilinearPolynomial::from(F.clone());
        let mut val = MultilinearPolynomial::from(val);
        const DEGREE: usize = 2;
        let num_rounds = K.log_2();
        let mut r_address: Vec<F> = Vec::with_capacity(num_rounds);
        let mut sumcheck_proof: Vec<CompressedUniPoly<F>> = Vec::with_capacity(num_rounds);
        for _ in 0..num_rounds {
            let uni_poly_evals: [F; 2] = (0..ra.len() / 2)
                .into_par_iter()
                .map(|index| {
                    let ra_evals = ra.sumcheck_evals(index, DEGREE, BindingOrder::LowToHigh);
                    let val_evals = val.sumcheck_evals(index, DEGREE, BindingOrder::LowToHigh);
                    [
                        ra_evals[0] * (z + val_evals[0]),
                        ra_evals[1] * (z + val_evals[1]),
                    ]
                })
                .reduce(
                    || [F::zero(); 2],
                    |acc, new| [acc[0] + new[0], acc[1] + new[1]],
                );
            let uni_poly = UniPoly::from_evals(&[
                uni_poly_evals[0],
                prev_claim - uni_poly_evals[0],
                uni_poly_evals[1],
            ]);
            let compressed_poly = uni_poly.compress();
            compressed_poly.append_to_transcript(transcript);
            sumcheck_proof.push(compressed_poly);
            let r: F = transcript.challenge_scalar();
            ra.bind_parallel(r, BindingOrder::LowToHigh);
            val.bind_parallel(r, BindingOrder::LowToHigh);
            prev_claim = uni_poly.evaluate(&r);
            r_address.push(r);
        }
        let ra_claim = ra.final_sumcheck_claim();
        let core_piop_shout_proof: SumcheckInstanceProof<F, ProofTranscript> =
            SumcheckInstanceProof::new(sumcheck_proof);
        // --- Booleanity check ---
        let (booleanity_sumcheck_proof, _r_address_prime, _r_cycle_prime, ra_claim_prime) =
            prove_booleanity(trace, &r_address, E, F, transcript);

        // --- raf evaluation ---
        BytecodeProof {
            core_piop_sumcheck: core_piop_shout_proof,
            booleanity_sumcheck: booleanity_sumcheck_proof,
            ra_claim,
            ra_claim_prime,
            rv_claim,
        }
    }

    /// Reed-solomon fingerprint each instr in the program bytecode
    fn bytecode_to_val(program_bytecode: &[ONNXInstr], gamma: &F) -> Vec<F> {
        let mut gamma_pows = [F::one(); 4];
        for i in 1..4 {
            gamma_pows[i] *= *gamma * gamma_pows[i - 1];
        }
        program_bytecode
            .iter()
            .map(|instr| {
                let mut linear_combination = F::zero();
                linear_combination += instr.opcode.clone().into_bitflag().field_mul(gamma_pows[0]);
                linear_combination += (instr.address as u64).field_mul(gamma_pows[1]);
                linear_combination +=
                    (instr.ts1.unwrap_or_default() as u64).field_mul(gamma_pows[2]);
                linear_combination +=
                    (instr.ts2.unwrap_or_default() as u64).field_mul(gamma_pows[3]);
                // TODO: Add td
                linear_combination
            })
            .collect()
    }
}

/// # Returns
/// - `SumcheckInstanceProof<F, ProofTranscript>`: The proof for the booleanity check.
/// - `Vec<F>`: r_address_prime.
/// - `Vec<F>`: r_cycle_prime.
/// - `F`: ra_claim_prime.
pub fn prove_booleanity<F, ProofTranscript>(
    trace: &[JoltONNXCycle],
    r: &[F],
    D: Vec<F>,
    G: Vec<F>,
    transcript: &mut ProofTranscript,
) -> (SumcheckInstanceProof<F, ProofTranscript>, Vec<F>, Vec<F>, F)
where
    ProofTranscript: Transcript,
    F: JoltField,
{
    const DEGREE: usize = 3;
    let K = r.len().pow2();
    let T = trace.len();
    let mut B = MultilinearPolynomial::from(EqPolynomial::evals(r));
    let num_rounds = K.log_2() + T.log_2();
    let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::with_capacity(num_rounds);

    // --- First log(K) rounds of sumcheck ---
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
    let mut F = vec![F::zero(); K];
    F[0] = F::one();
    let mut r_address_prime: Vec<F> = Vec::with_capacity(K.log_2());
    for round in 0..K.log_2() {
        let m = round + 1;
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
        let compressed_poly = univariate_poly.compress();
        compressed_poly.append_to_transcript(transcript);
        compressed_polys.push(compressed_poly);
        let r_j = transcript.challenge_scalar::<F>();
        r_address_prime.push(r_j);
        previous_claim = univariate_poly.evaluate(&r_j);
        B.bind_parallel(r_j, BindingOrder::LowToHigh);

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

    // Last log(T) rounds of sumcheck
    let eq_r_r = B.final_sumcheck_claim();
    let H: Vec<F> = trace.iter().map(|cycle| F[cycle.instr.address]).collect();
    let mut H = MultilinearPolynomial::from(H);
    let mut D = MultilinearPolynomial::from(D);
    let mut r_cycle_prime: Vec<F> = Vec::with_capacity(T.log_2());
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
        let mut univariate_poly_evals: [F; 3] = (0..D.len() / 2)
            .into_par_iter()
            .map(|i| {
                let D_evals = D.sumcheck_evals(i, DEGREE, BindingOrder::LowToHigh);
                let H_evals = H.sumcheck_evals(i, DEGREE, BindingOrder::LowToHigh);

                [
                    D_evals[0] * (H_evals[0] * H_evals[0] - H_evals[0]),
                    D_evals[1] * (H_evals[1] * H_evals[1] - H_evals[1]),
                    D_evals[2] * (H_evals[2] * H_evals[2] - H_evals[2]),
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

pub fn prove_raf_eval<F, ProofTranscript>(y: F, F: Vec<F>, transcript: &mut ProofTranscript)
where
    ProofTranscript: Transcript,
    F: JoltField,
{
    const DEGREE: usize = 2;
    let K = F.len();
    let num_rounds = K.log_2();
    let mut ra = MultilinearPolynomial::from(F);
    let mut int = IdentityPolynomial::<F>::new(num_rounds);
    let mut prev_claim = y;
    let mut compressed_polys: Vec<CompressedUniPoly<F>> = Vec::with_capacity(num_rounds);
    let mut r_address_double_prime: Vec<F> = Vec::with_capacity(num_rounds);
    for _ in 0..num_rounds {
        let uni_poly_evals: [F; 2] = (0..ra.len() / 2)
            .into_par_iter()
            .map(|index| {
                let ra_evals = ra.sumcheck_evals(index, DEGREE, BindingOrder::HighToLow);
                let int_evals = int.sumcheck_evals(index, DEGREE, BindingOrder::HighToLow);
                [ra_evals[0] * int_evals[0], ra_evals[1] * int_evals[1]]
            })
            .reduce(
                || [F::zero(); 2],
                |running, new| [running[0] + new[0], running[1] + new[1]],
            );
        let uni_poly = UniPoly::from_evals(&[
            uni_poly_evals[0],
            prev_claim - uni_poly_evals[0],
            uni_poly_evals[1],
        ]);
        let compressed_poly = uni_poly.compress();
        compressed_poly.append_to_transcript(transcript);
        compressed_polys.push(compressed_poly);

        let r_j = transcript.challenge_scalar::<F>();
        r_address_double_prime.push(r_j);
        prev_claim = uni_poly.evaluate(&r_j);

        // Bind polynomials
        rayon::join(
            || ra.bind_parallel(r_j, BindingOrder::HighToLow),
            || int.bind_parallel(r_j, BindingOrder::HighToLow),
        );
    }
    let _sc: SumcheckInstanceProof<F, ProofTranscript> =
        SumcheckInstanceProof::new(compressed_polys);
}

impl<F, ProofTranscript> BytecodeProof<F, ProofTranscript>
where
    ProofTranscript: Transcript,
    F: JoltField,
{
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
        let val: Vec<F> = Self::bytecode_to_val(&preprocessing.bytecode, &gamma);
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
        let eq_eval_address = EqPolynomial::mle(&r_address, r_address_prime);
        let r_cycle: Vec<_> = r_cycle.iter().copied().rev().collect();
        let eq_eval_cycle = EqPolynomial::mle(&r_cycle, r_cycle_prime);
        assert_eq!(
            eq_eval_address * eq_eval_cycle * (self.ra_claim_prime.square() - self.ra_claim_prime),
            sumcheck_claim,
            "Booleanity sumcheck failed"
        );
        Ok(())
    }
}
