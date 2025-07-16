use itertools::Itertools;
use jolt_core::{
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
        math::Math,
        transcript::{self, AppendToTranscript, Transcript},
    },
};
use onnx_tracer::trace_types::{ONNXCycle, ONNXInstr};
use rayon::prelude::*;
use std::marker::PhantomData;

pub struct BytecodePreprocessing {
    code_size: usize,
    bytecode: Vec<ONNXInstr>,
}

pub struct BytecodeProof<F, ProofTranscript>
where
    ProofTranscript: Transcript,
    F: JoltField,
{
    _p: PhantomData<(F, ProofTranscript)>,
}

impl<F, ProofTranscript> BytecodeProof<F, ProofTranscript>
where
    ProofTranscript: Transcript,
    F: JoltField,
{
    pub fn prove(
        preprocessing: &BytecodePreprocessing,
        trace: &[ONNXCycle],
        transcript: &mut ProofTranscript,
    ) {
        let K = preprocessing.code_size;
        let T = trace.len();

        // --- Shout PIOP ---
        let r_cycle: Vec<F> = transcript.challenge_vector(T.log_2());
        let E = EqPolynomial::evals(&r_cycle);
        let mut F = vec![F::zero(); K];
        for (j, cycle) in trace.iter().enumerate() {
            let k = cycle.instr.address;
            F[k] += E[j]
        }
        let gamma: F = transcript.challenge_scalar();
        let val = Self::bytecode_to_val(&preprocessing.bytecode, &gamma);
        // sum-check setup
        let sumcheck_claim: F = F.iter().zip_eq(val.iter()).map(|(f, v)| *f * v).sum();
        let mut prev_claim = sumcheck_claim;
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
                    let ra_evals = ra.sumcheck_evals(index, DEGREE, BindingOrder::HighToLow);
                    let val_evals = val.sumcheck_evals(index, DEGREE, BindingOrder::HighToLow);
                    [ra_evals[0] * val_evals[0], ra_evals[1] * val_evals[1]]
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
            ra.bind_parallel(r, BindingOrder::HighToLow);
            val.bind_parallel(r, BindingOrder::HighToLow);
            prev_claim = uni_poly.evaluate(&r);
            r_address.push(r);
        }
        let core_piop_shout_proof: SumcheckInstanceProof<F, ProofTranscript> =
            SumcheckInstanceProof::new(sumcheck_proof);

        // --- Hamming weight check ---
        // sum-check setup
        let sumcheck_claim = F::one();
        let mut prev_claim = sumcheck_claim;
        let mut ra = MultilinearPolynomial::from(F);
        let mut r_address_prime: Vec<F> = Vec::with_capacity(num_rounds);
        let mut sumcheck_proof: Vec<CompressedUniPoly<F>> = Vec::with_capacity(num_rounds);
        for _ in 0..num_rounds {
            let uni_poly_eval: F = (0..ra.len() / 2)
                .into_par_iter()
                .map(|b| ra.get_bound_coeff(b))
                .sum();
            let uni_poly = UniPoly::from_evals(&[uni_poly_eval, prev_claim - uni_poly_eval]);
            let compressed_poly = uni_poly.compress();
            compressed_poly.append_to_transcript(transcript);
            sumcheck_proof.push(compressed_poly);
            let r_j = transcript.challenge_scalar::<F>();
            r_address_prime.push(r_j);
            prev_claim = uni_poly.evaluate(&r_j);
            ra.bind_parallel(r_j, BindingOrder::HighToLow);
        }

        // --- Booleanity check ---
        // --- raf evaluation ---
    }

    /// Reed-solomon fingerprint each instr in the program bytecode
    fn bytecode_to_val(program_bytecode: &[ONNXInstr], gamma: &F) -> Vec<F> {
        let mut gamma_pows = [F::one(); 4];
        for i in 1..4 {
            gamma_pows[i] *= gamma_pows[i - 1];
        }
        program_bytecode
            .iter()
            .map(|instr| {
                let mut linear_combination = F::zero();
                linear_combination += instr.opcode.into_bitflag().field_mul(gamma_pows[0]);
                linear_combination += (instr.address as u64).field_mul(gamma_pows[1]);
                linear_combination +=
                    (instr.ts1.unwrap_or_default() as u64).field_mul(gamma_pows[2]);
                linear_combination +=
                    (instr.ts2.unwrap_or_default() as u64).field_mul(gamma_pows[3]);
                linear_combination
            })
            .collect()
    }
}

pub fn prove_booleanity<F, ProofTranscript>(
    r: &[F],
    D: Vec<F>,
    G: Vec<F>,
    transcript: &mut ProofTranscript,
) where
    ProofTranscript: Transcript,
    F: JoltField,
{
    let B = EqPolynomial::evals(r);
}
