use crate::host;
use crate::poly::multilinear_polynomial::MultilinearPolynomial;
use crate::subprotocols::large_degree_sumcheck::{
    compute_initial_eval_claim, AppendixCSumCheckProof, LargeDMulSumCheckProof, NaiveSumCheckProof,
};
use crate::subprotocols::toom::FieldMulSmall;
use crate::transcripts::{Blake2bTranscript, Transcript};
use crate::utils::math::Math;
use crate::utils::thread::unsafe_allocate_zero_vec;
use crate::zkvm::JoltVerifierPreprocessing;
use crate::zkvm::{Jolt, JoltRV32IM};
use ark_bn254::Fr;
use ark_std::test_rng;
use rand_core::RngCore;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

#[derive(Debug, Copy, Clone, clap::ValueEnum)]
pub enum BenchType {
    Btreemap,
    Fibonacci,
    Sha2,
    Sha3,
    Sha2Chain,
    LargeDSumCheck,
}

pub fn benchmarks(bench_type: BenchType) -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    match bench_type {
        BenchType::Btreemap => btreemap(),
        BenchType::Sha2 => sha2(),
        BenchType::Sha3 => sha3(),
        BenchType::Sha2Chain => sha2_chain(),
        BenchType::Fibonacci => fibonacci(),
        BenchType::LargeDSumCheck => large_d_sumcheck::<Fr, Blake2bTranscript>(),
    }
}

fn fibonacci() -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    prove_example("fibonacci-guest", postcard::to_stdvec(&400000u32).unwrap())
}

fn sha2() -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    #[cfg(feature = "host")]
    use sha2_inline as _;
    prove_example("sha2-guest", postcard::to_stdvec(&vec![5u8; 2048]).unwrap())
}

fn sha3() -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    #[cfg(feature = "host")]
    use sha3_inline as _;
    prove_example("sha3-guest", postcard::to_stdvec(&vec![5u8; 2048]).unwrap())
}

fn btreemap() -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    prove_example("btreemap-guest", postcard::to_stdvec(&50u32).unwrap())
}

fn sha2_chain() -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    #[cfg(feature = "host")]
    use sha2_inline as _;
    let mut inputs = vec![];
    inputs.append(&mut postcard::to_stdvec(&[5u8; 32]).unwrap());
    inputs.append(&mut postcard::to_stdvec(&1000u32).unwrap());
    prove_example("sha2-chain-guest", inputs)
}

fn prove_example(
    example_name: &str,
    serialized_input: Vec<u8>,
) -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    let mut tasks = Vec::new();
    let mut program = host::Program::new(example_name);
    let (bytecode, init_memory_state, _) = program.decode();
    let (_, _, program_io) = program.trace(&serialized_input);

    let task = move || {
        let preprocessing = JoltRV32IM::prover_preprocess(
            bytecode.clone(),
            program_io.memory_layout.clone(),
            init_memory_state,
            1 << 24,
        );

        let (jolt_proof, program_io, _) =
            JoltRV32IM::prove(&preprocessing, &mut program, &serialized_input);

        let verifier_preprocessing = JoltVerifierPreprocessing::from(&preprocessing);
        let verification_result =
            JoltRV32IM::verify(&verifier_preprocessing, jolt_proof, program_io, None);
        assert!(
            verification_result.is_ok(),
            "Verification failed with error: {:?}",
            verification_result.err()
        );
    };

    tasks.push((
        tracing::info_span!("e2e benchmark"),
        Box::new(task) as Box<dyn FnOnce()>,
    ));

    tasks
}

fn large_d_sumcheck<F, ProofTranscript>() -> Vec<(tracing::Span, Box<dyn FnOnce()>)>
where
    F: FieldMulSmall,
    ProofTranscript: Transcript,
{
    let mut tasks = Vec::new();

    let T = 1 << 20;

    let task = move || {
        compare_sumcheck_implementations::<F, ProofTranscript, 31>(32, T);
        compare_sumcheck_implementations::<F, ProofTranscript, 15>(16, T);
        compare_sumcheck_implementations::<F, ProofTranscript, 7>(8, T);
        compare_sumcheck_implementations::<F, ProofTranscript, 3>(4, T);
    };

    tasks.push((
        tracing::info_span!("large_d_e2e"),
        Box::new(task) as Box<dyn FnOnce()>,
    ));

    tasks
}

fn compare_sumcheck_implementations<F, ProofTranscript, const D_MINUS_ONE: usize>(
    D: usize,
    T: usize,
) where
    F: FieldMulSmall,
    ProofTranscript: Transcript,
{
    let NUM_COPIES: usize = 3;

    let ra = {
        let mut rng = test_rng();
        let mut val_vec: Vec<Vec<F>> = vec![unsafe_allocate_zero_vec(T); D];

        for j in 0..T {
            for i in 0..D {
                val_vec[i][j] = F::from_u32(rng.next_u32());
            }
        }

        val_vec
            .into_par_iter()
            .map(MultilinearPolynomial::from)
            .collect::<Vec<_>>()
    };

    let mut transcript = ProofTranscript::new(b"test_transcript");
    let r_cycle: Vec<F> = transcript.challenge_vector(T.log_2());

    let previous_claim = compute_initial_eval_claim(&ra.iter().collect::<Vec<_>>(), &r_cycle);

    let (mut ra, mut transcript, mut previous_claim) = (
        vec![ra; NUM_COPIES],
        vec![transcript; NUM_COPIES],
        vec![previous_claim; NUM_COPIES],
    );

    let _proof = AppendixCSumCheckProof::<F, ProofTranscript>::prove::<D_MINUS_ONE>(
        &mut ra[0].iter_mut().collect::<Vec<_>>(),
        &r_cycle,
        &mut previous_claim[0],
        &mut transcript[0],
    );

    let _proof = NaiveSumCheckProof::<F, ProofTranscript>::prove(
        &mut ra[1].iter_mut().collect::<Vec<_>>(),
        &r_cycle,
        &mut previous_claim[1],
        &mut transcript[1],
    );

    let _proof = LargeDMulSumCheckProof::<F, ProofTranscript>::prove(
        &mut ra[2].to_vec(),
        &r_cycle,
        &mut previous_claim[2],
        &mut transcript[2],
    );
}
