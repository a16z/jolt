use ark_serialize::CanonicalSerialize;
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::time::Instant;
use tracing_chrome::ChromeLayerBuilder;
use tracing_subscriber::layer::SubscriberExt;

use crate::host;
use crate::poly::multilinear_polynomial::MultilinearPolynomial;
use crate::subprotocols::large_degree_sumcheck::{
    compute_initial_eval_claim, AppendixCSumCheckProof, LargeDMulSumCheckProof, NaiveSumCheckProof,
};
use crate::subprotocols::toom::FieldMulSmall;
use crate::transcripts::{Blake2bTranscript, Transcript};
use crate::utils::math::Math;
use crate::utils::thread::unsafe_allocate_zero_vec;
use crate::zkvm::dag::state_manager::ProofKeys;
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
    Suite,
}

pub fn benchmarks(bench_type: BenchType) -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    match bench_type {
        BenchType::Btreemap => btreemap(),
        BenchType::Sha2 => sha2(),
        BenchType::Sha3 => sha3(),
        BenchType::Sha2Chain => sha2_chain(),
        BenchType::Fibonacci => fibonacci(),
        BenchType::LargeDSumCheck => large_d_sumcheck::<Fr, Blake2bTranscript>(),
        BenchType::Suite => benchmark_suite(),
    }
}

fn fibonacci() -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    prove_example("fibonacci-guest", postcard::to_stdvec(&400000u32).unwrap())
}

fn sha2() -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    prove_example("sha2-guest", postcard::to_stdvec(&vec![5u8; 2048]).unwrap())
}

fn sha3() -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    prove_example("sha3-guest", postcard::to_stdvec(&vec![5u8; 2048]).unwrap())
}

fn btreemap() -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    prove_example("btreemap-guest", postcard::to_stdvec(&50u32).unwrap())
}

fn sha2_chain() -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    let mut inputs = vec![];
    inputs.append(&mut postcard::to_stdvec(&[5u8; 32]).unwrap());
    inputs.append(&mut postcard::to_stdvec(&1000u32).unwrap());
    prove_example("sha2-chain-guest", inputs)
}

fn get_fib_input(scale: usize) -> u32 {
    let scale_factor = 1 << (scale - 18);
    10000u32 * scale_factor as u32
}

fn get_sha2_chain_iterations(scale: usize) -> u32 {
    30 * (1 << (scale - 18)) as u32
}

fn get_sha3_chain_iterations(scale: usize) -> u32 {
    10 * (1 << (scale - 18)) as u32
}

fn get_btreemap_ops(scale: usize) -> u32 {
    let scale_factor = 1 << (scale - 18);
    150u32 * scale_factor as u32
}

fn benchmark_suite() -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    let bench_scales: Vec<usize> = vec![26];

    if let Err(e) = fs::create_dir_all("perfetto_traces") {
        eprintln!("Warning: Failed to create perfetto_traces directory: {e}");
    }

    let task = move || {
        let mut benchmark_data: HashMap<String, Vec<(usize, f64, usize, usize)>> = HashMap::new();

        // Load existing data from CSV if available
        if let Ok(contents) = fs::read_to_string("perfetto_traces/timings.csv") {
            for line in contents.lines() {
                let parts: Vec<&str> = line.split(',').collect();
                if parts.len() == 5 {
                    if let (Ok(scale), Ok(time), Ok(proof_size), Ok(proof_size_comp)) = (
                        parts[1].parse::<usize>(),
                        parts[2].parse::<f64>(),
                        parts[3].parse::<usize>(),
                        parts[4].parse::<usize>(),
                    ) {
                        let bench_name = match parts[0] {
                            "fib" => "Fibonacci",
                            "sha2" | "sha2-chain" => "SHA2 chain",
                            "sha3" | "sha3-chain" => "SHA3 chain",
                            "btreemap" => "BTreeMap",
                            other => other,
                        };
                        benchmark_data
                            .entry(bench_name.to_string())
                            .or_default()
                            .push((scale, time, proof_size, proof_size_comp));
                    }
                }
            }
        }

        let benchmarks_to_run = vec!["fib", "sha2-chain", "sha3-chain", "btreemap"];

        for bench_scale in bench_scales {
            println!("\n=== Running benchmarks at scale 2^{bench_scale} ===");
            let max_trace_length = 1 << bench_scale;

            for current_bench in &benchmarks_to_run {
                let (duration, proof_size, proof_size_comp) = match *current_bench {
                    "fib" => {
                        println!("Running Fibonacci benchmark at scale 2^{bench_scale}");
                        let (chrome_layer, _guard) = ChromeLayerBuilder::new()
                            .file(format!("perfetto_traces/fib_{bench_scale}.json"))
                            .build();
                        let subscriber = tracing_subscriber::registry().with(chrome_layer);
                        let _guard = tracing::subscriber::set_default(subscriber);

                        let fib_input = get_fib_input(bench_scale);
                        prove_example_with_trace(
                            "fibonacci-guest",
                            postcard::to_stdvec(&fib_input).unwrap(),
                            max_trace_length,
                            "Fibonacci",
                            bench_scale,
                        )
                    }
                    "sha2" | "sha2-chain" => {
                        println!("Running SHA2-chain benchmark at scale 2^{bench_scale}");
                        let (chrome_layer, _guard) = ChromeLayerBuilder::new()
                            .file(format!("perfetto_traces/sha2_chain_{bench_scale}.json"))
                            .build();
                        let subscriber = tracing_subscriber::registry().with(chrome_layer);
                        let _guard = tracing::subscriber::set_default(subscriber);

                        let iterations = get_sha2_chain_iterations(bench_scale);
                        let mut inputs = vec![];
                        inputs.append(&mut postcard::to_stdvec(&[5u8; 32]).unwrap());
                        inputs.append(&mut postcard::to_stdvec(&iterations).unwrap());
                        prove_example_with_trace(
                            "sha2-chain-guest",
                            inputs,
                            max_trace_length,
                            "SHA2_chain",
                            bench_scale,
                        )
                    }
                    "sha3" | "sha3-chain" => {
                        println!("Running SHA3-chain benchmark at scale 2^{bench_scale}");
                        let (chrome_layer, _guard) = ChromeLayerBuilder::new()
                            .file(format!("perfetto_traces/sha3_chain_{bench_scale}.json"))
                            .build();
                        let subscriber = tracing_subscriber::registry().with(chrome_layer);
                        let _guard = tracing::subscriber::set_default(subscriber);

                        let iterations = get_sha3_chain_iterations(bench_scale);
                        let mut inputs = vec![];
                        inputs.append(&mut postcard::to_stdvec(&[5u8; 32]).unwrap());
                        inputs.append(&mut postcard::to_stdvec(&iterations).unwrap());
                        prove_example_with_trace(
                            "sha3-chain-guest",
                            inputs,
                            max_trace_length,
                            "SHA3_chain",
                            bench_scale,
                        )
                    }
                    "btreemap" => {
                        println!("Running BTreeMap benchmark at scale 2^{bench_scale}");
                        let (chrome_layer, _guard) = ChromeLayerBuilder::new()
                            .file(format!("perfetto_traces/btreemap_{bench_scale}.json"))
                            .build();
                        let subscriber = tracing_subscriber::registry().with(chrome_layer);
                        let _guard = tracing::subscriber::set_default(subscriber);

                        let btreemap_ops = get_btreemap_ops(bench_scale);
                        prove_example_with_trace(
                            "btreemap-guest",
                            postcard::to_stdvec(&btreemap_ops).unwrap(),
                            max_trace_length,
                            "BTreeMap",
                            bench_scale,
                        )
                    }
                    _ => {
                        eprintln!("Unknown benchmark type: {current_bench}");
                        continue;
                    }
                };

                println!("  Prover completed in {:.2}s", duration.as_secs_f64());

                let bench_name = match *current_bench {
                    "fib" => "Fibonacci",
                    "sha2" | "sha2-chain" => "SHA2 chain",
                    "sha3" | "sha3-chain" => "SHA3 chain",
                    "btreemap" => "BTreeMap",
                    _ => *current_bench,
                };

                benchmark_data
                    .entry(bench_name.to_string())
                    .or_default()
                    .push((
                        bench_scale,
                        duration.as_secs_f64(),
                        proof_size,
                        proof_size_comp,
                    ));

                let summary_line = format!(
                    "{},{},{:.2},{},{}\n",
                    current_bench,
                    bench_scale,
                    duration.as_secs_f64(),
                    proof_size,
                    proof_size_comp,
                );
                if let Err(e) = fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open("perfetto_traces/timings.csv")
                    .and_then(|mut f| f.write_all(summary_line.as_bytes()))
                {
                    eprintln!("Failed to write timing: {e}");
                }
            }
        }
    };

    vec![(
        tracing::info_span!("Benchmark suite"),
        Box::new(task) as Box<dyn FnOnce()>,
    )]
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

        let elf_contents_opt = program.get_elf_contents();
        let elf_contents = elf_contents_opt.as_deref().expect("elf contents is None");
        let (jolt_proof, program_io, _) =
            JoltRV32IM::prove(&preprocessing, elf_contents, &serialized_input);

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

fn prove_example_with_trace(
    example_name: &str,
    serialized_input: Vec<u8>,
    max_trace_length: usize,
    _bench_name: &str,
    _scale: usize,
) -> (std::time::Duration, usize, usize) {
    let mut program = host::Program::new(example_name);
    let (bytecode, init_memory_state, _) = program.decode();
    let (_, _, program_io) = program.trace(&serialized_input);

    let preprocessing = JoltRV32IM::prover_preprocess(
        bytecode.clone(),
        program_io.memory_layout.clone(),
        init_memory_state,
        max_trace_length,
    );

    let elf_contents_opt = program.get_elf_contents();
    let elf_contents = elf_contents_opt.as_deref().expect("elf contents is None");

    let span = tracing::info_span!("E2E");

    let start = Instant::now();
    let (jolt_proof, _, _) =
        span.in_scope(|| JoltRV32IM::prove(&preprocessing, &elf_contents, &serialized_input));
    let prove_duration = start.elapsed();

    let proof_size = jolt_proof.serialized_size(ark_serialize::Compress::Yes);
    let proof_size_projected = proof_size
        - jolt_proof.proofs[&ProofKeys::ReducedOpeningProof]
            .serialized_size(ark_serialize::Compress::Yes)
        + (jolt_proof.proofs[&ProofKeys::ReducedOpeningProof]
            .serialized_size(ark_serialize::Compress::No)
            / 3)
        - jolt_proof
            .commitments
            .serialized_size(ark_serialize::Compress::Yes)
        + (jolt_proof
            .commitments
            .serialized_size(ark_serialize::Compress::No)
            / 3);

    (prove_duration, proof_size, proof_size_projected)
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
