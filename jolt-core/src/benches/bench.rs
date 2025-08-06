use crate::field::JoltField;
use crate::host;
use crate::subprotocols::twist::{TwistAlgorithm, TwistProof};
use crate::utils::math::Math;
use crate::utils::transcript::{KeccakTranscript, Transcript};
use crate::zkvm::JoltVerifierPreprocessing;
use crate::zkvm::{Jolt, JoltRV32IM};
use ark_bn254::Fr;
use ark_std::test_rng;
use rand_core::RngCore;
use rand_distr::{Distribution, Zipf};
use std::env;
use std::fs;
use std::io::Write;
use std::time::Instant;
use tracing_chrome::ChromeLayerBuilder;
use tracing_subscriber::prelude::*;

#[derive(Debug, Copy, Clone, clap::ValueEnum)]
pub enum BenchType {
    Btreemap,
    Fibonacci,
    Sha2,
    Sha3,
    Sha2Chain,
    Shout,
    Twist,
    MasterBenchmark,
}

pub fn benchmarks(bench_type: BenchType) -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    match bench_type {
        BenchType::Btreemap => btreemap(),
        BenchType::Sha2 => sha2(),
        BenchType::Sha3 => sha3(),
        BenchType::Sha2Chain => sha2_chain(),
        BenchType::Fibonacci => fibonacci(),
        BenchType::Shout => shout(),
        BenchType::Twist => twist::<Fr, KeccakTranscript>(),
        BenchType::MasterBenchmark => master_benchmark(),
    }
}

fn shout() -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    todo!()
}

fn twist<F, ProofTranscript>() -> Vec<(tracing::Span, Box<dyn FnOnce()>)>
where
    F: JoltField,
    ProofTranscript: Transcript,
{
    let small_value_lookup_tables = F::compute_lookup_tables();
    F::initialize_lookup_tables(small_value_lookup_tables);

    let mut tasks = Vec::new();

    const K: usize = 1 << 10;
    const T: usize = 1 << 20;
    const ZIPF_S: f64 = 0.0;
    let zipf = Zipf::new(K as u64, ZIPF_S).unwrap();

    let mut rng = test_rng();

    let mut registers = [0u32; K];
    let mut read_addresses: Vec<usize> = Vec::with_capacity(T);
    let mut read_values: Vec<u32> = Vec::with_capacity(T);
    let mut write_addresses: Vec<usize> = Vec::with_capacity(T);
    let mut write_values: Vec<u32> = Vec::with_capacity(T);
    let mut write_increments: Vec<i64> = Vec::with_capacity(T);
    for _ in 0..T {
        // Random read register
        let read_address = zipf.sample(&mut rng) as usize - 1;
        // Random write register
        let write_address = zipf.sample(&mut rng) as usize - 1;
        read_addresses.push(read_address);
        write_addresses.push(write_address);
        // Read the value currently in the read register
        read_values.push(registers[read_address]);
        // Random write value
        let write_value = rng.next_u32();
        write_values.push(write_value);
        // The increment is the difference between the new value and the old value
        let write_increment = (write_value as i64) - (registers[write_address] as i64);
        write_increments.push(write_increment);
        // Write the new value to the write register
        registers[write_address] = write_value;
    }

    let mut prover_transcript = ProofTranscript::new(b"test_transcript");
    let r: Vec<F> = prover_transcript.challenge_vector(K.log_2());
    let r_prime: Vec<F> = prover_transcript.challenge_vector(T.log_2());

    let task = move || {
        let _proof = TwistProof::prove(
            read_addresses,
            read_values,
            write_addresses,
            write_values,
            write_increments,
            r.clone(),
            r_prime.clone(),
            &mut prover_transcript,
            TwistAlgorithm::Local,
        );
    };

    tasks.push((
        tracing::info_span!("Twist d=1"),
        Box::new(task) as Box<dyn FnOnce()>,
    ));

    tasks
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

// Helper functions for input scaling
fn get_fib_input(scale: usize) -> u32 {
    let scale_factor = 1 << (scale - 20);
    35000u32 * scale_factor as u32
}

fn get_sha_input_size(scale: usize) -> usize {
    let scale_factor = 1 << (scale - 20);
    2048
}

fn get_btreemap_ops(scale: usize) -> u32 {
    let scale_factor = 1 << (scale - 20);
    250u32 * scale_factor as u32
}

fn master_benchmark() -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    // Read environment variables
    let bench_type = env::var("BENCH_TYPE").unwrap_or_else(|_| "all".to_string());
    let bench_scale = env::var("BENCH_SCALE")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(20);
    
    // Create perfetto_traces directory
    if let Err(e) = fs::create_dir_all("perfetto_traces") {
        eprintln!("Warning: Failed to create perfetto_traces directory: {}", e);
    }
    
    let task = move || {
        let max_trace_length = 1 << bench_scale;
        
        let duration = match bench_type.as_str() {
            "fib" => {
                println!("Running Fibonacci benchmark at scale 2^{}", bench_scale);
                let (chrome_layer, _guard) = ChromeLayerBuilder::new()
                    .file(format!("perfetto_traces/fib_{}.json", bench_scale))
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
            "sha2" => {
                println!("Running SHA2 benchmark at scale 2^{}", bench_scale);
                let (chrome_layer, _guard) = ChromeLayerBuilder::new()
                    .file(format!("perfetto_traces/sha2_{}.json", bench_scale))
                    .build();
                let subscriber = tracing_subscriber::registry().with(chrome_layer);
                let _guard = tracing::subscriber::set_default(subscriber);
                
                let sha2_len = get_sha_input_size(bench_scale);
                let sha2_input = vec![5u8; sha2_len];
                prove_example_with_trace(
                    "sha2-guest",
                    postcard::to_stdvec(&sha2_input).unwrap(),
                    max_trace_length,
                    "SHA2",
                    bench_scale,
                )
            }
            "sha3" => {
                println!("Running SHA3 benchmark at scale 2^{}", bench_scale);
                let (chrome_layer, _guard) = ChromeLayerBuilder::new()
                    .file(format!("perfetto_traces/sha3_{}.json", bench_scale))
                    .build();
                let subscriber = tracing_subscriber::registry().with(chrome_layer);
                let _guard = tracing::subscriber::set_default(subscriber);
                
                let sha3_len = get_sha_input_size(bench_scale);
                let sha3_input = vec![5u8; sha3_len];
                prove_example_with_trace(
                    "sha3-guest",
                    postcard::to_stdvec(&sha3_input).unwrap(),
                    max_trace_length,
                    "SHA3",
                    bench_scale,
                )
            }
            "btreemap" => {
                println!("Running BTreeMap benchmark at scale 2^{}", bench_scale);
                let (chrome_layer, _guard) = ChromeLayerBuilder::new()
                    .file(format!("perfetto_traces/btreemap_{}.json", bench_scale))
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
                eprintln!("Unknown benchmark type: {}. Use fib, sha2, sha3, or btreemap", bench_type);
                return;
            }
        };
        
        println!("  Prover completed in {:.2}s", duration.as_secs_f64());
        
        let summary_line = format!("{},{},{:.2}\n", bench_type, bench_scale, duration.as_secs_f64());
        if let Err(e) = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open("perfetto_traces/timings.csv")
            .and_then(|mut f| f.write_all(summary_line.as_bytes()))
        {
            eprintln!("Failed to write timing: {}", e);
        }
    };
    
    vec![(
        tracing::info_span!("MasterBenchmark"),
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
        tracing::info_span!("Example_E2E"),
        Box::new(task) as Box<dyn FnOnce()>,
    ));

    tasks
}

fn prove_example_with_trace(
    example_name: &str,
    serialized_input: Vec<u8>,
    max_trace_length: usize,
    bench_name: &str,
    scale: usize,
) -> std::time::Duration {
    let mut program = host::Program::new(example_name);
    let (bytecode, init_memory_state, _) = program.decode();
    let (_, _, program_io) = program.trace(&serialized_input);

    let preprocessing = JoltRV32IM::prover_preprocess(
        bytecode.clone(),
        program_io.memory_layout.clone(),
        init_memory_state,
        max_trace_length,
    );

    let span = tracing::info_span!("{}_2^{}", bench_name, scale);
    let start = Instant::now();
    let (jolt_proof, program_io, _) = span.in_scope(|| {
        JoltRV32IM::prove(&preprocessing, &mut program, &serialized_input)
    });
    let prove_duration = start.elapsed();

    let verifier_preprocessing = JoltVerifierPreprocessing::from(&preprocessing);
    let verification_result =
        JoltRV32IM::verify(&verifier_preprocessing, jolt_proof, program_io, None);
    assert!(
        verification_result.is_ok(),
        "Verification failed with error: {:?}",
        verification_result.err()
    );

    prove_duration
}
