use ark_serialize::CanonicalSerialize;
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::time::Instant;
use tracing_chrome::ChromeLayerBuilder;
use tracing_subscriber::layer::SubscriberExt;

use crate::host;
use crate::jolt::vm::rv32i_vm::{RV32IJoltVM, C, M};
use crate::jolt::vm::Jolt;
use crate::poly::commitment::hyperkzg::HyperKZG;
use crate::utils::transcript::KeccakTranscript;
use ark_bn254::{Bn254, Fr};

#[derive(Debug, Copy, Clone, clap::ValueEnum)]
pub enum BenchType {
    Btreemap,
    Fibonacci,
    Sha2,
    Sha3,
    Sha2Chain,
    Suite,
}

pub fn benchmarks(bench_type: BenchType) -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    match bench_type {
        BenchType::Suite => benchmark_suite(),
        _ => panic!(),
    }
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
    let bench_scales: Vec<usize> = vec![24, 25];

    if let Err(e) = fs::create_dir_all("perfetto_traces_baseline") {
        eprintln!("Warning: Failed to create perfetto_traces_baseline directory: {e}");
    }

    let task = move || {
        let mut benchmark_data: HashMap<String, Vec<(usize, f64, usize)>> = HashMap::new();

        // Load existing data from CSV if available
        if let Ok(contents) = fs::read_to_string("perfetto_traces_baseline/timings.csv") {
            for line in contents.lines() {
                let parts: Vec<&str> = line.split(',').collect();
                if parts.len() == 5 {
                    if let (Ok(scale), Ok(time), Ok(proof_size)) = (
                        parts[1].parse::<usize>(),
                        parts[2].parse::<f64>(),
                        parts[3].parse::<usize>(),
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
                            .push((scale, time, proof_size));
                    }
                }
            }
        }

        let benchmarks_to_run = vec!["fib", "sha2-chain", "sha3-chain", "btreemap"];

        for bench_scale in bench_scales {
            println!("\n=== Running benchmarks at scale 2^{bench_scale} ===");
            let max_trace_length = 1 << bench_scale;

            for current_bench in &benchmarks_to_run {
                let (duration, proof_size) = match *current_bench {
                    "fib" => {
                        println!("Running Fibonacci benchmark at scale 2^{bench_scale}");
                        let (chrome_layer, _guard) = ChromeLayerBuilder::new()
                            .file(format!("perfetto_traces_baseline/fib_{bench_scale}.json"))
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
                            .file(format!(
                                "perfetto_traces_baseline/sha2_chain_{bench_scale}.json"
                            ))
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
                            .file(format!(
                                "perfetto_traces_baseline/sha3_chain_{bench_scale}.json"
                            ))
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
                            .file(format!(
                                "perfetto_traces_baseline/btreemap_{bench_scale}.json"
                            ))
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
                    .push((bench_scale, duration.as_secs_f64(), proof_size));

                let summary_line = format!(
                    "{},{},{:.2},{}\n",
                    current_bench,
                    bench_scale,
                    duration.as_secs_f64(),
                    proof_size,
                );
                if let Err(e) = fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open("perfetto_traces_baseline/timings.csv")
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

fn prove_example_with_trace(
    example_name: &str,
    serialized_input: Vec<u8>,
    max_trace_length: usize,
    _bench_name: &str,
    _scale: usize,
) -> (std::time::Duration, usize) {
    let mut program = host::Program::new(example_name);

    let (io_device, trace) = program.trace(&serialized_input);
    let (bytecode, memory_init) = program.decode();

    let preprocessing: crate::jolt::vm::JoltProverPreprocessing<
        C,
        Fr,
        HyperKZG<Bn254, KeccakTranscript>,
        KeccakTranscript,
    > = RV32IJoltVM::prover_preprocess(
        bytecode.clone(),
        io_device.memory_layout.clone(),
        memory_init,
        1 << 18,
        1 << 18,
        max_trace_length,
    );

    let span = tracing::info_span!("E2E");

    let start = Instant::now();
    let (jolt_proof, jolt_commitments, _, _) = span.in_scope(|| {
        <RV32IJoltVM as Jolt<_, HyperKZG<Bn254, KeccakTranscript>, C, M, KeccakTranscript>>::prove(
            io_device,
            trace,
            preprocessing.clone(),
        )
    });
    let prove_duration = start.elapsed();

    let proof_size = jolt_proof.serialized_size(ark_serialize::Compress::Yes)
        + jolt_commitments.serialized_size(ark_serialize::Compress::Yes);

    (prove_duration, proof_size)
}
