use crate::field::JoltField;
use crate::host;
use crate::subprotocols::twist::{TwistAlgorithm, TwistProof};
use crate::utils::math::Math;
use crate::utils::transcript::{KeccakTranscript, Transcript};
use crate::zkvm::JoltVerifierPreprocessing;
use crate::zkvm::{Jolt, JoltRV32IM};
use ark_bn254::Fr;
use ark_serialize::CanonicalSerialize;
use ark_std::test_rng;
use plotly::{Layout, Plot, Scatter};
use rand_core::RngCore;
use rand_distr::{Distribution, Zipf};
use std::collections::HashMap;
use std::env;
use std::fs;
use std::io::Write;
use tracing_chrome::ChromeLayerBuilder;
use tracing_subscriber::prelude::*;

#[derive(Debug, Copy, Clone, clap::ValueEnum)]
pub enum BenchType {
    Btreemap,
    Fibonacci,
    Sha2,
    Sha3,
    Sha2Chain,
    Sha3Chain,
    Shout,
    Twist,
    MasterBenchmark,
    Plot,
}

pub fn benchmarks(bench_type: BenchType) -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    match bench_type {
        BenchType::Btreemap => btreemap(),
        BenchType::Sha2 => sha2(),
        BenchType::Sha3 => sha3(),
        BenchType::Sha2Chain => sha2_chain(),
        BenchType::Sha3Chain => sha3_chain(),
        BenchType::Fibonacci => fibonacci(),
        BenchType::Shout => shout(),
        BenchType::Twist => twist::<Fr, KeccakTranscript>(),
        BenchType::MasterBenchmark => master_benchmark(),
        BenchType::Plot => plot_from_csv(),
    }
}

fn plot_from_csv() -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    let task = move || {
        let mut benchmark_data: HashMap<String, Vec<(usize, f64, usize)>> = HashMap::new();

        // Load existing data from CSV
        if let Ok(contents) = fs::read_to_string("perfetto_traces/timings.csv") {
            for line in contents.lines() {
                let parts: Vec<&str> = line.split(',').collect();
                if parts.len() == 4 {
                    if let (Ok(scale), Ok(time), Ok(proof_size)) = (
                        parts[1].parse::<usize>(),
                        parts[2].parse::<f64>(),
                        parts[3].parse::<usize>(),
                    ) {
                        let bench_name = match parts[0] {
                            "fib" => "Fibonacci",
                            "sha2" | "sha2-chain" => "SHA2-chain",
                            "sha3" | "sha3-chain" => "SHA3-chain",
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

            if benchmark_data.is_empty() {
                eprintln!("No data found in perfetto_traces/timings.csv");
            } else {
                println!("Loaded {} benchmark types from CSV", benchmark_data.len());
                if let Err(e) = create_benchmark_plot(&benchmark_data) {
                    eprintln!("Failed to create clock speed plot: {e}");
                }
                if let Err(e) = create_proof_size_plot(&benchmark_data) {
                    eprintln!("Failed to create proof size plot: {e}");
                }
            }
        } else {
            eprintln!("Could not read perfetto_traces/timings.csv. Run benchmarks first to generate data.");
        }
    };

    vec![(
        tracing::info_span!("PlotFromCSV"),
        Box::new(task) as Box<dyn FnOnce()>,
    )]
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
    #[cfg(feature = "host")]
    extern crate sha2_inline;
    let mut inputs = vec![];
    inputs.append(&mut postcard::to_stdvec(&[5u8; 32]).unwrap());
    inputs.append(&mut postcard::to_stdvec(&50u32).unwrap());
    prove_example("sha2-chain-guest", inputs)
}

fn sha3_chain() -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    let mut inputs = vec![];
    inputs.append(&mut postcard::to_stdvec(&[5u8; 32]).unwrap());
    inputs.append(&mut postcard::to_stdvec(&20u32).unwrap());
    prove_example("sha3-chain-guest", inputs)
}

fn get_fib_input(scale: usize) -> u32 {
    let scale_factor = 1 << (scale - 18);
    20000u32 * scale_factor as u32
}

fn get_sha2_chain_iterations(scale: usize) -> u32 {
    200 * (1 << (scale - 18)) as u32
}

fn get_sha3_chain_iterations(scale: usize) -> u32 {
    20 * (1 << (scale - 18)) as u32
}

fn get_btreemap_ops(scale: usize) -> u32 {
    let scale_factor = 1 << (scale - 18);
    300u32 * scale_factor as u32
}

fn create_benchmark_plot(
    data: &HashMap<String, Vec<(usize, f64, usize)>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut plot = Plot::new();

    // Define colors for different benchmarks
    let colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"];

    for (color_idx, (bench_name, points)) in data.iter().enumerate() {
        let mut sorted_points = points.clone();
        sorted_points.sort_by_key(|(scale, _, _)| *scale);

        let color = colors[color_idx % colors.len()];

        // Collect data points at scale n positions
        let mut x_values: Vec<f64> = Vec::new();
        let mut y_values: Vec<f64> = Vec::new();

        for (scale, time, _) in sorted_points.iter() {
            // For benchmark at scale n, plot at position n
            let cycles = (1 << *scale) as f64;

            // Clock speed in KHz (cycles per millisecond)
            let clock_speed = cycles / (*time * 1000.0);

            // Use the scale value (n) for x-axis
            x_values.push(*scale as f64);
            y_values.push(clock_speed);
        }

        // Add only markers, no lines
        let trace = Scatter::new(x_values, y_values)
            .name(bench_name)
            .mode(plotly::common::Mode::Markers)
            .marker(plotly::common::Marker::new().size(10).color(color));

        plot.add_trace(trace);
    }

    let layout = Layout::new()
        .title("Jolt zkVM Benchmark Results")
        .x_axis(
            plotly::layout::Axis::new()
                .title("RISCV32IM Cycles (2^n)")
                .type_(plotly::layout::AxisType::Linear)
                .dtick(1.0) // Tick at every integer
                .tick_mode(plotly::common::TickMode::Linear),
        )
        .y_axis(
            plotly::layout::Axis::new().title("Prover Clock Speed (Cycles per millisecond, KHz)"),
        )
        .width(1200)
        .height(800);

    plot.set_layout(layout);

    let html_path = "perfetto_traces/benchmark_plot.html";
    plot.write_html(html_path);
    println!("Interactive plot saved to {html_path}");

    Ok(())
}

fn create_proof_size_plot(
    data: &HashMap<String, Vec<(usize, f64, usize)>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut plot = Plot::new();

    // Define colors for different benchmarks
    let colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"];

    for (color_idx, (bench_name, points)) in data.iter().enumerate() {
        let mut sorted_points = points.clone();
        sorted_points.sort_by_key(|(scale, _, _)| *scale);

        let color = colors[color_idx % colors.len()];

        // Collect data points
        let mut x_values: Vec<f64> = Vec::new();
        let mut y_values: Vec<f64> = Vec::new();

        for (scale, _, proof_size) in sorted_points.iter() {
            // Convert 2^scale to millions of cycles for x-axis
            let cycles_millions = (1 << *scale) as f64 / 1_000_000.0;

            // Convert proof size from bytes to KB
            let proof_size_kb = *proof_size as f64 / 1024.0;

            x_values.push(cycles_millions);
            y_values.push(proof_size_kb);
        }

        // Add only markers, no lines
        let trace = Scatter::new(x_values, y_values)
            .name(bench_name)
            .mode(plotly::common::Mode::Markers)
            .marker(plotly::common::Marker::new().size(10).color(color));

        plot.add_trace(trace);
    }

    let layout = Layout::new()
        .title("Jolt zkVM Proof Size")
        .x_axis(
            plotly::layout::Axis::new()
                .title("RISCV32IM Cycles (millions)")
                .type_(plotly::layout::AxisType::Linear),
        )
        .y_axis(plotly::layout::Axis::new().title("Proof Size (KB)"))
        .width(1200)
        .height(800);

    plot.set_layout(layout);

    let html_path = "perfetto_traces/proof_size_plot.html";
    plot.write_html(html_path);
    println!("Proof size plot saved to {html_path}");

    Ok(())
}

fn master_benchmark() -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    // Ensure SHA2 inline library is linked and auto-registered
    #[cfg(feature = "host")]
    extern crate sha2_inline;
    let bench_type = env::var("BENCH_TYPE").unwrap_or_else(|_| "all".to_string());
    let bench_scales_str = env::var("BENCH_SCALES").unwrap_or_else(|_| {
        env::var("BENCH_SCALE")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .map(|s| s.to_string())
            .unwrap_or_else(|| "18".to_string())
    });

    let bench_scales: Vec<usize> = bench_scales_str
        .split(',')
        .filter_map(|s| s.trim().parse::<usize>().ok())
        .collect();

    if let Err(e) = fs::create_dir_all("perfetto_traces") {
        eprintln!("Warning: Failed to create perfetto_traces directory: {e}");
    }

    let task = move || {
        let mut benchmark_data: HashMap<String, Vec<(usize, f64, usize)>> = HashMap::new();

        // Load existing data from CSV if available
        if let Ok(contents) = fs::read_to_string("perfetto_traces/timings.csv") {
            for line in contents.lines() {
                let parts: Vec<&str> = line.split(',').collect();
                if parts.len() == 4 {
                    if let (Ok(scale), Ok(time), Ok(proof_size)) = (
                        parts[1].parse::<usize>(),
                        parts[2].parse::<f64>(),
                        parts[3].parse::<usize>(),
                    ) {
                        let bench_name = match parts[0] {
                            "fib" => "Fibonacci",
                            "sha2" | "sha2-chain" => "SHA2-chain",
                            "sha3" | "sha3-chain" => "SHA3-chain",
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

        let benchmarks_to_run = if bench_type == "all" {
            vec!["fib", "sha2-chain", "sha3-chain", "btreemap"]
        } else {
            vec![bench_type.as_str()]
        };

        for bench_scale in bench_scales {
            println!("\n=== Running benchmarks at scale 2^{bench_scale} ===");
            let max_trace_length = 1 << bench_scale;

            for current_bench in &benchmarks_to_run {
                let (duration, proof_size) = match *current_bench {
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
                    "sha2" | "sha2-chain" => "SHA2-chain",
                    "sha3" | "sha3-chain" => "SHA3-chain",
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
                    proof_size
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

        if !benchmark_data.is_empty() {
            if let Err(e) = create_benchmark_plot(&benchmark_data) {
                eprintln!("Failed to create clock speed plot: {e}");
            }
            if let Err(e) = create_proof_size_plot(&benchmark_data) {
                eprintln!("Failed to create proof size plot: {e}");
            }
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

        let (jolt_proof, program_io, _, _) =
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
    _bench_name: &str,
    _scale: usize,
) -> (std::time::Duration, usize) {
    let mut program = host::Program::new(example_name);
    let (bytecode, init_memory_state, _) = program.decode();
    let (_, _, program_io) = program.trace(&serialized_input);

    let preprocessing = JoltRV32IM::prover_preprocess(
        bytecode.clone(),
        program_io.memory_layout.clone(),
        init_memory_state,
        max_trace_length,
    );

    let span = tracing::info_span!("E2E");
    let (jolt_proof, program_io, _, start) =
        span.in_scope(|| JoltRV32IM::prove(&preprocessing, &mut program, &serialized_input));
    let prove_duration = start.elapsed();
    let proof_size = jolt_proof.serialized_size(ark_serialize::Compress::Yes);

    let verifier_preprocessing = JoltVerifierPreprocessing::from(&preprocessing);
    let verification_result =
        JoltRV32IM::verify(&verifier_preprocessing, jolt_proof, program_io, None);
    assert!(
        verification_result.is_ok(),
        "Verification failed with error: {:?}",
        verification_result.err()
    );

    (prove_duration, proof_size)
}
