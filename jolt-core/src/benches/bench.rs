use ark_serialize::CanonicalSerialize;
use plotly::layout::{Axis, BarMode};
use plotly::{Bar, Layout, Plot, Scatter};
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
    Sha3Chain,
    LargeDSumCheck,
    Suite,
}

pub fn benchmarks(bench_type: BenchType) -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    match bench_type {
        BenchType::Btreemap => btreemap(),
        BenchType::Sha2 => sha2(),
        BenchType::Sha3 => sha3(),
        BenchType::Sha2Chain => sha2_chain(),
        BenchType::Sha3Chain => sha3_chain(),
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
    let bench_scale = 23;
    let btreemap_ops = get_btreemap_ops(bench_scale);
    prove_example(
        "btreemap-guest",
        postcard::to_stdvec(&btreemap_ops).unwrap(),
    )
}

fn sha2_chain() -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    let iterations = get_sha2_chain_iterations(26);
    let mut inputs = vec![];
    inputs.append(&mut postcard::to_stdvec(&[5u8; 32]).unwrap());
    inputs.append(&mut postcard::to_stdvec(&iterations).unwrap());
    prove_example("sha2-chain-guest", inputs)
}

fn sha3_chain() -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    let mut inputs = vec![];
    inputs.append(&mut postcard::to_stdvec(&[5u8; 32]).unwrap());
    inputs.append(&mut postcard::to_stdvec(&300u32).unwrap());
    prove_example("sha3-chain-guest", inputs)
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
            1 << 26,
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

fn baseline_data() -> HashMap<String, Vec<(usize, f64, usize)>> {
    let mut data: HashMap<String, Vec<(usize, f64, usize)>> = HashMap::new();
    data.insert(
        "Fibonacci (Spice/Lasso)".to_string(),
        vec![
            (18, 2.18, 187711),
            (19, 3.90, 197895),
            (20, 7.43, 208495),
            (21, 14.62, 219511),
            (22, 27.49, 230943),
            (23, 54.96, 242791),
            (24, 111.35, 255055),
            (25, 289.54, 267735),
        ],
    );
    data.insert(
        "SHA2 chain (Spice/Lasso)".to_string(),
        vec![
            (18, 2.19, 191359),
            (19, 4.08, 201543),
            (20, 7.65, 212143),
            (21, 14.34, 223159),
            (22, 29.65, 234591),
            (23, 60.89, 246439),
            (24, 112.07, 258703),
            (25, 309.63, 271383),
        ],
    );
    data.insert(
        "SHA3 chain (Spice/Lasso)".to_string(),
        vec![
            (18, 2.21, 191359),
            (19, 3.93, 201543),
            (20, 7.50, 212143),
            (21, 14.29, 223159),
            (22, 27.32, 234591),
            (23, 55.40, 246439),
            (24, 111.63, 258703),
            (25, 314.68, 271383),
        ],
    );
    data.insert(
        "BTreeMap (Spice/Lasso)".to_string(),
        vec![
            (18, 2.11, 192991),
            (19, 3.80, 203175),
            (20, 7.08, 213775),
            (21, 13.89, 226527),
            (22, 36.21, 239799),
            (23, 57.81, 251647),
            (24, 109.58, 265855),
            (25, 297.00, 280583),
        ],
    );
    data
}

fn twist_shout_data() -> HashMap<String, Vec<(usize, f64, usize, usize)>> {
    let mut data: HashMap<String, Vec<(usize, f64, usize, usize)>> = HashMap::new();
    data.insert(
        "Fibonacci (Twist/Shout)".to_string(),
        vec![
            (18, 3.17, 69876, 40550),
            (19, 4.79, 73148, 42142),
            (20, 6.79, 73828, 42774),
            (21, 11.32, 77100, 44366),
            (22, 17.79, 77780, 44998),
            (23, 30.89, 81052, 46590),
            (24, 51.07, 81732, 47222),
            (25, 86.75, 85004, 48814),
        ],
    );
    data.insert(
        "SHA2 chain (Twist/Shout)".to_string(),
        vec![
            (18, 3.41, 70020, 40694),
            (19, 4.99, 73292, 42286),
            (20, 7.24, 73972, 42918),
            (21, 11.93, 77244, 44510),
            (22, 19.19, 77924, 45142),
            (23, 32.53, 81196, 46734),
            (24, 55.35, 81876, 47366),
            (25, 96.43, 85148, 48958),
        ],
    );
    data.insert(
        "SHA3 chain (Twist/Shout)".to_string(),
        vec![
            (18, 3.43, 70020, 40694),
            (19, 4.97, 73292, 42286),
            (20, 7.32, 73972, 42918),
            (21, 12.07, 77244, 44510),
            (22, 19.27, 77924, 45142),
            (23, 32.75, 81196, 46734),
            (24, 57.30, 81876, 47366),
            (25, 97.94, 85148, 48958),
        ],
    );
    data.insert(
        "BTreeMap (Twist/Shout)".to_string(),
        vec![
            (18, 2.84, 70124, 40798),
            (19, 4.55, 73396, 42390),
            (20, 6.73, 74076, 43022),
            (21, 11.27, 77452, 44718),
            (22, 18.17, 78236, 45454),
            (23, 32.37, 81508, 47046),
            (24, 61.23, 82898, 48068),
            (25, 98.78, 86274, 49764),
        ],
    );
    data
}

pub fn create_prover_speed_plot() -> Result<(), Box<dyn std::error::Error>> {
    let baseline_data = baseline_data();
    let twist_shout_data = twist_shout_data();

    let mut plot = Plot::new();

    // Define colors for different benchmarks
    let colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"];

    for (color_idx, (bench_name, points)) in baseline_data.iter().enumerate() {
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
            .marker(
                plotly::common::Marker::new()
                    .size(10)
                    .color(color)
                    .symbol(plotly::common::MarkerSymbol::X),
            );

        plot.add_trace(trace);
    }

    for (color_idx, (bench_name, points)) in twist_shout_data.iter().enumerate() {
        let mut sorted_points = points.clone();
        sorted_points.sort_by_key(|(scale, _, _, _)| *scale);

        let color = colors[color_idx % colors.len()];

        // Collect data points at scale n positions
        let mut x_values: Vec<f64> = Vec::new();
        let mut y_values: Vec<f64> = Vec::new();

        for (scale, time, _, _) in sorted_points.iter() {
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
            .marker(
                plotly::common::Marker::new()
                    .size(10)
                    .color(color)
                    .symbol(plotly::common::MarkerSymbol::Circle),
            );

        plot.add_trace(trace);
    }

    // Create custom tick labels
    let mut tick_vals: Vec<f64> = Vec::new();
    let mut tick_text: Vec<String> = Vec::new();

    // Add ticks for all scale values from 16 to 30
    for n in 16..=25 {
        tick_vals.push(n as f64);
        tick_text.push(format!("2^{n}"));
    }

    let layout = Layout::new()
        .title("Jolt zkVM Benchmark<br><sub>Hardware: 2023 M3 Max Macbook Pro 16 cores, 128 GB RAM</sub>")
        .x_axis(
            plotly::layout::Axis::new()
                .title("Trace length (RV32IM cycles)")
                .type_(plotly::layout::AxisType::Linear)
                .tick_values(tick_vals)
                .tick_text(tick_text),
        )
        .y_axis(
            plotly::layout::Axis::new()
                .title("Prover Speed, kHz (Cycles proved per millisecond)"),
        )
        .width(1200)
        .height(800);

    plot.set_layout(layout);

    let html_path = "perfetto_traces/benchmark_plot.html";
    plot.write_html(html_path);
    println!("Interactive plot saved to {html_path}");

    Ok(())
}

pub fn create_proof_size_plot() -> Result<(), Box<dyn std::error::Error>> {
    let baseline_data = baseline_data();
    let twist_shout_data = twist_shout_data();

    let mut plot = Plot::new();

    // Define colors for different benchmarks
    let colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"];

    for (color_idx, (bench_name, points)) in baseline_data.iter().enumerate() {
        let mut sorted_points = points.clone();
        sorted_points.sort_by_key(|(scale, _, _)| *scale);

        let color = colors[color_idx % colors.len()];

        // Collect data points for uncompressed proof size
        let mut x_values: Vec<f64> = Vec::new();
        let mut y_values: Vec<f64> = Vec::new();

        for (scale, _, proof_size) in sorted_points.iter() {
            let cycles = (1 << *scale) as f64;

            // Convert proof sizes from bytes to KB
            let proof_size_kb = *proof_size as f64 / 1024.0;

            x_values.push(cycles);
            y_values.push(proof_size_kb);
        }

        // Add uncompressed proof size with filled markers
        let trace_uncompressed = Scatter::new(x_values, y_values)
            .name(format!("{bench_name}"))
            .mode(plotly::common::Mode::Markers)
            .marker(
                plotly::common::Marker::new()
                    .size(10)
                    .color(color)
                    .symbol(plotly::common::MarkerSymbol::X),
            );

        plot.add_trace(trace_uncompressed);
    }

    for (color_idx, (bench_name, points)) in twist_shout_data.iter().enumerate() {
        let mut sorted_points = points.clone();
        sorted_points.sort_by_key(|(scale, _, _, _)| *scale);

        let color = colors[color_idx % colors.len()];

        // Collect data points for uncompressed proof size
        let mut x_values: Vec<f64> = Vec::new();
        let mut y_values: Vec<f64> = Vec::new();

        // Collect data points for compressed proof size
        let mut x_values_comp: Vec<f64> = Vec::new();
        let mut y_values_comp: Vec<f64> = Vec::new();

        for (scale, _, proof_size, proof_size_comp) in sorted_points.iter() {
            let cycles = (1 << *scale) as f64;

            // Convert proof sizes from bytes to KB
            let proof_size_kb = *proof_size as f64 / 1024.0;
            let proof_size_comp_kb = *proof_size_comp as f64 / 1024.0;

            x_values.push(cycles);
            y_values.push(proof_size_kb);

            x_values_comp.push(cycles);
            y_values_comp.push(proof_size_comp_kb);
        }

        // Add uncompressed proof size with filled markers
        let trace_uncompressed = Scatter::new(x_values, y_values)
            .name(format!("{bench_name} uncompressed"))
            .mode(plotly::common::Mode::Markers)
            .marker(plotly::common::Marker::new().size(10).color(color));

        plot.add_trace(trace_uncompressed);

        // Add compressed proof size with hollow markers (white fill with colored outline)
        let trace_compressed = Scatter::new(x_values_comp, y_values_comp)
            .name(format!("{bench_name} compressed"))
            .mode(plotly::common::Mode::Markers)
            .marker(
                plotly::common::Marker::new()
                    .size(10)
                    .color("white")
                    .line(plotly::common::Line::new().color(color).width(2.0)),
            );

        plot.add_trace(trace_compressed);
    }

    // Create custom tick labels for x-axis (in millions, but labeled as 2^n)
    let mut tick_vals: Vec<f64> = Vec::new();
    let mut tick_text: Vec<String> = Vec::new();

    for n in [18, 21, 22, 23, 24, 25] {
        tick_vals.push((1 << n) as f64);
        tick_text.push(format!("2^{n}"));
    }

    let layout = Layout::new()
        .title("Jolt zkVM Proof Size")
        .x_axis(
            plotly::layout::Axis::new()
                .title("Trace length (RV32IM Cycles)")
                .type_(plotly::layout::AxisType::Linear)
                .tick_values(tick_vals)
                .tick_text(tick_text),
        )
        .y_axis(plotly::layout::Axis::new().title("Proof size, kB"))
        .width(1200)
        .height(800);

    plot.set_layout(layout);

    let html_path = "perfetto_traces/proof_size_plot.html";
    plot.write_html(html_path);
    println!("Proof size plot saved to {html_path}");

    Ok(())
}

pub fn create_prover_time_stacked_bar_chart() {
    let x_vals = vec!["2^18", "2^20", "2^22", "2^24", "2^26"];

    let spartan_times = vec![
        0.063218167,
        0.241247792,
        0.918651959,
        4.018379291,
        16.883696959,
    ];
    let total_prove_times = vec![
        3.455596291,
        7.249402208,
        17.665766708,
        52.122210833,
        183.01000275,
    ];
    let dory_commitment_times = vec![
        0.760738916,
        1.62285075,
        3.745405291,
        9.76091225,
        29.817534042,
    ];
    let registers_twist_times = vec![
        0.050287377,
        0.169915421,
        0.553899621,
        2.094848582,
        8.968360504,
    ];
    let ram_twist_times = vec![
        0.075968592,
        0.168778755,
        0.48967716,
        1.693429088,
        6.842295842,
    ];
    let bytecode_shout_times = vec![
        0.057913128,
        0.121835336,
        0.397341205,
        1.606799953,
        6.50301608,
    ];
    let instruction_shout_times = vec![
        0.298935496,
        0.586434921,
        1.665092297,
        5.584993702,
        22.535375962,
    ];
    let dory_opening_times = vec![
        1.235668416,
        2.444058125,
        4.76064775,
        10.265910917,
        21.44526775,
    ];

    let dory_commitment_percentages: Vec<_> = dory_commitment_times
        .iter()
        .zip(total_prove_times.iter())
        .map(|(commit, total)| commit / total * 100.0)
        .collect();
    let dory_opening_percentages: Vec<_> = dory_opening_times
        .iter()
        .zip(total_prove_times.iter())
        .map(|(opening, total)| opening / total * 100.0)
        .collect();
    let remaining_percentages: Vec<_> = dory_commitment_percentages
        .iter()
        .zip(dory_opening_percentages.iter())
        .map(|(commit, open)| 100.0 - commit - open)
        .collect();

    let trace_commit = Bar::new(x_vals.clone(), dory_commitment_percentages)
        .name("Dory commitment")
        .width(0.5);
    let trace_open = Bar::new(x_vals.clone(), dory_opening_percentages)
        .name("Dory opening proof")
        .width(0.5);
    let trace_other = Bar::new(x_vals.clone(), remaining_percentages)
        .name("Other")
        .width(0.5);

    let layout = Layout::new()
        .title("Jolt zkVM Prover Time Breakdown")
        .bar_mode(BarMode::Stack)
        .x_axis(Axis::new().title("Trace length (RV32IM cycles)"))
        .y_axis(
            Axis::new()
                .title("Percentage of prover time")
                .range(vec![0.0, 100.0])
                .dtick(20.0),
        )
        .width(800);

    let mut plot = Plot::new();
    plot.add_trace(trace_commit);
    plot.add_trace(trace_open);
    plot.add_trace(trace_other);
    plot.set_layout(layout);

    let html_path = "perfetto_traces/prover_time_breakdown.html";
    plot.write_html(html_path);
    println!("Proof size plot saved to {html_path}");
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
