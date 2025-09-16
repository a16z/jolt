use ark_serialize::CanonicalSerialize;
use jolt_core::host;
use jolt_core::zkvm::dag::state_manager::ProofKeys;
use jolt_core::zkvm::JoltVerifierPreprocessing;
use jolt_core::zkvm::{Jolt, JoltRV32IM};
use plotly::{Layout, Plot, Scatter};
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
        BenchType::MasterBenchmark => master_benchmark(),
        BenchType::Plot => plot_from_csv(),
    }
}

fn plot_from_csv() -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    let task = move || {
        let mut benchmark_data: HashMap<String, Vec<(usize, f64, usize, usize)>> = HashMap::new();

        // Load existing data from CSV
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
                            "sha2" | "sha2-chain" => "SHA2-chain",
                            "sha3" | "sha3-chain" => "SHA3-chain",
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
    #[cfg(feature = "host")]
    extern crate sha3_inline;
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
    70 * (1 << (scale - 18)) as u32
}

fn get_sha3_chain_iterations(scale: usize) -> u32 {
    50 * (1 << (scale - 18)) as u32
}

fn get_btreemap_ops(scale: usize) -> u32 {
    let scale_factor = 1 << (scale - 18);
    150u32 * scale_factor as u32
}

fn create_benchmark_plot(
    data: &HashMap<String, Vec<(usize, f64, usize, usize)>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut plot = Plot::new();

    // Define colors for different benchmarks
    let colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"];

    for (color_idx, (bench_name, points)) in data.iter().enumerate() {
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
            .marker(plotly::common::Marker::new().size(10).color(color));

        plot.add_trace(trace);
    }

    // Create custom tick labels
    let mut tick_vals: Vec<f64> = Vec::new();
    let mut tick_text: Vec<String> = Vec::new();

    // Add ticks for all scale values from 16 to 30
    for n in 16..=30 {
        tick_vals.push(n as f64);

        // Special formatting for specific values
        let label = match n {
            20 => "2^20 (1 million)".to_string(),
            24 => "2^24 (16.8 million)".to_string(),
            26 => "2^26 (67 million)".to_string(),
            27 => "2^27 (134 million)".to_string(),
            28 => "2^28 (268 million)".to_string(),
            _ => format!("2^{n}"),
        };
        tick_text.push(label);
    }

    let layout = Layout::new()
        .title("Jolt zkVM Benchmark<br><sub>Hardware: AMD Threadripper PRO 7975WX 32 cores, 768 GB DDR5 RAM</sub>")
        .x_axis(
            plotly::layout::Axis::new()
                .title("Trace length (RISCV64IMAC Cycles)")
                .type_(plotly::layout::AxisType::Linear)
                .tick_values(tick_vals)
                .tick_text(tick_text),
        )
        .y_axis(
            plotly::layout::Axis::new()
                .title("Prover Speed (Cycles proved per millisecond, aka KHz)"),
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
    data: &HashMap<String, Vec<(usize, f64, usize, usize)>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut plot = Plot::new();

    // Define colors for different benchmarks
    let colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"];

    for (color_idx, (bench_name, points)) in data.iter().enumerate() {
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
            // Convert 2^scale to millions of cycles for x-axis
            let cycles_millions = (1 << *scale) as f64 / 1_000_000.0;

            // Convert proof sizes from bytes to KB
            let proof_size_kb = *proof_size as f64 / 1024.0;
            let proof_size_comp_kb = *proof_size_comp as f64 / 1024.0;

            x_values.push(cycles_millions);
            y_values.push(proof_size_kb);

            x_values_comp.push(cycles_millions);
            y_values_comp.push(proof_size_comp_kb);
        }

        // Add uncompressed proof size with filled markers
        let trace_uncompressed = Scatter::new(x_values, y_values)
            .name(format!("{bench_name} (uncompressed)"))
            .mode(plotly::common::Mode::Markers)
            .marker(plotly::common::Marker::new().size(10).color(color));

        plot.add_trace(trace_uncompressed);

        // Add compressed proof size with hollow markers (white fill with colored outline)
        let trace_compressed = Scatter::new(x_values_comp, y_values_comp)
            .name(format!("{bench_name} (compressed)"))
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

    // Add only specific ticks: 2^18, 2^22, 2^24, and 2^25 through 2^30
    let tick_scales = vec![18, 22, 23, 24, 25, 26, 27, 28, 29, 30];
    for n in tick_scales {
        let cycles_millions = (1_u64 << n) as f64 / 1_000_000.0;
        tick_vals.push(cycles_millions);

        // Special formatting for specific values
        let label = match n {
            20 => "2^20 (1 million)".to_string(),
            24 => "2^24 (16.8 million)".to_string(),
            26 => "2^26 (67 million)".to_string(),
            27 => "2^27 (134 million)".to_string(),
            28 => "2^28 (268 million)".to_string(),
            _ => format!("2^{n}"),
        };
        tick_text.push(label);
    }

    let layout = Layout::new()
        .title("Jolt zkVM Proof Size")
        .x_axis(
            plotly::layout::Axis::new()
                .title("Trace length (RISCV64IMAC Cycles)")
                .type_(plotly::layout::AxisType::Linear)
                .tick_values(tick_vals)
                .tick_text(tick_text),
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
    #[cfg(feature = "host")]
    extern crate sha3_inline;
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
                            "sha2" | "sha2-chain" => "SHA2-chain",
                            "sha3" | "sha3-chain" => "SHA3-chain",
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

        let benchmarks_to_run = if bench_type == "all" {
            vec!["fib", "sha2-chain", "sha3-chain", "btreemap"]
        } else {
            vec![bench_type.as_str()]
        };

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
                    "sha2" | "sha2-chain" => "SHA2-chain",
                    "sha3" | "sha3-chain" => "SHA3-chain",
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

        let elf_contents_opt = program.get_elf_contents();
        let elf_contents = elf_contents_opt.as_deref().expect("elf contents is None");
        let (jolt_proof, program_io, _, _) =
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
    let (jolt_proof, program_io, _, start) =
        span.in_scope(|| JoltRV32IM::prove(&preprocessing, elf_contents, &serialized_input));
    let prove_duration = start.elapsed();
    let proof_size = jolt_proof.serialized_size(ark_serialize::Compress::Yes);
    let proof_size_full_compressed = proof_size
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

    let verifier_preprocessing = JoltVerifierPreprocessing::from(&preprocessing);
    let verification_result =
        JoltRV32IM::verify(&verifier_preprocessing, jolt_proof, program_io, None);
    assert!(
        verification_result.is_ok(),
        "Verification failed with error: {:?}",
        verification_result.err()
    );

    (prove_duration, proof_size, proof_size_full_compressed)
}
