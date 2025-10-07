use ark_serialize::CanonicalSerialize;
use jolt_core::host;
use jolt_core::zkvm::dag::state_manager::ProofKeys;
use jolt_core::zkvm::JoltVerifierPreprocessing;
use jolt_core::zkvm::{Jolt, JoltRV64IMAC};
use plotly::{Layout, Plot, Scatter};
use std::collections::HashMap;
use std::fs;
use std::io::Write;

// Empirically measured cycles per operation for RV64IMAC
const CYCLES_PER_SHA256: f64 = 3396.0;
const CYCLES_PER_SHA3: f64 = 4330.0;
const CYCLES_PER_BTREEMAP_OP: f64 = 1550.0;
const CYCLES_PER_FIBONACCI_UNIT: f64 = 12.0;
const SAFETY_MARGIN: f64 = 0.9; // Use 90% of max trace capacity

/// Calculate number of operations to target a specific cycle count
fn scale_to_target_ops(target_cycles: usize, cycles_per_op: f64) -> u32 {
    std::cmp::max(1, (target_cycles as f64 / cycles_per_op) as u32)
}

#[derive(Debug, Copy, Clone, clap::ValueEnum, strum_macros::Display)]
pub enum BenchType {
    BTreeMap,
    Fibonacci,
    Sha2,
    Sha3,
    #[strum(serialize = "SHA2 Chain")]
    Sha2Chain,
    #[strum(serialize = "SHA3 Chain")]
    Sha3Chain,
    Plot,
}

pub fn benchmarks(bench_type: BenchType) -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    match bench_type {
        BenchType::BTreeMap => btreemap(),
        BenchType::Sha2 => sha2(),
        BenchType::Sha3 => sha3(),
        BenchType::Sha2Chain => sha2_chain(),
        BenchType::Sha3Chain => sha3_chain(),
        BenchType::Fibonacci => fibonacci(),
        BenchType::Plot => plot_from_csv(),
    }
}

fn plot_from_csv() -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    let task = move || {
        let mut benchmark_data: HashMap<String, Vec<(usize, f64, usize, usize)>> = HashMap::new();

        // Load existing data from CSV
        if let Ok(contents) = fs::read_to_string("benchmark-runs/timings.csv") {
            for line in contents.lines() {
                let parts: Vec<&str> = line.split(',').collect();
                if parts.len() == 5 {
                    if let (Ok(scale), Ok(time), Ok(proof_size), Ok(proof_size_comp)) = (
                        parts[1].parse::<usize>(),
                        parts[2].parse::<f64>(),
                        parts[3].parse::<usize>(),
                        parts[4].parse::<usize>(),
                    ) {
                        benchmark_data
                            .entry(parts[0].to_string())
                            .or_default()
                            .push((scale, time, proof_size, proof_size_comp));
                    }
                }
            }

            if benchmark_data.is_empty() {
                eprintln!("No data found in benchmark-runs/timings.csv");
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
            eprintln!("Could not read benchmark-runs/timings.csv. Run benchmarks first to generate data.");
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
    #[cfg(feature = "host")]
    use jolt_inlines_sha2 as _;
    prove_example("sha2-guest", postcard::to_stdvec(&vec![5u8; 2048]).unwrap())
}

fn sha3() -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    #[cfg(feature = "host")]
    use jolt_inlines_keccak256 as _;
    prove_example("sha3-guest", postcard::to_stdvec(&vec![5u8; 2048]).unwrap())
}

fn btreemap() -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    prove_example("btreemap-guest", postcard::to_stdvec(&50u32).unwrap())
}

fn sha2_chain() -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    #[cfg(feature = "host")]
    use jolt_inlines_sha2 as _;
    let mut inputs = vec![];
    inputs.append(&mut postcard::to_stdvec(&[5u8; 32]).unwrap());
    inputs.append(&mut postcard::to_stdvec(&4480u32).unwrap());
    prove_example("sha2-chain-guest", inputs)
}

fn sha3_chain() -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    #[cfg(feature = "host")]
    extern crate jolt_inlines_keccak256;
    let mut inputs = vec![];
    inputs.append(&mut postcard::to_stdvec(&[5u8; 32]).unwrap());
    inputs.append(&mut postcard::to_stdvec(&20u32).unwrap());
    prove_example("sha3-chain-guest", inputs)
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

    let html_path = "benchmark-runs/benchmark_plot.html";
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

    let html_path = "benchmark-runs/proof_size_plot.html";
    plot.write_html(html_path);
    println!("Proof size plot saved to {html_path}");

    Ok(())
}

pub fn master_benchmark(
    bench_type: BenchType,
    bench_scale: usize,
    target_trace_size: Option<usize>,
) -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    // Ensure SHA2 inline library is linked and auto-registered
    #[cfg(feature = "host")]
    extern crate jolt_inlines_sha2;
    #[cfg(feature = "host")]
    extern crate jolt_inlines_keccak256;

    if let Err(e) = fs::create_dir_all("benchmark-runs") {
        eprintln!("Warning: Failed to create benchmark-runs directory: {e}");
    }

    let task = move || {
        let mut benchmark_data: HashMap<String, Vec<(usize, f64, usize, usize)>> = HashMap::new();

        // Load existing data from CSV if available
        if let Ok(contents) = fs::read_to_string("benchmark-runs/timings.csv") {
            for line in contents.lines() {
                let parts: Vec<&str> = line.split(',').collect();
                if parts.len() == 5 {
                    if let (Ok(scale), Ok(time), Ok(proof_size), Ok(proof_size_comp)) = (
                        parts[1].parse::<usize>(),
                        parts[2].parse::<f64>(),
                        parts[3].parse::<usize>(),
                        parts[4].parse::<usize>(),
                    ) {
                        benchmark_data
                            .entry(parts[0].to_string())
                            .or_default()
                            .push((scale, time, proof_size, proof_size_comp));
                    }
                }
            }
        }

        println!("\n=== Running benchmark at scale 2^{bench_scale} ===");
        let max_trace_length = 1 << bench_scale;
        let bench_target = target_trace_size.unwrap_or(((1 << bench_scale) as f64 * SAFETY_MARGIN) as usize);

        let display_name = bench_type.to_string();
        
        // Map benchmark type to canonical name + input closure
        let (bench_name, input_fn): (&str, fn(usize) -> Vec<u8>) = match bench_type {
            BenchType::Fibonacci => ("fibonacci", |target| {
                postcard::to_stdvec(&scale_to_target_ops(target, CYCLES_PER_FIBONACCI_UNIT)).unwrap()
            }),
            BenchType::Sha2Chain => ("sha2-chain", |target| {
                let iterations = scale_to_target_ops(target, CYCLES_PER_SHA256);
                [postcard::to_stdvec(&[5u8; 32]).unwrap(),
                 postcard::to_stdvec(&iterations).unwrap()].concat()
            }),
            BenchType::Sha3Chain => ("sha3-chain", |target| {
                let iterations = scale_to_target_ops(target, CYCLES_PER_SHA3);
                [postcard::to_stdvec(&[5u8; 32]).unwrap(),
                 postcard::to_stdvec(&iterations).unwrap()].concat()
            }),
            BenchType::BTreeMap => ("btreemap", |target| {
                postcard::to_stdvec(&scale_to_target_ops(target, CYCLES_PER_BTREEMAP_OP)).unwrap()
            }),
            BenchType::Sha2 => panic!("Use sha2-chain instead"),
            BenchType::Sha3 => panic!("Use sha3-chain instead"),
            _ => unreachable!("Unsupported benchmark type"),
        };
        
        // Derive names from canonical bench_name
        let guest_name = format!("{bench_name}-guest");
        // Generate input and run benchmark
        println!("Running {bench_name} benchmark at scale 2^{bench_scale}");
        let input = input_fn(bench_target);
        let (duration, proof_size, proof_size_comp) = prove_example_with_trace(
            &guest_name,
            input,
            max_trace_length,
            bench_name,
            bench_scale,
        );
        
        println!("  Prover completed in {:.2}s", duration.as_secs_f64());
        
        // Store results
        benchmark_data
            .entry(display_name.clone())
            .or_default()
            .push((bench_scale, duration.as_secs_f64(), proof_size, proof_size_comp));
        
        // Write CSV
        let summary_line = format!(
            "{},{},{:.2},{},{}\n",
            bench_name, bench_scale, duration.as_secs_f64(), proof_size, proof_size_comp
        );
        if let Err(e) = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open("benchmark-runs/timings.csv")
            .and_then(|mut f| f.write_all(summary_line.as_bytes()))
        {
            eprintln!("Failed to write timing: {e}");
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
    let (trace, _, program_io) = program.trace(&serialized_input);

    let task = move || {
        let preprocessing = JoltRV64IMAC::prover_preprocess(
            bytecode.clone(),
            program_io.memory_layout.clone(),
            init_memory_state,
            (trace.len() + 1).next_power_of_two(),
        );

        let elf_contents_opt = program.get_elf_contents();
        let elf_contents = elf_contents_opt.as_deref().expect("elf contents is None");
        let (jolt_proof, program_io, _, _) =
            JoltRV64IMAC::prove(&preprocessing, elf_contents, &serialized_input);

        let verifier_preprocessing = JoltVerifierPreprocessing::from(&preprocessing);
        let verification_result =
            JoltRV64IMAC::verify(&verifier_preprocessing, jolt_proof, program_io, None);
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
    let (trace, _, program_io) = program.trace(&serialized_input);

    assert!(
        trace.len().next_power_of_two() <= max_trace_length,
        "Trace is longer than expected"
    );

    let preprocessing = JoltRV64IMAC::prover_preprocess(
        bytecode.clone(),
        program_io.memory_layout.clone(),
        init_memory_state,
        trace.len().next_power_of_two(),
    );

    let elf_contents_opt = program.get_elf_contents();
    let elf_contents = elf_contents_opt.as_deref().expect("elf contents is None");

    let span = tracing::info_span!("E2E");
    let (jolt_proof, program_io, _, start) =
        span.in_scope(|| JoltRV64IMAC::prove(&preprocessing, elf_contents, &serialized_input));
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
        JoltRV64IMAC::verify(&verifier_preprocessing, jolt_proof, program_io, None);
    assert!(
        verification_result.is_ok(),
        "Verification failed with error: {:?}",
        verification_result.err()
    );

    (prove_duration, proof_size, proof_size_full_compressed)
}
