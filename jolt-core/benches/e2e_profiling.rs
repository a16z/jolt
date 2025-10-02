use common::constants::{DEFAULT_MAX_INPUT_SIZE, DEFAULT_MAX_OUTPUT_SIZE};
use common::jolt_device::MemoryConfig;
use jolt_core::host;
use jolt_core::zkvm::JoltVerifierPreprocessing;
use jolt_core::zkvm::{Jolt, JoltRV64IMAC};

use std::fs;
use std::io::Write;
use std::time::Instant;

#[derive(Debug, Copy, Clone, clap::ValueEnum)]
pub enum BenchType {
    Btreemap,
    Fibonacci,
    Sha2,
    Sha3,
    Sha2Chain,
    Sha3Chain,
}

pub fn benchmarks(bench_type: BenchType) -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    match bench_type {
        BenchType::Btreemap => btreemap(),
        BenchType::Sha2 => sha2(),
        BenchType::Sha3 => sha3(),
        BenchType::Sha2Chain => sha2_chain(),
        BenchType::Sha3Chain => sha3_chain(),
        BenchType::Fibonacci => fibonacci(),
    }
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
    inputs.append(&mut postcard::to_stdvec(&50u32).unwrap());
    prove_example("sha2-chain-guest", inputs)
}

fn sha3_chain() -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    let mut inputs = vec![];
    inputs.append(&mut postcard::to_stdvec(&[5u8; 32]).unwrap());
    inputs.append(&mut postcard::to_stdvec(&20u32).unwrap());
    prove_example("sha3-chain-guest", inputs)
}

// Use 95% of trace capacity for benchmark purposes
const SAFETY_MARGIN: f64 = 0.9;
// Measured empirically for RV64
const CYCLES_PER_SHA256: f64 = 3396.0;
const CYCLES_PER_SHA3: f64 = 4330.0;
// Based on empirical data: 1406 cycles/op (2^24), 1531 cycles/op (2^27), 1568 cycles/op (2^28)
const CYCLES_PER_BTREEMAP_OP: f64 = 1550.0;
const CYCLES_PER_FIBONACCI_UNIT: f64 = 12.0;

fn scale_to_ops(target_trace_size: usize, cycles_per_op: f64) -> u32 {
    std::cmp::max(1, (target_trace_size as f64 / cycles_per_op) as u32)
}

fn get_memory_params(bench_type: &str, _bench_scale: usize) -> MemoryConfig {
    let base_stack_size = 1024 * 1024 * 10;
    let base_heap_size = 1024 * 1024 * 10;

    match bench_type {
        "btreemap" => MemoryConfig {
            max_input_size: 10000,
            max_output_size: 10000000,
            stack_size: base_stack_size,
            memory_size: base_heap_size,
            program_size: None,
        },
        _ => MemoryConfig {
            max_input_size: DEFAULT_MAX_INPUT_SIZE,
            max_output_size: DEFAULT_MAX_OUTPUT_SIZE,
            stack_size: base_stack_size,
            memory_size: base_heap_size,
            program_size: None,
        },
    }
}

pub fn master_benchmark(
    bench_type: BenchType,
    bench_scale: usize,
    target_trace_size: Option<usize>,
) -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    if let Err(e) = fs::create_dir_all("benchmark-runs/perfetto_traces") {
        eprintln!("Warning: Failed to create benchmark-runs/perfetto_traces directory: {e}");
    }
    if let Err(e) = fs::create_dir_all("benchmark-runs/results") {
        eprintln!("Warning: Failed to create benchmark-runs/results directory: {e}");
    }
    let bench_target_trace_size =
        target_trace_size.unwrap_or(((1 << bench_scale) as f64 * SAFETY_MARGIN) as usize);

    let task = move || {
        let (bench_name, input_fn): (&str, fn(usize) -> Vec<u8>) = match bench_type {
            BenchType::Fibonacci => ("fibonacci", |target_trace_size_| {
                postcard::to_stdvec(&scale_to_ops(target_trace_size_, CYCLES_PER_FIBONACCI_UNIT))
                    .unwrap()
            }),
            BenchType::Sha2Chain => ("sha2-chain", |target_trace_size_| {
                let iterations = scale_to_ops(target_trace_size_, CYCLES_PER_SHA256);
                [
                    postcard::to_stdvec(&[5u8; 32]).unwrap(),
                    postcard::to_stdvec(&iterations).unwrap(),
                ]
                .concat()
            }),
            BenchType::Sha3Chain => ("sha3-chain", |target_trace_size_| {
                let iterations = scale_to_ops(target_trace_size_, CYCLES_PER_SHA3);
                println!("number of sha3 iterations: {:?}", iterations);
                [
                    postcard::to_stdvec(&[5u8; 32]).unwrap(),
                    postcard::to_stdvec(&iterations).unwrap(),
                ]
                .concat()
            }),
            BenchType::Sha2 => panic!("Use sha2-chain instead"),
            BenchType::Sha3 => panic!("Use sha3-chain instead"),
            BenchType::Btreemap => ("btreemap", |target_trace_size_| {
                postcard::to_stdvec(&scale_to_ops(target_trace_size_, CYCLES_PER_BTREEMAP_OP))
                    .unwrap()
            }),
        };

        let input = input_fn(bench_target_trace_size);
        let max_trace_length = 1 << bench_scale;
        let (trace_length, duration) = prove_example_with_trace(
            &format!("{bench_name}-guest"),
            input,
            max_trace_length,
            bench_name,
            bench_scale,
        );

        println!(
            "  Prover completed in {:.2}s ({:.2} kHz)",
            duration.as_secs_f64(),
            trace_length as f64 / duration.as_secs_f64() / 1000.0
        );

        let summary_line = format!(
            "{},{},{:.2},{},{:.2}",
            bench_name,
            bench_scale,
            duration.as_secs_f64(),
            trace_length,
            trace_length as f64 / duration.as_secs_f64()
        );

        // Write individual result file for resume detection
        let individual_file = format!("benchmark-runs/results/{}_{}.csv", bench_name, bench_scale);
        if let Err(e) = fs::write(&individual_file, &summary_line) {
            eprintln!(
                "Failed to write individual result file {}: {e}",
                individual_file
            );
        }

        // Also append to consolidated timings file
        if let Err(e) = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open("benchmark-runs/results/timings.csv")
            .and_then(|mut f| f.write_all(format!("{}\n", summary_line).as_bytes()))
        {
            eprintln!("Failed to write consolidated timing: {e}");
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
    let mut program: host::Program = host::Program::new(example_name);
    let (bytecode, init_memory_state, _) = program.decode();
    let (_, _, program_io) = program.trace(&serialized_input);

    let task = move || {
        let preprocessing = JoltRV64IMAC::prover_preprocess(
            bytecode.clone(),
            program_io.memory_layout.clone(),
            init_memory_state,
            1 << 24,
        );

        let elf_contents_opt = program.get_elf_contents();
        let elf_contents = elf_contents_opt.as_deref().expect("elf contents is None");
        let (jolt_proof, program_io, _) =
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
        tracing::info_span!("e2e benchmark"),
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
) -> (usize, std::time::Duration) {
    let mut program: host::Program = host::Program::new(example_name);
    let memory_config = get_memory_params(bench_name, scale);
    program.set_memory_config(memory_config);
    let (bytecode, init_memory_state, _) = program.decode();
    let (trace, _, program_io) = program.trace(&serialized_input);
    let trace_length = trace.len();

    println!(
        "Trace length: {trace_length} ({:.1}% of 2^{scale})",
        (trace_length as f64 / max_trace_length as f64) * 100.0
    );
    assert!(
        trace_length <= max_trace_length,
        "Trace length is greater than max trace length"
    );
    let preprocessing = JoltRV64IMAC::prover_preprocess(
        bytecode.clone(),
        program_io.memory_layout.clone(),
        init_memory_state,
        max_trace_length,
    );

    let span = tracing::info_span!("prove_benchmark", bench_name, scale);
    let start = Instant::now();
    let elf_contents_opt = program.get_elf_contents();
    let elf_contents = elf_contents_opt.as_deref().expect("elf contents is None");
    let (jolt_proof, program_io, _) =
        span.in_scope(|| JoltRV64IMAC::prove(&preprocessing, &elf_contents, &serialized_input));
    let prove_duration = start.elapsed();

    let verifier_preprocessing = JoltVerifierPreprocessing::from(&preprocessing);
    let verification_result =
        JoltRV64IMAC::verify(&verifier_preprocessing, jolt_proof, program_io, None);
    assert!(
        verification_result.is_ok(),
        "Verification failed with error: {:?}",
        verification_result.err()
    );

    (trace_length, prove_duration)
}
