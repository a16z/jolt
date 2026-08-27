//! End-to-end benchmark harness for the MODULAR prover, mirroring the legacy
//! `jolt_prover_legacy benchmark` CLI so the two produce comparable numbers
//! and Perfetto traces.
//!
//! ```text
//! cargo run --release -p jolt-prover --example modular_benchmark \
//!     --features prover-fixtures -- \
//!     --name sha2-chain --scale 16 --format chrome
//! ```
//!
//! Pipeline: legacy-side guest compile/decode (the modular stack has no host
//! toolchain), legacy preprocessing → verifier preprocessing, modular trace
//! (`TracerBackend`), derived `ProverConfig`, `TraceBackend` witness,
//! `jolt_prover::prove` over `JoltBackend::reference()`, and a full
//! `jolt_verifier::verify` as the correctness gate. Only `prove` is timed —
//! the same window the legacy harness times. Reports wall-clock + kHz,
//! process-lifetime peak RSS (`getrusage`), and the per-stage RSS table
//! collected from the `prove_stage*` spans.

#![expect(
    clippy::expect_used,
    clippy::print_stdout,
    clippy::print_stderr,
    reason = "benchmark harness: fail loudly and report to stdout"
)]

use std::fs;
use std::io::Write as _;
use std::time::Instant;

use clap::{Parser, ValueEnum};
use common::jolt_device::MemoryConfig;
use jolt_claims::protocols::jolt::TracePolynomialOrder;
use jolt_crypto::{Bn254G1, Pedersen};
use jolt_dory::DoryScheme;
use jolt_field::Fr;
// Keep the inline libraries linked so their host-side registrations reach the
// tracer (both the legacy and modular trace runs), exactly as the legacy
// harness does.
use jolt_inlines_keccak256 as _;
use jolt_inlines_sha2 as _;
use jolt_profiling::{
    format_memory_size, peak_rss_bytes, report_stage_memory, setup_tracing, TracingFormat,
    BYTES_PER_GIB,
};
use jolt_program::execution::{
    ExecutionBackend, JoltProgram, OwnedTrace, TraceInputs, TraceOutput, TraceRow,
};
use jolt_prover::{JoltBackend, JoltProverPreprocessing, ProverConfig};
use jolt_prover_legacy::curve::Bn254Curve;
use jolt_prover_legacy::host;
use jolt_prover_legacy::poly::commitment::dory::DoryCommitmentScheme;
use jolt_prover_legacy::zkvm::preprocessing::JoltSharedPreprocessing;
use jolt_prover_legacy::zkvm::program::ProgramPreprocessing as LegacyProgramPreprocessing;
use jolt_prover_legacy::zkvm::proof::verifier_preprocessing_from_prover;
use jolt_prover_legacy::zkvm::prover::JoltProverPreprocessing as LegacyProverPreprocessing;
use jolt_transcript::LegacyBlake2bTranscript as Blake2bTranscript;
use jolt_witness::{JoltVmWitnessConfig, JoltVmWitnessInputs, TraceBackend};
use tracer::execution_backend::TracerBackend;

// Empirically measured cycles per operation for RV64IMAC — copied from the
// legacy harness (`benches/e2e_profiling.rs`) so both harnesses construct
// identical guest inputs for a given scale.
const CYCLES_PER_SHA256: f64 = 3396.0;
const CYCLES_PER_SHA3: f64 = 4330.0;
const CYCLES_PER_BTREEMAP_OP: f64 = 1550.0;
const CYCLES_PER_FIBONACCI_UNIT: f64 = 12.0;
const SAFETY_MARGIN: f64 = 0.9; // Use 90% of max trace capacity

fn scale_to_target_ops(target_cycles: usize, cycles_per_op: f64) -> u32 {
    std::cmp::max(1, (target_cycles as f64 / cycles_per_op) as u32)
}

#[derive(Debug, Copy, Clone, ValueEnum)]
enum BenchName {
    Fibonacci,
    Sha2Chain,
    Sha3Chain,
    #[value(name = "btreemap")]
    BTreeMap,
}

impl BenchName {
    /// The canonical bench name, also the guest crate prefix (`{name}-guest`).
    const fn as_str(self) -> &'static str {
        match self {
            Self::Fibonacci => "fibonacci",
            Self::Sha2Chain => "sha2-chain",
            Self::Sha3Chain => "sha3-chain",
            Self::BTreeMap => "btreemap",
        }
    }

    /// The guest input targeting `target` trace cycles — the same mapping as
    /// the legacy harness's `master_benchmark`.
    fn input(self, target: usize) -> Vec<u8> {
        match self {
            Self::Fibonacci => {
                postcard::to_stdvec(&scale_to_target_ops(target, CYCLES_PER_FIBONACCI_UNIT))
                    .expect("serialize input")
            }
            Self::Sha2Chain => [
                postcard::to_stdvec(&[5u8; 32]).expect("serialize input"),
                postcard::to_stdvec(&scale_to_target_ops(target, CYCLES_PER_SHA256))
                    .expect("serialize input"),
            ]
            .concat(),
            Self::Sha3Chain => [
                postcard::to_stdvec(&[5u8; 32]).expect("serialize input"),
                postcard::to_stdvec(&scale_to_target_ops(target, CYCLES_PER_SHA3))
                    .expect("serialize input"),
            ]
            .concat(),
            Self::BTreeMap => {
                postcard::to_stdvec(&scale_to_target_ops(target, CYCLES_PER_BTREEMAP_OP))
                    .expect("serialize input")
            }
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq)]
enum Format {
    Default,
    Chrome,
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Default)]
enum Backend {
    #[default]
    Reference,
    Optimized,
    #[cfg(all(feature = "metal", target_os = "macos"))]
    Metal,
}

/// Benchmark the modular prover end to end, mirroring
/// `jolt_prover_legacy benchmark` semantics.
#[derive(Parser, Debug)]
struct Cli {
    /// Benchmark to run.
    #[clap(long, value_enum)]
    name: BenchName,

    /// Max trace length as 2^scale (optional if target-trace-size is provided).
    #[clap(short, long)]
    scale: Option<usize>,

    /// Target specific cycle count (optional, defaults to 90% of 2^scale).
    #[clap(short, long)]
    target_trace_size: Option<usize>,

    /// Output formats.
    #[clap(short, long, value_enum)]
    format: Option<Vec<Format>>,

    /// Kernel backend to prove with.
    #[clap(short, long, value_enum, default_value = "reference")]
    backend: Backend,
}

fn main() {
    let cli = Cli::parse();
    let scale = match (cli.scale, cli.target_trace_size) {
        (Some(scale), _) => scale,
        (None, Some(target)) => target.next_power_of_two().trailing_zeros() as usize,
        (None, None) => {
            eprintln!("Error: Must provide either --scale or --target-trace-size");
            std::process::exit(1);
        }
    };
    let bench_name = cli.name.as_str();

    let formats: Vec<TracingFormat> = cli
        .format
        .unwrap_or_default()
        .iter()
        .map(|format| match format {
            Format::Default => TracingFormat::Default,
            Format::Chrome => TracingFormat::Chrome,
        })
        .collect();
    let backend_suffix = match cli.backend {
        Backend::Reference => "",
        Backend::Optimized => "_optimized",
        #[cfg(all(feature = "metal", target_os = "macos"))]
        Backend::Metal => "_metal",
    };
    let trace_name = format!(
        "modular_{}_{scale}{backend_suffix}",
        bench_name.replace('-', "_")
    );
    let _guards = setup_tracing(&formats, &trace_name);

    run_benchmark(cli.name, scale, cli.target_trace_size, cli.backend);
}

fn run_benchmark(
    bench: BenchName,
    scale: usize,
    target_trace_size: Option<usize>,
    backend_choice: Backend,
) {
    let bench_name = bench.as_str();
    let max_trace_length = 1usize << scale;
    let bench_target =
        target_trace_size.unwrap_or((max_trace_length as f64 * SAFETY_MARGIN) as usize);
    tracing::info!("Running modular {bench_name} benchmark at scale 2^{scale}");
    fs::create_dir_all("benchmark-runs/results").expect("create results directory");

    let input = bench.input(bench_target);

    // --- Guest + preprocessing (untimed, mirrors the legacy harness): the
    // guest is compiled/decoded through the legacy host toolchain, and the
    // verifier preprocessing (program view + digest) comes from the legacy
    // preprocessing exactly as in the byte-diff tests.
    let mut program = host::Program::new(&format!("{bench_name}-guest"));
    let (bytecode, init_memory_state, _, entry_address) = program.decode();
    let (_, legacy_trace, _, io_device) = program.trace(&input, &[], &[]);
    assert!(
        legacy_trace.len().next_power_of_two() <= max_trace_length,
        "Trace is longer than expected"
    );
    drop(legacy_trace);
    let elf_contents = program.get_elf_contents().expect("elf contents");
    let memory_layout = io_device.memory_layout.clone();

    let program_data =
        LegacyProgramPreprocessing::preprocess(bytecode, init_memory_state, entry_address)
            .expect("legacy preprocess");
    let shared_preprocessing =
        JoltSharedPreprocessing::new(program_data, memory_layout.clone(), max_trace_length);
    let legacy_preprocessing = LegacyProverPreprocessing::<
        jolt_prover_legacy::ark_bn254::Fr,
        Bn254Curve,
        DoryCommitmentScheme,
    >::new(shared_preprocessing);
    let verifier_preprocessing = verifier_preprocessing_from_prover(&legacy_preprocessing);
    let program_preprocessing = verifier_preprocessing
        .program
        .as_full_arc()
        .expect("full program preprocessing");
    let jolt_program = std::sync::Arc::new(JoltProgram::from_elf_bytes(elf_contents));

    let span = tracing::info_span!("E2E").entered();

    // --- Modular trace (untimed, like legacy's `gen_from_elf` emulation).
    let trace_output = trace_modular(&jolt_program, &memory_layout, &input);
    let trace_length = trace_output.trace.rows().len();

    let mut config = ProverConfig::derive::<Fr>(
        trace_output.trace.rows(),
        &memory_layout,
        verifier_preprocessing.program.min_bytecode_address(),
        verifier_preprocessing.program.program_image_len_words(),
        max_trace_length,
    )
    .expect("derive config");
    // Bench-only A/B ablation: `JOLT_BOOLEANITY_ANCHOR=legacy` pins the
    // stage-5 anchor so both arms run from one binary.
    if std::env::var("JOLT_BOOLEANITY_ANCHOR").as_deref() == Ok("legacy") {
        config.booleanity_anchor = jolt_verifier::config::BooleanityAnchor::Stage5Instruction;
    }
    if std::env::var("JOLT_TRACE_ORDER").as_deref() == Ok("address") {
        config.trace_polynomial_order = TracePolynomialOrder::AddressMajor;
    }
    let public_io = trace_output.device.clone();
    let padded_output = pad_trace(trace_output, config.trace_length);

    let witness = TraceBackend::new(
        JoltVmWitnessConfig::new(
            config.trace_length.ilog2() as usize,
            config.ram_K,
            config.one_hot_config,
        ),
        JoltVmWitnessInputs::new(&jolt_program, &program_preprocessing, padded_output),
    );

    // PCS setup sized like the byte-diff harness: the main one-hot matrix
    // maxed with both advice candidates (always included in setup sizing,
    // present or not — the SRS is prefix-stable).
    let total_vars = (config.one_hot_config.committed_chunk_bits()
        + config.trace_length.ilog2() as usize)
        .max(advice_vars(memory_layout.max_trusted_advice_size))
        .max(advice_vars(memory_layout.max_untrusted_advice_size));
    let prover_preprocessing = JoltProverPreprocessing::<DoryScheme, Pedersen<Bn254G1>> {
        verifier: verifier_preprocessing,
        pcs_setup: DoryScheme::setup_prover(total_vars),
        committed_program: None,
    };
    let backend = match backend_choice {
        Backend::Reference => JoltBackend::<Fr, DoryScheme>::reference(),
        Backend::Optimized => JoltBackend::<Fr, DoryScheme>::optimized(),
        #[cfg(all(feature = "metal", target_os = "macos"))]
        Backend::Metal => JoltBackend::<Fr, DoryScheme>::metal().expect("metal backend"),
    };

    // --- The timed window: the full modular prove (witness materialization,
    // commitment, all sumcheck stages, joint opening) — the same window the
    // legacy harness times around `prover.prove()`.
    let now = Instant::now();
    let proof = jolt_prover::prove::<Fr, DoryScheme, Pedersen<Bn254G1>, Blake2bTranscript, _>(
        &backend,
        &prover_preprocessing,
        &config,
        None,
        &witness,
        &public_io,
    )
    .expect("modular prove");
    let duration = now.elapsed();
    drop(span);

    let proof_size = bincode::serde::encode_to_vec(&proof, bincode::config::standard())
        .expect("serialize proof")
        .len();

    // --- Correctness gate (untimed): the proof must verify.
    jolt_verifier::verify::<Fr, DoryScheme, Pedersen<Bn254G1>, Blake2bTranscript>(
        &prover_preprocessing.verifier,
        &public_io,
        &proof,
        None,
    )
    .expect("modular proof verifies");

    let backend_label = match backend_choice {
        Backend::Reference => "reference",
        Backend::Optimized => "optimized",
        #[cfg(all(feature = "metal", target_os = "macos"))]
        Backend::Metal => "metal",
    };
    let proving_hz = trace_length as f64 / duration.as_secs_f64();
    let padded_proving_hz = trace_length.next_power_of_two() as f64 / duration.as_secs_f64();
    println!(
        "modular {} (2^{}, {backend_label}): Prover completed in {:.2}s ({:.1} kHz / padded {:.1} kHz)",
        bench_name,
        scale,
        duration.as_secs_f64(),
        proving_hz / 1000.0,
        padded_proving_hz / 1000.0,
    );
    if let Some(peak) = peak_rss_bytes() {
        println!(
            "modular {} (2^{}, {backend_label}): Peak RSS {}",
            bench_name,
            scale,
            format_memory_size(peak as f64 / BYTES_PER_GIB),
        );
    }
    if let Some(footprint) = jolt_profiling::phys_footprint() {
        println!(
            "modular {} (2^{}, {backend_label}): Peak footprint {}",
            bench_name,
            scale,
            format_memory_size(footprint.lifetime_peak_bytes as f64 / BYTES_PER_GIB),
        );
    }
    report_stage_memory();

    // The same 7-field CSV line the legacy harness writes, under a
    // modular-prefixed file name.
    let summary_line = format!(
        "{},{},{:.2},{},{:.2},{},{}\n",
        bench_name,
        scale,
        duration.as_secs_f64(),
        trace_length.next_power_of_two(),
        padded_proving_hz,
        proof_size,
        proof_size,
    );
    let individual_file = format!("benchmark-runs/results/modular_{bench_name}_{scale}.csv");
    if let Err(e) = fs::write(&individual_file, &summary_line) {
        eprintln!("Failed to write individual result file {individual_file}: {e}");
    }
    if let Err(e) = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("benchmark-runs/results/modular_timings.csv")
        .and_then(|mut f| f.write_all(summary_line.as_bytes()))
    {
        eprintln!("Failed to write consolidated timing: {e}");
    }
}

/// Trace the guest through the modular stack (`TracerBackend`), with the
/// memory config mirrored off the legacy layout — the byte-diff wiring.
fn trace_modular(
    program: &JoltProgram,
    memory_layout: &common::jolt_device::MemoryLayout,
    inputs: &[u8],
) -> TraceOutput<OwnedTrace> {
    let memory_config = MemoryConfig {
        max_untrusted_advice_size: memory_layout.max_untrusted_advice_size,
        max_trusted_advice_size: memory_layout.max_trusted_advice_size,
        max_input_size: memory_layout.max_input_size,
        max_output_size: memory_layout.max_output_size,
        stack_size: memory_layout.stack_size,
        heap_size: memory_layout.heap_size,
        program_size: Some(memory_layout.program_size),
    };
    TracerBackend::new()
        .trace(
            program,
            TraceInputs {
                inputs: inputs.to_vec(),
                untrusted_advice: Vec::new(),
                trusted_advice: Vec::new(),
                memory_config,
                advice_tape: None,
            },
        )
        .expect("modular trace")
}

/// Pad to the padded trace length with no-op rows, as legacy does.
fn pad_trace(
    trace_output: TraceOutput<OwnedTrace>,
    trace_length: usize,
) -> TraceOutput<OwnedTrace> {
    let mut rows = trace_output.trace.rows().to_vec();
    rows.resize(trace_length, TraceRow::default());
    TraceOutput::new(
        OwnedTrace::new(rows),
        trace_output.device,
        trace_output.final_memory,
        trace_output.advice_tape,
    )
}

/// A word-aligned advice buffer's balanced Dory matrix variable count.
fn advice_vars(max_advice_size_bytes: u64) -> usize {
    ((max_advice_size_bytes / 8) as usize)
        .next_power_of_two()
        .max(1)
        .ilog2() as usize
}
