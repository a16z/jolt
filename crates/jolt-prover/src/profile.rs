//! The profile harness behind the `profiling` feature: one documented
//! command proves a named workload on the modular stack and emits both
//! telemetry artifacts from the same span stream —
//! `benchmark-runs/perfetto_traces/{trace_name}.json` (Perfetto UI /
//! `trace_processor` SQL) and `{trace_name}.summary.json` (machine-queryable
//! aggregates for `jolt-eval` and `jq`).
//!
//! ```text
//! cargo run --release -p jolt-prover --features profiling -- \
//!     profile --name fibonacci --format chrome
//! ```
//!
//! Pipeline (promoted from the retired `examples/modular_benchmark.rs`):
//! legacy-side guest compile/decode (the modular stack has no host
//! toolchain), legacy preprocessing → verifier preprocessing, modular trace
//! (`TracerBackend`), derived `ProverConfig`, `TraceBackend` witness,
//! [`jolt_prover::prove`](crate::prove) over the selected backend, and a full
//! `jolt_verifier::verify` as the correctness gate. Only `prove` is measured
//! — guest compilation, tracer execution, and preprocessing are excluded
//! from every reported metric.

#![expect(
    clippy::expect_used,
    clippy::print_stdout,
    clippy::print_stderr,
    reason = "profile harness: fail loudly and report to stdout"
)]

use std::fs;
use std::io::Write as _;
use std::path::PathBuf;
use std::time::Instant;

use clap::ValueEnum;
use common::jolt_device::MemoryConfig;
use jolt_crypto::{Bn254G1, Pedersen};
use jolt_dory::DoryScheme;
use jolt_field::Fr;
// Keep the inline libraries linked so their host-side registrations reach the
// tracer, exactly as the legacy harness does.
use jolt_inlines_keccak256 as _;
use jolt_inlines_sha2 as _;
use jolt_profiling::summary::{finalize_trace, ProfileSummary, SummaryContext};
use jolt_profiling::{
    format_memory_size, peak_rss_bytes, report_stage_memory, setup_tracing, TracingFormat,
    BYTES_PER_GIB,
};
use jolt_program::execution::{
    ExecutionBackend, JoltProgram, OwnedTrace, TraceInputs, TraceOutput, TraceRow,
};
use jolt_prover_legacy::host;
use jolt_prover_legacy::poly::commitment::dory::DoryCommitmentScheme;
use jolt_prover_legacy::zkvm::preprocessing::JoltSharedPreprocessing;
use jolt_prover_legacy::zkvm::program::ProgramPreprocessing as LegacyProgramPreprocessing;
use jolt_prover_legacy::zkvm::proof::verifier_preprocessing_from_prover;
use jolt_prover_legacy::zkvm::prover::JoltProverPreprocessing as LegacyProverPreprocessing;
use jolt_transcript::LegacyBlake2bTranscript as Blake2bTranscript;
use jolt_witness::{JoltVmWitnessConfig, JoltVmWitnessInputs, TraceBackend};
use tracer::execution_backend::TracerBackend;

use crate::{JoltBackend, JoltProverPreprocessing, ProverConfig};

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

/// The scalable workloads the harness supports, with the default scales
/// pinned in `specs/prover-telemetry.md` (`jolt-eval` owns the normative
/// measurement-scale table and always passes `--scale` explicitly).
#[derive(Debug, Copy, Clone, ValueEnum)]
pub enum Workload {
    Fibonacci,
    Sha2Chain,
    Sha3Chain,
    #[value(name = "btreemap")]
    BTreeMap,
}

impl Workload {
    /// The canonical name, also the guest crate prefix (`{name}-guest`).
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Fibonacci => "fibonacci",
            Self::Sha2Chain => "sha2-chain",
            Self::Sha3Chain => "sha3-chain",
            Self::BTreeMap => "btreemap",
        }
    }

    /// Default log2 trace length when `--scale` is omitted.
    pub const fn default_scale(self) -> u32 {
        match self {
            Self::Fibonacci => 16,
            Self::Sha2Chain => 22,
            Self::Sha3Chain => 22,
            Self::BTreeMap => 20,
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

/// Subscriber stack selector.
#[derive(Debug, Clone, Copy, ValueEnum, PartialEq)]
pub enum OutputFormat {
    /// Console span-close timings only; no artifacts.
    Default,
    /// Full stack: chrome trace + summary.json + monitor counters — the
    /// format `jolt-eval` invokes.
    Chrome,
    /// No subscriber at all; times `prove()` with `std::time::Instant` — the
    /// overhead-budget baseline.
    None,
}

/// Prover backend selector. `reference` is the only backend today and is a
/// test oracle: absolute numbers are provisional, attribution is meaningful
/// relatively, and optimized backends slot into the same instrumented seams.
#[derive(Debug, Clone, Copy, ValueEnum, PartialEq)]
pub enum BackendKind {
    Reference,
}

/// `profile` subcommand arguments.
#[derive(Debug, clap::Args)]
pub struct ProfileArgs {
    /// Workload to prove.
    #[clap(long, value_enum)]
    pub name: Workload,

    /// log2 of the max (padded) trace length; per-workload default when
    /// omitted (fibonacci 16, sha2-chain 22, sha3-chain 22, btreemap 20).
    #[clap(long)]
    pub scale: Option<u32>,

    #[clap(long, value_enum, default_value = "chrome")]
    pub format: OutputFormat,

    #[clap(long, value_enum, default_value = "reference")]
    pub backend: BackendKind,
}

/// Artifact paths of one profile run (`None` unless `--format chrome`).
#[derive(Debug, Default)]
pub struct ProfileArtifacts {
    pub trace_path: Option<PathBuf>,
    pub summary_path: Option<PathBuf>,
    pub summary: Option<ProfileSummary>,
}

/// Runs one profile invocation end to end. The bin's `main` is a thin
/// wrapper over this so the smoke test can call it in-process.
///
/// # Panics
///
/// Panics on any pipeline failure (harness semantics), and if called twice
/// in one process with a subscriber-installing format (the global tracing
/// subscriber can only be set once).
pub fn run(args: &ProfileArgs) -> ProfileArtifacts {
    let scale = args.scale.unwrap_or_else(|| args.name.default_scale());
    let trace_name = format!("modular_{}_{scale}", args.name.as_str().replace('-', "_"));

    // Per-stage heap flamegraphs (allocative feature): opt in before the
    // prove so the cfg-gated hooks inside `prove()` see the prefix.
    #[cfg(feature = "allocative")]
    {
        std::fs::create_dir_all("benchmark-runs/flamegraphs")
            .expect("create flamegraphs directory");
        jolt_profiling::set_flamegraph_prefix(format!("benchmark-runs/flamegraphs/{trace_name}_"));
    }

    let guards = match args.format {
        OutputFormat::None => None,
        OutputFormat::Default => Some(setup_tracing(&[TracingFormat::Default], &trace_name)),
        OutputFormat::Chrome => Some(setup_tracing(&[TracingFormat::Chrome], &trace_name)),
    };

    run_workload(args.name, scale, args.backend);

    // Dropping the guards flushes the chrome trace; only then can the
    // flush-time pipeline parse it.
    drop(guards);

    if args.format != OutputFormat::Chrome {
        report_stage_memory();
        return ProfileArtifacts::default();
    }

    let trace_path = PathBuf::from(format!("benchmark-runs/perfetto_traces/{trace_name}.json"));
    let ctx = SummaryContext {
        workload: args.name.as_str().to_string(),
        scale_log2: scale,
        backend: "reference".to_string(),
    };
    let (summary_file, summary) = finalize_trace(&trace_path, &ctx).expect("finalize chrome trace");

    if let Some(root) = &summary.root {
        println!(
            "modular {} (2^{scale}): root span {:.2}s, dark time {:.1}%",
            args.name.as_str(),
            root.wall_time_ns as f64 / 1e9,
            root.dark_time_fraction * 100.0,
        );
    }
    println!("Trace:   {}", trace_path.display());
    println!("Summary: {}", summary_file.display());

    ProfileArtifacts {
        trace_path: Some(trace_path),
        summary_path: Some(summary_file),
        summary: Some(summary),
    }
}

fn run_workload(workload: Workload, scale: u32, backend: BackendKind) {
    let bench_name = workload.as_str();
    let max_trace_length = 1usize << scale;
    let bench_target = (max_trace_length as f64 * SAFETY_MARGIN) as usize;
    tracing::info!("Running modular {bench_name} profile at scale 2^{scale}");
    fs::create_dir_all("benchmark-runs/results").expect("create results directory");

    let input = workload.input(bench_target);

    // --- Guest + preprocessing (unmeasured): the guest is compiled/decoded
    // through the legacy host toolchain, and the verifier preprocessing
    // (program view + digest) comes from the legacy preprocessing exactly as
    // in the byte-diff tests.
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
        jolt_prover_legacy::curve::Bn254Curve,
        DoryCommitmentScheme,
    >::new(shared_preprocessing);
    let verifier_preprocessing = verifier_preprocessing_from_prover(&legacy_preprocessing);
    let program_preprocessing = verifier_preprocessing
        .program
        .as_full()
        .expect("full program preprocessing")
        .clone();
    let jolt_program = JoltProgram::from_elf_bytes(elf_contents);

    // --- Modular trace (unmeasured, like legacy's `gen_from_elf` emulation).
    let trace_output = trace_modular(&jolt_program, &memory_layout, &input);
    let trace_length = trace_output.trace.rows().len();

    let config = ProverConfig::derive::<Fr>(
        trace_output.trace.rows(),
        &memory_layout,
        verifier_preprocessing.program.min_bytecode_address(),
        verifier_preprocessing.program.program_image_len_words(),
        max_trace_length,
    )
    .expect("derive config");
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
    let backend = match backend {
        BackendKind::Reference => JoltBackend::<Fr, DoryScheme>::reference(),
    };

    // --- The measured window: the full modular prove (witness
    // materialization, commitment, all sumcheck stages, joint opening). The
    // `jolt_prover::prove` root span covers exactly this interval; the
    // Instant is the `--format none` no-subscriber baseline.
    let now = Instant::now();
    let proof = crate::prove::<Fr, DoryScheme, Pedersen<Bn254G1>, Blake2bTranscript, _>(
        &backend,
        &prover_preprocessing,
        &config,
        None,
        &witness,
        &public_io,
    )
    .expect("modular prove");
    let duration = now.elapsed();

    let proof_size = bincode::serde::encode_to_vec(&proof, bincode::config::standard())
        .expect("serialize proof")
        .len();

    // --- Correctness gate (unmeasured): the proof must verify.
    jolt_verifier::verify::<Fr, DoryScheme, Pedersen<Bn254G1>, Blake2bTranscript>(
        &prover_preprocessing.verifier,
        &public_io,
        &proof,
        None,
    )
    .expect("modular proof verifies");

    let proving_hz = trace_length as f64 / duration.as_secs_f64();
    let padded_proving_hz = trace_length.next_power_of_two() as f64 / duration.as_secs_f64();
    println!(
        "modular {} (2^{}): Prover completed in {:.2}s ({:.1} kHz / padded {:.1} kHz)",
        bench_name,
        scale,
        duration.as_secs_f64(),
        proving_hz / 1000.0,
        padded_proving_hz / 1000.0,
    );
    if let Some(peak) = peak_rss_bytes() {
        println!(
            "modular {} (2^{}): Peak RSS {}",
            bench_name,
            scale,
            format_memory_size(peak as f64 / BYTES_PER_GIB),
        );
    }

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
    )
}

/// A word-aligned advice buffer's balanced Dory matrix variable count.
fn advice_vars(max_advice_size_bytes: u64) -> usize {
    ((max_advice_size_bytes / 8) as usize)
        .next_power_of_two()
        .max(1)
        .ilog2() as usize
}
