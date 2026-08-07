//! The profile harness behind the `profiling` feature: one documented
//! command proves a named workload on the modular stack and emits both
//! telemetry artifacts from the same span stream —
//! one per-run directory `benchmark-runs/{timestamp}_{trace_name}/` (with a
//! `latest_{trace_name}` symlink flipped to it on success) holding
//! `trace.json` (Perfetto UI / `trace_processor` SQL) and `summary.json`
//! (machine-queryable aggregates for `jolt-eval` and `jq`) — the directory
//! name carries the run identity, so the files inside use fixed names.
//!
//! ```text
//! cargo run --release -p jolt-prover --features profiling -- \
//!     profile --name fibonacci --format chrome
//! ```
//!
//! Pipeline (promoted from the retired `examples/modular_benchmark.rs`):
//! legacy-side guest compile/decode (the modular stack has no host
//! toolchain), legacy preprocessing → verifier preprocessing, modular trace
//! (`TracerBackend`), derived `ProverConfig`, `TraceBackend` witness, the
//! compiled protocol's prove over the selected backend — `dory::prove`, or
//! `akita::prove` on the packed build (artifact names gain an `_akita`
//! suffix so the two protocols' runs never collide) — and a full
//! `jolt_verifier::verify` as the correctness gate. Only `prove` is measured
//! — guest compilation, tracer execution, and preprocessing are excluded
//! from every reported metric.

#![expect(
    clippy::expect_used,
    clippy::panic,
    clippy::print_stdout,
    clippy::print_stderr,
    reason = "profile harness: fail loudly and report to stdout"
)]

use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use clap::ValueEnum;
use common::jolt_device::MemoryConfig;
#[cfg(not(feature = "akita"))]
use jolt_crypto::{Bn254G1, Pedersen};
#[cfg(not(feature = "akita"))]
use jolt_dory::DoryScheme;
#[cfg(not(feature = "akita"))]
use jolt_field::Fr;
// Keep the inline libraries linked so their host-side registrations reach the
// tracer, exactly as the legacy harness does.
use jolt_inlines_keccak256 as _;
use jolt_inlines_sha2 as _;
use jolt_profiling::summary::{finalize_trace, ProfileSummary, SummaryContext};
use jolt_profiling::{
    format_memory_size, peak_rss_bytes, report_stage_memory, setup_tracing_with_trace_path,
    TracingFormat, BYTES_PER_GIB,
};
use jolt_program::execution::{
    ExecutionBackend, JoltProgram, OwnedTrace, TraceInputs, TraceOutput, TraceRow,
};
use jolt_prover_legacy::host;
#[cfg(not(feature = "akita"))]
use jolt_prover_legacy::poly::commitment::dory::DoryCommitmentScheme;
use jolt_prover_legacy::zkvm::preprocessing::JoltSharedPreprocessing;
use jolt_prover_legacy::zkvm::program::ProgramPreprocessing as LegacyProgramPreprocessing;
#[cfg(not(feature = "akita"))]
use jolt_prover_legacy::zkvm::proof::verifier_preprocessing_from_prover;
use jolt_prover_legacy::zkvm::prover::JoltProverPreprocessing as LegacyProverPreprocessing;
#[cfg(not(feature = "akita"))]
use jolt_transcript::LegacyBlake2bTranscript as Blake2bTranscript;
use jolt_witness::{JoltVmWitnessConfig, JoltVmWitnessInputs, TraceBackend};
use tracer::execution_backend::TracerBackend;

#[cfg(not(feature = "akita"))]
use crate::JoltBackend;
use crate::{JoltProverPreprocessing, ProverConfig};

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

/// The compiled protocol's artifact-name suffix: one compiled harness proves
/// exactly one protocol, and the packed runs must never collide with the
/// homomorphic runs' trace names, `latest_` links, or CSV rows (the Dory
/// names stay unsuffixed — they are the paths existing consumers key on).
#[cfg(not(feature = "akita"))]
const PROTOCOL_SUFFIX: &str = "";
#[cfg(feature = "akita")]
const PROTOCOL_SUFFIX: &str = "_akita";

/// The run identity `modular_{workload}{protocol}_{scale}{backend}` — the
/// trace-name stem shared by the run directory, the `latest_` link (which
/// the sweep's resume check reads), and the artifact lock. The reference
/// Dory names stay bare `modular_{workload}_{scale}` — the deterministic
/// paths `jolt-eval` telemetry reads.
fn trace_name(workload: Workload, scale: u32, backend: BackendKind) -> String {
    format!(
        "modular_{}{PROTOCOL_SUFFIX}_{scale}{}",
        workload.as_str().replace('-', "_"),
        backend.trace_suffix()
    )
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

impl OutputFormat {
    /// The clap value name, for the sweep's self-exec.
    const fn as_cli_str(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::Chrome => "chrome",
            Self::None => "none",
        }
    }
}

/// Prover backend selector. `reference` is the naive test oracle (absolute
/// numbers provisional, attribution meaningful relatively); `optimized` is
/// the performance tier, slotting into the same instrumented seams. The
/// packed (akita) build supports `reference` only.
#[derive(Debug, Clone, Copy, ValueEnum, PartialEq)]
pub enum BackendKind {
    Reference,
    Optimized,
}

impl BackendKind {
    /// The canonical name — the `run.backend` value telemetry consumers key
    /// on, and the CSV identity column. Adding a backend variant forces an
    /// arm here, which keeps the summary metadata honest without a
    /// hand-maintained string elsewhere.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Reference => "reference",
            Self::Optimized => "optimized",
        }
    }

    /// Artifact-name suffix: reference keeps the bare `modular_{name}_{scale}`
    /// identity (the deterministic path `jolt-eval` telemetry reads);
    /// optimized runs get their own artifact set next to it.
    const fn trace_suffix(self) -> &'static str {
        match self {
            Self::Reference => "",
            Self::Optimized => "_optimized",
        }
    }
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

/// `benchmark` subcommand arguments: a multi-scale sweep over the workload
/// table, one `profile` subprocess per (workload, scale) — the port of the
/// retired `scripts/jolt_benchmarks.sh` (subprocess-per-run keeps the global
/// tracing subscriber and the per-run `getrusage` peak RSS correct).
#[derive(Debug, clap::Args)]
pub struct BenchmarkArgs {
    /// Workloads to sweep (comma-separated; default: all four).
    #[clap(long, value_enum, value_delimiter = ',')]
    pub benchmarks: Option<Vec<Workload>>,

    /// Smallest log2 trace length in the sweep.
    #[clap(long, default_value_t = 18)]
    pub min_scale: u32,

    /// Largest log2 trace length in the sweep (inclusive).
    #[clap(long, default_value_t = 21)]
    pub max_scale: u32,

    /// Skip (workload, scale) pairs whose `latest_` link already exists
    /// (i.e. some run of that pair completed).
    #[clap(long)]
    pub resume: bool,

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

/// Largest supported `--scale`: keeps `1usize << scale` (and the derived
/// Dory variable counts) far from shift overflow; 2^40 rows is already
/// orders of magnitude past any provable trace.
const MAX_SCALE: u32 = 40;

/// Rejects out-of-range log2 trace lengths before they wrap a shift.
fn validate_scale(scale: u32) {
    assert!(
        (1..=MAX_SCALE).contains(&scale),
        "--scale {scale} out of range: expected a log2 trace length in 1..={MAX_SCALE}"
    );
}

/// Exclusive-run guard: `benchmark-runs/{trace_name}.lock`, created with
/// `create_new` and removed on drop. Two concurrent runs of the same
/// (workload, scale) would race on the trace/summary/CSV artifact paths and
/// corrupt them silently; failing loudly is the honest alternative for a
/// deterministic-path harness.
struct RunLock(PathBuf);

impl RunLock {
    fn acquire(trace_name: &str) -> Self {
        fs::create_dir_all("benchmark-runs").expect("create benchmark-runs directory");
        let path = PathBuf::from(format!("benchmark-runs/{trace_name}.lock"));
        match fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
        {
            Ok(_) => Self(path),
            Err(e) => panic!(
                "another profile run for {trace_name} appears active ({}: {e}); \
                 if no run is alive, delete the stale lock file",
                path.display()
            ),
        }
    }
}

impl Drop for RunLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

/// Runs one profile invocation end to end. The bin's `main` is a thin
/// wrapper over this so the smoke test can call it in-process.
///
/// # Panics
///
/// Panics on any pipeline failure (harness semantics), if `--scale` is out
/// of range, if another run of the same (workload, scale) holds the
/// artifact lock, and if called twice in one process with a
/// subscriber-installing format (the global tracing subscriber can only be
/// set once).
pub fn run(args: &ProfileArgs) -> ProfileArtifacts {
    let scale = args.scale.unwrap_or_else(|| args.name.default_scale());
    validate_scale(scale);
    let trace_name = trace_name(args.name, scale, args.backend);
    let _run_lock = RunLock::acquire(&trace_name);

    // One directory per run — benchmark-runs/{timestamp}_{trace_name}/ —
    // holding every artifact the run produces; `latest_{trace_name}` is
    // flipped to it on success (the stable path consumers read, so history
    // accumulates without breaking deterministic paths).
    let run_dir = PathBuf::from(format!(
        "benchmark-runs/{}_{trace_name}",
        chrono::Utc::now().format("%Y%m%d-%H%M%S")
    ));
    fs::create_dir_all(&run_dir).expect("create run directory");

    // Per-batch heap snapshots (allocative feature): opt in before the
    // prove so the cfg-gated hooks inside `prove()` see the prefix.
    #[cfg(feature = "allocative")]
    jolt_profiling::set_flamegraph_prefix(format!("{}/", run_dir.display()));

    let trace_path = run_dir.join("trace.json");
    let guards = match args.format {
        OutputFormat::None => None,
        OutputFormat::Default => Some(setup_tracing_with_trace_path(
            &[TracingFormat::Default],
            &trace_path,
        )),
        OutputFormat::Chrome => Some(setup_tracing_with_trace_path(
            &[TracingFormat::Chrome],
            &trace_path,
        )),
    };

    run_workload(args.name, scale, args.backend, &run_dir);

    // The workload's high-water mark, sampled before the flush-time trace
    // parse/rewrite below can inflate it with tooling allocations.
    let peak_rss = peak_rss_bytes();

    // Dropping the guards flushes the chrome trace; only then can the
    // flush-time pipeline parse it.
    drop(guards);

    if args.format != OutputFormat::Chrome {
        report_stage_memory();
        update_latest_link(&trace_name, &run_dir);
        return ProfileArtifacts::default();
    }

    let ctx = SummaryContext {
        workload: args.name.as_str().to_string(),
        scale_log2: scale,
        backend: args.backend.as_str().to_string(),
    };
    let (summary_file, summary) =
        finalize_trace(&trace_path, &ctx, peak_rss).expect("finalize chrome trace");

    if let Some(root) = &summary.root {
        println!(
            "modular {} (2^{scale}): root span {:.2}s, dark time {:.1}%",
            args.name.as_str(),
            root.wall_time_ns as f64 / 1e9,
            root.dark_time_fraction * 100.0,
        );
    }
    update_latest_link(&trace_name, &run_dir);
    println!(
        "Run:     {} (-> benchmark-runs/latest_{trace_name})",
        run_dir.display()
    );
    println!("Trace:   {}", trace_path.display());
    println!("Summary: {}", summary_file.display());

    ProfileArtifacts {
        trace_path: Some(trace_path),
        summary_path: Some(summary_file),
        summary: Some(summary),
    }
}

/// Points `benchmark-runs/latest_{trace_name}` at this run's directory —
/// the stable path `jolt-eval` and the documented `jq` queries read.
/// Flipped only after the run's artifacts are complete; `jolt-eval` removes
/// the link before spawning a run, so a failed candidate can never expose a
/// previous run's artifacts.
fn update_latest_link(trace_name: &str, run_dir: &Path) {
    let link = PathBuf::from(format!("benchmark-runs/latest_{trace_name}"));
    let _ = fs::remove_file(&link);
    let target = run_dir.file_name().expect("run directory has a name");
    #[cfg(unix)]
    if let Err(e) = std::os::unix::fs::symlink(target, &link) {
        eprintln!("warning: could not update {}: {e}", link.display());
    }
    // Non-unix: no symlink; consumers fall back to globbing the newest
    // {timestamp}_{trace_name} directory (timestamps sort lexicographically).
    #[cfg(not(unix))]
    let _ = target;
}

/// Runs the multi-scale benchmark sweep: one `profile` subprocess (this same
/// executable) per (workload, scale), continuing past failures. Returns
/// `true` when every run succeeded.
///
/// Results accumulate in `benchmark-runs/modular_timings.csv`;
/// render them with `scripts/benchmark_summary.py`,
/// `scripts/plot_benchmarks.py`, and `scripts/plot_memory_usage.py`.
pub fn run_sweep(args: &BenchmarkArgs) -> bool {
    validate_scale(args.min_scale);
    validate_scale(args.max_scale);
    let workloads = args.benchmarks.clone().unwrap_or_else(|| {
        vec![
            Workload::Fibonacci,
            Workload::Sha2Chain,
            Workload::Sha3Chain,
            Workload::BTreeMap,
        ]
    });
    let exe = std::env::current_exe().expect("resolve current executable");

    let mut completed = 0u32;
    let mut skipped = 0u32;
    let mut failed: Vec<String> = Vec::new();

    for scale in args.min_scale..=args.max_scale {
        println!("=== Running benchmarks at scale 2^{scale} ===");
        for &workload in &workloads {
            let name = workload.as_str();
            let backend = args.backend.as_str();
            // A completed run flips the `latest_` link, so its presence is
            // the resume marker (dangling links read as absent).
            let latest_link = format!(
                "benchmark-runs/latest_{}",
                trace_name(workload, scale, args.backend)
            );
            if args.resume && std::path::Path::new(&latest_link).exists() {
                println!("  ⏭ Skipping {name} (found {latest_link})");
                skipped += 1;
                continue;
            }

            let scale_arg = scale.to_string();
            let command_line = format!(
                "{} profile --name {name} --scale {scale_arg} --format {} --backend {backend}",
                exe.display(),
                args.format.as_cli_str(),
            );
            let status = std::process::Command::new(&exe)
                .args([
                    "profile",
                    "--name",
                    name,
                    "--scale",
                    &scale_arg,
                    "--format",
                    args.format.as_cli_str(),
                    "--backend",
                    backend,
                ])
                .status();
            match status {
                Ok(status) if status.success() => completed += 1,
                Ok(status) => {
                    eprintln!("  ❌ FAILED ({status}): {command_line}");
                    failed.push(command_line);
                }
                Err(e) => {
                    eprintln!("  ❌ FAILED to spawn ({e}): {command_line}");
                    failed.push(command_line);
                }
            }
        }
        println!();
    }

    println!("================================================");
    println!("Benchmark sweep summary:");
    println!("  ✓ Completed: {completed}");
    if skipped > 0 {
        println!("  ⏭ Skipped: {skipped}");
    }
    if !failed.is_empty() {
        println!("  ❌ Failed: {}", failed.len());
        for command_line in &failed {
            println!("     {command_line}");
        }
    }
    println!();
    println!("Render results with:");
    println!("  python3 scripts/benchmark_summary.py");
    println!("  python3 scripts/plot_benchmarks.py");
    println!("  python3 scripts/plot_memory_usage.py");

    failed.is_empty()
}

/// One proved workload, as the reporting tail consumes it. `proof_size` is
/// `None` on the packed build: the upstream akita field type has no serde
/// support at the pinned revision, so the packed proof has no byte encoding
/// to measure yet (the CSV rows carry 0 as an explicit placeholder).
struct ProvenRun {
    duration: std::time::Duration,
    proof_size: Option<usize>,
}

fn run_workload(workload: Workload, scale: u32, backend: BackendKind, run_dir: &Path) {
    let bench_name = workload.as_str();
    let backend_label = backend.as_str();
    let max_trace_length = 1usize << scale;
    let bench_target = (max_trace_length as f64 * SAFETY_MARGIN) as usize;
    tracing::info!("Running modular {bench_name} profile at scale 2^{scale}");

    let input = workload.input(bench_target);

    // --- Guest (unmeasured): compiled/decoded through the legacy host
    // toolchain (the modular stack has none of its own).
    let mut program = host::Program::new(&format!("{bench_name}-guest"));
    let (_, legacy_trace, _, io_device) = program.trace(&input, &[], &[]);
    assert!(
        legacy_trace.len().next_power_of_two() <= max_trace_length,
        "Trace is longer than expected"
    );
    drop(legacy_trace);
    let elf_contents = program.get_elf_contents().expect("elf contents");
    let memory_layout = io_device.memory_layout.clone();
    let jolt_program = Arc::new(JoltProgram::from_elf_bytes(elf_contents));

    // --- Modular trace (unmeasured, like legacy's `gen_from_elf` emulation).
    let trace_output = trace_modular(&jolt_program, &memory_layout, &input);
    let trace_length = trace_output.trace.rows().len();

    // --- The compiled protocol's preprocessing + prove + verify.
    let run = prove_workload(
        &mut program,
        &jolt_program,
        &memory_layout,
        max_trace_length,
        trace_output,
        backend,
    );
    let (duration, proof_size) = (run.duration, run.proof_size.unwrap_or(0));
    if run.proof_size.is_none() {
        println!(
            "modular {bench_name} (2^{scale}): proof size unavailable \
             (no packed-proof byte encoding yet); CSV carries 0"
        );
    }

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

    // The legacy harness's 7 CSV fields plus a trailing backend column, in
    // the run directory. Field 7 (`proof_size_compressed`)
    // duplicates the raw size exactly as legacy does — its
    // `prove_example_with_trace` returns `proof_size` for both fields, the
    // compressed encoding having been retired — so the columns stay
    // directly comparable across the two harnesses.
    let summary_line = format!(
        "{}{PROTOCOL_SUFFIX},{},{:.2},{},{:.2},{},{},{backend_label}\n",
        bench_name,
        scale,
        duration.as_secs_f64(),
        trace_length.next_power_of_two(),
        padded_proving_hz,
        proof_size,
        proof_size,
    );
    let individual_file = run_dir.join("timings.csv");
    if let Err(e) = fs::write(&individual_file, &summary_line) {
        eprintln!(
            "Failed to write individual result file {}: {e}",
            individual_file.display()
        );
    }
    // Header on creation: the summary/plot scripts read this by column name.
    // Cross-run by nature, so it lives at the benchmark-runs root rather
    // than inside any run directory.
    let consolidated = "benchmark-runs/modular_timings.csv";
    let line = if std::path::Path::new(consolidated).exists() {
        summary_line
    } else {
        format!(
            "benchmark_name,scale,prover_time_s,trace_length,proving_hz,\
             proof_size,proof_size_compressed,backend\n{summary_line}"
        )
    };
    if let Err(e) = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(consolidated)
        .and_then(|mut f| f.write_all(line.as_bytes()))
    {
        eprintln!("Failed to write consolidated timing: {e}");
    }
}

/// The homomorphic (Dory) arm: legacy preprocessing → verifier
/// preprocessing, derived config, `TraceBackend` witness, RLC setup, and
/// `dory::prove` over the selected backend + `jolt_verifier::verify` —
/// exactly as in the byte-diff tests. Only the prove is measured.
#[cfg(not(feature = "akita"))]
fn prove_workload(
    program: &mut host::Program,
    jolt_program: &Arc<JoltProgram>,
    memory_layout: &common::jolt_device::MemoryLayout,
    max_trace_length: usize,
    trace_output: TraceOutput<OwnedTrace>,
    backend: BackendKind,
) -> ProvenRun {
    let (bytecode, init_memory_state, _, entry_address) = program.decode();
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
        .as_full_arc()
        .expect("full program preprocessing");

    let config = ProverConfig::derive::<Fr>(
        trace_output.trace.rows(),
        memory_layout,
        verifier_preprocessing.program.min_bytecode_address(),
        verifier_preprocessing.program.program_image_len_words(),
        max_trace_length,
    )
    .expect("derive config");
    let public_io = trace_output.device.clone();
    let padded_output = pad_trace(trace_output, config.trace_length);

    let witness = Arc::new(TraceBackend::new(
        JoltVmWitnessConfig::new(
            config.trace_length.ilog2() as usize,
            config.ram_K,
            config.one_hot_config,
        ),
        JoltVmWitnessInputs::new(jolt_program, &program_preprocessing, padded_output),
    ));

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
        BackendKind::Optimized => JoltBackend::<Fr, DoryScheme>::optimized(),
    };

    // --- The measured window: the full modular prove (witness
    // materialization, commitment, all sumcheck stages, joint opening). The
    // `jolt_prover::prove` root span covers exactly this interval; the
    // Instant is the `--format none` no-subscriber baseline.
    let now = Instant::now();
    let proof = crate::dory::prove::<Fr, DoryScheme, Pedersen<Bn254G1>, Blake2bTranscript, _>(
        &backend,
        &prover_preprocessing,
        &config,
        None,
        Arc::clone(&witness),
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

    ProvenRun {
        duration,
        proof_size: Some(proof_size),
    }
}

/// The packed (Akita) arm: packed legacy preprocessing, the transparent
/// `OneHotTrace` setup derived from the config + program shape (no legacy
/// prover instance), and `akita::prove` + `jolt_verifier::verify`. Only the
/// prove is measured. Reference backend only — the packed path has no
/// optimized kernel tier yet.
#[cfg(feature = "akita")]
fn prove_workload(
    program: &mut host::Program,
    jolt_program: &Arc<JoltProgram>,
    memory_layout: &common::jolt_device::MemoryLayout,
    max_trace_length: usize,
    trace_output: TraceOutput<OwnedTrace>,
    backend: BackendKind,
) -> ProvenRun {
    use jolt_openings::CommitmentScheme as VerifierCommitmentScheme;
    use jolt_prover_legacy::zkvm::packed::{
        akita_verifier_preprocessing, AkitaField, AkitaPackedScheme, AkitaScheme, AkitaTranscript,
        AkitaVc,
    };

    let backend = match backend {
        BackendKind::Reference => crate::akita::JoltAkitaBackend::reference(),
        BackendKind::Optimized => panic!(
            "--backend optimized is not available on the packed (akita) build: \
             the packed path has no optimized kernel tier yet"
        ),
    };

    let (bytecode, init_memory_state, _, entry_address) = program.decode();
    let program_data =
        LegacyProgramPreprocessing::preprocess(bytecode, init_memory_state, entry_address)
            .expect("legacy preprocess");
    let shared_preprocessing: JoltSharedPreprocessing<AkitaPackedScheme> =
        JoltSharedPreprocessing::new(program_data, memory_layout.clone(), max_trace_length);
    let legacy_preprocessing = LegacyProverPreprocessing::new(shared_preprocessing);

    let config = ProverConfig::derive::<AkitaField>(
        trace_output.trace.rows(),
        memory_layout,
        legacy_preprocessing
            .shared
            .program_meta
            .min_bytecode_address,
        legacy_preprocessing
            .shared
            .program
            .program_image_len_words(),
        max_trace_length,
    )
    .expect("derive config");

    // The transparent OneHotTrace setup, from the config + program shape.
    let (setup_shape, layout_digest, one_hot_k) = crate::akita::one_hot_trace_setup_shape(
        &config,
        legacy_preprocessing.shared.bytecode_size(),
    )
    .expect("OneHotTrace setup shape");
    let params = <<AkitaScheme as VerifierCommitmentScheme>::SetupParams>::one_hot_only(
        setup_shape.num_vars,
        setup_shape.num_polys,
        layout_digest,
        one_hot_k,
    );
    let (object_setup, verifier_setup) = <AkitaScheme as VerifierCommitmentScheme>::setup(params)
        .expect("the transparent packed setup must derive");
    let verifier_preprocessing =
        akita_verifier_preprocessing(&legacy_preprocessing, verifier_setup, None);
    let program_preprocessing = verifier_preprocessing
        .program
        .as_full_arc()
        .expect("full program preprocessing");

    let public_io = trace_output.device.clone();
    let padded_output = pad_trace(trace_output, config.trace_length);
    let witness = TraceBackend::new(
        JoltVmWitnessConfig::new(
            config.trace_length.ilog2() as usize,
            config.ram_K,
            config.one_hot_config,
        ),
        JoltVmWitnessInputs::new(jolt_program, &program_preprocessing, padded_output),
    );
    let prover_preprocessing = JoltProverPreprocessing::<AkitaScheme, AkitaVc> {
        verifier: verifier_preprocessing,
        pcs_setup: object_setup,
        committed_program: None,
    };

    // --- The measured window: the full packed prove (OneHotTrace assembly
    // and native commit, all sumcheck stages, reconstruction, the native
    // same-point opening). The `jolt_prover::prove` root span covers exactly
    // this interval; the Instant is the `--format none` baseline.
    let now = Instant::now();
    let proof = crate::akita::prove::<AkitaField, AkitaScheme, AkitaVc, AkitaTranscript, _>(
        &backend,
        &prover_preprocessing,
        &config,
        None,
        None,
        &witness,
        &public_io,
    )
    .expect("modular packed prove");
    let duration = now.elapsed();

    // --- Correctness gate (unmeasured): the proof must verify.
    jolt_verifier::verify::<AkitaField, AkitaScheme, AkitaVc, AkitaTranscript>(
        &prover_preprocessing.verifier,
        &public_io,
        &proof,
        None,
    )
    .expect("modular packed proof verifies");

    ProvenRun {
        duration,
        // The packed proof has no byte encoding to measure: the upstream
        // akita field type carries no serde support at the pinned revision.
        proof_size: None,
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
    let source = trace_output.trace.rows();
    let mut rows = Vec::with_capacity(trace_length.max(source.len()));
    rows.extend_from_slice(source);
    rows.resize(trace_length, TraceRow::default());
    TraceOutput::new(
        OwnedTrace::new(rows),
        trace_output.device,
        trace_output.final_memory,
        trace_output.advice_tape,
    )
}

/// A word-aligned advice buffer's balanced Dory matrix variable count.
#[cfg(not(feature = "akita"))]
fn advice_vars(max_advice_size_bytes: u64) -> usize {
    ((max_advice_size_bytes / 8) as usize)
        .next_power_of_two()
        .max(1)
        .ilog2() as usize
}
