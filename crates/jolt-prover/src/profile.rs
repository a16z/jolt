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
//! Pipeline: guest compile/decode, modular preprocessing and trace
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
// Keep the inline libraries linked so their host-side registrations reach the tracer.
use jolt_host::{JoltProgramSource, Program};
use jolt_inlines_keccak256 as _;
use jolt_inlines_sha2 as _;
use jolt_profiling::summary::{finalize_trace, ProfileSummary, SummaryContext};
use jolt_profiling::{
    format_memory_size, peak_rss_bytes, report_stage_memory, setup_tracing_with_trace_path,
    TracingFormat, BYTES_PER_GIB,
};
use jolt_program::execution::{JoltProgram, OwnedTrace, TraceInputs, TraceOutput};
use jolt_program::preprocess::{BytecodePreprocessing, JoltProgramPreprocessing};
use jolt_riscv::JoltTraceRow;
#[cfg(not(feature = "akita"))]
use jolt_transcript::LegacyBlake2bTranscript as Blake2bTranscript;
use jolt_witness::{JoltVmWitnessConfig, JoltVmWitnessInputs, TraceBackend};
use tracer::execution_backend::TracerBackend;

#[cfg(not(feature = "akita"))]
use crate::JoltBackend;
#[cfg(not(feature = "akita"))]
use crate::JoltSharedPreprocessing;
use crate::ProverConfig;

// Empirically measured RV64IMAC cycles per operation. These values keep
// current benchmark scales comparable with prior results.
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

    /// The guest input targeting `target` trace cycles.
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
/// the performance tier, slotting into the same instrumented seams.
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

    // --- Guest compilation and trace sizing (unmeasured).
    let mut program = Program::new(&format!("{bench_name}-guest"));
    let (_, sizing_trace, _, io_device) = program.trace(&input, &[], &[]);
    assert!(
        sizing_trace.len().next_power_of_two() <= max_trace_length,
        "Trace is longer than expected"
    );
    drop(sizing_trace);
    let memory_layout = io_device.memory_layout.clone();
    let jolt_program = Arc::new(program.build_jolt_program().expect("build Jolt program"));
    let program_preprocessing = JoltProgramPreprocessing::new(
        jolt_program.expanded_bytecode.clone(),
        jolt_program.memory_init.clone(),
        memory_layout.clone(),
        jolt_program.entry_address,
        max_trace_length,
        program.instruction_profile(),
    )
    .expect("program preprocessing");

    // --- Modular trace (unmeasured).
    let trace_output = trace_modular(
        &jolt_program,
        &memory_layout,
        &program_preprocessing.bytecode,
        &input,
    );
    let trace_length = trace_output.trace.len();

    // --- The compiled protocol's preprocessing + prove + verify.
    let run = prove_workload(&jolt_program, program_preprocessing, trace_output, backend);
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

    // The CSV has seven historical fields plus the backend column. With no
    // compressed encoding, field 7 repeats the raw proof size.
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

/// The homomorphic (Dory) arm. Only the proof is measured.
#[cfg(not(feature = "akita"))]
fn prove_workload(
    jolt_program: &Arc<JoltProgram>,
    program_preprocessing: JoltProgramPreprocessing,
    trace_output: TraceOutput<Arc<Vec<JoltTraceRow>>>,
    backend: BackendKind,
) -> ProvenRun {
    let memory_layout = program_preprocessing.memory_layout.clone();
    let max_trace_length = program_preprocessing.max_padded_trace_length;

    let config = ProverConfig::derive_compact::<Fr>(
        trace_output.trace.as_slice(),
        &memory_layout,
        program_preprocessing.ram.min_bytecode_address,
        program_preprocessing.ram.bytecode_words.len(),
        max_trace_length,
    )
    .expect("derive config");
    let shared_preprocessing =
        JoltSharedPreprocessing::new(program_preprocessing).expect("shared preprocessing");
    let prover_preprocessing = crate::dory::from_shared(shared_preprocessing);
    let program_preprocessing = prover_preprocessing
        .program_arc()
        .expect("full program preprocessing");
    let public_io = trace_output.device.clone();
    let witness = Arc::new(TraceBackend::<OwnedTrace>::from_compact(
        JoltVmWitnessConfig::new(
            config.trace_length.ilog2() as usize,
            config.ram_K,
            config.one_hot_config,
        ),
        JoltVmWitnessInputs::new(jolt_program, &program_preprocessing, trace_output),
    ));

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
        witness.as_ref(),
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

/// The packed (Akita) arm. Only the proof is measured.
#[cfg(feature = "akita")]
fn prove_workload(
    jolt_program: &Arc<JoltProgram>,
    program_preprocessing: JoltProgramPreprocessing,
    trace_output: TraceOutput<Arc<Vec<JoltTraceRow>>>,
    backend: BackendKind,
) -> ProvenRun {
    use crate::akita::preprocessing::{preprocess_full, AkitaTranscript, AkitaVc};
    use jolt_akita::{AkitaField, AkitaScheme};

    let backend = match backend {
        BackendKind::Reference => crate::akita::JoltAkitaBackend::reference(),
        BackendKind::Optimized => crate::akita::JoltAkitaBackend::optimized(),
    };

    let memory_layout = program_preprocessing.memory_layout.clone();
    let max_trace_length = program_preprocessing.max_padded_trace_length;

    let config = ProverConfig::derive_compact::<AkitaField>(
        trace_output.trace.as_slice(),
        &memory_layout,
        program_preprocessing.ram.min_bytecode_address,
        program_preprocessing.ram.bytecode_words.len(),
        max_trace_length,
    )
    .expect("derive config");
    let prover_preprocessing =
        preprocess_full(program_preprocessing, &config).expect("Akita preprocessing");
    let program_preprocessing = prover_preprocessing
        .program_arc()
        .expect("full program preprocessing");

    let public_io = trace_output.device.clone();
    let witness = TraceBackend::<OwnedTrace>::from_compact(
        JoltVmWitnessConfig::new(
            config.trace_length.ilog2() as usize,
            config.ram_K,
            config.one_hot_config,
        ),
        JoltVmWitnessInputs::new(jolt_program, &program_preprocessing, trace_output),
    );
    // --- The measured window: the full packed prove (OneHotTrace assembly
    // and native commit, all sumcheck stages, and the native grouped opening).
    // The `jolt_prover::prove` root span covers exactly
    // this interval; the Instant is the `--format none` baseline.
    let now = Instant::now();
    let proof = crate::akita::prove::<AkitaField, AkitaScheme, AkitaVc, AkitaTranscript, _>(
        &backend,
        &prover_preprocessing,
        &config,
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

/// Trace the guest through the modular stack (`TracerBackend`).
fn trace_modular(
    program: &JoltProgram,
    memory_layout: &common::jolt_device::MemoryLayout,
    bytecode: &BytecodePreprocessing,
    inputs: &[u8],
) -> TraceOutput<Arc<Vec<JoltTraceRow>>> {
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
        .trace_compact(
            program,
            TraceInputs {
                inputs: inputs.to_vec(),
                untrusted_advice: Vec::new(),
                trusted_advice: Vec::new(),
                memory_config,
                advice_tape: None,
            },
            bytecode,
        )
        .expect("modular trace")
}
