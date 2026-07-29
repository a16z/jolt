//! Trace-generation baseline harness: eager reference-backend throughput and
//! peak RSS per profile guest (`specs/x86-tracer-backend.md`, AC1/AC10).
//!
//! Modes:
//! - no args: re-spawn self once per guest (subprocess-per-measurement, so
//!   peak RSS is isolated per guest) and print a markdown baseline table.
//! - `--guest <name>`: measure one guest in this process and print one JSON
//!   line (the subprocess mode).

use clap::Parser;
use jolt_eval::guests::{BTreeMapOps, Fibonacci, GuestConfig, Sha2, Sha2Chain, Sha3, Sha3Chain};
use jolt_eval::objective::performance::trace_gen::build_trace_setup;
use tracer::TracerBackend;

// Ensure inline libraries are linked and auto-registered.
use jolt_inlines_keccak256 as _;
use jolt_inlines_sha2 as _;

const GUESTS: &[&str] = &[
    "fibonacci",
    "sha2",
    "sha3",
    "sha2-chain",
    "sha3-chain",
    "btreemap",
];

#[derive(Parser)]
struct Args {
    /// Measure a single guest in-process and print one JSON line.
    #[arg(long)]
    guest: Option<String>,
    /// Timed trace repetitions per guest (median wall-clock reported).
    #[arg(long, default_value_t = 5)]
    runs: usize,
}

fn main() {
    let args = Args::parse();
    match args.guest {
        Some(name) => {
            let result = measure_named(&name, args.runs);
            println!("{result}");
        }
        None => orchestrate(args.runs),
    }
}

fn measure_named(name: &str, runs: usize) -> serde_json::Value {
    match name {
        "fibonacci" => measure(&Fibonacci(400000), runs),
        "sha2" => measure(&Sha2::default(), runs),
        "sha3" => measure(&Sha3::default(), runs),
        "sha2-chain" => measure(&Sha2Chain::profiling_default(), runs),
        "sha3-chain" => measure(&Sha3Chain::default(), runs),
        "btreemap" => measure(&BTreeMapOps::default(), runs),
        other => panic!("unknown guest: {other} (expected one of {GUESTS:?})"),
    }
}

fn measure<G: GuestConfig>(guest: &G, runs: usize) -> serde_json::Value {
    let (program, inputs) = build_trace_setup(guest);
    let mut wall_times = Vec::with_capacity(runs);
    let mut rows = 0usize;
    for _ in 0..runs.max(1) {
        let mut backend = TracerBackend::new();
        let start = std::time::Instant::now();
        let output = program
            .trace_with(&mut backend, inputs.clone())
            .expect("reference trace failed");
        wall_times.push(start.elapsed().as_secs_f64());
        rows = output.trace.rows().len();
    }
    wall_times.sort_by(f64::total_cmp);
    let median_wall_s = wall_times[wall_times.len() / 2];
    serde_json::json!({
        "guest": guest.label(),
        "rows": rows,
        "median_wall_s": median_wall_s,
        "mhz": rows as f64 / median_wall_s / 1e6,
        "peak_rss_bytes": peak_rss_bytes(),
        "runs": wall_times.len(),
    })
}

fn orchestrate(runs: usize) {
    let exe = std::env::current_exe().expect("cannot resolve current executable");
    println!("| guest | rows | median wall (s) | MHz | peak RSS (MiB) |");
    println!("|---|---:|---:|---:|---:|");
    for guest in GUESTS {
        let output = std::process::Command::new(&exe)
            .args(["--guest", guest, "--runs", &runs.to_string()])
            .output()
            .expect("failed to spawn measurement subprocess");
        if !output.status.success() {
            eprintln!("{}", String::from_utf8_lossy(&output.stderr));
            panic!("measurement subprocess for {guest} failed");
        }
        let stdout = String::from_utf8_lossy(&output.stdout);
        let line = stdout
            .lines()
            .last()
            .unwrap_or_else(|| panic!("no output from {guest} subprocess"));
        let v: serde_json::Value = serde_json::from_str(line).expect("malformed measurement JSON");
        println!(
            "| {} | {} | {:.3} | {:.3} | {:.1} |",
            v["guest"].as_str().unwrap_or(guest),
            v["rows"],
            v["median_wall_s"].as_f64().unwrap_or(f64::NAN),
            v["mhz"].as_f64().unwrap_or(f64::NAN),
            v["peak_rss_bytes"].as_f64().unwrap_or(f64::NAN) / (1024.0 * 1024.0),
        );
    }
    println!(
        "\nReference backend, eager `ExecutionBackend::trace`; median of {runs} runs per guest. \
         Platform: {} {}.",
        std::env::consts::ARCH,
        std::env::consts::OS
    );
}

/// Peak resident set size of this process, in bytes.
///
/// Linux reads `VmHWM` from `/proc/self/status` (the measurement the spec
/// gates on); other unix targets fall back to `getrusage` so the harness
/// stays runnable on dev machines.
#[cfg(target_os = "linux")]
fn peak_rss_bytes() -> u64 {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    status
        .lines()
        .find_map(|line| line.strip_prefix("VmHWM:"))
        .and_then(|rest| rest.trim().strip_suffix("kB"))
        .and_then(|kb| kb.trim().parse::<u64>().ok())
        .map(|kb| kb * 1024)
        .unwrap_or(0)
}

#[cfg(all(unix, not(target_os = "linux")))]
fn peak_rss_bytes() -> u64 {
    // SAFETY: getrusage only writes into the zeroed out-param we own.
    unsafe {
        let mut usage: libc::rusage = std::mem::zeroed();
        if libc::getrusage(libc::RUSAGE_SELF, &mut usage) == 0 {
            // ru_maxrss is bytes on macOS, kilobytes on other unix.
            #[cfg(target_os = "macos")]
            return usage.ru_maxrss as u64;
            #[cfg(not(target_os = "macos"))]
            return (usage.ru_maxrss as u64) * 1024;
        }
        0
    }
}

#[cfg(not(unix))]
fn peak_rss_bytes() -> u64 {
    0
}
