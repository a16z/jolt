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
use jolt_eval::objective::performance::trace_gen::{build_trace_setup, raw_trace_cycles};
use tracer::TracerBackend;

// Ensure inline libraries are linked and auto-registered.
use jolt_inlines_keccak256 as _;
use jolt_inlines_sha2 as _;

/// Prefix identifying the measurement JSON on a stdout shared with guest
/// output. `rsplit_once` on it also strips any guest text that ran on without
/// a newline.
const JSON_MARKER: &str = "BASELINE_JSON: ";

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
    // The tracer env-dispatches to parallel mode; pin serial so
    // baselines are environment-independent.
    std::env::remove_var("TRACER_PARALLEL");
    let args = Args::parse();
    match args.guest {
        Some(name) => {
            let result = measure_named(&name, args.runs);
            // Marker-prefixed: guests share this stdout and `handle_jolt_print`
            // emits without a trailing newline, so a guest ending in `print!`
            // would otherwise prepend itself to the JSON line.
            println!("{JSON_MARKER}{result}");
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
    // Raw tracing, timed separately so the table decomposes the seam into
    // "produce cycles" and "convert them to rows". Run after the seam loop so
    // the reported peak RSS still reflects the seam's peak (raw tracing holds
    // only Vec<Cycle>, so it cannot raise the high-water mark).
    let mut raw_times = Vec::with_capacity(runs);
    for _ in 0..runs.max(1) {
        let start = std::time::Instant::now();
        raw_trace_cycles(&program, &inputs);
        raw_times.push(start.elapsed().as_secs_f64());
    }
    wall_times.sort_by(f64::total_cmp);
    raw_times.sort_by(f64::total_cmp);
    let median_wall_s = wall_times[wall_times.len() / 2];
    let median_raw_s = raw_times[raw_times.len() / 2];
    serde_json::json!({
        "guest": guest.label(),
        "rows": rows,
        "min_wall_s": wall_times[0],
        "max_wall_s": wall_times[wall_times.len() - 1],
        "median_wall_s": median_wall_s,
        "mhz": rows as f64 / median_wall_s / 1e6,
        "median_raw_s": median_raw_s,
        "raw_mhz": rows as f64 / median_raw_s / 1e6,
        "conversion_share": (median_wall_s - median_raw_s) / median_wall_s,
        "peak_rss_bytes": peak_rss_bytes(),
        "runs": wall_times.len(),
    })
}

fn orchestrate(runs: usize) {
    let exe = std::env::current_exe().expect("cannot resolve current executable");
    println!(
        "| guest | rows | seam (s) | seam spread (s) | seam MHz | raw trace (s) | raw MHz | conversion share | peak RSS (MiB) |"
    );
    println!("|---|---:|---:|---:|---:|---:|---:|---:|---:|");
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
            .filter_map(|l| l.rsplit_once(JSON_MARKER).map(|(_, json)| json))
            .next_back()
            .unwrap_or_else(|| panic!("no marked measurement line from {guest} subprocess"));
        let v: serde_json::Value = serde_json::from_str(line).expect("malformed measurement JSON");
        println!(
            "| {} | {} | {:.3} | {:.3}–{:.3} | {:.1} | {:.3} | {:.1} | {:.0}% | {} |",
            v["guest"].as_str().unwrap_or(guest),
            v["rows"],
            v["median_wall_s"].as_f64().unwrap_or(f64::NAN),
            v["min_wall_s"].as_f64().unwrap_or(f64::NAN),
            v["max_wall_s"].as_f64().unwrap_or(f64::NAN),
            v["mhz"].as_f64().unwrap_or(f64::NAN),
            v["median_raw_s"].as_f64().unwrap_or(f64::NAN),
            v["raw_mhz"].as_f64().unwrap_or(f64::NAN),
            v["conversion_share"].as_f64().unwrap_or(f64::NAN) * 100.0,
            v["peak_rss_bytes"]
                .as_f64()
                .map(|b| format!("{:.1}", b / (1024.0 * 1024.0)))
                .unwrap_or_else(|| "n/a".to_string()),
        );
    }
    println!(
        "\nReference backend, serial (`TRACER_PARALLEL` unset); median of {runs} runs per guest.\n\
         `seam` is `ExecutionBackend::trace` (raw tracing + the `Cycle` to `TraceRow` \
         conversion); `raw trace` is `tracer::trace` alone. A backend that emits `TraceRow` \
         directly skips the conversion, so its speedup over `seam` includes that share and \
         only its gain over `raw trace` is attributable to codegen. Report both when \
         grading AC8/AC9.\n\
         Peak RSS is the seam's: during conversion `Vec<Cycle>` (96 B/row) and \
         `Vec<TraceRow>` (160 B/row) coexist, so it sits above the 160 B/row floor a \
         direct-emit backend would have. AC10 comparisons should account for that rather \
         than read the difference as an allocation win.\n\
         Memory config is `GuestConfig`'s (stack 4 KiB, heap 32 KiB), not the 32 MiB heap \
         `Program::new` defaults to; this tracks `e2e_profiling.rs` and does not move the \
         numbers, but the denominator should say what it ran.\n\
         MHz is comparable only within a guest, never across rows: at these pinned inputs \
         four of the six guests are under 10^5 rows, where per-trace fixed costs (the \
         capacity reserve, ELF re-parse, device setup) are a visible share of wall time and \
         depress MHz relative to the two large guests."
    );
    println!("\n{}", provenance());
}

/// Machine and revision provenance, so a table pasted into a PR stays
/// reproducible months later.
fn provenance() -> String {
    let cores = std::thread::available_parallelism()
        .map(|n| n.get().to_string())
        .unwrap_or_else(|_| "unknown".to_string());
    format!(
        "Platform: {} {} | CPU: {} | cores: {} | commit: {}",
        std::env::consts::ARCH,
        std::env::consts::OS,
        cpu_model(),
        cores,
        commit_hash(),
    )
}

fn cpu_model() -> String {
    #[cfg(target_os = "linux")]
    {
        if let Ok(info) = std::fs::read_to_string("/proc/cpuinfo") {
            if let Some(model) = info.lines().find_map(|l| {
                l.strip_prefix("model name")
                    .and_then(|r| r.split(':').nth(1))
            }) {
                return model.trim().to_string();
            }
        }
    }
    #[cfg(target_os = "macos")]
    {
        if let Ok(out) = std::process::Command::new("sysctl")
            .args(["-n", "machdep.cpu.brand_string"])
            .output()
        {
            if out.status.success() {
                return String::from_utf8_lossy(&out.stdout).trim().to_string();
            }
        }
    }
    "unknown".to_string()
}

fn commit_hash() -> String {
    let out = std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output();
    match out {
        Ok(out) if out.status.success() => {
            let hash = String::from_utf8_lossy(&out.stdout).trim().to_string();
            let dirty = std::process::Command::new("git")
                .args(["status", "--porcelain"])
                .output()
                .map(|o| !o.stdout.is_empty())
                .unwrap_or(false);
            if dirty {
                format!("{hash} (dirty)")
            } else {
                hash
            }
        }
        _ => "unknown".to_string(),
    }
}

/// Peak resident set size of this process, in bytes, or `None` where it
/// cannot be measured.
///
/// Linux reads `VmHWM` from `/proc/self/status` (the measurement the spec
/// gates on); other unix targets fall back to `getrusage` so the harness
/// stays runnable on dev machines. WHY `Option`: a failed parse or an
/// unsupported platform must not render as `0.0 MiB`, which reads like a
/// measurement rather than the absence of one.
#[cfg(target_os = "linux")]
fn peak_rss_bytes() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    status
        .lines()
        .find_map(|line| line.strip_prefix("VmHWM:"))
        .and_then(|rest| rest.trim().strip_suffix("kB"))
        .and_then(|kb| kb.trim().parse::<u64>().ok())
        .map(|kb| kb * 1024)
}

#[cfg(all(unix, not(target_os = "linux")))]
fn peak_rss_bytes() -> Option<u64> {
    // SAFETY: getrusage only writes into the zeroed out-param we own.
    unsafe {
        let mut usage: libc::rusage = std::mem::zeroed();
        if libc::getrusage(libc::RUSAGE_SELF, &mut usage) == 0 {
            // ru_maxrss is bytes on macOS, kilobytes on other unix.
            #[cfg(target_os = "macos")]
            return Some(usage.ru_maxrss as u64);
            #[cfg(not(target_os = "macos"))]
            return Some((usage.ru_maxrss as u64) * 1024);
        }
        None
    }
}

#[cfg(not(unix))]
fn peak_rss_bytes() -> Option<u64> {
    None
}
