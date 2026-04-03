#![allow(clippy::print_stdout, clippy::print_stderr, clippy::undocumented_unsafe_blocks)]
//! Fast iteration benchmark for jolt-cpu kernel optimization.
//!
//! Runs in ~15-20s total. Outputs JSON to stdout, human-readable table to stderr.
//! Auto-calibrates iteration counts to fill a target time window per benchmark.
//!
//! Usage:
//!   cargo bench -p jolt-cpu --bench quick_bench -- [OPTIONS]
//!
//! Options:
//!   --json              Output JSON to stdout (default: human table to stdout)
//!   --baseline <path>   Compare against a baseline JSON file
//!   --filter <pattern>  Only run benchmarks matching pattern

use std::time::Instant;

use jolt_compiler::kernel_spec::Iteration;
use jolt_compiler::{BindingOrder, Factor, Formula, KernelSpec, ProductTerm};
use jolt_compute::{Buf, ComputeBackend, DeviceBuffer};
use jolt_cpu::{compile, CpuBackend};
use jolt_field::{Field, FieldAccumulator, Fr};
use num_traits::{One, Zero};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Target measurement time per benchmark.
const TARGET_SECS: f64 = 1.0;

/// Minimum iterations to run (for statistical stability).
const MIN_ITERS: u64 = 5;

/// Size for reduce/bind benchmarks.
const LOG_N: usize = 18;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn random_field_vec(n: usize, seed: u64) -> Vec<Fr> {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    (0..n).map(|_| Fr::random(&mut rng)).collect()
}

fn product_sum_formula(d: usize) -> Formula {
    let terms: Vec<_> = (0..1)
        .map(|g| ProductTerm {
            coefficient: 1,
            factors: (0..d).map(|j| Factor::Input((g * d + j) as u32)).collect(),
        })
        .collect();
    Formula::from_terms(terms)
}

// ---------------------------------------------------------------------------
// Measurement
// ---------------------------------------------------------------------------

struct BenchResult {
    name: String,
    iterations: u64,
    total_ns: u64,
    ops_per_iter: u64,
    ns_per_op: f64,
    throughput_mops: f64,
}

/// Auto-calibrate and measure. `setup` is called once, `body` is the hot loop.
/// `ops_per_iter` is how many logical operations one iteration performs.
fn measure<S, F>(name: &str, ops_per_iter: u64, setup: impl FnOnce() -> S, body: F) -> BenchResult
where
    F: Fn(&mut S),
{
    let mut state = setup();

    // Warmup: single iteration to prime caches and JIT.
    body(&mut state);

    // Calibrate: how many iterations to fill TARGET_SECS?
    let cal_start = Instant::now();
    body(&mut state);
    let single_ns = cal_start.elapsed().as_nanos().max(1) as f64;
    let target_iters = ((TARGET_SECS * 1e9) / single_ns).ceil() as u64;
    let iters = target_iters.max(MIN_ITERS);

    // Measure
    let start = Instant::now();
    for _ in 0..iters {
        body(&mut state);
    }
    let elapsed = start.elapsed();
    let total_ns = elapsed.as_nanos() as u64;
    let total_ops = iters * ops_per_iter;
    let ns_per_op = total_ns as f64 / total_ops as f64;
    let throughput_mops = total_ops as f64 / (total_ns as f64 / 1e9) / 1e6;

    BenchResult {
        name: name.to_string(),
        iterations: iters,
        total_ns,
        ops_per_iter,
        ns_per_op,
        throughput_mops,
    }
}

/// Like `measure`, but calls `reset` before each iteration to restore state.
/// Only the `body` is timed, not the reset.
fn measure_with_reset<S, R, B>(
    name: &str,
    ops_per_iter: u64,
    setup: impl FnOnce() -> S,
    reset: R,
    body: B,
) -> BenchResult
where
    R: Fn(&mut S),
    B: Fn(&mut S),
{
    let mut state = setup();

    // Warmup
    reset(&mut state);
    body(&mut state);

    // Calibrate
    reset(&mut state);
    let cal_start = Instant::now();
    body(&mut state);
    let single_ns = cal_start.elapsed().as_nanos().max(1) as f64;
    let target_iters = ((TARGET_SECS * 1e9) / single_ns).ceil() as u64;
    let iters = target_iters.max(MIN_ITERS);

    // Measure: time only body, not reset
    let mut total_ns = 0u64;
    for _ in 0..iters {
        reset(&mut state);
        let start = Instant::now();
        body(&mut state);
        total_ns += start.elapsed().as_nanos() as u64;
    }
    let total_ops = iters * ops_per_iter;
    let ns_per_op = total_ns as f64 / total_ops as f64;
    let throughput_mops = total_ops as f64 / (total_ns as f64 / 1e9) / 1e6;

    BenchResult {
        name: name.to_string(),
        iterations: iters,
        total_ns,
        ops_per_iter,
        ns_per_op,
        throughput_mops,
    }
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

fn bench_field_mul() -> BenchResult {
    let ops = 10_000u64;
    measure(
        "field_mul",
        ops,
        || {
            let mut rng = ChaCha20Rng::seed_from_u64(42);
            let a: Vec<Fr> = (0..ops as usize).map(|_| Fr::random(&mut rng)).collect();
            let b: Vec<Fr> = (0..ops as usize).map(|_| Fr::random(&mut rng)).collect();
            (a, b, Fr::zero())
        },
        |(a, b, sink)| {
            for i in 0..a.len() {
                *sink = unsafe { std::hint::black_box(a.get_unchecked(i) * b.get_unchecked(i)) };
            }
        },
    )
}

fn bench_field_add() -> BenchResult {
    let ops = 10_000u64;
    measure(
        "field_add",
        ops,
        || {
            let mut rng = ChaCha20Rng::seed_from_u64(43);
            let a: Vec<Fr> = (0..ops as usize).map(|_| Fr::random(&mut rng)).collect();
            let b: Vec<Fr> = (0..ops as usize).map(|_| Fr::random(&mut rng)).collect();
            (a, b, Fr::zero())
        },
        |(a, b, sink)| {
            for i in 0..a.len() {
                *sink = unsafe { std::hint::black_box(*a.get_unchecked(i) + *b.get_unchecked(i)) };
            }
        },
    )
}

fn bench_fmadd_one() -> BenchResult {
    let ops = 10_000u64;
    measure(
        "fmadd_one",
        ops,
        || {
            let mut rng = ChaCha20Rng::seed_from_u64(44);
            let vals: Vec<Fr> = (0..ops as usize).map(|_| Fr::random(&mut rng)).collect();
            let one = Fr::one();
            let acc = <Fr as Field>::Accumulator::default();
            (vals, one, acc)
        },
        |(vals, one, acc)| {
            *acc = <Fr as Field>::Accumulator::default();
            for v in vals.iter() {
                acc.fmadd(*one, *v);
            }
            let _ = std::hint::black_box(&acc);
        },
    )
}

fn bench_toom_eval(d: usize) -> BenchResult {
    let ops = 10_000u64;
    let name = format!("toom{d}_eval");
    measure(
        &name,
        ops,
        || {
            let mut rng = ChaCha20Rng::seed_from_u64(100 + d as u64);
            let lo: Vec<Fr> = (0..d).map(|_| Fr::random(&mut rng)).collect();
            let hi: Vec<Fr> = (0..d).map(|_| Fr::random(&mut rng)).collect();
            let formula = product_sum_formula(d);
            let spec = KernelSpec::new(formula, Iteration::Dense, BindingOrder::LowToHigh);
            let kernel = compile::<Fr>(&spec);
            let out = vec![Fr::zero(); d];
            (lo, hi, kernel, out)
        },
        |(lo, hi, kernel, out)| {
            for _ in 0..ops {
                for slot in out.iter_mut() {
                    *slot = Fr::zero();
                }
                kernel.evaluate(lo, hi, &[], out);
            }
            let _ = std::hint::black_box(&out);
        },
    )
}

fn bench_dense_reduce(d: usize) -> BenchResult {
    let backend = CpuBackend;
    let n = 1usize << LOG_N;
    let pairs = (n / 2) as u64;
    let name = format!("dense_reduce_d{d}");
    measure(
        &name,
        pairs,
        || {
            let formula = product_sum_formula(d);
            let spec = KernelSpec::new(formula, Iteration::Dense, BindingOrder::LowToHigh);
            let kernel = compile::<Fr>(&spec);
            let bufs: Vec<Buf<CpuBackend, Fr>> = (0..d)
                .map(|i| DeviceBuffer::Field(random_field_vec(n, 500 + i as u64)))
                .collect();
            (kernel, bufs)
        },
        |(kernel, bufs)| {
            let buf_refs: Vec<&Buf<CpuBackend, Fr>> = bufs.iter().collect();
            let _ = std::hint::black_box(backend.reduce(kernel, &buf_refs, &[]));
        },
    )
}

fn bench_bind(order: BindingOrder) -> BenchResult {
    let backend = CpuBackend;
    let n = 1usize << LOG_N;
    let elems = (n / 2) as u64;
    let tag = match order {
        BindingOrder::LowToHigh => "l2h",
        BindingOrder::HighToLow => "h2l",
    };
    let name = format!("bind_{tag}");

    let source: Vec<Vec<Fr>> = (0..4)
        .map(|i| random_field_vec(n, 800 + i))
        .collect();
    let mut rng = ChaCha20Rng::seed_from_u64(700);
    let scalar = Fr::random(&mut rng);

    measure_with_reset(
        &name,
        elems,
        || {
            let formula = product_sum_formula(4);
            let spec = KernelSpec::new(formula, Iteration::Dense, order);
            let kernel = compile::<Fr>(&spec);
            let bufs: Vec<Buf<CpuBackend, Fr>> = source
                .iter()
                .map(|v| DeviceBuffer::Field(v.clone()))
                .collect();
            (kernel, bufs, source.clone(), scalar)
        },
        // Reset: restore buffers from source (clone cost excluded from timing)
        |(_, bufs, source, _)| {
            for (buf, src) in bufs.iter_mut().zip(source.iter()) {
                let field_buf = buf.as_field_mut();
                field_buf.resize(src.len(), Fr::zero());
                field_buf.copy_from_slice(src);
            }
        },
        // Body: only the bind is timed
        |(kernel, bufs, _, scalar)| {
            backend.bind(kernel, bufs, *scalar);
            let _ = std::hint::black_box(&bufs);
        },
    )
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

fn print_table(results: &[BenchResult], baseline: Option<&Vec<BenchResult>>) {
    eprintln!(
        "{:<22} {:>10} {:>12} {:>12} {}",
        "Benchmark", "ns/op", "Mops/s", "iters",
        if baseline.is_some() { "   delta" } else { "" }
    );
    eprintln!("{}", "-".repeat(if baseline.is_some() { 72 } else { 60 }));
    for r in results {
        let delta = baseline.and_then(|bl| {
            bl.iter()
                .find(|b| b.name == r.name)
                .map(|b| (r.ns_per_op - b.ns_per_op) / b.ns_per_op * 100.0)
        });
        eprintln!(
            "{:<22} {:>10.2} {:>12.2} {:>12}{}",
            r.name,
            r.ns_per_op,
            r.throughput_mops,
            r.iterations,
            delta.map_or_else(String::new, |d| format!("  {:>+7.1}%", d))
        );
    }
}

fn to_json(results: &[BenchResult]) -> String {
    let git_sha = std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .unwrap_or_default()
        .trim()
        .to_string();

    let git_dirty = std::process::Command::new("git")
        .args(["diff", "--quiet", "HEAD"])
        .status()
        .map(|s| !s.success())
        .unwrap_or(true);

    let rayon_threads = rayon::current_num_threads();
    let timestamp = chrono_lite_now();

    let benchmarks: Vec<String> = results
        .iter()
        .map(|r| {
            format!(
                r#"    "{}": {{
      "iterations": {},
      "total_ns": {},
      "ops_per_iter": {},
      "ns_per_op": {:.2},
      "throughput_mops": {:.2}
    }}"#,
                r.name, r.iterations, r.total_ns, r.ops_per_iter, r.ns_per_op, r.throughput_mops
            )
        })
        .collect();

    format!(
        r#"{{
  "metadata": {{
    "timestamp": "{}",
    "git_sha": "{}",
    "git_dirty": {},
    "rayon_threads": {},
    "log_n": {}
  }},
  "benchmarks": {{
{}
  }}
}}"#,
        timestamp,
        git_sha,
        git_dirty,
        rayon_threads,
        LOG_N,
        benchmarks.join(",\n")
    )
}

/// Minimal timestamp without pulling in chrono.
fn chrono_lite_now() -> String {
    let output = std::process::Command::new("date")
        .args(["+%Y-%m-%dT%H:%M:%S"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .unwrap_or_default();
    output.trim().to_string()
}

fn parse_baseline(path: &str) -> Option<Vec<BenchResult>> {
    let content = std::fs::read_to_string(path).ok()?;
    // Minimal JSON parsing — extract benchmark entries.
    let mut results = Vec::new();
    // Find each benchmark block: "name": { ... "ns_per_op": X ... }
    for line in content.lines() {
        let line = line.trim();
        if line.starts_with('"') && line.ends_with('{') {
            let name = line
                .trim_start_matches('"')
                .split('"')
                .next()
                .unwrap_or("")
                .to_string();
            // Read ahead for ns_per_op and throughput_mops from subsequent lines
            // (handled below by accumulating into current entry)
            results.push(BenchResult {
                name,
                iterations: 0,
                total_ns: 0,
                ops_per_iter: 0,
                ns_per_op: 0.0,
                throughput_mops: 0.0,
            });
        }
        if let Some(r) = results.last_mut() {
            if line.starts_with("\"ns_per_op\"") {
                if let Some(val) = line.split(':').nth(1) {
                    let val = val.trim().trim_end_matches(',');
                    r.ns_per_op = val.parse().unwrap_or(0.0);
                }
            }
            if line.starts_with("\"throughput_mops\"") {
                if let Some(val) = line.split(':').nth(1) {
                    let val = val.trim().trim_end_matches(',');
                    r.throughput_mops = val.parse().unwrap_or(0.0);
                }
            }
        }
    }
    results.retain(|r| r.ns_per_op > 0.0);
    Some(results)
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let json_mode = args.iter().any(|a| a == "--json");
    let baseline_path = args
        .windows(2)
        .find(|w| w[0] == "--baseline")
        .map(|w| w[1].as_str());
    let filter = args
        .windows(2)
        .find(|w| w[0] == "--filter")
        .map(|w| w[1].as_str());

    let baseline = baseline_path.and_then(parse_baseline);

    let should_run = |name: &str| -> bool {
        filter.is_none_or(|f| name.contains(f))
    };

    let mut results: Vec<BenchResult> = Vec::new();

    // Tier 1: Field arithmetic roofline
    if should_run("field_mul") {
        results.push(bench_field_mul());
    }
    if should_run("field_add") {
        results.push(bench_field_add());
    }
    if should_run("fmadd_one") {
        results.push(bench_fmadd_one());
    }

    // Tier 2: Per-pair kernel eval
    for d in [4, 8, 16] {
        let name = format!("toom{d}_eval");
        if should_run(&name) {
            results.push(bench_toom_eval(d));
        }
    }

    // Tier 3: Full parallel reduce
    for d in [4, 8, 16] {
        let name = format!("dense_reduce_d{d}");
        if should_run(&name) {
            results.push(bench_dense_reduce(d));
        }
    }

    // Tier 4: Bind
    if should_run("bind_l2h") {
        results.push(bench_bind(BindingOrder::LowToHigh));
    }
    if should_run("bind_h2l") {
        results.push(bench_bind(BindingOrder::HighToLow));
    }

    // Output
    if json_mode {
        println!("{}", to_json(&results));
        print_table(&results, baseline.as_ref());
    } else {
        print_table(&results, baseline.as_ref());
    }
}
