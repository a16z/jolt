//! Tracer throughput benchmark: measures emulation/trace-generation rate only.
//!
//! Run: cargo run --release -p tracer --example trace_bench [-- <filter>]
//!
//! The optional positional argument is a substring filter over guest names
//! (e.g. `sha2-chain`, `fib`). No argument runs every guest.
//!
//! Three arms per guest: serial trace, execute-only (no row emission), and
//! two-pass parallel trace. Knobs:
//! * `JOLT_BENCH_SCALE=<n>` — multiply the ~9.6M-cycle target trace length
//! * `TRACER_PARALLEL=<w>` — worker count for the parallel arm (default:
//!   available parallelism − 1); `0` skips the arm
//! * `JOLT_TRACER_CHUNK_ROWS`, `JOLT_TRACER_CAPACITY_ROWS`,
//!   `JOLT_TRACER_TIMING` pass through

// Link inline crates so their inventory registrations reach the tracer.
extern crate jolt_inlines_keccak256 as _;
extern crate jolt_inlines_sha2 as _;

#[path = "support/mod.rs"]
mod support;

use std::time::Instant;
use support::chain_input;

// Empirically measured cycles per operation (see
// crates/jolt-prover-legacy/benches/e2e_profiling.rs).
const CYCLES_PER_SHA256: f64 = 3396.0;
const CYCLES_PER_SHA3: f64 = 4330.0;
const CYCLES_PER_BTREEMAP_OP: f64 = 1550.0;
const CYCLES_PER_FIBONACCI_UNIT: f64 = 12.0;

/// Target trace length per benchmark: ~9.6M cycles, comfortably above 2^23.
const TARGET_CYCLES: f64 = (1u64 << 23) as f64 * 1.15;

fn bench_scale() -> f64 {
    std::env::var("JOLT_BENCH_SCALE")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|&scale| scale > 0.0)
        .unwrap_or(1.0)
}

fn ops_for_target(cycles_per_op: f64) -> u32 {
    (TARGET_CYCLES * bench_scale() / cycles_per_op).ceil() as u32
}

fn parallel_workers() -> usize {
    match std::env::var("TRACER_PARALLEL")
        .ok()
        .and_then(|v| v.parse().ok())
    {
        Some(workers) => workers,
        None => std::thread::available_parallelism()
            .map(|n| n.get().saturating_sub(1))
            .unwrap_or(0),
    }
}

fn median(mut times: Vec<f64>) -> f64 {
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times[times.len() / 2]
}

fn bench(guest: &str, input: Vec<u8>, runs: usize) {
    // Build guest once (excluded from timing)
    let (_, elf_path, _) = support::build_guest(guest);
    // Each run re-reads and re-decodes the ELF, matching what the legacy
    // host::Program::trace timed region did — keeps MHz comparable with
    // numbers measured through that harness.
    let run_trace = |input: &[u8]| {
        let (elf, memory_config) = support::load_guest(&elf_path);
        let (_, trace, _, _, _) =
            tracer::trace(&elf, Some(&elf_path), input, &[], &[], &memory_config, None);
        trace
    };

    // Serial arm (explicitly, regardless of ambient env).
    std::env::remove_var("TRACER_PARALLEL");
    // Warmup trace
    let trace = run_trace(&input);
    let len = trace.len();
    drop(trace);

    let mut times = Vec::new();
    for _ in 0..runs {
        let start = Instant::now();
        let trace = run_trace(&input);
        let dt = start.elapsed().as_secs_f64();
        assert_eq!(trace.len(), len);
        drop(trace);
        times.push(dt);
    }
    let serial_median = median(times.clone());
    let mhz = len as f64 / serial_median / 1e6;
    println!(
        "{guest}: {len} cycles, times(s)={times:.3?}, median={serial_median:.3}s => {mhz:.2} MHz"
    );

    // Execute-only pass: same program span, no Cycle construction. Rate is
    // trace rows over the wall time of the execute-only run — the pass-1
    // rate of the two-pass parallel tracer.
    let mut exec_times = Vec::new();
    for _ in 0..runs {
        let start = Instant::now();
        let (elf, memory_config) = support::load_guest(&elf_path);
        let (executed, _, _) = tracer::execute(
            &elf,
            Some(&elf_path),
            &input,
            &[],
            &[],
            &memory_config,
            None,
        );
        let dt = start.elapsed().as_secs_f64();
        assert_eq!(executed, len);
        exec_times.push(dt);
    }
    let exec_median = median(exec_times.clone());
    let exec_mhz = len as f64 / exec_median / 1e6;
    println!(
        "{guest} (execute-only): times(s)={exec_times:.3?}, median={exec_median:.3}s => {exec_mhz:.2} MHz row-equiv"
    );

    // Parallel arm: the same trace through the two-pass pipeline.
    let workers = parallel_workers();
    if workers > 1 {
        std::env::set_var("TRACER_PARALLEL", workers.to_string());
        // Warmup (thread/pool/page state)
        let trace = run_trace(&input);
        assert_eq!(trace.len(), len);
        drop(trace);
        let mut par_times = Vec::new();
        for _ in 0..runs {
            let start = Instant::now();
            let trace = run_trace(&input);
            let dt = start.elapsed().as_secs_f64();
            assert_eq!(trace.len(), len);
            drop(trace);
            par_times.push(dt);
        }
        std::env::remove_var("TRACER_PARALLEL");
        let par_median = median(par_times.clone());
        let par_mhz = len as f64 / par_median / 1e6;
        println!(
            "{guest} (parallel x{workers}): times(s)={par_times:.3?}, median={par_median:.3}s => {par_mhz:.2} MHz"
        );
    }
}

fn main() {
    let filter = std::env::args().nth(1);
    let selected = |name: &str| filter.as_deref().is_none_or(|f| name.contains(f));

    type InputFn = fn() -> Vec<u8>;
    let benches: [(&str, InputFn); 4] = [
        ("sha2-chain-guest", || {
            chain_input(ops_for_target(CYCLES_PER_SHA256))
        }),
        ("sha3-chain-guest", || {
            chain_input(ops_for_target(CYCLES_PER_SHA3))
        }),
        ("fibonacci-guest", || {
            postcard::to_stdvec(&ops_for_target(CYCLES_PER_FIBONACCI_UNIT)).unwrap()
        }),
        ("btreemap-guest", || {
            postcard::to_stdvec(&ops_for_target(CYCLES_PER_BTREEMAP_OP)).unwrap()
        }),
    ];

    let mut ran = false;
    for (guest, input_fn) in benches {
        if selected(guest) {
            bench(guest, input_fn(), 3);
            ran = true;
        }
    }
    if !ran {
        eprintln!("No guest matched filter {filter:?}");
        std::process::exit(1);
    }
}
