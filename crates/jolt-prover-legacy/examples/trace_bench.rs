//! Tracer throughput benchmark: measures emulation/trace-generation rate only.
//!
//! Run: cargo run --release -p jolt-prover-legacy --features host --example trace_bench [-- <filter>]
//!
//! The optional positional argument is a substring filter over guest names
//! (e.g. `sha2-chain`, `fib`). No argument runs every guest.

// Link inline crates so their inventory registrations reach the tracer.
extern crate jolt_inlines_keccak256 as _;
extern crate jolt_inlines_sha2 as _;

use jolt_prover_legacy::host;
use std::time::Instant;

// Empirically measured cycles per operation (see benches/e2e_profiling.rs).
const CYCLES_PER_SHA256: f64 = 3396.0;
const CYCLES_PER_SHA3: f64 = 4330.0;
const CYCLES_PER_BTREEMAP_OP: f64 = 1550.0;
const CYCLES_PER_FIBONACCI_UNIT: f64 = 12.0;

/// Target trace length per benchmark: ~9.6M cycles, comfortably above 2^23.
const TARGET_CYCLES: f64 = (1u64 << 23) as f64 * 1.15;

fn ops_for_target(cycles_per_op: f64) -> u32 {
    (TARGET_CYCLES / cycles_per_op).ceil() as u32
}

/// Input encoding for the `(input: [u8; 32], num_iters: u32)` chain guests.
fn chain_input(iters: u32) -> Vec<u8> {
    let mut input = postcard::to_stdvec(&[5u8; 32]).unwrap();
    input.extend(postcard::to_stdvec(&iters).unwrap());
    input
}

fn bench(guest: &str, input: Vec<u8>, runs: usize) {
    let mut program = host::Program::new(guest);
    // Build guest + decode once (excluded from timing)
    let _ = program.decode();
    // Warmup trace
    let (_, trace, _, _) = program.trace(&input, &[], &[]);
    let len = trace.len();
    drop(trace);

    let mut times = Vec::new();
    for _ in 0..runs {
        let start = Instant::now();
        let (_, trace, _, _) = program.trace(&input, &[], &[]);
        let dt = start.elapsed().as_secs_f64();
        assert_eq!(trace.len(), len);
        drop(trace);
        times.push(dt);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = times[times.len() / 2];
    let mhz = len as f64 / median / 1e6;
    println!("{guest}: {len} cycles, times(s)={times:.3?}, median={median:.3}s => {mhz:.2} MHz");

    // Execute-only pass: same program span, no Cycle construction. MHz is
    // reported in trace-row-equivalents (rows the traced run produced over
    // the wall time of the execute-only run) — the rate that matters for a
    // two-pass parallel tracer's first pass.
    let mut exec_times = Vec::new();
    for _ in 0..runs {
        let start = Instant::now();
        let executed = program.execute(&input, &[], &[]);
        let dt = start.elapsed().as_secs_f64();
        assert!(executed > 0);
        exec_times.push(dt);
    }
    exec_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let exec_median = exec_times[exec_times.len() / 2];
    let exec_mhz = len as f64 / exec_median / 1e6;
    println!(
        "{guest} (execute-only): times(s)={exec_times:.3?}, median={exec_median:.3}s => {exec_mhz:.2} MHz row-equiv"
    );
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
