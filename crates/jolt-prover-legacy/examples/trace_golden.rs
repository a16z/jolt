//! Golden-trace equivalence gate for tracer changes.
//!
//! Runs a fixed set of (guest, input) pairs, hashes every emitted `Cycle` in
//! stream order (postcard-serialized, the same codec `tracer` uses to persist
//! traces), plus the final memory state and the `JoltDevice` I/O, and compares
//! against fixtures recorded from the unmodified tracer.
//!
//! Usage:
//!   cargo run --release -p jolt-prover-legacy --features host --example trace_golden -- record [filter]
//!   cargo run --release -p jolt-prover-legacy --features host --example trace_golden -- check [filter]
//!
//! `record` (re)writes matching fixture entries; `check` re-runs and prints
//! PASS/FAIL per guest, exiting non-zero on any mismatch. The optional filter
//! is a substring over guest names.

// Link inline crates so their inventory registrations reach the tracer.
extern crate jolt_inlines_keccak256 as _;
extern crate jolt_inlines_sha2 as _;

use jolt_prover_legacy::host;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;

const FIXTURE_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/golden_traces.json"
);

#[derive(Serialize, Deserialize, PartialEq, Eq, Debug, Clone)]
struct GoldenRecord {
    row_count: usize,
    /// blake3 over the postcard bytes of every Cycle, in stream order.
    trace_hash: String,
    /// blake3 over postcard of `Memory::materialized_nonzero_bytes()` (address-sorted).
    memory_hash: String,
    /// blake3 over postcard of the final `JoltDevice` (inputs, outputs, panic, layout).
    io_hash: String,
}

/// Input encoding for the `(input: [u8; 32], num_iters: u32)` chain guests.
fn chain_input(iters: u32) -> Vec<u8> {
    let mut input = postcard::to_stdvec(&[5u8; 32]).unwrap();
    input.extend(postcard::to_stdvec(&iters).unwrap());
    input
}

/// Fixed (guest, input) pairs. Sizes target ~1M cycles except muldiv, which is
/// a tiny guest covering M-extension mul/div and compressed instructions.
fn golden_cases() -> Vec<(&'static str, Vec<u8>)> {
    vec![
        // ~3396 cycles/hash
        ("sha2-chain-guest", chain_input(300)),
        // ~4330 cycles/hash
        ("sha3-chain-guest", chain_input(235)),
        // ~12 cycles/unit
        ("fibonacci-guest", postcard::to_stdvec(&84_000u32).unwrap()),
        // ~1550 cycles/op; alloc-heavy, exercises rem via wyhash indexing
        ("btreemap-guest", postcard::to_stdvec(&650u32).unwrap()),
        // M-extension mul/div, compressed instructions
        (
            "muldiv-guest",
            postcard::to_stdvec(&[9u32, 5u32, 3u32]).unwrap(),
        ),
    ]
}

fn run_case(guest: &str, input: &[u8]) -> GoldenRecord {
    let mut program = host::Program::new(guest);
    let (_, trace, memory, io_device) = program.trace(input, &[], &[]);

    let mut hasher = blake3::Hasher::new();
    let mut buf: Vec<u8> = Vec::with_capacity(256);
    for cycle in &trace {
        buf.clear();
        postcard::to_io(cycle, &mut buf).unwrap();
        hasher.update(&buf);
    }
    let trace_hash = hasher.finalize().to_hex().to_string();

    let memory_hash =
        blake3::hash(&postcard::to_stdvec(&memory.materialized_nonzero_bytes()).unwrap())
            .to_hex()
            .to_string();
    let io_hash = blake3::hash(&postcard::to_stdvec(&io_device).unwrap())
        .to_hex()
        .to_string();

    GoldenRecord {
        row_count: trace.len(),
        trace_hash,
        memory_hash,
        io_hash,
    }
}

fn load_fixtures() -> BTreeMap<String, GoldenRecord> {
    match std::fs::read_to_string(FIXTURE_PATH) {
        Ok(contents) => serde_json::from_str(&contents).expect("malformed golden_traces.json"),
        Err(_) => BTreeMap::new(),
    }
}

fn main() {
    let mode = std::env::args().nth(1).unwrap_or_default();
    let filter = std::env::args().nth(2);
    let selected = |name: &str| filter.as_deref().is_none_or(|f| name.contains(f));

    match mode.as_str() {
        "record" => {
            let mut fixtures = load_fixtures();
            for (guest, input) in golden_cases() {
                if !selected(guest) {
                    continue;
                }
                let record = run_case(guest, &input);
                println!(
                    "recorded {guest}: {} rows, trace={}",
                    record.row_count, record.trace_hash
                );
                let _ = fixtures.insert(guest.to_string(), record);
            }
            let parent = Path::new(FIXTURE_PATH).parent().unwrap();
            std::fs::create_dir_all(parent).unwrap();
            let mut file = std::fs::File::create(FIXTURE_PATH).unwrap();
            serde_json::to_writer_pretty(&mut file, &fixtures).unwrap();
            file.write_all(b"\n").unwrap();
            println!("wrote {FIXTURE_PATH}");
        }
        "check" => {
            let fixtures = load_fixtures();
            assert!(
                !fixtures.is_empty(),
                "no fixtures found at {FIXTURE_PATH}; run `record` first"
            );
            let mut failures = 0usize;
            for (guest, input) in golden_cases() {
                if !selected(guest) {
                    continue;
                }
                let Some(expected) = fixtures.get(guest) else {
                    println!("SKIP {guest}: no fixture entry");
                    continue;
                };
                let actual = run_case(guest, &input);
                if actual == *expected {
                    println!("PASS {guest} ({} rows)", actual.row_count);
                } else {
                    failures += 1;
                    println!("FAIL {guest}");
                    println!("  expected: {expected:?}");
                    println!("  actual:   {actual:?}");
                }
            }
            if failures > 0 {
                println!("{failures} guest(s) FAILED golden-trace check");
                std::process::exit(1);
            }
            println!("all golden-trace checks passed");
        }
        _ => {
            eprintln!("usage: trace_golden <record|check> [guest-filter]");
            std::process::exit(2);
        }
    }
}
