//! Serial/parallel trace equivalence gate for tracer changes.
//!
//! Runs a fixed set of (guest, input) pairs under several tracer
//! configurations and requires every configuration to produce byte-identical
//! output: the same `Cycle` stream (postcard-serialized, the codec `tracer`
//! uses to persist traces), the same final memory, and the same `JoltDevice`.
//!
//! Usage:
//!   cargo run --release -p tracer --example trace_equivalence
//!
//! WHY no recorded fixtures: guest ELFs are rebuilt from source, and their
//! `.rodata` embeds absolute build paths (panic locations, dependency paths).
//! A digest recorded on one machine therefore cannot match another machine's
//! even when the tracer is bit-identical, so committed hashes would only ever
//! be valid for whoever recorded them. Comparing configurations within a
//! single run needs no fixtures and is what the parallel pipeline actually
//! has to guarantee.
//!
//! Each configuration runs in a child process (`emit` mode) because the
//! tracer selects its pipeline from the environment; spawning avoids mutating
//! the environment of a process that already has worker threads.

// Link inline crates so their inventory registrations reach the tracer.
extern crate jolt_inlines_keccak256 as _;
extern crate jolt_inlines_sha2 as _;

#[path = "support/mod.rs"]
mod support;

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use support::chain_input;

#[derive(Serialize, Deserialize, PartialEq, Eq, Debug, Clone)]
struct TraceDigest {
    row_count: usize,
    /// blake3 over the postcard bytes of every Cycle, in stream order.
    trace_hash: String,
    /// blake3 over postcard of `Memory::materialized_nonzero_bytes()` (address-sorted).
    memory_hash: String,
    /// blake3 over postcard of the final `JoltDevice` (inputs, outputs, panic, layout).
    io_hash: String,
}

/// A tracer configuration, expressed as the environment that selects it.
struct Config {
    name: &'static str,
    env: &'static [(&'static str, &'static str)],
}

/// The serial interpreter is the reference; every parallel configuration must
/// match it. Chunk sizes are deliberately small so that a ~1M-row guest
/// crosses many chunk boundaries (`TRACER_PARALLEL=1` maps to the serial
/// path, so 2 workers is the smallest real pipeline), and the tiny
/// capacity reservation forces the overflow copy-assembly path.
const CONFIGS: &[Config] = &[
    Config {
        name: "serial",
        env: &[],
    },
    Config {
        name: "parallel/2-worker/128-row chunks",
        env: &[("TRACER_PARALLEL", "2"), ("JOLT_TRACER_CHUNK_ROWS", "128")],
    },
    Config {
        name: "parallel/4-worker/128-row chunks",
        env: &[("TRACER_PARALLEL", "4"), ("JOLT_TRACER_CHUNK_ROWS", "128")],
    },
    Config {
        name: "parallel/4-worker/64k-row chunks",
        env: &[
            ("TRACER_PARALLEL", "4"),
            ("JOLT_TRACER_CHUNK_ROWS", "65536"),
        ],
    },
    Config {
        name: "parallel/4-worker/64k-row chunks/1k-row reserve (overflow assembly)",
        env: &[
            ("TRACER_PARALLEL", "4"),
            ("JOLT_TRACER_CHUNK_ROWS", "65536"),
            ("JOLT_TRACER_CAPACITY_ROWS", "1000"),
        ],
    },
];

/// Fixed (guest, input) pairs. Sizes target ~1M cycles except muldiv, which is
/// a tiny guest covering M-extension mul/div and compressed instructions.
fn cases() -> Vec<(&'static str, Vec<u8>)> {
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

fn run_case(guest: &str, input: &[u8]) -> TraceDigest {
    let (elf, elf_path, memory_config) = support::build_guest(guest);
    let (_, trace, memory, io_device, _) =
        tracer::trace(&elf, Some(&elf_path), input, &[], &[], &memory_config, None);

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

    TraceDigest {
        row_count: trace.len(),
        trace_hash,
        memory_hash,
        io_hash,
    }
}

/// Re-run this binary under `config` and collect its digests.
fn digests_under(config: &Config, filter: Option<&str>) -> BTreeMap<String, TraceDigest> {
    let exe = std::env::current_exe().expect("cannot locate own executable");
    let mut command = std::process::Command::new(exe);
    command.arg("emit");
    if let Some(filter) = filter {
        command.arg(filter);
    }
    // Start from a clean slate so a developer's ambient TRACER_PARALLEL cannot
    // silently turn the serial reference into a parallel run.
    command.env_remove("TRACER_PARALLEL");
    command.env_remove("JOLT_TRACER_CHUNK_ROWS");
    command.env_remove("JOLT_TRACER_CAPACITY_ROWS");
    for (key, value) in config.env {
        command.env(key, value);
    }
    let output = command
        .output()
        .expect("failed to re-run self in emit mode");
    if !output.status.success() {
        eprintln!("{}", String::from_utf8_lossy(&output.stderr));
        panic!("emit run failed for config `{}`", config.name);
    }
    serde_json::from_slice(&output.stdout).expect("malformed digest JSON from emit run")
}

/// Report every component that differs, not just the first.
fn describe_divergence(guest: &str, reference: &TraceDigest, actual: &TraceDigest) -> Vec<String> {
    let mut parts = Vec::new();
    if reference.row_count != actual.row_count {
        parts.push(format!(
            "{guest}: row count {} != {}",
            actual.row_count, reference.row_count
        ));
    }
    if reference.trace_hash != actual.trace_hash {
        parts.push(format!(
            "{guest}: cycle stream {} != {}",
            actual.trace_hash, reference.trace_hash
        ));
    }
    if reference.memory_hash != actual.memory_hash {
        parts.push(format!(
            "{guest}: final memory {} != {}",
            actual.memory_hash, reference.memory_hash
        ));
    }
    if reference.io_hash != actual.io_hash {
        parts.push(format!(
            "{guest}: device I/O {} != {}",
            actual.io_hash, reference.io_hash
        ));
    }
    parts
}

fn main() {
    let mode = std::env::args().nth(1).unwrap_or_default();
    let filter = std::env::args().nth(2);
    let selected = |name: &str| filter.as_deref().is_none_or(|f| name.contains(f));

    if mode == "emit" {
        let digests: BTreeMap<String, TraceDigest> = cases()
            .into_iter()
            .filter(|(guest, _)| selected(guest))
            .map(|(guest, input)| (guest.to_string(), run_case(guest, &input)))
            .collect();
        println!("{}", serde_json::to_string(&digests).unwrap());
        return;
    }
    if !mode.is_empty() && mode != "check" {
        eprintln!("usage: trace_equivalence [check] [guest-filter]");
        std::process::exit(2);
    }

    let reference_config = &CONFIGS[0];
    let reference = digests_under(reference_config, filter.as_deref());
    assert!(!reference.is_empty(), "no guests selected");
    for (guest, digest) in &reference {
        println!(
            "{} [{}]: {} rows",
            guest, reference_config.name, digest.row_count
        );
    }

    let mut failures = 0usize;
    for config in &CONFIGS[1..] {
        let actual = digests_under(config, filter.as_deref());
        let mut divergences = Vec::new();
        for (guest, expected) in &reference {
            match actual.get(guest) {
                Some(got) => divergences.extend(describe_divergence(guest, expected, got)),
                None => divergences.push(format!("{guest}: missing from {}", config.name)),
            }
        }
        if divergences.is_empty() {
            println!("PASS {} matches {}", config.name, reference_config.name);
        } else {
            failures += 1;
            println!(
                "FAIL {} diverges from {}",
                config.name, reference_config.name
            );
            for line in divergences {
                println!("  {line}");
            }
        }
    }

    if failures > 0 {
        println!("{failures} configuration(s) diverged from the serial tracer");
        std::process::exit(1);
    }
    println!(
        "all {} configuration(s) produce byte-identical traces",
        CONFIGS.len()
    );
}
