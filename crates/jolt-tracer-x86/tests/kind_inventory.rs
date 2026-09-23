//! Dev tool: histogram of `JoltInstructionKind`s in a guest's expanded
//! bytecode. Portable (no native codegen); drives template coverage for the
//! transpiler bring-up. Run with:
//!
//! ```sh
//! cargo nextest run -p jolt-tracer-x86 kind_inventory --cargo-quiet --run-ignored all --no-capture
//! ```

#![expect(clippy::print_stdout)]

use std::collections::BTreeMap;

// Link inline registrations for inline-bearing guests.
use jolt_inlines_keccak256 as _;
use jolt_inlines_sha2 as _;

mod common;
use common::setup;

fn kind_histogram(package: &str, func: &str) -> BTreeMap<String, usize> {
    let Some((program, _)) = setup(package, func, Vec::new()) else {
        return BTreeMap::new();
    };
    let mut histogram = BTreeMap::new();
    for row in &program.expanded_bytecode {
        *histogram
            .entry(format!("{:?}", row.instruction_kind))
            .or_insert(0) += 1;
    }
    histogram
}

#[test]
#[ignore = "dev tool: prints the static-bytecode kind histogram for a guest"]
fn fibonacci_kind_inventory() {
    let histogram = kind_histogram("fibonacci-guest", "fib");
    println!(
        "fibonacci-guest expanded-bytecode kinds ({}):",
        histogram.len()
    );
    for (kind, count) in &histogram {
        println!("  {kind:40} {count}");
    }
    assert!(!histogram.is_empty());
}

#[test]
#[ignore = "dev tool: prints the static-bytecode kind histogram for a guest"]
fn sha2_chain_kind_inventory() {
    let histogram = kind_histogram("sha2-chain-guest", "sha2_chain");
    println!(
        "sha2-chain-guest expanded-bytecode kinds ({}):",
        histogram.len()
    );
    for (kind, count) in &histogram {
        println!("  {kind:40} {count}");
    }
    assert!(!histogram.is_empty());
}

#[test]
#[ignore = "dev tool: prints the static-bytecode kind histogram for a guest"]
fn sha3_chain_kind_inventory() {
    let histogram = kind_histogram("sha3-chain-guest", "sha3_chain");
    println!(
        "sha3-chain-guest expanded-bytecode kinds ({}):",
        histogram.len()
    );
    for (kind, count) in &histogram {
        println!("  {kind:40} {count}");
    }
    assert!(!histogram.is_empty());
}

#[test]
#[ignore = "dev tool: prints the static-bytecode kind histogram for a guest"]
fn btreemap_kind_inventory() {
    let histogram = kind_histogram("btreemap-guest", "btreemap");
    println!(
        "btreemap-guest expanded-bytecode kinds ({}):",
        histogram.len()
    );
    for (kind, count) in &histogram {
        println!("  {kind:40} {count}");
    }
    assert!(!histogram.is_empty());
}

#[test]
#[ignore = "dev tool: prints the static-bytecode kind histogram for a guest"]
fn muldiv_kind_inventory() {
    let histogram = kind_histogram("muldiv-guest", "muldiv");
    println!(
        "muldiv-guest expanded-bytecode kinds ({}):",
        histogram.len()
    );
    for (kind, count) in &histogram {
        println!("  {kind:40} {count}");
    }
    assert!(!histogram.is_empty());
}
