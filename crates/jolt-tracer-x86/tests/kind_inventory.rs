//! Dev tool: histogram of `JoltInstructionKind`s in a guest's expanded
//! bytecode. Portable (no native codegen); drives template coverage for the
//! transpiler bring-up. Run with:
//!
//! ```sh
//! cargo nextest run -p jolt-tracer-x86 kind_inventory --cargo-quiet --run-ignored all --no-capture
//! ```

#![expect(clippy::expect_used, clippy::print_stdout, clippy::panic)]

use std::collections::BTreeMap;

// Link inline registrations for inline-bearing guests.
use jolt_inlines_keccak256 as _;
use jolt_inlines_sha2 as _;

fn build_guest_elf(package: &str, func: &str) -> Vec<u8> {
    let target_dir = format!("/tmp/jolt-guest-targets/{package}-{func}");
    let output = std::process::Command::new("jolt")
        .args([
            "build",
            "-p",
            package,
            "--stack-size",
            &common::constants::DEFAULT_STACK_SIZE.to_string(),
            "--heap-size",
            &common::constants::DEFAULT_HEAP_SIZE.to_string(),
            "--",
            "--release",
            "--target-dir",
            &target_dir,
            "--features",
            "guest",
        ])
        .env("JOLT_FUNC_NAME", func)
        .output()
        .expect("failed to run jolt CLI — install with: cargo install --path .");
    assert!(
        output.status.success(),
        "failed to build {package}:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let elf_path = format!("{target_dir}/riscv64imac-unknown-none-elf/release/{package}");
    std::fs::read(&elf_path).unwrap_or_else(|e| panic!("failed to read ELF at {elf_path}: {e}"))
}

fn kind_histogram(package: &str, func: &str) -> BTreeMap<String, usize> {
    let elf = build_guest_elf(package, func);
    let mut provider = tracer::TracerInlineExpansionProvider::new();
    let program = jolt_program::build_jolt_program_with_inline_provider(
        &elf,
        &mut provider,
        jolt_riscv::RV64IMAC_JOLT_ALL_INLINES,
    )
    .expect("failed to build Jolt program");
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
