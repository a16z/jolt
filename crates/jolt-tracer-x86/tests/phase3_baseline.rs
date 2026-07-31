//! Phase-3 baseline: both engines, one platform, one harness.
//! Measures, per guest (median of 3, in-process):
//! - reference serial `ExecutionBackend::trace` (modular seam, includes the
//!   Cycle to TraceRow conversion),
//! - reference fast pass `ChunkedExecutionBackend::execute` at 2^18 rows
//!   (post-#1717 this runs the parallel machinery's PassOne, execute mode),
//! - the AOT x86 fast pass (`fast_run`), where the guest's kinds are
//!   supported (fibonacci yes; sha2-chain uses the SHA2 inline and
//!   fail-fasts by design until slice 3).
//!
//! Numbers from a Rosetta container are PROVISIONAL; the gate platform is a
//! real linux-x86_64 workstation.

#![cfg(all(target_arch = "x86_64", target_os = "linux"))]
#![expect(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::print_stdout
)]

use jolt_program::execution::{
    ChunkedExecutionBackend, ExecutionBackend, JoltProgram, TraceInputs,
};
use jolt_tracer_x86::X86TracerBackend;
use std::time::Instant;
use tracer::TracerBackend;

// Link the SHA2 inline registration for the sha2-chain guest.
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
        .expect("failed to run jolt CLI");
    assert!(output.status.success(), "guest build failed");
    let elf_path = format!("{target_dir}/riscv64imac-unknown-none-elf/release/{package}");
    std::fs::read(&elf_path).unwrap_or_else(|e| panic!("failed to read ELF at {elf_path}: {e}"))
}

fn setup(package: &str, func: &str, input: Vec<u8>) -> (JoltProgram, TraceInputs) {
    let elf = build_guest_elf(package, func);
    // Inline-bearing guests (sha2-chain) need the tracer's inline provider.
    let mut provider = tracer::TracerInlineExpansionProvider::new();
    let program = jolt_program::build_jolt_program_with_inline_provider(
        &elf,
        &mut provider,
        jolt_riscv::RV64IMAC_JOLT_ALL_INLINES,
    )
    .expect("failed to build Jolt program");
    let memory_config = common::jolt_device::MemoryConfig {
        program_size: Some(program.program_end - common::constants::RAM_START_ADDRESS),
        ..Default::default()
    };
    (
        program,
        TraceInputs::new(input, Vec::new(), Vec::new(), memory_config),
    )
}

fn median3(mut f: impl FnMut() -> f64) -> f64 {
    let mut times = [f(), f(), f()];
    times.sort_by(f64::total_cmp);
    times[1]
}

fn report(guest: &str, program: &JoltProgram, inputs: &TraceInputs, x86: bool) {
    std::env::remove_var("TRACER_PARALLEL");

    // Reference serial (modular seam).
    let mut rows = 0usize;
    let serial = median3(|| {
        let mut backend = TracerBackend::new();
        let start = Instant::now();
        let out = backend
            .trace(program, inputs.clone())
            .expect("trace failed");
        let elapsed = start.elapsed().as_secs_f64();
        rows = out.trace.rows().len();
        elapsed
    });

    // Reference fast pass (execute mode + per-chunk checkpoints).
    let fast = median3(|| {
        let mut backend = TracerBackend::new();
        let start = Instant::now();
        let summary = backend
            .execute(program, inputs.clone(), 1 << 18)
            .expect("execute failed");
        assert_eq!(summary.trace_len, rows);
        start.elapsed().as_secs_f64()
    });

    // AOT x86 fast pass (supported guests only).
    let x86_fast = x86.then(|| {
        let mut backend = X86TracerBackend::new();
        // Warm the compile cache so steady-state is measured.
        let _ = backend
            .fast_run(program, inputs.clone())
            .expect("fast_run failed");
        median3(|| {
            let start = Instant::now();
            let out = backend
                .fast_run(program, inputs.clone())
                .expect("fast_run failed");
            assert_eq!(out.trace_len, rows);
            start.elapsed().as_secs_f64()
        })
    });

    let mhz = |seconds: f64| rows as f64 / seconds / 1e6;
    println!(
        "| {guest} | {rows} | {:.3} ({:.1} MHz) | {:.3} ({:.1} MHz) | {} |",
        serial,
        mhz(serial),
        fast,
        mhz(fast),
        match x86_fast {
            Some(t) => format!("{:.3} ({:.1} MHz)", t, mhz(t)),
            None => "n/a (inline kinds, slice 3)".to_string(),
        }
    );
}

#[test]
#[ignore = "phase-3 baseline measurement; run explicitly"]
fn phase3_baseline() {
    println!(
        "| guest | rows | ref serial s (MHz) | ref fast pass s (MHz) | x86 fast pass s (MHz) |"
    );
    println!("|---|---:|---:|---:|---:|");

    let (program, inputs) = setup(
        "fibonacci-guest",
        "fib",
        postcard::to_stdvec(&400_000_u32).unwrap(),
    );
    report("fibonacci_400000", &program, &inputs, true);

    let mut chain_input = postcard::to_stdvec(&[5u8; 32]).unwrap();
    chain_input.append(&mut postcard::to_stdvec(&4_446u32).unwrap());
    let (program, inputs) = setup("sha2-chain-guest", "sha2_chain", chain_input);
    report("sha2_chain_4446", &program, &inputs, true);
}
