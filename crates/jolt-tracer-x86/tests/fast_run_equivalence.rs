//! Whole-guest fast-pass equivalence: run the fibonacci guest through the
//! x86 backend's fast (non-recording) pass and compare row count, device
//! outputs, and final memory against the reference interpreter.
//!
//! Native-only: on other targets this file compiles to nothing.

#![cfg(all(target_arch = "x86_64", target_os = "linux"))]
#![expect(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use jolt_program::execution::{ExecutionBackend, JoltProgram, TraceInputs};
use jolt_tracer_x86::X86TracerBackend;
use tracer::TracerBackend;

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

fn setup(package: &str, func: &str, input: Vec<u8>) -> (JoltProgram, TraceInputs) {
    let elf = build_guest_elf(package, func);
    let program =
        jolt_program::execution::build_jolt_program(&elf).expect("failed to build Jolt program");
    let memory_config = common::jolt_device::MemoryConfig {
        program_size: Some(program.program_end - common::constants::RAM_START_ADDRESS),
        ..Default::default()
    };
    let inputs = TraceInputs::new(input, Vec::new(), Vec::new(), memory_config);
    (program, inputs)
}

#[test]
fn fibonacci_fast_run_matches_reference() {
    // Pin the reference to serial mode (the tracer env-dispatches to the
    // parallel pipeline).
    std::env::remove_var("TRACER_PARALLEL");
    let (program, inputs) = setup(
        "fibonacci-guest",
        "fib",
        postcard::to_stdvec(&100u32).unwrap(),
    );

    let reference = TracerBackend::new()
        .trace(&program, inputs.clone())
        .expect("reference trace failed");
    let reference_rows = reference.trace.rows();

    let mut backend = X86TracerBackend::new();
    let fast = backend
        .fast_run(&program, inputs)
        .expect("x86 fast run failed");

    assert_eq!(fast.trace_len, reference_rows.len(), "row count");
    assert_eq!(fast.device.outputs, reference.device.outputs, "outputs");
    assert_eq!(fast.device.panic, reference.device.panic, "panic flag");
    assert_eq!(
        Some(fast.final_memory),
        reference.final_memory,
        "final memory"
    );
    assert_eq!(Some(fast.advice_tape), reference.advice_tape, "advice tape");
}
