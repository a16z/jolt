//! Chunked execution for the AOT backend (spec slice 4, invariant 3).
//!
//! The concatenation of replayed chunks must equal the eager trace for every
//! chunk size, including degenerate ones, and independently of replay order.

#![cfg(all(target_arch = "x86_64", target_os = "linux"))]
#![expect(clippy::expect_used)]

use jolt_program::execution::{
    ChunkedExecutionBackend, ExecutionBackend, JoltProgram, TraceInputs, TraceRow,
};
use jolt_tracer_x86::X86TracerBackend;

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
        .expect("failed to run jolt CLI");
    assert!(output.status.success(), "guest build failed");
    let elf_path = format!("{target_dir}/riscv64imac-unknown-none-elf/release/{package}");
    std::fs::read(&elf_path).expect("failed to read guest ELF")
}

fn setup(package: &str, func: &str, input: Vec<u8>) -> (JoltProgram, TraceInputs) {
    let elf = build_guest_elf(package, func);
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

/// Invariant 3: chunks compose to the eager trace, for any chunk size and in
/// any replay order.
fn assert_chunks_compose(package: &str, func: &str, input: Vec<u8>, chunk_sizes: &[usize]) {
    std::env::remove_var("TRACER_PARALLEL");
    let (program, inputs) = setup(package, func, input);

    let mut backend = X86TracerBackend::new();
    let eager = backend
        .trace(&program, inputs.clone())
        .expect("eager record trace failed");
    let expected = eager.trace.rows();
    assert!(!expected.is_empty());

    for &chunk_size in chunk_sizes {
        let summary = backend
            .execute(&program, inputs.clone(), chunk_size)
            .expect("chunked execute failed");
        assert_eq!(
            summary.trace_len,
            expected.len(),
            "chunk_size {chunk_size}: trace length"
        );
        assert_eq!(
            summary.checkpoints.len(),
            expected.len().div_ceil(chunk_size),
            "chunk_size {chunk_size}: checkpoint count"
        );

        // Replay in reverse to show order independence.
        let mut chunks: Vec<Vec<TraceRow>> = summary
            .checkpoints
            .iter()
            .rev()
            .map(|checkpoint| {
                backend
                    .replay_chunk(checkpoint)
                    .expect("replay failed")
                    .into_rows()
            })
            .collect();
        chunks.reverse();

        let concatenated: Vec<TraceRow> = chunks.into_iter().flatten().collect();
        assert_eq!(
            concatenated.len(),
            expected.len(),
            "chunk_size {chunk_size}: concatenated length"
        );
        for (index, (got, want)) in concatenated.iter().zip(expected.iter()).enumerate() {
            assert_eq!(got, want, "chunk_size {chunk_size}: row {index} diverged");
        }
    }
}

#[test]
fn fibonacci_chunks_compose() {
    assert_chunks_compose(
        "fibonacci-guest",
        "fib",
        postcard::to_stdvec(&40u32).expect("postcard"),
        // Degenerate sizes included: 1 forces boundaries inside multi-row
        // groups, and a size past the trace length is a single chunk.
        &[1, 7, 100, 1 << 18],
    );
}

#[test]
fn muldiv_chunks_compose() {
    let mut input = postcard::to_stdvec(&7u32).expect("postcard");
    input.extend(postcard::to_stdvec(&11u32).expect("postcard"));
    input.extend(postcard::to_stdvec(&3u32).expect("postcard"));
    // Exercises the DIV/REM advice groups across chunk boundaries.
    assert_chunks_compose("muldiv-guest", "muldiv", input, &[1, 13, 1000]);
}
