//! Chunked execution for the AOT backend (spec slice 4, invariant 3).
//!
//! The concatenation of replayed chunks must equal the eager trace for every
//! chunk size, including degenerate ones, and independently of replay order.

#![cfg(all(target_arch = "x86_64", target_os = "linux"))]
#![expect(clippy::expect_used)]

use jolt_program::execution::{ChunkedExecutionBackend, ExecutionBackend, TraceRow};
use jolt_tracer_x86::X86TracerBackend;

use jolt_inlines_keccak256 as _;
use jolt_inlines_sha2 as _;

mod common;
use common::setup;

/// Invariant 3: chunks compose to the eager trace, for any chunk size and in
/// any replay order. With `dense_boundaries`, additionally shrink the
/// checkpoint spacing to a fraction of the trace and assert every checkpoint
/// resumes from a nearby boundary — proving the pause/resume machinery
/// (`Paused` exits, multi-boundary selection, boundary restore) actually ran,
/// which the production spacing of 2^16 rows never does on guests this small.
fn assert_chunks_compose_spaced(
    package: &str,
    func: &str,
    input: Vec<u8>,
    chunk_sizes: &[usize],
    dense_boundaries: bool,
) {
    std::env::remove_var("TRACER_PARALLEL");
    let Some((program, inputs)) = setup(package, func, input) else {
        return;
    };

    let mut backend = X86TracerBackend::new();
    let eager = backend
        .trace(&program, inputs.clone())
        .expect("eager record trace failed");
    let expected = eager.trace.rows();
    assert!(!expected.is_empty());

    // Sized from the measured trace so several pauses are guaranteed
    // regardless of how many rows the guest's startup contributes.
    let min_spacing = dense_boundaries.then(|| (expected.len() / 8).max(8));
    if let Some(rows) = min_spacing {
        backend.set_min_checkpoint_spacing_rows(rows);
    }

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
        if let Some(rows) = min_spacing {
            // A boundary exists at most spacing + one group past any mark, so
            // every checkpoint must resume from nearby. With a single (initial)
            // boundary — i.e. if pausing never fired — late chunks would skip
            // nearly the whole trace and this bound fails.
            let spacing = chunk_size.max(rows);
            let max_skip = summary
                .checkpoints
                .iter()
                .map(|checkpoint| checkpoint.skip_rows())
                .max()
                .unwrap_or(0);
            assert!(
                max_skip < 2 * spacing,
                "chunk_size {chunk_size}: max skip_rows {max_skip} exceeds the \
                 boundary spacing bound {} — checkpoints are not resuming from \
                 paused boundaries",
                2 * spacing
            );
        }

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

fn assert_chunks_compose(package: &str, func: &str, input: Vec<u8>, chunk_sizes: &[usize]) {
    assert_chunks_compose_spaced(package, func, input, chunk_sizes, false);
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

/// The pause/resume machinery under dense checkpoints: a small spacing forces
/// many `Paused` exits and resumes on a small guest, so replay restores
/// registers, memory, device state, and the advice cursor from real
/// mid-program boundaries — the paths production spacing (2^16 rows) only
/// reaches on million-row traces.
#[test]
fn fibonacci_chunks_compose_with_dense_boundaries() {
    assert_chunks_compose_spaced(
        "fibonacci-guest",
        "fib",
        postcard::to_stdvec(&60u32).expect("postcard"),
        &[1, 7, 64, 100],
        true,
    );
}

/// Dense boundaries across DIV/REM advice groups: resuming at a group whose
/// advice computation re-runs from restored registers must reproduce the
/// eager rows exactly.
#[test]
fn muldiv_chunks_compose_with_dense_boundaries() {
    let mut input = postcard::to_stdvec(&1_000_003u32).expect("postcard");
    input.extend(postcard::to_stdvec(&997u32).expect("postcard"));
    input.extend(postcard::to_stdvec(&13u32).expect("postcard"));
    assert_chunks_compose_spaced("muldiv-guest", "muldiv", input, &[1, 13, 100], true);
}
