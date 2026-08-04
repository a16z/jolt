//! Whole-guest fast-pass equivalence: run the fibonacci guest through the
//! x86 backend's fast (non-recording) pass and compare row count, device
//! outputs, and final memory against the reference interpreter.
//!
//! Native-only: on other targets this file compiles to nothing.

#![cfg(all(target_arch = "x86_64", target_os = "linux"))]
#![expect(clippy::unwrap_used, clippy::expect_used)]

use jolt_program::execution::ExecutionBackend;

// Link inline registrations for the inline-bearing guests.
use jolt_inlines_keccak256 as _;
use jolt_inlines_sha2 as _;
use jolt_tracer_x86::X86TracerBackend;
use tracer::TracerBackend;

mod common;
use common::setup;

/// Run a guest through both engines and assert the fast pass agrees with the
/// reference on everything the fast pass observes.
fn assert_fast_run_matches(package: &str, func: &str, input: Vec<u8>) {
    // Pin the reference to serial mode (the tracer env-dispatches to the
    // parallel pipeline).
    std::env::remove_var("TRACER_PARALLEL");
    let Some((program, inputs)) = setup(package, func, input) else {
        return;
    };

    let reference = TracerBackend::new()
        .trace(&program, inputs.clone())
        .expect("reference trace failed");
    let reference_rows = reference.trace.rows();

    let mut backend = X86TracerBackend::new();
    let fast = backend
        .fast_run(&program, inputs)
        .expect("x86 fast run failed");

    assert_eq!(fast.trace_len, reference_rows.len(), "{package}: row count");
    assert_eq!(
        fast.device.outputs, reference.device.outputs,
        "{package}: outputs"
    );
    assert_eq!(
        fast.device.panic, reference.device.panic,
        "{package}: panic flag"
    );
    assert_eq!(
        Some(fast.final_memory),
        reference.final_memory,
        "{package}: final memory"
    );
    assert_eq!(
        Some(fast.advice_tape),
        reference.advice_tape,
        "{package}: advice tape"
    );
}

/// Record mode: the full `TraceRow` stream must be identical to the
/// reference interpreter's, row for row. This is the strongest equivalence
/// statement the backend can make (spec invariant 1) and what proof
/// byte-equality rests on.
fn assert_record_matches(package: &str, func: &str, input: Vec<u8>) {
    std::env::remove_var("TRACER_PARALLEL");
    let Some((program, inputs)) = setup(package, func, input) else {
        return;
    };

    let reference = TracerBackend::new()
        .trace(&program, inputs.clone())
        .expect("reference trace failed");
    let expected = reference.trace.rows();

    let mut backend = X86TracerBackend::new();
    let actual_output = backend
        .trace(&program, inputs)
        .expect("x86 record trace failed");
    let actual = actual_output.trace.rows();

    assert_eq!(actual.len(), expected.len(), "{package}: row count");
    for (index, (got, want)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            got, want,
            "{package}: row {index} diverged\n  got:  {got:?}\n  want: {want:?}"
        );
    }
    assert_eq!(
        actual_output.device.outputs, reference.device.outputs,
        "{package}: outputs"
    );
    assert_eq!(
        actual_output.final_memory, reference.final_memory,
        "{package}: final memory"
    );
}

#[test]
fn fibonacci_record_matches_reference() {
    assert_record_matches(
        "fibonacci-guest",
        "fib",
        postcard::to_stdvec(&50u32).unwrap(),
    );
}

#[test]
fn muldiv_record_matches_reference() {
    // DIV/REM advice groups plus RAM traffic.
    assert_record_matches("muldiv-guest", "muldiv", {
        let mut bytes = postcard::to_stdvec(&7u32).unwrap();
        bytes.extend(postcard::to_stdvec(&11u32).unwrap());
        bytes.extend(postcard::to_stdvec(&3u32).unwrap());
        bytes
    });
}

#[test]
fn sha2_chain_record_matches_reference() {
    let mut input = postcard::to_stdvec(&[5u8; 32]).unwrap();
    input.append(&mut postcard::to_stdvec(&2u32).unwrap());
    assert_record_matches("sha2-chain-guest", "sha2_chain", input);
}

/// Production-scale row-stream equality. The small-guest tests above pin
/// semantics; this one exercises the same comparison over millions of rows
/// with realistic memory and control-flow patterns, which is the substance
/// AC7's proof byte-equality would provide (a proof is a deterministic
/// function of the trace, so byte-identical rows imply byte-identical
/// proofs). Ignored by default: it needs a few GB and ~10s.
#[test]
#[ignore = "scale test: several GB of trace rows"]
fn fibonacci_scale_record_matches_reference() {
    assert_record_matches(
        "fibonacci-guest",
        "fib",
        postcard::to_stdvec(&400_000_u32).unwrap(),
    );
}

#[test]
#[ignore = "scale test: several GB of trace rows"]
fn sha2_chain_scale_record_matches_reference() {
    let mut input = postcard::to_stdvec(&[5u8; 32]).unwrap();
    input.append(&mut postcard::to_stdvec(&256u32).unwrap());
    assert_record_matches("sha2-chain-guest", "sha2_chain", input);
}

#[test]
fn fibonacci_fast_run_matches_reference() {
    assert_fast_run_matches(
        "fibonacci-guest",
        "fib",
        postcard::to_stdvec(&100u32).unwrap(),
    );
}

#[test]
fn sha2_chain_fast_run_matches_reference() {
    let mut input = postcard::to_stdvec(&[5u8; 32]).unwrap();
    input.append(&mut postcard::to_stdvec(&8u32).unwrap());
    assert_fast_run_matches("sha2-chain-guest", "sha2_chain", input);
}

#[test]
fn sha3_chain_fast_run_matches_reference() {
    let mut input = postcard::to_stdvec(&[5u8; 32]).unwrap();
    input.append(&mut postcard::to_stdvec(&4u32).unwrap());
    assert_fast_run_matches("sha3-chain-guest", "sha3_chain", input);
}

#[test]
fn muldiv_fast_run_matches_reference() {
    // Exercises the DIV/REM advice groups (VirtualAdvice slots).
    assert_fast_run_matches(
        "muldiv-guest",
        "muldiv",
        vec![0xbd, 0xaa, 0xde, 0x5, 0x11, 0x5c],
    );
}

#[test]
fn btreemap_fast_run_matches_reference() {
    assert_fast_run_matches(
        "btreemap-guest",
        "btreemap",
        postcard::to_stdvec(&20u32).unwrap(),
    );
}
