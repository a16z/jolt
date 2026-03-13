//! Pipeline tests using tracer-level Cycle types.
//!
//! Constructs execution traces from RISC-V instruction types (ADD, SUB, JAL)
//! and runs them through generate_witnesses → prove → verify.

mod common;

use common::*;

/// ADD instructions terminated by JAL-to-self.
#[test]
fn trace_add_pipeline() {
    run_trace_e2e(
        vec![
            make_add_cycle(0x1000, 3, 4),
            make_add_cycle(0x1004, 10, 20),
            make_add_cycle(0x1008, 100, 200),
            make_jal_terminal(0x100C),
        ],
        b"jolt-trace-add",
    );
}

/// Pure NoOp trace.
#[test]
fn trace_noop_pipeline() {
    run_trace_e2e(
        vec![
            tracer::instruction::Cycle::NoOp,
            tracer::instruction::Cycle::NoOp,
        ],
        b"jolt-trace-noop",
    );
}

/// Mixed ADD + SUB with JAL termination.
#[test]
fn trace_mixed_pipeline() {
    run_trace_e2e(
        vec![
            make_add_cycle(0x1000, 7, 3),
            make_sub_cycle(0x1004, 42, 8),
            make_add_cycle(0x1008, 1, 2),
            make_jal_terminal(0x100C),
        ],
        b"jolt-trace-mixed",
    );
}

/// Single ADD instruction with JAL termination (minimum real trace).
#[test]
fn trace_single_add() {
    run_trace_e2e(
        vec![make_add_cycle(0x1000, 5, 5), make_jal_terminal(0x1004)],
        b"jolt-trace-single-add",
    );
}

/// Sequence of SUBs: exercises subtraction constraints.
#[test]
fn trace_sub_sequence() {
    run_trace_e2e(
        vec![
            make_sub_cycle(0x1000, 100, 30),
            make_sub_cycle(0x1004, 50, 50),
            make_sub_cycle(0x1008, 1, 0),
            make_jal_terminal(0x100C),
        ],
        b"jolt-trace-sub-seq",
    );
}

/// Alternating ADD/SUB pattern.
#[test]
fn trace_alternating_add_sub() {
    run_trace_e2e(
        vec![
            make_add_cycle(0x1000, 10, 20),
            make_sub_cycle(0x1004, 30, 5),
            make_add_cycle(0x1008, 25, 75),
            make_sub_cycle(0x100C, 100, 1),
            make_add_cycle(0x1010, 99, 1),
            make_sub_cycle(0x1014, 100, 100),
            make_add_cycle(0x1018, 0, 0),
            make_jal_terminal(0x101C),
        ],
        b"jolt-trace-alt",
    );
}

/// Large wrapping addition (tests u64 wrapping behavior).
#[test]
fn trace_wrapping_add() {
    run_trace_e2e(
        vec![
            make_add_cycle(0x1000, u64::MAX, 1),
            make_add_cycle(0x1004, u64::MAX / 2, u64::MAX / 2 + 1),
            make_jal_terminal(0x1008),
        ],
        b"jolt-trace-wrap-add",
    );
}

/// Large wrapping subtraction.
#[test]
fn trace_wrapping_sub() {
    run_trace_e2e(
        vec![
            make_sub_cycle(0x1000, 0, 1),
            make_sub_cycle(0x1004, 0, u64::MAX),
            make_jal_terminal(0x1008),
        ],
        b"jolt-trace-wrap-sub",
    );
}

/// 4 NoOps — tests padding for non-power-of-two that becomes power-of-two.
#[test]
fn trace_four_noops() {
    run_trace_e2e(
        vec![
            tracer::instruction::Cycle::NoOp,
            tracer::instruction::Cycle::NoOp,
            tracer::instruction::Cycle::NoOp,
            tracer::instruction::Cycle::NoOp,
        ],
        b"jolt-trace-4noop",
    );
}

/// 3 cycles → padded to 4 (non-power-of-two).
#[test]
fn trace_three_cycles_padded() {
    run_trace_e2e(
        vec![
            make_add_cycle(0x1000, 1, 2),
            make_sub_cycle(0x1004, 10, 3),
            make_jal_terminal(0x1008),
        ],
        b"jolt-trace-3pad",
    );
}

/// 5 cycles → padded to 8.
#[test]
fn trace_five_cycles_padded() {
    run_trace_e2e(
        vec![
            make_add_cycle(0x1000, 1, 1),
            make_add_cycle(0x1004, 2, 2),
            make_sub_cycle(0x1008, 4, 1),
            make_add_cycle(0x100C, 3, 3),
            make_jal_terminal(0x1010),
        ],
        b"jolt-trace-5pad",
    );
}

/// Zero-operand arithmetic (all zeros except flags and PC).
#[test]
fn trace_zero_operands() {
    run_trace_e2e(
        vec![
            make_add_cycle(0x1000, 0, 0),
            make_sub_cycle(0x1004, 0, 0),
            make_add_cycle(0x1008, 0, 0),
            make_jal_terminal(0x100C),
        ],
        b"jolt-trace-zeros",
    );
}
