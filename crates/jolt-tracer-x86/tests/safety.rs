//! Safety properties of the generated code (spec AC11 / invariant 6).
//!
//! Two claims that must hold for a backend that executes guest-controlled
//! data in-process:
//! 1. the finalized code mapping is never writable and executable at once;
//! 2. a guest access outside the memory plane surfaces as a defined error,
//!    not as undefined behaviour.

#![cfg(all(target_arch = "x86_64", target_os = "linux"))]
#![expect(clippy::expect_used, clippy::panic)]

use jolt_riscv::{JoltInstructionKind, JoltInstructionRow, NormalizedOperands};
use jolt_tracer_x86::harness::{compile_program, run_program, single_row_program, TEST_ADDR};

const REGS: usize = common::constants::REGISTER_COUNT as usize;

fn row(
    kind: JoltInstructionKind,
    rs1: Option<u8>,
    rd: Option<u8>,
    imm: i128,
) -> JoltInstructionRow {
    JoltInstructionRow {
        instruction_kind: kind,
        address: TEST_ADDR as usize,
        operands: NormalizedOperands {
            rs1,
            rs2: None,
            rd,
            imm,
        },
        virtual_sequence_remaining: None,
        is_first_in_sequence: true,
        is_compressed: false,
    }
}

/// The mapping holding generated code must be executable and not writable.
/// dynasm-rs assembles into a read-write mapping and flips it to read-execute
/// on finalize; this asserts the flip actually happened in the running
/// process rather than trusting the library.
#[test]
fn generated_code_mapping_is_not_writable() {
    let program = single_row_program(row(JoltInstructionKind::ADDI, Some(1), Some(2), 7));
    // The artifact must stay alive: dropping it unmaps the very region whose
    // permissions we are asserting.
    let compiled = compile_program(&program).expect("compile failed");
    let address = compiled.code_address();

    let maps = std::fs::read_to_string("/proc/self/maps").expect("cannot read /proc/self/maps");
    let mut found = None;
    for line in maps.lines() {
        let mut fields = line.split_whitespace();
        let Some(range) = fields.next() else { continue };
        let Some(perms) = fields.next() else { continue };
        let Some((start, end)) = range.split_once('-') else {
            continue;
        };
        let (Ok(start), Ok(end)) = (
            usize::from_str_radix(start, 16),
            usize::from_str_radix(end, 16),
        ) else {
            continue;
        };
        if (start..end).contains(&address) {
            found = Some(perms.to_string());
            break;
        }
    }

    let perms = found.unwrap_or_else(|| {
        panic!("no /proc/self/maps entry contains the generated code at {address:#x}")
    });
    assert!(
        perms.contains('x'),
        "generated code mapping is not executable: {perms}"
    );
    assert!(
        !perms.contains('w'),
        "generated code mapping is writable and executable (W^X violated): {perms}"
    );
}

/// A load whose effective address lies outside the guest memory plane must
/// produce the defined out-of-bounds exit, not a host segfault.
#[test]
fn out_of_bounds_load_reports_a_fault() {
    let program = single_row_program(row(JoltInstructionKind::LD, Some(1), Some(2), 0));
    let mut pre = [0u64; REGS];
    // Far past the plane, but still in the RAM region so it is not routed to
    // the device helpers: the bounds check is what must catch it.
    pre[1] = common::constants::RAM_START_ADDRESS + (1u64 << 40);

    let outcome = run_program(&program, &pre, &[], &[]).expect("run should not error");
    assert_eq!(
        outcome.exit, 2,
        "expected the out-of-bounds exit reason, got {} ({:?})",
        outcome.exit, outcome.helper_error
    );
    assert_eq!(
        outcome.fault_addr, pre[1],
        "the faulting guest address should be reported"
    );
}

/// A store outside the plane must likewise be caught rather than corrupting
/// host memory.
#[test]
fn out_of_bounds_store_reports_a_fault() {
    let mut store = row(JoltInstructionKind::SD, Some(1), None, 0);
    store.operands.rs2 = Some(3);
    let program = single_row_program(store);
    let mut pre = [0u64; REGS];
    pre[1] = common::constants::RAM_START_ADDRESS + (1u64 << 40);
    pre[3] = 0xdead_beef;

    let outcome = run_program(&program, &pre, &[], &[]).expect("run should not error");
    assert_eq!(
        outcome.exit, 2,
        "expected the out-of-bounds exit reason, got {} ({:?})",
        outcome.exit, outcome.helper_error
    );
}

/// An aligned address inside the text span but between compiled group starts
/// must take the jump-table filler path and report the bad target.
#[test]
fn in_range_unmapped_jump_reports_bad_target() {
    let program = single_row_program(row(JoltInstructionKind::JALR, Some(1), None, 0));
    let mut pre = [0u64; REGS];
    pre[1] = TEST_ADDR + 2;

    let outcome = run_program(&program, &pre, &[], &[]).expect("run should not error");
    assert_eq!(
        outcome.exit, 3,
        "expected the bad-jump exit reason, got {} ({:?})",
        outcome.exit, outcome.helper_error
    );
    assert_eq!(outcome.fault_addr, TEST_ADDR + 2);
}
