//! Safety properties of the generated code (spec AC11 / invariant 6).
//!
//! Two claims that must hold for a backend that executes guest-controlled
//! data in-process:
//! 1. the finalized code mapping is never writable and executable at once;
//! 2. a guest access outside the memory plane surfaces as a defined error,
//!    not as undefined behaviour.

#![cfg(all(target_arch = "x86_64", target_os = "linux"))]
#![expect(clippy::expect_used, clippy::panic)]

use common::constants::{RAM_START_ADDRESS, STACK_CANARY_SIZE};
use common::jolt_device::{JoltDevice, MemoryConfig};
use jolt_platform::{JOLT_ADVICE_WRITE_CALL_ID, JOLT_PRINT_CALL_ID, JOLT_PRINT_STRING};
use jolt_program::execution::{
    ChunkedExecutionBackend, ExecutionBackend, JoltProgram, TraceError, TraceInputs,
};
use jolt_riscv::instructions::VirtualHostIO;
use jolt_riscv::{JoltInstructionKind as Kind, JoltInstructionRow, NormalizedOperands};
use jolt_tracer_x86::harness::{
    compile_program, memory_config, run_program, single_row_program, terminal_row, ExitReason,
    Outcome, SCRATCH_START, TEST_ADDR,
};

use jolt_tracer_x86::{RunOutput, X86TracerBackend};

const REGS: usize = common::constants::REGISTER_COUNT as usize;

fn row(kind: Kind, rs1: Option<u8>, rd: Option<u8>, imm: i128) -> JoltInstructionRow {
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
    let program = single_row_program(row(Kind::ADDI, Some(1), Some(2), 7));
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
    let program = single_row_program(row(Kind::LD, Some(1), Some(2), 0));
    let mut pre = [0u64; REGS];
    // Far past the plane, but still in the RAM region so it is not routed to
    // the device helpers: the bounds check is what must catch it.
    pre[1] = RAM_START_ADDRESS + (1u64 << 40);

    let outcome = run_program(&program, &pre, &[], &[]).expect("run should not error");
    assert_eq!(
        outcome.exit,
        ExitReason::FaultOutOfBounds as u64,
        "expected the out-of-bounds exit reason, got {} ({:?})",
        outcome.exit,
        outcome.helper_error
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
    let mut store = row(Kind::SD, Some(1), None, 0);
    store.operands.rs2 = Some(3);
    let program = single_row_program(store);
    let mut pre = [0u64; REGS];
    pre[1] = RAM_START_ADDRESS + (1u64 << 40);
    pre[3] = 0xdead_beef;

    let outcome = run_program(&program, &pre, &[], &[]).expect("run should not error");
    assert_eq!(
        outcome.exit,
        ExitReason::FaultOutOfBounds as u64,
        "expected the out-of-bounds exit reason, got {} ({:?})",
        outcome.exit,
        outcome.helper_error
    );
}

/// Run a single `SD` of `0xdead_beef` to `address` in the harness layout.
fn store_to(address: u64) -> Outcome {
    let mut store = row(Kind::SD, Some(1), None, 0);
    store.operands.rs2 = Some(3);
    let program = single_row_program(store);
    let mut pre = [0u64; REGS];
    pre[1] = address;
    pre[3] = 0xdead_beef;
    run_program(&program, &pre, &[], &[]).expect("run should not error")
}

fn harness_stack_end() -> u64 {
    JoltDevice::new(&memory_config()).memory_layout.stack_end
}

/// A store into the stack canary `[stack_end, stack_end + STACK_CANARY_SIZE)`
/// is a stack overflow. The interpreter panics on it
/// (`Mmu::assert_effective_address`); the native backend must fault instead
/// of performing the write.
#[test]
fn store_into_stack_canary_reports_a_fault() {
    let stack_end = harness_stack_end();
    let last_doubleword = stack_end + STACK_CANARY_SIZE - 8;
    for address in [stack_end, stack_end + 8, last_doubleword] {
        let outcome = store_to(address);
        assert_eq!(
            outcome.exit,
            ExitReason::FaultHelper as u64,
            "store to {address:#x}: expected the helper-fault exit reason, got {} ({:?})",
            outcome.exit,
            outcome.helper_error
        );
        let message = outcome.helper_error.expect("a helper error message");
        assert!(
            message.contains("Stack overflow")
                && message.contains("stack canary region")
                && message.contains(&format!("{address:#X}")),
            "unexpected helper error: {message}"
        );
    }
}

/// The doublewords on either side of the canary are ordinary RAM: the check
/// must not reject the top of the program image or the bottom of the stack.
#[test]
fn stores_next_to_stack_canary_succeed() {
    let stack_end = harness_stack_end();
    for address in [stack_end - 8, stack_end + STACK_CANARY_SIZE] {
        let outcome = store_to(address);
        assert_eq!(
            outcome.exit,
            ExitReason::HaltedSuccess as u64,
            "store to {address:#x}: expected the terminal halt, got {} ({:?})",
            outcome.exit,
            outcome.helper_error
        );
    }
}

/// An aligned address inside the text span but between compiled group starts
/// must take the jump-table filler path and report the bad target.
#[test]
fn in_range_unmapped_jump_reports_bad_target() {
    let program = single_row_program(row(Kind::JALR, Some(1), None, 0));
    let mut pre = [0u64; REGS];
    pre[1] = TEST_ADDR + 2;

    let outcome = run_program(&program, &pre, &[], &[]).expect("run should not error");
    assert_eq!(
        outcome.exit,
        ExitReason::FaultBadJumpTarget as u64,
        "expected the bad-jump exit reason, got {} ({:?})",
        outcome.exit,
        outcome.helper_error
    );
    assert_eq!(outcome.fault_addr, TEST_ADDR + 2);
}

/// A static branch whose target has no compiled group must compile: it
/// executes normally when not taken and reports the bad target when taken.
#[test]
fn static_branch_to_unmapped_target_faults_only_when_taken() {
    let target = TEST_ADDR + 12;
    let program = single_row_program(row(Kind::BEQ, Some(1), None, (target - TEST_ADDR).into()));
    for (x1, exit) in [
        (1, ExitReason::HaltedSuccess),
        (0, ExitReason::FaultBadJumpTarget),
    ] {
        let mut pre = [0u64; REGS];
        pre[1] = x1;
        let outcome = run_program(&program, &pre, &[], &[]).expect("run should not error");
        assert_eq!(
            outcome.exit, exit as u64,
            "x1 = {x1}: unexpected exit reason ({:?})",
            outcome.helper_error
        );
        if exit == ExitReason::FaultBadJumpTarget {
            assert_eq!(outcome.fault_addr, target);
        }
    }
}

#[test]
fn loads_reject_the_padding_between_io_end_and_ram() {
    let config = MemoryConfig {
        max_input_size: 24,
        max_output_size: 0,
        max_trusted_advice_size: 0,
        max_untrusted_advice_size: 0,
        ..memory_config()
    };
    let layout = JoltDevice::new(&config).memory_layout;
    assert_eq!(layout.io_end, RAM_START_ADDRESS - 24);
    for kind in [
        Kind::LD,
        Kind::VirtualLoadB,
        Kind::VirtualLoadH,
        Kind::VirtualLoadW,
    ] {
        for (address, accepted) in [
            (layout.io_end - 8, true),
            (layout.io_end, false),
            (layout.io_end + 8, false),
        ] {
            let mut load = row(
                kind,
                Some(1),
                Some(2),
                i128::from(address) - i128::from(TEST_ADDR),
            );
            load.address += 4;
            let program = JoltProgram::from_parts(
                minimal_elf(TEST_ADDR),
                vec![
                    row(Kind::AUIPC, None, Some(1), 0),
                    load,
                    terminal_row(TEST_ADDR + 8),
                ],
                Vec::new(),
                RAM_START_ADDRESS + 1024,
                TEST_ADDR,
            );
            let inputs = TraceInputs::new(Vec::new(), Vec::new(), Vec::new(), config);
            let result = X86TracerBackend::new()
                .run_machine(&program, inputs)
                .and_then(RunOutput::require_halt);
            if accepted {
                let _ = result.expect("load of the final I/O word");
            } else {
                assert!(matches!(
                    result,
                    Err(TraceError::Backend("host helper reported an error"))
                ));
            }
        }
    }
}

#[test]
fn device_load_failures_report_io_overflow() {
    let layout = JoltDevice::new(&memory_config()).memory_layout;
    let mut pre = [0u64; REGS];
    pre[1] = layout.io_end + 8;
    let program = single_row_program(row(Kind::LD, Some(1), Some(2), 0));
    let load = run_program(&program, &pre, &[], &[]).expect("run overflowing load");
    assert_eq!(load.exit, ExitReason::FaultHelper as u64);
    assert!(load
        .helper_error
        .expect("load error")
        .contains("I/O overflow"));

    // A host call never fails: it ignores a range a guest load would reject,
    // as the interpreter does.
    let program = single_row_program(row(Kind::VirtualHostIO(VirtualHostIO(())), None, None, 0));
    for call in [JOLT_ADVICE_WRITE_CALL_ID, JOLT_PRINT_CALL_ID] {
        pre[10] = u64::from(call);
        pre[11] = layout.io_end + 8;
        pre[12] = 1;
        pre[13] = u64::from(JOLT_PRINT_STRING);
        let read = run_program(&program, &pre, &[], &[]).expect("run overflowing host read");
        assert_eq!(
            read.exit,
            ExitReason::HaltedSuccess as u64,
            "{:?}",
            read.helper_error
        );
        assert_eq!(read.advice_tape, [] as [u8; 0]);
    }
}

#[test]
fn advice_write_follows_guest_load_bounds() {
    let layout = JoltDevice::new(&memory_config()).memory_layout;
    let program = single_row_program(row(Kind::VirtualHostIO(VirtualHostIO(())), None, None, 0));
    // A rejected range leaves the tape empty and the run continues.
    for (address, tape) in [
        (layout.get_lowest_address() - 1, Some(0)),
        (layout.heap_end, None),
        (layout.get_lowest_address(), Some(0)),
        (SCRATCH_START, Some(0xab)),
    ] {
        let mut pre = [0u64; REGS];
        pre[10] = u64::from(JOLT_ADVICE_WRITE_CALL_ID);
        pre[11] = address;
        pre[12] = 1;
        let outcome = run_program(&program, &pre, &[0xab], &[]).expect("run advice write");
        assert_eq!(
            outcome.exit,
            ExitReason::HaltedSuccess as u64,
            "address {address:#x}: {:?}",
            outcome.helper_error
        );
        assert_eq!(
            outcome.advice_tape,
            Vec::from_iter(tape),
            "address {address:#x}"
        );
    }
}

fn partial_word_program(config: &MemoryConfig, offset: u64, value: u64) -> JoltProgram {
    let layout = JoltDevice::new(config).memory_layout;
    let mut store = row(Kind::SD, Some(1), None, offset.into());
    store.operands.rs2 = Some(3);
    let mut terminate = row(
        Kind::SD,
        Some(1),
        None,
        i128::from(layout.termination) - i128::from(RAM_START_ADDRESS),
    );
    terminate.operands.rs2 = Some(4);
    let mut rows = vec![
        row(Kind::AUIPC, None, Some(1), 0),
        row(Kind::ADDI, Some(0), Some(3), value.into()),
        store,
        row(Kind::LD, Some(1), Some(2), offset.into()),
        row(Kind::ADDI, Some(0), Some(4), 1),
        terminate,
    ];
    for (i, row) in rows.iter_mut().enumerate() {
        row.address = RAM_START_ADDRESS as usize + 4 * i;
    }
    rows.push(terminal_row(RAM_START_ADDRESS + 4 * rows.len() as u64));
    JoltProgram::from_parts(
        minimal_elf(RAM_START_ADDRESS),
        rows,
        Vec::new(),
        RAM_START_ADDRESS + config.program_size.expect("program size"),
        RAM_START_ADDRESS,
    )
}

#[test]
fn last_heap_word_preserves_backing_bytes_across_checkpoints() {
    let config = MemoryConfig {
        program_size: Some(1025),
        stack_size: 0,
        heap_size: 8,
        ..Default::default()
    };
    let inputs = TraceInputs::new(Vec::new(), Vec::new(), Vec::new(), config);
    let program = partial_word_program(&config, 1160, 256);
    let mut backend = X86TracerBackend::new();
    backend.set_min_checkpoint_spacing_rows(1);
    let trace = backend
        .trace(&program, inputs.clone())
        .expect("last heap word");
    assert_eq!(
        trace.trace.rows()[3]
            .registers()
            .rd
            .expect("load result")
            .post_value,
        256
    );
    assert!(trace
        .final_memory
        .as_ref()
        .expect("final memory")
        .bytes
        .contains(&(1161, 1)));
    let summary = backend
        .execute(&program, inputs, 1)
        .expect("checkpoint heap word");
    assert_eq!(summary.checkpoints.len(), trace.trace.rows().len());
    for (checkpoint, expected) in summary.checkpoints.iter().zip(trace.trace.rows()) {
        let replay = backend.replay_chunk(checkpoint).expect("replay heap word");
        assert_eq!(replay.rows(), std::slice::from_ref(expected));
    }
    for offset in [1168, 1176] {
        let outside = partial_word_program(&config, offset, 0);
        let inputs = TraceInputs::new(Vec::new(), Vec::new(), Vec::new(), config);
        assert!(X86TracerBackend::new()
            .run_machine(&outside, inputs)
            .and_then(RunOutput::require_halt)
            .is_err());
    }
}

#[test]
fn run_machine_rejects_programs_without_elf_bytes() {
    let program = single_row_program(terminal_row(TEST_ADDR));
    let inputs = TraceInputs::new(Vec::new(), Vec::new(), Vec::new(), memory_config());
    assert!(matches!(
        X86TracerBackend::new().run_machine(&program, inputs),
        Err(TraceError::MissingElfBytes)
    ));
}

#[test]
fn run_machine_accepts_stalls_until_halt_is_required() {
    for kind in [Kind::JoltSuccess, Kind::JoltPanic, Kind::BEQ] {
        let terminal = if kind == Kind::BEQ {
            let mut branch = row(kind, Some(0), None, 0);
            branch.operands.rs2 = Some(0);
            branch
        } else {
            let mut halt = terminal_row(TEST_ADDR);
            halt.instruction_kind = kind;
            halt
        };
        let program = JoltProgram::from_parts(
            minimal_elf(TEST_ADDR),
            vec![terminal],
            Vec::new(),
            RAM_START_ADDRESS + 1024,
            TEST_ADDR,
        );
        let inputs = TraceInputs::new(Vec::new(), Vec::new(), Vec::new(), memory_config());
        let output = X86TracerBackend::new()
            .run_machine(&program, inputs)
            .expect("machine run");
        assert_eq!(output.trace_len, 1);
        if kind == Kind::BEQ {
            assert!(matches!(
                output.require_halt(),
                Err(TraceError::EndedWithoutHalt { pc: TEST_ADDR })
            ));
        } else {
            assert_eq!(output.require_halt().expect("halted run").trace_len, 1);
        }
    }
}

#[test]
fn tracing_rejects_a_stall() {
    let mut stall = row(Kind::BEQ, Some(0), None, 0);
    stall.operands.rs2 = Some(0);
    let program = JoltProgram::from_parts(
        minimal_elf(TEST_ADDR),
        vec![stall],
        Vec::new(),
        RAM_START_ADDRESS + 1024,
        TEST_ADDR,
    );
    let inputs = TraceInputs::new(Vec::new(), Vec::new(), Vec::new(), memory_config());
    let mut backend = X86TracerBackend::new();
    assert!(matches!(
        backend.trace(&program, inputs.clone()),
        Err(TraceError::EndedWithoutHalt { pc: TEST_ADDR })
    ));
    assert!(matches!(
        backend.execute(&program, inputs, 1),
        Err(TraceError::EndedWithoutHalt { pc: TEST_ADDR })
    ));
}

fn minimal_elf(entry: u64) -> Vec<u8> {
    // Source recovery parses this sectionless ELF; execution uses the rows above.
    let mut elf = vec![0; 64];
    elf[..7].copy_from_slice(b"\x7fELF\x02\x01\x01");
    elf[16..18].copy_from_slice(&2u16.to_le_bytes());
    elf[18..20].copy_from_slice(&243u16.to_le_bytes());
    elf[20..24].copy_from_slice(&1u32.to_le_bytes());
    elf[24..32].copy_from_slice(&entry.to_le_bytes());
    elf[52..54].copy_from_slice(&64u16.to_le_bytes());
    elf[54..56].copy_from_slice(&56u16.to_le_bytes());
    elf[58..60].copy_from_slice(&64u16.to_le_bytes());
    elf
}
