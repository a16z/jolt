//! DIAGNOSTIC: compare LIVE trace-derived data (the way `jolt_host::
//! prove_program` constructs it) against the fixture data emitted by
//! `core_muldiv_commitment_fixture_at_log_t(9)`. Earlier elimination has
//! ruled out:
//!   - golden vs. fresh program plans (byte-identical at fixture shape),
//!   - preprocessing_digest=[0u8;32] (bolt_oracle succeeds with zero digest),
//!   - commitment-direct vs. replay (commitments byte-identical through S6).
//!
//! The remaining suspect is the LIVE data construction inside
//! `prove_program`. This test independently rebuilds the same data via
//! `jolt_trace::Program` (the host path) and compares every component
//! against the fixture, reporting the FIRST divergence. Don't try to
//! fix the bug here — just locate it.
//!
//! Run via:
//!   LLVM_SYS_220_PREFIX=/opt/homebrew/opt/llvm cargo nextest run \
//!     -p jolt-equivalence prove_program_live_vs_fixture_data \
//!     --run-ignored only --cargo-quiet --no-capture

#![expect(
    clippy::expect_used,
    clippy::panic,
    clippy::print_stderr,
    clippy::too_many_lines,
    clippy::similar_names,
    unfulfilled_lint_expectations,
    reason = "diagnostic test reports the first divergence loudly"
)]

use common::constants::{BYTES_PER_INSTRUCTION, RAM_START_ADDRESS};
use common::jolt_device::MemoryLayout;
use jolt_equivalence::core_oracle::core_muldiv_commitment_fixture_at_log_t;
use jolt_field::Fr;
use jolt_kernels::trace::{
    stage1_rv64_cycles, stage2_instruction_lookup_cycles, stage2_product_virtual_cycles,
    stage2_ram_accesses, stage3_cycles, stage4_register_accesses, stage5_lookup_trace,
    stage6_bytecode_entries,
};
use jolt_lookup_tables::traits::InstructionLookupTable;
use jolt_lookup_tables::XLEN;
use jolt_r1cs::{constraints::rv64, R1csKey};
use jolt_trace::ram::build_ram_states;
use jolt_trace::{extract_trace, with_isa_struct, BytecodePreprocessing, CycleRow, Program};
use tracer::instruction::Instruction;

fn instruction_lookup_table_index(instr: &Instruction) -> Option<usize> {
    with_isa_struct!(
        instr,
        |i| <_ as InstructionLookupTable<XLEN>>::lookup_table(&i).map(|t| t.index()),
        noop => None
    )
}

fn compute_min_ram_k(
    init_mem: &[(u64, u8)],
    trace: &[impl CycleRow],
    memory_layout: &MemoryLayout,
) -> usize {
    let lowest_addr = memory_layout.get_lowest_address();
    let min_bc_addr = init_mem
        .iter()
        .map(|(a, _)| *a)
        .min()
        .unwrap_or(lowest_addr);
    let max_bc_addr = init_mem
        .iter()
        .map(|(a, _)| *a)
        .max()
        .unwrap_or(lowest_addr)
        + (BYTES_PER_INSTRUCTION as u64 - 1);
    let num_bc_words = max_bc_addr.div_ceil(8) - min_bc_addr / 8 + 1;
    let bytecode_start_remapped = if min_bc_addr >= lowest_addr && min_bc_addr != 0 {
        (min_bc_addr - lowest_addr) / 8
    } else {
        0
    };
    let trace_max_remapped: u64 = trace
        .iter()
        .filter_map(|cycle| {
            let addr = cycle.ram_access_address()?;
            if addr == 0 || addr < lowest_addr {
                None
            } else {
                Some((addr - lowest_addr) / 8)
            }
        })
        .max()
        .unwrap_or(0);
    let io_end_remapped = if RAM_START_ADDRESS >= lowest_addr {
        (RAM_START_ADDRESS - lowest_addr) / 8
    } else {
        0
    };
    let ram_k_min = trace_max_remapped
        .max(bytecode_start_remapped + num_bc_words + 1)
        .max(io_end_remapped);
    (ram_k_min as usize).next_power_of_two()
}

/// Reports the first divergence between two slices via Debug formatting.
fn assert_debug_eq_slices<T: std::fmt::Debug>(label: &str, fixture: &[T], live: &[T]) {
    if fixture.len() != live.len() {
        panic!(
            "{label}: LENGTH differs (fixture={}, live={})",
            fixture.len(),
            live.len(),
        );
    }
    for (i, (a, b)) in fixture.iter().zip(live.iter()).enumerate() {
        let da = format!("{a:?}");
        let db = format!("{b:?}");
        if da != db {
            panic!(
                "{label}: first divergence at index {i} of {}\n  fixture: {da}\n  live:    {db}",
                fixture.len()
            );
        }
    }
    eprintln!("[live-vs-fixture] {label}: MATCH ({} elems)", fixture.len());
}

fn assert_eq_slices<T: std::fmt::Debug + PartialEq>(label: &str, fixture: &[T], live: &[T]) {
    if fixture.len() != live.len() {
        panic!(
            "{label}: LENGTH differs (fixture={}, live={})",
            fixture.len(),
            live.len(),
        );
    }
    for (i, (a, b)) in fixture.iter().zip(live.iter()).enumerate() {
        if a != b {
            panic!(
                "{label}: first divergence at index {i} of {}\n  fixture: {a:?}\n  live:    {b:?}",
                fixture.len()
            );
        }
    }
    eprintln!("[live-vs-fixture] {label}: MATCH ({} elems)", fixture.len());
}

#[test]
#[ignore = "diagnostic: compare LIVE prove_program data vs fixture data"]
fn prove_program_live_vs_fixture_data() {
    let log_t = 9usize;
    let fixture = core_muldiv_commitment_fixture_at_log_t(log_t);

    // Same inputs as the fixture (`core_oracle.rs:233-234`).
    let inputs = postcard::to_stdvec(&[9u32, 5u32, 3u32]).expect("muldiv inputs");

    // Build LIVE data via jolt_trace::Program (the prove_program path).
    let mut program = Program::new("muldiv-guest");
    let (_lazy_trace, trace, final_memory, io_device) = program.trace(&inputs, &[], &[]);
    let (bytecode_raw, init_mem, _program_size, entry_address) = program.decode();

    let bytecode = BytecodePreprocessing::preprocess(bytecode_raw, entry_address);
    let trace_length = trace.len().next_power_of_two().max(256);
    eprintln!(
        "[live-vs-fixture] live trace_length={}, raw trace.len()={}",
        trace_length,
        trace.len()
    );
    eprintln!(
        "[live-vs-fixture] fixture.proof.trace_length={}",
        fixture.proof.trace_length
    );
    if trace_length != fixture.proof.trace_length {
        panic!(
            "trace_length: live={}, fixture={} — top-level shape diverges before any data check",
            trace_length, fixture.proof.trace_length
        );
    }

    let memory_layout = io_device.memory_layout.clone();
    let ram_k = compute_min_ram_k(&init_mem, &trace, &memory_layout);
    eprintln!(
        "[live-vs-fixture] live ram_k={}, fixture.proof.ram_K={}",
        ram_k, fixture.proof.ram_K
    );
    if ram_k != fixture.proof.ram_K as usize {
        panic!(
            "ram_k: live={}, fixture={}",
            ram_k, fixture.proof.ram_K
        );
    }

    let r1cs_key = R1csKey::new(rv64::rv64_constraints::<Fr>(), trace_length);

    let (live_cycle_inputs, live_r1cs_witness, _) = extract_trace::<_, Fr>(
        &trace,
        trace_length,
        &bytecode,
        &memory_layout,
        r1cs_key.num_vars_padded,
        &[],
    );

    let live_rv64_cycles = stage1_rv64_cycles(&trace, trace_length, &bytecode);
    let live_product_virtual_cycles = stage2_product_virtual_cycles(&trace, trace_length);
    let live_instruction_lookup_cycles = stage2_instruction_lookup_cycles(&trace, trace_length);

    let lowest_addr = memory_layout.get_lowest_address();
    let remap_addr = |addr: u64| {
        if addr == 0 || addr < lowest_addr {
            None
        } else {
            Some(((addr - lowest_addr) / 8) as usize)
        }
    };
    let live_ram_accesses = stage2_ram_accesses(&trace, trace_length, |a| remap_addr(a));
    let live_stage3_cycles = stage3_cycles(&trace, trace_length, &bytecode);
    let live_stage4_register_accesses = stage4_register_accesses(&trace, trace_length);
    let live_stage5_lookup_trace = stage5_lookup_trace(&trace, trace_length, |cycle| {
        let instr = cycle.instruction();
        instruction_lookup_table_index(&instr)
    });
    let (live_initial_ram_state, live_final_ram_state) =
        build_ram_states(&init_mem, &final_memory, &io_device, ram_k);
    let live_stage6_bytecode_entries: Vec<_> =
        stage6_bytecode_entries::<Fr, _>(&bytecode, |i| instruction_lookup_table_index(i));
    let live_entry_bytecode_index = bytecode.entry_bytecode_index();

    // ===== Field-by-field assertions, reporting first divergence =====

    // CycleInput / Stage1Rv64Cycle / Stage2RamAccess / Stage3Cycle /
    // Stage4RegisterAccess don't derive PartialEq — use Debug.
    assert_debug_eq_slices("cycle_inputs", &fixture.cycle_inputs, &live_cycle_inputs);
    assert_eq_slices("r1cs_witness", &fixture.r1cs_witness, &live_r1cs_witness);
    assert_debug_eq_slices("rv64_cycles", &fixture.rv64_cycles, &live_rv64_cycles);
    assert_debug_eq_slices(
        "product_virtual_cycles",
        &fixture.product_virtual_cycles,
        &live_product_virtual_cycles,
    );
    assert_debug_eq_slices(
        "instruction_lookup_cycles",
        &fixture.instruction_lookup_cycles,
        &live_instruction_lookup_cycles,
    );
    assert_debug_eq_slices("ram_accesses", &fixture.ram_accesses, &live_ram_accesses);
    assert_debug_eq_slices("stage3_cycles", &fixture.stage3_cycles, &live_stage3_cycles);
    assert_debug_eq_slices(
        "stage4_register_accesses",
        &fixture.stage4_register_accesses,
        &live_stage4_register_accesses,
    );
    assert_eq_slices(
        "stage5_lookup_indices",
        &fixture.stage5_lookup_indices,
        &live_stage5_lookup_trace.lookup_indices,
    );
    assert_eq_slices(
        "stage5_lookup_table_indices",
        &fixture.stage5_lookup_table_indices,
        &live_stage5_lookup_trace.lookup_table_indices,
    );
    assert_eq_slices(
        "stage5_is_interleaved_operands",
        &fixture.stage5_is_interleaved_operands,
        &live_stage5_lookup_trace.is_interleaved_operands,
    );
    assert_eq_slices(
        "initial_ram_state",
        &fixture.initial_ram_state,
        &live_initial_ram_state,
    );
    assert_eq_slices(
        "final_ram_state",
        &fixture.final_ram_state,
        &live_final_ram_state,
    );
    assert_debug_eq_slices(
        "stage6_bytecode_entries",
        &fixture.stage6_bytecode_entries,
        &live_stage6_bytecode_entries,
    );
    assert_eq!(
        fixture.stage6_entry_bytecode_index, live_entry_bytecode_index,
        "stage6_entry_bytecode_index: fixture={}, live={}",
        fixture.stage6_entry_bytecode_index, live_entry_bytecode_index
    );

    eprintln!(
        "[live-vs-fixture] ALL trace-derived data matches between fixture and live \
         prove_program-style construction. Divergence cause must be upstream of \
         trace-derived data (memory_layout? entry_address? bytecode preprocessing?) \
         or downstream (per-stage opening inputs / FR-event injection / \
         JoltDevice content beyond memory_layout)."
    );
}
