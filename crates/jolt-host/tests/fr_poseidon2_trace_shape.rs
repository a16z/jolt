//! Shape probe for the FR Poseidon2 SDK guest.
//!
//! Mirrors the trace/preprocess/shape-compute steps inside
//! `jolt_host::prove_program` (without running any proof) so we can plan a
//! goldens regen at the natural shape this guest produces.

use common::constants::{BYTES_PER_INSTRUCTION, RAM_START_ADDRESS};
use common::jolt_device::MemoryLayout;
use jolt_trace::{BytecodePreprocessing, CycleRow, Program};

/// Mirror of `jolt_host::compute_min_ram_k` (lib.rs:587-631). Re-implemented
/// here because the upstream helper is `pub(crate)`.
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

#[test]
#[ignore = "shape probe; requires jolt CLI installed"]
fn fr_poseidon2_trace_shape() {
    // Same memory config baked into the `#[jolt::provable]` attribute on
    // `fr_poseidon2_sdk` (examples/bn254-fr-poseidon2-sdk/guest/src/lib.rs:13-18).
    let mut program = Program::new("bn254-fr-poseidon2-sdk-guest");
    let _ = program
        .set_func("fr_poseidon2_sdk")
        .set_stack_size(65_536)
        .set_heap_size(131_072)
        .set_max_input_size(8_192);

    // Same inputs as examples/bn254-fr-poseidon2-sdk/src/main.rs.
    let s0: [u64; 4] = [1, 0, 0, 0];
    let s1: [u64; 4] = [2, 0, 0, 0];
    let s2: [u64; 4] = [3, 0, 0, 0];
    let inputs = postcard::to_stdvec(&(s0, s1, s2)).expect("postcard encode FR inputs");

    // Two-pass: pass 1 (compute_advice ELF) populates the advice tape with
    // ark-bn254 results; pass 2 emits real FR coprocessor opcodes that
    // consume that tape. Required for FR-active guests — single-pass
    // panics inside the inline FieldOp handler when the tape is empty.
    let (_lazy, trace, _final_memory, _io_device, field_reg_events) =
        program.trace_two_pass_advice(&inputs, &[], &[]);
    let (bytecode_raw, init_mem, _program_size, entry_address) = program.decode();

    let raw_trace_len = trace.len();
    let trace_length = raw_trace_len.next_power_of_two().max(256);
    let bytecode = BytecodePreprocessing::preprocess(bytecode_raw, entry_address);
    let bytecode_k = bytecode.code_size;
    let log_t = trace_length.trailing_zeros() as usize;
    let log_k_bytecode = bytecode_k.trailing_zeros() as usize;

    // We don't have direct access to JoltDevice's MemoryLayout from the
    // returned io_device without retracing — but the layout is fully
    // determined by the MemoryConfig we set above plus program_size, so
    // reconstruct it the same way Program::trace does.
    let (_lazy2, _trace2, _final_memory2, io_device, _events2) =
        program.trace_two_pass_advice(&inputs, &[], &[]);
    let memory_layout = io_device.memory_layout.clone();
    let ram_k = compute_min_ram_k(&init_mem, &trace, &memory_layout);
    let log_k_ram = ram_k.trailing_zeros() as usize;

    eprintln!("=== FR Poseidon2 SDK guest natural shape ===");
    eprintln!("raw trace.len()       = {raw_trace_len}");
    eprintln!("trace_length (padded) = {trace_length}");
    eprintln!("bytecode.code_size    = {bytecode_k}");
    eprintln!("ram_K                 = {ram_k}");
    eprintln!("log_t                 = {log_t}");
    eprintln!("log_k_bytecode        = {log_k_bytecode}");
    eprintln!("log_k_ram             = {log_k_ram}");
    eprintln!("entry_address         = 0x{entry_address:x}");
    eprintln!("field_reg_events.len  = {}", field_reg_events.len());
    eprintln!(
        "memory_layout         = input_start=0x{:x} ram_start=0x{RAM_START_ADDRESS:x}",
        memory_layout.input_start
    );
}
