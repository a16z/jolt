//! The I/O gap `[io_end, RAM_START_ADDRESS)` and the heap padding
//! `[heap_end, end of the RAM domain)` are zero-initialized read/write
//! memory, which this backend must trace exactly as the interpreter does.

#![cfg(all(target_arch = "x86_64", target_os = "linux"))]
#![expect(clippy::expect_used)]

use common::constants::RAM_START_ADDRESS;
use common::jolt_device::MemoryConfig;
use jolt_program::execution::{ChunkedExecutionBackend, ExecutionBackend, TraceInputs};
use jolt_riscv::{JoltTraceRow, JOLT_SUCCESS_WORD};
use jolt_tracer_x86::X86TracerBackend;
use tracer::TracerBackend;

mod text_elf;
use text_elf::text_elf;

/// `ld rd, offset(x1)`.
fn ld(rd: u32, offset: i32) -> u32 {
    (((offset as u32) & 0xfff) << 20) | (1 << 15) | (0b011 << 12) | (rd << 7) | 0x03
}

/// `sd` (`funct3` 3) or `sb` (`funct3` 0) `rs2, offset(x1)`.
fn store(funct3: u32, rs2: u32, offset: i32) -> u32 {
    let imm = (offset as u32) & 0xfff;
    ((imm >> 5) << 25) | (rs2 << 20) | (1 << 15) | (funct3 << 12) | ((imm & 0x1f) << 7) | 0x23
}

/// `addi rd, x0, imm`.
fn li(rd: u32, imm: i32) -> u32 {
    (((imm as u32) & 0xfff) << 20) | (rd << 7) | 0x13
}

/// Both ends of the gap and of the padding are loaded, stored and reloaded,
/// and a byte store merges into a gap word. Rows, dimensions, the final
/// image and the device match the interpreter's, and one-row chunks replay
/// across the stores.
#[test]
fn io_gap_and_heap_padding_match_the_interpreter() {
    // `io_end` is RAM_START_ADDRESS - 32 and `heap_end` RAM_START_ADDRESS +
    // 1216; the 1280 bytes from the lowest address, RAM_START_ADDRESS - 64,
    // pad to a 256-word domain ending at RAM_START_ADDRESS + 1984.
    let memory_config = MemoryConfig {
        max_input_size: 16,
        max_output_size: 16,
        max_trusted_advice_size: 0,
        max_untrusted_advice_size: 0,
        stack_size: 0,
        heap_size: 64,
        program_size: Some(1024),
    };
    // `auipc x1, 0` sets x1 to RAM_START_ADDRESS.
    let mut text = vec![0x0000_0097];
    for offset in [-32, -8, 1216, 1976] {
        text.extend([
            ld(3, offset),
            li(2, offset),
            store(3, 2, offset),
            ld(3, offset),
        ]);
    }
    text.extend([store(0, 2, -31), ld(3, -32), JOLT_SUCCESS_WORD]);
    let program = jolt_program::build_jolt_program(&text_elf(&text)).expect("build program");
    let inputs = TraceInputs::new(Vec::new(), Vec::new(), Vec::new(), memory_config);

    let reference = TracerBackend::new()
        .trace(&program, inputs.clone())
        .expect("reference trace");
    let image = &reference.final_memory.as_ref().expect("final memory").bytes;
    for byte in [
        (RAM_START_ADDRESS - 32, 0xe0),
        (RAM_START_ADDRESS - 31, 0xb8),
        (RAM_START_ADDRESS - 8, 0xf8),
        (RAM_START_ADDRESS + 1216, 0xc0),
        (RAM_START_ADDRESS + 1976, 0xb8),
    ] {
        assert!(image.contains(&byte), "{byte:x?}");
    }

    let mut backend = X86TracerBackend::new();
    backend.set_min_checkpoint_spacing_rows(1);
    let native = backend
        .trace(&program, inputs.clone())
        .expect("native trace");
    assert_eq!(native.trace.rows(), reference.trace.rows());
    assert_eq!(native.dimensions, reference.dimensions);
    assert_eq!(native.final_memory, reference.final_memory);
    assert_eq!(native.device, reference.device);

    let summary = backend
        .execute(&program, inputs, 1)
        .expect("native execute");
    assert_eq!(summary.final_memory, reference.final_memory);
    let replayed: Vec<JoltTraceRow> = summary
        .checkpoints
        .iter()
        .flat_map(|checkpoint| {
            backend
                .replay_chunk(checkpoint)
                .expect("native replay")
                .rows()
                .to_vec()
        })
        .collect();
    assert_eq!(replayed, reference.trace.rows());
}
