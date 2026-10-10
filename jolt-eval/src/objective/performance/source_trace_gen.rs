use common::jolt_device::MemoryConfig;
use jolt_program::execution::{JoltProgram, OwnedTrace, SourceTraceRow, TraceInputs};
use jolt_riscv::RV64I;
use tracer::emulator::elf_analyzer::test_elf::{self, StrtabOrder};
use tracer::SourceTracerBackend;

use super::trace_gen::raw_trace_cycles;
use crate::objective::{Objective, OptimizationObjective, PerformanceObjective};

pub const SOURCE_TRACE_GEN: OptimizationObjective = OptimizationObjective::Performance(
    PerformanceObjective::SourceTraceGen(SourceTraceGenObjective),
);

const ALU: &[u32] = &[
    0x000802b7, // lui x5, 0x80
    0x00100313, // addi x6, x0, 1
    0xfff00393, // addi x7, x0, -1
    0x00730333, // add x6, x6, x7
    0x0063c3b3, // xor x7, x7, x6
    0x01f31413, // slli x8, x6, 31
    0x02045413, // srli x8, x8, 32
    0xfff3049b, // addiw x9, x6, -1
    0x4074d4bb, // sraw x9, x9, x7
    0xfff28293, // addi x5, x5, -1
    0xfe0292e3, // bne x5, x0, -28
    0x0000006f, // jal x0, 0
];

const MEMORY: &[u32] = &[
    0x80010337, // lui x6, 0x80010
    0x02031313, // slli x6, x6, 32
    0x02035313, // srli x6, x6, 32
    0x000402b7, // lui x5, 0x40
    0xfff00393, // addi x7, x0, -1
    0x00730023, // sb x7, 0(x6)
    0x00731023, // sh x7, 0(x6)
    0x00732023, // sw x7, 0(x6)
    0x00733023, // sd x7, 0(x6)
    0x00030403, // lb x8, 0(x6)
    0x00034403, // lbu x8, 0(x6)
    0x00031403, // lh x8, 0(x6)
    0x00035403, // lhu x8, 0(x6)
    0x00032403, // lw x8, 0(x6)
    0x00036403, // lwu x8, 0(x6)
    0x00033403, // ld x8, 0(x6)
    0x00138393, // addi x7, x7, 1
    0x0083c4b3, // xor x9, x7, x8
    0x0ff4f493, // andi x9, x9, 255
    0xfff28293, // addi x5, x5, -1
    0xfc0292e3, // bne x5, x0, -60
    0x0000006f, // jal x0, 0
];

const CALL_FRAME: &[u32] = &[
    0x80002137, // lui x2, 0x80002
    0x02011113, // slli x2, x2, 32
    0x02015113, // srli x2, x2, 32
    0x80010113, // addi x2, x2, -2048
    0x000602b7, // lui x5, 0x60
    0x010000ef, // jal x1, 16
    0xfff28293, // addi x5, x5, -1
    0xfe029ce3, // bne x5, x0, -8
    0x0000006f, // jal x0, 0
    0xff010113, // addi x2, x2, -16
    0x00113423, // sd x1, 8(x2)
    0x00613023, // sd x6, 0(x2)
    0x00130313, // addi x6, x6, 1
    0x00013303, // ld x6, 0(x2)
    0x00813083, // ld x1, 8(x2)
    0x01010113, // addi x2, x2, 16
    0x00008067, // jalr x0, 0(x1)
];

// Counts include the prologue and final self-jump, checked by the Criterion setup.
pub(super) const PROGRAMS: [(&str, &[u32], usize); 3] = [
    ("alu", ALU, 3 + 8 * (1 << 19) + 1),
    ("memory", MEMORY, 5 + 16 * (1 << 18) + 1),
    ("call_frame", CALL_FRAME, 5 + 11 * (0x60 << 12) + 1),
];

/// A hand-assembled RV64I benchmark and its executed source-instruction count.
pub struct SourceTraceGenSetup {
    pub label: &'static str,
    pub program: JoltProgram,
    pub inputs: TraceInputs,
    pub row_count: usize,
}

/// Measures source trace generation for ALU, memory, and call/frame loops.
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct SourceTraceGenObjective;

impl SourceTraceGenObjective {
    /// Emits a reserved source trace; each program's count is checked before timing.
    pub fn run_source(&self, setup: &SourceTraceGenSetup) -> OwnedTrace<SourceTraceRow> {
        let mut backend = SourceTracerBackend::with_row_capacity(setup.row_count);
        setup
            .program
            .trace_with(&mut backend, setup.inputs.clone())
            .expect("source trace benchmark program failed")
            .trace
    }

    /// Emits the reference tracer's cycles without converting them to seam rows.
    pub fn run_reference(&self, setup: &SourceTraceGenSetup) -> usize {
        raw_trace_cycles(&setup.program, &setup.inputs)
    }

    /// Reads all ten numeric accessors exactly once per source row.
    pub fn run_scan(&self, rows: &[SourceTraceRow]) -> u64 {
        let mut sum = 0u64;
        for row in rows {
            sum = sum
                .wrapping_add(u64::from(row.instruction_index()))
                .wrapping_add(row.pc())
                .wrapping_add(row.next_pc())
                .wrapping_add(row.rs1_value())
                .wrapping_add(row.rs2_value())
                .wrapping_add(row.rd_pre_value())
                .wrapping_add(row.rd_post_value())
                .wrapping_add(row.ram_address())
                .wrapping_add(row.ram_pre_value())
                .wrapping_add(row.ram_post_value());
        }
        std::hint::black_box(sum)
    }
}

impl Objective for SourceTraceGenObjective {
    type Setup = [SourceTraceGenSetup; 3];

    fn name(&self) -> &str {
        "source_trace_gen"
    }

    fn description(&self) -> String {
        "Source trace generation over three hand-assembled RV64I programs (about 2^22 instructions each)".to_owned()
    }

    fn setup(&self) -> Self::Setup {
        PROGRAMS.map(|(label, words, row_count)| {
            let program = JoltProgram::from_elf_bytes_with_profile(
                test_elf::build_elf64(words, &[], StrtabOrder::GnuLd),
                RV64I,
            );
            let config = MemoryConfig {
                program_size: Some(0x1000),
                stack_size: 0x1000,
                heap_size: 0x10000,
                max_input_size: 8,
                max_output_size: 8,
                max_trusted_advice_size: 0,
                max_untrusted_advice_size: 0,
            };
            SourceTraceGenSetup {
                label,
                program,
                inputs: TraceInputs::new(Vec::new(), Vec::new(), Vec::new(), config),
                row_count,
            }
        })
    }

    fn run(&self, setup: Self::Setup) {
        for program in &setup {
            std::hint::black_box(self.run_source(program));
        }
    }

    fn units(&self) -> Option<&str> {
        Some("s")
    }
}
