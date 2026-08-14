//! Per-instruction iai-callgrind microbenchmarks: callgrind instruction
//! counts per row kind, measured over a straight-line program of 4096 copies
//! of the row (compilation and memory-plane setup stay outside the measured
//! region via `Prepared`). Control-flow kinds that cannot repeat
//! straight-line use a `Jal +4` chain or a single dispatched row (`Jalr`).
//!
//! Requires valgrind and the `iai` feature; linux-x86_64 only (the feature
//! is meaningless elsewhere):
//!
//! ```sh
//! cargo bench -p jolt-tracer-x86 --features iai --bench iai
//! ```

#![cfg(all(target_arch = "x86_64", target_os = "linux"))]
#![expect(clippy::expect_used)]

use common::constants::REGISTER_COUNT;
use iai_callgrind::{library_benchmark, library_benchmark_group, main};
use jolt_riscv::{JoltInstructionKind, JoltInstructionRow, NormalizedOperands};
use jolt_tracer_x86::harness::{
    single_row_program, straight_line_program, Prepared, SCRATCH_START, TEST_ADDR,
};
use std::hint::black_box;

const COUNT: usize = 4096;
const REGS: usize = REGISTER_COUNT as usize;

type Operands = (Option<u8>, Option<u8>, Option<u8>, i128);

fn row(kind: JoltInstructionKind, operands: Operands) -> JoltInstructionRow {
    let (rs1, rs2, rd, imm) = operands;
    JoltInstructionRow {
        instruction_kind: kind,
        address: TEST_ADDR as usize,
        operands: NormalizedOperands { rs1, rs2, rd, imm },
        virtual_sequence_remaining: None,
        is_first_in_sequence: true,
        is_compressed: false,
    }
}

fn bench_regs() -> [u64; REGS] {
    let mut pre = [0u64; REGS];
    // Distinct operand values; the branch benches rely on x5 != x6.
    pre[5] = 1;
    pre[6] = 2;
    pre[7] = 1 << 20; // one-hot bitmask for the register-form shift
    pre[8] = SCRATCH_START; // aligned base for memory ops
    pre
}

/// Setup: straight-line repetition of a non-control-flow row (or a `Jal +4`
/// chain, which is also address-sequential).
fn straight(kind: JoltInstructionKind, operands: Operands) -> Prepared {
    let program = straight_line_program(row(kind, operands), COUNT);
    Prepared::new(&program, bench_regs()).expect("prepare failed")
}

/// Setup: one dispatched `Jalr` through the jump table.
fn jalr_single() -> Prepared {
    let mut pre = bench_regs();
    pre[5] = TEST_ADDR + 8;
    let program = single_row_program(row(JoltInstructionKind::JALR, (Some(5), None, Some(9), 0)));
    Prepared::new(&program, pre).expect("prepare failed")
}

fn run(mut prepared: Prepared) -> u64 {
    black_box(prepared.run_once().expect("run failed"))
}

use JoltInstructionKind as K;

/// For kinds whose associated-const name collides with the enum variant name
/// (the variant constructor wins path resolution), resolve by name instead.
fn kind_by_name(name: &str) -> JoltInstructionKind {
    JoltInstructionKind::from_name(name).expect("unknown kind name")
}

macro_rules! straight_benches {
    ($($bench:ident => ($kind:expr, $rs1:expr, $rs2:expr, $rd:expr, $imm:expr);)*) => {
        $(
            #[library_benchmark]
            #[bench::steady(args = ($kind, ($rs1, $rs2, $rd, $imm)), setup = straight)]
            fn $bench(prepared: Prepared) -> u64 {
                run(prepared)
            }
        )*

        library_benchmark_group!(
            name = per_instruction;
            benchmarks = $($bench),*
        );
    };
}

straight_benches! {
    iai_add => (K::ADD, Some(5), Some(6), Some(9), 0);
    iai_sub => (K::SUB, Some(5), Some(6), Some(9), 0);
    iai_and => (K::AND, Some(5), Some(6), Some(9), 0);
    iai_or => (K::OR, Some(5), Some(6), Some(9), 0);
    iai_xor => (K::XOR, Some(5), Some(6), Some(9), 0);
    iai_mul => (K::MUL, Some(5), Some(6), Some(9), 0);
    iai_mulhu => (K::MULHU, Some(5), Some(6), Some(9), 0);
    iai_sltu => (K::SLTU, Some(5), Some(6), Some(9), 0);
    iai_addi => (K::ADDI, Some(5), None, Some(9), 42);
    iai_andi => (K::ANDI, Some(5), None, Some(9), 42);
    iai_ori => (K::ORI, Some(5), None, Some(9), 42);
    iai_xori => (K::XORI, Some(5), None, Some(9), 42);
    iai_slti => (K::SLTI, Some(5), None, Some(9), 42);
    iai_sltiu => (K::SLTIU, Some(5), None, Some(9), 42);
    iai_muli => (K::VirtualMULI, Some(5), None, Some(9), 42);
    iai_lui => (K::LUI, None, None, Some(9), 0x1234_5000);
    iai_auipc => (K::AUIPC, None, None, Some(9), 0x1000);
    iai_pow2 => (K::VirtualPow2, Some(5), None, Some(9), 0);
    iai_shift_right_bitmask => (kind_by_name("VirtualShiftRightBitmask"), Some(5), None, Some(9), 0);
    iai_sign_extend_word => (kind_by_name("VirtualSignExtendWord"), Some(5), None, Some(9), 0);
    iai_zero_extend_word => (kind_by_name("VirtualZeroExtendWord"), Some(5), None, Some(9), 0);
    iai_srai => (K::VirtualSRAI, Some(5), None, Some(9), 1 << 7);
    iai_srli => (K::VirtualSRLI, Some(5), None, Some(9), 1 << 7);
    iai_srl => (K::VirtualSRL, Some(5), Some(7), Some(9), 0);
    iai_beq_not_taken => (K::BEQ, Some(5), Some(6), None, 8);
    iai_bne_taken_next => (K::BNE, Some(5), Some(6), None, 4);
    iai_ld => (K::LD, Some(8), None, Some(9), 0);
    iai_sd => (K::SD, Some(8), Some(5), None, 0);
    iai_assert_halfword_alignment => (K::VirtualAssertHalfwordAlignment, Some(8), None, None, 0);
    iai_assert_word_alignment => (K::VirtualAssertWordAlignment, Some(8), None, None, 0);
    iai_assert_lte => (K::VirtualAssertLTE, Some(5), Some(6), None, 0);
    iai_fence => (K::FENCE, None, None, None, 0);
    iai_host_io_unknown => (kind_by_name("VirtualHostIO"), None, None, None, 0);
    iai_jal_chain => (K::JAL, None, None, Some(9), 4);
}

#[library_benchmark]
#[bench::single(setup = jalr_single)]
fn iai_jalr(prepared: Prepared) -> u64 {
    run(prepared)
}

library_benchmark_group!(
    name = per_instruction_control;
    benchmarks = iai_jalr
);

main!(
    library_benchmark_groups = per_instruction,
    per_instruction_control
);
