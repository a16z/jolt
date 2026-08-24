#![expect(
    clippy::expect_used,
    reason = "trace coverage assertions should fail loudly"
)]

use jolt_program::execution::TraceRow;
use jolt_riscv::{instructions::VirtualShiftRightBitmaskW, JoltInstructionKind as InstructionKind};

pub fn assert_word_shift_trace_coverage(rows: &[TraceRow]) {
    let mask_kind = InstructionKind::VirtualShiftRightBitmaskW(VirtualShiftRightBitmaskW(()));
    let mask_shifts = rows
        .iter()
        .filter(|row| row.instruction.instruction_kind == mask_kind)
        .filter_map(|row| row.registers.rs1.map(|read| read.value))
        .collect::<Vec<_>>();

    for shift in [0, 31] {
        assert!(
            mask_shifts.iter().filter(|&&value| value == shift).count() >= 2,
            "word-shift mask lookup must cover shift {shift} for SRLW and SRAW"
        );
    }

    let boundary_masks = [(1u64 << 32) - 1, 1u64 << 31];
    for kind in [
        InstructionKind::VirtualSRLW,
        InstructionKind::VirtualSRAW,
        InstructionKind::VirtualSRLIW,
        InstructionKind::VirtualSRAIW,
    ] {
        let operands = rows
            .iter()
            .filter(|row| row.instruction.instruction_kind == kind)
            .map(|row| match kind {
                InstructionKind::VirtualSRLW | InstructionKind::VirtualSRAW => {
                    row.registers
                        .rs2
                        .expect("variable W-shift must read its mask")
                        .value
                }
                InstructionKind::VirtualSRLIW | InstructionKind::VirtualSRAIW => {
                    u64::try_from(row.instruction.operands.imm)
                        .expect("immediate W-shift mask must be non-negative")
                }
                _ => unreachable!(),
            })
            .collect::<Vec<_>>();

        for bitmask in boundary_masks {
            assert!(
                operands.contains(&bitmask),
                "{kind:?} must cover the shift boundary encoded by mask {bitmask:#x}"
            );
        }
    }
}
