use serde::{Deserialize, Serialize};
use tracer::{ELFInstruction, RVTraceRow, RegisterState, RV32IM};

use super::VirtualInstructionSequence;
use crate::jolt::instruction::{
    add::ADDInstruction, movsign::MOVSIGNInstruction, mulhu::MULHUInstruction,
    mulu::MULUInstruction, JoltInstruction,
};

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize)]
pub struct MULHInstruction<const WORD_SIZE: usize>(pub u64, pub u64);

impl<const WORD_SIZE: usize> VirtualInstructionSequence for MULHInstruction<WORD_SIZE> {
    fn virtual_sequence(trace_row: tracer::RVTraceRow) -> Vec<tracer::RVTraceRow> {
        assert_eq!(trace_row.instruction.opcode, RV32IM::MULH);
        // MULH operands
        let x = trace_row.register_state.rs1_val.unwrap();
        let y = trace_row.register_state.rs2_val.unwrap();
        // MULH source registers
        let r_x = trace_row.instruction.rs1;
        let r_y = trace_row.instruction.rs2;
        // Virtual registers used in sequence
        let v_sx = Some(32);
        let v_sy = Some(33);
        let v_0 = Some(34);
        let v_1 = Some(35);
        let v_2 = Some(36);
        let v_3 = Some(37);

        let mut virtual_sequence = vec![];

        let s_x = MOVSIGNInstruction::<WORD_SIZE>(x).lookup_entry();
        virtual_sequence.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::VIRTUAL_MOVSIGN,
                rs1: r_x,
                rs2: None,
                rd: v_sx,
                imm: None,
                virtual_sequence_index: Some(0),
            },
            register_state: RegisterState {
                rs1_val: Some(x),
                rs2_val: None,
                rd_post_val: Some(s_x),
            },
            memory_state: None,
        });

        let s_y = MOVSIGNInstruction::<WORD_SIZE>(y).lookup_entry();
        virtual_sequence.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::VIRTUAL_MOVSIGN,
                rs1: r_y,
                rs2: None,
                rd: v_sy,
                imm: None,
                virtual_sequence_index: Some(1),
            },
            register_state: RegisterState {
                rs1_val: Some(y),
                rs2_val: None,
                rd_post_val: Some(s_y),
            },
            memory_state: None,
        });

        let xy_high_bits = MULHUInstruction::<WORD_SIZE>(x, y).lookup_entry();
        virtual_sequence.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::MULHU,
                rs1: r_x,
                rs2: r_y,
                rd: v_0,
                imm: None,
                virtual_sequence_index: Some(2),
            },
            register_state: RegisterState {
                rs1_val: Some(x),
                rs2_val: Some(y),
                rd_post_val: Some(xy_high_bits),
            },
            memory_state: None,
        });

        let sx_y_low_bits = MULUInstruction::<WORD_SIZE>(s_x, y).lookup_entry();
        virtual_sequence.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::MULU,
                rs1: v_sx,
                rs2: r_y,
                rd: v_1,
                imm: None,
                virtual_sequence_index: Some(3),
            },
            register_state: RegisterState {
                rs1_val: Some(s_x),
                rs2_val: Some(y),
                rd_post_val: Some(sx_y_low_bits),
            },
            memory_state: None,
        });

        let sy_x_low_bits = MULUInstruction::<WORD_SIZE>(s_y, x).lookup_entry();
        virtual_sequence.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::MULU,
                rs1: v_sy,
                rs2: r_x,
                rd: v_2,
                imm: None,
                virtual_sequence_index: Some(4),
            },
            register_state: RegisterState {
                rs1_val: Some(s_y),
                rs2_val: Some(x),
                rd_post_val: Some(sy_x_low_bits),
            },
            memory_state: None,
        });

        let partial_sum = ADDInstruction::<WORD_SIZE>(xy_high_bits, sx_y_low_bits).lookup_entry();
        virtual_sequence.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::ADD,
                rs1: v_0,
                rs2: v_1,
                rd: v_3,
                imm: None,
                virtual_sequence_index: Some(5),
            },
            register_state: RegisterState {
                rs1_val: Some(xy_high_bits),
                rs2_val: Some(sx_y_low_bits),
                rd_post_val: Some(partial_sum),
            },
            memory_state: None,
        });

        let result = ADDInstruction::<WORD_SIZE>(partial_sum, sy_x_low_bits).lookup_entry();
        virtual_sequence.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::ADD,
                rs1: v_3,
                rs2: v_2,
                rd: trace_row.instruction.rd,
                imm: None,
                virtual_sequence_index: Some(6),
            },
            register_state: RegisterState {
                rs1_val: Some(partial_sum),
                rs2_val: Some(sy_x_low_bits),
                rd_post_val: Some(result),
            },
            memory_state: None,
        });
        virtual_sequence
    }
}
