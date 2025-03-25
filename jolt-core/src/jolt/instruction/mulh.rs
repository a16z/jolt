use common::constants::virtual_register_index;
use tracer::{ELFInstruction, RVTraceRow, RegisterState, RV32IM};

use super::VirtualInstructionSequence;
use crate::jolt::instruction::{
    add::ADDInstruction, mulhu::MULHUInstruction, mulu::MULUInstruction,
    virtual_movsign::MOVSIGNInstruction, JoltInstruction,
};

pub struct MULHInstruction<const WORD_SIZE: usize>;

impl<const WORD_SIZE: usize> VirtualInstructionSequence for MULHInstruction<WORD_SIZE> {
    const SEQUENCE_LENGTH: usize = 7;

    fn virtual_trace(trace_row: RVTraceRow) -> Vec<RVTraceRow> {
        assert_eq!(trace_row.instruction.opcode, RV32IM::MULH);
        // MULH source registers
        let r_x = trace_row.instruction.rs1;
        let r_y = trace_row.instruction.rs2;
        // Virtual registers used in sequence
        let v_sx = Some(virtual_register_index(0));
        let v_sy = Some(virtual_register_index(1));
        let v_0 = Some(virtual_register_index(2));
        let v_1 = Some(virtual_register_index(3));
        let v_2 = Some(virtual_register_index(4));
        let v_3 = Some(virtual_register_index(5));
        // MULH operands
        let x = trace_row.register_state.rs1_val.unwrap();
        let y = trace_row.register_state.rs2_val.unwrap();

        let mut virtual_trace = vec![];

        let s_x = MOVSIGNInstruction::<WORD_SIZE>(x).lookup_entry();
        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::VIRTUAL_MOVSIGN,
                rs1: r_x,
                rs2: None,
                rd: v_sx,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
            },
            register_state: RegisterState {
                rs1_val: Some(x),
                rs2_val: None,
                rd_post_val: Some(s_x),
            },
            memory_state: None,
            advice_value: None,
            precompile_input: None,
            precompile_output_address: None,
        });

        let s_y = MOVSIGNInstruction::<WORD_SIZE>(y).lookup_entry();
        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::VIRTUAL_MOVSIGN,
                rs1: r_y,
                rs2: None,
                rd: v_sy,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
            },
            register_state: RegisterState {
                rs1_val: Some(y),
                rs2_val: None,
                rd_post_val: Some(s_y),
            },
            memory_state: None,
            advice_value: None,
            precompile_input: None,
            precompile_output_address: None,
        });

        let xy_high_bits = MULHUInstruction::<WORD_SIZE>(x, y).lookup_entry();
        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::MULHU,
                rs1: r_x,
                rs2: r_y,
                rd: v_0,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
            },
            register_state: RegisterState {
                rs1_val: Some(x),
                rs2_val: Some(y),
                rd_post_val: Some(xy_high_bits),
            },
            memory_state: None,
            advice_value: None,
            precompile_input: None,
            precompile_output_address: None,
        });

        let sx_y_low_bits = MULUInstruction::<WORD_SIZE>(s_x, y).lookup_entry();
        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::MULU,
                rs1: v_sx,
                rs2: r_y,
                rd: v_1,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
            },
            register_state: RegisterState {
                rs1_val: Some(s_x),
                rs2_val: Some(y),
                rd_post_val: Some(sx_y_low_bits),
            },
            memory_state: None,
            advice_value: None,
            precompile_input: None,
            precompile_output_address: None,
        });

        let sy_x_low_bits = MULUInstruction::<WORD_SIZE>(s_y, x).lookup_entry();
        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::MULU,
                rs1: v_sy,
                rs2: r_x,
                rd: v_2,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
            },
            register_state: RegisterState {
                rs1_val: Some(s_y),
                rs2_val: Some(x),
                rd_post_val: Some(sy_x_low_bits),
            },
            memory_state: None,
            advice_value: None,
            precompile_input: None,
            precompile_output_address: None,
        });

        let partial_sum = ADDInstruction::<WORD_SIZE>(xy_high_bits, sx_y_low_bits).lookup_entry();
        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::ADD,
                rs1: v_0,
                rs2: v_1,
                rd: v_3,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
            },
            register_state: RegisterState {
                rs1_val: Some(xy_high_bits),
                rs2_val: Some(sx_y_low_bits),
                rd_post_val: Some(partial_sum),
            },
            memory_state: None,
            advice_value: None,
            precompile_input: None,
            precompile_output_address: None,
        });

        let result = ADDInstruction::<WORD_SIZE>(partial_sum, sy_x_low_bits).lookup_entry();
        virtual_trace.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::ADD,
                rs1: v_3,
                rs2: v_2,
                rd: trace_row.instruction.rd,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
            },
            register_state: RegisterState {
                rs1_val: Some(partial_sum),
                rs2_val: Some(sy_x_low_bits),
                rd_post_val: Some(result),
            },
            memory_state: None,
            advice_value: None,
            precompile_input: None,
            precompile_output_address: None,
        });
        virtual_trace
    }

    fn sequence_output(x: u64, y: u64) -> u64 {
        match WORD_SIZE {
            32 => {
                let result = ((x as i32 as i64) * (y as i32 as i64)) >> 32;
                result as u32 as u64
            }
            64 => {
                let result = ((x as i64 as i128) * (y as i64 as i128)) >> 64;
                result as u64
            }
            _ => panic!("Unsupported WORD_SIZE: {}", WORD_SIZE),
        }
    }
}

#[cfg(test)]
mod test {
    use crate::jolt::instruction::test::jolt_virtual_sequence_test;

    use super::*;

    #[test]
    fn mulh_virtual_sequence_32() {
        jolt_virtual_sequence_test::<MULHInstruction<32>>(RV32IM::MULH);
    }
}
