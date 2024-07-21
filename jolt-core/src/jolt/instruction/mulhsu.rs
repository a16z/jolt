use common::constants::virtual_register_index;
use tracer::{ELFInstruction, RVTraceRow, RegisterState, RV32IM};

use super::VirtualInstructionSequence;
use crate::jolt::instruction::{
    add::ADDInstruction, mulhu::MULHUInstruction, mulu::MULUInstruction,
    virtual_movsign::MOVSIGNInstruction, JoltInstruction,
};

/// Perform signed*unsigned multiplication and return the upper WORD_SIZE bits
pub struct MULHSUInstruction<const WORD_SIZE: usize>;

impl<const WORD_SIZE: usize> VirtualInstructionSequence for MULHSUInstruction<WORD_SIZE> {
    fn virtual_sequence(trace_row: RVTraceRow) -> Vec<RVTraceRow> {
        assert_eq!(trace_row.instruction.opcode, RV32IM::MULHSU);
        // MULHSU operands
        let x = trace_row.register_state.rs1_val.unwrap();
        let y = trace_row.register_state.rs2_val.unwrap();
        // MULHSU source registers
        let r_x = trace_row.instruction.rs1;
        let r_y = trace_row.instruction.rs2;
        // Virtual registers used in sequence
        let v_sx = Some(virtual_register_index(0));
        let v_1 = Some(virtual_register_index(1));
        let v_2 = Some(virtual_register_index(2));

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
            advice_value: None,
        });

        let xy_high_bits = MULHUInstruction::<WORD_SIZE>(x, y).lookup_entry();
        virtual_sequence.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::MULHU,
                rs1: r_x,
                rs2: r_y,
                rd: v_1,
                imm: None,
                virtual_sequence_index: Some(1),
            },
            register_state: RegisterState {
                rs1_val: Some(x),
                rs2_val: Some(y),
                rd_post_val: Some(xy_high_bits),
            },
            memory_state: None,
            advice_value: None,
        });

        let sx_y_low_bits = MULUInstruction::<WORD_SIZE>(s_x, y).lookup_entry();
        virtual_sequence.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::MULU,
                rs1: v_sx,
                rs2: r_y,
                rd: v_2,
                imm: None,
                virtual_sequence_index: Some(2),
            },
            register_state: RegisterState {
                rs1_val: Some(s_x),
                rs2_val: Some(y),
                rd_post_val: Some(sx_y_low_bits),
            },
            memory_state: None,
            advice_value: None,
        });

        let result = ADDInstruction::<WORD_SIZE>(xy_high_bits, sx_y_low_bits).lookup_entry();
        virtual_sequence.push(RVTraceRow {
            instruction: ELFInstruction {
                address: trace_row.instruction.address,
                opcode: RV32IM::ADD,
                rs1: v_1,
                rs2: v_2,
                rd: trace_row.instruction.rd,
                imm: None,
                virtual_sequence_index: Some(3),
            },
            register_state: RegisterState {
                rs1_val: Some(xy_high_bits),
                rs2_val: Some(sx_y_low_bits),
                rd_post_val: Some(result),
            },
            memory_state: None,
            advice_value: None,
        });
        virtual_sequence
    }
}

#[cfg(test)]
mod test {
    use ark_std::test_rng;
    use rand_chacha::rand_core::RngCore;

    use crate::{jolt::instruction::JoltInstruction, jolt_virtual_sequence_test};

    use super::*;

    #[test]
    fn mulhsu_virtual_sequence_32() {
        let mut rng = test_rng();

        let r_x = rng.next_u64() % 32;
        let r_y = rng.next_u64() % 32;
        let rd = rng.next_u64() % 32;

        let x = rng.next_u32() as u64;
        let y = if r_x == r_y { x } else { rng.next_u32() as u64 };
        let result = ((i128::from(x as i32) * i128::from(y)) >> 32) as u32;

        jolt_virtual_sequence_test!(
            MULHSUInstruction::<32>,
            RV32IM::MULHSU, 
            x, 
            y, 
            r_x, 
            r_y, 
            rd, 
            result
        );
    }
}
