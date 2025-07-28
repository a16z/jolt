use jolt_core::jolt::instruction::LookupQuery;
use onnx_tracer::{
    constants::virtual_tensor_index,
    tensor::Tensor,
    trace_types::{MemoryState, ONNXCycle, ONNXInstr, ONNXOpcode},
};

use crate::jolt::instruction::{
    VirtualInstructionSequence, add::ADD, beq::BEQInstruction, mul::MUL,
    virtual_advice::ADVICEInstruction, virtual_assert_valid_div0::AssertValidDiv0Instruction,
    virtual_assert_valid_signed_remainder::AssertValidSignedRemainderInstruction,
};

/// Perform signed division and return the result
pub struct DIVInstruction<const WORD_SIZE: usize>;

impl<const WORD_SIZE: usize> VirtualInstructionSequence for DIVInstruction<WORD_SIZE> {
    const SEQUENCE_LENGTH: usize = 8;

    fn virtual_trace(cycle: ONNXCycle) -> Vec<ONNXCycle> {
        assert_eq!(cycle.instr.opcode, ONNXOpcode::Div);
        // DIV source registers
        let r_x = cycle.instr.ts1;
        let r_y = cycle.instr.ts2;
        // Virtual registers used in sequence
        let v_0 = Some(virtual_tensor_index(0));
        let v_q: Option<usize> = Some(virtual_tensor_index(1));
        let v_r: Option<usize> = Some(virtual_tensor_index(2));
        let v_qy = Some(virtual_tensor_index(3));
        // DIV operands
        let x = cycle.memory_state.ts1_val.unwrap()[0];
        let y = cycle.memory_state.ts2_val.unwrap()[0];

        let mut virtual_trace = vec![];

        let (quotient, remainder) = match WORD_SIZE {
            32 => {
                if y == 0 {
                    (u32::MAX as u64, x as u64) // TODO
                } else {
                    let mut quotient = x as i32 / y as i32;
                    let mut remainder = x as i32 % y as i32;
                    if (remainder < 0 && (y as i32) > 0) || (remainder > 0 && (y as i32) < 0) {
                        remainder += y as i32;
                        quotient -= 1;
                    }
                    (quotient as u32 as u64, remainder as u32 as u64)
                }
            }
            64 => {
                if y == 0 {
                    (u64::MAX, x as u64)
                } else {
                    let mut quotient = x as i64 / y as i64;
                    let mut remainder = x as i64 % y as i64;
                    if (remainder < 0 && (y as i64) > 0) || (remainder > 0 && (y as i64) < 0) {
                        remainder += y as i64;
                        quotient -= 1;
                    }
                    (quotient as u64, remainder as u64)
                }
            }
            _ => panic!("Unsupported WORD_SIZE: {}", WORD_SIZE),
        };

        let q = ADVICEInstruction::<WORD_SIZE>(quotient).to_lookup_output();
        virtual_trace.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::VirtualAdvice,
                ts1: None,
                ts2: None,
                td: v_q,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
            },
            memory_state: MemoryState {
                ts1_val: None,
                ts2_val: None,
                td_post_val: Some(Tensor::from(vec![q as i128].into_iter())), // FIXME
            },
            advice_value: Some(quotient as i128), // FIXME
        });

        let r = ADVICEInstruction::<WORD_SIZE>(remainder).to_lookup_output();
        virtual_trace.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::VirtualAdvice,
                ts1: None,
                ts2: None,
                td: v_r,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
            },
            memory_state: MemoryState {
                ts1_val: None,
                ts2_val: None,
                td_post_val: Some(Tensor::from(vec![r as i128].into_iter())), // FIXME
            },
            advice_value: Some(remainder as i128), // FIXME: not sure i128 is the right type here
        });

        let is_valid: u64 =
            AssertValidSignedRemainderInstruction::<WORD_SIZE>(r, y as u64 /* FIXME */)
                .to_lookup_output();
        assert_eq!(is_valid, 1);
        // virtual_trace.push(RVTraceRow {
        //     instruction: ELFInstruction {
        //         address: trace_row.instruction.address,
        //         opcode: RV32IM::VIRTUAL_ASSERT_VALID_SIGNED_REMAINDER,
        //         rs1: v_r,
        //         rs2: r_y,
        //         rd: None,
        //         imm: None,
        //         virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
        //     },
        //     register_state: RegisterState {
        //         rs1_val: Some(r),
        //         rs2_val: Some(y),
        //         rd_post_val: None,
        //     },
        //     memory_state: None,
        //     advice_value: None,
        // });
        virtual_trace.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::VirtualAssertValidSignedRemainder,
                ts1: v_r,
                ts2: r_y,
                td: None,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
            },
            memory_state: MemoryState {
                ts1_val: Some(Tensor::from(vec![r as i128].into_iter())),
                ts2_val: Some(Tensor::from(vec![y as i128].into_iter())),
                td_post_val: None,
            },
            advice_value: None,
        });

        let is_valid: u64 =
            AssertValidDiv0Instruction::<WORD_SIZE>(y as u64 /* FIXME */, q).to_lookup_output();
        assert_eq!(is_valid, 1);
        // virtual_trace.push(RVTraceRow {
        //     instruction: ELFInstruction {
        //         address: trace_row.instruction.address,
        //         opcode: RV32IM::VIRTUAL_ASSERT_VALID_DIV0,
        //         rs1: r_y,
        //         rs2: v_q,
        //         rd: None,
        //         imm: None,
        //         virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
        //     },
        //     register_state: RegisterState {
        //         rs1_val: Some(y),
        //         rs2_val: Some(q),
        //         rd_post_val: None,
        //     },
        //     memory_state: None,
        //     advice_value: None,
        // });
        virtual_trace.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::VirtualAssertValidSignedRemainder,
                ts1: r_y,
                ts2: v_q,
                td: None,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
            },
            memory_state: MemoryState {
                ts1_val: Some(Tensor::from(vec![y as i128].into_iter())),
                ts2_val: Some(Tensor::from(vec![q as i128].into_iter())),
                td_post_val: None,
            },
            advice_value: None,
        });

        let q_y = MUL::<WORD_SIZE>(q, y as u64 /* FIXME */).to_lookup_output();
        virtual_trace.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::Mul,
                ts1: v_q,
                ts2: r_y,
                td: v_qy,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
            },
            memory_state: MemoryState {
                ts1_val: Some(Tensor::from(vec![q as i128].into_iter())),
                ts2_val: Some(Tensor::from(vec![y as i128].into_iter())),
                td_post_val: Some(Tensor::from(vec![q_y as i128].into_iter())),
            },
            advice_value: None,
        });

        let add_0 = ADD::<WORD_SIZE>(q_y, r).to_lookup_output();
        // virtual_trace.push(RVTraceRow {
        //     instruction: ELFInstruction {
        //         address: trace_row.instruction.address,
        //         opcode: RV32IM::ADD,
        //         rs1: v_qy,
        //         rs2: v_r,
        //         rd: v_0,
        //         imm: None,
        //         virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
        //     },
        //     register_state: RegisterState {
        //         rs1_val: Some(q_y),
        //         rs2_val: Some(r),
        //         rd_post_val: Some(add_0),
        //     },
        //     memory_state: None,
        //     advice_value: None,
        // });
        virtual_trace.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::Add,
                ts1: v_qy,
                ts2: v_r,
                td: v_0,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
            },
            memory_state: MemoryState {
                ts1_val: Some(Tensor::from(vec![q_y as i128].into_iter())),
                ts2_val: Some(Tensor::from(vec![r as i128].into_iter())),
                td_post_val: Some(Tensor::from(vec![add_0 as i128].into_iter())),
            },
            advice_value: None,
        });

        let _assert_eq =
            BEQInstruction::<WORD_SIZE>(add_0, x as u64 /* FIXME */).to_lookup_output();
        // virtual_trace.push(RVTraceRow {
        //     instruction: ELFInstruction {
        //         address: trace_row.instruction.address,
        //         opcode: RV32IM::VIRTUAL_ASSERT_EQ,
        //         rs1: v_0,
        //         rs2: r_x,
        //         rd: None,
        //         imm: None,
        //         virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
        //     },
        //     register_state: RegisterState {
        //         rs1_val: Some(add_0),
        //         rs2_val: Some(x),
        //         rd_post_val: None,
        //     },
        //     memory_state: None,
        //     advice_value: None,
        // });
        virtual_trace.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::VirtualAssertEq,
                ts1: v_0,
                ts2: r_x,
                td: None,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
            },
            memory_state: MemoryState {
                ts1_val: Some(Tensor::from(vec![add_0 as i128].into_iter())),
                ts2_val: Some(Tensor::from(vec![x as i128].into_iter())),
                td_post_val: None,
            },
            advice_value: None,
        });

        // virtual_trace.push(RVTraceRow {
        //     instruction: ELFInstruction {
        //         address: trace_row.instruction.address,
        //         opcode: RV32IM::VIRTUAL_MOVE,
        //         rs1: v_q,
        //         rs2: None,
        //         rd: trace_row.instruction.rd,
        //         imm: None,
        //         virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
        //     },
        //     register_state: RegisterState {
        //         rs1_val: Some(q),
        //         rs2_val: None,
        //         rd_post_val: Some(q),
        //     },
        //     memory_state: None,
        //     advice_value: None,
        // });
        virtual_trace.push(ONNXCycle {
            instr: ONNXInstr {
                address: cycle.instr.address,
                opcode: ONNXOpcode::VirtualMove,
                ts1: v_q,
                ts2: None,
                td: cycle.instr.td,
                imm: None,
                virtual_sequence_remaining: Some(Self::SEQUENCE_LENGTH - virtual_trace.len() - 1),
            },
            memory_state: MemoryState {
                ts1_val: Some(Tensor::from(vec![q as i128].into_iter())),
                ts2_val: None,
                td_post_val: Some(Tensor::from(vec![q as i128].into_iter())),
            },
            advice_value: None,
        });

        virtual_trace
    }

    fn sequence_output(x: u64, y: u64) -> u64 {
        let x = x as i32;
        let y = y as i32;
        let mut quotient = x / y;
        let remainder = x % y;
        if (remainder < 0 && y > 0) || (remainder > 0 && y < 0) {
            quotient -= 1;
        }
        quotient as u32 as u64
    }
}

// #[cfg(test)]
// mod test {
//     use super::*;
//     use crate::{jolt::instruction::JoltInstruction, jolt_virtual_sequence_test};

//     #[test]
//     fn div_virtual_sequence_32() {
//         jolt_virtual_sequence_test!(DIVInstruction::<32>, RV32IM::DIV);
//     }
// }
