use crate::{
    instruction::ELFInstruction,
    rv_trace::{RVTraceRow, RegisterState},
};

pub trait VirtualInstructionSequence {
    const SEQUENCE_LENGTH: usize;

    fn virtual_sequence(instruction: ELFInstruction) -> Vec<ELFInstruction> {
        let dummy_trace_row = RVTraceRow {
            instruction,
            register_state: RegisterState {
                rs1_val: Some(0),
                rs2_val: Some(0),
                rd_post_val: Some(0),
                rd_pre_val: Some(0),
            },
            memory_state: None,
            advice_value: None,
            precompile_input: None,
            precompile_output_address: None,
        };
        Self::virtual_trace(dummy_trace_row)
            .into_iter()
            .map(|trace_row| trace_row.instruction)
            .collect()
    }

    fn virtual_trace(trace_row: RVTraceRow) -> Vec<RVTraceRow> {
        vec![trace_row]
    }

    fn sequence_output(x: u64, y: u64) -> u64;
}

pub mod div;
pub mod divu;
pub mod lb;
pub mod lbu;
pub mod lh;
pub mod lhu;
pub mod mulh;
pub mod mulhsu;
pub mod rem;
pub mod remu;
pub mod sb;
pub mod sh;
pub mod sll;
pub mod sra;
pub mod srl;
