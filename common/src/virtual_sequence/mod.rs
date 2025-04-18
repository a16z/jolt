use crate::instruction::{RISCVCycle, RISCVInstruction, RV32IMCycle, RV32IMInstruction};

pub trait VirtualInstructionSequence: RISCVInstruction {
    fn virtual_sequence(&self) -> Vec<RV32IMInstruction> {
        let dummy_cycle = RISCVCycle {
            instruction: *self,
            register_state: Default::default(),
            memory_state: Default::default(),
        };
        Self::virtual_trace(dummy_cycle)
            .into_iter()
            .map(|cycle| cycle.instruction())
            .collect()
    }

    fn virtual_trace(cycle: RISCVCycle<Self>) -> Vec<RV32IMCycle>;
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
