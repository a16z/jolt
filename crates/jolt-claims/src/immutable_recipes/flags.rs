use jolt_riscv::{CircuitFlags, InstructionFlags};

use super::{KeySink, KeyVisit};

impl KeyVisit for CircuitFlags {
    fn visit(&self, sink: &mut dyn KeySink) -> bool {
        sink.word(u64::from(*self as u8))
    }
}

crate::recipe_id! { InstructionFlags {
    LeftOperandIsPC = 0,
    RightOperandIsImm = 1,
    LeftOperandIsRs1Value = 2,
    RightOperandIsRs2Value = 3,
    Branch = 4,
    IsNoop = 5,
    RightOperandIsRdWriteValue = 6,
} }
