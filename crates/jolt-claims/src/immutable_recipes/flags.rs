use jolt_riscv::{CircuitFlags, InstructionFlags};

use super::{KeySink, KeyVisit};

impl KeyVisit for CircuitFlags {
    fn visit(&self, sink: &mut dyn KeySink) -> bool {
        sink.word(u64::from(*self as u8))
    }
}

impl KeyVisit for InstructionFlags {
    fn visit(&self, sink: &mut dyn KeySink) -> bool {
        sink.word(u64::from(*self as u8))
    }
}
