use onnx_tracer::trace_types::{ONNXCycle, ONNXOpcode};

use crate::jolt::{
    instruction::{add::ADD, mul::MUL, sub::SUB},
    zkvm::onnx_vm::ONNXLookup,
};

pub trait LookupTrace {
    fn to_lookup(&self) -> Option<ONNXLookup>;
}

impl LookupTrace for ONNXCycle {
    fn to_lookup(&self) -> Option<ONNXLookup> {
        match self.instr.opcode {
            ONNXOpcode::Add => Some(ONNXLookup::Add(ADD(self.ts1_val(), self.ts2_val()))),
            ONNXOpcode::Sub => Some(ONNXLookup::Sub(SUB(self.ts1_val(), self.ts2_val()))),
            ONNXOpcode::Mul => Some(ONNXLookup::Mul(MUL(self.ts1_val(), self.ts2_val()))),
            _ => None,
        }
    }
}
