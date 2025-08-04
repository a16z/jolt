use onnx_tracer::trace_types::{ONNXCycle, ONNXOpcode};

use crate::jolt::instruction::{add::ADD, mul::MUL, sub::SUB};
use jolt_core::jolt::{
    instruction::{InstructionLookup, LookupQuery},
    lookup_table::LookupTables,
};

pub const WORD_SIZE: usize = 32;

macro_rules! define_lookup_enum {
    (
        enum $enum_name:ident,
        const $word_size:ident,
        trait $trait_name:ident,
        $($variant:ident : $inner:ty),+ $(,)?
    ) => {
        #[derive(Debug)]
        pub enum $enum_name {
            $(
                $variant($inner),
            )+
        }

        impl $trait_name<$word_size> for $enum_name {
            fn to_instruction_inputs(&self) -> (u64, i64) {
                match self {
                    $(
                        $enum_name::$variant(inner) => inner.to_instruction_inputs(),
                    )+
                }
            }

            fn to_lookup_index(&self) -> u64 {
                match self {
                    $(
                        $enum_name::$variant(inner) => inner.to_lookup_index(),
                    )+
                }
            }

            fn to_lookup_operands(&self) -> (u64, u64) {
                match self {
                    $(
                        $enum_name::$variant(inner) => inner.to_lookup_operands(),
                    )+
                }
            }

            fn to_lookup_output(&self) -> u64 {
                match self {
                    $(
                        $enum_name::$variant(inner) => inner.to_lookup_output(),
                    )+
                }
            }
        }
    };
}

define_lookup_enum!(
    enum ONNXLookup,
    const WORD_SIZE,
    trait LookupQuery,
    Add: ADD<WORD_SIZE>,
    Sub: SUB<WORD_SIZE>,
    Mul: MUL<WORD_SIZE>,
);

impl InstructionLookup<WORD_SIZE> for ONNXLookup {
    fn lookup_table(&self) -> Option<LookupTables<WORD_SIZE>> {
        match self {
            ONNXLookup::Add(add) => add.lookup_table(),
            ONNXLookup::Sub(sub) => sub.lookup_table(),
            ONNXLookup::Mul(mul) => mul.lookup_table(),
        }
    }
}

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
