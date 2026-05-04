use serde::{Deserialize, Serialize};

macro_rules! define_instruction_kind {
    (
        instructions: [$($instr:ident),* $(,)?]
    ) => {
        #[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
        pub enum InstructionKind {
            #[default]
            NoOp,
            Unimpl,
            $(
                $instr,
            )*
            Inline,
        }
    };
}

crate::for_each_instruction_kind!(define_instruction_kind);
