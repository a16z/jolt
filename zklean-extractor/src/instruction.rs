use jolt_core::zkvm::{
    instruction::{
        CircuitFlags, InstructionFlags as _, InstructionLookup as _, InterleavedBitsMarker as _,
    },
    r1cs::inputs::JoltR1CSInputs,
};
use strum::IntoEnumIterator as _;
use tracer::instruction::RV32IMInstruction;

use crate::{
    constants::JoltParameterSet,
    lookups::ZkLeanLookupTable,
    mle_ast::DefaultMleAst,
    modules::{AsModule, Module},
    r1cs::input_to_field_name,
    util::{indent, ZkLeanReprField},
};

/// Represents how an instructions operands should be combined into a single vector.
#[derive(Debug, Clone, Copy)]
pub enum OperandInterleaving {
    /// Indicates that the operands should be concatenated:
    ///     rs1 || rs2
    Concatenated,
    /// Indicates that the operands should be interleaved:
    ///     rs1[0] || rs2[0] || rs1[1] || rs2[1] || ...
    Interleaved,
}

impl OperandInterleaving {
    /// Extract the operand interleaving for an instruction
    fn instruction_interleaving(instr: &RV32IMInstruction) -> Self {
        if instr.circuit_flags().is_interleaved_operands() {
            Self::Interleaved
        } else {
            Self::Concatenated
        }
    }
}

impl std::fmt::Display for OperandInterleaving {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Interleaved => write!(f, "Interleaved"),
            Self::Concatenated => write!(f, "Concatenated"),
        }
    }
}

/// Wrapper around a JoltInstruction
// TODO: Make this generic over the instruction set
#[derive(Debug, Clone)]
pub struct ZkLeanInstruction<J> {
    instruction: tracer::instruction::RV32IMInstruction,
    interleaving: OperandInterleaving,
    phantom: std::marker::PhantomData<J>,
}

impl<J> From<RV32IMInstruction> for ZkLeanInstruction<J> {
    fn from(value: RV32IMInstruction) -> Self {
        let interleaving = OperandInterleaving::instruction_interleaving(&value);

        Self {
            instruction: value,
            interleaving,
            phantom: std::marker::PhantomData,
        }
    }
}

impl<J: JoltParameterSet> ZkLeanInstruction<J> {
    pub fn name(&self) -> String {
        let name = <&'static str>::from(&self.instruction);
        let word_size = J::WORD_SIZE;

        format!("{name}_{word_size}")
    }

    pub fn iter() -> impl Iterator<Item = Self> {
        RV32IMInstruction::iter().filter_map(|instr| match instr {
            RV32IMInstruction::NoOp | RV32IMInstruction::UNIMPL
                // Inline sequences
                | RV32IMInstruction::DIV(_)
                | RV32IMInstruction::DIVU(_)
                | RV32IMInstruction::LB(_)
                | RV32IMInstruction::LBU(_)
                | RV32IMInstruction::LH(_)
                | RV32IMInstruction::LHU(_)
                | RV32IMInstruction::MULH(_)
                | RV32IMInstruction::MULHSU(_)
                | RV32IMInstruction::REM(_)
                | RV32IMInstruction::REMU(_)
                | RV32IMInstruction::SB(_)
                | RV32IMInstruction::SH(_)
                | RV32IMInstruction::SLL(_)
                | RV32IMInstruction::SLLI(_)
                | RV32IMInstruction::SRA(_)
                | RV32IMInstruction::SRAI(_)
                | RV32IMInstruction::SRL(_)
                | RV32IMInstruction::SRLI(_)
                | RV32IMInstruction::INLINE(_)

                // RV64I
                | RV32IMInstruction::ADDIW(_)
                | RV32IMInstruction::SLLIW(_)
                | RV32IMInstruction::SRLIW(_)
                | RV32IMInstruction::SRAIW(_)
                | RV32IMInstruction::ADDW(_)
                | RV32IMInstruction::SUBW(_)
                | RV32IMInstruction::SLLW(_)
                | RV32IMInstruction::SRLW(_)
                | RV32IMInstruction::SRAW(_)
                | RV32IMInstruction::LWU(_)
                | RV32IMInstruction::LD(_)
                | RV32IMInstruction::SD(_)

                // RV64M
                | RV32IMInstruction::DIVUW(_)
                | RV32IMInstruction::DIVW(_)
                | RV32IMInstruction::MULW(_)
                | RV32IMInstruction::REMUW(_)
                | RV32IMInstruction::REMW(_)

                // RV32A
                | RV32IMInstruction::LRW(_)
                | RV32IMInstruction::SCW(_)
                | RV32IMInstruction::AMOSWAPW(_)
                | RV32IMInstruction::AMOADDW(_)
                | RV32IMInstruction::AMOANDW(_)
                | RV32IMInstruction::AMOORW(_)
                | RV32IMInstruction::AMOXORW(_)
                | RV32IMInstruction::AMOMINW(_)
                | RV32IMInstruction::AMOMAXW(_)
                | RV32IMInstruction::AMOMINUW(_)
                | RV32IMInstruction::AMOMAXUW(_)

                // RV64A
                | RV32IMInstruction::LRD(_)
                | RV32IMInstruction::SCD(_)
                | RV32IMInstruction::AMOSWAPD(_)
                | RV32IMInstruction::AMOADDD(_)
                | RV32IMInstruction::AMOANDD(_)
                | RV32IMInstruction::AMOORD(_)
                | RV32IMInstruction::AMOXORD(_)
                | RV32IMInstruction::AMOMIND(_)
                | RV32IMInstruction::AMOMAXD(_)
                | RV32IMInstruction::AMOMINUD(_)
                | RV32IMInstruction::AMOMAXUD(_)
                => None,
            _ => Some(Self::from(instr)),
        })
    }

    /// Pretty print an instruction as a ZkLean `ComposedLookupTable`.
    pub fn zklean_pretty_print<F: ZkLeanReprField>(
        &self,
        f: &mut impl std::io::Write,
        mut indent_level: usize,
    ) -> std::io::Result<()> {
        let name = self.name();
        let interleaving = self.interleaving;
        let lookup_table = match self
            .instruction
            .lookup_table()
            .map(|t| ZkLeanLookupTable::from(t).name())
        {
            None => String::from("none"),
            Some(t) => format!("(some {t})"),
        };
        let circuit_flags = CircuitFlags::iter()
            .filter_map(|f| {
                if self.instruction.circuit_flags()[f] {
                    Some(input_to_field_name(&JoltR1CSInputs::OpFlags(f)))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join(", ");

        f.write_fmt(format_args!(
            "{}def {name} [Field f] : Instruction f :=\n",
            indent(indent_level),
        ))?;
        indent_level += 1;
        f.write_fmt(format_args!(
            "{}instructionFromMLE {interleaving} {lookup_table}\n",
            indent(indent_level),
        ))?;
        f.write_fmt(format_args!(
            "{}-- Circuit flags: {circuit_flags}\n",
            indent(indent_level),
        ))?;

        Ok(())
    }
}

pub struct ZkLeanInstructions<J> {
    instructions: Vec<ZkLeanInstruction<J>>,
}

impl<J: JoltParameterSet> ZkLeanInstructions<J> {
    pub fn extract() -> Self {
        Self {
            instructions: ZkLeanInstruction::<J>::iter().collect(),
        }
    }

    pub fn zklean_pretty_print(
        &self,
        f: &mut impl std::io::Write,
        indent_level: usize,
    ) -> std::io::Result<()> {
        for instruction in &self.instructions {
            instruction.zklean_pretty_print::<DefaultMleAst>(f, indent_level)?;
        }
        Ok(())
    }

    pub fn zklean_imports(&self) -> Vec<String> {
        vec![String::from("ZkLean"), String::from("Jolt.LookupTables")]
    }
}

impl<J: JoltParameterSet> AsModule for ZkLeanInstructions<J> {
    fn as_module(&self) -> std::io::Result<Module> {
        let mut contents: Vec<u8> = vec![];
        self.zklean_pretty_print(&mut contents, 0)?;

        Ok(Module {
            name: String::from("Instructions"),
            imports: self.zklean_imports(),
            contents,
        })
    }
}
