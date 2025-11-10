use jolt_core::zkvm::{
    instruction::{CircuitFlags, Flags as _, InstructionLookup as _, InterleavedBitsMarker as _},
    r1cs::inputs::JoltR1CSInputs,
};
use strum::IntoEnumIterator as _;
use tracer::instruction::Instruction;

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
    fn instruction_interleaving(instr: &Instruction) -> Self {
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
    instruction: tracer::instruction::Instruction,
    interleaving: OperandInterleaving,
    phantom: std::marker::PhantomData<J>,
}

impl<J> From<Instruction> for ZkLeanInstruction<J> {
    fn from(value: Instruction) -> Self {
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
        let word_size = J::XLEN;

        format!("{name}_{word_size}")
    }

    pub fn iter() -> impl Iterator<Item = Self> {
        Instruction::iter().filter_map(|instr| {
            // Needed to call `Instruction::inline_sequence`
            let allocator = tracer::utils::virtual_registers::VirtualRegisterAllocator::new();
            let xlen = match J::XLEN {
                32 => tracer::emulator::cpu::Xlen::Bit32,
                64 => tracer::emulator::cpu::Xlen::Bit64,
                _ => panic!("Unsupported bit width"),
            };

            match instr {
                Instruction::NoOp | Instruction::UNIMPL | Instruction::INLINE(_) => None,
                // NOTE: The functions we would like to call on each instruction (`lookup_table`
                // and `circuit_flags`) panic on inline sequences. We check to see if
                // `inline_sequence` returns a non-empty vector in order to filter out inline
                // sequences here.
                _ if !instr.inline_sequence(&allocator, xlen).is_empty() => None,
                _ => Some(Self::from(instr)),
            }
        })
    }

    /// Pretty print an instruction as a ZkLean `ComposedLookupTable`.
    pub fn zklean_pretty_print<F: ZkLeanReprField>(
        &self,
        f: &mut impl std::io::Write,
        mut indent_level: usize,
    ) -> std::io::Result<()> {
        let name = self.name();
        let num_variables = 2 * J::XLEN;
        let interleaving = self.interleaving;
        let lookup_table = match self
            .instruction
            .lookup_table()
            .map(|t| ZkLeanLookupTable::from(t).name())
        {
            None => String::from("sorry /-No lookup table for this instruction-/"),
            Some(t) => format!("{t} : Vector f {num_variables} -> f"),
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
            "{}def {name} [Field f] : LookupTableMLE f {num_variables} :=\n",
            indent(indent_level),
        ))?;
        indent_level += 1;
        f.write_fmt(format_args!(
            "{}-- Circuit flags: {circuit_flags}\n",
            indent(indent_level),
        ))?;
        f.write_fmt(format_args!(
            "{}LookupTableMLE.mk Interleaving.{interleaving} ({lookup_table})\n",
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
