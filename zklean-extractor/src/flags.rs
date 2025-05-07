use crate::instruction::NamedInstruction;
use crate::modules::{Module, AsModule};
use jolt_core::r1cs::inputs::JoltR1CSInputs;

use crate::{r1cs::input_to_field_name, util::indent};

/// This represents a mapping between a [`JoltR1CSInputs`] variable and an instruction
pub struct ZkLeanInstructionFlags<const WORD_SIZE: usize, const CHUNKS: usize, const REG_SIZE: usize> {
    r1cs_input: JoltR1CSInputs,
    instruction: NamedInstruction<WORD_SIZE, CHUNKS, REG_SIZE>,
}

impl ZkLeanInstructionFlags<32, 4, 16> {
    /// Extract the [`JoltR1CSInputs::InstructionFlags`] variable for a given instruction.
    pub fn from_instruction(instruction: NamedInstruction<32, 4, 16>) -> Self {
        let opcode = instruction.to_instruction_set();
        Self {
            r1cs_input: JoltR1CSInputs::InstructionFlags(opcode),
            instruction,
        }
    }

    pub fn to_string(
        &self,
        input_var: &str,
    ) -> String {
        let r1cs_input = input_to_field_name(&self.r1cs_input);
        let instruction_name = self.instruction.name();
        format!("({input_var}.{r1cs_input}, {instruction_name})")
    }
}

/// The R1CS-variable <-> instruction mappings for the entire instruction set.
pub struct ZkLeanLookupCases<const WORD_SIZE: usize, const CHUNKS: usize, const REG_SIZE: usize> {
    instruction_flags: Vec<ZkLeanInstructionFlags<WORD_SIZE, CHUNKS, REG_SIZE>>,
}

impl ZkLeanLookupCases<32, 4, 16> {
    /// Iterate over the instruction set and extract each R1CS input variable.
    pub fn extract() -> Self {
        Self {
            instruction_flags: NamedInstruction::<32, 4, 16>::iter()
                .map(ZkLeanInstructionFlags::from_instruction)
                .collect(),
        }
    }

    pub fn zklean_pretty_print(
        &self,
        f: &mut impl std::io::Write,
        mut indent_level: usize,
    ) -> std::io::Result<()> {
        let input_var = String::from("inputs");

        f.write_fmt(format_args!(
                "{}def lookup_step [JoltField f] ({input_var}: JoltR1CSInputs f): ZKBuilder f PUnit := do\n",
                indent(indent_level),
        ))?;
        indent_level += 1;
        f.write_fmt(format_args!("{}let res <- mux_lookup\n", indent(indent_level)))?;
        // TODO(hamlinb) Extract these too?
        indent_level += 1;
        f.write_fmt(format_args!(
                "{}(#v[inputs.ChunksQuery_0, inputs.ChunksQuery_1, inputs.ChunksQuery_2, inputs.ChunksQuery_3])\n",
                indent(indent_level),
        ))?;
        f.write_fmt(format_args!(
                "{}(#[\n",
                indent(indent_level),
        ))?;
        indent_level += 1;
        for (i, iflags) in self.instruction_flags.iter().enumerate() {
            f.write_fmt(format_args!(
                    "{}{}{}",
                    indent(indent_level),
                    iflags.to_string(&input_var),
                    if i < self.instruction_flags.len() - 1 { ",\n" } else { "\n" },
            ))?;
        }
        indent_level -= 1;
        f.write_fmt(format_args!(
                "{}])\n",
                indent(indent_level),
        ))?;
        indent_level -= 1;
        f.write_fmt(format_args!(
                "{}constrainEq res inputs.LookupOutput\n",
                indent(indent_level),
        ))?;
        Ok(())
    }

    pub fn zklean_imports(&self) -> Vec<String> {
        vec![
            String::from("ZkLean"),
            String::from("Jolt.Subtables"),
            String::from("Jolt.Instructions"),
            String::from("Jolt.R1CS"),
        ]
    }
}

impl AsModule for ZkLeanLookupCases<32, 4, 16> {
    fn as_module(&self) -> std::io::Result<Module> {
        let mut contents: Vec<u8> = vec![];
        self.zklean_pretty_print(&mut contents, 0)?;

        Ok(Module {
            name: String::from("InstructionFlags"),
            imports: self.zklean_imports(),
            contents,
        })
    }
}
