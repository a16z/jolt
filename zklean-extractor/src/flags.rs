use jolt_core::{r1cs::inputs::JoltR1CSInputs};
use crate::instruction::NamedInstruction;

use crate::r1cs::input_to_field_name;

/// This represents a mapping between a [`JoltR1CSInputs`] variable and an instruction
pub struct ZkLeanInstructionFlags<const WORD_SIZE: usize> {
    r1cs_input: JoltR1CSInputs,
    instruction: NamedInstruction<WORD_SIZE>,
}

impl ZkLeanInstructionFlags<32> {
    /// Extract the [`JoltR1CSInputs::InstructionFlags`] variable for a given instruction.
    pub fn extract(instruction: NamedInstruction<32>) -> Self {
        let opcode = instruction.to_instruction_set();
        Self {
            r1cs_input: JoltR1CSInputs::InstructionFlags(opcode),
            instruction,
        }
    }

    pub fn zklean_pretty_print(&self, f: &mut impl std::io::Write) -> std::io::Result<()> {
        let r1cs_input = input_to_field_name(&self.r1cs_input);
        let composed_lookup_table = self.instruction.name();
        let _ = f.write(format!("({r1cs_input}, {composed_lookup_table})").as_bytes())?;
        Ok(())
    }
}

/// The R1CS-variable <-> instruction mappings for the entire instruction set.
pub struct ZkLeanLookupCases<const WORD_SIZE: usize> {
    instruction_flags: Vec<ZkLeanInstructionFlags<WORD_SIZE>>,
}

impl ZkLeanLookupCases<32> {
    /// Iterate over the instruction set and extract each R1CS input variable.
    pub fn extract() -> Self {
        Self {
            instruction_flags: NamedInstruction::<32>::enumerate()
                .into_iter()
                .map(ZkLeanInstructionFlags::extract)
                .collect(),
        }
    }

    pub fn zklean_pretty_print(&self, f: &mut impl std::io::Write) -> std::io::Result<()> {
        f.write(b"def lookup_step [JoltField f] (inputs: JoltR1CSInputs f): ZKBuilder f PUnit :=\n")?;
        f.write(b"  mux_lookup\n")?;
        // TODO(hamlinb) Extract these too?
        f.write(b"    (#v[inputs.ChunksQuery_0, inputs.ChunksQuery_1, inputs.ChunksQuery_2, inputs.ChunksQuery_3])\n")?;
        f.write(b"    (#[\n")?;
        for iflags in &self.instruction_flags {
            f.write(b"      ")?;
            iflags.zklean_pretty_print(f)?;
            f.write(b"\n")?;
        }
        f.write(b"    ])\n")?;
        Ok(())
    }
}
