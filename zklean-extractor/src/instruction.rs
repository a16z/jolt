use jolt_core::jolt::instruction;
use zklean_extractor::declare_instructions_enum;

use crate::{modules::{AsModule, Module}, util::{indent, ZkLeanReprField}, MleAst};

declare_instructions_enum! {
    NamedInstruction,
    instruction::add::ADDInstruction,
    instruction::and::ANDInstruction,
    instruction::beq::BEQInstruction,
    instruction::bge::BGEInstruction,
    instruction::bgeu::BGEUInstruction,
    instruction::bne::BNEInstruction,
    instruction::mul::MULInstruction,
    instruction::mulhu::MULHUInstruction,
    instruction::mulu::MULUInstruction,
    instruction::or::ORInstruction,
    instruction::sll::SLLInstruction,
    instruction::slt::SLTInstruction,
    instruction::sltu::SLTUInstruction,
    instruction::sra::SRAInstruction,
    instruction::srl::SRLInstruction,
    instruction::sub::SUBInstruction,
    instruction::xor::XORInstruction,
    instruction::virtual_advice::ADVICEInstruction,
    // The const-generic corresponds to the allignment. This instruction is only instantiated for
    // alignments of 2 and 4 in the ISA.
    instruction::virtual_assert_aligned_memory_access::AssertAlignedMemoryAccessInstruction<2>,
    instruction::virtual_assert_aligned_memory_access::AssertAlignedMemoryAccessInstruction<4>,
    instruction::virtual_assert_lte::ASSERTLTEInstruction,
    instruction::virtual_assert_valid_div0::AssertValidDiv0Instruction,
    instruction::virtual_assert_valid_signed_remainder::AssertValidSignedRemainderInstruction,
    instruction::virtual_assert_valid_unsigned_remainder::AssertValidUnsignedRemainderInstruction,
    instruction::virtual_move::MOVEInstruction,
    instruction::virtual_movsign::MOVSIGNInstruction,
}

/// Helper function to print a list of subtables as a Lean Vector.
fn zklean_write_subtables(subtables: &Vec<(String, usize)>, log_m: usize) -> String {
    std::iter::once("#[ ".to_string())
        .chain(subtables.iter().map(|(s, i)| format!("({s}_{log_m}, {i})")).intersperse(", ".to_string()))
        .chain(std::iter::once(" ].toVector".to_string()))
        .fold(String::new(), |acc, s| format!("{acc}{s}"))
}

impl<const WORD_SIZE: usize, const C: usize, const LOG_M: usize> NamedInstruction<WORD_SIZE, C, LOG_M> {
    /// Pretty print an instruction as a ZkLean `ComposedLookupTable`.
    pub fn zklean_pretty_print<F: ZkLeanReprField>(
        &self,
        f: &mut impl std::io::Write,
        mut indent_level: usize,
    ) -> std::io::Result<()> {
        let name = self.name();
        let mle = self.combine_lookups::<F>('x').as_computation();
        let subtables = zklean_write_subtables(&self.subtables::<F>(), LOG_M);

        f.write_fmt(format_args!(
                "{}def {name}_{WORD_SIZE} [Field f] : ComposedLookupTable f {LOG_M} {C}\n",
                indent(indent_level),
        ))?;
        indent_level += 1;
        f.write_fmt(format_args!(
                "{}:= mkComposedLookupTable {subtables} (fun x => {mle})\n",
                indent(indent_level),
        ))?;

        Ok(())
    }
}

pub struct ZkLeanInstructions<const WORD_SIZE: usize, const C: usize, const LOG_M: usize> {
    instructions: Vec<NamedInstruction<WORD_SIZE, C, LOG_M>>,
}

impl<const WORD_SIZE: usize, const C: usize, const LOG_M: usize> ZkLeanInstructions<WORD_SIZE, C, LOG_M> {
    pub fn extract() -> Self {
        Self {
            instructions: NamedInstruction::<WORD_SIZE, C, LOG_M>::variants(),
        }
    }

    pub fn zklean_pretty_print(
        &self,
        f: &mut impl std::io::Write,
        indent_level: usize,
    ) -> std::io::Result<()> {
        for instruction in &self.instructions {
            instruction.zklean_pretty_print::<MleAst<2048>>(f, indent_level)?;
        }
        Ok(())
    }

    pub fn zklean_imports(&self) -> Vec<String> {
        vec![
            String::from("ZkLean"),
            String::from("Jolt.Subtables"),
        ]
    }
}

impl<const WORD_SIZE: usize, const C: usize, const LOG_M: usize> AsModule for ZkLeanInstructions<WORD_SIZE, C, LOG_M> {
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
