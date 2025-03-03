use jolt_core::jolt::instruction;
use zklean_extractor::declare_instructions_enum;

use crate::util::ZkLeanReprField;

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
    instruction::or::ORInstruction,
    instruction::sll::SLLInstruction,
    instruction::slt::SLTInstruction,
    instruction::sltu::SLTUInstruction,
    instruction::sra::SRAInstruction,
    instruction::srl::SRLInstruction,
    instruction::sub::SUBInstruction,
    instruction::xor::XORInstruction,
}

/// Helper function to print a list of subtables as a Lean Vector.
fn zklean_write_subtables(subtables: &Vec<(String, usize)>, log_m: usize) -> String {
    std::iter::once("#[ ".to_string())
        .chain(subtables.iter().map(|(s, i)| format!("({s}_{log_m}, {i})")).intersperse(", ".to_string()))
        .chain(std::iter::once(" ].toVector".to_string()))
        .fold(String::new(), |acc, s| format!("{acc}{s}"))
}

impl<const WORD_SIZE: usize> NamedInstruction<WORD_SIZE> {
    /// Pretty print an instruction as a ZkLean `ComposedLookupTable`.
    pub fn zklean_pretty_print<F: ZkLeanReprField>(
        &self,
        f: &mut impl std::io::Write,
        c: usize,
        log_m: usize
    ) -> std::io::Result<()> {
        let m = 1 << log_m;
        let name = self.name();
        let mle = self.combine_lookups::<F>('x', c, m).as_computation();
        let subtables = zklean_write_subtables(&self.subtables::<F>(c, m), log_m);

        f.write_fmt(format_args!("def {name}_{WORD_SIZE} [Field f] : ComposedLookupTable f {log_m} {c}"))?;
        f.write_fmt(format_args!(" := mkComposedLookupTable {subtables} (fun x => {mle})\n"))?;

        Ok(())
    }
}
