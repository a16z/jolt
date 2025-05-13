use jolt_core::jolt::{instruction::JoltInstruction, vm::rv32i_vm::RV32I};
use strum::IntoEnumIterator as _;

use crate::{constants::JoltParameterSet, modules::{AsModule, Module}, subtable::ZkLeanSubtable, util::{indent, ZkLeanReprField}, MleAst};

/// Wrapper around a JoltInstruction
// TODO: Make this generic over the instruction set
#[derive(Debug)]
pub struct ZkLeanInstruction<J> {
    instruction: RV32I,
    phantom: std::marker::PhantomData<J>,
}

impl<J> From<RV32I> for ZkLeanInstruction<J> {
    fn from(value: RV32I) -> Self {
        Self {
            instruction: value,
            phantom: std::marker::PhantomData,
        }
    }
}

impl<J: JoltParameterSet> ZkLeanInstruction<J> {
    pub fn name(&self) -> String {
        let name = <&'static str>::from(&self.instruction);
        let word_size = J::WORD_SIZE;
        let c = J::C;
        let log_m = J::LOG_M;

        format!("{name}_{word_size}_{c}_{log_m}")
    }

    fn combine_lookups<F: ZkLeanReprField>(&self, reg_name: char) -> F {
        // We need one wire for each subtable evaluation
        let reg_size = self.subtables::<F>().count();
        let reg = F::register(reg_name, reg_size);
        self.instruction.combine_lookups(&reg, J::C, 1 << J::LOG_M)
    }

    fn subtables<F: ZkLeanReprField>(&self) -> impl Iterator<Item = (ZkLeanSubtable<F, J>, usize)> {
        self.instruction
            .subtables(J::C, 1 << J::LOG_M)
            .into_iter()
            .flat_map(|(subtable, ixs)|
                ixs.iter()
                    .map(|ix| (ZkLeanSubtable::<F, J>::from(&subtable), ix))
                    .collect::<Vec<_>>()
            )
    }

    pub fn iter() -> impl Iterator<Item = Self> {
        RV32I::iter().map(Self::from)
    }

    pub fn to_instruction_set(&self) -> RV32I {
        self.instruction
    }

    /// Pretty print an instruction as a ZkLean `ComposedLookupTable`.
    pub fn zklean_pretty_print<F: ZkLeanReprField>(
        &self,
        f: &mut impl std::io::Write,
        mut indent_level: usize,
    ) -> std::io::Result<()> {
        let name = self.name();
        let log_m = J::LOG_M;
        let c = J::C;
        let mle = self.combine_lookups::<F>('x').as_computation();
        let subtables = std::iter::once("#[ ".to_string())
            .chain(self.subtables::<F>().map(|(subtable, ix)|
                format!("({}, {ix})", subtable.name()))
                    .intersperse(", ".to_string())
            )
            .chain(std::iter::once(" ].toVector".to_string()))
            .fold(String::new(), |acc, s| format!("{acc}{s}"));

        f.write_fmt(format_args!(
                "{}def {name} [Field f] : ComposedLookupTable f {log_m} {c}\n",
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

pub struct ZkLeanInstructions<J> {
    instructions: Vec<ZkLeanInstruction<J>>,
    phantom: std::marker::PhantomData<J>,
}

impl<J: JoltParameterSet> ZkLeanInstructions<J> {
    pub fn extract() -> Self {
        Self {
            instructions: ZkLeanInstruction::<J>::iter().collect(),
            phantom: std::marker::PhantomData,
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
