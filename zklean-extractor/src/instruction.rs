use jolt_core::jolt::{instruction::JoltInstruction, vm::rv32i_vm::RV32I};
use strum::IntoEnumIterator as _;

use crate::{modules::{AsModule, Module}, subtable::ZkLeanSubtable, util::{indent, ZkLeanReprField}, MleAst};

/// Wrapper around a JoltInstruction
// TODO: Make this generic over the instruction set
#[derive(Debug)]
pub struct ZkLeanInstruction<const WORD_SIZE: usize, const C: usize, const LOG_M: usize>(RV32I);

impl<const WORD_SIZE: usize, const C: usize, const LOG_M: usize> From<RV32I> for ZkLeanInstruction<WORD_SIZE, C, LOG_M> {
    fn from(value: RV32I) -> Self {
        Self(value)
    }
}

impl<const WORD_SIZE: usize, const C: usize, const LOG_M: usize> ZkLeanInstruction<WORD_SIZE, C, LOG_M> {
    pub fn name(&self) -> String {
        format!("{}_{WORD_SIZE}_{C}_{LOG_M}", <&'static str>::from(&self.0))
    }

    fn combine_lookups<F: ZkLeanReprField>(&self, reg_name: char) -> F {
        // We need one wire for each subtable evaluation
        let reg_size = self.subtables::<F>().count();
        let reg = F::register(reg_name, reg_size);
        self.0.combine_lookups(&reg, C, 1 << LOG_M)
    }

    fn subtables<F: ZkLeanReprField>(&self) -> impl Iterator<Item = (ZkLeanSubtable<F, LOG_M>, usize)> {
        self.0
            .subtables(C, 1 << LOG_M)
            .into_iter()
            .flat_map(|(subtable, ixs)|
                ixs.iter()
                    .map(|ix| (ZkLeanSubtable::<F, LOG_M>::from(&subtable), ix))
                    .collect::<Vec<_>>()
            )
    }

    pub fn iter() -> impl Iterator<Item = Self> {
        RV32I::iter().map(Self::from)
    }

    pub fn to_instruction_set(&self) -> RV32I {
        self.0
    }

    /// Pretty print an instruction as a ZkLean `ComposedLookupTable`.
    pub fn zklean_pretty_print<F: ZkLeanReprField>(
        &self,
        f: &mut impl std::io::Write,
        mut indent_level: usize,
    ) -> std::io::Result<()> {
        let name = self.name();
        let mle = self.combine_lookups::<F>('x').as_computation();
        let subtables = std::iter::once("#[ ".to_string())
            .chain(self.subtables::<F>().map(|(subtable, ix)|
                format!("({}, {ix})", subtable.name()))
                    .intersperse(", ".to_string())
            )
            .chain(std::iter::once(" ].toVector".to_string()))
            .fold(String::new(), |acc, s| format!("{acc}{s}"));

        f.write_fmt(format_args!(
                "{}def {name} [Field f] : ComposedLookupTable f {LOG_M} {C}\n",
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
    instructions: Vec<ZkLeanInstruction<WORD_SIZE, C, LOG_M>>,
}

impl<const WORD_SIZE: usize, const C: usize, const LOG_M: usize> ZkLeanInstructions<WORD_SIZE, C, LOG_M> {
    pub fn extract() -> Self {
        Self {
            instructions: ZkLeanInstruction::<WORD_SIZE, C, LOG_M>::iter().collect(),
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
