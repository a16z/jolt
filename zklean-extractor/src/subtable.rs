use jolt_core::jolt::{subtable::LassoSubtable, vm::rv32i_vm::RV32ISubtables};
use strum::IntoEnumIterator as _;

use crate::{modules::{AsModule, Module}, util::{indent, ZkLeanReprField}};

/// Wrapper around a LassoSubtable
// TODO: Make generic over LassoSubtableSet
#[derive(Debug)]
pub struct ZkLeanSubtable<F: ZkLeanReprField, const LOG_M: usize>(RV32ISubtables<F>);

impl<F: ZkLeanReprField, const LOG_M: usize> From<RV32ISubtables<F>> for ZkLeanSubtable<F, LOG_M> {
    fn from(value: RV32ISubtables<F>) -> Self {
        Self(value)
    }
}

impl<F: ZkLeanReprField, const LOG_M: usize> From<&Box<dyn LassoSubtable<F>>> for ZkLeanSubtable<F, LOG_M> {
    fn from(value: &Box<dyn LassoSubtable<F>>) -> Self {
        Self(value.subtable_id().into())
    }
}

impl<F: ZkLeanReprField, const LOG_M: usize> ZkLeanSubtable<F, LOG_M> {
    pub fn name(&self) -> String {
        format!("{}_{LOG_M}", <&'static str>::from(&self.0))
    }

    pub fn evaluate_mle(&self, reg_name: char) -> F {
       let reg = F::register(reg_name, LOG_M);
       self.0.evaluate_mle(&reg)
    }

    pub fn iter() -> impl Iterator<Item = Self> {
        RV32ISubtables::iter().map(Self::from)
    }

    /// Pretty print a subtable as a ZkLean `Subtable`.
    pub fn zklean_pretty_print(
        &self,
        f: &mut impl std::io::Write,
        mut indent_level: usize,
    ) -> std::io::Result<()> {
        let name = self.name();
        let mle = self.evaluate_mle('x').as_computation();

        f.write_fmt(format_args!(
                "{}def {name} [Field f] : Subtable f {LOG_M} :=\n",
                indent(indent_level),
        ))?;
        indent_level += 1;
        f.write_fmt(format_args!(
                "{}subtableFromMLE (fun x => {mle})\n",
                indent(indent_level),
        ))?;

        Ok(())
    }
}

pub struct ZkLeanSubtables<F: ZkLeanReprField, const LOG_M: usize> {
    subtables: Vec<ZkLeanSubtable<F, LOG_M>>,
}

impl<F: ZkLeanReprField, const LOG_M: usize> ZkLeanSubtables<F, LOG_M> {
    pub fn extract() -> Self {
        Self {
            subtables: ZkLeanSubtable::<F, LOG_M>::iter().collect(),
        }
    }

    pub fn zklean_pretty_print(&self, f: &mut impl std::io::Write, indent_level: usize) -> std::io::Result<()> {
        for subtable in &self.subtables {
            subtable.zklean_pretty_print(f, indent_level)?;
        }
        Ok(())
    }

    pub fn zklean_imports(&self) -> Vec<String> {
        vec![
            String::from("ZkLean"),
        ]
    }
}

impl<F: ZkLeanReprField, const LOG_M: usize> AsModule for ZkLeanSubtables<F, LOG_M> {
    fn as_module(&self) -> std::io::Result<Module> {
        let mut contents: Vec<u8> = vec![];
        self.zklean_pretty_print(&mut contents, 0)?;

        Ok(Module {
            name: String::from("Subtables"),
            imports: self.zklean_imports(),
            contents,
        })
    }
}
