use jolt_core::jolt::{subtable::LassoSubtable, vm::rv32i_vm::RV32ISubtables};
use strum::IntoEnumIterator as _;

use crate::{constants::JoltParameterSet, modules::{AsModule, Module}, util::{indent, ZkLeanReprField}};

/// Wrapper around a LassoSubtable
// TODO: Make generic over LassoSubtableSet
#[derive(Debug)]
pub struct ZkLeanSubtable<F: ZkLeanReprField, J> {
    subtables: RV32ISubtables<F>,
    phantom: std::marker::PhantomData<J>,
}

impl<F: ZkLeanReprField, J> From<RV32ISubtables<F>> for ZkLeanSubtable<F, J> {
    fn from(value: RV32ISubtables<F>) -> Self {
        Self {
            subtables: value,
            phantom: std::marker::PhantomData,
        }
    }
}

impl<F: ZkLeanReprField, J> From<&Box<dyn LassoSubtable<F>>> for ZkLeanSubtable<F, J> {
    fn from(value: &Box<dyn LassoSubtable<F>>) -> Self {
        Self {
            subtables: value.subtable_id().into(),
            phantom: std::marker::PhantomData,
        }
    }
}

impl<F: ZkLeanReprField, J: JoltParameterSet> ZkLeanSubtable<F, J> {
    pub fn name(&self) -> String {
        let name = <&'static str>::from(&self.subtables);
        let log_m = J::LOG_M;

        format!("{name}_{log_m}")
    }

    pub fn evaluate_mle(&self, reg_name: char) -> F {
       let reg = F::register(reg_name, J::LOG_M);
       self.subtables.evaluate_mle(&reg)
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
        let log_m = J::LOG_M;
        let mle = self.evaluate_mle('x').as_computation();

        f.write_fmt(format_args!(
                "{}def {name} [Field f] : Subtable f {log_m} :=\n",
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

pub struct ZkLeanSubtables<F: ZkLeanReprField, J> {
    subtables: Vec<ZkLeanSubtable<F, J>>,
    phantom: std::marker::PhantomData<J>,
}

impl<F: ZkLeanReprField, J: JoltParameterSet> ZkLeanSubtables<F, J> {
    pub fn extract() -> Self {
        Self {
            subtables: ZkLeanSubtable::<F, J>::iter().collect(),
            phantom: std::marker::PhantomData,
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

impl<F: ZkLeanReprField, J: JoltParameterSet> AsModule for ZkLeanSubtables<F, J> {
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
