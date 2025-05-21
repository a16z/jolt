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

#[cfg(test)]
mod test {
    use super::*;
    use crate::util::arb_field_elem;

    use jolt_core::{field::JoltField, jolt::subtable::LassoSubtable};

    use proptest::{prelude::*, collection::vec};
    use strum::EnumCount as _;

    type RefField = ark_bn254::Fr;
    type TestField = crate::mle_ast::MleAst<4096>;
    type ParamSet = crate::constants::RV32IParameterSet;

    struct TestableSubtable<R: JoltField, T: ZkLeanReprField, J: JoltParameterSet> {
        reference: RV32ISubtables<R>,
        test: ZkLeanSubtable<T, J>,
    }

    impl<R: JoltField, T: ZkLeanReprField, J: JoltParameterSet> std::fmt::Debug for TestableSubtable<R, T, J> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_fmt(format_args!("{}", self.test.name()))
        }
    }

    impl<R: JoltField, T: ZkLeanReprField, J: JoltParameterSet> TestableSubtable<R, T, J> {
        fn iter() -> impl Iterator<Item = Self> {
            RV32ISubtables::iter()
                .zip(ZkLeanSubtable::iter())
                .map(|(reference, test)| Self { reference, test })
        }

        fn reference_evaluate_mle(&self, inputs: &[R]) -> R {
            assert_eq!(inputs.len(), J::LOG_M);

            self.reference.evaluate_mle(inputs)
        }

        fn test_evaluate_mle(&self, inputs: &[R]) -> R {
            assert_eq!(inputs.len(), J::LOG_M);

            let ast = self.test.evaluate_mle('x');
            ast.evaluate(inputs)
        }
    }

    fn arb_subtable<R: JoltField, T: ZkLeanReprField, J: JoltParameterSet>()
        -> impl Strategy<Value = TestableSubtable<R, T, J>>
    {
        (0..RV32ISubtables::<R>::COUNT)
            .prop_map(|n| TestableSubtable::iter().nth(n).unwrap())
    }

    proptest! {
        #[test]
        fn evaluate_mle(
            subtable in arb_subtable::<RefField, TestField, ParamSet>(),
            inputs in vec(arb_field_elem::<RefField>(), ParamSet::LOG_M),
        ) {
            // NOTE: Omitting this causes index OOB errors when converting from `uXX`
            crate::util::initialize_fields();

            prop_assert_eq!(
                subtable.test_evaluate_mle(&inputs),
                subtable.reference_evaluate_mle(&inputs),
            );
        }
    }
}
