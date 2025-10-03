use jolt_core::zkvm::lookup_table::LookupTables;
use strum::IntoEnumIterator as _;

use crate::{
    modules::{AsModule, Module},
    util::{indent, ZkLeanReprField},
    MleAst,
};

/// Wrapper around a JoltInstruction
// TODO: Can we tie the WORD_SIZE to the JoltParameterSet somehow? Seem hard w/o const generic
// exprs...
#[derive(Debug, Clone)]
pub struct ZkLeanLookupTable<const WORD_SIZE: usize> {
    lookup_table: LookupTables<WORD_SIZE>,
}

impl<const WORD_SIZE: usize> From<LookupTables<WORD_SIZE>> for ZkLeanLookupTable<WORD_SIZE> {
    fn from(value: LookupTables<WORD_SIZE>) -> Self {
        Self {
            lookup_table: value,
        }
    }
}

impl<const WORD_SIZE: usize> ZkLeanLookupTable<WORD_SIZE> {
    pub fn name(&self) -> String {
        let name = <&'static str>::from(&self.lookup_table);
        let word_size = WORD_SIZE;

        format!("{name}_{word_size}")
    }

    pub fn evaluate_mle<F: ZkLeanReprField>(&self, reg_name: char) -> F {
        let num_variables = 2 * WORD_SIZE;
        let reg = F::register(reg_name, num_variables);

        self.lookup_table.evaluate_mle::<F>(&reg)
    }

    pub fn iter() -> impl Iterator<Item = Self> {
        LookupTables::<WORD_SIZE>::iter().filter_map(|lt| match lt {
            // XXX Temporarily disabled. Too many nodes.
            // See https://gitlab-ext.galois.com/jb4/jolt-fork/-/issues/14
            LookupTables::SignedGreaterThanEqual(_) => None,
            LookupTables::UnsignedGreaterThanEqual(_) => None,
            LookupTables::SignedLessThan(_) => None,
            LookupTables::UnsignedLessThan(_) => None,
            LookupTables::LessThanEqual(_) => None,
            LookupTables::ValidSignedRemainder(_) => None,
            LookupTables::ValidUnsignedRemainder(_) => None,
            LookupTables::Equal(_) => None,
            LookupTables::NotEqual(_) => None,
            _ => Some(lt.into()),
        })
    }

    /// Pretty print an instruction as a ZkLean `ComposedLookupTable`.
    pub fn zklean_pretty_print<F: ZkLeanReprField>(
        &self,
        f: &mut impl std::io::Write,
        mut indent_level: usize,
    ) -> std::io::Result<()> {
        let name = self.name();
        let num_variables = 2 * WORD_SIZE;
        let mle = self.evaluate_mle::<F>('x').as_computation();

        f.write_fmt(format_args!(
            "{}def {name} [Field f] : LookupTable f {num_variables} :=\n",
            indent(indent_level),
        ))?;
        indent_level += 1;
        f.write_fmt(format_args!(
            "{}lookupTableFromMLE (fun x => {mle})\n",
            indent(indent_level),
        ))?;

        Ok(())
    }
}

pub struct ZkLeanLookupTables<const WORD_SIZE: usize> {
    instructions: Vec<ZkLeanLookupTable<WORD_SIZE>>,
}

impl<const WORD_SIZE: usize> ZkLeanLookupTables<WORD_SIZE> {
    pub fn extract() -> Self {
        Self {
            instructions: ZkLeanLookupTable::iter().collect(),
        }
    }

    pub fn zklean_pretty_print(
        &self,
        f: &mut impl std::io::Write,
        indent_level: usize,
    ) -> std::io::Result<()> {
        for instruction in &self.instructions {
            instruction.zklean_pretty_print::<MleAst<5400>>(f, indent_level)?;
        }
        Ok(())
    }

    pub fn zklean_imports(&self) -> Vec<String> {
        vec![String::from("ZkLean")]
    }
}

impl<const WORD_SIZE: usize> AsModule for ZkLeanLookupTables<WORD_SIZE> {
    fn as_module(&self) -> std::io::Result<Module> {
        let mut contents: Vec<u8> = vec![];
        self.zklean_pretty_print(&mut contents, 0)?;

        Ok(Module {
            name: String::from("LookupTables"),
            imports: self.zklean_imports(),
            contents,
        })
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::util::arb_field_elem;

    use jolt_core::field::JoltField;

    use proptest::{collection::vec, prelude::*};

    type RefField = ark_bn254::Fr;
    type TestField = crate::mle_ast::MleAst<5400>;

    const WORD_SIZE: usize = 32;

    #[derive(Clone)]
    struct TestableLookupTable<const WORD_SIZE: usize> {
        reference: LookupTables<WORD_SIZE>,
        test: ZkLeanLookupTable<WORD_SIZE>,
    }

    impl<const WORD_SIZE: usize> std::fmt::Debug for TestableLookupTable<WORD_SIZE> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_fmt(format_args!("{}", self.test.name()))
        }
    }

    impl<const WORD_SIZE: usize> TestableLookupTable<WORD_SIZE> {
        fn iter() -> impl Iterator<Item = Self> {
            ZkLeanLookupTable::iter().map(|instr| Self {
                reference: instr.lookup_table.clone(),
                test: instr,
            })
        }

        fn evaluate_reference_mle<R: JoltField>(&self, inputs: &[R]) -> R {
            assert_eq!(inputs.len(), 2 * WORD_SIZE);

            self.reference.evaluate_mle(inputs)
        }

        fn evaluate_test_mle<R: JoltField, T: ZkLeanReprField>(&self, inputs: &[R]) -> R {
            assert_eq!(inputs.len(), 2 * WORD_SIZE);

            let ast: T = self.test.evaluate_mle('x');
            ast.evaluate(inputs)
        }
    }

    fn arb_instruction<const WORD_SIZE: usize>(
    ) -> impl Strategy<Value = TestableLookupTable<WORD_SIZE>> {
        let num_instrs = TestableLookupTable::<WORD_SIZE>::iter().count();

        (0..num_instrs).prop_map(|n| TestableLookupTable::iter().nth(n).unwrap())
    }

    fn arb_instruction_and_input<R: JoltField, const WORD_SIZE: usize>(
    ) -> impl Strategy<Value = (TestableLookupTable<WORD_SIZE>, Vec<R>)> {
        arb_instruction().prop_flat_map(|instr| {
            let input_len = 2 * WORD_SIZE;
            let inputs = vec(arb_field_elem::<R>(), input_len);

            (Just(instr), inputs)
        })
    }

    proptest! {
        #[test]
        fn evaluate_mle(
            (instr, inputs) in arb_instruction_and_input::<RefField, WORD_SIZE>(),
        ) {
            prop_assert_eq!(
                instr.evaluate_test_mle::<_, TestField>(&inputs),
                instr.evaluate_reference_mle(&inputs),
            );
        }
    }
}
