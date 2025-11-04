use jolt_core::zkvm::lookup_table::LookupTables;
use strum::IntoEnumIterator as _;

use crate::{
    modules::{AsModule, Module},
    util::ZkLeanReprField,
    DefaultMleAst,
};

/// Wrapper around a JoltInstruction
// TODO: Can we tie the XLEN to the JoltParameterSet somehow? Seem hard w/o const generic
// exprs...
#[derive(Debug, Clone)]
pub struct ZkLeanLookupTable<const XLEN: usize> {
    pub lookup_table: LookupTables<XLEN>,
}

impl<const XLEN: usize> From<LookupTables<XLEN>> for ZkLeanLookupTable<XLEN> {
    fn from(value: LookupTables<XLEN>) -> Self {
        Self {
            lookup_table: value,
        }
    }
}

/// This structure is merely here to gather all the information needed for displaying a MLE.
struct DisplayZkLean<F: ZkLeanReprField> {
    mle: F,
    name: String,
    num_variables: usize,
}

impl<F: ZkLeanReprField> std::fmt::Display for DisplayZkLean<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.mle.format_for_lean(f, &self.name, self.num_variables)
    }
}

impl<const XLEN: usize> ZkLeanLookupTable<XLEN> {
    pub fn name(&self) -> String {
        let name = <&'static str>::from(&self.lookup_table);
        let word_size = XLEN;

        format!("{name}_{word_size}_lookup_table")
    }

    pub fn evaluate_mle<F: ZkLeanReprField>(&self, reg_name: char) -> F {
        let num_variables = 2 * XLEN;
        let reg = F::register(reg_name, num_variables);

        self.lookup_table.evaluate_mle::<F, F>(&reg)
    }

    pub fn iter() -> impl Iterator<Item = Self> {
        LookupTables::<XLEN>::iter().map(Self::from)
    }

    /// Pretty print an instruction as a ZkLean `ComposedLookupTable`.
    pub fn zklean_pretty_print<F: ZkLeanReprField>(
        &self,
        f: &mut impl std::io::Write,
    ) -> std::io::Result<()> {
        let printable = DisplayZkLean {
            name: self.name(),
            num_variables: 2 * XLEN,
            mle: self.evaluate_mle::<F>('x'),
        };
        let _ = write!(f, "{printable}");
        Ok(())
    }
}

pub struct ZkLeanLookupTables<const XLEN: usize> {
    instructions: Vec<ZkLeanLookupTable<XLEN>>,
}

impl<const XLEN: usize> ZkLeanLookupTables<XLEN> {
    pub fn extract() -> Self {
        Self {
            instructions: ZkLeanLookupTable::iter().collect(),
        }
    }

    pub fn zklean_pretty_print(&self, f: &mut impl std::io::Write) -> std::io::Result<()> {
        for instruction in &self.instructions {
            instruction.zklean_pretty_print::<DefaultMleAst>(f)?;
        }
        Ok(())
    }

    pub fn zklean_imports(&self) -> Vec<String> {
        vec![String::from("ZkLean")]
    }
}

impl<const XLEN: usize> AsModule for ZkLeanLookupTables<XLEN> {
    fn as_module(&self) -> std::io::Result<Module> {
        let mut contents: Vec<u8> = vec![];
        self.zklean_pretty_print(&mut contents)?;

        Ok(Module {
            name: String::from("LookupTables"),
            imports: self.zklean_imports(),
            contents,
        })
    }
}

#[cfg(test)]
mod test {
    use std::collections::HashMap;

    use super::*;
    use crate::util::{arb_field_elem, Environment};

    use jolt_core::field::JoltField;

    use proptest::{collection::vec, prelude::*};

    type RefField = ark_bn254::Fr;
    type TestField = crate::mle_ast::DefaultMleAst;

    const XLEN: usize = 32;

    #[derive(Clone)]
    struct TestableLookupTable<const XLEN: usize> {
        reference: LookupTables<XLEN>,
        test: ZkLeanLookupTable<XLEN>,
    }

    impl<const XLEN: usize> std::fmt::Debug for TestableLookupTable<XLEN> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_fmt(format_args!("{}", self.test.name()))
        }
    }

    impl<const XLEN: usize> TestableLookupTable<XLEN> {
        fn iter() -> impl Iterator<Item = Self> {
            ZkLeanLookupTable::iter().map(|instr| Self {
                reference: instr.lookup_table,
                test: instr,
            })
        }

        fn evaluate_reference_mle<R: JoltField>(&self, inputs: &[R]) -> R {
            assert_eq!(inputs.len(), 2 * XLEN);

            self.reference.evaluate_mle(inputs)
        }

        fn evaluate_test_mle<R: JoltField, T: ZkLeanReprField>(&self, inputs: &[R]) -> R {
            assert_eq!(inputs.len(), 2 * XLEN);

            let ast: T = self.test.evaluate_mle('x');
            ast.evaluate(&Environment {
                let_bindings: &HashMap::new(),
                vars: inputs,
            })
        }
    }

    fn arb_instruction<const XLEN: usize>() -> impl Strategy<Value = TestableLookupTable<XLEN>> {
        let num_instrs = TestableLookupTable::<XLEN>::iter().count();

        (0..num_instrs).prop_map(|n| TestableLookupTable::iter().nth(n).unwrap())
    }

    fn arb_instruction_and_input<R: JoltField, const XLEN: usize>(
    ) -> impl Strategy<Value = (TestableLookupTable<XLEN>, Vec<R>)> {
        arb_instruction().prop_flat_map(|instr| {
            let input_len = 2 * XLEN;
            let inputs = vec(arb_field_elem::<R>(), input_len);

            (Just(instr), inputs)
        })
    }

    proptest! {
        #[test]
        fn evaluate_mle(
            (instr, inputs) in arb_instruction_and_input::<RefField, XLEN>(),
        ) {
            prop_assert_eq!(
                instr.evaluate_test_mle::<_, TestField>(&inputs),
                instr.evaluate_reference_mle(&inputs),
            );
        }
    }
}
