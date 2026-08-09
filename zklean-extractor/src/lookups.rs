use jolt_lookup_tables::{LookupEval, LookupTableKind};
use jolt_prover_legacy::zkvm::lookup_table::LookupTables;
use strum::IntoEnumIterator as _;

use crate::{
    lookup_graph::LookupGraph,
    modules::{AsModule, Module},
    util::ZkLeanReprField,
    DefaultMleAst,
};

/// Wrapper around a JoltInstructionRowData
// TODO: Can we tie the XLEN to the JoltParameterSet somehow? Seem hard w/o const generic
// exprs...
#[derive(Debug, Clone)]
pub struct ZkLeanLookupTable<const XLEN: usize> {
    pub lookup_table: LookupTableKind<XLEN>,
}

impl<const XLEN: usize> From<LookupTableKind<XLEN>> for ZkLeanLookupTable<XLEN> {
    fn from(value: LookupTableKind<XLEN>) -> Self {
        Self {
            lookup_table: value,
        }
    }
}

impl<const XLEN: usize> From<LookupTables<XLEN>> for ZkLeanLookupTable<XLEN> {
    fn from(value: LookupTables<XLEN>) -> Self {
        let index = LookupTables::enum_index(&value);
        Self::from(
            LookupTableKind::iter()
                .nth(index)
                .expect("legacy and modular lookup table catalogs must have matching ordinals"),
        )
    }
}

impl<const XLEN: usize> ZkLeanLookupTable<XLEN> {
    pub fn name(&self) -> String {
        let legacy_table = LookupTables::<XLEN>::iter()
            .nth(self.lookup_table.index())
            .expect("legacy and modular lookup table catalogs must have matching ordinals");
        let name = <&'static str>::from(&legacy_table);
        let word_size = XLEN;

        format!("{name}_{word_size}_lookup_table")
    }

    pub fn evaluate_mle<F: ZkLeanReprField + LookupEval>(&self, reg_name: char) -> F {
        let num_variables = 2 * XLEN;
        let reg = F::register(reg_name, num_variables);

        self.lookup_table.evaluate_mle::<F, F>(&reg)
    }

    pub fn iter() -> impl Iterator<Item = Self> {
        LookupTableKind::<XLEN>::iter().map(Self::from)
    }

    /// Emit one graph-backed Lean lookup evaluator.
    pub fn zklean_pretty_print(&self, f: &mut impl std::io::Write) -> std::io::Result<()> {
        let table_name = self.name();
        let graph_name = format!("{table_name}_graph");
        let num_inputs = 2 * XLEN;
        let ast = self.evaluate_mle::<DefaultMleAst>('x');
        let graph = LookupGraph::from_mle_ast(&ast, num_inputs).map_err(std::io::Error::other)?;
        let graph_data = graph.format_for_lean().map_err(std::io::Error::other)?;

        writeln!(
            f,
            "/-- Shared algebraic graph extracted from the Rust lookup evaluator. -/"
        )?;
        writeln!(
            f,
            "def {graph_name} : Jolt.LookupGraph.Graph {num_inputs} :="
        )?;
        writeln!(f, "  {graph_data}")?;
        writeln!(f)?;
        writeln!(
            f,
            "/-- Lean checks that every graph reference points backward and every input is in range. -/"
        )?;
        writeln!(
            f,
            "theorem {graph_name}_wellFormed : {graph_name}.WellFormed := by"
        )?;
        writeln!(f, "  set_option maxRecDepth 4096 in decide")?;
        writeln!(f)?;
        writeln!(
            f,
            "def {table_name} [Field f] (point : Vector f {num_inputs}) : f :="
        )?;
        writeln!(f, "  {graph_name}.evalVector {graph_name}_wellFormed point")?;
        writeln!(f)?;
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
            if matches!(instruction.lookup_table, LookupTableKind::And(_)) {
                continue;
            }
            instruction.zklean_pretty_print(f)?;
        }
        Ok(())
    }

    pub fn zklean_imports(&self) -> Vec<String> {
        vec![
            String::from("Jolt.LookupGraph"),
            String::from("Mathlib.Algebra.Field.Defs"),
        ]
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

    use jolt_prover_legacy::field::JoltField;

    use proptest::{collection::vec, prelude::*};

    type RefField = ark_bn254::Fr;
    type TestField = crate::mle_ast::DefaultMleAst;

    const XLEN: usize = 64;

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
                reference: LookupTables::<XLEN>::iter()
                    .nth(instr.lookup_table.index())
                    .expect("legacy and modular lookup table catalogs must have matching ordinals"),
                test: instr,
            })
        }

        fn evaluate_reference_mle<R: JoltField>(&self, inputs: &[R]) -> R {
            assert_eq!(inputs.len(), 2 * XLEN);

            self.reference.evaluate_mle(inputs)
        }

        fn evaluate_test_mle<R: JoltField, T: ZkLeanReprField + LookupEval>(
            &self,
            inputs: &[R],
        ) -> (R, R) {
            assert_eq!(inputs.len(), 2 * XLEN);

            let ast: T = self.test.evaluate_mle('x');
            let ast_value = ast.evaluate(&Environment {
                let_bindings: &HashMap::new(),
                vars: inputs,
            });
            let graph =
                LookupGraph::from_mle_ast(&self.test.evaluate_mle::<DefaultMleAst>('x'), 2 * XLEN)
                    .unwrap();

            (ast_value, graph.evaluate(inputs))
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
            let reference = instr.evaluate_reference_mle(&inputs);
            let (ast, graph) = instr.evaluate_test_mle::<_, TestField>(&inputs);
            prop_assert_eq!(
                ast,
                reference,
            );
            prop_assert_eq!(graph, reference);
        }
    }

    #[test]
    fn extracted_graphs_preserve_lookup_sharing() {
        let mut total_nodes = 0;
        let mut largest = (String::new(), 0);

        for table in ZkLeanLookupTable::<XLEN>::iter() {
            let ast = table.evaluate_mle::<DefaultMleAst>('x');
            let graph = LookupGraph::from_mle_ast(&ast, 2 * XLEN).unwrap();
            total_nodes += graph.len();
            if graph.len() > largest.1 {
                largest = (table.name(), graph.len());
            }
        }

        eprintln!("lookup graph nodes: total={total_nodes}, largest={largest:?}");
        assert!(total_nodes < 20_000);
        assert!(largest.1 < 1_000);
    }
}
