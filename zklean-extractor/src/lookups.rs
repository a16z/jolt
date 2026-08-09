use jolt_lookup_tables::{LookupMaterializer, LookupTableKind};
use jolt_prover_legacy::zkvm::lookup_table::LookupTables;
use strum::IntoEnumIterator as _;

use crate::{
    lookup_graph::LookupGraph,
    materializer_ast::{MaterializerAst, NatExpr},
    modules::{AsModule, Module},
    DefaultMleAst,
};

/// A modular lookup table together with its generated-name compatibility adapter.
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

    pub fn evaluate_mle(&self) -> Result<DefaultMleAst, String> {
        let inputs = (0..2 * XLEN)
            .map(|index| {
                u16::try_from(index)
                    .map(DefaultMleAst::from_var)
                    .map_err(|_| format!("lookup input index {index} does not fit in MleAst"))
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(self
            .lookup_table
            .evaluate_mle::<DefaultMleAst, DefaultMleAst>(&inputs))
    }

    pub fn iter() -> impl Iterator<Item = Self> {
        LookupTableKind::<XLEN>::iter().map(Self::from)
    }

    fn materializer(&self) -> Option<NatExpr> {
        let mut backend = MaterializerAst::new(2 * XLEN);
        match self.lookup_table {
            LookupTableKind::And(table) => Some(table.materialize(&mut backend)),
            _ => None,
        }
    }
}

/// One canonical extracted record for a modular lookup table.
struct ExtractedLookup {
    table_name: String,
    graph: LookupGraph,
    materializer: Option<NatExpr>,
}

impl ExtractedLookup {
    fn extract<const XLEN: usize>(table: ZkLeanLookupTable<XLEN>) -> Result<Self, String> {
        let table_name = table.name();
        let num_inputs = 2 * XLEN;
        let graph = LookupGraph::from_mle_ast(&table.evaluate_mle()?, num_inputs)
            .map_err(|error| format!("failed to build {table_name}: {error}"))?;

        Ok(Self {
            table_name,
            graph,
            materializer: table.materializer(),
        })
    }

    /// Emit one graph-backed Lean lookup evaluator and its optional certificate.
    fn zklean_pretty_print<const XLEN: usize>(
        &self,
        f: &mut impl std::io::Write,
    ) -> std::io::Result<()> {
        let table_name = &self.table_name;
        let graph_name = format!("{table_name}_graph");
        let num_inputs = 2 * XLEN;
        let graph_data = self
            .graph
            .format_for_lean()
            .map_err(std::io::Error::other)?;

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

        if let Some(materializer) = &self.materializer {
            self.zklean_pretty_print_certificate::<XLEN>(f, materializer)?;
        } else {
            writeln!(
                f,
                "def {table_name} [Field f] (point : Vector f {num_inputs}) : f :="
            )?;
            writeln!(f, "  {graph_name}.evalVector {graph_name}_wellFormed point")?;
            writeln!(f)?;
        }
        Ok(())
    }

    fn zklean_pretty_print_certificate<const XLEN: usize>(
        &self,
        f: &mut impl std::io::Write,
        materializer: &NatExpr,
    ) -> std::io::Result<()> {
        let table_name = &self.table_name;
        let graph_name = format!("{table_name}_graph");
        let materializer_name = format!("{table_name}_materializer");
        let materializer_well_formed_name = format!("{materializer_name}_wellFormed");
        let correspondence_name = format!("{table_name}_correspondence");
        let program_name = format!("{table_name}_program");
        let num_inputs = 2 * XLEN;
        let materializer = materializer
            .format_for_lean(num_inputs)
            .map_err(std::io::Error::other)?;

        writeln!(
            f,
            "/-- Boolean materializer extracted from the same Rust lookup table. -/"
        )?;
        writeln!(
            f,
            "def {materializer_name} : Jolt.LookupExpression.NatExpr {num_inputs} :="
        )?;
        writeln!(f, "  {materializer}")?;
        writeln!(f)?;

        writeln!(
            f,
            "/-- Lean checks that every materializer input is in range. -/"
        )?;
        writeln!(
            f,
            "theorem {materializer_well_formed_name} : {materializer_name}.WellFormed := by"
        )?;
        writeln!(f, "  set_option maxRecDepth 4096 in decide")?;
        writeln!(f)?;

        writeln!(f, "set_option maxRecDepth 4096 in")?;
        writeln!(f, "set_option maxHeartbeats 1000000 in")?;
        writeln!(
            f,
            "/-- The extracted graph and materializer define the same polynomial over every ring. -/"
        )?;
        writeln!(
            f,
            "theorem {correspondence_name} {{f : Type*}} [CommRing f]"
        )?;
        writeln!(f, "    (point : Fin {num_inputs} → f) :")?;
        writeln!(
            f,
            "    {graph_name}.toExpr.eval point = {materializer_name}.arithmetize.eval point := by"
        )?;
        writeln!(
            f,
            "  prove_lookup_program_correspondence {graph_name} {materializer_name}"
        )?;
        writeln!(f)?;

        writeln!(
            f,
            "/-- The field evaluator and materializer extracted from shared Rust semantics. -/"
        )?;
        writeln!(
            f,
            "def {program_name} : Jolt.LookupExpression.LookupProgram {num_inputs} :="
        )?;
        writeln!(f, "  {{ mle := {graph_name}")?;
        writeln!(f, "    mleWellFormed := {graph_name}_wellFormed")?;
        writeln!(f, "    materializer := {materializer_name}")?;
        writeln!(
            f,
            "    materializerWellFormed := {materializer_well_formed_name}"
        )?;
        writeln!(f, "  }}")?;
        writeln!(f)?;

        writeln!(
            f,
            "/-- Evaluate the extracted lookup polynomial on a field vector. -/"
        )?;
        writeln!(
            f,
            "def {table_name} [Field f] (point : Vector f {num_inputs}) : f :="
        )?;
        writeln!(f, "  {program_name}.evalVector point")?;
        writeln!(f)?;

        writeln!(
            f,
            "/-- The extracted evaluator is the multilinear extension of its materializer. -/"
        )?;
        writeln!(f, "theorem {table_name}_isLookupTableMLE [Field f] :")?;
        writeln!(f, "    Jolt.LookupExpression.IsLookupTableMLE")?;
        writeln!(
            f,
            "      (fun point : Fin {num_inputs} → f => {table_name} (Vector.ofFn point))"
        )?;
        writeln!(f, "      {program_name}.materializer.eval :=")?;
        writeln!(
            f,
            "  {program_name}.isLookupTableMLE (by decide) {correspondence_name}"
        )?;

        Ok(())
    }
}

pub struct ZkLeanLookupTables<const XLEN: usize> {
    tables: Vec<ExtractedLookup>,
}

impl<const XLEN: usize> ZkLeanLookupTables<XLEN> {
    pub fn extract() -> Result<Self, String> {
        let tables = ZkLeanLookupTable::<XLEN>::iter()
            .map(ExtractedLookup::extract)
            .collect::<Result<_, _>>()?;
        Ok(Self { tables })
    }

    pub fn zklean_pretty_print(&self, f: &mut impl std::io::Write) -> std::io::Result<()> {
        for table in &self.tables {
            table.zklean_pretty_print::<XLEN>(f)?;
        }
        Ok(())
    }

    pub fn zklean_imports(&self) -> Vec<String> {
        vec![
            String::from("Jolt.LookupProgram"),
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
    use crate::util::{arb_field_elem, Environment, ZkLeanReprField};

    use jolt_prover_legacy::field::JoltField;

    use proptest::{collection::vec, prelude::*};

    type RefField = ark_bn254::Fr;
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

        fn evaluate_test_mle<R: JoltField>(&self, inputs: &[R]) -> (R, R) {
            assert_eq!(inputs.len(), 2 * XLEN);

            let ast = self.test.evaluate_mle().unwrap();
            let ast_value = ast.evaluate(&Environment {
                let_bindings: &HashMap::new(),
                vars: inputs,
            });
            let graph = LookupGraph::from_mle_ast(&ast, 2 * XLEN).unwrap();

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
            let (ast, graph) = instr.evaluate_test_mle(&inputs);
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
            let ast = table.evaluate_mle().unwrap();
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

    #[test]
    fn emits_all_lookups_through_one_canonical_module() {
        let lookups = ZkLeanLookupTables::<XLEN>::extract().unwrap();
        let module = lookups.as_module().unwrap();
        let lean = String::from_utf8(module.contents).unwrap();

        assert_eq!(module.name, "LookupTables");
        assert!(lean.contains("And_64_lookup_table_program"));
        assert!(lean.contains("And_64_lookup_table_graph"));
        assert!(lean.contains("mle := And_64_lookup_table_graph"));
        assert!(lean.contains("And_64_lookup_table_materializer"));
        assert!(lean.contains("And_64_lookup_table_correspondence"));
        assert!(lean.contains("prove_lookup_program_correspondence"));
        assert!(lean.contains(".isLookupTableMLE (by decide)"));
        assert!(lean.contains("Xor_64_lookup_table_graph"));
    }
}
