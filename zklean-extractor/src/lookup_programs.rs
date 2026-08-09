use jolt_lookup_tables::{tables::and::AndTable, LookupMaterializer, LookupTableKind};

use crate::{
    lookup_graph::LookupGraph,
    lookups::ZkLeanLookupTable,
    materializer_ast::MaterializerAst,
    modules::{AsModule, Module},
    DefaultMleAst,
};

/// Extracted lookup semantics checked by the static Lean library.
pub struct ZkLeanLookupPrograms<const XLEN: usize> {
    table_name: String,
    mle: String,
    materializer: String,
}

impl<const XLEN: usize> ZkLeanLookupPrograms<XLEN> {
    pub fn extract() -> Result<Self, String> {
        let table = ZkLeanLookupTable::from(LookupTableKind::And(AndTable::<XLEN>));
        let num_inputs = 2 * XLEN;
        let mle: DefaultMleAst = table.evaluate_mle('x');
        let mle = LookupGraph::from_mle_ast(&mle, num_inputs)
            .map_err(|error| format!("failed to build AND lookup graph: {error}"))?
            .format_for_lean()
            .map_err(|error| format!("failed to format AND lookup graph: {error}"))?;
        let materializer = AndTable::<XLEN>.materialize(&mut MaterializerAst::new(num_inputs));

        Ok(Self {
            table_name: table.name(),
            mle,
            materializer: materializer.format_for_lean(num_inputs)?,
        })
    }

    pub fn zklean_pretty_print(&self, f: &mut impl std::io::Write) -> std::io::Result<()> {
        let table_name = &self.table_name;
        let graph_name = format!("{table_name}_graph");
        let materializer_name = format!("{table_name}_materializer");
        let materializer_well_formed_name = format!("{materializer_name}_wellFormed");
        let correspondence_name = format!("{table_name}_correspondence");
        let program_name = format!("{table_name}_program");
        let num_inputs = 2 * XLEN;

        writeln!(
            f,
            "/-- Shared algebraic graph extracted from the Rust AND evaluator. -/"
        )?;
        writeln!(
            f,
            "def {graph_name} : Jolt.LookupGraph.Graph {num_inputs} :="
        )?;
        writeln!(f, "  {}", self.mle)?;
        writeln!(f)?;
        writeln!(
            f,
            "/-- Lean checks that the extracted AND graph is structurally well formed. -/"
        )?;
        writeln!(
            f,
            "theorem {graph_name}_wellFormed : {graph_name}.WellFormed := by"
        )?;
        writeln!(f, "  set_option maxRecDepth 4096 in decide")?;
        writeln!(f)?;
        writeln!(
            f,
            "/-- Independent Boolean materializer extracted from the Rust AND instruction. -/"
        )?;
        writeln!(
            f,
            "def {materializer_name} : Jolt.LookupExpression.NatExpr {num_inputs} :="
        )?;
        writeln!(f, "  {}", self.materializer)?;
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
            "/-- The extracted field graph is exactly the arithmetic form of the materializer. -/"
        )?;
        writeln!(f, "theorem {correspondence_name} :")?;
        writeln!(
            f,
            "    {graph_name}.toExpr = {materializer_name}.arithmetize := by"
        )?;
        writeln!(
            f,
            "  prove_lookup_program_correspondence {graph_name} {materializer_name}"
        )?;
        writeln!(f)?;
        writeln!(
            f,
            "/-- The field evaluator and materializer extracted from the shared AND semantics. -/"
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
        writeln!(f, "    mleCorrespondence := {correspondence_name} }}")?;
        writeln!(f)?;
        writeln!(
            f,
            "/-- Evaluate the extracted AND lookup polynomial on a field vector. -/"
        )?;
        writeln!(
            f,
            "def {table_name} [Field f] (point : Vector f {num_inputs}) : f :="
        )?;
        writeln!(f, "  {program_name}.evalVector point")?;
        writeln!(f)?;
        writeln!(
            f,
            "/-- The extracted AND evaluator is the multilinear extension of its materializer. -/"
        )?;
        writeln!(f, "theorem {table_name}_isLookupTableMLE [Field f] :")?;
        writeln!(f, "    Jolt.LookupExpression.IsLookupTableMLE")?;
        writeln!(
            f,
            "      (fun point : Fin {num_inputs} → f => {table_name} (Vector.ofFn point))"
        )?;
        writeln!(f, "      {program_name}.materializer.eval :=")?;
        writeln!(f, "  {program_name}.isLookupTableMLE (by decide)")?;

        Ok(())
    }
}

impl<const XLEN: usize> AsModule for ZkLeanLookupPrograms<XLEN> {
    fn as_module(&self) -> std::io::Result<Module> {
        let mut contents = Vec::new();
        self.zklean_pretty_print(&mut contents)?;

        Ok(Module {
            name: String::from("LookupPrograms"),
            imports: vec![
                String::from("Jolt.LookupProgram"),
                String::from("Mathlib.Algebra.Field.Defs"),
            ],
            contents,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_and_as_data_checked_by_one_static_theorem() {
        let program = ZkLeanLookupPrograms::<64>::extract().unwrap();
        let module = program.as_module().unwrap();
        let lean = String::from_utf8(module.contents).unwrap();

        assert!(lean.contains("And_64_lookup_table_program"));
        assert!(lean.contains("And_64_lookup_table_graph"));
        assert!(lean.contains("{ nodeChunks := ["));
        assert!(lean.contains("mle := And_64_lookup_table_graph"));
        assert!(lean.contains("mleWellFormed := And_64_lookup_table_graph_wellFormed"));
        assert!(lean.contains("And_64_lookup_table_materializer"));
        assert!(
            lean.contains("materializerWellFormed := And_64_lookup_table_materializer_wellFormed")
        );
        assert!(lean.contains("mleCorrespondence := And_64_lookup_table_correspondence"));
        assert!(lean.contains("  (.ofBitsBE"));
        assert!(lean.contains(
            "prove_lookup_program_correspondence And_64_lookup_table_graph And_64_lookup_table_materializer"
        ));
        assert!(lean.contains(".isLookupTableMLE (by"));
        assert!(!lean.contains("simp"));
        assert!(!lean.contains("ring"));
        assert!(lean.contains("set_option maxHeartbeats 1000000 in"));
    }
}
