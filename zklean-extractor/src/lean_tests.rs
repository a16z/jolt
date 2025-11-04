use jolt_core::field::JoltField;
use rand_core::RngCore;

use crate::{
    lookups::ZkLeanLookupTable,
    modules::{AsModule, Module},
    util::indent,
};

// XXX Extract this? Or make it generic?
type TestField = ark_bn254::Fr;

#[derive(Debug, Clone)]
pub struct ZkLeanLookupTableTest<const XLEN: usize> {
    lookup_table_ident: String,
    output: TestField,
}

impl<const XLEN: usize> ZkLeanLookupTableTest<XLEN> {
    fn extract(lookup_table: &ZkLeanLookupTable<XLEN>, input: &[TestField]) -> Self {
        Self {
            lookup_table_ident: lookup_table.name(),
            output: lookup_table.lookup_table.evaluate_mle(input),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ZkLeanTests<const XLEN: usize> {
    input: Vec<TestField>,
    tests: Vec<ZkLeanLookupTableTest<XLEN>>,
}

impl<const XLEN: usize> ZkLeanTests<XLEN> {
    pub fn extract(rng: &mut impl RngCore) -> Self {
        let num_variables = 2 * XLEN;
        let input: Vec<_> = (0..num_variables).map(|_| TestField::random(rng)).collect();
        let tests: Vec<_> = ZkLeanLookupTable::iter()
            .map(|table| ZkLeanLookupTableTest::extract(&table, &input))
            .collect();

        Self { input, tests }
    }

    fn zklean_imports(&self) -> Vec<String> {
        vec![String::from("ArkLib"), String::from("Jolt.LookupTables")]
    }

    fn zklean_pretty_print(
        &self,
        f: &mut impl std::io::Write,
        indent_level: usize,
    ) -> std::io::Result<()> {
        // We need to set maxHeartbeats here because the Lean compiler times out when trying to run
        // the guards otherwise.
        let lean_max_heartbeats = 8_000_000;
        writeln!(f, "set_option maxHeartbeats {lean_max_heartbeats}")?;
        writeln!(f)?;

        writeln!(f, "abbrev TestField := BN254.ScalarField")?;
        writeln!(f)?;

        let num_variables = 2 * XLEN;
        write!(
            f,
            "{}def input : Vector TestField {num_variables} := #v[",
            indent(indent_level)
        )?;
        for (i, value) in self.input.iter().enumerate() {
            write!(
                f,
                " {value}{}",
                if i == self.input.len() - 1 { "" } else { "," }
            )?;
        }
        writeln!(f, " ]")?;
        writeln!(f)?;

        for test in &self.tests {
            let ident = &test.lookup_table_ident;
            let output = test.output;
            writeln!(f, "{}#guard {ident} input = {output}", indent(indent_level))?;
        }

        Ok(())
    }
}

impl<const XLEN: usize> AsModule for ZkLeanTests<XLEN> {
    fn as_module(&self) -> std::io::Result<Module> {
        let mut contents: Vec<u8> = vec![];
        self.zklean_pretty_print(&mut contents, 0)?;

        Ok(Module {
            name: String::from("Tests"),
            imports: self.zklean_imports(),
            contents,
        })
    }
}
