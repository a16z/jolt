use jolt_core::{
    poly::opening_proof::{PolynomialId, SumcheckId},
    zkvm::{lookup_table::LookupTables, witness::VirtualPolynomial},
};
use strum::IntoEnumIterator as _;

use crate::{
    lookups::ZkLeanLookupTable,
    modules::{AsModule, Module},
    sumchecks::pretty_print_opening_ref,
    util::indent,
};

/// Each lookup-table corresponds to a flag, in the form of a polynomial opening. This struct
/// captures the correspondence between one table and its flag.
pub struct ZkLeanLookupTableFlag<const XLEN: usize> {
    sumcheck_id: SumcheckId,
    polynomial_id: PolynomialId,
    lookup_table: ZkLeanLookupTable<XLEN>,
}

impl<const XLEN: usize> ZkLeanLookupTableFlag<XLEN> {
    pub fn iter() -> impl Iterator<Item = Self> {
        let sumcheck_id = SumcheckId::InstructionReadRaf;

        LookupTables::<XLEN>::iter()
            .enumerate()
            .map(move |(i, lookup_table)| {
                let polynomial_id = PolynomialId::Virtual(VirtualPolynomial::LookupTableFlag(i));

                ZkLeanLookupTableFlag {
                    sumcheck_id,
                    polynomial_id,
                    lookup_table: lookup_table.into(),
                }
            })
    }

    pub fn zklean_pretty_print(
        &self,
        f: &mut impl std::io::Write,
        vars_ident: &str,
        indent_level: usize,
    ) -> std::io::Result<()> {
        let indent = indent(indent_level);
        let name = self.lookup_table.name();

        write!(f, "{indent}(")?;
        pretty_print_opening_ref(f, vars_ident, self.sumcheck_id, self.polynomial_id)?;
        write!(f, ", {name})")?;

        Ok(())
    }
}

/// This struct contains the list of input chunks, as well as the association between each
/// lookup table and its corresponding flag opening.
//
// NOTE: At the moment, this associates the lookup tables with their flags in the same way as the
// instruction_read_raf sumcheck: It enumerates all lookup tables and associates them with
// `LookupTableFlags(i)` where `i` is the ordinal index of the lookup table in enumerated order.
// This makes the assumption that the lookup tables and flags aren't associated in a different way
// in the future. A better way to extract this would be to extract the instruction_read_raf
// sumcheck itself.
pub struct ZkLeanLookupTableFlags<const XLEN: usize> {
    chunks: Vec<(SumcheckId, PolynomialId)>,
    flags: Vec<ZkLeanLookupTableFlag<XLEN>>,
}

impl<const XLEN: usize> ZkLeanLookupTableFlags<XLEN> {
    pub fn extract() -> Self {
        let chunks = vec![
            (
                SumcheckId::SpartanOuter,
                PolynomialId::Virtual(VirtualPolynomial::LeftLookupOperand),
            ),
            (
                SumcheckId::SpartanOuter,
                PolynomialId::Virtual(VirtualPolynomial::RightLookupOperand),
            ),
        ];
        let flags = ZkLeanLookupTableFlag::<XLEN>::iter().collect();

        Self { chunks, flags }
    }

    pub fn zklean_pretty_print(
        &self,
        f: &mut impl std::io::Write,
        mut indent_level: usize,
    ) -> std::io::Result<()> {
        let vars_ident = "inputs";
        writeln!(
            f,
            "{}def lookup_step [ZKField f] ({vars_ident}: SumcheckVars f): ZKBuilder f PUnit := do",
            indent(indent_level),
        )?;
        indent_level += 1;
        writeln!(f, "{}let res <- ZKBuilder.muxLookup", indent(indent_level))?;
        indent_level += 1;

        writeln!(f, "{}(#v[", indent(indent_level))?;
        indent_level += 1;
        for chunk in self.chunks.iter() {
            write!(f, "{}", indent(indent_level))?;
            pretty_print_opening_ref(f, vars_ident, chunk.0, chunk.1)?;
            writeln!(f, ",")?;
        }
        indent_level -= 1;
        writeln!(f, "{}])", indent(indent_level))?;

        writeln!(f, "{}(#v[", indent(indent_level))?;
        indent_level += 1;
        for flag in self.flags.iter() {
            flag.zklean_pretty_print(f, &vars_ident, indent_level)?;
            writeln!(f, ",")?;
        }
        indent_level -= 1;
        writeln!(f, "{}])", indent(indent_level))?;

        indent_level -= 1;
        write!(
            f,
            "{}ZKBuilder.constrainEq res ",
            indent(indent_level)
        )?;
        pretty_print_opening_ref(
            f,
            vars_ident,
            SumcheckId::SpartanOuter,
            PolynomialId::Virtual(VirtualPolynomial::LookupOutput),
        )?;

        Ok(())
    }
}

impl<const XLEN: usize> AsModule for ZkLeanLookupTableFlags<XLEN> {
    fn as_module(&self) -> std::io::Result<Module> {
        let mut contents: Vec<u8> = vec![];
        self.zklean_pretty_print(&mut contents, 0)?;

        Ok(Module {
            name: String::from("LookupTableFlags"),
            imports: vec![
                String::from("ZkLean"),
                String::from("Jolt.Sumchecks"),
                String::from("Jolt.LookupTables"),
            ],
            contents,
        })
    }
}
