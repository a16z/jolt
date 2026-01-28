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
    lookup_table_ident: String,
}

impl<const XLEN: usize> ZkLeanLookupTableFlag<XLEN> {
    pub fn iter() -> impl Iterator<Item = Self> {
        let sumcheck_id = SumcheckId::InstructionReadRaf;

        LookupTables::<XLEN>::iter()
            .enumerate()
            .map(move |(i, lookup_table)| {
                let polynomial_id = PolynomialId::Virtual(VirtualPolynomial::LookupTableFlag(i));
                let lookup_table_ident = ZkLeanLookupTable::from(lookup_table).name();

                ZkLeanLookupTableFlag {
                    sumcheck_id,
                    polynomial_id,
                    lookup_table_ident,
                }
            })
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
//
// TODO It would be good to have a dedicated struct, rather than using (SumcheckId, PolynomialId).
pub struct ZkLeanLookupTableFlags<const XLEN: usize> {
    left_operand: (SumcheckId, PolynomialId),
    right_operand: (SumcheckId, PolynomialId),
    /// A Boolean flag variable, set as follows:
    /// - 0: MLE input is the interleaving of the left and right operands
    /// - 1: MLE input is the concatenation of the left and right operands
    interleaving_flag: (SumcheckId, PolynomialId),
    /// Boolean flags for each lookup table, along with the corresponding lookup-table identifiers
    lookup_table_flags: Vec<ZkLeanLookupTableFlag<XLEN>>,
    /// Variable to constrain equal to muxed lookups
    lookup_output: (SumcheckId, PolynomialId),
}

impl<const XLEN: usize> ZkLeanLookupTableFlags<XLEN> {
    pub fn extract() -> Self {
        let left_operand = (
            SumcheckId::SpartanOuter,
            PolynomialId::Virtual(VirtualPolynomial::LeftLookupOperand),
        );
        let right_operand = (
            SumcheckId::SpartanOuter,
            PolynomialId::Virtual(VirtualPolynomial::RightLookupOperand),
        );
        let interleaving_flag = (
            SumcheckId::InstructionReadRaf,
            PolynomialId::Virtual(VirtualPolynomial::InstructionRafFlag),
        );
        let lookup_table_flags = ZkLeanLookupTableFlag::<XLEN>::iter().collect();
        let lookup_output = (
            SumcheckId::SpartanOuter,
            PolynomialId::Virtual(VirtualPolynomial::LookupOutput),
        );

        Self {
            left_operand,
            right_operand,
            interleaving_flag,
            lookup_table_flags,
            lookup_output,
        }
    }

    pub fn zklean_pretty_print(
        &self,
        f: &mut impl std::io::Write,
        mut indent_level: usize,
    ) -> std::io::Result<()> {
        let vars_ident = "inputs";

        writeln!(
            f,
            "{}def mux_lookup_flags [ZKField f] ",
            indent(indent_level)
        )?;
        indent_level += 1;
        writeln!(f, "{}({vars_ident} : SumcheckVars f)", indent(indent_level))?;
        writeln!(f, "{}(interleaving : Interleaving)", indent(indent_level))?;
        writeln!(f, "{}(right left : ZKExpr f)", indent(indent_level))?;
        writeln!(f, "{}: ZKBuilder f (ZKExpr f) := do", indent(indent_level))?;
        writeln!(f, "{}ZKBuilder.mux", indent(indent_level))?;
        indent_level += 1;
        writeln!(f, "{}#[", indent(indent_level))?;
        indent_level += 1;
        for flag in self.lookup_table_flags.iter() {
            writeln!(f, "{}(", indent(indent_level))?;
            indent_level += 1;
            write!(f, "{}", indent(indent_level))?;
            pretty_print_opening_ref(f, vars_ident, flag.sumcheck_id, flag.polynomial_id)?;
            writeln!(f, ",")?;
            writeln!(
                f,
                "{}evalLookupTableMLE (LookupTable interleaving {}) left right",
                indent(indent_level),
                flag.lookup_table_ident
            )?;
            indent_level -= 1;
            writeln!(f, "{}),", indent(indent_level))?;
        }
        indent_level -= 1;
        writeln!(f, "{}]", indent(indent_level))?;
        indent_level -= 1;
        indent_level -= 1;

        writeln!(f)?;
        writeln!(
            f,
            "{}def lookup_step [ZKField f] ({vars_ident} : SumcheckVars f): ZKBuilder f PUnit := do",
            indent(indent_level),
        )?;
        indent_level += 1;
        write!(f, "{}let left := ", indent(indent_level))?;
        pretty_print_opening_ref(f, vars_ident, self.left_operand.0, self.left_operand.1)?;
        writeln!(f)?;
        write!(f, "{}let right := ", indent(indent_level))?;
        pretty_print_opening_ref(f, vars_ident, self.right_operand.0, self.right_operand.1)?;
        writeln!(f)?;
        writeln!(f, "{}let concated_eval <- mux_lookup_flags {vars_ident} Interleaving.Concatenated left right", indent(indent_level))?;
        writeln!(f, "{}let interleaved_eval <- mux_lookup_flags {vars_ident} Interleaving.Interleaved left right", indent(indent_level))?;
        writeln!(f, "{}let res <- ZkBuilder.mux #[", indent(indent_level))?;
        indent_level += 1;
        write!(f, "{}(", indent(indent_level))?;
        pretty_print_opening_ref(
            f,
            vars_ident,
            self.interleaving_flag.0,
            self.interleaving_flag.1,
        )?;
        writeln!(f, ", concatenated_eval),")?;
        write!(f, "{}(1 - ", indent(indent_level))?;
        pretty_print_opening_ref(
            f,
            vars_ident,
            self.interleaving_flag.0,
            self.interleaving_flag.1,
        )?;
        writeln!(f, ", interleaved_eval),")?;
        indent_level -= 1;
        writeln!(f, "{}]", indent(indent_level))?;
        write!(f, "{}ZKBuilder.constrainEq res ", indent(indent_level))?;
        pretty_print_opening_ref(f, vars_ident, self.lookup_output.0, self.lookup_output.1)?;

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
