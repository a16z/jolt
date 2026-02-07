use jolt_core::{
    poly::opening_proof::SumcheckId,
    zkvm::{lookup_table::LookupTables, witness::VirtualPolynomial},
};
use strum::IntoEnumIterator as _;

use crate::{
    lookups::ZkLeanLookupTable,
    modules::{AsModule, Module},
    sumchecks::ZkLeanVarRef,
    util::indent,
};

/// Each lookup-table corresponds to a flag, in the form of a polynomial opening. This struct
/// captures the correspondence between one table and its flag.
pub struct ZkLeanLookupTableFlag<const XLEN: usize> {
    var: ZkLeanVarRef,
    lookup_table_ident: String,
}

impl<const XLEN: usize> ZkLeanLookupTableFlag<XLEN> {
    pub fn iter() -> impl Iterator<Item = Self> {
        let sumcheck_id = SumcheckId::InstructionReadRaf;

        LookupTables::<XLEN>::iter()
            .enumerate()
            .map(move |(i, lookup_table)| {
                let var =
                    ZkLeanVarRef::virtual_var(sumcheck_id, VirtualPolynomial::LookupTableFlag(i));
                let lookup_table_ident = ZkLeanLookupTable::from(lookup_table).name();

                ZkLeanLookupTableFlag {
                    var,
                    lookup_table_ident,
                }
            })
    }

    pub fn zklean_pretty_print(
        &self,
        f: &mut impl std::io::Write,
        vars_ident: &str,
        indent_level: usize,
    ) -> std::io::Result<()> {
        writeln!(
            f,
            "{}({vars_ident}.{}, {}),",
            indent(indent_level),
            self.var,
            self.lookup_table_ident
        )
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
    left_operand: ZkLeanVarRef,
    right_operand: ZkLeanVarRef,
    /// A Boolean flag variable, set as follows:
    /// - 0: MLE input is the interleaving of the left and right operands
    /// - 1: MLE input is the concatenation of the left and right operands
    interleaving_flag: ZkLeanVarRef,
    /// Boolean flags for each lookup table, along with the corresponding lookup-table identifiers
    lookup_table_flags: Vec<ZkLeanLookupTableFlag<XLEN>>,
    /// Variable to constrain equal to muxed lookups
    lookup_output: ZkLeanVarRef,
}

impl<const XLEN: usize> ZkLeanLookupTableFlags<XLEN> {
    pub fn extract() -> Self {
        let left_operand = ZkLeanVarRef::virtual_var(
            SumcheckId::SpartanOuter,
            VirtualPolynomial::LeftLookupOperand,
        );
        let right_operand = ZkLeanVarRef::virtual_var(
            SumcheckId::SpartanOuter,
            VirtualPolynomial::RightLookupOperand,
        );
        let interleaving_flag = ZkLeanVarRef::virtual_var(
            SumcheckId::InstructionReadRaf,
            VirtualPolynomial::InstructionRafFlag,
        );
        let lookup_table_flags = ZkLeanLookupTableFlag::<XLEN>::iter().collect();
        let lookup_output =
            ZkLeanVarRef::virtual_var(SumcheckId::SpartanOuter, VirtualPolynomial::LookupOutput);

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
        writeln!(f, "{}: ZKBuilder f (ZKExpr f) :=", indent(indent_level))?;
        writeln!(
            f,
            "{}mux_mles interleaving left right #[",
            indent(indent_level)
        )?;
        indent_level += 1;
        for flag in self.lookup_table_flags.iter() {
            flag.zklean_pretty_print(f, vars_ident, indent_level)?;
        }
        indent_level -= 1;
        writeln!(f, "{}]", indent(indent_level))?;
        indent_level -= 1;

        writeln!(f)?;
        writeln!(
            f,
            "{}def lookup_step [ZKField f] ({vars_ident} : SumcheckVars f): ZKBuilder f PUnit := do",
            indent(indent_level),
        )?;
        indent_level += 1;
        writeln!(
            f,
            "{}let left := {vars_ident}.{}",
            indent(indent_level),
            self.left_operand
        )?;
        writeln!(
            f,
            "{}let right := {vars_ident}.{}",
            indent(indent_level),
            self.right_operand
        )?;
        writeln!(f, "{}let concatenated_eval <- mux_lookup_flags {vars_ident} Interleaving.Concatenated left right", indent(indent_level))?;
        writeln!(f, "{}let interleaved_eval <- mux_lookup_flags {vars_ident} Interleaving.Interleaved left right", indent(indent_level))?;
        writeln!(f, "{}let res <- mux #[", indent(indent_level))?;
        indent_level += 1;
        writeln!(
            f,
            "{}({vars_ident}.{}, concatenated_eval),",
            indent(indent_level),
            self.interleaving_flag
        )?;
        writeln!(
            f,
            "{}(1 - {vars_ident}.{}, interleaved_eval),",
            indent(indent_level),
            self.interleaving_flag
        )?;
        indent_level -= 1;
        writeln!(f, "{}]", indent(indent_level))?;
        writeln!(
            f,
            "{}ZKBuilder.constrainEq res {vars_ident}.{}",
            indent(indent_level),
            self.lookup_output
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
                String::from("zkLean"),
                String::from("Jolt.Sumchecks"),
                String::from("Jolt.LookupTables"),
                String::from("Jolt.Util"),
            ],
            contents,
        })
    }
}
