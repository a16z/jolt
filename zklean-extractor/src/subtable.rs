use jolt_core::jolt::subtable;
use zklean_extractor::declare_subtables_enum;

use crate::{modules::{AsModule, Module}, util::{indent, ZkLeanReprField}};

declare_subtables_enum! {
    NamedSubtable,
    subtable::and::AndSubtable,
    subtable::div_by_zero::DivByZeroSubtable,
    subtable::eq::EqSubtable,
    subtable::eq_abs::EqAbsSubtable,
    subtable::identity::IdentitySubtable,
    subtable::left_is_zero::LeftIsZeroSubtable,
    subtable::left_msb::LeftMSBSubtable,
    subtable::low_bit::LowBitSubtable<0>,
    subtable::low_bit::LowBitSubtable<1>,
    subtable::lt_abs::LtAbsSubtable,
    subtable::ltu::LtuSubtable,
    subtable::or::OrSubtable,
    subtable::right_is_zero::RightIsZeroSubtable,
    subtable::right_msb::RightMSBSubtable,
    subtable::sign_extend::SignExtendSubtable<8>,
    subtable::sign_extend::SignExtendSubtable<16>,
    subtable::sll::SllSubtable<0, 32>,
    subtable::sll::SllSubtable<1, 32>,
    subtable::sll::SllSubtable<2, 32>,
    subtable::sll::SllSubtable<3, 32>,
    subtable::sra_sign::SraSignSubtable<32>,
    subtable::srl::SrlSubtable<0, 32>,
    subtable::srl::SrlSubtable<1, 32>,
    subtable::srl::SrlSubtable<2, 32>,
    subtable::srl::SrlSubtable<3, 32>,
    subtable::xor::XorSubtable,
}

impl<F: ZkLeanReprField, const LOG_M: usize> NamedSubtable<F, LOG_M> {
    /// Pretty print a subtable as a ZkLean `Subtable`.
    pub fn zklean_pretty_print(
        &self,
        f: &mut impl std::io::Write,
        mut indent_level: usize,
    ) -> std::io::Result<()> {
        let name = self.name();
        let mle = self.evaluate_mle('x').as_computation();

        f.write_fmt(format_args!(
                "{}def {name}_{LOG_M} [Field f] : Subtable f {LOG_M} :=\n",
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

pub struct ZkLeanSubtables<F: ZkLeanReprField, const LOG_M: usize> {
    subtables: Vec<NamedSubtable<F, LOG_M>>,
}

impl<F: ZkLeanReprField, const LOG_M: usize> ZkLeanSubtables<F, LOG_M> {
    pub fn extract() -> Self {
        Self {
            subtables: NamedSubtable::<F, LOG_M>::variants(),
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

impl<F: ZkLeanReprField, const LOG_M: usize> AsModule for ZkLeanSubtables<F, LOG_M> {
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
