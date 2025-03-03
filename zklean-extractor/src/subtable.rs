use jolt_core::jolt::subtable;
use zklean_extractor::declare_subtables_enum;

use crate::util::ZkLeanReprField;

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
    subtable::sll::SllSubtable<0, 32>,
    subtable::sll::SllSubtable<1, 32>,
    subtable::sll::SllSubtable<2, 32>,
    subtable::sll::SllSubtable<3, 32>,
    //subtable::sll::SllSubtable<0, 64>,
    //subtable::sll::SllSubtable<1, 64>,
    //subtable::sll::SllSubtable<2, 64>,
    //subtable::sll::SllSubtable<3, 64>,
    //subtable::sll::SllSubtable<4, 64>,
    //subtable::sll::SllSubtable<5, 64>,
    //subtable::sll::SllSubtable<6, 64>,
    //subtable::sll::SllSubtable<7, 64>,
    subtable::sra_sign::SraSignSubtable<32>,
    //subtable::sra_sign::SraSignSubtable<64>,
    subtable::srl::SrlSubtable<0, 32>,
    subtable::srl::SrlSubtable<1, 32>,
    subtable::srl::SrlSubtable<2, 32>,
    subtable::srl::SrlSubtable<3, 32>,
    //subtable::srl::SrlSubtable<0, 64>,
    //subtable::srl::SrlSubtable<1, 64>,
    //subtable::srl::SrlSubtable<2, 64>,
    //subtable::srl::SrlSubtable<3, 64>,
    //subtable::srl::SrlSubtable<4, 64>,
    //subtable::srl::SrlSubtable<5, 64>,
    //subtable::srl::SrlSubtable<6, 64>,
    //subtable::srl::SrlSubtable<7, 64>,
    subtable::truncate_overflow::TruncateOverflowSubtable<32>,
    //subtable::truncate_overflow::TruncateOverflowSubtable<64>,
    subtable::xor::XorSubtable,
}

impl<F: ZkLeanReprField> NamedSubtable<F> {
    /// Pretty print a subtable as a ZkLean `Subtable`.
    pub fn zklean_pretty_print(&self, f: &mut impl std::io::Write, reg_size: usize) -> std::io::Result<()> {
        let name = self.name();
        let mle = self.evaluate_mle('x', reg_size).as_computation();

        f.write_fmt(format_args!("def {name}_{reg_size} [Field f] : Subtable f {reg_size}"))?;
        f.write_fmt(format_args!(" := subtableFromMLE (fun x => {mle})\n"))?;

        Ok(())
    }
}
