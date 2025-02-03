use jolt_core::jolt::subtable::{self, LassoSubtable};
use jolt_core::field::JoltField;

/// A [`JoltField`] that can be used to write a ZKLean representation of a computation.
pub trait ZkLeanReprField: JoltField + Sized {
    fn register(name: char, size: usize) -> Vec<Self>;

    fn as_computation(&self) -> String;

    fn write_lean_mle(&self, f: &mut impl std::io::Write, name: &String, reg_size: usize) -> std::io::Result<()> {
        f.write_fmt(format_args!("def {name} [Field f]: Subtable f "))?;
        f.write_fmt(format_args!(":= SubtableFromMLE {reg_size} (fun x => {})\n", self.as_computation()))?;
        Ok(())
    }
}

/// A collection of all the [`LassoSubtable`] instances that are used in tests within the subtables
/// module, including the different values used for the const generics. The tests instantiate these
/// with one or both of 8 or 16 for the bitwidth.
pub fn subtables<F: JoltField>(bitwidth: usize) -> Vec<(&'static str, Box<dyn LassoSubtable<F>>)> {
    let v: Vec<Option<(&'static str, Box<dyn LassoSubtable<F>>)>> = vec![
        Some(("and", Box::new(subtable::and::AndSubtable::new()))),
        Some(("div_by_zero", Box::new(subtable::div_by_zero::DivByZeroSubtable::new()))),
        Some(("eq", Box::new(subtable::eq::EqSubtable::new()))),
        Some(("eq_abs", Box::new(subtable::eq_abs::EqAbsSubtable::new()))),
        Some(("identity", Box::new(subtable::identity::IdentitySubtable::new()))),
        Some(("left_is_zero", Box::new(subtable::left_is_zero::LeftIsZeroSubtable::new()))),
        Some(("left_msb", Box::new(subtable::left_msb::LeftMSBSubtable::new()))),
        Some(("low_bit_0", Box::new(subtable::low_bit::LowBitSubtable::<_, 0>::new()))),
        Some(("low_bit_1", Box::new(subtable::low_bit::LowBitSubtable::<_, 1>::new()))),
        Some(("lt_abs", Box::new(subtable::lt_abs::LtAbsSubtable::new()))),
        Some(("ltu", Box::new(subtable::ltu::LtuSubtable::new()))),
        Some(("or", Box::new(subtable::or::OrSubtable::new()))),
        Some(("right_is_zero", Box::new(subtable::right_is_zero::RightIsZeroSubtable::new()))),
        Some(("right_msb", Box::new(subtable::right_msb::RightMSBSubtable::new()))),
        Some(("sign_extend_8", Box::new(subtable::sign_extend::SignExtendSubtable::<_, 8>::new()))),
        (bitwidth >= 16).then_some(("sign_extend_16", Box::new(subtable::sign_extend::SignExtendSubtable::<_, 16>::new()))),
        Some(("sll_0_32", Box::new(subtable::sll::SllSubtable::<_, 0, 32>::new()))),
        Some(("sll_1_32", Box::new(subtable::sll::SllSubtable::<_, 1, 32>::new()))),
        Some(("sll_2_32", Box::new(subtable::sll::SllSubtable::<_, 2, 32>::new()))),
        Some(("sll_3_32", Box::new(subtable::sll::SllSubtable::<_, 3, 32>::new()))),
        Some(("sra_sign_32", Box::new(subtable::sra_sign::SraSignSubtable::<_, 32>::new()))),
        Some(("srl_0_32", Box::new(subtable::srl::SrlSubtable::<_, 0, 32>::new()))),
        Some(("srl_1_32", Box::new(subtable::srl::SrlSubtable::<_, 1, 32>::new()))),
        Some(("srl_2_32", Box::new(subtable::srl::SrlSubtable::<_, 2, 32>::new()))),
        Some(("srl_3_32", Box::new(subtable::srl::SrlSubtable::<_, 3, 32>::new()))),
        Some(("truncate_overflow_32", Box::new(subtable::truncate_overflow::TruncateOverflowSubtable::<_, 32>::new()))),
        Some(("xor", Box::new(subtable::xor::XorSubtable::new()))),
    ];

    let v: Vec<(&'static str, Box<dyn LassoSubtable<F>>)> = v.into_iter().flatten().collect();
    v
}

#[cfg(test)]
pub mod test {
    use jolt_core::field::JoltField;

    /// A [`JoltField`] that can be evaluated over another [`JoltField`] (e.g., [`crate::mle_ast::MleAst`]).
    pub trait Evaluatable: Sized {
        fn evaluate<F: JoltField>(&self, vars: &[F]) -> F;

        /// Create a register named 'x' of size `size`.
        fn x_register(size: usize) -> Vec<Self>;
    }

    /// Given a function, try running it over a standard [`JoltField`], as well as constructing an
    /// [`Evaluatable`] [`JoltField`]. Check whether running the latter result over the standard
    /// [`JoltField`] gives the same result as the former.
    macro_rules! test_evaluate {
        ($eval_field:ty, $ref_field:ty, $fun:expr, $values:expr$(,)?) => {
            let reg_size = $values.len();
            let register = <$eval_field as Evaluatable>::x_register(reg_size);
            let expected_result = $fun(&$values);
            let to_evaluate = $fun(&register);
            let actual_result = to_evaluate.evaluate(&$values);
            prop_assert_eq!(actual_result, expected_result, "\n   mle: {}:", to_evaluate);
        }
    }

    pub(crate) use test_evaluate;

    /// Construct a proptest for a given field and subtable.
    macro_rules! make_subtable_test {
        ($eval_field:ty, $ref_field:ty, $mod:ident, $subtable:ident < $( $generic:literal ),* >, $reg_size:literal$(,)?) => {
            proptest! {
                #[test]
                fn $mod(values_u64 in proptest::collection::vec(proptest::num::u64::ANY, $reg_size)) {
                    use jolt_core::jolt::subtable::LassoSubtable;
                    let values = values_u64
                        .into_iter()
                        .map(jolt_core::field::JoltField::from_u64)
                        .collect::<Vec<$ref_field>>();
                    $crate::util::test::test_evaluate!(
                        $eval_field,
                        $ref_field,
                        |reg| jolt_core::jolt::subtable::$mod::$subtable::<_, $($generic),*>::new().evaluate_mle(reg),
                        values,
                    );
                }
            }
        };

        ($eval_field:ty, $ref_field:ty, $mod:ident, $subtable:ident, $reg_size:literal$(,)?) => {
            $crate::util::test::make_subtable_test!($eval_field, $ref_field, $mod, $subtable<>, $reg_size);
        };
    }

    pub(crate) use make_subtable_test;

    /// All the [`jolt_core::jolt::subtable::LassoSubtable`] instances to run tests over. These are
    /// all the test instances used in the subtable module, itself.
    macro_rules! make_subtable_test_module {
        ($eval_field:ty, $ref_field:ty) => {
            mod subtables {
                use super::*;
                $crate::util::test::make_subtable_test!($eval_field, $ref_field, and, AndSubtable, 8);
                $crate::util::test::make_subtable_test!($eval_field, $ref_field, div_by_zero, DivByZeroSubtable, 8);
                $crate::util::test::make_subtable_test!($eval_field, $ref_field, eq, EqSubtable, 8);
                $crate::util::test::make_subtable_test!($eval_field, $ref_field, eq_abs, EqAbsSubtable, 8);
                $crate::util::test::make_subtable_test!($eval_field, $ref_field, identity, IdentitySubtable, 8);
                $crate::util::test::make_subtable_test!($eval_field, $ref_field, left_is_zero, LeftIsZeroSubtable, 8);
                $crate::util::test::make_subtable_test!($eval_field, $ref_field, left_msb, LeftMSBSubtable, 8);
                $crate::util::test::make_subtable_test!($eval_field, $ref_field, low_bit, LowBitSubtable<2>, 8);
                $crate::util::test::make_subtable_test!($eval_field, $ref_field, lt_abs, LtAbsSubtable, 8);
                $crate::util::test::make_subtable_test!($eval_field, $ref_field, ltu, LtuSubtable, 8);
                $crate::util::test::make_subtable_test!($eval_field, $ref_field, or, OrSubtable, 8);
                $crate::util::test::make_subtable_test!($eval_field, $ref_field, right_is_zero, RightIsZeroSubtable, 8);
                $crate::util::test::make_subtable_test!($eval_field, $ref_field, right_msb, RightMSBSubtable, 8);
                $crate::util::test::make_subtable_test!($eval_field, $ref_field, sign_extend, SignExtendSubtable<8>, 8);
                $crate::util::test::make_subtable_test!($eval_field, $ref_field, sll, SllSubtable<2, 32>, 8);
                $crate::util::test::make_subtable_test!($eval_field, $ref_field, sra_sign, SraSignSubtable<32>, 8);
                $crate::util::test::make_subtable_test!($eval_field, $ref_field, srl, SrlSubtable<2, 32>, 8);
                $crate::util::test::make_subtable_test!($eval_field, $ref_field, truncate_overflow, TruncateOverflowSubtable<32>, 8);
                $crate::util::test::make_subtable_test!($eval_field, $ref_field, xor, XorSubtable, 8);
            }
        }
    }

    pub(crate) use make_subtable_test_module;
}
