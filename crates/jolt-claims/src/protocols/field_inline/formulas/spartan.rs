use super::super::{
    FieldInlineOpFlag, FieldInlineOpeningId, FieldInlineRelationId, FieldInlineVirtualPolynomial,
};

pub const FIELD_INLINE_SPARTAN_OUTER_R1CS_INPUTS: [FieldInlineVirtualPolynomial; 13] = [
    FieldInlineVirtualPolynomial::FieldRs1Value,
    FieldInlineVirtualPolynomial::FieldRs2Value,
    FieldInlineVirtualPolynomial::FieldRdValue,
    FieldInlineVirtualPolynomial::FieldProduct,
    FieldInlineVirtualPolynomial::FieldInvProduct,
    FieldInlineVirtualPolynomial::FieldOpFlag(FieldInlineOpFlag::Add),
    FieldInlineVirtualPolynomial::FieldOpFlag(FieldInlineOpFlag::Sub),
    FieldInlineVirtualPolynomial::FieldOpFlag(FieldInlineOpFlag::Mul),
    FieldInlineVirtualPolynomial::FieldOpFlag(FieldInlineOpFlag::Inv),
    FieldInlineVirtualPolynomial::FieldOpFlag(FieldInlineOpFlag::AssertEq),
    FieldInlineVirtualPolynomial::FieldOpFlag(FieldInlineOpFlag::LoadFromX),
    FieldInlineVirtualPolynomial::FieldOpFlag(FieldInlineOpFlag::StoreToX),
    FieldInlineVirtualPolynomial::FieldOpFlag(FieldInlineOpFlag::LoadImm),
];

pub fn outer_opening(polynomial: FieldInlineVirtualPolynomial) -> FieldInlineOpeningId {
    FieldInlineOpeningId::virtual_polynomial(
        polynomial,
        FieldInlineRelationId::FieldRegistersSpartanOuter,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn field_inline_spartan_inputs_match_appended_r1cs_order() {
        assert_eq!(
            FIELD_INLINE_SPARTAN_OUTER_R1CS_INPUTS,
            [
                FieldInlineVirtualPolynomial::FieldRs1Value,
                FieldInlineVirtualPolynomial::FieldRs2Value,
                FieldInlineVirtualPolynomial::FieldRdValue,
                FieldInlineVirtualPolynomial::FieldProduct,
                FieldInlineVirtualPolynomial::FieldInvProduct,
                FieldInlineVirtualPolynomial::FieldOpFlag(FieldInlineOpFlag::Add),
                FieldInlineVirtualPolynomial::FieldOpFlag(FieldInlineOpFlag::Sub),
                FieldInlineVirtualPolynomial::FieldOpFlag(FieldInlineOpFlag::Mul),
                FieldInlineVirtualPolynomial::FieldOpFlag(FieldInlineOpFlag::Inv),
                FieldInlineVirtualPolynomial::FieldOpFlag(FieldInlineOpFlag::AssertEq),
                FieldInlineVirtualPolynomial::FieldOpFlag(FieldInlineOpFlag::LoadFromX),
                FieldInlineVirtualPolynomial::FieldOpFlag(FieldInlineOpFlag::StoreToX),
                FieldInlineVirtualPolynomial::FieldOpFlag(FieldInlineOpFlag::LoadImm),
            ]
        );
    }
}
