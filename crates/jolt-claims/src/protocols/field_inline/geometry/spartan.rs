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

pub const FIELD_INLINE_SPARTAN_OUTER_R1CS_INPUT_COUNT: usize =
    FIELD_INLINE_SPARTAN_OUTER_R1CS_INPUTS.len();

pub fn outer_opening(polynomial: FieldInlineVirtualPolynomial) -> FieldInlineOpeningId {
    FieldInlineOpeningId::virtual_polynomial(
        polynomial,
        FieldInlineRelationId::FieldRegistersSpartanOuter,
    )
}

pub fn outer_output_openings() -> [FieldInlineOpeningId; FIELD_INLINE_SPARTAN_OUTER_R1CS_INPUT_COUNT]
{
    [
        outer_opening(FieldInlineVirtualPolynomial::FieldRs1Value),
        outer_opening(FieldInlineVirtualPolynomial::FieldRs2Value),
        outer_opening(FieldInlineVirtualPolynomial::FieldRdValue),
        outer_opening(FieldInlineVirtualPolynomial::FieldProduct),
        outer_opening(FieldInlineVirtualPolynomial::FieldInvProduct),
        outer_opening(FieldInlineVirtualPolynomial::FieldOpFlag(
            FieldInlineOpFlag::Add,
        )),
        outer_opening(FieldInlineVirtualPolynomial::FieldOpFlag(
            FieldInlineOpFlag::Sub,
        )),
        outer_opening(FieldInlineVirtualPolynomial::FieldOpFlag(
            FieldInlineOpFlag::Mul,
        )),
        outer_opening(FieldInlineVirtualPolynomial::FieldOpFlag(
            FieldInlineOpFlag::Inv,
        )),
        outer_opening(FieldInlineVirtualPolynomial::FieldOpFlag(
            FieldInlineOpFlag::AssertEq,
        )),
        outer_opening(FieldInlineVirtualPolynomial::FieldOpFlag(
            FieldInlineOpFlag::LoadFromX,
        )),
        outer_opening(FieldInlineVirtualPolynomial::FieldOpFlag(
            FieldInlineOpFlag::StoreToX,
        )),
        outer_opening(FieldInlineVirtualPolynomial::FieldOpFlag(
            FieldInlineOpFlag::LoadImm,
        )),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spartan_outer_openings_follow_input_order() {
        let openings = outer_output_openings();
        assert_eq!(openings.len(), FIELD_INLINE_SPARTAN_OUTER_R1CS_INPUT_COUNT);

        let expected = FIELD_INLINE_SPARTAN_OUTER_R1CS_INPUTS
            .into_iter()
            .map(outer_opening)
            .collect::<Vec<_>>();
        assert_eq!(openings.to_vec(), expected);
    }

    #[test]
    fn spartan_outer_openings_use_field_registers_relation() {
        for opening in outer_output_openings() {
            let FieldInlineOpeningId::Polynomial { relation, .. } = opening;
            assert_eq!(relation, FieldInlineRelationId::FieldRegistersSpartanOuter);
        }
    }
}
