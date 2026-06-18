use jolt_field::Field;

use crate::protocols::jolt::{
    weighted_byte_decode_terms, LatticePackedFamilyId, LatticePackedViewFormula,
    LatticePackedViewTerm,
};

pub fn field_rd_inc_lattice_view_formula<F: Field>(
    byte_width: usize,
) -> LatticePackedViewFormula<F> {
    LatticePackedViewFormula::linear_decoded(field_rd_inc_byte_terms(byte_width))
}

pub fn field_rd_inc_byte_terms<F: Field>(byte_width: usize) -> Vec<LatticePackedViewTerm<F>> {
    let mut terms = Vec::with_capacity(byte_width * 256);
    let mut place = F::one();
    for index in 0..byte_width {
        terms.extend(weighted_byte_decode_terms(
            LatticePackedFamilyId::FieldRdIncByte { index },
            [(0, place)],
        ));
        place *= F::from_u64(256);
    }
    terms
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{Fr, FromPrimitiveInt};

    #[test]
    fn field_rd_inc_terms_decode_separate_little_endian_byte_families() {
        let terms = field_rd_inc_byte_terms::<Fr>(2);

        assert_eq!(terms.len(), 512);
        assert_eq!(terms[7].coefficient, Fr::from_u64(7));
        assert_eq!(
            terms[7].family,
            LatticePackedFamilyId::FieldRdIncByte { index: 0 }
        );
        assert_eq!(terms[7].limb, 0);
        assert_eq!(terms[7].symbol, 7);

        let second_byte = &terms[256 + 3];
        assert_eq!(second_byte.coefficient, Fr::from_u64(3 * 256));
        assert_eq!(
            second_byte.family,
            LatticePackedFamilyId::FieldRdIncByte { index: 1 }
        );
        assert_eq!(second_byte.limb, 0);
        assert_eq!(second_byte.symbol, 3);
    }
}
