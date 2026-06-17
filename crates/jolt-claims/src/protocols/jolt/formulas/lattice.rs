use jolt_field::Field;
use serde::{Deserialize, Serialize};

use super::super::{JoltAdviceKind, JoltCommittedPolynomial, JoltOpeningId, JoltRelationId};

#[derive(Hash, PartialEq, Eq, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum LatticePackedFamilyId {
    InstructionRa { index: usize },
    BytecodeRa { index: usize },
    RamRa { index: usize },
    IncByte { index: usize },
    IncSign,
    RamIncByte { index: usize },
    RamIncSign,
    RdIncByte { index: usize },
    RdIncSign,
    FieldRdIncByte { index: usize },
    FieldRdIncSign,
    AdviceBytes { kind: JoltAdviceKind, index: usize },
    BytecodeChunk { index: usize },
    ProgramImageInit,
    Custom { namespace: u32, index: usize },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LatticePackedViewFormula<F> {
    Direct {
        family: LatticePackedFamilyId,
        limb: usize,
        symbol: usize,
    },
    LinearDecoded {
        terms: Vec<LatticePackedViewTerm<F>>,
    },
    ReducedMasked {
        relation: JoltRelationId,
        output_openings: Vec<JoltOpeningId>,
    },
    MaskedDecoded,
}

impl<F> LatticePackedViewFormula<F> {
    pub fn direct(family: LatticePackedFamilyId, limb: usize, symbol: usize) -> Self {
        Self::Direct {
            family,
            limb,
            symbol,
        }
    }

    pub fn linear_decoded(terms: Vec<LatticePackedViewTerm<F>>) -> Self {
        Self::LinearDecoded { terms }
    }

    pub fn reduced_masked(relation: JoltRelationId, output_openings: Vec<JoltOpeningId>) -> Self {
        Self::ReducedMasked {
            relation,
            output_openings,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LatticePackedViewTerm<F> {
    pub coefficient: F,
    pub family: LatticePackedFamilyId,
    pub limb: usize,
    pub symbol: usize,
}

impl<F> LatticePackedViewTerm<F> {
    pub fn new(coefficient: F, family: LatticePackedFamilyId, limb: usize, symbol: usize) -> Self {
        Self {
            coefficient,
            family,
            limb,
            symbol,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LatticeFinalOpeningRequirement {
    PackedFamily {
        family: LatticePackedFamilyId,
        relation: JoltRelationId,
    },
    RequiresTranslation {
        relation: JoltRelationId,
    },
}

pub fn final_opening_lattice_requirement(
    polynomial: JoltCommittedPolynomial,
) -> LatticeFinalOpeningRequirement {
    match polynomial {
        JoltCommittedPolynomial::RdInc | JoltCommittedPolynomial::RamInc => {
            LatticeFinalOpeningRequirement::RequiresTranslation {
                relation: JoltRelationId::IncClaimReduction,
            }
        }
        JoltCommittedPolynomial::InstructionRa(index) => packed_family_requirement(
            LatticePackedFamilyId::InstructionRa { index },
            JoltRelationId::HammingWeightClaimReduction,
        ),
        JoltCommittedPolynomial::BytecodeRa(index) => packed_family_requirement(
            LatticePackedFamilyId::BytecodeRa { index },
            JoltRelationId::HammingWeightClaimReduction,
        ),
        JoltCommittedPolynomial::RamRa(index) => packed_family_requirement(
            LatticePackedFamilyId::RamRa { index },
            JoltRelationId::HammingWeightClaimReduction,
        ),
        JoltCommittedPolynomial::TrustedAdvice => packed_family_requirement(
            LatticePackedFamilyId::AdviceBytes {
                kind: JoltAdviceKind::Trusted,
                index: 0,
            },
            JoltRelationId::AdviceClaimReduction,
        ),
        JoltCommittedPolynomial::UntrustedAdvice => packed_family_requirement(
            LatticePackedFamilyId::AdviceBytes {
                kind: JoltAdviceKind::Untrusted,
                index: 0,
            },
            JoltRelationId::AdviceClaimReduction,
        ),
        JoltCommittedPolynomial::BytecodeChunk(index) => packed_family_requirement(
            LatticePackedFamilyId::BytecodeChunk { index },
            JoltRelationId::BytecodeClaimReduction,
        ),
        JoltCommittedPolynomial::ProgramImageInit => packed_family_requirement(
            LatticePackedFamilyId::ProgramImageInit,
            JoltRelationId::ProgramImageClaimReduction,
        ),
    }
}

fn packed_family_requirement(
    family: LatticePackedFamilyId,
    relation: JoltRelationId,
) -> LatticeFinalOpeningRequirement {
    LatticeFinalOpeningRequirement::PackedFamily { family, relation }
}

pub fn byte_decode_terms<F: Field>(
    family: LatticePackedFamilyId,
    limb: usize,
) -> Vec<LatticePackedViewTerm<F>> {
    symbol_decode_terms(family, limb, 256)
}

pub fn symbol_decode_terms<F: Field>(
    family: LatticePackedFamilyId,
    limb: usize,
    alphabet_size: usize,
) -> Vec<LatticePackedViewTerm<F>> {
    (0..alphabet_size)
        .map(|symbol| {
            LatticePackedViewTerm::new(F::from_u64(symbol as u64), family.clone(), limb, symbol)
        })
        .collect()
}

pub fn little_endian_byte_decode_terms<F: Field>(
    family: LatticePackedFamilyId,
    limb_count: usize,
) -> Vec<LatticePackedViewTerm<F>> {
    let mut terms = Vec::with_capacity(256 * limb_count);
    let mut place = F::one();
    for limb in 0..limb_count {
        terms.extend((0..256).map(|symbol| {
            LatticePackedViewTerm::new(
                place * F::from_u64(symbol as u64),
                family.clone(),
                limb,
                symbol,
            )
        }));
        place *= F::from_u64(256);
    }
    terms
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{Fr, FromPrimitiveInt};

    #[test]
    fn final_opening_lattice_requirement_marks_increments_as_translation() {
        assert_eq!(
            final_opening_lattice_requirement(JoltCommittedPolynomial::RamInc),
            LatticeFinalOpeningRequirement::RequiresTranslation {
                relation: JoltRelationId::IncClaimReduction
            }
        );
        assert_eq!(
            final_opening_lattice_requirement(JoltCommittedPolynomial::RdInc),
            LatticeFinalOpeningRequirement::RequiresTranslation {
                relation: JoltRelationId::IncClaimReduction
            }
        );
    }

    #[test]
    fn final_opening_lattice_requirement_names_packed_families() {
        assert_eq!(
            final_opening_lattice_requirement(JoltCommittedPolynomial::InstructionRa(2)),
            LatticeFinalOpeningRequirement::PackedFamily {
                family: LatticePackedFamilyId::InstructionRa { index: 2 },
                relation: JoltRelationId::HammingWeightClaimReduction,
            }
        );
        assert_eq!(
            final_opening_lattice_requirement(JoltCommittedPolynomial::ProgramImageInit),
            LatticeFinalOpeningRequirement::PackedFamily {
                family: LatticePackedFamilyId::ProgramImageInit,
                relation: JoltRelationId::ProgramImageClaimReduction,
            }
        );
    }

    #[test]
    fn byte_decode_terms_are_little_endian_symbol_weights() {
        let terms = byte_decode_terms::<Fr>(LatticePackedFamilyId::BytecodeChunk { index: 0 }, 3);

        assert_eq!(terms.len(), 256);
        assert_eq!(terms[7].coefficient, Fr::from_u64(7));
        assert_eq!(
            terms[7].family,
            LatticePackedFamilyId::BytecodeChunk { index: 0 }
        );
        assert_eq!(terms[7].limb, 3);
        assert_eq!(terms[7].symbol, 7);
    }

    #[test]
    fn symbol_decode_terms_support_non_byte_alphabets() {
        let terms = symbol_decode_terms::<Fr>(LatticePackedFamilyId::RamRa { index: 1 }, 0, 4);

        assert_eq!(terms.len(), 4);
        assert_eq!(terms[3].coefficient, Fr::from_u64(3));
        assert_eq!(terms[3].family, LatticePackedFamilyId::RamRa { index: 1 });
        assert_eq!(terms[3].limb, 0);
        assert_eq!(terms[3].symbol, 3);
    }

    #[test]
    fn little_endian_byte_decode_terms_weight_limbs_by_place_value() {
        let terms =
            little_endian_byte_decode_terms::<Fr>(LatticePackedFamilyId::ProgramImageInit, 2);

        assert_eq!(terms.len(), 512);
        assert_eq!(terms[7].coefficient, Fr::from_u64(7));
        assert_eq!(terms[7].limb, 0);
        assert_eq!(terms[7].symbol, 7);
        assert_eq!(terms[256 + 7].coefficient, Fr::from_u64(256 * 7));
        assert_eq!(terms[256 + 7].limb, 1);
        assert_eq!(terms[256 + 7].symbol, 7);
        assert_eq!(
            terms[256 + 7].family,
            LatticePackedFamilyId::ProgramImageInit
        );
    }
}
