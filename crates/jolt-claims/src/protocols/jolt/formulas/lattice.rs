use jolt_field::Field;
use jolt_lookup_tables::{LookupTableKind, XLEN};
use jolt_poly::EqPolynomial;
use jolt_riscv::{NUM_CIRCUIT_FLAGS, NUM_INSTRUCTION_FLAGS};
use serde::{Deserialize, Serialize};

use super::super::{JoltAdviceKind, JoltCommittedPolynomial, JoltOpeningId, JoltRelationId};
use super::claim_reductions::bytecode;
use super::dimensions::{JoltFormulaPointError, TracePolynomialOrder, REGISTER_ADDRESS_BITS};

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
    BytecodeRegisterSelector { chunk: usize, selector: usize },
    BytecodeCircuitFlag { chunk: usize, flag: usize },
    BytecodeInstructionFlag { chunk: usize, flag: usize },
    BytecodeLookupSelector { chunk: usize },
    BytecodeRafFlag { chunk: usize },
    BytecodeUnexpandedPcBytes { chunk: usize },
    BytecodeImmBytes { chunk: usize },
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
    weighted_byte_decode_terms(family, [(limb, F::one())])
}

pub fn symbol_decode_terms<F: Field>(
    family: LatticePackedFamilyId,
    limb: usize,
    alphabet_size: usize,
) -> Vec<LatticePackedViewTerm<F>> {
    weighted_symbol_terms(
        family,
        limb,
        (0..alphabet_size).map(|symbol| F::from_u64(symbol as u64)),
    )
}

pub fn weighted_symbol_terms<F>(
    family: LatticePackedFamilyId,
    limb: usize,
    weights: impl IntoIterator<Item = F>,
) -> Vec<LatticePackedViewTerm<F>> {
    weights
        .into_iter()
        .enumerate()
        .map(|(symbol, coefficient)| {
            LatticePackedViewTerm::new(coefficient, family.clone(), limb, symbol)
        })
        .collect()
}

pub fn weighted_byte_decode_terms<F: Field>(
    family: LatticePackedFamilyId,
    limb_weights: impl IntoIterator<Item = (usize, F)>,
) -> Vec<LatticePackedViewTerm<F>> {
    limb_weights
        .into_iter()
        .flat_map(|(limb, limb_weight)| {
            let family = family.clone();
            (0..256).map(move |symbol| {
                LatticePackedViewTerm::new(
                    limb_weight * F::from_u64(symbol as u64),
                    family.clone(),
                    limb,
                    symbol,
                )
            })
        })
        .collect()
}

pub fn little_endian_byte_decode_terms<F: Field>(
    family: LatticePackedFamilyId,
    limb_count: usize,
) -> Vec<LatticePackedViewTerm<F>> {
    let mut limb_weights = Vec::with_capacity(limb_count);
    let mut place = F::one();
    for limb in 0..limb_count {
        limb_weights.push((limb, place));
        place *= F::from_u64(256);
    }
    weighted_byte_decode_terms(family, limb_weights)
}

pub fn bytecode_chunk_lattice_view_formula<F: Field>(
    chunk: usize,
    opening_point: &[F],
    trace_order: TracePolynomialOrder,
    log_bytecode: usize,
    field_byte_width: usize,
) -> Result<LatticePackedViewFormula<F>, JoltFormulaPointError> {
    let lane_vars = bytecode::committed_lane_vars();
    let expected = lane_vars + log_bytecode;
    if opening_point.len() != expected {
        return Err(JoltFormulaPointError::OpeningPointLengthMismatch {
            expected,
            got: opening_point.len(),
        });
    }
    let lane_point = match trace_order {
        TracePolynomialOrder::CycleMajor => &opening_point[..lane_vars],
        TracePolynomialOrder::AddressMajor => &opening_point[log_bytecode..],
    };
    let lane_weights = EqPolynomial::<F>::evals(lane_point, None);
    let lane_layout = bytecode::BYTECODE_LANE_LAYOUT;
    let register_count = 1usize << REGISTER_ADDRESS_BITS;
    let mut terms = Vec::new();

    for selector in 0..3 {
        let start = match selector {
            0 => lane_layout.rs1_start,
            1 => lane_layout.rs2_start,
            _ => lane_layout.rd_start,
        };
        terms.extend(weighted_symbol_terms(
            LatticePackedFamilyId::BytecodeRegisterSelector { chunk, selector },
            0,
            lane_weights[start..start + register_count].iter().copied(),
        ));
    }
    terms.extend(weighted_byte_decode_terms(
        LatticePackedFamilyId::BytecodeUnexpandedPcBytes { chunk },
        byte_limb_weights(lane_weights[lane_layout.unexp_pc_idx], 8),
    ));
    terms.extend(weighted_byte_decode_terms(
        LatticePackedFamilyId::BytecodeImmBytes { chunk },
        byte_limb_weights(lane_weights[lane_layout.imm_idx], field_byte_width),
    ));
    for flag in 0..NUM_CIRCUIT_FLAGS {
        terms.push(LatticePackedViewTerm::new(
            lane_weights[lane_layout.circuit_start + flag],
            LatticePackedFamilyId::BytecodeCircuitFlag { chunk, flag },
            0,
            1,
        ));
    }
    for flag in 0..NUM_INSTRUCTION_FLAGS {
        terms.push(LatticePackedViewTerm::new(
            lane_weights[lane_layout.instr_start + flag],
            LatticePackedFamilyId::BytecodeInstructionFlag { chunk, flag },
            0,
            1,
        ));
    }
    terms.extend(weighted_symbol_terms(
        LatticePackedFamilyId::BytecodeLookupSelector { chunk },
        0,
        lane_weights
            [lane_layout.lookup_start..lane_layout.lookup_start + LookupTableKind::<XLEN>::COUNT]
            .iter()
            .copied(),
    ));
    terms.push(LatticePackedViewTerm::new(
        lane_weights[lane_layout.raf_flag_idx],
        LatticePackedFamilyId::BytecodeRafFlag { chunk },
        0,
        1,
    ));

    Ok(LatticePackedViewFormula::linear_decoded(terms))
}

fn byte_limb_weights<F: Field>(lane_weight: F, limb_count: usize) -> Vec<(usize, F)> {
    let mut weights = Vec::with_capacity(limb_count);
    let mut place = F::one();
    for limb in 0..limb_count {
        weights.push((limb, lane_weight * place));
        place *= F::from_u64(256);
    }
    weights
}

#[cfg(test)]
mod tests {
    #![expect(
        clippy::panic,
        clippy::unwrap_used,
        reason = "tests fail loudly on unexpected errors"
    )]

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
    fn committed_bytecode_lattice_family_ids_name_lane_classes() {
        assert_ne!(
            LatticePackedFamilyId::BytecodeRegisterSelector {
                chunk: 0,
                selector: 0,
            },
            LatticePackedFamilyId::BytecodeRegisterSelector {
                chunk: 0,
                selector: 1,
            }
        );
        assert_ne!(
            LatticePackedFamilyId::BytecodeUnexpandedPcBytes { chunk: 0 },
            LatticePackedFamilyId::BytecodeImmBytes { chunk: 0 }
        );
        assert_ne!(
            LatticePackedFamilyId::BytecodeCircuitFlag { chunk: 0, flag: 0 },
            LatticePackedFamilyId::BytecodeInstructionFlag { chunk: 0, flag: 0 }
        );
    }

    #[test]
    fn bytecode_chunk_lattice_view_formula_uses_cycle_major_lane_weights() {
        let lane_vars = bytecode::committed_lane_vars();
        let log_bytecode = 2;
        let lane_point = (1..=lane_vars as u64).map(Fr::from_u64).collect::<Vec<_>>();
        let mut opening_point = lane_point.clone();
        opening_point.extend([Fr::from_u64(101), Fr::from_u64(103)]);
        let lane_weights = EqPolynomial::<Fr>::evals(&lane_point, None);
        let lane_layout = bytecode::BYTECODE_LANE_LAYOUT;

        let formula = bytecode_chunk_lattice_view_formula(
            2,
            &opening_point,
            TracePolynomialOrder::CycleMajor,
            log_bytecode,
            2,
        )
        .unwrap();
        let terms = linear_decoded_terms(&formula);

        assert_eq!(
            find_term(
                terms,
                LatticePackedFamilyId::BytecodeRegisterSelector {
                    chunk: 2,
                    selector: 2
                },
                0,
                5
            )
            .coefficient,
            lane_weights[lane_layout.rd_start + 5]
        );
        assert_eq!(
            find_term(
                terms,
                LatticePackedFamilyId::BytecodeCircuitFlag { chunk: 2, flag: 0 },
                0,
                1
            )
            .coefficient,
            lane_weights[lane_layout.circuit_start]
        );
        assert_eq!(
            find_term(
                terms,
                LatticePackedFamilyId::BytecodeLookupSelector { chunk: 2 },
                0,
                3
            )
            .coefficient,
            lane_weights[lane_layout.lookup_start + 3]
        );
        assert_eq!(
            find_term(
                terms,
                LatticePackedFamilyId::BytecodeUnexpandedPcBytes { chunk: 2 },
                1,
                7
            )
            .coefficient,
            lane_weights[lane_layout.unexp_pc_idx] * Fr::from_u64(256 * 7)
        );
        assert_eq!(
            find_term(
                terms,
                LatticePackedFamilyId::BytecodeImmBytes { chunk: 2 },
                1,
                9
            )
            .coefficient,
            lane_weights[lane_layout.imm_idx] * Fr::from_u64(256 * 9)
        );
        assert_eq!(
            find_term(
                terms,
                LatticePackedFamilyId::BytecodeRafFlag { chunk: 2 },
                0,
                1
            )
            .coefficient,
            lane_weights[lane_layout.raf_flag_idx]
        );
    }

    #[test]
    fn bytecode_chunk_lattice_view_formula_uses_address_major_lane_suffix() {
        let lane_vars = bytecode::committed_lane_vars();
        let log_bytecode = 2;
        let lane_point = (11..11 + lane_vars as u64)
            .map(Fr::from_u64)
            .collect::<Vec<_>>();
        let mut opening_point = vec![Fr::from_u64(101), Fr::from_u64(103)];
        opening_point.extend(lane_point.iter().copied());
        let lane_weights = EqPolynomial::<Fr>::evals(&lane_point, None);

        let formula = bytecode_chunk_lattice_view_formula(
            1,
            &opening_point,
            TracePolynomialOrder::AddressMajor,
            log_bytecode,
            1,
        )
        .unwrap();
        let terms = linear_decoded_terms(&formula);

        assert_eq!(
            find_term(
                terms,
                LatticePackedFamilyId::BytecodeRegisterSelector {
                    chunk: 1,
                    selector: 0
                },
                0,
                1
            )
            .coefficient,
            lane_weights[bytecode::BYTECODE_LANE_LAYOUT.rs1_start + 1]
        );
    }

    #[test]
    fn bytecode_chunk_lattice_view_formula_rejects_bad_point_length() {
        let expected = bytecode::committed_lane_vars() + 3;

        let err = bytecode_chunk_lattice_view_formula::<Fr>(
            0,
            &vec![Fr::from_u64(0); expected - 1],
            TracePolynomialOrder::CycleMajor,
            3,
            1,
        )
        .unwrap_err();

        assert_eq!(
            err,
            JoltFormulaPointError::OpeningPointLengthMismatch {
                expected,
                got: expected - 1
            }
        );
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
    fn weighted_symbol_terms_use_supplied_coefficients() {
        let terms = weighted_symbol_terms(
            LatticePackedFamilyId::InstructionRa { index: 0 },
            2,
            [Fr::from_u64(11), Fr::from_u64(13), Fr::from_u64(17)],
        );

        assert_eq!(terms.len(), 3);
        assert_eq!(terms[1].coefficient, Fr::from_u64(13));
        assert_eq!(
            terms[1].family,
            LatticePackedFamilyId::InstructionRa { index: 0 }
        );
        assert_eq!(terms[1].limb, 2);
        assert_eq!(terms[1].symbol, 1);
    }

    #[test]
    fn weighted_byte_decode_terms_scale_symbols_by_limb_weights() {
        let terms = weighted_byte_decode_terms(
            LatticePackedFamilyId::BytecodeChunk { index: 2 },
            [(3, Fr::from_u64(5)), (8, Fr::from_u64(7))],
        );

        assert_eq!(terms.len(), 512);
        assert_eq!(terms[9].coefficient, Fr::from_u64(45));
        assert_eq!(terms[9].limb, 3);
        assert_eq!(terms[9].symbol, 9);
        assert_eq!(terms[256 + 9].coefficient, Fr::from_u64(63));
        assert_eq!(terms[256 + 9].limb, 8);
        assert_eq!(
            terms[256 + 9].family,
            LatticePackedFamilyId::BytecodeChunk { index: 2 }
        );
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

    fn linear_decoded_terms(
        formula: &LatticePackedViewFormula<Fr>,
    ) -> &[LatticePackedViewTerm<Fr>] {
        match formula {
            LatticePackedViewFormula::LinearDecoded { terms } => terms,
            _ => panic!("expected linear decoded formula"),
        }
    }

    fn find_term(
        terms: &[LatticePackedViewTerm<Fr>],
        family: LatticePackedFamilyId,
        limb: usize,
        symbol: usize,
    ) -> &LatticePackedViewTerm<Fr> {
        terms
            .iter()
            .find(|term| term.family == family && term.limb == limb && term.symbol == symbol)
            .unwrap()
    }
}
