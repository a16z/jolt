use jolt_field::{Field, FromPrimitiveInt, RingCore};
use jolt_lookup_tables::{LookupTableKind, XLEN};
use jolt_poly::EqPolynomial;
use jolt_riscv::{NUM_CIRCUIT_FLAGS, NUM_INSTRUCTION_FLAGS};
use serde::{Deserialize, Serialize};

use crate::{challenge, constant, opening, public};

use super::super::{
    IncVirtualizationChallenge, IncVirtualizationPublic, JoltChallengeId, JoltExpr, JoltPublicId,
    JoltRelationClaims, UnsignedIncChunkReconstructionChallenge,
    UnsignedIncChunkReconstructionPublic,
};
use super::super::{JoltAdviceKind, JoltCommittedPolynomial, JoltOpeningId, JoltRelationId};
use super::claim_reductions::bytecode as bytecode_reduction;
use super::dimensions::{
    JoltFormulaPointError, JoltSumcheckSpec, TraceDimensions, TracePolynomialOrder,
    REGISTER_ADDRESS_BITS,
};
use jolt_riscv::CircuitFlags;

pub const UNSIGNED_INC_BITS: usize = 64;

pub use jolt_openings::{
    packing_validity_digest, PackingAdviceKind, PackingFamilyId, PackingValidityDigest,
    PackingValidityKind, PackingValidityRequirement, PackingViewFormula, PackingViewTerm,
};

pub fn inc_virtualization_relation() -> JoltRelationId {
    JoltRelationId::IncVirtualization
}

pub fn unsigned_inc_claim_reduction_relation() -> JoltRelationId {
    JoltRelationId::UnsignedIncClaimReduction
}

pub fn unsigned_inc_chunk_reconstruction_relation() -> JoltRelationId {
    JoltRelationId::UnsignedIncChunkReconstruction
}

pub fn inc_virtualization_claim<F>(dimensions: TraceDimensions) -> JoltRelationClaims<F>
where
    F: RingCore,
{
    let gamma = inc_virtualization_challenge(IncVirtualizationChallenge::Gamma);

    let input = opening(inc_virtualization_ram_read_write_opening())
        + gamma.clone() * opening(inc_virtualization_ram_val_check_opening())
        + gamma.clone().pow(2) * opening(inc_virtualization_rd_read_write_opening())
        + gamma.clone().pow(3) * opening(inc_virtualization_rd_val_evaluation_opening());

    let ram_coeff = inc_virtualization_public(IncVirtualizationPublic::EqRamReadWrite)
        + gamma.clone() * inc_virtualization_public(IncVirtualizationPublic::EqRamValCheck);
    let gamma_2 = gamma.clone().pow(2);
    let rd_coeff = inc_virtualization_public(IncVirtualizationPublic::EqRegistersReadWrite)
        + gamma.clone()
            * inc_virtualization_public(IncVirtualizationPublic::EqRegistersValEvaluation);
    let store = opening(inc_virtualization_store_opening());
    let output = opening(inc_virtualization_inc_opening())
        * (ram_coeff * store.clone() + gamma_2 * rd_coeff * (JoltExpr::one() - store));

    JoltRelationClaims::new(
        JoltRelationId::IncVirtualization,
        dimensions.sumcheck(3),
        input,
        output,
    )
}

pub fn unsigned_inc_claim_reduction_claim<F>(dimensions: TraceDimensions) -> JoltRelationClaims<F>
where
    F: RingCore + FromPrimitiveInt,
{
    let input = opening(inc_virtualization_inc_opening()) + constant(F::from_u128(1u128 << 64));
    let output = opening(unsigned_inc_opening());

    JoltRelationClaims::new(
        JoltRelationId::UnsignedIncClaimReduction,
        dimensions.sumcheck(2),
        input,
        output,
    )
}

pub fn unsigned_inc_msb_booleanity_claim<F>(dimensions: TraceDimensions) -> JoltRelationClaims<F>
where
    F: RingCore,
{
    let msb = opening(unsigned_inc_msb_opening());
    JoltRelationClaims::new(
        JoltRelationId::Booleanity,
        dimensions.sumcheck(2),
        JoltExpr::zero(),
        msb.clone() * msb.clone() - msb,
    )
}

fn inc_virtualization_challenge<F>(id: IncVirtualizationChallenge) -> JoltExpr<F>
where
    F: RingCore,
{
    challenge(JoltChallengeId::from(id))
}

fn inc_virtualization_public<F>(id: IncVirtualizationPublic) -> JoltExpr<F>
where
    F: RingCore,
{
    public(JoltPublicId::from(id))
}

fn unsigned_inc_chunk_reconstruction_challenge<F>(
    id: UnsignedIncChunkReconstructionChallenge,
) -> JoltExpr<F>
where
    F: RingCore,
{
    challenge(JoltChallengeId::from(id))
}

fn unsigned_inc_chunk_reconstruction_public<F>(
    id: UnsignedIncChunkReconstructionPublic,
) -> JoltExpr<F>
where
    F: RingCore,
{
    public(JoltPublicId::from(id))
}

pub fn inc_virtualization_input_openings() -> [JoltOpeningId; 4] {
    [
        inc_virtualization_ram_read_write_opening(),
        inc_virtualization_ram_val_check_opening(),
        inc_virtualization_rd_read_write_opening(),
        inc_virtualization_rd_val_evaluation_opening(),
    ]
}

pub fn inc_virtualization_output_openings() -> [JoltOpeningId; 2] {
    [
        inc_virtualization_inc_opening(),
        inc_virtualization_store_opening(),
    ]
}

pub fn inc_virtualization_ram_read_write_opening() -> JoltOpeningId {
    JoltOpeningId::committed(
        JoltCommittedPolynomial::RamInc,
        JoltRelationId::RamReadWriteChecking,
    )
}

pub fn inc_virtualization_ram_val_check_opening() -> JoltOpeningId {
    JoltOpeningId::committed(JoltCommittedPolynomial::RamInc, JoltRelationId::RamValCheck)
}

pub fn inc_virtualization_rd_read_write_opening() -> JoltOpeningId {
    JoltOpeningId::committed(
        JoltCommittedPolynomial::RdInc,
        JoltRelationId::RegistersReadWriteChecking,
    )
}

pub fn inc_virtualization_rd_val_evaluation_opening() -> JoltOpeningId {
    JoltOpeningId::committed(
        JoltCommittedPolynomial::RdInc,
        JoltRelationId::RegistersValEvaluation,
    )
}

pub fn inc_virtualization_inc_opening() -> JoltOpeningId {
    JoltOpeningId::lattice(JoltRelationId::IncVirtualization, 0)
}

pub fn inc_virtualization_store_opening() -> JoltOpeningId {
    JoltOpeningId::lattice(JoltRelationId::IncVirtualization, 1)
}

pub fn unsigned_inc_input_opening() -> JoltOpeningId {
    inc_virtualization_inc_opening()
}

pub fn unsigned_inc_opening() -> JoltOpeningId {
    JoltOpeningId::lattice(JoltRelationId::UnsignedIncClaimReduction, 0)
}

pub fn unsigned_inc_msb_opening() -> JoltOpeningId {
    JoltOpeningId::lattice(JoltRelationId::UnsignedIncClaimReduction, 1)
}

pub fn unsigned_inc_chunk_opening(index: usize) -> JoltOpeningId {
    JoltOpeningId::lattice(JoltRelationId::UnsignedIncClaimReduction, 2 + index)
}

pub fn unsigned_inc_lower_chunk_count(log_k_chunk: usize) -> Option<usize> {
    (log_k_chunk != 0 && UNSIGNED_INC_BITS.is_multiple_of(log_k_chunk))
        .then_some(UNSIGNED_INC_BITS / log_k_chunk)
}

pub fn unsigned_inc_chunk_reconstruction_claim<F>(
    log_k_chunk: usize,
) -> Option<JoltRelationClaims<F>>
where
    F: RingCore + FromPrimitiveInt,
{
    let chunk_count = unsigned_inc_lower_chunk_count(log_k_chunk)?;
    let gamma =
        unsigned_inc_chunk_reconstruction_challenge(UnsignedIncChunkReconstructionChallenge::Gamma);
    let eq_booleanity_address = unsigned_inc_chunk_reconstruction_public(
        UnsignedIncChunkReconstructionPublic::EqBooleanityAddress,
    );
    let identity_at_address = unsigned_inc_chunk_reconstruction_public(
        UnsignedIncChunkReconstructionPublic::IdentityAtAddress,
    );
    let delta = gamma.clone().pow(2 * chunk_count);
    let lower_value = opening(unsigned_inc_opening())
        - constant(F::from_u128(1u128 << 64)) * opening(unsigned_inc_msb_opening());

    let mut input = delta.clone() * lower_value;
    let mut output = JoltExpr::zero();
    let mut place = F::one();
    for index in 0..chunk_count {
        input = input
            + gamma.clone().pow(2 * index)
            + gamma.clone().pow(2 * index + 1) * opening(unsigned_inc_chunk_opening(index));
        let output_coeff = gamma.clone().pow(2 * index)
            + gamma.clone().pow(2 * index + 1) * eq_booleanity_address.clone()
            + delta.clone() * constant(place) * identity_at_address.clone();
        output = output + output_coeff * opening(unsigned_inc_chunk_opening(index));
        place *= F::from_u64(1u64 << log_k_chunk);
    }

    Some(
        JoltRelationClaims::new(
            JoltRelationId::UnsignedIncChunkReconstruction,
            JoltSumcheckSpec::boolean(log_k_chunk, 3),
            input,
            output,
        )
        .with_input_challenges([JoltChallengeId::from(
            UnsignedIncChunkReconstructionChallenge::Gamma,
        )]),
    )
}

pub fn bytecode_store_flag_lattice_view_formula<F: Field>(chunk: usize) -> PackingViewFormula<F> {
    PackingViewFormula::direct(
        PackingFamilyId::BytecodeCircuitFlag {
            chunk,
            flag: CircuitFlags::Store as usize,
        },
        0,
        1,
    )
}

pub fn bytecode_rd_present_lattice_view_formula<F: Field>(chunk: usize) -> PackingViewFormula<F> {
    PackingViewFormula::linear_decoded(weighted_symbol_terms(
        PackingFamilyId::BytecodeRegisterSelector { chunk, selector: 2 },
        0,
        [F::one(); 1 << REGISTER_ADDRESS_BITS],
    ))
}

pub fn unsigned_inc_lower_value_lattice_view_formula<F: Field>(
    log_k_chunk: usize,
) -> Option<PackingViewFormula<F>> {
    Some(PackingViewFormula::linear_decoded(
        unsigned_inc_lower_value_terms(log_k_chunk)?,
    ))
}

pub fn unsigned_inc_lower_value_terms<F: Field>(
    log_k_chunk: usize,
) -> Option<Vec<PackingViewTerm<F>>> {
    let chunk_count = unsigned_inc_lower_chunk_count(log_k_chunk)?;
    let alphabet_size = 1usize << log_k_chunk;
    let mut terms = Vec::with_capacity(chunk_count * alphabet_size);
    let mut place = F::one();
    for index in 0..chunk_count {
        terms.extend(weighted_symbol_terms(
            PackingFamilyId::UnsignedIncChunk { index },
            0,
            (0..alphabet_size).map(|symbol| place * F::from_u64(symbol as u64)),
        ));
        place *= F::from_u64(1u64 << log_k_chunk);
    }
    Some(terms)
}

pub fn unsigned_inc_msb_lattice_view_formula<F: Field>() -> PackingViewFormula<F> {
    PackingViewFormula::direct(PackingFamilyId::UnsignedIncMsb, 0, 1)
}

pub fn unsigned_inc_validity_requirements(
    log_k_chunk: usize,
) -> Option<Vec<PackingValidityRequirement>> {
    let chunk_count = unsigned_inc_lower_chunk_count(log_k_chunk)?;
    let alphabet_size = 1usize << log_k_chunk;
    let mut requirements = (0..chunk_count)
        .map(|index| {
            PackingValidityRequirement::exact_one_hot(
                PackingFamilyId::UnsignedIncChunk { index },
                1,
                alphabet_size,
            )
        })
        .collect::<Vec<_>>();
    requirements.push(PackingValidityRequirement::boolean_indicator(
        PackingFamilyId::UnsignedIncMsb,
        1,
        2,
        1,
    ));
    Some(requirements)
}

pub fn advice_bytes_validity_requirement(kind: JoltAdviceKind) -> PackingValidityRequirement {
    byte_validity_requirement(
        PackingFamilyId::AdviceBytes {
            kind: packing_advice_kind(kind),
            index: 0,
        },
        1,
    )
}

pub fn program_image_validity_requirement() -> PackingValidityRequirement {
    byte_validity_requirement(PackingFamilyId::ProgramImageInit, 8)
}

pub fn bytecode_validity_requirements(
    chunk: usize,
    field_byte_width: usize,
) -> Vec<PackingValidityRequirement> {
    let register_count = 1usize << REGISTER_ADDRESS_BITS;
    let mut requirements = Vec::new();
    for selector in 0..3 {
        requirements.push(PackingValidityRequirement::optional_one_hot(
            PackingFamilyId::BytecodeRegisterSelector { chunk, selector },
            1,
            register_count,
        ));
    }
    for flag in 0..NUM_CIRCUIT_FLAGS {
        requirements.push(PackingValidityRequirement::boolean_indicator(
            PackingFamilyId::BytecodeCircuitFlag { chunk, flag },
            1,
            2,
            1,
        ));
    }
    for flag in 0..NUM_INSTRUCTION_FLAGS {
        requirements.push(PackingValidityRequirement::boolean_indicator(
            PackingFamilyId::BytecodeInstructionFlag { chunk, flag },
            1,
            2,
            1,
        ));
    }
    requirements.push(PackingValidityRequirement::optional_one_hot(
        PackingFamilyId::BytecodeLookupSelector { chunk },
        1,
        LookupTableKind::<XLEN>::COUNT.next_power_of_two(),
    ));
    requirements.push(PackingValidityRequirement::boolean_indicator(
        PackingFamilyId::BytecodeRafFlag { chunk },
        1,
        2,
        1,
    ));
    requirements.push(byte_validity_requirement(
        PackingFamilyId::BytecodeUnexpandedPcBytes { chunk },
        8,
    ));
    requirements.push(byte_validity_requirement(
        PackingFamilyId::BytecodeImmBytes { chunk },
        field_byte_width,
    ));
    requirements.push(PackingValidityRequirement::bytecode_store_rd_disjoint(
        chunk,
        CircuitFlags::Store as usize,
    ));
    requirements
}

pub fn bytecode_imm_canonical_bytes_requirement(
    chunk: usize,
    byte_width: usize,
    modulus: u128,
) -> PackingValidityRequirement {
    PackingValidityRequirement::field_element_canonical_bytes(
        PackingFamilyId::BytecodeImmBytes { chunk },
        byte_width,
        modulus,
    )
}

fn byte_validity_requirement(family: PackingFamilyId, limbs: usize) -> PackingValidityRequirement {
    PackingValidityRequirement::exact_one_hot(family, limbs, 256)
}

fn packing_advice_kind(kind: JoltAdviceKind) -> PackingAdviceKind {
    match kind {
        JoltAdviceKind::Trusted => PackingAdviceKind::Trusted,
        JoltAdviceKind::Untrusted => PackingAdviceKind::Untrusted,
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LatticeFinalOpeningRequirement {
    PackingLayoutFamily {
        family: PackingFamilyId,
        relation: JoltRelationId,
    },
    LogicalOnly,
}

pub fn final_opening_lattice_requirement(
    polynomial: JoltCommittedPolynomial,
) -> LatticeFinalOpeningRequirement {
    match polynomial {
        JoltCommittedPolynomial::RdInc | JoltCommittedPolynomial::RamInc => {
            LatticeFinalOpeningRequirement::LogicalOnly
        }
        JoltCommittedPolynomial::InstructionRa(index) => packed_family_requirement(
            PackingFamilyId::InstructionRa { index },
            JoltRelationId::HammingWeightClaimReduction,
        ),
        JoltCommittedPolynomial::BytecodeRa(index) => packed_family_requirement(
            PackingFamilyId::BytecodeRa { index },
            JoltRelationId::HammingWeightClaimReduction,
        ),
        JoltCommittedPolynomial::RamRa(index) => packed_family_requirement(
            PackingFamilyId::RamRa { index },
            JoltRelationId::HammingWeightClaimReduction,
        ),
        JoltCommittedPolynomial::TrustedAdvice => packed_family_requirement(
            PackingFamilyId::AdviceBytes {
                kind: PackingAdviceKind::Trusted,
                index: 0,
            },
            JoltRelationId::AdviceClaimReduction,
        ),
        JoltCommittedPolynomial::UntrustedAdvice => packed_family_requirement(
            PackingFamilyId::AdviceBytes {
                kind: PackingAdviceKind::Untrusted,
                index: 0,
            },
            JoltRelationId::AdviceClaimReduction,
        ),
        JoltCommittedPolynomial::BytecodeChunk(index) => packed_family_requirement(
            PackingFamilyId::BytecodeChunk { index },
            JoltRelationId::BytecodeClaimReduction,
        ),
        JoltCommittedPolynomial::ProgramImageInit => packed_family_requirement(
            PackingFamilyId::ProgramImageInit,
            JoltRelationId::ProgramImageClaimReduction,
        ),
    }
}

fn packed_family_requirement(
    family: PackingFamilyId,
    relation: JoltRelationId,
) -> LatticeFinalOpeningRequirement {
    LatticeFinalOpeningRequirement::PackingLayoutFamily { family, relation }
}

pub fn byte_decode_terms<F: Field>(
    family: PackingFamilyId,
    limb: usize,
) -> Vec<PackingViewTerm<F>> {
    weighted_byte_decode_terms(family, [(limb, F::one())])
}

pub fn symbol_decode_terms<F: Field>(
    family: PackingFamilyId,
    limb: usize,
    alphabet_size: usize,
) -> Vec<PackingViewTerm<F>> {
    weighted_symbol_terms(
        family,
        limb,
        (0..alphabet_size).map(|symbol| F::from_u64(symbol as u64)),
    )
}

pub fn weighted_symbol_terms<F>(
    family: PackingFamilyId,
    limb: usize,
    weights: impl IntoIterator<Item = F>,
) -> Vec<PackingViewTerm<F>> {
    weights
        .into_iter()
        .enumerate()
        .map(|(symbol, coefficient)| {
            PackingViewTerm::new(coefficient, family.clone(), limb, symbol)
        })
        .collect()
}

pub fn weighted_byte_decode_terms<F: Field>(
    family: PackingFamilyId,
    limb_weights: impl IntoIterator<Item = (usize, F)>,
) -> Vec<PackingViewTerm<F>> {
    limb_weights
        .into_iter()
        .flat_map(|(limb, limb_weight)| {
            let family = family.clone();
            (0..256).map(move |symbol| {
                PackingViewTerm::new(
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
    family: PackingFamilyId,
    limb_count: usize,
) -> Vec<PackingViewTerm<F>> {
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
) -> Result<PackingViewFormula<F>, JoltFormulaPointError> {
    let lane_vars = bytecode_reduction::committed_lane_vars();
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
    let lane_layout = bytecode_reduction::BYTECODE_LANE_LAYOUT;
    let register_count = 1usize << REGISTER_ADDRESS_BITS;
    let mut terms = Vec::new();

    for selector in 0..3 {
        let start = match selector {
            0 => lane_layout.rs1_start,
            1 => lane_layout.rs2_start,
            _ => lane_layout.rd_start,
        };
        terms.extend(weighted_symbol_terms(
            PackingFamilyId::BytecodeRegisterSelector { chunk, selector },
            0,
            lane_weights[start..start + register_count].iter().copied(),
        ));
    }
    terms.extend(weighted_byte_decode_terms(
        PackingFamilyId::BytecodeUnexpandedPcBytes { chunk },
        byte_limb_weights(lane_weights[lane_layout.unexp_pc_idx], 8),
    ));
    terms.extend(weighted_byte_decode_terms(
        PackingFamilyId::BytecodeImmBytes { chunk },
        byte_limb_weights(lane_weights[lane_layout.imm_idx], field_byte_width),
    ));
    for flag in 0..NUM_CIRCUIT_FLAGS {
        terms.push(PackingViewTerm::new(
            lane_weights[lane_layout.circuit_start + flag],
            PackingFamilyId::BytecodeCircuitFlag { chunk, flag },
            0,
            1,
        ));
    }
    for flag in 0..NUM_INSTRUCTION_FLAGS {
        terms.push(PackingViewTerm::new(
            lane_weights[lane_layout.instr_start + flag],
            PackingFamilyId::BytecodeInstructionFlag { chunk, flag },
            0,
            1,
        ));
    }
    terms.extend(weighted_symbol_terms(
        PackingFamilyId::BytecodeLookupSelector { chunk },
        0,
        lane_weights
            [lane_layout.lookup_start..lane_layout.lookup_start + LookupTableKind::<XLEN>::COUNT]
            .iter()
            .copied(),
    ));
    terms.push(PackingViewTerm::new(
        lane_weights[lane_layout.raf_flag_idx],
        PackingFamilyId::BytecodeRafFlag { chunk },
        0,
        1,
    ));

    Ok(PackingViewFormula::linear_decoded(terms))
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
        clippy::expect_used,
        clippy::panic,
        clippy::unwrap_used,
        reason = "tests fail loudly on unexpected errors"
    )]

    use super::*;
    use jolt_field::{Fr, FromPrimitiveInt};

    #[test]
    fn final_opening_lattice_requirement_marks_increments_as_logical_only() {
        assert_eq!(
            final_opening_lattice_requirement(JoltCommittedPolynomial::RamInc),
            LatticeFinalOpeningRequirement::LogicalOnly
        );
        assert_eq!(
            final_opening_lattice_requirement(JoltCommittedPolynomial::RdInc),
            LatticeFinalOpeningRequirement::LogicalOnly
        );
    }

    #[test]
    fn final_opening_lattice_requirement_names_packed_families() {
        assert_eq!(
            final_opening_lattice_requirement(JoltCommittedPolynomial::InstructionRa(2)),
            LatticeFinalOpeningRequirement::PackingLayoutFamily {
                family: PackingFamilyId::InstructionRa { index: 2 },
                relation: JoltRelationId::HammingWeightClaimReduction,
            }
        );
        assert_eq!(
            final_opening_lattice_requirement(JoltCommittedPolynomial::ProgramImageInit),
            LatticeFinalOpeningRequirement::PackingLayoutFamily {
                family: PackingFamilyId::ProgramImageInit,
                relation: JoltRelationId::ProgramImageClaimReduction,
            }
        );
    }

    #[test]
    fn inc_virtualization_claim_exposes_expected_dependencies() {
        let claims = inc_virtualization_claim::<Fr>(TraceDimensions::new(5));

        assert_eq!(claims.id, JoltRelationId::IncVirtualization);
        assert_eq!(claims.sumcheck, TraceDimensions::new(5).sumcheck(3));
        assert_eq!(
            claims.input.required_openings,
            inc_virtualization_input_openings()
        );
        assert_eq!(
            claims.output.required_openings,
            inc_virtualization_output_openings()
        );
        assert_eq!(
            claims.required_challenges(),
            vec![JoltChallengeId::from(IncVirtualizationChallenge::Gamma)]
        );
        assert_eq!(
            claims.required_publics(),
            vec![
                JoltPublicId::from(IncVirtualizationPublic::EqRamReadWrite),
                JoltPublicId::from(IncVirtualizationPublic::EqRamValCheck),
                JoltPublicId::from(IncVirtualizationPublic::EqRegistersReadWrite),
                JoltPublicId::from(IncVirtualizationPublic::EqRegistersValEvaluation),
            ]
        );
    }

    #[test]
    fn inc_virtualization_claim_evaluates_store_selected_inc() {
        let claims = inc_virtualization_claim::<Fr>(TraceDimensions::new(5));
        let ram_rw = Fr::from_u64(3);
        let ram_val = Fr::from_u64(5);
        let rd_rw = Fr::from_u64(7);
        let rd_val = Fr::from_u64(11);
        let inc = Fr::from_u64(13);
        let store = Fr::from_u64(0);
        let eq_ram_rw = Fr::from_u64(17);
        let eq_ram_val = Fr::from_u64(19);
        let eq_rd_rw = Fr::from_u64(23);
        let eq_rd_val = Fr::from_u64(29);
        let gamma = Fr::from_u64(31);
        let zero = Fr::from_u64(0);

        let input = claims.input.expression().evaluate(
            |id| match *id {
                id if id == inc_virtualization_ram_read_write_opening() => ram_rw,
                id if id == inc_virtualization_ram_val_check_opening() => ram_val,
                id if id == inc_virtualization_rd_read_write_opening() => rd_rw,
                id if id == inc_virtualization_rd_val_evaluation_opening() => rd_val,
                _ => zero,
            },
            |id| match id {
                JoltChallengeId::IncVirtualization(IncVirtualizationChallenge::Gamma) => gamma,
                _ => zero,
            },
            |_| zero,
        );
        let output = claims.output.expression().evaluate(
            |id| match *id {
                id if id == inc_virtualization_inc_opening() => inc,
                id if id == inc_virtualization_store_opening() => store,
                _ => zero,
            },
            |id| match id {
                JoltChallengeId::IncVirtualization(IncVirtualizationChallenge::Gamma) => gamma,
                _ => zero,
            },
            |id| match *id {
                JoltPublicId::IncVirtualization(IncVirtualizationPublic::EqRamReadWrite) => {
                    eq_ram_rw
                }
                JoltPublicId::IncVirtualization(IncVirtualizationPublic::EqRamValCheck) => {
                    eq_ram_val
                }
                JoltPublicId::IncVirtualization(IncVirtualizationPublic::EqRegistersReadWrite) => {
                    eq_rd_rw
                }
                JoltPublicId::IncVirtualization(
                    IncVirtualizationPublic::EqRegistersValEvaluation,
                ) => eq_rd_val,
                _ => zero,
            },
        );

        let gamma_2 = gamma * gamma;
        assert_eq!(
            input,
            ram_rw + gamma * ram_val + gamma_2 * rd_rw + gamma_2 * gamma * rd_val
        );
        assert_eq!(output, inc * gamma_2 * (eq_rd_rw + gamma * eq_rd_val));

        let store_output = claims.output.expression().evaluate(
            |id| match *id {
                id if id == inc_virtualization_inc_opening() => inc,
                id if id == inc_virtualization_store_opening() => Fr::from_u64(1),
                _ => zero,
            },
            |id| match id {
                JoltChallengeId::IncVirtualization(IncVirtualizationChallenge::Gamma) => gamma,
                _ => zero,
            },
            |id| match *id {
                JoltPublicId::IncVirtualization(IncVirtualizationPublic::EqRamReadWrite) => {
                    eq_ram_rw
                }
                JoltPublicId::IncVirtualization(IncVirtualizationPublic::EqRamValCheck) => {
                    eq_ram_val
                }
                JoltPublicId::IncVirtualization(IncVirtualizationPublic::EqRegistersReadWrite) => {
                    eq_rd_rw
                }
                JoltPublicId::IncVirtualization(
                    IncVirtualizationPublic::EqRegistersValEvaluation,
                ) => eq_rd_val,
                _ => zero,
            },
        );
        assert_eq!(store_output, inc * (eq_ram_rw + gamma * eq_ram_val));
    }

    #[test]
    fn unsigned_inc_claim_reduction_offsets_inc_by_two_to_64() {
        let claims = unsigned_inc_claim_reduction_claim::<Fr>(TraceDimensions::new(5));
        let inc = Fr::from_u64(13);
        let unsigned_inc = Fr::from_u128((1u128 << 64) + 13);
        let zero = Fr::from_u64(0);

        assert_eq!(claims.id, JoltRelationId::UnsignedIncClaimReduction);
        assert_eq!(claims.sumcheck, TraceDimensions::new(5).sumcheck(2));
        assert_eq!(
            claims.input.required_openings,
            vec![inc_virtualization_inc_opening()]
        );
        assert_eq!(
            claims.output.required_openings,
            vec![unsigned_inc_opening()]
        );
        assert!(claims.required_challenges().is_empty());
        assert!(claims.required_publics().is_empty());

        let input = claims.input.expression().evaluate(
            |id| match *id {
                id if id == inc_virtualization_inc_opening() => inc,
                _ => zero,
            },
            |_| zero,
            |_| zero,
        );
        let output = claims.output.expression().evaluate(
            |id| match *id {
                id if id == unsigned_inc_opening() => unsigned_inc,
                _ => zero,
            },
            |_| zero,
            |_| zero,
        );

        assert_eq!(input, unsigned_inc);
        assert_eq!(output, unsigned_inc);
    }

    #[test]
    fn unsigned_inc_msb_booleanity_claim_checks_cycle_bit() {
        let claims = unsigned_inc_msb_booleanity_claim::<Fr>(TraceDimensions::new(5));
        let zero = Fr::from_u64(0);
        let msb = Fr::from_u64(7);

        assert_eq!(claims.id, JoltRelationId::Booleanity);
        assert_eq!(claims.sumcheck, TraceDimensions::new(5).sumcheck(2));
        assert!(claims.input.required_openings.is_empty());
        assert_eq!(
            claims.output.required_openings,
            vec![unsigned_inc_msb_opening()]
        );
        assert!(claims.required_challenges().is_empty());
        assert!(claims.required_publics().is_empty());

        let output = claims.output.expression().evaluate(
            |id| match *id {
                id if id == unsigned_inc_msb_opening() => msb,
                _ => zero,
            },
            |_| zero,
            |_| zero,
        );

        assert_eq!(output, msb * msb - msb);
    }

    #[test]
    fn unsigned_inc_chunk_reconstruction_claim_batches_hamming_point_and_value() {
        let claims = unsigned_inc_chunk_reconstruction_claim::<Fr>(8)
            .expect("8-bit chunking should be valid");

        assert_eq!(claims.id, JoltRelationId::UnsignedIncChunkReconstruction);
        assert_eq!(claims.sumcheck, JoltSumcheckSpec::boolean(8, 3));
        assert!(claims
            .input
            .required_openings
            .contains(&unsigned_inc_opening()));
        assert!(claims
            .input
            .required_openings
            .contains(&unsigned_inc_msb_opening()));
        assert_eq!(
            claims.output.required_openings,
            (0..8).map(unsigned_inc_chunk_opening).collect::<Vec<_>>()
        );
        assert_eq!(
            claims.required_challenges(),
            vec![JoltChallengeId::from(
                UnsignedIncChunkReconstructionChallenge::Gamma,
            )]
        );
        assert_eq!(
            claims.required_publics(),
            vec![
                JoltPublicId::from(UnsignedIncChunkReconstructionPublic::EqBooleanityAddress),
                JoltPublicId::from(UnsignedIncChunkReconstructionPublic::IdentityAtAddress),
            ]
        );
    }

    #[test]
    fn unsigned_inc_decode_formulas_use_configured_chunks_and_msb() {
        assert_eq!(
            unsigned_inc_msb_lattice_view_formula::<Fr>(),
            PackingViewFormula::direct(PackingFamilyId::UnsignedIncMsb, 0, 1)
        );

        let lower = unsigned_inc_lower_value_lattice_view_formula::<Fr>(4)
            .expect("4-bit chunking should be valid");
        let terms = linear_decoded_terms(&lower);
        assert_eq!(terms.len(), 16 * 16);
        assert_eq!(
            find_term(terms, PackingFamilyId::UnsignedIncChunk { index: 0 }, 0, 7).coefficient,
            Fr::from_u64(7)
        );
        assert_eq!(
            find_term(terms, PackingFamilyId::UnsignedIncChunk { index: 1 }, 0, 3).coefficient,
            Fr::from_u64(16 * 3)
        );
        assert_eq!(
            find_term(terms, PackingFamilyId::UnsignedIncChunk { index: 15 }, 0, 2).coefficient,
            Fr::from_u64(1u64 << 60) * Fr::from_u64(2)
        );
    }

    #[test]
    fn unsigned_inc_validity_requirements_cover_chunks_and_msb() {
        let requirements =
            unsigned_inc_validity_requirements(4).expect("4-bit chunking should be valid");

        assert_eq!(requirements.len(), 17);
        assert_eq!(
            requirements[0],
            PackingValidityRequirement::exact_one_hot(
                PackingFamilyId::UnsignedIncChunk { index: 0 },
                1,
                16,
            )
        );
        assert_eq!(
            requirements[15],
            PackingValidityRequirement::exact_one_hot(
                PackingFamilyId::UnsignedIncChunk { index: 15 },
                1,
                16,
            )
        );
        assert_eq!(
            requirements[16],
            PackingValidityRequirement::boolean_indicator(PackingFamilyId::UnsignedIncMsb, 1, 2, 1,)
        );
    }

    #[test]
    fn packed_validity_digest_is_order_stable_and_kind_sensitive() {
        let exact = PackingValidityRequirement::exact_one_hot(
            PackingFamilyId::UnsignedIncChunk { index: 0 },
            1,
            256,
        );
        let msb =
            PackingValidityRequirement::boolean_indicator(PackingFamilyId::UnsignedIncMsb, 1, 2, 1);
        let optional = PackingValidityRequirement::optional_one_hot(
            PackingFamilyId::UnsignedIncChunk { index: 0 },
            1,
            256,
        );

        assert_eq!(
            packing_validity_digest(&[exact.clone(), msb.clone()]),
            packing_validity_digest(&[msb, exact.clone()])
        );
        assert_ne!(
            packing_validity_digest(&[exact]),
            packing_validity_digest(&[optional])
        );
    }

    #[test]
    fn bytecode_validity_requirements_cover_committed_program_facts() {
        let chunk = 2;
        let field_byte_width = 16;
        let requirements = bytecode_validity_requirements(chunk, field_byte_width);

        assert_eq!(
            requirements.len(),
            3 + NUM_CIRCUIT_FLAGS + NUM_INSTRUCTION_FLAGS + 5
        );
        assert!(
            requirements.contains(&PackingValidityRequirement::optional_one_hot(
                PackingFamilyId::BytecodeRegisterSelector { chunk, selector: 2 },
                1,
                1 << REGISTER_ADDRESS_BITS,
            ))
        );
        assert!(
            requirements.contains(&PackingValidityRequirement::boolean_indicator(
                PackingFamilyId::BytecodeCircuitFlag { chunk, flag: 0 },
                1,
                2,
                1,
            ))
        );
        assert!(
            requirements.contains(&PackingValidityRequirement::boolean_indicator(
                PackingFamilyId::BytecodeInstructionFlag {
                    chunk,
                    flag: NUM_INSTRUCTION_FLAGS - 1,
                },
                1,
                2,
                1,
            ))
        );
        assert!(
            requirements.contains(&PackingValidityRequirement::optional_one_hot(
                PackingFamilyId::BytecodeLookupSelector { chunk },
                1,
                LookupTableKind::<XLEN>::COUNT.next_power_of_two(),
            ))
        );
        assert!(
            requirements.contains(&PackingValidityRequirement::boolean_indicator(
                PackingFamilyId::BytecodeRafFlag { chunk },
                1,
                2,
                1,
            ))
        );
        assert!(
            requirements.contains(&PackingValidityRequirement::exact_one_hot(
                PackingFamilyId::BytecodeUnexpandedPcBytes { chunk },
                8,
                256,
            ))
        );
        assert!(
            requirements.contains(&PackingValidityRequirement::exact_one_hot(
                PackingFamilyId::BytecodeImmBytes { chunk },
                field_byte_width,
                256,
            ))
        );
        assert!(
            requirements.contains(&PackingValidityRequirement::bytecode_store_rd_disjoint(
                chunk,
                CircuitFlags::Store as usize
            ))
        );
    }

    #[test]
    fn bytecode_imm_canonical_bytes_requirement_anchors_bytecode_immediates() {
        assert_eq!(
            bytecode_imm_canonical_bytes_requirement(2, 16, 97),
            PackingValidityRequirement::field_element_canonical_bytes(
                PackingFamilyId::BytecodeImmBytes { chunk: 2 },
                16,
                97,
            )
        );
    }

    #[test]
    fn advice_and_program_image_validity_requirements_are_byte_facts() {
        assert_eq!(
            advice_bytes_validity_requirement(JoltAdviceKind::Trusted),
            PackingValidityRequirement::exact_one_hot(
                PackingFamilyId::AdviceBytes {
                    kind: PackingAdviceKind::Trusted,
                    index: 0,
                },
                1,
                256,
            )
        );
        assert_eq!(
            program_image_validity_requirement(),
            PackingValidityRequirement::exact_one_hot(PackingFamilyId::ProgramImageInit, 8, 256,)
        );
    }

    #[test]
    fn byte_decode_terms_are_little_endian_symbol_weights() {
        let terms = byte_decode_terms::<Fr>(PackingFamilyId::BytecodeChunk { index: 0 }, 3);

        assert_eq!(terms.len(), 256);
        assert_eq!(terms[7].coefficient, Fr::from_u64(7));
        assert_eq!(terms[7].family, PackingFamilyId::BytecodeChunk { index: 0 });
        assert_eq!(terms[7].limb, 3);
        assert_eq!(terms[7].symbol, 7);
    }

    #[test]
    fn committed_bytecode_lattice_family_ids_name_lane_classes() {
        assert_ne!(
            PackingFamilyId::BytecodeRegisterSelector {
                chunk: 0,
                selector: 0,
            },
            PackingFamilyId::BytecodeRegisterSelector {
                chunk: 0,
                selector: 1,
            }
        );
        assert_ne!(
            PackingFamilyId::BytecodeUnexpandedPcBytes { chunk: 0 },
            PackingFamilyId::BytecodeImmBytes { chunk: 0 }
        );
        assert_ne!(
            PackingFamilyId::BytecodeCircuitFlag { chunk: 0, flag: 0 },
            PackingFamilyId::BytecodeInstructionFlag { chunk: 0, flag: 0 }
        );
    }

    #[test]
    fn bytecode_chunk_lattice_view_formula_uses_cycle_major_lane_weights() {
        let lane_vars = bytecode_reduction::committed_lane_vars();
        let log_bytecode = 2;
        let lane_point = (1..=lane_vars as u64).map(Fr::from_u64).collect::<Vec<_>>();
        let mut opening_point = lane_point.clone();
        opening_point.extend([Fr::from_u64(101), Fr::from_u64(103)]);
        let lane_weights = EqPolynomial::<Fr>::evals(&lane_point, None);
        let lane_layout = bytecode_reduction::BYTECODE_LANE_LAYOUT;

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
                PackingFamilyId::BytecodeRegisterSelector {
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
                PackingFamilyId::BytecodeCircuitFlag { chunk: 2, flag: 0 },
                0,
                1
            )
            .coefficient,
            lane_weights[lane_layout.circuit_start]
        );
        assert_eq!(
            find_term(
                terms,
                PackingFamilyId::BytecodeLookupSelector { chunk: 2 },
                0,
                3
            )
            .coefficient,
            lane_weights[lane_layout.lookup_start + 3]
        );
        assert_eq!(
            find_term(
                terms,
                PackingFamilyId::BytecodeUnexpandedPcBytes { chunk: 2 },
                1,
                7
            )
            .coefficient,
            lane_weights[lane_layout.unexp_pc_idx] * Fr::from_u64(256 * 7)
        );
        assert_eq!(
            find_term(terms, PackingFamilyId::BytecodeImmBytes { chunk: 2 }, 1, 9).coefficient,
            lane_weights[lane_layout.imm_idx] * Fr::from_u64(256 * 9)
        );
        assert_eq!(
            find_term(terms, PackingFamilyId::BytecodeRafFlag { chunk: 2 }, 0, 1).coefficient,
            lane_weights[lane_layout.raf_flag_idx]
        );
    }

    #[test]
    fn bytecode_chunk_lattice_view_formula_uses_address_major_lane_suffix() {
        let lane_vars = bytecode_reduction::committed_lane_vars();
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
                PackingFamilyId::BytecodeRegisterSelector {
                    chunk: 1,
                    selector: 0
                },
                0,
                1
            )
            .coefficient,
            lane_weights[bytecode_reduction::BYTECODE_LANE_LAYOUT.rs1_start + 1]
        );
    }

    #[test]
    fn bytecode_chunk_lattice_view_formula_rejects_bad_point_length() {
        let expected = bytecode_reduction::committed_lane_vars() + 3;

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
        let terms = symbol_decode_terms::<Fr>(PackingFamilyId::RamRa { index: 1 }, 0, 4);

        assert_eq!(terms.len(), 4);
        assert_eq!(terms[3].coefficient, Fr::from_u64(3));
        assert_eq!(terms[3].family, PackingFamilyId::RamRa { index: 1 });
        assert_eq!(terms[3].limb, 0);
        assert_eq!(terms[3].symbol, 3);
    }

    #[test]
    fn weighted_symbol_terms_use_supplied_coefficients() {
        let terms = weighted_symbol_terms(
            PackingFamilyId::InstructionRa { index: 0 },
            2,
            [Fr::from_u64(11), Fr::from_u64(13), Fr::from_u64(17)],
        );

        assert_eq!(terms.len(), 3);
        assert_eq!(terms[1].coefficient, Fr::from_u64(13));
        assert_eq!(terms[1].family, PackingFamilyId::InstructionRa { index: 0 });
        assert_eq!(terms[1].limb, 2);
        assert_eq!(terms[1].symbol, 1);
    }

    #[test]
    fn weighted_byte_decode_terms_scale_symbols_by_limb_weights() {
        let terms = weighted_byte_decode_terms(
            PackingFamilyId::BytecodeChunk { index: 2 },
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
            PackingFamilyId::BytecodeChunk { index: 2 }
        );
    }

    #[test]
    fn little_endian_byte_decode_terms_weight_limbs_by_place_value() {
        let terms = little_endian_byte_decode_terms::<Fr>(PackingFamilyId::ProgramImageInit, 2);

        assert_eq!(terms.len(), 512);
        assert_eq!(terms[7].coefficient, Fr::from_u64(7));
        assert_eq!(terms[7].limb, 0);
        assert_eq!(terms[7].symbol, 7);
        assert_eq!(terms[256 + 7].coefficient, Fr::from_u64(256 * 7));
        assert_eq!(terms[256 + 7].limb, 1);
        assert_eq!(terms[256 + 7].symbol, 7);
        assert_eq!(terms[256 + 7].family, PackingFamilyId::ProgramImageInit);
    }

    fn linear_decoded_terms(formula: &PackingViewFormula<Fr>) -> &[PackingViewTerm<Fr>] {
        match formula {
            PackingViewFormula::LinearDecoded { terms, .. } => terms,
            _ => panic!("expected linear decoded formula"),
        }
    }

    fn find_term(
        terms: &[PackingViewTerm<Fr>],
        family: PackingFamilyId,
        limb: usize,
        symbol: usize,
    ) -> &PackingViewTerm<Fr> {
        terms
            .iter()
            .find(|term| term.family == family && term.limb == limb && term.symbol == symbol)
            .unwrap()
    }
}
