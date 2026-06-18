use blake2::digest::consts::U32;
use blake2::{Blake2b, Digest};
use jolt_field::{Field, RingCore};
use jolt_lookup_tables::{LookupTableKind, XLEN};
use jolt_poly::EqPolynomial;
use jolt_riscv::{NUM_CIRCUIT_FLAGS, NUM_INSTRUCTION_FLAGS};
use serde::{Deserialize, Serialize};

use crate::{challenge, opening};

use super::super::{
    FusedIncrementInactiveSourceLinkChallenge, FusedIncrementInactiveZeroChallenge,
    FusedIncrementSourceLinkChallenge, FusedIncrementTranslationChallenge, JoltChallengeId,
    JoltExpr, JoltRelationClaims,
};
use super::super::{JoltAdviceKind, JoltCommittedPolynomial, JoltOpeningId, JoltRelationId};
use super::bytecode::BytecodeReadRafDimensions;
use super::claim_reductions::bytecode as bytecode_reduction;
use super::dimensions::{
    JoltFormulaPointError, TraceDimensions, TracePolynomialOrder, REGISTER_ADDRESS_BITS,
};
use jolt_riscv::CircuitFlags;

pub const FUSED_INCREMENT_BYTE_LIMBS: usize = 8;

#[derive(Hash, PartialEq, Eq, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum LatticePackedFamilyId {
    InstructionRa { index: usize },
    BytecodeRa { index: usize },
    RamRa { index: usize },
    IncByte { index: usize },
    IncSign,
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
        terms: Vec<LatticePackedViewTerm<F>>,
    },
    MaskedDecoded {
        relation: JoltRelationId,
    },
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

    pub fn reduced_masked(relation: JoltRelationId, terms: Vec<LatticePackedViewTerm<F>>) -> Self {
        Self::ReducedMasked { relation, terms }
    }

    pub fn masked_decoded(relation: JoltRelationId) -> Self {
        Self::MaskedDecoded { relation }
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
pub enum LatticePackedValidityKind {
    ExactOneHot,
    OptionalOneHot,
    BooleanIndicator { symbol: usize },
    FusedIncrementCanonicalZero,
    BytecodeStoreRdDisjoint,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LatticePackedValidityRequirement {
    pub family: LatticePackedFamilyId,
    pub limbs: usize,
    pub alphabet_size: usize,
    pub kind: LatticePackedValidityKind,
}

impl LatticePackedValidityRequirement {
    pub fn exact_one_hot(
        family: LatticePackedFamilyId,
        limbs: usize,
        alphabet_size: usize,
    ) -> Self {
        Self {
            family,
            limbs,
            alphabet_size,
            kind: LatticePackedValidityKind::ExactOneHot,
        }
    }

    pub fn optional_one_hot(
        family: LatticePackedFamilyId,
        limbs: usize,
        alphabet_size: usize,
    ) -> Self {
        Self {
            family,
            limbs,
            alphabet_size,
            kind: LatticePackedValidityKind::OptionalOneHot,
        }
    }

    pub fn boolean_indicator(
        family: LatticePackedFamilyId,
        limbs: usize,
        alphabet_size: usize,
        symbol: usize,
    ) -> Self {
        Self {
            family,
            limbs,
            alphabet_size,
            kind: LatticePackedValidityKind::BooleanIndicator { symbol },
        }
    }

    pub fn fused_increment_canonical_zero() -> Self {
        Self {
            family: LatticePackedFamilyId::IncSign,
            limbs: 1,
            alphabet_size: 2,
            kind: LatticePackedValidityKind::FusedIncrementCanonicalZero,
        }
    }

    pub fn bytecode_store_rd_disjoint(chunk: usize) -> Self {
        Self {
            family: LatticePackedFamilyId::BytecodeCircuitFlag {
                chunk,
                flag: CircuitFlags::Store as usize,
            },
            limbs: 1,
            alphabet_size: 2,
            kind: LatticePackedValidityKind::BytecodeStoreRdDisjoint,
        }
    }
}

pub type LatticePackedValidityDigest = [u8; 32];

pub fn lattice_packed_validity_digest(
    requirements: &[LatticePackedValidityRequirement],
) -> LatticePackedValidityDigest {
    let mut encoded_requirements = requirements
        .iter()
        .map(encode_validity_requirement)
        .collect::<Vec<_>>();
    encoded_requirements.sort();

    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"jolt-claims/lattice-packed-validity/v1");
    write_usize(&mut bytes, encoded_requirements.len());
    for requirement in encoded_requirements {
        bytes.extend_from_slice(&requirement);
    }

    let mut hasher = Blake2b::<U32>::new();
    hasher.update(&bytes);
    let result = hasher.finalize();
    let mut digest = [0u8; 32];
    digest.copy_from_slice(&result);
    digest
}

fn encode_validity_requirement(requirement: &LatticePackedValidityRequirement) -> Vec<u8> {
    let mut bytes = Vec::new();
    write_family_id(&mut bytes, &requirement.family);
    write_usize(&mut bytes, requirement.limbs);
    write_usize(&mut bytes, requirement.alphabet_size);
    write_validity_kind(&mut bytes, &requirement.kind);
    bytes
}

fn write_validity_kind(bytes: &mut Vec<u8>, kind: &LatticePackedValidityKind) {
    match kind {
        LatticePackedValidityKind::ExactOneHot => bytes.push(0),
        LatticePackedValidityKind::OptionalOneHot => bytes.push(1),
        LatticePackedValidityKind::BooleanIndicator { symbol } => {
            bytes.push(2);
            write_usize(bytes, *symbol);
        }
        LatticePackedValidityKind::FusedIncrementCanonicalZero => bytes.push(3),
        LatticePackedValidityKind::BytecodeStoreRdDisjoint => bytes.push(4),
    }
}

fn write_family_id(bytes: &mut Vec<u8>, id: &LatticePackedFamilyId) {
    match id {
        LatticePackedFamilyId::InstructionRa { index } => {
            bytes.push(0);
            write_usize(bytes, *index);
        }
        LatticePackedFamilyId::BytecodeRa { index } => {
            bytes.push(1);
            write_usize(bytes, *index);
        }
        LatticePackedFamilyId::RamRa { index } => {
            bytes.push(2);
            write_usize(bytes, *index);
        }
        LatticePackedFamilyId::IncByte { index } => {
            bytes.push(3);
            write_usize(bytes, *index);
        }
        LatticePackedFamilyId::IncSign => bytes.push(4),
        LatticePackedFamilyId::FieldRdIncByte { index } => {
            bytes.push(9);
            write_usize(bytes, *index);
        }
        LatticePackedFamilyId::FieldRdIncSign => bytes.push(10),
        LatticePackedFamilyId::AdviceBytes { kind, index } => {
            bytes.push(11);
            bytes.push(advice_kind_tag(*kind));
            write_usize(bytes, *index);
        }
        LatticePackedFamilyId::BytecodeChunk { index } => {
            bytes.push(12);
            write_usize(bytes, *index);
        }
        LatticePackedFamilyId::ProgramImageInit => bytes.push(13),
        LatticePackedFamilyId::BytecodeRegisterSelector { chunk, selector } => {
            bytes.push(15);
            write_usize(bytes, *chunk);
            write_usize(bytes, *selector);
        }
        LatticePackedFamilyId::BytecodeCircuitFlag { chunk, flag } => {
            bytes.push(16);
            write_usize(bytes, *chunk);
            write_usize(bytes, *flag);
        }
        LatticePackedFamilyId::BytecodeInstructionFlag { chunk, flag } => {
            bytes.push(17);
            write_usize(bytes, *chunk);
            write_usize(bytes, *flag);
        }
        LatticePackedFamilyId::BytecodeLookupSelector { chunk } => {
            bytes.push(18);
            write_usize(bytes, *chunk);
        }
        LatticePackedFamilyId::BytecodeRafFlag { chunk } => {
            bytes.push(19);
            write_usize(bytes, *chunk);
        }
        LatticePackedFamilyId::BytecodeUnexpandedPcBytes { chunk } => {
            bytes.push(20);
            write_usize(bytes, *chunk);
        }
        LatticePackedFamilyId::BytecodeImmBytes { chunk } => {
            bytes.push(21);
            write_usize(bytes, *chunk);
        }
        LatticePackedFamilyId::Custom { namespace, index } => {
            bytes.push(14);
            bytes.extend_from_slice(&namespace.to_le_bytes());
            write_usize(bytes, *index);
        }
    }
}

fn write_usize(bytes: &mut Vec<u8>, value: usize) {
    bytes.extend_from_slice(&(value as u64).to_le_bytes());
}

fn advice_kind_tag(kind: JoltAdviceKind) -> u8 {
    match kind {
        JoltAdviceKind::Trusted => 0,
        JoltAdviceKind::Untrusted => 1,
    }
}

#[derive(Hash, PartialEq, Eq, Copy, Clone, Debug, PartialOrd, Ord, Serialize, Deserialize)]
pub enum LatticeFusedIncrementTarget {
    Ram,
    Rd,
}

pub fn fused_increment_translation_relation() -> JoltRelationId {
    JoltRelationId::FusedIncrementTranslation
}

pub fn fused_increment_source_link_relation() -> JoltRelationId {
    JoltRelationId::FusedIncrementSourceLink
}

pub fn fused_increment_inactive_zero_relation() -> JoltRelationId {
    JoltRelationId::FusedIncrementInactiveZero
}

pub fn fused_increment_inactive_source_link_relation() -> JoltRelationId {
    JoltRelationId::FusedIncrementInactiveSourceLink
}

pub fn fused_increment_translation_claim<F>(dimensions: TraceDimensions) -> JoltRelationClaims<F>
where
    F: RingCore,
{
    let gamma = fused_increment_translation_challenge(FusedIncrementTranslationChallenge::Gamma);
    let input = opening(fused_increment_translation_input_opening(
        LatticeFusedIncrementTarget::Ram,
    )) + gamma.clone()
        * opening(fused_increment_translation_input_opening(
            LatticeFusedIncrementTarget::Rd,
        ));
    let output = signed_source_output(LatticeFusedIncrementTarget::Ram)
        + gamma * signed_source_output(LatticeFusedIncrementTarget::Rd);

    JoltRelationClaims::new(
        JoltRelationId::FusedIncrementTranslation,
        dimensions.sumcheck(4),
        input,
        output,
    )
}

pub fn fused_increment_source_link_claim<F>(
    dimensions: BytecodeReadRafDimensions,
) -> JoltRelationClaims<F>
where
    F: RingCore,
{
    let gamma = fused_increment_source_link_challenge(FusedIncrementSourceLinkChallenge::Gamma);
    let input = opening(fused_increment_source_opening(
        LatticeFusedIncrementTarget::Ram,
    )) + gamma.clone()
        * opening(fused_increment_source_opening(
            LatticeFusedIncrementTarget::Rd,
        ));
    let source_output = opening(fused_increment_bytecode_source_opening(
        LatticeFusedIncrementTarget::Ram,
    )) + gamma
        * opening(fused_increment_bytecode_source_opening(
            LatticeFusedIncrementTarget::Rd,
        ));
    let output = fused_increment_source_link_bytecode_ra_product(dimensions) * source_output;

    JoltRelationClaims::new(
        JoltRelationId::FusedIncrementSourceLink,
        dimensions.sumcheck(),
        input,
        output,
    )
    .with_input_challenges([JoltChallengeId::from(
        FusedIncrementSourceLinkChallenge::Gamma,
    )])
}

pub fn fused_increment_inactive_zero_claim<F>(dimensions: TraceDimensions) -> JoltRelationClaims<F>
where
    F: RingCore,
{
    let beta = fused_increment_inactive_zero_challenge(FusedIncrementInactiveZeroChallenge::Beta);
    let inactive = JoltExpr::one()
        - opening(fused_increment_inactive_source_opening(
            LatticeFusedIncrementTarget::Ram,
        ))
        - opening(fused_increment_inactive_source_opening(
            LatticeFusedIncrementTarget::Rd,
        ));
    let value = opening(fused_increment_inactive_magnitude_opening())
        + beta * opening(fused_increment_inactive_sign_opening());

    JoltRelationClaims::new(
        JoltRelationId::FusedIncrementInactiveZero,
        dimensions.sumcheck(3),
        JoltExpr::zero(),
        inactive * value,
    )
}

pub fn fused_increment_inactive_source_link_claim<F>(
    dimensions: BytecodeReadRafDimensions,
) -> JoltRelationClaims<F>
where
    F: RingCore,
{
    let gamma = fused_increment_inactive_source_link_challenge(
        FusedIncrementInactiveSourceLinkChallenge::Gamma,
    );
    let input = opening(fused_increment_inactive_source_opening(
        LatticeFusedIncrementTarget::Ram,
    )) + gamma.clone()
        * opening(fused_increment_inactive_source_opening(
            LatticeFusedIncrementTarget::Rd,
        ));
    let source_output = opening(fused_increment_inactive_bytecode_source_opening(
        LatticeFusedIncrementTarget::Ram,
    )) + gamma
        * opening(fused_increment_inactive_bytecode_source_opening(
            LatticeFusedIncrementTarget::Rd,
        ));
    let output =
        fused_increment_inactive_source_link_bytecode_ra_product(dimensions) * source_output;

    JoltRelationClaims::new(
        JoltRelationId::FusedIncrementInactiveSourceLink,
        dimensions.sumcheck(),
        input,
        output,
    )
    .with_input_challenges([JoltChallengeId::from(
        FusedIncrementInactiveSourceLinkChallenge::Gamma,
    )])
}

fn fused_increment_translation_challenge<F>(id: FusedIncrementTranslationChallenge) -> JoltExpr<F>
where
    F: RingCore,
{
    challenge(JoltChallengeId::from(id))
}

fn fused_increment_source_link_challenge<F>(id: FusedIncrementSourceLinkChallenge) -> JoltExpr<F>
where
    F: RingCore,
{
    challenge(JoltChallengeId::from(id))
}

fn fused_increment_inactive_zero_challenge<F>(
    id: FusedIncrementInactiveZeroChallenge,
) -> JoltExpr<F>
where
    F: RingCore,
{
    challenge(JoltChallengeId::from(id))
}

fn fused_increment_inactive_source_link_challenge<F>(
    id: FusedIncrementInactiveSourceLinkChallenge,
) -> JoltExpr<F>
where
    F: RingCore,
{
    challenge(JoltChallengeId::from(id))
}

pub fn fused_increment_translation_input_opening(
    target: LatticeFusedIncrementTarget,
) -> JoltOpeningId {
    let polynomial = match target {
        LatticeFusedIncrementTarget::Ram => JoltCommittedPolynomial::RamInc,
        LatticeFusedIncrementTarget::Rd => JoltCommittedPolynomial::RdInc,
    };
    JoltOpeningId::committed(polynomial, JoltRelationId::IncClaimReduction)
}

pub fn fused_increment_translation_output_openings() -> [JoltOpeningId; 4] {
    [
        fused_increment_source_opening(LatticeFusedIncrementTarget::Ram),
        fused_increment_magnitude_opening(),
        fused_increment_sign_opening(),
        fused_increment_source_opening(LatticeFusedIncrementTarget::Rd),
    ]
}

pub fn fused_increment_source_opening(target: LatticeFusedIncrementTarget) -> JoltOpeningId {
    JoltOpeningId::lattice(
        JoltRelationId::FusedIncrementTranslation,
        match target {
            LatticeFusedIncrementTarget::Ram => 0,
            LatticeFusedIncrementTarget::Rd => 1,
        },
    )
}

pub fn fused_increment_inactive_source_opening(
    target: LatticeFusedIncrementTarget,
) -> JoltOpeningId {
    JoltOpeningId::lattice(
        JoltRelationId::FusedIncrementInactiveZero,
        match target {
            LatticeFusedIncrementTarget::Ram => 0,
            LatticeFusedIncrementTarget::Rd => 1,
        },
    )
}

pub fn fused_increment_bytecode_source_opening(
    target: LatticeFusedIncrementTarget,
) -> JoltOpeningId {
    JoltOpeningId::lattice(
        JoltRelationId::FusedIncrementSourceLink,
        match target {
            LatticeFusedIncrementTarget::Ram => 0,
            LatticeFusedIncrementTarget::Rd => 1,
        },
    )
}

pub fn fused_increment_inactive_bytecode_source_opening(
    target: LatticeFusedIncrementTarget,
) -> JoltOpeningId {
    JoltOpeningId::lattice(
        JoltRelationId::FusedIncrementInactiveSourceLink,
        match target {
            LatticeFusedIncrementTarget::Ram => 0,
            LatticeFusedIncrementTarget::Rd => 1,
        },
    )
}

pub fn fused_increment_source_link_output_openings(
    dimensions: BytecodeReadRafDimensions,
) -> Vec<JoltOpeningId> {
    let mut openings = fused_increment_source_link_bytecode_ra_openings(dimensions);
    openings.extend([
        fused_increment_bytecode_source_opening(LatticeFusedIncrementTarget::Ram),
        fused_increment_bytecode_source_opening(LatticeFusedIncrementTarget::Rd),
    ]);
    openings
}

pub fn fused_increment_inactive_zero_output_openings() -> [JoltOpeningId; 4] {
    [
        fused_increment_inactive_magnitude_opening(),
        fused_increment_inactive_sign_opening(),
        fused_increment_inactive_source_opening(LatticeFusedIncrementTarget::Ram),
        fused_increment_inactive_source_opening(LatticeFusedIncrementTarget::Rd),
    ]
}

pub fn fused_increment_inactive_source_link_output_openings(
    dimensions: BytecodeReadRafDimensions,
) -> Vec<JoltOpeningId> {
    let mut openings = fused_increment_inactive_source_link_bytecode_ra_openings(dimensions);
    openings.extend([
        fused_increment_inactive_bytecode_source_opening(LatticeFusedIncrementTarget::Ram),
        fused_increment_inactive_bytecode_source_opening(LatticeFusedIncrementTarget::Rd),
    ]);
    openings
}

pub fn fused_increment_magnitude_opening() -> JoltOpeningId {
    JoltOpeningId::lattice(JoltRelationId::FusedIncrementTranslation, 2)
}

pub fn fused_increment_sign_opening() -> JoltOpeningId {
    JoltOpeningId::lattice(JoltRelationId::FusedIncrementTranslation, 3)
}

pub fn fused_increment_inactive_magnitude_opening() -> JoltOpeningId {
    JoltOpeningId::lattice(JoltRelationId::FusedIncrementInactiveZero, 2)
}

pub fn fused_increment_inactive_sign_opening() -> JoltOpeningId {
    JoltOpeningId::lattice(JoltRelationId::FusedIncrementInactiveZero, 3)
}

fn signed_source_output<F>(target: LatticeFusedIncrementTarget) -> JoltExpr<F>
where
    F: RingCore,
{
    let source = opening(fused_increment_source_opening(target));
    let magnitude = opening(fused_increment_magnitude_opening());
    let sign = opening(fused_increment_sign_opening());

    let source_magnitude = source.clone() * magnitude.clone();
    let sign_correction = source * sign * magnitude;
    source_magnitude - sign_correction.clone() - sign_correction
}

fn fused_increment_source_link_bytecode_ra_product<F>(
    dimensions: BytecodeReadRafDimensions,
) -> JoltExpr<F>
where
    F: RingCore,
{
    fused_increment_bytecode_ra_product(dimensions, JoltRelationId::FusedIncrementSourceLink)
}

fn fused_increment_inactive_source_link_bytecode_ra_product<F>(
    dimensions: BytecodeReadRafDimensions,
) -> JoltExpr<F>
where
    F: RingCore,
{
    fused_increment_bytecode_ra_product(
        dimensions,
        JoltRelationId::FusedIncrementInactiveSourceLink,
    )
}

fn fused_increment_bytecode_ra_product<F>(
    dimensions: BytecodeReadRafDimensions,
    relation: JoltRelationId,
) -> JoltExpr<F>
where
    F: RingCore,
{
    let mut product = JoltExpr::one();
    for opening_id in fused_increment_bytecode_ra_openings(dimensions, relation) {
        product = product * opening(opening_id);
    }
    product
}

fn fused_increment_source_link_bytecode_ra_openings(
    dimensions: BytecodeReadRafDimensions,
) -> Vec<JoltOpeningId> {
    fused_increment_bytecode_ra_openings(dimensions, JoltRelationId::FusedIncrementSourceLink)
}

fn fused_increment_inactive_source_link_bytecode_ra_openings(
    dimensions: BytecodeReadRafDimensions,
) -> Vec<JoltOpeningId> {
    fused_increment_bytecode_ra_openings(
        dimensions,
        JoltRelationId::FusedIncrementInactiveSourceLink,
    )
}

fn fused_increment_bytecode_ra_openings(
    dimensions: BytecodeReadRafDimensions,
    relation: JoltRelationId,
) -> Vec<JoltOpeningId> {
    (0..dimensions.num_committed_ra_polys())
        .map(|index| JoltOpeningId::committed(JoltCommittedPolynomial::BytecodeRa(index), relation))
        .collect()
}

pub fn fused_increment_source_lattice_view_formula<F: Field>(
    target: LatticeFusedIncrementTarget,
    bytecode_chunk: usize,
) -> LatticePackedViewFormula<F> {
    match target {
        LatticeFusedIncrementTarget::Ram => {
            bytecode_store_flag_lattice_view_formula(bytecode_chunk)
        }
        LatticeFusedIncrementTarget::Rd => bytecode_rd_present_lattice_view_formula(bytecode_chunk),
    }
}

pub fn bytecode_store_flag_lattice_view_formula<F>(chunk: usize) -> LatticePackedViewFormula<F> {
    LatticePackedViewFormula::direct(
        LatticePackedFamilyId::BytecodeCircuitFlag {
            chunk,
            flag: CircuitFlags::Store as usize,
        },
        0,
        1,
    )
}

pub fn bytecode_rd_present_lattice_view_formula<F: Field>(
    chunk: usize,
) -> LatticePackedViewFormula<F> {
    LatticePackedViewFormula::linear_decoded(weighted_symbol_terms(
        LatticePackedFamilyId::BytecodeRegisterSelector { chunk, selector: 2 },
        0,
        [F::one(); 1 << REGISTER_ADDRESS_BITS],
    ))
}

pub fn fused_increment_magnitude_lattice_view_formula<F: Field>() -> LatticePackedViewFormula<F> {
    LatticePackedViewFormula::linear_decoded(fused_increment_magnitude_terms())
}

pub fn fused_increment_magnitude_terms<F: Field>() -> Vec<LatticePackedViewTerm<F>> {
    let mut terms = Vec::with_capacity(FUSED_INCREMENT_BYTE_LIMBS * 256);
    let mut place = F::one();
    for index in 0..FUSED_INCREMENT_BYTE_LIMBS {
        terms.extend(weighted_byte_decode_terms(
            LatticePackedFamilyId::IncByte { index },
            [(0, place)],
        ));
        place *= F::from_u64(256);
    }
    terms
}

pub fn fused_increment_sign_lattice_view_formula<F>() -> LatticePackedViewFormula<F> {
    LatticePackedViewFormula::direct(LatticePackedFamilyId::IncSign, 0, 1)
}

pub fn fused_increment_validity_requirements() -> Vec<LatticePackedValidityRequirement> {
    let mut requirements = (0..FUSED_INCREMENT_BYTE_LIMBS)
        .map(|index| {
            LatticePackedValidityRequirement::exact_one_hot(
                LatticePackedFamilyId::IncByte { index },
                1,
                256,
            )
        })
        .collect::<Vec<_>>();
    requirements.push(LatticePackedValidityRequirement::boolean_indicator(
        LatticePackedFamilyId::IncSign,
        1,
        2,
        1,
    ));
    requirements.push(LatticePackedValidityRequirement::fused_increment_canonical_zero());
    requirements
}

pub fn advice_bytes_validity_requirement(kind: JoltAdviceKind) -> LatticePackedValidityRequirement {
    byte_validity_requirement(LatticePackedFamilyId::AdviceBytes { kind, index: 0 }, 1)
}

pub fn program_image_validity_requirement() -> LatticePackedValidityRequirement {
    byte_validity_requirement(LatticePackedFamilyId::ProgramImageInit, 8)
}

pub fn bytecode_validity_requirements(
    chunk: usize,
    field_byte_width: usize,
) -> Vec<LatticePackedValidityRequirement> {
    let register_count = 1usize << REGISTER_ADDRESS_BITS;
    let mut requirements = Vec::new();
    for selector in 0..3 {
        requirements.push(LatticePackedValidityRequirement::optional_one_hot(
            LatticePackedFamilyId::BytecodeRegisterSelector { chunk, selector },
            1,
            register_count,
        ));
    }
    for flag in 0..NUM_CIRCUIT_FLAGS {
        requirements.push(LatticePackedValidityRequirement::boolean_indicator(
            LatticePackedFamilyId::BytecodeCircuitFlag { chunk, flag },
            1,
            2,
            1,
        ));
    }
    for flag in 0..NUM_INSTRUCTION_FLAGS {
        requirements.push(LatticePackedValidityRequirement::boolean_indicator(
            LatticePackedFamilyId::BytecodeInstructionFlag { chunk, flag },
            1,
            2,
            1,
        ));
    }
    requirements.push(LatticePackedValidityRequirement::optional_one_hot(
        LatticePackedFamilyId::BytecodeLookupSelector { chunk },
        1,
        LookupTableKind::<XLEN>::COUNT.next_power_of_two(),
    ));
    requirements.push(LatticePackedValidityRequirement::boolean_indicator(
        LatticePackedFamilyId::BytecodeRafFlag { chunk },
        1,
        2,
        1,
    ));
    requirements.push(byte_validity_requirement(
        LatticePackedFamilyId::BytecodeUnexpandedPcBytes { chunk },
        8,
    ));
    requirements.push(byte_validity_requirement(
        LatticePackedFamilyId::BytecodeImmBytes { chunk },
        field_byte_width,
    ));
    requirements.push(LatticePackedValidityRequirement::bytecode_store_rd_disjoint(chunk));
    requirements
}

fn byte_validity_requirement(
    family: LatticePackedFamilyId,
    limbs: usize,
) -> LatticePackedValidityRequirement {
    LatticePackedValidityRequirement::exact_one_hot(family, limbs, 256)
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
                relation: JoltRelationId::FusedIncrementTranslation,
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
    use crate::protocols::jolt::JoltPolynomialId;
    use jolt_field::{Fr, FromPrimitiveInt};

    #[test]
    fn final_opening_lattice_requirement_marks_increments_as_translation() {
        assert_eq!(
            final_opening_lattice_requirement(JoltCommittedPolynomial::RamInc),
            LatticeFinalOpeningRequirement::RequiresTranslation {
                relation: JoltRelationId::FusedIncrementTranslation
            }
        );
        assert_eq!(
            final_opening_lattice_requirement(JoltCommittedPolynomial::RdInc),
            LatticeFinalOpeningRequirement::RequiresTranslation {
                relation: JoltRelationId::FusedIncrementTranslation
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
    fn fused_increment_translation_names_existing_inc_claim_outputs() {
        assert_eq!(
            fused_increment_translation_relation(),
            JoltRelationId::FusedIncrementTranslation
        );
        assert_eq!(
            fused_increment_translation_input_opening(LatticeFusedIncrementTarget::Ram),
            JoltOpeningId::committed(
                JoltCommittedPolynomial::RamInc,
                JoltRelationId::IncClaimReduction
            )
        );
        assert_eq!(
            fused_increment_translation_input_opening(LatticeFusedIncrementTarget::Rd),
            JoltOpeningId::committed(
                JoltCommittedPolynomial::RdInc,
                JoltRelationId::IncClaimReduction
            )
        );
        assert_eq!(
            fused_increment_translation_output_openings(),
            [
                fused_increment_source_opening(LatticeFusedIncrementTarget::Ram),
                fused_increment_magnitude_opening(),
                fused_increment_sign_opening(),
                fused_increment_source_opening(LatticeFusedIncrementTarget::Rd),
            ]
        );
        assert_eq!(
            fused_increment_source_opening(LatticeFusedIncrementTarget::Ram),
            JoltOpeningId::lattice(JoltRelationId::FusedIncrementTranslation, 0)
        );
        assert_eq!(
            fused_increment_source_opening(LatticeFusedIncrementTarget::Rd),
            JoltOpeningId::lattice(JoltRelationId::FusedIncrementTranslation, 1)
        );
    }

    #[test]
    fn fused_increment_translation_claim_batches_ram_and_rd() {
        let claims = fused_increment_translation_claim::<Fr>(TraceDimensions::new(5));

        assert_eq!(claims.id, JoltRelationId::FusedIncrementTranslation);
        assert_eq!(claims.sumcheck, TraceDimensions::new(5).sumcheck(4));
        assert_eq!(
            claims.input.required_openings,
            vec![
                fused_increment_translation_input_opening(LatticeFusedIncrementTarget::Ram),
                fused_increment_translation_input_opening(LatticeFusedIncrementTarget::Rd),
            ]
        );
        assert_eq!(
            claims.output.required_openings,
            fused_increment_translation_output_openings()
        );
        assert_eq!(
            claims.required_challenges(),
            vec![JoltChallengeId::from(
                FusedIncrementTranslationChallenge::Gamma
            )]
        );
    }

    #[test]
    fn fused_increment_source_link_claim_batches_translation_sources() {
        let dimensions = BytecodeReadRafDimensions::new(5, 10, 2);
        let claims = fused_increment_source_link_claim::<Fr>(dimensions);
        let expected_bytecode_ra = (0..2)
            .map(|index| {
                JoltOpeningId::committed(
                    JoltCommittedPolynomial::BytecodeRa(index),
                    JoltRelationId::FusedIncrementSourceLink,
                )
            })
            .collect::<Vec<_>>();
        let mut expected_output = expected_bytecode_ra.clone();
        expected_output.extend([
            fused_increment_bytecode_source_opening(LatticeFusedIncrementTarget::Ram),
            fused_increment_bytecode_source_opening(LatticeFusedIncrementTarget::Rd),
        ]);

        assert_eq!(
            fused_increment_source_link_relation(),
            JoltRelationId::FusedIncrementSourceLink
        );
        assert_eq!(claims.id, JoltRelationId::FusedIncrementSourceLink);
        assert_eq!(claims.sumcheck, dimensions.sumcheck());
        assert_eq!(
            claims.input.required_openings,
            vec![
                fused_increment_source_opening(LatticeFusedIncrementTarget::Ram),
                fused_increment_source_opening(LatticeFusedIncrementTarget::Rd),
            ]
        );
        assert_eq!(claims.output.required_openings, expected_output);
        assert_eq!(
            fused_increment_source_link_output_openings(dimensions),
            claims.output.required_openings
        );
        assert_eq!(
            claims.required_challenges(),
            vec![JoltChallengeId::from(
                FusedIncrementSourceLinkChallenge::Gamma
            )]
        );
    }

    #[test]
    fn fused_increment_inactive_zero_claim_batches_magnitude_and_sign() {
        let claims = fused_increment_inactive_zero_claim::<Fr>(TraceDimensions::new(5));

        assert_eq!(
            fused_increment_inactive_zero_relation(),
            JoltRelationId::FusedIncrementInactiveZero
        );
        assert_eq!(claims.id, JoltRelationId::FusedIncrementInactiveZero);
        assert_eq!(claims.sumcheck, TraceDimensions::new(5).sumcheck(3));
        assert!(claims.input.required_openings.is_empty());
        assert_eq!(
            claims.output.required_openings,
            fused_increment_inactive_zero_output_openings()
        );
        assert_eq!(
            claims.required_challenges(),
            vec![JoltChallengeId::from(
                FusedIncrementInactiveZeroChallenge::Beta
            )]
        );
    }

    #[test]
    fn fused_increment_inactive_source_link_claim_batches_inactive_sources() {
        let dimensions = BytecodeReadRafDimensions::new(5, 10, 2);
        let claims = fused_increment_inactive_source_link_claim::<Fr>(dimensions);
        let expected_bytecode_ra = (0..2)
            .map(|index| {
                JoltOpeningId::committed(
                    JoltCommittedPolynomial::BytecodeRa(index),
                    JoltRelationId::FusedIncrementInactiveSourceLink,
                )
            })
            .collect::<Vec<_>>();
        let mut expected_output = expected_bytecode_ra.clone();
        expected_output.extend([
            fused_increment_inactive_bytecode_source_opening(LatticeFusedIncrementTarget::Ram),
            fused_increment_inactive_bytecode_source_opening(LatticeFusedIncrementTarget::Rd),
        ]);

        assert_eq!(
            fused_increment_inactive_source_link_relation(),
            JoltRelationId::FusedIncrementInactiveSourceLink
        );
        assert_eq!(claims.id, JoltRelationId::FusedIncrementInactiveSourceLink);
        assert_eq!(claims.sumcheck, dimensions.sumcheck());
        assert_eq!(
            claims.input.required_openings,
            vec![
                fused_increment_inactive_source_opening(LatticeFusedIncrementTarget::Ram),
                fused_increment_inactive_source_opening(LatticeFusedIncrementTarget::Rd),
            ]
        );
        assert_eq!(claims.output.required_openings, expected_output);
        assert_eq!(
            fused_increment_inactive_source_link_output_openings(dimensions),
            claims.output.required_openings
        );
        assert_eq!(
            claims.required_challenges(),
            vec![JoltChallengeId::from(
                FusedIncrementInactiveSourceLinkChallenge::Gamma
            )]
        );
    }

    #[test]
    fn fused_increment_source_link_claim_evaluates_read_raf_source_formula() {
        let dimensions = BytecodeReadRafDimensions::new(5, 10, 2);
        let claims = fused_increment_source_link_claim::<Fr>(dimensions);
        let ram_source = Fr::from_u64(13);
        let rd_source = Fr::from_u64(17);
        let bytecode_ra = [Fr::from_u64(2), Fr::from_u64(3)];
        let store_flag = Fr::from_u64(5);
        let rd_present = Fr::from_u64(7);
        let gamma = Fr::from_u64(11);
        let zero = Fr::from_u64(0);

        let input = claims.input.expression().evaluate(
            |id| match *id {
                id if id == fused_increment_source_opening(LatticeFusedIncrementTarget::Ram) => {
                    ram_source
                }
                id if id == fused_increment_source_opening(LatticeFusedIncrementTarget::Rd) => {
                    rd_source
                }
                _ => zero,
            },
            |id| match id {
                JoltChallengeId::FusedIncrementSourceLink(
                    FusedIncrementSourceLinkChallenge::Gamma,
                ) => gamma,
                _ => zero,
            },
            |_| zero,
        );
        let output = claims.output.expression().evaluate(
            |id| match *id {
                JoltOpeningId::Polynomial {
                    polynomial:
                        JoltPolynomialId::Committed(JoltCommittedPolynomial::BytecodeRa(index)),
                    relation: JoltRelationId::FusedIncrementSourceLink,
                } => bytecode_ra[index],
                id if id
                    == fused_increment_bytecode_source_opening(
                        LatticeFusedIncrementTarget::Ram,
                    ) =>
                {
                    store_flag
                }
                id if id
                    == fused_increment_bytecode_source_opening(LatticeFusedIncrementTarget::Rd) =>
                {
                    rd_present
                }
                _ => zero,
            },
            |id| match id {
                JoltChallengeId::FusedIncrementSourceLink(
                    FusedIncrementSourceLinkChallenge::Gamma,
                ) => gamma,
                _ => zero,
            },
            |_| zero,
        );

        assert_eq!(input, ram_source + gamma * rd_source);
        assert_eq!(
            output,
            bytecode_ra[0] * bytecode_ra[1] * (store_flag + gamma * rd_present)
        );
    }

    #[test]
    fn fused_increment_inactive_zero_claim_evaluates_inactive_selector() {
        let claims = fused_increment_inactive_zero_claim::<Fr>(TraceDimensions::new(5));
        let ram_source = Fr::from_u64(0);
        let rd_source = Fr::from_u64(0);
        let magnitude = Fr::from_u64(19);
        let sign = Fr::from_u64(1);
        let beta = Fr::from_u64(23);
        let zero = Fr::from_u64(0);

        let input = claims
            .input
            .expression()
            .evaluate(|_| zero, |_| zero, |_| zero);
        let output = claims.output.expression().evaluate(
            |id| match *id {
                id if id
                    == fused_increment_inactive_source_opening(
                        LatticeFusedIncrementTarget::Ram,
                    ) =>
                {
                    ram_source
                }
                id if id
                    == fused_increment_inactive_source_opening(LatticeFusedIncrementTarget::Rd) =>
                {
                    rd_source
                }
                id if id == fused_increment_inactive_magnitude_opening() => magnitude,
                id if id == fused_increment_inactive_sign_opening() => sign,
                _ => zero,
            },
            |id| match id {
                JoltChallengeId::FusedIncrementInactiveZero(
                    FusedIncrementInactiveZeroChallenge::Beta,
                ) => beta,
                _ => zero,
            },
            |_| zero,
        );

        assert_eq!(input, zero);
        assert_eq!(output, magnitude + beta * sign);
    }

    #[test]
    fn fused_increment_translation_claim_evaluates_signed_source_formula() {
        let claims = fused_increment_translation_claim::<Fr>(TraceDimensions::new(5));
        let ram_inc = Fr::from_u64(11);
        let rd_inc = Fr::from_u64(13);
        let ram_source = Fr::from_u64(1);
        let rd_source = Fr::from_u64(0);
        let magnitude = Fr::from_u64(19);
        let sign = Fr::from_u64(1);
        let gamma = Fr::from_u64(23);
        let zero = Fr::from_u64(0);

        let input = claims.input.expression().evaluate(
            |id| match *id {
                id if id
                    == fused_increment_translation_input_opening(
                        LatticeFusedIncrementTarget::Ram,
                    ) =>
                {
                    ram_inc
                }
                id if id
                    == fused_increment_translation_input_opening(
                        LatticeFusedIncrementTarget::Rd,
                    ) =>
                {
                    rd_inc
                }
                _ => zero,
            },
            |id| match id {
                JoltChallengeId::FusedIncrementTranslation(
                    FusedIncrementTranslationChallenge::Gamma,
                ) => gamma,
                _ => zero,
            },
            |_| zero,
        );
        let output = claims.output.expression().evaluate(
            |id| match *id {
                id if id == fused_increment_source_opening(LatticeFusedIncrementTarget::Ram) => {
                    ram_source
                }
                id if id == fused_increment_source_opening(LatticeFusedIncrementTarget::Rd) => {
                    rd_source
                }
                id if id == fused_increment_magnitude_opening() => magnitude,
                id if id == fused_increment_sign_opening() => sign,
                _ => zero,
            },
            |id| match id {
                JoltChallengeId::FusedIncrementTranslation(
                    FusedIncrementTranslationChallenge::Gamma,
                ) => gamma,
                _ => zero,
            },
            |_| zero,
        );

        assert_eq!(input, ram_inc + gamma * rd_inc);
        assert_eq!(output, -magnitude);
    }

    #[test]
    fn fused_increment_source_formulas_use_committed_bytecode_lanes() {
        assert_eq!(
            fused_increment_source_lattice_view_formula::<Fr>(LatticeFusedIncrementTarget::Ram, 3),
            LatticePackedViewFormula::direct(
                LatticePackedFamilyId::BytecodeCircuitFlag {
                    chunk: 3,
                    flag: CircuitFlags::Store as usize,
                },
                0,
                1
            )
        );

        let rd_present =
            fused_increment_source_lattice_view_formula::<Fr>(LatticeFusedIncrementTarget::Rd, 2);
        let terms = linear_decoded_terms(&rd_present);
        assert_eq!(terms.len(), 1 << REGISTER_ADDRESS_BITS);
        assert_eq!(
            find_term(
                terms,
                LatticePackedFamilyId::BytecodeRegisterSelector {
                    chunk: 2,
                    selector: 2,
                },
                0,
                0
            )
            .coefficient,
            Fr::from_u64(1)
        );
        assert_eq!(
            find_term(
                terms,
                LatticePackedFamilyId::BytecodeRegisterSelector {
                    chunk: 2,
                    selector: 2,
                },
                0,
                (1 << REGISTER_ADDRESS_BITS) - 1
            )
            .coefficient,
            Fr::from_u64(1)
        );
    }

    #[test]
    fn fused_increment_decode_formulas_use_sign_magnitude_families() {
        assert_eq!(
            fused_increment_sign_lattice_view_formula::<Fr>(),
            LatticePackedViewFormula::direct(LatticePackedFamilyId::IncSign, 0, 1)
        );

        let magnitude = fused_increment_magnitude_lattice_view_formula::<Fr>();
        let terms = linear_decoded_terms(&magnitude);
        assert_eq!(terms.len(), FUSED_INCREMENT_BYTE_LIMBS * 256);
        assert_eq!(
            find_term(terms, LatticePackedFamilyId::IncByte { index: 0 }, 0, 7).coefficient,
            Fr::from_u64(7)
        );
        assert_eq!(
            find_term(terms, LatticePackedFamilyId::IncByte { index: 1 }, 0, 3).coefficient,
            Fr::from_u64(256 * 3)
        );
        assert_eq!(
            find_term(terms, LatticePackedFamilyId::IncByte { index: 7 }, 0, 2).coefficient,
            Fr::from_u64(1u64 << 57)
        );
    }

    #[test]
    fn fused_increment_validity_requirements_cover_bytes_and_sign() {
        let requirements = fused_increment_validity_requirements();

        assert_eq!(requirements.len(), FUSED_INCREMENT_BYTE_LIMBS + 2);
        assert_eq!(
            requirements[0],
            LatticePackedValidityRequirement::exact_one_hot(
                LatticePackedFamilyId::IncByte { index: 0 },
                1,
                256,
            )
        );
        assert_eq!(
            requirements[FUSED_INCREMENT_BYTE_LIMBS - 1],
            LatticePackedValidityRequirement::exact_one_hot(
                LatticePackedFamilyId::IncByte {
                    index: FUSED_INCREMENT_BYTE_LIMBS - 1
                },
                1,
                256,
            )
        );
        assert_eq!(
            requirements[FUSED_INCREMENT_BYTE_LIMBS],
            LatticePackedValidityRequirement::boolean_indicator(
                LatticePackedFamilyId::IncSign,
                1,
                2,
                1,
            )
        );
        assert_eq!(
            requirements[FUSED_INCREMENT_BYTE_LIMBS + 1],
            LatticePackedValidityRequirement::fused_increment_canonical_zero()
        );
    }

    #[test]
    fn packed_validity_digest_is_order_stable_and_kind_sensitive() {
        let exact = LatticePackedValidityRequirement::exact_one_hot(
            LatticePackedFamilyId::IncByte { index: 0 },
            1,
            256,
        );
        let sign = LatticePackedValidityRequirement::boolean_indicator(
            LatticePackedFamilyId::IncSign,
            1,
            2,
            1,
        );
        let optional = LatticePackedValidityRequirement::optional_one_hot(
            LatticePackedFamilyId::IncByte { index: 0 },
            1,
            256,
        );

        assert_eq!(
            lattice_packed_validity_digest(&[exact.clone(), sign.clone()]),
            lattice_packed_validity_digest(&[sign, exact.clone()])
        );
        assert_ne!(
            lattice_packed_validity_digest(&[exact]),
            lattice_packed_validity_digest(&[optional])
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
            requirements.contains(&LatticePackedValidityRequirement::optional_one_hot(
                LatticePackedFamilyId::BytecodeRegisterSelector { chunk, selector: 2 },
                1,
                1 << REGISTER_ADDRESS_BITS,
            ))
        );
        assert!(
            requirements.contains(&LatticePackedValidityRequirement::boolean_indicator(
                LatticePackedFamilyId::BytecodeCircuitFlag { chunk, flag: 0 },
                1,
                2,
                1,
            ))
        );
        assert!(
            requirements.contains(&LatticePackedValidityRequirement::boolean_indicator(
                LatticePackedFamilyId::BytecodeInstructionFlag {
                    chunk,
                    flag: NUM_INSTRUCTION_FLAGS - 1,
                },
                1,
                2,
                1,
            ))
        );
        assert!(
            requirements.contains(&LatticePackedValidityRequirement::optional_one_hot(
                LatticePackedFamilyId::BytecodeLookupSelector { chunk },
                1,
                LookupTableKind::<XLEN>::COUNT.next_power_of_two(),
            ))
        );
        assert!(
            requirements.contains(&LatticePackedValidityRequirement::boolean_indicator(
                LatticePackedFamilyId::BytecodeRafFlag { chunk },
                1,
                2,
                1,
            ))
        );
        assert!(
            requirements.contains(&LatticePackedValidityRequirement::exact_one_hot(
                LatticePackedFamilyId::BytecodeUnexpandedPcBytes { chunk },
                8,
                256,
            ))
        );
        assert!(
            requirements.contains(&LatticePackedValidityRequirement::exact_one_hot(
                LatticePackedFamilyId::BytecodeImmBytes { chunk },
                field_byte_width,
                256,
            ))
        );
        assert!(requirements
            .contains(&LatticePackedValidityRequirement::bytecode_store_rd_disjoint(chunk)));
    }

    #[test]
    fn advice_and_program_image_validity_requirements_are_byte_facts() {
        assert_eq!(
            advice_bytes_validity_requirement(JoltAdviceKind::Trusted),
            LatticePackedValidityRequirement::exact_one_hot(
                LatticePackedFamilyId::AdviceBytes {
                    kind: JoltAdviceKind::Trusted,
                    index: 0,
                },
                1,
                256,
            )
        );
        assert_eq!(
            program_image_validity_requirement(),
            LatticePackedValidityRequirement::exact_one_hot(
                LatticePackedFamilyId::ProgramImageInit,
                8,
                256,
            )
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
                LatticePackedFamilyId::BytecodeRegisterSelector {
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
