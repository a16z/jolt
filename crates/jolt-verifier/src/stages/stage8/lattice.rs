use crate::{
    config::{validate_protocol_config, JoltProtocolConfig, PcsFamily},
    stages::PrecommittedSchedule,
    VerifierError,
};
use jolt_akita::{
    AkitaField, PackedAdviceKind, PackedAlphabet, PackedFactDomain, PackedFamilyId,
    PackedFamilySpec, PackedViewError, PackedViewFormula, PackedViewTerm, PackedWitnessLayout,
    AKITA_FIELD_MODULUS,
};
#[cfg(feature = "field-inline")]
use jolt_claims::protocols::field_inline::{
    formulas::{claim_reductions::increments as field_increments, lattice as field_lattice},
    FieldInlineOpeningId,
};
use jolt_claims::protocols::jolt::{
    advice_bytes_validity_requirement, byte_decode_terms, bytecode_chunk_lattice_view_formula,
    bytecode_imm_canonical_bytes_requirement, bytecode_validity_requirements,
    formulas::{dimensions::REGISTER_ADDRESS_BITS, ra::JoltRaPolynomialLayout},
    fused_increment_bytecode_source_opening, fused_increment_inactive_bytecode_source_opening,
    fused_increment_inactive_magnitude_opening, fused_increment_inactive_sign_opening,
    fused_increment_inactive_source_opening, fused_increment_magnitude_lattice_view_formula,
    fused_increment_magnitude_opening, fused_increment_sign_lattice_view_formula,
    fused_increment_sign_opening, fused_increment_source_lattice_view_formula,
    fused_increment_source_opening, fused_increment_validity_requirements,
    lattice_packed_validity_digest, little_endian_byte_decode_terms,
    program_image_validity_requirement, weighted_symbol_terms, AdviceClaimReductionLayout,
    JoltAdviceKind, JoltCommittedPolynomial, JoltOpeningId, JoltPolynomialId, JoltRelationId,
    LatticeFusedIncrementTarget, LatticePackedFamilyId, LatticePackedValidityKind,
    LatticePackedValidityRequirement, LatticePackedViewFormula, LatticePackedViewTerm,
    ProgramImageClaimReductionLayout, TracePolynomialOrder, FUSED_INCREMENT_BYTE_LIMBS,
};
use jolt_field::{Field, FixedByteSize};
use jolt_openings::{
    BatchOpeningClaim, BatchOpeningScheme, BatchOpeningStatement, CommitmentScheme,
    PackedLinearTerm, PhysicalView,
};
use jolt_poly::{try_eq_mle, EqPolynomial};
use jolt_riscv::CircuitFlags;
use jolt_sumcheck::{
    BatchedEvaluationClaim, BatchedSumcheckVerifier, ClearProof, SumcheckClaim, SumcheckProof,
};
use jolt_transcript::{Label, LabelWithCount, Transcript, U64Word};

use super::outputs::{Stage8LogicalManifest, Stage8OpeningId, Stage8PhysicalManifest};

pub type JoltLatticeViewFormulaWithRowPoint<F> =
    (Stage8OpeningId, LatticePackedViewFormula<F>, Vec<F>);

pub type LatticePackedValidityBatchStatement<F, C> = BatchOpeningStatement<F, C, usize, usize, F>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LatticePackedValidityStatement {
    pub requirement: LatticePackedValidityRequirement,
    pub kind: LatticePackedValidityStatementKind,
    pub num_vars: usize,
    pub degree: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LatticePackedValidityStatementKind {
    CellBooleanity,
    ExactOneHotRowSum,
    OptionalOneHotRowSum,
    BooleanIndicator,
    FusedIncrementCanonicalZero,
    BytecodeStoreRdDisjoint,
    FieldElementCanonicalBytes,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum FieldCanonicalFactor {
    Range {
        byte_index: usize,
        family: PackedFamilyId,
        limb: usize,
        start_symbol: usize,
    },
    Eq {
        byte_index: usize,
        family: PackedFamilyId,
        limb: usize,
        symbol: usize,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LatticePackedValidityBatch<F: Field, C> {
    pub statement: LatticePackedValidityBatchStatement<F, C>,
    pub expected_final_claim: F,
}

pub fn derive_akita_packed_witness_layout(
    config: &JoltProtocolConfig,
    log_t: usize,
    log_k_chunk: usize,
    ra_layout: JoltRaPolynomialLayout,
    precommitted: &PrecommittedSchedule,
) -> Result<PackedWitnessLayout, VerifierError> {
    if validate_protocol_config(config)? != PcsFamily::Lattice {
        return Err(invalid_lattice_config(
            "Akita packed witness layout derivation requires lattice PCS mode",
        ));
    }

    let trace = PackedFactDomain::TraceRows { log_t };
    let ra_alphabet = one_hot_alphabet(log_k_chunk)?;
    let mut specs = Vec::new();
    specs.extend((0..ra_layout.instruction()).map(|index| {
        PackedFamilySpec::direct(
            PackedFamilyId::InstructionRa { index },
            trace,
            1,
            ra_alphabet,
        )
    }));
    specs.extend((0..ra_layout.bytecode()).map(|index| {
        PackedFamilySpec::direct(PackedFamilyId::BytecodeRa { index }, trace, 1, ra_alphabet)
    }));
    specs.extend((0..ra_layout.ram()).map(|index| {
        PackedFamilySpec::direct(PackedFamilyId::RamRa { index }, trace, 1, ra_alphabet)
    }));

    extend_validity_requirement_families(
        &mut specs,
        &fused_increment_validity_requirements(),
        trace,
    )?;

    if config.lattice.field_inline.enabled {
        extend_field_rd_inc_families(&mut specs, trace)?;
    }

    if config.lattice.advice.trusted {
        specs.push(advice_family(
            JoltAdviceKind::Trusted,
            require_advice_layout(precommitted, JoltAdviceKind::Trusted)?,
        )?);
    }
    if config.lattice.advice.untrusted {
        specs.push(advice_family(
            JoltAdviceKind::Untrusted,
            require_advice_layout(precommitted, JoltAdviceKind::Untrusted)?,
        )?);
    }

    let bytecode_layout = precommitted.bytecode.as_ref().ok_or_else(|| {
        invalid_precommitted_schedule(
            "lattice committed-program mode requires a bytecode claim-reduction layout",
        )
    })?;
    for index in 0..bytecode_layout.chunk_count() {
        extend_bytecode_families(&mut specs, index, bytecode_layout.log_bytecode_chunk_size())?;
    }

    let program_image_layout = precommitted.program_image.as_ref().ok_or_else(|| {
        invalid_precommitted_schedule(
            "lattice committed-program mode requires a program-image claim-reduction layout",
        )
    })?;
    specs.push(program_image_family(program_image_layout)?);

    PackedWitnessLayout::new(specs).map_err(|error| invalid_lattice_config(error.to_string()))
}

pub fn derive_akita_packed_validity_requirements(
    config: &JoltProtocolConfig,
    precommitted: &PrecommittedSchedule,
) -> Result<Vec<LatticePackedValidityRequirement>, VerifierError> {
    if validate_protocol_config(config)? != PcsFamily::Lattice {
        return Err(invalid_lattice_config(
            "Akita packed witness validity derivation requires lattice PCS mode",
        ));
    }

    let mut requirements = fused_increment_validity_requirements();
    if config.lattice.field_inline.enabled {
        requirements.extend(field_rd_inc_validity_requirements());
    }
    if config.lattice.advice.trusted {
        let _ = require_advice_layout(precommitted, JoltAdviceKind::Trusted)?;
        requirements.push(advice_bytes_validity_requirement(JoltAdviceKind::Trusted));
    }
    if config.lattice.advice.untrusted {
        let _ = require_advice_layout(precommitted, JoltAdviceKind::Untrusted)?;
        requirements.push(advice_bytes_validity_requirement(JoltAdviceKind::Untrusted));
    }

    let bytecode_layout = precommitted.bytecode.as_ref().ok_or_else(|| {
        invalid_precommitted_schedule(
            "lattice committed-program mode requires a bytecode claim-reduction layout",
        )
    })?;
    for index in 0..bytecode_layout.chunk_count() {
        requirements.extend(bytecode_validity_requirements(index, AkitaField::NUM_BYTES));
        requirements.push(bytecode_imm_canonical_bytes_requirement(
            index,
            AkitaField::NUM_BYTES,
            AKITA_FIELD_MODULUS,
        ));
    }

    let _ = precommitted.program_image.as_ref().ok_or_else(|| {
        invalid_precommitted_schedule(
            "lattice committed-program mode requires a program-image claim-reduction layout",
        )
    })?;
    requirements.push(program_image_validity_requirement());

    Ok(requirements)
}

pub fn derive_akita_packed_validity_statements(
    layout: &PackedWitnessLayout,
    requirements: &[LatticePackedValidityRequirement],
) -> Result<Vec<LatticePackedValidityStatement>, VerifierError> {
    let mut statements = Vec::new();
    for requirement in requirements {
        if matches!(
            requirement.kind,
            LatticePackedValidityKind::FieldElementCanonicalBytes { .. }
        ) {
            let row_vars = validate_field_element_canonical_bytes_layout(layout, requirement)?;
            statements.push(LatticePackedValidityStatement {
                requirement: requirement.clone(),
                kind: LatticePackedValidityStatementKind::FieldElementCanonicalBytes,
                num_vars: row_vars,
                degree: canonical_field_byte_width(requirement)?,
            });
            continue;
        }

        let family_id = akita_packed_family_id(&requirement.family);
        let family = layout.family(&family_id).ok_or_else(|| {
            invalid_lattice_config(format!(
                "packed validity requirement references missing family {family_id:?}"
            ))
        })?;
        if family.limbs != requirement.limbs {
            return Err(invalid_lattice_config(format!(
                "packed validity family {family_id:?} limb count mismatch: layout has {}, requirement has {}",
                family.limbs, requirement.limbs
            )));
        }
        if family.alphabet.size() != requirement.alphabet_size {
            return Err(invalid_lattice_config(format!(
                "packed validity family {family_id:?} alphabet mismatch: layout has {}, requirement has {}",
                family.alphabet.size(),
                requirement.alphabet_size
            )));
        }

        let rows = family.domain.rows().map_err(|error| {
            invalid_lattice_config(format!(
                "packed validity family {family_id:?} has invalid row domain: {error}"
            ))
        })?;
        let row_vars = power_of_two_log(rows, "packed validity row count")?;
        let limb_vars = power_of_two_log(requirement.limbs, "packed validity limb count")?;
        let symbol_vars =
            power_of_two_log(requirement.alphabet_size, "packed validity alphabet size")?;

        match requirement.kind {
            LatticePackedValidityKind::ExactOneHot => {
                statements.push(LatticePackedValidityStatement {
                    requirement: requirement.clone(),
                    kind: LatticePackedValidityStatementKind::CellBooleanity,
                    num_vars: row_vars + limb_vars + symbol_vars,
                    degree: 3,
                });
                statements.push(LatticePackedValidityStatement {
                    requirement: requirement.clone(),
                    kind: LatticePackedValidityStatementKind::ExactOneHotRowSum,
                    num_vars: row_vars + limb_vars,
                    degree: 3,
                });
            }
            LatticePackedValidityKind::OptionalOneHot => {
                statements.push(LatticePackedValidityStatement {
                    requirement: requirement.clone(),
                    kind: LatticePackedValidityStatementKind::CellBooleanity,
                    num_vars: row_vars + limb_vars + symbol_vars,
                    degree: 3,
                });
                statements.push(LatticePackedValidityStatement {
                    requirement: requirement.clone(),
                    kind: LatticePackedValidityStatementKind::OptionalOneHotRowSum,
                    num_vars: row_vars + limb_vars,
                    degree: 3,
                });
            }
            LatticePackedValidityKind::BooleanIndicator { symbol } => {
                if symbol >= requirement.alphabet_size {
                    return Err(invalid_lattice_config(format!(
                        "packed validity boolean indicator symbol {symbol} is outside alphabet size {}",
                        requirement.alphabet_size
                    )));
                }
                statements.push(LatticePackedValidityStatement {
                    requirement: requirement.clone(),
                    kind: LatticePackedValidityStatementKind::BooleanIndicator,
                    num_vars: row_vars + limb_vars,
                    degree: 3,
                });
            }
            LatticePackedValidityKind::FusedIncrementCanonicalZero => {
                let row_vars = validate_fused_increment_canonical_zero_layout(layout, requirement)?;
                statements.push(LatticePackedValidityStatement {
                    requirement: requirement.clone(),
                    kind: LatticePackedValidityStatementKind::FusedIncrementCanonicalZero,
                    num_vars: row_vars,
                    degree: FUSED_INCREMENT_BYTE_LIMBS + 2,
                });
            }
            LatticePackedValidityKind::BytecodeStoreRdDisjoint => {
                let row_vars = validate_bytecode_store_rd_disjoint_layout(layout, requirement)?;
                statements.push(LatticePackedValidityStatement {
                    requirement: requirement.clone(),
                    kind: LatticePackedValidityStatementKind::BytecodeStoreRdDisjoint,
                    num_vars: row_vars,
                    degree: 3,
                });
            }
            LatticePackedValidityKind::FieldElementCanonicalBytes { .. } => unreachable!(
                "field canonical-byte validity is handled before family shape validation"
            ),
        }
    }
    Ok(statements)
}

fn validate_fused_increment_canonical_zero_layout(
    layout: &PackedWitnessLayout,
    requirement: &LatticePackedValidityRequirement,
) -> Result<usize, VerifierError> {
    if requirement.family != LatticePackedFamilyId::IncSign
        || requirement.limbs != 1
        || requirement.alphabet_size != 2
    {
        return Err(invalid_lattice_config(
            "fused increment canonical-zero validity must be anchored on IncSign",
        ));
    }
    let sign = layout.family(&PackedFamilyId::IncSign).ok_or_else(|| {
        invalid_lattice_config("fused increment canonical-zero validity requires IncSign")
    })?;
    if sign.limbs != 1 || sign.alphabet.size() != 2 {
        return Err(invalid_lattice_config(
            "fused increment canonical-zero validity requires a boolean IncSign family",
        ));
    }
    let rows = sign.domain.rows().map_err(|error| {
        invalid_lattice_config(format!(
            "fused increment canonical-zero IncSign row domain is invalid: {error}"
        ))
    })?;
    let row_vars = power_of_two_log(rows, "fused increment canonical-zero row count")?;
    for index in 0..FUSED_INCREMENT_BYTE_LIMBS {
        let family_id = PackedFamilyId::IncByte { index };
        let family = layout.family(&family_id).ok_or_else(|| {
            invalid_lattice_config(format!(
                "fused increment canonical-zero validity requires {family_id:?}"
            ))
        })?;
        if family.domain != sign.domain || family.limbs != 1 || family.alphabet.size() != 256 {
            return Err(invalid_lattice_config(format!(
                "fused increment canonical-zero validity requires {family_id:?} to be a byte family over the IncSign row domain"
            )));
        }
    }
    Ok(row_vars)
}

fn validate_bytecode_store_rd_disjoint_layout(
    layout: &PackedWitnessLayout,
    requirement: &LatticePackedValidityRequirement,
) -> Result<usize, VerifierError> {
    let chunk = bytecode_store_rd_disjoint_chunk(requirement)?;
    let store_id = PackedFamilyId::BytecodeCircuitFlag {
        chunk,
        flag: CircuitFlags::Store as usize,
    };
    let store = layout.family(&store_id).ok_or_else(|| {
        invalid_lattice_config(format!(
            "bytecode Store/Rd disjointness requires {store_id:?}"
        ))
    })?;
    if store.limbs != 1 || store.alphabet.size() != 2 {
        return Err(invalid_lattice_config(
            "bytecode Store/Rd disjointness requires a boolean Store flag family",
        ));
    }

    let rd_id = PackedFamilyId::BytecodeRegisterSelector { chunk, selector: 2 };
    let rd = layout.family(&rd_id).ok_or_else(|| {
        invalid_lattice_config(format!("bytecode Store/Rd disjointness requires {rd_id:?}"))
    })?;
    if rd.domain != store.domain
        || rd.limbs != 1
        || rd.alphabet.size() != 1 << REGISTER_ADDRESS_BITS
    {
        return Err(invalid_lattice_config(format!(
            "bytecode Store/Rd disjointness requires {rd_id:?} to be an rd selector family over the Store flag row domain"
        )));
    }

    let rows = store.domain.rows().map_err(|error| {
        invalid_lattice_config(format!(
            "bytecode Store/Rd disjointness row domain is invalid: {error}"
        ))
    })?;
    power_of_two_log(rows, "bytecode Store/Rd disjointness row count")
}

fn bytecode_store_rd_disjoint_chunk(
    requirement: &LatticePackedValidityRequirement,
) -> Result<usize, VerifierError> {
    let LatticePackedFamilyId::BytecodeCircuitFlag { chunk, flag } = &requirement.family else {
        return Err(invalid_lattice_config(
            "bytecode Store/Rd disjointness must be anchored on the Store circuit flag",
        ));
    };
    if *flag != CircuitFlags::Store as usize
        || requirement.limbs != 1
        || requirement.alphabet_size != 2
    {
        return Err(invalid_lattice_config(
            "bytecode Store/Rd disjointness must be anchored on a boolean Store circuit flag",
        ));
    }
    Ok(*chunk)
}

fn validate_field_element_canonical_bytes_layout(
    layout: &PackedWitnessLayout,
    requirement: &LatticePackedValidityRequirement,
) -> Result<usize, VerifierError> {
    let byte_width = canonical_field_byte_width(requirement)?;
    if requirement.limbs != 1 || requirement.alphabet_size != 256 {
        return Err(invalid_lattice_config(
            "field-element canonical-byte validity must use one byte limb and byte alphabet",
        ));
    }

    match &requirement.family {
        LatticePackedFamilyId::FieldRdIncByte { index: 0 } => {
            let first_id = PackedFamilyId::FieldRdIncByte { index: 0 };
            let first = layout.family(&first_id).ok_or_else(|| {
                invalid_lattice_config(
                    "field-element canonical-byte validity requires FieldRdIncByte[0]",
                )
            })?;
            if first.limbs != 1 || first.alphabet.size() != 256 {
                return Err(invalid_lattice_config(
                    "field-element canonical-byte validity requires byte one-hot families",
                ));
            }
            let rows = first.domain.rows().map_err(|error| {
                invalid_lattice_config(format!(
                    "field-element canonical-byte row domain is invalid: {error}"
                ))
            })?;
            let row_vars = power_of_two_log(rows, "field-element canonical-byte row count")?;
            for index in 1..byte_width {
                let family_id = PackedFamilyId::FieldRdIncByte { index };
                let family = layout.family(&family_id).ok_or_else(|| {
                    invalid_lattice_config(format!(
                        "field-element canonical-byte validity requires {family_id:?}"
                    ))
                })?;
                if family.domain != first.domain
                    || family.limbs != 1
                    || family.alphabet.size() != 256
                {
                    return Err(invalid_lattice_config(format!(
                        "field-element canonical-byte validity requires {family_id:?} to be a byte family over the FieldRdIncByte[0] row domain"
                    )));
                }
            }
            Ok(row_vars)
        }
        LatticePackedFamilyId::BytecodeImmBytes { chunk } => {
            let family_id = PackedFamilyId::BytecodeImmBytes { chunk: *chunk };
            let family = layout.family(&family_id).ok_or_else(|| {
                invalid_lattice_config(format!(
                    "field-element canonical-byte validity requires {family_id:?}"
                ))
            })?;
            if family.limbs != byte_width || family.alphabet.size() != 256 {
                return Err(invalid_lattice_config(format!(
                    "field-element canonical-byte validity requires {family_id:?} to expose field bytes as byte limbs"
                )));
            }
            let rows = family.domain.rows().map_err(|error| {
                invalid_lattice_config(format!(
                    "field-element canonical-byte row domain is invalid: {error}"
                ))
            })?;
            power_of_two_log(rows, "field-element canonical-byte row count")
        }
        _ => Err(invalid_lattice_config(format!(
            "field-element canonical-byte validity cannot be anchored on {:?}",
            requirement.family
        ))),
    }
}

fn canonical_field_byte_width(
    requirement: &LatticePackedValidityRequirement,
) -> Result<usize, VerifierError> {
    let LatticePackedValidityKind::FieldElementCanonicalBytes {
        byte_width,
        modulus,
    } = requirement.kind
    else {
        return Err(invalid_lattice_config(
            "field-element canonical-byte statement has a non-canonical requirement",
        ));
    };
    if byte_width == 0 || byte_width > u128::BITS as usize / 8 || modulus == 0 {
        return Err(invalid_lattice_config(
            "field-element canonical-byte validity requires a nonzero u128 modulus and 1..=16 bytes",
        ));
    }
    if byte_width < u128::BITS as usize / 8 && modulus >= (1u128 << (8 * byte_width)) {
        return Err(invalid_lattice_config(
            "field-element canonical-byte modulus does not fit in the declared byte width",
        ));
    }
    Ok(byte_width)
}

fn canonical_field_byte_location(
    requirement: &LatticePackedValidityRequirement,
    byte_index: usize,
) -> Result<(PackedFamilyId, usize), VerifierError> {
    let byte_width = canonical_field_byte_width(requirement)?;
    if byte_index >= byte_width {
        return Err(invalid_lattice_config(format!(
            "field-element canonical-byte index {byte_index} is outside byte width {byte_width}",
        )));
    }
    match &requirement.family {
        LatticePackedFamilyId::FieldRdIncByte { index: 0 } => {
            Ok((PackedFamilyId::FieldRdIncByte { index: byte_index }, 0))
        }
        LatticePackedFamilyId::BytecodeImmBytes { chunk } => Ok((
            PackedFamilyId::BytecodeImmBytes { chunk: *chunk },
            byte_index,
        )),
        _ => Err(invalid_lattice_config(format!(
            "field-element canonical-byte validity cannot be anchored on {:?}",
            requirement.family
        ))),
    }
}

pub(crate) fn field_element_canonical_factors(
    requirement: &LatticePackedValidityRequirement,
) -> Result<Vec<FieldCanonicalFactor>, VerifierError> {
    let LatticePackedValidityKind::FieldElementCanonicalBytes {
        byte_width: _,
        modulus,
    } = requirement.kind
    else {
        return Err(invalid_lattice_config(
            "field-element canonical-byte statement has a non-canonical requirement",
        ));
    };
    let byte_width = canonical_field_byte_width(requirement)?;
    let modulus_bytes = modulus.to_le_bytes();
    let mut factors = Vec::with_capacity(2 * byte_width - 1);
    for byte_index in (0..byte_width).rev() {
        let modulus_byte = modulus_bytes[byte_index] as usize;
        let (family, limb) = canonical_field_byte_location(requirement, byte_index)?;
        let start_symbol = if byte_index == 0 {
            modulus_byte
        } else {
            modulus_byte + 1
        };
        if start_symbol < 256 {
            factors.push(FieldCanonicalFactor::Range {
                byte_index,
                family: family.clone(),
                limb,
                start_symbol,
            });
        }
        if byte_index > 0 {
            factors.push(FieldCanonicalFactor::Eq {
                byte_index,
                family,
                limb,
                symbol: modulus_byte,
            });
        }
    }
    Ok(factors)
}

pub fn lattice_packed_validity_claims<F>(
    statements: &[LatticePackedValidityStatement],
) -> Vec<SumcheckClaim<F>>
where
    F: Field,
{
    statements
        .iter()
        .map(|statement| SumcheckClaim {
            num_vars: statement.num_vars,
            degree: statement.degree,
            claimed_sum: F::zero(),
        })
        .collect()
}

pub fn lattice_packed_validity_opening_count(
    statements: &[LatticePackedValidityStatement],
) -> usize {
    statements
        .iter()
        .map(validity_statement_opening_count)
        .sum()
}

fn validity_statement_opening_count(statement: &LatticePackedValidityStatement) -> usize {
    match statement.kind {
        LatticePackedValidityStatementKind::CellBooleanity
        | LatticePackedValidityStatementKind::ExactOneHotRowSum
        | LatticePackedValidityStatementKind::OptionalOneHotRowSum
        | LatticePackedValidityStatementKind::BooleanIndicator => 1,
        LatticePackedValidityStatementKind::FusedIncrementCanonicalZero => {
            FUSED_INCREMENT_BYTE_LIMBS + 1
        }
        LatticePackedValidityStatementKind::BytecodeStoreRdDisjoint => 2,
        LatticePackedValidityStatementKind::FieldElementCanonicalBytes => {
            field_element_canonical_factors(&statement.requirement)
                .map_or(0, |factors| factors.len())
        }
    }
}

pub fn sample_lattice_packed_validity_eq_points<F, T>(
    transcript: &mut T,
    layout: &PackedWitnessLayout,
    statements: &[LatticePackedValidityStatement],
) -> Vec<Vec<F>>
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    absorb_lattice_packed_validity_metadata(transcript, layout, statements);
    statements
        .iter()
        .map(|statement| transcript.challenge_vector(statement.num_vars))
        .collect()
}

pub fn build_lattice_packed_validity_batch<F, C>(
    layout: &PackedWitnessLayout,
    statements: &[LatticePackedValidityStatement],
    commitment: C,
    eq_points: &[Vec<F>],
    reduction: &BatchedEvaluationClaim<F>,
    opening_claims: &[F],
) -> Result<LatticePackedValidityBatch<F, C>, VerifierError>
where
    F: Field,
    C: Clone,
{
    if eq_points.len() != statements.len() {
        return Err(invalid_lattice_config(format!(
            "packed validity equality point count {} does not match statement count {}",
            eq_points.len(),
            statements.len()
        )));
    }
    let expected_opening_claims = lattice_packed_validity_opening_count(statements);
    if opening_claims.len() != expected_opening_claims {
        return Err(VerifierError::AkitaPackedValidityClaimCountMismatch {
            expected: expected_opening_claims,
            got: opening_claims.len(),
        });
    }
    if reduction.batching_coefficients.len() != statements.len() {
        return Err(VerifierError::AkitaPackedValiditySumcheckFailed {
            reason: format!(
                "batch verifier returned {} coefficients for {} packed validity statements",
                reduction.batching_coefficients.len(),
                statements.len()
            ),
        });
    }

    let mut expected_final_claim = F::zero();
    let mut claims = Vec::with_capacity(expected_opening_claims);
    let mut opening_offset = 0;
    for (index, statement) in statements.iter().enumerate() {
        let point = reduction
            .try_instance_point(statement.num_vars)
            .map_err(|error| VerifierError::AkitaPackedValiditySumcheckFailed {
                reason: error.to_string(),
            })?;
        let opening_count = validity_statement_opening_count(statement);
        let statement_openings = &opening_claims[opening_offset..opening_offset + opening_count];
        let eq_mask = try_eq_mle(point, &eq_points[index]).map_err(|error| {
            VerifierError::AkitaPackedValiditySumcheckFailed {
                reason: error.to_string(),
            }
        })?;
        expected_final_claim += reduction.batching_coefficients[index]
            * eq_mask
            * validity_statement_value_from_openings(statement, statement_openings)?;

        for (factor, opening_claim) in statement_openings.iter().copied().enumerate() {
            claims.push(BatchOpeningClaim {
                id: opening_offset + factor,
                relation: index,
                commitment: commitment.clone(),
                claim: opening_claim,
                view: validity_factor_physical_view(layout, statement, point, factor)?,
                scale: F::one(),
            });
        }
        opening_offset += opening_count;
    }

    let point = reduction.reduction.point.as_slice().to_vec();
    Ok(LatticePackedValidityBatch {
        statement: BatchOpeningStatement {
            logical_point: point.clone(),
            pcs_point: point,
            layout_digest: layout.digest,
            claims,
        },
        expected_final_claim,
    })
}

#[expect(
    clippy::too_many_arguments,
    reason = "Verifier helper mirrors the top-level proof, layout, transcript, and PCS inputs without introducing another wrapper type."
)]
pub fn verify_lattice_packed_validity_proof<F, PCS, T, RoundCommitment>(
    setup: &PCS::VerifierSetup,
    transcript: &mut T,
    config: &JoltProtocolConfig,
    precommitted: &PrecommittedSchedule,
    layout: &PackedWitnessLayout,
    commitment: PCS::Output,
    sumcheck_proof: &SumcheckProof<F, RoundCommitment>,
    opening_claims: &[F],
    opening_proof: &PCS::Proof,
) -> Result<(), VerifierError>
where
    F: Field,
    PCS: CommitmentScheme<Field = F> + BatchOpeningScheme,
    PCS::Output: Clone,
    T: Transcript<Challenge = F>,
{
    let requirements = derive_akita_packed_validity_requirements(config, precommitted)?;
    let statements = derive_akita_packed_validity_statements(layout, &requirements)?;
    let expected_opening_claims = lattice_packed_validity_opening_count(&statements);
    if opening_claims.len() != expected_opening_claims {
        return Err(VerifierError::AkitaPackedValidityClaimCountMismatch {
            expected: expected_opening_claims,
            got: opening_claims.len(),
        });
    }

    let eq_points = sample_lattice_packed_validity_eq_points(transcript, layout, &statements);
    let sumcheck_claims = lattice_packed_validity_claims(&statements);
    let compressed = match sumcheck_proof {
        SumcheckProof::Clear(ClearProof::Compressed(proof)) => proof,
        SumcheckProof::Clear(ClearProof::Full(_)) => {
            return Err(VerifierError::AkitaPackedValiditySumcheckFailed {
                reason: "expected compressed clear proof, got full clear".to_string(),
            });
        }
        SumcheckProof::Committed(_) => {
            return Err(VerifierError::ExpectedClearProof {
                field: "lattice_packed_validity_sumcheck_proof",
            });
        }
    };
    let reduction =
        BatchedSumcheckVerifier::verify_compressed(&sumcheck_claims, compressed, transcript)
            .map_err(|error| VerifierError::AkitaPackedValiditySumcheckFailed {
                reason: error.to_string(),
            })?;
    let batch = build_lattice_packed_validity_batch(
        layout,
        &statements,
        commitment,
        &eq_points,
        &reduction,
        opening_claims,
    )?;
    if reduction.reduction.value != batch.expected_final_claim {
        return Err(VerifierError::AkitaPackedValidityOutputMismatch);
    }
    PCS::verify_batch(setup, transcript, &batch.statement, opening_proof)
        .map_err(
            |error| VerifierError::AkitaPackedValidityOpeningVerificationFailed {
                reason: error.to_string(),
            },
        )
        .map(|_| ())
}

fn absorb_lattice_packed_validity_metadata<F, T>(
    transcript: &mut T,
    layout: &PackedWitnessLayout,
    statements: &[LatticePackedValidityStatement],
) where
    F: Field,
    T: Transcript<Challenge = F>,
{
    transcript.append(&Label(b"AkitaPackedValidity"));
    transcript.append(&LabelWithCount(
        b"ak_val_layout",
        layout.digest.len() as u64,
    ));
    transcript.append_bytes(&layout.digest);
    transcript.append(&U64Word(layout.dimension as u64));
    transcript.append(&U64Word(layout.cells as u64));
    transcript.append(&LabelWithCount(b"ak_val_stmts", statements.len() as u64));
    for (index, statement) in statements.iter().enumerate() {
        let family = akita_packed_family_id(&statement.requirement.family).physical_ref();
        transcript.append(&U64Word(index as u64));
        transcript.append(&U64Word(family.namespace));
        transcript.append(&U64Word(family.id));
        transcript.append(&U64Word(family.index));
        transcript.append(&U64Word(statement.requirement.limbs as u64));
        transcript.append(&U64Word(statement.requirement.alphabet_size as u64));
        transcript.append(&U64Word(statement.num_vars as u64));
        transcript.append(&U64Word(statement.degree as u64));
        transcript.append(&U64Word(validity_statement_kind_tag(statement.kind)));
        match statement.requirement.kind {
            LatticePackedValidityKind::ExactOneHot => {
                transcript.append(&U64Word(0));
                transcript.append(&U64Word(0));
            }
            LatticePackedValidityKind::OptionalOneHot => {
                transcript.append(&U64Word(1));
                transcript.append(&U64Word(0));
            }
            LatticePackedValidityKind::BooleanIndicator { symbol } => {
                transcript.append(&U64Word(2));
                transcript.append(&U64Word(symbol as u64));
            }
            LatticePackedValidityKind::FusedIncrementCanonicalZero => {
                transcript.append(&U64Word(3));
                transcript.append(&U64Word(0));
            }
            LatticePackedValidityKind::BytecodeStoreRdDisjoint => {
                transcript.append(&U64Word(4));
                transcript.append(&U64Word(0));
            }
            LatticePackedValidityKind::FieldElementCanonicalBytes {
                byte_width,
                modulus,
            } => {
                transcript.append(&U64Word(5));
                transcript.append(&U64Word(byte_width as u64));
                transcript.append(&U64Word((modulus & u64::MAX as u128) as u64));
                transcript.append(&U64Word((modulus >> u64::BITS) as u64));
            }
        }
    }
}

fn validity_statement_kind_tag(kind: LatticePackedValidityStatementKind) -> u64 {
    match kind {
        LatticePackedValidityStatementKind::CellBooleanity => 0,
        LatticePackedValidityStatementKind::ExactOneHotRowSum => 1,
        LatticePackedValidityStatementKind::OptionalOneHotRowSum => 2,
        LatticePackedValidityStatementKind::BooleanIndicator => 3,
        LatticePackedValidityStatementKind::FusedIncrementCanonicalZero => 4,
        LatticePackedValidityStatementKind::BytecodeStoreRdDisjoint => 5,
        LatticePackedValidityStatementKind::FieldElementCanonicalBytes => 6,
    }
}

fn validity_statement_value_from_openings<F>(
    statement: &LatticePackedValidityStatement,
    openings: &[F],
) -> Result<F, VerifierError>
where
    F: Field,
{
    let value = match statement.kind {
        LatticePackedValidityStatementKind::CellBooleanity
        | LatticePackedValidityStatementKind::ExactOneHotRowSum
        | LatticePackedValidityStatementKind::OptionalOneHotRowSum
        | LatticePackedValidityStatementKind::BooleanIndicator => {
            validity_violation(statement.kind, openings[0])
        }
        LatticePackedValidityStatementKind::FusedIncrementCanonicalZero => openings
            .iter()
            .copied()
            .fold(F::one(), |acc, opening| acc * opening),
        LatticePackedValidityStatementKind::BytecodeStoreRdDisjoint => openings[0] * openings[1],
        LatticePackedValidityStatementKind::FieldElementCanonicalBytes => {
            field_element_canonical_value_from_openings(statement, openings)?
        }
    };
    Ok(value)
}

pub(crate) fn field_element_canonical_value_from_openings<F>(
    statement: &LatticePackedValidityStatement,
    openings: &[F],
) -> Result<F, VerifierError>
where
    F: Field,
{
    let byte_width = canonical_field_byte_width(&statement.requirement)?;
    let factors = field_element_canonical_factors(&statement.requirement)?;
    if openings.len() != factors.len() {
        return Err(invalid_lattice_config(format!(
            "field-element canonical-byte statement expects {} openings, got {}",
            factors.len(),
            openings.len()
        )));
    }

    let mut equality = vec![None; byte_width];
    let mut range = vec![None; byte_width];
    for (factor, opening) in factors.iter().zip(openings.iter().copied()) {
        match factor {
            FieldCanonicalFactor::Eq { byte_index, .. } => equality[*byte_index] = Some(opening),
            FieldCanonicalFactor::Range { byte_index, .. } => range[*byte_index] = Some(opening),
        }
    }

    let mut equal_higher_bytes = F::one();
    let mut invalid = F::zero();
    for byte_index in (0..byte_width).rev() {
        if let Some(range_opening) = range[byte_index] {
            invalid += equal_higher_bytes * range_opening;
        }
        if byte_index > 0 {
            let Some(equality_opening) = equality[byte_index] else {
                return Err(invalid_lattice_config(format!(
                    "field-element canonical-byte statement is missing equality factor for byte {byte_index}",
                )));
            };
            equal_higher_bytes *= equality_opening;
        }
    }
    Ok(invalid)
}

fn validity_violation<F>(kind: LatticePackedValidityStatementKind, opening: F) -> F
where
    F: Field,
{
    match kind {
        LatticePackedValidityStatementKind::CellBooleanity
        | LatticePackedValidityStatementKind::BooleanIndicator
        | LatticePackedValidityStatementKind::OptionalOneHotRowSum => {
            opening * (opening - F::one())
        }
        LatticePackedValidityStatementKind::ExactOneHotRowSum => {
            let difference = opening - F::one();
            difference * difference
        }
        LatticePackedValidityStatementKind::FusedIncrementCanonicalZero
        | LatticePackedValidityStatementKind::BytecodeStoreRdDisjoint
        | LatticePackedValidityStatementKind::FieldElementCanonicalBytes => opening,
    }
}

fn validity_factor_physical_view<F>(
    layout: &PackedWitnessLayout,
    statement: &LatticePackedValidityStatement,
    point: &[F],
    factor: usize,
) -> Result<PhysicalView<F>, VerifierError>
where
    F: Field,
{
    if statement.kind == LatticePackedValidityStatementKind::FusedIncrementCanonicalZero {
        return fused_increment_canonical_zero_physical_view(layout, point, factor);
    }
    if statement.kind == LatticePackedValidityStatementKind::BytecodeStoreRdDisjoint {
        return bytecode_store_rd_disjoint_physical_view(layout, statement, point, factor);
    }
    if statement.kind == LatticePackedValidityStatementKind::FieldElementCanonicalBytes {
        return field_element_canonical_physical_view(layout, statement, point, factor);
    }
    if factor != 0 {
        return Err(invalid_lattice_config(format!(
            "packed validity statement {:?} has no opening factor {factor}",
            statement.kind
        )));
    }
    let family_id = akita_packed_family_id(&statement.requirement.family);
    let shape = validity_statement_shape(layout, statement, &family_id)?;
    let point_parts = split_validity_point(statement.kind, point, shape)?;
    let limb_weights = EqPolynomial::<F>::evals(point_parts.limb, None);
    let family = family_id.physical_ref();
    let terms = match statement.kind {
        LatticePackedValidityStatementKind::CellBooleanity => {
            let symbol_weights = EqPolynomial::<F>::evals(point_parts.symbol, None);
            let mut terms = Vec::with_capacity(shape.limbs * shape.alphabet_size);
            for (limb, limb_weight) in limb_weights.iter().copied().enumerate() {
                for (symbol, symbol_weight) in symbol_weights.iter().copied().enumerate() {
                    terms.push(
                        PackedLinearTerm::new(limb_weight * symbol_weight, family, limb, symbol)
                            .with_row_point(point_parts.row.to_vec()),
                    );
                }
            }
            terms
        }
        LatticePackedValidityStatementKind::ExactOneHotRowSum
        | LatticePackedValidityStatementKind::OptionalOneHotRowSum => {
            let mut terms = Vec::with_capacity(shape.limbs * shape.alphabet_size);
            for (limb, limb_weight) in limb_weights.iter().copied().enumerate() {
                for symbol in 0..shape.alphabet_size {
                    terms.push(
                        PackedLinearTerm::new(limb_weight, family, limb, symbol)
                            .with_row_point(point_parts.row.to_vec()),
                    );
                }
            }
            terms
        }
        LatticePackedValidityStatementKind::BooleanIndicator => {
            let LatticePackedValidityKind::BooleanIndicator { symbol } = statement.requirement.kind
            else {
                return Err(invalid_lattice_config(
                    "boolean-indicator validity statement has non-indicator requirement",
                ));
            };
            let mut terms = Vec::with_capacity(shape.limbs);
            for (limb, limb_weight) in limb_weights.iter().copied().enumerate() {
                terms.push(
                    PackedLinearTerm::new(limb_weight, family, limb, symbol)
                        .with_row_point(point_parts.row.to_vec()),
                );
            }
            terms
        }
        LatticePackedValidityStatementKind::FusedIncrementCanonicalZero => {
            return fused_increment_canonical_zero_physical_view(layout, point, factor);
        }
        LatticePackedValidityStatementKind::BytecodeStoreRdDisjoint => {
            return bytecode_store_rd_disjoint_physical_view(layout, statement, point, factor);
        }
        LatticePackedValidityStatementKind::FieldElementCanonicalBytes => {
            return field_element_canonical_physical_view(layout, statement, point, factor);
        }
    };

    Ok(PhysicalView::PackedLinear {
        layout_digest: layout.digest,
        terms,
    })
}

fn fused_increment_canonical_zero_physical_view<F>(
    layout: &PackedWitnessLayout,
    point: &[F],
    factor: usize,
) -> Result<PhysicalView<F>, VerifierError>
where
    F: Field,
{
    let (family_id, symbol) = fused_increment_canonical_zero_factor(factor)?;
    let family = layout.family(&family_id).ok_or_else(|| {
        invalid_lattice_config(format!(
            "fused increment canonical-zero factor requires {family_id:?}"
        ))
    })?;
    if family.limbs != 1 {
        return Err(invalid_lattice_config(format!(
            "fused increment canonical-zero factor {family_id:?} must have one limb"
        )));
    }
    let rows = family.domain.rows().map_err(|error| {
        invalid_lattice_config(format!(
            "fused increment canonical-zero factor {family_id:?} has invalid row domain: {error}"
        ))
    })?;
    let row_vars = power_of_two_log(rows, "fused increment canonical-zero row count")?;
    if point.len() != row_vars {
        return Err(VerifierError::AkitaPackedValiditySumcheckFailed {
            reason: format!(
                "fused increment canonical-zero point has {} variables but statement requires {row_vars}",
                point.len()
            ),
        });
    }

    Ok(PhysicalView::PackedLinear {
        layout_digest: layout.digest,
        terms: vec![
            PackedLinearTerm::new(F::one(), family_id.physical_ref(), 0, symbol)
                .with_row_point(point.to_vec()),
        ],
    })
}

fn field_element_canonical_physical_view<F>(
    layout: &PackedWitnessLayout,
    statement: &LatticePackedValidityStatement,
    point: &[F],
    factor: usize,
) -> Result<PhysicalView<F>, VerifierError>
where
    F: Field,
{
    let factors = field_element_canonical_factors(&statement.requirement)?;
    let factor = factors.get(factor).cloned().ok_or_else(|| {
        invalid_lattice_config(format!(
            "field-element canonical-byte statement has no opening factor {factor}"
        ))
    })?;
    let (family_id, limb) = match &factor {
        FieldCanonicalFactor::Eq { family, limb, .. }
        | FieldCanonicalFactor::Range { family, limb, .. } => (family.clone(), *limb),
    };
    let family = layout.family(&family_id).ok_or_else(|| {
        invalid_lattice_config(format!(
            "field-element canonical-byte factor requires {family_id:?}"
        ))
    })?;
    if limb >= family.limbs || family.alphabet.size() != 256 {
        return Err(invalid_lattice_config(format!(
            "field-element canonical-byte factor {family_id:?} must be a byte family",
        )));
    }
    let rows = family.domain.rows().map_err(|error| {
        invalid_lattice_config(format!(
            "field-element canonical-byte factor {family_id:?} has invalid row domain: {error}"
        ))
    })?;
    let row_vars = power_of_two_log(rows, "field-element canonical-byte row count")?;
    if point.len() != row_vars {
        return Err(VerifierError::AkitaPackedValiditySumcheckFailed {
            reason: format!(
                "field-element canonical-byte point has {} variables but statement requires {row_vars}",
                point.len()
            ),
        });
    }

    let terms = match factor {
        FieldCanonicalFactor::Eq { symbol, .. } => {
            vec![
                PackedLinearTerm::new(F::one(), family_id.physical_ref(), limb, symbol)
                    .with_row_point(point.to_vec()),
            ]
        }
        FieldCanonicalFactor::Range { start_symbol, .. } => (start_symbol..256)
            .map(|symbol| {
                PackedLinearTerm::new(F::one(), family_id.physical_ref(), limb, symbol)
                    .with_row_point(point.to_vec())
            })
            .collect(),
    };

    Ok(PhysicalView::PackedLinear {
        layout_digest: layout.digest,
        terms,
    })
}

fn fused_increment_canonical_zero_factor(
    factor: usize,
) -> Result<(PackedFamilyId, usize), VerifierError> {
    if factor == 0 {
        return Ok((PackedFamilyId::IncSign, 1));
    }
    let byte_index = factor - 1;
    if byte_index < FUSED_INCREMENT_BYTE_LIMBS {
        return Ok((PackedFamilyId::IncByte { index: byte_index }, 0));
    }
    Err(invalid_lattice_config(format!(
        "fused increment canonical-zero has no opening factor {factor}"
    )))
}

fn bytecode_store_rd_disjoint_physical_view<F>(
    layout: &PackedWitnessLayout,
    statement: &LatticePackedValidityStatement,
    point: &[F],
    factor: usize,
) -> Result<PhysicalView<F>, VerifierError>
where
    F: Field,
{
    let chunk = bytecode_store_rd_disjoint_chunk(&statement.requirement)?;
    let store_id = PackedFamilyId::BytecodeCircuitFlag {
        chunk,
        flag: CircuitFlags::Store as usize,
    };
    let store = layout.family(&store_id).ok_or_else(|| {
        invalid_lattice_config(format!(
            "bytecode Store/Rd disjointness requires {store_id:?}"
        ))
    })?;
    let rows = store.domain.rows().map_err(|error| {
        invalid_lattice_config(format!(
            "bytecode Store/Rd disjointness row domain is invalid: {error}"
        ))
    })?;
    let row_vars = power_of_two_log(rows, "bytecode Store/Rd disjointness row count")?;
    if point.len() != row_vars {
        return Err(VerifierError::AkitaPackedValiditySumcheckFailed {
            reason: format!(
                "bytecode Store/Rd disjointness point has {} variables but statement requires {row_vars}",
                point.len()
            ),
        });
    }

    let terms = match factor {
        0 => vec![
            PackedLinearTerm::new(F::one(), store_id.physical_ref(), 0, 1)
                .with_row_point(point.to_vec()),
        ],
        1 => {
            let rd_id = PackedFamilyId::BytecodeRegisterSelector { chunk, selector: 2 };
            let rd = layout.family(&rd_id).ok_or_else(|| {
                invalid_lattice_config(format!("bytecode Store/Rd disjointness requires {rd_id:?}"))
            })?;
            if rd.domain != store.domain || rd.limbs != 1 {
                return Err(invalid_lattice_config(
                    "bytecode Store/Rd disjointness rd selector layout mismatch",
                ));
            }
            (0..rd.alphabet.size())
                .map(|symbol| {
                    PackedLinearTerm::new(F::one(), rd_id.physical_ref(), 0, symbol)
                        .with_row_point(point.to_vec())
                })
                .collect()
        }
        _ => {
            return Err(invalid_lattice_config(format!(
                "bytecode Store/Rd disjointness has no opening factor {factor}"
            )));
        }
    };

    Ok(PhysicalView::PackedLinear {
        layout_digest: layout.digest,
        terms,
    })
}

#[derive(Clone, Copy)]
struct ValidityStatementShape {
    row_vars: usize,
    limb_vars: usize,
    symbol_vars: usize,
    limbs: usize,
    alphabet_size: usize,
}

struct ValidityPointParts<'a, F> {
    row: &'a [F],
    limb: &'a [F],
    symbol: &'a [F],
}

fn validity_statement_shape(
    layout: &PackedWitnessLayout,
    statement: &LatticePackedValidityStatement,
    family_id: &PackedFamilyId,
) -> Result<ValidityStatementShape, VerifierError> {
    let family = layout.family(family_id).ok_or_else(|| {
        invalid_lattice_config(format!(
            "packed validity statement references missing family {family_id:?}"
        ))
    })?;
    if family.limbs != statement.requirement.limbs {
        return Err(invalid_lattice_config(format!(
            "packed validity family {family_id:?} limb count mismatch: layout has {}, statement has {}",
            family.limbs, statement.requirement.limbs
        )));
    }
    if family.alphabet.size() != statement.requirement.alphabet_size {
        return Err(invalid_lattice_config(format!(
            "packed validity family {family_id:?} alphabet mismatch: layout has {}, statement has {}",
            family.alphabet.size(),
            statement.requirement.alphabet_size
        )));
    }
    let rows = family.domain.rows().map_err(|error| {
        invalid_lattice_config(format!(
            "packed validity family {family_id:?} has invalid row domain: {error}"
        ))
    })?;
    let row_vars = power_of_two_log(rows, "packed validity row count")?;
    let limb_vars = power_of_two_log(statement.requirement.limbs, "packed validity limb count")?;
    let symbol_vars = power_of_two_log(
        statement.requirement.alphabet_size,
        "packed validity alphabet size",
    )?;
    Ok(ValidityStatementShape {
        row_vars,
        limb_vars,
        symbol_vars,
        limbs: statement.requirement.limbs,
        alphabet_size: statement.requirement.alphabet_size,
    })
}

fn split_validity_point<F>(
    kind: LatticePackedValidityStatementKind,
    point: &[F],
    shape: ValidityStatementShape,
) -> Result<ValidityPointParts<'_, F>, VerifierError>
where
    F: Field,
{
    let expected = match kind {
        LatticePackedValidityStatementKind::CellBooleanity => {
            shape.row_vars + shape.limb_vars + shape.symbol_vars
        }
        LatticePackedValidityStatementKind::ExactOneHotRowSum
        | LatticePackedValidityStatementKind::OptionalOneHotRowSum
        | LatticePackedValidityStatementKind::BooleanIndicator => shape.row_vars + shape.limb_vars,
        LatticePackedValidityStatementKind::FusedIncrementCanonicalZero
        | LatticePackedValidityStatementKind::BytecodeStoreRdDisjoint
        | LatticePackedValidityStatementKind::FieldElementCanonicalBytes => shape.row_vars,
    };
    if point.len() != expected {
        return Err(VerifierError::AkitaPackedValiditySumcheckFailed {
            reason: format!(
                "packed validity point has {} variables but statement requires {expected}",
                point.len()
            ),
        });
    }
    let row_end = shape.row_vars;
    let limb_end = row_end + shape.limb_vars;
    let symbol_end = limb_end + shape.symbol_vars;
    let symbol_point = if matches!(kind, LatticePackedValidityStatementKind::CellBooleanity) {
        &point[limb_end..symbol_end]
    } else {
        &[]
    };
    Ok(ValidityPointParts {
        row: &point[..row_end],
        limb: &point[row_end..limb_end],
        symbol: symbol_point,
    })
}

pub fn validate_akita_packed_witness_validity_config(
    config: &JoltProtocolConfig,
    precommitted: &PrecommittedSchedule,
) -> Result<(), VerifierError> {
    let requirements = derive_akita_packed_validity_requirements(config, precommitted)?;
    let digest = lattice_packed_validity_digest(&requirements);
    if config.lattice.packed_witness.validity_digest != Some(digest) {
        return Err(invalid_lattice_config(
            "configured Akita packed witness validity digest does not match derived requirements",
        ));
    }
    Ok(())
}

pub fn validate_akita_packed_witness_layout_config(
    config: &JoltProtocolConfig,
    layout: &PackedWitnessLayout,
) -> Result<(), VerifierError> {
    if validate_protocol_config(config)? != PcsFamily::Lattice {
        return Err(invalid_lattice_config(
            "Akita packed witness layout validation requires lattice PCS mode",
        ));
    }
    if config.lattice.packed_witness.layout_digest != Some(layout.digest) {
        return Err(invalid_lattice_config(
            "configured Akita packed witness layout digest does not match derived layout",
        ));
    }
    if config.lattice.packed_witness.d_pack != Some(layout.dimension) {
        return Err(invalid_lattice_config(
            "configured Akita packed witness D_pack does not match derived layout",
        ));
    }
    Ok(())
}

pub fn jolt_lattice_view_formulas<F>(
    logical: &Stage8LogicalManifest<F>,
    log_k_chunk: usize,
    precommitted: &PrecommittedSchedule,
) -> Result<Vec<JoltLatticeViewFormulaWithRowPoint<F>>, VerifierError>
where
    F: Field,
{
    logical
        .openings
        .iter()
        .map(|opening| {
            let (formula, row_point) =
                stage8_lattice_view_formula(opening.id, &opening.point, log_k_chunk, precommitted)?;
            Ok((opening.id, formula, row_point))
        })
        .collect()
}

fn stage8_lattice_view_formula<F>(
    id: Stage8OpeningId,
    point: &[F],
    log_k_chunk: usize,
    precommitted: &PrecommittedSchedule,
) -> Result<(LatticePackedViewFormula<F>, Vec<F>), VerifierError>
where
    F: Field,
{
    match id {
        Stage8OpeningId::Jolt(id) => Ok((
            jolt_lattice_view_formula(id, point, log_k_chunk, precommitted)?,
            jolt_lattice_row_point(id, point, log_k_chunk, precommitted)?,
        )),
        #[cfg(feature = "field-inline")]
        Stage8OpeningId::FieldInline(id) => Ok((
            field_inline_lattice_view_formula(id)?,
            field_inline_lattice_row_point(id, point)?,
        )),
    }
}

pub fn jolt_lattice_physical_manifest<F>(
    logical: &Stage8LogicalManifest<F>,
    layout: &PackedWitnessLayout,
    log_k_chunk: usize,
    precommitted: &PrecommittedSchedule,
) -> Result<Stage8PhysicalManifest<F>, VerifierError>
where
    F: Field,
{
    let formulas = jolt_lattice_view_formulas(logical, log_k_chunk, precommitted)?;
    Stage8PhysicalManifest::from_jolt_lattice_view_formulas(logical, layout, formulas)
        .map_err(lattice_view_resolution_error)
}

pub fn jolt_lattice_physical_manifest_with_validity<F>(
    logical: &Stage8LogicalManifest<F>,
    layout: &PackedWitnessLayout,
    log_k_chunk: usize,
    precommitted: &PrecommittedSchedule,
    validity_requirements: &[LatticePackedValidityRequirement],
) -> Result<Stage8PhysicalManifest<F>, VerifierError>
where
    F: Field,
{
    let formulas = jolt_lattice_view_formulas(logical, log_k_chunk, precommitted)?;
    validate_lattice_view_validity_coverage(&formulas, validity_requirements)?;
    Stage8PhysicalManifest::from_jolt_lattice_view_formulas(logical, layout, formulas)
        .map_err(lattice_view_resolution_error)
}

pub fn validate_lattice_view_validity_coverage<F>(
    formulas: &[JoltLatticeViewFormulaWithRowPoint<F>],
    requirements: &[LatticePackedValidityRequirement],
) -> Result<(), VerifierError> {
    for (id, formula, _) in formulas {
        validate_lattice_formula_validity_coverage(*id, formula, requirements)?;
    }
    Ok(())
}

fn validate_lattice_formula_validity_coverage<F>(
    id: Stage8OpeningId,
    formula: &LatticePackedViewFormula<F>,
    requirements: &[LatticePackedValidityRequirement],
) -> Result<(), VerifierError> {
    match formula {
        LatticePackedViewFormula::Direct {
            family,
            limb,
            symbol,
        } => validate_lattice_term_validity_coverage(id, family, *limb, *symbol, requirements),
        LatticePackedViewFormula::LinearDecoded { terms }
        | LatticePackedViewFormula::ReducedMasked { terms, .. } => {
            for term in terms {
                validate_lattice_term_validity_coverage(
                    id,
                    &term.family,
                    term.limb,
                    term.symbol,
                    requirements,
                )?;
            }
            Ok(())
        }
        LatticePackedViewFormula::MaskedDecoded { relation } => Err(unsupported_lattice_view(
            format!("opening {id:?} still has unresolved masked relation {relation:?}"),
        )),
    }
}

fn validate_lattice_term_validity_coverage(
    id: Stage8OpeningId,
    family: &LatticePackedFamilyId,
    limb: usize,
    symbol: usize,
    requirements: &[LatticePackedValidityRequirement],
) -> Result<(), VerifierError> {
    if core_jolt_ra_family(family) {
        return Ok(());
    }
    let has_value_validity = requirements
        .iter()
        .any(|requirement| requirement_covers_term(requirement, family, limb, symbol));
    if !has_value_validity {
        return Err(unsupported_lattice_view(format!(
            "opening {id:?} uses packed family {family:?} limb {limb} symbol {symbol} without a bound validity requirement"
        )));
    }
    if term_requires_canonical_bytes(family)
        && !requirements
            .iter()
            .any(|requirement| canonical_requirement_covers_term(requirement, family, limb))
    {
        return Err(unsupported_lattice_view(format!(
            "opening {id:?} uses field-byte packed family {family:?} limb {limb} without a bound canonical-byte validity requirement"
        )));
    }
    if term_requires_fused_increment_canonical_zero(family)
        && !requirements
            .iter()
            .any(fused_increment_canonical_zero_requirement)
    {
        return Err(unsupported_lattice_view(format!(
            "opening {id:?} uses fused increment family {family:?} without a bound canonical-zero validity requirement"
        )));
    }
    Ok(())
}

fn core_jolt_ra_family(family: &LatticePackedFamilyId) -> bool {
    matches!(
        family,
        LatticePackedFamilyId::InstructionRa { .. }
            | LatticePackedFamilyId::BytecodeRa { .. }
            | LatticePackedFamilyId::RamRa { .. }
    )
}

fn requirement_covers_term(
    requirement: &LatticePackedValidityRequirement,
    family: &LatticePackedFamilyId,
    limb: usize,
    symbol: usize,
) -> bool {
    if &requirement.family != family || limb >= requirement.limbs {
        return false;
    }
    match requirement.kind {
        LatticePackedValidityKind::ExactOneHot | LatticePackedValidityKind::OptionalOneHot => {
            symbol < requirement.alphabet_size
        }
        LatticePackedValidityKind::BooleanIndicator { symbol: indicator } => {
            symbol == indicator && indicator < requirement.alphabet_size
        }
        LatticePackedValidityKind::FusedIncrementCanonicalZero
        | LatticePackedValidityKind::BytecodeStoreRdDisjoint
        | LatticePackedValidityKind::FieldElementCanonicalBytes { .. } => false,
    }
}

fn term_requires_canonical_bytes(family: &LatticePackedFamilyId) -> bool {
    matches!(
        family,
        LatticePackedFamilyId::FieldRdIncByte { .. }
            | LatticePackedFamilyId::BytecodeImmBytes { .. }
    )
}

fn canonical_requirement_covers_term(
    requirement: &LatticePackedValidityRequirement,
    family: &LatticePackedFamilyId,
    limb: usize,
) -> bool {
    let Ok(byte_width) = canonical_field_byte_width(requirement) else {
        return false;
    };
    match (&requirement.family, family) {
        (
            LatticePackedFamilyId::FieldRdIncByte { index: 0 },
            LatticePackedFamilyId::FieldRdIncByte { index },
        ) => *index < byte_width && limb == 0,
        (
            LatticePackedFamilyId::BytecodeImmBytes { chunk: expected },
            LatticePackedFamilyId::BytecodeImmBytes { chunk },
        ) => expected == chunk && limb < byte_width,
        _ => false,
    }
}

fn term_requires_fused_increment_canonical_zero(family: &LatticePackedFamilyId) -> bool {
    matches!(
        family,
        LatticePackedFamilyId::IncByte { .. } | LatticePackedFamilyId::IncSign
    )
}

fn fused_increment_canonical_zero_requirement(
    requirement: &LatticePackedValidityRequirement,
) -> bool {
    matches!(
        requirement.kind,
        LatticePackedValidityKind::FusedIncrementCanonicalZero
    )
}

fn jolt_lattice_row_point<F>(
    id: JoltOpeningId,
    point: &[F],
    log_k_chunk: usize,
    precommitted: &PrecommittedSchedule,
) -> Result<Vec<F>, VerifierError>
where
    F: Field,
{
    match id {
        JoltOpeningId::Polynomial {
            polynomial: JoltPolynomialId::Committed(polynomial),
            relation,
        } => committed_lattice_row_point(polynomial, relation, point, log_k_chunk, precommitted),
        JoltOpeningId::TrustedAdvice {
            relation: JoltRelationId::AdviceClaimReduction,
        }
        | JoltOpeningId::UntrustedAdvice {
            relation: JoltRelationId::AdviceClaimReduction,
        } => Ok(point.to_vec()),
        id if id == fused_increment_magnitude_opening()
            || id == fused_increment_sign_opening()
            || id == fused_increment_inactive_magnitude_opening()
            || id == fused_increment_inactive_sign_opening() =>
        {
            Ok(point.to_vec())
        }
        id if id == fused_increment_bytecode_source_opening(LatticeFusedIncrementTarget::Ram)
            || id == fused_increment_bytecode_source_opening(LatticeFusedIncrementTarget::Rd)
            || id
                == fused_increment_inactive_bytecode_source_opening(
                    LatticeFusedIncrementTarget::Ram,
                )
            || id
                == fused_increment_inactive_bytecode_source_opening(
                    LatticeFusedIncrementTarget::Rd,
                ) =>
        {
            bytecode_address_row_point(point, precommitted)
        }
        id if id == fused_increment_source_opening(LatticeFusedIncrementTarget::Ram)
            || id == fused_increment_source_opening(LatticeFusedIncrementTarget::Rd)
            || id == fused_increment_inactive_source_opening(LatticeFusedIncrementTarget::Ram)
            || id == fused_increment_inactive_source_opening(LatticeFusedIncrementTarget::Rd) =>
        {
            Ok(point.to_vec())
        }
        _ => Err(unsupported_lattice_view(format!(
            "final opening {id:?} has no supported lattice packed row point"
        ))),
    }
}

fn committed_lattice_row_point<F>(
    polynomial: JoltCommittedPolynomial,
    relation: JoltRelationId,
    point: &[F],
    log_k_chunk: usize,
    precommitted: &PrecommittedSchedule,
) -> Result<Vec<F>, VerifierError>
where
    F: Field,
{
    match (polynomial, relation) {
        (
            JoltCommittedPolynomial::InstructionRa(_)
            | JoltCommittedPolynomial::BytecodeRa(_)
            | JoltCommittedPolynomial::RamRa(_),
            JoltRelationId::HammingWeightClaimReduction,
        )
        | (
            JoltCommittedPolynomial::BytecodeRa(_),
            JoltRelationId::FusedIncrementSourceLink
            | JoltRelationId::FusedIncrementInactiveSourceLink,
        ) => ra_row_point(point, log_k_chunk),
        (
            JoltCommittedPolynomial::RamInc | JoltCommittedPolynomial::RdInc,
            JoltRelationId::IncClaimReduction,
        ) => Ok(point.to_vec()),
        (
            JoltCommittedPolynomial::ProgramImageInit,
            JoltRelationId::ProgramImageClaimReduction,
        )
        | (
            JoltCommittedPolynomial::TrustedAdvice | JoltCommittedPolynomial::UntrustedAdvice,
            JoltRelationId::AdviceClaimReduction,
        ) => Ok(point.to_vec()),
        (JoltCommittedPolynomial::BytecodeChunk(index), JoltRelationId::BytecodeClaimReduction) => {
            let layout = precommitted.bytecode.as_ref().ok_or_else(|| {
                unsupported_lattice_view(format!(
                    "BytecodeChunk({index}) row point requires committed-bytecode layout"
                ))
            })?;
            if index >= layout.chunk_count() {
                return Err(unsupported_lattice_view(format!(
                    "BytecodeChunk({index}) is outside committed-bytecode chunk count {}",
                    layout.chunk_count()
                )));
            }
            bytecode_chunk_row_point(point, layout.trace_order(), layout.log_bytecode_chunk_size())
        }
        _ => Err(unsupported_lattice_view(format!(
            "committed polynomial {polynomial:?} under relation {relation:?} has no supported lattice packed row point"
        ))),
    }
}

fn ra_row_point<F>(point: &[F], log_k_chunk: usize) -> Result<Vec<F>, VerifierError>
where
    F: Field,
{
    if point.len() < log_k_chunk {
        return Err(unsupported_lattice_view(format!(
            "RA lattice opening point has {} variables but needs at least {log_k_chunk}",
            point.len()
        )));
    }
    Ok(point[log_k_chunk..].to_vec())
}

fn bytecode_address_row_point<F>(
    point: &[F],
    precommitted: &PrecommittedSchedule,
) -> Result<Vec<F>, VerifierError>
where
    F: Field,
{
    let layout = precommitted.bytecode.as_ref().ok_or_else(|| {
        unsupported_lattice_view(
            "bytecode-derived packed row point requires committed-bytecode layout",
        )
    })?;
    layout
        .split_address_point(point)
        .map(|address| address.r_bc)
        .map_err(|error| unsupported_lattice_view(error.to_string()))
}

fn bytecode_chunk_row_point<F>(
    point: &[F],
    trace_order: TracePolynomialOrder,
    log_bytecode: usize,
) -> Result<Vec<F>, VerifierError>
where
    F: Field,
{
    if point.len() < log_bytecode {
        return Err(unsupported_lattice_view(format!(
            "bytecode chunk opening point has {} variables but needs at least {log_bytecode}",
            point.len()
        )));
    }
    let lane_vars = point.len() - log_bytecode;
    match trace_order {
        TracePolynomialOrder::CycleMajor => Ok(point[lane_vars..].to_vec()),
        TracePolynomialOrder::AddressMajor => Ok(point[..log_bytecode].to_vec()),
    }
}

#[cfg(feature = "field-inline")]
fn field_inline_lattice_row_point<F>(
    id: FieldInlineOpeningId,
    point: &[F],
) -> Result<Vec<F>, VerifierError>
where
    F: Field,
{
    if id == field_increments::field_rd_inc_reduced_opening() {
        return Ok(point.to_vec());
    }
    Err(unsupported_lattice_view(format!(
        "field-inline opening {id:?} has no supported lattice packed row point"
    )))
}

#[cfg(feature = "field-inline")]
fn field_inline_lattice_view_formula<F>(
    id: FieldInlineOpeningId,
) -> Result<LatticePackedViewFormula<F>, VerifierError>
where
    F: Field,
{
    if id == field_increments::field_rd_inc_reduced_opening() {
        return Ok(field_lattice::field_rd_inc_lattice_view_formula(
            AkitaField::NUM_BYTES,
        ));
    }
    Err(unsupported_lattice_view(format!(
        "field-inline opening {id:?} has no supported lattice packed view"
    )))
}

pub fn jolt_lattice_view_formula<F>(
    id: JoltOpeningId,
    point: &[F],
    log_k_chunk: usize,
    precommitted: &PrecommittedSchedule,
) -> Result<LatticePackedViewFormula<F>, VerifierError>
where
    F: Field,
{
    match id {
        JoltOpeningId::Polynomial {
            polynomial: JoltPolynomialId::Committed(polynomial),
            relation,
        } => committed_lattice_view_formula(polynomial, relation, point, log_k_chunk, precommitted),
        JoltOpeningId::TrustedAdvice {
            relation: JoltRelationId::AdviceClaimReduction,
        } => Ok(advice_lattice_view_formula(JoltAdviceKind::Trusted)),
        JoltOpeningId::UntrustedAdvice {
            relation: JoltRelationId::AdviceClaimReduction,
        } => Ok(advice_lattice_view_formula(JoltAdviceKind::Untrusted)),
        id if id == fused_increment_magnitude_opening() => {
            Ok(fused_increment_magnitude_lattice_view_formula())
        }
        id if id == fused_increment_inactive_magnitude_opening() => {
            Ok(fused_increment_magnitude_lattice_view_formula())
        }
        id if id == fused_increment_sign_opening() => {
            Ok(fused_increment_sign_lattice_view_formula())
        }
        id if id == fused_increment_inactive_sign_opening() => {
            Ok(fused_increment_sign_lattice_view_formula())
        }
        id if id == fused_increment_bytecode_source_opening(LatticeFusedIncrementTarget::Ram) => {
            fused_increment_bytecode_source_lattice_view_formula(
                LatticeFusedIncrementTarget::Ram,
                point,
                precommitted,
            )
        }
        id if id == fused_increment_bytecode_source_opening(LatticeFusedIncrementTarget::Rd) => {
            fused_increment_bytecode_source_lattice_view_formula(
                LatticeFusedIncrementTarget::Rd,
                point,
                precommitted,
            )
        }
        id if id
            == fused_increment_inactive_bytecode_source_opening(
                LatticeFusedIncrementTarget::Ram,
            ) =>
        {
            fused_increment_bytecode_source_lattice_view_formula(
                LatticeFusedIncrementTarget::Ram,
                point,
                precommitted,
            )
        }
        id if id
            == fused_increment_inactive_bytecode_source_opening(
                LatticeFusedIncrementTarget::Rd,
            ) =>
        {
            fused_increment_bytecode_source_lattice_view_formula(
                LatticeFusedIncrementTarget::Rd,
                point,
                precommitted,
            )
        }
        id if id == fused_increment_source_opening(LatticeFusedIncrementTarget::Ram)
            || id == fused_increment_source_opening(LatticeFusedIncrementTarget::Rd)
            || id == fused_increment_inactive_source_opening(LatticeFusedIncrementTarget::Ram)
            || id == fused_increment_inactive_source_opening(LatticeFusedIncrementTarget::Rd) =>
        {
            Err(unsupported_lattice_view(
                "fused increment source outputs require a bytecode-derived packed view relation",
            ))
        }
        _ => Err(unsupported_lattice_view(format!(
            "final opening {id:?} has no supported lattice packed view"
        ))),
    }
}

pub fn akita_packed_family_id(family: &LatticePackedFamilyId) -> PackedFamilyId {
    match family {
        LatticePackedFamilyId::InstructionRa { index } => {
            PackedFamilyId::InstructionRa { index: *index }
        }
        LatticePackedFamilyId::BytecodeRa { index } => PackedFamilyId::BytecodeRa { index: *index },
        LatticePackedFamilyId::RamRa { index } => PackedFamilyId::RamRa { index: *index },
        LatticePackedFamilyId::IncByte { index } => PackedFamilyId::IncByte { index: *index },
        LatticePackedFamilyId::IncSign => PackedFamilyId::IncSign,
        LatticePackedFamilyId::FieldRdIncByte { index } => {
            PackedFamilyId::FieldRdIncByte { index: *index }
        }
        LatticePackedFamilyId::FieldRdIncSign => PackedFamilyId::FieldRdIncSign,
        LatticePackedFamilyId::AdviceBytes { kind, index } => PackedFamilyId::AdviceBytes {
            kind: akita_advice_kind(*kind),
            index: *index,
        },
        LatticePackedFamilyId::BytecodeChunk { index } => {
            PackedFamilyId::BytecodeChunk { index: *index }
        }
        LatticePackedFamilyId::BytecodeRegisterSelector { chunk, selector } => {
            PackedFamilyId::BytecodeRegisterSelector {
                chunk: *chunk,
                selector: *selector,
            }
        }
        LatticePackedFamilyId::BytecodeCircuitFlag { chunk, flag } => {
            PackedFamilyId::BytecodeCircuitFlag {
                chunk: *chunk,
                flag: *flag,
            }
        }
        LatticePackedFamilyId::BytecodeInstructionFlag { chunk, flag } => {
            PackedFamilyId::BytecodeInstructionFlag {
                chunk: *chunk,
                flag: *flag,
            }
        }
        LatticePackedFamilyId::BytecodeLookupSelector { chunk } => {
            PackedFamilyId::BytecodeLookupSelector { chunk: *chunk }
        }
        LatticePackedFamilyId::BytecodeRafFlag { chunk } => {
            PackedFamilyId::BytecodeRafFlag { chunk: *chunk }
        }
        LatticePackedFamilyId::BytecodeUnexpandedPcBytes { chunk } => {
            PackedFamilyId::BytecodeUnexpandedPcBytes { chunk: *chunk }
        }
        LatticePackedFamilyId::BytecodeImmBytes { chunk } => {
            PackedFamilyId::BytecodeImmBytes { chunk: *chunk }
        }
        LatticePackedFamilyId::ProgramImageInit => PackedFamilyId::ProgramImageInit,
        LatticePackedFamilyId::Custom { namespace, index } => PackedFamilyId::Custom {
            namespace: *namespace,
            index: *index,
        },
    }
}

fn committed_lattice_view_formula<F>(
    polynomial: JoltCommittedPolynomial,
    relation: JoltRelationId,
    point: &[F],
    log_k_chunk: usize,
    precommitted: &PrecommittedSchedule,
) -> Result<LatticePackedViewFormula<F>, VerifierError>
where
    F: Field,
{
    match (polynomial, relation) {
        (
            JoltCommittedPolynomial::InstructionRa(index),
            JoltRelationId::HammingWeightClaimReduction,
        ) => ra_lattice_view_formula(
            LatticePackedFamilyId::InstructionRa { index },
            point,
            log_k_chunk,
        ),
        (
            JoltCommittedPolynomial::BytecodeRa(index),
            JoltRelationId::HammingWeightClaimReduction
            | JoltRelationId::FusedIncrementSourceLink
            | JoltRelationId::FusedIncrementInactiveSourceLink,
        ) => {
            ra_lattice_view_formula(
                LatticePackedFamilyId::BytecodeRa { index },
                point,
                log_k_chunk,
            )
        }
        (JoltCommittedPolynomial::RamRa(index), JoltRelationId::HammingWeightClaimReduction) => {
            ra_lattice_view_formula(
                LatticePackedFamilyId::RamRa { index },
                point,
                log_k_chunk,
            )
        }
        (JoltCommittedPolynomial::RamInc | JoltCommittedPolynomial::RdInc, JoltRelationId::IncClaimReduction) => {
            Ok(LatticePackedViewFormula::masked_decoded(
                JoltRelationId::FusedIncrementTranslation,
            ))
        }
        (JoltCommittedPolynomial::ProgramImageInit, JoltRelationId::ProgramImageClaimReduction) => {
            Ok(LatticePackedViewFormula::linear_decoded(
                little_endian_byte_decode_terms(LatticePackedFamilyId::ProgramImageInit, 8),
            ))
        }
        (JoltCommittedPolynomial::TrustedAdvice, JoltRelationId::AdviceClaimReduction) => {
            Ok(advice_lattice_view_formula(JoltAdviceKind::Trusted))
        }
        (JoltCommittedPolynomial::UntrustedAdvice, JoltRelationId::AdviceClaimReduction) => {
            Ok(advice_lattice_view_formula(JoltAdviceKind::Untrusted))
        }
        (JoltCommittedPolynomial::BytecodeChunk(index), JoltRelationId::BytecodeClaimReduction) => {
            let layout = precommitted.bytecode.as_ref().ok_or_else(|| {
                unsupported_lattice_view(format!(
                    "BytecodeChunk({index}) lattice view requires committed-bytecode layout"
                ))
            })?;
            if index >= layout.chunk_count() {
                return Err(unsupported_lattice_view(format!(
                    "BytecodeChunk({index}) is outside committed-bytecode chunk count {}",
                    layout.chunk_count()
                )));
            }
            bytecode_chunk_lattice_view_formula(
                index,
                point,
                layout.trace_order(),
                layout.log_bytecode_chunk_size(),
                AkitaField::NUM_BYTES,
            )
            .map_err(|error| {
                unsupported_lattice_view(format!(
                    "BytecodeChunk({index}) lattice view formula failed: {error}"
                ))
            })
        }
        _ => Err(unsupported_lattice_view(format!(
            "committed polynomial {polynomial:?} under relation {relation:?} has no supported lattice packed view"
        ))),
    }
}

fn ra_lattice_view_formula<F>(
    family: LatticePackedFamilyId,
    point: &[F],
    log_k_chunk: usize,
) -> Result<LatticePackedViewFormula<F>, VerifierError>
where
    F: Field,
{
    if log_k_chunk == 0 {
        return Err(unsupported_lattice_view(
            "RA lattice view requires a nonzero one-hot chunk size",
        ));
    }
    if point.len() < log_k_chunk {
        return Err(unsupported_lattice_view(format!(
            "RA lattice opening point has {} variables but needs at least {log_k_chunk}",
            point.len()
        )));
    }
    Ok(LatticePackedViewFormula::linear_decoded(
        weighted_symbol_terms(
            family,
            0,
            EqPolynomial::<F>::evals(&point[..log_k_chunk], None),
        ),
    ))
}

fn advice_lattice_view_formula<F>(kind: JoltAdviceKind) -> LatticePackedViewFormula<F>
where
    F: Field,
{
    LatticePackedViewFormula::linear_decoded(byte_decode_terms(
        LatticePackedFamilyId::AdviceBytes { kind, index: 0 },
        0,
    ))
}

fn fused_increment_bytecode_source_lattice_view_formula<F>(
    target: LatticeFusedIncrementTarget,
    point: &[F],
    precommitted: &PrecommittedSchedule,
) -> Result<LatticePackedViewFormula<F>, VerifierError>
where
    F: Field,
{
    let layout = precommitted.bytecode.as_ref().ok_or_else(|| {
        unsupported_lattice_view(
            "fused increment bytecode source view requires committed-bytecode layout",
        )
    })?;
    let address = layout
        .split_address_point(point)
        .map_err(|error| unsupported_lattice_view(error.to_string()))?;
    let mut terms = Vec::new();
    for (chunk, weight) in address.chunk_rbc_weights.into_iter().enumerate() {
        let formula = fused_increment_source_lattice_view_formula(target, chunk);
        extend_scaled_lattice_terms(&mut terms, formula, weight)?;
    }
    Ok(LatticePackedViewFormula::linear_decoded(terms))
}

fn extend_scaled_lattice_terms<F>(
    terms: &mut Vec<LatticePackedViewTerm<F>>,
    formula: LatticePackedViewFormula<F>,
    scale: F,
) -> Result<(), VerifierError>
where
    F: Field,
{
    match formula {
        LatticePackedViewFormula::Direct {
            family,
            limb,
            symbol,
        } => terms.push(LatticePackedViewTerm::new(scale, family, limb, symbol)),
        LatticePackedViewFormula::LinearDecoded {
            terms: formula_terms,
        } => {
            terms.extend(formula_terms.into_iter().map(|term| {
                LatticePackedViewTerm::new(
                    scale * term.coefficient,
                    term.family,
                    term.limb,
                    term.symbol,
                )
            }));
        }
        LatticePackedViewFormula::ReducedMasked { .. }
        | LatticePackedViewFormula::MaskedDecoded { .. } => {
            return Err(unsupported_lattice_view(
                "fused increment bytecode source view must lower to direct or linear decoded terms",
            ));
        }
    }
    Ok(())
}

pub fn akita_packed_view_formula<F>(
    formula: &LatticePackedViewFormula<F>,
) -> Result<PackedViewFormula<F>, PackedViewError>
where
    F: Field,
{
    match formula {
        LatticePackedViewFormula::Direct {
            family,
            limb,
            symbol,
        } => Ok(PackedViewFormula::direct(
            akita_packed_family_id(family),
            *limb,
            *symbol,
        )),
        LatticePackedViewFormula::LinearDecoded { terms } => Ok(PackedViewFormula::linear_decoded(
            terms
                .iter()
                .map(|term| {
                    PackedViewTerm::new(
                        term.coefficient,
                        akita_packed_family_id(&term.family),
                        term.limb,
                        term.symbol,
                    )
                })
                .collect(),
        )),
        LatticePackedViewFormula::ReducedMasked { terms, .. } => {
            Ok(PackedViewFormula::reduced_masked(
                terms
                    .iter()
                    .map(|term| {
                        PackedViewTerm::new(
                            term.coefficient,
                            akita_packed_family_id(&term.family),
                            term.limb,
                            term.symbol,
                        )
                    })
                    .collect(),
            ))
        }
        LatticePackedViewFormula::MaskedDecoded { .. } => {
            Err(PackedViewError::MaskedViewRequiresTranslation)
        }
    }
}

fn akita_advice_kind(kind: JoltAdviceKind) -> PackedAdviceKind {
    match kind {
        JoltAdviceKind::Trusted => PackedAdviceKind::Trusted,
        JoltAdviceKind::Untrusted => PackedAdviceKind::Untrusted,
    }
}

fn advice_family(
    kind: JoltAdviceKind,
    layout: &AdviceClaimReductionLayout,
) -> Result<PackedFamilySpec, VerifierError> {
    let packed_kind = akita_advice_kind(kind);
    let requirement = advice_bytes_validity_requirement(kind);
    Ok(PackedFamilySpec::direct(
        akita_packed_family_id(&requirement.family),
        PackedFactDomain::AdviceBytes {
            kind: packed_kind,
            log_bytes: layout.advice_shape().total_vars() + 3,
        },
        requirement.limbs,
        packed_alphabet_with_size(requirement.alphabet_size)?,
    ))
}

fn extend_validity_requirement_families(
    specs: &mut Vec<PackedFamilySpec>,
    requirements: &[LatticePackedValidityRequirement],
    domain: PackedFactDomain,
) -> Result<(), VerifierError> {
    for requirement in requirements {
        if matches!(
            requirement.kind,
            LatticePackedValidityKind::FusedIncrementCanonicalZero
                | LatticePackedValidityKind::BytecodeStoreRdDisjoint
                | LatticePackedValidityKind::FieldElementCanonicalBytes { .. }
        ) {
            continue;
        }
        specs.push(PackedFamilySpec::direct(
            akita_packed_family_id(&requirement.family),
            domain,
            requirement.limbs,
            packed_alphabet_with_size(requirement.alphabet_size)?,
        ));
    }
    Ok(())
}

#[cfg(feature = "field-inline")]
fn field_rd_inc_validity_requirements() -> Vec<LatticePackedValidityRequirement> {
    let mut requirements = field_lattice::field_rd_inc_validity_requirements(AkitaField::NUM_BYTES);
    requirements.push(field_lattice::field_rd_inc_canonical_bytes_requirement(
        AkitaField::NUM_BYTES,
        AKITA_FIELD_MODULUS,
    ));
    requirements
}

#[cfg(not(feature = "field-inline"))]
fn field_rd_inc_validity_requirements() -> Vec<LatticePackedValidityRequirement> {
    (0..AkitaField::NUM_BYTES)
        .map(|index| {
            LatticePackedValidityRequirement::exact_one_hot(
                LatticePackedFamilyId::FieldRdIncByte { index },
                1,
                256,
            )
        })
        .collect()
}

#[cfg(feature = "field-inline")]
fn extend_field_rd_inc_families(
    specs: &mut Vec<PackedFamilySpec>,
    domain: PackedFactDomain,
) -> Result<(), VerifierError> {
    extend_validity_requirement_families(specs, &field_rd_inc_validity_requirements(), domain)
}

#[cfg(not(feature = "field-inline"))]
fn extend_field_rd_inc_families(
    specs: &mut Vec<PackedFamilySpec>,
    domain: PackedFactDomain,
) -> Result<(), VerifierError> {
    extend_validity_requirement_families(specs, &field_rd_inc_validity_requirements(), domain)
}

fn extend_bytecode_families(
    specs: &mut Vec<PackedFamilySpec>,
    chunk: usize,
    log_bytecode: usize,
) -> Result<(), VerifierError> {
    let domain = PackedFactDomain::BytecodeRows { log_bytecode };
    extend_validity_requirement_families(
        specs,
        &bytecode_validity_requirements(chunk, AkitaField::NUM_BYTES),
        domain,
    )
}

fn program_image_family(
    layout: &ProgramImageClaimReductionLayout,
) -> Result<PackedFamilySpec, VerifierError> {
    let requirement = program_image_validity_requirement();
    Ok(PackedFamilySpec::direct(
        akita_packed_family_id(&requirement.family),
        PackedFactDomain::ProgramImageWords {
            log_words: power_of_two_log(layout.padded_len_words(), "program image length")?,
        },
        requirement.limbs,
        packed_alphabet_with_size(requirement.alphabet_size)?,
    ))
}

fn require_advice_layout(
    precommitted: &PrecommittedSchedule,
    kind: JoltAdviceKind,
) -> Result<&AdviceClaimReductionLayout, VerifierError> {
    precommitted.advice(kind).ok_or_else(|| {
        invalid_precommitted_schedule(format!(
            "lattice advice mode requires a {kind:?} advice claim-reduction layout",
        ))
    })
}

fn one_hot_alphabet(log_k_chunk: usize) -> Result<PackedAlphabet, VerifierError> {
    match log_k_chunk {
        0 => Err(invalid_lattice_config(
            "lattice one-hot chunk size must be nonzero",
        )),
        1 => Ok(PackedAlphabet::Bit),
        8 => Ok(PackedAlphabet::Byte),
        bits if bits < usize::BITS as usize => Ok(PackedAlphabet::Fixed {
            size: 1usize << bits,
        }),
        _ => Err(invalid_lattice_config(
            "lattice one-hot chunk size is too large",
        )),
    }
}

fn packed_alphabet_with_size(size: usize) -> Result<PackedAlphabet, VerifierError> {
    match size {
        0 => Err(invalid_lattice_config(
            "packed validity requirement alphabet size must be nonzero",
        )),
        2 => Ok(PackedAlphabet::Bit),
        256 => Ok(PackedAlphabet::Byte),
        size => Ok(PackedAlphabet::Fixed { size }),
    }
}

fn power_of_two_log(value: usize, name: &'static str) -> Result<usize, VerifierError> {
    if value.is_power_of_two() {
        Ok(value.trailing_zeros() as usize)
    } else {
        Err(invalid_lattice_config(format!(
            "{name} must be a power of two, got {value}",
        )))
    }
}

fn invalid_lattice_config(reason: impl Into<String>) -> VerifierError {
    VerifierError::InvalidProtocolConfig {
        reason: reason.into(),
    }
}

fn invalid_precommitted_schedule(reason: impl Into<String>) -> VerifierError {
    VerifierError::InvalidPrecommittedSchedule {
        reason: reason.into(),
    }
}

fn unsupported_lattice_view(reason: impl Into<String>) -> VerifierError {
    VerifierError::FinalOpeningBatchFailed {
        reason: format!("unsupported lattice final opening view: {}", reason.into()),
    }
}

fn lattice_view_resolution_error(error: PackedViewError) -> VerifierError {
    VerifierError::FinalOpeningBatchFailed {
        reason: format!("lattice packed view resolution failed: {error}"),
    }
}

#[cfg(test)]
mod tests {
    #![expect(clippy::panic, reason = "tests fail loudly on setup errors")]

    use super::super::outputs::{Stage8LogicalManifest, Stage8LogicalOpening, Stage8OpeningId};
    use super::*;
    use crate::{
        config::{IncrementCommitmentMode, PackedWitnessConfig, ProgramMode},
        stages::CommittedProgramSchedule,
    };
    use jolt_claims::protocols::jolt::formulas::claim_reductions::bytecode;
    use jolt_claims::protocols::jolt::{
        byte_decode_terms, formulas::dimensions::REGISTER_ADDRESS_BITS, JoltCommittedPolynomial,
        JoltOpeningId, JoltRelationId, LatticePackedFamilyId, LatticePackedViewFormula,
        LatticePackedViewTerm, TracePolynomialOrder,
    };
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_lookup_tables::{LookupTableKind, XLEN as RISCV_XLEN};
    use jolt_openings::{PackedLinearTerm, PhysicalView};
    use jolt_poly::{EqPolynomial, Point};
    use jolt_riscv::CircuitFlags;
    use jolt_sumcheck::{BatchedEvaluationClaim, EvaluationClaim};

    fn lattice_config() -> JoltProtocolConfig {
        let mut config = JoltProtocolConfig::for_zk(false).with_pcs_family(PcsFamily::Lattice);
        config.lattice.program_mode = ProgramMode::Committed;
        config.lattice.increment_mode = IncrementCommitmentMode::FusedOneHot;
        config.lattice.packed_witness = PackedWitnessConfig {
            layout_digest: Some([0; 32]),
            d_pack: Some(0),
            validity_digest: Some([0; 32]),
            field_rd_inc_family: false,
            trusted_advice_family: false,
            untrusted_advice_family: false,
        };
        #[cfg(feature = "field-inline")]
        {
            config.lattice.field_inline.enabled = true;
            config.lattice.packed_witness.field_rd_inc_family = true;
        }
        config
    }

    fn precommitted_schedule(trusted_max_advice_bytes: Option<usize>) -> PrecommittedSchedule {
        precommitted_schedule_with_advice(trusted_max_advice_bytes, None)
    }

    fn precommitted_schedule_with_advice(
        trusted_max_advice_bytes: Option<usize>,
        untrusted_max_advice_bytes: Option<usize>,
    ) -> PrecommittedSchedule {
        PrecommittedSchedule::new(
            TracePolynomialOrder::CycleMajor,
            2,
            8,
            trusted_max_advice_bytes,
            untrusted_max_advice_bytes,
            Some(CommittedProgramSchedule {
                bytecode_len: 16,
                bytecode_chunk_count: 2,
                program_image_len_words: 4,
                program_image_start_index: 0,
            }),
        )
        .unwrap_or_else(|error| panic!("precommitted schedule should build: {error}"))
    }

    fn precommitted_schedule_without_committed_program() -> PrecommittedSchedule {
        PrecommittedSchedule::new(TracePolynomialOrder::CycleMajor, 2, 8, None, None, None)
            .unwrap_or_else(|error| panic!("precommitted schedule should build: {error}"))
    }

    fn ra_layout() -> JoltRaPolynomialLayout {
        JoltRaPolynomialLayout::new(2, 1, 1)
            .unwrap_or_else(|error| panic!("RA layout should build: {error}"))
    }

    fn logical_manifest_for_stage8(
        id: Stage8OpeningId,
        point: Vec<Fr>,
    ) -> Stage8LogicalManifest<Fr> {
        Stage8LogicalManifest {
            openings: vec![Stage8LogicalOpening {
                id,
                point,
                claim: Some(Fr::from_u64(2)),
                scale: Fr::from_u64(3),
            }],
            pcs_opening_point: Point::high_to_low(vec![Fr::from_u64(4)]),
        }
    }

    fn logical_manifest(id: JoltOpeningId, point: Vec<Fr>) -> Stage8LogicalManifest<Fr> {
        logical_manifest_for_stage8(Stage8OpeningId::from(id), point)
    }

    fn bytecode_chunk_opening_point() -> (Vec<Fr>, Vec<Fr>) {
        let lane_vars = bytecode::committed_lane_vars();
        let lane_point = (1..=lane_vars as u64).map(Fr::from_u64).collect::<Vec<_>>();
        let mut point = lane_point.clone();
        point.extend([Fr::from_u64(101), Fr::from_u64(103), Fr::from_u64(107)]);
        let lane_weights = EqPolynomial::<Fr>::evals(&lane_point, None);
        (point, lane_weights)
    }

    fn linear_decoded_terms(
        formula: &LatticePackedViewFormula<Fr>,
    ) -> &[LatticePackedViewTerm<Fr>] {
        match formula {
            LatticePackedViewFormula::LinearDecoded { terms } => terms,
            _ => panic!("expected linear decoded formula"),
        }
    }

    fn find_lattice_term(
        terms: &[LatticePackedViewTerm<Fr>],
        family: LatticePackedFamilyId,
        limb: usize,
        symbol: usize,
    ) -> &LatticePackedViewTerm<Fr> {
        terms
            .iter()
            .find(|term| term.family == family && term.limb == limb && term.symbol == symbol)
            .unwrap_or_else(|| panic!("missing lattice term"))
    }

    fn find_physical_term(
        terms: &[PackedLinearTerm<Fr>],
        family: PackedFamilyId,
        limb: usize,
        symbol: usize,
    ) -> &PackedLinearTerm<Fr> {
        let family = family.physical_ref();
        terms
            .iter()
            .find(|term| term.family == family && term.limb == limb && term.symbol == symbol)
            .unwrap_or_else(|| panic!("missing physical term"))
    }

    fn fused_increment_validity_layout(log_t: usize) -> PackedWitnessLayout {
        let trace = PackedFactDomain::TraceRows { log_t };
        let mut specs = (0..FUSED_INCREMENT_BYTE_LIMBS)
            .map(|index| {
                PackedFamilySpec::direct(
                    PackedFamilyId::IncByte { index },
                    trace,
                    1,
                    PackedAlphabet::Byte,
                )
            })
            .collect::<Vec<_>>();
        specs.push(PackedFamilySpec::direct(
            PackedFamilyId::IncSign,
            trace,
            1,
            PackedAlphabet::Bit,
        ));
        PackedWitnessLayout::new(specs)
            .unwrap_or_else(|error| panic!("fused increment layout should build: {error}"))
    }

    fn bytecode_source_validity_layout(chunk: usize, log_bytecode: usize) -> PackedWitnessLayout {
        let domain = PackedFactDomain::BytecodeRows { log_bytecode };
        PackedWitnessLayout::new([
            PackedFamilySpec::direct(
                PackedFamilyId::BytecodeCircuitFlag {
                    chunk,
                    flag: CircuitFlags::Store as usize,
                },
                domain,
                1,
                PackedAlphabet::Bit,
            ),
            PackedFamilySpec::direct(
                PackedFamilyId::BytecodeRegisterSelector { chunk, selector: 2 },
                domain,
                1,
                PackedAlphabet::Fixed {
                    size: 1 << REGISTER_ADDRESS_BITS,
                },
            ),
        ])
        .unwrap_or_else(|error| panic!("bytecode source layout should build: {error}"))
    }

    #[test]
    fn derive_layout_includes_base_lattice_families() {
        let config = lattice_config();
        let layout = derive_akita_packed_witness_layout(
            &config,
            2,
            8,
            ra_layout(),
            &precommitted_schedule(None),
        )
        .unwrap_or_else(|error| panic!("layout derivation should succeed: {error}"));

        assert!(layout
            .family(&PackedFamilyId::InstructionRa { index: 0 })
            .is_some());
        assert!(layout.family(&PackedFamilyId::RamRa { index: 0 }).is_some());
        assert!(layout
            .family(&PackedFamilyId::IncByte { index: 7 })
            .is_some());
        assert!(layout.family(&PackedFamilyId::IncSign).is_some());
        assert!(layout
            .family(&PackedFamilyId::BytecodeRegisterSelector {
                chunk: 0,
                selector: 0,
            })
            .is_some());
        assert!(layout
            .family(&PackedFamilyId::BytecodeCircuitFlag { chunk: 0, flag: 0 })
            .is_some());
        let lookup_selector = layout
            .family(&PackedFamilyId::BytecodeLookupSelector { chunk: 0 })
            .unwrap_or_else(|| panic!("bytecode lookup selector family should be present"));
        assert_eq!(
            lookup_selector.alphabet,
            PackedAlphabet::Fixed {
                size: LookupTableKind::<RISCV_XLEN>::COUNT.next_power_of_two(),
            }
        );
        assert!(layout
            .family(&PackedFamilyId::BytecodeUnexpandedPcBytes { chunk: 0 })
            .is_some());
        assert!(layout
            .family(&PackedFamilyId::BytecodeChunk { index: 0 })
            .is_none());
        assert!(layout.family(&PackedFamilyId::ProgramImageInit).is_some());
        assert_eq!(layout.audit().d_pack, layout.dimension);

        let mut matching_config = lattice_config();
        matching_config.lattice.packed_witness.layout_digest = Some(layout.digest);
        matching_config.lattice.packed_witness.d_pack = Some(layout.dimension);

        validate_akita_packed_witness_layout_config(&matching_config, &layout)
            .unwrap_or_else(|error| panic!("layout config should validate: {error}"));
    }

    #[test]
    fn validate_validity_config_rejects_mismatched_digest() {
        let mut config = lattice_config();
        let schedule = precommitted_schedule(None);
        let requirements = derive_akita_packed_validity_requirements(&config, &schedule)
            .unwrap_or_else(|error| panic!("validity requirements should derive: {error}"));
        let digest = lattice_packed_validity_digest(&requirements);
        config.lattice.packed_witness.validity_digest = Some(digest);

        validate_akita_packed_witness_validity_config(&config, &schedule).unwrap_or_else(|error| {
            panic!("validity config should match derived requirements: {error}")
        });

        let mut wrong_digest = digest;
        wrong_digest[0] ^= 1;
        config.lattice.packed_witness.validity_digest = Some(wrong_digest);

        assert!(matches!(
            validate_akita_packed_witness_validity_config(&config, &schedule),
            Err(VerifierError::InvalidProtocolConfig { reason })
                if reason.contains("validity digest")
        ));
    }

    #[test]
    fn derive_validity_statements_matches_requirement_semantics() {
        let layout = PackedWitnessLayout::new([
            PackedFamilySpec::direct(
                PackedFamilyId::ProgramImageInit,
                PackedFactDomain::ProgramImageWords { log_words: 2 },
                8,
                PackedAlphabet::Byte,
            ),
            PackedFamilySpec::direct(
                PackedFamilyId::BytecodeLookupSelector { chunk: 0 },
                PackedFactDomain::BytecodeRows { log_bytecode: 3 },
                1,
                PackedAlphabet::Fixed { size: 8 },
            ),
            PackedFamilySpec::direct(
                PackedFamilyId::IncSign,
                PackedFactDomain::TraceRows { log_t: 4 },
                1,
                PackedAlphabet::Bit,
            ),
        ])
        .unwrap_or_else(|error| panic!("layout should build: {error}"));
        let requirements = vec![
            LatticePackedValidityRequirement::exact_one_hot(
                LatticePackedFamilyId::ProgramImageInit,
                8,
                256,
            ),
            LatticePackedValidityRequirement::optional_one_hot(
                LatticePackedFamilyId::BytecodeLookupSelector { chunk: 0 },
                1,
                8,
            ),
            LatticePackedValidityRequirement::boolean_indicator(
                LatticePackedFamilyId::IncSign,
                1,
                2,
                1,
            ),
        ];

        let statements = derive_akita_packed_validity_statements(&layout, &requirements)
            .unwrap_or_else(|error| panic!("validity statements should derive: {error}"));

        assert_eq!(statements.len(), 5);
        assert_eq!(
            statements[0].kind,
            LatticePackedValidityStatementKind::CellBooleanity
        );
        assert_eq!(statements[0].num_vars, 2 + 3 + 8);
        assert_eq!(
            statements[1].kind,
            LatticePackedValidityStatementKind::ExactOneHotRowSum
        );
        assert_eq!(statements[1].num_vars, 2 + 3);
        assert_eq!(
            statements[2].kind,
            LatticePackedValidityStatementKind::CellBooleanity
        );
        assert_eq!(statements[2].num_vars, 3 + 3);
        assert_eq!(
            statements[3].kind,
            LatticePackedValidityStatementKind::OptionalOneHotRowSum
        );
        assert_eq!(statements[3].num_vars, 3);
        assert_eq!(
            statements[4].kind,
            LatticePackedValidityStatementKind::BooleanIndicator
        );
        assert_eq!(statements[4].num_vars, 4);
        assert!(statements.iter().all(|statement| statement.degree == 3));
    }

    #[test]
    fn derive_validity_statements_adds_fused_increment_canonical_zero() {
        let layout = fused_increment_validity_layout(4);
        let requirement = LatticePackedValidityRequirement::fused_increment_canonical_zero();

        let statements =
            derive_akita_packed_validity_statements(&layout, std::slice::from_ref(&requirement))
                .unwrap_or_else(|error| {
                    panic!("canonical-zero validity statement should derive: {error}")
                });

        assert_eq!(statements.len(), 1);
        assert_eq!(statements[0].requirement, requirement);
        assert_eq!(
            statements[0].kind,
            LatticePackedValidityStatementKind::FusedIncrementCanonicalZero
        );
        assert_eq!(statements[0].num_vars, 4);
        assert_eq!(statements[0].degree, FUSED_INCREMENT_BYTE_LIMBS + 2);
        assert_eq!(
            lattice_packed_validity_opening_count(&statements),
            FUSED_INCREMENT_BYTE_LIMBS + 1
        );
    }

    #[test]
    fn derive_validity_statements_adds_bytecode_store_rd_disjointness() {
        let chunk = 2;
        let layout = bytecode_source_validity_layout(chunk, 5);
        let requirement = LatticePackedValidityRequirement::bytecode_store_rd_disjoint(chunk);

        let statements =
            derive_akita_packed_validity_statements(&layout, std::slice::from_ref(&requirement))
                .unwrap_or_else(|error| {
                    panic!("Store/Rd disjointness validity statement should derive: {error}")
                });

        assert_eq!(statements.len(), 1);
        assert_eq!(statements[0].requirement, requirement);
        assert_eq!(
            statements[0].kind,
            LatticePackedValidityStatementKind::BytecodeStoreRdDisjoint
        );
        assert_eq!(statements[0].num_vars, 5);
        assert_eq!(statements[0].degree, 3);
        assert_eq!(lattice_packed_validity_opening_count(&statements), 2);
    }

    #[test]
    fn derive_validity_statements_adds_field_element_canonical_bytes() {
        let layout = PackedWitnessLayout::new((0..2).map(|index| {
            PackedFamilySpec::direct(
                PackedFamilyId::FieldRdIncByte { index },
                PackedFactDomain::TraceRows { log_t: 3 },
                1,
                PackedAlphabet::Byte,
            )
        }))
        .unwrap_or_else(|error| panic!("layout should build: {error}"));
        let requirement = LatticePackedValidityRequirement::field_element_canonical_bytes(
            LatticePackedFamilyId::FieldRdIncByte { index: 0 },
            2,
            257,
        );

        let statements =
            derive_akita_packed_validity_statements(&layout, std::slice::from_ref(&requirement))
                .unwrap_or_else(|error| {
                    panic!("field canonical-byte validity statement should derive: {error}")
                });

        assert_eq!(statements.len(), 1);
        assert_eq!(statements[0].requirement, requirement);
        assert_eq!(
            statements[0].kind,
            LatticePackedValidityStatementKind::FieldElementCanonicalBytes
        );
        assert_eq!(statements[0].num_vars, 3);
        assert_eq!(statements[0].degree, 2);
        assert_eq!(lattice_packed_validity_opening_count(&statements), 3);
    }

    #[test]
    fn derive_validity_statements_adds_bytecode_imm_canonical_bytes() {
        let chunk = 2;
        let layout = PackedWitnessLayout::new([PackedFamilySpec::direct(
            PackedFamilyId::BytecodeImmBytes { chunk },
            PackedFactDomain::BytecodeRows { log_bytecode: 3 },
            2,
            PackedAlphabet::Byte,
        )])
        .unwrap_or_else(|error| panic!("layout should build: {error}"));
        let requirement = bytecode_imm_canonical_bytes_requirement(chunk, 2, 257);

        let statements =
            derive_akita_packed_validity_statements(&layout, std::slice::from_ref(&requirement))
                .unwrap_or_else(|error| {
                    panic!("bytecode imm canonical-byte validity statement should derive: {error}")
                });

        assert_eq!(statements.len(), 1);
        assert_eq!(statements[0].requirement, requirement);
        assert_eq!(
            statements[0].kind,
            LatticePackedValidityStatementKind::FieldElementCanonicalBytes
        );
        assert_eq!(statements[0].num_vars, 3);
        assert_eq!(statements[0].degree, 2);
        assert_eq!(lattice_packed_validity_opening_count(&statements), 3);
    }

    #[test]
    fn validity_batch_builder_lowers_cell_booleanity_to_packed_terms() {
        let layout = PackedWitnessLayout::new([PackedFamilySpec::direct(
            PackedFamilyId::ProgramImageInit,
            PackedFactDomain::ProgramImageWords { log_words: 1 },
            2,
            PackedAlphabet::Fixed { size: 4 },
        )])
        .unwrap_or_else(|error| panic!("layout should build: {error}"));
        let requirement = LatticePackedValidityRequirement::exact_one_hot(
            LatticePackedFamilyId::ProgramImageInit,
            2,
            4,
        );
        let statement = LatticePackedValidityStatement {
            requirement,
            kind: LatticePackedValidityStatementKind::CellBooleanity,
            num_vars: 4,
            degree: 3,
        };
        let point = vec![
            Fr::from_u64(2),
            Fr::from_u64(3),
            Fr::from_u64(5),
            Fr::from_u64(7),
        ];
        let eq_point = vec![
            Fr::from_u64(11),
            Fr::from_u64(13),
            Fr::from_u64(17),
            Fr::from_u64(19),
        ];
        let batching_coefficient = Fr::from_u64(23);
        let opening_claim = Fr::from_u64(29);
        let reduction = BatchedEvaluationClaim {
            reduction: EvaluationClaim::new(point.clone(), Fr::from_u64(31)),
            batching_coefficients: vec![batching_coefficient],
            max_num_vars: 4,
            max_degree: 3,
        };

        let batch = build_lattice_packed_validity_batch(
            &layout,
            std::slice::from_ref(&statement),
            99_u64,
            std::slice::from_ref(&eq_point),
            &reduction,
            &[opening_claim],
        )
        .unwrap_or_else(|error| panic!("validity batch should build: {error}"));

        let expected_eq = try_eq_mle(&point, &eq_point)
            .unwrap_or_else(|error| panic!("eq mask should evaluate: {error}"));
        assert_eq!(
            batch.expected_final_claim,
            batching_coefficient * expected_eq * opening_claim * (opening_claim - Fr::from_u64(1))
        );
        assert_eq!(batch.statement.claims.len(), 1);
        assert_eq!(batch.statement.claims[0].commitment, 99);
        assert_eq!(batch.statement.claims[0].claim, opening_claim);

        let PhysicalView::PackedLinear {
            layout_digest,
            terms,
        } = &batch.statement.claims[0].view
        else {
            panic!("validity opening should use a packed linear view");
        };
        assert_eq!(layout_digest, &layout.digest);
        assert_eq!(terms.len(), 8);
        let limb_weights = EqPolynomial::<Fr>::evals(&point[1..2], None);
        let symbol_weights = EqPolynomial::<Fr>::evals(&point[2..], None);
        let term = find_physical_term(terms, PackedFamilyId::ProgramImageInit, 1, 3);
        assert_eq!(term.row_point, vec![Fr::from_u64(2)]);
        assert_eq!(term.coefficient, limb_weights[1] * symbol_weights[3]);
    }

    #[test]
    fn validity_batch_builder_lowers_fused_increment_canonical_zero_factors() {
        let layout = fused_increment_validity_layout(4);
        let statement = LatticePackedValidityStatement {
            requirement: LatticePackedValidityRequirement::fused_increment_canonical_zero(),
            kind: LatticePackedValidityStatementKind::FusedIncrementCanonicalZero,
            num_vars: 4,
            degree: FUSED_INCREMENT_BYTE_LIMBS + 2,
        };
        let point = vec![
            Fr::from_u64(2),
            Fr::from_u64(3),
            Fr::from_u64(5),
            Fr::from_u64(7),
        ];
        let eq_point = vec![
            Fr::from_u64(11),
            Fr::from_u64(13),
            Fr::from_u64(17),
            Fr::from_u64(19),
        ];
        let batching_coefficient = Fr::from_u64(23);
        let opening_claims = (0..=FUSED_INCREMENT_BYTE_LIMBS)
            .map(|index| Fr::from_u64(29 + index as u64))
            .collect::<Vec<_>>();
        let reduction = BatchedEvaluationClaim {
            reduction: EvaluationClaim::new(point.clone(), Fr::from_u64(31)),
            batching_coefficients: vec![batching_coefficient],
            max_num_vars: 4,
            max_degree: FUSED_INCREMENT_BYTE_LIMBS + 2,
        };

        let batch = build_lattice_packed_validity_batch(
            &layout,
            std::slice::from_ref(&statement),
            99_u64,
            std::slice::from_ref(&eq_point),
            &reduction,
            &opening_claims,
        )
        .unwrap_or_else(|error| panic!("canonical-zero batch should build: {error}"));

        let expected_eq = try_eq_mle(&point, &eq_point)
            .unwrap_or_else(|error| panic!("eq mask should evaluate: {error}"));
        let opening_product = opening_claims
            .iter()
            .copied()
            .fold(Fr::from_u64(1), |acc, opening| acc * opening);
        assert_eq!(
            batch.expected_final_claim,
            batching_coefficient * expected_eq * opening_product
        );
        assert_eq!(batch.statement.claims.len(), FUSED_INCREMENT_BYTE_LIMBS + 1);

        let PhysicalView::PackedLinear {
            layout_digest,
            terms,
        } = &batch.statement.claims[0].view
        else {
            panic!("canonical-zero sign factor should use a packed linear view");
        };
        assert_eq!(layout_digest, &layout.digest);
        assert_eq!(terms.len(), 1);
        assert_eq!(terms[0].family, PackedFamilyId::IncSign.physical_ref());
        assert_eq!(terms[0].limb, 0);
        assert_eq!(terms[0].symbol, 1);
        assert_eq!(terms[0].coefficient, Fr::from_u64(1));
        assert_eq!(terms[0].row_point, point);

        for index in 0..FUSED_INCREMENT_BYTE_LIMBS {
            let PhysicalView::PackedLinear { terms, .. } = &batch.statement.claims[index + 1].view
            else {
                panic!("canonical-zero byte factor should use a packed linear view");
            };
            assert_eq!(terms.len(), 1);
            assert_eq!(
                terms[0].family,
                PackedFamilyId::IncByte { index }.physical_ref()
            );
            assert_eq!(terms[0].limb, 0);
            assert_eq!(terms[0].symbol, 0);
        }
    }

    #[test]
    fn validity_batch_builder_lowers_bytecode_store_rd_disjointness_factors() {
        let chunk = 2;
        let layout = bytecode_source_validity_layout(chunk, 3);
        let statement = LatticePackedValidityStatement {
            requirement: LatticePackedValidityRequirement::bytecode_store_rd_disjoint(chunk),
            kind: LatticePackedValidityStatementKind::BytecodeStoreRdDisjoint,
            num_vars: 3,
            degree: 3,
        };
        let point = vec![Fr::from_u64(2), Fr::from_u64(3), Fr::from_u64(5)];
        let eq_point = vec![Fr::from_u64(11), Fr::from_u64(13), Fr::from_u64(17)];
        let batching_coefficient = Fr::from_u64(19);
        let opening_claims = [Fr::from_u64(23), Fr::from_u64(29)];
        let reduction = BatchedEvaluationClaim {
            reduction: EvaluationClaim::new(point.clone(), Fr::from_u64(31)),
            batching_coefficients: vec![batching_coefficient],
            max_num_vars: 3,
            max_degree: 3,
        };

        let batch = build_lattice_packed_validity_batch(
            &layout,
            std::slice::from_ref(&statement),
            99_u64,
            std::slice::from_ref(&eq_point),
            &reduction,
            &opening_claims,
        )
        .unwrap_or_else(|error| panic!("Store/Rd disjointness batch should build: {error}"));

        let expected_eq = try_eq_mle(&point, &eq_point)
            .unwrap_or_else(|error| panic!("eq mask should evaluate: {error}"));
        assert_eq!(
            batch.expected_final_claim,
            batching_coefficient * expected_eq * opening_claims[0] * opening_claims[1]
        );
        assert_eq!(batch.statement.claims.len(), 2);

        let PhysicalView::PackedLinear { terms, .. } = &batch.statement.claims[0].view else {
            panic!("Store factor should use a packed linear view");
        };
        assert_eq!(terms.len(), 1);
        assert_eq!(
            terms[0].family,
            PackedFamilyId::BytecodeCircuitFlag {
                chunk,
                flag: CircuitFlags::Store as usize
            }
            .physical_ref()
        );
        assert_eq!(terms[0].symbol, 1);
        assert_eq!(terms[0].row_point, point);

        let PhysicalView::PackedLinear { terms, .. } = &batch.statement.claims[1].view else {
            panic!("Rd-present factor should use a packed linear view");
        };
        assert_eq!(terms.len(), 1 << REGISTER_ADDRESS_BITS);
        let term = find_physical_term(
            terms,
            PackedFamilyId::BytecodeRegisterSelector { chunk, selector: 2 },
            0,
            (1 << REGISTER_ADDRESS_BITS) - 1,
        );
        assert_eq!(term.coefficient, Fr::from_u64(1));
        assert_eq!(term.row_point, point);
    }

    #[test]
    fn validity_batch_builder_lowers_field_element_canonical_byte_factors() {
        let layout = PackedWitnessLayout::new((0..2).map(|index| {
            PackedFamilySpec::direct(
                PackedFamilyId::FieldRdIncByte { index },
                PackedFactDomain::TraceRows { log_t: 2 },
                1,
                PackedAlphabet::Byte,
            )
        }))
        .unwrap_or_else(|error| panic!("layout should build: {error}"));
        let requirement = LatticePackedValidityRequirement::field_element_canonical_bytes(
            LatticePackedFamilyId::FieldRdIncByte { index: 0 },
            2,
            257,
        );
        let statement = LatticePackedValidityStatement {
            requirement,
            kind: LatticePackedValidityStatementKind::FieldElementCanonicalBytes,
            num_vars: 2,
            degree: 2,
        };
        let point = vec![Fr::from_u64(2), Fr::from_u64(3)];
        let eq_point = vec![Fr::from_u64(5), Fr::from_u64(7)];
        let batching_coefficient = Fr::from_u64(11);
        let opening_claims = [Fr::from_u64(13), Fr::from_u64(17), Fr::from_u64(19)];
        let reduction = BatchedEvaluationClaim {
            reduction: EvaluationClaim::new(point.clone(), Fr::from_u64(23)),
            batching_coefficients: vec![batching_coefficient],
            max_num_vars: 2,
            max_degree: 2,
        };

        let batch = build_lattice_packed_validity_batch(
            &layout,
            std::slice::from_ref(&statement),
            99_u64,
            std::slice::from_ref(&eq_point),
            &reduction,
            &opening_claims,
        )
        .unwrap_or_else(|error| panic!("field canonical-byte batch should build: {error}"));

        let expected_eq = try_eq_mle(&point, &eq_point)
            .unwrap_or_else(|error| panic!("eq mask should evaluate: {error}"));
        let expected_invalid_indicator = opening_claims[0] + opening_claims[1] * opening_claims[2];
        assert_eq!(
            batch.expected_final_claim,
            batching_coefficient * expected_eq * expected_invalid_indicator
        );
        assert_eq!(batch.statement.claims.len(), 3);

        let PhysicalView::PackedLinear { terms, .. } = &batch.statement.claims[0].view else {
            panic!("high-byte range factor should use a packed linear view");
        };
        assert_eq!(terms.len(), 254);
        assert_eq!(
            terms[0].family,
            PackedFamilyId::FieldRdIncByte { index: 1 }.physical_ref()
        );
        assert_eq!(terms[0].symbol, 2);
        assert_eq!(terms[253].symbol, 255);

        let PhysicalView::PackedLinear { terms, .. } = &batch.statement.claims[1].view else {
            panic!("high-byte equality factor should use a packed linear view");
        };
        assert_eq!(terms.len(), 1);
        assert_eq!(
            terms[0].family,
            PackedFamilyId::FieldRdIncByte { index: 1 }.physical_ref()
        );
        assert_eq!(terms[0].symbol, 1);

        let PhysicalView::PackedLinear { terms, .. } = &batch.statement.claims[2].view else {
            panic!("low-byte range factor should use a packed linear view");
        };
        assert_eq!(terms.len(), 255);
        assert_eq!(
            terms[0].family,
            PackedFamilyId::FieldRdIncByte { index: 0 }.physical_ref()
        );
        assert_eq!(terms[0].symbol, 1);
        assert_eq!(terms[254].symbol, 255);
    }

    #[test]
    fn validity_batch_builder_lowers_bytecode_imm_canonical_byte_factors() {
        let chunk = 2;
        let layout = PackedWitnessLayout::new([PackedFamilySpec::direct(
            PackedFamilyId::BytecodeImmBytes { chunk },
            PackedFactDomain::BytecodeRows { log_bytecode: 2 },
            2,
            PackedAlphabet::Byte,
        )])
        .unwrap_or_else(|error| panic!("layout should build: {error}"));
        let requirement = bytecode_imm_canonical_bytes_requirement(chunk, 2, 257);
        let statement = LatticePackedValidityStatement {
            requirement,
            kind: LatticePackedValidityStatementKind::FieldElementCanonicalBytes,
            num_vars: 2,
            degree: 2,
        };
        let point = vec![Fr::from_u64(2), Fr::from_u64(3)];
        let eq_point = vec![Fr::from_u64(5), Fr::from_u64(7)];
        let batching_coefficient = Fr::from_u64(11);
        let opening_claims = [Fr::from_u64(13), Fr::from_u64(17), Fr::from_u64(19)];
        let reduction = BatchedEvaluationClaim {
            reduction: EvaluationClaim::new(point.clone(), Fr::from_u64(23)),
            batching_coefficients: vec![batching_coefficient],
            max_num_vars: 2,
            max_degree: 2,
        };

        let batch = build_lattice_packed_validity_batch(
            &layout,
            std::slice::from_ref(&statement),
            99_u64,
            std::slice::from_ref(&eq_point),
            &reduction,
            &opening_claims,
        )
        .unwrap_or_else(|error| panic!("bytecode imm canonical-byte batch should build: {error}"));

        let expected_eq = try_eq_mle(&point, &eq_point)
            .unwrap_or_else(|error| panic!("eq mask should evaluate: {error}"));
        let expected_invalid_indicator = opening_claims[0] + opening_claims[1] * opening_claims[2];
        assert_eq!(
            batch.expected_final_claim,
            batching_coefficient * expected_eq * expected_invalid_indicator
        );
        assert_eq!(batch.statement.claims.len(), 3);

        let PhysicalView::PackedLinear { terms, .. } = &batch.statement.claims[0].view else {
            panic!("high-byte range factor should use a packed linear view");
        };
        assert_eq!(terms.len(), 254);
        assert_eq!(
            terms[0].family,
            PackedFamilyId::BytecodeImmBytes { chunk }.physical_ref()
        );
        assert_eq!(terms[0].limb, 1);
        assert_eq!(terms[0].symbol, 2);
        assert_eq!(terms[253].limb, 1);
        assert_eq!(terms[253].symbol, 255);

        let PhysicalView::PackedLinear { terms, .. } = &batch.statement.claims[1].view else {
            panic!("high-byte equality factor should use a packed linear view");
        };
        assert_eq!(terms.len(), 1);
        assert_eq!(
            terms[0].family,
            PackedFamilyId::BytecodeImmBytes { chunk }.physical_ref()
        );
        assert_eq!(terms[0].limb, 1);
        assert_eq!(terms[0].symbol, 1);

        let PhysicalView::PackedLinear { terms, .. } = &batch.statement.claims[2].view else {
            panic!("low-byte range factor should use a packed linear view");
        };
        assert_eq!(terms.len(), 255);
        assert_eq!(
            terms[0].family,
            PackedFamilyId::BytecodeImmBytes { chunk }.physical_ref()
        );
        assert_eq!(terms[0].limb, 0);
        assert_eq!(terms[0].symbol, 1);
        assert_eq!(terms[254].limb, 0);
        assert_eq!(terms[254].symbol, 255);
    }

    #[test]
    fn derive_validity_statements_rejects_layout_requirement_mismatch() {
        let layout = PackedWitnessLayout::new([PackedFamilySpec::direct(
            PackedFamilyId::IncSign,
            PackedFactDomain::TraceRows { log_t: 4 },
            1,
            PackedAlphabet::Bit,
        )])
        .unwrap_or_else(|error| panic!("layout should build: {error}"));
        let requirements = [LatticePackedValidityRequirement::exact_one_hot(
            LatticePackedFamilyId::IncSign,
            8,
            256,
        )];

        assert!(matches!(
            derive_akita_packed_validity_statements(&layout, &requirements),
            Err(VerifierError::InvalidProtocolConfig { reason })
                if reason.contains("limb count mismatch")
        ));
    }

    #[test]
    fn validity_coverage_accepts_bound_decoded_families() {
        let id = Stage8OpeningId::from(JoltOpeningId::committed(
            JoltCommittedPolynomial::ProgramImageInit,
            JoltRelationId::ProgramImageClaimReduction,
        ));
        let formula = LatticePackedViewFormula::linear_decoded(byte_decode_terms::<Fr>(
            LatticePackedFamilyId::ProgramImageInit,
            0,
        ));

        validate_lattice_view_validity_coverage(
            &[(id, formula, vec![Fr::from_u64(1)])],
            &[program_image_validity_requirement()],
        )
        .unwrap_or_else(|error| panic!("bound decoded family should validate: {error}"));
    }

    #[test]
    fn validity_coverage_rejects_unbound_decoded_families() {
        let id = Stage8OpeningId::from(JoltOpeningId::committed(
            JoltCommittedPolynomial::ProgramImageInit,
            JoltRelationId::ProgramImageClaimReduction,
        ));
        let formula = LatticePackedViewFormula::linear_decoded(byte_decode_terms::<Fr>(
            LatticePackedFamilyId::ProgramImageInit,
            0,
        ));

        assert!(matches!(
            validate_lattice_view_validity_coverage(&[(id, formula, Vec::new())], &[]),
            Err(VerifierError::FinalOpeningBatchFailed { reason })
                if reason.contains("without a bound validity requirement")
        ));
    }

    #[test]
    fn validity_coverage_requires_canonical_byte_requirements_for_field_bytes() {
        let id = Stage8OpeningId::from(JoltOpeningId::committed(
            JoltCommittedPolynomial::BytecodeChunk(0),
            JoltRelationId::BytecodeClaimReduction,
        ));
        let formula = LatticePackedViewFormula::linear_decoded(byte_decode_terms::<Fr>(
            LatticePackedFamilyId::BytecodeImmBytes { chunk: 0 },
            0,
        ));
        let one_hot = LatticePackedValidityRequirement::exact_one_hot(
            LatticePackedFamilyId::BytecodeImmBytes { chunk: 0 },
            AkitaField::NUM_BYTES,
            256,
        );
        let canonical =
            bytecode_imm_canonical_bytes_requirement(0, AkitaField::NUM_BYTES, AKITA_FIELD_MODULUS);

        assert!(matches!(
            validate_lattice_view_validity_coverage(
                &[(id, formula.clone(), Vec::new())],
                std::slice::from_ref(&one_hot),
            ),
            Err(VerifierError::FinalOpeningBatchFailed { reason })
                if reason.contains("without a bound canonical-byte validity requirement")
        ));

        validate_lattice_view_validity_coverage(
            &[(id, formula, Vec::new())],
            &[one_hot, canonical],
        )
        .unwrap_or_else(|error| panic!("canonical byte family should validate: {error}"));
    }

    #[test]
    fn validity_coverage_requires_fused_increment_canonical_zero() {
        let id = Stage8OpeningId::from(fused_increment_magnitude_opening());
        let formula = fused_increment_magnitude_lattice_view_formula::<Fr>();
        let without_canonical_zero = fused_increment_validity_requirements()
            .into_iter()
            .filter(|requirement| {
                !matches!(
                    requirement.kind,
                    LatticePackedValidityKind::FusedIncrementCanonicalZero
                )
            })
            .collect::<Vec<_>>();
        let with_canonical_zero = fused_increment_validity_requirements();

        assert!(matches!(
            validate_lattice_view_validity_coverage(
                &[(id, formula.clone(), Vec::new())],
                &without_canonical_zero,
            ),
            Err(VerifierError::FinalOpeningBatchFailed { reason })
                if reason.contains("without a bound canonical-zero validity requirement")
        ));

        validate_lattice_view_validity_coverage(&[(id, formula, Vec::new())], &with_canonical_zero)
            .unwrap_or_else(|error| {
                panic!("fused increment canonical-zero coverage should validate: {error}")
            });
    }

    #[test]
    fn validity_coverage_allows_core_ra_families() {
        let id = Stage8OpeningId::from(JoltOpeningId::committed(
            JoltCommittedPolynomial::InstructionRa(0),
            JoltRelationId::HammingWeightClaimReduction,
        ));
        let formula = LatticePackedViewFormula::linear_decoded(vec![LatticePackedViewTerm::new(
            Fr::from_u64(1),
            LatticePackedFamilyId::InstructionRa { index: 0 },
            0,
            0,
        )]);

        validate_lattice_view_validity_coverage(&[(id, formula, Vec::new())], &[]).unwrap_or_else(
            |error| panic!("core RA family should not need lattice validity: {error}"),
        );
    }

    #[test]
    fn validity_coverage_checks_boolean_indicator_symbol() {
        let id = Stage8OpeningId::from(fused_increment_sign_opening());
        let requirement = LatticePackedValidityRequirement::boolean_indicator(
            LatticePackedFamilyId::IncSign,
            1,
            2,
            1,
        );
        let canonical_zero = LatticePackedValidityRequirement::fused_increment_canonical_zero();

        validate_lattice_view_validity_coverage(
            &[(
                id,
                LatticePackedViewFormula::<Fr>::direct(LatticePackedFamilyId::IncSign, 0, 1),
                Vec::new(),
            )],
            &[requirement.clone(), canonical_zero.clone()],
        )
        .unwrap_or_else(|error| panic!("covered boolean indicator should validate: {error}"));

        assert!(matches!(
            validate_lattice_view_validity_coverage(
                &[(
                    id,
                    LatticePackedViewFormula::<Fr>::direct(
                        LatticePackedFamilyId::IncSign,
                        0,
                        0,
                    ),
                    Vec::new(),
                )],
                &[requirement, canonical_zero],
            ),
            Err(VerifierError::FinalOpeningBatchFailed { reason })
                if reason.contains("without a bound validity requirement")
        ));
    }

    #[test]
    fn derive_layout_uses_fused_increment_validity_requirements() {
        let config = lattice_config();
        let log_t = 3;
        let layout = derive_akita_packed_witness_layout(
            &config,
            log_t,
            8,
            ra_layout(),
            &precommitted_schedule(None),
        )
        .unwrap_or_else(|error| panic!("layout derivation should succeed: {error}"));

        for requirement in fused_increment_validity_requirements() {
            let family_id = akita_packed_family_id(&requirement.family);
            let family = layout
                .family(&family_id)
                .unwrap_or_else(|| panic!("validity family {family_id:?} should be present"));

            assert_eq!(family.domain, PackedFactDomain::TraceRows { log_t });
            assert_eq!(family.limbs, requirement.limbs);
            assert_eq!(family.alphabet.size(), requirement.alphabet_size);
        }
    }

    #[test]
    fn derive_layout_uses_committed_program_validity_requirements() {
        let mut config = lattice_config();
        config.lattice.advice.trusted = true;
        config.lattice.packed_witness.trusted_advice_family = true;
        let precommitted = precommitted_schedule(Some(8));
        let layout = derive_akita_packed_witness_layout(&config, 2, 8, ra_layout(), &precommitted)
            .unwrap_or_else(|error| panic!("layout derivation should succeed: {error}"));
        let bytecode_domain = PackedFactDomain::BytecodeRows {
            log_bytecode: precommitted
                .bytecode
                .as_ref()
                .unwrap_or_else(|| panic!("bytecode layout should exist"))
                .log_bytecode_chunk_size(),
        };
        let validity_requirements =
            derive_akita_packed_validity_requirements(&config, &precommitted)
                .unwrap_or_else(|error| panic!("validity requirements should derive: {error}"));
        assert!(
            validity_requirements.contains(&bytecode_imm_canonical_bytes_requirement(
                0,
                AkitaField::NUM_BYTES,
                AKITA_FIELD_MODULUS,
            ))
        );

        for requirement in bytecode_validity_requirements(0, AkitaField::NUM_BYTES) {
            let family_id = akita_packed_family_id(&requirement.family);
            let family = layout
                .family(&family_id)
                .unwrap_or_else(|| panic!("bytecode validity family {family_id:?} should exist"));

            assert_eq!(family.domain, bytecode_domain);
            assert_eq!(family.limbs, requirement.limbs);
            assert_eq!(family.alphabet.size(), requirement.alphabet_size);
        }

        let advice_requirement = advice_bytes_validity_requirement(JoltAdviceKind::Trusted);
        let advice_family = layout
            .family(&akita_packed_family_id(&advice_requirement.family))
            .unwrap_or_else(|| panic!("trusted advice family should exist"));
        assert_eq!(advice_family.limbs, advice_requirement.limbs);
        assert_eq!(
            advice_family.alphabet.size(),
            advice_requirement.alphabet_size
        );

        let program_image_requirement = program_image_validity_requirement();
        let program_image_family = layout
            .family(&akita_packed_family_id(&program_image_requirement.family))
            .unwrap_or_else(|| panic!("program image family should exist"));
        assert_eq!(program_image_family.limbs, program_image_requirement.limbs);
        assert_eq!(
            program_image_family.alphabet.size(),
            program_image_requirement.alphabet_size
        );
    }

    #[test]
    fn lattice_family_ids_convert_to_akita_family_ids() {
        assert_eq!(
            akita_packed_family_id(&LatticePackedFamilyId::AdviceBytes {
                kind: JoltAdviceKind::Trusted,
                index: 3,
            }),
            PackedFamilyId::AdviceBytes {
                kind: PackedAdviceKind::Trusted,
                index: 3,
            }
        );
        assert_eq!(
            akita_packed_family_id(&LatticePackedFamilyId::Custom {
                namespace: 17,
                index: 5,
            }),
            PackedFamilyId::Custom {
                namespace: 17,
                index: 5,
            }
        );
        assert_eq!(
            akita_packed_family_id(&LatticePackedFamilyId::BytecodeRegisterSelector {
                chunk: 2,
                selector: 1,
            }),
            PackedFamilyId::BytecodeRegisterSelector {
                chunk: 2,
                selector: 1,
            }
        );
        assert_eq!(
            akita_packed_family_id(&LatticePackedFamilyId::BytecodeImmBytes { chunk: 2 }),
            PackedFamilyId::BytecodeImmBytes { chunk: 2 }
        );
    }

    #[test]
    fn lattice_direct_view_converts_to_akita_view_formula() {
        let formula = LatticePackedViewFormula::<Fr>::direct(LatticePackedFamilyId::IncSign, 0, 1);

        assert_eq!(
            akita_packed_view_formula(&formula)
                .unwrap_or_else(|error| panic!("direct view should convert: {error}")),
            PackedViewFormula::direct(PackedFamilyId::IncSign, 0, 1)
        );
    }

    #[test]
    fn lattice_linear_view_converts_terms_to_akita_view_formula() {
        let formula = LatticePackedViewFormula::linear_decoded(byte_decode_terms::<Fr>(
            LatticePackedFamilyId::BytecodeChunk { index: 2 },
            4,
        ));

        let converted = akita_packed_view_formula(&formula)
            .unwrap_or_else(|error| panic!("linear view should convert: {error}"));

        assert!(matches!(
            converted,
            PackedViewFormula::LinearDecoded { terms, .. }
                if terms.len() == 256
                    && terms[7].coefficient == Fr::from_u64(7)
                    && terms[7].family == (PackedFamilyId::BytecodeChunk { index: 2 })
                    && terms[7].limb == 4
                    && terms[7].symbol == 7
        ));
    }

    #[test]
    fn lattice_masked_views_require_prior_translation() {
        assert!(matches!(
            akita_packed_view_formula::<Fr>(&LatticePackedViewFormula::masked_decoded(
                JoltRelationId::FusedIncrementTranslation,
            )),
            Err(PackedViewError::MaskedViewRequiresTranslation)
        ));
    }

    #[test]
    fn lattice_reduced_masked_view_converts_terms_to_akita_formula() {
        let formula = LatticePackedViewFormula::reduced_masked(
            JoltRelationId::FusedIncrementTranslation,
            vec![jolt_claims::protocols::jolt::LatticePackedViewTerm::new(
                Fr::from_u64(9),
                LatticePackedFamilyId::IncSign,
                0,
                1,
            )],
        );
        assert!(matches!(
            akita_packed_view_formula::<Fr>(&formula),
            Ok(PackedViewFormula::ReducedMasked { terms })
                if terms.len() == 1
                    && terms[0].coefficient == Fr::from_u64(9)
                    && terms[0].family == PackedFamilyId::IncSign
                    && terms[0].limb == 0
                    && terms[0].symbol == 1
        ));
    }

    #[test]
    fn jolt_lattice_resolver_weights_ra_symbols_by_address_point() {
        let id = JoltOpeningId::committed(
            JoltCommittedPolynomial::InstructionRa(1),
            JoltRelationId::HammingWeightClaimReduction,
        );
        let point = vec![Fr::from_u64(2), Fr::from_u64(3), Fr::from_u64(5)];
        let expected_weights = EqPolynomial::<Fr>::evals(&point[..2], None);
        let formulas = jolt_lattice_view_formulas(
            &logical_manifest(id, point),
            2,
            &precommitted_schedule(None),
        )
        .unwrap_or_else(|error| panic!("RA lattice formula should resolve: {error}"));

        assert_eq!(formulas[0].0, Stage8OpeningId::from(id));
        assert!(matches!(
            &formulas[0].1,
            LatticePackedViewFormula::LinearDecoded { terms }
                if terms.len() == 4
                    && terms[2].coefficient == expected_weights[2]
                    && terms[2].family == LatticePackedFamilyId::InstructionRa { index: 1 }
                    && terms[2].limb == 0
                    && terms[2].symbol == 2
        ));
    }

    #[cfg(feature = "field-inline")]
    #[test]
    fn field_inline_rd_inc_resolves_to_packed_byte_families() {
        let id = field_increments::field_rd_inc_reduced_opening();
        let point = vec![Fr::from_u64(2), Fr::from_u64(3)];
        let formulas = jolt_lattice_view_formulas(
            &logical_manifest_for_stage8(Stage8OpeningId::from(id), point.clone()),
            8,
            &precommitted_schedule(None),
        )
        .unwrap_or_else(|error| panic!("field rd inc lattice formula should resolve: {error}"));

        assert_eq!(formulas[0].0, Stage8OpeningId::from(id));
        assert_eq!(formulas[0].2, point);

        let terms = linear_decoded_terms(&formulas[0].1);
        assert_eq!(terms.len(), AkitaField::NUM_BYTES * 256);
        let byte_1_symbol_3 = find_lattice_term(
            terms,
            LatticePackedFamilyId::FieldRdIncByte { index: 1 },
            0,
            3,
        );
        assert_eq!(byte_1_symbol_3.coefficient, Fr::from_u64(3 * 256));
    }

    #[test]
    fn jolt_lattice_resolver_decodes_advice_and_program_image_bytes() {
        let trusted = JoltOpeningId::trusted_advice(JoltRelationId::AdviceClaimReduction);
        let program_image = JoltOpeningId::committed(
            JoltCommittedPolynomial::ProgramImageInit,
            JoltRelationId::ProgramImageClaimReduction,
        );

        let schedule = precommitted_schedule(None);
        let trusted_formula = jolt_lattice_view_formula(trusted, &[Fr::from_u64(1)], 8, &schedule)
            .unwrap_or_else(|error| panic!("trusted advice view should resolve: {error}"));
        assert!(matches!(
            trusted_formula,
            LatticePackedViewFormula::LinearDecoded { terms }
                if terms.len() == 256
                    && terms[7].coefficient == Fr::from_u64(7)
                    && terms[7].family == (LatticePackedFamilyId::AdviceBytes {
                        kind: JoltAdviceKind::Trusted,
                        index: 0,
                    })
                    && terms[7].limb == 0
                    && terms[7].symbol == 7
        ));

        let program_image_formula =
            jolt_lattice_view_formula(program_image, &[Fr::from_u64(1)], 8, &schedule)
                .unwrap_or_else(|error| panic!("program image view should resolve: {error}"));
        assert!(matches!(
            program_image_formula,
            LatticePackedViewFormula::LinearDecoded { terms }
                if terms.len() == 8 * 256
                    && terms[256 + 7].coefficient == Fr::from_u64(256 * 7)
                    && terms[256 + 7].family == LatticePackedFamilyId::ProgramImageInit
                    && terms[256 + 7].limb == 1
                    && terms[256 + 7].symbol == 7
        ));
    }

    #[test]
    fn jolt_lattice_resolver_marks_increments_as_masked() {
        let id = JoltOpeningId::committed(
            JoltCommittedPolynomial::RamInc,
            JoltRelationId::IncClaimReduction,
        );
        assert!(matches!(
            jolt_lattice_view_formula(id, &[Fr::from_u64(1)], 8, &precommitted_schedule(None))
                .unwrap_or_else(|error| panic!("increment view should resolve: {error}")),
            LatticePackedViewFormula::MaskedDecoded {
                relation: JoltRelationId::FusedIncrementTranslation
            }
        ));
    }

    #[test]
    fn jolt_lattice_resolver_lowers_fused_increment_decode_outputs() {
        let magnitude = jolt_lattice_view_formula(
            fused_increment_magnitude_opening(),
            &[Fr::from_u64(1)],
            8,
            &precommitted_schedule(None),
        )
        .unwrap_or_else(|error| panic!("magnitude view should resolve: {error}"));
        let magnitude_terms = linear_decoded_terms(&magnitude);
        assert_eq!(
            find_lattice_term(
                magnitude_terms,
                LatticePackedFamilyId::IncByte { index: 7 },
                0,
                3,
            )
            .coefficient,
            Fr::from_u64(256_u64.pow(7) * 3)
        );

        assert!(matches!(
            jolt_lattice_view_formula(
                fused_increment_sign_opening(),
                &[Fr::from_u64(1)],
                8,
                &precommitted_schedule(None),
            )
            .unwrap_or_else(|error| panic!("sign view should resolve: {error}")),
            LatticePackedViewFormula::Direct {
                family: LatticePackedFamilyId::IncSign,
                limb: 0,
                symbol: 1
            }
        ));
    }

    #[test]
    fn jolt_lattice_resolver_rejects_fused_increment_source_until_bytecode_relation() {
        let error = match jolt_lattice_view_formula::<Fr>(
            fused_increment_source_opening(LatticeFusedIncrementTarget::Ram),
            &[Fr::from_u64(1)],
            8,
            &precommitted_schedule(None),
        ) {
            Ok(_) => panic!("source view should require a bytecode-derived relation"),
            Err(error) => error,
        };
        assert!(error
            .to_string()
            .contains("bytecode-derived packed view relation"));
    }

    #[test]
    fn jolt_lattice_resolver_lowers_source_link_bytecode_ra() {
        let id = JoltOpeningId::committed(
            JoltCommittedPolynomial::BytecodeRa(0),
            JoltRelationId::FusedIncrementSourceLink,
        );
        let point = (1..=9).map(Fr::from_u64).collect::<Vec<_>>();
        let formula = jolt_lattice_view_formula(id, &point, 8, &precommitted_schedule(None))
            .unwrap_or_else(|error| panic!("source-link BytecodeRa should resolve: {error}"));
        let terms = linear_decoded_terms(&formula);

        assert_eq!(
            find_lattice_term(terms, LatticePackedFamilyId::BytecodeRa { index: 0 }, 0, 7,)
                .coefficient,
            EqPolynomial::<Fr>::evals(&point[..8], None)[7]
        );
    }

    #[test]
    fn jolt_lattice_resolver_lowers_fused_increment_bytecode_sources() {
        let schedule = precommitted_schedule(None);
        let point = [
            Fr::from_u64(3),
            Fr::from_u64(5),
            Fr::from_u64(7),
            Fr::from_u64(11),
        ];
        let chunk_weights = EqPolynomial::<Fr>::evals(&point[..1], None);
        let store_formula = jolt_lattice_view_formula(
            fused_increment_bytecode_source_opening(LatticeFusedIncrementTarget::Ram),
            &point,
            8,
            &schedule,
        )
        .unwrap_or_else(|error| panic!("store source view should resolve: {error}"));
        let store_terms = linear_decoded_terms(&store_formula);
        assert_eq!(store_terms.len(), 2);
        assert_eq!(
            find_lattice_term(
                store_terms,
                LatticePackedFamilyId::BytecodeCircuitFlag {
                    chunk: 0,
                    flag: CircuitFlags::Store as usize,
                },
                0,
                1,
            )
            .coefficient,
            chunk_weights[0]
        );
        assert_eq!(
            find_lattice_term(
                store_terms,
                LatticePackedFamilyId::BytecodeCircuitFlag {
                    chunk: 1,
                    flag: CircuitFlags::Store as usize,
                },
                0,
                1,
            )
            .coefficient,
            chunk_weights[1]
        );

        let rd_formula = jolt_lattice_view_formula(
            fused_increment_bytecode_source_opening(LatticeFusedIncrementTarget::Rd),
            &point,
            8,
            &schedule,
        )
        .unwrap_or_else(|error| panic!("rd-present source view should resolve: {error}"));
        let rd_terms = linear_decoded_terms(&rd_formula);
        assert_eq!(rd_terms.len(), 2 * (1 << REGISTER_ADDRESS_BITS));
        assert_eq!(
            find_lattice_term(
                rd_terms,
                LatticePackedFamilyId::BytecodeRegisterSelector {
                    chunk: 1,
                    selector: 2,
                },
                0,
                31,
            )
            .coefficient,
            chunk_weights[1]
        );
    }

    #[test]
    fn jolt_lattice_resolver_decodes_bytecode_chunk_lanes() {
        let schedule = precommitted_schedule(None);
        let id = JoltOpeningId::committed(
            JoltCommittedPolynomial::BytecodeChunk(0),
            JoltRelationId::BytecodeClaimReduction,
        );
        let (point, lane_weights) = bytecode_chunk_opening_point();
        let formula = jolt_lattice_view_formula(id, &point, 8, &schedule)
            .unwrap_or_else(|error| panic!("bytecode view should resolve: {error}"));
        let terms = linear_decoded_terms(&formula);
        let lane_layout = bytecode::BYTECODE_LANE_LAYOUT;

        assert_eq!(
            find_lattice_term(
                terms,
                LatticePackedFamilyId::BytecodeRegisterSelector {
                    chunk: 0,
                    selector: 1,
                },
                0,
                5,
            )
            .coefficient,
            lane_weights[lane_layout.rs2_start + 5]
        );
        assert_eq!(
            find_lattice_term(
                terms,
                LatticePackedFamilyId::BytecodeImmBytes { chunk: 0 },
                1,
                7,
            )
            .coefficient,
            lane_weights[lane_layout.imm_idx] * Fr::from_u64(256 * 7)
        );
    }

    #[test]
    fn jolt_lattice_resolver_rejects_bytecode_chunks_without_layout() {
        let id = JoltOpeningId::committed(
            JoltCommittedPolynomial::BytecodeChunk(0),
            JoltRelationId::BytecodeClaimReduction,
        );
        assert!(matches!(
            jolt_lattice_view_formula::<Fr>(
                id,
                &[Fr::from_u64(1)],
                8,
                &precommitted_schedule_without_committed_program(),
            ),
            Err(VerifierError::FinalOpeningBatchFailed { reason })
                if reason.contains("BytecodeChunk")
        ));
    }

    #[test]
    fn jolt_lattice_physical_manifest_lowers_supported_ra_view() {
        let id = JoltOpeningId::committed(
            JoltCommittedPolynomial::InstructionRa(1),
            JoltRelationId::HammingWeightClaimReduction,
        );
        let point = vec![Fr::from_u64(2), Fr::from_u64(3), Fr::from_u64(5)];
        let expected_weights = EqPolynomial::<Fr>::evals(&point[..2], None);
        let logical = logical_manifest(id, point);
        let layout = PackedWitnessLayout::new([PackedFamilySpec::direct(
            PackedFamilyId::InstructionRa { index: 1 },
            PackedFactDomain::TraceRows { log_t: 1 },
            1,
            PackedAlphabet::Fixed { size: 4 },
        )])
        .unwrap_or_else(|error| panic!("test layout should be valid: {error}"));

        let physical =
            jolt_lattice_physical_manifest(&logical, &layout, 2, &precommitted_schedule(None))
                .unwrap_or_else(|error| panic!("physical manifest should resolve: {error}"));

        assert_eq!(physical.layout_digest, layout.digest);
        assert_eq!(physical.openings[0].id, Stage8OpeningId::from(id));
        assert!(matches!(
            &physical.openings[0].view,
            PhysicalView::PackedLinear {
                layout_digest,
                terms
            } if *layout_digest == layout.digest
                && terms.len() == 4
                && terms[2].coefficient == expected_weights[2]
                && terms[2].family == (PackedFamilyId::InstructionRa { index: 1 }).physical_ref()
                && terms[2].limb == 0
                && terms[2].symbol == 2
        ));
    }

    #[test]
    fn jolt_lattice_physical_manifest_lowers_bytecode_chunk_view() {
        let schedule = precommitted_schedule(None);
        let layout =
            derive_akita_packed_witness_layout(&lattice_config(), 2, 8, ra_layout(), &schedule)
                .unwrap_or_else(|error| panic!("layout derivation should succeed: {error}"));
        let id = JoltOpeningId::committed(
            JoltCommittedPolynomial::BytecodeChunk(0),
            JoltRelationId::BytecodeClaimReduction,
        );
        let (point, lane_weights) = bytecode_chunk_opening_point();
        let logical = logical_manifest(id, point);
        let physical = jolt_lattice_physical_manifest(&logical, &layout, 8, &schedule)
            .unwrap_or_else(|error| panic!("physical manifest should resolve: {error}"));

        assert_eq!(physical.layout_digest, layout.digest);
        assert_eq!(physical.openings[0].id, Stage8OpeningId::from(id));
        match &physical.openings[0].view {
            PhysicalView::PackedLinear {
                layout_digest,
                terms,
            } => {
                assert_eq!(*layout_digest, layout.digest);
                assert_eq!(
                    find_physical_term(terms, PackedFamilyId::BytecodeImmBytes { chunk: 0 }, 1, 7,)
                        .coefficient,
                    lane_weights[bytecode::BYTECODE_LANE_LAYOUT.imm_idx] * Fr::from_u64(256 * 7)
                );
                assert!(layout
                    .family(&PackedFamilyId::BytecodeChunk { index: 0 })
                    .is_none());
            }
            PhysicalView::Direct => panic!("expected packed linear view"),
        }
    }

    #[test]
    fn jolt_lattice_physical_manifest_rejects_masked_increment_view() {
        let id = JoltOpeningId::committed(
            JoltCommittedPolynomial::RamInc,
            JoltRelationId::IncClaimReduction,
        );
        let layout = PackedWitnessLayout::new([PackedFamilySpec::direct(
            PackedFamilyId::IncSign,
            PackedFactDomain::TraceRows { log_t: 0 },
            1,
            PackedAlphabet::Bit,
        )])
        .unwrap_or_else(|error| panic!("test layout should be valid: {error}"));

        assert!(matches!(
            jolt_lattice_physical_manifest(
                &logical_manifest(id, vec![Fr::from_u64(1)]),
                &layout,
                8,
                &precommitted_schedule(None),
            ),
            Err(VerifierError::FinalOpeningBatchFailed { reason })
                if reason.contains("masked packed view")
        ));
    }

    #[cfg(feature = "field-inline")]
    #[test]
    fn jolt_lattice_physical_manifest_resolves_field_inline_rd_inc() {
        let mut config = lattice_config();
        config.lattice.field_inline.enabled = true;
        config.lattice.packed_witness.field_rd_inc_family = true;
        let layout = derive_akita_packed_witness_layout(
            &config,
            2,
            8,
            ra_layout(),
            &precommitted_schedule(None),
        )
        .unwrap_or_else(|error| panic!("layout derivation should succeed: {error}"));
        let id = field_increments::field_rd_inc_reduced_opening();
        let row_point = vec![Fr::from_u64(11), Fr::from_u64(13)];

        let manifest = jolt_lattice_physical_manifest(
            &logical_manifest_for_stage8(Stage8OpeningId::from(id), row_point.clone()),
            &layout,
            8,
            &precommitted_schedule(None),
        )
        .unwrap_or_else(|error| panic!("field rd inc physical manifest should resolve: {error}"));

        let PhysicalView::PackedLinear { terms, .. } = &manifest.openings[0].view else {
            panic!("field rd inc should lower to a packed linear view");
        };
        let term = find_physical_term(terms, PackedFamilyId::FieldRdIncByte { index: 1 }, 0, 3);
        assert_eq!(term.coefficient, Fr::from_u64(3 * 256));
        assert_eq!(term.row_point, row_point);
    }

    #[test]
    fn layout_config_mismatch_rejects() {
        let layout = derive_akita_packed_witness_layout(
            &lattice_config(),
            2,
            8,
            ra_layout(),
            &precommitted_schedule(None),
        )
        .unwrap_or_else(|error| panic!("layout derivation should succeed: {error}"));

        assert!(matches!(
            validate_akita_packed_witness_layout_config(&lattice_config(), &layout),
            Err(VerifierError::InvalidProtocolConfig { .. })
        ));
    }

    #[test]
    fn advice_layout_requires_precommitted_schedule() {
        let mut config = lattice_config();
        config.lattice.advice.trusted = true;
        config.lattice.packed_witness.trusted_advice_family = true;

        assert!(matches!(
            derive_akita_packed_witness_layout(
                &config,
                2,
                8,
                ra_layout(),
                &precommitted_schedule(None),
            ),
            Err(VerifierError::InvalidPrecommittedSchedule { .. })
        ));

        let layout = derive_akita_packed_witness_layout(
            &config,
            2,
            8,
            ra_layout(),
            &precommitted_schedule(Some(64)),
        )
        .unwrap_or_else(|error| panic!("layout derivation should succeed: {error}"));
        assert!(layout
            .family(&PackedFamilyId::AdviceBytes {
                kind: PackedAdviceKind::Trusted,
                index: 0,
            })
            .is_some());
    }

    #[cfg(feature = "field-inline")]
    #[test]
    fn field_inline_layout_uses_separate_rd_inc_families() {
        let mut config = lattice_config();
        config.lattice.field_inline.enabled = true;
        config.lattice.packed_witness.field_rd_inc_family = true;

        let layout = derive_akita_packed_witness_layout(
            &config,
            2,
            8,
            ra_layout(),
            &precommitted_schedule(None),
        )
        .unwrap_or_else(|error| panic!("layout derivation should succeed: {error}"));

        assert!(layout.family(&PackedFamilyId::IncSign).is_some());
        assert!(layout.family(&PackedFamilyId::FieldRdIncSign).is_none());
        assert!(layout
            .family(&PackedFamilyId::FieldRdIncByte { index: 7 })
            .is_some());
        assert!(layout
            .family(&PackedFamilyId::FieldRdIncByte {
                index: AkitaField::NUM_BYTES - 1
            })
            .is_some());
    }

    #[test]
    fn advice_and_committed_program_use_non_trace_domains() {
        let mut config = lattice_config();
        config.lattice.advice.trusted = true;
        config.lattice.advice.untrusted = true;
        config.lattice.packed_witness.trusted_advice_family = true;
        config.lattice.packed_witness.untrusted_advice_family = true;

        let layout = derive_akita_packed_witness_layout(
            &config,
            2,
            8,
            ra_layout(),
            &precommitted_schedule_with_advice(Some(64), Some(128)),
        )
        .unwrap_or_else(|error| panic!("layout derivation should succeed: {error}"));

        let trusted = layout
            .family(&PackedFamilyId::AdviceBytes {
                kind: PackedAdviceKind::Trusted,
                index: 0,
            })
            .unwrap_or_else(|| panic!("trusted advice family should be present"));
        assert!(matches!(
            trusted.domain,
            PackedFactDomain::AdviceBytes {
                kind: PackedAdviceKind::Trusted,
                ..
            }
        ));

        let untrusted = layout
            .family(&PackedFamilyId::AdviceBytes {
                kind: PackedAdviceKind::Untrusted,
                index: 0,
            })
            .unwrap_or_else(|| panic!("untrusted advice family should be present"));
        assert!(matches!(
            untrusted.domain,
            PackedFactDomain::AdviceBytes {
                kind: PackedAdviceKind::Untrusted,
                ..
            }
        ));

        let bytecode = layout
            .family(&PackedFamilyId::BytecodeRegisterSelector {
                chunk: 0,
                selector: 0,
            })
            .unwrap_or_else(|| panic!("bytecode register selector family should be present"));
        assert_eq!(
            bytecode.domain,
            PackedFactDomain::BytecodeRows { log_bytecode: 3 }
        );
        assert_eq!(
            bytecode.alphabet,
            PackedAlphabet::Fixed {
                size: 1usize << REGISTER_ADDRESS_BITS,
            }
        );
        let lookup_selector = layout
            .family(&PackedFamilyId::BytecodeLookupSelector { chunk: 0 })
            .unwrap_or_else(|| panic!("bytecode lookup selector family should be present"));
        assert_eq!(
            lookup_selector.alphabet,
            PackedAlphabet::Fixed {
                size: LookupTableKind::<RISCV_XLEN>::COUNT.next_power_of_two(),
            }
        );
        assert!(lookup_selector.alphabet.size().is_power_of_two());
        assert!(lookup_selector.alphabet.size() >= LookupTableKind::<RISCV_XLEN>::COUNT);

        let imm = layout
            .family(&PackedFamilyId::BytecodeImmBytes { chunk: 0 })
            .unwrap_or_else(|| panic!("bytecode immediate byte family should be present"));
        assert_eq!(
            imm.domain,
            PackedFactDomain::BytecodeRows { log_bytecode: 3 }
        );
        assert_eq!(imm.limbs, AkitaField::NUM_BYTES);
        assert_eq!(imm.alphabet, PackedAlphabet::Byte);

        let program_image = layout
            .family(&PackedFamilyId::ProgramImageInit)
            .unwrap_or_else(|| panic!("program-image family should be present"));
        assert_eq!(
            program_image.domain,
            PackedFactDomain::ProgramImageWords { log_words: 2 }
        );
        assert_eq!(program_image.limbs, 8);
    }

    #[cfg(feature = "field-inline")]
    #[test]
    fn single_packed_witness_layout_includes_all_supported_lattice_families() {
        let mut config = lattice_config();
        config.lattice.field_inline.enabled = true;
        config.lattice.advice.trusted = true;
        config.lattice.advice.untrusted = true;
        config.lattice.packed_witness.field_rd_inc_family = true;
        config.lattice.packed_witness.trusted_advice_family = true;
        config.lattice.packed_witness.untrusted_advice_family = true;

        let layout = derive_akita_packed_witness_layout(
            &config,
            2,
            8,
            ra_layout(),
            &precommitted_schedule_with_advice(Some(64), Some(128)),
        )
        .unwrap_or_else(|error| panic!("layout derivation should succeed: {error}"));
        let audit = layout.audit();

        assert_eq!(audit.d_pack, layout.dimension);
        assert!(audit.cells_by_domain.trace_rows > 0);
        assert!(audit.cells_by_domain.bytecode_rows > 0);
        assert!(audit.cells_by_domain.program_image_words > 0);
        assert!(audit.cells_by_domain.advice_bytes > 0);
        assert!(layout.family(&PackedFamilyId::IncSign).is_some());
        assert!(layout
            .family(&PackedFamilyId::IncByte { index: 7 })
            .is_some());
        assert!(layout
            .family(&PackedFamilyId::FieldRdIncByte { index: 0 })
            .is_some());
        assert!(layout
            .family(&PackedFamilyId::FieldRdIncByte {
                index: AkitaField::NUM_BYTES - 1,
            })
            .is_some());
        assert!(layout
            .family(&PackedFamilyId::AdviceBytes {
                kind: PackedAdviceKind::Trusted,
                index: 0,
            })
            .is_some());
        assert!(layout
            .family(&PackedFamilyId::AdviceBytes {
                kind: PackedAdviceKind::Untrusted,
                index: 0,
            })
            .is_some());
        assert!(layout
            .family(&PackedFamilyId::BytecodeRegisterSelector {
                chunk: 0,
                selector: 2,
            })
            .is_some());
        assert!(layout
            .family(&PackedFamilyId::BytecodeCircuitFlag { chunk: 0, flag: 0 })
            .is_some());
        assert!(layout
            .family(&PackedFamilyId::BytecodeLookupSelector { chunk: 0 })
            .is_some());
        assert!(layout
            .family(&PackedFamilyId::BytecodeImmBytes { chunk: 0 })
            .is_some());
        assert!(layout.family(&PackedFamilyId::ProgramImageInit).is_some());
    }
}
