use crate::{
    config::{validate_protocol_config, JoltProtocolConfig, PcsFamily},
    stages::PrecommittedSchedule,
    VerifierError,
};
use jolt_akita::AkitaField;
#[cfg(feature = "field-inline")]
use jolt_akita::AKITA_FIELD_MODULUS;
#[cfg(feature = "field-inline")]
use jolt_claims::protocols::field_inline::{
    formulas::{claim_reductions::increments as field_increments, lattice as field_lattice},
    FieldInlineOpeningId,
};
use jolt_claims::protocols::jolt::{
    advice_bytes_validity_requirement, byte_decode_terms,
    formulas::{dimensions::REGISTER_ADDRESS_BITS, ra::JoltRaPolynomialLayout},
    lattice_packed_validity_digest, unsigned_inc_msb_lattice_view_formula,
    unsigned_inc_msb_opening, unsigned_inc_validity_requirements, weighted_symbol_terms,
    AdviceClaimReductionLayout, JoltAdviceKind, JoltCommittedPolynomial, JoltOpeningId,
    JoltPolynomialId, JoltRelationId, LatticePackedFamilyId, LatticePackedValidityKind,
    LatticePackedValidityRequirement, LatticePackedViewFormula,
};
use jolt_field::{Field, FixedByteSize};
use jolt_openings::{
    BatchOpeningClaim, BatchOpeningScheme, BatchOpeningStatement, CommitmentScheme,
    PackedAdviceKind, PackedAlphabet, PackedFactDomain, PackedFamilyId, PackedFamilySpec,
    PackedLinearTerm, PackedViewError, PackedViewFormula, PackedViewTerm, PackedWitnessLayout,
    PhysicalView,
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

pub fn derive_lattice_packed_witness_layout(
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

    let unsigned_inc_requirements =
        unsigned_inc_validity_requirements(log_k_chunk).ok_or_else(|| {
            invalid_lattice_config(format!(
                "unsigned increment chunk reconstruction requires log_k_chunk to divide 64, got {log_k_chunk}",
            ))
        })?;
    extend_validity_requirement_families(&mut specs, &unsigned_inc_requirements, trace)?;

    if config.lattice.field_inline.enabled {
        extend_field_rd_inc_families(&mut specs, trace)?;
    }

    if config.lattice.advice.untrusted {
        specs.push(advice_family(
            JoltAdviceKind::Untrusted,
            require_advice_layout(precommitted, JoltAdviceKind::Untrusted)?,
        )?);
    }

    let _ = precommitted.bytecode.as_ref().ok_or_else(|| {
        invalid_precommitted_schedule(
            "lattice committed-program mode requires a bytecode claim-reduction layout",
        )
    })?;

    let _ = precommitted.program_image.as_ref().ok_or_else(|| {
        invalid_precommitted_schedule(
            "lattice committed-program mode requires a program-image claim-reduction layout",
        )
    })?;

    PackedWitnessLayout::new(specs).map_err(|error| invalid_lattice_config(error.to_string()))
}

pub fn derive_lattice_packed_validity_requirements(
    config: &JoltProtocolConfig,
    log_k_chunk: usize,
    precommitted: &PrecommittedSchedule,
) -> Result<Vec<LatticePackedValidityRequirement>, VerifierError> {
    if validate_protocol_config(config)? != PcsFamily::Lattice {
        return Err(invalid_lattice_config(
            "Akita packed witness validity derivation requires lattice PCS mode",
        ));
    }

    let mut requirements = unsigned_inc_validity_requirements(log_k_chunk).ok_or_else(|| {
        invalid_lattice_config(format!(
            "unsigned increment chunk reconstruction requires log_k_chunk to divide 64, got {log_k_chunk}",
        ))
    })?;
    if config.lattice.field_inline.enabled {
        requirements.extend(field_rd_inc_validity_requirements());
    }
    if config.lattice.advice.untrusted {
        let _ = require_advice_layout(precommitted, JoltAdviceKind::Untrusted)?;
        requirements.push(advice_bytes_validity_requirement(JoltAdviceKind::Untrusted));
    }

    let _ = precommitted.bytecode.as_ref().ok_or_else(|| {
        invalid_precommitted_schedule(
            "lattice committed-program mode requires a bytecode claim-reduction layout",
        )
    })?;

    let _ = precommitted.program_image.as_ref().ok_or_else(|| {
        invalid_precommitted_schedule(
            "lattice committed-program mode requires a program-image claim-reduction layout",
        )
    })?;

    Ok(requirements)
}

pub fn derive_lattice_packed_validity_statements(
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

        let family_id = lattice_packing_family_id(&requirement.family);
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
        return Err(VerifierError::LatticePackedValidityClaimCountMismatch {
            expected: expected_opening_claims,
            got: opening_claims.len(),
        });
    }
    if reduction.batching_coefficients.len() != statements.len() {
        return Err(VerifierError::LatticePackedValiditySumcheckFailed {
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
            .map_err(|error| VerifierError::LatticePackedValiditySumcheckFailed {
                reason: error.to_string(),
            })?;
        let opening_count = validity_statement_opening_count(statement);
        let statement_openings = &opening_claims[opening_offset..opening_offset + opening_count];
        let eq_mask = try_eq_mle(point, &eq_points[index]).map_err(|error| {
            VerifierError::LatticePackedValiditySumcheckFailed {
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
    log_k_chunk: usize,
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
    let requirements =
        derive_lattice_packed_validity_requirements(config, log_k_chunk, precommitted)?;
    let statements = derive_lattice_packed_validity_statements(layout, &requirements)?;
    let expected_opening_claims = lattice_packed_validity_opening_count(&statements);
    if opening_claims.len() != expected_opening_claims {
        return Err(VerifierError::LatticePackedValidityClaimCountMismatch {
            expected: expected_opening_claims,
            got: opening_claims.len(),
        });
    }

    let eq_points = sample_lattice_packed_validity_eq_points(transcript, layout, &statements);
    let sumcheck_claims = lattice_packed_validity_claims(&statements);
    let compressed = match sumcheck_proof {
        SumcheckProof::Clear(ClearProof::Compressed(proof)) => proof,
        SumcheckProof::Clear(ClearProof::Full(_)) => {
            return Err(VerifierError::LatticePackedValiditySumcheckFailed {
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
            .map_err(|error| VerifierError::LatticePackedValiditySumcheckFailed {
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
        return Err(VerifierError::LatticePackedValidityOutputMismatch);
    }
    PCS::verify_batch(setup, transcript, &batch.statement, opening_proof)
        .map_err(
            |error| VerifierError::LatticePackedValidityOpeningVerificationFailed {
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
    transcript.append(&Label(b"LatticePackedValidity"));
    transcript.append(&LabelWithCount(
        b"lattice_validity_layout",
        layout.digest.len() as u64,
    ));
    transcript.append_bytes(&layout.digest);
    transcript.append(&U64Word(layout.dimension as u64));
    transcript.append(&U64Word(layout.cells as u64));
    transcript.append(&LabelWithCount(
        b"lattice_validity_stmts",
        statements.len() as u64,
    ));
    for (index, statement) in statements.iter().enumerate() {
        let family = lattice_packing_family_id(&statement.requirement.family).physical_ref();
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
            LatticePackedValidityKind::BytecodeStoreRdDisjoint => {
                transcript.append(&U64Word(3));
                transcript.append(&U64Word(0));
            }
            LatticePackedValidityKind::FieldElementCanonicalBytes {
                byte_width,
                modulus,
            } => {
                transcript.append(&U64Word(4));
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
        LatticePackedValidityStatementKind::BytecodeStoreRdDisjoint => 4,
        LatticePackedValidityStatementKind::FieldElementCanonicalBytes => 5,
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
        LatticePackedValidityStatementKind::BytecodeStoreRdDisjoint
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
    let family_id = lattice_packing_family_id(&statement.requirement.family);
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
        return Err(VerifierError::LatticePackedValiditySumcheckFailed {
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
        return Err(VerifierError::LatticePackedValiditySumcheckFailed {
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
        LatticePackedValidityStatementKind::BytecodeStoreRdDisjoint
        | LatticePackedValidityStatementKind::FieldElementCanonicalBytes => shape.row_vars,
    };
    if point.len() != expected {
        return Err(VerifierError::LatticePackedValiditySumcheckFailed {
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

pub fn validate_lattice_packed_witness_validity_config(
    config: &JoltProtocolConfig,
    log_k_chunk: usize,
    precommitted: &PrecommittedSchedule,
) -> Result<(), VerifierError> {
    let requirements =
        derive_lattice_packed_validity_requirements(config, log_k_chunk, precommitted)?;
    let digest = lattice_packed_validity_digest(&requirements);
    if config.lattice.packed_witness.validity_digest != Some(digest) {
        return Err(invalid_lattice_config(
            "configured Akita packed witness validity digest does not match derived requirements",
        ));
    }
    Ok(())
}

pub fn validate_lattice_packed_witness_layout_config(
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
    for family in &layout.families {
        if packed_family_is_precommitted(&family.id) {
            return Err(invalid_lattice_config(format!(
                "precommitted family {:?} cannot be included in the Akita packed witness layout",
                family.id
            )));
        }
    }
    Ok(())
}

fn packed_family_is_precommitted(family: &PackedFamilyId) -> bool {
    matches!(
        family,
        PackedFamilyId::AdviceBytes {
            kind: PackedAdviceKind::Trusted,
            ..
        } | PackedFamilyId::BytecodeChunk { .. }
            | PackedFamilyId::BytecodeRegisterSelector { .. }
            | PackedFamilyId::BytecodeCircuitFlag { .. }
            | PackedFamilyId::BytecodeInstructionFlag { .. }
            | PackedFamilyId::BytecodeLookupSelector { .. }
            | PackedFamilyId::BytecodeRafFlag { .. }
            | PackedFamilyId::BytecodeUnexpandedPcBytes { .. }
            | PackedFamilyId::BytecodeImmBytes { .. }
            | PackedFamilyId::ProgramImageInit
    )
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
    if term_requires_bytecode_store_rd_disjoint(family)
        && !requirements.iter().any(|requirement| {
            bytecode_store_rd_disjoint_requirement_covers_term(requirement, family)
        })
    {
        return Err(unsupported_lattice_view(format!(
            "opening {id:?} uses bytecode source family {family:?} without a bound Store/Rd disjointness requirement"
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
        LatticePackedValidityKind::BytecodeStoreRdDisjoint
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

fn term_requires_bytecode_store_rd_disjoint(family: &LatticePackedFamilyId) -> bool {
    match family {
        LatticePackedFamilyId::BytecodeCircuitFlag { flag, .. } => {
            *flag == CircuitFlags::Store as usize
        }
        LatticePackedFamilyId::BytecodeRegisterSelector { selector, .. } => *selector == 2,
        _ => false,
    }
}

fn bytecode_store_rd_disjoint_requirement_covers_term(
    requirement: &LatticePackedValidityRequirement,
    family: &LatticePackedFamilyId,
) -> bool {
    let LatticePackedValidityKind::BytecodeStoreRdDisjoint = requirement.kind else {
        return false;
    };
    let LatticePackedFamilyId::BytecodeCircuitFlag {
        chunk: requirement_chunk,
        flag,
    } = &requirement.family
    else {
        return false;
    };
    if *flag != CircuitFlags::Store as usize {
        return false;
    }
    match family {
        LatticePackedFamilyId::BytecodeCircuitFlag { chunk, flag } => {
            requirement_chunk == chunk && *flag == CircuitFlags::Store as usize
        }
        LatticePackedFamilyId::BytecodeRegisterSelector { chunk, selector } => {
            requirement_chunk == chunk && *selector == 2
        }
        _ => false,
    }
}

fn jolt_lattice_row_point<F>(
    id: JoltOpeningId,
    point: &[F],
    log_k_chunk: usize,
    _precommitted: &PrecommittedSchedule,
) -> Result<Vec<F>, VerifierError>
where
    F: Field,
{
    match id {
        JoltOpeningId::Polynomial {
            polynomial: JoltPolynomialId::Committed(polynomial),
            relation,
        } => committed_lattice_row_point(polynomial, relation, point, log_k_chunk),
        JoltOpeningId::TrustedAdvice {
            relation: JoltRelationId::AdviceClaimReduction,
        } => Err(unsupported_lattice_view(
            "trusted advice uses a separate precommitted opening, not a packed witness row point",
        )),
        JoltOpeningId::UntrustedAdvice {
            relation: JoltRelationId::AdviceClaimReduction,
        } => Ok(point.to_vec()),
        id if id == unsigned_inc_msb_opening() => Ok(point.to_vec()),
        id if unsigned_inc_chunk_index(id).is_some() => ra_row_point(point, log_k_chunk),
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
        ) => ra_row_point(point, log_k_chunk),
        (
            JoltCommittedPolynomial::RamInc | JoltCommittedPolynomial::RdInc,
            JoltRelationId::IncClaimReduction,
        ) => Err(unsupported_lattice_view(
            "lattice mode opens increment chunks/MSB through unsigned increment reconstruction, not dense IncClaimReduction polynomials",
        )),
        (JoltCommittedPolynomial::ProgramImageInit, JoltRelationId::ProgramImageClaimReduction) => {
            Err(unsupported_lattice_view(
                "ProgramImageInit uses a separate precommitted opening, not a packed witness row point",
            ))
        }
        (JoltCommittedPolynomial::TrustedAdvice, JoltRelationId::AdviceClaimReduction) => {
            Err(unsupported_lattice_view(
                "TrustedAdvice uses a separate precommitted opening, not a packed witness row point",
            ))
        }
        (JoltCommittedPolynomial::UntrustedAdvice, JoltRelationId::AdviceClaimReduction) => {
            Ok(point.to_vec())
        }
        (JoltCommittedPolynomial::BytecodeChunk(index), JoltRelationId::BytecodeClaimReduction) => {
            Err(unsupported_lattice_view(format!(
                "BytecodeChunk({index}) uses a separate precommitted opening, not a packed witness row point"
            )))
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
    _precommitted: &PrecommittedSchedule,
) -> Result<LatticePackedViewFormula<F>, VerifierError>
where
    F: Field,
{
    match id {
        JoltOpeningId::Polynomial {
            polynomial: JoltPolynomialId::Committed(polynomial),
            relation,
        } => committed_lattice_view_formula(polynomial, relation, point, log_k_chunk),
        JoltOpeningId::TrustedAdvice {
            relation: JoltRelationId::AdviceClaimReduction,
        } => Err(unsupported_lattice_view(
            "trusted advice uses a separate precommitted opening, not a packed witness view",
        )),
        JoltOpeningId::UntrustedAdvice {
            relation: JoltRelationId::AdviceClaimReduction,
        } => Ok(advice_lattice_view_formula(JoltAdviceKind::Untrusted)),
        id if id == unsigned_inc_msb_opening() => Ok(unsigned_inc_msb_lattice_view_formula()),
        id => {
            if let Some(index) = unsigned_inc_chunk_index(id) {
                return unsigned_inc_chunk_lattice_view_formula(index, point, log_k_chunk);
            }
            Err(unsupported_lattice_view(format!(
                "final opening {id:?} has no supported lattice packed view"
            )))
        }
    }
}

pub fn lattice_packing_family_id(family: &LatticePackedFamilyId) -> PackedFamilyId {
    match family {
        LatticePackedFamilyId::InstructionRa { index } => {
            PackedFamilyId::InstructionRa { index: *index }
        }
        LatticePackedFamilyId::BytecodeRa { index } => PackedFamilyId::BytecodeRa { index: *index },
        LatticePackedFamilyId::RamRa { index } => PackedFamilyId::RamRa { index: *index },
        LatticePackedFamilyId::UnsignedIncChunk { index } => {
            PackedFamilyId::UnsignedIncChunk { index: *index }
        }
        LatticePackedFamilyId::UnsignedIncMsb => PackedFamilyId::UnsignedIncMsb,
        LatticePackedFamilyId::FieldRdIncByte { index } => {
            PackedFamilyId::FieldRdIncByte { index: *index }
        }
        LatticePackedFamilyId::FieldRdIncSign => PackedFamilyId::FieldRdIncSign,
        LatticePackedFamilyId::AdviceBytes { kind, index } => PackedFamilyId::AdviceBytes {
            kind: lattice_packing_advice_kind(*kind),
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
            JoltRelationId::HammingWeightClaimReduction,
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
            Err(unsupported_lattice_view(
                "lattice mode opens increment chunks/MSB through unsigned increment reconstruction, not dense IncClaimReduction polynomials",
            ))
        }
        (JoltCommittedPolynomial::ProgramImageInit, JoltRelationId::ProgramImageClaimReduction) => {
            Err(unsupported_lattice_view(
                "ProgramImageInit uses a separate precommitted opening, not a packed witness view",
            ))
        }
        (JoltCommittedPolynomial::TrustedAdvice, JoltRelationId::AdviceClaimReduction) => {
            Err(unsupported_lattice_view(
                "TrustedAdvice uses a separate precommitted opening, not a packed witness view",
            ))
        }
        (JoltCommittedPolynomial::UntrustedAdvice, JoltRelationId::AdviceClaimReduction) => {
            Ok(advice_lattice_view_formula(JoltAdviceKind::Untrusted))
        }
        (JoltCommittedPolynomial::BytecodeChunk(index), JoltRelationId::BytecodeClaimReduction) => {
            Err(unsupported_lattice_view(format!(
                "BytecodeChunk({index}) uses a separate precommitted opening, not a packed witness view"
            )))
        }
        _ => Err(unsupported_lattice_view(format!(
            "committed polynomial {polynomial:?} under relation {relation:?} has no supported lattice packed view"
        ))),
    }
}

fn unsigned_inc_chunk_index(id: JoltOpeningId) -> Option<usize> {
    let JoltOpeningId::Lattice {
        relation: JoltRelationId::UnsignedIncClaimReduction,
        index,
    } = id
    else {
        return None;
    };
    (index >= 2).then_some(index - 2)
}

fn unsigned_inc_chunk_lattice_view_formula<F>(
    index: usize,
    point: &[F],
    log_k_chunk: usize,
) -> Result<LatticePackedViewFormula<F>, VerifierError>
where
    F: Field,
{
    if log_k_chunk == 0 {
        return Err(unsupported_lattice_view(
            "unsigned increment chunk view requires a nonzero chunk size",
        ));
    }
    if point.len() < log_k_chunk {
        return Err(unsupported_lattice_view(format!(
            "unsigned increment chunk opening point has {} variables but needs at least {log_k_chunk}",
            point.len()
        )));
    }
    ra_lattice_view_formula(
        LatticePackedFamilyId::UnsignedIncChunk { index },
        point,
        log_k_chunk,
    )
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

pub fn lattice_packing_view_formula<F>(
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
            lattice_packing_family_id(family),
            *limb,
            *symbol,
        )),
        LatticePackedViewFormula::LinearDecoded { terms } => Ok(PackedViewFormula::linear_decoded(
            terms
                .iter()
                .map(|term| {
                    PackedViewTerm::new(
                        term.coefficient,
                        lattice_packing_family_id(&term.family),
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
                            lattice_packing_family_id(&term.family),
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

fn lattice_packing_advice_kind(kind: JoltAdviceKind) -> PackedAdviceKind {
    match kind {
        JoltAdviceKind::Trusted => PackedAdviceKind::Trusted,
        JoltAdviceKind::Untrusted => PackedAdviceKind::Untrusted,
    }
}

fn advice_family(
    kind: JoltAdviceKind,
    layout: &AdviceClaimReductionLayout,
) -> Result<PackedFamilySpec, VerifierError> {
    let packed_kind = lattice_packing_advice_kind(kind);
    let requirement = advice_bytes_validity_requirement(kind);
    Ok(PackedFamilySpec::direct(
        lattice_packing_family_id(&requirement.family),
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
            LatticePackedValidityKind::BytecodeStoreRdDisjoint
                | LatticePackedValidityKind::FieldElementCanonicalBytes { .. }
        ) {
            continue;
        }
        specs.push(PackedFamilySpec::direct(
            lattice_packing_family_id(&requirement.family),
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
#[path = "lattice_tests.rs"]
mod tests;
