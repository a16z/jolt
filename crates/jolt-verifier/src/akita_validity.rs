use crate::{
    akita::{
        prove_akita_packing_openings, AkitaClearVectorCommitment, AkitaJoltProof,
        AkitaPackingBatchProof, AkitaPackingProverSetup, AkitaPackingWitnessArtifacts,
        AkitaVerifierPreprocessing,
    },
    akita_packing::AkitaPackingScheme,
    akita_validation::validate_akita_artifacts_for_proof,
    proof::{ClearOnlyCommitment, JoltProofClaims},
    stages::{
        stage7::inputs::LatticePackedValidityOutputClaims,
        stage8::{
            build_lattice_packed_validity_batch, derive_lattice_packed_validity_requirements,
            derive_lattice_packed_validity_statements, field_element_canonical_factors,
            field_element_canonical_value_from_openings, lattice_packed_validity_claims,
            lattice_packed_validity_opening_count, lattice_packing_family_id,
            sample_lattice_packed_validity_eq_points, FieldCanonicalFactor,
            LatticePackedValidityStatement, LatticePackedValidityStatementKind,
        },
        PrecommittedSchedule,
    },
    VerifierError,
};
use common::jolt_device::JoltDevice;
use jolt_akita::{AkitaCommitment, AkitaField};
use jolt_claims::protocols::jolt::{
    LatticePackedFamilyId, LatticePackedValidityKind, LatticePackedValidityRequirement,
};
use jolt_openings::{PackedFamilyId, PackedWitnessLayout, PackedWitnessSource};
use jolt_poly::{try_eq_mle, EqPolynomial, UnivariatePoly};
use jolt_riscv::CircuitFlags;
use jolt_sumcheck::{
    append_sumcheck_claim, BatchedEvaluationClaim, ClearProof, CompressedLabeledRoundPoly,
    CompressedSumcheckProof, EvaluationClaim, RoundMessage, SumcheckProof,
    SUMCHECK_ROUND_TRANSCRIPT_LABEL,
};
use jolt_transcript::Transcript;

#[derive(Clone, Debug)]
pub struct AkitaPackingValidityProofArtifacts {
    pub sumcheck_proof: SumcheckProof<AkitaField, ClearOnlyCommitment>,
    pub opening_claims: LatticePackedValidityOutputClaims<AkitaField>,
    pub opening_proof: AkitaPackingBatchProof,
}

pub fn prove_akita_packing_validity<T, S>(
    setup: &AkitaPackingProverSetup,
    transcript: &mut T,
    artifacts: &AkitaPackingWitnessArtifacts,
    source: &S,
    log_k_chunk: usize,
    precommitted: &PrecommittedSchedule,
) -> Result<AkitaPackingValidityProofArtifacts, VerifierError>
where
    T: Transcript<Challenge = AkitaField>,
    S: PackedWitnessSource<AkitaField>,
{
    if source.layout() != &artifacts.layout {
        return Err(
            VerifierError::LatticePackedValidityOpeningVerificationFailed {
                reason: "Akita packing validity source layout does not match committed artifact"
                    .to_string(),
            },
        );
    }

    let requirements = derive_lattice_packed_validity_requirements(
        &artifacts.protocol,
        log_k_chunk,
        precommitted,
    )?;
    let statements = derive_lattice_packed_validity_statements(&artifacts.layout, &requirements)?;
    let eq_points =
        sample_lattice_packed_validity_eq_points(transcript, &artifacts.layout, &statements);
    let sumcheck_claims = lattice_packed_validity_claims::<AkitaField>(&statements);
    for claim in &sumcheck_claims {
        append_sumcheck_claim(transcript, &claim.claimed_sum);
    }
    let batching_coefficients = (0..statements.len())
        .map(|_| transcript.challenge_scalar())
        .collect::<Vec<_>>();
    let max_num_vars = statements
        .iter()
        .map(|statement| statement.num_vars)
        .max()
        .ok_or_else(|| VerifierError::LatticePackedValiditySumcheckFailed {
            reason: "cannot prove an empty Akita packing validity batch".to_string(),
        })?;
    let max_degree = statements
        .iter()
        .map(|statement| statement.degree)
        .max()
        .ok_or_else(|| VerifierError::LatticePackedValiditySumcheckFailed {
            reason: "cannot prove an empty Akita packing validity batch".to_string(),
        })?;

    let (compressed, reduction) = prove_combined_validity_sumcheck(
        source,
        &statements,
        &eq_points,
        &batching_coefficients,
        max_num_vars,
        max_degree,
        transcript,
    )?;
    let mut opening_claims = Vec::with_capacity(lattice_packed_validity_opening_count(&statements));
    for statement in &statements {
        let point = reduction
            .try_instance_point(statement.num_vars)
            .map_err(|error| VerifierError::LatticePackedValiditySumcheckFailed {
                reason: error.to_string(),
            })?;
        opening_claims.extend(validity_opening_values(source, statement, point)?);
    }
    let batch = build_lattice_packed_validity_batch(
        &artifacts.layout,
        &statements,
        artifacts
            .payload()
            .ok_or_else(
                || VerifierError::LatticePackedValidityOpeningVerificationFailed {
                    reason: "Akita packing validity artifacts do not carry a lattice payload"
                        .to_string(),
                },
            )?
            .packed_witness
            .clone(),
        &eq_points,
        &reduction,
        &opening_claims,
    )?;
    if reduction.reduction.value != batch.expected_final_claim {
        return Err(VerifierError::LatticePackedValidityOutputMismatch);
    }
    let opening_proof =
        prove_akita_packing_openings(setup, transcript, artifacts, source, &batch.statement)?;

    Ok(AkitaPackingValidityProofArtifacts {
        sumcheck_proof: SumcheckProof::Clear(ClearProof::Compressed(compressed)),
        opening_claims: LatticePackedValidityOutputClaims { opening_claims },
        opening_proof,
    })
}

pub fn attach_akita_packing_validity_proof(
    proof: &mut AkitaJoltProof,
    validity: AkitaPackingValidityProofArtifacts,
) -> Result<(), VerifierError> {
    proof.stages.lattice_packed_validity_sumcheck_proof = Some(validity.sumcheck_proof);
    proof.lattice_packed_validity_opening_proof = Some(validity.opening_proof);
    let JoltProofClaims::Clear(claims) = &mut proof.claims else {
        return Err(VerifierError::UnexpectedBlindFoldProof);
    };
    claims.stage7.lattice_packed_validity = Some(validity.opening_claims);
    Ok(())
}

pub fn prove_akita_jolt_packed_validity<T, S>(
    setup: &AkitaPackingProverSetup,
    preprocessing: &AkitaVerifierPreprocessing,
    public_io: &JoltDevice,
    proof: &AkitaJoltProof,
    trusted_advice_commitment: Option<&AkitaCommitment>,
    artifacts: &AkitaPackingWitnessArtifacts,
    source: &S,
) -> Result<AkitaPackingValidityProofArtifacts, VerifierError>
where
    T: Transcript<Challenge = AkitaField>,
    S: PackedWitnessSource<AkitaField>,
{
    validate_akita_artifacts_for_proof(
        &preprocessing.pcs_setup,
        &proof.protocol,
        &proof.commitments,
        artifacts,
    )?;
    let (checked, mut transcript) =
        crate::verifier::lattice_packed_validity_transcript_with_config::<
            AkitaField,
            AkitaPackingScheme,
            AkitaClearVectorCommitment,
            T,
            _,
        >(
            preprocessing,
            public_io,
            proof,
            trusted_advice_commitment,
            &artifacts.protocol,
        )?;
    prove_akita_packing_validity(
        setup,
        &mut transcript,
        artifacts,
        source,
        proof.one_hot_config.committed_chunk_bits(),
        &checked.precommitted,
    )
}

fn prove_combined_validity_sumcheck<T, S>(
    source: &S,
    statements: &[LatticePackedValidityStatement],
    eq_points: &[Vec<AkitaField>],
    batching_coefficients: &[AkitaField],
    max_num_vars: usize,
    max_degree: usize,
    transcript: &mut T,
) -> Result<
    (
        CompressedSumcheckProof<AkitaField>,
        BatchedEvaluationClaim<AkitaField>,
    ),
    VerifierError,
>
where
    T: Transcript<Challenge = AkitaField>,
    S: PackedWitnessSource<AkitaField>,
{
    let mut challenges = Vec::with_capacity(max_num_vars);
    let mut round_polynomials = Vec::with_capacity(max_num_vars);
    for _ in 0..max_num_vars {
        let remaining = max_num_vars - challenges.len() - 1;
        let round_evals = (0..=max_degree)
            .map(|point| {
                let mut prefix = challenges.clone();
                prefix.push(AkitaField::from_u64(point as u64));
                sum_combined_validity_over_suffix(
                    source,
                    statements,
                    eq_points,
                    batching_coefficients,
                    max_num_vars,
                    &prefix,
                    remaining,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let round_poly = UnivariatePoly::from_evals(&round_evals);
        let compressed =
            CompressedLabeledRoundPoly::new(&round_poly, SUMCHECK_ROUND_TRANSCRIPT_LABEL);
        <CompressedLabeledRoundPoly<'_, AkitaField> as RoundMessage>::append_to_transcript(
            &compressed,
            transcript,
        );
        let challenge = transcript.challenge();
        round_polynomials.push(round_poly.compress());
        challenges.push(challenge);
    }

    let value = combined_validity_value(
        source,
        statements,
        eq_points,
        batching_coefficients,
        max_num_vars,
        &challenges,
    )?;

    Ok((
        CompressedSumcheckProof { round_polynomials },
        BatchedEvaluationClaim {
            reduction: EvaluationClaim::new(challenges, value),
            batching_coefficients: batching_coefficients.to_vec(),
            max_num_vars,
            max_degree,
        },
    ))
}

fn sum_combined_validity_over_suffix<S>(
    source: &S,
    statements: &[LatticePackedValidityStatement],
    eq_points: &[Vec<AkitaField>],
    batching_coefficients: &[AkitaField],
    max_num_vars: usize,
    prefix: &[AkitaField],
    remaining: usize,
) -> Result<AkitaField, VerifierError>
where
    S: PackedWitnessSource<AkitaField>,
{
    let suffix_count = checked_power_of_two(remaining, "packed validity suffix")?;
    let mut sum = AkitaField::zero();
    for suffix in 0..suffix_count {
        let mut point = prefix.to_vec();
        append_boolean_bits(&mut point, suffix, remaining);
        sum += combined_validity_value(
            source,
            statements,
            eq_points,
            batching_coefficients,
            max_num_vars,
            &point,
        )?;
    }
    Ok(sum)
}

fn combined_validity_value<S>(
    source: &S,
    statements: &[LatticePackedValidityStatement],
    eq_points: &[Vec<AkitaField>],
    batching_coefficients: &[AkitaField],
    max_num_vars: usize,
    point: &[AkitaField],
) -> Result<AkitaField, VerifierError>
where
    S: PackedWitnessSource<AkitaField>,
{
    let mut value = AkitaField::zero();
    for ((statement, eq_point), coefficient) in
        statements.iter().zip(eq_points).zip(batching_coefficients)
    {
        let offset = max_num_vars
            .checked_sub(statement.num_vars)
            .ok_or_else(|| VerifierError::LatticePackedValiditySumcheckFailed {
                reason: "packed validity statement has more variables than the combined batch"
                    .to_string(),
            })?;
        let instance_point = point
            .get(offset..offset + statement.num_vars)
            .ok_or_else(|| VerifierError::LatticePackedValiditySumcheckFailed {
                reason: "packed validity instance point is out of range".to_string(),
            })?;
        value += *coefficient * validity_value(source, statement, eq_point, instance_point)?;
    }
    Ok(value)
}

pub(crate) fn validity_value<S>(
    source: &S,
    statement: &LatticePackedValidityStatement,
    eq_point: &[AkitaField],
    point: &[AkitaField],
) -> Result<AkitaField, VerifierError>
where
    S: PackedWitnessSource<AkitaField>,
{
    let eq_mask = try_eq_mle(point, eq_point).map_err(|error| {
        VerifierError::LatticePackedValiditySumcheckFailed {
            reason: error.to_string(),
        }
    })?;
    let value = match statement.kind {
        LatticePackedValidityStatementKind::BytecodeStoreRdDisjoint => {
            let openings = validity_opening_values(source, statement, point)?;
            openings[0] * openings[1]
        }
        LatticePackedValidityStatementKind::FieldElementCanonicalBytes => {
            let openings = validity_opening_values(source, statement, point)?;
            field_element_canonical_value_from_openings(statement, &openings)?
        }
        _ => validity_violation(
            statement.kind,
            validity_opening_value(source, statement, point)?,
        ),
    };
    Ok(eq_mask * value)
}

fn validity_opening_values<S>(
    source: &S,
    statement: &LatticePackedValidityStatement,
    point: &[AkitaField],
) -> Result<Vec<AkitaField>, VerifierError>
where
    S: PackedWitnessSource<AkitaField>,
{
    if statement.kind == LatticePackedValidityStatementKind::BytecodeStoreRdDisjoint {
        return Ok(vec![
            bytecode_store_rd_disjoint_factor_value(source, statement, point, 0)?,
            bytecode_store_rd_disjoint_factor_value(source, statement, point, 1)?,
        ]);
    }
    if statement.kind == LatticePackedValidityStatementKind::FieldElementCanonicalBytes {
        let factors = field_element_canonical_factors(&statement.requirement)?;
        return factors
            .into_iter()
            .map(|factor| field_element_canonical_factor_value(source, point, factor))
            .collect();
    }
    validity_opening_value(source, statement, point).map(|value| vec![value])
}

fn validity_opening_value<S>(
    source: &S,
    statement: &LatticePackedValidityStatement,
    point: &[AkitaField],
) -> Result<AkitaField, VerifierError>
where
    S: PackedWitnessSource<AkitaField>,
{
    let family_id = lattice_packing_family_id(&statement.requirement.family);
    let shape = validity_statement_shape(source.layout(), statement, &family_id)?;
    let point_parts = split_validity_point(statement.kind, point, shape)?;
    let row_weights = EqPolynomial::<AkitaField>::evals(point_parts.row, None);
    let limb_weights = EqPolynomial::<AkitaField>::evals(point_parts.limb, None);
    match statement.kind {
        LatticePackedValidityStatementKind::CellBooleanity => {
            let symbol_weights = EqPolynomial::<AkitaField>::evals(point_parts.symbol, None);
            weighted_family_value(
                source,
                &family_id,
                &row_weights,
                &limb_weights,
                SymbolWeights::Point(&symbol_weights),
            )
        }
        LatticePackedValidityStatementKind::ExactOneHotRowSum
        | LatticePackedValidityStatementKind::OptionalOneHotRowSum => weighted_family_value(
            source,
            &family_id,
            &row_weights,
            &limb_weights,
            SymbolWeights::All,
        ),
        LatticePackedValidityStatementKind::BooleanIndicator => {
            let LatticePackedValidityKind::BooleanIndicator { symbol } = statement.requirement.kind
            else {
                return Err(VerifierError::InvalidProtocolConfig {
                    reason: "boolean-indicator validity statement has non-indicator requirement"
                        .to_string(),
                });
            };
            weighted_family_value(
                source,
                &family_id,
                &row_weights,
                &limb_weights,
                SymbolWeights::Fixed(symbol),
            )
        }
        LatticePackedValidityStatementKind::BytecodeStoreRdDisjoint => {
            Err(VerifierError::InvalidProtocolConfig {
                reason: "bytecode Store/Rd disjointness has multiple opening factors".to_string(),
            })
        }
        LatticePackedValidityStatementKind::FieldElementCanonicalBytes => {
            Err(VerifierError::InvalidProtocolConfig {
                reason: "field-element canonical-byte validity has multiple opening factors"
                    .to_string(),
            })
        }
    }
}

fn validity_violation(kind: LatticePackedValidityStatementKind, opening: AkitaField) -> AkitaField {
    match kind {
        LatticePackedValidityStatementKind::CellBooleanity
        | LatticePackedValidityStatementKind::BooleanIndicator
        | LatticePackedValidityStatementKind::OptionalOneHotRowSum => {
            opening * (opening - AkitaField::one())
        }
        LatticePackedValidityStatementKind::ExactOneHotRowSum => {
            let difference = opening - AkitaField::one();
            difference * difference
        }
        LatticePackedValidityStatementKind::BytecodeStoreRdDisjoint
        | LatticePackedValidityStatementKind::FieldElementCanonicalBytes => opening,
    }
}

fn weighted_direct_symbol_value<S>(
    source: &S,
    family_id: &PackedFamilyId,
    row_weights: &[AkitaField],
    symbol: usize,
) -> Result<AkitaField, VerifierError>
where
    S: PackedWitnessSource<AkitaField>,
{
    weighted_direct_limb_symbol_value(source, family_id, row_weights, 0, symbol)
}

fn weighted_direct_limb_symbol_value<S>(
    source: &S,
    family_id: &PackedFamilyId,
    row_weights: &[AkitaField],
    limb: usize,
    symbol: usize,
) -> Result<AkitaField, VerifierError>
where
    S: PackedWitnessSource<AkitaField>,
{
    let mut value = AkitaField::zero();
    let mut error = None;
    source.for_each_nonzero(|rank, cell| {
        if error.is_some() {
            return;
        }
        let Some(address) = source.layout().unrank(rank) else {
            error = Some(
                VerifierError::LatticePackedValidityOpeningVerificationFailed {
                    reason: format!("packed validity source emitted out-of-layout rank {rank}"),
                },
            );
            return;
        };
        if &address.family != family_id || address.limb != limb || address.symbol != symbol {
            return;
        }
        let Some(row_weight) = row_weights.get(address.row).copied() else {
            error = Some(
                VerifierError::LatticePackedValidityOpeningVerificationFailed {
                    reason: format!("packed validity row {} is outside row weights", address.row),
                },
            );
            return;
        };
        value += row_weight * cell;
    });
    if let Some(error) = error {
        return Err(error);
    }
    Ok(value)
}

fn field_element_canonical_factor_value<S>(
    source: &S,
    point: &[AkitaField],
    factor: FieldCanonicalFactor,
) -> Result<AkitaField, VerifierError>
where
    S: PackedWitnessSource<AkitaField>,
{
    let (family, limb, symbol_filter) = match factor {
        FieldCanonicalFactor::Eq {
            family,
            limb,
            symbol,
            ..
        } => {
            return weighted_field_canonical_symbol_value(source, point, &family, limb, symbol);
        }
        FieldCanonicalFactor::Range {
            family,
            limb,
            start_symbol,
            ..
        } => (family, limb, start_symbol..256),
    };

    let mut value = AkitaField::zero();
    for symbol in symbol_filter {
        value += weighted_field_canonical_symbol_value(source, point, &family, limb, symbol)?;
    }
    Ok(value)
}

fn weighted_field_canonical_symbol_value<S>(
    source: &S,
    point: &[AkitaField],
    family_id: &PackedFamilyId,
    limb: usize,
    symbol: usize,
) -> Result<AkitaField, VerifierError>
where
    S: PackedWitnessSource<AkitaField>,
{
    let family =
        source
            .layout()
            .family(family_id)
            .ok_or_else(|| VerifierError::InvalidProtocolConfig {
                reason: format!("field-element canonical-byte factor requires {family_id:?}"),
            })?;
    if limb >= family.limbs || family.alphabet.size() != 256 {
        return Err(VerifierError::InvalidProtocolConfig {
            reason: format!(
                "field-element canonical-byte factor {family_id:?} must be a byte family"
            ),
        });
    }
    let rows = family
        .domain
        .rows()
        .map_err(|error| VerifierError::InvalidProtocolConfig {
            reason: format!(
                "field-element canonical-byte factor {family_id:?} has invalid row domain: {error}"
            ),
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
    let row_weights = EqPolynomial::<AkitaField>::evals(point, None);
    weighted_direct_limb_symbol_value(source, family_id, &row_weights, limb, symbol)
}

fn bytecode_store_rd_disjoint_factor_value<S>(
    source: &S,
    statement: &LatticePackedValidityStatement,
    point: &[AkitaField],
    factor: usize,
) -> Result<AkitaField, VerifierError>
where
    S: PackedWitnessSource<AkitaField>,
{
    let chunk = bytecode_store_rd_disjoint_chunk(&statement.requirement)?;
    let store_id = PackedFamilyId::BytecodeCircuitFlag {
        chunk,
        flag: CircuitFlags::Store as usize,
    };
    let store =
        source
            .layout()
            .family(&store_id)
            .ok_or_else(|| VerifierError::InvalidProtocolConfig {
                reason: format!("bytecode Store/Rd disjointness requires {store_id:?}"),
            })?;
    let rows = store
        .domain
        .rows()
        .map_err(|error| VerifierError::InvalidProtocolConfig {
            reason: format!("bytecode Store/Rd disjointness row domain is invalid: {error}"),
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
    let row_weights = EqPolynomial::<AkitaField>::evals(point, None);
    match factor {
        0 => weighted_direct_symbol_value(source, &store_id, &row_weights, 1),
        1 => {
            let rd_id = PackedFamilyId::BytecodeRegisterSelector { chunk, selector: 2 };
            let rd = source.layout().family(&rd_id).ok_or_else(|| {
                VerifierError::InvalidProtocolConfig {
                    reason: format!("bytecode Store/Rd disjointness requires {rd_id:?}"),
                }
            })?;
            if rd.domain != store.domain || rd.limbs != 1 {
                return Err(VerifierError::InvalidProtocolConfig {
                    reason: "bytecode Store/Rd disjointness rd selector layout mismatch"
                        .to_string(),
                });
            }
            let limb_weights = [AkitaField::one()];
            weighted_family_value(
                source,
                &rd_id,
                &row_weights,
                &limb_weights,
                SymbolWeights::All,
            )
        }
        _ => Err(VerifierError::InvalidProtocolConfig {
            reason: format!("bytecode Store/Rd disjointness has no opening factor {factor}"),
        }),
    }
}

fn bytecode_store_rd_disjoint_chunk(
    requirement: &LatticePackedValidityRequirement,
) -> Result<usize, VerifierError> {
    let LatticePackedFamilyId::BytecodeCircuitFlag { chunk, flag } = &requirement.family else {
        return Err(VerifierError::InvalidProtocolConfig {
            reason: "bytecode Store/Rd disjointness must be anchored on the Store circuit flag"
                .to_string(),
        });
    };
    if *flag != CircuitFlags::Store as usize
        || requirement.limbs != 1
        || requirement.alphabet_size != 2
    {
        return Err(VerifierError::InvalidProtocolConfig {
            reason:
                "bytecode Store/Rd disjointness must be anchored on a boolean Store circuit flag"
                    .to_string(),
        });
    }
    Ok(*chunk)
}

#[derive(Clone, Copy)]
struct ValidityStatementShape {
    row: usize,
    limb: usize,
    symbol: usize,
}

struct ValidityPointParts<'a> {
    row: &'a [AkitaField],
    limb: &'a [AkitaField],
    symbol: &'a [AkitaField],
}

fn validity_statement_shape(
    layout: &PackedWitnessLayout,
    statement: &LatticePackedValidityStatement,
    family_id: &PackedFamilyId,
) -> Result<ValidityStatementShape, VerifierError> {
    let family = layout
        .family(family_id)
        .ok_or_else(|| VerifierError::InvalidProtocolConfig {
            reason: format!("packed validity statement references missing family {family_id:?}"),
        })?;
    if family.limbs != statement.requirement.limbs {
        return Err(VerifierError::InvalidProtocolConfig {
            reason: format!(
                "packed validity family {family_id:?} limb count mismatch: layout has {}, statement has {}",
                family.limbs, statement.requirement.limbs
            ),
        });
    }
    if family.alphabet.size() != statement.requirement.alphabet_size {
        return Err(VerifierError::InvalidProtocolConfig {
            reason: format!(
                "packed validity family {family_id:?} alphabet mismatch: layout has {}, statement has {}",
                family.alphabet.size(),
                statement.requirement.alphabet_size
            ),
        });
    }
    let rows = family
        .domain
        .rows()
        .map_err(|error| VerifierError::InvalidProtocolConfig {
            reason: format!("packed validity family {family_id:?} has invalid row domain: {error}"),
        })?;
    Ok(ValidityStatementShape {
        row: power_of_two_log(rows, "packed validity row count")?,
        limb: power_of_two_log(statement.requirement.limbs, "packed validity limb count")?,
        symbol: power_of_two_log(
            statement.requirement.alphabet_size,
            "packed validity alphabet size",
        )?,
    })
}

fn split_validity_point(
    kind: LatticePackedValidityStatementKind,
    point: &[AkitaField],
    shape: ValidityStatementShape,
) -> Result<ValidityPointParts<'_>, VerifierError> {
    let expected = match kind {
        LatticePackedValidityStatementKind::CellBooleanity => shape.row + shape.limb + shape.symbol,
        LatticePackedValidityStatementKind::ExactOneHotRowSum
        | LatticePackedValidityStatementKind::OptionalOneHotRowSum
        | LatticePackedValidityStatementKind::BooleanIndicator => shape.row + shape.limb,
        LatticePackedValidityStatementKind::BytecodeStoreRdDisjoint
        | LatticePackedValidityStatementKind::FieldElementCanonicalBytes => shape.row,
    };
    if point.len() != expected {
        return Err(VerifierError::LatticePackedValiditySumcheckFailed {
            reason: format!(
                "packed validity point has {} variables but statement requires {expected}",
                point.len()
            ),
        });
    }
    let row_end = shape.row;
    let limb_end = row_end + shape.limb;
    let symbol_end = limb_end + shape.symbol;
    let symbol = if matches!(kind, LatticePackedValidityStatementKind::CellBooleanity) {
        &point[limb_end..symbol_end]
    } else {
        &[]
    };
    Ok(ValidityPointParts {
        row: &point[..row_end],
        limb: &point[row_end..limb_end],
        symbol,
    })
}

enum SymbolWeights<'a> {
    Point(&'a [AkitaField]),
    All,
    Fixed(usize),
}

fn weighted_family_value<S>(
    source: &S,
    family_id: &PackedFamilyId,
    row_weights: &[AkitaField],
    limb_weights: &[AkitaField],
    symbol_weights: SymbolWeights<'_>,
) -> Result<AkitaField, VerifierError>
where
    S: PackedWitnessSource<AkitaField>,
{
    let mut value = AkitaField::zero();
    let mut error = None;
    source.for_each_nonzero(|rank, cell| {
        if error.is_some() {
            return;
        }
        let Some(address) = source.layout().unrank(rank) else {
            error = Some(
                VerifierError::LatticePackedValidityOpeningVerificationFailed {
                    reason: format!("packed validity source emitted out-of-layout rank {rank}"),
                },
            );
            return;
        };
        if &address.family != family_id {
            return;
        }
        let Some(row_weight) = row_weights.get(address.row).copied() else {
            error = Some(
                VerifierError::LatticePackedValidityOpeningVerificationFailed {
                    reason: format!("packed validity row {} is outside row weights", address.row),
                },
            );
            return;
        };
        let Some(limb_weight) = limb_weights.get(address.limb).copied() else {
            error = Some(
                VerifierError::LatticePackedValidityOpeningVerificationFailed {
                    reason: format!(
                        "packed validity limb {} is outside limb weights",
                        address.limb
                    ),
                },
            );
            return;
        };
        let symbol_weight = match symbol_weights {
            SymbolWeights::Point(weights) => {
                let Some(weight) = weights.get(address.symbol).copied() else {
                    error = Some(
                        VerifierError::LatticePackedValidityOpeningVerificationFailed {
                            reason: format!(
                                "packed validity symbol {} is outside symbol weights",
                                address.symbol
                            ),
                        },
                    );
                    return;
                };
                weight
            }
            SymbolWeights::All => AkitaField::one(),
            SymbolWeights::Fixed(symbol) => {
                if address.symbol == symbol {
                    AkitaField::one()
                } else {
                    AkitaField::zero()
                }
            }
        };
        value += row_weight * limb_weight * symbol_weight * cell;
    });
    if let Some(error) = error {
        return Err(error);
    }
    Ok(value)
}

fn append_boolean_bits(point: &mut Vec<AkitaField>, index: usize, bits: usize) {
    for bit in 0..bits {
        let shift = bits - 1 - bit;
        point.push(AkitaField::from_u64(((index >> shift) & 1) as u64));
    }
}

fn checked_power_of_two(bits: usize, name: &'static str) -> Result<usize, VerifierError> {
    1usize.checked_shl(bits as u32).ok_or_else(|| {
        VerifierError::LatticePackedValiditySumcheckFailed {
            reason: format!("{name} dimension is too large"),
        }
    })
}

fn power_of_two_log(value: usize, name: &'static str) -> Result<usize, VerifierError> {
    if value.is_power_of_two() {
        Ok(value.trailing_zeros() as usize)
    } else {
        Err(VerifierError::InvalidProtocolConfig {
            reason: format!("{name} must be a power of two, got {value}"),
        })
    }
}
