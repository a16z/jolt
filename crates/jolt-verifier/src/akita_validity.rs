use crate::{
    akita::{
        prove_akita_packing_openings, AkitaClearVectorCommitment, AkitaJoltProof,
        AkitaPackingBatchProof, AkitaPackingProverSetup, AkitaPackingWitnessArtifacts,
        AkitaVerifierPreprocessing,
    },
    akita_packing::AkitaPackingScheme,
    akita_validation::validate_akita_artifacts_for_proof,
    akita_validity_sumcheck::prove_combined_validity_sumcheck,
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
use jolt_openings::{PackingFamilyId, PackingWitnessLayout, PackingWitnessSource};
use jolt_poly::{try_eq_mle, EqPolynomial};
use jolt_riscv::CircuitFlags;
use jolt_sumcheck::{append_sumcheck_claim, ClearProof, SumcheckProof};
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
    S: PackingWitnessSource<AkitaField>,
{
    if source.layout() != &artifacts.layout {
        return Err(
            VerifierError::LatticePackedValidityOpeningVerificationFailed {
                reason: "lattice packed validity source layout does not match committed artifact"
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
            reason: "cannot prove an empty lattice packed validity batch".to_string(),
        })?;
    let max_degree = statements
        .iter()
        .map(|statement| statement.degree)
        .max()
        .ok_or_else(|| VerifierError::LatticePackedValiditySumcheckFailed {
            reason: "cannot prove an empty lattice packed validity batch".to_string(),
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
                    reason: "lattice packed validity artifacts do not carry a lattice payload"
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
    S: PackingWitnessSource<AkitaField>,
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

pub(crate) fn validity_value<S>(
    source: &S,
    statement: &LatticePackedValidityStatement,
    eq_point: &[AkitaField],
    point: &[AkitaField],
) -> Result<AkitaField, VerifierError>
where
    S: PackingWitnessSource<AkitaField>,
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
    S: PackingWitnessSource<AkitaField>,
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
    S: PackingWitnessSource<AkitaField>,
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
    family_id: &PackingFamilyId,
    row_weights: &[AkitaField],
    symbol: usize,
) -> Result<AkitaField, VerifierError>
where
    S: PackingWitnessSource<AkitaField>,
{
    weighted_direct_limb_symbol_value(source, family_id, row_weights, 0, symbol)
}

fn weighted_direct_limb_symbol_value<S>(
    source: &S,
    family_id: &PackingFamilyId,
    row_weights: &[AkitaField],
    limb: usize,
    symbol: usize,
) -> Result<AkitaField, VerifierError>
where
    S: PackingWitnessSource<AkitaField>,
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
    S: PackingWitnessSource<AkitaField>,
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
    family_id: &PackingFamilyId,
    limb: usize,
    symbol: usize,
) -> Result<AkitaField, VerifierError>
where
    S: PackingWitnessSource<AkitaField>,
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
    S: PackingWitnessSource<AkitaField>,
{
    let chunk = bytecode_store_rd_disjoint_chunk(&statement.requirement)?;
    let store_id = PackingFamilyId::BytecodeCircuitFlag {
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
            let rd_id = PackingFamilyId::BytecodeRegisterSelector { chunk, selector: 2 };
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
    layout: &PackingWitnessLayout,
    statement: &LatticePackedValidityStatement,
    family_id: &PackingFamilyId,
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
    family_id: &PackingFamilyId,
    row_weights: &[AkitaField],
    limb_weights: &[AkitaField],
    symbol_weights: SymbolWeights<'_>,
) -> Result<AkitaField, VerifierError>
where
    S: PackingWitnessSource<AkitaField>,
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

fn power_of_two_log(value: usize, name: &'static str) -> Result<usize, VerifierError> {
    if value.is_power_of_two() {
        Ok(value.trailing_zeros() as usize)
    } else {
        Err(VerifierError::InvalidProtocolConfig {
            reason: format!("{name} must be a power of two, got {value}"),
        })
    }
}

#[cfg(test)]
mod tests {
    #![expect(
        clippy::expect_used,
        reason = "tests assert successful validity fixture construction"
    )]

    use super::*;
    use crate::{
        akita::{commit_akita_packing_witness_with_config, AkitaPackingVerifierSetup},
        akita_packing::AkitaPackingScheme,
        config::{IncrementCommitmentMode, JoltProtocolConfig, PcsFamily, ProgramMode},
        proof::ClearOnlyCommitment,
        stages::{
            stage8::{
                derive_lattice_packed_witness_layout,
                lattice_protocol_config_for_packed_witness_layout,
                lattice_validity_requirements_for_packed_witness_layout,
            },
            CommittedProgramSchedule, PrecommittedSchedule,
        },
    };
    use jolt_akita::AkitaSetupParams;
    use jolt_claims::protocols::jolt::{
        bytecode_imm_canonical_bytes_requirement,
        formulas::{
            dimensions::{TracePolynomialOrder, REGISTER_ADDRESS_BITS},
            ra::JoltRaPolynomialLayout,
        },
        lattice_packed_validity_digest, JoltAdviceKind,
    };
    use jolt_field::FixedByteSize;
    use jolt_openings::{
        CommitmentScheme, PackingAdviceKind, PackingAlphabet, PackingCellAddress,
        PackingFactDomain, PackingFamilySpec, PackingSetupParams, SparsePackingWitness,
    };
    use jolt_transcript::{Blake2bTranscript, Transcript};

    fn run_on_large_stack(test: impl FnOnce() + Send + 'static) {
        std::thread::Builder::new()
            .stack_size(256 * 1024 * 1024)
            .spawn(test)
            .expect("failed to spawn test thread")
            .join()
            .expect("test thread panicked");
    }

    fn akita_packing_params(
        layout: &PackingWitnessLayout,
        max_num_polys_per_commitment_group: usize,
    ) -> PackingSetupParams<AkitaSetupParams, PackingWitnessLayout> {
        PackingSetupParams {
            pcs: AkitaSetupParams::new(
                layout.dimension,
                max_num_polys_per_commitment_group,
                layout.digest,
            ),
            layout: layout.clone(),
        }
    }

    #[test]
    #[cfg_attr(
        feature = "field-inline",
        ignore = "field-inline canonical-byte validity makes the real Akita proof fixture expensive; run explicitly with --run-ignored"
    )]
    fn packed_validity_helper_proves_real_akita_opening_proof() {
        run_on_large_stack(|| {
            let log_t = 0;
            let log_k_chunk = 1;
            let precommitted = PrecommittedSchedule::new(
                TracePolynomialOrder::CycleMajor,
                log_t,
                log_k_chunk,
                None,
                None,
                Some(CommittedProgramSchedule {
                    bytecode_len: 1,
                    bytecode_chunk_count: 1,
                    program_image_len_words: 1,
                    program_image_start_index: 0,
                }),
            )
            .expect("precommitted schedule should build");
            let mut config = JoltProtocolConfig::for_zk(false).with_pcs_family(PcsFamily::Lattice);
            config.lattice.program_mode = ProgramMode::Committed;
            config.lattice.increment_mode = IncrementCommitmentMode::FusedOneHot;
            config.lattice.packed_witness.layout_digest = Some([0; 32]);
            config.lattice.packed_witness.d_pack = Some(0);
            config.lattice.packed_witness.validity_digest = Some([0; 32]);
            #[cfg(feature = "field-inline")]
            {
                config.lattice.field_inline.enabled = true;
            }

            let layout = derive_lattice_packed_witness_layout(
                &config,
                log_t,
                log_k_chunk,
                JoltRaPolynomialLayout::new(1, 1, 1).expect("RA layout should build"),
                &precommitted,
            )
            .expect("layout should derive");
            config.lattice.packed_witness.layout_digest = Some(layout.digest);
            config.lattice.packed_witness.d_pack = Some(layout.dimension);
            let requirements =
                derive_lattice_packed_validity_requirements(&config, log_k_chunk, &precommitted)
                    .expect("validity requirements should derive");
            config.lattice.packed_witness.validity_digest =
                Some(lattice_packed_validity_digest(&requirements));
            let source = validity_default_source(&layout, &requirements);
            let params = akita_packing_params(&layout, 1);
            let (prover_setup, verifier_setup) = AkitaPackingScheme::setup(params);
            let artifacts =
                commit_akita_packing_witness_with_config(config, &prover_setup, &source)
                    .expect("valid packed witness should commit");

            let mut prover_transcript = Blake2bTranscript::new(b"akita-validity");
            let validity = prove_akita_packing_validity(
                &prover_setup,
                &mut prover_transcript,
                &artifacts,
                &source,
                log_k_chunk,
                &precommitted,
            )
            .expect("validity proof should prove");

            let mut verifier_transcript = Blake2bTranscript::new(b"akita-validity");
            verify_validity_artifacts(
                &verifier_setup,
                &mut verifier_transcript,
                &artifacts,
                log_k_chunk,
                &precommitted,
                &validity,
            )
            .expect("validity proof should verify");
            assert_eq!(prover_transcript.state(), verifier_transcript.state());

            let mut tampered = validity.clone();
            tampered.opening_claims.opening_claims[0] += AkitaField::one();
            let mut tampered_transcript = Blake2bTranscript::new(b"akita-validity");
            let error = verify_validity_artifacts(
                &verifier_setup,
                &mut tampered_transcript,
                &artifacts,
                log_k_chunk,
                &precommitted,
                &tampered,
            )
            .expect_err("tampered validity opening claim should reject");
            assert!(matches!(
                error,
                VerifierError::LatticePackedValidityOutputMismatch
                    | VerifierError::LatticePackedValidityOpeningVerificationFailed { .. }
            ));
        });
    }

    #[cfg(feature = "field-inline")]
    #[test]
    #[ignore = "real Akita negative canonical-byte proof takes over two minutes; run explicitly with --run-ignored"]
    fn packed_validity_rejects_noncanonical_field_rd_inc_bytes() {
        run_on_large_stack(|| {
            let log_t = 0;
            let log_k_chunk = 1;
            let precommitted = PrecommittedSchedule::new(
                TracePolynomialOrder::CycleMajor,
                log_t,
                log_k_chunk,
                None,
                None,
                Some(CommittedProgramSchedule {
                    bytecode_len: 1,
                    bytecode_chunk_count: 1,
                    program_image_len_words: 1,
                    program_image_start_index: 0,
                }),
            )
            .expect("precommitted schedule should build");
            let mut config = JoltProtocolConfig::for_zk(false).with_pcs_family(PcsFamily::Lattice);
            config.lattice.program_mode = ProgramMode::Committed;
            config.lattice.increment_mode = IncrementCommitmentMode::FusedOneHot;
            config.lattice.field_inline.enabled = true;
            config.lattice.packed_witness.layout_digest = Some([0; 32]);
            config.lattice.packed_witness.d_pack = Some(0);
            config.lattice.packed_witness.validity_digest = Some([0; 32]);

            let layout = derive_lattice_packed_witness_layout(
                &config,
                log_t,
                log_k_chunk,
                JoltRaPolynomialLayout::new(1, 1, 1).expect("RA layout should build"),
                &precommitted,
            )
            .expect("layout should derive");
            config.lattice.packed_witness.layout_digest = Some(layout.digest);
            config.lattice.packed_witness.d_pack = Some(layout.dimension);
            let requirements =
                derive_lattice_packed_validity_requirements(&config, log_k_chunk, &precommitted)
                    .expect("validity requirements should derive");
            config.lattice.packed_witness.validity_digest =
                Some(lattice_packed_validity_digest(&requirements));

            let modulus_bytes = jolt_akita::AKITA_FIELD_MODULUS.to_le_bytes();
            let source =
                validity_source_with_field_rd_inc_bytes(&layout, &requirements, &modulus_bytes);
            let params = akita_packing_params(&layout, 1);
            let (prover_setup, verifier_setup) = AkitaPackingScheme::setup(params);
            let artifacts =
                commit_akita_packing_witness_with_config(config, &prover_setup, &source)
                    .expect("packed witness should commit");

            let mut prover_transcript = Blake2bTranscript::new(b"akita-validity");
            let validity = prove_akita_packing_validity(
                &prover_setup,
                &mut prover_transcript,
                &artifacts,
                &source,
                log_k_chunk,
                &precommitted,
            )
            .expect("invalid packed witness can still produce a proof transcript");

            let mut verifier_transcript = Blake2bTranscript::new(b"akita-validity");
            let error = verify_validity_artifacts(
                &verifier_setup,
                &mut verifier_transcript,
                &artifacts,
                log_k_chunk,
                &precommitted,
                &validity,
            )
            .expect_err("noncanonical field bytes should reject");
            assert!(matches!(
                error,
                VerifierError::LatticePackedValidityOutputMismatch
                    | VerifierError::LatticePackedValiditySumcheckFailed { .. }
                    | VerifierError::LatticePackedValidityOpeningVerificationFailed { .. }
            ));
        });
    }

    #[test]
    fn packed_validity_rejects_precommitted_bytecode_layout_config() {
        let (layout, _, requirements) = small_bytecode_validity_context();
        let source = validity_source_with_bytecode_imm_bytes(
            &layout,
            &requirements,
            &jolt_akita::AKITA_FIELD_MODULUS.to_le_bytes(),
        );
        let mut config = lattice_protocol_config_for_packed_witness_layout(&layout);
        config.lattice.packed_witness.validity_digest =
            Some(lattice_packed_validity_digest(&requirements));
        let params = akita_packing_params(&layout, 1);
        let (prover_setup, _) = AkitaPackingScheme::setup(params);

        let error = commit_akita_packing_witness_with_config(config, &prover_setup, &source)
            .expect_err("precommitted bytecode families should reject");

        assert!(matches!(
            error,
            VerifierError::InvalidProtocolConfig { reason }
                if reason.contains("precommitted family")
        ));
    }

    #[cfg(feature = "field-inline")]
    #[test]
    fn packed_validity_value_detects_noncanonical_field_rd_inc_bytes() {
        let log_t = 0;
        let log_k_chunk = 1;
        let precommitted = PrecommittedSchedule::new(
            TracePolynomialOrder::CycleMajor,
            log_t,
            log_k_chunk,
            None,
            None,
            Some(CommittedProgramSchedule {
                bytecode_len: 1,
                bytecode_chunk_count: 1,
                program_image_len_words: 1,
                program_image_start_index: 0,
            }),
        )
        .expect("precommitted schedule should build");
        let mut config = JoltProtocolConfig::for_zk(false).with_pcs_family(PcsFamily::Lattice);
        config.lattice.program_mode = ProgramMode::Committed;
        config.lattice.increment_mode = IncrementCommitmentMode::FusedOneHot;
        config.lattice.field_inline.enabled = true;
        config.lattice.packed_witness.layout_digest = Some([0; 32]);
        config.lattice.packed_witness.d_pack = Some(0);
        config.lattice.packed_witness.validity_digest = Some([0; 32]);

        let layout = derive_lattice_packed_witness_layout(
            &config,
            log_t,
            log_k_chunk,
            JoltRaPolynomialLayout::new(1, 1, 1).expect("RA layout should build"),
            &precommitted,
        )
        .expect("layout should derive");
        let requirements =
            derive_lattice_packed_validity_requirements(&config, log_k_chunk, &precommitted)
                .expect("validity requirements should derive");
        let statements = derive_lattice_packed_validity_statements(&layout, &requirements)
            .expect("validity statements should derive");
        let source = validity_source_with_field_rd_inc_bytes(
            &layout,
            &requirements,
            &jolt_akita::AKITA_FIELD_MODULUS.to_le_bytes(),
        );
        let statement = statements
            .iter()
            .find(|statement| {
                matches!(
                    statement.requirement.family,
                    LatticePackedFamilyId::FieldRdIncByte { index: 0 }
                ) && statement.kind
                    == LatticePackedValidityStatementKind::FieldElementCanonicalBytes
            })
            .expect("FieldRdInc canonical-byte statement should exist");
        let point = vec![AkitaField::zero(); statement.num_vars];
        let value = validity_value(&source, statement, &point, &point)
            .expect("validity value should evaluate");

        assert_ne!(value, AkitaField::zero());
    }

    #[test]
    fn packed_validity_value_detects_noncanonical_bytecode_imm_bytes() {
        let (layout, statements, requirements) = small_bytecode_validity_context();
        let source = validity_source_with_bytecode_imm_bytes(
            &layout,
            &requirements,
            &jolt_akita::AKITA_FIELD_MODULUS.to_le_bytes(),
        );
        let statement = statements
            .iter()
            .find(|statement| {
                matches!(
                    statement.requirement.family,
                    LatticePackedFamilyId::BytecodeImmBytes { chunk: 0 }
                ) && statement.kind
                    == LatticePackedValidityStatementKind::FieldElementCanonicalBytes
            })
            .expect("bytecode immediate canonical-byte statement should exist");
        let point = vec![AkitaField::zero(); statement.num_vars];
        let value = validity_value(&source, statement, &point, &point)
            .expect("validity value should evaluate");

        assert_ne!(value, AkitaField::zero());
    }

    #[test]
    fn packed_validity_value_detects_malformed_advice_byte_onehot() {
        let (layout, statements) = small_validity_context();
        let family = PackingFamilyId::AdviceBytes {
            kind: PackingAdviceKind::Untrusted,
            index: 0,
        };
        let source = SparsePackingWitness::try_from_cells(
            layout,
            [
                (packed_cell_at(family.clone(), 0, 0, 7), AkitaField::one()),
                (packed_cell_at(family, 0, 0, 8), AkitaField::one()),
            ],
        )
        .expect("malformed advice source should build");
        let statement = validity_statement(
            &statements,
            LatticePackedFamilyId::AdviceBytes {
                kind: JoltAdviceKind::Untrusted,
                index: 0,
            },
            LatticePackedValidityStatementKind::ExactOneHotRowSum,
        );

        assert_ne!(
            validity_value_at_zero(&source, statement),
            AkitaField::zero()
        );
    }

    #[test]
    fn packed_validity_value_detects_malformed_bytecode_optional_selector() {
        let (layout, statements, _) = small_bytecode_validity_context();
        let family = PackingFamilyId::BytecodeRegisterSelector {
            chunk: 0,
            selector: 0,
        };
        let source = SparsePackingWitness::try_from_cells(
            layout,
            [
                (packed_cell_at(family.clone(), 0, 0, 3), AkitaField::one()),
                (packed_cell_at(family, 0, 0, 4), AkitaField::one()),
            ],
        )
        .expect("malformed bytecode selector source should build");
        let statement = validity_statement(
            &statements,
            LatticePackedFamilyId::BytecodeRegisterSelector {
                chunk: 0,
                selector: 0,
            },
            LatticePackedValidityStatementKind::OptionalOneHotRowSum,
        );

        assert_ne!(
            validity_value_at_zero(&source, statement),
            AkitaField::zero()
        );
    }

    #[test]
    fn packed_validity_value_detects_malformed_bytecode_boolean_flag() {
        let (layout, statements, _) = small_bytecode_validity_context();
        let flag = CircuitFlags::Store as usize;
        let family = PackingFamilyId::BytecodeCircuitFlag { chunk: 0, flag };
        let source = SparsePackingWitness::try_from_cells(
            layout,
            [(packed_cell_at(family, 0, 0, 1), af(2))],
        )
        .expect("malformed bytecode flag source should build");
        let statement = validity_statement(
            &statements,
            LatticePackedFamilyId::BytecodeCircuitFlag { chunk: 0, flag },
            LatticePackedValidityStatementKind::BooleanIndicator,
        );

        assert_ne!(
            validity_value_at_zero(&source, statement),
            AkitaField::zero()
        );
    }

    fn small_validity_context() -> (PackingWitnessLayout, Vec<LatticePackedValidityStatement>) {
        let log_t = 0;
        let log_k_chunk = 1;
        let precommitted = PrecommittedSchedule::new(
            TracePolynomialOrder::CycleMajor,
            log_t,
            log_k_chunk,
            None,
            Some(1),
            Some(CommittedProgramSchedule {
                bytecode_len: 1,
                bytecode_chunk_count: 1,
                program_image_len_words: 1,
                program_image_start_index: 0,
            }),
        )
        .expect("precommitted schedule should build");
        let mut config = JoltProtocolConfig::for_zk(false).with_pcs_family(PcsFamily::Lattice);
        config.lattice.program_mode = ProgramMode::Committed;
        config.lattice.increment_mode = IncrementCommitmentMode::FusedOneHot;
        config.lattice.advice.untrusted = true;
        config.lattice.packed_witness.layout_digest = Some([0; 32]);
        config.lattice.packed_witness.d_pack = Some(0);
        config.lattice.packed_witness.validity_digest = Some([0; 32]);
        #[cfg(feature = "field-inline")]
        {
            config.lattice.field_inline.enabled = true;
        }

        let layout = derive_lattice_packed_witness_layout(
            &config,
            log_t,
            log_k_chunk,
            JoltRaPolynomialLayout::new(1, 1, 1).expect("RA layout should build"),
            &precommitted,
        )
        .expect("layout should derive");
        let requirements =
            derive_lattice_packed_validity_requirements(&config, log_k_chunk, &precommitted)
                .expect("validity requirements should derive");
        let statements = derive_lattice_packed_validity_statements(&layout, &requirements)
            .expect("validity statements should derive");
        (layout, statements)
    }

    fn small_bytecode_validity_context() -> (
        PackingWitnessLayout,
        Vec<LatticePackedValidityStatement>,
        Vec<LatticePackedValidityRequirement>,
    ) {
        let specs = vec![
            PackingFamilySpec::direct(
                PackingFamilyId::BytecodeRegisterSelector {
                    chunk: 0,
                    selector: 0,
                },
                PackingFactDomain::BytecodeRows { log_bytecode: 0 },
                1,
                PackingAlphabet::Fixed {
                    size: 1 << REGISTER_ADDRESS_BITS,
                },
            ),
            PackingFamilySpec::direct(
                PackingFamilyId::BytecodeRegisterSelector {
                    chunk: 0,
                    selector: 2,
                },
                PackingFactDomain::BytecodeRows { log_bytecode: 0 },
                1,
                PackingAlphabet::Fixed {
                    size: 1 << REGISTER_ADDRESS_BITS,
                },
            ),
            PackingFamilySpec::direct(
                PackingFamilyId::BytecodeCircuitFlag {
                    chunk: 0,
                    flag: CircuitFlags::Store as usize,
                },
                PackingFactDomain::BytecodeRows { log_bytecode: 0 },
                1,
                PackingAlphabet::Bit,
            ),
            PackingFamilySpec::direct(
                PackingFamilyId::BytecodeImmBytes { chunk: 0 },
                PackingFactDomain::BytecodeRows { log_bytecode: 0 },
                AkitaField::NUM_BYTES,
                PackingAlphabet::Byte,
            ),
        ];
        #[cfg(feature = "field-inline")]
        let specs = {
            let mut specs = specs;
            specs.extend((0..AkitaField::NUM_BYTES).map(|index| {
                PackingFamilySpec::direct(
                    PackingFamilyId::FieldRdIncByte { index },
                    PackingFactDomain::TraceRows { log_t: 0 },
                    1,
                    PackingAlphabet::Byte,
                )
            }));
            specs
        };
        let layout =
            PackingWitnessLayout::new(specs).expect("manual bytecode validity layout should build");
        let mut requirements = lattice_validity_requirements_for_packed_witness_layout(&layout);
        requirements.push(bytecode_imm_canonical_bytes_requirement(
            0,
            AkitaField::NUM_BYTES,
            jolt_akita::AKITA_FIELD_MODULUS,
        ));
        let statements = derive_lattice_packed_validity_statements(&layout, &requirements)
            .expect("manual bytecode validity statements should derive");
        (layout, statements, requirements)
    }

    fn validity_statement(
        statements: &[LatticePackedValidityStatement],
        family: LatticePackedFamilyId,
        kind: LatticePackedValidityStatementKind,
    ) -> &LatticePackedValidityStatement {
        statements
            .iter()
            .find(|statement| statement.requirement.family == family && statement.kind == kind)
            .expect("validity statement should exist")
    }

    fn validity_value_at_zero(
        source: &SparsePackingWitness<AkitaField>,
        statement: &LatticePackedValidityStatement,
    ) -> AkitaField {
        let point = vec![AkitaField::zero(); statement.num_vars];
        validity_value(source, statement, &point, &point).expect("validity value should evaluate")
    }

    fn packed_cell_at(
        family: PackingFamilyId,
        row: usize,
        limb: usize,
        symbol: usize,
    ) -> PackingCellAddress {
        PackingCellAddress {
            family,
            row,
            limb,
            symbol,
        }
    }

    fn af(value: u64) -> AkitaField {
        AkitaField::from_u64(value)
    }

    fn validity_default_source(
        layout: &PackingWitnessLayout,
        requirements: &[LatticePackedValidityRequirement],
    ) -> SparsePackingWitness<AkitaField> {
        validity_source_with_symbols(layout, requirements, |_, _| 0)
    }

    #[cfg(feature = "field-inline")]
    fn validity_source_with_field_rd_inc_bytes(
        layout: &PackingWitnessLayout,
        requirements: &[LatticePackedValidityRequirement],
        bytes: &[u8],
    ) -> SparsePackingWitness<AkitaField> {
        validity_source_with_symbols(layout, requirements, |family, _| match family {
            LatticePackedFamilyId::FieldRdIncByte { index } => bytes[*index] as usize,
            _ => 0,
        })
    }

    fn validity_source_with_bytecode_imm_bytes(
        layout: &PackingWitnessLayout,
        requirements: &[LatticePackedValidityRequirement],
        bytes: &[u8],
    ) -> SparsePackingWitness<AkitaField> {
        validity_source_with_symbols(layout, requirements, |family, limb| match family {
            LatticePackedFamilyId::BytecodeImmBytes { .. } => bytes[limb] as usize,
            _ => 0,
        })
    }

    fn validity_source_with_symbols(
        layout: &PackingWitnessLayout,
        requirements: &[LatticePackedValidityRequirement],
        mut symbol_for: impl FnMut(&LatticePackedFamilyId, usize) -> usize,
    ) -> SparsePackingWitness<AkitaField> {
        let mut cells = Vec::new();
        for requirement in requirements {
            let family_id = lattice_packing_family_id(&requirement.family);
            let family = layout
                .family(&family_id)
                .expect("validity family should exist");
            let rows = family.domain.rows().expect("family rows should derive");
            if !matches!(requirement.kind, LatticePackedValidityKind::ExactOneHot) {
                continue;
            }
            for row in 0..rows {
                for limb in 0..requirement.limbs {
                    let symbol = symbol_for(&requirement.family, limb);
                    cells.push((
                        PackingCellAddress {
                            family: family_id.clone(),
                            row,
                            limb,
                            symbol,
                        },
                        AkitaField::one(),
                    ));
                }
            }
        }
        SparsePackingWitness::try_from_cells(layout.clone(), cells)
            .expect("validity source should build")
    }

    fn verify_validity_artifacts<T>(
        setup: &AkitaPackingVerifierSetup,
        transcript: &mut T,
        artifacts: &AkitaPackingWitnessArtifacts,
        log_k_chunk: usize,
        precommitted: &PrecommittedSchedule,
        validity: &AkitaPackingValidityProofArtifacts,
    ) -> Result<(), VerifierError>
    where
        T: Transcript<Challenge = AkitaField>,
    {
        crate::stages::stage8::verify_lattice_packed_validity_proof::<
            AkitaField,
            AkitaPackingScheme,
            T,
            ClearOnlyCommitment,
        >(
            setup,
            transcript,
            &artifacts.protocol,
            log_k_chunk,
            precommitted,
            &artifacts.layout,
            artifacts
                .payload()
                .expect("artifact should carry lattice payload")
                .packed_witness
                .clone(),
            &validity.sumcheck_proof,
            &validity.opening_claims.opening_claims,
            &validity.opening_proof,
        )
    }
}
