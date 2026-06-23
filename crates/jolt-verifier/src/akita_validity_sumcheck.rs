use std::collections::BTreeMap;

use crate::{
    stages::stage8::{
        field_element_canonical_factors, field_element_canonical_value_from_openings,
        lattice_packing_family_id, FieldCanonicalFactor, LatticePackedValidityStatement,
        LatticePackedValidityStatementKind,
    },
    VerifierError,
};
use jolt_akita::AkitaField;
use jolt_claims::protocols::jolt::{LatticePackedFamilyId, LatticePackedValidityKind};
use jolt_openings::{PackingFamilyId, PackingWitnessLayout, PackingWitnessSource};
use jolt_poly::{try_eq_mle, EqPolynomial, UnivariatePoly};
use jolt_riscv::CircuitFlags;
use jolt_sumcheck::{
    BatchedEvaluationClaim, CompressedLabeledRoundPoly, CompressedSumcheckProof, EvaluationClaim,
    RoundMessage, SUMCHECK_ROUND_TRANSCRIPT_LABEL,
};
use jolt_transcript::Transcript;

pub(crate) fn prove_combined_validity_sumcheck<T, S>(
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
    S: PackingWitnessSource<AkitaField>,
{
    let indexed_source = IndexedValiditySource::new(source)?;
    let mut challenges = Vec::with_capacity(max_num_vars);
    let mut round_polynomials = Vec::with_capacity(max_num_vars);
    for _ in 0..max_num_vars {
        let remaining = max_num_vars - challenges.len() - 1;
        let round_evals = (0..=max_degree)
            .map(|point| {
                let mut prefix = challenges.clone();
                prefix.push(AkitaField::from_u64(point as u64));
                sum_combined_validity_over_suffix(
                    &indexed_source,
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
        &indexed_source,
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

fn sum_combined_validity_over_suffix(
    source: &IndexedValiditySource<'_>,
    statements: &[LatticePackedValidityStatement],
    eq_points: &[Vec<AkitaField>],
    batching_coefficients: &[AkitaField],
    max_num_vars: usize,
    prefix: &[AkitaField],
    remaining: usize,
) -> Result<AkitaField, VerifierError> {
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

fn combined_validity_value(
    source: &IndexedValiditySource<'_>,
    statements: &[LatticePackedValidityStatement],
    eq_points: &[Vec<AkitaField>],
    batching_coefficients: &[AkitaField],
    max_num_vars: usize,
    point: &[AkitaField],
) -> Result<AkitaField, VerifierError> {
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
        value +=
            *coefficient * validity_value_indexed(source, statement, eq_point, instance_point)?;
    }
    Ok(value)
}

#[cfg(test)]
pub(crate) fn validity_value_for_testing<S>(
    source: &S,
    statement: &LatticePackedValidityStatement,
    eq_point: &[AkitaField],
    point: &[AkitaField],
) -> Result<AkitaField, VerifierError>
where
    S: PackingWitnessSource<AkitaField>,
{
    let indexed_source = IndexedValiditySource::new(source)?;
    validity_value_indexed(&indexed_source, statement, eq_point, point)
}

struct IndexedValiditySource<'a> {
    layout: &'a PackingWitnessLayout,
    entries: BTreeMap<PackingFamilyId, Vec<PackingEntry>>,
}

#[derive(Clone, Copy)]
struct PackingEntry {
    row: usize,
    limb: usize,
    symbol: usize,
    value: AkitaField,
}

impl<'a> IndexedValiditySource<'a> {
    fn new<S>(source: &'a S) -> Result<Self, VerifierError>
    where
        S: PackingWitnessSource<AkitaField>,
    {
        let layout = source.layout();
        let mut entries: BTreeMap<PackingFamilyId, Vec<PackingEntry>> = BTreeMap::new();
        let mut error = None;
        source.for_each_nonzero(|rank, value| {
            if error.is_some() {
                return;
            }
            let Some(address) = layout.unrank(rank) else {
                error = Some(
                    VerifierError::LatticePackedValidityOpeningVerificationFailed {
                        reason: format!("packed validity source emitted out-of-layout rank {rank}"),
                    },
                );
                return;
            };
            entries
                .entry(address.family)
                .or_default()
                .push(PackingEntry {
                    row: address.row,
                    limb: address.limb,
                    symbol: address.symbol,
                    value,
                });
        });
        if let Some(error) = error {
            return Err(error);
        }
        Ok(Self { layout, entries })
    }

    fn entries(&self, family_id: &PackingFamilyId) -> &[PackingEntry] {
        self.entries.get(family_id).map_or(&[], Vec::as_slice)
    }
}

fn validity_value_indexed(
    source: &IndexedValiditySource<'_>,
    statement: &LatticePackedValidityStatement,
    eq_point: &[AkitaField],
    point: &[AkitaField],
) -> Result<AkitaField, VerifierError> {
    let eq_mask = try_eq_mle(point, eq_point).map_err(|error| {
        VerifierError::LatticePackedValiditySumcheckFailed {
            reason: error.to_string(),
        }
    })?;
    let value = match statement.kind {
        LatticePackedValidityStatementKind::BytecodeStoreRdDisjoint => {
            let openings = validity_opening_values_indexed(source, statement, point)?;
            openings[0] * openings[1]
        }
        LatticePackedValidityStatementKind::FieldElementCanonicalBytes => {
            let openings = validity_opening_values_indexed(source, statement, point)?;
            field_element_canonical_value_from_openings(statement, &openings)?
        }
        _ => validity_violation(
            statement.kind,
            validity_opening_value_indexed(source, statement, point)?,
        ),
    };
    Ok(eq_mask * value)
}

fn validity_opening_values_indexed(
    source: &IndexedValiditySource<'_>,
    statement: &LatticePackedValidityStatement,
    point: &[AkitaField],
) -> Result<Vec<AkitaField>, VerifierError> {
    if statement.kind == LatticePackedValidityStatementKind::BytecodeStoreRdDisjoint {
        return Ok(vec![
            bytecode_store_rd_disjoint_factor_value_indexed(source, statement, point, 0)?,
            bytecode_store_rd_disjoint_factor_value_indexed(source, statement, point, 1)?,
        ]);
    }
    if statement.kind == LatticePackedValidityStatementKind::FieldElementCanonicalBytes {
        let factors = field_element_canonical_factors(&statement.requirement)?;
        return factors
            .into_iter()
            .map(|factor| field_element_canonical_factor_value_indexed(source, point, factor))
            .collect();
    }
    validity_opening_value_indexed(source, statement, point).map(|value| vec![value])
}

fn validity_opening_value_indexed(
    source: &IndexedValiditySource<'_>,
    statement: &LatticePackedValidityStatement,
    point: &[AkitaField],
) -> Result<AkitaField, VerifierError> {
    let family_id = lattice_packing_family_id(&statement.requirement.family);
    let shape = validity_statement_shape(source.layout, statement, &family_id)?;
    let point_parts = split_validity_point(statement.kind, point, shape)?;
    let row_weights = EqPolynomial::<AkitaField>::evals(point_parts.row, None);
    let limb_weights = EqPolynomial::<AkitaField>::evals(point_parts.limb, None);
    match statement.kind {
        LatticePackedValidityStatementKind::CellBooleanity => {
            let symbol_weights = EqPolynomial::<AkitaField>::evals(point_parts.symbol, None);
            weighted_family_value_indexed(
                source,
                &family_id,
                &row_weights,
                &limb_weights,
                SymbolWeights::Point(&symbol_weights),
            )
        }
        LatticePackedValidityStatementKind::ExactOneHotRowSum
        | LatticePackedValidityStatementKind::OptionalOneHotRowSum => {
            weighted_family_value_indexed(
                source,
                &family_id,
                &row_weights,
                &limb_weights,
                SymbolWeights::All,
            )
        }
        LatticePackedValidityStatementKind::BooleanIndicator => {
            let LatticePackedValidityKind::BooleanIndicator { symbol } = statement.requirement.kind
            else {
                return Err(VerifierError::InvalidProtocolConfig {
                    reason: "boolean-indicator validity statement has non-indicator requirement"
                        .to_string(),
                });
            };
            weighted_family_value_indexed(
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

fn field_element_canonical_factor_value_indexed(
    source: &IndexedValiditySource<'_>,
    point: &[AkitaField],
    factor: FieldCanonicalFactor,
) -> Result<AkitaField, VerifierError> {
    let (family, limb, symbol_filter) = match factor {
        FieldCanonicalFactor::Eq {
            family,
            limb,
            symbol,
            ..
        } => {
            return weighted_field_canonical_symbol_value_indexed(
                source, point, &family, limb, symbol,
            );
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
        value +=
            weighted_field_canonical_symbol_value_indexed(source, point, &family, limb, symbol)?;
    }
    Ok(value)
}

fn weighted_field_canonical_symbol_value_indexed(
    source: &IndexedValiditySource<'_>,
    point: &[AkitaField],
    family_id: &PackingFamilyId,
    limb: usize,
    symbol: usize,
) -> Result<AkitaField, VerifierError> {
    let family =
        source
            .layout
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
    let row_vars = rows.ilog2() as usize;
    if point.len() != row_vars {
        return Err(VerifierError::LatticePackedValiditySumcheckFailed {
            reason: format!(
                "field-element canonical-byte point has {} variables but statement requires {row_vars}",
                point.len()
            ),
        });
    }
    let row_weights = EqPolynomial::<AkitaField>::evals(point, None);
    weighted_direct_limb_symbol_value_indexed(source, family_id, &row_weights, limb, symbol)
}

fn bytecode_store_rd_disjoint_factor_value_indexed(
    source: &IndexedValiditySource<'_>,
    statement: &LatticePackedValidityStatement,
    point: &[AkitaField],
    factor: usize,
) -> Result<AkitaField, VerifierError> {
    let chunk = bytecode_store_rd_disjoint_chunk(&statement.requirement)?;
    let store_id = PackingFamilyId::BytecodeCircuitFlag {
        chunk,
        flag: CircuitFlags::Store as usize,
    };
    let store =
        source
            .layout
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
    let row_vars = rows.ilog2() as usize;
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
        0 => weighted_direct_symbol_value_indexed(source, &store_id, &row_weights, 1),
        1 => {
            let rd_id = PackingFamilyId::BytecodeRegisterSelector { chunk, selector: 2 };
            let rd = source.layout.family(&rd_id).ok_or_else(|| {
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
            weighted_family_value_indexed(
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
    requirement: &jolt_claims::protocols::jolt::LatticePackedValidityRequirement,
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
        row: rows.ilog2() as usize,
        limb: statement.requirement.limbs.ilog2() as usize,
        symbol: statement.requirement.alphabet_size.ilog2() as usize,
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

fn weighted_family_value_indexed(
    source: &IndexedValiditySource<'_>,
    family_id: &PackingFamilyId,
    row_weights: &[AkitaField],
    limb_weights: &[AkitaField],
    symbol_weights: SymbolWeights<'_>,
) -> Result<AkitaField, VerifierError> {
    let mut value = AkitaField::zero();
    for entry in source.entries(family_id) {
        let Some(row_weight) = row_weights.get(entry.row).copied() else {
            return Err(
                VerifierError::LatticePackedValidityOpeningVerificationFailed {
                    reason: format!("packed validity row {} is outside row weights", entry.row),
                },
            );
        };
        let Some(limb_weight) = limb_weights.get(entry.limb).copied() else {
            return Err(
                VerifierError::LatticePackedValidityOpeningVerificationFailed {
                    reason: format!(
                        "packed validity limb {} is outside limb weights",
                        entry.limb
                    ),
                },
            );
        };
        let symbol_weight = match symbol_weights {
            SymbolWeights::Point(weights) => {
                let Some(weight) = weights.get(entry.symbol).copied() else {
                    return Err(
                        VerifierError::LatticePackedValidityOpeningVerificationFailed {
                            reason: format!(
                                "packed validity symbol {} is outside symbol weights",
                                entry.symbol
                            ),
                        },
                    );
                };
                weight
            }
            SymbolWeights::All => AkitaField::one(),
            SymbolWeights::Fixed(symbol) => {
                if entry.symbol == symbol {
                    AkitaField::one()
                } else {
                    continue;
                }
            }
        };
        value += row_weight * limb_weight * symbol_weight * entry.value;
    }
    Ok(value)
}

fn weighted_direct_symbol_value_indexed(
    source: &IndexedValiditySource<'_>,
    family_id: &PackingFamilyId,
    row_weights: &[AkitaField],
    symbol: usize,
) -> Result<AkitaField, VerifierError> {
    weighted_direct_limb_symbol_value_indexed(source, family_id, row_weights, 0, symbol)
}

fn weighted_direct_limb_symbol_value_indexed(
    source: &IndexedValiditySource<'_>,
    family_id: &PackingFamilyId,
    row_weights: &[AkitaField],
    limb: usize,
    symbol: usize,
) -> Result<AkitaField, VerifierError> {
    let mut value = AkitaField::zero();
    for entry in source.entries(family_id) {
        if entry.limb != limb || entry.symbol != symbol {
            continue;
        }
        let Some(row_weight) = row_weights.get(entry.row).copied() else {
            return Err(
                VerifierError::LatticePackedValidityOpeningVerificationFailed {
                    reason: format!("packed validity row {} is outside row weights", entry.row),
                },
            );
        };
        value += row_weight * entry.value;
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
