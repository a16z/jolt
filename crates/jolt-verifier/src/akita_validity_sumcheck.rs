use std::collections::BTreeMap;

#[cfg(test)]
use crate::stages::stage8::field_element_canonical_value_from_openings;
use crate::{
    stages::stage8::{
        field_element_canonical_factors, lattice_packing_family_id, FieldCanonicalFactor,
        LatticePackedValidityStatement, LatticePackedValidityStatementKind,
    },
    VerifierError,
};
use jolt_akita::AkitaField;
use jolt_claims::protocols::jolt::{LatticePackedFamilyId, LatticePackedValidityKind};
use jolt_openings::{PackingFamilyId, PackingWitnessLayout, PackingWitnessSource};
#[cfg(test)]
use jolt_poly::try_eq_mle;
use jolt_poly::{EqPolynomial, UnivariatePoly};
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
    let two_inv = AkitaField::from_u64(2).inverse().ok_or_else(|| {
        VerifierError::LatticePackedValiditySumcheckFailed {
            reason: "failed to invert 2 in Akita field".to_string(),
        }
    })?;
    let mut instances = statements
        .iter()
        .zip(eq_points)
        .zip(batching_coefficients)
        .map(|((statement, eq_point), coefficient)| {
            DenseValidityInstance::new(
                &indexed_source,
                statement,
                eq_point,
                *coefficient,
                max_num_vars,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut challenges = Vec::with_capacity(max_num_vars);
    let mut round_polynomials = Vec::with_capacity(max_num_vars);
    for round in 0..max_num_vars {
        let mut round_evals = vec![AkitaField::zero(); max_degree + 1];
        let mut instance_round_evals = Vec::with_capacity(instances.len());
        for instance in &instances {
            let evals = if instance.is_dummy_round(round) {
                let dummy_eval = instance.current_claim * two_inv;
                vec![dummy_eval; max_degree + 1]
            } else {
                instance.compute_round_evals(max_degree)?
            };
            for (combined, eval) in round_evals.iter_mut().zip(&evals) {
                *combined += instance.coefficient * *eval;
            }
            instance_round_evals.push(evals);
        }
        let round_poly = UnivariatePoly::from_evals(&round_evals);
        let compressed =
            CompressedLabeledRoundPoly::new(&round_poly, SUMCHECK_ROUND_TRANSCRIPT_LABEL);
        <CompressedLabeledRoundPoly<'_, AkitaField> as RoundMessage>::append_to_transcript(
            &compressed,
            transcript,
        );
        let challenge = transcript.challenge();
        round_polynomials.push(round_poly.compress());
        for (instance, evals) in instances.iter_mut().zip(instance_round_evals) {
            if instance.is_dummy_round(round) {
                instance.current_claim *= two_inv;
            } else {
                let next_claim = UnivariatePoly::from_evals(&evals).evaluate(challenge);
                instance.bind(challenge, next_claim)?;
            }
        }
        challenges.push(challenge);
    }

    let value = instances
        .iter()
        .map(|instance| instance.coefficient * instance.current_claim)
        .sum();

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

struct DenseValidityInstance {
    offset: usize,
    coefficient: AkitaField,
    factors: Vec<Vec<AkitaField>>,
    terms: Vec<DenseProductTerm>,
    current_claim: AkitaField,
}

struct DenseProductTerm {
    coefficient: AkitaField,
    factors: Vec<usize>,
}

impl DenseValidityInstance {
    fn new(
        source: &IndexedValiditySource<'_>,
        statement: &LatticePackedValidityStatement,
        eq_point: &[AkitaField],
        coefficient: AkitaField,
        max_num_vars: usize,
    ) -> Result<Self, VerifierError> {
        let offset = max_num_vars
            .checked_sub(statement.num_vars)
            .ok_or_else(|| VerifierError::LatticePackedValiditySumcheckFailed {
                reason: "packed validity statement has more variables than the combined batch"
                    .to_string(),
            })?;
        let expected_len = checked_power_of_two(statement.num_vars, "packed validity statement")?;
        let mut factors = vec![EqPolynomial::<AkitaField>::evals(eq_point, None)];
        if factors[0].len() != expected_len {
            return Err(VerifierError::LatticePackedValiditySumcheckFailed {
                reason: format!(
                    "packed validity eq point has {} evaluations but statement requires {expected_len}",
                    factors[0].len()
                ),
            });
        }
        let eq = 0;
        let mut terms = Vec::new();
        match statement.kind {
            LatticePackedValidityStatementKind::CellBooleanity
            | LatticePackedValidityStatementKind::BooleanIndicator
            | LatticePackedValidityStatementKind::OptionalOneHotRowSum => {
                let opening = push_factor(
                    &mut factors,
                    dense_validity_opening_value(source, statement)?,
                    expected_len,
                )?;
                terms.push(DenseProductTerm {
                    coefficient: AkitaField::one(),
                    factors: vec![eq, opening, opening],
                });
                terms.push(DenseProductTerm {
                    coefficient: -AkitaField::one(),
                    factors: vec![eq, opening],
                });
            }
            LatticePackedValidityStatementKind::ExactOneHotRowSum => {
                let opening = push_factor(
                    &mut factors,
                    dense_validity_opening_value(source, statement)?,
                    expected_len,
                )?;
                terms.push(DenseProductTerm {
                    coefficient: AkitaField::one(),
                    factors: vec![eq, opening, opening],
                });
                terms.push(DenseProductTerm {
                    coefficient: -AkitaField::from_u64(2),
                    factors: vec![eq, opening],
                });
                terms.push(DenseProductTerm {
                    coefficient: AkitaField::one(),
                    factors: vec![eq],
                });
            }
            LatticePackedValidityStatementKind::BytecodeStoreRdDisjoint => {
                let openings =
                    dense_bytecode_store_rd_disjoint_openings(source, statement, expected_len)?;
                let left = push_factor(&mut factors, openings.0, expected_len)?;
                let right = push_factor(&mut factors, openings.1, expected_len)?;
                terms.push(DenseProductTerm {
                    coefficient: AkitaField::one(),
                    factors: vec![eq, left, right],
                });
            }
            LatticePackedValidityStatementKind::FieldElementCanonicalBytes => {
                append_field_canonical_terms(
                    source,
                    statement,
                    expected_len,
                    eq,
                    &mut factors,
                    &mut terms,
                )?;
            }
        }

        let mut current_claim = sum_dense_terms(&factors, &terms)?;
        current_claim *= pow2_field(offset);
        Ok(Self {
            offset,
            coefficient,
            factors,
            terms,
            current_claim,
        })
    }

    fn is_dummy_round(&self, round: usize) -> bool {
        round < self.offset
    }

    fn compute_round_evals(&self, max_degree: usize) -> Result<Vec<AkitaField>, VerifierError> {
        let current_len = self.factors.first().map(Vec::len).ok_or_else(|| {
            VerifierError::LatticePackedValiditySumcheckFailed {
                reason: "packed validity instance has no factors".to_string(),
            }
        })?;
        if current_len < 2 || current_len % 2 != 0 {
            return Err(VerifierError::LatticePackedValiditySumcheckFailed {
                reason: format!(
                    "packed validity active round requires an even factor length, got {current_len}"
                ),
            });
        }
        let mut evals = vec![AkitaField::zero(); max_degree + 1];
        let half = current_len / 2;
        for row in 0..half {
            let lo = row;
            let hi = row + half;
            for (point, eval) in evals.iter_mut().enumerate() {
                let point = AkitaField::from_u64(point as u64);
                for term in &self.terms {
                    let mut product = term.coefficient;
                    for &factor in &term.factors {
                        let low = self.factors[factor][lo];
                        let high = self.factors[factor][hi];
                        product *= low + point * (high - low);
                    }
                    *eval += product;
                }
            }
        }
        Ok(evals)
    }

    fn bind(&mut self, challenge: AkitaField, next_claim: AkitaField) -> Result<(), VerifierError> {
        for factor in &mut self.factors {
            if factor.len() < 2 || factor.len() % 2 != 0 {
                return Err(VerifierError::LatticePackedValiditySumcheckFailed {
                    reason: format!(
                        "packed validity bind requires an even factor length, got {}",
                        factor.len()
                    ),
                });
            }
            let half = factor.len() / 2;
            let next = (0..half)
                .map(|row| {
                    let low = factor[row];
                    let high = factor[row + half];
                    low + challenge * (high - low)
                })
                .collect();
            *factor = next;
        }
        self.current_claim = next_claim;
        Ok(())
    }
}

fn push_factor(
    factors: &mut Vec<Vec<AkitaField>>,
    factor: Vec<AkitaField>,
    expected_len: usize,
) -> Result<usize, VerifierError> {
    if factor.len() != expected_len {
        return Err(VerifierError::LatticePackedValiditySumcheckFailed {
            reason: format!(
                "packed validity factor has {} evaluations but statement requires {expected_len}",
                factor.len()
            ),
        });
    }
    let index = factors.len();
    factors.push(factor);
    Ok(index)
}

fn sum_dense_terms(
    factors: &[Vec<AkitaField>],
    terms: &[DenseProductTerm],
) -> Result<AkitaField, VerifierError> {
    let first_factor =
        factors
            .first()
            .ok_or_else(|| VerifierError::LatticePackedValiditySumcheckFailed {
                reason: "packed validity instance has no factors".to_string(),
            })?;
    let mut sum = AkitaField::zero();
    for (row, _) in first_factor.iter().enumerate() {
        for term in terms {
            let mut product = term.coefficient;
            for &factor in &term.factors {
                product *= factors[factor][row];
            }
            sum += product;
        }
    }
    Ok(sum)
}

fn pow2_field(bits: usize) -> AkitaField {
    let mut value = AkitaField::one();
    for _ in 0..bits {
        value += value;
    }
    value
}

enum DenseSymbolFilter {
    Point,
    All,
    Fixed(usize),
}

fn dense_validity_opening_value(
    source: &IndexedValiditySource<'_>,
    statement: &LatticePackedValidityStatement,
) -> Result<Vec<AkitaField>, VerifierError> {
    let family_id = lattice_packing_family_id(&statement.requirement.family);
    let shape = validity_statement_shape(source.layout, statement, &family_id)?;
    match statement.kind {
        LatticePackedValidityStatementKind::CellBooleanity => dense_family_factor(
            source,
            &family_id,
            shape,
            shape.symbol,
            DenseSymbolFilter::Point,
        ),
        LatticePackedValidityStatementKind::ExactOneHotRowSum
        | LatticePackedValidityStatementKind::OptionalOneHotRowSum => {
            dense_family_factor(source, &family_id, shape, 0, DenseSymbolFilter::All)
        }
        LatticePackedValidityStatementKind::BooleanIndicator => {
            let LatticePackedValidityKind::BooleanIndicator { symbol } = statement.requirement.kind
            else {
                return Err(VerifierError::InvalidProtocolConfig {
                    reason: "boolean-indicator validity statement has non-indicator requirement"
                        .to_string(),
                });
            };
            dense_family_factor(
                source,
                &family_id,
                shape,
                0,
                DenseSymbolFilter::Fixed(symbol),
            )
        }
        LatticePackedValidityStatementKind::BytecodeStoreRdDisjoint
        | LatticePackedValidityStatementKind::FieldElementCanonicalBytes => {
            Err(VerifierError::InvalidProtocolConfig {
                reason:
                    "multi-factor packed validity statement requires specialized dense openings"
                        .to_string(),
            })
        }
    }
}

fn dense_family_factor(
    source: &IndexedValiditySource<'_>,
    family_id: &PackingFamilyId,
    shape: ValidityStatementShape,
    output_symbol_bits: usize,
    symbol_filter: DenseSymbolFilter,
) -> Result<Vec<AkitaField>, VerifierError> {
    let len = checked_power_of_two(
        shape.row + shape.limb + output_symbol_bits,
        "packed validity dense factor",
    )?;
    let row_count = checked_power_of_two(shape.row, "packed validity rows")?;
    let limb_count = checked_power_of_two(shape.limb, "packed validity limbs")?;
    let symbol_count = checked_power_of_two(shape.symbol, "packed validity symbols")?;
    let output_symbol_count =
        checked_power_of_two(output_symbol_bits, "packed validity output symbols")?;
    let mut evals = vec![AkitaField::zero(); len];
    for entry in source.entries(family_id) {
        if entry.row >= row_count || entry.limb >= limb_count {
            return Err(
                VerifierError::LatticePackedValidityOpeningVerificationFailed {
                    reason: format!(
                        "packed validity entry is outside dense factor shape: row {}, limb {}",
                        entry.row, entry.limb
                    ),
                },
            );
        }
        let symbol = match symbol_filter {
            DenseSymbolFilter::Point => {
                if entry.symbol >= symbol_count {
                    return Err(
                        VerifierError::LatticePackedValidityOpeningVerificationFailed {
                            reason: format!(
                                "packed validity symbol {} is outside dense factor shape",
                                entry.symbol
                            ),
                        },
                    );
                }
                entry.symbol
            }
            DenseSymbolFilter::All => 0,
            DenseSymbolFilter::Fixed(symbol) => {
                if entry.symbol != symbol {
                    continue;
                }
                0
            }
        };
        if symbol >= output_symbol_count {
            return Err(
                VerifierError::LatticePackedValidityOpeningVerificationFailed {
                    reason: format!(
                        "packed validity output symbol {symbol} is outside dense factor shape"
                    ),
                },
            );
        }
        let index = (entry.row << (shape.limb + output_symbol_bits))
            | (entry.limb << output_symbol_bits)
            | symbol;
        evals[index] += entry.value;
    }
    Ok(evals)
}

fn dense_bytecode_store_rd_disjoint_openings(
    source: &IndexedValiditySource<'_>,
    statement: &LatticePackedValidityStatement,
    expected_len: usize,
) -> Result<(Vec<AkitaField>, Vec<AkitaField>), VerifierError> {
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
    let store_factor = dense_direct_limb_symbol_value(source, &store_id, row_vars, 0, 1)?;
    let rd_id = PackingFamilyId::BytecodeRegisterSelector { chunk, selector: 2 };
    let rd = source
        .layout
        .family(&rd_id)
        .ok_or_else(|| VerifierError::InvalidProtocolConfig {
            reason: format!("bytecode Store/Rd disjointness requires {rd_id:?}"),
        })?;
    if rd.domain != store.domain || rd.limbs != 1 {
        return Err(VerifierError::InvalidProtocolConfig {
            reason: "bytecode Store/Rd disjointness rd selector layout mismatch".to_string(),
        });
    }
    let rd_factor = dense_family_factor(
        source,
        &rd_id,
        ValidityStatementShape {
            row: row_vars,
            limb: 0,
            symbol: rd.alphabet.size().ilog2() as usize,
        },
        0,
        DenseSymbolFilter::All,
    )?;
    if store_factor.len() != expected_len || rd_factor.len() != expected_len {
        return Err(VerifierError::LatticePackedValiditySumcheckFailed {
            reason: "bytecode Store/Rd dense factors have unexpected length".to_string(),
        });
    }
    Ok((store_factor, rd_factor))
}

fn append_field_canonical_terms(
    source: &IndexedValiditySource<'_>,
    statement: &LatticePackedValidityStatement,
    expected_len: usize,
    eq: usize,
    factors: &mut Vec<Vec<AkitaField>>,
    terms: &mut Vec<DenseProductTerm>,
) -> Result<(), VerifierError> {
    let canonical_factors = field_element_canonical_factors(&statement.requirement)?;
    let byte_width = canonical_factors
        .iter()
        .map(|factor| match factor {
            FieldCanonicalFactor::Eq { byte_index, .. }
            | FieldCanonicalFactor::Range { byte_index, .. } => *byte_index,
        })
        .max()
        .map_or(0, |index| index + 1);
    let mut equality = vec![None; byte_width];
    let mut range = vec![None; byte_width];
    for factor in canonical_factors {
        let dense = match &factor {
            FieldCanonicalFactor::Eq {
                family,
                limb,
                symbol,
                ..
            } => {
                dense_direct_limb_symbol_value(source, family, statement.num_vars, *limb, *symbol)?
            }
            FieldCanonicalFactor::Range {
                family,
                limb,
                start_symbol,
                ..
            } => dense_direct_limb_symbol_range(
                source,
                family,
                statement.num_vars,
                *limb,
                *start_symbol..256,
            )?,
        };
        let factor_index = push_factor(factors, dense, expected_len)?;
        match factor {
            FieldCanonicalFactor::Eq { byte_index, .. } => {
                equality[byte_index] = Some(factor_index);
            }
            FieldCanonicalFactor::Range { byte_index, .. } => {
                range[byte_index] = Some(factor_index);
            }
        }
    }

    for (byte_index, range_factor) in range.iter().copied().enumerate() {
        let Some(range_factor) = range_factor else {
            continue;
        };
        let mut term_factors = vec![eq, range_factor];
        for (higher_byte, equality_factor) in
            equality.iter().copied().enumerate().skip(byte_index + 1)
        {
            let Some(equality_factor) = equality_factor else {
                return Err(VerifierError::InvalidProtocolConfig {
                    reason: format!(
                        "field-element canonical-byte statement is missing equality factor for byte {higher_byte}"
                    ),
                });
            };
            term_factors.push(equality_factor);
        }
        terms.push(DenseProductTerm {
            coefficient: AkitaField::one(),
            factors: term_factors,
        });
    }
    Ok(())
}

fn dense_direct_limb_symbol_value(
    source: &IndexedValiditySource<'_>,
    family_id: &PackingFamilyId,
    row_vars: usize,
    limb: usize,
    symbol: usize,
) -> Result<Vec<AkitaField>, VerifierError> {
    dense_direct_limb_symbol_range(source, family_id, row_vars, limb, symbol..symbol + 1)
}

fn dense_direct_limb_symbol_range(
    source: &IndexedValiditySource<'_>,
    family_id: &PackingFamilyId,
    row_vars: usize,
    limb: usize,
    symbols: std::ops::Range<usize>,
) -> Result<Vec<AkitaField>, VerifierError> {
    let len = checked_power_of_two(row_vars, "packed validity direct rows")?;
    let mut evals = vec![AkitaField::zero(); len];
    for entry in source.entries(family_id) {
        if entry.limb != limb || !symbols.contains(&entry.symbol) {
            continue;
        }
        let Some(eval) = evals.get_mut(entry.row) else {
            return Err(
                VerifierError::LatticePackedValidityOpeningVerificationFailed {
                    reason: format!(
                        "packed validity row {} is outside dense direct factor",
                        entry.row
                    ),
                },
            );
        };
        *eval += entry.value;
    }
    Ok(evals)
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

#[cfg(test)]
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

#[cfg(test)]
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

#[cfg(test)]
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

#[cfg(test)]
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

#[cfg(test)]
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

#[cfg(test)]
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

#[cfg(test)]
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

#[cfg(test)]
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

#[cfg(test)]
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

#[cfg(test)]
enum SymbolWeights<'a> {
    Point(&'a [AkitaField]),
    All,
    Fixed(usize),
}

#[cfg(test)]
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

#[cfg(test)]
fn weighted_direct_symbol_value_indexed(
    source: &IndexedValiditySource<'_>,
    family_id: &PackingFamilyId,
    row_weights: &[AkitaField],
    symbol: usize,
) -> Result<AkitaField, VerifierError> {
    weighted_direct_limb_symbol_value_indexed(source, family_id, row_weights, 0, symbol)
}

#[cfg(test)]
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

fn checked_power_of_two(bits: usize, name: &'static str) -> Result<usize, VerifierError> {
    1usize.checked_shl(bits as u32).ok_or_else(|| {
        VerifierError::LatticePackedValiditySumcheckFailed {
            reason: format!("{name} dimension is too large"),
        }
    })
}
