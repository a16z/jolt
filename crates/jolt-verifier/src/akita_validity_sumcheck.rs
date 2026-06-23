use crate::{
    akita_validity::validity_value, stages::stage8::LatticePackedValidityStatement, VerifierError,
};
use jolt_akita::AkitaField;
use jolt_openings::PackingWitnessSource;
use jolt_poly::UnivariatePoly;
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
    S: PackingWitnessSource<AkitaField>,
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
    S: PackingWitnessSource<AkitaField>,
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
