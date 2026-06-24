use jolt_field::{Field, RingCore};
use jolt_poly::{boolean_point_msb, EqPolynomial, Polynomial};

use crate::{challenge, public};

use super::super::super::{
    IncClaimReductionChallenge, IncClaimReductionPublic, JoltChallengeId, JoltCommittedPolynomial,
    JoltExpr, JoltOpeningId, JoltPublicId, JoltRelationClaims, JoltRelationId,
};
use super::super::{dimensions::TraceDimensions, error::JoltFormulaPointError};

pub fn claim_reduction<F>(dimensions: TraceDimensions) -> JoltRelationClaims<F>
where
    F: RingCore,
{
    use crate::protocols::jolt::relations::claim_reductions::increments::ClaimReduction;
    use crate::SymbolicSumcheck;
    let r = ClaimReduction::new(dimensions);
    JoltRelationClaims::new(
        ClaimReduction::id(),
        r.spec(),
        r.input_expression::<F>(),
        r.output_expression::<F>(),
    )
}

pub(crate) fn inc_challenge<F>(id: IncClaimReductionChallenge) -> JoltExpr<F>
where
    F: RingCore,
{
    challenge(JoltChallengeId::from(id))
}

pub(crate) fn inc_public<F>(id: IncClaimReductionPublic) -> JoltExpr<F>
where
    F: RingCore,
{
    public(JoltPublicId::from(id))
}

pub fn claim_reduction_input_openings() -> [JoltOpeningId; 4] {
    [
        ram_inc_read_write(),
        ram_inc_val_check(),
        rd_inc_read_write(),
        rd_inc_val_evaluation(),
    ]
}

pub fn claim_reduction_output_openings() -> [JoltOpeningId; 2] {
    [ram_inc_reduced(), rd_inc_reduced()]
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClaimReductionOutputCoefficients<F> {
    pub ram: F,
    pub rd: F,
}

#[derive(Clone, Copy, Debug)]
pub struct ClaimReductionOutputCoefficientInputs<'a, F> {
    pub opening_point: &'a [F],
    pub ram_read_write_cycle: &'a [F],
    pub ram_val_check_cycle: &'a [F],
    pub registers_read_write_cycle: &'a [F],
    pub registers_val_evaluation_cycle: &'a [F],
    pub gamma: F,
}

#[derive(Clone, Copy, Debug)]
pub struct ClaimReductionOutputClaimInputs<'a, F> {
    pub coefficients: ClaimReductionOutputCoefficientInputs<'a, F>,
    pub ram_inc: F,
    pub rd_inc: F,
}

#[derive(Clone, Copy, Debug)]
pub struct ClaimReductionOutputCoefficientPolynomialInputs<'a, F> {
    pub num_vars: usize,
    pub trace_dimensions: TraceDimensions,
    pub ram_read_write_cycle: &'a [F],
    pub ram_val_check_cycle: &'a [F],
    pub registers_read_write_cycle: &'a [F],
    pub registers_val_evaluation_cycle: &'a [F],
    pub gamma: F,
}

pub fn claim_reduction_output_coefficients<F: Field>(
    inputs: ClaimReductionOutputCoefficientInputs<'_, F>,
) -> Result<ClaimReductionOutputCoefficients<F>, JoltFormulaPointError> {
    let eq_ram_read_write = eq_mle(inputs.opening_point, inputs.ram_read_write_cycle)?;
    let eq_ram_val_check = eq_mle(inputs.opening_point, inputs.ram_val_check_cycle)?;
    let eq_registers_read_write = eq_mle(inputs.opening_point, inputs.registers_read_write_cycle)?;
    let eq_registers_val_evaluation =
        eq_mle(inputs.opening_point, inputs.registers_val_evaluation_cycle)?;

    Ok(ClaimReductionOutputCoefficients {
        ram: eq_ram_read_write + inputs.gamma * eq_ram_val_check,
        rd: eq_registers_read_write + inputs.gamma * eq_registers_val_evaluation,
    })
}

pub fn claim_reduction_output_claim<F: Field>(
    inputs: ClaimReductionOutputClaimInputs<'_, F>,
) -> Result<F, JoltFormulaPointError> {
    let coefficients = claim_reduction_output_coefficients(inputs.coefficients)?;
    Ok(inputs.ram_inc * coefficients.ram
        + inputs.coefficients.gamma * inputs.coefficients.gamma * inputs.rd_inc * coefficients.rd)
}

pub fn claim_reduction_output_coefficient_polynomials<F: Field>(
    inputs: ClaimReductionOutputCoefficientPolynomialInputs<'_, F>,
) -> Result<(Polynomial<F>, Polynomial<F>), JoltFormulaPointError> {
    let rows = 1usize.checked_shl(inputs.num_vars as u32).ok_or(
        JoltFormulaPointError::EvaluationDomainSizeOverflow {
            num_vars: inputs.num_vars,
        },
    )?;
    let mut ram_coeff = Vec::with_capacity(rows);
    let mut rd_coeff = Vec::with_capacity(rows);
    for index in 0..rows {
        let point = boolean_point_msb(inputs.num_vars, index);
        let opening_point = inputs.trace_dimensions.cycle_opening_point(&point)?;
        let coefficients =
            claim_reduction_output_coefficients(ClaimReductionOutputCoefficientInputs {
                opening_point: &opening_point,
                ram_read_write_cycle: inputs.ram_read_write_cycle,
                ram_val_check_cycle: inputs.ram_val_check_cycle,
                registers_read_write_cycle: inputs.registers_read_write_cycle,
                registers_val_evaluation_cycle: inputs.registers_val_evaluation_cycle,
                gamma: inputs.gamma,
            })?;
        ram_coeff.push(coefficients.ram);
        rd_coeff.push(coefficients.rd);
    }
    Ok((Polynomial::new(ram_coeff), Polynomial::new(rd_coeff)))
}

pub fn ram_inc_read_write_opening() -> JoltOpeningId {
    ram_inc_read_write()
}

pub fn ram_inc_val_check_opening() -> JoltOpeningId {
    ram_inc_val_check()
}

pub fn rd_inc_read_write_opening() -> JoltOpeningId {
    rd_inc_read_write()
}

pub fn rd_inc_val_evaluation_opening() -> JoltOpeningId {
    rd_inc_val_evaluation()
}

pub fn ram_inc_reduced_opening() -> JoltOpeningId {
    ram_inc_reduced()
}

pub fn rd_inc_reduced_opening() -> JoltOpeningId {
    rd_inc_reduced()
}

pub(crate) fn ram_inc_read_write() -> JoltOpeningId {
    JoltOpeningId::committed(
        JoltCommittedPolynomial::RamInc,
        JoltRelationId::RamReadWriteChecking,
    )
}

pub(crate) fn ram_inc_val_check() -> JoltOpeningId {
    JoltOpeningId::committed(JoltCommittedPolynomial::RamInc, JoltRelationId::RamValCheck)
}

pub(crate) fn rd_inc_read_write() -> JoltOpeningId {
    JoltOpeningId::committed(
        JoltCommittedPolynomial::RdInc,
        JoltRelationId::RegistersReadWriteChecking,
    )
}

pub(crate) fn rd_inc_val_evaluation() -> JoltOpeningId {
    JoltOpeningId::committed(
        JoltCommittedPolynomial::RdInc,
        JoltRelationId::RegistersValEvaluation,
    )
}

pub(crate) fn ram_inc_reduced() -> JoltOpeningId {
    JoltOpeningId::committed(
        JoltCommittedPolynomial::RamInc,
        JoltRelationId::IncClaimReduction,
    )
}

pub(crate) fn rd_inc_reduced() -> JoltOpeningId {
    JoltOpeningId::committed(
        JoltCommittedPolynomial::RdInc,
        JoltRelationId::IncClaimReduction,
    )
}

fn eq_mle<F: Field>(
    opening_point: &[F],
    reference_point: &[F],
) -> Result<F, JoltFormulaPointError> {
    if opening_point.len() != reference_point.len() {
        return Err(JoltFormulaPointError::OpeningPointLengthMismatch {
            expected: opening_point.len(),
            got: reference_point.len(),
        });
    }
    Ok(EqPolynomial::mle(opening_point, reference_point))
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{Fr, FromPrimitiveInt};

    fn dimensions() -> TraceDimensions {
        TraceDimensions::new(5)
    }

    #[test]
    fn claim_reduction_evaluates_like_core_formula() {
        let claims = claim_reduction::<Fr>(dimensions());

        let ram_rw = Fr::from_u64(3);
        let ram_val = Fr::from_u64(5);
        let rd_rw = Fr::from_u64(7);
        let rd_val = Fr::from_u64(11);
        let ram_reduced = Fr::from_u64(13);
        let rd_reduced = Fr::from_u64(17);
        let eq_ram_rw = Fr::from_u64(19);
        let eq_ram_val = Fr::from_u64(23);
        let eq_rd_rw = Fr::from_u64(29);
        let eq_rd_val = Fr::from_u64(31);
        let gamma = Fr::from_u64(37);
        let zero = Fr::from_u64(0);

        let input = claims.input.expression().evaluate(
            |id| match *id {
                id if id == ram_inc_read_write() => ram_rw,
                id if id == ram_inc_val_check() => ram_val,
                id if id == rd_inc_read_write() => rd_rw,
                id if id == rd_inc_val_evaluation() => rd_val,
                _ => zero,
            },
            |id| match *id {
                JoltChallengeId::IncClaimReduction(IncClaimReductionChallenge::Gamma) => gamma,
                _ => zero,
            },
            |_| zero,
        );

        let output = claims.output.expression().evaluate(
            |id| match *id {
                id if id == ram_inc_reduced() => ram_reduced,
                id if id == rd_inc_reduced() => rd_reduced,
                _ => zero,
            },
            |id| match *id {
                JoltChallengeId::IncClaimReduction(IncClaimReductionChallenge::Gamma) => gamma,
                _ => zero,
            },
            |id| match *id {
                JoltPublicId::IncClaimReduction(IncClaimReductionPublic::EqRamReadWrite) => {
                    eq_ram_rw
                }
                JoltPublicId::IncClaimReduction(IncClaimReductionPublic::EqRamValCheck) => {
                    eq_ram_val
                }
                JoltPublicId::IncClaimReduction(IncClaimReductionPublic::EqRegistersReadWrite) => {
                    eq_rd_rw
                }
                JoltPublicId::IncClaimReduction(
                    IncClaimReductionPublic::EqRegistersValEvaluation,
                ) => eq_rd_val,
                _ => zero,
            },
        );

        let gamma_2 = gamma * gamma;
        assert_eq!(
            input,
            ram_rw + gamma * ram_val + gamma_2 * rd_rw + gamma_2 * gamma * rd_val
        );
        assert_eq!(
            output,
            ram_reduced * (eq_ram_rw + gamma * eq_ram_val)
                + gamma_2 * rd_reduced * (eq_rd_rw + gamma * eq_rd_val)
        );
    }

    #[test]
    fn output_coefficients_evaluate_eq_weighted_cycles() {
        let opening_point = vec![Fr::from_u64(0), Fr::from_u64(1)];
        let ram_read_write_cycle = vec![Fr::from_u64(0), Fr::from_u64(1)];
        let ram_val_check_cycle = vec![Fr::from_u64(1), Fr::from_u64(0)];
        let registers_read_write_cycle = vec![Fr::from_u64(0), Fr::from_u64(0)];
        let registers_val_evaluation_cycle = vec![Fr::from_u64(0), Fr::from_u64(1)];
        let gamma = Fr::from_u64(7);

        let coefficients =
            claim_reduction_output_coefficients(ClaimReductionOutputCoefficientInputs {
                opening_point: &opening_point,
                ram_read_write_cycle: &ram_read_write_cycle,
                ram_val_check_cycle: &ram_val_check_cycle,
                registers_read_write_cycle: &registers_read_write_cycle,
                registers_val_evaluation_cycle: &registers_val_evaluation_cycle,
                gamma,
            });

        assert_eq!(
            coefficients,
            Ok(ClaimReductionOutputCoefficients {
                ram: Fr::from_u64(1),
                rd: gamma,
            })
        );
    }

    #[test]
    fn output_coefficient_polynomials_follow_boolean_sumcheck_order() {
        let ram_read_write_cycle = vec![Fr::from_u64(0), Fr::from_u64(0)];
        let ram_val_check_cycle = vec![Fr::from_u64(1), Fr::from_u64(0)];
        let registers_read_write_cycle = vec![Fr::from_u64(0), Fr::from_u64(1)];
        let registers_val_evaluation_cycle = vec![Fr::from_u64(1), Fr::from_u64(1)];
        let gamma = Fr::from_u64(7);

        let polynomials = claim_reduction_output_coefficient_polynomials(
            ClaimReductionOutputCoefficientPolynomialInputs {
                num_vars: 2,
                trace_dimensions: TraceDimensions::new(2),
                ram_read_write_cycle: &ram_read_write_cycle,
                ram_val_check_cycle: &ram_val_check_cycle,
                registers_read_write_cycle: &registers_read_write_cycle,
                registers_val_evaluation_cycle: &registers_val_evaluation_cycle,
                gamma,
            },
        )
        .map(|(ram, rd)| (ram.evals().to_vec(), rd.evals().to_vec()))
        .map_err(|error| error.to_string());

        assert_eq!(
            polynomials,
            Ok((
                vec![Fr::from_u64(1), gamma, Fr::from_u64(0), Fr::from_u64(0)],
                vec![Fr::from_u64(0), Fr::from_u64(0), Fr::from_u64(1), gamma],
            )),
        );
    }

    #[test]
    fn output_coefficient_polynomials_reject_trace_dimension_mismatch() {
        let cycle = vec![Fr::from_u64(0)];

        assert_eq!(
            claim_reduction_output_coefficient_polynomials(
                ClaimReductionOutputCoefficientPolynomialInputs {
                    num_vars: 1,
                    trace_dimensions: TraceDimensions::new(2),
                    ram_read_write_cycle: &cycle,
                    ram_val_check_cycle: &cycle,
                    registers_read_write_cycle: &cycle,
                    registers_val_evaluation_cycle: &cycle,
                    gamma: Fr::from_u64(7),
                },
            ),
            Err(JoltFormulaPointError::ChallengeLengthMismatch {
                expected: 2,
                got: 1,
            })
        );
    }

    #[test]
    fn output_claim_uses_shared_coefficients() {
        let opening_point = vec![Fr::from_u64(0), Fr::from_u64(1)];
        let ram_read_write_cycle = vec![Fr::from_u64(0), Fr::from_u64(1)];
        let ram_val_check_cycle = vec![Fr::from_u64(1), Fr::from_u64(0)];
        let registers_read_write_cycle = vec![Fr::from_u64(0), Fr::from_u64(0)];
        let registers_val_evaluation_cycle = vec![Fr::from_u64(0), Fr::from_u64(1)];
        let gamma = Fr::from_u64(7);

        let output = claim_reduction_output_claim(ClaimReductionOutputClaimInputs {
            coefficients: ClaimReductionOutputCoefficientInputs {
                opening_point: &opening_point,
                ram_read_write_cycle: &ram_read_write_cycle,
                ram_val_check_cycle: &ram_val_check_cycle,
                registers_read_write_cycle: &registers_read_write_cycle,
                registers_val_evaluation_cycle: &registers_val_evaluation_cycle,
                gamma,
            },
            ram_inc: Fr::from_u64(3),
            rd_inc: Fr::from_u64(5),
        });

        assert_eq!(
            output,
            Ok(Fr::from_u64(3) + gamma * gamma * gamma * Fr::from_u64(5))
        );
    }

    #[test]
    fn output_coefficients_reject_cycle_length_mismatch() {
        let point = vec![Fr::from_u64(0), Fr::from_u64(1)];
        let short = vec![Fr::from_u64(0)];

        assert_eq!(
            claim_reduction_output_coefficients(ClaimReductionOutputCoefficientInputs {
                opening_point: &point,
                ram_read_write_cycle: &short,
                ram_val_check_cycle: &point,
                registers_read_write_cycle: &point,
                registers_val_evaluation_cycle: &point,
                gamma: Fr::from_u64(7),
            }),
            Err(JoltFormulaPointError::OpeningPointLengthMismatch {
                expected: 2,
                got: 1,
            })
        );
    }
}
