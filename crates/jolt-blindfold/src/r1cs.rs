use jolt_field::Field;
use jolt_r1cs::{assert_claim_expr_eq, ClaimSources, R1csBuilder, Variable};
use jolt_sumcheck::{
    append_sumcheck_r1cs_constraints, SumcheckR1csError, SumcheckR1csLayout,
    SumcheckR1csRoundLayout,
};

use crate::{Error, Inputs, InstanceClaims, LayoutError};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layout {
    pub stages: Vec<StageLayout>,
}

impl Layout {
    pub fn stage_count(&self) -> usize {
        self.stages.len()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StageLayout {
    pub sumcheck: SumcheckR1csLayout,
}

impl StageLayout {
    pub fn input_claim(&self) -> Variable {
        self.sumcheck.input_claim
    }

    pub fn round_count(&self) -> usize {
        self.sumcheck.round_count()
    }

    pub fn output_claim(&self) -> Variable {
        self.sumcheck.output_claim
    }
}

pub type RoundLayout = SumcheckR1csRoundLayout;

pub fn build<F, O, P, Ch, C, R>(
    builder: &mut R1csBuilder<F>,
    claims: &InstanceClaims<F, O, P, Ch>,
    inputs: &Inputs<F, C>,
    claim_sources: &mut R,
) -> Result<Layout, Error>
where
    F: Field,
    R: ClaimSources<F, Opening = O, Challenge = Ch, Public = P>,
{
    let layout = allocate_layout(builder, claims, inputs)?;
    append(builder, claims, inputs, &layout, claim_sources)?;
    Ok(layout)
}

pub fn append<F, O, P, Ch, C, R>(
    builder: &mut R1csBuilder<F>,
    claims: &InstanceClaims<F, O, P, Ch>,
    inputs: &Inputs<F, C>,
    layout: &Layout,
    claim_sources: &mut R,
) -> Result<(), Error>
where
    F: Field,
    R: ClaimSources<F, Opening = O, Challenge = Ch, Public = P>,
{
    validate_stage_counts(claims, inputs, layout)?;

    for (stage_index, ((stage, stage_input), stage_layout)) in claims
        .stages
        .iter()
        .zip(&inputs.stages)
        .zip(&layout.stages)
        .enumerate()
    {
        assert_claim_expr_eq(
            builder,
            &stage.input_claim,
            stage_layout.sumcheck.input_claim,
            claim_sources,
        )?;

        append_sumcheck_r1cs_constraints(
            builder,
            stage.statement,
            &stage_input.consistency.rounds,
            &stage_layout.sumcheck,
        )
        .map_err(|source| Error::Sumcheck {
            stage_index,
            source,
        })?;

        assert_claim_expr_eq(
            builder,
            &stage.output_claim,
            stage_layout.sumcheck.output_claim,
            claim_sources,
        )?;
    }

    Ok(())
}

pub fn allocate_layout<F, O, P, Ch, C>(
    builder: &mut R1csBuilder<F>,
    claims: &InstanceClaims<F, O, P, Ch>,
    inputs: &Inputs<F, C>,
) -> Result<Layout, LayoutError>
where
    F: Field,
{
    if claims.stages.len() != inputs.stages.len() {
        return Err(LayoutError::StageCountMismatch {
            claim_stages: claims.stages.len(),
            input_stages: inputs.stages.len(),
        });
    }

    let coefficient_row_len = inputs
        .stages
        .iter()
        .flat_map(|stage| &stage.consistency.rounds)
        .map(|round| round.degree.saturating_add(1))
        .max()
        .unwrap_or(1)
        .next_power_of_two();

    let coefficients = claims
        .stages
        .iter()
        .zip(&inputs.stages)
        .enumerate()
        .map(|(stage_index, (stage, stage_input))| {
            validate_stage_statement(stage.statement, &stage_input.consistency.rounds).map_err(
                |source| LayoutError::Sumcheck {
                    stage_index,
                    source,
                },
            )?;
            Ok(stage_input
                .consistency
                .rounds
                .iter()
                .map(|round| {
                    let coefficients = (0..=round.degree)
                        .map(|_| builder.alloc_unknown())
                        .collect::<Vec<_>>();
                    for _ in coefficients.len()..coefficient_row_len {
                        let _ = builder.alloc(F::zero());
                    }
                    coefficients
                })
                .collect::<Vec<_>>())
        })
        .collect::<Result<Vec<_>, _>>()?;

    let stages = inputs
        .stages
        .iter()
        .zip(coefficients)
        .map(|(stage_input, stage_coefficients)| {
            let input_claim = builder.alloc_unknown();
            let mut claim_in = input_claim;
            let mut rounds = Vec::with_capacity(stage_input.consistency.rounds.len());

            for coefficients in stage_coefficients {
                let claim_out = builder.alloc_unknown();
                rounds.push(SumcheckR1csRoundLayout {
                    claim_in,
                    coefficients,
                    claim_out,
                });
                claim_in = claim_out;
            }

            Ok(StageLayout {
                sumcheck: SumcheckR1csLayout {
                    input_claim,
                    rounds,
                    output_claim: claim_in,
                },
            })
        })
        .collect::<Result<Vec<_>, LayoutError>>()?;

    Ok(Layout { stages })
}

fn validate_stage_statement<F, C>(
    statement: jolt_sumcheck::SumcheckStatement,
    rounds: &[jolt_sumcheck::VerifiedCommittedRound<F, C>],
) -> Result<(), SumcheckR1csError> {
    if statement.num_vars != rounds.len() {
        return Err(SumcheckR1csError::WrongNumberOfRounds {
            expected: statement.num_vars,
            actual: rounds.len(),
        });
    }

    for (round_index, round) in rounds.iter().enumerate() {
        if round.degree > statement.degree {
            return Err(SumcheckR1csError::DegreeBoundExceeded {
                round_index,
                bound: statement.degree,
                actual: round.degree,
            });
        }
    }

    Ok(())
}

fn validate_stage_counts<F, O, P, Ch, C>(
    claims: &InstanceClaims<F, O, P, Ch>,
    inputs: &Inputs<F, C>,
    layout: &Layout,
) -> Result<(), Error> {
    if claims.stages.len() != inputs.stages.len() {
        return Err(LayoutError::StageCountMismatch {
            claim_stages: claims.stages.len(),
            input_stages: inputs.stages.len(),
        }
        .into());
    }

    if claims.stages.len() != layout.stages.len() {
        return Err(Error::LayoutStageCountMismatch {
            claim_stages: claims.stages.len(),
            layout_stages: layout.stages.len(),
        });
    }

    Ok(())
}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "tests may panic on assertion failures")]
mod tests {
    use super::*;
    use crate::{Inputs, InstanceClaims, StageClaims, StageInput};
    use jolt_claims::{challenge, constant, opening, public, Expr};
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_r1cs::{ClaimLoweringError, ClaimSourceTable, R1csBuilderError};
    use jolt_sumcheck::{
        CommittedSumcheckConsistency, SumcheckR1csError, SumcheckStatement, VerifiedCommittedRound,
    };

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Opening {
        Input,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Public {
        Scale,
    }

    fn empty_stage(num_vars: usize, degree: usize) -> StageClaims<Fr, ()> {
        let claim: Expr<Fr, ()> = constant(Fr::from_u64(0));
        StageClaims::new(
            "stage",
            SumcheckStatement::new(num_vars, degree),
            claim.clone(),
            claim,
        )
    }

    fn claim_stage(
        num_vars: usize,
        degree: usize,
        input_claim: Expr<Fr, Opening, Public>,
        output_claim: Expr<Fr, Opening, Public>,
    ) -> StageClaims<Fr, Opening, Public> {
        StageClaims::new(
            "stage",
            SumcheckStatement::new(num_vars, degree),
            input_claim,
            output_claim,
        )
    }

    fn stage_input(degrees: &[usize]) -> StageInput<Fr, ()> {
        StageInput::new(CommittedSumcheckConsistency {
            rounds: degrees
                .iter()
                .enumerate()
                .map(|(index, &degree)| VerifiedCommittedRound {
                    commitment: (),
                    degree,
                    challenge: Fr::from_u64(index as u64 + 11),
                })
                .collect(),
        })
    }

    fn committed_stage_input(rounds: &[(usize, u64)]) -> StageInput<Fr, ()> {
        StageInput::new(CommittedSumcheckConsistency {
            rounds: rounds
                .iter()
                .map(|&(degree, challenge)| VerifiedCommittedRound {
                    commitment: (),
                    degree,
                    challenge: Fr::from_u64(challenge),
                })
                .collect(),
        })
    }

    fn assign(builder: &mut R1csBuilder<Fr>, variable: Variable, value: u64) {
        builder
            .assign(variable, Fr::from_u64(value))
            .expect("assignment succeeds");
    }

    fn assign_round(
        builder: &mut R1csBuilder<Fr>,
        round: &RoundLayout,
        coefficients: &[u64],
        claim_out: u64,
    ) {
        for (&variable, &coefficient) in round.coefficients.iter().zip(coefficients) {
            assign(builder, variable, coefficient);
        }
        assign(builder, round.claim_out, claim_out);
    }

    #[test]
    fn allocates_claim_chain_and_coefficients() {
        let claims = InstanceClaims::new(vec![empty_stage(2, 3)]);
        let inputs = Inputs::new(vec![stage_input(&[1, 3])]);
        let mut builder = R1csBuilder::<Fr>::new();

        let layout = allocate_layout(&mut builder, &claims, &inputs).expect("layout allocates");

        assert_eq!(layout.stage_count(), 1);
        let stage = &layout.stages[0];
        assert_eq!(stage.round_count(), 2);
        assert_eq!(stage.sumcheck.rounds[0].degree(), 1);
        assert_eq!(stage.sumcheck.rounds[1].degree(), 3);
        assert_eq!(stage.input_claim(), stage.sumcheck.rounds[0].claim_in);
        assert_eq!(
            stage.sumcheck.rounds[0].claim_out,
            stage.sumcheck.rounds[1].claim_in
        );
        assert_eq!(stage.output_claim(), stage.sumcheck.rounds[1].claim_out);
        assert_eq!(
            builder.witness().expect_err("layout is witness-only"),
            R1csBuilderError::MissingWitnessValue {
                variable: Variable(1),
            },
        );
    }

    #[test]
    fn rejects_stage_count_mismatch() {
        let claims = InstanceClaims::new(vec![empty_stage(1, 2)]);
        let inputs = Inputs::<Fr, ()>::new(Vec::new());
        let mut builder = R1csBuilder::<Fr>::new();

        let error =
            allocate_layout(&mut builder, &claims, &inputs).expect_err("stage counts differ");

        assert_eq!(
            error,
            LayoutError::StageCountMismatch {
                claim_stages: 1,
                input_stages: 0,
            }
        );
    }

    #[test]
    fn rejects_round_count_mismatch() {
        let claims = InstanceClaims::new(vec![empty_stage(2, 2)]);
        let inputs = Inputs::new(vec![stage_input(&[2])]);
        let mut builder = R1csBuilder::<Fr>::new();

        let error =
            allocate_layout(&mut builder, &claims, &inputs).expect_err("round counts differ");

        assert_eq!(
            error,
            LayoutError::Sumcheck {
                stage_index: 0,
                source: SumcheckR1csError::WrongNumberOfRounds {
                    expected: 2,
                    actual: 1,
                },
            }
        );
    }

    #[test]
    fn rejects_degree_above_stage_bound() {
        let claims = InstanceClaims::new(vec![empty_stage(1, 2)]);
        let inputs = Inputs::new(vec![stage_input(&[3])]);
        let mut builder = R1csBuilder::<Fr>::new();

        let error =
            allocate_layout(&mut builder, &claims, &inputs).expect_err("degree exceeds bound");

        assert_eq!(
            error,
            LayoutError::Sumcheck {
                stage_index: 0,
                source: SumcheckR1csError::DegreeBoundExceeded {
                    round_index: 0,
                    bound: 2,
                    actual: 3,
                },
            }
        );
    }

    #[test]
    fn lowers_claim_sources_and_sumcheck_constraints() {
        let mut builder = R1csBuilder::<Fr>::new();
        let input = builder.alloc(Fr::from_u64(3));
        let mut sources = ClaimSourceTable::<Fr, Opening, Public>::new();
        sources.insert_opening(Opening::Input, input);
        sources.insert_challenge(0, Fr::from_u64(4));
        sources.insert_public(Public::Scale, Fr::from_u64(2));

        let claims = InstanceClaims::new(vec![claim_stage(
            1,
            1,
            opening(Opening::Input) * public(Public::Scale) + challenge(0),
            constant(Fr::from_u64(11)),
        )]);
        let inputs = Inputs::new(vec![committed_stage_input(&[(1, 2)])]);

        let layout =
            build(&mut builder, &claims, &inputs, &mut sources).expect("constraints should build");

        let stage_layout = &layout.stages[0].sumcheck;
        assign(&mut builder, stage_layout.input_claim, 10);
        assign_round(&mut builder, &stage_layout.rounds[0], &[3, 4], 11);

        let witness = builder.witness().expect("witness is assigned");
        assert!(builder.into_matrices().check_witness(&witness).is_ok());
    }

    #[test]
    fn reports_missing_claim_source() {
        let claims = InstanceClaims::new(vec![claim_stage(
            0,
            1,
            opening(Opening::Input),
            constant(Fr::from_u64(0)),
        )]);
        let inputs = Inputs::new(vec![committed_stage_input(&[])]);
        let mut builder = R1csBuilder::<Fr>::new();
        let mut sources = ClaimSourceTable::<Fr, Opening, Public>::new();

        let error =
            build(&mut builder, &claims, &inputs, &mut sources).expect_err("opening is missing");

        assert_eq!(error, Error::Claim(ClaimLoweringError::MissingOpening));
    }

    #[test]
    fn reports_sumcheck_layout_errors_with_stage_index() {
        let claims = InstanceClaims::new(vec![claim_stage(
            1,
            1,
            constant(Fr::from_u64(10)),
            constant(Fr::from_u64(11)),
        )]);
        let inputs = Inputs::new(vec![committed_stage_input(&[(1, 2)])]);
        let mut builder = R1csBuilder::<Fr>::new();
        let mut layout = allocate_layout(&mut builder, &claims, &inputs).expect("layout allocates");
        layout.stages[0].sumcheck.rounds[0].claim_in =
            layout.stages[0].sumcheck.rounds[0].claim_out;
        let mut sources = ClaimSourceTable::<Fr, Opening, Public>::new();

        let error = append(&mut builder, &claims, &inputs, &layout, &mut sources)
            .expect_err("layout claim chain is broken");

        assert_eq!(
            error,
            Error::Sumcheck {
                stage_index: 0,
                source: SumcheckR1csError::RoundClaimLinkMismatch {
                    round_index: 0,
                    expected: layout.stages[0].sumcheck.input_claim,
                    actual: layout.stages[0].sumcheck.rounds[0].claim_in,
                },
            }
        );
    }
}
