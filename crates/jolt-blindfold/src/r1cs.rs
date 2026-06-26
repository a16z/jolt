use jolt_field::Field;
use jolt_r1cs::{
    assert_claim_expr_eq, ClaimSourceTable, ClaimSources, LinearCombination, R1csBuilder, Variable,
};
use jolt_sumcheck::{
    append_sumcheck_r1cs_constraints_for_domain, SumcheckR1csError, SumcheckR1csLayout,
    SumcheckR1csRoundLayout,
};

use crate::{BlindFoldStage, BlindFoldStatement, Error, LayoutError};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layout {
    pub witness_row_len: usize,
    pub stages: Vec<StageLayout>,
    pub final_openings: Vec<FinalOpeningLayout>,
}

impl Layout {
    pub fn stage_count(&self) -> usize {
        self.stages.len()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StageLayout {
    pub sumcheck: SumcheckR1csLayout,
    pub output_claim_rows: Vec<OutputClaimRowLayout>,
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OutputClaimRowLayout {
    pub variables: Vec<Variable>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FinalOpeningLayout {
    pub evaluation: Option<Variable>,
    pub blinding: Option<Variable>,
}

impl<F, O, Com, P, Ch> BlindFoldStatement<F, O, Com, P, Ch>
where
    F: Field,
    O: Clone + PartialEq,
    P: Clone + PartialEq,
    Ch: Clone + PartialEq,
{
    pub fn build_with_sources(
        &self,
        builder: &mut R1csBuilder<F>,
        publics: &[(P, F)],
        challenges: &[(Ch, F)],
    ) -> Result<Layout, Error> {
        let layout = self.allocate_layout(builder)?;
        let mut claim_sources = ClaimSourceTable::<F, O, P, Ch>::new();
        insert_output_claim_sources(self, &layout, &mut claim_sources)?;
        for (id, value) in publics {
            claim_sources.insert_public(id.clone(), *value);
        }
        for (id, value) in challenges {
            claim_sources.insert_challenge(id.clone(), *value);
        }
        self.append(builder, &layout, &mut claim_sources)?;
        Ok(layout)
    }
}

impl<F, O, Com, P, Ch> BlindFoldStatement<F, O, Com, P, Ch>
where
    F: Field,
{
    pub fn build<R>(
        &self,
        builder: &mut R1csBuilder<F>,
        claim_sources: &mut R,
    ) -> Result<Layout, Error>
    where
        R: ClaimSources<F, Opening = O, Challenge = Ch, Public = P>,
    {
        let layout = self.allocate_layout(builder)?;
        self.append(builder, &layout, claim_sources)?;
        Ok(layout)
    }

    pub fn append<R>(
        &self,
        builder: &mut R1csBuilder<F>,
        layout: &Layout,
        claim_sources: &mut R,
    ) -> Result<(), Error>
    where
        R: ClaimSources<F, Opening = O, Challenge = Ch, Public = P>,
    {
        validate_stage_count(self, layout)?;
        validate_final_opening_count(self, layout)?;

        for (stage_index, (stage, stage_layout)) in
            self.stages.iter().zip(&layout.stages).enumerate()
        {
            assert_claim_expr_eq(
                builder,
                &stage.input_claim,
                stage_layout.sumcheck.input_claim,
                claim_sources,
            )?;

            append_sumcheck_r1cs_constraints_for_domain(
                builder,
                stage.statement,
                &stage.consistency.rounds,
                &stage_layout.sumcheck,
                stage.domain,
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

        for (binding, binding_layout) in self.final_openings.iter().zip(&layout.final_openings) {
            let Some(evaluation) = binding_layout.evaluation else {
                continue;
            };
            let mut combined = LinearCombination::zero();
            for (opening_id, &coefficient) in binding.opening_ids.iter().zip(&binding.coefficients)
            {
                combined = combined
                    + claim_sources
                        .opening(opening_id)?
                        .into_linear_combination()
                        .scale(coefficient);
            }
            builder.assert_equal(combined, evaluation);
        }

        Ok(())
    }

    pub fn allocate_layout(&self, builder: &mut R1csBuilder<F>) -> Result<Layout, LayoutError> {
        let witness_row_len = witness_row_len(self)?;

        let coefficients = self
            .stages
            .iter()
            .enumerate()
            .map(|(stage_index, stage)| {
                validate_stage_statement(stage.statement, &stage.consistency.rounds).map_err(
                    |source| LayoutError::Sumcheck {
                        stage_index,
                        source,
                    },
                )?;
                Ok(stage
                    .consistency
                    .rounds
                    .iter()
                    .map(|round| {
                        let coefficients = (0..=round.degree)
                            .map(|_| builder.alloc_unknown())
                            .collect::<Vec<_>>();
                        for _ in coefficients.len()..witness_row_len {
                            let _ = builder.alloc(F::zero());
                        }
                        coefficients
                    })
                    .collect::<Vec<_>>())
            })
            .collect::<Result<Vec<_>, _>>()?;

        let output_claim_rows = self
            .stages
            .iter()
            .map(|stage| allocate_output_claim_rows(builder, stage, witness_row_len))
            .collect::<Vec<_>>();

        let stages = self
            .stages
            .iter()
            .zip(coefficients)
            .zip(output_claim_rows)
            .map(|((stage, stage_coefficients), output_claim_rows)| {
                let input_claim = builder.alloc_unknown();
                let mut claim_in = input_claim;
                let mut rounds = Vec::with_capacity(stage.consistency.rounds.len());

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
                    output_claim_rows,
                    sumcheck: SumcheckR1csLayout {
                        input_claim,
                        rounds,
                        output_claim: claim_in,
                    },
                })
            })
            .collect::<Result<Vec<_>, LayoutError>>()?;

        let final_openings = self
            .final_openings
            .iter()
            .map(|binding| FinalOpeningLayout {
                evaluation: (!binding.opening_ids.is_empty())
                    .then(|| allocate_private_row_scalar(builder, witness_row_len)),
                blinding: (!binding.opening_ids.is_empty())
                    .then(|| allocate_private_row_scalar(builder, witness_row_len)),
            })
            .collect();

        Ok(Layout {
            witness_row_len,
            stages,
            final_openings,
        })
    }
}

fn allocate_private_row_scalar<F: Field>(
    builder: &mut R1csBuilder<F>,
    witness_row_len: usize,
) -> Variable {
    // Final evaluation bindings are opened at fixed coordinates. Keep them in
    // dedicated rows so those openings cannot reveal unrelated witness values.
    pad_to_witness_row_boundary(builder, witness_row_len);
    let variable = builder.alloc_unknown();
    pad_to_witness_row_boundary(builder, witness_row_len);
    variable
}

fn pad_to_witness_row_boundary<F: Field>(builder: &mut R1csBuilder<F>, witness_row_len: usize) {
    while !(builder.num_vars() - 1).is_multiple_of(witness_row_len) {
        let _ = builder.alloc(F::zero());
    }
}

fn witness_row_len<F, O, P, Ch, C>(
    statement: &BlindFoldStatement<F, O, C, P, Ch>,
) -> Result<usize, LayoutError> {
    let round_coefficients = statement
        .stages
        .iter()
        .flat_map(|stage| &stage.consistency.rounds)
        .map(|round| round.degree.saturating_add(1))
        .max()
        .unwrap_or(1);
    let output_claim_row_len = statement
        .stages
        .iter()
        .map(|stage| stage.output_claim_rows.row_len)
        .max()
        .unwrap_or(0);
    let row_len = round_coefficients.max(output_claim_row_len).max(1);
    row_len
        .checked_next_power_of_two()
        .ok_or(LayoutError::DimensionOverflow {
            name: "witness row length",
            value: row_len,
        })
}

fn allocate_output_claim_rows<F, O, P, Ch, C>(
    builder: &mut R1csBuilder<F>,
    stage: &BlindFoldStage<F, O, C, P, Ch>,
    witness_row_len: usize,
) -> Vec<OutputClaimRowLayout>
where
    F: Field,
{
    let row_count = stage.output_claim_rows.commitments.commitments.len();
    let row_len = stage.output_claim_rows.row_len;
    let mut remaining_openings = stage.output_claim_rows.opening_ids.len();
    let mut rows = Vec::with_capacity(row_count);

    for _ in 0..row_count {
        let opening_slots = remaining_openings.min(row_len);
        let mut variables = Vec::with_capacity(witness_row_len);
        for slot in 0..witness_row_len {
            let variable = if slot < opening_slots {
                builder.alloc_unknown()
            } else {
                builder.alloc(F::zero())
            };
            variables.push(variable);
        }
        remaining_openings -= opening_slots;
        rows.push(OutputClaimRowLayout { variables });
    }

    rows
}

fn insert_output_claim_sources<F, O, P, Ch, C>(
    statement: &BlindFoldStatement<F, O, C, P, Ch>,
    layout: &Layout,
    claim_sources: &mut ClaimSourceTable<F, O, P, Ch>,
) -> Result<(), Error>
where
    F: Field,
    O: Clone + PartialEq,
{
    let mut inserted = Vec::<(O, Variable)>::new();
    for (stage, stage_layout) in statement.stages.iter().zip(&layout.stages) {
        let row_len = stage.output_claim_rows.row_len;
        let variables = stage_layout
            .output_claim_rows
            .iter()
            .flat_map(|row| row.variables.iter().take(row_len));
        for (opening_id, &variable) in stage.output_claim_rows.opening_ids.iter().zip(variables) {
            claim_sources.insert_opening(opening_id.clone(), variable);
            inserted.push((opening_id.clone(), variable));
        }
        for alias in &stage.output_claim_rows.opening_aliases {
            let variable = inserted
                .iter()
                .find_map(|(opening_id, variable)| {
                    (opening_id == &alias.source).then_some(*variable)
                })
                .ok_or(Error::MissingOpeningAliasSource)?;
            claim_sources.insert_opening(alias.alias.clone(), variable);
            inserted.push((alias.alias.clone(), variable));
        }
    }
    Ok(())
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

fn validate_stage_count<F, O, P, Ch, C>(
    statement: &BlindFoldStatement<F, O, C, P, Ch>,
    layout: &Layout,
) -> Result<(), Error> {
    if statement.stages.len() != layout.stages.len() {
        return Err(Error::LayoutStageCountMismatch {
            statement_stages: statement.stages.len(),
            layout_stages: layout.stages.len(),
        });
    }

    Ok(())
}

fn validate_final_opening_count<F, O, P, Ch, C>(
    statement: &BlindFoldStatement<F, O, C, P, Ch>,
    layout: &Layout,
) -> Result<(), Error> {
    if statement.final_openings.len() != layout.final_openings.len() {
        return Err(Error::LayoutStageCountMismatch {
            statement_stages: statement.final_openings.len(),
            layout_stages: layout.final_openings.len(),
        });
    }

    Ok(())
}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "tests may panic on assertion failures")]
mod tests {
    use super::*;
    use crate::{BlindFoldStage, BlindFoldStatement, CommittedClaimRows, OpeningAlias};
    use jolt_claims::{challenge, constant, opening, derived, Expr};
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_r1cs::{ClaimLoweringError, ClaimSourceTable, R1csBuilderError};
    use jolt_sumcheck::{
        CommittedOutputClaims, CommittedSumcheckConsistency, SumcheckDomainSpec, SumcheckR1csError,
        SumcheckR1csRoundLayout, SumcheckStatement, VerifiedCommittedRound,
    };

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Opening {
        Input,
        Output,
        Alias,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Public {
        Scale,
    }

    fn consistency(degrees: &[usize]) -> CommittedSumcheckConsistency<Fr, ()> {
        CommittedSumcheckConsistency {
            rounds: degrees
                .iter()
                .enumerate()
                .map(|(index, &degree)| VerifiedCommittedRound {
                    commitment: (),
                    degree,
                    challenge: Fr::from_u64(index as u64 + 11),
                })
                .collect(),
        }
    }

    fn committed_consistency(rounds: &[(usize, u64)]) -> CommittedSumcheckConsistency<Fr, ()> {
        CommittedSumcheckConsistency {
            rounds: rounds
                .iter()
                .map(|&(degree, challenge)| VerifiedCommittedRound {
                    commitment: (),
                    degree,
                    challenge: Fr::from_u64(challenge),
                })
                .collect(),
        }
    }

    fn output_claim_rows() -> CommittedClaimRows<(), ()> {
        CommittedClaimRows::empty()
    }

    fn empty_stage(
        num_vars: usize,
        degree: usize,
        round_degrees: &[usize],
    ) -> BlindFoldStage<Fr, (), ()> {
        let claim: Expr<Fr, ()> = constant(Fr::from_u64(0));
        BlindFoldStage::new(
            "stage",
            SumcheckStatement::new(num_vars, degree),
            SumcheckDomainSpec::BooleanHypercube,
            consistency(round_degrees),
            output_claim_rows(),
            claim.clone(),
            claim,
        )
    }

    fn claim_stage(
        num_vars: usize,
        degree: usize,
        rounds: &[(usize, u64)],
        input_claim: Expr<Fr, Opening, Public>,
        output_claim: Expr<Fr, Opening, Public>,
    ) -> BlindFoldStage<Fr, Opening, (), Public> {
        BlindFoldStage::new(
            "stage",
            SumcheckStatement::new(num_vars, degree),
            SumcheckDomainSpec::BooleanHypercube,
            committed_consistency(rounds),
            CommittedClaimRows::empty(),
            input_claim,
            output_claim,
        )
    }

    fn assign(builder: &mut R1csBuilder<Fr>, variable: Variable, value: u64) {
        builder
            .assign(variable, Fr::from_u64(value))
            .expect("assignment succeeds");
    }

    fn assign_round(
        builder: &mut R1csBuilder<Fr>,
        round: &SumcheckR1csRoundLayout,
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
        let statement = BlindFoldStatement::new(vec![empty_stage(2, 3, &[1, 3])], Vec::new());
        let mut builder = R1csBuilder::<Fr>::new();

        let layout = statement
            .allocate_layout(&mut builder)
            .expect("layout allocates");

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
                variable: Variable::new(1),
            },
        );
    }

    #[test]
    fn append_rejects_layout_stage_count_mismatch() {
        let statement = BlindFoldStatement::new(vec![empty_stage(1, 2, &[2])], Vec::new());
        let mut builder = R1csBuilder::<Fr>::new();
        let mut sources = ClaimSourceTable::<Fr, (), ()>::new();
        let layout = Layout {
            witness_row_len: 1,
            stages: Vec::new(),
            final_openings: Vec::new(),
        };

        let error = statement
            .append(&mut builder, &layout, &mut sources)
            .expect_err("stage counts differ");

        assert_eq!(
            error,
            Error::LayoutStageCountMismatch {
                statement_stages: 1,
                layout_stages: 0,
            }
        );
    }

    #[test]
    fn rejects_round_count_mismatch() {
        let statement = BlindFoldStatement::new(vec![empty_stage(2, 2, &[2])], Vec::new());
        let mut builder = R1csBuilder::<Fr>::new();

        let error = statement
            .allocate_layout(&mut builder)
            .expect_err("round counts differ");

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
        let statement = BlindFoldStatement::new(vec![empty_stage(1, 2, &[3])], Vec::new());
        let mut builder = R1csBuilder::<Fr>::new();

        let error = statement
            .allocate_layout(&mut builder)
            .expect_err("degree exceeds bound");

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

        let statement = BlindFoldStatement::new(
            vec![claim_stage(
                1,
                1,
                &[(1, 2)],
                opening(Opening::Input) * derived(Public::Scale) + challenge(0usize),
                constant(Fr::from_u64(11)),
            )],
            Vec::new(),
        );

        let layout = statement
            .build(&mut builder, &mut sources)
            .expect("constraints should build");

        let stage_layout = &layout.stages[0].sumcheck;
        assign(&mut builder, stage_layout.input_claim, 10);
        assign_round(&mut builder, &stage_layout.rounds[0], &[3, 4], 11);

        let witness = builder.witness().expect("witness is assigned");
        assert!(builder.into_matrices().check_witness(&witness).is_ok());
    }

    #[test]
    fn lowers_output_claim_alias_to_source_variable() {
        let input_claim: Expr<Fr, Opening, Public> = constant(Fr::from_u64(10));
        let output_claim: Expr<Fr, Opening, Public> = opening(Opening::Alias);
        let statement = BlindFoldStatement::new(
            vec![BlindFoldStage::new(
                "alias",
                SumcheckStatement::new(1, 1),
                SumcheckDomainSpec::BooleanHypercube,
                committed_consistency(&[(1, 2)]),
                CommittedClaimRows::new(
                    vec![Opening::Output],
                    1,
                    CommittedOutputClaims {
                        commitments: vec![()],
                    },
                )
                .with_aliases([OpeningAlias::new(Opening::Alias, Opening::Output)]),
                input_claim,
                output_claim,
            )],
            Vec::new(),
        );
        let mut builder = R1csBuilder::<Fr>::new();

        let layout = statement
            .build_with_sources(&mut builder, &[], &[])
            .expect("constraints should build");
        let stage_layout = &layout.stages[0].sumcheck;
        assign(&mut builder, stage_layout.input_claim, 10);
        assign_round(&mut builder, &stage_layout.rounds[0], &[3, 4], 11);
        assign(
            &mut builder,
            layout.stages[0].output_claim_rows[0].variables[0],
            11,
        );

        let witness = builder.witness().expect("witness is assigned");
        assert!(builder.into_matrices().check_witness(&witness).is_ok());
    }

    #[test]
    fn lowers_centered_integer_domain_round_sums() {
        let mut builder = R1csBuilder::<Fr>::new();
        let statement = BlindFoldStatement::new(
            vec![BlindFoldStage::new(
                "centered",
                SumcheckStatement::new(1, 2),
                SumcheckDomainSpec::centered_integer(4),
                committed_consistency(&[(2, 5)]),
                CommittedClaimRows::empty(),
                constant(Fr::from_u64(26)),
                constant(Fr::from_u64(86)),
            )],
            Vec::new(),
        );
        let mut sources = ClaimSourceTable::<Fr, (), ()>::new();

        let layout = statement
            .build(&mut builder, &mut sources)
            .expect("constraints should build for centered domain");
        let stage_layout = &layout.stages[0].sumcheck;
        assign(&mut builder, stage_layout.input_claim, 26);
        assign_round(&mut builder, &stage_layout.rounds[0], &[1, 2, 3], 86);

        let witness = builder.witness().expect("witness is assigned");
        assert!(builder.into_matrices().check_witness(&witness).is_ok());
    }

    #[test]
    fn reports_missing_claim_source() {
        let statement = BlindFoldStatement::new(
            vec![claim_stage(
                0,
                1,
                &[],
                opening(Opening::Input),
                constant(Fr::from_u64(0)),
            )],
            Vec::new(),
        );
        let mut builder = R1csBuilder::<Fr>::new();
        let mut sources = ClaimSourceTable::<Fr, Opening, Public>::new();

        let error = statement
            .build(&mut builder, &mut sources)
            .expect_err("opening is missing");

        assert_eq!(error, Error::Claim(ClaimLoweringError::MissingOpening));
    }

    #[test]
    fn reports_sumcheck_layout_errors_with_stage_index() {
        let statement = BlindFoldStatement::new(
            vec![claim_stage(
                1,
                1,
                &[(1, 2)],
                constant(Fr::from_u64(10)),
                constant(Fr::from_u64(11)),
            )],
            Vec::new(),
        );
        let mut builder = R1csBuilder::<Fr>::new();
        let mut layout = statement
            .allocate_layout(&mut builder)
            .expect("layout allocates");
        layout.stages[0].sumcheck.rounds[0].claim_in =
            layout.stages[0].sumcheck.rounds[0].claim_out;
        let mut sources = ClaimSourceTable::<Fr, Opening, Public>::new();

        let error = statement
            .append(&mut builder, &layout, &mut sources)
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
