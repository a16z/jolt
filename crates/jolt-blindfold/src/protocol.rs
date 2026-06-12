use std::ops::Range;

use jolt_crypto::HomomorphicCommitment;
use jolt_field::Field;
use jolt_r1cs::{ConstraintMatrices, R1csBuilder, Variable};
use jolt_sumcheck::{CommittedOutputClaims, CommittedSumcheckConsistency};

use crate::{
    r1cs::{build_with_sources, Layout},
    BlindFoldProtocolBuilder, BlindFoldStatement, Error, RelaxedError, RelaxedInstance,
    VerificationError,
};

#[derive(Clone, Debug)]
pub struct BlindFoldProtocol<F: Field, Com> {
    pub sumcheck_consistency: Vec<CommittedSumcheckConsistency<F, Com>>,
    pub committed_output_claims: Vec<CommittedOutputClaims<Com>>,
    pub r1cs: ConstraintMatrices<F>,
    pub layout: Layout,
    pub dimensions: BlindFoldDimensions,
    pub eval_commitments: Vec<Com>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RowDimensions {
    pub row_len: usize,
    pub row_count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlindFoldDimensions {
    pub witness: RowDimensions,
    pub error: RowDimensions,
    pub witness_rows: WitnessRowLayout,
    pub coefficient_rows: usize,
    pub output_claim_rows: usize,
    pub auxiliary_rows: usize,
    pub coefficient_values: usize,
    pub auxiliary_values: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WitnessRowLayout {
    pub coefficients: Range<usize>,
    pub output_claims: Range<usize>,
    pub auxiliary: Range<usize>,
    pub padding: Range<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WitnessCoordinate {
    pub row: usize,
    pub column: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FinalOpeningWitnessCoordinates {
    pub evaluation: Option<WitnessCoordinate>,
    pub blinding: Option<WitnessCoordinate>,
}

impl<F, Com> BlindFoldProtocol<F, Com>
where
    F: Field,
{
    pub fn builder<O, P, Ch>() -> BlindFoldProtocolBuilder<F, O, Com, P, Ch> {
        BlindFoldProtocolBuilder::new()
    }
}

impl<F, Com> BlindFoldProtocol<F, Com>
where
    F: Field + Clone,
    Com: Clone,
{
    pub(crate) fn from_parts<O, P, Ch>(
        statement: &BlindFoldStatement<F, O, Com, P, Ch>,
        publics: &[(P, F)],
        challenges: &[(Ch, F)],
    ) -> Result<Self, VerificationError<F>>
    where
        O: Clone + PartialEq,
        P: Clone + PartialEq,
        Ch: Clone + PartialEq,
    {
        statement.validate()?;

        let sumcheck_consistency = statement.sumcheck_consistency();
        let committed_output_claims = statement.committed_output_claims();
        let (r1cs, layout) = statement.build_constraints(publics, challenges)?;
        let dimensions =
            layout.dimensions(&r1cs, &sumcheck_consistency, &committed_output_claims)?;

        Ok(Self {
            sumcheck_consistency,
            committed_output_claims,
            r1cs,
            layout,
            dimensions,
            eval_commitments: statement.final_opening_commitments(),
        })
    }
}

impl<F, Com> BlindFoldProtocol<F, Com>
where
    F: Field,
    Com: Clone + HomomorphicCommitment<F>,
{
    pub fn committed_relaxed_instance(
        &self,
        auxiliary_row_commitments: &[Com],
    ) -> Result<RelaxedInstance<F, Com>, RelaxedError> {
        if auxiliary_row_commitments.len() != self.dimensions.auxiliary_rows {
            return Err(RelaxedError::LengthMismatch {
                name: "auxiliary row commitments",
                expected: self.dimensions.auxiliary_rows,
                actual: auxiliary_row_commitments.len(),
            });
        }

        let mut witness_row_commitments = Vec::with_capacity(self.dimensions.witness.row_count);

        witness_row_commitments.extend(
            self.sumcheck_consistency
                .iter()
                .flat_map(|consistency| consistency.rounds.iter())
                .map(|round| round.commitment.clone()),
        );
        pad_rows::<F, Com>(
            &mut witness_row_commitments,
            self.dimensions.witness_rows.coefficients.end,
            "coefficient row commitments",
        )?;

        witness_row_commitments.extend(
            self.committed_output_claims
                .iter()
                .flat_map(|claims| claims.commitments.iter().cloned()),
        );
        pad_rows::<F, Com>(
            &mut witness_row_commitments,
            self.dimensions.witness_rows.output_claims.end,
            "output claim row commitments",
        )?;

        witness_row_commitments.extend_from_slice(auxiliary_row_commitments);
        pad_rows::<F, Com>(
            &mut witness_row_commitments,
            self.dimensions.witness_rows.auxiliary.end,
            "auxiliary row commitments",
        )?;
        pad_rows::<F, Com>(
            &mut witness_row_commitments,
            self.dimensions.witness.row_count,
            "witness row commitments",
        )?;

        Ok(RelaxedInstance::new(
            F::one(),
            witness_row_commitments,
            vec![Com::default(); self.dimensions.error.row_count],
            self.eval_commitments.clone(),
        ))
    }
}

impl<F, Com> BlindFoldProtocol<F, Com>
where
    F: Field,
{
    pub fn validate_cross_term_error_rows(
        &self,
        cross_term_error_row_commitments: &[Com],
    ) -> Result<(), RelaxedError> {
        ensure_len(
            "cross-term error row commitments",
            self.dimensions.error.row_count,
            cross_term_error_row_commitments.len(),
        )
    }

    pub fn final_opening_witness_coordinates(
        &self,
    ) -> Result<Vec<FinalOpeningWitnessCoordinates>, RelaxedError> {
        self.layout
            .final_openings
            .iter()
            .map(|layout| {
                Ok(FinalOpeningWitnessCoordinates {
                    evaluation: layout
                        .evaluation
                        .map(|variable| self.witness_coordinate(variable))
                        .transpose()?,
                    blinding: layout
                        .blinding
                        .map(|variable| self.witness_coordinate(variable))
                        .transpose()?,
                })
            })
            .collect()
    }

    fn witness_coordinate(&self, variable: Variable) -> Result<WitnessCoordinate, RelaxedError> {
        let witness_index =
            variable
                .index()
                .checked_sub(1)
                .ok_or(RelaxedError::InconsistentDimensions {
                    name: "witness variable",
                    total: self.dimensions.witness.row_count * self.dimensions.witness.row_len,
                    used: 0,
                })?;
        let witness_values = self
            .dimensions
            .witness
            .row_count
            .checked_mul(self.dimensions.witness.row_len)
            .ok_or(RelaxedError::DimensionOverflow {
                name: "witness values",
                value: self.dimensions.witness.row_count,
            })?;
        if witness_index >= witness_values {
            return Err(RelaxedError::InconsistentDimensions {
                name: "witness variable",
                total: witness_values,
                used: witness_index + 1,
            });
        }

        Ok(WitnessCoordinate {
            row: witness_index / self.dimensions.witness.row_len,
            column: witness_index % self.dimensions.witness.row_len,
        })
    }
}

impl<F, Com> BlindFoldProtocol<F, Com>
where
    F: Field,
    Com: Clone + HomomorphicCommitment<F>,
{
    pub fn random_relaxed_instance(
        &self,
        round_commitments: &[Com],
        output_claim_row_commitments: &[Com],
        auxiliary_row_commitments: &[Com],
        error_row_commitments: &[Com],
        eval_commitments: &[Com],
        u: F,
    ) -> Result<RelaxedInstance<F, Com>, RelaxedError> {
        ensure_len(
            "random round commitments",
            self.dimensions.coefficient_rows,
            round_commitments.len(),
        )?;
        ensure_len(
            "random output claim row commitments",
            self.dimensions.output_claim_rows,
            output_claim_row_commitments.len(),
        )?;
        ensure_len(
            "random auxiliary row commitments",
            self.dimensions.auxiliary_rows,
            auxiliary_row_commitments.len(),
        )?;
        ensure_len(
            "random error row commitments",
            self.dimensions.error.row_count,
            error_row_commitments.len(),
        )?;
        ensure_len(
            "random eval commitments",
            self.eval_commitments.len(),
            eval_commitments.len(),
        )?;

        let mut witness_row_commitments = Vec::with_capacity(self.dimensions.witness.row_count);
        witness_row_commitments.extend_from_slice(round_commitments);
        pad_rows::<F, Com>(
            &mut witness_row_commitments,
            self.dimensions.witness_rows.coefficients.end,
            "random coefficient row commitments",
        )?;
        witness_row_commitments.extend_from_slice(output_claim_row_commitments);
        pad_rows::<F, Com>(
            &mut witness_row_commitments,
            self.dimensions.witness_rows.output_claims.end,
            "random output claim row commitments",
        )?;
        witness_row_commitments.extend_from_slice(auxiliary_row_commitments);
        pad_rows::<F, Com>(
            &mut witness_row_commitments,
            self.dimensions.witness_rows.auxiliary.end,
            "random auxiliary row commitments",
        )?;
        pad_rows::<F, Com>(
            &mut witness_row_commitments,
            self.dimensions.witness.row_count,
            "random witness row commitments",
        )?;

        Ok(RelaxedInstance::new(
            u,
            witness_row_commitments,
            error_row_commitments.to_vec(),
            eval_commitments.to_vec(),
        ))
    }
}

impl<F, O, Com, P, Ch> BlindFoldStatement<F, O, Com, P, Ch>
where
    F: Field + Clone,
    O: Clone + PartialEq,
    Com: Clone,
    P: Clone + PartialEq,
    Ch: Clone + PartialEq,
{
    fn validate(&self) -> Result<(), VerificationError<F>> {
        self.validate_unique_openings()?;
        self.validate_output_claim_rows()?;
        for binding in &self.final_openings {
            if binding.opening_ids.is_empty() {
                return Err(Error::EmptyFinalOpeningBinding.into());
            }
            if binding.opening_ids.len() != binding.coefficients.len() {
                return Err(RelaxedError::LengthMismatch {
                    name: "final opening coefficients",
                    expected: binding.opening_ids.len(),
                    actual: binding.coefficients.len(),
                }
                .into());
            }
        }
        Ok(())
    }

    fn validate_unique_openings(&self) -> Result<(), Error> {
        let mut openings = Vec::new();
        for stage in &self.stages {
            for opening_id in &stage.output_claim_rows.opening_ids {
                if openings.contains(opening_id) {
                    return Err(Error::DuplicateOpeningSource);
                }
                openings.push(opening_id.clone());
            }
            for alias in &stage.output_claim_rows.opening_aliases {
                if openings.contains(&alias.alias) {
                    return Err(Error::DuplicateOpeningSource);
                }
                if !openings.contains(&alias.source) {
                    return Err(Error::MissingOpeningAliasSource);
                }
                openings.push(alias.alias.clone());
            }
        }
        Ok(())
    }

    fn validate_output_claim_rows(&self) -> Result<(), VerificationError<F>> {
        for stage in &self.stages {
            let rows = &stage.output_claim_rows;
            let row_count = rows.commitments.commitments.len();
            if row_count == 0 {
                if !rows.opening_ids.is_empty() {
                    return Err(Error::OpeningRowCapacityExceeded {
                        name: "output claim rows",
                        ids: rows.opening_ids.len(),
                        slots: 0,
                    }
                    .into());
                }
                continue;
            }
            if rows.opening_ids.is_empty() {
                if row_count != 0 {
                    return Err(Error::CommittedRowCountMismatch {
                        name: "output claim rows",
                        expected: 0,
                        actual: row_count,
                    }
                    .into());
                }
                continue;
            }
            if rows.row_len == 0 {
                return Err(Error::MissingRowLength {
                    name: "output claim rows",
                }
                .into());
            }
            let slots =
                row_count
                    .checked_mul(rows.row_len)
                    .ok_or(RelaxedError::DimensionOverflow {
                        name: "output claim row slots",
                        value: row_count,
                    })?;
            if rows.opening_ids.len() > slots {
                return Err(Error::OpeningRowCapacityExceeded {
                    name: "output claim rows",
                    ids: rows.opening_ids.len(),
                    slots,
                }
                .into());
            }
            let expected_rows = rows.opening_ids.len().div_ceil(rows.row_len);
            if row_count != expected_rows {
                return Err(Error::CommittedRowCountMismatch {
                    name: "output claim rows",
                    expected: expected_rows,
                    actual: row_count,
                }
                .into());
            }
        }
        Ok(())
    }

    fn sumcheck_consistency(&self) -> Vec<CommittedSumcheckConsistency<F, Com>> {
        self.stages
            .iter()
            .map(|stage| stage.consistency.clone())
            .collect()
    }

    fn committed_output_claims(&self) -> Vec<CommittedOutputClaims<Com>> {
        self.stages
            .iter()
            .map(|stage| stage.output_claim_rows.commitments.clone())
            .collect()
    }

    fn build_constraints(
        &self,
        publics: &[(P, F)],
        challenges: &[(Ch, F)],
    ) -> Result<(ConstraintMatrices<F>, Layout), VerificationError<F>> {
        let mut r1cs = R1csBuilder::new();
        let layout = self.build_with_sources(&mut r1cs, publics, challenges)?;

        Ok((r1cs.into_matrices(), layout))
    }
}

impl Layout {
    fn dimensions<F: Field, Com>(
        &self,
        r1cs: &ConstraintMatrices<F>,
        sumcheck_consistency: &[CommittedSumcheckConsistency<F, Com>],
        output_claims: &[CommittedOutputClaims<Com>],
    ) -> Result<BlindFoldDimensions, RelaxedError> {
        let total_rounds = checked_sum(
            "total rounds",
            sumcheck_consistency
                .iter()
                .map(|consistency| consistency.rounds.len()),
        )?;
        let witness_row_len = self.witness_row_len;
        let coefficient_values =
            total_rounds
                .checked_mul(witness_row_len)
                .ok_or(RelaxedError::DimensionOverflow {
                    name: "coefficient values",
                    value: total_rounds,
                })?;
        let coefficient_rows = total_rounds;

        let output_claim_rows = checked_sum(
            "output claim rows",
            output_claims.iter().map(|claims| claims.commitments.len()),
        )?;
        let output_claim_values = output_claim_rows.checked_mul(witness_row_len).ok_or(
            RelaxedError::DimensionOverflow {
                name: "output claim values",
                value: output_claim_rows,
            },
        )?;

        let r1cs_witness_values = r1cs.num_vars.saturating_sub(1);
        let used_values = checked_sum(
            "committed witness values",
            [coefficient_values, output_claim_values],
        )?;
        let auxiliary_values = r1cs_witness_values.checked_sub(used_values).ok_or(
            RelaxedError::InconsistentDimensions {
                name: "auxiliary values",
                total: r1cs_witness_values,
                used: used_values,
            },
        )?;
        let auxiliary_rows = auxiliary_values.div_ceil(witness_row_len);
        let occupied_witness_rows = checked_sum(
            "occupied witness rows",
            [coefficient_rows, output_claim_rows, auxiliary_rows],
        )?;
        let witness_row_count =
            checked_next_power_of_two("witness rows", occupied_witness_rows.max(1))?;
        let witness_rows = WitnessRowLayout::from_counts(
            coefficient_rows,
            output_claim_rows,
            auxiliary_rows,
            witness_row_count,
        )?;

        let padded_constraints =
            checked_next_power_of_two("error values", r1cs.num_constraints.max(1))?;
        let error_row_len = witness_row_len.min(padded_constraints);
        let error_row_count = padded_constraints / error_row_len;

        Ok(BlindFoldDimensions {
            witness: RowDimensions {
                row_len: witness_row_len,
                row_count: witness_row_count,
            },
            error: RowDimensions {
                row_len: error_row_len,
                row_count: error_row_count,
            },
            witness_rows,
            coefficient_rows,
            output_claim_rows,
            auxiliary_rows,
            coefficient_values,
            auxiliary_values,
        })
    }
}

impl WitnessRowLayout {
    fn from_counts(
        coefficient_rows: usize,
        output_claim_rows: usize,
        auxiliary_rows: usize,
        witness_row_count: usize,
    ) -> Result<Self, RelaxedError> {
        let output_claim_start = coefficient_rows;
        let auxiliary_start =
            checked_sum("witness row layout", [coefficient_rows, output_claim_rows])?;
        let padding_start = checked_sum(
            "witness row layout",
            [coefficient_rows, auxiliary_rows, output_claim_rows],
        )?;
        if padding_start > witness_row_count {
            return Err(RelaxedError::InconsistentDimensions {
                name: "witness row layout",
                total: witness_row_count,
                used: padding_start,
            });
        }

        Ok(Self {
            coefficients: 0..coefficient_rows,
            output_claims: output_claim_start..auxiliary_start,
            auxiliary: auxiliary_start..padding_start,
            padding: padding_start..witness_row_count,
        })
    }
}

fn pad_rows<F, Com>(
    rows: &mut Vec<Com>,
    target_len: usize,
    name: &'static str,
) -> Result<(), RelaxedError>
where
    F: Field,
    Com: Clone + HomomorphicCommitment<F>,
{
    if rows.len() > target_len {
        return Err(RelaxedError::InconsistentDimensions {
            name,
            total: target_len,
            used: rows.len(),
        });
    }
    rows.resize_with(target_len, Com::default);
    Ok(())
}

fn ensure_len(name: &'static str, expected: usize, actual: usize) -> Result<(), RelaxedError> {
    if expected != actual {
        return Err(RelaxedError::LengthMismatch {
            name,
            expected,
            actual,
        });
    }
    Ok(())
}

fn checked_next_power_of_two(name: &'static str, value: usize) -> Result<usize, RelaxedError> {
    value
        .checked_next_power_of_two()
        .ok_or(RelaxedError::DimensionOverflow { name, value })
}

fn checked_sum(
    name: &'static str,
    values: impl IntoIterator<Item = usize>,
) -> Result<usize, RelaxedError> {
    values.into_iter().try_fold(0usize, |sum, value| {
        sum.checked_add(value)
            .ok_or(RelaxedError::DimensionOverflow { name, value: sum })
    })
}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "tests may panic on assertion failures")]
mod tests {
    use super::*;
    use crate::{
        BlindFoldStage, BlindFoldStatement, CommittedClaimRows, FinalOpeningBinding, OpeningAlias,
    };
    use jolt_claims::{constant, opening, Expr};
    use jolt_crypto::{Bn254, Bn254G1, JoltGroup, Pedersen, PedersenSetup, VectorCommitment};
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_sumcheck::{
        CommittedOutputClaims, CommittedRound, CommittedSumcheckProof, SumcheckDomainSpec,
        SumcheckError, SumcheckStatement,
    };
    use jolt_transcript::{AppendToTranscript, Blake2bTranscript, Transcript};

    #[derive(Clone, Debug)]
    struct TestStage {
        name: String,
        statement: SumcheckStatement,
        input_claim: Expr<Fr, usize>,
        output_claim: Expr<Fr, usize>,
    }

    fn stage(num_vars: usize, degree: usize) -> TestStage {
        let claim: Expr<Fr, usize> = constant(Fr::from_u64(0));
        TestStage {
            name: "stage".to_string(),
            statement: SumcheckStatement::new(num_vars, degree),
            input_claim: claim.clone(),
            output_claim: claim,
        }
    }

    fn proof(rounds: &[(u64, usize)], output_claims: &[u64]) -> CommittedSumcheckProof<Fr> {
        CommittedSumcheckProof {
            rounds: rounds
                .iter()
                .map(|&(commitment, degree)| CommittedRound {
                    commitment: Fr::from_u64(commitment),
                    degree,
                })
                .collect(),
            output_claims: CommittedOutputClaims {
                commitments: output_claims
                    .iter()
                    .map(|&commitment| Fr::from_u64(commitment))
                    .collect(),
            },
        }
    }

    fn commitment_proof(
        setup: &PedersenSetup<Bn254G1>,
        rounds: &[(u64, usize)],
        output_claims: &[u64],
    ) -> CommittedSumcheckProof<Bn254G1> {
        CommittedSumcheckProof {
            rounds: rounds
                .iter()
                .map(|&(commitment, degree)| CommittedRound {
                    commitment: pedersen_commitment(setup, commitment),
                    degree,
                })
                .collect(),
            output_claims: CommittedOutputClaims {
                commitments: output_claims
                    .iter()
                    .map(|&commitment| pedersen_commitment(setup, commitment))
                    .collect(),
            },
        }
    }

    fn pedersen_setup() -> PedersenSetup<Bn254G1> {
        let generator = Bn254::g1_generator();
        let message_generators = (1..=4)
            .map(|i| generator.scalar_mul(&Fr::from_u64(i)))
            .collect();
        PedersenSetup::new(message_generators, generator.scalar_mul(&Fr::from_u64(99)))
    }

    fn pedersen_commitment(setup: &PedersenSetup<Bn254G1>, value: u64) -> Bn254G1 {
        Pedersen::<Bn254G1>::commit(setup, &[Fr::from_u64(value)], &Fr::from_u64(value + 1000))
    }

    fn try_statement_from_proofs<Com>(
        stages: &[TestStage],
        proofs: &[CommittedSumcheckProof<Com>],
        final_openings: Vec<FinalOpeningBinding<Fr, usize, Com>>,
    ) -> Result<BlindFoldStatement<Fr, usize, Com>, VerificationError<Fr>>
    where
        Com: Clone + AppendToTranscript,
    {
        if stages.len() != proofs.len() {
            return Err(VerificationError::StageCountMismatch {
                claim_stages: stages.len(),
                proof_stages: proofs.len(),
            });
        }

        let mut transcript = Blake2bTranscript::<Fr>::new(b"blindfold");
        let mut next_opening_id = 0usize;
        let stages = stages
            .iter()
            .zip(proofs)
            .enumerate()
            .map(|(stage_index, (stage, proof))| {
                let consistency = proof
                    .verify_committed_consistency(stage.statement, &mut transcript)
                    .map_err(|source| VerificationError::Sumcheck {
                        stage_index,
                        source,
                    })?;
                let row_len = stage.statement.degree + 1;
                let output_opening_count = proof.output_claims.commitments.len() * row_len;
                let opening_ids =
                    (next_opening_id..next_opening_id + output_opening_count).collect();
                next_opening_id += output_opening_count;
                Ok::<BlindFoldStage<Fr, usize, Com>, VerificationError<Fr>>(BlindFoldStage::new(
                    stage.name.clone(),
                    stage.statement,
                    SumcheckDomainSpec::BooleanHypercube,
                    consistency,
                    CommittedClaimRows::new(opening_ids, row_len, proof.output_claims.clone()),
                    stage.input_claim.clone(),
                    stage.output_claim.clone(),
                ))
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(BlindFoldStatement::new(stages, final_openings))
    }

    fn protocol_from_proofs<Com>(
        stages: &[TestStage],
        proofs: &[CommittedSumcheckProof<Com>],
        final_openings: Vec<FinalOpeningBinding<Fr, usize, Com>>,
    ) -> BlindFoldProtocol<Fr, Com>
    where
        Com: Clone + AppendToTranscript,
    {
        let statement = try_statement_from_proofs(stages, proofs, final_openings)
            .expect("statement builds from committed proofs");
        protocol_from_statement(&statement)
    }

    fn protocol_from_statement<Com>(
        statement: &BlindFoldStatement<Fr, usize, Com>,
    ) -> BlindFoldProtocol<Fr, Com>
    where
        Com: Clone,
    {
        let mut builder = BlindFoldProtocol::<Fr, Com>::builder::<usize, (), usize>();
        for stage in &statement.stages {
            builder = builder
                .stage(stage.name.clone())
                .sumcheck(stage.statement)
                .domain(stage.domain)
                .consistency(stage.consistency.clone())
                .output_claim_rows(
                    stage.output_claim_rows.opening_ids.clone(),
                    stage.output_claim_rows.row_len,
                    stage.output_claim_rows.commitments.clone(),
                )
                .input_claim(stage.input_claim.clone())
                .output_claim(stage.output_claim.clone())
                .finish_stage()
                .expect("test stage statement is complete");
        }
        for binding in &statement.final_openings {
            builder = builder.final_opening(
                binding.opening_ids.clone(),
                binding.coefficients.clone(),
                binding.evaluation_commitment.clone(),
            );
        }
        builder.build().expect("BlindFold protocol builds")
    }

    #[test]
    fn verifies_committed_stages_in_claim_order() {
        let stages = vec![stage(2, 2), stage(1, 1)];
        let proofs = vec![
            proof(&[(11, 1), (12, 2)], &[21]),
            proof(&[(13, 1)], &[34, 55]),
        ];

        let verified =
            try_statement_from_proofs(&stages, &proofs, Vec::new()).expect("proofs verify");

        assert_eq!(verified.stage_count(), 2);
        assert_eq!(verified.stages[0].consistency.rounds.len(), 2);
        assert_eq!(verified.stages[1].consistency.rounds.len(), 1);
        let round_commitments = verified
            .stages
            .iter()
            .flat_map(|stage| {
                stage
                    .consistency
                    .rounds
                    .iter()
                    .map(|round| round.commitment)
            })
            .collect::<Vec<_>>();
        let output_claim_commitments = verified
            .stages
            .iter()
            .flat_map(|stage| {
                stage
                    .output_claim_rows
                    .commitments
                    .commitments
                    .iter()
                    .copied()
            })
            .collect::<Vec<_>>();
        assert_eq!(
            round_commitments,
            vec![Fr::from_u64(11), Fr::from_u64(12), Fr::from_u64(13)]
        );
        assert_eq!(
            output_claim_commitments,
            vec![Fr::from_u64(21), Fr::from_u64(34), Fr::from_u64(55)]
        );
    }

    #[test]
    fn rejects_stage_count_mismatch() {
        let stages = vec![stage(1, 1), stage(1, 1)];
        let proofs = vec![proof(&[(11, 1)], &[])];
        let error =
            try_statement_from_proofs(&stages, &proofs, Vec::new()).expect_err("counts differ");

        assert!(matches!(
            error,
            VerificationError::StageCountMismatch {
                claim_stages: 2,
                proof_stages: 1,
            }
        ));
    }

    #[test]
    fn reports_sumcheck_error_with_stage_index() {
        let stages = vec![stage(1, 1), stage(1, 1)];
        let proofs = vec![proof(&[(11, 1)], &[]), proof(&[(12, 2)], &[])];
        let error =
            try_statement_from_proofs(&stages, &proofs, Vec::new()).expect_err("degree fails");

        assert!(matches!(
            error,
            VerificationError::Sumcheck {
                stage_index: 1,
                source: SumcheckError::DegreeBoundExceeded { got: 2, max: 1 },
            }
        ));
    }

    #[test]
    fn blindfold_protocol_builder_constructs_protocol() {
        let stages = vec![stage(1, 1)];
        let proofs = vec![proof(&[(11, 1)], &[21])];
        let protocol = protocol_from_proofs(&stages, &proofs, Vec::new());

        assert_eq!(protocol.sumcheck_consistency.len(), 1);
        assert_eq!(protocol.committed_output_claims.len(), 1);
        assert_eq!(protocol.layout.stage_count(), 1);
        assert!(protocol.eval_commitments.is_empty());
        assert!(protocol.r1cs.num_vars > 1);
        assert_eq!(
            protocol.dimensions,
            BlindFoldDimensions {
                witness: RowDimensions {
                    row_len: 2,
                    row_count: 4,
                },
                error: RowDimensions {
                    row_len: 2,
                    row_count: 2,
                },
                witness_rows: WitnessRowLayout {
                    coefficients: 0..1,
                    output_claims: 1..2,
                    auxiliary: 2..3,
                    padding: 3..4,
                },
                coefficient_rows: 1,
                output_claim_rows: 1,
                auxiliary_rows: 1,
                coefficient_values: 2,
                auxiliary_values: 2,
            }
        );
    }

    #[test]
    fn blindfold_protocol_builder_resolves_output_claim_aliases() {
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        enum Opening {
            Source,
            Alias,
        }

        let statement = SumcheckStatement::new(1, 1);
        let proof = proof(&[(11, 1)], &[21]);
        let mut transcript = Blake2bTranscript::<Fr>::new(b"blindfold-alias");
        let consistency = proof
            .verify_committed_consistency(statement, &mut transcript)
            .expect("committed proof is consistent");
        let protocol = BlindFoldProtocol::<Fr, Fr>::builder::<Opening, (), usize>()
            .stage("alias")
            .sumcheck(statement)
            .domain(SumcheckDomainSpec::BooleanHypercube)
            .consistency(consistency)
            .output_claim_rows(vec![Opening::Source], 1, proof.output_claims.clone())
            .output_claim_aliases([OpeningAlias::new(Opening::Alias, Opening::Source)])
            .input_claim(constant(Fr::from_u64(0)))
            .output_claim(opening(Opening::Alias))
            .finish_stage()
            .expect("stage is complete")
            .build()
            .expect("alias resolves to a committed output claim");

        assert_eq!(protocol.dimensions.output_claim_rows, 1);
    }

    #[test]
    fn blindfold_protocol_builder_rejects_missing_output_claim_alias_source() {
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        enum Opening {
            Source,
            Alias,
            Other,
        }

        let statement = SumcheckStatement::new(1, 1);
        let proof = proof(&[(11, 1)], &[21]);
        let mut transcript = Blake2bTranscript::<Fr>::new(b"blindfold-alias");
        let consistency = proof
            .verify_committed_consistency(statement, &mut transcript)
            .expect("committed proof is consistent");
        let error = BlindFoldProtocol::<Fr, Fr>::builder::<Opening, (), usize>()
            .stage("alias")
            .sumcheck(statement)
            .domain(SumcheckDomainSpec::BooleanHypercube)
            .consistency(consistency)
            .output_claim_rows(vec![Opening::Other], 1, proof.output_claims.clone())
            .output_claim_aliases([OpeningAlias::new(Opening::Alias, Opening::Source)])
            .input_claim(constant(Fr::from_u64(0)))
            .output_claim(opening(Opening::Alias))
            .finish_stage()
            .expect("stage is complete")
            .build()
            .expect_err("alias source is absent");

        assert!(matches!(
            error,
            VerificationError::R1cs(Error::MissingOpeningAliasSource)
        ));
    }

    #[test]
    fn rejects_malformed_final_opening_binding() {
        let stages = vec![stage(1, 1)];
        let proofs = vec![proof(&[(11, 1)], &[21])];
        let statement = try_statement_from_proofs(
            &stages,
            &proofs,
            vec![FinalOpeningBinding::new(
                vec![0],
                Vec::new(),
                Fr::from_u64(99),
            )],
        )
        .expect("statement construction verifies committed proof");

        let error = BlindFoldProtocol::<Fr, Fr>::builder::<usize, (), usize>()
            .stage("stage")
            .sumcheck(stages[0].statement)
            .domain(SumcheckDomainSpec::BooleanHypercube)
            .consistency(statement.stages[0].consistency.clone())
            .output_claim_rows(
                vec![0, 1],
                stages[0].statement.degree + 1,
                proofs[0].output_claims.clone(),
            )
            .input_claim(stages[0].input_claim.clone())
            .output_claim(stages[0].output_claim.clone())
            .finish_stage()
            .expect("stage is complete")
            .final_opening(vec![0], Vec::new(), Fr::from_u64(99))
            .build()
            .expect_err("final opening binding is malformed");

        assert!(matches!(
            error,
            VerificationError::Relaxed(RelaxedError::LengthMismatch {
                name: "final opening coefficients",
                expected: 1,
                actual: 0,
            })
        ));
    }

    #[test]
    fn rejects_empty_final_opening_binding() {
        let stages = vec![stage(1, 1)];
        let proofs = vec![proof(&[(11, 1)], &[21])];
        let error = BlindFoldProtocol::<Fr, Fr>::builder::<usize, (), usize>()
            .stage("stage")
            .sumcheck(stages[0].statement)
            .domain(SumcheckDomainSpec::BooleanHypercube)
            .consistency(
                try_statement_from_proofs(&stages, &proofs, Vec::new())
                    .expect("statement construction verifies committed proof")
                    .stages[0]
                    .consistency
                    .clone(),
            )
            .output_claim_rows(
                vec![0, 1],
                stages[0].statement.degree + 1,
                proofs[0].output_claims.clone(),
            )
            .input_claim(stages[0].input_claim.clone())
            .output_claim(stages[0].output_claim.clone())
            .finish_stage()
            .expect("stage is complete")
            .final_opening(Vec::new(), Vec::new(), Fr::from_u64(99))
            .build()
            .expect_err("empty final opening binding is malformed");

        assert!(matches!(
            error,
            VerificationError::R1cs(Error::EmptyFinalOpeningBinding)
        ));
    }

    #[test]
    fn rejects_extra_output_claim_rows_without_typed_openings() {
        let statement = SumcheckStatement::new(1, 1);
        let proof = proof(&[(11, 1)], &[21, 22]);
        let mut transcript = Blake2bTranscript::<Fr>::new(b"blindfold-output-row-count");
        let consistency = proof
            .verify_committed_consistency(statement, &mut transcript)
            .expect("committed proof is consistent");
        let error = BlindFoldProtocol::<Fr, Fr>::builder::<usize, (), usize>()
            .stage("row-count")
            .sumcheck(statement)
            .domain(SumcheckDomainSpec::BooleanHypercube)
            .consistency(consistency)
            .output_claim_rows(vec![0, 1], 2, proof.output_claims.clone())
            .input_claim(constant(Fr::from_u64(0)))
            .output_claim(constant(Fr::from_u64(0)))
            .finish_stage()
            .expect("stage is complete")
            .build()
            .expect_err("anonymous extra output row is rejected");

        assert!(matches!(
            error,
            VerificationError::R1cs(Error::CommittedRowCountMismatch {
                name: "output claim rows",
                expected: 1,
                actual: 2,
            })
        ));
    }

    #[test]
    fn dimensions_account_for_multiple_stages_and_padding() {
        let stages = vec![stage(2, 2), stage(1, 1)];
        let proofs = vec![
            proof(&[(11, 1), (12, 2)], &[21, 22, 23]),
            proof(&[(13, 1)], &[34, 55]),
        ];
        let protocol = protocol_from_proofs(&stages, &proofs, Vec::new());

        assert_eq!(
            protocol.dimensions,
            BlindFoldDimensions {
                witness: RowDimensions {
                    row_len: 4,
                    row_count: 16,
                },
                error: RowDimensions {
                    row_len: 4,
                    row_count: 4,
                },
                witness_rows: WitnessRowLayout {
                    coefficients: 0..3,
                    output_claims: 3..8,
                    auxiliary: 8..10,
                    padding: 10..16,
                },
                coefficient_rows: 3,
                output_claim_rows: 5,
                auxiliary_rows: 2,
                coefficient_values: 12,
                auxiliary_values: 5,
            }
        );
    }

    #[test]
    fn committed_relaxed_instance_assembles_witness_rows_in_layout_order() {
        let setup = pedersen_setup();
        let stages = vec![stage(2, 2), stage(1, 1)];
        let proofs = vec![
            commitment_proof(&setup, &[(11, 1), (12, 2)], &[21, 22]),
            commitment_proof(&setup, &[(13, 1)], &[34]),
        ];
        let protocol = protocol_from_proofs(&stages, &proofs, Vec::new());

        let auxiliary_rows = vec![
            pedersen_commitment(&setup, 41),
            pedersen_commitment(&setup, 42),
        ];
        let relaxed = protocol
            .committed_relaxed_instance(&auxiliary_rows)
            .expect("relaxed instance builds");

        assert_eq!(relaxed.u, Fr::from_u64(1));
        assert_eq!(
            protocol.dimensions.witness_rows,
            WitnessRowLayout {
                coefficients: 0..3,
                output_claims: 3..6,
                auxiliary: 6..8,
                padding: 8..8,
            }
        );
        let identity = <Bn254G1 as JoltGroup>::identity();
        assert_eq!(
            relaxed.witness_row_commitments,
            vec![
                pedersen_commitment(&setup, 11),
                pedersen_commitment(&setup, 12),
                pedersen_commitment(&setup, 13),
                pedersen_commitment(&setup, 21),
                pedersen_commitment(&setup, 22),
                pedersen_commitment(&setup, 34),
                pedersen_commitment(&setup, 41),
                pedersen_commitment(&setup, 42),
            ]
        );
        assert_eq!(
            relaxed.error_row_commitments,
            vec![identity; protocol.dimensions.error.row_count]
        );
        assert!(relaxed.eval_commitments.is_empty());
    }

    #[test]
    fn committed_relaxed_instance_rejects_auxiliary_row_count_mismatch() {
        let setup = pedersen_setup();
        let stages = vec![stage(1, 1)];
        let proofs = vec![commitment_proof(&setup, &[(11, 1)], &[21])];
        let protocol = protocol_from_proofs(&stages, &proofs, Vec::new());

        let error = protocol
            .committed_relaxed_instance(&[])
            .expect_err("auxiliary row is missing");

        assert_eq!(
            error,
            RelaxedError::LengthMismatch {
                name: "auxiliary row commitments",
                expected: 1,
                actual: 0,
            }
        );
    }

    #[test]
    fn random_relaxed_instance_accepts_exact_dimensions() {
        let setup = pedersen_setup();
        let stages = vec![stage(1, 1)];
        let proofs = vec![commitment_proof(&setup, &[(11, 1)], &[21])];
        let protocol = protocol_from_proofs(&stages, &proofs, Vec::new());

        let round_rows = vec![pedersen_commitment(&setup, 7); protocol.dimensions.coefficient_rows];
        let output_claim_rows =
            vec![pedersen_commitment(&setup, 71); protocol.dimensions.output_claim_rows];
        let auxiliary_rows =
            vec![pedersen_commitment(&setup, 72); protocol.dimensions.auxiliary_rows];
        let error_rows = vec![pedersen_commitment(&setup, 8); protocol.dimensions.error.row_count];
        let random_eval_commitments = Vec::new();
        let random = protocol
            .random_relaxed_instance(
                &round_rows,
                &output_claim_rows,
                &auxiliary_rows,
                &error_rows,
                &random_eval_commitments,
                Fr::from_u64(3),
            )
            .expect("random instance dimensions match");

        assert_eq!(random.u, Fr::from_u64(3));
        let identity = <Bn254G1 as JoltGroup>::identity();
        let mut expected_witness_rows = Vec::new();
        expected_witness_rows.extend_from_slice(&round_rows);
        expected_witness_rows.resize(protocol.dimensions.witness_rows.coefficients.end, identity);
        expected_witness_rows.extend_from_slice(&output_claim_rows);
        expected_witness_rows.resize(protocol.dimensions.witness_rows.output_claims.end, identity);
        expected_witness_rows.extend_from_slice(&auxiliary_rows);
        expected_witness_rows.resize(protocol.dimensions.witness_rows.auxiliary.end, identity);
        expected_witness_rows.resize(protocol.dimensions.witness.row_count, identity);
        assert_eq!(random.witness_row_commitments, expected_witness_rows);
        assert_eq!(random.error_row_commitments, error_rows);
        assert_eq!(random.eval_commitments, random_eval_commitments);
    }

    #[test]
    fn random_relaxed_instance_rejects_round_row_count_mismatch() {
        let setup = pedersen_setup();
        let protocol = one_stage_protocol(&setup);
        let round_rows = vec![
            pedersen_commitment(&setup, 7);
            protocol.dimensions.coefficient_rows.saturating_sub(1)
        ];
        let output_claim_rows =
            vec![pedersen_commitment(&setup, 71); protocol.dimensions.output_claim_rows];
        let auxiliary_rows =
            vec![pedersen_commitment(&setup, 72); protocol.dimensions.auxiliary_rows];
        let error_rows = vec![pedersen_commitment(&setup, 8); protocol.dimensions.error.row_count];

        let error = protocol
            .random_relaxed_instance(
                &round_rows,
                &output_claim_rows,
                &auxiliary_rows,
                &error_rows,
                &[],
                Fr::from_u64(3),
            )
            .expect_err("witness row count differs");

        assert_eq!(
            error,
            RelaxedError::LengthMismatch {
                name: "random round commitments",
                expected: protocol.dimensions.coefficient_rows,
                actual: protocol.dimensions.coefficient_rows - 1,
            }
        );
    }

    #[test]
    fn random_relaxed_instance_rejects_error_row_count_mismatch() {
        let setup = pedersen_setup();
        let protocol = one_stage_protocol(&setup);
        let round_rows = vec![pedersen_commitment(&setup, 7); protocol.dimensions.coefficient_rows];
        let output_claim_rows =
            vec![pedersen_commitment(&setup, 71); protocol.dimensions.output_claim_rows];
        let auxiliary_rows =
            vec![pedersen_commitment(&setup, 72); protocol.dimensions.auxiliary_rows];
        let error_rows = vec![
            pedersen_commitment(&setup, 8);
            protocol.dimensions.error.row_count.saturating_sub(1)
        ];

        let error = protocol
            .random_relaxed_instance(
                &round_rows,
                &output_claim_rows,
                &auxiliary_rows,
                &error_rows,
                &[],
                Fr::from_u64(3),
            )
            .expect_err("error row count differs");

        assert_eq!(
            error,
            RelaxedError::LengthMismatch {
                name: "random error row commitments",
                expected: protocol.dimensions.error.row_count,
                actual: protocol.dimensions.error.row_count - 1,
            }
        );
    }

    #[test]
    fn random_relaxed_instance_rejects_eval_commitment_count_mismatch() {
        let setup = pedersen_setup();
        let protocol = one_stage_protocol(&setup);
        let round_rows = vec![pedersen_commitment(&setup, 7); protocol.dimensions.coefficient_rows];
        let output_claim_rows =
            vec![pedersen_commitment(&setup, 71); protocol.dimensions.output_claim_rows];
        let auxiliary_rows =
            vec![pedersen_commitment(&setup, 72); protocol.dimensions.auxiliary_rows];
        let error_rows = vec![pedersen_commitment(&setup, 8); protocol.dimensions.error.row_count];

        let error = protocol
            .random_relaxed_instance(
                &round_rows,
                &output_claim_rows,
                &auxiliary_rows,
                &error_rows,
                &[pedersen_commitment(&setup, 100)],
                Fr::from_u64(3),
            )
            .expect_err("eval commitment count differs");

        assert_eq!(
            error,
            RelaxedError::LengthMismatch {
                name: "random eval commitments",
                expected: 0,
                actual: 1,
            }
        );
    }

    #[test]
    fn validate_cross_term_error_rows_accepts_exact_count() {
        let setup = pedersen_setup();
        let protocol = one_stage_protocol(&setup);
        let cross_term_rows =
            vec![pedersen_commitment(&setup, 9); protocol.dimensions.error.row_count];

        protocol
            .validate_cross_term_error_rows(&cross_term_rows)
            .expect("cross-term row count matches");
    }

    #[test]
    fn validate_cross_term_error_rows_rejects_count_mismatch() {
        let setup = pedersen_setup();
        let protocol = one_stage_protocol(&setup);
        let cross_term_rows = vec![
            pedersen_commitment(&setup, 9);
            protocol.dimensions.error.row_count.saturating_sub(1)
        ];

        let error = protocol
            .validate_cross_term_error_rows(&cross_term_rows)
            .expect_err("cross-term row count differs");

        assert_eq!(
            error,
            RelaxedError::LengthMismatch {
                name: "cross-term error row commitments",
                expected: protocol.dimensions.error.row_count,
                actual: protocol.dimensions.error.row_count - 1,
            }
        );
    }

    fn one_stage_protocol(setup: &PedersenSetup<Bn254G1>) -> BlindFoldProtocol<Fr, Bn254G1> {
        let stages = vec![stage(1, 1)];
        let proofs = vec![commitment_proof(setup, &[(11, 1)], &[21])];

        protocol_from_proofs(&stages, &proofs, Vec::new())
    }
}
