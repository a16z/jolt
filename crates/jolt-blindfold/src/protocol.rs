use std::ops::Range;

use jolt_crypto::HomomorphicCommitment;
use jolt_field::Field;
use jolt_r1cs::{ClaimSources, ConstraintMatrices, R1csBuilder};
use jolt_sumcheck::{CommittedOutputClaims, CommittedSumcheckProof, SumcheckScalar};
use jolt_transcript::{AppendToTranscript, Transcript};

use crate::{
    r1cs, Inputs, InstanceClaims, Layout, RelaxedError, RelaxedInstance, StageInput,
    VerificationError,
};

#[derive(Clone, Debug)]
pub struct BlindFoldProtocol<F: Field, C> {
    pub sumcheck_inputs: Inputs<F, C>,
    pub output_claims: Vec<CommittedOutputClaims<C>>,
    pub r1cs: ConstraintMatrices<F>,
    pub layout: Layout,
    pub dimensions: BlindFoldDimensions,
    pub eval_commitments: Vec<C>,
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

impl<F, C> BlindFoldProtocol<F, C>
where
    F: Field + SumcheckScalar,
    C: Clone + AppendToTranscript,
{
    pub fn from_committed_proofs<T, R>(
        claims: &InstanceClaims<F, R::Opening, R::Public, R::Challenge>,
        sumcheck_proofs: &[CommittedSumcheckProof<C>],
        eval_commitments: &[C],
        transcript: &mut T,
        claim_sources: &mut R,
    ) -> Result<Self, VerificationError<F>>
    where
        T: Transcript<Challenge = F>,
        R: ClaimSources<F>,
    {
        let committed = verify_committed_stages(claims, sumcheck_proofs, transcript)?;
        let (r1cs, layout) = build_constraints(claims, &committed.inputs, claim_sources)?;
        let dimensions = compute_dimensions(&r1cs, &committed.inputs, &committed.output_claims)?;

        Ok(Self {
            sumcheck_inputs: committed.inputs,
            output_claims: committed.output_claims,
            r1cs,
            layout,
            dimensions,
            eval_commitments: eval_commitments.to_vec(),
        })
    }
}

impl<F, C> BlindFoldProtocol<F, C>
where
    F: Field,
    C: Clone + HomomorphicCommitment<F>,
{
    pub fn committed_relaxed_instance(
        &self,
        auxiliary_row_commitments: &[C],
    ) -> Result<RelaxedInstance<F, C>, RelaxedError> {
        if auxiliary_row_commitments.len() != self.dimensions.auxiliary_rows {
            return Err(RelaxedError::LengthMismatch {
                name: "auxiliary row commitments",
                expected: self.dimensions.auxiliary_rows,
                actual: auxiliary_row_commitments.len(),
            });
        }

        let mut witness_row_commitments = Vec::with_capacity(self.dimensions.witness.row_count);

        witness_row_commitments.extend(
            self.sumcheck_inputs
                .stages
                .iter()
                .flat_map(|stage| stage.check.rounds.iter())
                .map(|round| round.commitment.clone()),
        );
        pad_rows::<F, C>(
            &mut witness_row_commitments,
            self.dimensions.witness_rows.coefficients.end,
            "coefficient row commitments",
        )?;

        witness_row_commitments.extend_from_slice(auxiliary_row_commitments);
        pad_rows::<F, C>(
            &mut witness_row_commitments,
            self.dimensions.witness_rows.auxiliary.end,
            "auxiliary row commitments",
        )?;

        witness_row_commitments.extend(
            self.output_claims
                .iter()
                .flat_map(|claims| claims.commitments.iter().cloned()),
        );
        pad_rows::<F, C>(
            &mut witness_row_commitments,
            self.dimensions.witness_rows.output_claims.end,
            "output claim row commitments",
        )?;
        pad_rows::<F, C>(
            &mut witness_row_commitments,
            self.dimensions.witness.row_count,
            "witness row commitments",
        )?;

        Ok(RelaxedInstance::new(
            F::one(),
            witness_row_commitments,
            vec![C::identity(); self.dimensions.error.row_count],
            self.eval_commitments.clone(),
        ))
    }
}

impl<F, C> BlindFoldProtocol<F, C>
where
    F: Field,
{
    pub fn validate_cross_term_error_rows(
        &self,
        cross_term_error_row_commitments: &[C],
    ) -> Result<(), RelaxedError> {
        ensure_len(
            "cross-term error row commitments",
            self.dimensions.error.row_count,
            cross_term_error_row_commitments.len(),
        )
    }
}

impl<F, C> BlindFoldProtocol<F, C>
where
    F: Field,
    C: Clone,
{
    pub fn random_relaxed_instance(
        &self,
        witness_row_commitments: &[C],
        error_row_commitments: &[C],
        eval_commitments: &[C],
        u: F,
    ) -> Result<RelaxedInstance<F, C>, RelaxedError> {
        ensure_len(
            "random witness row commitments",
            self.dimensions.witness.row_count,
            witness_row_commitments.len(),
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

        Ok(RelaxedInstance::new(
            u,
            witness_row_commitments.to_vec(),
            error_row_commitments.to_vec(),
            eval_commitments.to_vec(),
        ))
    }
}

#[derive(Clone, Debug)]
struct CommittedStages<F, C> {
    inputs: Inputs<F, C>,
    output_claims: Vec<CommittedOutputClaims<C>>,
}

fn verify_committed_stages<F, O, P, Ch, C, T>(
    claims: &InstanceClaims<F, O, P, Ch>,
    proofs: &[CommittedSumcheckProof<C>],
    transcript: &mut T,
) -> Result<CommittedStages<F, C>, VerificationError<F>>
where
    F: SumcheckScalar,
    T: Transcript<Challenge = F>,
    C: Clone + AppendToTranscript,
{
    if claims.stages.len() != proofs.len() {
        return Err(VerificationError::StageCountMismatch {
            claim_stages: claims.stages.len(),
            proof_stages: proofs.len(),
        });
    }

    let mut inputs = Vec::with_capacity(claims.stages.len());
    let mut output_claims = Vec::with_capacity(claims.stages.len());
    for (stage_index, (stage, proof)) in claims.stages.iter().zip(proofs).enumerate() {
        let check = proof
            .verify_committed(stage.shape, transcript)
            .map_err(|source| VerificationError::Sumcheck {
                stage_index,
                source,
            })?;
        inputs.push(StageInput::new(check));
        output_claims.push(proof.output_claims.clone());
    }

    Ok(CommittedStages {
        inputs: Inputs::new(inputs),
        output_claims,
    })
}

fn build_constraints<F, O, P, Ch, C, R>(
    claims: &InstanceClaims<F, O, P, Ch>,
    inputs: &Inputs<F, C>,
    claim_sources: &mut R,
) -> Result<(ConstraintMatrices<F>, Layout), VerificationError<F>>
where
    F: Field,
    R: ClaimSources<F, Opening = O, Public = P, Challenge = Ch>,
{
    let mut r1cs = R1csBuilder::new();
    let layout = r1cs::build(&mut r1cs, claims, inputs, claim_sources)?;
    Ok((r1cs.into_matrices(), layout))
}

fn compute_dimensions<F: Field, C>(
    r1cs: &ConstraintMatrices<F>,
    inputs: &Inputs<F, C>,
    output_claims: &[CommittedOutputClaims<C>],
) -> Result<BlindFoldDimensions, RelaxedError> {
    let total_rounds = checked_sum(
        "total rounds",
        inputs.stages.iter().map(|stage| stage.check.rounds.len()),
    )?;
    let round_coefficients = inputs
        .stages
        .iter()
        .flat_map(|stage| &stage.check.rounds)
        .map(|round| {
            round
                .degree
                .checked_add(1)
                .ok_or(RelaxedError::DimensionOverflow {
                    name: "round coefficient count",
                    value: round.degree,
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let max_round_coefficients = round_coefficients.iter().copied().max().unwrap_or(1);
    let witness_row_len = checked_next_power_of_two("witness row length", max_round_coefficients)?;
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

    let r1cs_witness_values = r1cs.num_vars.saturating_sub(1);
    let auxiliary_values = r1cs_witness_values.checked_sub(coefficient_values).ok_or(
        RelaxedError::InconsistentDimensions {
            name: "auxiliary values",
            total: r1cs_witness_values,
            used: coefficient_values,
        },
    )?;
    let auxiliary_rows = auxiliary_values.div_ceil(witness_row_len);
    let occupied_witness_rows = checked_sum(
        "occupied witness rows",
        [coefficient_rows, output_claim_rows, auxiliary_rows],
    )?;
    let witness_row_count =
        checked_next_power_of_two("witness rows", occupied_witness_rows.max(1))?;
    let witness_rows = witness_row_layout(
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

fn witness_row_layout(
    coefficient_rows: usize,
    output_claim_rows: usize,
    auxiliary_rows: usize,
    witness_row_count: usize,
) -> Result<WitnessRowLayout, RelaxedError> {
    let auxiliary_start = coefficient_rows;
    let output_claim_start = checked_sum("witness row layout", [coefficient_rows, auxiliary_rows])?;
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

    Ok(WitnessRowLayout {
        coefficients: 0..coefficient_rows,
        auxiliary: auxiliary_start..output_claim_start,
        output_claims: output_claim_start..padding_start,
        padding: padding_start..witness_row_count,
    })
}

fn pad_rows<F, C>(
    rows: &mut Vec<C>,
    target_len: usize,
    name: &'static str,
) -> Result<(), RelaxedError>
where
    F: Field,
    C: Clone + HomomorphicCommitment<F>,
{
    if rows.len() > target_len {
        return Err(RelaxedError::InconsistentDimensions {
            name,
            total: target_len,
            used: rows.len(),
        });
    }
    rows.resize_with(target_len, C::identity);
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
    use crate::{InstanceClaims, StageClaims};
    use jolt_claims::{constant, Expr};
    use jolt_crypto::{
        Bn254, Bn254G1, HomomorphicCommitment, JoltGroup, Pedersen, PedersenSetup, VectorCommitment,
    };
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_r1cs::ClaimSourceTable;
    use jolt_sumcheck::{CommittedOutputClaims, CommittedRound, SumcheckError, SumcheckShape};
    use jolt_transcript::{Blake2bTranscript, Transcript};

    fn stage(num_vars: usize, degree: usize) -> StageClaims<Fr, ()> {
        let claim: Expr<Fr, ()> = constant(Fr::from_u64(0));
        StageClaims::new(
            "stage",
            SumcheckShape::new(num_vars, degree),
            claim.clone(),
            claim,
        )
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

    #[test]
    fn verifies_committed_stages_in_claim_order() {
        let claims = InstanceClaims::new(vec![stage(2, 2), stage(1, 1)]);
        let proofs = vec![
            proof(&[(11, 1), (12, 2)], &[21]),
            proof(&[(13, 1)], &[34, 55]),
        ];

        let mut manual = Blake2bTranscript::<Fr>::new(b"blindfold");
        for (stage, proof) in claims.stages.iter().zip(&proofs) {
            let _ = proof
                .verify_committed(stage.shape, &mut manual)
                .expect("manual replay succeeds");
        }

        let mut transcript = Blake2bTranscript::<Fr>::new(b"blindfold");
        let verified =
            verify_committed_stages(&claims, &proofs, &mut transcript).expect("proofs verify");

        assert_eq!(verified.inputs.stage_count(), 2);
        assert_eq!(verified.inputs.stages[0].check.rounds.len(), 2);
        assert_eq!(verified.inputs.stages[1].check.rounds.len(), 1);
        let round_commitments = verified
            .inputs
            .stages
            .iter()
            .flat_map(|input| input.check.rounds.iter().map(|round| round.commitment))
            .collect::<Vec<_>>();
        let output_claim_commitments = verified
            .output_claims
            .iter()
            .flat_map(|output_claims| output_claims.commitments.iter().copied())
            .collect::<Vec<_>>();
        assert_eq!(
            round_commitments,
            vec![Fr::from_u64(11), Fr::from_u64(12), Fr::from_u64(13)]
        );
        assert_eq!(
            output_claim_commitments,
            vec![Fr::from_u64(21), Fr::from_u64(34), Fr::from_u64(55)]
        );
        assert_eq!(transcript.state(), manual.state());
    }

    #[test]
    fn rejects_stage_count_mismatch() {
        let claims = InstanceClaims::new(vec![stage(1, 1), stage(1, 1)]);
        let proofs = vec![proof(&[(11, 1)], &[])];
        let mut transcript = Blake2bTranscript::<Fr>::new(b"blindfold");

        let error =
            verify_committed_stages(&claims, &proofs, &mut transcript).expect_err("counts differ");

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
        let claims = InstanceClaims::new(vec![stage(1, 1), stage(1, 1)]);
        let proofs = vec![proof(&[(11, 1)], &[]), proof(&[(12, 2)], &[])];
        let mut transcript = Blake2bTranscript::<Fr>::new(b"blindfold");

        let error =
            verify_committed_stages(&claims, &proofs, &mut transcript).expect_err("degree fails");

        assert!(matches!(
            error,
            VerificationError::Sumcheck {
                stage_index: 1,
                source: SumcheckError::DegreeBoundExceeded { got: 2, max: 1 },
            }
        ));
    }

    #[test]
    fn blindfold_protocol_constructs_from_committed_proofs() {
        let claims = InstanceClaims::new(vec![stage(1, 1)]);
        let proofs = vec![proof(&[(11, 1)], &[21])];
        let eval_commitments = vec![Fr::from_u64(99)];
        let mut transcript = Blake2bTranscript::<Fr>::new(b"blindfold");
        let mut claim_sources = ClaimSourceTable::<Fr, (), (), usize>::new();

        let protocol = BlindFoldProtocol::from_committed_proofs(
            &claims,
            &proofs,
            &eval_commitments,
            &mut transcript,
            &mut claim_sources,
        )
        .expect("BlindFold protocol builds");

        assert_eq!(protocol.sumcheck_inputs.stage_count(), 1);
        assert_eq!(protocol.output_claims.len(), 1);
        assert_eq!(protocol.layout.stage_count(), 1);
        assert_eq!(protocol.eval_commitments, eval_commitments);
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
                    auxiliary: 1..2,
                    output_claims: 2..3,
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
    fn dimensions_account_for_multiple_stages_and_padding() {
        let claims = InstanceClaims::new(vec![stage(2, 2), stage(1, 1)]);
        let proofs = vec![
            proof(&[(11, 1), (12, 2)], &[21, 22, 23]),
            proof(&[(13, 1)], &[34, 55]),
        ];
        let mut transcript = Blake2bTranscript::<Fr>::new(b"blindfold");
        let mut claim_sources = ClaimSourceTable::<Fr, (), (), usize>::new();

        let protocol = BlindFoldProtocol::from_committed_proofs(
            &claims,
            &proofs,
            &[],
            &mut transcript,
            &mut claim_sources,
        )
        .expect("BlindFold protocol builds");

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
                    auxiliary: 3..5,
                    output_claims: 5..10,
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
        let claims = InstanceClaims::new(vec![stage(2, 2), stage(1, 1)]);
        let proofs = vec![
            commitment_proof(&setup, &[(11, 1), (12, 2)], &[21, 22]),
            commitment_proof(&setup, &[(13, 1)], &[34]),
        ];
        let eval_commitments = vec![pedersen_commitment(&setup, 99)];
        let mut transcript = Blake2bTranscript::<Fr>::new(b"blindfold");
        let mut claim_sources = ClaimSourceTable::<Fr, (), (), usize>::new();

        let protocol = BlindFoldProtocol::from_committed_proofs(
            &claims,
            &proofs,
            &eval_commitments,
            &mut transcript,
            &mut claim_sources,
        )
        .expect("BlindFold protocol builds");

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
                auxiliary: 3..5,
                output_claims: 5..8,
                padding: 8..8,
            }
        );
        let identity = <Bn254G1 as HomomorphicCommitment<Fr>>::identity();
        assert_eq!(
            relaxed.witness_row_commitments,
            vec![
                pedersen_commitment(&setup, 11),
                pedersen_commitment(&setup, 12),
                pedersen_commitment(&setup, 13),
                pedersen_commitment(&setup, 41),
                pedersen_commitment(&setup, 42),
                pedersen_commitment(&setup, 21),
                pedersen_commitment(&setup, 22),
                pedersen_commitment(&setup, 34),
            ]
        );
        assert_eq!(
            relaxed.error_row_commitments,
            vec![identity; protocol.dimensions.error.row_count]
        );
        assert_eq!(relaxed.eval_commitments, eval_commitments);
    }

    #[test]
    fn committed_relaxed_instance_rejects_auxiliary_row_count_mismatch() {
        let setup = pedersen_setup();
        let claims = InstanceClaims::new(vec![stage(1, 1)]);
        let proofs = vec![commitment_proof(&setup, &[(11, 1)], &[21])];
        let mut transcript = Blake2bTranscript::<Fr>::new(b"blindfold");
        let mut claim_sources = ClaimSourceTable::<Fr, (), (), usize>::new();

        let protocol = BlindFoldProtocol::from_committed_proofs(
            &claims,
            &proofs,
            &[],
            &mut transcript,
            &mut claim_sources,
        )
        .expect("BlindFold protocol builds");

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
        let claims = InstanceClaims::new(vec![stage(1, 1)]);
        let proofs = vec![commitment_proof(&setup, &[(11, 1)], &[21])];
        let eval_commitments = vec![pedersen_commitment(&setup, 99)];
        let mut transcript = Blake2bTranscript::<Fr>::new(b"blindfold");
        let mut claim_sources = ClaimSourceTable::<Fr, (), (), usize>::new();

        let protocol = BlindFoldProtocol::from_committed_proofs(
            &claims,
            &proofs,
            &eval_commitments,
            &mut transcript,
            &mut claim_sources,
        )
        .expect("BlindFold protocol builds");

        let witness_rows =
            vec![pedersen_commitment(&setup, 7); protocol.dimensions.witness.row_count];
        let error_rows = vec![pedersen_commitment(&setup, 8); protocol.dimensions.error.row_count];
        let random_eval_commitments = vec![pedersen_commitment(&setup, 100)];
        let random = protocol
            .random_relaxed_instance(
                &witness_rows,
                &error_rows,
                &random_eval_commitments,
                Fr::from_u64(3),
            )
            .expect("random instance dimensions match");

        assert_eq!(random.u, Fr::from_u64(3));
        assert_eq!(random.witness_row_commitments, witness_rows);
        assert_eq!(random.error_row_commitments, error_rows);
        assert_eq!(random.eval_commitments, random_eval_commitments);
    }

    #[test]
    fn random_relaxed_instance_rejects_witness_row_count_mismatch() {
        let setup = pedersen_setup();
        let protocol = one_stage_protocol(&setup);
        let witness_rows = vec![
            pedersen_commitment(&setup, 7);
            protocol.dimensions.witness.row_count.saturating_sub(1)
        ];
        let error_rows = vec![pedersen_commitment(&setup, 8); protocol.dimensions.error.row_count];

        let error = protocol
            .random_relaxed_instance(&witness_rows, &error_rows, &[], Fr::from_u64(3))
            .expect_err("witness row count differs");

        assert_eq!(
            error,
            RelaxedError::LengthMismatch {
                name: "random witness row commitments",
                expected: protocol.dimensions.witness.row_count,
                actual: protocol.dimensions.witness.row_count - 1,
            }
        );
    }

    #[test]
    fn random_relaxed_instance_rejects_error_row_count_mismatch() {
        let setup = pedersen_setup();
        let protocol = one_stage_protocol(&setup);
        let witness_rows =
            vec![pedersen_commitment(&setup, 7); protocol.dimensions.witness.row_count];
        let error_rows = vec![
            pedersen_commitment(&setup, 8);
            protocol.dimensions.error.row_count.saturating_sub(1)
        ];

        let error = protocol
            .random_relaxed_instance(&witness_rows, &error_rows, &[], Fr::from_u64(3))
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
        let witness_rows =
            vec![pedersen_commitment(&setup, 7); protocol.dimensions.witness.row_count];
        let error_rows = vec![pedersen_commitment(&setup, 8); protocol.dimensions.error.row_count];

        let error = protocol
            .random_relaxed_instance(
                &witness_rows,
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
        let claims = InstanceClaims::new(vec![stage(1, 1)]);
        let proofs = vec![commitment_proof(setup, &[(11, 1)], &[21])];
        let mut transcript = Blake2bTranscript::<Fr>::new(b"blindfold");
        let mut claim_sources = ClaimSourceTable::<Fr, (), (), usize>::new();

        BlindFoldProtocol::from_committed_proofs(
            &claims,
            &proofs,
            &[],
            &mut transcript,
            &mut claim_sources,
        )
        .expect("BlindFold protocol builds")
    }
}
