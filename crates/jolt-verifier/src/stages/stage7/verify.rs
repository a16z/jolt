use jolt_claims::protocols::jolt::{
    formulas::{
        claim_reductions::{
            advice,
            bytecode::{self as bytecode_reduction, BytecodeOutputWeightInputs},
            hamming_weight, program_image,
        },
        dimensions::JoltFormulaDimensions,
    },
    AdviceClaimReductionLayout, BytecodeClaimReductionLayout, JoltAdviceKind,
    JoltCommittedPolynomial, JoltRelationClaims, JoltRelationId, PrecommittedReductionLayout,
    ProgramImageClaimReductionLayout,
};
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_openings::CommitmentScheme;
use jolt_sumcheck::{
    BatchedCommittedSumcheckConsistency, BatchedEvaluationClaim, BatchedSumcheckVerifier,
    SumcheckClaim, SumcheckStatement,
};
use jolt_transcript::Transcript;

use super::advice_address_phase::{
    AdviceAddressPhase, AdviceAddressPhaseInputClaims, AdviceAddressPhaseOutputClaims,
};
use super::committed_reduction_address_phase::{
    BytecodeReductionAddressPhase, BytecodeReductionAddressPhaseInputClaims,
    ProgramImageReductionAddressPhase, ProgramImageReductionAddressPhaseInputClaims,
};
use super::hamming_weight_claim_reduction::{
    HammingWeightClaimReduction, HammingWeightClaimReductionInputClaims,
};
use super::outputs::{
    PrecommittedFinalOpening, Stage7ClearOutput, Stage7Output, Stage7OutputClaims,
    Stage7PublicOutput, Stage7ZkOutput,
};
use crate::{
    proof::JoltProof,
    stages::{
        relations::{
            check_relation_boolean_hypercube, zip_openings, OpeningClaim, SumcheckInstance,
        },
        stage4::{Stage4ClearOutput, Stage4Output},
        stage6::{
            outputs::BytecodeCyclePhaseOutputClaims, Stage6ClearOutput, Stage6Output,
            Stage6ZkOutput,
        },
        zk::committed,
    },
    verifier::CheckedInputs,
    VerifierError,
};

pub fn verify<PCS, VC, T, ZkProof>(
    checked: &CheckedInputs,
    proof: &JoltProof<PCS, VC, ZkProof>,
    formula_dimensions: &JoltFormulaDimensions,
    transcript: &mut T,
    stage4: &Stage4Output<PCS::Field, VC::Output>,
    stage6: &Stage6Output<PCS::Field, VC::Output>,
) -> Result<Stage7Output<PCS::Field, VC::Output>, VerifierError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
    T: Transcript<Challenge = PCS::Field>,
{
    let hamming_dimensions = hamming_weight::HammingWeightClaimReductionDimensions::new(
        formula_dimensions.ra_layout,
        proof.one_hot_config.committed_chunk_bits(),
    );
    let hamming_claims = hamming_weight::claim_reduction::<PCS::Field>(hamming_dimensions);

    let layouts = Stage7Layouts {
        trusted_advice: checked.precommitted.trusted_advice.as_ref(),
        untrusted_advice: checked.precommitted.untrusted_advice.as_ref(),
        bytecode: checked.precommitted.bytecode.as_ref(),
        program_image: checked.precommitted.program_image.as_ref(),
    };
    let trusted_advice_claims = layouts.trusted_advice.and_then(|layout| {
        layout.dimensions().has_address_phase().then(|| {
            advice::address_phase::<PCS::Field>(JoltAdviceKind::Trusted, layout.dimensions())
        })
    });
    let untrusted_advice_claims = layouts.untrusted_advice.and_then(|layout| {
        layout.dimensions().has_address_phase().then(|| {
            advice::address_phase::<PCS::Field>(JoltAdviceKind::Untrusted, layout.dimensions())
        })
    });
    let bytecode_reduction_claims = layouts.bytecode.and_then(|layout| {
        layout.dimensions().has_address_phase().then(|| {
            bytecode_reduction::address_phase::<PCS::Field>(
                layout.dimensions(),
                layout.chunk_count(),
            )
        })
    });
    let program_image_reduction_claims = layouts.program_image.and_then(|layout| {
        layout
            .dimensions()
            .has_address_phase()
            .then(|| program_image::address_phase::<PCS::Field>(layout.dimensions()))
    });

    check_relation_boolean_hypercube(&hamming_claims)?;
    for claim in [
        &trusted_advice_claims,
        &untrusted_advice_claims,
        &bytecode_reduction_claims,
        &program_image_reduction_claims,
    ]
    .into_iter()
    .flatten()
    {
        check_relation_boolean_hypercube(claim)?;
    }

    let hamming_gamma = transcript.challenge_scalar();

    if checked.zk {
        let stage6 = stage6.zk()?;
        let mut statements = vec![SumcheckStatement::new(
            hamming_claims.sumcheck.rounds,
            hamming_claims.sumcheck.degree,
        )];
        for claim in [
            &trusted_advice_claims,
            &untrusted_advice_claims,
            &bytecode_reduction_claims,
            &program_image_reduction_claims,
        ]
        .into_iter()
        .flatten()
        {
            statements.push(SumcheckStatement::new(
                claim.sumcheck.rounds,
                claim.sumcheck.degree,
            ));
        }

        let batch_consistency = BatchedSumcheckVerifier::verify_committed_consistency(
            &statements,
            &proof.stages.stage7_sumcheck_proof,
            transcript,
        )
        .map_err(|error| VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::HammingWeightClaimReduction,
            reason: error.to_string(),
        })?;

        let output_openings = hamming_weight::claim_reduction_output_openings(hamming_dimensions);
        let committed_output_claims = output_openings.instruction_ra.len()
            + output_openings.bytecode_ra.len()
            + output_openings.ram_ra.len()
            + usize::from(trusted_advice_claims.is_some())
            + usize::from(untrusted_advice_claims.is_some())
            + bytecode_reduction_claims
                .as_ref()
                .and(layouts.bytecode)
                .map_or(0, BytecodeClaimReductionLayout::chunk_count)
            + usize::from(program_image_reduction_claims.is_some());
        let batch_output_claims =
            committed::verify_output_claim_commitments(committed::CommittedOutputClaimInputs {
                checked,
                proof: &proof.stages.stage7_sumcheck_proof,
                proof_label: "stage7_sumcheck_proof",
                output_claim_count: committed_output_claims,
                stage: JoltRelationId::HammingWeightClaimReduction,
            })?;

        let hamming_point = batch_consistency
            .try_instance_point(hamming_claims.sumcheck.rounds)
            .map_err(|error| VerifierError::StageClaimSumcheckFailed {
                stage: JoltRelationId::HammingWeightClaimReduction,
                reason: error.to_string(),
            })?;
        let booleanity_opening = stage6.output_points.booleanity_opening_point().ok_or(
            VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::HammingWeightClaimReduction,
                reason: "Stage 6 booleanity produced no opening point".to_string(),
            },
        )?;
        let booleanity_r_cycle = &booleanity_opening[hamming_dimensions.log_k_chunk..];
        let hamming_opening_point = hamming_dimensions
            .opening_point(&hamming_point, booleanity_r_cycle)
            .map_err(|error| VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::HammingWeightClaimReduction,
                reason: error.to_string(),
            })?;
        // The advice / committed-program address-phase opening points. Their
        // sumcheck points and `FinalScale`/`ChunkOutputWeight` publics are
        // recomputed by BlindFold from `batch_consistency`, so only the opening
        // points (needed for the precommitted finals stage 8 consumes) are
        // recovered here.
        let trusted_advice = if let (Some(layout), Some(claim)) =
            (layouts.trusted_advice, trusted_advice_claims.as_ref())
        {
            Some(advice_address_phase_opening_point(
                &batch_consistency,
                claim,
                layout,
                JoltAdviceKind::Trusted,
                stage6,
            )?)
        } else {
            None
        };
        let untrusted_advice = if let (Some(layout), Some(claim)) =
            (layouts.untrusted_advice, untrusted_advice_claims.as_ref())
        {
            Some(advice_address_phase_opening_point(
                &batch_consistency,
                claim,
                layout,
                JoltAdviceKind::Untrusted,
                stage6,
            )?)
        } else {
            None
        };
        let bytecode_address_phase = if let (Some(layout), Some(claim)) =
            (layouts.bytecode, bytecode_reduction_claims.as_ref())
        {
            let cycle_phase_variables = stage6
                .output_points
                .bytecode_cycle_phase_variables()
                .ok_or(VerifierError::MissingOpeningClaim {
                    id: bytecode_reduction::cycle_phase_intermediate_opening(),
                })?;
            Some(committed_reduction_address_phase_opening_point(
                &batch_consistency,
                claim,
                layout.precommitted(),
                &cycle_phase_variables,
                JoltRelationId::BytecodeClaimReduction,
            )?)
        } else {
            None
        };
        let program_image_address_phase = if let (Some(layout), Some(claim)) = (
            layouts.program_image,
            program_image_reduction_claims.as_ref(),
        ) {
            let cycle_phase_variables = stage6
                .output_points
                .program_image_cycle_phase_variables()
                .ok_or(VerifierError::MissingOpeningClaim {
                    id: program_image::cycle_phase_program_image_opening(),
                })?;
            Some(committed_reduction_address_phase_opening_point(
                &batch_consistency,
                claim,
                layout.precommitted(),
                &cycle_phase_variables,
                JoltRelationId::ProgramImageClaimReduction,
            )?)
        } else {
            None
        };

        let mut precommitted_final_openings = Vec::new();
        for (kind, layout, address_phase, cycle_phase) in [
            (
                JoltAdviceKind::Trusted,
                layouts.trusted_advice,
                trusted_advice.as_deref().map(PrecommittedFinalSource::zk),
                stage6
                    .output_points
                    .advice_cycle_phase_opening_point(JoltAdviceKind::Trusted)
                    .map(PrecommittedFinalSource::zk),
            ),
            (
                JoltAdviceKind::Untrusted,
                layouts.untrusted_advice,
                untrusted_advice.as_deref().map(PrecommittedFinalSource::zk),
                stage6
                    .output_points
                    .advice_cycle_phase_opening_point(JoltAdviceKind::Untrusted)
                    .map(PrecommittedFinalSource::zk),
            ),
        ] {
            if let Some(layout) = layout {
                precommitted_final_openings.push(advice_final_opening(
                    kind,
                    layout,
                    address_phase,
                    cycle_phase,
                )?);
            }
        }
        if let Some(layout) = layouts.bytecode {
            precommitted_final_openings.extend(bytecode_final_openings(
                layout,
                bytecode_address_phase
                    .as_deref()
                    .map(PrecommittedFinalSource::zk),
                stage6
                    .output_points
                    .bytecode_reduction_opening_point()
                    .map(PrecommittedFinalSource::zk),
            )?);
        }
        if let Some(layout) = layouts.program_image {
            precommitted_final_openings.push(program_image_final_opening(
                layout,
                program_image_address_phase
                    .as_deref()
                    .map(PrecommittedFinalSource::zk),
                stage6
                    .output_points
                    .program_image_opening_point()
                    .map(PrecommittedFinalSource::zk),
            )?);
        }

        return Ok(Stage7Output::Zk(Stage7ZkOutput {
            public: Stage7PublicOutput { hamming_gamma },
            batch_consistency,
            batch_output_claims,
            hamming_weight_opening_point: hamming_opening_point,
            precommitted_final_openings,
        }));
    }

    let stage4 = stage4.clear()?;
    let stage6 = stage6.clear()?;
    let claims = &proof.clear_claims()?.stage7;

    let relations =
        Stage7Relations::build(hamming_dimensions, hamming_gamma, &layouts, stage4, stage6)?;

    // Reject opening claims supplied for phases that did not run.
    if relations.trusted_advice.is_none() && claims.advice_address_phase.trusted.is_some() {
        return Err(VerifierError::UnexpectedOpeningClaim {
            id: advice::final_advice_opening(JoltAdviceKind::Trusted),
        });
    }
    if relations.untrusted_advice.is_none() && claims.advice_address_phase.untrusted.is_some() {
        return Err(VerifierError::UnexpectedOpeningClaim {
            id: advice::final_advice_opening(JoltAdviceKind::Untrusted),
        });
    }
    if relations.bytecode.is_none() && claims.bytecode_address_phase.is_some() {
        return Err(VerifierError::UnexpectedOpeningClaim {
            id: bytecode_reduction::final_bytecode_chunk_opening(0),
        });
    }
    if relations.program_image.is_none() && claims.program_image_address_phase.is_some() {
        return Err(VerifierError::UnexpectedOpeningClaim {
            id: program_image::final_program_image_opening(),
        });
    }

    let sumcheck_claims = relations.sumcheck_claims(&hamming_claims)?;
    let batch = BatchedSumcheckVerifier::verify_compressed_boolean(
        &sumcheck_claims,
        &proof.stages.stage7_sumcheck_proof,
        transcript,
    )
    .map_err(|error| VerifierError::StageClaimSumcheckFailed {
        stage: JoltRelationId::HammingWeightClaimReduction,
        reason: error.to_string(),
    })?;

    let points = relations.instance_points_from_batch(&batch, hamming_claims.sumcheck.rounds)?;
    let parts = relations.clear_output(&points, claims, stage6, &layouts)?;

    if batch.batching_coefficients.len() != parts.expected_outputs.len() {
        return Err(VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::HammingWeightClaimReduction,
            reason: format!(
                "Stage 7 batch verifier returned {} coefficients for {} instances",
                batch.batching_coefficients.len(),
                parts.expected_outputs.len()
            ),
        });
    }
    let expected_final_claim = batch
        .batching_coefficients
        .iter()
        .zip(&parts.expected_outputs)
        .map(|(coefficient, output)| *coefficient * *output)
        .sum();
    if batch.reduction.value != expected_final_claim {
        return Err(VerifierError::StageClaimOutputMismatch {
            stage: JoltRelationId::HammingWeightClaimReduction,
        });
    }

    claims.append_to_transcript(transcript);

    Ok(Stage7Output::Clear(parts.output))
}

/// The committed-program claim-reduction layouts present in this proof
/// configuration (each `Some` only when the corresponding precommitted poly is
/// committed). Bundled so the verifier and prover build the same relation set.
#[derive(Clone, Copy)]
pub struct Stage7Layouts<'a> {
    pub trusted_advice: Option<&'a AdviceClaimReductionLayout>,
    pub untrusted_advice: Option<&'a AdviceClaimReductionLayout>,
    pub bytecode: Option<&'a BytecodeClaimReductionLayout>,
    pub program_image: Option<&'a ProgramImageClaimReductionLayout>,
}

/// The stage-7 relation objects and their wired input claims. Built once from the
/// stage-4/stage-6 outputs and shared by the verifier and the prover, so the
/// input-claim and output-claim algebra cannot drift between them.
pub struct Stage7Relations<F: Field> {
    hamming_gamma: F,
    pub hamming: HammingWeightClaimReduction<F>,
    pub hamming_inputs: HammingWeightClaimReductionInputClaims<OpeningClaim<F>>,
    pub trusted_advice: Option<AdviceAddressPhase<F>>,
    pub untrusted_advice: Option<AdviceAddressPhase<F>>,
    pub advice_inputs: AdviceAddressPhaseInputClaims<OpeningClaim<F>>,
    pub bytecode: Option<BytecodeReductionAddressPhase<F>>,
    pub bytecode_inputs: Option<BytecodeReductionAddressPhaseInputClaims<OpeningClaim<F>>>,
    pub program_image: Option<ProgramImageReductionAddressPhase<F>>,
    pub program_image_inputs: Option<ProgramImageReductionAddressPhaseInputClaims<OpeningClaim<F>>>,
}

/// Each present instance's sumcheck point, in batch order. The hamming reduction
/// is suffix-aligned within the batch; the address phases are prefix-aligned.
pub struct Stage7InstancePoints<'a, F> {
    pub hamming: &'a [F],
    pub trusted_advice: Option<&'a [F]>,
    pub untrusted_advice: Option<&'a [F]>,
    pub bytecode: Option<&'a [F]>,
    pub program_image: Option<&'a [F]>,
}

/// The produced stage-7 clear output plus the per-instance expected output claims
/// (batch order) the verifier folds against the batching coefficients.
pub struct Stage7ClearOutputParts<F: Field> {
    pub output: Stage7ClearOutput<F>,
    pub expected_outputs: Vec<F>,
}

impl<F: Field> Stage7Relations<F> {
    pub fn build(
        hamming_dimensions: hamming_weight::HammingWeightClaimReductionDimensions,
        hamming_gamma: F,
        layouts: &Stage7Layouts<'_>,
        stage4: &Stage4ClearOutput<F>,
        stage6: &Stage6ClearOutput<F>,
    ) -> Result<Self, VerifierError> {
        let booleanity_opening = stage6.output_points.booleanity_opening_point().ok_or(
            VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::HammingWeightClaimReduction,
                reason: "Stage 6 booleanity produced no opening point".to_string(),
            },
        )?;
        let (booleanity_r_address, booleanity_r_cycle) =
            booleanity_opening.split_at(hamming_dimensions.log_k_chunk);
        let hamming = HammingWeightClaimReduction::new(
            hamming_dimensions,
            hamming_gamma,
            booleanity_r_cycle.to_vec(),
            booleanity_r_address.to_vec(),
            stage7_hamming_virtualization_address_points(hamming_dimensions, stage6)?,
        );
        let bytecode = clear_bytecode_relation(layouts.bytecode, stage6)?;
        let bytecode_inputs = bytecode
            .as_ref()
            .map(|_| clear_bytecode_input_claims(stage6))
            .transpose()?;
        let program_image = clear_program_image_relation(layouts.program_image, stage4, stage6)?;
        let program_image_inputs = program_image
            .as_ref()
            .map(|_| clear_program_image_input_claims(stage6))
            .transpose()?;
        Ok(Self {
            hamming_gamma,
            hamming,
            hamming_inputs: hamming_input_claims(stage6),
            trusted_advice: clear_advice_relation(
                JoltAdviceKind::Trusted,
                layouts.trusted_advice,
                stage4,
                stage6,
            )?,
            untrusted_advice: clear_advice_relation(
                JoltAdviceKind::Untrusted,
                layouts.untrusted_advice,
                stage4,
                stage6,
            )?,
            advice_inputs: clear_advice_input_claims(stage6),
            bytecode,
            bytecode_inputs,
            program_image,
            program_image_inputs,
        })
    }

    /// The hamming-weight reduction's batching gamma, drawn before the stage-7
    /// batch and recorded in the stage-7 public output.
    pub fn hamming_gamma(&self) -> F {
        self.hamming_gamma
    }

    /// The per-instance batched-sumcheck claims, in canonical batch order: hamming,
    /// trusted/untrusted advice, bytecode, program image.
    pub fn sumcheck_claims(
        &self,
        hamming_claims: &JoltRelationClaims<F>,
    ) -> Result<Vec<SumcheckClaim<F>>, VerifierError> {
        let mut claims = vec![SumcheckClaim::new(
            hamming_claims.sumcheck.rounds,
            hamming_claims.sumcheck.degree,
            self.hamming.input_claim(&self.hamming_inputs)?,
        )];
        for relation in [&self.trusted_advice, &self.untrusted_advice]
            .into_iter()
            .flatten()
        {
            let spec = &relation.sumcheck_relation().sumcheck;
            claims.push(SumcheckClaim::new(
                spec.rounds,
                spec.degree,
                relation.input_claim(&self.advice_inputs)?,
            ));
        }
        if let (Some(relation), Some(inputs)) = (&self.bytecode, &self.bytecode_inputs) {
            let spec = &relation.sumcheck_relation().sumcheck;
            claims.push(SumcheckClaim::new(
                spec.rounds,
                spec.degree,
                relation.input_claim(inputs)?,
            ));
        }
        if let (Some(relation), Some(inputs)) = (&self.program_image, &self.program_image_inputs) {
            let spec = &relation.sumcheck_relation().sumcheck;
            claims.push(SumcheckClaim::new(
                spec.rounds,
                spec.degree,
                relation.input_claim(inputs)?,
            ));
        }
        Ok(claims)
    }

    /// Slice each present instance's sumcheck point out of the verifier's batched
    /// evaluation claim (hamming suffix-aligned, address phases prefix-aligned).
    pub fn instance_points_from_batch<'a>(
        &self,
        batch: &'a BatchedEvaluationClaim<F>,
        hamming_rounds: usize,
    ) -> Result<Stage7InstancePoints<'a, F>, VerifierError> {
        let address = |relation: &Option<AdviceAddressPhase<F>>| {
            relation
                .as_ref()
                .map(|relation| {
                    address_phase_point(
                        batch,
                        relation.sumcheck_relation().sumcheck.rounds,
                        relation.id(),
                    )
                })
                .transpose()
        };
        Ok(Stage7InstancePoints {
            hamming: batch.try_instance_point(hamming_rounds).map_err(|error| {
                sumcheck_failed(JoltRelationId::HammingWeightClaimReduction, error)
            })?,
            trusted_advice: address(&self.trusted_advice)?,
            untrusted_advice: address(&self.untrusted_advice)?,
            bytecode: self
                .bytecode
                .as_ref()
                .map(|relation| {
                    address_phase_point(
                        batch,
                        relation.sumcheck_relation().sumcheck.rounds,
                        relation.id(),
                    )
                })
                .transpose()?,
            program_image: self
                .program_image
                .as_ref()
                .map(|relation| {
                    address_phase_point(
                        batch,
                        relation.sumcheck_relation().sumcheck.rounds,
                        relation.id(),
                    )
                })
                .transpose()?,
        })
    }

    /// Build the produced clear output from the wire opening values (`claims`) and
    /// the per-instance sumcheck points: derive each relation's opening points,
    /// pair them with the values, evaluate the expected output claims, and resolve
    /// the precommitted final openings. Shared by the verifier and the prover.
    pub fn clear_output(
        &self,
        points: &Stage7InstancePoints<'_, F>,
        claims: &Stage7OutputClaims<F>,
        stage6: &Stage6ClearOutput<F>,
        layouts: &Stage7Layouts<'_>,
    ) -> Result<Stage7ClearOutputParts<F>, VerifierError> {
        let hamming_output_points = self
            .hamming
            .derive_opening_points(points.hamming, &self.hamming_inputs)?;
        let hamming_weight_opening_point = hamming_output_points
            .instruction_ra
            .first()
            .or_else(|| hamming_output_points.bytecode_ra.first())
            .or_else(|| hamming_output_points.ram_ra.first())
            .cloned()
            .ok_or_else(|| VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::HammingWeightClaimReduction,
                reason: "stage 7 produced no hamming-weight openings".to_string(),
            })?;
        let hamming_output = zip_openings(
            &claims.hamming_weight_claim_reduction,
            &hamming_output_points,
        );

        let advice_points = AdviceAddressPhaseOutputClaims {
            trusted: advice_kind_point(
                self.trusted_advice.as_ref(),
                points.trusted_advice,
                &self.advice_inputs,
            )?,
            untrusted: advice_kind_point(
                self.untrusted_advice.as_ref(),
                points.untrusted_advice,
                &self.advice_inputs,
            )?,
        };
        let advice_output = zip_openings(&claims.advice_address_phase, &advice_points);

        let bytecode_output = match (&self.bytecode, &self.bytecode_inputs, points.bytecode) {
            (Some(relation), Some(inputs), Some(point)) => {
                let derived = relation.derive_opening_points(point, inputs)?;
                let wire = claims.bytecode_address_phase.as_ref().ok_or(
                    VerifierError::MissingOpeningClaim {
                        id: bytecode_reduction::final_bytecode_chunk_opening(0),
                    },
                )?;
                Some(zip_openings(wire, &derived))
            }
            _ => None,
        };
        let program_image_output = match (
            &self.program_image,
            &self.program_image_inputs,
            points.program_image,
        ) {
            (Some(relation), Some(inputs), Some(point)) => {
                let derived = relation.derive_opening_points(point, inputs)?;
                let wire = claims.program_image_address_phase.as_ref().ok_or(
                    VerifierError::MissingOpeningClaim {
                        id: program_image::final_program_image_opening(),
                    },
                )?;
                Some(zip_openings(wire, &derived))
            }
            _ => None,
        };

        let output_claims = Stage7OutputClaims {
            hamming_weight_claim_reduction: hamming_output,
            advice_address_phase: advice_output,
            bytecode_address_phase: bytecode_output,
            program_image_address_phase: program_image_output,
        };

        let mut expected_outputs = vec![self.hamming.expected_output(
            &self.hamming_inputs,
            &output_claims.hamming_weight_claim_reduction,
        )?];
        for relation in [&self.trusted_advice, &self.untrusted_advice]
            .into_iter()
            .flatten()
        {
            expected_outputs.push(
                relation
                    .expected_output(&self.advice_inputs, &output_claims.advice_address_phase)?,
            );
        }
        if let (Some(relation), Some(inputs), Some(output)) = (
            &self.bytecode,
            &self.bytecode_inputs,
            &output_claims.bytecode_address_phase,
        ) {
            expected_outputs.push(relation.expected_output(inputs, output)?);
        }
        if let (Some(relation), Some(inputs), Some(output)) = (
            &self.program_image,
            &self.program_image_inputs,
            &output_claims.program_image_address_phase,
        ) {
            expected_outputs.push(relation.expected_output(inputs, output)?);
        }

        let precommitted_final_openings =
            clear_precommitted_final_openings(layouts, &output_claims, stage6)?;

        Ok(Stage7ClearOutputParts {
            output: Stage7ClearOutput {
                output_claims,
                hamming_weight_opening_point,
                precommitted_final_openings,
            },
            expected_outputs,
        })
    }
}

fn sumcheck_failed<F: Field>(
    stage: JoltRelationId,
    error: jolt_sumcheck::SumcheckError<F>,
) -> VerifierError {
    VerifierError::StageClaimSumcheckFailed {
        stage,
        reason: error.to_string(),
    }
}

/// The prefix-aligned instance point of an address-phase relation within the
/// stage-7 batch (address phases start at offset 0).
fn address_phase_point<F: Field>(
    batch: &BatchedEvaluationClaim<F>,
    rounds: usize,
    stage: JoltRelationId,
) -> Result<&[F], VerifierError> {
    batch
        .try_instance_point_at(0, rounds)
        .map_err(|error| sumcheck_failed(stage, error))
}

/// The produced address-phase opening point for `relation`'s kind, or `None` when
/// the kind's advice address phase did not run.
fn advice_kind_point<F: Field>(
    relation: Option<&AdviceAddressPhase<F>>,
    point: Option<&[F]>,
    inputs: &AdviceAddressPhaseInputClaims<OpeningClaim<F>>,
) -> Result<Option<Vec<F>>, VerifierError> {
    let (Some(relation), Some(point)) = (relation, point) else {
        return Ok(None);
    };
    let derived = relation.derive_opening_points(point, inputs)?;
    Ok(match relation.kind() {
        JoltAdviceKind::Trusted => derived.trusted,
        JoltAdviceKind::Untrusted => derived.untrusted,
    })
}

fn input_opening<F: Field>(value: F) -> OpeningClaim<F> {
    OpeningClaim {
        point: Vec::new(),
        value,
    }
}

/// The hamming reduction's consumed claims, wired from stage 6. The relation reads
/// only their values (its produced points are derived from its own sumcheck
/// point), so the input opening points are left empty.
fn hamming_input_claims<F: Field>(
    stage6: &Stage6ClearOutput<F>,
) -> HammingWeightClaimReductionInputClaims<OpeningClaim<F>> {
    let openings = |values: &[F]| values.iter().copied().map(input_opening).collect();
    HammingWeightClaimReductionInputClaims {
        ram_hamming_weight: input_opening(
            stage6
                .output_claims
                .ram_hamming_booleanity
                .ram_hamming_weight,
        ),
        instruction_booleanity: openings(&stage6.output_claims.booleanity.instruction_ra),
        bytecode_booleanity: openings(&stage6.output_claims.booleanity.bytecode_ra),
        ram_booleanity: openings(&stage6.output_claims.booleanity.ram_ra),
        instruction_virtualization: openings(
            &stage6
                .output_claims
                .instruction_ra_virtualization
                .committed_instruction_ra,
        ),
        bytecode_virtualization: openings(&stage6.output_claims.bytecode_read_raf.bytecode_ra),
        ram_virtualization: openings(&stage6.output_claims.ram_ra_virtualization.ram_ra),
    }
}

fn clear_advice_relation<F: Field>(
    kind: JoltAdviceKind,
    layout: Option<&AdviceClaimReductionLayout>,
    stage4: &Stage4ClearOutput<F>,
    stage6: &Stage6ClearOutput<F>,
) -> Result<Option<AdviceAddressPhase<F>>, VerifierError> {
    let Some(layout) = layout.filter(|layout| layout.dimensions().has_address_phase()) else {
        return Ok(None);
    };
    let cycle_phase_variables = stage6
        .output_points
        .advice_cycle_phase_variables(kind)
        .ok_or_else(|| VerifierError::MissingOpeningClaim {
            id: advice::cycle_phase_advice_opening(kind),
        })?;
    let contribution = stage4
        .ram_val_check_init
        .advice_contribution(kind)
        .ok_or_else(|| VerifierError::MissingOpeningClaim {
            id: advice::ram_val_check_advice_opening(kind),
        })?;
    Ok(Some(AdviceAddressPhase::new(
        kind,
        layout,
        contribution.opening.point.clone(),
        cycle_phase_variables,
    )))
}

/// The consumed cycle-phase advice openings (both kinds, present only when that
/// kind ran). Each address-phase relation reads its own kind's field.
fn clear_advice_input_claims<F: Field>(
    stage6: &Stage6ClearOutput<F>,
) -> AdviceAddressPhaseInputClaims<OpeningClaim<F>> {
    AdviceAddressPhaseInputClaims {
        trusted: stage6_advice_cycle_phase_claim(stage6, JoltAdviceKind::Trusted)
            .ok()
            .map(input_opening),
        untrusted: stage6_advice_cycle_phase_claim(stage6, JoltAdviceKind::Untrusted)
            .ok()
            .map(input_opening),
    }
}

fn clear_bytecode_relation<F: Field>(
    layout: Option<&BytecodeClaimReductionLayout>,
    stage6: &Stage6ClearOutput<F>,
) -> Result<Option<BytecodeReductionAddressPhase<F>>, VerifierError> {
    let Some(layout) = layout.filter(|layout| layout.dimensions().has_address_phase()) else {
        return Ok(None);
    };
    let weights =
        stage6
            .bytecode_reduction_weights
            .as_ref()
            .ok_or(VerifierError::MissingOpeningClaim {
                id: bytecode_reduction::cycle_phase_intermediate_opening(),
            })?;
    let cycle_phase_variables = stage6
        .output_points
        .bytecode_cycle_phase_variables()
        .ok_or(VerifierError::MissingOpeningClaim {
            id: bytecode_reduction::cycle_phase_intermediate_opening(),
        })?;
    Ok(Some(BytecodeReductionAddressPhase::new(
        layout,
        BytecodeOutputWeightInputs {
            r_bc: &weights.r_bc,
            chunk_rbc_weights: &weights.chunk_rbc_weights,
            lane_weights: &weights.lane_weights,
        },
        cycle_phase_variables,
    )))
}

fn clear_bytecode_input_claims<F: Field>(
    stage6: &Stage6ClearOutput<F>,
) -> Result<BytecodeReductionAddressPhaseInputClaims<OpeningClaim<F>>, VerifierError> {
    let value = match stage6.output_claims.bytecode_claim_reduction.as_ref() {
        Some(BytecodeCyclePhaseOutputClaims::Intermediate(value)) => *value,
        _ => {
            return Err(VerifierError::MissingOpeningClaim {
                id: bytecode_reduction::cycle_phase_intermediate_opening(),
            })
        }
    };
    Ok(BytecodeReductionAddressPhaseInputClaims {
        cycle_phase_intermediate: input_opening(value),
    })
}

fn clear_program_image_relation<F: Field>(
    layout: Option<&ProgramImageClaimReductionLayout>,
    stage4: &Stage4ClearOutput<F>,
    stage6: &Stage6ClearOutput<F>,
) -> Result<Option<ProgramImageReductionAddressPhase<F>>, VerifierError> {
    let Some(layout) = layout.filter(|layout| layout.dimensions().has_address_phase()) else {
        return Ok(None);
    };
    let cycle_phase_variables = stage6
        .output_points
        .program_image_cycle_phase_variables()
        .ok_or(VerifierError::MissingOpeningClaim {
            id: program_image::cycle_phase_program_image_opening(),
        })?;
    let contribution = stage4
        .ram_val_check_init
        .program_image_contribution
        .as_ref()
        .ok_or(VerifierError::MissingOpeningClaim {
            id: program_image::ram_val_check_contribution_opening(),
        })?;
    Ok(Some(ProgramImageReductionAddressPhase::new(
        layout,
        contribution.point.clone(),
        cycle_phase_variables,
    )))
}

fn clear_program_image_input_claims<F: Field>(
    stage6: &Stage6ClearOutput<F>,
) -> Result<ProgramImageReductionAddressPhaseInputClaims<OpeningClaim<F>>, VerifierError> {
    let value = stage6
        .output_claims
        .program_image_claim_reduction
        .as_ref()
        .map(|claim| claim.opening_claim)
        .ok_or(VerifierError::MissingOpeningClaim {
            id: program_image::cycle_phase_program_image_opening(),
        })?;
    Ok(ProgramImageReductionAddressPhaseInputClaims {
        cycle_phase: input_opening(value),
    })
}

/// Resolve the final openings of the precommitted polynomials from whichever phase
/// completed each reduction: this stage's address phase (when it ran) or the
/// stage 6b cycle phase. Mirrors the ZK arm's assembly but reads the clear address
/// openings from the produced [`Stage7OutputClaims`].
fn clear_precommitted_final_openings<F: Field>(
    layouts: &Stage7Layouts<'_>,
    output_claims: &Stage7OutputClaims<OpeningClaim<F>>,
    stage6: &Stage6ClearOutput<F>,
) -> Result<Vec<PrecommittedFinalOpening<F>>, VerifierError> {
    let mut openings = Vec::new();
    for (kind, layout, address, cycle) in [
        (
            JoltAdviceKind::Trusted,
            layouts.trusted_advice,
            output_claims.advice_address_phase.trusted.as_ref(),
            stage6
                .output_points
                .advice_cycle_phase_opening_point(JoltAdviceKind::Trusted)
                .zip(stage6.output_claims.advice_cycle_phase.trusted.as_ref()),
        ),
        (
            JoltAdviceKind::Untrusted,
            layouts.untrusted_advice,
            output_claims.advice_address_phase.untrusted.as_ref(),
            stage6
                .output_points
                .advice_cycle_phase_opening_point(JoltAdviceKind::Untrusted)
                .zip(stage6.output_claims.advice_cycle_phase.untrusted.as_ref()),
        ),
    ] {
        if let Some(layout) = layout {
            let address_phase = address
                .map(|opening| PrecommittedFinalSource::clear(&opening.point, opening.value));
            let cycle_phase = cycle.map(|(opening_point, claim)| {
                PrecommittedFinalSource::clear(opening_point, claim.opening_claim)
            });
            openings.push(advice_final_opening(
                kind,
                layout,
                address_phase,
                cycle_phase,
            )?);
        }
    }
    if let Some(layout) = layouts.bytecode {
        let address_phase = output_claims
            .bytecode_address_phase
            .as_ref()
            .and_then(|output| output.chunks.first().map(|first| (output, first)))
            .map(|(output, first)| {
                PrecommittedFinalSource::clear_chunks(
                    &first.point,
                    output.chunks.iter().map(|chunk| chunk.value).collect(),
                )
            });
        let cycle_phase = match (
            stage6.output_points.bytecode_reduction_opening_point(),
            &stage6.output_claims.bytecode_claim_reduction,
        ) {
            (Some(opening_point), Some(BytecodeCyclePhaseOutputClaims::Chunks(chunks))) => Some(
                PrecommittedFinalSource::clear_chunks(opening_point, chunks.clone()),
            ),
            _ => None,
        };
        openings.extend(bytecode_final_openings(layout, address_phase, cycle_phase)?);
    }
    if let Some(layout) = layouts.program_image {
        let address_phase = output_claims
            .program_image_address_phase
            .as_ref()
            .map(|output| {
                PrecommittedFinalSource::clear(
                    &output.program_image.point,
                    output.program_image.value,
                )
            });
        let cycle_phase = stage6
            .output_points
            .program_image_opening_point()
            .zip(stage6.output_claims.program_image_claim_reduction.as_ref())
            .map(|(opening_point, claim)| {
                PrecommittedFinalSource::clear(opening_point, claim.opening_claim)
            });
        openings.push(program_image_final_opening(
            layout,
            address_phase,
            cycle_phase,
        )?);
    }
    Ok(openings)
}

pub fn stage7_hamming_virtualization_address_points<F: Field>(
    dimensions: hamming_weight::HammingWeightClaimReductionDimensions,
    stage6: &Stage6ClearOutput<F>,
) -> Result<Vec<Vec<F>>, VerifierError> {
    let instruction_ra_points = &stage6
        .output_points
        .instruction_ra_virtualization
        .committed_instruction_ra;
    let bytecode_ra_points = &stage6.output_points.bytecode_read_raf.bytecode_ra;
    let ram_ra_points = &stage6.output_points.ram_ra_virtualization.ram_ra;
    if instruction_ra_points.len() != dimensions.layout.instruction()
        || bytecode_ra_points.len() != dimensions.layout.bytecode()
        || ram_ra_points.len() != dimensions.layout.ram()
    {
        return Err(VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::HammingWeightClaimReduction,
            reason: "Stage 6 RA opening point count mismatch for Stage 7".to_string(),
        });
    }

    let mut points = Vec::with_capacity(dimensions.layout.total());
    for point in instruction_ra_points {
        points.push(hamming_virtualization_address_point(
            dimensions.log_k_chunk,
            point,
        )?);
    }
    for point in bytecode_ra_points {
        points.push(hamming_virtualization_address_point(
            dimensions.log_k_chunk,
            point,
        )?);
    }
    for point in ram_ra_points {
        points.push(hamming_virtualization_address_point(
            dimensions.log_k_chunk,
            point,
        )?);
    }
    Ok(points)
}

fn hamming_virtualization_address_point<F: Field>(
    log_k_chunk: usize,
    point: &[F],
) -> Result<Vec<F>, VerifierError> {
    if point.len() < log_k_chunk {
        return Err(VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::HammingWeightClaimReduction,
            reason: format!(
                "Stage 6 RA opening point is too short for HammingWeight address chunk: expected at least {log_k_chunk}, got {}",
                point.len()
            ),
        });
    }
    Ok(point[..log_k_chunk].to_vec())
}

/// Opening point and (clear-mode) claim payload recorded by the stage that
/// completed a precommitted claim reduction. `T` is a single claim for advice and
/// the program image, and the per-chunk claim slice for the committed bytecode.
struct PrecommittedFinalSource<'a, F, T = F> {
    point: &'a [F],
    opening_claim: Option<T>,
}

impl<'a, F, T> PrecommittedFinalSource<'a, F, T> {
    fn zk(point: &'a [F]) -> Self {
        Self {
            point,
            opening_claim: None,
        }
    }

    fn clear(point: &'a [F], opening_claim: T) -> Self {
        Self {
            point,
            opening_claim: Some(opening_claim),
        }
    }
}

impl<'a, F> PrecommittedFinalSource<'a, F, Vec<F>> {
    fn clear_chunks(point: &'a [F], chunks: Vec<F>) -> Self {
        Self {
            point,
            opening_claim: Some(chunks),
        }
    }
}

/// Resolves the final opening of an advice polynomial from whichever phase
/// completed its reduction: this stage's address phase, or the stage 6b cycle
/// phase when no active address rounds remain.
fn advice_final_opening<F: Field>(
    kind: JoltAdviceKind,
    layout: &AdviceClaimReductionLayout,
    address_phase: Option<PrecommittedFinalSource<'_, F>>,
    cycle_phase: Option<PrecommittedFinalSource<'_, F>>,
) -> Result<PrecommittedFinalOpening<F>, VerifierError> {
    let source = if layout.dimensions().has_address_phase() {
        address_phase
    } else {
        cycle_phase
    };
    let source = source.ok_or(VerifierError::MissingOpeningClaim {
        id: advice::final_advice_opening(kind),
    })?;
    let polynomial = match kind {
        JoltAdviceKind::Trusted => JoltCommittedPolynomial::TrustedAdvice,
        JoltAdviceKind::Untrusted => JoltCommittedPolynomial::UntrustedAdvice,
    };
    Ok(PrecommittedFinalOpening {
        polynomial,
        point: source.point.to_vec(),
        opening_claim: source.opening_claim,
    })
}

/// The advice address-phase opening point (the point at which the committed
/// advice polynomial is finally opened), recovered for the ZK precommitted finals.
/// BlindFold recomputes the sumcheck point and `FinalScale` independently.
fn advice_address_phase_opening_point<F: Field, C>(
    batch: &BatchedCommittedSumcheckConsistency<F, C>,
    claim: &JoltRelationClaims<F>,
    layout: &AdviceClaimReductionLayout,
    kind: JoltAdviceKind,
    stage6: &Stage6ZkOutput<F, C>,
) -> Result<Vec<F>, VerifierError> {
    let advice_point = batch
        .try_instance_point_at(0, claim.sumcheck.rounds)
        .map_err(|error| VerifierError::StageClaimSumcheckFailed {
            stage: JoltRelationId::AdviceClaimReduction,
            reason: error.to_string(),
        })?;
    let cycle_phase_variables = stage6
        .output_points
        .advice_cycle_phase_variables(kind)
        .ok_or_else(|| VerifierError::MissingOpeningClaim {
            id: advice::cycle_phase_advice_opening(kind),
        })?;
    layout
        .address_phase_opening_point(&cycle_phase_variables, &advice_point)
        .map_err(|error| VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::AdviceClaimReduction,
            reason: error.to_string(),
        })
}

fn stage6_advice_cycle_phase_claim<F: Field>(
    stage6: &Stage6ClearOutput<F>,
    kind: JoltAdviceKind,
) -> Result<F, VerifierError> {
    let claim = match kind {
        JoltAdviceKind::Trusted => stage6.output_claims.advice_cycle_phase.trusted.as_ref(),
        JoltAdviceKind::Untrusted => stage6.output_claims.advice_cycle_phase.untrusted.as_ref(),
    };
    claim
        .map(|claim| claim.opening_claim)
        .ok_or_else(|| VerifierError::MissingOpeningClaim {
            id: advice::cycle_phase_advice_opening(kind),
        })
}

/// The committed bytecode / program-image address-phase opening point, recovered
/// for the ZK precommitted finals (BlindFold recomputes the rest).
fn committed_reduction_address_phase_opening_point<F: Field, C>(
    batch: &BatchedCommittedSumcheckConsistency<F, C>,
    claim: &JoltRelationClaims<F>,
    precommitted: &jolt_claims::protocols::jolt::PrecommittedClaimReduction,
    cycle_phase_variables: &[F],
    stage: JoltRelationId,
) -> Result<Vec<F>, VerifierError> {
    let point = batch
        .try_instance_point_at(0, claim.sumcheck.rounds)
        .map_err(|error| VerifierError::StageClaimSumcheckFailed {
            stage,
            reason: error.to_string(),
        })?;
    precommitted
        .address_phase_opening_point(cycle_phase_variables, &point)
        .map_err(|error| VerifierError::StageClaimPublicInputFailed {
            stage,
            reason: error.to_string(),
        })
}

/// Resolves the final per-chunk openings of the committed bytecode from whichever
/// phase completed the reduction: this stage's address phase, or the stage 6b
/// cycle phase when no active address rounds remain.
fn bytecode_final_openings<F: Field>(
    layout: &BytecodeClaimReductionLayout,
    address_phase: Option<PrecommittedFinalSource<'_, F, Vec<F>>>,
    cycle_phase: Option<PrecommittedFinalSource<'_, F, Vec<F>>>,
) -> Result<Vec<PrecommittedFinalOpening<F>>, VerifierError> {
    let source = if layout.dimensions().has_address_phase() {
        address_phase
    } else {
        cycle_phase
    };
    let source = source.ok_or(VerifierError::MissingOpeningClaim {
        id: bytecode_reduction::final_bytecode_chunk_opening(0),
    })?;
    if let Some(chunk_claims) = &source.opening_claim {
        if chunk_claims.len() != layout.chunk_count() {
            return Err(VerifierError::StageClaimPublicInputFailed {
                stage: JoltRelationId::BytecodeClaimReduction,
                reason: format!(
                    "final bytecode chunk claim count mismatch: expected {}, got {}",
                    layout.chunk_count(),
                    chunk_claims.len()
                ),
            });
        }
    }
    Ok((0..layout.chunk_count())
        .map(|chunk_idx| PrecommittedFinalOpening {
            polynomial: JoltCommittedPolynomial::BytecodeChunk(chunk_idx),
            point: source.point.to_vec(),
            opening_claim: source
                .opening_claim
                .as_ref()
                .map(|chunk_claims| chunk_claims[chunk_idx]),
        })
        .collect())
}

/// Resolves the final opening of the committed program image from whichever phase
/// completed the reduction: this stage's address phase, or the stage 6b cycle
/// phase when no active address rounds remain.
fn program_image_final_opening<F: Field>(
    layout: &ProgramImageClaimReductionLayout,
    address_phase: Option<PrecommittedFinalSource<'_, F>>,
    cycle_phase: Option<PrecommittedFinalSource<'_, F>>,
) -> Result<PrecommittedFinalOpening<F>, VerifierError> {
    let source = if layout.dimensions().has_address_phase() {
        address_phase
    } else {
        cycle_phase
    };
    let source = source.ok_or(VerifierError::MissingOpeningClaim {
        id: program_image::final_program_image_opening(),
    })?;
    Ok(PrecommittedFinalOpening {
        polynomial: JoltCommittedPolynomial::ProgramImageInit,
        point: source.point.to_vec(),
        opening_claim: source.opening_claim,
    })
}
