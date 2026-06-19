use jolt_claims::protocols::jolt::{
    formulas::{booleanity, bytecode},
    JoltCommittedPolynomial,
};
use jolt_claims::protocols::jolt::{
    formulas::{claim_reductions::advice, claim_reductions::increments, instruction, ram},
    AdviceClaimReductionLayout, JoltAdviceKind, PrecommittedReductionLayout,
};
use jolt_field::Field;
use jolt_verifier::stages::relations::{zip_openings, SumcheckInstance};
use jolt_verifier::stages::stage6::batch::{Stage6Relations, Stage6RelationsParams};
use jolt_verifier::stages::stage6::inputs::{
    AdviceCyclePhaseOutputClaim, Stage6AddressPhaseClaims, Stage6AdviceCyclePhaseClaims,
    Stage6OutputClaims,
};
use jolt_verifier::stages::stage6::{
    stage6_advice_cycle_phase_reference, stage6_bytecode_cycle_points,
    stage6_bytecode_register_points, stage6_inc_claim_reduction_cycle_points,
    stage6_instruction_read_raf_point, stage6_stage1_cycle_binding,
    stage6_stage5_ram_reduced_opening_point, AdviceCyclePhaseOutputClaims,
    Stage6AdviceCyclePhaseReference, Stage6BatchExpectedOutputClaims, Stage6BatchInputClaims,
};
use jolt_verifier::stages::{
    stage1::Stage1ClearOutput, stage2::Stage2ClearOutput, stage3::Stage3ClearOutput,
    stage4::Stage4ClearOutput, stage5::Stage5ClearOutput,
};
use jolt_witness::protocols::jolt_vm::JoltVmNamespace;
use jolt_witness::{OracleRef, WitnessProvider};

use super::io::{Stage6ProverConfig, Stage6RegularBatchPrefixOutput};
use super::relation_state::{AdviceCyclePhaseRelationState, Stage6RelationState};
use crate::stages::invalid_sumcheck_output;
use crate::ProverError;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Stage6InstanceKind {
    BytecodeReadRaf,
    Booleanity,
    RamHammingBooleanity,
    RamRaVirtualization,
    InstructionRaVirtualization,
    IncClaimReduction,
    AdviceCyclePhase(JoltAdviceKind),
}

#[derive(Clone, Debug)]
pub(super) struct Stage6BatchInstance<F: Field> {
    pub(super) kind: Stage6InstanceKind,
    pub(super) input_claim: F,
    pub(super) num_vars: usize,
    pub(super) degree: usize,
    pub(super) offset: usize,
}

impl<F: Field> Stage6BatchInstance<F> {
    pub(super) fn is_active(&self, round: usize) -> bool {
        (self.offset..self.offset + self.num_vars).contains(&round)
    }
}

pub(super) struct Stage6BatchContext<'a, F: Field, W> {
    pub(super) config: Stage6ProverConfig,
    pub(super) witness: &'a W,
    pub(super) stage1: &'a Stage1ClearOutput<F>,
    pub(super) stage2: &'a Stage2ClearOutput<F>,
    pub(super) stage3: &'a Stage3ClearOutput<F>,
    pub(super) stage4: &'a Stage4ClearOutput<F>,
    pub(super) stage5: &'a Stage5ClearOutput<F>,
    pub(super) prefix: &'a Stage6RegularBatchPrefixOutput<F>,
    pub(super) instances: Vec<Stage6BatchInstance<F>>,
    pub(super) max_num_vars: usize,
    /// Stage 6a address-phase output openings used as the bytecode/booleanity
    /// cycle-phase input claims.
    address_phase: Stage6AddressPhaseClaims<F>,
    /// Stage 6a bytecode address-phase sumcheck challenges; reversed, they are the
    /// bytecode cycle relation's `r_address`.
    bytecode_address_challenges: Vec<F>,
    /// Stage 6a booleanity address-phase sumcheck challenges; reversed, they are
    /// the booleanity cycle relation's `r_address`.
    booleanity_address_challenges: Vec<F>,
}

struct Stage6InstanceSpec<F: Field> {
    kind: Stage6InstanceKind,
    input_claim: F,
    num_vars: usize,
    degree: usize,
}

impl<'a, F, W> Stage6BatchContext<'a, F, W>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>,
{
    /// Builds the metadata for the stage 6b **cycle-phase** batch.
    #[expect(
        clippy::too_many_arguments,
        reason = "Stage 6 context owns verifier-aligned dependencies from stages 1-5."
    )]
    pub(super) fn new_metadata(
        config: Stage6ProverConfig,
        witness: &'a W,
        stage1: &'a Stage1ClearOutput<F>,
        stage2: &'a Stage2ClearOutput<F>,
        stage3: &'a Stage3ClearOutput<F>,
        stage4: &'a Stage4ClearOutput<F>,
        stage5: &'a Stage5ClearOutput<F>,
        prefix: &'a Stage6RegularBatchPrefixOutput<F>,
        address_phase: &Stage6AddressPhaseClaims<F>,
        bytecode_address_challenges: &[F],
        booleanity_address_challenges: &[F],
    ) -> Result<Self, ProverError> {
        let bytecode_claims =
            bytecode::read_raf_cycle_phase::<F>(config.bytecode_read_raf_dimensions);
        let booleanity_claims =
            booleanity::booleanity_cycle_phase::<F>(config.booleanity_dimensions);
        let ram_hamming_claims = ram::hamming_booleanity::<F>(config.trace_dimensions());
        let ram_ra_claims = ram::ra_virtualization::<F>(config.ram_ra_virtualization_dimensions);
        let instruction_ra_claims =
            instruction::ra_virtualization::<F>(config.instruction_ra_virtualization_dimensions);
        let inc_claims = increments::claim_reduction::<F>(config.trace_dimensions());

        let mut specs = vec![
            Stage6InstanceSpec {
                kind: Stage6InstanceKind::BytecodeReadRaf,
                input_claim: address_phase.bytecode_read_raf,
                num_vars: bytecode_claims.sumcheck.rounds,
                degree: bytecode_claims.sumcheck.degree,
            },
            Stage6InstanceSpec {
                kind: Stage6InstanceKind::Booleanity,
                input_claim: address_phase.booleanity,
                num_vars: booleanity_claims.sumcheck.rounds,
                degree: booleanity_claims.sumcheck.degree,
            },
            Stage6InstanceSpec {
                kind: Stage6InstanceKind::RamHammingBooleanity,
                input_claim: prefix.input_claims.ram_hamming_booleanity,
                num_vars: ram_hamming_claims.sumcheck.rounds,
                degree: ram_hamming_claims.sumcheck.degree,
            },
            Stage6InstanceSpec {
                kind: Stage6InstanceKind::RamRaVirtualization,
                input_claim: prefix.input_claims.ram_ra_virtualization,
                num_vars: ram_ra_claims.sumcheck.rounds,
                degree: ram_ra_claims.sumcheck.degree,
            },
            Stage6InstanceSpec {
                kind: Stage6InstanceKind::InstructionRaVirtualization,
                input_claim: prefix.input_claims.instruction_ra_virtualization,
                num_vars: instruction_ra_claims.sumcheck.rounds,
                degree: instruction_ra_claims.sumcheck.degree,
            },
            Stage6InstanceSpec {
                kind: Stage6InstanceKind::IncClaimReduction,
                input_claim: prefix.input_claims.inc_claim_reduction,
                num_vars: inc_claims.sumcheck.rounds,
                degree: inc_claims.sumcheck.degree,
            },
        ];

        if let Some(input_claim) = prefix.input_claims.trusted_advice_cycle_phase {
            let layout = config.trusted_advice_layout.as_ref().ok_or_else(|| {
                invalid_stage_request("Stage 6 trusted advice input has no configured layout")
            })?;
            let claims = advice::cycle_phase::<F>(JoltAdviceKind::Trusted, layout.dimensions());
            specs.push(Stage6InstanceSpec {
                kind: Stage6InstanceKind::AdviceCyclePhase(JoltAdviceKind::Trusted),
                input_claim,
                num_vars: claims.sumcheck.rounds,
                degree: claims.sumcheck.degree,
            });
        }
        if let Some(input_claim) = prefix.input_claims.untrusted_advice_cycle_phase {
            let layout = config.untrusted_advice_layout.as_ref().ok_or_else(|| {
                invalid_stage_request("Stage 6 untrusted advice input has no configured layout")
            })?;
            let claims = advice::cycle_phase::<F>(JoltAdviceKind::Untrusted, layout.dimensions());
            specs.push(Stage6InstanceSpec {
                kind: Stage6InstanceKind::AdviceCyclePhase(JoltAdviceKind::Untrusted),
                input_claim,
                num_vars: claims.sumcheck.rounds,
                degree: claims.sumcheck.degree,
            });
        }

        let first = specs
            .first()
            .ok_or_else(|| invalid_stage_request("Stage 6 batch has no sumcheck instances"))?;
        let max_num_vars = specs
            .iter()
            .fold(first.num_vars, |max: usize, spec| max.max(spec.num_vars));

        let mut instances = Vec::with_capacity(specs.len());
        for spec in specs {
            let offset = match spec.kind {
                Stage6InstanceKind::AdviceCyclePhase(kind) => {
                    advice_cycle_phase_offset(&config, kind, max_num_vars)?
                }
                _ => max_num_vars.checked_sub(spec.num_vars).ok_or_else(|| {
                    invalid_stage_request("Stage 6 instance has more variables than batch")
                })?,
            };
            if offset + spec.num_vars > max_num_vars {
                return Err(invalid_stage_request(format!(
                    "Stage 6 instance {:?} at offset {offset} with {} variables exceeds batch size {max_num_vars}",
                    spec.kind, spec.num_vars
                )));
            }
            instances.push(Stage6BatchInstance {
                kind: spec.kind,
                input_claim: spec.input_claim,
                num_vars: spec.num_vars,
                degree: spec.degree,
                offset,
            });
        }

        Ok(Self {
            config,
            witness,
            stage1,
            stage2,
            stage3,
            stage4,
            stage5,
            prefix,
            instances,
            max_num_vars,
            address_phase: address_phase.clone(),
            bytecode_address_challenges: bytecode_address_challenges.to_vec(),
            booleanity_address_challenges: booleanity_address_challenges.to_vec(),
        })
    }

    /// Builds the stage-6b cycle relation bundle shared with the verifier, sourcing
    /// its construction points from the prover's config and the stage-1..5 clear
    /// outputs. The bytecode/booleanity `r_address` are the reversed stage-6a
    /// address challenges (matching the verifier's reversed address opening point).
    /// The modular prover only supports full programs, so there are no committed
    /// claim reductions.
    fn cycle_relations(&self) -> Result<Stage6Relations<'_, F>, ProverError> {
        let config = &self.config;
        let bytecode_context = config.bytecode_context.as_ref().ok_or_else(|| {
            invalid_stage_request("Stage 6 bytecode context is required for the cycle relations")
        })?;
        let stage_cycle_points = stage6_bytecode_cycle_points(
            self.stage1,
            self.stage2,
            self.stage3,
            self.stage4,
            self.stage5,
        )
        .map_err(invalid_stage_request)?;
        let register_points = stage6_bytecode_register_points(self.stage4, self.stage5)
            .map_err(invalid_stage_request)?;
        let ram_reduced =
            stage6_stage5_ram_reduced_opening_point(self.stage5, config.log_k, config.log_t)
                .map_err(invalid_stage_request)?;
        let instruction_read_raf = stage6_instruction_read_raf_point(self.stage5);
        let inc_cycles = stage6_inc_claim_reduction_cycle_points(
            self.stage2,
            self.stage4,
            self.stage5,
            config.log_k,
        )
        .map_err(invalid_stage_request)?;
        let challenges = &self.prefix.challenges;
        let reversed = |challenges: &[F]| challenges.iter().rev().copied().collect::<Vec<_>>();

        Stage6Relations::build(
            Stage6RelationsParams {
                bytecode_dimensions: config.bytecode_read_raf_dimensions,
                booleanity_dimensions: config.booleanity_dimensions,
                trace_dimensions: config.trace_dimensions(),
                ram_ra_dimensions: config.ram_ra_virtualization_dimensions,
                instruction_ra_dimensions: config.instruction_ra_virtualization_dimensions,
                committed_chunk_bits: config.committed_chunk_bits,
                bytecode_table: Some(&bytecode_context.rows),
                entry_bytecode_index: bytecode_context.entry_bytecode_index,
                bytecode_r_address: reversed(&self.bytecode_address_challenges),
                booleanity_r_address: reversed(&self.booleanity_address_challenges),
                address_bytecode_read_raf: self.address_phase.bytecode_read_raf,
                address_booleanity: self.address_phase.booleanity,
                address_val_stages: self
                    .address_phase
                    .bytecode_val_stages
                    .map_or_else(Vec::new, |stages| stages.to_vec()),
                bytecode_gamma: challenges.bytecode_gamma_powers[1],
                instruction_ra_gamma: challenges.instruction_ra_gamma,
                inc_gamma: challenges.inc_gamma,
                booleanity_gamma: challenges.booleanity_gamma,
                eta: None,
                stage_cycle_points,
                register_read_write_point: register_points.register_read_write_address.to_vec(),
                register_val_evaluation_point: register_points
                    .register_val_evaluation_address
                    .to_vec(),
                stage_gammas: [
                    challenges.stage1_gammas.clone(),
                    challenges.stage2_gammas.clone(),
                    challenges.stage3_gammas.clone(),
                    challenges.stage4_gammas.clone(),
                    challenges.stage5_gammas.clone(),
                ],
                booleanity_reference_address: challenges.booleanity_reference.address.clone(),
                booleanity_reference_cycle: challenges.booleanity_reference.cycle.clone(),
                stage1_cycle_binding: stage6_stage1_cycle_binding(self.stage1)
                    .map_err(invalid_stage_request)?
                    .to_vec(),
                ram_reduced_address: ram_reduced.address.to_vec(),
                ram_reduced_cycle: ram_reduced.cycle.to_vec(),
                instruction_r_address: instruction_read_raf.address.to_vec(),
                instruction_r_cycle: instruction_read_raf.cycle.to_vec(),
                inc_cycle_points: [
                    inc_cycles.ram_read_write_cycle.to_vec(),
                    inc_cycles.ram_val_check_cycle.to_vec(),
                    inc_cycles.registers_read_write_cycle.to_vec(),
                    inc_cycles.registers_val_evaluation_cycle.to_vec(),
                ],
                trusted_advice_layout: config.trusted_advice_layout.as_ref(),
                untrusted_advice_layout: config.untrusted_advice_layout.as_ref(),
                bytecode_reduction_layout: None,
                program_image_reduction_layout: None,
                bytecode_reduction_weights: None,
                program_image_r_addr_rw: Vec::new(),
            },
            self.stage2,
            self.stage4,
            self.stage5,
        )
        .map_err(invalid_stage_request)
    }

    pub(super) fn cycle_input_claims(&self) -> Stage6BatchInputClaims<F> {
        let mut input_claims = self.prefix.input_claims.clone();
        input_claims.bytecode_read_raf = self.address_phase.bytecode_read_raf;
        input_claims.booleanity = self.address_phase.booleanity;
        input_claims
    }

    pub(super) fn materialize_relation(
        &self,
        instance: &Stage6BatchInstance<F>,
    ) -> Result<Stage6RelationState<F>, ProverError> {
        let relation = self.materialize_relation_state(instance)?;
        let input_sum = relation.round_sum(0, F::zero())? + relation.round_sum(0, F::one())?;
        if input_sum != instance.input_claim {
            return Err(invalid_sumcheck_output(format!(
                "Stage 6 instance {:?} materialized sum does not match input claim: expected {}, got {}",
                instance.kind, instance.input_claim, input_sum
            )));
        }
        Ok(relation)
    }

    fn materialize_relation_state(
        &self,
        instance: &Stage6BatchInstance<F>,
    ) -> Result<Stage6RelationState<F>, ProverError> {
        // Only the advice cycle-phase relations are materialized in-crate; every
        // other Stage 6 relation's round polynomials are produced by jolt-backends.
        match instance.kind {
            Stage6InstanceKind::AdviceCyclePhase(kind) => {
                self.materialize_advice_relation(instance, kind)
            }
            other => Err(invalid_stage_request(format!(
                "Stage 6 {other:?} relation is backend-owned, not materialized in-crate"
            ))),
        }
    }

    fn materialize_advice_relation(
        &self,
        _instance: &Stage6BatchInstance<F>,
        kind: JoltAdviceKind,
    ) -> Result<Stage6RelationState<F>, ProverError> {
        let layout = self.advice_layout(kind).ok_or_else(|| {
            invalid_stage_request(format!("Stage 6 {kind:?} advice layout is missing"))
        })?;
        let reference = self.advice_cycle_phase_reference(kind)?.opening_point;
        let values = materialize_advice_values(self.witness, kind)?;
        let (advice, eq) = layout
            .cycle_phase_polynomials(reference, values)
            .map_err(|error| {
                invalid_stage_request(format!(
                    "Stage 6 {kind:?} advice cycle-phase polynomials are invalid: {error}"
                ))
            })?;
        Ok(Stage6RelationState::advice(
            AdviceCyclePhaseRelationState::new(
                advice,
                eq,
                layout.cycle_phase_col_rounds(),
                layout.cycle_phase_row_rounds(),
            ),
        ))
    }

    /// The per-instance expected output claims, single-sourced through the same
    /// [`Stage6Relations`] bundle the verifier uses: each relation derives its
    /// opening points from the 6b cycle instance point and evaluates its output
    /// `Expr` against the produced openings — mirroring the clear verifier exactly.
    /// The stage-6b cycle batch's expected output claims AND the produced opening
    /// *points*, both single-sourced through the [`Stage6Relations`] bundle the
    /// verifier uses: each relation derives its opening points once, which feed
    /// both the output-claim check and the `Stage6ClearOutput::output_points` the
    /// prover hands to stages 7/8. Mirrors the clear verifier exactly.
    pub(super) fn cycle_outputs(
        &self,
        sumcheck_point: &[F],
        openings: &Stage6OutputClaims<F>,
    ) -> Result<(Stage6BatchExpectedOutputClaims<F>, Stage6OutputClaims<Vec<F>>), ProverError> {
        let relations = self.cycle_relations()?;
        let algebra =
            |error: jolt_verifier::VerifierError| invalid_sumcheck_output(error.to_string());

        let bytecode_point =
            self.instance_point(sumcheck_point, Stage6InstanceKind::BytecodeReadRaf)?;
        let bytecode_derived = relations
            .bytecode_read_raf
            .derive_opening_points(&bytecode_point, &relations.bytecode_read_raf_inputs)
            .map_err(algebra)?;
        let bytecode_read_raf = relations
            .bytecode_read_raf
            .expected_output(
                &relations.bytecode_read_raf_inputs,
                &zip_openings(&openings.bytecode_read_raf, &bytecode_derived),
            )
            .map_err(algebra)?;

        let booleanity_point =
            self.instance_point(sumcheck_point, Stage6InstanceKind::Booleanity)?;
        let booleanity_derived = relations
            .booleanity
            .derive_opening_points(&booleanity_point, &relations.booleanity_inputs)
            .map_err(algebra)?;
        let booleanity = relations
            .booleanity
            .expected_output(
                &relations.booleanity_inputs,
                &zip_openings(&openings.booleanity, &booleanity_derived),
            )
            .map_err(algebra)?;

        let ram_hamming_point =
            self.instance_point(sumcheck_point, Stage6InstanceKind::RamHammingBooleanity)?;
        let ram_hamming_derived = relations
            .ram_hamming
            .derive_opening_points(&ram_hamming_point, &relations.ram_hamming_inputs)
            .map_err(algebra)?;
        let ram_hamming_booleanity = relations
            .ram_hamming
            .expected_output(
                &relations.ram_hamming_inputs,
                &zip_openings(&openings.ram_hamming_booleanity, &ram_hamming_derived),
            )
            .map_err(algebra)?;

        let ram_ra_point =
            self.instance_point(sumcheck_point, Stage6InstanceKind::RamRaVirtualization)?;
        let ram_ra_derived = relations
            .ram_ra
            .derive_opening_points(&ram_ra_point, &relations.ram_ra_inputs)
            .map_err(algebra)?;
        let ram_ra_virtualization = relations
            .ram_ra
            .expected_output(
                &relations.ram_ra_inputs,
                &zip_openings(&openings.ram_ra_virtualization, &ram_ra_derived),
            )
            .map_err(algebra)?;

        let instruction_ra_point = self.instance_point(
            sumcheck_point,
            Stage6InstanceKind::InstructionRaVirtualization,
        )?;
        let instruction_ra_derived = relations
            .instruction_ra
            .derive_opening_points(&instruction_ra_point, &relations.instruction_ra_inputs)
            .map_err(algebra)?;
        let instruction_ra_virtualization = relations
            .instruction_ra
            .expected_output(
                &relations.instruction_ra_inputs,
                &zip_openings(
                    &openings.instruction_ra_virtualization,
                    &instruction_ra_derived,
                ),
            )
            .map_err(algebra)?;

        let inc_point =
            self.instance_point(sumcheck_point, Stage6InstanceKind::IncClaimReduction)?;
        let inc_derived = relations
            .inc
            .derive_opening_points(&inc_point, &relations.inc_inputs)
            .map_err(algebra)?;
        let inc_claim_reduction = relations
            .inc
            .expected_output(
                &relations.inc_inputs,
                &zip_openings(&openings.inc_claim_reduction, &inc_derived),
            )
            .map_err(algebra)?;

        // Each advice kind yields both its expected output claim and its produced
        // opening point (for `output_points`).
        let advice = |kind: JoltAdviceKind,
                      relation: Option<&jolt_verifier::stages::stage6::AdviceCyclePhase<F>>|
         -> Result<Option<(F, Vec<F>)>, ProverError> {
            let claim = match kind {
                JoltAdviceKind::Trusted => openings.advice_cycle_phase.trusted.as_ref(),
                JoltAdviceKind::Untrusted => openings.advice_cycle_phase.untrusted.as_ref(),
            };
            match (relation, claim) {
                (Some(relation), Some(claim)) => {
                    let point = self.instance_point(
                        sumcheck_point,
                        Stage6InstanceKind::AdviceCyclePhase(kind),
                    )?;
                    let derived = relation
                        .derive_opening_points(&point, &relations.advice_inputs)
                        .map_err(algebra)?;
                    let opening_point = match kind {
                        JoltAdviceKind::Trusted => derived.trusted.clone(),
                        JoltAdviceKind::Untrusted => derived.untrusted.clone(),
                    }
                    .ok_or_else(|| {
                        invalid_sumcheck_output("Stage 6 advice cycle phase produced no opening")
                    })?;
                    let values = match kind {
                        JoltAdviceKind::Trusted => AdviceCyclePhaseOutputClaims {
                            trusted: Some(claim.opening_claim),
                            untrusted: None,
                        },
                        JoltAdviceKind::Untrusted => AdviceCyclePhaseOutputClaims {
                            trusted: None,
                            untrusted: Some(claim.opening_claim),
                        },
                    };
                    let expected = relation
                        .expected_output(&relations.advice_inputs, &zip_openings(&values, &derived))
                        .map_err(algebra)?;
                    Ok(Some((expected, opening_point)))
                }
                (None, None) => Ok(None),
                _ => Err(invalid_stage_request(format!(
                    "Stage 6 {kind:?} advice relation/opening presence mismatch"
                ))),
            }
        };
        let trusted_advice = advice(JoltAdviceKind::Trusted, relations.trusted_advice.as_ref())?;
        let untrusted_advice =
            advice(JoltAdviceKind::Untrusted, relations.untrusted_advice.as_ref())?;

        let reversed = |challenges: &[F]| challenges.iter().rev().copied().collect::<Vec<_>>();
        let output_points = Stage6OutputClaims {
            address_phase: Stage6AddressPhaseClaims {
                bytecode_read_raf: reversed(&self.bytecode_address_challenges),
                booleanity: reversed(&self.booleanity_address_challenges),
                // The modular prover only supports full programs (no staged Val columns).
                bytecode_val_stages: None,
            },
            bytecode_read_raf: bytecode_derived,
            booleanity: booleanity_derived,
            ram_hamming_booleanity: ram_hamming_derived,
            ram_ra_virtualization: ram_ra_derived,
            instruction_ra_virtualization: instruction_ra_derived,
            inc_claim_reduction: inc_derived,
            advice_cycle_phase: Stage6AdviceCyclePhaseClaims {
                trusted: trusted_advice
                    .as_ref()
                    .map(|(_, point)| AdviceCyclePhaseOutputClaim {
                        opening_claim: point.clone(),
                    }),
                untrusted: untrusted_advice
                    .as_ref()
                    .map(|(_, point)| AdviceCyclePhaseOutputClaim {
                        opening_claim: point.clone(),
                    }),
            },
            // Committed-program reductions; the modular prover is full-only.
            bytecode_claim_reduction: None,
            program_image_claim_reduction: None,
        };
        let expected = Stage6BatchExpectedOutputClaims {
            bytecode_read_raf,
            booleanity,
            ram_hamming_booleanity,
            ram_ra_virtualization,
            instruction_ra_virtualization,
            inc_claim_reduction,
            trusted_advice_cycle_phase: trusted_advice.map(|(expected, _)| expected),
            untrusted_advice_cycle_phase: untrusted_advice.map(|(expected, _)| expected),
        };
        Ok((expected, output_points))
    }

    /// The advice cycle-phase produced opening point for `kind` (the reversed
    /// active-cycle challenges), used to evaluate the advice witness opening
    /// before the output claims are assembled. Matches the `output_points` cell
    /// `cycle_outputs` later produces.
    pub(super) fn advice_cycle_phase_opening_point(
        &self,
        sumcheck_point: &[F],
        kind: JoltAdviceKind,
    ) -> Result<Option<Vec<F>>, ProverError> {
        let Some(layout) = self.advice_layout(kind) else {
            return Ok(None);
        };
        if self
            .instances
            .iter()
            .all(|instance| instance.kind != Stage6InstanceKind::AdviceCyclePhase(kind))
        {
            return Ok(None);
        }
        let point = self.instance_point(sumcheck_point, Stage6InstanceKind::AdviceCyclePhase(kind))?;
        Ok(Some(
            layout
                .cycle_phase_opening_point(&point)
                .map_err(|error| invalid_sumcheck_output(error.to_string()))?,
        ))
    }

    pub(super) fn instance(
        &self,
        kind: Stage6InstanceKind,
    ) -> Result<&Stage6BatchInstance<F>, ProverError> {
        self.instances
            .iter()
            .find(|instance| instance.kind == kind)
            .ok_or_else(|| invalid_stage_request(format!("Stage 6 instance {kind:?} is missing")))
    }

    fn instance_point(
        &self,
        sumcheck_point: &[F],
        kind: Stage6InstanceKind,
    ) -> Result<Vec<F>, ProverError> {
        let instance = self.instance(kind)?;
        let end = instance.offset + instance.num_vars;
        sumcheck_point
            .get(instance.offset..end)
            .map(<[F]>::to_vec)
            .ok_or_else(|| {
                invalid_sumcheck_output(format!(
                    "Stage 6 instance {:?} point range {}..{end} is out of range for {} variables",
                    kind,
                    instance.offset,
                    sumcheck_point.len()
                ))
            })
    }

    fn advice_cycle_phase_reference(
        &self,
        kind: JoltAdviceKind,
    ) -> Result<Stage6AdviceCyclePhaseReference<'_, F>, ProverError> {
        stage6_advice_cycle_phase_reference(self.stage4, kind).map_err(invalid_stage_request)
    }

    pub(super) fn advice_cycle_phase_reference_opening_point(
        &self,
        kind: JoltAdviceKind,
    ) -> Result<Option<&[F]>, ProverError> {
        if self.advice_layout(kind).is_none() {
            return Ok(None);
        }
        Ok(Some(self.advice_cycle_phase_reference(kind)?.opening_point))
    }

    fn advice_layout(&self, kind: JoltAdviceKind) -> Option<&AdviceClaimReductionLayout> {
        match kind {
            JoltAdviceKind::Trusted => self.config.trusted_advice_layout.as_ref(),
            JoltAdviceKind::Untrusted => self.config.untrusted_advice_layout.as_ref(),
        }
    }
}

fn advice_cycle_phase_offset(
    config: &Stage6ProverConfig,
    kind: JoltAdviceKind,
    max_num_vars: usize,
) -> Result<usize, ProverError> {
    let layout = match kind {
        JoltAdviceKind::Trusted => config.trusted_advice_layout.as_ref(),
        JoltAdviceKind::Untrusted => config.untrusted_advice_layout.as_ref(),
    }
    .ok_or_else(|| invalid_stage_request(format!("Stage 6 {kind:?} advice layout is missing")))?;
    layout
        .cycle_phase_batch_offset(max_num_vars)
        .map_err(|error| {
            invalid_stage_request(format!(
                "Stage 6 {kind:?} advice cycle-phase offset is invalid: {error}"
            ))
        })
}

pub(super) fn evaluate_advice_cycle_phase_opening<F, W>(
    layout: Option<&AdviceClaimReductionLayout>,
    witness: &W,
    kind: JoltAdviceKind,
    reference_opening_point: Option<&[F]>,
    opening_point: Option<&[F]>,
) -> Result<Option<AdviceCyclePhaseOutputClaim<F>>, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>,
{
    let Some(layout) = layout else {
        if reference_opening_point.is_some() || opening_point.is_some() {
            return Err(invalid_stage_request(format!(
                "Stage 6 {kind:?} advice opening was supplied without configured advice"
            )));
        }
        return Ok(None);
    };
    let reference_opening_point = reference_opening_point.ok_or_else(|| {
        invalid_stage_request(format!(
            "Stage 6 {kind:?} advice reference opening point is missing"
        ))
    })?;
    let opening_point = opening_point.ok_or_else(|| {
        invalid_stage_request(format!(
            "Stage 6 {kind:?} advice cycle-phase opening point is missing"
        ))
    })?;
    let values = materialize_advice_values(witness, kind)?;
    let opening_claim = layout
        .cycle_phase_opening_claim(reference_opening_point, values, opening_point)
        .map_err(|error| {
            invalid_stage_request(format!(
                "Stage 6 {kind:?} advice cycle-phase opening claim is invalid: {error}"
            ))
        })?;
    Ok(Some(AdviceCyclePhaseOutputClaim { opening_claim }))
}

fn materialize_advice_values<F, W>(witness: &W, kind: JoltAdviceKind) -> Result<Vec<F>, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>,
{
    let oracle = match kind {
        JoltAdviceKind::Trusted => OracleRef::committed(JoltCommittedPolynomial::TrustedAdvice),
        JoltAdviceKind::Untrusted => OracleRef::committed(JoltCommittedPolynomial::UntrustedAdvice),
    };
    let requirement = witness
        .view_requirements(oracle)?
        .into_iter()
        .next()
        .ok_or_else(|| {
            invalid_stage_request(format!(
                "witness returned no view requirement for Stage 6 {kind:?} advice"
            ))
        })?;
    let view = witness.oracle_view(requirement)?;
    view.as_slice().map(<[F]>::to_vec).ok_or_else(|| {
        invalid_stage_request(format!("Stage 6 {kind:?} advice view is not concrete"))
    })
}

fn invalid_stage_request(error: impl std::fmt::Display) -> ProverError {
    ProverError::InvalidStageRequest {
        reason: error.to_string(),
    }
}
