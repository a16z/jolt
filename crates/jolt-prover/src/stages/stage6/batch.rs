use jolt_claims::protocols::jolt::{
    formulas::{booleanity, bytecode},
    JoltCommittedPolynomial,
};
use jolt_claims::protocols::jolt::{
    formulas::{claim_reductions::advice, claim_reductions::increments, instruction, ram},
    AdviceClaimReductionLayout, JoltAdviceKind, PrecommittedReductionLayout,
};
use jolt_field::Field;
use jolt_verifier::stages::stage6::inputs::Stage6AddressPhaseClaims;
use jolt_verifier::stages::stage6::inputs::{AdviceCyclePhaseOutputClaim, Stage6Claims};
use jolt_verifier::stages::stage6::outputs::AdviceCyclePhasePublicOutput;
use jolt_verifier::stages::stage6::{
    stage6_advice_cycle_phase_expected_output, stage6_advice_cycle_phase_reference,
    stage6_batch_points, stage6_booleanity_expected_output, stage6_bytecode_cycle_points,
    stage6_bytecode_read_raf_expected_output, stage6_bytecode_register_points,
    stage6_inc_claim_reduction_cycle_points, stage6_inc_claim_reduction_expected_output,
    stage6_instruction_ra_virtualization_expected_output, stage6_instruction_read_raf_point,
    stage6_ram_hamming_booleanity_expected_output, stage6_ram_ra_virtualization_expected_output,
    stage6_stage1_cycle_binding, stage6_stage5_ram_reduced_opening_point,
    Stage6AdviceCyclePhaseReference, Stage6BatchExpectedOutputClaims, Stage6BatchInputClaims,
    Stage6BatchPointContext, Stage6BatchPointInputs, Stage6BatchPoints,
    Stage6BooleanityExpectedOutputInputs, Stage6BytecodeReadRafExpectedOutputInputs,
    Stage6IncClaimReductionExpectedOutputInputs,
    Stage6InstructionRaVirtualizationExpectedOutputInputs, Stage6InstructionReadRafPoint,
    Stage6RamRaVirtualizationExpectedOutputInputs, Stage6RamReducedOpeningPoint,
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
        })
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

    /// Builds the per-instance opening points for the stage 6b cycle batch.
    pub(super) fn derived_points(
        &self,
        sumcheck_point: &[F],
        bytecode_address_challenges: &[F],
        booleanity_address_challenges: &[F],
    ) -> Result<Stage6BatchPoints<F>, ProverError> {
        if sumcheck_point.len() != self.max_num_vars {
            return Err(invalid_sumcheck_output(format!(
                "Stage 6 batch sumcheck point has {} variables, expected {}",
                sumcheck_point.len(),
                self.max_num_vars
            )));
        }

        let bytecode_read_raf = [
            bytecode_address_challenges,
            self.instance_point(sumcheck_point, Stage6InstanceKind::BytecodeReadRaf)?
                .as_slice(),
        ]
        .concat();
        let booleanity = [
            booleanity_address_challenges,
            self.instance_point(sumcheck_point, Stage6InstanceKind::Booleanity)?
                .as_slice(),
        ]
        .concat();
        let ram_hamming_booleanity =
            self.instance_point(sumcheck_point, Stage6InstanceKind::RamHammingBooleanity)?;
        let ram_ra_virtualization =
            self.instance_point(sumcheck_point, Stage6InstanceKind::RamRaVirtualization)?;
        let instruction_ra_virtualization = self.instance_point(
            sumcheck_point,
            Stage6InstanceKind::InstructionRaVirtualization,
        )?;
        let inc_claim_reduction =
            self.instance_point(sumcheck_point, Stage6InstanceKind::IncClaimReduction)?;

        let trusted_advice_cycle_phase =
            self.advice_cycle_phase_sumcheck_point(sumcheck_point, JoltAdviceKind::Trusted)?;
        let untrusted_advice_cycle_phase =
            self.advice_cycle_phase_sumcheck_point(sumcheck_point, JoltAdviceKind::Untrusted)?;

        let ram_reduced = self.ram_reduced_opening_point()?;
        let instruction_read_raf = self.instruction_read_raf_point();
        stage6_batch_points(
            Stage6BatchPointInputs {
                bytecode_read_raf: &bytecode_read_raf,
                booleanity: &booleanity,
                ram_hamming_booleanity: &ram_hamming_booleanity,
                ram_ra_virtualization: &ram_ra_virtualization,
                instruction_ra_virtualization: &instruction_ra_virtualization,
                inc_claim_reduction: &inc_claim_reduction,
                trusted_advice_cycle_phase: trusted_advice_cycle_phase.as_deref(),
                untrusted_advice_cycle_phase: untrusted_advice_cycle_phase.as_deref(),
            },
            Stage6BatchPointContext {
                trace_dimensions: self.config.trace_dimensions(),
                bytecode_read_raf_dimensions: self.config.bytecode_read_raf_dimensions,
                booleanity_dimensions: self.config.booleanity_dimensions,
                committed_chunk_bits: self.config.committed_chunk_bits,
                ram_reduced_opening_point: ram_reduced,
                instruction_read_raf,
                trusted_advice_layout: self.config.trusted_advice_layout.as_ref(),
                untrusted_advice_layout: self.config.untrusted_advice_layout.as_ref(),
            },
        )
        .map_err(|error| invalid_sumcheck_output(error.to_string()))
    }

    fn advice_cycle_phase_sumcheck_point(
        &self,
        sumcheck_point: &[F],
        kind: JoltAdviceKind,
    ) -> Result<Option<Vec<F>>, ProverError> {
        if self
            .instances
            .iter()
            .all(|instance| instance.kind != Stage6InstanceKind::AdviceCyclePhase(kind))
        {
            return Ok(None);
        }
        let _layout = self.advice_layout(kind).ok_or_else(|| {
            invalid_stage_request(format!("Stage 6 {kind:?} advice layout is missing"))
        })?;
        self.instance_point(sumcheck_point, Stage6InstanceKind::AdviceCyclePhase(kind))
            .map(Some)
    }

    pub(super) fn expected_outputs(
        &self,
        points: &Stage6BatchPoints<F>,
        openings: &Stage6Claims<F>,
    ) -> Result<Stage6BatchExpectedOutputClaims<F>, ProverError> {
        Ok(Stage6BatchExpectedOutputClaims {
            bytecode_read_raf: self.expected_bytecode_output(
                &points.bytecode_read_raf_sumcheck_point,
                &openings.bytecode_read_raf.bytecode_ra,
            )?,
            booleanity: self.expected_booleanity_output(
                &points.booleanity_sumcheck_point,
                &openings.booleanity.instruction_ra,
                &openings.booleanity.bytecode_ra,
                &openings.booleanity.ram_ra,
            )?,
            ram_hamming_booleanity: self.expected_ram_hamming_output(
                &points.ram_hamming_booleanity_sumcheck_point,
                openings.ram_hamming_booleanity.ram_hamming_weight,
            )?,
            ram_ra_virtualization: self.expected_ram_ra_virtualization_output(
                &points.ram_ra_virtualization_sumcheck_point,
                &openings.ram_ra_virtualization.ram_ra,
            )?,
            instruction_ra_virtualization: self.expected_instruction_ra_virtualization_output(
                &points.instruction_ra_virtualization_sumcheck_point,
                &openings
                    .instruction_ra_virtualization
                    .committed_instruction_ra,
            )?,
            inc_claim_reduction: self.expected_inc_claim_reduction_output(
                &points.inc_claim_reduction_sumcheck_point,
                openings.inc_claim_reduction.ram_inc,
                openings.inc_claim_reduction.rd_inc,
            )?,
            trusted_advice_cycle_phase: self.expected_advice_output(
                JoltAdviceKind::Trusted,
                points.trusted_advice_cycle_phase.as_ref(),
                openings.advice_cycle_phase.trusted.as_ref(),
            )?,
            untrusted_advice_cycle_phase: self.expected_advice_output(
                JoltAdviceKind::Untrusted,
                points.untrusted_advice_cycle_phase.as_ref(),
                openings.advice_cycle_phase.untrusted.as_ref(),
            )?,
        })
    }

    fn expected_bytecode_output(&self, point: &[F], bytecode_ra: &[F]) -> Result<F, ProverError> {
        let public_values = self.bytecode_public_values(point)?;
        stage6_bytecode_read_raf_expected_output(Stage6BytecodeReadRafExpectedOutputInputs {
            dimensions: self.config.bytecode_read_raf_dimensions,
            public_values: &public_values,
            bytecode_ra,
            gamma: self.prefix.challenges.bytecode_gamma_powers[1],
        })
        .map_err(|error| invalid_sumcheck_output(error.to_string()))
    }

    fn expected_booleanity_output(
        &self,
        point: &[F],
        instruction_ra: &[F],
        bytecode_ra: &[F],
        ram_ra: &[F],
    ) -> Result<F, ProverError> {
        stage6_booleanity_expected_output(Stage6BooleanityExpectedOutputInputs {
            dimensions: self.config.booleanity_dimensions,
            sumcheck_point: point,
            reference: &self.prefix.challenges.booleanity_reference,
            instruction_ra,
            bytecode_ra,
            ram_ra,
            gamma: self.prefix.challenges.booleanity_gamma,
        })
        .map_err(|error| invalid_sumcheck_output(error.to_string()))
    }

    fn expected_ram_hamming_output(
        &self,
        point: &[F],
        ram_hamming_weight: F,
    ) -> Result<F, ProverError> {
        stage6_ram_hamming_booleanity_expected_output(
            point,
            stage6_stage1_cycle_binding(self.stage1).map_err(invalid_stage_request)?,
            ram_hamming_weight,
        )
        .map_err(|error| invalid_sumcheck_output(error.to_string()))
    }

    fn expected_ram_ra_virtualization_output(
        &self,
        point: &[F],
        ram_ra: &[F],
    ) -> Result<F, ProverError> {
        let r_cycle = self
            .config
            .trace_dimensions()
            .cycle_opening_point(point)
            .map_err(|error| invalid_sumcheck_output(error.to_string()))?;
        let ram_reduced = self.ram_reduced_opening_point()?;
        stage6_ram_ra_virtualization_expected_output(
            Stage6RamRaVirtualizationExpectedOutputInputs {
                dimensions: self.config.ram_ra_virtualization_dimensions,
                r_cycle: &r_cycle,
                ram_reduced_cycle: ram_reduced.cycle,
                ram_ra,
            },
        )
        .map_err(|error| invalid_sumcheck_output(error.to_string()))
    }

    fn expected_instruction_ra_virtualization_output(
        &self,
        point: &[F],
        committed_instruction_ra: &[F],
    ) -> Result<F, ProverError> {
        let r_cycle = self
            .config
            .trace_dimensions()
            .cycle_opening_point(point)
            .map_err(|error| invalid_sumcheck_output(error.to_string()))?;
        stage6_instruction_ra_virtualization_expected_output(
            Stage6InstructionRaVirtualizationExpectedOutputInputs {
                dimensions: self.config.instruction_ra_virtualization_dimensions,
                instruction_read_raf_cycle: self.instruction_read_raf_point().cycle,
                r_cycle: &r_cycle,
                committed_instruction_ra,
                gamma: self.prefix.challenges.instruction_ra_gamma,
            },
        )
        .map_err(|error| invalid_sumcheck_output(error.to_string()))
    }

    fn expected_inc_claim_reduction_output(
        &self,
        point: &[F],
        ram_inc: F,
        rd_inc: F,
    ) -> Result<F, ProverError> {
        let opening_point = self
            .config
            .trace_dimensions()
            .cycle_opening_point(point)
            .map_err(|error| invalid_sumcheck_output(error.to_string()))?;
        let cycles = stage6_inc_claim_reduction_cycle_points(
            self.stage2,
            self.stage4,
            self.stage5,
            self.config.log_k,
        )
        .map_err(|error| invalid_sumcheck_output(error.to_string()))?;
        stage6_inc_claim_reduction_expected_output(Stage6IncClaimReductionExpectedOutputInputs {
            opening_point: &opening_point,
            ram_read_write_cycle: cycles.ram_read_write_cycle,
            ram_val_check_cycle: cycles.ram_val_check_cycle,
            registers_read_write_cycle: cycles.registers_read_write_cycle,
            registers_val_evaluation_cycle: cycles.registers_val_evaluation_cycle,
            ram_inc,
            rd_inc,
            gamma: self.prefix.challenges.inc_gamma,
        })
        .map_err(|error| invalid_sumcheck_output(error.to_string()))
    }

    fn expected_advice_output(
        &self,
        kind: JoltAdviceKind,
        point: Option<&AdviceCyclePhasePublicOutput<F>>,
        opening_claim: Option<&AdviceCyclePhaseOutputClaim<F>>,
    ) -> Result<Option<F>, ProverError> {
        match (point, opening_claim) {
            (Some(point), Some(opening_claim)) => {
                let layout = self.advice_layout(kind).ok_or_else(|| {
                    invalid_stage_request(format!("Stage 6 {kind:?} advice layout is missing"))
                })?;
                let reference = self.advice_cycle_phase_reference(kind)?.opening_point;
                let output = stage6_advice_cycle_phase_expected_output(
                    layout,
                    kind,
                    reference,
                    &point.sumcheck_point,
                    opening_claim.opening_claim,
                )
                .map_err(|error| invalid_sumcheck_output(error.to_string()))?;
                Ok(Some(output))
            }
            (None, None) => Ok(None),
            _ => Err(invalid_stage_request(format!(
                "Stage 6 {kind:?} advice point/opening presence mismatch"
            ))),
        }
    }

    fn bytecode_public_values(
        &self,
        point: &[F],
    ) -> Result<bytecode::BytecodeReadRafPublicValues<F>, ProverError> {
        let context = self.config.bytecode_context.as_ref().ok_or_else(|| {
            invalid_stage_request("Stage 6 bytecode context is required for read-RAF evaluation")
        })?;
        let opening = self
            .config
            .bytecode_read_raf_dimensions
            .opening_point(point)
            .map_err(|error| invalid_sumcheck_output(error.to_string()))?;
        let stage_cycles = stage6_bytecode_cycle_points(
            self.stage1,
            self.stage2,
            self.stage3,
            self.stage4,
            self.stage5,
        )
        .map_err(invalid_stage_request)?;
        let register_points = stage6_bytecode_register_points(self.stage4, self.stage5)
            .map_err(invalid_stage_request)?;
        let stage_cycle_points = [
            stage_cycles[0].as_slice(),
            stage_cycles[1].as_slice(),
            stage_cycles[2].as_slice(),
            stage_cycles[3].as_slice(),
            stage_cycles[4].as_slice(),
        ];
        let bytecode_rows = context.rows.as_slice();

        let public_values = if let Some(public_values) =
            bytecode::read_raf_public_values_at_boolean_point::<F>(
                bytecode::BytecodeReadRafBooleanEvaluationInputs {
                    bytecode: bytecode_rows,
                    r_address: &opening.r_address,
                    r_cycle: &opening.r_cycle,
                    stage_cycle_points,
                    register_read_write_point: register_points.register_read_write_address,
                    register_val_evaluation_point: register_points.register_val_evaluation_address,
                    entry_bytecode_index: context.entry_bytecode_index,
                    stage1_gammas: &self.prefix.challenges.stage1_gammas,
                    stage2_gammas: &self.prefix.challenges.stage2_gammas,
                    stage3_gammas: &self.prefix.challenges.stage3_gammas,
                    stage4_gammas: &self.prefix.challenges.stage4_gammas,
                    stage5_gammas: &self.prefix.challenges.stage5_gammas,
                },
            )
            .map_err(|error| invalid_sumcheck_output(error.to_string()))?
        {
            public_values
        } else {
            bytecode::read_raf_public_values::<F>(bytecode::BytecodeReadRafEvaluationInputs {
                bytecode: bytecode_rows,
                r_address: &opening.r_address,
                r_cycle: &opening.r_cycle,
                stage_cycle_points,
                register_read_write_point: register_points.register_read_write_address,
                register_val_evaluation_point: register_points.register_val_evaluation_address,
                entry_bytecode_index: context.entry_bytecode_index,
                stage1_gammas: &self.prefix.challenges.stage1_gammas,
                stage2_gammas: &self.prefix.challenges.stage2_gammas,
                stage3_gammas: &self.prefix.challenges.stage3_gammas,
                stage4_gammas: &self.prefix.challenges.stage4_gammas,
                stage5_gammas: &self.prefix.challenges.stage5_gammas,
            })
            .map_err(|error| invalid_sumcheck_output(error.to_string()))?
        };

        Ok(public_values)
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

    pub(super) fn ram_reduced_opening_point(
        &self,
    ) -> Result<Stage6RamReducedOpeningPoint<'_, F>, ProverError> {
        stage6_stage5_ram_reduced_opening_point(self.stage5, self.config.log_k, self.config.log_t)
            .map_err(invalid_stage_request)
    }

    fn instruction_read_raf_point(&self) -> Stage6InstructionReadRafPoint<'_, F> {
        stage6_instruction_read_raf_point(self.stage5)
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
