#[cfg(feature = "field-inline")]
use jolt_claims::protocols::field_inline::{
    formulas::{bytecode as field_bytecode, claim_reductions::increments as field_increments},
    FieldRegistersTraceDimensions,
};
use jolt_claims::protocols::jolt::{
    formulas::{booleanity, bytecode},
    JoltCommittedPolynomial, JoltOpeningId,
};
use jolt_claims::protocols::jolt::{
    formulas::{claim_reductions::advice, claim_reductions::increments, instruction, ram},
    AdviceClaimReductionLayout, JoltAdviceKind,
};
#[cfg(feature = "zk")]
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_poly::UnivariatePoly;
use jolt_poly::{boolean_index_msb, boolean_point_msb, eq_index_msb, Polynomial};
#[cfg(feature = "field-inline")]
use jolt_riscv::JoltInstructionRow;
use jolt_sumcheck::{
    append_sumcheck_claim, ClearProof, CompressedLabeledRoundPoly, CompressedSumcheckProof,
    RoundMessage, SumcheckProof,
};
use jolt_transcript::Transcript;
use jolt_verifier::stages::stage6::inputs::{AdviceCyclePhaseOutputClaim, Stage6Claims};
use jolt_verifier::stages::stage6::outputs::AdviceCyclePhasePublicOutput;
#[cfg(feature = "zk")]
use jolt_verifier::stages::stage6::stage6_output_claim_values;
use jolt_verifier::stages::stage6::{
    append_stage6_opening_claims, stage6_advice_cycle_phase_expected_output,
    stage6_advice_cycle_phase_reference, stage6_batch_points, stage6_booleanity_expected_output,
    stage6_bytecode_cycle_points, stage6_bytecode_ra_point,
    stage6_bytecode_read_raf_expected_output, stage6_bytecode_read_raf_output_coefficient,
    stage6_bytecode_read_raf_point, stage6_bytecode_register_points,
    stage6_inc_claim_reduction_cycle_points, stage6_inc_claim_reduction_expected_output,
    stage6_instruction_ra_virtualization_expected_output, stage6_instruction_read_raf_point,
    stage6_ram_hamming_booleanity_expected_output, stage6_ram_ra_virtualization_expected_output,
    stage6_stage1_cycle_binding, stage6_stage5_ram_reduced_opening_point,
    stage6_validate_dependencies, Stage6AdviceCyclePhaseReference, Stage6BatchExpectedOutputClaims,
    Stage6BatchPointContext, Stage6BatchPointInputs, Stage6BatchPoints,
    Stage6BooleanityExpectedOutputInputs, Stage6BytecodeRaPoint,
    Stage6BytecodeReadRafExpectedOutputInputs, Stage6BytecodeReadRafOutputCoefficientInputs,
    Stage6IncClaimReductionExpectedOutputInputs,
    Stage6InstructionRaVirtualizationExpectedOutputInputs, Stage6InstructionReadRafPoint,
    Stage6RamRaVirtualizationExpectedOutputInputs, Stage6RamReducedOpeningPoint,
};
#[cfg(feature = "field-inline")]
use jolt_verifier::stages::stage6::{
    stage6_field_inline_bytecode_register_points,
    stage6_field_registers_inc_claim_reduction_cycle_points,
    stage6_field_registers_inc_claim_reduction_expected_output,
    FieldInlineStage6IncClaimReductionExpectedOutputInputs,
};
use jolt_verifier::stages::{
    stage1::Stage1ClearOutput, stage2::Stage2ClearOutput, stage3::Stage3ClearOutput,
    stage4::Stage4ClearOutput, stage5::Stage5ClearOutput,
};
use jolt_witness::protocols::jolt_vm::{jolt_opening_oracle_ref, JoltVmNamespace};
use jolt_witness::{OracleRef, OracleViewRequest, WitnessProvider};
use std::{cell::RefCell, collections::HashMap, marker::PhantomData};

use super::io::{Stage6ProverConfig, Stage6RegularBatchPrefixOutput};
#[cfg(feature = "zk")]
use crate::committed::{CommittedSumcheckBuilder, CommittedSumcheckWitness};
use crate::ProverError;

pub(super) struct Stage6ProofComponents<F: Field, C> {
    pub(super) proof: SumcheckProof<F, C>,
    #[cfg(feature = "zk")]
    pub(super) committed_witness: Option<CommittedSumcheckWitness<F>>,
    #[cfg(feature = "zk")]
    pub(super) output_claim_values: Option<Vec<F>>,
}

pub(super) trait Stage6ProofRecorder<F: Field> {
    type Commitment;

    fn absorb_input_claims<T>(&mut self, input_claims: &[F], transcript: &mut T)
    where
        T: Transcript<Challenge = F>;

    fn absorb_round<T>(
        &mut self,
        round_poly: &UnivariatePoly<F>,
        transcript: &mut T,
    ) -> Result<F, ProverError>
    where
        T: Transcript<Challenge = F>;

    fn finish<T>(
        self,
        output_openings: &Stage6Claims<F>,
        transcript: &mut T,
    ) -> Result<Stage6ProofComponents<F, Self::Commitment>, ProverError>
    where
        T: Transcript<Challenge = F>;
}

pub(super) struct ClearStage6ProofRecorder<F: Field, C> {
    round_polynomials: Vec<jolt_poly::CompressedPoly<F>>,
    _marker: PhantomData<C>,
}

impl<F, C> ClearStage6ProofRecorder<F, C>
where
    F: Field,
{
    pub(super) fn new(round_capacity: usize) -> Self {
        Self {
            round_polynomials: Vec::with_capacity(round_capacity),
            _marker: PhantomData,
        }
    }
}

impl<F, C> Stage6ProofRecorder<F> for ClearStage6ProofRecorder<F, C>
where
    F: Field,
{
    type Commitment = C;

    fn absorb_input_claims<T>(&mut self, input_claims: &[F], transcript: &mut T)
    where
        T: Transcript<Challenge = F>,
    {
        for input_claim in input_claims {
            append_sumcheck_claim(transcript, input_claim);
        }
    }

    fn absorb_round<T>(
        &mut self,
        round_poly: &UnivariatePoly<F>,
        transcript: &mut T,
    ) -> Result<F, ProverError>
    where
        T: Transcript<Challenge = F>,
    {
        CompressedLabeledRoundPoly::sumcheck(round_poly).append_to_transcript(transcript);
        let challenge = transcript.challenge();
        self.round_polynomials.push(round_poly.compress());
        Ok(challenge)
    }

    fn finish<T>(
        self,
        output_openings: &Stage6Claims<F>,
        transcript: &mut T,
    ) -> Result<Stage6ProofComponents<F, Self::Commitment>, ProverError>
    where
        T: Transcript<Challenge = F>,
    {
        append_stage6_opening_claims(transcript, output_openings);
        Ok(Stage6ProofComponents {
            proof: SumcheckProof::Clear(ClearProof::Compressed(CompressedSumcheckProof {
                round_polynomials: self.round_polynomials,
            })),
            #[cfg(feature = "zk")]
            committed_witness: None,
            #[cfg(feature = "zk")]
            output_claim_values: None,
        })
    }
}

#[cfg(feature = "zk")]
pub(super) struct CommittedStage6ProofRecorder<'a, F, VC>
where
    F: Field,
    VC: VectorCommitment<Field = F>,
{
    builder: CommittedSumcheckBuilder<'a, F, VC>,
}

#[cfg(feature = "zk")]
impl<'a, F, VC> CommittedStage6ProofRecorder<'a, F, VC>
where
    F: Field,
    VC: VectorCommitment<Field = F>,
{
    pub(super) fn new(setup: &'a VC::Setup) -> Result<Self, ProverError> {
        Ok(Self {
            builder: CommittedSumcheckBuilder::new(setup, 0)?,
        })
    }
}

#[cfg(feature = "zk")]
impl<F, VC> Stage6ProofRecorder<F> for CommittedStage6ProofRecorder<'_, F, VC>
where
    F: Field,
    VC: VectorCommitment<Field = F>,
{
    type Commitment = VC::Output;

    fn absorb_input_claims<T>(&mut self, _input_claims: &[F], _transcript: &mut T)
    where
        T: Transcript<Challenge = F>,
    {
    }

    fn absorb_round<T>(
        &mut self,
        round_poly: &UnivariatePoly<F>,
        transcript: &mut T,
    ) -> Result<F, ProverError>
    where
        T: Transcript<Challenge = F>,
    {
        self.builder.commit_round(round_poly, transcript)
    }

    fn finish<T>(
        self,
        output_openings: &Stage6Claims<F>,
        transcript: &mut T,
    ) -> Result<Stage6ProofComponents<F, Self::Commitment>, ProverError>
    where
        T: Transcript<Challenge = F>,
    {
        let output_claim_values = stage6_output_claim_values(output_openings);
        let built = self.builder.finish(&output_claim_values, transcript)?;
        Ok(Stage6ProofComponents {
            proof: built.proof,
            committed_witness: Some(built.witness),
            output_claim_values: Some(output_claim_values),
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Stage6InstanceKind {
    BytecodeReadRaf,
    Booleanity,
    RamHammingBooleanity,
    RamRaVirtualization,
    InstructionRaVirtualization,
    IncClaimReduction,
    #[cfg(feature = "field-inline")]
    FieldRegistersIncClaimReduction,
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
    oracle_cache: RefCell<HashMap<OracleRef<JoltVmNamespace>, Polynomial<F>>>,
    pub(super) max_num_vars: usize,
}

pub(super) enum Stage6RelationState<F: Field> {
    BytecodeReadRaf(bytecode::BytecodeReadRafRelationState<F>),
    Booleanity(booleanity::BooleanityRelationState<F>),
    RamHammingBooleanity(ram::RamHammingBooleanityRelationState<F>),
    RamRaVirtualization(ram::RamRaVirtualizationRelationState<F>),
    InstructionRaVirtualization(instruction::InstructionRaVirtualizationRelationState<F>),
    IncClaimReduction(increments::IncClaimReductionRelationState<F>),
    AdviceCyclePhase(advice::AdviceCyclePhaseRelationState<F>),
}

struct Stage6InstanceSpec<F: Field> {
    kind: Stage6InstanceKind,
    input_claim: F,
    num_vars: usize,
    degree: usize,
}

type Stage6RaValueGroups<F> = (Vec<F>, Vec<F>, Vec<F>);

impl<F: Field> Stage6RelationState<F> {
    pub(super) fn round_sum(&self, local_round: usize, point: F) -> Result<F, ProverError> {
        let rows = self.round_rows(local_round);
        let mut sum = F::zero();
        for index in 0..rows {
            sum += self.round_eval(local_round, index, point)?;
        }
        Ok(sum)
    }

    pub(super) fn round_degree(&self, local_round: usize, fallback: usize) -> usize {
        match self {
            Self::BytecodeReadRaf(bytecode_state) => {
                bytecode_state.round_degree(local_round, fallback)
            }
            _ => fallback,
        }
    }

    pub(super) fn bind(&mut self, local_round: usize, challenge: F) {
        match self {
            Self::BytecodeReadRaf(bytecode_state) => bytecode_state.bind(local_round, challenge),
            Self::Booleanity(booleanity_state) => booleanity_state.bind(challenge),
            Self::RamHammingBooleanity(ram_state) => ram_state.bind(challenge),
            Self::RamRaVirtualization(ram_state) => ram_state.bind(challenge),
            Self::InstructionRaVirtualization(instruction_state) => {
                instruction_state.bind(challenge);
            }
            Self::IncClaimReduction(increment_state) => increment_state.bind(challenge),
            Self::AdviceCyclePhase(advice_state) => advice_state.bind(local_round, challenge),
        }
    }

    fn round_rows(&self, local_round: usize) -> usize {
        match self {
            Self::BytecodeReadRaf(bytecode_state) => bytecode_state.round_rows(local_round),
            Self::Booleanity(booleanity_state) => booleanity_state.round_rows(),
            Self::RamHammingBooleanity(ram_state) => ram_state.round_rows(),
            Self::RamRaVirtualization(ram_state) => ram_state.round_rows(),
            Self::InstructionRaVirtualization(instruction_state) => instruction_state.round_rows(),
            Self::IncClaimReduction(increment_state) => increment_state.round_rows(),
            Self::AdviceCyclePhase(advice_state) => advice_state.round_rows(local_round),
        }
    }

    fn round_eval(&self, local_round: usize, index: usize, point: F) -> Result<F, ProverError> {
        match self {
            Self::BytecodeReadRaf(bytecode_state) => {
                Ok(bytecode_state.round_eval(local_round, index, point))
            }
            Self::Booleanity(booleanity_state) => Ok(booleanity_state.round_eval(index, point)),
            Self::RamHammingBooleanity(ram_state) => Ok(ram_state.round_eval(index, point)),
            Self::RamRaVirtualization(ram_state) => Ok(ram_state.round_eval(index, point)),
            Self::InstructionRaVirtualization(instruction_state) => {
                Ok(instruction_state.round_eval(index, point))
            }
            Self::IncClaimReduction(increment_state) => {
                Ok(increment_state.round_eval(index, point))
            }
            Self::AdviceCyclePhase(advice_state) => {
                Ok(advice_state.round_eval(local_round, index, point))
            }
        }
    }
}

impl<'a, F, W> Stage6BatchContext<'a, F, W>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>,
{
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
    ) -> Result<Self, ProverError> {
        stage6_validate_dependencies(stage3).map_err(invalid_stage_request)?;

        let bytecode_claims = bytecode::read_raf::<F>(config.bytecode_read_raf_dimensions);
        let booleanity_claims = booleanity::booleanity::<F>(config.booleanity_dimensions);
        let ram_hamming_claims = ram::hamming_booleanity::<F>(config.trace_dimensions());
        let ram_ra_claims = ram::ra_virtualization::<F>(config.ram_ra_virtualization_dimensions);
        let instruction_ra_claims =
            instruction::ra_virtualization::<F>(config.instruction_ra_virtualization_dimensions);
        let inc_claims = increments::claim_reduction::<F>(config.trace_dimensions());
        #[cfg(feature = "field-inline")]
        let field_inc_claims = field_increments::claim_reduction::<F>(
            FieldRegistersTraceDimensions::new(config.log_t),
        );

        let mut specs = vec![
            Stage6InstanceSpec {
                kind: Stage6InstanceKind::BytecodeReadRaf,
                input_claim: prefix.input_claims.bytecode_read_raf,
                num_vars: bytecode_claims.sumcheck.rounds,
                degree: bytecode_claims.sumcheck.degree,
            },
            Stage6InstanceSpec {
                kind: Stage6InstanceKind::Booleanity,
                input_claim: prefix.input_claims.booleanity,
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
        #[cfg(feature = "field-inline")]
        specs.push(Stage6InstanceSpec {
            kind: Stage6InstanceKind::FieldRegistersIncClaimReduction,
            input_claim: prefix.input_claims.field_registers_inc_claim_reduction,
            num_vars: field_inc_claims.sumcheck.rounds,
            degree: field_inc_claims.sumcheck.degree,
        });

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
            oracle_cache: RefCell::new(HashMap::new()),
            max_num_vars,
        })
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
        match instance.kind {
            Stage6InstanceKind::BytecodeReadRaf => self.materialize_bytecode_relation(instance),
            Stage6InstanceKind::Booleanity => self.materialize_booleanity_relation(instance),
            Stage6InstanceKind::RamHammingBooleanity => {
                self.materialize_ram_hamming_relation(instance)
            }
            Stage6InstanceKind::RamRaVirtualization => self.materialize_ram_ra_relation(instance),
            Stage6InstanceKind::InstructionRaVirtualization => {
                self.materialize_instruction_ra_relation(instance)
            }
            Stage6InstanceKind::IncClaimReduction => self.materialize_inc_relation(instance),
            #[cfg(feature = "field-inline")]
            Stage6InstanceKind::FieldRegistersIncClaimReduction => Err(invalid_stage_request(
                "Stage 6 field-register increment relation is backend-owned",
            )),
            Stage6InstanceKind::AdviceCyclePhase(kind) => {
                self.materialize_advice_relation(instance, kind)
            }
        }
    }

    fn relation_rows(instance: &Stage6BatchInstance<F>) -> Result<usize, ProverError> {
        1usize.checked_shl(instance.num_vars as u32).ok_or_else(|| {
            invalid_sumcheck_output(format!(
                "Stage 6 instance {:?} materialization row count overflowed",
                instance.kind
            ))
        })
    }

    fn materialize_values(
        instance: &Stage6BatchInstance<F>,
        mut evaluate: impl FnMut(&[F]) -> Result<F, ProverError>,
    ) -> Result<Polynomial<F>, ProverError> {
        let rows = Self::relation_rows(instance)?;
        let mut values = Vec::with_capacity(rows);
        for index in 0..rows {
            values.push(evaluate(&boolean_point_msb(instance.num_vars, index))?);
        }
        Ok(Polynomial::new(values))
    }

    fn materialize_bytecode_relation(
        &self,
        instance: &Stage6BatchInstance<F>,
    ) -> Result<Stage6RelationState<F>, ProverError> {
        let rows = Self::relation_rows(instance)?;
        let ra_count = self
            .config
            .bytecode_read_raf_dimensions
            .num_committed_ra_polys();
        let mut public_coeff = Vec::with_capacity(rows);
        let mut bytecode_ra = (0..ra_count)
            .map(|_| Vec::with_capacity(rows))
            .collect::<Vec<_>>();
        for index in 0..rows {
            let point = boolean_point_msb(instance.num_vars, index);
            public_coeff.push(self.bytecode_public_coeff(&point)?);
            for (target, value) in bytecode_ra.iter_mut().zip(self.bytecode_ra_values(&point)?) {
                target.push(value);
            }
        }
        Ok(Stage6RelationState::BytecodeReadRaf(
            bytecode::BytecodeReadRafRelationState::new(
                Polynomial::new(public_coeff),
                bytecode_ra.into_iter().map(Polynomial::new).collect(),
                Some(self.materialize_bytecode_address_phase()?),
            ),
        ))
    }

    fn materialize_bytecode_address_phase(
        &self,
    ) -> Result<bytecode::BytecodeReadRafAddressPhaseState<F>, ProverError> {
        let context = self.config.bytecode_context.as_ref().ok_or_else(|| {
            invalid_stage_request("Stage 6 bytecode context is required for read-RAF evaluation")
        })?;
        let log_k = self.config.bytecode_read_raf_dimensions.log_k();
        let rows = 1usize.checked_shl(log_k as u32).ok_or_else(|| {
            invalid_sumcheck_output("Stage 6 bytecode address phase row count overflowed")
        })?;
        if context.rows.len() != rows {
            return Err(invalid_stage_request(format!(
                "Stage 6 bytecode context has {} rows, expected {rows}",
                context.rows.len()
            )));
        }

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
        let register_eq = bytecode::read_raf_register_eq_evals(
            register_points.register_read_write_address,
            register_points.register_val_evaluation_address,
        );
        let zero_cycle = vec![F::zero(); self.config.bytecode_read_raf_dimensions.log_t()];

        let mut stage_f = (0..5).map(|_| Vec::with_capacity(rows)).collect::<Vec<_>>();
        let mut stage_val = (0..5).map(|_| Vec::with_capacity(rows)).collect::<Vec<_>>();
        let mut entry_trace = Vec::with_capacity(rows);
        let mut entry_expected = Vec::with_capacity(rows);

        let identity_coefficients = bytecode::read_raf_address_phase_identity_coefficients(
            &self.prefix.challenges.bytecode_gamma_powers,
        )
        .map_err(|error| invalid_sumcheck_output(error.to_string()))?;
        for index in 0..rows {
            let address_point = boolean_point_msb(log_k, index);
            let r_address = address_point.iter().rev().copied().collect::<Vec<_>>();
            let row_index = boolean_index_msb(&r_address).ok_or_else(|| {
                invalid_sumcheck_output("Stage 6 bytecode address phase expected Boolean address")
            })?;
            let row = context.rows.get(row_index).ok_or_else(|| {
                invalid_stage_request(format!(
                    "Stage 6 bytecode row index {row_index} is out of range for {rows} rows"
                ))
            })?;
            for (stage, cycle) in stage_cycles.iter().enumerate() {
                stage_f[stage].push(self.bytecode_ra_indicator_sum_at(&r_address, cycle)?);
            }

            #[cfg(feature = "field-inline")]
            let row = field_bytecode::base_jolt_bytecode_row(row);
            #[cfg(feature = "field-inline")]
            let row = &row;
            let row_values = bytecode::read_raf_row_values::<F>(
                row,
                &register_eq.read_write,
                &register_eq.val_evaluation,
                &self.prefix.challenges.stage1_gammas,
                &self.prefix.challenges.stage2_gammas,
                &self.prefix.challenges.stage3_gammas,
                &self.prefix.challenges.stage4_gammas,
                &self.prefix.challenges.stage5_gammas,
            );
            let row_values = bytecode::read_raf_address_phase_stage_values(
                row_values,
                row_index,
                identity_coefficients,
            );
            for (target, value) in stage_val.iter_mut().zip(row_values) {
                target.push(value);
            }
            entry_trace.push(
                self.bytecode_ra_product_at(stage6_bytecode_ra_point(&r_address, &zero_cycle))?,
            );
            entry_expected.push(if row_index == context.entry_bytecode_index {
                F::one()
            } else {
                F::zero()
            });
        }

        Ok(bytecode::BytecodeReadRafAddressPhaseState::new(
            log_k,
            stage_f.into_iter().map(Polynomial::new).collect(),
            stage_val.into_iter().map(Polynomial::new).collect(),
            Polynomial::new(entry_trace),
            Polynomial::new(entry_expected),
            self.prefix.challenges.bytecode_gamma_powers.clone(),
        ))
    }

    fn materialize_booleanity_relation(
        &self,
        instance: &Stage6BatchInstance<F>,
    ) -> Result<Stage6RelationState<F>, ProverError> {
        let rows = Self::relation_rows(instance)?;
        let mut ra = (0..self.config.booleanity_dimensions.layout.total())
            .map(|_| Vec::with_capacity(rows))
            .collect::<Vec<_>>();
        for index in 0..rows {
            let point = boolean_point_msb(instance.num_vars, index);
            let (instruction, bytecode, ram) = self.booleanity_ra_values(&point)?;
            for (target, value) in ra
                .iter_mut()
                .zip(instruction.into_iter().chain(bytecode).chain(ram))
            {
                target.push(value);
            }
        }
        Ok(Stage6RelationState::Booleanity(
            booleanity::BooleanityRelationState::new(
                booleanity::eq_address_cycle_polynomial(
                    &self.prefix.challenges.booleanity_reference.address,
                    &self.prefix.challenges.booleanity_reference.cycle,
                ),
                ra.into_iter().map(Polynomial::new).collect(),
                self.prefix.challenges.booleanity_gamma * self.prefix.challenges.booleanity_gamma,
            ),
        ))
    }

    fn materialize_ram_hamming_relation(
        &self,
        instance: &Stage6BatchInstance<F>,
    ) -> Result<Stage6RelationState<F>, ProverError> {
        let [hamming_opening] = ram::hamming_booleanity_output_openings();
        let hamming_weight = Self::materialize_values(instance, |point| {
            let opening_point = self
                .config
                .trace_dimensions()
                .cycle_opening_point(point)
                .map_err(|error| invalid_sumcheck_output(error.to_string()))?;
            self.evaluate_opening(hamming_opening, &opening_point)
        })?;
        Ok(Stage6RelationState::RamHammingBooleanity(
            ram::RamHammingBooleanityRelationState::new(
                ram::hamming_booleanity_eq_cycle_polynomial(
                    stage6_stage1_cycle_binding(self.stage1).map_err(invalid_stage_request)?,
                ),
                hamming_weight,
            ),
        ))
    }

    fn materialize_ram_ra_relation(
        &self,
        instance: &Stage6BatchInstance<F>,
    ) -> Result<Stage6RelationState<F>, ProverError> {
        let rows = Self::relation_rows(instance)?;
        let ra_count = self
            .config
            .ram_ra_virtualization_dimensions
            .num_committed_ra_polys();
        let mut ram_ra = (0..ra_count)
            .map(|_| Vec::with_capacity(rows))
            .collect::<Vec<_>>();
        for index in 0..rows {
            let point = boolean_point_msb(instance.num_vars, index);
            for (target, value) in ram_ra
                .iter_mut()
                .zip(self.ram_ra_virtualization_values(&point)?)
            {
                target.push(value);
            }
        }
        Ok(Stage6RelationState::RamRaVirtualization(
            ram::RamRaVirtualizationRelationState::new(
                ram::ra_virtualization_eq_cycle_polynomial(self.ram_reduced_opening_point()?.cycle),
                ram_ra.into_iter().map(Polynomial::new).collect(),
            ),
        ))
    }

    fn materialize_instruction_ra_relation(
        &self,
        instance: &Stage6BatchInstance<F>,
    ) -> Result<Stage6RelationState<F>, ProverError> {
        let rows = Self::relation_rows(instance)?;
        let dimensions = self.config.instruction_ra_virtualization_dimensions;
        let mut groups = (0..dimensions.num_virtual_ra_polys())
            .map(|_| {
                (0..dimensions.num_committed_per_virtual())
                    .map(|_| Vec::with_capacity(rows))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        for index in 0..rows {
            let point = boolean_point_msb(instance.num_vars, index);
            let values = self.instruction_ra_virtualization_values(&point)?;
            for (virtual_index, group) in groups.iter_mut().enumerate() {
                let start = virtual_index * dimensions.num_committed_per_virtual();
                for (target, value) in group
                    .iter_mut()
                    .zip(values[start..start + dimensions.num_committed_per_virtual()].iter())
                {
                    target.push(*value);
                }
            }
        }
        Ok(Stage6RelationState::InstructionRaVirtualization(
            instruction::InstructionRaVirtualizationRelationState::new(
                instruction::ra_virtualization_eq_cycle_polynomial(
                    self.instruction_read_raf_point().cycle,
                ),
                self.prefix.challenges.instruction_ra_gamma_powers.clone(),
                groups
                    .into_iter()
                    .map(|group| group.into_iter().map(Polynomial::new).collect())
                    .collect(),
            ),
        ))
    }

    fn materialize_inc_relation(
        &self,
        instance: &Stage6BatchInstance<F>,
    ) -> Result<Stage6RelationState<F>, ProverError> {
        let [ram_inc_opening, rd_inc_opening] = increments::claim_reduction_output_openings();
        let ram_inc = Self::materialize_values(instance, |point| {
            let opening_point = self
                .config
                .trace_dimensions()
                .cycle_opening_point(point)
                .map_err(|error| invalid_sumcheck_output(error.to_string()))?;
            self.evaluate_opening(ram_inc_opening, &opening_point)
        })?;
        let rd_inc = Self::materialize_values(instance, |point| {
            let opening_point = self
                .config
                .trace_dimensions()
                .cycle_opening_point(point)
                .map_err(|error| invalid_sumcheck_output(error.to_string()))?;
            self.evaluate_opening(rd_inc_opening, &opening_point)
        })?;
        let cycles = stage6_inc_claim_reduction_cycle_points(
            self.stage2,
            self.stage4,
            self.stage5,
            self.config.log_k,
        )
        .map_err(|error| invalid_sumcheck_output(error.to_string()))?;
        let (ram_coeff, rd_coeff) = increments::claim_reduction_output_coefficient_polynomials(
            increments::ClaimReductionOutputCoefficientPolynomialInputs {
                num_vars: instance.num_vars,
                trace_dimensions: self.config.trace_dimensions(),
                ram_read_write_cycle: cycles.ram_read_write_cycle,
                ram_val_check_cycle: cycles.ram_val_check_cycle,
                registers_read_write_cycle: cycles.registers_read_write_cycle,
                registers_val_evaluation_cycle: cycles.registers_val_evaluation_cycle,
                gamma: self.prefix.challenges.inc_gamma,
            },
        )
        .map_err(|error| invalid_sumcheck_output(error.to_string()))?;
        Ok(Stage6RelationState::IncClaimReduction(
            increments::IncClaimReductionRelationState::new(
                ram_coeff,
                rd_coeff,
                ram_inc,
                rd_inc,
                self.prefix.challenges.inc_gamma * self.prefix.challenges.inc_gamma,
            ),
        ))
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
        Ok(Stage6RelationState::AdviceCyclePhase(
            advice::AdviceCyclePhaseRelationState::new(
                advice,
                eq,
                layout.cycle_phase_col_rounds(),
                layout.cycle_phase_row_rounds(),
            ),
        ))
    }

    pub(super) fn derived_points(
        &self,
        sumcheck_point: &[F],
    ) -> Result<Stage6BatchPoints<F>, ProverError> {
        if sumcheck_point.len() != self.max_num_vars {
            return Err(invalid_sumcheck_output(format!(
                "Stage 6 batch sumcheck point has {} variables, expected {}",
                sumcheck_point.len(),
                self.max_num_vars
            )));
        }

        let bytecode_read_raf =
            self.instance_point(sumcheck_point, Stage6InstanceKind::BytecodeReadRaf)?;
        let booleanity = self.instance_point(sumcheck_point, Stage6InstanceKind::Booleanity)?;
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
        #[cfg(feature = "field-inline")]
        let field_registers_inc_claim_reduction = self.instance_point(
            sumcheck_point,
            Stage6InstanceKind::FieldRegistersIncClaimReduction,
        )?;

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
                #[cfg(feature = "field-inline")]
                field_registers_inc_claim_reduction: &field_registers_inc_claim_reduction,
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
            #[cfg(feature = "field-inline")]
            field_registers_inc_claim_reduction: self
                .expected_field_registers_inc_claim_reduction_output(
                    &points.field_registers_inc_claim_reduction_opening_point,
                    openings
                        .field_inline
                        .field_registers_inc_claim_reduction
                        .field_rd_inc,
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

    fn bytecode_ra_values(&self, point: &[F]) -> Result<Vec<F>, ProverError> {
        let opening =
            stage6_bytecode_read_raf_point(self.config.bytecode_read_raf_dimensions, point)
                .map_err(|error| invalid_sumcheck_output(error.to_string()))?;
        self.bytecode_ra_values_at(opening.ra_point())
    }

    fn bytecode_ra_values_at(
        &self,
        point: Stage6BytecodeRaPoint<'_, F>,
    ) -> Result<Vec<F>, ProverError> {
        point
            .committed_opening_points(self.config.committed_chunk_bits)
            .into_iter()
            .enumerate()
            .map(|(index, opening_point)| {
                self.evaluate_oracle(
                    OracleRef::committed(JoltCommittedPolynomial::BytecodeRa(index)),
                    &opening_point,
                )
            })
            .collect()
    }

    fn bytecode_ra_product_at(
        &self,
        point: Stage6BytecodeRaPoint<'_, F>,
    ) -> Result<F, ProverError> {
        Ok(self
            .bytecode_ra_values_at(point)?
            .into_iter()
            .fold(F::one(), |acc, value| acc * value))
    }

    fn bytecode_ra_indicator_sum_at(
        &self,
        r_address: &[F],
        r_cycle: &[F],
    ) -> Result<F, ProverError> {
        let log_t = self.config.bytecode_read_raf_dimensions.log_t();
        let cycles = 1usize.checked_shl(log_t as u32).ok_or_else(|| {
            invalid_sumcheck_output("Stage 6 bytecode cycle row count overflowed")
        })?;
        let mut sum = F::zero();
        for cycle_index in 0..cycles {
            let weight = eq_index_msb(r_cycle, cycle_index);
            if weight.is_zero() {
                continue;
            }
            let cycle_point = boolean_point_msb(log_t, cycle_index);
            sum += weight
                * self.bytecode_ra_product_at(stage6_bytecode_ra_point(r_address, &cycle_point))?;
        }
        Ok(sum)
    }

    fn booleanity_ra_values(&self, point: &[F]) -> Result<Stage6RaValueGroups<F>, ProverError> {
        let opening = self
            .config
            .booleanity_dimensions
            .opening_point(point)
            .map_err(|error| invalid_sumcheck_output(error.to_string()))?;
        let mut instruction =
            Vec::with_capacity(self.config.booleanity_dimensions.layout.instruction());
        for index in 0..self.config.booleanity_dimensions.layout.instruction() {
            instruction.push(self.evaluate_oracle(
                OracleRef::committed(JoltCommittedPolynomial::InstructionRa(index)),
                &opening.opening_point,
            )?);
        }
        let mut bytecode = Vec::with_capacity(self.config.booleanity_dimensions.layout.bytecode());
        for index in 0..self.config.booleanity_dimensions.layout.bytecode() {
            bytecode.push(self.evaluate_oracle(
                OracleRef::committed(JoltCommittedPolynomial::BytecodeRa(index)),
                &opening.opening_point,
            )?);
        }
        let mut ram = Vec::with_capacity(self.config.booleanity_dimensions.layout.ram());
        for index in 0..self.config.booleanity_dimensions.layout.ram() {
            ram.push(self.evaluate_oracle(
                OracleRef::committed(JoltCommittedPolynomial::RamRa(index)),
                &opening.opening_point,
            )?);
        }
        Ok((instruction, bytecode, ram))
    }

    fn ram_ra_virtualization_values(&self, point: &[F]) -> Result<Vec<F>, ProverError> {
        let r_cycle = self
            .config
            .trace_dimensions()
            .cycle_opening_point(point)
            .map_err(|error| invalid_sumcheck_output(error.to_string()))?;
        let ram_reduced = self.ram_reduced_opening_point()?;
        ram_reduced
            .committed_opening_points(&r_cycle, self.config.committed_chunk_bits)
            .into_iter()
            .enumerate()
            .map(|(index, point)| {
                self.evaluate_oracle(
                    OracleRef::committed(JoltCommittedPolynomial::RamRa(index)),
                    &point,
                )
            })
            .collect()
    }

    fn instruction_ra_virtualization_values(&self, point: &[F]) -> Result<Vec<F>, ProverError> {
        let r_cycle = self
            .config
            .trace_dimensions()
            .cycle_opening_point(point)
            .map_err(|error| invalid_sumcheck_output(error.to_string()))?;
        let instruction_read_raf = self.instruction_read_raf_point();
        instruction_read_raf
            .committed_opening_points(&r_cycle, self.config.committed_chunk_bits)
            .into_iter()
            .enumerate()
            .map(|(index, point)| {
                self.evaluate_oracle(
                    OracleRef::committed(JoltCommittedPolynomial::InstructionRa(index)),
                    &point,
                )
            })
            .collect()
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

    #[cfg(feature = "field-inline")]
    fn expected_field_registers_inc_claim_reduction_output(
        &self,
        opening_point: &[F],
        field_rd_inc: F,
    ) -> Result<F, ProverError> {
        let field_log_k = self.config.field_inline.field_register_log_k;
        let cycles = stage6_field_registers_inc_claim_reduction_cycle_points(
            self.stage4,
            self.stage5,
            field_log_k,
            self.config.log_t,
        )
        .map_err(|error| invalid_sumcheck_output(error.to_string()))?;
        stage6_field_registers_inc_claim_reduction_expected_output(
            FieldInlineStage6IncClaimReductionExpectedOutputInputs {
                opening_point,
                read_write_cycle: cycles.read_write_cycle,
                val_evaluation_cycle: cycles.val_evaluation_cycle,
                field_rd_inc,
                gamma: self.prefix.challenges.field_inc_gamma,
            },
        )
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
        #[cfg(feature = "field-inline")]
        let base_bytecode_rows = context
            .rows
            .iter()
            .map(field_bytecode::base_jolt_bytecode_row)
            .collect::<Vec<_>>();
        #[cfg(feature = "field-inline")]
        let bytecode_rows = base_bytecode_rows.as_slice();
        #[cfg(not(feature = "field-inline"))]
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

        #[cfg(feature = "field-inline")]
        let public_values = {
            let mut public_values = public_values;
            self.add_field_inline_bytecode_public_values(
                &mut public_values,
                &context.rows,
                &opening.r_address,
                &opening.r_cycle,
                &stage_cycles[0],
            )?;
            public_values
        };

        Ok(public_values)
    }

    #[cfg(feature = "field-inline")]
    fn add_field_inline_bytecode_public_values(
        &self,
        bytecode_public_values: &mut bytecode::BytecodeReadRafPublicValues<F>,
        bytecode_rows: &[JoltInstructionRow],
        r_address: &[F],
        r_cycle: &[F],
        stage1_cycle: &[F],
    ) -> Result<(), ProverError> {
        let field_log_k = self.config.field_inline.field_register_log_k;
        let field_register_points = stage6_field_inline_bytecode_register_points(
            self.stage4,
            self.stage5,
            field_log_k,
            self.config.log_t,
        )
        .map_err(|error| invalid_sumcheck_output(error.to_string()))?;
        let field_inline_bytecode = bytecode_rows
            .iter()
            .map(field_bytecode::field_inline_bytecode_row)
            .collect::<Vec<_>>();
        let field_values = field_bytecode::read_raf_public_values(
            field_bytecode::FieldInlineBytecodeReadRafEvaluationInputs {
                bytecode: &field_inline_bytecode,
                field_register_log_k: field_log_k,
                r_address,
                r_cycle,
                stage1_cycle_point: stage1_cycle,
                field_register_read_write_point: field_register_points.read_write_address,
                field_register_read_write_cycle_point: field_register_points.read_write_cycle,
                field_register_val_evaluation_point: field_register_points.val_evaluation_address,
                field_register_val_evaluation_cycle_point: field_register_points
                    .val_evaluation_cycle,
                stage1_gammas: &self.prefix.challenges.stage1_gammas,
                stage4_gammas: &self.prefix.challenges.stage4_gammas,
                stage5_gammas: &self.prefix.challenges.stage5_gammas,
            },
        )
        .map_err(|error| invalid_sumcheck_output(error.to_string()))?;
        field_bytecode::merge_read_raf_public_values(bytecode_public_values, field_values);
        Ok(())
    }

    fn bytecode_public_coeff(&self, point: &[F]) -> Result<F, ProverError> {
        require_len(
            "Stage 6 bytecode gamma powers",
            self.prefix.challenges.bytecode_gamma_powers.len(),
            8,
        )?;
        let public_values = self.bytecode_public_values(point)?;
        stage6_bytecode_read_raf_output_coefficient(Stage6BytecodeReadRafOutputCoefficientInputs {
            dimensions: self.config.bytecode_read_raf_dimensions,
            public_values: &public_values,
            gamma: self.prefix.challenges.bytecode_gamma_powers[1],
        })
        .map_err(|error| invalid_sumcheck_output(error.to_string()))
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

    fn evaluate_opening(&self, opening: JoltOpeningId, point: &[F]) -> Result<F, ProverError> {
        let oracle = jolt_opening_oracle_ref(opening)?;
        self.evaluate_oracle(oracle, point)
    }

    fn evaluate_oracle(
        &self,
        oracle: OracleRef<JoltVmNamespace>,
        point: &[F],
    ) -> Result<F, ProverError> {
        let is_boolean_point = boolean_index_msb(point).is_some();
        let requirement = self
            .witness
            .view_requirements(oracle)?
            .into_iter()
            .next()
            .ok_or_else(|| {
                invalid_stage_request(format!(
                    "witness returned no view requirement for Stage 6 oracle {:?}",
                    oracle.kind
                ))
            })?;
        let request = OracleViewRequest::new(requirement);
        if let Some(polynomial) = self.oracle_cache.borrow().get(&oracle) {
            return evaluate_cached_polynomial(oracle, polynomial, point);
        }
        if !is_boolean_point {
            if let Some(value) = self
                .witness
                .try_evaluate_oracle_view(request.clone(), point)?
            {
                return Ok(value);
            }
        }
        let view = self.witness.oracle_view(request)?;
        let values = view.as_slice().ok_or_else(|| {
            invalid_stage_request(format!(
                "Stage 6 oracle {:?} did not materialize a concrete view",
                oracle.kind
            ))
        })?;
        let polynomial = Polynomial::new(values.to_vec());
        if polynomial.num_vars() != point.len() {
            return Err(invalid_stage_request(format!(
                "Stage 6 oracle {:?} has {} variables, evaluated at {} variables",
                oracle.kind,
                polynomial.num_vars(),
                point.len()
            )));
        }
        let value = evaluate_cached_polynomial(oracle, &polynomial, point)?;
        let _ = self.oracle_cache.borrow_mut().insert(oracle, polynomial);
        Ok(value)
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

fn evaluate_cached_polynomial<F: Field>(
    oracle: OracleRef<JoltVmNamespace>,
    polynomial: &Polynomial<F>,
    point: &[F],
) -> Result<F, ProverError> {
    if polynomial.num_vars() != point.len() {
        return Err(invalid_stage_request(format!(
            "Stage 6 oracle {:?} has {} variables, evaluated at {} variables",
            oracle.kind,
            polynomial.num_vars(),
            point.len()
        )));
    }
    Ok(polynomial.evaluate_with_msb_boolean_fast_path(point))
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
    let view = witness.oracle_view(OracleViewRequest::new(requirement))?;
    view.as_slice().map(<[F]>::to_vec).ok_or_else(|| {
        invalid_stage_request(format!("Stage 6 {kind:?} advice view is not concrete"))
    })
}

fn require_len(label: &'static str, actual: usize, expected: usize) -> Result<(), ProverError> {
    if actual < expected {
        return Err(invalid_stage_request(format!(
            "{label} has {actual} values, expected at least {expected}"
        )));
    }
    Ok(())
}

fn invalid_stage_request(error: impl std::fmt::Display) -> ProverError {
    ProverError::InvalidStageRequest {
        reason: error.to_string(),
    }
}

fn invalid_sumcheck_output(error: impl std::fmt::Display) -> ProverError {
    ProverError::InvalidSumcheckOutput {
        reason: error.to_string(),
    }
}
