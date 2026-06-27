//! The stage 6b cycle-phase relation bundle.
//!
//! Stage 6b batches the cycle-phase sumcheck instances: the base relations
//! (bytecode read-RAF, booleanity, RAM hamming booleanity, RAM/instruction RA
//! virtualization, increment claim reduction) and — in committed-program mode —
//! the per-kind advice, bytecode, and program-image claim-reduction cycle phases.
//!
//! [`Stage6Relations`] is built once (from the stage-1..5 clear outputs and the
//! stage-6a address-phase results) and shared by the verifier and the prover, so
//! the per-instance input claims (claimed sums) and the produced opening points /
//! expected outputs cannot drift between them. The bundle holds the relation
//! objects plus their wired [`InputClaims`](crate::stages::relations::InputClaims);
//! both sides read [`Stage6Relations::sumcheck_claims`] for the batch's input
//! claims and reuse the relation objects for `derive_opening_points` /
//! `expected_output`.

use jolt_claims::protocols::jolt::{
    geometry::{
        booleanity::BooleanityDimensions, bytecode::BytecodeReadRafDimensions,
        dimensions::TraceDimensions, instruction::InstructionRaVirtualizationDimensions,
        ram::RamRaVirtualizationDimensions,
    },
    AdviceClaimReductionLayout, BytecodeClaimReductionLayout, JoltAdviceKind, JoltSumcheckSpec,
    ProgramImageClaimReductionLayout,
};
use jolt_field::Field;
use jolt_riscv::JoltInstructionRow;
use jolt_sumcheck::SumcheckClaim;

use super::booleanity::{booleanity_inputs_from_upstream, Booleanity, BooleanityInputClaims};
use super::bytecode_read_raf::{
    bytecode_read_raf_inputs_from_upstream, BytecodeReadRaf, BytecodeReadRafCommitted,
    BytecodeReadRafCommittedCycleInputs, BytecodeReadRafCycleInputs, BytecodeReadRafInputClaims,
    BytecodeReadRafOutputClaims,
};
use super::committed_reduction_cycle_phase::{
    advice_cycle_phase_inputs_from_upstream, bytecode_reduction_cycle_phase_inputs_from_values,
    program_image_reduction_cycle_phase_inputs_from_upstream, AdviceCyclePhase,
    AdviceCyclePhaseInputClaims, BytecodeReductionCyclePhase,
    BytecodeReductionCyclePhaseInputClaims, ProgramImageReductionCyclePhase,
    ProgramImageReductionCyclePhaseInputClaims,
};
use super::inc_claim_reduction::{
    inc_claim_reduction_inputs_from_upstream, IncClaimReduction, IncClaimReductionInputClaims,
};
use super::instruction_ra_virtualization::{
    instruction_ra_virtualization_inputs_from_upstream, InstructionRaVirtualization,
    InstructionRaVirtualizationInputClaims,
};
use super::outputs::BytecodeReductionWeights;
use super::ram_hamming_booleanity::{
    ram_hamming_booleanity_inputs_from_upstream, RamHammingBooleanity,
    RamHammingBooleanityInputClaims,
};
use super::ram_ra_virtualization::{
    ram_ra_virtualization_inputs_from_upstream, RamRaVirtualization, RamRaVirtualizationInputClaims,
};
use crate::stages::relations::{ConcreteSumcheck, OpeningClaim};
use crate::stages::{
    stage2::Stage2ClearOutput, stage4::Stage4ClearOutput, stage5::Stage5ClearOutput,
};
use crate::VerifierError;

/// The stage-6b bytecode read-RAF cycle relation, dispatching over full-program
/// mode ([`BytecodeReadRaf`], which borrows the bytecode table) and
/// committed-program mode ([`BytecodeReadRafCommitted`]). Both variants share the
/// same input/output claim cells, so the bundle treats them uniformly.
pub enum BytecodeReadRafCycle<'a, F: Field> {
    Full(BytecodeReadRaf<'a, F>),
    Committed(BytecodeReadRafCommitted<F>),
}

impl<F: Field> BytecodeReadRafCycle<'_, F> {
    pub fn spec(&self) -> JoltSumcheckSpec {
        match self {
            Self::Full(relation) => relation.spec(),
            Self::Committed(relation) => relation.spec(),
        }
    }

    pub fn input_claim(
        &self,
        inputs: &BytecodeReadRafInputClaims<OpeningClaim<F>>,
    ) -> Result<F, VerifierError> {
        match self {
            Self::Full(relation) => relation.input_claim(inputs),
            Self::Committed(relation) => relation.input_claim(inputs),
        }
    }

    pub fn derive_opening_points(
        &self,
        sumcheck_point: &[F],
        inputs: &BytecodeReadRafInputClaims<OpeningClaim<F>>,
    ) -> Result<BytecodeReadRafOutputClaims<Vec<F>>, VerifierError> {
        match self {
            Self::Full(relation) => relation.derive_opening_points(sumcheck_point, inputs),
            Self::Committed(relation) => relation.derive_opening_points(sumcheck_point, inputs),
        }
    }

    pub fn expected_output(
        &self,
        inputs: &BytecodeReadRafInputClaims<OpeningClaim<F>>,
        outputs: &BytecodeReadRafOutputClaims<OpeningClaim<F>>,
    ) -> Result<F, VerifierError> {
        match self {
            Self::Full(relation) => relation.expected_output(inputs, outputs),
            Self::Committed(relation) => relation.expected_output(inputs, outputs),
        }
    }
}

/// Construction inputs for [`Stage6Relations::build`]. Carries the per-stage cycle
/// bindings, the stage-6a address-phase results, the Fiat-Shamir gammas, and the
/// committed-program reduction layouts/weights. Both the verifier and the prover
/// populate this identically from their own prior-stage state.
pub struct Stage6RelationsParams<'a, F: Field> {
    pub bytecode_dimensions: BytecodeReadRafDimensions,
    pub booleanity_dimensions: BooleanityDimensions,
    pub trace_dimensions: TraceDimensions,
    pub ram_ra_dimensions: RamRaVirtualizationDimensions,
    pub instruction_ra_dimensions: InstructionRaVirtualizationDimensions,
    pub committed_chunk_bits: usize,
    /// Full bytecode table; `Some` in full-program mode, `None` in committed mode.
    pub bytecode_table: Option<&'a [JoltInstructionRow]>,
    pub entry_bytecode_index: usize,
    /// The stage-6a bytecode read-RAF address opening (`bytecode_r_address`).
    pub bytecode_r_address: Vec<F>,
    /// The stage-6a booleanity address opening (`booleanity_r_address`).
    pub booleanity_r_address: Vec<F>,
    /// The address-phase `BytecodeReadRafAddrClaim` intermediate (cycle input).
    pub address_bytecode_read_raf: F,
    /// The address-phase `BooleanityAddrClaim` intermediate (cycle input).
    pub address_booleanity: F,
    /// The address-phase staged `BytecodeValStage` opening values (committed mode).
    pub address_val_stages: Vec<F>,
    pub bytecode_gamma: F,
    pub instruction_ra_gamma: F,
    pub inc_gamma: F,
    pub booleanity_gamma: F,
    pub eta: Option<F>,
    /// Per-stage (1..=5) cycle bindings used by the bytecode read-RAF table fold.
    pub stage_cycle_points: [Vec<F>; 5],
    pub register_read_write_point: Vec<F>,
    pub register_val_evaluation_point: Vec<F>,
    /// Per-stage (1..=5) Fiat-Shamir gamma powers.
    pub stage_gammas: [Vec<F>; 5],
    pub booleanity_reference_address: Vec<F>,
    pub booleanity_reference_cycle: Vec<F>,
    /// The stage-1 Spartan-outer cycle binding (RAM hamming booleanity reference).
    pub stage1_cycle_binding: Vec<F>,
    /// The stage-5 reduced RAM address prefix.
    pub ram_reduced_address: Vec<F>,
    /// The stage-5 reduced RAM cycle suffix.
    pub ram_reduced_cycle: Vec<F>,
    /// The stage-5 instruction RA reduced address prefix.
    pub instruction_r_address: Vec<F>,
    /// The stage-5 instruction RA reduced cycle suffix.
    pub instruction_r_cycle: Vec<F>,
    /// Increment claim-reduction per-source cycle bindings, in source order:
    /// RAM read-write, RAM value-check, register read-write, register value-eval.
    pub inc_cycle_points: [Vec<F>; 4],
    pub trusted_advice_layout: Option<&'a AdviceClaimReductionLayout>,
    pub untrusted_advice_layout: Option<&'a AdviceClaimReductionLayout>,
    pub bytecode_reduction_layout: Option<&'a BytecodeClaimReductionLayout>,
    pub program_image_reduction_layout: Option<&'a ProgramImageClaimReductionLayout>,
    /// The bytecode claim-reduction output weights (committed mode), precomputed by
    /// the caller via `bytecode_reduction_weights`.
    pub bytecode_reduction_weights: Option<BytecodeReductionWeights<F>>,
    /// The RAM read-write `RamVal` address prefix (program-image `FinalScale`).
    pub program_image_r_addr_rw: Vec<F>,
}

/// The stage-6b cycle-phase relation objects and their wired input claims. Built
/// once from the stage-1..5 clear outputs and the stage-6a results, and shared by
/// the verifier and the prover.
pub struct Stage6Relations<'a, F: Field> {
    pub bytecode_read_raf: BytecodeReadRafCycle<'a, F>,
    pub bytecode_read_raf_inputs: BytecodeReadRafInputClaims<OpeningClaim<F>>,
    pub booleanity: Booleanity<F>,
    pub booleanity_inputs: BooleanityInputClaims<OpeningClaim<F>>,
    pub ram_hamming: RamHammingBooleanity<F>,
    pub ram_hamming_inputs: RamHammingBooleanityInputClaims<OpeningClaim<F>>,
    pub ram_ra: RamRaVirtualization<F>,
    pub ram_ra_inputs: RamRaVirtualizationInputClaims<OpeningClaim<F>>,
    pub instruction_ra: InstructionRaVirtualization<F>,
    pub instruction_ra_inputs: InstructionRaVirtualizationInputClaims<OpeningClaim<F>>,
    pub inc: IncClaimReduction<F>,
    pub inc_inputs: IncClaimReductionInputClaims<OpeningClaim<F>>,
    pub trusted_advice: Option<AdviceCyclePhase<F>>,
    pub untrusted_advice: Option<AdviceCyclePhase<F>>,
    pub advice_inputs: AdviceCyclePhaseInputClaims<OpeningClaim<F>>,
    pub bytecode_reduction: Option<BytecodeReductionCyclePhase<F>>,
    pub bytecode_reduction_inputs: Option<BytecodeReductionCyclePhaseInputClaims<OpeningClaim<F>>>,
    pub program_image_reduction: Option<ProgramImageReductionCyclePhase<F>>,
    pub program_image_reduction_inputs:
        Option<ProgramImageReductionCyclePhaseInputClaims<OpeningClaim<F>>>,
}

impl<'a, F: Field> Stage6Relations<'a, F> {
    pub fn build(
        params: Stage6RelationsParams<'a, F>,
        stage2: &Stage2ClearOutput<F>,
        stage4: &Stage4ClearOutput<F>,
        stage5: &Stage5ClearOutput<F>,
    ) -> Result<Self, VerifierError> {
        let bytecode_read_raf = match params.bytecode_table {
            Some(bytecode) => {
                BytecodeReadRafCycle::Full(BytecodeReadRaf::new(BytecodeReadRafCycleInputs {
                    dimensions: params.bytecode_dimensions,
                    gamma: params.bytecode_gamma,
                    bytecode,
                    r_address: params.bytecode_r_address.clone(),
                    stage_cycle_points: params.stage_cycle_points.clone(),
                    register_read_write_point: params.register_read_write_point.clone(),
                    register_val_evaluation_point: params.register_val_evaluation_point.clone(),
                    entry_bytecode_index: params.entry_bytecode_index,
                    stage_gammas: params.stage_gammas.clone(),
                    committed_chunk_bits: params.committed_chunk_bits,
                }))
            }
            None => BytecodeReadRafCycle::Committed(BytecodeReadRafCommitted::new(
                BytecodeReadRafCommittedCycleInputs {
                    dimensions: params.bytecode_dimensions,
                    gamma: params.bytecode_gamma,
                    r_address: params.bytecode_r_address.clone(),
                    stage_cycle_points: params.stage_cycle_points.clone(),
                    entry_bytecode_index: params.entry_bytecode_index,
                    committed_chunk_bits: params.committed_chunk_bits,
                    val_stages: params.address_val_stages.clone(),
                },
            )),
        };
        let bytecode_read_raf_inputs = bytecode_read_raf_inputs_from_upstream(OpeningClaim {
            point: Vec::new(),
            value: params.address_bytecode_read_raf,
        });

        let booleanity = Booleanity::new(
            params.booleanity_dimensions,
            params.booleanity_gamma,
            params.booleanity_r_address.clone(),
            params.booleanity_reference_address.clone(),
            params.booleanity_reference_cycle.clone(),
        );
        let booleanity_inputs = booleanity_inputs_from_upstream(OpeningClaim {
            point: Vec::new(),
            value: params.address_booleanity,
        });

        let ram_hamming =
            RamHammingBooleanity::new(params.trace_dimensions, params.stage1_cycle_binding.clone());
        let ram_hamming_inputs = ram_hamming_booleanity_inputs_from_upstream();

        let ram_ra = RamRaVirtualization::new(
            params.ram_ra_dimensions,
            params.ram_reduced_address.clone(),
            params.ram_reduced_cycle.clone(),
            params.committed_chunk_bits,
        );
        let ram_ra_inputs = ram_ra_virtualization_inputs_from_upstream(stage5);

        let instruction_ra = InstructionRaVirtualization::new(
            params.instruction_ra_dimensions,
            params.instruction_ra_gamma,
            params.instruction_r_address.clone(),
            params.instruction_r_cycle.clone(),
            params.committed_chunk_bits,
        );
        let instruction_ra_inputs = instruction_ra_virtualization_inputs_from_upstream(stage5);

        let [ram_read_write_cycle, ram_val_check_cycle, registers_read_write_cycle, registers_val_evaluation_cycle] =
            params.inc_cycle_points.clone();
        let inc = IncClaimReduction::new(
            params.trace_dimensions,
            params.inc_gamma,
            ram_read_write_cycle,
            ram_val_check_cycle,
            registers_read_write_cycle,
            registers_val_evaluation_cycle,
        );
        let inc_inputs = inc_claim_reduction_inputs_from_upstream(stage2, stage4, stage5);

        let advice_relation =
            |kind: JoltAdviceKind, layout: Option<&AdviceClaimReductionLayout>| {
                layout.and_then(|layout| {
                    stage4
                        .ram_val_check_init
                        .advice_contributions
                        .iter()
                        .find(|contribution| contribution.kind == kind)
                        .map(|contribution| {
                            AdviceCyclePhase::new(kind, layout, contribution.opening.point.clone())
                        })
                })
            };
        let trusted_advice = advice_relation(JoltAdviceKind::Trusted, params.trusted_advice_layout);
        let untrusted_advice =
            advice_relation(JoltAdviceKind::Untrusted, params.untrusted_advice_layout);
        let advice_inputs = advice_cycle_phase_inputs_from_upstream(stage4);

        let bytecode_reduction = match (
            params.bytecode_reduction_layout,
            params.eta,
            &params.bytecode_reduction_weights,
        ) {
            (Some(layout), Some(eta), Some(weights)) => Some(BytecodeReductionCyclePhase::new(
                layout,
                eta,
                weights.clone(),
            )),
            _ => None,
        };
        let bytecode_reduction_inputs = bytecode_reduction.as_ref().map(|_| {
            bytecode_reduction_cycle_phase_inputs_from_values(
                params
                    .address_val_stages
                    .iter()
                    .map(|value| OpeningClaim {
                        point: Vec::new(),
                        value: *value,
                    })
                    .collect(),
            )
        });

        let program_image_reduction = params.program_image_reduction_layout.map(|layout| {
            ProgramImageReductionCyclePhase::new(layout, params.program_image_r_addr_rw.clone())
        });
        let program_image_reduction_inputs = program_image_reduction
            .as_ref()
            .map(|_| program_image_reduction_cycle_phase_inputs_from_upstream(stage4))
            .transpose()?;

        Ok(Self {
            bytecode_read_raf,
            bytecode_read_raf_inputs,
            booleanity,
            booleanity_inputs,
            ram_hamming,
            ram_hamming_inputs,
            ram_ra,
            ram_ra_inputs,
            instruction_ra,
            instruction_ra_inputs,
            inc,
            inc_inputs,
            trusted_advice,
            untrusted_advice,
            advice_inputs,
            bytecode_reduction,
            bytecode_reduction_inputs,
            program_image_reduction,
            program_image_reduction_inputs,
        })
    }

    /// The per-instance batched-sumcheck claims, in canonical stage-6b batch order:
    /// bytecode read-RAF, booleanity, RAM hamming booleanity, RAM/instruction RA
    /// virtualization, increment claim reduction, then (committed mode) trusted /
    /// untrusted advice, bytecode reduction, program-image reduction. The input
    /// claim of each instance is the claimed sum of its sumcheck.
    pub fn sumcheck_claims(&self) -> Result<Vec<SumcheckClaim<F>>, VerifierError> {
        let claim =
            |spec: JoltSumcheckSpec, input: F| SumcheckClaim::new(spec.rounds, spec.degree, input);
        let mut claims = vec![
            claim(
                self.bytecode_read_raf.spec(),
                self.bytecode_read_raf
                    .input_claim(&self.bytecode_read_raf_inputs)?,
            ),
            claim(
                self.booleanity.spec(),
                self.booleanity.input_claim(&self.booleanity_inputs)?,
            ),
            claim(
                self.ram_hamming.spec(),
                self.ram_hamming.input_claim(&self.ram_hamming_inputs)?,
            ),
            claim(
                self.ram_ra.spec(),
                self.ram_ra.input_claim(&self.ram_ra_inputs)?,
            ),
            claim(
                self.instruction_ra.spec(),
                self.instruction_ra
                    .input_claim(&self.instruction_ra_inputs)?,
            ),
            claim(self.inc.spec(), self.inc.input_claim(&self.inc_inputs)?),
        ];
        for relation in [&self.trusted_advice, &self.untrusted_advice]
            .into_iter()
            .flatten()
        {
            claims.push(claim(
                relation.spec(),
                relation.input_claim(&self.advice_inputs)?,
            ));
        }
        if let (Some(relation), Some(inputs)) =
            (&self.bytecode_reduction, &self.bytecode_reduction_inputs)
        {
            claims.push(claim(relation.spec(), relation.input_claim(inputs)?));
        }
        if let (Some(relation), Some(inputs)) = (
            &self.program_image_reduction,
            &self.program_image_reduction_inputs,
        ) {
            claims.push(claim(relation.spec(), relation.input_claim(inputs)?));
        }
        Ok(claims)
    }
}
