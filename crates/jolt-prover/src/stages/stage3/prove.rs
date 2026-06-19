use std::collections::BTreeMap;

use jolt_backends::{
    stage3_instruction_input_opening_slot, stage3_registers_claim_reduction_opening_slot,
    stage3_shift_opening_slot, stage3_shift_rows, BackendValueSlot, Stage3SpartanSumcheckBackend,
    SumcheckBackend, SumcheckMaterializationOutput, SumcheckMaterializationRequest,
    SumcheckRegularBatchInstance, SumcheckRegularBatchLinearFactor, SumcheckRegularBatchLinearTerm,
    SumcheckRegularBatchProduct, SumcheckRegularBatchState, SumcheckStage3ShiftStateRequest,
    SumcheckViewMaterializationRequest,
};
use jolt_claims::protocols::jolt::{
    formulas::{claim_reductions::registers as registers_claim_reduction, instruction, spartan},
    JoltOpeningId, TraceDimensions,
};
#[cfg(feature = "zk")]
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_poly::{
    thread::unsafe_allocate_zero_vec, EqPlusOnePolynomial, EqPolynomial, Polynomial, UnivariatePoly,
};
use jolt_sumcheck::{
    append_sumcheck_claim, ClearProof, CompressedLabeledRoundPoly, CompressedSumcheckProof,
    RoundMessage, SumcheckProof,
};
use jolt_transcript::Transcript;
use jolt_verifier::stages::relations::SumcheckInstance;
use jolt_verifier::stages::{
    stage1::Stage1ClearOutput,
    stage2::Stage2ClearOutput,
    stage3::inputs::{
        InstructionInputOutputClaims, RegistersClaimReductionOutputClaims,
        SpartanShiftOutputClaims, Stage3OutputClaims,
    },
    stage3::instruction_input::{InstructionInput, InstructionInputInputClaims},
    stage3::outputs::{Stage3Challenges, Stage3ClearOutput},
    stage3::registers_claim_reduction::{
        RegistersClaimReduction, RegistersClaimReductionInputClaims,
    },
    stage3::spartan_shift::{SpartanShift, SpartanShiftInputClaims},
    stage3::{stage3_expected_final_claim, stage3_output_claims_with_points},
};
use jolt_verifier::{CheckedInputs, VerifierError};
use jolt_witness::{
    protocols::jolt_vm::{
        JoltVmNamespace, JoltVmStage3InstructionRegisterRows, JoltVmStage3ShiftRows,
    },
    WitnessProvider,
};
use rayon::prelude::*;

#[cfg(feature = "zk")]
use crate::committed::CommittedSumcheckWitness;
use crate::stages::invalid_sumcheck_output;
#[cfg(feature = "zk")]
use crate::stages::recorder::CommittedSumcheckRecorder;
use crate::stages::recorder::{ClearSumcheckRecorder, SumcheckRecorder};
use crate::stages::{
    collect_backend_materializations, take_backend_materialization,
    view_requirement_from_jolt_opening,
};
use crate::ProverError;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage3ProverConfig {
    pub log_t: usize,
}

impl Stage3ProverConfig {
    pub const fn new(log_t: usize) -> Self {
        Self { log_t }
    }
}

/// Canonical Stage 3 prover input.
#[derive(Clone, Copy, Debug)]
pub struct Stage3ProverInput<'a, F: Field, W> {
    pub config: Stage3ProverConfig,
    pub checked: &'a CheckedInputs,
    pub stage1: &'a Stage1ClearOutput<F>,
    pub stage2: &'a Stage2ClearOutput<F>,
    pub witness: &'a W,
}

impl<'a, F: Field, W> Stage3ProverInput<'a, F, W> {
    pub const fn new(
        config: Stage3ProverConfig,
        checked: &'a CheckedInputs,
        stage1: &'a Stage1ClearOutput<F>,
        stage2: &'a Stage2ClearOutput<F>,
        witness: &'a W,
    ) -> Self {
        Self {
            config,
            checked,
            stage1,
            stage2,
            witness,
        }
    }
}

/// The three Stage 3 sumcheck input claims (claimed sums), one per batched
/// relation. Computed via the relation objects' `input_claim` so prover and
/// verifier derive them identically.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Stage3InputClaims<F: Field> {
    shift: F,
    instruction_input: F,
    registers_claim_reduction: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Stage3RegularBatchPrefixOutput<F: Field> {
    input_claims: Stage3InputClaims<F>,
    shift_gamma: F,
    instruction_gamma: F,
    registers_gamma: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Stage3ShiftMaterializedOpenings<F: Field> {
    unexpanded_pc: Vec<F>,
    pc: Vec<F>,
    is_virtual: Vec<F>,
    is_first_in_sequence: Vec<F>,
    is_noop: Vec<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Stage3InstructionInputMaterializedOpenings<F: Field> {
    right_operand_is_rs2: Vec<F>,
    rs2_value: Vec<F>,
    right_operand_is_imm: Vec<F>,
    imm: Vec<F>,
    left_operand_is_rs1: Vec<F>,
    rs1_value: Vec<F>,
    left_operand_is_pc: Vec<F>,
    unexpanded_pc: Vec<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Stage3RegistersClaimReductionMaterializedOpenings<F: Field> {
    rd_write: Vec<F>,
    rs1: Vec<F>,
    rs2: Vec<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Stage3RegularBatchMaterializedOpenings<F: Field> {
    shift: Stage3ShiftMaterializedOpenings<F>,
    instruction_input: Stage3InstructionInputMaterializedOpenings<F>,
    registers_claim_reduction: Stage3RegistersClaimReductionMaterializedOpenings<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage3ProofComponent<F: Field, Proof> {
    pub stage3_sumcheck_proof: Proof,
    pub claims: Stage3OutputClaims<F>,
    pub verifier_output: Stage3ClearOutput<F>,
}

#[cfg(feature = "zk")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage3CommittedProofComponent<F, VC>
where
    F: Field,
    VC: VectorCommitment<Field = F>,
{
    pub stage3_sumcheck_proof: SumcheckProof<F, VC::Output>,
    pub verifier_output: Stage3ClearOutput<F>,
    pub output_claim_values: Vec<F>,
    pub(crate) committed_witness: CommittedSumcheckWitness<F>,
}

const STAGE3_INSTRUCTION_REGISTER_MATERIALIZE_CHUNK_ROWS: usize = 1 << 14;

fn stage3_materialized_openings_from_outputs<F: Field>(
    materializations: Vec<SumcheckMaterializationOutput<F>>,
) -> Result<Stage3RegularBatchMaterializedOpenings<F>, ProverError> {
    let mut values = collect_materialized_values(materializations)?;
    let materialized = Stage3RegularBatchMaterializedOpenings {
        shift: Stage3ShiftMaterializedOpenings {
            unexpanded_pc: take_shift_materialization(&mut values, 0, "unexpanded PC")?,
            pc: take_shift_materialization(&mut values, 1, "PC")?,
            is_virtual: take_shift_materialization(&mut values, 2, "virtual-instruction flag")?,
            is_first_in_sequence: take_shift_materialization(
                &mut values,
                3,
                "first-in-sequence flag",
            )?,
            is_noop: take_shift_materialization(&mut values, 4, "noop flag")?,
        },
        instruction_input: Stage3InstructionInputMaterializedOpenings {
            right_operand_is_rs2: take_instruction_input_materialization(
                &mut values,
                0,
                "right operand is rs2",
            )?,
            rs2_value: take_instruction_input_materialization(&mut values, 1, "rs2 value")?,
            right_operand_is_imm: take_instruction_input_materialization(
                &mut values,
                2,
                "right operand is imm",
            )?,
            imm: take_instruction_input_materialization(&mut values, 3, "immediate")?,
            left_operand_is_rs1: take_instruction_input_materialization(
                &mut values,
                4,
                "left operand is rs1",
            )?,
            rs1_value: take_instruction_input_materialization(&mut values, 5, "rs1 value")?,
            left_operand_is_pc: take_instruction_input_materialization(
                &mut values,
                6,
                "left operand is PC",
            )?,
            unexpanded_pc: take_instruction_input_materialization(&mut values, 7, "unexpanded PC")?,
        },
        registers_claim_reduction: Stage3RegistersClaimReductionMaterializedOpenings {
            rd_write: take_registers_claim_reduction_materialization(
                &mut values,
                0,
                "rd write value",
            )?,
            rs1: take_registers_claim_reduction_materialization(&mut values, 1, "rs1 value")?,
            rs2: take_registers_claim_reduction_materialization(&mut values, 2, "rs2 value")?,
        },
    };
    if let Some(slot) = values.keys().next() {
        return Err(invalid_sumcheck_output(format!(
            "unexpected Stage 3 output materialization slot {slot:?}"
        )));
    }
    Ok(materialized)
}

fn collect_materialized_values<F: Field>(
    materializations: Vec<SumcheckMaterializationOutput<F>>,
) -> Result<BTreeMap<BackendValueSlot, Vec<F>>, ProverError> {
    collect_backend_materializations(materializations, "Stage 3 materialization")
}

fn take_shift_materialization<F: Field>(
    values: &mut BTreeMap<BackendValueSlot, Vec<F>>,
    index: usize,
    label: &'static str,
) -> Result<Vec<F>, ProverError> {
    let slot = stage3_shift_opening_slot(index);
    take_backend_materialization(
        values,
        slot,
        format!("missing Stage 3 shift materialization for {label}"),
    )
}

fn take_instruction_input_materialization<F: Field>(
    values: &mut BTreeMap<BackendValueSlot, Vec<F>>,
    index: usize,
    label: &'static str,
) -> Result<Vec<F>, ProverError> {
    let slot = stage3_instruction_input_opening_slot(index);
    take_backend_materialization(
        values,
        slot,
        format!("missing Stage 3 instruction-input materialization for {label}"),
    )
}

fn take_registers_claim_reduction_materialization<F: Field>(
    values: &mut BTreeMap<BackendValueSlot, Vec<F>>,
    index: usize,
    label: &'static str,
) -> Result<Vec<F>, ProverError> {
    let slot = stage3_registers_claim_reduction_opening_slot(index);
    take_backend_materialization(
        values,
        slot,
        format!("missing Stage 3 registers claim-reduction materialization for {label}"),
    )
}

/// Canonical Stage 3 prover entrypoint.
///
/// Mirrors `jolt-verifier/src/stages/stage3/verify.rs` in prover execution order:
/// derive the shift/instruction/registers gammas, prove the regular batched
/// Boolean sumcheck over the three Stage 3 statements, evaluate the output
/// openings, then assemble the verifier-owned `stage3_sumcheck_proof`,
/// [`Stage3OutputClaims`], and [`Stage3ClearOutput`] for Stage 4 and later stages.
///
/// This single entrypoint is shared across feature modes. ZK committed proof
/// component assembly is layered on top of the same gamma derivation and
/// statement order.
struct Stage3RegularBatchProofOutput<F: Field, C> {
    proof: SumcheckProof<F, C>,
    #[cfg(feature = "zk")]
    committed_witness: Option<CommittedSumcheckWitness<F>>,
    #[cfg(feature = "zk")]
    output_claim_values: Option<Vec<F>>,
    claims: Stage3OutputClaims<F>,
    verifier_output: Stage3ClearOutput<F>,
}

fn prove_stage3_regular_batch_sumcheck_with_recorder<F, W, B, T, S>(
    input: &Stage3ProverInput<'_, F, W>,
    prefix: &Stage3RegularBatchPrefixOutput<F>,
    transcript: &mut T,
    backend: &mut B,
    mut proof_recorder: S,
) -> Result<Stage3RegularBatchProofOutput<F, S::Commitment>, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>
        + JoltVmStage3ShiftRows
        + JoltVmStage3InstructionRegisterRows,
    B: SumcheckBackend<F, JoltVmNamespace> + Stage3SpartanSumcheckBackend<F>,
    T: Transcript<Challenge = F>,
    S: SumcheckRecorder<F>,
{
    let shift_request =
        build_stage3_shift_state_request(input.config, input.stage2, prefix, input.witness)?;
    let mut shift_state = backend.materialize_sumcheck_stage3_shift_state(&shift_request)?;
    let instances = build_stage3_instruction_registers_regular_batch_instances(
        input.config,
        input.stage2,
        prefix,
        input.witness,
        backend,
    )?;
    let mut regular_state =
        SumcheckRegularBatchState::new("stage3.instruction_registers_regular_batch", instances);
    let max_num_rounds = input.config.log_t;
    for instance in &regular_state.instances {
        if instance.num_rounds() != max_num_rounds {
            return Err(invalid_sumcheck_output(format!(
                "Stage 3 instance {} has {} rounds, expected {max_num_rounds}",
                instance.label,
                instance.num_rounds()
            )));
        }
    }

    proof_recorder.absorb_input_claims(
        &[
            prefix.input_claims.shift,
            prefix.input_claims.instruction_input,
            prefix.input_claims.registers_claim_reduction,
        ],
        transcript,
    );
    let batching_coefficients = (0..3)
        .map(|_| transcript.challenge_scalar())
        .collect::<Vec<_>>();
    let [shift_coefficient, instruction_coefficient, registers_coefficient] =
        batching_coefficients.as_slice()
    else {
        return Err(invalid_sumcheck_output(
            "Stage 3 batch expected exactly three batching coefficients",
        ));
    };

    let mut shift_claim = prefix.input_claims.shift;
    let mut regular_claims = vec![
        prefix.input_claims.instruction_input,
        prefix.input_claims.registers_claim_reduction,
    ];
    let mut running_claim = *shift_coefficient * shift_claim
        + *instruction_coefficient * regular_claims[0]
        + *registers_coefficient * regular_claims[1];

    let mut challenges = Vec::with_capacity(max_num_rounds);
    for round in 0..max_num_rounds {
        let shift_poly = backend.evaluate_sumcheck_stage3_shift_round(&shift_state, shift_claim)?;
        let messages = backend.evaluate_sumcheck_regular_batch_round(
            &mut regular_state,
            round,
            max_num_rounds,
            &regular_claims,
        )?;
        if messages.len() != regular_claims.len() {
            return Err(invalid_sumcheck_output(format!(
                "Stage 3 instruction/register batch round {round} returned {} messages, expected {}",
                messages.len(),
                regular_claims.len()
            )));
        }
        let mut regular_polys = Vec::with_capacity(regular_claims.len());
        for (expected_index, message) in messages.into_iter().enumerate() {
            if message.instance_index != expected_index {
                return Err(invalid_sumcheck_output(format!(
                    "Stage 3 instruction/register batch round {round} returned instance {}, expected {expected_index}",
                    message.instance_index
                )));
            }
            regular_polys.push(message.polynomial);
        }

        let mut batched_poly = &shift_poly * *shift_coefficient;
        batched_poly += &(&regular_polys[0] * *instruction_coefficient);
        batched_poly += &(&regular_polys[1] * *registers_coefficient);
        let round_sum = batched_poly.evaluate(F::zero()) + batched_poly.evaluate(F::one());
        if round_sum != running_claim {
            return Err(invalid_sumcheck_output(format!(
                "Stage 3 regular batch round {round} sumcheck invariant failed"
            )));
        }

        let batched_poly = trim_round_polynomial(batched_poly);
        let challenge = proof_recorder.absorb_round(&batched_poly, transcript)?;
        running_claim = batched_poly.evaluate(challenge);
        challenges.push(challenge);

        shift_claim = shift_poly.evaluate(challenge);
        for (claim, poly) in regular_claims.iter_mut().zip(regular_polys) {
            *claim = poly.evaluate(challenge);
        }
        backend.bind_sumcheck_stage3_shift_state(&mut shift_state, challenge)?;
        backend.bind_sumcheck_regular_batch_state(
            &mut regular_state,
            round,
            max_num_rounds,
            challenge,
        )?;
    }

    let output_openings =
        stage3_output_openings_from_bound_states(backend, &shift_state, &regular_state)?;

    let dimensions = TraceDimensions::new(input.config.log_t);
    let shift_relation = SpartanShift::new(
        dimensions,
        prefix.shift_gamma,
        input.stage2.product_uniskip.tau_low.clone(),
        input.stage2.batch.product_remainder.opening_point.clone(),
    );
    let instruction_relation = InstructionInput::new(
        dimensions,
        prefix.instruction_gamma,
        input.stage2.batch.product_remainder.opening_point.clone(),
    );
    let registers_relation = RegistersClaimReduction::new(
        dimensions,
        prefix.registers_gamma,
        input.stage2.product_uniskip.tau_low.clone(),
    );
    let shift_inputs = SpartanShiftInputClaims::from_upstream(input.stage1, input.stage2);
    let instruction_inputs = InstructionInputInputClaims::from_upstream(input.stage2);
    let registers_inputs = RegistersClaimReductionInputClaims::from_upstream(input.stage1);

    let to_prover_error = |error: VerifierError| invalid_sumcheck_output(error.to_string());
    let shift_points = shift_relation
        .derive_opening_points(&challenges, &shift_inputs)
        .map_err(to_prover_error)?;
    let instruction_points = instruction_relation
        .derive_opening_points(&challenges, &instruction_inputs)
        .map_err(to_prover_error)?;
    let registers_points = registers_relation
        .derive_opening_points(&challenges, &registers_inputs)
        .map_err(to_prover_error)?;
    let output_claims = stage3_output_claims_with_points(
        &output_openings,
        &shift_points,
        &instruction_points,
        &registers_points,
    );

    let shift_output = shift_relation
        .expected_output(&shift_inputs, &output_claims.shift)
        .map_err(to_prover_error)?;
    let instruction_output = instruction_relation
        .expected_output(&instruction_inputs, &output_claims.instruction_input)
        .map_err(to_prover_error)?;
    let registers_output = registers_relation
        .expected_output(&registers_inputs, &output_claims.registers_claim_reduction)
        .map_err(to_prover_error)?;
    let expected_final_claim = stage3_expected_final_claim(
        &batching_coefficients,
        shift_output,
        instruction_output,
        registers_output,
    )
    .map_err(to_prover_error)?;
    if running_claim != expected_final_claim {
        return Err(invalid_sumcheck_output(
            "Stage 3 batch final claim did not match output openings",
        ));
    }

    // Enforce the cross-relation opening aliases (mirrors the verifier's validate).
    output_openings.validate().map_err(to_prover_error)?;

    let output_claim_values = output_openings.opening_values();
    let verifier_output = Stage3ClearOutput {
        challenges: Stage3Challenges {
            shift_gamma: prefix.shift_gamma,
            instruction_gamma: prefix.instruction_gamma,
            registers_gamma: prefix.registers_gamma,
        },
        output_claims,
    };
    let recorded = proof_recorder.finish(&output_claim_values, transcript)?;

    Ok(Stage3RegularBatchProofOutput {
        proof: recorded.proof,
        #[cfg(feature = "zk")]
        committed_witness: recorded.committed_witness,
        #[cfg(feature = "zk")]
        output_claim_values: recorded.output_claim_values,
        claims: output_openings,
        verifier_output,
    })
}

pub fn prove<F, W, B, T, C>(
    input: Stage3ProverInput<'_, F, W>,
    backend: &mut B,
    transcript: &mut T,
) -> Result<Stage3ProofComponent<F, SumcheckProof<F, C>>, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>
        + JoltVmStage3ShiftRows
        + JoltVmStage3InstructionRegisterRows,
    B: SumcheckBackend<F, JoltVmNamespace> + Stage3SpartanSumcheckBackend<F>,
    T: Transcript<Challenge = F>,
{
    if input.checked.zk {
        return Err(ProverError::InvalidStageRequest {
            reason: "Stage 3 clear prover received ZK checked inputs".to_owned(),
        });
    }
    if input.checked.trace_length != (1usize << input.config.log_t) {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 3 checked trace length {} does not match log_t {}",
                input.checked.trace_length, input.config.log_t
            ),
        });
    }

    let prefix =
        derive_stage3_regular_batch_prefix(input.config, input.stage1, input.stage2, transcript)?;
    let output = prove_stage3_regular_batch_sumcheck_with_recorder(
        &input,
        &prefix,
        transcript,
        backend,
        ClearSumcheckRecorder::<F, C>::new(input.config.log_t),
    )?;

    Ok(Stage3ProofComponent {
        stage3_sumcheck_proof: output.proof,
        claims: output.claims,
        verifier_output: output.verifier_output,
    })
}

#[cfg(feature = "zk")]
pub fn prove_committed_proof_component<F, W, B, T, VC>(
    input: Stage3ProverInput<'_, F, W>,
    backend: &mut B,
    transcript: &mut T,
    vc_setup: &VC::Setup,
) -> Result<Stage3CommittedProofComponent<F, VC>, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>
        + JoltVmStage3ShiftRows
        + JoltVmStage3InstructionRegisterRows,
    B: SumcheckBackend<F, JoltVmNamespace> + Stage3SpartanSumcheckBackend<F>,
    T: Transcript<Challenge = F>,
    VC: VectorCommitment<Field = F>,
{
    if !input.checked.zk {
        return Err(ProverError::InvalidStageRequest {
            reason: "Stage 3 committed proof component prover received transparent checked inputs"
                .to_owned(),
        });
    }
    if input.checked.trace_length != (1usize << input.config.log_t) {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 3 checked trace length {} does not match log_t {}",
                input.checked.trace_length, input.config.log_t
            ),
        });
    }

    let prefix =
        derive_stage3_regular_batch_prefix(input.config, input.stage1, input.stage2, transcript)?;
    let output = prove_stage3_regular_batch_sumcheck_with_recorder(
        &input,
        &prefix,
        transcript,
        backend,
        CommittedSumcheckRecorder::<F, VC>::new(vc_setup)?,
    )?;

    Ok(Stage3CommittedProofComponent {
        stage3_sumcheck_proof: output.proof,
        output_claim_values: output.output_claim_values.ok_or_else(|| {
            invalid_sumcheck_output("Stage 3 committed output claim values are missing")
        })?,
        verifier_output: output.verifier_output,
        committed_witness: output.committed_witness.ok_or_else(|| {
            invalid_sumcheck_output("Stage 3 committed witness material is missing")
        })?,
    })
}

#[expect(
    dead_code,
    reason = "return type of the retained parity-audit reference prover"
)]
struct Stage3RegularBatch<F: Field, C> {
    proof: SumcheckProof<F, C>,
    challenges: Vec<F>,
    batching_coefficients: Vec<F>,
    output_claim: F,
    output_openings: Option<Stage3OutputClaims<F>>,
}

fn build_stage3_output_opening_materialization_request<F, W>(
    witness: &W,
) -> Result<SumcheckMaterializationRequest<JoltVmNamespace>, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>,
{
    let mut views = Vec::new();
    views.extend(build_stage3_opening_materializations(
        witness,
        &spartan::shift_output_openings(),
        stage3_shift_opening_slot,
    )?);
    views.extend(build_stage3_opening_materializations(
        witness,
        &instruction::input_virtualization_output_openings(),
        stage3_instruction_input_opening_slot,
    )?);
    views.extend(build_stage3_opening_materializations(
        witness,
        &registers_claim_reduction::claim_reduction_output_openings(),
        stage3_registers_claim_reduction_opening_slot,
    )?);

    Ok(SumcheckMaterializationRequest::new(
        "stage3.batch.output_materializations",
        views,
    ))
}

fn build_stage3_opening_materializations<F, W>(
    witness: &W,
    openings: &[JoltOpeningId],
    slot_for_index: impl Fn(usize) -> BackendValueSlot,
) -> Result<Vec<SumcheckViewMaterializationRequest<JoltVmNamespace>>, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>,
{
    openings
        .iter()
        .copied()
        .enumerate()
        .map(|(index, opening)| {
            let view = view_requirement_from_jolt_opening::<F, W>(witness, opening)?;
            Ok(SumcheckViewMaterializationRequest::new(
                slot_for_index(index),
                view,
            ))
        })
        .collect()
}

/// Builds the three Stage 3 regular-batch instances (Spartan shift, instruction
/// input, register claim reduction) for the backend sumcheck kernel.
///
/// Each statement is a sum of products, so it maps onto one
/// [`SumcheckRegularBatchInstance`] carrying several product terms. The
/// instance polynomials are bit-reversed so the kernel's `HighToLow` binding
/// reproduces the canonical low-to-high cycle order (matching Stage 2).
#[expect(
    dead_code,
    reason = "dense Stage 3 batch assembly is retained as a parity-audit reference"
)]
fn build_stage3_regular_batch_instances<F, W, B>(
    config: Stage3ProverConfig,
    stage2: &Stage2ClearOutput<F>,
    prefix: &Stage3RegularBatchPrefixOutput<F>,
    witness: &W,
    backend: &mut B,
) -> Result<Vec<SumcheckRegularBatchInstance<F>>, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>,
    B: SumcheckBackend<F, JoltVmNamespace>,
{
    let request = build_stage3_output_opening_materialization_request::<F, W>(witness)?;
    let materializations = backend.materialize_sumcheck_views(&request, witness)?;
    let materialized = stage3_materialized_openings_from_outputs(materializations)?;

    let eq_plus_one_outer = EqPlusOnePolynomial::evals(&stage2.product_uniskip.tau_low, None).1;
    let eq_plus_one_product =
        EqPlusOnePolynomial::evals(&stage2.batch.product_remainder.opening_point, None).1;
    let eq_product =
        EqPolynomial::new(stage2.batch.product_remainder.opening_point.clone()).evaluations();
    let eq_spartan = EqPolynomial::new(stage2.product_uniskip.tau_low.clone()).evaluations();

    let shift_gamma = prefix.shift_gamma;
    let shift_gamma2 = shift_gamma * shift_gamma;
    let shift_gamma3 = shift_gamma2 * shift_gamma;
    let shift_gamma4 = shift_gamma3 * shift_gamma;
    let instruction_gamma = prefix.instruction_gamma;
    let registers_gamma = prefix.registers_gamma;
    let registers_gamma2 = registers_gamma * registers_gamma;

    let shift = SumcheckRegularBatchInstance::new_products(
        "stage3.shift",
        prefix.input_claims.shift,
        vec![
            stage3_reversed_polynomial(config, "shift outer eq+1", eq_plus_one_outer)?,
            stage3_reversed_polynomial(config, "shift product eq+1", eq_plus_one_product)?,
            stage3_reversed_polynomial(
                config,
                "shift unexpanded PC",
                materialized.shift.unexpanded_pc,
            )?,
            stage3_reversed_polynomial(config, "shift PC", materialized.shift.pc)?,
            stage3_reversed_polynomial(
                config,
                "shift virtual flag",
                materialized.shift.is_virtual,
            )?,
            stage3_reversed_polynomial(
                config,
                "shift first-in-sequence flag",
                materialized.shift.is_first_in_sequence,
            )?,
            stage3_reversed_polynomial(config, "shift noop flag", materialized.shift.is_noop)?,
        ],
        vec![
            SumcheckRegularBatchProduct::new(
                F::one(),
                vec![
                    batch_factor(vec![batch_term(0, F::one())]),
                    batch_factor(vec![
                        batch_term(2, F::one()),
                        batch_term(3, shift_gamma),
                        batch_term(4, shift_gamma2),
                        batch_term(5, shift_gamma3),
                    ]),
                ],
            ),
            SumcheckRegularBatchProduct::new(
                shift_gamma4,
                vec![
                    batch_factor(vec![batch_term(1, F::one())]),
                    SumcheckRegularBatchLinearFactor::new(F::one(), vec![batch_term(6, neg_one())]),
                ],
            ),
        ],
    );

    let instruction_input = SumcheckRegularBatchInstance::new_products(
        "stage3.instruction_input",
        prefix.input_claims.instruction_input,
        vec![
            stage3_reversed_polynomial(config, "instruction product eq", eq_product)?,
            stage3_reversed_polynomial(
                config,
                "instruction right operand is rs2",
                materialized.instruction_input.right_operand_is_rs2,
            )?,
            stage3_reversed_polynomial(
                config,
                "instruction rs2 value",
                materialized.instruction_input.rs2_value,
            )?,
            stage3_reversed_polynomial(
                config,
                "instruction right operand is imm",
                materialized.instruction_input.right_operand_is_imm,
            )?,
            stage3_reversed_polynomial(
                config,
                "instruction immediate",
                materialized.instruction_input.imm,
            )?,
            stage3_reversed_polynomial(
                config,
                "instruction left operand is rs1",
                materialized.instruction_input.left_operand_is_rs1,
            )?,
            stage3_reversed_polynomial(
                config,
                "instruction rs1 value",
                materialized.instruction_input.rs1_value,
            )?,
            stage3_reversed_polynomial(
                config,
                "instruction left operand is PC",
                materialized.instruction_input.left_operand_is_pc,
            )?,
            stage3_reversed_polynomial(
                config,
                "instruction unexpanded PC",
                materialized.instruction_input.unexpanded_pc,
            )?,
        ],
        vec![
            SumcheckRegularBatchProduct::new(
                F::one(),
                vec![
                    batch_factor(vec![batch_term(0, F::one())]),
                    batch_factor(vec![batch_term(1, F::one())]),
                    batch_factor(vec![batch_term(2, F::one())]),
                ],
            ),
            SumcheckRegularBatchProduct::new(
                F::one(),
                vec![
                    batch_factor(vec![batch_term(0, F::one())]),
                    batch_factor(vec![batch_term(3, F::one())]),
                    batch_factor(vec![batch_term(4, F::one())]),
                ],
            ),
            SumcheckRegularBatchProduct::new(
                instruction_gamma,
                vec![
                    batch_factor(vec![batch_term(0, F::one())]),
                    batch_factor(vec![batch_term(5, F::one())]),
                    batch_factor(vec![batch_term(6, F::one())]),
                ],
            ),
            SumcheckRegularBatchProduct::new(
                instruction_gamma,
                vec![
                    batch_factor(vec![batch_term(0, F::one())]),
                    batch_factor(vec![batch_term(7, F::one())]),
                    batch_factor(vec![batch_term(8, F::one())]),
                ],
            ),
        ],
    );

    let registers_claim_reduction = SumcheckRegularBatchInstance::new_products(
        "stage3.registers_claim_reduction",
        prefix.input_claims.registers_claim_reduction,
        vec![
            stage3_reversed_polynomial(config, "registers Spartan eq", eq_spartan)?,
            stage3_reversed_polynomial(
                config,
                "registers rd write value",
                materialized.registers_claim_reduction.rd_write,
            )?,
            stage3_reversed_polynomial(
                config,
                "registers rs1 value",
                materialized.registers_claim_reduction.rs1,
            )?,
            stage3_reversed_polynomial(
                config,
                "registers rs2 value",
                materialized.registers_claim_reduction.rs2,
            )?,
        ],
        vec![SumcheckRegularBatchProduct::new(
            F::one(),
            vec![
                batch_factor(vec![batch_term(0, F::one())]),
                batch_factor(vec![
                    batch_term(1, F::one()),
                    batch_term(2, registers_gamma),
                    batch_term(3, registers_gamma2),
                ]),
            ],
        )],
    );

    Ok(vec![shift, instruction_input, registers_claim_reduction])
}

fn build_stage3_shift_state_request<F, W>(
    config: Stage3ProverConfig,
    stage2: &Stage2ClearOutput<F>,
    prefix: &Stage3RegularBatchPrefixOutput<F>,
    witness: &W,
) -> Result<SumcheckStage3ShiftStateRequest<F>, ProverError>
where
    F: Field,
    W: JoltVmStage3ShiftRows,
{
    let rows = stage3_shift_rows(witness, config.log_t)?;
    Ok(SumcheckStage3ShiftStateRequest::new(
        "stage3.shift.prefix_suffix",
        config.log_t,
        stage2.product_uniskip.tau_low.clone(),
        stage2.batch.product_remainder.opening_point.clone(),
        prefix.shift_gamma,
        rows,
    ))
}

fn stage3_instruction_register_reversed_columns_from_rows<F, W>(
    config: Stage3ProverConfig,
    witness: &W,
    bit_reversed_indices: &[usize],
    eq_product: &[F],
    eq_spartan: &[F],
) -> Result<Vec<Vec<F>>, ProverError>
where
    F: Field,
    W: JoltVmStage3InstructionRegisterRows,
{
    let rows = witness.stage3_instruction_register_rows(config.log_t)?;
    let row_count = checked_stage3_len(config.log_t, "Stage 3 instruction/register trace length")?;
    validate_stage3_row_count(config, rows.len(), "instruction/register row cache")?;
    if bit_reversed_indices.len() != row_count {
        return Err(invalid_sumcheck_output(format!(
            "Stage 3 instruction/register bit-reversal table has {} entries, expected {row_count}",
            bit_reversed_indices.len()
        )));
    }
    if eq_product.len() != row_count || eq_spartan.len() != row_count {
        return Err(invalid_sumcheck_output(format!(
            "Stage 3 instruction/register eq tables have lengths {} and {}, expected {row_count}",
            eq_product.len(),
            eq_spartan.len()
        )));
    }
    let mut instruction_eq_product = unsafe_allocate_zero_vec(row_count);
    let mut right_operand_is_rs2 = unsafe_allocate_zero_vec(row_count);
    let mut rs2_value = unsafe_allocate_zero_vec(row_count);
    let mut right_operand_is_imm = unsafe_allocate_zero_vec(row_count);
    let mut imm = unsafe_allocate_zero_vec(row_count);
    let mut left_operand_is_rs1 = unsafe_allocate_zero_vec(row_count);
    let mut rs1_value = unsafe_allocate_zero_vec(row_count);
    let mut left_operand_is_pc = unsafe_allocate_zero_vec(row_count);
    let mut unexpanded_pc = unsafe_allocate_zero_vec(row_count);
    let mut registers_eq_spartan = unsafe_allocate_zero_vec(row_count);
    let mut rd_write_value = unsafe_allocate_zero_vec(row_count);
    let mut registers_rs1_value = unsafe_allocate_zero_vec(row_count);
    let mut registers_rs2_value = unsafe_allocate_zero_vec(row_count);

    instruction_eq_product
        .par_chunks_mut(STAGE3_INSTRUCTION_REGISTER_MATERIALIZE_CHUNK_ROWS)
        .zip(
            right_operand_is_rs2.par_chunks_mut(STAGE3_INSTRUCTION_REGISTER_MATERIALIZE_CHUNK_ROWS),
        )
        .zip(rs2_value.par_chunks_mut(STAGE3_INSTRUCTION_REGISTER_MATERIALIZE_CHUNK_ROWS))
        .zip(
            right_operand_is_imm.par_chunks_mut(STAGE3_INSTRUCTION_REGISTER_MATERIALIZE_CHUNK_ROWS),
        )
        .zip(imm.par_chunks_mut(STAGE3_INSTRUCTION_REGISTER_MATERIALIZE_CHUNK_ROWS))
        .zip(left_operand_is_rs1.par_chunks_mut(STAGE3_INSTRUCTION_REGISTER_MATERIALIZE_CHUNK_ROWS))
        .zip(rs1_value.par_chunks_mut(STAGE3_INSTRUCTION_REGISTER_MATERIALIZE_CHUNK_ROWS))
        .zip(left_operand_is_pc.par_chunks_mut(STAGE3_INSTRUCTION_REGISTER_MATERIALIZE_CHUNK_ROWS))
        .zip(unexpanded_pc.par_chunks_mut(STAGE3_INSTRUCTION_REGISTER_MATERIALIZE_CHUNK_ROWS))
        .zip(
            registers_eq_spartan.par_chunks_mut(STAGE3_INSTRUCTION_REGISTER_MATERIALIZE_CHUNK_ROWS),
        )
        .zip(rd_write_value.par_chunks_mut(STAGE3_INSTRUCTION_REGISTER_MATERIALIZE_CHUNK_ROWS))
        .zip(registers_rs1_value.par_chunks_mut(STAGE3_INSTRUCTION_REGISTER_MATERIALIZE_CHUNK_ROWS))
        .zip(registers_rs2_value.par_chunks_mut(STAGE3_INSTRUCTION_REGISTER_MATERIALIZE_CHUNK_ROWS))
        .enumerate()
        .for_each(|(chunk_index, chunks)| {
            let (
                (
                    (
                        (
                            (
                                (
                                    (
                                        (
                                            (
                                                (
                                                    (
                                                        (
                                                            instruction_eq_product,
                                                            right_operand_is_rs2,
                                                        ),
                                                        rs2_value,
                                                    ),
                                                    right_operand_is_imm,
                                                ),
                                                imm,
                                            ),
                                            left_operand_is_rs1,
                                        ),
                                        rs1_value,
                                    ),
                                    left_operand_is_pc,
                                ),
                                unexpanded_pc,
                            ),
                            registers_eq_spartan,
                        ),
                        rd_write_value,
                    ),
                    registers_rs1_value,
                ),
                registers_rs2_value,
            ) = chunks;
            let start = chunk_index * STAGE3_INSTRUCTION_REGISTER_MATERIALIZE_CHUNK_ROWS;
            for offset in 0..instruction_eq_product.len() {
                let cycle = bit_reversed_indices[start + offset];
                let row = &rows[cycle];
                let rs2 = F::from_u64(row.rs2_value);
                let rs1 = F::from_u64(row.rs1_value);

                instruction_eq_product[offset] = eq_product[cycle];
                right_operand_is_rs2[offset] = F::from_bool(row.right_operand_is_rs2);
                rs2_value[offset] = rs2;
                right_operand_is_imm[offset] = F::from_bool(row.right_operand_is_imm);
                imm[offset] = F::from_i128(row.imm);
                left_operand_is_rs1[offset] = F::from_bool(row.left_operand_is_rs1);
                rs1_value[offset] = rs1;
                left_operand_is_pc[offset] = F::from_bool(row.left_operand_is_pc);
                unexpanded_pc[offset] = F::from_u64(row.unexpanded_pc);
                registers_eq_spartan[offset] = eq_spartan[cycle];
                rd_write_value[offset] = F::from_u64(row.rd_write_value);
                registers_rs1_value[offset] = rs1;
                registers_rs2_value[offset] = rs2;
            }
        });

    Ok(vec![
        instruction_eq_product,
        right_operand_is_rs2,
        rs2_value,
        right_operand_is_imm,
        imm,
        left_operand_is_rs1,
        rs1_value,
        left_operand_is_pc,
        unexpanded_pc,
        registers_eq_spartan,
        rd_write_value,
        registers_rs1_value,
        registers_rs2_value,
    ])
}

fn build_stage3_instruction_registers_regular_batch_instances<F, W, B>(
    config: Stage3ProverConfig,
    stage2: &Stage2ClearOutput<F>,
    prefix: &Stage3RegularBatchPrefixOutput<F>,
    witness: &W,
    _backend: &mut B,
) -> Result<Vec<SumcheckRegularBatchInstance<F>>, ProverError>
where
    F: Field,
    W: JoltVmStage3InstructionRegisterRows,
    B: SumcheckBackend<F, JoltVmNamespace>,
{
    let bit_reversed_indices = stage3_bit_reversed_indices(config)?;

    let eq_product =
        EqPolynomial::new(stage2.batch.product_remainder.opening_point.clone()).evaluations();
    let eq_spartan = EqPolynomial::new(stage2.product_uniskip.tau_low.clone()).evaluations();
    let mut columns = stage3_instruction_register_reversed_columns_from_rows(
        config,
        witness,
        &bit_reversed_indices,
        &eq_product,
        &eq_spartan,
    )?
    .into_iter();
    let mut next_column = |label: &'static str| {
        columns.next().ok_or_else(|| {
            invalid_sumcheck_output(format!("Stage 3 missing materialized column {label}"))
        })
    };
    let instruction_gamma = prefix.instruction_gamma;
    let registers_gamma = prefix.registers_gamma;
    let registers_gamma2 = registers_gamma * registers_gamma;

    let instruction_input = SumcheckRegularBatchInstance::new_products(
        "stage3.instruction_input",
        prefix.input_claims.instruction_input,
        vec![
            Polynomial::new(next_column("instruction product eq")?),
            Polynomial::new(next_column("instruction right operand is rs2")?),
            Polynomial::new(next_column("instruction rs2 value")?),
            Polynomial::new(next_column("instruction right operand is imm")?),
            Polynomial::new(next_column("instruction immediate")?),
            Polynomial::new(next_column("instruction left operand is rs1")?),
            Polynomial::new(next_column("instruction rs1 value")?),
            Polynomial::new(next_column("instruction left operand is PC")?),
            Polynomial::new(next_column("instruction unexpanded PC")?),
        ],
        vec![
            SumcheckRegularBatchProduct::new(
                F::one(),
                vec![
                    batch_factor(vec![batch_term(0, F::one())]),
                    batch_factor(vec![batch_term(1, F::one())]),
                    batch_factor(vec![batch_term(2, F::one())]),
                ],
            ),
            SumcheckRegularBatchProduct::new(
                F::one(),
                vec![
                    batch_factor(vec![batch_term(0, F::one())]),
                    batch_factor(vec![batch_term(3, F::one())]),
                    batch_factor(vec![batch_term(4, F::one())]),
                ],
            ),
            SumcheckRegularBatchProduct::new(
                instruction_gamma,
                vec![
                    batch_factor(vec![batch_term(0, F::one())]),
                    batch_factor(vec![batch_term(5, F::one())]),
                    batch_factor(vec![batch_term(6, F::one())]),
                ],
            ),
            SumcheckRegularBatchProduct::new(
                instruction_gamma,
                vec![
                    batch_factor(vec![batch_term(0, F::one())]),
                    batch_factor(vec![batch_term(7, F::one())]),
                    batch_factor(vec![batch_term(8, F::one())]),
                ],
            ),
        ],
    );

    let registers_claim_reduction = SumcheckRegularBatchInstance::new_products(
        "stage3.registers_claim_reduction",
        prefix.input_claims.registers_claim_reduction,
        vec![
            Polynomial::new(next_column("registers Spartan eq")?),
            Polynomial::new(next_column("registers rd write value")?),
            Polynomial::new(next_column("registers rs1 value")?),
            Polynomial::new(next_column("registers rs2 value")?),
        ],
        vec![SumcheckRegularBatchProduct::new(
            F::one(),
            vec![
                batch_factor(vec![batch_term(0, F::one())]),
                batch_factor(vec![
                    batch_term(1, F::one()),
                    batch_term(2, registers_gamma),
                    batch_term(3, registers_gamma2),
                ]),
            ],
        )],
    );

    Ok(vec![instruction_input, registers_claim_reduction])
}

fn stage3_output_openings_from_bound_states<F, B>(
    backend: &mut B,
    shift_state: &B::Stage3ShiftState,
    regular_state: &SumcheckRegularBatchState<F>,
) -> Result<Stage3OutputClaims<F>, ProverError>
where
    F: Field,
    B: Stage3SpartanSumcheckBackend<F>,
{
    let [unexpanded_pc, pc, is_virtual, is_first_in_sequence, is_noop] =
        backend.stage3_shift_output_openings(shift_state)?;
    let instruction_instance = regular_state.instances.first().ok_or_else(|| {
        invalid_sumcheck_output("Stage 3 regular state is missing instruction input instance")
    })?;
    let registers_instance = regular_state.instances.get(1).ok_or_else(|| {
        invalid_sumcheck_output("Stage 3 regular state is missing registers instance")
    })?;

    Ok(Stage3OutputClaims {
        shift: SpartanShiftOutputClaims {
            unexpanded_pc,
            pc,
            is_virtual,
            is_first_in_sequence,
            is_noop,
        },
        instruction_input: InstructionInputOutputClaims {
            right_operand_is_rs2: final_regular_polynomial_value(
                instruction_instance,
                1,
                "instruction right operand is rs2",
            )?,
            rs2_value: final_regular_polynomial_value(
                instruction_instance,
                2,
                "instruction rs2 value",
            )?,
            right_operand_is_imm: final_regular_polynomial_value(
                instruction_instance,
                3,
                "instruction right operand is imm",
            )?,
            imm: final_regular_polynomial_value(instruction_instance, 4, "instruction immediate")?,
            left_operand_is_rs1: final_regular_polynomial_value(
                instruction_instance,
                5,
                "instruction left operand is rs1",
            )?,
            rs1_value: final_regular_polynomial_value(
                instruction_instance,
                6,
                "instruction rs1 value",
            )?,
            left_operand_is_pc: final_regular_polynomial_value(
                instruction_instance,
                7,
                "instruction left operand is PC",
            )?,
            unexpanded_pc: final_regular_polynomial_value(
                instruction_instance,
                8,
                "instruction unexpanded PC",
            )?,
        },
        registers_claim_reduction: RegistersClaimReductionOutputClaims {
            rd_write_value: final_regular_polynomial_value(
                registers_instance,
                1,
                "registers rd write value",
            )?,
            rs1_value: final_regular_polynomial_value(
                registers_instance,
                2,
                "registers rs1 value",
            )?,
            rs2_value: final_regular_polynomial_value(
                registers_instance,
                3,
                "registers rs2 value",
            )?,
        },
    })
}

fn final_regular_polynomial_value<F: Field>(
    instance: &SumcheckRegularBatchInstance<F>,
    polynomial_index: usize,
    label: &'static str,
) -> Result<F, ProverError> {
    let polynomial = instance.polynomials.get(polynomial_index).ok_or_else(|| {
        invalid_sumcheck_output(format!(
            "Stage 3 regular instance {} is missing polynomial {polynomial_index} ({label})",
            instance.label
        ))
    })?;
    let [value] = polynomial.evaluations() else {
        return Err(invalid_sumcheck_output(format!(
            "Stage 3 regular instance {} polynomial {polynomial_index} ({label}) has {} evaluations, expected 1",
            instance.label,
            polynomial.len()
        )));
    };
    Ok(*value)
}

/// Dense/materialized Stage 3 batch driver retained as a reference path for
/// correctness tests. Production proving should use the prefix/suffix shift
/// path above.
#[expect(
    dead_code,
    reason = "dense Stage 3 batch prover is retained as a parity-audit reference"
)]
fn prove_stage3_regular_batch_sumcheck_dense_reference<F, T, C, B>(
    config: Stage3ProverConfig,
    instances: Vec<SumcheckRegularBatchInstance<F>>,
    transcript: &mut T,
    backend: &mut B,
) -> Result<Stage3RegularBatch<F, C>, ProverError>
where
    F: Field,
    T: Transcript<Challenge = F>,
    B: SumcheckBackend<F, JoltVmNamespace>,
{
    let mut state = SumcheckRegularBatchState::new("stage3.regular_batch", instances);
    let max_num_rounds = config.log_t;
    for instance in &state.instances {
        if instance.num_rounds() != max_num_rounds {
            return Err(invalid_sumcheck_output(format!(
                "Stage 3 instance {} has {} rounds, expected {max_num_rounds}",
                instance.label,
                instance.num_rounds()
            )));
        }
        append_sumcheck_claim(transcript, &instance.input_claim);
    }
    let instance_count = state.instances.len();
    let batching_coefficients = (0..instance_count)
        .map(|_| transcript.challenge_scalar())
        .collect::<Vec<_>>();
    let mut individual_claims = state
        .instances
        .iter()
        .map(|instance| instance.input_claim)
        .collect::<Vec<_>>();
    let mut running_claim = individual_claims
        .iter()
        .zip(&batching_coefficients)
        .map(|(claim, coefficient)| *claim * *coefficient)
        .sum::<F>();

    let mut challenges = Vec::with_capacity(max_num_rounds);
    let mut round_polynomials = Vec::with_capacity(max_num_rounds);
    for round in 0..max_num_rounds {
        let messages = backend.evaluate_sumcheck_regular_batch_round(
            &mut state,
            round,
            max_num_rounds,
            &individual_claims,
        )?;
        if messages.len() != instance_count {
            return Err(invalid_sumcheck_output(format!(
                "Stage 3 regular batch round {round} returned {} messages, expected {instance_count}",
                messages.len()
            )));
        }
        let mut univariate_polys = Vec::with_capacity(instance_count);
        for (expected_index, message) in messages.into_iter().enumerate() {
            if message.instance_index != expected_index {
                return Err(invalid_sumcheck_output(format!(
                    "Stage 3 regular batch round {round} returned instance {}, expected {expected_index}",
                    message.instance_index
                )));
            }
            univariate_polys.push(message.polynomial);
        }
        let mut batched_poly = UnivariatePoly::zero();
        for (poly, coefficient) in univariate_polys.iter().zip(&batching_coefficients) {
            batched_poly += &(poly * *coefficient);
        }
        let round_sum = batched_poly.evaluate(F::zero()) + batched_poly.evaluate(F::one());
        if round_sum != running_claim {
            return Err(invalid_sumcheck_output(format!(
                "Stage 3 regular batch round {round} sumcheck invariant failed"
            )));
        }
        let batched_poly = trim_round_polynomial(batched_poly);
        CompressedLabeledRoundPoly::sumcheck(&batched_poly).append_to_transcript(transcript);
        let challenge = transcript.challenge();
        running_claim = batched_poly.evaluate(challenge);
        challenges.push(challenge);
        round_polynomials.push(batched_poly.compress());
        for (claim, poly) in individual_claims.iter_mut().zip(univariate_polys) {
            *claim = poly.evaluate(challenge);
        }
        backend.bind_sumcheck_regular_batch_state(&mut state, round, max_num_rounds, challenge)?;
    }

    Ok(Stage3RegularBatch {
        proof: SumcheckProof::Clear(ClearProof::Compressed(CompressedSumcheckProof {
            round_polynomials,
        })),
        challenges,
        batching_coefficients,
        output_claim: running_claim,
        output_openings: None,
    })
}

fn batch_term<F: Field>(polynomial: usize, coefficient: F) -> SumcheckRegularBatchLinearTerm<F> {
    SumcheckRegularBatchLinearTerm::new(polynomial, coefficient)
}

fn batch_factor<F: Field>(
    terms: Vec<SumcheckRegularBatchLinearTerm<F>>,
) -> SumcheckRegularBatchLinearFactor<F> {
    SumcheckRegularBatchLinearFactor::from_terms(terms)
}

fn neg_one<F: Field>() -> F {
    F::zero() - F::one()
}

fn stage3_reversed_polynomial<F: Field>(
    config: Stage3ProverConfig,
    label: &'static str,
    values: Vec<F>,
) -> Result<Polynomial<F>, ProverError> {
    let bit_reversed_indices = stage3_bit_reversed_indices(config)?;
    stage3_reversed_polynomial_with_indices(config, label, values, &bit_reversed_indices)
}

fn stage3_reversed_polynomial_with_indices<F: Field>(
    config: Stage3ProverConfig,
    label: &'static str,
    values: Vec<F>,
    bit_reversed_indices: &[usize],
) -> Result<Polynomial<F>, ProverError> {
    let expected_len = 1usize.checked_shl(config.log_t as u32).ok_or_else(|| {
        invalid_sumcheck_output(format!(
            "Stage 3 {label} trace length overflows for log_t={}",
            config.log_t
        ))
    })?;
    if values.len() != expected_len {
        return Err(invalid_sumcheck_output(format!(
            "Stage 3 {label} materialized {} rows, expected {expected_len}",
            values.len()
        )));
    }
    if bit_reversed_indices.len() != expected_len {
        return Err(invalid_sumcheck_output(format!(
            "Stage 3 {label} bit-reversal table has {} entries, expected {expected_len}",
            bit_reversed_indices.len()
        )));
    }
    let reversed = bit_reversed_indices
        .iter()
        .map(|&index| values[index])
        .collect::<Vec<_>>();
    Ok(Polynomial::from(reversed))
}

fn stage3_bit_reversed_indices(config: Stage3ProverConfig) -> Result<Vec<usize>, ProverError> {
    let expected_len = 1usize.checked_shl(config.log_t as u32).ok_or_else(|| {
        invalid_sumcheck_output(format!(
            "Stage 3 trace length overflows for log_t={}",
            config.log_t
        ))
    })?;
    Ok((0..expected_len)
        .map(|index| bit_reverse(index, config.log_t))
        .collect())
}

fn bit_reverse(index: usize, bits: usize) -> usize {
    let mut reversed = 0usize;
    for position in 0..bits {
        reversed <<= 1;
        reversed |= (index >> position) & 1;
    }
    reversed
}

fn trim_round_polynomial<F: Field>(poly: UnivariatePoly<F>) -> UnivariatePoly<F> {
    let mut coefficients = poly.into_coefficients();
    while coefficients.len() > 2 && coefficients.last().is_some_and(|value| *value == F::zero()) {
        let _ = coefficients.pop();
    }
    UnivariatePoly::new(coefficients)
}

fn derive_stage3_regular_batch_prefix<F, T>(
    config: Stage3ProverConfig,
    stage1: &Stage1ClearOutput<F>,
    stage2: &Stage2ClearOutput<F>,
    transcript: &mut T,
) -> Result<Stage3RegularBatchPrefixOutput<F>, ProverError>
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    let expected_rounds = config.log_t;
    if stage2.batch.product_remainder.opening_point.len() != expected_rounds {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 3 product-remainder dependency has {} variables, expected {expected_rounds}",
                stage2.batch.product_remainder.opening_point.len()
            ),
        });
    }
    if stage2.batch.instruction_claim_reduction.opening_point
        != stage2.batch.product_remainder.opening_point
    {
        return Err(ProverError::InvalidStageRequest {
            reason: "Stage 3 instruction input dependencies use different opening points"
                .to_owned(),
        });
    }
    if stage2.product_uniskip.tau_low.len() != expected_rounds {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 3 product uni-skip dependency has {} variables, expected {expected_rounds}",
                stage2.product_uniskip.tau_low.len()
            ),
        });
    }

    let shift_gamma = transcript.challenge_scalar();
    let instruction_gamma = transcript.challenge_scalar();
    let registers_gamma = transcript.challenge_scalar();

    let dimensions = TraceDimensions::new(expected_rounds);
    let shift_relation = SpartanShift::new(
        dimensions,
        shift_gamma,
        stage2.product_uniskip.tau_low.clone(),
        stage2.batch.product_remainder.opening_point.clone(),
    );
    let instruction_relation = InstructionInput::new(
        dimensions,
        instruction_gamma,
        stage2.batch.product_remainder.opening_point.clone(),
    );
    let registers_relation = RegistersClaimReduction::new(
        dimensions,
        registers_gamma,
        stage2.product_uniskip.tau_low.clone(),
    );
    let to_prover_error = |error: VerifierError| invalid_sumcheck_output(error.to_string());
    let input_claims = Stage3InputClaims {
        shift: shift_relation
            .input_claim(&SpartanShiftInputClaims::from_upstream(stage1, stage2))
            .map_err(to_prover_error)?,
        instruction_input: instruction_relation
            .input_claim(&InstructionInputInputClaims::from_upstream(stage2))
            .map_err(to_prover_error)?,
        registers_claim_reduction: registers_relation
            .input_claim(&RegistersClaimReductionInputClaims::from_upstream(stage1))
            .map_err(to_prover_error)?,
    };

    Ok(Stage3RegularBatchPrefixOutput {
        input_claims,
        shift_gamma,
        instruction_gamma,
        registers_gamma,
    })
}

fn validate_stage3_row_count(
    config: Stage3ProverConfig,
    actual: usize,
    label: &'static str,
) -> Result<(), ProverError> {
    let expected = checked_stage3_len(config.log_t, "Stage 3 trace length")?;
    if actual == expected {
        return Ok(());
    }
    Err(ProverError::InvalidStageRequest {
        reason: format!("Stage 3 {label} has {actual} rows, expected {expected}"),
    })
}

fn checked_stage3_len(log_len: usize, label: &str) -> Result<usize, ProverError> {
    1usize
        .checked_shl(log_len as u32)
        .ok_or_else(|| ProverError::InvalidStageRequest {
            reason: format!("{label} overflows for log length {log_len}"),
        })
}
