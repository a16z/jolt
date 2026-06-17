#[cfg(feature = "zk")]
use crate::committed::{CommittedSumcheckBuilder, CommittedSumcheckWitness};
use crate::stages::invalid_sumcheck_output;
#[cfg(feature = "zk")]
use crate::stages::recorder::CommittedSumcheckRecorder;
use crate::stages::recorder::{ClearSumcheckRecorder, SumcheckRecorder};
use crate::ProverError;
use jolt_backends::{
    ram_read_write_rows_from_trace, stage2_product_instruction_openings_from_rows,
    stage2_product_uniskip_first_round, stage2_ram_state_requests, RamReadWriteSumcheckBackend,
    Stage2ProductUniskipFirstRoundRequest, Stage2RamStateRequests, Stage2RamStateRequestsRequest,
    Stage2RegularBatchInstanceRequest, SumcheckBackend, SumcheckRamOutputCheckStateRequest,
    SumcheckRamRafStateRequest, SumcheckRamReadWriteRow, SumcheckRamReadWriteStateRequest,
    SumcheckRegularBatchInstance, SumcheckRegularBatchState,
};
#[cfg(feature = "field-inline")]
use jolt_backends::{
    stage2_field_inline_factor_openings, stage2_field_inline_materialize_product_factors,
    stage2_field_inline_product_uniskip_extended_evals,
    stage2_field_inline_regular_batch_instances, Stage2FieldInlineMaterializedFactors,
    Stage2FieldInlineProductUniskipEvalRequest, Stage2FieldInlineRegularBatchInstanceRequest,
};
#[cfg(not(feature = "field-inline"))]
use jolt_backends::{
    stage2_product_uniskip_extended_eval_outputs, stage2_product_uniskip_extended_eval_request,
    stage2_product_uniskip_rows_from_stage2_trace, stage2_regular_batch_instances,
};
use jolt_claims::protocols::jolt::JoltReadWriteConfig;
#[cfg(feature = "zk")]
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_poly::UnivariatePoly;
use jolt_program::preprocess::PublicIoMemory;
use jolt_r1cs::constraints::jolt::{
    SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE, SPARTAN_PRODUCT_UNISKIP_FIRST_ROUND_DEGREE,
};
use jolt_sumcheck::{
    ClearProof, ClearSumcheckProof, LabeledRoundPoly, RoundMessage, SumcheckProof,
};
use jolt_transcript::Transcript;
use jolt_verifier::stages::stage1::Stage1ClearOutput;
use jolt_verifier::stages::stage2::inputs::{
    product_uniskip_input_claim, InstructionClaimReductionOutputOpeningClaims,
    ProductRemainderOutputOpeningClaims, RamReadWriteOutputOpeningClaims,
    Stage2BatchOutputOpeningClaims, Stage2Claims, Stage2ProductUniSkipInputValues,
};
#[cfg(feature = "field-inline")]
use jolt_verifier::stages::stage2::inputs::{
    FieldInlineProductOutputOpeningClaims, FieldInlineStage2OutputOpeningClaims,
};
use jolt_verifier::stages::stage2::outputs::Stage2ClearOutput;
#[cfg(feature = "zk")]
use jolt_verifier::stages::stage2::outputs::Stage2PublicOutput;
use jolt_verifier::stages::stage2::stage2_output_claim_values;
use jolt_verifier::stages::stage2::{
    stage2_batch_input_claims, stage2_batch_opening_points, stage2_clear_output,
    stage2_expected_final_claim, stage2_expected_outputs, Stage2BatchExpectedOutputClaims,
    Stage2BatchInputClaimRequest, Stage2BatchInputClaims, Stage2BatchOpeningPoints,
    Stage2BatchPointRequest, Stage2ClearOutputRequest, Stage2ExpectedOutputRequest,
    Stage2ProductUniSkipClearRequest, Stage2ProductUniSkipOutputClaimData,
    Stage2RegularBatchClearRequest,
};
use jolt_verifier::CheckedInputs;
#[cfg(feature = "field-inline")]
use jolt_witness::protocols::jolt_vm::field_inline::FieldInlineNamespace;
use jolt_witness::{
    protocols::jolt_vm::{JoltVmNamespace, JoltVmStage2Rows, JoltVmStage2TraceRow},
    WitnessProvider,
};

const PRODUCT_UNISKIP_EXTENDED_EVAL_COUNT: usize = SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE - 1;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage2ProverConfig {
    pub log_t: usize,
}

impl Stage2ProverConfig {
    pub const fn new(log_t: usize) -> Self {
        Self { log_t }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage2BatchProverConfig {
    pub log_t: usize,
    pub log_k: usize,
    pub rw_config: JoltReadWriteConfig,
}

impl Stage2BatchProverConfig {
    pub const fn new(log_t: usize, log_k: usize, rw_config: JoltReadWriteConfig) -> Self {
        Self {
            log_t,
            log_k,
            rw_config,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage2ProductUniSkipInput<F: Field> {
    pub tau_low: Vec<F>,
    pub product: F,
    pub should_branch: F,
    pub should_jump: F,
    #[cfg(feature = "field-inline")]
    pub field_product: F,
    #[cfg(feature = "field-inline")]
    pub field_inv_product: F,
}

impl<F: Field> Stage2ProductUniSkipInput<F> {
    pub fn from_stage1(stage1: &Stage1ClearOutput<F>) -> Self {
        let mut tau_low = stage1.public.remainder_challenges[1..].to_vec();
        tau_low.reverse();
        Self {
            tau_low,
            product: stage1.outer.product,
            should_branch: stage1.outer.should_branch,
            should_jump: stage1.outer.should_jump,
            #[cfg(feature = "field-inline")]
            field_product: stage1.field_inline.field_product,
            #[cfg(feature = "field-inline")]
            field_inv_product: stage1.field_inline.field_inv_product,
        }
    }

    fn input_values(&self) -> Stage2ProductUniSkipInputValues<F> {
        Stage2ProductUniSkipInputValues {
            product: self.product,
            should_branch: self.should_branch,
            should_jump: self.should_jump,
            #[cfg(feature = "field-inline")]
            field_product: self.field_product,
            #[cfg(feature = "field-inline")]
            field_inv_product: self.field_inv_product,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Stage2ProverInput<'a, F: Field, W> {
    pub config: Stage2BatchProverConfig,
    pub checked: &'a CheckedInputs,
    pub stage1: &'a Stage1ClearOutput<F>,
    pub witness: &'a W,
}

impl<'a, F: Field, W> Stage2ProverInput<'a, F, W> {
    pub const fn new(
        config: Stage2BatchProverConfig,
        checked: &'a CheckedInputs,
        stage1: &'a Stage1ClearOutput<F>,
        witness: &'a W,
    ) -> Self {
        Self {
            config,
            checked,
            stage1,
            witness,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage2ProductUniSkipOutput<F: Field, C> {
    pub proof: SumcheckProof<F, C>,
    pub input_claim: F,
    pub output_claim: F,
    pub challenge: F,
    pub tau_high: F,
    pub tau_low: Vec<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage2ProofComponent<F: Field, Proof> {
    pub product_uniskip_proof: Proof,
    pub regular_batch_proof: Proof,
    pub claims: Stage2Claims<F>,
    pub verifier_output: Stage2ClearOutput<F>,
}

#[cfg(feature = "zk")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage2CommittedProofComponent<F, VC>
where
    F: Field,
    VC: VectorCommitment<Field = F>,
{
    pub product_uniskip_proof: SumcheckProof<F, VC::Output>,
    pub regular_batch_proof: SumcheckProof<F, VC::Output>,
    pub public: Stage2PublicOutput<F>,
    pub verifier_output: Stage2ClearOutput<F>,
    pub product_uniskip_output_claim_values: Vec<F>,
    pub batch_output_claim_values: Vec<F>,
    pub(crate) product_uniskip_committed_witness: CommittedSumcheckWitness<F>,
    pub(crate) batch_committed_witness: CommittedSumcheckWitness<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage2RegularBatchPrefixOutput<F: Field> {
    pub input_claims: Stage2BatchInputClaims<F>,
    pub ram_read_write_gamma: F,
    pub instruction_gamma: F,
    #[cfg(feature = "field-inline")]
    pub field_registers_claim_reduction_gamma: F,
    pub output_address_challenges: Vec<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage2RamTerminalOutputOpeningClaims<F: Field> {
    pub ram_raf_evaluation: F,
    pub ram_output_check: F,
}

struct Stage2VerifierOutputInput<'a, F: Field, C> {
    output_claims: Stage2BatchOutputOpeningClaims<F>,
    product_uniskip: &'a Stage2ProductUniSkipOutput<F, C>,
    batch_prefix: &'a Stage2RegularBatchPrefixOutput<F>,
    batch_challenges: &'a [F],
    batching_coefficients: &'a [F],
    batch_output_claim: F,
    opening_points: Stage2BatchOpeningPoints<F>,
    expected_outputs: Stage2BatchExpectedOutputClaims<F>,
}

fn stage2_verifier_output<F, C>(
    input: Stage2VerifierOutputInput<'_, F, C>,
) -> Result<Stage2ClearOutput<F>, ProverError>
where
    F: Field,
{
    stage2_clear_output(Stage2ClearOutputRequest {
        output_claims: input.output_claims,
        product_uniskip: Stage2ProductUniSkipClearRequest {
            tau_low: input.product_uniskip.tau_low.clone(),
            tau_high: input.product_uniskip.tau_high,
            input_claim: input.product_uniskip.input_claim,
            challenge: input.product_uniskip.challenge,
            output_claim: input.product_uniskip.output_claim,
        },
        batch: Stage2RegularBatchClearRequest {
            challenges: input.batch_challenges.to_vec(),
            batching_coefficients: input.batching_coefficients.to_vec(),
            output_claim: input.batch_output_claim,
            ram_read_write_gamma: input.batch_prefix.ram_read_write_gamma,
            instruction_gamma: input.batch_prefix.instruction_gamma,
            #[cfg(feature = "field-inline")]
            field_registers_claim_reduction_gamma: input
                .batch_prefix
                .field_registers_claim_reduction_gamma,
            output_address_challenges: input.batch_prefix.output_address_challenges.clone(),
            input_claims: input.batch_prefix.input_claims.clone(),
            opening_points: input.opening_points,
            expected_outputs: input.expected_outputs,
        },
    })
    .map_err(|error| invalid_sumcheck_output(error.to_string()))
}

fn stage2_opening_points<F: Field>(
    config: Stage2BatchProverConfig,
    challenges: &[F],
    product_tau_low: &[F],
) -> Result<Stage2BatchOpeningPoints<F>, ProverError> {
    stage2_batch_opening_points(Stage2BatchPointRequest {
        log_t: config.log_t,
        log_k: config.log_k,
        rw_config: config.rw_config,
        challenges,
        product_tau_low,
    })
    .map_err(|error| invalid_sumcheck_output(error.to_string()))
}

fn validate_stage2_request(
    checked: &CheckedInputs,
    config: Stage2BatchProverConfig,
    expected_zk: bool,
    mode_label: &'static str,
) -> Result<(), ProverError> {
    if checked.zk != expected_zk {
        let received = if checked.zk { "ZK" } else { "non-ZK" };
        return Err(ProverError::InvalidStageRequest {
            reason: format!("Stage 2 {mode_label} prover received {received} checked inputs"),
        });
    }
    if checked.trace_length != (1usize << config.log_t) {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 2 checked trace length {} does not match log_t {}",
                checked.trace_length, config.log_t
            ),
        });
    }
    if checked.ram_K != (1usize << config.log_k) {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 2 checked RAM K {} does not match log_k {}",
                checked.ram_K, config.log_k
            ),
        });
    }

    Ok(())
}

struct PreparedStage2RegularBatch<F: Field> {
    prefix: Stage2RegularBatchPrefixOutput<F>,
    ram_read_write: SumcheckRamReadWriteStateRequest<F>,
    ram_raf: SumcheckRamRafStateRequest<F>,
    ram_output_check: SumcheckRamOutputCheckStateRequest<F>,
    instances: Vec<SumcheckRegularBatchInstance<F>>,
}

struct Stage2RegularBatchPrepareInput<'a, F: Field, C> {
    config: Stage2BatchProverConfig,
    checked: &'a CheckedInputs,
    stage1: &'a Stage1ClearOutput<F>,
    rows: &'a [JoltVmStage2TraceRow],
    initial_ram_state: &'a [u64],
    final_ram_state: &'a [u64],
    product_uniskip: &'a Stage2ProductUniSkipOutput<F, C>,
}

#[cfg(not(feature = "field-inline"))]
fn prepare_stage2_regular_batch<F, T, C>(
    input: Stage2RegularBatchPrepareInput<'_, F, C>,
    transcript: &mut T,
) -> Result<PreparedStage2RegularBatch<F>, ProverError>
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    let prefix = derive_stage2_regular_batch_prefix(
        input.config,
        input.stage1,
        input.product_uniskip,
        transcript,
    )?;
    let backend_rows = ram_read_write_rows_from_trace(input.rows);
    let ram_requests = build_ram_state_requests(
        input.config,
        input.checked,
        backend_rows,
        input.initial_ram_state,
        input.final_ram_state,
        input.product_uniskip,
        &prefix,
    )?;
    let instances =
        build_regular_batch_instances(input.config, input.rows, input.product_uniskip, &prefix)?;

    Ok(PreparedStage2RegularBatch {
        prefix,
        ram_read_write: ram_requests.ram_read_write,
        ram_raf: ram_requests.ram_raf,
        ram_output_check: ram_requests.ram_output_check,
        instances,
    })
}

#[cfg(feature = "field-inline")]
fn prepare_stage2_regular_batch<F, T, C>(
    input: Stage2RegularBatchPrepareInput<'_, F, C>,
    field_factors: &Stage2FieldInlineMaterializedFactors<F>,
    transcript: &mut T,
) -> Result<PreparedStage2RegularBatch<F>, ProverError>
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    let prefix = derive_stage2_regular_batch_prefix(
        input.config,
        input.stage1,
        input.product_uniskip,
        transcript,
    )?;
    let backend_rows = ram_read_write_rows_from_trace(input.rows);
    let ram_requests = build_ram_state_requests(
        input.config,
        input.checked,
        backend_rows,
        input.initial_ram_state,
        input.final_ram_state,
        input.product_uniskip,
        &prefix,
    )?;
    let instances = build_regular_batch_instances(
        input.config,
        input.rows,
        input.product_uniskip,
        &prefix,
        field_factors,
    )?;

    Ok(PreparedStage2RegularBatch {
        prefix,
        ram_read_write: ram_requests.ram_read_write,
        ram_raf: ram_requests.ram_raf,
        ram_output_check: ram_requests.ram_output_check,
        instances,
    })
}

#[cfg(not(feature = "field-inline"))]
fn stage2_batch_output_claims<F: Field>(
    ram_read_write: RamReadWriteOutputOpeningClaims<F>,
    tail: Stage2TailOutputOpenings<F>,
    terminal: Stage2RamTerminalOutputOpeningClaims<F>,
) -> Stage2BatchOutputOpeningClaims<F> {
    Stage2BatchOutputOpeningClaims {
        ram_read_write,
        product_remainder: tail.product_remainder,
        instruction_claim_reduction: tail.instruction_claim_reduction,
        ram_raf_evaluation: terminal.ram_raf_evaluation,
        ram_output_check: terminal.ram_output_check,
    }
}

#[cfg(feature = "field-inline")]
fn stage2_batch_output_claims<F: Field>(
    ram_read_write: RamReadWriteOutputOpeningClaims<F>,
    tail: Stage2TailOutputOpenings<F>,
    field_inline: FieldInlineStage2OutputOpeningClaims<F>,
    terminal: Stage2RamTerminalOutputOpeningClaims<F>,
) -> Stage2BatchOutputOpeningClaims<F> {
    Stage2BatchOutputOpeningClaims {
        ram_read_write,
        product_remainder: tail.product_remainder,
        field_inline,
        instruction_claim_reduction: tail.instruction_claim_reduction,
        ram_raf_evaluation: terminal.ram_raf_evaluation,
        ram_output_check: terminal.ram_output_check,
    }
}

#[cfg(not(feature = "field-inline"))]
pub fn prove<F, W, B, T, C>(
    input: Stage2ProverInput<'_, F, W>,
    backend: &mut B,
    transcript: &mut T,
) -> Result<Stage2ProofComponent<F, SumcheckProof<F, C>>, ProverError>
where
    F: Field,
    W: JoltVmStage2Rows + WitnessProvider<F, JoltVmNamespace>,
    B: SumcheckBackend<F, JoltVmNamespace> + RamReadWriteSumcheckBackend<F>,
    T: Transcript<Challenge = F>,
{
    validate_stage2_request(input.checked, input.config, false, "clear")?;

    let stage2_rows = input.witness.stage2_rows()?;
    let initial_ram_state = input.witness.initial_ram_state_words()?;
    let final_ram_state = input.witness.final_ram_state_words()?;

    let product_input = Stage2ProductUniSkipInput::from_stage1(input.stage1);
    let product_uniskip = prove_stage2_product_uniskip_from_stage2_rows::<F, B, T, C>(
        Stage2ProverConfig::new(input.config.log_t),
        &product_input,
        &stage2_rows,
        backend,
        transcript,
    )?;

    let prepared = prepare_stage2_regular_batch(
        Stage2RegularBatchPrepareInput {
            config: input.config,
            checked: input.checked,
            stage1: input.stage1,
            rows: &stage2_rows,
            initial_ram_state: &initial_ram_state,
            final_ram_state: &final_ram_state,
            product_uniskip: &product_uniskip,
        },
        transcript,
    )?;
    let batch = prove_regular_batch_sumcheck::<F, T, C, B>(
        prepared.ram_read_write,
        prepared.ram_raf,
        prepared.ram_output_check,
        prepared.instances,
        backend,
        transcript,
    )?;
    let opening_points =
        stage2_opening_points(input.config, &batch.challenges, &product_uniskip.tau_low)?;

    let ram_read_write = batch.ram_read_write.clone();
    let tail_openings = evaluate_stage2_tail_openings_from_rows(
        input.config,
        &stage2_rows,
        &opening_points.product_opening,
        &opening_points.instruction_opening,
    )?;
    let terminal = Stage2RamTerminalOutputOpeningClaims {
        ram_raf_evaluation: batch.ram_raf_evaluation,
        ram_output_check: batch.ram_output_check,
    };

    let output_claims = stage2_batch_output_claims(ram_read_write, tail_openings, terminal);
    let (expected_outputs, expected_final_claim) = expected_regular_batch_outputs(
        input.config,
        input.checked,
        &product_uniskip,
        &prepared.prefix,
        &batch.batching_coefficients,
        &opening_points,
        &output_claims,
    )?;
    if batch.output_claim != expected_final_claim {
        return Err(stage2_regular_batch_output_mismatch(
            batch.output_claim,
            expected_final_claim,
            &expected_outputs,
        ));
    }

    let recorded = batch
        .proof
        .finish(&stage2_output_claim_values(&output_claims), transcript)?;

    let claims = Stage2Claims {
        product_uniskip_output_claim: product_uniskip.output_claim,
        batch_outputs: output_claims.clone(),
    };
    let verifier_output = stage2_verifier_output(Stage2VerifierOutputInput {
        output_claims,
        product_uniskip: &product_uniskip,
        batch_prefix: &prepared.prefix,
        batch_challenges: &batch.challenges,
        batching_coefficients: &batch.batching_coefficients,
        batch_output_claim: batch.output_claim,
        opening_points,
        expected_outputs,
    })?;

    Ok(Stage2ProofComponent {
        product_uniskip_proof: product_uniskip.proof,
        regular_batch_proof: recorded.proof,
        claims,
        verifier_output,
    })
}

#[cfg(feature = "field-inline")]
pub fn prove<F, W, B, T, C>(
    input: Stage2ProverInput<'_, F, W>,
    backend: &mut B,
    transcript: &mut T,
) -> Result<Stage2ProofComponent<F, SumcheckProof<F, C>>, ProverError>
where
    F: Field,
    W: JoltVmStage2Rows
        + WitnessProvider<F, JoltVmNamespace>
        + WitnessProvider<F, FieldInlineNamespace>,
    B: SumcheckBackend<F, JoltVmNamespace>
        + SumcheckBackend<F, FieldInlineNamespace>
        + RamReadWriteSumcheckBackend<F>,
    T: Transcript<Challenge = F>,
{
    validate_stage2_request(input.checked, input.config, false, "clear")?;

    let stage2_rows = input.witness.stage2_rows()?;
    let initial_ram_state = input.witness.initial_ram_state_words()?;
    let final_ram_state = input.witness.final_ram_state_words()?;
    let field_factors = stage2_field_inline_materialize_product_factors(
        input.config.log_t,
        input.witness,
        backend,
    )?;

    let product_input = Stage2ProductUniSkipInput::from_stage1(input.stage1);
    let product_uniskip = prove_stage2_product_uniskip_from_stage2_rows_field_inline::<F, T, C>(
        Stage2ProverConfig::new(input.config.log_t),
        &product_input,
        &stage2_rows,
        &field_factors,
        transcript,
    )?;

    let prepared = prepare_stage2_regular_batch(
        Stage2RegularBatchPrepareInput {
            config: input.config,
            checked: input.checked,
            stage1: input.stage1,
            rows: &stage2_rows,
            initial_ram_state: &initial_ram_state,
            final_ram_state: &final_ram_state,
            product_uniskip: &product_uniskip,
        },
        &field_factors,
        transcript,
    )?;
    let batch = prove_regular_batch_sumcheck::<F, T, C, B>(
        prepared.ram_read_write,
        prepared.ram_raf,
        prepared.ram_output_check,
        prepared.instances,
        backend,
        transcript,
    )?;
    let opening_points =
        stage2_opening_points(input.config, &batch.challenges, &product_uniskip.tau_low)?;

    let ram_read_write = batch.ram_read_write.clone();
    let tail_openings = evaluate_stage2_tail_openings_from_rows(
        input.config,
        &stage2_rows,
        &opening_points.product_opening,
        &opening_points.instruction_opening,
    )?;
    let field_inline = evaluate_stage2_field_inline_openings_from_factors(
        input.config,
        &field_factors,
        &opening_points.field_registers_claim_reduction_opening,
    )?;
    let terminal = Stage2RamTerminalOutputOpeningClaims {
        ram_raf_evaluation: batch.ram_raf_evaluation,
        ram_output_check: batch.ram_output_check,
    };

    let output_claims =
        stage2_batch_output_claims(ram_read_write, tail_openings, field_inline, terminal);
    let (expected_outputs, expected_final_claim) = expected_regular_batch_outputs(
        input.config,
        input.checked,
        &product_uniskip,
        &prepared.prefix,
        &batch.batching_coefficients,
        &opening_points,
        &output_claims,
    )?;
    if batch.output_claim != expected_final_claim {
        return Err(stage2_regular_batch_output_mismatch(
            batch.output_claim,
            expected_final_claim,
            &expected_outputs,
        ));
    }

    let recorded = batch
        .proof
        .finish(&stage2_output_claim_values(&output_claims), transcript)?;

    let claims = Stage2Claims {
        product_uniskip_output_claim: product_uniskip.output_claim,
        batch_outputs: output_claims.clone(),
    };
    let verifier_output = stage2_verifier_output(Stage2VerifierOutputInput {
        output_claims,
        product_uniskip: &product_uniskip,
        batch_prefix: &prepared.prefix,
        batch_challenges: &batch.challenges,
        batching_coefficients: &batch.batching_coefficients,
        batch_output_claim: batch.output_claim,
        opening_points,
        expected_outputs,
    })?;

    Ok(Stage2ProofComponent {
        product_uniskip_proof: product_uniskip.proof,
        regular_batch_proof: recorded.proof,
        claims,
        verifier_output,
    })
}

#[cfg(all(feature = "zk", not(feature = "field-inline")))]
pub fn prove_committed_proof_component<F, W, B, T, VC>(
    input: Stage2ProverInput<'_, F, W>,
    backend: &mut B,
    vc_setup: &VC::Setup,
    transcript: &mut T,
) -> Result<Stage2CommittedProofComponent<F, VC>, ProverError>
where
    F: Field,
    W: JoltVmStage2Rows + WitnessProvider<F, JoltVmNamespace>,
    B: SumcheckBackend<F, JoltVmNamespace> + RamReadWriteSumcheckBackend<F>,
    T: Transcript<Challenge = F>,
    VC: VectorCommitment<Field = F>,
{
    validate_stage2_request(input.checked, input.config, true, "committed")?;

    let stage2_rows = input.witness.stage2_rows()?;
    let initial_ram_state = input.witness.initial_ram_state_words()?;
    let final_ram_state = input.witness.final_ram_state_words()?;

    let product_input = Stage2ProductUniSkipInput::from_stage1(input.stage1);
    let product_uniskip = prove_stage2_product_uniskip_committed_from_stage2_rows::<F, B, T, VC>(
        Stage2ProverConfig::new(input.config.log_t),
        &product_input,
        &stage2_rows,
        backend,
        vc_setup,
        transcript,
    )?;

    let prepared = prepare_stage2_regular_batch(
        Stage2RegularBatchPrepareInput {
            config: input.config,
            checked: input.checked,
            stage1: input.stage1,
            rows: &stage2_rows,
            initial_ram_state: &initial_ram_state,
            final_ram_state: &final_ram_state,
            product_uniskip: &product_uniskip.output,
        },
        transcript,
    )?;
    let batch = prove_regular_batch_sumcheck_committed::<F, T, B, VC>(
        prepared.ram_read_write,
        prepared.ram_raf,
        prepared.ram_output_check,
        prepared.instances,
        backend,
        vc_setup,
        transcript,
    )?;
    let opening_points = stage2_opening_points(
        input.config,
        &batch.challenges,
        &product_uniskip.output.tau_low,
    )?;

    let ram_read_write = batch.ram_read_write.clone();
    let tail_openings = evaluate_stage2_tail_openings_from_rows(
        input.config,
        &stage2_rows,
        &opening_points.product_opening,
        &opening_points.instruction_opening,
    )?;
    let terminal = Stage2RamTerminalOutputOpeningClaims {
        ram_raf_evaluation: batch.ram_raf_evaluation,
        ram_output_check: batch.ram_output_check,
    };

    let output_claims = stage2_batch_output_claims(ram_read_write, tail_openings, terminal);
    let (expected_outputs, expected_final_claim) = expected_regular_batch_outputs(
        input.config,
        input.checked,
        &product_uniskip.output,
        &prepared.prefix,
        &batch.batching_coefficients,
        &opening_points,
        &output_claims,
    )?;
    if batch.output_claim != expected_final_claim {
        return Err(stage2_regular_batch_output_mismatch(
            batch.output_claim,
            expected_final_claim,
            &expected_outputs,
        ));
    }

    let batch_output_claim_values = stage2_output_claim_values(&output_claims);
    let verifier_output = stage2_verifier_output(Stage2VerifierOutputInput {
        output_claims,
        product_uniskip: &product_uniskip.output,
        batch_prefix: &prepared.prefix,
        batch_challenges: &batch.challenges,
        batching_coefficients: &batch.batching_coefficients,
        batch_output_claim: batch.output_claim,
        opening_points,
        expected_outputs,
    })?;
    let recorded = batch.proof.finish(&batch_output_claim_values, transcript)?;

    Ok(Stage2CommittedProofComponent {
        product_uniskip_proof: product_uniskip.output.proof,
        regular_batch_proof: recorded.proof,
        public: verifier_output.public.clone(),
        verifier_output,
        product_uniskip_output_claim_values: product_uniskip.output_claim_values,
        batch_output_claim_values,
        product_uniskip_committed_witness: product_uniskip.committed_witness,
        batch_committed_witness: recorded.committed_witness.ok_or_else(|| {
            invalid_sumcheck_output("Stage 2 committed batch witness material is missing")
        })?,
    })
}

#[cfg(all(feature = "zk", feature = "field-inline"))]
pub fn prove_committed_proof_component<F, W, B, T, VC>(
    input: Stage2ProverInput<'_, F, W>,
    backend: &mut B,
    vc_setup: &VC::Setup,
    transcript: &mut T,
) -> Result<Stage2CommittedProofComponent<F, VC>, ProverError>
where
    F: Field,
    W: JoltVmStage2Rows
        + WitnessProvider<F, JoltVmNamespace>
        + WitnessProvider<F, FieldInlineNamespace>,
    B: SumcheckBackend<F, JoltVmNamespace>
        + SumcheckBackend<F, FieldInlineNamespace>
        + RamReadWriteSumcheckBackend<F>,
    T: Transcript<Challenge = F>,
    VC: VectorCommitment<Field = F>,
{
    validate_stage2_request(input.checked, input.config, true, "committed")?;

    let stage2_rows = input.witness.stage2_rows()?;
    let initial_ram_state = input.witness.initial_ram_state_words()?;
    let final_ram_state = input.witness.final_ram_state_words()?;
    let field_factors = stage2_field_inline_materialize_product_factors(
        input.config.log_t,
        input.witness,
        backend,
    )?;

    let product_input = Stage2ProductUniSkipInput::from_stage1(input.stage1);
    let product_uniskip =
        prove_stage2_product_uniskip_committed_from_stage2_rows_field_inline::<F, T, VC>(
            Stage2ProverConfig::new(input.config.log_t),
            &product_input,
            &stage2_rows,
            &field_factors,
            vc_setup,
            transcript,
        )?;

    let prepared = prepare_stage2_regular_batch(
        Stage2RegularBatchPrepareInput {
            config: input.config,
            checked: input.checked,
            stage1: input.stage1,
            rows: &stage2_rows,
            initial_ram_state: &initial_ram_state,
            final_ram_state: &final_ram_state,
            product_uniskip: &product_uniskip.output,
        },
        &field_factors,
        transcript,
    )?;
    let batch = prove_regular_batch_sumcheck_committed::<F, T, B, VC>(
        prepared.ram_read_write,
        prepared.ram_raf,
        prepared.ram_output_check,
        prepared.instances,
        backend,
        vc_setup,
        transcript,
    )?;
    let opening_points = stage2_opening_points(
        input.config,
        &batch.challenges,
        &product_uniskip.output.tau_low,
    )?;

    let ram_read_write = batch.ram_read_write.clone();
    let tail_openings = evaluate_stage2_tail_openings_from_rows(
        input.config,
        &stage2_rows,
        &opening_points.product_opening,
        &opening_points.instruction_opening,
    )?;
    let field_inline = evaluate_stage2_field_inline_openings_from_factors(
        input.config,
        &field_factors,
        &opening_points.field_registers_claim_reduction_opening,
    )?;
    let terminal = Stage2RamTerminalOutputOpeningClaims {
        ram_raf_evaluation: batch.ram_raf_evaluation,
        ram_output_check: batch.ram_output_check,
    };

    let output_claims =
        stage2_batch_output_claims(ram_read_write, tail_openings, field_inline, terminal);
    let (expected_outputs, expected_final_claim) = expected_regular_batch_outputs(
        input.config,
        input.checked,
        &product_uniskip.output,
        &prepared.prefix,
        &batch.batching_coefficients,
        &opening_points,
        &output_claims,
    )?;
    if batch.output_claim != expected_final_claim {
        return Err(stage2_regular_batch_output_mismatch(
            batch.output_claim,
            expected_final_claim,
            &expected_outputs,
        ));
    }

    let batch_output_claim_values = stage2_output_claim_values(&output_claims);
    let verifier_output = stage2_verifier_output(Stage2VerifierOutputInput {
        output_claims,
        product_uniskip: &product_uniskip.output,
        batch_prefix: &prepared.prefix,
        batch_challenges: &batch.challenges,
        batching_coefficients: &batch.batching_coefficients,
        batch_output_claim: batch.output_claim,
        opening_points,
        expected_outputs,
    })?;
    let recorded = batch.proof.finish(&batch_output_claim_values, transcript)?;

    Ok(Stage2CommittedProofComponent {
        product_uniskip_proof: product_uniskip.output.proof,
        regular_batch_proof: recorded.proof,
        public: verifier_output.public.clone(),
        verifier_output,
        product_uniskip_output_claim_values: product_uniskip.output_claim_values,
        batch_output_claim_values,
        product_uniskip_committed_witness: product_uniskip.committed_witness,
        batch_committed_witness: recorded.committed_witness.ok_or_else(|| {
            invalid_sumcheck_output("Stage 2 committed batch witness material is missing")
        })?,
    })
}

#[cfg(not(feature = "field-inline"))]
fn prove_stage2_product_uniskip_from_stage2_rows<F, B, T, C>(
    config: Stage2ProverConfig,
    input: &Stage2ProductUniSkipInput<F>,
    stage2_rows: &[JoltVmStage2TraceRow],
    backend: &mut B,
    transcript: &mut T,
) -> Result<Stage2ProductUniSkipOutput<F, C>, ProverError>
where
    F: Field,
    B: SumcheckBackend<F, JoltVmNamespace>,
    T: Transcript<Challenge = F>,
{
    let base_evals = [input.product, input.should_branch, input.should_jump];
    let extended_evals = product_uniskip_extended_evals_from_stage2_rows(
        config,
        stage2_rows,
        backend,
        &input.tau_low,
    )?;
    prove_stage2_product_uniskip_with_extended_evals(
        input,
        &base_evals,
        &extended_evals,
        transcript,
    )
}

#[cfg(all(feature = "zk", not(feature = "field-inline")))]
fn prove_stage2_product_uniskip_committed_from_stage2_rows<F, B, T, VC>(
    config: Stage2ProverConfig,
    input: &Stage2ProductUniSkipInput<F>,
    stage2_rows: &[JoltVmStage2TraceRow],
    backend: &mut B,
    vc_setup: &VC::Setup,
    transcript: &mut T,
) -> Result<CommittedProductUniSkip<F, VC::Output>, ProverError>
where
    F: Field,
    B: SumcheckBackend<F, JoltVmNamespace>,
    T: Transcript<Challenge = F>,
    VC: VectorCommitment<Field = F>,
{
    let base_evals = [input.product, input.should_branch, input.should_jump];
    let extended_evals = product_uniskip_extended_evals_from_stage2_rows(
        config,
        stage2_rows,
        backend,
        &input.tau_low,
    )?;
    prove_stage2_product_uniskip_committed_with_extended_evals::<F, T, VC>(
        input,
        &base_evals,
        &extended_evals,
        vc_setup,
        transcript,
    )
}

#[cfg(feature = "field-inline")]
fn prove_stage2_product_uniskip_from_stage2_rows_field_inline<F, T, C>(
    config: Stage2ProverConfig,
    input: &Stage2ProductUniSkipInput<F>,
    stage2_rows: &[JoltVmStage2TraceRow],
    field_factors: &Stage2FieldInlineMaterializedFactors<F>,
    transcript: &mut T,
) -> Result<Stage2ProductUniSkipOutput<F, C>, ProverError>
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    let base_evals = [
        input.product,
        input.should_branch,
        input.should_jump,
        input.field_product,
        input.field_inv_product,
    ];
    let extended_evals = product_uniskip_extended_evals_field_inline_from_stage2_rows(
        config,
        stage2_rows,
        field_factors,
        &input.tau_low,
    )?;
    prove_stage2_product_uniskip_with_extended_evals(
        input,
        &base_evals,
        &extended_evals,
        transcript,
    )
}

#[cfg(all(feature = "zk", feature = "field-inline"))]
fn prove_stage2_product_uniskip_committed_from_stage2_rows_field_inline<F, T, VC>(
    config: Stage2ProverConfig,
    input: &Stage2ProductUniSkipInput<F>,
    stage2_rows: &[JoltVmStage2TraceRow],
    field_factors: &Stage2FieldInlineMaterializedFactors<F>,
    vc_setup: &VC::Setup,
    transcript: &mut T,
) -> Result<CommittedProductUniSkip<F, VC::Output>, ProverError>
where
    F: Field,
    T: Transcript<Challenge = F>,
    VC: VectorCommitment<Field = F>,
{
    let base_evals = [
        input.product,
        input.should_branch,
        input.should_jump,
        input.field_product,
        input.field_inv_product,
    ];
    let extended_evals = product_uniskip_extended_evals_field_inline_from_stage2_rows(
        config,
        stage2_rows,
        field_factors,
        &input.tau_low,
    )?;
    prove_stage2_product_uniskip_committed_with_extended_evals::<F, T, VC>(
        input,
        &base_evals,
        &extended_evals,
        vc_setup,
        transcript,
    )
}

fn prove_stage2_product_uniskip_with_extended_evals<F, T, C>(
    input: &Stage2ProductUniSkipInput<F>,
    base_evals: &[F; SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE],
    extended_evals: &[F],
    transcript: &mut T,
) -> Result<Stage2ProductUniSkipOutput<F, C>, ProverError>
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    let (tau_high, input_claim, polynomial) =
        stage2_product_uniskip_round(input, base_evals, extended_evals, transcript)?;
    LabeledRoundPoly::uniskip(&polynomial).append_to_transcript(transcript);
    let challenge = transcript.challenge();
    let output_claim = polynomial.evaluate(challenge);
    transcript.append_labeled(b"opening_claim", &output_claim);

    Ok(Stage2ProductUniSkipOutput {
        proof: SumcheckProof::Clear(ClearProof::Full(ClearSumcheckProof {
            round_polynomials: vec![polynomial],
        })),
        input_claim,
        output_claim,
        challenge,
        tau_high,
        tau_low: input.tau_low.clone(),
    })
}

#[cfg(feature = "zk")]
fn prove_stage2_product_uniskip_committed_with_extended_evals<F, T, VC>(
    input: &Stage2ProductUniSkipInput<F>,
    base_evals: &[F; SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE],
    extended_evals: &[F],
    vc_setup: &VC::Setup,
    transcript: &mut T,
) -> Result<CommittedProductUniSkip<F, VC::Output>, ProverError>
where
    F: Field,
    T: Transcript<Challenge = F>,
    VC: VectorCommitment<Field = F>,
{
    let (tau_high, input_claim, polynomial) =
        stage2_product_uniskip_round(input, base_evals, extended_evals, transcript)?;
    let mut builder = CommittedSumcheckBuilder::<F, VC>::new(vc_setup, 1)?;
    let challenge = builder.commit_round(&polynomial, transcript)?;
    let output_claim = polynomial.evaluate(challenge);
    let output_claim_values = vec![output_claim];
    let built = builder.finish(&output_claim_values, transcript)?;

    Ok(CommittedProductUniSkip {
        output: Stage2ProductUniSkipOutput {
            proof: built.proof,
            input_claim,
            output_claim,
            challenge,
            tau_high,
            tau_low: input.tau_low.clone(),
        },
        output_claim_values,
        committed_witness: built.witness,
    })
}

fn stage2_product_uniskip_round<F, T>(
    input: &Stage2ProductUniSkipInput<F>,
    base_evals: &[F; SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE],
    extended_evals: &[F],
    transcript: &mut T,
) -> Result<(F, F, UnivariatePoly<F>), ProverError>
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    let tau_high = transcript.challenge();
    let first_round = stage2_product_uniskip_first_round(&Stage2ProductUniskipFirstRoundRequest {
        domain_size: SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE,
        first_round_degree: SPARTAN_PRODUCT_UNISKIP_FIRST_ROUND_DEGREE,
        base_evals,
        extended_evals,
        tau_high,
    })
    .map_err(ProverError::from)?;

    let input_claim =
        product_uniskip_input_claim(input.input_values(), &first_round.lagrange_weights)?;
    if first_round.round_sum != input_claim {
        return Err(invalid_sumcheck_output(
            "Stage 2 product uni-skip first-round polynomial does not sum to the input claim",
        ));
    }

    Ok((tau_high, input_claim, first_round.polynomial))
}

pub fn derive_stage2_regular_batch_prefix<F, T, C>(
    config: Stage2BatchProverConfig,
    stage1: &Stage1ClearOutput<F>,
    product_uniskip: &Stage2ProductUniSkipOutput<F, C>,
    transcript: &mut T,
) -> Result<Stage2RegularBatchPrefixOutput<F>, ProverError>
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    let ram_read_write_gamma = transcript.challenge_scalar();
    let instruction_gamma = transcript.challenge_scalar();
    #[cfg(feature = "field-inline")]
    let field_registers_claim_reduction_gamma = transcript.challenge_scalar();
    let output_address_challenges = (0..config.log_k)
        .map(|_| transcript.challenge())
        .collect::<Vec<_>>();

    let input_claims = stage2_batch_input_claims(Stage2BatchInputClaimRequest {
        log_t: config.log_t,
        log_k: config.log_k,
        rw_config: config.rw_config,
        stage1,
        product_uniskip_output_claim: product_uniskip.output_claim,
        ram_read_write_gamma,
        instruction_gamma,
        #[cfg(feature = "field-inline")]
        field_registers_claim_reduction_gamma,
    })?;

    Ok(Stage2RegularBatchPrefixOutput {
        input_claims,
        ram_read_write_gamma,
        instruction_gamma,
        #[cfg(feature = "field-inline")]
        field_registers_claim_reduction_gamma,
        output_address_challenges,
    })
}

fn evaluate_stage2_tail_openings_from_rows<F>(
    config: Stage2BatchProverConfig,
    rows: &[JoltVmStage2TraceRow],
    product_opening_point: &[F],
    instruction_opening_point: &[F],
) -> Result<Stage2TailOutputOpenings<F>, ProverError>
where
    F: Field,
{
    let openings = stage2_product_instruction_openings_from_rows(
        config.log_t,
        rows,
        product_opening_point,
        instruction_opening_point,
    )?;

    Ok(Stage2TailOutputOpenings {
        product_remainder: ProductRemainderOutputOpeningClaims {
            left_instruction_input: openings.product_remainder.left_instruction_input,
            right_instruction_input: openings.product_remainder.right_instruction_input,
            jump_flag: openings.product_remainder.jump_flag,
            write_lookup_output_to_rd: openings.product_remainder.write_lookup_output_to_rd,
            lookup_output: openings.product_remainder.lookup_output,
            branch_flag: openings.product_remainder.branch_flag,
            next_is_noop: openings.product_remainder.next_is_noop,
            virtual_instruction: openings.product_remainder.virtual_instruction,
        },
        instruction_claim_reduction: InstructionClaimReductionOutputOpeningClaims {
            lookup_output: None,
            left_lookup_operand: openings.instruction_claim_reduction.left_lookup_operand,
            right_lookup_operand: openings.instruction_claim_reduction.right_lookup_operand,
            left_instruction_input: None,
            right_instruction_input: None,
        },
    })
}

#[cfg(feature = "field-inline")]
fn evaluate_stage2_field_inline_openings_from_factors<F>(
    config: Stage2BatchProverConfig,
    factors: &Stage2FieldInlineMaterializedFactors<F>,
    opening_point: &[F],
) -> Result<FieldInlineStage2OutputOpeningClaims<F>, ProverError>
where
    F: Field,
{
    let openings = stage2_field_inline_factor_openings(config.log_t, factors, opening_point)?;
    Ok(FieldInlineStage2OutputOpeningClaims {
        product: FieldInlineProductOutputOpeningClaims {
            field_rs1_value: openings.field_rs1_value,
            field_rs2_value: openings.field_rs2_value,
            field_rd_value: openings.field_rd_value,
        },
    })
}

struct RegularBatchProof<F: Field, Proof> {
    proof: Proof,
    challenges: Vec<F>,
    batching_coefficients: Vec<F>,
    output_claim: F,
    ram_read_write: RamReadWriteOutputOpeningClaims<F>,
    ram_raf_evaluation: F,
    ram_output_check: F,
}

#[derive(Clone)]
struct Stage2TailOutputOpenings<F: Field> {
    product_remainder: ProductRemainderOutputOpeningClaims<F>,
    instruction_claim_reduction: InstructionClaimReductionOutputOpeningClaims<F>,
}

#[cfg(feature = "zk")]
struct CommittedProductUniSkip<F: Field, C> {
    output: Stage2ProductUniSkipOutput<F, C>,
    output_claim_values: Vec<F>,
    committed_witness: CommittedSumcheckWitness<F>,
}

fn stage2_regular_batch_output_mismatch<F: Field>(
    batch_output_claim: F,
    expected_final_claim: F,
    expected_outputs: &Stage2BatchExpectedOutputClaims<F>,
) -> ProverError {
    #[cfg(not(feature = "field-inline"))]
    let reason = format!(
        "Stage 2 regular batch final claim did not match output openings: got {}, expected {}; components ram_read_write={}, product_remainder={}, instruction_claim_reduction={}, ram_raf_evaluation={}, ram_output_check={}",
        batch_output_claim,
        expected_final_claim,
        expected_outputs.ram_read_write,
        expected_outputs.product_remainder,
        expected_outputs.instruction_claim_reduction,
        expected_outputs.ram_raf_evaluation,
        expected_outputs.ram_output_check,
    );
    #[cfg(feature = "field-inline")]
    let reason = format!(
        "Stage 2 regular batch final claim did not match output openings: got {}, expected {}; components ram_read_write={}, product_remainder={}, instruction_claim_reduction={}, field_registers_claim_reduction={}, ram_raf_evaluation={}, ram_output_check={}",
        batch_output_claim,
        expected_final_claim,
        expected_outputs.ram_read_write,
        expected_outputs.product_remainder,
        expected_outputs.instruction_claim_reduction,
        expected_outputs.field_registers_claim_reduction,
        expected_outputs.ram_raf_evaluation,
        expected_outputs.ram_output_check,
    );
    invalid_sumcheck_output(reason)
}

fn prove_regular_batch_sumcheck<F, T, C, B>(
    ram_read_write: SumcheckRamReadWriteStateRequest<F>,
    ram_raf: SumcheckRamRafStateRequest<F>,
    ram_output_check: SumcheckRamOutputCheckStateRequest<F>,
    instances: Vec<SumcheckRegularBatchInstance<F>>,
    backend: &mut B,
    transcript: &mut T,
) -> Result<RegularBatchProof<F, ClearSumcheckRecorder<F, C>>, ProverError>
where
    F: Field,
    T: Transcript<Challenge = F>,
    B: SumcheckBackend<F, JoltVmNamespace> + RamReadWriteSumcheckBackend<F>,
{
    prove_regular_batch_sumcheck_with_recorder(
        ram_read_write,
        ram_raf,
        ram_output_check,
        instances,
        backend,
        transcript,
        ClearSumcheckRecorder::<F, C>::new(0),
    )
}

fn prove_regular_batch_sumcheck_with_recorder<F, T, B, S>(
    ram_read_write: SumcheckRamReadWriteStateRequest<F>,
    ram_raf: SumcheckRamRafStateRequest<F>,
    ram_output_check: SumcheckRamOutputCheckStateRequest<F>,
    instances: Vec<SumcheckRegularBatchInstance<F>>,
    backend: &mut B,
    transcript: &mut T,
    mut proof_recorder: S,
) -> Result<RegularBatchProof<F, S>, ProverError>
where
    F: Field,
    T: Transcript<Challenge = F>,
    B: SumcheckBackend<F, JoltVmNamespace> + RamReadWriteSumcheckBackend<F>,
    S: SumcheckRecorder<F>,
{
    let mut ram_state = backend.materialize_sumcheck_ram_read_write_state(&ram_read_write)?;
    let mut ram_raf_state = backend.materialize_sumcheck_ram_raf_state(&ram_raf)?;
    let mut ram_output_check_state =
        backend.materialize_sumcheck_ram_output_check_state(&ram_output_check)?;
    let mut state = SumcheckRegularBatchState::new("stage2.regular_batch.tail", instances);
    let ram_read_write_rounds = ram_read_write.log_t + ram_read_write.log_k;
    let ram_raf_rounds = ram_raf.log_t + ram_raf.log_k - ram_raf.phase1_num_rounds;
    let ram_output_check_rounds =
        ram_output_check.log_t + ram_output_check.log_k - ram_output_check.phase1_num_rounds;
    let max_num_rounds = std::iter::once(ram_read_write_rounds)
        .chain(
            state
                .instances
                .iter()
                .map(SumcheckRegularBatchInstance::num_rounds),
        )
        .chain([ram_raf_rounds, ram_output_check_rounds])
        .max()
        .ok_or_else(|| invalid_sumcheck_output("Stage 2 regular batch has no instances"))?;
    let ram_raf_offset = max_num_rounds - ram_raf_rounds;
    let ram_output_check_offset = max_num_rounds - ram_output_check_rounds;

    // Canonical Stage 2 regular-batch input-claim order: RAM read-write, then
    // each tail instance, then RAM RAF, then the RAM output-check (claim zero).
    let mut input_claim_values = Vec::with_capacity(state.instances.len() + 3);
    input_claim_values.push(ram_read_write.input_claim);
    input_claim_values.extend(state.instances.iter().map(|instance| instance.input_claim));
    input_claim_values.push(ram_raf.input_claim);
    input_claim_values.push(F::zero());
    proof_recorder.absorb_input_claims(&input_claim_values, transcript);

    let tail_start = 1;
    let terminal_start = tail_start + state.instances.len();
    let ram_raf_index = terminal_start;
    let ram_output_check_index = terminal_start + 1;
    let instance_count = state.instances.len() + 3;
    let batching_coefficients = (0..instance_count)
        .map(|_| transcript.challenge_scalar())
        .collect::<Vec<_>>();
    let mut individual_claims = std::iter::once(ram_read_write.input_claim)
        .chain(state.instances.iter().map(|instance| {
            instance
                .input_claim
                .mul_pow_2(max_num_rounds - instance.num_rounds())
        }))
        .chain([
            ram_raf.input_claim.mul_pow_2(ram_raf_offset),
            F::zero().mul_pow_2(ram_output_check_offset),
        ])
        .collect::<Vec<_>>();
    let mut running_claim = individual_claims
        .iter()
        .zip(&batching_coefficients)
        .map(|(claim, coefficient)| *claim * *coefficient)
        .sum::<F>();
    let two_inv = F::from_u64(2).inv_or_zero();

    let mut challenges = Vec::with_capacity(max_num_rounds);
    for round in 0..max_num_rounds {
        let ram_poly =
            backend.evaluate_sumcheck_ram_read_write_round(&ram_state, individual_claims[0])?;
        let tail_messages = backend
            .evaluate_sumcheck_regular_batch_round(
                &mut state,
                round,
                max_num_rounds,
                &individual_claims[tail_start..terminal_start],
            )?
            .into_iter()
            .collect::<Vec<_>>();
        if tail_messages.len() != state.instances.len() {
            return Err(invalid_sumcheck_output(format!(
                "Stage 2 regular batch round {round} returned {} instance messages, expected {}",
                tail_messages.len() + 1,
                state.instances.len() + 1
            )));
        }
        let mut univariate_polys = Vec::with_capacity(instance_count);
        univariate_polys.push(ram_poly);
        for (expected_index, round_message) in tail_messages.into_iter().enumerate() {
            if round_message.instance_index != expected_index {
                return Err(invalid_sumcheck_output(format!(
                    "Stage 2 regular batch round {round} returned instance {}, expected {expected_index}",
                    round_message.instance_index
                )));
            }
            univariate_polys.push(round_message.polynomial);
        }
        if round < ram_raf_offset {
            univariate_polys.push(UnivariatePoly::new(vec![
                individual_claims[ram_raf_index] * two_inv,
            ]));
        } else {
            univariate_polys.push(backend.evaluate_sumcheck_ram_raf_round(
                &ram_raf_state,
                individual_claims[ram_raf_index],
            )?);
        }
        if round < ram_output_check_offset {
            univariate_polys.push(UnivariatePoly::new(vec![
                individual_claims[ram_output_check_index] * two_inv,
            ]));
        } else {
            univariate_polys.push(backend.evaluate_sumcheck_ram_output_check_round(
                &ram_output_check_state,
                individual_claims[ram_output_check_index],
            )?);
        }

        let mut batched_poly = UnivariatePoly::zero();
        for (poly, coefficient) in univariate_polys.iter().zip(&batching_coefficients) {
            batched_poly += &(poly * *coefficient);
        }
        let round_sum = batched_poly.evaluate(F::zero()) + batched_poly.evaluate(F::one());
        if round_sum != running_claim {
            return Err(invalid_sumcheck_output(format!(
                "Stage 2 regular batch round {round} sumcheck invariant failed"
            )));
        }
        let batched_poly = trim_round_polynomial(batched_poly);
        let challenge = proof_recorder.absorb_round(&batched_poly, transcript)?;
        running_claim = batched_poly.evaluate(challenge);
        challenges.push(challenge);

        for (claim, poly) in individual_claims.iter_mut().zip(univariate_polys) {
            *claim = poly.evaluate(challenge);
        }
        backend.bind_sumcheck_ram_read_write_state(&mut ram_state, challenge)?;
        backend.bind_sumcheck_regular_batch_state(&mut state, round, max_num_rounds, challenge)?;
        if round >= ram_raf_offset {
            backend.bind_sumcheck_ram_raf_state(&mut ram_raf_state, challenge)?;
        }
        if round >= ram_output_check_offset {
            backend.bind_sumcheck_ram_output_check_state(&mut ram_output_check_state, challenge)?;
        }
    }
    let [val, ra, inc] = backend.output_sumcheck_ram_read_write_state(&ram_state)?;
    let ram_raf_evaluation = backend.output_sumcheck_ram_raf_state(&ram_raf_state)?;
    let ram_output_check =
        backend.output_sumcheck_ram_output_check_state(&ram_output_check_state)?;

    // The output-claim values span batch results plus caller-assembled
    // components (product remainder, instruction claim reduction, ...), so the
    // recorder is finished by the caller via `RegularBatchProof::proof.finish`.
    Ok(RegularBatchProof {
        proof: proof_recorder,
        challenges,
        batching_coefficients,
        output_claim: running_claim,
        ram_read_write: RamReadWriteOutputOpeningClaims { val, ra, inc },
        ram_raf_evaluation,
        ram_output_check,
    })
}

#[cfg(feature = "zk")]
fn prove_regular_batch_sumcheck_committed<'a, F, T, B, VC>(
    ram_read_write: SumcheckRamReadWriteStateRequest<F>,
    ram_raf: SumcheckRamRafStateRequest<F>,
    ram_output_check: SumcheckRamOutputCheckStateRequest<F>,
    instances: Vec<SumcheckRegularBatchInstance<F>>,
    backend: &mut B,
    vc_setup: &'a VC::Setup,
    transcript: &mut T,
) -> Result<RegularBatchProof<F, CommittedSumcheckRecorder<'a, F, VC>>, ProverError>
where
    F: Field,
    T: Transcript<Challenge = F>,
    B: SumcheckBackend<F, JoltVmNamespace> + RamReadWriteSumcheckBackend<F>,
    VC: VectorCommitment<Field = F>,
{
    prove_regular_batch_sumcheck_with_recorder(
        ram_read_write,
        ram_raf,
        ram_output_check,
        instances,
        backend,
        transcript,
        CommittedSumcheckRecorder::<F, VC>::new(vc_setup)?,
    )
}

fn trim_round_polynomial<F: Field>(poly: UnivariatePoly<F>) -> UnivariatePoly<F> {
    let mut coefficients = poly.into_coefficients();
    while coefficients.len() > 2 && coefficients.last().is_some_and(|value| *value == F::zero()) {
        let _ = coefficients.pop();
    }
    UnivariatePoly::new(coefficients)
}

#[cfg(not(feature = "field-inline"))]
fn build_regular_batch_instances<F, C>(
    config: Stage2BatchProverConfig,
    rows: &[JoltVmStage2TraceRow],
    product_uniskip: &Stage2ProductUniSkipOutput<F, C>,
    prefix: &Stage2RegularBatchPrefixOutput<F>,
) -> Result<Vec<SumcheckRegularBatchInstance<F>>, ProverError>
where
    F: Field,
{
    let request = Stage2RegularBatchInstanceRequest {
        log_t: config.log_t,
        rows,
        tau_low: &product_uniskip.tau_low,
        tau_high: product_uniskip.tau_high,
        product_challenge: product_uniskip.challenge,
        product_output_claim: product_uniskip.output_claim,
        instruction_claim_reduction_input_claim: prefix.input_claims.instruction_claim_reduction,
        instruction_gamma: prefix.instruction_gamma,
    };
    stage2_regular_batch_instances(&request).map_err(ProverError::from)
}

#[cfg(feature = "field-inline")]
fn build_regular_batch_instances<F, C>(
    config: Stage2BatchProverConfig,
    rows: &[JoltVmStage2TraceRow],
    product_uniskip: &Stage2ProductUniSkipOutput<F, C>,
    prefix: &Stage2RegularBatchPrefixOutput<F>,
    field_factors: &Stage2FieldInlineMaterializedFactors<F>,
) -> Result<Vec<SumcheckRegularBatchInstance<F>>, ProverError>
where
    F: Field,
{
    let base = Stage2RegularBatchInstanceRequest {
        log_t: config.log_t,
        rows,
        tau_low: &product_uniskip.tau_low,
        tau_high: product_uniskip.tau_high,
        product_challenge: product_uniskip.challenge,
        product_output_claim: product_uniskip.output_claim,
        instruction_claim_reduction_input_claim: prefix.input_claims.instruction_claim_reduction,
        instruction_gamma: prefix.instruction_gamma,
    };
    let request = Stage2FieldInlineRegularBatchInstanceRequest {
        base,
        field_factors,
        field_registers_claim_reduction_input_claim: prefix
            .input_claims
            .field_registers_claim_reduction,
        field_registers_claim_reduction_gamma: prefix.field_registers_claim_reduction_gamma,
    };
    stage2_field_inline_regular_batch_instances(&request).map_err(ProverError::from)
}

fn build_ram_state_requests<F: Field>(
    config: Stage2BatchProverConfig,
    checked: &CheckedInputs,
    rows: Vec<SumcheckRamReadWriteRow>,
    initial_ram_state: &[u64],
    final_ram_state: &[u64],
    product_uniskip: &Stage2ProductUniSkipOutput<F, impl Sized>,
    prefix: &Stage2RegularBatchPrefixOutput<F>,
) -> Result<Stage2RamStateRequests<F>, ProverError> {
    let public_memory = PublicIoMemory::new(&checked.public_io).map_err(|error| {
        ProverError::InvalidStageRequest {
            reason: format!("invalid public IO memory for Stage 2 output check: {error}"),
        }
    })?;
    stage2_ram_state_requests(&Stage2RamStateRequestsRequest {
        log_t: config.log_t,
        log_k: config.log_k,
        phase1_num_rounds: config.rw_config.ram_rw_phase1_num_rounds as usize,
        phase2_num_rounds: config.rw_config.ram_rw_phase2_num_rounds as usize,
        rows: &rows,
        initial_ram_state,
        final_ram_state,
        tau_low: &product_uniskip.tau_low,
        ram_read_write_gamma: prefix.ram_read_write_gamma,
        ram_read_write_input_claim: prefix.input_claims.ram_read_write,
        ram_raf_input_claim: prefix.input_claims.ram_raf_evaluation,
        start_address: checked.public_io.memory_layout.get_lowest_address(),
        public_memory: &public_memory,
        output_address_challenges: &prefix.output_address_challenges,
    })
    .map_err(ProverError::from)
}

fn expected_regular_batch_outputs<F: Field>(
    config: Stage2BatchProverConfig,
    checked: &jolt_verifier::CheckedInputs,
    product_uniskip: &Stage2ProductUniSkipOutput<F, impl Sized>,
    prefix: &Stage2RegularBatchPrefixOutput<F>,
    batching_coefficients: &[F],
    opening_points: &Stage2BatchOpeningPoints<F>,
    claims: &Stage2BatchOutputOpeningClaims<F>,
) -> Result<(Stage2BatchExpectedOutputClaims<F>, F), ProverError> {
    let expected_outputs = stage2_expected_outputs(Stage2ExpectedOutputRequest {
        log_k: config.log_k,
        checked,
        product_uniskip: Stage2ProductUniSkipOutputClaimData {
            tau_low: &product_uniskip.tau_low,
            tau_high: product_uniskip.tau_high,
            challenge: product_uniskip.challenge,
        },
        ram_read_write_gamma: prefix.ram_read_write_gamma,
        instruction_gamma: prefix.instruction_gamma,
        #[cfg(feature = "field-inline")]
        field_registers_claim_reduction_gamma: prefix.field_registers_claim_reduction_gamma,
        output_address_challenges: &prefix.output_address_challenges,
        opening_points,
        claims,
    })
    .map_err(|error| invalid_sumcheck_output(error.to_string()))?;
    let final_claim = stage2_expected_final_claim(batching_coefficients, &expected_outputs)
        .map_err(|error| invalid_sumcheck_output(error.to_string()))?;

    Ok((expected_outputs, final_claim))
}

#[cfg(not(feature = "field-inline"))]
fn product_uniskip_extended_evals_from_stage2_rows<F, B>(
    config: Stage2ProverConfig,
    stage2_rows: &[JoltVmStage2TraceRow],
    backend: &mut B,
    tau_low: &[F],
) -> Result<Vec<F>, ProverError>
where
    F: Field,
    B: SumcheckBackend<F, JoltVmNamespace>,
{
    let product_rows = stage2_product_uniskip_rows_from_stage2_trace(config.log_t, stage2_rows)?;
    let request =
        stage2_product_uniskip_extended_eval_request(config.log_t, &product_rows, tau_low)?;
    let outputs = backend.evaluate_sumcheck_product_uniskip_rows(&request)?;
    stage2_product_uniskip_extended_eval_outputs(outputs, PRODUCT_UNISKIP_EXTENDED_EVAL_COUNT)
        .map_err(ProverError::from)
}

#[cfg(feature = "field-inline")]
fn product_uniskip_extended_evals_field_inline_from_stage2_rows<F>(
    config: Stage2ProverConfig,
    stage2_rows: &[JoltVmStage2TraceRow],
    field_factors: &Stage2FieldInlineMaterializedFactors<F>,
    tau_low: &[F],
) -> Result<Vec<F>, ProverError>
where
    F: Field,
{
    let request = Stage2FieldInlineProductUniskipEvalRequest {
        log_t: config.log_t,
        stage2_rows,
        field_factors,
        tau_low,
        domain_size: SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE,
        extended_eval_count: PRODUCT_UNISKIP_EXTENDED_EVAL_COUNT,
    };
    stage2_field_inline_product_uniskip_extended_evals(&request).map_err(ProverError::from)
}
