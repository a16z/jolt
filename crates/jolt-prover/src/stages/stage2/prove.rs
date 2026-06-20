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
use jolt_backends::{
    stage2_product_uniskip_extended_eval_outputs, stage2_product_uniskip_extended_eval_request,
    stage2_product_uniskip_rows_from_stage2_trace, stage2_regular_batch_instances,
};
use jolt_claims::protocols::jolt::{
    formulas::{
        dimensions::{ReadWriteDimensions, TraceDimensions},
        ram::RamRafEvaluationDimensions,
        spartan::SpartanProductDimensions,
    },
    JoltReadWriteConfig,
};
#[cfg(feature = "zk")]
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_poly::{Point, UnivariatePoly};
use jolt_program::preprocess::PublicIoMemory;
use jolt_r1cs::constraints::jolt::{
    SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE, SPARTAN_PRODUCT_UNISKIP_FIRST_ROUND_DEGREE,
};
use jolt_sumcheck::{
    ClearProof, ClearSumcheckProof, LabeledRoundPoly, RoundMessage, SumcheckProof,
};
use jolt_transcript::Transcript;
use jolt_verifier::stages::relations::{OpeningClaim, SumcheckInstance};
use jolt_verifier::stages::stage1::Stage1ClearOutput;
use jolt_verifier::stages::stage2::inputs::{
    product_uniskip_input_claim, InstructionClaimReductionOutputClaims,
    ProductRemainderOutputClaims, RamReadWriteOutputClaims, Stage2BatchOutputClaims,
    Stage2OutputClaims, Stage2ProductUniSkipInputValues,
};
use jolt_verifier::stages::stage2::outputs::{Stage2ClearOutput, Stage2PublicOutput};
use jolt_verifier::stages::stage2::{
    stage2_batch_output_claims_with_points, stage2_expected_final_claim, InstructionClaimReduction,
    InstructionClaimReductionInputClaims, ProductRemainder, ProductRemainderInputClaims,
    RamOutputCheck, RamOutputCheckInputClaims, RamOutputCheckOutputClaims, RamRafEvaluation,
    RamRafEvaluationInputClaims, RamRafEvaluationOutputClaims, RamReadWriteChecking,
    RamReadWriteInputClaims, VerifiedProductUniSkip,
};
use jolt_verifier::{CheckedInputs, VerifierError};
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
        }
    }

    fn input_values(&self) -> Stage2ProductUniSkipInputValues<F> {
        Stage2ProductUniSkipInputValues {
            product: self.product,
            should_branch: self.should_branch,
            should_jump: self.should_jump,
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
    pub claims: Stage2OutputClaims<F>,
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

/// The five stage 2 batch sumcheck input claims (claimed sums), one per batched
/// relation. Computed via the relation objects' `input_claim` so the prover and
/// verifier derive them identically. Prover-local mirror of the deleted verifier
/// struct (cf. stage 3's local `Stage3InputClaims`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage2BatchInputClaims<F: Field> {
    pub ram_read_write: F,
    pub product_remainder: F,
    pub instruction_claim_reduction: F,
    pub ram_raf_evaluation: F,
    pub ram_output_check: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage2RegularBatchPrefixOutput<F: Field> {
    pub input_claims: Stage2BatchInputClaims<F>,
    pub ram_read_write_gamma: F,
    pub instruction_gamma: F,
    pub output_address_challenges: Vec<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage2RamTerminalOutputOpeningClaims<F: Field> {
    pub ram_raf_evaluation: F,
    pub ram_output_check: F,
}

/// The five stage 2 batch relation objects bundled with their wired input claims.
/// The prover builds them inline (mirroring the verifier's `verify_regular_batch`
/// clear arm) and reuses them for both the input-claim derivation (before the
/// batched sumcheck) and the opening-point/expected-output derivation (after).
struct Stage2BatchRelations<F: Field> {
    ram_read_write: RamReadWriteChecking<F>,
    product_remainder: ProductRemainder<F>,
    instruction_reduction: InstructionClaimReduction<F>,
    ram_raf: RamRafEvaluation<F>,
    ram_output: RamOutputCheck<F>,
    ram_read_write_inputs: RamReadWriteInputClaims<OpeningClaim<F>>,
    product_remainder_inputs: ProductRemainderInputClaims<OpeningClaim<F>>,
    instruction_reduction_inputs: InstructionClaimReductionInputClaims<OpeningClaim<F>>,
    ram_raf_inputs: RamRafEvaluationInputClaims<OpeningClaim<F>>,
    ram_output_inputs: RamOutputCheckInputClaims<OpeningClaim<F>>,
}

/// The per-instance sumcheck points derived from the flat batch `challenges`,
/// before each relation maps them to its produced opening points. Recreates the
/// split the deleted `stage2_batch_opening_points` performed.
struct Stage2BatchSumcheckPoints<F: Field> {
    ram_read_write: Vec<F>,
    product: Vec<F>,
    instruction: Vec<F>,
    terminal: Vec<F>,
}

fn to_prover_error(error: VerifierError) -> ProverError {
    invalid_sumcheck_output(error.to_string())
}

impl<F: Field> Stage2BatchRelations<F> {
    fn input_claims(&self) -> Result<Stage2BatchInputClaims<F>, ProverError> {
        Ok(Stage2BatchInputClaims {
            ram_read_write: self
                .ram_read_write
                .input_claim(&self.ram_read_write_inputs)
                .map_err(to_prover_error)?,
            product_remainder: self
                .product_remainder
                .input_claim(&self.product_remainder_inputs)
                .map_err(to_prover_error)?,
            instruction_claim_reduction: self
                .instruction_reduction
                .input_claim(&self.instruction_reduction_inputs)
                .map_err(to_prover_error)?,
            ram_raf_evaluation: self
                .ram_raf
                .input_claim(&self.ram_raf_inputs)
                .map_err(to_prover_error)?,
            ram_output_check: self
                .ram_output
                .input_claim(&self.ram_output_inputs)
                .map_err(to_prover_error)?,
        })
    }

    /// Map each instance's sumcheck point to its produced opening points, in the
    /// same `Stage2BatchOutputClaims<Vec<F>>` aggregate the verifier feeds to
    /// `stage2_batch_output_claims_with_points`.
    fn derive_opening_points(
        &self,
        points: &Stage2BatchSumcheckPoints<F>,
    ) -> Result<Stage2BatchOutputClaims<Vec<F>>, ProverError> {
        Ok(Stage2BatchOutputClaims {
            ram_read_write: self
                .ram_read_write
                .derive_opening_points(&points.ram_read_write, &self.ram_read_write_inputs)
                .map_err(to_prover_error)?,
            product_remainder: self
                .product_remainder
                .derive_opening_points(&points.product, &self.product_remainder_inputs)
                .map_err(to_prover_error)?,
            instruction_claim_reduction: self
                .instruction_reduction
                .derive_opening_points(&points.instruction, &self.instruction_reduction_inputs)
                .map_err(to_prover_error)?,
            ram_raf_evaluation: self
                .ram_raf
                .derive_opening_points(&points.terminal, &self.ram_raf_inputs)
                .map_err(to_prover_error)?,
            ram_output_check: self
                .ram_output
                .derive_opening_points(&points.terminal, &self.ram_output_inputs)
                .map_err(to_prover_error)?,
        })
    }
}

fn read_write_dimensions(config: Stage2BatchProverConfig) -> ReadWriteDimensions {
    config.rw_config.ram_dimensions(config.log_t, config.log_k)
}

/// Recreate the deleted `stage2_batch_opening_points` per-instance sumcheck-point
/// split from the flat batch `challenges`: RAM read-write reads every challenge,
/// the product and instruction instances share the last `log_t` challenges, and
/// the RAM RAF / output-check instances share the terminal point
/// `challenges[phase1_num_rounds .. phase1_num_rounds + (log_t + log_k - phase1_num_rounds)]`.
fn stage2_batch_sumcheck_points<F: Field>(
    config: Stage2BatchProverConfig,
    challenges: &[F],
) -> Result<Stage2BatchSumcheckPoints<F>, ProverError> {
    let dimensions = read_write_dimensions(config);
    let read_write_rounds = config.log_t + config.log_k;
    if challenges.len() != read_write_rounds {
        return Err(invalid_sumcheck_output(format!(
            "Stage 2 regular batch returned {} challenges, expected {read_write_rounds}",
            challenges.len()
        )));
    }

    let product_offset = read_write_rounds - config.log_t;
    let product = challenges[product_offset..].to_vec();
    let instruction = product.clone();

    let phase1_num_rounds = dimensions.phase1_num_rounds();
    let terminal_rounds = read_write_rounds - phase1_num_rounds;
    let terminal_end = phase1_num_rounds + terminal_rounds;
    let terminal = challenges
        .get(phase1_num_rounds..terminal_end)
        .ok_or_else(|| {
            invalid_sumcheck_output(format!(
                "Stage 2 terminal point range {phase1_num_rounds}..{terminal_end} exceeds {} challenges",
                challenges.len()
            ))
        })?
        .to_vec();

    Ok(Stage2BatchSumcheckPoints {
        ram_read_write: challenges.to_vec(),
        product,
        instruction,
        terminal,
    })
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
        input.checked,
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

fn stage2_batch_output_claims<F: Field>(
    ram_read_write: RamReadWriteOutputClaims<F>,
    tail: Stage2TailOutputOpenings<F>,
    terminal: Stage2RamTerminalOutputOpeningClaims<F>,
) -> Stage2BatchOutputClaims<F> {
    Stage2BatchOutputClaims {
        ram_read_write,
        product_remainder: tail.product_remainder,
        instruction_claim_reduction: tail.instruction_claim_reduction,
        ram_raf_evaluation: RamRafEvaluationOutputClaims {
            ram_ra: terminal.ram_raf_evaluation,
        },
        ram_output_check: RamOutputCheckOutputClaims {
            val_final: terminal.ram_output_check,
        },
    }
}

struct Stage2BatchAssembly<F: Field> {
    claims: Stage2OutputClaims<F>,
    verifier_output: Stage2ClearOutput<F>,
    batch_output_claim_values: Vec<F>,
}

struct Stage2BatchAssemblyInput<'a, F: Field, C, P> {
    config: Stage2BatchProverConfig,
    checked: &'a CheckedInputs,
    stage1: &'a Stage1ClearOutput<F>,
    stage2_rows: &'a [JoltVmStage2TraceRow],
    product_uniskip: &'a Stage2ProductUniSkipOutput<F, C>,
    prefix: &'a Stage2RegularBatchPrefixOutput<F>,
    batch: &'a RegularBatchProof<F, P>,
}

/// Mirror the verifier's `verify_regular_batch` clear arm: split the batch
/// sumcheck point, build the five relations inline, derive each relation's
/// opening claims, check the combined final claim against the prover's batch
/// `output_claim`, and assemble the `Stage2ClearOutput` for later stages.
///
/// Shared by the clear `prove` and the committed `prove_committed_proof_component`
/// paths; pure with respect to the transcript (no Fiat-Shamir appends).
fn assemble_stage2_batch_output<F, C, P>(
    input: Stage2BatchAssemblyInput<'_, F, C, P>,
) -> Result<Stage2BatchAssembly<F>, ProverError>
where
    F: Field,
{
    let sumcheck_points = stage2_batch_sumcheck_points(input.config, &input.batch.challenges)?;
    let relations = stage2_batch_relations(
        input.config,
        input.checked,
        input.stage1,
        input.product_uniskip,
        input.prefix.ram_read_write_gamma,
        input.prefix.instruction_gamma,
        &input.prefix.output_address_challenges,
    )?;
    let opening_points = relations.derive_opening_points(&sumcheck_points)?;

    let tail_openings = evaluate_stage2_tail_openings_from_rows(
        input.config,
        input.stage2_rows,
        &opening_points.product_remainder.left_instruction_input,
        &opening_points
            .instruction_claim_reduction
            .left_lookup_operand,
    )?;
    let terminal = Stage2RamTerminalOutputOpeningClaims {
        ram_raf_evaluation: input.batch.ram_raf_evaluation,
        ram_output_check: input.batch.ram_output_check,
    };
    let wire_claims =
        stage2_batch_output_claims(input.batch.ram_read_write.clone(), tail_openings, terminal);
    let batch_output_claim_values = wire_claims.opening_values();

    let output_claims = stage2_batch_output_claims_with_points(&wire_claims, &opening_points);

    let expected_final_claim = stage2_expected_final_claim(
        &input.batch.batching_coefficients,
        relations
            .ram_read_write
            .expected_output(
                &relations.ram_read_write_inputs,
                &output_claims.ram_read_write,
            )
            .map_err(to_prover_error)?,
        relations
            .product_remainder
            .expected_output(
                &relations.product_remainder_inputs,
                &output_claims.product_remainder,
            )
            .map_err(to_prover_error)?,
        relations
            .instruction_reduction
            .expected_output(
                &relations.instruction_reduction_inputs,
                &output_claims.instruction_claim_reduction,
            )
            .map_err(to_prover_error)?,
        relations
            .ram_raf
            .expected_output(&relations.ram_raf_inputs, &output_claims.ram_raf_evaluation)
            .map_err(to_prover_error)?,
        relations
            .ram_output
            .expected_output(
                &relations.ram_output_inputs,
                &output_claims.ram_output_check,
            )
            .map_err(to_prover_error)?,
    )
    .map_err(to_prover_error)?;
    if input.batch.output_claim != expected_final_claim {
        return Err(invalid_sumcheck_output(format!(
            "Stage 2 regular batch final claim did not match output openings: got {}, expected {}",
            input.batch.output_claim, expected_final_claim,
        )));
    }

    let claims = Stage2OutputClaims {
        product_uniskip_output_claim: input.product_uniskip.output_claim,
        batch_outputs: wire_claims,
    };
    let public = Stage2PublicOutput {
        challenges: input.batch.challenges.clone(),
        batching_coefficients: input.batch.batching_coefficients.clone(),
        product_uniskip_challenge: input.product_uniskip.challenge,
        product_tau_low: input.product_uniskip.tau_low.clone(),
        product_tau_high: input.product_uniskip.tau_high,
        ram_read_write_gamma: input.prefix.ram_read_write_gamma,
        instruction_gamma: input.prefix.instruction_gamma,
        output_address_challenges: input.prefix.output_address_challenges.clone(),
    };
    let verifier_output = Stage2ClearOutput {
        public,
        output_claims,
        product_uniskip: VerifiedProductUniSkip {
            tau_low: input.product_uniskip.tau_low.clone(),
            tau_high: input.product_uniskip.tau_high,
            sumcheck_point: Point::high_to_low(vec![input.product_uniskip.challenge]),
        },
    };

    Ok(Stage2BatchAssembly {
        claims,
        verifier_output,
        batch_output_claim_values,
    })
}

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

    let assembly = assemble_stage2_batch_output(Stage2BatchAssemblyInput {
        config: input.config,
        checked: input.checked,
        stage1: input.stage1,
        stage2_rows: &stage2_rows,
        product_uniskip: &product_uniskip,
        prefix: &prepared.prefix,
        batch: &batch,
    })?;

    let recorded = batch
        .proof
        .finish(&assembly.batch_output_claim_values, transcript)?;

    Ok(Stage2ProofComponent {
        product_uniskip_proof: product_uniskip.proof,
        regular_batch_proof: recorded.proof,
        claims: assembly.claims,
        verifier_output: assembly.verifier_output,
    })
}

#[cfg(feature = "zk")]
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

    let assembly = assemble_stage2_batch_output(Stage2BatchAssemblyInput {
        config: input.config,
        checked: input.checked,
        stage1: input.stage1,
        stage2_rows: &stage2_rows,
        product_uniskip: &product_uniskip.output,
        prefix: &prepared.prefix,
        batch: &batch,
    })?;

    let batch_output_claim_values = assembly.batch_output_claim_values.clone();
    let recorded = batch.proof.finish(&batch_output_claim_values, transcript)?;

    Ok(Stage2CommittedProofComponent {
        product_uniskip_proof: product_uniskip.output.proof,
        regular_batch_proof: recorded.proof,
        public: assembly.verifier_output.public.clone(),
        verifier_output: assembly.verifier_output,
        product_uniskip_output_claim_values: product_uniskip.output_claim_values,
        batch_output_claim_values,
        product_uniskip_committed_witness: product_uniskip.committed_witness,
        batch_committed_witness: recorded.committed_witness.ok_or_else(|| {
            invalid_sumcheck_output("Stage 2 committed batch witness material is missing")
        })?,
    })
}

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

#[cfg(feature = "zk")]
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
    checked: &CheckedInputs,
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
    let output_address_challenges = (0..config.log_k)
        .map(|_| transcript.challenge())
        .collect::<Vec<_>>();

    let input_claims = stage2_batch_relations(
        config,
        checked,
        stage1,
        product_uniskip,
        ram_read_write_gamma,
        instruction_gamma,
        &output_address_challenges,
    )?
    .input_claims()?;

    Ok(Stage2RegularBatchPrefixOutput {
        input_claims,
        ram_read_write_gamma,
        instruction_gamma,
        output_address_challenges,
    })
}

/// Build the five stage 2 batch relation objects (bundled with their wired input
/// claims) inline, exactly as the verifier's `verify_regular_batch` clear arm does.
/// The prover constructs them twice — once for the input claims (before the batched
/// sumcheck) and once for the opening points / expected outputs (after) — since the
/// relations carry no proof state.
fn stage2_batch_relations<F, C>(
    config: Stage2BatchProverConfig,
    checked: &CheckedInputs,
    stage1: &Stage1ClearOutput<F>,
    product_uniskip: &Stage2ProductUniSkipOutput<F, C>,
    ram_read_write_gamma: F,
    instruction_gamma: F,
    output_address_challenges: &[F],
) -> Result<Stage2BatchRelations<F>, ProverError>
where
    F: Field,
{
    let log_t = config.log_t;
    let log_k = config.log_k;
    let trace_dimensions = TraceDimensions::new(log_t);
    let rw_dimensions = read_write_dimensions(config);
    let product_dimensions = SpartanProductDimensions::new(log_t);
    let raf_dimensions = RamRafEvaluationDimensions::try_from(rw_dimensions).map_err(|error| {
        invalid_sumcheck_output(format!("Stage 2 RAM RAF dimensions invalid: {error}"))
    })?;

    let lowest_address = checked.public_io.memory_layout.get_lowest_address();
    let public_memory = PublicIoMemory::new(&checked.public_io).map_err(|error| {
        ProverError::InvalidStageRequest {
            reason: format!("invalid public IO memory for Stage 2 output check: {error}"),
        }
    })?;

    let ram_read_write = RamReadWriteChecking::new(
        rw_dimensions,
        log_k,
        ram_read_write_gamma,
        product_uniskip.tau_low.clone(),
    );
    let product_remainder = ProductRemainder::new(
        product_dimensions,
        product_uniskip.challenge,
        product_uniskip.tau_high,
        product_uniskip.tau_low.clone(),
    );
    let instruction_reduction = InstructionClaimReduction::new(
        trace_dimensions,
        instruction_gamma,
        product_uniskip.tau_low.clone(),
    );
    let ram_raf = RamRafEvaluation::new(
        rw_dimensions,
        raf_dimensions,
        log_k,
        lowest_address,
        product_uniskip.tau_low.clone(),
    );
    let ram_output = RamOutputCheck::new(
        rw_dimensions,
        output_address_challenges.to_vec(),
        public_memory,
    );

    Ok(Stage2BatchRelations {
        ram_read_write,
        product_remainder,
        instruction_reduction,
        ram_raf,
        ram_output,
        ram_read_write_inputs: RamReadWriteInputClaims::from_upstream(stage1),
        product_remainder_inputs: ProductRemainderInputClaims::from_uniskip_output(
            product_uniskip.output_claim,
        ),
        instruction_reduction_inputs: InstructionClaimReductionInputClaims::from_upstream(stage1),
        ram_raf_inputs: RamRafEvaluationInputClaims::from_upstream(stage1),
        ram_output_inputs: RamOutputCheckInputClaims::from_upstream(),
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
        product_remainder: ProductRemainderOutputClaims {
            left_instruction_input: openings.product_remainder.left_instruction_input,
            right_instruction_input: openings.product_remainder.right_instruction_input,
            jump_flag: openings.product_remainder.jump_flag,
            write_lookup_output_to_rd: openings.product_remainder.write_lookup_output_to_rd,
            lookup_output: openings.product_remainder.lookup_output,
            branch_flag: openings.product_remainder.branch_flag,
            next_is_noop: openings.product_remainder.next_is_noop,
            virtual_instruction: openings.product_remainder.virtual_instruction,
        },
        instruction_claim_reduction: InstructionClaimReductionOutputClaims {
            lookup_output: None,
            left_lookup_operand: openings.instruction_claim_reduction.left_lookup_operand,
            right_lookup_operand: openings.instruction_claim_reduction.right_lookup_operand,
            left_instruction_input: None,
            right_instruction_input: None,
        },
    })
}

struct RegularBatchProof<F: Field, Proof> {
    proof: Proof,
    challenges: Vec<F>,
    batching_coefficients: Vec<F>,
    output_claim: F,
    ram_read_write: RamReadWriteOutputClaims<F>,
    ram_raf_evaluation: F,
    ram_output_check: F,
}

#[derive(Clone)]
struct Stage2TailOutputOpenings<F: Field> {
    product_remainder: ProductRemainderOutputClaims<F>,
    instruction_claim_reduction: InstructionClaimReductionOutputClaims<F>,
}

#[cfg(feature = "zk")]
struct CommittedProductUniSkip<F: Field, C> {
    output: Stage2ProductUniSkipOutput<F, C>,
    output_claim_values: Vec<F>,
    committed_witness: CommittedSumcheckWitness<F>,
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
        ram_read_write: RamReadWriteOutputClaims { val, ra, inc },
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
