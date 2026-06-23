use jolt_backends::poly::{Stage8JointRlcConfig, Stage8JointRlcSource};
use jolt_claims::protocols::jolt::{
    formulas::dimensions::TracePolynomialOrder, formulas::ra::JoltRaPolynomialLayout,
    AdviceClaimReductionLayout,
};
use jolt_crypto::HomomorphicCommitment;
use jolt_field::Field;
#[cfg(feature = "zk")]
use jolt_openings::ZkOpeningScheme;
use jolt_openings::{AdditivelyHomomorphic, CommitmentScheme};
use jolt_transcript::Transcript;
#[cfg(feature = "zk")]
use jolt_verifier::stages::stage8::stage8_zk_final_opening_batch;
#[cfg(feature = "zk")]
use jolt_verifier::stages::stage8::Stage8FinalOpeningStructure as Stage8OpeningStructure;
use jolt_verifier::stages::{
    stage6::Stage6ClearOutput,
    stage7::outputs::Stage7ClearOutput,
    stage8::{stage8_clear_final_opening_batch, Stage8FinalOpeningBatchInput},
};
use jolt_witness::{
    protocols::jolt_vm::{JoltVmNamespace, JoltVmStage6Rows},
    WitnessProvider,
};

use crate::ProverError;

/// Canonical Stage 8 prover configuration.
///
/// Carries the verifier-equivalent dimensions, RA layout, trace polynomial order,
/// and advice layouts needed to build the final batched-opening order exactly as
/// `jolt-verifier/src/stages/stage8/verify.rs` does.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage8ProverConfig {
    pub log_t: usize,
    pub committed_chunk_bits: usize,
    pub layout: JoltRaPolynomialLayout,
    pub trace_polynomial_order: TracePolynomialOrder,
    pub trusted_advice_layout: Option<AdviceClaimReductionLayout>,
    pub untrusted_advice_layout: Option<AdviceClaimReductionLayout>,
}

impl Stage8ProverConfig {
    pub const fn new(
        log_t: usize,
        committed_chunk_bits: usize,
        layout: JoltRaPolynomialLayout,
        trace_polynomial_order: TracePolynomialOrder,
        trusted_advice_layout: Option<AdviceClaimReductionLayout>,
        untrusted_advice_layout: Option<AdviceClaimReductionLayout>,
    ) -> Self {
        Self {
            log_t,
            committed_chunk_bits,
            layout,
            trace_polynomial_order,
            trusted_advice_layout,
            untrusted_advice_layout,
        }
    }
}

/// Canonical Stage 8 prover output (clear path): the generated
/// `joint_opening_proof` that becomes the `JoltProof` PCS artifact.
#[derive(Clone, Debug)]
pub(crate) struct Stage8ProofOutput<Proof> {
    pub(crate) joint_opening_proof: Proof,
}

/// Canonical Stage 8 prover output for ZK mode: the verifier-equivalent opening
/// structure, generated PCS proof, and prover-side opening blind retained for
/// BlindFold.
#[cfg(feature = "zk")]
#[derive(Clone, Debug)]
pub(crate) struct Stage8ZkProofOutput<F: Field, Proof, Blind> {
    pub(crate) structure: Stage8OpeningStructure<F>,
    pub(crate) joint_opening_proof: Proof,
    pub(crate) hiding_evaluation_blind: Blind,
}

#[cfg(feature = "zk")]
type Stage8ZkOutputFor<F, PCS> =
    Stage8ZkProofOutput<F, <PCS as CommitmentScheme>::Proof, <PCS as ZkOpeningScheme>::Blind>;

#[expect(
    clippy::too_many_arguments,
    reason = "Stage 8 needs prior-stage outputs plus the PCS commitments, retained hints, and setup, which are PCS-generic and cannot bundle into the non-generic prover input."
)]
pub(crate) fn prove_stage8<F, PCS, W, T>(
    config: &Stage8ProverConfig,
    stage6: &Stage6ClearOutput<F>,
    stage7: &Stage7ClearOutput<F>,
    witness: &W,
    commitments: &[PCS::Output],
    hints: Vec<PCS::OpeningHint>,
    setup: &PCS::ProverSetup,
    transcript: &mut T,
) -> Result<Stage8ProofOutput<PCS::Proof>, ProverError>
where
    F: Field,
    <F as jolt_field::WithAccumulator>::Accumulator: jolt_field::RingAccumulator<Element = F>,
    PCS: CommitmentScheme<Field = F> + AdditivelyHomomorphic,
    PCS::Output: HomomorphicCommitment<F>,
    W: WitnessProvider<F, JoltVmNamespace> + JoltVmStage6Rows + Sync,
    T: Transcript<Challenge = F>,
{
    let final_opening_batch = stage8_clear_final_opening_batch(
        stage8_final_opening_batch_input(config, stage6, stage7),
        transcript,
    )?;
    let structure = final_opening_batch.structure;
    let gamma_powers = final_opening_batch.gamma_powers;
    require_stage8_batch_size(
        "Stage 8",
        commitments.len(),
        hints.len(),
        structure.opening_ids.len(),
    )?;

    let combined_hint = PCS::combine_hints(hints, &gamma_powers);
    let joint_polynomial = Stage8JointRlcSource::new(
        stage8_joint_rlc_config(config),
        witness,
        gamma_powers.clone(),
    )?;
    let joint_opening_proof = PCS::open_poly(
        &joint_polynomial,
        structure.pcs_opening_point.as_slice(),
        structure.joint_claim,
        setup,
        Some(combined_hint),
        transcript,
    );
    PCS::bind_opening_inputs(
        transcript,
        structure.opening_point.as_slice(),
        &structure.joint_claim,
    );

    Ok(Stage8ProofOutput {
        joint_opening_proof,
    })
}

#[expect(
    clippy::too_many_arguments,
    reason = "Stage 8 ZK opening uses the same verifier-order inputs as clear Stage 8 plus hidden PCS output material."
)]
#[cfg(feature = "zk")]
pub(crate) fn prove_stage8_zk<F, PCS, W, T>(
    config: &Stage8ProverConfig,
    stage6: &Stage6ClearOutput<F>,
    stage7: &Stage7ClearOutput<F>,
    witness: &W,
    commitments: &[PCS::Output],
    hints: Vec<PCS::OpeningHint>,
    setup: &PCS::ProverSetup,
    transcript: &mut T,
) -> Result<Stage8ZkOutputFor<F, PCS>, ProverError>
where
    F: Field,
    <F as jolt_field::WithAccumulator>::Accumulator: jolt_field::RingAccumulator<Element = F>,
    PCS: CommitmentScheme<Field = F> + AdditivelyHomomorphic + ZkOpeningScheme,
    PCS::Output: HomomorphicCommitment<F>,
    W: WitnessProvider<F, JoltVmNamespace> + JoltVmStage6Rows + Sync,
    T: Transcript<Challenge = F>,
{
    let final_opening_batch = stage8_zk_final_opening_batch(
        stage8_final_opening_batch_input(config, stage6, stage7),
        transcript,
    )?;
    let structure = final_opening_batch.structure;
    let gamma_powers = final_opening_batch.gamma_powers;
    require_stage8_batch_size(
        "Stage 8 ZK",
        commitments.len(),
        hints.len(),
        structure.opening_ids.len(),
    )?;

    let combined_hint = PCS::combine_hints(hints, &gamma_powers);
    let joint_polynomial =
        Stage8JointRlcSource::new(stage8_joint_rlc_config(config), witness, gamma_powers)?;
    let (joint_opening_proof, hiding_evaluation_commitment, hiding_evaluation_blind) =
        PCS::open_zk_poly(
            &joint_polynomial,
            structure.pcs_opening_point.as_slice(),
            structure.joint_claim,
            setup,
            combined_hint,
            transcript,
        );
    PCS::bind_zk_opening_inputs(
        transcript,
        structure.opening_point.as_slice(),
        &hiding_evaluation_commitment,
    );

    Ok(Stage8ZkProofOutput {
        structure,
        joint_opening_proof,
        hiding_evaluation_blind,
    })
}

fn stage8_joint_rlc_config(config: &Stage8ProverConfig) -> Stage8JointRlcConfig<'_> {
    Stage8JointRlcConfig {
        log_t: config.log_t,
        committed_chunk_bits: config.committed_chunk_bits,
        layout: config.layout,
        trace_polynomial_order: config.trace_polynomial_order,
        trusted_advice_layout: config.trusted_advice_layout.as_ref(),
        untrusted_advice_layout: config.untrusted_advice_layout.as_ref(),
    }
}

fn stage8_final_opening_batch_input<'a, F: Field>(
    config: &'a Stage8ProverConfig,
    stage6: &'a Stage6ClearOutput<F>,
    stage7: &'a Stage7ClearOutput<F>,
) -> Stage8FinalOpeningBatchInput<'a, F> {
    Stage8FinalOpeningBatchInput {
        log_t: config.log_t,
        committed_chunk_bits: config.committed_chunk_bits,
        layout: config.layout,
        trace_polynomial_order: config.trace_polynomial_order,
        trusted_advice_layout: config.trusted_advice_layout.as_ref(),
        untrusted_advice_layout: config.untrusted_advice_layout.as_ref(),
        stage6,
        stage7,
    }
}

fn require_stage8_batch_size(
    label: &'static str,
    commitments: usize,
    hints: usize,
    opening_ids: usize,
) -> Result<(), ProverError> {
    if commitments == opening_ids && hints == opening_ids {
        return Ok(());
    }
    Err(ProverError::InvalidStageRequest {
        reason: format!(
            "{label} batch size mismatch: {commitments} commitments, {hints} hints, {opening_ids} opening ids"
        ),
    })
}
