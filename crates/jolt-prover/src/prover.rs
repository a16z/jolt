#[cfg(feature = "zk")]
use common::constants::MAX_BLINDFOLD_GENERATORS;
use common::jolt_device::JoltDevice;
#[cfg(feature = "zk")]
use jolt_backends::{
    BlindFoldBackend, BlindFoldCrossTermErrorRowsRequest, BlindFoldErrorRowsRequest,
    BlindFoldFoldErrorRowsRequest, BlindFoldFoldErrorScalarsRequest, BlindFoldFoldRowsRequest,
    BlindFoldFoldScalarsRequest, BlindFoldRowCommitmentRequest, BlindFoldRowOpeningRequest,
};
use jolt_backends::{
    RamReadWriteSumcheckBackend, Stage3SpartanSumcheckBackend, Stage4ReadWriteSumcheckBackend,
    Stage5ValueEvaluationSumcheckBackend, Stage6RegularBatchSumcheckBackend, SumcheckBackend,
};
#[cfg(feature = "field-inline")]
use jolt_claims::protocols::field_inline::formulas::bytecode as field_bytecode;
use jolt_claims::protocols::jolt::JoltFormulaDimensions;
use jolt_claims::protocols::jolt::{AdviceClaimReductionLayout, JoltAdviceKind};
use jolt_crypto::{HomomorphicCommitment, VectorCommitment};
use jolt_field::Field;
#[cfg(feature = "zk")]
use jolt_openings::ZkOpeningScheme;
use jolt_openings::{AdditivelyHomomorphic, CommitmentScheme};
use jolt_poly::{block_selector_mle_msb, sparse_segments_mle_msb};
use jolt_program::preprocess::PublicInitialRam;
use jolt_transcript::AppendToTranscript;
#[cfg(feature = "zk")]
use jolt_transcript::Label;
use jolt_transcript::{Blake2bTranscript, Transcript};
use jolt_verifier::{JoltProof, JOLT_VERIFIER_CONFIG};
#[cfg(feature = "field-inline")]
use jolt_witness::protocols::jolt_vm::field_inline::{
    FieldInlineNamespace, FieldInlineRegisterReadWriteRows,
};
use jolt_witness::{
    protocols::jolt_vm::{
        JoltVmNamespace, JoltVmProductUniskipRows, JoltVmRegisterReadWriteRows,
        JoltVmSpartanOuterRows, JoltVmStage2Rows, JoltVmStage3InstructionRegisterRows,
        JoltVmStage3ShiftRows, JoltVmStage5InstructionReadRafRows, JoltVmStage6Rows,
        RV64_LOOKUP_ADDRESS_BITS,
    },
    CommittedWitnessProvider, WitnessProvider,
};

use crate::assembly::ProofAssembly;
use crate::stages::stage0::{
    self, CommitmentStageConfig, CommitmentStageInput, CommitmentStageOutput,
};
use crate::stages::stage1::{input::Stage1ProverConfig, input::Stage1ProverInput};
use crate::stages::stage2::input::{Stage2BatchProverConfig, Stage2ProverInput};
use crate::stages::stage3::input::{Stage3ProverConfig, Stage3ProverInput};
use crate::stages::stage4::{
    input::{Stage4ProverConfig, Stage4ProverInput},
    output::{Stage4RamValCheckAdviceContribution, Stage4RamValCheckInitialEvaluation},
};
use crate::stages::stage5::input::{Stage5ProverConfig, Stage5ProverInput};
use crate::stages::stage6::input::{Stage6ProverConfig, Stage6ProverInput};
use crate::stages::stage7::input::{Stage7ProverConfig, Stage7ProverInput};
use crate::stages::stage8::input::Stage8ProverConfig;
use crate::{
    JoltProverPreprocessing, ProverConfig, ProverError, ProverFeatureSet, ProverProofShape,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProverOutput<PCS, VC>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    pub proof: JoltProof<PCS, VC>,
    pub trusted_advice_commitment: Option<PCS::Output>,
}

impl<PCS, VC> ProverOutput<PCS, VC>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    pub fn into_proof(self) -> JoltProof<PCS, VC> {
        self.proof
    }
}

#[cfg(feature = "frontier-harness")]
fn timed_stage<T, E>(label: &'static str, f: impl FnOnce() -> Result<T, E>) -> Result<T, E> {
    let start = std::time::Instant::now();
    let result = f();
    crate::timing::record_stage_timing(label, start.elapsed().as_secs_f64() * 1000.0);
    result
}

#[cfg(not(feature = "frontier-harness"))]
fn timed_stage<T, E>(_label: &'static str, f: impl FnOnce() -> Result<T, E>) -> Result<T, E> {
    f()
}

#[cfg(feature = "field-inline")]
trait FieldInlineProverWitness<F>:
    CommittedWitnessProvider<F, FieldInlineNamespace>
    + WitnessProvider<F, FieldInlineNamespace>
    + FieldInlineRegisterReadWriteRows<F>
    + Sync
where
    F: Field,
{
}

#[cfg(feature = "field-inline")]
impl<F, T> FieldInlineProverWitness<F> for T
where
    F: Field,
    T: CommittedWitnessProvider<F, FieldInlineNamespace>
        + WitnessProvider<F, FieldInlineNamespace>
        + FieldInlineRegisterReadWriteRows<F>
        + Sync,
{
}

#[cfg(not(feature = "field-inline"))]
trait FieldInlineProverWitness<F> {}

#[cfg(not(feature = "field-inline"))]
impl<F, T> FieldInlineProverWitness<F> for T {}

#[cfg(feature = "field-inline")]
trait ClearProverBackend<F>:
    SumcheckBackend<F, JoltVmNamespace>
    + SumcheckBackend<F, FieldInlineNamespace>
    + RamReadWriteSumcheckBackend<F>
    + Stage3SpartanSumcheckBackend<F>
    + Stage4ReadWriteSumcheckBackend<F>
    + Stage5ValueEvaluationSumcheckBackend<F>
    + Stage6RegularBatchSumcheckBackend<F>
where
    F: Field,
{
}

#[cfg(feature = "field-inline")]
impl<F, T> ClearProverBackend<F> for T
where
    F: Field,
    T: SumcheckBackend<F, JoltVmNamespace>
        + SumcheckBackend<F, FieldInlineNamespace>
        + RamReadWriteSumcheckBackend<F>
        + Stage3SpartanSumcheckBackend<F>
        + Stage4ReadWriteSumcheckBackend<F>
        + Stage5ValueEvaluationSumcheckBackend<F>
        + Stage6RegularBatchSumcheckBackend<F>,
{
}

#[cfg(not(feature = "field-inline"))]
trait ClearProverBackend<F>:
    SumcheckBackend<F, JoltVmNamespace>
    + RamReadWriteSumcheckBackend<F>
    + Stage3SpartanSumcheckBackend<F>
    + Stage4ReadWriteSumcheckBackend<F>
    + Stage5ValueEvaluationSumcheckBackend<F>
    + Stage6RegularBatchSumcheckBackend<F>
where
    F: Field,
{
}

#[cfg(not(feature = "field-inline"))]
impl<F, T> ClearProverBackend<F> for T
where
    F: Field,
    T: SumcheckBackend<F, JoltVmNamespace>
        + RamReadWriteSumcheckBackend<F>
        + Stage3SpartanSumcheckBackend<F>
        + Stage4ReadWriteSumcheckBackend<F>
        + Stage5ValueEvaluationSumcheckBackend<F>
        + Stage6RegularBatchSumcheckBackend<F>,
{
}

#[doc(hidden)]
#[cfg(feature = "zk")]
pub trait BlindFoldProverBackend<F>: BlindFoldBackend<F>
where
    F: Field,
{
}

#[cfg(feature = "zk")]
impl<F, T> BlindFoldProverBackend<F> for T
where
    F: Field,
    T: BlindFoldBackend<F>,
{
}

#[doc(hidden)]
#[cfg(not(feature = "zk"))]
pub trait BlindFoldProverBackend<F> {}

#[cfg(not(feature = "zk"))]
impl<F, T> BlindFoldProverBackend<F> for T {}

#[cfg(feature = "zk")]
pub trait ProverPcs<VC>:
    CommitmentScheme
    + AdditivelyHomomorphic
    + ZkOpeningScheme<
        HidingCommitment = <VC as jolt_crypto::Commitment>::Output,
        Blind = <Self as CommitmentScheme>::Field,
    >
where
    VC: VectorCommitment<Field = <Self as CommitmentScheme>::Field>,
    <Self as jolt_crypto::Commitment>::Output:
        HomomorphicCommitment<<Self as CommitmentScheme>::Field>,
    <VC as jolt_crypto::Commitment>::Output:
        HomomorphicCommitment<<Self as CommitmentScheme>::Field>,
{
}

#[cfg(feature = "zk")]
impl<PCS, VC> ProverPcs<VC> for PCS
where
    PCS: CommitmentScheme
        + AdditivelyHomomorphic
        + ZkOpeningScheme<
            HidingCommitment = <VC as jolt_crypto::Commitment>::Output,
            Blind = <PCS as CommitmentScheme>::Field,
        >,
    VC: VectorCommitment<Field = <PCS as CommitmentScheme>::Field>,
    <PCS as jolt_crypto::Commitment>::Output:
        HomomorphicCommitment<<PCS as CommitmentScheme>::Field>,
    <VC as jolt_crypto::Commitment>::Output:
        HomomorphicCommitment<<PCS as CommitmentScheme>::Field>,
{
}

#[cfg(not(feature = "zk"))]
pub trait ProverPcs<VC>: CommitmentScheme + AdditivelyHomomorphic
where
    VC: VectorCommitment<Field = <Self as CommitmentScheme>::Field>,
    <Self as jolt_crypto::Commitment>::Output:
        HomomorphicCommitment<<Self as CommitmentScheme>::Field>,
{
}

#[cfg(not(feature = "zk"))]
impl<PCS, VC> ProverPcs<VC> for PCS
where
    PCS: CommitmentScheme + AdditivelyHomomorphic,
    VC: VectorCommitment<Field = <PCS as CommitmentScheme>::Field>,
    <PCS as jolt_crypto::Commitment>::Output:
        HomomorphicCommitment<<PCS as CommitmentScheme>::Field>,
{
}

#[cfg(not(feature = "field-inline"))]
pub fn prove<PCS, VC, B, W>(
    preprocessing: &JoltProverPreprocessing<PCS, VC>,
    public_io: &JoltDevice,
    witness: &W,
    config: ProverConfig,
    backend: &mut B,
) -> Result<JoltProof<PCS, VC>, ProverError>
where
    PCS: ProverPcs<VC>,
    PCS::Output: AppendToTranscript + HomomorphicCommitment<PCS::Field>,
    <PCS::Field as jolt_field::WithAccumulator>::Accumulator:
        jolt_field::RingAccumulator<Element = PCS::Field>,
    VC: VectorCommitment<Field = PCS::Field>,
    VC::Output: HomomorphicCommitment<PCS::Field>,
    B: stage0::CommitmentStageBackend<PCS::Field, PCS>
        + SumcheckBackend<PCS::Field, JoltVmNamespace>
        + RamReadWriteSumcheckBackend<PCS::Field>
        + Stage3SpartanSumcheckBackend<PCS::Field>
        + Stage4ReadWriteSumcheckBackend<PCS::Field>
        + Stage5ValueEvaluationSumcheckBackend<PCS::Field>
        + Stage6RegularBatchSumcheckBackend<PCS::Field>
        + BlindFoldProverBackend<PCS::Field>,
    W: CommittedWitnessProvider<PCS::Field, JoltVmNamespace>
        + WitnessProvider<PCS::Field, JoltVmNamespace>
        + JoltVmSpartanOuterRows
        + JoltVmProductUniskipRows
        + JoltVmStage2Rows
        + JoltVmStage3ShiftRows
        + JoltVmStage3InstructionRegisterRows
        + JoltVmRegisterReadWriteRows
        + JoltVmStage5InstructionReadRafRows
        + JoltVmStage6Rows
        + Sync,
{
    prove_with_output(preprocessing, public_io, witness, config, backend)
        .map(ProverOutput::into_proof)
}

#[cfg(feature = "field-inline")]
pub fn prove<PCS, VC, B, W, FI>(
    preprocessing: &JoltProverPreprocessing<PCS, VC>,
    public_io: &JoltDevice,
    witness: &W,
    field_inline_witness: &FI,
    config: ProverConfig,
    backend: &mut B,
) -> Result<JoltProof<PCS, VC>, ProverError>
where
    PCS: ProverPcs<VC>,
    PCS::Output: AppendToTranscript + HomomorphicCommitment<PCS::Field>,
    <PCS::Field as jolt_field::WithAccumulator>::Accumulator:
        jolt_field::RingAccumulator<Element = PCS::Field>,
    VC: VectorCommitment<Field = PCS::Field>,
    VC::Output: HomomorphicCommitment<PCS::Field>,
    B: stage0::CommitmentStageBackend<PCS::Field, PCS>
        + SumcheckBackend<PCS::Field, JoltVmNamespace>
        + SumcheckBackend<PCS::Field, FieldInlineNamespace>
        + RamReadWriteSumcheckBackend<PCS::Field>
        + Stage3SpartanSumcheckBackend<PCS::Field>
        + Stage4ReadWriteSumcheckBackend<PCS::Field>
        + Stage5ValueEvaluationSumcheckBackend<PCS::Field>
        + Stage6RegularBatchSumcheckBackend<PCS::Field>
        + BlindFoldProverBackend<PCS::Field>,
    W: CommittedWitnessProvider<PCS::Field, JoltVmNamespace>
        + WitnessProvider<PCS::Field, JoltVmNamespace>
        + JoltVmSpartanOuterRows
        + JoltVmProductUniskipRows
        + JoltVmStage2Rows
        + JoltVmStage3ShiftRows
        + JoltVmStage3InstructionRegisterRows
        + JoltVmRegisterReadWriteRows
        + JoltVmStage5InstructionReadRafRows
        + JoltVmStage6Rows
        + Sync,
    FI: CommittedWitnessProvider<PCS::Field, FieldInlineNamespace>
        + WitnessProvider<PCS::Field, FieldInlineNamespace>
        + FieldInlineRegisterReadWriteRows<PCS::Field>
        + Sync,
{
    prove_with_output(
        preprocessing,
        public_io,
        witness,
        field_inline_witness,
        config,
        backend,
    )
    .map(ProverOutput::into_proof)
}

#[cfg(not(feature = "field-inline"))]
pub fn prove_with_output<PCS, VC, B, W>(
    preprocessing: &JoltProverPreprocessing<PCS, VC>,
    public_io: &JoltDevice,
    witness: &W,
    config: ProverConfig,
    backend: &mut B,
) -> Result<ProverOutput<PCS, VC>, ProverError>
where
    PCS: ProverPcs<VC>,
    PCS::Output: AppendToTranscript + HomomorphicCommitment<PCS::Field>,
    <PCS::Field as jolt_field::WithAccumulator>::Accumulator:
        jolt_field::RingAccumulator<Element = PCS::Field>,
    VC: VectorCommitment<Field = PCS::Field>,
    VC::Output: HomomorphicCommitment<PCS::Field>,
    B: stage0::CommitmentStageBackend<PCS::Field, PCS>
        + SumcheckBackend<PCS::Field, JoltVmNamespace>
        + RamReadWriteSumcheckBackend<PCS::Field>
        + Stage3SpartanSumcheckBackend<PCS::Field>
        + Stage4ReadWriteSumcheckBackend<PCS::Field>
        + Stage5ValueEvaluationSumcheckBackend<PCS::Field>
        + Stage6RegularBatchSumcheckBackend<PCS::Field>
        + BlindFoldProverBackend<PCS::Field>,
    W: CommittedWitnessProvider<PCS::Field, JoltVmNamespace>
        + WitnessProvider<PCS::Field, JoltVmNamespace>
        + JoltVmSpartanOuterRows
        + JoltVmProductUniskipRows
        + JoltVmStage2Rows
        + JoltVmStage3ShiftRows
        + JoltVmStage3InstructionRegisterRows
        + JoltVmRegisterReadWriteRows
        + JoltVmStage5InstructionReadRafRows
        + JoltVmStage6Rows
        + Sync,
{
    prove_with_output_inner(preprocessing, public_io, witness, &(), config, backend)
}

#[cfg(feature = "field-inline")]
pub fn prove_with_output<PCS, VC, B, W, FI>(
    preprocessing: &JoltProverPreprocessing<PCS, VC>,
    public_io: &JoltDevice,
    witness: &W,
    field_inline_witness: &FI,
    config: ProverConfig,
    backend: &mut B,
) -> Result<ProverOutput<PCS, VC>, ProverError>
where
    PCS: ProverPcs<VC>,
    PCS::Output: AppendToTranscript + HomomorphicCommitment<PCS::Field>,
    <PCS::Field as jolt_field::WithAccumulator>::Accumulator:
        jolt_field::RingAccumulator<Element = PCS::Field>,
    VC: VectorCommitment<Field = PCS::Field>,
    VC::Output: HomomorphicCommitment<PCS::Field>,
    B: stage0::CommitmentStageBackend<PCS::Field, PCS>
        + SumcheckBackend<PCS::Field, JoltVmNamespace>
        + SumcheckBackend<PCS::Field, FieldInlineNamespace>
        + RamReadWriteSumcheckBackend<PCS::Field>
        + Stage3SpartanSumcheckBackend<PCS::Field>
        + Stage4ReadWriteSumcheckBackend<PCS::Field>
        + Stage5ValueEvaluationSumcheckBackend<PCS::Field>
        + Stage6RegularBatchSumcheckBackend<PCS::Field>
        + BlindFoldProverBackend<PCS::Field>,
    W: CommittedWitnessProvider<PCS::Field, JoltVmNamespace>
        + WitnessProvider<PCS::Field, JoltVmNamespace>
        + JoltVmSpartanOuterRows
        + JoltVmProductUniskipRows
        + JoltVmStage2Rows
        + JoltVmStage3ShiftRows
        + JoltVmStage3InstructionRegisterRows
        + JoltVmRegisterReadWriteRows
        + JoltVmStage5InstructionReadRafRows
        + JoltVmStage6Rows
        + Sync,
    FI: CommittedWitnessProvider<PCS::Field, FieldInlineNamespace>
        + WitnessProvider<PCS::Field, FieldInlineNamespace>
        + FieldInlineRegisterReadWriteRows<PCS::Field>
        + Sync,
{
    prove_with_output_inner(
        preprocessing,
        public_io,
        witness,
        field_inline_witness,
        config,
        backend,
    )
}

fn prove_with_output_inner<PCS, VC, B, W, FI>(
    preprocessing: &JoltProverPreprocessing<PCS, VC>,
    public_io: &JoltDevice,
    witness: &W,
    field_inline_witness: &FI,
    config: ProverConfig,
    backend: &mut B,
) -> Result<ProverOutput<PCS, VC>, ProverError>
where
    PCS: ProverPcs<VC>,
    PCS::Output: AppendToTranscript + HomomorphicCommitment<PCS::Field>,
    <PCS::Field as jolt_field::WithAccumulator>::Accumulator:
        jolt_field::RingAccumulator<Element = PCS::Field>,
    VC: VectorCommitment<Field = PCS::Field>,
    VC::Output: HomomorphicCommitment<PCS::Field>,
    B: stage0::CommitmentStageBackend<PCS::Field, PCS>
        + ClearProverBackend<PCS::Field>
        + BlindFoldProverBackend<PCS::Field>,
    W: CommittedWitnessProvider<PCS::Field, JoltVmNamespace>
        + WitnessProvider<PCS::Field, JoltVmNamespace>
        + JoltVmSpartanOuterRows
        + JoltVmProductUniskipRows
        + JoltVmStage2Rows
        + JoltVmStage3ShiftRows
        + JoltVmStage3InstructionRegisterRows
        + JoltVmRegisterReadWriteRows
        + JoltVmStage5InstructionReadRafRows
        + JoltVmStage6Rows
        + Sync,
    FI: FieldInlineProverWitness<PCS::Field>,
{
    validate_config(&config)?;
    let mut assembly = ProofAssembly::<PCS, VC>::new(config.clone(), public_io);
    let stage0 = timed_stage("stage0", || {
        prove_stage0(
            preprocessing,
            public_io,
            witness,
            field_inline_witness,
            &config,
            backend,
        )
    })?;
    assembly.record_stage0(stage0)?;

    #[cfg(feature = "zk")]
    if config.features.zk {
        let blindfold_proof = prove_zk_stages(
            preprocessing,
            public_io,
            witness,
            field_inline_witness,
            &config,
            backend,
            &mut assembly,
        )?;
        let (proof, trusted_advice_commitment) = timed_stage("assemble_zk_proof", || {
            assembly.into_zk_proof(blindfold_proof)
        })?;
        return Ok(ProverOutput {
            proof,
            trusted_advice_commitment,
        });
    }

    prove_clear_stages(
        preprocessing,
        public_io,
        witness,
        field_inline_witness,
        &config,
        backend,
        &mut assembly,
    )?;

    if !config.features.zk {
        let (proof, trusted_advice_commitment) =
            timed_stage("assemble_clear_proof", || assembly.into_clear_proof())?;
        return Ok(ProverOutput {
            proof,
            trusted_advice_commitment,
        });
    }

    Err(ProverError::FrontierNotImplemented {
        frontier: assembly.next_frontier(),
    })
}

fn validate_config(config: &ProverConfig) -> Result<(), ProverError> {
    if config.features != ProverFeatureSet::COMPILED {
        return Err(ProverError::InvalidProverConfig {
            reason: format!(
                "requested features {:?} do not match compiled features {:?}",
                config.features,
                ProverFeatureSet::COMPILED
            ),
        });
    }

    let protocol_features = ProverFeatureSet::from_protocol(&config.protocol);
    if protocol_features != config.features {
        return Err(ProverError::InvalidProverConfig {
            reason: format!(
                "protocol {:?} implies features {:?}, but requested features are {:?}",
                config.protocol, protocol_features, config.features
            ),
        });
    }

    if config.protocol != JOLT_VERIFIER_CONFIG {
        return Err(ProverError::InvalidProverConfig {
            reason: format!(
                "requested protocol {:?} does not match compiled verifier protocol {:?}",
                config.protocol, JOLT_VERIFIER_CONFIG
            ),
        });
    }

    if let Some(proof_shape) = config.proof_shape {
        if proof_shape.trace_length == 0 || !proof_shape.trace_length.is_power_of_two() {
            return Err(ProverError::InvalidProverConfig {
                reason: format!(
                    "proof trace_length must be a nonzero power of two, got {}",
                    proof_shape.trace_length
                ),
            });
        }
        if proof_shape.ram_k == 0 || !proof_shape.ram_k.is_power_of_two() {
            return Err(ProverError::InvalidProverConfig {
                reason: format!(
                    "proof ram_k must be a nonzero power of two, got {}",
                    proof_shape.ram_k
                ),
            });
        }
    }

    Ok(())
}

#[cfg(feature = "zk")]
fn prove_zk_stages<PCS, VC, B, W, FI>(
    preprocessing: &JoltProverPreprocessing<PCS, VC>,
    public_io: &JoltDevice,
    witness: &W,
    field_inline_witness: &FI,
    config: &ProverConfig,
    backend: &mut B,
    assembly: &mut ProofAssembly<PCS, VC>,
) -> Result<jolt_blindfold::BlindFoldProof<PCS::Field, VC::Output>, ProverError>
where
    PCS: ProverPcs<VC>,
    PCS::Output: AppendToTranscript + HomomorphicCommitment<PCS::Field>,
    <PCS::Field as jolt_field::WithAccumulator>::Accumulator:
        jolt_field::RingAccumulator<Element = PCS::Field>,
    VC: VectorCommitment<Field = PCS::Field>,
    VC::Output: HomomorphicCommitment<PCS::Field>,
    B: ClearProverBackend<PCS::Field> + BlindFoldBackend<PCS::Field>,
    W: JoltVmSpartanOuterRows
        + JoltVmProductUniskipRows
        + JoltVmStage2Rows
        + JoltVmStage3ShiftRows
        + JoltVmStage3InstructionRegisterRows
        + JoltVmRegisterReadWriteRows
        + JoltVmStage5InstructionReadRafRows
        + JoltVmStage6Rows
        + WitnessProvider<PCS::Field, JoltVmNamespace>
        + Sync,
    FI: FieldInlineProverWitness<PCS::Field>,
{
    #[cfg(not(feature = "field-inline"))]
    let _ = field_inline_witness;

    let proof_shape = required_proof_shape(config)?;
    let checked = zk_checked_inputs(preprocessing, public_io, config)?;
    let vc_setup = preprocessing.verifier.vc_setup.as_ref().ok_or_else(|| {
        ProverError::InvalidProverConfig {
            reason: "ZK proving requires verifier preprocessing with vector-commitment setup"
                .to_owned(),
        }
    })?;
    let mut transcript = Blake2bTranscript::<PCS::Field>::new(b"Jolt");
    assembly.absorb_stage0(&checked, &mut transcript)?;
    let log_t = proof_shape_log_t(proof_shape);

    let stage1 = timed_stage("stage1", || {
        crate::stages::stage1::prove::prove_committed_boundary(
            Stage1ProverInput::new(
                Stage1ProverConfig::new(log_t),
                witness,
                #[cfg(feature = "field-inline")]
                field_inline_witness,
            ),
            backend,
            &mut transcript,
            vc_setup,
        )
    })?;
    let stage1_verifier_output = stage1.verifier_output.clone();
    assembly.record_stage1_committed(stage1)?;

    let log_k = proof_shape_log_k(proof_shape);
    let stage2 = timed_stage("stage2", || {
        crate::stages::stage2::prove::prove_committed_boundary(
            Stage2ProverInput::new(
                Stage2BatchProverConfig::new(log_t, log_k, proof_shape.rw_config),
                &checked,
                &stage1_verifier_output,
                witness,
                #[cfg(feature = "field-inline")]
                field_inline_witness,
            ),
            backend,
            vc_setup,
            &mut transcript,
        )
    })?;
    let stage2_verifier_output = stage2.verifier_output.clone();
    assembly.record_stage2_committed(stage2)?;

    let stage3 = timed_stage("stage3", || {
        crate::stages::stage3::prove::prove_committed_boundary(
            Stage3ProverInput::new(
                Stage3ProverConfig::new(log_t),
                &checked,
                &stage1_verifier_output,
                &stage2_verifier_output,
                witness,
            ),
            backend,
            &mut transcript,
            vc_setup,
        )
    })?;
    let stage3_verifier_output = stage3.verifier_output.clone();
    assembly.record_stage3_committed(stage3)?;

    let ram_val_check_init = timed_stage("stage4_init", || {
        stage4_ram_val_check_initial_evaluation(
            preprocessing,
            &checked,
            &stage2_verifier_output,
            log_k,
        )
    })?;
    let stage4 = timed_stage("stage4", || {
        crate::stages::stage4::prove::prove_committed_boundary(
            Stage4ProverInput::new(
                Stage4ProverConfig::new(log_t, log_k, proof_shape.rw_config),
                &checked,
                &stage2_verifier_output,
                &stage3_verifier_output,
                ram_val_check_init,
                witness,
                #[cfg(feature = "field-inline")]
                field_inline_witness,
            ),
            backend,
            &mut transcript,
            vc_setup,
        )
    })?;
    let stage4_verifier_output = stage4.verifier_output.clone();
    assembly.record_stage4_committed(stage4)?;

    let formula_dimensions = proof_shape_formula_dimensions(preprocessing, proof_shape)?;
    let stage5 = timed_stage("stage5", || {
        crate::stages::stage5::prove::prove_committed_boundary(
            Stage5ProverInput::new(
                Stage5ProverConfig::new(log_t, log_k, formula_dimensions.instruction_read_raf),
                &checked,
                &stage2_verifier_output,
                &stage4_verifier_output,
                witness,
                #[cfg(feature = "field-inline")]
                field_inline_witness,
            ),
            backend,
            &mut transcript,
            vc_setup,
        )
    })?;
    let stage5_verifier_output = stage5.verifier_output.clone();
    assembly.record_stage5_committed(stage5)?;

    let stage6_config = stage6_config(preprocessing, public_io, proof_shape)?;
    let stage6 = timed_stage("stage6", || {
        crate::stages::stage6::prove::prove_committed_boundary(
            Stage6ProverInput::new(
                &stage6_config,
                &checked,
                &stage1_verifier_output,
                &stage2_verifier_output,
                &stage3_verifier_output,
                &stage4_verifier_output,
                &stage5_verifier_output,
                witness,
                #[cfg(feature = "field-inline")]
                field_inline_witness,
            ),
            backend,
            &mut transcript,
            vc_setup,
        )
    })?;
    let stage6_verifier_output = stage6.verifier_output.clone();
    assembly.record_stage6_committed(stage6)?;

    let stage7_config = stage7_config(preprocessing, public_io, proof_shape)?;
    let stage7 = timed_stage("stage7", || {
        crate::stages::stage7::prove::prove_committed_boundary(
            Stage7ProverInput::new(
                &stage7_config,
                &checked,
                &stage4_verifier_output,
                &stage6_verifier_output,
                witness,
            ),
            backend,
            &mut transcript,
            vc_setup,
        )
    })?;
    let stage7_verifier_output = stage7.verifier_output.clone();
    assembly.record_stage7_committed(stage7)?;

    timed_stage("assemble_zk_stage_payloads", || {
        assembly.assemble_zk_stage_payloads()
    })?;
    let stage8_config = stage8_config(preprocessing, public_io, proof_shape)?;
    let (commitments, hints) = assembly.stage8_clear_opening_inputs(stage8_config.layout)?;
    let stage8 = timed_stage("stage8", || {
        crate::stages::stage8::prove::prove_stage8_zk::<
            PCS::Field,
            PCS,
            W,
            Blake2bTranscript<PCS::Field>,
        >(
            &stage8_config,
            &stage6_verifier_output,
            &stage7_verifier_output,
            witness,
            #[cfg(feature = "field-inline")]
            field_inline_witness,
            commitments.as_slice(),
            hints,
            &preprocessing.pcs_setup,
            &mut transcript,
        )
    })?;
    assembly.record_stage8_zk(stage8)?;
    let blindfold = timed_stage("blindfold_build_protocol", || {
        assembly.build_blindfold_protocol(&preprocessing.verifier)
    })?;
    let mut rng = rand_core::OsRng;
    let blindfold_witness = timed_stage("blindfold_witness", || {
        assembly.assemble_blindfold_witness(&blindfold, &mut rng)
    })?;

    transcript.append(&Label(b"BlindFold"));
    let mut row_committer = BackendBlindFoldRowCommitter { backend };
    timed_stage("blindfold_prove", || {
        jolt_blindfold::prove_with_row_committer::<PCS::Field, VC, _, _, _>(
            vc_setup,
            &blindfold.protocol,
            &mut transcript,
            jolt_blindfold::BlindFoldWitness {
                rows: &blindfold_witness.rows,
                blindings: &blindfold_witness.blindings,
                eval_outputs: &blindfold_witness.eval_outputs,
                eval_blindings: &blindfold_witness.eval_blindings,
            },
            &mut rng,
            &mut row_committer,
        )
    })
    .map_err(|error| ProverError::InvalidStageRequest {
        reason: format!("BlindFold proof generation failed: {error}"),
    })
}

#[cfg(feature = "zk")]
struct BackendBlindFoldRowCommitter<'a, B> {
    backend: &'a mut B,
}

#[cfg(feature = "zk")]
impl<F, VC, B> jolt_blindfold::BlindFoldRowCommitter<F, VC> for BackendBlindFoldRowCommitter<'_, B>
where
    F: Field,
    VC: VectorCommitment<Field = F>,
    B: BlindFoldBackend<F>,
{
    fn commit_rows(
        &mut self,
        setup: &VC::Setup,
        rows: &[Vec<F>],
        blindings: &[F],
        name: &'static str,
    ) -> Result<Vec<VC::Output>, jolt_blindfold::ProverError<F>> {
        self.backend
            .commit_blindfold_rows::<VC>(
                BlindFoldRowCommitmentRequest::new(name, rows, blindings),
                setup,
            )
            .map(|result| result.commitments)
            .map_err(|error| jolt_blindfold::ProverError::RowCommitmentBackend {
                name,
                reason: error.to_string(),
            })
    }

    fn compute_error_rows(
        &mut self,
        r1cs: &jolt_r1cs::ConstraintMatrices<F>,
        u: F,
        witness: &[F],
        row_count: usize,
        row_len: usize,
        name: &'static str,
    ) -> Result<Vec<Vec<F>>, jolt_blindfold::ProverError<F>> {
        self.backend
            .compute_blindfold_error_rows(BlindFoldErrorRowsRequest::new(
                name, r1cs, u, witness, row_count, row_len,
            ))
            .map(|result| result.rows)
            .map_err(|error| jolt_blindfold::ProverError::BackendKernel {
                name,
                reason: error.to_string(),
            })
    }

    fn compute_cross_term_error_rows(
        &mut self,
        r1cs: &jolt_r1cs::ConstraintMatrices<F>,
        real_u: F,
        real_witness: &[F],
        random_u: F,
        random_witness: &[F],
        row_count: usize,
        row_len: usize,
        name: &'static str,
    ) -> Result<Vec<Vec<F>>, jolt_blindfold::ProverError<F>> {
        self.backend
            .compute_blindfold_cross_term_error_rows(BlindFoldCrossTermErrorRowsRequest::new(
                name,
                r1cs,
                real_u,
                real_witness,
                random_u,
                random_witness,
                row_count,
                row_len,
            ))
            .map(|result| result.rows)
            .map_err(|error| jolt_blindfold::ProverError::BackendKernel {
                name,
                reason: error.to_string(),
            })
    }

    fn fold_rows(
        &mut self,
        real: &[Vec<F>],
        random: &[Vec<F>],
        challenge: F,
        name: &'static str,
    ) -> Result<Vec<Vec<F>>, jolt_blindfold::ProverError<F>> {
        self.backend
            .fold_blindfold_rows(BlindFoldFoldRowsRequest::new(name, real, random, challenge))
            .map(|result| result.rows)
            .map_err(|error| jolt_blindfold::ProverError::BackendKernel {
                name,
                reason: error.to_string(),
            })
    }

    fn fold_scalars(
        &mut self,
        real: &[F],
        random: &[F],
        challenge: F,
        name: &'static str,
    ) -> Result<Vec<F>, jolt_blindfold::ProverError<F>> {
        self.backend
            .fold_blindfold_scalars(BlindFoldFoldScalarsRequest::new(
                name, real, random, challenge,
            ))
            .map(|result| result.scalars)
            .map_err(|error| jolt_blindfold::ProverError::BackendKernel {
                name,
                reason: error.to_string(),
            })
    }

    fn fold_error_rows(
        &mut self,
        real: &[Vec<F>],
        cross: &[Vec<F>],
        random: &[Vec<F>],
        challenge: F,
        name: &'static str,
    ) -> Result<Vec<Vec<F>>, jolt_blindfold::ProverError<F>> {
        self.backend
            .fold_blindfold_error_rows(BlindFoldFoldErrorRowsRequest::new(
                name, real, cross, random, challenge,
            ))
            .map(|result| result.rows)
            .map_err(|error| jolt_blindfold::ProverError::BackendKernel {
                name,
                reason: error.to_string(),
            })
    }

    fn fold_error_scalars(
        &mut self,
        real: &[F],
        cross: &[F],
        random: &[F],
        challenge: F,
        name: &'static str,
    ) -> Result<Vec<F>, jolt_blindfold::ProverError<F>> {
        self.backend
            .fold_blindfold_error_scalars(BlindFoldFoldErrorScalarsRequest::new(
                name, real, cross, random, challenge,
            ))
            .map(|result| result.scalars)
            .map_err(|error| jolt_blindfold::ProverError::BackendKernel {
                name,
                reason: error.to_string(),
            })
    }

    fn open_rows(
        &mut self,
        setup: &VC::Setup,
        rows: &[Vec<F>],
        blindings: &[F],
        row_point: &[F],
        entry_point: &[F],
        name: &'static str,
    ) -> Result<(jolt_crypto::VectorCommitmentOpening<F>, F), jolt_blindfold::ProverError<F>> {
        self.backend
            .open_blindfold_rows::<VC>(
                BlindFoldRowOpeningRequest::new(name, rows, blindings, row_point, entry_point),
                setup,
            )
            .map(|result| (result.opening, result.evaluation))
            .map_err(|error| jolt_blindfold::ProverError::BackendKernel {
                name,
                reason: error.to_string(),
            })
    }
}

fn prove_clear_stages<PCS, VC, B, W, FI>(
    preprocessing: &JoltProverPreprocessing<PCS, VC>,
    public_io: &JoltDevice,
    witness: &W,
    field_inline_witness: &FI,
    config: &ProverConfig,
    backend: &mut B,
    assembly: &mut ProofAssembly<PCS, VC>,
) -> Result<(), ProverError>
where
    PCS: CommitmentScheme + AdditivelyHomomorphic,
    PCS::Output: AppendToTranscript + HomomorphicCommitment<PCS::Field>,
    <PCS::Field as jolt_field::WithAccumulator>::Accumulator:
        jolt_field::RingAccumulator<Element = PCS::Field>,
    VC: VectorCommitment<Field = PCS::Field>,
    B: ClearProverBackend<PCS::Field>,
    W: JoltVmSpartanOuterRows
        + JoltVmProductUniskipRows
        + JoltVmStage2Rows
        + JoltVmStage3ShiftRows
        + JoltVmStage3InstructionRegisterRows
        + JoltVmRegisterReadWriteRows
        + JoltVmStage5InstructionReadRafRows
        + JoltVmStage6Rows
        + WitnessProvider<PCS::Field, JoltVmNamespace>
        + Sync,
    FI: FieldInlineProverWitness<PCS::Field>,
{
    #[cfg(not(feature = "field-inline"))]
    let _ = field_inline_witness;

    let proof_shape = required_proof_shape(config)?;
    let checked = clear_checked_inputs(preprocessing, public_io, config)?;
    let mut transcript = Blake2bTranscript::<PCS::Field>::new(b"Jolt");
    assembly.absorb_stage0(&checked, &mut transcript)?;
    let log_t = proof_shape_log_t(proof_shape);
    let stage1 = timed_stage("stage1", || {
        crate::stages::stage1::prove::prove(
            Stage1ProverInput::new(
                Stage1ProverConfig::new(log_t),
                witness,
                #[cfg(feature = "field-inline")]
                field_inline_witness,
            ),
            backend,
            &mut transcript,
        )
    })?;
    assembly.record_stage1_clear(stage1)?;

    let log_k = proof_shape_log_k(proof_shape);
    let stage1 = assembly.stage1_clear_output()?;
    let stage2 = timed_stage("stage2", || {
        crate::stages::stage2::prove::prove(
            Stage2ProverInput::new(
                Stage2BatchProverConfig::new(log_t, log_k, proof_shape.rw_config),
                &checked,
                stage1,
                witness,
                #[cfg(feature = "field-inline")]
                field_inline_witness,
            ),
            backend,
            &mut transcript,
        )
    })?;
    assembly.record_stage2_clear(stage2)?;

    let stage1 = assembly.stage1_clear_output()?;
    let stage2 = assembly.stage2_clear_output()?;
    let stage3 = timed_stage("stage3", || {
        crate::stages::stage3::prove::prove(
            Stage3ProverInput::new(
                Stage3ProverConfig::new(log_t),
                &checked,
                stage1,
                stage2,
                witness,
            ),
            backend,
            &mut transcript,
        )
    })?;
    assembly.record_stage3_clear(stage3)?;

    let stage2 = assembly.stage2_clear_output()?;
    let stage3 = assembly.stage3_clear_output()?;
    let ram_val_check_init = timed_stage("stage4_init", || {
        stage4_ram_val_check_initial_evaluation(preprocessing, &checked, stage2, log_k)
    })?;
    let stage4 = timed_stage("stage4", || {
        crate::stages::stage4::prove::prove(
            Stage4ProverInput::new(
                Stage4ProverConfig::new(log_t, log_k, proof_shape.rw_config),
                &checked,
                stage2,
                stage3,
                ram_val_check_init,
                witness,
                #[cfg(feature = "field-inline")]
                field_inline_witness,
            ),
            backend,
            &mut transcript,
        )
    })?;
    assembly.record_stage4_clear(stage4)?;

    let formula_dimensions = proof_shape_formula_dimensions(preprocessing, proof_shape)?;
    let stage2 = assembly.stage2_clear_output()?;
    let stage4 = assembly.stage4_clear_output()?;
    let stage5 = timed_stage("stage5", || {
        crate::stages::stage5::prove::prove(
            Stage5ProverInput::new(
                Stage5ProverConfig::new(log_t, log_k, formula_dimensions.instruction_read_raf),
                &checked,
                stage2,
                stage4,
                witness,
                #[cfg(feature = "field-inline")]
                field_inline_witness,
            ),
            backend,
            &mut transcript,
        )
    })?;
    assembly.record_stage5_clear(stage5)?;

    let stage6_config = stage6_config(preprocessing, public_io, proof_shape)?;
    let stage1 = assembly.stage1_clear_output()?;
    let stage2 = assembly.stage2_clear_output()?;
    let stage3 = assembly.stage3_clear_output()?;
    let stage4 = assembly.stage4_clear_output()?;
    let stage5 = assembly.stage5_clear_output()?;
    let stage6 = timed_stage("stage6", || {
        crate::stages::stage6::prove::prove(
            Stage6ProverInput::new(
                &stage6_config,
                &checked,
                stage1,
                stage2,
                stage3,
                stage4,
                stage5,
                witness,
                #[cfg(feature = "field-inline")]
                field_inline_witness,
            ),
            backend,
            &mut transcript,
        )
    })?;
    assembly.record_stage6_clear(stage6)?;

    let stage7_config = stage7_config(preprocessing, public_io, proof_shape)?;
    let stage4 = assembly.stage4_clear_output()?;
    let stage6 = assembly.stage6_clear_output()?;
    let stage7 = timed_stage("stage7", || {
        crate::stages::stage7::prove::prove(
            Stage7ProverInput::new(&stage7_config, &checked, stage4, stage6, witness),
            backend,
            &mut transcript,
        )
    })?;
    assembly.record_stage7_clear(stage7)?;

    if !config.features.zk {
        timed_stage("assemble_clear_stage_payloads", || {
            assembly.assemble_clear_stage_payloads()
        })?;
        let stage8_config = stage8_config(preprocessing, public_io, proof_shape)?;
        let (commitments, hints) = assembly.stage8_clear_opening_inputs(stage8_config.layout)?;
        let stage6 = assembly.stage6_clear_output()?;
        let stage7 = assembly.stage7_clear_output()?;
        let stage8 = timed_stage("stage8", || {
            crate::stages::stage8::prove::prove_stage8::<
                PCS::Field,
                PCS,
                W,
                Blake2bTranscript<PCS::Field>,
            >(
                &stage8_config,
                stage6,
                stage7,
                witness,
                #[cfg(feature = "field-inline")]
                field_inline_witness,
                commitments.as_slice(),
                hints,
                &preprocessing.pcs_setup,
                &mut transcript,
            )
        })?;
        assembly.record_stage8_clear(stage8)?;
    }

    Ok(())
}

fn stage8_config<PCS, VC>(
    preprocessing: &JoltProverPreprocessing<PCS, VC>,
    public_io: &JoltDevice,
    proof_shape: ProverProofShape,
) -> Result<Stage8ProverConfig, ProverError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    let log_t = proof_shape_log_t(proof_shape);
    let formula_dimensions = proof_shape_formula_dimensions(preprocessing, proof_shape)?;
    let committed_chunk_bits = proof_shape.one_hot_config.committed_chunk_bits();
    let trusted_advice_layout = (!public_io.trusted_advice.is_empty()).then(|| {
        AdviceClaimReductionLayout::balanced(
            proof_shape.trace_polynomial_order,
            log_t,
            committed_chunk_bits,
            public_io.memory_layout.max_trusted_advice_size as usize,
        )
    });
    let untrusted_advice_layout = (!public_io.untrusted_advice.is_empty()).then(|| {
        AdviceClaimReductionLayout::balanced(
            proof_shape.trace_polynomial_order,
            log_t,
            committed_chunk_bits,
            public_io.memory_layout.max_untrusted_advice_size as usize,
        )
    });

    Ok(Stage8ProverConfig::new(
        log_t,
        committed_chunk_bits,
        formula_dimensions.ra_layout,
        proof_shape.trace_polynomial_order,
        trusted_advice_layout,
        untrusted_advice_layout,
    ))
}

fn stage7_config<PCS, VC>(
    preprocessing: &JoltProverPreprocessing<PCS, VC>,
    public_io: &JoltDevice,
    proof_shape: ProverProofShape,
) -> Result<Stage7ProverConfig, ProverError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    let log_t = proof_shape_log_t(proof_shape);
    let formula_dimensions = proof_shape_formula_dimensions(preprocessing, proof_shape)?;
    let committed_chunk_bits = proof_shape.one_hot_config.committed_chunk_bits();
    let hamming_dimensions =
        jolt_claims::protocols::jolt::formulas::claim_reductions::hamming_weight::HammingWeightClaimReductionDimensions::new(
            formula_dimensions.ra_layout,
            committed_chunk_bits,
        );
    let trusted_advice_layout = (!public_io.trusted_advice.is_empty()).then(|| {
        AdviceClaimReductionLayout::balanced(
            proof_shape.trace_polynomial_order,
            log_t,
            committed_chunk_bits,
            public_io.memory_layout.max_trusted_advice_size as usize,
        )
    });
    let untrusted_advice_layout = (!public_io.untrusted_advice.is_empty()).then(|| {
        AdviceClaimReductionLayout::balanced(
            proof_shape.trace_polynomial_order,
            log_t,
            committed_chunk_bits,
            public_io.memory_layout.max_untrusted_advice_size as usize,
        )
    });

    Ok(Stage7ProverConfig::new(
        log_t,
        hamming_dimensions,
        trusted_advice_layout,
        untrusted_advice_layout,
    ))
}

fn stage6_config<PCS, VC>(
    preprocessing: &JoltProverPreprocessing<PCS, VC>,
    public_io: &JoltDevice,
    proof_shape: ProverProofShape,
) -> Result<Stage6ProverConfig, ProverError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    let log_t = proof_shape_log_t(proof_shape);
    let log_k = proof_shape_log_k(proof_shape);
    let formula_dimensions = proof_shape_formula_dimensions(preprocessing, proof_shape)?;
    let committed_chunk_bits = proof_shape.one_hot_config.committed_chunk_bits();
    let trusted_advice_layout = (!public_io.trusted_advice.is_empty()).then(|| {
        AdviceClaimReductionLayout::balanced(
            proof_shape.trace_polynomial_order,
            log_t,
            committed_chunk_bits,
            public_io.memory_layout.max_trusted_advice_size as usize,
        )
    });
    let untrusted_advice_layout = (!public_io.untrusted_advice.is_empty()).then(|| {
        AdviceClaimReductionLayout::balanced(
            proof_shape.trace_polynomial_order,
            log_t,
            committed_chunk_bits,
            public_io.memory_layout.max_untrusted_advice_size as usize,
        )
    });
    let entry_bytecode_index = preprocessing
        .verifier
        .program
        .bytecode
        .entry_bytecode_index()
        .ok_or_else(|| ProverError::InvalidProverConfig {
            reason: "entry address was not found in bytecode preprocessing".to_owned(),
        })?;

    Ok(Stage6ProverConfig::new(
        log_t,
        log_k,
        committed_chunk_bits,
        formula_dimensions.bytecode_read_raf,
        jolt_claims::protocols::jolt::formulas::booleanity::BooleanityDimensions::new(
            formula_dimensions.ra_layout,
            log_t,
            committed_chunk_bits,
        ),
        formula_dimensions.ram_ra_virtualization,
        formula_dimensions.instruction_ra_virtualization,
        trusted_advice_layout,
        untrusted_advice_layout,
    )
    .with_bytecode_context(
        preprocessing.verifier.program.bytecode.bytecode.clone(),
        entry_bytecode_index,
    ))
}

fn stage4_ram_val_check_initial_evaluation<PCS, VC>(
    preprocessing: &JoltProverPreprocessing<PCS, VC>,
    checked: &jolt_verifier::CheckedInputs,
    stage2: &jolt_verifier::stages::stage2::outputs::Stage2ClearOutput<PCS::Field>,
    log_k: usize,
) -> Result<Stage4RamValCheckInitialEvaluation<PCS::Field>, ProverError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    let opening_point = &stage2.batch.ram_read_write.opening_point;
    if opening_point.len() < log_k {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 4 RAM read-write opening point has {} variables, fewer than log_k {log_k}",
                opening_point.len()
            ),
        });
    }
    let (r_address, _) = opening_point.split_at(log_k);
    let public_initial_ram =
        PublicInitialRam::new(&preprocessing.verifier.program.ram, &checked.public_io).map_err(
            |error| ProverError::InvalidStageRequest {
                reason: format!("Stage 4 public initial RAM construction failed: {error}"),
            },
        )?;
    for segment in &public_initial_ram.segments {
        let end = segment.start_index + segment.words.len();
        if end > checked.ram_K {
            return Err(ProverError::InvalidStageRequest {
                reason: format!(
                    "Stage 4 public initial RAM segment [{}, {}) exceeds RAM domain {}",
                    segment.start_index, end, checked.ram_K
                ),
            });
        }
    }

    let public_eval = sparse_segments_mle_msb(
        public_initial_ram
            .segments
            .iter()
            .map(|segment| (segment.start_index, segment.words.as_slice())),
        r_address,
    );
    let mut full_eval = public_eval;
    let mut advice_contributions = Vec::new();
    collect_stage4_advice_contribution(
        JoltAdviceKind::Untrusted,
        !checked.public_io.untrusted_advice.is_empty(),
        &checked.public_io.untrusted_advice,
        checked,
        r_address,
        &mut full_eval,
        &mut advice_contributions,
    )?;
    collect_stage4_advice_contribution(
        JoltAdviceKind::Trusted,
        checked.trusted_advice_commitment_present,
        &checked.public_io.trusted_advice,
        checked,
        r_address,
        &mut full_eval,
        &mut advice_contributions,
    )?;

    Ok(Stage4RamValCheckInitialEvaluation {
        public_eval,
        advice_contributions,
        full_eval,
    })
}

fn collect_stage4_advice_contribution<F: Field>(
    kind: JoltAdviceKind,
    present: bool,
    bytes: &[u8],
    checked: &jolt_verifier::CheckedInputs,
    r_address: &[F],
    full_eval: &mut F,
    contributions: &mut Vec<Stage4RamValCheckAdviceContribution<F>>,
) -> Result<(), ProverError> {
    if !present {
        return Ok(());
    }

    let layout = &checked.public_io.memory_layout;
    let (start_address, max_size) = match kind {
        JoltAdviceKind::Trusted => (layout.trusted_advice_start, layout.max_trusted_advice_size),
        JoltAdviceKind::Untrusted => (
            layout.untrusted_advice_start,
            layout.max_untrusted_advice_size,
        ),
    };
    if max_size == 0 {
        return Err(ProverError::InvalidStageRequest {
            reason: format!("Stage 4 {kind:?} advice is present but configured size is zero"),
        });
    }
    if bytes.len() > max_size as usize {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 4 {kind:?} advice has {} bytes, exceeding configured max {max_size}",
                bytes.len()
            ),
        });
    }

    let start_index = layout
        .remapped_word_address(start_address)
        .map_err(|error| ProverError::InvalidStageRequest {
            reason: format!("Stage 4 {kind:?} advice start address is invalid: {error}"),
        })? as usize;
    let advice_num_vars = ((max_size as usize) / 8).next_power_of_two().ilog2() as usize;
    if advice_num_vars > r_address.len() {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 4 {kind:?} advice point needs {advice_num_vars} variables but RAM address has {}",
                r_address.len()
            ),
        });
    }
    let selector =
        block_selector_mle_msb(start_index, advice_num_vars, r_address).map_err(|error| {
            ProverError::InvalidStageRequest {
                reason: format!("Stage 4 {kind:?} advice selector failed: {error}"),
            }
        })?;
    let opening_point = r_address[r_address.len() - advice_num_vars..].to_vec();
    let words = advice_words_le(bytes);
    let opening_claim = sparse_segments_mle_msb([(0, words.as_slice())], &opening_point);
    *full_eval += selector * opening_claim;
    contributions.push(Stage4RamValCheckAdviceContribution {
        kind,
        selector,
        opening_claim,
        opening_point,
    });
    Ok(())
}

fn advice_words_le(bytes: &[u8]) -> Vec<u64> {
    bytes
        .chunks(8)
        .map(|chunk| {
            let mut word = [0_u8; 8];
            word[..chunk.len()].copy_from_slice(chunk);
            u64::from_le_bytes(word)
        })
        .collect()
}

fn clear_checked_inputs<PCS, VC>(
    preprocessing: &JoltProverPreprocessing<PCS, VC>,
    public_io: &JoltDevice,
    config: &ProverConfig,
) -> Result<jolt_verifier::CheckedInputs, ProverError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    let proof_shape = required_proof_shape(config)?;
    if public_io.memory_layout != preprocessing.verifier.program.memory_layout {
        return Err(ProverError::InvalidProverConfig {
            reason: "public I/O memory layout does not match preprocessing".to_owned(),
        });
    }
    let max_input_size = preprocessing.verifier.program.memory_layout.max_input_size as usize;
    if public_io.inputs.len() > max_input_size {
        return Err(ProverError::InvalidProverConfig {
            reason: format!(
                "public input is too large: got {}, max {max_input_size}",
                public_io.inputs.len()
            ),
        });
    }
    let max_output_size = preprocessing.verifier.program.memory_layout.max_output_size as usize;
    if public_io.outputs.len() > max_output_size {
        return Err(ProverError::InvalidProverConfig {
            reason: format!(
                "public output is too large: got {}, max {max_output_size}",
                public_io.outputs.len()
            ),
        });
    }
    if proof_shape.trace_length > preprocessing.verifier.program.max_padded_trace_length {
        return Err(ProverError::InvalidProverConfig {
            reason: format!(
                "proof trace_length {} exceeds preprocessing max {}",
                proof_shape.trace_length, preprocessing.verifier.program.max_padded_trace_length
            ),
        });
    }

    let mut normalized_public_io = public_io.clone();
    normalized_public_io.outputs.truncate(
        normalized_public_io
            .outputs
            .iter()
            .rposition(|&byte| byte != 0)
            .map_or(0, |position| position + 1),
    );

    #[cfg(feature = "field-inline")]
    let field_inline_bytecode_transcript = {
        let field_inline_bytecode = preprocessing
            .verifier
            .field_inline_bytecode
            .as_deref()
            .ok_or_else(|| ProverError::InvalidProverConfig {
                reason: "field-inline bytecode metadata is missing".to_owned(),
            })?;
        field_bytecode::validate_bytecode_rows(
            field_inline_bytecode,
            preprocessing.verifier.program.bytecode.code_size,
            config.protocol.field_inline.field_register_log_k,
        )
        .map_err(|error| ProverError::InvalidProverConfig {
            reason: format!("invalid field-inline bytecode metadata: {error}"),
        })?;
        field_bytecode::bytecode_transcript_bytes(
            field_inline_bytecode,
            config.protocol.field_inline.field_register_log_k,
        )
    };

    Ok(jolt_verifier::CheckedInputs {
        public_io: normalized_public_io,
        zk: false,
        trace_length: proof_shape.trace_length,
        ram_K: proof_shape.ram_k,
        entry_address: preprocessing.verifier.program.bytecode.entry_address,
        preprocessing_digest: preprocessing.verifier.preprocessing_digest,
        trusted_advice_commitment_present: !public_io.trusted_advice.is_empty(),
        vc_capacity: None,
        #[cfg(feature = "field-inline")]
        field_inline_bytecode_transcript,
    })
}

#[cfg(feature = "zk")]
fn zk_checked_inputs<PCS, VC>(
    preprocessing: &JoltProverPreprocessing<PCS, VC>,
    public_io: &JoltDevice,
    config: &ProverConfig,
) -> Result<jolt_verifier::CheckedInputs, ProverError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    let mut checked = clear_checked_inputs(preprocessing, public_io, config)?;
    let vc_setup = preprocessing.verifier.vc_setup.as_ref().ok_or_else(|| {
        ProverError::InvalidProverConfig {
            reason: "ZK checked inputs require vector-commitment setup".to_owned(),
        }
    })?;
    let vc_capacity = VC::capacity(vc_setup);
    if vc_capacity < MAX_BLINDFOLD_GENERATORS {
        return Err(ProverError::InvalidProverConfig {
            reason: format!(
                "vector-commitment setup capacity {vc_capacity} is below required BlindFold capacity {MAX_BLINDFOLD_GENERATORS}"
            ),
        });
    }
    checked.zk = true;
    checked.vc_capacity = Some(vc_capacity);
    Ok(checked)
}

fn prove_stage0<PCS, VC, B, W, FI>(
    preprocessing: &JoltProverPreprocessing<PCS, VC>,
    public_io: &JoltDevice,
    witness: &W,
    field_inline_witness: &FI,
    config: &ProverConfig,
    backend: &mut B,
) -> Result<CommitmentStageOutput<PCS>, ProverError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
    B: stage0::CommitmentStageBackend<PCS::Field, PCS>,
    W: CommittedWitnessProvider<PCS::Field, JoltVmNamespace> + Sync,
    FI: FieldInlineProverWitness<PCS::Field>,
{
    #[cfg(not(feature = "field-inline"))]
    let _ = field_inline_witness;

    let stage0_config = stage0_config(preprocessing, public_io, config)?;
    stage0::prove::<PCS::Field, _, _, PCS>(
        CommitmentStageInput::new(
            witness,
            &preprocessing.pcs_setup,
            stage0_config,
            config.protocol.clone(),
            #[cfg(feature = "field-inline")]
            field_inline_witness,
        ),
        backend,
    )
}

fn required_proof_shape(config: &ProverConfig) -> Result<ProverProofShape, ProverError> {
    config
        .proof_shape
        .ok_or_else(|| ProverError::InvalidProverConfig {
            reason: "proof_shape is required before proof assembly".to_owned(),
        })
}

fn stage0_config<PCS, VC>(
    preprocessing: &JoltProverPreprocessing<PCS, VC>,
    public_io: &JoltDevice,
    config: &ProverConfig,
) -> Result<CommitmentStageConfig, ProverError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    let proof_shape = required_proof_shape(config)?;
    let log_t = proof_shape_log_t(proof_shape);
    let dimensions = proof_shape_formula_dimensions(preprocessing, proof_shape)?;

    Ok(CommitmentStageConfig::new(
        dimensions.ra_layout,
        !public_io.trusted_advice.is_empty(),
        !public_io.untrusted_advice.is_empty(),
    )
    .with_final_opening_trace_embedding(
        log_t,
        proof_shape.one_hot_config.committed_chunk_bits(),
        proof_shape.trace_polynomial_order,
    ))
}

fn proof_shape_log_t(proof_shape: ProverProofShape) -> usize {
    proof_shape.trace_length.trailing_zeros() as usize
}

fn proof_shape_log_k(proof_shape: ProverProofShape) -> usize {
    proof_shape.ram_k.trailing_zeros() as usize
}

fn proof_shape_formula_dimensions<PCS, VC>(
    preprocessing: &JoltProverPreprocessing<PCS, VC>,
    proof_shape: ProverProofShape,
) -> Result<JoltFormulaDimensions, ProverError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    let log_t = proof_shape_log_t(proof_shape);
    JoltFormulaDimensions::try_from(proof_shape.one_hot_config.dimensions(
        log_t,
        RV64_LOOKUP_ADDRESS_BITS,
        preprocessing.verifier.program.bytecode.code_size,
        proof_shape.ram_k,
    ))
    .map_err(|error| ProverError::InvalidProverConfig {
        reason: format!("invalid proof shape formula dimensions: {error}"),
    })
}
