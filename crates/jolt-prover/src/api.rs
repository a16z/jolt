use common::jolt_device::JoltDevice;
#[cfg(feature = "zk")]
use jolt_backends::BlindFoldBackend;
use jolt_backends::{
    RamReadWriteSumcheckBackend, Stage3SpartanSumcheckBackend, Stage4ReadWriteSumcheckBackend,
    Stage5ValueEvaluationSumcheckBackend, Stage6RegularBatchSumcheckBackend, SumcheckBackend,
};
use jolt_crypto::{HomomorphicCommitment, VectorCommitment};
use jolt_field::Field;
#[cfg(feature = "zk")]
use jolt_openings::ZkOpeningScheme;
use jolt_openings::{AdditivelyHomomorphic, CommitmentScheme};
use jolt_transcript::AppendToTranscript;
use jolt_verifier::JoltProof;
use jolt_witness::{
    protocols::jolt_vm::{
        JoltVmNamespace, JoltVmRegisterReadWriteRows, JoltVmSpartanOuterRows, JoltVmStage2Rows,
        JoltVmStage3InstructionRegisterRows, JoltVmStage3ShiftRows,
        JoltVmStage5InstructionReadRafRows, JoltVmStage6Rows,
    },
    CommittedWitnessProvider, WitnessProvider,
};

use crate::stages::stage0;
use crate::{JoltProverPreprocessing, ProverConfig, ProverError};

#[doc(hidden)]
pub trait JoltVmProverWitness<F>:
    CommittedWitnessProvider<F, JoltVmNamespace>
    + WitnessProvider<F, JoltVmNamespace>
    + jolt_witness::RaFamilyCycleIndexSource<F, JoltVmNamespace>
    + JoltVmSpartanOuterRows
    + JoltVmStage2Rows
    + JoltVmStage3ShiftRows
    + JoltVmStage3InstructionRegisterRows
    + JoltVmRegisterReadWriteRows
    + JoltVmStage5InstructionReadRafRows
    + JoltVmStage6Rows
    + Sync
where
    F: Field,
{
}

impl<F, T> JoltVmProverWitness<F> for T
where
    F: Field,
    T: CommittedWitnessProvider<F, JoltVmNamespace>
        + WitnessProvider<F, JoltVmNamespace>
        + jolt_witness::RaFamilyCycleIndexSource<F, JoltVmNamespace>
        + JoltVmSpartanOuterRows
        + JoltVmStage2Rows
        + JoltVmStage3ShiftRows
        + JoltVmStage3InstructionRegisterRows
        + JoltVmRegisterReadWriteRows
        + JoltVmStage5InstructionReadRafRows
        + JoltVmStage6Rows
        + Sync,
{
}

#[doc(hidden)]
pub trait ClearProverBackend<F>:
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProofResult<PCS, VC>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    pub proof: JoltProof<PCS, VC>,
    pub trusted_advice_commitment: Option<PCS::Output>,
}

impl<PCS, VC> ProofResult<PCS, VC>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    pub fn into_proof(self) -> JoltProof<PCS, VC> {
        self.proof
    }
}

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
        + ClearProverBackend<PCS::Field>
        + BlindFoldProverBackend<PCS::Field>,
    W: JoltVmProverWitness<PCS::Field>,
{
    prove_with_components(preprocessing, public_io, witness, config, backend)
        .map(ProofResult::into_proof)
}

pub fn prove_with_components<PCS, VC, B, W>(
    preprocessing: &JoltProverPreprocessing<PCS, VC>,
    public_io: &JoltDevice,
    witness: &W,
    config: ProverConfig,
    backend: &mut B,
) -> Result<ProofResult<PCS, VC>, ProverError>
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
    W: JoltVmProverWitness<PCS::Field>,
{
    crate::prover::prove_with_components_inner(preprocessing, public_io, witness, config, backend)
}
