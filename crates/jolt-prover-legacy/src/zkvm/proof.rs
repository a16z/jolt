#![expect(
    clippy::module_name_repetitions,
    reason = "verifier-facing aliases intentionally include protocol/module names"
)]
//! Native proof construction for prover outputs accepted by `jolt-verifier`.

#[cfg(not(feature = "zk"))]
use crate::zkvm::clear_claims::build_clear_claims;
pub use jolt_verifier::VerifierError;
use jolt_verifier::{
    preprocessing::{
        CommittedProgramPreprocessing, JoltVerifierPreprocessing, ProgramPreprocessing,
    },
    proof::{JoltProof, JoltProofClaims, TracePolynomialOrder},
};

#[cfg(not(feature = "zk"))]
use jolt_claims::protocols::jolt::{
    JoltCommittedPolynomial, JoltOpeningId, JoltPolynomialId, JoltRelationId, JoltVirtualPolynomial,
};
use jolt_claims::protocols::jolt::{JoltOneHotConfig, JoltReadWriteConfig};
use jolt_crypto::{
    Bn254G1, Bn254GT, Commitment as VerifierCommitment, HomomorphicCommitment, Pedersen,
    PedersenSetup, VectorCommitment as VerifierVectorCommitment,
};
use jolt_dory::{DoryCommitment, DoryProof, DoryScheme, DoryVerifierSetup};
use jolt_field::{Field as VerifierFieldTrait, Fr as VerifierFr};
use jolt_openings::CommitmentScheme as VerifierCommitmentScheme;
use jolt_program::preprocess::{JoltProgramPreprocessing, ProgramMetadata};
use jolt_transcript::DuplexSpongeInterface;

use crate::{
    curve::{Bn254Curve, JoltCurve},
    field::JoltField,
    poly::commitment::{
        commitment_scheme::{CommitmentScheme, ZkEvalCommitment},
        dory::{
            ArkGT as ProverDoryCommitment, DoryCommitmentScheme, DoryLayout as ProverDoryLayout,
        },
    },
    zkvm::{
        config::{OneHotConfig as ProverOneHotConfig, ReadWriteConfig as ProverReadWriteConfig},
        preprocessing::{BlindfoldSetup, JoltSharedPreprocessing},
        program::ProgramPreprocessing as ProverProgramPreprocessing,
        proof_parts::JoltProofParts as ProverProofParts,
        prover::JoltProverPreprocessing,
    },
};
#[cfg(not(feature = "zk"))]
use crate::{
    poly::opening_proof as prover_opening,
    zkvm::{instruction as prover_instruction, witness as prover_witness},
};

pub type RV64IMACProof = JoltProof<DoryScheme, Pedersen<Bn254G1>>;

pub trait ProofField: JoltField {
    type VerifierField: VerifierFieldTrait;

    fn into_verifier_field(self) -> Self::VerifierField;
}

impl ProofField for crate::ark_bn254::Fr {
    type VerifierField = VerifierFr;

    fn into_verifier_field(self) -> Self::VerifierField {
        self.into()
    }
}

pub trait ProofCommitmentScheme<F>: CommitmentScheme<Field = F>
where
    F: ProofField,
{
    type VerifierPcs: VerifierCommitmentScheme<Field = F::VerifierField>;

    fn commitment_into_verifier(
        commitment: Self::Commitment,
    ) -> <Self::VerifierPcs as VerifierCommitment>::Output;

    fn opening_proof_into_verifier(
        proof: Self::Proof,
    ) -> <Self::VerifierPcs as VerifierCommitmentScheme>::Proof;

    fn verifier_setup_into_verifier(
        setup: Self::VerifierSetup,
    ) -> <Self::VerifierPcs as VerifierCommitmentScheme>::VerifierSetup;
}

impl ProofCommitmentScheme<crate::ark_bn254::Fr> for DoryCommitmentScheme {
    type VerifierPcs = DoryScheme;

    fn commitment_into_verifier(commitment: Self::Commitment) -> DoryCommitment {
        DoryCommitment(prover_dory_commitment_into_verifier(&commitment))
    }

    fn opening_proof_into_verifier(proof: Self::Proof) -> DoryProof {
        DoryProof(proof)
    }

    fn verifier_setup_into_verifier(setup: Self::VerifierSetup) -> DoryVerifierSetup {
        DoryVerifierSetup(setup)
    }
}

pub trait ProofCurve<F>: JoltCurve<F = F> + Sized
where
    F: ProofField,
{
    type VerifierVectorCommitment: VerifierVectorCommitment<
        Field = F::VerifierField,
        Output = Self::VerifierRoundCommitment,
    >;
    type VerifierRoundCommitment: Copy
        + HomomorphicCommitment<F::VerifierField>
        + ark_serialize::CanonicalSerialize
        + serde::Serialize
        + serde::de::DeserializeOwned;

    fn g1_into_verifier(commitment: Self::G1) -> Self::VerifierRoundCommitment;

    fn vc_setup_from_prover_blindfold(
        setup: &crate::poly::commitment::pedersen::PedersenGenerators<Self>,
    ) -> <Self::VerifierVectorCommitment as VerifierVectorCommitment>::Setup;
}

impl ProofCurve<crate::ark_bn254::Fr> for Bn254Curve {
    type VerifierVectorCommitment = Pedersen<Bn254G1>;
    type VerifierRoundCommitment = Bn254G1;

    fn g1_into_verifier(commitment: Self::G1) -> Self::VerifierRoundCommitment {
        commitment.0.into()
    }

    fn vc_setup_from_prover_blindfold(
        setup: &crate::poly::commitment::pedersen::PedersenGenerators<Self>,
    ) -> <Self::VerifierVectorCommitment as VerifierVectorCommitment>::Setup {
        PedersenSetup::new(
            setup
                .message_generators
                .iter()
                .copied()
                .map(Self::g1_into_verifier)
                .collect(),
            Self::g1_into_verifier(setup.blinding_generator),
        )
    }
}

fn verifier_preprocessing_from_parts<F, C, PCS>(
    shared: &JoltSharedPreprocessing<PCS>,
    generators: PCS::VerifierSetup,
    blindfold_setup: Option<&BlindfoldSetup<C>>,
) -> JoltVerifierPreprocessing<
    <PCS as ProofCommitmentScheme<F>>::VerifierPcs,
    <C as ProofCurve<F>>::VerifierVectorCommitment,
>
where
    F: ProofField,
    C: ProofCurve<F>,
    PCS: ProofCommitmentScheme<F>,
{
    JoltVerifierPreprocessing::new(
        verifier_program_from_shared::<F, PCS>(shared),
        shared.digest(),
        PCS::verifier_setup_into_verifier(generators),
        convert_vc_setup::<F, C>(blindfold_setup),
    )
}

fn verifier_program_from_shared<F, PCS>(
    shared: &JoltSharedPreprocessing<PCS>,
) -> ProgramPreprocessing<<PCS as ProofCommitmentScheme<F>>::VerifierPcs>
where
    F: ProofField,
    PCS: ProofCommitmentScheme<F>,
{
    match &shared.program {
        ProverProgramPreprocessing::Full(full) => {
            ProgramPreprocessing::Full(JoltProgramPreprocessing {
                bytecode: full.bytecode.as_ref().clone(),
                ram: full.ram.clone(),
                memory_layout: shared.memory_layout.clone(),
                max_padded_trace_length: shared.max_padded_trace_length,
            })
        }
        ProverProgramPreprocessing::Committed(committed) => {
            ProgramPreprocessing::Committed(CommittedProgramPreprocessing {
                meta: ProgramMetadata {
                    entry_address: committed.meta.entry_address,
                    min_bytecode_address: committed.meta.min_bytecode_address,
                    entry_bytecode_index: committed.meta.entry_bytecode_index,
                    program_image_len_words: committed.meta.program_image_len_words,
                    bytecode_len: committed.meta.bytecode_len,
                },
                memory_layout: shared.memory_layout.clone(),
                max_padded_trace_length: shared.max_padded_trace_length,
                bytecode_chunk_commitments: committed
                    .bytecode_commitments
                    .commitments
                    .iter()
                    .cloned()
                    .map(PCS::commitment_into_verifier)
                    .collect(),
                program_image_commitment: PCS::commitment_into_verifier(
                    committed
                        .program_commitments
                        .program_image_commitment
                        .clone(),
                ),
            })
        }
    }
}

pub fn verifier_preprocessing_from_shared<F, C, PCS>(
    shared: JoltSharedPreprocessing<PCS>,
    generators: PCS::VerifierSetup,
    blindfold_setup: Option<BlindfoldSetup<C>>,
) -> JoltVerifierPreprocessing<
    <PCS as ProofCommitmentScheme<F>>::VerifierPcs,
    <C as ProofCurve<F>>::VerifierVectorCommitment,
>
where
    F: ProofField,
    C: ProofCurve<F>,
    PCS: ProofCommitmentScheme<F>,
{
    verifier_preprocessing_from_parts(&shared, generators, blindfold_setup.as_ref())
}

pub fn verifier_preprocessing_from_prover<F, C, PCS>(
    preprocessing: &JoltProverPreprocessing<F, C, PCS>,
) -> JoltVerifierPreprocessing<
    <PCS as ProofCommitmentScheme<F>>::VerifierPcs,
    <C as ProofCurve<F>>::VerifierVectorCommitment,
>
where
    F: ProofField,
    C: ProofCurve<F>,
    PCS: ProofCommitmentScheme<F> + ZkEvalCommitment<C>,
{
    let generators = PCS::setup_verifier(&preprocessing.generators);
    #[cfg(not(feature = "zk"))]
    let blindfold_setup: Option<BlindfoldSetup<C>> = None;
    #[cfg(feature = "zk")]
    let blindfold_setup = Some(preprocessing.blindfold_setup());

    verifier_preprocessing_from_parts(&preprocessing.shared, generators, blindfold_setup.as_ref())
}

#[cfg(not(feature = "zk"))]
fn convert_vc_setup<F, C>(
    _blindfold_setup: Option<&BlindfoldSetup<C>>,
) -> Option<<C::VerifierVectorCommitment as VerifierVectorCommitment>::Setup>
where
    F: ProofField,
    C: ProofCurve<F>,
{
    None
}

#[cfg(feature = "zk")]
#[expect(
    clippy::expect_used,
    reason = "prover ZK verifier preprocessing must carry the BlindFold setup needed by modular verification"
)]
fn convert_vc_setup<F, C>(
    blindfold_setup: Option<&BlindfoldSetup<C>>,
) -> Option<<C::VerifierVectorCommitment as VerifierVectorCommitment>::Setup>
where
    F: ProofField,
    C: ProofCurve<F>,
{
    Some(C::vc_setup_from_prover_blindfold(
        &blindfold_setup
            .expect("ZK prover preprocessing must carry BlindFold setup")
            .0,
    ))
}

fn convert_read_write_config(config: ProverReadWriteConfig) -> JoltReadWriteConfig {
    JoltReadWriteConfig {
        ram_rw_phase1_num_rounds: config.ram_rw_phase1_num_rounds,
        ram_rw_phase2_num_rounds: config.ram_rw_phase2_num_rounds,
        registers_rw_phase1_num_rounds: config.registers_rw_phase1_num_rounds,
        registers_rw_phase2_num_rounds: config.registers_rw_phase2_num_rounds,
    }
}

fn convert_one_hot_config(config: ProverOneHotConfig) -> JoltOneHotConfig {
    JoltOneHotConfig {
        log_k_chunk: config.log_k_chunk,
        lookups_ra_virtual_log_k_chunk: config.lookups_ra_virtual_log_k_chunk,
    }
}

fn convert_trace_polynomial_order(layout: ProverDoryLayout) -> TracePolynomialOrder {
    match layout {
        ProverDoryLayout::CycleMajor => TracePolynomialOrder::CycleMajor,
        ProverDoryLayout::AddressMajor => TracePolynomialOrder::AddressMajor,
    }
}

#[cfg(not(feature = "zk"))]
#[expect(
    clippy::type_complexity,
    reason = "private converter returns the verifier-native proof with projected backend types"
)]
pub(crate) fn proof_parts_into_verifier<F, C, PCS, H>(
    proof: ProverProofParts<F, C, PCS, H>,
) -> Result<
    JoltProof<
        <PCS as ProofCommitmentScheme<F>>::VerifierPcs,
        <C as ProofCurve<F>>::VerifierVectorCommitment,
    >,
    VerifierError,
>
where
    F: ProofField,
    C: ProofCurve<F>,
    PCS: ProofCommitmentScheme<F>,
    <PCS::VerifierPcs as VerifierCommitment>::Output: ark_serialize::CanonicalSerialize + Clone,
    H: DuplexSpongeInterface,
{
    let one_hot_config = convert_one_hot_config(proof.one_hot_config);

    Ok(JoltProof::new(
        proof.narg,
        PCS::opening_proof_into_verifier(proof.joint_opening_proof),
        JoltProofClaims::Clear(convert_opening_claims(
            proof.opening_claims,
            proof.trace_length,
        )),
        proof.trace_length,
        proof.ram_K,
        convert_read_write_config(proof.rw_config),
        one_hot_config,
        convert_trace_polynomial_order(proof.dory_layout),
    ))
}

#[cfg(feature = "zk")]
#[expect(
    clippy::type_complexity,
    reason = "private converter returns the verifier-native proof with projected backend types"
)]
pub(crate) fn proof_parts_into_verifier<F, C, PCS, H>(
    proof: ProverProofParts<F, C, PCS, H>,
) -> Result<
    JoltProof<
        <PCS as ProofCommitmentScheme<F>>::VerifierPcs,
        <C as ProofCurve<F>>::VerifierVectorCommitment,
    >,
    VerifierError,
>
where
    F: ProofField,
    C: ProofCurve<F>,
    PCS: ProofCommitmentScheme<F>,
    <PCS::VerifierPcs as VerifierCommitment>::Output: ark_serialize::CanonicalSerialize + Clone,
    H: DuplexSpongeInterface,
{
    let one_hot_config = convert_one_hot_config(proof.one_hot_config);

    Ok(JoltProof::new(
        proof.narg,
        PCS::opening_proof_into_verifier(proof.joint_opening_proof),
        JoltProofClaims::Zk,
        proof.trace_length,
        proof.ram_K,
        convert_read_write_config(proof.rw_config),
        one_hot_config,
        convert_trace_polynomial_order(proof.dory_layout),
    ))
}

fn prover_dory_commitment_into_verifier(commitment: &ProverDoryCommitment) -> Bn254GT {
    // `ArkGT` (prover Dory) and `Bn254GT` (modular `jolt-dory`) both wrap the same
    // patched `ark_bn254::Fq12`, so convert through the shared inner element. This
    // is equivalent to the previous `transmute_copy` but layout-independent: a
    // future change to either wrapper becomes a type error here instead of silent
    // memory corruption on the soundness-critical commitment path.
    Bn254GT::from(commitment.0)
}

#[cfg(not(feature = "zk"))]
#[expect(
    clippy::expect_used,
    reason = "standard prover proofs are expected to carry a complete clear-claim payload"
)]
fn convert_opening_claims<F>(
    claims: crate::zkvm::proof_parts::ProverOpeningClaims<F>,
    trace_length: usize,
) -> jolt_verifier::proof::ClearProofClaims<F::VerifierField>
where
    F: ProofField,
{
    build_clear_claims(
        claims.0.into_iter().map(|(id, (_point, claim))| {
            (
                convert_opening_id(id),
                ProofField::into_verifier_field(claim),
            )
        }),
        trace_length,
    )
    .expect("standard prover proof must contain all typed clear opening claims")
}

#[cfg(not(feature = "zk"))]
fn convert_opening_id(id: prover_opening::OpeningId) -> JoltOpeningId {
    match id {
        prover_opening::OpeningId::Polynomial(poly, sumcheck) => {
            JoltOpeningId::polynomial(convert_polynomial_id(poly), convert_sumcheck_id(sumcheck))
        }
        prover_opening::OpeningId::UntrustedAdvice(sumcheck) => {
            JoltOpeningId::untrusted_advice(convert_sumcheck_id(sumcheck))
        }
        prover_opening::OpeningId::TrustedAdvice(sumcheck) => {
            JoltOpeningId::trusted_advice(convert_sumcheck_id(sumcheck))
        }
    }
}

#[cfg(not(feature = "zk"))]
fn convert_polynomial_id(id: prover_opening::PolynomialId) -> JoltPolynomialId {
    match id {
        prover_opening::PolynomialId::Committed(poly) => {
            JoltPolynomialId::Committed(convert_committed_polynomial(poly))
        }
        prover_opening::PolynomialId::Virtual(poly) => {
            JoltPolynomialId::Virtual(convert_virtual_polynomial(poly))
        }
    }
}

#[cfg(not(feature = "zk"))]
fn convert_sumcheck_id(id: prover_opening::SumcheckId) -> JoltRelationId {
    match id {
        prover_opening::SumcheckId::SpartanOuter => JoltRelationId::SpartanOuter,
        prover_opening::SumcheckId::SpartanProductVirtualization => {
            JoltRelationId::SpartanProductVirtualization
        }
        prover_opening::SumcheckId::SpartanShift => JoltRelationId::SpartanShift,
        prover_opening::SumcheckId::InstructionClaimReduction => {
            JoltRelationId::InstructionClaimReduction
        }
        prover_opening::SumcheckId::InstructionInputVirtualization => {
            JoltRelationId::InstructionInputVirtualization
        }
        prover_opening::SumcheckId::InstructionReadRaf => JoltRelationId::InstructionReadRaf,
        prover_opening::SumcheckId::InstructionRaVirtualization => {
            JoltRelationId::InstructionRaVirtualization
        }
        prover_opening::SumcheckId::RamReadWriteChecking => JoltRelationId::RamReadWriteChecking,
        prover_opening::SumcheckId::RamRafEvaluation => JoltRelationId::RamRafEvaluation,
        prover_opening::SumcheckId::RamOutputCheck => JoltRelationId::RamOutputCheck,
        prover_opening::SumcheckId::RamValCheck => JoltRelationId::RamValCheck,
        prover_opening::SumcheckId::RamRaClaimReduction => JoltRelationId::RamRaClaimReduction,
        prover_opening::SumcheckId::RamHammingBooleanity => JoltRelationId::RamHammingBooleanity,
        prover_opening::SumcheckId::RamRaVirtualization => JoltRelationId::RamRaVirtualization,
        prover_opening::SumcheckId::RegistersClaimReduction => {
            JoltRelationId::RegistersClaimReduction
        }
        prover_opening::SumcheckId::RegistersReadWriteChecking => {
            JoltRelationId::RegistersReadWriteChecking
        }
        prover_opening::SumcheckId::RegistersValEvaluation => {
            JoltRelationId::RegistersValEvaluation
        }
        prover_opening::SumcheckId::BytecodeReadRafAddressPhase => JoltRelationId::BytecodeReadRaf,
        prover_opening::SumcheckId::BytecodeReadRaf => JoltRelationId::BytecodeReadRaf,
        prover_opening::SumcheckId::BooleanityAddressPhase => JoltRelationId::Booleanity,
        prover_opening::SumcheckId::Booleanity => JoltRelationId::Booleanity,
        prover_opening::SumcheckId::AdviceClaimReductionCyclePhase => {
            JoltRelationId::AdviceClaimReductionCyclePhase
        }
        prover_opening::SumcheckId::AdviceClaimReduction => JoltRelationId::AdviceClaimReduction,
        prover_opening::SumcheckId::BytecodeClaimReductionCyclePhase => {
            JoltRelationId::BytecodeClaimReductionCyclePhase
        }
        prover_opening::SumcheckId::BytecodeClaimReduction => {
            JoltRelationId::BytecodeClaimReduction
        }
        prover_opening::SumcheckId::ProgramImageClaimReductionCyclePhase => {
            JoltRelationId::ProgramImageClaimReductionCyclePhase
        }
        prover_opening::SumcheckId::ProgramImageClaimReduction => {
            JoltRelationId::ProgramImageClaimReduction
        }
        prover_opening::SumcheckId::IncClaimReduction => JoltRelationId::IncClaimReduction,
        prover_opening::SumcheckId::HammingWeightClaimReduction => {
            JoltRelationId::HammingWeightClaimReduction
        }
    }
}

#[cfg(not(feature = "zk"))]
fn convert_committed_polynomial(
    poly: prover_witness::CommittedPolynomial,
) -> JoltCommittedPolynomial {
    match poly {
        prover_witness::CommittedPolynomial::RdInc => JoltCommittedPolynomial::RdInc,
        prover_witness::CommittedPolynomial::RamInc => JoltCommittedPolynomial::RamInc,
        prover_witness::CommittedPolynomial::InstructionRa(index) => {
            JoltCommittedPolynomial::InstructionRa(index)
        }
        prover_witness::CommittedPolynomial::BytecodeRa(index) => {
            JoltCommittedPolynomial::BytecodeRa(index)
        }
        prover_witness::CommittedPolynomial::BytecodeChunk(index) => {
            JoltCommittedPolynomial::BytecodeChunk(index)
        }
        prover_witness::CommittedPolynomial::RamRa(index) => JoltCommittedPolynomial::RamRa(index),
        prover_witness::CommittedPolynomial::TrustedAdvice => {
            JoltCommittedPolynomial::TrustedAdvice
        }
        prover_witness::CommittedPolynomial::UntrustedAdvice => {
            JoltCommittedPolynomial::UntrustedAdvice
        }
        prover_witness::CommittedPolynomial::ProgramImageInit => {
            JoltCommittedPolynomial::ProgramImageInit
        }
    }
}

#[cfg(not(feature = "zk"))]
fn convert_virtual_polynomial(poly: prover_witness::VirtualPolynomial) -> JoltVirtualPolynomial {
    match poly {
        prover_witness::VirtualPolynomial::PC => JoltVirtualPolynomial::PC,
        prover_witness::VirtualPolynomial::UnexpandedPC => JoltVirtualPolynomial::UnexpandedPC,
        prover_witness::VirtualPolynomial::NextPC => JoltVirtualPolynomial::NextPC,
        prover_witness::VirtualPolynomial::NextUnexpandedPC => {
            JoltVirtualPolynomial::NextUnexpandedPC
        }
        prover_witness::VirtualPolynomial::NextIsNoop => JoltVirtualPolynomial::NextIsNoop,
        prover_witness::VirtualPolynomial::NextIsVirtual => JoltVirtualPolynomial::NextIsVirtual,
        prover_witness::VirtualPolynomial::NextIsFirstInSequence => {
            JoltVirtualPolynomial::NextIsFirstInSequence
        }
        prover_witness::VirtualPolynomial::LeftLookupOperand => {
            JoltVirtualPolynomial::LeftLookupOperand
        }
        prover_witness::VirtualPolynomial::RightLookupOperand => {
            JoltVirtualPolynomial::RightLookupOperand
        }
        prover_witness::VirtualPolynomial::LeftInstructionInput => {
            JoltVirtualPolynomial::LeftInstructionInput
        }
        prover_witness::VirtualPolynomial::RightInstructionInput => {
            JoltVirtualPolynomial::RightInstructionInput
        }
        prover_witness::VirtualPolynomial::Product => JoltVirtualPolynomial::Product,
        prover_witness::VirtualPolynomial::ShouldJump => JoltVirtualPolynomial::ShouldJump,
        prover_witness::VirtualPolynomial::ShouldBranch => JoltVirtualPolynomial::ShouldBranch,
        prover_witness::VirtualPolynomial::Rd => JoltVirtualPolynomial::Rd,
        prover_witness::VirtualPolynomial::Imm => JoltVirtualPolynomial::Imm,
        prover_witness::VirtualPolynomial::Rs1Value => JoltVirtualPolynomial::Rs1Value,
        prover_witness::VirtualPolynomial::Rs2Value => JoltVirtualPolynomial::Rs2Value,
        prover_witness::VirtualPolynomial::RdWriteValue => JoltVirtualPolynomial::RdWriteValue,
        prover_witness::VirtualPolynomial::Rs1Ra => JoltVirtualPolynomial::Rs1Ra,
        prover_witness::VirtualPolynomial::Rs2Ra => JoltVirtualPolynomial::Rs2Ra,
        prover_witness::VirtualPolynomial::RdWa => JoltVirtualPolynomial::RdWa,
        prover_witness::VirtualPolynomial::LookupOutput => JoltVirtualPolynomial::LookupOutput,
        prover_witness::VirtualPolynomial::InstructionRaf => JoltVirtualPolynomial::InstructionRaf,
        prover_witness::VirtualPolynomial::InstructionRafFlag => {
            JoltVirtualPolynomial::InstructionRafFlag
        }
        prover_witness::VirtualPolynomial::InstructionRa(index) => {
            JoltVirtualPolynomial::InstructionRa(index)
        }
        prover_witness::VirtualPolynomial::RegistersVal => JoltVirtualPolynomial::RegistersVal,
        prover_witness::VirtualPolynomial::RamAddress => JoltVirtualPolynomial::RamAddress,
        prover_witness::VirtualPolynomial::RamRa => JoltVirtualPolynomial::RamRa,
        prover_witness::VirtualPolynomial::RamReadValue => JoltVirtualPolynomial::RamReadValue,
        prover_witness::VirtualPolynomial::RamWriteValue => JoltVirtualPolynomial::RamWriteValue,
        prover_witness::VirtualPolynomial::RamVal => JoltVirtualPolynomial::RamVal,
        prover_witness::VirtualPolynomial::RamValInit => JoltVirtualPolynomial::RamValInit,
        prover_witness::VirtualPolynomial::RamValFinal => JoltVirtualPolynomial::RamValFinal,
        prover_witness::VirtualPolynomial::RamHammingWeight => {
            JoltVirtualPolynomial::RamHammingWeight
        }
        prover_witness::VirtualPolynomial::UnivariateSkip => JoltVirtualPolynomial::UnivariateSkip,
        prover_witness::VirtualPolynomial::OpFlags(flag) => {
            JoltVirtualPolynomial::OpFlags(convert_circuit_flag(flag))
        }
        prover_witness::VirtualPolynomial::InstructionFlags(flag) => {
            JoltVirtualPolynomial::InstructionFlags(convert_instruction_flag(flag))
        }
        prover_witness::VirtualPolynomial::LookupTableFlag(index) => {
            JoltVirtualPolynomial::LookupTableFlag(index)
        }
        prover_witness::VirtualPolynomial::BytecodeValStage(stage) => {
            JoltVirtualPolynomial::BytecodeValStage(stage)
        }
        prover_witness::VirtualPolynomial::BytecodeReadRafAddrClaim => {
            JoltVirtualPolynomial::BytecodeReadRafAddrClaim
        }
        prover_witness::VirtualPolynomial::BooleanityAddrClaim => {
            JoltVirtualPolynomial::BooleanityAddrClaim
        }
        prover_witness::VirtualPolynomial::BytecodeClaimReductionIntermediate => {
            JoltVirtualPolynomial::BytecodeClaimReductionIntermediate
        }
        prover_witness::VirtualPolynomial::ProgramImageInitContributionRw => {
            JoltVirtualPolynomial::ProgramImageInitContributionRw
        }
    }
}

#[cfg(not(feature = "zk"))]
fn convert_circuit_flag(flag: prover_instruction::CircuitFlags) -> jolt_riscv::CircuitFlags {
    match flag {
        prover_instruction::CircuitFlags::AddOperands => jolt_riscv::CircuitFlags::AddOperands,
        prover_instruction::CircuitFlags::SubtractOperands => {
            jolt_riscv::CircuitFlags::SubtractOperands
        }
        prover_instruction::CircuitFlags::MultiplyOperands => {
            jolt_riscv::CircuitFlags::MultiplyOperands
        }
        prover_instruction::CircuitFlags::Load => jolt_riscv::CircuitFlags::Load,
        prover_instruction::CircuitFlags::Store => jolt_riscv::CircuitFlags::Store,
        prover_instruction::CircuitFlags::Jump => jolt_riscv::CircuitFlags::Jump,
        prover_instruction::CircuitFlags::WriteLookupOutputToRD => {
            jolt_riscv::CircuitFlags::WriteLookupOutputToRD
        }
        prover_instruction::CircuitFlags::VirtualInstruction => {
            jolt_riscv::CircuitFlags::VirtualInstruction
        }
        prover_instruction::CircuitFlags::Assert => jolt_riscv::CircuitFlags::Assert,
        prover_instruction::CircuitFlags::DoNotUpdateUnexpandedPC => {
            jolt_riscv::CircuitFlags::DoNotUpdateUnexpandedPC
        }
        prover_instruction::CircuitFlags::Advice => jolt_riscv::CircuitFlags::Advice,
        prover_instruction::CircuitFlags::IsCompressed => jolt_riscv::CircuitFlags::IsCompressed,
        prover_instruction::CircuitFlags::IsFirstInSequence => {
            jolt_riscv::CircuitFlags::IsFirstInSequence
        }
        prover_instruction::CircuitFlags::IsLastInSequence => {
            jolt_riscv::CircuitFlags::IsLastInSequence
        }
    }
}

#[cfg(not(feature = "zk"))]
fn convert_instruction_flag(
    flag: prover_instruction::InstructionFlags,
) -> jolt_riscv::InstructionFlags {
    match flag {
        prover_instruction::InstructionFlags::LeftOperandIsPC => {
            jolt_riscv::InstructionFlags::LeftOperandIsPC
        }
        prover_instruction::InstructionFlags::RightOperandIsImm => {
            jolt_riscv::InstructionFlags::RightOperandIsImm
        }
        prover_instruction::InstructionFlags::LeftOperandIsRs1Value => {
            jolt_riscv::InstructionFlags::LeftOperandIsRs1Value
        }
        prover_instruction::InstructionFlags::RightOperandIsRs2Value => {
            jolt_riscv::InstructionFlags::RightOperandIsRs2Value
        }
        prover_instruction::InstructionFlags::Branch => jolt_riscv::InstructionFlags::Branch,
        prover_instruction::InstructionFlags::IsNoop => jolt_riscv::InstructionFlags::IsNoop,
    }
}
