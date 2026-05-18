//! Conversions from imported `jolt-core` types into verifier-owned model types.

use crate::proof::{JoltProof, JoltStageProofs};
#[cfg(not(feature = "zk"))]
use crate::{compat::claims::LegacyOpeningClaims, compat::ids as verifier_ids};
use jolt_claims::protocols::jolt::{JoltOneHotConfig, JoltReadWriteConfig};
use jolt_crypto::{
    Bn254G1, Bn254GT, Commitment as ModularCommitment, Pedersen,
    VectorCommitment as ModularVectorCommitment,
};
use jolt_dory::{DoryCommitment, DoryProof, DoryScheme};
use jolt_field::{Field as ModularField, Fr as ModularFr};
use jolt_openings::CommitmentScheme as ModularCommitmentScheme;
use jolt_poly::{CompressedPoly, UnivariatePoly};
use jolt_sumcheck::{
    ClearProof, ClearSumcheckProof, CommittedOutputClaims, CommittedRound, CommittedSumcheckProof,
    CompressedSumcheckProof, SumcheckProof,
};

use jolt_core::{
    curve::{Bn254Curve as CoreBn254Curve, JoltCurve},
    field::JoltField,
    poly::commitment::{
        commitment_scheme::CommitmentScheme as CoreCommitmentScheme,
        dory::{ArkGT as CoreDoryCommitment, DoryCommitmentScheme as CoreDoryCommitmentScheme},
    },
    subprotocols::{
        sumcheck::SumcheckInstanceProof, univariate_skip::UniSkipFirstRoundProofVariant,
    },
    transcripts::Transcript,
    zkvm::{
        config::{OneHotConfig as CoreOneHotConfig, ReadWriteConfig as CoreReadWriteConfig},
        proof_serialization::JoltProof as CoreJoltProof,
    },
};
#[cfg(not(feature = "zk"))]
use jolt_core::{
    poly::opening_proof as core_opening,
    zkvm::{instruction as core_instruction, witness as core_witness},
};
#[cfg(feature = "zk")]
use serde::{Deserialize, Deserializer, Serialize, Serializer};
#[cfg(feature = "zk")]
use std::fmt::{Debug, Formatter, Result as FmtResult};

#[cfg(feature = "zk")]
use jolt_core::subprotocols::blindfold::BlindFoldProof;
#[cfg(not(feature = "zk"))]
use jolt_core::zkvm::proof_serialization::Claims as CoreClaims;

pub type JoltCoreProof<F, C, PCS, FS> = CoreJoltProof<F, C, PCS, FS>;

pub trait CoreFieldBridge: JoltField {
    type VerifierField: ModularField;

    fn into_verifier_field(self) -> Self::VerifierField;
}

impl CoreFieldBridge for ark_bn254::Fr {
    type VerifierField = ModularFr;

    fn into_verifier_field(self) -> Self::VerifierField {
        self.into()
    }
}

pub trait CorePcsBridge<F>: CoreCommitmentScheme<Field = F>
where
    F: CoreFieldBridge,
{
    type VerifierPcs: ModularCommitmentScheme<Field = F::VerifierField>;

    fn commitment_into_verifier(
        commitment: Self::Commitment,
    ) -> <Self::VerifierPcs as ModularCommitment>::Output;

    fn proof_into_verifier(
        proof: Self::Proof,
    ) -> <Self::VerifierPcs as ModularCommitmentScheme>::Proof;
}

impl CorePcsBridge<ark_bn254::Fr> for CoreDoryCommitmentScheme {
    type VerifierPcs = DoryScheme;

    fn commitment_into_verifier(commitment: Self::Commitment) -> DoryCommitment {
        DoryCommitment(core_dory_commitment_into_verifier(&commitment))
    }

    fn proof_into_verifier(proof: Self::Proof) -> DoryProof {
        DoryProof(proof)
    }
}

pub trait CoreCurveBridge<F>: JoltCurve<F = F>
where
    F: CoreFieldBridge,
{
    type VerifierVectorCommitment: ModularVectorCommitment<
        Field = F::VerifierField,
        Output = Self::VerifierRoundCommitment,
    >;
    type VerifierRoundCommitment: Copy
        + jolt_transcript::AppendToTranscript
        + serde::Serialize
        + serde::de::DeserializeOwned;

    fn g1_into_verifier(commitment: Self::G1) -> Self::VerifierRoundCommitment;
}

impl CoreCurveBridge<ark_bn254::Fr> for CoreBn254Curve {
    type VerifierVectorCommitment = Pedersen<Bn254G1>;
    type VerifierRoundCommitment = Bn254G1;

    fn g1_into_verifier(commitment: Self::G1) -> Self::VerifierRoundCommitment {
        commitment.0.into()
    }
}

#[cfg(not(feature = "zk"))]
pub type ImportedCoreProof<F, C, PCS> = JoltProof<
    <PCS as CorePcsBridge<F>>::VerifierPcs,
    <C as CoreCurveBridge<F>>::VerifierVectorCommitment,
    LegacyOpeningClaims<<F as CoreFieldBridge>::VerifierField>,
    (),
>;

#[cfg(feature = "zk")]
pub type ImportedCoreProof<F, C, PCS> = JoltProof<
    <PCS as CorePcsBridge<F>>::VerifierPcs,
    <C as CoreCurveBridge<F>>::VerifierVectorCommitment,
    (),
    LegacyBlindFoldProof<F, C>,
>;

fn convert_read_write_config(config: CoreReadWriteConfig) -> JoltReadWriteConfig {
    JoltReadWriteConfig {
        ram_rw_phase1_num_rounds: config.ram_rw_phase1_num_rounds,
        ram_rw_phase2_num_rounds: config.ram_rw_phase2_num_rounds,
        registers_rw_phase1_num_rounds: config.registers_rw_phase1_num_rounds,
        registers_rw_phase2_num_rounds: config.registers_rw_phase2_num_rounds,
    }
}

fn convert_one_hot_config(config: CoreOneHotConfig) -> JoltOneHotConfig {
    JoltOneHotConfig {
        log_k_chunk: config.log_k_chunk,
        lookups_ra_virtual_log_k_chunk: config.lookups_ra_virtual_log_k_chunk,
    }
}

#[cfg(not(feature = "zk"))]
impl<F, C, PCS, FS> From<JoltCoreProof<F, C, PCS, FS>> for ImportedCoreProof<F, C, PCS>
where
    F: CoreFieldBridge,
    C: CoreCurveBridge<F>,
    PCS: CorePcsBridge<F>,
    FS: Transcript,
{
    fn from(proof: JoltCoreProof<F, C, PCS, FS>) -> Self {
        let stages = JoltStageProofs {
            stage1_uni_skip_first_round_proof: convert_uniskip(
                proof.stage1_uni_skip_first_round_proof,
            ),
            stage1_sumcheck_proof: convert_sumcheck(proof.stage1_sumcheck_proof),
            stage2_uni_skip_first_round_proof: convert_uniskip(
                proof.stage2_uni_skip_first_round_proof,
            ),
            stage2_sumcheck_proof: convert_sumcheck(proof.stage2_sumcheck_proof),
            stage3_sumcheck_proof: convert_sumcheck(proof.stage3_sumcheck_proof),
            stage4_sumcheck_proof: convert_sumcheck(proof.stage4_sumcheck_proof),
            stage5_sumcheck_proof: convert_sumcheck(proof.stage5_sumcheck_proof),
            stage6_sumcheck_proof: convert_sumcheck(proof.stage6_sumcheck_proof),
            stage7_sumcheck_proof: convert_sumcheck(proof.stage7_sumcheck_proof),
        };

        Self {
            commitments: proof
                .commitments
                .into_iter()
                .map(PCS::commitment_into_verifier)
                .collect(),
            stages,
            joint_opening_proof: PCS::proof_into_verifier(proof.joint_opening_proof),
            untrusted_advice_commitment: proof
                .untrusted_advice_commitment
                .map(PCS::commitment_into_verifier),
            opening_claims: Some(convert_opening_claims(proof.opening_claims)),
            blindfold_proof: None,
            trace_length: proof.trace_length,
            ram_K: proof.ram_K,
            rw_config: convert_read_write_config(proof.rw_config),
            one_hot_config: convert_one_hot_config(proof.one_hot_config),
        }
    }
}

#[cfg(feature = "zk")]
impl<F, C, PCS, FS> From<JoltCoreProof<F, C, PCS, FS>> for ImportedCoreProof<F, C, PCS>
where
    F: CoreFieldBridge,
    C: CoreCurveBridge<F>,
    PCS: CorePcsBridge<F>,
    FS: Transcript,
{
    fn from(proof: JoltCoreProof<F, C, PCS, FS>) -> Self {
        let stages = JoltStageProofs {
            stage1_uni_skip_first_round_proof: convert_uniskip(
                proof.stage1_uni_skip_first_round_proof,
            ),
            stage1_sumcheck_proof: convert_sumcheck(proof.stage1_sumcheck_proof),
            stage2_uni_skip_first_round_proof: convert_uniskip(
                proof.stage2_uni_skip_first_round_proof,
            ),
            stage2_sumcheck_proof: convert_sumcheck(proof.stage2_sumcheck_proof),
            stage3_sumcheck_proof: convert_sumcheck(proof.stage3_sumcheck_proof),
            stage4_sumcheck_proof: convert_sumcheck(proof.stage4_sumcheck_proof),
            stage5_sumcheck_proof: convert_sumcheck(proof.stage5_sumcheck_proof),
            stage6_sumcheck_proof: convert_sumcheck(proof.stage6_sumcheck_proof),
            stage7_sumcheck_proof: convert_sumcheck(proof.stage7_sumcheck_proof),
        };

        Self {
            commitments: proof
                .commitments
                .into_iter()
                .map(PCS::commitment_into_verifier)
                .collect(),
            stages,
            joint_opening_proof: PCS::proof_into_verifier(proof.joint_opening_proof),
            untrusted_advice_commitment: proof
                .untrusted_advice_commitment
                .map(PCS::commitment_into_verifier),
            opening_claims: None,
            blindfold_proof: Some(LegacyBlindFoldProof(proof.blindfold_proof)),
            trace_length: proof.trace_length,
            ram_K: proof.ram_K,
            rw_config: convert_read_write_config(proof.rw_config),
            one_hot_config: convert_one_hot_config(proof.one_hot_config),
        }
    }
}

fn convert_uniskip<F, C, FS>(
    proof: UniSkipFirstRoundProofVariant<F, C, FS>,
) -> SumcheckProof<F::VerifierField, C::VerifierRoundCommitment>
where
    F: CoreFieldBridge,
    C: CoreCurveBridge<F>,
    FS: Transcript,
{
    match proof {
        UniSkipFirstRoundProofVariant::Standard(proof) => {
            SumcheckProof::Clear(ClearProof::Full(ClearSumcheckProof {
                round_polynomials: vec![convert_univariate(proof.uni_poly)],
            }))
        }
        UniSkipFirstRoundProofVariant::Zk(proof) => {
            SumcheckProof::Committed(committed_proof_from_parts::<F, C>(
                vec![proof.commitment],
                vec![proof.poly_degree],
                proof.output_claims_commitments,
            ))
        }
    }
}

fn convert_sumcheck<F, C, FS>(
    proof: SumcheckInstanceProof<F, C, FS>,
) -> SumcheckProof<F::VerifierField, C::VerifierRoundCommitment>
where
    F: CoreFieldBridge,
    C: CoreCurveBridge<F>,
    FS: Transcript,
{
    match proof {
        SumcheckInstanceProof::Clear(proof) => {
            SumcheckProof::Clear(ClearProof::Compressed(CompressedSumcheckProof {
                round_polynomials: proof
                    .compressed_polys
                    .into_iter()
                    .map(convert_compressed_poly)
                    .collect(),
            }))
        }
        SumcheckInstanceProof::Zk(proof) => {
            SumcheckProof::Committed(committed_proof_from_parts::<F, C>(
                proof.round_commitments,
                proof.poly_degrees,
                proof.output_claims_commitments,
            ))
        }
    }
}

fn committed_proof_from_parts<F, C>(
    round_commitments: Vec<C::G1>,
    poly_degrees: Vec<usize>,
    output_claims_commitments: Vec<C::G1>,
) -> CommittedSumcheckProof<C::VerifierRoundCommitment>
where
    F: CoreFieldBridge,
    C: CoreCurveBridge<F>,
{
    let rounds = round_commitments
        .into_iter()
        .zip(poly_degrees)
        .map(|(commitment, degree)| CommittedRound {
            commitment: C::g1_into_verifier(commitment),
            degree,
        })
        .collect();

    CommittedSumcheckProof {
        rounds,
        output_claims: CommittedOutputClaims {
            commitments: output_claims_commitments
                .into_iter()
                .map(C::g1_into_verifier)
                .collect(),
        },
    }
}

fn convert_univariate<F>(
    poly: jolt_core::poly::unipoly::UniPoly<F>,
) -> UnivariatePoly<F::VerifierField>
where
    F: CoreFieldBridge,
{
    UnivariatePoly::new(convert_field_vec(poly.coeffs))
}

fn convert_compressed_poly<F>(
    poly: jolt_core::poly::unipoly::CompressedUniPoly<F>,
) -> CompressedPoly<F::VerifierField>
where
    F: CoreFieldBridge,
{
    CompressedPoly::new(convert_field_vec(poly.coeffs_except_linear_term))
}

fn convert_field_vec<F>(values: Vec<F>) -> Vec<F::VerifierField>
where
    F: CoreFieldBridge,
{
    values
        .into_iter()
        .map(CoreFieldBridge::into_verifier_field)
        .collect()
}

fn core_dory_commitment_into_verifier(commitment: &CoreDoryCommitment) -> Bn254GT {
    // SAFETY: `jolt-core` Dory and modular `jolt-dory` use transparent wrappers
    // over the same arkworks `Fq12` target-group element.
    unsafe { std::mem::transmute_copy(commitment) }
}

#[cfg(feature = "zk")]
#[derive(Clone)]
pub struct LegacyBlindFoldProof<F: JoltField, C: JoltCurve<F = F>>(pub BlindFoldProof<F, C>);

#[cfg(feature = "zk")]
impl<F, C> Debug for LegacyBlindFoldProof<F, C>
where
    F: JoltField,
    C: JoltCurve<F = F>,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.debug_tuple("LegacyBlindFoldProof")
            .finish_non_exhaustive()
    }
}

#[cfg(feature = "zk")]
impl<F, C> Serialize for LegacyBlindFoldProof<F, C>
where
    F: JoltField,
    C: JoltCurve<F = F>,
{
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use ark_serialize::CanonicalSerialize;

        let mut bytes = Vec::new();
        self.0
            .serialize_compressed(&mut bytes)
            .map_err(serde::ser::Error::custom)?;
        serializer.serialize_bytes(&bytes)
    }
}

#[cfg(feature = "zk")]
impl<'de, F, C> Deserialize<'de> for LegacyBlindFoldProof<F, C>
where
    F: JoltField,
    C: JoltCurve<F = F>,
{
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use ark_serialize::CanonicalDeserialize;

        let bytes = <Vec<u8> as Deserialize>::deserialize(deserializer)?;
        BlindFoldProof::deserialize_compressed(&bytes[..])
            .map(Self)
            .map_err(serde::de::Error::custom)
    }
}

#[cfg(not(feature = "zk"))]
fn convert_opening_claims<F>(claims: CoreClaims<F>) -> LegacyOpeningClaims<F::VerifierField>
where
    F: CoreFieldBridge,
{
    LegacyOpeningClaims(
        claims
            .0
            .into_iter()
            .map(|(id, (_point, claim))| {
                (
                    convert_opening_id(id),
                    CoreFieldBridge::into_verifier_field(claim),
                )
            })
            .collect::<std::collections::BTreeMap<_, _>>(),
    )
}

#[cfg(not(feature = "zk"))]
fn convert_opening_id(id: core_opening::OpeningId) -> verifier_ids::OpeningId {
    match id {
        core_opening::OpeningId::Polynomial(poly, sumcheck) => verifier_ids::OpeningId::Polynomial(
            convert_polynomial_id(poly),
            convert_sumcheck_id(sumcheck),
        ),
        core_opening::OpeningId::UntrustedAdvice(sumcheck) => {
            verifier_ids::OpeningId::UntrustedAdvice(convert_sumcheck_id(sumcheck))
        }
        core_opening::OpeningId::TrustedAdvice(sumcheck) => {
            verifier_ids::OpeningId::TrustedAdvice(convert_sumcheck_id(sumcheck))
        }
    }
}

#[cfg(not(feature = "zk"))]
fn convert_polynomial_id(id: core_opening::PolynomialId) -> verifier_ids::PolynomialId {
    match id {
        core_opening::PolynomialId::Committed(poly) => {
            verifier_ids::PolynomialId::Committed(convert_committed_polynomial(poly))
        }
        core_opening::PolynomialId::Virtual(poly) => {
            verifier_ids::PolynomialId::Virtual(convert_virtual_polynomial(poly))
        }
    }
}

#[cfg(not(feature = "zk"))]
fn convert_sumcheck_id(id: core_opening::SumcheckId) -> verifier_ids::SumcheckId {
    match id {
        core_opening::SumcheckId::SpartanOuter => verifier_ids::SumcheckId::SpartanOuter,
        core_opening::SumcheckId::SpartanProductVirtualization => {
            verifier_ids::SumcheckId::SpartanProductVirtualization
        }
        core_opening::SumcheckId::SpartanShift => verifier_ids::SumcheckId::SpartanShift,
        core_opening::SumcheckId::InstructionClaimReduction => {
            verifier_ids::SumcheckId::InstructionClaimReduction
        }
        core_opening::SumcheckId::InstructionInputVirtualization => {
            verifier_ids::SumcheckId::InstructionInputVirtualization
        }
        core_opening::SumcheckId::InstructionReadRaf => {
            verifier_ids::SumcheckId::InstructionReadRaf
        }
        core_opening::SumcheckId::InstructionRaVirtualization => {
            verifier_ids::SumcheckId::InstructionRaVirtualization
        }
        core_opening::SumcheckId::RamReadWriteChecking => {
            verifier_ids::SumcheckId::RamReadWriteChecking
        }
        core_opening::SumcheckId::RamRafEvaluation => verifier_ids::SumcheckId::RamRafEvaluation,
        core_opening::SumcheckId::RamOutputCheck => verifier_ids::SumcheckId::RamOutputCheck,
        core_opening::SumcheckId::RamValCheck => verifier_ids::SumcheckId::RamValCheck,
        core_opening::SumcheckId::RamRaClaimReduction => {
            verifier_ids::SumcheckId::RamRaClaimReduction
        }
        core_opening::SumcheckId::RamHammingBooleanity => {
            verifier_ids::SumcheckId::RamHammingBooleanity
        }
        core_opening::SumcheckId::RamRaVirtualization => {
            verifier_ids::SumcheckId::RamRaVirtualization
        }
        core_opening::SumcheckId::RegistersClaimReduction => {
            verifier_ids::SumcheckId::RegistersClaimReduction
        }
        core_opening::SumcheckId::RegistersReadWriteChecking => {
            verifier_ids::SumcheckId::RegistersReadWriteChecking
        }
        core_opening::SumcheckId::RegistersValEvaluation => {
            verifier_ids::SumcheckId::RegistersValEvaluation
        }
        core_opening::SumcheckId::BytecodeReadRaf => verifier_ids::SumcheckId::BytecodeReadRaf,
        core_opening::SumcheckId::Booleanity => verifier_ids::SumcheckId::Booleanity,
        core_opening::SumcheckId::AdviceClaimReductionCyclePhase => {
            verifier_ids::SumcheckId::AdviceClaimReductionCyclePhase
        }
        core_opening::SumcheckId::AdviceClaimReduction => {
            verifier_ids::SumcheckId::AdviceClaimReduction
        }
        core_opening::SumcheckId::IncClaimReduction => verifier_ids::SumcheckId::IncClaimReduction,
        core_opening::SumcheckId::HammingWeightClaimReduction => {
            verifier_ids::SumcheckId::HammingWeightClaimReduction
        }
    }
}

#[cfg(not(feature = "zk"))]
fn convert_committed_polynomial(
    poly: core_witness::CommittedPolynomial,
) -> verifier_ids::CommittedPolynomial {
    match poly {
        core_witness::CommittedPolynomial::RdInc => verifier_ids::CommittedPolynomial::RdInc,
        core_witness::CommittedPolynomial::RamInc => verifier_ids::CommittedPolynomial::RamInc,
        core_witness::CommittedPolynomial::InstructionRa(index) => {
            verifier_ids::CommittedPolynomial::InstructionRa(index)
        }
        core_witness::CommittedPolynomial::BytecodeRa(index) => {
            verifier_ids::CommittedPolynomial::BytecodeRa(index)
        }
        core_witness::CommittedPolynomial::RamRa(index) => {
            verifier_ids::CommittedPolynomial::RamRa(index)
        }
        core_witness::CommittedPolynomial::TrustedAdvice => {
            verifier_ids::CommittedPolynomial::TrustedAdvice
        }
        core_witness::CommittedPolynomial::UntrustedAdvice => {
            verifier_ids::CommittedPolynomial::UntrustedAdvice
        }
    }
}

#[cfg(not(feature = "zk"))]
fn convert_virtual_polynomial(
    poly: core_witness::VirtualPolynomial,
) -> verifier_ids::VirtualPolynomial {
    match poly {
        core_witness::VirtualPolynomial::PC => verifier_ids::VirtualPolynomial::PC,
        core_witness::VirtualPolynomial::UnexpandedPC => {
            verifier_ids::VirtualPolynomial::UnexpandedPC
        }
        core_witness::VirtualPolynomial::NextPC => verifier_ids::VirtualPolynomial::NextPC,
        core_witness::VirtualPolynomial::NextUnexpandedPC => {
            verifier_ids::VirtualPolynomial::NextUnexpandedPC
        }
        core_witness::VirtualPolynomial::NextIsNoop => verifier_ids::VirtualPolynomial::NextIsNoop,
        core_witness::VirtualPolynomial::NextIsVirtual => {
            verifier_ids::VirtualPolynomial::NextIsVirtual
        }
        core_witness::VirtualPolynomial::NextIsFirstInSequence => {
            verifier_ids::VirtualPolynomial::NextIsFirstInSequence
        }
        core_witness::VirtualPolynomial::LeftLookupOperand => {
            verifier_ids::VirtualPolynomial::LeftLookupOperand
        }
        core_witness::VirtualPolynomial::RightLookupOperand => {
            verifier_ids::VirtualPolynomial::RightLookupOperand
        }
        core_witness::VirtualPolynomial::LeftInstructionInput => {
            verifier_ids::VirtualPolynomial::LeftInstructionInput
        }
        core_witness::VirtualPolynomial::RightInstructionInput => {
            verifier_ids::VirtualPolynomial::RightInstructionInput
        }
        core_witness::VirtualPolynomial::Product => verifier_ids::VirtualPolynomial::Product,
        core_witness::VirtualPolynomial::ShouldJump => verifier_ids::VirtualPolynomial::ShouldJump,
        core_witness::VirtualPolynomial::ShouldBranch => {
            verifier_ids::VirtualPolynomial::ShouldBranch
        }
        core_witness::VirtualPolynomial::Rd => verifier_ids::VirtualPolynomial::Rd,
        core_witness::VirtualPolynomial::Imm => verifier_ids::VirtualPolynomial::Imm,
        core_witness::VirtualPolynomial::Rs1Value => verifier_ids::VirtualPolynomial::Rs1Value,
        core_witness::VirtualPolynomial::Rs2Value => verifier_ids::VirtualPolynomial::Rs2Value,
        core_witness::VirtualPolynomial::RdWriteValue => {
            verifier_ids::VirtualPolynomial::RdWriteValue
        }
        core_witness::VirtualPolynomial::Rs1Ra => verifier_ids::VirtualPolynomial::Rs1Ra,
        core_witness::VirtualPolynomial::Rs2Ra => verifier_ids::VirtualPolynomial::Rs2Ra,
        core_witness::VirtualPolynomial::RdWa => verifier_ids::VirtualPolynomial::RdWa,
        core_witness::VirtualPolynomial::LookupOutput => {
            verifier_ids::VirtualPolynomial::LookupOutput
        }
        core_witness::VirtualPolynomial::InstructionRaf => {
            verifier_ids::VirtualPolynomial::InstructionRaf
        }
        core_witness::VirtualPolynomial::InstructionRafFlag => {
            verifier_ids::VirtualPolynomial::InstructionRafFlag
        }
        core_witness::VirtualPolynomial::InstructionRa(index) => {
            verifier_ids::VirtualPolynomial::InstructionRa(index)
        }
        core_witness::VirtualPolynomial::RegistersVal => {
            verifier_ids::VirtualPolynomial::RegistersVal
        }
        core_witness::VirtualPolynomial::RamAddress => verifier_ids::VirtualPolynomial::RamAddress,
        core_witness::VirtualPolynomial::RamRa => verifier_ids::VirtualPolynomial::RamRa,
        core_witness::VirtualPolynomial::RamReadValue => {
            verifier_ids::VirtualPolynomial::RamReadValue
        }
        core_witness::VirtualPolynomial::RamWriteValue => {
            verifier_ids::VirtualPolynomial::RamWriteValue
        }
        core_witness::VirtualPolynomial::RamVal => verifier_ids::VirtualPolynomial::RamVal,
        core_witness::VirtualPolynomial::RamValInit => verifier_ids::VirtualPolynomial::RamValInit,
        core_witness::VirtualPolynomial::RamValFinal => {
            verifier_ids::VirtualPolynomial::RamValFinal
        }
        core_witness::VirtualPolynomial::RamHammingWeight => {
            verifier_ids::VirtualPolynomial::RamHammingWeight
        }
        core_witness::VirtualPolynomial::UnivariateSkip => {
            verifier_ids::VirtualPolynomial::UnivariateSkip
        }
        core_witness::VirtualPolynomial::OpFlags(flag) => {
            verifier_ids::VirtualPolynomial::OpFlags(convert_circuit_flag(flag))
        }
        core_witness::VirtualPolynomial::InstructionFlags(flag) => {
            verifier_ids::VirtualPolynomial::InstructionFlags(convert_instruction_flag(flag))
        }
        core_witness::VirtualPolynomial::LookupTableFlag(index) => {
            verifier_ids::VirtualPolynomial::LookupTableFlag(index)
        }
    }
}

#[cfg(not(feature = "zk"))]
fn convert_circuit_flag(flag: core_instruction::CircuitFlags) -> jolt_riscv::CircuitFlags {
    match flag {
        core_instruction::CircuitFlags::AddOperands => jolt_riscv::CircuitFlags::AddOperands,
        core_instruction::CircuitFlags::SubtractOperands => {
            jolt_riscv::CircuitFlags::SubtractOperands
        }
        core_instruction::CircuitFlags::MultiplyOperands => {
            jolt_riscv::CircuitFlags::MultiplyOperands
        }
        core_instruction::CircuitFlags::Load => jolt_riscv::CircuitFlags::Load,
        core_instruction::CircuitFlags::Store => jolt_riscv::CircuitFlags::Store,
        core_instruction::CircuitFlags::Jump => jolt_riscv::CircuitFlags::Jump,
        core_instruction::CircuitFlags::WriteLookupOutputToRD => {
            jolt_riscv::CircuitFlags::WriteLookupOutputToRD
        }
        core_instruction::CircuitFlags::VirtualInstruction => {
            jolt_riscv::CircuitFlags::VirtualInstruction
        }
        core_instruction::CircuitFlags::Assert => jolt_riscv::CircuitFlags::Assert,
        core_instruction::CircuitFlags::DoNotUpdateUnexpandedPC => {
            jolt_riscv::CircuitFlags::DoNotUpdateUnexpandedPC
        }
        core_instruction::CircuitFlags::Advice => jolt_riscv::CircuitFlags::Advice,
        core_instruction::CircuitFlags::IsCompressed => jolt_riscv::CircuitFlags::IsCompressed,
        core_instruction::CircuitFlags::IsFirstInSequence => {
            jolt_riscv::CircuitFlags::IsFirstInSequence
        }
        core_instruction::CircuitFlags::IsLastInSequence => {
            jolt_riscv::CircuitFlags::IsLastInSequence
        }
    }
}

#[cfg(not(feature = "zk"))]
fn convert_instruction_flag(
    flag: core_instruction::InstructionFlags,
) -> jolt_riscv::InstructionFlags {
    match flag {
        core_instruction::InstructionFlags::LeftOperandIsPC => {
            jolt_riscv::InstructionFlags::LeftOperandIsPC
        }
        core_instruction::InstructionFlags::RightOperandIsImm => {
            jolt_riscv::InstructionFlags::RightOperandIsImm
        }
        core_instruction::InstructionFlags::LeftOperandIsRs1Value => {
            jolt_riscv::InstructionFlags::LeftOperandIsRs1Value
        }
        core_instruction::InstructionFlags::RightOperandIsRs2Value => {
            jolt_riscv::InstructionFlags::RightOperandIsRs2Value
        }
        core_instruction::InstructionFlags::Branch => jolt_riscv::InstructionFlags::Branch,
        core_instruction::InstructionFlags::IsNoop => jolt_riscv::InstructionFlags::IsNoop,
    }
}
