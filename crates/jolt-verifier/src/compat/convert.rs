//! Conversions from imported `jolt-core` types into verifier-owned model types.

use crate::{
    compat::config::{OneHotConfig, ReadWriteConfig},
    proof::{DoryLayout, JoltProof, JoltStageProofs},
};

use jolt_core::{
    curve::JoltCurve,
    field::JoltField,
    poly::commitment::{commitment_scheme::CommitmentScheme, dory::DoryLayout as CoreDoryLayout},
    subprotocols::{
        sumcheck::SumcheckInstanceProof, univariate_skip::UniSkipFirstRoundProofVariant,
    },
    transcripts::Transcript,
    zkvm::{
        config::{OneHotConfig as CoreOneHotConfig, ReadWriteConfig as CoreReadWriteConfig},
        proof_serialization::JoltProof as CoreJoltProof,
    },
};

#[cfg(feature = "zk")]
use jolt_core::subprotocols::blindfold::BlindFoldProof;
#[cfg(not(feature = "zk"))]
use jolt_core::zkvm::proof_serialization::Claims as CoreClaims;

pub type JoltCoreProof<F, C, PCS, FS> = CoreJoltProof<F, C, PCS, FS>;

#[cfg(not(feature = "zk"))]
pub type ImportedCoreProof<F, C, PCS, FS> = JoltProof<
    <PCS as CommitmentScheme>::Commitment,
    UniSkipFirstRoundProofVariant<F, C, FS>,
    SumcheckInstanceProof<F, C, FS>,
    <PCS as CommitmentScheme>::Proof,
    CoreClaims<F>,
>;

#[cfg(feature = "zk")]
pub type ImportedCoreProof<F, C, PCS, FS> = JoltProof<
    <PCS as CommitmentScheme>::Commitment,
    UniSkipFirstRoundProofVariant<F, C, FS>,
    SumcheckInstanceProof<F, C, FS>,
    <PCS as CommitmentScheme>::Proof,
    (),
    BlindFoldProof<F, C>,
>;

impl From<CoreReadWriteConfig> for ReadWriteConfig {
    fn from(config: CoreReadWriteConfig) -> Self {
        Self {
            ram_rw_phase1_num_rounds: config.ram_rw_phase1_num_rounds,
            ram_rw_phase2_num_rounds: config.ram_rw_phase2_num_rounds,
            registers_rw_phase1_num_rounds: config.registers_rw_phase1_num_rounds,
            registers_rw_phase2_num_rounds: config.registers_rw_phase2_num_rounds,
        }
    }
}

impl From<CoreOneHotConfig> for OneHotConfig {
    fn from(config: CoreOneHotConfig) -> Self {
        Self {
            log_k_chunk: config.log_k_chunk,
            lookups_ra_virtual_log_k_chunk: config.lookups_ra_virtual_log_k_chunk,
        }
    }
}

impl From<CoreDoryLayout> for DoryLayout {
    fn from(layout: CoreDoryLayout) -> Self {
        match layout {
            CoreDoryLayout::CycleMajor => Self::CycleMajor,
            CoreDoryLayout::AddressMajor => Self::AddressMajor,
        }
    }
}

#[cfg(not(feature = "zk"))]
impl<F, C, PCS, FS> From<JoltCoreProof<F, C, PCS, FS>> for ImportedCoreProof<F, C, PCS, FS>
where
    F: JoltField,
    C: JoltCurve<F = F>,
    PCS: CommitmentScheme<Field = F>,
    FS: Transcript,
{
    fn from(proof: JoltCoreProof<F, C, PCS, FS>) -> Self {
        let stages = JoltStageProofs {
            stage1_uni_skip_first_round_proof: proof.stage1_uni_skip_first_round_proof,
            stage1_sumcheck_proof: proof.stage1_sumcheck_proof,
            stage2_uni_skip_first_round_proof: proof.stage2_uni_skip_first_round_proof,
            stage2_sumcheck_proof: proof.stage2_sumcheck_proof,
            stage3_sumcheck_proof: proof.stage3_sumcheck_proof,
            stage4_sumcheck_proof: proof.stage4_sumcheck_proof,
            stage5_sumcheck_proof: proof.stage5_sumcheck_proof,
            stage6_sumcheck_proof: proof.stage6_sumcheck_proof,
            stage7_sumcheck_proof: proof.stage7_sumcheck_proof,
        };

        Self {
            commitments: proof.commitments,
            stages,
            joint_opening_proof: proof.joint_opening_proof,
            untrusted_advice_commitment: proof.untrusted_advice_commitment,
            opening_claims: Some(proof.opening_claims),
            blindfold_proof: None,
            trace_length: proof.trace_length,
            ram_K: proof.ram_K,
            rw_config: proof.rw_config.into(),
            one_hot_config: proof.one_hot_config.into(),
            dory_layout: proof.dory_layout.into(),
        }
    }
}

#[cfg(feature = "zk")]
impl<F, C, PCS, FS> From<JoltCoreProof<F, C, PCS, FS>> for ImportedCoreProof<F, C, PCS, FS>
where
    F: JoltField,
    C: JoltCurve<F = F>,
    PCS: CommitmentScheme<Field = F>,
    FS: Transcript,
{
    fn from(proof: JoltCoreProof<F, C, PCS, FS>) -> Self {
        let stages = JoltStageProofs {
            stage1_uni_skip_first_round_proof: proof.stage1_uni_skip_first_round_proof,
            stage1_sumcheck_proof: proof.stage1_sumcheck_proof,
            stage2_uni_skip_first_round_proof: proof.stage2_uni_skip_first_round_proof,
            stage2_sumcheck_proof: proof.stage2_sumcheck_proof,
            stage3_sumcheck_proof: proof.stage3_sumcheck_proof,
            stage4_sumcheck_proof: proof.stage4_sumcheck_proof,
            stage5_sumcheck_proof: proof.stage5_sumcheck_proof,
            stage6_sumcheck_proof: proof.stage6_sumcheck_proof,
            stage7_sumcheck_proof: proof.stage7_sumcheck_proof,
        };

        Self {
            commitments: proof.commitments,
            stages,
            joint_opening_proof: proof.joint_opening_proof,
            untrusted_advice_commitment: proof.untrusted_advice_commitment,
            opening_claims: None,
            blindfold_proof: Some(proof.blindfold_proof),
            trace_length: proof.trace_length,
            ram_K: proof.ram_K,
            rw_config: proof.rw_config.into(),
            one_hot_config: proof.one_hot_config.into(),
            dory_layout: proof.dory_layout.into(),
        }
    }
}
