#[cfg(feature = "zk")]
use std::ops::Range;

use common::jolt_device::JoltDevice;
#[cfg(feature = "field-inline")]
use jolt_claims::protocols::field_inline::FieldInlineCommittedPolynomial;
use jolt_claims::protocols::jolt::{formulas::ra::JoltRaPolynomialLayout, JoltCommittedPolynomial};
use jolt_claims::protocols::jolt::{JoltOneHotConfig, JoltReadWriteConfig};
#[cfg(feature = "zk")]
use jolt_crypto::HomomorphicCommitment;
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
#[cfg(feature = "zk")]
use jolt_field::FromPrimitiveInt;
#[cfg(feature = "zk")]
use jolt_field::RandomSampling;
#[cfg(feature = "zk")]
use jolt_openings::AdditivelyHomomorphic;
use jolt_openings::CommitmentScheme;
#[cfg(feature = "zk")]
use jolt_openings::ZkOpeningScheme;
#[cfg(feature = "zk")]
use jolt_r1cs::constraints::jolt::{
    SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE, SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE,
};
#[cfg(feature = "zk")]
use jolt_r1cs::{ConstraintMatrices, SparseRow, Variable};
use jolt_sumcheck::SumcheckProof;
#[cfg(feature = "zk")]
use jolt_sumcheck::{SumcheckDomain, SumcheckDomainSpec};
#[cfg(feature = "zk")]
use jolt_transcript::Blake2bTranscript;
use jolt_transcript::{AppendToTranscript, Transcript};
use jolt_verifier::proof::{
    ClearProofClaims, JoltCommitments, JoltStageProofs, TracePolynomialOrder,
};
use jolt_verifier::proof::{JoltProof, JoltProofClaims};
#[cfg(feature = "zk")]
use jolt_verifier::stages::stage1::outputs::Stage1PublicOutput;
use jolt_verifier::stages::stage1::{inputs::Stage1Claims, outputs::Stage1ClearOutput};
#[cfg(feature = "zk")]
use jolt_verifier::stages::stage2::outputs::Stage2PublicOutput;
use jolt_verifier::stages::stage2::{inputs::Stage2Claims, outputs::Stage2ClearOutput};
#[cfg(feature = "zk")]
use jolt_verifier::stages::stage3::outputs::Stage3PublicOutput;
use jolt_verifier::stages::stage3::{inputs::Stage3Claims, outputs::Stage3ClearOutput};
#[cfg(feature = "zk")]
use jolt_verifier::stages::stage4::outputs::Stage4PublicOutput;
use jolt_verifier::stages::stage4::{inputs::Stage4Claims, outputs::Stage4ClearOutput};
#[cfg(feature = "zk")]
use jolt_verifier::stages::stage5::outputs::Stage5PublicOutput;
use jolt_verifier::stages::stage5::{inputs::Stage5Claims, outputs::Stage5ClearOutput};
#[cfg(feature = "zk")]
use jolt_verifier::stages::stage6::outputs::Stage6PublicOutput;
use jolt_verifier::stages::stage6::{inputs::Stage6Claims, outputs::Stage6ClearOutput};
#[cfg(feature = "zk")]
use jolt_verifier::stages::stage7::outputs::Stage7PublicOutput;
#[cfg(feature = "zk")]
use jolt_verifier::stages::zk::{
    blindfold,
    inputs::BlindFoldInputs,
    outputs::{zk_stage_outputs, BlindFoldOutput},
};
#[cfg(feature = "zk")]
use jolt_verifier::JoltVerifierPreprocessing;
use jolt_verifier::{
    stages::stage7::{inputs::Stage7Claims, outputs::Stage7ClearOutput},
    CheckedInputs,
};

#[cfg(feature = "zk")]
use crate::committed::CommittedSumcheckWitness;
#[cfg(feature = "zk")]
use crate::stages::stage1::output::Stage1CommittedBoundaryOutput;
use crate::stages::stage1::{output::stage1_claims_from_r1cs_inputs, output::Stage1SumcheckOutput};
#[cfg(feature = "zk")]
use crate::stages::stage2::output::Stage2CommittedBoundaryOutput;
use crate::stages::stage2::output::Stage2ProverOutput;
#[cfg(feature = "zk")]
use crate::stages::stage3::output::Stage3CommittedBoundaryOutput;
use crate::stages::stage3::output::Stage3ProverOutput;
#[cfg(feature = "zk")]
use crate::stages::stage4::output::Stage4CommittedBoundaryOutput;
use crate::stages::stage4::output::Stage4ProverOutput;
#[cfg(feature = "zk")]
use crate::stages::stage5::output::Stage5CommittedBoundaryOutput;
use crate::stages::stage5::output::Stage5ProverOutput;
#[cfg(feature = "zk")]
use crate::stages::stage6::output::Stage6CommittedBoundaryOutput;
use crate::stages::stage6::output::Stage6ProverOutput;
#[cfg(feature = "zk")]
use crate::stages::stage7::output::Stage7CommittedBoundaryOutput;
use crate::stages::stage7::output::Stage7ProverOutput;
#[cfg(feature = "zk")]
use crate::stages::stage8::output::Stage8OpeningStructure;
use crate::stages::stage8::output::Stage8ProofOutput;
#[cfg(feature = "zk")]
use crate::stages::stage8::output::Stage8ZkProofOutput;
use crate::ProverError;
use crate::{absorb_stage0_transcript, Stage0TranscriptContext};
use crate::{
    stages::stage0::{CommitmentStageOutput, CommitmentStageProverState},
    ProverConfig,
};

type Stage8OpeningInputs<PCS> = (
    Vec<<PCS as jolt_crypto::Commitment>::Output>,
    Vec<<PCS as CommitmentScheme>::OpeningHint>,
);

#[cfg(feature = "zk")]
type Stage8ZkOpeningOutput<PCS, VC> = Stage8ZkProofOutput<
    <PCS as CommitmentScheme>::Field,
    <PCS as CommitmentScheme>::Proof,
    <PCS as jolt_crypto::Commitment>::Output,
    <VC as jolt_crypto::Commitment>::Output,
    <PCS as CommitmentScheme>::Field,
>;

type ClearProofAssemblyOutput<PCS, VC> = (
    JoltProof<PCS, VC>,
    Option<<PCS as jolt_crypto::Commitment>::Output>,
);

#[cfg(feature = "zk")]
type ZkProofAssemblyOutput<PCS, VC, ZkProof> = (
    JoltProof<PCS, VC, ZkProof>,
    Option<<PCS as jolt_crypto::Commitment>::Output>,
);

pub(crate) struct ProofAssembly<PCS, VC>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    config: ProverConfig,
    public_io: JoltDevice,
    commitments: Option<JoltCommitments<PCS::Output>>,
    stage0_prover_state: Option<CommitmentStageProverState<PCS::OpeningHint>>,
    clear_stages: ClearStageAssembly<PCS::Field, VC::Output>,
    #[cfg(feature = "zk")]
    zk_stages: ZkStageAssembly<PCS::Field, VC::Output>,
    stage_proofs: Option<JoltStageProofs<PCS::Field, VC>>,
    clear_claims: Option<ClearProofClaims<PCS::Field>>,
    #[cfg(feature = "zk")]
    stage8_zk: Option<Stage8ZkAssembly<PCS::Field, PCS::Output, VC::Output>>,
    #[cfg(feature = "zk")]
    blindfold_witness: BlindFoldWitnessAssembly<PCS::Field, VC::Output>,
    joint_opening_proof: Option<PCS::Proof>,
    trusted_advice_commitment: Option<PCS::Output>,
    untrusted_advice_commitment: Option<PCS::Output>,
    trace_length: Option<usize>,
    ram_k: Option<usize>,
    rw_config: Option<JoltReadWriteConfig>,
    one_hot_config: Option<JoltOneHotConfig>,
    trace_polynomial_order: Option<TracePolynomialOrder>,
}

impl<PCS, VC> ProofAssembly<PCS, VC>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    pub(crate) fn new(config: ProverConfig, public_io: &JoltDevice) -> Self {
        let proof_shape = config.proof_shape;
        Self {
            config,
            public_io: public_io.clone(),
            commitments: None,
            stage0_prover_state: None,
            clear_stages: ClearStageAssembly::default(),
            #[cfg(feature = "zk")]
            zk_stages: ZkStageAssembly::default(),
            stage_proofs: None,
            clear_claims: None,
            #[cfg(feature = "zk")]
            stage8_zk: None,
            #[cfg(feature = "zk")]
            blindfold_witness: BlindFoldWitnessAssembly::default(),
            joint_opening_proof: None,
            trusted_advice_commitment: None,
            untrusted_advice_commitment: None,
            trace_length: proof_shape.map(|shape| shape.trace_length),
            ram_k: proof_shape.map(|shape| shape.ram_k),
            rw_config: proof_shape.map(|shape| shape.rw_config),
            one_hot_config: proof_shape.map(|shape| shape.one_hot_config),
            trace_polynomial_order: proof_shape.map(|shape| shape.trace_polynomial_order),
        }
    }

    pub(crate) fn record_stage0(
        &mut self,
        output: CommitmentStageOutput<PCS>,
    ) -> Result<(), ProverError> {
        if self.commitments.is_some() || self.stage0_prover_state.is_some() {
            return Err(ProverError::InvalidStageRequest {
                reason: "Stage 0 output was already recorded in proof assembly".to_owned(),
            });
        }

        self.commitments = Some(output.commitments);
        self.stage0_prover_state = Some(output.prover_state);
        self.trusted_advice_commitment = output.trusted_advice_commitment;
        self.untrusted_advice_commitment = output.untrusted_advice_commitment;
        Ok(())
    }

    pub(crate) fn absorb_stage0<T>(
        &self,
        checked: &CheckedInputs,
        transcript: &mut T,
    ) -> Result<(), ProverError>
    where
        PCS::Output: AppendToTranscript,
        T: Transcript,
    {
        let commitments =
            self.commitments
                .as_ref()
                .ok_or_else(|| ProverError::InvalidStageRequest {
                    reason: "Stage 0 commitments are required before transcript initialization"
                        .to_owned(),
                })?;
        let context = Stage0TranscriptContext::new(
            self.rw_config
                .ok_or_else(|| missing_metadata("read/write config"))?,
            self.one_hot_config
                .ok_or_else(|| missing_metadata("one-hot config"))?,
            self.trace_polynomial_order
                .ok_or_else(|| missing_metadata("trace polynomial order"))?,
        );
        absorb_stage0_transcript(
            checked,
            context,
            commitments,
            self.untrusted_advice_commitment.as_ref(),
            self.trusted_advice_commitment.as_ref(),
            transcript,
        );
        Ok(())
    }

    pub(crate) fn stage1_clear_output(
        &self,
    ) -> Result<&Stage1ClearOutput<PCS::Field>, ProverError> {
        self.clear_stages
            .stage1
            .as_ref()
            .map(|stage| &stage.verifier_output)
            .ok_or_else(|| ProverError::InvalidStageRequest {
                reason: "Stage 1 clear output is required before Stage 2".to_owned(),
            })
    }

    pub(crate) fn stage2_clear_output(
        &self,
    ) -> Result<&Stage2ClearOutput<PCS::Field>, ProverError> {
        self.clear_stages
            .stage2
            .as_ref()
            .map(|stage| &stage.verifier_output)
            .ok_or_else(|| ProverError::InvalidStageRequest {
                reason: "Stage 2 clear output is required before Stage 3".to_owned(),
            })
    }

    pub(crate) fn stage3_clear_output(
        &self,
    ) -> Result<&Stage3ClearOutput<PCS::Field>, ProverError> {
        self.clear_stages
            .stage3
            .as_ref()
            .map(|stage| &stage.verifier_output)
            .ok_or_else(|| ProverError::InvalidStageRequest {
                reason: "Stage 3 clear output is required before Stage 4".to_owned(),
            })
    }

    pub(crate) fn stage4_clear_output(
        &self,
    ) -> Result<&Stage4ClearOutput<PCS::Field>, ProverError> {
        self.clear_stages
            .stage4
            .as_ref()
            .map(|stage| &stage.verifier_output)
            .ok_or_else(|| ProverError::InvalidStageRequest {
                reason: "Stage 4 clear output is required before Stage 5".to_owned(),
            })
    }

    pub(crate) fn stage5_clear_output(
        &self,
    ) -> Result<&Stage5ClearOutput<PCS::Field>, ProverError> {
        self.clear_stages
            .stage5
            .as_ref()
            .map(|stage| &stage.verifier_output)
            .ok_or_else(|| ProverError::InvalidStageRequest {
                reason: "Stage 5 clear output is required before Stage 6".to_owned(),
            })
    }

    pub(crate) fn stage6_clear_output(
        &self,
    ) -> Result<&Stage6ClearOutput<PCS::Field>, ProverError> {
        self.clear_stages
            .stage6
            .as_ref()
            .map(|stage| &stage.verifier_output)
            .ok_or_else(|| ProverError::InvalidStageRequest {
                reason: "Stage 6 clear output is required before Stage 7".to_owned(),
            })
    }

    pub(crate) fn stage7_clear_output(
        &self,
    ) -> Result<&Stage7ClearOutput<PCS::Field>, ProverError> {
        self.clear_stages
            .stage7
            .as_ref()
            .map(|stage| &stage.verifier_output)
            .ok_or_else(|| ProverError::InvalidStageRequest {
                reason: "Stage 7 clear output is required before Stage 8".to_owned(),
            })
    }

    pub(crate) fn record_stage1_clear(
        &mut self,
        output: Stage1SumcheckOutput<PCS::Field, SumcheckProof<PCS::Field, VC::Output>>,
    ) -> Result<(), ProverError> {
        if self.clear_stages.stage1.is_some() {
            return Err(ProverError::InvalidStageRequest {
                reason: "Stage 1 clear output was already recorded in proof assembly".to_owned(),
            });
        }
        let verifier_output =
            output
                .verifier_output
                .ok_or_else(|| ProverError::InvalidStageRequest {
                    reason: "Stage 1 clear prover did not return verifier output".to_owned(),
                })?;
        let claims = stage1_claims_from_r1cs_inputs(
            output.uniskip_output_claim,
            &output.r1cs_input_claims,
            #[cfg(feature = "field-inline")]
            &output.field_inline_r1cs_input_claims,
        )?;
        self.clear_stages.stage1 = Some(Stage1ClearAssembly {
            verifier_output,
            claims,
            uniskip_proof: output.uniskip_proof,
            sumcheck_proof: output.remainder_proof,
        });
        Ok(())
    }

    pub(crate) fn record_stage2_clear(
        &mut self,
        output: Stage2ProverOutput<PCS::Field, SumcheckProof<PCS::Field, VC::Output>>,
    ) -> Result<(), ProverError> {
        if self.clear_stages.stage2.is_some() {
            return Err(ProverError::InvalidStageRequest {
                reason: "Stage 2 clear output was already recorded in proof assembly".to_owned(),
            });
        }
        self.clear_stages.stage2 = Some(Stage2ClearAssembly {
            verifier_output: output.verifier_output,
            claims: output.claims,
            uniskip_proof: output.product_uniskip_proof,
            sumcheck_proof: output.regular_batch_proof,
        });
        Ok(())
    }

    pub(crate) fn record_stage3_clear(
        &mut self,
        output: Stage3ProverOutput<PCS::Field, SumcheckProof<PCS::Field, VC::Output>>,
    ) -> Result<(), ProverError> {
        if self.clear_stages.stage3.is_some() {
            return Err(ProverError::InvalidStageRequest {
                reason: "Stage 3 clear output was already recorded in proof assembly".to_owned(),
            });
        }
        self.clear_stages.stage3 = Some(Stage3ClearAssembly {
            verifier_output: output.verifier_output,
            claims: output.claims,
            sumcheck_proof: output.stage3_sumcheck_proof,
        });
        Ok(())
    }

    pub(crate) fn record_stage4_clear(
        &mut self,
        output: Stage4ProverOutput<PCS::Field, SumcheckProof<PCS::Field, VC::Output>>,
    ) -> Result<(), ProverError> {
        if self.clear_stages.stage4.is_some() {
            return Err(ProverError::InvalidStageRequest {
                reason: "Stage 4 clear output was already recorded in proof assembly".to_owned(),
            });
        }
        self.clear_stages.stage4 = Some(Stage4ClearAssembly {
            verifier_output: output.verifier_output,
            claims: output.claims,
            sumcheck_proof: output.stage4_sumcheck_proof,
        });
        Ok(())
    }

    pub(crate) fn record_stage5_clear(
        &mut self,
        output: Stage5ProverOutput<PCS::Field, SumcheckProof<PCS::Field, VC::Output>>,
    ) -> Result<(), ProverError> {
        if self.clear_stages.stage5.is_some() {
            return Err(ProverError::InvalidStageRequest {
                reason: "Stage 5 clear output was already recorded in proof assembly".to_owned(),
            });
        }
        self.clear_stages.stage5 = Some(Stage5ClearAssembly {
            verifier_output: output.verifier_output,
            claims: output.claims,
            sumcheck_proof: output.stage5_sumcheck_proof,
        });
        Ok(())
    }

    pub(crate) fn record_stage6_clear(
        &mut self,
        output: Stage6ProverOutput<PCS::Field, SumcheckProof<PCS::Field, VC::Output>>,
    ) -> Result<(), ProverError> {
        if self.clear_stages.stage6.is_some() {
            return Err(ProverError::InvalidStageRequest {
                reason: "Stage 6 clear output was already recorded in proof assembly".to_owned(),
            });
        }
        self.clear_stages.stage6 = Some(Stage6ClearAssembly {
            verifier_output: output.verifier_output,
            claims: output.claims,
            sumcheck_proof: output.stage6_sumcheck_proof,
        });
        Ok(())
    }

    pub(crate) fn record_stage7_clear(
        &mut self,
        output: Stage7ProverOutput<PCS::Field, SumcheckProof<PCS::Field, VC::Output>>,
    ) -> Result<(), ProverError> {
        if self.clear_stages.stage7.is_some() {
            return Err(ProverError::InvalidStageRequest {
                reason: "Stage 7 clear output was already recorded in proof assembly".to_owned(),
            });
        }
        self.clear_stages.stage7 = Some(Stage7ClearAssembly {
            verifier_output: output.verifier_output,
            claims: output.claims,
            sumcheck_proof: output.stage7_sumcheck_proof,
        });
        Ok(())
    }

    #[cfg(feature = "zk")]
    pub(crate) fn record_stage1_committed(
        &mut self,
        output: Stage1CommittedBoundaryOutput<PCS::Field, VC>,
    ) -> Result<(), ProverError> {
        if self.zk_stages.stage1.is_some() {
            return Err(ProverError::InvalidStageRequest {
                reason: "Stage 1 committed output was already recorded in proof assembly"
                    .to_owned(),
            });
        }
        self.zk_stages.stage1 = Some(Stage1ZkAssembly {
            verifier_output: output.verifier_output,
            public: output.public,
            uniskip_proof: output.uniskip_proof,
            sumcheck_proof: output.remainder_proof,
            uniskip_output_claim_values: output.uniskip_output_claim_values,
            remainder_output_claim_values: output.remainder_output_claim_values,
            uniskip_committed_witness: output.uniskip_committed_witness,
            remainder_committed_witness: output.remainder_committed_witness,
        });
        Ok(())
    }

    #[cfg(feature = "zk")]
    pub(crate) fn record_stage2_committed(
        &mut self,
        output: Stage2CommittedBoundaryOutput<PCS::Field, VC>,
    ) -> Result<(), ProverError> {
        if self.zk_stages.stage2.is_some() {
            return Err(ProverError::InvalidStageRequest {
                reason: "Stage 2 committed output was already recorded in proof assembly"
                    .to_owned(),
            });
        }
        self.zk_stages.stage2 = Some(Stage2ZkAssembly {
            verifier_output: output.verifier_output,
            public: output.public,
            uniskip_proof: output.product_uniskip_proof,
            sumcheck_proof: output.regular_batch_proof,
            product_uniskip_output_claim_values: output.product_uniskip_output_claim_values,
            batch_output_claim_values: output.batch_output_claim_values,
            product_uniskip_committed_witness: output.product_uniskip_committed_witness,
            batch_committed_witness: output.batch_committed_witness,
        });
        Ok(())
    }

    #[cfg(feature = "zk")]
    pub(crate) fn record_stage3_committed(
        &mut self,
        output: Stage3CommittedBoundaryOutput<PCS::Field, VC>,
    ) -> Result<(), ProverError> {
        if self.zk_stages.stage3.is_some() {
            return Err(ProverError::InvalidStageRequest {
                reason: "Stage 3 committed output was already recorded in proof assembly"
                    .to_owned(),
            });
        }
        self.zk_stages.stage3 = Some(Stage3ZkAssembly {
            verifier_output: output.verifier_output,
            public: output.public,
            sumcheck_proof: output.stage3_sumcheck_proof,
            output_claim_values: output.output_claim_values,
            committed_witness: output.committed_witness,
        });
        Ok(())
    }

    #[cfg(feature = "zk")]
    pub(crate) fn record_stage4_committed(
        &mut self,
        output: Stage4CommittedBoundaryOutput<PCS::Field, VC>,
    ) -> Result<(), ProverError> {
        if self.zk_stages.stage4.is_some() {
            return Err(ProverError::InvalidStageRequest {
                reason: "Stage 4 committed output was already recorded in proof assembly"
                    .to_owned(),
            });
        }
        self.zk_stages.stage4 = Some(Stage4ZkAssembly {
            verifier_output: output.verifier_output,
            public: output.public,
            sumcheck_proof: output.stage4_sumcheck_proof,
            output_claim_values: output.output_claim_values,
            committed_witness: output.committed_witness,
        });
        Ok(())
    }

    #[cfg(feature = "zk")]
    pub(crate) fn record_stage5_committed(
        &mut self,
        output: Stage5CommittedBoundaryOutput<PCS::Field, VC>,
    ) -> Result<(), ProverError> {
        if self.zk_stages.stage5.is_some() {
            return Err(ProverError::InvalidStageRequest {
                reason: "Stage 5 committed output was already recorded in proof assembly"
                    .to_owned(),
            });
        }
        self.zk_stages.stage5 = Some(Stage5ZkAssembly {
            verifier_output: output.verifier_output,
            public: output.public,
            sumcheck_proof: output.stage5_sumcheck_proof,
            output_claim_values: output.output_claim_values,
            committed_witness: output.committed_witness,
        });
        Ok(())
    }

    #[cfg(feature = "zk")]
    pub(crate) fn record_stage6_committed(
        &mut self,
        output: Stage6CommittedBoundaryOutput<PCS::Field, VC>,
    ) -> Result<(), ProverError> {
        if self.zk_stages.stage6.is_some() {
            return Err(ProverError::InvalidStageRequest {
                reason: "Stage 6 committed output was already recorded in proof assembly"
                    .to_owned(),
            });
        }
        self.zk_stages.stage6 = Some(Stage6ZkAssembly {
            verifier_output: output.verifier_output,
            public: output.public,
            sumcheck_proof: output.stage6_sumcheck_proof,
            output_claim_values: output.output_claim_values,
            committed_witness: output.committed_witness,
        });
        Ok(())
    }

    #[cfg(feature = "zk")]
    pub(crate) fn record_stage7_committed(
        &mut self,
        output: Stage7CommittedBoundaryOutput<PCS::Field, VC>,
    ) -> Result<(), ProverError> {
        if self.zk_stages.stage7.is_some() {
            return Err(ProverError::InvalidStageRequest {
                reason: "Stage 7 committed output was already recorded in proof assembly"
                    .to_owned(),
            });
        }
        self.zk_stages.stage7 = Some(Stage7ZkAssembly {
            verifier_output: output.verifier_output,
            public: output.public,
            sumcheck_proof: output.stage7_sumcheck_proof,
            output_claim_values: output.output_claim_values,
            committed_witness: output.committed_witness,
        });
        Ok(())
    }

    pub(crate) fn assemble_clear_stage_payloads(&mut self) -> Result<(), ProverError> {
        if self.stage_proofs.is_some() || self.clear_claims.is_some() {
            return Err(ProverError::InvalidStageRequest {
                reason: "clear stage proof payloads were already assembled".to_owned(),
            });
        }

        let stage1 =
            self.clear_stages
                .stage1
                .as_ref()
                .ok_or_else(|| ProverError::InvalidStageRequest {
                    reason: "Stage 1 clear assembly is required before proof payload assembly"
                        .to_owned(),
                })?;
        let stage2 =
            self.clear_stages
                .stage2
                .as_ref()
                .ok_or_else(|| ProverError::InvalidStageRequest {
                    reason: "Stage 2 clear assembly is required before proof payload assembly"
                        .to_owned(),
                })?;
        let stage3 =
            self.clear_stages
                .stage3
                .as_ref()
                .ok_or_else(|| ProverError::InvalidStageRequest {
                    reason: "Stage 3 clear assembly is required before proof payload assembly"
                        .to_owned(),
                })?;
        let stage4 =
            self.clear_stages
                .stage4
                .as_ref()
                .ok_or_else(|| ProverError::InvalidStageRequest {
                    reason: "Stage 4 clear assembly is required before proof payload assembly"
                        .to_owned(),
                })?;
        let stage5 =
            self.clear_stages
                .stage5
                .as_ref()
                .ok_or_else(|| ProverError::InvalidStageRequest {
                    reason: "Stage 5 clear assembly is required before proof payload assembly"
                        .to_owned(),
                })?;
        let stage6 =
            self.clear_stages
                .stage6
                .as_ref()
                .ok_or_else(|| ProverError::InvalidStageRequest {
                    reason: "Stage 6 clear assembly is required before proof payload assembly"
                        .to_owned(),
                })?;
        let stage7 =
            self.clear_stages
                .stage7
                .as_ref()
                .ok_or_else(|| ProverError::InvalidStageRequest {
                    reason: "Stage 7 clear assembly is required before proof payload assembly"
                        .to_owned(),
                })?;

        self.stage_proofs = Some(JoltStageProofs {
            stage1_uni_skip_first_round_proof: stage1.uniskip_proof.clone(),
            stage1_sumcheck_proof: stage1.sumcheck_proof.clone(),
            stage2_uni_skip_first_round_proof: stage2.uniskip_proof.clone(),
            stage2_sumcheck_proof: stage2.sumcheck_proof.clone(),
            stage3_sumcheck_proof: stage3.sumcheck_proof.clone(),
            stage4_sumcheck_proof: stage4.sumcheck_proof.clone(),
            stage5_sumcheck_proof: stage5.sumcheck_proof.clone(),
            stage6_sumcheck_proof: stage6.sumcheck_proof.clone(),
            stage7_sumcheck_proof: stage7.sumcheck_proof.clone(),
        });
        self.clear_claims = Some(ClearProofClaims {
            stage1: stage1.claims.clone(),
            stage2: stage2.claims.clone(),
            stage3: stage3.claims.clone(),
            stage4: stage4.claims.clone(),
            stage5: stage5.claims.clone(),
            stage6: stage6.claims.clone(),
            stage7: stage7.claims.clone(),
        });
        Ok(())
    }

    #[cfg(feature = "zk")]
    pub(crate) fn assemble_zk_stage_payloads(&mut self) -> Result<(), ProverError> {
        if self.stage_proofs.is_some() || self.clear_claims.is_some() {
            return Err(ProverError::InvalidStageRequest {
                reason: "ZK stage proof payloads were already assembled".to_owned(),
            });
        }
        if !self.blindfold_witness.committed_sumchecks.is_empty() {
            return Err(ProverError::InvalidStageRequest {
                reason: "BlindFold committed sumcheck witnesses were already assembled".to_owned(),
            });
        }

        let stage1 =
            self.zk_stages
                .stage1
                .as_ref()
                .ok_or_else(|| ProverError::InvalidStageRequest {
                    reason: "Stage 1 committed assembly is required before proof payload assembly"
                        .to_owned(),
                })?;
        let stage2 =
            self.zk_stages
                .stage2
                .as_ref()
                .ok_or_else(|| ProverError::InvalidStageRequest {
                    reason: "Stage 2 committed assembly is required before proof payload assembly"
                        .to_owned(),
                })?;
        let stage3 =
            self.zk_stages
                .stage3
                .as_ref()
                .ok_or_else(|| ProverError::InvalidStageRequest {
                    reason: "Stage 3 committed assembly is required before proof payload assembly"
                        .to_owned(),
                })?;
        let stage4 =
            self.zk_stages
                .stage4
                .as_ref()
                .ok_or_else(|| ProverError::InvalidStageRequest {
                    reason: "Stage 4 committed assembly is required before proof payload assembly"
                        .to_owned(),
                })?;
        let stage5 =
            self.zk_stages
                .stage5
                .as_ref()
                .ok_or_else(|| ProverError::InvalidStageRequest {
                    reason: "Stage 5 committed assembly is required before proof payload assembly"
                        .to_owned(),
                })?;
        let stage6 =
            self.zk_stages
                .stage6
                .as_ref()
                .ok_or_else(|| ProverError::InvalidStageRequest {
                    reason: "Stage 6 committed assembly is required before proof payload assembly"
                        .to_owned(),
                })?;
        let stage7 =
            self.zk_stages
                .stage7
                .as_ref()
                .ok_or_else(|| ProverError::InvalidStageRequest {
                    reason: "Stage 7 committed assembly is required before proof payload assembly"
                        .to_owned(),
                })?;

        self.stage_proofs = Some(JoltStageProofs {
            stage1_uni_skip_first_round_proof: stage1.uniskip_proof.clone(),
            stage1_sumcheck_proof: stage1.sumcheck_proof.clone(),
            stage2_uni_skip_first_round_proof: stage2.uniskip_proof.clone(),
            stage2_sumcheck_proof: stage2.sumcheck_proof.clone(),
            stage3_sumcheck_proof: stage3.sumcheck_proof.clone(),
            stage4_sumcheck_proof: stage4.sumcheck_proof.clone(),
            stage5_sumcheck_proof: stage5.sumcheck_proof.clone(),
            stage6_sumcheck_proof: stage6.sumcheck_proof.clone(),
            stage7_sumcheck_proof: stage7.sumcheck_proof.clone(),
        });
        self.blindfold_witness.committed_sumchecks = vec![
            stage1.uniskip_committed_witness.clone(),
            stage1.remainder_committed_witness.clone(),
            stage2.product_uniskip_committed_witness.clone(),
            stage2.batch_committed_witness.clone(),
            stage3.committed_witness.clone(),
            stage4.committed_witness.clone(),
            stage5.committed_witness.clone(),
            stage6.committed_witness.clone(),
            stage7.committed_witness.clone(),
        ];
        Ok(())
    }

    pub(crate) fn stage8_clear_opening_inputs(
        &self,
        layout: JoltRaPolynomialLayout,
    ) -> Result<Stage8OpeningInputs<PCS>, ProverError> {
        let commitments =
            self.commitments
                .as_ref()
                .ok_or_else(|| ProverError::InvalidStageRequest {
                    reason: "Stage 0 commitments are required before Stage 8".to_owned(),
                })?;
        let prover_state =
            self.stage0_prover_state
                .as_ref()
                .ok_or_else(|| ProverError::InvalidStageRequest {
                    reason: "Stage 0 opening hints are required before Stage 8".to_owned(),
                })?;

        let mut ordered_commitments = Vec::with_capacity(
            2 + field_inline_final_opening_count()
                + layout.total()
                + usize::from(self.trusted_advice_commitment.is_some())
                + usize::from(self.untrusted_advice_commitment.is_some()),
        );
        let mut opening_hints = Vec::with_capacity(ordered_commitments.capacity());
        push_jolt_stage8_opening(
            &mut ordered_commitments,
            &mut opening_hints,
            &commitments.ram_inc,
            prover_state,
            JoltCommittedPolynomial::RamInc,
        )?;
        push_jolt_stage8_opening(
            &mut ordered_commitments,
            &mut opening_hints,
            &commitments.rd_inc,
            prover_state,
            JoltCommittedPolynomial::RdInc,
        )?;
        #[cfg(feature = "field-inline")]
        push_field_inline_stage8_opening(
            &mut ordered_commitments,
            &mut opening_hints,
            &commitments.field_inline.field_registers.rd_inc,
            prover_state,
            FieldInlineCommittedPolynomial::FieldRdInc,
        )?;
        for (index, commitment) in commitments.ra.instruction.iter().enumerate() {
            push_jolt_stage8_opening(
                &mut ordered_commitments,
                &mut opening_hints,
                commitment,
                prover_state,
                JoltCommittedPolynomial::InstructionRa(index),
            )?;
        }
        for (index, commitment) in commitments.ra.bytecode.iter().enumerate() {
            push_jolt_stage8_opening(
                &mut ordered_commitments,
                &mut opening_hints,
                commitment,
                prover_state,
                JoltCommittedPolynomial::BytecodeRa(index),
            )?;
        }
        for (index, commitment) in commitments.ra.ram.iter().enumerate() {
            push_jolt_stage8_opening(
                &mut ordered_commitments,
                &mut opening_hints,
                commitment,
                prover_state,
                JoltCommittedPolynomial::RamRa(index),
            )?;
        }
        if let Some(commitment) = &self.trusted_advice_commitment {
            push_jolt_stage8_opening(
                &mut ordered_commitments,
                &mut opening_hints,
                commitment,
                prover_state,
                JoltCommittedPolynomial::TrustedAdvice,
            )?;
        }
        if let Some(commitment) = &self.untrusted_advice_commitment {
            push_jolt_stage8_opening(
                &mut ordered_commitments,
                &mut opening_hints,
                commitment,
                prover_state,
                JoltCommittedPolynomial::UntrustedAdvice,
            )?;
        }

        Ok((ordered_commitments, opening_hints))
    }

    pub(crate) fn record_stage8_clear(
        &mut self,
        output: Stage8ProofOutput<PCS::Field, PCS::Proof, PCS::Output>,
    ) -> Result<(), ProverError> {
        if self.joint_opening_proof.is_some() {
            return Err(ProverError::InvalidStageRequest {
                reason: "Stage 8 joint opening proof was already recorded".to_owned(),
            });
        }
        let Stage8ProofOutput {
            structure,
            joint_opening_proof,
            joint_commitment,
        } = output;
        let _ = (structure, joint_commitment);
        self.joint_opening_proof = Some(joint_opening_proof);
        Ok(())
    }

    #[cfg(feature = "zk")]
    pub(crate) fn record_stage8_zk(
        &mut self,
        output: Stage8ZkOpeningOutput<PCS, VC>,
    ) -> Result<(), ProverError>
    where
        PCS: ZkOpeningScheme<
            HidingCommitment = VC::Output,
            Blind = <PCS as CommitmentScheme>::Field,
        >,
    {
        if self.joint_opening_proof.is_some() || self.stage8_zk.is_some() {
            return Err(ProverError::InvalidStageRequest {
                reason: "Stage 8 ZK opening output was already recorded".to_owned(),
            });
        }
        let Stage8ZkProofOutput {
            structure,
            joint_opening_proof,
            joint_commitment,
            hiding_evaluation_commitment,
            hiding_evaluation_blind,
        } = output;
        self.joint_opening_proof = Some(joint_opening_proof);
        self.blindfold_witness.stage8_hiding_evaluation_commitment =
            Some(hiding_evaluation_commitment);
        self.blindfold_witness.stage8_hiding_evaluation_blind = Some(hiding_evaluation_blind);
        self.stage8_zk = Some(Stage8ZkAssembly {
            structure,
            joint_commitment,
            hiding_evaluation_commitment,
        });
        Ok(())
    }

    #[cfg(feature = "zk")]
    pub(crate) fn build_blindfold_protocol(
        &self,
        preprocessing: &JoltVerifierPreprocessing<PCS, VC>,
    ) -> Result<BlindFoldOutput<PCS::Field, VC::Output>, ProverError>
    where
        PCS: AdditivelyHomomorphic
            + ZkOpeningScheme<HidingCommitment = VC::Output, Blind = <PCS as CommitmentScheme>::Field>,
        PCS::Output: AppendToTranscript + HomomorphicCommitment<PCS::Field>,
    {
        if !self.config.features.zk {
            return Err(ProverError::InvalidStageRequest {
                reason: "BlindFold protocol construction requires ZK proof assembly".to_owned(),
            });
        }
        if !self
            .stage8_zk
            .as_ref()
            .is_some_and(Stage8ZkAssembly::is_ready)
        {
            return Err(missing_proof_component("Stage 8 ZK assembly"));
        }

        let proof = self.zk_proof_shell()?;
        let jolt_verifier::PreStage1VerifierState {
            checked,
            mut transcript,
        } = jolt_verifier::verify_until_stage1::<
            PCS,
            VC,
            Blake2bTranscript<PCS::Field>,
            (),
            jolt_verifier::NoPcsAssist,
        >(
            preprocessing,
            &self.public_io,
            &proof,
            self.trusted_advice_commitment.as_ref(),
            true,
        )?;

        let stage1 = jolt_verifier::stages::stage1::verify(
            &checked,
            preprocessing,
            &proof,
            &mut transcript,
        )?;
        let stage2 = jolt_verifier::stages::stage2::verify(
            &checked,
            preprocessing,
            &proof,
            &mut transcript,
            jolt_verifier::stages::stage2::deps(&stage1),
        )?;
        let stage3 = jolt_verifier::stages::stage3::verify(
            &checked,
            preprocessing,
            &proof,
            &mut transcript,
            jolt_verifier::stages::stage3::deps(&stage1, &stage2)?,
        )?;
        let stage4 = jolt_verifier::stages::stage4::verify(
            &checked,
            preprocessing,
            &proof,
            &mut transcript,
            jolt_verifier::stages::stage4::deps(&stage2, &stage3)?,
        )?;
        let stage5 = jolt_verifier::stages::stage5::verify(
            &checked,
            preprocessing,
            &proof,
            &mut transcript,
            jolt_verifier::stages::stage5::deps(&stage2, &stage4)?,
        )?;
        let stage6 = jolt_verifier::stages::stage6::verify(
            &checked,
            preprocessing,
            &proof,
            &mut transcript,
            jolt_verifier::stages::stage6::deps(&stage1, &stage2, &stage3, &stage4, &stage5)?,
        )?;
        let stage7 = jolt_verifier::stages::stage7::verify(
            &checked,
            preprocessing,
            &proof,
            &mut transcript,
            jolt_verifier::stages::stage7::deps(&stage4, &stage6)?,
        )?;
        let stage8 = jolt_verifier::stages::stage8::verify(
            &checked,
            &proof.protocol,
            preprocessing,
            &proof,
            self.trusted_advice_commitment.as_ref(),
            &mut transcript,
            jolt_verifier::stages::stage8::deps(&stage6, &stage7)?,
        )?;

        let zk_stages = zk_stage_outputs::<PCS, VC>(
            &stage1, &stage2, &stage3, &stage4, &stage5, &stage6, &stage7, &stage8,
        )?;
        Ok(blindfold::build(BlindFoldInputs {
            checked: &checked,
            preprocessing,
            proof: &proof,
            stage1: zk_stages.stage1,
            stage2: zk_stages.stage2,
            stage3: zk_stages.stage3,
            stage4: zk_stages.stage4,
            stage5: zk_stages.stage5,
            stage6: zk_stages.stage6,
            stage7: zk_stages.stage7,
            stage8: zk_stages.stage8,
        })?)
    }

    #[cfg(feature = "zk")]
    fn zk_proof_shell(&self) -> Result<JoltProof<PCS, VC, ()>, ProverError> {
        let commitments = self
            .commitments
            .clone()
            .ok_or_else(|| missing_proof_component("commitments"))?;
        let stages = self
            .stage_proofs
            .clone()
            .ok_or_else(|| missing_proof_component("stage proofs"))?;
        let joint_opening_proof = self
            .joint_opening_proof
            .clone()
            .ok_or_else(|| missing_proof_component("joint opening proof"))?;
        let trace_length = self
            .trace_length
            .ok_or_else(|| missing_metadata("trace length"))?;
        let ram_k = self
            .ram_k
            .ok_or_else(|| missing_metadata("RAM domain size"))?;
        let rw_config = self
            .rw_config
            .ok_or_else(|| missing_metadata("read/write config"))?;
        let one_hot_config = self
            .one_hot_config
            .ok_or_else(|| missing_metadata("one-hot config"))?;
        let trace_polynomial_order = self
            .trace_polynomial_order
            .ok_or_else(|| missing_metadata("trace polynomial order"))?;

        Ok(JoltProof::<PCS, VC, ()>::new(
            commitments,
            stages,
            joint_opening_proof,
            self.untrusted_advice_commitment.clone(),
            JoltProofClaims::Zk {
                blindfold_proof: (),
            },
            trace_length,
            ram_k,
            rw_config,
            one_hot_config,
            trace_polynomial_order,
        ))
    }

    #[cfg(feature = "zk")]
    pub(crate) fn assemble_blindfold_witness<R>(
        &self,
        blindfold: &BlindFoldOutput<PCS::Field, VC::Output>,
        rng: &mut R,
    ) -> Result<BlindFoldProverWitness<PCS::Field>, ProverError>
    where
        R: rand_core::RngCore,
    {
        let protocol = &blindfold.protocol;
        let row_len = protocol.dimensions.witness.row_len;
        let row_count = protocol.dimensions.witness.row_count;
        let value_count =
            row_len
                .checked_mul(row_count)
                .ok_or_else(|| ProverError::InvalidStageRequest {
                    reason: "BlindFold witness dimensions overflow".to_owned(),
                })?;
        if protocol.r1cs.num_vars > value_count + 1 {
            return Err(ProverError::InvalidStageRequest {
                reason: format!(
                    "BlindFold R1CS variable count {} exceeds padded witness capacity {}",
                    protocol.r1cs.num_vars,
                    value_count + 1
                ),
            });
        }

        let stage8 = self
            .stage8_zk
            .as_ref()
            .ok_or_else(|| missing_proof_component("Stage 8 ZK assembly"))?;
        let eval_outputs = vec![stage8.structure.joint_claim];
        let eval_blind = self
            .blindfold_witness
            .stage8_hiding_evaluation_blind
            .ok_or_else(|| missing_proof_component("Stage 8 hidden evaluation blinding"))?;
        let eval_blindings = vec![eval_blind];
        if protocol.eval_commitments.len() != eval_outputs.len() {
            return Err(ProverError::InvalidStageRequest {
                reason: format!(
                    "BlindFold final-opening count mismatch: expected {}, got {}",
                    eval_outputs.len(),
                    protocol.eval_commitments.len()
                ),
            });
        }

        let committed_sumchecks = &self.blindfold_witness.committed_sumchecks;
        if committed_sumchecks.len() != protocol.layout.stages.len()
            || committed_sumchecks.len() != protocol.sumcheck_consistency.len()
        {
            return Err(ProverError::InvalidStageRequest {
                reason: format!(
                    "BlindFold committed sumcheck count mismatch: witnesses {}, layouts {}, protocol {}",
                    committed_sumchecks.len(),
                    protocol.layout.stages.len(),
                    protocol.sumcheck_consistency.len()
                ),
            });
        }

        let mut witness_values = vec![None; value_count + 1];
        witness_values[0] = Some(PCS::Field::from_u64(1));
        let mut blindings = vec![PCS::Field::from_u64(0); row_count];

        assign_committed_blindfold_rows(
            &mut witness_values,
            &mut blindings,
            row_len,
            protocol.dimensions.witness_rows.coefficients.clone(),
            committed_sumchecks,
            BlindFoldCommittedRows::Coefficients,
        )?;
        assign_committed_blindfold_rows(
            &mut witness_values,
            &mut blindings,
            row_len,
            protocol.dimensions.witness_rows.output_claims.clone(),
            committed_sumchecks,
            BlindFoldCommittedRows::OutputClaims,
        )?;

        let domains = blindfold_sumcheck_domains();
        if domains.len() != committed_sumchecks.len() {
            return Err(ProverError::InvalidStageRequest {
                reason: format!(
                    "BlindFold domain count mismatch: expected {}, got {}",
                    committed_sumchecks.len(),
                    domains.len()
                ),
            });
        }
        for ((stage_layout, consistency), (witness, domain)) in protocol
            .layout
            .stages
            .iter()
            .zip(&protocol.sumcheck_consistency)
            .zip(committed_sumchecks.iter().zip(domains))
        {
            assign_sumcheck_layout_witness(
                &mut witness_values,
                &stage_layout.sumcheck,
                consistency,
                witness,
                domain,
            )?;
        }

        let final_coordinates = protocol
            .final_opening_witness_coordinates()
            .map_err(|error| ProverError::InvalidStageRequest {
                reason: format!("BlindFold final-opening coordinate derivation failed: {error}"),
            })?;
        if final_coordinates.len() != eval_outputs.len() {
            return Err(ProverError::InvalidStageRequest {
                reason: format!(
                    "BlindFold final-opening coordinate count mismatch: expected {}, got {}",
                    eval_outputs.len(),
                    final_coordinates.len()
                ),
            });
        }
        for (index, coordinates) in final_coordinates.iter().enumerate() {
            if let Some(coordinate) = coordinates.evaluation {
                assign_witness_cell(
                    &mut witness_values,
                    row_len,
                    coordinate.row,
                    coordinate.column,
                    eval_outputs[index],
                    "BlindFold final-opening evaluation",
                )?;
            }
            if let Some(coordinate) = coordinates.blinding {
                assign_witness_cell(
                    &mut witness_values,
                    row_len,
                    coordinate.row,
                    coordinate.column,
                    eval_blindings[index],
                    "BlindFold final-opening blinding",
                )?;
            }
        }

        complete_blindfold_auxiliary_witness(&protocol.r1cs, &mut witness_values)?;
        for row in protocol.dimensions.witness_rows.auxiliary.clone() {
            blindings[row] = PCS::Field::random(&mut *rng);
        }

        let flat_witness = witness_values
            .into_iter()
            .map(Option::unwrap_or_default)
            .collect::<Vec<_>>();
        protocol
            .r1cs
            .check_witness(&flat_witness)
            .map_err(|constraint| ProverError::InvalidStageRequest {
                reason: format!("BlindFold witness does not satisfy constraint {constraint}"),
            })?;
        let rows = flat_witness[1..]
            .chunks(row_len)
            .map(<[PCS::Field]>::to_vec)
            .collect::<Vec<_>>();
        if rows.len() != row_count {
            return Err(ProverError::InvalidStageRequest {
                reason: format!(
                    "BlindFold witness row count mismatch: expected {row_count}, got {}",
                    rows.len()
                ),
            });
        }

        Ok(BlindFoldProverWitness {
            rows,
            blindings,
            eval_outputs,
            eval_blindings,
        })
    }

    pub(crate) fn into_clear_proof(self) -> Result<ClearProofAssemblyOutput<PCS, VC>, ProverError> {
        if self.config.features.zk {
            return Err(ProverError::InvalidStageRequest {
                reason: "clear proof assembly cannot finish a ZK proof".to_owned(),
            });
        }

        let commitments = self
            .commitments
            .ok_or_else(|| missing_proof_component("commitments"))?;
        let stages = self
            .stage_proofs
            .ok_or_else(|| missing_proof_component("stage proofs"))?;
        let joint_opening_proof = self
            .joint_opening_proof
            .ok_or_else(|| missing_proof_component("joint opening proof"))?;
        let claims = self
            .clear_claims
            .ok_or_else(|| missing_proof_component("clear proof claims"))?;
        let trace_length = self
            .trace_length
            .ok_or_else(|| missing_metadata("trace length"))?;
        let ram_k = self
            .ram_k
            .ok_or_else(|| missing_metadata("RAM domain size"))?;
        let rw_config = self
            .rw_config
            .ok_or_else(|| missing_metadata("read/write config"))?;
        let one_hot_config = self
            .one_hot_config
            .ok_or_else(|| missing_metadata("one-hot config"))?;
        let trace_polynomial_order = self
            .trace_polynomial_order
            .ok_or_else(|| missing_metadata("trace polynomial order"))?;

        let proof = JoltProof::new(
            commitments,
            stages,
            joint_opening_proof,
            self.untrusted_advice_commitment,
            JoltProofClaims::Clear(claims),
            trace_length,
            ram_k,
            rw_config,
            one_hot_config,
            trace_polynomial_order,
        );
        Ok((proof, self.trusted_advice_commitment))
    }

    #[cfg(feature = "zk")]
    pub(crate) fn into_zk_proof<ZkProof>(
        self,
        blindfold_proof: ZkProof,
    ) -> Result<ZkProofAssemblyOutput<PCS, VC, ZkProof>, ProverError> {
        if !self.config.features.zk {
            return Err(ProverError::InvalidStageRequest {
                reason: "ZK proof assembly cannot finish a clear proof".to_owned(),
            });
        }

        let commitments = self
            .commitments
            .ok_or_else(|| missing_proof_component("commitments"))?;
        let stages = self
            .stage_proofs
            .ok_or_else(|| missing_proof_component("stage proofs"))?;
        let joint_opening_proof = self
            .joint_opening_proof
            .ok_or_else(|| missing_proof_component("joint opening proof"))?;
        if !self.blindfold_witness.is_ready() {
            return Err(missing_proof_component("BlindFold witness material"));
        }
        let trace_length = self
            .trace_length
            .ok_or_else(|| missing_metadata("trace length"))?;
        let ram_k = self
            .ram_k
            .ok_or_else(|| missing_metadata("RAM domain size"))?;
        let rw_config = self
            .rw_config
            .ok_or_else(|| missing_metadata("read/write config"))?;
        let one_hot_config = self
            .one_hot_config
            .ok_or_else(|| missing_metadata("one-hot config"))?;
        let trace_polynomial_order = self
            .trace_polynomial_order
            .ok_or_else(|| missing_metadata("trace polynomial order"))?;

        let proof = JoltProof::new(
            commitments,
            stages,
            joint_opening_proof,
            self.untrusted_advice_commitment,
            JoltProofClaims::Zk { blindfold_proof },
            trace_length,
            ram_k,
            rw_config,
            one_hot_config,
            trace_polynomial_order,
        );
        Ok((proof, self.trusted_advice_commitment))
    }

    pub(crate) fn next_frontier(&self) -> &'static str {
        let _protocol = self.config.protocol;
        let _public_io = &self.public_io;
        let _trusted_advice_present = self.trusted_advice_commitment.is_some();
        let _untrusted_advice_present = self.untrusted_advice_commitment.is_some();

        if self.commitments.is_none() || self.stage0_prover_state.is_none() {
            return "full Jolt proof: Stage 0 assembly";
        }
        if self.config.features.zk {
            #[cfg(feature = "zk")]
            {
                if !self.zk_stages.stage1_ready() {
                    return "full Jolt proof: Stage 1 committed proof assembly";
                }
                if !self.zk_stages.stage2_ready() {
                    return "full Jolt proof: Stage 2 committed proof assembly";
                }
                if !self.zk_stages.stage3_ready() {
                    return "full Jolt proof: Stage 3 committed proof assembly";
                }
                if !self.zk_stages.stage4_ready() {
                    return "full Jolt proof: Stage 4 committed proof assembly";
                }
                if !self.zk_stages.stage5_ready() {
                    return "full Jolt proof: Stage 5 committed proof assembly";
                }
                if !self.zk_stages.stage6_ready() {
                    return "full Jolt proof: Stage 6 committed proof assembly";
                }
                if !self.zk_stages.stage7_ready() {
                    return "full Jolt proof: Stage 7 committed proof assembly";
                }
                if self.stage_proofs.is_none() {
                    return "full Jolt proof: committed JoltStageProofs assembly";
                }
                if self.joint_opening_proof.is_none()
                    || !self
                        .stage8_zk
                        .as_ref()
                        .is_some_and(Stage8ZkAssembly::is_ready)
                {
                    return "full Jolt proof: Stage 8 ZK joint opening";
                }
                if !self.blindfold_witness.is_ready() {
                    return "full Jolt proof: BlindFold witness assembly";
                }
                if !self.metadata_ready() {
                    return "full Jolt proof: trace/config metadata";
                }
                return "full Jolt proof: BlindFold proof generation";
            }
            #[cfg(not(feature = "zk"))]
            {
                return "full Jolt proof: ZK feature not compiled";
            }
        }
        if !self.clear_stages.stage1_ready() {
            return "full Jolt proof: Stage 1 clear proof assembly";
        }
        if !self.clear_stages.stage2_ready() {
            return "full Jolt proof: Stage 2 clear proof assembly";
        }
        if !self.clear_stages.stage3_ready() {
            return "full Jolt proof: Stage 3 clear proof assembly";
        }
        if !self.clear_stages.stage4_ready() {
            return "full Jolt proof: Stage 4 clear proof assembly";
        }
        if !self.clear_stages.stage5_ready() {
            return "full Jolt proof: Stage 5 clear proof assembly";
        }
        if !self.clear_stages.stage6_ready() {
            return "full Jolt proof: Stage 6 clear proof assembly";
        }
        if !self.clear_stages.stage7_ready() {
            return "full Jolt proof: Stage 7 clear proof assembly";
        }
        if self.stage_proofs.is_none() {
            return "full Jolt proof: JoltStageProofs assembly";
        }
        if !self.claims_ready() {
            return "full Jolt proof: proof-claim assembly";
        }
        if self.joint_opening_proof.is_none() {
            return "full Jolt proof: Stage 8 joint opening";
        }
        if !self.metadata_ready() {
            return "full Jolt proof: trace/config metadata";
        }

        "full Jolt proof"
    }

    fn claims_ready(&self) -> bool {
        if self.config.features.zk {
            #[cfg(feature = "zk")]
            {
                return self.blindfold_witness.is_ready();
            }
            #[cfg(not(feature = "zk"))]
            {
                return false;
            }
        }

        self.clear_claims.is_some()
    }

    fn metadata_ready(&self) -> bool {
        self.trace_length.is_some()
            && self.ram_k.is_some()
            && self.rw_config.is_some()
            && self.one_hot_config.is_some()
            && self.trace_polynomial_order.is_some()
    }
}

struct ClearStageAssembly<F, C>
where
    F: Field,
{
    stage1: Option<Stage1ClearAssembly<F, C>>,
    stage2: Option<Stage2ClearAssembly<F, C>>,
    stage3: Option<Stage3ClearAssembly<F, C>>,
    stage4: Option<Stage4ClearAssembly<F, C>>,
    stage5: Option<Stage5ClearAssembly<F, C>>,
    stage6: Option<Stage6ClearAssembly<F, C>>,
    stage7: Option<Stage7ClearAssembly<F, C>>,
}

impl<F, C> Default for ClearStageAssembly<F, C>
where
    F: Field,
{
    fn default() -> Self {
        Self {
            stage1: None,
            stage2: None,
            stage3: None,
            stage4: None,
            stage5: None,
            stage6: None,
            stage7: None,
        }
    }
}

impl<F, C> ClearStageAssembly<F, C>
where
    F: Field,
{
    fn stage1_ready(&self) -> bool {
        self.stage1
            .as_ref()
            .is_some_and(Stage1ClearAssembly::is_ready)
    }

    fn stage2_ready(&self) -> bool {
        self.stage2
            .as_ref()
            .is_some_and(Stage2ClearAssembly::is_ready)
    }

    fn stage3_ready(&self) -> bool {
        self.stage3
            .as_ref()
            .is_some_and(Stage3ClearAssembly::is_ready)
    }

    fn stage4_ready(&self) -> bool {
        self.stage4
            .as_ref()
            .is_some_and(Stage4ClearAssembly::is_ready)
    }

    fn stage5_ready(&self) -> bool {
        self.stage5
            .as_ref()
            .is_some_and(Stage5ClearAssembly::is_ready)
    }

    fn stage6_ready(&self) -> bool {
        self.stage6
            .as_ref()
            .is_some_and(Stage6ClearAssembly::is_ready)
    }

    fn stage7_ready(&self) -> bool {
        self.stage7
            .as_ref()
            .is_some_and(Stage7ClearAssembly::is_ready)
    }
}

#[cfg(feature = "zk")]
struct ZkStageAssembly<F, C>
where
    F: Field,
{
    stage1: Option<Stage1ZkAssembly<F, C>>,
    stage2: Option<Stage2ZkAssembly<F, C>>,
    stage3: Option<Stage3ZkAssembly<F, C>>,
    stage4: Option<Stage4ZkAssembly<F, C>>,
    stage5: Option<Stage5ZkAssembly<F, C>>,
    stage6: Option<Stage6ZkAssembly<F, C>>,
    stage7: Option<Stage7ZkAssembly<F, C>>,
}

#[cfg(feature = "zk")]
impl<F, C> Default for ZkStageAssembly<F, C>
where
    F: Field,
{
    fn default() -> Self {
        Self {
            stage1: None,
            stage2: None,
            stage3: None,
            stage4: None,
            stage5: None,
            stage6: None,
            stage7: None,
        }
    }
}

#[cfg(feature = "zk")]
impl<F, C> ZkStageAssembly<F, C>
where
    F: Field,
{
    fn stage1_ready(&self) -> bool {
        self.stage1.as_ref().is_some_and(Stage1ZkAssembly::is_ready)
    }

    fn stage2_ready(&self) -> bool {
        self.stage2.as_ref().is_some_and(Stage2ZkAssembly::is_ready)
    }

    fn stage3_ready(&self) -> bool {
        self.stage3.as_ref().is_some_and(Stage3ZkAssembly::is_ready)
    }

    fn stage4_ready(&self) -> bool {
        self.stage4.as_ref().is_some_and(Stage4ZkAssembly::is_ready)
    }

    fn stage5_ready(&self) -> bool {
        self.stage5.as_ref().is_some_and(Stage5ZkAssembly::is_ready)
    }

    fn stage6_ready(&self) -> bool {
        self.stage6.as_ref().is_some_and(Stage6ZkAssembly::is_ready)
    }

    fn stage7_ready(&self) -> bool {
        self.stage7.as_ref().is_some_and(Stage7ZkAssembly::is_ready)
    }
}

struct Stage1ClearAssembly<F, C>
where
    F: Field,
{
    verifier_output: Stage1ClearOutput<F>,
    claims: Stage1Claims<F>,
    uniskip_proof: SumcheckProof<F, C>,
    sumcheck_proof: SumcheckProof<F, C>,
}

impl<F, C> Stage1ClearAssembly<F, C>
where
    F: Field,
{
    fn is_ready(&self) -> bool {
        let _ = (
            &self.verifier_output,
            &self.claims,
            &self.uniskip_proof,
            &self.sumcheck_proof,
        );
        true
    }
}

struct Stage2ClearAssembly<F, C>
where
    F: Field,
{
    verifier_output: Stage2ClearOutput<F>,
    claims: Stage2Claims<F>,
    uniskip_proof: SumcheckProof<F, C>,
    sumcheck_proof: SumcheckProof<F, C>,
}

impl<F, C> Stage2ClearAssembly<F, C>
where
    F: Field,
{
    fn is_ready(&self) -> bool {
        let _ = (
            &self.verifier_output,
            &self.claims,
            &self.uniskip_proof,
            &self.sumcheck_proof,
        );
        true
    }
}

struct Stage3ClearAssembly<F, C>
where
    F: Field,
{
    verifier_output: Stage3ClearOutput<F>,
    claims: Stage3Claims<F>,
    sumcheck_proof: SumcheckProof<F, C>,
}

impl<F, C> Stage3ClearAssembly<F, C>
where
    F: Field,
{
    fn is_ready(&self) -> bool {
        let _ = (&self.verifier_output, &self.claims, &self.sumcheck_proof);
        true
    }
}

struct Stage4ClearAssembly<F, C>
where
    F: Field,
{
    verifier_output: Stage4ClearOutput<F>,
    claims: Stage4Claims<F>,
    sumcheck_proof: SumcheckProof<F, C>,
}

impl<F, C> Stage4ClearAssembly<F, C>
where
    F: Field,
{
    fn is_ready(&self) -> bool {
        let _ = (&self.verifier_output, &self.claims, &self.sumcheck_proof);
        true
    }
}

struct Stage5ClearAssembly<F, C>
where
    F: Field,
{
    verifier_output: Stage5ClearOutput<F>,
    claims: Stage5Claims<F>,
    sumcheck_proof: SumcheckProof<F, C>,
}

impl<F, C> Stage5ClearAssembly<F, C>
where
    F: Field,
{
    fn is_ready(&self) -> bool {
        let _ = (&self.verifier_output, &self.claims, &self.sumcheck_proof);
        true
    }
}

struct Stage6ClearAssembly<F, C>
where
    F: Field,
{
    verifier_output: Stage6ClearOutput<F>,
    claims: Stage6Claims<F>,
    sumcheck_proof: SumcheckProof<F, C>,
}

impl<F, C> Stage6ClearAssembly<F, C>
where
    F: Field,
{
    fn is_ready(&self) -> bool {
        let _ = (&self.verifier_output, &self.claims, &self.sumcheck_proof);
        true
    }
}

struct Stage7ClearAssembly<F, C>
where
    F: Field,
{
    verifier_output: Stage7ClearOutput<F>,
    claims: Stage7Claims<F>,
    sumcheck_proof: SumcheckProof<F, C>,
}

impl<F, C> Stage7ClearAssembly<F, C>
where
    F: Field,
{
    fn is_ready(&self) -> bool {
        let _ = (&self.verifier_output, &self.claims, &self.sumcheck_proof);
        true
    }
}

#[cfg(feature = "zk")]
struct Stage1ZkAssembly<F, C>
where
    F: Field,
{
    verifier_output: Stage1ClearOutput<F>,
    public: Stage1PublicOutput<F>,
    uniskip_proof: SumcheckProof<F, C>,
    sumcheck_proof: SumcheckProof<F, C>,
    uniskip_output_claim_values: Vec<F>,
    remainder_output_claim_values: Vec<F>,
    uniskip_committed_witness: CommittedSumcheckWitness<F>,
    remainder_committed_witness: CommittedSumcheckWitness<F>,
}

#[cfg(feature = "zk")]
impl<F, C> Stage1ZkAssembly<F, C>
where
    F: Field,
{
    fn is_ready(&self) -> bool {
        let _ = (
            &self.verifier_output,
            &self.public,
            &self.uniskip_proof,
            &self.sumcheck_proof,
            &self.uniskip_output_claim_values,
            &self.remainder_output_claim_values,
            &self.uniskip_committed_witness,
            &self.remainder_committed_witness,
        );
        true
    }
}

#[cfg(feature = "zk")]
struct Stage2ZkAssembly<F, C>
where
    F: Field,
{
    verifier_output: Stage2ClearOutput<F>,
    public: Stage2PublicOutput<F>,
    uniskip_proof: SumcheckProof<F, C>,
    sumcheck_proof: SumcheckProof<F, C>,
    product_uniskip_output_claim_values: Vec<F>,
    batch_output_claim_values: Vec<F>,
    product_uniskip_committed_witness: CommittedSumcheckWitness<F>,
    batch_committed_witness: CommittedSumcheckWitness<F>,
}

#[cfg(feature = "zk")]
impl<F, C> Stage2ZkAssembly<F, C>
where
    F: Field,
{
    fn is_ready(&self) -> bool {
        let _ = (
            &self.verifier_output,
            &self.public,
            &self.uniskip_proof,
            &self.sumcheck_proof,
            &self.product_uniskip_output_claim_values,
            &self.batch_output_claim_values,
            &self.product_uniskip_committed_witness,
            &self.batch_committed_witness,
        );
        true
    }
}

#[cfg(feature = "zk")]
struct Stage3ZkAssembly<F, C>
where
    F: Field,
{
    verifier_output: Stage3ClearOutput<F>,
    public: Stage3PublicOutput<F>,
    sumcheck_proof: SumcheckProof<F, C>,
    output_claim_values: Vec<F>,
    committed_witness: CommittedSumcheckWitness<F>,
}

#[cfg(feature = "zk")]
impl<F, C> Stage3ZkAssembly<F, C>
where
    F: Field,
{
    fn is_ready(&self) -> bool {
        let _ = (
            &self.verifier_output,
            &self.public,
            &self.sumcheck_proof,
            &self.output_claim_values,
            &self.committed_witness,
        );
        true
    }
}

#[cfg(feature = "zk")]
struct Stage4ZkAssembly<F, C>
where
    F: Field,
{
    verifier_output: Stage4ClearOutput<F>,
    public: Stage4PublicOutput<F>,
    sumcheck_proof: SumcheckProof<F, C>,
    output_claim_values: Vec<F>,
    committed_witness: CommittedSumcheckWitness<F>,
}

#[cfg(feature = "zk")]
impl<F, C> Stage4ZkAssembly<F, C>
where
    F: Field,
{
    fn is_ready(&self) -> bool {
        let _ = (
            &self.verifier_output,
            &self.public,
            &self.sumcheck_proof,
            &self.output_claim_values,
            &self.committed_witness,
        );
        true
    }
}

#[cfg(feature = "zk")]
struct Stage5ZkAssembly<F, C>
where
    F: Field,
{
    verifier_output: Stage5ClearOutput<F>,
    public: Stage5PublicOutput<F>,
    sumcheck_proof: SumcheckProof<F, C>,
    output_claim_values: Vec<F>,
    committed_witness: CommittedSumcheckWitness<F>,
}

#[cfg(feature = "zk")]
impl<F, C> Stage5ZkAssembly<F, C>
where
    F: Field,
{
    fn is_ready(&self) -> bool {
        let _ = (
            &self.verifier_output,
            &self.public,
            &self.sumcheck_proof,
            &self.output_claim_values,
            &self.committed_witness,
        );
        true
    }
}

#[cfg(feature = "zk")]
struct Stage6ZkAssembly<F, C>
where
    F: Field,
{
    verifier_output: Stage6ClearOutput<F>,
    public: Stage6PublicOutput<F>,
    sumcheck_proof: SumcheckProof<F, C>,
    output_claim_values: Vec<F>,
    committed_witness: CommittedSumcheckWitness<F>,
}

#[cfg(feature = "zk")]
impl<F, C> Stage6ZkAssembly<F, C>
where
    F: Field,
{
    fn is_ready(&self) -> bool {
        let _ = (
            &self.verifier_output,
            &self.public,
            &self.sumcheck_proof,
            &self.output_claim_values,
            &self.committed_witness,
        );
        true
    }
}

#[cfg(feature = "zk")]
struct Stage7ZkAssembly<F, C>
where
    F: Field,
{
    verifier_output: Stage7ClearOutput<F>,
    public: Stage7PublicOutput<F>,
    sumcheck_proof: SumcheckProof<F, C>,
    output_claim_values: Vec<F>,
    committed_witness: CommittedSumcheckWitness<F>,
}

#[cfg(feature = "zk")]
impl<F, C> Stage7ZkAssembly<F, C>
where
    F: Field,
{
    fn is_ready(&self) -> bool {
        let _ = (
            &self.verifier_output,
            &self.public,
            &self.sumcheck_proof,
            &self.output_claim_values,
            &self.committed_witness,
        );
        true
    }
}

#[cfg(feature = "zk")]
struct Stage8ZkAssembly<F, C, H>
where
    F: Field,
{
    structure: Stage8OpeningStructure<F>,
    joint_commitment: C,
    hiding_evaluation_commitment: H,
}

#[cfg(feature = "zk")]
impl<F, C, H> Stage8ZkAssembly<F, C, H>
where
    F: Field,
{
    fn is_ready(&self) -> bool {
        let _ = (
            &self.structure,
            &self.joint_commitment,
            &self.hiding_evaluation_commitment,
        );
        true
    }
}

#[cfg(feature = "zk")]
pub(crate) struct BlindFoldProverWitness<F: Field> {
    pub(crate) rows: Vec<Vec<F>>,
    pub(crate) blindings: Vec<F>,
    pub(crate) eval_outputs: Vec<F>,
    pub(crate) eval_blindings: Vec<F>,
}

#[cfg(feature = "zk")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BlindFoldCommittedRows {
    Coefficients,
    OutputClaims,
}

#[cfg(feature = "zk")]
fn assign_committed_blindfold_rows<F>(
    witness_values: &mut [Option<F>],
    blindings: &mut [F],
    row_len: usize,
    target_rows: Range<usize>,
    committed_sumchecks: &[CommittedSumcheckWitness<F>],
    row_kind: BlindFoldCommittedRows,
) -> Result<(), ProverError>
where
    F: Field,
{
    let mut row = target_rows.start;
    for witness in committed_sumchecks {
        let (rows, row_blindings) = match row_kind {
            BlindFoldCommittedRows::Coefficients => {
                (&witness.round_coefficients, &witness.round_blindings)
            }
            BlindFoldCommittedRows::OutputClaims => {
                (&witness.output_claim_rows, &witness.output_claim_blindings)
            }
        };
        if rows.len() != row_blindings.len() {
            return Err(ProverError::InvalidStageRequest {
                reason: format!(
                    "BlindFold {row_kind:?} row/blinding count mismatch: rows {}, blindings {}",
                    rows.len(),
                    row_blindings.len()
                ),
            });
        }
        for (values, &blinding) in rows.iter().zip(row_blindings) {
            if row >= target_rows.end {
                return Err(ProverError::InvalidStageRequest {
                    reason: format!(
                        "BlindFold {row_kind:?} rows exceed reserved range {:?}",
                        target_rows
                    ),
                });
            }
            if values.len() > row_len {
                return Err(ProverError::InvalidStageRequest {
                    reason: format!(
                        "BlindFold {row_kind:?} row length {} exceeds witness row length {row_len}",
                        values.len()
                    ),
                });
            }
            for column in 0..row_len {
                let value = values.get(column).copied().unwrap_or_else(F::zero);
                assign_witness_cell(
                    witness_values,
                    row_len,
                    row,
                    column,
                    value,
                    "BlindFold committed row",
                )?;
            }
            blindings[row] = blinding;
            row += 1;
        }
    }
    if row != target_rows.end {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "BlindFold {row_kind:?} row count mismatch: expected {}, got {}",
                target_rows.end - target_rows.start,
                row - target_rows.start
            ),
        });
    }
    Ok(())
}

#[cfg(feature = "zk")]
fn assign_sumcheck_layout_witness<F, C>(
    witness_values: &mut [Option<F>],
    layout: &jolt_sumcheck::SumcheckR1csLayout,
    consistency: &jolt_sumcheck::CommittedSumcheckConsistency<F, C>,
    witness: &CommittedSumcheckWitness<F>,
    domain: SumcheckDomainSpec,
) -> Result<(), ProverError>
where
    F: Field,
{
    let round_count = consistency.rounds.len();
    if layout.rounds.len() != round_count
        || witness.round_coefficients.len() != round_count
        || witness.round_blindings.len() != round_count
    {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "BlindFold sumcheck round count mismatch: layout {}, consistency {}, coefficients {}, blindings {}",
                layout.rounds.len(),
                round_count,
                witness.round_coefficients.len(),
                witness.round_blindings.len()
            ),
        });
    }

    let Some(first_round) = witness.round_coefficients.first() else {
        return Err(ProverError::InvalidStageRequest {
            reason: "BlindFold sumcheck witness has no rounds".to_owned(),
        });
    };
    let input_claim = round_sum_over_domain(domain, first_round)?;
    assign_witness_variable(
        witness_values,
        layout.input_claim,
        input_claim,
        "BlindFold sumcheck input claim",
    )?;

    for ((round_layout, round), coefficients) in layout
        .rounds
        .iter()
        .zip(&consistency.rounds)
        .zip(&witness.round_coefficients)
    {
        if round_layout.coefficients.len() != coefficients.len() {
            return Err(ProverError::InvalidStageRequest {
                reason: format!(
                    "BlindFold round coefficient count mismatch: layout {}, witness {}",
                    round_layout.coefficients.len(),
                    coefficients.len()
                ),
            });
        }
        for (&variable, &coefficient) in round_layout.coefficients.iter().zip(coefficients) {
            assign_witness_variable(
                witness_values,
                variable,
                coefficient,
                "BlindFold sumcheck round coefficient",
            )?;
        }
        let claim_out = evaluate_univariate_coefficients(coefficients, round.challenge);
        assign_witness_variable(
            witness_values,
            round_layout.claim_out,
            claim_out,
            "BlindFold sumcheck output claim",
        )?;
    }
    Ok(())
}

#[cfg(feature = "zk")]
fn blindfold_sumcheck_domains() -> [SumcheckDomainSpec; 9] {
    [
        SumcheckDomainSpec::centered_integer(SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE),
        SumcheckDomainSpec::BooleanHypercube,
        SumcheckDomainSpec::centered_integer(SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE),
        SumcheckDomainSpec::BooleanHypercube,
        SumcheckDomainSpec::BooleanHypercube,
        SumcheckDomainSpec::BooleanHypercube,
        SumcheckDomainSpec::BooleanHypercube,
        SumcheckDomainSpec::BooleanHypercube,
        SumcheckDomainSpec::BooleanHypercube,
    ]
}

#[cfg(feature = "zk")]
fn round_sum_over_domain<F>(
    domain: SumcheckDomainSpec,
    coefficients: &[F],
) -> Result<F, ProverError>
where
    F: Field,
{
    let Some(degree) = coefficients.len().checked_sub(1) else {
        return Err(ProverError::InvalidStageRequest {
            reason: "BlindFold round has no coefficients".to_owned(),
        });
    };
    let weights =
        <SumcheckDomainSpec as SumcheckDomain<F>>::round_sum_coefficients(&domain, degree)
            .map_err(|error| ProverError::InvalidStageRequest {
                reason: format!("BlindFold round-sum coefficient derivation failed: {error}"),
            })?;
    if weights.len() != coefficients.len() {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "BlindFold round-sum coefficient count mismatch: expected {}, got {}",
                coefficients.len(),
                weights.len()
            ),
        });
    }
    Ok(coefficients
        .iter()
        .zip(weights)
        .fold(F::zero(), |acc, (&coefficient, weight)| {
            acc + coefficient * weight
        }))
}

#[cfg(feature = "zk")]
fn evaluate_univariate_coefficients<F>(coefficients: &[F], point: F) -> F
where
    F: Field,
{
    coefficients
        .iter()
        .rev()
        .copied()
        .fold(F::zero(), |acc, coefficient| acc * point + coefficient)
}

#[cfg(feature = "zk")]
fn assign_witness_cell<F>(
    witness_values: &mut [Option<F>],
    row_len: usize,
    row: usize,
    column: usize,
    value: F,
    label: &'static str,
) -> Result<(), ProverError>
where
    F: Field,
{
    if column >= row_len {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "{label} column {column} exceeds BlindFold witness row length {row_len}"
            ),
        });
    }
    let index = 1 + row * row_len + column;
    assign_witness_index(witness_values, index, value, label)
}

#[cfg(feature = "zk")]
fn assign_witness_variable<F>(
    witness_values: &mut [Option<F>],
    variable: Variable,
    value: F,
    label: &'static str,
) -> Result<(), ProverError>
where
    F: Field,
{
    assign_witness_index(witness_values, variable.index(), value, label)
}

#[cfg(feature = "zk")]
fn assign_witness_index<F>(
    witness_values: &mut [Option<F>],
    index: usize,
    value: F,
    label: &'static str,
) -> Result<(), ProverError>
where
    F: Field,
{
    let slot = witness_values
        .get_mut(index)
        .ok_or_else(|| ProverError::InvalidStageRequest {
            reason: format!("{label} variable {index} is outside the BlindFold witness"),
        })?;
    match *slot {
        Some(existing) if existing != value => Err(ProverError::InvalidStageRequest {
            reason: format!("{label} variable {index} was assigned conflicting values"),
        }),
        Some(_) => Ok(()),
        None => {
            *slot = Some(value);
            Ok(())
        }
    }
}

#[cfg(feature = "zk")]
fn complete_blindfold_auxiliary_witness<F>(
    matrices: &ConstraintMatrices<F>,
    witness_values: &mut [Option<F>],
) -> Result<(), ProverError>
where
    F: Field,
{
    let mut progressed = true;
    while progressed {
        progressed = false;
        for constraint in 0..matrices.num_constraints {
            let Some(a_value) = known_sparse_row_value(&matrices.a[constraint], witness_values)?
            else {
                continue;
            };
            let Some(b_value) = known_sparse_row_value(&matrices.b[constraint], witness_values)?
            else {
                continue;
            };
            let c_row = sparse_row_state(&matrices.c[constraint], witness_values)?;
            match c_row {
                SparseRowState::Known(c_value) => {
                    if a_value * b_value != c_value {
                        return Err(ProverError::InvalidStageRequest {
                            reason: format!(
                                "BlindFold witness violates solved constraint {constraint} of {}: A vars {:?}, B vars {:?}, C vars {:?}",
                                matrices.num_constraints,
                                sparse_row_variables(&matrices.a[constraint]),
                                sparse_row_variables(&matrices.b[constraint]),
                                sparse_row_variables(&matrices.c[constraint])
                            ),
                        });
                    }
                }
                SparseRowState::SingleUnknown {
                    index,
                    coefficient,
                    known,
                } => {
                    if coefficient.is_zero() {
                        continue;
                    }
                    if coefficient != F::one() {
                        return Err(ProverError::InvalidStageRequest {
                            reason: format!(
                                "BlindFold auxiliary constraint {constraint} has unsupported non-unit output coefficient"
                            ),
                        });
                    }
                    let value = a_value * b_value - known;
                    assign_witness_index(
                        witness_values,
                        index,
                        value,
                        "BlindFold auxiliary witness",
                    )?;
                    progressed = true;
                }
                SparseRowState::MultipleUnknowns => {}
            }
        }
    }
    Ok(())
}

#[cfg(feature = "zk")]
enum SparseRowState<F> {
    Known(F),
    SingleUnknown {
        index: usize,
        coefficient: F,
        known: F,
    },
    MultipleUnknowns,
}

#[cfg(feature = "zk")]
fn known_sparse_row_value<F>(
    row: &SparseRow<F>,
    witness_values: &[Option<F>],
) -> Result<Option<F>, ProverError>
where
    F: Field,
{
    match sparse_row_state(row, witness_values)? {
        SparseRowState::Known(value) => Ok(Some(value)),
        SparseRowState::SingleUnknown { .. } | SparseRowState::MultipleUnknowns => Ok(None),
    }
}

#[cfg(feature = "zk")]
fn sparse_row_state<F>(
    row: &SparseRow<F>,
    witness_values: &[Option<F>],
) -> Result<SparseRowState<F>, ProverError>
where
    F: Field,
{
    let mut known = F::zero();
    let mut unknown = None;
    for &(index, coefficient) in row {
        let value = witness_values
            .get(index)
            .ok_or_else(|| ProverError::InvalidStageRequest {
                reason: format!("BlindFold constraint references out-of-range variable {index}"),
            })?;
        if let Some(value) = value {
            known += coefficient * *value;
        } else if unknown.is_some() {
            return Ok(SparseRowState::MultipleUnknowns);
        } else {
            unknown = Some((index, coefficient));
        }
    }
    Ok(match unknown {
        Some((index, coefficient)) => SparseRowState::SingleUnknown {
            index,
            coefficient,
            known,
        },
        None => SparseRowState::Known(known),
    })
}

#[cfg(feature = "zk")]
fn sparse_row_variables<F>(row: &SparseRow<F>) -> Vec<usize> {
    row.iter().map(|&(index, _)| index).collect()
}

fn missing_metadata(name: &'static str) -> ProverError {
    ProverError::InvalidStageRequest {
        reason: format!("{name} metadata is required for proof assembly"),
    }
}

fn missing_proof_component(name: &'static str) -> ProverError {
    ProverError::InvalidStageRequest {
        reason: format!("{name} is required for proof construction"),
    }
}

#[cfg(feature = "field-inline")]
const fn field_inline_final_opening_count() -> usize {
    1
}

#[cfg(not(feature = "field-inline"))]
const fn field_inline_final_opening_count() -> usize {
    0
}

fn push_jolt_stage8_opening<C, H>(
    commitments: &mut Vec<C>,
    hints: &mut Vec<H>,
    commitment: &C,
    prover_state: &CommitmentStageProverState<H>,
    polynomial: JoltCommittedPolynomial,
) -> Result<(), ProverError>
where
    C: Clone,
    H: Clone,
{
    commitments.push(commitment.clone());
    let hint = prover_state
        .opening_hints
        .get(&polynomial)
        .cloned()
        .ok_or_else(|| ProverError::InvalidStageRequest {
            reason: format!("Stage 8 opening hint is missing for {polynomial:?}"),
        })?;
    hints.push(hint);
    Ok(())
}

#[cfg(feature = "field-inline")]
fn push_field_inline_stage8_opening<C, H>(
    commitments: &mut Vec<C>,
    hints: &mut Vec<H>,
    commitment: &C,
    prover_state: &CommitmentStageProverState<H>,
    polynomial: FieldInlineCommittedPolynomial,
) -> Result<(), ProverError>
where
    C: Clone,
    H: Clone,
{
    commitments.push(commitment.clone());
    let hint = prover_state
        .field_inline_opening_hints
        .get(&polynomial)
        .cloned()
        .ok_or_else(|| ProverError::InvalidStageRequest {
            reason: format!("Stage 8 field-inline opening hint is missing for {polynomial:?}"),
        })?;
    hints.push(hint);
    Ok(())
}

#[cfg(feature = "zk")]
struct BlindFoldWitnessAssembly<F, C>
where
    F: Field,
{
    committed_sumchecks: Vec<CommittedSumcheckWitness<F>>,
    stage8_hiding_evaluation_commitment: Option<C>,
    stage8_hiding_evaluation_blind: Option<F>,
}

#[cfg(feature = "zk")]
impl<F, C> Default for BlindFoldWitnessAssembly<F, C>
where
    F: Field,
{
    fn default() -> Self {
        Self {
            committed_sumchecks: Vec::new(),
            stage8_hiding_evaluation_commitment: None,
            stage8_hiding_evaluation_blind: None,
        }
    }
}

#[cfg(feature = "zk")]
impl<F, C> BlindFoldWitnessAssembly<F, C>
where
    F: Field,
{
    fn is_ready(&self) -> bool {
        !self.committed_sumchecks.is_empty()
            && self.stage8_hiding_evaluation_commitment.is_some()
            && self.stage8_hiding_evaluation_blind.is_some()
    }
}
