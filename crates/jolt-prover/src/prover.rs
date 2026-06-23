use common::jolt_device::JoltDevice;
#[cfg(feature = "zk")]
use jolt_backends::BlindFoldBackend;
use jolt_claims::protocols::jolt::JoltFormulaDimensions;
use jolt_crypto::{HomomorphicCommitment, VectorCommitment};
use jolt_openings::{AdditivelyHomomorphic, CommitmentScheme};
use jolt_sumcheck::SumcheckProof;
use jolt_transcript::AppendToTranscript;
use jolt_transcript::{Blake2bTranscript, Transcript};
use jolt_verifier::proof::{ClearProofClaims, JoltProofClaims, JoltStageProofs};
use jolt_verifier::{
    absorb_transcript_commitments, absorb_transcript_preamble, JoltProof, ProofTranscriptConfig,
    JOLT_VERIFIER_CONFIG,
};
use jolt_witness::{
    protocols::jolt_vm::{JoltVmNamespace, RV64_LOOKUP_ADDRESS_BITS},
    CommittedWitnessProvider,
};

use crate::api::{
    BlindFoldProverBackend, ClearProverBackend, JoltVmProverWitness, ProofResult, ProverPcs,
};
#[cfg(feature = "zk")]
use crate::committed::CommittedSumcheckWitness;
use crate::stages::stage0::{self, prepare as stage0_prepare};
use crate::stages::stage0::{CommitmentComponent, CommitmentStageInput};
#[cfg(feature = "zk")]
use crate::stages::stage1::prove::Stage1CommittedProofComponent;
use crate::stages::stage1::prove::{Stage1ProofComponent, Stage1ProverConfig, Stage1ProverInput};
#[cfg(feature = "zk")]
use crate::stages::stage2::prove::Stage2CommittedProofComponent;
use crate::stages::stage2::prove::{
    Stage2BatchProverConfig, Stage2ProofComponent, Stage2ProverInput,
};
#[cfg(feature = "zk")]
use crate::stages::stage3::prove::Stage3CommittedProofComponent;
use crate::stages::stage3::prove::{Stage3ProofComponent, Stage3ProverConfig, Stage3ProverInput};
use crate::stages::stage4::prepare as stage4_prepare;
#[cfg(feature = "zk")]
use crate::stages::stage4::prove::Stage4CommittedProofComponent;
use crate::stages::stage4::prove::{Stage4ProofComponent, Stage4ProverConfig, Stage4ProverInput};
#[cfg(feature = "zk")]
use crate::stages::stage5::prove::Stage5CommittedProofComponent;
use crate::stages::stage5::prove::{Stage5ProofComponent, Stage5ProverConfig, Stage5ProverInput};
use crate::stages::stage6::prepare as stage6_prepare;
#[cfg(feature = "zk")]
use crate::stages::stage6::Stage6CommittedProofComponent;
use crate::stages::stage6::{Stage6ProofComponent, Stage6ProverInput};
use crate::stages::stage7::prepare as stage7_prepare;
#[cfg(feature = "zk")]
use crate::stages::stage7::prove::Stage7CommittedProofComponent;
use crate::stages::stage7::prove::{Stage7ProofComponent, Stage7ProverInput};
use crate::stages::stage8::prepare as stage8_prepare;
#[cfg(feature = "zk")]
use crate::zk;
use crate::{JoltProverPreprocessing, ProverConfig, ProverError};
use jolt_verifier::stages::stage1::stage1_claims_from_r1cs_inputs;

type ClearProofPayload<PCS, VC> = (
    JoltStageProofs<<PCS as CommitmentScheme>::Field, VC>,
    ClearProofClaims<<PCS as CommitmentScheme>::Field>,
    <PCS as CommitmentScheme>::Proof,
);

type ClearStagePayload<PCS, VC> = (
    JoltStageProofs<<PCS as CommitmentScheme>::Field, VC>,
    ClearProofClaims<<PCS as CommitmentScheme>::Field>,
);

type ProofBuildOutput<PCS, VC> = (
    JoltProof<PCS, VC>,
    Option<<PCS as jolt_crypto::Commitment>::Output>,
);

type StageSumcheckProof<F, VC> = SumcheckProof<F, <VC as jolt_crypto::Commitment>::Output>;

#[cfg(feature = "zk")]
type ZkProofPayload<PCS, VC> = (
    JoltStageProofs<<PCS as CommitmentScheme>::Field, VC>,
    <PCS as CommitmentScheme>::Proof,
    jolt_blindfold::BlindFoldProof<
        <PCS as CommitmentScheme>::Field,
        <VC as jolt_crypto::Commitment>::Output,
    >,
);

#[cfg(feature = "zk")]
type ZkStagePayload<PCS, VC> = (
    JoltStageProofs<<PCS as CommitmentScheme>::Field, VC>,
    Vec<CommittedSumcheckWitness<<PCS as CommitmentScheme>::Field>>,
);

pub(crate) fn prove_with_components_inner<PCS, VC, B, W>(
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
    config.validate_for_proving()?;
    let stage0 = prove_stage0(preprocessing, public_io, witness, &config, backend)?;

    #[cfg(feature = "zk")]
    if ProverConfig::features().zk {
        let (stage_proofs, joint_opening_proof, blindfold_proof) =
            prove_zk_stages(preprocessing, public_io, witness, &config, backend, &stage0)?;
        let (proof, trusted_advice_commitment) = build_zk_proof(
            &config,
            stage0,
            stage_proofs,
            joint_opening_proof,
            blindfold_proof,
        )?;
        return Ok(ProofResult {
            proof,
            trusted_advice_commitment,
        });
    }

    let (stage_proofs, clear_claims, joint_opening_proof) =
        prove_clear_stages(preprocessing, public_io, witness, &config, backend, &stage0)?;

    if !ProverConfig::features().zk {
        let (proof, trusted_advice_commitment) = build_clear_proof(
            &config,
            stage0,
            stage_proofs,
            clear_claims,
            joint_opening_proof,
        )?;
        return Ok(ProofResult {
            proof,
            trusted_advice_commitment,
        });
    }

    Err(ProverError::ProverPathNotImplemented {
        path: "full Jolt proof",
    })
}

fn absorb_stage0_transcript<PCS, T>(
    checked: &jolt_verifier::CheckedInputs,
    proof_parameters: ProverConfig,
    stage0: &CommitmentComponent<PCS>,
    transcript: &mut T,
) where
    PCS: CommitmentScheme,
    PCS::Output: AppendToTranscript,
    T: Transcript,
{
    let config = ProofTranscriptConfig::new(
        proof_parameters.rw_config,
        proof_parameters.one_hot_config,
        proof_parameters.trace_polynomial_order,
    );
    absorb_transcript_preamble(checked, config, transcript);
    absorb_transcript_commitments(
        &stage0.commitments,
        stage0.untrusted_advice_commitment.as_ref(),
        stage0.trusted_advice_commitment.as_ref(),
        transcript,
    );
}

fn build_clear_proof<PCS, VC>(
    config: &ProverConfig,
    stage0: CommitmentComponent<PCS>,
    stages: JoltStageProofs<PCS::Field, VC>,
    claims: ClearProofClaims<PCS::Field>,
    joint_opening_proof: PCS::Proof,
) -> Result<ProofBuildOutput<PCS, VC>, ProverError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    if ProverConfig::features().zk {
        return Err(ProverError::InvalidStageRequest {
            reason: "clear proof construction cannot finish a ZK proof".to_owned(),
        });
    }
    let proof_parameters = *config;
    let proof = JoltProof::new(
        stage0.commitments,
        stages,
        joint_opening_proof,
        stage0.untrusted_advice_commitment,
        JoltProofClaims::Clear(claims),
        proof_parameters.trace_length,
        proof_parameters.ram_k,
        proof_parameters.rw_config,
        proof_parameters.one_hot_config,
        proof_parameters.trace_polynomial_order,
    );
    Ok((proof, stage0.trusted_advice_commitment))
}

#[cfg(feature = "zk")]
fn build_zk_proof<PCS, VC>(
    config: &ProverConfig,
    stage0: CommitmentComponent<PCS>,
    stages: JoltStageProofs<PCS::Field, VC>,
    joint_opening_proof: PCS::Proof,
    blindfold_proof: jolt_blindfold::BlindFoldProof<PCS::Field, VC::Output>,
) -> Result<ProofBuildOutput<PCS, VC>, ProverError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    if !ProverConfig::features().zk {
        return Err(ProverError::InvalidStageRequest {
            reason: "ZK proof construction cannot finish a clear proof".to_owned(),
        });
    }
    let proof_parameters = *config;
    let proof = JoltProof::new(
        stage0.commitments,
        stages,
        joint_opening_proof,
        stage0.untrusted_advice_commitment,
        JoltProofClaims::Zk { blindfold_proof },
        proof_parameters.trace_length,
        proof_parameters.ram_k,
        proof_parameters.rw_config,
        proof_parameters.one_hot_config,
        proof_parameters.trace_polynomial_order,
    );
    Ok((proof, stage0.trusted_advice_commitment))
}

fn clear_stage_payload<PCS, VC>(
    stage1: Stage1ProofComponent<PCS::Field, StageSumcheckProof<PCS::Field, VC>>,
    stage2: Stage2ProofComponent<PCS::Field, StageSumcheckProof<PCS::Field, VC>>,
    stage3: Stage3ProofComponent<PCS::Field, StageSumcheckProof<PCS::Field, VC>>,
    stage4: Stage4ProofComponent<PCS::Field, StageSumcheckProof<PCS::Field, VC>>,
    stage5: Stage5ProofComponent<PCS::Field, StageSumcheckProof<PCS::Field, VC>>,
    stage6: Stage6ProofComponent<PCS::Field, StageSumcheckProof<PCS::Field, VC>>,
    stage7: Stage7ProofComponent<PCS::Field, StageSumcheckProof<PCS::Field, VC>>,
) -> Result<ClearStagePayload<PCS, VC>, ProverError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    let stage1_claims = stage1_claims_from_r1cs_inputs(
        stage1.uniskip_output_claim,
        stage1
            .r1cs_input_claims
            .iter()
            .map(crate::stages::stage1::prove::Stage1R1csInputClaim::verifier_input),
    )?;
    let stage_proofs = JoltStageProofs {
        stage1_uni_skip_first_round_proof: stage1.uniskip_proof,
        stage1_sumcheck_proof: stage1.remainder_proof,
        stage2_uni_skip_first_round_proof: stage2.product_uniskip_proof,
        stage2_sumcheck_proof: stage2.regular_batch_proof,
        stage3_sumcheck_proof: stage3.stage3_sumcheck_proof,
        stage4_sumcheck_proof: stage4.stage4_sumcheck_proof,
        stage5_sumcheck_proof: stage5.stage5_sumcheck_proof,
        stage6a_sumcheck_proof: stage6.stage6a_sumcheck_proof,
        stage6b_sumcheck_proof: stage6.stage6b_sumcheck_proof,
        stage7_sumcheck_proof: stage7.stage7_sumcheck_proof,
    };
    let clear_claims = ClearProofClaims {
        stage1: stage1_claims,
        stage2: stage2.claims,
        stage3: stage3.claims,
        stage4: stage4.claims,
        stage5: stage5.claims,
        stage6: stage6.claims,
        stage7: stage7.claims,
    };

    Ok((stage_proofs, clear_claims))
}

#[cfg(feature = "zk")]
fn zk_stage_payload<PCS, VC>(
    stage1: Stage1CommittedProofComponent<PCS::Field, VC>,
    stage2: Stage2CommittedProofComponent<PCS::Field, VC>,
    stage3: Stage3CommittedProofComponent<PCS::Field, VC>,
    stage4: Stage4CommittedProofComponent<PCS::Field, VC>,
    stage5: Stage5CommittedProofComponent<PCS::Field, VC>,
    stage6: Stage6CommittedProofComponent<PCS::Field, VC>,
    stage7: Stage7CommittedProofComponent<PCS::Field, VC>,
) -> ZkStagePayload<PCS, VC>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    let stage_proofs = JoltStageProofs {
        stage1_uni_skip_first_round_proof: stage1.uniskip_proof,
        stage1_sumcheck_proof: stage1.remainder_proof,
        stage2_uni_skip_first_round_proof: stage2.product_uniskip_proof,
        stage2_sumcheck_proof: stage2.regular_batch_proof,
        stage3_sumcheck_proof: stage3.stage3_sumcheck_proof,
        stage4_sumcheck_proof: stage4.stage4_sumcheck_proof,
        stage5_sumcheck_proof: stage5.stage5_sumcheck_proof,
        stage6a_sumcheck_proof: stage6.stage6a_sumcheck_proof,
        stage6b_sumcheck_proof: stage6.stage6b_sumcheck_proof,
        stage7_sumcheck_proof: stage7.stage7_sumcheck_proof,
    };
    let committed_sumchecks = vec![
        stage1.uniskip_committed_witness,
        stage1.remainder_committed_witness,
        stage2.product_uniskip_committed_witness,
        stage2.batch_committed_witness,
        stage3.committed_witness,
        stage4.committed_witness,
        stage5.committed_witness,
        stage6.stage6a_committed_witness,
        stage6.committed_witness,
        stage7.committed_witness,
    ];

    (stage_proofs, committed_sumchecks)
}

#[cfg(feature = "zk")]
fn prove_zk_stages<PCS, VC, B, W>(
    preprocessing: &JoltProverPreprocessing<PCS, VC>,
    public_io: &JoltDevice,
    witness: &W,
    config: &ProverConfig,
    backend: &mut B,
    stage0: &CommitmentComponent<PCS>,
) -> Result<ZkProofPayload<PCS, VC>, ProverError>
where
    PCS: ProverPcs<VC>,
    PCS::Output: AppendToTranscript + HomomorphicCommitment<PCS::Field>,
    <PCS::Field as jolt_field::WithAccumulator>::Accumulator:
        jolt_field::RingAccumulator<Element = PCS::Field>,
    VC: VectorCommitment<Field = PCS::Field>,
    VC::Output: HomomorphicCommitment<PCS::Field>,
    B: ClearProverBackend<PCS::Field> + BlindFoldBackend<PCS::Field>,
    W: JoltVmProverWitness<PCS::Field>,
{
    let proof_parameters = *config;
    let checked = checked_inputs(preprocessing, public_io, config, true)?;
    let vc_setup = preprocessing.verifier.vc_setup.as_ref().ok_or_else(|| {
        ProverError::InvalidProverConfig {
            reason: "ZK proving requires verifier preprocessing with vector-commitment setup"
                .to_owned(),
        }
    })?;
    let mut transcript = Blake2bTranscript::<PCS::Field>::new(b"Jolt");
    absorb_stage0_transcript(&checked, proof_parameters, stage0, &mut transcript);
    let log_t = proof_parameters_log_t(proof_parameters);

    let stage1: Stage1CommittedProofComponent<PCS::Field, VC> =
        crate::stages::stage1::prove::prove_committed_proof_component(
            Stage1ProverInput::new(Stage1ProverConfig::new(log_t), witness),
            backend,
            &mut transcript,
            vc_setup,
        )?;
    let stage1_verifier_output = stage1.verifier_output.clone();

    let log_k = proof_parameters_log_k(proof_parameters);
    let stage2: Stage2CommittedProofComponent<PCS::Field, VC> =
        crate::stages::stage2::prove::prove_committed_proof_component(
            Stage2ProverInput::new(
                Stage2BatchProverConfig::new(log_t, log_k, proof_parameters.rw_config),
                &checked,
                &stage1_verifier_output,
                witness,
            ),
            backend,
            vc_setup,
            &mut transcript,
        )?;
    let stage2_verifier_output = stage2.verifier_output.clone();

    let stage3: Stage3CommittedProofComponent<PCS::Field, VC> =
        crate::stages::stage3::prove::prove_committed_proof_component(
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
        )?;
    let stage3_verifier_output = stage3.verifier_output.clone();

    let ram_val_check_init = stage4_prepare::ram_val_check_initial_evaluation(
        preprocessing,
        &checked,
        &stage2_verifier_output,
        log_k,
    )?;
    let stage4: Stage4CommittedProofComponent<PCS::Field, VC> =
        crate::stages::stage4::prove::prove_committed_proof_component(
            Stage4ProverInput::new(
                Stage4ProverConfig::new(log_t, log_k, proof_parameters.rw_config),
                &checked,
                &stage2_verifier_output,
                &stage3_verifier_output,
                ram_val_check_init,
                witness,
            ),
            backend,
            &mut transcript,
            vc_setup,
        )?;
    let stage4_verifier_output = stage4.verifier_output.clone();

    let formula_dimensions = proof_parameters_formula_dimensions(preprocessing, proof_parameters)?;
    let stage5: Stage5CommittedProofComponent<PCS::Field, VC> =
        crate::stages::stage5::prove::prove_committed_proof_component(
            Stage5ProverInput::new(
                Stage5ProverConfig::new(log_t, log_k, formula_dimensions.instruction_read_raf),
                &checked,
                &stage2_verifier_output,
                &stage4_verifier_output,
                witness,
            ),
            backend,
            &mut transcript,
            vc_setup,
        )?;
    let stage5_verifier_output = stage5.verifier_output.clone();

    let stage6_config = stage6_prepare::prover_config(
        preprocessing,
        public_io,
        proof_parameters,
        formula_dimensions,
    )?;
    let stage6: Stage6CommittedProofComponent<PCS::Field, VC> =
        crate::stages::stage6::prove::prove_committed_proof_component(
            Stage6ProverInput::new(
                &stage6_config,
                &checked,
                &stage1_verifier_output,
                &stage2_verifier_output,
                &stage3_verifier_output,
                &stage4_verifier_output,
                &stage5_verifier_output,
                witness,
            ),
            backend,
            &mut transcript,
            vc_setup,
        )?;
    let stage6_verifier_output = stage6.verifier_output.clone();

    let stage7_config =
        stage7_prepare::prover_config(public_io, proof_parameters, formula_dimensions)?;
    let stage7: Stage7CommittedProofComponent<PCS::Field, VC> =
        crate::stages::stage7::prove::prove_committed_proof_component(
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
        )?;
    let stage7_verifier_output = stage7.verifier_output.clone();
    let (stage_proofs, committed_sumchecks) =
        zk_stage_payload::<PCS, VC>(stage1, stage2, stage3, stage4, stage5, stage6, stage7);

    let stage8_config =
        stage8_prepare::prover_config(public_io, proof_parameters, formula_dimensions)?;
    let (commitments, hints) = stage0.stage8_opening_inputs(stage8_config.layout)?;
    let stage8 = crate::stages::stage8::prove::prove_stage8_zk::<
        PCS::Field,
        PCS,
        W,
        Blake2bTranscript<PCS::Field>,
    >(
        &stage8_config,
        &stage6_verifier_output,
        &stage7_verifier_output,
        witness,
        commitments.as_slice(),
        hints,
        &preprocessing.pcs_setup,
        &mut transcript,
    )?;
    let blindfold = zk::build_blindfold_protocol(
        config,
        &preprocessing.verifier,
        public_io,
        stage0,
        &stage_proofs,
        &stage8.joint_opening_proof,
    )?;
    let mut rng = rand_core::OsRng;
    let blindfold_witness = zk::assemble_blindfold_witness::<PCS, VC, _>(
        &blindfold,
        &committed_sumchecks,
        &stage8,
        &mut rng,
    )?;

    let blindfold_proof = zk::prove_blindfold::<PCS::Field, VC, _, _, _>(
        vc_setup,
        &blindfold,
        &blindfold_witness,
        &mut transcript,
        &mut rng,
        backend,
    )?;
    Ok((stage_proofs, stage8.joint_opening_proof, blindfold_proof))
}

fn prove_clear_stages<PCS, VC, B, W>(
    preprocessing: &JoltProverPreprocessing<PCS, VC>,
    public_io: &JoltDevice,
    witness: &W,
    config: &ProverConfig,
    backend: &mut B,
    stage0: &CommitmentComponent<PCS>,
) -> Result<ClearProofPayload<PCS, VC>, ProverError>
where
    PCS: CommitmentScheme + AdditivelyHomomorphic,
    PCS::Output: AppendToTranscript + HomomorphicCommitment<PCS::Field>,
    <PCS::Field as jolt_field::WithAccumulator>::Accumulator:
        jolt_field::RingAccumulator<Element = PCS::Field>,
    VC: VectorCommitment<Field = PCS::Field>,
    B: ClearProverBackend<PCS::Field>,
    W: JoltVmProverWitness<PCS::Field>,
{
    let proof_parameters = *config;
    let checked = checked_inputs(preprocessing, public_io, config, false)?;
    let mut transcript = Blake2bTranscript::<PCS::Field>::new(b"Jolt");
    absorb_stage0_transcript(&checked, proof_parameters, stage0, &mut transcript);

    let log_t = proof_parameters_log_t(proof_parameters);
    let stage1 = crate::stages::stage1::prove::prove(
        Stage1ProverInput::new(Stage1ProverConfig::new(log_t), witness),
        backend,
        &mut transcript,
    )?;
    let stage1_verifier_output =
        stage1
            .verifier_output
            .clone()
            .ok_or_else(|| ProverError::InvalidStageRequest {
                reason: "Stage 1 clear prover did not return verifier output".to_owned(),
            })?;

    let log_k = proof_parameters_log_k(proof_parameters);
    let stage2 = crate::stages::stage2::prove::prove(
        Stage2ProverInput::new(
            Stage2BatchProverConfig::new(log_t, log_k, proof_parameters.rw_config),
            &checked,
            &stage1_verifier_output,
            witness,
        ),
        backend,
        &mut transcript,
    )?;
    let stage2_verifier_output = stage2.verifier_output.clone();

    let stage3 = crate::stages::stage3::prove::prove(
        Stage3ProverInput::new(
            Stage3ProverConfig::new(log_t),
            &checked,
            &stage1_verifier_output,
            &stage2_verifier_output,
            witness,
        ),
        backend,
        &mut transcript,
    )?;
    let stage3_verifier_output = stage3.verifier_output.clone();

    let ram_val_check_init = stage4_prepare::ram_val_check_initial_evaluation(
        preprocessing,
        &checked,
        &stage2_verifier_output,
        log_k,
    )?;
    let stage4 = crate::stages::stage4::prove::prove(
        Stage4ProverInput::new(
            Stage4ProverConfig::new(log_t, log_k, proof_parameters.rw_config),
            &checked,
            &stage2_verifier_output,
            &stage3_verifier_output,
            ram_val_check_init,
            witness,
        ),
        backend,
        &mut transcript,
    )?;
    let stage4_verifier_output = stage4.verifier_output.clone();

    let formula_dimensions = proof_parameters_formula_dimensions(preprocessing, proof_parameters)?;
    let stage5 = crate::stages::stage5::prove::prove(
        Stage5ProverInput::new(
            Stage5ProverConfig::new(log_t, log_k, formula_dimensions.instruction_read_raf),
            &checked,
            &stage2_verifier_output,
            &stage4_verifier_output,
            witness,
        ),
        backend,
        &mut transcript,
    )?;
    let stage5_verifier_output = stage5.verifier_output.clone();

    let stage6_config = stage6_prepare::prover_config(
        preprocessing,
        public_io,
        proof_parameters,
        formula_dimensions,
    )?;
    let stage6 = crate::stages::stage6::prove::prove(
        Stage6ProverInput::new(
            &stage6_config,
            &checked,
            &stage1_verifier_output,
            &stage2_verifier_output,
            &stage3_verifier_output,
            &stage4_verifier_output,
            &stage5_verifier_output,
            witness,
        ),
        backend,
        &mut transcript,
    )?;
    let stage6_verifier_output = stage6.verifier_output.clone();

    let stage7_config =
        stage7_prepare::prover_config(public_io, proof_parameters, formula_dimensions)?;
    let stage7 = crate::stages::stage7::prove::prove(
        Stage7ProverInput::new(
            &stage7_config,
            &checked,
            &stage4_verifier_output,
            &stage6_verifier_output,
            witness,
        ),
        backend,
        &mut transcript,
    )?;
    let stage7_verifier_output = stage7.verifier_output.clone();

    let (stage_proofs, clear_claims) =
        clear_stage_payload::<PCS, VC>(stage1, stage2, stage3, stage4, stage5, stage6, stage7)?;

    let stage8_config =
        stage8_prepare::prover_config(public_io, proof_parameters, formula_dimensions)?;
    let (commitments, hints) = stage0.stage8_opening_inputs(stage8_config.layout)?;
    let stage8 = crate::stages::stage8::prove::prove_stage8::<
        PCS::Field,
        PCS,
        W,
        Blake2bTranscript<PCS::Field>,
    >(
        &stage8_config,
        &stage6_verifier_output,
        &stage7_verifier_output,
        witness,
        commitments.as_slice(),
        hints,
        &preprocessing.pcs_setup,
        &mut transcript,
    )?;
    let joint_opening_proof = stage8.joint_opening_proof;

    Ok((stage_proofs, clear_claims, joint_opening_proof))
}

fn checked_inputs<PCS, VC>(
    preprocessing: &JoltProverPreprocessing<PCS, VC>,
    public_io: &JoltDevice,
    config: &ProverConfig,
    zk: bool,
) -> Result<jolt_verifier::CheckedInputs, ProverError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    let proof_parameters = *config;
    Ok(jolt_verifier::validate_inputs_from_parts(
        &preprocessing.verifier,
        public_io,
        proof_parameters.trace_length,
        proof_parameters.ram_k,
        proof_parameters.trace_polynomial_order,
        proof_parameters.one_hot_config,
        !public_io.trusted_advice.is_empty(),
        !public_io.untrusted_advice.is_empty(),
        zk,
    )?)
}

fn prove_stage0<PCS, VC, B, W>(
    preprocessing: &JoltProverPreprocessing<PCS, VC>,
    public_io: &JoltDevice,
    witness: &W,
    config: &ProverConfig,
    backend: &mut B,
) -> Result<CommitmentComponent<PCS>, ProverError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
    B: stage0::CommitmentStageBackend<PCS::Field, PCS>,
    W: CommittedWitnessProvider<PCS::Field, JoltVmNamespace> + Sync,
{
    let proof_parameters = *config;
    let formula_dimensions = proof_parameters_formula_dimensions(preprocessing, proof_parameters)?;
    let stage0_config =
        stage0_prepare::commitment_config(public_io, proof_parameters, formula_dimensions);
    stage0::prove::<PCS::Field, _, _, PCS>(
        CommitmentStageInput::new(
            witness,
            &preprocessing.pcs_setup,
            stage0_config,
            JOLT_VERIFIER_CONFIG,
        ),
        backend,
    )
}

fn proof_parameters_log_t(proof_parameters: ProverConfig) -> usize {
    proof_parameters.trace_length.trailing_zeros() as usize
}

fn proof_parameters_log_k(proof_parameters: ProverConfig) -> usize {
    proof_parameters.ram_k.trailing_zeros() as usize
}

fn proof_parameters_formula_dimensions<PCS, VC>(
    preprocessing: &JoltProverPreprocessing<PCS, VC>,
    proof_parameters: ProverConfig,
) -> Result<JoltFormulaDimensions, ProverError>
where
    PCS: CommitmentScheme,
    VC: VectorCommitment<Field = PCS::Field>,
{
    let log_t = proof_parameters_log_t(proof_parameters);
    JoltFormulaDimensions::try_from(proof_parameters.one_hot_config.dimensions(
        log_t,
        RV64_LOOKUP_ADDRESS_BITS,
        preprocessing.verifier.program.bytecode_len(),
        proof_parameters.ram_k,
    ))
    .map_err(|error| ProverError::InvalidProverConfig {
        reason: format!("invalid proof formula dimensions: {error}"),
    })
}
