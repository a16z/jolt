//! Core proof conversion helpers for Bolt equivalence tests.
//!
//! The helpers here normalize field elements, commitments, and sumcheck
//! proofs into jolt-core shapes so Bolt equivalence tests can compare
//! generated artifacts against the reference implementation.
//!
#![expect(
    clippy::expect_used,
    clippy::panic,
    clippy::too_many_arguments,
    reason = "core proof adapters are test-oracle bridges and fail fast on malformed artifacts"
)]

use ark_bn254::Fr as ArkFr;
use ark_serialize::CanonicalSerialize;
use bolt::protocols::jolt::TranscriptStep;
use jolt_core::curve::Bn254Curve;
use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme;
use jolt_core::poly::commitment::dory::DoryCommitmentScheme;
use jolt_core::poly::unipoly::UniPoly;
use jolt_core::subprotocols::sumcheck::SumcheckInstanceProof;
use jolt_core::subprotocols::univariate_skip::{
    UniSkipFirstRoundProof, UniSkipFirstRoundProofVariant,
};
use jolt_core::transcripts::{Blake2bTranscript, Transcript as _};
use jolt_core::zkvm::proof_serialization::{Claims, JoltProof as CoreJoltProof};
use jolt_field::Fr as NewFr;
use jolt_poly::UnivariatePoly;
use jolt_verifier::JoltStageProof;

use crate::TranscriptEvent;

pub type CoreCommitment = <DoryCommitmentScheme as CommitmentScheme>::Commitment;
pub type CoreProofForConversion =
    CoreJoltProof<ArkFr, Bn254Curve, DoryCommitmentScheme, Blake2bTranscript>;

/// Convert `NewFr` → `ArkFr`. Both are the BN254 scalar field; this is
/// a representation-only cast.
pub(crate) fn to_ark(f: NewFr) -> ArkFr {
    f.into()
}

/// Convert a modular `UnivariatePoly<NewFr>` into a jolt-core
/// `CompressedUniPoly<ArkFr>`. `CompressedUniPoly` stores `[c0, c2, c3,
/// ...]` (linear term `c1` omitted; verifier reconstructs from `s(0) + s(1)`).
fn to_compressed_uni_poly(
    poly: &UnivariatePoly<NewFr>,
) -> jolt_core::poly::unipoly::CompressedUniPoly<ArkFr> {
    let coeffs = poly.coefficients();
    assert!(
        coeffs.len() >= 2,
        "round poly must have at least 2 coefficients"
    );
    let mut compressed = Vec::with_capacity(coeffs.len() - 1);
    compressed.push(to_ark(coeffs[0]));
    for c in &coeffs[2..] {
        compressed.push(to_ark(*c));
    }
    jolt_core::poly::unipoly::CompressedUniPoly {
        coeffs_except_linear_term: compressed,
    }
}

fn to_core_sumcheck_proof(
    round_polys: &[UnivariatePoly<NewFr>],
) -> SumcheckInstanceProof<ArkFr, Bn254Curve, Blake2bTranscript> {
    let compressed: Vec<_> = round_polys.iter().map(to_compressed_uni_poly).collect();
    SumcheckInstanceProof::Clear(jolt_core::subprotocols::sumcheck::ClearSumcheckProof::new(
        compressed,
    ))
}

fn to_core_uniskip_proof_from_round_polys(
    round_polys: &[UnivariatePoly<NewFr>],
) -> UniSkipFirstRoundProofVariant<ArkFr, Bn254Curve, Blake2bTranscript> {
    assert_eq!(round_polys.len(), 1);
    let coefficients = round_polys[0]
        .coefficients()
        .iter()
        .copied()
        .map(to_ark)
        .collect();
    UniSkipFirstRoundProofVariant::Standard(UniSkipFirstRoundProof::new(UniPoly::from_coeff(
        coefficients,
    )))
}

fn to_core_uniskip_proof(
    output: &jolt_kernels::stage1::Stage1SumcheckOutput<NewFr>,
) -> UniSkipFirstRoundProofVariant<ArkFr, Bn254Curve, Blake2bTranscript> {
    to_core_uniskip_proof_from_round_polys(&output.proof.round_polynomials)
}

fn to_core_stage2_uniskip_proof(
    output: &jolt_kernels::stage2::Stage2SumcheckOutput<NewFr>,
) -> UniSkipFirstRoundProofVariant<ArkFr, Bn254Curve, Blake2bTranscript> {
    to_core_uniskip_proof_from_round_polys(&output.proof.round_polynomials)
}

/// Clone a jolt-core proof without depending on private verifier state.
pub(crate) fn clone_core_proof(proof: &CoreProofForConversion) -> CoreProofForConversion {
    CoreJoltProof {
        commitments: proof.commitments.clone(),
        stage1_uni_skip_first_round_proof: proof.stage1_uni_skip_first_round_proof.clone(),
        stage1_sumcheck_proof: proof.stage1_sumcheck_proof.clone(),
        stage2_uni_skip_first_round_proof: proof.stage2_uni_skip_first_round_proof.clone(),
        stage2_sumcheck_proof: proof.stage2_sumcheck_proof.clone(),
        stage3_sumcheck_proof: proof.stage3_sumcheck_proof.clone(),
        stage4_sumcheck_proof: proof.stage4_sumcheck_proof.clone(),
        stage5_sumcheck_proof: proof.stage5_sumcheck_proof.clone(),
        stage6_sumcheck_proof: proof.stage6_sumcheck_proof.clone(),
        stage7_sumcheck_proof: proof.stage7_sumcheck_proof.clone(),
        joint_opening_proof: proof.joint_opening_proof.clone(),
        untrusted_advice_commitment: proof.untrusted_advice_commitment,
        opening_claims: Claims(proof.opening_claims.0.clone()),
        trace_length: proof.trace_length,
        ram_K: proof.ram_K,
        rw_config: proof.rw_config.clone(),
        one_hot_config: proof.one_hot_config.clone(),
        dory_layout: proof.dory_layout,
    }
}

pub(crate) fn core_proof_with_bolt_stage1(
    base: &CoreProofForConversion,
    artifacts: &jolt_kernels::stage1::Stage1ExecutionArtifacts<NewFr>,
) -> CoreProofForConversion {
    let mut proof = clone_core_proof(base);
    proof.stage1_uni_skip_first_round_proof = to_core_uniskip_proof(&artifacts.sumchecks[0]);
    proof.stage1_sumcheck_proof =
        to_core_sumcheck_proof(&artifacts.sumchecks[1].proof.round_polynomials);
    proof
}

pub(crate) fn core_proof_with_bolt_stage2(
    base: &CoreProofForConversion,
    stage1_artifacts: &jolt_kernels::stage1::Stage1ExecutionArtifacts<NewFr>,
    stage2_artifacts: &jolt_kernels::stage2::Stage2ExecutionArtifacts<NewFr>,
) -> CoreProofForConversion {
    let mut proof = core_proof_with_bolt_stage1(base, stage1_artifacts);
    proof.stage2_uni_skip_first_round_proof =
        to_core_stage2_uniskip_proof(&stage2_artifacts.sumchecks[0]);
    proof.stage2_sumcheck_proof =
        to_core_sumcheck_proof(&stage2_artifacts.sumchecks[1].proof.round_polynomials);
    proof
}

pub(crate) fn core_proof_with_bolt_stage3(
    base: &CoreProofForConversion,
    stage1_artifacts: &jolt_kernels::stage1::Stage1ExecutionArtifacts<NewFr>,
    stage2_artifacts: &jolt_kernels::stage2::Stage2ExecutionArtifacts<NewFr>,
    stage3_artifacts: &jolt_kernels::stage3::Stage3ExecutionArtifacts<NewFr>,
) -> CoreProofForConversion {
    let mut proof = core_proof_with_bolt_stage2(base, stage1_artifacts, stage2_artifacts);
    proof.stage3_sumcheck_proof =
        to_core_sumcheck_proof(&stage3_artifacts.sumchecks[0].proof.round_polynomials);
    proof
}

pub(crate) fn core_proof_with_bolt_stage4(
    base: &CoreProofForConversion,
    stage1_artifacts: &jolt_kernels::stage1::Stage1ExecutionArtifacts<NewFr>,
    stage2_artifacts: &jolt_kernels::stage2::Stage2ExecutionArtifacts<NewFr>,
    stage3_artifacts: &jolt_kernels::stage3::Stage3ExecutionArtifacts<NewFr>,
    stage4_artifacts: &jolt_kernels::stage4::Stage4ExecutionArtifacts<NewFr>,
) -> CoreProofForConversion {
    let mut proof =
        core_proof_with_bolt_stage3(base, stage1_artifacts, stage2_artifacts, stage3_artifacts);
    proof.stage4_sumcheck_proof =
        to_core_sumcheck_proof(&stage4_artifacts.sumchecks[0].proof.round_polynomials);
    proof
}

pub(crate) fn core_proof_with_bolt_stage5(
    base: &CoreProofForConversion,
    stage1_artifacts: &jolt_kernels::stage1::Stage1ExecutionArtifacts<NewFr>,
    stage2_artifacts: &jolt_kernels::stage2::Stage2ExecutionArtifacts<NewFr>,
    stage3_artifacts: &jolt_kernels::stage3::Stage3ExecutionArtifacts<NewFr>,
    stage4_artifacts: &jolt_kernels::stage4::Stage4ExecutionArtifacts<NewFr>,
    stage5_proof: &JoltStageProof,
) -> CoreProofForConversion {
    let mut proof = core_proof_with_bolt_stage4(
        base,
        stage1_artifacts,
        stage2_artifacts,
        stage3_artifacts,
        stage4_artifacts,
    );
    proof.stage5_sumcheck_proof =
        to_core_sumcheck_proof(&stage5_proof.sumchecks[0].proof.round_polynomials);
    proof
}

pub(crate) fn core_proof_with_bolt_stage6(
    base: &CoreProofForConversion,
    stage1_artifacts: &jolt_kernels::stage1::Stage1ExecutionArtifacts<NewFr>,
    stage2_artifacts: &jolt_kernels::stage2::Stage2ExecutionArtifacts<NewFr>,
    stage3_artifacts: &jolt_kernels::stage3::Stage3ExecutionArtifacts<NewFr>,
    stage4_artifacts: &jolt_kernels::stage4::Stage4ExecutionArtifacts<NewFr>,
    stage5_proof: &JoltStageProof,
    stage6_proof: &JoltStageProof,
) -> CoreProofForConversion {
    let mut proof = core_proof_with_bolt_stage5(
        base,
        stage1_artifacts,
        stage2_artifacts,
        stage3_artifacts,
        stage4_artifacts,
        stage5_proof,
    );
    proof.stage6_sumcheck_proof =
        to_core_sumcheck_proof(&stage6_proof.sumchecks[0].proof.round_polynomials);
    proof
}

pub(crate) fn core_proof_with_bolt_stage7(
    base: &CoreProofForConversion,
    stage1_artifacts: &jolt_kernels::stage1::Stage1ExecutionArtifacts<NewFr>,
    stage2_artifacts: &jolt_kernels::stage2::Stage2ExecutionArtifacts<NewFr>,
    stage3_artifacts: &jolt_kernels::stage3::Stage3ExecutionArtifacts<NewFr>,
    stage4_artifacts: &jolt_kernels::stage4::Stage4ExecutionArtifacts<NewFr>,
    stage5_proof: &JoltStageProof,
    stage6_proof: &JoltStageProof,
    stage7_proof: &JoltStageProof,
) -> CoreProofForConversion {
    let mut proof = core_proof_with_bolt_stage6(
        base,
        stage1_artifacts,
        stage2_artifacts,
        stage3_artifacts,
        stage4_artifacts,
        stage5_proof,
        stage6_proof,
    );
    proof.stage7_sumcheck_proof =
        to_core_sumcheck_proof(&stage7_proof.sumchecks[0].proof.round_polynomials);
    proof
}

pub(crate) fn core_proof_with_bolt_evaluation(
    base: &CoreProofForConversion,
    evaluation: &jolt_verifier::JoltEvaluationProof,
) -> CoreProofForConversion {
    let mut proof = clone_core_proof(base);
    proof.joint_opening_proof = evaluation.joint_opening_proof.0.clone();
    proof
}

pub(crate) fn core_proof_with_full_bolt(
    base: &CoreProofForConversion,
    proof: &jolt_verifier::JoltProof,
    artifacts: &jolt_prover::JoltProverArtifacts,
) -> CoreProofForConversion {
    let stage5_proof = jolt_prover::stage5_proof(&artifacts.stage5);
    let stage6_proof = jolt_prover::stage6_proof(&artifacts.stage6);
    let stage7_proof = jolt_prover::stage7_proof(&artifacts.stage7);
    let mut core_proof = core_proof_with_bolt_stage7(
        base,
        &artifacts.stage1_outer,
        &artifacts.stage2,
        &artifacts.stage3,
        &artifacts.stage4,
        &stage5_proof,
        &stage6_proof,
        &stage7_proof,
    );
    core_proof.commitments = proof
        .commitments
        .iter()
        .filter_map(|commitment| commitment.as_ref().map(commitment_to_ark))
        .collect();
    core_proof.joint_opening_proof = proof
        .evaluation
        .as_ref()
        .expect("Bolt proof includes evaluation proof")
        .joint_opening_proof
        .0
        .clone();
    core_proof
}

pub fn core_commitment_log<'a>(
    records: impl IntoIterator<Item = (&'a str, Option<&'a jolt_dory::DoryCommitment>)>,
    transcript_steps: &[TranscriptStep],
) -> Vec<TranscriptEvent> {
    let records = records.into_iter().collect::<Vec<_>>();
    let mut transcript = Blake2bTranscript::new(b"Jolt");
    let mut events = Vec::new();

    for step in transcript_steps {
        let mut appended = false;
        for (artifact, commitment) in &records {
            if *artifact != step.source {
                continue;
            }
            if let Some(commitment) = commitment {
                let core_commitment = commitment_to_ark(commitment);
                let label = static_transcript_label(&step.label);
                for bytes in core_append_serializable_bytes(label, &core_commitment) {
                    transcript.raw_append_bytes(&bytes);
                    events.push(TranscriptEvent::Append {
                        bytes,
                        state_after: transcript.state,
                    });
                }
                appended = true;
            }
        }
        assert!(step.optional || appended, "missing core transcript source");
    }

    events
}

pub fn core_commitments_transcript_log(
    commitments: &[CoreCommitment],
    transcript_steps: &[TranscriptStep],
) -> Vec<TranscriptEvent> {
    let mut transcript = Blake2bTranscript::new(b"Jolt");
    let mut events = Vec::new();

    for step in transcript_steps {
        if step.source != "jolt.main_witness_commitments" {
            assert!(step.optional, "unexpected non-main commitment source");
            continue;
        }
        for commitment in commitments {
            let label = static_transcript_label(&step.label);
            for bytes in core_append_serializable_bytes(label, commitment) {
                transcript.raw_append_bytes(&bytes);
                events.push(TranscriptEvent::Append {
                    bytes,
                    state_after: transcript.state,
                });
            }
        }
    }

    events
}

fn core_append_serializable_bytes<T: CanonicalSerialize>(
    label: &'static [u8],
    data: &T,
) -> Vec<Vec<u8>> {
    let mut payload = Vec::new();
    data.serialize_uncompressed(&mut payload)
        .expect("core commitment serialization");
    let mut header = [0u8; 32];
    header[..label.len()].copy_from_slice(label);
    let len = (payload.len() as u64).to_be_bytes();
    header[24..].copy_from_slice(&len);
    payload.reverse();
    vec![header.to_vec(), payload]
}

fn static_transcript_label(label: &str) -> &'static [u8] {
    match label {
        "commitment" => b"commitment",
        "untrusted_advice" => b"untrusted_advice",
        "trusted_advice" => b"trusted_advice",
        _ => panic!("unsupported transcript label `{label}`"),
    }
}

/// Convert a jolt-dory `DoryCommitment` to jolt-core's `ArkGT` shape.
///
/// Both are repr(transparent) wrappers over the same `Fq12` type.
pub(crate) fn commitment_to_ark(c: &jolt_dory::types::DoryCommitment) -> CoreCommitment {
    // SAFETY: Bn254GT and ArkGT are both repr(transparent) over Fq12.
    unsafe { std::mem::transmute_copy(&c.0) }
}
