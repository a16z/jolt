//! Focused equivalence assertions shared by tests.

#![expect(
    clippy::expect_used,
    clippy::panic,
    reason = "equivalence assertions should fail fast with precise mismatch context"
)]

use std::collections::BTreeMap;

use jolt_core::curve::Bn254Curve;
use jolt_core::poly::commitment::dory::DoryCommitmentScheme;
use jolt_core::subprotocols::univariate_skip::UniSkipFirstRoundProofVariant;
use jolt_core::transcripts::Blake2bTranscript as CoreBlake2bTranscript;
use jolt_core::zkvm::proof_serialization::JoltProof as CoreJoltProof;
use jolt_dory::DoryProof;
use jolt_field::{Field, Fr};
use jolt_kernels::{
    stage1::{
        outer_uniskip_extended_evals_from_round_poly, outer_uniskip_targets,
        Stage1ExecutionArtifacts, Stage1OuterR1csData, Stage1OuterRemainingEvaluator,
        Stage1OuterRv64Data, Stage1SumcheckOutput,
    },
    stage2::Stage2ExecutionArtifacts,
    stage3::Stage3ExecutionArtifacts,
};
use jolt_poly::UnivariatePoly;
use jolt_verifier::stages::{common::OpeningClaimPlan, stage2, stage3, stage6};
use jolt_verifier::{JoltStageExecutionArtifacts, JoltStageProof, JoltSumcheckOutput};

use crate::adapters::{
    canonical_generated_stage5_proof, canonical_generated_stage6_execution_artifacts,
    canonical_generated_stage6_proof, canonical_generated_stage7_execution_artifacts,
    canonical_generated_stage7_proof,
};
use crate::artifacts::{EquivalenceRun, StageArtifacts};
use crate::core_conversion::to_ark;
use crate::core_opening_ids::core_opening_id;

pub type CoreProofForChecks =
    CoreJoltProof<ark_bn254::Fr, Bn254Curve, DoryCommitmentScheme, CoreBlake2bTranscript>;

macro_rules! assert_core_compressed_sumcheck_match {
    ($stage:literal, $core_proof:expr, $output:expr) => {{
        let core_polys = match $core_proof {
            jolt_core::subprotocols::sumcheck::SumcheckInstanceProof::Clear(proof) => {
                &proof.compressed_polys
            }
            jolt_core::subprotocols::sumcheck::SumcheckInstanceProof::Zk(_) => {
                panic!("standard {} proof expected", $stage)
            }
        };
        assert_eq!(
            core_polys.len(),
            $output.proof.round_polynomials.len(),
            "{} round count mismatch",
            $stage
        );
        for (round, (core, bolt)) in core_polys
            .iter()
            .zip(&$output.proof.round_polynomials)
            .enumerate()
        {
            let bolt_coeffs = bolt.compress();
            let bolt_coeffs = bolt_coeffs
                .coeffs_except_linear_term()
                .iter()
                .copied()
                .map(to_ark)
                .collect::<Vec<_>>();
            assert_eq!(
                core.coeffs_except_linear_term, bolt_coeffs,
                "{} compressed coefficient mismatch at round {round}",
                $stage
            );
        }
    }};
}

/// Assert byte-for-byte equality of Dory opening proofs.
pub(crate) fn assert_dory_proofs_match(expected: &DoryProof, actual: &DoryProof) {
    assert_eq!(
        dory_proof_bytes(expected),
        dory_proof_bytes(actual),
        "Dory joint opening proof mismatch"
    );
}

fn dory_proof_bytes(proof: &DoryProof) -> Vec<u8> {
    postcard::to_stdvec(proof).expect("serialize Dory proof")
}

fn inverse_nonzero(value: Fr) -> Fr {
    match value.inverse() {
        Some(inverse) => inverse,
        None => unreachable!("nonzero field element has an inverse"),
    }
}

fn assert_core_uniskip_coefficients_match(
    stage: &str,
    proof: &UniSkipFirstRoundProofVariant<ark_bn254::Fr, Bn254Curve, CoreBlake2bTranscript>,
    round_polynomials: &[UnivariatePoly<Fr>],
) {
    let core_coefficients = core_uniskip_coefficients(stage, proof);
    assert_eq!(round_polynomials.len(), 1);
    let bolt_coefficients = round_polynomials[0].coefficients();
    if let Some(index) = bolt_coefficients
        .iter()
        .zip(core_coefficients.iter())
        .position(|(bolt, core)| bolt != core)
    {
        let ratio = if core_coefficients[index] != Fr::from_u64(0) {
            Some(bolt_coefficients[index] * inverse_nonzero(core_coefficients[index]))
        } else {
            None
        };
        let next_ratio = bolt_coefficients
            .iter()
            .zip(core_coefficients.iter())
            .enumerate()
            .skip(index + 1)
            .find(|(_, (_, core))| **core != Fr::from_u64(0))
            .map(|(_, (bolt, core))| *bolt * inverse_nonzero(*core));
        panic!(
            "{stage} uni-skip coefficient mismatch at {index}: bolt={:?} core={:?} ratio={:?} next_ratio={:?}",
            bolt_coefficients[index], core_coefficients[index], ratio, next_ratio
        );
    }
    assert_eq!(
        bolt_coefficients.len(),
        core_coefficients.len(),
        "{stage} uni-skip coefficient count mismatch"
    );
}

fn core_uniskip_coefficients(
    stage: &str,
    proof: &UniSkipFirstRoundProofVariant<ark_bn254::Fr, Bn254Curve, CoreBlake2bTranscript>,
) -> Vec<Fr> {
    match proof {
        UniSkipFirstRoundProofVariant::Standard(proof) => proof
            .uni_poly
            .coeffs
            .iter()
            .copied()
            .map(Fr::from)
            .collect(),
        UniSkipFirstRoundProofVariant::Zk(_) => panic!("standard {stage} proof expected"),
    }
}

pub fn assert_stage1_uniskip_extended_evals_match_core(
    proof: &CoreProofForChecks,
    typed_data: &Stage1OuterRv64Data<'_>,
    generic_data: &Stage1OuterR1csData<'_, Fr>,
    artifacts: &Stage1ExecutionArtifacts<Fr>,
) {
    let tau = artifacts
        .challenge_vectors
        .iter()
        .find(|vector| vector.symbol == "stage1.tau")
        .expect("Bolt stage1 tau")
        .values
        .as_slice();
    let typed_evals = typed_data
        .uniskip_extended_evals(tau)
        .expect("typed Stage 1 extended evals");
    let generic_evals = generic_data
        .uniskip_extended_evals(tau)
        .expect("generic Stage 1 extended evals");
    assert_stage1_extended_eval_vecs_match(
        "typed RV64 vs generic R1CS",
        &typed_evals,
        &generic_evals,
    );

    let core_evals = core_stage1_uniskip_extended_evals(proof, tau[tau.len() - 1]);
    assert_stage1_extended_eval_vecs_match(
        "Bolt typed RV64 vs jolt-core",
        &typed_evals,
        &core_evals,
    );
}

fn core_stage1_uniskip_extended_evals(proof: &CoreProofForChecks, tau_high: Fr) -> Vec<Fr> {
    let coefficients =
        core_uniskip_coefficients("Stage 1", &proof.stage1_uni_skip_first_round_proof);
    let s1 = UnivariatePoly::new(coefficients);
    outer_uniskip_extended_evals_from_round_poly(&s1, tau_high)
}

fn assert_stage1_extended_eval_vecs_match(label: &str, actual: &[Fr], expected: &[Fr]) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label} extended eval count mismatch"
    );
    let targets = outer_uniskip_targets();
    if let Some(index) = actual
        .iter()
        .zip(expected.iter())
        .position(|(actual, expected)| actual != expected)
    {
        panic!(
            "{label} Stage 1 extended eval mismatch at target {} (index {index}): actual={:?} expected={:?}",
            targets[index], actual[index], expected[index]
        );
    }
}

fn assert_core_opening_claim_plans_match_bolt(
    stage: &str,
    proof: &CoreProofForChecks,
    evals: impl IntoIterator<Item = (&'static str, Fr)>,
    opening_claims: &[OpeningClaimPlan],
) {
    let evals = evals.into_iter().collect::<BTreeMap<_, _>>();
    let mut matched_claims = 0usize;

    for claim in opening_claims {
        let Some(opening_id) = core_opening_id(claim) else {
            panic!(
                "{stage} opening claim has no core mapping: {}",
                claim.symbol
            );
        };
        let Some((_, core_claim)) = proof.opening_claims.0.get(&opening_id) else {
            continue;
        };
        let Some(value) = evals.get(claim.eval_source) else {
            panic!(
                "{stage} proof missing eval {} for opening claim {}",
                claim.eval_source, claim.symbol
            );
        };
        matched_claims += 1;
        assert_eq!(
            *value,
            Fr::from(*core_claim),
            "{stage} opening claim mismatch for {}",
            claim.symbol,
        );
    }
    assert!(
        matched_claims > 0,
        "{stage} opening claim check matched no core public claims"
    );
}

/// Assert Stage 2 opening-claim evals against jolt-core public proof claims.
pub(crate) fn assert_core_stage2_opening_claims_match_bolt(
    proof: &CoreProofForChecks,
    artifacts: &Stage2ExecutionArtifacts<Fr>,
) {
    assert_core_opening_claim_plans_match_bolt(
        "Stage 2",
        proof,
        artifacts
            .sumchecks
            .iter()
            .flat_map(|output| output.evals.iter().map(|eval| (eval.name, eval.value))),
        stage2::STAGE2_OPENING_CLAIMS,
    );
}

/// Assert Stage 3 opening-claim evals against jolt-core public proof claims.
pub(crate) fn assert_core_stage3_opening_claims_match_bolt(
    proof: &CoreProofForChecks,
    artifacts: &Stage3ExecutionArtifacts<Fr>,
) {
    assert_core_opening_claim_plans_match_bolt(
        "Stage 3",
        proof,
        artifacts
            .sumchecks
            .iter()
            .flat_map(|output| output.evals.iter().map(|eval| (eval.name, eval.value))),
        stage3::STAGE3_OPENING_CLAIMS,
    );
}

/// Assert Stage 6 opening-claim evals against jolt-core public proof claims.
pub(crate) fn assert_core_stage6_opening_claims_match_bolt(
    proof: &CoreProofForChecks,
    stage6_proof: &JoltStageProof,
) {
    assert_core_opening_claim_plans_match_bolt(
        "Stage 6",
        proof,
        stage6_proof
            .sumchecks
            .iter()
            .flat_map(|output| output.evals.iter().map(|eval| (eval.name, eval.value))),
        stage6::STAGE6_OPENING_CLAIMS,
    );
    assert!(
        !stage6::STAGE6_OPENING_CLAIMS.is_empty(),
        "Stage 6 opening claim check was empty"
    );
}

/// Assert Stage 1 uni-skip proof coefficients match jolt-core.
pub fn assert_core_stage1_uniskip_proof_matches_bolt(
    proof: &CoreProofForChecks,
    output: &Stage1SumcheckOutput<Fr>,
) {
    assert_core_uniskip_coefficients_match(
        "Stage 1",
        &proof.stage1_uni_skip_first_round_proof,
        &output.proof.round_polynomials,
    );
}

/// Assert Stage 2 uni-skip proof coefficients match jolt-core.
pub fn assert_core_stage2_uniskip_proof_matches_bolt(
    proof: &CoreProofForChecks,
    output: &jolt_kernels::stage2::Stage2SumcheckOutput<Fr>,
) {
    assert_core_uniskip_coefficients_match(
        "Stage 2",
        &proof.stage2_uni_skip_first_round_proof,
        &output.proof.round_polynomials,
    );
}

/// Assert Stage 2 compressed round polynomials match jolt-core.
pub(crate) fn assert_core_stage2_sumcheck_proof_matches_bolt(
    proof: &CoreProofForChecks,
    output: &jolt_kernels::stage2::Stage2SumcheckOutput<Fr>,
) {
    assert_core_compressed_sumcheck_match!("Stage 2", &proof.stage2_sumcheck_proof, output);
}

/// Assert Stage 3 compressed round polynomials match jolt-core.
pub(crate) fn assert_core_stage3_sumcheck_proof_matches_bolt(
    proof: &CoreProofForChecks,
    output: &jolt_kernels::stage3::Stage3SumcheckOutput<Fr>,
) {
    assert_core_compressed_sumcheck_match!("Stage 3", &proof.stage3_sumcheck_proof, output);
}

/// Assert Stage 6 compressed round polynomials match jolt-core.
pub(crate) fn assert_core_stage6_sumcheck_proof_matches_bolt(
    proof: &CoreProofForChecks,
    output: &JoltSumcheckOutput,
) {
    assert_core_compressed_sumcheck_match!("Stage 6", &proof.stage6_sumcheck_proof, output);
}

pub(crate) fn assert_canonical_stage_artifacts_match(
    stage: &str,
    expected: StageArtifacts<Fr>,
    actual: StageArtifacts<Fr>,
) {
    assert_eq!(expected, actual, "{stage} artifact mismatch");
}

pub(crate) fn assert_equivalence_run_artifacts_match(
    label: &str,
    expected: &EquivalenceRun<Fr>,
    actual: &EquivalenceRun<Fr>,
) {
    assert_eq!(
        expected.commitments, actual.commitments,
        "{label} commitment trace mismatch"
    );
    assert_eq!(expected.stages, actual.stages, "{label} stage mismatch");
    assert_eq!(
        expected.opening_claims, actual.opening_claims,
        "{label} opening-claim mismatch"
    );
    assert_eq!(
        expected.verifier_result, actual.verifier_result,
        "{label} verifier result mismatch"
    );
}

macro_rules! define_stage_artifacts_match {
    ($fn_name:ident, $stage:literal, $expected_ty:ty, $actual_ty:ty, $expected_adapter:path, $actual_adapter:path) => {
        pub(crate) fn $fn_name(expected: &$expected_ty, actual: &$actual_ty) {
            assert_canonical_stage_artifacts_match(
                $stage,
                $expected_adapter(expected),
                $actual_adapter(actual),
            );
        }
    };
}

define_stage_artifacts_match!(
    assert_stage5_artifacts_match,
    "Stage 5",
    JoltStageProof,
    JoltStageProof,
    canonical_generated_stage5_proof,
    canonical_generated_stage5_proof
);
define_stage_artifacts_match!(
    assert_stage6_artifacts_match,
    "Stage 6",
    JoltStageProof,
    JoltStageExecutionArtifacts,
    canonical_generated_stage6_proof,
    canonical_generated_stage6_execution_artifacts
);
define_stage_artifacts_match!(
    assert_stage7_artifacts_match,
    "Stage 7",
    JoltStageProof,
    JoltStageExecutionArtifacts,
    canonical_generated_stage7_proof,
    canonical_generated_stage7_execution_artifacts
);
