//! Focused equivalence assertions shared by tests.

#![expect(
    clippy::expect_used,
    clippy::panic,
    reason = "equivalence assertions should fail fast with precise mismatch context"
)]

#[cfg(not(feature = "zk"))]
use std::collections::BTreeMap;

use jolt_core::curve::Bn254Curve;
use jolt_core::poly::commitment::dory::DoryCommitmentScheme;
#[cfg(not(feature = "zk"))]
use jolt_core::poly::opening_proof::SumcheckId as S;
#[cfg(not(feature = "zk"))]
use jolt_core::poly::opening_proof::{OpeningId, SumcheckId};
use jolt_core::subprotocols::univariate_skip::UniSkipFirstRoundProofVariant;
use jolt_core::transcripts::Blake2bTranscript as CoreBlake2bTranscript;
#[cfg(not(feature = "zk"))]
use jolt_core::zkvm::instruction::{CircuitFlags, InstructionFlags};
use jolt_core::zkvm::proof_serialization::JoltProof as CoreJoltProof;
#[cfg(not(feature = "zk"))]
use jolt_core::zkvm::witness::{CommittedPolynomial, VirtualPolynomial};
#[cfg(not(feature = "zk"))]
use jolt_core::zkvm::witness::{CommittedPolynomial as C, VirtualPolynomial as V};
use jolt_dory::DoryProof;
use jolt_field::{Fr, Invertible};
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
#[cfg(not(feature = "zk"))]
use jolt_verifier::stages::stage6 as generated_stage6;
use jolt_verifier::{JoltStageExecutionArtifacts, JoltStageProof, JoltSumcheckOutput};

use crate::adapters::{
    canonical_generated_stage5_proof, canonical_generated_stage6_execution_artifacts,
    canonical_generated_stage6_proof, canonical_generated_stage7_execution_artifacts,
    canonical_generated_stage7_proof,
};
use crate::artifacts::{EquivalenceRun, StageArtifacts};
use crate::core_conversion::to_ark;

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

#[cfg(not(feature = "zk"))]
type CoreOpeningExpectation = (&'static str, OpeningId);

#[cfg(not(feature = "zk"))]
fn expected_virtual_opening(
    name: &'static str,
    polynomial: VirtualPolynomial,
    sumcheck: SumcheckId,
) -> CoreOpeningExpectation {
    (name, OpeningId::virt(polynomial, sumcheck))
}

#[cfg(not(feature = "zk"))]
fn expected_committed_opening(
    name: &'static str,
    polynomial: CommittedPolynomial,
    sumcheck: SumcheckId,
) -> CoreOpeningExpectation {
    (name, OpeningId::committed(polynomial, sumcheck))
}

#[cfg(not(feature = "zk"))]
macro_rules! expected_opening {
    (v, $name:literal, $polynomial:expr, $sumcheck:expr) => {
        expected_virtual_opening($name, $polynomial, $sumcheck)
    };
    (c, $name:literal, $polynomial:expr, $sumcheck:expr) => {
        expected_committed_opening($name, $polynomial, $sumcheck)
    };
}

#[cfg(not(feature = "zk"))]
macro_rules! expected_openings {
    ($($kind:ident $name:literal $polynomial:expr => $sumcheck:expr;)+) => {
        [$(expected_opening!($kind, $name, $polynomial, $sumcheck)),+]
    };
}

#[cfg(not(feature = "zk"))]
fn assert_core_opening_claim_evals_match(
    stage: &str,
    proof: &CoreProofForChecks,
    evals: impl IntoIterator<Item = (&'static str, Fr)>,
    expected: impl IntoIterator<Item = CoreOpeningExpectation>,
) {
    let evals = evals.into_iter().collect::<BTreeMap<_, _>>();
    let mut matched_claims = 0usize;

    for (name, opening_id) in expected {
        let Some((_, core_claim)) = proof.opening_claims.0.get(&opening_id) else {
            continue;
        };
        let Some(value) = evals.get(name) else {
            panic!("{stage} proof missing expected opening eval {name}");
        };
        matched_claims += 1;
        assert_eq!(
            *value,
            Fr::from(*core_claim),
            "{stage} opening claim mismatch for {name}",
        );
    }
    assert!(
        matched_claims > 0,
        "{stage} opening claim check matched no core public claims"
    );
}

/// Assert Stage 2 opening-claim evals against jolt-core public proof claims.
#[cfg(not(feature = "zk"))]
pub(crate) fn assert_core_stage2_opening_claims_match_bolt(
    proof: &CoreProofForChecks,
    artifacts: &Stage2ExecutionArtifacts<Fr>,
) {
    let expected = expected_openings! {
        v "stage2.product_virtual.uniskip.eval.UnivariateSkip" V::UnivariateSkip => S::SpartanProductVirtualization;
        v "stage2.ram_read_write.eval.RamVal" V::RamVal => S::RamReadWriteChecking;
        v "stage2.ram_read_write.eval.RamRa" V::RamRa => S::RamReadWriteChecking;
        c "stage2.ram_read_write.eval.RamInc" C::RamInc => S::RamReadWriteChecking;
        v "stage2.product_virtual.remainder.eval.LeftInstructionInput" V::LeftInstructionInput => S::SpartanProductVirtualization;
        v "stage2.product_virtual.remainder.eval.RightInstructionInput" V::RightInstructionInput => S::SpartanProductVirtualization;
        v "stage2.product_virtual.remainder.eval.OpFlagJump" V::OpFlags(CircuitFlags::Jump) => S::SpartanProductVirtualization;
        v "stage2.product_virtual.remainder.eval.OpFlagWriteLookupOutputToRD" V::OpFlags(CircuitFlags::WriteLookupOutputToRD) => S::SpartanProductVirtualization;
        v "stage2.product_virtual.remainder.eval.LookupOutput" V::LookupOutput => S::SpartanProductVirtualization;
        v "stage2.product_virtual.remainder.eval.InstructionFlagBranch" V::InstructionFlags(InstructionFlags::Branch) => S::SpartanProductVirtualization;
        v "stage2.product_virtual.remainder.eval.NextIsNoop" V::NextIsNoop => S::SpartanProductVirtualization;
        v "stage2.product_virtual.remainder.eval.OpFlagVirtualInstruction" V::OpFlags(CircuitFlags::VirtualInstruction) => S::SpartanProductVirtualization;
        v "stage2.instruction_lookup.claim_reduction.eval.LookupOutput" V::LookupOutput => S::InstructionClaimReduction;
        v "stage2.instruction_lookup.claim_reduction.eval.LeftLookupOperand" V::LeftLookupOperand => S::InstructionClaimReduction;
        v "stage2.instruction_lookup.claim_reduction.eval.RightLookupOperand" V::RightLookupOperand => S::InstructionClaimReduction;
        v "stage2.instruction_lookup.claim_reduction.eval.LeftInstructionInput" V::LeftInstructionInput => S::InstructionClaimReduction;
        v "stage2.instruction_lookup.claim_reduction.eval.RightInstructionInput" V::RightInstructionInput => S::InstructionClaimReduction;
        v "stage2.ram_raf.eval.RamRa" V::RamRa => S::RamRafEvaluation;
        v "stage2.ram_output.eval.RamValFinal" V::RamValFinal => S::RamOutputCheck;
    };
    assert_core_opening_claim_evals_match(
        "Stage 2",
        proof,
        artifacts
            .sumchecks
            .iter()
            .flat_map(|output| output.evals.iter().map(|eval| (eval.name, eval.value))),
        expected,
    );
}

#[cfg(feature = "zk")]
pub(crate) fn assert_core_stage2_opening_claims_match_bolt(
    _proof: &CoreProofForChecks,
    _artifacts: &Stage2ExecutionArtifacts<Fr>,
) {
    panic!("opening-claim parity requires non-ZK jolt-core proofs");
}

/// Assert Stage 3 opening-claim evals against jolt-core public proof claims.
#[cfg(not(feature = "zk"))]
pub(crate) fn assert_core_stage3_opening_claims_match_bolt(
    proof: &CoreProofForChecks,
    artifacts: &Stage3ExecutionArtifacts<Fr>,
) {
    let expected = expected_openings! {
        v "stage3.spartan_shift.eval.UnexpandedPC" V::UnexpandedPC => S::SpartanShift;
        v "stage3.spartan_shift.eval.PC" V::PC => S::SpartanShift;
        v "stage3.spartan_shift.eval.OpFlagVirtualInstruction" V::OpFlags(CircuitFlags::VirtualInstruction) => S::SpartanShift;
        v "stage3.spartan_shift.eval.OpFlagIsFirstInSequence" V::OpFlags(CircuitFlags::IsFirstInSequence) => S::SpartanShift;
        v "stage3.spartan_shift.eval.InstructionFlagIsNoop" V::InstructionFlags(InstructionFlags::IsNoop) => S::SpartanShift;
        v "stage3.instruction_input.eval.InstructionFlagLeftOperandIsRs1Value" V::InstructionFlags(InstructionFlags::LeftOperandIsRs1Value) => S::InstructionInputVirtualization;
        v "stage3.instruction_input.eval.Rs1Value" V::Rs1Value => S::InstructionInputVirtualization;
        v "stage3.instruction_input.eval.InstructionFlagLeftOperandIsPC" V::InstructionFlags(InstructionFlags::LeftOperandIsPC) => S::InstructionInputVirtualization;
        v "stage3.instruction_input.eval.UnexpandedPC" V::UnexpandedPC => S::InstructionInputVirtualization;
        v "stage3.instruction_input.eval.InstructionFlagRightOperandIsRs2Value" V::InstructionFlags(InstructionFlags::RightOperandIsRs2Value) => S::InstructionInputVirtualization;
        v "stage3.instruction_input.eval.Rs2Value" V::Rs2Value => S::InstructionInputVirtualization;
        v "stage3.instruction_input.eval.InstructionFlagRightOperandIsImm" V::InstructionFlags(InstructionFlags::RightOperandIsImm) => S::InstructionInputVirtualization;
        v "stage3.instruction_input.eval.Imm" V::Imm => S::InstructionInputVirtualization;
        v "stage3.registers_claim_reduction.eval.RdWriteValue" V::RdWriteValue => S::RegistersClaimReduction;
        v "stage3.registers_claim_reduction.eval.Rs1Value" V::Rs1Value => S::RegistersClaimReduction;
        v "stage3.registers_claim_reduction.eval.Rs2Value" V::Rs2Value => S::RegistersClaimReduction;
    };
    assert_core_opening_claim_evals_match(
        "Stage 3",
        proof,
        artifacts
            .sumchecks
            .iter()
            .flat_map(|output| output.evals.iter().map(|eval| (eval.name, eval.value))),
        expected,
    );
}

#[cfg(feature = "zk")]
pub(crate) fn assert_core_stage3_opening_claims_match_bolt(
    _proof: &CoreProofForChecks,
    _artifacts: &Stage3ExecutionArtifacts<Fr>,
) {
    panic!("opening-claim parity requires non-ZK jolt-core proofs");
}

#[cfg(not(feature = "zk"))]
fn stage6_oracle_index(oracle: &'static str, prefix: &'static str) -> Option<usize> {
    oracle.strip_prefix(prefix)?.parse().ok()
}

#[cfg(not(feature = "zk"))]
fn stage6_committed_polynomial(oracle: &'static str) -> Option<CommittedPolynomial> {
    if let Some(index) = stage6_oracle_index(oracle, "InstructionRa_") {
        return Some(CommittedPolynomial::InstructionRa(index));
    }
    if let Some(index) = stage6_oracle_index(oracle, "BytecodeRa_") {
        return Some(CommittedPolynomial::BytecodeRa(index));
    }
    if let Some(index) = stage6_oracle_index(oracle, "RamRa_") {
        return Some(CommittedPolynomial::RamRa(index));
    }
    match oracle {
        "RamInc" => Some(CommittedPolynomial::RamInc),
        "RdInc" => Some(CommittedPolynomial::RdInc),
        _ => None,
    }
}

#[cfg(not(feature = "zk"))]
fn stage6_opening_claim_id(claim: &generated_stage6::Stage6OpeningClaimPlan) -> Option<OpeningId> {
    if claim
        .symbol
        .starts_with("stage6.bytecode_read_raf.opening.")
    {
        return stage6_committed_polynomial(claim.oracle)
            .map(|polynomial| OpeningId::committed(polynomial, SumcheckId::BytecodeReadRaf));
    }
    if claim.symbol.starts_with("stage6.booleanity.opening.") {
        return stage6_committed_polynomial(claim.oracle)
            .map(|polynomial| OpeningId::committed(polynomial, SumcheckId::Booleanity));
    }
    if claim.symbol == "stage6.hamming_booleanity.opening.HammingWeight" {
        return Some(OpeningId::virt(
            VirtualPolynomial::RamHammingWeight,
            SumcheckId::RamHammingBooleanity,
        ));
    }
    if claim.symbol.starts_with("stage6.ram_ra_virtual.opening.") {
        return stage6_committed_polynomial(claim.oracle)
            .map(|polynomial| OpeningId::committed(polynomial, SumcheckId::RamRaVirtualization));
    }
    if claim
        .symbol
        .starts_with("stage6.instruction_ra_virtual.opening.")
    {
        return stage6_committed_polynomial(claim.oracle).map(|polynomial| {
            OpeningId::committed(polynomial, SumcheckId::InstructionRaVirtualization)
        });
    }
    match claim.symbol {
        "stage6.inc_claim_reduction.opening.RamInc" => Some(OpeningId::committed(
            CommittedPolynomial::RamInc,
            SumcheckId::IncClaimReduction,
        )),
        "stage6.inc_claim_reduction.opening.RdInc" => Some(OpeningId::committed(
            CommittedPolynomial::RdInc,
            SumcheckId::IncClaimReduction,
        )),
        _ => None,
    }
}

/// Assert Stage 6 opening-claim evals against jolt-core public proof claims.
#[cfg(not(feature = "zk"))]
pub(crate) fn assert_core_stage6_opening_claims_match_bolt(
    proof: &CoreProofForChecks,
    stage6_proof: &JoltStageProof,
) {
    let evals = stage6_proof
        .sumchecks
        .iter()
        .flat_map(|output| output.evals.iter())
        .map(|eval| (eval.name, eval.value))
        .collect::<BTreeMap<_, _>>();

    for claim in generated_stage6::STAGE6_OPENING_CLAIMS {
        let Some(opening_id) = stage6_opening_claim_id(claim) else {
            panic!(
                "Stage 6 opening claim has no core mapping: {}",
                claim.symbol
            );
        };
        let Some(value) = evals.get(claim.eval_source) else {
            panic!(
                "Stage 6 proof missing eval {} for opening claim {}",
                claim.eval_source, claim.symbol
            );
        };
        let Some((_, core_claim)) = proof.opening_claims.0.get(&opening_id) else {
            panic!("Stage 6 core opening claim missing for {}", claim.symbol);
        };
        assert_eq!(
            *value,
            Fr::from(*core_claim),
            "Stage 6 opening claim mismatch for {}",
            claim.symbol,
        );
    }
    assert!(
        !generated_stage6::STAGE6_OPENING_CLAIMS.is_empty(),
        "Stage 6 opening claim check was empty"
    );
}

#[cfg(feature = "zk")]
pub(crate) fn assert_core_stage6_opening_claims_match_bolt(
    _proof: &CoreProofForChecks,
    _stage6_proof: &JoltStageProof,
) {
    panic!("opening-claim parity requires non-ZK jolt-core proofs");
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

/// Assert Stage 5 compressed round polynomials match jolt-core.
pub(crate) fn assert_core_stage5_sumcheck_proof_matches_bolt(
    proof: &CoreProofForChecks,
    output: &JoltSumcheckOutput,
) {
    assert_core_compressed_sumcheck_match!("Stage 5", &proof.stage5_sumcheck_proof, output);
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
