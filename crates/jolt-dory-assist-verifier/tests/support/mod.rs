#![expect(
    clippy::expect_used,
    clippy::panic,
    reason = "test fixtures may panic on invalid local setup"
)]

use jolt_claims::protocols::dory_assist::{
    formulas::{
        composition, dory_reduce,
        protocol::{protocol_claims, CANONICAL_RELATION_ORDER},
        transcript_scalars,
    },
    DoryAssistChallengeId, DoryAssistCopyConstraint, DoryAssistDimensions, DoryAssistOpeningId,
    DoryAssistRelationId, DoryAssistValueRef, DoryAssistVirtualPolynomial, DoryReduceDimensions,
    DoryReducePolynomial, GtDimensions, PrefixPackingDimensions,
};
use jolt_crypto::{Bn254Fq12, Bn254G1, Bn254G2, Bn254GT, Grumpkin, JoltGroup};
use jolt_dory::{DoryCommitment, DoryProof, DoryScheme, DoryVerifierSetup};
use jolt_dory_assist_verifier::{
    artifacts::{
        DoryProofArtifactLayout, DORY_PROOF_DIGEST_INDEX, DORY_REDUCE_ROUNDS_START,
        DORY_VMV_C_START, DORY_VMV_E1_START, DORY_ZK_E2_START, GT_ARTIFACT_COEFFS,
    },
    derive_hyrax_prover_setup,
    native_final::{
        transparent_native_final_input_claims, transparent_replayed_final_pairing_check,
        zk_native_final_input_claims, zk_replayed_final_pairing_check,
    },
    proof::{NATIVE_FINAL_D1_START, NATIVE_FINAL_GT_C_START},
    verify_clear, verify_zk, ClearOpeningStatement, DoryAssist, DoryAssistConfig, DoryAssistHyrax,
    DoryAssistInputPublicClaims, DoryAssistOpeningClaim, DoryAssistProof, DoryAssistStage,
    DoryAssistStage1Proof, DoryAssistStage2Proof, DoryAssistVerifierError, ZkOpeningStatement,
};
use jolt_field::{CanonicalBytes, FixedByteSize, Fq, Fr, FromPrimitiveInt, Invertible};
use jolt_hyrax::HyraxDimensions;
use jolt_openings::{CommitmentScheme, ZkOpeningScheme};
use jolt_poly::{EqPolynomial, Polynomial};
use jolt_sumcheck::SUMCHECK_ROUND_TRANSCRIPT_LABEL;
use jolt_transcript::{Blake2bTranscript, Label, LabelWithCount, Transcript, U64Word};
use jolt_verifier::{PcsAssistClearInput, PcsAssistZkInput, PcsProofAssist};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum VerifierPhase {
    CheckedInputs,
    Stage1,
    Stage2,
    Stage3,
    Opening,
    NativeOutput,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FixtureId {
    ClearBase,
    ClearMultiround,
    ZkBase,
    ZkMultiround,
    ClearInputMismatch,
    ZkInputMismatch,
    StagePayloadMismatch,
    OpeningClaimMismatch,
    HyraxOpeningMismatch,
    DenseCommitmentMismatch,
    PublicOutputMismatch,
    ZkPublicOutputMismatch,
    NativeFinalInputMismatch,
    ZkNativeFinalInputMismatch,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TestCase {
    pub name: &'static str,
    pub zk: bool,
    pub fixture: FixtureId,
    pub checked_at: VerifierPhase,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FixtureMetadata {
    pub id: FixtureId,
    pub name: &'static str,
    pub zk: bool,
    pub expected_accepts: bool,
    pub notes: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TamperTarget {
    pub name: &'static str,
    pub fixture: FixtureId,
    pub checked_at: VerifierPhase,
    pub coverage_note: &'static str,
}

pub struct DoryAssistVerifierCase {
    pub verifier_setup: DoryVerifierSetup,
    pub pcs_proof: DoryProof,
    pub commitment: DoryCommitment,
    pub point: Vec<Fr>,
    pub eval: Fr,
    pub assist_proof: DoryAssistProof,
}

impl DoryAssistVerifierCase {
    pub fn clear_input(&self) -> PcsAssistClearInput<'_, DoryScheme> {
        PcsAssistClearInput {
            setup: &self.verifier_setup,
            pcs_proof: &self.pcs_proof,
            commitment: &self.commitment,
            point: &self.point,
            eval: self.eval,
        }
    }

    pub fn zk_input(&self) -> PcsAssistZkInput<'_, DoryScheme> {
        PcsAssistZkInput {
            setup: &self.verifier_setup,
            pcs_proof: &self.pcs_proof,
            commitment: &self.commitment,
            point: &self.point,
        }
    }

    pub fn verify_clear(&self) -> Result<(), DoryAssistVerifierError> {
        self.verify_clear_with_transcript::<Blake2bTranscript<Fr>>()
    }

    pub fn verify_clear_with_transcript<T>(&self) -> Result<(), DoryAssistVerifierError>
    where
        T: Transcript<Challenge = Fr>,
    {
        let mut transcript = T::new(b"dory-assist-oracle");
        verify_clear(
            &DoryAssistConfig,
            self.clear_input(),
            &self.assist_proof,
            &mut transcript,
        )
    }

    pub fn verify_zk(&self) -> Result<jolt_crypto::Bn254G1, DoryAssistVerifierError> {
        self.verify_zk_with_transcript::<Blake2bTranscript<Fr>>()
    }

    pub fn verify_zk_with_transcript<T>(
        &self,
    ) -> Result<jolt_crypto::Bn254G1, DoryAssistVerifierError>
    where
        T: Transcript<Challenge = Fr>,
    {
        let mut transcript = T::new(b"dory-assist-oracle");
        verify_zk(
            &DoryAssistConfig,
            self.zk_input(),
            &self.assist_proof,
            &mut transcript,
        )
    }

    pub fn verify_clear_via_pcs_assist(&self) -> Result<(), DoryAssistVerifierError> {
        self.verify_clear_via_pcs_assist_with_transcript::<Blake2bTranscript<Fr>>()
    }

    pub fn verify_clear_via_pcs_assist_with_transcript<T>(
        &self,
    ) -> Result<(), DoryAssistVerifierError>
    where
        T: Transcript<Challenge = Fr>,
    {
        let mut transcript = T::new(b"dory-assist-oracle");
        <DoryAssist as PcsProofAssist<DoryScheme>>::verify_clear(
            &DoryAssistConfig,
            self.clear_input(),
            &self.assist_proof,
            &mut transcript,
        )
    }

    pub fn verify_zk_via_pcs_assist(
        &self,
    ) -> Result<jolt_crypto::Bn254G1, DoryAssistVerifierError> {
        self.verify_zk_via_pcs_assist_with_transcript::<Blake2bTranscript<Fr>>()
    }

    pub fn verify_zk_via_pcs_assist_with_transcript<T>(
        &self,
    ) -> Result<jolt_crypto::Bn254G1, DoryAssistVerifierError>
    where
        T: Transcript<Challenge = Fr>,
    {
        let mut transcript = T::new(b"dory-assist-oracle");
        <DoryAssist as PcsProofAssist<DoryScheme>>::verify_zk(
            &DoryAssistConfig,
            self.zk_input(),
            &self.assist_proof,
            &mut transcript,
        )
    }
}

pub fn clear_base_case() -> DoryAssistVerifierCase {
    base_case(false)
}

pub fn clear_multiround_case() -> DoryAssistVerifierCase {
    base_case_with_num_vars::<Blake2bTranscript<Fr>>(false, 4)
}

pub fn clear_multiround_case_with_transcript<T>() -> DoryAssistVerifierCase
where
    T: Transcript<Challenge = Fr>,
{
    base_case_with_num_vars::<T>(false, 4)
}

pub fn clear_base_case_with_transcript<T>() -> DoryAssistVerifierCase
where
    T: Transcript<Challenge = Fr>,
{
    base_case_with_transcript::<T>(false)
}

pub fn clear_shift_public_kernel_case() -> DoryAssistVerifierCase {
    let mut case = base_case(false);
    let vmv_c0 = case
        .assist_proof
        .claims
        .stage1
        .public
        .input
        .dory_proof_artifacts[DORY_VMV_C_START];
    case.assist_proof
        .claims
        .stage1
        .gt_exponentiation
        .accumulator = vmv_c0;
    case.assist_proof
        .claims
        .stage1
        .gt_exponentiation_shift
        .accumulator = vmv_c0;
    case.assist_proof
        .claims
        .stage1
        .gt_exponentiation_boundary
        .accumulator = vmv_c0;
    case.assist_proof.claims.stage1.public.gt_shift_eq_kernel = Fq::default();
    case.assist_proof
        .claims
        .stage1
        .public
        .gt_exponentiation_boundary
        .initial_value = vmv_c0;
    bind_native_public_output_fixture(&mut case, false);
    case
}

pub fn zk_base_case() -> DoryAssistVerifierCase {
    base_case(true)
}

pub fn zk_multiround_case() -> DoryAssistVerifierCase {
    base_case_with_num_vars::<Blake2bTranscript<Fr>>(true, 4)
}

pub fn zk_multiround_case_with_transcript<T>() -> DoryAssistVerifierCase
where
    T: Transcript<Challenge = Fr>,
{
    base_case_with_num_vars::<T>(true, 4)
}

pub fn zk_base_case_with_transcript<T>() -> DoryAssistVerifierCase
where
    T: Transcript<Challenge = Fr>,
{
    base_case_with_transcript::<T>(true)
}

pub fn assert_unique_case_names(cases: &[TestCase]) {
    for (index, case) in cases.iter().enumerate() {
        for other in &cases[index + 1..] {
            assert_ne!(case.name, other.name, "duplicate test case name");
        }
    }
}

pub fn assert_unique_tamper_target_names(targets: &[TamperTarget]) {
    for (index, target) in targets.iter().enumerate() {
        for other in &targets[index + 1..] {
            assert_ne!(target.name, other.name, "duplicate tamper target name");
        }
    }
}

pub fn assert_case_metadata_matches(case: TestCase, metadata: FixtureMetadata) {
    assert_eq!(case.fixture, metadata.id);
    assert_eq!(case.zk, metadata.zk);
}

pub fn assert_accepts<T: core::fmt::Debug>(result: Result<T, DoryAssistVerifierError>) {
    assert!(
        result.is_ok(),
        "valid assist proof was rejected: {result:?}"
    );
}

pub fn assert_rejects<T: core::fmt::Debug>(result: Result<T, DoryAssistVerifierError>) {
    let result_debug = format!("{result:?}");

    assert!(
        result.is_err(),
        "tampered assist proof was accepted: {result_debug}",
    );
}

pub fn assert_rejects_at_stage<T: core::fmt::Debug>(
    result: Result<T, DoryAssistVerifierError>,
    expected_stage: DoryAssistStage,
) {
    let result_debug = format!("{result:?}");
    let actual_stage = match result {
        Err(
            DoryAssistVerifierError::StageClaimMismatch { stage, .. }
            | DoryAssistVerifierError::StageSumcheckFailed { stage, .. }
            | DoryAssistVerifierError::StageOutputMismatch { stage, .. },
        ) => Some(stage),
        Err(_) | Ok(_) => None,
    };

    assert_eq!(
        actual_stage,
        Some(expected_stage),
        "tampered assist proof was not rejected at {expected_stage}: {result_debug}",
    );
}

pub fn tamper_clear_eval(case: &mut DoryAssistVerifierCase) {
    case.eval += Fr::from_u64(1);
}

pub fn tamper_opening_point(case: &mut DoryAssistVerifierCase) {
    case.point[0] += Fr::from_u64(1);
}

pub fn tamper_checked_input_digest(case: &mut DoryAssistVerifierCase) {
    case.assist_proof
        .claims
        .stage1
        .public
        .input
        .checked_input_digest += Fq::from_u64(1);
}

pub fn tamper_verifier_setup_digest(case: &mut DoryAssistVerifierCase) {
    case.assist_proof
        .claims
        .stage1
        .public
        .input
        .verifier_setup_digest += Fq::from_u64(1);
}

pub fn tamper_verifier_setup_artifact(case: &mut DoryAssistVerifierCase) {
    case.assist_proof
        .claims
        .stage1
        .public
        .input
        .verifier_setup_artifacts[0] += Fq::from_u64(1);
}

pub fn tamper_dory_proof_artifact(case: &mut DoryAssistVerifierCase) {
    case.assist_proof
        .claims
        .stage1
        .public
        .input
        .dory_proof_artifacts[DORY_PROOF_DIGEST_INDEX] += Fq::from_u64(1);
}

pub fn tamper_dory_vmv_c_artifact(case: &mut DoryAssistVerifierCase) {
    case.assist_proof
        .claims
        .stage1
        .public
        .input
        .dory_proof_artifacts[DORY_VMV_C_START] += Fq::from_u64(1);
}

pub fn tamper_dory_vmv_e1_artifact(case: &mut DoryAssistVerifierCase) {
    case.assist_proof
        .claims
        .stage1
        .public
        .input
        .dory_proof_artifacts[DORY_VMV_E1_START] += Fq::from_u64(1);
}

pub fn tamper_dory_zk_artifact(case: &mut DoryAssistVerifierCase) {
    case.assist_proof
        .claims
        .stage1
        .public
        .input
        .dory_proof_artifacts[DORY_ZK_E2_START] += Fq::from_u64(1);
}

pub fn tamper_dory_zk_y_com_artifact(case: &mut DoryAssistVerifierCase) {
    let layout = DoryProofArtifactLayout::for_proof(&case.pcs_proof);
    case.assist_proof
        .claims
        .stage1
        .public
        .input
        .dory_proof_artifacts[layout.zk_y_com().start] += Fq::from_u64(1);
}

pub fn tamper_dory_scalar_product_artifact(case: &mut DoryAssistVerifierCase) {
    let layout = DoryProofArtifactLayout::for_proof(&case.pcs_proof);
    case.assist_proof
        .claims
        .stage1
        .public
        .input
        .dory_proof_artifacts[layout.scalar_product_p1().start] += Fq::from_u64(1);
}

pub fn tamper_dory_reduce_round_artifact(case: &mut DoryAssistVerifierCase) {
    case.assist_proof
        .claims
        .stage1
        .public
        .input
        .dory_proof_artifacts[DORY_REDUCE_ROUNDS_START] += Fq::from_u64(1);
}

pub fn tamper_dory_reduce_dimensions(case: &mut DoryAssistVerifierCase) {
    let dimensions = case.assist_proof.dimensions.dory_reduce;
    case.assist_proof.dimensions.dory_reduce =
        DoryReduceDimensions::new(dimensions.point_len(), dimensions.reduce_rounds() + 1);
}

pub fn tamper_gt_dimensions(case: &mut DoryAssistVerifierCase) {
    let dimensions = case.assist_proof.dimensions.gt;
    case.assist_proof.dimensions.gt = GtDimensions::new(
        dimensions.exp_step_vars() + 1,
        dimensions.exp_instance_vars(),
        dimensions.mul_instance_vars(),
    );
}

pub fn tamper_packing_dimensions(case: &mut DoryAssistVerifierCase) {
    let dimensions = case.assist_proof.dimensions.packing;
    case.assist_proof.dimensions.packing = PrefixPackingDimensions::new(
        dimensions.packed_vars() + 1,
        dimensions.max_poly_vars(),
        dimensions.num_claims(),
    )
    .expect("valid non-minimal packing dimensions");
}

pub fn tamper_dory_final_artifact(case: &mut DoryAssistVerifierCase) {
    let artifacts = &mut case
        .assist_proof
        .claims
        .stage1
        .public
        .input
        .dory_proof_artifacts;
    let layout = DoryProofArtifactLayout::for_proof(&case.pcs_proof);
    artifacts[layout.final_e2_start()] += Fq::from_u64(1);
}

pub fn tamper_jolt_commitment_claim(case: &mut DoryAssistVerifierCase) {
    case.assist_proof
        .claims
        .stage1
        .public
        .input
        .jolt_commitments[0] += Fq::from_u64(1);
}

pub fn tamper_jolt_commitment_gt_claim(case: &mut DoryAssistVerifierCase) {
    case.assist_proof
        .claims
        .stage1
        .public
        .input
        .jolt_commitments[1] += Fq::from_u64(1);
}

pub fn tamper_jolt_evaluation_claim(case: &mut DoryAssistVerifierCase) {
    case.assist_proof
        .claims
        .stage1
        .public
        .input
        .jolt_evaluation_claims[0] += Fq::from_u64(1);
}

pub fn tamper_transcript_scalar_claim(case: &mut DoryAssistVerifierCase) {
    case.assist_proof
        .claims
        .stage1
        .public
        .input
        .transcript_scalars[0] += Fq::from_u64(1);
}

pub fn tamper_zk_sigma_c_transcript_scalar_claim(case: &mut DoryAssistVerifierCase) {
    let index = transcript_scalars::dory_scalar_product_sigma_c(
        case.point.len(),
        case.pcs_proof.reduce_round_count(),
    );
    case.assist_proof
        .claims
        .stage1
        .public
        .input
        .transcript_scalars[index] += Fq::from_u64(1);
}

pub fn tamper_stage1_payload(case: &mut DoryAssistVerifierCase) {
    case.assist_proof.stages.stage1.relations[0].sumcheck.degree += 1;
}

pub fn tamper_stage1_sumcheck_round_count(case: &mut DoryAssistVerifierCase) {
    let _ = case.assist_proof.stages.stage1.relations[0]
        .sumcheck_proof
        .round_polynomials
        .pop();
}

pub fn tamper_stage1_relation_output(case: &mut DoryAssistVerifierCase) {
    case.assist_proof
        .claims
        .stage1
        .gt_exponentiation
        .accumulator = Fq::from_u64(1);
    case.assist_proof
        .claims
        .stage1
        .gt_exponentiation
        .digit_selector = Fq::from_u64(1);
}

pub fn tamper_stage1_digit_selector_output(case: &mut DoryAssistVerifierCase) {
    case.assist_proof
        .claims
        .stage1
        .gt_exponentiation_digit_selector
        .digit_lo = Fq::default();
}

pub fn tamper_stage1_shift_output(case: &mut DoryAssistVerifierCase) {
    case.assist_proof
        .claims
        .stage1
        .gt_exponentiation_shift
        .accumulator = Fq::from_u64(1);
}

pub fn tamper_stage1_shift_public(case: &mut DoryAssistVerifierCase) {
    case.assist_proof.claims.stage1.public.gt_shift_eq_kernel = Fq::from_u64(1);
}

pub fn tamper_stage1_boundary_output(case: &mut DoryAssistVerifierCase) {
    case.assist_proof
        .claims
        .stage1
        .gt_exponentiation_boundary
        .accumulator = Fq::from_u64(1);
}

pub fn tamper_stage1_boundary_public(case: &mut DoryAssistVerifierCase) {
    case.assist_proof
        .claims
        .stage1
        .public
        .gt_exponentiation_boundary
        .initial_value = Fq::from_u64(1);
}

pub fn tamper_stage1_multiplication_output(case: &mut DoryAssistVerifierCase) {
    case.assist_proof
        .claims
        .stage1
        .gt_multiplication
        .opening
        .output = Fq::from_u64(1);
}

pub fn tamper_stage1_g1_scalar_multiplication_output(case: &mut DoryAssistVerifierCase) {
    case.assist_proof
        .claims
        .stage1
        .g1
        .scalar_multiplication
        .doubled
        .x = Fq::from_u64(1);
}

pub fn tamper_stage1_g1_shift_output(case: &mut DoryAssistVerifierCase) {
    case.assist_proof
        .claims
        .stage1
        .g1
        .scalar_multiplication_shift
        .accumulator
        .x = Fq::from_u64(1);
}

pub fn tamper_stage1_g1_boundary_public(case: &mut DoryAssistVerifierCase) {
    case.assist_proof
        .claims
        .stage1
        .public
        .g1
        .scalar_multiplication_boundary
        .initial_value
        .x = Fq::from_u64(1);
}

pub fn tamper_stage1_g1_addition_output(case: &mut DoryAssistVerifierCase) {
    case.assist_proof.claims.stage1.g1.addition.output.x = Fq::from_u64(1);
}

pub fn tamper_stage1_g2_scalar_multiplication_output(case: &mut DoryAssistVerifierCase) {
    case.assist_proof
        .claims
        .stage1
        .g2
        .scalar_multiplication
        .doubled
        .x[0] = Fq::from_u64(1);
}

pub fn tamper_stage1_g2_shift_output(case: &mut DoryAssistVerifierCase) {
    case.assist_proof
        .claims
        .stage1
        .g2
        .scalar_multiplication_shift
        .accumulator
        .x[0] = Fq::from_u64(1);
}

pub fn tamper_stage1_g2_boundary_public(case: &mut DoryAssistVerifierCase) {
    case.assist_proof
        .claims
        .stage1
        .public
        .g2
        .scalar_multiplication_boundary
        .initial_value
        .x[0] = Fq::from_u64(1);
}

pub fn tamper_stage1_g2_addition_output(case: &mut DoryAssistVerifierCase) {
    case.assist_proof.claims.stage1.g2.addition.output.x[0] = Fq::from_u64(1);
}

pub fn tamper_stage1_line_step_output(case: &mut DoryAssistVerifierCase) {
    case.assist_proof
        .claims
        .stage1
        .miller_loop
        .line_step
        .shifted_state_x[0] = Fq::from_u64(1);
}

pub fn tamper_stage1_line_evaluation_output(case: &mut DoryAssistVerifierCase) {
    case.assist_proof
        .claims
        .stage1
        .miller_loop
        .line_evaluation
        .line_evaluation_coeffs[6] = Fq::from_u64(1);
}

pub fn tamper_stage1_pair_product_output(case: &mut DoryAssistVerifierCase) {
    case.assist_proof
        .claims
        .stage1
        .miller_loop
        .pair_product
        .shifted_accumulator[0] = Fq::from_u64(1);
}

pub fn tamper_stage1_accumulator_output(case: &mut DoryAssistVerifierCase) {
    case.assist_proof
        .claims
        .stage1
        .miller_loop
        .accumulator
        .accumulator[0] = Fq::from_u64(1);
}

pub fn tamper_stage1_miller_boundary_output(case: &mut DoryAssistVerifierCase) {
    case.assist_proof
        .claims
        .stage1
        .miller_loop
        .boundary
        .accumulator[0] = Fq::from_u64(1);
}

pub fn tamper_stage1_dory_reduce_scalar_fold_output(case: &mut DoryAssistVerifierCase) {
    case.assist_proof
        .claims
        .stage1
        .dory_reduce
        .scalar_fold
        .s1_next_accumulator = Fq::from_u64(1);
}

pub fn tamper_stage1_dory_reduce_state_chain_output(case: &mut DoryAssistVerifierCase) {
    set_dory_reduce_opening(
        &mut case.assist_proof,
        dory_reduce_opening(
            DoryAssistRelationId::DoryReduceStateChain,
            DoryReducePolynomial::S1Accumulator,
        ),
        Fq::from_u64(7),
    );
}

pub fn tamper_stage1_dory_reduce_boundary_output(case: &mut DoryAssistVerifierCase) {
    set_dory_reduce_opening(
        &mut case.assist_proof,
        dory_reduce_opening(
            DoryAssistRelationId::DoryReduceBoundary,
            DoryReducePolynomial::S1Accumulator,
        ),
        Fq::from_u64(9),
    );
}

pub fn tamper_stage1_dory_reduce_gt_transition_output(case: &mut DoryAssistVerifierCase) {
    tamper_dory_reduce_transition_claim(
        case,
        DoryAssistRelationId::DoryReduceGtTransition,
        DoryReducePolynomial::NextC(0),
    );
}

pub fn tamper_stage1_dory_reduce_g1_transition_output(case: &mut DoryAssistVerifierCase) {
    tamper_dory_reduce_transition_claim(
        case,
        DoryAssistRelationId::DoryReduceG1Transition,
        DoryReducePolynomial::NextE1X,
    );
}

pub fn tamper_stage1_dory_reduce_g2_transition_output(case: &mut DoryAssistVerifierCase) {
    tamper_dory_reduce_transition_claim(
        case,
        DoryAssistRelationId::DoryReduceG2Transition,
        DoryReducePolynomial::NextE2X0,
    );
}

fn tamper_dory_reduce_transition_claim(
    case: &mut DoryAssistVerifierCase,
    relation: DoryAssistRelationId,
    polynomial: DoryReducePolynomial,
) {
    let id = DoryAssistOpeningId::virtual_polynomial(
        DoryAssistVirtualPolynomial::DoryReduce(polynomial),
        relation,
    );
    let claim = case
        .assist_proof
        .claims
        .stage1
        .dory_reduce
        .transitions
        .iter_mut()
        .find(|claim| claim.id == id)
        .expect("fixture contains Dory-reduce transition claim");
    claim.value = Fq::from_u64(1);
}

pub fn tamper_stage2_payload(case: &mut DoryAssistVerifierCase) {
    let _ = case.assist_proof.stages.stage2.copy_constraints.pop();
}

pub fn tamper_stage2_copy_value(case: &mut DoryAssistVerifierCase) {
    case.assist_proof
        .claims
        .stage1
        .gt_exponentiation_digit_bitness
        .digit_lo = Fq::default();
}

pub fn tamper_stage2_public_vmv_c_copy_value(case: &mut DoryAssistVerifierCase) {
    let changed = case
        .assist_proof
        .claims
        .stage1
        .public
        .input
        .dory_proof_artifacts[DORY_VMV_C_START]
        + Fq::from_u64(1);
    case.assist_proof
        .claims
        .stage1
        .gt_exponentiation
        .accumulator = changed;
    case.assist_proof
        .claims
        .stage1
        .gt_exponentiation_shift
        .accumulator = changed;
    case.assist_proof
        .claims
        .stage1
        .gt_exponentiation_boundary
        .accumulator = changed;
    case.assist_proof
        .claims
        .stage1
        .public
        .gt_exponentiation_boundary
        .initial_value = changed;
    case.assist_proof.claims.stage1.public.gt_shift_eq_kernel = Fq::default();
    case.assist_proof
        .claims
        .stage1
        .public
        .miller_loop
        .accumulator_shift_eq_kernel =
        accumulator_zero_sumcheck_kernel::<Blake2bTranscript<Fr>>(case, false);
    rebalance_dory_reduce_transition_relations(case, false);
}

pub fn tamper_stage2_public_vmv_e1_copy_value(case: &mut DoryAssistVerifierCase) {
    case.assist_proof
        .claims
        .stage1
        .miller_loop
        .line_evaluation
        .g1_point_x = case
        .assist_proof
        .claims
        .stage1
        .public
        .input
        .dory_proof_artifacts[DORY_VMV_E1_START]
        + Fq::from_u64(1);
    case.assist_proof
        .claims
        .stage1
        .public
        .miller_loop
        .accumulator_shift_eq_kernel =
        accumulator_zero_sumcheck_kernel::<Blake2bTranscript<Fr>>(case, false);
    rebalance_dory_reduce_transition_relations(case, false);
}

pub fn tamper_stage2_line_copy_value(case: &mut DoryAssistVerifierCase) {
    case.assist_proof
        .claims
        .stage1
        .miller_loop
        .line_evaluation
        .line_coefficients[0][0] = Fq::from_u64(1);
}

pub fn tamper_stage2_pair_product_copy_value(case: &mut DoryAssistVerifierCase) {
    case.assist_proof
        .claims
        .stage1
        .miller_loop
        .pair_product
        .accumulator[0] = Fq::from_u64(1);
    case.assist_proof
        .claims
        .stage1
        .public
        .miller_loop
        .pair_product_shift_eq_kernel = Fq::default();
    case.assist_proof
        .claims
        .stage1
        .public
        .miller_loop
        .pair_product_initial_selector = Fq::default();
    case.assist_proof
        .claims
        .stage1
        .public
        .miller_loop
        .pair_product_final_selector = Fq::default();
}

pub fn tamper_stage2_pair_product_quotient_copy_value(case: &mut DoryAssistVerifierCase) {
    case.assist_proof
        .claims
        .stage1
        .miller_loop
        .pair_product
        .quotient[0] = Fq::from_u64(1);
}

pub fn tamper_stage2_accumulator_copy_value(case: &mut DoryAssistVerifierCase) {
    case.assist_proof
        .claims
        .stage1
        .miller_loop
        .accumulator
        .accumulator[0] = Fq::from_u64(1);
    case.assist_proof
        .claims
        .stage1
        .public
        .miller_loop
        .accumulator_shift_eq_kernel = Fq::default();
}

pub fn tamper_stage2_accumulator_quotient_copy_value(case: &mut DoryAssistVerifierCase) {
    case.assist_proof
        .claims
        .stage1
        .miller_loop
        .accumulator
        .quotient[0] = Fq::from_u64(1);
}

pub fn tamper_stage2_boundary_copy_value(case: &mut DoryAssistVerifierCase) {
    case.assist_proof
        .claims
        .stage1
        .miller_loop
        .boundary
        .accumulator[0] = Fq::from_u64(1);
    case.assist_proof
        .claims
        .stage1
        .public
        .miller_loop
        .boundary_initial_selector = Fq::default();
}

pub fn tamper_stage2_g1_shift_copy_value(case: &mut DoryAssistVerifierCase) {
    case.assist_proof
        .claims
        .stage1
        .g1
        .scalar_multiplication_shift
        .shifted_accumulator
        .x = Fq::from_u64(1);
    let final_claim = stage1_zero_sumcheck_final_claim(
        case,
        false,
        DoryAssistRelationId::G1ScalarMultiplicationShift,
    );
    case.assist_proof
        .claims
        .stage1
        .g1
        .scalar_multiplication_shift
        .accumulator
        .x = final_claim;
    case.assist_proof
        .claims
        .stage1
        .public
        .miller_loop
        .accumulator_shift_eq_kernel =
        accumulator_zero_sumcheck_kernel::<Blake2bTranscript<Fr>>(case, false);
    rebalance_dory_reduce_transition_relations(case, false);
}

pub fn tamper_stage2_g1_boundary_copy_value(case: &mut DoryAssistVerifierCase) {
    case.assist_proof
        .claims
        .stage1
        .g1
        .scalar_multiplication_boundary
        .accumulator
        .infinity = Fq::from_u64(1);
    case.assist_proof
        .claims
        .stage1
        .public
        .g1
        .scalar_multiplication_boundary
        .initial_value
        .infinity = Fq::from_u64(1);
    case.assist_proof
        .claims
        .stage1
        .public
        .miller_loop
        .accumulator_shift_eq_kernel =
        accumulator_zero_sumcheck_kernel::<Blake2bTranscript<Fr>>(case, false);
    rebalance_dory_reduce_transition_relations(case, false);
}

pub fn tamper_stage2_g2_shift_copy_value(case: &mut DoryAssistVerifierCase) {
    case.assist_proof
        .claims
        .stage1
        .g2
        .scalar_multiplication_shift
        .shifted_accumulator
        .x[0] = Fq::from_u64(1);
    let final_claim = stage1_zero_sumcheck_final_claim(
        case,
        false,
        DoryAssistRelationId::G2ScalarMultiplicationShift,
    );
    case.assist_proof
        .claims
        .stage1
        .g2
        .scalar_multiplication_shift
        .accumulator
        .x[0] = final_claim;
    case.assist_proof
        .claims
        .stage1
        .public
        .miller_loop
        .accumulator_shift_eq_kernel =
        accumulator_zero_sumcheck_kernel::<Blake2bTranscript<Fr>>(case, false);
    rebalance_dory_reduce_transition_relations(case, false);
}

pub fn tamper_stage2_g2_boundary_copy_value(case: &mut DoryAssistVerifierCase) {
    case.assist_proof
        .claims
        .stage1
        .g2
        .scalar_multiplication_boundary
        .accumulator
        .infinity = Fq::from_u64(1);
    case.assist_proof
        .claims
        .stage1
        .public
        .g2
        .scalar_multiplication_boundary
        .initial_value
        .infinity = Fq::from_u64(1);
    case.assist_proof
        .claims
        .stage1
        .public
        .miller_loop
        .accumulator_shift_eq_kernel =
        accumulator_zero_sumcheck_kernel::<Blake2bTranscript<Fr>>(case, false);
    rebalance_dory_reduce_transition_relations(case, false);
}

pub fn tamper_stage2_dory_reduce_scalar_fold_copy_value(case: &mut DoryAssistVerifierCase) {
    case.assist_proof
        .claims
        .stage1
        .dory_reduce
        .scalar_fold
        .s1_fold_factor += Fq::from_u64(1);
    rebalance_dory_reduce_scalar_fold_relation::<Blake2bTranscript<Fr>>(case, false);
}

pub fn tamper_stage2_dory_reduce_initial_state_copy_value(case: &mut DoryAssistVerifierCase) {
    add_to_dory_reduce_transition_opening(
        &mut case.assist_proof,
        dory_reduce_opening(
            DoryAssistRelationId::DoryReduceGtTransition,
            DoryReducePolynomial::CurrentC(0),
        ),
        Fq::from_u64(1),
    );
    rebalance_dory_reduce_transition_relations(case, false);
}

pub fn tamper_stage2_dory_reduce_proof_artifact_copy_value(case: &mut DoryAssistVerifierCase) {
    add_to_dory_reduce_transition_opening(
        &mut case.assist_proof,
        dory_reduce_opening(
            DoryAssistRelationId::DoryReduceGtTransition,
            DoryReducePolynomial::MessageD1Left(0),
        ),
        Fq::from_u64(1),
    );
    rebalance_dory_reduce_transition_relations(case, false);
}

pub fn tamper_stage2_dory_reduce_setup_artifact_copy_value(case: &mut DoryAssistVerifierCase) {
    add_to_dory_reduce_transition_opening(
        &mut case.assist_proof,
        dory_reduce_opening(
            DoryAssistRelationId::DoryReduceGtTransition,
            DoryReducePolynomial::SetupChi(0),
        ),
        Fq::from_u64(1),
    );
    rebalance_dory_reduce_transition_relations(case, false);
}

pub fn tamper_stage2_dory_reduce_transcript_scalar_copy_value(case: &mut DoryAssistVerifierCase) {
    add_to_dory_reduce_transition_opening(
        &mut case.assist_proof,
        dory_reduce_opening(
            DoryAssistRelationId::DoryReduceGtTransition,
            DoryReducePolynomial::Beta,
        ),
        Fq::from_u64(1),
    );
    rebalance_dory_reduce_transition_relations(case, false);
}

pub fn tamper_stage2_dory_reduce_public_fold_value(case: &mut DoryAssistVerifierCase) {
    tamper_stage2_dory_reduce_public_fold_value_for_mode(case, false);
}

pub fn tamper_stage2_zk_dory_reduce_public_fold_value(case: &mut DoryAssistVerifierCase) {
    tamper_stage2_dory_reduce_public_fold_value_for_mode(case, true);
}

fn tamper_stage2_dory_reduce_public_fold_value_for_mode(
    case: &mut DoryAssistVerifierCase,
    zk: bool,
) {
    let opening = dory_reduce::s1_fold_factor_opening();
    let tampered = get_dory_reduce_opening(&case.assist_proof, opening) + Fq::from_u64(1);
    set_dory_reduce_opening(&mut case.assist_proof, opening, tampered);
    rebalance_dory_reduce_scalar_fold_relation::<Blake2bTranscript<Fr>>(case, zk);
}

pub fn tamper_stage3_payload(case: &mut DoryAssistVerifierCase) {
    case.assist_proof.stages.stage3.packed_eval += Fq::from_u64(1);
}

pub fn tamper_stage3_reduced_openings(case: &mut DoryAssistVerifierCase) {
    case.assist_proof.stages.stage3.reduced_openings.swap(0, 1);
}

pub fn tamper_opening_claim_point(case: &mut DoryAssistVerifierCase) {
    case.assist_proof.claims.opening.packed_point[0] += Fq::from_u64(1);
}

pub fn tamper_opening_claim_eval(case: &mut DoryAssistVerifierCase) {
    case.assist_proof.claims.opening.packed_eval += Fq::from_u64(1);
}

pub fn tamper_hyrax_opening_row(case: &mut DoryAssistVerifierCase) {
    case.assist_proof.opening_proof.combined_row[0] += Fq::from_u64(1);
}

pub fn tamper_hyrax_opening_scalar(case: &mut DoryAssistVerifierCase) {
    case.assist_proof.opening_proof.combined_row_opening_scalar += Fq::from_u64(1);
}

pub fn tamper_dense_commitment(case: &mut DoryAssistVerifierCase) {
    case.assist_proof.dense_commitment.rows[0] =
        Grumpkin::generator().scalar_mul(&Fq::from_u64(31));
}

pub fn tamper_public_output(case: &mut DoryAssistVerifierCase) {
    case.assist_proof.public_outputs.pre_final_exponentiation = Bn254Fq12::default();
}

pub fn tamper_native_final_input(case: &mut DoryAssistVerifierCase) {
    let inputs = &mut case.assist_proof.claims.stage1.public.native_final.inputs;
    let identity = Bn254GT::identity().fq12_coefficients();
    let current_is_identity = inputs
        [NATIVE_FINAL_GT_C_START..NATIVE_FINAL_GT_C_START + Bn254GT::FQ12_COEFFICIENTS]
        .iter()
        .copied()
        .eq(identity);
    if current_is_identity {
        inputs.copy_within(
            NATIVE_FINAL_D1_START..NATIVE_FINAL_D1_START + GT_ARTIFACT_COEFFS,
            NATIVE_FINAL_GT_C_START,
        );
        return;
    }

    inputs[NATIVE_FINAL_GT_C_START..NATIVE_FINAL_GT_C_START + Bn254GT::FQ12_COEFFICIENTS]
        .copy_from_slice(&identity);
    inputs[NATIVE_FINAL_GT_C_START + Bn254GT::FQ12_COEFFICIENTS
        ..NATIVE_FINAL_GT_C_START + GT_ARTIFACT_COEFFS]
        .fill(Fq::default());
}

fn base_case(zk: bool) -> DoryAssistVerifierCase {
    base_case_with_transcript::<Blake2bTranscript<Fr>>(zk)
}

fn base_case_with_transcript<T>(zk: bool) -> DoryAssistVerifierCase
where
    T: Transcript<Challenge = Fr>,
{
    base_case_with_num_vars::<T>(zk, 2)
}

fn base_case_with_num_vars<T>(zk: bool, num_vars: usize) -> DoryAssistVerifierCase
where
    T: Transcript<Challenge = Fr>,
{
    let (prover_setup, verifier_setup) = DoryScheme::setup(num_vars);
    let poly = Polynomial::<Fr>::from(
        (0..(1usize << num_vars))
            .map(|index| Fr::from_u64(1 + index as u64))
            .collect::<Vec<_>>(),
    );
    let point = (0..num_vars)
        .map(|index| Fr::from_u64(5 + 2 * index as u64))
        .collect::<Vec<_>>();
    let eval = poly.evaluate(&point);
    let mut transcript = T::new(b"dory-assist-oracle");
    let (commitment, pcs_proof) = if zk {
        let (commitment, hint) =
            <DoryScheme as ZkOpeningScheme>::commit_zk(poly.evaluations(), &prover_setup);
        let (proof, _hiding_commitment, _blind) =
            DoryScheme::open_zk(&poly, &point, eval, &prover_setup, hint, &mut transcript);
        (commitment, proof)
    } else {
        let (commitment, hint) = DoryScheme::commit(poly.evaluations(), &prover_setup);
        let proof = DoryScheme::open(
            &poly,
            &point,
            eval,
            &prover_setup,
            Some(hint),
            &mut transcript,
        );
        (commitment, proof)
    };

    let dimensions =
        dory_assist_dimensions_for_opening(point.len(), pcs_proof.reduce_round_count());
    let mut assist_proof = DoryAssistProof {
        dimensions,
        ..DoryAssistProof::default()
    };
    assist_proof.stages.stage1 =
        DoryAssistStage1Proof::canonical_for_dimensions(assist_proof.dimensions);
    assist_proof.stages.stage2 =
        DoryAssistStage2Proof::canonical_for_dimensions(assist_proof.dimensions);
    bind_zero_dory_reduce_transition_fixture(&mut assist_proof);
    populate_valid_hyrax_opening(&mut assist_proof);

    let mut case = DoryAssistVerifierCase {
        verifier_setup,
        pcs_proof,
        commitment,
        point,
        eval,
        assist_proof,
    };
    bind_checked_input_public_claims_fixture_with_transcript::<T>(&mut case, zk);
    bind_native_public_output_fixture_with_transcript::<T>(&mut case, zk);
    case
}

fn dory_assist_dimensions_for_opening(
    point_len: usize,
    reduce_rounds: usize,
) -> DoryAssistDimensions {
    let supported = jolt_dory_assist_verifier::proof::default_dory_assist_dimensions();
    let unpacked = DoryAssistDimensions::new(
        supported.gt,
        supported.g1,
        supported.g2,
        supported.miller_loop,
        DoryReduceDimensions::new(point_len, reduce_rounds),
        supported.wiring,
        PrefixPackingDimensions::new(0, 0, 0).expect("valid empty packing dimensions"),
    );
    let packing = composition::prefix_packing_catalog(unpacked)
        .minimal_dimensions()
        .expect("valid checked Dory-assist packing dimensions");

    DoryAssistDimensions::new(
        unpacked.gt,
        unpacked.g1,
        unpacked.g2,
        unpacked.miller_loop,
        unpacked.dory_reduce,
        unpacked.wiring,
        packing,
    )
}

fn bind_checked_input_public_claims_fixture(case: &mut DoryAssistVerifierCase, zk: bool) {
    bind_checked_input_public_claims_fixture_with_transcript::<Blake2bTranscript<Fr>>(case, zk);
}

fn bind_checked_input_public_claims_fixture_with_transcript<T>(
    case: &mut DoryAssistVerifierCase,
    zk: bool,
) where
    T: Transcript<Challenge = Fr>,
{
    case.assist_proof.claims.stage1.public.input =
        checked_input_public_claims_for_fixture::<T>(case, zk);
    bind_dory_reduce_scalar_fold_fixture(case);
    bind_public_input_copy_fixture(case);
}

fn bind_dory_reduce_scalar_fold_fixture(case: &mut DoryAssistVerifierCase) {
    let point_len = case.point.len();
    let input_claims = &case.assist_proof.claims.stage1.public.input;
    let s1_fold_factor = input_claims.transcript_scalars
        [transcript_scalars::dory_reduce_s1_fold_factor(point_len, 0)];
    let s2_fold_factor = input_claims.transcript_scalars
        [transcript_scalars::dory_reduce_s2_fold_factor(point_len, 0)];
    case.assist_proof
        .claims
        .stage1
        .dory_reduce
        .scalar_fold
        .s1_fold_factor = s1_fold_factor;
    case.assist_proof
        .claims
        .stage1
        .dory_reduce
        .scalar_fold
        .s2_fold_factor = s2_fold_factor;
}

fn bind_zero_dory_reduce_transition_fixture(assist_proof: &mut DoryAssistProof) {
    assist_proof.claims.stage1.dory_reduce.transitions.clear();
    assist_proof.claims.stage1.dory_reduce.state_chain.clear();
    assist_proof.claims.stage1.dory_reduce.boundary.clear();
    let protocol = protocol_claims::<Fq>(assist_proof.dimensions);
    for relation_id in [
        DoryAssistRelationId::DoryReduceGtTransition,
        DoryAssistRelationId::DoryReduceG1Transition,
        DoryAssistRelationId::DoryReduceG2Transition,
    ] {
        let relation = protocol
            .relation(relation_id)
            .expect("Dory-reduce transition relation is in the protocol catalog");
        assist_proof.claims.stage1.dory_reduce.transitions.extend(
            relation
                .required_openings()
                .into_iter()
                .map(|id| DoryAssistOpeningClaim {
                    id,
                    value: Fq::default(),
                }),
        );
    }
    if assist_proof.dimensions.dory_reduce.reduce_rounds() > 1 {
        for relation_id in [
            DoryAssistRelationId::DoryReduceStateChain,
            DoryAssistRelationId::DoryReduceBoundary,
        ] {
            let relation = protocol
                .relation(relation_id)
                .expect("Dory-reduce multi-round relation is in the protocol catalog");
            let claims =
                relation
                    .required_openings()
                    .into_iter()
                    .map(|id| DoryAssistOpeningClaim {
                        id,
                        value: Fq::default(),
                    });
            match relation_id {
                DoryAssistRelationId::DoryReduceStateChain => {
                    assist_proof
                        .claims
                        .stage1
                        .dory_reduce
                        .state_chain
                        .extend(claims);
                }
                DoryAssistRelationId::DoryReduceBoundary => {
                    assist_proof
                        .claims
                        .stage1
                        .dory_reduce
                        .boundary
                        .extend(claims);
                }
                _ => unreachable!("only multi-round Dory-reduce relations are handled here"),
            }
        }
    }
}

fn bind_public_input_copy_fixture(case: &mut DoryAssistVerifierCase) {
    let vmv_c0 = case
        .assist_proof
        .claims
        .stage1
        .public
        .input
        .dory_proof_artifacts[DORY_VMV_C_START];
    case.assist_proof
        .claims
        .stage1
        .gt_exponentiation
        .accumulator = vmv_c0;
    case.assist_proof
        .claims
        .stage1
        .gt_exponentiation_shift
        .accumulator = vmv_c0;
    case.assist_proof
        .claims
        .stage1
        .gt_exponentiation_boundary
        .accumulator = vmv_c0;
    case.assist_proof.claims.stage1.public.gt_shift_eq_kernel = Fq::default();
    case.assist_proof
        .claims
        .stage1
        .public
        .gt_exponentiation_boundary
        .initial_value = vmv_c0;
    case.assist_proof
        .claims
        .stage1
        .miller_loop
        .line_evaluation
        .g1_point_x = case
        .assist_proof
        .claims
        .stage1
        .public
        .input
        .dory_proof_artifacts[DORY_VMV_E1_START];
    case.assist_proof
        .claims
        .stage1
        .miller_loop
        .line_evaluation
        .g1_point_y = case
        .assist_proof
        .claims
        .stage1
        .public
        .input
        .dory_proof_artifacts[DORY_VMV_E1_START + 1];
}

fn bind_native_public_output_fixture(case: &mut DoryAssistVerifierCase, zk: bool) {
    bind_native_public_output_fixture_with_transcript::<Blake2bTranscript<Fr>>(case, zk);
}

fn bind_native_public_output_fixture_with_transcript<T>(case: &mut DoryAssistVerifierCase, zk: bool)
where
    T: Transcript<Challenge = Fr>,
{
    if zk {
        bind_zk_pre_final_output_fixture::<T>(case);
    } else {
        bind_transparent_pre_final_output_fixture::<T>(case);
    }

    let output_coefficients = case
        .assist_proof
        .public_outputs
        .pre_final_exponentiation_coefficients();
    case.assist_proof
        .claims
        .stage1
        .public
        .miller_loop
        .bind_pre_final_exponentiation(&case.assist_proof.public_outputs);
    case.assist_proof
        .claims
        .stage1
        .public
        .miller_loop
        .boundary_initial_value = output_coefficients;
    case.assist_proof
        .claims
        .stage1
        .miller_loop
        .accumulator
        .accumulator = output_coefficients;
    case.assist_proof
        .claims
        .stage1
        .miller_loop
        .accumulator
        .shifted_accumulator = output_coefficients;
    case.assist_proof
        .claims
        .stage1
        .miller_loop
        .boundary
        .accumulator = output_coefficients;
    case.assist_proof
        .claims
        .stage1
        .miller_loop
        .boundary
        .shifted_accumulator = output_coefficients;

    let square_row = &mut case.assist_proof.claims.stage1.gt_multiplication.rows
        [composition::ACCUMULATOR_SQUARE_GT_ROW];
    square_row.left = output_coefficients;
    square_row.right = output_coefficients;
    square_row.output = output_coefficients;

    let mul_row = &mut case.assist_proof.claims.stage1.gt_multiplication.rows
        [composition::ACCUMULATOR_MUL_GT_ROW];
    mul_row.left = output_coefficients;
    mul_row.output = output_coefficients;

    case.assist_proof
        .claims
        .stage1
        .public
        .miller_loop
        .accumulator_shift_eq_kernel = accumulator_zero_sumcheck_kernel::<T>(case, zk);
    bind_dory_reduce_transition_copy_fixture::<T>(case, zk);
    populate_valid_hyrax_opening(&mut case.assist_proof);
}

fn bind_transparent_pre_final_output_fixture<T>(case: &mut DoryAssistVerifierCase)
where
    T: Transcript<Challenge = Fr>,
{
    let transcript = T::new(b"dory-assist-oracle");
    let scalars = case
        .pcs_proof
        .verifier_transcript_scalars(&transcript, &case.point);
    let statement = ClearOpeningStatement {
        setup: &case.verifier_setup,
        pcs_proof: &case.pcs_proof,
        commitment: &case.commitment,
        point: &case.point,
        eval: case.eval,
    };
    case.assist_proof.claims.stage1.public.native_final.bind(
        transparent_native_final_input_claims(&statement, &scalars)
            .expect("transparent native-final fixture is well shaped"),
    );
    case.assist_proof.public_outputs.pre_final_exponentiation =
        transparent_replayed_final_pairing_check(&statement, &scalars)
            .expect("transparent final fixture is well shaped")
            .pre_final_exponentiation();
}

fn bind_zk_pre_final_output_fixture<T>(case: &mut DoryAssistVerifierCase)
where
    T: Transcript<Challenge = Fr>,
{
    let transcript = T::new(b"dory-assist-oracle");
    let scalars = case
        .pcs_proof
        .verifier_transcript_scalars(&transcript, &case.point);
    let statement = ZkOpeningStatement {
        setup: &case.verifier_setup,
        pcs_proof: &case.pcs_proof,
        commitment: &case.commitment,
        point: &case.point,
    };
    case.assist_proof.claims.stage1.public.native_final.bind(
        zk_native_final_input_claims(&statement, &scalars)
            .expect("ZK native-final fixture is well shaped"),
    );
    case.assist_proof.public_outputs.pre_final_exponentiation =
        zk_replayed_final_pairing_check(&statement, &scalars)
            .expect("ZK final fixture is well shaped")
            .pre_final_exponentiation();
}

fn bind_dory_reduce_transition_copy_fixture<T>(case: &mut DoryAssistVerifierCase, zk: bool)
where
    T: Transcript<Challenge = Fr>,
{
    for constraint in dory_reduce::initial_state_copy_constraints() {
        bind_dory_reduce_copy_target(case, constraint);
    }
    if case.assist_proof.dimensions.dory_reduce.reduce_rounds() == 1 {
        for constraint in dory_reduce_transition_copy_constraints(case.assist_proof.dimensions) {
            bind_dory_reduce_copy_target(case, constraint);
        }
    }
    bind_dory_reduce_public_fold_fixture::<T>(
        case,
        zk,
        DoryAssistRelationId::DoryReduceGtTransition,
    );
    rebalance_dory_reduce_transition_relation::<T>(
        case,
        zk,
        DoryAssistRelationId::DoryReduceGtTransition,
    );
    bind_dory_reduce_public_fold_fixture::<T>(
        case,
        zk,
        DoryAssistRelationId::DoryReduceG1Transition,
    );
    rebalance_dory_reduce_transition_relation::<T>(
        case,
        zk,
        DoryAssistRelationId::DoryReduceG1Transition,
    );
    bind_dory_reduce_public_fold_fixture::<T>(
        case,
        zk,
        DoryAssistRelationId::DoryReduceG2Transition,
    );
    rebalance_dory_reduce_transition_relation::<T>(
        case,
        zk,
        DoryAssistRelationId::DoryReduceG2Transition,
    );
    bind_dory_reduce_public_fold_fixture::<T>(case, zk, DoryAssistRelationId::DoryReduceScalarFold);
    rebalance_dory_reduce_scalar_fold_relation::<T>(case, zk);
    if case.assist_proof.dimensions.dory_reduce.reduce_rounds() > 1 {
        bind_dory_reduce_boundary_fixture(case);
    }
}

fn dory_reduce_transition_copy_constraints(
    dimensions: jolt_claims::protocols::dory_assist::DoryAssistDimensions,
) -> Vec<DoryAssistCopyConstraint> {
    dory_reduce::proof_artifact_copy_constraints(0)
        .into_iter()
        .chain(dory_reduce::round_setup_artifact_copy_constraints(
            dimensions.dory_reduce.reduce_rounds(),
            0,
        ))
        .chain(dory_reduce::transition_transcript_scalar_copy_constraints(
            dimensions.dory_reduce.point_len(),
            0,
        ))
        .collect()
}

fn bind_dory_reduce_copy_target(
    case: &mut DoryAssistVerifierCase,
    constraint: DoryAssistCopyConstraint,
) {
    let value = match constraint.source {
        DoryAssistValueRef::Public { id, .. } => case
            .assist_proof
            .claims
            .stage1
            .public
            .claim(&id)
            .expect("fixture public claim exists"),
        DoryAssistValueRef::Constant(value) => Fq::from_u64(value as u64),
        DoryAssistValueRef::Witness { .. } | DoryAssistValueRef::Challenge(_) => {
            panic!("Dory-reduce fixture copy source must be public or constant")
        }
    };
    let opening = constraint
        .target
        .witness_opening()
        .expect("Dory-reduce transition fixture copy target must be witness");
    set_dory_reduce_opening(&mut case.assist_proof, opening, value);
}

fn bind_dory_reduce_public_fold_fixture<T>(
    case: &mut DoryAssistVerifierCase,
    zk: bool,
    relation_id: DoryAssistRelationId,
) where
    T: Transcript<Challenge = Fr>,
{
    let context = stage1_relation_context_for_fixture::<T>(case, zk, relation_id);
    let weights = EqPolynomial::new(context.sumcheck_point).evaluations();
    for constraint in dory_reduce::public_fold_constraints(case.assist_proof.dimensions.dory_reduce)
    {
        let opening = constraint
            .target
            .witness_opening()
            .expect("Dory-reduce public fold target is a witness opening");
        if opening_relation(opening) != relation_id {
            continue;
        }
        assert!(
            constraint.sources.len() <= weights.len(),
            "public fold sources fit the relation point domain"
        );
        let value =
            constraint
                .sources
                .iter()
                .zip(&weights)
                .fold(Fq::default(), |acc, (id, weight)| {
                    let public = case
                        .assist_proof
                        .claims
                        .stage1
                        .public_claim(id)
                        .expect("fixture public-fold source exists");
                    acc + public * *weight
                });
        set_dory_reduce_opening(&mut case.assist_proof, opening, value);
    }
}

fn bind_dory_reduce_boundary_fixture(case: &mut DoryAssistVerifierCase) {
    for term in dory_reduce::initial_boundary_terms()
        .into_iter()
        .chain(dory_reduce::final_boundary_terms())
    {
        let value = match term.value {
            dory_reduce::DoryReduceBoundaryValue::ConstantOne => Fq::from_u64(1),
            dory_reduce::DoryReduceBoundaryValue::Public(id) => case
                .assist_proof
                .claims
                .stage1
                .public_claim(&id)
                .expect("fixture Dory-reduce boundary public claim exists"),
        };
        set_dory_reduce_opening(&mut case.assist_proof, term.opening, value);
    }
}

fn rebalance_dory_reduce_transition_relation<T>(
    case: &mut DoryAssistVerifierCase,
    zk: bool,
    relation_id: DoryAssistRelationId,
) where
    T: Transcript<Challenge = Fr>,
{
    let protocol = protocol_claims::<Fq>(case.assist_proof.dimensions);
    let relation = protocol
        .relation(relation_id)
        .expect("Dory-reduce transition relation is in the protocol catalog");
    let context = stage1_relation_context_for_fixture::<T>(case, zk, relation_id);
    let target = dory_reduce_transition_target_opening(relation_id);
    let input = relation
        .input
        .expression()
        .try_evaluate(
            |id| {
                case.assist_proof
                    .claims
                    .stage1
                    .opening_claim(id)
                    .ok_or("missing opening")
            },
            |id| resolve_fixture_challenge(&context.relation_challenges, id),
            |id| {
                case.assist_proof
                    .claims
                    .stage1
                    .public_claim(id)
                    .ok_or("missing public")
            },
        )
        .expect("fixture Dory-reduce transition input evaluates");
    let output = relation
        .output
        .expression()
        .try_evaluate(
            |id| {
                case.assist_proof
                    .claims
                    .stage1
                    .opening_claim(id)
                    .ok_or("missing opening")
            },
            |id| resolve_fixture_challenge(&context.relation_challenges, id),
            |id| {
                case.assist_proof
                    .claims
                    .stage1
                    .public_claim(id)
                    .ok_or("missing public")
            },
        )
        .expect("fixture Dory-reduce transition output evaluates");
    let factor = sumcheck_linear_factor(&context.sumcheck_point);
    let delta = (output - input * factor)
        * factor
            .inverse()
            .expect("fixture Dory-reduce sumcheck factor is nonzero");
    let adjusted = get_dory_reduce_opening(&case.assist_proof, target) + delta;
    set_dory_reduce_opening(&mut case.assist_proof, target, adjusted);
}

fn rebalance_dory_reduce_transition_relations(case: &mut DoryAssistVerifierCase, zk: bool) {
    rebalance_dory_reduce_transition_relation::<Blake2bTranscript<Fr>>(
        case,
        zk,
        DoryAssistRelationId::DoryReduceGtTransition,
    );
    rebalance_dory_reduce_transition_relation::<Blake2bTranscript<Fr>>(
        case,
        zk,
        DoryAssistRelationId::DoryReduceG1Transition,
    );
    rebalance_dory_reduce_transition_relation::<Blake2bTranscript<Fr>>(
        case,
        zk,
        DoryAssistRelationId::DoryReduceG2Transition,
    );
    rebalance_dory_reduce_scalar_fold_relation::<Blake2bTranscript<Fr>>(case, zk);
}

fn rebalance_dory_reduce_scalar_fold_relation<T>(case: &mut DoryAssistVerifierCase, zk: bool)
where
    T: Transcript<Challenge = Fr>,
{
    let protocol = protocol_claims::<Fq>(case.assist_proof.dimensions);
    let relation = protocol
        .relation(DoryAssistRelationId::DoryReduceScalarFold)
        .expect("Dory-reduce scalar-fold relation is in the protocol catalog");
    let context = stage1_relation_context_for_fixture::<T>(
        case,
        zk,
        DoryAssistRelationId::DoryReduceScalarFold,
    );
    let input = relation
        .input
        .expression()
        .try_evaluate(
            |id| {
                case.assist_proof
                    .claims
                    .stage1
                    .opening_claim(id)
                    .ok_or("missing opening")
            },
            |id| resolve_fixture_challenge(&context.relation_challenges, id),
            |id| {
                case.assist_proof
                    .claims
                    .stage1
                    .public_claim(id)
                    .ok_or("missing public")
            },
        )
        .expect("fixture Dory-reduce scalar-fold input evaluates");
    let output = relation
        .output
        .expression()
        .try_evaluate(
            |id| {
                case.assist_proof
                    .claims
                    .stage1
                    .opening_claim(id)
                    .ok_or("missing opening")
            },
            |id| resolve_fixture_challenge(&context.relation_challenges, id),
            |id| {
                case.assist_proof
                    .claims
                    .stage1
                    .public_claim(id)
                    .ok_or("missing public")
            },
        )
        .expect("fixture Dory-reduce scalar-fold output evaluates");
    let factor = sumcheck_linear_factor(&context.sumcheck_point);
    let target = dory_reduce::s1_next_accumulator_opening();
    let delta = (output - input * factor)
        * factor
            .inverse()
            .expect("fixture Dory-reduce scalar-fold sumcheck factor is nonzero");
    let adjusted = get_dory_reduce_opening(&case.assist_proof, target) + delta;
    set_dory_reduce_opening(&mut case.assist_proof, target, adjusted);
}

fn dory_reduce_transition_target_opening(relation: DoryAssistRelationId) -> DoryAssistOpeningId {
    let polynomial = match relation {
        DoryAssistRelationId::DoryReduceGtTransition => DoryReducePolynomial::NextC(0),
        DoryAssistRelationId::DoryReduceG1Transition => DoryReducePolynomial::NextE1X,
        DoryAssistRelationId::DoryReduceG2Transition => DoryReducePolynomial::NextE2X0,
        _ => panic!("not a Dory-reduce transition relation"),
    };
    dory_reduce_opening(relation, polynomial)
}

fn dory_reduce_opening(
    relation: DoryAssistRelationId,
    polynomial: DoryReducePolynomial,
) -> DoryAssistOpeningId {
    DoryAssistOpeningId::virtual_polynomial(
        DoryAssistVirtualPolynomial::DoryReduce(polynomial),
        relation,
    )
}

fn set_dory_reduce_transition_opening(
    assist_proof: &mut DoryAssistProof,
    opening: DoryAssistOpeningId,
    value: Fq,
) {
    let claim = assist_proof
        .claims
        .stage1
        .dory_reduce
        .transitions
        .iter_mut()
        .find(|claim| claim.id == opening)
        .expect("fixture contains Dory-reduce transition opening");
    claim.value = value;
}

fn set_dory_reduce_opening_claim(
    claims: &mut [DoryAssistOpeningClaim],
    opening: DoryAssistOpeningId,
    value: Fq,
) {
    let claim = claims
        .iter_mut()
        .find(|claim| claim.id == opening)
        .expect("fixture contains Dory-reduce relation opening");
    claim.value = value;
}

fn set_dory_reduce_opening(
    assist_proof: &mut DoryAssistProof,
    opening: DoryAssistOpeningId,
    value: Fq,
) {
    if opening == dory_reduce::s1_accumulator_opening() {
        assist_proof
            .claims
            .stage1
            .dory_reduce
            .scalar_fold
            .s1_accumulator = value;
    } else if opening == dory_reduce::s1_next_accumulator_opening() {
        assist_proof
            .claims
            .stage1
            .dory_reduce
            .scalar_fold
            .s1_next_accumulator = value;
    } else if opening == dory_reduce::s1_fold_factor_opening() {
        assist_proof
            .claims
            .stage1
            .dory_reduce
            .scalar_fold
            .s1_fold_factor = value;
    } else if opening == dory_reduce::s2_accumulator_opening() {
        assist_proof
            .claims
            .stage1
            .dory_reduce
            .scalar_fold
            .s2_accumulator = value;
    } else if opening == dory_reduce::s2_next_accumulator_opening() {
        assist_proof
            .claims
            .stage1
            .dory_reduce
            .scalar_fold
            .s2_next_accumulator = value;
    } else if opening == dory_reduce::s2_fold_factor_opening() {
        assist_proof
            .claims
            .stage1
            .dory_reduce
            .scalar_fold
            .s2_fold_factor = value;
    } else {
        match opening_relation(opening) {
            DoryAssistRelationId::DoryReduceGtTransition
            | DoryAssistRelationId::DoryReduceG1Transition
            | DoryAssistRelationId::DoryReduceG2Transition => {
                set_dory_reduce_transition_opening(assist_proof, opening, value);
            }
            DoryAssistRelationId::DoryReduceStateChain => set_dory_reduce_opening_claim(
                &mut assist_proof.claims.stage1.dory_reduce.state_chain,
                opening,
                value,
            ),
            DoryAssistRelationId::DoryReduceBoundary => set_dory_reduce_opening_claim(
                &mut assist_proof.claims.stage1.dory_reduce.boundary,
                opening,
                value,
            ),
            relation => panic!("fixture cannot set non-Dory-reduce opening {relation:?}"),
        }
    }
}

fn get_dory_reduce_opening(assist_proof: &DoryAssistProof, opening: DoryAssistOpeningId) -> Fq {
    assist_proof
        .claims
        .stage1
        .opening_claim(&opening)
        .expect("fixture Dory-reduce opening exists")
}

fn opening_relation(opening: DoryAssistOpeningId) -> DoryAssistRelationId {
    let DoryAssistOpeningId::Polynomial { relation, .. } = opening;
    relation
}

fn add_to_dory_reduce_transition_opening(
    assist_proof: &mut DoryAssistProof,
    opening: DoryAssistOpeningId,
    delta: Fq,
) {
    let claim = assist_proof
        .claims
        .stage1
        .dory_reduce
        .transitions
        .iter_mut()
        .find(|claim| claim.id == opening)
        .expect("fixture contains Dory-reduce transition opening");
    claim.value += delta;
}

struct Stage1RelationContextForFixture {
    relation_challenges: Vec<(DoryAssistChallengeId, Fq)>,
    sumcheck_point: Vec<Fq>,
}

fn stage1_relation_context_for_fixture<T>(
    case: &DoryAssistVerifierCase,
    zk: bool,
    target: DoryAssistRelationId,
) -> Stage1RelationContextForFixture
where
    T: Transcript<Challenge = Fr>,
{
    let mut transcript = T::new(b"dory-assist-oracle");
    let _ = absorb_checked_inputs_for_fixture(case, zk, &mut transcript);
    let _ = squeeze_checked_input_digest_for_fixture(&mut transcript);
    absorb_stage1_preamble_for_fixture(
        if zk { &b"zk"[..] } else { &b"clear"[..] },
        case.assist_proof.stages.stage1.relation_count(),
        &mut transcript,
    );

    let protocol = protocol_claims::<Fq>(case.assist_proof.dimensions);
    for relation in &case.assist_proof.stages.stage1.relations {
        let relation_claims = protocol
            .relation(relation.id)
            .expect("stage 1 relation belongs to Dory-assist protocol");
        absorb_stage1_relation_for_fixture(relation.id, &relation.sumcheck, &mut transcript);
        let relation_challenges = relation_claims
            .required_challenges()
            .into_iter()
            .map(|id| (id, squeeze_fq_for_fixture(&mut transcript)))
            .collect::<Vec<_>>();

        if relation.id == target {
            let mut sumcheck_point = Vec::with_capacity(relation.sumcheck.rounds);
            for round_proof in &relation.sumcheck_proof.round_polynomials {
                absorb_sumcheck_round_for_fixture(round_proof, &mut transcript);
                sumcheck_point.push(squeeze_fq_for_fixture(&mut transcript));
            }
            return Stage1RelationContextForFixture {
                relation_challenges,
                sumcheck_point,
            };
        }

        for round_proof in &relation.sumcheck_proof.round_polynomials {
            absorb_sumcheck_round_for_fixture(round_proof, &mut transcript);
            let _ = squeeze_fq_for_fixture(&mut transcript);
        }
        for id in relation_claims.required_openings() {
            let value = case
                .assist_proof
                .claims
                .stage1
                .opening_claim(&id)
                .expect("fixture has canonical opening claim");
            transcript.append_labeled(b"opening_claim", &value);
        }
    }

    panic!("target relation {target:?} is absent from the canonical Stage 1 catalog");
}

fn sumcheck_linear_factor(point: &[Fq]) -> Fq {
    point
        .iter()
        .copied()
        .fold(Fq::from_u64(1), |acc, challenge| acc * challenge)
}

fn accumulator_zero_sumcheck_kernel<T>(case: &DoryAssistVerifierCase, zk: bool) -> Fq
where
    T: Transcript<Challenge = Fr>,
{
    let mut transcript = T::new(b"dory-assist-oracle");
    let _ = absorb_checked_inputs_for_fixture(case, zk, &mut transcript);
    let _ = squeeze_checked_input_digest_for_fixture(&mut transcript);
    absorb_stage1_preamble_for_fixture(
        if zk { &b"zk"[..] } else { &b"clear"[..] },
        case.assist_proof.stages.stage1.relation_count(),
        &mut transcript,
    );

    let protocol = protocol_claims::<Fq>(case.assist_proof.dimensions);
    for relation in &case.assist_proof.stages.stage1.relations {
        let relation_claims = protocol
            .relation(relation.id)
            .expect("stage 1 relation belongs to Dory-assist protocol");
        absorb_stage1_relation_for_fixture(relation.id, &relation.sumcheck, &mut transcript);
        let relation_challenges = relation_claims
            .required_challenges()
            .into_iter()
            .map(|id| (id, squeeze_fq_for_fixture(&mut transcript)))
            .collect::<Vec<_>>();

        if relation.id == DoryAssistRelationId::MillerLoopAccumulator {
            let input_claim = relation_claims
                .input
                .expression()
                .try_evaluate(
                    |id| {
                        case.assist_proof
                            .claims
                            .stage1
                            .opening_claim(id)
                            .ok_or("missing opening")
                    },
                    |id| resolve_fixture_challenge(&relation_challenges, id),
                    |id| {
                        case.assist_proof
                            .claims
                            .stage1
                            .public_claim(id)
                            .ok_or("missing public")
                    },
                )
                .expect("fixture accumulator input evaluates");
            let final_claim = relation.sumcheck_proof.round_polynomials.iter().fold(
                input_claim,
                |running_sum, round_proof| {
                    absorb_sumcheck_round_for_fixture(round_proof, &mut transcript);
                    let challenge = squeeze_fq_for_fixture(&mut transcript);
                    round_proof.evaluate_with_hint(running_sum, challenge)
                },
            );
            return final_claim
                * input_claim
                    .inverse()
                    .expect("native-output fixture has nonzero accumulator claim");
        }

        for round_proof in &relation.sumcheck_proof.round_polynomials {
            absorb_sumcheck_round_for_fixture(round_proof, &mut transcript);
            let _ = squeeze_fq_for_fixture(&mut transcript);
        }
        for id in relation_claims.required_openings() {
            let value = case
                .assist_proof
                .claims
                .stage1
                .opening_claim(&id)
                .expect("fixture has canonical opening claim");
            transcript.append_labeled(b"opening_claim", &value);
        }
    }

    panic!("canonical Stage 1 relation catalog has no Miller-loop accumulator relation");
}

fn stage1_zero_sumcheck_final_claim(
    case: &DoryAssistVerifierCase,
    zk: bool,
    target: DoryAssistRelationId,
) -> Fq {
    stage1_zero_sumcheck_final_claim_with_transcript::<Blake2bTranscript<Fr>>(case, zk, target)
}

fn stage1_zero_sumcheck_final_claim_with_transcript<T>(
    case: &DoryAssistVerifierCase,
    zk: bool,
    target: DoryAssistRelationId,
) -> Fq
where
    T: Transcript<Challenge = Fr>,
{
    let mut transcript = T::new(b"dory-assist-oracle");
    let _ = absorb_checked_inputs_for_fixture(case, zk, &mut transcript);
    let _ = squeeze_checked_input_digest_for_fixture(&mut transcript);
    absorb_stage1_preamble_for_fixture(
        if zk { &b"zk"[..] } else { &b"clear"[..] },
        case.assist_proof.stages.stage1.relation_count(),
        &mut transcript,
    );

    let protocol = protocol_claims::<Fq>(case.assist_proof.dimensions);
    for relation in &case.assist_proof.stages.stage1.relations {
        let relation_claims = protocol
            .relation(relation.id)
            .expect("stage 1 relation belongs to Dory-assist protocol");
        absorb_stage1_relation_for_fixture(relation.id, &relation.sumcheck, &mut transcript);
        let relation_challenges = relation_claims
            .required_challenges()
            .into_iter()
            .map(|id| (id, squeeze_fq_for_fixture(&mut transcript)))
            .collect::<Vec<_>>();

        let input_claim = relation_claims
            .input
            .expression()
            .try_evaluate(
                |id| {
                    case.assist_proof
                        .claims
                        .stage1
                        .opening_claim(id)
                        .ok_or("missing opening")
                },
                |id| resolve_fixture_challenge(&relation_challenges, id),
                |id| {
                    case.assist_proof
                        .claims
                        .stage1
                        .public_claim(id)
                        .ok_or("missing public")
                },
            )
            .expect("fixture input evaluates");
        let final_claim = relation.sumcheck_proof.round_polynomials.iter().fold(
            input_claim,
            |running_sum, round_proof| {
                absorb_sumcheck_round_for_fixture(round_proof, &mut transcript);
                let challenge = squeeze_fq_for_fixture(&mut transcript);
                round_proof.evaluate_with_hint(running_sum, challenge)
            },
        );

        if relation.id == target {
            return final_claim;
        }

        for id in relation_claims.required_openings() {
            let value = case
                .assist_proof
                .claims
                .stage1
                .opening_claim(&id)
                .expect("fixture has canonical opening claim");
            transcript.append_labeled(b"opening_claim", &value);
        }
    }

    panic!("target relation {target:?} is absent from the canonical Stage 1 catalog");
}

fn resolve_fixture_challenge(
    challenges: &[(DoryAssistChallengeId, Fq)],
    id: &DoryAssistChallengeId,
) -> Result<Fq, &'static str> {
    challenges
        .iter()
        .find(|(candidate, _)| candidate == id)
        .map(|(_, value)| *value)
        .ok_or("missing challenge")
}

fn absorb_checked_inputs_for_fixture(
    case: &DoryAssistVerifierCase,
    zk: bool,
    transcript: &mut impl Transcript<Challenge = Fr>,
) -> DoryAssistInputPublicClaims {
    let dory_verifier_scalars =
        dory_verifier_transcript_scalar_claims_for_fixture(case, transcript);
    let mut input_public_claims = DoryAssistInputPublicClaims::default();
    transcript.append(&Label(b"DoryAssist"));
    transcript.append(&Label(b"checked_inputs"));
    transcript.append(&Label(if zk { &b"zk"[..] } else { &b"clear"[..] }));
    transcript.append(&Label(b"dory_assist_setup"));
    transcript.append(&case.verifier_setup);
    input_public_claims.verifier_setup_digest =
        forked_fq_challenge_for_fixture(transcript, b"dory_assist_setup_digest");
    append_dory_verifier_setup_artifacts_for_fixture(
        &mut input_public_claims.verifier_setup_artifacts,
        &case.verifier_setup,
    );

    transcript.append(&Label(b"dory_assist_pcs_proof"));
    transcript.append(&case.pcs_proof);
    input_public_claims
        .dory_proof_artifacts
        .push(forked_fq_challenge_for_fixture(
            transcript,
            b"dory_assist_proof_digest",
        ));
    append_dory_proof_artifacts_for_fixture(
        &mut input_public_claims.dory_proof_artifacts,
        &case.pcs_proof,
    );

    transcript.append(&Label(b"dory_assist_commitment"));
    transcript.append(&case.commitment);
    input_public_claims
        .jolt_commitments
        .push(forked_fq_challenge_for_fixture(
            transcript,
            b"dory_assist_commitment_digest",
        ));
    input_public_claims
        .jolt_commitments
        .extend(gt_artifact_coefficients_for_fixture(&case.commitment.0));

    transcript.append(&LabelWithCount(
        b"dory_assist_point",
        case.point.len() as u64,
    ));
    for point_coordinate in &case.point {
        transcript.append(point_coordinate);
    }
    input_public_claims
        .transcript_scalars
        .extend(case.point.iter().copied().map(inject_fr_to_fq_for_fixture));
    input_public_claims
        .transcript_scalars
        .extend(dory_verifier_scalars);

    if !zk {
        transcript.append(&Label(b"dory_assist_eval"));
        transcript.append(&case.eval);
        input_public_claims
            .jolt_evaluation_claims
            .push(inject_fr_to_fq_for_fixture(case.eval));
        input_public_claims
            .dory_reduce_initial_e2
            .extend(g2_artifact_coordinates_for_fixture(
                case.verifier_setup.artifacts().g2_0.scalar_mul(&case.eval),
            ));
    } else {
        input_public_claims
            .dory_reduce_initial_e2
            .extend(g2_artifact_coordinates_for_fixture(
                case.pcs_proof.zk_artifacts().e2.unwrap_or_default(),
            ));
    }

    input_public_claims
}

fn dory_verifier_transcript_scalar_claims_for_fixture<T>(
    case: &DoryAssistVerifierCase,
    transcript: &T,
) -> Vec<Fq>
where
    T: Transcript<Challenge = Fr>,
{
    let scalars = case
        .pcs_proof
        .verifier_transcript_scalars(transcript, &case.point);
    let mut claims = Vec::with_capacity(
        8 * scalars.reduce_rounds.len() + 4 + usize::from(scalars.scalar_product_sigma_c.is_some()),
    );
    for round in scalars.reduce_rounds {
        claims.push(inject_fr_to_fq_for_fixture(round.beta));
        claims.push(inject_fr_to_fq_for_fixture(round.beta_inverse));
        claims.push(inject_fr_to_fq_for_fixture(round.alpha));
        claims.push(inject_fr_to_fq_for_fixture(round.alpha_inverse));
        claims.push(inject_fr_to_fq_for_fixture(round.alpha_beta));
        claims.push(inject_fr_to_fq_for_fixture(
            round.alpha_inverse_beta_inverse,
        ));
        claims.push(inject_fr_to_fq_for_fixture(round.s1_fold_factor));
        claims.push(inject_fr_to_fq_for_fixture(round.s2_fold_factor));
    }
    claims.push(inject_fr_to_fq_for_fixture(scalars.gamma));
    claims.push(inject_fr_to_fq_for_fixture(scalars.gamma_inverse));
    if let Some(sigma_c) = scalars.scalar_product_sigma_c {
        claims.push(inject_fr_to_fq_for_fixture(sigma_c));
    }
    claims.push(inject_fr_to_fq_for_fixture(scalars.d));
    claims.push(inject_fr_to_fq_for_fixture(scalars.d_inverse));
    claims.push(inject_fr_to_fq_for_fixture(scalars.d_squared));
    claims
}

fn append_dory_proof_artifacts_for_fixture(artifacts: &mut Vec<Fq>, pcs_proof: &DoryProof) {
    let vmv = pcs_proof.vmv_artifacts();
    artifacts.extend(gt_artifact_coefficients_for_fixture(&vmv.c));
    artifacts.extend(gt_artifact_coefficients_for_fixture(&vmv.d2));
    artifacts.extend(g1_artifact_coordinates_for_fixture(vmv.e1));

    let zk = pcs_proof.zk_artifacts();
    artifacts.extend(match zk.e2 {
        Some(e2) => g2_artifact_coordinates_for_fixture(e2),
        None => identity_g2_artifact_coordinates_for_fixture(),
    });
    artifacts.extend(match zk.y_com {
        Some(y_com) => g1_artifact_coordinates_for_fixture(y_com),
        None => identity_g1_artifact_coordinates_for_fixture(),
    });
    if let Some(scalar_product) = pcs_proof.scalar_product_artifacts() {
        artifacts.extend(gt_artifact_coefficients_for_fixture(&scalar_product.p1));
        artifacts.extend(gt_artifact_coefficients_for_fixture(&scalar_product.p2));
        artifacts.extend(gt_artifact_coefficients_for_fixture(&scalar_product.q));
        artifacts.extend(gt_artifact_coefficients_for_fixture(&scalar_product.r));
        artifacts.extend(g1_artifact_coordinates_for_fixture(scalar_product.e1));
        artifacts.extend(g2_artifact_coordinates_for_fixture(scalar_product.e2));
        artifacts.push(inject_fr_to_fq_for_fixture(scalar_product.r1));
        artifacts.push(inject_fr_to_fq_for_fixture(scalar_product.r2));
        artifacts.push(inject_fr_to_fq_for_fixture(scalar_product.r3));
    } else {
        let identity_gt = Bn254GT::default();
        artifacts.extend(gt_artifact_coefficients_for_fixture(&identity_gt));
        artifacts.extend(gt_artifact_coefficients_for_fixture(&identity_gt));
        artifacts.extend(gt_artifact_coefficients_for_fixture(&identity_gt));
        artifacts.extend(gt_artifact_coefficients_for_fixture(&identity_gt));
        artifacts.extend(identity_g1_artifact_coordinates_for_fixture());
        artifacts.extend(identity_g2_artifact_coordinates_for_fixture());
        artifacts.extend([Fq::default(), Fq::default(), Fq::default()]);
    }

    for round in pcs_proof.reduce_round_artifacts() {
        artifacts.extend(gt_artifact_coefficients_for_fixture(&round.first.d1_left));
        artifacts.extend(gt_artifact_coefficients_for_fixture(&round.first.d1_right));
        artifacts.extend(gt_artifact_coefficients_for_fixture(&round.first.d2_left));
        artifacts.extend(gt_artifact_coefficients_for_fixture(&round.first.d2_right));
        artifacts.extend(g1_artifact_coordinates_for_fixture(round.first.e1_beta));
        artifacts.extend(g2_artifact_coordinates_for_fixture(round.first.e2_beta));

        artifacts.extend(gt_artifact_coefficients_for_fixture(&round.second.c_plus));
        artifacts.extend(gt_artifact_coefficients_for_fixture(&round.second.c_minus));
        artifacts.extend(g1_artifact_coordinates_for_fixture(round.second.e1_plus));
        artifacts.extend(g1_artifact_coordinates_for_fixture(round.second.e1_minus));
        artifacts.extend(g2_artifact_coordinates_for_fixture(round.second.e2_plus));
        artifacts.extend(g2_artifact_coordinates_for_fixture(round.second.e2_minus));
    }

    let final_artifacts = pcs_proof.final_artifacts();
    artifacts.extend(g1_artifact_coordinates_for_fixture(final_artifacts.e1));
    artifacts.extend(g2_artifact_coordinates_for_fixture(final_artifacts.e2));
}

fn append_dory_verifier_setup_artifacts_for_fixture(
    artifacts: &mut Vec<Fq>,
    setup: &DoryVerifierSetup,
) {
    let setup_artifacts = setup.artifacts();
    for value in &setup_artifacts.chi {
        artifacts.extend(gt_artifact_coefficients_for_fixture(value));
    }
    for value in &setup_artifacts.delta_1l {
        artifacts.extend(gt_artifact_coefficients_for_fixture(value));
    }
    for value in &setup_artifacts.delta_1r {
        artifacts.extend(gt_artifact_coefficients_for_fixture(value));
    }
    for value in &setup_artifacts.delta_2l {
        artifacts.extend(gt_artifact_coefficients_for_fixture(value));
    }
    for value in &setup_artifacts.delta_2r {
        artifacts.extend(gt_artifact_coefficients_for_fixture(value));
    }
    artifacts.extend(g1_artifact_coordinates_for_fixture(setup_artifacts.g1_0));
    artifacts.extend(g2_artifact_coordinates_for_fixture(setup_artifacts.g2_0));
    artifacts.extend(g1_artifact_coordinates_for_fixture(setup_artifacts.h1));
    artifacts.extend(g2_artifact_coordinates_for_fixture(setup_artifacts.h2));
    artifacts.extend(gt_artifact_coefficients_for_fixture(&setup_artifacts.ht));
}

fn gt_artifact_coefficients_for_fixture(value: &Bn254GT) -> [Fq; 16] {
    let mut coefficients = [Fq::default(); 16];
    coefficients[..Bn254GT::FQ12_COEFFICIENTS].copy_from_slice(&value.fq12_coefficients());
    coefficients
}

fn g1_artifact_coordinates_for_fixture(value: Bn254G1) -> [Fq; 3] {
    value.affine_coordinates_with_infinity()
}

fn g2_artifact_coordinates_for_fixture(value: Bn254G2) -> [Fq; 5] {
    value.affine_coordinates_with_infinity()
}

fn identity_g1_artifact_coordinates_for_fixture() -> [Fq; 3] {
    [Fq::default(), Fq::default(), Fq::from_u64(1)]
}

fn identity_g2_artifact_coordinates_for_fixture() -> [Fq; 5] {
    [
        Fq::default(),
        Fq::default(),
        Fq::default(),
        Fq::default(),
        Fq::from_u64(1),
    ]
}

fn checked_input_public_claims_for_fixture<T>(
    case: &DoryAssistVerifierCase,
    zk: bool,
) -> DoryAssistInputPublicClaims
where
    T: Transcript<Challenge = Fr>,
{
    let mut transcript = T::new(b"dory-assist-oracle");
    let mut input_public_claims = absorb_checked_inputs_for_fixture(case, zk, &mut transcript);
    input_public_claims.checked_input_digest =
        squeeze_checked_input_digest_for_fixture(&mut transcript);
    input_public_claims
}

fn squeeze_checked_input_digest_for_fixture(
    transcript: &mut impl Transcript<Challenge = Fr>,
) -> Fq {
    transcript.append(&Label(b"dory_assist_checked_input_digest"));
    squeeze_fq_for_fixture(transcript)
}

fn forked_fq_challenge_for_fixture<T>(transcript: &T, label: &'static [u8]) -> Fq
where
    T: Transcript<Challenge = Fr>,
{
    let mut fork = transcript.clone();
    fork.append(&Label(label));
    squeeze_fq_for_fixture(&mut fork)
}

fn inject_fr_to_fq_for_fixture(value: Fr) -> Fq {
    let mut bytes = [0_u8; Fr::NUM_BYTES];
    value.to_bytes_le(&mut bytes);
    Fq::from_le_bytes_mod_order(&bytes)
}

fn absorb_stage1_preamble_for_fixture(
    mode_name: &'static [u8],
    relation_count: u32,
    transcript: &mut impl Transcript<Challenge = Fr>,
) {
    transcript.append(&Label(b"dory_assist_stage1"));
    transcript.append(&Label(mode_name));
    transcript.append(&Label(b"stage1_relations"));
    transcript.append(&U64Word(relation_count as u64));
}

fn absorb_stage1_relation_for_fixture(
    id: DoryAssistRelationId,
    sumcheck: &jolt_claims::protocols::dory_assist::DoryAssistSumcheckSpec,
    transcript: &mut impl Transcript<Challenge = Fr>,
) {
    transcript.append(&Label(b"stage1_relation_id"));
    transcript.append(&U64Word(relation_transcript_tag_for_fixture(id) as u64));
    transcript.append(&Label(b"stage1_sumcheck_domain"));
    transcript.append(&U64Word(0));
    transcript.append(&Label(b"stage1_sumcheck_rounds"));
    transcript.append(&U64Word(sumcheck.rounds as u64));
    transcript.append(&Label(b"stage1_sumcheck_degree"));
    transcript.append(&U64Word(sumcheck.degree as u64));
}

fn absorb_sumcheck_round_for_fixture(
    round_proof: &jolt_poly::CompressedPoly<Fq>,
    transcript: &mut impl Transcript<Challenge = Fr>,
) {
    let coeffs = round_proof.coeffs_except_linear_term();
    transcript.append(&LabelWithCount(
        SUMCHECK_ROUND_TRANSCRIPT_LABEL,
        coeffs.len() as u64,
    ));
    for coeff in coeffs {
        transcript.append(coeff);
    }
}

fn relation_transcript_tag_for_fixture(id: DoryAssistRelationId) -> usize {
    CANONICAL_RELATION_ORDER
        .iter()
        .position(|candidate| *candidate == id)
        .expect("stage 1 relation has a canonical transcript tag")
}

fn squeeze_fq_for_fixture(transcript: &mut impl Transcript<Challenge = Fr>) -> Fq {
    let value = transcript.challenge_scalar();
    let mut bytes = [0_u8; Fr::NUM_BYTES];
    value.to_bytes_le(&mut bytes);
    Fq::from_le_bytes_mod_order(&bytes)
}

fn populate_valid_hyrax_opening(assist_proof: &mut DoryAssistProof) {
    let reduced_claims = reduced_opening_claims(assist_proof);
    let poly_len = reduced_claims.len().next_power_of_two();
    let num_vars = poly_len.trailing_zeros() as usize;
    let row_vars = num_vars / 2;
    let col_vars = num_vars - row_vars;
    let dimensions =
        HyraxDimensions::new(num_vars, row_vars, col_vars).expect("valid Hyrax dimensions");
    let hyrax_setup = derive_hyrax_prover_setup(dimensions).expect("seed-derived Hyrax setup");
    let mut evaluations = vec![Fq::default(); poly_len];
    for (slot, claim) in evaluations.iter_mut().zip(&reduced_claims) {
        *slot = claim.value;
    }
    let packed_poly = Polynomial::<Fq>::from(evaluations);
    let packed_point = (0..num_vars)
        .map(|index| Fq::from_u64(13 + 6 * index as u64))
        .collect::<Vec<_>>();
    let packed_eval = packed_poly.evaluate(&packed_point);
    let (dense_commitment, hint) = DoryAssistHyrax::commit(&packed_poly, &hyrax_setup);
    let mut transcript = Blake2bTranscript::new(b"dory-assist-hyrax-fixture");
    let opening_proof = DoryAssistHyrax::open(
        &packed_poly,
        &packed_point,
        packed_eval,
        &hyrax_setup,
        Some(hint),
        &mut transcript,
    );

    assist_proof.stages.stage3.packed_eval = packed_eval;
    assist_proof.stages.stage3.reduced_openings =
        reduced_claims.iter().map(|claim| claim.id).collect();
    assist_proof.claims.opening.packed_point = packed_point;
    assist_proof.claims.opening.packed_eval = packed_eval;
    assist_proof.opening_proof = opening_proof;
    assist_proof.dense_commitment = dense_commitment;
}

fn reduced_opening_claims(assist_proof: &DoryAssistProof) -> Vec<DoryAssistOpeningClaim> {
    let protocol = protocol_claims::<Fq>(assist_proof.dimensions);
    let mut reduced_claims = Vec::new();
    for relation in &assist_proof.stages.stage1.relations {
        let relation_claims = protocol
            .relation(relation.id)
            .expect("stage 1 relation belongs to Dory-assist protocol");
        for id in relation_claims.required_openings() {
            if reduced_claims
                .iter()
                .any(|claim: &DoryAssistOpeningClaim| claim.id == id)
            {
                continue;
            }
            let value = assist_proof
                .claims
                .stage1
                .opening_claim(&id)
                .expect("stage 1 claim value exists for canonical opening");
            reduced_claims.push(DoryAssistOpeningClaim { id, value });
        }
    }
    reduced_claims
}
