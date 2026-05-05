//! Malformed proof fixtures for verifier rejection tests.

#![expect(
    clippy::expect_used,
    reason = "tamper gates should fail fast when honest prefix setup fails"
)]

use jolt_dory::{DoryProof, DoryScheme};
use jolt_field::{Field, Fr};
use jolt_kernels::stage1::{
    Stage1CpuProgramPlan as KernelStage1CpuProgramPlan, Stage1ExecutionArtifacts, Stage1Proof,
};
use jolt_kernels::stage2::{
    Stage2CpuProgramPlan as KernelStage2CpuProgramPlan, Stage2ExecutionArtifacts,
    Stage2OpeningInputValue, Stage2Proof, Stage2RamData,
};
use jolt_kernels::stage3::Stage3ExecutionArtifacts;
use jolt_kernels::stage4::Stage4ExecutionArtifacts;
use jolt_openings::CommitmentScheme;
use jolt_poly::Polynomial;
use jolt_poly::UnivariatePoly;
use jolt_transcript::Transcript;
use jolt_verifier::{
    JoltStage6VerifierData, JoltStageOpeningInputValue, JoltStageProof, Stage1VerifierProgramPlan,
    Stage2VerifierProgramPlan, Stage3VerifierProgramPlan, Stage4VerifierProgramPlan,
    Stage5VerifierProgramPlan, Stage6VerifierProgramPlan, Stage7VerifierProgramPlan,
};

use crate::checkpoint::assert_state_history_match;
use crate::commitment_oracle::{
    transcript_with_bolt_commitment_trace, transcript_with_bolt_preamble, BoltCommitmentTrace,
    BoltPreambleSource, BoltTranscript,
};
use crate::core_oracle::CoreMuldivCommitmentFixture;

fn stage2_opening_inputs_from_artifacts(
    program: &'static KernelStage2CpuProgramPlan,
    stage1_artifacts: &Stage1ExecutionArtifacts<Fr>,
) -> Vec<Stage2OpeningInputValue<Fr>> {
    jolt_prover::stage2_opening_inputs_from_artifacts(program, stage1_artifacts)
        .expect("generated prover derives Stage 2 opening inputs from artifacts")
}

macro_rules! tampered_sumcheck_coefficient {
    ($proof:expr, $sumcheck_index:expr) => {{
        let mut tampered = $proof.clone();
        let round_poly = &mut tampered.sumchecks[$sumcheck_index].proof.round_polynomials[0];
        let mut coefficients = round_poly.clone().into_coefficients();
        coefficients[0] += Fr::from_u64(1);
        *round_poly = UnivariatePoly::new(coefficients);
        tampered
    }};
}

macro_rules! tampered_sumcheck_eval {
    ($proof:expr, $sumcheck_index:expr) => {{
        let mut tampered = $proof.clone();
        tampered.sumchecks[$sumcheck_index].evals[0].value += Fr::from_u64(1);
        tampered
    }};
}

macro_rules! tampered_sumcheck_point {
    ($proof:expr, $sumcheck_index:expr) => {{
        let mut tampered = $proof.clone();
        tampered.sumchecks[$sumcheck_index].point[0] += Fr::from_u64(1);
        tampered
    }};
}

macro_rules! assert_batched_sumcheck_tampers {
    ($assert_fn:ident, $proof:expr, $sumcheck_index:expr, $prefix:literal) => {{
        $assert_fn(
            tampered_sumcheck_coefficient!($proof, $sumcheck_index),
            concat!($prefix, " accepted a tampered batched sumcheck coefficient"),
        );
        $assert_fn(
            tampered_sumcheck_eval!($proof, $sumcheck_index),
            concat!($prefix, " accepted a tampered batched opening evaluation"),
        );
        $assert_fn(
            tampered_sumcheck_point!($proof, $sumcheck_index),
            concat!($prefix, " accepted a tampered batched sumcheck point"),
        );
    }};
}

fn assert_opening_input_tampers<F>(
    openings: &[jolt_verifier::JoltStageOpeningInputValue],
    stage: &str,
    mut assert_tamper_rejected: F,
) where
    F: FnMut(&[jolt_verifier::JoltStageOpeningInputValue], &str),
{
    let tampered_openings = tampered_opening_input_eval(openings);
    assert_tamper_rejected(
        &tampered_openings,
        &format!("{stage} verifier accepted a tampered opening-claim input evaluation"),
    );
    let missing_openings = opening_inputs_without_first(openings);
    assert_tamper_rejected(
        &missing_openings,
        &format!("{stage} verifier accepted a missing opening-claim input"),
    );
    let extra_openings = opening_inputs_with_extra_first(openings);
    assert_tamper_rejected(
        &extra_openings,
        &format!("{stage} verifier accepted an extra opening-claim input"),
    );
    let short_point_openings = opening_inputs_with_short_first_point(openings);
    assert_tamper_rejected(
        &short_point_openings,
        &format!("{stage} verifier accepted an opening-claim input with invalid point arity"),
    );
}

fn tampered_opening_input_eval(
    openings: &[jolt_verifier::JoltStageOpeningInputValue],
) -> Vec<jolt_verifier::JoltStageOpeningInputValue> {
    let mut tampered = openings.to_vec();
    tampered
        .get_mut(0)
        .expect("stage has at least one opening input")
        .eval += Fr::from_u64(1);
    tampered
}

fn opening_inputs_without_first<T: Clone>(openings: &[T]) -> Vec<T> {
    assert!(!openings.is_empty(), "stage has at least one opening input");
    openings[1..].to_vec()
}

fn opening_inputs_with_extra_first<T: Clone>(openings: &[T]) -> Vec<T> {
    let mut tampered = openings.to_vec();
    tampered.push(
        openings
            .first()
            .expect("stage has at least one opening input")
            .clone(),
    );
    tampered
}

fn opening_inputs_with_short_first_point(
    openings: &[jolt_verifier::JoltStageOpeningInputValue],
) -> Vec<jolt_verifier::JoltStageOpeningInputValue> {
    let mut tampered = openings.to_vec();
    let _removed_coordinate = tampered
        .first_mut()
        .expect("stage has at least one opening input")
        .point
        .pop()
        .expect("opening input has at least one point coordinate");
    tampered
}

fn tampered_opening_input_suffix_point(
    openings: &[jolt_verifier::JoltStageOpeningInputValue],
    symbol: &'static str,
    prefix_len: usize,
) -> Vec<jolt_verifier::JoltStageOpeningInputValue> {
    let mut tampered = openings.to_vec();
    let point = tampered
        .iter_mut()
        .find(|opening| opening.symbol == symbol)
        .expect("stage has opening input symbol")
        .point
        .get_mut(prefix_len)
        .expect("opening input has a suffix point coordinate");
    *point += Fr::from_u64(1);
    tampered
}

fn assert_generated_jolt_prefix_tamper_rejected<P>(
    preamble: &P,
    proof: &jolt_verifier::JoltProof,
    inputs: jolt_verifier::JoltVerifierInputs<'_>,
    programs: jolt_verifier::JoltVerifierPrograms,
    message: &str,
) where
    P: BoltPreambleSource,
{
    let mut transcript = transcript_with_bolt_preamble(preamble);
    let result = if !inputs.stage7_openings.is_empty() {
        jolt_verifier::verify_jolt_through_stage7_with_programs(
            proof,
            inputs,
            programs,
            &mut transcript,
        )
    } else if !inputs.stage6_openings.is_empty() {
        jolt_verifier::verify_jolt_through_stage6_with_programs(
            proof,
            inputs,
            programs,
            &mut transcript,
        )
    } else {
        jolt_verifier::verify_jolt_through_stage5_with_programs(
            proof,
            inputs,
            programs,
            &mut transcript,
        )
    };
    assert!(result.is_err(), "generated monolithic {message}");
}

macro_rules! assert_generated_stage_and_prefix_tamper_rejected {
    ($preamble:expr, $message:expr, $inputs:expr, $programs:expr, $stage_result:expr, $prefix_proof:expr $(,)?) => {{
        assert!($stage_result.is_err(), "generated {}", $message);
        assert_generated_jolt_prefix_tamper_rejected(
            $preamble,
            &$prefix_proof,
            $inputs,
            $programs,
            $message,
        );
    }};
}

/// Produces a valid Dory proof for an unrelated polynomial/opening claim.
fn unrelated_dory_proof() -> DoryProof {
    let prover_setup = DoryScheme::setup_prover(1);
    let poly = Polynomial::new(vec![Fr::from_u64(0), Fr::from_u64(1)]);
    let point = vec![Fr::from_u64(7)];
    let mut transcript = jolt_transcript::Blake2bTranscript::new(b"unrelated-dory-proof");
    DoryScheme::open(
        &poly,
        &point,
        point[0],
        &prover_setup,
        None,
        &mut transcript,
    )
}

pub(crate) struct MonolithicJoltTamperInput<'a> {
    pub(crate) preamble: &'a CoreMuldivCommitmentFixture,
    pub(crate) proof: &'a jolt_verifier::JoltProof,
    pub(crate) inputs: jolt_verifier::JoltVerifierInputs<'a>,
    pub(crate) programs: jolt_verifier::JoltVerifierPrograms,
}

pub(crate) fn assert_monolithic_jolt_tamper_rejected(input: MonolithicJoltTamperInput<'_>) {
    let mut inputs_without_setup = input.inputs;
    inputs_without_setup.evaluation_setup = None;

    let verify = |proof: &jolt_verifier::JoltProof,
                  inputs: jolt_verifier::JoltVerifierInputs<'_>| {
        let mut transcript = transcript_with_bolt_preamble(input.preamble);
        jolt_verifier::verify_jolt_with_programs(proof, inputs, input.programs, &mut transcript)
    };

    macro_rules! assert_verify_error {
        ($result:expr, $error:pat, $message:expr $(,)?) => {
            assert!(matches!($result, $error), $message);
        };
    }

    assert_verify_error!(
        verify(input.proof, inputs_without_setup),
        Err(jolt_verifier::JoltVerifyError::Evaluation(
            jolt_verifier::JoltEvaluationProofError::MissingVerifierSetup
        )),
        "generated monolithic verifier accepted evaluation proof without verifier setup",
    );

    let mut missing_evaluation_proof = input.proof.clone();
    missing_evaluation_proof.evaluation = None;
    assert_verify_error!(
        verify(&missing_evaluation_proof, input.inputs),
        Err(jolt_verifier::JoltVerifyError::Evaluation(
            jolt_verifier::JoltEvaluationProofError::MissingProof
        )),
        "generated monolithic verifier accepted missing evaluation proof with verifier setup",
    );

    assert_verify_error!(
        verify(&missing_evaluation_proof, inputs_without_setup),
        Err(jolt_verifier::JoltVerifyError::Evaluation(
            jolt_verifier::JoltEvaluationProofError::MissingProof
        )),
        "generated monolithic verifier accepted missing evaluation proof without verifier setup",
    );

    let mut tampered_evaluation_proof = input.proof.clone();
    tampered_evaluation_proof
        .evaluation
        .as_mut()
        .expect("evaluation proof")
        .joint_opening_proof = unrelated_dory_proof();
    assert_verify_error!(
        verify(&tampered_evaluation_proof, input.inputs),
        Err(jolt_verifier::JoltVerifyError::Evaluation(_)),
        "generated monolithic verifier accepted a tampered evaluation proof",
    );

    let stage8_source_symbol = input.programs.stage8.evaluation_point_source.source_claim;
    let stage7_address_prefix_len = input
        .proof
        .stage7
        .sumchecks
        .first()
        .expect("monolithic proof has a Stage 7 sumcheck")
        .point
        .len();
    let tampered_stage7_openings = tampered_opening_input_suffix_point(
        input.inputs.stage7_openings,
        stage8_source_symbol,
        stage7_address_prefix_len,
    );
    let mut tampered_stage7_opening_inputs = input.inputs;
    tampered_stage7_opening_inputs.stage7_openings = &tampered_stage7_openings;
    assert_verify_error!(
        verify(input.proof, tampered_stage7_opening_inputs),
        Err(jolt_verifier::JoltVerifyError::Evaluation(_)),
        "generated monolithic verifier accepted a tampered Stage 8 opening-point suffix",
    );

    let mut missing_commitment_proof = input.proof.clone();
    *missing_commitment_proof
        .commitments
        .get_mut(0)
        .expect("monolithic proof has a main commitment") = None;
    assert_verify_error!(
        verify(&missing_commitment_proof, input.inputs),
        Err(jolt_verifier::JoltVerifyError::Commitment(_)),
        "generated monolithic verifier accepted a missing required commitment",
    );

    macro_rules! assert_missing_stage_rejected {
        ($field:ident, $variant:ident, $stage:literal) => {{
            let mut missing_stage_proof = input.proof.clone();
            missing_stage_proof.$field.sumchecks.clear();
            assert_verify_error!(
                verify(&missing_stage_proof, input.inputs),
                Err(jolt_verifier::JoltVerifyError::$variant(_)),
                concat!(
                    "generated monolithic verifier accepted a missing ",
                    $stage,
                    " proof"
                ),
            );
        }};
    }

    assert_missing_stage_rejected!(stage1_outer, Stage1Outer, "Stage 1 outer");
    assert_missing_stage_rejected!(stage2, Stage2, "Stage 2");
    assert_missing_stage_rejected!(stage3, Stage3, "Stage 3");
    assert_missing_stage_rejected!(stage4, Stage4, "Stage 4");
    assert_missing_stage_rejected!(stage5, Stage5, "Stage 5");
    assert_missing_stage_rejected!(stage6, Stage6, "Stage 6");
    assert_missing_stage_rejected!(stage7, Stage7, "Stage 7");

    let mut wrong_stage_slot_proof = input.proof.clone();
    std::mem::swap(
        &mut wrong_stage_slot_proof.stage6,
        &mut wrong_stage_slot_proof.stage7,
    );
    assert_verify_error!(
        verify(&wrong_stage_slot_proof, input.inputs),
        Err(jolt_verifier::JoltVerifyError::Stage6(_)),
        "generated monolithic verifier accepted a stage proof in the wrong slot"
    );
}

pub fn assert_bolt_stage1_tamper_rejected(
    stage1_verifier_plan: &'static KernelStage1CpuProgramPlan,
    generated_stage1_verifier_plan: &'static Stage1VerifierProgramPlan,
    proof: &Stage1Proof<Fr>,
    stage1_start_transcript: &BoltTranscript,
) {
    let assert_stage1_tamper_rejected = |tampered: Stage1Proof<Fr>, message: &str| {
        let mut transcript = stage1_start_transcript.clone();

        let result = jolt_prover::replay_stage1_outer_proof_with_program(
            stage1_verifier_plan,
            &tampered,
            &mut transcript,
        );
        assert!(result.is_err(), "{message}");

        let mut generated_transcript = stage1_start_transcript.clone();
        let generated_tampered = jolt_prover::stage1_outer_proof_from_kernel_proof(&tampered);
        let generated_result = jolt_verifier::verify_stage1_outer_with_program(
            generated_stage1_verifier_plan,
            &generated_tampered,
            &mut generated_transcript,
        );
        assert!(generated_result.is_err(), "generated {message}");
    };

    assert_stage1_tamper_rejected(
        tampered_sumcheck_coefficient!(proof, 1),
        "Bolt Stage 1 verifier accepted a tampered remaining sumcheck coefficient",
    );

    assert_stage1_tamper_rejected(
        tampered_sumcheck_coefficient!(proof, 0),
        "Bolt Stage 1 verifier accepted a tampered uniskip sumcheck coefficient",
    );

    assert_stage1_tamper_rejected(
        tampered_sumcheck_point!(proof, 0),
        "Bolt Stage 1 verifier accepted a tampered uniskip point",
    );

    assert_stage1_tamper_rejected(
        tampered_sumcheck_eval!(proof, 0),
        "Bolt Stage 1 verifier accepted a tampered uniskip eval",
    );

    assert_stage1_tamper_rejected(
        tampered_sumcheck_point!(proof, 1),
        "Bolt Stage 1 verifier accepted a tampered remaining sumcheck point",
    );
}

pub struct BoltStage2ChainVerifierInput<'a> {
    pub fixture: &'a CoreMuldivCommitmentFixture,
    pub commitment_verifier_trace: &'a BoltCommitmentTrace,
    pub stage1_prover_plan: &'static KernelStage1CpuProgramPlan,
    pub stage2_prover_plan: &'static KernelStage2CpuProgramPlan,
    pub generated_stage2_verifier_plan: &'static Stage2VerifierProgramPlan,
    pub stage1_artifacts: &'a Stage1ExecutionArtifacts<Fr>,
    pub stage2_artifacts: &'a Stage2ExecutionArtifacts<Fr>,
    pub ram_data: &'a Stage2RamData<'a>,
    pub prover_transcript: &'a BoltTranscript,
}

pub fn assert_bolt_chain_verifier_accepts_stage2_product_uniskip(
    input: BoltStage2ChainVerifierInput<'_>,
) {
    let mut verifier_transcript =
        transcript_with_bolt_commitment_trace(input.fixture, input.commitment_verifier_trace);

    let stage1_proof = Stage1Proof::from(input.stage1_artifacts.clone());
    let verified_stage1 = jolt_prover::replay_stage1_outer_proof_with_program(
        input.stage1_prover_plan,
        &stage1_proof,
        &mut verifier_transcript,
    )
    .expect("Bolt Stage 1 verifier accepts Bolt prover proof");
    assert_eq!(
        input.stage1_artifacts.sumchecks.len(),
        verified_stage1.sumchecks.len()
    );
    assert_eq!(
        input.stage1_artifacts.opening_values.len(),
        verified_stage1.opening_values.len()
    );

    let stage2_openings =
        stage2_opening_inputs_from_artifacts(input.stage2_prover_plan, &verified_stage1);
    let generated_stage2_openings =
        jolt_prover::verifier_opening_inputs_from_kernel(&stage2_openings);
    let generated_ram_data_storage = jolt_prover::stage2_verifier_ram_data(input.ram_data);
    let generated_ram_data = generated_ram_data_storage.as_input();
    let stage2_proof = Stage2Proof::from(input.stage2_artifacts.clone());
    let stage2_start_transcript = verifier_transcript.clone();
    let verified_stage2 = jolt_prover::replay_stage2_proof_with_program(
        input.stage2_prover_plan,
        &stage2_proof,
        &stage2_openings,
        Some(input.ram_data),
        &mut verifier_transcript,
    )
    .expect("Bolt Stage 2 verifier accepts Bolt prover proof");

    assert_eq!(
        input.stage2_artifacts.sumchecks.len(),
        verified_stage2.sumchecks.len()
    );
    assert_eq!(
        input.stage2_artifacts.sumchecks[0].point,
        verified_stage2.sumchecks[0].point
    );
    assert_eq!(
        input.stage2_artifacts.sumchecks[0].evals[0].value,
        verified_stage2.sumchecks[0].evals[0].value
    );
    assert_state_history_match(input.prover_transcript.log(), verifier_transcript.log());

    let assert_stage2_product_tamper_rejected =
        |tampered_stage2_artifacts: Stage2ExecutionArtifacts<Fr>, message: &str| {
            let mut tamper_transcript = stage2_start_transcript.clone();
            let tampered_stage2_proof = Stage2Proof::from(tampered_stage2_artifacts.clone());
            let tamper_result = jolt_prover::replay_stage2_proof_with_program(
                input.stage2_prover_plan,
                &tampered_stage2_proof,
                &stage2_openings,
                Some(input.ram_data),
                &mut tamper_transcript,
            );
            assert!(tamper_result.is_err(), "{message}");

            let mut generated_tamper_transcript = stage2_start_transcript.clone();
            let generated_tampered_stage2_proof =
                jolt_prover::stage2_proof(&tampered_stage2_artifacts);
            let generated_tamper_result = jolt_verifier::verify_stage2_with_program(
                input.generated_stage2_verifier_plan,
                &generated_tampered_stage2_proof,
                &generated_stage2_openings,
                Some(&generated_ram_data),
                &mut generated_tamper_transcript,
            );
            assert!(generated_tamper_result.is_err(), "generated {message}");
        };

    assert_stage2_product_tamper_rejected(
        tampered_sumcheck_coefficient!(input.stage2_artifacts, 0),
        "Bolt Stage 2 verifier accepted a tampered product uni-skip coefficient",
    );

    assert_stage2_product_tamper_rejected(
        tampered_sumcheck_eval!(input.stage2_artifacts, 0),
        "Bolt Stage 2 verifier accepted a tampered product uni-skip opening evaluation",
    );

    assert_stage2_product_tamper_rejected(
        tampered_sumcheck_point!(input.stage2_artifacts, 0),
        "Bolt Stage 2 verifier accepted a tampered product uni-skip point",
    );
}

pub struct Stage2BatchedTamperInput<'a> {
    pub stage2_prover_plan: &'static KernelStage2CpuProgramPlan,
    pub generated_stage2_verifier_plan: &'static Stage2VerifierProgramPlan,
    pub stage2_start_transcript: &'a BoltTranscript,
    pub stage2_openings: &'a [Stage2OpeningInputValue<Fr>],
    pub stage2_artifacts: &'a Stage2ExecutionArtifacts<Fr>,
    pub ram_data: &'a Stage2RamData<'a>,
}

pub fn assert_bolt_stage2_batched_tamper_rejected(input: Stage2BatchedTamperInput<'_>) {
    let generated_stage2_openings =
        jolt_prover::verifier_opening_inputs_from_kernel(input.stage2_openings);
    let generated_ram_data_storage = jolt_prover::stage2_verifier_ram_data(input.ram_data);
    let generated_ram_data = generated_ram_data_storage.as_input();

    let assert_stage2_tamper_rejected =
        |tampered_stage2_artifacts: Stage2ExecutionArtifacts<Fr>, message: &str| {
            let mut tamper_transcript = input.stage2_start_transcript.clone();
            let tampered_stage2_proof = Stage2Proof::from(tampered_stage2_artifacts.clone());
            let tamper_result = jolt_prover::replay_stage2_proof_with_program(
                input.stage2_prover_plan,
                &tampered_stage2_proof,
                input.stage2_openings,
                Some(input.ram_data),
                &mut tamper_transcript,
            );
            assert!(tamper_result.is_err(), "{message}");

            let mut generated_tamper_transcript = input.stage2_start_transcript.clone();
            let generated_tampered_stage2_proof =
                jolt_prover::stage2_proof(&tampered_stage2_artifacts);
            let generated_tamper_result = jolt_verifier::verify_stage2_with_program(
                input.generated_stage2_verifier_plan,
                &generated_tampered_stage2_proof,
                &generated_stage2_openings,
                Some(&generated_ram_data),
                &mut generated_tamper_transcript,
            );
            assert!(generated_tamper_result.is_err(), "generated {message}");
        };

    assert_batched_sumcheck_tampers!(
        assert_stage2_tamper_rejected,
        input.stage2_artifacts,
        1,
        "Bolt Stage 2 verifier"
    );
}

pub(crate) struct Stage6TamperInput<'a> {
    pub(crate) preamble: &'a CoreMuldivCommitmentFixture,
    pub(crate) commitment_verifier_trace: &'a BoltCommitmentTrace,
    pub(crate) verifier_transcript: &'a BoltTranscript,
    pub(crate) verifier_plan: &'static Stage6VerifierProgramPlan,
    pub(crate) proof: &'a JoltStageProof,
    pub(crate) openings: &'a [JoltStageOpeningInputValue],
    pub(crate) data: &'a JoltStage6VerifierData,
    pub(crate) stage1_artifacts: &'a Stage1ExecutionArtifacts<Fr>,
    pub(crate) stage2_artifacts: &'a Stage2ExecutionArtifacts<Fr>,
    pub(crate) stage3_artifacts: &'a Stage3ExecutionArtifacts<Fr>,
    pub(crate) stage4_artifacts: &'a Stage4ExecutionArtifacts<Fr>,
    pub(crate) stage5_proof: &'a JoltStageProof,
    pub(crate) jolt_inputs: jolt_verifier::JoltVerifierInputs<'a>,
    pub(crate) programs: jolt_verifier::JoltVerifierPrograms,
}

pub(crate) fn assert_bolt_stage6_tamper_rejected(input: Stage6TamperInput<'_>) {
    let assert_tamper_rejected = |tampered_stage6_proof: JoltStageProof, message: &str| {
        assert_generated_stage_and_prefix_tamper_rejected!(
            input.preamble,
            message,
            input.jolt_inputs,
            input.programs,
            {
                let mut transcript = input.verifier_transcript.clone();
                jolt_verifier::verify_stage6_with_program(
                    input.verifier_plan,
                    &tampered_stage6_proof,
                    input.openings,
                    Some(input.data),
                    &mut transcript,
                )
            },
            jolt_prover::jolt_proof_through_stage6(
                &input.commitment_verifier_trace.commitments,
                input.stage1_artifacts,
                input.stage2_artifacts,
                input.stage3_artifacts,
                input.stage4_artifacts,
                input.stage5_proof,
                &tampered_stage6_proof,
            ),
        );
    };

    assert_batched_sumcheck_tampers!(assert_tamper_rejected, input.proof, 0, "Stage 6 verifier");

    let assert_opening_input_tamper_rejected =
        |tampered_openings: &[JoltStageOpeningInputValue], message: &str| {
            let mut generated_tamper_transcript = input.verifier_transcript.clone();
            let generated_tamper_result = jolt_verifier::verify_stage6_with_program(
                input.verifier_plan,
                input.proof,
                tampered_openings,
                Some(input.data),
                &mut generated_tamper_transcript,
            );
            assert!(generated_tamper_result.is_err(), "generated {message}");

            let generated_jolt_proof = jolt_prover::jolt_proof_through_stage6(
                &input.commitment_verifier_trace.commitments,
                input.stage1_artifacts,
                input.stage2_artifacts,
                input.stage3_artifacts,
                input.stage4_artifacts,
                input.stage5_proof,
                input.proof,
            );
            let mut tampered_inputs = input.jolt_inputs;
            tampered_inputs.stage6_openings = tampered_openings;
            assert_generated_jolt_prefix_tamper_rejected(
                input.preamble,
                &generated_jolt_proof,
                tampered_inputs,
                input.programs,
                message,
            );
        };

    assert_opening_input_tampers(
        input.openings,
        "Stage 6",
        assert_opening_input_tamper_rejected,
    );
}

pub(crate) struct Stage7TamperInput<'a> {
    pub(crate) preamble: &'a CoreMuldivCommitmentFixture,
    pub(crate) commitment_verifier_trace: &'a BoltCommitmentTrace,
    pub(crate) verifier_transcript: &'a BoltTranscript,
    pub(crate) verifier_plan: &'static Stage7VerifierProgramPlan,
    pub(crate) proof: &'a JoltStageProof,
    pub(crate) openings: &'a [JoltStageOpeningInputValue],
    pub(crate) stage1_artifacts: &'a Stage1ExecutionArtifacts<Fr>,
    pub(crate) stage2_artifacts: &'a Stage2ExecutionArtifacts<Fr>,
    pub(crate) stage3_artifacts: &'a Stage3ExecutionArtifacts<Fr>,
    pub(crate) stage4_artifacts: &'a Stage4ExecutionArtifacts<Fr>,
    pub(crate) stage5_proof: &'a JoltStageProof,
    pub(crate) stage6_proof: &'a JoltStageProof,
    pub(crate) jolt_inputs: jolt_verifier::JoltVerifierInputs<'a>,
    pub(crate) programs: jolt_verifier::JoltVerifierPrograms,
}

pub(crate) fn assert_bolt_stage7_tamper_rejected(input: Stage7TamperInput<'_>) {
    let assert_tamper_rejected = |tampered_stage7_proof: JoltStageProof, message: &str| {
        assert_generated_stage_and_prefix_tamper_rejected!(
            input.preamble,
            message,
            input.jolt_inputs,
            input.programs,
            {
                let mut transcript = input.verifier_transcript.clone();
                jolt_verifier::verify_stage7_with_program(
                    input.verifier_plan,
                    &tampered_stage7_proof,
                    input.openings,
                    &mut transcript,
                )
            },
            jolt_prover::jolt_proof_through_stage7(
                &input.commitment_verifier_trace.commitments,
                input.stage1_artifacts,
                input.stage2_artifacts,
                input.stage3_artifacts,
                input.stage4_artifacts,
                input.stage5_proof,
                input.stage6_proof,
                &tampered_stage7_proof,
            ),
        );
    };

    assert_batched_sumcheck_tampers!(assert_tamper_rejected, input.proof, 0, "Stage 7 verifier");

    let assert_opening_input_tamper_rejected =
        |tampered_openings: &[JoltStageOpeningInputValue], message: &str| {
            let mut generated_tamper_transcript = input.verifier_transcript.clone();
            let generated_tamper_result = jolt_verifier::verify_stage7_with_program(
                input.verifier_plan,
                input.proof,
                tampered_openings,
                &mut generated_tamper_transcript,
            );
            assert!(generated_tamper_result.is_err(), "generated {message}");

            let generated_jolt_proof = jolt_prover::jolt_proof_through_stage7(
                &input.commitment_verifier_trace.commitments,
                input.stage1_artifacts,
                input.stage2_artifacts,
                input.stage3_artifacts,
                input.stage4_artifacts,
                input.stage5_proof,
                input.stage6_proof,
                input.proof,
            );
            let mut tampered_inputs = input.jolt_inputs;
            tampered_inputs.stage7_openings = tampered_openings;
            assert_generated_jolt_prefix_tamper_rejected(
                input.preamble,
                &generated_jolt_proof,
                tampered_inputs,
                input.programs,
                message,
            );
        };

    assert_opening_input_tampers(
        input.openings,
        "Stage 7",
        assert_opening_input_tamper_rejected,
    );
}

pub(crate) struct Stage345TamperInput<'a> {
    pub(crate) preamble: &'a CoreMuldivCommitmentFixture,
    pub(crate) commitment_verifier_trace: &'a BoltCommitmentTrace,
    pub(crate) stage1_artifacts: &'a Stage1ExecutionArtifacts<Fr>,
    pub(crate) stage2_artifacts: &'a Stage2ExecutionArtifacts<Fr>,
    pub(crate) stage3_artifacts: &'a Stage3ExecutionArtifacts<Fr>,
    pub(crate) stage4_artifacts: &'a Stage4ExecutionArtifacts<Fr>,
    pub(crate) generated_stage3_verifier_plan: &'static Stage3VerifierProgramPlan,
    pub(crate) generated_stage3_openings: &'a [JoltStageOpeningInputValue],
    pub(crate) generated_stage3_start_transcript: &'a BoltTranscript,
    pub(crate) generated_stage4_verifier_plan: &'static Stage4VerifierProgramPlan,
    pub(crate) generated_stage4_start_transcript: &'a BoltTranscript,
    pub(crate) generated_stage5_verifier_plan: &'static Stage5VerifierProgramPlan,
    pub(crate) generated_stage4_openings: &'a [JoltStageOpeningInputValue],
    pub(crate) generated_stage5_openings: &'a [JoltStageOpeningInputValue],
    pub(crate) generated_stage5_proof: &'a JoltStageProof,
    pub(crate) generated_stage5_start_transcript: &'a BoltTranscript,
    pub(crate) generated_jolt_inputs: jolt_verifier::JoltVerifierInputs<'a>,
    pub(crate) generated_programs: jolt_verifier::JoltVerifierPrograms,
}

pub(crate) fn assert_bolt_stage3_4_5_tamper_rejected(input: Stage345TamperInput<'_>) {
    let Stage345TamperInput {
        preamble: fixture,
        commitment_verifier_trace,
        stage1_artifacts,
        stage2_artifacts,
        stage3_artifacts,
        stage4_artifacts,
        generated_stage3_verifier_plan,
        generated_stage3_openings,
        generated_stage3_start_transcript,
        generated_stage4_verifier_plan,
        generated_stage4_start_transcript,
        generated_stage5_verifier_plan,
        generated_stage4_openings,
        generated_stage5_openings,
        generated_stage5_proof,
        generated_stage5_start_transcript,
        generated_jolt_inputs,
        generated_programs,
    } = input;
    let assert_stage3_tamper_rejected =
        |tampered_stage3_artifacts: Stage3ExecutionArtifacts<Fr>, message: &str| {
            assert_generated_stage_and_prefix_tamper_rejected!(
                fixture,
                message,
                generated_jolt_inputs.through_stage5(),
                generated_programs,
                {
                    let mut transcript = generated_stage3_start_transcript.clone();
                    let proof = jolt_prover::stage3_proof(&tampered_stage3_artifacts);
                    jolt_verifier::verify_stage3_with_program(
                        generated_stage3_verifier_plan,
                        &proof,
                        generated_stage3_openings,
                        &mut transcript,
                    )
                },
                jolt_prover::jolt_proof_through_stage5(
                    &commitment_verifier_trace.commitments,
                    stage1_artifacts,
                    stage2_artifacts,
                    &tampered_stage3_artifacts,
                    stage4_artifacts,
                    generated_stage5_proof,
                ),
            );
        };

    assert_batched_sumcheck_tampers!(
        assert_stage3_tamper_rejected,
        stage3_artifacts,
        0,
        "Bolt Stage 3 verifier"
    );

    let assert_stage4_tamper_rejected =
        |tampered_stage4_artifacts: Stage4ExecutionArtifacts<Fr>, message: &str| {
            assert_generated_stage_and_prefix_tamper_rejected!(
                fixture,
                message,
                generated_jolt_inputs.through_stage5(),
                generated_programs,
                {
                    let mut transcript = generated_stage4_start_transcript.clone();
                    let proof = jolt_prover::stage4_proof(&tampered_stage4_artifacts);
                    jolt_verifier::verify_stage4_with_program(
                        generated_stage4_verifier_plan,
                        &proof,
                        generated_stage4_openings,
                        &mut transcript,
                    )
                },
                jolt_prover::jolt_proof_through_stage5(
                    &commitment_verifier_trace.commitments,
                    stage1_artifacts,
                    stage2_artifacts,
                    stage3_artifacts,
                    &tampered_stage4_artifacts,
                    generated_stage5_proof,
                ),
            );
        };

    assert_batched_sumcheck_tampers!(
        assert_stage4_tamper_rejected,
        stage4_artifacts,
        0,
        "Stage 4 verifier"
    );

    let assert_stage5_tamper_rejected = |tampered_stage5_proof: JoltStageProof, message: &str| {
        assert_generated_stage_and_prefix_tamper_rejected!(
            fixture,
            message,
            generated_jolt_inputs.through_stage5(),
            generated_programs,
            {
                let mut transcript = generated_stage5_start_transcript.clone();
                jolt_verifier::verify_stage5_with_program(
                    generated_stage5_verifier_plan,
                    &tampered_stage5_proof,
                    generated_stage5_openings,
                    &mut transcript,
                )
            },
            jolt_prover::jolt_proof_through_stage5(
                &commitment_verifier_trace.commitments,
                stage1_artifacts,
                stage2_artifacts,
                stage3_artifacts,
                stage4_artifacts,
                &tampered_stage5_proof,
            ),
        );
    };

    assert_batched_sumcheck_tampers!(
        assert_stage5_tamper_rejected,
        generated_stage5_proof,
        0,
        "Stage 5 verifier"
    );
}
