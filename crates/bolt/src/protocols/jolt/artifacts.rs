use std::path::Path;

use crate::emit::rust::{
    assemble_generated_crates, assemble_workspace_generated_crates, protocol_rust_artifact,
    validate_rust_artifact_imports, write_generated_crates, EmitError, GeneratedCrate,
    GeneratedFile, ProtocolArtifactConfig, ProtocolArtifactExtension, ProtocolCrateRef,
    ProtocolProverApiExtension, ProtocolRuntimeModule, ProtocolRustArtifact, ProtocolStage,
    ProtocolStageKind, ProtocolStandaloneDependency, ProtocolVerifierApiExtension, RustSourceFile,
    RustTypeRef,
};
use crate::ir::Role;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JoltProtocolStage {
    Commitment,
    Stage1Outer,
    Stage2,
    Stage3,
    Stage4,
    Stage5,
    Stage6,
    Stage7,
    Stage8,
}

impl JoltProtocolStage {
    pub fn name(self) -> &'static str {
        match self {
            Self::Commitment => "commitment",
            Self::Stage1Outer => "stage1_outer",
            Self::Stage2 => "stage2",
            Self::Stage3 => "stage3",
            Self::Stage4 => "stage4",
            Self::Stage5 => "stage5",
            Self::Stage6 => "stage6",
            Self::Stage7 => "stage7",
            Self::Stage8 => "stage8",
        }
    }

    fn expected_filename(self, role: &Role) -> &'static str {
        match (self, role) {
            (Self::Commitment, Role::Prover) => "prove_commitment_phase.rs",
            (Self::Commitment, Role::Verifier) => "verify_commitment_phase.rs",
            (Self::Stage1Outer, Role::Prover) => "prove_stage1_outer.rs",
            (Self::Stage1Outer, Role::Verifier) => "verify_stage1_outer.rs",
            (Self::Stage2, Role::Prover) => "prove_stage2.rs",
            (Self::Stage2, Role::Verifier) => "verify_stage2.rs",
            (Self::Stage3, Role::Prover) => "prove_stage3.rs",
            (Self::Stage3, Role::Verifier) => "verify_stage3.rs",
            (Self::Stage4, Role::Prover) => "prove_stage4.rs",
            (Self::Stage4, Role::Verifier) => "verify_stage4.rs",
            (Self::Stage5, Role::Prover) => "prove_stage5.rs",
            (Self::Stage5, Role::Verifier) => "verify_stage5.rs",
            (Self::Stage6, Role::Prover) => "prove_stage6.rs",
            (Self::Stage6, Role::Verifier) => "verify_stage6.rs",
            (Self::Stage7, Role::Prover) => "prove_stage7.rs",
            (Self::Stage7, Role::Verifier) => "verify_stage7.rs",
            (Self::Stage8, Role::Prover) => "prove_stage8.rs",
            (Self::Stage8, Role::Verifier) => "verify_stage8.rs",
        }
    }
}

impl From<JoltProtocolStage> for ProtocolStage {
    fn from(stage: JoltProtocolStage) -> Self {
        match stage {
            JoltProtocolStage::Commitment => {
                ProtocolStage::new("commitment", "commitment", 0, ProtocolStageKind::Commitment)
            }
            JoltProtocolStage::Stage1Outer => {
                ProtocolStage::new("stage1_outer", "stage1_outer", 1, ProtocolStageKind::Proof)
            }
            JoltProtocolStage::Stage2 => {
                ProtocolStage::new("stage2", "stage2", 2, ProtocolStageKind::Proof)
            }
            JoltProtocolStage::Stage3 => {
                ProtocolStage::new("stage3", "stage3", 3, ProtocolStageKind::Proof)
            }
            JoltProtocolStage::Stage4 => {
                ProtocolStage::new("stage4", "stage4", 4, ProtocolStageKind::Proof)
            }
            JoltProtocolStage::Stage5 => {
                ProtocolStage::new("stage5", "stage5", 5, ProtocolStageKind::Proof)
            }
            JoltProtocolStage::Stage6 => {
                ProtocolStage::new("stage6", "stage6", 6, ProtocolStageKind::Proof)
            }
            JoltProtocolStage::Stage7 => {
                ProtocolStage::new("stage7", "stage7", 7, ProtocolStageKind::Proof)
            }
            JoltProtocolStage::Stage8 => {
                ProtocolStage::new("stage8", "stage8", 8, ProtocolStageKind::Evaluation)
            }
        }
    }
}

impl PartialEq<JoltProtocolStage> for ProtocolStage {
    fn eq(&self, other: &JoltProtocolStage) -> bool {
        self == &ProtocolStage::from(*other)
    }
}

impl PartialEq<ProtocolStage> for JoltProtocolStage {
    fn eq(&self, other: &ProtocolStage) -> bool {
        other == self
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JoltArtifactCrate {
    Prover,
    Verifier,
}

impl JoltArtifactCrate {
    pub fn for_role(role: &Role) -> Self {
        match role {
            Role::Prover => Self::Prover,
            Role::Verifier => Self::Verifier,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Prover => "jolt-prover",
            Self::Verifier => "jolt-verifier",
        }
    }
}

pub type JoltRustArtifact = ProtocolRustArtifact;
pub type JoltGeneratedCrate = GeneratedCrate;
pub type JoltGeneratedFile = GeneratedFile;

pub fn write_jolt_generated_crates(
    generated_crates: &[GeneratedCrate],
    output_root: impl AsRef<Path>,
) -> Result<(), EmitError> {
    write_generated_crates(generated_crates, output_root)
}

pub fn jolt_rust_artifact(
    stage: JoltProtocolStage,
    role: Role,
    source: RustSourceFile,
) -> Result<ProtocolRustArtifact, EmitError> {
    let expected = stage.expected_filename(&role);
    if source.filename != expected {
        return Err(EmitError::new(format!(
            "generated {} artifact for {} expected filename `{expected}`, got `{}`",
            role,
            stage.name(),
            source.filename
        )));
    }

    Ok(protocol_rust_artifact(
        &jolt_artifact_config(),
        ProtocolStage::from(stage),
        role,
        source,
    ))
}

pub fn validate_jolt_rust_artifact_imports(
    artifact: &ProtocolRustArtifact,
) -> Result<(), EmitError> {
    validate_rust_artifact_imports(&jolt_artifact_config(), artifact)
}

pub fn assemble_jolt_generated_crates(
    artifacts: Vec<ProtocolRustArtifact>,
    dependency_root: &str,
) -> Result<Vec<GeneratedCrate>, EmitError> {
    assemble_generated_crates(&jolt_artifact_config(), artifacts, dependency_root)
}

pub fn assemble_jolt_workspace_generated_crates(
    artifacts: Vec<ProtocolRustArtifact>,
) -> Result<Vec<GeneratedCrate>, EmitError> {
    assemble_workspace_generated_crates(&jolt_artifact_config(), artifacts)
}

pub fn jolt_artifact_config() -> ProtocolArtifactConfig {
    ProtocolArtifactConfig {
        protocol_name: "Jolt".to_owned(),
        type_prefix: "Jolt".to_owned(),
        transcript_label: "Jolt".to_owned(),
        repository: Some("https://github.com/a16z/jolt".to_owned()),
        prover_crate_name: "jolt-prover".to_owned(),
        verifier_crate_name: "jolt-verifier".to_owned(),
        crates_io_patches: vec![
            "ark-bn254 = { git = \"https://github.com/a16z/arkworks-algebra\", branch = \"dev/twist-shout\" }".to_owned(),
            "ark-ec = { git = \"https://github.com/a16z/arkworks-algebra\", branch = \"dev/twist-shout\" }".to_owned(),
            "ark-ff = { git = \"https://github.com/a16z/arkworks-algebra\", branch = \"dev/twist-shout\" }".to_owned(),
            "ark-serialize = { git = \"https://github.com/a16z/arkworks-algebra\", branch = \"dev/twist-shout\" }".to_owned(),
        ],
        standalone_dependency_overrides: vec![ProtocolStandaloneDependency::new(
            "rayon",
            "rayon = \"1.12.0\"",
        )],
        common_dependencies: vec![
            "jolt-field".to_owned(),
            "jolt-openings".to_owned(),
            "jolt-poly".to_owned(),
            "jolt-transcript".to_owned(),
        ],
        prover_dependencies: vec![
            "jolt-dory".to_owned(),
            "jolt-kernels".to_owned(),
            "jolt-witness".to_owned(),
            "rayon".to_owned(),
        ],
        verifier_dependencies: vec![
            "jolt-dory".to_owned(),
            "jolt-lookup-tables".to_owned(),
            "jolt-sumcheck".to_owned(),
        ],
        prover_forbidden_imports: PROVER_FORBIDDEN_IMPORTS
            .iter()
            .map(ToString::to_string)
            .collect(),
        verifier_forbidden_imports: VERIFIER_FORBIDDEN_IMPORTS
            .iter()
            .map(ToString::to_string)
            .collect(),
        kernel_crate: Some(ProtocolCrateRef::new("jolt-kernels", "jolt_kernels")),
        field_type: RustTypeRef::new("jolt_field::Fr"),
        default_transcript_type: RustTypeRef::new("jolt_transcript::Blake2bTranscript"),
        transcript_trait: RustTypeRef::new("jolt_transcript::Transcript"),
        commitment_type: RustTypeRef::new("jolt_dory::DoryCommitment"),
        prover_setup_type: RustTypeRef::new("jolt_dory::DoryProverSetup"),
        role_api_extension: Some(jolt_evaluation_role_api_extension()),
        verifier_runtime_modules: vec![ProtocolRuntimeModule {
            module_name: "common".to_owned(),
            file: GeneratedFile {
                path: "src/stages/common.rs".to_owned(),
                source: include_str!("verifier_common.rs.template").to_owned(),
            },
        }],
        verifier_named_eval_type: RustTypeRef::new("crate::stages::common::StageNamedEval"),
        verifier_sumcheck_output_type: RustTypeRef::new(
            "crate::stages::common::StageSumcheckOutput",
        ),
        verifier_stage_proof_type: RustTypeRef::new("crate::stages::common::StageProof"),
    }
}

fn jolt_evaluation_role_api_extension() -> ProtocolArtifactExtension {
    ProtocolArtifactExtension {
        required_commitment: true,
        required_proof_stages: vec!["stage6".to_owned(), "stage7".to_owned()],
        required_artifact_stages: vec!["stage8".to_owned()],
        prover: ProtocolProverApiExtension {
            lib_module: "#[rustfmt::skip]\npub mod prover;\npub mod stages;\n\npub use prover::{\n    default_prover_programs, prove_jolt, prove_jolt_evaluation_proof, prove_jolt_with_programs,\n    DefaultJoltTranscript, JoltEvaluationProveError, JoltProveError, JoltProverArtifacts,\n    JoltProverInputs, JoltProverPrograms,\n};".to_owned(),
            imports: "use jolt_dory::{DoryHint, DoryProverSetup, DoryScheme};\nuse jolt_field::{Field, Fr};\nuse jolt_kernels::{stage1, stage2, stage3, stage4, stage5, stage6, stage7};\nuse jolt_openings::{AdditivelyHomomorphic, CommitmentScheme};\nuse jolt_poly::{EqPolynomial, Polynomial};\nuse jolt_transcript::{AppendToTranscript, Blake2bTranscript, LabelWithCount, Transcript};\nuse jolt_verifier::{JoltEvaluationProof, JoltNamedEval, JoltProof, JoltStageProof, JoltSumcheckOutput};\nuse rayon::prelude::*;\n\n".to_owned(),
            input_fields:
                "    pub stage7_openings: Option<&'a [stage7::Stage7OpeningInputValue<Fr>]>,\n"
                    .to_owned(),
            program_fields:
                "    pub stage8: &'static stage8_stage::Stage8EvaluationProgramPlan,\n".to_owned(),
            default_program_fields: "        stage8: &stage8_stage::STAGE8_PROGRAM,\n".to_owned(),
            error_variants: "    Evaluation(JoltEvaluationProveError),\n".to_owned(),
            error_items: "#[derive(Debug)]\npub enum JoltEvaluationProveError {\n    MissingOracle { oracle: &'static str },\n    MissingOpeningHint { oracle: &'static str },\n    MissingStageEval { stage: &'static str, eval: &'static str },\n    MissingStage7RaEval,\n    MissingStage7EvaluationPoint,\n    InvalidPointLength {\n        artifact: &'static str,\n        expected: usize,\n        actual: usize,\n    },\n    TargetSizeOverflow { num_vars: usize },\n}\n\n".to_owned(),
            error_conversions: "impl From<JoltEvaluationProveError> for JoltProveError {\n    fn from(error: JoltEvaluationProveError) -> Self {\n        Self::Evaluation(error)\n    }\n}\n\n".to_owned(),
            after_stage_execution: "    let evaluation = if let Some(stage7_openings) = inputs.stage7_openings {\n        Some(prove_jolt_evaluation_proof(\n            programs.stage8,\n            inputs.commitment_inputs,\n            inputs.prover_setup,\n            &commitment,\n            &stage6,\n            &stage7,\n            stage7_openings,\n            transcript,\n        )?)\n    } else {\n        None\n    };\n".to_owned(),
            proof_fields: "        evaluation,\n".to_owned(),
            helper_items: jolt_prover_evaluation_helpers("Fr"),
        },
        verifier: ProtocolVerifierApiExtension {
            lib_module: "pub mod stages;\n#[rustfmt::skip]\npub mod verifier;\n\npub use verifier::{\n    default_verifier_programs, verify_jolt, verify_jolt_evaluation_proof, verify_jolt_prefix,\n    verify_jolt_prefix_with_programs, verify_jolt_with_programs, JoltEvaluationProof,\n    JoltEvaluationProofError, JoltNamedEval, JoltProof, JoltStageProof, JoltSumcheckOutput,\n    JoltVerificationArtifacts, JoltVerifierInputs, JoltVerifierPrograms, JoltVerifyError,\n};".to_owned(),
            imports: "use std::collections::BTreeMap;\n\nuse jolt_dory::{DoryCommitment, DoryProof, DoryScheme, DoryVerifierSetup};\nuse jolt_field::{Field, Fr};\nuse jolt_openings::{AdditivelyHomomorphic, CommitmentScheme, OpeningsError};\nuse jolt_poly::EqPolynomial;\nuse jolt_transcript::{AppendToTranscript, LabelWithCount, Transcript};\n".to_owned(),
            proof_fields: "    pub evaluation: Option<JoltEvaluationProof>,\n".to_owned(),
            proof_items: "#[derive(Clone, Debug)]\npub struct JoltEvaluationProof {\n    pub joint_opening_proof: DoryProof,\n}\n\n".to_owned(),
            inputs_derive: Some("#[derive(Clone, Copy)]".to_owned()),
            input_fields: "    pub evaluation_setup: Option<&'a DoryVerifierSetup>,\n".to_owned(),
            program_fields:
                "    pub stage8: &'static stage8_stage::Stage8EvaluationProgramPlan,\n".to_owned(),
            default_program_fields: "        stage8: &stage8_stage::STAGE8_PROGRAM,\n".to_owned(),
            error_variants: "    Evaluation(JoltEvaluationProofError),\n".to_owned(),
            error_items: "#[derive(Debug)]\npub enum JoltEvaluationProofError {\n    MissingProof,\n    MissingVerifierSetup,\n    MissingStageEval { stage: &'static str, eval: &'static str },\n    MissingStage7RaEval,\n    MissingStage7EvaluationPoint,\n    MissingCommitment { oracle: &'static str },\n    InvalidPointLength {\n        artifact: &'static str,\n        expected: usize,\n        actual: usize,\n    },\n    Opening(OpeningsError),\n}\n\n".to_owned(),
            error_conversions: "impl From<JoltEvaluationProofError> for JoltVerifyError {\n    fn from(error: JoltEvaluationProofError) -> Self {\n        Self::Evaluation(error)\n    }\n}\n\nimpl From<OpeningsError> for JoltEvaluationProofError {\n    fn from(error: OpeningsError) -> Self {\n        Self::Opening(error)\n    }\n}\n\n".to_owned(),
            after_default_verify: "pub fn verify_jolt_prefix<T>(\n    proof: &JoltProof,\n    inputs: JoltVerifierInputs<'_>,\n    transcript: &mut T,\n) -> Result<JoltVerificationArtifacts, JoltVerifyError>\nwhere\n    T: Transcript<Challenge = Fr>,\n{\n    verify_jolt_prefix_with_programs(proof, inputs, default_verifier_programs(), transcript)\n}\n\n".to_owned(),
            with_programs_body_intro: "    verify_jolt_with_programs_inner(proof, inputs, programs, transcript, true)\n}\n\npub fn verify_jolt_prefix_with_programs<T>(\n    proof: &JoltProof,\n    inputs: JoltVerifierInputs<'_>,\n    programs: JoltVerifierPrograms,\n    transcript: &mut T,\n) -> Result<JoltVerificationArtifacts, JoltVerifyError>\nwhere\n    T: Transcript<Challenge = Fr>,\n{\n    verify_jolt_with_programs_inner(proof, inputs, programs, transcript, false)\n}\n\nfn verify_jolt_with_programs_inner<T>(\n    proof: &JoltProof,\n    inputs: JoltVerifierInputs<'_>,\n    programs: JoltVerifierPrograms,\n    transcript: &mut T,\n    require_evaluation: bool,\n) -> Result<JoltVerificationArtifacts, JoltVerifyError>\nwhere\n    T: Transcript<Challenge = Fr>,\n{\n".to_owned(),
            after_stage_verification: "    match (&proof.evaluation, inputs.evaluation_setup) {\n        (Some(evaluation), Some(setup)) => {\n            verify_jolt_evaluation_proof(\n                programs.stage8,\n                evaluation,\n                &commitment,\n                &proof.stage6,\n                &proof.stage7,\n                inputs.stage7_openings,\n                setup,\n                transcript,\n            )?;\n        }\n        (Some(_), None) => return Err(JoltEvaluationProofError::MissingVerifierSetup.into()),\n        (None, Some(_)) => return Err(JoltEvaluationProofError::MissingProof.into()),\n        (None, None) if require_evaluation => return Err(JoltEvaluationProofError::MissingProof.into()),\n        (None, None) => {}\n    }\n".to_owned(),
            helper_items: jolt_verifier_evaluation_helpers("Jolt", "Fr"),
        },
    }
}

fn jolt_verifier_evaluation_helpers(prefix: &str, field_type: &str) -> String {
    format!(
        r#"pub fn verify_jolt_evaluation_proof<T>(
    program: &'static stage8_stage::Stage8EvaluationProgramPlan,
    proof: &{prefix}EvaluationProof,
    commitments: &commitment_stage::CommitmentArtifacts,
    stage6: &{prefix}StageProof,
    stage7: &{prefix}StageProof,
    stage7_openings: &[stage7_stage::Stage7OpeningInputValue<{field_type}>],
    verifier_setup: &DoryVerifierSetup,
    transcript: &mut T,
) -> Result<(), {prefix}EvaluationProofError>
where
    T: Transcript<Challenge = {field_type}>,
{{
    let state =
        evaluation_proof_state(program, commitments, stage6, stage7, stage7_openings, transcript)?;
    <DoryScheme as CommitmentScheme>::verify(
        &state.joint_commitment,
        &state.opening_point,
        state.joint_claim,
        &proof.joint_opening_proof,
        verifier_setup,
        transcript,
    )?;
    <DoryScheme as CommitmentScheme>::bind_opening_inputs(
        transcript,
        &state.opening_point,
        &state.joint_claim,
    );
    Ok(())
}}

struct EvaluationProofState {{
    opening_point: Vec<{field_type}>,
    joint_claim: {field_type},
    joint_commitment: DoryCommitment,
}}

struct EvaluationClaim {{
    oracle: &'static str,
    value: {field_type},
}}

fn evaluation_proof_state<T>(
    program: &'static stage8_stage::Stage8EvaluationProgramPlan,
    commitments: &commitment_stage::CommitmentArtifacts,
    stage6: &{prefix}StageProof,
    stage7: &{prefix}StageProof,
    stage7_openings: &[stage7_stage::Stage7OpeningInputValue<{field_type}>],
    transcript: &mut T,
) -> Result<EvaluationProofState, {prefix}EvaluationProofError>
where
    T: Transcript<Challenge = {field_type}>,
{{
    let (sumcheck_address_point, stage7_values) = stage7_claim_values(program, stage7)?;
    let address_point = reverse_point(&sumcheck_address_point);
    let opening_point = stage7_evaluation_opening_point(program, &address_point, stage7_openings)?;
    let lagrange_factor = EqPolynomial::<{field_type}>::zero_selector(&address_point);
    let claims = evaluation_claims(program, stage6, &stage7_values, lagrange_factor)?;

    append_rlc_claims(transcript, &claims);
    let gamma_powers = gamma_powers(transcript, claims.len());
    let joint_claim = claims
        .iter()
        .zip(&gamma_powers)
        .map(|(claim, gamma)| claim.value * *gamma)
        .sum();
    let joint_commitment = joint_commitment(commitments, &claims, &gamma_powers)?;

    Ok(EvaluationProofState {{
        opening_point,
        joint_claim,
        joint_commitment,
    }})
}}

fn stage_eval(
    proof: &{prefix}StageProof,
    stage: &'static str,
    eval_name: &'static str,
) -> Result<{field_type}, {prefix}EvaluationProofError> {{
    for output in &proof.sumchecks {{
        if let Some(eval) = output.evals.iter().find(|eval| eval.name == eval_name) {{
            return Ok(eval.value);
        }}
    }}
    Err({prefix}EvaluationProofError::MissingStageEval {{
        stage,
        eval: eval_name,
    }})
}}

fn evaluation_claims(
    program: &'static stage8_stage::Stage8EvaluationProgramPlan,
    stage6: &{prefix}StageProof,
    stage7_values: &BTreeMap<&'static str, {field_type}>,
    lagrange_factor: {field_type},
) -> Result<Vec<EvaluationClaim>, {prefix}EvaluationProofError> {{
    let mut claims = Vec::with_capacity(program.opening_claims.len());
    for plan in program.opening_claims {{
        let value = match plan.source_stage {{
            "stage6" => stage_eval(stage6, plan.source_stage, plan.source_claim)? * lagrange_factor,
            "stage7" => *stage7_values.get(plan.source_claim).ok_or(
                {prefix}EvaluationProofError::MissingStageEval {{
                    stage: plan.source_stage,
                    eval: plan.source_claim,
                }},
            )?,
            _ => {{
                return Err({prefix}EvaluationProofError::MissingStageEval {{
                    stage: plan.source_stage,
                    eval: plan.source_claim,
                }});
            }}
        }};
        claims.push(EvaluationClaim {{
            oracle: plan.oracle,
            value,
        }});
    }}
    Ok(claims)
}}

fn stage7_claim_values(
    program: &'static stage8_stage::Stage8EvaluationProgramPlan,
    proof: &{prefix}StageProof,
) -> Result<(Vec<{field_type}>, BTreeMap<&'static str, {field_type}>), {prefix}EvaluationProofError> {{
    let stage7_plans = program
        .opening_claims
        .iter()
        .filter(|plan| plan.source_stage == "stage7")
        .collect::<Vec<_>>();
    for output in &proof.sumchecks {{
        let mut values = BTreeMap::new();
        for plan in &stage7_plans {{
            if let Some(eval) = output.evals.iter().find(|eval| eval.name == plan.source_claim) {{
                let _ = values.insert(plan.source_claim, eval.value);
            }}
        }}
        if values.len() == stage7_plans.len() {{
            return Ok((output.point.clone(), values));
        }}
    }}
    Err({prefix}EvaluationProofError::MissingStage7RaEval)
}}

fn reverse_point(point: &[{field_type}]) -> Vec<{field_type}> {{
    point.iter().rev().copied().collect()
}}

fn stage7_evaluation_opening_point(
    program: &'static stage8_stage::Stage8EvaluationProgramPlan,
    address_point: &[{field_type}],
    stage7_openings: &[stage7_stage::Stage7OpeningInputValue<{field_type}>],
) -> Result<Vec<{field_type}>, {prefix}EvaluationProofError> {{
    let cycle_source_symbol = program.evaluation_point_source.source_claim;
    let cycle_source = stage7_openings
        .iter()
        .find(|input| input.symbol == cycle_source_symbol)
        .ok_or({prefix}EvaluationProofError::MissingStage7EvaluationPoint)?;
    if cycle_source.point.len() < address_point.len() {{
        return Err({prefix}EvaluationProofError::InvalidPointLength {{
            artifact: cycle_source_symbol,
            expected: address_point.len(),
            actual: cycle_source.point.len(),
        }});
    }}
    let mut point = Vec::with_capacity(cycle_source.point.len());
    point.extend_from_slice(address_point);
    point.extend_from_slice(&cycle_source.point[address_point.len()..]);
    Ok(point)
}}

fn append_rlc_claims<T>(transcript: &mut T, claims: &[EvaluationClaim])
where
    T: Transcript<Challenge = {field_type}>,
{{
    transcript.append(&LabelWithCount(b"rlc_claims", claims.len() as u64));
    for claim in claims {{
        claim.value.append_to_transcript(transcript);
    }}
}}

fn gamma_powers<T>(transcript: &mut T, count: usize) -> Vec<{field_type}>
where
    T: Transcript<Challenge = {field_type}>,
{{
    let gamma = transcript.challenge();
    let mut powers = Vec::with_capacity(count);
    let mut power = {field_type}::from_u64(1);
    for _ in 0..count {{
        powers.push(power);
        power *= gamma;
    }}
    powers
}}

fn joint_commitment(
    commitments: &commitment_stage::CommitmentArtifacts,
    claims: &[EvaluationClaim],
    gamma_powers: &[{field_type}],
) -> Result<DoryCommitment, {prefix}EvaluationProofError> {{
    let mut coefficients = BTreeMap::<&'static str, {field_type}>::new();
    for (claim, gamma) in claims.iter().zip(gamma_powers) {{
        let coefficient = coefficients.entry(claim.oracle).or_insert({field_type}::from_u64(0));
        *coefficient += *gamma;
    }}
    let mut commitment_values = Vec::with_capacity(coefficients.len());
    let mut scalars = Vec::with_capacity(coefficients.len());
    for (oracle, coefficient) in coefficients {{
        commitment_values.push(commitment_for_oracle(commitments, oracle)?);
        scalars.push(coefficient);
    }}
    Ok(<DoryScheme as AdditivelyHomomorphic>::combine(
        &commitment_values,
        &scalars,
    ))
}}

fn commitment_for_oracle(
    commitments: &commitment_stage::CommitmentArtifacts,
    oracle: &'static str,
) -> Result<DoryCommitment, {prefix}EvaluationProofError> {{
    for (record, commitment) in commitments.records.iter().zip(&commitments.commitments) {{
        if record.oracle == oracle {{
            return commitment
                .clone()
                .ok_or({prefix}EvaluationProofError::MissingCommitment {{ oracle }});
        }}
    }}
    Err({prefix}EvaluationProofError::MissingCommitment {{ oracle }})
}}

"#
    )
}

fn jolt_prover_evaluation_helpers(field_type: &str) -> String {
    format!(
        r#"pub fn prove_jolt_evaluation_proof<I, T>(
    program: &'static stage8_stage::Stage8EvaluationProgramPlan,
    commitment_inputs: &mut I,
    prover_setup: &DoryProverSetup,
    commitments: &commitment_stage::CommitmentArtifacts,
    stage6: &stage6::Stage6ExecutionArtifacts<{field_type}>,
    stage7: &stage7::Stage7ExecutionArtifacts<{field_type}>,
    stage7_openings: &[stage7::Stage7OpeningInputValue<{field_type}>],
    transcript: &mut T,
) -> Result<JoltEvaluationProof, JoltEvaluationProveError>
where
    I: commitment_stage::CommitmentInputProvider,
    T: Transcript<Challenge = {field_type}>,
{{
    let (sumcheck_address_point, stage7_values) = stage7_claim_values(program, stage7)?;
    let address_point = reverse_point(&sumcheck_address_point);
    let (opening_point, log_t) =
        stage7_evaluation_opening_point(program, &address_point, stage7_openings)?;
    let lagrange_factor = EqPolynomial::<{field_type}>::zero_selector(&address_point);
    let claims = evaluation_claims(program, stage6, &stage7_values, lagrange_factor)?;

    append_rlc_claims(transcript, &claims);
    let gamma_powers = gamma_powers(transcript, claims.len());
    let joint_claim = claims
        .iter()
        .zip(&gamma_powers)
        .map(|(claim, gamma)| claim.value * *gamma)
        .sum();
    let joint_evals = materialize_joint_polynomial(
        commitment_inputs,
        &claims,
        &gamma_powers,
        log_t,
        opening_point.len(),
    )?;
    let joint_poly = Polynomial::new(joint_evals);
    let joint_hint = joint_opening_hint(commitments, &claims, &gamma_powers)?;
    let joint_opening_proof = <jolt_dory::DoryScheme as CommitmentScheme>::open(
        &joint_poly,
        &opening_point,
        joint_claim,
        prover_setup,
        Some(joint_hint),
        transcript,
    );
    <jolt_dory::DoryScheme as CommitmentScheme>::bind_opening_inputs(
        transcript,
        &opening_point,
        &joint_claim,
    );
    Ok(JoltEvaluationProof {{ joint_opening_proof }})
}}

struct EvaluationClaim {{
    oracle: &'static str,
    source_stage: &'static str,
    value: {field_type},
}}

fn stage6_eval_claim(
    artifacts: &stage6::Stage6ExecutionArtifacts<{field_type}>,
    eval_name: &'static str,
) -> Result<{field_type}, JoltEvaluationProveError> {{
    for output in &artifacts.sumchecks {{
        if let Some(eval) = output.evals.iter().find(|eval| eval.name == eval_name) {{
            return Ok(eval.value);
        }}
    }}
    Err(JoltEvaluationProveError::MissingStageEval {{
        stage: "stage6",
        eval: eval_name,
    }})
}}

fn evaluation_claims(
    program: &'static stage8_stage::Stage8EvaluationProgramPlan,
    stage6: &stage6::Stage6ExecutionArtifacts<{field_type}>,
    stage7_values: &std::collections::BTreeMap<&'static str, {field_type}>,
    lagrange_factor: {field_type},
) -> Result<Vec<EvaluationClaim>, JoltEvaluationProveError> {{
    let mut claims = Vec::with_capacity(program.opening_claims.len());
    for plan in program.opening_claims {{
        let value = match plan.source_stage {{
            "stage6" => stage6_eval_claim(stage6, plan.source_claim)? * lagrange_factor,
            "stage7" => *stage7_values.get(plan.source_claim).ok_or(
                JoltEvaluationProveError::MissingStageEval {{
                    stage: plan.source_stage,
                    eval: plan.source_claim,
                }},
            )?,
            _ => {{
                return Err(JoltEvaluationProveError::MissingStageEval {{
                    stage: plan.source_stage,
                    eval: plan.source_claim,
                }});
            }}
        }};
        claims.push(EvaluationClaim {{
            oracle: plan.oracle,
            source_stage: plan.source_stage,
            value,
        }});
    }}
    Ok(claims)
}}

fn stage7_claim_values(
    program: &'static stage8_stage::Stage8EvaluationProgramPlan,
    artifacts: &stage7::Stage7ExecutionArtifacts<{field_type}>,
) -> Result<(Vec<{field_type}>, std::collections::BTreeMap<&'static str, {field_type}>), JoltEvaluationProveError> {{
    let stage7_plans = program
        .opening_claims
        .iter()
        .filter(|plan| plan.source_stage == "stage7")
        .collect::<Vec<_>>();
    for output in &artifacts.sumchecks {{
        let mut values = std::collections::BTreeMap::new();
        for plan in &stage7_plans {{
            if let Some(eval) = output.evals.iter().find(|eval| eval.name == plan.source_claim) {{
                let _ = values.insert(plan.source_claim, eval.value);
            }}
        }}
        if values.len() == stage7_plans.len() {{
            return Ok((output.point.clone(), values));
        }}
    }}
    Err(JoltEvaluationProveError::MissingStage7RaEval)
}}

fn reverse_point(point: &[{field_type}]) -> Vec<{field_type}> {{
    point.iter().rev().copied().collect()
}}

fn stage7_evaluation_opening_point(
    program: &'static stage8_stage::Stage8EvaluationProgramPlan,
    address_point: &[{field_type}],
    stage7_openings: &[stage7::Stage7OpeningInputValue<{field_type}>],
) -> Result<(Vec<{field_type}>, usize), JoltEvaluationProveError> {{
    let cycle_source_symbol = program.evaluation_point_source.source_claim;
    let cycle_source = stage7_openings
        .iter()
        .find(|input| input.symbol == cycle_source_symbol)
        .ok_or(JoltEvaluationProveError::MissingStage7EvaluationPoint)?;
    if cycle_source.point.len() < address_point.len() {{
        return Err(JoltEvaluationProveError::InvalidPointLength {{
            artifact: cycle_source_symbol,
            expected: address_point.len(),
            actual: cycle_source.point.len(),
        }});
    }}
    let cycle_len = cycle_source.point.len() - address_point.len();
    let mut point = Vec::with_capacity(cycle_source.point.len());
    point.extend_from_slice(address_point);
    point.extend_from_slice(&cycle_source.point[address_point.len()..]);
    Ok((point, cycle_len))
}}

fn append_rlc_claims<T>(transcript: &mut T, claims: &[EvaluationClaim])
where
    T: Transcript<Challenge = {field_type}>,
{{
    transcript.append(&LabelWithCount(b"rlc_claims", claims.len() as u64));
    for claim in claims {{
        claim.value.append_to_transcript(transcript);
    }}
}}

fn gamma_powers<T>(transcript: &mut T, count: usize) -> Vec<{field_type}>
where
    T: Transcript<Challenge = {field_type}>,
{{
    let gamma = transcript.challenge();
    let mut powers = Vec::with_capacity(count);
    let mut power = {field_type}::from_u64(1);
    for _ in 0..count {{
        powers.push(power);
        power *= gamma;
    }}
    powers
}}

fn materialize_joint_polynomial<I>(
    commitment_inputs: &mut I,
    claims: &[EvaluationClaim],
    gamma_powers: &[{field_type}],
    log_t: usize,
    main_num_vars: usize,
) -> Result<Vec<{field_type}>, JoltEvaluationProveError>
where
    I: commitment_stage::CommitmentInputProvider,
{{
    let trace_len = target_len(log_t)?;
    let main_len = target_len(main_num_vars)?;
    let mut joint = vec![{field_type}::from_u64(0); main_len];
    for (claim, gamma) in claims.iter().zip(gamma_powers) {{
        if claim.source_stage == "stage6" {{
            add_oracle_scaled(commitment_inputs, &mut joint, claim.oracle, log_t, trace_len, *gamma)?;
        }} else {{
            add_oracle_scaled(
                commitment_inputs,
                &mut joint,
                claim.oracle,
                main_num_vars,
                main_len,
                *gamma,
            )?;
        }}
    }}
    Ok(joint)
}}

fn add_oracle_scaled<I>(
    commitment_inputs: &mut I,
    joint: &mut [{field_type}],
    oracle: &'static str,
    num_vars: usize,
    limit: usize,
    scalar: {field_type},
) -> Result<(), JoltEvaluationProveError>
where
    I: commitment_stage::CommitmentInputProvider,
{{
    if commitment_inputs.add_scaled_to_joint(oracle, joint, num_vars, limit, scalar) {{
        return Ok(());
    }}
    let target_len = target_len(num_vars)?;
    let data = commitment_inputs
        .materialize_with_num_vars(oracle, num_vars)
        .ok_or(JoltEvaluationProveError::MissingOracle {{ oracle }})?;
    if data.len() > target_len {{
        return Err(JoltEvaluationProveError::InvalidPointLength {{
            artifact: oracle,
            expected: target_len,
            actual: data.len(),
        }});
    }}
    let zero = {field_type}::from_u64(0);
    let one = {field_type}::from_u64(1);
    let len = limit.min(joint.len()).min(data.len());
    if len >= 1 << 15 {{
        joint[..len]
            .par_iter_mut()
            .zip(data[..len].par_iter())
            .for_each(|(dst, value)| {{
                if *value == zero {{
                    return;
                }}
                if *value == one {{
                    *dst += scalar;
                }} else {{
                    *dst += *value * scalar;
                }}
            }});
    }} else {{
        for (dst, value) in joint.iter_mut().take(len).zip(data.iter()) {{
            if *value == zero {{
                continue;
            }}
            if *value == one {{
                *dst += scalar;
            }} else {{
                *dst += *value * scalar;
            }}
        }}
    }}
    Ok(())
}}

fn joint_opening_hint(
    commitments: &commitment_stage::CommitmentArtifacts,
    claims: &[EvaluationClaim],
    gamma_powers: &[{field_type}],
) -> Result<DoryHint, JoltEvaluationProveError> {{
    let mut coefficients = std::collections::BTreeMap::<&'static str, {field_type}>::new();
    for (claim, gamma) in claims.iter().zip(gamma_powers) {{
        let coefficient = coefficients.entry(claim.oracle).or_insert({field_type}::from_u64(0));
        *coefficient += *gamma;
    }}

    let mut hints = Vec::with_capacity(coefficients.len());
    let mut scalars = Vec::with_capacity(coefficients.len());
    for (oracle, coefficient) in coefficients {{
        hints.push(opening_hint_for_oracle(commitments, oracle)?);
        scalars.push(coefficient);
    }}

    Ok(<DoryScheme as AdditivelyHomomorphic>::combine_hints(
        hints, &scalars,
    ))
}}

fn opening_hint_for_oracle(
    commitments: &commitment_stage::CommitmentArtifacts,
    oracle: &'static str,
) -> Result<DoryHint, JoltEvaluationProveError> {{
    commitments
        .hints
        .iter()
        .find(|hint| hint.oracle == oracle)
        .map(|hint| hint.hint.clone())
        .ok_or(JoltEvaluationProveError::MissingOpeningHint {{ oracle }})
}}

fn target_len(num_vars: usize) -> Result<usize, JoltEvaluationProveError> {{
    if num_vars >= usize::BITS as usize {{
        return Err(JoltEvaluationProveError::TargetSizeOverflow {{ num_vars }});
    }}
    Ok(1usize << num_vars)
}}

"#
    )
}

const PROVER_FORBIDDEN_IMPORTS: &[&str] = &[
    "use jolt_core",
    "jolt_core::",
    "use jolt_verifier::stages",
    "jolt_verifier::stages::",
    "use jolt_equivalence",
    "jolt_equivalence::",
    "use jolt_bench",
    "jolt_bench::",
];

const VERIFIER_FORBIDDEN_IMPORTS: &[&str] = &[
    "use jolt_kernels",
    "jolt_kernels::",
    "use jolt_prover",
    "jolt_prover::",
    "use jolt_core",
    "jolt_core::",
    "use jolt_equivalence",
    "jolt_equivalence::",
    "use jolt_bench",
    "jolt_bench::",
    "tracer::",
];
