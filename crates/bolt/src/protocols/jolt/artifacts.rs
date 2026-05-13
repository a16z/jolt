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
        ), ProtocolStandaloneDependency::new(
            "serde",
            "serde = { version = \"1.0\", default-features = false, features = [\"derive\"] }",
        ), ProtocolStandaloneDependency::new(
            "tracing",
            "tracing = { version = \"0.1.37\", default-features = false, features = [\"attributes\"] }",
        )],
        common_dependencies: vec![
            "jolt-field".to_owned(),
            "jolt-openings".to_owned(),
            "jolt-poly".to_owned(),
            "jolt-transcript".to_owned(),
            "tracing".to_owned(),
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
            "serde".to_owned(),
        ],
        instrumentation_prefix: Some("bolt".to_owned()),
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

fn jolt_prover_lib_module() -> String {
    r"#[rustfmt::skip]
pub mod prover;
pub mod stages;

pub use prover::{
    default_prover_programs, jolt_proof_through_stage5, jolt_proof_through_stage6,
    jolt_proof_through_stage7, prove_jolt, prove_jolt_evaluation_proof, prove_jolt_with_programs,
    prove_jolt_with_stage_inputs, prove_jolt_with_witness_inputs,
    prove_stage1_outer_inputs_with_program, prove_stage2_inputs_with_program,
    prove_stage3_inputs_with_program, prove_stage4_inputs_with_program,
    prove_stage5_inputs_with_program, prove_stage6_inputs_with_program,
    prove_stage7_inputs_with_program, replay_stage1_outer_proof_with_program,
    replay_stage2_proof_with_program, replay_stage3_proof_with_program,
    replay_stage4_proof_with_program, replay_stage5_proof_with_program,
    replay_stage6_proof_with_program, replay_stage7_proof_with_program, stage1_outer_proof,
    stage1_outer_proof_from_kernel_proof, stage1_outer_prover_inputs,
    stage2_opening_inputs_from_artifacts, stage2_proof, stage2_prover_inputs,
    stage2_verifier_ram_data, stage3_opening_inputs_from_artifacts, stage3_proof,
    stage3_prover_inputs, stage4_opening_inputs_from_artifacts, stage4_proof, stage4_prover_inputs,
    stage5_kernel_proof, stage5_opening_inputs_from_artifacts, stage5_proof, stage5_prover_inputs,
    stage6_bytecode_read_raf_data_from_witness_entries, stage6_execution_artifacts,
    stage6_kernel_proof, stage6_opening_inputs_from_artifacts, stage6_proof, stage6_prover_inputs,
    stage6_witness_from_opening_inputs, stage7_execution_artifacts, stage7_kernel_proof,
    stage7_opening_inputs_from_stage6_artifacts,
    stage7_opening_inputs_from_stage6_artifacts_with_program, stage7_proof, stage7_prover_inputs,
    verifier_opening_inputs_from_kernel, DefaultJoltTranscript, JoltEvaluationProveError,
    JoltKernelOpeningInput, JoltOpeningInputError, JoltProveError, JoltProverArtifacts,
    JoltProverInputs, JoltProverPrograms, JoltProverStageInputs, JoltProverWitnessInputs,
    JoltStage2RamDataStorage,
};

pub use prover::{
    prove_stage1_outer_with_witness_inputs, prove_stage2_with_witness_inputs,
    prove_stage3_with_witness_inputs, prove_stage4_with_trace_witness_inputs,
    prove_stage4_with_witness_inputs, prove_stage5_with_trace_witness_inputs,
    prove_stage5_with_witness_inputs, prove_stage6_with_trace_witness_inputs,
    prove_stage6_with_witness_inputs, prove_stage7_with_trace_witness_inputs,
    prove_stage7_with_witness_inputs, stage6_verifier_data_from_witness_entries,
};"
    .to_owned()
}

fn jolt_verifier_lib_module() -> String {
    r"pub mod stages;
#[rustfmt::skip]
pub mod verifier;

pub use stages::{
    stage1_outer::{verify_stage1_outer_with_program, Stage1VerifierProgramPlan},
    stage2::{verify_stage2_with_program, Stage2VerifierProgramPlan},
    stage3::{verify_stage3_with_program, Stage3VerifierProgramPlan},
    stage4::{verify_stage4_with_program, Stage4VerifierProgramPlan},
    stage5::{verify_stage5_with_program, Stage5VerifierProgramPlan},
    stage6::{verify_stage6_with_program, Stage6VerifierProgramPlan},
    stage7::{verify_stage7_with_program, Stage7VerifierProgramPlan},
};

pub use verifier::{
    default_verifier_programs, verify_jolt, verify_jolt_evaluation_proof, verify_jolt_prefix,
    verify_jolt_prefix_with_programs, verify_jolt_through_stage5,
    verify_jolt_through_stage5_with_programs, verify_jolt_through_stage6,
    verify_jolt_through_stage6_with_programs, verify_jolt_through_stage7,
    verify_jolt_through_stage7_with_programs, verify_jolt_with_programs, JoltEvaluationProof,
    JoltEvaluationProofError, JoltNamedEval, JoltProof, JoltStage2RamAccess, JoltStage2RamData,
    JoltStage2RamOutputLayout, JoltStage6BytecodeEntry, JoltStage6BytecodeReadRafData,
    JoltStage6VerifierData, JoltStageChallengeVector, JoltStageExecutionArtifacts,
    JoltStageOpeningInputValue, JoltStageProof, JoltSumcheckOutput, JoltVerificationArtifacts,
    JoltVerifierInputs, JoltVerifierPrograms, JoltVerifierTarget, JoltVerifyError,
};"
    .to_owned()
}

fn jolt_evaluation_role_api_extension() -> ProtocolArtifactExtension {
    ProtocolArtifactExtension {
        required_commitment: true,
        required_proof_stages: vec!["stage6".to_owned(), "stage7".to_owned()],
        required_artifact_stages: vec!["stage8".to_owned()],
        prover: ProtocolProverApiExtension {
            lib_module: jolt_prover_lib_module(),
            imports: "#![expect(\n    clippy::too_many_arguments,\n    reason = \"generated prover helpers mirror staged protocol ABIs\"\n)]\n\nuse jolt_dory::{DoryCommitment, DoryHint, DoryProverSetup, DoryScheme};\nuse jolt_field::Fr;\nuse jolt_kernels::{stage1, stage2, stage3, stage4, stage5, stage6, stage7};\nuse jolt_openings::{AdditivelyHomomorphic, CommitmentScheme};\nuse jolt_poly::{EqPolynomial, Polynomial};\nuse jolt_transcript::{Blake2bTranscript, Transcript};\nuse jolt_verifier::{JoltEvaluationProof, JoltNamedEval, JoltProof, JoltStage2RamAccess, JoltStage2RamData, JoltStage2RamOutputLayout, JoltStage6BytecodeEntry, JoltStage6BytecodeReadRafData, JoltStage6VerifierData, JoltStageChallengeVector, JoltStageExecutionArtifacts, JoltStageOpeningInputValue, JoltStageProof, JoltSumcheckOutput};\nuse jolt_witness::{stage4_ram_val_init_opening, CycleInput, Stage45SparseTraceWitness, Stage6BytecodeEntry as WitnessStage6BytecodeEntry, Stage6WitnessParams, Stage6WitnessPolynomials, Stage6WitnessSlices};\nuse rayon::prelude::*;\n\n".to_owned(),
            input_fields:
                "    pub stage7_openings: Option<&'a [stage7::Stage7OpeningInputValue<Fr>]>,\n"
                    .to_owned(),
            program_fields:
                "    pub stage8: &'static stage8_stage::Stage8EvaluationProgramPlan,\n".to_owned(),
            default_program_fields: "        stage8: &stage8_stage::STAGE8_PROGRAM,\n".to_owned(),
            error_variants: "    Evaluation(JoltEvaluationProveError),\n".to_owned(),
            error_items: "#[derive(Debug)]\npub enum JoltEvaluationProveError {\n    MissingOracle { oracle: &'static str },\n    MissingOpeningHint { oracle: &'static str },\n    MissingStageEval { stage: &'static str, eval: &'static str },\n    MissingStage7RaEval,\n    MissingStage7EvaluationPoint,\n    InvalidPointLength {\n        artifact: &'static str,\n        expected: usize,\n        actual: usize,\n    },\n    TargetSizeOverflow { num_vars: usize },\n}\n\n#[derive(Debug)]\npub enum JoltOpeningInputError {\n    MissingOpeningClaim { stage: &'static str, source_claim: &'static str },\n    MissingStage6OpeningClaim { source_claim: &'static str },\n    UnsupportedOpeningInputSource { stage: &'static str, symbol: &'static str, source_stage: &'static str },\n    UnsupportedStage7InputSource { symbol: &'static str, source_stage: &'static str },\n    InvalidPointLength {\n        symbol: &'static str,\n        expected: usize,\n        actual: usize,\n    },\n}\n\n".to_owned(),
            error_conversions: "impl From<JoltEvaluationProveError> for JoltProveError {\n    fn from(error: JoltEvaluationProveError) -> Self {\n        Self::Evaluation(error)\n    }\n}\n\n".to_owned(),
            after_stage_execution: "    let evaluation = if let Some(stage7_openings) = inputs.stage7_openings {\n        let _stage8_span = tracing::info_span!(\"bolt.stage8\").entered();\n        let _evaluate_span = tracing::info_span!(\"bolt.evaluate\").entered();\n        Some(prove_jolt_evaluation_proof(\n            programs.stage8,\n            inputs.commitment_inputs,\n            inputs.prover_setup,\n            &commitment,\n            &stage6,\n            &stage7,\n            stage7_openings,\n            transcript,\n        )?)\n    } else {\n        None\n    };\n".to_owned(),
            proof_fields: "        evaluation,\n".to_owned(),
            helper_items: format!(
                "{}{}",
                jolt_prover_evaluation_helpers("Fr"),
                jolt_prover_stage7_opening_input_helpers("Fr")
            ),
        },
        verifier: ProtocolVerifierApiExtension {
            lib_module: jolt_verifier_lib_module(),
            imports: "use std::collections::BTreeMap;\n\nuse jolt_dory::{DoryCommitment, DoryProof, DoryScheme, DoryVerifierSetup};\nuse jolt_field::Fr;\nuse jolt_openings::{AdditivelyHomomorphic, CommitmentScheme, OpeningsError};\nuse jolt_poly::EqPolynomial;\nuse jolt_transcript::Transcript;\n".to_owned(),
            proof_fields: "    pub evaluation: Option<JoltEvaluationProof>,\n".to_owned(),
            proof_items: "pub type JoltStage2RamAccess = crate::stages::stage2::Stage2RamAccess;\npub type JoltStage2RamOutputLayout = crate::stages::stage2::Stage2RamOutputLayout;\npub type JoltStage2RamData<'a> = crate::stages::stage2::Stage2RamData<'a>;\npub type JoltStageChallengeVector = crate::stages::common::StageChallengeVector<Fr>;\npub type JoltStageExecutionArtifacts = crate::stages::common::StageExecutionArtifacts<Fr>;\npub type JoltStageOpeningInputValue = crate::stages::common::StageOpeningInputValue<Fr>;\n\n#[derive(Clone, Debug)]\npub struct JoltEvaluationProof {\n    pub joint_opening_proof: DoryProof,\n}\n\n".to_owned(),
            inputs_derive: Some("#[derive(Clone, Copy)]".to_owned()),
            input_fields: "    pub evaluation_setup: Option<&'a DoryVerifierSetup>,\n".to_owned(),
            program_fields:
                "    pub stage8: &'static stage8_stage::Stage8EvaluationProgramPlan,\n".to_owned(),
            default_program_fields: "        stage8: &stage8_stage::STAGE8_PROGRAM,\n".to_owned(),
            error_variants: "    Evaluation(JoltEvaluationProofError),\n".to_owned(),
            error_items: format!("{}{}", jolt_verifier_target_items(), "#[derive(Debug)]\npub enum JoltEvaluationProofError {\n    MissingProof,\n    MissingVerifierSetup,\n    MissingStageEval { stage: &'static str, eval: &'static str },\n    MissingStage7RaEval,\n    MissingStage7EvaluationPoint,\n    MissingCommitment { oracle: &'static str },\n    InvalidPointLength {\n        artifact: &'static str,\n        expected: usize,\n        actual: usize,\n    },\n    Opening(OpeningsError),\n}\n\n"),
            error_conversions: "impl From<JoltEvaluationProofError> for JoltVerifyError {\n    fn from(error: JoltEvaluationProofError) -> Self {\n        Self::Evaluation(error)\n    }\n}\n\nimpl From<OpeningsError> for JoltEvaluationProofError {\n    fn from(error: OpeningsError) -> Self {\n        Self::Opening(error)\n    }\n}\n\n".to_owned(),
            after_default_verify: "pub fn verify_jolt_prefix<T: Transcript<Challenge = Fr>>(proof: &JoltProof, inputs: JoltVerifierInputs<'_>, transcript: &mut T) -> Result<JoltVerificationArtifacts, JoltVerifyError> { verify_jolt_prefix_with_programs(proof, inputs, default_verifier_programs(), transcript) }\n\npub fn verify_jolt_through_stage5<T: Transcript<Challenge = Fr>>(proof: &JoltProof, inputs: JoltVerifierInputs<'_>, transcript: &mut T) -> Result<JoltVerificationArtifacts, JoltVerifyError> { verify_jolt_through_stage5_with_programs(proof, inputs, default_verifier_programs(), transcript) }\n\npub fn verify_jolt_through_stage6<T: Transcript<Challenge = Fr>>(proof: &JoltProof, inputs: JoltVerifierInputs<'_>, transcript: &mut T) -> Result<JoltVerificationArtifacts, JoltVerifyError> { verify_jolt_through_stage6_with_programs(proof, inputs, default_verifier_programs(), transcript) }\n\npub fn verify_jolt_through_stage7<T: Transcript<Challenge = Fr>>(proof: &JoltProof, inputs: JoltVerifierInputs<'_>, transcript: &mut T) -> Result<JoltVerificationArtifacts, JoltVerifyError> { verify_jolt_through_stage7_with_programs(proof, inputs, default_verifier_programs(), transcript) }\n\n".to_owned(),
            with_programs_body_intro: "    verify_jolt_with_programs_inner(proof, inputs, programs, transcript, JoltVerifierTarget::Full)\n}\n\npub fn verify_jolt_through_stage5_with_programs<T: Transcript<Challenge = Fr>>(proof: &JoltProof, inputs: JoltVerifierInputs<'_>, programs: JoltVerifierPrograms, transcript: &mut T) -> Result<JoltVerificationArtifacts, JoltVerifyError> { verify_jolt_with_programs_inner(proof, inputs, programs, transcript, JoltVerifierTarget::ThroughStage5) }\n\npub fn verify_jolt_through_stage6_with_programs<T: Transcript<Challenge = Fr>>(proof: &JoltProof, inputs: JoltVerifierInputs<'_>, programs: JoltVerifierPrograms, transcript: &mut T) -> Result<JoltVerificationArtifacts, JoltVerifyError> { verify_jolt_with_programs_inner(proof, inputs, programs, transcript, JoltVerifierTarget::ThroughStage6) }\n\npub fn verify_jolt_through_stage7_with_programs<T: Transcript<Challenge = Fr>>(proof: &JoltProof, inputs: JoltVerifierInputs<'_>, programs: JoltVerifierPrograms, transcript: &mut T) -> Result<JoltVerificationArtifacts, JoltVerifyError> { verify_jolt_with_programs_inner(proof, inputs, programs, transcript, JoltVerifierTarget::ThroughStage7) }\n\npub fn verify_jolt_prefix_with_programs<T: Transcript<Challenge = Fr>>(proof: &JoltProof, inputs: JoltVerifierInputs<'_>, programs: JoltVerifierPrograms, transcript: &mut T) -> Result<JoltVerificationArtifacts, JoltVerifyError> { verify_jolt_through_stage7_with_programs(proof, inputs, programs, transcript) }\n\nfn verify_jolt_with_programs_inner<T: Transcript<Challenge = Fr>>(proof: &JoltProof, inputs: JoltVerifierInputs<'_>, programs: JoltVerifierPrograms, transcript: &mut T, target: JoltVerifierTarget) -> Result<JoltVerificationArtifacts, JoltVerifyError> {\n".to_owned(),
            stage_verification_override: jolt_verifier_stage_verification(),
            after_stage_verification: jolt_verifier_evaluation_check(),
            helper_items: format!(
                "{}{}",
                jolt_verifier_input_helpers("Jolt"),
                jolt_verifier_evaluation_helpers("Jolt", "Fr")
            ),
        },
    }
}

fn jolt_verifier_target_items() -> String {
    "#[derive(Clone, Copy, Debug, PartialEq, Eq)]\npub enum JoltVerifierTarget {\n    ThroughStage5,\n    ThroughStage6,\n    ThroughStage7,\n    Full,\n}\n\nimpl JoltVerifierTarget {\n    fn verifies_stage6(self) -> bool { matches!(self, Self::ThroughStage6 | Self::ThroughStage7 | Self::Full) }\n    fn verifies_stage7(self) -> bool { matches!(self, Self::ThroughStage7 | Self::Full) }\n    fn verifies_evaluation(self) -> bool { matches!(self, Self::Full) }\n    fn allows_optional_evaluation(self) -> bool { matches!(self, Self::ThroughStage7 | Self::Full) }\n}\n\n".to_owned()
}

fn jolt_verifier_stage_verification() -> String {
    "    let stage1_outer = stage1_outer_stage::verify_stage1_outer_with_program(programs.stage1_outer, &proof.stage1_outer, transcript)?;\n    let stage2 = stage2_stage::verify_stage2_with_program(programs.stage2, &proof.stage2, inputs.stage2_openings, inputs.stage2_ram, transcript)?;\n    let stage3 = stage3_stage::verify_stage3_with_program(programs.stage3, &proof.stage3, inputs.stage3_openings, transcript)?;\n    let stage4 = stage4_stage::verify_stage4_with_program(programs.stage4, &proof.stage4, inputs.stage4_openings, transcript)?;\n    let stage5 = stage5_stage::verify_stage5_with_program(programs.stage5, &proof.stage5, inputs.stage5_openings, transcript)?;\n    let stage6 = if target.verifies_stage6() {\n        stage6_stage::verify_stage6_with_program(programs.stage6, &proof.stage6, inputs.stage6_openings, inputs.stage6_data, transcript)?\n    } else {\n        stage6_stage::Stage6ExecutionArtifacts::default()\n    };\n    let stage7 = if target.verifies_stage7() {\n        stage7_stage::verify_stage7_with_program(programs.stage7, &proof.stage7, inputs.stage7_openings, transcript)?\n    } else {\n        stage7_stage::Stage7ExecutionArtifacts::default()\n    };\n".to_owned()
}

fn jolt_verifier_evaluation_check() -> String {
    "    if target.allows_optional_evaluation() {\n        match (&proof.evaluation, inputs.evaluation_setup) {\n            (Some(evaluation), Some(setup)) => {\n                verify_jolt_evaluation_proof(\n                    programs.stage8,\n                    evaluation,\n                    &commitment,\n                    &proof.stage6,\n                    &proof.stage7,\n                    inputs.stage7_openings,\n                    setup,\n                    transcript,\n                )?;\n            }\n            (Some(_), None) => return Err(JoltEvaluationProofError::MissingVerifierSetup.into()),\n            (None, Some(_)) => return Err(JoltEvaluationProofError::MissingProof.into()),\n            (None, None) if target.verifies_evaluation() => return Err(JoltEvaluationProofError::MissingProof.into()),\n            (None, None) => {}\n        }\n    }\n".to_owned()
}

fn jolt_verifier_input_helpers(prefix: &str) -> String {
    format!(
        "impl<'a> {prefix}VerifierInputs<'a> {{\n    pub fn through_stage5(mut self) -> Self {{ self.stage6_openings = &[]; self.stage7_openings = &[]; self.evaluation_setup = None; self }}\n    pub fn through_stage6(mut self) -> Self {{ self.stage7_openings = &[]; self.evaluation_setup = None; self }}\n    pub fn through_stage7(mut self) -> Self {{ self.evaluation_setup = None; self }}\n    pub fn full(mut self, evaluation_setup: &'a DoryVerifierSetup) -> Self {{ self.evaluation_setup = Some(evaluation_setup); self }}\n}}\n\n"
    )
}

fn jolt_verifier_evaluation_helpers(prefix: &str, field_type: &str) -> String {
    format!(
        r#"pub type {prefix}Stage6BytecodeEntry = crate::stages::stage6::Stage6BytecodeEntry;
pub type {prefix}Stage6BytecodeReadRafData = crate::stages::stage6::Stage6BytecodeReadRafData;
pub type {prefix}Stage6VerifierData = crate::stages::stage6::Stage6VerifierData;

impl stage8_stage::Stage8NamedEvalView<{field_type}> for {prefix}NamedEval {{
    fn name(&self) -> &'static str {{
        self.name
    }}

    fn value(&self) -> {field_type} {{
        self.value
    }}
}}

impl stage8_stage::Stage8SumcheckOutputView<{field_type}> for {prefix}SumcheckOutput {{
    type Eval = {prefix}NamedEval;

    fn point(&self) -> &[{field_type}] {{
        &self.point
    }}

    fn evals(&self) -> &[Self::Eval] {{
        &self.evals
    }}
}}

impl stage8_stage::Stage8OpeningInputView<{field_type}>
    for stage7_stage::Stage7OpeningInputValue<{field_type}>
{{
    fn symbol(&self) -> &'static str {{
        self.symbol
    }}

    fn point(&self) -> &[{field_type}] {{
        &self.point
    }}
}}

impl From<stage8_stage::Stage8EvaluationOpeningPointError> for {prefix}EvaluationProofError {{
    fn from(error: stage8_stage::Stage8EvaluationOpeningPointError) -> Self {{
        match error {{
            stage8_stage::Stage8EvaluationOpeningPointError::MissingStage7EvaluationPoint => {{
                Self::MissingStage7EvaluationPoint
            }}
            stage8_stage::Stage8EvaluationOpeningPointError::InvalidPointLength {{
                artifact,
                expected,
                actual,
            }} => Self::InvalidPointLength {{
                artifact,
                expected,
                actual,
            }},
        }}
    }}
}}

#[expect(
    clippy::too_many_arguments,
    reason = "generated verifier entry point follows the Jolt proof artifact boundary"
)]
pub fn verify_jolt_evaluation_proof<T>(
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
    let _state_span = tracing::info_span!("bolt.verify.evaluation_state").entered();
    let state =
        evaluation_proof_state(program, commitments, stage6, stage7, stage7_openings, transcript)?;
    drop(_state_span);
    let _dory_verify_span = tracing::info_span!("bolt.verify.dory_verify").entered();
    <DoryScheme as CommitmentScheme>::verify(
        &state.joint_commitment,
        &state.opening_point,
        state.joint_claim,
        &proof.joint_opening_proof,
        verifier_setup,
        transcript,
    )?;
    drop(_dory_verify_span);
    let _bind_span = tracing::info_span!("bolt.verify.bind_opening_inputs").entered();
    <DoryScheme as CommitmentScheme>::bind_opening_inputs(
        transcript,
        &state.opening_point,
        &state.joint_claim,
    );
    drop(_bind_span);
    Ok(())
}}

struct EvaluationProofState {{
    opening_point: Vec<{field_type}>,
    joint_claim: {field_type},
    joint_commitment: DoryCommitment,
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
    let (sumcheck_address_point, stage7_values) =
        stage8_stage::stage7_claim_values(program, &stage7.sumchecks)
            .ok_or({prefix}EvaluationProofError::MissingStage7RaEval)?;
    let address_point = stage8_stage::reverse_point(&sumcheck_address_point);
    let (opening_point, _) =
        stage8_stage::stage7_evaluation_opening_point(program, &address_point, stage7_openings)?;
    let lagrange_factor = EqPolynomial::<{field_type}>::zero_selector(&address_point);
    let claims =
        stage8_stage::evaluation_claims(program, &stage6.sumchecks, &stage7_values, lagrange_factor)
            .map_err(|error| {prefix}EvaluationProofError::MissingStageEval {{
                stage: error.stage,
                eval: error.eval,
            }})?;

    stage8_stage::append_rlc_claims(transcript, &claims);
    let gamma_powers = stage8_stage::gamma_powers(transcript, claims.len());
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

fn joint_commitment(
    commitments: &commitment_stage::CommitmentArtifacts,
    claims: &[stage8_stage::Stage8EvaluationClaim<{field_type}>],
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

fn jolt_prover_stage7_opening_input_helpers(field_type: &str) -> String {
    format!(
        r#"pub struct JoltProverStageInputs<'a, CommitmentInputs> {{
    pub commitment_inputs: &'a mut CommitmentInputs,
    pub prover_setup: &'a DoryProverSetup,
    pub stage1_outer: stage1::Stage1ProverInputs<'a, {field_type}>,
    pub stage2: stage2::Stage2ProverInputs<'a, {field_type}>,
    pub stage3: stage3::Stage3ProverInputs<'a, {field_type}>,
    pub stage4: stage4::Stage4ProverInputs<'a, {field_type}>,
    pub stage5: stage5::Stage5ProverInputs<'a, {field_type}>,
    pub stage6: stage6::Stage6ProverInputs<'a, {field_type}>,
    pub stage7: stage7::Stage7ProverInputs<'a, {field_type}>,
    pub stage7_openings: Option<&'a [stage7::Stage7OpeningInputValue<{field_type}>]>,
}}

pub fn prove_jolt_with_stage_inputs<CommitmentInputs, T>(
    inputs: JoltProverStageInputs<'_, CommitmentInputs>,
    programs: JoltProverPrograms,
    transcript: &mut T,
) -> Result<(JoltProof, JoltProverArtifacts), JoltProveError>
where
    CommitmentInputs: commitment_stage::CommitmentInputProvider,
    T: Transcript<Challenge = {field_type}>,
{{
    let JoltProverStageInputs {{
        commitment_inputs,
        prover_setup,
        stage1_outer,
        stage2,
        stage3,
        stage4,
        stage5,
        stage6,
        stage7,
        stage7_openings,
    }} = inputs;
    let mut stage1_outer_executor = stage1::Stage1ProverKernelExecutor::new(stage1_outer);
    let mut stage2_executor = stage2::Stage2ProverKernelExecutor::new(stage2);
    let mut stage3_executor = stage3::Stage3ProverKernelExecutor::new(stage3);
    let mut stage4_executor = stage4::Stage4ProverKernelExecutor::new(stage4);
    let mut stage5_executor = stage5::Stage5ProverKernelExecutor::new(stage5);
    let mut stage6_executor = stage6::Stage6ProverKernelExecutor::new(stage6);
    let mut stage7_executor = stage7::Stage7ProverKernelExecutor::new(stage7);
    prove_jolt_with_programs(
        JoltProverInputs {{
            commitment_inputs,
            prover_setup,
            stage1_outer_executor: &mut stage1_outer_executor,
            stage2_executor: &mut stage2_executor,
            stage3_executor: &mut stage3_executor,
            stage4_executor: &mut stage4_executor,
            stage5_executor: &mut stage5_executor,
            stage6_executor: &mut stage6_executor,
            stage7_executor: &mut stage7_executor,
            stage7_openings,
        }},
        programs,
        transcript,
    )
}}

pub struct JoltProverWitnessInputs<'a, CommitmentInputs> {{
    pub commitment_inputs: &'a mut CommitmentInputs,
    pub prover_setup: &'a DoryProverSetup,
    pub stage1_trace_num_vars: usize,
    pub stage1_outer_evaluator: &'a dyn stage1::Stage1OuterRemainingEvaluator<{field_type}>,
    pub stage2_openings: &'a [stage2::Stage2OpeningInputValue<{field_type}>],
    pub product_virtual_cycles: &'a [stage2::Stage2ProductVirtualCycle],
    pub instruction_lookup_cycles: &'a [stage2::Stage2InstructionLookupCycle],
    pub ram: &'a stage2::Stage2RamData<'a>,
    pub stage3_openings: &'a [stage3::Stage3OpeningInputValue<{field_type}>],
    pub stage3_cycles: &'a [stage3::Stage3Cycle],
    pub stage4_openings: &'a [stage4::Stage4OpeningInputValue<{field_type}>],
    pub register_count: usize,
    pub trace_len: usize,
    pub ram_k: usize,
    pub register_accesses: &'a [stage4::Stage4RegisterAccess],
    pub stage5_openings: &'a [stage5::Stage5OpeningInputValue<{field_type}>],
    pub lookup_indices: &'a [u128],
    pub lookup_table_indices: &'a [Option<usize>],
    pub is_interleaved_operands: &'a [bool],
    pub ra_virtual_log_k_chunk: usize,
    pub stage6_openings: &'a [stage6::Stage6OpeningInputValue<{field_type}>],
    pub stage6_bytecode_data: stage6::Stage6BytecodeReadRafData<'a, {field_type}>,
    pub stage6_witness_params: Stage6WitnessParams,
    pub cycle_inputs: &'a [CycleInput],
    pub instruction_ra_virtual_d: usize,
    pub stage7_openings: &'a [stage7::Stage7OpeningInputValue<{field_type}>],
    pub evaluation_openings: Option<&'a [stage7::Stage7OpeningInputValue<{field_type}>]>,
}}

pub fn prove_jolt_with_witness_inputs<CommitmentInputs, T>(
    inputs: JoltProverWitnessInputs<'_, CommitmentInputs>,
    programs: JoltProverPrograms,
    transcript: &mut T,
) -> Result<(JoltProof, JoltProverArtifacts), JoltProveError>
where
    CommitmentInputs: commitment_stage::CommitmentInputProvider,
    T: Transcript<Challenge = {field_type}>,
{{
    let _input_span = tracing::info_span!("bolt.prove.inputs").entered();
    let _stage1_input_span = tracing::info_span!("bolt.prove.inputs.stage1").entered();
    let stage1_outer =
        stage1_outer_prover_inputs(inputs.stage1_trace_num_vars, inputs.stage1_outer_evaluator);
    drop(_stage1_input_span);
    let _stage2_input_span = tracing::info_span!("bolt.prove.inputs.stage2").entered();
    let stage2 = stage2_prover_inputs(
        inputs.stage2_openings,
        inputs.product_virtual_cycles,
        inputs.instruction_lookup_cycles,
        inputs.ram,
    )?;
    drop(_stage2_input_span);
    let _stage3_input_span = tracing::info_span!("bolt.prove.inputs.stage3").entered();
    let stage3 = stage3_prover_inputs(inputs.stage3_openings, inputs.stage3_cycles);
    drop(_stage3_input_span);
    let _stage45_witness_span = tracing::info_span!("bolt.prove.inputs.stage45_witness").entered();
    let stage45_witness = stage4::stage4_5_sparse_trace_witness_from_accesses(
        inputs.register_accesses,
        inputs.ram.accesses,
    );
    drop(_stage45_witness_span);
    let _stage4_input_span = tracing::info_span!("bolt.prove.inputs.stage4").entered();
    let stage4 = stage4_prover_inputs(
        inputs.stage4_openings,
        inputs.register_count,
        inputs.trace_len,
        inputs.ram_k,
        inputs.register_accesses,
        &stage45_witness,
    );
    drop(_stage4_input_span);
    let _stage5_input_span = tracing::info_span!("bolt.prove.inputs.stage5").entered();
    let stage5 = stage5_prover_inputs(
        inputs.stage5_openings,
        inputs.trace_len,
        inputs.ram_k,
        inputs.register_count,
        inputs.lookup_indices,
        inputs.lookup_table_indices,
        inputs.is_interleaved_operands,
        inputs.ra_virtual_log_k_chunk,
        &stage45_witness,
    );
    drop(_stage5_input_span);
    let _stage6_witness_span = tracing::info_span!("bolt.prove.inputs.stage6_witness").entered();
    let stage6_witness = stage6_witness_from_opening_inputs(
        inputs.stage6_witness_params,
        inputs.cycle_inputs,
        inputs.stage6_openings,
    );
    let stage6_witness_slices = stage6_witness.slices();
    drop(_stage6_witness_span);
    let _stage6_input_span = tracing::info_span!("bolt.prove.inputs.stage6").entered();
    let stage6 = stage6_prover_inputs(
        inputs.stage6_openings,
        inputs.stage6_bytecode_data,
        &stage6_witness,
        &stage6_witness_slices,
        inputs.instruction_ra_virtual_d,
    );
    drop(_stage6_input_span);
    let _stage7_input_span = tracing::info_span!("bolt.prove.inputs.stage7").entered();
    let stage7 = stage7_prover_inputs(inputs.stage7_openings, &stage6_witness_slices);
    drop(_stage7_input_span);
    drop(_input_span);
    prove_jolt_with_stage_inputs(
        JoltProverStageInputs {{
            commitment_inputs: inputs.commitment_inputs,
            prover_setup: inputs.prover_setup,
            stage1_outer,
            stage2,
            stage3,
            stage4,
            stage5,
            stage6,
            stage7,
            stage7_openings: inputs.evaluation_openings,
        }},
        programs,
        transcript,
    )
}}

pub fn stage1_outer_prover_inputs(
    trace_num_vars: usize,
    evaluator: &dyn stage1::Stage1OuterRemainingEvaluator<{field_type}>,
) -> stage1::Stage1ProverInputs<'_, {field_type}> {{
    stage1::Stage1ProverInputs::empty(trace_num_vars).with_outer_remaining_evaluator(evaluator)
}}

pub fn prove_stage1_outer_inputs_with_program<T>(
    program: &'static stage1::Stage1CpuProgramPlan,
    inputs: stage1::Stage1ProverInputs<'_, {field_type}>,
    transcript: &mut T,
) -> Result<stage1::Stage1ExecutionArtifacts<{field_type}>, stage1::Stage1KernelError>
where
    T: Transcript<Challenge = {field_type}>,
{{
    let mut executor = stage1::Stage1ProverKernelExecutor::new(inputs);
    stage1_outer_stage::prove_stage1_outer_with_program(program, &mut executor, transcript)
}}

pub fn prove_stage1_outer_with_witness_inputs<T>(
    program: &'static stage1::Stage1CpuProgramPlan,
    trace_num_vars: usize,
    evaluator: &dyn stage1::Stage1OuterRemainingEvaluator<{field_type}>,
    transcript: &mut T,
) -> Result<stage1::Stage1ExecutionArtifacts<{field_type}>, stage1::Stage1KernelError>
where
    T: Transcript<Challenge = {field_type}>,
{{
    let inputs = stage1_outer_prover_inputs(trace_num_vars, evaluator);
    prove_stage1_outer_inputs_with_program(program, inputs, transcript)
}}

pub fn replay_stage1_outer_proof_with_program<T>(
    program: &'static stage1::Stage1CpuProgramPlan,
    proof: &stage1::Stage1Proof<{field_type}>,
    transcript: &mut T,
) -> Result<stage1::Stage1ExecutionArtifacts<{field_type}>, stage1::Stage1KernelError>
where
    T: Transcript<Challenge = {field_type}>,
{{
    let mut executor = stage1::Stage1VerifierKernelExecutor::new(proof);
    stage1::execute_stage1_program(
        program,
        stage1::Stage1ExecutionMode::Verifier,
        &mut executor,
        transcript,
    )
}}

pub fn stage1_outer_proof_from_kernel_proof(
    proof: &stage1::Stage1Proof<{field_type}>,
) -> JoltStageProof {{
    JoltStageProof {{
        sumchecks: proof
            .sumchecks
            .iter()
            .map(stage1_outer_sumcheck)
            .collect(),
    }}
}}

pub fn stage2_prover_inputs<'a>(
    opening_inputs: &'a [stage2::Stage2OpeningInputValue<{field_type}>],
    product_virtual_cycles: &'a [stage2::Stage2ProductVirtualCycle],
    instruction_lookup_cycles: &'a [stage2::Stage2InstructionLookupCycle],
    ram: &'a stage2::Stage2RamData<'a>,
) -> Result<stage2::Stage2ProverInputs<'a, {field_type}>, stage2::Stage2KernelError> {{
    Ok(stage2::Stage2ProverInputs::new(opening_inputs)
        .with_product_virtual_witness(product_virtual_cycles)?
        .with_instruction_lookup_cycles(instruction_lookup_cycles)
        .with_ram_data(ram))
}}

pub struct JoltStage2RamDataStorage<'a> {{
    log_k: usize,
    start_address: u64,
    initial_ram: &'a [u64],
    final_ram: &'a [u64],
    accesses: Vec<JoltStage2RamAccess>,
    output_layout: Option<JoltStage2RamOutputLayout>,
}}

impl<'a> JoltStage2RamDataStorage<'a> {{
    pub fn from_kernel(ram: &stage2::Stage2RamData<'a>) -> Self {{
        Self {{
            log_k: ram.log_k,
            start_address: ram.start_address,
            initial_ram: ram.initial_ram,
            final_ram: ram.final_ram,
            accesses: ram
                .accesses
                .iter()
                .map(|access| JoltStage2RamAccess {{
                    remapped_address: access.remapped_address,
                    read_value: access.read_value,
                    write_value: access.write_value,
                }})
                .collect(),
            output_layout: ram.output_layout.map(|layout| JoltStage2RamOutputLayout {{
                io_start: layout.io_start,
                io_end: layout.io_end,
            }}),
        }}
    }}

    pub fn as_input(&self) -> JoltStage2RamData<'_> {{
        JoltStage2RamData {{
            log_k: self.log_k,
            start_address: self.start_address,
            initial_ram: self.initial_ram,
            final_ram: self.final_ram,
            accesses: &self.accesses,
            output_layout: self.output_layout,
        }}
    }}
}}

pub fn stage2_verifier_ram_data<'a>(
    ram: &stage2::Stage2RamData<'a>,
) -> JoltStage2RamDataStorage<'a> {{
    JoltStage2RamDataStorage::from_kernel(ram)
}}

pub trait JoltKernelOpeningInput {{
    fn symbol(&self) -> &'static str;
    fn point(&self) -> &[{field_type}];
    fn eval(&self) -> {field_type};
}}

macro_rules! impl_jolt_kernel_opening_input {{
    ($opening:ty) => {{
        impl JoltKernelOpeningInput for $opening {{
            fn symbol(&self) -> &'static str {{
                self.symbol
            }}

            fn point(&self) -> &[{field_type}] {{
                &self.point
            }}

            fn eval(&self) -> {field_type} {{
                self.eval
            }}
        }}
    }};
}}

impl_jolt_kernel_opening_input!(stage2::Stage2OpeningInputValue<{field_type}>);
impl_jolt_kernel_opening_input!(stage3::Stage3OpeningInputValue<{field_type}>);
impl_jolt_kernel_opening_input!(stage4::Stage4OpeningInputValue<{field_type}>);

pub fn verifier_opening_inputs_from_kernel<I>(inputs: &[I]) -> Vec<JoltStageOpeningInputValue>
where
    I: JoltKernelOpeningInput,
{{
    inputs
        .iter()
        .map(|input| JoltStageOpeningInputValue {{
            symbol: input.symbol(),
            point: input.point().to_vec(),
            eval: input.eval(),
        }})
        .collect()
}}

pub fn prove_stage2_inputs_with_program<T>(
    program: &'static stage2::Stage2CpuProgramPlan,
    inputs: stage2::Stage2ProverInputs<'_, {field_type}>,
    transcript: &mut T,
) -> Result<stage2::Stage2ExecutionArtifacts<{field_type}>, stage2::Stage2KernelError>
where
    T: Transcript<Challenge = {field_type}>,
{{
    let mut executor = stage2::Stage2ProverKernelExecutor::new(inputs);
    stage2_stage::execute_stage2_prover_with_program(program, &mut executor, transcript)
}}

pub fn prove_stage2_with_witness_inputs<'a, T>(
    program: &'static stage2::Stage2CpuProgramPlan,
    opening_inputs: &'a [stage2::Stage2OpeningInputValue<{field_type}>],
    product_virtual_cycles: &'a [stage2::Stage2ProductVirtualCycle],
    instruction_lookup_cycles: &'a [stage2::Stage2InstructionLookupCycle],
    ram: &'a stage2::Stage2RamData<'a>,
    transcript: &mut T,
) -> Result<stage2::Stage2ExecutionArtifacts<{field_type}>, stage2::Stage2KernelError>
where
    T: Transcript<Challenge = {field_type}>,
{{
    let inputs = stage2_prover_inputs(
        opening_inputs,
        product_virtual_cycles,
        instruction_lookup_cycles,
        ram,
    )?;
    prove_stage2_inputs_with_program(program, inputs, transcript)
}}

pub fn stage2_opening_inputs_from_artifacts(
    program: &'static stage2::Stage2CpuProgramPlan,
    stage1_artifacts: &stage1::Stage1ExecutionArtifacts<{field_type}>,
) -> Result<Vec<stage2::Stage2OpeningInputValue<{field_type}>>, JoltOpeningInputError> {{
    program
        .opening_inputs
        .iter()
        .map(|input| {{
            let (point, eval) = match input.source_stage {{
                "stage1" => stage1_opening_claim(stage1_artifacts, input.source_claim)?,
                source_stage => {{
                    return Err(JoltOpeningInputError::UnsupportedOpeningInputSource {{
                        stage: "stage2",
                        symbol: input.symbol,
                        source_stage,
                    }});
                }}
            }};
            validate_point_len(input.symbol, input.point_arity, point.len())?;
            Ok(stage2::Stage2OpeningInputValue {{
                symbol: input.symbol,
                point,
                eval,
            }})
        }})
        .collect()
}}

pub fn replay_stage2_proof_with_program<'a, T>(
    program: &'static stage2::Stage2CpuProgramPlan,
    proof: &'a stage2::Stage2Proof<{field_type}>,
    opening_inputs: &'a [stage2::Stage2OpeningInputValue<{field_type}>],
    ram: Option<&'a stage2::Stage2RamData<'a>>,
    transcript: &mut T,
) -> Result<stage2::Stage2ExecutionArtifacts<{field_type}>, stage2::Stage2KernelError>
where
    T: Transcript<Challenge = {field_type}>,
{{
    let mut executor = stage2::Stage2VerifierKernelExecutor::new(proof, opening_inputs);
    if let Some(ram) = ram {{
        executor = executor.with_ram_data(ram);
    }}
    stage2::execute_stage2_program(
        program,
        stage2::Stage2ExecutionMode::Verifier,
        &mut executor,
        transcript,
    )
}}

pub fn stage3_prover_inputs<'a>(
    opening_inputs: &'a [stage3::Stage3OpeningInputValue<{field_type}>],
    cycles: &'a [stage3::Stage3Cycle],
) -> stage3::Stage3ProverInputs<'a, {field_type}> {{
    stage3::Stage3ProverInputs::new(opening_inputs).with_cycles(cycles)
}}

pub fn prove_stage3_inputs_with_program<T>(
    program: &'static stage3::Stage3CpuProgramPlan,
    inputs: stage3::Stage3ProverInputs<'_, {field_type}>,
    transcript: &mut T,
) -> Result<stage3::Stage3ExecutionArtifacts<{field_type}>, stage3::Stage3KernelError>
where
    T: Transcript<Challenge = {field_type}>,
{{
    let mut executor = stage3::Stage3ProverKernelExecutor::new(inputs);
    stage3_stage::execute_stage3_prover_with_program(program, &mut executor, transcript)
}}

pub fn prove_stage3_with_witness_inputs<T>(
    program: &'static stage3::Stage3CpuProgramPlan,
    opening_inputs: &[stage3::Stage3OpeningInputValue<{field_type}>],
    cycles: &[stage3::Stage3Cycle],
    transcript: &mut T,
) -> Result<stage3::Stage3ExecutionArtifacts<{field_type}>, stage3::Stage3KernelError>
where
    T: Transcript<Challenge = {field_type}>,
{{
    let inputs = stage3_prover_inputs(opening_inputs, cycles);
    prove_stage3_inputs_with_program(program, inputs, transcript)
}}

pub fn stage3_opening_inputs_from_artifacts(
    program: &'static stage3::Stage3CpuProgramPlan,
    stage1_artifacts: &stage1::Stage1ExecutionArtifacts<{field_type}>,
    stage2_artifacts: &stage2::Stage2ExecutionArtifacts<{field_type}>,
) -> Result<Vec<stage3::Stage3OpeningInputValue<{field_type}>>, JoltOpeningInputError> {{
    program
        .opening_inputs
        .iter()
        .map(|input| {{
            let (point, eval) = match input.source_stage {{
                "stage1" => stage1_opening_claim(stage1_artifacts, input.source_claim)?,
                "stage2" => stage2_opening_claim(stage2_artifacts, input.source_claim)?,
                source_stage => {{
                    return Err(JoltOpeningInputError::UnsupportedOpeningInputSource {{
                        stage: "stage3",
                        symbol: input.symbol,
                        source_stage,
                    }});
                }}
            }};
            validate_point_len(input.symbol, input.point_arity, point.len())?;
            Ok(stage3::Stage3OpeningInputValue {{
                symbol: input.symbol,
                point,
                eval,
            }})
        }})
        .collect()
}}

pub fn replay_stage3_proof_with_program<T>(
    program: &'static stage3::Stage3CpuProgramPlan,
    proof: &stage3::Stage3Proof<{field_type}>,
    opening_inputs: &[stage3::Stage3OpeningInputValue<{field_type}>],
    transcript: &mut T,
) -> Result<stage3::Stage3ExecutionArtifacts<{field_type}>, stage3::Stage3KernelError>
where
    T: Transcript<Challenge = {field_type}>,
{{
    let mut executor = stage3::Stage3VerifierKernelExecutor::new(proof, opening_inputs);
    stage3::execute_stage3_program(
        program,
        stage3::Stage3ExecutionMode::Verifier,
        &mut executor,
        transcript,
    )
}}

pub fn stage4_prover_inputs<'a>(
    opening_inputs: &'a [stage4::Stage4OpeningInputValue<{field_type}>],
    register_count: usize,
    trace_len: usize,
    ram_k: usize,
    register_accesses: &'a [stage4::Stage4RegisterAccess],
    witness: &'a Stage45SparseTraceWitness<{field_type}>,
) -> stage4::Stage4ProverInputs<'a, {field_type}> {{
    stage4::Stage4ProverInputs::new(opening_inputs).with_stage45_sparse_trace_witness(
        register_count,
        trace_len,
        ram_k,
        register_accesses,
        witness,
    )
}}

pub fn prove_stage4_inputs_with_program<T>(
    program: &'static stage4::Stage4CpuProgramPlan,
    inputs: stage4::Stage4ProverInputs<'_, {field_type}>,
    transcript: &mut T,
) -> Result<stage4::Stage4ExecutionArtifacts<{field_type}>, stage4::Stage4KernelError>
where
    T: Transcript<Challenge = {field_type}>,
{{
    let mut executor = stage4::Stage4ProverKernelExecutor::new(inputs);
    stage4_stage::execute_stage4_prover_with_program(program, &mut executor, transcript)
}}

pub fn prove_stage4_with_witness_inputs<T>(
    program: &'static stage4::Stage4CpuProgramPlan,
    opening_inputs: &[stage4::Stage4OpeningInputValue<{field_type}>],
    register_count: usize,
    trace_len: usize,
    ram_k: usize,
    register_accesses: &[stage4::Stage4RegisterAccess],
    witness: &Stage45SparseTraceWitness<{field_type}>,
    transcript: &mut T,
) -> Result<stage4::Stage4ExecutionArtifacts<{field_type}>, stage4::Stage4KernelError>
where
    T: Transcript<Challenge = {field_type}>,
{{
    let inputs = stage4_prover_inputs(
        opening_inputs,
        register_count,
        trace_len,
        ram_k,
        register_accesses,
        witness,
    );
    prove_stage4_inputs_with_program(program, inputs, transcript)
}}

pub fn prove_stage4_with_trace_witness_inputs<T>(
    program: &'static stage4::Stage4CpuProgramPlan,
    opening_inputs: &[stage4::Stage4OpeningInputValue<{field_type}>],
    register_count: usize,
    trace_len: usize,
    ram_k: usize,
    register_accesses: &[stage4::Stage4RegisterAccess],
    ram_accesses: &[stage2::Stage2RamAccess],
    transcript: &mut T,
) -> Result<stage4::Stage4ExecutionArtifacts<{field_type}>, stage4::Stage4KernelError>
where
    T: Transcript<Challenge = {field_type}>,
{{
    let witness = stage4::stage4_5_sparse_trace_witness_from_accesses(
        register_accesses,
        ram_accesses,
    );
    prove_stage4_with_witness_inputs(
        program,
        opening_inputs,
        register_count,
        trace_len,
        ram_k,
        register_accesses,
        &witness,
        transcript,
    )
}}

pub fn stage4_opening_inputs_from_artifacts(
    program: &'static stage4::Stage4CpuProgramPlan,
    initial_ram_state: &[u64],
    stage2_artifacts: &stage2::Stage2ExecutionArtifacts<{field_type}>,
    stage3_artifacts: &stage3::Stage3ExecutionArtifacts<{field_type}>,
) -> Result<Vec<stage4::Stage4OpeningInputValue<{field_type}>>, JoltOpeningInputError> {{
    program
        .opening_inputs
        .iter()
        .map(|input| {{
            let (point, eval) = match input.source_stage {{
                "stage2" => stage2_opening_claim(stage2_artifacts, input.source_claim)?,
                "stage3" => stage3_opening_claim(stage3_artifacts, input.source_claim)?,
                "stage4_precomputed" => {{
                    let (point, _) = stage2_opening_claim(
                        stage2_artifacts,
                        "stage2.ram_output.opening.RamValFinal",
                    )?;
                    stage4_ram_val_init_opening(initial_ram_state, &point)
                }}
                source_stage => {{
                    return Err(JoltOpeningInputError::UnsupportedOpeningInputSource {{
                        stage: "stage4",
                        symbol: input.symbol,
                        source_stage,
                    }});
                }}
            }};
            opening_input_value(input.symbol, input.point_arity, point, eval)
        }})
        .collect()
}}

pub fn replay_stage4_proof_with_program<T>(
    program: &'static stage4::Stage4CpuProgramPlan,
    proof: &stage4::Stage4Proof<{field_type}>,
    opening_inputs: &[stage4::Stage4OpeningInputValue<{field_type}>],
    transcript: &mut T,
) -> Result<stage4::Stage4ExecutionArtifacts<{field_type}>, stage4::Stage4KernelError>
where
    T: Transcript<Challenge = {field_type}>,
{{
    let mut executor = stage4::Stage4VerifierKernelExecutor::new(proof, opening_inputs);
    stage4::execute_stage4_program(
        program,
        stage4::Stage4ExecutionMode::Verifier,
        &mut executor,
        transcript,
    )
}}

pub fn stage5_prover_inputs<'a>(
    opening_inputs: &'a [stage5::Stage5OpeningInputValue<{field_type}>],
    trace_len: usize,
    ram_k: usize,
    register_count: usize,
    lookup_indices: &'a [u128],
    lookup_table_indices: &'a [Option<usize>],
    is_interleaved_operands: &'a [bool],
    ra_virtual_log_k_chunk: usize,
    witness: &'a Stage45SparseTraceWitness<{field_type}>,
) -> stage5::Stage5ProverInputs<'a, {field_type}> {{
    stage5::Stage5ProverInputs::new(opening_inputs).with_stage45_sparse_trace_witness(
        trace_len,
        ram_k,
        register_count,
        lookup_indices,
        lookup_table_indices,
        is_interleaved_operands,
        ra_virtual_log_k_chunk,
        witness,
    )
}}

pub fn prove_stage5_inputs_with_program<T>(
    program: &'static stage5::Stage5CpuProgramPlan,
    inputs: stage5::Stage5ProverInputs<'_, {field_type}>,
    transcript: &mut T,
) -> Result<stage5::Stage5ExecutionArtifacts<{field_type}>, stage5::Stage5KernelError>
where
    T: Transcript<Challenge = {field_type}>,
{{
    let mut executor = stage5::Stage5ProverKernelExecutor::new(inputs);
    stage5_stage::execute_stage5_prover_with_program(program, &mut executor, transcript)
}}

pub fn prove_stage5_with_witness_inputs<T>(
    program: &'static stage5::Stage5CpuProgramPlan,
    opening_inputs: &[stage5::Stage5OpeningInputValue<{field_type}>],
    trace_len: usize,
    ram_k: usize,
    register_count: usize,
    lookup_indices: &[u128],
    lookup_table_indices: &[Option<usize>],
    is_interleaved_operands: &[bool],
    ra_virtual_log_k_chunk: usize,
    witness: &Stage45SparseTraceWitness<{field_type}>,
    transcript: &mut T,
) -> Result<stage5::Stage5ExecutionArtifacts<{field_type}>, stage5::Stage5KernelError>
where
    T: Transcript<Challenge = {field_type}>,
{{
    let inputs = stage5_prover_inputs(
        opening_inputs,
        trace_len,
        ram_k,
        register_count,
        lookup_indices,
        lookup_table_indices,
        is_interleaved_operands,
        ra_virtual_log_k_chunk,
        witness,
    );
    prove_stage5_inputs_with_program(program, inputs, transcript)
}}

pub fn prove_stage5_with_trace_witness_inputs<T>(
    program: &'static stage5::Stage5CpuProgramPlan,
    opening_inputs: &[stage5::Stage5OpeningInputValue<{field_type}>],
    trace_len: usize,
    ram_k: usize,
    register_count: usize,
    lookup_indices: &[u128],
    lookup_table_indices: &[Option<usize>],
    is_interleaved_operands: &[bool],
    ra_virtual_log_k_chunk: usize,
    register_accesses: &[stage4::Stage4RegisterAccess],
    ram_accesses: &[stage2::Stage2RamAccess],
    transcript: &mut T,
) -> Result<stage5::Stage5ExecutionArtifacts<{field_type}>, stage5::Stage5KernelError>
where
    T: Transcript<Challenge = {field_type}>,
{{
    let witness = stage4::stage4_5_sparse_trace_witness_from_accesses(
        register_accesses,
        ram_accesses,
    );
    prove_stage5_with_witness_inputs(
        program,
        opening_inputs,
        trace_len,
        ram_k,
        register_count,
        lookup_indices,
        lookup_table_indices,
        is_interleaved_operands,
        ra_virtual_log_k_chunk,
        &witness,
        transcript,
    )
}}

pub fn stage5_opening_inputs_from_artifacts(
    program: &'static stage5::Stage5CpuProgramPlan,
    stage2_artifacts: &stage2::Stage2ExecutionArtifacts<{field_type}>,
    stage4_artifacts: &stage4::Stage4ExecutionArtifacts<{field_type}>,
) -> Result<Vec<stage5::Stage5OpeningInputValue<{field_type}>>, JoltOpeningInputError> {{
    program
        .opening_inputs
        .iter()
        .map(|input| {{
            let (point, eval) = match input.source_stage {{
                "stage2" => stage2_opening_claim(stage2_artifacts, input.source_claim)?,
                "stage4" => stage4_opening_claim(stage4_artifacts, input.source_claim)?,
                source_stage => {{
                    return Err(JoltOpeningInputError::UnsupportedOpeningInputSource {{
                        stage: "stage5",
                        symbol: input.symbol,
                        source_stage,
                    }});
                }}
            }};
            opening_input_value(input.symbol, input.point_arity, point, eval)
        }})
        .collect()
}}

pub fn stage5_kernel_proof(
    artifacts: &stage5::Stage5ExecutionArtifacts<{field_type}>,
) -> stage5::Stage5Proof<{field_type}> {{
    stage5::Stage5Proof {{
        sumchecks: artifacts.sumchecks.clone(),
    }}
}}

pub fn jolt_proof_through_stage5(
    commitments: &[Option<DoryCommitment>],
    stage1_artifacts: &stage1::Stage1ExecutionArtifacts<{field_type}>,
    stage2_artifacts: &stage2::Stage2ExecutionArtifacts<{field_type}>,
    stage3_artifacts: &stage3::Stage3ExecutionArtifacts<{field_type}>,
    stage4_artifacts: &stage4::Stage4ExecutionArtifacts<{field_type}>,
    stage5_proof: &JoltStageProof,
) -> JoltProof {{
    JoltProof {{
        commitments: commitments.to_vec(),
        stage1_outer: stage1_outer_proof(stage1_artifacts),
        stage2: stage2_proof(stage2_artifacts),
        stage3: stage3_proof(stage3_artifacts),
        stage4: stage4_proof(stage4_artifacts),
        stage5: stage5_proof.clone(),
        stage6: JoltStageProof::default(),
        stage7: JoltStageProof::default(),
        evaluation: None,
    }}
}}

pub fn jolt_proof_through_stage6(
    commitments: &[Option<DoryCommitment>],
    stage1_artifacts: &stage1::Stage1ExecutionArtifacts<{field_type}>,
    stage2_artifacts: &stage2::Stage2ExecutionArtifacts<{field_type}>,
    stage3_artifacts: &stage3::Stage3ExecutionArtifacts<{field_type}>,
    stage4_artifacts: &stage4::Stage4ExecutionArtifacts<{field_type}>,
    stage5_proof: &JoltStageProof,
    stage6_proof: &JoltStageProof,
) -> JoltProof {{
    let mut proof = jolt_proof_through_stage5(
        commitments,
        stage1_artifacts,
        stage2_artifacts,
        stage3_artifacts,
        stage4_artifacts,
        stage5_proof,
    );
    proof.stage6 = stage6_proof.clone();
    proof
}}

pub fn jolt_proof_through_stage7(
    commitments: &[Option<DoryCommitment>],
    stage1_artifacts: &stage1::Stage1ExecutionArtifacts<{field_type}>,
    stage2_artifacts: &stage2::Stage2ExecutionArtifacts<{field_type}>,
    stage3_artifacts: &stage3::Stage3ExecutionArtifacts<{field_type}>,
    stage4_artifacts: &stage4::Stage4ExecutionArtifacts<{field_type}>,
    stage5_proof: &JoltStageProof,
    stage6_proof: &JoltStageProof,
    stage7_proof: &JoltStageProof,
) -> JoltProof {{
    let mut proof = jolt_proof_through_stage6(
        commitments,
        stage1_artifacts,
        stage2_artifacts,
        stage3_artifacts,
        stage4_artifacts,
        stage5_proof,
        stage6_proof,
    );
    proof.stage7 = stage7_proof.clone();
    proof
}}

pub fn replay_stage5_proof_with_program<T>(
    program: &'static stage5::Stage5CpuProgramPlan,
    proof: &stage5::Stage5Proof<{field_type}>,
    opening_inputs: &[stage5::Stage5OpeningInputValue<{field_type}>],
    transcript: &mut T,
) -> Result<stage5::Stage5ExecutionArtifacts<{field_type}>, stage5::Stage5KernelError>
where
    T: Transcript<Challenge = {field_type}>,
{{
    let mut executor = stage5::Stage5ProofCarryingKernelExecutor::new(proof, opening_inputs);
    stage5_stage::execute_stage5_prover_with_program(program, &mut executor, transcript)
}}

pub fn stage6_witness_from_opening_inputs(
    params: Stage6WitnessParams,
    cycle_inputs: &[CycleInput],
    opening_inputs: &[stage6::Stage6OpeningInputValue<{field_type}>],
) -> Stage6WitnessPolynomials<{field_type}> {{
    stage6::stage6_witness_from_opening_inputs(params, cycle_inputs, opening_inputs)
}}

pub fn stage6_bytecode_read_raf_data_from_witness_entries(
    entries: &[WitnessStage6BytecodeEntry<{field_type}>],
    entry_bytecode_index: usize,
    num_lookup_tables: usize,
) -> stage6::Stage6BytecodeReadRafDataStorage<{field_type}> {{
    stage6::Stage6BytecodeReadRafDataStorage::from_witness_entries(
        entries,
        entry_bytecode_index,
        num_lookup_tables,
    )
}}

pub fn stage6_verifier_data_from_witness_entries(
    entries: &[WitnessStage6BytecodeEntry<{field_type}>],
    entry_bytecode_index: usize,
    num_lookup_tables: usize,
) -> JoltStage6VerifierData {{
    JoltStage6VerifierData {{
        bytecode_read_raf: Some(JoltStage6BytecodeReadRafData {{
            entries: entries
                .iter()
                .map(|entry| JoltStage6BytecodeEntry {{
                    address: entry.address,
                    imm: entry.imm,
                    circuit_flags: entry.circuit_flags,
                    rd: entry.rd,
                    rs1: entry.rs1,
                    rs2: entry.rs2,
                    lookup_table: entry.lookup_table,
                    is_interleaved: entry.is_interleaved,
                    is_branch: entry.is_branch,
                    left_is_rs1: entry.left_is_rs1,
                    left_is_pc: entry.left_is_pc,
                    right_is_rs2: entry.right_is_rs2,
                    right_is_imm: entry.right_is_imm,
                    is_noop: entry.is_noop,
                }})
                .collect(),
            entry_bytecode_index,
            num_lookup_tables,
        }}),
    }}
}}

pub fn stage6_prover_inputs<'a>(
    opening_inputs: &'a [stage6::Stage6OpeningInputValue<{field_type}>],
    bytecode_data: stage6::Stage6BytecodeReadRafData<'a, {field_type}>,
    witness: &'a Stage6WitnessPolynomials<{field_type}>,
    slices: &'a Stage6WitnessSlices<'a, {field_type}>,
    instruction_ra_virtual_d: usize,
) -> stage6::Stage6ProverInputs<'a, {field_type}> {{
    stage6::Stage6ProverInputs::new(opening_inputs).with_stage6_witness(
        bytecode_data,
        witness,
        slices,
        instruction_ra_virtual_d,
    )
}}

pub fn prove_stage6_inputs_with_program<T>(
    program: &'static stage6::Stage6CpuProgramPlan,
    inputs: stage6::Stage6ProverInputs<'_, {field_type}>,
    transcript: &mut T,
) -> Result<stage6::Stage6ExecutionArtifacts<{field_type}>, stage6::Stage6KernelError>
where
    T: Transcript<Challenge = {field_type}>,
{{
    let mut executor = stage6::Stage6ProverKernelExecutor::new(inputs);
    stage6_stage::execute_stage6_prover_with_program(program, &mut executor, transcript)
}}

pub fn prove_stage6_with_witness_inputs<T>(
    program: &'static stage6::Stage6CpuProgramPlan,
    opening_inputs: &[stage6::Stage6OpeningInputValue<{field_type}>],
    bytecode_data: stage6::Stage6BytecodeReadRafData<'_, {field_type}>,
    witness: &Stage6WitnessPolynomials<{field_type}>,
    slices: &Stage6WitnessSlices<'_, {field_type}>,
    instruction_ra_virtual_d: usize,
    transcript: &mut T,
) -> Result<stage6::Stage6ExecutionArtifacts<{field_type}>, stage6::Stage6KernelError>
where
    T: Transcript<Challenge = {field_type}>,
{{
    let inputs = stage6_prover_inputs(
        opening_inputs,
        bytecode_data,
        witness,
        slices,
        instruction_ra_virtual_d,
    );
    prove_stage6_inputs_with_program(program, inputs, transcript)
}}

pub fn prove_stage6_with_trace_witness_inputs<T>(
    program: &'static stage6::Stage6CpuProgramPlan,
    opening_inputs: &[stage6::Stage6OpeningInputValue<{field_type}>],
    bytecode_data: stage6::Stage6BytecodeReadRafData<'_, {field_type}>,
    witness_params: Stage6WitnessParams,
    cycle_inputs: &[CycleInput],
    instruction_ra_virtual_d: usize,
    transcript: &mut T,
) -> Result<stage6::Stage6ExecutionArtifacts<{field_type}>, stage6::Stage6KernelError>
where
    T: Transcript<Challenge = {field_type}>,
{{
    let witness = stage6_witness_from_opening_inputs(witness_params, cycle_inputs, opening_inputs);
    let slices = witness.slices();
    prove_stage6_with_witness_inputs(
        program,
        opening_inputs,
        bytecode_data,
        &witness,
        &slices,
        instruction_ra_virtual_d,
        transcript,
    )
}}

pub fn stage6_opening_inputs_from_artifacts(
    program: &'static stage6::Stage6CpuProgramPlan,
    stage1_artifacts: &stage1::Stage1ExecutionArtifacts<{field_type}>,
    stage2_artifacts: &stage2::Stage2ExecutionArtifacts<{field_type}>,
    stage3_artifacts: &stage3::Stage3ExecutionArtifacts<{field_type}>,
    stage4_artifacts: &stage4::Stage4ExecutionArtifacts<{field_type}>,
    stage5_artifacts: &stage5::Stage5ExecutionArtifacts<{field_type}>,
) -> Result<Vec<stage6::Stage6OpeningInputValue<{field_type}>>, JoltOpeningInputError> {{
    program
        .opening_inputs
        .iter()
        .map(|input| {{
            let (point, eval) = match input.source_stage {{
                "stage1" => stage1_opening_claim(stage1_artifacts, input.source_claim)?,
                "stage2" => stage2_opening_claim(stage2_artifacts, input.source_claim)?,
                "stage3" => stage3_opening_claim(stage3_artifacts, input.source_claim)?,
                "stage4" => stage4_opening_claim(stage4_artifacts, input.source_claim)?,
                "stage5" => stage5_opening_claim(stage5_artifacts, input.source_claim)?,
                source_stage => {{
                    return Err(JoltOpeningInputError::UnsupportedOpeningInputSource {{
                        stage: "stage6",
                        symbol: input.symbol,
                        source_stage,
                    }});
                }}
            }};
            opening_input_value(input.symbol, input.point_arity, point, eval)
        }})
        .collect()
}}

pub fn stage6_kernel_proof(proof: &JoltStageProof) -> stage6::Stage6Proof<{field_type}> {{
    stage6::Stage6Proof {{
        sumchecks: proof
            .sumchecks
            .iter()
            .map(stage6_kernel_sumcheck_output)
            .collect(),
    }}
}}

fn stage6_kernel_sumcheck_output(
    output: &JoltSumcheckOutput,
) -> stage6::Stage6SumcheckOutput<{field_type}> {{
    stage6::Stage6SumcheckOutput {{
        driver: output.driver,
        point: output.point.clone(),
        evals: output.evals.iter().map(stage6_kernel_eval).collect(),
        opening_claims: Vec::new(),
        proof: output.proof.clone(),
    }}
}}

fn stage6_kernel_eval(eval: &JoltNamedEval) -> stage6::Stage6NamedEval<{field_type}> {{
    stage6::Stage6NamedEval {{
        name: eval.name,
        oracle: eval.oracle,
        value: eval.value,
    }}
}}

pub fn stage6_execution_artifacts(
    artifacts: &stage6::Stage6ExecutionArtifacts<{field_type}>,
) -> JoltStageExecutionArtifacts {{
    JoltStageExecutionArtifacts {{
        challenge_vectors: artifacts
            .challenge_vectors
            .iter()
            .map(|challenge| JoltStageChallengeVector {{
                symbol: challenge.symbol,
                values: challenge.values.clone(),
            }})
            .collect(),
        sumchecks: stage6_proof(artifacts).sumchecks,
        opening_batches: Vec::new(),
    }}
}}

pub fn replay_stage6_proof_with_program<'a, T>(
    program: &'static stage6::Stage6CpuProgramPlan,
    proof: &'a stage6::Stage6Proof<{field_type}>,
    opening_inputs: &'a [stage6::Stage6OpeningInputValue<{field_type}>],
    bytecode_data: Option<stage6::Stage6BytecodeReadRafData<'a, {field_type}>>,
    transcript: &mut T,
) -> Result<stage6::Stage6ExecutionArtifacts<{field_type}>, stage6::Stage6KernelError>
where
    T: Transcript<Challenge = {field_type}>,
{{
    let mut executor = stage6::Stage6ProofCarryingKernelExecutor::new(proof, opening_inputs);
    if let Some(bytecode_data) = bytecode_data {{
        executor = executor.with_bytecode_read_raf_data(bytecode_data);
    }}
    stage6_stage::execute_stage6_prover_with_program(program, &mut executor, transcript)
}}

pub fn stage7_prover_inputs<'a>(
    opening_inputs: &'a [stage7::Stage7OpeningInputValue<{field_type}>],
    slices: &'a Stage6WitnessSlices<'a, {field_type}>,
) -> stage7::Stage7ProverInputs<'a, {field_type}> {{
    stage7::Stage7ProverInputs::new(opening_inputs).with_stage6_witness_indices(slices)
}}

pub fn prove_stage7_inputs_with_program<T>(
    program: &'static stage7::Stage7CpuProgramPlan,
    inputs: stage7::Stage7ProverInputs<'_, {field_type}>,
    transcript: &mut T,
) -> Result<stage7::Stage7ExecutionArtifacts<{field_type}>, stage7::Stage7KernelError>
where
    T: Transcript<Challenge = {field_type}>,
{{
    let mut executor = stage7::Stage7ProverKernelExecutor::new(inputs);
    stage7_stage::execute_stage7_prover_with_program(program, &mut executor, transcript)
}}

pub fn prove_stage7_with_witness_inputs<T>(
    program: &'static stage7::Stage7CpuProgramPlan,
    opening_inputs: &[stage7::Stage7OpeningInputValue<{field_type}>],
    slices: &Stage6WitnessSlices<'_, {field_type}>,
    transcript: &mut T,
) -> Result<stage7::Stage7ExecutionArtifacts<{field_type}>, stage7::Stage7KernelError>
where
    T: Transcript<Challenge = {field_type}>,
{{
    let inputs = stage7_prover_inputs(opening_inputs, slices);
    prove_stage7_inputs_with_program(program, inputs, transcript)
}}

pub fn prove_stage7_with_trace_witness_inputs<T>(
    program: &'static stage7::Stage7CpuProgramPlan,
    opening_inputs: &[stage7::Stage7OpeningInputValue<{field_type}>],
    witness_params: Stage6WitnessParams,
    cycle_inputs: &[CycleInput],
    stage6_openings: &[stage6::Stage6OpeningInputValue<{field_type}>],
    transcript: &mut T,
) -> Result<stage7::Stage7ExecutionArtifacts<{field_type}>, stage7::Stage7KernelError>
where
    T: Transcript<Challenge = {field_type}>,
{{
    let witness = stage6_witness_from_opening_inputs(witness_params, cycle_inputs, stage6_openings);
    let slices = witness.slices();
    prove_stage7_with_witness_inputs(program, opening_inputs, &slices, transcript)
}}

pub fn stage7_kernel_proof(proof: &JoltStageProof) -> stage7::Stage7Proof<{field_type}> {{
    stage7::Stage7Proof {{
        sumchecks: proof
            .sumchecks
            .iter()
            .map(stage7_kernel_sumcheck_output)
            .collect(),
    }}
}}

fn stage7_kernel_sumcheck_output(
    output: &JoltSumcheckOutput,
) -> stage7::Stage7SumcheckOutput<{field_type}> {{
    stage7::Stage7SumcheckOutput {{
        driver: output.driver,
        point: output.point.clone(),
        evals: output.evals.iter().map(stage7_kernel_eval).collect(),
        opening_claims: Vec::new(),
        proof: output.proof.clone(),
    }}
}}

fn stage7_kernel_eval(eval: &JoltNamedEval) -> stage7::Stage7NamedEval<{field_type}> {{
    stage7::Stage7NamedEval {{
        name: eval.name,
        oracle: eval.oracle,
        value: eval.value,
    }}
}}

pub fn stage7_execution_artifacts(
    artifacts: &stage7::Stage7ExecutionArtifacts<{field_type}>,
) -> JoltStageExecutionArtifacts {{
    JoltStageExecutionArtifacts {{
        challenge_vectors: artifacts
            .challenge_vectors
            .iter()
            .map(|challenge| JoltStageChallengeVector {{
                symbol: challenge.symbol,
                values: challenge.values.clone(),
            }})
            .collect(),
        sumchecks: stage7_proof(artifacts).sumchecks,
        opening_batches: Vec::new(),
    }}
}}

pub fn replay_stage7_proof_with_program<T>(
    program: &'static stage7::Stage7CpuProgramPlan,
    proof: &stage7::Stage7Proof<{field_type}>,
    opening_inputs: &[stage7::Stage7OpeningInputValue<{field_type}>],
    transcript: &mut T,
) -> Result<stage7::Stage7ExecutionArtifacts<{field_type}>, stage7::Stage7KernelError>
where
    T: Transcript<Challenge = {field_type}>,
{{
    let mut executor = stage7::Stage7ProofCarryingKernelExecutor::new(proof, opening_inputs);
    stage7_stage::execute_stage7_prover_with_program(program, &mut executor, transcript)
}}

pub fn stage7_opening_inputs_from_stage6_artifacts(
    artifacts: &stage6::Stage6ExecutionArtifacts<{field_type}>,
) -> Result<Vec<stage7::Stage7OpeningInputValue<{field_type}>>, JoltOpeningInputError> {{
    stage7_opening_inputs_from_stage6_artifacts_with_program(&stage7_stage::STAGE7_PROGRAM, artifacts)
}}

pub fn stage7_opening_inputs_from_stage6_artifacts_with_program(
    program: &'static stage7::Stage7CpuProgramPlan,
    artifacts: &stage6::Stage6ExecutionArtifacts<{field_type}>,
) -> Result<Vec<stage7::Stage7OpeningInputValue<{field_type}>>, JoltOpeningInputError> {{
    program
        .opening_inputs
        .iter()
        .map(|input| {{
            let (point, eval) = stage6_opening_claim(artifacts, input.symbol, input.source_stage, input.source_claim, input.point_arity)?;
            Ok(stage7::Stage7OpeningInputValue {{
                symbol: input.symbol,
                point,
                eval,
            }})
        }})
        .collect()
}}

fn stage6_opening_claim(
    artifacts: &stage6::Stage6ExecutionArtifacts<{field_type}>,
    symbol: &'static str,
    source_stage: &'static str,
    source_claim: &'static str,
    point_arity: usize,
) -> Result<(Vec<{field_type}>, {field_type}), JoltOpeningInputError> {{
    if source_stage != "stage6" {{
        return Err(JoltOpeningInputError::UnsupportedStage7InputSource {{
            symbol,
            source_stage,
        }});
    }}
    let opening = artifacts
        .opening_claims
        .iter()
        .find(|opening| opening.symbol == source_claim)
        .ok_or(JoltOpeningInputError::MissingStage6OpeningClaim {{ source_claim }})?;
    if opening.point.len() != point_arity {{
        return Err(JoltOpeningInputError::InvalidPointLength {{
            symbol,
            expected: point_arity,
            actual: opening.point.len(),
        }});
    }}
    Ok((opening.point.clone(), opening.eval))
}}

fn opening_input_value(
    symbol: &'static str,
    point_arity: usize,
    point: Vec<{field_type}>,
    eval: {field_type},
) -> Result<stage4::Stage4OpeningInputValue<{field_type}>, JoltOpeningInputError> {{
    validate_point_len(symbol, point_arity, point.len())?;
    Ok(stage4::Stage4OpeningInputValue {{
        symbol,
        point,
        eval,
    }})
}}

fn validate_point_len(
    symbol: &'static str,
    expected: usize,
    actual: usize,
) -> Result<(), JoltOpeningInputError> {{
    if actual != expected {{
        return Err(JoltOpeningInputError::InvalidPointLength {{
            symbol,
            expected,
            actual,
        }});
    }}
    Ok(())
}}

fn stage1_opening_claim(
    artifacts: &stage1::Stage1ExecutionArtifacts<{field_type}>,
    source_claim: &'static str,
) -> Result<(Vec<{field_type}>, {field_type}), JoltOpeningInputError> {{
    let opening = artifacts.opening_value(source_claim).ok_or(
        JoltOpeningInputError::MissingOpeningClaim {{
            stage: "stage1",
            source_claim,
        }},
    )?;
    Ok((opening.point.clone(), opening.eval))
}}

fn stage2_opening_claim(
    artifacts: &stage2::Stage2ExecutionArtifacts<{field_type}>,
    source_claim: &'static str,
) -> Result<(Vec<{field_type}>, {field_type}), JoltOpeningInputError> {{
    artifacts
        .opening_claims
        .iter()
        .find(|opening| opening.symbol == source_claim)
        .map(|opening| (opening.point.clone(), opening.eval))
        .ok_or(JoltOpeningInputError::MissingOpeningClaim {{
            stage: "stage2",
            source_claim,
        }})
}}

fn stage3_opening_claim(
    artifacts: &stage3::Stage3ExecutionArtifacts<{field_type}>,
    source_claim: &'static str,
) -> Result<(Vec<{field_type}>, {field_type}), JoltOpeningInputError> {{
    artifacts
        .opening_claims
        .iter()
        .find(|opening| opening.symbol == source_claim)
        .map(|opening| (opening.point.clone(), opening.eval))
        .ok_or(JoltOpeningInputError::MissingOpeningClaim {{
            stage: "stage3",
            source_claim,
        }})
}}

fn stage4_opening_claim(
    artifacts: &stage4::Stage4ExecutionArtifacts<{field_type}>,
    source_claim: &'static str,
) -> Result<(Vec<{field_type}>, {field_type}), JoltOpeningInputError> {{
    artifacts
        .opening_claims
        .iter()
        .find(|opening| opening.symbol == source_claim)
        .map(|opening| (opening.point.clone(), opening.eval))
        .ok_or(JoltOpeningInputError::MissingOpeningClaim {{
            stage: "stage4",
            source_claim,
        }})
}}

fn stage5_opening_claim(
    artifacts: &stage5::Stage5ExecutionArtifacts<{field_type}>,
    source_claim: &'static str,
) -> Result<(Vec<{field_type}>, {field_type}), JoltOpeningInputError> {{
    artifacts
        .opening_claims
        .iter()
        .find(|opening| opening.symbol == source_claim)
        .map(|opening| (opening.point.clone(), opening.eval))
        .ok_or(JoltOpeningInputError::MissingOpeningClaim {{
            stage: "stage5",
            source_claim,
        }})
}}

"#
    )
}

fn jolt_prover_evaluation_helpers(field_type: &str) -> String {
    format!(
        r#"impl stage8_stage::Stage8NamedEvalView<{field_type}> for stage7::Stage7NamedEval<{field_type}> {{
    fn name(&self) -> &'static str {{
        self.name
    }}

    fn value(&self) -> {field_type} {{
        self.value
    }}
}}

impl stage8_stage::Stage8SumcheckOutputView<{field_type}>
    for stage7::Stage7SumcheckOutput<{field_type}>
{{
    type Eval = stage7::Stage7NamedEval<{field_type}>;

    fn point(&self) -> &[{field_type}] {{
        &self.point
    }}

    fn evals(&self) -> &[Self::Eval] {{
        &self.evals
    }}
}}

impl stage8_stage::Stage8OpeningInputView<{field_type}>
    for stage7::Stage7OpeningInputValue<{field_type}>
{{
    fn symbol(&self) -> &'static str {{
        self.symbol
    }}

    fn point(&self) -> &[{field_type}] {{
        &self.point
    }}
}}

impl From<stage8_stage::Stage8EvaluationOpeningPointError> for JoltEvaluationProveError {{
    fn from(error: stage8_stage::Stage8EvaluationOpeningPointError) -> Self {{
        match error {{
            stage8_stage::Stage8EvaluationOpeningPointError::MissingStage7EvaluationPoint => {{
                Self::MissingStage7EvaluationPoint
            }}
            stage8_stage::Stage8EvaluationOpeningPointError::InvalidPointLength {{
                artifact,
                expected,
                actual,
            }} => Self::InvalidPointLength {{
                artifact,
                expected,
                actual,
            }},
        }}
    }}
}}

pub fn prove_jolt_evaluation_proof<I, T>(
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
    let _claims_span = tracing::info_span!("bolt.evaluate.claims").entered();
    let (sumcheck_address_point, stage7_values) =
        stage8_stage::stage7_claim_values(program, &stage7.sumchecks)
            .ok_or(JoltEvaluationProveError::MissingStage7RaEval)?;
    let address_point = stage8_stage::reverse_point(&sumcheck_address_point);
    let (opening_point, log_t) =
        stage8_stage::stage7_evaluation_opening_point(program, &address_point, stage7_openings)?;
    let lagrange_factor = EqPolynomial::<{field_type}>::zero_selector(&address_point);
    let claims =
        stage8_stage::evaluation_claims(program, &stage6.sumchecks, &stage7_values, lagrange_factor)
            .map_err(|error| JoltEvaluationProveError::MissingStageEval {{
                stage: error.stage,
                eval: error.eval,
            }})?;
    drop(_claims_span);

    let _rlc_span = tracing::info_span!("bolt.evaluate.rlc_claims").entered();
    stage8_stage::append_rlc_claims(transcript, &claims);
    let gamma_powers = stage8_stage::gamma_powers(transcript, claims.len());
    let joint_claim = claims
        .iter()
        .zip(&gamma_powers)
        .map(|(claim, gamma)| claim.value * *gamma)
        .sum();
    drop(_rlc_span);
    let _materialize_span =
        tracing::info_span!("bolt.evaluate.materialize_joint_polynomial").entered();
    let joint_evals = materialize_joint_polynomial(
        commitment_inputs,
        &claims,
        &gamma_powers,
        log_t,
        opening_point.len(),
    )?;
    drop(_materialize_span);
    let joint_poly = Polynomial::new(joint_evals);
    let _hint_span = tracing::info_span!("bolt.evaluate.joint_opening_hint").entered();
    let joint_hint = joint_opening_hint(commitments, &claims, &gamma_powers)?;
    drop(_hint_span);
    let _dory_open_span = tracing::info_span!("bolt.evaluate.dory_open").entered();
    let joint_opening_proof = <jolt_dory::DoryScheme as CommitmentScheme>::open(
        &joint_poly,
        &opening_point,
        joint_claim,
        prover_setup,
        Some(joint_hint),
        transcript,
    );
    drop(_dory_open_span);
    let _bind_span = tracing::info_span!("bolt.evaluate.bind_opening_inputs").entered();
    <jolt_dory::DoryScheme as CommitmentScheme>::bind_opening_inputs(
        transcript,
        &opening_point,
        &joint_claim,
    );
    drop(_bind_span);
    Ok(JoltEvaluationProof {{ joint_opening_proof }})
}}

fn materialize_joint_polynomial<I>(
    commitment_inputs: &mut I,
    claims: &[stage8_stage::Stage8EvaluationClaim<{field_type}>],
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
        if claim.source_stage == stage8_stage::Stage8SourceStage::Stage6 {{
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
    claims: &[stage8_stage::Stage8EvaluationClaim<{field_type}>],
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

    Ok(DoryScheme::combine_hint_refs(&hints, &scalars))
}}

fn opening_hint_for_oracle<'a>(
    commitments: &'a commitment_stage::CommitmentArtifacts,
    oracle: &'static str,
) -> Result<&'a DoryHint, JoltEvaluationProveError> {{
    commitments
        .hints
        .iter()
        .find(|hint| hint.oracle == oracle)
        .map(|hint| &hint.hint)
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
    "use jolt_profiling",
    "jolt_profiling::",
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
    "use jolt_profiling",
    "jolt_profiling::",
    "tracer::",
];
