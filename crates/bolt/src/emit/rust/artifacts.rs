use std::path::{Component, Path};

use crate::ir::Role;

use super::{EmitError, RustSourceFile};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProtocolArtifactConfig {
    pub protocol_name: String,
    pub type_prefix: String,
    pub transcript_label: String,
    pub prover_crate_name: String,
    pub verifier_crate_name: String,
    pub common_dependencies: Vec<String>,
    pub prover_dependencies: Vec<String>,
    pub verifier_dependencies: Vec<String>,
    pub prover_forbidden_imports: Vec<String>,
    pub verifier_forbidden_imports: Vec<String>,
    pub kernel_crate: Option<ProtocolCrateRef>,
    pub field_type: RustTypeRef,
    pub default_transcript_type: RustTypeRef,
    pub transcript_trait: RustTypeRef,
    pub sumcheck_proof_type: RustTypeRef,
    pub commitment_type: RustTypeRef,
    pub prover_setup_type: RustTypeRef,
}

impl ProtocolArtifactConfig {
    fn protocol_snake(&self) -> String {
        snake_case(&self.protocol_name)
    }

    fn crate_name(&self, role: &Role) -> &str {
        match role {
            Role::Prover => &self.prover_crate_name,
            Role::Verifier => &self.verifier_crate_name,
        }
    }

    fn dependencies(&self, role: &Role) -> Vec<String> {
        let mut dependencies = self.common_dependencies.clone();
        match role {
            Role::Prover => {
                dependencies.extend(self.prover_dependencies.clone());
                if !dependencies.contains(&self.verifier_crate_name) {
                    dependencies.push(self.verifier_crate_name.clone());
                }
            }
            Role::Verifier => dependencies.extend(self.verifier_dependencies.clone()),
        }
        dependencies.sort();
        dependencies.dedup();
        dependencies
    }

    fn forbidden_imports(&self, role: &Role) -> &[String] {
        match role {
            Role::Prover => &self.prover_forbidden_imports,
            Role::Verifier => &self.verifier_forbidden_imports,
        }
    }

    fn verifier_crate_import(&self) -> String {
        rust_crate_ident(&self.verifier_crate_name)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProtocolCrateRef {
    pub package: String,
    pub import: String,
}

impl ProtocolCrateRef {
    pub fn new(package: impl Into<String>, import: impl Into<String>) -> Self {
        Self {
            package: package.into(),
            import: import.into(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RustTypeRef {
    pub path: String,
}

impl RustTypeRef {
    pub fn new(path: impl Into<String>) -> Self {
        Self { path: path.into() }
    }

    fn ident(&self) -> &str {
        self.path.rsplit("::").next().unwrap_or(&self.path)
    }

    fn use_line(&self) -> String {
        format!("use {};\n", self.path)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProtocolStageKind {
    Commitment,
    Proof,
    Evaluation,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProtocolStage {
    name: String,
    module_name: String,
    ordinal: usize,
    kind: ProtocolStageKind,
}

impl ProtocolStage {
    pub fn new(
        name: impl Into<String>,
        module_name: impl Into<String>,
        ordinal: usize,
        kind: ProtocolStageKind,
    ) -> Self {
        Self {
            name: name.into(),
            module_name: module_name.into(),
            ordinal,
            kind,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn module_name(&self) -> &str {
        &self.module_name
    }

    pub fn order(&self) -> usize {
        self.ordinal
    }

    pub fn is_commitment(&self) -> bool {
        self.kind == ProtocolStageKind::Commitment
    }

    pub fn is_proof(&self) -> bool {
        self.kind == ProtocolStageKind::Proof
    }

    pub fn is_evaluation(&self) -> bool {
        self.kind == ProtocolStageKind::Evaluation
    }
}

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
pub enum ArtifactCrateRole {
    Prover,
    Verifier,
}

impl ArtifactCrateRole {
    pub fn for_role(role: &Role) -> Self {
        match role {
            Role::Prover => Self::Prover,
            Role::Verifier => Self::Verifier,
        }
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProtocolRustArtifact {
    pub role: Role,
    pub stage: ProtocolStage,
    pub crate_name: String,
    pub path: String,
    pub source: RustSourceFile,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GeneratedCrate {
    pub crate_name: String,
    pub files: Vec<GeneratedFile>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GeneratedFile {
    pub path: String,
    pub source: String,
}

pub type JoltRustArtifact = ProtocolRustArtifact;
pub type JoltGeneratedCrate = GeneratedCrate;
pub type JoltGeneratedFile = GeneratedFile;

impl GeneratedCrate {
    pub fn write_to(&self, output_root: impl AsRef<Path>) -> Result<(), EmitError> {
        let crate_root = output_root.as_ref().join(&self.crate_name);
        for file in &self.files {
            let path = generated_file_path(&crate_root, &file.path)?;
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).map_err(|error| {
                    EmitError::new(format!(
                        "failed to create generated crate directory `{}`: {error}",
                        parent.display()
                    ))
                })?;
            }
            std::fs::write(&path, &file.source).map_err(|error| {
                EmitError::new(format!(
                    "failed to write generated crate file `{}`: {error}",
                    path.display()
                ))
            })?;
        }
        Ok(())
    }
}

pub fn write_generated_crates(
    generated_crates: &[GeneratedCrate],
    output_root: impl AsRef<Path>,
) -> Result<(), EmitError> {
    for generated_crate in generated_crates {
        generated_crate.write_to(output_root.as_ref())?;
    }
    Ok(())
}

pub fn write_jolt_generated_crates(
    generated_crates: &[GeneratedCrate],
    output_root: impl AsRef<Path>,
) -> Result<(), EmitError> {
    write_generated_crates(generated_crates, output_root)
}

pub fn protocol_rust_artifact(
    config: &ProtocolArtifactConfig,
    stage: ProtocolStage,
    role: Role,
    source: RustSourceFile,
) -> ProtocolRustArtifact {
    let crate_name = config.crate_name(&role).to_owned();
    let path = format!("{crate_name}/src/stages/{}.rs", stage.module_name());
    ProtocolRustArtifact {
        role,
        stage,
        crate_name,
        path,
        source,
    }
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

pub fn validate_rust_artifact_imports(
    config: &ProtocolArtifactConfig,
    artifact: &ProtocolRustArtifact,
) -> Result<(), EmitError> {
    for import in config.forbidden_imports(&artifact.role) {
        if artifact.source.source.contains(import) {
            return Err(EmitError::new(format!(
                "{} artifact `{}` for {} imports forbidden `{import}`",
                artifact.crate_name,
                artifact.path,
                artifact.stage.name()
            )));
        }
    }
    Ok(())
}

pub fn validate_jolt_rust_artifact_imports(
    artifact: &ProtocolRustArtifact,
) -> Result<(), EmitError> {
    validate_rust_artifact_imports(&jolt_artifact_config(), artifact)
}

pub fn assemble_generated_crates(
    config: &ProtocolArtifactConfig,
    artifacts: Vec<ProtocolRustArtifact>,
    dependency_root: &str,
) -> Result<Vec<GeneratedCrate>, EmitError> {
    assemble_generated_crates_with_manifest(
        config,
        artifacts,
        ManifestMode::Standalone { dependency_root },
    )
}

pub fn assemble_workspace_generated_crates(
    config: &ProtocolArtifactConfig,
    artifacts: Vec<ProtocolRustArtifact>,
) -> Result<Vec<GeneratedCrate>, EmitError> {
    assemble_generated_crates_with_manifest(config, artifacts, ManifestMode::Workspace)
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

fn assemble_generated_crates_with_manifest(
    config: &ProtocolArtifactConfig,
    artifacts: Vec<ProtocolRustArtifact>,
    manifest_mode: ManifestMode<'_>,
) -> Result<Vec<GeneratedCrate>, EmitError> {
    let mut prover = Vec::new();
    let mut verifier = Vec::new();
    for artifact in artifacts {
        validate_rust_artifact_imports(config, &artifact)?;
        match artifact.role {
            Role::Prover => prover.push(artifact),
            Role::Verifier => verifier.push(artifact),
        }
    }
    Ok(vec![
        generated_crate(config, Role::Prover, prover, manifest_mode),
        generated_crate(config, Role::Verifier, verifier, manifest_mode),
    ])
}

fn generated_crate(
    config: &ProtocolArtifactConfig,
    role: Role,
    mut artifacts: Vec<ProtocolRustArtifact>,
    manifest_mode: ManifestMode<'_>,
) -> GeneratedCrate {
    artifacts.sort_by_key(|artifact| artifact.stage.order());
    let crate_name = config.crate_name(&role).to_owned();
    let stage_modules = artifacts
        .iter()
        .map(|artifact| {
            format!(
                "#[rustfmt::skip]\npub mod {};",
                artifact.stage.module_name()
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    let mut files = vec![
        GeneratedFile {
            path: "Cargo.toml".to_owned(),
            source: generated_manifest(config, &role, manifest_mode),
        },
        GeneratedFile {
            path: "src/lib.rs".to_owned(),
            source: generated_lib(config, &role, &artifacts),
        },
        generated_role_api_file(config, &role, &artifacts),
        GeneratedFile {
            path: "src/stages/mod.rs".to_owned(),
            source: format!("{stage_modules}\n"),
        },
    ];
    files.extend(artifacts.into_iter().map(|artifact| GeneratedFile {
        path: format!("src/stages/{}.rs", artifact.stage.module_name()),
        source: artifact.source.source,
    }));
    GeneratedCrate { crate_name, files }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ManifestMode<'a> {
    Standalone { dependency_root: &'a str },
    Workspace,
}

fn generated_manifest(
    config: &ProtocolArtifactConfig,
    role: &Role,
    manifest_mode: ManifestMode<'_>,
) -> String {
    let crate_name = config.crate_name(role);
    let dependencies = config.dependencies(role);
    match manifest_mode {
        ManifestMode::Standalone { dependency_root } => {
            let dependencies = dependencies
                .into_iter()
                .map(|name| format!("{name} = {{ path = \"{dependency_root}/{name}\" }}"))
                .collect::<Vec<_>>()
                .join("\n");
            format!(
                "[package]\nname = \"{crate_name}\"\nversion = \"0.0.0\"\nedition = \"2021\"\n\n[patch.crates-io]\nark-bn254 = {{ git = \"https://github.com/a16z/arkworks-algebra\", branch = \"dev/twist-shout\" }}\nark-ec = {{ git = \"https://github.com/a16z/arkworks-algebra\", branch = \"dev/twist-shout\" }}\nark-ff = {{ git = \"https://github.com/a16z/arkworks-algebra\", branch = \"dev/twist-shout\" }}\nark-serialize = {{ git = \"https://github.com/a16z/arkworks-algebra\", branch = \"dev/twist-shout\" }}\n\n[dependencies]\n{dependencies}\n"
            )
        }
        ManifestMode::Workspace => {
            let dependencies = dependencies
                .into_iter()
                .map(|name| format!("{name}.workspace = true"))
                .collect::<Vec<_>>()
                .join("\n");
            let role_name = match role {
                Role::Prover => "prover",
                Role::Verifier => "verifier",
            };
            format!(
                "[package]\nname = \"{crate_name}\"\nversion = \"0.0.0\"\nedition = \"2021\"\nlicense = \"MIT OR Apache-2.0\"\ndescription = \"Bolt-generated {} {role_name} role crate\"\nrepository = \"https://github.com/a16z/jolt\"\n\n[lints]\nworkspace = true\n\n[dependencies]\n{dependencies}\n",
                config.protocol_name
            )
        }
    }
}

fn generated_lib(
    config: &ProtocolArtifactConfig,
    role: &Role,
    artifacts: &[ProtocolRustArtifact],
) -> String {
    let protocol_snake = config.protocol_snake();
    let prefix = &config.type_prefix;
    let stage_apis = stage_apis(config, artifacts);
    let commitment_api = commitment_api(artifacts);
    let supports_evaluation =
        supports_jolt_evaluation(config, &stage_apis, &commitment_api, artifacts);
    let role_module = match (role, supports_evaluation) {
        (Role::Prover, true) => format!(
            "#[rustfmt::skip]\npub mod prover;\npub mod stages;\n\npub use prover::{{\n    default_prover_programs, prove_{protocol_snake}, prove_{protocol_snake}_evaluation_proof, prove_{protocol_snake}_with_programs,\n    Default{prefix}Transcript, {prefix}EvaluationProveError, {prefix}ProveError, {prefix}ProverArtifacts,\n    {prefix}ProverInputs, {prefix}ProverPrograms,\n}};"
        ),
        (Role::Prover, false) => format!(
            "#[rustfmt::skip]\npub mod prover;\npub mod stages;\n\npub use prover::{{\n    default_prover_programs, prove_{protocol_snake}, prove_{protocol_snake}_with_programs,\n    Default{prefix}Transcript, {prefix}ProveError, {prefix}ProverArtifacts, {prefix}ProverInputs,\n    {prefix}ProverPrograms,\n}};"
        ),
        (Role::Verifier, true) => format!(
            "pub mod stages;\n#[rustfmt::skip]\npub mod verifier;\n\npub use verifier::{{\n    default_verifier_programs, verify_{protocol_snake}, verify_{protocol_snake}_evaluation_proof,\n    verify_{protocol_snake}_with_programs, {prefix}EvaluationProof, {prefix}EvaluationProofError, {prefix}NamedEval,\n    {prefix}Proof, {prefix}StageProof, {prefix}SumcheckOutput, {prefix}VerificationArtifacts, {prefix}VerifierInputs,\n    {prefix}VerifierPrograms, {prefix}VerifyError,\n}};"
        ),
        (Role::Verifier, false) => format!(
            "pub mod stages;\n#[rustfmt::skip]\npub mod verifier;\n\npub use verifier::{{\n    default_verifier_programs, verify_{protocol_snake}, verify_{protocol_snake}_with_programs, {prefix}NamedEval, {prefix}Proof,\n    {prefix}StageProof, {prefix}SumcheckOutput, {prefix}VerificationArtifacts, {prefix}VerifierInputs,\n    {prefix}VerifierPrograms, {prefix}VerifyError,\n}};"
        ),
    };
    let stages = artifacts
        .iter()
        .map(|artifact| {
            format!(
                "    GeneratedStage {{\n        name: \"{}\",\n        module: \"{}\",\n        ordinal: {},\n    }},",
                artifact.stage.name(),
                artifact.stage.module_name(),
                artifact.stage.order()
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "{role_module}\n\npub const TRANSCRIPT_LABEL: &[u8] = {};\n\n#[derive(Clone, Copy, Debug, PartialEq, Eq)]\npub struct GeneratedStage {{\n    pub name: &'static str,\n    pub module: &'static str,\n    pub ordinal: usize,\n}}\n\npub const GENERATED_STAGES: &[GeneratedStage] = &[\n{stages}\n];\n\npub fn generated_stage_names() -> impl Iterator<Item = &'static str> {{\n    GENERATED_STAGES.iter().map(|stage| stage.name)\n}}\n",
        byte_string_literal(&config.transcript_label)
    )
}

fn generated_role_api_file(
    config: &ProtocolArtifactConfig,
    role: &Role,
    artifacts: &[ProtocolRustArtifact],
) -> GeneratedFile {
    match role {
        Role::Prover => GeneratedFile {
            path: "src/prover.rs".to_owned(),
            source: generated_prover_api(config, artifacts),
        },
        Role::Verifier => GeneratedFile {
            path: "src/verifier.rs".to_owned(),
            source: generated_verifier_api(config, artifacts),
        },
    }
}

#[derive(Clone, Debug)]
struct StageRustApi {
    field_name: String,
    module_alias: String,
    variant_name: String,
    proof_type: String,
    output_type: String,
    eval_type: String,
    artifacts_type: String,
    error_type: String,
    verifier_fn: Option<String>,
    with_program_verifier_fn: Option<String>,
    program_type: Option<String>,
    program_const: Option<String>,
    prover_fn: Option<String>,
    with_program_prover_fn: Option<String>,
    kernel_module: Option<String>,
    opening_input_type: Option<String>,
    ram_data_type: Option<String>,
    verifier_data_type: Option<String>,
}

#[derive(Clone, Debug)]
struct CommitmentRustApi {
    field_name: String,
    module_alias: String,
    variant_name: String,
    artifacts_type: String,
    error_type: String,
    verifier_fn: Option<String>,
    with_program_verifier_fn: Option<String>,
    program_type: Option<String>,
    program_const: Option<String>,
    prover_fn: Option<String>,
    with_program_prover_fn: Option<String>,
    input_provider_trait: Option<String>,
}

fn generated_verifier_api(
    config: &ProtocolArtifactConfig,
    artifacts: &[ProtocolRustArtifact],
) -> String {
    let stages = stage_apis(config, artifacts);
    let modules = role_modules(artifacts);
    let commitment = commitment_api(artifacts);
    let supports_evaluation = supports_jolt_evaluation(config, &stages, &commitment, artifacts);
    let prefix = &config.type_prefix;
    let protocol_snake = config.protocol_snake();
    let field_type = config.field_type.ident();
    let sumcheck_proof_type = config.sumcheck_proof_type.ident();
    let transcript_trait = config.transcript_trait.ident();
    let commitment_type = config.commitment_type.ident();
    let named_eval_type = format!("{prefix}NamedEval");
    let sumcheck_output_type = format!("{prefix}SumcheckOutput");
    let stage_proof_type = format!("{prefix}StageProof");
    let proof_type = format!("{prefix}Proof");
    let verifier_inputs_type = format!("{prefix}VerifierInputs");
    let verifier_programs_type = format!("{prefix}VerifierPrograms");
    let verification_artifacts_type = format!("{prefix}VerificationArtifacts");
    let verify_error_type = format!("{prefix}VerifyError");

    let mut source = String::new();
    if supports_evaluation {
        source.push_str("use std::collections::BTreeMap;\n\n");
        source.push_str(
            "use jolt_dory::{DoryCommitment, DoryProof, DoryScheme, DoryVerifierSetup};\n",
        );
        source.push_str("use jolt_field::{Field, Fr};\n");
        source.push_str(
            "use jolt_openings::{AdditivelyHomomorphic, CommitmentScheme, OpeningsError};\n",
        );
        source.push_str("use jolt_poly::EqPolynomial;\n");
        if !stages.is_empty() {
            source.push_str(&config.sumcheck_proof_type.use_line());
        }
        source.push_str("use jolt_transcript::{AppendToTranscript, LabelWithCount, Transcript};\n");
    } else {
        if commitment.is_some() {
            source.push_str(&config.commitment_type.use_line());
        }
        source.push_str(&config.field_type.use_line());
        if !stages.is_empty() {
            source.push_str(&config.sumcheck_proof_type.use_line());
        }
        source.push_str(&config.transcript_trait.use_line());
    }
    source.push('\n');
    if !modules.is_empty() {
        source.push_str(&format!(
            "use crate::stages::{{{}}};\n\n",
            aliased_modules(&modules).join(", ")
        ));
    }

    source.push_str(&format!(
        "#[derive(Clone, Debug)]\npub struct {named_eval_type} {{\n    pub name: &'static str,\n    pub oracle: &'static str,\n    pub value: {field_type},\n}}\n\n",
    ));
    source.push_str(&format!(
        "#[derive(Clone, Debug)]\npub struct {sumcheck_output_type} {{\n    pub driver: &'static str,\n    pub point: Vec<{field_type}>,\n    pub evals: Vec<{named_eval_type}>,\n    pub proof: {sumcheck_proof_type}<{field_type}>,\n}}\n\n",
    ));
    source.push_str(&format!(
        "#[derive(Clone, Debug, Default)]\npub struct {stage_proof_type} {{\n    pub sumchecks: Vec<{sumcheck_output_type}>,\n}}\n\n",
    ));
    source.push_str(&format!(
        "#[derive(Clone, Debug)]\npub struct {proof_type} {{\n"
    ));
    if commitment.is_some() {
        source.push_str(&format!(
            "    pub commitments: Vec<Option<{commitment_type}>>,\n"
        ));
    }
    for stage in &stages {
        source.push_str(&format!(
            "    pub {}: {stage_proof_type},\n",
            stage.field_name
        ));
    }
    if supports_evaluation {
        source.push_str(&format!(
            "    pub evaluation: Option<{prefix}EvaluationProof>,\n"
        ));
    }
    source.push_str("}\n\n");

    if supports_evaluation {
        source.push_str(&format!(
            "#[derive(Clone, Debug)]\npub struct {prefix}EvaluationProof {{\n    pub joint_opening_proof: DoryProof,\n}}\n\n",
        ));
    }

    let verifier_inputs_derive = if supports_evaluation {
        "#[derive(Clone, Copy)]"
    } else {
        "#[derive(Clone, Copy, Debug)]"
    };
    source.push_str(&format!(
        "{verifier_inputs_derive}\npub struct {verifier_inputs_type}<'a> {{\n"
    ));
    for stage in &stages {
        if let Some(opening_type) = &stage.opening_input_type {
            source.push_str(&format!(
                "    pub {}_openings: &'a [{}::{}<{field_type}>],\n",
                stage.field_name, stage.module_alias, opening_type
            ));
        }
        if let Some(ram_type) = &stage.ram_data_type {
            source.push_str(&format!(
                "    pub {}_ram: Option<&'a {}::{}<'a>>,\n",
                stage.field_name, stage.module_alias, ram_type
            ));
        }
        if let Some(data_type) = &stage.verifier_data_type {
            source.push_str(&format!(
                "    pub {}_data: Option<&'a {}::{}>,\n",
                stage.field_name, stage.module_alias, data_type
            ));
        }
    }
    if supports_evaluation {
        source.push_str("    pub evaluation_setup: Option<&'a DoryVerifierSetup>,\n");
    }
    source.push_str("}\n\n");

    source.push_str(&format!(
        "#[derive(Clone, Copy, Debug)]\npub struct {verifier_programs_type} {{\n"
    ));
    if let Some(commitment) = &commitment {
        if let (Some(program_type), Some(_), Some(_)) = (
            &commitment.program_type,
            &commitment.program_const,
            &commitment.with_program_verifier_fn,
        ) {
            source.push_str(&format!(
                "    pub {}: &'static {}::{},\n",
                commitment.field_name, commitment.module_alias, program_type
            ));
        }
    }
    for stage in &stages {
        if let (Some(program_type), Some(_), Some(_)) = (
            &stage.program_type,
            &stage.program_const,
            &stage.with_program_verifier_fn,
        ) {
            source.push_str(&format!(
                "    pub {}: &'static {}::{},\n",
                stage.field_name, stage.module_alias, program_type
            ));
        }
    }
    if supports_evaluation {
        source.push_str("    pub stage8: &'static stage8_stage::Stage8EvaluationProgramPlan,\n");
    }
    source.push_str("}\n\n");

    source.push_str(&format!(
        "pub fn default_verifier_programs() -> {verifier_programs_type} {{\n    {verifier_programs_type} {{\n"
    ));
    if let Some(commitment) = &commitment {
        if let (Some(_), Some(program_const), Some(_)) = (
            &commitment.program_type,
            &commitment.program_const,
            &commitment.with_program_verifier_fn,
        ) {
            source.push_str(&format!(
                "        {}: &{}::{},\n",
                commitment.field_name, commitment.module_alias, program_const
            ));
        }
    }
    for stage in &stages {
        if let (Some(_), Some(program_const), Some(_)) = (
            &stage.program_type,
            &stage.program_const,
            &stage.with_program_verifier_fn,
        ) {
            source.push_str(&format!(
                "        {}: &{}::{},\n",
                stage.field_name, stage.module_alias, program_const
            ));
        }
    }
    if supports_evaluation {
        source.push_str("        stage8: &stage8_stage::STAGE8_PROGRAM,\n");
    }
    source.push_str("    }\n}\n\n");

    source.push_str(&format!(
        "#[derive(Clone, Debug)]\npub struct {verification_artifacts_type} {{\n"
    ));
    if let Some(commitment) = &commitment {
        source.push_str(&format!(
            "    pub {}: {}::{},\n",
            commitment.field_name, commitment.module_alias, commitment.artifacts_type
        ));
    }
    for stage in &stages {
        source.push_str(&format!(
            "    pub {}: {}::{}<{field_type}>,\n",
            stage.field_name, stage.module_alias, stage.artifacts_type
        ));
    }
    source.push_str("}\n\n");

    source.push_str(&format!(
        "#[derive(Debug)]\npub enum {verify_error_type} {{\n"
    ));
    if let Some(commitment) = &commitment {
        source.push_str(&format!(
            "    {}({}::{}),\n",
            commitment.variant_name, commitment.module_alias, commitment.error_type
        ));
    }
    for stage in &stages {
        source.push_str(&format!(
            "    {}({}::{}),\n",
            stage.variant_name, stage.module_alias, stage.error_type
        ));
    }
    if supports_evaluation {
        source.push_str(&format!("    Evaluation({prefix}EvaluationProofError),\n"));
    }
    source.push_str("}\n\n");

    if supports_evaluation {
        source.push_str(&format!(
            "#[derive(Debug)]\npub enum {prefix}EvaluationProofError {{\n    MissingProof,\n    MissingVerifierSetup,\n    MissingStageEval {{ stage: &'static str, eval: &'static str }},\n    MissingStage7RaEval,\n    MissingStage7EvaluationPoint,\n    MissingCommitment {{ oracle: &'static str }},\n    InvalidPointLength {{\n        artifact: &'static str,\n        expected: usize,\n        actual: usize,\n    }},\n    Opening(OpeningsError),\n}}\n\n",
        ));
    }

    if let Some(commitment) = &commitment {
        source.push_str(&format!(
            "impl From<{module}::{error}> for {verify_error_type} {{\n    fn from(error: {module}::{error}) -> Self {{\n        Self::{variant}(error)\n    }}\n}}\n\n",
            module = commitment.module_alias,
            error = commitment.error_type,
            variant = commitment.variant_name,
        ));
    }
    for stage in &stages {
        source.push_str(&format!(
            "impl From<{}::{}> for {verify_error_type} {{\n    fn from(error: {}::{}) -> Self {{\n        Self::{}(error)\n    }}\n}}\n\n",
            stage.module_alias,
            stage.error_type,
            stage.module_alias,
            stage.error_type,
            stage.variant_name
        ));
    }
    if supports_evaluation {
        source.push_str(&format!(
            "impl From<{prefix}EvaluationProofError> for {verify_error_type} {{\n    fn from(error: {prefix}EvaluationProofError) -> Self {{\n        Self::Evaluation(error)\n    }}\n}}\n\nimpl From<OpeningsError> for {prefix}EvaluationProofError {{\n    fn from(error: OpeningsError) -> Self {{\n        Self::Opening(error)\n    }}\n}}\n\n",
        ));
    }

    source.push_str(&format!(
        "pub fn verify_{protocol_snake}<T>(\n    proof: &{proof_type},\n    inputs: {verifier_inputs_type}<'_>,\n    transcript: &mut T,\n) -> Result<{verification_artifacts_type}, {verify_error_type}>\nwhere\n    T: {transcript_trait}<Challenge = {field_type}>,\n{{\n",
    ));
    source.push_str(&format!(
        "    verify_{protocol_snake}_with_programs(proof, inputs, default_verifier_programs(), transcript)\n}}\n\n"
    ));
    source.push_str(&format!(
        "pub fn verify_{protocol_snake}_with_programs<T>(\n    proof: &{proof_type},\n    inputs: {verifier_inputs_type}<'_>,\n    programs: {verifier_programs_type},\n    transcript: &mut T,\n) -> Result<{verification_artifacts_type}, {verify_error_type}>\nwhere\n    T: {transcript_trait}<Challenge = {field_type}>,\n{{\n",
    ));
    if let Some(commitment) = &commitment {
        let verifier_fn = commitment
            .with_program_verifier_fn
            .as_deref()
            .or(commitment.verifier_fn.as_deref())
            .unwrap_or("missing_commitment_verifier_function");
        let program_arg = if commitment.with_program_verifier_fn.is_some()
            && commitment.program_type.is_some()
            && commitment.program_const.is_some()
        {
            format!("programs.{}, ", commitment.field_name)
        } else {
            String::new()
        };
        source.push_str(&format!(
            "    let {field} = {module}::{verifier_fn}({program_arg}&proof.commitments, transcript)?;\n",
            field = commitment.field_name,
            module = commitment.module_alias,
        ));
    }
    for stage in &stages {
        source.push_str(&format!(
            "    let {}_proof = {}_proof(&proof.{});\n",
            stage.field_name, stage.field_name, stage.field_name
        ));
    }
    source.push('\n');
    for stage in &stages {
        let verifier_fn = stage
            .with_program_verifier_fn
            .as_deref()
            .or(stage.verifier_fn.as_deref())
            .unwrap_or("missing_verifier_function");
        let mut args = vec![format!("&{}_proof", stage.field_name)];
        if stage.with_program_verifier_fn.is_some()
            && stage.program_type.is_some()
            && stage.program_const.is_some()
        {
            args.insert(0, format!("programs.{}", stage.field_name));
        }
        if stage.opening_input_type.is_some() {
            args.push(format!("inputs.{}_openings", stage.field_name));
        }
        if stage.ram_data_type.is_some() {
            args.push(format!("inputs.{}_ram", stage.field_name));
        }
        if stage.verifier_data_type.is_some() {
            args.push(format!("inputs.{}_data", stage.field_name));
        }
        args.push("transcript".to_owned());
        source.push_str(&format!(
            "    let {} = {}::{}({})?;\n",
            stage.field_name,
            stage.module_alias,
            verifier_fn,
            args.join(", ")
        ));
    }
    if supports_evaluation {
        source.push_str(
            "    match (&proof.evaluation, inputs.evaluation_setup) {\n        (Some(evaluation), Some(setup)) => {\n            verify_jolt_evaluation_proof(\n                programs.stage8,\n                evaluation,\n                &commitment,\n                &proof.stage6,\n                &proof.stage7,\n                inputs.stage7_openings,\n                setup,\n                transcript,\n            )?;\n        }\n        (Some(_), None) => return Err(JoltEvaluationProofError::MissingVerifierSetup.into()),\n        (None, Some(_)) => return Err(JoltEvaluationProofError::MissingProof.into()),\n        (None, None) => {}\n    }\n",
        );
    }
    source.push_str(&format!("\n    Ok({verification_artifacts_type} {{\n"));
    if let Some(commitment) = &commitment {
        source.push_str(&format!("        {},\n", commitment.field_name));
    }
    for stage in &stages {
        source.push_str(&format!("        {},\n", stage.field_name));
    }
    source.push_str("    })\n}\n\n");

    if supports_evaluation {
        source.push_str(&jolt_verifier_evaluation_helpers(prefix, field_type));
    }

    for stage in &stages {
        source.push_str(&format!(
            "fn {field}_proof(proof: &{stage_proof_type}) -> {module}::{proof_ty}<{field_type}> {{\n    {module}::{proof_ty} {{\n        sumchecks: proof.sumchecks.iter().map({field}_sumcheck).collect(),\n    }}\n}}\n\n",
            field = stage.field_name,
            module = stage.module_alias,
            proof_ty = stage.proof_type
        ));
        source.push_str(&format!(
            "fn {field}_sumcheck(output: &{sumcheck_output_type}) -> {module}::{output_ty}<{field_type}> {{\n    {module}::{output_ty} {{\n        driver: output.driver,\n        point: output.point.clone(),\n        evals: output.evals.iter().map({field}_eval).collect(),\n        proof: output.proof.clone(),\n    }}\n}}\n\n",
            field = stage.field_name,
            module = stage.module_alias,
            output_ty = stage.output_type
        ));
        source.push_str(&format!(
            "fn {field}_eval(eval: &{named_eval_type}) -> {module}::{eval_ty}<{field_type}> {{\n    {module}::{eval_ty} {{\n        name: eval.name,\n        oracle: eval.oracle,\n        value: eval.value,\n    }}\n}}\n\n",
            field = stage.field_name,
            module = stage.module_alias,
            eval_ty = stage.eval_type
        ));
    }
    source
}

fn generated_prover_api(
    config: &ProtocolArtifactConfig,
    artifacts: &[ProtocolRustArtifact],
) -> String {
    let stages = stage_apis(config, artifacts);
    let modules = role_modules(artifacts);
    let kernel_modules = unique_kernel_modules(&stages);
    let commitment = commitment_api(artifacts);
    let has_commitment = commitment.is_some();
    let supports_evaluation = supports_jolt_evaluation(config, &stages, &commitment, artifacts);
    let generic_params = prover_generic_params(&stages, has_commitment);
    let prefix = &config.type_prefix;
    let protocol_snake = config.protocol_snake();
    let field_type = config.field_type.ident();
    let default_transcript_type = config.default_transcript_type.ident();
    let transcript_trait = config.transcript_trait.ident();
    let prover_setup_type = config.prover_setup_type.ident();
    let verifier_import = config.verifier_crate_import();
    let named_eval_type = format!("{prefix}NamedEval");
    let sumcheck_output_type = format!("{prefix}SumcheckOutput");
    let stage_proof_type = format!("{prefix}StageProof");
    let proof_type = format!("{prefix}Proof");
    let prover_inputs_type = format!("{prefix}ProverInputs");
    let prover_programs_type = format!("{prefix}ProverPrograms");
    let prover_artifacts_type = format!("{prefix}ProverArtifacts");
    let prove_error_type = format!("{prefix}ProveError");
    let default_transcript_alias = format!("Default{prefix}Transcript");

    let mut source = String::new();
    if supports_evaluation {
        source.push_str("use jolt_dory::{DoryHint, DoryProverSetup, DoryScheme};\n");
        source.push_str("use jolt_field::{Field, Fr};\n");
        if !kernel_modules.is_empty() {
            let kernel_crate = config
                .kernel_crate
                .as_ref()
                .map(|kernel_crate| kernel_crate.import.as_str())
                .unwrap_or("jolt_kernels");
            source.push_str(&format!(
                "use {kernel_crate}::{{{}}};\n",
                kernel_modules.join(", ")
            ));
        }
        source.push_str("use jolt_openings::{AdditivelyHomomorphic, CommitmentScheme};\n");
        source.push_str("use jolt_poly::{EqPolynomial, Polynomial};\n");
        source.push_str(
            "use jolt_transcript::{AppendToTranscript, Blake2bTranscript, LabelWithCount, Transcript};\n",
        );
        source.push_str(&format!(
            "use {verifier_import}::{{JoltEvaluationProof, {named_eval_type}, {proof_type}, {stage_proof_type}, {sumcheck_output_type}}};\n\n",
        ));
    } else {
        if has_commitment {
            source.push_str(&config.prover_setup_type.use_line());
        }
        source.push_str(&config.field_type.use_line());
        if !kernel_modules.is_empty() {
            let kernel_crate = config
                .kernel_crate
                .as_ref()
                .map(|kernel_crate| kernel_crate.import.as_str())
                .unwrap_or("jolt_kernels");
            source.push_str(&format!(
                "use {kernel_crate}::{{{}}};\n",
                kernel_modules.join(", ")
            ));
        }
        source.push_str(&config.default_transcript_type.use_line());
        source.push_str(&config.transcript_trait.use_line());
        source.push_str(&format!(
            "use {verifier_import}::{{{named_eval_type}, {proof_type}, {stage_proof_type}, {sumcheck_output_type}}};\n\n",
        ));
    }
    if !modules.is_empty() {
        source.push_str(&format!(
            "use crate::stages::{{{}}};\n\n",
            aliased_modules(&modules).join(", ")
        ));
    }
    source.push_str(&format!(
        "pub type {default_transcript_alias} = {default_transcript_type}<{field_type}>;\n\n"
    ));

    source.push_str(&format!(
        "pub struct {prover_inputs_type}<'a, {}> {{\n",
        generic_params.join(", ")
    ));
    if has_commitment {
        source.push_str("    pub commitment_inputs: &'a mut CommitmentInputs,\n");
        source.push_str(&format!("    pub prover_setup: &'a {prover_setup_type},\n"));
    }
    for stage in &stages {
        source.push_str(&format!(
            "    pub {}_executor: &'a mut {}Executor,\n",
            stage.field_name, stage.variant_name
        ));
    }
    if supports_evaluation {
        let stage7_kernel = stage_api_by_field(&stages, "stage7")
            .and_then(|stage| stage.kernel_module.as_deref())
            .unwrap_or("stage7");
        source.push_str(&format!(
            "    pub stage7_openings: Option<&'a [{stage7_kernel}::Stage7OpeningInputValue<{field_type}>]>,\n"
        ));
    }
    source.push_str("}\n\n");

    source.push_str(&format!(
        "#[derive(Clone, Copy, Debug)]\npub struct {prover_programs_type} {{\n"
    ));
    if let Some(commitment) = &commitment {
        if let (Some(program_type), Some(_), Some(_)) = (
            &commitment.program_type,
            &commitment.program_const,
            &commitment.with_program_prover_fn,
        ) {
            source.push_str(&format!(
                "    pub {}: &'static {}::{},\n",
                commitment.field_name, commitment.module_alias, program_type
            ));
        }
    }
    for stage in &stages {
        if let (Some(program_type), Some(_), Some(_)) = (
            &stage.program_type,
            &stage.program_const,
            &stage.with_program_prover_fn,
        ) {
            let program_module = stage
                .kernel_module
                .as_deref()
                .unwrap_or(stage.module_alias.as_str());
            source.push_str(&format!(
                "    pub {}: &'static {}::{},\n",
                stage.field_name, program_module, program_type
            ));
        }
    }
    if supports_evaluation {
        source.push_str("    pub stage8: &'static stage8_stage::Stage8EvaluationProgramPlan,\n");
    }
    source.push_str("}\n\n");

    source.push_str(&format!(
        "pub fn default_prover_programs() -> {prover_programs_type} {{\n    {prover_programs_type} {{\n"
    ));
    if let Some(commitment) = &commitment {
        if let (Some(_), Some(program_const), Some(_)) = (
            &commitment.program_type,
            &commitment.program_const,
            &commitment.with_program_prover_fn,
        ) {
            source.push_str(&format!(
                "        {}: &{}::{},\n",
                commitment.field_name, commitment.module_alias, program_const
            ));
        }
    }
    for stage in &stages {
        if let (Some(_), Some(program_const), Some(_)) = (
            &stage.program_type,
            &stage.program_const,
            &stage.with_program_prover_fn,
        ) {
            source.push_str(&format!(
                "        {}: &{}::{},\n",
                stage.field_name, stage.module_alias, program_const
            ));
        }
    }
    if supports_evaluation {
        source.push_str("        stage8: &stage8_stage::STAGE8_PROGRAM,\n");
    }
    source.push_str("    }\n}\n\n");

    source.push_str(&format!(
        "#[derive(Clone, Debug)]\npub struct {prover_artifacts_type} {{\n"
    ));
    if let Some(commitment) = &commitment {
        source.push_str(&format!(
            "    pub {}: {}::{},\n",
            commitment.field_name, commitment.module_alias, commitment.artifacts_type
        ));
    }
    for stage in &stages {
        let kernel_module = stage
            .kernel_module
            .as_deref()
            .unwrap_or(stage.module_alias.as_str());
        source.push_str(&format!(
            "    pub {}: {}::{}<{field_type}>,\n",
            stage.field_name, kernel_module, stage.artifacts_type
        ));
    }
    source.push_str("}\n\n");

    source.push_str(&format!(
        "#[derive(Debug)]\npub enum {prove_error_type} {{\n"
    ));
    if let Some(commitment) = &commitment {
        source.push_str(&format!(
            "    {}({}::{}),\n",
            commitment.variant_name, commitment.module_alias, commitment.error_type
        ));
    }
    for stage in &stages {
        let kernel_module = stage
            .kernel_module
            .as_deref()
            .unwrap_or(stage.module_alias.as_str());
        source.push_str(&format!(
            "    {}({}::{}),\n",
            stage.variant_name, kernel_module, stage.error_type
        ));
    }
    if supports_evaluation {
        source.push_str(&format!("    Evaluation({prefix}EvaluationProveError),\n"));
    }
    source.push_str("}\n\n");

    if supports_evaluation {
        source.push_str(&format!(
            "#[derive(Debug)]\npub enum {prefix}EvaluationProveError {{\n    MissingOracle {{ oracle: &'static str }},\n    MissingOpeningHint {{ oracle: &'static str }},\n    MissingStageEval {{ stage: &'static str, eval: &'static str }},\n    MissingStage7RaEval,\n    MissingStage7EvaluationPoint,\n    InvalidPointLength {{\n        artifact: &'static str,\n        expected: usize,\n        actual: usize,\n    }},\n    TargetSizeOverflow {{ num_vars: usize }},\n}}\n\n",
        ));
    }

    if let Some(commitment) = &commitment {
        source.push_str(&format!(
            "impl From<{module}::{error}> for {prove_error_type} {{\n    fn from(error: {module}::{error}) -> Self {{\n        Self::{variant}(error)\n    }}\n}}\n\n",
            module = commitment.module_alias,
            error = commitment.error_type,
            variant = commitment.variant_name,
        ));
    }
    for stage in &stages {
        let kernel_module = stage
            .kernel_module
            .as_deref()
            .unwrap_or(stage.module_alias.as_str());
        source.push_str(&format!(
            "impl From<{}::{}> for {prove_error_type} {{\n    fn from(error: {}::{}) -> Self {{\n        Self::{}(error)\n    }}\n}}\n\n",
            kernel_module, stage.error_type, kernel_module, stage.error_type, stage.variant_name
        ));
    }
    if supports_evaluation {
        source.push_str(&format!(
            "impl From<{prefix}EvaluationProveError> for {prove_error_type} {{\n    fn from(error: {prefix}EvaluationProveError) -> Self {{\n        Self::Evaluation(error)\n    }}\n}}\n\n",
        ));
    }

    source.push_str(&format!(
        "pub fn prove_{protocol_snake}<{}, T>(\n    inputs: {prover_inputs_type}<'_, {}>,\n    transcript: &mut T,\n) -> Result<({proof_type}, {prover_artifacts_type}), {prove_error_type}>\nwhere\n",
        generic_params.join(", "),
        generic_params.join(", ")
    ));
    if let Some(commitment) = &commitment {
        let input_provider = commitment
            .input_provider_trait
            .as_deref()
            .unwrap_or("MissingCommitmentInputProvider");
        source.push_str(&format!(
            "    CommitmentInputs: {}::{input_provider},\n",
            commitment.module_alias
        ));
    }
    for stage in &stages {
        let kernel_module = stage
            .kernel_module
            .as_deref()
            .unwrap_or(stage.module_alias.as_str());
        let kernel_trait = kernel_executor_type(&stage.error_type);
        source.push_str(&format!(
            "    {}Executor: {}::{}<{field_type}>,\n",
            stage.variant_name, kernel_module, kernel_trait
        ));
    }
    source.push_str(&format!(
        "    T: {transcript_trait}<Challenge = {field_type}>,\n"
    ));
    source.push_str("{\n");
    source.push_str(&format!(
        "    prove_{protocol_snake}_with_programs(inputs, default_prover_programs(), transcript)\n}}\n\n"
    ));

    source.push_str(&format!(
        "pub fn prove_{protocol_snake}_with_programs<{}, T>(\n    inputs: {prover_inputs_type}<'_, {}>,\n    programs: {prover_programs_type},\n    transcript: &mut T,\n) -> Result<({proof_type}, {prover_artifacts_type}), {prove_error_type}>\nwhere\n",
        generic_params.join(", "),
        generic_params.join(", ")
    ));
    if let Some(commitment) = &commitment {
        let input_provider = commitment
            .input_provider_trait
            .as_deref()
            .unwrap_or("MissingCommitmentInputProvider");
        source.push_str(&format!(
            "    CommitmentInputs: {}::{input_provider},\n",
            commitment.module_alias
        ));
    }
    for stage in &stages {
        let kernel_module = stage
            .kernel_module
            .as_deref()
            .unwrap_or(stage.module_alias.as_str());
        let kernel_trait = kernel_executor_type(&stage.error_type);
        source.push_str(&format!(
            "    {}Executor: {}::{}<{field_type}>,\n",
            stage.variant_name, kernel_module, kernel_trait
        ));
    }
    source.push_str(&format!(
        "    T: {transcript_trait}<Challenge = {field_type}>,\n"
    ));
    source.push_str("{\n");
    if let Some(commitment) = &commitment {
        let prover_fn = commitment
            .with_program_prover_fn
            .as_deref()
            .or(commitment.prover_fn.as_deref())
            .unwrap_or("missing_commitment_prover_function");
        let program_arg = if commitment.with_program_prover_fn.is_some()
            && commitment.program_type.is_some()
            && commitment.program_const.is_some()
        {
            format!("programs.{}, ", commitment.field_name)
        } else {
            String::new()
        };
        source.push_str(&format!(
            "    let {field} = {module}::{prover_fn}(\n        {program_arg}inputs.commitment_inputs,\n        inputs.prover_setup,\n        transcript,\n    )?;\n",
            field = commitment.field_name,
            module = commitment.module_alias
        ));
    }
    for stage in &stages {
        let prover_fn = stage
            .with_program_prover_fn
            .as_deref()
            .or(stage.prover_fn.as_deref())
            .unwrap_or("missing_prover_function");
        let program_arg = if stage.with_program_prover_fn.is_some()
            && stage.program_type.is_some()
            && stage.program_const.is_some()
        {
            format!("programs.{}, ", stage.field_name)
        } else {
            String::new()
        };
        source.push_str(&format!(
            "    let {} = {}::{}({program_arg}inputs.{}_executor, transcript)?;\n",
            stage.field_name, stage.module_alias, prover_fn, stage.field_name
        ));
    }
    if supports_evaluation {
        source.push_str(
            "    let evaluation = if let Some(stage7_openings) = inputs.stage7_openings {\n        Some(prove_jolt_evaluation_proof(\n            programs.stage8,\n            inputs.commitment_inputs,\n            inputs.prover_setup,\n            &commitment,\n            &stage6,\n            &stage7,\n            stage7_openings,\n            transcript,\n        )?)\n    } else {\n        None\n    };\n",
        );
    }
    source.push_str(&format!("\n    let proof = {proof_type} {{\n"));
    if let Some(commitment) = &commitment {
        source.push_str(&format!(
            "        commitments: {}.commitments.clone(),\n",
            commitment.field_name
        ));
    }
    for stage in &stages {
        source.push_str(&format!(
            "        {}: {}_proof(&{}),\n",
            stage.field_name, stage.field_name, stage.field_name
        ));
    }
    if supports_evaluation {
        source.push_str("        evaluation,\n");
    }
    source.push_str(&format!(
        "    }};\n    let artifacts = {prover_artifacts_type} {{\n"
    ));
    if let Some(commitment) = &commitment {
        source.push_str(&format!("        {},\n", commitment.field_name));
    }
    for stage in &stages {
        source.push_str(&format!("        {},\n", stage.field_name));
    }
    source.push_str("    };\n    Ok((proof, artifacts))\n}\n\n");

    if supports_evaluation {
        source.push_str(&jolt_prover_evaluation_helpers(field_type));
    }

    for stage in &stages {
        let kernel_module = stage
            .kernel_module
            .as_deref()
            .unwrap_or(stage.module_alias.as_str());
        source.push_str(&format!(
            "fn {field}_proof(artifacts: &{kernel}::{artifacts_ty}<{field_type}>) -> {stage_proof_type} {{\n    {stage_proof_type} {{\n        sumchecks: artifacts.sumchecks.iter().map({field}_sumcheck).collect(),\n    }}\n}}\n\n",
            field = stage.field_name,
            kernel = kernel_module,
            artifacts_ty = stage.artifacts_type
        ));
        source.push_str(&format!(
            "fn {field}_sumcheck(output: &{kernel}::{output_ty}<{field_type}>) -> {sumcheck_output_type} {{\n    {sumcheck_output_type} {{\n        driver: output.driver,\n        point: output.point.clone(),\n        evals: output.evals.iter().map({field}_eval).collect(),\n        proof: output.proof.clone(),\n    }}\n}}\n\n",
            field = stage.field_name,
            kernel = kernel_module,
            output_ty = stage.output_type
        ));
        source.push_str(&format!(
            "fn {field}_eval(eval: &{kernel}::{eval_ty}<{field_type}>) -> {named_eval_type} {{\n    {named_eval_type} {{\n        name: eval.name,\n        oracle: eval.oracle,\n        value: eval.value,\n    }}\n}}\n\n",
            field = stage.field_name,
            kernel = kernel_module,
            eval_ty = stage.eval_type
        ));
    }
    source
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
    let target_len = target_len(num_vars)?;
    let data = commitment_inputs
        .materialize(oracle)
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
    for (dst, value) in joint.iter_mut().take(limit).zip(data.iter()) {{
        if *value == zero {{
            continue;
        }}
        if *value == one {{
            *dst += scalar;
        }} else {{
            *dst += *value * scalar;
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

fn stage_apis(
    config: &ProtocolArtifactConfig,
    artifacts: &[ProtocolRustArtifact],
) -> Vec<StageRustApi> {
    artifacts
        .iter()
        .filter(|artifact| artifact.stage.is_proof())
        .map(|artifact| stage_api(config, artifact))
        .collect()
}

fn stage_api(config: &ProtocolArtifactConfig, artifact: &ProtocolRustArtifact) -> StageRustApi {
    let source = artifact.source.source.as_str();
    let artifacts_type = find_type_with_suffix(source, "ExecutionArtifacts").unwrap_or_else(|| {
        format!(
            "{}ExecutionArtifacts",
            upper_camel(artifact.stage.module_name())
        )
    });
    let prefix = artifacts_type
        .strip_suffix("ExecutionArtifacts")
        .unwrap_or(&artifacts_type);
    let proof_type = find_public_item(source, "pub struct ", "Proof")
        .unwrap_or_else(|| format!("{}Proof", upper_camel(artifact.stage.module_name())));
    let opening_input_name = format!("{prefix}OpeningInputValue");
    let ram_data_name = format!("{prefix}RamData");
    let verifier_data_name = format!("{prefix}VerifierData");
    let program_type_suffix = match artifact.role {
        Role::Prover => "CpuProgramPlan",
        Role::Verifier => "VerifierProgramPlan",
    };
    let program_type = find_type_with_suffix(source, program_type_suffix);
    let program_const = program_type
        .as_deref()
        .and_then(|program_type| find_public_const_of_type(source, program_type));
    let error_type = match artifact.role {
        Role::Prover => find_type_with_suffix(source, "KernelError")
            .unwrap_or_else(|| format!("{prefix}KernelError")),
        Role::Verifier => find_public_item(source, "pub enum ", "Error")
            .unwrap_or_else(|| format!("Verify{prefix}Error")),
    };
    StageRustApi {
        field_name: artifact.stage.module_name().to_owned(),
        module_alias: module_alias(artifact.stage.module_name()),
        variant_name: upper_camel(artifact.stage.module_name()),
        proof_type,
        output_type: format!("{prefix}SumcheckOutput"),
        eval_type: format!("{prefix}NamedEval"),
        artifacts_type,
        error_type,
        verifier_fn: find_public_fn(source, &["verify_"]),
        with_program_verifier_fn: find_public_fn_containing(source, &["verify_"], "_with_program"),
        program_type,
        program_const,
        prover_fn: find_public_fn(source, &["prove_", "execute_"]),
        with_program_prover_fn: find_public_fn_containing(
            source,
            &["prove_", "execute_"],
            "_with_program",
        ),
        kernel_module: find_kernel_module(config, source),
        opening_input_type: source
            .contains(&format!("pub struct {opening_input_name}"))
            .then_some(opening_input_name),
        ram_data_type: source
            .contains(&format!("pub struct {ram_data_name}"))
            .then_some(ram_data_name),
        verifier_data_type: source
            .contains(&format!("pub struct {verifier_data_name}"))
            .then_some(verifier_data_name),
    }
}

fn commitment_api(artifacts: &[ProtocolRustArtifact]) -> Option<CommitmentRustApi> {
    let artifact = artifacts
        .iter()
        .find(|artifact| artifact.stage.is_commitment())?;
    let source = artifact.source.source.as_str();
    let artifacts_type = find_type_with_suffix(source, "Artifacts")
        .unwrap_or_else(|| format!("{}Artifacts", upper_camel(artifact.stage.module_name())));
    let error_type = find_public_item(source, "pub enum ", "Error")
        .unwrap_or_else(|| format!("{}Error", upper_camel(artifact.stage.module_name())));
    let input_provider_trait = find_public_item(source, "pub trait ", "InputProvider");
    let program_type_suffix = match artifact.role {
        Role::Prover => "ProverProgramPlan",
        Role::Verifier => "VerifierProgramPlan",
    };
    let program_type = find_type_with_suffix(source, program_type_suffix);
    let program_const = program_type
        .as_deref()
        .and_then(|program_type| find_public_const_of_type(source, program_type));
    Some(CommitmentRustApi {
        field_name: artifact.stage.module_name().to_owned(),
        module_alias: module_alias(artifact.stage.module_name()),
        variant_name: upper_camel(artifact.stage.module_name()),
        artifacts_type,
        error_type,
        verifier_fn: find_public_fn(source, &["verify_"]),
        with_program_verifier_fn: find_public_fn_containing(source, &["verify_"], "_with_program"),
        program_type,
        program_const,
        prover_fn: find_public_fn(source, &["prove_"]),
        with_program_prover_fn: find_public_fn_containing(source, &["prove_"], "_with_program"),
        input_provider_trait,
    })
}

fn supports_jolt_evaluation(
    config: &ProtocolArtifactConfig,
    stages: &[StageRustApi],
    commitment: &Option<CommitmentRustApi>,
    artifacts: &[ProtocolRustArtifact],
) -> bool {
    config.type_prefix == "Jolt"
        && commitment.is_some()
        && stages.iter().any(|stage| stage.field_name == "stage6")
        && stages.iter().any(|stage| stage.field_name == "stage7")
        && artifacts.iter().any(|artifact| {
            artifact.stage.is_evaluation() && artifact.stage.module_name() == "stage8"
        })
}

fn stage_api_by_field<'a>(stages: &'a [StageRustApi], field: &str) -> Option<&'a StageRustApi> {
    stages.iter().find(|stage| stage.field_name == field)
}

fn role_modules(artifacts: &[ProtocolRustArtifact]) -> Vec<String> {
    artifacts
        .iter()
        .map(|artifact| artifact.stage.module_name().to_owned())
        .collect()
}

fn aliased_modules(modules: &[String]) -> Vec<String> {
    modules
        .iter()
        .map(|module| format!("{module} as {}", module_alias(module)))
        .collect()
}

fn module_alias(module: &str) -> String {
    format!("{module}_stage")
}

fn unique_kernel_modules(stages: &[StageRustApi]) -> Vec<String> {
    let mut modules = Vec::new();
    for stage in stages {
        if let Some(kernel_module) = &stage.kernel_module {
            if !modules.contains(kernel_module) {
                modules.push(kernel_module.clone());
            }
        }
    }
    modules
}

fn prover_generic_params(stages: &[StageRustApi], has_commitment: bool) -> Vec<String> {
    let mut params = if has_commitment {
        vec!["CommitmentInputs".to_owned()]
    } else {
        Vec::new()
    };
    params.extend(
        stages
            .iter()
            .map(|stage| format!("{}Executor", stage.variant_name)),
    );
    params
}

fn find_public_item(source: &str, prefix: &str, suffix: &str) -> Option<String> {
    source.lines().find_map(|line| {
        let trimmed = line.trim_start();
        let rest = trimmed.strip_prefix(prefix)?;
        let name = rest
            .split(|character: char| {
                matches!(character, '<' | '(' | '{') || character.is_whitespace()
            })
            .next()?;
        name.ends_with(suffix).then(|| name.to_owned())
    })
}

fn find_type_with_suffix(source: &str, suffix: &str) -> Option<String> {
    source
        .split(|character: char| !character.is_ascii_alphanumeric() && character != '_')
        .find(|token| token.ends_with(suffix) && token.len() > suffix.len())
        .map(ToOwned::to_owned)
}

fn find_public_fn(source: &str, prefixes: &[&str]) -> Option<String> {
    source.lines().find_map(|line| {
        let trimmed = line.trim_start();
        let rest = trimmed.strip_prefix("pub fn ")?;
        let name = rest
            .split(|character: char| matches!(character, '<' | '(') || character.is_whitespace())
            .next()?;
        prefixes
            .iter()
            .any(|prefix| name.starts_with(prefix))
            .then(|| name.to_owned())
    })
}

fn find_public_fn_containing(source: &str, prefixes: &[&str], needle: &str) -> Option<String> {
    source.lines().find_map(|line| {
        let trimmed = line.trim_start();
        let rest = trimmed.strip_prefix("pub fn ")?;
        let name = rest
            .split(|character: char| matches!(character, '<' | '(') || character.is_whitespace())
            .next()?;
        (name.contains(needle) && prefixes.iter().any(|prefix| name.starts_with(prefix)))
            .then(|| name.to_owned())
    })
}

fn find_public_const_of_type(source: &str, type_name: &str) -> Option<String> {
    source.lines().find_map(|line| {
        let trimmed = line.trim_start();
        let rest = trimmed.strip_prefix("pub const ")?;
        let name = rest
            .split(|character: char| character == ':' || character.is_whitespace())
            .next()?;
        rest.contains(&format!(": {type_name}"))
            .then(|| name.to_owned())
    })
}

fn kernel_executor_type(error_type: &str) -> String {
    error_type.strip_suffix("KernelError").map_or_else(
        || {
            error_type
                .replace("Verify", "")
                .replace("Error", "KernelExecutor")
        },
        |prefix| format!("{prefix}KernelExecutor"),
    )
}

fn find_kernel_module(config: &ProtocolArtifactConfig, source: &str) -> Option<String> {
    let kernel_import = config.kernel_crate.as_ref()?.import.as_str();
    let prefix = format!("use {kernel_import}::");
    source.lines().find_map(|line| {
        let rest = line.trim_start().strip_prefix(&prefix)?;
        rest.split(|character: char| matches!(character, ':' | '{') || character.is_whitespace())
            .next()
            .filter(|name| !name.is_empty())
            .map(ToOwned::to_owned)
    })
}

fn upper_camel(name: &str) -> String {
    let mut output = String::new();
    for segment in name.split('_') {
        let mut chars = segment.chars();
        if let Some(first) = chars.next() {
            output.extend(first.to_uppercase());
            output.push_str(chars.as_str());
        }
    }
    output
}

fn snake_case(name: &str) -> String {
    let mut output = String::new();
    for (index, character) in name.chars().enumerate() {
        if character.is_ascii_uppercase() {
            if index != 0 {
                output.push('_');
            }
            output.extend(character.to_lowercase());
        } else if character == '-' || character == ' ' {
            output.push('_');
        } else {
            output.push(character);
        }
    }
    output
}

fn rust_crate_ident(package_name: &str) -> String {
    package_name.replace('-', "_")
}

fn byte_string_literal(label: &str) -> String {
    let escaped = label.escape_default().to_string();
    format!("b\"{escaped}\"")
}

pub fn jolt_artifact_config() -> ProtocolArtifactConfig {
    ProtocolArtifactConfig {
        protocol_name: "Jolt".to_owned(),
        type_prefix: "Jolt".to_owned(),
        transcript_label: "Jolt".to_owned(),
        prover_crate_name: "jolt-prover".to_owned(),
        verifier_crate_name: "jolt-verifier".to_owned(),
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
        sumcheck_proof_type: RustTypeRef::new("jolt_sumcheck::SumcheckProof"),
        commitment_type: RustTypeRef::new("jolt_dory::DoryCommitment"),
        prover_setup_type: RustTypeRef::new("jolt_dory::DoryProverSetup"),
    }
}

fn generated_file_path(root: &Path, relative_path: &str) -> Result<std::path::PathBuf, EmitError> {
    let path = Path::new(relative_path);
    if path.is_absolute()
        || path.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
    {
        return Err(EmitError::new(format!(
            "generated crate file path `{relative_path}` must be relative and stay inside the crate"
        )));
    }
    Ok(root.join(path))
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
