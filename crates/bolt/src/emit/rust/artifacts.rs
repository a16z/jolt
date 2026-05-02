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
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JoltProtocolStage {
    Commitment,
    Stage1Outer,
    Stage2,
    Stage3,
}

impl JoltProtocolStage {
    pub fn name(self) -> &'static str {
        match self {
            Self::Commitment => "commitment",
            Self::Stage1Outer => "stage1_outer",
            Self::Stage2 => "stage2",
            Self::Stage3 => "stage3",
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
    let role_module = match role {
        Role::Prover => format!(
            "#[rustfmt::skip]\npub mod prover;\npub mod stages;\n\npub use prover::{{\n    prove_{protocol_snake}, Default{prefix}Transcript, {prefix}ProveError, {prefix}ProverArtifacts, {prefix}ProverInputs,\n}};"
        ),
        Role::Verifier => format!(
            "pub mod stages;\n#[rustfmt::skip]\npub mod verifier;\n\npub use verifier::{{\n    verify_{protocol_snake}, {prefix}NamedEval, {prefix}Proof, {prefix}StageProof, {prefix}SumcheckOutput,\n    {prefix}VerificationArtifacts, {prefix}VerifierInputs, {prefix}VerifyError,\n}};"
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
    prover_fn: Option<String>,
    kernel_module: Option<String>,
    opening_input_type: Option<String>,
    ram_data_type: Option<String>,
}

#[derive(Clone, Debug)]
struct CommitmentRustApi {
    field_name: String,
    module_alias: String,
    variant_name: String,
    artifacts_type: String,
    error_type: String,
    verifier_fn: Option<String>,
    prover_fn: Option<String>,
    input_provider_trait: Option<String>,
}

fn generated_verifier_api(
    config: &ProtocolArtifactConfig,
    artifacts: &[ProtocolRustArtifact],
) -> String {
    let stages = stage_apis(config, artifacts);
    let modules = role_modules(artifacts);
    let commitment = commitment_api(artifacts);
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
    let verification_artifacts_type = format!("{prefix}VerificationArtifacts");
    let verify_error_type = format!("{prefix}VerifyError");

    let mut source = String::new();
    if commitment.is_some() {
        source.push_str(&config.commitment_type.use_line());
    }
    source.push_str(&config.field_type.use_line());
    if !stages.is_empty() {
        source.push_str(&config.sumcheck_proof_type.use_line());
    }
    source.push_str(&config.transcript_trait.use_line());
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
    source.push_str("}\n\n");

    source.push_str(&format!(
        "#[derive(Clone, Copy, Debug)]\npub struct {verifier_inputs_type}<'a> {{\n"
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
    }
    source.push_str("}\n\n");

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
    source.push_str("}\n\n");

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

    source.push_str(&format!(
        "pub fn verify_{protocol_snake}<T>(\n    proof: &{proof_type},\n    inputs: {verifier_inputs_type}<'_>,\n    transcript: &mut T,\n) -> Result<{verification_artifacts_type}, {verify_error_type}>\nwhere\n    T: {transcript_trait}<Challenge = {field_type}>,\n{{\n",
    ));
    if let Some(commitment) = &commitment {
        let verifier_fn = commitment
            .verifier_fn
            .as_deref()
            .unwrap_or("missing_commitment_verifier_function");
        source.push_str(&format!(
            "    let {field} = {module}::{verifier_fn}(&proof.commitments, transcript)?;\n",
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
            .verifier_fn
            .as_deref()
            .unwrap_or("missing_verifier_function");
        let mut args = vec![format!("&{}_proof", stage.field_name)];
        if stage.opening_input_type.is_some() {
            args.push(format!("inputs.{}_openings", stage.field_name));
        }
        if stage.ram_data_type.is_some() {
            args.push(format!("inputs.{}_ram", stage.field_name));
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
    source.push_str(&format!("\n    Ok({verification_artifacts_type} {{\n"));
    if let Some(commitment) = &commitment {
        source.push_str(&format!("        {},\n", commitment.field_name));
    }
    for stage in &stages {
        source.push_str(&format!("        {},\n", stage.field_name));
    }
    source.push_str("    })\n}\n\n");

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
    let generic_params = prover_generic_params(&stages, has_commitment);
    let prefix = &config.type_prefix;
    let protocol_snake = config.protocol_snake();
    let field_type = config.field_type.ident();
    let default_transcript_type = config.default_transcript_type.ident();
    let prover_setup_type = config.prover_setup_type.ident();
    let verifier_import = config.verifier_crate_import();
    let named_eval_type = format!("{prefix}NamedEval");
    let sumcheck_output_type = format!("{prefix}SumcheckOutput");
    let stage_proof_type = format!("{prefix}StageProof");
    let proof_type = format!("{prefix}Proof");
    let prover_inputs_type = format!("{prefix}ProverInputs");
    let prover_artifacts_type = format!("{prefix}ProverArtifacts");
    let prove_error_type = format!("{prefix}ProveError");
    let default_transcript_alias = format!("Default{prefix}Transcript");

    let mut source = String::new();
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
    source.push_str(&format!(
        "use {verifier_import}::{{{named_eval_type}, {proof_type}, {stage_proof_type}, {sumcheck_output_type}}};\n\n",
    ));
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
    source.push_str("}\n\n");

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
    source.push_str("}\n\n");

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

    source.push_str(&format!(
        "pub fn prove_{protocol_snake}<{}>(\n    inputs: {prover_inputs_type}<'_, {}>,\n    transcript: &mut {default_transcript_alias},\n) -> Result<({proof_type}, {prover_artifacts_type}), {prove_error_type}>\nwhere\n",
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
    source.push_str("{\n");
    if let Some(commitment) = &commitment {
        let prover_fn = commitment
            .prover_fn
            .as_deref()
            .unwrap_or("missing_commitment_prover_function");
        source.push_str(&format!(
            "    let {field} = {module}::{prover_fn}(\n        inputs.commitment_inputs,\n        inputs.prover_setup,\n        transcript,\n    )?;\n",
            field = commitment.field_name,
            module = commitment.module_alias
        ));
    }
    for stage in &stages {
        let prover_fn = stage
            .prover_fn
            .as_deref()
            .unwrap_or("missing_prover_function");
        source.push_str(&format!(
            "    let {} = {}::{}(inputs.{}_executor, transcript)?;\n",
            stage.field_name, stage.module_alias, prover_fn, stage.field_name
        ));
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

fn stage_apis(
    config: &ProtocolArtifactConfig,
    artifacts: &[ProtocolRustArtifact],
) -> Vec<StageRustApi> {
    artifacts
        .iter()
        .filter(|artifact| !artifact.stage.is_commitment())
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
        prover_fn: find_public_fn(source, &["prove_", "execute_"]),
        kernel_module: find_kernel_module(config, source),
        opening_input_type: source
            .contains(&format!("pub struct {opening_input_name}"))
            .then_some(opening_input_name),
        ram_data_type: source
            .contains(&format!("pub struct {ram_data_name}"))
            .then_some(ram_data_name),
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
    Some(CommitmentRustApi {
        field_name: artifact.stage.module_name().to_owned(),
        module_alias: module_alias(artifact.stage.module_name()),
        variant_name: upper_camel(artifact.stage.module_name()),
        artifacts_type,
        error_type,
        verifier_fn: find_public_fn(source, &["verify_"]),
        prover_fn: find_public_fn(source, &["prove_"]),
        input_provider_trait,
    })
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
        common_dependencies: vec!["jolt-field".to_owned(), "jolt-transcript".to_owned()],
        prover_dependencies: vec![
            "jolt-dory".to_owned(),
            "jolt-kernels".to_owned(),
            "jolt-witness".to_owned(),
        ],
        verifier_dependencies: vec![
            "jolt-dory".to_owned(),
            "jolt-poly".to_owned(),
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
