use std::path::{Component, Path};

use crate::ir::Role;

use super::{EmitError, RustSourceFile};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProtocolArtifactConfig {
    pub protocol_name: String,
    pub type_prefix: String,
    pub transcript_label: String,
    pub repository: Option<String>,
    pub prover_crate_name: String,
    pub verifier_crate_name: String,
    pub crates_io_patches: Vec<String>,
    pub standalone_dependency_overrides: Vec<ProtocolStandaloneDependency>,
    pub common_dependencies: Vec<String>,
    pub prover_dependencies: Vec<String>,
    pub verifier_dependencies: Vec<String>,
    pub prover_forbidden_imports: Vec<String>,
    pub verifier_forbidden_imports: Vec<String>,
    pub kernel_crate: Option<ProtocolCrateRef>,
    pub field_type: RustTypeRef,
    pub default_transcript_type: RustTypeRef,
    pub transcript_trait: RustTypeRef,
    pub commitment_type: RustTypeRef,
    pub prover_setup_type: RustTypeRef,
    pub role_api_extension: Option<ProtocolArtifactExtension>,
    pub verifier_runtime_modules: Vec<ProtocolRuntimeModule>,
    pub verifier_named_eval_type: RustTypeRef,
    pub verifier_sumcheck_output_type: RustTypeRef,
    pub verifier_stage_proof_type: RustTypeRef,
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
pub struct ProtocolStandaloneDependency {
    pub package: String,
    pub manifest_entry: String,
}

impl ProtocolStandaloneDependency {
    pub fn new(package: impl Into<String>, manifest_entry: impl Into<String>) -> Self {
        Self {
            package: package.into(),
            manifest_entry: manifest_entry.into(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProtocolArtifactExtension {
    pub required_commitment: bool,
    pub required_proof_stages: Vec<String>,
    pub required_artifact_stages: Vec<String>,
    pub prover: ProtocolProverApiExtension,
    pub verifier: ProtocolVerifierApiExtension,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ProtocolProverApiExtension {
    pub lib_module: String,
    pub imports: String,
    pub input_fields: String,
    pub program_fields: String,
    pub default_program_fields: String,
    pub error_variants: String,
    pub error_items: String,
    pub error_conversions: String,
    pub after_stage_execution: String,
    pub proof_fields: String,
    pub helper_items: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ProtocolVerifierApiExtension {
    pub lib_module: String,
    pub imports: String,
    pub proof_fields: String,
    pub proof_items: String,
    pub inputs_derive: Option<String>,
    pub input_fields: String,
    pub program_fields: String,
    pub default_program_fields: String,
    pub error_variants: String,
    pub error_items: String,
    pub error_conversions: String,
    pub after_default_verify: String,
    pub with_programs_body_intro: String,
    pub after_stage_verification: String,
    pub helper_items: String,
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProtocolRuntimeModule {
    pub module_name: String,
    pub file: GeneratedFile,
}

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
    let mut stage_module_lines = Vec::new();
    if role == Role::Verifier {
        stage_module_lines.extend(
            config
                .verifier_runtime_modules
                .iter()
                .map(|module| format!("pub mod {};", module.module_name)),
        );
    }
    stage_module_lines.extend(artifacts.iter().map(|artifact| {
        format!(
            "#[rustfmt::skip]\npub mod {};",
            artifact.stage.module_name()
        )
    }));
    let stage_modules = stage_module_lines.join("\n");
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
    if role == Role::Verifier {
        files.extend(
            config
                .verifier_runtime_modules
                .iter()
                .map(|module| module.file.clone()),
        );
    }
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
            let patch_section = if config.crates_io_patches.is_empty() {
                String::new()
            } else {
                format!(
                    "\n[patch.crates-io]\n{}\n",
                    config.crates_io_patches.join("\n")
                )
            };
            let dependencies = dependencies
                .into_iter()
                .map(|name| standalone_dependency_entry(config, dependency_root, &name))
                .collect::<Vec<_>>()
                .join("\n");
            format!(
                "[package]\nname = \"{crate_name}\"\nversion = \"0.0.0\"\nedition = \"2021\"\n{patch_section}\n[dependencies]\n{dependencies}\n"
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
            let repository = config
                .repository
                .as_ref()
                .map(|repository| format!("repository = \"{repository}\"\n"))
                .unwrap_or_default();
            format!(
                "[package]\nname = \"{crate_name}\"\nversion = \"0.0.0\"\nedition = \"2021\"\nlicense = \"MIT OR Apache-2.0\"\ndescription = \"Bolt-generated {} {role_name} role crate\"\n{repository}\n[lints]\nworkspace = true\n\n[dependencies]\n{dependencies}\n",
                config.protocol_name
            )
        }
    }
}

fn standalone_dependency_entry(
    config: &ProtocolArtifactConfig,
    dependency_root: &str,
    package: &str,
) -> String {
    config
        .standalone_dependency_overrides
        .iter()
        .find(|dependency| dependency.package == package)
        .map(|dependency| dependency.manifest_entry.clone())
        .unwrap_or_else(|| format!("{package} = {{ path = \"{dependency_root}/{package}\" }}"))
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
    let extension = active_role_api_extension(config, &stage_apis, &commitment_api, artifacts);
    let role_module = match (role, extension) {
        (Role::Prover, Some(extension)) => extension.prover.lib_module.clone(),
        (Role::Prover, None) => format!(
            "#[rustfmt::skip]\npub mod prover;\npub mod stages;\n\npub use prover::{{\n    default_prover_programs, prove_{protocol_snake}, prove_{protocol_snake}_with_programs,\n    Default{prefix}Transcript, {prefix}ProveError, {prefix}ProverArtifacts, {prefix}ProverInputs,\n    {prefix}ProverPrograms,\n}};"
        ),
        (Role::Verifier, Some(extension)) => extension.verifier.lib_module.clone(),
        (Role::Verifier, None) => format!(
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
    let extension = active_role_api_extension(config, &stages, &commitment, artifacts);
    let prefix = &config.type_prefix;
    let protocol_snake = config.protocol_snake();
    let field_type = config.field_type.ident();
    let transcript_trait = config.transcript_trait.ident();
    let commitment_type = config.commitment_type.ident();
    let runtime_named_eval_type = &config.verifier_named_eval_type.path;
    let runtime_sumcheck_output_type = &config.verifier_sumcheck_output_type.path;
    let runtime_stage_proof_type = &config.verifier_stage_proof_type.path;
    let named_eval_type = format!("{prefix}NamedEval");
    let sumcheck_output_type = format!("{prefix}SumcheckOutput");
    let stage_proof_type = format!("{prefix}StageProof");
    let proof_type = format!("{prefix}Proof");
    let verifier_inputs_type = format!("{prefix}VerifierInputs");
    let verifier_programs_type = format!("{prefix}VerifierPrograms");
    let verification_artifacts_type = format!("{prefix}VerificationArtifacts");
    let verify_error_type = format!("{prefix}VerifyError");

    let mut source = String::new();
    if let Some(extension) = extension {
        source.push_str(&extension.verifier.imports);
    } else {
        if commitment.is_some() {
            source.push_str(&config.commitment_type.use_line());
        }
        source.push_str(&config.field_type.use_line());
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
        "pub type {named_eval_type} = {runtime_named_eval_type}<{field_type}>;\n\
         pub type {sumcheck_output_type} = {runtime_sumcheck_output_type}<{field_type}>;\n\
         pub type {stage_proof_type} = {runtime_stage_proof_type}<{field_type}>;\n\n",
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
    if let Some(extension) = extension {
        source.push_str(&extension.verifier.proof_fields);
    }
    source.push_str("}\n\n");

    if let Some(extension) = extension {
        source.push_str(&extension.verifier.proof_items);
    }

    let verifier_inputs_derive = extension
        .and_then(|extension| extension.verifier.inputs_derive.as_deref())
        .unwrap_or("#[derive(Clone, Copy, Debug)]");
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
    if let Some(extension) = extension {
        source.push_str(&extension.verifier.input_fields);
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
    if let Some(extension) = extension {
        source.push_str(&extension.verifier.program_fields);
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
    if let Some(extension) = extension {
        source.push_str(&extension.verifier.default_program_fields);
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
    if let Some(extension) = extension {
        source.push_str(&extension.verifier.error_variants);
    }
    source.push_str("}\n\n");

    if let Some(extension) = extension {
        source.push_str(&extension.verifier.error_items);
    }

    source.push_str(&format!(
        "macro_rules! define_{protocol_snake}_verify_error_from {{\n    ($module:ident, $error_ty:ident, $variant:ident) => {{\n        impl From<$module::$error_ty> for {verify_error_type} {{\n            fn from(error: $module::$error_ty) -> Self {{\n                Self::$variant(error)\n            }}\n        }}\n    }};\n}}\n\n"
    ));
    if let Some(commitment) = &commitment {
        source.push_str(&format!(
            "define_{protocol_snake}_verify_error_from!({module}, {error}, {variant});\n",
            module = commitment.module_alias,
            error = commitment.error_type,
            variant = commitment.variant_name,
        ));
    }
    for stage in &stages {
        source.push_str(&format!(
            "define_{protocol_snake}_verify_error_from!({}, {}, {});\n",
            stage.module_alias, stage.error_type, stage.variant_name
        ));
    }
    if commitment.is_some() || !stages.is_empty() {
        source.push('\n');
    }
    if let Some(extension) = extension {
        source.push_str(&extension.verifier.error_conversions);
    }

    source.push_str(&format!(
        "pub fn verify_{protocol_snake}<T>(\n    proof: &{proof_type},\n    inputs: {verifier_inputs_type}<'_>,\n    transcript: &mut T,\n) -> Result<{verification_artifacts_type}, {verify_error_type}>\nwhere\n    T: {transcript_trait}<Challenge = {field_type}>,\n{{\n",
    ));
    source.push_str(&format!(
        "    verify_{protocol_snake}_with_programs(proof, inputs, default_verifier_programs(), transcript)\n}}\n\n"
    ));
    if let Some(extension) = extension {
        source.push_str(&extension.verifier.after_default_verify);
    }
    source.push_str(&format!(
        "pub fn verify_{protocol_snake}_with_programs<T>(\n    proof: &{proof_type},\n    inputs: {verifier_inputs_type}<'_>,\n    programs: {verifier_programs_type},\n    transcript: &mut T,\n) -> Result<{verification_artifacts_type}, {verify_error_type}>\nwhere\n    T: {transcript_trait}<Challenge = {field_type}>,\n{{\n",
    ));
    if let Some(extension) = extension {
        source.push_str(&extension.verifier.with_programs_body_intro);
    }
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
        let verifier_fn = stage
            .with_program_verifier_fn
            .as_deref()
            .or(stage.verifier_fn.as_deref())
            .unwrap_or("missing_verifier_function");
        let mut args = vec![format!("&proof.{}", stage.field_name)];
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
    if let Some(extension) = extension {
        source.push_str(&extension.verifier.after_stage_verification);
    }
    source.push_str(&format!("\n    Ok({verification_artifacts_type} {{\n"));
    if let Some(commitment) = &commitment {
        source.push_str(&format!("        {},\n", commitment.field_name));
    }
    for stage in &stages {
        source.push_str(&format!("        {},\n", stage.field_name));
    }
    source.push_str("    })\n}\n\n");

    if let Some(extension) = extension {
        source.push_str(&extension.verifier.helper_items);
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
    let extension = active_role_api_extension(config, &stages, &commitment, artifacts);
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
    if let Some(extension) = extension {
        source.push_str(&extension.prover.imports);
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
                .unwrap_or("missing_kernel_crate");
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
    if let Some(extension) = extension {
        source.push_str(&extension.prover.input_fields);
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
    if let Some(extension) = extension {
        source.push_str(&extension.prover.program_fields);
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
    if let Some(extension) = extension {
        source.push_str(&extension.prover.default_program_fields);
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
    if let Some(extension) = extension {
        source.push_str(&extension.prover.error_variants);
    }
    source.push_str("}\n\n");

    if let Some(extension) = extension {
        source.push_str(&extension.prover.error_items);
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
    if let Some(extension) = extension {
        source.push_str(&extension.prover.error_conversions);
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
    if let Some(extension) = extension {
        source.push_str(&extension.prover.after_stage_execution);
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
    if let Some(extension) = extension {
        source.push_str(&extension.prover.proof_fields);
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

    if let Some(extension) = extension {
        source.push_str(&extension.prover.helper_items);
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
        opening_input_type: has_public_type_name(source, &opening_input_name)
            .then_some(opening_input_name),
        ram_data_type: has_public_type_name(source, &ram_data_name).then_some(ram_data_name),
        verifier_data_type: has_public_type_name(source, &verifier_data_name)
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

fn active_role_api_extension<'a>(
    config: &'a ProtocolArtifactConfig,
    stages: &[StageRustApi],
    commitment: &Option<CommitmentRustApi>,
    artifacts: &[ProtocolRustArtifact],
) -> Option<&'a ProtocolArtifactExtension> {
    let extension = config.role_api_extension.as_ref()?;
    if extension.required_commitment && commitment.is_none() {
        return None;
    }
    if !extension
        .required_proof_stages
        .iter()
        .all(|required| stages.iter().any(|stage| &stage.field_name == required))
    {
        return None;
    }
    if !extension.required_artifact_stages.iter().all(|required| {
        artifacts
            .iter()
            .any(|artifact| artifact.stage.module_name() == required)
    }) {
        return None;
    }
    Some(extension)
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

fn has_public_type_name(source: &str, type_name: &str) -> bool {
    source.contains(&format!("pub struct {type_name}"))
        || source.contains(&format!("pub type {type_name}"))
        || source.contains(&format!(" as {type_name}"))
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
