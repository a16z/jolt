mod dependencies;
mod naming;
mod roles;

use super::extensions::ProtocolArtifactExtension;
use super::generated::ProtocolRuntimeModule;
use super::references::{ProtocolCrateRef, RustTypeRef};

pub use dependencies::ProtocolStandaloneDependency;

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
    pub instrumentation_prefix: Option<String>,
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
