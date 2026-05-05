use crate::ir::Role;

use super::super::super::{EmitError, RustSourceFile};
use super::super::types::{ProtocolArtifactConfig, ProtocolRustArtifact, ProtocolStage};

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
