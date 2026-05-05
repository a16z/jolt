mod context;
mod declarations;
mod discovery;
mod imports;
mod items;
mod lib_module;
mod names;
mod programs;
mod prover;
mod role;
mod source_scan;
mod types;
mod verifier;

use crate::ir::Role;

use super::types::{GeneratedFile, ProtocolArtifactConfig, ProtocolRustArtifact};
pub(super) use lib_module::generated_lib;
use role::RoleApiRole;
use types::{
    CommitmentRustApi, RoleApiProgram, RoleApiProgramBinding, RoleApiProgramSource, StageRustApi,
    VerifierStageInputKind,
};

pub(super) fn generated_role_api_file(
    config: &ProtocolArtifactConfig,
    role: &Role,
    artifacts: &[ProtocolRustArtifact],
) -> GeneratedFile {
    let role = RoleApiRole::from_role(role);
    GeneratedFile {
        path: role.source_path().to_owned(),
        source: generated_role_api_source(config, role, artifacts),
    }
}

fn generated_role_api_source(
    config: &ProtocolArtifactConfig,
    role: RoleApiRole,
    artifacts: &[ProtocolRustArtifact],
) -> String {
    match role {
        RoleApiRole::Prover => prover::generated_prover_api(config, artifacts),
        RoleApiRole::Verifier => verifier::generated_verifier_api(config, artifacts),
    }
}
