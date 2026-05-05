use super::super::types::{
    ProtocolArtifactConfig, ProtocolArtifactExtension, ProtocolRustArtifact,
};
use super::declarations::RoleDeclarationTypes;
use super::discovery::{active_role_api_extension, commitment_api, role_modules, stage_apis};
use super::names::RoleApiNames;
use super::role::RoleApiRole;
use super::{CommitmentRustApi, StageRustApi};

pub(super) struct RoleApiSourceContext<'a> {
    pub(super) stages: Vec<StageRustApi>,
    pub(super) modules: Vec<String>,
    pub(super) commitment: Option<CommitmentRustApi>,
    pub(super) extension: Option<&'a ProtocolArtifactExtension>,
    pub(super) names: RoleApiNames,
    pub(super) protocol_snake: String,
    pub(super) field_type: String,
    pub(super) transcript_trait: String,
}

impl<'a> RoleApiSourceContext<'a> {
    pub(super) fn new(
        config: &'a ProtocolArtifactConfig,
        artifacts: &[ProtocolRustArtifact],
    ) -> Self {
        let stages = stage_apis(config, artifacts);
        let modules = role_modules(artifacts);
        let commitment = commitment_api(artifacts);
        let extension = active_role_api_extension(config, &stages, &commitment, artifacts);
        Self {
            stages,
            modules,
            commitment,
            extension,
            names: RoleApiNames::new(&config.type_prefix),
            protocol_snake: config.protocol_snake(),
            field_type: config.field_type.ident().to_owned(),
            transcript_trait: config.transcript_trait.ident().to_owned(),
        }
    }

    pub(super) fn declaration_types(&self, role: RoleApiRole) -> RoleDeclarationTypes<'_> {
        let names = self.names.role(role);
        RoleDeclarationTypes {
            programs: names.programs,
            artifacts: names.artifacts,
            field: &self.field_type,
        }
    }
}
