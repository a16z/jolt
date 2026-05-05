use super::super::super::role::RoleApiRole;
use super::super::super::{RoleApiProgram, StageRustApi};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ProgramFieldStyle {
    Struct,
    Default,
}

impl ProgramFieldStyle {
    pub(super) fn stage_module_alias<'a>(
        self,
        stage: &'a StageRustApi,
        role: RoleApiRole,
    ) -> &'a str {
        match self {
            Self::Struct => role.stage_module_alias(stage),
            Self::Default => stage.module_alias.as_str(),
        }
    }

    pub(super) fn push(
        self,
        source: &mut String,
        field_name: &str,
        module_alias: &str,
        program: RoleApiProgram<'_>,
    ) {
        match self {
            Self::Struct => {
                source.push_str(&format!(
                    "    pub {field_name}: &'static {module_alias}::{},\n",
                    program.type_name
                ));
            }
            Self::Default => {
                source.push_str(&format!(
                    "        {field_name}: &{module_alias}::{},\n",
                    program.const_name
                ));
            }
        }
    }
}
