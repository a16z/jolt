use super::super::super::{CommitmentRustApi, StageRustApi};

pub(in crate::emit::rust::artifacts::role_api) fn push_verifier_error_conversions(
    source: &mut String,
    protocol_snake: &str,
    commitment: Option<&CommitmentRustApi>,
    stages: &[StageRustApi],
    verify_error_type: &str,
) {
    source.push_str(&format!(
        "macro_rules! define_{protocol_snake}_verify_error_from {{\n    ($module:ident, $error_ty:ident, $variant:ident) => {{\n        impl From<$module::$error_ty> for {verify_error_type} {{\n            fn from(error: $module::$error_ty) -> Self {{\n                Self::$variant(error)\n            }}\n        }}\n    }};\n}}\n\n"
    ));
    if let Some(commitment) = commitment {
        source.push_str(&format!(
            "define_{protocol_snake}_verify_error_from!({module}, {error}, {variant});\n",
            module = commitment.module_alias,
            error = commitment.error_type,
            variant = commitment.variant_name,
        ));
    }
    for stage in stages {
        source.push_str(&format!(
            "define_{protocol_snake}_verify_error_from!({}, {}, {});\n",
            stage.module_alias, stage.error_type, stage.variant_name
        ));
    }
    if commitment.is_some() || !stages.is_empty() {
        source.push('\n');
    }
}
