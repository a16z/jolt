use super::discovery::aliased_modules;

pub(in crate::emit::rust::artifacts::role_api) fn push_stage_imports(
    source: &mut String,
    modules: &[String],
) {
    if !modules.is_empty() {
        source.push_str(&format!(
            "use crate::stages::{{{}}};\n\n",
            aliased_modules(modules).join(", ")
        ));
    }
}
