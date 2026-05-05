use super::super::names::RoleApiNames;

pub(super) fn push_type_aliases(
    source: &mut String,
    names: &RoleApiNames,
    default_transcript_type: &str,
    field_type: &str,
) {
    source.push_str(&format!(
        "pub type {} = {default_transcript_type}<{field_type}>;\n\n",
        names.default_transcript_alias
    ));
}
