pub(in crate::emit::rust::artifacts::role_api) fn find_type_with_suffix(
    source: &str,
    suffix: &str,
) -> Option<String> {
    source
        .split(|character: char| !character.is_ascii_alphanumeric() && character != '_')
        .find(|token| token.ends_with(suffix) && token.len() > suffix.len())
        .map(ToOwned::to_owned)
}

pub(in crate::emit::rust::artifacts::role_api) fn has_public_type_name(
    source: &str,
    type_name: &str,
) -> bool {
    source.contains(&format!("pub struct {type_name}"))
        || source.contains(&format!("pub type {type_name}"))
        || source.contains(&format!(" as {type_name}"))
}
