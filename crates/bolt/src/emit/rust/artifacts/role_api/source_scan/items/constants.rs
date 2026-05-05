pub(in crate::emit::rust::artifacts::role_api) fn find_public_const_of_type(
    source: &str,
    type_name: &str,
) -> Option<String> {
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
