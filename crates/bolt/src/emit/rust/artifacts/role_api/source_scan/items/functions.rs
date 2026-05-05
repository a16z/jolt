pub(in crate::emit::rust::artifacts::role_api) fn find_public_item(
    source: &str,
    prefix: &str,
    suffix: &str,
) -> Option<String> {
    source.lines().find_map(|line| {
        let trimmed = line.trim_start();
        let rest = trimmed.strip_prefix(prefix)?;
        let name = rest
            .split(|character: char| {
                matches!(character, '<' | '(' | '{') || character.is_whitespace()
            })
            .next()?;
        name.ends_with(suffix).then(|| name.to_owned())
    })
}

pub(in crate::emit::rust::artifacts::role_api) fn find_public_fn(
    source: &str,
    prefixes: &[&str],
) -> Option<String> {
    find_public_fn_matching(source, |name| {
        prefixes.iter().any(|prefix| name.starts_with(prefix))
    })
}

pub(in crate::emit::rust::artifacts::role_api) fn find_public_fn_containing(
    source: &str,
    prefixes: &[&str],
    needle: &str,
) -> Option<String> {
    find_public_fn_matching(source, |name| {
        name.contains(needle) && prefixes.iter().any(|prefix| name.starts_with(prefix))
    })
}

fn find_public_fn_matching(
    source: &str,
    mut predicate: impl FnMut(&str) -> bool,
) -> Option<String> {
    source.lines().find_map(|line| {
        let trimmed = line.trim_start();
        let rest = trimmed.strip_prefix("pub fn ")?;
        let name = rest
            .split(|character: char| matches!(character, '<' | '(') || character.is_whitespace())
            .next()?;
        predicate(name).then(|| name.to_owned())
    })
}
