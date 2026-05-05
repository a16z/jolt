use std::path::{Component, Path};

use super::super::EmitError;

pub(super) fn upper_camel(name: &str) -> String {
    let mut output = String::new();
    for segment in name.split('_') {
        let mut chars = segment.chars();
        if let Some(first) = chars.next() {
            output.extend(first.to_uppercase());
            output.push_str(chars.as_str());
        }
    }
    output
}

pub(super) fn snake_case(name: &str) -> String {
    let mut output = String::new();
    for (index, character) in name.chars().enumerate() {
        if character.is_ascii_uppercase() {
            if index != 0 {
                output.push('_');
            }
            output.extend(character.to_lowercase());
        } else if character == '-' || character == ' ' {
            output.push('_');
        } else {
            output.push(character);
        }
    }
    output
}

pub(super) fn rust_crate_ident(package_name: &str) -> String {
    package_name.replace('-', "_")
}

pub(super) fn byte_string_literal(label: &str) -> String {
    let escaped = label.escape_default().to_string();
    format!("b\"{escaped}\"")
}

pub(super) fn generated_file_path(
    root: &Path,
    relative_path: &str,
) -> Result<std::path::PathBuf, EmitError> {
    let path = Path::new(relative_path);
    if path.is_absolute()
        || path.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
    {
        return Err(EmitError::new(format!(
            "generated crate file path `{relative_path}` must be relative and stay inside the crate"
        )));
    }
    Ok(root.join(path))
}
