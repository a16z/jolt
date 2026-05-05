use super::rust_str;

pub(in crate::protocols::jolt::emit::rust) fn emit_plan_array(
    name: &str,
    plan_type: &str,
    rows: impl IntoIterator<Item = String>,
) -> String {
    emit_plan_array_with_suffix(name, plan_type, rows, "\n\n")
}

pub(in crate::protocols::jolt::emit::rust) fn emit_plan_array_compact(
    name: &str,
    plan_type: &str,
    rows: impl IntoIterator<Item = String>,
) -> String {
    emit_plan_array_with_suffix(name, plan_type, rows, "\n")
}

pub(in crate::protocols::jolt::emit::rust) fn emit_rustfmt_skip_macro_plan_array(
    macro_rules: &str,
    name: &str,
    plan_type: &str,
    rows: impl AsRef<str>,
    suffix: &str,
) -> String {
    let rows = rows.as_ref();
    format!("{macro_rules}\n\n#[rustfmt::skip]\npub const {name}: &[{plan_type}] = &[\n{rows}\n];{suffix}")
}

pub(in crate::protocols::jolt::emit::rust) fn emit_str_array(
    name: &str,
    values: &[String],
) -> String {
    if values.is_empty() {
        return format!("pub const {name}: &[&str] = &[];\n\n");
    }
    if let [value] = values {
        return format!("pub const {name}: &[&str] = &[{}];\n\n", rust_str(value));
    }
    let entries = values
        .iter()
        .map(|value| format!("    {},", rust_str(value)))
        .collect::<Vec<_>>()
        .join("\n");
    format!("pub const {name}: &[&str] = &[\n{entries}\n];\n\n")
}

pub(in crate::protocols::jolt::emit::rust) fn emit_str_array_compact(
    name: &str,
    values: &[String],
) -> String {
    let entries = values
        .iter()
        .map(|value| format!("    {},", rust_str(value)))
        .collect::<Vec<_>>()
        .join("\n");
    format!("pub const {name}: &[&str] = &[\n{entries}\n];\n")
}

pub(in crate::protocols::jolt::emit::rust) fn emit_usize_array(
    name: &str,
    values: &[usize],
) -> String {
    let entries = values
        .iter()
        .map(|value| format!("    {value},"))
        .collect::<Vec<_>>()
        .join("\n");
    format!("pub const {name}: &[usize] = &[\n{entries}\n];\n\n")
}

pub(in crate::protocols::jolt::emit::rust) fn intern_str_array(
    source: &mut String,
    arrays: &mut Vec<(Vec<String>, String)>,
    name_prefix: &str,
    values: &[String],
) -> String {
    if let Some((_, name)) = arrays
        .iter()
        .find(|(existing, _)| existing.as_slice() == values)
    {
        return name.clone();
    }
    let name = format!("{name_prefix}_{}", arrays.len());
    source.push_str(&emit_str_array(&name, values));
    arrays.push((values.to_vec(), name.clone()));
    name
}

fn emit_plan_array_with_suffix(
    name: &str,
    plan_type: &str,
    rows: impl IntoIterator<Item = String>,
    suffix: &str,
) -> String {
    let rows = rows.into_iter().collect::<Vec<_>>().join("\n");
    format!("pub const {name}: &[{plan_type}] = &[\n{rows}\n];{suffix}")
}
