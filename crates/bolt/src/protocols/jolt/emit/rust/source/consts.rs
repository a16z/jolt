use super::rust_str;

pub(in crate::protocols::jolt::emit::rust) fn emit_params_const(
    name: &str,
    params_type: &str,
    field: &str,
    pcs: &str,
    transcript: &str,
) -> String {
    let field = rust_str(field);
    let pcs = rust_str(pcs);
    let transcript = rust_str(transcript);
    emit_struct_const(
        name,
        params_type,
        &[
            ("field", &field),
            ("pcs", &pcs),
            ("transcript", &transcript),
        ],
    )
}

pub(in crate::protocols::jolt::emit::rust) fn emit_struct_const(
    name: &str,
    type_name: &str,
    fields: &[(&str, &str)],
) -> String {
    emit_struct_const_with_literal(name, type_name, type_name, fields)
}

pub(in crate::protocols::jolt::emit::rust) fn emit_struct_const_with_literal(
    name: &str,
    const_type: &str,
    literal_type: &str,
    fields: &[(&str, &str)],
) -> String {
    let fields = fields
        .iter()
        .map(|(field, value)| format!("    {field}: {value},"))
        .collect::<Vec<_>>()
        .join("\n");
    format!("pub const {name}: {const_type} = {literal_type} {{\n{fields}\n}};\n")
}

pub(in crate::protocols::jolt::emit::rust) fn emit_inline_struct_const(
    name: &str,
    type_name: &str,
    fields: &[(&str, &str)],
    suffix: &str,
) -> String {
    let fields = fields
        .iter()
        .map(|(field, value)| format!("{field}: {value}"))
        .collect::<Vec<_>>()
        .join(", ");
    format!("pub const {name}: {type_name} = {type_name} {{ {fields} }};{suffix}")
}

pub(in crate::protocols::jolt::emit::rust) fn emit_value_const(
    name: &str,
    type_name: &str,
    value: &str,
    suffix: &str,
) -> String {
    format!("pub const {name}: {type_name} = {value};{suffix}")
}
