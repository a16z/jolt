pub(in crate::protocols::jolt::emit::rust) fn rust_str(value: &str) -> String {
    format!("{value:?}")
}

pub(in crate::protocols::jolt::emit::rust) fn rust_option_str(value: Option<&str>) -> String {
    value.map_or_else(
        || "None".to_owned(),
        |value| format!("Some({})", rust_str(value)),
    )
}

pub(in crate::protocols::jolt::emit::rust) fn rust_str_array(values: &[String]) -> String {
    let values = values
        .iter()
        .map(|value| rust_str(value))
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{values}]")
}
