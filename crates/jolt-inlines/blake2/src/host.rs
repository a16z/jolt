use crate::sequence_builder::Blake2bCompression;

jolt_inlines_sdk::register_inlines! {
    trace_file: "blake2_trace.joltinline",
    extension: jolt_inlines_sdk::host::InlineExtension::Blake2,
    ops: [Blake2bCompression],
}
