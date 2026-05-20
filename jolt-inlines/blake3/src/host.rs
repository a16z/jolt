use crate::sequence_builder::{Blake3Compression, Blake3Keyed64Compression};

jolt_inlines_sdk::register_inlines! {
    trace_file: "blake3_trace.joltinline",
    extension: jolt_inlines_sdk::host::InlineExtension::Blake3,
    ops: [Blake3Compression, Blake3Keyed64Compression],
}
