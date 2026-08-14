use crate::sequence_builder::{Sha256Compression, Sha256CompressionInitial};

jolt_inlines_sdk::register_inlines! {
    trace_file: "sha256_trace.joltinline",
    extension: jolt_inlines_sdk::host::InlineExtension::Sha2,
    ops: [Sha256Compression, Sha256CompressionInitial],
}
