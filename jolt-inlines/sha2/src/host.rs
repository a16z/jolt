use crate::sequence_builder::{Sha256Compression, Sha256CompressionInitial};

jolt_inlines_sdk::register_inlines! {
    trace_file: "sha256_trace.joltinline",
    ops: [Sha256Compression, Sha256CompressionInitial],
}
