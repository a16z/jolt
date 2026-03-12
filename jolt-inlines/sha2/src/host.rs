use crate::sequence_builder::{Sha256Compression, Sha256CompressionInitial};

jolt_inlines_common::register_inlines! {
    crate_name: "SHA256",
    trace_file: "sha256_trace.joltinline",
    ops: [Sha256Compression, Sha256CompressionInitial],
}
