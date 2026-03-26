use crate::sequence_builder::{Blake3Compression, Blake3Keyed64Compression};

jolt_inlines_sdk::register_inlines! {
    trace_file: "blake3_trace.joltinline",
    ops: [Blake3Compression, Blake3Keyed64Compression],
}
