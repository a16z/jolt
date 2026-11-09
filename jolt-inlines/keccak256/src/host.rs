use crate::sequence_builder::{
    Keccak256AbsorbDigest, Keccak256AbsorbPermutation, Keccak256Digest, Keccak256Permutation,
};

jolt_inlines_sdk::register_inlines! {
    trace_file: "keccak256_trace.joltinline",
    extension: jolt_inlines_sdk::host::InlineExtension::Keccak256,
    ops: [Keccak256Permutation, Keccak256AbsorbPermutation, Keccak256Digest, Keccak256AbsorbDigest],
}
