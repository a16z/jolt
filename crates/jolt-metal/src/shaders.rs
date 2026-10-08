//! MSL headers for the types in [`field`](crate::field), their accumulators,
//! and the simdgroup and threadgroup reductions over accumulators.
//!
//! Libraries are compiled from source at runtime, where `#include` of a
//! repository path does not resolve. Add these units to a
//! [`LibrarySpec`](crate::runtime::LibrarySpec), in order, before any source
//! that uses them.

/// `(name, text)` of every field header, in dependency order.
pub const FIELD_HEADERS: [(&str, &str); 6] = [
    (
        "jolt/field/fp128.h",
        include_str!("../shaders/jolt/field/fp128.h"),
    ),
    (
        "jolt/field/fp64.h",
        include_str!("../shaders/jolt/field/fp64.h"),
    ),
    (
        "jolt/field/ext2.h",
        include_str!("../shaders/jolt/field/ext2.h"),
    ),
    (
        "jolt/field/accum.h",
        include_str!("../shaders/jolt/field/accum.h"),
    ),
    (
        "jolt/field/fp128_accum.h",
        include_str!("../shaders/jolt/field/fp128_accum.h"),
    ),
    (
        "jolt/field/reduce.h",
        include_str!("../shaders/jolt/field/reduce.h"),
    ),
];
