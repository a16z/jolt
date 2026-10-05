//! MSL headers for the types in [`field`](crate::field).
//!
//! Libraries are compiled from source at runtime, where `#include` of a
//! repository path does not resolve. Add these units to a
//! [`LibrarySpec`](crate::runtime::LibrarySpec), in order, before any source
//! that uses them.

/// `(name, text)` of every field header, in dependency order.
pub const FIELD_HEADERS: [(&str, &str); 1] = [(
    "jolt/field/fp128.h",
    include_str!("../shaders/jolt/field/fp128.h"),
)];
