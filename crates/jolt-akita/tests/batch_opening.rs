#![expect(clippy::expect_used, reason = "tests assert successful proof setup")]

#[path = "batch_opening/support.rs"]
mod support;

#[path = "batch_opening/native.rs"]
mod native;

#[path = "batch_opening/packed_combine.rs"]
mod packed_combine;

#[path = "batch_opening/packed_scheme.rs"]
mod packed_scheme;

#[path = "batch_opening/packed_source.rs"]
mod packed_source;
