#![expect(
    dead_code,
    reason = "the shared support module is compiled into every integration-test target but only partially used per feature configuration."
)]

#[path = "completeness/mod.rs"]
mod completeness;
mod support;
