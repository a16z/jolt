#![allow(clippy::print_stderr, clippy::print_stdout)]

pub mod baseline;
pub mod cli;
pub mod generated;
pub mod measure;
pub mod output;
pub mod programs;
#[cfg(feature = "legacy-stack-bench")]
pub mod stacks;
