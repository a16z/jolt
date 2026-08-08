//! Address-major bytecode read/RAF carrier and Metal worker.
//!
//! The worker is executable in isolation. Production integration still must
//! attach [`carrier`] receipts to the stage-5 resident rows.

pub mod carrier;
pub mod model;
pub mod oracle;
mod runtime;
pub(crate) mod worklist;
mod worklist_runtime;

pub use runtime::*;

pub(super) const SOURCE: &str = include_str!("shader.metal");

#[cfg(test)]
mod tests;
