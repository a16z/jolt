//! Address-major bytecode read/RAF carriers and Metal workers.
//!
//! Production uses the sparse Stage-1 owner; the dense carrier remains a
//! diagnostic oracle for isolated kernel experiments.

pub mod carrier;
pub mod model;
pub mod oracle;
mod runtime;
mod stage1_topology;
pub(crate) mod worklist;
mod worklist_owner;
mod worklist_runtime;

pub use runtime::*;
pub(crate) use stage1_topology::*;
pub(crate) use worklist_owner::*;

pub(super) const SOURCE: &str = include_str!("shader.metal");

#[cfg(test)]
mod tests;
