//! Address-major bytecode read/RAF carriers and Metal workers.

pub mod carrier;
mod stage1_topology;
pub(crate) mod worklist;
mod worklist_owner;
mod worklist_runtime;

pub(crate) use stage1_topology::*;
pub(crate) use worklist_owner::*;

pub(super) const SOURCE: &str = include_str!("shader.metal");

#[cfg(test)]
mod tests;
