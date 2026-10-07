//! Native CPU implementations of the Akita backend interfaces.

pub mod commitment;
mod witness;

#[cfg(feature = "field-inline")]
mod field_inline;
