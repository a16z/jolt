//! Committed polynomial buffers for the Jolt zkVM.
//!
//! This crate materializes committed polynomial evaluation buffers from
//! per-cycle execution data. It is the data layer between trace backends
//! and the proving pipeline.
//!
//! # Usage
//!
//! ```ignore
//! let mut polys = Polynomials::new(config);
//! polys.push(&cycles);       // batch or streaming
//! polys.finish();
//! let rd_inc = polys.get(PolynomialId::RdInc);
//! ```

mod config;
mod cycle_input;
pub mod polynomial_id;
mod polynomials;

pub use config::PolynomialConfig;
pub use cycle_input::CycleInput;
pub use polynomial_id::PolynomialId;
pub use polynomials::Polynomials;
