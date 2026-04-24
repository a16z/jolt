//! Polynomial data providers and witness buffers for the Jolt zkVM.
//!
//! This crate is the data layer between trace backends and the proving
//! pipeline. It materializes committed polynomial evaluation buffers from
//! per-cycle execution data and provides on-demand computation of derived,
//! preprocessed, and bytecode polynomials.
//!
//! # Usage
//!
//! ```ignore
//! let mut polys = Polynomials::new(config);
//! polys.push(&cycles);       // batch or streaming
//! polys.finish();
//! let rd_inc = polys.get(PolynomialId::RdInc);
//! ```

pub mod bytecode_raf;
mod config;
mod cycle_input;
pub mod derived;
pub mod field_reg;
pub mod polynomial_id;
mod polynomials;
pub mod preprocessed;
pub mod provider;

pub use config::{PolynomialConfig, LOG_K_FR};
pub use cycle_input::CycleInput;
pub use field_reg::{replay_field_regs, FieldRegEvent, FrCycleBytecode, FrCycleData, FrLimbs};
pub use polynomial_id::PolynomialId;
pub use polynomials::Polynomials;
