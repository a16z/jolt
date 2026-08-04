//! Apple Metal compute kernels.
//!
//! This module currently contains a field-arithmetic backend and measurement
//! probes. It has no prover or sumcheck integration; [`SPEC.md`](SPEC.md) defines
//! the limits that must be measured before that interface is designed.

pub mod solinas;
