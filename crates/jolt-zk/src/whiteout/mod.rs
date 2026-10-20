//! Whiteout, the lattice zero-knowledge back end.
//!
//! This module holds the ring-level building blocks moved from Akita `dev`'s
//! `akita-zk-verifier` and `akita-zk-prover` crates (LayerZero-Labs/akita#120).
//! It exposes no zero-knowledge mode yet. The verifier-side pieces build under
//! the `whiteout` feature; the prover-side sampler additionally needs
//! `whiteout-prover`, so a verifier that enables `whiteout` never compiles it.
//! Randomness here comes from a caller-supplied cryptographic RNG.

mod error;
mod ring;
#[cfg(feature = "whiteout-prover")]
mod sample;

pub use error::WhiteoutError;
pub use ring::LiftedFunctional;
#[cfg(feature = "whiteout-prover")]
pub use sample::sample_zero_constant_term;
