//! Signed big integer types for the Jolt prover.
//!
//! These types represent signed integers with configurable bit widths using
//! sign-magnitude representation. They are used extensively in R1CS evaluation,
//! accumulation, and increment witness generation where intermediate values
//! can be negative but have bounded magnitude.
//!
//! Two families are provided:
//!
//! - [`SignedBigInt<N>`]: magnitude stored as `BigInt<N>` (width = `N * 64` bits)
//! - [`SignedBigIntHi32<N>`]: magnitude stored as `[u64; N]` + `u32` tail (width = `N * 64 + 32` bits)
//!
//! Common type aliases:
//! - `S64`, `S128`, `S192`, `S256` (from `SignedBigInt`)
//! - `S96`, `S160`, `S224` (from `SignedBigIntHi32`)

mod signed_bigint;
mod signed_bigint_hi32;

pub use signed_bigint::*;
pub use signed_bigint_hi32::*;
