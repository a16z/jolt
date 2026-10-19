//! Sampling primitives. Under the `lattice` feature: constant-time discrete
//! Gaussian sampling and the joint rejection rule of Whiteout's Σ-protocol
//! (`specs/whiteout-gaussian.md`), which are prover-only and stay out of the
//! default verifier closure.
//!
//! Neither the sampler's running time, branches, nor memory addresses depend
//! on the value it returns, and [`JointRejection::accept`] reads its secret
//! inputs only through fixed-length arithmetic and constant-time selects. The
//! only value that leaves the rejection step is the accept bit. The prover's
//! retry loop around it lives in `jolt-zk::whiteout`.
//!
//! All arithmetic on secrets is 192-bit fixed point; every runtime constant
//! is computed from public parameters at 320-bit precision and floored to
//! 192 bits (the note's §2 "Constants" and "Precision").

#[cfg(feature = "lattice")]
mod fixed;
#[cfg(feature = "lattice")]
mod gaussian;
#[cfg(feature = "lattice")]
mod rejection;
#[cfg(all(test, feature = "lattice"))]
mod stats;

#[cfg(feature = "lattice")]
pub use gaussian::{GaussianSampler, GaussianWidth};
#[cfg(feature = "lattice")]
pub use rejection::{BlockParams, JointRejection};

#[cfg(feature = "lattice")]
use std::error::Error;
#[cfg(feature = "lattice")]
use std::fmt::{Display, Formatter, Result as FmtResult};

/// Errors of the sampling layer. Every variant except
/// [`SampleError::ShiftOutOfRange`] concerns public parameters or shapes.
#[cfg(feature = "lattice")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SampleError {
    /// A Gaussian width multiplier outside `[1, 2^31]`, or a requested width
    /// that is not a positive finite number.
    WidthOutOfRange,
    /// A rejection rule with no Gaussian-masked block.
    NoBlocks,
    /// `α = 0` or `ε_bits = 0`.
    BadRejectionParameters,
    /// The block parameters admit an `i128` overflow in the rejection
    /// exponent, or a denominator `Q ≥ 2^{126}`.
    ArithmeticBound,
    /// `z` or `v` does not have the total length of the blocks.
    LengthMismatch { expected: usize, got: usize },
    /// Some `|v_i|` exceeds its block's `shift_bound`. An honest `v = cs`
    /// never does; the check is evaluated over every coordinate in constant
    /// time and only its overall result is branched on.
    ShiftOutOfRange,
}

#[cfg(feature = "lattice")]
impl Display for SampleError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            Self::WidthOutOfRange => write!(f, "Gaussian width out of range"),
            Self::NoBlocks => write!(f, "rejection rule has no blocks"),
            Self::BadRejectionParameters => write!(f, "alpha and epsilon bits must be positive"),
            Self::ArithmeticBound => {
                write!(f, "block parameters exceed the rejection arithmetic bound")
            }
            Self::LengthMismatch { expected, got } => {
                write!(f, "expected {expected} coordinates, got {got}")
            }
            Self::ShiftOutOfRange => write!(f, "shift coordinate exceeds its block bound"),
        }
    }
}

#[cfg(feature = "lattice")]
impl Error for SampleError {}
