//! Univariate skip optimization for the first sumcheck round.
//!
//! In the standard sumcheck protocol, the first round iterates over
//! $2^{n-1}$ evaluation pairs. When the domain is structured (e.g.,
//! a subgroup of size $N$), the first round polynomial can be computed
//! via univariate techniques in $O(N)$ instead of $O(2^{n-1})$.
//!
//! This module defines the strategy enum. The actual optimization
//! is a future extension point.

/// Strategy for the first round of the outer sumcheck.
///
/// Controls whether the prover uses the standard Boolean-hypercube
/// enumeration or a univariate skip that exploits subgroup structure.
#[derive(Clone, Copy, Debug, Default)]
pub enum FirstRoundStrategy {
    /// Standard first-round evaluation over the Boolean hypercube.
    ///
    /// Iterates over all $2^{n-1}$ variable assignments. This is the
    /// baseline approach with no algebraic shortcuts.
    #[default]
    Standard,

    /// Univariate skip: evaluate the first-round polynomial via a
    /// degree-$d$ univariate interpolation over a multiplicative
    /// subgroup of the given size.
    ///
    /// Requires `domain_size` to be a power of two and at least as
    /// large as the number of constraints.
    UnivariateSkip {
        /// Size of the multiplicative subgroup used for interpolation.
        domain_size: usize,
    },
}
