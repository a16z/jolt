/// Minimal group operations required by MSM algorithms.
/// Implementors: curve point types (e.g., GrumpkinPoint).
pub trait MsmGroup: Clone + Sized {
    /// The additive identity (point at infinity).
    fn identity() -> Self;

    /// Check if this is the identity element.
    fn is_identity(&self) -> bool;

    /// Group addition: self + other.
    fn add(&self, other: &Self) -> Self;

    /// Negation: -self.
    fn neg(&self) -> Self;

    /// Fused double-and-add: 2*self + other.
    /// Default impl provided; override if curve has optimized version.
    #[inline(always)]
    fn double_and_add(&self, other: &Self) -> Self {
        self.add(self).add(other)
    }

    /// Point doubling: 2*self.
    /// Default impl provided; override if curve has optimized version.
    #[inline(always)]
    fn double(&self) -> Self {
        self.add(self)
    }
}

/// Interface for extracting windows of bits from scalars.
/// Abstracts away limb layout and bit-width differences.
pub trait WindowedScalar: Clone {
    /// Total number of bits in this scalar representation.
    fn bit_len(&self) -> usize;

    /// Extract `width` bits starting at bit position `offset` (LSB = bit 0).
    /// Returns the window value as u16 (max window size = 16 bits).
    fn window(&self, offset: usize, width: usize) -> u16;
}

/// Marker trait for curves with efficient GLV endomorphism.
/// Enables the GLV-accelerated MSM path.
pub trait GlvCapable: MsmGroup {
    /// Half-scalar type after decomposition (e.g., u128).
    /// Must implement Default for buffer initialization.
    type HalfScalar: WindowedScalar + Default;

    /// Full scalar type before decomposition (e.g., GrumpkinFr).
    /// Must be Clone to allow iteration over scalar slices.
    type FullScalar: Clone;

    /// Apply the curve endomorphism: φ(P) where φ(P) = [λ]P.
    fn endomorphism(&self) -> Self;

    /// Decompose scalar k into signed half-scalars.
    /// Returns [(sign1, |k1|), (sign2, |k2|)] where k ≡ k1 + k2·λ (mod n).
    fn decompose_scalar(k: &Self::FullScalar) -> [(bool, Self::HalfScalar); 2];
}
