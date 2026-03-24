use jolt_field::Field;

use crate::JoltGroup;

/// Pairing-friendly group for schemes that require bilinear maps (Dory, KZG).
///
/// Not all groups need this — Pedersen commitments only require `JoltGroup`.
/// The trait is parameterised over four associated types: the scalar field,
/// G1, G2, and the target group GT.
///
/// G1, G2, and GT all implement `JoltGroup` (additive notation). GT uses
/// additive notation for uniformity, even though the underlying operation
/// is Fq12 multiplication. See `Bn254GT` for the mapping.
pub trait PairingGroup: Clone + Sync + Send + 'static {
    /// Scalar field for G1 and G2 (e.g., BN254 Fr).
    type ScalarField: Field;
    type G1: JoltGroup;
    type G2: JoltGroup;
    type GT: JoltGroup;

    /// Computes the bilinear pairing `e(g1, g2)`.
    #[must_use]
    fn pairing(g1: &Self::G1, g2: &Self::G2) -> Self::GT;

    /// Computes the multi-pairing `Π e(g1s[i], g2s[i])`.
    #[must_use]
    fn multi_pairing(g1s: &[Self::G1], g2s: &[Self::G2]) -> Self::GT;
}
