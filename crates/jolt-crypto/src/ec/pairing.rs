use jolt_field::JoltField;
use std::fmt::Debug;

use super::group::JoltGroup;

/// Pairing-friendly group for schemes that require bilinear maps (Dory, KZG).
///
/// Not all groups need this — Pedersen commitments only require `JoltGroup`.
/// The trait is parameterised over four associated types: the scalar field,
/// G1, G2, and the target group GT.
///
/// G1, G2, and GT all implement `JoltGroup` (additive notation). GT uses
/// additive notation for uniformity, even though the underlying operation
/// is Fq12 multiplication. See `Bn254GT` for the mapping.
pub trait PairingGroup: Clone + Debug + Eq + Sync + Send + 'static {
    /// Scalar field for G1 and G2 (e.g., BN254 Fr).
    type ScalarField: JoltField;
    type G1: JoltGroup;
    type G1Affine: Clone
        + Copy
        + Debug
        + Eq
        + Send
        + Sync
        + serde::Serialize
        + for<'de> serde::Deserialize<'de>;
    type G2: JoltGroup;
    type GT: JoltGroup;

    /// Batch-converts G1 elements to the backend's persistent MSM form.
    #[must_use]
    fn g1_to_affine(bases: &[Self::G1]) -> Vec<Self::G1Affine>;

    /// Converts one prepared G1 element back to the group representation.
    #[must_use]
    fn g1_from_affine(base: &Self::G1Affine) -> Self::G1;

    /// Computes a G1 MSM over prepared bases.
    #[must_use]
    fn g1_affine_msm(bases: &[Self::G1Affine], scalars: &[Self::ScalarField]) -> Self::G1;

    /// Computes a G1 MSM using the pairing backend's scalar representation.
    #[must_use]
    fn g1_msm(bases: &[Self::G1], scalars: &[Self::ScalarField]) -> Self::G1 {
        Self::g1_affine_msm(&Self::g1_to_affine(bases), scalars)
    }

    /// Computes the bilinear pairing `e(g1, g2)`.
    #[must_use]
    fn pairing(g1: &Self::G1, g2: &Self::G2) -> Self::GT;

    /// Computes the multi-pairing `Π e(g1s[i], g2s[i])`.
    #[must_use]
    fn multi_pairing(g1s: &[Self::G1], g2s: &[Self::G2]) -> Self::GT;
}
