use std::fmt::Debug;
use std::ops::{Mul, MulAssign};

use rand_core::RngCore;
use serde::{Deserialize, Serialize};

use crate::JoltGroup;

/// Pairing-friendly group for schemes that require bilinear maps (Dory, KZG).
///
/// Not all groups need this — Pedersen commitments only require `JoltGroup`.
/// The trait is parameterised over three associated types: G1, G2, and the
/// target group GT. GT uses **multiplicative** notation to match the
/// mathematical convention for pairing targets.
pub trait PairingGroup: Clone + Sync + Send + 'static {
    type G1: JoltGroup;
    type G2: JoltGroup;
    type GT: Clone
        + Debug
        + Eq
        + Send
        + Sync
        + 'static
        + Mul<Output = Self::GT>
        + MulAssign
        + Serialize
        + for<'de> Deserialize<'de>;

    /// Multiplicative identity in GT.
    fn gt_one() -> Self::GT;

    /// Generator of G1.
    fn g1_generator() -> Self::G1;

    /// Generator of G2.
    fn g2_generator() -> Self::G2;

    /// Computes the bilinear pairing `e(g1, g2)`.
    fn pairing(g1: &Self::G1, g2: &Self::G2) -> Self::GT;

    /// Computes the multi-pairing `Π e(g1s[i], g2s[i])`.
    fn multi_pairing(g1s: &[Self::G1], g2s: &[Self::G2]) -> Self::GT;

    /// Samples a uniformly random G1 element.
    fn random_g1<R: RngCore>(rng: &mut R) -> Self::G1;
}
