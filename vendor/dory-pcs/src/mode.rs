//! Mode trait for transparent vs zero-knowledge proofs.
use crate::primitives::arithmetic::{Field, Group};

/// Determines whether protocol messages are blinded (ZK) or unblinded (transparent).
pub trait Mode: 'static {
    /// Whether this mode produces blinding values that callers must retain.
    const BLINDING: bool;
    /// Sample a blinding scalar: zero in Transparent mode, random in ZK mode.
    fn sample<F: Field>() -> F;
    /// Mask a group element: identity in Transparent mode, `value + base * blind` in ZK mode.
    fn mask<G: Group>(value: G, base: &G, blind: &G::Scalar) -> G;
}

/// Transparent mode: no blinding, non-hiding proofs.
pub struct Transparent;
impl Mode for Transparent {
    const BLINDING: bool = false;
    fn sample<F: Field>() -> F {
        F::zero()
    }
    fn mask<G: Group>(value: G, _base: &G, _blind: &G::Scalar) -> G {
        value
    }
}

/// Zero-knowledge mode: samples blinds from RNG for hiding proofs.
#[cfg(feature = "zk")]
pub struct ZK;
#[cfg(feature = "zk")]
impl Mode for ZK {
    const BLINDING: bool = true;
    fn sample<F: Field>() -> F {
        F::random()
    }
    fn mask<G: Group>(value: G, base: &G, blind: &G::Scalar) -> G {
        value + base.scale(blind)
    }
}
