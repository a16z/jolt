//! Utilities for Dory commitment scheme, including type-safe wrappers for BN254-specific operations.

use ark_bn254::{Bn254, Fq12, Fr};
use ark_ec::pairing::{Pairing as ArkPairing, PairingOutput};
use ark_ff::Field;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use jolt_optimizations::ExponentiationSteps;
use std::marker::PhantomData;

/// Marker trait for BN254 pairing.
pub trait IsBn254: ArkPairing {}

impl IsBn254 for Bn254 {}

/// Safe wrapper for GT scalar multiplication with exponentiation steps.
///
/// This provides a type-safe way to perform GT scalar multiplications
/// with exponentiation step extraction, which is only implemented for BN254.
pub struct GtScalarMultiplier<P: ArkPairing> {
    _phantom: PhantomData<P>,
}

impl GtScalarMultiplier<Bn254> {
    /// Performs GT scalar multiplication with exponentiation steps for BN254.
    ///
    /// This is a safe wrapper that avoids unsafe transmutes by directly
    /// working with the concrete BN254 types.
    pub fn scale_with_steps(
        gt: PairingOutput<Bn254>,
        scalar: Fr,
    ) -> (PairingOutput<Bn254>, ExponentiationSteps) {
        let fq12_val: Fq12 = gt.0;
        let steps = ExponentiationSteps::new(fq12_val, scalar.into());

        // Sanity check: verify naive pow() equals steps.result
        let scalar_bigint: ark_ff::BigInt<4> = scalar.into();
        let naive_result = fq12_val.pow(scalar_bigint);
        assert_eq!(
            naive_result, steps.result,
            "Mismatch between naive pow() and ExponentiationSteps::new result"
        );

        let result = PairingOutput(steps.result);
        (result, steps)
    }
}

/// Type-safe wrapper for Dory-specific GT operations.
///
/// This allows us to constrain operations to BN254 at the type level
/// rather than using runtime checks and unsafe transmutes.
#[derive(Debug, Clone, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
pub struct DoryGT<P: ArkPairing>(pub P::TargetField);

impl<P: ArkPairing> From<PairingOutput<P>> for DoryGT<P> {
    fn from(value: PairingOutput<P>) -> Self {
        Self(value.0)
    }
}

impl<P: ArkPairing> Default for DoryGT<P> {
    fn default() -> Self {
        use ark_ff::One;
        Self(<P::TargetField as One>::one())
    }
}

impl<P> std::iter::Sum for DoryGT<P>
where
    P: ArkPairing,
{
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::default(), |acc, x| Self(acc.0 + x.0))
    }
}

/// Extension trait for BN254-specific operations.
///
/// This trait can only be implemented for types that are proven to be BN254,
/// avoiding the need for runtime type checks.
pub trait Bn254Ops: Sized {
    fn scale_with_steps(&self, scalar: &Fr) -> (Self, ExponentiationSteps);
}

impl Bn254Ops for DoryGT<Bn254> {
    fn scale_with_steps(&self, scalar: &Fr) -> (Self, ExponentiationSteps) {
        let pairing_output = PairingOutput(self.0);
        let (result, steps) = GtScalarMultiplier::scale_with_steps(pairing_output, *scalar);
        (DoryGT(result.0), steps)
    }
}
