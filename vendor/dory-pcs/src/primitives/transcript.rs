//! Transcript trait for Fiat-Shamir transformations

#![allow(missing_docs)]

use crate::primitives::arithmetic::{Group, PairingCurve};
use crate::primitives::DorySerialize;

/// Transcript to standardize fiat shamir across different transcript implementations
pub trait Transcript {
    type Curve: PairingCurve;
    fn append_bytes(&mut self, label: &[u8], bytes: &[u8]);

    fn append_field(
        &mut self,
        label: &[u8],
        x: &<<Self::Curve as PairingCurve>::G1 as Group>::Scalar,
    );

    fn append_group<G: Group>(&mut self, label: &[u8], g: &G);

    fn append_serde<S: DorySerialize>(&mut self, label: &[u8], s: &S);

    fn challenge_scalar(
        &mut self,
        label: &[u8],
    ) -> <<Self::Curve as PairingCurve>::G1 as Group>::Scalar;

    fn reset(&mut self, domain_label: &[u8]);
}
