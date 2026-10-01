//! Duplex sponges a transcript can run on.

use spongefish::DuplexSpongeInterface;

/// A byte-oriented duplex sponge with a stable identity.
///
/// Protocol code is generic over `H: Sponge`; swapping the sponge is a change
/// of type argument. [`ID`](Self::ID) is bound into every
/// [`ProtocolId`](crate::ProtocolId), so a proof produced under one sponge
/// never verifies under another.
pub trait Sponge: DuplexSpongeInterface<U = u8> + Default + Clone + Send + Sync + 'static {
    /// Stable name of this sponge construction.
    const ID: &'static str;
}

#[cfg(feature = "transcript-blake2b")]
impl Sponge for spongefish::instantiations::Blake2b512 {
    const ID: &'static str = "blake2b512";
}

#[cfg(feature = "transcript-keccak")]
impl Sponge for spongefish::instantiations::Keccak {
    const ID: &'static str = "keccak-f1600";
}

#[cfg(feature = "transcript-poseidon")]
impl Sponge for crate::PoseidonSponge {
    const ID: &'static str = "poseidon-bn254-circom-t4";
}
