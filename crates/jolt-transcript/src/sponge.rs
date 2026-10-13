//! Duplex sponges a transcript can run on.

#[cfg(feature = "transcript-poseidon")]
use crate::PoseidonSponge;
#[cfg(feature = "transcript-blake2b")]
use spongefish::instantiations::Blake2b512;
#[cfg(feature = "transcript-keccak")]
use spongefish::instantiations::Keccak;
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

    /// Whether an absorb erases how many bytes were squeezed within the
    /// current output block. An overwrite-mode duplex squeezes by reading its
    /// rate without permuting, so squeezing one byte or two before the same
    /// absorb leaves the same state. On such a sponge the transcript absorbs
    /// the width of every variable-length squeeze (the exact-challenge
    /// rejection loop), so distinct retry histories cannot reconverge.
    const ABSORB_FORGETS_SQUEEZE_WIDTH: bool;
}

#[cfg(feature = "transcript-blake2b")]
impl Sponge for Blake2b512 {
    const ID: &'static str = "blake2b512";
    const ABSORB_FORGETS_SQUEEZE_WIDTH: bool = false;
}

#[cfg(feature = "transcript-keccak")]
impl Sponge for Keccak {
    const ID: &'static str = "keccak-f1600";
    const ABSORB_FORGETS_SQUEEZE_WIDTH: bool = true;
}

#[cfg(feature = "transcript-poseidon")]
impl Sponge for PoseidonSponge {
    const ID: &'static str = "poseidon-bn254-circom-t4";
    const ABSORB_FORGETS_SQUEEZE_WIDTH: bool = true;
}
