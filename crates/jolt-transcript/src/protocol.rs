//! Domain separation: the identity a transcript starts from.

use crate::Sponge;

/// Byte width of a [`ProtocolId`].
pub const PROTOCOL_ID_LEN: usize = 64;

/// The 64-byte domain separator every transcript absorbs first.
///
/// Encodes `len(name) || name || len(Sponge::ID) || Sponge::ID`, zero padded.
/// Both lengths are one byte, so the encoding is injective in `(name, sponge)`.
/// Callers put everything that must separate proofs into `name`: protocol,
/// version, and proof mode (for example `"jolt/v1/zk"`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProtocolId([u8; PROTOCOL_ID_LEN]);

impl ProtocolId {
    /// Derives the identity of protocol `name` running on sponge `H`.
    ///
    /// # Panics
    ///
    /// Panics if `name` and `H::ID` do not fit in 62 bytes together. Evaluate
    /// it in a `const` item to turn that into a compile error.
    #[must_use]
    #[expect(
        clippy::indexing_slicing,
        reason = "every index is below the asserted total length, which fits the buffer"
    )]
    pub const fn new<H: Sponge>(name: &str) -> Self {
        let name = name.as_bytes();
        let sponge = H::ID.as_bytes();
        assert!(
            name.len() + sponge.len() + 2 <= PROTOCOL_ID_LEN,
            "protocol name and sponge id exceed the protocol id width"
        );
        let mut id = [0u8; PROTOCOL_ID_LEN];
        id[0] = name.len() as u8;
        let mut i = 0;
        while i < name.len() {
            id[1 + i] = name[i];
            i += 1;
        }
        let sponge_start = 1 + name.len();
        id[sponge_start] = sponge.len() as u8;
        let mut j = 0;
        while j < sponge.len() {
            id[sponge_start + 1 + j] = sponge[j];
            j += 1;
        }
        Self(id)
    }

    /// The encoded identity.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; PROTOCOL_ID_LEN] {
        &self.0
    }
}
