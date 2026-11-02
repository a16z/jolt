//! The hash behind every digest a statement or a key binds.
//!
//! SHA3-256 (FIPS 202) by default. The `keccak256-digests` feature selects
//! Keccak-256 with the original `0x01` pad byte, the hash Ethereum uses, for
//! applications that need it. The two share Keccak-f[1600] and differ only in
//! that pad byte, so a guest with a Keccak permutation instruction pays the
//! same for either. Cargo unifies features across a build, so a build selects
//! exactly one. [`PROTOCOL_ID`](crate::PROTOCOL_ID) names the selection, so a
//! proof made under one never verifies under the other.

#[cfg(feature = "keccak256-digests")]
use sha3::Keccak256;
#[cfg(not(feature = "keccak256-digests"))]
use sha3::Sha3_256;

/// The hashes a build can select for its digests.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DigestHash {
    /// SHA3-256 (FIPS 202), the default.
    Sha3_256,
    /// Keccak-256 with the `0x01` pad byte, under `keccak256-digests`.
    Keccak256,
}

/// The digest hash this build selects.
#[cfg(not(feature = "keccak256-digests"))]
pub const DIGEST_HASH: DigestHash = DigestHash::Sha3_256;
/// The digest hash this build selects.
#[cfg(feature = "keccak256-digests")]
pub const DIGEST_HASH: DigestHash = DigestHash::Keccak256;

/// The hasher for [`DIGEST_HASH`].
#[cfg(not(feature = "keccak256-digests"))]
pub type DigestHasher = Sha3_256;
/// The hasher for [`DIGEST_HASH`].
#[cfg(feature = "keccak256-digests")]
pub type DigestHasher = Keccak256;

#[cfg(test)]
mod tests {
    use sha3::Digest;

    use super::*;

    /// FIPS 202 and Keccak team known answers for the empty input.
    #[test]
    fn digest_hasher_is_the_selected_hash() {
        let expected: [u8; 32] = match DIGEST_HASH {
            DigestHash::Sha3_256 => [
                0xa7, 0xff, 0xc6, 0xf8, 0xbf, 0x1e, 0xd7, 0x66, 0x51, 0xc1, 0x47, 0x56, 0xa0, 0x61,
                0xd6, 0x62, 0xf5, 0x80, 0xff, 0x4d, 0xe4, 0x3b, 0x49, 0xfa, 0x82, 0xd8, 0x0a, 0x4b,
                0x80, 0xf8, 0x43, 0x4a,
            ],
            DigestHash::Keccak256 => [
                0xc5, 0xd2, 0x46, 0x01, 0x86, 0xf7, 0x23, 0x3c, 0x92, 0x7e, 0x7d, 0xb2, 0xdc, 0xc7,
                0x03, 0xc0, 0xe5, 0x00, 0xb6, 0x53, 0xca, 0x82, 0x27, 0x3b, 0x7b, 0xfa, 0xd8, 0x04,
                0x5d, 0x85, 0xa4, 0x70,
            ],
        };
        assert_eq!(DigestHasher::digest(b"")[..], expected);
    }
}
