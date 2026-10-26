//! The hash behind every digest a statement or a key binds.
//!
//! SHA3-256 (FIPS 202) by default. The `keccak256-digests` feature selects
//! Keccak-256 with the original `0x01` pad byte, the hash Ethereum uses, for
//! applications that need it. The two share Keccak-f[1600] and differ only in
//! that pad byte, so a guest with a Keccak permutation instruction pays the
//! same for either. Cargo unifies features across a build, so a build selects
//! exactly one. [`PROTOCOL_ID`](crate::PROTOCOL_ID) names the selection, so a
//! proof made under one never verifies under the other.

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
pub type DigestHasher = sha3::Sha3_256;
/// The hasher for [`DIGEST_HASH`].
#[cfg(feature = "keccak256-digests")]
pub type DigestHasher = sha3::Keccak256;

#[cfg(test)]
mod tests {
    use sha3::Digest;

    use super::*;

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    /// FIPS 202 and Keccak team known answers for the empty input.
    #[test]
    fn digest_hasher_is_the_selected_hash() {
        let expected = match DIGEST_HASH {
            DigestHash::Sha3_256 => {
                "a7ffc6f8bf1ed76651c14756a061d662f580ff4de43b49fa82d80a4b80f8434a"
            }
            DigestHash::Keccak256 => {
                "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470"
            }
        };
        assert_eq!(hex(&DigestHasher::digest(b"")), expected);
    }
}
