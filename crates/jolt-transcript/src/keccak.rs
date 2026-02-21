//! Keccak-256 based Fiat-Shamir transcript (Ethereum/EVM compatible).

use digest::Digest;
use sha3::Keccak256;

use crate::impl_transcript::impl_transcript;

impl_transcript!(KeccakTranscript, Keccak256, Keccak256::new());
