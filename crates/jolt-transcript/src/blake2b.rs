//! Blake2b-256 based Fiat-Shamir transcript.

use blake2::digest::consts::U32;
use blake2::Blake2b;
use digest::Digest;

use crate::impl_transcript::impl_transcript;

type Blake2b256 = Blake2b<U32>;

impl_transcript!(Blake2bTranscript, Blake2b256, Blake2b256::new());
