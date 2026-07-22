#![cfg_attr(not(feature = "host"), no_std)]

pub use jolt_platform::{spoil_proof, UnwrapOrSpoilProof};
pub use spec::InlineReference;

#[cfg(feature = "test-utils")]
pub use spec::{
    assert_edge_cases_match_reference, assert_random_cases_match_reference,
    assert_reference_matches_harness, InlineSpec,
};

/// Decode a GLV/Fake-GLV sign word: must be exactly 0 (positive) or 1 (negative).
/// Returns `None` for any other value.
pub fn decode_sign_word(w: u64) -> Option<bool> {
    match w {
        0 => Some(false),
        1 => Some(true),
        _ => None,
    }
}

#[cfg(feature = "elliptic-curve")]
pub mod ec;

#[cfg(feature = "host")]
pub mod host;

pub mod spec;
