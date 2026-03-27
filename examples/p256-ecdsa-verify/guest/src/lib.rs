#![cfg_attr(feature = "guest", no_std)]

use jolt_inlines_p256::{ecdsa_verify, P256Fr, P256Point, UnwrapOrSpoilProof};

#[jolt::provable(heap_size = 100000, max_trace_length = 524288)]
fn p256_ecdsa_verify(z: [u64; 4], r: [u64; 4], s: [u64; 4], q: [u64; 8]) {
    let z = P256Fr::from_u64_arr(&z).unwrap_or_spoil_proof();
    let r = P256Fr::from_u64_arr(&r).unwrap_or_spoil_proof();
    let s = P256Fr::from_u64_arr(&s).unwrap_or_spoil_proof();
    let q = P256Point::from_u64_arr(&q).unwrap_or_spoil_proof();
    ecdsa_verify(z, r, s, q).unwrap_or_spoil_proof()
}
