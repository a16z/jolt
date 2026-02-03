#![cfg_attr(feature = "guest", no_std)]

use jolt_inlines_secp256k1::{ecdsa_verify, Secp256k1Fr, Secp256k1Point, UnwrapOrSpoilProof};

// verifies an secp256k1 ECDSA signature
// given message hash z, signature (r, s), and public key Q
// all inputs are little-endian u64 arrays in normal form, validated by the ecdsa_verify function
// returns Ok(()) if signature is valid, Err(SignatureError) otherwise
#[jolt::provable(heap_size = 100000, max_trace_length = 250000)]
fn secp256k1_ecdsa_verify(z: [u64; 4], r: [u64; 4], s: [u64; 4], q: [u64; 8]) {
    let z = Secp256k1Fr::from_u64_arr(&z).unwrap_or_spoil_proof();
    let r = Secp256k1Fr::from_u64_arr(&r).unwrap_or_spoil_proof();
    let s = Secp256k1Fr::from_u64_arr(&s).unwrap_or_spoil_proof();
    let q = Secp256k1Point::from_u64_arr(&q).unwrap_or_spoil_proof();
    ecdsa_verify(z, r, s, q).unwrap_or_spoil_proof()
}
