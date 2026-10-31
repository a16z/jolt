#![cfg_attr(feature = "guest", no_std)]

use jolt_inlines_secp256k1::{ecdsa_verify, Secp256k1Fr, Secp256k1Point, UnwrapOrSpoilProof};

#[jolt::provable]
fn verify_batch(count: u32, z: [u64; 4], r: [u64; 4], s: [u64; 4], q: [u64; 8]) {
    for _ in 0..count {
        let z = Secp256k1Fr::from_u64_arr(&z).unwrap_or_spoil_proof();
        let r = Secp256k1Fr::from_u64_arr(&r).unwrap_or_spoil_proof();
        let s = Secp256k1Fr::from_u64_arr(&s).unwrap_or_spoil_proof();
        let q = Secp256k1Point::from_u64_arr(&q).unwrap_or_spoil_proof();
        ecdsa_verify(z, r, s, q).unwrap_or_spoil_proof();
    }
}
