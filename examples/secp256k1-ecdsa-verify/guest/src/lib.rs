#![cfg_attr(feature = "guest", no_std)]

use jolt_inlines_secp256k1::{ecdsa_verify, Secp256k1Error, Secp256k1Fr, Secp256k1Point};

// verifies an secp256k1 ECDSA signature
// given message hash z, signature (r, s), and public key Q
// all inputs are little-endian u64 arrays in normal form
// returns Ok(()) if signature is valid, Err(SignatureError) otherwise
#[jolt::provable(memory_size = 100000, max_trace_length = 400000)]
fn secp256k1_ecdsa_verify(
    z: [u64; 4],
    r: [u64; 4],
    s: [u64; 4],
    q: [u64; 8],
) -> Result<(), Secp256k1Error> {
    let z = Secp256k1Fr::from_u64_arr(&z)?;
    let r = Secp256k1Fr::from_u64_arr(&r)?;
    let s = Secp256k1Fr::from_u64_arr(&s)?;
    let q = Secp256k1Point::from_u64_arr(&q)?;
    ecdsa_verify(z, r, s, q)
}
