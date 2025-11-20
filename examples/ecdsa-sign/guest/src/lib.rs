// ECDSA signing over BN254 using arkworks (pure Rust, no C dependencies)
// NOTE: secp256k1 crate doesn't work in guest due to C library dependencies
// arkworks provides pure-Rust elliptic curve operations for RISC-V targets

use ark_bn254::{Fq, Fr, G1Projective as G1};
use ark_ec::{AffineRepr, CurveGroup, PrimeGroup};
use ark_ff::{BigInteger, Field, PrimeField, Zero};

#[jolt::provable(
    stack_size = 16777216,
    memory_size = 33554432,
    max_trace_length = 2097152
)]
fn ecdsa_sign(private_key_bytes: [u8; 32], message_hash: [u8; 32]) -> ([u8; 32], [u8; 32]) {
    // ECDSA signing: (r, s) = sign(message_hash, private_key, nonce)
    // For deterministic nonce, we use hash(private_key || message_hash)

    // Convert private key to field element
    let private_key = Fr::from_le_bytes_mod_order(&private_key_bytes);

    // Generate deterministic nonce k from private key and message
    // k = hash(private_key || message_hash) mod order
    use sha2::{Digest, Sha256};
    let mut nonce_input = [0u8; 64];
    nonce_input[..32].copy_from_slice(&private_key_bytes);
    nonce_input[32..].copy_from_slice(&message_hash);
    let mut hasher = Sha256::new();
    hasher.update(&nonce_input);
    let nonce_hash = hasher.finalize();
    let k = Fr::from_be_bytes_mod_order(&nonce_hash);

    // Ensure k is non-zero
    let k = if k.is_zero() { Fr::from(1u64) } else { k };

    // Step 1: R = k * G
    let r_point = G1::generator() * k;
    let r_point_affine = r_point.into_affine();

    // Step 2: r = R.x mod n
    let r_x: Fq = r_point_affine.x().unwrap();
    let r_x_bigint = r_x.into_bigint();
    let r_x_bytes = r_x_bigint.to_bytes_le();
    let r = Fr::from_le_bytes_mod_order(&r_x_bytes);

    // Note: In production, handle r == 0 by regenerating k
    // For this example, we'll just use Fr::from(1) if it happens
    let r = if r.is_zero() { Fr::from(1u64) } else { r };

    // Step 3: z = message_hash
    let z = Fr::from_le_bytes_mod_order(&message_hash);

    // Step 4: s = k^(-1) * (z + r * private_key)
    let k_inv = k.inverse().unwrap();
    let s = k_inv * (z + r * private_key);

    // Note: In production, handle s == 0 by regenerating k
    let s = if s.is_zero() { Fr::from(1u64) } else { s };

    // Convert to bytes
    let r_bigint = r.into_bigint();
    let s_bigint = s.into_bigint();
    let r_bytes = r_bigint.to_bytes_le();
    let s_bytes = s_bigint.to_bytes_le();

    let mut r_array = [0u8; 32];
    let mut s_array = [0u8; 32];
    let r_len = r_bytes.len().min(32);
    let s_len = s_bytes.len().min(32);
    r_array[..r_len].copy_from_slice(&r_bytes[..r_len]);
    s_array[..s_len].copy_from_slice(&s_bytes[..s_len]);

    (r_array, s_array)
}
