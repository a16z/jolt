#![no_main]
use jolt_eval::invariant::jolt_crypto_glv_vector::JoltCryptoGlvVectorInvariant;
jolt_eval::fuzz_invariant!(JoltCryptoGlvVectorInvariant::default());
