#![no_main]
use jolt_eval::invariant::jolt_crypto_scalar_decomp::JoltCryptoScalarDecompInvariant;
jolt_eval::fuzz_invariant!(JoltCryptoScalarDecompInvariant::default());
