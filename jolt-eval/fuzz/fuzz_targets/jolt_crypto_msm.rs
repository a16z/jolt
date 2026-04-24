#![no_main]
use jolt_eval::invariant::jolt_crypto_msm::JoltCryptoMsmInvariant;
jolt_eval::fuzz_invariant!(JoltCryptoMsmInvariant::default());
