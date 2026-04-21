#![no_main]
use jolt_eval::invariant::jolt_crypto_batch_addition::JoltCryptoBatchAdditionInvariant;
jolt_eval::fuzz_invariant!(JoltCryptoBatchAdditionInvariant::default());
