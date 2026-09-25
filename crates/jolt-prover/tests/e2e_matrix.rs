//! The end-to-end acceptance matrix: guests × protocol modes.
//!
//! One guest table; the mode is whichever protocol this crate was compiled
//! for (Dory clear by default, Dory ZK under `zk`, Akita under `akita`), so
//! the three prover lanes in CI prove the same guests and a guest missing from
//! a lane is a build-matrix gap, not a test-file gap. Every case checks the
//! guest's output against a natively computed value, proves with the
//! optimized backend with full address-first read/write binding, and verifies
//! through the public verifier API. The clear arm deliberately overlaps the
//! jolt-verifier fixture completeness cases so the table stays identical
//! across modes. Mode-specific behavior (tampering, committed programs,
//! forced one-hot sizes) lives in `zk_e2e.rs` and `akita_e2e.rs`.

#[cfg(feature = "prover-fixtures")]
mod support;

#[cfg(feature = "prover-fixtures")]
#[expect(clippy::expect_used, reason = "end-to-end fixtures fail loudly")]
mod matrix {
    // Host-side inline registrations for the hashing guests.
    extern crate jolt_inlines_keccak256;
    extern crate jolt_inlines_sha2;

    use std::collections::BTreeMap;

    use serde::Serialize;
    use sha2::{Digest, Sha256};
    use sha3::Keccak256;

    use crate::support::GuestCase;

    fn encode<T: Serialize>(value: &T) -> Vec<u8> {
        postcard::to_stdvec(value).expect("serialize guest value")
    }

    fn sha256(bytes: &[u8]) -> [u8; 32] {
        Sha256::digest(bytes).into()
    }

    fn keccak256(bytes: &[u8]) -> [u8; 32] {
        Keccak256::digest(bytes).into()
    }

    fn message(len: usize) -> Vec<u8> {
        (0..len).map(|i| i as u8).collect()
    }

    /// Two full Keccak rate blocks as the `sha3_aligned` guest takes them; the
    /// guest hashes their little-endian bytes.
    fn keccak_blocks() -> [[u64; 17]; 2] {
        let mut blocks = [[0u64; 17]; 2];
        for (index, lane) in blocks.iter_mut().flatten().enumerate() {
            *lane = (index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        }
        blocks
    }

    fn keccak_block_bytes(blocks: &[[u64; 17]; 2]) -> Vec<u8> {
        blocks
            .iter()
            .flatten()
            .flat_map(|lane| lane.to_le_bytes())
            .collect()
    }

    /// Native replica of the btreemap guest's workload.
    fn btreemap_reference(n: u32) -> u128 {
        fn wyhash64(mut x: u64) -> u64 {
            x ^= x >> 32;
            x = x.wrapping_mul(0xd6e8_feb8_6659_fd93);
            x ^= x >> 32;
            x.wrapping_mul(0xd6e8_feb8_6659_fd93)
        }
        let mut map = BTreeMap::new();
        let inserted: Vec<u64> = (0..n).map(|i| wyhash64(u64::from(i))).collect();
        for (i, key) in inserted.iter().enumerate() {
            let _ = map.insert(*key, i as u64);
        }
        for key in &inserted[..(n / 4) as usize] {
            let _ = map.remove(key);
        }
        for i in 0..(n / 2) {
            let _ = map.insert(wyhash64(u64::from(i + n * 2)), u64::from(i + n));
        }
        let mut range_sum = 0u64;
        if let (Some((&min_key, _)), Some((&max_key, _))) =
            (map.first_key_value(), map.last_key_value())
        {
            let range_size = (max_key - min_key) / 4;
            let start = min_key + range_size;
            for (_, value) in map.range(start..start + range_size) {
                range_sum = range_sum.wrapping_add(*value);
            }
        }
        (map.len() as u128).wrapping_add(u128::from(range_sum))
    }

    macro_rules! guests {
        ($emit:ident) => {
            $emit! {
                muldiv => GuestCase {
                    inputs: encode(&[9u32, 5, 3]),
                    expected_output: Some(encode(&15u32)),
                    ..GuestCase::new("muldiv-guest")
                };
                fibonacci => GuestCase {
                    inputs: encode(&100u32),
                    expected_output: Some(encode(&354_224_848_179_261_915_075u128)),
                    ..GuestCase::new("fibonacci-guest")
                };
                // The guest stores 0x12 and 0x3456 and reloads them signed; the
                // untouched bytes read back as 0.
                memory_ops => GuestCase {
                    expected_output: Some(encode(&(0x12i32, 0u32, 0x3456i32, 0u32))),
                    ..GuestCase::new("memory-ops-guest")
                };
                // 127 bytes: one full block through the initial compression,
                // then a 63-byte tail that needs the two-block padding, so both
                // SHA-256 inline instructions run.
                sha2 => GuestCase {
                    inputs: encode(&message(127)),
                    expected_output: Some(encode(&sha256(&message(127)))),
                    ..GuestCase::new("sha2-guest")
                };
                // 300 bytes behind postcard's length prefix: two full rate
                // blocks reach the fused absorb through stack staging (the
                // unaligned path), then the padded final permutation.
                sha3 => GuestCase {
                    func: Some("sha3"),
                    inputs: encode(&message(300)),
                    expected_output: Some(encode(&keccak256(&message(300)))),
                    ..GuestCase::new("sha3-guest")
                };
                // Two aligned rate blocks fed to the fused absorb straight from
                // the caller's buffer.
                sha3_aligned => GuestCase {
                    func: Some("sha3_aligned"),
                    inputs: encode(&keccak_blocks()),
                    expected_output: Some(encode(&keccak256(&keccak_block_bytes(
                        &keccak_blocks(),
                    )))),
                    ..GuestCase::new("sha3-guest")
                };
                btreemap => GuestCase {
                    stack_size: Some(10_000),
                    inputs: encode(&50u32),
                    expected_output: Some(encode(&btreemap_reference(50))),
                    ..GuestCase::new("btreemap-guest")
                };
                stdlib => GuestCase {
                    func: Some("string_concat"),
                    std: true,
                    stack_size: Some(1 << 20),
                    inputs: encode(&16i32),
                    expected_output: Some(encode(
                        &(0..16).map(|i| i.to_string()).collect::<String>(),
                    )),
                    ..GuestCase::new("stdlib-guest")
                };
                advice_consumer => GuestCase {
                    inputs: encode(&12u64),
                    untrusted_advice: encode(&5u64),
                    trusted_advice: encode(&7u64),
                    expected_output: Some(encode(&(7u64 * 3 + 5))),
                    ..GuestCase::new("advice-consumer-guest")
                };
            }
        };
    }

    macro_rules! emit_tests {
        ($($test:ident => $case:expr;)*) => {
            $(
                #[test]
                fn $test() {
                    super::mode::prove_and_verify(&$case);
                }
            )*
        };
    }

    // The mode is part of every test's name, so a lane's output states which
    // matrix cell it proved.
    #[cfg(not(any(feature = "zk", feature = "akita")))]
    mod clear {
        use super::*;
        guests!(emit_tests);
    }

    #[cfg(all(feature = "zk", not(feature = "akita")))]
    mod zk {
        use super::*;
        guests!(emit_tests);
    }

    #[cfg(feature = "akita")]
    mod akita {
        use super::*;
        guests!(emit_tests);
    }

    #[cfg(not(feature = "akita"))]
    mod mode {
        use jolt_crypto::{Bn254G1, Pedersen};
        use jolt_dory::DoryScheme;
        use jolt_field::Fr;
        use jolt_program::execution::OwnedTrace;
        use jolt_prover::{dory, JoltBackend, JoltSharedPreprocessing, ProverConfig};
        use jolt_transcript::LegacyBlake2bTranscript as Blake2bTranscript;
        use jolt_witness::{JoltVmWitnessConfig, JoltVmWitnessInputs, TraceBackend};

        use crate::support::{self, GuestCase};

        pub fn prove_and_verify(case: &GuestCase) {
            #[cfg(feature = "zk")]
            {
                let case = case.clone();
                support::with_zk_stack(move || prove_case(&case));
            }
            #[cfg(not(feature = "zk"))]
            prove_case(case);
        }

        fn prove_case(case: &GuestCase) {
            let prepared = support::prepare(case);
            let mut config = ProverConfig::derive_compact::<Fr>(
                prepared.trace.trace.as_slice(),
                &prepared.preprocessing.memory_layout,
                prepared.preprocessing.ram.min_bytecode_address,
                prepared.preprocessing.ram.bytecode_words.len(),
                prepared.preprocessing.max_padded_trace_length,
            )
            .expect("derive config");
            config.rw_config.ram_rw_phase1_num_rounds = 0;
            config.rw_config.registers_rw_phase1_num_rounds = 0;
            let preprocessing = dory::from_shared(
                JoltSharedPreprocessing::new(prepared.preprocessing).expect("shared preprocessing"),
            )
            .expect("Dory preprocessing");
            let program_preprocessing = preprocessing
                .program_arc()
                .expect("full program preprocessing");
            let trusted = (!case.trusted_advice.is_empty()).then(|| {
                dory::commit_trusted_advice(&preprocessing, &case.trusted_advice)
                    .expect("trusted advice commitment")
            });
            let public_io = prepared.trace.device.clone();
            let witness = TraceBackend::<OwnedTrace>::from_compact(
                JoltVmWitnessConfig::new(
                    config.trace_length.ilog2() as usize,
                    config.ram_K,
                    config.one_hot_config,
                )
                .include_untrusted_advice(!case.untrusted_advice.is_empty())
                .include_trusted_advice(trusted.is_some()),
                JoltVmWitnessInputs::new(&prepared.program, &program_preprocessing, prepared.trace),
            );
            let proof = dory::prove::<Fr, DoryScheme, Pedersen<Bn254G1>, Blake2bTranscript, _>(
                &JoltBackend::optimized(),
                &preprocessing,
                &config,
                trusted.as_ref(),
                &witness,
                &public_io,
            )
            .expect("Dory proof");
            jolt_verifier::verify::<Fr, DoryScheme, Pedersen<Bn254G1>, Blake2bTranscript>(
                &preprocessing.verifier,
                &public_io,
                &proof,
                trusted.as_ref().map(|advice| &advice.commitment),
            )
            .expect("Dory proof must verify");
        }
    }

    #[cfg(feature = "akita")]
    mod mode {
        use jolt_akita::{AkitaField, AkitaScheduleArtifacts, AkitaScheme};
        use jolt_program::execution::OwnedTrace;
        use jolt_prover::akita::preprocessing::{self, AkitaTranscript, AkitaVc};
        use jolt_prover::akita::{self, JoltAkitaBackend};
        use jolt_prover::ProverConfig;
        use jolt_witness::{JoltVmWitnessConfig, JoltVmWitnessInputs, TraceBackend};

        use crate::support::{self, GuestCase};

        pub fn prove_and_verify(case: &GuestCase) {
            let prepared = support::prepare(case);
            let mut config = ProverConfig::derive_compact::<AkitaField>(
                prepared.trace.trace.as_slice(),
                &prepared.preprocessing.memory_layout,
                prepared.preprocessing.ram.min_bytecode_address,
                prepared.preprocessing.ram.bytecode_words.len(),
                prepared.preprocessing.max_padded_trace_length,
            )
            .expect("derive config");
            config.rw_config.ram_rw_phase1_num_rounds = 0;
            config.rw_config.registers_rw_phase1_num_rounds = 0;
            let untrusted_advice = !case.untrusted_advice.is_empty();
            let trusted_advice = !case.trusted_advice.is_empty();
            let preprocessing = preprocessing::preprocess_full_with_advice(
                &AkitaScheduleArtifacts::shared_from_default_directory(),
                prepared.preprocessing,
                &config,
                untrusted_advice,
                trusted_advice,
            )
            .expect("Akita preprocessing");
            let trusted = trusted_advice.then(|| {
                preprocessing::commit_trusted_advice(&preprocessing, &case.trusted_advice)
                    .expect("trusted advice commitment")
            });
            let program_preprocessing = preprocessing
                .program_arc()
                .expect("full program preprocessing");
            let public_io = prepared.trace.device.clone();
            let witness = TraceBackend::<OwnedTrace>::from_compact(
                JoltVmWitnessConfig::new(
                    config.trace_length.ilog2() as usize,
                    config.ram_K,
                    config.one_hot_config,
                )
                .include_untrusted_advice(untrusted_advice)
                .include_trusted_advice(trusted_advice),
                JoltVmWitnessInputs::new(&prepared.program, &program_preprocessing, prepared.trace),
            );
            let proof = akita::prove::<AkitaField, AkitaScheme, AkitaVc, AkitaTranscript, _>(
                &JoltAkitaBackend::optimized(),
                &preprocessing,
                &config,
                trusted.as_ref(),
                &witness,
                &public_io,
            )
            .expect("Akita proof");
            jolt_verifier::verify::<AkitaField, AkitaScheme, AkitaVc, AkitaTranscript>(
                &preprocessing.verifier,
                &public_io,
                &proof,
                trusted.as_ref().map(|object| &object.commitment),
            )
            .expect("Akita proof must verify");
        }
    }
}
