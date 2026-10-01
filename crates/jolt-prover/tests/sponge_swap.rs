//! The proof sponge is a type parameter: the stock prover and verifier run end
//! to end on Keccak, and the sponge's identity separates proofs, so a Keccak
//! proof does not verify under Blake2b.

#[cfg(all(
    feature = "prover-fixtures",
    not(feature = "akita"),
    not(feature = "field-inline")
))]
mod support;

#[cfg(all(
    feature = "prover-fixtures",
    not(feature = "akita"),
    not(feature = "field-inline")
))]
#[expect(clippy::expect_used, reason = "end-to-end fixtures fail loudly")]
mod keccak {
    use jolt_crypto::{Bn254G1, Pedersen};
    use jolt_dory::DoryScheme;
    use jolt_field::Fr;
    use jolt_program::execution::OwnedTrace;
    use jolt_prover::{dory, JoltBackend, JoltSharedPreprocessing, ProverConfig};
    use jolt_transcript::{Blake2b512, Keccak};
    use jolt_witness::{JoltVmWitnessConfig, JoltVmWitnessInputs, TraceBackend};

    use crate::support::{self, GuestCase};

    #[test]
    fn muldiv_proves_and_verifies_on_keccak_only() {
        let case = GuestCase {
            inputs: postcard::to_stdvec(&[9u32, 5u32, 3u32]).expect("serialize inputs"),
            ..GuestCase::new("muldiv-guest")
        };
        let prepared = support::prepare(&case);
        let config = ProverConfig::derive_compact::<Fr>(
            prepared.trace.trace.as_slice(),
            &prepared.preprocessing.memory_layout,
            prepared.preprocessing.ram.min_bytecode_address,
            prepared.preprocessing.ram.bytecode_words.len(),
            prepared.preprocessing.max_padded_trace_length,
        )
        .expect("derive config");
        let preprocessing = dory::from_shared(
            JoltSharedPreprocessing::new(prepared.preprocessing).expect("shared preprocessing"),
        )
        .expect("Dory preprocessing");
        let program_preprocessing = preprocessing
            .program_arc()
            .expect("full program preprocessing");
        let public_io = prepared.trace.device.clone();
        let witness = TraceBackend::<OwnedTrace>::from_compact(
            JoltVmWitnessConfig::new(
                config.trace_length.ilog2() as usize,
                config.ram_K,
                config.one_hot_config,
            ),
            JoltVmWitnessInputs::new(&prepared.program, &program_preprocessing, prepared.trace),
        );

        let proof = dory::prove::<Fr, DoryScheme, Pedersen<Bn254G1>, Keccak, _>(
            &JoltBackend::optimized(),
            &preprocessing,
            &config,
            None,
            &witness,
            &public_io,
        )
        .expect("Keccak proof");
        jolt_verifier::verify::<Fr, DoryScheme, Pedersen<Bn254G1>, Keccak>(
            &preprocessing.verifier,
            &public_io,
            &proof,
            None,
        )
        .expect("a Keccak proof verifies under Keccak");
        assert!(
            jolt_verifier::verify::<Fr, DoryScheme, Pedersen<Bn254G1>, Blake2b512>(
                &preprocessing.verifier,
                &public_io,
                &proof,
                None,
            )
            .is_err(),
            "a Keccak proof must not verify under Blake2b"
        );
    }
}
