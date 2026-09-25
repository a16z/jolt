//! ZK end-to-end coverage for the modular prover and verifier: the
//! mode-specific checks (BlindFold tampering, the reference backend, the
//! unaligned SHA3 inline expansion, committed programs). Plain acceptance
//! across guests is `e2e_matrix.rs`.

#[cfg(all(
    feature = "prover-fixtures",
    feature = "zk",
    not(feature = "field-inline")
))]
mod support;

#[cfg(all(
    feature = "prover-fixtures",
    feature = "zk",
    not(feature = "field-inline")
))]
#[expect(
    clippy::expect_used,
    clippy::panic,
    reason = "integration tests should fail loudly"
)]
mod zk {
    extern crate jolt_inlines_keccak256;

    use common::jolt_device::JoltDevice;
    use jolt_crypto::{Bn254G1, Pedersen};
    use jolt_dory::{DoryCommitment, DoryScheme};
    use jolt_field::{Fr, Ring};
    use jolt_program::execution::OwnedTrace;
    use jolt_prover::dory::DoryProverPreprocessing;
    use jolt_prover::{JoltBackend, JoltSharedPreprocessing, ProverConfig};
    use jolt_riscv::{JoltInstructionKind, JoltTraceRow};
    use jolt_transcript::LegacyBlake2bTranscript as Blake2bTranscript;
    use jolt_verifier::proof::{JoltProof, JoltProofClaims};
    use jolt_verifier::VerifierError;
    use jolt_witness::{JoltVmWitnessConfig, JoltVmWitnessInputs, TraceBackend};

    use crate::support::{self, with_zk_stack, GuestCase, PreparedGuest};

    // 24 rounds x 24 ROTRI per Keccak-f permutation (theta-D XORs use VirtualXORROTL1).
    const KECCAK_ROTRI_ROWS: usize = 576;
    // The `&[u8]` guest input sits behind postcard's 2-byte length prefix, so
    // `digest` takes its unaligned path: two fused absorb-permute blocks staged
    // through stack copies, then a padded final block.
    const SHA3_INPUT_LEN: usize = 300;
    const SHA3_PERMUTATIONS: usize = 3;

    type Proof = JoltProof<DoryScheme, Pedersen<Bn254G1>>;

    struct ProvedGuest {
        preprocessing: DoryProverPreprocessing,
        public_io: JoltDevice,
        proof: Proof,
        trusted_advice_commitment: Option<DoryCommitment>,
    }

    fn muldiv_case() -> GuestCase {
        GuestCase {
            inputs: postcard::to_stdvec(&[9u32, 5u32, 3u32]).expect("serialize inputs"),
            ..GuestCase::new("muldiv-guest")
        }
    }

    fn derive_config(run: &PreparedGuest) -> ProverConfig {
        ProverConfig::derive_compact::<Fr>(
            run.trace.trace.as_slice(),
            &run.preprocessing.memory_layout,
            run.preprocessing.ram.min_bytecode_address,
            run.preprocessing.ram.bytecode_words.len(),
            run.preprocessing.max_padded_trace_length,
        )
        .expect("derive config")
    }

    fn prove_guest(
        case: GuestCase,
        backend: JoltBackend<Fr, DoryScheme>,
        inspect_trace: impl FnOnce(&[JoltTraceRow]),
    ) -> ProvedGuest {
        let run = support::prepare(&case);
        inspect_trace(run.trace.trace.as_slice());
        let config = derive_config(&run);
        let shared = JoltSharedPreprocessing::new(run.preprocessing).expect("shared preprocessing");
        let preprocessing = jolt_prover::dory::from_shared(shared).expect("Dory preprocessing");
        assert!(preprocessing.verifier.vc_setup.is_some());
        let program_preprocessing = preprocessing
            .program_arc()
            .expect("full program preprocessing");
        let public_io = run.trace.device.clone();
        let witness = TraceBackend::<OwnedTrace>::from_compact(
            JoltVmWitnessConfig::new(
                config.trace_length.ilog2() as usize,
                config.ram_K,
                config.one_hot_config,
            )
            .include_trusted_advice(!case.trusted_advice.is_empty())
            .include_untrusted_advice(!case.untrusted_advice.is_empty()),
            JoltVmWitnessInputs::new(&run.program, &program_preprocessing, run.trace),
        );
        let trusted = (!case.trusted_advice.is_empty()).then(|| {
            jolt_prover::dory::commit_trusted_advice(&preprocessing, &case.trusted_advice)
                .expect("trusted advice commitment")
        });
        let proof =
            jolt_prover::dory::prove::<Fr, DoryScheme, Pedersen<Bn254G1>, Blake2bTranscript, _>(
                &backend,
                &preprocessing,
                &config,
                trusted.as_ref(),
                &witness,
                &public_io,
            )
            .expect("modular ZK prove");
        ProvedGuest {
            trusted_advice_commitment: trusted.map(|entry| entry.commitment),
            preprocessing,
            public_io,
            proof,
        }
    }

    fn prove_muldiv(backend: JoltBackend<Fr, DoryScheme>) -> ProvedGuest {
        prove_guest(muldiv_case(), backend, |_| {})
    }

    fn verify(proved: &ProvedGuest) -> Result<(), VerifierError> {
        jolt_verifier::verify::<Fr, DoryScheme, Pedersen<Bn254G1>, Blake2bTranscript>(
            &proved.preprocessing.verifier,
            &proved.public_io,
            &proved.proof,
            proved.trusted_advice_commitment.as_ref(),
        )
    }

    /// The reference kernel tier under the ZK envelope; the optimized tier is
    /// covered by the guest matrix.
    #[test]
    fn zk_muldiv_reference_backend_proof_is_accepted() {
        with_zk_stack(|| {
            let proved = prove_muldiv(JoltBackend::reference());
            assert!(matches!(proved.proof.claims, JoltProofClaims::Zk { .. }));
            verify(&proved).expect("modular ZK proof must verify");
        });
    }

    #[test]
    fn zk_sha3_inline_modular_proof_is_accepted() {
        with_zk_stack(|| {
            let message: Vec<u8> = (0..SHA3_INPUT_LEN).map(|i| i as u8).collect();
            let proved = prove_guest(
                GuestCase {
                    func: Some("sha3"),
                    inputs: postcard::to_stdvec(&message).expect("serialize input"),
                    ..GuestCase::new("sha3-guest")
                },
                JoltBackend::optimized(),
                |rows| {
                    assert_eq!(
                        rows.iter()
                            .filter(|row| {
                                row.instruction_kind() == Some(JoltInstructionKind::VirtualROTRI)
                            })
                            .count(),
                        KECCAK_ROTRI_ROWS * SHA3_PERMUTATIONS,
                        "two unaligned fused-absorb blocks and the padded final Keccak permutation must be expanded into the modular trace",
                    );
                },
            );
            verify(&proved).expect("modular SHA3 ZK proof must verify");
        });
    }

    #[test]
    fn zk_muldiv_tampered_blindfold_is_rejected() {
        with_zk_stack(|| {
            let mut proved = prove_muldiv(JoltBackend::reference());
            let JoltProofClaims::Zk { blindfold_proof } = &mut proved.proof.claims else {
                panic!("ZK proof must carry BlindFold claims");
            };
            blindfold_proof.random_u += Fr::from_u64(1);
            assert!(verify(&proved).is_err());
        });
    }

    #[test]
    fn zk_advice_consumer_modular_proof_is_accepted() {
        with_zk_stack(|| {
            let proved = prove_guest(
                GuestCase {
                    inputs: postcard::to_stdvec(&12u64).expect("serialize input"),
                    untrusted_advice: postcard::to_stdvec(&5u64)
                        .expect("serialize untrusted advice"),
                    trusted_advice: postcard::to_stdvec(&7u64).expect("serialize trusted advice"),
                    ..GuestCase::new("advice-consumer-guest")
                },
                JoltBackend::reference(),
                |_| {},
            );
            assert!(proved.proof.untrusted_advice_commitment.is_some());
            verify(&proved).expect("modular ZK advice proof must verify");
        });
    }

    #[test]
    fn zk_committed_muldiv_modular_proof_is_accepted() {
        with_zk_stack(|| {
            let run = support::prepare(&muldiv_case());
            let config = derive_config(&run);
            let preprocessing = jolt_prover::dory::preprocess_committed(run.preprocessing, 2)
                .expect("committed preprocessing");
            let program_preprocessing = preprocessing.program_arc().expect("retained full program");
            let public_io = run.trace.device.clone();
            let witness = TraceBackend::<OwnedTrace>::from_compact(
                JoltVmWitnessConfig::new(
                    config.trace_length.ilog2() as usize,
                    config.ram_K,
                    config.one_hot_config,
                ),
                JoltVmWitnessInputs::new(&run.program, &program_preprocessing, run.trace),
            );
            let proof = jolt_prover::dory::prove::<
                Fr,
                DoryScheme,
                Pedersen<Bn254G1>,
                Blake2bTranscript,
                _,
            >(
                &JoltBackend::reference(),
                &preprocessing,
                &config,
                None,
                &witness,
                &public_io,
            )
            .expect("committed ZK prove");
            jolt_verifier::verify::<Fr, DoryScheme, Pedersen<Bn254G1>, Blake2bTranscript>(
                &preprocessing.verifier,
                &public_io,
                &proof,
                None,
            )
            .expect("committed ZK proof must verify");
        });
    }
}
