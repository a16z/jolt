//! Field-inline end-to-end: the modular prover's FR-composed proofs against
//! the full `jolt_verifier::verify` entry, in both proof modes.
//!
//! Two guests span the composed protocol's envelope: the eq-MLE guest
//! (`eqpoly-field-guest`) exercises every shipped FR instruction family —
//! LoadImm, both x-register bridges, add/sub/mul, and FIELD_ASSERT_EQ — and
//! the FR-profile muldiv build is the uniform-shape degenerate case (an FR-on
//! proof over a trace with zero FR instructions, so every FR column including
//! the committed `FieldRdInc` is identically zero). Clear-mode tampers hit
//! the FR-specific wire surface — a stage-1 FR opening, the `FieldRdInc`
//! commitment, a stage-2 FR product appendage value, and the stage-2 batch
//! round polynomial at the FR claim-reduction's gamma position — and every
//! mutation must reject.
//!
//! Every suite runs over BOTH kernel backends: `JoltBackend::reference` (the
//! byte-parity oracle) and `JoltBackend::optimized` (the sparse/composed
//! tier) — one prover, two backends, the same verifier entry, so an
//! optimized/reference wire divergence fails here as a verification error
//! even before the kernel-level parity tests localize it.

#[cfg(all(
    feature = "prover-fixtures",
    feature = "field-inline",
    not(feature = "akita")
))]
#[expect(clippy::expect_used, reason = "integration tests should fail loudly")]
mod support {
    use std::sync::Arc;
    #[cfg(feature = "zk")]
    use std::thread::Builder;

    use common::jolt_device::{JoltDevice, MemoryConfig, MemoryLayout};
    use jolt_crypto::{Bn254G1, Pedersen};
    use jolt_dory::DoryScheme;
    use jolt_field::{CanonicalBytes, Fr, Ring};
    use jolt_program::execution::{
        ExecutionBackend, JoltProgram, OwnedTrace, TraceInputs, TraceOutput, TraceRow,
    };
    use jolt_prover::{JoltBackend, JoltProverPreprocessing, ProverConfig};
    use jolt_prover_legacy::ark_bn254::Fr as LegacyFr;
    use jolt_prover_legacy::curve::Bn254Curve;
    use jolt_prover_legacy::host::Program;
    use jolt_prover_legacy::poly::commitment::dory::DoryCommitmentScheme;
    use jolt_prover_legacy::zkvm::preprocessing::JoltSharedPreprocessing;
    use jolt_prover_legacy::zkvm::program::ProgramPreprocessing as LegacyProgramPreprocessing;
    use jolt_prover_legacy::zkvm::proof::verifier_preprocessing_from_prover;
    use jolt_prover_legacy::zkvm::prover::JoltProverPreprocessing as LegacyProverPreprocessing;
    use jolt_transcript::LegacyBlake2bTranscript as Blake2bTranscript;
    use jolt_verifier::proof::JoltProof;
    use jolt_verifier::{JoltVerifierPreprocessing, VerifierError};
    use jolt_witness::{JoltVmWitnessConfig, JoltVmWitnessInputs, TraceBackend};
    use tracer::execution_backend::TracerBackend;

    pub const MAX_PADDED_TRACE_LENGTH: usize = 1 << 16;

    pub type Proof = JoltProof<DoryScheme, Pedersen<Bn254G1>>;
    pub type VerifierPreprocessing = JoltVerifierPreprocessing<DoryScheme, Pedersen<Bn254G1>>;
    pub type LegacyPreprocessing =
        LegacyProverPreprocessing<LegacyFr, Bn254Curve, DoryCommitmentScheme>;

    pub const EQ_PAIRS: [[u64; 2]; 4] = [[3, 5], [7, 2], [11, 13], [1, 9]];

    /// eq(r, x) = prod_i (r_i·x_i + (1 − r_i)(1 − x_i)) — the host-side
    /// reference the guest's FIELD_ASSERT_EQ checks against.
    fn eq_mle(pairs: &[[u64; 2]; 4]) -> Fr {
        let one = Fr::from_u64(1);
        pairs.iter().fold(one, |acc, [r, x]| {
            let r = Fr::from_u64(*r);
            let x = Fr::from_u64(*x);
            acc * (r * x + (one - r) * (one - x))
        })
    }

    /// The eq-MLE guest's inputs: the coordinate pairs plus the expected
    /// value as canonical little-endian u64 limbs (each provable-fn argument
    /// postcard-encoded and concatenated).
    pub fn eqpoly_inputs() -> Vec<u8> {
        let value = eq_mle(&EQ_PAIRS);
        let mut bytes = [0u8; 32];
        value.to_bytes_le(&mut bytes);
        let mut limbs = [0u64; 4];
        for (limb, chunk) in limbs.iter_mut().zip(bytes.chunks_exact(8)) {
            *limb = u64::from_le_bytes(chunk.try_into().expect("8-byte chunk"));
        }
        let mut inputs = postcard::to_stdvec(&EQ_PAIRS).expect("serialize pairs");
        inputs.extend(postcard::to_stdvec(&limbs).expect("serialize limbs"));
        inputs
    }

    pub struct FrGuest {
        pub verifier_preprocessing: VerifierPreprocessing,
        pub trace_output: TraceOutput<OwnedTrace>,
        pub program: Arc<JoltProgram>,
    }

    /// Build `guest_name` under the FR instruction profile, preprocess with
    /// that profile (the profile carry `preprocess_with_profile` exists for),
    /// and re-trace through the modular tracer backend.
    pub fn fr_guest(guest_name: &str, inputs: &[u8]) -> FrGuest {
        let mut program = Program::new(guest_name);
        program.enable_field_inline();

        let (bytecode, memory_init, _, entry_address) = program.decode();
        let (_, _, _, io_device) = program.trace(inputs, &[], &[]);
        let elf_contents = program.get_elf_contents().expect("elf contents");
        let preprocessed = LegacyProgramPreprocessing::preprocess_with_profile(
            bytecode,
            memory_init,
            entry_address,
            program.instruction_profile(),
        )
        .expect("FR-profile preprocess");
        let shared = JoltSharedPreprocessing::new(
            preprocessed,
            io_device.memory_layout.clone(),
            MAX_PADDED_TRACE_LENGTH,
        );
        let legacy_preprocessing: LegacyPreprocessing = LegacyProverPreprocessing::new(shared);
        let verifier_preprocessing = verifier_preprocessing_from_prover(&legacy_preprocessing);

        let jolt_program = Arc::new(JoltProgram::from_elf_bytes_with_profile(
            elf_contents,
            program.instruction_profile(),
        ));
        let trace_output = trace_modular(&jolt_program, &io_device.memory_layout, inputs);
        FrGuest {
            verifier_preprocessing,
            trace_output,
            program: jolt_program,
        }
    }

    fn trace_modular(
        program: &JoltProgram,
        memory_layout: &MemoryLayout,
        inputs: &[u8],
    ) -> TraceOutput<OwnedTrace> {
        let memory_config = MemoryConfig {
            max_untrusted_advice_size: memory_layout.max_untrusted_advice_size,
            max_trusted_advice_size: memory_layout.max_trusted_advice_size,
            max_input_size: memory_layout.max_input_size,
            max_output_size: memory_layout.max_output_size,
            stack_size: memory_layout.stack_size,
            heap_size: memory_layout.heap_size,
            program_size: Some(memory_layout.program_size),
        };
        TracerBackend::new()
            .trace(
                program,
                TraceInputs {
                    inputs: inputs.to_vec(),
                    untrusted_advice: Vec::new(),
                    trusted_advice: Vec::new(),
                    memory_config,
                    advice_tape: None,
                },
            )
            .expect("modular trace")
    }

    fn advice_vars(max_advice_size_bytes: u64) -> usize {
        ((max_advice_size_bytes / 8) as usize)
            .next_power_of_two()
            .max(1)
            .ilog2() as usize
    }

    fn setup_total_vars(memory_layout: &MemoryLayout) -> usize {
        let max_log_t = MAX_PADDED_TRACE_LENGTH.ilog2() as usize;
        let max_log_k_chunk = 4usize; // max_log_t = 16 < the 25-bit threshold
        (max_log_k_chunk + max_log_t)
            .max(advice_vars(memory_layout.max_trusted_advice_size))
            .max(advice_vars(memory_layout.max_untrusted_advice_size))
    }

    pub fn field_inline_rows(rows: &[TraceRow]) -> usize {
        rows.iter().filter(|row| row.field_inline.is_some()).count()
    }

    /// Prove `guest` FR-on with the modular prover (the field-inline witness
    /// view attached — the FR-on build refuses classic-profile witnesses).
    pub fn prove_fr(
        guest: FrGuest,
        backend: JoltBackend<Fr, DoryScheme>,
    ) -> (VerifierPreprocessing, JoltDevice, Proof) {
        let FrGuest {
            verifier_preprocessing,
            trace_output,
            program,
        } = guest;
        let memory_layout = trace_output.device.memory_layout.clone();
        let public_io = trace_output.device.clone();
        let config = ProverConfig::derive::<Fr>(
            trace_output.trace.rows(),
            &memory_layout,
            verifier_preprocessing.program.min_bytecode_address(),
            verifier_preprocessing.program.program_image_len_words(),
            MAX_PADDED_TRACE_LENGTH,
        )
        .expect("derive config");

        let mut rows = trace_output.trace.rows().to_vec();
        rows.resize(config.trace_length, TraceRow::default());
        let padded_output = TraceOutput::new(
            OwnedTrace::new(rows),
            trace_output.device,
            trace_output.final_memory,
            trace_output.advice_tape,
        );

        let program_preprocessing = verifier_preprocessing
            .program
            .as_full_arc()
            .expect("full program preprocessing");
        let witness = TraceBackend::new(
            JoltVmWitnessConfig::new(
                config.trace_length.ilog2() as usize,
                config.ram_K,
                config.one_hot_config,
            ),
            JoltVmWitnessInputs::new(&program, &program_preprocessing, padded_output),
        )
        .with_field_inline()
        .expect("field-inline witness view");
        let witness = Arc::new(witness);

        // Sized off MAX_PADDED_TRACE_LENGTH like the legacy preprocessing's
        // generators: the verifier setup derives from those, and Dory URS
        // generators are seeded per exact size — a prover setup sized off the
        // derived config would commit under a different generator set than
        // the verifier checks against.
        let prover_preprocessing = JoltProverPreprocessing::<DoryScheme, Pedersen<Bn254G1>> {
            verifier: verifier_preprocessing,
            pcs_setup: DoryScheme::setup_prover(setup_total_vars(&memory_layout)),
            committed_program: None,
        };
        let proof = jolt_prover::prove::<Fr, DoryScheme, Pedersen<Bn254G1>, Blake2bTranscript, _>(
            &backend,
            &prover_preprocessing,
            &config,
            None,
            witness.as_ref(),
            &public_io,
        )
        .expect("modular FR prove");
        (prover_preprocessing.verifier, public_io, proof)
    }

    pub fn verify_full(
        preprocessing: &VerifierPreprocessing,
        public_io: &JoltDevice,
        proof: &Proof,
    ) -> Result<(), VerifierError> {
        jolt_verifier::verify::<Fr, DoryScheme, Pedersen<Bn254G1>, Blake2bTranscript>(
            preprocessing,
            public_io,
            proof,
            None,
        )
    }

    /// A labeled kernel-backend constructor.
    pub type BackendCase = (&'static str, fn() -> JoltBackend<Fr, DoryScheme>);

    /// The two kernel backends every FR e2e case runs over, labeled for
    /// assertion messages.
    pub fn backends() -> [BackendCase; 2] {
        [
            ("reference", JoltBackend::reference),
            ("optimized", JoltBackend::optimized),
        ]
    }

    /// BlindFold verification (and the prover's replay of it) recurses over a
    /// large folded R1CS — run on a dedicated wide stack like the
    /// jolt-verifier ZK suites.
    #[cfg(feature = "zk")]
    pub fn with_zk_stack<R: Send + 'static>(body: impl FnOnce() -> R + Send + 'static) -> R {
        Builder::new()
            .stack_size(128 * 1024 * 1024)
            .spawn(body)
            .expect("spawn ZK test thread")
            .join()
            .expect("ZK test thread panicked")
    }
}

#[cfg(all(
    feature = "prover-fixtures",
    feature = "field-inline",
    not(feature = "zk"),
    not(feature = "akita")
))]
#[expect(
    clippy::expect_used,
    clippy::panic,
    reason = "integration tests should fail loudly"
)]
mod clear {
    use common::jolt_device::JoltDevice;
    use jolt_dory::DoryScheme;
    use jolt_field::{Fr, Ring};
    use jolt_poly::CompressedPoly;
    use jolt_prover::JoltBackend;
    use jolt_sumcheck::{ClearProof, SumcheckProof};
    use jolt_verifier::proof::JoltProofClaims;

    use super::support::{self, Proof, VerifierPreprocessing};

    fn prove_eqpoly(
        backend: JoltBackend<Fr, DoryScheme>,
    ) -> (VerifierPreprocessing, JoltDevice, Proof) {
        let guest = support::fr_guest("eqpoly-field-guest", &support::eqpoly_inputs());
        assert!(
            support::field_inline_rows(guest.trace_output.trace.rows()) > 0,
            "the eq-MLE guest must trace FR-active",
        );
        support::prove_fr(guest, backend)
    }

    /// Both backends' proofs must verify AND be equal wire objects — clear
    /// mode draws nothing outside Fiat-Shamir, so reference/optimized
    /// divergence anywhere in the composed pipeline shows up here as a proof
    /// inequality even when both sides individually verify.
    #[test]
    fn field_inline_eqpoly_proof_is_accepted() {
        let mut proofs = Vec::new();
        for (label, backend) in support::backends() {
            let (preprocessing, public_io, proof) = prove_eqpoly(backend());
            assert!(
                proof.commitments.field_inline.is_some(),
                "FR-on proofs must carry the field-inline commitment payload ({label})",
            );
            assert!(matches!(proof.claims, JoltProofClaims::Clear(_)));
            support::verify_full(&preprocessing, &public_io, &proof)
                .unwrap_or_else(|error| panic!("modular FR proof must verify ({label}): {error}"));
            proofs.push(proof);
        }
        assert!(
            proofs[0] == proofs[1],
            "reference and optimized FR proofs must be identical wire objects",
        );
    }

    /// The uniform-shape degenerate case: an FR-profile guest executing zero
    /// FR instructions still proves under the composed protocol, with an
    /// all-zero `FieldRdInc` commitment and zero FR openings.
    #[test]
    fn field_inline_inactive_muldiv_proof_is_accepted() {
        let mut proofs = Vec::new();
        for (label, backend) in support::backends() {
            let inputs = postcard::to_stdvec(&[9u32, 5u32, 3u32]).expect("serialize inputs");
            let guest = support::fr_guest("muldiv-guest", &inputs);
            assert_eq!(
                support::field_inline_rows(guest.trace_output.trace.rows()),
                0,
                "the FR-profile muldiv trace must contain no FR instructions",
            );
            let (preprocessing, public_io, proof) = support::prove_fr(guest, backend());
            assert!(proof.commitments.field_inline.is_some());
            support::verify_full(&preprocessing, &public_io, &proof).unwrap_or_else(|error| {
                panic!("FR-inactive modular proof must verify ({label}): {error}")
            });
            proofs.push(proof);
        }
        assert!(
            proofs[0] == proofs[1],
            "reference and optimized FR-inactive proofs must be identical wire objects",
        );
    }

    /// Every FR-specific single-field tamper must reject: one proof, four
    /// mutations on fresh clones. The optimized backend proves here — its
    /// wire bytes equal the reference's (the accept tests pin both), so one
    /// backend's tamper matrix covers both.
    #[test]
    fn field_inline_tampered_proofs_are_rejected() {
        let (preprocessing, public_io, proof) = prove_eqpoly(JoltBackend::optimized());
        support::verify_full(&preprocessing, &public_io, &proof)
            .expect("base proof must verify before tampering");
        let one = Fr::from_u64(1);

        type Tamper = (&'static str, Box<dyn Fn(&mut Proof)>);
        let tampers: Vec<Tamper> = vec![
            (
                "stage1 FR rs1_value opening",
                Box::new(move |proof| {
                    let JoltProofClaims::Clear(claims) = &mut proof.claims else {
                        panic!("clear proof expected");
                    };
                    let outer = claims
                        .stage1
                        .field_inline_outer
                        .as_mut()
                        .expect("FR-on proof carries stage-1 FR openings");
                    outer.rs1_value += one;
                }),
            ),
            (
                "FieldRdInc commitment",
                Box::new(|proof| {
                    let replacement = proof.commitments.ram_inc.clone();
                    let field_inline = proof
                        .commitments
                        .field_inline
                        .as_mut()
                        .expect("FR-on proof carries the field-inline payload");
                    assert_ne!(
                        field_inline.field_registers.rd_inc, replacement,
                        "replacement commitment must differ",
                    );
                    field_inline.field_registers.rd_inc = replacement;
                }),
            ),
            (
                "stage2 FR product appendage rd_value",
                Box::new(move |proof| {
                    let JoltProofClaims::Clear(claims) = &mut proof.claims else {
                        panic!("clear proof expected");
                    };
                    let product = claims
                        .stage2
                        .field_inline_product
                        .as_mut()
                        .expect("FR-on proof carries the stage-2 FR product appendage");
                    product.rd_value += one;
                }),
            ),
            (
                // The composed stage-2 batch (FR claim reduction + product
                // appendage) rejects a corrupted round polynomial like the
                // base batch does; FR-on, no legacy-fixture suite covers the
                // round polynomials, so this is the composed batch's guard.
                "stage2 composed batch round polynomial corrupted",
                Box::new(|proof| {
                    let SumcheckProof::Clear(ClearProof::Compressed(batch)) =
                        &mut proof.stages.stage2_sumcheck_proof
                    else {
                        panic!("clear compressed stage-2 batch expected");
                    };
                    let round = batch
                        .round_polynomials
                        .first_mut()
                        .expect("stage-2 batch has a first round");
                    *round = CompressedPoly::new(vec![Fr::from_u64(7)]);
                }),
            ),
        ];
        for (name, tamper) in tampers {
            let mut tampered = proof.clone();
            tamper(&mut tampered);
            assert!(
                support::verify_full(&preprocessing, &public_io, &tampered).is_err(),
                "tampered proof must be rejected: {name}",
            );
        }
    }
}

#[cfg(all(
    feature = "prover-fixtures",
    feature = "field-inline",
    feature = "zk",
    not(feature = "akita")
))]
#[expect(
    clippy::expect_used,
    clippy::panic,
    reason = "integration tests should fail loudly"
)]
mod zk {
    use jolt_field::{Fr, Ring};
    use jolt_prover::JoltBackend;
    use jolt_verifier::proof::JoltProofClaims;

    use super::support;

    /// ZK accept plus the FR tampers that exist on the ZK wire (clear claims
    /// don't): the FieldRdInc commitment and the BlindFold payload. One
    /// proof, mutations on clones — ZK proving is the expensive step.
    #[test]
    fn zk_field_inline_eqpoly_accepts_and_tampers_reject() {
        support::with_zk_stack(|| {
            // The optimized backend proves the tampered matrix; the
            // reference ZK path is pinned by the muldiv accept below (ZK
            // blindings randomize the wire, so proofs are verify-only here —
            // clear mode owns the byte-equality statement).
            let guest = support::fr_guest("eqpoly-field-guest", &support::eqpoly_inputs());
            assert!(
                support::field_inline_rows(guest.trace_output.trace.rows()) > 0,
                "the eq-MLE guest must trace FR-active",
            );
            let (preprocessing, public_io, proof) =
                support::prove_fr(guest, JoltBackend::optimized());
            assert!(matches!(proof.claims, JoltProofClaims::Zk { .. }));
            assert!(proof.commitments.field_inline.is_some());
            support::verify_full(&preprocessing, &public_io, &proof)
                .expect("modular FR ZK proof must verify");

            let mut commitment_tampered = proof.clone();
            let replacement = commitment_tampered.commitments.ram_inc.clone();
            let field_inline = commitment_tampered
                .commitments
                .field_inline
                .as_mut()
                .expect("FR-on proof carries the field-inline payload");
            assert_ne!(field_inline.field_registers.rd_inc, replacement);
            field_inline.field_registers.rd_inc = replacement;
            assert!(
                support::verify_full(&preprocessing, &public_io, &commitment_tampered).is_err(),
                "a tampered FieldRdInc commitment must be rejected in ZK mode",
            );

            let mut blindfold_tampered = proof;
            let JoltProofClaims::Zk { blindfold_proof } = &mut blindfold_tampered.claims else {
                panic!("ZK proof must carry the BlindFold claims variant");
            };
            blindfold_proof.random_u += Fr::from_u64(1);
            assert!(
                support::verify_full(&preprocessing, &public_io, &blindfold_tampered).is_err(),
                "a tampered BlindFold proof must be rejected",
            );
        });
    }

    #[test]
    fn zk_field_inline_inactive_muldiv_proof_is_accepted() {
        support::with_zk_stack(|| {
            for (label, backend) in support::backends() {
                let inputs = postcard::to_stdvec(&[9u32, 5u32, 3u32]).expect("serialize inputs");
                let guest = support::fr_guest("muldiv-guest", &inputs);
                assert_eq!(
                    support::field_inline_rows(guest.trace_output.trace.rows()),
                    0,
                    "the FR-profile muldiv trace must contain no FR instructions",
                );
                let (preprocessing, public_io, proof) = support::prove_fr(guest, backend());
                support::verify_full(&preprocessing, &public_io, &proof).unwrap_or_else(|error| {
                    panic!("FR-inactive modular ZK proof must verify ({label}): {error}")
                });
            }
        });
    }
}

#[cfg(not(all(
    feature = "prover-fixtures",
    feature = "field-inline",
    not(feature = "akita")
)))]
#[test]
#[ignore = "enable --features prover-fixtures,field-inline (optionally +zk) to run the dory \
            field-inline e2e; the packed suite is akita_field_inline_e2e.rs"]
fn field_inline_e2e() {}
