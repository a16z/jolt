//! Packed (Akita) field-inline end-to-end: FR-composed proofs over fp128
//! against the full `jolt_verifier::verify` entry — the packed sibling of
//! `field_inline_e2e.rs` (the akita axis proves exclusively over fp128, so
//! the FR guests re-fixture here at the 16-byte value encoding).
//!
//! Accept: the eq-MLE guest (every shipped FR instruction family, a live
//! `FieldRdInc` column) and the FR-profile muldiv (zero FR instructions,
//! `FieldRdInc` identically zero, the limb group PRESENT with all-zero
//! content — the always-present rule, pinning the all-zero dense open), each
//! over both kernel backends with wire equality.
//! Tamper (all must reject): a limb-evaluation offset (the stage-8 linear
//! recomposition check), a limb-commitment layout-digest byte flip, a
//! batch-proof mutation, the limb group stripped from the proof, and a
//! spurious second FR-role group in the heterogeneous batch statement.

#[cfg(all(
    feature = "prover-fixtures",
    feature = "field-inline",
    feature = "akita"
))]
#[expect(clippy::expect_used, reason = "integration tests should fail loudly")]
mod support {
    use std::sync::Arc;

    use common::jolt_device::{JoltDevice, MemoryConfig, MemoryLayout};
    use jolt_claims::protocols::field_inline::{
        FieldInlineCommittedPolynomial, FieldInlinePolynomialId,
    };
    use jolt_field::CanonicalBytes;
    use jolt_openings::CommitmentScheme as VerifierCommitmentScheme;
    use jolt_program::execution::{
        ExecutionBackend, JoltProgram, OwnedTrace, TraceInputs, TraceOutput, TraceRow,
    };
    use jolt_prover::{akita, JoltProverPreprocessing, ProverConfig};
    use jolt_prover_legacy::host;
    use jolt_prover_legacy::zkvm::packed::{
        akita_verifier_preprocessing, field_inc_limb_schedule_params, AkitaField, AkitaJoltProof,
        AkitaPackedScheme, AkitaScheme, AkitaTranscript, AkitaVc,
    };
    use jolt_prover_legacy::zkvm::preprocessing::JoltSharedPreprocessing;
    use jolt_prover_legacy::zkvm::program::ProgramPreprocessing as LegacyProgramPreprocessing;
    use jolt_prover_legacy::zkvm::prover::JoltProverPreprocessing as LegacyProverPreprocessing;
    use jolt_witness::{JoltVmWitnessConfig, JoltVmWitnessInputs, TraceBackend};
    use tracer::execution_backend::TracerBackend;

    pub const MAX_PADDED_TRACE_LENGTH: usize = 1 << 16;

    pub type Proof = AkitaJoltProof;

    pub const EQ_PAIRS: [[u64; 2]; 4] = [[3, 5], [7, 2], [11, 13], [1, 9]];

    /// eq(r, x) = prod_i (r_i·x_i + (1 − r_i)(1 − x_i)) over the packed
    /// axis's proof field (fp128, p = 2^128 − 2^32 + 22537) — the host-side
    /// reference the guest's FIELD_ASSERT_EQ checks against.
    fn eq_mle(pairs: &[[u64; 2]; 4]) -> AkitaField {
        let one = AkitaField::from_u64(1);
        pairs.iter().fold(one, |acc, [r, x]| {
            let r = AkitaField::from_u64(*r);
            let x = AkitaField::from_u64(*x);
            acc * (r * x + (one - r) * (one - x))
        })
    }

    /// The eq-MLE guest's inputs. The guest is field-agnostic (it Horner-
    /// recomposes the limbs in whatever field the build proves over), so the
    /// pinned expected value is the fp128 evaluation: its 16-byte canonical
    /// form fills the low two u64 limbs, the high two stay zero.
    pub fn eqpoly_inputs() -> Vec<u8> {
        let value = eq_mle(&EQ_PAIRS);
        let bytes = value.to_bytes_le_vec();
        assert_eq!(bytes.len(), <AkitaField as CanonicalBytes>::NUM_BYTES);
        let mut limbs = [0u64; 4];
        for (limb, chunk) in limbs.iter_mut().zip(bytes.chunks_exact(8)) {
            *limb = u64::from_le_bytes(chunk.try_into().expect("8-byte chunk"));
        }
        let mut inputs = postcard::to_stdvec(&EQ_PAIRS).expect("serialize pairs");
        inputs.extend(postcard::to_stdvec(&limbs).expect("serialize limbs"));
        inputs
    }

    type LegacyPreprocessing = LegacyProverPreprocessing<
        jolt_prover_legacy::field::akita::AkitaFp128,
        jolt_prover_legacy::zkvm::packed::AkitaNoCurve,
        AkitaPackedScheme,
    >;

    pub struct FrGuest {
        legacy_preprocessing: LegacyPreprocessing,
        pub trace_output: TraceOutput<OwnedTrace>,
        program: Arc<JoltProgram>,
    }

    /// Build `guest_name` under the FR instruction profile, preprocess with
    /// that profile on the packed scheme, and trace through the modular
    /// tracer backend (which executes FR ops over fp128 on this build — the
    /// eq-MLE guest's FIELD_ASSERT_EQ already validates the fixture inputs
    /// at trace time).
    pub fn fr_guest(guest_name: &str, inputs: &[u8]) -> FrGuest {
        let mut program = host::Program::new(guest_name);
        program.enable_field_inline();

        let (bytecode, memory_init, _, entry_address) = program.decode();
        let (_, _, _, io_device) = program.trace(inputs, &[], &[]);
        let elf_contents = program.get_elf_contents().expect("elf contents");
        let preprocessed =
            LegacyProgramPreprocessing::<AkitaPackedScheme>::preprocess_with_profile(
                bytecode,
                memory_init,
                entry_address,
                program.instruction_profile(),
            )
            .expect("FR-profile packed preprocess");
        let shared: JoltSharedPreprocessing<AkitaPackedScheme> = JoltSharedPreprocessing::new(
            preprocessed,
            io_device.memory_layout.clone(),
            MAX_PADDED_TRACE_LENGTH,
        );
        let legacy_preprocessing: LegacyPreprocessing = LegacyProverPreprocessing::new(shared);

        let jolt_program = Arc::new(JoltProgram::from_elf_bytes_with_profile(
            elf_contents,
            program.instruction_profile(),
        ));
        let trace_output = trace_modular(&jolt_program, &io_device.memory_layout, inputs);
        FrGuest {
            legacy_preprocessing,
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

    pub fn field_inline_rows(rows: &[TraceRow]) -> usize {
        rows.iter().filter(|row| row.field_inline.is_some()).count()
    }

    /// Everything the tamper matrix needs beyond the proof: the verifier
    /// preprocessing, the proof shape's config, and the honest per-cycle
    /// `FieldRdInc` values (the base material for forging limb commitments
    /// through the real commit path).
    pub struct ProveOutput {
        pub verifier_preprocessing: jolt_verifier::JoltVerifierPreprocessing<AkitaScheme, AkitaVc>,
        pub public_io: JoltDevice,
        pub proof: Proof,
        pub config: ProverConfig,
        pub rd_inc: Vec<AkitaField>,
    }

    /// Prove `guest` FR-on with the modular packed prover. The transparent
    /// grouped setup carries the FR limb arity line, so both fronts
    /// provision the [FieldIncLimbs] grouped rows every FR proof resolves.
    pub fn prove_fr(
        guest: FrGuest,
        backend: akita::JoltAkitaBackend<AkitaField, AkitaScheme>,
    ) -> ProveOutput {
        let FrGuest {
            legacy_preprocessing,
            trace_output,
            program,
        } = guest;
        let memory_layout = trace_output.device.memory_layout.clone();
        let public_io = trace_output.device.clone();
        let config = ProverConfig::derive::<AkitaField>(
            trace_output.trace.rows(),
            &memory_layout,
            legacy_preprocessing
                .shared
                .program_meta
                .min_bytecode_address,
            legacy_preprocessing
                .shared
                .program
                .program_image_len_words(),
            MAX_PADDED_TRACE_LENGTH,
        )
        .expect("derive config");

        let log_t = config.trace_length.ilog2() as usize;
        let bytecode_len = legacy_preprocessing.shared.bytecode_size();
        let (setup_shape, layout_digest, one_hot_k) =
            akita::one_hot_trace_setup_shape(&config, bytecode_len)
                .expect("OneHotTrace setup shape");
        // Preprocessing must cover every arity a proof of this program can
        // select, up to the padded ceiling (the derivation legacy's
        // one_hot_trace_setup_params performs; the trace overhead over log_T
        // is constant per K).
        let ceiling_num_vars =
            setup_shape.num_vars - log_t + MAX_PADDED_TRACE_LENGTH.ilog2() as usize;
        let advice_schedule = jolt_akita::AdviceScheduleParams::new(None, None, ceiling_num_vars)
            .with_field_inc_limbs(field_inc_limb_schedule_params(setup_shape.num_vars, log_t));
        let params = <<AkitaScheme as VerifierCommitmentScheme>::SetupParams>::one_hot_only_grouped(
            setup_shape.num_vars,
            setup_shape.num_polys,
            2,
            layout_digest,
            one_hot_k,
            Some(advice_schedule),
        );
        let (object_setup, verifier_setup) =
            <AkitaScheme as VerifierCommitmentScheme>::setup(params)
                .expect("the transparent packed setup must derive");
        let verifier_preprocessing =
            akita_verifier_preprocessing(&legacy_preprocessing, verifier_setup, None);

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
            JoltVmWitnessConfig::new(log_t, config.ram_K, config.one_hot_config),
            JoltVmWitnessInputs::new(&program, &program_preprocessing, padded_output),
        )
        .with_field_inline()
        .expect("field-inline witness view");
        let rd_inc: Vec<AkitaField> = witness
            .field_inline_witness()
            .expect("field-inline oracle")
            .oracle_table(FieldInlinePolynomialId::Committed(
                FieldInlineCommittedPolynomial::FieldRdInc,
            ))
            .expect("FieldRdInc oracle table");

        let prover_preprocessing = JoltProverPreprocessing::<AkitaScheme, AkitaVc> {
            verifier: verifier_preprocessing,
            pcs_setup: object_setup,
            committed_program: None,
        };
        let proof = akita::prove::<AkitaField, AkitaScheme, AkitaVc, AkitaTranscript, _>(
            &backend,
            &prover_preprocessing,
            &config,
            None,
            &witness,
            &public_io,
        )
        .expect("packed FR prove");
        ProveOutput {
            verifier_preprocessing: prover_preprocessing.verifier,
            public_io,
            proof,
            config,
            rd_inc,
        }
    }

    pub fn verify_full(
        preprocessing: &jolt_verifier::JoltVerifierPreprocessing<AkitaScheme, AkitaVc>,
        public_io: &JoltDevice,
        proof: &Proof,
    ) -> Result<(), jolt_verifier::VerifierError> {
        jolt_verifier::verify::<AkitaField, AkitaScheme, AkitaVc, AkitaTranscript>(
            preprocessing,
            public_io,
            proof,
            None,
        )
    }

    /// A labeled packed kernel-backend constructor.
    pub type BackendCase = (
        &'static str,
        fn() -> akita::JoltAkitaBackend<AkitaField, AkitaScheme>,
    );

    pub fn backends() -> [BackendCase; 2] {
        [
            ("reference", akita::JoltAkitaBackend::reference),
            ("optimized", akita::JoltAkitaBackend::optimized),
        ]
    }

    /// Commit the honest limb-word polynomial under `digest` through the real
    /// dense commit path, returning the commitment a tamper splices into a
    /// proof (same content, mutated identity).
    pub fn commit_limb_words_with_digest(
        output: &ProveOutput,
        digest: [u8; 32],
    ) -> jolt_akita::AkitaCommitment {
        use jolt_claims::protocols::field_inline::lattice::canonical_limbs;
        use jolt_openings::TransparentObjectSetup;
        use jolt_poly::Polynomial;
        use jolt_verifier::stages::stage8::field_inline_packed::limb_plan;

        let log_t = output.config.trace_length.ilog2() as usize;
        let plan = limb_plan::<AkitaField>(log_t).expect("canonical limb plan");
        let mut evaluations =
            vec![AkitaField::from_u64(0); 1usize << plan.packing().packed_num_vars()];
        for (cycle, value) in output.rd_inc.iter().enumerate() {
            for (limb, word) in canonical_limbs(value).into_iter().enumerate() {
                evaluations[(limb << log_t) | cycle] = AkitaField::from_u64(word);
            }
        }
        let polynomial = Polynomial::new(evaluations);
        let (setup, _) = <AkitaScheme as TransparentObjectSetup>::transparent_object_setup(
            plan.packing().packed_num_vars(),
            digest,
        )
        .expect("transparent limb setup");
        let (commitment, _hint) =
            <AkitaScheme as VerifierCommitmentScheme>::commit(&polynomial, &setup)
                .expect("forged limb commit");
        commitment
    }
}

#[cfg(all(
    feature = "prover-fixtures",
    feature = "field-inline",
    feature = "akita"
))]
#[expect(
    clippy::expect_used,
    clippy::panic,
    reason = "integration tests should fail loudly"
)]
mod clear {
    use jolt_prover::akita::JoltAkitaBackend;
    use jolt_prover_legacy::zkvm::packed::AkitaField;
    use jolt_verifier::proof::JoltProofClaims;

    use super::support;

    fn prove_eqpoly(
        backend: JoltAkitaBackend<AkitaField, jolt_akita::AkitaScheme>,
    ) -> support::ProveOutput {
        let guest = support::fr_guest("eqpoly-field-guest", &support::eqpoly_inputs());
        assert!(
            support::field_inline_rows(guest.trace_output.trace.rows()) > 0,
            "the eq-MLE guest must trace FR-active",
        );
        support::prove_fr(guest, backend)
    }

    fn clear_limb_claims(
        proof: &support::Proof,
    ) -> &jolt_verifier::stages::stage8::field_inline_packed::FieldIncLimbClaims<AkitaField> {
        let JoltProofClaims::Clear(claims) = &proof.claims else {
            panic!("packed proofs carry clear claims");
        };
        claims
            .field_inc_limbs
            .as_ref()
            .expect("packed FR proofs carry the limb claims")
    }

    /// Both backends' packed FR proofs must verify AND be equal wire objects.
    #[test]
    fn akita_field_inline_eqpoly_proof_is_accepted() {
        let mut proofs = Vec::new();
        for (label, backend) in support::backends() {
            let output = prove_eqpoly(backend());
            assert!(
                output.proof.field_inc_limbs_commitment.is_some(),
                "packed FR proofs must carry the limb-group commitment ({label})",
            );
            assert_eq!(
                clear_limb_claims(&output.proof).limbs.len(),
                2,
                "fp128 decomposes FieldRdInc into two u64 limbs ({label})",
            );
            support::verify_full(
                &output.verifier_preprocessing,
                &output.public_io,
                &output.proof,
            )
            .unwrap_or_else(|error| panic!("packed FR proof must verify ({label}): {error}"));
            proofs.push(output.proof);
        }
        assert!(
            proofs[0] == proofs[1],
            "reference and optimized packed FR proofs must be identical wire objects",
        );
    }

    /// The uniform-shape degenerate case: an FR-profile guest executing zero
    /// FR instructions — `FieldRdInc` identically zero, every limb word zero
    /// — still proves and verifies with the limb group PRESENT (all-zero
    /// content is legal: dense schedules are keyed by shape, never content).
    /// This pins the all-zero dense open.
    #[test]
    fn akita_field_inline_inactive_muldiv_proof_is_accepted() {
        let mut proofs = Vec::new();
        for (label, backend) in support::backends() {
            let inputs = postcard::to_stdvec(&[9u32, 5u32, 3u32]).expect("serialize inputs");
            let guest = support::fr_guest("muldiv-guest", &inputs);
            assert_eq!(
                support::field_inline_rows(guest.trace_output.trace.rows()),
                0,
                "the FR-profile muldiv trace must contain no FR instructions",
            );
            let output = support::prove_fr(guest, backend());
            assert!(output
                .rd_inc
                .iter()
                .all(|value| *value == AkitaField::from_u64(0)));
            assert!(
                output.proof.field_inc_limbs_commitment.is_some(),
                "a zero FieldRdInc still commits its limb group ({label})",
            );
            assert!(
                clear_limb_claims(&output.proof)
                    .limbs
                    .iter()
                    .all(|limb| *limb == AkitaField::from_u64(0)),
                "an all-zero group opens to all-zero limb evaluations ({label})",
            );
            support::verify_full(
                &output.verifier_preprocessing,
                &output.public_io,
                &output.proof,
            )
            .unwrap_or_else(|error| {
                panic!("FR-inactive packed proof must verify ({label}): {error}")
            });
            proofs.push(output.proof);
        }
        assert!(
            proofs[0] == proofs[1],
            "reference and optimized FR-inactive packed proofs must be identical wire objects",
        );
    }

    /// The packed FR tamper matrix: one honest proof, mutations on fresh
    /// clones, every one rejected.
    #[test]
    fn akita_field_inline_tampered_proofs_are_rejected() {
        let output = prove_eqpoly(JoltAkitaBackend::optimized());
        support::verify_full(
            &output.verifier_preprocessing,
            &output.public_io,
            &output.proof,
        )
        .expect("base proof must verify before tampering");
        let one = AkitaField::from_u64(1);

        // The honest limb polynomial under a corrupted layout digest, through
        // the real commit path: the metadata gate must reject before any
        // opening verification.
        let wrong_digest_commitment = {
            let honest = output
                .proof
                .field_inc_limbs_commitment
                .as_ref()
                .expect("packed FR proofs carry the limb-group commitment");
            let mut digest = jolt_openings::GroupCommitmentMetadata::layout_digest(honest);
            digest[0] ^= 0x01;
            support::commit_limb_words_with_digest(&output, digest)
        };

        type Tamper = (&'static str, Box<dyn Fn(&mut support::Proof)>);
        let tampers: Vec<Tamper> = vec![
            (
                "limb evaluation offset (linear recomposition check)",
                Box::new(move |proof| {
                    let JoltProofClaims::Clear(claims) = &mut proof.claims else {
                        panic!("clear proof expected");
                    };
                    let limbs = claims
                        .field_inc_limbs
                        .as_mut()
                        .expect("packed FR proof carries limb claims");
                    *limbs.limbs.first_mut().expect("two limbs") += one;
                }),
            ),
            (
                "limb commitment layout-digest byte flip",
                Box::new(move |proof| {
                    proof.field_inc_limbs_commitment = Some(wrong_digest_commitment.clone());
                }),
            ),
            (
                "batched opening proof mutation",
                Box::new(|proof| {
                    let mut value = serde_json::to_value(&proof.joint_opening_proof.main_batch)
                        .expect("serialize batch proof");
                    let bytes = value
                        .get_mut("serialized_akita_proof")
                        .and_then(serde_json::Value::as_array_mut)
                        .expect("batch proof carries the serialized backend proof");
                    let mid = bytes.len() / 2;
                    let byte = bytes.get_mut(mid).expect("nonempty backend proof");
                    let flipped = byte.as_u64().expect("byte value") ^ 0x01;
                    *byte = serde_json::Value::from(flipped);
                    proof.joint_opening_proof.main_batch =
                        serde_json::from_value(value).expect("deserialize mutated batch proof");
                }),
            ),
            (
                "limb group stripped from the proof",
                Box::new(|proof| {
                    proof.field_inc_limbs_commitment = None;
                    let JoltProofClaims::Clear(claims) = &mut proof.claims else {
                        panic!("clear proof expected");
                    };
                    claims.field_inc_limbs = None;
                }),
            ),
            (
                "limb claims stripped while the commitment stays",
                Box::new(|proof| {
                    let JoltProofClaims::Clear(claims) = &mut proof.claims else {
                        panic!("clear proof expected");
                    };
                    claims.field_inc_limbs = None;
                }),
            ),
        ];
        for (name, tamper) in tampers {
            let mut tampered = output.proof.clone();
            tamper(&mut tampered);
            assert!(
                support::verify_full(&output.verifier_preprocessing, &output.public_io, &tampered)
                    .is_err(),
                "tampered packed FR proof must be rejected: {name}",
            );
        }

        // The limb-evaluation offset must reject through the linear
        // recomposition check specifically, not some later transcript
        // divergence.
        let mut offset_limb = output.proof.clone();
        {
            let JoltProofClaims::Clear(claims) = &mut offset_limb.claims else {
                panic!("clear proof expected");
            };
            let limbs = claims
                .field_inc_limbs
                .as_mut()
                .expect("packed FR proof carries limb claims");
            *limbs.limbs.first_mut().expect("two limbs") += one;
        }
        assert!(matches!(
            support::verify_full(
                &output.verifier_preprocessing,
                &output.public_io,
                &offset_limb
            ),
            Err(jolt_verifier::VerifierError::FieldIncLimbRecompositionMismatch)
        ));
    }

    /// A spurious second FR-role group in the heterogeneous batch statement
    /// must be rejected by the strictly-ascending role order — the layer that
    /// makes the verifier-assembled single FR entry canonical.
    #[test]
    fn akita_field_inline_second_fr_group_is_rejected() {
        use jolt_claims::protocols::field_inline::lattice::field_inc_limbs_precommitted_role;
        use jolt_openings::CommitmentScheme as VerifierCommitmentScheme;
        use jolt_openings::{GroupOpeningClaim, PrecommittedClaim};
        use jolt_prover_legacy::zkvm::packed::{AkitaScheme, AkitaTranscript};
        use jolt_transcript::Transcript;

        let output = prove_eqpoly(JoltAkitaBackend::optimized());
        let commitment = output
            .proof
            .field_inc_limbs_commitment
            .clone()
            .expect("packed FR proofs carry the limb-group commitment");
        let point = vec![
            AkitaField::from_u64(3);
            jolt_openings::GroupCommitmentMetadata::num_vars(&commitment)
        ];
        let fr_claim = PrecommittedClaim::new(
            field_inc_limbs_precommitted_role(),
            GroupOpeningClaim::new(commitment, point, vec![AkitaField::from_u64(0)]),
        );
        let main = GroupOpeningClaim::new(
            output.proof.commitments.clone(),
            vec![
                AkitaField::from_u64(3);
                jolt_openings::GroupCommitmentMetadata::num_vars(&output.proof.commitments)
            ],
            vec![AkitaField::from_u64(0)],
        );
        let mut transcript = AkitaTranscript::new(b"spurious-fr-group");
        assert!(
            <AkitaScheme as VerifierCommitmentScheme>::verify_batch(
                &output.verifier_preprocessing.pcs_setup,
                &[fr_claim.clone(), fr_claim],
                &main,
                &output.proof.joint_opening_proof.main_batch,
                &mut transcript,
            )
            .is_err(),
            "a duplicated FR-role group must be rejected by the canonical order",
        );
    }
}

#[cfg(not(all(
    feature = "prover-fixtures",
    feature = "field-inline",
    feature = "akita"
)))]
#[test]
#[ignore = "enable --features prover-fixtures,field-inline,akita to run the packed field-inline e2e"]
fn akita_field_inline_e2e() {}
