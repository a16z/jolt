//! Whiteout end to end on the Akita build (`specs/whiteout-jolt-integration.md`):
//! honest proofs verify and tampering rejects, for muldiv alone, beside
//! untrusted and trusted advice, and beside a committed program. The
//! verifier-side suites (census, sweeps, compensating shifts) are
//! jolt-verifier's `soundness::whiteout`.

#[cfg(all(feature = "prover-fixtures", feature = "akita", feature = "zk"))]
#[cfg_attr(
    feature = "field-inline",
    expect(
        dead_code,
        reason = "the field-inline Whiteout test proves only field-ops"
    )
)]
mod support;

#[cfg(all(
    feature = "prover-fixtures",
    feature = "akita",
    feature = "zk",
    not(feature = "field-inline")
))]
#[expect(
    clippy::expect_used,
    clippy::indexing_slicing,
    reason = "integration tests should fail loudly"
)]
mod whiteout {
    use common::jolt_device::JoltDevice;
    use jolt_akita::{
        AkitaChunkProfile, AkitaCommitment, AkitaField, AkitaScheduleArtifacts, AkitaScheme,
        HidingRandomness, WhiteoutPlans,
    };
    use jolt_algebra::Ring;
    use jolt_prover::akita;
    use jolt_prover::akita::preprocessing::{
        self, AkitaProverPreprocessing, AkitaVc, AkitaVerifierPreprocessing,
    };
    use jolt_prover::akita::witness::{AdviceObject, DirectProgramOpenings, HidingOpening};
    use jolt_prover::{JoltBackend, PreprocessingError, ProverConfig, ProverError};
    use jolt_transcript::{Keccak, ProverState, VerifierState};
    use jolt_verifier::proof::{JoltProof, JoltProofClaims};
    use jolt_verifier::VerifierError;
    use jolt_witness::{JoltVmWitnessConfig, JoltVmWitnessInputs, TraceBackend};

    use crate::support::{self, GuestCase, PreparedGuest};

    type F = AkitaField;
    /// A Whiteout proof: no clear opening proof, the tail in the NARG string.
    type WhiteoutProof = JoltProof<AkitaScheme, AkitaVc, (), ()>;

    /// How a case is preprocessed.
    #[derive(Clone, Copy)]
    enum Setup {
        /// The production setup.
        Muldiv,
        /// Production, with a committed program.
        Committed,
        /// Production, with untrusted and trusted advice.
        Advice,
    }

    struct Proved {
        preprocessing: AkitaProverPreprocessing,
        public_io: JoltDevice,
        trusted: Option<AkitaCommitment>,
        proof: Result<WhiteoutProof, ProverError<F>>,
    }

    fn prepare(setup: Setup) -> PreparedGuest {
        match setup {
            Setup::Muldiv | Setup::Committed => support::prepare(&GuestCase {
                inputs: postcard::to_stdvec(&[9u32, 5u32, 3u32]).expect("serialize inputs"),
                ..GuestCase::new("muldiv-guest")
            }),
            Setup::Advice => support::prepare(&GuestCase {
                inputs: postcard::to_stdvec(&12u64).expect("serialize inputs"),
                untrusted_advice: postcard::to_stdvec(&5u64).expect("serialize advice"),
                trusted_advice: postcard::to_stdvec(&7u64).expect("serialize advice"),
                ..GuestCase::new("advice-consumer-guest")
            }),
        }
    }

    fn config(run: &PreparedGuest) -> ProverConfig {
        ProverConfig::derive_from_dimensions::<F>(
            run.trace.dimensions,
            &run.preprocessing.memory_layout,
            run.preprocessing.ram.min_bytecode_address,
            run.preprocessing.ram.bytecode_words.len(),
            run.preprocessing.max_padded_trace_length,
        )
        .expect("derive config")
    }

    fn preprocess(
        setup: Setup,
        run: &PreparedGuest,
        config: &ProverConfig,
    ) -> AkitaProverPreprocessing {
        let artifacts =
            AkitaScheduleArtifacts::shared_from_default_directory(AkitaChunkProfile::new(8, 2));
        let advice = matches!(setup, Setup::Advice);
        match setup {
            Setup::Muldiv | Setup::Advice => preprocessing::preprocess_full_with_advice(
                &artifacts,
                run.preprocessing.clone(),
                config,
                advice,
                advice,
            ),
            Setup::Committed => {
                preprocessing::preprocess_committed(&artifacts, run.preprocessing.clone(), config)
            }
        }
        .expect("Akita preprocessing")
    }

    fn prove(setup: Setup) -> Proved {
        let run = prepare(setup);
        let config = config(&run);
        let preprocessing = preprocess(setup, &run, &config);
        let trusted = matches!(setup, Setup::Advice).then(|| {
            preprocessing::commit_trusted_advice(&preprocessing, &trusted_advice())
                .expect("trusted advice commitment")
        });
        prove_with(run, &config, preprocessing, trusted)
    }

    fn trusted_advice() -> Vec<u8> {
        postcard::to_stdvec(&7u64).expect("serialize advice")
    }

    /// Proves `run` on `preprocessing`, beside `trusted` advice.
    fn prove_with(
        run: PreparedGuest,
        config: &ProverConfig,
        preprocessing: AkitaProverPreprocessing,
        trusted: Option<AdviceObject<AkitaScheme>>,
    ) -> Proved {
        let advice = trusted.is_some();
        let program = preprocessing.program_arc().expect("full program");
        let public_io = run.trace.device.clone();
        let witness = TraceBackend::new(
            JoltVmWitnessConfig::new(
                config.trace_length.ilog2() as usize,
                config.ram_K,
                config.one_hot_config,
            )
            .include_untrusted_advice(advice)
            .include_trusted_advice(advice),
            JoltVmWitnessInputs::new(&run.program, &program, run.trace),
        );
        let proof = akita::prove::<F, AkitaScheme, AkitaVc, ProverState<Keccak>>(
            JoltBackend::optimized().with_witness(&witness),
            &preprocessing,
            config,
            trusted.as_ref(),
            &public_io,
        );
        Proved {
            preprocessing,
            public_io,
            trusted: trusted.map(|object| object.commitment),
            proof,
        }
    }

    fn verify(proved: &Proved, proof: &WhiteoutProof) -> Result<(), VerifierError> {
        verify_with(proved, proof, proved.trusted.as_ref())
    }

    fn verify_with(
        proved: &Proved,
        proof: &WhiteoutProof,
        trusted: Option<&AkitaCommitment>,
    ) -> Result<(), VerifierError> {
        jolt_verifier::verify::<F, AkitaScheme, AkitaVc, VerifierState<'_, Keccak>>(
            &proved.preprocessing.verifier,
            &proved.public_io,
            proof,
            trusted,
        )
    }

    /// The verifier preprocessing after a serde round trip: the copy shares
    /// no derived state with the prover's. It loads the persisted catalog
    /// plan; a setup persisted without response plans plans its response
    /// schedule on first use.
    fn transported(preprocessing: &AkitaVerifierPreprocessing) -> AkitaVerifierPreprocessing {
        let encoded = bincode::serde::encode_to_vec(preprocessing, bincode::config::standard())
            .expect("encode verifier preprocessing");
        bincode::serde::decode_from_slice(&encoded, bincode::config::standard())
            .expect("decode verifier preprocessing")
            .0
    }

    /// An honest proof verifies against transported preprocessing.
    fn assert_verifies_transported(proved: &Proved, proof: &WhiteoutProof) {
        jolt_verifier::verify::<F, AkitaScheme, AkitaVc, VerifierState<'_, Keccak>>(
            &transported(&proved.preprocessing.verifier),
            &proved.public_io,
            proof,
            proved.trusted.as_ref(),
        )
        .expect("an honest proof verifies against transported preprocessing");
    }

    fn honest(setup: Setup) -> (Proved, WhiteoutProof) {
        let mut proved = prove(setup);
        let proof = std::mem::replace(
            &mut proved.proof,
            Err(ProverError::InvariantViolation { reason: "taken" }),
        )
        .expect("Whiteout proof");
        (proved, proof)
    }

    #[test]
    fn whiteout_muldiv_verifies_and_rejects_tampering() {
        let (proved, proof) = honest(Setup::Muldiv);
        verify(&proved, &proof).expect("an honest Whiteout proof verifies");
        assert_verifies_transported(&proved, &proof);

        let len = proof.narg.len();
        for position in [len / 4, len / 2, 3 * len / 4, len - 1] {
            let mut tampered = proof.clone();
            tampered.narg[position] ^= 1;
            assert!(
                verify(&proved, &tampered).is_err(),
                "a flipped bit at {position} of {len} must reject"
            );
        }
        let mut truncated = proof.clone();
        let _ = truncated.narg.pop();
        assert!(verify(&proved, &truncated).is_err());
        let mut extended = proof.clone();
        extended.narg.push(0);
        assert!(verify(&proved, &extended).is_err());
        assert_tape_tampers_reject(&proved, &proof);

        // A second proof of the same statement draws fresh hiding
        // randomness, pads and Σ coins.
        let (again, second) = honest(Setup::Muldiv);
        verify(&again, &second).expect("a second Whiteout proof verifies");
        assert_ne!(proof.narg, second.narg);
    }

    fn encode(preprocessing: &AkitaVerifierPreprocessing) -> Vec<u8> {
        bincode::serde::encode_to_vec(preprocessing, bincode::config::standard())
            .expect("encode verifier preprocessing")
    }

    /// Serialized preprocessing is a function of its setup, not of the
    /// proofs made on it: a proved setup, an independent preprocessing run
    /// and a decode and re-encode give the same bytes, with or without
    /// exported plans. A prover and a verifier built from exported plans load
    /// them: a proof on them verifies against the transported preprocessing,
    /// and plans of another format are refused when the prover first uses
    /// them, never replanned.
    #[test]
    fn whiteout_plans_persist_and_are_consumed() {
        let (proved, _) = honest(Setup::Muldiv);
        let run = prepare(Setup::Muldiv);
        let config = config(&run);
        let bytes = encode(&proved.preprocessing.verifier);
        assert_eq!(
            encode(&preprocess(Setup::Muldiv, &run, &config).verifier),
            bytes
        );
        assert_eq!(encode(&transported(&proved.preprocessing.verifier)), bytes);

        let plans = proved
            .preprocessing
            .verifier
            .pcs_setup
            .whiteout_plans()
            .expect("the proved setup's plans");
        let artifacts =
            AkitaScheduleArtifacts::shared_from_default_directory(AkitaChunkProfile::new(8, 2));
        let from_plans = |plans: WhiteoutPlans| {
            preprocessing::preprocess_full_from_plans(
                &artifacts,
                run.preprocessing.clone(),
                &config,
                false,
                false,
                plans,
            )
            .expect("preprocessing from plans")
        };
        let loaded = from_plans(plans.clone());
        let persisted = transported(&loaded.verifier);
        assert_eq!(encode(&persisted), encode(&loaded.verifier));
        assert_ne!(encode(&persisted), bytes);

        // `WhiteoutPlans`' wire form opens with its 4-byte tag, then its
        // version as a one-byte varint.
        let mut wire = bincode::serde::encode_to_vec(&plans, bincode::config::standard())
            .expect("encode plans");
        assert_eq!(wire[4], 1);
        wire[4] = 2;
        let (versioned, _): (WhiteoutPlans, _) =
            bincode::serde::decode_from_slice(&wire, bincode::config::standard())
                .expect("decode plans");
        let refused = prove_with(prepare(Setup::Muldiv), &config, from_plans(versioned), None);
        // The trace commitment, Stage 0, is the first use; the commitment
        // path carries the refusal as an `OpeningsError` message.
        let error = refused
            .proof
            .expect_err("plans of another format must be refused");
        assert!(
            error
                .to_string()
                .contains("version 2 are not this build's format"),
            "{error}"
        );

        let reproved = prove_with(prepare(Setup::Muldiv), &config, loaded, None);
        let proof = reproved.proof.expect("Whiteout proof on persisted plans");
        jolt_verifier::verify::<F, AkitaScheme, AkitaVc, VerifierState<'_, Keccak>>(
            &persisted,
            &reproved.public_io,
            &proof,
            None,
        )
        .expect("a proof on persisted plans verifies against the transported preprocessing");
    }

    /// Flipped bits across the argument string, truncation and an appended
    /// byte reject.
    fn assert_byte_tampers_reject(proved: &Proved, proof: &WhiteoutProof) {
        let len = proof.narg.len();
        for position in [len / 8, len / 4, len / 2, 3 * len / 4, len - 1] {
            let mut tampered = proof.clone();
            tampered.narg[position] ^= 1;
            assert!(
                verify(proved, &tampered).is_err(),
                "a flipped bit at {position} of {len} must reject"
            );
        }
        let mut truncated = proof.clone();
        let _ = truncated.narg.pop();
        assert!(verify(proved, &truncated).is_err());
        assert_tape_tampers_reject(proved, proof);
    }

    /// Every masked value is bound: a shifted value at several positions of
    /// the masked tape, a dropped value and an appended one reject.
    #[expect(clippy::panic, reason = "a clear proof here is a test failure")]
    fn assert_tape_tampers_reject(proved: &Proved, proof: &WhiteoutProof) {
        let JoltProofClaims::Zk { masked, .. } = &proof.claims else {
            panic!("a Whiteout proof carries a masked tape");
        };
        let len = masked.values.len();
        assert!(len > 0, "a Whiteout proof masks its stage messages");
        let tamper = |edit: &dyn Fn(&mut Vec<F>)| {
            let mut tampered = proof.clone();
            let JoltProofClaims::Zk { masked, .. } = &mut tampered.claims else {
                unreachable!("checked above");
            };
            edit(&mut masked.values);
            tampered
        };
        for position in [0, len / 4, len / 2, 3 * len / 4, len - 1] {
            assert!(
                verify(
                    proved,
                    &tamper(&|values| values[position] += F::from_u64(1))
                )
                .is_err(),
                "a shifted masked value at {position} of {len} must reject"
            );
        }
        assert!(verify(
            proved,
            &tamper(&|values| {
                let _ = values.pop();
            })
        )
        .is_err());
        assert!(verify(proved, &tamper(&|values| values.push(F::from_u64(0)))).is_err());
    }

    /// Untrusted and trusted advice open as hiding precommitted groups: an
    /// honest proof verifies; byte tampers reject, as do a trusted-advice
    /// commitment of other advice, the same advice recommitted (fresh
    /// hiding randomness, so another commitment), and a missing one.
    #[test]
    fn whiteout_advice_verifies_and_rejects_tampering() {
        let (proved, proof) = honest(Setup::Advice);
        verify(&proved, &proof).expect("an honest Whiteout advice proof verifies");
        assert_verifies_transported(&proved, &proof);
        assert_byte_tampers_reject(&proved, &proof);
        let trusted = proved.trusted.as_ref().expect("trusted advice");
        let other = preprocessing::commit_trusted_advice(
            &proved.preprocessing,
            &postcard::to_stdvec(&8u64).expect("serialize advice"),
        )
        .expect("other trusted advice")
        .commitment;
        let again = preprocessing::commit_trusted_advice(
            &proved.preprocessing,
            &postcard::to_stdvec(&7u64).expect("serialize advice"),
        )
        .expect("the same trusted advice")
        .commitment;
        assert_ne!(&again, trusted, "a hiding recommit draws fresh randomness");
        for (label, commitment) in [("other advice", Some(&other)), ("a recommit", Some(&again))] {
            assert!(
                verify_with(&proved, &proof, commitment).is_err(),
                "a trusted-advice commitment of {label} is accepted"
            );
        }
        assert!(verify_with(&proved, &proof, None).is_err());
    }

    /// A committed program opens as two hiding precommitted groups: an
    /// honest proof verifies; byte tampers reject, as does verifier
    /// preprocessing that holds a recommitted program (fresh randomness).
    #[test]
    fn whiteout_committed_program_verifies_and_rejects_tampering() {
        let (proved, proof) = honest(Setup::Committed);
        verify(&proved, &proof).expect("an honest Whiteout committed-program proof verifies");
        assert_verifies_transported(&proved, &proof);
        assert_byte_tampers_reject(&proved, &proof);
        let run = prepare(Setup::Committed);
        let recommitted = preprocess(Setup::Committed, &run, &config(&run));
        assert_ne!(
            recommitted.verifier.program.committed(),
            proved.preprocessing.verifier.program.committed(),
            "a hiding recommit draws fresh randomness"
        );
        assert!(
            jolt_verifier::verify::<F, AkitaScheme, AkitaVc, VerifierState<'_, Keccak>>(
                &recommitted.verifier,
                &proved.public_io,
                &proof,
                None,
            )
            .is_err(),
            "a proof verifies against another commitment to its program"
        );
    }

    /// The opening a trusted party ships, as bytes for the randomness: the
    /// committing process's handle is gone once it returns.
    fn ship(opening: HidingOpening<AkitaScheme>) -> (AkitaCommitment, Vec<u8>) {
        (opening.commitment, opening.randomness.to_bytes().to_vec())
    }

    fn receive(commitment: &AkitaCommitment, randomness: &[u8]) -> HidingOpening<AkitaScheme> {
        HidingOpening {
            commitment: commitment.clone(),
            randomness: HidingRandomness::from_bytes(randomness).expect("decode randomness"),
        }
    }

    /// A copy of `randomness` with its first digit (after the 13-byte
    /// header) moved by one, still inside the balanced alphabet.
    fn altered(randomness: &[u8]) -> Vec<u8> {
        let mut altered = randomness.to_vec();
        altered[13] ^= 1;
        altered
    }

    /// A trusted party commits trusted advice hiding and ships the
    /// commitment and its randomness. The prover, in another process,
    /// rebuilds its hint from (advice, randomness) and proves; the verifier
    /// accepts against the shipped commitment. Altered randomness or other
    /// advice is refused at rebuild, and an auditor's rebuild reproduces
    /// the commitment.
    #[test]
    #[expect(clippy::panic, reason = "an unexpected refusal is a test failure")]
    fn whiteout_trusted_advice_reopens_from_shipped_randomness() {
        let (commitment, randomness) = {
            let run = prepare(Setup::Advice);
            let trusted_party = preprocess(Setup::Advice, &run, &config(&run));
            let object = preprocessing::commit_trusted_advice(&trusted_party, &trusted_advice())
                .expect("trusted advice commitment");
            ship(object.hiding_opening().expect("export the advice opening"))
        };

        let run = prepare(Setup::Advice);
        let config = config(&run);
        let prover = preprocess(Setup::Advice, &run, &config);
        let reopen = |advice: &[u8], randomness: &[u8]| {
            preprocessing::reopen_trusted_advice(&prover, advice, &receive(&commitment, randomness))
        };
        let refusal = |result: Result<AdviceObject<AkitaScheme>, _>| match result {
            Err(PreprocessingError::InvalidAdvice { reason }) => reason,
            Err(error) => panic!("unexpected refusal: {error:?}"),
            Ok(_) => panic!("a mismatched rebuild was accepted"),
        };
        // The digit stays inside the alphabet, so the refusal is the
        // commitment mismatch, not the randomness decoding.
        assert!(
            refusal(reopen(&trusted_advice(), &altered(&randomness)))
                .contains("do not open the expected commitment"),
            "altered randomness rebuilds the commitment"
        );
        assert!(
            refusal(reopen(
                &postcard::to_stdvec(&8u64).expect("serialize advice"),
                &randomness
            ))
            .contains("do not open the expected commitment"),
            "other advice rebuilds the commitment"
        );
        // A shipped commitment whose backend payload matches but whose
        // header (the layout digest, right after the flavor byte) does not.
        let mut header = postcard::to_stdvec(&commitment).expect("serialize commitment");
        header[1] ^= 1;
        let header: AkitaCommitment = postcard::from_bytes(&header).expect("decode commitment");
        assert!(
            refusal(preprocessing::reopen_trusted_advice(
                &prover,
                &trusted_advice(),
                &receive(&header, &randomness)
            ))
            .contains("header differs"),
            "a shipped commitment with another header is refused"
        );
        let audited = reopen(&trusted_advice(), &randomness).expect("the auditor's rebuild");
        assert_eq!(audited.commitment, commitment);

        let object = reopen(&trusted_advice(), &randomness).expect("the prover's rebuild");
        let mut proved = prove_with(run, &config, prover, Some(object));
        let proof = std::mem::replace(
            &mut proved.proof,
            Err(ProverError::InvariantViolation { reason: "taken" }),
        )
        .expect("Whiteout proof on reopened trusted advice");
        verify_with(&proved, &proof, Some(&commitment))
            .expect("the verifier accepts against the shipped commitment");
    }

    /// A trusted party commits a program hiding and ships its verifier
    /// preprocessing and the objects' randomness. The prover rebuilds its
    /// preprocessing from (program, randomness) and proves; the trusted
    /// party's verifier preprocessing accepts. Altered or swapped randomness
    /// is refused at rebuild.
    #[test]
    fn whiteout_committed_program_reopens_from_shipped_randomness() {
        let run = prepare(Setup::Committed);
        let config = config(&run);
        let artifacts =
            AkitaScheduleArtifacts::shared_from_default_directory(AkitaChunkProfile::new(8, 2));
        let (verifier, bytecode, program_image) = {
            let trusted_party = preprocess(Setup::Committed, &run, &config);
            let openings = trusted_party
                .committed_program
                .as_ref()
                .expect("committed program")
                .direct_program
                .hiding_openings()
                .expect("export the program openings");
            (
                transported(&trusted_party.verifier),
                ship(openings.bytecode),
                ship(openings.program_image),
            )
        };
        let reopen = |bytecode_randomness: &[u8], program_image_randomness: &[u8]| {
            preprocessing::preprocess_committed_reopened(
                &artifacts,
                run.preprocessing.clone(),
                &config,
                false,
                false,
                &DirectProgramOpenings {
                    bytecode: receive(&bytecode.0, bytecode_randomness),
                    program_image: receive(&program_image.0, program_image_randomness),
                },
            )
        };
        assert!(
            reopen(&altered(&bytecode.1), &program_image.1).is_err(),
            "altered bytecode randomness rebuilds the commitment"
        );
        assert!(
            reopen(&bytecode.1, &altered(&program_image.1)).is_err(),
            "altered program-image randomness rebuilds the commitment"
        );
        assert!(
            reopen(&program_image.1, &bytecode.1).is_err(),
            "swapped randomness rebuilds the commitments"
        );
        let prover = reopen(&bytecode.1, &program_image.1).expect("the prover's rebuild");
        assert_eq!(
            prover.verifier.program.committed(),
            verifier.program.committed()
        );

        let mut proved = prove_with(run, &config, prover, None);
        let proof = std::mem::replace(
            &mut proved.proof,
            Err(ProverError::InvariantViolation { reason: "taken" }),
        )
        .expect("Whiteout proof on a reopened committed program");
        jolt_verifier::verify::<F, AkitaScheme, AkitaVc, VerifierState<'_, Keccak>>(
            &verifier,
            &proved.public_io,
            &proof,
            None,
        )
        .expect("the trusted party's verifier preprocessing accepts");
    }
}

#[cfg(all(
    feature = "prover-fixtures",
    feature = "akita",
    feature = "zk",
    feature = "field-inline"
))]
#[expect(
    clippy::expect_used,
    clippy::indexing_slicing,
    reason = "integration tests should fail loudly"
)]
mod whiteout_field_inline {
    use jolt_prover::JoltBackend;

    use crate::support::field_inline::akita::Proof;
    use crate::support::field_inline::{akita, field_ops};

    /// The field increments open as a full-field hiding precommitted group
    /// beside untrusted advice: an honest field-ops proof, whose increments
    /// leave the `u64` envelope, verifies, and flipped bits across the
    /// argument string and truncation reject.
    #[test]
    fn whiteout_field_inline_verifies_and_rejects_tampering() {
        let mut case = field_ops();
        case.untrusted_advice = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let (output, ()) = akita::prove(&case, JoltBackend::optimized(), |_, _, _| ());
        let verify = |proof: &Proof| {
            akita::verify_full(&output.verifier_preprocessing, &output.public_io, proof)
        };
        verify(&output.proof).expect("an honest Whiteout field-inline proof verifies");
        let len = output.proof.narg.len();
        for position in [len / 8, len / 4, len / 2, 3 * len / 4, len - 1] {
            let mut tampered = output.proof.clone();
            tampered.narg[position] ^= 1;
            assert!(
                verify(&tampered).is_err(),
                "a flipped bit at {position} of {len} must reject"
            );
        }
        let mut truncated = output.proof.clone();
        let _ = truncated.narg.pop();
        assert!(verify(&truncated).is_err());
    }
}
