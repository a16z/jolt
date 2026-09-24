//! Packed (Akita) field-inline parity and tamper tests over fp128.
//!
//! Parity and limb-group invariants: the eq-MLE guest (every shipped field-inline
//! instruction family, a live `FieldRdInc` column) and the field-inline muldiv
//! (zero field-inline instructions, `FieldRdInc` identically zero, the limb group PRESENT with all-zero
//! content — the always-present rule, pinning the all-zero dense open), each
//! over both kernel backends with wire equality. Guest acceptance across modes
//! lives in `e2e_matrix.rs`; all suites share guest preparation.
//! Tamper (all must reject): a limb-evaluation offset (the stage-8 linear
//! recomposition check), a limb-commitment layout-digest byte flip, a
//! batch-proof mutation, the limb group stripped from the proof, and a
//! spurious second field-inline-role group in the heterogeneous batch statement.

#[cfg(all(
    feature = "prover-fixtures",
    feature = "field-inline",
    feature = "akita"
))]
mod support;

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
    use jolt_akita::{AkitaCommitment, AkitaField, AkitaScheduleArtifacts, AkitaScheme};
    use jolt_claims::protocols::field_inline::{
        FieldInlineCommittedPolynomial, FieldInlinePolynomialId,
    };
    use jolt_field::Ring;
    use jolt_openings::{CommitmentScheme, GroupCommitmentMetadata};
    use jolt_prover::akita::JoltAkitaBackend;
    use jolt_prover::ProverConfig;
    use jolt_verifier::proof::JoltProofClaims;
    use jolt_verifier::stages::stage8::field_inline_packed::FieldIncLimbClaims;
    use jolt_verifier::VerifierError;
    use jolt_witness::field_inline::FieldInlineWitnessOracle;
    use serde_json::Value;

    use crate::support::field_inline::akita::Proof;
    use crate::support::field_inline::{akita, field_ops, muldiv};

    /// A labeled packed kernel-backend constructor.
    type BackendCase = (
        &'static str,
        fn() -> JoltAkitaBackend<AkitaField, AkitaScheme>,
    );

    fn backends() -> [BackendCase; 2] {
        [
            ("reference", JoltAkitaBackend::reference),
            ("optimized", JoltAkitaBackend::optimized),
        ]
    }

    struct LimbFixture {
        log_t: usize,
        rd_inc: Vec<AkitaField>,
    }

    fn collect_limbs(
        config: &ProverConfig,
        oracle: &dyn FieldInlineWitnessOracle<AkitaField>,
    ) -> LimbFixture {
        LimbFixture {
            log_t: config.trace_length.ilog2() as usize,
            rd_inc: oracle
                .oracle_table(FieldInlinePolynomialId::Committed(
                    FieldInlineCommittedPolynomial::FieldRdInc,
                ))
                .expect("FieldRdInc oracle table"),
        }
    }

    /// Commit the honest limb-word polynomial under `digest` through the real
    /// dense commit path, returning the commitment a tamper splices into a
    /// proof (same content, mutated identity).
    fn commit_limb_words_with_digest(fixture: &LimbFixture, digest: [u8; 32]) -> AkitaCommitment {
        use jolt_claims::protocols::field_inline::lattice::canonical_limbs;
        use jolt_openings::TransparentObjectSetup;
        use jolt_poly::Polynomial;
        use jolt_verifier::stages::stage8::field_inline_packed::limb_plan;

        let log_t = fixture.log_t;
        let plan = limb_plan::<AkitaField>(log_t).expect("canonical limb plan");
        let mut evaluations =
            vec![AkitaField::from_u64(0); 1usize << plan.packing().packed_num_vars()];
        for (cycle, value) in fixture.rd_inc.iter().enumerate() {
            for (limb, word) in canonical_limbs(value).into_iter().enumerate() {
                evaluations[(limb << log_t) | cycle] = AkitaField::from_u64(word);
            }
        }
        let polynomial = Polynomial::new(evaluations);
        let (setup, _) = <AkitaScheme as TransparentObjectSetup>::transparent_object_setup(
            &AkitaScheduleArtifacts::shared_from_default_directory(),
            plan.packing().packed_num_vars(),
            digest,
        )
        .expect("transparent limb setup");
        let (commitment, _hint) = <AkitaScheme as CommitmentScheme>::commit(&polynomial, &setup)
            .expect("forged limb commit");
        commitment
    }

    fn clear_limb_claims(proof: &Proof) -> &FieldIncLimbClaims<AkitaField> {
        let JoltProofClaims::Clear(claims) = &proof.claims else {
            panic!("packed proofs carry clear claims");
        };
        claims
            .field_inc_limbs
            .as_ref()
            .expect("packed field-inline proofs carry the limb claims")
    }

    /// Both backends' packed field-inline proofs must verify AND be equal wire objects.
    #[test]
    fn akita_field_inline_field_ops_backends_have_identical_proofs() {
        let mut proofs = Vec::new();
        for (label, backend) in backends() {
            let (output, ()) = akita::prove(&field_ops(), backend(), |_, _| ());
            assert!(
                output.proof.field_inc_limbs_commitment.is_some(),
                "packed field-inline proofs must carry the limb-group commitment ({label})",
            );
            assert_eq!(
                clear_limb_claims(&output.proof).limbs.len(),
                2,
                "fp128 decomposes FieldRdInc into two u64 limbs ({label})",
            );
            akita::verify_full(
                &output.verifier_preprocessing,
                &output.public_io,
                &output.proof,
            )
            .unwrap_or_else(|error| {
                panic!("packed field-inline proof must verify ({label}): {error}")
            });
            proofs.push(output.proof);
        }
        assert!(
            proofs[0] == proofs[1],
            "reference and optimized packed field-inline proofs must be identical wire objects",
        );
    }

    /// The uniform-shape degenerate case: a field-inline guest executing zero
    /// field-inline instructions — `FieldRdInc` identically zero, every limb word zero
    /// — still proves and verifies with the limb group PRESENT (all-zero
    /// content is legal: dense schedules are keyed by shape, never content).
    /// This pins the all-zero dense open.
    #[test]
    fn akita_field_inline_muldiv_backends_have_identical_zero_limb_proofs() {
        let mut proofs = Vec::new();
        for (label, backend) in backends() {
            let (output, limbs) = akita::prove(&muldiv(), backend(), collect_limbs);
            assert!(limbs
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
            akita::verify_full(
                &output.verifier_preprocessing,
                &output.public_io,
                &output.proof,
            )
            .unwrap_or_else(|error| {
                panic!("field-inactive packed proof must verify ({label}): {error}")
            });
            proofs.push(output.proof);
        }
        assert!(
            proofs[0] == proofs[1],
            "reference and optimized field-inactive packed proofs must be identical wire objects",
        );
    }

    /// The packed field-inline tamper matrix: one honest proof, mutations on fresh
    /// clones, every one rejected.
    #[test]
    fn akita_field_inline_tampered_proofs_are_rejected() {
        let (output, limbs) =
            akita::prove(&field_ops(), JoltAkitaBackend::optimized(), collect_limbs);
        akita::verify_full(
            &output.verifier_preprocessing,
            &output.public_io,
            &output.proof,
        )
        .expect("base proof must verify before tampering");
        let one = AkitaField::from_u64(1);

        // The honest limb polynomial under a corrupted layout digest, through
        // the real commit path. The digest is part of the commitment's
        // absorbed identity, so the stage-0 transcript already diverges; the
        // stage-8 metadata gate is the backstop that pins the digest to the
        // canonical plan even off-transcript. Either layer rejects.
        let wrong_digest_commitment = {
            let honest = output
                .proof
                .field_inc_limbs_commitment
                .as_ref()
                .expect("packed field-inline proofs carry the limb-group commitment");
            let digest = GroupCommitmentMetadata::layout_digest(honest);
            // The forgery path reproduces the prover's commit exactly under
            // the honest digest, so the flipped-digest commitment below
            // differs from the honest one only in the digest.
            assert_eq!(
                &commit_limb_words_with_digest(&limbs, digest),
                honest,
                "the test's limb commit must reproduce the prover's under the honest digest",
            );
            let mut digest = digest;
            digest[0] ^= 0x01;
            commit_limb_words_with_digest(&limbs, digest)
        };

        type Tamper = (&'static str, Box<dyn Fn(&mut Proof)>);
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
                        .expect("packed field-inline proof carries limb claims");
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
                    let mut value = serde_json::to_value(&proof.joint_opening_proof)
                        .expect("serialize batch proof");
                    let bytes = value
                        .get_mut("backend_proof")
                        .and_then(Value::as_array_mut)
                        .expect("batch proof carries the headerless backend proof body");
                    let mid = bytes.len() / 2;
                    let byte = bytes.get_mut(mid).expect("nonempty backend proof");
                    let flipped = byte.as_u64().expect("byte value") ^ 0x01;
                    *byte = Value::from(flipped);
                    proof.joint_opening_proof =
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
                akita::verify_full(&output.verifier_preprocessing, &output.public_io, &tampered)
                    .is_err(),
                "tampered packed field-inline proof must be rejected: {name}",
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
                .expect("packed field-inline proof carries limb claims");
            *limbs.limbs.first_mut().expect("two limbs") += one;
        }
        assert!(matches!(
            akita::verify_full(
                &output.verifier_preprocessing,
                &output.public_io,
                &offset_limb
            ),
            Err(VerifierError::FieldIncLimbRecompositionMismatch)
        ));
    }

    /// A spurious second field-inline-role group in the heterogeneous batch statement
    /// must be rejected by the strictly-ascending role order — the layer that
    /// makes the verifier-assembled single field-inline entry canonical.
    #[test]
    fn akita_field_inline_duplicate_limb_group_is_rejected() {
        use jolt_claims::protocols::field_inline::lattice::field_inc_limbs_precommitted_role;
        use jolt_openings::{GroupOpeningClaim, PrecommittedClaim};
        use jolt_prover::akita::preprocessing::AkitaTranscript;
        use jolt_transcript::Transcript;

        let (output, ()) = akita::prove(&field_ops(), JoltAkitaBackend::optimized(), |_, _| ());
        let commitment = output
            .proof
            .field_inc_limbs_commitment
            .clone()
            .expect("packed field-inline proofs carry the limb-group commitment");
        let point = vec![AkitaField::from_u64(3); GroupCommitmentMetadata::num_vars(&commitment)];
        let field_claim = PrecommittedClaim::new(
            field_inc_limbs_precommitted_role(),
            GroupOpeningClaim::new(commitment, point, vec![AkitaField::from_u64(0)]),
        );
        let main = GroupOpeningClaim::new(
            output.proof.commitments.clone(),
            vec![
                AkitaField::from_u64(3);
                GroupCommitmentMetadata::num_vars(&output.proof.commitments)
            ],
            vec![AkitaField::from_u64(0)],
        );
        let mut transcript = AkitaTranscript::new(b"spurious-field_inline-group");
        let error = <AkitaScheme as CommitmentScheme>::verify_batch(
            &output.verifier_preprocessing.pcs_setup,
            &[field_claim.clone(), field_claim],
            &main,
            &output.proof.joint_opening_proof,
            &mut transcript,
        )
        .expect_err("a duplicated field-inline-role group must be rejected");
        // The rejection must come from the canonical role-order check, not
        // from the fake statement failing later in the batch.
        assert!(
            error.to_string().contains("canonical ascending order"),
            "duplicated field-inline role rejected for the wrong reason: {error}",
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
