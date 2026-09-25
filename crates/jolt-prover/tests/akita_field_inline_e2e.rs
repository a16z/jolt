//! Packed (Akita) field-inline parity and tamper tests over fp128.
//!
//! Both kernel backends prove the field-ops guest with full-width `FieldRdInc`
//! values and muldiv with an identically zero `FieldRdInc`, producing identical
//! wire objects. The field-increment commitment is present in both cases.
//! Guest acceptance across modes lives in `e2e_matrix.rs`.
//! Tamper cases cover the reduced increment claim, commitment layout digest,
//! batched opening proof, missing commitment, and duplicate batch role.

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
    use jolt_claims::protocols::field_inline::lattice::FieldIncLayout;
    use jolt_claims::protocols::field_inline::{
        FieldInlineCommittedPolynomial, FieldInlinePolynomialId,
    };
    use jolt_field::Ring;
    use jolt_openings::{CommitmentScheme, GroupCommitmentMetadata};
    use jolt_prover::akita::JoltAkitaBackend;
    use jolt_prover::ProverConfig;
    use jolt_verifier::proof::JoltProofClaims;
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

    struct IncFixture {
        log_t: usize,
        rd_inc: Vec<AkitaField>,
    }

    fn collect_inc(
        config: &ProverConfig,
        oracle: &dyn FieldInlineWitnessOracle<AkitaField>,
    ) -> IncFixture {
        IncFixture {
            log_t: config.trace_length.ilog2() as usize,
            rd_inc: oracle
                .oracle_table(FieldInlinePolynomialId::Committed(
                    FieldInlineCommittedPolynomial::FieldRdInc,
                ))
                .expect("FieldRdInc oracle table"),
        }
    }

    /// Commit the honest increment polynomial under a supplied layout digest.
    fn commit_inc_with_digest(fixture: &IncFixture, digest: [u8; 32]) -> AkitaCommitment {
        use jolt_openings::TransparentObjectSetup;
        use jolt_poly::Polynomial;

        let layout = FieldIncLayout::new(fixture.log_t);
        let mut evaluations = fixture.rd_inc.clone();
        evaluations.resize(1usize << layout.num_vars(), AkitaField::from_u64(0));
        let polynomial = Polynomial::new(evaluations);
        let (commitment, _hint) =
            <AkitaScheme as TransparentObjectSetup>::commit_full_width_object(
                &AkitaScheduleArtifacts::shared_from_default_directory(),
                &polynomial,
                digest,
            )
            .expect("full-width field-increment commit");
        commitment
    }

    /// Both backends' packed field-inline proofs must verify AND be equal wire objects.
    #[test]
    fn akita_field_inline_field_ops_backends_have_identical_proofs() {
        let mut case = field_ops();
        // Exercise a joint opening with both bounded advice and full-width increments.
        case.untrusted_advice = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let mut proofs = Vec::new();
        for (label, backend) in backends() {
            let (output, inc) = akita::prove(&case, backend(), collect_inc);
            assert!(
                output.proof.untrusted_advice_commitment.is_some(),
                "the mixed batch must include bounded advice ({label})",
            );
            assert!(
                output.proof.field_inc_commitment.is_some(),
                "packed field-inline proofs must carry the field-increment commitment ({label})",
            );
            assert!(
                inc.rd_inc.iter().any(|value| {
                    value.to_canonical_u128() > u128::from(u64::MAX)
                        && (-*value).to_canonical_u128() > u128::from(u64::MAX)
                }),
                "the guest must exercise increments outside the bounded dense envelope ({label})",
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

    /// Dense schedules depend on shape, so an inactive field register file
    /// still carries a commitment and opens its all-zero increment polynomial.
    #[test]
    fn akita_field_inline_muldiv_backends_have_identical_zero_inc_proofs() {
        let mut proofs = Vec::new();
        for (label, backend) in backends() {
            let (output, inc) = akita::prove(&muldiv(), backend(), collect_inc);
            assert!(inc
                .rd_inc
                .iter()
                .all(|value| *value == AkitaField::from_u64(0)));
            assert!(
                output.proof.field_inc_commitment.is_some(),
                "a zero FieldRdInc still carries its commitment ({label})",
            );
            let JoltProofClaims::Clear(claims) = &output.proof.claims else {
                panic!("packed proofs carry clear claims");
            };
            assert_eq!(
                claims.stage6b.field_registers_inc_claim_reduction.rd_inc,
                AkitaField::from_u64(0),
                "an all-zero polynomial opens to zero ({label})",
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
        let (output, inc) = akita::prove(&field_ops(), JoltAkitaBackend::optimized(), collect_inc);
        akita::verify_full(
            &output.verifier_preprocessing,
            &output.public_io,
            &output.proof,
        )
        .expect("base proof must verify before tampering");
        let one = AkitaField::from_u64(1);

        // Bind the layout identity independently of the committed values.
        let wrong_digest_commitment = {
            let honest = output
                .proof
                .field_inc_commitment
                .as_ref()
                .expect("packed field-inline proofs carry the field-increment commitment");
            let digest = GroupCommitmentMetadata::layout_digest(honest);
            // The forgery path reproduces the prover's commit exactly under
            // the honest digest, so the flipped-digest commitment below
            // differs from the honest one only in the digest.
            assert_eq!(
                &commit_inc_with_digest(&inc, digest),
                honest,
                "the test's increment commit must reproduce the prover's under the honest digest",
            );
            let mut digest = digest;
            digest[0] ^= 0x01;
            commit_inc_with_digest(&inc, digest)
        };

        type Tamper = (&'static str, Box<dyn Fn(&mut Proof)>);
        let tampers: Vec<Tamper> = vec![
            (
                "reduced field-increment claim offset",
                Box::new(move |proof| {
                    let JoltProofClaims::Clear(claims) = &mut proof.claims else {
                        panic!("clear proof expected");
                    };
                    claims.stage6b.field_registers_inc_claim_reduction.rd_inc += one;
                }),
            ),
            (
                "field-increment commitment layout-digest byte flip",
                Box::new(move |proof| {
                    proof.field_inc_commitment = Some(wrong_digest_commitment.clone());
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
                "field-increment commitment stripped from the proof",
                Box::new(|proof| {
                    proof.field_inc_commitment = None;
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
    }

    /// A spurious second field-inline-role group in the heterogeneous batch statement
    /// must be rejected by the strictly-ascending role order — the layer that
    /// makes the verifier-assembled single field-inline entry canonical.
    #[test]
    fn akita_field_inline_duplicate_inc_group_is_rejected() {
        use jolt_claims::protocols::field_inline::lattice::field_inc_precommitted_role;
        use jolt_openings::{GroupOpeningClaim, PrecommittedClaim};
        use jolt_prover::akita::preprocessing::AkitaTranscript;
        use jolt_transcript::Transcript;

        let (output, ()) = akita::prove(&field_ops(), JoltAkitaBackend::optimized(), |_, _| ());
        let commitment = output
            .proof
            .field_inc_commitment
            .clone()
            .expect("packed field-inline proofs carry the field-increment commitment");
        let point = vec![AkitaField::from_u64(3); GroupCommitmentMetadata::num_vars(&commitment)];
        let field_claim = PrecommittedClaim::new(
            field_inc_precommitted_role(),
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
