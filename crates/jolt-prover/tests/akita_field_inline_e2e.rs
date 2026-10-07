//! Packed (Akita) field-inline parity and tamper tests over fp128.
//!
//! Both kernel backends prove the field-ops guest with full-width `FieldRdInc`
//! values and muldiv with an identically zero `FieldRdInc`, producing identical
//! wire objects. The field-increment commitment is present in both cases.
//! Guest acceptance across modes lives in `e2e_matrix.rs`.
//! Tamper cases cover the stage spine, the field-increment commitment's layout
//! digest, the batched opening, a missing commitment, and a duplicate batch
//! role.

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
    use jolt_akita::{AkitaCommitment, AkitaField, AkitaProverSetup, AkitaScheme};
    use jolt_claims::protocols::field_inline::lattice::FieldIncLayout;
    use jolt_claims::protocols::field_inline::{
        FieldInlineCommittedPolynomial, FieldInlinePolynomialId,
    };
    use jolt_field::Ring;
    use jolt_openings::{CommitmentScheme, GroupCommitmentMetadata};
    use jolt_prover::akita::JoltAkitaBackend;
    use jolt_prover::ProverConfig;
    use std::ops::Range;

    use jolt_prover::akita::preprocessing::AkitaVc;
    use jolt_transcript::{ProverTranscript, VerifierTranscript};
    use jolt_verifier::proof::ProofHeader;
    use jolt_verifier::{jolt_protocol_id, seed_transcript, JoltSponge, JOLT_SESSION};
    use jolt_witness::field_inline::FieldInlineWitnessOracle;

    use crate::support::field_inline::akita::{Proof, ProveOutput};
    use crate::support::field_inline::{akita, field_ops, muldiv};

    /// One sent commitment and the argument-string bytes it occupies.
    struct SentCommitment {
        commitment: AkitaCommitment,
        bytes: Range<usize>,
    }

    /// The commitments a packed field-inline proof sends, in
    /// `ProofCommitments::send` order, as the verifier's own stage-0 read sees
    /// them.
    struct SentCommitments {
        one_hot_trace: SentCommitment,
        field_inc: SentCommitment,
        untrusted_advice: Option<SentCommitment>,
    }

    /// The byte width of `commitment`'s wire form.
    fn commitment_width(commitment: &AkitaCommitment) -> usize {
        let mut transcript =
            ProverTranscript::<JoltSponge>::new(&jolt_protocol_id::<JoltSponge>(), b"");
        AkitaScheme::send_commitment(commitment, &mut transcript);
        transcript.narg().len()
    }

    fn sent_commitments(output: &ProveOutput) -> SentCommitments {
        let narg = output.proof.narg.as_slice();
        let consumed =
            |transcript: &VerifierTranscript<'_, JoltSponge>| narg.len() - transcript.remaining();
        let protocol = jolt_protocol_id::<JoltSponge>();

        let mut transcript = VerifierTranscript::<JoltSponge>::new(&protocol, JOLT_SESSION, narg);
        let _header = ProofHeader::receive(&mut transcript).expect("proof header");
        let header_end = consumed(&transcript);

        let mut transcript = VerifierTranscript::<JoltSponge>::new(&protocol, JOLT_SESSION, narg);
        let commitments = seed_transcript::<AkitaScheme, AkitaVc, JoltSponge>(
            &output.verifier_preprocessing,
            &output.public_io,
            None,
            &mut transcript,
        )
        .expect("proof commitments")
        .commitments;
        let mut next = header_end;
        let mut place = |commitment: AkitaCommitment| {
            let start = next;
            next += commitment_width(&commitment);
            SentCommitment {
                commitment,
                bytes: start..next,
            }
        };
        let sent = SentCommitments {
            one_hot_trace: place(commitments.one_hot_trace),
            field_inc: place(commitments.field_inc),
            untrusted_advice: commitments.untrusted_advice.map(&mut place),
        };
        assert_eq!(
            next,
            consumed(&transcript),
            "commitments end the seeding reads"
        );
        sent
    }

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
        setup: AkitaProverSetup,
        log_t: usize,
        rd_inc: Vec<AkitaField>,
    }

    fn collect_inc(
        config: &ProverConfig,
        oracle: &dyn FieldInlineWitnessOracle<AkitaField>,
        setup: &AkitaProverSetup,
    ) -> IncFixture {
        IncFixture {
            setup: setup.clone(),
            log_t: config.trace_length.ilog2() as usize,
            rd_inc: oracle
                .oracle_table(FieldInlinePolynomialId::Committed(
                    FieldInlineCommittedPolynomial::FieldRdInc,
                ))
                .expect("FieldRdInc oracle table"),
        }
    }

    fn commit_inc_with_digest(fixture: &IncFixture, digest: [u8; 32]) -> AkitaCommitment {
        use jolt_openings::TransparentObjectSetup;
        use jolt_poly::Polynomial;

        let layout = FieldIncLayout::new(fixture.log_t);
        let mut evaluations = fixture.rd_inc.clone();
        evaluations.resize(1usize << layout.num_vars(), AkitaField::from_u64(0));
        let polynomial = Polynomial::new(evaluations);
        let (commitment, _hint) =
            <AkitaScheme as TransparentObjectSetup>::commit_full_width_object(
                &fixture.setup,
                &polynomial,
                digest,
            )
            .expect("full-width field-increment commit");
        commitment
    }

    #[test]
    fn akita_field_inline_field_ops_backends_have_identical_proofs() {
        let mut case = field_ops();
        // Exercise a joint opening with both bounded advice and full-width increments.
        case.untrusted_advice = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let mut proofs = Vec::new();
        for (label, backend) in backends() {
            let (output, inc) = akita::prove(&case, backend(), collect_inc);
            assert!(
                sent_commitments(&output).untrusted_advice.is_some(),
                "the mixed batch must include bounded advice ({label})",
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
            assert_eq!(
                sent_commitments(&output).field_inc.commitment,
                commit_inc_with_digest(&inc, FieldIncLayout::new(inc.log_t).layout_digest()),
                "a zero FieldRdInc still carries its commitment ({label})",
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

    #[test]
    fn akita_field_inline_tampered_proofs_are_rejected() {
        let (output, inc) = akita::prove(&field_ops(), JoltAkitaBackend::optimized(), collect_inc);
        akita::verify_full(
            &output.verifier_preprocessing,
            &output.public_io,
            &output.proof,
        )
        .expect("base proof must verify before tampering");
        let sent = sent_commitments(&output);

        // Bind the layout identity independently of the committed values.
        let wrong_digest_commitment = {
            let honest = &sent.field_inc.commitment;
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
        let mut wrong_digest_bytes =
            ProverTranscript::<JoltSponge>::new(&jolt_protocol_id::<JoltSponge>(), b"");
        AkitaScheme::send_commitment(&wrong_digest_commitment, &mut wrong_digest_bytes);
        let wrong_digest_bytes = wrong_digest_bytes.finish();
        let field_inc_bytes = sent.field_inc.bytes.clone();

        type Tamper = (&'static str, Box<dyn Fn(&mut Proof)>);
        let tampers: Vec<Tamper> = vec![
            (
                "stage-spine bit flip",
                Box::new(|proof| {
                    let mid = proof.narg.len() / 2;
                    proof.narg[mid] ^= 0x01;
                }),
            ),
            (
                "field-increment commitment layout-digest byte flip",
                Box::new({
                    let range = field_inc_bytes.clone();
                    move |proof| {
                        let _ = proof
                            .narg
                            .splice(range.clone(), wrong_digest_bytes.iter().copied());
                    }
                }),
            ),
            (
                "batched opening proof mutation",
                Box::new(|proof| {
                    *proof.narg.last_mut().expect("nonempty proof") ^= 0x01;
                }),
            ),
            (
                "field-increment commitment stripped from the proof",
                Box::new(move |proof| {
                    let _ = proof.narg.drain(field_inc_bytes.clone());
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
        use jolt_claims::protocols::field_inline::lattice::field_inc_group_role;
        use jolt_openings::{GroupOpeningClaim, TaggedGroupOpeningClaim};

        let (output, ()) = akita::prove(&field_ops(), JoltAkitaBackend::optimized(), |_, _, _| ());
        let sent = sent_commitments(&output);
        let commitment = sent.field_inc.commitment;
        let point = vec![AkitaField::from_u64(3); GroupCommitmentMetadata::num_vars(&commitment)];
        let field_claim = TaggedGroupOpeningClaim::new(
            field_inc_group_role(),
            GroupOpeningClaim::new(commitment, point, vec![AkitaField::from_u64(0)]),
        );
        let trace = sent.one_hot_trace.commitment;
        let main = GroupOpeningClaim::new(
            trace.clone(),
            vec![AkitaField::from_u64(3); GroupCommitmentMetadata::num_vars(&trace)],
            vec![AkitaField::from_u64(0)],
        );
        let mut transcript = VerifierTranscript::<JoltSponge>::new(
            &jolt_protocol_id::<JoltSponge>(),
            b"spurious-field_inline-group",
            &output.proof.narg,
        );
        let error = <AkitaScheme as CommitmentScheme>::verify_batch(
            &output.verifier_preprocessing.pcs_setup,
            &[field_claim.clone(), field_claim],
            &main,
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
