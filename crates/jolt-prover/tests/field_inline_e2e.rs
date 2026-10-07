//! Field-inline Dory backend parity and tamper rejection.
//!
//! Acceptance across protocol modes lives in `e2e_matrix.rs`. These tests use
//! the same guest cases and preparation, and cover distinct field-inline wire
//! properties: reference/optimized proof equality in clear mode and the binding
//! of the `FieldRdInc` commitment in both modes. Claim and round-polynomial
//! mutations live in the verifier fixture matrix; BlindFold tampering lives in
//! `zk_e2e.rs`.

#[cfg(all(
    feature = "prover-fixtures",
    feature = "field-inline",
    not(feature = "akita")
))]
mod support;

#[cfg(all(
    feature = "prover-fixtures",
    feature = "field-inline",
    not(feature = "akita")
))]
#[expect(clippy::expect_used, reason = "integration tests should fail loudly")]
mod tamper {
    use common::jolt_device::JoltDevice;
    use jolt_crypto::{Bn254G1, Pedersen};
    use jolt_dory::DoryScheme;
    use jolt_transcript::VerifierTranscript;
    use jolt_verifier::proof::ProofHeader;
    use jolt_verifier::{jolt_protocol_id, seed_transcript, JoltSponge, JOLT_SESSION};

    use crate::support::field_inline::dory::{Proof, VerifierPreprocessing};

    /// `proof` with its `FieldRdInc` commitment bytes overwritten by its `RamInc`
    /// commitment bytes: a well-formed commitment to a different polynomial. The
    /// byte ranges come from the verifier's own header and commitment reads, in
    /// `ProofCommitments::send` order (`RdInc`, `RamInc`, the RA commitments,
    /// then `FieldRdInc`; the fixtures carry no untrusted advice).
    pub(crate) fn with_field_inline_commitment_replaced(
        preprocessing: &VerifierPreprocessing,
        public_io: &JoltDevice,
        proof: &Proof,
    ) -> Proof {
        let narg = proof.narg.as_slice();
        let consumed =
            |transcript: &VerifierTranscript<'_, JoltSponge>| narg.len() - transcript.remaining();
        let protocol = jolt_protocol_id::<JoltSponge>();

        let mut transcript = VerifierTranscript::<JoltSponge>::new(&protocol, JOLT_SESSION, narg);
        let _header = ProofHeader::receive(&mut transcript).expect("proof header");
        let header_end = consumed(&transcript);

        let mut transcript = VerifierTranscript::<JoltSponge>::new(&protocol, JOLT_SESSION, narg);
        let seeded = seed_transcript::<DoryScheme, Pedersen<Bn254G1>, JoltSponge>(
            preprocessing,
            public_io,
            None,
            &mut transcript,
        )
        .expect("proof commitments");
        let commitments_end = consumed(&transcript);
        assert!(seeded.commitments.untrusted_advice.is_none());
        let trace = &seeded.commitments.trace;
        let count = 3 + trace.instruction_ra.len() + trace.ram_ra.len() + trace.bytecode_ra.len();
        let span = commitments_end - header_end;
        assert_eq!(span % count, 0, "Dory commitments have one width");
        let width = span / count;
        assert_ne!(trace.ram_inc, trace.field_inline.field_registers.rd_inc);

        let mut tampered = proof.clone();
        let ram_inc = header_end + width..header_end + 2 * width;
        tampered.narg.copy_within(ram_inc, commitments_end - width);
        tampered
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
    use jolt_dory::DoryScheme;
    use jolt_field::Fr;
    use jolt_prover::JoltBackend;

    use crate::support::field_inline::dory;
    use crate::support::field_inline::{field_ops, muldiv};

    type BackendCase = (&'static str, fn() -> JoltBackend<Fr, DoryScheme>);

    fn backends() -> [BackendCase; 2] {
        [
            ("reference", JoltBackend::reference),
            ("optimized", JoltBackend::optimized),
        ]
    }

    /// Both backends' proofs must verify AND be equal argument strings — clear
    /// mode draws nothing outside Fiat-Shamir, so reference/optimized
    /// divergence anywhere in the composed pipeline shows up here as a proof
    /// inequality even when both sides individually verify.
    #[test]
    fn field_inline_eqpoly_reference_matches_optimized() {
        let mut proofs = Vec::new();
        for (label, backend) in backends() {
            let (preprocessing, public_io, proof) = dory::prove(&field_ops(), backend());
            dory::verify_full(&preprocessing, &public_io, &proof).unwrap_or_else(|error| {
                panic!("modular field-inline proof must verify ({label}): {error}")
            });
            proofs.push(proof);
        }
        assert!(
            proofs[0] == proofs[1],
            "reference and optimized field-inline proofs must be identical argument strings",
        );
    }

    /// The uniform-shape degenerate case: a field-inline guest executing zero
    /// field-inline instructions still proves under the composed protocol, with an
    /// all-zero `FieldRdInc` commitment and zero field-inline openings.
    #[test]
    fn field_inline_muldiv_reference_matches_optimized() {
        let mut proofs = Vec::new();
        for (label, backend) in backends() {
            let (preprocessing, public_io, proof) = dory::prove(&muldiv(), backend());
            dory::verify_full(&preprocessing, &public_io, &proof).unwrap_or_else(|error| {
                panic!("field-inactive modular proof must verify ({label}): {error}")
            });
            proofs.push(proof);
        }
        assert!(
            proofs[0] == proofs[1],
            "reference and optimized field-inactive proofs must be identical argument strings",
        );
    }

    /// The verifier fixture matrix covers claim and round-polynomial mutations;
    /// this checks that the prover's emitted commitment is transcript-bound.
    #[test]
    fn field_inline_tampered_commitment_is_rejected() {
        let (preprocessing, public_io, proof) = dory::prove(&field_ops(), JoltBackend::optimized());
        dory::verify_full(&preprocessing, &public_io, &proof).expect("honest proof");
        let tampered = crate::tamper::with_field_inline_commitment_replaced(
            &preprocessing,
            &public_io,
            &proof,
        );
        assert!(dory::verify_full(&preprocessing, &public_io, &tampered).is_err());
    }
}

#[cfg(all(
    feature = "prover-fixtures",
    feature = "field-inline",
    feature = "zk",
    not(feature = "akita")
))]
#[expect(clippy::expect_used, reason = "integration tests should fail loudly")]
mod zk {
    use jolt_prover::JoltBackend;

    use crate::support;
    use crate::support::field_inline::{dory, field_ops, muldiv};

    /// The FieldRdInc commitment is bound on the ZK wire too.
    #[test]
    fn field_inline_tampered_commitment_is_rejected() {
        support::with_zk_stack(|| {
            let (preprocessing, public_io, proof) =
                dory::prove(&field_ops(), JoltBackend::optimized());
            dory::verify_full(&preprocessing, &public_io, &proof)
                .expect("modular field-inline ZK proof must verify");
            let tampered = crate::tamper::with_field_inline_commitment_replaced(
                &preprocessing,
                &public_io,
                &proof,
            );
            assert!(
                dory::verify_full(&preprocessing, &public_io, &tampered).is_err(),
                "a tampered FieldRdInc commitment must be rejected in ZK mode",
            );
        });
    }

    /// The acceptance matrix uses optimized kernels; retain the reference ZK
    /// path separately because randomized ZK proofs cannot be compared by bytes.
    #[test]
    fn field_inline_muldiv_reference_proof_is_accepted() {
        support::with_zk_stack(|| {
            let (preprocessing, public_io, proof) =
                dory::prove(&muldiv(), JoltBackend::reference());
            dory::verify_full(&preprocessing, &public_io, &proof)
                .expect("field-inactive reference ZK proof must verify");
        });
    }
}
