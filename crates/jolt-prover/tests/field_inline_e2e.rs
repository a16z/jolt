//! Field-inline Dory backend parity and tamper rejection.
//!
//! Acceptance across protocol modes lives in `e2e_matrix.rs`. These tests use
//! the same guest cases and preparation, and cover distinct field-inline wire
//! properties: reference/optimized proof equality in clear mode, field commitment
//! presence and binding, and rejection of corrupted BlindFold payloads. Claim and
//! round-polynomial mutations live in the verifier fixture matrix.

#[cfg(all(
    feature = "prover-fixtures",
    feature = "field-inline",
    not(feature = "akita")
))]
mod support;

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
    use jolt_prover::{JoltBackend, ProverError};
    use jolt_verifier::proof::JoltProofClaims;
    use jolt_verifier::VerifierError;

    use crate::support::field_inline::dory;
    use crate::support::field_inline::{field_ops, muldiv};

    #[test]
    fn noncanonical_limb_advice_cannot_produce_an_accepted_proof() {
        use jolt_program::execution::{ExecutionBackend, TraceInputs};
        use tracer::execution_backend::TracerBackend;
        use tracer::instruction::field_inline::advice_limb::inject_advice_limbs;

        use crate::support::{self, GuestCase};

        let inputs = postcard::to_stdvec(&(Some([0u64; 6]), Some(6u8)))
            .expect("serialize conversion request");
        let case = GuestCase {
            inputs,
            field_inline_active: true,
            ..GuestCase::new("field-limbs-guest")
        };
        let mut prepared = support::prepare(&case);
        let trace_inputs = TraceInputs::new(
            case.inputs.clone(),
            vec![],
            vec![],
            support::memory_config(&prepared.preprocessing.memory_layout),
        );
        let injection = inject_advice_limbs(&Fr::MODULUS_LIMBS);
        prepared.trace = TracerBackend::new()
            .trace(&prepared.program, trace_inputs)
            .expect("dishonest advice produces a spoiled trace");
        assert_eq!(
            injection.consumed_addresses().len(),
            Fr::MODULUS_LIMBS.len()
        );
        assert!(prepared.trace.device.panic);
        drop(injection);
        match dory::try_prove(prepared, JoltBackend::optimized()) {
            Ok((preprocessing, public_io, proof)) => {
                assert!(dory::verify_full(&preprocessing, &public_io, &proof).is_err());
            }
            Err(error) => assert!(
                matches!(&error, ProverError::Verifier(
                    VerifierError::StageClaimSumcheckFailed { stage, .. }
                ) if stage == "Stage1Batch"),
                "unexpected rejection before checking the spoiled constraints: {error:?}",
            ),
        }
    }

    type BackendCase = (&'static str, fn() -> JoltBackend<Fr, DoryScheme>);

    fn backends() -> [BackendCase; 2] {
        [
            ("reference", JoltBackend::reference),
            ("optimized", JoltBackend::optimized),
        ]
    }

    /// Both backends' proofs must verify AND be equal wire objects — clear
    /// mode draws nothing outside Fiat-Shamir, so reference/optimized
    /// divergence anywhere in the composed pipeline shows up here as a proof
    /// inequality even when both sides individually verify.
    #[test]
    fn field_inline_eqpoly_reference_matches_optimized() {
        let mut proofs = Vec::new();
        for (label, backend) in backends() {
            let (preprocessing, public_io, proof) = dory::prove(&field_ops(), backend());
            assert!(
                proof.commitments.field_inline.is_some(),
                "field-inline proofs must carry the field-inline commitment payload ({label})",
            );
            assert!(matches!(proof.claims, JoltProofClaims::Clear(_)));
            dory::verify_full(&preprocessing, &public_io, &proof).unwrap_or_else(|error| {
                panic!("modular field-inline proof must verify ({label}): {error}")
            });
            proofs.push(proof);
        }
        assert!(
            proofs[0] == proofs[1],
            "reference and optimized field-inline proofs must be identical wire objects",
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
            assert!(proof.commitments.field_inline.is_some());
            dory::verify_full(&preprocessing, &public_io, &proof).unwrap_or_else(|error| {
                panic!("field-inactive modular proof must verify ({label}): {error}")
            });
            proofs.push(proof);
        }
        assert!(
            proofs[0] == proofs[1],
            "reference and optimized field-inactive proofs must be identical wire objects",
        );
    }

    /// The verifier fixture matrix covers claim and round-polynomial mutations;
    /// this checks that the prover's emitted commitment is transcript-bound.
    #[test]
    fn field_inline_tampered_commitment_is_rejected() {
        let (preprocessing, public_io, mut proof) =
            dory::prove(&field_ops(), JoltBackend::optimized());
        dory::verify_full(&preprocessing, &public_io, &proof).expect("honest proof");
        let replacement = proof.commitments.ram_inc.clone();
        let field_inline = proof
            .commitments
            .field_inline
            .as_mut()
            .expect("field-inline commitment");
        assert_ne!(field_inline.field_registers.rd_inc, replacement);
        field_inline.field_registers.rd_inc = replacement;
        assert!(dory::verify_full(&preprocessing, &public_io, &proof).is_err());
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

    use crate::support;
    use crate::support::field_inline::{dory, field_ops, muldiv};

    #[test]
    fn field_inline_tampered_proofs_are_rejected() {
        support::with_zk_stack(|| {
            let (preprocessing, public_io, proof) =
                dory::prove(&field_ops(), JoltBackend::optimized());
            assert!(matches!(proof.claims, JoltProofClaims::Zk { .. }));
            assert!(proof.commitments.field_inline.is_some());
            dory::verify_full(&preprocessing, &public_io, &proof)
                .expect("modular field-inline ZK proof must verify");

            let mut commitment_tampered = proof.clone();
            let replacement = commitment_tampered.commitments.ram_inc.clone();
            let field_inline = commitment_tampered
                .commitments
                .field_inline
                .as_mut()
                .expect("field-inline proof carries the field-inline payload");
            assert_ne!(field_inline.field_registers.rd_inc, replacement);
            field_inline.field_registers.rd_inc = replacement;
            assert!(
                dory::verify_full(&preprocessing, &public_io, &commitment_tampered).is_err(),
                "a tampered FieldRdInc commitment must be rejected in ZK mode",
            );

            let mut blindfold_tampered = proof;
            let JoltProofClaims::Zk { blindfold_proof } = &mut blindfold_tampered.claims else {
                panic!("ZK proof must carry the BlindFold claims variant");
            };
            blindfold_proof.random_u += Fr::from_u64(1);
            assert!(
                dory::verify_full(&preprocessing, &public_io, &blindfold_tampered).is_err(),
                "a tampered BlindFold proof must be rejected",
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
