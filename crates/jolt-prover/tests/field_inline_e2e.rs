//! Field-inline Dory backend parity and tamper rejection.
//!
//! Acceptance across protocol modes lives in `e2e_matrix.rs`. These tests use
//! the same guest cases and preparation, and cover distinct field-inline wire
//! properties: backend proof equality against reference in clear mode, field commitment
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
    not(feature = "akita")
))]
#[path = "support/dory_backend.rs"]
mod dory_backend;

#[cfg(all(
    feature = "prover-fixtures",
    feature = "field-inline",
    not(feature = "zk"),
    not(feature = "akita")
))]
#[expect(clippy::expect_used, reason = "integration tests should fail loudly")]
mod clear {
    use jolt_dory::DoryScheme;
    use jolt_field::Fr;
    #[cfg(feature = "cuda")]
    use jolt_kernels::cuda::CudaDoryScheme;
    use jolt_prover::{JoltBackend, ProverError};
    use jolt_verifier::proof::JoltProofClaims;
    use jolt_verifier::VerifierError;

    use crate::dory_backend::DoryBackend;
    use crate::support::field_inline::dory;
    use crate::support::field_inline::{field_ops, muldiv};
    use crate::support::GuestCase;

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
        match dory::try_prove(prepared, JoltBackend::<Fr, DoryScheme>::optimized()) {
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

    /// Clear mode draws nothing outside Fiat-Shamir: a backend must produce
    /// the reference proof, not just a proof the verifier accepts.
    fn assert_backend_equivalence(case: &GuestCase, backend: impl DoryBackend) {
        let (reference_preprocessing, reference_io, reference) =
            dory::prove(case, JoltBackend::<Fr, DoryScheme>::reference());
        dory::verify_full(&reference_preprocessing, &reference_io, &reference)
            .expect("reference proof must verify");
        let (preprocessing, public_io, proof) = dory::prove(case, backend);
        assert!(proof.commitments.field_inline.is_some());
        assert!(matches!(proof.claims, JoltProofClaims::Clear(_)));
        dory::verify_full(&preprocessing, &public_io, &proof).expect("backend proof must verify");
        assert!(
            reference == proof,
            "backend and reference proofs must be identical wire objects",
        );
    }

    #[test]
    fn field_inline_eqpoly_reference_matches_optimized() {
        assert_backend_equivalence(&field_ops(), JoltBackend::<Fr, DoryScheme>::optimized());
    }

    /// The uniform-shape degenerate case: a field-inline guest executing zero
    /// field-inline instructions still proves under the composed protocol, with an
    /// all-zero `FieldRdInc` commitment and zero field-inline openings.
    #[test]
    fn field_inline_muldiv_reference_matches_optimized() {
        assert_backend_equivalence(&muldiv(), JoltBackend::<Fr, DoryScheme>::optimized());
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn field_inline_eqpoly_reference_matches_cuda() {
        assert_backend_equivalence(&field_ops(), JoltBackend::<Fr, CudaDoryScheme>::cuda());
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn field_inline_muldiv_reference_matches_cuda() {
        assert_backend_equivalence(&muldiv(), JoltBackend::<Fr, CudaDoryScheme>::cuda());
    }

    /// The verifier fixture matrix covers claim and round-polynomial mutations;
    /// this checks that the prover's emitted commitment is transcript-bound.
    #[test]
    fn field_inline_tampered_commitment_is_rejected() {
        let (preprocessing, public_io, mut proof) =
            dory::prove(&field_ops(), JoltBackend::<Fr, DoryScheme>::optimized());
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
    use jolt_dory::DoryScheme;
    use jolt_field::{Fr, Ring};
    use jolt_prover::JoltBackend;
    use jolt_verifier::proof::JoltProofClaims;

    use crate::support;
    use crate::support::field_inline::{dory, field_ops, muldiv};

    #[test]
    fn field_inline_tampered_proofs_are_rejected() {
        support::with_zk_stack(|| {
            let (preprocessing, public_io, proof) =
                dory::prove(&field_ops(), JoltBackend::<Fr, DoryScheme>::optimized());
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
                dory::prove(&muldiv(), JoltBackend::<Fr, DoryScheme>::reference());
            dory::verify_full(&preprocessing, &public_io, &proof)
                .expect("field-inactive reference ZK proof must verify");
        });
    }
}
