//! Field-inline Dory backend parity and tamper rejection.
//!
//! Acceptance across protocol modes lives in `e2e_matrix.rs`. These tests use
//! the same guest cases and preparation, and cover distinct field-inline wire
//! properties: reference/optimized proof equality in clear mode, field commitment
//! presence, and rejection of corrupted field openings, commitments, and sumchecks.

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
    use jolt_field::{Fr, Ring};
    use jolt_poly::CompressedPoly;
    use jolt_prover::JoltBackend;
    use jolt_sumcheck::{ClearProof, SumcheckProof};
    use jolt_verifier::proof::JoltProofClaims;

    use crate::support::field_inline::dory::{self, Proof};
    use crate::support::field_inline::{field_ops, muldiv};

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

    /// Every field-inline-specific single-field tamper must reject: one proof, four
    /// mutations on fresh clones. The optimized backend proves here — its
    /// wire bytes equal the reference's (the parity tests pin both), so one
    /// backend's tamper matrix covers both.
    #[test]
    fn field_inline_tampered_proofs_are_rejected() {
        let (preprocessing, public_io, proof) = dory::prove(&field_ops(), JoltBackend::optimized());
        dory::verify_full(&preprocessing, &public_io, &proof)
            .expect("base proof must verify before tampering");
        let one = Fr::from_u64(1);

        type Tamper = (&'static str, Box<dyn Fn(&mut Proof)>);
        let tampers: Vec<Tamper> = vec![
            (
                "stage1 field-inline rs1_value opening",
                Box::new(move |proof| {
                    let JoltProofClaims::Clear(claims) = &mut proof.claims else {
                        panic!("clear proof expected");
                    };
                    let outer = &mut claims.stage1.outer.outer_remainder.field_inline;
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
                        .expect("field-inline proof carries the field-inline payload");
                    assert_ne!(
                        field_inline.field_registers.rd_inc, replacement,
                        "replacement commitment must differ",
                    );
                    field_inline.field_registers.rd_inc = replacement;
                }),
            ),
            (
                "stage2 field-inline product appendage rd_value",
                Box::new(move |proof| {
                    let JoltProofClaims::Clear(claims) = &mut proof.claims else {
                        panic!("clear proof expected");
                    };
                    let product = &mut claims.stage2.batch_outputs.product_remainder.field_inline;
                    product.rd_value += one;
                }),
            ),
            (
                // The composed stage-2 batch (field-inline claim reduction + product
                // appendage) rejects a corrupted round polynomial like the
                // base batch does; field-inline, no legacy-fixture suite covers the
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
                dory::verify_full(&preprocessing, &public_io, &tampered).is_err(),
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

    use crate::support;
    use crate::support::field_inline::{dory, field_ops, muldiv};

    /// The field-inline tampers on the ZK wire: the FieldRdInc commitment and
    /// the BlindFold payload. Mutate clones of one accepted base proof.
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

#[cfg(not(all(
    feature = "prover-fixtures",
    feature = "field-inline",
    not(feature = "akita")
)))]
#[test]
#[ignore = "enable --features prover-fixtures,field-inline (optionally +zk) to run the dory \
            field-inline e2e; the packed suite is akita_field_inline_e2e.rs"]
fn field_inline_e2e() {}
