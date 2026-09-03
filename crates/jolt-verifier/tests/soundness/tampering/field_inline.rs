//! Fixture-driven tamper suite for the field-inline path: every FR wire cell
//! of the clear claims is offset by one over a real modular-prover fixture
//! (the eq-MLE FR guest — the FR-capable prover is the modular one), and the
//! verifier must reject each mutation. These are the active coverage behind
//! the field-inline `TamperCoverage::Active` manifest entries.

#![cfg_attr(
    all(
        feature = "prover-fixtures",
        feature = "field-inline",
        not(feature = "zk")
    ),
    expect(
        clippy::expect_used,
        clippy::panic,
        reason = "fixture tamper tests should fail loudly when the stored proof shape changes"
    )
)]

#[cfg(all(
    feature = "prover-fixtures",
    feature = "field-inline",
    not(feature = "zk")
))]
mod active {
    use jolt_claims::protocols::field_inline::relations::product::FieldRegistersProductOutputClaims;
    use jolt_claims::protocols::field_inline::relations::spartan::FieldRegistersSpartanOuterOutputClaims;
    use jolt_field::{Fr, Ring};
    use jolt_verifier::proof::{ClearProofClaims, JoltProofClaims};

    use crate::support::tamper_manifest::{
        assert_verifier_fixture_tamper_rejects, required_target,
    };
    use crate::support::verifier_fixtures::{
        standard_field_inline_eqpoly_case, VerifierFixtureCase,
    };

    type CellSelector = fn(&mut ClearProofClaims<Fr>) -> &mut Fr;

    fn clear_claims(case: &mut VerifierFixtureCase) -> &mut ClearProofClaims<Fr> {
        let JoltProofClaims::Clear(claims) = &mut case.proof.claims else {
            panic!("field-inline fixture must carry clear claims");
        };
        claims
    }

    /// Offset each listed FR claim cell by one on a fresh clone of `base`;
    /// each mutation must reject, under the named (Active) manifest target.
    fn offset_each_cell(base: &VerifierFixtureCase, target: &str, cells: &[CellSelector]) {
        for select in cells {
            assert_verifier_fixture_tamper_rejects(required_target(target), base, |case| {
                *select(clear_claims(case)) += Fr::from_u64(1);
            });
        }
    }

    #[test]
    fn tampered_field_inline_stage1_outer_claims_reject() {
        fn outer(
            claims: &mut ClearProofClaims<Fr>,
        ) -> &mut FieldRegistersSpartanOuterOutputClaims<Fr> {
            claims
                .stage1
                .field_inline_outer
                .as_mut()
                .expect("FR-on fixture carries stage-1 FR openings")
        }
        offset_each_cell(
            &standard_field_inline_eqpoly_case(),
            "stage1.claims.field_inline_outer",
            &[
                |claims| &mut outer(claims).rs1_value,
                |claims| &mut outer(claims).rs2_value,
                |claims| &mut outer(claims).rd_value,
                |claims| &mut outer(claims).product,
                |claims| &mut outer(claims).inv_product,
                |claims| &mut outer(claims).add,
                |claims| &mut outer(claims).sub,
                |claims| &mut outer(claims).mul,
                |claims| &mut outer(claims).inv,
                |claims| &mut outer(claims).assert_eq,
                |claims| &mut outer(claims).load_from_x,
                |claims| &mut outer(claims).store_to_x,
                |claims| &mut outer(claims).load_imm,
            ],
        );
    }

    #[test]
    fn tampered_field_inline_stage2_claim_reduction_claims_reject() {
        offset_each_cell(
            &standard_field_inline_eqpoly_case(),
            "stage2.claims.batch_outputs.field_registers_claim_reduction",
            &[
                |claims| {
                    &mut claims
                        .stage2
                        .batch_outputs
                        .field_registers_claim_reduction
                        .rd_value
                },
                |claims| {
                    &mut claims
                        .stage2
                        .batch_outputs
                        .field_registers_claim_reduction
                        .rs1_value
                },
                |claims| {
                    &mut claims
                        .stage2
                        .batch_outputs
                        .field_registers_claim_reduction
                        .rs2_value
                },
            ],
        );
    }

    #[test]
    fn tampered_field_inline_stage2_product_appendage_claims_reject() {
        fn product(
            claims: &mut ClearProofClaims<Fr>,
        ) -> &mut FieldRegistersProductOutputClaims<Fr> {
            claims
                .stage2
                .field_inline_product
                .as_mut()
                .expect("FR-on fixture carries the stage-2 FR product appendage")
        }
        offset_each_cell(
            &standard_field_inline_eqpoly_case(),
            "stage2.claims.field_inline_product",
            &[
                |claims| &mut product(claims).rs1_value,
                |claims| &mut product(claims).rs2_value,
                |claims| &mut product(claims).rd_value,
            ],
        );
    }

    #[test]
    fn tampered_field_inline_stage4_read_write_claims_reject() {
        offset_each_cell(
            &standard_field_inline_eqpoly_case(),
            "stage4.claims.field_registers_read_write",
            &[
                |claims| &mut claims.stage4.field_registers_read_write.registers_val,
                |claims| &mut claims.stage4.field_registers_read_write.rs1_ra,
                |claims| &mut claims.stage4.field_registers_read_write.rs2_ra,
                |claims| &mut claims.stage4.field_registers_read_write.rd_wa,
                |claims| &mut claims.stage4.field_registers_read_write.rd_inc,
            ],
        );
    }

    #[test]
    fn tampered_field_inline_stage5_val_evaluation_claims_reject() {
        offset_each_cell(
            &standard_field_inline_eqpoly_case(),
            "stage5.claims.field_registers_val_evaluation",
            &[
                |claims| &mut claims.stage5.field_registers_val_evaluation.rd_inc,
                |claims| &mut claims.stage5.field_registers_val_evaluation.rd_wa,
            ],
        );
    }

    #[test]
    fn tampered_field_inline_stage6_inc_claim_reduction_claim_rejects() {
        offset_each_cell(
            &standard_field_inline_eqpoly_case(),
            "stage6.claims.field_registers_inc_claim_reduction.rd_inc",
            &[|claims| &mut claims.stage6b.field_registers_inc_claim_reduction.rd_inc],
        );
    }
}
