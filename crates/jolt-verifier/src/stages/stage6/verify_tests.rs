
#![expect(
    clippy::expect_used,
    reason = "test setup should fail loudly when helper contracts change"
)]

use super::*;
use crate::config::{IncrementCommitmentMode, JoltProtocolConfig, PcsFamily, ProgramMode};
use jolt_field::Fr;

fn lattice_config() -> JoltProtocolConfig {
    let mut config = JoltProtocolConfig::for_zk(false).with_pcs_family(PcsFamily::Lattice);
    config.lattice.program_mode = ProgramMode::Committed;
    config.lattice.increment_mode = IncrementCommitmentMode::FusedOneHot;
    config.lattice.packed_witness.layout_digest = Some([7; 32]);
    config.lattice.packed_witness.d_pack = Some(8);
    config.lattice.packed_witness.validity_digest = Some([11; 32]);
    #[cfg(feature = "field-inline")]
    {
        config.lattice.field_inline.enabled = true;
    }
    config
}

fn curve_config() -> JoltProtocolConfig {
    JoltProtocolConfig::for_zk(false)
}

fn trace_dimensions() -> TraceDimensions {
    TraceDimensions::new(2)
}

#[test]
fn lattice_stage6_requires_unsigned_increment_claims() {
    let error =
        unsigned_inc_claims_for_protocol::<Fr>(&lattice_config(), trace_dimensions(), false)
            .expect_err("lattice mode should require unsigned increment claims");
    assert!(matches!(
        error,
        VerifierError::MissingOpeningClaim { id } if id == lattice::unsigned_inc_opening()
    ));
}

#[test]
fn lattice_stage6_builds_unsigned_increment_claims() {
    let claims =
        unsigned_inc_claims_for_protocol::<Fr>(&lattice_config(), trace_dimensions(), true)
            .expect("lattice mode with unsigned claim should build stage claims");
    assert_eq!(
        claims.expect("unsigned increment claims should exist").id,
        JoltRelationId::UnsignedIncClaimReduction
    );
}

#[test]
fn lattice_stage6_rejects_wrong_unsigned_increment_chunk_count() {
    let claims =
        unsigned_inc_claims_for_protocol::<Fr>(&lattice_config(), trace_dimensions(), true)
            .expect("lattice mode with unsigned claim should build stage claims");
    let output_claims = super::super::inputs::UnsignedIncClaimReductionOutputOpeningClaims {
        unsigned_inc: Fr::zero(),
        unsigned_inc_msb: Fr::zero(),
    };
    let booleanity_claims = super::super::inputs::BooleanityOutputOpeningClaims {
        instruction_ra: Vec::new(),
        bytecode_ra: Vec::new(),
        ram_ra: Vec::new(),
        unsigned_inc_chunks: vec![Fr::zero(); 7],
    };

    let error = validate_lattice_increment_claim_shape(
        claims.as_ref(),
        Some(&output_claims),
        &booleanity_claims,
        8,
    )
    .expect_err("lattice booleanity must require all lower chunks");
    assert!(matches!(
        error,
        VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::Booleanity,
            ..
        }
    ));
}

#[test]
fn lattice_stage6_rejects_missing_unsigned_increment_output_claims() {
    let claims =
        unsigned_inc_claims_for_protocol::<Fr>(&lattice_config(), trace_dimensions(), true)
            .expect("lattice mode with unsigned claim should build stage claims");
    let booleanity_claims = super::super::inputs::BooleanityOutputOpeningClaims {
        instruction_ra: Vec::new(),
        bytecode_ra: Vec::new(),
        ram_ra: Vec::new(),
        unsigned_inc_chunks: vec![Fr::zero(); 8],
    };

    let error =
        validate_lattice_increment_claim_shape(claims.as_ref(), None, &booleanity_claims, 8)
            .expect_err("lattice unsigned increment output claims are required");
    assert!(matches!(
        error,
        VerifierError::MissingOpeningClaim { id } if id == lattice::unsigned_inc_opening()
    ));
}

#[test]
fn lattice_stage6_rejects_dense_increment_claims() {
    let dense_claims = super::super::inputs::IncClaimReductionOutputOpeningClaims {
        ram_inc: Fr::zero(),
        rd_inc: Fr::zero(),
    };

    let error = validate_dense_increment_claim_shape(true, Some(&dense_claims))
        .expect_err("lattice mode must not accept dense increment claims");
    assert!(matches!(
        error,
        VerifierError::UnexpectedOpeningClaim { id }
            if id == increments::claim_reduction_output_openings()[0]
    ));
}

#[test]
fn curve_stage6_requires_dense_increment_claims() {
    let error = validate_dense_increment_claim_shape::<Fr>(false, None)
        .expect_err("curve mode must require dense increment claims");
    assert!(matches!(
        error,
        VerifierError::MissingOpeningClaim { id }
            if id == increments::claim_reduction_output_openings()[0]
    ));
}

#[test]
fn curve_stage6_rejects_unsigned_increment_chunk_claims() {
    let booleanity_claims = super::super::inputs::BooleanityOutputOpeningClaims {
        instruction_ra: Vec::new(),
        bytecode_ra: Vec::new(),
        ram_ra: Vec::new(),
        unsigned_inc_chunks: vec![Fr::zero()],
    };

    let error = validate_lattice_increment_claim_shape(None, None, &booleanity_claims, 8)
        .expect_err("curve mode must not accept lattice unsigned increment chunk claims");
    assert!(matches!(
        error,
        VerifierError::UnexpectedOpeningClaim { id }
            if id == lattice::unsigned_inc_chunk_opening(0)
    ));
}

#[test]
fn curve_stage6_rejects_unsigned_increment_claims() {
    let error = unsigned_inc_claims_for_protocol::<Fr>(&curve_config(), trace_dimensions(), true)
        .expect_err("curve mode should reject unsigned increment claims");
    assert!(matches!(
        error,
        VerifierError::UnexpectedOpeningClaim { id } if id == lattice::unsigned_inc_opening()
    ));
}

#[test]
fn bytecode_val_stage_count_keeps_curve_and_lattice_shapes_distinct() {
    let five = vec![Fr::zero(); bytecode_reduction::NUM_BYTECODE_VAL_STAGES];
    let six = vec![Fr::zero(); bytecode_reduction::LATTICE_BYTECODE_VAL_STAGES];

    validate_bytecode_val_stage_claim_count(true, false, Some(&five))
        .expect("curve committed bytecode keeps five staged values");
    validate_bytecode_val_stage_claim_count(true, true, Some(&six))
        .expect("lattice committed bytecode requires Store binding stage");

    let error = validate_bytecode_val_stage_claim_count(true, true, Some(&five))
        .expect_err("lattice committed bytecode must reject five staged values");
    assert!(matches!(
        error,
        VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::BytecodeReadRaf,
            ..
        }
    ));
}
