//! Stage 1's field-inline seam: every FR-specific divergence of the stage-1
//! verifier in one place. `verify.rs` interacts with the FR protocol only
//! through the functions here (plus the FR carrier fields on the outputs,
//! which are proof shape).

use jolt_claims::protocols::field_inline::geometry::spartan::FIELD_INLINE_SPARTAN_OUTER_R1CS_INPUT_COUNT;
use jolt_claims::protocols::field_inline::relations::spartan::FieldRegistersSpartanOuterOutputClaims;
use jolt_claims::protocols::jolt::JoltRelationId;
use jolt_field::JoltField;
use jolt_transcript::Transcript;

use super::outputs::{Stage1BatchSumchecks, Stage1OutputClaims};
use crate::stages::relations::OutputClaims as _;
use crate::VerifierError;

/// Extract the FR Spartan-outer appendage from the stage-1 claims (fail-closed
/// on an FR-on proof without it) and supply it to the composed remainder
/// relation: the composed R1CS appends 13 FR-local columns whose openings ride
/// the same remainder sumcheck and feed the composed expected-output check.
pub fn attach_outer_outputs<F: JoltField>(
    sumchecks: &Stage1BatchSumchecks<F>,
    claims: &Stage1OutputClaims<F>,
) -> Result<FieldRegistersSpartanOuterOutputClaims<F>, VerifierError> {
    let field_inline_outer =
        claims
            .field_inline_outer
            .clone()
            .ok_or(VerifierError::MissingProofPayload {
                field: "claims.stage1.field_inline_outer",
            })?;
    sumchecks
        .outer_remainder
        .set_field_inline_outputs(field_inline_outer.opening_values())?;
    Ok(field_inline_outer)
}

/// Absorb the FR-local openings after the ordinary ones, in appended-column
/// order — the same append the prover must perform.
pub fn append_outer_openings<F: JoltField, T: Transcript<Challenge = F>>(
    transcript: &mut T,
    outputs: &FieldRegistersSpartanOuterOutputClaims<F>,
) {
    for value in outputs.opening_values() {
        transcript.append_labeled(b"opening_claim", &value);
    }
}

/// The composed stage-1 committed row count: the ordinary member openings plus
/// the 13 FR-local appendage rows — the same rows the clear path absorbs after
/// the member openings.
pub fn composed_output_claim_count(base: usize) -> Result<usize, VerifierError> {
    base.checked_add(FIELD_INLINE_SPARTAN_OUTER_R1CS_INPUT_COUNT)
        .ok_or_else(|| VerifierError::StageClaimSumcheckFailed {
            stage: format!("{:?}", JoltRelationId::SpartanOuter),
            reason: "composed stage-1 output-claim count overflows usize".to_string(),
        })
}
