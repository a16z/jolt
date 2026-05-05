use super::shapes::StageVerifierErrorShape;

pub(in crate::protocols::jolt::emit::rust) fn stage_verifier_error_enum(
    stage: usize,
    shape: StageVerifierErrorShape,
) -> String {
    let missing_ram_variant = if shape.has_missing_ram() {
        "    MissingRam { relation: &'static str },\n"
    } else {
        ""
    };

    format!(
        r#"
#[derive(Debug)]
pub enum VerifyStage{stage}Error {{
    UnexpectedProofCount {{ expected: usize, got: usize }},
    MissingProof {{ driver: &'static str }},
    MissingBatch {{ driver: &'static str, batch: &'static str }},
    MissingClaim {{ batch: &'static str, claim: &'static str }},
    MissingValue {{ symbol: &'static str }},
    InvalidInputLength {{ input: &'static str, expected: usize, actual: usize }},
    InvalidProof {{ driver: &'static str, reason: &'static str }},
    UnsupportedFieldExpr {{ symbol: &'static str, formula: &'static str }},
    UnsupportedRelation {{ relation: &'static str }},
{missing_ram_variant}    Sumcheck {{ driver: &'static str, error: SumcheckError<Fr> }},
}}

super::common::impl_runtime_plan_error_conversion!(VerifyStage{stage}Error);
"#
    )
}
