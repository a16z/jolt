use super::{Stage8CpuProgram, Stage8OpeningInputPlan, EVALUATION_POINT_SOURCE_SYMBOL};
use crate::emit::rust::EmitError;
use crate::protocols::jolt::emit::rust::source::{
    emit_inline_struct_const, emit_plan_array, emit_struct_const, emit_value_const, rust_str,
    rust_str_array,
};

impl Stage8CpuProgram {
    pub(super) fn emit_constants(&self) -> Result<String, EmitError> {
        let params_field = rust_str(&self.params.field);
        let params_pcs = rust_str(&self.params.pcs);
        let params_transcript = rust_str(&self.params.transcript);
        let mut source = emit_inline_struct_const(
            "STAGE8_PARAMS",
            "Stage8Params",
            &[
                ("field", &params_field),
                ("pcs", &params_pcs),
                ("transcript", &params_transcript),
            ],
            "\n\n",
        );
        let point_source = self
            .opening_inputs
            .iter()
            .find(|input| input.symbol == EVALUATION_POINT_SOURCE_SYMBOL)
            .ok_or_else(|| {
                EmitError::new(format!(
                    "evaluation program missing `{EVALUATION_POINT_SOURCE_SYMBOL}` opening-point source"
                ))
            })?;
        source.push_str(&emit_value_const(
            "STAGE8_EVALUATION_POINT_SOURCE",
            "Stage8OpeningInputPlan",
            &opening_input_literal(point_source),
            "\n\n",
        ));
        source.push_str(&emit_plan_array(
            "STAGE8_OPENING_INPUTS",
            "Stage8OpeningInputPlan",
            self.opening_inputs
                .iter()
                .map(|input| format!("    {},", opening_input_literal(input))),
        ));
        source.push_str(&emit_plan_array(
            "STAGE8_OPENING_CLAIMS",
            "Stage8OpeningClaimPlan",
            self.opening_claims.iter().map(|claim| {
                format!(
                    "    Stage8OpeningClaimPlan {{ symbol: {}, oracle: {}, family: {}, domain: {}, point_arity: {}, point_source: {}, eval_source: {}, source_stage: {}, source_claim: {} }},",
                    rust_str(&claim.symbol),
                    rust_str(&claim.oracle),
                    rust_str(&claim.family),
                    rust_str(&claim.domain),
                    claim.point_arity,
                    rust_str(&claim.point_source),
                    rust_str(&claim.eval_source),
                    rust_str(&claim.source_stage),
                    rust_str(&claim.source_claim),
                )
            }),
        ));
        let batch = &self.opening_batches[0];
        let batch_ordered_claims = format!("&{}", rust_str_array(&batch.ordered_claims));
        source.push_str(&emit_value_const(
            "STAGE8_OPENING_BATCH_ORDERED_CLAIMS",
            "&[&str]",
            &batch_ordered_claims,
            "\n\n",
        ));
        let batch_symbol = rust_str(&batch.symbol);
        let batch_proof_slot = rust_str(&batch.proof_slot);
        let batch_policy = rust_str(&batch.policy);
        let batch_count = batch.count.to_string();
        source.push_str(&emit_inline_struct_const(
            "STAGE8_OPENING_BATCH",
            "Stage8OpeningBatchPlan",
            &[
                ("symbol", &batch_symbol),
                ("proof_slot", &batch_proof_slot),
                ("policy", &batch_policy),
                ("count", &batch_count),
                ("ordered_claims", "STAGE8_OPENING_BATCH_ORDERED_CLAIMS"),
            ],
            "\n\n",
        ));
        let proof = &self.pcs_proofs[0];
        let proof_symbol = rust_str(&proof.symbol);
        let proof_mode = rust_str(&proof.mode);
        let proof_pcs = rust_str(&proof.pcs);
        let proof_slot = rust_str(&proof.proof_slot);
        let proof_transcript_label = rust_str(&proof.transcript_label);
        let proof_batch = rust_str(&proof.batch);
        source.push_str(&emit_inline_struct_const(
            "STAGE8_PCS_PROOF",
            "Stage8PcsProofPlan",
            &[
                ("symbol", &proof_symbol),
                ("mode", &proof_mode),
                ("pcs", &proof_pcs),
                ("proof_slot", &proof_slot),
                ("transcript_label", &proof_transcript_label),
                ("batch", &proof_batch),
            ],
            "\n\n",
        ));
        let role = rust_str(self.role.as_str());
        let function = rust_str(&self.function);
        source.push_str(&emit_struct_const(
            "STAGE8_PROGRAM",
            "Stage8EvaluationProgramPlan",
            &[
                ("role", &role),
                ("function", &function),
                ("params", "STAGE8_PARAMS"),
                ("evaluation_point_source", "STAGE8_EVALUATION_POINT_SOURCE"),
                ("opening_inputs", "STAGE8_OPENING_INPUTS"),
                ("opening_claims", "STAGE8_OPENING_CLAIMS"),
                ("opening_batch", "STAGE8_OPENING_BATCH"),
                ("pcs_proof", "STAGE8_PCS_PROOF"),
            ],
        ));
        Ok(source)
    }
}

fn opening_input_literal(input: &Stage8OpeningInputPlan) -> String {
    format!(
        "Stage8OpeningInputPlan {{ symbol: {}, source_stage: {}, source_claim: {}, oracle: {}, domain: {}, point_arity: {}, claim_kind: {} }}",
        rust_str(&input.symbol),
        rust_str(&input.source_stage),
        rust_str(&input.source_claim),
        rust_str(&input.oracle),
        rust_str(&input.domain),
        input.point_arity,
        rust_str(&input.claim_kind),
    )
}
