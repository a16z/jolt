use super::Stage8CpuProgram;
use crate::emit::rust::EmitError;

impl Stage8CpuProgram {
    pub(super) fn emit_source(&self) -> Result<String, EmitError> {
        let mut source = Self::emit_types().to_owned();
        source.push_str(&self.emit_constants()?);
        Ok(source)
    }

    fn emit_types() -> &'static str {
        "#![allow(clippy::too_many_lines)]\n\n\
         #[derive(Clone, Copy, Debug, PartialEq, Eq)]\n\
         pub struct Stage8Params {\n    pub field: &'static str,\n    pub pcs: &'static str,\n    pub transcript: &'static str,\n}\n\n\
         #[derive(Clone, Copy, Debug, PartialEq, Eq)]\n\
         pub struct Stage8OpeningInputPlan {\n    pub symbol: &'static str,\n    pub source_stage: &'static str,\n    pub source_claim: &'static str,\n    pub oracle: &'static str,\n    pub domain: &'static str,\n    pub point_arity: usize,\n    pub claim_kind: &'static str,\n}\n\n\
         #[derive(Clone, Copy, Debug, PartialEq, Eq)]\n\
         pub struct Stage8OpeningClaimPlan {\n    pub symbol: &'static str,\n    pub oracle: &'static str,\n    pub family: &'static str,\n    pub domain: &'static str,\n    pub point_arity: usize,\n    pub point_source: &'static str,\n    pub eval_source: &'static str,\n    pub source_stage: &'static str,\n    pub source_claim: &'static str,\n}\n\n\
         #[derive(Clone, Copy, Debug, PartialEq, Eq)]\n\
         pub struct Stage8OpeningBatchPlan {\n    pub symbol: &'static str,\n    pub proof_slot: &'static str,\n    pub policy: &'static str,\n    pub count: usize,\n    pub ordered_claims: &'static [&'static str],\n}\n\n\
         #[derive(Clone, Copy, Debug, PartialEq, Eq)]\n\
         pub struct Stage8PcsProofPlan {\n    pub symbol: &'static str,\n    pub mode: &'static str,\n    pub pcs: &'static str,\n    pub proof_slot: &'static str,\n    pub transcript_label: &'static str,\n    pub batch: &'static str,\n}\n\n\
         #[derive(Clone, Copy, Debug, PartialEq, Eq)]\n\
         pub struct Stage8EvaluationProgramPlan {\n    pub role: &'static str,\n    pub function: &'static str,\n    pub params: Stage8Params,\n    pub evaluation_point_source: Stage8OpeningInputPlan,\n    pub opening_inputs: &'static [Stage8OpeningInputPlan],\n    pub opening_claims: &'static [Stage8OpeningClaimPlan],\n    pub opening_batch: Stage8OpeningBatchPlan,\n    pub pcs_proof: Stage8PcsProofPlan,\n}\n\n"
    }
}
