use super::{CommitmentCpuProgram, OptionalSkipPolicy};
use crate::ir::Role;
use crate::protocols::jolt::emit::rust::source::{
    emit_params_const, emit_plan_array_compact, emit_str_array_compact, emit_struct_const, rust_str,
};

impl CommitmentCpuProgram {
    pub(super) fn emit_constants(&self) -> String {
        let mut source = emit_params_const(
            "COMMITMENT_PARAMS",
            "CommitmentParams",
            &self.params.field,
            &self.params.pcs,
            &self.params.transcript,
        );

        source.push_str(&emit_plan_array_compact(
            "ORACLE_PLANS",
            "OraclePlan",
            self.oracle_plans.iter().map(|plan| {
                format!(
                    "    OraclePlan {{ oracle: {}, domain: {}, num_vars: {} }},",
                    rust_str(&plan.oracle),
                    rust_str(&plan.domain),
                    plan.num_vars
                )
            }),
        ));

        for (index, plan) in self.batch_plans.iter().enumerate() {
            source.push_str(&emit_str_array_compact(
                &format!("COMMITMENT_BATCH_{index}_ORACLES"),
                &plan.oracles,
            ));
        }

        source.push_str(&emit_plan_array_compact(
            "COMMITMENT_BATCH_PLANS",
            "CommitmentBatchPlan",
            self.batch_plans.iter().enumerate().map(|(index, plan)| {
                format!(
                    "    CommitmentBatchPlan {{ artifact: {}, pcs: {}, oracle_family: {}, label: {}, oracles: COMMITMENT_BATCH_{index}_ORACLES, count: {}, domain: {}, num_vars: {} }},",
                    rust_str(&plan.artifact),
                    rust_str(&plan.pcs),
                    rust_str(&plan.oracle_family),
                    rust_str(&plan.label),
                    plan.count,
                    rust_str(&plan.domain),
                    plan.num_vars
                )
            }),
        ));

        source.push_str(&emit_plan_array_compact(
            "OPTIONAL_COMMITMENT_PLANS",
            "OptionalCommitmentPlan",
            self.optional_plans.iter().map(|plan| {
                format!(
                    "    OptionalCommitmentPlan {{ artifact: {}, pcs: {}, oracle: {}, label: {}, domain: {}, num_vars: {}, skip_policy: {} }},",
                    rust_str(&plan.artifact),
                    rust_str(&plan.pcs),
                    rust_str(&plan.oracle),
                    rust_str(&plan.label),
                    rust_str(&plan.domain),
                    plan.num_vars,
                    plan.skip_policy.rust_variant()
                )
            }),
        ));

        source.push_str(&emit_plan_array_compact(
            "TRANSCRIPT_PLAN",
            "TranscriptStep",
            self.transcript_steps.iter().map(|step| {
                format!(
                    "    TranscriptStep {{ label: {}, source: {}, optional: {} }},",
                    rust_str(&step.label),
                    rust_str(&step.source),
                    step.optional
                )
            }),
        ));
        let program_type = match self.role {
            Role::Prover => "CommitmentProverProgramPlan",
            Role::Verifier => "CommitmentVerifierProgramPlan",
        };
        source.push_str(&emit_struct_const(
            "COMMITMENT_PROGRAM",
            program_type,
            &[
                ("params", "COMMITMENT_PARAMS"),
                ("oracle_plans", "ORACLE_PLANS"),
                ("batch_plans", "COMMITMENT_BATCH_PLANS"),
                ("optional_plans", "OPTIONAL_COMMITMENT_PLANS"),
                ("transcript_steps", "TRANSCRIPT_PLAN"),
            ],
        ));

        source
    }
}

impl OptionalSkipPolicy {
    fn rust_variant(&self) -> &'static str {
        match self {
            Self::MissingOrZero => "OptionalSkipPolicy::MissingOrZero",
        }
    }
}
