use bolt::protocols::jolt::{CommitmentCpuProgram, OptionalSkipPolicy};
use jolt_prover::stages::commitment as generated_prover_commitment;
use jolt_verifier::stages::commitment as generated_commitment;

use super::{leak_slice, leak_str, leak_str_slice};

macro_rules! define_generated_commitment_adapter {
    ($function:ident, $module:ident, $program_plan:ident) => {
        pub fn $function(program: &CommitmentCpuProgram) -> &'static $module::$program_plan {
            Box::leak(Box::new($module::$program_plan {
                params: $module::CommitmentParams {
                    field: leak_str(&program.params.field),
                    pcs: leak_str(&program.params.pcs),
                    transcript: leak_str(&program.params.transcript),
                },
                oracle_plans: leak_slice(
                    program
                        .oracle_plans
                        .iter()
                        .map(|plan| $module::OraclePlan {
                            oracle: leak_str(&plan.oracle),
                            domain: leak_str(&plan.domain),
                            num_vars: plan.num_vars,
                        })
                        .collect(),
                ),
                batch_plans: leak_slice(
                    program
                        .batch_plans
                        .iter()
                        .map(|plan| $module::CommitmentBatchPlan {
                            artifact: leak_str(&plan.artifact),
                            pcs: leak_str(&plan.pcs),
                            oracle_family: leak_str(&plan.oracle_family),
                            label: leak_str(&plan.label),
                            oracles: leak_str_slice(&plan.oracles),
                            count: plan.count,
                            domain: leak_str(&plan.domain),
                            num_vars: plan.num_vars,
                        })
                        .collect(),
                ),
                optional_plans: leak_slice(
                    program
                        .optional_plans
                        .iter()
                        .map(|plan| $module::OptionalCommitmentPlan {
                            artifact: leak_str(&plan.artifact),
                            pcs: leak_str(&plan.pcs),
                            oracle: leak_str(&plan.oracle),
                            label: leak_str(&plan.label),
                            domain: leak_str(&plan.domain),
                            num_vars: plan.num_vars,
                            skip_policy: match plan.skip_policy {
                                OptionalSkipPolicy::MissingOrZero => {
                                    $module::OptionalSkipPolicy::MissingOrZero
                                }
                            },
                        })
                        .collect(),
                ),
                transcript_steps: leak_slice(
                    program
                        .transcript_steps
                        .iter()
                        .map(|step| $module::TranscriptStep {
                            label: leak_str(&step.label),
                            source: leak_str(&step.source),
                            optional: step.optional,
                        })
                        .collect(),
                ),
            }))
        }
    };
}

define_generated_commitment_adapter!(
    leak_generated_commitment_prover_program,
    generated_prover_commitment,
    CommitmentProverProgramPlan
);
define_generated_commitment_adapter!(
    leak_generated_commitment_verifier_program,
    generated_commitment,
    CommitmentVerifierProgramPlan
);
