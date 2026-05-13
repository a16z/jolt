//! Static plan adapters from Bolt compiler plans to generated/kernel plans.
//!
//! These are compatibility shims for the equivalence oracle. They translate
//! Bolt's owned compiler plans into the currently generated static plan shape
//! expected by jolt-kernels, jolt-prover, and jolt-verifier.

macro_rules! stage_list {
    (kernel, $values:expr) => {
        super::leak_str_slice($values)
    };
    (generated, $values:expr) => {
        super::leak_symbol_list($values)
    };
}

macro_rules! stage_field_expr {
    (kernel, $module:ident, $field_expr:ident, $plan:ident) => {
        $module::$field_expr {
            symbol: super::leak_str(&$plan.symbol),
            kind: super::leak_str(&$plan.kind),
            formula: super::leak_str(&$plan.formula),
            operand_names: super::leak_str_slice(&$plan.operand_names),
            operands: super::leak_str_slice(&$plan.operands),
        }
    };
    (generated, $module:ident, $field_expr:ident, $plan:ident) => {
        $module::$field_expr {
            symbol: super::leak_str(&$plan.symbol),
            kind: super::leak_str(&$plan.kind),
            formula: super::leak_str(&$plan.formula),
            operands: super::leak_symbol_list(&$plan.operands),
        }
    };
}

macro_rules! stage_claim {
    ($mode:ident, $module:ident, $claim:ident, $plan:ident) => {
        $module::$claim {
            symbol: super::leak_str(&$plan.symbol),
            stage: super::leak_str(&$plan.stage),
            domain: super::leak_str(&$plan.domain),
            num_rounds: $plan.num_rounds,
            degree: $plan.degree,
            claim: super::leak_str(&$plan.claim),
            kernel: $plan.kernel.as_deref().map(super::leak_str),
            relation: $plan.relation.as_deref().map(super::leak_str),
            claim_value: super::leak_str(&$plan.claim_value),
            input_openings: stage_list!($mode, &$plan.input_openings),
        }
    };
}

macro_rules! stage_driver {
    ($module:ident, $driver:ident, $plan:ident) => {
        $module::$driver {
            symbol: super::leak_str(&$plan.symbol),
            stage: super::leak_str(&$plan.stage),
            proof_slot: super::leak_str(&$plan.proof_slot),
            kernel: $plan.kernel.as_deref().map(super::leak_str),
            relation: $plan.relation.as_deref().map(super::leak_str),
            batch: super::leak_str(&$plan.batch),
            policy: super::leak_str(&$plan.policy),
            round_schedule: super::leak_usize_slice(&$plan.round_schedule),
            claim_label: super::leak_str(&$plan.claim_label),
            round_label: super::leak_str(&$plan.round_label),
            num_rounds: $plan.num_rounds,
            degree: $plan.degree,
        }
    };
}

macro_rules! define_stage_adapter_impl {
    (
        $mode:ident,
        $function:ident,
        $compiler:ty,
        $module:ident,
        $program:ident,
        $params:ident,
        $step:ident,
        $squeeze:ident,
        $opening_input:ident,
        $field_constant:ident,
        $field_expr:ident,
        $claim:ident,
        $batch:ident,
        $driver:ident,
        $instance_result:ident,
        $eval:ident,
        $point_slice:ident,
        $point_concat:ident,
        $opening_claim:ident,
        $opening_batch:ident
        $(, role = $role_field:ident)?
        $(, transcript_absorb_bytes = $absorb:ident)?
        $(, kernels = $kernel:ident)?
        $(, point_zeros = $point_zero:ident)?
        $(, opening_equalities = $opening_equality:ident)?
    ) => {
        pub fn $function(program: &$compiler) -> &'static $module::$program {
            Box::leak(Box::new($module::$program {
                $(
                $role_field: super::role_name(&program.role),
                )?
                params: $module::$params {
                    field: super::leak_str(&program.params.field),
                    pcs: super::leak_str(&program.params.pcs),
                    transcript: super::leak_str(&program.params.transcript),
                },
                steps: super::leak_slice(
                    program
                        .steps
                        .iter()
                        .map(|plan| $module::$step {
                            kind: super::leak_str(&plan.kind),
                            symbol: super::leak_str(&plan.symbol),
                        })
                        .collect(),
                ),
                transcript_squeezes: super::leak_slice(
                    program
                        .transcript_squeezes
                        .iter()
                        .map(|plan| $module::$squeeze {
                            symbol: super::leak_str(&plan.symbol),
                            label: super::leak_str(&plan.label),
                            kind: super::leak_str(&plan.kind),
                            count: plan.count,
                        })
                        .collect(),
                ),
                $(
                transcript_absorb_bytes: super::leak_slice(
                    program
                        .transcript_absorb_bytes
                        .iter()
                        .map(|plan| $module::$absorb {
                            symbol: super::leak_str(&plan.symbol),
                            label: super::leak_str(&plan.label),
                            payload: super::leak_str(&plan.payload),
                        })
                        .collect(),
                ),
                )?
                opening_inputs: super::leak_slice(
                    program
                        .opening_inputs
                        .iter()
                        .map(|plan| $module::$opening_input {
                            symbol: super::leak_str(&plan.symbol),
                            source_stage: super::leak_str(&plan.source_stage),
                            source_claim: super::leak_str(&plan.source_claim),
                            oracle: super::leak_str(&plan.oracle),
                            domain: super::leak_str(&plan.domain),
                            point_arity: plan.point_arity,
                            claim_kind: super::leak_str(&plan.claim_kind),
                        })
                        .collect(),
                ),
                field_constants: super::leak_slice(
                    program
                        .field_constants
                        .iter()
                        .map(|plan| $module::$field_constant {
                            symbol: super::leak_str(&plan.symbol),
                            field: super::leak_str(&plan.field),
                            value: plan.value,
                        })
                        .collect(),
                ),
                field_exprs: super::leak_slice(
                    program
                        .field_exprs
                        .iter()
                        .map(|plan| stage_field_expr!($mode, $module, $field_expr, plan))
                        .collect(),
                ),
                $(
                kernels: super::leak_slice(
                    program
                        .kernels
                        .iter()
                        .map(|plan| $module::$kernel {
                            symbol: super::leak_str(&plan.symbol),
                            relation: super::leak_str(&plan.relation),
                            kind: super::leak_str(&plan.kind),
                            backend: super::leak_str(&plan.backend),
                            abi: super::leak_str(&plan.abi),
                        })
                        .collect(),
                ),
                )?
                claims: super::leak_slice(
                    program
                        .claims
                        .iter()
                        .map(|plan| stage_claim!($mode, $module, $claim, plan))
                        .collect(),
                ),
                batches: super::leak_slice(
                    program
                        .batches
                        .iter()
                        .map(|plan| $module::$batch {
                            symbol: super::leak_str(&plan.symbol),
                            stage: super::leak_str(&plan.stage),
                            proof_slot: super::leak_str(&plan.proof_slot),
                            policy: super::leak_str(&plan.policy),
                            count: plan.count,
                            ordered_claims: stage_list!($mode, &plan.ordered_claims),
                            claim_operands: stage_list!($mode, &plan.claim_operands),
                            claim_label: super::leak_str(&plan.claim_label),
                            round_label: super::leak_str(&plan.round_label),
                            round_schedule: super::leak_usize_slice(&plan.round_schedule),
                        })
                        .collect(),
                ),
                drivers: super::leak_slice(
                    program
                        .drivers
                        .iter()
                        .map(|plan| stage_driver!($module, $driver, plan))
                        .collect(),
                ),
                instance_results: super::leak_slice(
                    program
                        .instance_results
                        .iter()
                        .map(|plan| $module::$instance_result {
                            symbol: super::leak_str(&plan.symbol),
                            source: super::leak_str(&plan.source),
                            claim: super::leak_str(&plan.claim),
                            relation: super::leak_str(&plan.relation),
                            index: plan.index,
                            point_arity: plan.point_arity,
                            num_rounds: plan.num_rounds,
                            round_offset: plan.round_offset,
                            point_order: super::leak_str(&plan.point_order),
                            degree: plan.degree,
                        })
                        .collect(),
                ),
                evals: super::leak_slice(
                    program
                        .evals
                        .iter()
                        .map(|plan| $module::$eval {
                            symbol: super::leak_str(&plan.symbol),
                            source: super::leak_str(&plan.source),
                            name: super::leak_str(&plan.name),
                            index: plan.index,
                            oracle: super::leak_str(&plan.oracle),
                        })
                        .collect(),
                ),
                $(
                point_zeros: super::leak_slice(
                    program
                        .point_zeros
                        .iter()
                        .map(|plan| $module::$point_zero {
                            symbol: super::leak_str(&plan.symbol),
                            field: super::leak_str(&plan.field),
                            arity: plan.arity,
                        })
                        .collect(),
                ),
                )?
                point_slices: super::leak_slice(
                    program
                        .point_slices
                        .iter()
                        .map(|plan| $module::$point_slice {
                            symbol: super::leak_str(&plan.symbol),
                            source: super::leak_str(&plan.source),
                            offset: plan.offset,
                            length: plan.length,
                            input: super::leak_str(&plan.input),
                        })
                        .collect(),
                ),
                point_concats: super::leak_slice(
                    program
                        .point_concats
                        .iter()
                        .map(|plan| $module::$point_concat {
                            symbol: super::leak_str(&plan.symbol),
                            layout: super::leak_str(&plan.layout),
                            arity: plan.arity,
                            inputs: stage_list!($mode, &plan.inputs),
                        })
                        .collect(),
                ),
                opening_claims: super::leak_slice(
                    program
                        .opening_claims
                        .iter()
                        .map(|plan| $module::$opening_claim {
                            symbol: super::leak_str(&plan.symbol),
                            oracle: super::leak_str(&plan.oracle),
                            domain: super::leak_str(&plan.domain),
                            point_arity: plan.point_arity,
                            claim_kind: super::leak_str(&plan.claim_kind),
                            point_source: super::leak_str(&plan.point_source),
                            eval_source: super::leak_str(&plan.eval_source),
                        })
                        .collect(),
                ),
                $(
                opening_equalities: super::leak_slice(
                    program
                        .opening_equalities
                        .iter()
                        .map(|plan| $module::$opening_equality {
                            symbol: super::leak_str(&plan.symbol),
                            mode: super::leak_str(&plan.mode),
                            lhs: super::leak_str(&plan.lhs),
                            rhs: super::leak_str(&plan.rhs),
                        })
                        .collect(),
                ),
                )?
                opening_batches: super::leak_slice(
                    program
                        .opening_batches
                        .iter()
                        .map(|plan| $module::$opening_batch {
                            symbol: super::leak_str(&plan.symbol),
                            stage: super::leak_str(&plan.stage),
                            proof_slot: super::leak_str(&plan.proof_slot),
                            policy: super::leak_str(&plan.policy),
                            count: plan.count,
                            ordered_claims: stage_list!($mode, &plan.ordered_claims),
                            claim_operands: stage_list!($mode, &plan.claim_operands),
                        })
                        .collect(),
                ),
            }))
        }
    };
}

macro_rules! define_stage_adapter {
    (
        $mode:ident,
        $function:ident,
        $compiler:ty,
        $module:ident,
        $program:ident,
        $params:ident,
        $step:ident,
        $squeeze:ident,
        $absorb:ident,
        $opening_input:ident,
        $field_constant:ident,
        $field_expr:ident,
        $kernel:ident,
        $claim:ident,
        $batch:ident,
        $driver:ident,
        $instance_result:ident,
        $eval:ident,
        $point_slice:ident,
        $point_concat:ident,
        $opening_claim:ident,
        $opening_equality:ident,
        $opening_batch:ident
        $(, point_zero = $point_zero:ident)?
    ) => {
        define_stage_adapter_impl!(
            $mode,
            $function,
            $compiler,
            $module,
            $program,
            $params,
            $step,
            $squeeze,
            $opening_input,
            $field_constant,
            $field_expr,
            $claim,
            $batch,
            $driver,
            $instance_result,
            $eval,
            $point_slice,
            $point_concat,
            $opening_claim,
            $opening_batch,
            role = role,
            transcript_absorb_bytes = $absorb,
            kernels = $kernel
            $(, point_zeros = $point_zero)?,
            opening_equalities = $opening_equality
        );
    };
}

macro_rules! define_stage_adapter_no_absorb {
    (
        $mode:ident,
        $function:ident,
        $compiler:ty,
        $module:ident,
        $program:ident,
        $params:ident,
        $step:ident,
        $squeeze:ident,
        $opening_input:ident,
        $field_constant:ident,
        $field_expr:ident,
        $claim:ident,
        $batch:ident,
        $driver:ident,
        $instance_result:ident,
        $eval:ident,
        $point_slice:ident,
        $point_concat:ident,
        $opening_claim:ident,
        $opening_batch:ident
        $(, kernels = $kernel:ident)?
        $(, opening_equalities = $opening_equality:ident)?
    ) => {
        define_stage_adapter_impl!(
            $mode,
            $function,
            $compiler,
            $module,
            $program,
            $params,
            $step,
            $squeeze,
            $opening_input,
            $field_constant,
            $field_expr,
            $claim,
            $batch,
            $driver,
            $instance_result,
            $eval,
            $point_slice,
            $point_concat,
            $opening_claim,
            $opening_batch
            $(, kernels = $kernel)?
            $(, opening_equalities = $opening_equality)?
        );
    };
}

macro_rules! define_stage1_adapter {
    (
        $mode:ident,
        $function:ident,
        $compiler:ty,
        $module:ident,
        $program:ident,
        $params:ident,
        $squeeze:ident,
        $claim:ident,
        $batch:ident,
        $driver:ident,
        $instance_result:ident,
        $eval:ident,
        $opening_claim:ident,
        $opening_batch:ident
        $(, kernels = $kernel:ident)?
    ) => {
        pub fn $function(program: &$compiler) -> &'static $module::$program {
            Box::leak(Box::new($module::$program {
                params: $module::$params {
                    field: super::leak_str(&program.params.field),
                    pcs: super::leak_str(&program.params.pcs),
                    transcript: super::leak_str(&program.params.transcript),
                },
                transcript_squeezes: super::leak_slice(
                    program
                        .transcript_squeezes
                        .iter()
                        .map(|plan| $module::$squeeze {
                            symbol: super::leak_str(&plan.symbol),
                            label: super::leak_str(&plan.label),
                            kind: super::leak_str(&plan.kind),
                            count: plan.count,
                        })
                        .collect(),
                ),
                $(
                kernels: super::leak_slice(
                    program
                        .kernels
                        .iter()
                        .map(|plan| $module::$kernel {
                            symbol: super::leak_str(&plan.symbol),
                            relation: super::leak_str(&plan.relation),
                            kind: super::leak_str(&plan.kind),
                            backend: super::leak_str(&plan.backend),
                            abi: super::leak_str(&plan.abi),
                        })
                        .collect(),
                ),
                )?
                claims: super::leak_slice(
                    program
                        .claims
                        .iter()
                        .map(|plan| stage_claim!($mode, $module, $claim, plan))
                        .collect(),
                ),
                batches: super::leak_slice(
                    program
                        .batches
                        .iter()
                        .map(|plan| $module::$batch {
                            symbol: super::leak_str(&plan.symbol),
                            stage: super::leak_str(&plan.stage),
                            proof_slot: super::leak_str(&plan.proof_slot),
                            policy: super::leak_str(&plan.policy),
                            count: plan.count,
                            ordered_claims: stage_list!($mode, &plan.ordered_claims),
                            claim_operands: stage_list!($mode, &plan.claim_operands),
                            claim_label: super::leak_str(&plan.claim_label),
                            round_label: super::leak_str(&plan.round_label),
                            round_schedule: super::leak_usize_slice(&plan.round_schedule),
                        })
                        .collect(),
                ),
                drivers: super::leak_slice(
                    program
                        .drivers
                        .iter()
                        .map(|plan| stage_driver!($module, $driver, plan))
                        .collect(),
                ),
                instance_results: super::leak_slice(
                    program
                        .instance_results
                        .iter()
                        .map(|plan| $module::$instance_result {
                            symbol: super::leak_str(&plan.symbol),
                            source: super::leak_str(&plan.source),
                            claim: super::leak_str(&plan.claim),
                            relation: super::leak_str(&plan.relation),
                            index: plan.index,
                            point_arity: plan.point_arity,
                            num_rounds: plan.num_rounds,
                            round_offset: plan.round_offset,
                            point_order: super::leak_str(&plan.point_order),
                            degree: plan.degree,
                        })
                        .collect(),
                ),
                evals: super::leak_slice(
                    program
                        .evals
                        .iter()
                        .map(|plan| $module::$eval {
                            symbol: super::leak_str(&plan.symbol),
                            source: super::leak_str(&plan.source),
                            name: super::leak_str(&plan.name),
                            index: plan.index,
                            oracle: super::leak_str(&plan.oracle),
                        })
                        .collect(),
                ),
                opening_claims: super::leak_slice(
                    program
                        .opening_claims
                        .iter()
                        .map(|plan| $module::$opening_claim {
                            symbol: super::leak_str(&plan.symbol),
                            oracle: super::leak_str(&plan.oracle),
                            domain: super::leak_str(&plan.domain),
                            point_arity: plan.point_arity,
                            claim_kind: super::leak_str(&plan.claim_kind),
                            point_source: super::leak_str(&plan.point_source),
                            eval_source: super::leak_str(&plan.eval_source),
                        })
                        .collect(),
                ),
                opening_batches: super::leak_slice(
                    program
                        .opening_batches
                        .iter()
                        .map(|plan| $module::$opening_batch {
                            symbol: super::leak_str(&plan.symbol),
                            stage: super::leak_str(&plan.stage),
                            proof_slot: super::leak_str(&plan.proof_slot),
                            policy: super::leak_str(&plan.policy),
                            count: plan.count,
                            ordered_claims: stage_list!($mode, &plan.ordered_claims),
                            claim_operands: stage_list!($mode, &plan.claim_operands),
                        })
                        .collect(),
                ),
            }))
        }
    };
}

mod generated_commitment;
mod generated_stage1;
mod generated_stage2;
mod generated_stage3;
mod generated_stage4;
mod generated_stage5;
mod generated_stage6;
mod generated_stage7;
mod stage1;
mod stage2;
mod stage3;
mod stage4;
mod stage5;
mod stage6;
mod stage7;

pub use generated_commitment::{
    leak_generated_commitment_prover_program, leak_generated_commitment_verifier_program,
};
pub use generated_stage1::leak_generated_stage1_verifier_program;
pub use generated_stage2::leak_generated_stage2_verifier_program;
pub use generated_stage3::leak_generated_stage3_verifier_program;
pub use generated_stage4::leak_generated_stage4_verifier_program;
pub use generated_stage5::leak_generated_stage5_verifier_program;
pub use generated_stage6::leak_generated_stage6_verifier_program;
pub use generated_stage7::leak_generated_stage7_verifier_program;
pub use stage1::leak_stage1_program;
pub use stage2::leak_stage2_program;
pub use stage3::leak_stage3_program;
pub use stage4::leak_stage4_program;
pub use stage5::leak_stage5_program;
pub use stage6::leak_stage6_program;
pub use stage7::leak_stage7_program;

use bolt::protocols::jolt::Stage8CpuProgram as CompilerStage8CpuProgram;
use bolt::Role;
use jolt_prover::stages::stage8 as generated_prover_stage8;
use jolt_verifier::stages::stage8 as generated_stage8;

macro_rules! define_stage8_adapter {
    ($function:ident, $module:ident) => {
        pub(crate) fn $function(
            program: &CompilerStage8CpuProgram,
        ) -> &'static $module::Stage8EvaluationProgramPlan {
            let evaluation_point_source = program
                .opening_inputs
                .iter()
                .find(|input| input.symbol == "stage8.evaluation.point_source")
                .expect("stage8 evaluation point source exists");
            Box::leak(Box::new($module::Stage8EvaluationProgramPlan {
                role: role_name(&program.role),
                function: leak_str(&program.function),
                params: $module::Stage8Params {
                    field: leak_str(&program.params.field),
                    pcs: leak_str(&program.params.pcs),
                    transcript: leak_str(&program.params.transcript),
                },
                evaluation_point_source: $module::Stage8OpeningInputPlan {
                    symbol: leak_str(&evaluation_point_source.symbol),
                    source_stage: leak_str(&evaluation_point_source.source_stage),
                    source_claim: leak_str(&evaluation_point_source.source_claim),
                    oracle: leak_str(&evaluation_point_source.oracle),
                    domain: leak_str(&evaluation_point_source.domain),
                    point_arity: evaluation_point_source.point_arity,
                    claim_kind: leak_str(&evaluation_point_source.claim_kind),
                },
                opening_inputs: leak_slice(
                    program
                        .opening_inputs
                        .iter()
                        .map(|plan| $module::Stage8OpeningInputPlan {
                            symbol: leak_str(&plan.symbol),
                            source_stage: leak_str(&plan.source_stage),
                            source_claim: leak_str(&plan.source_claim),
                            oracle: leak_str(&plan.oracle),
                            domain: leak_str(&plan.domain),
                            point_arity: plan.point_arity,
                            claim_kind: leak_str(&plan.claim_kind),
                        })
                        .collect(),
                ),
                opening_claims: leak_slice(
                    program
                        .opening_claims
                        .iter()
                        .map(|plan| $module::Stage8OpeningClaimPlan {
                            symbol: leak_str(&plan.symbol),
                            oracle: leak_str(&plan.oracle),
                            family: leak_str(&plan.family),
                            domain: leak_str(&plan.domain),
                            point_arity: plan.point_arity,
                            point_source: leak_str(&plan.point_source),
                            eval_source: leak_str(&plan.eval_source),
                            source_stage: leak_str(&plan.source_stage),
                            source_claim: leak_str(&plan.source_claim),
                        })
                        .collect(),
                ),
                opening_batch: $module::Stage8OpeningBatchPlan {
                    symbol: leak_str(&program.opening_batches[0].symbol),
                    proof_slot: leak_str(&program.opening_batches[0].proof_slot),
                    policy: leak_str(&program.opening_batches[0].policy),
                    count: program.opening_batches[0].count,
                    ordered_claims: leak_str_slice(&program.opening_batches[0].ordered_claims),
                },
                pcs_proof: $module::Stage8PcsProofPlan {
                    symbol: leak_str(&program.pcs_proofs[0].symbol),
                    mode: leak_str(&program.pcs_proofs[0].mode),
                    pcs: leak_str(&program.pcs_proofs[0].pcs),
                    proof_slot: leak_str(&program.pcs_proofs[0].proof_slot),
                    transcript_label: leak_str(&program.pcs_proofs[0].transcript_label),
                    batch: leak_str(&program.pcs_proofs[0].batch),
                },
            }))
        }
    };
}

define_stage8_adapter!(
    leak_generated_stage8_prover_program,
    generated_prover_stage8
);
define_stage8_adapter!(leak_generated_stage8_verifier_program, generated_stage8);

fn role_name(role: &Role) -> &'static str {
    match role {
        Role::Prover => "prover",
        Role::Verifier => "verifier",
    }
}

fn leak_str(value: &str) -> &'static str {
    Box::leak(value.to_owned().into_boxed_str())
}

fn leak_str_slice(values: &[String]) -> &'static [&'static str] {
    let leaked = values
        .iter()
        .map(|value| leak_str(value))
        .collect::<Vec<_>>();
    Box::leak(leaked.into_boxed_slice())
}

fn leak_symbol_list(values: &[String]) -> &'static str {
    leak_str(&values.join("|"))
}

fn leak_usize_slice(values: &[usize]) -> &'static [usize] {
    Box::leak(values.to_vec().into_boxed_slice())
}

fn leak_slice<T>(values: Vec<T>) -> &'static [T] {
    Box::leak(values.into_boxed_slice())
}
