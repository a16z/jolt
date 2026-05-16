//! Static plan adapters from Bolt compiler plans to generated/kernel plans.
//!
//! The equivalence oracle compares Bolt's owned compiler plans against the
//! generated static plans expected by jolt-kernels, jolt-prover, and
//! jolt-verifier. These adapters keep that comparison explicit.

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
            kind: super::generated_field_expr_kind($plan.formula.as_str()),
            operands: super::leak_str_slice(&$plan.operands),
        }
    };
}

macro_rules! stage_optional_relation_kind {
    (kernel, $value:expr) => {
        $value.map(super::leak_str)
    };
    (generated, $value:expr) => {
        $value.map(super::generated_relation_kind)
    };
}

macro_rules! stage_sumcheck_point_order {
    (kernel, $value:expr) => {
        super::leak_str($value)
    };
    (generated, $value:expr) => {
        super::generated_sumcheck_point_order($value)
    };
}

macro_rules! stage_claim {
    (kernel, $module:ident, $claim:ident, $plan:ident) => {
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
            input_openings: super::leak_str_slice(&$plan.input_openings),
        }
    };
    (generated, $module:ident, $claim:ident, $plan:ident) => {
        $module::$claim {
            symbol: super::leak_str(&$plan.symbol),
            stage: super::leak_str(&$plan.stage),
            domain: super::leak_str(&$plan.domain),
            num_rounds: $plan.num_rounds,
            degree: $plan.degree,
            claim: super::leak_str(&$plan.claim),
            kernel: $plan.kernel.as_deref().map(super::leak_str),
            relation: stage_optional_relation_kind!(generated, $plan.relation.as_deref()),
            claim_value: super::leak_str(&$plan.claim_value),
        }
    };
}

macro_rules! stage_sumcheck_batch {
    (kernel, $module:ident, $batch:ident, $plan:ident) => {
        $module::$batch {
            symbol: super::leak_str(&$plan.symbol),
            stage: super::leak_str(&$plan.stage),
            proof_slot: super::leak_str(&$plan.proof_slot),
            policy: super::leak_str(&$plan.policy),
            count: $plan.count,
            ordered_claims: super::leak_str_slice(&$plan.ordered_claims),
            claim_operands: super::leak_str_slice(&$plan.claim_operands),
            claim_label: super::leak_str(&$plan.claim_label),
            round_label: super::leak_str(&$plan.round_label),
            round_schedule: super::leak_usize_slice(&$plan.round_schedule),
        }
    };
    (generated, $module:ident, $batch:ident, $plan:ident) => {
        $module::$batch {
            symbol: super::leak_str(&$plan.symbol),
            stage: super::leak_str(&$plan.stage),
            proof_slot: super::leak_str(&$plan.proof_slot),
            policy: super::leak_str(&$plan.policy),
            count: $plan.count,
            claim_operands: super::leak_str_slice(&$plan.claim_operands),
            claim_label: super::leak_str(&$plan.claim_label),
            round_label: super::leak_str(&$plan.round_label),
            round_schedule: super::leak_usize_slice(&$plan.round_schedule),
        }
    };
}

macro_rules! stage_driver {
    (kernel, $module:ident, $driver:ident, $plan:ident) => {
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
    (generated, $module:ident, $driver:ident, $plan:ident) => {
        $module::$driver {
            symbol: super::leak_str(&$plan.symbol),
            stage: super::leak_str(&$plan.stage),
            proof_slot: super::leak_str(&$plan.proof_slot),
            kernel: $plan.kernel.as_deref().map(super::leak_str),
            relation: stage_optional_relation_kind!(generated, $plan.relation.as_deref()),
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

#[expect(
    clippy::panic,
    reason = "equivalence adapters fail fast when a compiler plan contains an unsupported generated verifier enum tag"
)]
fn generated_program_step_kind(value: &str) -> bolt_verifier_runtime::ProgramStepKind {
    match value {
        "transcript_squeeze" => bolt_verifier_runtime::ProgramStepKind::TranscriptSqueeze,
        "transcript_absorb_bytes" => bolt_verifier_runtime::ProgramStepKind::TranscriptAbsorbBytes,
        "sumcheck_driver" => bolt_verifier_runtime::ProgramStepKind::SumcheckDriver,
        value => panic!("unsupported generated program step kind `{value}`"),
    }
}

#[expect(
    clippy::panic,
    reason = "equivalence adapters fail fast when a compiler plan contains an unsupported generated verifier enum tag"
)]
fn generated_transcript_squeeze_kind(value: &str) -> bolt_verifier_runtime::TranscriptSqueezeKind {
    match value {
        "challenge_scalar" => bolt_verifier_runtime::TranscriptSqueezeKind::ChallengeScalar,
        "challenge_vector" => bolt_verifier_runtime::TranscriptSqueezeKind::ChallengeVector,
        "scalar" => bolt_verifier_runtime::TranscriptSqueezeKind::Scalar,
        value => panic!("unsupported generated transcript squeeze kind `{value}`"),
    }
}

#[expect(
    clippy::panic,
    reason = "equivalence adapters fail fast when a compiler plan contains an unsupported generated verifier enum tag"
)]
fn generated_claim_kind(value: &str) -> bolt_verifier_runtime::ClaimKind {
    match value {
        "committed" => bolt_verifier_runtime::ClaimKind::Committed,
        "virtual" => bolt_verifier_runtime::ClaimKind::Virtual,
        value => panic!("unsupported generated claim kind `{value}`"),
    }
}

#[expect(
    clippy::panic,
    reason = "equivalence adapters fail fast when a compiler plan contains an unsupported generated verifier enum tag"
)]
fn generated_relation_kind(value: &str) -> jolt_verifier::stages::jolt_relations::JoltRelationKind {
    match value {
        "jolt.stage1.outer.uniskip" => {
            jolt_verifier::stages::jolt_relations::JoltRelationKind::Stage1OuterUniskip
        }
        "jolt.stage1.outer.remaining" => {
            jolt_verifier::stages::jolt_relations::JoltRelationKind::Stage1OuterRemaining
        }
        "jolt.stage2.product_virtual.uniskip" => {
            jolt_verifier::stages::jolt_relations::JoltRelationKind::Stage2ProductVirtualUniskip
        }
        "jolt.stage2.ram.read_write" => {
            jolt_verifier::stages::jolt_relations::JoltRelationKind::Stage2RamReadWrite
        }
        "jolt.stage2.product_virtual.remainder" => {
            jolt_verifier::stages::jolt_relations::JoltRelationKind::Stage2ProductVirtualRemainder
        }
        "jolt.stage2.instruction_lookup.claim_reduction" => {
            jolt_verifier::stages::jolt_relations::JoltRelationKind::Stage2InstructionLookupClaimReduction
        }
        "jolt.stage2.ram.raf_evaluation" => {
            jolt_verifier::stages::jolt_relations::JoltRelationKind::Stage2RamRafEvaluation
        }
        "jolt.stage2.ram.output_check" => {
            jolt_verifier::stages::jolt_relations::JoltRelationKind::Stage2RamOutputCheck
        }
        "jolt.stage2.batched" => jolt_verifier::stages::jolt_relations::JoltRelationKind::Stage2Batched,
        "jolt.stage3.spartan_shift" => {
            jolt_verifier::stages::jolt_relations::JoltRelationKind::Stage3SpartanShift
        }
        "jolt.stage3.instruction_input" => {
            jolt_verifier::stages::jolt_relations::JoltRelationKind::Stage3InstructionInput
        }
        "jolt.stage3.registers_claim_reduction" => {
            jolt_verifier::stages::jolt_relations::JoltRelationKind::Stage3RegistersClaimReduction
        }
        "jolt.stage3.batched" => jolt_verifier::stages::jolt_relations::JoltRelationKind::Stage3Batched,
        "jolt.stage4.registers_read_write" => {
            jolt_verifier::stages::jolt_relations::JoltRelationKind::Stage4RegistersReadWrite
        }
        "jolt.stage4.ram_val_check" => {
            jolt_verifier::stages::jolt_relations::JoltRelationKind::Stage4RamValCheck
        }
        "jolt.stage4.batched" => jolt_verifier::stages::jolt_relations::JoltRelationKind::Stage4Batched,
        "jolt.stage5.instruction_read_raf" => {
            jolt_verifier::stages::jolt_relations::JoltRelationKind::Stage5InstructionReadRaf
        }
        "jolt.stage5.ram_ra_claim_reduction" => {
            jolt_verifier::stages::jolt_relations::JoltRelationKind::Stage5RamRaClaimReduction
        }
        "jolt.stage5.registers_val_evaluation" => {
            jolt_verifier::stages::jolt_relations::JoltRelationKind::Stage5RegistersValEvaluation
        }
        "jolt.stage5.batched" => jolt_verifier::stages::jolt_relations::JoltRelationKind::Stage5Batched,
        "jolt.stage6.bytecode_read_raf" => {
            jolt_verifier::stages::jolt_relations::JoltRelationKind::Stage6BytecodeReadRaf
        }
        "jolt.stage6.booleanity" => jolt_verifier::stages::jolt_relations::JoltRelationKind::Stage6Booleanity,
        "jolt.stage6.hamming_booleanity" => {
            jolt_verifier::stages::jolt_relations::JoltRelationKind::Stage6HammingBooleanity
        }
        "jolt.stage6.ram_ra_virtual" => {
            jolt_verifier::stages::jolt_relations::JoltRelationKind::Stage6RamRaVirtual
        }
        "jolt.stage6.instruction_ra_virtual" => {
            jolt_verifier::stages::jolt_relations::JoltRelationKind::Stage6InstructionRaVirtual
        }
        "jolt.stage6.inc_claim_reduction" => {
            jolt_verifier::stages::jolt_relations::JoltRelationKind::Stage6IncClaimReduction
        }
        "jolt.stage6.batched" => jolt_verifier::stages::jolt_relations::JoltRelationKind::Stage6Batched,
        "jolt.stage7.hamming_weight_claim_reduction" => {
            jolt_verifier::stages::jolt_relations::JoltRelationKind::Stage7HammingWeightClaimReduction
        }
        "jolt.stage7.batched" => jolt_verifier::stages::jolt_relations::JoltRelationKind::Stage7Batched,
        value => panic!("unsupported generated relation `{value}`"),
    }
}

#[expect(
    clippy::expect_used,
    clippy::panic,
    reason = "equivalence adapters fail fast when a compiler plan contains an unsupported generated verifier field expression tag"
)]
fn generated_field_expr_kind(value: &str) -> bolt_verifier_runtime::FieldExprKind {
    match value {
        "opening_eval" => bolt_verifier_runtime::FieldExprKind::OpeningEval,
        "field.add" => bolt_verifier_runtime::FieldExprKind::Add,
        "field.sub" => bolt_verifier_runtime::FieldExprKind::Sub,
        "field.mul" => bolt_verifier_runtime::FieldExprKind::Mul,
        "field.neg" => bolt_verifier_runtime::FieldExprKind::Neg,
        value if value.starts_with("field.pow:") => {
            let exponent = value
                .strip_prefix("field.pow:")
                .expect("field pow expression has prefix")
                .parse::<usize>()
                .expect("field pow expression has usize exponent");
            bolt_verifier_runtime::FieldExprKind::Pow(exponent)
        }
        value if value.starts_with("poly.lagrange_basis_eval:") => {
            let spec = value
                .strip_prefix("poly.lagrange_basis_eval:")
                .expect("lagrange expression has prefix");
            let parts = spec.split(':').collect::<Vec<_>>();
            assert!(parts.len() == 3, "lagrange expression has three fields");
            bolt_verifier_runtime::FieldExprKind::LagrangeBasisEval(
                parts[0]
                    .parse::<i64>()
                    .expect("lagrange domain start is i64"),
                parts[1]
                    .parse::<usize>()
                    .expect("lagrange domain size is usize"),
                parts[2].parse::<usize>().expect("lagrange index is usize"),
            )
        }
        value => panic!("unsupported generated field expression kind `{value}`"),
    }
}

#[expect(
    clippy::panic,
    reason = "equivalence adapters fail fast when a compiler plan contains an unsupported generated verifier enum tag"
)]
fn generated_opening_equality_mode(value: &str) -> bolt_verifier_runtime::OpeningEqualityMode {
    match value {
        "point_and_eval" => bolt_verifier_runtime::OpeningEqualityMode::PointAndEval,
        value => panic!("unsupported generated opening equality mode `{value}`"),
    }
}

#[expect(
    clippy::panic,
    reason = "equivalence adapters fail fast when a compiler plan contains an unsupported generated verifier enum tag"
)]
fn generated_structured_polynomial_kind(
    value: &str,
) -> bolt_verifier_runtime::StructuredPolynomialKind {
    match value {
        "eq" => bolt_verifier_runtime::StructuredPolynomialKind::Eq,
        "eq_plus_one" => bolt_verifier_runtime::StructuredPolynomialKind::EqPlusOne,
        "lt" => bolt_verifier_runtime::StructuredPolynomialKind::Lt,
        value => panic!("unsupported generated structured polynomial `{value}`"),
    }
}

#[expect(
    clippy::panic,
    reason = "equivalence adapters fail fast when a compiler plan contains an unsupported generated verifier enum tag"
)]
fn generated_structured_polynomial_point_segment(
    value: &str,
) -> bolt_verifier_runtime::StructuredPolynomialPointSegment {
    match value {
        "full" => bolt_verifier_runtime::StructuredPolynomialPointSegment::Full,
        "prefix" => bolt_verifier_runtime::StructuredPolynomialPointSegment::Prefix,
        "suffix" => bolt_verifier_runtime::StructuredPolynomialPointSegment::Suffix,
        value => panic!("unsupported generated structured polynomial point segment `{value}`"),
    }
}

#[expect(
    clippy::panic,
    reason = "equivalence adapters fail fast when a compiler plan contains an unsupported generated verifier enum tag"
)]
fn generated_structured_polynomial_point_length(
    value: &str,
) -> bolt_verifier_runtime::StructuredPolynomialPointLength {
    match value {
        "full" => bolt_verifier_runtime::StructuredPolynomialPointLength::Full,
        "x_point" => bolt_verifier_runtime::StructuredPolynomialPointLength::XPoint,
        "y_point" => bolt_verifier_runtime::StructuredPolynomialPointLength::YPoint,
        value => panic!("unsupported generated structured polynomial point length `{value}`"),
    }
}

#[expect(
    clippy::panic,
    reason = "equivalence adapters fail fast when a compiler plan contains an unsupported generated verifier enum tag"
)]
fn generated_structured_polynomial_point_order(
    value: &str,
) -> bolt_verifier_runtime::StructuredPolynomialPointOrder {
    match value {
        "as_is" => bolt_verifier_runtime::StructuredPolynomialPointOrder::AsIs,
        "reverse" => bolt_verifier_runtime::StructuredPolynomialPointOrder::Reverse,
        value => panic!("unsupported generated structured polynomial point order `{value}`"),
    }
}

#[expect(
    clippy::panic,
    reason = "equivalence adapters fail fast when a compiler plan contains an unsupported generated verifier enum tag"
)]
fn generated_sumcheck_point_order(value: &str) -> bolt_verifier_runtime::SumcheckPointOrder {
    match value {
        "as_is" => bolt_verifier_runtime::SumcheckPointOrder::AsIs,
        "reverse" => bolt_verifier_runtime::SumcheckPointOrder::Reverse,
        "stage4_registers_rw" => {
            bolt_verifier_runtime::SumcheckPointOrder::Stage4RegistersReadWrite
        }
        "instruction_read_raf" => bolt_verifier_runtime::SumcheckPointOrder::InstructionReadRaf,
        "bytecode_read_raf" => bolt_verifier_runtime::SumcheckPointOrder::BytecodeReadRaf,
        "stage6_booleanity" => bolt_verifier_runtime::SumcheckPointOrder::Stage6Booleanity,
        value => panic!("unsupported generated sumcheck point order `{value}`"),
    }
}

#[expect(
    clippy::panic,
    reason = "equivalence adapters fail fast when a compiler plan contains an unsupported generated verifier enum tag"
)]
fn generated_relation_output_function_kind(
    value: &str,
) -> bolt_verifier_runtime::RelationOutputFunctionKind {
    match value {
        "boolean_zero" => bolt_verifier_runtime::RelationOutputFunctionKind::BooleanZero,
        value => panic!("unsupported generated output function `{value}`"),
    }
}

macro_rules! stage_program_step_kind {
    (kernel, $module:ident, $value:expr) => {
        super::leak_str($value)
    };
    (generated, $module:ident, $value:expr) => {
        super::generated_program_step_kind($value)
    };
}

macro_rules! stage_transcript_squeeze_kind {
    (kernel, $module:ident, $value:expr) => {
        super::leak_str($value)
    };
    (generated, $module:ident, $value:expr) => {
        super::generated_transcript_squeeze_kind($value)
    };
}

macro_rules! stage_claim_kind {
    (kernel, $module:ident, $value:expr) => {
        super::leak_str($value)
    };
    (generated, $module:ident, $value:expr) => {
        super::generated_claim_kind($value)
    };
}

macro_rules! stage_relation_kind {
    (kernel, $module:ident, $value:expr) => {
        super::leak_str($value)
    };
    (generated, $module:ident, $value:expr) => {
        super::generated_relation_kind($value)
    };
}

macro_rules! stage_opening_equality_mode {
    (kernel, $module:ident, $value:expr) => {
        super::leak_str($value)
    };
    (generated, $module:ident, $value:expr) => {
        super::generated_opening_equality_mode($value)
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
        $(, relation_outputs = $relation_output:ident, relation_output_values = $relation_output_value:ident)?
        $(, empty_relation_outputs = $empty_relation_outputs:ident)?
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
                            kind: stage_program_step_kind!($mode, $module, plan.kind.as_str()),
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
                            kind: stage_transcript_squeeze_kind!($mode, $module, plan.kind.as_str()),
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
                            claim_kind: stage_claim_kind!($mode, $module, plan.claim_kind.as_str()),
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
                        .map(|plan| stage_sumcheck_batch!($mode, $module, $batch, plan))
                        .collect(),
                ),
                drivers: super::leak_slice(
                    program
                        .drivers
                        .iter()
                        .map(|plan| stage_driver!($mode, $module, $driver, plan))
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
                            relation: stage_relation_kind!($mode, $module, plan.relation.as_str()),
                            index: plan.index,
                            point_arity: plan.point_arity,
                            num_rounds: plan.num_rounds,
                            round_offset: plan.round_offset,
                            point_order: stage_sumcheck_point_order!($mode, &plan.point_order),
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
                relation_output_values: super::leak_slice(
                    program
                        .relation_output_values
                        .iter()
                        .map(|value| $module::$relation_output_value {
                            symbol: super::leak_str(&value.symbol),
                            polynomial: super::generated_structured_polynomial_kind(value.polynomial.as_str()),
                            x_point: bolt_verifier_runtime::StructuredPolynomialPointPlan {
                                source: super::leak_str(&value.x_point.source),
                                segment: super::generated_structured_polynomial_point_segment(value.x_point.segment.as_str()),
                                length: super::generated_structured_polynomial_point_length(value.x_point.length.as_str()),
                                order: super::generated_structured_polynomial_point_order(value.x_point.order.as_str()),
                            },
                            y_point: bolt_verifier_runtime::StructuredPolynomialPointPlan {
                                source: super::leak_str(&value.y_point.source),
                                segment: super::generated_structured_polynomial_point_segment(value.y_point.segment.as_str()),
                                length: super::generated_structured_polynomial_point_length(value.y_point.length.as_str()),
                                order: super::generated_structured_polynomial_point_order(value.y_point.order.as_str()),
                            },
                        })
                        .collect(),
                ),
                relation_outputs: super::leak_slice(
                    program
                        .relation_outputs
                        .iter()
                        .map(|plan| $module::$relation_output {
                            relation: super::generated_relation_kind(&plan.relation),
                            structured_polynomial_evals: super::leak_slice(
                                plan.structured_polynomial_evals
                                    .iter()
                                    .map(|value| bolt_verifier_runtime::StructuredPolynomialEvalRef {
                                        symbol: super::leak_str(&value.symbol),
                                        index: value.index,
                                    })
                                    .collect(),
                            ),
                            eval_families: super::leak_slice(
                                plan.eval_families
                                    .iter()
                                    .map(|family| bolt_verifier_runtime::RelationOutputEvalFamilyPlan {
                                        symbol: super::leak_str(&family.symbol),
                                        gamma: super::leak_str(&family.gamma),
                                        evals: super::leak_slice(
                                            family
                                                .evals
                                                .iter()
                                                .map(|symbol| super::leak_str(symbol))
                                                .collect(),
                                        ),
                                        power_stride: family.power_stride,
                                        value_term_offsets: super::leak_slice(
                                            family.value_term_offsets.clone(),
                                        ),
                                        shared_terms: super::leak_slice(
                                            family
                                                .shared_terms
                                                .iter()
                                                .map(|term| bolt_verifier_runtime::RelationOutputEvalFamilySharedTermPlan {
                                                    gamma_power_offset: term.gamma_power_offset,
                                                    factor: super::leak_str(&term.factor),
                                                })
                                                .collect(),
                                        ),
                                        item_terms: super::leak_slice(
                                            family
                                                .item_terms
                                                .iter()
                                                .map(|term| bolt_verifier_runtime::RelationOutputEvalFamilyItemTermPlan {
                                                    gamma_power_offset: term.gamma_power_offset,
                                                    factors: super::leak_slice(
                                                        term
                                                            .factors
                                                            .iter()
                                                            .map(|symbol| super::leak_str(symbol))
                                                            .collect(),
                                                    ),
                                                })
                                                .collect(),
                                        ),
                                    })
                                    .collect(),
                            ),
                            product_families: super::leak_slice(
                                plan.product_families
                                    .iter()
                                    .map(|family| bolt_verifier_runtime::RelationOutputProductFamilyPlan {
                                        symbol: super::leak_str(&family.symbol),
                                        gamma: family.gamma.as_ref().map(|gamma| super::leak_str(gamma)),
                                        terms: super::leak_slice(
                                            family
                                                .terms
                                                .iter()
                                                .map(|term| bolt_verifier_runtime::RelationOutputProductFamilyTermPlan {
                                                    gamma_power_offset: term.gamma_power_offset,
                                                    evals: super::leak_slice(
                                                        term
                                                            .evals
                                                            .iter()
                                                            .map(|symbol| super::leak_str(symbol))
                                                            .collect(),
                                                    ),
                                                    eval_families: super::leak_slice(
                                                        term
                                                            .eval_families
                                                            .iter()
                                                            .map(|symbol| super::leak_str(symbol))
                                                            .collect(),
                                                    ),
                                                    factors: super::leak_slice(
                                                        term
                                                            .factors
                                                            .iter()
                                                            .map(|symbol| super::leak_str(symbol))
                                                            .collect(),
                                                    ),
                                                })
                                                .collect(),
                                        ),
                                    })
                                    .collect(),
                            ),
                            function_families: super::leak_slice(
                                plan.function_families
                                    .iter()
                                    .map(|family| bolt_verifier_runtime::RelationOutputFunctionFamilyPlan {
                                        symbol: super::leak_str(&family.symbol),
                                        gamma: family.gamma.as_ref().map(|gamma| super::leak_str(gamma)),
                                        terms: super::leak_slice(
                                            family
                                                .terms
                                                .iter()
                                                .map(|term| bolt_verifier_runtime::RelationOutputFunctionFamilyTermPlan {
                                                    gamma_power_offset: term.gamma_power_offset,
                                                    function: super::generated_relation_output_function_kind(term.function.as_str()),
                                                    eval: super::leak_str(&term.eval),
                                                    factors: super::leak_slice(
                                                        term
                                                            .factors
                                                            .iter()
                                                            .map(|symbol| super::leak_str(symbol))
                                                            .collect(),
                                                    ),
                                                })
                                                .collect(),
                                        ),
                                    })
                                    .collect(),
                            ),
                            local_scalars: super::leak_str_slice(&plan.local_scalars),
                            expected_output: super::leak_str(&plan.expected_output),
                        })
                        .collect(),
                ),
                )?
                $(
                relation_output_values: {
                    let _ = stringify!($empty_relation_outputs);
                    &[]
                },
                relation_outputs: {
                    let _ = stringify!($empty_relation_outputs);
                    &[]
                },
                )?
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
                            inputs: super::leak_str_slice(&plan.inputs),
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
                            claim_kind: stage_claim_kind!($mode, $module, plan.claim_kind.as_str()),
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
                            mode: stage_opening_equality_mode!($mode, $module, plan.mode.as_str()),
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
                            ordered_claims: super::leak_str_slice(&plan.ordered_claims),
                            claim_operands: super::leak_str_slice(&plan.claim_operands),
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
        $(, relation_outputs = $relation_output:ident, relation_output_values = $relation_output_value:ident)?
        $(, empty_relation_outputs = $empty_relation_outputs:ident)?
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
            $(, point_zeros = $point_zero)?
            $(, relation_outputs = $relation_output, relation_output_values = $relation_output_value)?
            $(, empty_relation_outputs = $empty_relation_outputs)?
            ,
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
        $(, relation_outputs = $relation_output:ident, relation_output_values = $relation_output_value:ident)?
        $(, empty_relation_outputs = $empty_relation_outputs:ident)?
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
            $(, relation_outputs = $relation_output, relation_output_values = $relation_output_value)?
            $(, empty_relation_outputs = $empty_relation_outputs)?
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
                            kind: stage_transcript_squeeze_kind!($mode, $module, plan.kind.as_str()),
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
                        .map(|plan| stage_sumcheck_batch!($mode, $module, $batch, plan))
                        .collect(),
                ),
                drivers: super::leak_slice(
                    program
                        .drivers
                        .iter()
                        .map(|plan| stage_driver!($mode, $module, $driver, plan))
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
                            relation: stage_relation_kind!($mode, $module, plan.relation.as_str()),
                            index: plan.index,
                            point_arity: plan.point_arity,
                            num_rounds: plan.num_rounds,
                            round_offset: plan.round_offset,
                            point_order: stage_sumcheck_point_order!($mode, &plan.point_order),
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
                            claim_kind: stage_claim_kind!($mode, $module, plan.claim_kind.as_str()),
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
                            ordered_claims: super::leak_str_slice(&plan.ordered_claims),
                            claim_operands: super::leak_str_slice(&plan.claim_operands),
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

pub(crate) use generated_commitment::{
    leak_generated_commitment_prover_program, leak_generated_commitment_verifier_program,
};
pub use generated_stage1::leak_generated_stage1_verifier_program;
pub use generated_stage2::leak_generated_stage2_verifier_program;
pub(crate) use generated_stage3::leak_generated_stage3_verifier_program;
pub(crate) use generated_stage4::leak_generated_stage4_verifier_program;
pub(crate) use generated_stage5::leak_generated_stage5_verifier_program;
pub(crate) use generated_stage6::leak_generated_stage6_verifier_program;
pub(crate) use generated_stage7::leak_generated_stage7_verifier_program;
pub use stage1::leak_stage1_program;
pub use stage2::leak_stage2_program;
pub(crate) use stage3::leak_stage3_program;
pub(crate) use stage4::leak_stage4_program;
pub(crate) use stage5::leak_stage5_program;
pub(crate) use stage6::leak_stage6_program;
pub(crate) use stage7::leak_stage7_program;

use bolt::protocols::jolt::Stage8CpuProgram as CompilerStage8CpuProgram;
use bolt::Role;
use jolt_prover::stages::stage8 as generated_prover_stage8;
use jolt_verifier::stages::stage8 as generated_stage8;

#[expect(
    clippy::panic,
    reason = "equivalence adapters fail fast when a compiler plan contains an unsupported generated Stage 8 enum tag"
)]
fn generated_prover_stage8_source_stage(value: &str) -> generated_prover_stage8::Stage8SourceStage {
    match value {
        "stage6" => generated_prover_stage8::Stage8SourceStage::Stage6,
        "stage7" => generated_prover_stage8::Stage8SourceStage::Stage7,
        value => panic!("unsupported Stage 8 source stage `{value}`"),
    }
}

#[expect(
    clippy::panic,
    reason = "equivalence adapters fail fast when a compiler plan contains an unsupported generated Stage 8 enum tag"
)]
fn generated_stage8_source_stage(value: &str) -> generated_stage8::Stage8SourceStage {
    match value {
        "stage6" => generated_stage8::Stage8SourceStage::Stage6,
        "stage7" => generated_stage8::Stage8SourceStage::Stage7,
        value => panic!("unsupported Stage 8 source stage `{value}`"),
    }
}

#[expect(
    clippy::panic,
    reason = "equivalence adapters fail fast when a compiler plan contains an unsupported generated Stage 8 enum tag"
)]
fn generated_prover_stage8_claim_kind(value: &str) -> generated_prover_stage8::Stage8ClaimKind {
    match value {
        "committed" => generated_prover_stage8::Stage8ClaimKind::Committed,
        "virtual" => generated_prover_stage8::Stage8ClaimKind::Virtual,
        value => panic!("unsupported Stage 8 claim kind `{value}`"),
    }
}

#[expect(
    clippy::panic,
    reason = "equivalence adapters fail fast when a compiler plan contains an unsupported generated Stage 8 enum tag"
)]
fn generated_stage8_claim_kind(value: &str) -> generated_stage8::Stage8ClaimKind {
    match value {
        "committed" => generated_stage8::Stage8ClaimKind::Committed,
        "virtual" => generated_stage8::Stage8ClaimKind::Virtual,
        value => panic!("unsupported Stage 8 claim kind `{value}`"),
    }
}

#[expect(
    clippy::panic,
    reason = "equivalence adapters fail fast when a compiler plan contains an unsupported generated Stage 8 enum tag"
)]
fn generated_prover_stage8_pcs_proof_mode(
    value: &str,
) -> generated_prover_stage8::Stage8PcsProofMode {
    match value {
        "open" => generated_prover_stage8::Stage8PcsProofMode::Open,
        "verify" => generated_prover_stage8::Stage8PcsProofMode::Verify,
        value => panic!("unsupported Stage 8 PCS proof mode `{value}`"),
    }
}

#[expect(
    clippy::panic,
    reason = "equivalence adapters fail fast when a compiler plan contains an unsupported generated Stage 8 enum tag"
)]
fn generated_stage8_pcs_proof_mode(value: &str) -> generated_stage8::Stage8PcsProofMode {
    match value {
        "open" => generated_stage8::Stage8PcsProofMode::Open,
        "verify" => generated_stage8::Stage8PcsProofMode::Verify,
        value => panic!("unsupported Stage 8 PCS proof mode `{value}`"),
    }
}

#[expect(
    clippy::expect_used,
    reason = "Stage 8 adapters consume compiler-validated plans and fail fast if required generated plan rows are missing"
)]
fn stage8_evaluation_point_source_index(program: &CompilerStage8CpuProgram) -> usize {
    program
        .opening_inputs
        .iter()
        .position(|input| input.symbol == "stage8.evaluation.point_source")
        .expect("stage8 evaluation point source exists")
}

#[expect(
    clippy::expect_used,
    reason = "Stage 8 adapters consume compiler-validated plans and fail fast if required generated plan rows are missing"
)]
fn stage8_ordered_claim_indices(program: &CompilerStage8CpuProgram) -> Vec<usize> {
    let claim_index_by_symbol = program
        .opening_claims
        .iter()
        .enumerate()
        .map(|(index, claim)| (claim.symbol.as_str(), index))
        .collect::<std::collections::BTreeMap<_, _>>();
    program
        .opening_batches
        .first()
        .expect("stage8 opening batch exists")
        .ordered_claims
        .iter()
        .map(|symbol| {
            *claim_index_by_symbol
                .get(symbol.as_str())
                .expect("stage8 opening batch claim exists")
        })
        .collect()
}

#[expect(
    clippy::expect_used,
    reason = "Stage 8 adapters consume compiler-validated plans and fail fast if required generated plan rows are missing"
)]
pub(crate) fn leak_generated_stage8_prover_program(
    program: &CompilerStage8CpuProgram,
) -> &'static generated_prover_stage8::Stage8EvaluationProgramPlan {
    let evaluation_point_source_index = stage8_evaluation_point_source_index(program);
    let ordered_claim_indices = stage8_ordered_claim_indices(program);
    let opening_inputs = leak_slice(
        program
            .opening_inputs
            .iter()
            .map(|plan| generated_prover_stage8::Stage8OpeningInputPlan {
                symbol: generated_prover_stage8::Stage8OpeningInputSymbol::new(leak_str(
                    &plan.symbol,
                )),
                source_stage: generated_prover_stage8_source_stage(plan.source_stage.as_str()),
                source_claim: generated_prover_stage8::Stage8SourceClaim::new(leak_str(
                    &plan.source_claim,
                )),
                oracle: leak_str(&plan.oracle),
                domain: leak_str(&plan.domain),
                point_arity: plan.point_arity,
                claim_kind: generated_prover_stage8_claim_kind(plan.claim_kind.as_str()),
            })
            .collect(),
    );
    let opening_claims = leak_slice(
        program
            .opening_claims
            .iter()
            .map(|plan| generated_prover_stage8::Stage8OpeningClaimPlan {
                symbol: generated_prover_stage8::Stage8OpeningClaimSymbol::new(leak_str(
                    &plan.symbol,
                )),
                oracle: leak_str(&plan.oracle),
                family: leak_str(&plan.family),
                domain: leak_str(&plan.domain),
                point_arity: plan.point_arity,
                point_source: generated_prover_stage8::Stage8OpeningInputSymbol::new(leak_str(
                    &plan.point_source,
                )),
                eval_source: generated_prover_stage8::Stage8OpeningInputSymbol::new(leak_str(
                    &plan.eval_source,
                )),
                source_stage: generated_prover_stage8_source_stage(plan.source_stage.as_str()),
                source_claim: generated_prover_stage8::Stage8SourceClaim::new(leak_str(
                    &plan.source_claim,
                )),
            })
            .collect(),
    );
    let evaluation_point_source = *opening_inputs
        .get(evaluation_point_source_index)
        .expect("stage8 evaluation point source exists");
    let ordered_claims = leak_slice(
        ordered_claim_indices
            .iter()
            .map(|index| {
                *opening_claims
                    .get(*index)
                    .expect("stage8 opening batch claim exists")
            })
            .collect(),
    );
    let opening_batch = program
        .opening_batches
        .first()
        .expect("stage8 opening batch exists");
    let pcs_proof = program.pcs_proofs.first().expect("stage8 PCS proof exists");
    Box::leak(Box::new(
        generated_prover_stage8::Stage8EvaluationProgramPlan {
            role: role_name(&program.role),
            function: leak_str(&program.function),
            params: generated_prover_stage8::Stage8Params {
                field: leak_str(&program.params.field),
                pcs: leak_str(&program.params.pcs),
                transcript: leak_str(&program.params.transcript),
            },
            evaluation_point_source,
            opening_inputs,
            opening_claims,
            opening_batch: generated_prover_stage8::Stage8OpeningBatchPlan {
                symbol: generated_prover_stage8::Stage8OpeningBatchSymbol::new(leak_str(
                    &opening_batch.symbol,
                )),
                proof_slot: leak_str(&opening_batch.proof_slot),
                policy: leak_str(&opening_batch.policy),
                count: opening_batch.count,
                ordered_claims,
            },
            pcs_proof: generated_prover_stage8::Stage8PcsProofPlan {
                symbol: leak_str(&pcs_proof.symbol),
                mode: generated_prover_stage8_pcs_proof_mode(pcs_proof.mode.as_str()),
                pcs: leak_str(&pcs_proof.pcs),
                proof_slot: leak_str(&pcs_proof.proof_slot),
                transcript_label: leak_str(&pcs_proof.transcript_label),
                batch: generated_prover_stage8::Stage8OpeningBatchSymbol::new(leak_str(
                    &pcs_proof.batch,
                )),
            },
        },
    ))
}

#[expect(
    clippy::expect_used,
    reason = "Stage 8 adapters consume compiler-validated plans and fail fast if required generated plan rows are missing"
)]
pub(crate) fn leak_generated_stage8_verifier_program(
    program: &CompilerStage8CpuProgram,
) -> &'static generated_stage8::Stage8EvaluationProgramPlan {
    let evaluation_point_source_index = stage8_evaluation_point_source_index(program);
    let ordered_claim_indices = stage8_ordered_claim_indices(program);
    let opening_inputs = leak_slice(
        program
            .opening_inputs
            .iter()
            .map(|plan| generated_stage8::Stage8OpeningInputPlan {
                symbol: generated_stage8::Stage8OpeningInputSymbol::new(leak_str(&plan.symbol)),
                source_stage: generated_stage8_source_stage(plan.source_stage.as_str()),
                source_claim: generated_stage8::Stage8SourceClaim::new(leak_str(
                    &plan.source_claim,
                )),
                oracle: leak_str(&plan.oracle),
                domain: leak_str(&plan.domain),
                point_arity: plan.point_arity,
                claim_kind: generated_stage8_claim_kind(plan.claim_kind.as_str()),
            })
            .collect(),
    );
    let opening_claims = leak_slice(
        program
            .opening_claims
            .iter()
            .map(|plan| generated_stage8::Stage8OpeningClaimPlan {
                symbol: generated_stage8::Stage8OpeningClaimSymbol::new(leak_str(&plan.symbol)),
                oracle: leak_str(&plan.oracle),
                family: leak_str(&plan.family),
                domain: leak_str(&plan.domain),
                point_arity: plan.point_arity,
                point_source: generated_stage8::Stage8OpeningInputSymbol::new(leak_str(
                    &plan.point_source,
                )),
                eval_source: generated_stage8::Stage8OpeningInputSymbol::new(leak_str(
                    &plan.eval_source,
                )),
                source_stage: generated_stage8_source_stage(plan.source_stage.as_str()),
                source_claim: generated_stage8::Stage8SourceClaim::new(leak_str(
                    &plan.source_claim,
                )),
            })
            .collect(),
    );
    let evaluation_point_source = *opening_inputs
        .get(evaluation_point_source_index)
        .expect("stage8 evaluation point source exists");
    let ordered_claims = leak_slice(
        ordered_claim_indices
            .iter()
            .map(|index| {
                *opening_claims
                    .get(*index)
                    .expect("stage8 opening batch claim exists")
            })
            .collect(),
    );
    let opening_batch = program
        .opening_batches
        .first()
        .expect("stage8 opening batch exists");
    let pcs_proof = program.pcs_proofs.first().expect("stage8 PCS proof exists");
    Box::leak(Box::new(generated_stage8::Stage8EvaluationProgramPlan {
        role: role_name(&program.role),
        function: leak_str(&program.function),
        params: generated_stage8::Stage8Params {
            field: leak_str(&program.params.field),
            pcs: leak_str(&program.params.pcs),
            transcript: leak_str(&program.params.transcript),
        },
        evaluation_point_source,
        opening_inputs,
        opening_claims,
        opening_batch: generated_stage8::Stage8OpeningBatchPlan {
            symbol: generated_stage8::Stage8OpeningBatchSymbol::new(leak_str(
                &opening_batch.symbol,
            )),
            proof_slot: leak_str(&opening_batch.proof_slot),
            policy: leak_str(&opening_batch.policy),
            count: opening_batch.count,
            ordered_claims,
        },
        pcs_proof: generated_stage8::Stage8PcsProofPlan {
            symbol: leak_str(&pcs_proof.symbol),
            mode: generated_stage8_pcs_proof_mode(pcs_proof.mode.as_str()),
            pcs: leak_str(&pcs_proof.pcs),
            proof_slot: leak_str(&pcs_proof.proof_slot),
            transcript_label: leak_str(&pcs_proof.transcript_label),
            batch: generated_stage8::Stage8OpeningBatchSymbol::new(leak_str(&pcs_proof.batch)),
        },
    }))
}

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

fn leak_usize_slice(values: &[usize]) -> &'static [usize] {
    Box::leak(values.to_vec().into_boxed_slice())
}

fn leak_slice<T>(values: Vec<T>) -> &'static [T] {
    Box::leak(values.into_boxed_slice())
}
