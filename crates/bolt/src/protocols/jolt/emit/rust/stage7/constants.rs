use super::Stage7CpuProgram;
use crate::ir::Role;
use crate::protocols::jolt::emit::rust::source::{
    emit_params_const, emit_plan_array, emit_plan_array_compact,
    emit_rustfmt_skip_macro_plan_array, emit_str_array, emit_struct_const_with_literal,
    emit_usize_array, intern_str_array, rust_option_str, rust_str,
};

impl Stage7CpuProgram {
    pub(super) fn emit_constants(&self) -> String {
        let mut source = self.emit_shared_constants();
        source.push_str(&self.emit_kernel_constants());
        source.push_str(&self.emit_sumcheck_claim_constants());
        source.push_str(&self.emit_sumcheck_batch_constants());
        source.push_str(&self.emit_sumcheck_driver_constants());
        source.push_str(&self.emit_tail_constants());
        let role = rust_str(self.role_label());
        source.push_str(&emit_struct_const_with_literal(
            "STAGE7_PROGRAM",
            self.program_plan_type(),
            "Stage7CpuProgramPlan",
            &[
                ("role", &role),
                ("params", "STAGE7_PARAMS"),
                ("steps", "STAGE7_PROGRAM_STEPS"),
                ("transcript_squeezes", "STAGE7_TRANSCRIPT_SQUEEZES"),
                ("transcript_absorb_bytes", "STAGE7_TRANSCRIPT_ABSORB_BYTES"),
                ("opening_inputs", "STAGE7_OPENING_INPUTS"),
                ("field_constants", "STAGE7_FIELD_CONSTANTS"),
                ("field_exprs", "STAGE7_FIELD_EXPRS"),
                ("kernels", "STAGE7_KERNELS"),
                ("claims", "STAGE7_SUMCHECK_CLAIMS"),
                ("batches", "STAGE7_SUMCHECK_BATCHES"),
                ("drivers", "STAGE7_SUMCHECK_DRIVERS"),
                ("instance_results", "STAGE7_SUMCHECK_INSTANCE_RESULTS"),
                ("evals", "STAGE7_SUMCHECK_EVALS"),
                ("point_zeros", "STAGE7_POINT_ZEROS"),
                ("point_slices", "STAGE7_POINT_SLICES"),
                ("point_concats", "STAGE7_POINT_CONCATS"),
                ("opening_claims", "STAGE7_OPENING_CLAIMS"),
                ("opening_equalities", "STAGE7_OPENING_EQUALITIES"),
                ("opening_batches", "STAGE7_OPENING_BATCHES"),
            ],
        ));
        source
    }

    fn emit_shared_constants(&self) -> String {
        let mut source = emit_params_const(
            "STAGE7_PARAMS",
            "Stage7Params",
            &self.params.field,
            &self.params.pcs,
            &self.params.transcript,
        );
        source.push_str(&self.emit_program_step_constants());
        source.push_str(&self.emit_transcript_squeeze_constants());
        source.push_str(&self.emit_transcript_absorb_bytes_constants());
        source.push_str(&self.emit_opening_input_constants());
        source.push_str(&self.emit_field_constant_constants());
        source.push_str(&self.emit_field_expr_constants());
        source
    }

    fn emit_program_step_constants(&self) -> String {
        emit_plan_array(
            "STAGE7_PROGRAM_STEPS",
            "Stage7ProgramStepPlan",
            self.steps.iter().map(|step| {
                format!(
                    "    Stage7ProgramStepPlan {{ kind: {}, symbol: {} }},",
                    rust_str(&step.kind),
                    rust_str(&step.symbol),
                )
            }),
        )
    }

    fn emit_transcript_squeeze_constants(&self) -> String {
        emit_plan_array(
            "STAGE7_TRANSCRIPT_SQUEEZES",
            "Stage7TranscriptSqueezePlan",
            self.transcript_squeezes.iter().map(|squeeze| {
                format!(
                    "    Stage7TranscriptSqueezePlan {{ symbol: {}, label: {}, kind: {}, count: {} }},",
                    rust_str(&squeeze.symbol),
                    rust_str(&squeeze.label),
                    rust_str(&squeeze.kind),
                    squeeze.count,
                )
            }),
        )
    }

    fn emit_transcript_absorb_bytes_constants(&self) -> String {
        emit_plan_array(
            "STAGE7_TRANSCRIPT_ABSORB_BYTES",
            "Stage7TranscriptAbsorbBytesPlan",
            self.transcript_absorb_bytes.iter().map(|absorb| {
                format!(
                    "    Stage7TranscriptAbsorbBytesPlan {{ symbol: {}, label: {}, payload: {} }},",
                    rust_str(&absorb.symbol),
                    rust_str(&absorb.label),
                    rust_str(&absorb.payload),
                )
            }),
        )
    }

    fn emit_opening_input_constants(&self) -> String {
        emit_plan_array(
            "STAGE7_OPENING_INPUTS",
            "Stage7OpeningInputPlan",
            self.opening_inputs.iter().map(|input| {
                format!(
                    "    Stage7OpeningInputPlan {{ symbol: {}, source_stage: {}, source_claim: {}, oracle: {}, domain: {}, point_arity: {}, claim_kind: {} }},",
                    rust_str(&input.symbol),
                    rust_str(&input.source_stage),
                    rust_str(&input.source_claim),
                    rust_str(&input.oracle),
                    rust_str(&input.domain),
                    input.point_arity,
                    rust_str(&input.claim_kind)
                )
            }),
        )
    }

    fn emit_field_constant_constants(&self) -> String {
        emit_plan_array(
            "STAGE7_FIELD_CONSTANTS",
            "Stage7FieldConstantPlan",
            self.field_constants.iter().map(|constant| {
                format!(
                    "    Stage7FieldConstantPlan {{ symbol: {}, field: {}, value: {} }},",
                    rust_str(&constant.symbol),
                    rust_str(&constant.field),
                    constant.value
                )
            }),
        )
    }

    fn emit_field_expr_constants(&self) -> String {
        if self.role == Role::Verifier {
            let rows = self
                .field_exprs
                .chunks(8)
                .map(|chunk| {
                    let exprs = chunk
                        .iter()
                        .map(|expr| {
                            format!(
                                "stage7_field_expr!({}, {}, {})",
                                rust_str(&expr.symbol),
                                rust_str(&expr.formula),
                                rust_str(&expr.operands.join("|"))
                            )
                        })
                        .collect::<Vec<_>>()
                        .join(", ");
                    format!("    {exprs},")
                })
                .collect::<Vec<_>>()
                .join("\n");
            return emit_rustfmt_skip_macro_plan_array(
                "macro_rules! stage7_field_expr {\n    ($symbol:literal, $formula:literal, $operands:literal) => {\n        Stage7FieldExprPlan { symbol: $symbol, kind: \"op\", formula: $formula, operands: $operands }\n    };\n}",
                "STAGE7_FIELD_EXPRS",
                "Stage7FieldExprPlan",
                rows,
                "\n",
            );
        }

        let mut source = String::new();
        let mut arrays = Vec::new();
        let mut array_refs = Vec::new();
        for (index, expr) in self.field_exprs.iter().enumerate() {
            let operands = intern_str_array(
                &mut source,
                &mut arrays,
                "STAGE7_FIELD_EXPR_OPERANDS",
                &expr.operands,
            );
            let operand_names = intern_str_array(
                &mut source,
                &mut arrays,
                "STAGE7_FIELD_EXPR_OPERANDS",
                &expr.operand_names,
            );
            array_refs.push((index, operand_names, operands));
        }
        source.push_str(&emit_plan_array_compact(
            "STAGE7_FIELD_EXPRS",
            "Stage7FieldExprPlan",
            self.field_exprs.iter().enumerate().map(|(index, expr)| {
                let (_, operand_names, operands) = &array_refs[index];
                format!(
                    "    Stage7FieldExprPlan {{ symbol: {}, kind: {}, formula: {}, operand_names: {operand_names}, operands: {operands} }},",
                    rust_str(&expr.symbol),
                    rust_str(&expr.kind),
                    rust_str(&expr.formula)
                )
            }),
        ));
        source
    }

    fn emit_kernel_constants(&self) -> String {
        emit_plan_array(
            "STAGE7_KERNELS",
            "Stage7KernelPlan",
            self.kernels.iter().map(|kernel| {
                format!(
                    "    Stage7KernelPlan {{ symbol: {}, relation: {}, kind: {}, backend: {}, abi: {} }},",
                    rust_str(&kernel.symbol),
                    rust_str(&kernel.relation),
                    rust_str(&kernel.kind),
                    rust_str(&kernel.backend),
                    rust_str(&kernel.abi)
                )
            }),
        )
    }

    fn emit_sumcheck_claim_constants(&self) -> String {
        if self.role == Role::Verifier {
            return emit_plan_array_compact(
                "STAGE7_SUMCHECK_CLAIMS",
                "Stage7SumcheckClaimPlan",
                self.claims.iter().map(|claim| {
                    format!(
                        "    Stage7SumcheckClaimPlan {{ symbol: {}, stage: {}, domain: {}, num_rounds: {}, degree: {}, claim: {}, kernel: {}, relation: {}, claim_value: {}, input_openings: {} }},",
                        rust_str(&claim.symbol),
                        rust_str(&claim.stage),
                        rust_str(&claim.domain),
                        claim.num_rounds,
                        claim.degree,
                        rust_str(&claim.claim),
                        rust_option_str(claim.kernel.as_deref()),
                        rust_option_str(claim.relation.as_deref()),
                        rust_str(&claim.claim_value),
                        rust_str(&claim.input_openings.join("|"))
                    )
                }),
            );
        }

        let mut source = String::new();
        for (index, claim) in self.claims.iter().enumerate() {
            source.push_str(&emit_str_array(
                &format!("STAGE7_SUMCHECK_CLAIM_{index}_INPUT_OPENINGS"),
                &claim.input_openings,
            ));
        }
        source.push_str(&emit_plan_array_compact(
            "STAGE7_SUMCHECK_CLAIMS",
            "Stage7SumcheckClaimPlan",
            self.claims.iter().enumerate().map(|(index, claim)| {
                format!(
                    "    Stage7SumcheckClaimPlan {{ symbol: {}, stage: {}, domain: {}, num_rounds: {}, degree: {}, claim: {}, kernel: {}, relation: {}, claim_value: {}, input_openings: STAGE7_SUMCHECK_CLAIM_{index}_INPUT_OPENINGS }},",
                    rust_str(&claim.symbol),
                    rust_str(&claim.stage),
                    rust_str(&claim.domain),
                    claim.num_rounds,
                    claim.degree,
                    rust_str(&claim.claim),
                    rust_option_str(claim.kernel.as_deref()),
                    rust_option_str(claim.relation.as_deref()),
                    rust_str(&claim.claim_value)
                )
            }),
        ));
        source
    }

    fn emit_sumcheck_batch_constants(&self) -> String {
        if self.role == Role::Verifier {
            let mut source = String::new();
            for (index, batch) in self.batches.iter().enumerate() {
                source.push_str(&emit_usize_array(
                    &format!("STAGE7_SUMCHECK_BATCH_{index}_ROUND_SCHEDULE"),
                    &batch.round_schedule,
                ));
            }
            source.push_str(&emit_plan_array_compact(
                "STAGE7_SUMCHECK_BATCHES",
                "Stage7SumcheckBatchPlan",
                self.batches.iter().enumerate().map(|(index, batch)| {
                    format!(
                        "    Stage7SumcheckBatchPlan {{ symbol: {}, stage: {}, proof_slot: {}, policy: {}, count: {}, ordered_claims: {}, claim_operands: {}, claim_label: {}, round_label: {}, round_schedule: STAGE7_SUMCHECK_BATCH_{index}_ROUND_SCHEDULE }},",
                        rust_str(&batch.symbol),
                        rust_str(&batch.stage),
                        rust_str(&batch.proof_slot),
                        rust_str(&batch.policy),
                        batch.count,
                        rust_str(&batch.ordered_claims.join("|")),
                        rust_str(&batch.claim_operands.join("|")),
                        rust_str(&batch.claim_label),
                        rust_str(&batch.round_label)
                    )
                }),
            ));
            return source;
        }

        let mut source = String::new();
        for (index, batch) in self.batches.iter().enumerate() {
            source.push_str(&emit_str_array(
                &format!("STAGE7_SUMCHECK_BATCH_{index}_ORDERED_CLAIMS"),
                &batch.ordered_claims,
            ));
            source.push_str(&emit_str_array(
                &format!("STAGE7_SUMCHECK_BATCH_{index}_CLAIM_OPERANDS"),
                &batch.claim_operands,
            ));
            source.push_str(&emit_usize_array(
                &format!("STAGE7_SUMCHECK_BATCH_{index}_ROUND_SCHEDULE"),
                &batch.round_schedule,
            ));
        }
        source.push_str(&emit_plan_array_compact(
            "STAGE7_SUMCHECK_BATCHES",
            "Stage7SumcheckBatchPlan",
            self.batches.iter().enumerate().map(|(index, batch)| {
                format!(
                    "    Stage7SumcheckBatchPlan {{ symbol: {}, stage: {}, proof_slot: {}, policy: {}, count: {}, ordered_claims: STAGE7_SUMCHECK_BATCH_{index}_ORDERED_CLAIMS, claim_operands: STAGE7_SUMCHECK_BATCH_{index}_CLAIM_OPERANDS, claim_label: {}, round_label: {}, round_schedule: STAGE7_SUMCHECK_BATCH_{index}_ROUND_SCHEDULE }},",
                    rust_str(&batch.symbol),
                    rust_str(&batch.stage),
                    rust_str(&batch.proof_slot),
                    rust_str(&batch.policy),
                    batch.count,
                    rust_str(&batch.claim_label),
                    rust_str(&batch.round_label)
                )
            }),
        ));
        source
    }

    fn emit_sumcheck_driver_constants(&self) -> String {
        let mut source = String::new();
        for (index, driver) in self.drivers.iter().enumerate() {
            source.push_str(&emit_usize_array(
                &format!("STAGE7_SUMCHECK_DRIVER_{index}_ROUND_SCHEDULE"),
                &driver.round_schedule,
            ));
        }
        source.push_str(&emit_plan_array_compact(
            "STAGE7_SUMCHECK_DRIVERS",
            "Stage7SumcheckDriverPlan",
            self.drivers.iter().enumerate().map(|(index, driver)| {
                format!(
                    "    Stage7SumcheckDriverPlan {{ symbol: {}, stage: {}, proof_slot: {}, kernel: {}, relation: {}, batch: {}, policy: {}, round_schedule: STAGE7_SUMCHECK_DRIVER_{index}_ROUND_SCHEDULE, claim_label: {}, round_label: {}, num_rounds: {}, degree: {} }},",
                    rust_str(&driver.symbol),
                    rust_str(&driver.stage),
                    rust_str(&driver.proof_slot),
                    rust_option_str(driver.kernel.as_deref()),
                    rust_option_str(driver.relation.as_deref()),
                    rust_str(&driver.batch),
                    rust_str(&driver.policy),
                    rust_str(&driver.claim_label),
                    rust_str(&driver.round_label),
                    driver.num_rounds,
                    driver.degree
                )
            }),
        ));
        source
    }

    fn emit_tail_constants(&self) -> String {
        let mut source = String::new();
        source.push_str(&self.emit_sumcheck_instance_result_constants());
        source.push_str(&self.emit_sumcheck_eval_constants());
        source.push_str(&self.emit_point_zero_constants());
        source.push_str(&self.emit_point_slice_constants());
        source.push_str(&self.emit_point_concat_constants());
        source.push_str(&self.emit_opening_claim_constants());
        source.push_str(&self.emit_opening_claim_equality_constants());
        source.push_str(&self.emit_opening_batch_constants());
        source
    }

    fn emit_sumcheck_instance_result_constants(&self) -> String {
        emit_plan_array(
            "STAGE7_SUMCHECK_INSTANCE_RESULTS",
            "Stage7SumcheckInstanceResultPlan",
            self.instance_results.iter().map(|instance| {
                format!(
                    "    Stage7SumcheckInstanceResultPlan {{ symbol: {}, source: {}, claim: {}, relation: {}, index: {}, point_arity: {}, num_rounds: {}, round_offset: {}, point_order: {}, degree: {} }},",
                    rust_str(&instance.symbol),
                    rust_str(&instance.source),
                    rust_str(&instance.claim),
                    rust_str(&instance.relation),
                    instance.index,
                    instance.point_arity,
                    instance.num_rounds,
                    instance.round_offset,
                    rust_str(&instance.point_order),
                    instance.degree
                )
            }),
        )
    }

    fn emit_sumcheck_eval_constants(&self) -> String {
        let rows = self
            .evals
            .chunks(4)
            .map(|chunk| {
                let evals = chunk
                    .iter()
                    .map(|eval| {
                        format!(
                            "stage7_sumcheck_eval!({}, {}, {}, {}, {})",
                            rust_str(&eval.symbol),
                            rust_str(&eval.source),
                            rust_str(&eval.name),
                            eval.index,
                            rust_str(&eval.oracle)
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("    {evals},")
            })
            .collect::<Vec<_>>()
            .join("\n");
        emit_rustfmt_skip_macro_plan_array(
            "macro_rules! stage7_sumcheck_eval {\n    ($symbol:literal, $source:literal, $name:literal, $index:literal, $oracle:literal) => {\n        Stage7SumcheckEvalPlan { symbol: $symbol, source: $source, name: $name, index: $index, oracle: $oracle }\n    };\n}",
            "STAGE7_SUMCHECK_EVALS",
            "Stage7SumcheckEvalPlan",
            rows,
            "\n\n",
        )
    }

    fn emit_point_zero_constants(&self) -> String {
        emit_plan_array(
            "STAGE7_POINT_ZEROS",
            "Stage7PointZeroPlan",
            self.point_zeros.iter().map(|zero| {
                format!(
                    "    Stage7PointZeroPlan {{ symbol: {}, field: {}, arity: {} }},",
                    rust_str(&zero.symbol),
                    rust_str(&zero.field),
                    zero.arity
                )
            }),
        )
    }

    fn emit_point_slice_constants(&self) -> String {
        emit_plan_array(
            "STAGE7_POINT_SLICES",
            "Stage7PointSlicePlan",
            self.point_slices.iter().map(|slice| {
                format!(
                    "    Stage7PointSlicePlan {{ symbol: {}, source: {}, offset: {}, length: {}, input: {} }},",
                    rust_str(&slice.symbol),
                    rust_str(&slice.source),
                    slice.offset,
                    slice.length,
                    rust_str(&slice.input)
                )
            }),
        )
    }

    fn emit_point_concat_constants(&self) -> String {
        if self.role == Role::Verifier {
            return emit_plan_array_compact(
                "STAGE7_POINT_CONCATS",
                "Stage7PointConcatPlan",
                self.point_concats.iter().map(|concat| {
                    format!(
                        "    Stage7PointConcatPlan {{ symbol: {}, layout: {}, arity: {}, inputs: {} }},",
                        rust_str(&concat.symbol),
                        rust_str(&concat.layout),
                        concat.arity,
                        rust_str(&concat.inputs.join("|"))
                    )
                }),
            );
        }

        let mut source = String::new();
        for (index, concat) in self.point_concats.iter().enumerate() {
            source.push_str(&emit_str_array(
                &format!("STAGE7_POINT_CONCAT_{index}_INPUTS"),
                &concat.inputs,
            ));
        }
        source.push_str(&emit_plan_array_compact(
            "STAGE7_POINT_CONCATS",
            "Stage7PointConcatPlan",
            self.point_concats.iter().enumerate().map(|(index, concat)| {
                format!(
                    "    Stage7PointConcatPlan {{ symbol: {}, layout: {}, arity: {}, inputs: STAGE7_POINT_CONCAT_{index}_INPUTS }},",
                    rust_str(&concat.symbol),
                    rust_str(&concat.layout),
                    concat.arity
                )
            }),
        ));
        source
    }

    fn emit_opening_claim_constants(&self) -> String {
        emit_plan_array(
            "STAGE7_OPENING_CLAIMS",
            "Stage7OpeningClaimPlan",
            self.opening_claims.iter().map(|claim| {
                format!(
                    "    Stage7OpeningClaimPlan {{ symbol: {}, oracle: {}, domain: {}, point_arity: {}, claim_kind: {}, point_source: {}, eval_source: {} }},",
                    rust_str(&claim.symbol),
                    rust_str(&claim.oracle),
                    rust_str(&claim.domain),
                    claim.point_arity,
                    rust_str(&claim.claim_kind),
                    rust_str(&claim.point_source),
                    rust_str(&claim.eval_source)
                )
            }),
        )
    }

    fn emit_opening_claim_equality_constants(&self) -> String {
        emit_plan_array(
            "STAGE7_OPENING_EQUALITIES",
            "Stage7OpeningClaimEqualityPlan",
            self.opening_equalities.iter().map(|equality| {
                format!(
                    "    Stage7OpeningClaimEqualityPlan {{ symbol: {}, mode: {}, lhs: {}, rhs: {} }},",
                    rust_str(&equality.symbol),
                    rust_str(&equality.mode),
                    rust_str(&equality.lhs),
                    rust_str(&equality.rhs)
                )
            }),
        )
    }

    fn emit_opening_batch_constants(&self) -> String {
        if self.role == Role::Verifier {
            return emit_plan_array_compact(
                "STAGE7_OPENING_BATCHES",
                "Stage7OpeningBatchPlan",
                self.opening_batches.iter().map(|batch| {
                    format!(
                        "    Stage7OpeningBatchPlan {{ symbol: {}, stage: {}, proof_slot: {}, policy: {}, count: {}, ordered_claims: {}, claim_operands: {} }},",
                        rust_str(&batch.symbol),
                        rust_str(&batch.stage),
                        rust_str(&batch.proof_slot),
                        rust_str(&batch.policy),
                        batch.count,
                        rust_str(&batch.ordered_claims.join("|")),
                        rust_str(&batch.claim_operands.join("|"))
                    )
                }),
            );
        }

        let mut source = String::new();
        for (index, batch) in self.opening_batches.iter().enumerate() {
            source.push_str(&emit_str_array(
                &format!("STAGE7_OPENING_BATCH_{index}_ORDERED_CLAIMS"),
                &batch.ordered_claims,
            ));
            source.push_str(&emit_str_array(
                &format!("STAGE7_OPENING_BATCH_{index}_CLAIM_OPERANDS"),
                &batch.claim_operands,
            ));
        }
        source.push_str(&emit_plan_array_compact(
            "STAGE7_OPENING_BATCHES",
            "Stage7OpeningBatchPlan",
            self.opening_batches.iter().enumerate().map(|(index, batch)| {
                format!(
                    "    Stage7OpeningBatchPlan {{ symbol: {}, stage: {}, proof_slot: {}, policy: {}, count: {}, ordered_claims: STAGE7_OPENING_BATCH_{index}_ORDERED_CLAIMS, claim_operands: STAGE7_OPENING_BATCH_{index}_CLAIM_OPERANDS }},",
                    rust_str(&batch.symbol),
                    rust_str(&batch.stage),
                    rust_str(&batch.proof_slot),
                    rust_str(&batch.policy),
                    batch.count
                )
            }),
        ));
        source
    }
}
