use super::Stage3CpuProgram;
use crate::emit::rust::EmitError;
use crate::ir::Role;
use crate::protocols::jolt::emit::rust::checks::missing_role_binding;
use crate::protocols::jolt::emit::rust::source::{
    emit_params_const, emit_plan_array, emit_plan_array_compact, emit_str_array, emit_struct_const,
    emit_usize_array, intern_str_array, rust_str,
};

impl Stage3CpuProgram {
    pub(super) fn emit_prover_constants(&self) -> Result<String, EmitError> {
        let mut source = self.emit_shared_constants();
        source.push_str(&self.emit_kernel_constants());
        source.push_str(&self.emit_prover_sumcheck_claim_constants()?);
        source.push_str(&self.emit_sumcheck_batch_constants());
        source.push_str(&self.emit_prover_sumcheck_driver_constants()?);
        source.push_str(&self.emit_tail_constants());
        source.push_str(&emit_struct_const(
            "STAGE3_PROGRAM",
            "Stage3CpuProgramPlan",
            &[
                ("params", "STAGE3_PARAMS"),
                ("steps", "STAGE3_PROGRAM_STEPS"),
                ("transcript_squeezes", "STAGE3_TRANSCRIPT_SQUEEZES"),
                ("opening_inputs", "STAGE3_OPENING_INPUTS"),
                ("field_constants", "STAGE3_FIELD_CONSTANTS"),
                ("field_exprs", "STAGE3_FIELD_EXPRS"),
                ("kernels", "STAGE3_KERNELS"),
                ("claims", "STAGE3_SUMCHECK_CLAIMS"),
                ("batches", "STAGE3_SUMCHECK_BATCHES"),
                ("drivers", "STAGE3_SUMCHECK_DRIVERS"),
                ("instance_results", "STAGE3_SUMCHECK_INSTANCE_RESULTS"),
                ("evals", "STAGE3_SUMCHECK_EVALS"),
                ("point_slices", "STAGE3_POINT_SLICES"),
                ("point_concats", "STAGE3_POINT_CONCATS"),
                ("opening_claims", "STAGE3_OPENING_CLAIMS"),
                ("opening_equalities", "STAGE3_OPENING_EQUALITIES"),
                ("opening_batches", "STAGE3_OPENING_BATCHES"),
            ],
        ));
        Ok(source)
    }

    pub(super) fn emit_verifier_constants(&self) -> Result<String, EmitError> {
        let mut source = self.emit_shared_constants();
        source.push_str(&self.emit_verifier_sumcheck_claim_constants()?);
        source.push_str(&self.emit_sumcheck_batch_constants());
        source.push_str(&self.emit_verifier_sumcheck_driver_constants()?);
        source.push_str(&self.emit_tail_constants());
        source.push_str(&emit_struct_const(
            "STAGE3_PROGRAM",
            "Stage3VerifierProgramPlan",
            &[
                ("params", "STAGE3_PARAMS"),
                ("steps", "STAGE3_PROGRAM_STEPS"),
                ("transcript_squeezes", "STAGE3_TRANSCRIPT_SQUEEZES"),
                ("opening_inputs", "STAGE3_OPENING_INPUTS"),
                ("field_constants", "STAGE3_FIELD_CONSTANTS"),
                ("field_exprs", "STAGE3_FIELD_EXPRS"),
                ("claims", "STAGE3_SUMCHECK_CLAIMS"),
                ("batches", "STAGE3_SUMCHECK_BATCHES"),
                ("drivers", "STAGE3_SUMCHECK_DRIVERS"),
                ("instance_results", "STAGE3_SUMCHECK_INSTANCE_RESULTS"),
                ("evals", "STAGE3_SUMCHECK_EVALS"),
                ("point_slices", "STAGE3_POINT_SLICES"),
                ("point_concats", "STAGE3_POINT_CONCATS"),
                ("opening_claims", "STAGE3_OPENING_CLAIMS"),
                ("opening_equalities", "STAGE3_OPENING_EQUALITIES"),
                ("opening_batches", "STAGE3_OPENING_BATCHES"),
            ],
        ));
        Ok(source)
    }

    fn emit_shared_constants(&self) -> String {
        let mut source = emit_params_const(
            "STAGE3_PARAMS",
            "Stage3Params",
            &self.params.field,
            &self.params.pcs,
            &self.params.transcript,
        );
        source.push_str(&self.emit_program_step_constants());
        source.push_str(&self.emit_transcript_squeeze_constants());
        source.push_str(&self.emit_opening_input_constants());
        source.push_str(&self.emit_field_constant_constants());
        source.push_str(&self.emit_field_expr_constants());
        source
    }

    fn emit_program_step_constants(&self) -> String {
        emit_plan_array(
            "STAGE3_PROGRAM_STEPS",
            "Stage3ProgramStepPlan",
            self.steps.iter().map(|step| {
                format!(
                    "    Stage3ProgramStepPlan {{ kind: {}, symbol: {} }},",
                    rust_str(&step.kind),
                    rust_str(&step.symbol),
                )
            }),
        )
    }

    fn emit_transcript_squeeze_constants(&self) -> String {
        emit_plan_array(
            "STAGE3_TRANSCRIPT_SQUEEZES",
            "Stage3TranscriptSqueezePlan",
            self.transcript_squeezes.iter().map(|squeeze| {
                format!(
                    "    Stage3TranscriptSqueezePlan {{ symbol: {}, label: {}, kind: {}, count: {} }},",
                    rust_str(&squeeze.symbol),
                    rust_str(&squeeze.label),
                    rust_str(&squeeze.kind),
                    squeeze.count,
                )
            }),
        )
    }

    fn emit_opening_input_constants(&self) -> String {
        emit_plan_array(
            "STAGE3_OPENING_INPUTS",
            "Stage3OpeningInputPlan",
            self.opening_inputs.iter().map(|input| {
                format!(
                    "    Stage3OpeningInputPlan {{ symbol: {}, source_stage: {}, source_claim: {}, oracle: {}, domain: {}, point_arity: {}, claim_kind: {} }},",
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
            "STAGE3_FIELD_CONSTANTS",
            "Stage3FieldConstantPlan",
            self.field_constants.iter().map(|constant| {
                format!(
                    "    Stage3FieldConstantPlan {{ symbol: {}, field: {}, value: {} }},",
                    rust_str(&constant.symbol),
                    rust_str(&constant.field),
                    constant.value
                )
            }),
        )
    }

    fn emit_field_expr_constants(&self) -> String {
        if self.role == Role::Verifier {
            return emit_plan_array_compact(
                "STAGE3_FIELD_EXPRS",
                "Stage3FieldExprPlan",
                self.field_exprs.iter().map(|expr| {
                    format!(
                        "    Stage3FieldExprPlan {{ symbol: {}, kind: {}, formula: {}, operands: {} }},",
                        rust_str(&expr.symbol),
                        rust_str(&expr.kind),
                        rust_str(&expr.formula),
                        rust_str(&expr.operands.join("|"))
                    )
                }),
            );
        }

        let mut source = String::new();
        let mut arrays = Vec::new();
        let mut array_refs = Vec::new();
        for (index, expr) in self.field_exprs.iter().enumerate() {
            let operands = intern_str_array(
                &mut source,
                &mut arrays,
                "STAGE3_FIELD_EXPR_OPERANDS",
                &expr.operands,
            );
            let operand_names = intern_str_array(
                &mut source,
                &mut arrays,
                "STAGE3_FIELD_EXPR_OPERANDS",
                &expr.operand_names,
            );
            array_refs.push((index, operand_names, operands));
        }
        source.push_str(&emit_plan_array_compact(
            "STAGE3_FIELD_EXPRS",
            "Stage3FieldExprPlan",
            self.field_exprs.iter().enumerate().map(|(index, expr)| {
                let (_, operand_names, operands) = &array_refs[index];
                format!(
                    "    Stage3FieldExprPlan {{ symbol: {}, kind: {}, formula: {}, operand_names: {operand_names}, operands: {operands} }},",
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
            "STAGE3_KERNELS",
            "Stage3KernelPlan",
            self.kernels.iter().map(|kernel| {
                format!(
                    "    Stage3KernelPlan {{ symbol: {}, relation: {}, kind: {}, backend: {}, abi: {} }},",
                    rust_str(&kernel.symbol),
                    rust_str(&kernel.relation),
                    rust_str(&kernel.kind),
                    rust_str(&kernel.backend),
                    rust_str(&kernel.abi)
                )
            }),
        )
    }

    fn emit_prover_sumcheck_claim_constants(&self) -> Result<String, EmitError> {
        self.emit_sumcheck_claim_constants(Role::Prover)
    }

    fn emit_verifier_sumcheck_claim_constants(&self) -> Result<String, EmitError> {
        self.emit_sumcheck_claim_constants(Role::Verifier)
    }

    fn emit_sumcheck_claim_constants(&self, role: Role) -> Result<String, EmitError> {
        let mut source = String::new();
        if role == Role::Prover {
            for (index, claim) in self.claims.iter().enumerate() {
                source.push_str(&emit_str_array(
                    &format!("STAGE3_SUMCHECK_CLAIM_{index}_INPUT_OPENINGS"),
                    &claim.input_openings,
                ));
            }
        }
        let mut claims = Vec::new();
        for (index, claim) in self.claims.iter().enumerate() {
            match role {
                Role::Prover => {
                    let kernel = claim.kernel.as_deref().ok_or_else(|| {
                        missing_role_binding("prover claim kernel", &claim.symbol)
                    })?;
                    claims.push(format!(
                        "    Stage3SumcheckClaimPlan {{ symbol: {}, stage: {}, domain: {}, num_rounds: {}, degree: {}, claim: {}, kernel: Some({}), relation: None, claim_value: {}, input_openings: STAGE3_SUMCHECK_CLAIM_{index}_INPUT_OPENINGS }},",
                        rust_str(&claim.symbol),
                        rust_str(&claim.stage),
                        rust_str(&claim.domain),
                        claim.num_rounds,
                        claim.degree,
                        rust_str(&claim.claim),
                        rust_str(kernel),
                        rust_str(&claim.claim_value)
                    ));
                }
                Role::Verifier => {
                    let relation = claim.relation.as_deref().ok_or_else(|| {
                        missing_role_binding("verifier claim relation", &claim.symbol)
                    })?;
                    claims.push(format!(
                        "    Stage3SumcheckClaimPlan {{ symbol: {}, stage: {}, domain: {}, num_rounds: {}, degree: {}, claim: {}, kernel: None, relation: Some({}), claim_value: {}, input_openings: {} }},",
                        rust_str(&claim.symbol),
                        rust_str(&claim.stage),
                        rust_str(&claim.domain),
                        claim.num_rounds,
                        claim.degree,
                        rust_str(&claim.claim),
                        rust_str(relation),
                        rust_str(&claim.claim_value),
                        rust_str(&claim.input_openings.join("|"))
                    ));
                }
            }
        }
        source.push_str(&emit_plan_array_compact(
            "STAGE3_SUMCHECK_CLAIMS",
            "Stage3SumcheckClaimPlan",
            claims,
        ));
        Ok(source)
    }

    fn emit_sumcheck_batch_constants(&self) -> String {
        if self.role == Role::Verifier {
            let mut source = String::new();
            for (index, batch) in self.batches.iter().enumerate() {
                source.push_str(&emit_usize_array(
                    &format!("STAGE3_SUMCHECK_BATCH_{index}_ROUND_SCHEDULE"),
                    &batch.round_schedule,
                ));
            }
            source.push_str(&emit_plan_array_compact(
                "STAGE3_SUMCHECK_BATCHES",
                "Stage3SumcheckBatchPlan",
                self.batches.iter().enumerate().map(|(index, batch)| {
                    format!(
                        "    Stage3SumcheckBatchPlan {{ symbol: {}, stage: {}, proof_slot: {}, policy: {}, count: {}, ordered_claims: {}, claim_operands: {}, claim_label: {}, round_label: {}, round_schedule: STAGE3_SUMCHECK_BATCH_{index}_ROUND_SCHEDULE }},",
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
                &format!("STAGE3_SUMCHECK_BATCH_{index}_ORDERED_CLAIMS"),
                &batch.ordered_claims,
            ));
            source.push_str(&emit_str_array(
                &format!("STAGE3_SUMCHECK_BATCH_{index}_CLAIM_OPERANDS"),
                &batch.claim_operands,
            ));
            source.push_str(&emit_usize_array(
                &format!("STAGE3_SUMCHECK_BATCH_{index}_ROUND_SCHEDULE"),
                &batch.round_schedule,
            ));
        }
        source.push_str(&emit_plan_array_compact(
            "STAGE3_SUMCHECK_BATCHES",
            "Stage3SumcheckBatchPlan",
            self.batches.iter().enumerate().map(|(index, batch)| {
                format!(
                    "    Stage3SumcheckBatchPlan {{ symbol: {}, stage: {}, proof_slot: {}, policy: {}, count: {}, ordered_claims: STAGE3_SUMCHECK_BATCH_{index}_ORDERED_CLAIMS, claim_operands: STAGE3_SUMCHECK_BATCH_{index}_CLAIM_OPERANDS, claim_label: {}, round_label: {}, round_schedule: STAGE3_SUMCHECK_BATCH_{index}_ROUND_SCHEDULE }},",
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

    fn emit_prover_sumcheck_driver_constants(&self) -> Result<String, EmitError> {
        self.emit_sumcheck_driver_constants(Role::Prover)
    }

    fn emit_verifier_sumcheck_driver_constants(&self) -> Result<String, EmitError> {
        self.emit_sumcheck_driver_constants(Role::Verifier)
    }

    fn emit_sumcheck_driver_constants(&self, role: Role) -> Result<String, EmitError> {
        let mut source = String::new();
        for (index, driver) in self.drivers.iter().enumerate() {
            source.push_str(&emit_usize_array(
                &format!("STAGE3_SUMCHECK_DRIVER_{index}_ROUND_SCHEDULE"),
                &driver.round_schedule,
            ));
        }
        let mut drivers = Vec::new();
        for (index, driver) in self.drivers.iter().enumerate() {
            match role {
                Role::Prover => {
                    let kernel = driver.kernel.as_deref().ok_or_else(|| {
                        missing_role_binding("prover driver kernel", &driver.symbol)
                    })?;
                    drivers.push(format!(
                        "    Stage3SumcheckDriverPlan {{ symbol: {}, stage: {}, proof_slot: {}, kernel: Some({}), relation: None, batch: {}, policy: {}, round_schedule: STAGE3_SUMCHECK_DRIVER_{index}_ROUND_SCHEDULE, claim_label: {}, round_label: {}, num_rounds: {}, degree: {} }},",
                        rust_str(&driver.symbol),
                        rust_str(&driver.stage),
                        rust_str(&driver.proof_slot),
                        rust_str(kernel),
                        rust_str(&driver.batch),
                        rust_str(&driver.policy),
                        rust_str(&driver.claim_label),
                        rust_str(&driver.round_label),
                        driver.num_rounds,
                        driver.degree
                    ));
                }
                Role::Verifier => {
                    let relation = driver.relation.as_deref().ok_or_else(|| {
                        missing_role_binding("verifier driver relation", &driver.symbol)
                    })?;
                    drivers.push(format!(
                        "    Stage3SumcheckDriverPlan {{ symbol: {}, stage: {}, proof_slot: {}, kernel: None, relation: Some({}), batch: {}, policy: {}, round_schedule: STAGE3_SUMCHECK_DRIVER_{index}_ROUND_SCHEDULE, claim_label: {}, round_label: {}, num_rounds: {}, degree: {} }},",
                        rust_str(&driver.symbol),
                        rust_str(&driver.stage),
                        rust_str(&driver.proof_slot),
                        rust_str(relation),
                        rust_str(&driver.batch),
                        rust_str(&driver.policy),
                        rust_str(&driver.claim_label),
                        rust_str(&driver.round_label),
                        driver.num_rounds,
                        driver.degree
                    ));
                }
            }
        }
        source.push_str(&emit_plan_array_compact(
            "STAGE3_SUMCHECK_DRIVERS",
            "Stage3SumcheckDriverPlan",
            drivers,
        ));
        Ok(source)
    }

    fn emit_tail_constants(&self) -> String {
        let mut source = String::new();
        source.push_str(&self.emit_sumcheck_instance_result_constants());
        source.push_str(&self.emit_sumcheck_eval_constants());
        source.push_str(&self.emit_point_slice_constants());
        source.push_str(&self.emit_point_concat_constants());
        source.push_str(&self.emit_opening_claim_constants());
        source.push_str(&self.emit_opening_claim_equality_constants());
        source.push_str(&self.emit_opening_batch_constants());
        source
    }

    fn emit_sumcheck_instance_result_constants(&self) -> String {
        emit_plan_array(
            "STAGE3_SUMCHECK_INSTANCE_RESULTS",
            "Stage3SumcheckInstanceResultPlan",
            self.instance_results.iter().map(|instance| {
                format!(
                    "    Stage3SumcheckInstanceResultPlan {{ symbol: {}, source: {}, claim: {}, relation: {}, index: {}, point_arity: {}, num_rounds: {}, round_offset: {}, point_order: {}, degree: {} }},",
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
        emit_plan_array(
            "STAGE3_SUMCHECK_EVALS",
            "Stage3SumcheckEvalPlan",
            self.evals.iter().map(|eval| {
                format!(
                    "    Stage3SumcheckEvalPlan {{ symbol: {}, source: {}, name: {}, index: {}, oracle: {} }},",
                    rust_str(&eval.symbol),
                    rust_str(&eval.source),
                    rust_str(&eval.name),
                    eval.index,
                    rust_str(&eval.oracle)
                )
            }),
        )
    }

    fn emit_point_slice_constants(&self) -> String {
        emit_plan_array(
            "STAGE3_POINT_SLICES",
            "Stage3PointSlicePlan",
            self.point_slices.iter().map(|slice| {
                format!(
                    "    Stage3PointSlicePlan {{ symbol: {}, source: {}, offset: {}, length: {}, input: {} }},",
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
                "STAGE3_POINT_CONCATS",
                "Stage3PointConcatPlan",
                self.point_concats.iter().map(|concat| {
                    format!(
                        "    Stage3PointConcatPlan {{ symbol: {}, layout: {}, arity: {}, inputs: {} }},",
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
                &format!("STAGE3_POINT_CONCAT_{index}_INPUTS"),
                &concat.inputs,
            ));
        }
        source.push_str(&emit_plan_array_compact(
            "STAGE3_POINT_CONCATS",
            "Stage3PointConcatPlan",
            self.point_concats.iter().enumerate().map(|(index, concat)| {
                format!(
                    "    Stage3PointConcatPlan {{ symbol: {}, layout: {}, arity: {}, inputs: STAGE3_POINT_CONCAT_{index}_INPUTS }},",
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
            "STAGE3_OPENING_CLAIMS",
            "Stage3OpeningClaimPlan",
            self.opening_claims.iter().map(|claim| {
                format!(
                    "    Stage3OpeningClaimPlan {{ symbol: {}, oracle: {}, domain: {}, point_arity: {}, claim_kind: {}, point_source: {}, eval_source: {} }},",
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
            "STAGE3_OPENING_EQUALITIES",
            "Stage3OpeningClaimEqualityPlan",
            self.opening_equalities.iter().map(|equality| {
                format!(
                    "    Stage3OpeningClaimEqualityPlan {{ symbol: {}, mode: {}, lhs: {}, rhs: {} }},",
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
                "STAGE3_OPENING_BATCHES",
                "Stage3OpeningBatchPlan",
                self.opening_batches.iter().map(|batch| {
                    format!(
                        "    Stage3OpeningBatchPlan {{ symbol: {}, stage: {}, proof_slot: {}, policy: {}, count: {}, ordered_claims: {}, claim_operands: {} }},",
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
                &format!("STAGE3_OPENING_BATCH_{index}_ORDERED_CLAIMS"),
                &batch.ordered_claims,
            ));
            source.push_str(&emit_str_array(
                &format!("STAGE3_OPENING_BATCH_{index}_CLAIM_OPERANDS"),
                &batch.claim_operands,
            ));
        }
        source.push_str(&emit_plan_array_compact(
            "STAGE3_OPENING_BATCHES",
            "Stage3OpeningBatchPlan",
            self.opening_batches.iter().enumerate().map(|(index, batch)| {
                format!(
                    "    Stage3OpeningBatchPlan {{ symbol: {}, stage: {}, proof_slot: {}, policy: {}, count: {}, ordered_claims: STAGE3_OPENING_BATCH_{index}_ORDERED_CLAIMS, claim_operands: STAGE3_OPENING_BATCH_{index}_CLAIM_OPERANDS }},",
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
