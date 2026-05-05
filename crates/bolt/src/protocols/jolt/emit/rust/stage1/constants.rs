use super::Stage1CpuProgram;
use crate::emit::rust::EmitError;
use crate::ir::Role;
use crate::protocols::jolt::emit::rust::checks::missing_role_binding;
use crate::protocols::jolt::emit::rust::source::{
    emit_params_const, emit_plan_array, emit_plan_array_compact, emit_str_array, emit_struct_const,
    emit_usize_array, rust_str,
};

impl Stage1CpuProgram {
    pub(super) fn emit_prover_constants(&self) -> Result<String, EmitError> {
        let mut source = emit_params_const(
            "STAGE1_PARAMS",
            "Stage1Params",
            &self.params.field,
            &self.params.pcs,
            &self.params.transcript,
        );

        source.push_str(&self.emit_transcript_squeeze_constants());
        source.push_str(&self.emit_kernel_constants());
        source.push_str(&self.emit_sumcheck_claim_constants()?);
        source.push_str(&self.emit_sumcheck_batch_constants());
        source.push_str(&self.emit_sumcheck_driver_constants()?);
        source.push_str(&self.emit_sumcheck_instance_result_constants());
        source.push_str(&self.emit_sumcheck_eval_constants());
        source.push_str(&self.emit_opening_claim_constants());
        source.push_str(&self.emit_opening_batch_constants());
        source.push_str(&emit_struct_const(
            "STAGE1_PROGRAM",
            "Stage1CpuProgramPlan",
            &[
                ("params", "STAGE1_PARAMS"),
                ("transcript_squeezes", "STAGE1_TRANSCRIPT_SQUEEZES"),
                ("kernels", "STAGE1_KERNELS"),
                ("claims", "STAGE1_SUMCHECK_CLAIMS"),
                ("batches", "STAGE1_SUMCHECK_BATCHES"),
                ("drivers", "STAGE1_SUMCHECK_DRIVERS"),
                ("instance_results", "STAGE1_SUMCHECK_INSTANCE_RESULTS"),
                ("evals", "STAGE1_SUMCHECK_EVALS"),
                ("opening_claims", "STAGE1_OPENING_CLAIMS"),
                ("opening_batches", "STAGE1_OPENING_BATCHES"),
            ],
        ));
        Ok(source)
    }

    fn emit_sumcheck_instance_result_constants(&self) -> String {
        emit_plan_array(
            "STAGE1_SUMCHECK_INSTANCE_RESULTS",
            "Stage1SumcheckInstanceResultPlan",
            self.instance_results.iter().map(|instance| {
                format!(
                    "    Stage1SumcheckInstanceResultPlan {{ symbol: {}, source: {}, claim: {}, relation: {}, index: {}, point_arity: {}, num_rounds: {}, round_offset: {}, point_order: {}, degree: {} }},",
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

    fn emit_transcript_squeeze_constants(&self) -> String {
        emit_plan_array(
            "STAGE1_TRANSCRIPT_SQUEEZES",
            "Stage1TranscriptSqueezePlan",
            self.transcript_squeezes.iter().map(|squeeze| {
                format!(
                    "    Stage1TranscriptSqueezePlan {{ symbol: {}, label: {}, kind: {}, count: {} }},",
                    rust_str(&squeeze.symbol),
                    rust_str(&squeeze.label),
                    rust_str(&squeeze.kind),
                    squeeze.count,
                )
            }),
        )
    }

    fn emit_kernel_constants(&self) -> String {
        emit_plan_array(
            "STAGE1_KERNELS",
            "Stage1KernelPlan",
            self.kernels.iter().map(|kernel| {
                format!(
                    "    Stage1KernelPlan {{ symbol: {}, relation: {}, kind: {}, backend: {}, abi: {} }},",
                    rust_str(&kernel.symbol),
                    rust_str(&kernel.relation),
                    rust_str(&kernel.kind),
                    rust_str(&kernel.backend),
                    rust_str(&kernel.abi)
                )
            }),
        )
    }

    fn emit_sumcheck_claim_constants(&self) -> Result<String, EmitError> {
        let mut source = String::new();
        for (index, claim) in self.claims.iter().enumerate() {
            source.push_str(&emit_str_array(
                &format!("STAGE1_SUMCHECK_CLAIM_{index}_INPUT_OPENINGS"),
                &claim.input_openings,
            ));
        }
        let mut claims = Vec::new();
        for (index, claim) in self.claims.iter().enumerate() {
            let kernel = claim
                .kernel
                .as_deref()
                .ok_or_else(|| missing_role_binding("prover claim kernel", &claim.symbol))?;
            claims.push(format!(
                    "    Stage1SumcheckClaimPlan {{ symbol: {}, stage: {}, domain: {}, num_rounds: {}, degree: {}, claim: {}, kernel: Some({}), relation: None, claim_value: {}, input_openings: STAGE1_SUMCHECK_CLAIM_{index}_INPUT_OPENINGS }},",
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
        source.push_str(&emit_plan_array_compact(
            "STAGE1_SUMCHECK_CLAIMS",
            "Stage1SumcheckClaimPlan",
            claims,
        ));
        Ok(source)
    }

    fn emit_sumcheck_batch_constants(&self) -> String {
        if self.role == Role::Verifier {
            let mut source = String::new();
            for (index, batch) in self.batches.iter().enumerate() {
                source.push_str(&emit_usize_array(
                    &format!("STAGE1_SUMCHECK_BATCH_{index}_ROUND_SCHEDULE"),
                    &batch.round_schedule,
                ));
            }
            source.push_str(&emit_plan_array_compact(
                "STAGE1_SUMCHECK_BATCHES",
                "Stage1SumcheckBatchPlan",
                self.batches.iter().enumerate().map(|(index, batch)| {
                    format!(
                        "    Stage1SumcheckBatchPlan {{ symbol: {}, stage: {}, proof_slot: {}, policy: {}, count: {}, ordered_claims: {}, claim_operands: {}, claim_label: {}, round_label: {}, round_schedule: STAGE1_SUMCHECK_BATCH_{index}_ROUND_SCHEDULE }},",
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
                &format!("STAGE1_SUMCHECK_BATCH_{index}_ORDERED_CLAIMS"),
                &batch.ordered_claims,
            ));
            source.push_str(&emit_str_array(
                &format!("STAGE1_SUMCHECK_BATCH_{index}_CLAIM_OPERANDS"),
                &batch.claim_operands,
            ));
            source.push_str(&emit_usize_array(
                &format!("STAGE1_SUMCHECK_BATCH_{index}_ROUND_SCHEDULE"),
                &batch.round_schedule,
            ));
        }
        source.push_str(&emit_plan_array_compact(
            "STAGE1_SUMCHECK_BATCHES",
            "Stage1SumcheckBatchPlan",
            self.batches.iter().enumerate().map(|(index, batch)| {
                format!(
                    "    Stage1SumcheckBatchPlan {{ symbol: {}, stage: {}, proof_slot: {}, policy: {}, count: {}, ordered_claims: STAGE1_SUMCHECK_BATCH_{index}_ORDERED_CLAIMS, claim_operands: STAGE1_SUMCHECK_BATCH_{index}_CLAIM_OPERANDS, claim_label: {}, round_label: {}, round_schedule: STAGE1_SUMCHECK_BATCH_{index}_ROUND_SCHEDULE }},",
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

    fn emit_sumcheck_driver_constants(&self) -> Result<String, EmitError> {
        let mut source = String::new();
        for (index, driver) in self.drivers.iter().enumerate() {
            source.push_str(&emit_usize_array(
                &format!("STAGE1_SUMCHECK_DRIVER_{index}_ROUND_SCHEDULE"),
                &driver.round_schedule,
            ));
        }
        let mut drivers = Vec::new();
        for (index, driver) in self.drivers.iter().enumerate() {
            let kernel = driver
                .kernel
                .as_deref()
                .ok_or_else(|| missing_role_binding("prover driver kernel", &driver.symbol))?;
            drivers.push(format!(
                    "    Stage1SumcheckDriverPlan {{ symbol: {}, stage: {}, proof_slot: {}, kernel: Some({}), relation: None, batch: {}, policy: {}, round_schedule: STAGE1_SUMCHECK_DRIVER_{index}_ROUND_SCHEDULE, claim_label: {}, round_label: {}, num_rounds: {}, degree: {} }},",
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
        source.push_str(&emit_plan_array_compact(
            "STAGE1_SUMCHECK_DRIVERS",
            "Stage1SumcheckDriverPlan",
            drivers,
        ));
        Ok(source)
    }

    fn emit_sumcheck_eval_constants(&self) -> String {
        emit_plan_array(
            "STAGE1_SUMCHECK_EVALS",
            "Stage1SumcheckEvalPlan",
            self.evals.iter().map(|eval| {
                format!(
                    "    Stage1SumcheckEvalPlan {{ symbol: {}, source: {}, name: {}, index: {}, oracle: {} }},",
                    rust_str(&eval.symbol),
                    rust_str(&eval.source),
                    rust_str(&eval.name),
                    eval.index,
                    rust_str(&eval.oracle)
                )
            }),
        )
    }

    fn emit_opening_claim_constants(&self) -> String {
        emit_plan_array(
            "STAGE1_OPENING_CLAIMS",
            "Stage1OpeningClaimPlan",
            self.opening_claims.iter().map(|claim| {
                format!(
                    "    Stage1OpeningClaimPlan {{ symbol: {}, oracle: {}, domain: {}, point_arity: {}, claim_kind: {}, point_source: {}, eval_source: {} }},",
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

    fn emit_opening_batch_constants(&self) -> String {
        if self.role == Role::Verifier {
            return emit_plan_array_compact(
                "STAGE1_OPENING_BATCHES",
                "Stage1OpeningBatchPlan",
                self.opening_batches.iter().map(|batch| {
                    format!(
                        "    Stage1OpeningBatchPlan {{ symbol: {}, stage: {}, proof_slot: {}, policy: {}, count: {}, ordered_claims: {}, claim_operands: {} }},",
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
                &format!("STAGE1_OPENING_BATCH_{index}_ORDERED_CLAIMS"),
                &batch.ordered_claims,
            ));
            source.push_str(&emit_str_array(
                &format!("STAGE1_OPENING_BATCH_{index}_CLAIM_OPERANDS"),
                &batch.claim_operands,
            ));
        }
        source.push_str(&emit_plan_array_compact(
            "STAGE1_OPENING_BATCHES",
            "Stage1OpeningBatchPlan",
            self.opening_batches.iter().enumerate().map(|(index, batch)| {
                format!(
                    "    Stage1OpeningBatchPlan {{ symbol: {}, stage: {}, proof_slot: {}, policy: {}, count: {}, ordered_claims: STAGE1_OPENING_BATCH_{index}_ORDERED_CLAIMS, claim_operands: STAGE1_OPENING_BATCH_{index}_CLAIM_OPERANDS }},",
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

    pub(super) fn emit_verifier_constants(&self) -> Result<String, EmitError> {
        let mut source = emit_params_const(
            "STAGE1_PARAMS",
            "Stage1Params",
            &self.params.field,
            &self.params.pcs,
            &self.params.transcript,
        );

        source.push_str(&self.emit_transcript_squeeze_constants());
        source.push_str(&self.emit_verifier_sumcheck_claim_constants()?);
        source.push_str(&self.emit_sumcheck_batch_constants());
        source.push_str(&self.emit_verifier_sumcheck_driver_constants()?);
        source.push_str(&self.emit_sumcheck_instance_result_constants());
        source.push_str(&self.emit_sumcheck_eval_constants());
        source.push_str(&self.emit_opening_claim_constants());
        source.push_str(&self.emit_opening_batch_constants());
        source.push_str(&emit_struct_const(
            "STAGE1_PROGRAM",
            "Stage1VerifierProgramPlan",
            &[
                ("params", "STAGE1_PARAMS"),
                ("transcript_squeezes", "STAGE1_TRANSCRIPT_SQUEEZES"),
                ("claims", "STAGE1_SUMCHECK_CLAIMS"),
                ("batches", "STAGE1_SUMCHECK_BATCHES"),
                ("drivers", "STAGE1_SUMCHECK_DRIVERS"),
                ("instance_results", "STAGE1_SUMCHECK_INSTANCE_RESULTS"),
                ("evals", "STAGE1_SUMCHECK_EVALS"),
                ("opening_claims", "STAGE1_OPENING_CLAIMS"),
                ("opening_batches", "STAGE1_OPENING_BATCHES"),
            ],
        ));
        Ok(source)
    }

    fn emit_verifier_sumcheck_claim_constants(&self) -> Result<String, EmitError> {
        let mut claims = Vec::new();
        for claim in &self.claims {
            let relation = claim
                .relation
                .as_deref()
                .ok_or_else(|| missing_role_binding("verifier claim relation", &claim.symbol))?;
            claims.push(format!(
                    "    Stage1SumcheckClaimPlan {{ symbol: {}, stage: {}, domain: {}, num_rounds: {}, degree: {}, claim: {}, kernel: None, relation: Some({}), claim_value: {}, input_openings: {} }},",
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
        Ok(emit_plan_array_compact(
            "STAGE1_SUMCHECK_CLAIMS",
            "Stage1SumcheckClaimPlan",
            claims,
        ))
    }

    fn emit_verifier_sumcheck_driver_constants(&self) -> Result<String, EmitError> {
        let mut source = String::new();
        for (index, driver) in self.drivers.iter().enumerate() {
            source.push_str(&emit_usize_array(
                &format!("STAGE1_SUMCHECK_DRIVER_{index}_ROUND_SCHEDULE"),
                &driver.round_schedule,
            ));
        }
        let mut drivers = Vec::new();
        for (index, driver) in self.drivers.iter().enumerate() {
            let relation = driver
                .relation
                .as_deref()
                .ok_or_else(|| missing_role_binding("verifier driver relation", &driver.symbol))?;
            drivers.push(format!(
                    "    Stage1SumcheckDriverPlan {{ symbol: {}, stage: {}, proof_slot: {}, kernel: None, relation: Some({}), batch: {}, policy: {}, round_schedule: STAGE1_SUMCHECK_DRIVER_{index}_ROUND_SCHEDULE, claim_label: {}, round_label: {}, num_rounds: {}, degree: {} }},",
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
        source.push_str(&emit_plan_array_compact(
            "STAGE1_SUMCHECK_DRIVERS",
            "Stage1SumcheckDriverPlan",
            drivers,
        ));
        Ok(source)
    }
}
