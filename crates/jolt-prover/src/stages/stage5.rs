#![allow(dead_code)]

use jolt_field::Fr;
use jolt_kernels::stage5::{execute_stage5_program, Stage5CpuProgramPlan, Stage5ExecutionArtifacts, Stage5ExecutionMode, Stage5FieldConstantPlan, Stage5FieldExprPlan, Stage5KernelError, Stage5KernelExecutor, Stage5KernelPlan, Stage5OpeningBatchPlan, Stage5OpeningClaimEqualityPlan, Stage5OpeningClaimPlan, Stage5OpeningInputPlan, Stage5Params, Stage5PointConcatPlan, Stage5PointSlicePlan, Stage5ProgramStepPlan, Stage5SumcheckBatchPlan, Stage5SumcheckClaimPlan, Stage5SumcheckDriverPlan, Stage5SumcheckEvalPlan, Stage5SumcheckInstanceResultPlan, Stage5TranscriptAbsorbBytesPlan, Stage5TranscriptSqueezePlan};
use jolt_transcript::{Blake2bTranscript, Transcript};

pub type DefaultStage5Transcript = Blake2bTranscript<Fr>;

pub const STAGE5_PARAMS: Stage5Params = Stage5Params {
    field: "bn254_fr",
    pcs: "dory",
    transcript: "blake2b_transcript",
};
pub const STAGE5_PROGRAM_STEPS: &[Stage5ProgramStepPlan] = &[
    Stage5ProgramStepPlan { kind: "transcript_squeeze", symbol: "stage5.instruction_read_raf.gamma" },
    Stage5ProgramStepPlan { kind: "transcript_squeeze", symbol: "stage5.ram_ra_claim_reduction.gamma" },
    Stage5ProgramStepPlan { kind: "sumcheck_driver", symbol: "stage5.sumcheck" },
];

pub const STAGE5_TRANSCRIPT_SQUEEZES: &[Stage5TranscriptSqueezePlan] = &[
    Stage5TranscriptSqueezePlan { symbol: "stage5.instruction_read_raf.gamma", label: "instruction_read_raf_gamma", kind: "challenge_scalar", count: 1 },
    Stage5TranscriptSqueezePlan { symbol: "stage5.ram_ra_claim_reduction.gamma", label: "ram_ra_claim_reduction_gamma", kind: "challenge_scalar", count: 1 },
];

pub const STAGE5_TRANSCRIPT_ABSORB_BYTES: &[Stage5TranscriptAbsorbBytesPlan] = &[

];

pub const STAGE5_OPENING_INPUTS: &[Stage5OpeningInputPlan] = &[
    Stage5OpeningInputPlan { symbol: "stage5.input.stage2.instruction.LookupOutput", source_stage: "stage2", source_claim: "stage2.instruction_lookup.claim_reduction.opening.LookupOutput", oracle: "LookupOutput", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage5OpeningInputPlan { symbol: "stage5.input.stage2.product_virtual.LookupOutput", source_stage: "stage2", source_claim: "stage2.product_virtual.remainder.opening.LookupOutput", oracle: "LookupOutput", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage5OpeningInputPlan { symbol: "stage5.input.stage2.instruction.LeftLookupOperand", source_stage: "stage2", source_claim: "stage2.instruction_lookup.claim_reduction.opening.LeftLookupOperand", oracle: "LeftLookupOperand", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage5OpeningInputPlan { symbol: "stage5.input.stage2.instruction.RightLookupOperand", source_stage: "stage2", source_claim: "stage2.instruction_lookup.claim_reduction.opening.RightLookupOperand", oracle: "RightLookupOperand", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage5OpeningInputPlan { symbol: "stage5.input.stage2.ram_raf.RamRa", source_stage: "stage2", source_claim: "stage2.ram_raf.opening.RamRa", oracle: "RamRa", domain: "jolt.stage2_ram_rw_domain", point_arity: 32, claim_kind: "virtual" },
    Stage5OpeningInputPlan { symbol: "stage5.input.stage2.ram_read_write.RamRa", source_stage: "stage2", source_claim: "stage2.ram_read_write.opening.RamRa", oracle: "RamRa", domain: "jolt.stage2_ram_rw_domain", point_arity: 32, claim_kind: "virtual" },
    Stage5OpeningInputPlan { symbol: "stage5.input.stage4.ram_val_check.RamRa", source_stage: "stage4", source_claim: "stage4.ram_val_check.opening.RamRa", oracle: "RamRa", domain: "jolt.stage2_ram_rw_domain", point_arity: 32, claim_kind: "virtual" },
    Stage5OpeningInputPlan { symbol: "stage5.input.stage4.registers.RegistersVal", source_stage: "stage4", source_claim: "stage4.registers_read_write.opening.RegistersVal", oracle: "RegistersVal", domain: "jolt.stage4_registers_rw_domain", point_arity: 23, claim_kind: "virtual" },
];

pub const STAGE5_FIELD_CONSTANTS: &[Stage5FieldConstantPlan] = &[

];

pub const STAGE5_FIELD_EXPR_OPERANDS_0: &[&str] = &["stage5.instruction_read_raf.gamma"];

pub const STAGE5_FIELD_EXPR_OPERANDS_1: &[&str] = &[
    "stage5.instruction_read_raf.gamma",
    "stage5.input.stage2.instruction.LeftLookupOperand",
];

pub const STAGE5_FIELD_EXPR_OPERANDS_2: &[&str] = &[
    "stage5.instruction_read_raf.gamma2",
    "stage5.input.stage2.instruction.RightLookupOperand",
];

pub const STAGE5_FIELD_EXPR_OPERANDS_3: &[&str] = &[
    "stage5.input.stage2.instruction.LookupOutput",
    "stage5.instruction_read_raf.term.LeftLookupOperand",
];

pub const STAGE5_FIELD_EXPR_OPERANDS_4: &[&str] = &[
    "stage5.instruction_read_raf.partial.LookupOutputLeftOperand",
    "stage5.instruction_read_raf.term.RightLookupOperand",
];

pub const STAGE5_FIELD_EXPR_OPERANDS_5: &[&str] = &["stage5.ram_ra_claim_reduction.gamma"];

pub const STAGE5_FIELD_EXPR_OPERANDS_6: &[&str] = &[
    "stage5.ram_ra_claim_reduction.gamma",
    "stage5.input.stage2.ram_read_write.RamRa",
];

pub const STAGE5_FIELD_EXPR_OPERANDS_7: &[&str] = &[
    "stage5.ram_ra_claim_reduction.gamma2",
    "stage5.input.stage4.ram_val_check.RamRa",
];

pub const STAGE5_FIELD_EXPR_OPERANDS_8: &[&str] = &[
    "stage5.input.stage2.ram_raf.RamRa",
    "stage5.ram_ra_claim_reduction.term.RamRaReadWrite",
];

pub const STAGE5_FIELD_EXPR_OPERANDS_9: &[&str] = &[
    "stage5.ram_ra_claim_reduction.partial.RafReadWrite",
    "stage5.ram_ra_claim_reduction.term.RamRaValCheck",
];

pub const STAGE5_FIELD_EXPRS: &[Stage5FieldExprPlan] = &[
    Stage5FieldExprPlan { symbol: "stage5.instruction_read_raf.gamma2", kind: "op", formula: "field.pow:2", operand_names: STAGE5_FIELD_EXPR_OPERANDS_0, operands: STAGE5_FIELD_EXPR_OPERANDS_0 },
    Stage5FieldExprPlan { symbol: "stage5.instruction_read_raf.term.LeftLookupOperand", kind: "op", formula: "field.mul", operand_names: STAGE5_FIELD_EXPR_OPERANDS_1, operands: STAGE5_FIELD_EXPR_OPERANDS_1 },
    Stage5FieldExprPlan { symbol: "stage5.instruction_read_raf.term.RightLookupOperand", kind: "op", formula: "field.mul", operand_names: STAGE5_FIELD_EXPR_OPERANDS_2, operands: STAGE5_FIELD_EXPR_OPERANDS_2 },
    Stage5FieldExprPlan { symbol: "stage5.instruction_read_raf.partial.LookupOutputLeftOperand", kind: "op", formula: "field.add", operand_names: STAGE5_FIELD_EXPR_OPERANDS_3, operands: STAGE5_FIELD_EXPR_OPERANDS_3 },
    Stage5FieldExprPlan { symbol: "stage5.instruction_read_raf.claim_expr", kind: "op", formula: "field.add", operand_names: STAGE5_FIELD_EXPR_OPERANDS_4, operands: STAGE5_FIELD_EXPR_OPERANDS_4 },
    Stage5FieldExprPlan { symbol: "stage5.ram_ra_claim_reduction.gamma2", kind: "op", formula: "field.pow:2", operand_names: STAGE5_FIELD_EXPR_OPERANDS_5, operands: STAGE5_FIELD_EXPR_OPERANDS_5 },
    Stage5FieldExprPlan { symbol: "stage5.ram_ra_claim_reduction.term.RamRaReadWrite", kind: "op", formula: "field.mul", operand_names: STAGE5_FIELD_EXPR_OPERANDS_6, operands: STAGE5_FIELD_EXPR_OPERANDS_6 },
    Stage5FieldExprPlan { symbol: "stage5.ram_ra_claim_reduction.term.RamRaValCheck", kind: "op", formula: "field.mul", operand_names: STAGE5_FIELD_EXPR_OPERANDS_7, operands: STAGE5_FIELD_EXPR_OPERANDS_7 },
    Stage5FieldExprPlan { symbol: "stage5.ram_ra_claim_reduction.partial.RafReadWrite", kind: "op", formula: "field.add", operand_names: STAGE5_FIELD_EXPR_OPERANDS_8, operands: STAGE5_FIELD_EXPR_OPERANDS_8 },
    Stage5FieldExprPlan { symbol: "stage5.ram_ra_claim_reduction.claim_expr", kind: "op", formula: "field.add", operand_names: STAGE5_FIELD_EXPR_OPERANDS_9, operands: STAGE5_FIELD_EXPR_OPERANDS_9 },
];
pub const STAGE5_KERNELS: &[Stage5KernelPlan] = &[
    Stage5KernelPlan { symbol: "jolt.cpu.stage5.instruction_read_raf", relation: "jolt.stage5.instruction_read_raf", kind: "sumcheck", backend: "cpu", abi: "jolt_stage5_instruction_read_raf" },
    Stage5KernelPlan { symbol: "jolt.cpu.stage5.ram_ra_claim_reduction", relation: "jolt.stage5.ram_ra_claim_reduction", kind: "sumcheck", backend: "cpu", abi: "jolt_stage5_ram_ra_claim_reduction" },
    Stage5KernelPlan { symbol: "jolt.cpu.stage5.registers_val_evaluation", relation: "jolt.stage5.registers_val_evaluation", kind: "sumcheck", backend: "cpu", abi: "jolt_stage5_registers_val_evaluation" },
    Stage5KernelPlan { symbol: "jolt.cpu.stage5.batched", relation: "jolt.stage5.batched", kind: "sumcheck", backend: "cpu", abi: "jolt_stage5_batched" },
];

pub const STAGE5_SUMCHECK_CLAIM_0_INPUT_OPENINGS: &[&str] = &[
    "stage5.input.stage2.instruction.LookupOutput",
    "stage5.input.stage2.instruction.LeftLookupOperand",
    "stage5.input.stage2.instruction.RightLookupOperand",
];

pub const STAGE5_SUMCHECK_CLAIM_1_INPUT_OPENINGS: &[&str] = &[
    "stage5.input.stage2.ram_raf.RamRa",
    "stage5.input.stage2.ram_read_write.RamRa",
    "stage5.input.stage4.ram_val_check.RamRa",
];

pub const STAGE5_SUMCHECK_CLAIM_2_INPUT_OPENINGS: &[&str] = &["stage5.input.stage4.registers.RegistersVal"];

pub const STAGE5_SUMCHECK_CLAIMS: &[Stage5SumcheckClaimPlan] = &[
    Stage5SumcheckClaimPlan { symbol: "stage5.instruction_read_raf.input", stage: "stage5", domain: "jolt.stage5_instruction_read_raf_domain", num_rounds: 144, degree: 10, claim: "stage5.instruction_read_raf.weighted_lookup_values", kernel: Some("jolt.cpu.stage5.instruction_read_raf"), relation: None, claim_value: "stage5.instruction_read_raf.claim_expr", input_openings: STAGE5_SUMCHECK_CLAIM_0_INPUT_OPENINGS },
    Stage5SumcheckClaimPlan { symbol: "stage5.ram_ra_claim_reduction.input", stage: "stage5", domain: "jolt.trace_domain", num_rounds: 16, degree: 2, claim: "stage5.ram_ra_claim_reduction.weighted_ram_ra", kernel: Some("jolt.cpu.stage5.ram_ra_claim_reduction"), relation: None, claim_value: "stage5.ram_ra_claim_reduction.claim_expr", input_openings: STAGE5_SUMCHECK_CLAIM_1_INPUT_OPENINGS },
    Stage5SumcheckClaimPlan { symbol: "stage5.registers_val_evaluation.input", stage: "stage5", domain: "jolt.trace_domain", num_rounds: 16, degree: 3, claim: "stage5.registers_val_evaluation.registers_val", kernel: Some("jolt.cpu.stage5.registers_val_evaluation"), relation: None, claim_value: "stage5.input.stage4.registers.RegistersVal", input_openings: STAGE5_SUMCHECK_CLAIM_2_INPUT_OPENINGS },
];
pub const STAGE5_SUMCHECK_BATCH_0_ORDERED_CLAIMS: &[&str] = &[
    "stage5.instruction_read_raf.input",
    "stage5.ram_ra_claim_reduction.input",
    "stage5.registers_val_evaluation.input",
];

pub const STAGE5_SUMCHECK_BATCH_0_CLAIM_OPERANDS: &[&str] = &[
    "stage5.instruction_read_raf.input",
    "stage5.ram_ra_claim_reduction.input",
    "stage5.registers_val_evaluation.input",
];

pub const STAGE5_SUMCHECK_BATCH_0_ROUND_SCHEDULE: &[usize] = &[
    128,
    16,
];

pub const STAGE5_SUMCHECK_BATCHES: &[Stage5SumcheckBatchPlan] = &[
    Stage5SumcheckBatchPlan { symbol: "stage5.batch", stage: "stage5", proof_slot: "stage5.sumcheck", policy: "jolt_core_stage5_aligned", count: 3, ordered_claims: STAGE5_SUMCHECK_BATCH_0_ORDERED_CLAIMS, claim_operands: STAGE5_SUMCHECK_BATCH_0_CLAIM_OPERANDS, claim_label: "sumcheck_claim", round_label: "sumcheck_poly", round_schedule: STAGE5_SUMCHECK_BATCH_0_ROUND_SCHEDULE },
];
pub const STAGE5_SUMCHECK_DRIVER_0_ROUND_SCHEDULE: &[usize] = &[
    128,
    16,
];

pub const STAGE5_SUMCHECK_DRIVERS: &[Stage5SumcheckDriverPlan] = &[
    Stage5SumcheckDriverPlan { symbol: "stage5.sumcheck", stage: "stage5", proof_slot: "stage5.sumcheck", kernel: Some("jolt.cpu.stage5.batched"), relation: None, batch: "stage5.batch", policy: "jolt_core_stage5_aligned", round_schedule: STAGE5_SUMCHECK_DRIVER_0_ROUND_SCHEDULE, claim_label: "sumcheck_claim", round_label: "sumcheck_poly", num_rounds: 144, degree: 10 },
];
pub const STAGE5_SUMCHECK_INSTANCE_RESULTS: &[Stage5SumcheckInstanceResultPlan] = &[
    Stage5SumcheckInstanceResultPlan { symbol: "stage5.instruction_read_raf.instance", source: "stage5.sumcheck", claim: "stage5.instruction_read_raf.input", relation: "jolt.stage5.instruction_read_raf", index: 0, point_arity: 144, num_rounds: 144, round_offset: 0, point_order: "instruction_read_raf", degree: 10 },
    Stage5SumcheckInstanceResultPlan { symbol: "stage5.ram_ra_claim_reduction.instance", source: "stage5.sumcheck", claim: "stage5.ram_ra_claim_reduction.input", relation: "jolt.stage5.ram_ra_claim_reduction", index: 1, point_arity: 16, num_rounds: 16, round_offset: 128, point_order: "reverse", degree: 2 },
    Stage5SumcheckInstanceResultPlan { symbol: "stage5.registers_val_evaluation.instance", source: "stage5.sumcheck", claim: "stage5.registers_val_evaluation.input", relation: "jolt.stage5.registers_val_evaluation", index: 2, point_arity: 16, num_rounds: 16, round_offset: 128, point_order: "reverse", degree: 3 },
];

pub const STAGE5_SUMCHECK_EVALS: &[Stage5SumcheckEvalPlan] = &[
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.LookupTableFlag_0", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.LookupTableFlag_0", index: 0, oracle: "LookupTableFlag_0" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.LookupTableFlag_1", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.LookupTableFlag_1", index: 1, oracle: "LookupTableFlag_1" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.LookupTableFlag_2", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.LookupTableFlag_2", index: 2, oracle: "LookupTableFlag_2" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.LookupTableFlag_3", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.LookupTableFlag_3", index: 3, oracle: "LookupTableFlag_3" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.LookupTableFlag_4", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.LookupTableFlag_4", index: 4, oracle: "LookupTableFlag_4" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.LookupTableFlag_5", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.LookupTableFlag_5", index: 5, oracle: "LookupTableFlag_5" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.LookupTableFlag_6", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.LookupTableFlag_6", index: 6, oracle: "LookupTableFlag_6" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.LookupTableFlag_7", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.LookupTableFlag_7", index: 7, oracle: "LookupTableFlag_7" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.LookupTableFlag_8", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.LookupTableFlag_8", index: 8, oracle: "LookupTableFlag_8" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.LookupTableFlag_9", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.LookupTableFlag_9", index: 9, oracle: "LookupTableFlag_9" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.LookupTableFlag_10", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.LookupTableFlag_10", index: 10, oracle: "LookupTableFlag_10" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.LookupTableFlag_11", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.LookupTableFlag_11", index: 11, oracle: "LookupTableFlag_11" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.LookupTableFlag_12", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.LookupTableFlag_12", index: 12, oracle: "LookupTableFlag_12" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.LookupTableFlag_13", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.LookupTableFlag_13", index: 13, oracle: "LookupTableFlag_13" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.LookupTableFlag_14", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.LookupTableFlag_14", index: 14, oracle: "LookupTableFlag_14" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.LookupTableFlag_15", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.LookupTableFlag_15", index: 15, oracle: "LookupTableFlag_15" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.LookupTableFlag_16", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.LookupTableFlag_16", index: 16, oracle: "LookupTableFlag_16" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.LookupTableFlag_17", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.LookupTableFlag_17", index: 17, oracle: "LookupTableFlag_17" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.LookupTableFlag_18", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.LookupTableFlag_18", index: 18, oracle: "LookupTableFlag_18" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.LookupTableFlag_19", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.LookupTableFlag_19", index: 19, oracle: "LookupTableFlag_19" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.LookupTableFlag_20", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.LookupTableFlag_20", index: 20, oracle: "LookupTableFlag_20" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.LookupTableFlag_21", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.LookupTableFlag_21", index: 21, oracle: "LookupTableFlag_21" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.LookupTableFlag_22", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.LookupTableFlag_22", index: 22, oracle: "LookupTableFlag_22" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.LookupTableFlag_23", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.LookupTableFlag_23", index: 23, oracle: "LookupTableFlag_23" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.LookupTableFlag_24", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.LookupTableFlag_24", index: 24, oracle: "LookupTableFlag_24" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.LookupTableFlag_25", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.LookupTableFlag_25", index: 25, oracle: "LookupTableFlag_25" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.LookupTableFlag_26", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.LookupTableFlag_26", index: 26, oracle: "LookupTableFlag_26" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.LookupTableFlag_27", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.LookupTableFlag_27", index: 27, oracle: "LookupTableFlag_27" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.LookupTableFlag_28", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.LookupTableFlag_28", index: 28, oracle: "LookupTableFlag_28" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.LookupTableFlag_29", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.LookupTableFlag_29", index: 29, oracle: "LookupTableFlag_29" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.LookupTableFlag_30", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.LookupTableFlag_30", index: 30, oracle: "LookupTableFlag_30" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.LookupTableFlag_31", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.LookupTableFlag_31", index: 31, oracle: "LookupTableFlag_31" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.LookupTableFlag_32", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.LookupTableFlag_32", index: 32, oracle: "LookupTableFlag_32" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.LookupTableFlag_33", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.LookupTableFlag_33", index: 33, oracle: "LookupTableFlag_33" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.LookupTableFlag_34", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.LookupTableFlag_34", index: 34, oracle: "LookupTableFlag_34" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.LookupTableFlag_35", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.LookupTableFlag_35", index: 35, oracle: "LookupTableFlag_35" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.LookupTableFlag_36", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.LookupTableFlag_36", index: 36, oracle: "LookupTableFlag_36" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.LookupTableFlag_37", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.LookupTableFlag_37", index: 37, oracle: "LookupTableFlag_37" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.LookupTableFlag_38", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.LookupTableFlag_38", index: 38, oracle: "LookupTableFlag_38" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.LookupTableFlag_39", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.LookupTableFlag_39", index: 39, oracle: "LookupTableFlag_39" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.LookupTableFlag_40", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.LookupTableFlag_40", index: 40, oracle: "LookupTableFlag_40" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.InstructionRa_0", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.InstructionRa_0", index: 41, oracle: "InstructionRa_0" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.InstructionRa_1", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.InstructionRa_1", index: 42, oracle: "InstructionRa_1" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.InstructionRa_2", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.InstructionRa_2", index: 43, oracle: "InstructionRa_2" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.InstructionRa_3", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.InstructionRa_3", index: 44, oracle: "InstructionRa_3" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.InstructionRa_4", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.InstructionRa_4", index: 45, oracle: "InstructionRa_4" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.InstructionRa_5", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.InstructionRa_5", index: 46, oracle: "InstructionRa_5" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.InstructionRa_6", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.InstructionRa_6", index: 47, oracle: "InstructionRa_6" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.InstructionRa_7", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.InstructionRa_7", index: 48, oracle: "InstructionRa_7" },
    Stage5SumcheckEvalPlan { symbol: "stage5.instruction_read_raf.eval.InstructionRafFlag", source: "stage5.sumcheck", name: "stage5.instruction_read_raf.eval.InstructionRafFlag", index: 49, oracle: "InstructionRafFlag" },
    Stage5SumcheckEvalPlan { symbol: "stage5.ram_ra_claim_reduction.eval.RamRa", source: "stage5.sumcheck", name: "stage5.ram_ra_claim_reduction.eval.RamRa", index: 0, oracle: "RamRa" },
    Stage5SumcheckEvalPlan { symbol: "stage5.registers_val_evaluation.eval.RdInc", source: "stage5.sumcheck", name: "stage5.registers_val_evaluation.eval.RdInc", index: 0, oracle: "RdInc" },
    Stage5SumcheckEvalPlan { symbol: "stage5.registers_val_evaluation.eval.RdWa", source: "stage5.sumcheck", name: "stage5.registers_val_evaluation.eval.RdWa", index: 1, oracle: "RdWa" },
];

pub const STAGE5_POINT_SLICES: &[Stage5PointSlicePlan] = &[
    Stage5PointSlicePlan { symbol: "stage5.instruction_read_raf.point.Cycle", source: "stage5.instruction_read_raf.instance", offset: 128, length: 16, input: "stage5.instruction_read_raf.instance" },
    Stage5PointSlicePlan { symbol: "stage5.instruction_read_raf.point.InstructionRa_0.address", source: "stage5.instruction_read_raf.instance", offset: 0, length: 16, input: "stage5.instruction_read_raf.instance" },
    Stage5PointSlicePlan { symbol: "stage5.instruction_read_raf.point.InstructionRa_1.address", source: "stage5.instruction_read_raf.instance", offset: 16, length: 16, input: "stage5.instruction_read_raf.instance" },
    Stage5PointSlicePlan { symbol: "stage5.instruction_read_raf.point.InstructionRa_2.address", source: "stage5.instruction_read_raf.instance", offset: 32, length: 16, input: "stage5.instruction_read_raf.instance" },
    Stage5PointSlicePlan { symbol: "stage5.instruction_read_raf.point.InstructionRa_3.address", source: "stage5.instruction_read_raf.instance", offset: 48, length: 16, input: "stage5.instruction_read_raf.instance" },
    Stage5PointSlicePlan { symbol: "stage5.instruction_read_raf.point.InstructionRa_4.address", source: "stage5.instruction_read_raf.instance", offset: 64, length: 16, input: "stage5.instruction_read_raf.instance" },
    Stage5PointSlicePlan { symbol: "stage5.instruction_read_raf.point.InstructionRa_5.address", source: "stage5.instruction_read_raf.instance", offset: 80, length: 16, input: "stage5.instruction_read_raf.instance" },
    Stage5PointSlicePlan { symbol: "stage5.instruction_read_raf.point.InstructionRa_6.address", source: "stage5.instruction_read_raf.instance", offset: 96, length: 16, input: "stage5.instruction_read_raf.instance" },
    Stage5PointSlicePlan { symbol: "stage5.instruction_read_raf.point.InstructionRa_7.address", source: "stage5.instruction_read_raf.instance", offset: 112, length: 16, input: "stage5.instruction_read_raf.instance" },
    Stage5PointSlicePlan { symbol: "stage5.ram_ra_claim_reduction.point.RamAddress", source: "stage5.input.stage2.ram_raf.RamRa", offset: 0, length: 16, input: "stage5.input.stage2.ram_raf.RamRa" },
    Stage5PointSlicePlan { symbol: "stage5.registers_val_evaluation.point.RegisterAddress", source: "stage5.input.stage4.registers.RegistersVal", offset: 0, length: 7, input: "stage5.input.stage4.registers.RegistersVal" },
];

pub const STAGE5_POINT_CONCAT_0_INPUTS: &[&str] = &[
    "stage5.instruction_read_raf.point.InstructionRa_0.address",
    "stage5.instruction_read_raf.point.Cycle",
];

pub const STAGE5_POINT_CONCAT_1_INPUTS: &[&str] = &[
    "stage5.instruction_read_raf.point.InstructionRa_1.address",
    "stage5.instruction_read_raf.point.Cycle",
];

pub const STAGE5_POINT_CONCAT_2_INPUTS: &[&str] = &[
    "stage5.instruction_read_raf.point.InstructionRa_2.address",
    "stage5.instruction_read_raf.point.Cycle",
];

pub const STAGE5_POINT_CONCAT_3_INPUTS: &[&str] = &[
    "stage5.instruction_read_raf.point.InstructionRa_3.address",
    "stage5.instruction_read_raf.point.Cycle",
];

pub const STAGE5_POINT_CONCAT_4_INPUTS: &[&str] = &[
    "stage5.instruction_read_raf.point.InstructionRa_4.address",
    "stage5.instruction_read_raf.point.Cycle",
];

pub const STAGE5_POINT_CONCAT_5_INPUTS: &[&str] = &[
    "stage5.instruction_read_raf.point.InstructionRa_5.address",
    "stage5.instruction_read_raf.point.Cycle",
];

pub const STAGE5_POINT_CONCAT_6_INPUTS: &[&str] = &[
    "stage5.instruction_read_raf.point.InstructionRa_6.address",
    "stage5.instruction_read_raf.point.Cycle",
];

pub const STAGE5_POINT_CONCAT_7_INPUTS: &[&str] = &[
    "stage5.instruction_read_raf.point.InstructionRa_7.address",
    "stage5.instruction_read_raf.point.Cycle",
];

pub const STAGE5_POINT_CONCAT_8_INPUTS: &[&str] = &[
    "stage5.ram_ra_claim_reduction.point.RamAddress",
    "stage5.ram_ra_claim_reduction.instance",
];

pub const STAGE5_POINT_CONCAT_9_INPUTS: &[&str] = &[
    "stage5.registers_val_evaluation.point.RegisterAddress",
    "stage5.registers_val_evaluation.instance",
];

pub const STAGE5_POINT_CONCATS: &[Stage5PointConcatPlan] = &[
    Stage5PointConcatPlan { symbol: "stage5.instruction_read_raf.point.InstructionRa_0", layout: "address_chunk_then_cycle", arity: 32, inputs: STAGE5_POINT_CONCAT_0_INPUTS },
    Stage5PointConcatPlan { symbol: "stage5.instruction_read_raf.point.InstructionRa_1", layout: "address_chunk_then_cycle", arity: 32, inputs: STAGE5_POINT_CONCAT_1_INPUTS },
    Stage5PointConcatPlan { symbol: "stage5.instruction_read_raf.point.InstructionRa_2", layout: "address_chunk_then_cycle", arity: 32, inputs: STAGE5_POINT_CONCAT_2_INPUTS },
    Stage5PointConcatPlan { symbol: "stage5.instruction_read_raf.point.InstructionRa_3", layout: "address_chunk_then_cycle", arity: 32, inputs: STAGE5_POINT_CONCAT_3_INPUTS },
    Stage5PointConcatPlan { symbol: "stage5.instruction_read_raf.point.InstructionRa_4", layout: "address_chunk_then_cycle", arity: 32, inputs: STAGE5_POINT_CONCAT_4_INPUTS },
    Stage5PointConcatPlan { symbol: "stage5.instruction_read_raf.point.InstructionRa_5", layout: "address_chunk_then_cycle", arity: 32, inputs: STAGE5_POINT_CONCAT_5_INPUTS },
    Stage5PointConcatPlan { symbol: "stage5.instruction_read_raf.point.InstructionRa_6", layout: "address_chunk_then_cycle", arity: 32, inputs: STAGE5_POINT_CONCAT_6_INPUTS },
    Stage5PointConcatPlan { symbol: "stage5.instruction_read_raf.point.InstructionRa_7", layout: "address_chunk_then_cycle", arity: 32, inputs: STAGE5_POINT_CONCAT_7_INPUTS },
    Stage5PointConcatPlan { symbol: "stage5.ram_ra_claim_reduction.point.RamRa", layout: "address_then_cycle", arity: 32, inputs: STAGE5_POINT_CONCAT_8_INPUTS },
    Stage5PointConcatPlan { symbol: "stage5.registers_val_evaluation.point.RdWa", layout: "register_address_then_cycle", arity: 23, inputs: STAGE5_POINT_CONCAT_9_INPUTS },
];
pub const STAGE5_OPENING_CLAIMS: &[Stage5OpeningClaimPlan] = &[
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_0", oracle: "LookupTableFlag_0", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_0" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_1", oracle: "LookupTableFlag_1", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_1" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_2", oracle: "LookupTableFlag_2", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_2" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_3", oracle: "LookupTableFlag_3", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_3" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_4", oracle: "LookupTableFlag_4", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_4" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_5", oracle: "LookupTableFlag_5", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_5" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_6", oracle: "LookupTableFlag_6", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_6" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_7", oracle: "LookupTableFlag_7", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_7" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_8", oracle: "LookupTableFlag_8", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_8" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_9", oracle: "LookupTableFlag_9", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_9" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_10", oracle: "LookupTableFlag_10", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_10" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_11", oracle: "LookupTableFlag_11", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_11" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_12", oracle: "LookupTableFlag_12", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_12" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_13", oracle: "LookupTableFlag_13", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_13" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_14", oracle: "LookupTableFlag_14", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_14" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_15", oracle: "LookupTableFlag_15", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_15" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_16", oracle: "LookupTableFlag_16", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_16" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_17", oracle: "LookupTableFlag_17", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_17" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_18", oracle: "LookupTableFlag_18", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_18" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_19", oracle: "LookupTableFlag_19", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_19" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_20", oracle: "LookupTableFlag_20", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_20" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_21", oracle: "LookupTableFlag_21", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_21" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_22", oracle: "LookupTableFlag_22", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_22" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_23", oracle: "LookupTableFlag_23", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_23" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_24", oracle: "LookupTableFlag_24", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_24" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_25", oracle: "LookupTableFlag_25", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_25" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_26", oracle: "LookupTableFlag_26", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_26" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_27", oracle: "LookupTableFlag_27", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_27" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_28", oracle: "LookupTableFlag_28", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_28" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_29", oracle: "LookupTableFlag_29", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_29" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_30", oracle: "LookupTableFlag_30", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_30" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_31", oracle: "LookupTableFlag_31", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_31" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_32", oracle: "LookupTableFlag_32", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_32" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_33", oracle: "LookupTableFlag_33", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_33" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_34", oracle: "LookupTableFlag_34", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_34" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_35", oracle: "LookupTableFlag_35", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_35" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_36", oracle: "LookupTableFlag_36", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_36" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_37", oracle: "LookupTableFlag_37", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_37" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_38", oracle: "LookupTableFlag_38", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_38" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_39", oracle: "LookupTableFlag_39", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_39" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.LookupTableFlag_40", oracle: "LookupTableFlag_40", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.LookupTableFlag_40" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.InstructionRa_0", oracle: "InstructionRa_0", domain: "jolt.stage5_instruction_ra_chunk_domain", point_arity: 32, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.InstructionRa_0", eval_source: "stage5.instruction_read_raf.eval.InstructionRa_0" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.InstructionRa_1", oracle: "InstructionRa_1", domain: "jolt.stage5_instruction_ra_chunk_domain", point_arity: 32, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.InstructionRa_1", eval_source: "stage5.instruction_read_raf.eval.InstructionRa_1" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.InstructionRa_2", oracle: "InstructionRa_2", domain: "jolt.stage5_instruction_ra_chunk_domain", point_arity: 32, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.InstructionRa_2", eval_source: "stage5.instruction_read_raf.eval.InstructionRa_2" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.InstructionRa_3", oracle: "InstructionRa_3", domain: "jolt.stage5_instruction_ra_chunk_domain", point_arity: 32, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.InstructionRa_3", eval_source: "stage5.instruction_read_raf.eval.InstructionRa_3" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.InstructionRa_4", oracle: "InstructionRa_4", domain: "jolt.stage5_instruction_ra_chunk_domain", point_arity: 32, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.InstructionRa_4", eval_source: "stage5.instruction_read_raf.eval.InstructionRa_4" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.InstructionRa_5", oracle: "InstructionRa_5", domain: "jolt.stage5_instruction_ra_chunk_domain", point_arity: 32, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.InstructionRa_5", eval_source: "stage5.instruction_read_raf.eval.InstructionRa_5" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.InstructionRa_6", oracle: "InstructionRa_6", domain: "jolt.stage5_instruction_ra_chunk_domain", point_arity: 32, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.InstructionRa_6", eval_source: "stage5.instruction_read_raf.eval.InstructionRa_6" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.InstructionRa_7", oracle: "InstructionRa_7", domain: "jolt.stage5_instruction_ra_chunk_domain", point_arity: 32, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.InstructionRa_7", eval_source: "stage5.instruction_read_raf.eval.InstructionRa_7" },
    Stage5OpeningClaimPlan { symbol: "stage5.instruction_read_raf.opening.InstructionRafFlag", oracle: "InstructionRafFlag", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage5.instruction_read_raf.point.Cycle", eval_source: "stage5.instruction_read_raf.eval.InstructionRafFlag" },
    Stage5OpeningClaimPlan { symbol: "stage5.ram_ra_claim_reduction.opening.RamRa", oracle: "RamRa", domain: "jolt.stage2_ram_rw_domain", point_arity: 32, claim_kind: "virtual", point_source: "stage5.ram_ra_claim_reduction.point.RamRa", eval_source: "stage5.ram_ra_claim_reduction.eval.RamRa" },
    Stage5OpeningClaimPlan { symbol: "stage5.registers_val_evaluation.opening.RdInc", oracle: "RdInc", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "committed", point_source: "stage5.registers_val_evaluation.instance", eval_source: "stage5.registers_val_evaluation.eval.RdInc" },
    Stage5OpeningClaimPlan { symbol: "stage5.registers_val_evaluation.opening.RdWa", oracle: "RdWa", domain: "jolt.stage4_registers_rw_domain", point_arity: 23, claim_kind: "virtual", point_source: "stage5.registers_val_evaluation.point.RdWa", eval_source: "stage5.registers_val_evaluation.eval.RdWa" },
];

pub const STAGE5_OPENING_EQUALITIES: &[Stage5OpeningClaimEqualityPlan] = &[
    Stage5OpeningClaimEqualityPlan { symbol: "stage5.instruction.lookup_output_claim_consistency", mode: "point_and_eval", lhs: "stage5.input.stage2.instruction.LookupOutput", rhs: "stage5.input.stage2.product_virtual.LookupOutput" },
];

pub const STAGE5_OPENING_BATCH_0_ORDERED_CLAIMS: &[&str] = &[
    "stage5.instruction_read_raf.opening.LookupTableFlag_0",
    "stage5.instruction_read_raf.opening.LookupTableFlag_1",
    "stage5.instruction_read_raf.opening.LookupTableFlag_2",
    "stage5.instruction_read_raf.opening.LookupTableFlag_3",
    "stage5.instruction_read_raf.opening.LookupTableFlag_4",
    "stage5.instruction_read_raf.opening.LookupTableFlag_5",
    "stage5.instruction_read_raf.opening.LookupTableFlag_6",
    "stage5.instruction_read_raf.opening.LookupTableFlag_7",
    "stage5.instruction_read_raf.opening.LookupTableFlag_8",
    "stage5.instruction_read_raf.opening.LookupTableFlag_9",
    "stage5.instruction_read_raf.opening.LookupTableFlag_10",
    "stage5.instruction_read_raf.opening.LookupTableFlag_11",
    "stage5.instruction_read_raf.opening.LookupTableFlag_12",
    "stage5.instruction_read_raf.opening.LookupTableFlag_13",
    "stage5.instruction_read_raf.opening.LookupTableFlag_14",
    "stage5.instruction_read_raf.opening.LookupTableFlag_15",
    "stage5.instruction_read_raf.opening.LookupTableFlag_16",
    "stage5.instruction_read_raf.opening.LookupTableFlag_17",
    "stage5.instruction_read_raf.opening.LookupTableFlag_18",
    "stage5.instruction_read_raf.opening.LookupTableFlag_19",
    "stage5.instruction_read_raf.opening.LookupTableFlag_20",
    "stage5.instruction_read_raf.opening.LookupTableFlag_21",
    "stage5.instruction_read_raf.opening.LookupTableFlag_22",
    "stage5.instruction_read_raf.opening.LookupTableFlag_23",
    "stage5.instruction_read_raf.opening.LookupTableFlag_24",
    "stage5.instruction_read_raf.opening.LookupTableFlag_25",
    "stage5.instruction_read_raf.opening.LookupTableFlag_26",
    "stage5.instruction_read_raf.opening.LookupTableFlag_27",
    "stage5.instruction_read_raf.opening.LookupTableFlag_28",
    "stage5.instruction_read_raf.opening.LookupTableFlag_29",
    "stage5.instruction_read_raf.opening.LookupTableFlag_30",
    "stage5.instruction_read_raf.opening.LookupTableFlag_31",
    "stage5.instruction_read_raf.opening.LookupTableFlag_32",
    "stage5.instruction_read_raf.opening.LookupTableFlag_33",
    "stage5.instruction_read_raf.opening.LookupTableFlag_34",
    "stage5.instruction_read_raf.opening.LookupTableFlag_35",
    "stage5.instruction_read_raf.opening.LookupTableFlag_36",
    "stage5.instruction_read_raf.opening.LookupTableFlag_37",
    "stage5.instruction_read_raf.opening.LookupTableFlag_38",
    "stage5.instruction_read_raf.opening.LookupTableFlag_39",
    "stage5.instruction_read_raf.opening.LookupTableFlag_40",
    "stage5.instruction_read_raf.opening.InstructionRa_0",
    "stage5.instruction_read_raf.opening.InstructionRa_1",
    "stage5.instruction_read_raf.opening.InstructionRa_2",
    "stage5.instruction_read_raf.opening.InstructionRa_3",
    "stage5.instruction_read_raf.opening.InstructionRa_4",
    "stage5.instruction_read_raf.opening.InstructionRa_5",
    "stage5.instruction_read_raf.opening.InstructionRa_6",
    "stage5.instruction_read_raf.opening.InstructionRa_7",
    "stage5.instruction_read_raf.opening.InstructionRafFlag",
    "stage5.ram_ra_claim_reduction.opening.RamRa",
    "stage5.registers_val_evaluation.opening.RdInc",
    "stage5.registers_val_evaluation.opening.RdWa",
];

pub const STAGE5_OPENING_BATCH_0_CLAIM_OPERANDS: &[&str] = &[
    "stage5.instruction_read_raf.opening.LookupTableFlag_0",
    "stage5.instruction_read_raf.opening.LookupTableFlag_1",
    "stage5.instruction_read_raf.opening.LookupTableFlag_2",
    "stage5.instruction_read_raf.opening.LookupTableFlag_3",
    "stage5.instruction_read_raf.opening.LookupTableFlag_4",
    "stage5.instruction_read_raf.opening.LookupTableFlag_5",
    "stage5.instruction_read_raf.opening.LookupTableFlag_6",
    "stage5.instruction_read_raf.opening.LookupTableFlag_7",
    "stage5.instruction_read_raf.opening.LookupTableFlag_8",
    "stage5.instruction_read_raf.opening.LookupTableFlag_9",
    "stage5.instruction_read_raf.opening.LookupTableFlag_10",
    "stage5.instruction_read_raf.opening.LookupTableFlag_11",
    "stage5.instruction_read_raf.opening.LookupTableFlag_12",
    "stage5.instruction_read_raf.opening.LookupTableFlag_13",
    "stage5.instruction_read_raf.opening.LookupTableFlag_14",
    "stage5.instruction_read_raf.opening.LookupTableFlag_15",
    "stage5.instruction_read_raf.opening.LookupTableFlag_16",
    "stage5.instruction_read_raf.opening.LookupTableFlag_17",
    "stage5.instruction_read_raf.opening.LookupTableFlag_18",
    "stage5.instruction_read_raf.opening.LookupTableFlag_19",
    "stage5.instruction_read_raf.opening.LookupTableFlag_20",
    "stage5.instruction_read_raf.opening.LookupTableFlag_21",
    "stage5.instruction_read_raf.opening.LookupTableFlag_22",
    "stage5.instruction_read_raf.opening.LookupTableFlag_23",
    "stage5.instruction_read_raf.opening.LookupTableFlag_24",
    "stage5.instruction_read_raf.opening.LookupTableFlag_25",
    "stage5.instruction_read_raf.opening.LookupTableFlag_26",
    "stage5.instruction_read_raf.opening.LookupTableFlag_27",
    "stage5.instruction_read_raf.opening.LookupTableFlag_28",
    "stage5.instruction_read_raf.opening.LookupTableFlag_29",
    "stage5.instruction_read_raf.opening.LookupTableFlag_30",
    "stage5.instruction_read_raf.opening.LookupTableFlag_31",
    "stage5.instruction_read_raf.opening.LookupTableFlag_32",
    "stage5.instruction_read_raf.opening.LookupTableFlag_33",
    "stage5.instruction_read_raf.opening.LookupTableFlag_34",
    "stage5.instruction_read_raf.opening.LookupTableFlag_35",
    "stage5.instruction_read_raf.opening.LookupTableFlag_36",
    "stage5.instruction_read_raf.opening.LookupTableFlag_37",
    "stage5.instruction_read_raf.opening.LookupTableFlag_38",
    "stage5.instruction_read_raf.opening.LookupTableFlag_39",
    "stage5.instruction_read_raf.opening.LookupTableFlag_40",
    "stage5.instruction_read_raf.opening.InstructionRa_0",
    "stage5.instruction_read_raf.opening.InstructionRa_1",
    "stage5.instruction_read_raf.opening.InstructionRa_2",
    "stage5.instruction_read_raf.opening.InstructionRa_3",
    "stage5.instruction_read_raf.opening.InstructionRa_4",
    "stage5.instruction_read_raf.opening.InstructionRa_5",
    "stage5.instruction_read_raf.opening.InstructionRa_6",
    "stage5.instruction_read_raf.opening.InstructionRa_7",
    "stage5.instruction_read_raf.opening.InstructionRafFlag",
    "stage5.ram_ra_claim_reduction.opening.RamRa",
    "stage5.registers_val_evaluation.opening.RdInc",
    "stage5.registers_val_evaluation.opening.RdWa",
];

pub const STAGE5_OPENING_BATCHES: &[Stage5OpeningBatchPlan] = &[
    Stage5OpeningBatchPlan { symbol: "stage5.openings", stage: "stage5", proof_slot: "stage5.openings", policy: "jolt_stage5_output_order", count: 53, ordered_claims: STAGE5_OPENING_BATCH_0_ORDERED_CLAIMS, claim_operands: STAGE5_OPENING_BATCH_0_CLAIM_OPERANDS },
];
pub const STAGE5_PROGRAM: Stage5CpuProgramPlan = Stage5CpuProgramPlan {
    role: "prover",
    params: STAGE5_PARAMS,
    steps: STAGE5_PROGRAM_STEPS,
    transcript_squeezes: STAGE5_TRANSCRIPT_SQUEEZES,
    transcript_absorb_bytes: STAGE5_TRANSCRIPT_ABSORB_BYTES,
    opening_inputs: STAGE5_OPENING_INPUTS,
    field_constants: STAGE5_FIELD_CONSTANTS,
    field_exprs: STAGE5_FIELD_EXPRS,
    kernels: STAGE5_KERNELS,
    claims: STAGE5_SUMCHECK_CLAIMS,
    batches: STAGE5_SUMCHECK_BATCHES,
    drivers: STAGE5_SUMCHECK_DRIVERS,
    instance_results: STAGE5_SUMCHECK_INSTANCE_RESULTS,
    evals: STAGE5_SUMCHECK_EVALS,
    point_slices: STAGE5_POINT_SLICES,
    point_concats: STAGE5_POINT_CONCATS,
    opening_claims: STAGE5_OPENING_CLAIMS,
    opening_equalities: STAGE5_OPENING_EQUALITIES,
    opening_batches: STAGE5_OPENING_BATCHES,
};

pub fn execute_stage5_prover<E, T>(
    executor: &mut E,
    transcript: &mut T,
) -> Result<Stage5ExecutionArtifacts<Fr>, Stage5KernelError>
where
    E: Stage5KernelExecutor<Fr>,
    T: Transcript<Challenge = Fr>,
{
    execute_stage5_prover_with_program(&STAGE5_PROGRAM, executor, transcript)
}

pub fn execute_stage5_prover_with_program<E, T>(
    program: &'static Stage5CpuProgramPlan,
    executor: &mut E,
    transcript: &mut T,
) -> Result<Stage5ExecutionArtifacts<Fr>, Stage5KernelError>
where
    E: Stage5KernelExecutor<Fr>,
    T: Transcript<Challenge = Fr>,
{
    execute_stage5_program(program, Stage5ExecutionMode::Prover, executor, transcript)
}
