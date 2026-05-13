#![allow(dead_code)]

use jolt_field::Fr;
use jolt_kernels::stage4::{execute_stage4_program, Stage4CpuProgramPlan, Stage4ExecutionArtifacts, Stage4ExecutionMode, Stage4FieldConstantPlan, Stage4FieldExprPlan, Stage4KernelError, Stage4KernelExecutor, Stage4KernelPlan, Stage4OpeningBatchPlan, Stage4OpeningClaimEqualityPlan, Stage4OpeningClaimPlan, Stage4OpeningInputPlan, Stage4Params, Stage4PointConcatPlan, Stage4PointSlicePlan, Stage4ProgramStepPlan, Stage4SumcheckBatchPlan, Stage4SumcheckClaimPlan, Stage4SumcheckDriverPlan, Stage4SumcheckEvalPlan, Stage4SumcheckInstanceResultPlan, Stage4TranscriptAbsorbBytesPlan, Stage4TranscriptSqueezePlan};
use jolt_transcript::{Blake2bTranscript, Transcript};

pub type DefaultStage4Transcript = Blake2bTranscript<Fr>;

pub const STAGE4_PARAMS: Stage4Params = Stage4Params {
    field: "bn254_fr",
    pcs: "dory",
    transcript: "blake2b_transcript",
};
pub const STAGE4_PROGRAM_STEPS: &[Stage4ProgramStepPlan] = &[
    Stage4ProgramStepPlan { kind: "transcript_squeeze", symbol: "stage4.registers_read_write.gamma" },
    Stage4ProgramStepPlan { kind: "transcript_squeeze", symbol: "stage4.field_registers_read_write.gamma" },
    Stage4ProgramStepPlan { kind: "transcript_absorb_bytes", symbol: "stage4.ram_val_check.domain_separator" },
    Stage4ProgramStepPlan { kind: "transcript_squeeze", symbol: "stage4.ram_val_check.gamma" },
    Stage4ProgramStepPlan { kind: "sumcheck_driver", symbol: "stage4.sumcheck" },
];

pub const STAGE4_TRANSCRIPT_SQUEEZES: &[Stage4TranscriptSqueezePlan] = &[
    Stage4TranscriptSqueezePlan { symbol: "stage4.registers_read_write.gamma", label: "registers_read_write_gamma", kind: "challenge_scalar", count: 1 },
    Stage4TranscriptSqueezePlan { symbol: "stage4.field_registers_read_write.gamma", label: "field_registers_read_write_gamma", kind: "challenge_scalar", count: 1 },
    Stage4TranscriptSqueezePlan { symbol: "stage4.ram_val_check.gamma", label: "ram_val_check_gamma", kind: "challenge_scalar", count: 1 },
];

pub const STAGE4_TRANSCRIPT_ABSORB_BYTES: &[Stage4TranscriptAbsorbBytesPlan] = &[
    Stage4TranscriptAbsorbBytesPlan { symbol: "stage4.ram_val_check.domain_separator", label: "ram_val_check_gamma", payload: "" },
];

pub const STAGE4_OPENING_INPUTS: &[Stage4OpeningInputPlan] = &[
    Stage4OpeningInputPlan { symbol: "stage4.input.stage3.registers.RdWriteValue", source_stage: "stage3", source_claim: "stage3.registers_claim_reduction.opening.RdWriteValue", oracle: "RdWriteValue", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual" },
    Stage4OpeningInputPlan { symbol: "stage4.input.stage3.registers.Rs1Value", source_stage: "stage3", source_claim: "stage3.registers_claim_reduction.opening.Rs1Value", oracle: "Rs1Value", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual" },
    Stage4OpeningInputPlan { symbol: "stage4.input.stage3.registers.Rs2Value", source_stage: "stage3", source_claim: "stage3.registers_claim_reduction.opening.Rs2Value", oracle: "Rs2Value", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual" },
    Stage4OpeningInputPlan { symbol: "stage4.input.stage3.instruction.Rs1Value", source_stage: "stage3", source_claim: "stage3.instruction_input.opening.Rs1Value", oracle: "Rs1Value", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual" },
    Stage4OpeningInputPlan { symbol: "stage4.input.stage3.instruction.Rs2Value", source_stage: "stage3", source_claim: "stage3.instruction_input.opening.Rs2Value", oracle: "Rs2Value", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual" },
    Stage4OpeningInputPlan { symbol: "stage4.input.stage2.RamVal", source_stage: "stage2", source_claim: "stage2.ram_read_write.opening.RamVal", oracle: "RamVal", domain: "jolt.stage2_ram_rw_domain", point_arity: 32, claim_kind: "virtual" },
    Stage4OpeningInputPlan { symbol: "stage4.input.stage2.RamValFinal", source_stage: "stage2", source_claim: "stage2.ram_output.opening.RamValFinal", oracle: "RamValFinal", domain: "jolt.ram_address_domain", point_arity: 14, claim_kind: "virtual" },
    Stage4OpeningInputPlan { symbol: "stage4.input.initial_ram.RamValInit", source_stage: "stage4_precomputed", source_claim: "stage4.ram_val_check.initial_ram_eval", oracle: "RamValInit", domain: "jolt.ram_address_domain", point_arity: 14, claim_kind: "virtual" },
    Stage4OpeningInputPlan { symbol: "stage4.input.stage3.field_registers.FieldRdValue", source_stage: "stage3", source_claim: "stage3.field_registers_claim_reduction.opening.FieldRdValue", oracle: "FieldRdValue", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual" },
    Stage4OpeningInputPlan { symbol: "stage4.input.stage3.field_registers.FieldRs1Value", source_stage: "stage3", source_claim: "stage3.field_registers_claim_reduction.opening.FieldRs1Value", oracle: "FieldRs1Value", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual" },
    Stage4OpeningInputPlan { symbol: "stage4.input.stage3.field_registers.FieldRs2Value", source_stage: "stage3", source_claim: "stage3.field_registers_claim_reduction.opening.FieldRs2Value", oracle: "FieldRs2Value", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "virtual" },
];

pub const STAGE4_FIELD_CONSTANTS: &[Stage4FieldConstantPlan] = &[

];

pub const STAGE4_FIELD_EXPR_OPERANDS_0: &[&str] = &["stage4.registers_read_write.gamma"];

pub const STAGE4_FIELD_EXPR_OPERANDS_1: &[&str] = &[
    "stage4.registers_read_write.gamma",
    "stage4.input.stage3.registers.Rs1Value",
];

pub const STAGE4_FIELD_EXPR_OPERANDS_2: &[&str] = &[
    "stage4.registers_read_write.gamma2",
    "stage4.input.stage3.registers.Rs2Value",
];

pub const STAGE4_FIELD_EXPR_OPERANDS_3: &[&str] = &[
    "stage4.input.stage3.registers.RdWriteValue",
    "stage4.registers_read_write.term.Rs1Value",
];

pub const STAGE4_FIELD_EXPR_OPERANDS_4: &[&str] = &[
    "stage4.registers_read_write.partial.RdWriteValueRs1Value",
    "stage4.registers_read_write.term.Rs2Value",
];

pub const STAGE4_FIELD_EXPR_OPERANDS_5: &[&str] = &["stage4.field_registers_read_write.gamma"];

pub const STAGE4_FIELD_EXPR_OPERANDS_6: &[&str] = &[
    "stage4.field_registers_read_write.gamma",
    "stage4.input.stage3.field_registers.FieldRs1Value",
];

pub const STAGE4_FIELD_EXPR_OPERANDS_7: &[&str] = &[
    "stage4.field_registers_read_write.gamma2",
    "stage4.input.stage3.field_registers.FieldRs2Value",
];

pub const STAGE4_FIELD_EXPR_OPERANDS_8: &[&str] = &[
    "stage4.input.stage3.field_registers.FieldRdValue",
    "stage4.field_registers_read_write.term.FieldRs1Value",
];

pub const STAGE4_FIELD_EXPR_OPERANDS_9: &[&str] = &[
    "stage4.field_registers_read_write.partial.FieldRdValueFieldRs1Value",
    "stage4.field_registers_read_write.term.FieldRs2Value",
];

pub const STAGE4_FIELD_EXPR_OPERANDS_10: &[&str] = &[
    "stage4.input.stage2.RamVal",
    "stage4.input.initial_ram.RamValInit",
];

pub const STAGE4_FIELD_EXPR_OPERANDS_11: &[&str] = &[
    "stage4.input.stage2.RamValFinal",
    "stage4.input.initial_ram.RamValInit",
];

pub const STAGE4_FIELD_EXPR_OPERANDS_12: &[&str] = &[
    "stage4.ram_val_check.gamma",
    "stage4.ram_val_check.delta.RamValFinal",
];

pub const STAGE4_FIELD_EXPR_OPERANDS_13: &[&str] = &[
    "stage4.ram_val_check.delta.RamVal",
    "stage4.ram_val_check.term.RamValFinal",
];

pub const STAGE4_FIELD_EXPRS: &[Stage4FieldExprPlan] = &[
    Stage4FieldExprPlan { symbol: "stage4.registers_read_write.gamma2", kind: "op", formula: "field.pow:2", operand_names: STAGE4_FIELD_EXPR_OPERANDS_0, operands: STAGE4_FIELD_EXPR_OPERANDS_0 },
    Stage4FieldExprPlan { symbol: "stage4.registers_read_write.term.Rs1Value", kind: "op", formula: "field.mul", operand_names: STAGE4_FIELD_EXPR_OPERANDS_1, operands: STAGE4_FIELD_EXPR_OPERANDS_1 },
    Stage4FieldExprPlan { symbol: "stage4.registers_read_write.term.Rs2Value", kind: "op", formula: "field.mul", operand_names: STAGE4_FIELD_EXPR_OPERANDS_2, operands: STAGE4_FIELD_EXPR_OPERANDS_2 },
    Stage4FieldExprPlan { symbol: "stage4.registers_read_write.partial.RdWriteValueRs1Value", kind: "op", formula: "field.add", operand_names: STAGE4_FIELD_EXPR_OPERANDS_3, operands: STAGE4_FIELD_EXPR_OPERANDS_3 },
    Stage4FieldExprPlan { symbol: "stage4.registers_read_write.claim_expr", kind: "op", formula: "field.add", operand_names: STAGE4_FIELD_EXPR_OPERANDS_4, operands: STAGE4_FIELD_EXPR_OPERANDS_4 },
    Stage4FieldExprPlan { symbol: "stage4.field_registers_read_write.gamma2", kind: "op", formula: "field.pow:2", operand_names: STAGE4_FIELD_EXPR_OPERANDS_5, operands: STAGE4_FIELD_EXPR_OPERANDS_5 },
    Stage4FieldExprPlan { symbol: "stage4.field_registers_read_write.term.FieldRs1Value", kind: "op", formula: "field.mul", operand_names: STAGE4_FIELD_EXPR_OPERANDS_6, operands: STAGE4_FIELD_EXPR_OPERANDS_6 },
    Stage4FieldExprPlan { symbol: "stage4.field_registers_read_write.term.FieldRs2Value", kind: "op", formula: "field.mul", operand_names: STAGE4_FIELD_EXPR_OPERANDS_7, operands: STAGE4_FIELD_EXPR_OPERANDS_7 },
    Stage4FieldExprPlan { symbol: "stage4.field_registers_read_write.partial.FieldRdValueFieldRs1Value", kind: "op", formula: "field.add", operand_names: STAGE4_FIELD_EXPR_OPERANDS_8, operands: STAGE4_FIELD_EXPR_OPERANDS_8 },
    Stage4FieldExprPlan { symbol: "stage4.field_registers_read_write.claim_expr", kind: "op", formula: "field.add", operand_names: STAGE4_FIELD_EXPR_OPERANDS_9, operands: STAGE4_FIELD_EXPR_OPERANDS_9 },
    Stage4FieldExprPlan { symbol: "stage4.ram_val_check.delta.RamVal", kind: "op", formula: "field.sub", operand_names: STAGE4_FIELD_EXPR_OPERANDS_10, operands: STAGE4_FIELD_EXPR_OPERANDS_10 },
    Stage4FieldExprPlan { symbol: "stage4.ram_val_check.delta.RamValFinal", kind: "op", formula: "field.sub", operand_names: STAGE4_FIELD_EXPR_OPERANDS_11, operands: STAGE4_FIELD_EXPR_OPERANDS_11 },
    Stage4FieldExprPlan { symbol: "stage4.ram_val_check.term.RamValFinal", kind: "op", formula: "field.mul", operand_names: STAGE4_FIELD_EXPR_OPERANDS_12, operands: STAGE4_FIELD_EXPR_OPERANDS_12 },
    Stage4FieldExprPlan { symbol: "stage4.ram_val_check.claim_expr", kind: "op", formula: "field.add", operand_names: STAGE4_FIELD_EXPR_OPERANDS_13, operands: STAGE4_FIELD_EXPR_OPERANDS_13 },
];
pub const STAGE4_KERNELS: &[Stage4KernelPlan] = &[
    Stage4KernelPlan { symbol: "jolt.cpu.stage4.registers_read_write", relation: "jolt.stage4.registers_read_write", kind: "sumcheck", backend: "cpu", abi: "jolt_stage4_registers_read_write" },
    Stage4KernelPlan { symbol: "jolt.cpu.stage4.field_registers_read_write", relation: "jolt.stage4.field_registers_read_write", kind: "sumcheck", backend: "cpu", abi: "jolt_stage4_field_registers_read_write" },
    Stage4KernelPlan { symbol: "jolt.cpu.stage4.ram_val_check", relation: "jolt.stage4.ram_val_check", kind: "sumcheck", backend: "cpu", abi: "jolt_stage4_ram_val_check" },
    Stage4KernelPlan { symbol: "jolt.cpu.stage4.batched", relation: "jolt.stage4.batched", kind: "sumcheck", backend: "cpu", abi: "jolt_stage4_batched" },
];

pub const STAGE4_SUMCHECK_CLAIM_0_INPUT_OPENINGS: &[&str] = &[
    "stage4.input.stage3.registers.RdWriteValue",
    "stage4.input.stage3.registers.Rs1Value",
    "stage4.input.stage3.registers.Rs2Value",
];

pub const STAGE4_SUMCHECK_CLAIM_1_INPUT_OPENINGS: &[&str] = &[
    "stage4.input.stage3.field_registers.FieldRdValue",
    "stage4.input.stage3.field_registers.FieldRs1Value",
    "stage4.input.stage3.field_registers.FieldRs2Value",
];

pub const STAGE4_SUMCHECK_CLAIM_2_INPUT_OPENINGS: &[&str] = &[
    "stage4.input.stage2.RamVal",
    "stage4.input.stage2.RamValFinal",
    "stage4.input.initial_ram.RamValInit",
];

pub const STAGE4_SUMCHECK_CLAIMS: &[Stage4SumcheckClaimPlan] = &[
    Stage4SumcheckClaimPlan { symbol: "stage4.registers_read_write.input", stage: "stage4", domain: "jolt.stage4_registers_rw_domain", num_rounds: 25, degree: 3, claim: "stage4.registers_read_write.weighted_values", kernel: Some("jolt.cpu.stage4.registers_read_write"), relation: None, claim_value: "stage4.registers_read_write.claim_expr", input_openings: STAGE4_SUMCHECK_CLAIM_0_INPUT_OPENINGS },
    Stage4SumcheckClaimPlan { symbol: "stage4.field_registers_read_write.input", stage: "stage4", domain: "jolt.stage4_field_registers_rw_domain", num_rounds: 22, degree: 3, claim: "stage4.field_registers_read_write.weighted_values", kernel: Some("jolt.cpu.stage4.field_registers_read_write"), relation: None, claim_value: "stage4.field_registers_read_write.claim_expr", input_openings: STAGE4_SUMCHECK_CLAIM_1_INPUT_OPENINGS },
    Stage4SumcheckClaimPlan { symbol: "stage4.ram_val_check.input", stage: "stage4", domain: "jolt.trace_domain", num_rounds: 18, degree: 3, claim: "stage4.ram_val_check.weighted_values", kernel: Some("jolt.cpu.stage4.ram_val_check"), relation: None, claim_value: "stage4.ram_val_check.claim_expr", input_openings: STAGE4_SUMCHECK_CLAIM_2_INPUT_OPENINGS },
];
pub const STAGE4_SUMCHECK_BATCH_0_ORDERED_CLAIMS: &[&str] = &[
    "stage4.registers_read_write.input",
    "stage4.field_registers_read_write.input",
    "stage4.ram_val_check.input",
];

pub const STAGE4_SUMCHECK_BATCH_0_CLAIM_OPERANDS: &[&str] = &[
    "stage4.registers_read_write.input",
    "stage4.field_registers_read_write.input",
    "stage4.ram_val_check.input",
];

pub const STAGE4_SUMCHECK_BATCH_0_ROUND_SCHEDULE: &[usize] = &[
    18,
    7,
];

pub const STAGE4_SUMCHECK_BATCHES: &[Stage4SumcheckBatchPlan] = &[
    Stage4SumcheckBatchPlan { symbol: "stage4.batch", stage: "stage4", proof_slot: "stage4.sumcheck", policy: "jolt_core_stage4_aligned", count: 3, ordered_claims: STAGE4_SUMCHECK_BATCH_0_ORDERED_CLAIMS, claim_operands: STAGE4_SUMCHECK_BATCH_0_CLAIM_OPERANDS, claim_label: "sumcheck_claim", round_label: "sumcheck_poly", round_schedule: STAGE4_SUMCHECK_BATCH_0_ROUND_SCHEDULE },
];
pub const STAGE4_SUMCHECK_DRIVER_0_ROUND_SCHEDULE: &[usize] = &[
    18,
    7,
];

pub const STAGE4_SUMCHECK_DRIVERS: &[Stage4SumcheckDriverPlan] = &[
    Stage4SumcheckDriverPlan { symbol: "stage4.sumcheck", stage: "stage4", proof_slot: "stage4.sumcheck", kernel: Some("jolt.cpu.stage4.batched"), relation: None, batch: "stage4.batch", policy: "jolt_core_stage4_aligned", round_schedule: STAGE4_SUMCHECK_DRIVER_0_ROUND_SCHEDULE, claim_label: "sumcheck_claim", round_label: "sumcheck_poly", num_rounds: 25, degree: 3 },
];
pub const STAGE4_SUMCHECK_INSTANCE_RESULTS: &[Stage4SumcheckInstanceResultPlan] = &[
    Stage4SumcheckInstanceResultPlan { symbol: "stage4.registers_read_write.instance", source: "stage4.sumcheck", claim: "stage4.registers_read_write.input", relation: "jolt.stage4.registers_read_write", index: 0, point_arity: 25, num_rounds: 25, round_offset: 0, point_order: "stage4_registers_rw", degree: 3 },
    Stage4SumcheckInstanceResultPlan { symbol: "stage4.field_registers_read_write.instance", source: "stage4.sumcheck", claim: "stage4.field_registers_read_write.input", relation: "jolt.stage4.field_registers_read_write", index: 1, point_arity: 22, num_rounds: 22, round_offset: 0, point_order: "stage4_field_registers_rw", degree: 3 },
    Stage4SumcheckInstanceResultPlan { symbol: "stage4.ram_val_check.instance", source: "stage4.sumcheck", claim: "stage4.ram_val_check.input", relation: "jolt.stage4.ram_val_check", index: 2, point_arity: 18, num_rounds: 18, round_offset: 7, point_order: "reverse", degree: 3 },
];

pub const STAGE4_SUMCHECK_EVALS: &[Stage4SumcheckEvalPlan] = &[
    Stage4SumcheckEvalPlan { symbol: "stage4.registers_read_write.eval.RegistersVal", source: "stage4.sumcheck", name: "stage4.registers_read_write.eval.RegistersVal", index: 0, oracle: "RegistersVal" },
    Stage4SumcheckEvalPlan { symbol: "stage4.registers_read_write.eval.Rs1Ra", source: "stage4.sumcheck", name: "stage4.registers_read_write.eval.Rs1Ra", index: 1, oracle: "Rs1Ra" },
    Stage4SumcheckEvalPlan { symbol: "stage4.registers_read_write.eval.Rs2Ra", source: "stage4.sumcheck", name: "stage4.registers_read_write.eval.Rs2Ra", index: 2, oracle: "Rs2Ra" },
    Stage4SumcheckEvalPlan { symbol: "stage4.registers_read_write.eval.RdWa", source: "stage4.sumcheck", name: "stage4.registers_read_write.eval.RdWa", index: 3, oracle: "RdWa" },
    Stage4SumcheckEvalPlan { symbol: "stage4.registers_read_write.eval.RdInc", source: "stage4.sumcheck", name: "stage4.registers_read_write.eval.RdInc", index: 4, oracle: "RdInc" },
    Stage4SumcheckEvalPlan { symbol: "stage4.field_registers_read_write.eval.FieldRegistersVal", source: "stage4.sumcheck", name: "stage4.field_registers_read_write.eval.FieldRegistersVal", index: 0, oracle: "FieldRegistersVal" },
    Stage4SumcheckEvalPlan { symbol: "stage4.field_registers_read_write.eval.FieldRs1Ra", source: "stage4.sumcheck", name: "stage4.field_registers_read_write.eval.FieldRs1Ra", index: 1, oracle: "FieldRs1Ra" },
    Stage4SumcheckEvalPlan { symbol: "stage4.field_registers_read_write.eval.FieldRs2Ra", source: "stage4.sumcheck", name: "stage4.field_registers_read_write.eval.FieldRs2Ra", index: 2, oracle: "FieldRs2Ra" },
    Stage4SumcheckEvalPlan { symbol: "stage4.field_registers_read_write.eval.FieldRdWa", source: "stage4.sumcheck", name: "stage4.field_registers_read_write.eval.FieldRdWa", index: 3, oracle: "FieldRdWa" },
    Stage4SumcheckEvalPlan { symbol: "stage4.field_registers_read_write.eval.FieldRdInc", source: "stage4.sumcheck", name: "stage4.field_registers_read_write.eval.FieldRdInc", index: 4, oracle: "FieldRdInc" },
    Stage4SumcheckEvalPlan { symbol: "stage4.ram_val_check.eval.RamRa", source: "stage4.sumcheck", name: "stage4.ram_val_check.eval.RamRa", index: 0, oracle: "RamRa" },
    Stage4SumcheckEvalPlan { symbol: "stage4.ram_val_check.eval.RamInc", source: "stage4.sumcheck", name: "stage4.ram_val_check.eval.RamInc", index: 1, oracle: "RamInc" },
];

pub const STAGE4_POINT_SLICES: &[Stage4PointSlicePlan] = &[
    Stage4PointSlicePlan { symbol: "stage4.registers_read_write.point.RdInc", source: "stage4.registers_read_write.instance", offset: 7, length: 18, input: "stage4.registers_read_write.instance" },
    Stage4PointSlicePlan { symbol: "stage4.field_registers_read_write.point.FieldRdInc", source: "stage4.field_registers_read_write.instance", offset: 4, length: 18, input: "stage4.field_registers_read_write.instance" },
    Stage4PointSlicePlan { symbol: "stage4.ram_val_check.point.RamAddress", source: "stage4.input.stage2.RamVal", offset: 0, length: 14, input: "stage4.input.stage2.RamVal" },
];

pub const STAGE4_POINT_CONCAT_0_INPUTS: &[&str] = &[
    "stage4.ram_val_check.point.RamAddress",
    "stage4.ram_val_check.instance",
];

pub const STAGE4_POINT_CONCATS: &[Stage4PointConcatPlan] = &[
    Stage4PointConcatPlan { symbol: "stage4.ram_val_check.point.RamRa", layout: "address_then_cycle", arity: 32, inputs: STAGE4_POINT_CONCAT_0_INPUTS },
];
pub const STAGE4_OPENING_CLAIMS: &[Stage4OpeningClaimPlan] = &[
    Stage4OpeningClaimPlan { symbol: "stage4.registers_read_write.opening.RegistersVal", oracle: "RegistersVal", domain: "jolt.stage4_registers_rw_domain", point_arity: 25, claim_kind: "virtual", point_source: "stage4.registers_read_write.instance", eval_source: "stage4.registers_read_write.eval.RegistersVal" },
    Stage4OpeningClaimPlan { symbol: "stage4.registers_read_write.opening.Rs1Ra", oracle: "Rs1Ra", domain: "jolt.stage4_registers_rw_domain", point_arity: 25, claim_kind: "virtual", point_source: "stage4.registers_read_write.instance", eval_source: "stage4.registers_read_write.eval.Rs1Ra" },
    Stage4OpeningClaimPlan { symbol: "stage4.registers_read_write.opening.Rs2Ra", oracle: "Rs2Ra", domain: "jolt.stage4_registers_rw_domain", point_arity: 25, claim_kind: "virtual", point_source: "stage4.registers_read_write.instance", eval_source: "stage4.registers_read_write.eval.Rs2Ra" },
    Stage4OpeningClaimPlan { symbol: "stage4.registers_read_write.opening.RdWa", oracle: "RdWa", domain: "jolt.stage4_registers_rw_domain", point_arity: 25, claim_kind: "virtual", point_source: "stage4.registers_read_write.instance", eval_source: "stage4.registers_read_write.eval.RdWa" },
    Stage4OpeningClaimPlan { symbol: "stage4.registers_read_write.opening.RdInc", oracle: "RdInc", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "committed", point_source: "stage4.registers_read_write.point.RdInc", eval_source: "stage4.registers_read_write.eval.RdInc" },
    Stage4OpeningClaimPlan { symbol: "stage4.field_registers_read_write.opening.FieldRegistersVal", oracle: "FieldRegistersVal", domain: "jolt.stage4_field_registers_rw_domain", point_arity: 22, claim_kind: "virtual", point_source: "stage4.field_registers_read_write.instance", eval_source: "stage4.field_registers_read_write.eval.FieldRegistersVal" },
    Stage4OpeningClaimPlan { symbol: "stage4.field_registers_read_write.opening.FieldRs1Ra", oracle: "FieldRs1Ra", domain: "jolt.stage4_field_registers_rw_domain", point_arity: 22, claim_kind: "virtual", point_source: "stage4.field_registers_read_write.instance", eval_source: "stage4.field_registers_read_write.eval.FieldRs1Ra" },
    Stage4OpeningClaimPlan { symbol: "stage4.field_registers_read_write.opening.FieldRs2Ra", oracle: "FieldRs2Ra", domain: "jolt.stage4_field_registers_rw_domain", point_arity: 22, claim_kind: "virtual", point_source: "stage4.field_registers_read_write.instance", eval_source: "stage4.field_registers_read_write.eval.FieldRs2Ra" },
    Stage4OpeningClaimPlan { symbol: "stage4.field_registers_read_write.opening.FieldRdWa", oracle: "FieldRdWa", domain: "jolt.stage4_field_registers_rw_domain", point_arity: 22, claim_kind: "virtual", point_source: "stage4.field_registers_read_write.instance", eval_source: "stage4.field_registers_read_write.eval.FieldRdWa" },
    Stage4OpeningClaimPlan { symbol: "stage4.field_registers_read_write.opening.FieldRdInc", oracle: "FieldRdInc", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "committed", point_source: "stage4.field_registers_read_write.point.FieldRdInc", eval_source: "stage4.field_registers_read_write.eval.FieldRdInc" },
    Stage4OpeningClaimPlan { symbol: "stage4.ram_val_check.opening.RamRa", oracle: "RamRa", domain: "jolt.stage2_ram_rw_domain", point_arity: 32, claim_kind: "virtual", point_source: "stage4.ram_val_check.point.RamRa", eval_source: "stage4.ram_val_check.eval.RamRa" },
    Stage4OpeningClaimPlan { symbol: "stage4.ram_val_check.opening.RamInc", oracle: "RamInc", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "committed", point_source: "stage4.ram_val_check.instance", eval_source: "stage4.ram_val_check.eval.RamInc" },
];

pub const STAGE4_OPENING_EQUALITIES: &[Stage4OpeningClaimEqualityPlan] = &[
    Stage4OpeningClaimEqualityPlan { symbol: "stage4.registers.rs1_claim_consistency", mode: "point_and_eval", lhs: "stage4.input.stage3.registers.Rs1Value", rhs: "stage4.input.stage3.instruction.Rs1Value" },
    Stage4OpeningClaimEqualityPlan { symbol: "stage4.registers.rs2_claim_consistency", mode: "point_and_eval", lhs: "stage4.input.stage3.registers.Rs2Value", rhs: "stage4.input.stage3.instruction.Rs2Value" },
];

pub const STAGE4_OPENING_BATCH_0_ORDERED_CLAIMS: &[&str] = &[
    "stage4.registers_read_write.opening.RegistersVal",
    "stage4.registers_read_write.opening.Rs1Ra",
    "stage4.registers_read_write.opening.Rs2Ra",
    "stage4.registers_read_write.opening.RdWa",
    "stage4.registers_read_write.opening.RdInc",
    "stage4.field_registers_read_write.opening.FieldRegistersVal",
    "stage4.field_registers_read_write.opening.FieldRs1Ra",
    "stage4.field_registers_read_write.opening.FieldRs2Ra",
    "stage4.field_registers_read_write.opening.FieldRdWa",
    "stage4.field_registers_read_write.opening.FieldRdInc",
    "stage4.ram_val_check.opening.RamRa",
    "stage4.ram_val_check.opening.RamInc",
];

pub const STAGE4_OPENING_BATCH_0_CLAIM_OPERANDS: &[&str] = &[
    "stage4.registers_read_write.opening.RegistersVal",
    "stage4.registers_read_write.opening.Rs1Ra",
    "stage4.registers_read_write.opening.Rs2Ra",
    "stage4.registers_read_write.opening.RdWa",
    "stage4.registers_read_write.opening.RdInc",
    "stage4.field_registers_read_write.opening.FieldRegistersVal",
    "stage4.field_registers_read_write.opening.FieldRs1Ra",
    "stage4.field_registers_read_write.opening.FieldRs2Ra",
    "stage4.field_registers_read_write.opening.FieldRdWa",
    "stage4.field_registers_read_write.opening.FieldRdInc",
    "stage4.ram_val_check.opening.RamRa",
    "stage4.ram_val_check.opening.RamInc",
];

pub const STAGE4_OPENING_BATCHES: &[Stage4OpeningBatchPlan] = &[
    Stage4OpeningBatchPlan { symbol: "stage4.openings", stage: "stage4", proof_slot: "stage4.openings", policy: "jolt_stage4_output_order", count: 12, ordered_claims: STAGE4_OPENING_BATCH_0_ORDERED_CLAIMS, claim_operands: STAGE4_OPENING_BATCH_0_CLAIM_OPERANDS },
];
pub const STAGE4_PROGRAM: Stage4CpuProgramPlan = Stage4CpuProgramPlan {
    role: "prover",
    params: STAGE4_PARAMS,
    steps: STAGE4_PROGRAM_STEPS,
    transcript_squeezes: STAGE4_TRANSCRIPT_SQUEEZES,
    transcript_absorb_bytes: STAGE4_TRANSCRIPT_ABSORB_BYTES,
    opening_inputs: STAGE4_OPENING_INPUTS,
    field_constants: STAGE4_FIELD_CONSTANTS,
    field_exprs: STAGE4_FIELD_EXPRS,
    kernels: STAGE4_KERNELS,
    claims: STAGE4_SUMCHECK_CLAIMS,
    batches: STAGE4_SUMCHECK_BATCHES,
    drivers: STAGE4_SUMCHECK_DRIVERS,
    instance_results: STAGE4_SUMCHECK_INSTANCE_RESULTS,
    evals: STAGE4_SUMCHECK_EVALS,
    point_slices: STAGE4_POINT_SLICES,
    point_concats: STAGE4_POINT_CONCATS,
    opening_claims: STAGE4_OPENING_CLAIMS,
    opening_equalities: STAGE4_OPENING_EQUALITIES,
    opening_batches: STAGE4_OPENING_BATCHES,
};

pub fn execute_stage4_prover<E, T>(
    executor: &mut E,
    transcript: &mut T,
) -> Result<Stage4ExecutionArtifacts<Fr>, Stage4KernelError>
where
    E: Stage4KernelExecutor<Fr>,
    T: Transcript<Challenge = Fr>,
{
    execute_stage4_prover_with_program(&STAGE4_PROGRAM, executor, transcript)
}

pub fn execute_stage4_prover_with_program<E, T>(
    program: &'static Stage4CpuProgramPlan,
    executor: &mut E,
    transcript: &mut T,
) -> Result<Stage4ExecutionArtifacts<Fr>, Stage4KernelError>
where
    E: Stage4KernelExecutor<Fr>,
    T: Transcript<Challenge = Fr>,
{
    execute_stage4_program(program, Stage4ExecutionMode::Prover, executor, transcript)
}
