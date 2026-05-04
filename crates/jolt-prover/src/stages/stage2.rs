#![allow(dead_code)]

use jolt_field::Fr;
use jolt_kernels::stage2::{execute_stage2_program, Stage2CpuProgramPlan, Stage2ExecutionArtifacts, Stage2ExecutionMode, Stage2FieldConstantPlan, Stage2FieldExprPlan, Stage2KernelError, Stage2KernelExecutor, Stage2KernelPlan, Stage2OpeningBatchPlan, Stage2OpeningClaimPlan, Stage2OpeningInputPlan, Stage2Params, Stage2PointConcatPlan, Stage2PointSlicePlan, Stage2ProgramStepPlan, Stage2SumcheckBatchPlan, Stage2SumcheckClaimPlan, Stage2SumcheckDriverPlan, Stage2SumcheckEvalPlan, Stage2SumcheckInstanceResultPlan, Stage2TranscriptSqueezePlan};
use jolt_transcript::{Blake2bTranscript, Transcript};

pub type DefaultStage2Transcript = Blake2bTranscript<Fr>;

pub const STAGE2_PARAMS: Stage2Params = Stage2Params {
    field: "bn254_fr",
    pcs: "dory",
    transcript: "blake2b_transcript",
};
pub const STAGE2_PROGRAM_STEPS: &[Stage2ProgramStepPlan] = &[
    Stage2ProgramStepPlan { kind: "transcript_squeeze", symbol: "stage2.product_virtual.tau_high" },
    Stage2ProgramStepPlan { kind: "sumcheck_driver", symbol: "stage2.product_virtual.uniskip.sumcheck" },
    Stage2ProgramStepPlan { kind: "transcript_squeeze", symbol: "stage2.ram_read_write.gamma" },
    Stage2ProgramStepPlan { kind: "transcript_squeeze", symbol: "stage2.instruction_lookup.gamma" },
    Stage2ProgramStepPlan { kind: "transcript_squeeze", symbol: "stage2.ram_output.r_address" },
    Stage2ProgramStepPlan { kind: "sumcheck_driver", symbol: "stage2.sumcheck" },
];

pub const STAGE2_TRANSCRIPT_SQUEEZES: &[Stage2TranscriptSqueezePlan] = &[
    Stage2TranscriptSqueezePlan { symbol: "stage2.product_virtual.tau_high", label: "product_virtual_tau_high", kind: "challenge_scalar", count: 1 },
    Stage2TranscriptSqueezePlan { symbol: "stage2.ram_read_write.gamma", label: "ram_read_write_gamma", kind: "challenge_scalar", count: 1 },
    Stage2TranscriptSqueezePlan { symbol: "stage2.instruction_lookup.gamma", label: "instruction_lookup_gamma", kind: "challenge_scalar", count: 1 },
    Stage2TranscriptSqueezePlan { symbol: "stage2.ram_output.r_address", label: "ram_output_r_address", kind: "challenge_vector", count: 16 },
];

pub const STAGE2_OPENING_INPUTS: &[Stage2OpeningInputPlan] = &[
    Stage2OpeningInputPlan { symbol: "stage2.input.stage1.Product", source_stage: "stage1", source_claim: "stage1.outer_remaining.opening.Product", oracle: "Product", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage2OpeningInputPlan { symbol: "stage2.input.stage1.ShouldBranch", source_stage: "stage1", source_claim: "stage1.outer_remaining.opening.ShouldBranch", oracle: "ShouldBranch", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage2OpeningInputPlan { symbol: "stage2.input.stage1.ShouldJump", source_stage: "stage1", source_claim: "stage1.outer_remaining.opening.ShouldJump", oracle: "ShouldJump", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage2OpeningInputPlan { symbol: "stage2.input.stage1.RamReadValue", source_stage: "stage1", source_claim: "stage1.outer_remaining.opening.RamReadValue", oracle: "RamReadValue", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage2OpeningInputPlan { symbol: "stage2.input.stage1.RamWriteValue", source_stage: "stage1", source_claim: "stage1.outer_remaining.opening.RamWriteValue", oracle: "RamWriteValue", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage2OpeningInputPlan { symbol: "stage2.input.stage1.LookupOutput", source_stage: "stage1", source_claim: "stage1.outer_remaining.opening.LookupOutput", oracle: "LookupOutput", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage2OpeningInputPlan { symbol: "stage2.input.stage1.LeftLookupOperand", source_stage: "stage1", source_claim: "stage1.outer_remaining.opening.LeftLookupOperand", oracle: "LeftLookupOperand", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage2OpeningInputPlan { symbol: "stage2.input.stage1.RightLookupOperand", source_stage: "stage1", source_claim: "stage1.outer_remaining.opening.RightLookupOperand", oracle: "RightLookupOperand", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage2OpeningInputPlan { symbol: "stage2.input.stage1.LeftInstructionInput", source_stage: "stage1", source_claim: "stage1.outer_remaining.opening.LeftInstructionInput", oracle: "LeftInstructionInput", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage2OpeningInputPlan { symbol: "stage2.input.stage1.RightInstructionInput", source_stage: "stage1", source_claim: "stage1.outer_remaining.opening.RightInstructionInput", oracle: "RightInstructionInput", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
    Stage2OpeningInputPlan { symbol: "stage2.input.stage1.RamAddress", source_stage: "stage1", source_claim: "stage1.outer_remaining.opening.RamAddress", oracle: "RamAddress", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual" },
];

pub const STAGE2_FIELD_CONSTANTS: &[Stage2FieldConstantPlan] = &[
    Stage2FieldConstantPlan { symbol: "stage2.ram_output.zero", field: "bn254_fr", value: 0 },
];

pub const STAGE2_FIELD_EXPR_OPERANDS_0: &[&str] = &["stage2.product_virtual.tau_high"];

pub const STAGE2_FIELD_EXPR_OPERANDS_1: &[&str] = &[
    "stage2.product_virtual.uniskip.weight.Product",
    "stage2.input.stage1.Product",
];

pub const STAGE2_FIELD_EXPR_OPERANDS_2: &[&str] = &[
    "stage2.product_virtual.uniskip.weight.ShouldBranch",
    "stage2.input.stage1.ShouldBranch",
];

pub const STAGE2_FIELD_EXPR_OPERANDS_3: &[&str] = &[
    "stage2.product_virtual.uniskip.weight.ShouldJump",
    "stage2.input.stage1.ShouldJump",
];

pub const STAGE2_FIELD_EXPR_OPERANDS_4: &[&str] = &[
    "stage2.product_virtual.uniskip.term.Product",
    "stage2.product_virtual.uniskip.term.ShouldBranch",
];

pub const STAGE2_FIELD_EXPR_OPERANDS_5: &[&str] = &[
    "stage2.product_virtual.uniskip.partial.ProductShouldBranch",
    "stage2.product_virtual.uniskip.term.ShouldJump",
];

pub const STAGE2_FIELD_EXPR_OPERANDS_6: &[&str] = &[
    "stage2.ram_read_write.gamma",
    "stage2.input.stage1.RamWriteValue",
];

pub const STAGE2_FIELD_EXPR_OPERANDS_7: &[&str] = &[
    "stage2.input.stage1.RamReadValue",
    "stage2.ram_read_write.term.RamWriteValue",
];

pub const STAGE2_FIELD_EXPR_OPERANDS_8: &[&str] = &[
    "stage2.instruction_lookup.gamma",
    "stage2.instruction_lookup.gamma",
];

pub const STAGE2_FIELD_EXPR_OPERANDS_9: &[&str] = &[
    "stage2.instruction_lookup.gamma2",
    "stage2.instruction_lookup.gamma",
];

pub const STAGE2_FIELD_EXPR_OPERANDS_10: &[&str] = &[
    "stage2.instruction_lookup.gamma2",
    "stage2.instruction_lookup.gamma2",
];

pub const STAGE2_FIELD_EXPR_OPERANDS_11: &[&str] = &[
    "stage2.instruction_lookup.gamma",
    "stage2.input.stage1.LeftLookupOperand",
];

pub const STAGE2_FIELD_EXPR_OPERANDS_12: &[&str] = &[
    "stage2.instruction_lookup.gamma2",
    "stage2.input.stage1.RightLookupOperand",
];

pub const STAGE2_FIELD_EXPR_OPERANDS_13: &[&str] = &[
    "stage2.instruction_lookup.gamma3",
    "stage2.input.stage1.LeftInstructionInput",
];

pub const STAGE2_FIELD_EXPR_OPERANDS_14: &[&str] = &[
    "stage2.instruction_lookup.gamma4",
    "stage2.input.stage1.RightInstructionInput",
];

pub const STAGE2_FIELD_EXPR_OPERANDS_15: &[&str] = &[
    "stage2.input.stage1.LookupOutput",
    "stage2.instruction_lookup.term.LeftLookupOperand",
];

pub const STAGE2_FIELD_EXPR_OPERANDS_16: &[&str] = &[
    "stage2.instruction_lookup.partial.LookupOutputLeftOperand",
    "stage2.instruction_lookup.term.RightLookupOperand",
];

pub const STAGE2_FIELD_EXPR_OPERANDS_17: &[&str] = &[
    "stage2.instruction_lookup.partial.RightOperand",
    "stage2.instruction_lookup.term.LeftInstructionInput",
];

pub const STAGE2_FIELD_EXPR_OPERANDS_18: &[&str] = &[
    "stage2.instruction_lookup.partial.LeftInstructionInput",
    "stage2.instruction_lookup.term.RightInstructionInput",
];

pub const STAGE2_FIELD_EXPRS: &[Stage2FieldExprPlan] = &[
    Stage2FieldExprPlan { symbol: "stage2.product_virtual.uniskip.weight.Product", kind: "op", formula: "poly.lagrange_basis_eval:-1:3:0", operand_names: STAGE2_FIELD_EXPR_OPERANDS_0, operands: STAGE2_FIELD_EXPR_OPERANDS_0 },
    Stage2FieldExprPlan { symbol: "stage2.product_virtual.uniskip.weight.ShouldBranch", kind: "op", formula: "poly.lagrange_basis_eval:-1:3:1", operand_names: STAGE2_FIELD_EXPR_OPERANDS_0, operands: STAGE2_FIELD_EXPR_OPERANDS_0 },
    Stage2FieldExprPlan { symbol: "stage2.product_virtual.uniskip.weight.ShouldJump", kind: "op", formula: "poly.lagrange_basis_eval:-1:3:2", operand_names: STAGE2_FIELD_EXPR_OPERANDS_0, operands: STAGE2_FIELD_EXPR_OPERANDS_0 },
    Stage2FieldExprPlan { symbol: "stage2.product_virtual.uniskip.term.Product", kind: "op", formula: "field.mul", operand_names: STAGE2_FIELD_EXPR_OPERANDS_1, operands: STAGE2_FIELD_EXPR_OPERANDS_1 },
    Stage2FieldExprPlan { symbol: "stage2.product_virtual.uniskip.term.ShouldBranch", kind: "op", formula: "field.mul", operand_names: STAGE2_FIELD_EXPR_OPERANDS_2, operands: STAGE2_FIELD_EXPR_OPERANDS_2 },
    Stage2FieldExprPlan { symbol: "stage2.product_virtual.uniskip.term.ShouldJump", kind: "op", formula: "field.mul", operand_names: STAGE2_FIELD_EXPR_OPERANDS_3, operands: STAGE2_FIELD_EXPR_OPERANDS_3 },
    Stage2FieldExprPlan { symbol: "stage2.product_virtual.uniskip.partial.ProductShouldBranch", kind: "op", formula: "field.add", operand_names: STAGE2_FIELD_EXPR_OPERANDS_4, operands: STAGE2_FIELD_EXPR_OPERANDS_4 },
    Stage2FieldExprPlan { symbol: "stage2.product_virtual.uniskip.claim_expr", kind: "op", formula: "field.add", operand_names: STAGE2_FIELD_EXPR_OPERANDS_5, operands: STAGE2_FIELD_EXPR_OPERANDS_5 },
    Stage2FieldExprPlan { symbol: "stage2.ram_read_write.term.RamWriteValue", kind: "op", formula: "field.mul", operand_names: STAGE2_FIELD_EXPR_OPERANDS_6, operands: STAGE2_FIELD_EXPR_OPERANDS_6 },
    Stage2FieldExprPlan { symbol: "stage2.ram_read_write.claim_expr", kind: "op", formula: "field.add", operand_names: STAGE2_FIELD_EXPR_OPERANDS_7, operands: STAGE2_FIELD_EXPR_OPERANDS_7 },
    Stage2FieldExprPlan { symbol: "stage2.instruction_lookup.gamma2", kind: "op", formula: "field.mul", operand_names: STAGE2_FIELD_EXPR_OPERANDS_8, operands: STAGE2_FIELD_EXPR_OPERANDS_8 },
    Stage2FieldExprPlan { symbol: "stage2.instruction_lookup.gamma3", kind: "op", formula: "field.mul", operand_names: STAGE2_FIELD_EXPR_OPERANDS_9, operands: STAGE2_FIELD_EXPR_OPERANDS_9 },
    Stage2FieldExprPlan { symbol: "stage2.instruction_lookup.gamma4", kind: "op", formula: "field.mul", operand_names: STAGE2_FIELD_EXPR_OPERANDS_10, operands: STAGE2_FIELD_EXPR_OPERANDS_10 },
    Stage2FieldExprPlan { symbol: "stage2.instruction_lookup.term.LeftLookupOperand", kind: "op", formula: "field.mul", operand_names: STAGE2_FIELD_EXPR_OPERANDS_11, operands: STAGE2_FIELD_EXPR_OPERANDS_11 },
    Stage2FieldExprPlan { symbol: "stage2.instruction_lookup.term.RightLookupOperand", kind: "op", formula: "field.mul", operand_names: STAGE2_FIELD_EXPR_OPERANDS_12, operands: STAGE2_FIELD_EXPR_OPERANDS_12 },
    Stage2FieldExprPlan { symbol: "stage2.instruction_lookup.term.LeftInstructionInput", kind: "op", formula: "field.mul", operand_names: STAGE2_FIELD_EXPR_OPERANDS_13, operands: STAGE2_FIELD_EXPR_OPERANDS_13 },
    Stage2FieldExprPlan { symbol: "stage2.instruction_lookup.term.RightInstructionInput", kind: "op", formula: "field.mul", operand_names: STAGE2_FIELD_EXPR_OPERANDS_14, operands: STAGE2_FIELD_EXPR_OPERANDS_14 },
    Stage2FieldExprPlan { symbol: "stage2.instruction_lookup.partial.LookupOutputLeftOperand", kind: "op", formula: "field.add", operand_names: STAGE2_FIELD_EXPR_OPERANDS_15, operands: STAGE2_FIELD_EXPR_OPERANDS_15 },
    Stage2FieldExprPlan { symbol: "stage2.instruction_lookup.partial.RightOperand", kind: "op", formula: "field.add", operand_names: STAGE2_FIELD_EXPR_OPERANDS_16, operands: STAGE2_FIELD_EXPR_OPERANDS_16 },
    Stage2FieldExprPlan { symbol: "stage2.instruction_lookup.partial.LeftInstructionInput", kind: "op", formula: "field.add", operand_names: STAGE2_FIELD_EXPR_OPERANDS_17, operands: STAGE2_FIELD_EXPR_OPERANDS_17 },
    Stage2FieldExprPlan { symbol: "stage2.instruction_lookup.claim_reduction.claim_expr", kind: "op", formula: "field.add", operand_names: STAGE2_FIELD_EXPR_OPERANDS_18, operands: STAGE2_FIELD_EXPR_OPERANDS_18 },
];
pub const STAGE2_KERNELS: &[Stage2KernelPlan] = &[
    Stage2KernelPlan { symbol: "jolt.cpu.stage2.product_virtual.uniskip", relation: "jolt.stage2.product_virtual.uniskip", kind: "sumcheck", backend: "cpu", abi: "jolt_stage2_product_virtual_uniskip" },
    Stage2KernelPlan { symbol: "jolt.cpu.stage2.ram.read_write", relation: "jolt.stage2.ram.read_write", kind: "sumcheck", backend: "cpu", abi: "jolt_stage2_ram_read_write" },
    Stage2KernelPlan { symbol: "jolt.cpu.stage2.product_virtual.remainder", relation: "jolt.stage2.product_virtual.remainder", kind: "sumcheck", backend: "cpu", abi: "jolt_stage2_product_virtual_remainder" },
    Stage2KernelPlan { symbol: "jolt.cpu.stage2.instruction_lookup.claim_reduction", relation: "jolt.stage2.instruction_lookup.claim_reduction", kind: "sumcheck", backend: "cpu", abi: "jolt_stage2_instruction_lookup_claim_reduction" },
    Stage2KernelPlan { symbol: "jolt.cpu.stage2.ram.raf_evaluation", relation: "jolt.stage2.ram.raf_evaluation", kind: "sumcheck", backend: "cpu", abi: "jolt_stage2_ram_raf_evaluation" },
    Stage2KernelPlan { symbol: "jolt.cpu.stage2.ram.output_check", relation: "jolt.stage2.ram.output_check", kind: "sumcheck", backend: "cpu", abi: "jolt_stage2_ram_output_check" },
    Stage2KernelPlan { symbol: "jolt.cpu.stage2.batched", relation: "jolt.stage2.batched", kind: "sumcheck", backend: "cpu", abi: "jolt_stage2_batched" },
];

pub const STAGE2_SUMCHECK_CLAIM_0_INPUT_OPENINGS: &[&str] = &[
    "stage2.input.stage1.Product",
    "stage2.input.stage1.ShouldBranch",
    "stage2.input.stage1.ShouldJump",
];

pub const STAGE2_SUMCHECK_CLAIM_1_INPUT_OPENINGS: &[&str] = &[
    "stage2.input.stage1.RamReadValue",
    "stage2.input.stage1.RamWriteValue",
];

pub const STAGE2_SUMCHECK_CLAIM_2_INPUT_OPENINGS: &[&str] = &["stage2.product_virtual.uniskip.opening.UnivariateSkip"];

pub const STAGE2_SUMCHECK_CLAIM_3_INPUT_OPENINGS: &[&str] = &[
    "stage2.input.stage1.LookupOutput",
    "stage2.input.stage1.LeftLookupOperand",
    "stage2.input.stage1.RightLookupOperand",
    "stage2.input.stage1.LeftInstructionInput",
    "stage2.input.stage1.RightInstructionInput",
];

pub const STAGE2_SUMCHECK_CLAIM_4_INPUT_OPENINGS: &[&str] = &["stage2.input.stage1.RamAddress"];

pub const STAGE2_SUMCHECK_CLAIM_5_INPUT_OPENINGS: &[&str] = &[];

pub const STAGE2_SUMCHECK_CLAIMS: &[Stage2SumcheckClaimPlan] = &[
    Stage2SumcheckClaimPlan { symbol: "stage2.product_virtual.uniskip.input", stage: "stage2", domain: "jolt.stage2_uniskip_domain", num_rounds: 1, degree: 6, claim: "stage2.product_virtual.weighted_stage1_outputs", kernel: "jolt.cpu.stage2.product_virtual.uniskip", claim_value: "stage2.product_virtual.uniskip.claim_expr", input_openings: STAGE2_SUMCHECK_CLAIM_0_INPUT_OPENINGS },
    Stage2SumcheckClaimPlan { symbol: "stage2.ram_read_write.input", stage: "stage2", domain: "jolt.stage2_ram_rw_domain", num_rounds: 32, degree: 3, claim: "stage2.ram_read_write.weighted_values", kernel: "jolt.cpu.stage2.ram.read_write", claim_value: "stage2.ram_read_write.claim_expr", input_openings: STAGE2_SUMCHECK_CLAIM_1_INPUT_OPENINGS },
    Stage2SumcheckClaimPlan { symbol: "stage2.product_virtual.remainder.input", stage: "stage2", domain: "jolt.trace_domain", num_rounds: 16, degree: 3, claim: "stage2.product_virtual.uniskip.opening", kernel: "jolt.cpu.stage2.product_virtual.remainder", claim_value: "stage2.product_virtual.uniskip.eval.UnivariateSkip", input_openings: STAGE2_SUMCHECK_CLAIM_2_INPUT_OPENINGS },
    Stage2SumcheckClaimPlan { symbol: "stage2.instruction_lookup.claim_reduction.input", stage: "stage2", domain: "jolt.trace_domain", num_rounds: 16, degree: 2, claim: "stage2.instruction_lookup.weighted_operands", kernel: "jolt.cpu.stage2.instruction_lookup.claim_reduction", claim_value: "stage2.instruction_lookup.claim_reduction.claim_expr", input_openings: STAGE2_SUMCHECK_CLAIM_3_INPUT_OPENINGS },
    Stage2SumcheckClaimPlan { symbol: "stage2.ram_raf.input", stage: "stage2", domain: "jolt.ram_address_domain", num_rounds: 16, degree: 2, claim: "stage2.ram_raf.ram_address", kernel: "jolt.cpu.stage2.ram.raf_evaluation", claim_value: "stage2.input.stage1.RamAddress", input_openings: STAGE2_SUMCHECK_CLAIM_4_INPUT_OPENINGS },
    Stage2SumcheckClaimPlan { symbol: "stage2.ram_output.input", stage: "stage2", domain: "jolt.ram_address_domain", num_rounds: 16, degree: 3, claim: "zero", kernel: "jolt.cpu.stage2.ram.output_check", claim_value: "stage2.ram_output.zero", input_openings: STAGE2_SUMCHECK_CLAIM_5_INPUT_OPENINGS },
];
pub const STAGE2_SUMCHECK_BATCH_0_ORDERED_CLAIMS: &[&str] = &["stage2.product_virtual.uniskip.input"];

pub const STAGE2_SUMCHECK_BATCH_0_CLAIM_OPERANDS: &[&str] = &["stage2.product_virtual.uniskip.input"];

pub const STAGE2_SUMCHECK_BATCH_0_ROUND_SCHEDULE: &[usize] = &[
    1,
];

pub const STAGE2_SUMCHECK_BATCH_1_ORDERED_CLAIMS: &[&str] = &[
    "stage2.ram_read_write.input",
    "stage2.product_virtual.remainder.input",
    "stage2.instruction_lookup.claim_reduction.input",
    "stage2.ram_raf.input",
    "stage2.ram_output.input",
];

pub const STAGE2_SUMCHECK_BATCH_1_CLAIM_OPERANDS: &[&str] = &[
    "stage2.ram_read_write.input",
    "stage2.product_virtual.remainder.input",
    "stage2.instruction_lookup.claim_reduction.input",
    "stage2.ram_raf.input",
    "stage2.ram_output.input",
];

pub const STAGE2_SUMCHECK_BATCH_1_ROUND_SCHEDULE: &[usize] = &[
    16,
    16,
];

pub const STAGE2_SUMCHECK_BATCHES: &[Stage2SumcheckBatchPlan] = &[
    Stage2SumcheckBatchPlan { symbol: "stage2.product_virtual.uniskip.batch", stage: "stage2", proof_slot: "stage2.product_virtual.uni_skip_first_round", policy: "single_instance", count: 1, ordered_claims: STAGE2_SUMCHECK_BATCH_0_ORDERED_CLAIMS, claim_operands: STAGE2_SUMCHECK_BATCH_0_CLAIM_OPERANDS, claim_label: "uniskip_claim", round_label: "uniskip_poly", round_schedule: STAGE2_SUMCHECK_BATCH_0_ROUND_SCHEDULE },
    Stage2SumcheckBatchPlan { symbol: "stage2.batch", stage: "stage2", proof_slot: "stage2.sumcheck", policy: "jolt_core_stage2_aligned", count: 5, ordered_claims: STAGE2_SUMCHECK_BATCH_1_ORDERED_CLAIMS, claim_operands: STAGE2_SUMCHECK_BATCH_1_CLAIM_OPERANDS, claim_label: "sumcheck_claim", round_label: "sumcheck_poly", round_schedule: STAGE2_SUMCHECK_BATCH_1_ROUND_SCHEDULE },
];
pub const STAGE2_SUMCHECK_DRIVER_0_ROUND_SCHEDULE: &[usize] = &[
    1,
];

pub const STAGE2_SUMCHECK_DRIVER_1_ROUND_SCHEDULE: &[usize] = &[
    16,
    16,
];

pub const STAGE2_SUMCHECK_DRIVERS: &[Stage2SumcheckDriverPlan] = &[
    Stage2SumcheckDriverPlan { symbol: "stage2.product_virtual.uniskip.sumcheck", stage: "stage2", proof_slot: "stage2.product_virtual.uni_skip_first_round", kernel: "jolt.cpu.stage2.product_virtual.uniskip", batch: "stage2.product_virtual.uniskip.batch", policy: "univariate_skip", round_schedule: STAGE2_SUMCHECK_DRIVER_0_ROUND_SCHEDULE, claim_label: "uniskip_claim", round_label: "uniskip_poly", num_rounds: 1, degree: 6 },
    Stage2SumcheckDriverPlan { symbol: "stage2.sumcheck", stage: "stage2", proof_slot: "stage2.sumcheck", kernel: "jolt.cpu.stage2.batched", batch: "stage2.batch", policy: "jolt_core_stage2_aligned", round_schedule: STAGE2_SUMCHECK_DRIVER_1_ROUND_SCHEDULE, claim_label: "sumcheck_claim", round_label: "sumcheck_poly", num_rounds: 32, degree: 3 },
];
pub const STAGE2_SUMCHECK_INSTANCE_RESULTS: &[Stage2SumcheckInstanceResultPlan] = &[
    Stage2SumcheckInstanceResultPlan { symbol: "stage2.product_virtual.uniskip.instance", source: "stage2.product_virtual.uniskip.sumcheck", claim: "stage2.product_virtual.uniskip.input", relation: "jolt.stage2.product_virtual.uniskip", index: 0, point_arity: 1, num_rounds: 1, round_offset: 0, point_order: "as_is", degree: 6 },
    Stage2SumcheckInstanceResultPlan { symbol: "stage2.ram_read_write.instance", source: "stage2.sumcheck", claim: "stage2.ram_read_write.input", relation: "jolt.stage2.ram.read_write", index: 0, point_arity: 32, num_rounds: 32, round_offset: 0, point_order: "as_is", degree: 3 },
    Stage2SumcheckInstanceResultPlan { symbol: "stage2.product_virtual.remainder.instance", source: "stage2.sumcheck", claim: "stage2.product_virtual.remainder.input", relation: "jolt.stage2.product_virtual.remainder", index: 1, point_arity: 16, num_rounds: 16, round_offset: 16, point_order: "reverse", degree: 3 },
    Stage2SumcheckInstanceResultPlan { symbol: "stage2.instruction_lookup.claim_reduction.instance", source: "stage2.sumcheck", claim: "stage2.instruction_lookup.claim_reduction.input", relation: "jolt.stage2.instruction_lookup.claim_reduction", index: 2, point_arity: 16, num_rounds: 16, round_offset: 16, point_order: "reverse", degree: 2 },
    Stage2SumcheckInstanceResultPlan { symbol: "stage2.ram_raf.instance", source: "stage2.sumcheck", claim: "stage2.ram_raf.input", relation: "jolt.stage2.ram.raf_evaluation", index: 3, point_arity: 16, num_rounds: 16, round_offset: 16, point_order: "reverse", degree: 2 },
    Stage2SumcheckInstanceResultPlan { symbol: "stage2.ram_output.instance", source: "stage2.sumcheck", claim: "stage2.ram_output.input", relation: "jolt.stage2.ram.output_check", index: 4, point_arity: 16, num_rounds: 16, round_offset: 16, point_order: "reverse", degree: 3 },
];

pub const STAGE2_SUMCHECK_EVALS: &[Stage2SumcheckEvalPlan] = &[
    Stage2SumcheckEvalPlan { symbol: "stage2.product_virtual.uniskip.eval.UnivariateSkip", source: "stage2.product_virtual.uniskip.sumcheck", name: "stage2.product_virtual.uniskip.eval.UnivariateSkip", index: 0, oracle: "UnivariateSkip" },
    Stage2SumcheckEvalPlan { symbol: "stage2.ram_read_write.eval.RamVal", source: "stage2.sumcheck", name: "stage2.ram_read_write.eval.RamVal", index: 0, oracle: "RamVal" },
    Stage2SumcheckEvalPlan { symbol: "stage2.ram_read_write.eval.RamRa", source: "stage2.sumcheck", name: "stage2.ram_read_write.eval.RamRa", index: 1, oracle: "RamRa" },
    Stage2SumcheckEvalPlan { symbol: "stage2.ram_read_write.eval.RamInc", source: "stage2.sumcheck", name: "stage2.ram_read_write.eval.RamInc", index: 2, oracle: "RamInc" },
    Stage2SumcheckEvalPlan { symbol: "stage2.product_virtual.remainder.eval.LeftInstructionInput", source: "stage2.sumcheck", name: "stage2.product_virtual.remainder.eval.LeftInstructionInput", index: 0, oracle: "LeftInstructionInput" },
    Stage2SumcheckEvalPlan { symbol: "stage2.product_virtual.remainder.eval.RightInstructionInput", source: "stage2.sumcheck", name: "stage2.product_virtual.remainder.eval.RightInstructionInput", index: 1, oracle: "RightInstructionInput" },
    Stage2SumcheckEvalPlan { symbol: "stage2.product_virtual.remainder.eval.OpFlagJump", source: "stage2.sumcheck", name: "stage2.product_virtual.remainder.eval.OpFlagJump", index: 2, oracle: "OpFlagJump" },
    Stage2SumcheckEvalPlan { symbol: "stage2.product_virtual.remainder.eval.OpFlagWriteLookupOutputToRD", source: "stage2.sumcheck", name: "stage2.product_virtual.remainder.eval.OpFlagWriteLookupOutputToRD", index: 3, oracle: "OpFlagWriteLookupOutputToRD" },
    Stage2SumcheckEvalPlan { symbol: "stage2.product_virtual.remainder.eval.LookupOutput", source: "stage2.sumcheck", name: "stage2.product_virtual.remainder.eval.LookupOutput", index: 4, oracle: "LookupOutput" },
    Stage2SumcheckEvalPlan { symbol: "stage2.product_virtual.remainder.eval.InstructionFlagBranch", source: "stage2.sumcheck", name: "stage2.product_virtual.remainder.eval.InstructionFlagBranch", index: 5, oracle: "InstructionFlagBranch" },
    Stage2SumcheckEvalPlan { symbol: "stage2.product_virtual.remainder.eval.NextIsNoop", source: "stage2.sumcheck", name: "stage2.product_virtual.remainder.eval.NextIsNoop", index: 6, oracle: "NextIsNoop" },
    Stage2SumcheckEvalPlan { symbol: "stage2.product_virtual.remainder.eval.OpFlagVirtualInstruction", source: "stage2.sumcheck", name: "stage2.product_virtual.remainder.eval.OpFlagVirtualInstruction", index: 7, oracle: "OpFlagVirtualInstruction" },
    Stage2SumcheckEvalPlan { symbol: "stage2.instruction_lookup.claim_reduction.eval.LookupOutput", source: "stage2.sumcheck", name: "stage2.instruction_lookup.claim_reduction.eval.LookupOutput", index: 0, oracle: "LookupOutput" },
    Stage2SumcheckEvalPlan { symbol: "stage2.instruction_lookup.claim_reduction.eval.LeftLookupOperand", source: "stage2.sumcheck", name: "stage2.instruction_lookup.claim_reduction.eval.LeftLookupOperand", index: 1, oracle: "LeftLookupOperand" },
    Stage2SumcheckEvalPlan { symbol: "stage2.instruction_lookup.claim_reduction.eval.RightLookupOperand", source: "stage2.sumcheck", name: "stage2.instruction_lookup.claim_reduction.eval.RightLookupOperand", index: 2, oracle: "RightLookupOperand" },
    Stage2SumcheckEvalPlan { symbol: "stage2.instruction_lookup.claim_reduction.eval.LeftInstructionInput", source: "stage2.sumcheck", name: "stage2.instruction_lookup.claim_reduction.eval.LeftInstructionInput", index: 3, oracle: "LeftInstructionInput" },
    Stage2SumcheckEvalPlan { symbol: "stage2.instruction_lookup.claim_reduction.eval.RightInstructionInput", source: "stage2.sumcheck", name: "stage2.instruction_lookup.claim_reduction.eval.RightInstructionInput", index: 4, oracle: "RightInstructionInput" },
    Stage2SumcheckEvalPlan { symbol: "stage2.ram_raf.eval.RamRa", source: "stage2.sumcheck", name: "stage2.ram_raf.eval.RamRa", index: 0, oracle: "RamRa" },
    Stage2SumcheckEvalPlan { symbol: "stage2.ram_output.eval.RamValFinal", source: "stage2.sumcheck", name: "stage2.ram_output.eval.RamValFinal", index: 0, oracle: "RamValFinal" },
];

pub const STAGE2_POINT_SLICES: &[Stage2PointSlicePlan] = &[
    Stage2PointSlicePlan { symbol: "stage2.ram_read_write.point.RamInc", source: "stage2.ram_read_write.instance", offset: 16, length: 16, input: "stage2.ram_read_write.instance" },
];

pub const STAGE2_POINT_CONCAT_0_INPUTS: &[&str] = &[
    "stage2.ram_raf.instance",
    "stage2.input.stage1.RamAddress",
];

pub const STAGE2_POINT_CONCATS: &[Stage2PointConcatPlan] = &[
    Stage2PointConcatPlan { symbol: "stage2.ram_raf.point.RamRa", layout: "address_then_cycle", arity: 32, inputs: STAGE2_POINT_CONCAT_0_INPUTS },
];
pub const STAGE2_OPENING_CLAIMS: &[Stage2OpeningClaimPlan] = &[
    Stage2OpeningClaimPlan { symbol: "stage2.product_virtual.uniskip.opening.UnivariateSkip", oracle: "UnivariateSkip", domain: "jolt.stage2_uniskip_domain", point_arity: 1, claim_kind: "virtual", point_source: "stage2.product_virtual.uniskip.instance", eval_source: "stage2.product_virtual.uniskip.eval.UnivariateSkip" },
    Stage2OpeningClaimPlan { symbol: "stage2.ram_read_write.opening.RamVal", oracle: "RamVal", domain: "jolt.stage2_ram_rw_domain", point_arity: 32, claim_kind: "virtual", point_source: "stage2.ram_read_write.instance", eval_source: "stage2.ram_read_write.eval.RamVal" },
    Stage2OpeningClaimPlan { symbol: "stage2.ram_read_write.opening.RamRa", oracle: "RamRa", domain: "jolt.stage2_ram_rw_domain", point_arity: 32, claim_kind: "virtual", point_source: "stage2.ram_read_write.instance", eval_source: "stage2.ram_read_write.eval.RamRa" },
    Stage2OpeningClaimPlan { symbol: "stage2.ram_read_write.opening.RamInc", oracle: "RamInc", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "committed", point_source: "stage2.ram_read_write.point.RamInc", eval_source: "stage2.ram_read_write.eval.RamInc" },
    Stage2OpeningClaimPlan { symbol: "stage2.product_virtual.remainder.opening.LeftInstructionInput", oracle: "LeftInstructionInput", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage2.product_virtual.remainder.instance", eval_source: "stage2.product_virtual.remainder.eval.LeftInstructionInput" },
    Stage2OpeningClaimPlan { symbol: "stage2.product_virtual.remainder.opening.RightInstructionInput", oracle: "RightInstructionInput", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage2.product_virtual.remainder.instance", eval_source: "stage2.product_virtual.remainder.eval.RightInstructionInput" },
    Stage2OpeningClaimPlan { symbol: "stage2.product_virtual.remainder.opening.OpFlagJump", oracle: "OpFlagJump", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage2.product_virtual.remainder.instance", eval_source: "stage2.product_virtual.remainder.eval.OpFlagJump" },
    Stage2OpeningClaimPlan { symbol: "stage2.product_virtual.remainder.opening.OpFlagWriteLookupOutputToRD", oracle: "OpFlagWriteLookupOutputToRD", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage2.product_virtual.remainder.instance", eval_source: "stage2.product_virtual.remainder.eval.OpFlagWriteLookupOutputToRD" },
    Stage2OpeningClaimPlan { symbol: "stage2.product_virtual.remainder.opening.LookupOutput", oracle: "LookupOutput", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage2.product_virtual.remainder.instance", eval_source: "stage2.product_virtual.remainder.eval.LookupOutput" },
    Stage2OpeningClaimPlan { symbol: "stage2.product_virtual.remainder.opening.InstructionFlagBranch", oracle: "InstructionFlagBranch", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage2.product_virtual.remainder.instance", eval_source: "stage2.product_virtual.remainder.eval.InstructionFlagBranch" },
    Stage2OpeningClaimPlan { symbol: "stage2.product_virtual.remainder.opening.NextIsNoop", oracle: "NextIsNoop", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage2.product_virtual.remainder.instance", eval_source: "stage2.product_virtual.remainder.eval.NextIsNoop" },
    Stage2OpeningClaimPlan { symbol: "stage2.product_virtual.remainder.opening.OpFlagVirtualInstruction", oracle: "OpFlagVirtualInstruction", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage2.product_virtual.remainder.instance", eval_source: "stage2.product_virtual.remainder.eval.OpFlagVirtualInstruction" },
    Stage2OpeningClaimPlan { symbol: "stage2.instruction_lookup.claim_reduction.opening.LookupOutput", oracle: "LookupOutput", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage2.instruction_lookup.claim_reduction.instance", eval_source: "stage2.instruction_lookup.claim_reduction.eval.LookupOutput" },
    Stage2OpeningClaimPlan { symbol: "stage2.instruction_lookup.claim_reduction.opening.LeftLookupOperand", oracle: "LeftLookupOperand", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage2.instruction_lookup.claim_reduction.instance", eval_source: "stage2.instruction_lookup.claim_reduction.eval.LeftLookupOperand" },
    Stage2OpeningClaimPlan { symbol: "stage2.instruction_lookup.claim_reduction.opening.RightLookupOperand", oracle: "RightLookupOperand", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage2.instruction_lookup.claim_reduction.instance", eval_source: "stage2.instruction_lookup.claim_reduction.eval.RightLookupOperand" },
    Stage2OpeningClaimPlan { symbol: "stage2.instruction_lookup.claim_reduction.opening.LeftInstructionInput", oracle: "LeftInstructionInput", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage2.instruction_lookup.claim_reduction.instance", eval_source: "stage2.instruction_lookup.claim_reduction.eval.LeftInstructionInput" },
    Stage2OpeningClaimPlan { symbol: "stage2.instruction_lookup.claim_reduction.opening.RightInstructionInput", oracle: "RightInstructionInput", domain: "jolt.trace_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage2.instruction_lookup.claim_reduction.instance", eval_source: "stage2.instruction_lookup.claim_reduction.eval.RightInstructionInput" },
    Stage2OpeningClaimPlan { symbol: "stage2.ram_raf.opening.RamRa", oracle: "RamRa", domain: "jolt.stage2_ram_rw_domain", point_arity: 32, claim_kind: "virtual", point_source: "stage2.ram_raf.point.RamRa", eval_source: "stage2.ram_raf.eval.RamRa" },
    Stage2OpeningClaimPlan { symbol: "stage2.ram_output.opening.RamValFinal", oracle: "RamValFinal", domain: "jolt.ram_address_domain", point_arity: 16, claim_kind: "virtual", point_source: "stage2.ram_output.instance", eval_source: "stage2.ram_output.eval.RamValFinal" },
];

pub const STAGE2_OPENING_BATCH_0_ORDERED_CLAIMS: &[&str] = &[
    "stage2.ram_read_write.opening.RamVal",
    "stage2.ram_read_write.opening.RamRa",
    "stage2.ram_read_write.opening.RamInc",
    "stage2.product_virtual.remainder.opening.LeftInstructionInput",
    "stage2.product_virtual.remainder.opening.RightInstructionInput",
    "stage2.product_virtual.remainder.opening.OpFlagJump",
    "stage2.product_virtual.remainder.opening.OpFlagWriteLookupOutputToRD",
    "stage2.product_virtual.remainder.opening.LookupOutput",
    "stage2.product_virtual.remainder.opening.InstructionFlagBranch",
    "stage2.product_virtual.remainder.opening.NextIsNoop",
    "stage2.product_virtual.remainder.opening.OpFlagVirtualInstruction",
    "stage2.instruction_lookup.claim_reduction.opening.LookupOutput",
    "stage2.instruction_lookup.claim_reduction.opening.LeftLookupOperand",
    "stage2.instruction_lookup.claim_reduction.opening.RightLookupOperand",
    "stage2.instruction_lookup.claim_reduction.opening.LeftInstructionInput",
    "stage2.instruction_lookup.claim_reduction.opening.RightInstructionInput",
    "stage2.ram_raf.opening.RamRa",
    "stage2.ram_output.opening.RamValFinal",
];

pub const STAGE2_OPENING_BATCH_0_CLAIM_OPERANDS: &[&str] = &[
    "stage2.ram_read_write.opening.RamVal",
    "stage2.ram_read_write.opening.RamRa",
    "stage2.ram_read_write.opening.RamInc",
    "stage2.product_virtual.remainder.opening.LeftInstructionInput",
    "stage2.product_virtual.remainder.opening.RightInstructionInput",
    "stage2.product_virtual.remainder.opening.OpFlagJump",
    "stage2.product_virtual.remainder.opening.OpFlagWriteLookupOutputToRD",
    "stage2.product_virtual.remainder.opening.LookupOutput",
    "stage2.product_virtual.remainder.opening.InstructionFlagBranch",
    "stage2.product_virtual.remainder.opening.NextIsNoop",
    "stage2.product_virtual.remainder.opening.OpFlagVirtualInstruction",
    "stage2.instruction_lookup.claim_reduction.opening.LookupOutput",
    "stage2.instruction_lookup.claim_reduction.opening.LeftLookupOperand",
    "stage2.instruction_lookup.claim_reduction.opening.RightLookupOperand",
    "stage2.instruction_lookup.claim_reduction.opening.LeftInstructionInput",
    "stage2.instruction_lookup.claim_reduction.opening.RightInstructionInput",
    "stage2.ram_raf.opening.RamRa",
    "stage2.ram_output.opening.RamValFinal",
];

pub const STAGE2_OPENING_BATCHES: &[Stage2OpeningBatchPlan] = &[
    Stage2OpeningBatchPlan { symbol: "stage2.openings", stage: "stage2", proof_slot: "stage2.openings", policy: "jolt_stage2_output_order", count: 18, ordered_claims: STAGE2_OPENING_BATCH_0_ORDERED_CLAIMS, claim_operands: STAGE2_OPENING_BATCH_0_CLAIM_OPERANDS },
];
pub const STAGE2_PROGRAM: Stage2CpuProgramPlan = Stage2CpuProgramPlan {
    params: STAGE2_PARAMS,
    steps: STAGE2_PROGRAM_STEPS,
    transcript_squeezes: STAGE2_TRANSCRIPT_SQUEEZES,
    opening_inputs: STAGE2_OPENING_INPUTS,
    field_constants: STAGE2_FIELD_CONSTANTS,
    field_exprs: STAGE2_FIELD_EXPRS,
    kernels: STAGE2_KERNELS,
    claims: STAGE2_SUMCHECK_CLAIMS,
    batches: STAGE2_SUMCHECK_BATCHES,
    drivers: STAGE2_SUMCHECK_DRIVERS,
    instance_results: STAGE2_SUMCHECK_INSTANCE_RESULTS,
    evals: STAGE2_SUMCHECK_EVALS,
    point_slices: STAGE2_POINT_SLICES,
    point_concats: STAGE2_POINT_CONCATS,
    opening_claims: STAGE2_OPENING_CLAIMS,
    opening_batches: STAGE2_OPENING_BATCHES,
};

pub fn execute_stage2_prover<E, T>(
    executor: &mut E,
    transcript: &mut T,
) -> Result<Stage2ExecutionArtifacts<Fr>, Stage2KernelError>
where
    E: Stage2KernelExecutor<Fr>,
    T: Transcript<Challenge = Fr>,
{
    execute_stage2_prover_with_program(&STAGE2_PROGRAM, executor, transcript)
}

pub fn execute_stage2_prover_with_program<E, T>(
    program: &'static Stage2CpuProgramPlan,
    executor: &mut E,
    transcript: &mut T,
) -> Result<Stage2ExecutionArtifacts<Fr>, Stage2KernelError>
where
    E: Stage2KernelExecutor<Fr>,
    T: Transcript<Challenge = Fr>,
{
    execute_stage2_program(program, Stage2ExecutionMode::Prover, executor, transcript)
}
