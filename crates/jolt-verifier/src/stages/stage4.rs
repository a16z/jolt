#![allow(dead_code)]

use super::common::{batch_claims, eval_by_name, find_batch, find_plan, lt_polynomial_eval, reverse_slice};
use jolt_field::{Field, Fr};
use jolt_poly::EqPolynomial;
use jolt_sumcheck::SumcheckError;
use jolt_transcript::{Blake2bTranscript, LabelWithCount, Transcript};

pub type Stage4NamedEval<F> = super::common::StageNamedEval<F>;
pub type Stage4SumcheckOutput<F> = super::common::StageSumcheckOutput<F>;
pub type Stage4ChallengeVector<F> = super::common::StageChallengeVector<F>;
pub type Stage4ExecutionArtifacts<F> = super::common::StageExecutionArtifacts<F>;
pub type Stage4Proof<F> = super::common::StageProof<F>;
pub type Stage4OpeningInputValue<F> = super::common::StageOpeningInputValue<F>;

pub use super::common::{
    FieldConstantPlan as Stage4FieldConstantPlan, FieldExprPlan as Stage4FieldExprPlan,
    KernelPlan as Stage4KernelPlan, OpeningBatchPlan as Stage4OpeningBatchPlan,
    OpeningClaimEqualityPlan as Stage4OpeningClaimEqualityPlan,
    OpeningClaimPlan as Stage4OpeningClaimPlan, OpeningInputPlan as Stage4OpeningInputPlan,
    PointConcatPlan as Stage4PointConcatPlan, PointSlicePlan as Stage4PointSlicePlan,
    ProgramStepPlan as Stage4ProgramStepPlan, StageParams as Stage4Params,
    StageProgramPlanNoPointZeros as Stage4CpuProgramPlan,
    SumcheckBatchPlan as Stage4SumcheckBatchPlan,
    SumcheckClaimPlan as Stage4SumcheckClaimPlan, SumcheckDriverPlan as Stage4SumcheckDriverPlan,
    SumcheckEvalPlan as Stage4SumcheckEvalPlan,
    SumcheckInstanceResultPlan as Stage4SumcheckInstanceResultPlan,
    TranscriptAbsorbBytesPlan as Stage4TranscriptAbsorbBytesPlan,
    TranscriptSqueezePlan as Stage4TranscriptSqueezePlan,
};

pub type DefaultStage4Transcript = Blake2bTranscript<Fr>;
pub type Stage4VerifierProgramPlan = Stage4CpuProgramPlan;

#[derive(Debug)]
pub enum VerifyStage4Error {
    UnexpectedProofCount { expected: usize, got: usize },
    MissingProof { driver: &'static str },
    MissingBatch { driver: &'static str, batch: &'static str },
    MissingClaim { batch: &'static str, claim: &'static str },
    MissingValue { symbol: &'static str },
    InvalidInputLength { input: &'static str, expected: usize, actual: usize },
    InvalidProof { driver: &'static str, reason: &'static str },
    UnsupportedFieldExpr { symbol: &'static str, formula: &'static str },
    UnsupportedRelation { relation: &'static str },
    Sumcheck { driver: &'static str, error: SumcheckError<Fr> },
}

super::common::impl_runtime_plan_error_conversion!(VerifyStage4Error);

pub const STAGE4_PARAMS: Stage4Params = Stage4Params {
    field: "bn254_fr",
    pcs: "dory",
    transcript: "blake2b_transcript",
};
pub const STAGE4_PROGRAM_STEPS: &[Stage4ProgramStepPlan] = &[
    Stage4ProgramStepPlan { kind: "transcript_squeeze", symbol: "stage4.registers_read_write.gamma" },
    Stage4ProgramStepPlan { kind: "transcript_absorb_bytes", symbol: "stage4.ram_val_check.domain_separator" },
    Stage4ProgramStepPlan { kind: "transcript_squeeze", symbol: "stage4.ram_val_check.gamma" },
    Stage4ProgramStepPlan { kind: "sumcheck_driver", symbol: "stage4.sumcheck" },
];

pub const STAGE4_TRANSCRIPT_SQUEEZES: &[Stage4TranscriptSqueezePlan] = &[
    Stage4TranscriptSqueezePlan { symbol: "stage4.registers_read_write.gamma", label: "registers_read_write_gamma", kind: "challenge_scalar", count: 1 },
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
];

pub const STAGE4_FIELD_CONSTANTS: &[Stage4FieldConstantPlan] = &[

];

pub const STAGE4_FIELD_EXPRS: &[Stage4FieldExprPlan] = &[
    Stage4FieldExprPlan { symbol: "stage4.registers_read_write.gamma2", kind: "op", formula: "field.pow:2", operands: "stage4.registers_read_write.gamma" },
    Stage4FieldExprPlan { symbol: "stage4.registers_read_write.term.Rs1Value", kind: "op", formula: "field.mul", operands: "stage4.registers_read_write.gamma|stage4.input.stage3.registers.Rs1Value" },
    Stage4FieldExprPlan { symbol: "stage4.registers_read_write.term.Rs2Value", kind: "op", formula: "field.mul", operands: "stage4.registers_read_write.gamma2|stage4.input.stage3.registers.Rs2Value" },
    Stage4FieldExprPlan { symbol: "stage4.registers_read_write.partial.RdWriteValueRs1Value", kind: "op", formula: "field.add", operands: "stage4.input.stage3.registers.RdWriteValue|stage4.registers_read_write.term.Rs1Value" },
    Stage4FieldExprPlan { symbol: "stage4.registers_read_write.claim_expr", kind: "op", formula: "field.add", operands: "stage4.registers_read_write.partial.RdWriteValueRs1Value|stage4.registers_read_write.term.Rs2Value" },
    Stage4FieldExprPlan { symbol: "stage4.ram_val_check.delta.RamVal", kind: "op", formula: "field.sub", operands: "stage4.input.stage2.RamVal|stage4.input.initial_ram.RamValInit" },
    Stage4FieldExprPlan { symbol: "stage4.ram_val_check.delta.RamValFinal", kind: "op", formula: "field.sub", operands: "stage4.input.stage2.RamValFinal|stage4.input.initial_ram.RamValInit" },
    Stage4FieldExprPlan { symbol: "stage4.ram_val_check.term.RamValFinal", kind: "op", formula: "field.mul", operands: "stage4.ram_val_check.gamma|stage4.ram_val_check.delta.RamValFinal" },
    Stage4FieldExprPlan { symbol: "stage4.ram_val_check.claim_expr", kind: "op", formula: "field.add", operands: "stage4.ram_val_check.delta.RamVal|stage4.ram_val_check.term.RamValFinal" },
];
pub const STAGE4_KERNELS: &[Stage4KernelPlan] = &[

];

pub const STAGE4_SUMCHECK_CLAIMS: &[Stage4SumcheckClaimPlan] = &[
    Stage4SumcheckClaimPlan { symbol: "stage4.registers_read_write.input", stage: "stage4", domain: "jolt.stage4_registers_rw_domain", num_rounds: 25, degree: 3, claim: "stage4.registers_read_write.weighted_values", kernel: None, relation: Some("jolt.stage4.registers_read_write"), claim_value: "stage4.registers_read_write.claim_expr", input_openings: "stage4.input.stage3.registers.RdWriteValue|stage4.input.stage3.registers.Rs1Value|stage4.input.stage3.registers.Rs2Value" },
    Stage4SumcheckClaimPlan { symbol: "stage4.ram_val_check.input", stage: "stage4", domain: "jolt.trace_domain", num_rounds: 18, degree: 3, claim: "stage4.ram_val_check.weighted_values", kernel: None, relation: Some("jolt.stage4.ram_val_check"), claim_value: "stage4.ram_val_check.claim_expr", input_openings: "stage4.input.stage2.RamVal|stage4.input.stage2.RamValFinal|stage4.input.initial_ram.RamValInit" },
];
pub const STAGE4_SUMCHECK_BATCH_0_ROUND_SCHEDULE: &[usize] = &[
    18,
    7,
];

pub const STAGE4_SUMCHECK_BATCHES: &[Stage4SumcheckBatchPlan] = &[
    Stage4SumcheckBatchPlan { symbol: "stage4.batch", stage: "stage4", proof_slot: "stage4.sumcheck", policy: "jolt_core_stage4_aligned", count: 2, ordered_claims: "stage4.registers_read_write.input|stage4.ram_val_check.input", claim_operands: "stage4.registers_read_write.input|stage4.ram_val_check.input", claim_label: "sumcheck_claim", round_label: "sumcheck_poly", round_schedule: STAGE4_SUMCHECK_BATCH_0_ROUND_SCHEDULE },
];
pub const STAGE4_SUMCHECK_DRIVER_0_ROUND_SCHEDULE: &[usize] = &[
    18,
    7,
];

pub const STAGE4_SUMCHECK_DRIVERS: &[Stage4SumcheckDriverPlan] = &[
    Stage4SumcheckDriverPlan { symbol: "stage4.sumcheck", stage: "stage4", proof_slot: "stage4.sumcheck", kernel: None, relation: Some("jolt.stage4.batched"), batch: "stage4.batch", policy: "jolt_core_stage4_aligned", round_schedule: STAGE4_SUMCHECK_DRIVER_0_ROUND_SCHEDULE, claim_label: "sumcheck_claim", round_label: "sumcheck_poly", num_rounds: 25, degree: 3 },
];
pub const STAGE4_SUMCHECK_INSTANCE_RESULTS: &[Stage4SumcheckInstanceResultPlan] = &[
    Stage4SumcheckInstanceResultPlan { symbol: "stage4.registers_read_write.instance", source: "stage4.sumcheck", claim: "stage4.registers_read_write.input", relation: "jolt.stage4.registers_read_write", index: 0, point_arity: 25, num_rounds: 25, round_offset: 0, point_order: "stage4_registers_rw", degree: 3 },
    Stage4SumcheckInstanceResultPlan { symbol: "stage4.ram_val_check.instance", source: "stage4.sumcheck", claim: "stage4.ram_val_check.input", relation: "jolt.stage4.ram_val_check", index: 1, point_arity: 18, num_rounds: 18, round_offset: 7, point_order: "reverse", degree: 3 },
];

pub const STAGE4_SUMCHECK_EVALS: &[Stage4SumcheckEvalPlan] = &[
    Stage4SumcheckEvalPlan { symbol: "stage4.registers_read_write.eval.RegistersVal", source: "stage4.sumcheck", name: "stage4.registers_read_write.eval.RegistersVal", index: 0, oracle: "RegistersVal" },
    Stage4SumcheckEvalPlan { symbol: "stage4.registers_read_write.eval.Rs1Ra", source: "stage4.sumcheck", name: "stage4.registers_read_write.eval.Rs1Ra", index: 1, oracle: "Rs1Ra" },
    Stage4SumcheckEvalPlan { symbol: "stage4.registers_read_write.eval.Rs2Ra", source: "stage4.sumcheck", name: "stage4.registers_read_write.eval.Rs2Ra", index: 2, oracle: "Rs2Ra" },
    Stage4SumcheckEvalPlan { symbol: "stage4.registers_read_write.eval.RdWa", source: "stage4.sumcheck", name: "stage4.registers_read_write.eval.RdWa", index: 3, oracle: "RdWa" },
    Stage4SumcheckEvalPlan { symbol: "stage4.registers_read_write.eval.RdInc", source: "stage4.sumcheck", name: "stage4.registers_read_write.eval.RdInc", index: 4, oracle: "RdInc" },
    Stage4SumcheckEvalPlan { symbol: "stage4.ram_val_check.eval.RamRa", source: "stage4.sumcheck", name: "stage4.ram_val_check.eval.RamRa", index: 0, oracle: "RamRa" },
    Stage4SumcheckEvalPlan { symbol: "stage4.ram_val_check.eval.RamInc", source: "stage4.sumcheck", name: "stage4.ram_val_check.eval.RamInc", index: 1, oracle: "RamInc" },
];

pub const STAGE4_POINT_SLICES: &[Stage4PointSlicePlan] = &[
    Stage4PointSlicePlan { symbol: "stage4.registers_read_write.point.RdInc", source: "stage4.registers_read_write.instance", offset: 7, length: 18, input: "stage4.registers_read_write.instance" },
    Stage4PointSlicePlan { symbol: "stage4.ram_val_check.point.RamAddress", source: "stage4.input.stage2.RamVal", offset: 0, length: 14, input: "stage4.input.stage2.RamVal" },
];

pub const STAGE4_POINT_CONCATS: &[Stage4PointConcatPlan] = &[
    Stage4PointConcatPlan { symbol: "stage4.ram_val_check.point.RamRa", layout: "address_then_cycle", arity: 32, inputs: "stage4.ram_val_check.point.RamAddress|stage4.ram_val_check.instance" },
];
pub const STAGE4_OPENING_CLAIMS: &[Stage4OpeningClaimPlan] = &[
    Stage4OpeningClaimPlan { symbol: "stage4.registers_read_write.opening.RegistersVal", oracle: "RegistersVal", domain: "jolt.stage4_registers_rw_domain", point_arity: 25, claim_kind: "virtual", point_source: "stage4.registers_read_write.instance", eval_source: "stage4.registers_read_write.eval.RegistersVal" },
    Stage4OpeningClaimPlan { symbol: "stage4.registers_read_write.opening.Rs1Ra", oracle: "Rs1Ra", domain: "jolt.stage4_registers_rw_domain", point_arity: 25, claim_kind: "virtual", point_source: "stage4.registers_read_write.instance", eval_source: "stage4.registers_read_write.eval.Rs1Ra" },
    Stage4OpeningClaimPlan { symbol: "stage4.registers_read_write.opening.Rs2Ra", oracle: "Rs2Ra", domain: "jolt.stage4_registers_rw_domain", point_arity: 25, claim_kind: "virtual", point_source: "stage4.registers_read_write.instance", eval_source: "stage4.registers_read_write.eval.Rs2Ra" },
    Stage4OpeningClaimPlan { symbol: "stage4.registers_read_write.opening.RdWa", oracle: "RdWa", domain: "jolt.stage4_registers_rw_domain", point_arity: 25, claim_kind: "virtual", point_source: "stage4.registers_read_write.instance", eval_source: "stage4.registers_read_write.eval.RdWa" },
    Stage4OpeningClaimPlan { symbol: "stage4.registers_read_write.opening.RdInc", oracle: "RdInc", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "committed", point_source: "stage4.registers_read_write.point.RdInc", eval_source: "stage4.registers_read_write.eval.RdInc" },
    Stage4OpeningClaimPlan { symbol: "stage4.ram_val_check.opening.RamRa", oracle: "RamRa", domain: "jolt.stage2_ram_rw_domain", point_arity: 32, claim_kind: "virtual", point_source: "stage4.ram_val_check.point.RamRa", eval_source: "stage4.ram_val_check.eval.RamRa" },
    Stage4OpeningClaimPlan { symbol: "stage4.ram_val_check.opening.RamInc", oracle: "RamInc", domain: "jolt.trace_domain", point_arity: 18, claim_kind: "committed", point_source: "stage4.ram_val_check.instance", eval_source: "stage4.ram_val_check.eval.RamInc" },
];

pub const STAGE4_OPENING_EQUALITIES: &[Stage4OpeningClaimEqualityPlan] = &[
    Stage4OpeningClaimEqualityPlan { symbol: "stage4.registers.rs1_claim_consistency", mode: "point_and_eval", lhs: "stage4.input.stage3.registers.Rs1Value", rhs: "stage4.input.stage3.instruction.Rs1Value" },
    Stage4OpeningClaimEqualityPlan { symbol: "stage4.registers.rs2_claim_consistency", mode: "point_and_eval", lhs: "stage4.input.stage3.registers.Rs2Value", rhs: "stage4.input.stage3.instruction.Rs2Value" },
];

pub const STAGE4_OPENING_BATCHES: &[Stage4OpeningBatchPlan] = &[
    Stage4OpeningBatchPlan { symbol: "stage4.openings", stage: "stage4", proof_slot: "stage4.openings", policy: "jolt_stage4_output_order", count: 7, ordered_claims: "stage4.registers_read_write.opening.RegistersVal|stage4.registers_read_write.opening.Rs1Ra|stage4.registers_read_write.opening.Rs2Ra|stage4.registers_read_write.opening.RdWa|stage4.registers_read_write.opening.RdInc|stage4.ram_val_check.opening.RamRa|stage4.ram_val_check.opening.RamInc", claim_operands: "stage4.registers_read_write.opening.RegistersVal|stage4.registers_read_write.opening.Rs1Ra|stage4.registers_read_write.opening.Rs2Ra|stage4.registers_read_write.opening.RdWa|stage4.registers_read_write.opening.RdInc|stage4.ram_val_check.opening.RamRa|stage4.ram_val_check.opening.RamInc" },
];
pub const STAGE4_PROGRAM: Stage4VerifierProgramPlan = Stage4CpuProgramPlan {
    role: "verifier",
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

pub fn verify_stage4<T>(
    proof: &Stage4Proof<Fr>,
    opening_inputs: &[Stage4OpeningInputValue<Fr>],
    transcript: &mut T,
) -> Result<Stage4ExecutionArtifacts<Fr>, VerifyStage4Error>
where
    T: Transcript<Challenge = Fr>,
{
    verify_stage4_with_program(&STAGE4_PROGRAM, proof, opening_inputs, transcript)
}

pub fn verify_stage4_with_program<T>(
    program: &'static Stage4VerifierProgramPlan,
    proof: &Stage4Proof<Fr>,
    opening_inputs: &[Stage4OpeningInputValue<Fr>],
    transcript: &mut T,
) -> Result<Stage4ExecutionArtifacts<Fr>, VerifyStage4Error>
where
    T: Transcript<Challenge = Fr>,
{
    if proof.sumchecks.len() != program.drivers.len() {
        return Err(VerifyStage4Error::UnexpectedProofCount {
            expected: program.drivers.len(),
            got: proof.sumchecks.len(),
        });
    }
    let mut store =
        super::common::ValueStore::with_opening_inputs(opening_inputs, program.opening_inputs)?;
    store.seed_constants(program.field_constants);
    let mut artifacts = Stage4ExecutionArtifacts::default();
    for step in program.steps {
        match step.kind {
            "transcript_squeeze" => {
                let squeeze =
                    find_plan(program.transcript_squeezes, step.symbol).ok_or(VerifyStage4Error::MissingValue {
                        symbol: step.symbol,
                    })?;
                verify_stage4_squeeze(program, squeeze, &mut store, transcript, &mut artifacts)?;
            }
            "transcript_absorb_bytes" => {
                let absorb = find_plan(program.transcript_absorb_bytes, step.symbol).ok_or(
                    VerifyStage4Error::MissingValue {
                        symbol: step.symbol,
                    },
                )?;
                absorb_stage4_bytes(absorb, transcript);
            }
            "sumcheck_driver" => {
                let driver =
                    find_plan(program.drivers, step.symbol).ok_or(VerifyStage4Error::MissingProof {
                        driver: step.symbol,
                    })?;
                verify_stage4_driver(program, driver, proof, &mut store, transcript, &mut artifacts)?;
            }
            _ => {
                return Err(VerifyStage4Error::InvalidProof {
                    driver: step.symbol,
                    reason: "unsupported stage4 program step",
                });
            }
        }
    }
    artifacts
        .opening_batches
        .extend(program.opening_batches.iter());
    Ok(artifacts)
}

pub fn stage4_verifier_program() -> &'static Stage4VerifierProgramPlan {
    &STAGE4_PROGRAM
}

fn verify_stage4_squeeze<T>(
    program: &'static Stage4VerifierProgramPlan,
    squeeze: &'static Stage4TranscriptSqueezePlan,
    store: &mut super::common::ValueStore<Fr>,
    transcript: &mut T,
    artifacts: &mut Stage4ExecutionArtifacts<Fr>,
) -> Result<(), VerifyStage4Error>
where
    T: Transcript<Challenge = Fr>,
{
    let values = transcript.challenge_vector(squeeze.count);
    store.observe_challenge_vector(squeeze, &values, |input, expected, actual| {
        VerifyStage4Error::InvalidInputLength {
            input,
            expected,
            actual,
        }
    })?;
    store
        .evaluate_available_field_exprs(program.field_exprs, super::common::evaluate_field_expr)
        .map_err(VerifyStage4Error::from)?;
    artifacts.challenge_vectors.push(Stage4ChallengeVector {
        symbol: squeeze.symbol,
        values,
    });
    Ok(())
}

fn absorb_stage4_bytes<T>(absorb: &'static Stage4TranscriptAbsorbBytesPlan, transcript: &mut T)
where
    T: Transcript<Challenge = Fr>,
{
    transcript.append(&LabelWithCount(
        absorb.label.as_bytes(),
        absorb.payload.len() as u64,
    ));
    transcript.append_bytes(absorb.payload.as_bytes());
}

fn verify_stage4_driver<T>(
    program: &'static Stage4VerifierProgramPlan,
    driver: &'static Stage4SumcheckDriverPlan,
    proof: &Stage4Proof<Fr>,
    store: &mut super::common::ValueStore<Fr>,
    transcript: &mut T,
    artifacts: &mut Stage4ExecutionArtifacts<Fr>,
) -> Result<(), VerifyStage4Error>
where
    T: Transcript<Challenge = Fr>,
{
    let proof = proof
        .sumchecks
        .get(artifacts.sumchecks.len())
        .ok_or(VerifyStage4Error::MissingProof {
            driver: driver.symbol,
        })?;
    let relation = driver.relation.unwrap_or("<missing>");
    let output = match relation {
        "jolt.stage4.batched" => {
            verify_batched_stage4(program, driver, proof, store, transcript)?
        }
        _ => return Err(VerifyStage4Error::UnsupportedRelation { relation }),
    };
    artifacts.sumchecks.push(output);
    Ok(())
}

fn verify_batched_stage4<T>(
    program: &'static Stage4VerifierProgramPlan,
    driver: &'static Stage4SumcheckDriverPlan,
    proof: &Stage4SumcheckOutput<Fr>,
    store: &mut super::common::ValueStore<Fr>,
    transcript: &mut T,
) -> Result<Stage4SumcheckOutput<Fr>, VerifyStage4Error>
where
    T: Transcript<Challenge = Fr>,
{
    super::common::verify_batched_sumcheck(
        driver,
        proof,
        program.claims,
        program.batches,
        program.field_exprs,
        program.opening_inputs,
        program.opening_claims,
        program.opening_batches,
        store,
        transcript,
        |store, evals, point, batching_coeffs| {
            expected_batched_output_claim(program, driver, store, evals, point, batching_coeffs)
        },
        |store, verified| observe_stage4_sumcheck_output(program, store, verified),
        |driver, error| VerifyStage4Error::Sumcheck { driver, error },
    )
}

fn observe_stage4_sumcheck_output<F: Field>(
    program: &'static Stage4VerifierProgramPlan,
    store: &mut super::common::ValueStore<F>,
    output: &Stage4SumcheckOutput<F>,
) -> Result<(), VerifyStage4Error> {
    store.observe_sumcheck_output(
        program.instance_results,
        program.evals,
        output,
        |instance, mut point| {
            match instance.point_order {
                "as_is" => {}
                "reverse" => point.reverse(),
                "stage4_registers_rw" => {
                    point = normalize_stage4_registers_rw_point(program, output.driver, &point)?;
                }
                _ => {
                    return Err(VerifyStage4Error::InvalidProof {
                        driver: output.driver,
                        reason: "unsupported point order",
                    });
                }
            }
            Ok(point)
        },
        |input, expected, actual| VerifyStage4Error::InvalidInputLength {
            input,
            expected,
            actual,
        },
        |symbol| VerifyStage4Error::MissingValue { symbol },
    )?;
    store.evaluate_available_points(
        program.point_slices,
        program.point_concats,
        |input, expected, actual| VerifyStage4Error::InvalidInputLength {
            input,
            expected,
            actual,
        },
    )?;
    store
        .evaluate_available_field_exprs(program.field_exprs, super::common::evaluate_field_expr)
        .map_err(VerifyStage4Error::from)?;
    store.verify_opening_equalities(
        program.opening_equalities,
        |driver, reason| VerifyStage4Error::InvalidProof { driver, reason },
        |symbol| VerifyStage4Error::MissingValue { symbol },
    )
}

fn expected_batched_output_claim(
    program: &'static Stage4VerifierProgramPlan,
    driver: &'static Stage4SumcheckDriverPlan,
    store: &super::common::ValueStore<Fr>,
    evals: &[Stage4NamedEval<Fr>],
    point: &[Fr],
    batching_coeffs: &[Fr],
) -> Result<Fr, VerifyStage4Error> {
    let batch = find_batch(program.batches, driver.symbol, driver.batch)?;
    let claims = batch_claims(program.claims, batch)?;
    let mut expected = Fr::from_u64(0);
    for (claim, coefficient) in claims.iter().zip(batching_coeffs) {
        let instance = program
            .instance_results
            .iter()
            .find(|instance| instance.claim == claim.symbol && instance.source == driver.symbol)
            .ok_or(VerifyStage4Error::MissingClaim {
                batch: batch.symbol,
                claim: claim.symbol,
            })?;
        let local_point = point
            .get(instance.round_offset..instance.round_offset + instance.num_rounds)
            .ok_or(VerifyStage4Error::InvalidInputLength {
                input: instance.symbol,
                expected: instance.round_offset + instance.num_rounds,
                actual: point.len(),
            })?;
        let relation = claim.relation.unwrap_or("<missing>");
        let value = match relation {
            "jolt.stage4.registers_read_write" => {
                expected_registers_read_write(store, evals, local_point)?
            }
            "jolt.stage4.ram_val_check" => {
                expected_ram_val_check(store, evals, local_point)?
            }
            _ => return Err(VerifyStage4Error::UnsupportedRelation { relation }),
        };
        expected += *coefficient * value;
    }
    Ok(expected)
}

fn expected_registers_read_write(
    store: &super::common::ValueStore<Fr>,
    evals: &[Stage4NamedEval<Fr>],
    local_point: &[Fr],
) -> Result<Fr, VerifyStage4Error> {
    let trace_point = super::common::store_point(store, "stage4.input.stage3.registers.RdWriteValue")?;
    let r_cycle = normalize_stage4_registers_rw_cycle_point(
        local_point,
        trace_point.len(),
        "stage4.registers_read_write.instance",
    )?;
    let eq_eval = EqPolynomial::<Fr>::mle(&r_cycle, trace_point);
    let registers_val = eval_by_name(
        evals,
        "stage4.registers_read_write.eval.RegistersVal",
    )?;
    let rs1_ra = eval_by_name(evals, "stage4.registers_read_write.eval.Rs1Ra")?;
    let rs2_ra = eval_by_name(evals, "stage4.registers_read_write.eval.Rs2Ra")?;
    let rd_wa = eval_by_name(evals, "stage4.registers_read_write.eval.RdWa")?;
    let rd_inc = eval_by_name(evals, "stage4.registers_read_write.eval.RdInc")?;
    let gamma = super::common::store_scalar(store, "stage4.registers_read_write.gamma")?;
    Ok(eq_eval
        * (rd_wa * (registers_val + rd_inc)
            + gamma * (rs1_ra * registers_val + gamma * rs2_ra * registers_val)))
}

fn expected_ram_val_check(
    store: &super::common::ValueStore<Fr>,
    evals: &[Stage4NamedEval<Fr>],
    local_point: &[Fr],
) -> Result<Fr, VerifyStage4Error> {
    let ram_val_point = super::common::store_point(store, "stage4.input.stage2.RamVal")?;
    let r_cycle_prime = reverse_slice(local_point);
    let r_cycle = suffix_point(
        ram_val_point,
        r_cycle_prime.len(),
        "stage4.input.stage2.RamVal",
    )?;
    let lt_eval = lt_polynomial_eval(&r_cycle_prime, r_cycle);
    let gamma = super::common::store_scalar(store, "stage4.ram_val_check.gamma")?;
    let ram_ra = eval_by_name(evals, "stage4.ram_val_check.eval.RamRa")?;
    let ram_inc = eval_by_name(evals, "stage4.ram_val_check.eval.RamInc")?;
    Ok(ram_inc * ram_ra * (lt_eval + gamma))
}

fn suffix_point<'a>(
    point: &'a [Fr],
    length: usize,
    input: &'static str,
) -> Result<&'a [Fr], VerifyStage4Error> {
    point
        .get(point.len().saturating_sub(length)..)
        .filter(|suffix| suffix.len() == length)
        .ok_or(VerifyStage4Error::InvalidInputLength {
            input,
            expected: length,
            actual: point.len(),
        })
}

fn normalize_stage4_registers_rw_point<F: Field>(
    program: &'static Stage4VerifierProgramPlan,
    driver: &'static str,
    point: &[F],
) -> Result<Vec<F>, VerifyStage4Error> {
    let driver_plan = find_plan(program.drivers, driver).ok_or(VerifyStage4Error::MissingProof {
        driver,
    })?;
    if driver_plan.round_schedule.len() != 2 {
        return Err(VerifyStage4Error::InvalidProof {
            driver,
            reason: "stage4 registers point normalization requires [cycle, address] schedule",
        });
    }
    let cycle_rounds = driver_plan.round_schedule[0];
    let address_rounds = driver_plan.round_schedule[1];
    if point.len() != cycle_rounds + address_rounds {
        return Err(VerifyStage4Error::InvalidInputLength {
            input: "stage4.registers_read_write.instance",
            expected: cycle_rounds + address_rounds,
            actual: point.len(),
        });
    }
    let (cycle, address) = point.split_at(cycle_rounds);
    Ok(address
        .iter()
        .rev()
        .copied()
        .chain(cycle.iter().rev().copied())
        .collect())
}

fn normalize_stage4_registers_rw_cycle_point<F: Field>(
    point: &[F],
    cycle_rounds: usize,
    input: &'static str,
) -> Result<Vec<F>, VerifyStage4Error> {
    let cycle = point
        .get(..cycle_rounds)
        .filter(|cycle| cycle.len() == cycle_rounds)
        .ok_or(VerifyStage4Error::InvalidInputLength {
            input,
            expected: cycle_rounds,
            actual: point.len(),
        })?;
    Ok(cycle.iter().rev().copied().collect())
}
