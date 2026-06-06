use jolt_backends::{
    Stage3SpartanSumcheckBackend, SumcheckBackend, SumcheckRegularBatchInstance,
    SumcheckRegularBatchLinearFactor, SumcheckRegularBatchLinearTerm, SumcheckRegularBatchProduct,
    SumcheckRegularBatchState, SumcheckStage3ShiftRow, SumcheckStage3ShiftStateRequest,
};
#[cfg(feature = "zk")]
use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_poly::{
    thread::unsafe_allocate_zero_vec, try_eq_mle, BindingOrder, EqPlusOnePolynomial, EqPolynomial,
    Point, Polynomial, UnivariatePoly,
};
use jolt_sumcheck::{
    append_sumcheck_claim, ClearProof, CompressedLabeledRoundPoly, CompressedSumcheckProof,
    RoundMessage, SumcheckProof,
};
use jolt_transcript::Transcript;
use jolt_verifier::stages::{
    stage1::Stage1ClearOutput,
    stage2::Stage2ClearOutput,
    stage3::inputs::{
        InstructionInputOutputOpeningClaims, RegistersClaimReductionOutputOpeningClaims,
        SpartanShiftOutputOpeningClaims, Stage3Claims,
    },
    stage3::outputs::{
        Stage3ClearOutput, Stage3PublicOutput, VerifiedStage3Batch, VerifiedStage3Sumcheck,
    },
};
use jolt_witness::{
    protocols::jolt_vm::{
        JoltVmNamespace, JoltVmStage3InstructionRegisterRows, JoltVmStage3ShiftRows,
    },
    WitnessProvider,
};
use rayon::prelude::*;

#[cfg(feature = "zk")]
use crate::committed::CommittedSumcheckBuilder;
use crate::ProverError;

#[cfg(feature = "zk")]
use super::output::Stage3CommittedBoundaryOutput;
use super::{
    input::{Stage3ProverConfig, Stage3ProverInput},
    output::{
        stage3_materialized_openings_from_outputs, stage3_output_openings_from_evaluations,
        Stage3InstructionInputMaterializedOpenings, Stage3ProverOutput,
        Stage3RegistersClaimReductionMaterializedOpenings, Stage3RegularBatchExpectedOutputs,
        Stage3RegularBatchInputClaims, Stage3RegularBatchMaterializedOpenings,
        Stage3RegularBatchOutputOpeningClaims, Stage3RegularBatchPrefixOutput,
        Stage3RegularBatchProofOutput, Stage3ShiftMaterializedOpenings,
    },
    request::{
        build_stage3_output_opening_evaluation_request,
        build_stage3_output_opening_materialization_request,
    },
};

const STAGE3_INSTRUCTION_INPUT_DEGREE: usize = 3;
const STAGE3_BATCH_DEGREE: usize = STAGE3_INSTRUCTION_INPUT_DEGREE;
const STAGE3_INSTRUCTION_REGISTER_MATERIALIZE_CHUNK_ROWS: usize = 1 << 14;

#[cfg(feature = "frontier-harness")]
fn timed_stage3<T, E>(label: &'static str, f: impl FnOnce() -> Result<T, E>) -> Result<T, E> {
    let start = std::time::Instant::now();
    let result = f();
    crate::timing::record_stage_timing(label, start.elapsed().as_secs_f64() * 1000.0);
    result
}

#[cfg(not(feature = "frontier-harness"))]
fn timed_stage3<T, E>(_label: &'static str, f: impl FnOnce() -> Result<T, E>) -> Result<T, E> {
    f()
}

#[cfg(feature = "frontier-harness")]
fn timed_stage3_value<T>(label: &'static str, f: impl FnOnce() -> T) -> T {
    let start = std::time::Instant::now();
    let result = f();
    crate::timing::record_stage_timing(label, start.elapsed().as_secs_f64() * 1000.0);
    result
}

#[cfg(not(feature = "frontier-harness"))]
fn timed_stage3_value<T>(_label: &'static str, f: impl FnOnce() -> T) -> T {
    f()
}

#[cfg(feature = "frontier-harness")]
fn timed_stage3_accumulate<T>(accumulator: &mut f64, f: impl FnOnce() -> T) -> T {
    let start = std::time::Instant::now();
    let result = f();
    *accumulator += start.elapsed().as_secs_f64() * 1000.0;
    result
}

#[cfg(not(feature = "frontier-harness"))]
fn timed_stage3_accumulate<T>(_accumulator: &mut f64, f: impl FnOnce() -> T) -> T {
    f()
}

#[cfg(feature = "frontier-harness")]
fn record_stage3_accumulated(label: &'static str, time_ms: f64) {
    crate::timing::record_stage_timing(label, time_ms);
}

#[cfg(not(feature = "frontier-harness"))]
fn record_stage3_accumulated(_label: &'static str, _time_ms: f64) {}

/// Canonical Stage 3 prover entrypoint.
///
/// Mirrors `jolt-verifier/src/stages/stage3/verify.rs` in prover execution order:
/// derive the shift/instruction/registers gammas, prove the regular batched
/// Boolean sumcheck over the three Stage 3 statements, evaluate the output
/// openings, then assemble the verifier-owned `stage3_sumcheck_proof`,
/// [`Stage3Claims`], and [`Stage3ClearOutput`] for Stage 4 and later stages.
///
/// Stage 3 has no field-inline-specific relation, so this single entrypoint is
/// shared across feature modes. ZK committed-boundary assembly is layered on top
/// of the same gamma derivation and statement order.
pub fn prove<F, W, B, T, C>(
    input: Stage3ProverInput<'_, F, W>,
    backend: &mut B,
    transcript: &mut T,
) -> Result<Stage3ProverOutput<F, SumcheckProof<F, C>>, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>
        + JoltVmStage3ShiftRows
        + JoltVmStage3InstructionRegisterRows,
    B: SumcheckBackend<F, JoltVmNamespace> + Stage3SpartanSumcheckBackend<F>,
    T: Transcript<Challenge = F>,
{
    if input.checked.zk {
        return Err(ProverError::InvalidStageRequest {
            reason: "Stage 3 clear prover received ZK checked inputs".to_owned(),
        });
    }
    if input.checked.trace_length != (1usize << input.config.log_t) {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 3 checked trace length {} does not match log_t {}",
                input.checked.trace_length, input.config.log_t
            ),
        });
    }

    let prefix = timed_stage3("stage3.prefix", || {
        derive_stage3_regular_batch_prefix(input.config, input.stage1, input.stage2, transcript)
    })?;
    let batch = timed_stage3("stage3.regular_batch", || {
        prove_stage3_regular_batch_sumcheck::<F, W, B, T, C>(&input, &prefix, transcript, backend)
    })?;

    let opening_point = batch.challenges.iter().rev().copied().collect::<Vec<_>>();
    let output_openings = batch.output_openings.clone().ok_or_else(|| {
        invalid_sumcheck_output("Stage 3 regular batch did not return bound output openings")
    })?;
    let expected = timed_stage3("stage3.expected_outputs", || {
        stage3_expected_outputs(input.stage2, &prefix, &opening_point, &output_openings)
    })?;
    let [shift_coefficient, instruction_coefficient, registers_coefficient] =
        batch.batching_coefficients.as_slice()
    else {
        return Err(invalid_sumcheck_output(
            "Stage 3 batch expected exactly three batching coefficients",
        ));
    };
    let expected_final_claim = *shift_coefficient * expected.shift
        + *instruction_coefficient * expected.instruction_input
        + *registers_coefficient * expected.registers_claim_reduction;
    if batch.output_claim != expected_final_claim {
        return Err(invalid_sumcheck_output(
            "Stage 3 batch final claim did not match output openings",
        ));
    }

    timed_stage3_value("stage3.opening_claim_transcript", || {
        append_stage3_opening_claims(transcript, &output_openings);
    });

    let claims = Stage3Claims {
        shift: output_openings.shift.clone(),
        instruction_input: output_openings.instruction_input.clone(),
        registers_claim_reduction: output_openings.registers_claim_reduction.clone(),
    };
    let public = Stage3PublicOutput {
        challenges: batch.challenges.clone(),
        batching_coefficients: batch.batching_coefficients.clone(),
        shift_gamma: prefix.shift_gamma,
        instruction_gamma: prefix.instruction_gamma,
        registers_gamma: prefix.registers_gamma,
    };
    let verifier_output = Stage3ClearOutput {
        public,
        output_claims: claims.clone(),
        batch: VerifiedStage3Batch {
            batching_coefficients: batch.batching_coefficients.clone(),
            sumcheck_point: Point::high_to_low(batch.challenges.clone()),
            sumcheck_final_claim: batch.output_claim,
            expected_final_claim,
            shift: VerifiedStage3Sumcheck {
                input_claim: prefix.input_claims.shift,
                sumcheck_point: batch.challenges.clone(),
                opening_point: opening_point.clone(),
                expected_output_claim: expected.shift,
            },
            instruction_input: VerifiedStage3Sumcheck {
                input_claim: prefix.input_claims.instruction_input,
                sumcheck_point: batch.challenges.clone(),
                opening_point: opening_point.clone(),
                expected_output_claim: expected.instruction_input,
            },
            registers_claim_reduction: VerifiedStage3Sumcheck {
                input_claim: prefix.input_claims.registers_claim_reduction,
                sumcheck_point: batch.challenges.clone(),
                opening_point,
                expected_output_claim: expected.registers_claim_reduction,
            },
        },
    };

    Ok(Stage3ProverOutput {
        stage3_sumcheck_proof: batch.proof,
        claims,
        verifier_output,
    })
}

#[cfg(feature = "zk")]
pub fn prove_committed_boundary<F, W, B, T, VC>(
    input: Stage3ProverInput<'_, F, W>,
    backend: &mut B,
    transcript: &mut T,
    vc_setup: &VC::Setup,
) -> Result<Stage3CommittedBoundaryOutput<F, VC>, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>
        + JoltVmStage3ShiftRows
        + JoltVmStage3InstructionRegisterRows,
    B: SumcheckBackend<F, JoltVmNamespace> + Stage3SpartanSumcheckBackend<F>,
    T: Transcript<Challenge = F>,
    VC: VectorCommitment<Field = F>,
{
    if !input.checked.zk {
        return Err(ProverError::InvalidStageRequest {
            reason: "Stage 3 committed-boundary prover received transparent checked inputs"
                .to_owned(),
        });
    }
    if input.checked.trace_length != (1usize << input.config.log_t) {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 3 checked trace length {} does not match log_t {}",
                input.checked.trace_length, input.config.log_t
            ),
        });
    }

    let prefix = timed_stage3("stage3.prefix", || {
        derive_stage3_regular_batch_prefix(input.config, input.stage1, input.stage2, transcript)
    })?;
    let shift_request = timed_stage3("stage3.shift_request", || {
        build_stage3_shift_state_request(input.config, input.stage2, &prefix, input.witness)
    })?;
    let mut shift_state = timed_stage3("stage3.shift_materialize", || {
        backend.materialize_sumcheck_stage3_shift_state(&shift_request)
    })?;
    let instances = timed_stage3("stage3.instruction_register_instances", || {
        build_stage3_instruction_registers_regular_batch_instances(
            input.config,
            input.stage2,
            &prefix,
            input.witness,
            backend,
        )
    })?;
    let mut regular_state =
        SumcheckRegularBatchState::new("stage3.instruction_registers_regular_batch", instances);
    let max_num_rounds = input.config.log_t;
    for instance in &regular_state.instances {
        if instance.num_rounds() != max_num_rounds {
            return Err(invalid_sumcheck_output(format!(
                "Stage 3 instance {} has {} rounds, expected {max_num_rounds}",
                instance.label,
                instance.num_rounds()
            )));
        }
    }

    let batching_coefficients = (0..3)
        .map(|_| transcript.challenge_scalar())
        .collect::<Vec<_>>();
    let [shift_coefficient, instruction_coefficient, registers_coefficient] =
        batching_coefficients.as_slice()
    else {
        return Err(invalid_sumcheck_output(
            "Stage 3 batch expected exactly three batching coefficients",
        ));
    };

    let mut shift_claim = prefix.input_claims.shift;
    let mut regular_claims = vec![
        prefix.input_claims.instruction_input,
        prefix.input_claims.registers_claim_reduction,
    ];
    let mut running_claim = *shift_coefficient * shift_claim
        + *instruction_coefficient * regular_claims[0]
        + *registers_coefficient * regular_claims[1];

    let mut builder = CommittedSumcheckBuilder::<F, VC>::new(vc_setup, max_num_rounds)?;
    let mut challenges = Vec::with_capacity(max_num_rounds);
    for round in 0..max_num_rounds {
        let shift_poly = backend.evaluate_sumcheck_stage3_shift_round(&shift_state, shift_claim)?;
        let messages = backend.evaluate_sumcheck_regular_batch_round(
            &mut regular_state,
            round,
            max_num_rounds,
            &regular_claims,
        )?;
        if messages.len() != regular_claims.len() {
            return Err(invalid_sumcheck_output(format!(
                "Stage 3 instruction/register batch round {round} returned {} messages, expected {}",
                messages.len(),
                regular_claims.len()
            )));
        }
        let mut regular_polys = Vec::with_capacity(regular_claims.len());
        for (expected_index, message) in messages.into_iter().enumerate() {
            if message.instance_index != expected_index {
                return Err(invalid_sumcheck_output(format!(
                    "Stage 3 instruction/register batch round {round} returned instance {}, expected {expected_index}",
                    message.instance_index
                )));
            }
            regular_polys.push(message.polynomial);
        }

        let mut batched_poly = &shift_poly * *shift_coefficient;
        batched_poly += &(&regular_polys[0] * *instruction_coefficient);
        batched_poly += &(&regular_polys[1] * *registers_coefficient);
        let round_sum = batched_poly.evaluate(F::zero()) + batched_poly.evaluate(F::one());
        if round_sum != running_claim {
            return Err(invalid_sumcheck_output(format!(
                "Stage 3 regular batch round {round} sumcheck invariant failed"
            )));
        }

        let batched_poly = trim_round_polynomial(batched_poly);
        let challenge = builder.commit_round(&batched_poly, transcript)?;
        running_claim = batched_poly.evaluate(challenge);
        challenges.push(challenge);

        shift_claim = shift_poly.evaluate(challenge);
        for (claim, poly) in regular_claims.iter_mut().zip(regular_polys) {
            *claim = poly.evaluate(challenge);
        }
        backend.bind_sumcheck_stage3_shift_state(&mut shift_state, challenge)?;
        backend.bind_sumcheck_regular_batch_state(
            &mut regular_state,
            round,
            max_num_rounds,
            challenge,
        )?;
    }

    let opening_point = challenges.iter().rev().copied().collect::<Vec<_>>();
    let output_openings = timed_stage3("stage3.bound_output_openings", || {
        stage3_output_openings_from_bound_states(backend, &shift_state, &regular_state)
    })?;
    let expected = timed_stage3("stage3.expected_outputs", || {
        stage3_expected_outputs(input.stage2, &prefix, &opening_point, &output_openings)
    })?;
    let expected_final_claim = *shift_coefficient * expected.shift
        + *instruction_coefficient * expected.instruction_input
        + *registers_coefficient * expected.registers_claim_reduction;
    if running_claim != expected_final_claim {
        return Err(invalid_sumcheck_output(
            "Stage 3 batch final claim did not match output openings",
        ));
    }

    let output_claim_values = stage3_committed_output_claim_values(&output_openings);
    let claims = Stage3Claims {
        shift: output_openings.shift.clone(),
        instruction_input: output_openings.instruction_input.clone(),
        registers_claim_reduction: output_openings.registers_claim_reduction.clone(),
    };
    let public = Stage3PublicOutput {
        challenges: challenges.clone(),
        batching_coefficients: batching_coefficients.clone(),
        shift_gamma: prefix.shift_gamma,
        instruction_gamma: prefix.instruction_gamma,
        registers_gamma: prefix.registers_gamma,
    };
    let verifier_output = Stage3ClearOutput {
        public: public.clone(),
        output_claims: claims,
        batch: VerifiedStage3Batch {
            batching_coefficients: batching_coefficients.clone(),
            sumcheck_point: Point::high_to_low(challenges.clone()),
            sumcheck_final_claim: running_claim,
            expected_final_claim,
            shift: VerifiedStage3Sumcheck {
                input_claim: prefix.input_claims.shift,
                sumcheck_point: challenges.clone(),
                opening_point: opening_point.clone(),
                expected_output_claim: expected.shift,
            },
            instruction_input: VerifiedStage3Sumcheck {
                input_claim: prefix.input_claims.instruction_input,
                sumcheck_point: challenges.clone(),
                opening_point: opening_point.clone(),
                expected_output_claim: expected.instruction_input,
            },
            registers_claim_reduction: VerifiedStage3Sumcheck {
                input_claim: prefix.input_claims.registers_claim_reduction,
                sumcheck_point: challenges.clone(),
                opening_point,
                expected_output_claim: expected.registers_claim_reduction,
            },
        },
    };
    let built = builder.finish(&output_claim_values, transcript)?;
    Ok(Stage3CommittedBoundaryOutput {
        stage3_sumcheck_proof: built.proof,
        public,
        verifier_output,
        output_claim_values,
        committed_witness: built.witness,
    })
}

struct Stage3RegularBatch<F: Field, C> {
    proof: SumcheckProof<F, C>,
    challenges: Vec<F>,
    batching_coefficients: Vec<F>,
    output_claim: F,
    output_openings: Option<Stage3RegularBatchOutputOpeningClaims<F>>,
}

/// Builds the three Stage 3 regular-batch instances (Spartan shift, instruction
/// input, register claim reduction) for the backend sumcheck kernel.
///
/// Each statement is a sum of products, so it maps onto one
/// [`SumcheckRegularBatchInstance`] carrying several product terms. The
/// instance polynomials are bit-reversed so the kernel's `HighToLow` binding
/// reproduces the canonical low-to-high cycle order (matching Stage 2).
#[expect(
    dead_code,
    reason = "dense Stage 3 batch assembly is retained as a parity-audit reference"
)]
fn build_stage3_regular_batch_instances<F, W, B>(
    config: Stage3ProverConfig,
    stage2: &Stage2ClearOutput<F>,
    prefix: &Stage3RegularBatchPrefixOutput<F>,
    witness: &W,
    backend: &mut B,
) -> Result<Vec<SumcheckRegularBatchInstance<F>>, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>,
    B: SumcheckBackend<F, JoltVmNamespace>,
{
    let request = build_stage3_output_opening_materialization_request::<F, W>(witness)?;
    let materializations =
        backend.materialize_sumcheck_views(&request.materializations, witness)?;
    let materialized = stage3_materialized_openings_from_outputs(&request, materializations)?;

    let eq_plus_one_outer = EqPlusOnePolynomial::evals(&stage2.product_uniskip.tau_low, None).1;
    let eq_plus_one_product =
        EqPlusOnePolynomial::evals(&stage2.batch.product_remainder.opening_point, None).1;
    let (eq_product, eq_spartan) = timed_stage3_value("stage3.eq_tables", || {
        let eq_product =
            EqPolynomial::new(stage2.batch.product_remainder.opening_point.clone()).evaluations();
        let eq_spartan = EqPolynomial::new(stage2.product_uniskip.tau_low.clone()).evaluations();
        (eq_product, eq_spartan)
    });

    let shift_gamma = prefix.shift_gamma;
    let shift_gamma2 = shift_gamma * shift_gamma;
    let shift_gamma3 = shift_gamma2 * shift_gamma;
    let shift_gamma4 = shift_gamma3 * shift_gamma;
    let instruction_gamma = prefix.instruction_gamma;
    let registers_gamma = prefix.registers_gamma;
    let registers_gamma2 = registers_gamma * registers_gamma;

    let shift = SumcheckRegularBatchInstance::new_products(
        "stage3.shift",
        prefix.input_claims.shift,
        vec![
            stage3_reversed_polynomial(config, "shift outer eq+1", eq_plus_one_outer)?,
            stage3_reversed_polynomial(config, "shift product eq+1", eq_plus_one_product)?,
            stage3_reversed_polynomial(
                config,
                "shift unexpanded PC",
                materialized.shift.unexpanded_pc,
            )?,
            stage3_reversed_polynomial(config, "shift PC", materialized.shift.pc)?,
            stage3_reversed_polynomial(
                config,
                "shift virtual flag",
                materialized.shift.is_virtual,
            )?,
            stage3_reversed_polynomial(
                config,
                "shift first-in-sequence flag",
                materialized.shift.is_first_in_sequence,
            )?,
            stage3_reversed_polynomial(config, "shift noop flag", materialized.shift.is_noop)?,
        ],
        vec![
            SumcheckRegularBatchProduct::new(
                F::one(),
                vec![
                    batch_factor(vec![batch_term(0, F::one())]),
                    batch_factor(vec![
                        batch_term(2, F::one()),
                        batch_term(3, shift_gamma),
                        batch_term(4, shift_gamma2),
                        batch_term(5, shift_gamma3),
                    ]),
                ],
            ),
            SumcheckRegularBatchProduct::new(
                shift_gamma4,
                vec![
                    batch_factor(vec![batch_term(1, F::one())]),
                    SumcheckRegularBatchLinearFactor::new(F::one(), vec![batch_term(6, neg_one())]),
                ],
            ),
        ],
    );

    let instruction_input = timed_stage3("stage3.build_instruction_instance", || {
        Ok::<_, ProverError>(SumcheckRegularBatchInstance::new_products(
            "stage3.instruction_input",
            prefix.input_claims.instruction_input,
            vec![
                stage3_reversed_polynomial(config, "instruction product eq", eq_product)?,
                stage3_reversed_polynomial(
                    config,
                    "instruction right operand is rs2",
                    materialized.instruction_input.right_operand_is_rs2,
                )?,
                stage3_reversed_polynomial(
                    config,
                    "instruction rs2 value",
                    materialized.instruction_input.rs2_value,
                )?,
                stage3_reversed_polynomial(
                    config,
                    "instruction right operand is imm",
                    materialized.instruction_input.right_operand_is_imm,
                )?,
                stage3_reversed_polynomial(
                    config,
                    "instruction immediate",
                    materialized.instruction_input.imm,
                )?,
                stage3_reversed_polynomial(
                    config,
                    "instruction left operand is rs1",
                    materialized.instruction_input.left_operand_is_rs1,
                )?,
                stage3_reversed_polynomial(
                    config,
                    "instruction rs1 value",
                    materialized.instruction_input.rs1_value,
                )?,
                stage3_reversed_polynomial(
                    config,
                    "instruction left operand is PC",
                    materialized.instruction_input.left_operand_is_pc,
                )?,
                stage3_reversed_polynomial(
                    config,
                    "instruction unexpanded PC",
                    materialized.instruction_input.unexpanded_pc,
                )?,
            ],
            vec![
                SumcheckRegularBatchProduct::new(
                    F::one(),
                    vec![
                        batch_factor(vec![batch_term(0, F::one())]),
                        batch_factor(vec![batch_term(1, F::one())]),
                        batch_factor(vec![batch_term(2, F::one())]),
                    ],
                ),
                SumcheckRegularBatchProduct::new(
                    F::one(),
                    vec![
                        batch_factor(vec![batch_term(0, F::one())]),
                        batch_factor(vec![batch_term(3, F::one())]),
                        batch_factor(vec![batch_term(4, F::one())]),
                    ],
                ),
                SumcheckRegularBatchProduct::new(
                    instruction_gamma,
                    vec![
                        batch_factor(vec![batch_term(0, F::one())]),
                        batch_factor(vec![batch_term(5, F::one())]),
                        batch_factor(vec![batch_term(6, F::one())]),
                    ],
                ),
                SumcheckRegularBatchProduct::new(
                    instruction_gamma,
                    vec![
                        batch_factor(vec![batch_term(0, F::one())]),
                        batch_factor(vec![batch_term(7, F::one())]),
                        batch_factor(vec![batch_term(8, F::one())]),
                    ],
                ),
            ],
        ))
    })?;

    let registers_claim_reduction = SumcheckRegularBatchInstance::new_products(
        "stage3.registers_claim_reduction",
        prefix.input_claims.registers_claim_reduction,
        vec![
            stage3_reversed_polynomial(config, "registers Spartan eq", eq_spartan)?,
            stage3_reversed_polynomial(
                config,
                "registers rd write value",
                materialized.registers_claim_reduction.rd_write_value,
            )?,
            stage3_reversed_polynomial(
                config,
                "registers rs1 value",
                materialized.registers_claim_reduction.rs1_value,
            )?,
            stage3_reversed_polynomial(
                config,
                "registers rs2 value",
                materialized.registers_claim_reduction.rs2_value,
            )?,
        ],
        vec![SumcheckRegularBatchProduct::new(
            F::one(),
            vec![
                batch_factor(vec![batch_term(0, F::one())]),
                batch_factor(vec![
                    batch_term(1, F::one()),
                    batch_term(2, registers_gamma),
                    batch_term(3, registers_gamma2),
                ]),
            ],
        )],
    );

    Ok(vec![shift, instruction_input, registers_claim_reduction])
}

fn build_stage3_shift_state_request<F, W>(
    config: Stage3ProverConfig,
    stage2: &Stage2ClearOutput<F>,
    prefix: &Stage3RegularBatchPrefixOutput<F>,
    witness: &W,
) -> Result<SumcheckStage3ShiftStateRequest<F>, ProverError>
where
    F: Field,
    W: JoltVmStage3ShiftRows,
{
    let rows = timed_stage3("stage3.shift_rows", || {
        witness.stage3_shift_rows(config.log_t)
    })?
    .into_iter()
    .map(|row| {
        SumcheckStage3ShiftRow::new(
            row.unexpanded_pc,
            row.pc,
            row.is_virtual,
            row.is_first_in_sequence,
            row.is_noop,
        )
    })
    .collect();
    Ok(SumcheckStage3ShiftStateRequest::new(
        "stage3.shift.prefix_suffix",
        config.log_t,
        stage2.product_uniskip.tau_low.clone(),
        stage2.batch.product_remainder.opening_point.clone(),
        prefix.shift_gamma,
        rows,
    ))
}

fn stage3_instruction_register_reversed_columns_from_rows<F, W>(
    config: Stage3ProverConfig,
    witness: &W,
    bit_reversed_indices: &[usize],
    eq_product: &[F],
    eq_spartan: &[F],
) -> Result<Vec<Vec<F>>, ProverError>
where
    F: Field,
    W: JoltVmStage3InstructionRegisterRows,
{
    let rows = timed_stage3("stage3.instruction_register_rows", || {
        witness.stage3_instruction_register_rows(config.log_t)
    })?;
    let row_count = checked_stage3_len(config.log_t, "Stage 3 instruction/register trace length")?;
    validate_stage3_row_count(config, rows.len(), "instruction/register row cache")?;
    if bit_reversed_indices.len() != row_count {
        return Err(invalid_sumcheck_output(format!(
            "Stage 3 instruction/register bit-reversal table has {} entries, expected {row_count}",
            bit_reversed_indices.len()
        )));
    }
    if eq_product.len() != row_count || eq_spartan.len() != row_count {
        return Err(invalid_sumcheck_output(format!(
            "Stage 3 instruction/register eq tables have lengths {} and {}, expected {row_count}",
            eq_product.len(),
            eq_spartan.len()
        )));
    }
    Ok(timed_stage3_value(
        "stage3.instruction_register_materialize_columns",
        || {
            let mut instruction_eq_product = unsafe_allocate_zero_vec(row_count);
            let mut right_operand_is_rs2 = unsafe_allocate_zero_vec(row_count);
            let mut rs2_value = unsafe_allocate_zero_vec(row_count);
            let mut right_operand_is_imm = unsafe_allocate_zero_vec(row_count);
            let mut imm = unsafe_allocate_zero_vec(row_count);
            let mut left_operand_is_rs1 = unsafe_allocate_zero_vec(row_count);
            let mut rs1_value = unsafe_allocate_zero_vec(row_count);
            let mut left_operand_is_pc = unsafe_allocate_zero_vec(row_count);
            let mut unexpanded_pc = unsafe_allocate_zero_vec(row_count);
            let mut registers_eq_spartan = unsafe_allocate_zero_vec(row_count);
            let mut rd_write_value = unsafe_allocate_zero_vec(row_count);
            let mut registers_rs1_value = unsafe_allocate_zero_vec(row_count);
            let mut registers_rs2_value = unsafe_allocate_zero_vec(row_count);

            instruction_eq_product
                .par_chunks_mut(STAGE3_INSTRUCTION_REGISTER_MATERIALIZE_CHUNK_ROWS)
                .zip(
                    right_operand_is_rs2
                        .par_chunks_mut(STAGE3_INSTRUCTION_REGISTER_MATERIALIZE_CHUNK_ROWS),
                )
                .zip(rs2_value.par_chunks_mut(STAGE3_INSTRUCTION_REGISTER_MATERIALIZE_CHUNK_ROWS))
                .zip(
                    right_operand_is_imm
                        .par_chunks_mut(STAGE3_INSTRUCTION_REGISTER_MATERIALIZE_CHUNK_ROWS),
                )
                .zip(imm.par_chunks_mut(STAGE3_INSTRUCTION_REGISTER_MATERIALIZE_CHUNK_ROWS))
                .zip(
                    left_operand_is_rs1
                        .par_chunks_mut(STAGE3_INSTRUCTION_REGISTER_MATERIALIZE_CHUNK_ROWS),
                )
                .zip(rs1_value.par_chunks_mut(STAGE3_INSTRUCTION_REGISTER_MATERIALIZE_CHUNK_ROWS))
                .zip(
                    left_operand_is_pc
                        .par_chunks_mut(STAGE3_INSTRUCTION_REGISTER_MATERIALIZE_CHUNK_ROWS),
                )
                .zip(
                    unexpanded_pc
                        .par_chunks_mut(STAGE3_INSTRUCTION_REGISTER_MATERIALIZE_CHUNK_ROWS),
                )
                .zip(
                    registers_eq_spartan
                        .par_chunks_mut(STAGE3_INSTRUCTION_REGISTER_MATERIALIZE_CHUNK_ROWS),
                )
                .zip(
                    rd_write_value
                        .par_chunks_mut(STAGE3_INSTRUCTION_REGISTER_MATERIALIZE_CHUNK_ROWS),
                )
                .zip(
                    registers_rs1_value
                        .par_chunks_mut(STAGE3_INSTRUCTION_REGISTER_MATERIALIZE_CHUNK_ROWS),
                )
                .zip(
                    registers_rs2_value
                        .par_chunks_mut(STAGE3_INSTRUCTION_REGISTER_MATERIALIZE_CHUNK_ROWS),
                )
                .enumerate()
                .for_each(|(chunk_index, chunks)| {
                    let (
                        (
                            (
                                (
                                    (
                                        (
                                            (
                                                (
                                                    (
                                                        (
                                                            (
                                                                (
                                                                    instruction_eq_product,
                                                                    right_operand_is_rs2,
                                                                ),
                                                                rs2_value,
                                                            ),
                                                            right_operand_is_imm,
                                                        ),
                                                        imm,
                                                    ),
                                                    left_operand_is_rs1,
                                                ),
                                                rs1_value,
                                            ),
                                            left_operand_is_pc,
                                        ),
                                        unexpanded_pc,
                                    ),
                                    registers_eq_spartan,
                                ),
                                rd_write_value,
                            ),
                            registers_rs1_value,
                        ),
                        registers_rs2_value,
                    ) = chunks;
                    let start = chunk_index * STAGE3_INSTRUCTION_REGISTER_MATERIALIZE_CHUNK_ROWS;
                    for offset in 0..instruction_eq_product.len() {
                        let cycle = bit_reversed_indices[start + offset];
                        let row = &rows[cycle];
                        let rs2 = F::from_u64(row.rs2_value);
                        let rs1 = F::from_u64(row.rs1_value);

                        instruction_eq_product[offset] = eq_product[cycle];
                        right_operand_is_rs2[offset] = F::from_bool(row.right_operand_is_rs2);
                        rs2_value[offset] = rs2;
                        right_operand_is_imm[offset] = F::from_bool(row.right_operand_is_imm);
                        imm[offset] = F::from_i128(row.imm);
                        left_operand_is_rs1[offset] = F::from_bool(row.left_operand_is_rs1);
                        rs1_value[offset] = rs1;
                        left_operand_is_pc[offset] = F::from_bool(row.left_operand_is_pc);
                        unexpanded_pc[offset] = F::from_u64(row.unexpanded_pc);
                        registers_eq_spartan[offset] = eq_spartan[cycle];
                        rd_write_value[offset] = F::from_u64(row.rd_write_value);
                        registers_rs1_value[offset] = rs1;
                        registers_rs2_value[offset] = rs2;
                    }
                });

            vec![
                instruction_eq_product,
                right_operand_is_rs2,
                rs2_value,
                right_operand_is_imm,
                imm,
                left_operand_is_rs1,
                rs1_value,
                left_operand_is_pc,
                unexpanded_pc,
                registers_eq_spartan,
                rd_write_value,
                registers_rs1_value,
                registers_rs2_value,
            ]
        },
    ))
}

fn build_stage3_instruction_registers_regular_batch_instances<F, W, B>(
    config: Stage3ProverConfig,
    stage2: &Stage2ClearOutput<F>,
    prefix: &Stage3RegularBatchPrefixOutput<F>,
    witness: &W,
    _backend: &mut B,
) -> Result<Vec<SumcheckRegularBatchInstance<F>>, ProverError>
where
    F: Field,
    W: JoltVmStage3InstructionRegisterRows,
    B: SumcheckBackend<F, JoltVmNamespace>,
{
    let bit_reversed_indices = stage3_bit_reversed_indices(config)?;

    let (eq_product, eq_spartan) = timed_stage3_value("stage3.eq_tables", || {
        let eq_product =
            EqPolynomial::new(stage2.batch.product_remainder.opening_point.clone()).evaluations();
        let eq_spartan = EqPolynomial::new(stage2.product_uniskip.tau_low.clone()).evaluations();
        (eq_product, eq_spartan)
    });
    let mut columns = stage3_instruction_register_reversed_columns_from_rows(
        config,
        witness,
        &bit_reversed_indices,
        &eq_product,
        &eq_spartan,
    )?
    .into_iter();
    let mut next_column = |label: &'static str| {
        columns.next().ok_or_else(|| {
            invalid_sumcheck_output(format!("Stage 3 missing materialized column {label}"))
        })
    };
    let instruction_gamma = prefix.instruction_gamma;
    let registers_gamma = prefix.registers_gamma;
    let registers_gamma2 = registers_gamma * registers_gamma;

    let instruction_input = timed_stage3("stage3.build_instruction_instance", || {
        Ok::<_, ProverError>(SumcheckRegularBatchInstance::new_products(
            "stage3.instruction_input",
            prefix.input_claims.instruction_input,
            vec![
                Polynomial::new(next_column("instruction product eq")?),
                Polynomial::new(next_column("instruction right operand is rs2")?),
                Polynomial::new(next_column("instruction rs2 value")?),
                Polynomial::new(next_column("instruction right operand is imm")?),
                Polynomial::new(next_column("instruction immediate")?),
                Polynomial::new(next_column("instruction left operand is rs1")?),
                Polynomial::new(next_column("instruction rs1 value")?),
                Polynomial::new(next_column("instruction left operand is PC")?),
                Polynomial::new(next_column("instruction unexpanded PC")?),
            ],
            vec![
                SumcheckRegularBatchProduct::new(
                    F::one(),
                    vec![
                        batch_factor(vec![batch_term(0, F::one())]),
                        batch_factor(vec![batch_term(1, F::one())]),
                        batch_factor(vec![batch_term(2, F::one())]),
                    ],
                ),
                SumcheckRegularBatchProduct::new(
                    F::one(),
                    vec![
                        batch_factor(vec![batch_term(0, F::one())]),
                        batch_factor(vec![batch_term(3, F::one())]),
                        batch_factor(vec![batch_term(4, F::one())]),
                    ],
                ),
                SumcheckRegularBatchProduct::new(
                    instruction_gamma,
                    vec![
                        batch_factor(vec![batch_term(0, F::one())]),
                        batch_factor(vec![batch_term(5, F::one())]),
                        batch_factor(vec![batch_term(6, F::one())]),
                    ],
                ),
                SumcheckRegularBatchProduct::new(
                    instruction_gamma,
                    vec![
                        batch_factor(vec![batch_term(0, F::one())]),
                        batch_factor(vec![batch_term(7, F::one())]),
                        batch_factor(vec![batch_term(8, F::one())]),
                    ],
                ),
            ],
        ))
    })?;

    let registers_claim_reduction = timed_stage3("stage3.build_registers_instance", || {
        Ok::<_, ProverError>(SumcheckRegularBatchInstance::new_products(
            "stage3.registers_claim_reduction",
            prefix.input_claims.registers_claim_reduction,
            vec![
                Polynomial::new(next_column("registers Spartan eq")?),
                Polynomial::new(next_column("registers rd write value")?),
                Polynomial::new(next_column("registers rs1 value")?),
                Polynomial::new(next_column("registers rs2 value")?),
            ],
            vec![SumcheckRegularBatchProduct::new(
                F::one(),
                vec![
                    batch_factor(vec![batch_term(0, F::one())]),
                    batch_factor(vec![
                        batch_term(1, F::one()),
                        batch_term(2, registers_gamma),
                        batch_term(3, registers_gamma2),
                    ]),
                ],
            )],
        ))
    })?;

    Ok(vec![instruction_input, registers_claim_reduction])
}

/// Drives the Stage 3 regular batched sumcheck through backend kernels.
///
/// The shift statement uses the core two-phase eq+1 prefix/suffix algorithm;
/// instruction-input and register claim-reduction use the shared regular-batch
/// kernel. Batching and transcript order stay verifier-canonical: shift,
/// instruction-input, registers claim-reduction.
fn prove_stage3_regular_batch_sumcheck<F, W, B, T, C>(
    input: &Stage3ProverInput<'_, F, W>,
    prefix: &Stage3RegularBatchPrefixOutput<F>,
    transcript: &mut T,
    backend: &mut B,
) -> Result<Stage3RegularBatch<F, C>, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>
        + JoltVmStage3ShiftRows
        + JoltVmStage3InstructionRegisterRows,
    T: Transcript<Challenge = F>,
    B: SumcheckBackend<F, JoltVmNamespace> + Stage3SpartanSumcheckBackend<F>,
{
    let shift_request = timed_stage3("stage3.shift_request", || {
        build_stage3_shift_state_request(input.config, input.stage2, prefix, input.witness)
    })?;
    let mut shift_state = timed_stage3("stage3.shift_materialize", || {
        backend.materialize_sumcheck_stage3_shift_state(&shift_request)
    })?;
    let instances = timed_stage3("stage3.instruction_register_instances", || {
        build_stage3_instruction_registers_regular_batch_instances(
            input.config,
            input.stage2,
            prefix,
            input.witness,
            backend,
        )
    })?;
    let mut regular_state =
        SumcheckRegularBatchState::new("stage3.instruction_registers_regular_batch", instances);
    let max_num_rounds = input.config.log_t;
    for instance in &regular_state.instances {
        if instance.num_rounds() != max_num_rounds {
            return Err(invalid_sumcheck_output(format!(
                "Stage 3 instance {} has {} rounds, expected {max_num_rounds}",
                instance.label,
                instance.num_rounds()
            )));
        }
    }

    append_sumcheck_claim(transcript, &prefix.input_claims.shift);
    append_sumcheck_claim(transcript, &prefix.input_claims.instruction_input);
    append_sumcheck_claim(transcript, &prefix.input_claims.registers_claim_reduction);
    let batching_coefficients = (0..3)
        .map(|_| transcript.challenge_scalar())
        .collect::<Vec<_>>();
    let [shift_coefficient, instruction_coefficient, registers_coefficient] =
        batching_coefficients.as_slice()
    else {
        return Err(invalid_sumcheck_output(
            "Stage 3 batch expected exactly three batching coefficients",
        ));
    };

    let mut shift_claim = prefix.input_claims.shift;
    let mut regular_claims = vec![
        prefix.input_claims.instruction_input,
        prefix.input_claims.registers_claim_reduction,
    ];
    let mut running_claim = *shift_coefficient * shift_claim
        + *instruction_coefficient * regular_claims[0]
        + *registers_coefficient * regular_claims[1];

    let mut challenges = Vec::with_capacity(max_num_rounds);
    let mut round_polynomials = Vec::with_capacity(max_num_rounds);
    let mut shift_round_ms = 0.0;
    let mut regular_round_ms = 0.0;
    let mut combine_ms = 0.0;
    let mut transcript_ms = 0.0;
    let mut bind_shift_ms = 0.0;
    let mut bind_regular_ms = 0.0;
    for round in 0..max_num_rounds {
        let shift_poly = timed_stage3_accumulate(&mut shift_round_ms, || {
            backend.evaluate_sumcheck_stage3_shift_round(&shift_state, shift_claim)
        })?;
        let messages = timed_stage3_accumulate(&mut regular_round_ms, || {
            backend.evaluate_sumcheck_regular_batch_round(
                &mut regular_state,
                round,
                max_num_rounds,
                &regular_claims,
            )
        })?;
        if messages.len() != regular_claims.len() {
            return Err(invalid_sumcheck_output(format!(
                "Stage 3 instruction/register batch round {round} returned {} messages, expected {}",
                messages.len(),
                regular_claims.len()
            )));
        }
        let mut regular_polys = Vec::with_capacity(regular_claims.len());
        for (expected_index, message) in messages.into_iter().enumerate() {
            if message.instance_index != expected_index {
                return Err(invalid_sumcheck_output(format!(
                    "Stage 3 instruction/register batch round {round} returned instance {}, expected {expected_index}",
                    message.instance_index
                )));
            }
            regular_polys.push(message.polynomial);
        }

        let batched_poly = timed_stage3_accumulate(&mut combine_ms, || {
            let mut batched_poly = &shift_poly * *shift_coefficient;
            batched_poly += &(&regular_polys[0] * *instruction_coefficient);
            batched_poly += &(&regular_polys[1] * *registers_coefficient);
            batched_poly
        });
        let round_sum = batched_poly.evaluate(F::zero()) + batched_poly.evaluate(F::one());
        if round_sum != running_claim {
            return Err(invalid_sumcheck_output(format!(
                "Stage 3 regular batch round {round} sumcheck invariant failed"
            )));
        }

        let batched_poly =
            timed_stage3_accumulate(&mut combine_ms, || trim_round_polynomial(batched_poly));
        let challenge = timed_stage3_accumulate(&mut transcript_ms, || {
            CompressedLabeledRoundPoly::sumcheck(&batched_poly).append_to_transcript(transcript);
            transcript.challenge()
        });
        running_claim = batched_poly.evaluate(challenge);
        challenges.push(challenge);
        round_polynomials.push(batched_poly.compress());

        shift_claim = shift_poly.evaluate(challenge);
        for (claim, poly) in regular_claims.iter_mut().zip(regular_polys) {
            *claim = poly.evaluate(challenge);
        }
        timed_stage3_accumulate(&mut bind_shift_ms, || {
            backend.bind_sumcheck_stage3_shift_state(&mut shift_state, challenge)
        })?;
        timed_stage3_accumulate(&mut bind_regular_ms, || {
            backend.bind_sumcheck_regular_batch_state(
                &mut regular_state,
                round,
                max_num_rounds,
                challenge,
            )
        })?;
    }
    record_stage3_accumulated("stage3.rounds.shift", shift_round_ms);
    record_stage3_accumulated("stage3.rounds.regular", regular_round_ms);
    record_stage3_accumulated("stage3.rounds.combine", combine_ms);
    record_stage3_accumulated("stage3.rounds.transcript", transcript_ms);
    record_stage3_accumulated("stage3.bind.shift", bind_shift_ms);
    record_stage3_accumulated("stage3.bind.regular", bind_regular_ms);
    let output_openings = timed_stage3("stage3.bound_output_openings", || {
        stage3_output_openings_from_bound_states(backend, &shift_state, &regular_state)
    })?;

    Ok(Stage3RegularBatch {
        proof: SumcheckProof::Clear(ClearProof::Compressed(CompressedSumcheckProof {
            round_polynomials,
        })),
        challenges,
        batching_coefficients,
        output_claim: running_claim,
        output_openings: Some(output_openings),
    })
}

fn stage3_output_openings_from_bound_states<F, B>(
    backend: &mut B,
    shift_state: &B::Stage3ShiftState,
    regular_state: &SumcheckRegularBatchState<F>,
) -> Result<Stage3RegularBatchOutputOpeningClaims<F>, ProverError>
where
    F: Field,
    B: Stage3SpartanSumcheckBackend<F>,
{
    let [unexpanded_pc, pc, is_virtual, is_first_in_sequence, is_noop] =
        backend.stage3_shift_output_openings(shift_state)?;
    let instruction_instance = regular_state.instances.first().ok_or_else(|| {
        invalid_sumcheck_output("Stage 3 regular state is missing instruction input instance")
    })?;
    let registers_instance = regular_state.instances.get(1).ok_or_else(|| {
        invalid_sumcheck_output("Stage 3 regular state is missing registers instance")
    })?;

    Ok(Stage3RegularBatchOutputOpeningClaims {
        shift: SpartanShiftOutputOpeningClaims {
            unexpanded_pc,
            pc,
            is_virtual,
            is_first_in_sequence,
            is_noop,
        },
        instruction_input: InstructionInputOutputOpeningClaims {
            right_operand_is_rs2: final_regular_polynomial_value(
                instruction_instance,
                1,
                "instruction right operand is rs2",
            )?,
            rs2_value: final_regular_polynomial_value(
                instruction_instance,
                2,
                "instruction rs2 value",
            )?,
            right_operand_is_imm: final_regular_polynomial_value(
                instruction_instance,
                3,
                "instruction right operand is imm",
            )?,
            imm: final_regular_polynomial_value(instruction_instance, 4, "instruction immediate")?,
            left_operand_is_rs1: final_regular_polynomial_value(
                instruction_instance,
                5,
                "instruction left operand is rs1",
            )?,
            rs1_value: final_regular_polynomial_value(
                instruction_instance,
                6,
                "instruction rs1 value",
            )?,
            left_operand_is_pc: final_regular_polynomial_value(
                instruction_instance,
                7,
                "instruction left operand is PC",
            )?,
            unexpanded_pc: final_regular_polynomial_value(
                instruction_instance,
                8,
                "instruction unexpanded PC",
            )?,
        },
        registers_claim_reduction: RegistersClaimReductionOutputOpeningClaims {
            rd_write_value: final_regular_polynomial_value(
                registers_instance,
                1,
                "registers rd write value",
            )?,
            rs1_value: final_regular_polynomial_value(
                registers_instance,
                2,
                "registers rs1 value",
            )?,
            rs2_value: final_regular_polynomial_value(
                registers_instance,
                3,
                "registers rs2 value",
            )?,
        },
    })
}

fn final_regular_polynomial_value<F: Field>(
    instance: &SumcheckRegularBatchInstance<F>,
    polynomial_index: usize,
    label: &'static str,
) -> Result<F, ProverError> {
    let polynomial = instance.polynomials.get(polynomial_index).ok_or_else(|| {
        invalid_sumcheck_output(format!(
            "Stage 3 regular instance {} is missing polynomial {polynomial_index} ({label})",
            instance.label
        ))
    })?;
    let [value] = polynomial.evaluations() else {
        return Err(invalid_sumcheck_output(format!(
            "Stage 3 regular instance {} polynomial {polynomial_index} ({label}) has {} evaluations, expected 1",
            instance.label,
            polynomial.len()
        )));
    };
    Ok(*value)
}

/// Dense/materialized Stage 3 batch driver retained as a reference path for
/// correctness tests. Production proving should use the prefix/suffix shift
/// path above.
#[expect(
    dead_code,
    reason = "dense Stage 3 batch prover is retained as a parity-audit reference"
)]
fn prove_stage3_regular_batch_sumcheck_dense_reference<F, T, C, B>(
    config: Stage3ProverConfig,
    instances: Vec<SumcheckRegularBatchInstance<F>>,
    transcript: &mut T,
    backend: &mut B,
) -> Result<Stage3RegularBatch<F, C>, ProverError>
where
    F: Field,
    T: Transcript<Challenge = F>,
    B: SumcheckBackend<F, JoltVmNamespace>,
{
    let mut state = SumcheckRegularBatchState::new("stage3.regular_batch", instances);
    let max_num_rounds = config.log_t;
    for instance in &state.instances {
        if instance.num_rounds() != max_num_rounds {
            return Err(invalid_sumcheck_output(format!(
                "Stage 3 instance {} has {} rounds, expected {max_num_rounds}",
                instance.label,
                instance.num_rounds()
            )));
        }
        append_sumcheck_claim(transcript, &instance.input_claim);
    }
    let instance_count = state.instances.len();
    let batching_coefficients = (0..instance_count)
        .map(|_| transcript.challenge_scalar())
        .collect::<Vec<_>>();
    let mut individual_claims = state
        .instances
        .iter()
        .map(|instance| instance.input_claim)
        .collect::<Vec<_>>();
    let mut running_claim = individual_claims
        .iter()
        .zip(&batching_coefficients)
        .map(|(claim, coefficient)| *claim * *coefficient)
        .sum::<F>();

    let mut challenges = Vec::with_capacity(max_num_rounds);
    let mut round_polynomials = Vec::with_capacity(max_num_rounds);
    for round in 0..max_num_rounds {
        let messages = backend.evaluate_sumcheck_regular_batch_round(
            &mut state,
            round,
            max_num_rounds,
            &individual_claims,
        )?;
        if messages.len() != instance_count {
            return Err(invalid_sumcheck_output(format!(
                "Stage 3 regular batch round {round} returned {} messages, expected {instance_count}",
                messages.len()
            )));
        }
        let mut univariate_polys = Vec::with_capacity(instance_count);
        for (expected_index, message) in messages.into_iter().enumerate() {
            if message.instance_index != expected_index {
                return Err(invalid_sumcheck_output(format!(
                    "Stage 3 regular batch round {round} returned instance {}, expected {expected_index}",
                    message.instance_index
                )));
            }
            univariate_polys.push(message.polynomial);
        }
        let mut batched_poly = UnivariatePoly::zero();
        for (poly, coefficient) in univariate_polys.iter().zip(&batching_coefficients) {
            batched_poly += &(poly * *coefficient);
        }
        let round_sum = batched_poly.evaluate(F::zero()) + batched_poly.evaluate(F::one());
        if round_sum != running_claim {
            return Err(invalid_sumcheck_output(format!(
                "Stage 3 regular batch round {round} sumcheck invariant failed"
            )));
        }
        let batched_poly = trim_round_polynomial(batched_poly);
        CompressedLabeledRoundPoly::sumcheck(&batched_poly).append_to_transcript(transcript);
        let challenge = transcript.challenge();
        running_claim = batched_poly.evaluate(challenge);
        challenges.push(challenge);
        round_polynomials.push(batched_poly.compress());
        for (claim, poly) in individual_claims.iter_mut().zip(univariate_polys) {
            *claim = poly.evaluate(challenge);
        }
        backend.bind_sumcheck_regular_batch_state(&mut state, round, max_num_rounds, challenge)?;
    }

    Ok(Stage3RegularBatch {
        proof: SumcheckProof::Clear(ClearProof::Compressed(CompressedSumcheckProof {
            round_polynomials,
        })),
        challenges,
        batching_coefficients,
        output_claim: running_claim,
        output_openings: None,
    })
}

/// Isolated Stage 3 regular-batch sumcheck proof, used by the prover harness to
/// benchmark the backend kernel against the `jolt-core` reference path.
#[cfg(feature = "frontier-harness")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage3RegularBatchFrontierProof<F: Field, C> {
    pub proof: SumcheckProof<F, C>,
    pub challenges: Vec<F>,
    pub batching_coefficients: Vec<F>,
    pub output_claim: F,
}

/// Builds the Stage 3 regular-batch instances and drives the backend sumcheck
/// kernel in isolation, without the surrounding gamma derivation or output
/// opening evaluation. Mirrors `prove_stage2_regular_batch_sumcheck_for_frontier`.
#[cfg(feature = "frontier-harness")]
pub fn prove_stage3_regular_batch_sumcheck_for_frontier<F, W, B, T, C>(
    input: &Stage3ProverInput<'_, F, W>,
    backend: &mut B,
    prefix: &Stage3RegularBatchPrefixOutput<F>,
    transcript: &mut T,
) -> Result<Stage3RegularBatchFrontierProof<F, C>, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>
        + JoltVmStage3ShiftRows
        + JoltVmStage3InstructionRegisterRows,
    B: SumcheckBackend<F, JoltVmNamespace> + Stage3SpartanSumcheckBackend<F>,
    T: Transcript<Challenge = F>,
{
    let batch =
        prove_stage3_regular_batch_sumcheck::<F, W, B, T, C>(input, prefix, transcript, backend)?;
    Ok(Stage3RegularBatchFrontierProof {
        proof: batch.proof,
        challenges: batch.challenges,
        batching_coefficients: batch.batching_coefficients,
        output_claim: batch.output_claim,
    })
}

fn stage3_expected_outputs<F: Field>(
    stage2: &Stage2ClearOutput<F>,
    prefix: &Stage3RegularBatchPrefixOutput<F>,
    opening_point: &[F],
    openings: &Stage3RegularBatchOutputOpeningClaims<F>,
) -> Result<Stage3RegularBatchExpectedOutputs<F>, ProverError> {
    let eq_plus_one_outer =
        EqPlusOnePolynomial::new(stage2.product_uniskip.tau_low.clone()).evaluate(opening_point);
    let eq_plus_one_product =
        EqPlusOnePolynomial::new(stage2.batch.product_remainder.opening_point.clone())
            .evaluate(opening_point);
    let eq_product = try_eq_mle(opening_point, &stage2.batch.product_remainder.opening_point)
        .map_err(invalid_sumcheck_output)?;
    let eq_spartan = try_eq_mle(opening_point, &stage2.product_uniskip.tau_low)
        .map_err(invalid_sumcheck_output)?;

    let shift_gamma = prefix.shift_gamma;
    let shift_gamma2 = shift_gamma * shift_gamma;
    let shift_gamma3 = shift_gamma2 * shift_gamma;
    let shift_gamma4 = shift_gamma3 * shift_gamma;
    let shift = eq_plus_one_outer
        * (openings.shift.unexpanded_pc
            + shift_gamma * openings.shift.pc
            + shift_gamma2 * openings.shift.is_virtual
            + shift_gamma3 * openings.shift.is_first_in_sequence)
        + eq_plus_one_product * shift_gamma4 * (F::one() - openings.shift.is_noop);

    let instruction_gamma = prefix.instruction_gamma;
    let instruction_right = openings.instruction_input.right_operand_is_rs2
        * openings.instruction_input.rs2_value
        + openings.instruction_input.right_operand_is_imm * openings.instruction_input.imm;
    let instruction_left = openings.instruction_input.left_operand_is_rs1
        * openings.instruction_input.rs1_value
        + openings.instruction_input.left_operand_is_pc * openings.instruction_input.unexpanded_pc;
    let instruction_input = eq_product * (instruction_right + instruction_gamma * instruction_left);

    let registers_gamma = prefix.registers_gamma;
    let registers_gamma2 = registers_gamma * registers_gamma;
    let registers_claim_reduction = eq_spartan
        * (openings.registers_claim_reduction.rd_write_value
            + registers_gamma * openings.registers_claim_reduction.rs1_value
            + registers_gamma2 * openings.registers_claim_reduction.rs2_value);

    Ok(Stage3RegularBatchExpectedOutputs {
        shift,
        instruction_input,
        registers_claim_reduction,
    })
}

fn batch_term<F: Field>(polynomial: usize, coefficient: F) -> SumcheckRegularBatchLinearTerm<F> {
    SumcheckRegularBatchLinearTerm::new(polynomial, coefficient)
}

fn batch_factor<F: Field>(
    terms: Vec<SumcheckRegularBatchLinearTerm<F>>,
) -> SumcheckRegularBatchLinearFactor<F> {
    SumcheckRegularBatchLinearFactor::from_terms(terms)
}

fn neg_one<F: Field>() -> F {
    F::zero() - F::one()
}

fn stage3_reversed_polynomial<F: Field>(
    config: Stage3ProverConfig,
    label: &'static str,
    values: Vec<F>,
) -> Result<Polynomial<F>, ProverError> {
    let bit_reversed_indices = stage3_bit_reversed_indices(config)?;
    stage3_reversed_polynomial_with_indices(config, label, values, &bit_reversed_indices)
}

fn stage3_reversed_polynomial_with_indices<F: Field>(
    config: Stage3ProverConfig,
    label: &'static str,
    values: Vec<F>,
    bit_reversed_indices: &[usize],
) -> Result<Polynomial<F>, ProverError> {
    let expected_len = 1usize.checked_shl(config.log_t as u32).ok_or_else(|| {
        invalid_sumcheck_output(format!(
            "Stage 3 {label} trace length overflows for log_t={}",
            config.log_t
        ))
    })?;
    if values.len() != expected_len {
        return Err(invalid_sumcheck_output(format!(
            "Stage 3 {label} materialized {} rows, expected {expected_len}",
            values.len()
        )));
    }
    if bit_reversed_indices.len() != expected_len {
        return Err(invalid_sumcheck_output(format!(
            "Stage 3 {label} bit-reversal table has {} entries, expected {expected_len}",
            bit_reversed_indices.len()
        )));
    }
    let reversed = bit_reversed_indices
        .iter()
        .map(|&index| values[index])
        .collect::<Vec<_>>();
    Ok(Polynomial::from(reversed))
}

fn stage3_bit_reversed_indices(config: Stage3ProverConfig) -> Result<Vec<usize>, ProverError> {
    let expected_len = 1usize.checked_shl(config.log_t as u32).ok_or_else(|| {
        invalid_sumcheck_output(format!(
            "Stage 3 trace length overflows for log_t={}",
            config.log_t
        ))
    })?;
    Ok((0..expected_len)
        .map(|index| bit_reverse(index, config.log_t))
        .collect())
}

fn bit_reverse(index: usize, bits: usize) -> usize {
    let mut reversed = 0usize;
    for position in 0..bits {
        reversed <<= 1;
        reversed |= (index >> position) & 1;
    }
    reversed
}

fn trim_round_polynomial<F: Field>(poly: UnivariatePoly<F>) -> UnivariatePoly<F> {
    let mut coefficients = poly.into_coefficients();
    while coefficients.len() > 2 && coefficients.last().is_some_and(|value| *value == F::zero()) {
        let _ = coefficients.pop();
    }
    UnivariatePoly::new(coefficients)
}

pub fn derive_stage3_regular_batch_prefix<F, T>(
    config: Stage3ProverConfig,
    stage1: &Stage1ClearOutput<F>,
    stage2: &Stage2ClearOutput<F>,
    transcript: &mut T,
) -> Result<Stage3RegularBatchPrefixOutput<F>, ProverError>
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    let expected_rounds = config.log_t;
    if stage2.batch.product_remainder.opening_point.len() != expected_rounds {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 3 product-remainder dependency has {} variables, expected {expected_rounds}",
                stage2.batch.product_remainder.opening_point.len()
            ),
        });
    }
    if stage2.batch.instruction_claim_reduction.opening_point
        != stage2.batch.product_remainder.opening_point
    {
        return Err(ProverError::InvalidStageRequest {
            reason: "Stage 3 instruction input dependencies use different opening points"
                .to_owned(),
        });
    }
    if stage2.product_uniskip.tau_low.len() != expected_rounds {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 3 product uni-skip dependency has {} variables, expected {expected_rounds}",
                stage2.product_uniskip.tau_low.len()
            ),
        });
    }

    let shift_gamma = transcript.challenge_scalar();
    let instruction_gamma = transcript.challenge_scalar();
    let registers_gamma = transcript.challenge_scalar();

    let shift_gamma2 = shift_gamma * shift_gamma;
    let shift_gamma3 = shift_gamma2 * shift_gamma;
    let shift_gamma4 = shift_gamma3 * shift_gamma;
    let registers_gamma2 = registers_gamma * registers_gamma;

    let product_left = stage2
        .output_claims
        .product_remainder
        .left_instruction_input;
    let product_right = stage2
        .output_claims
        .product_remainder
        .right_instruction_input;
    let reduced_left = stage2
        .output_claims
        .instruction_claim_reduction
        .left_instruction_input
        .unwrap_or(product_left);
    let reduced_right = stage2
        .output_claims
        .instruction_claim_reduction
        .right_instruction_input
        .unwrap_or(product_right);
    if reduced_left != product_left || reduced_right != product_right {
        return Err(ProverError::InvalidStageRequest {
            reason:
                "Stage 3 instruction input dependencies disagree on left/right instruction inputs"
                    .to_owned(),
        });
    }

    let input_claims = Stage3RegularBatchInputClaims {
        shift: stage1.outer.next_unexpanded_pc
            + shift_gamma * stage1.outer.next_pc
            + shift_gamma2 * stage1.outer.next_is_virtual
            + shift_gamma3 * stage1.outer.next_is_first_in_sequence
            + shift_gamma4 * (F::one() - stage2.output_claims.product_remainder.next_is_noop),
        instruction_input: product_right + instruction_gamma * product_left,
        registers_claim_reduction: stage1.outer.rd_write_value
            + registers_gamma * stage1.outer.rs1_value
            + registers_gamma2 * stage1.outer.rs2_value,
    };

    Ok(Stage3RegularBatchPrefixOutput {
        input_claims,
        shift_gamma,
        instruction_gamma,
        registers_gamma,
    })
}

fn validate_stage3_row_count(
    config: Stage3ProverConfig,
    actual: usize,
    label: &'static str,
) -> Result<(), ProverError> {
    let expected = checked_stage3_len(config.log_t, "Stage 3 trace length")?;
    if actual == expected {
        return Ok(());
    }
    Err(ProverError::InvalidStageRequest {
        reason: format!("Stage 3 {label} has {actual} rows, expected {expected}"),
    })
}

fn checked_stage3_len(log_len: usize, label: &str) -> Result<usize, ProverError> {
    1usize
        .checked_shl(log_len as u32)
        .ok_or_else(|| ProverError::InvalidStageRequest {
            reason: format!("{label} overflows for log length {log_len}"),
        })
}

pub fn evaluate_stage3_output_openings<F, W, B>(
    config: Stage3ProverConfig,
    witness: &W,
    backend: &mut B,
    shift_opening_point: Vec<F>,
    instruction_input_opening_point: Vec<F>,
    registers_claim_reduction_opening_point: Vec<F>,
) -> Result<Stage3RegularBatchOutputOpeningClaims<F>, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>,
    B: SumcheckBackend<F, JoltVmNamespace>,
{
    let request = build_stage3_output_opening_evaluation_request(
        config,
        witness,
        shift_opening_point,
        instruction_input_opening_point,
        registers_claim_reduction_opening_point,
    )?;
    let evaluations = backend.evaluate_sumcheck_views(&request.evaluations, witness)?;
    stage3_output_openings_from_evaluations(&request, evaluations)
}

pub fn prove_stage3_transparent_sumchecks<F, W, B, T, C>(
    config: Stage3ProverConfig,
    witness: &W,
    backend: &mut B,
    stage2: &Stage2ClearOutput<F>,
    prefix: &Stage3RegularBatchPrefixOutput<F>,
    transcript: &mut T,
) -> Result<Stage3RegularBatchProofOutput<F, C>, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>,
    B: SumcheckBackend<F, JoltVmNamespace>,
    T: Transcript<Challenge = F>,
{
    let request = build_stage3_output_opening_materialization_request::<F, W>(witness)?;
    let materializations =
        backend.materialize_sumcheck_views(&request.materializations, witness)?;
    let materialized = stage3_materialized_openings_from_outputs(&request, materializations)?;
    let mut context = Stage3BatchContext::new(config, stage2, prefix, materialized)?;

    append_sumcheck_claim(transcript, &prefix.input_claims.shift);
    append_sumcheck_claim(transcript, &prefix.input_claims.instruction_input);
    append_sumcheck_claim(transcript, &prefix.input_claims.registers_claim_reduction);
    let batching_coefficients = (0..3)
        .map(|_| transcript.challenge_scalar())
        .collect::<Vec<_>>();
    let [shift_coefficient, instruction_coefficient, registers_coefficient] =
        batching_coefficients.as_slice()
    else {
        return Err(invalid_sumcheck_output(
            "Stage 3 batch expected exactly three batching coefficients",
        ));
    };
    let coefficients = Stage3BatchCoefficients {
        shift: *shift_coefficient,
        instruction_input: *instruction_coefficient,
        registers_claim_reduction: *registers_coefficient,
    };

    let mut running_claim = coefficients.shift * prefix.input_claims.shift
        + coefficients.instruction_input * prefix.input_claims.instruction_input
        + coefficients.registers_claim_reduction * prefix.input_claims.registers_claim_reduction;
    let mut round_polynomials = Vec::with_capacity(config.log_t);
    let mut sumcheck_point = Vec::with_capacity(config.log_t);

    for round in 0..config.log_t {
        let evaluations = (0..=STAGE3_BATCH_DEGREE)
            .map(|point| context.round_sum(F::from_u64(point as u64), coefficients))
            .collect::<Vec<_>>();
        let round_poly = UnivariatePoly::interpolate_over_integers(&evaluations);
        let round_sum = round_poly.evaluate(F::zero()) + round_poly.evaluate(F::one());
        if round_sum != running_claim {
            return Err(invalid_sumcheck_output(format!(
                "Stage 3 batch round {round} sumcheck invariant failed"
            )));
        }

        CompressedLabeledRoundPoly::sumcheck(&round_poly).append_to_transcript(transcript);
        let challenge = transcript.challenge();
        running_claim = round_poly.evaluate(challenge);
        sumcheck_point.push(challenge);
        context.bind(challenge);
        round_polynomials.push(round_poly.compress());
    }

    let output_openings = context.output_openings()?;
    let expected_outputs = context.expected_outputs(&output_openings)?;
    let expected_final_claim = coefficients.shift * expected_outputs.shift
        + coefficients.instruction_input * expected_outputs.instruction_input
        + coefficients.registers_claim_reduction * expected_outputs.registers_claim_reduction;
    if running_claim != expected_final_claim {
        return Err(invalid_sumcheck_output(
            "Stage 3 batch final claim did not match output openings",
        ));
    }

    timed_stage3_value("stage3.opening_claim_transcript", || {
        append_stage3_opening_claims(transcript, &output_openings);
    });
    let opening_point = reversed_point(&sumcheck_point);

    Ok(Stage3RegularBatchProofOutput {
        prefix: prefix.clone(),
        proof: SumcheckProof::Clear(ClearProof::Compressed(CompressedSumcheckProof {
            round_polynomials,
        })),
        output_openings,
        expected_outputs: Stage3RegularBatchExpectedOutputs {
            shift: expected_outputs.shift,
            instruction_input: expected_outputs.instruction_input,
            registers_claim_reduction: expected_outputs.registers_claim_reduction,
        },
        batching_coefficients,
        sumcheck_point: sumcheck_point.clone(),
        sumcheck_final_claim: running_claim,
        expected_final_claim,
        shift_opening_point: opening_point.clone(),
        instruction_input_opening_point: opening_point.clone(),
        registers_claim_reduction_opening_point: opening_point,
    })
}

pub fn append_stage3_opening_claims<F, T>(
    transcript: &mut T,
    claims: &Stage3RegularBatchOutputOpeningClaims<F>,
) where
    F: Field,
    T: Transcript<Challenge = F>,
{
    transcript.append_labeled(b"opening_claim", &claims.shift.unexpanded_pc);
    transcript.append_labeled(b"opening_claim", &claims.shift.pc);
    transcript.append_labeled(b"opening_claim", &claims.shift.is_virtual);
    transcript.append_labeled(b"opening_claim", &claims.shift.is_first_in_sequence);
    transcript.append_labeled(b"opening_claim", &claims.shift.is_noop);
    transcript.append_labeled(
        b"opening_claim",
        &claims.instruction_input.left_operand_is_rs1,
    );
    transcript.append_labeled(b"opening_claim", &claims.instruction_input.rs1_value);
    transcript.append_labeled(
        b"opening_claim",
        &claims.instruction_input.left_operand_is_pc,
    );
    transcript.append_labeled(
        b"opening_claim",
        &claims.instruction_input.right_operand_is_rs2,
    );
    transcript.append_labeled(b"opening_claim", &claims.instruction_input.rs2_value);
    transcript.append_labeled(
        b"opening_claim",
        &claims.instruction_input.right_operand_is_imm,
    );
    transcript.append_labeled(b"opening_claim", &claims.instruction_input.imm);
    transcript.append_labeled(
        b"opening_claim",
        &claims.registers_claim_reduction.rd_write_value,
    );
}

#[cfg(feature = "zk")]
fn stage3_committed_output_claim_values<F: Field>(
    claims: &Stage3RegularBatchOutputOpeningClaims<F>,
) -> Vec<F> {
    vec![
        claims.shift.unexpanded_pc,
        claims.shift.pc,
        claims.shift.is_virtual,
        claims.shift.is_first_in_sequence,
        claims.shift.is_noop,
        claims.instruction_input.left_operand_is_rs1,
        claims.instruction_input.rs1_value,
        claims.instruction_input.left_operand_is_pc,
        claims.instruction_input.right_operand_is_rs2,
        claims.instruction_input.rs2_value,
        claims.instruction_input.right_operand_is_imm,
        claims.instruction_input.imm,
        claims.registers_claim_reduction.rd_write_value,
    ]
}

#[derive(Clone, Copy)]
struct Stage3BatchCoefficients<F: Field> {
    shift: F,
    instruction_input: F,
    registers_claim_reduction: F,
}

struct Stage3BatchContext<F: Field> {
    shift_gamma: F,
    shift_gamma2: F,
    shift_gamma3: F,
    shift_gamma4: F,
    instruction_gamma: F,
    registers_gamma: F,
    registers_gamma2: F,
    shift: Stage3ShiftPolys<F>,
    instruction_input: Stage3InstructionInputPolys<F>,
    registers_claim_reduction: Stage3RegistersClaimReductionPolys<F>,
    eq_plus_one_outer: Polynomial<F>,
    eq_plus_one_product: Polynomial<F>,
    eq_product: Polynomial<F>,
    eq_spartan: Polynomial<F>,
}

struct Stage3ShiftPolys<F: Field> {
    unexpanded_pc: Polynomial<F>,
    pc: Polynomial<F>,
    is_virtual: Polynomial<F>,
    is_first_in_sequence: Polynomial<F>,
    is_noop: Polynomial<F>,
}

struct Stage3InstructionInputPolys<F: Field> {
    right_operand_is_rs2: Polynomial<F>,
    rs2_value: Polynomial<F>,
    right_operand_is_imm: Polynomial<F>,
    imm: Polynomial<F>,
    left_operand_is_rs1: Polynomial<F>,
    rs1_value: Polynomial<F>,
    left_operand_is_pc: Polynomial<F>,
    unexpanded_pc: Polynomial<F>,
}

struct Stage3RegistersClaimReductionPolys<F: Field> {
    rd_write: Polynomial<F>,
    rs1: Polynomial<F>,
    rs2: Polynomial<F>,
}

struct Stage3ExpectedOutputs<F: Field> {
    shift: F,
    instruction_input: F,
    registers_claim_reduction: F,
}

impl<F: Field> Stage3BatchContext<F> {
    fn new(
        config: Stage3ProverConfig,
        stage2: &Stage2ClearOutput<F>,
        prefix: &Stage3RegularBatchPrefixOutput<F>,
        materialized: Stage3RegularBatchMaterializedOpenings<F>,
    ) -> Result<Self, ProverError> {
        let shift = stage3_shift_polys(config, materialized.shift)?;
        let instruction_input =
            stage3_instruction_input_polys(config, materialized.instruction_input)?;
        let registers_claim_reduction =
            stage3_registers_claim_reduction_polys(config, materialized.registers_claim_reduction)?;
        let eq_plus_one_outer = stage3_polynomial(
            config,
            "shift outer eq+1",
            EqPlusOnePolynomial::evals(&stage2.product_uniskip.tau_low, None).1,
        )?;
        let eq_plus_one_product = stage3_polynomial(
            config,
            "shift product eq+1",
            EqPlusOnePolynomial::evals(&stage2.batch.product_remainder.opening_point, None).1,
        )?;
        let eq_product = stage3_polynomial(
            config,
            "instruction product eq",
            EqPolynomial::new(stage2.batch.product_remainder.opening_point.clone()).evaluations(),
        )?;
        let eq_spartan = stage3_polynomial(
            config,
            "registers Spartan eq",
            EqPolynomial::new(stage2.product_uniskip.tau_low.clone()).evaluations(),
        )?;

        let shift_gamma2 = prefix.shift_gamma * prefix.shift_gamma;
        let shift_gamma3 = shift_gamma2 * prefix.shift_gamma;
        let shift_gamma4 = shift_gamma3 * prefix.shift_gamma;
        let registers_gamma2 = prefix.registers_gamma * prefix.registers_gamma;

        Ok(Self {
            shift_gamma: prefix.shift_gamma,
            shift_gamma2,
            shift_gamma3,
            shift_gamma4,
            instruction_gamma: prefix.instruction_gamma,
            registers_gamma: prefix.registers_gamma,
            registers_gamma2,
            shift,
            instruction_input,
            registers_claim_reduction,
            eq_plus_one_outer,
            eq_plus_one_product,
            eq_product,
            eq_spartan,
        })
    }

    fn round_sum(&self, point: F, coefficients: Stage3BatchCoefficients<F>) -> F {
        let terms = self.current_len() / 2;
        (0..terms)
            .map(|index| {
                coefficients.shift * self.shift_round_term(index, point)
                    + coefficients.instruction_input
                        * self.instruction_input_round_term(index, point)
                    + coefficients.registers_claim_reduction
                        * self.registers_claim_reduction_round_term(index, point)
            })
            .sum()
    }

    fn bind(&mut self, challenge: F) {
        self.shift.bind(challenge);
        self.instruction_input.bind(challenge);
        self.registers_claim_reduction.bind(challenge);
        self.eq_plus_one_outer
            .bind_with_order(challenge, BindingOrder::LowToHigh);
        self.eq_plus_one_product
            .bind_with_order(challenge, BindingOrder::LowToHigh);
        self.eq_product
            .bind_with_order(challenge, BindingOrder::LowToHigh);
        self.eq_spartan
            .bind_with_order(challenge, BindingOrder::LowToHigh);
    }

    fn output_openings(&self) -> Result<Stage3RegularBatchOutputOpeningClaims<F>, ProverError> {
        Ok(Stage3RegularBatchOutputOpeningClaims {
            shift: SpartanShiftOutputOpeningClaims {
                unexpanded_pc: final_value(
                    "Stage 3 shift unexpanded PC",
                    &self.shift.unexpanded_pc,
                )?,
                pc: final_value("Stage 3 shift PC", &self.shift.pc)?,
                is_virtual: final_value(
                    "Stage 3 shift virtual-instruction flag",
                    &self.shift.is_virtual,
                )?,
                is_first_in_sequence: final_value(
                    "Stage 3 shift first-in-sequence flag",
                    &self.shift.is_first_in_sequence,
                )?,
                is_noop: final_value("Stage 3 shift noop flag", &self.shift.is_noop)?,
            },
            instruction_input: InstructionInputOutputOpeningClaims {
                right_operand_is_rs2: final_value(
                    "Stage 3 instruction-input right operand is rs2",
                    &self.instruction_input.right_operand_is_rs2,
                )?,
                rs2_value: final_value(
                    "Stage 3 instruction-input rs2 value",
                    &self.instruction_input.rs2_value,
                )?,
                right_operand_is_imm: final_value(
                    "Stage 3 instruction-input right operand is imm",
                    &self.instruction_input.right_operand_is_imm,
                )?,
                imm: final_value(
                    "Stage 3 instruction-input immediate",
                    &self.instruction_input.imm,
                )?,
                left_operand_is_rs1: final_value(
                    "Stage 3 instruction-input left operand is rs1",
                    &self.instruction_input.left_operand_is_rs1,
                )?,
                rs1_value: final_value(
                    "Stage 3 instruction-input rs1 value",
                    &self.instruction_input.rs1_value,
                )?,
                left_operand_is_pc: final_value(
                    "Stage 3 instruction-input left operand is PC",
                    &self.instruction_input.left_operand_is_pc,
                )?,
                unexpanded_pc: final_value(
                    "Stage 3 instruction-input unexpanded PC",
                    &self.instruction_input.unexpanded_pc,
                )?,
            },
            registers_claim_reduction: RegistersClaimReductionOutputOpeningClaims {
                rd_write_value: final_value(
                    "Stage 3 registers claim-reduction rd write value",
                    &self.registers_claim_reduction.rd_write,
                )?,
                rs1_value: final_value(
                    "Stage 3 registers claim-reduction rs1 value",
                    &self.registers_claim_reduction.rs1,
                )?,
                rs2_value: final_value(
                    "Stage 3 registers claim-reduction rs2 value",
                    &self.registers_claim_reduction.rs2,
                )?,
            },
        })
    }

    fn expected_outputs(
        &self,
        openings: &Stage3RegularBatchOutputOpeningClaims<F>,
    ) -> Result<Stage3ExpectedOutputs<F>, ProverError> {
        let eq_plus_one_outer = final_value("Stage 3 shift outer eq+1", &self.eq_plus_one_outer)?;
        let eq_plus_one_product =
            final_value("Stage 3 shift product eq+1", &self.eq_plus_one_product)?;
        let eq_product = final_value("Stage 3 instruction product eq", &self.eq_product)?;
        let eq_spartan = final_value("Stage 3 registers Spartan eq", &self.eq_spartan)?;

        let shift = eq_plus_one_outer
            * (openings.shift.unexpanded_pc
                + self.shift_gamma * openings.shift.pc
                + self.shift_gamma2 * openings.shift.is_virtual
                + self.shift_gamma3 * openings.shift.is_first_in_sequence)
            + eq_plus_one_product * self.shift_gamma4 * (F::one() - openings.shift.is_noop);
        let instruction_right = openings.instruction_input.right_operand_is_rs2
            * openings.instruction_input.rs2_value
            + openings.instruction_input.right_operand_is_imm * openings.instruction_input.imm;
        let instruction_left = openings.instruction_input.left_operand_is_rs1
            * openings.instruction_input.rs1_value
            + openings.instruction_input.left_operand_is_pc
                * openings.instruction_input.unexpanded_pc;
        let instruction_input =
            eq_product * (instruction_right + self.instruction_gamma * instruction_left);
        let registers_claim_reduction = eq_spartan
            * (openings.registers_claim_reduction.rd_write_value
                + self.registers_gamma * openings.registers_claim_reduction.rs1_value
                + self.registers_gamma2 * openings.registers_claim_reduction.rs2_value);

        Ok(Stage3ExpectedOutputs {
            shift,
            instruction_input,
            registers_claim_reduction,
        })
    }

    fn shift_round_term(&self, index: usize, point: F) -> F {
        let eq_outer = multilinear_round_eval(&self.eq_plus_one_outer, index, point);
        let eq_product = multilinear_round_eval(&self.eq_plus_one_product, index, point);
        eq_outer
            * (multilinear_round_eval(&self.shift.unexpanded_pc, index, point)
                + self.shift_gamma * multilinear_round_eval(&self.shift.pc, index, point)
                + self.shift_gamma2 * multilinear_round_eval(&self.shift.is_virtual, index, point)
                + self.shift_gamma3
                    * multilinear_round_eval(&self.shift.is_first_in_sequence, index, point))
            + eq_product
                * self.shift_gamma4
                * (F::one() - multilinear_round_eval(&self.shift.is_noop, index, point))
    }

    fn instruction_input_round_term(&self, index: usize, point: F) -> F {
        let right =
            multilinear_round_eval(&self.instruction_input.right_operand_is_rs2, index, point)
                * multilinear_round_eval(&self.instruction_input.rs2_value, index, point)
                + multilinear_round_eval(
                    &self.instruction_input.right_operand_is_imm,
                    index,
                    point,
                ) * multilinear_round_eval(&self.instruction_input.imm, index, point);
        let left =
            multilinear_round_eval(&self.instruction_input.left_operand_is_rs1, index, point)
                * multilinear_round_eval(&self.instruction_input.rs1_value, index, point)
                + multilinear_round_eval(&self.instruction_input.left_operand_is_pc, index, point)
                    * multilinear_round_eval(&self.instruction_input.unexpanded_pc, index, point);
        multilinear_round_eval(&self.eq_product, index, point)
            * (right + self.instruction_gamma * left)
    }

    fn registers_claim_reduction_round_term(&self, index: usize, point: F) -> F {
        multilinear_round_eval(&self.eq_spartan, index, point)
            * (multilinear_round_eval(&self.registers_claim_reduction.rd_write, index, point)
                + self.registers_gamma
                    * multilinear_round_eval(&self.registers_claim_reduction.rs1, index, point)
                + self.registers_gamma2
                    * multilinear_round_eval(&self.registers_claim_reduction.rs2, index, point))
    }

    fn current_len(&self) -> usize {
        self.eq_product.len()
    }
}

impl<F: Field> Stage3ShiftPolys<F> {
    fn bind(&mut self, challenge: F) {
        self.unexpanded_pc
            .bind_with_order(challenge, BindingOrder::LowToHigh);
        self.pc.bind_with_order(challenge, BindingOrder::LowToHigh);
        self.is_virtual
            .bind_with_order(challenge, BindingOrder::LowToHigh);
        self.is_first_in_sequence
            .bind_with_order(challenge, BindingOrder::LowToHigh);
        self.is_noop
            .bind_with_order(challenge, BindingOrder::LowToHigh);
    }
}

impl<F: Field> Stage3InstructionInputPolys<F> {
    fn bind(&mut self, challenge: F) {
        self.right_operand_is_rs2
            .bind_with_order(challenge, BindingOrder::LowToHigh);
        self.rs2_value
            .bind_with_order(challenge, BindingOrder::LowToHigh);
        self.right_operand_is_imm
            .bind_with_order(challenge, BindingOrder::LowToHigh);
        self.imm.bind_with_order(challenge, BindingOrder::LowToHigh);
        self.left_operand_is_rs1
            .bind_with_order(challenge, BindingOrder::LowToHigh);
        self.rs1_value
            .bind_with_order(challenge, BindingOrder::LowToHigh);
        self.left_operand_is_pc
            .bind_with_order(challenge, BindingOrder::LowToHigh);
        self.unexpanded_pc
            .bind_with_order(challenge, BindingOrder::LowToHigh);
    }
}

impl<F: Field> Stage3RegistersClaimReductionPolys<F> {
    fn bind(&mut self, challenge: F) {
        self.rd_write
            .bind_with_order(challenge, BindingOrder::LowToHigh);
        self.rs1.bind_with_order(challenge, BindingOrder::LowToHigh);
        self.rs2.bind_with_order(challenge, BindingOrder::LowToHigh);
    }
}

fn stage3_shift_polys<F: Field>(
    config: Stage3ProverConfig,
    materialized: Stage3ShiftMaterializedOpenings<F>,
) -> Result<Stage3ShiftPolys<F>, ProverError> {
    Ok(Stage3ShiftPolys {
        unexpanded_pc: stage3_polynomial(
            config,
            "shift unexpanded PC",
            materialized.unexpanded_pc,
        )?,
        pc: stage3_polynomial(config, "shift PC", materialized.pc)?,
        is_virtual: stage3_polynomial(
            config,
            "shift virtual-instruction flag",
            materialized.is_virtual,
        )?,
        is_first_in_sequence: stage3_polynomial(
            config,
            "shift first-in-sequence flag",
            materialized.is_first_in_sequence,
        )?,
        is_noop: stage3_polynomial(config, "shift noop flag", materialized.is_noop)?,
    })
}

fn stage3_instruction_input_polys<F: Field>(
    config: Stage3ProverConfig,
    materialized: Stage3InstructionInputMaterializedOpenings<F>,
) -> Result<Stage3InstructionInputPolys<F>, ProverError> {
    Ok(Stage3InstructionInputPolys {
        right_operand_is_rs2: stage3_polynomial(
            config,
            "instruction-input right operand is rs2",
            materialized.right_operand_is_rs2,
        )?,
        rs2_value: stage3_polynomial(
            config,
            "instruction-input rs2 value",
            materialized.rs2_value,
        )?,
        right_operand_is_imm: stage3_polynomial(
            config,
            "instruction-input right operand is imm",
            materialized.right_operand_is_imm,
        )?,
        imm: stage3_polynomial(config, "instruction-input immediate", materialized.imm)?,
        left_operand_is_rs1: stage3_polynomial(
            config,
            "instruction-input left operand is rs1",
            materialized.left_operand_is_rs1,
        )?,
        rs1_value: stage3_polynomial(
            config,
            "instruction-input rs1 value",
            materialized.rs1_value,
        )?,
        left_operand_is_pc: stage3_polynomial(
            config,
            "instruction-input left operand is PC",
            materialized.left_operand_is_pc,
        )?,
        unexpanded_pc: stage3_polynomial(
            config,
            "instruction-input unexpanded PC",
            materialized.unexpanded_pc,
        )?,
    })
}

fn stage3_registers_claim_reduction_polys<F: Field>(
    config: Stage3ProverConfig,
    materialized: Stage3RegistersClaimReductionMaterializedOpenings<F>,
) -> Result<Stage3RegistersClaimReductionPolys<F>, ProverError> {
    Ok(Stage3RegistersClaimReductionPolys {
        rd_write: stage3_polynomial(
            config,
            "registers claim-reduction rd write value",
            materialized.rd_write_value,
        )?,
        rs1: stage3_polynomial(
            config,
            "registers claim-reduction rs1 value",
            materialized.rs1_value,
        )?,
        rs2: stage3_polynomial(
            config,
            "registers claim-reduction rs2 value",
            materialized.rs2_value,
        )?,
    })
}

fn stage3_polynomial<F: Field>(
    config: Stage3ProverConfig,
    label: &'static str,
    values: Vec<F>,
) -> Result<Polynomial<F>, ProverError> {
    let expected_len = 1usize.checked_shl(config.log_t as u32).ok_or_else(|| {
        invalid_sumcheck_output(format!(
            "Stage 3 {label} trace length overflows for log_t={}",
            config.log_t
        ))
    })?;
    if values.len() != expected_len {
        return Err(invalid_sumcheck_output(format!(
            "Stage 3 {label} materialized {} rows, expected {expected_len}",
            values.len()
        )));
    }
    Ok(Polynomial::from(values))
}

fn multilinear_round_eval<F: Field>(poly: &Polynomial<F>, index: usize, point: F) -> F {
    let (lo, hi) = poly.sumcheck_eval_pair(index, BindingOrder::LowToHigh);
    lo + point * (hi - lo)
}

fn final_value<F: Field>(label: &'static str, poly: &Polynomial<F>) -> Result<F, ProverError> {
    let [value] = poly.evals() else {
        return Err(invalid_sumcheck_output(format!(
            "{label} has {} remaining evaluations, expected 1",
            poly.len()
        )));
    };
    Ok(*value)
}

fn reversed_point<F: Field>(point: &[F]) -> Vec<F> {
    point.iter().rev().copied().collect()
}

fn invalid_sumcheck_output(error: impl std::fmt::Display) -> ProverError {
    ProverError::InvalidSumcheckOutput {
        reason: error.to_string(),
    }
}
