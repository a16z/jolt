use super::Stage3CpuProgram;

impl Stage3CpuProgram {
    pub(super) fn emit_prover_entrypoint() -> &'static str {
        "pub fn execute_stage3_prover<E, T>(\n\
         \x20   executor: &mut E,\n\
         \x20   transcript: &mut T,\n\
         ) -> Result<Stage3ExecutionArtifacts<Fr>, Stage3KernelError>\n\
         where\n\
         \x20   E: Stage3KernelExecutor<Fr>,\n\
         \x20   T: Transcript<Challenge = Fr>,\n\
         {\n\
         \x20   execute_stage3_prover_with_program(&STAGE3_PROGRAM, executor, transcript)\n\
         }\n\
         \n\
         pub fn execute_stage3_prover_with_program<E, T>(\n\
         \x20   program: &'static Stage3CpuProgramPlan,\n\
         \x20   executor: &mut E,\n\
         \x20   transcript: &mut T,\n\
         ) -> Result<Stage3ExecutionArtifacts<Fr>, Stage3KernelError>\n\
         where\n\
         \x20   E: Stage3KernelExecutor<Fr>,\n\
         \x20   T: Transcript<Challenge = Fr>,\n\
         {\n\
         \x20   execute_stage3_program(program, Stage3ExecutionMode::Prover, executor, transcript)\n\
         }\n"
    }

    pub(super) fn emit_verifier_entrypoint() -> &'static str {
        r#"pub fn verify_stage3<T>(
    proof: &Stage3Proof<Fr>,
    opening_inputs: &[Stage3OpeningInputValue<Fr>],
    transcript: &mut T,
) -> Result<Stage3ExecutionArtifacts<Fr>, VerifyStage3Error>
where
    T: Transcript<Challenge = Fr>,
{
    verify_stage3_with_program(&STAGE3_PROGRAM, proof, opening_inputs, transcript)
}

pub fn verify_stage3_with_program<T>(
    program: &'static Stage3VerifierProgramPlan,
    proof: &Stage3Proof<Fr>,
    opening_inputs: &[Stage3OpeningInputValue<Fr>],
    transcript: &mut T,
) -> Result<Stage3ExecutionArtifacts<Fr>, VerifyStage3Error>
where
    T: Transcript<Challenge = Fr>,
{
    if proof.sumchecks.len() != program.drivers.len() {
        return Err(VerifyStage3Error::UnexpectedProofCount {
            expected: program.drivers.len(),
            got: proof.sumchecks.len(),
        });
    }
    let mut store =
        super::common::ValueStore::with_opening_inputs(opening_inputs, program.opening_inputs)?;
    store.seed_constants(program.field_constants);
    let mut artifacts = Stage3ExecutionArtifacts::default();
    for step in program.steps {
        match step.kind {
            "transcript_squeeze" => {
                let squeeze =
                    find_plan(program.transcript_squeezes, step.symbol).ok_or(VerifyStage3Error::MissingValue {
                        symbol: step.symbol,
                    })?;
                verify_stage3_squeeze(program, squeeze, &mut store, transcript, &mut artifacts)?;
            }
            "sumcheck_driver" => {
                let driver =
                    find_plan(program.drivers, step.symbol).ok_or(VerifyStage3Error::MissingProof {
                        driver: step.symbol,
                    })?;
                verify_stage3_driver(program, driver, proof, &mut store, transcript, &mut artifacts)?;
            }
            _ => {
                return Err(VerifyStage3Error::InvalidProof {
                    driver: step.symbol,
                    reason: "unsupported stage3 program step",
                });
            }
        }
    }
    artifacts
        .opening_batches
        .extend(program.opening_batches.iter());
    Ok(artifacts)
}

pub fn stage3_verifier_program() -> &'static Stage3VerifierProgramPlan {
    &STAGE3_PROGRAM
}

fn verify_stage3_squeeze<T>(
    program: &'static Stage3VerifierProgramPlan,
    squeeze: &'static Stage3TranscriptSqueezePlan,
    store: &mut super::common::ValueStore<Fr>,
    transcript: &mut T,
    artifacts: &mut Stage3ExecutionArtifacts<Fr>,
) -> Result<(), VerifyStage3Error>
where
    T: Transcript<Challenge = Fr>,
{
    let values = transcript.challenge_vector(squeeze.count);
    store.observe_challenge_vector(squeeze, &values, |input, expected, actual| {
        VerifyStage3Error::InvalidInputLength {
            input,
            expected,
            actual,
        }
    })?;
    store
        .evaluate_available_field_exprs(program.field_exprs, super::common::evaluate_field_expr)
        .map_err(VerifyStage3Error::from)?;
    artifacts.challenge_vectors.push(Stage3ChallengeVector {
        symbol: squeeze.symbol,
        values,
    });
    Ok(())
}

fn verify_stage3_driver<T>(
    program: &'static Stage3VerifierProgramPlan,
    driver: &'static Stage3SumcheckDriverPlan,
    proof: &Stage3Proof<Fr>,
    store: &mut super::common::ValueStore<Fr>,
    transcript: &mut T,
    artifacts: &mut Stage3ExecutionArtifacts<Fr>,
) -> Result<(), VerifyStage3Error>
where
    T: Transcript<Challenge = Fr>,
{
    let proof = proof
        .sumchecks
        .get(artifacts.sumchecks.len())
        .ok_or(VerifyStage3Error::MissingProof {
            driver: driver.symbol,
        })?;
    let relation = driver.relation.unwrap_or("<missing>");
    let output = match relation {
        "jolt.stage3.batched" => {
            verify_batched_stage3(program, driver, proof, store, transcript)?
        }
        _ => {
            return Err(VerifyStage3Error::UnsupportedRelation {
                relation,
            });
        }
    };
    artifacts.sumchecks.push(output);
    Ok(())
}

fn verify_batched_stage3<T>(
    program: &'static Stage3VerifierProgramPlan,
    driver: &'static Stage3SumcheckDriverPlan,
    proof: &Stage3SumcheckOutput<Fr>,
    store: &mut super::common::ValueStore<Fr>,
    transcript: &mut T,
) -> Result<Stage3SumcheckOutput<Fr>, VerifyStage3Error>
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
        |store, verified| observe_stage3_sumcheck_output(program, store, verified),
        |driver, error| VerifyStage3Error::Sumcheck { driver, error },
    )
}

fn observe_stage3_sumcheck_output<F: Field>(
    program: &'static Stage3VerifierProgramPlan,
    store: &mut super::common::ValueStore<F>,
    output: &Stage3SumcheckOutput<F>,
) -> Result<(), VerifyStage3Error> {
    store.observe_sumcheck_output(
        program.instance_results,
        program.evals,
        output,
        |instance, mut point| {
            match instance.point_order {
                "as_is" => {}
                "reverse" => point.reverse(),
                _ => {
                    return Err(VerifyStage3Error::InvalidProof {
                        driver: output.driver,
                        reason: "unsupported point order",
                    });
                }
            }
            Ok(point)
        },
        |input, expected, actual| VerifyStage3Error::InvalidInputLength {
            input,
            expected,
            actual,
        },
        |symbol| VerifyStage3Error::MissingValue { symbol },
    )?;
    store.evaluate_available_points(
        program.point_slices,
        program.point_concats,
        |input, expected, actual| VerifyStage3Error::InvalidInputLength {
            input,
            expected,
            actual,
        },
    )?;
    store
        .evaluate_available_field_exprs(program.field_exprs, super::common::evaluate_field_expr)
        .map_err(VerifyStage3Error::from)?;
    store.verify_opening_equalities(
        program.opening_equalities,
        |driver, reason| VerifyStage3Error::InvalidProof { driver, reason },
        |symbol| VerifyStage3Error::MissingValue { symbol },
    )
}

fn expected_batched_output_claim(
    program: &'static Stage3VerifierProgramPlan,
    driver: &'static Stage3SumcheckDriverPlan,
    store: &super::common::ValueStore<Fr>,
    evals: &[Stage3NamedEval<Fr>],
    point: &[Fr],
    batching_coeffs: &[Fr],
) -> Result<Fr, VerifyStage3Error> {
    let batch = find_batch(program.batches, driver.symbol, driver.batch)?;
    let claims = batch_claims(program.claims, batch)?;
    let mut expected = Fr::from_u64(0);
    for (claim, coefficient) in claims.iter().zip(batching_coeffs) {
        let instance = program
            .instance_results
            .iter()
            .find(|instance| instance.claim == claim.symbol && instance.source == driver.symbol)
            .ok_or(VerifyStage3Error::MissingClaim {
                batch: batch.symbol,
                claim: claim.symbol,
            })?;
        let local_point = point
            .get(instance.round_offset..instance.round_offset + instance.num_rounds)
            .ok_or(VerifyStage3Error::InvalidInputLength {
                input: instance.symbol,
                expected: instance.round_offset + instance.num_rounds,
                actual: point.len(),
            })?;
        let value = match instance.relation {
            "jolt.stage3.spartan_shift" => {
                expected_spartan_shift(store, evals, local_point)?
            }
            "jolt.stage3.instruction_input" => {
                expected_instruction_input(store, evals, local_point)?
            }
            "jolt.stage3.registers_claim_reduction" => {
                expected_registers(store, evals, local_point)?
            }
            _ => {
                return Err(VerifyStage3Error::UnsupportedRelation {
                    relation: instance.relation,
                });
            }
        };
        expected += *coefficient * value;
    }
    Ok(expected)
}

fn expected_spartan_shift(
    store: &super::common::ValueStore<Fr>,
    evals: &[Stage3NamedEval<Fr>],
    local_point: &[Fr],
) -> Result<Fr, VerifyStage3Error> {
    let opening_point = reverse_slice(local_point);
    let eq_outer =
        EqPlusOnePolynomial::<Fr>::new(super::common::store_point(store, "stage3.input.stage1.NextPC")?.to_vec())
            .evaluate(&opening_point);
    let eq_product = EqPlusOnePolynomial::<Fr>::new(
        super::common::store_point(store, "stage3.input.stage2.product_virtual.NextIsNoop")?
            .to_vec(),
    )
    .evaluate(&opening_point);
    let weighted_outer = eval_by_name(evals, "stage3.spartan_shift.eval.UnexpandedPC")?
        + super::common::store_scalar(store, "stage3.spartan_shift.gamma")?
            * eval_by_name(evals, "stage3.spartan_shift.eval.PC")?
        + super::common::store_scalar(store, "stage3.spartan_shift.gamma2")?
            * eval_by_name(evals, "stage3.spartan_shift.eval.OpFlagVirtualInstruction")?
        + super::common::store_scalar(store, "stage3.spartan_shift.gamma3")?
            * eval_by_name(evals, "stage3.spartan_shift.eval.OpFlagIsFirstInSequence")?;
    Ok(eq_outer * weighted_outer
        + super::common::store_scalar(store, "stage3.spartan_shift.gamma4")?
            * eq_product
            * (Fr::from_u64(1)
                - eval_by_name(evals, "stage3.spartan_shift.eval.InstructionFlagIsNoop")?))
}

fn expected_instruction_input(
    store: &super::common::ValueStore<Fr>,
    evals: &[Stage3NamedEval<Fr>],
    local_point: &[Fr],
) -> Result<Fr, VerifyStage3Error> {
    let opening_point = reverse_slice(local_point);
    let eq_eval = EqPolynomial::<Fr>::mle(
        &opening_point,
        super::common::store_point(store, "stage3.input.stage2.product_virtual.LeftInstructionInput")?,
    );
    let left = eval_by_name(
        evals,
        "stage3.instruction_input.eval.InstructionFlagLeftOperandIsRs1Value",
    )? * eval_by_name(evals, "stage3.instruction_input.eval.Rs1Value")?
        + eval_by_name(
            evals,
            "stage3.instruction_input.eval.InstructionFlagLeftOperandIsPC",
        )? * eval_by_name(evals, "stage3.instruction_input.eval.UnexpandedPC")?;
    let right = eval_by_name(
        evals,
        "stage3.instruction_input.eval.InstructionFlagRightOperandIsRs2Value",
    )? * eval_by_name(evals, "stage3.instruction_input.eval.Rs2Value")?
        + eval_by_name(
            evals,
            "stage3.instruction_input.eval.InstructionFlagRightOperandIsImm",
        )? * eval_by_name(evals, "stage3.instruction_input.eval.Imm")?;
    Ok(eq_eval * (right + super::common::store_scalar(store, "stage3.instruction_input.gamma")? * left))
}

fn expected_registers(
    store: &super::common::ValueStore<Fr>,
    evals: &[Stage3NamedEval<Fr>],
    local_point: &[Fr],
) -> Result<Fr, VerifyStage3Error> {
    let opening_point = reverse_slice(local_point);
    let eq_eval = EqPolynomial::<Fr>::mle(
        &opening_point,
        super::common::store_point(store, "stage3.input.stage1.RdWriteValue")?,
    );
    Ok(eq_eval
        * (eval_by_name(evals, "stage3.registers_claim_reduction.eval.RdWriteValue")?
            + super::common::store_scalar(store, "stage3.registers.gamma")?
                * eval_by_name(evals, "stage3.registers_claim_reduction.eval.Rs1Value")?
            + super::common::store_scalar(store, "stage3.registers.gamma2")?
                * eval_by_name(evals, "stage3.registers_claim_reduction.eval.Rs2Value")?))
}

"#
    }
}
