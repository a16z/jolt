use super::Stage6CpuProgram;
use crate::ir::Role;

impl Stage6CpuProgram {
    pub(super) fn emit_entrypoint(&self) -> &'static str {
        match self.role {
            Role::Prover => {
                "pub fn execute_stage6_prover<E, T>(\n\
                 \x20   executor: &mut E,\n\
                 \x20   transcript: &mut T,\n\
                 ) -> Result<Stage6ExecutionArtifacts<Fr>, Stage6KernelError>\n\
                 where\n\
                 \x20   E: Stage6KernelExecutor<Fr>,\n\
                 \x20   T: Transcript<Challenge = Fr>,\n\
                 {\n\
                 \x20   execute_stage6_prover_with_program(&STAGE6_PROGRAM, executor, transcript)\n\
                 }\n\
                 \n\
                 pub fn execute_stage6_prover_with_program<E, T>(\n\
                 \x20   program: &'static Stage6CpuProgramPlan,\n\
                 \x20   executor: &mut E,\n\
                 \x20   transcript: &mut T,\n\
                 ) -> Result<Stage6ExecutionArtifacts<Fr>, Stage6KernelError>\n\
                 where\n\
                 \x20   E: Stage6KernelExecutor<Fr>,\n\
                 \x20   T: Transcript<Challenge = Fr>,\n\
                 {\n\
                 \x20   execute_stage6_program(program, Stage6ExecutionMode::Prover, executor, transcript)\n\
                 }\n"
            }
            Role::Verifier => {
                r#"pub fn verify_stage6<T>(
    proof: &Stage6Proof<Fr>,
    opening_inputs: &[Stage6OpeningInputValue<Fr>],
    verifier_data: Option<&Stage6VerifierData>,
    transcript: &mut T,
) -> Result<Stage6ExecutionArtifacts<Fr>, VerifyStage6Error>
where
    T: Transcript<Challenge = Fr>,
{
    verify_stage6_with_program(&STAGE6_PROGRAM, proof, opening_inputs, verifier_data, transcript)
}

pub fn verify_stage6_with_program<T>(
    program: &'static Stage6VerifierProgramPlan,
    proof: &Stage6Proof<Fr>,
    opening_inputs: &[Stage6OpeningInputValue<Fr>],
    verifier_data: Option<&Stage6VerifierData>,
    transcript: &mut T,
) -> Result<Stage6ExecutionArtifacts<Fr>, VerifyStage6Error>
where
    T: Transcript<Challenge = Fr>,
{
    if proof.sumchecks.len() != program.drivers.len() {
        return Err(VerifyStage6Error::UnexpectedProofCount {
            expected: program.drivers.len(),
            got: proof.sumchecks.len(),
        });
    }
    let mut store =
        super::common::ValueStore::with_opening_inputs(opening_inputs, program.opening_inputs)?;
    store.seed_constants(program.field_constants);
    store.seed_point_zeros(program.point_zeros);
    let mut artifacts = Stage6ExecutionArtifacts::default();
    for step in program.steps {
        match step.kind {
            "transcript_squeeze" => {
                let squeeze =
                    find_plan(program.transcript_squeezes, step.symbol).ok_or(VerifyStage6Error::MissingValue {
                        symbol: step.symbol,
                    })?;
                verify_stage6_squeeze(program, squeeze, &mut store, transcript, &mut artifacts)?;
            }
            "transcript_absorb_bytes" => {
                let absorb = find_plan(program.transcript_absorb_bytes, step.symbol).ok_or(
                    VerifyStage6Error::MissingValue {
                        symbol: step.symbol,
                    },
                )?;
                absorb_stage6_bytes(absorb, transcript);
            }
            "sumcheck_driver" => {
                let driver =
                    find_plan(program.drivers, step.symbol).ok_or(VerifyStage6Error::MissingProof {
                        driver: step.symbol,
                    })?;
                verify_stage6_driver(
                    program,
                    driver,
                    proof,
                    verifier_data,
                    &mut store,
                    transcript,
                    &mut artifacts,
                )?;
            }
            _ => {
                return Err(VerifyStage6Error::InvalidProof {
                    driver: step.symbol,
                    reason: "unsupported stage6 program step",
                });
            }
        }
    }
    artifacts
        .opening_batches
        .extend(program.opening_batches.iter());
    Ok(artifacts)
}

pub fn stage6_verifier_program() -> &'static Stage6VerifierProgramPlan {
    &STAGE6_PROGRAM
}

fn verify_stage6_squeeze<T>(
    program: &'static Stage6VerifierProgramPlan,
    squeeze: &'static Stage6TranscriptSqueezePlan,
    store: &mut super::common::ValueStore<Fr>,
    transcript: &mut T,
    artifacts: &mut Stage6ExecutionArtifacts<Fr>,
) -> Result<(), VerifyStage6Error>
where
    T: Transcript<Challenge = Fr>,
{
    let values = transcript.challenge_vector(squeeze.count);
    store.observe_challenge_vector(squeeze, &values, |input, expected, actual| {
        VerifyStage6Error::InvalidInputLength {
            input,
            expected,
            actual,
        }
    })?;
    store
        .evaluate_available_field_exprs(program.field_exprs, super::common::evaluate_field_expr)
        .map_err(VerifyStage6Error::from)?;
    artifacts.challenge_vectors.push(Stage6ChallengeVector {
        symbol: squeeze.symbol,
        values,
    });
    Ok(())
}

fn absorb_stage6_bytes<T>(absorb: &'static Stage6TranscriptAbsorbBytesPlan, transcript: &mut T)
where
    T: Transcript<Challenge = Fr>,
{
    transcript.append(&LabelWithCount(
        absorb.label.as_bytes(),
        absorb.payload.len() as u64,
    ));
    transcript.append_bytes(absorb.payload.as_bytes());
}

fn verify_stage6_driver<T>(
    program: &'static Stage6VerifierProgramPlan,
    driver: &'static Stage6SumcheckDriverPlan,
    proof: &Stage6Proof<Fr>,
    verifier_data: Option<&Stage6VerifierData>,
    store: &mut super::common::ValueStore<Fr>,
    transcript: &mut T,
    artifacts: &mut Stage6ExecutionArtifacts<Fr>,
) -> Result<(), VerifyStage6Error>
where
    T: Transcript<Challenge = Fr>,
{
    let proof = proof
        .sumchecks
        .get(artifacts.sumchecks.len())
        .ok_or(VerifyStage6Error::MissingProof {
            driver: driver.symbol,
        })?;
    let relation = driver.relation.unwrap_or("<missing>");
    let output = match relation {
        "jolt.stage6.batched" => {
            verify_batched_stage6(program, driver, proof, verifier_data, store, transcript)?
        }
        _ => return Err(VerifyStage6Error::UnsupportedRelation { relation }),
    };
    artifacts.sumchecks.push(output);
    Ok(())
}

fn verify_batched_stage6<T>(
    program: &'static Stage6VerifierProgramPlan,
    driver: &'static Stage6SumcheckDriverPlan,
    proof: &Stage6SumcheckOutput<Fr>,
    verifier_data: Option<&Stage6VerifierData>,
    store: &mut super::common::ValueStore<Fr>,
    transcript: &mut T,
) -> Result<Stage6SumcheckOutput<Fr>, VerifyStage6Error>
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
            expected_batched_output_claim(
                program,
                driver,
                verifier_data,
                store,
                evals,
                point,
                batching_coeffs,
            )
        },
        |store, verified| observe_stage6_sumcheck_output(program, store, verified),
        |driver, error| VerifyStage6Error::Sumcheck { driver, error },
    )
}

fn observe_stage6_sumcheck_output<F: Field>(
    program: &'static Stage6VerifierProgramPlan,
    store: &mut super::common::ValueStore<F>,
    output: &Stage6SumcheckOutput<F>,
) -> Result<(), VerifyStage6Error> {
    store.observe_sumcheck_output(
        program.instance_results,
        program.evals,
        output,
        |instance, mut point| {
            match instance.point_order {
                "as_is" => {}
                "reverse" => point.reverse(),
                "bytecode_read_raf" => point = normalize_bytecode_read_raf_point(&point, stage6_trace_rounds(program)?, "stage6.bytecode_read_raf.point")?,
                "stage6_booleanity" => {}
                "instruction_read_raf" => point = normalize_instruction_read_raf_point(&point, "stage6.instruction_read_raf.point")?,
                _ => {
                    return Err(VerifyStage6Error::InvalidProof {
                        driver: output.driver,
                        reason: "unsupported point order",
                    });
                }
            }
            Ok(point)
        },
        |input, expected, actual| VerifyStage6Error::InvalidInputLength {
            input,
            expected,
            actual,
        },
        |symbol| VerifyStage6Error::MissingValue { symbol },
    )?;
    store.evaluate_available_points(
        program.point_slices,
        program.point_concats,
        |input, expected, actual| VerifyStage6Error::InvalidInputLength {
            input,
            expected,
            actual,
        },
    )?;
    store
        .evaluate_available_field_exprs(program.field_exprs, super::common::evaluate_field_expr)
        .map_err(VerifyStage6Error::from)?;
    store.verify_opening_equalities(
        program.opening_equalities,
        |driver, reason| VerifyStage6Error::InvalidProof { driver, reason },
        |symbol| VerifyStage6Error::MissingValue { symbol },
    )
}

fn expected_batched_output_claim(
    program: &'static Stage6VerifierProgramPlan,
    driver: &'static Stage6SumcheckDriverPlan,
    verifier_data: Option<&Stage6VerifierData>,
    store: &super::common::ValueStore<Fr>,
    evals: &[Stage6NamedEval<Fr>],
    point: &[Fr],
    batching_coeffs: &[Fr],
) -> Result<Fr, VerifyStage6Error> {
    let batch = find_batch(program.batches, driver.symbol, driver.batch)?;
    let claims = batch_claims(program.claims, batch)?;
    let mut expected = Fr::from_u64(0);
    for (claim, coefficient) in claims.iter().zip(batching_coeffs) {
        let instance = program
            .instance_results
            .iter()
            .find(|instance| instance.claim == claim.symbol && instance.source == driver.symbol)
            .ok_or(VerifyStage6Error::MissingClaim {
                batch: batch.symbol,
                claim: claim.symbol,
            })?;
        let local_point = point
            .get(instance.round_offset..instance.round_offset + instance.num_rounds)
            .ok_or(VerifyStage6Error::InvalidInputLength {
                input: instance.symbol,
                expected: instance.round_offset + instance.num_rounds,
                actual: point.len(),
            })?;
        let relation = claim.relation.unwrap_or("<missing>");
        let value = match relation {
            "jolt.stage6.bytecode_read_raf" => {
                let data = verifier_data
                    .and_then(|data| data.bytecode_read_raf.as_ref())
                    .ok_or(VerifyStage6Error::MissingValue {
                        symbol: "stage6.bytecode_read_raf.data",
                })?;
                expected_bytecode_read_raf(program, data, store, evals, local_point)?
            }
            "jolt.stage6.booleanity" => {
                expected_booleanity(program, store, evals, local_point)?
            }
            "jolt.stage6.hamming_booleanity" => {
                expected_hamming_booleanity(store, evals, local_point)?
            }
            "jolt.stage6.ram_ra_virtual" => {
                expected_ram_ra_virtual(store, evals, local_point)?
            }
            "jolt.stage6.instruction_ra_virtual" => {
                expected_instruction_ra_virtual(program, store, evals, local_point)?
            }
            "jolt.stage6.inc_claim_reduction" => {
                expected_inc_claim_reduction(store, evals, local_point)?
            }
            _ => return Err(VerifyStage6Error::UnsupportedRelation { relation }),
        };
        expected += *coefficient * value;
    }
    Ok(expected)
}

fn expected_bytecode_read_raf(
    program: &'static Stage6VerifierProgramPlan,
    data: &Stage6BytecodeReadRafData,
    store: &super::common::ValueStore<Fr>,
    evals: &[Stage6NamedEval<Fr>],
    local_point: &[Fr],
) -> Result<Fr, VerifyStage6Error> {
    let log_t = stage6_trace_rounds(program)?;
    Ok(expected_stage67_bytecode_read_raf(
        &data.entries,
        data.entry_bytecode_index,
        data.num_lookup_tables,
        store,
        evals,
        local_point,
        log_t,
        &STAGE6_BYTECODE_SYMBOLS,
    )?)
}

fn expected_booleanity(
    program: &'static Stage6VerifierProgramPlan,
    store: &super::common::ValueStore<Fr>,
    evals: &[Stage6NamedEval<Fr>],
    local_point: &[Fr],
) -> Result<Fr, VerifyStage6Error> {
    let log_t = stage6_trace_rounds(program)?;
    Ok(expected_stage67_booleanity(store, evals, local_point, log_t, &STAGE6_RELATION_SYMBOLS)?)
}

fn expected_hamming_booleanity(
    store: &super::common::ValueStore<Fr>,
    evals: &[Stage6NamedEval<Fr>],
    local_point: &[Fr],
) -> Result<Fr, VerifyStage6Error> {
    Ok(expected_stage67_hamming_booleanity(store, evals, local_point, &STAGE6_RELATION_SYMBOLS)?)
}

fn expected_ram_ra_virtual(
    store: &super::common::ValueStore<Fr>,
    evals: &[Stage6NamedEval<Fr>],
    local_point: &[Fr],
) -> Result<Fr, VerifyStage6Error> {
    Ok(expected_stage67_ram_ra_virtual(store, evals, local_point, &STAGE6_RELATION_SYMBOLS)?)
}

fn expected_instruction_ra_virtual(
    program: &'static Stage6VerifierProgramPlan,
    store: &super::common::ValueStore<Fr>,
    evals: &[Stage6NamedEval<Fr>],
    local_point: &[Fr],
) -> Result<Fr, VerifyStage6Error> {
    Ok(expected_stage67_instruction_ra_virtual(program.opening_inputs, store, evals, local_point, &STAGE6_RELATION_SYMBOLS)?)
}

fn expected_inc_claim_reduction(
    store: &super::common::ValueStore<Fr>,
    evals: &[Stage6NamedEval<Fr>],
    local_point: &[Fr],
) -> Result<Fr, VerifyStage6Error> {
    Ok(expected_stage67_inc_claim_reduction(store, evals, local_point, &STAGE6_RELATION_SYMBOLS)?)
}

fn stage6_trace_rounds(
    program: &'static Stage6VerifierProgramPlan,
) -> Result<usize, VerifyStage6Error> {
    Ok(stage67_trace_rounds(program.instance_results, &STAGE6_RELATION_SYMBOLS)?)
}
"#
            }
        }
    }
}
