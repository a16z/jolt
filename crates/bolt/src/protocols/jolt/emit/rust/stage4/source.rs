use super::Stage4CpuProgram;
use crate::ir::Role;

impl Stage4CpuProgram {
    pub(super) fn emit_entrypoint(&self) -> &'static str {
        match self.role {
            Role::Prover => {
                "pub fn execute_stage4_prover<E, T>(\n\
                 \x20   executor: &mut E,\n\
                 \x20   transcript: &mut T,\n\
                 ) -> Result<Stage4ExecutionArtifacts<Fr>, Stage4KernelError>\n\
                 where\n\
                 \x20   E: Stage4KernelExecutor<Fr>,\n\
                 \x20   T: Transcript<Challenge = Fr>,\n\
                 {\n\
                 \x20   execute_stage4_prover_with_program(&STAGE4_PROGRAM, executor, transcript)\n\
                 }\n\
                 \n\
                 pub fn execute_stage4_prover_with_program<E, T>(\n\
                 \x20   program: &'static Stage4CpuProgramPlan,\n\
                 \x20   executor: &mut E,\n\
                 \x20   transcript: &mut T,\n\
                 ) -> Result<Stage4ExecutionArtifacts<Fr>, Stage4KernelError>\n\
                 where\n\
                 \x20   E: Stage4KernelExecutor<Fr>,\n\
                 \x20   T: Transcript<Challenge = Fr>,\n\
                 {\n\
                 \x20   execute_stage4_program(program, Stage4ExecutionMode::Prover, executor, transcript)\n\
                 }\n"
            }
            Role::Verifier => {
                r#"pub fn verify_stage4<T>(
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

"#
            }
        }
    }
}
