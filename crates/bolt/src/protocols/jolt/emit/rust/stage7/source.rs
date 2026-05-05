use super::Stage7CpuProgram;
use crate::ir::Role;

impl Stage7CpuProgram {
    pub(super) fn emit_entrypoint(&self) -> &'static str {
        match self.role {
            Role::Prover => {
                "pub fn execute_stage7_prover<E, T>(\n\
                 \x20   executor: &mut E,\n\
                 \x20   transcript: &mut T,\n\
                 ) -> Result<Stage7ExecutionArtifacts<Fr>, Stage7KernelError>\n\
                 where\n\
                 \x20   E: Stage7KernelExecutor<Fr>,\n\
                 \x20   T: Transcript<Challenge = Fr>,\n\
                 {\n\
                 \x20   execute_stage7_prover_with_program(&STAGE7_PROGRAM, executor, transcript)\n\
                 }\n\
                 \n\
                 pub fn execute_stage7_prover_with_program<E, T>(\n\
                 \x20   program: &'static Stage7CpuProgramPlan,\n\
                 \x20   executor: &mut E,\n\
                 \x20   transcript: &mut T,\n\
                 ) -> Result<Stage7ExecutionArtifacts<Fr>, Stage7KernelError>\n\
                 where\n\
                 \x20   E: Stage7KernelExecutor<Fr>,\n\
                 \x20   T: Transcript<Challenge = Fr>,\n\
                 {\n\
                 \x20   execute_stage7_program(program, Stage7ExecutionMode::Prover, executor, transcript)\n\
                 }\n"
            }
            Role::Verifier => {
                r#"pub fn verify_stage7<T>(
    proof: &Stage7Proof<Fr>,
    opening_inputs: &[Stage7OpeningInputValue<Fr>],
    transcript: &mut T,
) -> Result<Stage7ExecutionArtifacts<Fr>, VerifyStage7Error>
where
    T: Transcript<Challenge = Fr>,
{
    verify_stage7_with_program(&STAGE7_PROGRAM, proof, opening_inputs, transcript)
}

pub fn verify_stage7_with_program<T>(
    program: &'static Stage7VerifierProgramPlan,
    proof: &Stage7Proof<Fr>,
    opening_inputs: &[Stage7OpeningInputValue<Fr>],
    transcript: &mut T,
) -> Result<Stage7ExecutionArtifacts<Fr>, VerifyStage7Error>
where
    T: Transcript<Challenge = Fr>,
{
    if proof.sumchecks.len() != program.drivers.len() {
        return Err(VerifyStage7Error::UnexpectedProofCount {
            expected: program.drivers.len(),
            got: proof.sumchecks.len(),
        });
    }
    let mut store =
        super::common::ValueStore::with_opening_inputs(opening_inputs, program.opening_inputs)?;
    store.seed_constants(program.field_constants);
    store.seed_point_zeros(program.point_zeros);
    let mut artifacts = Stage7ExecutionArtifacts::default();
    for step in program.steps {
        match step.kind {
            "transcript_squeeze" => {
                let squeeze =
                    find_plan(program.transcript_squeezes, step.symbol).ok_or(VerifyStage7Error::MissingValue {
                        symbol: step.symbol,
                    })?;
                verify_stage7_squeeze(program, squeeze, &mut store, transcript, &mut artifacts)?;
            }
            "transcript_absorb_bytes" => {
                let absorb = find_plan(program.transcript_absorb_bytes, step.symbol).ok_or(
                    VerifyStage7Error::MissingValue {
                        symbol: step.symbol,
                    },
                )?;
                absorb_stage7_bytes(absorb, transcript);
            }
            "sumcheck_driver" => {
                let driver =
                    find_plan(program.drivers, step.symbol).ok_or(VerifyStage7Error::MissingProof {
                        driver: step.symbol,
                    })?;
                verify_stage7_driver(
                    program,
                    driver,
                    proof,
                    &mut store,
                    transcript,
                    &mut artifacts,
                )?;
            }
            _ => {
                return Err(VerifyStage7Error::InvalidProof {
                    driver: step.symbol,
                    reason: "unsupported stage7 program step",
                });
            }
        }
    }
    artifacts
        .opening_batches
        .extend(program.opening_batches.iter());
    Ok(artifacts)
}

pub fn stage7_verifier_program() -> &'static Stage7VerifierProgramPlan {
    &STAGE7_PROGRAM
}

fn verify_stage7_squeeze<T>(
    program: &'static Stage7VerifierProgramPlan,
    squeeze: &'static Stage7TranscriptSqueezePlan,
    store: &mut super::common::ValueStore<Fr>,
    transcript: &mut T,
    artifacts: &mut Stage7ExecutionArtifacts<Fr>,
) -> Result<(), VerifyStage7Error>
where
    T: Transcript<Challenge = Fr>,
{
    let values = transcript.challenge_vector(squeeze.count);
    store.observe_challenge_vector(squeeze, &values, |input, expected, actual| {
        VerifyStage7Error::InvalidInputLength {
            input,
            expected,
            actual,
        }
    })?;
    store
        .evaluate_available_field_exprs(program.field_exprs, super::common::evaluate_field_expr)
        .map_err(VerifyStage7Error::from)?;
    artifacts.challenge_vectors.push(Stage7ChallengeVector {
        symbol: squeeze.symbol,
        values,
    });
    Ok(())
}

fn absorb_stage7_bytes<T>(absorb: &'static Stage7TranscriptAbsorbBytesPlan, transcript: &mut T)
where
    T: Transcript<Challenge = Fr>,
{
    transcript.append(&LabelWithCount(
        absorb.label.as_bytes(),
        absorb.payload.len() as u64,
    ));
    transcript.append_bytes(absorb.payload.as_bytes());
}

fn verify_stage7_driver<T>(
    program: &'static Stage7VerifierProgramPlan,
    driver: &'static Stage7SumcheckDriverPlan,
    proof: &Stage7Proof<Fr>,
    store: &mut super::common::ValueStore<Fr>,
    transcript: &mut T,
    artifacts: &mut Stage7ExecutionArtifacts<Fr>,
) -> Result<(), VerifyStage7Error>
where
    T: Transcript<Challenge = Fr>,
{
    let proof = proof
        .sumchecks
        .get(artifacts.sumchecks.len())
        .ok_or(VerifyStage7Error::MissingProof {
            driver: driver.symbol,
        })?;
    let relation = driver.relation.unwrap_or("<missing>");
    let output = match relation {
        "jolt.stage7.batched" => {
            verify_batched_stage7(program, driver, proof, store, transcript)?
        }
        _ => return Err(VerifyStage7Error::UnsupportedRelation { relation }),
    };
    artifacts.sumchecks.push(output);
    Ok(())
}

fn verify_batched_stage7<T>(
    program: &'static Stage7VerifierProgramPlan,
    driver: &'static Stage7SumcheckDriverPlan,
    proof: &Stage7SumcheckOutput<Fr>,
    store: &mut super::common::ValueStore<Fr>,
    transcript: &mut T,
) -> Result<Stage7SumcheckOutput<Fr>, VerifyStage7Error>
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
        |store, verified| observe_stage7_sumcheck_output(program, store, verified),
        |driver, error| VerifyStage7Error::Sumcheck { driver, error },
    )
}

fn observe_stage7_sumcheck_output<F: Field>(
    program: &'static Stage7VerifierProgramPlan,
    store: &mut super::common::ValueStore<F>,
    output: &Stage7SumcheckOutput<F>,
) -> Result<(), VerifyStage7Error> {
    store.observe_sumcheck_output(
        program.instance_results,
        program.evals,
        output,
        |instance, mut point| {
            match instance.point_order {
                "as_is" => {}
                "reverse" => point.reverse(),
                "bytecode_read_raf" => point = normalize_bytecode_read_raf_point(&point, stage7_trace_rounds(program)?, "stage7.bytecode_read_raf.point")?,
                "stage7_booleanity" => {}
                "instruction_read_raf" => point = normalize_instruction_read_raf_point(&point, "stage7.instruction_read_raf.point")?,
                _ => {
                    return Err(VerifyStage7Error::InvalidProof {
                        driver: output.driver,
                        reason: "unsupported point order",
                    });
                }
            }
            Ok(point)
        },
        |input, expected, actual| VerifyStage7Error::InvalidInputLength {
            input,
            expected,
            actual,
        },
        |symbol| VerifyStage7Error::MissingValue { symbol },
    )?;
    store.evaluate_available_points(
        program.point_slices,
        program.point_concats,
        |input, expected, actual| VerifyStage7Error::InvalidInputLength {
            input,
            expected,
            actual,
        },
    )?;
    store
        .evaluate_available_field_exprs(program.field_exprs, super::common::evaluate_field_expr)
        .map_err(VerifyStage7Error::from)?;
    store.verify_opening_equalities(
        program.opening_equalities,
        |driver, reason| VerifyStage7Error::InvalidProof { driver, reason },
        |symbol| VerifyStage7Error::MissingValue { symbol },
    )
}

fn expected_batched_output_claim(
    program: &'static Stage7VerifierProgramPlan,
    driver: &'static Stage7SumcheckDriverPlan,
    store: &super::common::ValueStore<Fr>,
    evals: &[Stage7NamedEval<Fr>],
    point: &[Fr],
    batching_coeffs: &[Fr],
) -> Result<Fr, VerifyStage7Error> {
    let batch = find_batch(program.batches, driver.symbol, driver.batch)?;
    let claims = batch_claims(program.claims, batch)?;
    let mut expected = Fr::from_u64(0);
    for (claim, coefficient) in claims.iter().zip(batching_coeffs) {
        let instance = program
            .instance_results
            .iter()
            .find(|instance| instance.claim == claim.symbol && instance.source == driver.symbol)
            .ok_or(VerifyStage7Error::MissingClaim {
                batch: batch.symbol,
                claim: claim.symbol,
            })?;
        let local_point = point
            .get(instance.round_offset..instance.round_offset + instance.num_rounds)
            .ok_or(VerifyStage7Error::InvalidInputLength {
                input: instance.symbol,
                expected: instance.round_offset + instance.num_rounds,
                actual: point.len(),
            })?;
        let relation = claim.relation.unwrap_or("<missing>");
        let value = match relation {
            "jolt.stage7.hamming_weight_claim_reduction" => {
                expected_hamming_weight_claim_reduction(program, driver, store, evals, local_point)?
            }
            _ => return Err(VerifyStage7Error::UnsupportedRelation { relation }),
        };
        expected += *coefficient * value;
    }
    Ok(expected)
}

fn expected_hamming_weight_claim_reduction(
    program: &'static Stage7VerifierProgramPlan,
    driver: &'static Stage7SumcheckDriverPlan,
    store: &super::common::ValueStore<Fr>,
    evals: &[Stage7NamedEval<Fr>],
    local_point: &[Fr],
) -> Result<Fr, VerifyStage7Error> {
    let rho_rev = reverse_slice(local_point);
    let booleanity_point = super::common::store_point(store, "stage7.input.stage6.booleanity.InstructionRa_0")?;
    let r_addr_bool =
        booleanity_point
            .get(..local_point.len())
            .ok_or(VerifyStage7Error::InvalidInputLength {
                input: "stage7.input.stage6.booleanity.InstructionRa_0",
                expected: local_point.len(),
                actual: booleanity_point.len(),
            })?;
    let eq_bool = EqPolynomial::<Fr>::mle(&rho_rev, r_addr_bool);
    let gamma = super::common::store_scalar(store, "stage7.hamming_weight_claim_reduction.gamma")?;
    let mut gamma_power = Fr::from_u64(1);
    let mut expected = Fr::from_u64(0);
    let mut eval_plans = program
        .evals
        .iter()
        .filter(|eval| eval.source == driver.symbol)
        .collect::<Vec<_>>();
    eval_plans.sort_by_key(|eval| eval.index);
    for eval_plan in eval_plans {
        let g_i = eval_by_name(evals, eval_plan.name)?;
        let virt_point =
            stage7_virtualization_point(store, eval_plan.oracle, local_point.len())?;
        let eq_virt = EqPolynomial::<Fr>::mle(&rho_rev, virt_point);
        expected += g_i * (gamma_power + gamma_power * gamma * eq_bool
            + gamma_power * gamma.square() * eq_virt);
        gamma_power *= gamma;
        gamma_power *= gamma;
        gamma_power *= gamma;
    }
    Ok(expected)
}

fn stage7_virtualization_point<'a>(
    store: &'a super::common::ValueStore<Fr>,
    oracle: &str,
    log_k_chunk: usize,
) -> Result<&'a [Fr], VerifyStage7Error> {
    let symbol = if oracle.starts_with("InstructionRa_") {
        format!("stage7.input.stage6.instruction_ra_virtual.{oracle}")
    } else if oracle.starts_with("BytecodeRa_") {
        format!("stage7.input.stage6.bytecode_read_raf.{oracle}")
    } else if oracle.starts_with("RamRa_") {
        format!("stage7.input.stage6.ram_ra_virtual.{oracle}")
    } else {
        return Err(VerifyStage7Error::MissingValue {
            symbol: "stage7.hamming_weight_claim_reduction.oracle",
        });
    };
    let point = store.try_point(&symbol).ok_or(VerifyStage7Error::MissingValue {
        symbol: "stage7.hamming_weight_claim_reduction.virtualization_point",
    })?;
    point
        .get(..log_k_chunk)
        .ok_or(VerifyStage7Error::InvalidInputLength {
            input: "stage7.hamming_weight_claim_reduction.virtualization_point",
            expected: log_k_chunk,
            actual: point.len(),
        })
}

fn stage7_trace_rounds(
    program: &'static Stage7VerifierProgramPlan,
) -> Result<usize, VerifyStage7Error> {
    program
        .instance_results
        .iter()
        .find(|instance| instance.relation == "jolt.stage7.hamming_booleanity")
        .map(|instance| instance.num_rounds)
        .ok_or(VerifyStage7Error::MissingValue {
            symbol: "stage7.hamming_booleanity.instance",
        })
}
"#
            }
        }
    }
}
