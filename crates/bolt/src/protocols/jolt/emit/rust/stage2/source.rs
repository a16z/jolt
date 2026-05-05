use super::Stage2CpuProgram;

impl Stage2CpuProgram {
    pub(super) fn emit_prover_entrypoint() -> &'static str {
        "pub fn execute_stage2_prover<E, T>(\n\
         \x20   executor: &mut E,\n\
         \x20   transcript: &mut T,\n\
         ) -> Result<Stage2ExecutionArtifacts<Fr>, Stage2KernelError>\n\
         where\n\
         \x20   E: Stage2KernelExecutor<Fr>,\n\
         \x20   T: Transcript<Challenge = Fr>,\n\
         {\n\
         \x20   execute_stage2_prover_with_program(&STAGE2_PROGRAM, executor, transcript)\n\
         }\n\
         \n\
         pub fn execute_stage2_prover_with_program<E, T>(\n\
         \x20   program: &'static Stage2CpuProgramPlan,\n\
         \x20   executor: &mut E,\n\
         \x20   transcript: &mut T,\n\
         ) -> Result<Stage2ExecutionArtifacts<Fr>, Stage2KernelError>\n\
         where\n\
         \x20   E: Stage2KernelExecutor<Fr>,\n\
         \x20   T: Transcript<Challenge = Fr>,\n\
         {\n\
         \x20   execute_stage2_program(program, Stage2ExecutionMode::Prover, executor, transcript)\n\
         }\n"
    }

    pub(super) fn emit_verifier_entrypoint() -> &'static str {
        r#"const PRODUCT_VIRTUAL_UNISKIP_DOMAIN_START: i64 = -1;
const PRODUCT_VIRTUAL_UNISKIP_DOMAIN_SIZE: usize = 3;

pub fn verify_stage2<T>(
    proof: &Stage2Proof<Fr>,
    opening_inputs: &[Stage2OpeningInputValue<Fr>],
    ram: Option<&Stage2RamData<'_>>,
    transcript: &mut T,
) -> Result<Stage2ExecutionArtifacts<Fr>, VerifyStage2Error>
where
    T: Transcript<Challenge = Fr>,
{
    verify_stage2_with_program(&STAGE2_PROGRAM, proof, opening_inputs, ram, transcript)
}

pub fn verify_stage2_with_program<T>(
    program: &'static Stage2VerifierProgramPlan,
    proof: &Stage2Proof<Fr>,
    opening_inputs: &[Stage2OpeningInputValue<Fr>],
    ram: Option<&Stage2RamData<'_>>,
    transcript: &mut T,
) -> Result<Stage2ExecutionArtifacts<Fr>, VerifyStage2Error>
where
    T: Transcript<Challenge = Fr>,
{
    if proof.sumchecks.len() != program.drivers.len() {
        return Err(VerifyStage2Error::UnexpectedProofCount {
            expected: program.drivers.len(),
            got: proof.sumchecks.len(),
        });
    }
    let mut store = Stage2ValueStore::with_opening_inputs(opening_inputs);
    store.seed_constants(program);
    let mut artifacts = Stage2ExecutionArtifacts::default();
    if program.steps.is_empty() {
        for squeeze in program.transcript_squeezes {
            verify_stage2_squeeze(program, squeeze, &mut store, transcript, &mut artifacts)?;
        }
        for driver in program.drivers {
            verify_stage2_driver(program, driver, proof, ram, &mut store, transcript, &mut artifacts)?;
        }
    } else {
        for step in program.steps {
            match step.kind {
                "transcript_squeeze" => {
                    let squeeze = find_plan(program.transcript_squeezes, step.symbol).ok_or(VerifyStage2Error::MissingValue {
                        symbol: step.symbol,
                    })?;
                    verify_stage2_squeeze(program, squeeze, &mut store, transcript, &mut artifacts)?;
                }
                "sumcheck_driver" => {
                    let driver = find_plan(program.drivers, step.symbol).ok_or(VerifyStage2Error::MissingProof {
                        driver: step.symbol,
                    })?;
                    verify_stage2_driver(program, driver, proof, ram, &mut store, transcript, &mut artifacts)?;
                }
                _ => {
                    return Err(VerifyStage2Error::InvalidProof {
                        driver: step.symbol,
                        reason: "unsupported stage2 program step",
                    });
                }
            }
        }
    }
    artifacts
        .opening_batches
        .extend(program.opening_batches.iter());
    Ok(artifacts)
}

pub fn stage2_verifier_program() -> &'static Stage2VerifierProgramPlan {
    &STAGE2_PROGRAM
}

fn verify_stage2_squeeze<T>(
    program: &'static Stage2VerifierProgramPlan,
    squeeze: &'static Stage2TranscriptSqueezePlan,
    store: &mut Stage2ValueStore<Fr>,
    transcript: &mut T,
    artifacts: &mut Stage2ExecutionArtifacts<Fr>,
) -> Result<(), VerifyStage2Error>
where
    T: Transcript<Challenge = Fr>,
{
    let values = transcript.challenge_vector(squeeze.count);
    store.observe_challenge_vector(program, squeeze, &values)?;
    artifacts.challenge_vectors.push(Stage2ChallengeVector {
        symbol: squeeze.symbol,
        values,
    });
    Ok(())
}

fn verify_stage2_driver<T>(
    program: &'static Stage2VerifierProgramPlan,
    driver: &'static Stage2SumcheckDriverPlan,
    proof: &Stage2Proof<Fr>,
    ram: Option<&Stage2RamData<'_>>,
    store: &mut Stage2ValueStore<Fr>,
    transcript: &mut T,
    artifacts: &mut Stage2ExecutionArtifacts<Fr>,
) -> Result<(), VerifyStage2Error>
where
    T: Transcript<Challenge = Fr>,
{
    let proof = proof
        .sumchecks
        .get(artifacts.sumchecks.len())
        .ok_or(VerifyStage2Error::MissingProof {
            driver: driver.symbol,
        })?;
    let output = match driver.relation {
        "jolt.stage2.product_virtual.uniskip" => {
            verify_product_virtual_uniskip(program, driver, proof, store, transcript)?
        }
        "jolt.stage2.batched" => verify_batched_stage2(program, driver, proof, ram, store, transcript)?,
        relation => return Err(VerifyStage2Error::UnsupportedRelation { relation }),
    };
    artifacts.sumchecks.push(output);
    Ok(())
}

fn verify_product_virtual_uniskip<T>(
    program: &'static Stage2VerifierProgramPlan,
    driver: &'static Stage2SumcheckDriverPlan,
    proof: &Stage2SumcheckOutput<Fr>,
    store: &mut Stage2ValueStore<Fr>,
    transcript: &mut T,
) -> Result<Stage2SumcheckOutput<Fr>, VerifyStage2Error>
where
    T: Transcript<Challenge = Fr>,
{
    validate_driver_symbol(driver, proof)?;
    let [poly] = proof.proof.round_polynomials.as_slice() else {
        return Err(VerifyStage2Error::InvalidProof {
            driver: driver.symbol,
            reason: "unexpected product uniskip round count",
        });
    };
    if polynomial_degree(poly) > driver.degree {
        return Err(VerifyStage2Error::InvalidProof {
            driver: driver.symbol,
            reason: "product uniskip polynomial exceeds degree bound",
        });
    }
    let batch = find_batch(program.batches, driver.symbol, driver.batch)?;
    let claim = batch_claims(program.claims, batch)?
        .into_iter()
        .next()
        .ok_or(VerifyStage2Error::MissingClaim {
            batch: batch.symbol,
            claim: "stage2.product_virtual.uniskip.input",
        })?;
    let input_claim = store.claim_value(program, claim)?;
    if !product_uniskip_sum_matches(poly, input_claim) {
        return Err(VerifyStage2Error::InvalidProof {
            driver: driver.symbol,
            reason: "product uniskip input claim mismatch",
        });
    }
    append_univariate_poly(transcript, driver.round_label, poly);
    let r0 = transcript.challenge();
    if !proof.point.is_empty() && proof.point != [r0] {
        return Err(VerifyStage2Error::InvalidProof {
            driver: driver.symbol,
            reason: "product uniskip point mismatch",
        });
    }
    let eval = poly.evaluate(r0);
    append_labeled_scalar(transcript, "opening_claim", &eval);
    let output = Stage2SumcheckOutput {
        driver: driver.symbol,
        point: vec![r0],
        evals: driver_evals(program, driver.symbol, eval),
        proof: proof.proof.clone(),
    };
    verify_named_evals(driver.symbol, &output.evals, &proof.evals)?;
    store.observe_sumcheck_output(program, &output)?;
    Ok(output)
}

fn verify_batched_stage2<T>(
    program: &'static Stage2VerifierProgramPlan,
    driver: &'static Stage2SumcheckDriverPlan,
    proof: &Stage2SumcheckOutput<Fr>,
    ram: Option<&Stage2RamData<'_>>,
    store: &mut Stage2ValueStore<Fr>,
    transcript: &mut T,
) -> Result<Stage2SumcheckOutput<Fr>, VerifyStage2Error>
where
    T: Transcript<Challenge = Fr>,
{
    validate_driver_symbol(driver, proof)?;
    let batch = find_batch(program.batches, driver.symbol, driver.batch)?;
    let claims = batch_claims(program.claims, batch)?;
    let input_claims = store.batch_claim_values(program, batch)?;
    for claim in &input_claims {
        append_labeled_scalar(transcript, batch.claim_label, claim);
    }
    let batching_coeffs = transcript.challenge_vector(claims.len());
    let claimed_sum = input_claims
        .iter()
        .zip(claims.iter())
        .zip(&batching_coeffs)
        .map(|((claim, plan), coefficient)| {
            claim.mul_pow_2(driver.num_rounds - plan.num_rounds) * *coefficient
        })
        .sum::<Fr>();
    let claim = SumcheckClaim::new(driver.num_rounds, driver.degree, claimed_sum);
    let round_proofs = proof
        .proof
        .round_polynomials
        .iter()
        .map(|poly| CompressedLabeledRoundPoly::new(poly, driver.round_label.as_bytes()))
        .collect::<Vec<_>>();
    let output = SumcheckVerifier::verify(&claim, &round_proofs, transcript)
        .map_err(|error| VerifyStage2Error::Sumcheck {
            driver: driver.symbol,
            error,
        })?;
    if !proof.point.is_empty() && proof.point != output.point {
        return Err(VerifyStage2Error::InvalidProof {
            driver: driver.symbol,
            reason: "batched point mismatch",
        });
    }
    let expected =
        expected_batched_output_claim(program, driver, &*store, &proof.evals, &output.point, &batching_coeffs, ram)?;
    if output.value != expected {
        return Err(VerifyStage2Error::InvalidProof {
            driver: driver.symbol,
            reason: "batched output claim mismatch",
        });
    }
    let verified = Stage2SumcheckOutput {
        driver: driver.symbol,
        point: output.point,
        evals: proof.evals.clone(),
        proof: proof.proof.clone(),
    };
    store.observe_sumcheck_output(program, &verified)?;
    super::common::append_opening_claims(
        program.opening_inputs,
        program.opening_claims,
        program.opening_batches,
        &mut store.0,
        transcript,
        &verified.evals,
        |batch, claim| VerifyStage2Error::MissingClaim { batch, claim },
        |symbol| VerifyStage2Error::MissingValue { symbol },
    )?;
    Ok(verified)
}

impl<F: Field> Stage2ValueStore<F> {
    fn with_opening_inputs(inputs: &[Stage2OpeningInputValue<F>]) -> Self {
        Self(super::common::ValueStore::with_opening_inputs(inputs))
    }

    fn seed_constants(&mut self, program: &'static Stage2VerifierProgramPlan) {
        self.0.seed_constants(program.field_constants);
    }

    fn observe_challenge_vector(
        &mut self,
        program: &'static Stage2VerifierProgramPlan,
        plan: &'static Stage2TranscriptSqueezePlan,
        values: &[F],
    ) -> Result<(), VerifyStage2Error> {
        self.0.observe_challenge_vector(plan, values, |input, expected, actual| {
            VerifyStage2Error::InvalidInputLength { input, expected, actual }
        })?;
        self.evaluate_available_points(program)?;
        self.evaluate_available_field_exprs(program)?;
        Ok(())
    }

    fn observe_sumcheck_output(
        &mut self,
        program: &'static Stage2VerifierProgramPlan,
        output: &Stage2SumcheckOutput<F>,
    ) -> Result<(), VerifyStage2Error> {
        self.0.observe_sumcheck_output(
            program.instance_results,
            program.evals,
            output,
            |instance, mut point| {
                match instance.point_order {
                    "as_is" => {}
                    "reverse" => point.reverse(),
                    _ => {
                        return Err(VerifyStage2Error::InvalidProof {
                            driver: output.driver,
                            reason: "unsupported point order",
                        });
                    }
                }
                Ok(point)
            },
            |input, expected, actual| VerifyStage2Error::InvalidInputLength {
                input,
                expected,
                actual,
            },
            |symbol| VerifyStage2Error::MissingValue { symbol },
        )?;
        self.evaluate_available_points(program)?;
        self.evaluate_available_field_exprs(program)?;
        Ok(())
    }

    fn claim_value(
        &mut self,
        program: &'static Stage2VerifierProgramPlan,
        claim: &Stage2SumcheckClaimPlan,
    ) -> Result<F, VerifyStage2Error> {
        self.evaluate_available_field_exprs(program)?;
        self.scalar(claim.claim_value)
    }

    fn batch_claim_values(
        &mut self,
        program: &'static Stage2VerifierProgramPlan,
        batch: &Stage2SumcheckBatchPlan,
    ) -> Result<Vec<F>, VerifyStage2Error> {
        super::common::symbol_list(batch.claim_operands)
            .map(|symbol| {
                let claim = find_plan(program.claims, symbol).ok_or(VerifyStage2Error::MissingClaim {
                    batch: batch.symbol,
                    claim: symbol,
                })?;
                self.claim_value(program, claim)
            })
            .collect()
    }

    fn evaluate_available_points(
        &mut self,
        program: &'static Stage2VerifierProgramPlan,
    ) -> Result<(), VerifyStage2Error> {
        self.0.evaluate_available_points(
            program.point_slices,
            program.point_concats,
            |input, expected, actual| VerifyStage2Error::InvalidInputLength {
                input,
                expected,
                actual,
            },
        )
    }

    fn evaluate_available_field_exprs(
        &mut self,
        program: &'static Stage2VerifierProgramPlan,
    ) -> Result<(), VerifyStage2Error> {
        self.0
            .evaluate_available_field_exprs(program.field_exprs, evaluate_stage2_field_expr)
    }

    fn scalar(&self, symbol: &'static str) -> Result<F, VerifyStage2Error> {
        self.0
            .scalar_or(symbol, |symbol| VerifyStage2Error::MissingValue { symbol })
    }

    fn point(&self, symbol: &'static str) -> Result<&[F], VerifyStage2Error> {
        self.0
            .point_or(symbol, |symbol| VerifyStage2Error::MissingValue { symbol })
    }

    fn try_point(&self, symbol: &str) -> Option<&[F]> {
        self.0.try_point(symbol)
    }
}

fn evaluate_stage2_field_expr<F: Field>(
    expr: &Stage2FieldExprPlan,
    operands: &[F],
) -> Result<F, VerifyStage2Error> {
    match expr.formula {
        "opening_eval" => Ok(single_operand(expr.symbol, operands)?),
        "jolt_stage2_product_virtual_uniskip_input" => {
            require_operand_count(expr.symbol, 4, operands.len())?;
            let weights = lagrange_evals(
                PRODUCT_VIRTUAL_UNISKIP_DOMAIN_START,
                PRODUCT_VIRTUAL_UNISKIP_DOMAIN_SIZE,
                operands[0],
            );
            Ok(weights[0] * operands[1] + weights[1] * operands[2] + weights[2] * operands[3])
        }
        "jolt_stage2_ram_read_write_input" => {
            require_operand_count(expr.symbol, 3, operands.len())?;
            Ok(operands[1] + operands[0] * operands[2])
        }
        "jolt_stage2_instruction_lookup_input" => {
            require_operand_count(expr.symbol, 6, operands.len())?;
            let gamma = operands[0];
            let gamma2 = gamma.square();
            let gamma3 = gamma2 * gamma;
            let gamma4 = gamma2.square();
            Ok(operands[1]
                + gamma * operands[2]
                + gamma2 * operands[3]
                + gamma3 * operands[4]
                + gamma4 * operands[5])
        }
        "field.add" => {
            require_operand_count(expr.symbol, 2, operands.len())?;
            Ok(operands[0] + operands[1])
        }
        "field.sub" => {
            require_operand_count(expr.symbol, 2, operands.len())?;
            Ok(operands[0] - operands[1])
        }
        "field.mul" => {
            require_operand_count(expr.symbol, 2, operands.len())?;
            Ok(operands[0] * operands[1])
        }
        "field.neg" => {
            require_operand_count(expr.symbol, 1, operands.len())?;
            Ok(-operands[0])
        }
        formula => {
            if let Some(exponent) = formula.strip_prefix("field.pow:") {
                require_operand_count(expr.symbol, 1, operands.len())?;
                let exponent = exponent.parse::<usize>().map_err(|_| {
                    VerifyStage2Error::UnsupportedFieldExpr {
                        symbol: expr.symbol,
                        formula,
                    }
                })?;
                return Ok(pow_field(operands[0], exponent));
            }
            if let Some(spec) = formula.strip_prefix("poly.lagrange_basis_eval:") {
                require_operand_count(expr.symbol, 1, operands.len())?;
                let parts = spec.split(':').collect::<Vec<_>>();
                if parts.len() != 3 {
                    return Err(VerifyStage2Error::UnsupportedFieldExpr {
                        symbol: expr.symbol,
                        formula,
                    });
                }
                let domain_start = parts[0].parse::<i64>().map_err(|_| {
                    VerifyStage2Error::UnsupportedFieldExpr {
                        symbol: expr.symbol,
                        formula,
                    }
                })?;
                let domain_size = parts[1].parse::<usize>().map_err(|_| {
                    VerifyStage2Error::UnsupportedFieldExpr {
                        symbol: expr.symbol,
                        formula,
                    }
                })?;
                let index = parts[2].parse::<usize>().map_err(|_| {
                    VerifyStage2Error::UnsupportedFieldExpr {
                        symbol: expr.symbol,
                        formula,
                    }
                })?;
                let weights = lagrange_evals(domain_start, domain_size, operands[0]);
                return weights
                    .get(index)
                    .copied()
                    .ok_or(VerifyStage2Error::InvalidInputLength {
                        input: expr.symbol,
                        expected: index + 1,
                        actual: weights.len(),
                    });
            }
            Err(VerifyStage2Error::UnsupportedFieldExpr {
                symbol: expr.symbol,
                formula,
            })
        }
    }
}

fn expected_batched_output_claim(
    program: &'static Stage2VerifierProgramPlan,
    driver: &'static Stage2SumcheckDriverPlan,
    store: &Stage2ValueStore<Fr>,
    evals: &[Stage2NamedEval<Fr>],
    point: &[Fr],
    batching_coeffs: &[Fr],
    ram: Option<&Stage2RamData<'_>>,
) -> Result<Fr, VerifyStage2Error> {
    let batch = find_batch(program.batches, driver.symbol, driver.batch)?;
    let claims = batch_claims(program.claims, batch)?;
    let mut expected = Fr::from_u64(0);
    for (claim, coefficient) in claims.iter().zip(batching_coeffs) {
        let instance = program
            .instance_results
            .iter()
            .find(|instance| instance.claim == claim.symbol && instance.source == driver.symbol)
            .ok_or(VerifyStage2Error::MissingClaim {
                batch: batch.symbol,
                claim: claim.symbol,
            })?;
        let local_point = point
            .get(instance.round_offset..instance.round_offset + instance.num_rounds)
            .ok_or(VerifyStage2Error::InvalidInputLength {
                input: instance.symbol,
                expected: instance.round_offset + instance.num_rounds,
                actual: point.len(),
            })?;
        let value = match instance.relation {
            "jolt.stage2.ram.read_write" => expected_ram_read_write(store, evals, local_point)?,
            "jolt.stage2.product_virtual.remainder" => {
                expected_product_remainder(store, evals, local_point)?
            }
            "jolt.stage2.instruction_lookup.claim_reduction" => {
                expected_instruction_lookup(store, evals, local_point)?
            }
            "jolt.stage2.ram.raf_evaluation" => expected_ram_raf(evals, local_point, ram)?,
            "jolt.stage2.ram.output_check" => expected_ram_output(store, evals, local_point, ram)?,
            relation => return Err(VerifyStage2Error::UnsupportedRelation { relation }),
        };
        expected += *coefficient * value;
    }
    Ok(expected)
}

fn expected_ram_read_write(
    store: &Stage2ValueStore<Fr>,
    evals: &[Stage2NamedEval<Fr>],
    local_point: &[Fr],
) -> Result<Fr, VerifyStage2Error> {
    let r_cycle_stage1 = store.point("stage2.input.stage1.RamReadValue")?;
    let log_t = r_cycle_stage1.len();
    let r_cycle = reverse_slice(&local_point[..log_t]);
    let eq_eval = EqPolynomial::<Fr>::mle(r_cycle_stage1, &r_cycle);
    let gamma = store.scalar("stage2.ram_read_write.gamma")?;
    let val = eval_by_name(evals, "stage2.ram_read_write.eval.RamVal")?;
    let ra = eval_by_name(evals, "stage2.ram_read_write.eval.RamRa")?;
    let inc = eval_by_name(evals, "stage2.ram_read_write.eval.RamInc")?;
    Ok(eq_eval * ra * (val + gamma * (val + inc)))
}

fn expected_product_remainder(
    store: &Stage2ValueStore<Fr>,
    evals: &[Stage2NamedEval<Fr>],
    local_point: &[Fr],
) -> Result<Fr, VerifyStage2Error> {
    let tau_low = store.point("stage2.input.stage1.Product")?;
    let tau_high = store.scalar("stage2.product_virtual.tau_high")?;
    let r0 = *store
        .point("stage2.product_virtual.uniskip.sumcheck")?
        .first()
        .ok_or(VerifyStage2Error::MissingValue {
            symbol: "stage2.product_virtual.uniskip.sumcheck",
        })?;
    let r_tail = reverse_slice(local_point);
    let low = EqPolynomial::<Fr>::mle(tau_low, &r_tail);
    let high = lagrange_kernel_eval(
        PRODUCT_VIRTUAL_UNISKIP_DOMAIN_START,
        PRODUCT_VIRTUAL_UNISKIP_DOMAIN_SIZE,
        tau_high,
        r0,
    );
    let weights = lagrange_evals(
        PRODUCT_VIRTUAL_UNISKIP_DOMAIN_START,
        PRODUCT_VIRTUAL_UNISKIP_DOMAIN_SIZE,
        r0,
    );
    let left = weights[0]
        * eval_by_name(evals, "stage2.product_virtual.remainder.eval.LeftInstructionInput")?
        + weights[1] * eval_by_name(evals, "stage2.product_virtual.remainder.eval.LookupOutput")?
        + weights[2] * eval_by_name(evals, "stage2.product_virtual.remainder.eval.OpFlagJump")?;
    let right = weights[0]
        * eval_by_name(evals, "stage2.product_virtual.remainder.eval.RightInstructionInput")?
        + weights[1]
            * eval_by_name(evals, "stage2.product_virtual.remainder.eval.InstructionFlagBranch")?
        + weights[2]
            * (Fr::from_u64(1)
                - eval_by_name(evals, "stage2.product_virtual.remainder.eval.NextIsNoop")?);
    Ok(high * low * left * right)
}

fn expected_instruction_lookup(
    store: &Stage2ValueStore<Fr>,
    evals: &[Stage2NamedEval<Fr>],
    local_point: &[Fr],
) -> Result<Fr, VerifyStage2Error> {
    let opening_point = reverse_slice(local_point);
    let r_spartan = store.point("stage2.input.stage1.LookupOutput")?;
    let eq_eval = EqPolynomial::<Fr>::mle(&opening_point, r_spartan);
    let gamma = store.scalar("stage2.instruction_lookup.gamma")?;
    let gamma2 = gamma.square();
    let gamma3 = gamma2 * gamma;
    let gamma4 = gamma2.square();
    let weighted = eval_by_name(
        evals,
        "stage2.instruction_lookup.claim_reduction.eval.LookupOutput",
    )? + gamma
        * eval_by_name(
            evals,
            "stage2.instruction_lookup.claim_reduction.eval.LeftLookupOperand",
        )?
        + gamma2
            * eval_by_name(
                evals,
                "stage2.instruction_lookup.claim_reduction.eval.RightLookupOperand",
            )?
        + gamma3
            * eval_by_name(
                evals,
                "stage2.instruction_lookup.claim_reduction.eval.LeftInstructionInput",
            )?
        + gamma4
            * eval_by_name(
                evals,
                "stage2.instruction_lookup.claim_reduction.eval.RightInstructionInput",
            )?;
    Ok(eq_eval * weighted)
}

fn expected_ram_raf(
    evals: &[Stage2NamedEval<Fr>],
    local_point: &[Fr],
    ram: Option<&Stage2RamData<'_>>,
) -> Result<Fr, VerifyStage2Error> {
    let ram = ram.ok_or(VerifyStage2Error::MissingRam {
        relation: "jolt.stage2.ram.raf_evaluation",
    })?;
    let address = reverse_slice(local_point);
    let unmap = unmap_eval(ram.log_k, ram.start_address, &address);
    Ok(unmap * eval_by_name(evals, "stage2.ram_raf.eval.RamRa")?)
}

fn expected_ram_output(
    store: &Stage2ValueStore<Fr>,
    evals: &[Stage2NamedEval<Fr>],
    local_point: &[Fr],
    ram: Option<&Stage2RamData<'_>>,
) -> Result<Fr, VerifyStage2Error> {
    let ram = ram.ok_or(VerifyStage2Error::MissingRam {
        relation: "jolt.stage2.ram.output_check",
    })?;
    let layout = ram.output_layout.ok_or(VerifyStage2Error::MissingRam {
        relation: "jolt.stage2.ram.output_check.layout",
    })?;
    let r_address = store.point("stage2.ram_output.r_address")?;
    let opening_point = reverse_slice(local_point);
    let eq_eval = EqPolynomial::<Fr>::mle(r_address, &opening_point);
    let io_mask = range_mask_eval(layout.io_start, layout.io_end, &opening_point);
    let val_io = sparse_final_ram_eval(
        ram.final_ram,
        layout.io_start,
        layout.io_end,
        &opening_point,
    );
    let val_final = eval_by_name(evals, "stage2.ram_output.eval.RamValFinal")?;
    Ok(eq_eval * io_mask * (val_final - val_io))
}

fn driver_evals(
    program: &'static Stage2VerifierProgramPlan,
    driver: &'static str,
    value: Fr,
) -> Vec<Stage2NamedEval<Fr>> {
    program
        .evals
        .iter()
        .filter(|eval| eval.source == driver)
        .map(|eval| Stage2NamedEval {
            name: eval.name,
            oracle: eval.oracle,
            value,
        })
        .collect()
}

fn verify_named_evals(
    driver: &'static str,
    expected: &[Stage2NamedEval<Fr>],
    actual: &[Stage2NamedEval<Fr>],
) -> Result<(), VerifyStage2Error> {
    if expected.len() != actual.len() {
        return Err(VerifyStage2Error::InvalidProof {
            driver,
            reason: "eval count mismatch",
        });
    }
    for (expected, actual) in expected.iter().zip(actual) {
        if expected.name != actual.name || expected.oracle != actual.oracle || expected.value != actual.value {
            return Err(VerifyStage2Error::InvalidProof {
                driver,
                reason: "eval mismatch",
            });
        }
    }
    Ok(())
}

fn validate_driver_symbol(
    driver: &'static Stage2SumcheckDriverPlan,
    proof: &Stage2SumcheckOutput<Fr>,
) -> Result<(), VerifyStage2Error> {
    if proof.driver == driver.symbol {
        Ok(())
    } else {
        Err(VerifyStage2Error::InvalidProof {
            driver: driver.symbol,
            reason: "driver symbol mismatch",
        })
    }
}

fn append_univariate_poly<T>(transcript: &mut T, label: &'static str, poly: &UnivariatePoly<Fr>)
where
    T: Transcript<Challenge = Fr>,
{
    transcript.append(&LabelWithCount(
        label.as_bytes(),
        poly.coefficients().len() as u64,
    ));
    for coefficient in poly.coefficients() {
        transcript.append(coefficient);
    }
}

fn product_uniskip_sum_matches(poly: &UnivariatePoly<Fr>, claim: Fr) -> bool {
    (0..PRODUCT_VIRTUAL_UNISKIP_DOMAIN_SIZE)
        .map(|index| {
            poly.evaluate(Fr::from_i64(
                PRODUCT_VIRTUAL_UNISKIP_DOMAIN_START + index as i64,
            ))
        })
        .sum::<Fr>()
        == claim
}

fn polynomial_degree(poly: &UnivariatePoly<Fr>) -> usize {
    poly.coefficients()
        .iter()
        .rposition(|coefficient| *coefficient != Fr::from_u64(0))
        .unwrap_or(0)
}

fn unmap_eval(log_k: usize, start_address: u64, point: &[Fr]) -> Fr {
    point
        .iter()
        .enumerate()
        .fold(Fr::from_u64(start_address), |acc, (index, value)| {
            acc + value.mul_pow_2(log_k - 1 - index).mul_u64(8)
        })
}

fn range_mask_eval(start: usize, end: usize, point: &[Fr]) -> Fr {
    eq_prefix_sum(end, point) - eq_prefix_sum(start, point)
}

fn sparse_final_ram_eval(values: &[u64], start: usize, end: usize, point: &[Fr]) -> Fr {
    values[start..end]
        .iter()
        .enumerate()
        .filter(|(_, value)| **value != 0)
        .map(|(offset, value)| Fr::from_u64(*value) * eq_eval_at_index(start + offset, point))
        .sum()
}

fn eq_prefix_sum(end: usize, point: &[Fr]) -> Fr {
    let domain_len = 1usize << point.len();
    if end >= domain_len {
        return Fr::from_u64(1);
    }
    let mut sum = Fr::from_u64(0);
    let mut prefix = Fr::from_u64(1);
    for (bit, r) in point.iter().enumerate() {
        let mask = 1usize << (point.len() - 1 - bit);
        if end & mask == 0 {
            prefix *= Fr::from_u64(1) - *r;
        } else {
            sum += prefix * (Fr::from_u64(1) - *r);
            prefix *= *r;
        }
    }
    sum
}

fn eq_eval_at_index(index: usize, point: &[Fr]) -> Fr {
    point.iter().enumerate().fold(Fr::from_u64(1), |acc, (bit, r)| {
        let mask = 1usize << (point.len() - 1 - bit);
        if index & mask == 0 {
            acc * (Fr::from_u64(1) - *r)
        } else {
            acc * *r
        }
    })
}
"#
    }
}
