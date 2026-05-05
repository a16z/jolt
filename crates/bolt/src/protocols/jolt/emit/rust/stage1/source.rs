use super::Stage1CpuProgram;

impl Stage1CpuProgram {
    pub(super) fn emit_prover_entrypoint() -> &'static str {
        r"pub fn prove_stage1_outer<E, T>(
    executor: &mut E,
    transcript: &mut T,
) -> Result<Stage1ExecutionArtifacts<Fr>, Stage1KernelError>
where
    E: Stage1KernelExecutor<Fr>,
    T: Transcript<Challenge = Fr>,
{
    prove_stage1_outer_with_program(&STAGE1_PROGRAM, executor, transcript)
}

pub fn prove_stage1_outer_with_program<E, T>(
    program: &'static Stage1CpuProgramPlan,
    executor: &mut E,
    transcript: &mut T,
) -> Result<Stage1ExecutionArtifacts<Fr>, Stage1KernelError>
where
    E: Stage1KernelExecutor<Fr>,
    T: Transcript<Challenge = Fr>,
{
    execute_stage1_program(
        program,
        Stage1ExecutionMode::Prover,
        executor,
        transcript,
    )
}
"
    }

    pub(super) fn emit_verifier_entrypoint() -> &'static str {
        r#"pub fn verify_stage1_outer<T>(
    proof: &Stage1Proof<Fr>,
    transcript: &mut T,
) -> Result<Stage1ExecutionArtifacts<Fr>, VerifyStage1Error>
where
    T: Transcript<Challenge = Fr>,
{
    verify_stage1_outer_with_program(&STAGE1_PROGRAM, proof, transcript)
}

pub fn verify_stage1_outer_with_program<T>(
    program: &'static Stage1VerifierProgramPlan,
    proof: &Stage1Proof<Fr>,
    transcript: &mut T,
) -> Result<Stage1ExecutionArtifacts<Fr>, VerifyStage1Error>
where
    T: Transcript<Challenge = Fr>,
{
    if proof.sumchecks.len() != program.drivers.len() {
        return Err(VerifyStage1Error::UnexpectedProofCount {
            expected: program.drivers.len(),
            got: proof.sumchecks.len(),
        });
    }
    let mut artifacts = Stage1ExecutionArtifacts::default();
    for squeeze in program.transcript_squeezes {
        let values = transcript.challenge_vector(squeeze.count);
        artifacts.challenge_vectors.push(Stage1ChallengeVector {
            symbol: squeeze.symbol,
            values,
        });
    }
    for (index, driver) in program.drivers.iter().enumerate() {
        let proof = proof.sumchecks.get(index).ok_or(VerifyStage1Error::MissingProof {
            driver: driver.symbol,
        })?;
        let output = verify_stage1_driver(program, driver, proof, &artifacts.sumchecks, transcript)?;
        artifacts.sumchecks.push(output);
    }
    artifacts
        .opening_batches
        .extend(program.opening_batches.iter());
    Ok(artifacts)
}

pub fn stage1_outer_verifier_program() -> &'static Stage1VerifierProgramPlan {
    &STAGE1_PROGRAM
}

fn verify_stage1_driver<T>(
    program: &'static Stage1VerifierProgramPlan,
    driver: &'static Stage1SumcheckDriverPlan,
    proof: &Stage1SumcheckOutput<Fr>,
    completed: &[Stage1SumcheckOutput<Fr>],
    transcript: &mut T,
) -> Result<Stage1SumcheckOutput<Fr>, VerifyStage1Error>
where
    T: Transcript<Challenge = Fr>,
{
    if proof.driver != driver.symbol {
        return Err(VerifyStage1Error::InvalidProof {
            driver: driver.symbol,
            reason: "driver symbol mismatch",
        });
    }
    match driver.relation {
        "jolt.stage1.outer.uniskip" => verify_outer_uniskip(program, driver, proof, transcript),
        "jolt.stage1.outer.remaining" => {
            verify_outer_remaining(program, driver, proof, completed, transcript)
        }
        relation => Err(VerifyStage1Error::UnsupportedRelation { relation }),
    }
}

fn verify_outer_uniskip<T>(
    program: &'static Stage1VerifierProgramPlan,
    driver: &'static Stage1SumcheckDriverPlan,
    proof: &Stage1SumcheckOutput<Fr>,
    transcript: &mut T,
) -> Result<Stage1SumcheckOutput<Fr>, VerifyStage1Error>
where
    T: Transcript<Challenge = Fr>,
{
    let claim = SumcheckClaim::new(driver.num_rounds, driver.degree, Fr::from_u64(0));
    let round_proofs = proof
        .proof
        .round_polynomials
        .iter()
        .map(|poly| LabeledRoundPoly::new(poly, driver.round_label.as_bytes()))
        .collect::<Vec<_>>();
    let output = SumcheckVerifier::verify(&claim, &round_proofs, transcript)
        .map_err(|error| VerifyStage1Error::Sumcheck {
            driver: driver.symbol,
            error,
        })?;
    let eval = output.value;
    let point = output.point;
    if !proof.point.is_empty() && proof.point != point {
        return Err(VerifyStage1Error::InvalidProof {
            driver: driver.symbol,
            reason: "uniskip point mismatch",
        });
    }
    validate_eval_shape(program, driver, &proof.evals, Some(eval))?;
    append_labeled_scalar(transcript, "opening_claim", &eval);
    Ok(Stage1SumcheckOutput {
        driver: driver.symbol,
        point,
        evals: driver_evals(program, driver.symbol, eval),
        proof: proof.proof.clone(),
    })
}

fn verify_outer_remaining<T>(
    program: &'static Stage1VerifierProgramPlan,
    driver: &'static Stage1SumcheckDriverPlan,
    proof: &Stage1SumcheckOutput<Fr>,
    completed: &[Stage1SumcheckOutput<Fr>],
    transcript: &mut T,
) -> Result<Stage1SumcheckOutput<Fr>, VerifyStage1Error>
where
    T: Transcript<Challenge = Fr>,
{
    let input_claim = completed
        .iter()
        .find(|output| output.driver == "stage1.uniskip.sumcheck")
        .and_then(|output| output.evals.first())
        .map(|eval| eval.value)
        .ok_or(VerifyStage1Error::MissingDependency {
            driver: driver.symbol,
            dependency: "stage1.uniskip.eval",
        })?;
    append_labeled_scalar(transcript, driver.claim_label, &input_claim);
    let batching_coeff = transcript.challenge();
    let claim = SumcheckClaim::new(
        driver.num_rounds,
        driver.degree,
        input_claim * batching_coeff,
    );
    let round_proofs = proof
        .proof
        .round_polynomials
        .iter()
        .map(|poly| CompressedLabeledRoundPoly::new(poly, driver.round_label.as_bytes()))
        .collect::<Vec<_>>();
    let output = SumcheckVerifier::verify(&claim, &round_proofs, transcript)
        .map_err(|error| VerifyStage1Error::Sumcheck {
            driver: driver.symbol,
            error,
        })?;
    let point = output.point;
    if !proof.point.is_empty() && proof.point != point {
        return Err(VerifyStage1Error::InvalidProof {
            driver: driver.symbol,
            reason: "outer remaining point mismatch",
        });
    }
    validate_eval_shape(program, driver, &proof.evals, None)?;
    append_opening_claims(transcript, &proof.evals);
    Ok(Stage1SumcheckOutput {
        driver: driver.symbol,
        point,
        evals: proof.evals.clone(),
        proof: proof.proof.clone(),
    })
}

fn driver_evals(
    program: &'static Stage1VerifierProgramPlan,
    driver: &'static str,
    value: Fr,
) -> Vec<Stage1NamedEval<Fr>> {
    program
        .evals
        .iter()
        .filter(|eval| eval.source == driver)
        .map(|eval| Stage1NamedEval {
            name: eval.name,
            oracle: eval.oracle,
            value,
        })
        .collect()
}

fn validate_eval_shape(
    program: &'static Stage1VerifierProgramPlan,
    driver: &'static Stage1SumcheckDriverPlan,
    actual: &[Stage1NamedEval<Fr>],
    expected_value: Option<Fr>,
) -> Result<(), VerifyStage1Error> {
    let expected = program
        .evals
        .iter()
        .filter(|eval| eval.source == driver.symbol)
        .collect::<Vec<_>>();
    if actual.len() != expected.len() {
        return Err(VerifyStage1Error::InvalidProof {
            driver: driver.symbol,
            reason: "eval count mismatch",
        });
    }
    for (actual, expected) in actual.iter().zip(expected) {
        if actual.name != expected.name {
            return Err(VerifyStage1Error::InvalidProof {
                driver: driver.symbol,
                reason: "eval name mismatch",
            });
        }
        if actual.oracle != expected.oracle {
            return Err(VerifyStage1Error::InvalidProof {
                driver: driver.symbol,
                reason: "eval oracle mismatch",
            });
        }
        if expected_value.is_some_and(|value| actual.value != value) {
            return Err(VerifyStage1Error::InvalidProof {
                driver: driver.symbol,
                reason: "eval value mismatch",
            });
        }
    }
    Ok(())
}

fn append_opening_claims<T>(transcript: &mut T, evals: &[Stage1NamedEval<Fr>])
where
    T: Transcript<Challenge = Fr>,
{
    for eval in evals {
        append_labeled_scalar(transcript, "opening_claim", &eval.value);
    }
}
"#
    }
}
