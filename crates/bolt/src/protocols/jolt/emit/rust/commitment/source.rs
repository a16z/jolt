use super::CommitmentCpuProgram;
use crate::ir::Role;

impl CommitmentCpuProgram {
    pub(super) fn emit_entrypoint(&self) -> &'static str {
        match self.role {
            Role::Prover => Self::emit_prover_entrypoint(),
            Role::Verifier => Self::emit_verifier_entrypoint(),
        }
    }

    fn emit_prover_entrypoint() -> &'static str {
        r"pub fn prove_commitment_phase<I, T>(
    inputs: &mut I,
    prover_setup: &DoryProverSetup,
    transcript: &mut T,
) -> Result<CommitmentArtifacts, CommitmentPhaseError>
where
    I: CommitmentInputProvider,
    T: Transcript<Challenge = Fr>,
{
    prove_commitment_phase_with_program(&COMMITMENT_PROGRAM, inputs, prover_setup, transcript)
}

pub fn prove_commitment_phase_with_program<I, T>(
    program: &'static CommitmentProverProgramPlan,
    inputs: &mut I,
    prover_setup: &DoryProverSetup,
    transcript: &mut T,
) -> Result<CommitmentArtifacts, CommitmentPhaseError>
where
    I: CommitmentInputProvider,
    T: Transcript<Challenge = Fr>,
{
    let mut artifacts = CommitmentArtifacts::default();
    for plan in program.batch_plans {
        commit_batch(program, inputs, prover_setup, &mut artifacts, plan)?;
    }
    for plan in program.optional_plans {
        commit_optional(program, inputs, prover_setup, &mut artifacts, plan)?;
    }
    absorb_transcript(program, &artifacts, transcript)?;
    Ok(artifacts)
}

fn commit_batch<I>(
    program: &CommitmentProverProgramPlan,
    inputs: &mut I,
    prover_setup: &DoryProverSetup,
    artifacts: &mut CommitmentArtifacts,
    plan: &CommitmentBatchPlan,
) -> Result<(), CommitmentPhaseError>
where
    I: CommitmentInputProvider,
{
    if plan.count != plan.oracles.len() {
        return Err(CommitmentPhaseError::PlanCountMismatch {
            artifact: plan.artifact,
            expected: plan.count,
            actual: plan.oracles.len(),
        });
    }
    if let Some(committed) = inputs.commit_batch(program, plan, prover_setup) {
        for committed in committed? {
            artifacts.records.push(committed.record);
            artifacts.commitments.push(committed.commitment);
            if let Some(hint) = committed.hint {
                artifacts.hints.push(hint);
            }
        }
        return Ok(());
    }
    for &oracle in plan.oracles {
        let data = inputs
            .materialize_with_num_vars(oracle, oracle_num_vars(program, oracle, plan.num_vars))
            .ok_or(CommitmentPhaseError::MissingOracle { oracle })?;
        let oracle_num_vars = oracle_num_vars(program, oracle, plan.num_vars);
        let data = into_padded_oracle(oracle, oracle_num_vars, data)?;
        let (commitment, hint) = commit_with_layout(&data, plan.num_vars, prover_setup)?;
        artifacts.records.push(CommitmentRecord {
            artifact: plan.artifact,
            oracle,
            label: plan.label,
            num_vars: oracle_num_vars,
        });
        artifacts.commitments.push(Some(commitment));
        artifacts.hints.push(OracleOpeningHint { oracle, hint });
    }
    Ok(())
}

fn commit_optional<I>(
    program: &CommitmentProverProgramPlan,
    inputs: &mut I,
    prover_setup: &DoryProverSetup,
    artifacts: &mut CommitmentArtifacts,
    plan: &OptionalCommitmentPlan,
) -> Result<(), CommitmentPhaseError>
where
    I: CommitmentInputProvider,
{
    let Some(data) = inputs.materialize_with_num_vars(plan.oracle, plan.num_vars) else {
        return push_skipped_optional(program, artifacts, plan);
    };
    if should_skip_optional(plan.skip_policy, data.as_ref()) {
        return push_skipped_optional(program, artifacts, plan);
    }
    let data = into_padded_oracle(plan.oracle, plan.num_vars, data)?;
    let (commitment, hint) = commit_with_layout(&data, plan.num_vars, prover_setup)?;
    artifacts.records.push(CommitmentRecord {
        artifact: plan.artifact,
        oracle: plan.oracle,
        label: plan.label,
        num_vars: oracle_num_vars(program, plan.oracle, plan.num_vars),
    });
    artifacts.commitments.push(Some(commitment));
    artifacts.hints.push(OracleOpeningHint {
        oracle: plan.oracle,
        hint,
    });
    Ok(())
}

fn push_skipped_optional(
    program: &CommitmentProverProgramPlan,
    artifacts: &mut CommitmentArtifacts,
    plan: &OptionalCommitmentPlan,
) -> Result<(), CommitmentPhaseError> {
    artifacts.records.push(CommitmentRecord {
        artifact: plan.artifact,
        oracle: plan.oracle,
        label: plan.label,
        num_vars: oracle_num_vars(program, plan.oracle, plan.num_vars),
    });
    artifacts.commitments.push(None);
    Ok(())
}

fn should_skip_optional(policy: OptionalSkipPolicy, data: &[Fr]) -> bool {
    match policy {
        OptionalSkipPolicy::MissingOrZero => data.iter().all(|value| *value == Fr::from_u64(0)),
    }
}

fn into_padded_oracle(
    oracle: &'static str,
    num_vars: usize,
    data: Cow<'_, [Fr]>,
) -> Result<Vec<Fr>, CommitmentPhaseError> {
    let target_len = target_len(num_vars)?;
    if data.len() > target_len {
        return Err(CommitmentPhaseError::OracleTooLarge {
            oracle,
            len: data.len(),
            target_len,
        });
    }
    let mut data = data.into_owned();
    data.resize(target_len, Fr::from_u64(0));
    Ok(data)
}

fn oracle_num_vars(
    program: &CommitmentProverProgramPlan,
    oracle: &'static str,
    fallback: usize,
) -> usize {
    program
        .oracle_plans
        .iter()
        .find(|plan| plan.oracle == oracle)
        .map_or(fallback, |plan| plan.num_vars)
}

fn commit_with_layout(
    data: &[Fr],
    layout_num_vars: usize,
    prover_setup: &DoryProverSetup,
) -> Result<(DoryCommitment, DoryHint), CommitmentPhaseError> {
    let row_len = target_len(layout_num_vars.div_ceil(2))?;
    Ok(DoryScheme::commit_evaluations_with_row_len(
        data,
        row_len,
        prover_setup,
    ))
}

fn target_len(num_vars: usize) -> Result<usize, CommitmentPhaseError> {
    if num_vars >= usize::BITS as usize {
        return Err(CommitmentPhaseError::TargetSizeOverflow { num_vars });
    }
    Ok(1usize << num_vars)
}

fn absorb_transcript<T>(
    program: &CommitmentProverProgramPlan,
    artifacts: &CommitmentArtifacts,
    transcript: &mut T,
) -> Result<(), CommitmentPhaseError>
where
    T: Transcript<Challenge = Fr>,
{
    for step in program.transcript_steps {
        let mut appended = false;
        for (record, commitment) in artifacts.records.iter().zip(&artifacts.commitments) {
            if record.artifact != step.source {
                continue;
            }
            if let Some(commitment) = commitment {
                transcript.append(&LabelWithCount(step.label.as_bytes(), commitment.serialized_len()));
                commitment.append_to_transcript(transcript);
                appended = true;
            }
        }
        if !step.optional && !appended {
            return Err(CommitmentPhaseError::MissingTranscriptSource {
                source: step.source,
            });
        }
    }
    Ok(())
}
"
    }

    fn emit_verifier_entrypoint() -> &'static str {
        r"pub fn verify_commitment_phase<T>(
    proof_commitments: &[Option<DoryCommitment>],
    transcript: &mut T,
) -> Result<CommitmentArtifacts, CommitmentPhaseError>
where
    T: Transcript<Challenge = Fr>,
{
    verify_commitment_phase_with_program(&COMMITMENT_PROGRAM, proof_commitments, transcript)
}

pub fn verify_commitment_phase_with_program<T>(
    program: &'static CommitmentVerifierProgramPlan,
    proof_commitments: &[Option<DoryCommitment>],
    transcript: &mut T,
) -> Result<CommitmentArtifacts, CommitmentPhaseError>
where
    T: Transcript<Challenge = Fr>,
{
    let mut artifacts = CommitmentArtifacts::default();
    let mut cursor = 0usize;
    for plan in program.batch_plans {
        receive_batch(program, proof_commitments, &mut cursor, &mut artifacts, plan)?;
    }
    for plan in program.optional_plans {
        receive_optional(program, proof_commitments, &mut cursor, &mut artifacts, plan)?;
    }
    if cursor != proof_commitments.len() {
        return Err(CommitmentPhaseError::ProofCommitmentCountMismatch {
            expected: cursor,
            actual: proof_commitments.len(),
        });
    }
    absorb_transcript(program, &artifacts, transcript)?;
    Ok(artifacts)
}

fn receive_batch(
    program: &'static CommitmentVerifierProgramPlan,
    proof_commitments: &[Option<DoryCommitment>],
    cursor: &mut usize,
    artifacts: &mut CommitmentArtifacts,
    plan: &CommitmentBatchPlan,
) -> Result<(), CommitmentPhaseError> {
    if plan.count != plan.oracles.len() {
        return Err(CommitmentPhaseError::PlanCountMismatch {
            artifact: plan.artifact,
            expected: plan.count,
            actual: plan.oracles.len(),
        });
    }
    for &oracle in plan.oracles {
        let commitment = proof_commitments
            .get(*cursor)
            .ok_or(CommitmentPhaseError::MissingProofCommitmentSlot {
                artifact: plan.artifact,
                oracle,
            })?
            .as_ref()
            .ok_or(CommitmentPhaseError::MissingProofCommitment { oracle })?
            .clone();
        *cursor += 1;
        let oracle_num_vars = oracle_num_vars(program, oracle, plan.num_vars);
        artifacts.records.push(CommitmentRecord {
            artifact: plan.artifact,
            oracle,
            label: plan.label,
            num_vars: oracle_num_vars,
        });
        artifacts.commitments.push(Some(commitment));
    }
    Ok(())
}

fn receive_optional(
    program: &'static CommitmentVerifierProgramPlan,
    proof_commitments: &[Option<DoryCommitment>],
    cursor: &mut usize,
    artifacts: &mut CommitmentArtifacts,
    plan: &OptionalCommitmentPlan,
) -> Result<(), CommitmentPhaseError> {
    let commitment = proof_commitments
        .get(*cursor)
        .ok_or(CommitmentPhaseError::MissingProofCommitmentSlot {
            artifact: plan.artifact,
            oracle: plan.oracle,
        })?
        .clone();
    *cursor += 1;
    artifacts.records.push(CommitmentRecord {
        artifact: plan.artifact,
        oracle: plan.oracle,
        label: plan.label,
        num_vars: oracle_num_vars(program, plan.oracle, plan.num_vars),
    });
    artifacts.commitments.push(commitment);
    Ok(())
}

pub fn commitment_verifier_program() -> &'static CommitmentVerifierProgramPlan {
    &COMMITMENT_PROGRAM
}

fn oracle_num_vars(
    program: &'static CommitmentVerifierProgramPlan,
    oracle: &'static str,
    fallback: usize,
) -> usize {
    program
        .oracle_plans
        .iter()
        .find(|plan| plan.oracle == oracle)
        .map_or(fallback, |plan| plan.num_vars)
}

fn absorb_transcript<T>(
    program: &'static CommitmentVerifierProgramPlan,
    artifacts: &CommitmentArtifacts,
    transcript: &mut T,
) -> Result<(), CommitmentPhaseError>
where
    T: Transcript<Challenge = Fr>,
{
    for step in program.transcript_steps {
        let mut appended = false;
        for (record, commitment) in artifacts.records.iter().zip(&artifacts.commitments) {
            if record.artifact != step.source {
                continue;
            }
            if let Some(commitment) = commitment {
                transcript.append(&LabelWithCount(step.label.as_bytes(), commitment.serialized_len()));
                commitment.append_to_transcript(transcript);
                appended = true;
            }
        }
        if !step.optional && !appended {
            return Err(CommitmentPhaseError::MissingTranscriptSource {
                source: step.source,
            });
        }
    }
    Ok(())
}
"
    }
}
