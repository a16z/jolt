#![allow(dead_code)]

use jolt_dory::DoryCommitment;
use jolt_field::Fr;
use jolt_transcript::{AppendToTranscript, Blake2bTranscript, LabelWithCount, Transcript};

pub type DefaultCommitmentTranscript = Blake2bTranscript<Fr>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CommitmentParams {
    pub field: &'static str,
    pub pcs: &'static str,
    pub transcript: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OraclePlan {
    pub oracle: &'static str,
    pub domain: &'static str,
    pub num_vars: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CommitmentBatchPlan {
    pub artifact: &'static str,
    pub pcs: &'static str,
    pub oracle_family: &'static str,
    pub label: &'static str,
    pub oracles: &'static [&'static str],
    pub count: usize,
    pub domain: &'static str,
    pub num_vars: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OptionalSkipPolicy {
    MissingOrZero,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OptionalCommitmentPlan {
    pub artifact: &'static str,
    pub pcs: &'static str,
    pub oracle: &'static str,
    pub label: &'static str,
    pub domain: &'static str,
    pub num_vars: usize,
    pub skip_policy: OptionalSkipPolicy,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TranscriptStep {
    pub label: &'static str,
    pub source: &'static str,
    pub optional: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CommitmentVerifierProgramPlan {
    pub params: CommitmentParams,
    pub oracle_plans: &'static [OraclePlan],
    pub batch_plans: &'static [CommitmentBatchPlan],
    pub optional_plans: &'static [OptionalCommitmentPlan],
    pub transcript_steps: &'static [TranscriptStep],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CommitmentRecord {
    pub artifact: &'static str,
    pub oracle: &'static str,
    pub label: &'static str,
    pub num_vars: usize,
}

#[derive(Clone, Debug, Default)]
pub struct CommitmentArtifacts {
    pub commitments: Vec<Option<DoryCommitment>>,
    pub records: Vec<CommitmentRecord>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CommitmentPhaseError {
    MissingProofCommitment { oracle: &'static str },
    MissingProofCommitmentSlot { artifact: &'static str, oracle: &'static str },
    MissingTranscriptSource { source: &'static str },
    PlanCountMismatch { artifact: &'static str, expected: usize, actual: usize },
    ProofCommitmentCountMismatch { expected: usize, actual: usize },
}
pub const COMMITMENT_PARAMS: CommitmentParams = CommitmentParams {
    field: "bn254_fr",
    pcs: "dory",
    transcript: "blake2b_transcript",
};
pub const ORACLE_PLANS: &[OraclePlan] = &[
    OraclePlan { oracle: "RdInc", domain: "jolt.trace_domain", num_vars: 16 },
    OraclePlan { oracle: "RamInc", domain: "jolt.trace_domain", num_vars: 16 },
    OraclePlan { oracle: "InstructionRa_0", domain: "jolt.main_witness_commit_domain", num_vars: 20 },
    OraclePlan { oracle: "InstructionRa_1", domain: "jolt.main_witness_commit_domain", num_vars: 20 },
    OraclePlan { oracle: "InstructionRa_2", domain: "jolt.main_witness_commit_domain", num_vars: 20 },
    OraclePlan { oracle: "InstructionRa_3", domain: "jolt.main_witness_commit_domain", num_vars: 20 },
    OraclePlan { oracle: "InstructionRa_4", domain: "jolt.main_witness_commit_domain", num_vars: 20 },
    OraclePlan { oracle: "InstructionRa_5", domain: "jolt.main_witness_commit_domain", num_vars: 20 },
    OraclePlan { oracle: "InstructionRa_6", domain: "jolt.main_witness_commit_domain", num_vars: 20 },
    OraclePlan { oracle: "InstructionRa_7", domain: "jolt.main_witness_commit_domain", num_vars: 20 },
    OraclePlan { oracle: "InstructionRa_8", domain: "jolt.main_witness_commit_domain", num_vars: 20 },
    OraclePlan { oracle: "InstructionRa_9", domain: "jolt.main_witness_commit_domain", num_vars: 20 },
    OraclePlan { oracle: "InstructionRa_10", domain: "jolt.main_witness_commit_domain", num_vars: 20 },
    OraclePlan { oracle: "InstructionRa_11", domain: "jolt.main_witness_commit_domain", num_vars: 20 },
    OraclePlan { oracle: "InstructionRa_12", domain: "jolt.main_witness_commit_domain", num_vars: 20 },
    OraclePlan { oracle: "InstructionRa_13", domain: "jolt.main_witness_commit_domain", num_vars: 20 },
    OraclePlan { oracle: "InstructionRa_14", domain: "jolt.main_witness_commit_domain", num_vars: 20 },
    OraclePlan { oracle: "InstructionRa_15", domain: "jolt.main_witness_commit_domain", num_vars: 20 },
    OraclePlan { oracle: "InstructionRa_16", domain: "jolt.main_witness_commit_domain", num_vars: 20 },
    OraclePlan { oracle: "InstructionRa_17", domain: "jolt.main_witness_commit_domain", num_vars: 20 },
    OraclePlan { oracle: "InstructionRa_18", domain: "jolt.main_witness_commit_domain", num_vars: 20 },
    OraclePlan { oracle: "InstructionRa_19", domain: "jolt.main_witness_commit_domain", num_vars: 20 },
    OraclePlan { oracle: "InstructionRa_20", domain: "jolt.main_witness_commit_domain", num_vars: 20 },
    OraclePlan { oracle: "InstructionRa_21", domain: "jolt.main_witness_commit_domain", num_vars: 20 },
    OraclePlan { oracle: "InstructionRa_22", domain: "jolt.main_witness_commit_domain", num_vars: 20 },
    OraclePlan { oracle: "InstructionRa_23", domain: "jolt.main_witness_commit_domain", num_vars: 20 },
    OraclePlan { oracle: "InstructionRa_24", domain: "jolt.main_witness_commit_domain", num_vars: 20 },
    OraclePlan { oracle: "InstructionRa_25", domain: "jolt.main_witness_commit_domain", num_vars: 20 },
    OraclePlan { oracle: "InstructionRa_26", domain: "jolt.main_witness_commit_domain", num_vars: 20 },
    OraclePlan { oracle: "InstructionRa_27", domain: "jolt.main_witness_commit_domain", num_vars: 20 },
    OraclePlan { oracle: "InstructionRa_28", domain: "jolt.main_witness_commit_domain", num_vars: 20 },
    OraclePlan { oracle: "InstructionRa_29", domain: "jolt.main_witness_commit_domain", num_vars: 20 },
    OraclePlan { oracle: "InstructionRa_30", domain: "jolt.main_witness_commit_domain", num_vars: 20 },
    OraclePlan { oracle: "InstructionRa_31", domain: "jolt.main_witness_commit_domain", num_vars: 20 },
    OraclePlan { oracle: "RamRa_0", domain: "jolt.main_witness_commit_domain", num_vars: 20 },
    OraclePlan { oracle: "RamRa_1", domain: "jolt.main_witness_commit_domain", num_vars: 20 },
    OraclePlan { oracle: "RamRa_2", domain: "jolt.main_witness_commit_domain", num_vars: 20 },
    OraclePlan { oracle: "RamRa_3", domain: "jolt.main_witness_commit_domain", num_vars: 20 },
    OraclePlan { oracle: "BytecodeRa_0", domain: "jolt.main_witness_commit_domain", num_vars: 20 },
    OraclePlan { oracle: "BytecodeRa_1", domain: "jolt.main_witness_commit_domain", num_vars: 20 },
    OraclePlan { oracle: "BytecodeRa_2", domain: "jolt.main_witness_commit_domain", num_vars: 20 },
    OraclePlan { oracle: "UntrustedAdvice", domain: "jolt.trace_domain", num_vars: 16 },
    OraclePlan { oracle: "TrustedAdvice", domain: "jolt.trace_domain", num_vars: 16 },
];
pub const COMMITMENT_BATCH_0_ORACLES: &[&str] = &[
    "RdInc",
    "RamInc",
    "InstructionRa_0",
    "InstructionRa_1",
    "InstructionRa_2",
    "InstructionRa_3",
    "InstructionRa_4",
    "InstructionRa_5",
    "InstructionRa_6",
    "InstructionRa_7",
    "InstructionRa_8",
    "InstructionRa_9",
    "InstructionRa_10",
    "InstructionRa_11",
    "InstructionRa_12",
    "InstructionRa_13",
    "InstructionRa_14",
    "InstructionRa_15",
    "InstructionRa_16",
    "InstructionRa_17",
    "InstructionRa_18",
    "InstructionRa_19",
    "InstructionRa_20",
    "InstructionRa_21",
    "InstructionRa_22",
    "InstructionRa_23",
    "InstructionRa_24",
    "InstructionRa_25",
    "InstructionRa_26",
    "InstructionRa_27",
    "InstructionRa_28",
    "InstructionRa_29",
    "InstructionRa_30",
    "InstructionRa_31",
    "RamRa_0",
    "RamRa_1",
    "RamRa_2",
    "RamRa_3",
    "BytecodeRa_0",
    "BytecodeRa_1",
    "BytecodeRa_2",
];
pub const COMMITMENT_BATCH_PLANS: &[CommitmentBatchPlan] = &[
    CommitmentBatchPlan { artifact: "jolt.main_witness_commitments", pcs: "dory", oracle_family: "jolt.main_witness_polys", label: "commitment", oracles: COMMITMENT_BATCH_0_ORACLES, count: 41, domain: "jolt.main_witness_commit_domain", num_vars: 20 },
];
pub const OPTIONAL_COMMITMENT_PLANS: &[OptionalCommitmentPlan] = &[
    OptionalCommitmentPlan { artifact: "jolt.untrusted_advice_commitment", pcs: "dory", oracle: "UntrustedAdvice", label: "untrusted_advice", domain: "jolt.trace_domain", num_vars: 16, skip_policy: OptionalSkipPolicy::MissingOrZero },
    OptionalCommitmentPlan { artifact: "jolt.trusted_advice_commitment", pcs: "dory", oracle: "TrustedAdvice", label: "trusted_advice", domain: "jolt.trace_domain", num_vars: 16, skip_policy: OptionalSkipPolicy::MissingOrZero },
];
pub const TRANSCRIPT_PLAN: &[TranscriptStep] = &[
    TranscriptStep { label: "commitment", source: "jolt.main_witness_commitments", optional: false },
    TranscriptStep { label: "untrusted_advice", source: "jolt.untrusted_advice_commitment", optional: true },
    TranscriptStep { label: "trusted_advice", source: "jolt.trusted_advice_commitment", optional: true },
];
pub const COMMITMENT_PROGRAM: CommitmentVerifierProgramPlan = CommitmentVerifierProgramPlan {
    params: COMMITMENT_PARAMS,
    oracle_plans: ORACLE_PLANS,
    batch_plans: COMMITMENT_BATCH_PLANS,
    optional_plans: OPTIONAL_COMMITMENT_PLANS,
    transcript_steps: TRANSCRIPT_PLAN,
};

pub fn verify_commitment_phase<T>(
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
