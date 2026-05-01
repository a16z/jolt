#![allow(dead_code)]

use std::borrow::Cow;

use jolt_dory::{DoryCommitment, DoryHint, DoryProverSetup, DoryScheme};
use jolt_field::{Field, Fr};
use jolt_transcript::{AppendToTranscript, Blake2bTranscript, LabelWithCount, Transcript};
use jolt_witness::{dense_i128_column_to_field, one_hot_chunk_address_major, optional_field_oracle};

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
pub struct CommitmentRecord {
    pub artifact: &'static str,
    pub oracle: &'static str,
    pub label: &'static str,
    pub num_vars: usize,
}

#[derive(Clone, Debug)]
pub struct OracleOpeningHint {
    pub oracle: &'static str,
    pub hint: DoryHint,
}

#[derive(Clone, Debug, Default)]
pub struct CommitmentArtifacts {
    pub commitments: Vec<Option<DoryCommitment>>,
    pub records: Vec<CommitmentRecord>,
    pub hints: Vec<OracleOpeningHint>,
}

pub trait CommitmentInputProvider {
    fn materialize(&mut self, oracle: &'static str) -> Option<Cow<'_, [Fr]>>;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CommitmentPhaseError {
    MissingOracle { oracle: &'static str },
    MissingTranscriptSource { source: &'static str },
    PlanCountMismatch { artifact: &'static str, expected: usize, actual: usize },
    OracleTooLarge { oracle: &'static str, len: usize, target_len: usize },
    TargetSizeOverflow { num_vars: usize },
}

pub struct CommitmentOracleInputs<'a> {
    pub rd_inc: &'a [i128],
    pub ram_inc: &'a [i128],
    pub instruction_keys: &'a [Option<u128>],
    pub ram_addresses: &'a [Option<u128>],
    pub bytecode_indices: &'a [Option<u128>],
    pub untrusted_advice: Option<&'a [Fr]>,
    pub trusted_advice: Option<&'a [Fr]>,
}

#[derive(Clone, Debug, Default)]
pub struct CommitmentOracles {
    pub rd_inc: Vec<Fr>,
    pub ram_inc: Vec<Fr>,
    pub instruction_ra_0: Vec<Fr>,
    pub instruction_ra_1: Vec<Fr>,
    pub instruction_ra_2: Vec<Fr>,
    pub instruction_ra_3: Vec<Fr>,
    pub instruction_ra_4: Vec<Fr>,
    pub instruction_ra_5: Vec<Fr>,
    pub instruction_ra_6: Vec<Fr>,
    pub instruction_ra_7: Vec<Fr>,
    pub instruction_ra_8: Vec<Fr>,
    pub instruction_ra_9: Vec<Fr>,
    pub instruction_ra_10: Vec<Fr>,
    pub instruction_ra_11: Vec<Fr>,
    pub instruction_ra_12: Vec<Fr>,
    pub instruction_ra_13: Vec<Fr>,
    pub instruction_ra_14: Vec<Fr>,
    pub instruction_ra_15: Vec<Fr>,
    pub instruction_ra_16: Vec<Fr>,
    pub instruction_ra_17: Vec<Fr>,
    pub instruction_ra_18: Vec<Fr>,
    pub instruction_ra_19: Vec<Fr>,
    pub instruction_ra_20: Vec<Fr>,
    pub instruction_ra_21: Vec<Fr>,
    pub instruction_ra_22: Vec<Fr>,
    pub instruction_ra_23: Vec<Fr>,
    pub instruction_ra_24: Vec<Fr>,
    pub instruction_ra_25: Vec<Fr>,
    pub instruction_ra_26: Vec<Fr>,
    pub instruction_ra_27: Vec<Fr>,
    pub instruction_ra_28: Vec<Fr>,
    pub instruction_ra_29: Vec<Fr>,
    pub instruction_ra_30: Vec<Fr>,
    pub instruction_ra_31: Vec<Fr>,
    pub ram_ra_0: Vec<Fr>,
    pub ram_ra_1: Vec<Fr>,
    pub ram_ra_2: Vec<Fr>,
    pub ram_ra_3: Vec<Fr>,
    pub bytecode_ra_0: Vec<Fr>,
    pub bytecode_ra_1: Vec<Fr>,
    pub bytecode_ra_2: Vec<Fr>,
    pub untrusted_advice: Option<Vec<Fr>>,
    pub trusted_advice: Option<Vec<Fr>>,
}

impl CommitmentInputProvider for CommitmentOracles {
    fn materialize(&mut self, oracle: &'static str) -> Option<Cow<'_, [Fr]>> {
        match oracle {
            "RdInc" => Some(Cow::Borrowed(&self.rd_inc)),
            "RamInc" => Some(Cow::Borrowed(&self.ram_inc)),
            "InstructionRa_0" => Some(Cow::Borrowed(&self.instruction_ra_0)),
            "InstructionRa_1" => Some(Cow::Borrowed(&self.instruction_ra_1)),
            "InstructionRa_2" => Some(Cow::Borrowed(&self.instruction_ra_2)),
            "InstructionRa_3" => Some(Cow::Borrowed(&self.instruction_ra_3)),
            "InstructionRa_4" => Some(Cow::Borrowed(&self.instruction_ra_4)),
            "InstructionRa_5" => Some(Cow::Borrowed(&self.instruction_ra_5)),
            "InstructionRa_6" => Some(Cow::Borrowed(&self.instruction_ra_6)),
            "InstructionRa_7" => Some(Cow::Borrowed(&self.instruction_ra_7)),
            "InstructionRa_8" => Some(Cow::Borrowed(&self.instruction_ra_8)),
            "InstructionRa_9" => Some(Cow::Borrowed(&self.instruction_ra_9)),
            "InstructionRa_10" => Some(Cow::Borrowed(&self.instruction_ra_10)),
            "InstructionRa_11" => Some(Cow::Borrowed(&self.instruction_ra_11)),
            "InstructionRa_12" => Some(Cow::Borrowed(&self.instruction_ra_12)),
            "InstructionRa_13" => Some(Cow::Borrowed(&self.instruction_ra_13)),
            "InstructionRa_14" => Some(Cow::Borrowed(&self.instruction_ra_14)),
            "InstructionRa_15" => Some(Cow::Borrowed(&self.instruction_ra_15)),
            "InstructionRa_16" => Some(Cow::Borrowed(&self.instruction_ra_16)),
            "InstructionRa_17" => Some(Cow::Borrowed(&self.instruction_ra_17)),
            "InstructionRa_18" => Some(Cow::Borrowed(&self.instruction_ra_18)),
            "InstructionRa_19" => Some(Cow::Borrowed(&self.instruction_ra_19)),
            "InstructionRa_20" => Some(Cow::Borrowed(&self.instruction_ra_20)),
            "InstructionRa_21" => Some(Cow::Borrowed(&self.instruction_ra_21)),
            "InstructionRa_22" => Some(Cow::Borrowed(&self.instruction_ra_22)),
            "InstructionRa_23" => Some(Cow::Borrowed(&self.instruction_ra_23)),
            "InstructionRa_24" => Some(Cow::Borrowed(&self.instruction_ra_24)),
            "InstructionRa_25" => Some(Cow::Borrowed(&self.instruction_ra_25)),
            "InstructionRa_26" => Some(Cow::Borrowed(&self.instruction_ra_26)),
            "InstructionRa_27" => Some(Cow::Borrowed(&self.instruction_ra_27)),
            "InstructionRa_28" => Some(Cow::Borrowed(&self.instruction_ra_28)),
            "InstructionRa_29" => Some(Cow::Borrowed(&self.instruction_ra_29)),
            "InstructionRa_30" => Some(Cow::Borrowed(&self.instruction_ra_30)),
            "InstructionRa_31" => Some(Cow::Borrowed(&self.instruction_ra_31)),
            "RamRa_0" => Some(Cow::Borrowed(&self.ram_ra_0)),
            "RamRa_1" => Some(Cow::Borrowed(&self.ram_ra_1)),
            "RamRa_2" => Some(Cow::Borrowed(&self.ram_ra_2)),
            "RamRa_3" => Some(Cow::Borrowed(&self.ram_ra_3)),
            "BytecodeRa_0" => Some(Cow::Borrowed(&self.bytecode_ra_0)),
            "BytecodeRa_1" => Some(Cow::Borrowed(&self.bytecode_ra_1)),
            "BytecodeRa_2" => Some(Cow::Borrowed(&self.bytecode_ra_2)),
            "UntrustedAdvice" => self.untrusted_advice.as_deref().map(Cow::Borrowed),
            "TrustedAdvice" => self.trusted_advice.as_deref().map(Cow::Borrowed),
            _ => None,
        }
    }
}

pub fn build_commitment_oracles(
    inputs: &CommitmentOracleInputs<'_>,
) -> Result<CommitmentOracles, CommitmentPhaseError> {
    Ok(CommitmentOracles {
        rd_inc: dense_i128_column_to_field(inputs.rd_inc, target_len(16)?),
        ram_inc: dense_i128_column_to_field(inputs.ram_inc, target_len(16)?),
        instruction_ra_0: one_hot_chunk_address_major(inputs.instruction_keys, 0, 32, 4, target_len(16)?, Some(0)),
        instruction_ra_1: one_hot_chunk_address_major(inputs.instruction_keys, 1, 32, 4, target_len(16)?, Some(0)),
        instruction_ra_2: one_hot_chunk_address_major(inputs.instruction_keys, 2, 32, 4, target_len(16)?, Some(0)),
        instruction_ra_3: one_hot_chunk_address_major(inputs.instruction_keys, 3, 32, 4, target_len(16)?, Some(0)),
        instruction_ra_4: one_hot_chunk_address_major(inputs.instruction_keys, 4, 32, 4, target_len(16)?, Some(0)),
        instruction_ra_5: one_hot_chunk_address_major(inputs.instruction_keys, 5, 32, 4, target_len(16)?, Some(0)),
        instruction_ra_6: one_hot_chunk_address_major(inputs.instruction_keys, 6, 32, 4, target_len(16)?, Some(0)),
        instruction_ra_7: one_hot_chunk_address_major(inputs.instruction_keys, 7, 32, 4, target_len(16)?, Some(0)),
        instruction_ra_8: one_hot_chunk_address_major(inputs.instruction_keys, 8, 32, 4, target_len(16)?, Some(0)),
        instruction_ra_9: one_hot_chunk_address_major(inputs.instruction_keys, 9, 32, 4, target_len(16)?, Some(0)),
        instruction_ra_10: one_hot_chunk_address_major(inputs.instruction_keys, 10, 32, 4, target_len(16)?, Some(0)),
        instruction_ra_11: one_hot_chunk_address_major(inputs.instruction_keys, 11, 32, 4, target_len(16)?, Some(0)),
        instruction_ra_12: one_hot_chunk_address_major(inputs.instruction_keys, 12, 32, 4, target_len(16)?, Some(0)),
        instruction_ra_13: one_hot_chunk_address_major(inputs.instruction_keys, 13, 32, 4, target_len(16)?, Some(0)),
        instruction_ra_14: one_hot_chunk_address_major(inputs.instruction_keys, 14, 32, 4, target_len(16)?, Some(0)),
        instruction_ra_15: one_hot_chunk_address_major(inputs.instruction_keys, 15, 32, 4, target_len(16)?, Some(0)),
        instruction_ra_16: one_hot_chunk_address_major(inputs.instruction_keys, 16, 32, 4, target_len(16)?, Some(0)),
        instruction_ra_17: one_hot_chunk_address_major(inputs.instruction_keys, 17, 32, 4, target_len(16)?, Some(0)),
        instruction_ra_18: one_hot_chunk_address_major(inputs.instruction_keys, 18, 32, 4, target_len(16)?, Some(0)),
        instruction_ra_19: one_hot_chunk_address_major(inputs.instruction_keys, 19, 32, 4, target_len(16)?, Some(0)),
        instruction_ra_20: one_hot_chunk_address_major(inputs.instruction_keys, 20, 32, 4, target_len(16)?, Some(0)),
        instruction_ra_21: one_hot_chunk_address_major(inputs.instruction_keys, 21, 32, 4, target_len(16)?, Some(0)),
        instruction_ra_22: one_hot_chunk_address_major(inputs.instruction_keys, 22, 32, 4, target_len(16)?, Some(0)),
        instruction_ra_23: one_hot_chunk_address_major(inputs.instruction_keys, 23, 32, 4, target_len(16)?, Some(0)),
        instruction_ra_24: one_hot_chunk_address_major(inputs.instruction_keys, 24, 32, 4, target_len(16)?, Some(0)),
        instruction_ra_25: one_hot_chunk_address_major(inputs.instruction_keys, 25, 32, 4, target_len(16)?, Some(0)),
        instruction_ra_26: one_hot_chunk_address_major(inputs.instruction_keys, 26, 32, 4, target_len(16)?, Some(0)),
        instruction_ra_27: one_hot_chunk_address_major(inputs.instruction_keys, 27, 32, 4, target_len(16)?, Some(0)),
        instruction_ra_28: one_hot_chunk_address_major(inputs.instruction_keys, 28, 32, 4, target_len(16)?, Some(0)),
        instruction_ra_29: one_hot_chunk_address_major(inputs.instruction_keys, 29, 32, 4, target_len(16)?, Some(0)),
        instruction_ra_30: one_hot_chunk_address_major(inputs.instruction_keys, 30, 32, 4, target_len(16)?, Some(0)),
        instruction_ra_31: one_hot_chunk_address_major(inputs.instruction_keys, 31, 32, 4, target_len(16)?, Some(0)),
        ram_ra_0: one_hot_chunk_address_major(inputs.ram_addresses, 0, 4, 4, target_len(16)?, None),
        ram_ra_1: one_hot_chunk_address_major(inputs.ram_addresses, 1, 4, 4, target_len(16)?, None),
        ram_ra_2: one_hot_chunk_address_major(inputs.ram_addresses, 2, 4, 4, target_len(16)?, None),
        ram_ra_3: one_hot_chunk_address_major(inputs.ram_addresses, 3, 4, 4, target_len(16)?, None),
        bytecode_ra_0: one_hot_chunk_address_major(inputs.bytecode_indices, 0, 3, 4, target_len(16)?, Some(0)),
        bytecode_ra_1: one_hot_chunk_address_major(inputs.bytecode_indices, 1, 3, 4, target_len(16)?, Some(0)),
        bytecode_ra_2: one_hot_chunk_address_major(inputs.bytecode_indices, 2, 3, 4, target_len(16)?, Some(0)),
        untrusted_advice: optional_field_oracle(inputs.untrusted_advice, target_len(16)?),
        trusted_advice: optional_field_oracle(inputs.trusted_advice, target_len(16)?),
    })
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

pub fn prove_commitment_phase<I, T>(
    inputs: &mut I,
    prover_setup: &DoryProverSetup,
    transcript: &mut T,
) -> Result<CommitmentArtifacts, CommitmentPhaseError>
where
    I: CommitmentInputProvider,
    T: Transcript<Challenge = Fr>,
{
    let mut artifacts = CommitmentArtifacts::default();
    for plan in COMMITMENT_BATCH_PLANS {
        commit_batch(inputs, prover_setup, &mut artifacts, plan)?;
    }
    for plan in OPTIONAL_COMMITMENT_PLANS {
        commit_optional(inputs, prover_setup, &mut artifacts, plan)?;
    }
    absorb_transcript(&artifacts, transcript)?;
    Ok(artifacts)
}

fn commit_batch<I>(
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
    for &oracle in plan.oracles {
        let data = inputs
            .materialize(oracle)
            .ok_or(CommitmentPhaseError::MissingOracle { oracle })?;
        let oracle_num_vars = oracle_num_vars(oracle, plan.num_vars);
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
    inputs: &mut I,
    prover_setup: &DoryProverSetup,
    artifacts: &mut CommitmentArtifacts,
    plan: &OptionalCommitmentPlan,
) -> Result<(), CommitmentPhaseError>
where
    I: CommitmentInputProvider,
{
    let Some(data) = inputs.materialize(plan.oracle) else {
        return push_skipped_optional(artifacts, plan);
    };
    if should_skip_optional(plan.skip_policy, data.as_ref()) {
        return push_skipped_optional(artifacts, plan);
    }
    let data = into_padded_oracle(plan.oracle, plan.num_vars, data)?;
    let (commitment, hint) = commit_with_layout(&data, plan.num_vars, prover_setup)?;
    artifacts.records.push(CommitmentRecord {
        artifact: plan.artifact,
        oracle: plan.oracle,
        label: plan.label,
        num_vars: oracle_num_vars(plan.oracle, plan.num_vars),
    });
    artifacts.commitments.push(Some(commitment));
    artifacts.hints.push(OracleOpeningHint {
        oracle: plan.oracle,
        hint,
    });
    Ok(())
}

fn push_skipped_optional(
    artifacts: &mut CommitmentArtifacts,
    plan: &OptionalCommitmentPlan,
) -> Result<(), CommitmentPhaseError> {
    artifacts.records.push(CommitmentRecord {
        artifact: plan.artifact,
        oracle: plan.oracle,
        label: plan.label,
        num_vars: oracle_num_vars(plan.oracle, plan.num_vars),
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

fn oracle_num_vars(oracle: &'static str, fallback: usize) -> usize {
    ORACLE_PLANS
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
    artifacts: &CommitmentArtifacts,
    transcript: &mut T,
) -> Result<(), CommitmentPhaseError>
where
    T: Transcript<Challenge = Fr>,
{
    for step in TRANSCRIPT_PLAN {
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
