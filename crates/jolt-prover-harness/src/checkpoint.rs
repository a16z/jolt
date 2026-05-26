use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NamedValue {
    pub name: String,
    pub value: String,
}

impl NamedValue {
    pub fn new(name: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            value: value.into(),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommitmentCheckpoint {
    pub commitments: Vec<NamedValue>,
    pub opening_hints: Vec<NamedValue>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StageCheckpoint {
    pub stage: u8,
    pub transcript_values: Vec<NamedValue>,
    pub output_claims: Vec<NamedValue>,
    pub opening_events: Vec<NamedValue>,
}

impl StageCheckpoint {
    pub fn new(stage: u8) -> Self {
        Self {
            stage,
            transcript_values: Vec::new(),
            output_claims: Vec::new(),
            opening_events: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpeningCheckpoint {
    pub opening_order: Vec<String>,
    pub evaluations: Vec<NamedValue>,
    pub joint_claims: Vec<NamedValue>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlindFoldCheckpoint {
    pub committed_rounds: Vec<NamedValue>,
    pub output_claim_rows: Vec<NamedValue>,
    pub public_inputs: Vec<NamedValue>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FrontierCheckpoint {
    Commitments(CommitmentCheckpoint),
    Stage(StageCheckpoint),
    Openings(OpeningCheckpoint),
    BlindFold(BlindFoldCheckpoint),
}

impl FrontierCheckpoint {
    pub fn named_values(&self) -> Vec<NamedValue> {
        match self {
            Self::Commitments(checkpoint) => checkpoint
                .commitments
                .iter()
                .chain(&checkpoint.opening_hints)
                .cloned()
                .collect(),
            Self::Stage(checkpoint) => checkpoint
                .transcript_values
                .iter()
                .chain(&checkpoint.output_claims)
                .chain(&checkpoint.opening_events)
                .cloned()
                .collect(),
            Self::Openings(checkpoint) => checkpoint
                .evaluations
                .iter()
                .chain(&checkpoint.joint_claims)
                .cloned()
                .collect(),
            Self::BlindFold(checkpoint) => checkpoint
                .committed_rounds
                .iter()
                .chain(&checkpoint.output_claim_rows)
                .chain(&checkpoint.public_inputs)
                .cloned()
                .collect(),
        }
    }
}
