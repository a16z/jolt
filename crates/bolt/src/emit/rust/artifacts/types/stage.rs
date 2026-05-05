#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProtocolStageKind {
    Commitment,
    Proof,
    Evaluation,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProtocolStage {
    name: String,
    module_name: String,
    ordinal: usize,
    kind: ProtocolStageKind,
}

impl ProtocolStage {
    pub fn new(
        name: impl Into<String>,
        module_name: impl Into<String>,
        ordinal: usize,
        kind: ProtocolStageKind,
    ) -> Self {
        Self {
            name: name.into(),
            module_name: module_name.into(),
            ordinal,
            kind,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn module_name(&self) -> &str {
        &self.module_name
    }

    pub fn order(&self) -> usize {
        self.ordinal
    }

    pub fn is_commitment(&self) -> bool {
        self.kind == ProtocolStageKind::Commitment
    }

    pub fn is_proof(&self) -> bool {
        self.kind == ProtocolStageKind::Proof
    }

    pub fn is_evaluation(&self) -> bool {
        self.kind == ProtocolStageKind::Evaluation
    }
}
