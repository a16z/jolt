#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum TranscriptOp {
    State,
    Absorb,
    AbsorbBytes,
    ChallengeDriver,
    Ignored,
}

impl TranscriptOp {
    pub(super) fn classify(name: &str) -> Self {
        match name {
            "transcript.state" => Self::State,
            "transcript.absorb" | "transcript.absorb_optional" => Self::Absorb,
            "transcript.absorb_bytes" => Self::AbsorbBytes,
            "transcript.squeeze" | "piop.sumcheck" | "pcs.batch_open" | "pcs.batch_verify" => {
                Self::ChallengeDriver
            }
            _ => Self::Ignored,
        }
    }
}
