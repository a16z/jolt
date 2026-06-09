use thiserror::Error;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Error)]
pub enum JoltFormulaPointError {
    #[error(
        "invalid read-write phase split: phase1 {phase1_num_rounds}/{log_t}, phase2 {phase2_num_rounds}/{log_k}"
    )]
    InvalidReadWritePhaseSplit {
        phase1_num_rounds: usize,
        log_t: usize,
        phase2_num_rounds: usize,
        log_k: usize,
    },
    #[error("challenge length mismatch: expected {expected}, got {got}")]
    ChallengeLengthMismatch { expected: usize, got: usize },
    #[error("opening point length mismatch: expected {expected}, got {got}")]
    OpeningPointLengthMismatch { expected: usize, got: usize },
    #[error("evaluation domain length mismatch: expected {expected}, got {got}")]
    EvaluationDomainLengthMismatch { expected: usize, got: usize },
    #[error("evaluation domain size overflow for {num_vars} variables")]
    EvaluationDomainSizeOverflow { num_vars: usize },
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Error)]
pub enum JoltFormulaDimensionsError {
    #[error("{name} must be nonzero")]
    Zero { name: &'static str },
    #[error("{name} overflowed")]
    Overflow { name: &'static str },
    #[error(
        "lookup_virtual_chunk_bits ({lookup_virtual_chunk_bits}) must be >= committed_chunk_bits ({committed_chunk_bits})"
    )]
    InvalidChunkOrder {
        committed_chunk_bits: usize,
        lookup_virtual_chunk_bits: usize,
    },
    #[error("{name} underflowed")]
    Underflow { name: &'static str },
    #[error("{value_name} ({value}) must be divisible by {divisor_name} ({divisor})")]
    NotDivisible {
        value_name: &'static str,
        value: usize,
        divisor_name: &'static str,
        divisor: usize,
    },
    #[error("{value_name} ({value}) must be at most {max_name} ({max})")]
    Exceeds {
        value_name: &'static str,
        value: usize,
        max_name: &'static str,
        max: usize,
    },
    #[error("phase1_num_rounds ({phase1_num_rounds}) must be <= log_t ({log_t})")]
    InvalidPhaseRounds {
        phase1_num_rounds: usize,
        log_t: usize,
    },
}
