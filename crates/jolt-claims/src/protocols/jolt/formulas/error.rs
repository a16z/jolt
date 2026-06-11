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
    #[error("incompatible dominant precommitted anchors: {first} and {second} disagree")]
    IncompatibleDominantAnchors { first: usize, second: usize },
    #[error("stage 6 cycle challenges ({got}) shorter than native cycle vars ({expected})")]
    CycleChallengesShorterThanNativeCycle { expected: usize, got: usize },
    #[error(
        "cycle-major final opening expects the stage 6 cycle prefix to equal the native cycle vars"
    )]
    CycleMajorCyclePrefixMismatch,
    #[error(
        "cycle-phase final opening requested with {active_address_rounds} active address-phase rounds remaining"
    )]
    CyclePhaseNotFinal { active_address_rounds: usize },
    #[error("cycle round {round} is not active for this polynomial")]
    InactiveCycleRound { round: usize },
    #[error(
        "polynomial dims ({poly_row_vars}x{poly_col_vars}) exceed reference dims ({reference_row_vars}x{reference_col_vars})"
    )]
    PolyDimsExceedReference {
        poly_row_vars: usize,
        poly_col_vars: usize,
        reference_row_vars: usize,
        reference_col_vars: usize,
    },
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
    #[error("{value_name} ({value}) must be divisible by {divisor_name} ({divisor})")]
    NotDivisible {
        value_name: &'static str,
        value: usize,
        divisor_name: &'static str,
        divisor: usize,
    },
    #[error("phase1_num_rounds ({phase1_num_rounds}) must be <= log_t ({log_t})")]
    InvalidPhaseRounds {
        phase1_num_rounds: usize,
        log_t: usize,
    },
    #[error(
        "bytecode chunk count ({chunk_count}) must be a nonzero power of two at most 256 dividing the power-of-two bytecode length ({bytecode_len})"
    )]
    InvalidBytecodeChunking {
        bytecode_len: usize,
        chunk_count: usize,
    },
}
