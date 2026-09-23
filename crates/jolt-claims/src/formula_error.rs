//! Protocol-neutral point-geometry errors.
//!
//! [`JoltFormulaPointError`] ("Jolt" the zkVM, not the jolt protocol family) is
//! shared by every protocol family's geometry, so it lives in the framework
//! half of the crate: `protocols/jolt` re-exports it at its historical path,
//! and `protocols/field_inline` consumes it from here — neither protocol
//! module imports the other for it.

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
    #[error(
        "bytecode chunk count ({chunk_count}) must be a nonzero power of two at most 256 dividing the power-of-two bytecode length ({bytecode_len})"
    )]
    InvalidBytecodeChunking {
        bytecode_len: usize,
        chunk_count: usize,
    },
    #[error("evaluation domain size overflow for {num_vars} variables")]
    EvaluationDomainSizeOverflow { num_vars: usize },
}
