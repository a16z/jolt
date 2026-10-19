use core::convert::Infallible;

use jolt_field::Field;
use jolt_transcript::TranscriptError;

/// A sumcheck failure: an invariant violation, a rejected proof, or a failure
/// `K` of the kernel or round context driving a single-instance sumcheck.
/// Retrying with the same input fails again. The batched engine and the
/// uni-skip provers keep `K = Infallible`.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum SumcheckError<F: Field, K = Infallible> {
    /// The proof's transcript rejected a round message.
    #[error("sumcheck transcript: {0}")]
    Transcript(#[from] TranscriptError),

    /// Round check failed: the domain sum did not match the expected value
    /// carried forward from the previous round.
    #[error("round {round}: expected sum {expected}, got {actual}")]
    RoundCheckFailed {
        /// Zero-indexed round number where the check failed.
        round: usize,
        expected: F,
        /// The computed domain sum.
        actual: F,
    },

    /// A round polynomial exceeded the declared degree bound.
    #[error("degree bound exceeded: degree {got}, max {max}")]
    DegreeBoundExceeded {
        /// Actual degree of the offending round polynomial.
        got: usize,
        /// Maximum allowed degree from the claim.
        max: usize,
    },

    /// A batch with rounds to prove declared a maximum degree of zero. A
    /// sumcheck round polynomial must have degree at least 1 — the same
    /// invariant `SumcheckClaim::new` enforces on the verify side.
    #[error("batch declares max degree 0 with {max_num_vars} rounds to prove")]
    ZeroBatchDegree {
        /// Number of rounds that would need round polynomials.
        max_num_vars: usize,
    },

    /// A batch member declared more rounds than the batch contains.
    #[error(
        "batch member {member} has {rounds} rounds, exceeding the batch's {max_num_vars} rounds"
    )]
    BatchMemberRoundsOutOfRange {
        /// Zero-indexed member position (declaration order).
        member: usize,
        /// Round count declared for this member.
        rounds: usize,
        /// The batch's total round count.
        max_num_vars: usize,
    },

    /// The field implementation cannot represent this batch's padding scale.
    #[error("batch member {member} requires unsupported padding exponent {exponent}")]
    BatchPaddingExponentOutOfRange {
        /// Zero-indexed member position (declaration order).
        member: usize,
        /// `max_num_vars - rounds`, which `Ring::mul_pow_2` cannot represent above 255.
        exponent: usize,
    },

    #[error(
        "batch member {member}: activation window overflow for offset {offset}, rounds {rounds}"
    )]
    BatchMemberWindowOverflow {
        /// Zero-indexed member position (declaration order).
        member: usize,
        /// The member's activation offset.
        offset: usize,
        /// The member's round count.
        rounds: usize,
    },

    /// Adding one to the declared degree overflowed `usize`.
    #[error("sumcheck degree {degree} cannot be represented with its coefficient count")]
    DegreeOverflow { degree: usize },

    /// The domain-sum coefficient vector must have exactly one scalar per
    /// round-polynomial coefficient.
    #[error("round {round}: expected {expected} round-sum coefficients, got {got}")]
    RoundSumCoefficientCountMismatch {
        /// Zero-indexed round number where the mismatch appeared.
        round: usize,
        /// Expected number of coefficients, equal to `degree + 1`.
        expected: usize,
        /// Actual number of coefficients supplied by the domain.
        got: usize,
    },

    /// An integer-domain sumcheck round used an invalid domain size.
    #[error("integer sumcheck domain size must be between 1 and i64::MAX, got {domain_size}")]
    InvalidIntegerDomain {
        /// Number of integer points in the domain.
        domain_size: usize,
    },

    #[error("round polynomial must contain at least one coefficient")]
    EmptyRoundCoefficients,

    /// A prove-side round-message evaluation could not resolve one of its
    /// leaves (an opening/derived table or a challenge scalar). Provers
    /// validate their sources at construction, so this indicates state
    /// corruption between construction and the round loop.
    #[error("missing {kind} evaluation source during prove")]
    MissingEvaluationSource {
        /// Which leaf kind failed to resolve: "opening", "challenge", or "derived".
        kind: &'static str,
    },

    #[error("vector-commitment setup has zero capacity")]
    ZeroCommitmentCapacity,

    #[error("round polynomial has {coefficients} coefficients, but vector-commitment capacity is {capacity}")]
    RoundExceedsCommitmentCapacity {
        /// Number of coefficients in the offending round polynomial.
        coefficients: usize,
        /// The vector-commitment setup's capacity.
        capacity: usize,
    },

    /// A batched prove was given a member slice whose length disagrees with
    /// the batch prelude.
    #[error("batch prelude describes {expected} members, got {got}")]
    BatchMemberCountMismatch {
        /// Member count in the batch prelude.
        expected: usize,
        /// Member count in the supplied slice.
        got: usize,
    },

    /// A batch member's round count disagrees with the batch prelude.
    #[error("batch member {member}: prelude says {expected} rounds, member reports {got}")]
    BatchMemberRoundsMismatch {
        /// Zero-indexed member position (declaration order).
        member: usize,
        /// Round count recorded in the batch prelude.
        expected: usize,
        /// Round count the member reports.
        got: usize,
    },

    #[error(
        "batch member {member}: window [{offset}, {offset} + {rounds}) exceeds {max_num_vars} rounds"
    )]
    BatchMemberWindowOutOfRange {
        /// Zero-indexed member position (declaration order).
        member: usize,
        /// The member's activation offset.
        offset: usize,
        /// The member's round count.
        rounds: usize,
        /// The batch's total round count.
        max_num_vars: usize,
    },

    /// The batch's round scheduler returned no round message for an active
    /// member. Reported rather than folded as that member's `claim / 2`
    /// padding (which would surface only as a round-sum mismatch).
    #[error("batch member {member}: round scheduler produced no round message")]
    MissingRoundMessage {
        /// Zero-indexed member position (declaration order).
        member: usize,
    },

    /// A round handle came back carrying a member index the batch does not
    /// have. The engine assigns those indices, so this means the scheduler
    /// rewrote one; attributing the message by it would misfold the round.
    #[error("round scheduler returned member index {member}, but the batch has {members} members")]
    RoundMemberIndexOutOfRange {
        /// The out-of-range index carried by the returned handle.
        member: usize,
        /// Number of members in the batch.
        members: usize,
    },

    /// A batched committed consistency was asked for an impossible point slice.
    #[error("batched point range overflow: offset {offset}, num_vars {num_vars}")]
    BatchedPointRangeOverflow {
        /// Starting index into the batched challenge vector.
        offset: usize,
        /// Number of variables in the requested instance.
        num_vars: usize,
    },

    /// A batched committed consistency did not contain enough challenges for the requested instance.
    #[error(
        "batched point out of range: offset {offset}, num_vars {num_vars}, total challenges {total}"
    )]
    BatchedPointOutOfRange {
        /// Starting index into the batched challenge vector.
        offset: usize,
        /// Number of variables in the requested instance.
        num_vars: usize,
        /// Total number of available batched challenges.
        total: usize,
    },

    /// Consecutive integer domain points have colliding images in the field.
    #[error("integer sumcheck domain of size {domain_size} is not distinct in the field")]
    IntegerDomainNotDistinct { domain_size: usize },

    /// A batch's round challenges were supplied with the wrong count.
    #[error("expected {expected} round challenges, got {got}")]
    WrongNumberOfChallenges {
        /// The batch's round count.
        expected: usize,
        /// Number of challenges supplied.
        got: usize,
    },

    /// [`SumcheckDomainSpec::EqFactored`](crate::SumcheckDomainSpec::EqFactored)
    /// names a family of round domains; the weights need the round's `τ`
    /// ([`EqFactoredRound`](crate::EqFactoredRound)).
    #[error("equality-factored round weights need the round's tau")]
    EqFactoredRoundWithoutTau,

    /// A single-instance sumcheck declared degree bound zero.
    #[error("sumcheck degree bound must be at least 1")]
    ZeroDegreeBound,

    /// The prover kernel's round count disagrees with the instance.
    #[error("sumcheck instance declares {expected} rounds, prover reports {got}")]
    RoundCountMismatch { expected: usize, got: usize },

    /// An equality-factored instance's equality point has the wrong length.
    #[error("equality point has {got} coordinates for {expected} rounds")]
    EqualityPointLengthMismatch { expected: usize, got: usize },

    /// A round index does not fit the `u32` of a transcript site.
    #[error("sumcheck round {round} does not fit a transcript site")]
    RoundIndexOverflow { round: usize },

    /// The kernel or the round context failed.
    #[error("sumcheck kernel: {0}")]
    Kernel(K),
}

impl<F: Field> SumcheckError<F, Infallible> {
    /// Embeds a kernel-free error into a driver whose kernel fails with `K`.
    pub fn widen<K>(self) -> SumcheckError<F, K> {
        match self {
            Self::Transcript(error) => SumcheckError::Transcript(error),
            Self::RoundCheckFailed {
                round,
                expected,
                actual,
            } => SumcheckError::RoundCheckFailed {
                round,
                expected,
                actual,
            },
            Self::DegreeBoundExceeded { got, max } => {
                SumcheckError::DegreeBoundExceeded { got, max }
            }
            Self::ZeroBatchDegree { max_num_vars } => {
                SumcheckError::ZeroBatchDegree { max_num_vars }
            }
            Self::BatchMemberRoundsOutOfRange {
                member,
                rounds,
                max_num_vars,
            } => SumcheckError::BatchMemberRoundsOutOfRange {
                member,
                rounds,
                max_num_vars,
            },
            Self::BatchPaddingExponentOutOfRange { member, exponent } => {
                SumcheckError::BatchPaddingExponentOutOfRange { member, exponent }
            }
            Self::BatchMemberWindowOverflow {
                member,
                offset,
                rounds,
            } => SumcheckError::BatchMemberWindowOverflow {
                member,
                offset,
                rounds,
            },
            Self::DegreeOverflow { degree } => SumcheckError::DegreeOverflow { degree },
            Self::RoundSumCoefficientCountMismatch {
                round,
                expected,
                got,
            } => SumcheckError::RoundSumCoefficientCountMismatch {
                round,
                expected,
                got,
            },
            Self::InvalidIntegerDomain { domain_size } => {
                SumcheckError::InvalidIntegerDomain { domain_size }
            }
            Self::EmptyRoundCoefficients => SumcheckError::EmptyRoundCoefficients,
            Self::MissingEvaluationSource { kind } => {
                SumcheckError::MissingEvaluationSource { kind }
            }
            Self::ZeroCommitmentCapacity => SumcheckError::ZeroCommitmentCapacity,
            Self::RoundExceedsCommitmentCapacity {
                coefficients,
                capacity,
            } => SumcheckError::RoundExceedsCommitmentCapacity {
                coefficients,
                capacity,
            },
            Self::BatchMemberCountMismatch { expected, got } => {
                SumcheckError::BatchMemberCountMismatch { expected, got }
            }
            Self::BatchMemberRoundsMismatch {
                member,
                expected,
                got,
            } => SumcheckError::BatchMemberRoundsMismatch {
                member,
                expected,
                got,
            },
            Self::BatchMemberWindowOutOfRange {
                member,
                offset,
                rounds,
                max_num_vars,
            } => SumcheckError::BatchMemberWindowOutOfRange {
                member,
                offset,
                rounds,
                max_num_vars,
            },
            Self::MissingRoundMessage { member } => SumcheckError::MissingRoundMessage { member },
            Self::RoundMemberIndexOutOfRange { member, members } => {
                SumcheckError::RoundMemberIndexOutOfRange { member, members }
            }
            Self::BatchedPointRangeOverflow { offset, num_vars } => {
                SumcheckError::BatchedPointRangeOverflow { offset, num_vars }
            }
            Self::BatchedPointOutOfRange {
                offset,
                num_vars,
                total,
            } => SumcheckError::BatchedPointOutOfRange {
                offset,
                num_vars,
                total,
            },
            Self::IntegerDomainNotDistinct { domain_size } => {
                SumcheckError::IntegerDomainNotDistinct { domain_size }
            }
            Self::WrongNumberOfChallenges { expected, got } => {
                SumcheckError::WrongNumberOfChallenges { expected, got }
            }
            Self::EqFactoredRoundWithoutTau => SumcheckError::EqFactoredRoundWithoutTau,
            Self::ZeroDegreeBound => SumcheckError::ZeroDegreeBound,
            Self::RoundCountMismatch { expected, got } => {
                SumcheckError::RoundCountMismatch { expected, got }
            }
            Self::EqualityPointLengthMismatch { expected, got } => {
                SumcheckError::EqualityPointLengthMismatch { expected, got }
            }
            Self::RoundIndexOverflow { round } => SumcheckError::RoundIndexOverflow { round },
            Self::Kernel(never) => match never {},
        }
    }
}
