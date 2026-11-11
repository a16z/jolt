use crate::SumcheckKernelError;
use jolt_field::JoltField;
use jolt_sumcheck::SumcheckError;

#[derive(Clone, Copy)]
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
enum MessageState {
    Ready,
    PendingBind,
}

enum RoundRequest {
    Message { round: usize, has_bind: bool },
    Finish,
}

impl MessageState {
    fn validate<F: JoltField>(
        self,
        bound: usize,
        total: usize,
        request: RoundRequest,
    ) -> Result<(), SumcheckError<F>> {
        let pending = matches!(self, Self::PendingBind);
        let (round, valid) = match request {
            RoundRequest::Message { round, has_bind } => (
                round,
                round < total
                    && has_bind == pending
                    && bound.checked_add(usize::from(has_bind)) == Some(round),
            ),
            RoundRequest::Finish => (total, pending && bound.checked_add(1) == Some(total)),
        };
        if valid {
            Ok(())
        } else {
            Err(SumcheckError::InvalidEqRoundState {
                round,
                context: "round index, bind presence, or finish does not match the pending message",
            })
        }
    }
}

pub(crate) fn eq_round_error<F: JoltField>(round: usize) -> SumcheckError<F> {
    SumcheckError::InvalidEqRoundState {
        round,
        context: "equality state is exhausted",
    }
}

/// A kernel's bound-round count against its total — the one home of the
/// "claims only after every round is bound" invariant.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
pub(crate) struct RoundProgress {
    bound: usize,
    total: usize,
    message: MessageState,
}

impl RoundProgress {
    pub(crate) fn new(total: usize) -> Self {
        Self {
            bound: 0,
            total,
            message: MessageState::Ready,
        }
    }

    /// Total rounds — the kernel's `ProveRounds::num_rounds`.
    pub(crate) fn total(&self) -> usize {
        self.total
    }

    pub(crate) fn bound(&self) -> usize {
        self.bound
    }

    pub(crate) fn advance(&mut self) {
        self.bound += 1;
        self.message = MessageState::Ready;
    }

    pub(crate) fn validate_round<F: JoltField>(
        &self,
        round: usize,
        bind: bool,
    ) -> Result<(), SumcheckError<F>> {
        self.message.validate(
            self.bound(),
            self.total(),
            RoundRequest::Message {
                round,
                has_bind: bind,
            },
        )
    }

    pub(crate) fn validate_finish<F: JoltField>(&self) -> Result<(), SumcheckError<F>> {
        self.message
            .validate(self.bound(), self.total(), RoundRequest::Finish)
    }

    pub(crate) fn record_message(&mut self) {
        self.message = MessageState::PendingBind;
    }

    /// Gate for every output-claim / derived-table entry point.
    pub(crate) fn require_complete<F: JoltField>(&self) -> Result<(), SumcheckKernelError<F>> {
        if self.bound == self.total {
            Ok(())
        } else {
            Err(SumcheckKernelError::NotFullyBound {
                remaining: self.total - self.bound,
            })
        }
    }
}

/// The drawn challenges of a kernel's bound rounds, tracked against the
/// round total — one authority for both the challenge history and the
/// bound-rounds invariant. Kernels that never revisit their challenges use
/// [`RoundProgress`] instead.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
pub(crate) struct RoundChallenges<F> {
    challenges: Vec<F>,
    total: usize,
    message: MessageState,
}

impl<F: JoltField> RoundChallenges<F> {
    pub(crate) fn new(total: usize) -> Self {
        Self {
            challenges: Vec::with_capacity(total),
            total,
            message: MessageState::Ready,
        }
    }

    /// Total rounds — the kernel's `ProveRounds::num_rounds`.
    pub(crate) fn total(&self) -> usize {
        self.total
    }

    pub(crate) fn bound(&self) -> usize {
        self.challenges.len()
    }

    pub(crate) fn push(&mut self, challenge: F) {
        self.challenges.push(challenge);
        self.message = MessageState::Ready;
    }

    /// The challenges bound so far, in binding order.
    pub(crate) fn as_slice(&self) -> &[F] {
        &self.challenges
    }

    pub(crate) fn validate_round(&self, round: usize, bind: bool) -> Result<(), SumcheckError<F>> {
        self.message.validate(
            self.bound(),
            self.total(),
            RoundRequest::Message {
                round,
                has_bind: bind,
            },
        )
    }

    pub(crate) fn validate_finish(&self) -> Result<(), SumcheckError<F>> {
        self.message
            .validate(self.bound(), self.total(), RoundRequest::Finish)
    }

    pub(crate) fn record_message(&mut self) {
        self.message = MessageState::PendingBind;
    }

    /// Gate for every output-claim / derived-table entry point.
    pub(crate) fn require_complete(&self) -> Result<(), SumcheckKernelError<F>> {
        if self.challenges.len() == self.total {
            Ok(())
        } else {
            Err(SumcheckKernelError::NotFullyBound {
                remaining: self.total - self.challenges.len(),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{Fr, Ring};

    #[test]
    fn zero_and_one_round_schedules_reject_invalid_entries_without_advancing() {
        for total in [0, 1] {
            let mut progress = RoundProgress::new(total);
            let mut challenges = RoundChallenges::<Fr>::new(total);
            assert!(
                matches!(progress.validate_finish::<Fr>(), Err(SumcheckError::InvalidEqRoundState { round, .. }) if round == total)
            );
            assert!(
                matches!(challenges.validate_finish(), Err(SumcheckError::InvalidEqRoundState { round, .. }) if round == total)
            );
            assert!(progress.validate_round::<Fr>(0, true).is_err());
            assert!(challenges.validate_round(0, true).is_err());
            assert_eq!(progress.bound(), 0);
            assert_eq!(challenges.as_slice(), []);
            if total == 0 {
                assert!(progress.validate_round::<Fr>(0, false).is_err());
                assert!(challenges.validate_round(0, false).is_err());
                assert!(progress.require_complete::<Fr>().is_ok());
                assert!(challenges.require_complete().is_ok());
            } else {
                assert!(progress.validate_round::<Fr>(0, false).is_ok());
                assert!(challenges.validate_round(0, false).is_ok());
                progress.record_message();
                challenges.record_message();
                assert!(progress.validate_round::<Fr>(0, false).is_err());
                assert!(challenges.validate_round(0, false).is_err());
                assert!(progress.validate_finish::<Fr>().is_ok());
                assert!(challenges.validate_finish().is_ok());
                progress.advance();
                challenges.push(Fr::from_u64(7));
                assert!(progress.validate_finish::<Fr>().is_err());
                assert!(challenges.validate_finish().is_err());
                assert!(progress.require_complete::<Fr>().is_ok());
                assert!(challenges.require_complete().is_ok());
            }
        }
    }
}
