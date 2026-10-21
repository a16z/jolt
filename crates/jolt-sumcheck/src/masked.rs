//! Masked sumcheck rounds: every private coefficient arrives as m̂ = m + p,
//! so the verifier records each round's wire values, pad slots and
//! challenge, and leaves the round checks to the residual relation.

use jolt_algebra::{CanonicalEncoding, Field};
use jolt_transcript::{Channel, PadSlots, PrivateChannel, TranscriptError};

use crate::error::SumcheckError;
use crate::schedule::DegreeSchedule;

/// One masked round as the verifier received it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MaskedRound<F> {
    /// The masked wire coefficients: `[c0, c2, …]` for a compressed round,
    /// every coefficient for a full one.
    pub wire: Vec<F>,
    pub slots: PadSlots,
    pub challenge: F,
}

/// Which coefficients a round sends.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RoundWire {
    /// The compressed Boolean-hypercube form: `degree` values, the linear
    /// coefficient omitted.
    Compressed,
    /// All `degree + 1` coefficients.
    Full,
}

impl RoundWire {
    /// The number of values a round of degree bound `degree` sends.
    ///
    /// # Errors
    ///
    /// [`SumcheckError::DegreeOverflow`] if `degree + 1` overflows.
    pub fn width<F: Field>(self, degree: usize) -> Result<usize, SumcheckError<F>> {
        match self {
            Self::Compressed => Ok(degree.max(1)),
            Self::Full => degree
                .checked_add(1)
                .ok_or(SumcheckError::DegreeOverflow { degree }),
        }
    }
}

/// A batched masked sumcheck: its rounds, the degree schedule they were read
/// at, and the batching data the residual lowering folds them against.
///
/// A shorter member is active only inside its window. Tail-aligned members
/// (the default) have their challenge suffix begin at
/// `max_num_vars - num_vars`; head-aligned members supply their offset to
/// [`try_instance_point_at`](Self::try_instance_point_at).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchedMaskedRounds<F> {
    pub rounds: Vec<MaskedRound<F>>,
    pub batching_coefficients: Vec<F>,
    pub degrees: DegreeSchedule,
}

impl<F: Field> BatchedMaskedRounds<F> {
    pub fn challenges(&self) -> Vec<F> {
        self.rounds.iter().map(|round| round.challenge).collect()
    }

    /// The batch's round count.
    pub fn max_num_vars(&self) -> usize {
        self.degrees.num_rounds()
    }

    /// The tail-aligned offset `max_num_vars - num_vars`.
    ///
    /// # Errors
    ///
    /// [`SumcheckError::BatchedPointOutOfRange`] if `num_vars` exceeds the
    /// batch.
    pub fn try_round_offset(&self, num_vars: usize) -> Result<usize, SumcheckError<F>> {
        self.max_num_vars()
            .checked_sub(num_vars)
            .ok_or(SumcheckError::BatchedPointOutOfRange {
                offset: 0,
                num_vars,
                total: self.rounds.len(),
            })
    }

    /// The tail-aligned challenge suffix of a member with `num_vars` rounds.
    ///
    /// # Errors
    ///
    /// As [`try_round_offset`](Self::try_round_offset).
    pub fn try_instance_point(&self, num_vars: usize) -> Result<Vec<F>, SumcheckError<F>> {
        self.try_instance_point_at(self.try_round_offset(num_vars)?, num_vars)
    }

    /// The `num_vars` challenges starting at `offset`.
    ///
    /// # Errors
    ///
    /// [`SumcheckError::BatchedPointOutOfRange`] if the window leaves the
    /// batch.
    pub fn try_instance_point_at(
        &self,
        offset: usize,
        num_vars: usize,
    ) -> Result<Vec<F>, SumcheckError<F>> {
        let end = offset
            .checked_add(num_vars)
            .ok_or(SumcheckError::BatchedPointRangeOverflow { offset, num_vars })?;
        self.rounds
            .get(offset..end)
            .ok_or(SumcheckError::BatchedPointOutOfRange {
                offset,
                num_vars,
                total: self.rounds.len(),
            })
            .map(|rounds| rounds.iter().map(|round| round.challenge).collect())
    }
}

/// Reads one masked round per entry of `degrees`, at that degree's width in
/// the `wire` format, each followed by its challenge. Masked proofs reveal no
/// claim scalar, so nothing is checked here: the round checks become
/// residual rows.
///
/// # Errors
///
/// The channel's error on a truncated proof or an exhausted pad schedule.
pub fn receive_masked_rounds<F, C>(
    degrees: &DegreeSchedule,
    wire: RoundWire,
    channel: &mut C,
) -> Result<Vec<MaskedRound<F>>, C::Error>
where
    F: Field + CanonicalEncoding,
    C: Channel + PrivateChannel,
    C::Error: From<TranscriptError> + From<SumcheckError<F>>,
{
    let mut rounds = Vec::with_capacity(degrees.num_rounds());
    for &degree in degrees.degrees() {
        let mut values = vec![F::zero(); wire.width::<F>(degree)?];
        let slots = channel.exchange_private_base(&mut values)?;
        rounds.push(MaskedRound {
            wire: values,
            slots,
            challenge: channel.challenge_small(),
        });
    }
    Ok(rounds)
}
