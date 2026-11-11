//! Masked sumcheck rounds: every private coefficient arrives as m̂ = m + p,
//! so the verifier records each round's wire values, pad slots and
//! challenge, and leaves the round checks to the residual relation.

use jolt_field::Field;

use crate::batch::{instance_window, tail_offset};
use crate::error::SumcheckError;
use crate::schedule::DegreeSchedule;
use crate::single::{RoundSpec, VerifierRoundChannel};

/// Where a masked channel placed a private message's pads: `len` pad
/// coordinates from `start` in the pad schedule.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PadSlots {
    /// The first pad coordinate.
    pub start: usize,
    /// The number of pad coordinates.
    pub len: usize,
}

/// One masked round as the verifier received it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MaskedRound<F> {
    /// The masked wire coefficients: `[c0, c2, …]` for a compressed round,
    /// `[q1, …, qd]` for an eq-factored one, every coefficient for a full
    /// one.
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
    rounds: Vec<MaskedRound<F>>,
    batching_coefficients: Vec<F>,
    degrees: DegreeSchedule,
}

impl<F: Field> BatchedMaskedRounds<F> {
    /// Checks that `rounds` holds one round per entry of `degrees`, each
    /// carrying the `wire` format's width at that degree.
    ///
    /// # Errors
    ///
    /// [`SumcheckError::WrongNumberOfRounds`] on a round-count mismatch,
    /// [`SumcheckError::WrongRoundWidth`] on a wire-width mismatch,
    /// and [`SumcheckError::DegreeOverflow`] if a width overflows.
    pub fn try_new(
        rounds: Vec<MaskedRound<F>>,
        batching_coefficients: Vec<F>,
        degrees: DegreeSchedule,
        wire: RoundWire,
    ) -> Result<Self, SumcheckError<F>> {
        if rounds.len() != degrees.num_rounds() {
            return Err(SumcheckError::WrongNumberOfRounds {
                expected: degrees.num_rounds(),
                got: rounds.len(),
            });
        }
        for (index, (round, &degree)) in rounds.iter().zip(degrees.degrees()).enumerate() {
            let width = wire.width::<F>(degree)?;
            if round.wire.len() != width {
                return Err(SumcheckError::WrongRoundWidth {
                    round: index,
                    expected: width,
                    got: round.wire.len(),
                });
            }
        }
        Ok(Self {
            rounds,
            batching_coefficients,
            degrees,
        })
    }

    pub fn rounds(&self) -> &[MaskedRound<F>] {
        &self.rounds
    }

    pub fn batching_coefficients(&self) -> &[F] {
        &self.batching_coefficients
    }

    pub fn degrees(&self) -> &DegreeSchedule {
        &self.degrees
    }

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
        tail_offset(self.max_num_vars(), num_vars, self.rounds.len())
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
        Ok(instance_window(&self.rounds, offset, num_vars)?
            .iter()
            .map(|round| round.challenge)
            .collect())
    }
}

/// Reads one masked round per entry of `degrees` through a masked verifier
/// round channel: `wire`'s width at that degree, the round's proof of work
/// priced at the degree, then its challenge. Masked proofs reveal no claim
/// scalar, so nothing is checked here: the round checks become residual rows.
///
/// # Errors
///
/// The channel's error on a truncated proof, an exhausted pad schedule, a
/// rejected proof of work, or a wire of the wrong width.
pub fn receive_masked_rounds<F, C>(
    degrees: &DegreeSchedule,
    wire: RoundWire,
    channel: &mut C,
) -> Result<Vec<MaskedRound<F>>, C::Error>
where
    F: Field,
    C: VerifierRoundChannel<F, Slots = PadSlots>,
{
    let mut rounds = Vec::with_capacity(degrees.num_rounds());
    for (round, &degree) in degrees.degrees().iter().enumerate() {
        let width = wire
            .width::<F>(degree)
            .map_err(|error| channel.map_sumcheck_error(error))?;
        let received = channel
            .verify_round(
                round,
                RoundSpec { width, degree },
                |wire, slots, challenge| {
                    if wire.len() != width {
                        return Ok(Err(SumcheckError::WrongRoundWidth {
                            round,
                            expected: width,
                            got: wire.len(),
                        }));
                    }
                    Ok(Ok(MaskedRound {
                        wire: wire.into_owned(),
                        slots,
                        challenge,
                    }))
                },
            )?
            .map_err(|error| channel.map_sumcheck_error(error))?;
        rounds.push(received);
    }
    Ok(rounds)
}

#[cfg(test)]
#[expect(clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use std::borrow::Cow;

    use jolt_field::{Prime64Offset59, Ring};
    use jolt_poly::UnivariatePoly;

    use super::{receive_masked_rounds, BatchedMaskedRounds, MaskedRound, PadSlots, RoundWire};
    use crate::round_proof::padded_coefficients;
    use crate::single::{RoundSpec, SumcheckShape, VerifierRoundChannel};
    use crate::{DegreeSchedule, MemberDegrees, SumcheckError};

    type F = Prime64Offset59;

    /// Lends each round a wire of the width it is asked for (or `short`
    /// values fewer), places it at the next pad slots, and records the
    /// specs it was asked to read.
    struct Tape {
        next_slot: usize,
        specs: Vec<RoundSpec>,
        short: usize,
    }

    impl VerifierRoundChannel<F> for Tape {
        type Error = SumcheckError<F>;
        type Slots = PadSlots;
        fn validate_shape(&self, _shape: SumcheckShape) -> Result<(), Self::Error> {
            Ok(())
        }
        fn map_sumcheck_error(&self, error: SumcheckError<F>) -> Self::Error {
            error
        }
        fn verify_round<R>(
            &mut self,
            round: usize,
            spec: RoundSpec,
            visit: impl FnOnce(Cow<'_, [F]>, PadSlots, F) -> Result<R, Self::Error>,
        ) -> Result<R, Self::Error> {
            self.specs.push(spec);
            let len = spec.width - self.short;
            let slots = PadSlots {
                start: self.next_slot,
                len,
            };
            self.next_slot += len;
            let wire = vec![F::from_u64(round as u64); len];
            visit(Cow::Owned(wire), slots, F::from_u64(100 + round as u64))
        }
    }

    fn schedule() -> DegreeSchedule {
        DegreeSchedule::batched::<F>(
            3,
            [MemberDegrees {
                offset: 0,
                degree: 3,
                round_degrees: vec![3, 1, 2],
            }],
        )
        .unwrap()
    }

    #[test]
    fn rounds_are_read_at_their_format_width_and_scheduled_degree() {
        for (wire, widths) in [
            (RoundWire::Compressed, [3, 1, 2]),
            (RoundWire::Full, [4, 2, 3]),
        ] {
            let mut tape = Tape {
                next_slot: 0,
                specs: Vec::new(),
                short: 0,
            };
            let rounds = receive_masked_rounds(&schedule(), wire, &mut tape).unwrap();
            let expected: Vec<RoundSpec> = widths
                .iter()
                .zip([3, 1, 2])
                .map(|(&width, degree)| RoundSpec { width, degree })
                .collect();
            assert_eq!(tape.specs, expected);
            let starts: Vec<usize> = rounds.iter().map(|round| round.slots.start).collect();
            assert_eq!(starts, [0, widths[0], widths[0] + widths[1]]);
            assert_eq!(rounds[2].challenge, F::from_u64(102));
            let batch =
                BatchedMaskedRounds::try_new(rounds, vec![F::from_u64(1)], schedule(), wire)
                    .unwrap();
            assert_eq!(batch.max_num_vars(), 3);
        }
    }

    #[test]
    fn a_short_wire_is_rejected() {
        let mut tape = Tape {
            next_slot: 0,
            specs: Vec::new(),
            short: 1,
        };
        assert!(matches!(
            receive_masked_rounds(&schedule(), RoundWire::Full, &mut tape),
            Err(SumcheckError::WrongRoundWidth {
                round: 0,
                expected: 4,
                got: 3
            })
        ));
        assert_eq!(tape.specs.len(), 1);
    }

    #[test]
    fn batched_rounds_must_match_their_schedule() {
        let round = |width: usize| MaskedRound {
            wire: vec![F::from_u64(0); width],
            slots: PadSlots::default(),
            challenge: F::from_u64(0),
        };
        assert!(matches!(
            BatchedMaskedRounds::try_new(
                vec![round(4), round(2)],
                Vec::new(),
                schedule(),
                RoundWire::Full
            ),
            Err(SumcheckError::WrongNumberOfRounds {
                expected: 3,
                got: 2
            })
        ));
        assert!(matches!(
            BatchedMaskedRounds::try_new(
                vec![round(4), round(3), round(3)],
                Vec::new(),
                schedule(),
                RoundWire::Full
            ),
            Err(SumcheckError::WrongRoundWidth {
                round: 1,
                expected: 2,
                got: 3
            })
        ));
    }

    #[test]
    fn padding_ignores_zero_leading_coefficients_like_the_prover() {
        let [one, two] = [1, 2].map(F::from_u64);
        let trailing_zero = UnivariatePoly::new(vec![one, two, F::from_u64(0), F::from_u64(0)]);
        assert_eq!(
            padded_coefficients(&trailing_zero, 1).unwrap(),
            vec![one, two]
        );
        let cubic = UnivariatePoly::new(vec![one, two, F::from_u64(0), two]);
        assert!(matches!(
            padded_coefficients(&cubic, 2),
            Err(SumcheckError::DegreeBoundExceeded { got: 3, max: 2 })
        ));
    }
}
