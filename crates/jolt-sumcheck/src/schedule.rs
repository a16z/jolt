//! The public per-round degree schedule of a sumcheck.

use jolt_field::Field;

use crate::error::SumcheckError;

/// The degree bound of every round of a sumcheck, in round order. Round `j`'s
/// polynomial has degree at most `degrees()[j]`: the verifier rejects a
/// longer one, and the round's proof of work is priced at that degree.
///
/// A batch's schedule is derived from public data only:
/// [`batched`](Self::batched) takes each member's activation offset and its
/// declared per-round degrees. The prove loop and the clear verifier read the
/// same schedule, from the generated `degree_schedule`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DegreeSchedule {
    degrees: Vec<usize>,
}

/// One batch member's contribution to a [`DegreeSchedule`]: its activation
/// offset in the batch, its declared degree bound `degree`, and its declared
/// bound for each of its own rounds, each at most `degree`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemberDegrees {
    pub offset: usize,
    pub degree: usize,
    pub round_degrees: Vec<usize>,
}

impl DegreeSchedule {
    /// `num_rounds` rounds at the same degree bound.
    pub fn uniform(num_rounds: usize, degree: usize) -> Self {
        Self {
            degrees: vec![degree; num_rounds],
        }
    }

    /// The schedule of a `num_rounds`-round batch: each round's degree is the
    /// largest degree any member active in that round declares for it, and at
    /// least 1. Inactive members contribute at most degree 1 under either
    /// padding rule of [`BatchPrelude`](crate::BatchPrelude): a constant under
    /// constant extension, `p · m · (1 − X)` under zero extension.
    ///
    /// # Errors
    ///
    /// [`SumcheckError::BatchMemberWindowOverflow`] or
    /// [`SumcheckError::BatchMemberWindowOutOfRange`] if a member's rounds
    /// leave the batch, and [`SumcheckError::MemberRoundDegreeAboveBound`] if a
    /// member declares a round above its own degree bound.
    pub fn batched<F: Field>(
        num_rounds: usize,
        members: impl IntoIterator<Item = MemberDegrees>,
    ) -> Result<Self, SumcheckError<F>> {
        let mut degrees = vec![1; num_rounds];
        for (member, described) in members.into_iter().enumerate() {
            let offset = described.offset;
            let rounds = described.round_degrees.len();
            let window =
                offset
                    .checked_add(rounds)
                    .ok_or(SumcheckError::BatchMemberWindowOverflow {
                        member,
                        offset,
                        rounds,
                    })?;
            let slots = degrees.get_mut(offset..window).ok_or(
                SumcheckError::BatchMemberWindowOutOfRange {
                    member,
                    offset,
                    rounds,
                    max_num_vars: num_rounds,
                },
            )?;
            for (round, (slot, degree)) in slots.iter_mut().zip(described.round_degrees).enumerate()
            {
                if degree > described.degree {
                    return Err(SumcheckError::MemberRoundDegreeAboveBound {
                        member,
                        round,
                        got: degree,
                        max: described.degree,
                    });
                }
                *slot = (*slot).max(degree);
            }
        }
        Ok(Self { degrees })
    }

    pub fn num_rounds(&self) -> usize {
        self.degrees.len()
    }

    /// The degree bounds in round order.
    pub fn degrees(&self) -> &[usize] {
        &self.degrees
    }

    /// The largest round degree bound, or 0 for a schedule with no rounds.
    pub fn max_degree(&self) -> usize {
        self.degrees.iter().copied().max().unwrap_or(0)
    }

    /// Checks every round is provable: degree at least 1 and a representable
    /// coefficient count.
    ///
    /// # Errors
    ///
    /// [`SumcheckError::ZeroBatchDegree`] or [`SumcheckError::DegreeOverflow`]
    /// at the first offending round.
    pub(crate) fn validate<F: Field>(&self) -> Result<(), SumcheckError<F>> {
        for (round, &degree) in self.degrees.iter().enumerate() {
            if degree == 0 {
                return Err(SumcheckError::ZeroBatchDegree {
                    max_num_vars: self.num_rounds(),
                    round,
                });
            }
            if degree.checked_add(1).is_none() {
                return Err(SumcheckError::DegreeOverflow { degree });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use jolt_field::Prime64Offset59;

    use super::{DegreeSchedule, MemberDegrees};
    use crate::SumcheckError;

    type F = Prime64Offset59;

    fn member(offset: usize, round_degrees: &[usize]) -> MemberDegrees {
        MemberDegrees {
            offset,
            degree: round_degrees.iter().copied().max().unwrap_or(1),
            round_degrees: round_degrees.to_vec(),
        }
    }

    #[test]
    fn each_round_takes_the_largest_active_member_degree() {
        let schedule = DegreeSchedule::batched::<F>(
            6,
            [
                member(0, &[2, 2, 2, 2, 10, 10]),
                member(3, &[3, 3, 3]),
                member(1, &[1]),
            ],
        )
        .unwrap();
        assert_eq!(schedule.degrees(), &[2, 2, 2, 3, 10, 10]);
    }

    #[test]
    fn rounds_without_an_active_member_have_degree_one() {
        let schedule = DegreeSchedule::batched::<F>(4, [member(0, &[3]), member(3, &[2])]).unwrap();
        assert_eq!(schedule.degrees(), &[3, 1, 1, 2]);
    }

    #[test]
    fn members_outside_the_batch_are_rejected() {
        assert!(matches!(
            DegreeSchedule::batched::<F>(3, [member(2, &[1, 1])]),
            Err(SumcheckError::BatchMemberWindowOutOfRange {
                member: 0,
                offset: 2,
                rounds: 2,
                max_num_vars: 3
            })
        ));
        assert!(matches!(
            DegreeSchedule::batched::<F>(3, [member(usize::MAX, &[1])]),
            Err(SumcheckError::BatchMemberWindowOverflow { member: 0, .. })
        ));
    }

    #[test]
    fn a_round_above_the_member_bound_is_rejected() {
        let mut above = member(0, &[2, 3]);
        above.degree = 2;
        assert!(matches!(
            DegreeSchedule::batched::<F>(2, [above]),
            Err(SumcheckError::MemberRoundDegreeAboveBound {
                member: 0,
                round: 1,
                got: 3,
                max: 2
            })
        ));
    }

    #[test]
    fn validation_rejects_zero_and_unrepresentable_degrees() {
        assert!(matches!(
            DegreeSchedule::uniform(2, 0).validate::<F>(),
            Err(SumcheckError::ZeroBatchDegree {
                max_num_vars: 2,
                round: 0
            })
        ));
        assert!(matches!(
            DegreeSchedule::uniform(1, usize::MAX).validate::<F>(),
            Err(SumcheckError::DegreeOverflow { degree: usize::MAX })
        ));
        assert!(DegreeSchedule::uniform(0, 0).validate::<F>().is_ok());
    }
}
