//! The public per-round degree schedule of a sumcheck.

use jolt_algebra::Field;

use crate::error::SumcheckError;

/// The degree bound of every round of a sumcheck, in round order. Round `j`
/// sends its polynomial at `degree(j)`'s wire width (the compressed form
/// sends `degree(j)` coefficients), so prover and verifier agree on every
/// message's length without sending it.
///
/// A batch's schedule is derived from public data only:
/// [`batched`](Self::batched) takes each member's activation offset and its
/// declared per-round degrees. Every consumer — the prove loop, the clear
/// verifier, and the masked verifier and its residual lowering — reads the
/// same schedule.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DegreeSchedule {
    degrees: Vec<usize>,
}

/// One batch member's contribution to a [`DegreeSchedule`]: its activation
/// offset in the batch and its declared degree bound for each of its own
/// rounds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemberDegrees {
    pub offset: usize,
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
    /// leave the batch.
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
            for (slot, degree) in slots.iter_mut().zip(described.round_degrees) {
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

    /// Checks every round is provable: degree at least 1 and a representable
    /// coefficient count.
    ///
    /// # Errors
    ///
    /// [`SumcheckError::ZeroBatchDegree`] or [`SumcheckError::DegreeOverflow`]
    /// at the first offending round.
    pub(crate) fn validate<F: Field>(&self) -> Result<(), SumcheckError<F>> {
        for &degree in &self.degrees {
            if degree == 0 {
                return Err(SumcheckError::ZeroBatchDegree {
                    max_num_vars: self.num_rounds(),
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
    use jolt_algebra::Prime64Offset59;

    use super::{DegreeSchedule, MemberDegrees};
    use crate::SumcheckError;

    type F = Prime64Offset59;

    fn member(offset: usize, round_degrees: &[usize]) -> MemberDegrees {
        MemberDegrees {
            offset,
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
    fn validation_rejects_zero_and_unrepresentable_degrees() {
        assert!(matches!(
            DegreeSchedule::uniform(2, 0).validate::<F>(),
            Err(SumcheckError::ZeroBatchDegree { max_num_vars: 2 })
        ));
        assert!(matches!(
            DegreeSchedule::uniform(1, usize::MAX).validate::<F>(),
            Err(SumcheckError::DegreeOverflow { degree: usize::MAX })
        ));
        assert!(DegreeSchedule::uniform(0, 0).validate::<F>().is_ok());
    }
}
