//! Batched-sumcheck head data shared by the two sides of the protocol.
//!
//! A batched sumcheck's *head* — per-member input claims absorbed, one batching
//! coefficient drawn per member, the padded random linear combination — is
//! computed once by the generated per-stage `begin_batch` driver
//! (`#[derive(SumcheckBatch)]` in `jolt-verifier`) and consumed by both the
//! clear verify tail and the prove-side round loop. [`BatchPrelude`] is that
//! head's output in engine form: plain positional data with no per-stage
//! types, so this crate's provers can consume it without naming any stage.
//! Its field-dependent padding contract is documented on [`BatchPrelude`].

use jolt_algebra::Field;

use crate::{DegreeSchedule, SumcheckError};

/// One present batch member: its input claim (the member's initial running
/// claim), its batching coefficient, its round count, and its activation
/// offset. Members are ordered by stage declaration order — the Fiat-Shamir
/// absorb/draw order.
///
/// A member is active for rounds `[offset, offset + rounds)` and contributes
/// the padding polynomial specified by [`BatchPrelude`] outside that window.
/// Most members are
/// tail-aligned (`offset = max_num_vars - rounds`, the relation's default
/// `instance_point_offset`); the precommitted claim-reduction cycle phases
/// are head-aligned (`offset = 0`), binding the batch's leading challenges.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchMember<F> {
    pub input_claim: F,
    pub coefficient: F,
    pub rounds: usize,
    pub offset: usize,
}

/// The computed head of a batched sumcheck: the present members (declaration
/// order), the combined claim, and the batch's degree schedule, whose length
/// is the batch's round count.
///
/// # Padding rules
///
/// Write `n = max_num_vars` and `W = [offset, offset + rounds)` for a member's
/// active window. The field selects the rule: invertible `F::from_u64(2)` gives
/// constant extension; otherwise the summand is zero-extended.
///
/// Under constant extension the input claim is multiplied by `2^(n - rounds)`.
/// Each inactive round contributes the constant `claim / 2` and halves the
/// running member claim. Active members receive that running claim and must
/// return polynomials at its scale, including any remaining padding. The
/// output multiplier is one.
///
/// Under zero extension the summand is multiplied by `∏_{j ∉ W} (1 - x_j)`,
/// so the input claim has multiplier one. The engine keeps a native claim `m`
/// (initially the input claim) and a padding factor `p` (initially one).
/// An inactive round contributes `p * m * (1 - X)` and updates only
/// `p *= 1 - r_j`. An active round passes `m` to the member, folds its native
/// polynomial `s` with multiplier `p`, and updates only `m = s(r_j)`.
/// Neither value is recovered by division: both may be zero. The returned
/// member claim is `p * m`.
///
/// [`Self::member_output_scale`] exposes the output multiplier to verifiers.
/// Its `challenges` are in temporal batch-round order: `challenges[j] = r_j`,
/// with a member opening at the slice for `W`. The multiplier can be zero.
/// It rejects invalid batch dimensions, an out-of-range member index, or a
/// challenge count different from `n` with a typed [`SumcheckError`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchPrelude<F> {
    pub members: Vec<BatchMember<F>>,
    /// The padded random linear combination of the members' input claims —
    /// the batch's initial running claim, under the padding rule above.
    pub claimed_sum: F,
    pub degrees: DegreeSchedule,
}

impl<F> BatchMember<F> {
    pub(crate) fn is_active(&self, round: usize) -> bool {
        round >= self.offset && round < self.offset + self.rounds
    }
}

impl<F: Field> BatchPrelude<F> {
    /// Combine `members` into the initial claim using the [padding rule](Self).
    ///
    /// # Panics
    ///
    /// Panics if the batch dimensions or a member's activation window are
    /// invalid. External callers should use [`Self::try_new`] to receive a
    /// typed error.
    #[expect(
        clippy::expect_used,
        reason = "legacy constructor retained for generated in-repo callers; external callers use try_new"
    )]
    pub fn new(members: Vec<BatchMember<F>>, degrees: DegreeSchedule) -> Self {
        Self::try_new(members, degrees).expect("invalid sumcheck batch prelude")
    }

    pub fn try_new(
        members: Vec<BatchMember<F>>,
        degrees: DegreeSchedule,
    ) -> Result<Self, SumcheckError<F>> {
        validate_batch_dimensions(&members, &degrees)?;
        let max_num_vars = degrees.num_rounds();
        let padding = PaddingRule::for_field();
        let claimed_sum = members
            .iter()
            .map(|member| {
                member.coefficient
                    * padding.pad_input_claim(member.input_claim, max_num_vars - member.rounds)
            })
            .sum();
        Ok(Self {
            members,
            claimed_sum,
            degrees,
        })
    }

    /// The batch's round count.
    pub fn max_num_vars(&self) -> usize {
        self.degrees.num_rounds()
    }

    /// Returns `λ` such that an honest member's final claim is `λ * g(r_W)`.
    /// A verifier must check its reduced value against
    /// `Σ_i coefficient_i * member_output_scale(i, r) * g_i(r_W)` using the
    /// challenges returned by its own round verification. See the
    /// [padding contract](Self) for the rule and challenge order.
    ///
    /// # Errors
    ///
    /// Validates dimensions as [`crate::prove_batch`] does, including the
    /// padding-exponent limit. Then returns
    /// [`SumcheckError::RoundMemberIndexOutOfRange`] for an absent member or
    /// [`SumcheckError::WrongNumberOfChallenges`] for an incorrect challenge count.
    pub fn member_output_scale(
        &self,
        member: usize,
        challenges: &[F],
    ) -> Result<F, SumcheckError<F>> {
        self.validate()?;
        let described =
            self.members
                .get(member)
                .ok_or(SumcheckError::RoundMemberIndexOutOfRange {
                    member,
                    members: self.members.len(),
                })?;
        if challenges.len() != self.max_num_vars() {
            return Err(SumcheckError::WrongNumberOfChallenges {
                expected: self.max_num_vars(),
                got: challenges.len(),
            });
        }
        Ok(match PaddingRule::<F>::for_field() {
            PaddingRule::ConstantExtension { .. } => F::one(),
            PaddingRule::ZeroExtension => challenges
                .iter()
                .enumerate()
                .filter(|(round, _)| !described.is_active(*round))
                .map(|(_, challenge)| F::one() - challenge)
                .product(),
        })
    }

    pub(crate) fn validate(&self) -> Result<(), SumcheckError<F>> {
        validate_batch_dimensions(&self.members, &self.degrees)
    }
}

pub(crate) enum PaddingRule<F> {
    ConstantExtension { two_inv: F },
    ZeroExtension,
}

impl<F: Field> PaddingRule<F> {
    pub(crate) fn for_field() -> Self {
        match F::from_u64(2).inverse() {
            Some(two_inv) => Self::ConstantExtension { two_inv },
            None => Self::ZeroExtension,
        }
    }

    pub(crate) fn pad_input_claim(&self, claim: F, exponent: usize) -> F {
        match self {
            Self::ConstantExtension { .. } => claim.mul_pow_2(exponent),
            Self::ZeroExtension => claim,
        }
    }
}

fn validate_batch_dimensions<F: Field>(
    members: &[BatchMember<F>],
    degrees: &DegreeSchedule,
) -> Result<(), SumcheckError<F>> {
    degrees.validate()?;
    let max_num_vars = degrees.num_rounds();
    for (member, described) in members.iter().enumerate() {
        let exponent = max_num_vars.checked_sub(described.rounds).ok_or(
            SumcheckError::BatchMemberRoundsOutOfRange {
                member,
                rounds: described.rounds,
                max_num_vars,
            },
        )?;
        // `Ring::mul_pow_2` panics above 255.
        if exponent > 255 {
            return Err(SumcheckError::BatchPaddingExponentOutOfRange { member, exponent });
        }
        let window_end = described.offset.checked_add(described.rounds).ok_or(
            SumcheckError::BatchMemberWindowOverflow {
                member,
                offset: described.offset,
                rounds: described.rounds,
            },
        )?;
        if window_end > max_num_vars {
            return Err(SumcheckError::BatchMemberWindowOutOfRange {
                member,
                offset: described.offset,
                rounds: described.rounds,
                max_num_vars,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use jolt_algebra::{Prime64Offset59, Ring};

    use super::{BatchMember, BatchPrelude};
    use crate::{DegreeSchedule, SumcheckError};

    type F = Prime64Offset59;

    fn member(rounds: usize, offset: usize) -> BatchMember<F> {
        BatchMember {
            input_claim: F::from_u64(1),
            coefficient: F::from_u64(1),
            rounds,
            offset,
        }
    }

    #[test]
    fn checked_constructor_rejects_rounds_larger_than_batch() {
        assert!(matches!(
            BatchPrelude::try_new(vec![member(3, 0)], DegreeSchedule::uniform(2, 1)),
            Err(SumcheckError::BatchMemberRoundsOutOfRange {
                member: 0,
                rounds: 3,
                max_num_vars: 2
            })
        ));
    }

    #[test]
    fn checked_constructor_rejects_window_overflow() {
        assert!(matches!(
            BatchPrelude::try_new(vec![member(1, usize::MAX)], DegreeSchedule::uniform(2, 1)),
            Err(SumcheckError::BatchMemberWindowOverflow {
                member: 0,
                offset: usize::MAX,
                rounds: 1
            })
        ));
    }

    #[test]
    fn checked_constructor_rejects_window_outside_batch() {
        assert!(matches!(
            BatchPrelude::try_new(vec![member(2, 1)], DegreeSchedule::uniform(2, 1)),
            Err(SumcheckError::BatchMemberWindowOutOfRange {
                member: 0,
                offset: 1,
                rounds: 2,
                max_num_vars: 2
            })
        ));
    }

    #[test]
    fn checked_constructor_rejects_degree_overflow() {
        assert!(matches!(
            BatchPrelude::<F>::try_new(Vec::new(), DegreeSchedule::uniform(1, usize::MAX)),
            Err(SumcheckError::DegreeOverflow { degree: usize::MAX })
        ));
    }
}
