//! Batched-sumcheck head data shared by the two sides of the protocol.
//!
//! A batched sumcheck's *head* — per-member input claims absorbed, one batching
//! coefficient drawn per member, the padded random linear combination — is
//! computed once by the generated per-stage `begin_batch` driver
//! (`#[derive(SumcheckBatch)]` in `jolt-verifier`) and consumed by both the
//! clear verify tail and the prove-side round loop. [`BatchPrelude`] is that
//! head's output in engine form: plain positional data with no per-stage
//! types, so this crate's provers can consume it without naming any stage.

use jolt_field::JoltField;

use crate::SumcheckError;

/// One present batch member: its input claim (the member's initial running
/// claim), its batching coefficient, its round count, and its activation
/// offset. Members are ordered by stage declaration order — the Fiat-Shamir
/// absorb/draw order.
///
/// A member is active for rounds `[offset, offset + rounds)` and contributes
/// the constant `claim / 2` polynomial outside that window. Most members are
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
/// order), the combined claim, and the batch dimensions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchPrelude<F> {
    pub members: Vec<BatchMember<F>>,
    /// The padded random linear combination of the members' input claims —
    /// the batch's initial running claim:
    /// `Σ coefficient · input_claim · 2^(max_num_vars − rounds)`.
    pub claimed_sum: F,
    pub max_num_vars: usize,
    pub max_degree: usize,
}

impl<F: JoltField> BatchPrelude<F> {
    /// Combine `members` into the batch's initial running claim. The
    /// `2^(max_num_vars − rounds)` scale is each shorter member's dummy-round
    /// padding — its summand extended constantly over the batch's extra
    /// variables — and is independent of where the member's window sits. A
    /// tail-aligned member halves through its inactive rounds and enters its
    /// window at the unscaled input claim; a head-aligned member is active
    /// immediately at the padded scale, so its kernel must emit round
    /// polynomials carrying that scale.
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
    pub fn new(members: Vec<BatchMember<F>>, max_num_vars: usize, max_degree: usize) -> Self {
        Self::try_new(members, max_num_vars, max_degree).expect("invalid sumcheck batch prelude")
    }

    /// Validates and constructs a batched sumcheck prelude.
    pub fn try_new(
        members: Vec<BatchMember<F>>,
        max_num_vars: usize,
        max_degree: usize,
    ) -> Result<Self, SumcheckError<F>> {
        validate_batch_dimensions(&members, max_num_vars, max_degree)?;
        let claimed_sum = members
            .iter()
            .map(|member| {
                member.coefficient * member.input_claim.mul_pow_2(max_num_vars - member.rounds)
            })
            .sum();
        Ok(Self {
            members,
            claimed_sum,
            max_num_vars,
            max_degree,
        })
    }

    pub(crate) fn validate(&self) -> Result<(), SumcheckError<F>> {
        validate_batch_dimensions(&self.members, self.max_num_vars, self.max_degree)
    }
}

fn validate_batch_dimensions<F: JoltField>(
    members: &[BatchMember<F>],
    max_num_vars: usize,
    max_degree: usize,
) -> Result<(), SumcheckError<F>> {
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
    if max_num_vars > 0 && max_degree == 0 {
        return Err(SumcheckError::ZeroBatchDegree { max_num_vars });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use jolt_field::{Prime64Offset59, Ring};

    use super::{BatchMember, BatchPrelude};
    use crate::SumcheckError;

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
            BatchPrelude::try_new(vec![member(3, 0)], 2, 1),
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
            BatchPrelude::try_new(vec![member(1, usize::MAX)], 2, 1),
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
            BatchPrelude::try_new(vec![member(2, 1)], 2, 1),
            Err(SumcheckError::BatchMemberWindowOutOfRange {
                member: 0,
                offset: 1,
                rounds: 2,
                max_num_vars: 2
            })
        ));
    }
}
