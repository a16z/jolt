//! Batched-sumcheck head data shared by the two sides of the protocol.
//!
//! A batched sumcheck's *head* — per-member input claims absorbed, one batching
//! coefficient drawn per member, the padded random linear combination — is
//! computed once by the generated per-stage `begin_batch` driver
//! (`#[derive(SumcheckBatch)]` in `jolt-verifier`) and consumed by both the
//! clear verify tail and the prove-side round loop. [`BatchPrelude`] is that
//! head's output in engine form: plain positional data with no per-stage
//! types, so this crate's provers can consume it without naming any stage.

use jolt_field::Field;

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

impl<F: Field> BatchPrelude<F> {
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
    /// Panics if a member has more rounds than `max_num_vars` (a wiring bug —
    /// `max_num_vars` is defined as the members' maximum).
    pub fn new(members: Vec<BatchMember<F>>, max_num_vars: usize, max_degree: usize) -> Self {
        let claimed_sum = members
            .iter()
            .map(|member| {
                member.coefficient * member.input_claim.mul_pow_2(max_num_vars - member.rounds)
            })
            .sum();
        Self {
            members,
            claimed_sum,
            max_num_vars,
            max_degree,
        }
    }
}
