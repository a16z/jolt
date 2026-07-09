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
/// claim), its batching coefficient, and its round count. Members are ordered
/// by stage declaration order — the Fiat-Shamir absorb/draw order.
///
/// Under the front-loaded batching layout a member with fewer rounds than the
/// batch is inactive for the first `max_num_vars - rounds` rounds, so `rounds`
/// doubles as the member's activation window.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchMember<F> {
    pub input_claim: F,
    pub coefficient: F,
    pub rounds: usize,
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
    /// `2^(max_num_vars − rounds)` scale front-loads each shorter member's
    /// padding: an inactive round halves the member's contribution, so after
    /// its `max_num_vars − rounds` inactive rounds the member enters its first
    /// active round at its unscaled input claim.
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
