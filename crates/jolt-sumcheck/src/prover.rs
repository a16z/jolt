//! The prove-side sumcheck engine: the batched round loop and the uni-skip
//! first-round prover.
//!
//! [`prove_batch`] is the single batched-sumcheck prover, the mirror of the
//! generated per-stage `verify_clear`/`verify_zk` tails in `jolt-verifier`: it
//! consumes the [`BatchPrelude`] the generated `begin_batch` produced (so the
//! head's Fiat-Shamir sequence is shared code, not convention), drives the
//! members through the front-loaded-padding round loop, and records rounds
//! through a [`SumcheckRecorder`] — the clear/ZK seam. Only this engine and
//! the recorder touch the transcript; batch members compute pure field data.
//!
//! [`prove_uniskip_clear`] / [`prove_uniskip_committed`] mirror
//! `jolt-verifier/src/stages/uniskip.rs`'s two verify arms: a univariate-skip
//! round is a genuinely different round type (separate wire proof, single
//! degree-bounded round over a centered integer domain, full — not compressed
//! — coefficients in the clear), so it is not a batch member and does not go
//! through the recorder.

use jolt_crypto::VectorCommitment;
use jolt_field::Field;
use jolt_poly::UnivariatePoly;
use jolt_transcript::Transcript;

use crate::batch::BatchPrelude;
use crate::committed::{CommittedSumcheckBuilder, CommittedSumcheckWitness};
use crate::domain::{CenteredIntegerDomain, SumcheckDomain};
use crate::error::SumcheckError;
use crate::proof::{ClearProof, ClearSumcheckProof, SumcheckProof};
use crate::recorder::SumcheckRecorder;
use crate::round_proof::{LabeledRoundPoly, RoundMessage};
use crate::OPENING_CLAIM_TRANSCRIPT_LABEL;

/// One batch member's prove-side round interface, consumed by
/// [`prove_batch`]'s round loop. Object-safe on purpose: a stage's members are
/// heterogeneous, and the engine takes them as `&mut [&mut dyn ProveRounds<F>]`.
///
/// `round` indices are member-local (`0..num_rounds()`): under front-loaded
/// padding a member is consulted only during its active window, and never
/// learns the batch's earlier challenges (its opening point is the batch
/// point's suffix).
pub trait ProveRounds<F: Field> {
    /// The number of rounds/variables in this member's sumcheck.
    fn num_rounds(&self) -> usize;

    /// Compute this member's round polynomial for member-local `round`.
    /// `previous_claim` is the member's own running claim; the returned
    /// polynomial must satisfy `s(0) + s(1) == previous_claim`.
    fn compute_message(
        &mut self,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>>;

    /// Bind this member's state to the round challenge for member-local
    /// `round`.
    fn ingest_challenge(&mut self, challenge: F, round: usize) -> Result<(), SumcheckError<F>>;
}

/// A proved batch: the round challenges (the batch opening point), the final
/// combined running claim (what the verifier's `expected_final_claim` must
/// reproduce — stage recipes hard-check this), and each member's final bound
/// claim in declaration order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProvedBatch<F> {
    pub challenges: Vec<F>,
    pub final_claim: F,
    pub member_claims: Vec<F>,
}

/// Drop trailing zero coefficients down to the minimum two (degree 1) the
/// compressed wire form requires. The batched polynomial is assembled over
/// `max_degree + 1` slots, so rounds where every active member's degree is
/// lower carry trailing zeros that must not reach the wire.
fn trim_round_polynomial<F: Field>(mut coefficients: Vec<F>) -> UnivariatePoly<F> {
    while coefficients.len() > 2 && coefficients.last().is_some_and(|value| *value == F::zero()) {
        let _ = coefficients.pop();
    }
    UnivariatePoly::new(coefficients)
}

/// Prove one batched sumcheck, mirroring the generated verify drivers'
/// structure: per round, combine the active members' round polynomials with
/// their batching coefficients (an inactive member contributes the constant
/// `claim / 2`, halving its front-loaded padding scale), self-check
/// `s(0) + s(1)` against the running claim, record the round through the
/// recorder (clear: compressed append; committed: Pedersen), and bind every
/// active member to the squeezed challenge.
///
/// The caller finishes the proof afterwards — typed output-claim extraction is
/// stage-side, so the stage computes its canonical opening values and calls
/// `recorder.finish` itself.
///
/// # Panics
///
/// Panics if `prelude.max_degree == 0` and the batch has rounds to prove — a
/// sumcheck round polynomial must have degree at least 1 (the same invariant
/// `SumcheckClaim::new` enforces on the verify side).
pub fn prove_batch<F, R, T>(
    prelude: &BatchPrelude<F>,
    members: &mut [&mut dyn ProveRounds<F>],
    recorder: &mut R,
    transcript: &mut T,
) -> Result<ProvedBatch<F>, SumcheckError<F>>
where
    F: Field,
    R: SumcheckRecorder<F>,
    T: Transcript<Challenge = F>,
{
    if members.len() != prelude.members.len() {
        return Err(SumcheckError::BatchMemberCountMismatch {
            expected: prelude.members.len(),
            got: members.len(),
        });
    }
    for (index, (member, described)) in members.iter().zip(&prelude.members).enumerate() {
        if member.num_rounds() != described.rounds {
            return Err(SumcheckError::BatchMemberRoundsMismatch {
                member: index,
                expected: described.rounds,
                got: member.num_rounds(),
            });
        }
    }
    let max_num_vars = prelude.max_num_vars;
    assert!(
        max_num_vars == 0 || prelude.max_degree >= 1,
        "sumcheck round polynomial must have degree >= 1"
    );

    #[expect(
        clippy::unwrap_used,
        reason = "2 is invertible in any field of characteristic != 2, and Jolt fields are large-prime"
    )]
    let two_inv = F::from_u64(2).inverse().unwrap();
    // Each member's running claim, at the front-loaded padding scale: a member
    // starts at `input_claim * 2^(max - rounds)` and halves once per inactive
    // round, reaching its true input claim exactly when it activates.
    let mut member_claims: Vec<F> = prelude
        .members
        .iter()
        .map(|member| member.input_claim.mul_pow_2(max_num_vars - member.rounds))
        .collect();
    let mut running_claim = prelude.claimed_sum;
    let mut challenges = Vec::with_capacity(max_num_vars);

    for round in 0..max_num_vars {
        let mut batched_coefficients = vec![F::zero(); prelude.max_degree + 1];
        let mut round_polys: Vec<Option<UnivariatePoly<F>>> = Vec::with_capacity(members.len());

        for (index, member) in members.iter_mut().enumerate() {
            let described = &prelude.members[index];
            let activation = max_num_vars - described.rounds;
            if round < activation {
                // Inactive: the constant polynomial `claim / 2`, so
                // `s(0) + s(1)` preserves the member's claim and evaluation at
                // any challenge halves it.
                batched_coefficients[0] += described.coefficient * member_claims[index] * two_inv;
                round_polys.push(None);
                continue;
            }
            let poly = member.compute_message(round - activation, member_claims[index])?;
            let poly_degree = poly.degree();
            if poly_degree > prelude.max_degree {
                return Err(SumcheckError::DegreeBoundExceeded {
                    got: poly_degree,
                    max: prelude.max_degree,
                });
            }
            for (slot, coefficient) in batched_coefficients.iter_mut().zip(poly.coefficients()) {
                *slot += described.coefficient * *coefficient;
            }
            round_polys.push(Some(poly));
        }

        let batched_poly = trim_round_polynomial(batched_coefficients);
        let round_sum = batched_poly.evaluate(F::zero()) + batched_poly.evaluate(F::one());
        if round_sum != running_claim {
            return Err(SumcheckError::RoundCheckFailed {
                round,
                expected: running_claim,
                actual: round_sum,
            });
        }

        let challenge = recorder.absorb_round(&batched_poly, transcript)?;
        running_claim = batched_poly.evaluate(challenge);
        challenges.push(challenge);

        for (index, (member, poly)) in members.iter_mut().zip(round_polys).enumerate() {
            match poly {
                Some(poly) => {
                    member_claims[index] = poly.evaluate(challenge);
                    let activation = max_num_vars - prelude.members[index].rounds;
                    member.ingest_challenge(challenge, round - activation)?;
                }
                None => member_claims[index] *= two_inv,
            }
        }
    }

    Ok(ProvedBatch {
        challenges,
        final_claim: running_claim,
        member_claims,
    })
}

/// A proved clear uni-skip round: the one-round full-coefficient wire proof,
/// the reduction challenge, and the output claim (the round polynomial at the
/// challenge — the batch driver absorbs it again as the remainder's input
/// claim).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProvedUniskip<F: Field, C> {
    pub proof: SumcheckProof<F, C>,
    pub challenge: F,
    pub output_claim: F,
}

/// A proved committed uni-skip round: the committed wire proof, its retained
/// witness (for BlindFold), the reduction challenge, and the (prover-internal,
/// never absorbed) output claim.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProvedUniskipCommitted<F: Field, C> {
    pub proof: SumcheckProof<F, C>,
    pub witness: CommittedSumcheckWitness<F>,
    pub challenge: F,
    pub output_claim: F,
}

/// Self-check the uni-skip round polynomial against the verifier's round
/// checks before anything reaches the transcript: degree bound and
/// centered-integer-domain round sum.
fn check_uniskip_round<F: Field>(
    round_poly: &UnivariatePoly<F>,
    input_claim: F,
    degree: usize,
    domain_size: usize,
) -> Result<(), SumcheckError<F>> {
    if round_poly.degree() > degree {
        return Err(SumcheckError::DegreeBoundExceeded {
            got: round_poly.degree(),
            max: degree,
        });
    }
    CenteredIntegerDomain::new(domain_size).check_round_sum(
        0,
        input_claim,
        &LabeledRoundPoly::uniskip(round_poly),
    )
}

/// Prove a clear uni-skip first round, mirroring the verifier's
/// `uniskip::verify_clear`: absorb the full labeled round polynomial, squeeze
/// the reduction challenge, evaluate the output claim, and absorb it under
/// `b"opening_claim"` — before any post-uni-skip draw (the remainder batch's
/// coefficient squeeze in particular), which is why the absorb lives here and
/// not in the stage.
pub fn prove_uniskip_clear<F, C, T>(
    round_poly: UnivariatePoly<F>,
    input_claim: F,
    degree: usize,
    domain_size: usize,
    transcript: &mut T,
) -> Result<ProvedUniskip<F, C>, SumcheckError<F>>
where
    F: Field,
    T: Transcript<Challenge = F>,
{
    check_uniskip_round(&round_poly, input_claim, degree, domain_size)?;

    LabeledRoundPoly::uniskip(&round_poly).append_to_transcript(transcript);
    let challenge = transcript.challenge();
    let output_claim = round_poly.evaluate(challenge);
    transcript.append_labeled(OPENING_CLAIM_TRANSCRIPT_LABEL, &output_claim);

    Ok(ProvedUniskip {
        proof: SumcheckProof::Clear(ClearProof::Full(ClearSumcheckProof {
            round_polynomials: vec![round_poly],
        })),
        challenge,
        output_claim,
    })
}

/// Prove a committed uni-skip first round, mirroring the verifier's
/// `uniskip::verify_zk`: commit the round polynomial (absorbing only the
/// commitment), squeeze the reduction challenge, then commit and absorb the
/// output claim. The claim scalar never reaches the transcript.
pub fn prove_uniskip_committed<F, VC, T>(
    round_poly: UnivariatePoly<F>,
    input_claim: F,
    degree: usize,
    domain_size: usize,
    setup: &VC::Setup,
    transcript: &mut T,
) -> Result<ProvedUniskipCommitted<F, VC::Output>, SumcheckError<F>>
where
    F: Field,
    VC: VectorCommitment<Field = F>,
    T: Transcript<Challenge = F>,
{
    check_uniskip_round(&round_poly, input_claim, degree, domain_size)?;

    let mut builder = CommittedSumcheckBuilder::<F, VC>::new(setup)?;
    let challenge = builder.commit_round(&round_poly, transcript)?;
    let output_claim = round_poly.evaluate(challenge);
    let (proof, witness) = builder.finish(&[output_claim], transcript)?;

    Ok(ProvedUniskipCommitted {
        proof,
        witness,
        challenge,
        output_claim,
    })
}
