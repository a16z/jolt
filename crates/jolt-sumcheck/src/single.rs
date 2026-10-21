//! The single-instance sumcheck driver: one unbatched instance over a base
//! field `F` and its extension `E`, with transcript sites and round challenges
//! supplied by a round context that owns the transcript.
//!
//! Two round formats share the driver. [`SingleFormat::Compressed`] sends the
//! round polynomial `s` without its linear coefficient and checks
//! `s(0) + s(1) = claim`. [`SingleFormat::EqFactored`] factors round `i` as
//! `eq(τ_i, X) · q(X)`, sends `q` without its constant coefficient, and checks
//! `(1 − τ_i) q(0) + τ_i q(1) = claim` ([`EqFactoredRound`]); its running claim
//! is normalized to `q(r_i)`, and the caller applies `Π eq(τ_i, r_i)` at the
//! terminal check. Both formats send `degree` coefficients per round.

use jolt_algebra::{CanonicalDecode, CanonicalEncoding, ExtField, Field};
use jolt_poly::{CompressedPoly, OmittedConstantPoly, Point, UnivariatePoly, LOW_TO_HIGH};
use jolt_transcript::{Channel, PadSlots, PrivateChannel, SiteId};

use crate::domain::{BooleanHypercube, EqFactoredRound, SumcheckDomain};
use crate::error::SumcheckError;
use crate::masked::MaskedRound;
use crate::prover::ProveRounds;
use crate::round_proof::padded_coefficients;

/// Which atom of a sumcheck a transcript site tags. The discriminants are
/// part of the site encoding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum SumcheckRole {
    Claim = 1,
    RoundBody = 3,
    Challenge = 4,
}

/// One atom of one sumcheck invocation. The claim site is at round 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SumcheckSite {
    pub invocation: u32,
    pub round: u32,
    pub role: SumcheckRole,
}

/// The prover's round context: it owns the channel the rounds are sent on,
/// names sites, and draws each round challenge, applying whatever grinding
/// its caller schedules.
///
/// The channel is the clear transcript or a masked channel over committed
/// pads (Akita `specs/whiteout-akita.md` §4.5). Its errors reach the driver
/// as a [`SumcheckError`]: [`SumcheckError::Transcript`] for the clear
/// transcript, [`SumcheckError::PrivateChannel`] for a masked channel.
pub trait ProverRoundContext<E: Field> {
    type Channel: Channel + PrivateChannel<Error: Into<SumcheckError<E>>>;
    type Error;

    fn channel_mut(&mut self) -> &mut Self::Channel;

    /// The transcript site for `site`, or `None` to emit no site.
    fn site(&self, site: SumcheckSite) -> Option<SiteId>;

    /// Applies scheduled work and draws the challenge at a
    /// [`SumcheckRole::Challenge`] site.
    fn challenge(&mut self, site: SumcheckSite) -> Result<E, Self::Error>;
}

/// The verifier's mirror of [`ProverRoundContext`]: the clear transcript, or
/// the masked channel [`receive_masked_single`] records pad slots from.
pub trait VerifierRoundContext<E: Field> {
    type Channel: Channel + PrivateChannel<Error: Into<SumcheckError<E>>>;
    type Error;

    fn channel_mut(&mut self) -> &mut Self::Channel;

    /// The transcript site for `site`, or `None` to emit no site.
    fn site(&self, site: SumcheckSite) -> Option<SiteId>;

    /// Checks scheduled work and draws the challenge at a
    /// [`SumcheckRole::Challenge`] site.
    fn challenge(&mut self, site: SumcheckSite) -> Result<E, Self::Error>;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SingleFormat<E> {
    Compressed,
    /// Round `i` uses `τ_i = equality_point[i]`.
    EqFactored {
        equality_point: Point<LOW_TO_HIGH, E>,
    },
}

/// The public shape of one single-instance sumcheck. For
/// [`SingleFormat::EqFactored`], `degree` bounds `q`, and `input_claim` is
/// the unscaled sum.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SingleInstance<E> {
    pub input_claim: E,
    pub rounds: usize,
    pub degree: usize,
    pub format: SingleFormat<E>,
    pub invocation: u32,
}

/// The reduction a single instance proves or replays: the round challenges
/// and the final running claim, normalized for eq-factored instances. The
/// verifier gets it before any oracle check; the caller checks
/// `output_claim` against the oracle at `challenges`, after applying the eq
/// factors `Π eq(τ_i, r_i)` for eq-factored instances.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SingleReduction<E> {
    pub challenges: Vec<E>,
    pub output_claim: E,
}

impl<E: Field> SingleInstance<E> {
    /// Checks the public shape: a nonzero degree bound, round indices that fit
    /// a transcript site, a representable `degree × rounds` proof size, and,
    /// for eq-factored instances, one equality coordinate per round and at
    /// least one round.
    fn validate(&self) -> Result<(), SumcheckError<E>> {
        if self.degree == 0 {
            return Err(SumcheckError::ZeroDegreeBound);
        }
        if u32::try_from(self.rounds).is_err() || self.degree.checked_mul(self.rounds).is_none() {
            return Err(SumcheckError::ShapeOverflow {
                rounds: self.rounds,
                degree: self.degree,
            });
        }
        if let SingleFormat::EqFactored { equality_point } = &self.format {
            if self.rounds == 0 {
                return Err(SumcheckError::EqFactoredZeroRounds);
            }
            if equality_point.len() != self.rounds {
                return Err(SumcheckError::EqualityPointLengthMismatch {
                    expected: self.rounds,
                    got: equality_point.len(),
                });
            }
        }
        Ok(())
    }

    /// The site of `role` in `round`; `round < self.rounds <= u32::MAX` by
    /// [`validate`](Self::validate).
    fn site(&self, round: u32, role: SumcheckRole) -> SumcheckSite {
        SumcheckSite {
            invocation: self.invocation,
            round,
            role,
        }
    }

    /// `τ_round` for eq-factored instances, which [`validate`](Self::validate)
    /// gives one coordinate per round.
    fn tau(&self, round: usize) -> Result<Option<E>, SumcheckError<E>> {
        match &self.format {
            SingleFormat::Compressed => Ok(None),
            SingleFormat::EqFactored { equality_point } => equality_point
                .as_slice()
                .get(round)
                .copied()
                .map(Some)
                .ok_or(SumcheckError::EqualityPointLengthMismatch {
                    expected: self.rounds,
                    got: equality_point.len(),
                }),
        }
    }

    /// Self-checks `poly` against `claim` and returns its `degree` wire
    /// coefficients.
    fn wire_coefficients(
        &self,
        round: usize,
        claim: E,
        poly: &UnivariatePoly<E>,
    ) -> Result<Vec<E>, SumcheckError<E>> {
        let padded = UnivariatePoly::new(padded_coefficients(poly, self.degree)?);
        match self.tau(round)? {
            None => {
                BooleanHypercube.check_round_sum(round, claim, &padded)?;
                Ok(padded.compress().coeffs_except_linear_term().to_vec())
            }
            Some(tau) => {
                EqFactoredRound { tau }.check_round_sum(round, claim, &padded)?;
                Ok(
                    OmittedConstantPoly::from_q_coefficients(padded.into_coefficients())
                        .into_coefficients(),
                )
            }
        }
    }

    /// The running claim after a round whose wire coefficients are `wire`,
    /// recovering the omitted coefficient from `claim`.
    fn next_claim(
        &self,
        round: usize,
        claim: E,
        wire: Vec<E>,
        challenge: E,
    ) -> Result<E, SumcheckError<E>> {
        Ok(match self.tau(round)? {
            None => CompressedPoly::new(wire).eval_from_hint(&claim, &challenge),
            Some(tau) => OmittedConstantPoly::new(wire).evaluate_from_claim(claim, tau, challenge),
        })
    }
}

fn mark<C: Channel>(channel: &mut C, site: Option<SiteId>) {
    if let Some(site) = site {
        channel.site(site);
    }
}

/// Proves one instance: the input claim bound as a derived claim at the claim
/// site; per round, the self-checked round polynomial's `degree` wire
/// coefficients exchanged as private messages at the round-body site, then
/// the context's challenge. Sends no output claims. Masked mode is the
/// context's channel; the verifier reads it with [`receive_masked_single`].
///
/// The kernel's round API is fused, as for the batched engine: round `i > 0`
/// receives challenge `i − 1`, and [`ProveRounds::finish_rounds`] the last.
/// For eq-factored instances the kernel returns the full `q`. The kernel and
/// the context fail with the caller's `Err`, which embeds sumcheck errors.
#[tracing::instrument(skip_all, name = "prove_single", fields(rounds = instance.rounds))]
pub fn prove_single<F, E, Err, C>(
    member: &mut dyn ProveRounds<E, Err>,
    instance: &SingleInstance<E>,
    context: &mut C,
) -> Result<SingleReduction<E>, Err>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + CanonicalDecode,
    Err: From<SumcheckError<E>>,
    C: ProverRoundContext<E, Error = Err>,
{
    instance.validate()?;
    if member.num_rounds() != instance.rounds {
        return Err(SumcheckError::RoundCountMismatch {
            expected: instance.rounds,
            got: member.num_rounds(),
        }
        .into());
    }

    let mut claim = instance.input_claim;
    let claim_site = context.site(instance.site(0, SumcheckRole::Claim));
    let channel = context.channel_mut();
    mark(channel, claim_site);
    channel
        .bind_derived::<F, E>(&claim)
        .map_err(Into::into)?;

    let mut challenges: Vec<E> = Vec::with_capacity(instance.rounds);
    for (round, round_id) in (0..instance.rounds).zip(0u32..) {
        let poly = member.prove_round(challenges.last().copied(), round, claim)?;
        let wire = instance.wire_coefficients(round, claim, &poly)?;
        let body_site = context.site(instance.site(round_id, SumcheckRole::RoundBody));
        let channel = context.channel_mut();
        mark(channel, body_site);
        let _: PadSlots = channel
            .exchange_private::<F, E>(&mut wire.clone())
            .map_err(Into::into)?;
        let challenge = context.challenge(instance.site(round_id, SumcheckRole::Challenge))?;
        claim = instance.next_claim(round, claim, wire, challenge)?;
        challenges.push(challenge);
    }
    if let Some(&last) = challenges.last() {
        member.finish_rounds(last)?;
    }

    Ok(SingleReduction {
        challenges,
        output_claim: claim,
    })
}

/// Replays one instance's rounds and returns the reduction before any oracle
/// check, so a caller can exchange late claims first. Receives exactly what
/// [`prove_single`] sends, through [`receive_masked_single`]; the omitted
/// coefficient of each round is defined by its check, so the check holds by
/// construction and the oracle check carries soundness.
#[tracing::instrument(skip_all, name = "verify_single", fields(rounds = instance.rounds))]
pub fn verify_single<F, E, Err, C>(
    instance: &SingleInstance<E>,
    context: &mut C,
) -> Result<SingleReduction<E>, Err>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + CanonicalDecode,
    Err: From<SumcheckError<E>>,
    C: VerifierRoundContext<E, Error = Err>,
{
    let rounds = receive_masked_single::<F, E, Err, C>(instance, context)?;
    let mut claim = instance.input_claim;
    let mut challenges = Vec::with_capacity(rounds.len());
    for (
        round,
        MaskedRound {
            wire, challenge, ..
        },
    ) in rounds.into_iter().enumerate()
    {
        claim = instance.next_claim(round, claim, wire, challenge)?;
        challenges.push(challenge);
    }
    Ok(SingleReduction {
        challenges,
        output_claim: claim,
    })
}

/// Reads one instance's rounds as [`prove_single`] sends them, recording
/// each round's wire values, pad slots and challenge without folding them
/// into a claim. Each challenge is drawn through the context, so per-round
/// grinding applies, unlike [`receive_masked_rounds`](crate::receive_masked_rounds),
/// which draws `challenge_small` directly.
///
/// Over a masked channel the wire values are m̂ = m + p, the input claim binds
/// nothing (so `instance.input_claim` is unused), and the round checks are
/// left to the caller's residual lowering (Akita `specs/whiteout-akita.md`
/// §4.5). Over the clear transcript the slots are empty and
/// [`verify_single`] folds the rounds.
///
/// # Errors
///
/// A shape error before any transcript operation; the channel's or the
/// context's error on a truncated proof, an exhausted pad schedule or a
/// rejected grinding nonce.
pub fn receive_masked_single<F, E, Err, C>(
    instance: &SingleInstance<E>,
    context: &mut C,
) -> Result<Vec<MaskedRound<E>>, Err>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + CanonicalDecode,
    Err: From<SumcheckError<E>>,
    C: VerifierRoundContext<E, Error = Err>,
{
    instance.validate()?;

    let claim_site = context.site(instance.site(0, SumcheckRole::Claim));
    let channel = context.channel_mut();
    mark(channel, claim_site);
    channel
        .bind_derived::<F, E>(&instance.input_claim)
        .map_err(Into::into)?;

    let mut rounds = Vec::with_capacity(instance.rounds);
    for round_id in (0u32..).take(instance.rounds) {
        let body_site = context.site(instance.site(round_id, SumcheckRole::RoundBody));
        let channel = context.channel_mut();
        mark(channel, body_site);
        let mut wire = vec![E::zero(); instance.degree];
        let slots = channel
            .exchange_private::<F, E>(&mut wire)
            .map_err(Into::into)?;
        let challenge = context.challenge(instance.site(round_id, SumcheckRole::Challenge))?;
        rounds.push(MaskedRound {
            wire,
            slots,
            challenge,
        });
    }
    Ok(rounds)
}
