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

use jolt_field::{CanonicalDecode, CanonicalEncoding, ExtField, Field};
use jolt_poly::{CompressedPoly, OmittedConstantPoly, Point, UnivariatePoly, LOW_TO_HIGH};
use jolt_transcript::{Channel, ProverTranscript, SiteId, Sponge, VerifierTranscript};

use crate::domain::{BooleanHypercube, EqFactoredRound, SumcheckDomain};
use crate::error::SumcheckError;
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

/// The prover's round context: it owns the transcript, names sites, and draws
/// each round challenge, applying whatever grinding its caller schedules.
pub trait ProverRoundContext<E> {
    type Sponge: Sponge;
    type Error;

    fn transcript_mut(&mut self) -> &mut ProverTranscript<Self::Sponge>;

    /// The transcript site for `site`, or `None` to emit no site.
    fn site(&self, site: SumcheckSite) -> Option<SiteId>;

    /// Applies scheduled work and draws the challenge at a
    /// [`SumcheckRole::Challenge`] site.
    fn challenge(&mut self, site: SumcheckSite) -> Result<E, Self::Error>;
}

/// The verifier's mirror of [`ProverRoundContext`].
pub trait VerifierRoundContext<'proof, E> {
    type Sponge: Sponge;
    type Error;

    fn transcript_mut(&mut self) -> &mut VerifierTranscript<'proof, Self::Sponge>;

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

/// A proved single instance: the round challenges and the final (normalized,
/// for eq-factored) running claim.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProvedSingle<E> {
    pub challenges: Vec<E>,
    pub output_claim: E,
}

/// A replayed single instance, before any oracle check: the claim the caller
/// must check against the oracle (after applying eq factors, for
/// eq-factored), and the challenges it is checked at.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SingleReplay<E> {
    pub output_claim: E,
    pub challenges: Vec<E>,
}

impl<E: Field> SingleInstance<E> {
    fn validate(&self) -> Result<(), SumcheckError<E>> {
        if self.degree == 0 {
            return Err(SumcheckError::ZeroDegreeBound);
        }
        if let SingleFormat::EqFactored { equality_point } = &self.format {
            if equality_point.len() != self.rounds {
                return Err(SumcheckError::EqualityPointLengthMismatch {
                    expected: self.rounds,
                    got: equality_point.len(),
                });
            }
        }
        Ok(())
    }

    fn site(&self, round: usize, role: SumcheckRole) -> Result<SumcheckSite, SumcheckError<E>> {
        Ok(SumcheckSite {
            invocation: self.invocation,
            round: u32::try_from(round).map_err(|_| SumcheckError::RoundIndexOverflow { round })?,
            role,
        })
    }

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

/// Proves one instance: the input claim absorbed publicly at the claim site;
/// per round, the self-checked round polynomial's `degree` wire coefficients
/// sent at the round-body site, then the context's challenge. Sends no output
/// claims.
///
/// The kernel's round API is fused, as for the batched engine: round `i > 0`
/// receives challenge `i − 1`, and [`ProveRounds::finish_rounds`] the last.
/// For eq-factored instances the kernel returns the full `q`.
#[tracing::instrument(skip_all, name = "prove_single", fields(rounds = instance.rounds))]
pub fn prove_single<F, E, K, C>(
    member: &mut dyn ProveRounds<E, K>,
    instance: &SingleInstance<E>,
    context: &mut C,
) -> Result<ProvedSingle<E>, SumcheckError<E, K>>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + CanonicalDecode,
    C: ProverRoundContext<E, Error = K>,
{
    instance.validate().map_err(SumcheckError::widen)?;
    if member.num_rounds() != instance.rounds {
        return Err(SumcheckError::RoundCountMismatch {
            expected: instance.rounds,
            got: member.num_rounds(),
        });
    }

    let mut claim = instance.input_claim;
    let claim_site = context.site(
        instance
            .site(0, SumcheckRole::Claim)
            .map_err(SumcheckError::widen)?,
    );
    let transcript = context.transcript_mut();
    mark(transcript, claim_site);
    transcript.public(&claim);

    let mut challenges: Vec<E> = Vec::with_capacity(instance.rounds);
    for round in 0..instance.rounds {
        let poly = member.prove_round(challenges.last().copied(), round, claim)?;
        let wire = instance
            .wire_coefficients(round, claim, &poly)
            .map_err(SumcheckError::widen)?;
        let body_site = context.site(
            instance
                .site(round, SumcheckRole::RoundBody)
                .map_err(SumcheckError::widen)?,
        );
        let transcript = context.transcript_mut();
        mark(transcript, body_site);
        transcript.send_all(&wire);
        let challenge = context
            .challenge(
                instance
                    .site(round, SumcheckRole::Challenge)
                    .map_err(SumcheckError::widen)?,
            )
            .map_err(SumcheckError::Kernel)?;
        claim = instance
            .next_claim(round, claim, wire, challenge)
            .map_err(SumcheckError::widen)?;
        challenges.push(challenge);
    }
    if let Some(&last) = challenges.last() {
        member.finish_rounds(last)?;
    }

    Ok(ProvedSingle {
        challenges,
        output_claim: claim,
    })
}

/// Replays one instance's rounds and returns the output claim and the
/// challenges before any oracle check, so a caller can exchange late claims
/// first. Receives exactly what [`prove_single`] sends; the omitted
/// coefficient of each round is defined by its check, so the check holds by
/// construction and the oracle check carries soundness.
#[tracing::instrument(skip_all, name = "verify_single", fields(rounds = instance.rounds))]
pub fn verify_single<'proof, F, E, K, C>(
    instance: &SingleInstance<E>,
    context: &mut C,
) -> Result<SingleReplay<E>, SumcheckError<E, K>>
where
    F: Field + CanonicalEncoding,
    E: ExtField<F> + CanonicalDecode,
    C: VerifierRoundContext<'proof, E, Error = K>,
{
    instance.validate().map_err(SumcheckError::widen)?;

    let mut claim = instance.input_claim;
    let claim_site = context.site(
        instance
            .site(0, SumcheckRole::Claim)
            .map_err(SumcheckError::widen)?,
    );
    let transcript = context.transcript_mut();
    mark(transcript, claim_site);
    transcript.public(&claim);

    let mut challenges = Vec::with_capacity(instance.rounds);
    for round in 0..instance.rounds {
        let body_site = context.site(
            instance
                .site(round, SumcheckRole::RoundBody)
                .map_err(SumcheckError::widen)?,
        );
        let transcript = context.transcript_mut();
        mark(transcript, body_site);
        let wire = transcript.receive_n::<E>(instance.degree)?;
        let challenge = context
            .challenge(
                instance
                    .site(round, SumcheckRole::Challenge)
                    .map_err(SumcheckError::widen)?,
            )
            .map_err(SumcheckError::Kernel)?;
        claim = instance
            .next_claim(round, claim, wire, challenge)
            .map_err(SumcheckError::widen)?;
        challenges.push(challenge);
    }

    Ok(SingleReplay {
        output_claim: claim,
        challenges,
    })
}
