//! The single-instance sumcheck driver: one unbatched instance with a public
//! degree bound, proved by a [`ProveRounds`] kernel and replayed by the
//! verifier up to, not including, the oracle check.
//!
//! Two round formats share the driver. [`SingleFormat::Compressed`] sends the
//! round polynomial `s` without its linear coefficient and checks
//! `s(0) + s(1) = claim`. [`SingleFormat::EqFactored`] factors round `i` as
//! `eq(τ_i, X) · q(X)`, sends `q` without its constant coefficient, and checks
//! `(1 − τ_i) q(0) + τ_i q(1) = claim` ([`EqFactoredRound`]); its running claim
//! is normalized to `q(r_i)`, so the final claim is the summand `g(r)` itself
//! and the unnormalized claim is `Π eq(τ_i, r_i) · g(r)`.
//!
//! Both formats send exactly `degree` coefficients per round, so a round's
//! length is fixed by the public instance and absorbs with no count word. The
//! transcript sees, in order: the input claim as a derived claim, then per
//! round the wire coefficients as one private message, the round's proof of
//! work at the instance's degree, and the round challenge.

use jolt_algebra::Field;
#[cfg(feature = "prover")]
use jolt_poly::UnivariatePoly;
use jolt_poly::{CompressedPoly, OmittedConstantPoly, Point, LOW_TO_HIGH};
#[cfg(feature = "prover")]
use jolt_transcript::ProverTranscript;
use jolt_transcript::{Encoding, FieldTranscript, PrivateChannel};
use serde::{Deserialize, Serialize};

#[cfg(feature = "prover")]
use crate::domain::{BooleanHypercube, EqFactoredRound, SumcheckDomain};
use crate::error::SumcheckError;
#[cfg(feature = "prover")]
use crate::pow::grind_round;
use crate::pow::RoundNonces;
#[cfg(feature = "prover")]
use crate::prover::ProveRounds;

/// How each round of a single instance is factored and sent.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SingleFormat<E> {
    /// `s` sent without its linear coefficient.
    Compressed,
    /// `q` sent without its constant coefficient; round `i` uses
    /// `τ_i = equality_point[i]`.
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
}

/// A single-instance proof: each round's `degree` wire coefficients, and
/// one proof-of-work nonce per round when the round difficulty is positive.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(serialize = "E: Serialize", deserialize = "E: Deserialize<'de>"))]
pub struct SingleSumcheckProof<E> {
    pub rounds: Vec<Vec<E>>,
    pub pow_nonces: Vec<u64>,
}

/// The reduction a single instance proves or replays: the round challenges
/// `r` and the final running claim. The verifier gets it before any oracle
/// check; the caller checks `output_claim = g(r)` against the oracle. For
/// eq-factored instances the claim is normalized: `output_claim = g(r)`, and
/// the unnormalized claim is `Π eq(τ_i, r_i) · output_claim`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SingleReduction<E> {
    pub challenges: Vec<E>,
    pub output_claim: E,
}

impl<E: Field + Encoding> SingleInstance<E> {
    /// Checks the public shape: a nonzero degree bound, a representable
    /// `degree × rounds` proof size, and, for eq-factored instances, one
    /// equality coordinate per round and at least one round.
    fn validate(&self) -> Result<(), SumcheckError<E>> {
        if self.degree == 0 {
            return Err(SumcheckError::ZeroDegreeBound);
        }
        if self.degree.checked_mul(self.rounds).is_none() {
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
    #[cfg(feature = "prover")]
    fn wire_coefficients(
        &self,
        round: usize,
        claim: E,
        poly: &UnivariatePoly<E>,
    ) -> Result<Vec<E>, SumcheckError<E>> {
        // Kernels may sample a polynomial at more points than its degree
        // needs, leaving zero leading coefficients.
        let coefficients = poly.coefficients();
        let degree = coefficients
            .iter()
            .rposition(|coefficient| !coefficient.is_zero())
            .unwrap_or(0);
        if degree > self.degree {
            return Err(SumcheckError::DegreeBoundExceeded {
                got: degree,
                max: self.degree,
            });
        }
        let width = self.degree + 1;
        let mut padded: Vec<E> = coefficients.iter().copied().take(width).collect();
        padded.resize(width, E::zero());
        let padded = UnivariatePoly::new(padded);
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
        wire: &[E],
        challenge: E,
    ) -> Result<E, SumcheckError<E>> {
        Ok(match self.tau(round)? {
            None => CompressedPoly::new(wire.to_vec()).eval_from_hint(&claim, &challenge),
            Some(tau) => {
                OmittedConstantPoly::new(wire.to_vec()).evaluate_from_claim(claim, tau, challenge)
            }
        })
    }
}

/// Proves one instance: the input claim bound as a derived claim; per round,
/// the self-checked round polynomial's `degree` wire coefficients as one
/// private message, the round's proof of work, then the round challenge.
/// Sends no output claims.
///
/// `instance.input_claim` must be derivable from values already absorbed:
/// it is bound with [`PrivateChannel::bind_derived`], which a masked channel
/// does not absorb.
///
/// The kernel's round API is fused, as for the batched engine: round `i > 0`
/// receives challenge `i − 1`, and [`ProveRounds::finish_rounds`] the last.
/// For eq-factored instances the kernel returns the full `q`. The kernel
/// fails with the caller's `Err`, which embeds sumcheck errors.
#[cfg(feature = "prover")]
#[tracing::instrument(skip_all, name = "prove_single", fields(rounds = instance.rounds))]
pub fn prove_single<E, T, Err>(
    member: &mut dyn ProveRounds<E, Err>,
    instance: &SingleInstance<E>,
    transcript: &mut T,
) -> Result<(SingleSumcheckProof<E>, SingleReduction<E>), Err>
where
    E: Field + Encoding,
    T: FieldTranscript<E> + ProverTranscript,
    Err: From<SumcheckError<E>>,
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
    transcript.bind_derived(&claim);

    let mut proof = SingleSumcheckProof {
        rounds: Vec::with_capacity(instance.rounds),
        pow_nonces: Vec::new(),
    };
    let mut challenges: Vec<E> = Vec::with_capacity(instance.rounds);
    for round in 0..instance.rounds {
        let poly = member.prove_round(challenges.last().copied(), round, claim)?;
        let wire = instance.wire_coefficients(round, claim, &poly)?;
        transcript.append_private(&wire);
        grind_round(instance.degree, round, &mut proof.pow_nonces, transcript)?;
        let challenge = transcript.challenge().map_err(SumcheckError::from)?;
        claim = instance.next_claim(round, claim, &wire, challenge)?;
        challenges.push(challenge);
        proof.rounds.push(wire);
    }
    if let Some(&last) = challenges.last() {
        member.finish_rounds(last)?;
    }

    Ok((
        proof,
        SingleReduction {
            challenges,
            output_claim: claim,
        },
    ))
}

/// Replays one instance's rounds as [`prove_single`] absorbs them and
/// returns the reduction before any oracle check, so a caller can absorb
/// late claims first. The omitted coefficient of each round is defined by
/// its check, so the check holds by construction and the oracle check
/// carries soundness.
///
/// `instance.input_claim` must be derivable from values already absorbed:
/// it is bound with [`PrivateChannel::bind_derived`], which a masked channel
/// does not absorb.
///
/// # Errors
///
/// A shape error before any transcript operation; a round of the wrong
/// width or count; a rejected proof-of-work nonce or challenge draw.
#[tracing::instrument(skip_all, name = "verify_single", fields(rounds = instance.rounds))]
pub fn verify_single<E, T>(
    instance: &SingleInstance<E>,
    proof: &SingleSumcheckProof<E>,
    transcript: &mut T,
) -> Result<SingleReduction<E>, SumcheckError<E>>
where
    E: Field + Encoding,
    T: FieldTranscript<E>,
{
    instance.validate()?;
    if proof.rounds.len() != instance.rounds {
        return Err(SumcheckError::WrongNumberOfRounds {
            expected: instance.rounds,
            got: proof.rounds.len(),
        });
    }
    if let Some((round, wire)) = proof
        .rounds
        .iter()
        .enumerate()
        .find(|(_, wire)| wire.len() != instance.degree)
    {
        return Err(SumcheckError::WrongRoundWidth {
            round,
            expected: instance.degree,
            got: wire.len(),
        });
    }
    let pow = RoundNonces::new::<E, T>(instance.degree, instance.rounds, &proof.pow_nonces)?;

    let mut claim = instance.input_claim;
    transcript.bind_derived(&claim);

    let mut challenges = Vec::with_capacity(instance.rounds);
    for (round, wire) in proof.rounds.iter().enumerate() {
        transcript.append_private(wire);
        pow.check(round, transcript)?;
        let challenge = transcript.challenge()?;
        claim = instance.next_claim(round, claim, wire, challenge)?;
        challenges.push(challenge);
    }
    Ok(SingleReduction {
        challenges,
        output_claim: claim,
    })
}
