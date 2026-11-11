//! The single-instance sumcheck driver: one unbatched instance with a public
//! degree bound, proved by a [`ProveRounds`] kernel and replayed by the
//! verifier up to, not including, the oracle check.
//!
//! Two round formats share the driver. [`SingleFormat::Compressed`] sends the
//! round polynomial `s` without its linear coefficient and checks
//! `s(0) + s(1) = claim`. [`SingleFormat::EqFactored`] factors round `i` as
//! `eq(τ_i, X) · q(X)`, sends `q` without its constant coefficient, and checks
//! `(1 − τ_i) q(0) + τ_i q(1) = claim` ([`crate::EqFactoredRound`]); its running claim
//! is normalized to `q(r_i)`, so the final claim is the summand `g(r)` itself
//! and the unnormalized claim is `Π eq(τ_i, r_i) · g(r)`.
//!
//! Both formats send exactly `degree` coefficients per round, so a round's
//! length is fixed by the public instance and absorbs with no count word. The
//! transcript sees, in order: the input claim as a derived claim, then per
//! round the wire coefficients as one private message, the round's proof of
//! work at the instance's degree, and the round challenge.

use jolt_field::Field;
use jolt_poly::{eval_round_wire, Point, WireFormat, LOW_TO_HIGH};
#[cfg(feature = "prover")]
use jolt_transcript::ProverTranscript;
use jolt_transcript::{Encoding, FieldTranscript};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;

use crate::error::SumcheckError;
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

mod channels;
#[cfg(feature = "prover")]
mod prover;
#[cfg(feature = "prover")]
mod rounds;
mod shape;
pub use channels::{BindScalarClaim, ClearVerifierRounds, RoundSpec, VerifierRoundChannel};
#[cfg(feature = "prover")]
pub use prover::{ClearProverRounds, ProverRoundChannel};
#[cfg(feature = "prover")]
pub use rounds::{FullRounds, ProveSingleRounds, ProverRound};
pub use shape::{SumcheckShape, SumcheckShapeError};

impl<E: Field> SingleInstance<E> {
    pub fn shape(&self) -> Result<SumcheckShape, SumcheckError<E>> {
        let shape = SumcheckShape::new(self.rounds, self.degree)?;
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
        Ok(shape)
    }

    #[cfg(feature = "prover")]
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

    // Both drivers validate shape before entering their bounded round loops.
    #[expect(
        clippy::indexing_slicing,
        reason = "shape validates the equality-point length and drivers bound round by that shape"
    )]
    fn next_claim(&self, round: usize, claim: E, wire: &[E], challenge: E) -> E {
        let format = match &self.format {
            SingleFormat::Compressed => WireFormat::Compressed,
            SingleFormat::EqFactored { equality_point } => WireFormat::EqFactored {
                tau: equality_point.as_slice()[round],
            },
        };
        eval_round_wire(format, wire, claim, challenge)
    }
}

/// Drives a kernel through a privacy-aware transport. Channel errors and kernel
/// errors pass through unchanged; the channel classifies driver errors.
#[cfg(feature = "prover")]
pub fn prove_single_with<E, C, K>(
    kernel: &mut K,
    instance: &SingleInstance<E>,
    channel: &mut C,
) -> Result<SingleReduction<E>, C::Error>
where
    E: Field,
    C: ProverRoundChannel<E>,
    K: ProveSingleRounds<E, C::Error> + ?Sized,
{
    let shape = instance
        .shape()
        .map_err(|e| channel.map_sumcheck_error(e))?;
    if kernel.num_rounds() != shape.num_rounds() {
        return Err(
            channel.map_sumcheck_error(SumcheckError::RoundCountMismatch {
                expected: shape.num_rounds(),
                got: kernel.num_rounds(),
            }),
        );
    }
    channel.bind_input_claim(&instance.input_claim)?;
    let mut claim = instance.input_claim;
    let mut challenges = Vec::with_capacity(shape.num_rounds());
    for round in 0..shape.num_rounds() {
        let output = kernel.prove_round(challenges.last().copied(), round, claim)?;
        let mut wire = instance
            .wire_coefficients(round, claim, output)
            .map_err(|e| channel.map_sumcheck_error(e))?;
        channel.send_round(round, &mut wire)?;
        let challenge = channel.round_challenge(round, shape.degree_bound())?;
        claim = instance.next_claim(round, claim, &wire, challenge);
        channel.finish_round(wire);
        challenges.push(challenge);
    }
    if let Some(&last) = challenges.last() {
        kernel.finish_rounds(last)?;
    }
    Ok(SingleReduction {
        challenges,
        output_claim: claim,
    })
}

/// Streams each wire directly to the visitor. The channel may lend a reused
/// buffer or transfer ownership; the driver never collects round messages.
pub fn visit_single_rounds<E, C>(
    shape: SumcheckShape,
    channel: &mut C,
    mut visit: impl FnMut(usize, Cow<'_, [E]>, C::Slots, E) -> Result<(), C::Error>,
) -> Result<(), C::Error>
where
    E: Field,
    C: VerifierRoundChannel<E>,
{
    channel.validate_shape(shape)?;
    let spec = RoundSpec {
        width: shape.degree_bound(),
        degree: shape.degree_bound(),
    };
    for round in 0..shape.num_rounds() {
        channel
            .verify_round(round, spec, |wire, slots, challenge| {
                if wire.len() != shape.degree_bound() {
                    return Ok(Err(SumcheckError::WrongRoundWidth {
                        round,
                        expected: shape.degree_bound(),
                        got: wire.len(),
                    }));
                }
                visit(round, wire, slots, challenge).map(Ok)
            })?
            .map_err(|error| channel.map_sumcheck_error(error))?;
    }
    Ok(())
}

/// Folds a scalar claim over streamed clear rounds, before the caller's oracle check.
pub fn verify_single_with<E, C>(
    instance: &SingleInstance<E>,
    channel: &mut C,
) -> Result<SingleReduction<E>, C::Error>
where
    E: Field,
    C: BindScalarClaim<E>,
{
    let shape = instance
        .shape()
        .map_err(|e| channel.map_sumcheck_error(e))?;
    channel.validate_shape(shape)?;
    channel.bind_input_claim(&instance.input_claim)?;
    let mut claim = instance.input_claim;
    let mut challenges = Vec::with_capacity(shape.num_rounds());
    visit_single_rounds(shape, channel, |round, wire, (), challenge| {
        claim = instance.next_claim(round, claim, &wire, challenge);
        challenges.push(challenge);
        Ok(())
    })?;
    Ok(SingleReduction {
        challenges,
        output_claim: claim,
    })
}

/// Proves one instance: the input claim bound as a derived claim; per round,
/// the self-checked round polynomial's `degree` wire coefficients as one
/// private message, the round's proof of work, then the round challenge.
/// Sends no output claims.
///
/// `instance.input_claim` must be derivable from values already absorbed:
/// it is bound with [`jolt_transcript::PrivateChannel::bind_derived`], which
/// a clear transcript absorbs as a public value.
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
    let mut channel = ClearProverRounds::new(instance.shape()?, transcript);
    let reduction = prove_single_with(&mut FullRounds::new(member), instance, &mut channel)?;
    Ok((channel.into_proof(), reduction))
}

/// Replays one instance's rounds as [`prove_single`] absorbs them and
/// returns the reduction before any oracle check, so a caller can absorb
/// late claims first. The omitted coefficient of each round is defined by
/// its check, so the check holds by construction and the oracle check
/// carries soundness.
///
/// `instance.input_claim` must be derivable from values already absorbed:
/// it is bound with [`jolt_transcript::PrivateChannel::bind_derived`], which
/// a clear transcript absorbs as a public value.
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
    let mut channel = ClearVerifierRounds::new(instance.shape()?, proof, transcript)?;
    verify_single_with(instance, &mut channel)
}
