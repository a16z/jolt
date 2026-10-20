//! The shared uni-skip first-round verification step.
//!
//! Stages 1 and 2 each open with a univariate-skip round — a genuinely
//! different round type from the batched remainder sumchecks (separate wire
//! proof, degree-bounded single round over a centered integer domain) — before
//! their generated batch drivers run. The two stages differ only in their
//! degree/domain constants, error attribution, and how the input claim is
//! produced (stage 1: the constant zero; stage 2: the `ProductUniskip`
//! relation's fold of the stage-1 openings), so the verification core is
//! shared here.

use jolt_claims::protocols::composed::geometry::{
    SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE, SPARTAN_PRODUCT_UNISKIP_FIRST_ROUND_DEGREE,
};
use jolt_claims::protocols::composed::r1cs::{
    SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE, SPARTAN_OUTER_UNISKIP_FIRST_ROUND_DEGREE,
};
use jolt_claims::protocols::jolt::geometry::spartan::{
    outer_uniskip_opening, product_uniskip_opening,
};
use jolt_claims::protocols::jolt::{JoltOpeningId, JoltRelationId};
use jolt_field::JoltField;
use jolt_sumcheck::{
    receive_masked_rounds, receive_private_value, CenteredIntegerDomain, MaskedRound, RoundWire,
    SumcheckClaim, SumcheckError, SumcheckStatement, SumcheckVerifier,
};
use jolt_transcript::{Channel, PrivateChannel, Sponge, TranscriptError, VerifierTranscript};

use crate::stages::relations::{receive_masked_value, MaskedClaims};
use crate::VerifierError;

const UNISKIP_ROUNDS: usize = 1;

/// The per-stage uni-skip shape: the fixed first-round degree bound and
/// centered-domain size the wire format prescribes, plus error attribution.
/// Jolt has exactly two uni-skip rounds — [`spartan_outer`](Self::spartan_outer)
/// (stage 1) and [`spartan_product`](Self::spartan_product) (stage 2) — so the
/// two constructors are the only instances.
pub struct UniskipParams {
    stage: JoltRelationId,
    /// The round's single output opening.
    output_opening: JoltOpeningId,
    stage_number: usize,
    degree: usize,
    domain_size: usize,
}

impl UniskipParams {
    /// The stage-1 Spartan outer uni-skip shape.
    pub fn spartan_outer() -> Self {
        Self {
            stage: JoltRelationId::SpartanOuter,
            output_opening: outer_uniskip_opening(),
            stage_number: 1,
            degree: SPARTAN_OUTER_UNISKIP_FIRST_ROUND_DEGREE,
            domain_size: SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE,
        }
    }

    /// The stage-2 Spartan product-virtualization uni-skip shape.
    pub fn spartan_product() -> Self {
        Self {
            stage: JoltRelationId::SpartanProductVirtualization,
            output_opening: product_uniskip_opening(),
            stage_number: 2,
            degree: SPARTAN_PRODUCT_UNISKIP_FIRST_ROUND_DEGREE,
            domain_size: SPARTAN_PRODUCT_UNISKIP_DOMAIN_SIZE,
        }
    }

    /// The stage's uni-skip first-round degree bound (the transmitted
    /// polynomial's maximum degree the wire format admits).
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// The stage's centered integer domain size — the node count the
    /// first-round claim sums over.
    pub fn domain_size(&self) -> usize {
        self.domain_size
    }

    fn sumcheck_failed(&self, reason: impl ToString) -> VerifierError {
        VerifierError::StageClaimSumcheckFailed {
            stage: format!("{:?}", self.stage),
            reason: reason.to_string(),
        }
    }
}

/// The stage-1 outer tau — the `log_t + 2` challenges drawn before the outer
/// uni-skip round (the Spartan outer relation's cycle variables plus the two
/// uni-skip-collapsed row variables). Both fronts call this, so the draw is
/// single-sourced.
pub fn draw_spartan_outer_tau<F, C>(transcript: &mut C, log_t: usize) -> Vec<F>
where
    F: JoltField,
    C: Channel,
{
    #[expect(
        clippy::arithmetic_side_effects,
        reason = "log_t is an ilog2 result (< 64); log_t + 2 cannot overflow usize"
    )]
    transcript.challenges_small(log_t + 2)
}

/// The stage-2 product tau_high, drawn before the product uni-skip round from
/// the small challenge set. Both fronts call this, so the draw is
/// single-sourced.
pub fn draw_spartan_product_tau_high<F, C>(transcript: &mut C) -> F
where
    F: JoltField,
    C: Channel,
{
    transcript.challenge_small()
}

/// The masked uni-skip step's outputs: the masked round and output claim
/// (carried downstream for the residual relation), plus the reduction
/// challenge.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UniskipMasked<F> {
    pub rounds: Vec<MaskedRound<F>>,
    pub output_claims: MaskedClaims<F>,
    pub challenge: F,
}

/// The clear uni-skip step's outputs: the single reduction challenge and the
/// received output claim.
pub struct UniskipClear<F> {
    pub challenge: F,
    pub output_claim: F,
}

/// Verify a clear-mode uni-skip round against its input claim.
///
/// Protocol contract (the prover's uni-skip round must mirror it): the round
/// polynomial is read as a one-round sumcheck of `params`' degree over
/// `params`' centered integer domain, then the output claim is received and
/// hard-checked against the reduced value — BEFORE any post-uni-skip draw (the
/// remainder batch's coefficient draw in particular). Errors are attributed to
/// `params`' stage.
pub fn verify_clear<F, H>(
    params: &UniskipParams,
    input_claim: F,
    transcript: &mut VerifierTranscript<'_, H>,
) -> Result<UniskipClear<F>, VerifierError>
where
    F: JoltField,
    H: Sponge,
{
    let reduction = SumcheckVerifier::verify(
        &SumcheckClaim::new(UNISKIP_ROUNDS, params.degree, input_claim),
        CenteredIntegerDomain::new(params.domain_size),
        transcript,
    )
    .map_err(|error| params.sumcheck_failed(error))?;
    let (output_claim, _pads) = receive_private_value(transcript)?;
    if reduction.value != output_claim {
        return Err(VerifierError::StageClaimOutputMismatch {
            stage: params.stage_number,
        });
    }

    let [challenge] = reduction.point.as_slice() else {
        return Err(params.sumcheck_failed("uni-skip proof did not reduce to one challenge"));
    };
    Ok(UniskipClear {
        challenge: *challenge,
        output_claim,
    })
}

/// Verify a masked uni-skip round: read the masked round at full width, its
/// challenge, then the masked output claim. Nothing hidden is checked; the
/// residual relation states the round-sum check and the output equality.
pub fn verify_masked<F, C>(
    params: &UniskipParams,
    channel: &mut C,
) -> Result<UniskipMasked<F>, VerifierError>
where
    F: JoltField,
    C: Channel + PrivateChannel,
    C::Error: From<TranscriptError> + From<SumcheckError<F>>,
    VerifierError: From<C::Error>,
{
    let rounds = receive_masked_rounds::<F, C>(
        SumcheckStatement::new(UNISKIP_ROUNDS, params.degree),
        RoundWire::Full,
        channel,
    )?;
    let [round] = rounds.as_slice() else {
        return Err(params.sumcheck_failed("uni-skip proof did not reduce to one challenge"));
    };
    let challenge = round.challenge;
    let output_claims = MaskedClaims {
        received: vec![(params.output_opening.into(), receive_masked_value(channel)?)],
        ..MaskedClaims::default()
    };
    Ok(UniskipMasked {
        rounds,
        output_claims,
        challenge,
    })
}
