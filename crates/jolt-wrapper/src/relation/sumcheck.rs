//! Sumcheck verification with variable challenges: the batched head, the
//! compressed Horner rounds and the uni-skip first round.

use jolt_field::{Fr, One, Ring, Zero};
use jolt_poly::lagrange::centered_power_sums;
use jolt_sumcheck::{
    OPENING_CLAIM_TRANSCRIPT_LABEL, SUMCHECK_CLAIM_TRANSCRIPT_LABEL,
    SUMCHECK_ROUND_TRANSCRIPT_LABEL, UNISKIP_ROUND_TRANSCRIPT_LABEL,
};

use super::ctx::{lc_const, Accum, Ctx, Lc};
use super::gadgets::horner;
use super::replay::SqueezeKind;
use super::RelationError;

pub(crate) struct Member {
    pub input_claim: Lc,
    pub rounds: usize,
}

pub(crate) struct Batch {
    pub coefficients: Vec<Lc>,
    pub claimed_sum: Lc,
}

/// Absorbs every member's input claim, draws one batching coefficient per
/// member and forms `Σ coeff · 2^(max − rounds) · claim`.
pub(crate) fn begin_batch(ctx: &mut Ctx, members: &[Member]) -> Result<Batch, RelationError> {
    for member in members {
        ctx.absorb_label(SUMCHECK_CLAIM_TRANSCRIPT_LABEL)?;
        ctx.absorb_value(&member.input_claim)?;
    }
    let coefficients = ctx.squeeze_vector(SqueezeKind::Scalar, members.len())?;
    let max_rounds = members
        .iter()
        .map(|member| member.rounds)
        .max()
        .unwrap_or(0);
    let mut claimed_sum = Accum::default();
    for (member, coefficient) in members.iter().zip(&coefficients) {
        let scaled = member
            .input_claim
            .clone()
            .scale(Fr::pow2(max_rounds - member.rounds));
        let term = ctx.mul(coefficient, &scaled);
        claimed_sum.add(&term, Fr::one());
    }
    Ok(Batch {
        coefficients,
        claimed_sum: claimed_sum.finish(),
    })
}

/// The compressed Boolean-hypercube rounds: per round `degrees[r]` prover
/// coefficients (linear term omitted), one challenge, one Horner chain. The
/// prover trims each batched round polynomial to the highest degree any
/// member active in that round emits, so the profile is per round.
pub(crate) fn compressed_rounds(
    ctx: &mut Ctx,
    claimed_sum: &Lc,
    degrees: &[usize],
) -> Result<(Vec<Lc>, Lc), RelationError> {
    let mut claim = claimed_sum.clone();
    let mut challenges = Vec::with_capacity(degrees.len());
    for &degree in degrees {
        ctx.absorb_label_count(SUMCHECK_ROUND_TRANSCRIPT_LABEL, degree)?;
        let c0 = ctx.proof_fr()?;
        let mut high = Vec::with_capacity(degree - 1);
        for _ in 1..degree {
            high.push(ctx.proof_fr()?);
        }
        let r = ctx.squeeze(SqueezeKind::Challenge)?;
        // c1 = claim − 2·c0 − Σ_{k≥2} c_k recovers the omitted linear term.
        let mut linear = Accum::default();
        linear.add(&claim, Fr::one());
        linear.add(&c0, -Fr::from_u64(2));
        for coefficient in &high {
            linear.add(coefficient, -Fr::one());
        }
        let mut coefficients = vec![c0, linear.finish()];
        coefficients.extend(high);
        claim = horner(ctx, &coefficients, &r);
        challenges.push(r);
    }
    Ok((challenges, claim))
}

/// `Σ coeff_i · expected_i == final_claim`.
pub(crate) fn finish_batch(ctx: &mut Ctx, batch: &Batch, expected: &[Lc], final_claim: &Lc) {
    let mut fold = Accum::default();
    for (coefficient, expected) in batch.coefficients.iter().zip(expected) {
        let term = ctx.mul(coefficient, expected);
        fold.add(&term, Fr::one());
    }
    let fold = fold.finish();
    ctx.assert_eq(&fold, final_claim);
}

/// One univariate-skip round over the centered integer domain: the full
/// `degree + 1` coefficients are absorbed, their power-sum combination must
/// equal the input claim, the polynomial is evaluated at the challenge and
/// that output claim is absorbed under `opening_claim`.
pub(crate) fn uniskip(
    ctx: &mut Ctx,
    input_claim: &Lc,
    degree: usize,
    domain_size: usize,
) -> Result<(Lc, Lc), RelationError> {
    ctx.absorb_label_count(UNISKIP_ROUND_TRANSCRIPT_LABEL, degree + 1)?;
    let mut coefficients = Vec::with_capacity(degree + 1);
    for _ in 0..=degree {
        coefficients.push(ctx.proof_fr()?);
    }
    let power_sums = centered_power_sums(domain_size, degree + 1)
        .map_err(|error| RelationError::Geometry(format!("{error:?}")))?;
    let mut round_sum = Accum::default();
    for (coefficient, power_sum) in coefficients.iter().zip(power_sums) {
        round_sum.add(coefficient, Fr::from_i128(power_sum));
    }
    let round_sum = round_sum.finish();
    ctx.assert_eq(&round_sum, input_claim);
    let challenge = ctx.squeeze(SqueezeKind::Challenge)?;
    let output = horner(ctx, &coefficients, &challenge);
    ctx.absorb_label(OPENING_CLAIM_TRANSCRIPT_LABEL)?;
    let output = ctx.absorb_computed(&output)?;
    Ok((challenge, super::ctx::lc_var(output)))
}

/// Absorbs the produced opening claims (prover wires) under `opening_claim`.
pub(crate) fn absorb_openings(ctx: &mut Ctx, count: usize) -> Result<Vec<Lc>, RelationError> {
    (0..count)
        .map(|_| {
            ctx.absorb_label(OPENING_CLAIM_TRANSCRIPT_LABEL)?;
            ctx.proof_fr()
        })
        .collect()
}

pub(crate) fn zero() -> Lc {
    lc_const(Fr::zero())
}
