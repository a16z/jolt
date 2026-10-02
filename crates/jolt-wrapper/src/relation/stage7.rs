//! Stage 7: the Hamming-weight claim reduction over the `log_k_chunk`
//! address rounds; every RA polynomial's booleanity and virtualization claim
//! is folded into one opening at `rev(point) ‖ r_cycle`.

use jolt_claims::protocols::jolt::geometry::claim_reductions::hamming_weight::HammingWeightClaimReductionDimensions;
use jolt_claims::protocols::jolt::relations::claim_reductions::hamming_weight::ClaimReduction as HammingClaimReduction;
use jolt_claims::protocols::jolt::{
    HammingWeightClaimReductionChallenge, HammingWeightClaimReductionPublic,
};
use jolt_claims::SymbolicSumcheck;
use jolt_field::Fr;

use super::ctx::{Ctx, Lc};
use super::gadgets::{eq, reversed};
use super::lower::lower;
use super::replay::SqueezeKind;
use super::stage5::Stage5;
use super::stage6a::Stage6a;
use super::stage6b::Stage6b;
use super::sumcheck::finish_batch;
use super::wiring::{absorb_member, run_batch, Layout, Wires};
use super::RelationError;
use crate::profile::WrapperProfile;

pub(crate) struct Stage7 {
    /// `rev(sumcheck point) ‖ r_cycle`: the unified RA opening point.
    pub point: Vec<Lc>,
}

pub(crate) fn walk(
    ctx: &mut Ctx,
    profile: &WrapperProfile,
    wires: &mut Wires,
    stage5: &Stage5,
    stage6a: &Stage6a,
    stage6b: &Stage6b,
) -> Result<Stage7, RelationError> {
    let chunk_bits = profile.one_hot_config.committed_chunk_bits();
    let dimensions =
        HammingWeightClaimReductionDimensions::new(stage5.formula.ra_layout, chunk_bits);
    let hamming = HammingClaimReduction::new(dimensions);

    ctx.section("stage7/batch");
    let gamma = ctx.squeeze(SqueezeKind::Scalar)?;
    wires.challenge(HammingWeightClaimReductionChallenge::Gamma, gamma);
    let input = lower(ctx, &hamming.input_expression::<Fr>(), &wires.sources)?;
    let layout = Layout::uniform(chunk_bits, hamming.degree(), 0);
    let (batch, sumcheck_point, final_claim) = run_batch(ctx, &[input], &[layout])?;
    let rho = reversed(&sumcheck_point);
    let mut point = rho.clone();
    point.extend(stage6b.r_cycle.iter().cloned());
    absorb_member(ctx, wires, &hamming, &[], &[], |_| point.clone())?;

    ctx.section("stage7/expected");
    let eq_booleanity = eq(ctx, &rho, &stage6a.booleanity_address);
    wires.derived(
        HammingWeightClaimReductionPublic::EqBooleanity,
        eq_booleanity,
    );
    for (index, chunk) in stage6b.virtualization_chunks.iter().enumerate() {
        let eq_virtualization = eq(ctx, &rho, chunk);
        wires.derived(
            HammingWeightClaimReductionPublic::EqVirtualization(index),
            eq_virtualization,
        );
    }
    let expected = lower(ctx, &hamming.output_expression::<Fr>(), &wires.sources)?;
    finish_batch(ctx, &batch, &[expected], &final_claim);

    Ok(Stage7 { point })
}
