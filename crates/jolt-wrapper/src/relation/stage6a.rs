//! Stage 6a: the bytecode read-RAF and booleanity address phases. The
//! bytecode member emits degree-2 polynomials over its `log_k_bytecode`
//! rounds; booleanity binds the last `log_k_chunk` rounds at degree 3, so its
//! address is the first committed chunk of the bytecode address.

use jolt_claims::protocols::jolt::geometry::booleanity::BooleanityDimensions;
use jolt_claims::protocols::jolt::relations::booleanity::BooleanityAddressPhase;
use jolt_claims::protocols::jolt::relations::bytecode::ReadRafAddressPhase;
use jolt_claims::protocols::jolt::{BooleanityChallenge, BytecodeReadRafChallenge};
use jolt_claims::SymbolicSumcheck;
use jolt_field::Fr;

use super::ctx::{Ctx, Lc};
use super::gadgets::reversed;
use super::lower::lower;
use super::public_io::PublicSlots;
use super::replay::SqueezeKind;
use super::stage5::Stage5;
use super::sumcheck::finish_batch;
use super::wiring::{absorb_member, run_batch, Layout, Wires};
use super::RelationError;
use crate::profile::WrapperProfile;

/// The bytecode address-phase rounds emit `eq · (linear)` polynomials.
const BYTECODE_ADDRESS_ROUND_DEGREE: usize = 2;

pub(crate) struct Stage6a {
    pub bytecode_address: Vec<Lc>,
    pub booleanity_address: Vec<Lc>,
    pub reference_address: Vec<Lc>,
    /// `[gamma, stage1_gamma, …, stage5_gamma]`.
    pub bytecode_gammas: Vec<Lc>,
}

pub(crate) fn walk(
    ctx: &mut Ctx,
    profile: &WrapperProfile,
    public: &PublicSlots,
    wires: &mut Wires,
    stage5: &Stage5,
) -> Result<Stage6a, RelationError> {
    let log_t = profile.log_t;
    let log_k_bytecode = profile.log_k_bytecode;
    let chunk_bits = profile.one_hot_config.committed_chunk_bits();
    if chunk_bits > log_k_bytecode {
        return Err(RelationError::Unsupported(
            "booleanity address phase longer than the bytecode address phase",
        ));
    }
    let bytecode_dimensions = stage5.formula.bytecode_read_raf;
    let booleanity_dimensions =
        BooleanityDimensions::new(stage5.formula.ra_layout, log_t, chunk_bits);
    let bytecode = ReadRafAddressPhase::new(bytecode_dimensions);
    let booleanity = BooleanityAddressPhase::new(booleanity_dimensions);

    ctx.section("stage6a/batch");
    let bytecode_gammas = ctx.squeeze_vector(SqueezeKind::Scalar, 6)?;
    let gamma_ids = [
        BytecodeReadRafChallenge::Gamma,
        BytecodeReadRafChallenge::Stage1Gamma,
        BytecodeReadRafChallenge::Stage2Gamma,
        BytecodeReadRafChallenge::Stage3Gamma,
        BytecodeReadRafChallenge::Stage4Gamma,
        BytecodeReadRafChallenge::Stage5Gamma,
    ];
    for (id, gamma) in gamma_ids.into_iter().zip(&bytecode_gammas) {
        wires.challenge(id, gamma.clone());
    }
    // The booleanity reference address is the reversed stage-5 instruction
    // address, truncated to (or padded by fresh challenges up to) the chunk.
    let mut reference_address = reversed(&stage5.instruction_address);
    if reference_address.len() >= chunk_bits {
        reference_address = reference_address[reference_address.len() - chunk_bits..].to_vec();
    } else {
        let missing = chunk_bits - reference_address.len();
        reference_address.extend(ctx.squeeze_vector(SqueezeKind::Challenge, missing)?);
    }
    let booleanity_gamma = ctx.squeeze(SqueezeKind::Challenge)?;
    wires.challenge(BooleanityChallenge::Gamma, booleanity_gamma);

    let inputs = [
        lower(ctx, &bytecode.input_expression::<Fr>(), &wires.sources)?,
        lower(ctx, &booleanity.input_expression::<Fr>(), &wires.sources)?,
    ];
    let layouts = [
        Layout::uniform(log_k_bytecode, BYTECODE_ADDRESS_ROUND_DEGREE, 0),
        Layout::suffix(chunk_bits, booleanity.degree(), log_k_bytecode),
    ];
    let (batch, point, final_claim) = run_batch(ctx, &inputs, &layouts)?;
    let bytecode_address = reversed(&point);
    let booleanity_address = reversed(layouts[1].slice(&point));

    absorb_member(ctx, wires, &bytecode, &[], &[], |_| {
        bytecode_address.clone()
    })?;
    absorb_member(ctx, wires, &booleanity, &[], &[], |_| {
        booleanity_address.clone()
    })?;

    ctx.section("stage6a/expected");
    let expected = [
        lower(ctx, &bytecode.output_expression::<Fr>(), &wires.sources)?,
        lower(ctx, &booleanity.output_expression::<Fr>(), &wires.sources)?,
    ];
    finish_batch(ctx, &batch, &expected, &final_claim);

    ctx.section("stage6a/public");
    PublicSlots::bind_points(
        ctx,
        &public.evaluation_points().bytecode_address,
        &bytecode_address,
    )?;
    PublicSlots::bind_points(
        ctx,
        &public.evaluation_points().bytecode_gammas,
        &bytecode_gammas,
    )?;

    Ok(Stage6a {
        bytecode_address,
        booleanity_address,
        reference_address,
        bytecode_gammas,
    })
}
