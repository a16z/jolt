//! Stage 3: {Spartan shift, instruction-input virtualization, registers claim
//! reduction}, all `log_t` rounds at the same point.

use jolt_claims::protocols::jolt::geometry::bytecode::read_raf_consistency_openings;
use jolt_claims::protocols::jolt::geometry::claim_reductions::registers::{
    rs1_value_reduced, rs2_value_reduced,
};
use jolt_claims::protocols::jolt::geometry::dimensions::TraceDimensions;
use jolt_claims::protocols::jolt::geometry::instruction::{rs1_value, rs2_value};
use jolt_claims::protocols::jolt::relations::claim_reductions::registers::ClaimReduction as RegistersClaimReduction;
use jolt_claims::protocols::jolt::relations::instruction::InputVirtualization;
use jolt_claims::protocols::jolt::relations::spartan::Shift;
use jolt_claims::protocols::jolt::{
    InstructionInputChallenge, InstructionInputPublic, RegistersClaimReductionChallenge,
    RegistersClaimReductionPublic, SpartanShiftChallenge, SpartanShiftPublic,
};
use jolt_claims::SymbolicSumcheck;
use jolt_field::Fr;

use super::ctx::{Ctx, Lc};
use super::gadgets::{eq, eq_plus_one, reversed};
use super::lower::lower;
use super::replay::SqueezeKind;
use super::stage2::Stage2;
use super::sumcheck::finish_batch;
use super::wiring::{absorb_member, run_batch, Layout, Wires};
use super::RelationError;
use crate::profile::WrapperProfile;

pub(crate) struct Stage3 {
    /// The shared opening point of all three members.
    pub point: Vec<Lc>,
}

pub(crate) fn walk(
    ctx: &mut Ctx,
    profile: &WrapperProfile,
    wires: &mut Wires,
    stage2: &Stage2,
) -> Result<Stage3, RelationError> {
    let log_t = profile.log_t;
    let trace = TraceDimensions::new(log_t);
    let shift = Shift::new(trace);
    let instruction_input = InputVirtualization::new(trace);
    let registers_reduction = RegistersClaimReduction::new(trace);

    ctx.section("stage3/batch");
    let shift_gamma = ctx.squeeze(SqueezeKind::Scalar)?;
    wires.challenge(SpartanShiftChallenge::Gamma, shift_gamma);
    let input_gamma = ctx.squeeze(SqueezeKind::Scalar)?;
    wires.challenge(InstructionInputChallenge::Gamma, input_gamma);
    let registers_gamma = ctx.squeeze(SqueezeKind::Scalar)?;
    wires.challenge(RegistersClaimReductionChallenge::Gamma, registers_gamma);

    let inputs = [
        lower(ctx, &shift.input_expression::<Fr>(), &wires.sources)?,
        lower(
            ctx,
            &instruction_input.input_expression::<Fr>(),
            &wires.sources,
        )?,
        lower(
            ctx,
            &registers_reduction.input_expression::<Fr>(),
            &wires.sources,
        )?,
    ];
    let layouts = [
        Layout::uniform(log_t, shift.degree(), 0),
        Layout::uniform(log_t, instruction_input.degree(), 0),
        Layout::uniform(log_t, registers_reduction.degree(), 0),
    ];
    let (batch, sumcheck_point, final_claim) = run_batch(ctx, &inputs, &layouts)?;
    let point = reversed(&sumcheck_point);

    absorb_member(ctx, wires, &shift, &[], &[], |_| point.clone())?;
    let [(shift_unexpanded_pc, instruction_unexpanded_pc)] = read_raf_consistency_openings();
    absorb_member(
        ctx,
        wires,
        &instruction_input,
        &[],
        &[(instruction_unexpanded_pc, shift_unexpanded_pc)],
        |_| point.clone(),
    )?;
    absorb_member(
        ctx,
        wires,
        &registers_reduction,
        &[],
        &[
            (rs1_value_reduced(), rs1_value()),
            (rs2_value_reduced(), rs2_value()),
        ],
        |_| point.clone(),
    )?;

    ctx.section("stage3/expected");
    let eq_plus_one_outer = eq_plus_one(ctx, &stage2.tau_low, &point);
    wires.derived(SpartanShiftPublic::EqPlusOneOuter, eq_plus_one_outer);
    let eq_plus_one_product = eq_plus_one(ctx, &stage2.product_point, &point);
    wires.derived(SpartanShiftPublic::EqPlusOneProduct, eq_plus_one_product);
    let eq_product = eq(ctx, &point, &stage2.product_point);
    wires.derived(InstructionInputPublic::EqProduct, eq_product);
    let eq_spartan = eq(ctx, &point, &stage2.tau_low);
    wires.derived(RegistersClaimReductionPublic::EqSpartan, eq_spartan);

    let expected = [
        lower(ctx, &shift.output_expression::<Fr>(), &wires.sources)?,
        lower(
            ctx,
            &instruction_input.output_expression::<Fr>(),
            &wires.sources,
        )?,
        lower(
            ctx,
            &registers_reduction.output_expression::<Fr>(),
            &wires.sources,
        )?,
    ];
    finish_batch(ctx, &batch, &expected, &final_claim);

    Ok(Stage3 { point })
}
