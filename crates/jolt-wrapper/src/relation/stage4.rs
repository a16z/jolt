//! Stage 4: {registers read-write checking, RAM value check}; `InitEval` is
//! an outsourced public input.

use jolt_claims::protocols::jolt::geometry::dimensions::{TraceDimensions, REGISTER_ADDRESS_BITS};
use jolt_claims::protocols::jolt::relations::ram::{RamValCheck, RamValCheckShape};
use jolt_claims::protocols::jolt::relations::registers::ReadWriteChecking as RegistersReadWriteChecking;
use jolt_claims::protocols::jolt::{
    RamValCheckChallenge, RamValCheckPublic, RegistersReadWriteChallenge, RegistersReadWritePublic,
};
use jolt_claims::SymbolicSumcheck;
use jolt_field::Fr;
use jolt_transcript::LabelWithCount;

use super::ctx::{encode, Ctx, Lc};
use super::gadgets::{eq, lt, read_write_opening_point, reversed};
use super::lower::lower;
use super::public_io::{self, PublicSlots};
use super::replay::SqueezeKind;
use super::stage2::{values, Stage2};
use super::stage3::Stage3;
use super::sumcheck::finish_batch;
use super::wiring::{absorb_member, run_batch, Layout, Wires};
use super::{Native, RelationError};
use crate::profile::WrapperProfile;

pub(crate) struct Stage4 {
    pub register_address: Vec<Lc>,
    pub register_cycle: Vec<Lc>,
    /// The RAM value-check cycle point.
    pub ram_val_check_cycle: Vec<Lc>,
}

pub(crate) fn walk(
    ctx: &mut Ctx,
    profile: &WrapperProfile,
    native: Option<&Native<'_>>,
    public: &PublicSlots,
    wires: &mut Wires,
    stage2: &Stage2,
    stage3: &Stage3,
) -> Result<Stage4, RelationError> {
    let log_t = profile.log_t;
    let register_dimensions = profile
        .rw_config
        .register_dimensions(log_t, REGISTER_ADDRESS_BITS);
    let registers = RegistersReadWriteChecking::new(register_dimensions);
    let ram_val_check = RamValCheck::new(RamValCheckShape {
        dimensions: TraceDimensions::new(log_t),
        contributions: Vec::new(),
    });

    ctx.section("stage4/batch");
    let registers_gamma = ctx.squeeze(SqueezeKind::Scalar)?;
    wires.challenge(RegistersReadWriteChallenge::Gamma, registers_gamma);
    // The value-check gamma draw is domain-separated from the register draw.
    ctx.absorb_bytes(&encode(&LabelWithCount(b"ram_val_check_gamma", 0)))?;
    ctx.absorb_bytes(&[])?;
    let val_check_gamma = ctx.squeeze(SqueezeKind::Scalar)?;
    wires.challenge(RamValCheckChallenge::Gamma, val_check_gamma.clone());
    wires.derived(RamValCheckPublic::InitEval, public.init_eval());

    ctx.section("stage4/public");
    let init_eval = match native {
        Some(native) => Some(public_io::init_eval(
            native,
            &values(ctx, &stage2.ram_address)?,
        )?),
        None => None,
    };
    PublicSlots::set_input(ctx, public.init_eval_slot(), init_eval)?;

    ctx.section("stage4/batch");
    let inputs = [
        lower(ctx, &registers.input_expression::<Fr>(), &wires.sources)?,
        lower(ctx, &ram_val_check.input_expression::<Fr>(), &wires.sources)?,
    ];
    let max_rounds = register_dimensions.read_write_rounds();
    let layouts = [
        Layout::uniform(max_rounds, registers.degree(), 0),
        Layout::suffix(log_t, ram_val_check.degree(), max_rounds),
    ];
    let (batch, point, final_claim) = run_batch(ctx, &inputs, &layouts)?;

    let (register_address, register_cycle) = read_write_opening_point(register_dimensions, &point);
    let mut register_point = register_address.clone();
    register_point.extend(register_cycle.iter().cloned());
    let ram_val_check_cycle = reversed(layouts[1].slice(&point));
    let mut ram_val_check_point = stage2.ram_address.clone();
    ram_val_check_point.extend(ram_val_check_cycle.iter().cloned());

    absorb_member(ctx, wires, &registers, &[], &[], |_| register_point.clone())?;
    absorb_member(ctx, wires, &ram_val_check, &[], &[], |_| {
        ram_val_check_point.clone()
    })?;

    ctx.section("stage4/public");
    PublicSlots::bind_outputs(ctx, &public.outputs().register_address, &register_address)?;

    ctx.section("stage4/expected");
    let eq_cycle = eq(ctx, &stage3.point, &register_cycle);
    wires.derived(RegistersReadWritePublic::EqCycle, eq_cycle);
    let lt_cycle = lt(ctx, &ram_val_check_cycle, &stage2.ram_cycle);
    wires.derived(
        RamValCheckPublic::LtCyclePlusGamma,
        lt_cycle + val_check_gamma,
    );

    let expected = [
        lower(ctx, &registers.output_expression::<Fr>(), &wires.sources)?,
        lower(
            ctx,
            &ram_val_check.output_expression::<Fr>(),
            &wires.sources,
        )?,
    ];
    finish_batch(ctx, &batch, &expected, &final_claim);

    Ok(Stage4 {
        register_address,
        register_cycle,
        ram_val_check_cycle,
    })
}
