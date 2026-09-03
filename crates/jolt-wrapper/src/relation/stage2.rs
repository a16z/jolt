//! Stage 2: the product uni-skip round, then the batch {RAM read-write
//! checking, product remainder, instruction claim reduction, RAM RAF
//! evaluation, RAM output check}.

use common::constants::RAM_START_ADDRESS;
use jolt_claims::protocols::jolt::geometry::dimensions::TraceDimensions;
use jolt_claims::protocols::jolt::geometry::instruction::{
    input_virtualization_consistency_openings, read_raf_consistency_openings,
};
use jolt_claims::protocols::jolt::geometry::ram::RamRafEvaluationDimensions;
use jolt_claims::protocols::jolt::geometry::spartan::{
    product_uniskip_opening, virtual_instruction_product, write_lookup_output_to_rd_product,
    SpartanProductDimensions,
};
use jolt_claims::protocols::jolt::relations::claim_reductions::instruction::ClaimReduction as InstructionClaimReduction;
use jolt_claims::protocols::jolt::relations::ram::{OutputCheck, RafEvaluation, ReadWriteChecking};
use jolt_claims::protocols::jolt::relations::spartan::{ProductRemainder, ProductUniskip};
use jolt_claims::protocols::jolt::{
    InstructionClaimReductionChallenge, InstructionClaimReductionPublic, RamOutputCheckPublic,
    RamRafEvaluationPublic, RamReadWriteChallenge, RamReadWritePublic,
    SpartanProductVirtualizationPublic,
};
use jolt_claims::SymbolicSumcheck;
use jolt_field::{Fr, Ring};

use super::ctx::{Ctx, Lc};
use super::gadgets::{
    address_opening_point, centered_lagrange, centered_lagrange_kernel, eq, identity_msb,
    range_mask_msb, read_write_opening_point, reversed,
};
use super::lower::lower;
use super::public_io::{self, PublicSlots};
use super::replay::SqueezeKind;
use super::stage1::Stage1;
use super::sumcheck::{finish_batch, uniskip};
use super::wiring::{absorb_member, run_batch, Layout, Wires};
use super::{Native, RelationError};
use crate::profile::WrapperProfile;

const PRODUCT_UNISKIP_DOMAIN_SIZE: usize = 3;

pub(crate) struct Stage2 {
    /// `rev(stage-1 cycle binding)`: the Spartan cycle point every stage-2/3
    /// eq consumer binds against.
    pub tau_low: Vec<Lc>,
    pub ram_address: Vec<Lc>,
    pub ram_cycle: Vec<Lc>,
    /// The product remainder / instruction claim-reduction point.
    pub product_point: Vec<Lc>,
}

pub(crate) fn walk(
    ctx: &mut Ctx,
    profile: &WrapperProfile,
    native: Option<&Native<'_>>,
    public: &PublicSlots,
    wires: &mut Wires,
    stage1: &Stage1,
) -> Result<Stage2, RelationError> {
    let log_t = profile.log_t;
    let tau_low = reversed(stage1.cycle_binding());

    ctx.section("stage2/uniskip");
    let tau_high = ctx.squeeze(SqueezeKind::Challenge)?;
    let product_dimensions = SpartanProductDimensions::new(log_t);
    let uniskip_relation = ProductUniskip::new(product_dimensions);
    for (index, weight) in centered_lagrange(ctx, PRODUCT_UNISKIP_DOMAIN_SIZE, &tau_high)
        .into_iter()
        .enumerate()
    {
        wires.derived(
            SpartanProductVirtualizationPublic::UniskipLagrangeWeight(index),
            weight,
        );
    }
    let uniskip_input = lower(
        ctx,
        &uniskip_relation.input_expression::<Fr>(),
        &wires.sources,
    )?;
    let (uniskip_challenge, uniskip_output) = uniskip(
        ctx,
        &uniskip_input,
        uniskip_relation.degree(),
        PRODUCT_UNISKIP_DOMAIN_SIZE,
    )?;
    wires.set(product_uniskip_opening(), uniskip_output, Vec::new());

    ctx.section("stage2/batch");
    let rw = profile.rw_config.ram_dimensions(log_t, profile.log_k_ram);
    let raf_dimensions = RamRafEvaluationDimensions::try_from(rw)
        .map_err(|error| RelationError::Geometry(format!("{error:?}")))?;
    let ram_read_write = ReadWriteChecking::new(rw);
    let product_remainder = ProductRemainder::new(product_dimensions);
    let instruction_reduction = InstructionClaimReduction::new(TraceDimensions::new(log_t));
    let raf_evaluation = RafEvaluation::new(raf_dimensions);
    let output_check = OutputCheck::new(rw);

    let ram_gamma = ctx.squeeze(SqueezeKind::Scalar)?;
    wires.challenge(RamReadWriteChallenge::Gamma, ram_gamma);
    let instruction_gamma = ctx.squeeze(SqueezeKind::Scalar)?;
    wires.challenge(InstructionClaimReductionChallenge::Gamma, instruction_gamma);
    let output_address = ctx.squeeze_vector(SqueezeKind::Challenge, profile.log_k_ram)?;

    let inputs = [
        lower(
            ctx,
            &ram_read_write.input_expression::<Fr>(),
            &wires.sources,
        )?,
        lower(
            ctx,
            &product_remainder.input_expression::<Fr>(),
            &wires.sources,
        )?,
        lower(
            ctx,
            &instruction_reduction.input_expression::<Fr>(),
            &wires.sources,
        )?,
        lower(
            ctx,
            &raf_evaluation.input_expression::<Fr>(),
            &wires.sources,
        )?,
        lower(ctx, &output_check.input_expression::<Fr>(), &wires.sources)?,
    ];
    let max_rounds = rw.read_write_rounds();
    let phase1_offset = rw.phase1_num_rounds();
    let layouts = [
        Layout::uniform(max_rounds, ram_read_write.degree(), 0),
        Layout::suffix(log_t, product_remainder.degree(), max_rounds),
        Layout::suffix(log_t, instruction_reduction.degree(), max_rounds),
        Layout::uniform(
            rw.raf_evaluation_rounds(),
            raf_evaluation.degree(),
            phase1_offset,
        ),
        Layout::uniform(
            rw.output_check_rounds(),
            output_check.degree(),
            phase1_offset,
        ),
    ];
    let (batch, point, final_claim) = run_batch(ctx, &inputs, &layouts)?;

    let (ram_address, ram_cycle) = read_write_opening_point(rw, &point);
    let mut ram_point = ram_address.clone();
    ram_point.extend(ram_cycle.iter().cloned());
    let product_point = reversed(layouts[1].slice(&point));
    let raf_address = address_opening_point(rw, layouts[3].slice(&point));
    let mut raf_point = raf_address.clone();
    raf_point.extend(tau_low.iter().cloned());
    let output_check_address = address_opening_point(rw, layouts[4].slice(&point));

    absorb_member(ctx, wires, &ram_read_write, &[], &[], |_| ram_point.clone())?;
    absorb_member(
        ctx,
        wires,
        &product_remainder,
        &[
            write_lookup_output_to_rd_product(),
            virtual_instruction_product(),
        ],
        &[],
        |_| product_point.clone(),
    )?;
    let [lookup_alias] = read_raf_consistency_openings();
    let [left_alias, right_alias] = input_virtualization_consistency_openings();
    absorb_member(
        ctx,
        wires,
        &instruction_reduction,
        &[],
        &[lookup_alias, left_alias, right_alias],
        |_| product_point.clone(),
    )?;
    absorb_member(ctx, wires, &raf_evaluation, &[], &[], |_| raf_point.clone())?;
    absorb_member(ctx, wires, &output_check, &[], &[], |_| {
        output_check_address.clone()
    })?;

    ctx.section("stage2/public");
    // The output-check address is the read-write address wire for wire
    // (`address_opening_point` reorders the same batch coordinates), so one
    // public copy serves `ValIo` and `InitEval`.
    debug_assert_eq!(output_check_address, ram_address);
    PublicSlots::bind_outputs(ctx, &public.outputs().ram_address, &ram_address)?;
    let val_io = match native {
        Some(native) => {
            let address = values(ctx, &ram_address)?;
            Some(public_io::val_io(native, &address)?)
        }
        None => None,
    };
    PublicSlots::set_input(ctx, public.val_io_slot(), val_io)?;

    ctx.section("stage2/expected");
    let eq_cycle = eq(ctx, &tau_low, &ram_cycle);
    wires.derived(RamReadWritePublic::EqCycle, eq_cycle);
    for (index, weight) in centered_lagrange(ctx, PRODUCT_UNISKIP_DOMAIN_SIZE, &uniskip_challenge)
        .into_iter()
        .enumerate()
    {
        wires.derived(
            SpartanProductVirtualizationPublic::LagrangeWeight(index),
            weight,
        );
    }
    let kernel = centered_lagrange_kernel(
        ctx,
        PRODUCT_UNISKIP_DOMAIN_SIZE,
        &tau_high,
        &uniskip_challenge,
    );
    let eq_product = eq(ctx, &tau_low, &product_point);
    let tau_kernel = ctx.mul(&kernel, &eq_product);
    wires.derived(SpartanProductVirtualizationPublic::TauKernel, tau_kernel);
    let eq_spartan = eq(ctx, &product_point, &tau_low);
    wires.derived(InstructionClaimReductionPublic::EqSpartan, eq_spartan);
    let lowest_address = Fr::from_u64(profile.memory_layout.get_lowest_address());
    let unmap = identity_msb(&raf_address).scale(Fr::from_u64(8)) + Lc::constant(lowest_address);
    wires.derived(RamRafEvaluationPublic::UnmapAddress, unmap);
    let eq_address = eq(ctx, &output_address, &output_check_address);
    wires.derived(RamOutputCheckPublic::EqAddress, eq_address);
    let layout = &profile.memory_layout;
    let io_mask_start = layout
        .remapped_word_address(layout.input_start)
        .map_err(|error| RelationError::Geometry(error.to_string()))?
        as u128;
    let io_mask_end = layout
        .remapped_word_address(RAM_START_ADDRESS)
        .map_err(|error| RelationError::Geometry(error.to_string()))? as u128;
    let io_mask = range_mask_msb(ctx, io_mask_start, io_mask_end, &output_check_address);
    wires.derived(RamOutputCheckPublic::IoMask, io_mask);
    wires.derived(RamOutputCheckPublic::ValIo, public.val_io());

    let expected = [
        lower(
            ctx,
            &ram_read_write.output_expression::<Fr>(),
            &wires.sources,
        )?,
        lower(
            ctx,
            &product_remainder.output_expression::<Fr>(),
            &wires.sources,
        )?,
        lower(
            ctx,
            &instruction_reduction.output_expression::<Fr>(),
            &wires.sources,
        )?,
        lower(
            ctx,
            &raf_evaluation.output_expression::<Fr>(),
            &wires.sources,
        )?,
        lower(ctx, &output_check.output_expression::<Fr>(), &wires.sources)?,
    ];
    finish_batch(ctx, &batch, &expected, &final_claim);

    Ok(Stage2 {
        tau_low,
        ram_address,
        ram_cycle,
        product_point,
    })
}

/// The assigned values of a point (assign mode only).
pub(crate) fn values(ctx: &Ctx, point: &[Lc]) -> Result<Vec<Fr>, RelationError> {
    point
        .iter()
        .map(|lc| ctx.value(lc).ok_or(RelationError::Witness(0)))
        .collect()
}
