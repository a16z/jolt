//! Stage 6b: the cycle-phase batch {bytecode read-RAF, booleanity, RAM
//! Hamming booleanity, RAM RA virtualization, instruction RA virtualization,
//! increment claim reduction}, all `log_t` rounds at one point. The five
//! bytecode stage folds are outsourced public inputs.

use jolt_claims::protocols::jolt::geometry::booleanity::BooleanityDimensions;
use jolt_claims::protocols::jolt::geometry::bytecode::{bytecode_ra, read_raf_output_openings};
use jolt_claims::protocols::jolt::geometry::claim_reductions::bytecode::NUM_BYTECODE_VAL_STAGES;
use jolt_claims::protocols::jolt::geometry::dimensions::TraceDimensions;
use jolt_claims::protocols::jolt::geometry::ra::JoltRaPolynomial;
use jolt_claims::protocols::jolt::relations::booleanity::BooleanityCyclePhase;
use jolt_claims::protocols::jolt::relations::bytecode::ReadRafCyclePhase;
use jolt_claims::protocols::jolt::relations::claim_reductions::increments::ClaimReduction as IncClaimReduction;
use jolt_claims::protocols::jolt::relations::instruction::RaVirtualization as InstructionRaVirtualization;
use jolt_claims::protocols::jolt::relations::ram::{
    HammingBooleanity, RaVirtualization as RamRaVirtualization,
};
use jolt_claims::protocols::jolt::{
    BooleanityPublic, BytecodeReadRafPublic, IncClaimReductionChallenge, IncClaimReductionPublic,
    InstructionRaVirtualizationChallenge, InstructionRaVirtualizationPublic, JoltOpeningId,
    JoltRelationId, RamHammingBooleanityPublic, RamRaVirtualizationPublic,
};
use jolt_claims::SymbolicSumcheck;
use jolt_field::Fr;

use super::ctx::{Ctx, Lc};
use super::gadgets::{address_chunks, eq, eq_index_msb, identity_msb, reversed, zero_selector};
use super::lower::lower;
use super::public_io::{self, PublicSlots, StageValueInputs};
use super::replay::SqueezeKind;
use super::stage1::Stage1;
use super::stage2::{values, Stage2};
use super::stage3::Stage3;
use super::stage4::Stage4;
use super::stage5::Stage5;
use super::stage6a::Stage6a;
use super::sumcheck::finish_batch;
use super::wiring::{absorb_member, run_batch, Layout, Wires};
use super::{Native, RelationError, NUM_STAGE_VALUES};
use crate::profile::WrapperProfile;

pub(crate) struct Stage6b {
    pub r_cycle: Vec<Lc>,
    /// Per RA polynomial in layout order, the committed address chunk its
    /// virtualization opening binds (`log_k_chunk` coordinates).
    pub virtualization_chunks: Vec<Vec<Lc>>,
}

#[expect(clippy::too_many_arguments, reason = "one argument per upstream stage")]
pub(crate) fn walk(
    ctx: &mut Ctx,
    profile: &WrapperProfile,
    native: Option<&Native<'_>>,
    public: &PublicSlots,
    wires: &mut Wires,
    stage1: &Stage1,
    stage2: &Stage2,
    stage3: &Stage3,
    stage4: &Stage4,
    stage5: &Stage5,
    stage6a: &Stage6a,
) -> Result<Stage6b, RelationError> {
    let log_t = profile.log_t;
    let log_k_ram = profile.log_k_ram;
    let chunk_bits = profile.one_hot_config.committed_chunk_bits();
    let formula = &stage5.formula;
    let layout = formula.ra_layout;
    let trace = TraceDimensions::new(log_t);
    let bytecode_dimensions = formula.bytecode_read_raf;
    let bytecode = ReadRafCyclePhase::new((bytecode_dimensions, NUM_BYTECODE_VAL_STAGES));
    let booleanity =
        BooleanityCyclePhase::new(BooleanityDimensions::new(layout, log_t, chunk_bits));
    let ram_hamming = HammingBooleanity::new(trace);
    let ram_virtualization = RamRaVirtualization::new(formula.ram_ra_virtualization);
    let instruction_virtualization =
        InstructionRaVirtualization::new(formula.instruction_ra_virtualization);
    let inc_reduction = IncClaimReduction::new(trace);

    ctx.section("stage6b/batch");
    let instruction_gamma = ctx.squeeze(SqueezeKind::Scalar)?;
    wires.challenge(
        InstructionRaVirtualizationChallenge::Gamma,
        instruction_gamma,
    );
    let inc_gamma = ctx.squeeze(SqueezeKind::Scalar)?;
    wires.challenge(IncClaimReductionChallenge::Gamma, inc_gamma);

    let inputs = [
        lower(ctx, &bytecode.input_expression::<Fr>(), &wires.sources)?,
        lower(ctx, &booleanity.input_expression::<Fr>(), &wires.sources)?,
        lower(ctx, &ram_hamming.input_expression::<Fr>(), &wires.sources)?,
        lower(
            ctx,
            &ram_virtualization.input_expression::<Fr>(),
            &wires.sources,
        )?,
        lower(
            ctx,
            &instruction_virtualization.input_expression::<Fr>(),
            &wires.sources,
        )?,
        lower(ctx, &inc_reduction.input_expression::<Fr>(), &wires.sources)?,
    ];
    let layouts = [
        Layout::uniform(log_t, bytecode.degree(), 0),
        Layout::uniform(log_t, booleanity.degree(), 0),
        Layout::uniform(log_t, ram_hamming.degree(), 0),
        Layout::uniform(log_t, ram_virtualization.degree(), 0),
        Layout::uniform(log_t, instruction_virtualization.degree(), 0),
        Layout::uniform(log_t, inc_reduction.degree(), 0),
    ];
    let (batch, point, final_claim) = run_batch(ctx, &inputs, &layouts)?;
    let r_cycle = reversed(&point);
    let with_cycle = |prefix: &[Lc]| {
        let mut point = prefix.to_vec();
        point.extend(r_cycle.iter().cloned());
        point
    };

    let bytecode_chunks = address_chunks(&stage6a.bytecode_address, chunk_bits);
    let ram_chunks = address_chunks(&stage5.ram_reduced_point[..log_k_ram], chunk_bits);
    let instruction_chunks = address_chunks(&stage5.instruction_address, chunk_bits);
    let bytecode_points: Vec<Vec<Lc>> = bytecode_chunks
        .iter()
        .map(|chunk| with_cycle(chunk))
        .collect();
    let booleanity_point = with_cycle(&stage6a.booleanity_address);
    let bytecode_outputs = read_raf_output_openings(bytecode_dimensions);
    absorb_member(ctx, wires, &bytecode, &[], &[], |id| {
        let index = bytecode_outputs
            .bytecode_ra
            .iter()
            .position(|ra| ra == id)
            .unwrap_or(0);
        bytecode_points[index].clone()
    })?;
    // A booleanity bytecode opening whose chunk is the booleanity address
    // itself (the last chunk of the shared stage-6a point) is the same
    // polynomial at the same wires as the read-RAF's: aliased, not absorbed.
    let aliases: Vec<(JoltOpeningId, JoltOpeningId)> = (0..layout.bytecode())
        .filter(|&index| bytecode_points[index] == booleanity_point)
        .map(|index| {
            (
                JoltRaPolynomial::Bytecode(index).opening(JoltRelationId::Booleanity),
                bytecode_ra(index),
            )
        })
        .collect();
    absorb_member(ctx, wires, &booleanity, &[], &aliases, |_| {
        booleanity_point.clone()
    })?;
    absorb_member(ctx, wires, &ram_hamming, &[], &[], |_| r_cycle.clone())?;
    let ram_ids: Vec<JoltOpeningId> = (0..layout.ram())
        .map(|index| JoltRaPolynomial::Ram(index).opening(JoltRelationId::RamRaVirtualization))
        .collect();
    absorb_member(ctx, wires, &ram_virtualization, &[], &[], |id| {
        let index = ram_ids.iter().position(|ra| ra == id).unwrap_or(0);
        with_cycle(&ram_chunks[index])
    })?;
    let instruction_ids: Vec<JoltOpeningId> = (0..layout.instruction())
        .map(|index| {
            JoltRaPolynomial::Instruction(index)
                .opening(JoltRelationId::InstructionRaVirtualization)
        })
        .collect();
    absorb_member(ctx, wires, &instruction_virtualization, &[], &[], |id| {
        let index = instruction_ids.iter().position(|ra| ra == id).unwrap_or(0);
        with_cycle(&instruction_chunks[index])
    })?;
    absorb_member(ctx, wires, &inc_reduction, &[], &[], |_| r_cycle.clone())?;

    ctx.section("stage6b/public");
    let stage_values = match native {
        Some(native) => {
            let gammas = values(ctx, &stage6a.bytecode_gammas[1..])?;
            let stage_gammas: [Fr; NUM_STAGE_VALUES] = gammas
                .try_into()
                .map_err(|_| RelationError::Geometry("bytecode gamma count".into()))?;
            Some(public_io::stage_values(
                native,
                StageValueInputs {
                    bytecode_address: &values(ctx, &stage6a.bytecode_address)?,
                    register_address: &values(ctx, &stage4.register_address)?,
                    stage_gammas,
                },
            )?)
        }
        None => None,
    };
    for stage in 0..NUM_STAGE_VALUES {
        PublicSlots::set_input(
            ctx,
            public.stage_value_slot(stage),
            stage_values.map(|values| values[stage]),
        )?;
    }

    ctx.section("stage6b/expected");
    let stage_cycle_points: [&[Lc]; NUM_STAGE_VALUES] = [
        &stage2.tau_low,
        &stage2.product_point,
        &stage3.point,
        &stage4.register_cycle,
        &stage5.register_val_cycle,
    ];
    let mut stage_cycle_eq = Vec::with_capacity(NUM_STAGE_VALUES);
    for (stage, cycle_point) in stage_cycle_points.iter().enumerate() {
        let eq_cycle = eq(ctx, cycle_point, &r_cycle);
        let value = ctx.mul(&public.stage_value(stage), &eq_cycle);
        wires.derived(BytecodeReadRafPublic::StageValue(stage), value);
        wires.derived(BytecodeReadRafPublic::StageCycleEq(stage), eq_cycle.clone());
        stage_cycle_eq.push(eq_cycle);
    }
    let identity = identity_msb(&stage6a.bytecode_address);
    let outer_raf = ctx.mul(&identity, &stage_cycle_eq[0]);
    wires.derived(BytecodeReadRafPublic::SpartanOuterRaf, outer_raf);
    let shift_raf = ctx.mul(&identity, &stage_cycle_eq[2]);
    wires.derived(BytecodeReadRafPublic::SpartanShiftRaf, shift_raf);
    let entry_address = eq_index_msb(
        ctx,
        &stage6a.bytecode_address,
        profile.entry_bytecode_index as u128,
    );
    let first_cycle = zero_selector(ctx, &r_cycle);
    let entry = ctx.mul(&entry_address, &first_cycle);
    wires.derived(BytecodeReadRafPublic::Entry, entry);

    let mut sumcheck_point = reversed(&stage6a.booleanity_address);
    sumcheck_point.extend(reversed(&r_cycle));
    let mut reference_point = reversed(&stage6a.reference_address);
    reference_point.extend(stage5.instruction_cycle.iter().cloned());
    let eq_address_cycle = eq(ctx, &sumcheck_point, &reference_point);
    wires.derived(BooleanityPublic::EqAddressCycle, eq_address_cycle);
    let eq_hamming = eq(ctx, &point, stage1.cycle_binding());
    wires.derived(RamHammingBooleanityPublic::EqCycle, eq_hamming);
    let eq_ram = eq(ctx, &stage5.ram_reduced_point[log_k_ram..], &r_cycle);
    wires.derived(RamRaVirtualizationPublic::EqCycle, eq_ram);
    let eq_instruction = eq(ctx, &stage5.instruction_cycle, &r_cycle);
    wires.derived(InstructionRaVirtualizationPublic::EqCycle, eq_instruction);
    let eq_ram_rw = eq(ctx, &r_cycle, &stage2.ram_cycle);
    wires.derived(IncClaimReductionPublic::EqRamReadWrite, eq_ram_rw);
    let eq_ram_vc = eq(ctx, &r_cycle, &stage4.ram_val_check_cycle);
    wires.derived(IncClaimReductionPublic::EqRamValCheck, eq_ram_vc);
    let eq_reg_rw = eq(ctx, &r_cycle, &stage4.register_cycle);
    wires.derived(IncClaimReductionPublic::EqRegistersReadWrite, eq_reg_rw);
    let eq_reg_val = eq(ctx, &r_cycle, &stage5.register_val_cycle);
    wires.derived(
        IncClaimReductionPublic::EqRegistersValEvaluation,
        eq_reg_val,
    );

    let expected = [
        lower(ctx, &bytecode.output_expression::<Fr>(), &wires.sources)?,
        lower(ctx, &booleanity.output_expression::<Fr>(), &wires.sources)?,
        lower(ctx, &ram_hamming.output_expression::<Fr>(), &wires.sources)?,
        lower(
            ctx,
            &ram_virtualization.output_expression::<Fr>(),
            &wires.sources,
        )?,
        lower(
            ctx,
            &instruction_virtualization.output_expression::<Fr>(),
            &wires.sources,
        )?,
        lower(
            ctx,
            &inc_reduction.output_expression::<Fr>(),
            &wires.sources,
        )?,
    ];
    finish_batch(ctx, &batch, &expected, &final_claim);

    let mut virtualization_chunks = instruction_chunks;
    virtualization_chunks.extend(bytecode_chunks);
    virtualization_chunks.extend(ram_chunks);
    Ok(Stage6b {
        r_cycle,
        virtualization_chunks,
    })
}
