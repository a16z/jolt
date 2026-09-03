//! Stage 5: {instruction read-RAF, RAM RA claim reduction, registers value
//! evaluation}. The read-RAF member emits degree-2 polynomials over its 128
//! address rounds and degree `D + 2` over the cycle rounds.

use jolt_claims::protocols::jolt::geometry::dimensions::TraceDimensions;
use jolt_claims::protocols::jolt::geometry::instruction::read_raf_output_openings;
use jolt_claims::protocols::jolt::relations::instruction::ReadRaf;
use jolt_claims::protocols::jolt::relations::ram::RaClaimReduction;
use jolt_claims::protocols::jolt::relations::registers::ValEvaluation;
use jolt_claims::protocols::jolt::{
    InstructionReadRafChallenge, InstructionReadRafPublic, JoltFormulaDimensions, JoltRelationId,
    RamRaClaimReductionChallenge, RamRaClaimReductionPublic, RegistersValEvaluationPublic,
};
use jolt_claims::SymbolicSumcheck;
use jolt_field::Fr;
use jolt_verifier::stages::formula_dimensions_from_parts;

use super::ctx::{Ctx, Lc};
use super::gadgets::{address_chunks, eq, identity_msb, lt, operand, reversed};
use super::lower::lower;
use super::replay::SqueezeKind;
use super::stage2::Stage2;
use super::stage4::Stage4;
use super::sumcheck::finish_batch;
use super::tables::table_mles;
use super::wiring::{absorb_member, run_batch, Layout, Wires};
use super::RelationError;
use crate::profile::WrapperProfile;

/// Instruction lookups index a 128-bit interleaved operand address.
pub(crate) const INSTRUCTION_ADDRESS_BITS: usize = 128;
/// The read-RAF address-phase rounds emit `eq · (linear)` polynomials.
const READ_RAF_ADDRESS_ROUND_DEGREE: usize = 2;

pub(crate) struct Stage5 {
    pub formula: JoltFormulaDimensions,
    pub instruction_address: Vec<Lc>,
    pub instruction_cycle: Vec<Lc>,
    /// The RAM RA claim-reduction point (`ram_address ‖ cycle`).
    pub ram_reduced_point: Vec<Lc>,
    pub register_val_cycle: Vec<Lc>,
}

pub(crate) fn formula_dimensions(
    profile: &WrapperProfile,
) -> Result<JoltFormulaDimensions, RelationError> {
    Ok(formula_dimensions_from_parts(
        profile.one_hot_config,
        profile.log_t,
        profile.bytecode_len(),
        profile.ram_k(),
        JoltRelationId::InstructionReadRaf,
    )?)
}

pub(crate) fn walk(
    ctx: &mut Ctx,
    profile: &WrapperProfile,
    wires: &mut Wires,
    stage2: &Stage2,
    stage4: &Stage4,
) -> Result<Stage5, RelationError> {
    let log_t = profile.log_t;
    let formula = formula_dimensions(profile)?;
    let read_raf_dimensions = formula.instruction_read_raf;
    let read_raf = ReadRaf::new(read_raf_dimensions);
    let ram_reduction = RaClaimReduction::new(TraceDimensions::new(log_t));
    let registers_val = ValEvaluation::new(TraceDimensions::new(log_t));

    ctx.section("stage5/batch");
    let instruction_gamma = ctx.squeeze(SqueezeKind::Scalar)?;
    wires.challenge(
        InstructionReadRafChallenge::Gamma,
        instruction_gamma.clone(),
    );
    let ram_gamma = ctx.squeeze(SqueezeKind::Scalar)?;
    wires.challenge(RamRaClaimReductionChallenge::Gamma, ram_gamma);

    let inputs = [
        lower(ctx, &read_raf.input_expression::<Fr>(), &wires.sources)?,
        lower(ctx, &ram_reduction.input_expression::<Fr>(), &wires.sources)?,
        lower(ctx, &registers_val.input_expression::<Fr>(), &wires.sources)?,
    ];
    let max_rounds = INSTRUCTION_ADDRESS_BITS + log_t;
    let mut read_raf_degrees = vec![READ_RAF_ADDRESS_ROUND_DEGREE; INSTRUCTION_ADDRESS_BITS];
    read_raf_degrees.extend(std::iter::repeat_n(read_raf.degree(), log_t));
    let layouts = [
        Layout {
            rounds: max_rounds,
            offset: 0,
            degrees: read_raf_degrees,
        },
        Layout::suffix(log_t, ram_reduction.degree(), max_rounds),
        Layout::suffix(log_t, registers_val.degree(), max_rounds),
    ];
    let (batch, point, final_claim) = run_batch(ctx, &inputs, &layouts)?;

    let instruction_address = point[..INSTRUCTION_ADDRESS_BITS].to_vec();
    let instruction_cycle = reversed(&point[INSTRUCTION_ADDRESS_BITS..]);
    let virtual_chunks = address_chunks(
        &instruction_address,
        profile.one_hot_config.lookup_virtual_chunk_bits(),
    );
    let ram_cycle = reversed(layouts[1].slice(&point));
    let mut ram_reduced_point = stage2.ram_address.clone();
    ram_reduced_point.extend(ram_cycle.iter().cloned());
    let register_val_cycle = reversed(layouts[2].slice(&point));
    let mut register_val_point = stage4.register_address.clone();
    register_val_point.extend(register_val_cycle.iter().cloned());

    let outputs = read_raf_output_openings(read_raf_dimensions);
    absorb_member(ctx, wires, &read_raf, &[], &[], |id| {
        match outputs.instruction_ra.iter().position(|ra| ra == id) {
            Some(index) => {
                let mut point = virtual_chunks[index].clone();
                point.extend(instruction_cycle.iter().cloned());
                point
            }
            None => instruction_cycle.clone(),
        }
    })?;
    absorb_member(ctx, wires, &ram_reduction, &[], &[], |_| {
        ram_reduced_point.clone()
    })?;
    absorb_member(ctx, wires, &registers_val, &[], &[], |_| {
        register_val_point.clone()
    })?;

    ctx.section("stage5/tables");
    let eq_reduction = eq(ctx, &stage2.product_point, &instruction_cycle);
    for (index, table) in table_mles(ctx, &instruction_address)
        .into_iter()
        .enumerate()
    {
        let value = ctx.mul(&eq_reduction, &table);
        wires.derived(InstructionReadRafPublic::EqTableValue(index), value);
    }
    ctx.section("stage5/expected");
    let left = operand(&instruction_address, 0);
    let right = operand(&instruction_address, 1);
    let gamma_squared = ctx.mul(&instruction_gamma, &instruction_gamma);
    let gamma_left = ctx.mul(&instruction_gamma, &left);
    let gamma2_right = ctx.mul(&gamma_squared, &right);
    let raf_constant = ctx.mul(&eq_reduction, &(gamma_left.clone() + gamma2_right.clone()));
    wires.derived(InstructionReadRafPublic::EqRafConstant, raf_constant);
    let gamma2_identity = ctx.mul(&gamma_squared, &identity_msb(&instruction_address));
    let raf_flag = ctx.mul(
        &eq_reduction,
        &(gamma2_identity - gamma_left - gamma2_right),
    );
    wires.derived(InstructionReadRafPublic::EqRafFlag, raf_flag);
    let eq_raf = eq(ctx, &stage2.tau_low, &ram_cycle);
    wires.derived(RamRaClaimReductionPublic::EqCycleRaf, eq_raf);
    let eq_read_write = eq(ctx, &stage2.ram_cycle, &ram_cycle);
    wires.derived(RamRaClaimReductionPublic::EqCycleReadWrite, eq_read_write);
    let eq_val_check = eq(ctx, &stage4.ram_val_check_cycle, &ram_cycle);
    wires.derived(RamRaClaimReductionPublic::EqCycleValCheck, eq_val_check);
    let lt_cycle = lt(ctx, &register_val_cycle, &stage4.register_cycle);
    wires.derived(RegistersValEvaluationPublic::LtCycle, lt_cycle);

    let expected = [
        lower(ctx, &read_raf.output_expression::<Fr>(), &wires.sources)?,
        lower(
            ctx,
            &ram_reduction.output_expression::<Fr>(),
            &wires.sources,
        )?,
        lower(
            ctx,
            &registers_val.output_expression::<Fr>(),
            &wires.sources,
        )?,
    ];
    finish_batch(ctx, &batch, &expected, &final_claim);

    Ok(Stage5 {
        formula,
        instruction_address,
        instruction_cycle,
        ram_reduced_point,
        register_val_cycle,
    })
}
