use jolt_field::{Field, RingCore};
use jolt_lookup_tables::{InstructionLookupTable, LookupTableKind, XLEN};
use jolt_poly::{EqPolynomial, IdentityPolynomial, MultilinearEvaluation};
use jolt_riscv::{
    instructions::Noop, CircuitFlags, Flags, InstructionFlags, InterleavedBitsMarker,
    JoltInstruction, JoltInstructionRow, CIRCUIT_FLAGS, NUM_CIRCUIT_FLAGS,
};

use crate::{challenge, derived, opening};

use super::super::{
    BytecodeReadRafChallenge, BytecodeReadRafPublic, JoltCommittedPolynomial, JoltExpr,
    JoltOpeningId, JoltRelationId, JoltVirtualPolynomial,
};
use super::claim_reductions::bytecode::NUM_BYTECODE_VAL_STAGES;
use super::dimensions::JoltFormulaPointError;
use super::error::require_len;
use super::instruction::{imm, instruction_raf_flag, lookup_table_flag, unexpanded_pc};
use super::registers::{
    rd_wa_read_write, rd_wa_val_evaluation, rs1_ra_read_write, rs2_ra_read_write,
};
use super::spartan::{pc_shift, unexpanded_pc_shift};

/// Per-stage (1..=5) gamma-power vector lengths for the bytecode read-RAF stage
/// folds — the arities of the prover's `challenge_scalar_powers` draws. The
/// verifier stores each stage's single drawn scalar and expands it with
/// [`stage_gamma_powers`], so these lengths are single-sourced with the
/// fold-side `require_len` guards.
///
/// [`stage_gamma_powers`]: crate::protocols::jolt::relations::bytecode::BytecodeReadRafAddressPhaseChallenges::stage_gamma_powers
pub const BYTECODE_STAGE_GAMMA_COUNTS: [usize; 5] = [
    // Stage 1: UnexpandedPC, Imm, then one per circuit flag (all Spartan outer).
    2 + NUM_CIRCUIT_FLAGS,
    // Stage 2: the Jump, Branch, WriteLookupOutputToRD, and VirtualInstruction
    // product-virtualization flags.
    4,
    // Stage 3: Imm (instruction input), UnexpandedPC (shift), the four
    // operand-source flags, IsNoop, VirtualInstruction, IsFirstInSequence.
    9,
    // Stage 4: the RdWa, Rs1Ra, Rs2Ra register read-write openings.
    3,
    // Stage 5: RdWa (registers val evaluation), InstructionRafFlag, then one
    // per lookup table flag.
    2 + LookupTableKind::<XLEN>::COUNT,
];

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct BytecodeReadRafDimensions {
    log_t: usize,
    log_k: usize,
    committed_ra_polys: usize,
}

impl BytecodeReadRafDimensions {
    pub const fn new(log_t: usize, log_k: usize, committed_ra_polys: usize) -> Self {
        Self {
            log_t,
            log_k,
            committed_ra_polys,
        }
    }

    pub const fn log_t(self) -> usize {
        self.log_t
    }

    pub const fn log_k(self) -> usize {
        self.log_k
    }

    pub const fn num_committed_ra_polys(self) -> usize {
        self.committed_ra_polys
    }

    pub const fn sumcheck_rounds(self) -> usize {
        self.log_t + self.log_k
    }
}

/// The staged input fold shared by the bytecode read-RAF monolith and its
/// address phase: the five base staged claims at `γ^0..4`, then one raw claim
/// per `extra_stage_claims` entry at the following powers (the lattice
/// fused-inc consumer stages), then the Spartan outer/shift PC openings and
/// the constant entry term at the next three powers.
pub(crate) fn read_raf_address_input_fold<F>(extra_stage_claims: Vec<JoltExpr<F>>) -> JoltExpr<F>
where
    F: RingCore,
{
    let gamma = challenge(BytecodeReadRafChallenge::Gamma);
    let base_stages = BYTECODE_STAGE_GAMMA_COUNTS.len();
    let num_val_stages = base_stages + extra_stage_claims.len();

    let mut fold = gamma.clone().pow(num_val_stages + 2)
        + stage1_claim()
        + gamma.clone() * stage2_claim()
        + gamma.clone().pow(2) * stage3_claim()
        + gamma.clone().pow(3) * stage4_claim()
        + gamma.clone().pow(4) * stage5_claim::<F>();
    for (index, claim) in extra_stage_claims.into_iter().enumerate() {
        fold = fold + gamma.clone().pow(base_stages + index) * claim;
    }
    fold + gamma.clone().pow(num_val_stages) * opening(pc_spartan_outer())
        + gamma.pow(num_val_stages + 1) * opening(pc_shift())
}

pub(crate) fn read_raf_cycle_output<F>(
    dimensions: BytecodeReadRafDimensions,
    num_val_stages: usize,
) -> JoltExpr<F>
where
    F: RingCore,
{
    let gamma = challenge(BytecodeReadRafChallenge::Gamma);
    let mut output_coeff = JoltExpr::zero();
    for stage in 0..num_val_stages {
        output_coeff = output_coeff
            + gamma.clone().pow(stage) * derived(BytecodeReadRafPublic::StageValue(stage));
    }
    output_coeff = output_coeff
        + gamma.clone().pow(num_val_stages) * derived(BytecodeReadRafPublic::SpartanOuterRaf)
        + gamma.clone().pow(num_val_stages + 1) * derived(BytecodeReadRafPublic::SpartanShiftRaf)
        + gamma.pow(num_val_stages + 2) * derived(BytecodeReadRafPublic::Entry);

    output_coeff * bytecode_ra_product(dimensions)
}

pub(crate) fn read_raf_cycle_output_committed<F>(
    dimensions: BytecodeReadRafDimensions,
    num_val_stages: usize,
) -> JoltExpr<F>
where
    F: RingCore,
{
    let gamma = challenge(BytecodeReadRafChallenge::Gamma);
    // The staged Val factor multiplies after the RA product so the lowered
    // R1CS auxiliary chain matches core's `[ra..., val_stage]` factor order.
    let mut output = JoltExpr::zero();
    for stage in 0..num_val_stages {
        output = output
            + gamma.clone().pow(stage)
                * derived(BytecodeReadRafPublic::StageCycleEq(stage))
                * bytecode_ra_product(dimensions)
                * opening(super::claim_reductions::bytecode::bytecode_val_stage_opening(stage));
    }
    let raf_coeff = gamma.clone().pow(num_val_stages)
        * derived(BytecodeReadRafPublic::SpartanOuterRaf)
        + gamma.clone().pow(num_val_stages + 1) * derived(BytecodeReadRafPublic::SpartanShiftRaf)
        + gamma.pow(num_val_stages + 2) * derived(BytecodeReadRafPublic::Entry);

    output + raf_coeff * bytecode_ra_product(dimensions)
}

/// Fused-inc consumer stages appended after the five base flag stages in
/// lattice mode: `(store, r_ram_read_write)`, `(store, r_ram_val_check)`,
/// `(¬store, r_registers_read_write)`, `(¬store, r_registers_val_evaluation)`
/// — each cycle-weighted by the `FusedInc` stream, discharging the four base
/// inc claims inside the read-raf fold (per-cycle store/rd disjointness makes
/// `FusedInc·Store = RamInc` and `FusedInc·(1−Store) = RdInc`).
pub const LATTICE_FUSED_INC_STAGES: usize = 4;

/// The read-raf relation's val-stage count (one cycle point / cycle-eq public
/// per stage): the five base flag stages, plus (akita) the four fused-inc
/// consumer stages. Distinct from [`NUM_BYTECODE_VAL_STAGES`], the staged
/// `BytecodeValStage` *wire* count — the fused stages resolve through the
/// store wire and its complement, adding no wires.
#[cfg(not(feature = "akita"))]
pub const READ_RAF_CYCLE_STAGES: usize = BYTECODE_STAGE_GAMMA_COUNTS.len();
#[cfg(feature = "akita")]
pub const READ_RAF_CYCLE_STAGES: usize =
    BYTECODE_STAGE_GAMMA_COUNTS.len() + LATTICE_FUSED_INC_STAGES;

/// The `FusedInc` opening produced by the lattice read-raf cycle phase at its
/// bound cycle point (shared with the rest of the stage-6b batch).
pub fn fused_inc_read_raf_opening() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::FusedInc,
        JoltRelationId::BytecodeReadRaf,
    )
}

/// Lattice full-mode cycle output: the base five stage values ride the RA
/// product as usual; the four fused-inc stages additionally carry the
/// `FusedInc` opening as a cycle factor (degree +1 over the base relation).
pub(crate) fn read_raf_cycle_output_lattice<F>(dimensions: BytecodeReadRafDimensions) -> JoltExpr<F>
where
    F: RingCore,
{
    let gamma = challenge(BytecodeReadRafChallenge::Gamma);
    let base_stages = BYTECODE_STAGE_GAMMA_COUNTS.len();
    let num_val_stages = base_stages + LATTICE_FUSED_INC_STAGES;
    let mut base_coeff = JoltExpr::zero();
    for stage in 0..base_stages {
        base_coeff = base_coeff
            + gamma.clone().pow(stage) * derived(BytecodeReadRafPublic::StageValue(stage));
    }
    base_coeff = base_coeff
        + gamma.clone().pow(num_val_stages) * derived(BytecodeReadRafPublic::SpartanOuterRaf)
        + gamma.clone().pow(num_val_stages + 1) * derived(BytecodeReadRafPublic::SpartanShiftRaf)
        + gamma.clone().pow(num_val_stages + 2) * derived(BytecodeReadRafPublic::Entry);

    let mut fused_coeff = JoltExpr::zero();
    for stage in base_stages..num_val_stages {
        fused_coeff = fused_coeff
            + gamma.clone().pow(stage) * derived(BytecodeReadRafPublic::StageValue(stage));
    }

    (base_coeff + fused_coeff * opening(fused_inc_read_raf_opening()))
        * bytecode_ra_product(dimensions)
}

/// Lattice committed-mode cycle output: the base five stages fold their staged
/// `BytecodeValStage` openings; the four fused stages reuse the staged *store*
/// val (index [`BYTECODE_STAGE_GAMMA_COUNTS`]`.len()`) — directly for the two
/// RAM legs, complemented (`1 − store`) for the two register legs — so the
/// staged-val wire set is unchanged from the store-claim design.
pub(crate) fn read_raf_cycle_output_committed_lattice<F>(
    dimensions: BytecodeReadRafDimensions,
) -> JoltExpr<F>
where
    F: RingCore,
{
    let gamma = challenge(BytecodeReadRafChallenge::Gamma);
    let base_stages = BYTECODE_STAGE_GAMMA_COUNTS.len();
    let num_val_stages = base_stages + LATTICE_FUSED_INC_STAGES;
    let ra_product = bytecode_ra_product(dimensions);
    let fused = opening(fused_inc_read_raf_opening());
    let store = opening(super::claim_reductions::bytecode::bytecode_val_stage_opening(base_stages));

    let mut output = JoltExpr::zero();
    for stage in 0..base_stages {
        output = output
            + gamma.clone().pow(stage)
                * derived(BytecodeReadRafPublic::StageCycleEq(stage))
                * ra_product.clone()
                * opening(super::claim_reductions::bytecode::bytecode_val_stage_opening(stage));
    }
    let store_pair = gamma.clone().pow(base_stages)
        * derived(BytecodeReadRafPublic::StageCycleEq(base_stages))
        + gamma.clone().pow(base_stages + 1)
            * derived(BytecodeReadRafPublic::StageCycleEq(base_stages + 1));
    let notstore_pair = gamma.clone().pow(base_stages + 2)
        * derived(BytecodeReadRafPublic::StageCycleEq(base_stages + 2))
        + gamma.clone().pow(base_stages + 3)
            * derived(BytecodeReadRafPublic::StageCycleEq(base_stages + 3));
    output = output
        + store_pair * ra_product.clone() * fused.clone() * store.clone()
        + notstore_pair.clone() * ra_product.clone() * fused.clone()
        - notstore_pair * ra_product.clone() * fused * store;

    let raf_coeff = gamma.clone().pow(num_val_stages)
        * derived(BytecodeReadRafPublic::SpartanOuterRaf)
        + gamma.clone().pow(num_val_stages + 1) * derived(BytecodeReadRafPublic::SpartanShiftRaf)
        + gamma.pow(num_val_stages + 2) * derived(BytecodeReadRafPublic::Entry);

    output + raf_coeff * ra_product
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BytecodeReadRafOutputOpenings {
    pub bytecode_ra: Vec<JoltOpeningId>,
}

pub fn bytecode_read_raf_address_phase_opening() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::BytecodeReadRafAddrClaim,
        JoltRelationId::BytecodeReadRaf,
    )
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BytecodeReadRafPublicValues<F: Field> {
    pub stage_values: [F; 5],
    pub spartan_outer_raf: F,
    pub spartan_shift_raf: F,
    pub entry: F,
}

impl<F: Field> BytecodeReadRafPublicValues<F> {
    /// Returns `None` for committed-mode publics (`StageCycleEq`) and
    /// out-of-range stage indices so a wrong-mode formula fails loudly at the
    /// source instead of evaluating with a silently zeroed term.
    pub fn value(&self, id: BytecodeReadRafPublic) -> Option<F> {
        match id {
            BytecodeReadRafPublic::StageValue(index) => self.stage_values.get(index).copied(),
            BytecodeReadRafPublic::StageCycleEq(_) => None,
            BytecodeReadRafPublic::SpartanOuterRaf => Some(self.spartan_outer_raf),
            BytecodeReadRafPublic::SpartanShiftRaf => Some(self.spartan_shift_raf),
            BytecodeReadRafPublic::Entry => Some(self.entry),
        }
    }
}

/// Committed-program read-RAF publics: the bytecode table is not available,
/// so only the table-independent factors are computed. The per-stage Val
/// factors are openings; their cycle-eq coefficients are public. One cycle eq
/// per relation stage — five in base mode, nine in lattice mode (the four
/// fused-inc consumer stages follow the base five).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BytecodeReadRafCommittedPublicValues<F: Field> {
    pub stage_cycle_eqs: [F; READ_RAF_CYCLE_STAGES],
    pub spartan_outer_raf: F,
    pub spartan_shift_raf: F,
    pub entry: F,
}

impl<F: Field> BytecodeReadRafCommittedPublicValues<F> {
    /// Returns `None` for full-mode publics (`StageValue`) and out-of-range
    /// stage indices so a wrong-mode formula fails loudly at the source
    /// instead of evaluating with a silently zeroed term.
    pub fn value(&self, id: BytecodeReadRafPublic) -> Option<F> {
        match id {
            BytecodeReadRafPublic::StageValue(_) => None,
            BytecodeReadRafPublic::StageCycleEq(index) => self.stage_cycle_eqs.get(index).copied(),
            BytecodeReadRafPublic::SpartanOuterRaf => Some(self.spartan_outer_raf),
            BytecodeReadRafPublic::SpartanShiftRaf => Some(self.spartan_shift_raf),
            BytecodeReadRafPublic::Entry => Some(self.entry),
        }
    }
}

pub struct BytecodeReadRafCommittedEvaluationInputs<'a, F> {
    pub r_address: &'a [F],
    pub r_cycle: &'a [F],
    /// One cycle point per relation stage (five base, nine lattice — the four
    /// fused-inc consumer points follow the base five).
    pub stage_cycle_points: [&'a [F]; READ_RAF_CYCLE_STAGES],
    pub entry_bytecode_index: usize,
}

pub fn read_raf_committed_public_values<F>(
    inputs: BytecodeReadRafCommittedEvaluationInputs<'_, F>,
) -> BytecodeReadRafCommittedPublicValues<F>
where
    F: Field,
{
    let stage_cycle_eqs = inputs
        .stage_cycle_points
        .map(|stage_cycle_point| EqPolynomial::<F>::mle(stage_cycle_point, inputs.r_cycle));
    let (spartan_outer_raf, spartan_shift_raf, entry) = read_raf_raf_entry_publics(
        inputs.r_address,
        inputs.r_cycle,
        stage_cycle_eqs[0],
        stage_cycle_eqs[2],
        inputs.entry_bytecode_index,
    );

    BytecodeReadRafCommittedPublicValues {
        stage_cycle_eqs,
        spartan_outer_raf,
        spartan_shift_raf,
        entry,
    }
}

/// Table-independent read-RAF publics shared by the full and committed
/// evaluation paths: `(SpartanOuterRaf, SpartanShiftRaf, Entry)`, where the
/// RAF terms scale `Int(r_address)` by the stage-1/stage-3 cycle-eq factors.
fn read_raf_raf_entry_publics<F>(
    r_address: &[F],
    r_cycle: &[F],
    outer_stage_cycle_eq: F,
    shift_stage_cycle_eq: F,
    entry_bytecode_index: usize,
) -> (F, F, F)
where
    F: Field,
{
    let identity = IdentityPolynomial::new(r_address.len()).evaluate(r_address);
    let spartan_outer_raf = identity * outer_stage_cycle_eq;
    let spartan_shift_raf = identity * shift_stage_cycle_eq;

    let entry_bits = (0..r_address.len())
        .map(|i| F::from_u64(((entry_bytecode_index >> (r_address.len() - 1 - i)) & 1) as u64))
        .collect::<Vec<_>>();
    let zero_cycle = vec![F::zero(); r_cycle.len()];
    let entry = EqPolynomial::<F>::mle(&entry_bits, r_address)
        * EqPolynomial::<F>::mle(&zero_cycle, r_cycle);

    (spartan_outer_raf, spartan_shift_raf, entry)
}

pub struct BytecodeReadRafEvaluationInputs<'a, F> {
    pub bytecode: &'a [JoltInstructionRow],
    pub r_address: &'a [F],
    pub r_cycle: &'a [F],
    pub stage_cycle_points: [&'a [F]; 5],
    pub register_read_write_point: &'a [F],
    pub register_val_evaluation_point: &'a [F],
    pub entry_bytecode_index: usize,
    pub stage1_gammas: &'a [F],
    pub stage2_gammas: &'a [F],
    pub stage3_gammas: &'a [F],
    pub stage4_gammas: &'a [F],
    pub stage5_gammas: &'a [F],
}

pub struct BytecodeReadRafStageValueInputs<'a, F> {
    pub bytecode: &'a [JoltInstructionRow],
    pub register_read_write_point: &'a [F],
    pub register_val_evaluation_point: &'a [F],
    pub stage1_gammas: &'a [F],
    pub stage2_gammas: &'a [F],
    pub stage3_gammas: &'a [F],
    pub stage4_gammas: &'a [F],
    pub stage5_gammas: &'a [F],
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct BytecodeReadRafRegisterEqEvals<F> {
    pub read_write: Vec<F>,
    pub val_evaluation: Vec<F>,
}

fn read_raf_register_eq_evals<F>(
    register_read_write_point: &[F],
    register_val_evaluation_point: &[F],
) -> BytecodeReadRafRegisterEqEvals<F>
where
    F: Field,
{
    BytecodeReadRafRegisterEqEvals {
        read_write: EqPolynomial::<F>::evals(register_read_write_point, None),
        val_evaluation: EqPolynomial::<F>::evals(register_val_evaluation_point, None),
    }
}

/// Every bytecode row's staged values: the five gamma-folded stages, plus
/// (lattice) the store circuit flag as the sixth staged value, folded like
/// the others by the read-raf consumers.
pub fn read_raf_stage_values<F>(
    inputs: BytecodeReadRafStageValueInputs<'_, F>,
) -> Vec<[F; NUM_BYTECODE_VAL_STAGES]>
where
    F: Field,
{
    let register_eq = read_raf_register_eq_evals(
        inputs.register_read_write_point,
        inputs.register_val_evaluation_point,
    );
    inputs
        .bytecode
        .iter()
        .map(|instruction| {
            read_raf_row_values::<F>(
                instruction,
                &register_eq.read_write,
                &register_eq.val_evaluation,
                inputs.stage1_gammas,
                inputs.stage2_gammas,
                inputs.stage3_gammas,
                inputs.stage4_gammas,
                inputs.stage5_gammas,
            )
        })
        .collect()
}

pub fn read_raf_public_values<F>(
    inputs: BytecodeReadRafEvaluationInputs<'_, F>,
) -> Result<BytecodeReadRafPublicValues<F>, JoltFormulaPointError>
where
    F: Field,
{
    require_len(inputs.stage1_gammas, BYTECODE_STAGE_GAMMA_COUNTS[0])?;
    require_len(inputs.stage2_gammas, BYTECODE_STAGE_GAMMA_COUNTS[1])?;
    require_len(inputs.stage3_gammas, BYTECODE_STAGE_GAMMA_COUNTS[2])?;
    require_len(inputs.stage4_gammas, BYTECODE_STAGE_GAMMA_COUNTS[3])?;
    require_len(inputs.stage5_gammas, BYTECODE_STAGE_GAMMA_COUNTS[4])?;

    let expected_domain = 1usize << inputs.r_address.len();
    if inputs.bytecode.len() != expected_domain {
        return Err(JoltFormulaPointError::EvaluationDomainLengthMismatch {
            expected: expected_domain,
            got: inputs.bytecode.len(),
        });
    }

    let register_eq = read_raf_register_eq_evals(
        inputs.register_read_write_point,
        inputs.register_val_evaluation_point,
    );
    let address_eq_evals = EqPolynomial::<F>::evals(inputs.r_address, None);

    // The base monolith publics carry the five gamma'd stages only; the
    // lattice sixth (store) row value never flows through this path, so the
    // zip below is deliberately driven by the five-slot accumulator.
    let mut stage_values = [F::zero(); 5];
    for (instruction, eq_address) in inputs.bytecode.iter().zip(address_eq_evals) {
        let row_values = read_raf_row_values::<F>(
            instruction,
            &register_eq.read_write,
            &register_eq.val_evaluation,
            inputs.stage1_gammas,
            inputs.stage2_gammas,
            inputs.stage3_gammas,
            inputs.stage4_gammas,
            inputs.stage5_gammas,
        );
        for (stage_value, row_value) in stage_values.iter_mut().zip(row_values) {
            *stage_value += row_value * eq_address;
        }
    }

    let stage_cycle_eqs = inputs
        .stage_cycle_points
        .iter()
        .map(|stage_cycle_point| EqPolynomial::<F>::mle(stage_cycle_point, inputs.r_cycle))
        .collect::<Vec<_>>();
    for (stage_value, stage_cycle_eq) in stage_values.iter_mut().zip(&stage_cycle_eqs) {
        *stage_value *= *stage_cycle_eq;
    }

    let (spartan_outer_raf, spartan_shift_raf, entry) = read_raf_raf_entry_publics(
        inputs.r_address,
        inputs.r_cycle,
        stage_cycle_eqs[0],
        stage_cycle_eqs[2],
        inputs.entry_bytecode_index,
    );

    Ok(BytecodeReadRafPublicValues {
        stage_values,
        spartan_outer_raf,
        spartan_shift_raf,
        entry,
    })
}

#[expect(
    clippy::too_many_arguments,
    reason = "Each gamma slice corresponds to one protocol subexpression."
)]
fn read_raf_row_values<F>(
    instruction: &JoltInstructionRow,
    register_read_write_eq: &[F],
    register_val_evaluation_eq: &[F],
    stage1_gammas: &[F],
    stage2_gammas: &[F],
    stage3_gammas: &[F],
    stage4_gammas: &[F],
    stage5_gammas: &[F],
) -> [F; NUM_BYTECODE_VAL_STAGES]
where
    F: Field,
{
    let decoded = JoltInstruction::try_from(*instruction)
        .unwrap_or(JoltInstruction::Noop(Noop(*instruction)));
    let circuit_flags = decoded.circuit_flags();
    let instruction_flags = decoded.instruction_flags();

    let mut stage1 = F::from_u64(instruction.address as u64);
    stage1 += stage1_gammas[1].mul_i128(instruction.operands.imm);
    for (index, flag) in CIRCUIT_FLAGS.into_iter().enumerate() {
        if circuit_flags[flag] {
            stage1 += stage1_gammas[index + 2];
        }
    }

    let mut stage2 = F::zero();
    if circuit_flags[CircuitFlags::Jump] {
        stage2 += stage2_gammas[0];
    }
    if instruction_flags[InstructionFlags::Branch] {
        stage2 += stage2_gammas[1];
    }
    if circuit_flags[CircuitFlags::WriteLookupOutputToRD] {
        stage2 += stage2_gammas[2];
    }
    if circuit_flags[CircuitFlags::VirtualInstruction] {
        stage2 += stage2_gammas[3];
    }

    let mut stage3 = F::from_i128(instruction.operands.imm);
    stage3 += stage3_gammas[1].mul_u64(instruction.address as u64);
    if instruction_flags[InstructionFlags::LeftOperandIsRs1Value] {
        stage3 += stage3_gammas[2];
    }
    if instruction_flags[InstructionFlags::LeftOperandIsPC] {
        stage3 += stage3_gammas[3];
    }
    if instruction_flags[InstructionFlags::RightOperandIsRs2Value] {
        stage3 += stage3_gammas[4];
    }
    if instruction_flags[InstructionFlags::RightOperandIsImm] {
        stage3 += stage3_gammas[5];
    }
    if instruction_flags[InstructionFlags::IsNoop] {
        stage3 += stage3_gammas[6];
    }
    if circuit_flags[CircuitFlags::VirtualInstruction] {
        stage3 += stage3_gammas[7];
    }
    if circuit_flags[CircuitFlags::IsFirstInSequence] {
        stage3 += stage3_gammas[8];
    }

    let stage4 = register_eq(instruction.operands.rd, register_read_write_eq) * stage4_gammas[0]
        + register_eq(instruction.operands.rs1, register_read_write_eq) * stage4_gammas[1]
        + register_eq(instruction.operands.rs2, register_read_write_eq) * stage4_gammas[2];

    let mut stage5 = register_eq(instruction.operands.rd, register_val_evaluation_eq);
    if !circuit_flags.is_interleaved_operands() {
        stage5 += stage5_gammas[1];
    }
    if let Some(table) = InstructionLookupTable::<XLEN>::lookup_table(&decoded) {
        stage5 += stage5_gammas[2 + table.index()];
    }

    #[cfg(not(feature = "akita"))]
    {
        [stage1, stage2, stage3, stage4, stage5]
    }
    // The lattice sixth stage: the store circuit flag as a raw staged value
    // (the `IncVirtualization` destination selector), folded against
    // `eq(r_address)` like the five gamma'd stages by the read-raf consumers.
    #[cfg(feature = "akita")]
    {
        let store = F::from_u64(u64::from(circuit_flags[CircuitFlags::Store]));
        [stage1, stage2, stage3, stage4, stage5, store]
    }
}

fn register_eq<F: Field>(register: Option<u8>, eq: &[F]) -> F {
    register
        .and_then(|register| eq.get(register as usize))
        .copied()
        .unwrap_or_else(F::zero)
}

pub fn read_raf_output_openings(
    dimensions: BytecodeReadRafDimensions,
) -> BytecodeReadRafOutputOpenings {
    BytecodeReadRafOutputOpenings {
        bytecode_ra: (0..dimensions.num_committed_ra_polys())
            .map(bytecode_ra)
            .collect(),
    }
}

pub fn read_raf_consistency_openings() -> [(JoltOpeningId, JoltOpeningId); 1] {
    [(unexpanded_pc_shift(), unexpanded_pc())]
}

pub(crate) fn stage1_claim<F>() -> JoltExpr<F>
where
    F: RingCore,
{
    let beta = challenge(BytecodeReadRafChallenge::Stage1Gamma);
    let mut claim =
        opening(unexpanded_pc_spartan_outer()) + beta.clone() * opening(imm_spartan_outer());

    for (i, flag) in CIRCUIT_FLAGS.into_iter().enumerate() {
        claim = claim + beta.clone().pow(i + 2) * opening(op_flag_spartan_outer(flag));
    }

    claim
}

pub(crate) fn stage2_claim<F>() -> JoltExpr<F>
where
    F: RingCore,
{
    let beta = challenge(BytecodeReadRafChallenge::Stage2Gamma);

    opening(op_flag_product(CircuitFlags::Jump))
        + beta.clone() * opening(instruction_flag_product(InstructionFlags::Branch))
        + beta.clone().pow(2) * opening(op_flag_product(CircuitFlags::WriteLookupOutputToRD))
        + beta.pow(3) * opening(op_flag_product(CircuitFlags::VirtualInstruction))
}

pub(crate) fn stage3_claim<F>() -> JoltExpr<F>
where
    F: RingCore,
{
    let beta = challenge(BytecodeReadRafChallenge::Stage3Gamma);

    opening(imm())
        + beta.clone() * opening(unexpanded_pc_shift())
        + beta.clone().pow(2)
            * opening(instruction_flag_input(
                InstructionFlags::LeftOperandIsRs1Value,
            ))
        + beta.clone().pow(3) * opening(instruction_flag_input(InstructionFlags::LeftOperandIsPC))
        + beta.clone().pow(4)
            * opening(instruction_flag_input(
                InstructionFlags::RightOperandIsRs2Value,
            ))
        + beta.clone().pow(5) * opening(instruction_flag_input(InstructionFlags::RightOperandIsImm))
        + beta.clone().pow(6) * opening(instruction_flag_shift(InstructionFlags::IsNoop))
        + beta.clone().pow(7) * opening(op_flag_shift(CircuitFlags::VirtualInstruction))
        + beta.pow(8) * opening(op_flag_shift(CircuitFlags::IsFirstInSequence))
}

pub(crate) fn stage4_claim<F>() -> JoltExpr<F>
where
    F: RingCore,
{
    let beta = challenge(BytecodeReadRafChallenge::Stage4Gamma);

    opening(rd_wa_read_write())
        + beta.clone() * opening(rs1_ra_read_write())
        + beta.pow(2) * opening(rs2_ra_read_write())
}

pub(crate) fn stage5_claim<F>() -> JoltExpr<F>
where
    F: RingCore,
{
    let beta = challenge(BytecodeReadRafChallenge::Stage5Gamma);
    let mut claim =
        opening(rd_wa_val_evaluation()) + beta.clone() * opening(instruction_raf_flag());

    for (i, table) in LookupTableKind::<XLEN>::iter().enumerate() {
        claim = claim + beta.clone().pow(i + 2) * opening(lookup_table_flag(table));
    }

    claim
}

fn bytecode_ra_product<F>(dimensions: BytecodeReadRafDimensions) -> JoltExpr<F>
where
    F: RingCore,
{
    let mut product = JoltExpr::one();
    for i in 0..dimensions.num_committed_ra_polys() {
        product = product * opening(bytecode_ra(i));
    }
    product
}

pub(crate) fn unexpanded_pc_spartan_outer() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::UnexpandedPC,
        JoltRelationId::SpartanOuter,
    )
}

pub(crate) fn imm_spartan_outer() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(JoltVirtualPolynomial::Imm, JoltRelationId::SpartanOuter)
}

fn op_flag_spartan_outer(flag: CircuitFlags) -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::OpFlags(flag),
        JoltRelationId::SpartanOuter,
    )
}

pub(crate) fn op_flag_product(flag: CircuitFlags) -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::OpFlags(flag),
        JoltRelationId::SpartanProductVirtualization,
    )
}

pub(crate) fn instruction_flag_product(flag: InstructionFlags) -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::InstructionFlags(flag),
        JoltRelationId::SpartanProductVirtualization,
    )
}

pub(crate) fn instruction_flag_input(flag: InstructionFlags) -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::InstructionFlags(flag),
        JoltRelationId::InstructionInputVirtualization,
    )
}

pub(crate) fn instruction_flag_shift(flag: InstructionFlags) -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::InstructionFlags(flag),
        JoltRelationId::SpartanShift,
    )
}

pub(crate) fn op_flag_shift(flag: CircuitFlags) -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::OpFlags(flag),
        JoltRelationId::SpartanShift,
    )
}

pub(crate) fn pc_spartan_outer() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(JoltVirtualPolynomial::PC, JoltRelationId::SpartanOuter)
}

pub fn bytecode_ra(index: usize) -> JoltOpeningId {
    JoltOpeningId::committed(
        JoltCommittedPolynomial::BytecodeRa(index),
        JoltRelationId::BytecodeReadRaf,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_poly::EqPolynomial;
    use jolt_riscv::{JoltInstructionKind, NormalizedOperands};

    #[test]
    fn read_raf_register_eq_evals_builds_register_address_tables() {
        let read_write = vec![Fr::from_u64(2), Fr::from_u64(3)];
        let val_evaluation = vec![Fr::from_u64(5), Fr::from_u64(7)];
        let eq = read_raf_register_eq_evals(&read_write, &val_evaluation);

        assert_eq!(
            eq,
            BytecodeReadRafRegisterEqEvals {
                read_write: EqPolynomial::<Fr>::evals(&read_write, None),
                val_evaluation: EqPolynomial::<Fr>::evals(&val_evaluation, None),
            }
        );
    }

    #[test]
    fn read_raf_stage_values_match_row_formula() {
        let bytecode = vec![
            JoltInstructionRow {
                instruction_kind: JoltInstructionKind::ADD,
                address: 9,
                operands: NormalizedOperands {
                    rs1: Some(1),
                    rs2: Some(2),
                    rd: Some(3),
                    imm: 4,
                },
                virtual_sequence_remaining: None,
                is_first_in_sequence: false,
                is_compressed: false,
            },
            JoltInstructionRow::default(),
        ];
        let register_read_write_point = vec![Fr::from_u64(2); 4];
        let register_val_evaluation_point = vec![Fr::from_u64(3); 4];
        let stage1_gammas = (0..2 + NUM_CIRCUIT_FLAGS)
            .map(|value| Fr::from_u64(value as u64 + 1))
            .collect::<Vec<_>>();
        let stage2_gammas = (0..4)
            .map(|value| Fr::from_u64(value as u64 + 11))
            .collect::<Vec<_>>();
        let stage3_gammas = (0..9)
            .map(|value| Fr::from_u64(value as u64 + 17))
            .collect::<Vec<_>>();
        let stage4_gammas = (0..3)
            .map(|value| Fr::from_u64(value as u64 + 29))
            .collect::<Vec<_>>();
        let stage5_gammas = (0..2 + LookupTableKind::<XLEN>::COUNT)
            .map(|value| Fr::from_u64(value as u64 + 37))
            .collect::<Vec<_>>();
        let register_eq =
            read_raf_register_eq_evals(&register_read_write_point, &register_val_evaluation_point);

        let stage_values = read_raf_stage_values(BytecodeReadRafStageValueInputs {
            bytecode: &bytecode,
            register_read_write_point: &register_read_write_point,
            register_val_evaluation_point: &register_val_evaluation_point,
            stage1_gammas: &stage1_gammas,
            stage2_gammas: &stage2_gammas,
            stage3_gammas: &stage3_gammas,
            stage4_gammas: &stage4_gammas,
            stage5_gammas: &stage5_gammas,
        });
        let expected = bytecode
            .iter()
            .map(|row| {
                read_raf_row_values(
                    row,
                    &register_eq.read_write,
                    &register_eq.val_evaluation,
                    &stage1_gammas,
                    &stage2_gammas,
                    &stage3_gammas,
                    &stage4_gammas,
                    &stage5_gammas,
                )
            })
            .collect::<Vec<_>>();

        assert_eq!(stage_values, expected);
    }
}
