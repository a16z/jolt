use jolt_field::{JoltField, Ring};
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
use super::dimensions::PointGeometryError;
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

/// `challenge^exponent` as an expression leaf: the constant one, the
/// challenge itself, or the derived [`BytecodeReadRafPublic::ChallengePow`]
/// public for higher powers, so a fold over many powers stays one factor per
/// term instead of `exponent` repeated challenge factors.
pub(crate) fn challenge_pow_expr<F: Ring>(
    id: BytecodeReadRafChallenge,
    exponent: usize,
) -> JoltExpr<F> {
    match exponent {
        0 => JoltExpr::one(),
        1 => challenge(id),
        _ => derived(BytecodeReadRafPublic::ChallengePow {
            challenge: id,
            exponent,
        }),
    }
}

/// `γ^exponent` for the read-RAF batching challenge.
fn gamma_power_expr<F: Ring>(exponent: usize) -> JoltExpr<F> {
    challenge_pow_expr(BytecodeReadRafChallenge::Gamma, exponent)
}

/// The value behind [`BytecodeReadRafPublic::ChallengePow`].
pub fn challenge_pow<F: Ring>(value: F, exponent: usize) -> F {
    super::claim_reductions::hamming_weight::gamma_pow(value, exponent)
}

/// One past the largest exponent the read-RAF expressions raise each
/// challenge to, for `num_val_stages` val stages: the verifier registers the
/// [`BytecodeReadRafPublic::ChallengePow`] publics `2..bound` per challenge.
pub fn challenge_power_bounds(num_val_stages: usize) -> [(BytecodeReadRafChallenge, usize); 6] {
    let [stage1, stage2, stage3, stage4, stage5] = BYTECODE_STAGE_GAMMA_COUNTS;
    [
        (BytecodeReadRafChallenge::Gamma, num_val_stages + 3),
        (BytecodeReadRafChallenge::Stage1Gamma, stage1),
        (BytecodeReadRafChallenge::Stage2Gamma, stage2),
        (BytecodeReadRafChallenge::Stage3Gamma, stage3),
        (BytecodeReadRafChallenge::Stage4Gamma, stage4),
        (BytecodeReadRafChallenge::Stage5Gamma, stage5),
    ]
}

/// The staged input fold shared by the bytecode read-RAF monolith and its
/// address phase: the five base staged claims at `γ^0..4`, then one raw claim
/// per `extra_stage_claims` entry at the following powers (the lattice
/// fused-inc consumer stages), then the Spartan outer/shift PC openings and
/// the constant entry term at the next three powers.
pub(crate) fn read_raf_address_input_fold<F>(extra_stage_claims: Vec<JoltExpr<F>>) -> JoltExpr<F>
where
    F: Ring,
{
    let base_stages = BYTECODE_STAGE_GAMMA_COUNTS.len();
    let num_val_stages = base_stages + extra_stage_claims.len();

    let mut fold = gamma_power_expr(num_val_stages + 2)
        + stage1_claim()
        + gamma_power_expr(1) * stage2_claim()
        + gamma_power_expr(2) * stage3_claim()
        + gamma_power_expr(3) * stage4_claim()
        + gamma_power_expr(4) * stage5_claim::<F>();
    for (index, claim) in extra_stage_claims.into_iter().enumerate() {
        fold = fold + gamma_power_expr(base_stages + index) * claim;
    }
    fold + gamma_power_expr(num_val_stages) * opening(pc_spartan_outer())
        + gamma_power_expr(num_val_stages + 1) * opening(pc_shift())
}

pub(crate) fn read_raf_cycle_output<F>(
    dimensions: BytecodeReadRafDimensions,
    num_val_stages: usize,
) -> JoltExpr<F>
where
    F: Ring,
{
    let mut output_coeff = JoltExpr::zero();
    for stage in 0..num_val_stages {
        output_coeff = output_coeff
            + gamma_power_expr(stage) * derived(BytecodeReadRafPublic::StageValue(stage));
    }
    output_coeff = output_coeff
        + gamma_power_expr(num_val_stages) * derived(BytecodeReadRafPublic::SpartanOuterRaf)
        + gamma_power_expr(num_val_stages + 1) * derived(BytecodeReadRafPublic::SpartanShiftRaf)
        + gamma_power_expr(num_val_stages + 2) * derived(BytecodeReadRafPublic::Entry);

    output_coeff * bytecode_ra_product(dimensions)
}

pub(crate) fn read_raf_cycle_output_committed<F>(
    dimensions: BytecodeReadRafDimensions,
    num_val_stages: usize,
) -> JoltExpr<F>
where
    F: Ring,
{
    // The staged Val factor multiplies after the RA product so the lowered
    // R1CS auxiliary chain matches core's `[ra..., val_stage]` factor order.
    let mut output = JoltExpr::zero();
    for stage in 0..num_val_stages {
        output = output
            + gamma_power_expr(stage)
                * derived(BytecodeReadRafPublic::StageCycleEq(stage))
                * bytecode_ra_product(dimensions)
                * opening(super::claim_reductions::bytecode::bytecode_val_stage_opening(stage));
    }
    let raf_coeff = gamma_power_expr(num_val_stages)
        * derived(BytecodeReadRafPublic::SpartanOuterRaf)
        + gamma_power_expr(num_val_stages + 1) * derived(BytecodeReadRafPublic::SpartanShiftRaf)
        + gamma_power_expr(num_val_stages + 2) * derived(BytecodeReadRafPublic::Entry);

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
/// `BytecodeValClaim` *wire* count — the fused stages resolve through the
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
    F: Ring,
{
    let base_stages = BYTECODE_STAGE_GAMMA_COUNTS.len();
    let num_val_stages = base_stages + LATTICE_FUSED_INC_STAGES;
    let mut base_coeff = JoltExpr::zero();
    for stage in 0..base_stages {
        base_coeff = base_coeff
            + gamma_power_expr(stage) * derived(BytecodeReadRafPublic::StageValue(stage));
    }
    base_coeff = base_coeff
        + gamma_power_expr(num_val_stages) * derived(BytecodeReadRafPublic::SpartanOuterRaf)
        + gamma_power_expr(num_val_stages + 1) * derived(BytecodeReadRafPublic::SpartanShiftRaf)
        + gamma_power_expr(num_val_stages + 2) * derived(BytecodeReadRafPublic::Entry);

    let mut fused_coeff = JoltExpr::zero();
    for stage in base_stages..num_val_stages {
        fused_coeff = fused_coeff
            + gamma_power_expr(stage) * derived(BytecodeReadRafPublic::StageValue(stage));
    }

    (base_coeff + fused_coeff * opening(fused_inc_read_raf_opening()))
        * bytecode_ra_product(dimensions)
}

/// Lattice committed-mode cycle output: the base five stages fold their staged
/// `BytecodeValClaim` openings; the four fused stages reuse the staged *store*
/// val (index [`BYTECODE_STAGE_GAMMA_COUNTS`]`.len()`) — directly for the two
/// RAM legs, complemented (`1 − store`) for the two register legs — so the
/// staged-val wire set is unchanged from the store-claim design.
pub(crate) fn read_raf_cycle_output_committed_lattice<F>(
    dimensions: BytecodeReadRafDimensions,
) -> JoltExpr<F>
where
    F: Ring,
{
    let base_stages = BYTECODE_STAGE_GAMMA_COUNTS.len();
    let num_val_stages = base_stages + LATTICE_FUSED_INC_STAGES;
    let ra_product = bytecode_ra_product(dimensions);
    let fused = opening(fused_inc_read_raf_opening());
    let store = opening(super::claim_reductions::bytecode::bytecode_val_stage_opening(base_stages));

    let mut output = JoltExpr::zero();
    for stage in 0..base_stages {
        output = output
            + gamma_power_expr(stage)
                * derived(BytecodeReadRafPublic::StageCycleEq(stage))
                * ra_product.clone()
                * opening(super::claim_reductions::bytecode::bytecode_val_stage_opening(stage));
    }
    let store_pair = gamma_power_expr(base_stages)
        * derived(BytecodeReadRafPublic::StageCycleEq(base_stages))
        + gamma_power_expr(base_stages + 1)
            * derived(BytecodeReadRafPublic::StageCycleEq(base_stages + 1));
    let notstore_pair = gamma_power_expr(base_stages + 2)
        * derived(BytecodeReadRafPublic::StageCycleEq(base_stages + 2))
        + gamma_power_expr(base_stages + 3)
            * derived(BytecodeReadRafPublic::StageCycleEq(base_stages + 3));
    output = output
        + store_pair * ra_product.clone() * fused.clone() * store.clone()
        + notstore_pair.clone() * ra_product.clone() * fused.clone()
        - notstore_pair * ra_product.clone() * fused * store;

    let raf_coeff = gamma_power_expr(num_val_stages)
        * derived(BytecodeReadRafPublic::SpartanOuterRaf)
        + gamma_power_expr(num_val_stages + 1) * derived(BytecodeReadRafPublic::SpartanShiftRaf)
        + gamma_power_expr(num_val_stages + 2) * derived(BytecodeReadRafPublic::Entry);

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
pub struct BytecodeReadRafPublicValues<F: JoltField> {
    pub stage_values: [F; 5],
    pub spartan_outer_raf: F,
    pub spartan_shift_raf: F,
    pub entry: F,
}

impl<F: JoltField> BytecodeReadRafPublicValues<F> {
    /// Returns `None` for committed-mode publics (`StageCycleEq`) and
    /// out-of-range stage indices so a wrong-mode formula fails loudly at the
    /// source instead of evaluating with a silently zeroed term.
    pub fn value(&self, id: BytecodeReadRafPublic) -> Option<F> {
        match id {
            BytecodeReadRafPublic::StageValue(index) => self.stage_values.get(index).copied(),
            BytecodeReadRafPublic::StageCycleEq(_) | BytecodeReadRafPublic::ChallengePow { .. } => {
                None
            }
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
pub struct BytecodeReadRafCommittedPublicValues<F: JoltField> {
    pub stage_cycle_eqs: [F; READ_RAF_CYCLE_STAGES],
    pub spartan_outer_raf: F,
    pub spartan_shift_raf: F,
    pub entry: F,
}

impl<F: JoltField> BytecodeReadRafCommittedPublicValues<F> {
    /// Returns `None` for full-mode publics (`StageValue`) and out-of-range
    /// stage indices so a wrong-mode formula fails loudly at the source
    /// instead of evaluating with a silently zeroed term.
    pub fn value(&self, id: BytecodeReadRafPublic) -> Option<F> {
        match id {
            BytecodeReadRafPublic::StageValue(_) | BytecodeReadRafPublic::ChallengePow { .. } => {
                None
            }
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
    F: JoltField,
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
    F: JoltField,
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
    F: JoltField,
{
    BytecodeReadRafRegisterEqEvals {
        read_write: EqPolynomial::<F>::evals(register_read_write_point, None),
        val_evaluation: EqPolynomial::<F>::evals(register_val_evaluation_point, None),
    }
}

impl<F: JoltField> BytecodeReadRafRegisterEqEvals<F> {
    fn weighted_read_write(&self, gammas: &[F]) -> [Vec<F>; 3] {
        std::array::from_fn(|i| self.read_write.iter().map(|eq| *eq * gammas[i]).collect())
    }
}

/// Every bytecode row's staged values: the five gamma-folded stages, plus
/// (lattice) the store circuit flag as the sixth staged value, folded like
/// the others by the read-raf consumers.
pub fn read_raf_stage_values<F>(
    inputs: BytecodeReadRafStageValueInputs<'_, F>,
) -> Vec<[F; NUM_BYTECODE_VAL_STAGES]>
where
    F: JoltField,
{
    let register_eq = read_raf_register_eq_evals(
        inputs.register_read_write_point,
        inputs.register_val_evaluation_point,
    );
    let weighted_read_write = register_eq.weighted_read_write(inputs.stage4_gammas);
    let gammas = stage_gammas(&inputs);
    inputs
        .bytecode
        .iter()
        .map(|instruction| {
            read_raf_row_values::<F>(
                instruction,
                &weighted_read_write,
                &register_eq.val_evaluation,
                &gammas,
            )
        })
        .collect()
}

pub fn read_raf_public_values<F>(
    inputs: BytecodeReadRafEvaluationInputs<'_, F>,
) -> Result<BytecodeReadRafPublicValues<F>, PointGeometryError>
where
    F: JoltField,
{
    require_len(inputs.stage1_gammas, BYTECODE_STAGE_GAMMA_COUNTS[0])?;
    require_len(inputs.stage2_gammas, BYTECODE_STAGE_GAMMA_COUNTS[1])?;
    require_len(inputs.stage3_gammas, BYTECODE_STAGE_GAMMA_COUNTS[2])?;
    require_len(inputs.stage4_gammas, BYTECODE_STAGE_GAMMA_COUNTS[3])?;
    require_len(inputs.stage5_gammas, BYTECODE_STAGE_GAMMA_COUNTS[4])?;

    let expected_domain = 1usize << inputs.r_address.len();
    if inputs.bytecode.len() != expected_domain {
        return Err(PointGeometryError::EvaluationDomainLengthMismatch {
            expected: expected_domain,
            got: inputs.bytecode.len(),
        });
    }

    let address_eq_evals = EqPolynomial::<F>::evals(inputs.r_address, None);
    let folded = read_raf_folded_stage_values(
        BytecodeReadRafStageValueInputs {
            bytecode: inputs.bytecode,
            register_read_write_point: inputs.register_read_write_point,
            register_val_evaluation_point: inputs.register_val_evaluation_point,
            stage1_gammas: inputs.stage1_gammas,
            stage2_gammas: inputs.stage2_gammas,
            stage3_gammas: inputs.stage3_gammas,
            stage4_gammas: inputs.stage4_gammas,
            stage5_gammas: inputs.stage5_gammas,
        },
        &address_eq_evals,
    );
    // The base monolith publics carry the five gamma'd stages only; the
    // lattice sixth (store) staged value never flows through this path.
    let mut stage_values: [F; 5] = std::array::from_fn(|stage| folded[stage]);

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

/// Indices into [`BYTECODE_STAGE_GAMMA_COUNTS`] of the stages whose gamma
/// terms [`read_raf_flag_terms`] reports.
const STAGE1: usize = 0;
const STAGE2: usize = 1;
const STAGE3: usize = 2;
const STAGE5: usize = 4;

/// Decode one row's flags, report each gamma they select as `(stage, index)`,
/// and return its store flag.
///
/// With `γ_s` the stage-`s` gamma powers and each sum over the indices
/// reported for that stage, a row's stage values are
/// - stage 1: `address + γ_1[1]·imm + Σ γ_1[k]`,
/// - stage 2: `Σ γ_2[k]`,
/// - stage 3: `imm + γ_3[1]·address + Σ γ_3[k]`,
/// - stage 4: the stage-4-weighted read-write register eq at `rd`, `rs1`, `rs2`,
/// - stage 5: `eq_val(rd) + Σ γ_5[k]`,
///
/// plus (lattice) the store flag. Everything this reports depends only on the
/// row's [`JoltInstructionRow::flag_class`].
fn read_raf_flag_terms(
    instruction: &JoltInstructionRow,
    mut gamma: impl FnMut(usize, usize),
) -> bool {
    let decoded = JoltInstruction::try_from(*instruction)
        .unwrap_or(JoltInstruction::Noop(Noop(*instruction)));
    let circuit_flags = decoded.circuit_flags();
    let instruction_flags = decoded.instruction_flags();

    for (index, flag) in CIRCUIT_FLAGS.into_iter().enumerate() {
        if circuit_flags[flag] {
            gamma(STAGE1, index + 2);
        }
    }
    for (selected, index) in [
        (circuit_flags[CircuitFlags::Jump], 0),
        (instruction_flags[InstructionFlags::Branch], 1),
        (circuit_flags[CircuitFlags::WriteLookupOutputToRD], 2),
        (circuit_flags[CircuitFlags::VirtualInstruction], 3),
    ] {
        if selected {
            gamma(STAGE2, index);
        }
    }
    for (selected, index) in [
        (
            instruction_flags[InstructionFlags::LeftOperandIsRs1Value],
            2,
        ),
        (instruction_flags[InstructionFlags::LeftOperandIsPC], 3),
        (
            instruction_flags[InstructionFlags::RightOperandIsRs2Value],
            4,
        ),
        (instruction_flags[InstructionFlags::RightOperandIsImm], 5),
        (instruction_flags[InstructionFlags::IsNoop], 6),
        (circuit_flags[CircuitFlags::VirtualInstruction], 7),
        (circuit_flags[CircuitFlags::IsFirstInSequence], 8),
    ] {
        if selected {
            gamma(STAGE3, index);
        }
    }
    if !circuit_flags.is_interleaved_operands() {
        gamma(STAGE5, 1);
    }
    if let Some(table) = InstructionLookupTable::<XLEN>::lookup_table(&decoded) {
        gamma(STAGE5, 2 + table.index());
    }
    circuit_flags[CircuitFlags::Store]
}

/// Dense indices for the distinct [`JoltInstructionRow::flag_class`]es of a
/// bytecode table, in first-seen order (open addressing over the class key).
/// Slots and keys are word-sized: a Jolt guest expands sub-word stores and
/// loads into multi-row sequences.
struct FlagClasses {
    slots: Vec<usize>,
    keys: Vec<usize>,
}

impl FlagClasses {
    fn with_rows(rows: usize) -> Self {
        Self {
            slots: vec![usize::MAX; (2 * rows).next_power_of_two().max(16)],
            keys: Vec::new(),
        }
    }

    /// The index of `key`'s class, and whether this call created it.
    fn index(&mut self, key: u32) -> (usize, bool) {
        let key = key as usize;
        let mask = self.slots.len() - 1;
        let mut slot = key.wrapping_mul(0x9e37_79b9) & mask;
        loop {
            match self.slots[slot] {
                usize::MAX => {
                    let class = self.keys.len();
                    self.keys.push(key);
                    self.slots[slot] = class;
                    return (class, true);
                }
                class if self.keys[class] == key => return (class, false),
                _ => slot = (slot + 1) & mask,
            }
        }
    }
}

fn read_raf_row_values<F>(
    instruction: &JoltInstructionRow,
    register_read_write_eq: &[Vec<F>; 3],
    register_val_evaluation_eq: &[F],
    gammas: &[&[F]; 5],
) -> [F; NUM_BYTECODE_VAL_STAGES]
where
    F: JoltField,
{
    let mut sums = [F::zero(); 5];
    let is_store = read_raf_flag_terms(instruction, |stage, index| {
        sums[stage] += gammas[stage][index];
    });
    let operands = instruction.integer_operands();
    assemble_stage_values(
        F::from_u64(instruction.address as u64),
        F::from_i128(instruction.operands.imm),
        sums,
        gammas,
        register_eq(operands.rd, &register_read_write_eq[0])
            + register_eq(operands.rs1, &register_read_write_eq[1])
            + register_eq(operands.rs2, &register_read_write_eq[2]),
        register_eq(operands.rd, register_val_evaluation_eq),
        F::from_u64(u64::from(is_store)),
    )
}

/// The staged values from an address, immediate, and per-stage gamma sums,
/// the stage-4 register term, the stage-5 `rd` value-evaluation term, and the
/// store flag: one row's, or their `address_eq`-weighted sums over rows.
fn assemble_stage_values<F: JoltField>(
    address: F,
    imm: F,
    gamma_sums: [F; 5],
    gammas: &[&[F]; 5],
    stage4: F,
    stage5_register: F,
    store: F,
) -> [F; NUM_BYTECODE_VAL_STAGES] {
    let stage1 = address + gammas[STAGE1][1] * imm + gamma_sums[STAGE1];
    let stage2 = gamma_sums[STAGE2];
    let stage3 = imm + gammas[STAGE3][1] * address + gamma_sums[STAGE3];
    let stage5 = stage5_register + gamma_sums[STAGE5];

    #[cfg(not(feature = "akita"))]
    {
        let _ = store;
        [stage1, stage2, stage3, stage4, stage5]
    }
    // The lattice sixth stage: the store circuit flag as a raw staged value
    // (the `IncVirtualization` destination selector), folded against
    // `eq(r_address)` like the five gamma'd stages by the read-raf consumers.
    #[cfg(feature = "akita")]
    {
        [stage1, stage2, stage3, stage4, stage5, store]
    }
}

/// The per-stage gamma slices, with stage 4's register weights folded in
/// separately.
fn stage_gammas<'a, F>(inputs: &BytecodeReadRafStageValueInputs<'a, F>) -> [&'a [F]; 5] {
    [
        inputs.stage1_gammas,
        inputs.stage2_gammas,
        inputs.stage3_gammas,
        &[],
        inputs.stage5_gammas,
    ]
}

/// `Σ_r address_eq[r] · stage_values(r)` over the bytecode rows, folded
/// column-first: each row adds its eq weight into one bucket per selected
/// gamma and per register operand, and the buckets meet the gammas once. This
/// equals summing [`read_raf_stage_values`] weighted by `address_eq`, with
/// additions per row instead of one field multiplication per stage.
pub fn read_raf_folded_stage_values<F>(
    inputs: BytecodeReadRafStageValueInputs<'_, F>,
    address_eq: &[F],
) -> [F; NUM_BYTECODE_VAL_STAGES]
where
    F: JoltField,
{
    let register_eq = read_raf_register_eq_evals(
        inputs.register_read_write_point,
        inputs.register_val_evaluation_point,
    );
    let registers = register_eq.read_write.len();
    let mut gamma_buckets: [Vec<F>; 5] =
        std::array::from_fn(|stage| vec![F::zero(); BYTECODE_STAGE_GAMMA_COUNTS[stage]]);
    let mut register_buckets: [Vec<F>; 3] = std::array::from_fn(|_| vec![F::zero(); registers]);
    let mut store = F::zero();
    let rows = inputs.bytecode.len().min(address_eq.len());
    let mut addresses = Vec::with_capacity(rows);
    let mut imms = Vec::with_capacity(rows);
    let mut classes = FlagClasses::with_rows(rows);
    let mut class_weights: Vec<F> = Vec::new();
    let mut class_rows: Vec<&JoltInstructionRow> = Vec::new();
    for (instruction, &eq) in inputs.bytecode.iter().zip(address_eq) {
        let (class, created) = classes.index(instruction.flag_class());
        if created {
            class_weights.push(F::zero());
            class_rows.push(instruction);
        }
        class_weights[class] += eq;
        let operands = instruction.integer_operands();
        for (bucket, register) in
            register_buckets
                .iter_mut()
                .zip([operands.rd, operands.rs1, operands.rs2])
        {
            if let Some(slot) = register.and_then(|register| bucket.get_mut(register as usize)) {
                *slot += eq;
            }
        }
        addresses.push(F::from_u64(instruction.address as u64));
        imms.push(F::from_i128(instruction.operands.imm));
    }
    for (instruction, &weight) in class_rows.iter().zip(&class_weights) {
        if read_raf_flag_terms(instruction, |stage, index| {
            gamma_buckets[stage][index] += weight;
        }) {
            store += weight;
        }
    }
    let eq = &address_eq[..rows];
    let gammas = stage_gammas(&inputs);
    let weighted_read_write = register_eq.weighted_read_write(inputs.stage4_gammas);
    assemble_stage_values(
        F::dot_product(&addresses, eq),
        F::dot_product(&imms, eq),
        std::array::from_fn(|stage| F::dot_product(gammas[stage], &gamma_buckets[stage])),
        &gammas,
        weighted_read_write
            .iter()
            .zip(&register_buckets)
            .map(|(weights, bucket)| F::dot_product(weights, bucket))
            .fold(F::zero(), |sum, term| sum + term),
        F::dot_product(&register_eq.val_evaluation, &register_buckets[0]),
        store,
    )
}

fn register_eq<F: JoltField>(register: Option<u8>, eq: &[F]) -> F {
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
    F: Ring,
{
    let beta = challenge(BytecodeReadRafChallenge::Stage1Gamma);
    let mut claim =
        opening(unexpanded_pc_spartan_outer()) + beta.clone() * opening(imm_spartan_outer());

    for (i, flag) in CIRCUIT_FLAGS.into_iter().enumerate() {
        claim = claim
            + challenge_pow_expr(BytecodeReadRafChallenge::Stage1Gamma, i + 2)
                * opening(op_flag_spartan_outer(flag));
    }

    claim
}

pub(crate) fn stage2_claim<F>() -> JoltExpr<F>
where
    F: Ring,
{
    let beta = challenge(BytecodeReadRafChallenge::Stage2Gamma);

    opening(op_flag_product(CircuitFlags::Jump))
        + beta.clone() * opening(instruction_flag_product(InstructionFlags::Branch))
        + challenge_pow_expr(BytecodeReadRafChallenge::Stage2Gamma, 2)
            * opening(op_flag_product(CircuitFlags::WriteLookupOutputToRD))
        + challenge_pow_expr(BytecodeReadRafChallenge::Stage2Gamma, 3)
            * opening(op_flag_product(CircuitFlags::VirtualInstruction))
}

pub(crate) fn stage3_claim<F>() -> JoltExpr<F>
where
    F: Ring,
{
    let beta = challenge(BytecodeReadRafChallenge::Stage3Gamma);

    opening(imm())
        + beta.clone() * opening(unexpanded_pc_shift())
        + challenge_pow_expr(BytecodeReadRafChallenge::Stage3Gamma, 2)
            * opening(instruction_flag_input(
                InstructionFlags::LeftOperandIsRs1Value,
            ))
        + challenge_pow_expr(BytecodeReadRafChallenge::Stage3Gamma, 3)
            * opening(instruction_flag_input(InstructionFlags::LeftOperandIsPC))
        + challenge_pow_expr(BytecodeReadRafChallenge::Stage3Gamma, 4)
            * opening(instruction_flag_input(
                InstructionFlags::RightOperandIsRs2Value,
            ))
        + challenge_pow_expr(BytecodeReadRafChallenge::Stage3Gamma, 5)
            * opening(instruction_flag_input(InstructionFlags::RightOperandIsImm))
        + challenge_pow_expr(BytecodeReadRafChallenge::Stage3Gamma, 6)
            * opening(instruction_flag_shift(InstructionFlags::IsNoop))
        + challenge_pow_expr(BytecodeReadRafChallenge::Stage3Gamma, 7)
            * opening(op_flag_shift(CircuitFlags::VirtualInstruction))
        + challenge_pow_expr(BytecodeReadRafChallenge::Stage3Gamma, 8)
            * opening(op_flag_shift(CircuitFlags::IsFirstInSequence))
}

pub(crate) fn stage4_claim<F>() -> JoltExpr<F>
where
    F: Ring,
{
    let beta = challenge(BytecodeReadRafChallenge::Stage4Gamma);

    opening(rd_wa_read_write())
        + beta.clone() * opening(rs1_ra_read_write())
        + challenge_pow_expr(BytecodeReadRafChallenge::Stage4Gamma, 2)
            * opening(rs2_ra_read_write())
}

pub(crate) fn stage5_claim<F>() -> JoltExpr<F>
where
    F: Ring,
{
    let beta = challenge(BytecodeReadRafChallenge::Stage5Gamma);
    let mut claim =
        opening(rd_wa_val_evaluation()) + beta.clone() * opening(instruction_raf_flag());

    for (i, table) in LookupTableKind::<XLEN>::iter().enumerate() {
        claim = claim
            + challenge_pow_expr(BytecodeReadRafChallenge::Stage5Gamma, i + 2)
                * opening(lookup_table_flag(table));
    }

    claim
}

fn bytecode_ra_product<F>(dimensions: BytecodeReadRafDimensions) -> JoltExpr<F>
where
    F: Ring,
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
    use jolt_field::{Fr, Ring};
    use jolt_riscv::{JoltInstructionKind, NormalizedOperands};
    use std::collections::HashMap;

    /// The folded evaluation groups rows by `flag_class` and reads each
    /// class's flags off its first row, so equal classes must yield equal
    /// gamma terms and store flags whatever the operands and address.
    #[test]
    fn flag_class_determines_read_raf_flag_terms() {
        let operands = [
            NormalizedOperands::default(),
            NormalizedOperands {
                rs1: Some(1),
                rs2: Some(2),
                rd: Some(0),
                imm: -7,
            },
            NormalizedOperands {
                rs1: None,
                rs2: Some(31),
                rd: Some(5),
                imm: 1 << 40,
            },
        ];
        let mut terms_by_class = HashMap::new();
        for &instruction_kind in JoltInstructionKind::ALL {
            for virtual_sequence_remaining in [None, Some(0), Some(3)] {
                for is_compressed in [false, true] {
                    for is_first_in_sequence in [false, true] {
                        for (address, operands) in operands.into_iter().enumerate() {
                            let row = JoltInstructionRow {
                                instruction_kind,
                                address: 4 * address + 8,
                                operands,
                                virtual_sequence_remaining,
                                is_first_in_sequence,
                                is_compressed,
                            };
                            let mut hits = Vec::new();
                            let store =
                                read_raf_flag_terms(&row, |stage, index| hits.push((stage, index)));
                            let terms = (hits, store);
                            let first = terms_by_class
                                .entry(row.flag_class())
                                .or_insert_with(|| terms.clone());
                            assert_eq!(*first, terms, "{row:?}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn folded_stage_values_match_weighted_row_values() {
        let row = |instruction_kind, address, rs1, rs2, rd, imm| JoltInstructionRow {
            instruction_kind,
            address,
            operands: NormalizedOperands { rs1, rs2, rd, imm },
            virtual_sequence_remaining: None,
            is_first_in_sequence: false,
            is_compressed: false,
        };
        // Repeated classes with different operands, empty registers, and a
        // store, so every bucket of the fold sees more than one row.
        let bytecode = vec![
            row(JoltInstructionKind::ADD, 9, Some(1), Some(2), Some(3), 4),
            row(JoltInstructionKind::ADD, 13, Some(7), Some(0), Some(15), -5),
            row(JoltInstructionKind::SD, 17, Some(2), Some(9), None, 16),
            row(JoltInstructionKind::SD, 21, Some(4), Some(4), None, -8),
            JoltInstructionRow::default(),
            row(
                JoltInstructionKind::ADDI,
                25,
                Some(3),
                None,
                Some(3),
                1 << 33,
            ),
        ];
        let gammas = |count: usize, offset: u64| {
            (0..count)
                .map(|value| Fr::from_u64(value as u64 + offset))
                .collect::<Vec<_>>()
        };
        let stage1_gammas = gammas(2 + NUM_CIRCUIT_FLAGS, 1);
        let stage2_gammas = gammas(4, 11);
        let stage3_gammas = gammas(9, 17);
        let stage4_gammas = gammas(3, 29);
        let stage5_gammas = gammas(2 + LookupTableKind::<XLEN>::COUNT, 37);
        let register_read_write_point = gammas(4, 43);
        let register_val_evaluation_point = gammas(4, 53);
        let inputs = || BytecodeReadRafStageValueInputs {
            bytecode: &bytecode,
            register_read_write_point: &register_read_write_point,
            register_val_evaluation_point: &register_val_evaluation_point,
            stage1_gammas: &stage1_gammas,
            stage2_gammas: &stage2_gammas,
            stage3_gammas: &stage3_gammas,
            stage4_gammas: &stage4_gammas,
            stage5_gammas: &stage5_gammas,
        };
        let address_eq = gammas(bytecode.len(), 61);

        let mut expected = [Fr::from_u64(0); NUM_BYTECODE_VAL_STAGES];
        for (values, eq) in read_raf_stage_values(inputs()).into_iter().zip(&address_eq) {
            for (sum, value) in expected.iter_mut().zip(values) {
                *sum += *eq * value;
            }
        }
        assert_eq!(
            read_raf_folded_stage_values(inputs(), &address_eq),
            expected
        );
    }
}
