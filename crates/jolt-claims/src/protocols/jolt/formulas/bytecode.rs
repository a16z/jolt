use jolt_field::{Field, RingCore};
use jolt_lookup_tables::{InstructionLookupTable, LookupTableKind, XLEN};
use jolt_poly::{EqPolynomial, IdentityPolynomial, MultilinearEvaluation};
use jolt_riscv::{
    instructions::Noop, CircuitFlags, Flags, InstructionFlags, InterleavedBitsMarker,
    JoltInstruction, JoltInstructionRow, CIRCUIT_FLAGS, NUM_CIRCUIT_FLAGS,
};

use crate::{challenge, opening, public};

use super::super::{
    BytecodeReadRafChallenge, BytecodeReadRafPublic, JoltChallengeId, JoltCommittedPolynomial,
    JoltConsistencyClaim, JoltExpr, JoltOpeningId, JoltPublicId, JoltRelationClaims,
    JoltRelationId, JoltVirtualPolynomial,
};
use super::dimensions::{JoltFormulaPointError, JoltSumcheckSpec};

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

    pub const fn sumcheck(self) -> JoltSumcheckSpec {
        JoltSumcheckSpec::boolean(self.log_t + self.log_k, self.committed_ra_polys + 1)
    }

    pub fn opening_point<F: Field>(
        self,
        challenges: &[F],
    ) -> Result<BytecodeReadRafOpeningPoint<F>, JoltFormulaPointError> {
        let expected = self.log_k + self.log_t;
        if challenges.len() != expected {
            return Err(JoltFormulaPointError::ChallengeLengthMismatch {
                expected,
                got: challenges.len(),
            });
        }

        let (r_address, r_cycle) = challenges.split_at(self.log_k);
        let r_address = r_address.iter().rev().copied().collect::<Vec<_>>();
        let r_cycle = r_cycle.iter().rev().copied().collect::<Vec<_>>();
        let opening_point = [r_address.as_slice(), r_cycle.as_slice()].concat();

        Ok(BytecodeReadRafOpeningPoint {
            r_address,
            r_cycle,
            opening_point,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BytecodeReadRafOpeningPoint<F: Field> {
    pub r_address: Vec<F>,
    pub r_cycle: Vec<F>,
    pub opening_point: Vec<F>,
}

pub fn read_raf<F>(dimensions: BytecodeReadRafDimensions) -> JoltRelationClaims<F>
where
    F: RingCore,
{
    let gamma = bytecode_challenge(BytecodeReadRafChallenge::Gamma);

    let input = gamma.clone().pow(7)
        + stage1_claim()
        + gamma.clone() * stage2_claim()
        + gamma.clone().pow(2) * stage3_claim()
        + gamma.clone().pow(3) * stage4_claim()
        + gamma.clone().pow(4) * stage5_claim::<F>()
        + gamma.clone().pow(5) * opening(pc_spartan_outer())
        + gamma.pow(6) * opening(pc_spartan_shift());

    let gamma = bytecode_challenge(BytecodeReadRafChallenge::Gamma);
    let output_coeff = bytecode_public(BytecodeReadRafPublic::StageValue(0))
        + gamma.clone() * bytecode_public(BytecodeReadRafPublic::StageValue(1))
        + gamma.clone().pow(2) * bytecode_public(BytecodeReadRafPublic::StageValue(2))
        + gamma.clone().pow(3) * bytecode_public(BytecodeReadRafPublic::StageValue(3))
        + gamma.clone().pow(4) * bytecode_public(BytecodeReadRafPublic::StageValue(4))
        + gamma.clone().pow(5) * bytecode_public(BytecodeReadRafPublic::SpartanOuterRaf)
        + gamma.clone().pow(6) * bytecode_public(BytecodeReadRafPublic::SpartanShiftRaf)
        + gamma.pow(7) * bytecode_public(BytecodeReadRafPublic::Entry);

    JoltRelationClaims::new(
        JoltRelationId::BytecodeReadRaf,
        dimensions.sumcheck(),
        input,
        output_coeff * bytecode_ra_product(dimensions),
    )
    .with_input_challenges([
        JoltChallengeId::from(BytecodeReadRafChallenge::Gamma),
        JoltChallengeId::from(BytecodeReadRafChallenge::Stage1Gamma),
        JoltChallengeId::from(BytecodeReadRafChallenge::Stage2Gamma),
        JoltChallengeId::from(BytecodeReadRafChallenge::Stage3Gamma),
        JoltChallengeId::from(BytecodeReadRafChallenge::Stage4Gamma),
        JoltChallengeId::from(BytecodeReadRafChallenge::Stage5Gamma),
    ])
    .with_consistency([JoltConsistencyClaim::same_evaluation(
        unexpanded_pc_spartan_shift(),
        unexpanded_pc_instruction_input(),
    )])
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BytecodeReadRafInputOpenings {
    pub spartan_outer: BytecodeReadRafSpartanOuterOpenings,
    pub spartan_product: BytecodeReadRafSpartanProductOpenings,
    pub instruction_input: BytecodeReadRafInstructionInputOpenings,
    pub spartan_shift: BytecodeReadRafSpartanShiftOpenings,
    pub registers_read_write: BytecodeReadRafRegistersReadWriteOpenings,
    pub registers_val_evaluation: BytecodeReadRafRegistersValEvaluationOpenings,
    pub spartan_outer_pc: JoltOpeningId,
    pub spartan_shift_pc: JoltOpeningId,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BytecodeReadRafSpartanOuterOpenings {
    pub unexpanded_pc: JoltOpeningId,
    pub imm: JoltOpeningId,
    pub op_flags: Vec<(CircuitFlags, JoltOpeningId)>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BytecodeReadRafSpartanProductOpenings {
    pub jump: JoltOpeningId,
    pub branch: JoltOpeningId,
    pub write_lookup_output_to_rd: JoltOpeningId,
    pub virtual_instruction: JoltOpeningId,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BytecodeReadRafInstructionInputOpenings {
    pub imm: JoltOpeningId,
    pub unexpanded_pc_from_shift: JoltOpeningId,
    pub left_operand_is_rs1_value: JoltOpeningId,
    pub left_operand_is_pc: JoltOpeningId,
    pub right_operand_is_rs2_value: JoltOpeningId,
    pub right_operand_is_imm: JoltOpeningId,
    pub is_noop_from_shift: JoltOpeningId,
    pub virtual_instruction_from_shift: JoltOpeningId,
    pub is_first_in_sequence_from_shift: JoltOpeningId,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BytecodeReadRafSpartanShiftOpenings {
    pub unexpanded_pc: JoltOpeningId,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BytecodeReadRafRegistersReadWriteOpenings {
    pub rd_wa: JoltOpeningId,
    pub rs1_ra: JoltOpeningId,
    pub rs2_ra: JoltOpeningId,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BytecodeReadRafRegistersValEvaluationOpenings {
    pub rd_wa: JoltOpeningId,
    pub instruction_raf_flag: JoltOpeningId,
    pub lookup_table_flags: Vec<(LookupTableKind<XLEN>, JoltOpeningId)>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BytecodeReadRafOutputOpenings {
    pub bytecode_ra: Vec<JoltOpeningId>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BytecodeReadRafPublicValues<F: Field> {
    pub stage_values: [F; 5],
    pub spartan_outer_raf: F,
    pub spartan_shift_raf: F,
    pub entry: F,
}

impl<F: Field> BytecodeReadRafPublicValues<F> {
    pub fn value(&self, id: BytecodeReadRafPublic) -> F {
        match id {
            BytecodeReadRafPublic::StageValue(index) => self
                .stage_values
                .get(index)
                .copied()
                .unwrap_or_else(F::zero),
            BytecodeReadRafPublic::SpartanOuterRaf => self.spartan_outer_raf,
            BytecodeReadRafPublic::SpartanShiftRaf => self.spartan_shift_raf,
            BytecodeReadRafPublic::Entry => self.entry,
        }
    }
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

pub fn read_raf_public_values<F>(
    inputs: BytecodeReadRafEvaluationInputs<'_, F>,
) -> Result<BytecodeReadRafPublicValues<F>, JoltFormulaPointError>
where
    F: Field,
{
    require_len(inputs.stage1_gammas, 2 + NUM_CIRCUIT_FLAGS)?;
    require_len(inputs.stage2_gammas, 4)?;
    require_len(inputs.stage3_gammas, 9)?;
    require_len(inputs.stage4_gammas, 3)?;
    require_len(inputs.stage5_gammas, 2 + LookupTableKind::<XLEN>::COUNT)?;

    let expected_domain = 1usize << inputs.r_address.len();
    if inputs.bytecode.len() != expected_domain {
        return Err(JoltFormulaPointError::EvaluationDomainLengthMismatch {
            expected: expected_domain,
            got: inputs.bytecode.len(),
        });
    }

    let address_eq_evals = EqPolynomial::<F>::evals(inputs.r_address, None);
    let register_read_write_eq = EqPolynomial::<F>::evals(inputs.register_read_write_point, None);
    let register_val_evaluation_eq =
        EqPolynomial::<F>::evals(inputs.register_val_evaluation_point, None);

    let mut stage_values = [F::zero(); 5];
    for (instruction, eq_address) in inputs.bytecode.iter().zip(address_eq_evals) {
        let row_values = read_raf_row_values::<F>(
            instruction,
            &register_read_write_eq,
            &register_val_evaluation_eq,
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

    for (stage_value, stage_cycle_point) in stage_values.iter_mut().zip(inputs.stage_cycle_points) {
        *stage_value *= EqPolynomial::<F>::mle(stage_cycle_point, inputs.r_cycle);
    }

    let identity = IdentityPolynomial::new(inputs.r_address.len()).evaluate(inputs.r_address);
    let spartan_outer_raf =
        identity * EqPolynomial::<F>::mle(inputs.stage_cycle_points[0], inputs.r_cycle);
    let spartan_shift_raf =
        identity * EqPolynomial::<F>::mle(inputs.stage_cycle_points[2], inputs.r_cycle);

    let entry_bits = (0..inputs.r_address.len())
        .map(|i| {
            F::from_u64(
                ((inputs.entry_bytecode_index >> (inputs.r_address.len() - 1 - i)) & 1) as u64,
            )
        })
        .collect::<Vec<_>>();
    let zero_cycle = vec![F::zero(); inputs.r_cycle.len()];
    let entry = EqPolynomial::<F>::mle(&entry_bits, inputs.r_address)
        * EqPolynomial::<F>::mle(&zero_cycle, inputs.r_cycle);

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
) -> [F; 5]
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

    [stage1, stage2, stage3, stage4, stage5]
}

fn register_eq<F: Field>(register: Option<u8>, eq: &[F]) -> F {
    register
        .and_then(|register| eq.get(register as usize))
        .copied()
        .unwrap_or_else(F::zero)
}

fn require_len<F>(values: &[F], expected: usize) -> Result<(), JoltFormulaPointError> {
    if values.len() < expected {
        return Err(JoltFormulaPointError::ChallengeLengthMismatch {
            expected,
            got: values.len(),
        });
    }
    Ok(())
}

pub fn read_raf_input_openings() -> BytecodeReadRafInputOpenings {
    BytecodeReadRafInputOpenings {
        spartan_outer: BytecodeReadRafSpartanOuterOpenings {
            unexpanded_pc: unexpanded_pc_spartan_outer(),
            imm: imm_spartan_outer(),
            op_flags: CIRCUIT_FLAGS
                .into_iter()
                .map(|flag| (flag, op_flag_spartan_outer(flag)))
                .collect(),
        },
        spartan_product: BytecodeReadRafSpartanProductOpenings {
            jump: op_flag_product(CircuitFlags::Jump),
            branch: instruction_flag_product(InstructionFlags::Branch),
            write_lookup_output_to_rd: op_flag_product(CircuitFlags::WriteLookupOutputToRD),
            virtual_instruction: op_flag_product(CircuitFlags::VirtualInstruction),
        },
        instruction_input: BytecodeReadRafInstructionInputOpenings {
            imm: imm_instruction_input(),
            unexpanded_pc_from_shift: unexpanded_pc_spartan_shift(),
            left_operand_is_rs1_value: instruction_flag_input(
                InstructionFlags::LeftOperandIsRs1Value,
            ),
            left_operand_is_pc: instruction_flag_input(InstructionFlags::LeftOperandIsPC),
            right_operand_is_rs2_value: instruction_flag_input(
                InstructionFlags::RightOperandIsRs2Value,
            ),
            right_operand_is_imm: instruction_flag_input(InstructionFlags::RightOperandIsImm),
            is_noop_from_shift: instruction_flag_shift(InstructionFlags::IsNoop),
            virtual_instruction_from_shift: op_flag_shift(CircuitFlags::VirtualInstruction),
            is_first_in_sequence_from_shift: op_flag_shift(CircuitFlags::IsFirstInSequence),
        },
        spartan_shift: BytecodeReadRafSpartanShiftOpenings {
            unexpanded_pc: unexpanded_pc_spartan_shift(),
        },
        registers_read_write: BytecodeReadRafRegistersReadWriteOpenings {
            rd_wa: rd_wa_read_write(),
            rs1_ra: rs1_ra_read_write(),
            rs2_ra: rs2_ra_read_write(),
        },
        registers_val_evaluation: BytecodeReadRafRegistersValEvaluationOpenings {
            rd_wa: rd_wa_val_evaluation(),
            instruction_raf_flag: instruction_raf_flag(),
            lookup_table_flags: LookupTableKind::<XLEN>::iter()
                .map(|table| (table, lookup_table_flag(table)))
                .collect(),
        },
        spartan_outer_pc: pc_spartan_outer(),
        spartan_shift_pc: pc_spartan_shift(),
    }
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
    [(
        unexpanded_pc_spartan_shift(),
        unexpanded_pc_instruction_input(),
    )]
}

fn stage1_claim<F>() -> JoltExpr<F>
where
    F: RingCore,
{
    let beta = bytecode_challenge(BytecodeReadRafChallenge::Stage1Gamma);
    let mut claim =
        opening(unexpanded_pc_spartan_outer()) + beta.clone() * opening(imm_spartan_outer());

    for (i, flag) in CIRCUIT_FLAGS.into_iter().enumerate() {
        claim = claim + beta.clone().pow(i + 2) * opening(op_flag_spartan_outer(flag));
    }

    claim
}

fn stage2_claim<F>() -> JoltExpr<F>
where
    F: RingCore,
{
    let beta = bytecode_challenge(BytecodeReadRafChallenge::Stage2Gamma);

    opening(op_flag_product(CircuitFlags::Jump))
        + beta.clone() * opening(instruction_flag_product(InstructionFlags::Branch))
        + beta.clone().pow(2) * opening(op_flag_product(CircuitFlags::WriteLookupOutputToRD))
        + beta.pow(3) * opening(op_flag_product(CircuitFlags::VirtualInstruction))
}

fn stage3_claim<F>() -> JoltExpr<F>
where
    F: RingCore,
{
    let beta = bytecode_challenge(BytecodeReadRafChallenge::Stage3Gamma);

    opening(imm_instruction_input())
        + beta.clone() * opening(unexpanded_pc_spartan_shift())
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

fn stage4_claim<F>() -> JoltExpr<F>
where
    F: RingCore,
{
    let beta = bytecode_challenge(BytecodeReadRafChallenge::Stage4Gamma);

    opening(rd_wa_read_write())
        + beta.clone() * opening(rs1_ra_read_write())
        + beta.pow(2) * opening(rs2_ra_read_write())
}

fn stage5_claim<F>() -> JoltExpr<F>
where
    F: RingCore,
{
    let beta = bytecode_challenge(BytecodeReadRafChallenge::Stage5Gamma);
    let mut claim =
        opening(rd_wa_val_evaluation()) + beta.clone() * opening(instruction_raf_flag());

    for (i, table) in LookupTableKind::<XLEN>::iter().enumerate() {
        claim = claim + beta.clone().pow(i + 2) * opening(lookup_table_flag(table));
    }

    claim
}

fn bytecode_challenge<F>(id: BytecodeReadRafChallenge) -> JoltExpr<F>
where
    F: RingCore,
{
    challenge(JoltChallengeId::from(id))
}

fn bytecode_public<F>(id: BytecodeReadRafPublic) -> JoltExpr<F>
where
    F: RingCore,
{
    public(JoltPublicId::from(id))
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

fn unexpanded_pc_spartan_outer() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::UnexpandedPC,
        JoltRelationId::SpartanOuter,
    )
}

fn imm_spartan_outer() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(JoltVirtualPolynomial::Imm, JoltRelationId::SpartanOuter)
}

fn op_flag_spartan_outer(flag: CircuitFlags) -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::OpFlags(flag),
        JoltRelationId::SpartanOuter,
    )
}

fn op_flag_product(flag: CircuitFlags) -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::OpFlags(flag),
        JoltRelationId::SpartanProductVirtualization,
    )
}

fn instruction_flag_product(flag: InstructionFlags) -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::InstructionFlags(flag),
        JoltRelationId::SpartanProductVirtualization,
    )
}

fn imm_instruction_input() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::Imm,
        JoltRelationId::InstructionInputVirtualization,
    )
}

fn unexpanded_pc_instruction_input() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::UnexpandedPC,
        JoltRelationId::InstructionInputVirtualization,
    )
}

fn unexpanded_pc_spartan_shift() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::UnexpandedPC,
        JoltRelationId::SpartanShift,
    )
}

fn instruction_flag_input(flag: InstructionFlags) -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::InstructionFlags(flag),
        JoltRelationId::InstructionInputVirtualization,
    )
}

fn instruction_flag_shift(flag: InstructionFlags) -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::InstructionFlags(flag),
        JoltRelationId::SpartanShift,
    )
}

fn op_flag_shift(flag: CircuitFlags) -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::OpFlags(flag),
        JoltRelationId::SpartanShift,
    )
}

fn rd_wa_read_write() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::RdWa,
        JoltRelationId::RegistersReadWriteChecking,
    )
}

fn rs1_ra_read_write() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::Rs1Ra,
        JoltRelationId::RegistersReadWriteChecking,
    )
}

fn rs2_ra_read_write() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::Rs2Ra,
        JoltRelationId::RegistersReadWriteChecking,
    )
}

fn rd_wa_val_evaluation() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::RdWa,
        JoltRelationId::RegistersValEvaluation,
    )
}

fn instruction_raf_flag() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::InstructionRafFlag,
        JoltRelationId::InstructionReadRaf,
    )
}

fn lookup_table_flag(table: LookupTableKind<XLEN>) -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::LookupTableFlag(table.index()),
        JoltRelationId::InstructionReadRaf,
    )
}

fn pc_spartan_outer() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(JoltVirtualPolynomial::PC, JoltRelationId::SpartanOuter)
}

fn pc_spartan_shift() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(JoltVirtualPolynomial::PC, JoltRelationId::SpartanShift)
}

fn bytecode_ra(index: usize) -> JoltOpeningId {
    JoltOpeningId::committed(
        JoltCommittedPolynomial::BytecodeRa(index),
        JoltRelationId::BytecodeReadRaf,
    )
}

#[cfg(test)]
#[expect(clippy::panic)]
mod tests {
    use super::*;
    use crate::protocols::jolt::{JoltCommittedPolynomial, JoltConsistencyClaim, JoltPolynomialId};
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_riscv::{JoltInstructionKind, NormalizedOperands};

    fn dimensions(num_committed_ra_polys: usize) -> BytecodeReadRafDimensions {
        BytecodeReadRafDimensions::new(5, 10, num_committed_ra_polys)
    }

    fn gamma_power(gamma: Fr, exponent: usize) -> Fr {
        let mut value = Fr::from_u64(1);
        for _ in 0..exponent {
            value *= gamma;
        }
        value
    }

    fn stage1_openings() -> Vec<JoltOpeningId> {
        let mut openings = vec![unexpanded_pc_spartan_outer(), imm_spartan_outer()];
        openings.extend(CIRCUIT_FLAGS.into_iter().map(op_flag_spartan_outer));
        openings
    }

    fn stage5_lookup_flags() -> Vec<JoltOpeningId> {
        LookupTableKind::<XLEN>::iter()
            .map(lookup_table_flag)
            .collect()
    }

    #[test]
    fn read_raf_opening_point_matches_core_order() {
        let point = BytecodeReadRafDimensions::new(3, 2, 1)
            .opening_point(&(1..=5).map(Fr::from_u64).collect::<Vec<_>>())
            .unwrap_or_else(|error| panic!("bytecode read-raf point should normalize: {error}"));

        assert_eq!(point.r_address, vec![Fr::from_u64(2), Fr::from_u64(1)]);
        assert_eq!(
            point.r_cycle,
            vec![Fr::from_u64(5), Fr::from_u64(4), Fr::from_u64(3)]
        );
        assert_eq!(
            point.opening_point,
            vec![
                Fr::from_u64(2),
                Fr::from_u64(1),
                Fr::from_u64(5),
                Fr::from_u64(4),
                Fr::from_u64(3),
            ]
        );
    }

    #[test]
    fn read_raf_helpers_expose_typed_openings() {
        let input = read_raf_input_openings();
        let output = read_raf_output_openings(dimensions(2));

        assert_eq!(
            input.spartan_outer.unexpanded_pc,
            unexpanded_pc_spartan_outer()
        );
        assert_eq!(input.spartan_outer.imm, imm_spartan_outer());
        assert_eq!(input.spartan_outer.op_flags.len(), NUM_CIRCUIT_FLAGS);
        assert_eq!(
            input.spartan_product.jump,
            op_flag_product(CircuitFlags::Jump)
        );
        assert_eq!(
            input.spartan_product.branch,
            instruction_flag_product(InstructionFlags::Branch)
        );
        assert_eq!(
            input.registers_val_evaluation.lookup_table_flags,
            LookupTableKind::<XLEN>::iter()
                .map(|table| (table, lookup_table_flag(table)))
                .collect::<Vec<_>>()
        );
        assert_eq!(output.bytecode_ra, vec![bytecode_ra(0), bytecode_ra(1)]);
        assert_eq!(
            read_raf_consistency_openings(),
            [(
                unexpanded_pc_spartan_shift(),
                unexpanded_pc_instruction_input()
            )]
        );
    }

    #[test]
    fn read_raf_public_values_evaluate_bytecode_rows() {
        let bytecode = vec![
            JoltInstructionRow {
                instruction_kind: JoltInstructionKind::ADD,
                address: 9,
                operands: NormalizedOperands {
                    rs1: Some(0),
                    rs2: Some(0),
                    rd: Some(0),
                    imm: 4,
                },
                virtual_sequence_remaining: None,
                is_first_in_sequence: false,
                is_compressed: false,
            },
            JoltInstructionRow::default(),
        ];
        let one = Fr::from_u64(1);
        let zero = Fr::from_u64(0);
        let r_address = [zero];
        let r_cycle = [zero];
        let stage_cycle_points = [&r_cycle[..]; 5];
        let stage1_gammas = vec![one; 2 + NUM_CIRCUIT_FLAGS];
        let stage5_gammas = vec![one; 2 + LookupTableKind::<XLEN>::COUNT];
        let public_values = read_raf_public_values::<Fr>(BytecodeReadRafEvaluationInputs {
            bytecode: &bytecode,
            r_address: &r_address,
            r_cycle: &r_cycle,
            stage_cycle_points,
            register_read_write_point: &[],
            register_val_evaluation_point: &[],
            entry_bytecode_index: 0,
            stage1_gammas: &stage1_gammas,
            stage2_gammas: &[one; 4],
            stage3_gammas: &[one; 9],
            stage4_gammas: &[one; 3],
            stage5_gammas: &stage5_gammas,
        })
        .unwrap_or_else(|error| panic!("bytecode public values should evaluate: {error}"));

        assert_eq!(
            public_values.stage_values,
            [
                Fr::from_u64(15),
                Fr::from_u64(1),
                Fr::from_u64(15),
                Fr::from_u64(3),
                Fr::from_u64(3),
            ]
        );
        assert_eq!(public_values.spartan_outer_raf, zero);
        assert_eq!(public_values.spartan_shift_raf, zero);
        assert_eq!(public_values.entry, one);
    }

    #[test]
    fn read_raf_supports_empty_ra_product() {
        let claims = read_raf::<Fr>(dimensions(0));

        assert!(!claims.output.required_openings.iter().any(|opening_id| {
            matches!(
                opening_id,
                JoltOpeningId::Polynomial {
                    polynomial: JoltPolynomialId::Committed(JoltCommittedPolynomial::BytecodeRa(_)),
                    ..
                }
            )
        }));
    }

    #[test]
    fn read_raf_exposes_expected_dependencies() {
        let dimensions = dimensions(2);
        let claims = read_raf::<Fr>(dimensions);

        let mut expected_input = stage1_openings();
        expected_input.extend([
            op_flag_product(CircuitFlags::Jump),
            instruction_flag_product(InstructionFlags::Branch),
            op_flag_product(CircuitFlags::WriteLookupOutputToRD),
            op_flag_product(CircuitFlags::VirtualInstruction),
            imm_instruction_input(),
            unexpanded_pc_spartan_shift(),
            instruction_flag_input(InstructionFlags::LeftOperandIsRs1Value),
            instruction_flag_input(InstructionFlags::LeftOperandIsPC),
            instruction_flag_input(InstructionFlags::RightOperandIsRs2Value),
            instruction_flag_input(InstructionFlags::RightOperandIsImm),
            instruction_flag_shift(InstructionFlags::IsNoop),
            op_flag_shift(CircuitFlags::VirtualInstruction),
            op_flag_shift(CircuitFlags::IsFirstInSequence),
            rd_wa_read_write(),
            rs1_ra_read_write(),
            rs2_ra_read_write(),
            rd_wa_val_evaluation(),
            instruction_raf_flag(),
        ]);
        expected_input.extend(stage5_lookup_flags());
        expected_input.extend([pc_spartan_outer(), pc_spartan_shift()]);

        assert_eq!(claims.id, JoltRelationId::BytecodeReadRaf);
        assert_eq!(claims.sumcheck, JoltSumcheckSpec::boolean(15, 3));
        assert_eq!(claims.input.required_openings, expected_input);
        assert_eq!(
            claims.output.required_openings,
            vec![bytecode_ra(0), bytecode_ra(1)]
        );
        assert_eq!(
            claims.required_challenges(),
            vec![
                JoltChallengeId::from(BytecodeReadRafChallenge::Gamma),
                JoltChallengeId::from(BytecodeReadRafChallenge::Stage1Gamma),
                JoltChallengeId::from(BytecodeReadRafChallenge::Stage2Gamma),
                JoltChallengeId::from(BytecodeReadRafChallenge::Stage3Gamma),
                JoltChallengeId::from(BytecodeReadRafChallenge::Stage4Gamma),
                JoltChallengeId::from(BytecodeReadRafChallenge::Stage5Gamma),
            ]
        );
        assert_eq!(
            claims.required_publics(),
            vec![
                JoltPublicId::from(BytecodeReadRafPublic::StageValue(0)),
                JoltPublicId::from(BytecodeReadRafPublic::StageValue(1)),
                JoltPublicId::from(BytecodeReadRafPublic::StageValue(2)),
                JoltPublicId::from(BytecodeReadRafPublic::StageValue(3)),
                JoltPublicId::from(BytecodeReadRafPublic::StageValue(4)),
                JoltPublicId::from(BytecodeReadRafPublic::SpartanOuterRaf),
                JoltPublicId::from(BytecodeReadRafPublic::SpartanShiftRaf),
                JoltPublicId::from(BytecodeReadRafPublic::Entry),
            ]
        );
        assert_eq!(
            claims.consistency,
            vec![JoltConsistencyClaim::same_evaluation(
                unexpanded_pc_spartan_shift(),
                unexpanded_pc_instruction_input(),
            )]
        );
        assert_eq!(claims.num_challenges(), 6);
    }

    #[test]
    fn read_raf_evaluates_like_core_formula() {
        let dimensions = dimensions(2);
        let claims = read_raf::<Fr>(dimensions);

        let gamma = Fr::from_u64(3);
        let stage1_gamma = Fr::from_u64(5);
        let stage2_gamma = Fr::from_u64(7);
        let stage3_gamma = Fr::from_u64(11);
        let stage4_gamma = Fr::from_u64(13);
        let stage5_gamma = Fr::from_u64(17);
        let zero = Fr::from_u64(0);

        let input = claims.input.expression().evaluate(
            |id| match *id {
                id if id == unexpanded_pc_spartan_outer() => Fr::from_u64(19),
                id if id == imm_spartan_outer() => Fr::from_u64(23),
                id if id == op_flag_product(CircuitFlags::Jump) => Fr::from_u64(29),
                id if id == instruction_flag_product(InstructionFlags::Branch) => Fr::from_u64(31),
                id if id == op_flag_product(CircuitFlags::WriteLookupOutputToRD) => {
                    Fr::from_u64(37)
                }
                id if id == op_flag_product(CircuitFlags::VirtualInstruction) => Fr::from_u64(41),
                id if id == imm_instruction_input() => Fr::from_u64(43),
                id if id == unexpanded_pc_spartan_shift() => Fr::from_u64(47),
                id if id == instruction_flag_input(InstructionFlags::LeftOperandIsRs1Value) => {
                    Fr::from_u64(53)
                }
                id if id == instruction_flag_input(InstructionFlags::LeftOperandIsPC) => {
                    Fr::from_u64(59)
                }
                id if id == instruction_flag_input(InstructionFlags::RightOperandIsRs2Value) => {
                    Fr::from_u64(61)
                }
                id if id == instruction_flag_input(InstructionFlags::RightOperandIsImm) => {
                    Fr::from_u64(67)
                }
                id if id == instruction_flag_shift(InstructionFlags::IsNoop) => Fr::from_u64(71),
                id if id == op_flag_shift(CircuitFlags::VirtualInstruction) => Fr::from_u64(73),
                id if id == op_flag_shift(CircuitFlags::IsFirstInSequence) => Fr::from_u64(79),
                id if id == rd_wa_read_write() => Fr::from_u64(83),
                id if id == rs1_ra_read_write() => Fr::from_u64(89),
                id if id == rs2_ra_read_write() => Fr::from_u64(97),
                id if id == rd_wa_val_evaluation() => Fr::from_u64(101),
                id if id == instruction_raf_flag() => Fr::from_u64(103),
                id if id == pc_spartan_outer() => Fr::from_u64(107),
                id if id == pc_spartan_shift() => Fr::from_u64(109),
                JoltOpeningId::Polynomial {
                    polynomial: JoltPolynomialId::Virtual(JoltVirtualPolynomial::OpFlags(flag)),
                    relation: JoltRelationId::SpartanOuter,
                } => Fr::from_u64(200 + u64::from(flag as u8)),
                JoltOpeningId::Polynomial {
                    polynomial:
                        JoltPolynomialId::Virtual(JoltVirtualPolynomial::LookupTableFlag(index)),
                    relation: JoltRelationId::InstructionReadRaf,
                } => Fr::from_u64(300 + index as u64),
                _ => zero,
            },
            |id| match *id {
                JoltChallengeId::BytecodeReadRaf(BytecodeReadRafChallenge::Gamma) => gamma,
                JoltChallengeId::BytecodeReadRaf(BytecodeReadRafChallenge::Stage1Gamma) => {
                    stage1_gamma
                }
                JoltChallengeId::BytecodeReadRaf(BytecodeReadRafChallenge::Stage2Gamma) => {
                    stage2_gamma
                }
                JoltChallengeId::BytecodeReadRaf(BytecodeReadRafChallenge::Stage3Gamma) => {
                    stage3_gamma
                }
                JoltChallengeId::BytecodeReadRaf(BytecodeReadRafChallenge::Stage4Gamma) => {
                    stage4_gamma
                }
                JoltChallengeId::BytecodeReadRaf(BytecodeReadRafChallenge::Stage5Gamma) => {
                    stage5_gamma
                }
                _ => zero,
            },
            |_| zero,
        );

        let mut stage1 = Fr::from_u64(19) + stage1_gamma * Fr::from_u64(23);
        for flag in CIRCUIT_FLAGS {
            stage1 += gamma_power(stage1_gamma, usize::from(flag as u8) + 2)
                * Fr::from_u64(200 + u64::from(flag as u8));
        }
        let stage2 = Fr::from_u64(29)
            + stage2_gamma * Fr::from_u64(31)
            + gamma_power(stage2_gamma, 2) * Fr::from_u64(37)
            + gamma_power(stage2_gamma, 3) * Fr::from_u64(41);
        let stage3 = Fr::from_u64(43)
            + stage3_gamma * Fr::from_u64(47)
            + gamma_power(stage3_gamma, 2) * Fr::from_u64(53)
            + gamma_power(stage3_gamma, 3) * Fr::from_u64(59)
            + gamma_power(stage3_gamma, 4) * Fr::from_u64(61)
            + gamma_power(stage3_gamma, 5) * Fr::from_u64(67)
            + gamma_power(stage3_gamma, 6) * Fr::from_u64(71)
            + gamma_power(stage3_gamma, 7) * Fr::from_u64(73)
            + gamma_power(stage3_gamma, 8) * Fr::from_u64(79);
        let stage4 = Fr::from_u64(83)
            + stage4_gamma * Fr::from_u64(89)
            + gamma_power(stage4_gamma, 2) * Fr::from_u64(97);
        let mut stage5 = Fr::from_u64(101) + stage5_gamma * Fr::from_u64(103);
        for table in LookupTableKind::<XLEN>::iter() {
            stage5 += gamma_power(stage5_gamma, table.index() + 2)
                * Fr::from_u64(300 + table.index() as u64);
        }

        assert_eq!(
            input,
            gamma_power(gamma, 7)
                + stage1
                + gamma * stage2
                + gamma_power(gamma, 2) * stage3
                + gamma_power(gamma, 3) * stage4
                + gamma_power(gamma, 4) * stage5
                + gamma_power(gamma, 5) * Fr::from_u64(107)
                + gamma_power(gamma, 6) * Fr::from_u64(109)
        );

        let stage_values = [
            Fr::from_u64(2),
            Fr::from_u64(3),
            Fr::from_u64(5),
            Fr::from_u64(7),
            Fr::from_u64(11),
        ];
        let spartan_outer_raf = Fr::from_u64(13);
        let spartan_shift_raf = Fr::from_u64(17);
        let entry = Fr::from_u64(19);
        let bytecode_ra_0 = Fr::from_u64(23);
        let bytecode_ra_1 = Fr::from_u64(29);

        let output = claims.output.expression().evaluate(
            |id| match *id {
                id if id == bytecode_ra(0) => bytecode_ra_0,
                id if id == bytecode_ra(1) => bytecode_ra_1,
                _ => zero,
            },
            |id| match *id {
                JoltChallengeId::BytecodeReadRaf(BytecodeReadRafChallenge::Gamma) => gamma,
                _ => zero,
            },
            |id| match *id {
                JoltPublicId::BytecodeReadRaf(BytecodeReadRafPublic::StageValue(index)) => {
                    stage_values[index]
                }
                JoltPublicId::BytecodeReadRaf(BytecodeReadRafPublic::SpartanOuterRaf) => {
                    spartan_outer_raf
                }
                JoltPublicId::BytecodeReadRaf(BytecodeReadRafPublic::SpartanShiftRaf) => {
                    spartan_shift_raf
                }
                JoltPublicId::BytecodeReadRaf(BytecodeReadRafPublic::Entry) => entry,
                _ => zero,
            },
        );

        assert_eq!(
            output,
            (stage_values[0]
                + gamma * stage_values[1]
                + gamma_power(gamma, 2) * stage_values[2]
                + gamma_power(gamma, 3) * stage_values[3]
                + gamma_power(gamma, 4) * stage_values[4]
                + gamma_power(gamma, 5) * spartan_outer_raf
                + gamma_power(gamma, 6) * spartan_shift_raf
                + gamma_power(gamma, 7) * entry)
                * bytecode_ra_0
                * bytecode_ra_1
        );
    }
}
