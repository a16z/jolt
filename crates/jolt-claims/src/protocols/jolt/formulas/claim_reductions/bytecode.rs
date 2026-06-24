//! Two-phase committed-bytecode claim reduction (stage 6b cycle -> stage 7
//! address).
//!
//! In committed program mode the bytecode value columns are committed as dense
//! chunk polynomials instead of being evaluated directly by the verifier. The
//! address phase of bytecode read-RAF stages five `BytecodeValStage(i)` claims;
//! this reduction batches them with powers of `eta` and reduces the batch into
//! openings of `BytecodeChunk(i)` over the shared precommitted schedule.
//! Mirrors `jolt-prover-legacy`'s `zkvm/claim_reductions/bytecode.rs` and the
//! committed-bytecode geometry of `zkvm/bytecode/chunks.rs`.

use jolt_field::{Field, RingCore};
use jolt_lookup_tables::{LookupTableKind, XLEN};
use jolt_poly::EqPolynomial;
use jolt_riscv::{CircuitFlags, InstructionFlags, NUM_CIRCUIT_FLAGS, NUM_INSTRUCTION_FLAGS};

use crate::{challenge, opening, public};

use super::super::super::{
    BytecodeClaimReductionChallenge, BytecodeClaimReductionPublic, JoltChallengeId,
    JoltCommittedPolynomial, JoltExpr, JoltOpeningId, JoltPublicId, JoltRelationId,
    JoltVirtualPolynomial,
};
use super::super::dimensions::{
    log2_power_of_two, CommitmentMatrixShape, TracePolynomialOrder, REGISTER_ADDRESS_BITS,
};
use super::super::error::{require_len, require_opening_point_len, JoltFormulaPointError};
use super::precommitted::{
    precommitted_skip_round_scale, PrecommittedClaimReduction, PrecommittedReductionDimensions,
    PrecommittedReductionLayout, PrecommittedSchedulingReference,
};

/// Number of staged `BytecodeValStage(i)` claims batched into the reduction.
pub const NUM_BYTECODE_VAL_STAGES: usize = 5;

const REGISTER_COUNT: usize = 1 << REGISTER_ADDRESS_BITS;

/// Total number of lanes encoded by committed-bytecode rows.
pub const fn total_lanes() -> usize {
    3 * REGISTER_COUNT
        + 2
        + NUM_CIRCUIT_FLAGS
        + NUM_INSTRUCTION_FLAGS
        + LookupTableKind::<XLEN>::COUNT
        + 1
}

/// Fixed lane capacity for committed bytecode rows.
pub const COMMITTED_BYTECODE_LANE_CAPACITY: usize = total_lanes().next_power_of_two();

#[inline(always)]
pub const fn committed_lanes() -> usize {
    COMMITTED_BYTECODE_LANE_CAPACITY
}

pub const fn committed_lane_vars() -> usize {
    COMMITTED_BYTECODE_LANE_CAPACITY.trailing_zeros() as usize
}

/// Maximum chunk count representable by the `u8` proof serialization of
/// `BytecodeChunk(i)`.
pub const MAX_COMMITTED_BYTECODE_CHUNK_COUNT: usize = 256;

/// Committed bytecode chunking is valid when the chunk count is a nonzero
/// power of two no larger than [`MAX_COMMITTED_BYTECODE_CHUNK_COUNT`] that
/// divides the power-of-two bytecode length.
///
/// Deliberately stricter than core's same-named predicate: core leaves
/// `bytecode_len` unchecked because preprocessing pads it to a power of two,
/// while the chunk-size log derivations here rely on that invariant
/// explicitly.
#[inline(always)]
pub fn is_valid_committed_bytecode_chunking_for_len(
    bytecode_len: usize,
    chunk_count: usize,
) -> bool {
    is_valid_chunk_count(chunk_count)
        && bytecode_len.is_power_of_two()
        && bytecode_len.is_multiple_of(chunk_count)
}

/// Chunk-count half of the chunking rules, shared with the formula
/// constructors that validate a chunk count without the bytecode length.
const fn is_valid_chunk_count(chunk_count: usize) -> bool {
    chunk_count > 0
        && chunk_count <= MAX_COMMITTED_BYTECODE_CHUNK_COUNT
        && chunk_count.is_power_of_two()
}

/// Lane offsets of the committed bytecode row encoding. One-hot `rs1`/`rs2`/
/// `rd` blocks are followed by scalar unexpanded-PC and immediate lanes, the
/// circuit and instruction flag blocks, the lookup-table selector block, and
/// the RAF flag.
#[derive(Clone, Copy, Debug)]
pub struct BytecodeLaneLayout {
    pub rs1_start: usize,
    pub rs2_start: usize,
    pub rd_start: usize,
    pub unexp_pc_idx: usize,
    pub imm_idx: usize,
    pub circuit_start: usize,
    pub instr_start: usize,
    pub lookup_start: usize,
    pub raf_flag_idx: usize,
}

impl BytecodeLaneLayout {
    pub const fn new() -> Self {
        let rs1_start = 0usize;
        let rs2_start = rs1_start + REGISTER_COUNT;
        let rd_start = rs2_start + REGISTER_COUNT;
        let unexp_pc_idx = rd_start + REGISTER_COUNT;
        let imm_idx = unexp_pc_idx + 1;
        let circuit_start = imm_idx + 1;
        let instr_start = circuit_start + NUM_CIRCUIT_FLAGS;
        let lookup_start = instr_start + NUM_INSTRUCTION_FLAGS;
        let raf_flag_idx = lookup_start + LookupTableKind::<XLEN>::COUNT;
        Self {
            rs1_start,
            rs2_start,
            rd_start,
            unexp_pc_idx,
            imm_idx,
            circuit_start,
            instr_start,
            lookup_start,
            raf_flag_idx,
        }
    }
}

impl Default for BytecodeLaneLayout {
    fn default() -> Self {
        Self::new()
    }
}

pub const BYTECODE_LANE_LAYOUT: BytecodeLaneLayout = BytecodeLaneLayout::new();

/// Total-var count of one committed bytecode chunk polynomial, used as this
/// reduction's candidate in the shared precommitted scheduling reference.
pub fn precommitted_candidate(
    bytecode_len: usize,
    chunk_count: usize,
) -> Result<usize, JoltFormulaPointError> {
    if !is_valid_committed_bytecode_chunking_for_len(bytecode_len, chunk_count) {
        return Err(JoltFormulaPointError::InvalidBytecodeChunking {
            bytecode_len,
            chunk_count,
        });
    }
    Ok(committed_lane_vars() + log2_power_of_two(bytecode_len / chunk_count))
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BytecodeClaimReductionLayout {
    chunk_shape: CommitmentMatrixShape,
    precommitted: PrecommittedClaimReduction,
    trace_order: TracePolynomialOrder,
    chunk_count: usize,
    log_bytecode_chunk_size: usize,
    dropped_address_bits: usize,
}

impl BytecodeClaimReductionLayout {
    pub fn balanced(
        trace_order: TracePolynomialOrder,
        log_t: usize,
        scheduling_reference: PrecommittedSchedulingReference,
        bytecode_len: usize,
        chunk_count: usize,
    ) -> Result<Self, JoltFormulaPointError> {
        if !is_valid_committed_bytecode_chunking_for_len(bytecode_len, chunk_count) {
            return Err(JoltFormulaPointError::InvalidBytecodeChunking {
                bytecode_len,
                chunk_count,
            });
        }
        let log_bytecode_chunk_size = log2_power_of_two(bytecode_len / chunk_count);
        let dropped_address_bits = log2_power_of_two(bytecode_len) - log_bytecode_chunk_size;
        // Bytecode chunks use their own balanced dimensions (independent from
        // the main trace matrix); stage 8 embeds them as a top-left block in
        // the joint opening.
        let chunk_shape =
            CommitmentMatrixShape::balanced(committed_lane_vars() + log_bytecode_chunk_size);
        let precommitted = PrecommittedClaimReduction::new(
            chunk_shape.row_vars(),
            chunk_shape.column_vars(),
            scheduling_reference,
            trace_order,
            log_t,
        )?;
        Ok(Self {
            chunk_shape,
            precommitted,
            trace_order,
            chunk_count,
            log_bytecode_chunk_size,
            dropped_address_bits,
        })
    }

    pub const fn chunk_shape(&self) -> CommitmentMatrixShape {
        self.chunk_shape
    }

    pub const fn chunk_count(&self) -> usize {
        self.chunk_count
    }

    pub const fn log_bytecode_chunk_size(&self) -> usize {
        self.log_bytecode_chunk_size
    }

    /// Split the full bytecode address point (the `BytecodeReadRafAddrClaim`
    /// opening point) into per-chunk eq weights over the dropped high bits and
    /// the chunk-local cycle point shared by all chunks.
    pub fn split_address_point<F: Field>(
        &self,
        r_bc_full: &[F],
    ) -> Result<BytecodeAddressPoint<F>, JoltFormulaPointError> {
        let expected = self.dropped_address_bits + self.log_bytecode_chunk_size;
        if r_bc_full.len() != expected {
            return Err(JoltFormulaPointError::OpeningPointLengthMismatch {
                expected,
                got: r_bc_full.len(),
            });
        }
        let chunk_rbc_weights = if self.dropped_address_bits == 0 {
            vec![F::one()]
        } else {
            EqPolynomial::<F>::evals(&r_bc_full[..self.dropped_address_bits], None)
        };
        debug_assert_eq!(chunk_rbc_weights.len(), self.chunk_count);
        Ok(BytecodeAddressPoint {
            chunk_rbc_weights,
            r_bc: r_bc_full[self.dropped_address_bits..].to_vec(),
        })
    }

    /// `ChunkOutputWeight(i)` values when the reduction completes in the cycle
    /// phase (no active address-phase rounds remain).
    pub fn cycle_phase_final_output_weights<F: Field>(
        &self,
        inputs: BytecodeOutputWeightInputs<'_, F>,
        challenges: &[F],
    ) -> Result<Vec<F>, JoltFormulaPointError> {
        let opening_point = self
            .precommitted
            .cycle_phase_permuted_opening_point(challenges)?;
        let scale = self.eq_combined(&inputs, &opening_point)?
            * self.precommitted.cycle_phase_skip_scale::<F>();
        self.chunk_output_weights(inputs.chunk_rbc_weights, scale)
    }

    /// `ChunkOutputWeight(i)` values from the reduction's already-derived
    /// cycle-phase opening point, rather than re-deriving it from the sumcheck
    /// challenges. Lets the cycle-phase relation object's `resolve_public`
    /// recover the weights from the opening point it produced in
    /// `derive_opening_points`.
    pub fn cycle_phase_final_output_weights_at_opening_point<F: Field>(
        &self,
        inputs: BytecodeOutputWeightInputs<'_, F>,
        opening_point: &[F],
    ) -> Result<Vec<F>, JoltFormulaPointError> {
        let permuted = self
            .precommitted
            .cycle_phase_permuted_from_opening_point(opening_point)?;
        let scale =
            self.eq_combined(&inputs, &permuted)? * self.precommitted.cycle_phase_skip_scale::<F>();
        self.chunk_output_weights(inputs.chunk_rbc_weights, scale)
    }

    /// `ChunkOutputWeight(i)` values when the reduction completes in the
    /// address phase.
    pub fn address_phase_final_output_weights<F: Field>(
        &self,
        inputs: BytecodeOutputWeightInputs<'_, F>,
        cycle_var_challenges: &[F],
        challenges: &[F],
    ) -> Result<Vec<F>, JoltFormulaPointError> {
        let opening_point = self
            .precommitted
            .address_phase_opening_point(cycle_var_challenges, challenges)?;
        self.address_phase_final_output_weights_at_opening_point(inputs, &opening_point)
    }

    /// `ChunkOutputWeight(i)` values from the reduction's already-derived
    /// address-phase opening point, rather than re-deriving it from the
    /// cycle/sumcheck challenges. Lets the stage 7 relation object's
    /// `resolve_public` recover the weights from the opening point it produced in
    /// `derive_opening_points`.
    pub fn address_phase_final_output_weights_at_opening_point<F: Field>(
        &self,
        inputs: BytecodeOutputWeightInputs<'_, F>,
        opening_point: &[F],
    ) -> Result<Vec<F>, JoltFormulaPointError> {
        let scale = self.eq_combined(&inputs, opening_point)?
            * precommitted_skip_round_scale::<F>(&self.precommitted);
        self.chunk_output_weights(inputs.chunk_rbc_weights, scale)
    }

    /// Evaluate the gamma-weighted lane selector against the chunk opening
    /// point: `(sum_lane lane_weights[lane] * eq(r_lane)[lane]) * eq(r_cycle,
    /// r_bc)`, with the lane/cycle split determined by the trace layout.
    fn eq_combined<F: Field>(
        &self,
        inputs: &BytecodeOutputWeightInputs<'_, F>,
        opening_point: &[F],
    ) -> Result<F, JoltFormulaPointError> {
        let lane_vars = committed_lane_vars();
        let expected = lane_vars + self.log_bytecode_chunk_size;
        if opening_point.len() != expected {
            return Err(JoltFormulaPointError::OpeningPointLengthMismatch {
                expected,
                got: opening_point.len(),
            });
        }
        if inputs.r_bc.len() != self.log_bytecode_chunk_size {
            return Err(JoltFormulaPointError::OpeningPointLengthMismatch {
                expected: self.log_bytecode_chunk_size,
                got: inputs.r_bc.len(),
            });
        }
        if inputs.lane_weights.len() != committed_lanes() {
            return Err(JoltFormulaPointError::EvaluationDomainLengthMismatch {
                expected: committed_lanes(),
                got: inputs.lane_weights.len(),
            });
        }

        let (lane_challenges, cycle_challenges) = match self.trace_order {
            TracePolynomialOrder::CycleMajor => opening_point.split_at(lane_vars),
            TracePolynomialOrder::AddressMajor => {
                let (cycle, lane) = opening_point.split_at(self.log_bytecode_chunk_size);
                (lane, cycle)
            }
        };

        let eq_cycle = EqPolynomial::<F>::mle(cycle_challenges, inputs.r_bc);
        let eq_lane = EqPolynomial::<F>::evals(lane_challenges, None);
        let lane_weight_eval = inputs
            .lane_weights
            .iter()
            .zip(eq_lane)
            .map(|(weight, eq)| *weight * eq)
            .sum::<F>();

        Ok(lane_weight_eval * eq_cycle)
    }

    fn chunk_output_weights<F: Field>(
        &self,
        chunk_rbc_weights: &[F],
        scale: F,
    ) -> Result<Vec<F>, JoltFormulaPointError> {
        if chunk_rbc_weights.len() != self.chunk_count {
            return Err(JoltFormulaPointError::EvaluationDomainLengthMismatch {
                expected: self.chunk_count,
                got: chunk_rbc_weights.len(),
            });
        }
        Ok(chunk_rbc_weights
            .iter()
            .map(|weight| *weight * scale)
            .collect())
    }
}

impl PrecommittedReductionLayout for BytecodeClaimReductionLayout {
    fn precommitted(&self) -> &PrecommittedClaimReduction {
        &self.precommitted
    }
}

/// Per-chunk eq weights over the dropped high bytecode address bits, plus the
/// chunk-local cycle point.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BytecodeAddressPoint<F> {
    pub chunk_rbc_weights: Vec<F>,
    pub r_bc: Vec<F>,
}

/// Stage-6b inputs to the final committed-bytecode output weights.
pub struct BytecodeOutputWeightInputs<'a, F> {
    pub r_bc: &'a [F],
    pub chunk_rbc_weights: &'a [F],
    pub lane_weights: &'a [F],
}

/// Inputs to [`lane_weights`]; the gamma vectors are the same per-stage gamma
/// powers consumed by `bytecode::read_raf_public_values`. The register points
/// are the full `RdWa` opening points from registers read-write checking and
/// registers val evaluation; only their leading `REGISTER_ADDRESS_BITS`
/// address components are read (unlike `BytecodeReadRafEvaluationInputs`,
/// whose same-named fields take the address components alone).
pub struct BytecodeLaneWeightInputs<'a, F> {
    pub eta: F,
    pub stage1_gammas: &'a [F],
    pub stage2_gammas: &'a [F],
    pub stage3_gammas: &'a [F],
    pub stage4_gammas: &'a [F],
    pub stage5_gammas: &'a [F],
    pub register_read_write_point: &'a [F],
    pub register_val_evaluation_point: &'a [F],
}

/// Fold the five staged bytecode read-RAF combinations into one weight per
/// committed lane, so `sum_lane weights[lane] * lane_value(row, lane)` equals
/// `sum_stage eta^stage * stage_value(row)` for every bytecode row.
pub fn lane_weights<F: Field>(
    inputs: BytecodeLaneWeightInputs<'_, F>,
) -> Result<Vec<F>, JoltFormulaPointError> {
    require_len(inputs.stage1_gammas, 2 + NUM_CIRCUIT_FLAGS)?;
    require_len(inputs.stage2_gammas, 4)?;
    require_len(inputs.stage3_gammas, 9)?;
    require_len(inputs.stage4_gammas, 3)?;
    require_len(inputs.stage5_gammas, 2 + LookupTableKind::<XLEN>::COUNT)?;
    require_opening_point_len(inputs.register_read_write_point, REGISTER_ADDRESS_BITS)?;
    require_opening_point_len(inputs.register_val_evaluation_point, REGISTER_ADDRESS_BITS)?;

    let mut eta_powers = [F::one(); NUM_BYTECODE_VAL_STAGES];
    for stage in 1..NUM_BYTECODE_VAL_STAGES {
        eta_powers[stage] = eta_powers[stage - 1] * inputs.eta;
    }

    let layout = BYTECODE_LANE_LAYOUT;
    debug_assert_eq!(layout.raf_flag_idx + 1, total_lanes());

    let register_read_write_eq = EqPolynomial::<F>::evals(
        &inputs.register_read_write_point[..REGISTER_ADDRESS_BITS],
        None,
    );
    let register_val_evaluation_eq = EqPolynomial::<F>::evals(
        &inputs.register_val_evaluation_point[..REGISTER_ADDRESS_BITS],
        None,
    );

    let mut weights = vec![F::zero(); committed_lanes()];

    {
        let coeff = eta_powers[0];
        let g = inputs.stage1_gammas;
        weights[layout.unexp_pc_idx] += coeff * g[0];
        weights[layout.imm_idx] += coeff * g[1];
        for i in 0..NUM_CIRCUIT_FLAGS {
            weights[layout.circuit_start + i] += coeff * g[2 + i];
        }
    }
    {
        let coeff = eta_powers[1];
        let g = inputs.stage2_gammas;
        weights[layout.circuit_start + (CircuitFlags::Jump as usize)] += coeff * g[0];
        weights[layout.instr_start + (InstructionFlags::Branch as usize)] += coeff * g[1];
        weights[layout.circuit_start + (CircuitFlags::WriteLookupOutputToRD as usize)] +=
            coeff * g[2];
        weights[layout.circuit_start + (CircuitFlags::VirtualInstruction as usize)] += coeff * g[3];
    }
    {
        let coeff = eta_powers[2];
        let g = inputs.stage3_gammas;
        weights[layout.imm_idx] += coeff * g[0];
        weights[layout.unexp_pc_idx] += coeff * g[1];
        weights[layout.instr_start + (InstructionFlags::LeftOperandIsRs1Value as usize)] +=
            coeff * g[2];
        weights[layout.instr_start + (InstructionFlags::LeftOperandIsPC as usize)] += coeff * g[3];
        weights[layout.instr_start + (InstructionFlags::RightOperandIsRs2Value as usize)] +=
            coeff * g[4];
        weights[layout.instr_start + (InstructionFlags::RightOperandIsImm as usize)] +=
            coeff * g[5];
        weights[layout.instr_start + (InstructionFlags::IsNoop as usize)] += coeff * g[6];
        weights[layout.circuit_start + (CircuitFlags::VirtualInstruction as usize)] += coeff * g[7];
        weights[layout.circuit_start + (CircuitFlags::IsFirstInSequence as usize)] += coeff * g[8];
    }
    {
        let coeff = eta_powers[3];
        let g = inputs.stage4_gammas;
        for r in 0..REGISTER_COUNT {
            weights[layout.rd_start + r] += coeff * g[0] * register_read_write_eq[r];
            weights[layout.rs1_start + r] += coeff * g[1] * register_read_write_eq[r];
            weights[layout.rs2_start + r] += coeff * g[2] * register_read_write_eq[r];
        }
    }
    {
        let coeff = eta_powers[4];
        let g = inputs.stage5_gammas;
        for r in 0..REGISTER_COUNT {
            weights[layout.rd_start + r] += coeff * g[0] * register_val_evaluation_eq[r];
        }
        weights[layout.raf_flag_idx] += coeff * g[1];
        for i in 0..LookupTableKind::<XLEN>::COUNT {
            weights[layout.lookup_start + i] += coeff * g[2 + i];
        }
    }

    Ok(weights)
}

pub(crate) fn final_output_expr<F>(chunk_count: usize) -> JoltExpr<F>
where
    F: RingCore,
{
    let mut output = JoltExpr::zero();
    for chunk_idx in 0..chunk_count {
        output = output
            + public(JoltPublicId::from(
                BytecodeClaimReductionPublic::ChunkOutputWeight(chunk_idx),
            )) * opening(final_bytecode_chunk_opening(chunk_idx));
    }
    output
}

pub fn cycle_phase_output_openings(
    dimensions: PrecommittedReductionDimensions,
    chunk_count: usize,
) -> Vec<JoltOpeningId> {
    assert_valid_chunk_count(chunk_count);
    if dimensions.has_address_phase() {
        vec![cycle_phase_intermediate_opening()]
    } else {
        (0..chunk_count).map(final_bytecode_chunk_opening).collect()
    }
}

pub fn bytecode_val_stage_opening(stage: usize) -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::BytecodeValStage(stage),
        JoltRelationId::BytecodeReadRaf,
    )
}

pub fn cycle_phase_intermediate_opening() -> JoltOpeningId {
    JoltOpeningId::virtual_polynomial(
        JoltVirtualPolynomial::BytecodeClaimReductionIntermediate,
        JoltRelationId::BytecodeClaimReductionCyclePhase,
    )
}

pub fn final_bytecode_chunk_opening(chunk_idx: usize) -> JoltOpeningId {
    JoltOpeningId::committed(
        JoltCommittedPolynomial::BytecodeChunk(chunk_idx),
        JoltRelationId::BytecodeClaimReduction,
    )
}

pub(crate) fn bytecode_challenge<F>(id: BytecodeClaimReductionChallenge) -> JoltExpr<F>
where
    F: RingCore,
{
    challenge(JoltChallengeId::from(id))
}

/// Backstop for the formula constructors that take a raw chunk count without
/// the bytecode length needed for full chunking validation; layouts are the
/// validated source of this value.
pub(crate) fn assert_valid_chunk_count(chunk_count: usize) {
    assert!(
        is_valid_chunk_count(chunk_count),
        "bytecode chunk count ({chunk_count}) must be a nonzero power of two at most \
         {MAX_COMMITTED_BYTECODE_CHUNK_COUNT}"
    );
}

#[cfg(test)]
mod tests {
    #![expect(clippy::panic, reason = "tests fail loudly on unexpected errors")]

    use super::super::super::bytecode::{read_raf_public_values, BytecodeReadRafEvaluationInputs};
    use super::*;
    use crate::protocols::jolt::JoltPolynomialId;
    use jolt_field::{Fr, FromPrimitiveInt, Invertible};
    use jolt_lookup_tables::InstructionLookupTable;
    use jolt_riscv::{
        instructions::Noop, Flags, InterleavedBitsMarker, JoltInstruction, JoltInstructionKind,
        JoltInstructionRow, NormalizedOperands, CIRCUIT_FLAGS,
    };

    const INSTRUCTION_FLAG_ORDER: [InstructionFlags; NUM_INSTRUCTION_FLAGS] = [
        InstructionFlags::LeftOperandIsPC,
        InstructionFlags::RightOperandIsImm,
        InstructionFlags::LeftOperandIsRs1Value,
        InstructionFlags::RightOperandIsRs2Value,
        InstructionFlags::Branch,
        InstructionFlags::IsNoop,
    ];

    fn fr(value: u64) -> Fr {
        Fr::from_u64(value)
    }

    fn gamma_powers(base: u64, count: usize) -> Vec<Fr> {
        let base = fr(base);
        let mut powers = Vec::with_capacity(count);
        let mut power = fr(1);
        for _ in 0..count {
            powers.push(power);
            power *= base;
        }
        powers
    }

    /// Sparse `(lane, value)` encoding of one committed bytecode row, mirroring
    /// core's `for_each_active_lane_value`.
    fn lane_values(instruction: &JoltInstructionRow) -> Vec<(usize, Fr)> {
        let decoded = JoltInstruction::try_from(*instruction)
            .unwrap_or(JoltInstruction::Noop(Noop(*instruction)));
        let circuit_flags = decoded.circuit_flags();
        let instruction_flags = decoded.instruction_flags();
        let layout = BYTECODE_LANE_LAYOUT;
        let mut values = Vec::new();

        if let Some(register) = instruction.operands.rs1 {
            values.push((layout.rs1_start + register as usize, fr(1)));
        }
        if let Some(register) = instruction.operands.rs2 {
            values.push((layout.rs2_start + register as usize, fr(1)));
        }
        if let Some(register) = instruction.operands.rd {
            values.push((layout.rd_start + register as usize, fr(1)));
        }
        values.push((layout.unexp_pc_idx, fr(instruction.address as u64)));
        values.push((layout.imm_idx, Fr::from_i128(instruction.operands.imm)));
        for (index, flag) in CIRCUIT_FLAGS.into_iter().enumerate() {
            assert_eq!(index, flag as usize);
            if circuit_flags[flag] {
                values.push((layout.circuit_start + index, fr(1)));
            }
        }
        for (index, flag) in INSTRUCTION_FLAG_ORDER.into_iter().enumerate() {
            assert_eq!(index, flag as usize);
            if instruction_flags[flag] {
                values.push((layout.instr_start + index, fr(1)));
            }
        }
        if let Some(table) = InstructionLookupTable::<XLEN>::lookup_table(&decoded) {
            values.push((layout.lookup_start + table.index(), fr(1)));
        }
        if !circuit_flags.is_interleaved_operands() {
            values.push((layout.raf_flag_idx, fr(1)));
        }

        values
    }

    fn test_bytecode() -> Vec<JoltInstructionRow> {
        vec![
            JoltInstructionRow {
                instruction_kind: JoltInstructionKind::ADD,
                address: 9,
                operands: NormalizedOperands {
                    rs1: Some(3),
                    rs2: Some(5),
                    rd: Some(7),
                    imm: 4,
                },
                virtual_sequence_remaining: None,
                is_first_in_sequence: false,
                is_compressed: false,
            },
            JoltInstructionRow::default(),
        ]
    }

    #[test]
    fn lane_layout_is_contiguous_and_fits_capacity() {
        let layout = BYTECODE_LANE_LAYOUT;
        let register_count = 1 << REGISTER_ADDRESS_BITS;

        assert_eq!(layout.rs2_start, layout.rs1_start + register_count);
        assert_eq!(layout.rd_start, layout.rs2_start + register_count);
        assert_eq!(layout.unexp_pc_idx, layout.rd_start + register_count);
        assert_eq!(layout.imm_idx, layout.unexp_pc_idx + 1);
        assert_eq!(layout.circuit_start, layout.imm_idx + 1);
        assert_eq!(layout.instr_start, layout.circuit_start + NUM_CIRCUIT_FLAGS);
        assert_eq!(
            layout.lookup_start,
            layout.instr_start + NUM_INSTRUCTION_FLAGS
        );
        assert_eq!(
            layout.raf_flag_idx,
            layout.lookup_start + LookupTableKind::<XLEN>::COUNT
        );
        assert_eq!(layout.raf_flag_idx + 1, total_lanes());

        assert!(committed_lanes().is_power_of_two());
        assert!(committed_lanes() >= total_lanes());
        assert!(committed_lanes() < 2 * total_lanes());
        assert_eq!(1 << committed_lane_vars(), committed_lanes());
    }

    #[test]
    fn chunking_validation_rules() {
        assert!(is_valid_committed_bytecode_chunking_for_len(1024, 1));
        assert!(is_valid_committed_bytecode_chunking_for_len(1024, 4));
        assert!(is_valid_committed_bytecode_chunking_for_len(256, 256));
        assert!(!is_valid_committed_bytecode_chunking_for_len(1024, 0));
        assert!(!is_valid_committed_bytecode_chunking_for_len(1024, 3));
        assert!(!is_valid_committed_bytecode_chunking_for_len(1024, 512));
        assert!(!is_valid_committed_bytecode_chunking_for_len(1000, 4));

        assert_eq!(
            precommitted_candidate(1024, 4),
            Ok(committed_lane_vars() + 8)
        );
        assert_eq!(
            precommitted_candidate(1024, 3),
            Err(JoltFormulaPointError::InvalidBytecodeChunking {
                bytecode_len: 1024,
                chunk_count: 3,
            })
        );
    }

    #[test]
    fn lane_weights_reproduce_read_raf_stage_values() {
        let bytecode = test_bytecode();
        let r_address = [fr(29)];
        let register_read_write_point: Vec<Fr> =
            (1..=REGISTER_ADDRESS_BITS as u64).map(fr).collect();
        let register_val_evaluation_point: Vec<Fr> =
            (11..11 + REGISTER_ADDRESS_BITS as u64).map(fr).collect();
        let eta = fr(31);
        let stage1_gammas = gamma_powers(3, 2 + NUM_CIRCUIT_FLAGS);
        let stage2_gammas = gamma_powers(5, 4);
        let stage3_gammas = gamma_powers(7, 9);
        let stage4_gammas = gamma_powers(11, 3);
        let stage5_gammas = gamma_powers(13, 2 + LookupTableKind::<XLEN>::COUNT);

        let public_values = read_raf_public_values::<Fr>(BytecodeReadRafEvaluationInputs {
            bytecode: &bytecode,
            r_address: &r_address,
            r_cycle: &[],
            stage_cycle_points: [&[]; 5],
            register_read_write_point: &register_read_write_point,
            register_val_evaluation_point: &register_val_evaluation_point,
            entry_bytecode_index: 0,
            stage1_gammas: &stage1_gammas,
            stage2_gammas: &stage2_gammas,
            stage3_gammas: &stage3_gammas,
            stage4_gammas: &stage4_gammas,
            stage5_gammas: &stage5_gammas,
        })
        .unwrap_or_else(|error| panic!("read-raf public values should evaluate: {error}"));
        let expected = public_values
            .stage_values
            .iter()
            .enumerate()
            .map(|(stage, value)| {
                let mut eta_power = fr(1);
                for _ in 0..stage {
                    eta_power *= eta;
                }
                eta_power * *value
            })
            .sum::<Fr>();

        let weights = lane_weights::<Fr>(BytecodeLaneWeightInputs {
            eta,
            stage1_gammas: &stage1_gammas,
            stage2_gammas: &stage2_gammas,
            stage3_gammas: &stage3_gammas,
            stage4_gammas: &stage4_gammas,
            stage5_gammas: &stage5_gammas,
            register_read_write_point: &register_read_write_point,
            register_val_evaluation_point: &register_val_evaluation_point,
        })
        .unwrap_or_else(|error| panic!("lane weights should evaluate: {error}"));
        let address_eq = EqPolynomial::<Fr>::evals(&r_address, None);
        let weighted_rows = bytecode
            .iter()
            .zip(address_eq)
            .map(|(instruction, eq_address)| {
                lane_values(instruction)
                    .into_iter()
                    .map(|(lane, value)| weights[lane] * value)
                    .sum::<Fr>()
                    * eq_address
            })
            .sum::<Fr>();

        assert_eq!(weighted_rows, expected);
    }

    #[test]
    fn lane_weights_reject_short_inputs() {
        let stage1_gammas = gamma_powers(3, 2 + NUM_CIRCUIT_FLAGS);
        let stage5_gammas = gamma_powers(13, 2 + LookupTableKind::<XLEN>::COUNT);
        let register_point: Vec<Fr> = (1..=REGISTER_ADDRESS_BITS as u64).map(fr).collect();

        let result = lane_weights::<Fr>(BytecodeLaneWeightInputs {
            eta: fr(31),
            stage1_gammas: &stage1_gammas,
            stage2_gammas: &[fr(1); 4],
            stage3_gammas: &[fr(1); 9],
            stage4_gammas: &[fr(1); 3],
            stage5_gammas: &stage5_gammas,
            register_read_write_point: &register_point[..REGISTER_ADDRESS_BITS - 1],
            register_val_evaluation_point: &register_point,
        });

        assert_eq!(
            result,
            Err(JoltFormulaPointError::OpeningPointLengthMismatch {
                expected: REGISTER_ADDRESS_BITS,
                got: REGISTER_ADDRESS_BITS - 1,
            })
        );
    }

    fn bytecode_layout(
        trace_order: TracePolynomialOrder,
        bytecode_len: usize,
        chunk_count: usize,
    ) -> BytecodeClaimReductionLayout {
        let log_t = 8;
        let log_k_chunk = 4;
        let candidate = precommitted_candidate(bytecode_len, chunk_count)
            .unwrap_or_else(|error| panic!("chunking should be valid: {error}"));
        let scheduling_reference = PrecommittedClaimReduction::scheduling_reference(
            log_t + log_k_chunk,
            &[candidate],
            log_k_chunk,
        );
        BytecodeClaimReductionLayout::balanced(
            trace_order,
            log_t,
            scheduling_reference,
            bytecode_len,
            chunk_count,
        )
        .unwrap_or_else(|error| panic!("layout should build: {error}"))
    }

    #[test]
    fn split_address_point_weights_dropped_high_bits() {
        let layout = bytecode_layout(TracePolynomialOrder::CycleMajor, 8, 4);
        let r_bc_full: Vec<Fr> = (1..=3).map(fr).collect();

        let point = layout
            .split_address_point(&r_bc_full)
            .unwrap_or_else(|error| panic!("address point should split: {error}"));

        assert_eq!(
            point.chunk_rbc_weights,
            EqPolynomial::<Fr>::evals(&r_bc_full[..2], None)
        );
        assert_eq!(point.r_bc, vec![fr(3)]);

        let single_chunk = bytecode_layout(TracePolynomialOrder::CycleMajor, 8, 1);
        let point = single_chunk
            .split_address_point(&r_bc_full)
            .unwrap_or_else(|error| panic!("address point should split: {error}"));
        assert_eq!(point.chunk_rbc_weights, vec![fr(1)]);
        assert_eq!(point.r_bc, r_bc_full);
    }

    /// `eq_combined` must factorize the MLE of the coefficient grid
    /// `lane_weights[lane] * eq(r_bc)[cycle]` laid out in the active trace
    /// order, for any opening point of matching length.
    #[test]
    fn final_output_weights_match_naive_grid_evaluation() {
        for trace_order in [
            TracePolynomialOrder::CycleMajor,
            TracePolynomialOrder::AddressMajor,
        ] {
            let layout = bytecode_layout(trace_order, 2, 1);
            let precommitted = layout.precommitted();
            assert!(precommitted.num_address_phase_rounds() > 0);

            let cycle_var_challenges: Vec<Fr> = (0..precommitted.cycle_phase_rounds().len())
                .map(|index| fr(50 + index as u64))
                .collect();
            let challenges: Vec<Fr> = (0..precommitted.address_phase_total_rounds())
                .map(|index| fr(80 + index as u64))
                .collect();
            let opening_point = layout
                .address_phase_opening_point(&cycle_var_challenges, &challenges)
                .unwrap_or_else(|error| panic!("opening point should assemble: {error}"));

            let r_bc = [fr(23)];
            let mut lane_weight_values = vec![Fr::from_u64(0); committed_lanes()];
            for (lane, weight) in lane_weight_values.iter_mut().enumerate() {
                *weight = fr(3 + lane as u64);
            }

            let chunk_cycle_len = 2usize;
            let eq_rbc = EqPolynomial::<Fr>::evals(&r_bc, None);
            let mut grid = vec![Fr::from_u64(0); committed_lanes() * chunk_cycle_len];
            for (lane, lane_weight) in lane_weight_values.iter().enumerate() {
                for (cycle, eq_cycle) in eq_rbc.iter().enumerate() {
                    let index = trace_order.address_cycle_to_index(
                        lane,
                        cycle,
                        committed_lanes(),
                        chunk_cycle_len,
                    );
                    grid[index] = *lane_weight * *eq_cycle;
                }
            }
            let opening_eq = EqPolynomial::<Fr>::evals(&opening_point, None);
            let naive = grid
                .iter()
                .zip(opening_eq)
                .map(|(coeff, eq)| *coeff * eq)
                .sum::<Fr>();

            let weights = layout
                .address_phase_final_output_weights(
                    BytecodeOutputWeightInputs {
                        r_bc: &r_bc,
                        chunk_rbc_weights: &[fr(1)],
                        lane_weights: &lane_weight_values,
                    },
                    &cycle_var_challenges,
                    &challenges,
                )
                .unwrap_or_else(|error| panic!("output weights should evaluate: {error}"));

            assert_eq!(
                weights,
                vec![naive * precommitted_skip_round_scale::<Fr>(precommitted)]
            );
        }
    }

    #[test]
    fn cycle_phase_output_openings_track_address_phase_presence() {
        let with_address = PrecommittedReductionDimensions::new(4, 3, true);
        assert_eq!(
            cycle_phase_output_openings(with_address, 2),
            vec![cycle_phase_intermediate_opening()]
        );

        let without_address = PrecommittedReductionDimensions::new(4, 3, false);
        assert_eq!(
            cycle_phase_output_openings(without_address, 2),
            vec![
                final_bytecode_chunk_opening(0),
                final_bytecode_chunk_opening(1),
            ]
        );
        assert!(matches!(
            final_bytecode_chunk_opening(1),
            JoltOpeningId::Polynomial {
                polynomial: JoltPolynomialId::Committed(JoltCommittedPolynomial::BytecodeChunk(1)),
                relation: JoltRelationId::BytecodeClaimReduction,
            }
        ));
    }

    #[test]
    fn cycle_phase_skip_scale_matches_two_phase_gap() {
        let layout = bytecode_layout(TracePolynomialOrder::CycleMajor, 2, 1);
        let precommitted = layout.precommitted();
        let two_inv = fr(2).inv_or_zero();
        let gap = (precommitted.cycle_phase_total_rounds()
            - precommitted.cycle_phase_rounds().len())
            + (precommitted.address_phase_total_rounds()
                - precommitted.address_phase_rounds().len());

        assert_eq!(
            precommitted_skip_round_scale::<Fr>(precommitted),
            (0..gap).fold(fr(1), |scale, _| scale * two_inv)
        );
    }
}
