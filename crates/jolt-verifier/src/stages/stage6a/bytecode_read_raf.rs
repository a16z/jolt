//! The stage 6a bytecode read-RAF address-phase sumcheck instance.
//!
//! The **address phase** binds the `log_k` address variables. Its input claim is
//! the gamma-folded bind of the entire prior proof (every stage-1..5 opening plus
//! the two PC claims), wired by
//! [`bytecode_read_raf_address_phase_input_values_from_upstream`]. Its output is
//! the staged `BytecodeReadRafAddrClaim` intermediate (consumed by the stage-6b
//! cycle phase) followed, in committed mode, by the `BytecodeValStage` openings.

use jolt_claims::protocols::jolt::relations;
pub use jolt_claims::protocols::jolt::relations::bytecode::{
    BytecodeReadRafAddressPhaseInputClaims, BytecodeReadRafAddressPhaseOutputClaims,
};
use jolt_claims::protocols::jolt::{
    geometry::{
        bytecode::BytecodeReadRafDimensions, claim_reductions::bytecode as bytecode_reduction,
        spartan::outer_opening,
    },
    JoltOpeningId, JoltVirtualPolynomial,
};
use jolt_claims::SymbolicSumcheck;
use jolt_field::Field;
use jolt_riscv::CIRCUIT_FLAGS;

use crate::stages::relations::{ConcreteSumcheck, OutputClaims};
use crate::stages::{
    stage1::Stage1BatchOutputClaims, stage2::Stage2BatchOutputClaims, stage3::Stage3OutputClaims,
    stage4::Stage4OutputClaims, stage5::Stage5OutputClaims,
};
use crate::VerifierError;

/// Wire the prior-proof opening *values* the address-phase input claim binds
/// (every stage-1..5 opening folded by the `read_raf_address_phase` input `Expr`,
/// plus the two PC claims).
pub fn bytecode_read_raf_address_phase_input_values_from_upstream<F: Field>(
    stage1: &Stage1BatchOutputClaims<F>,
    stage2: &Stage2BatchOutputClaims<F>,
    stage3: &Stage3OutputClaims<F>,
    stage4: &Stage4OutputClaims<F>,
    stage5: &Stage5OutputClaims<F>,
) -> Result<BytecodeReadRafAddressPhaseInputClaims<F>, VerifierError> {
    let outer = &stage1.outer_remainder;
    let outer_op_flags = CIRCUIT_FLAGS
        .iter()
        .map(|flag| {
            let id = outer_opening(JoltVirtualPolynomial::OpFlags(*flag));
            outer
                .resolve_output(&id)
                .ok_or(VerifierError::MissingOpeningClaim { id })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let product = &stage2.product_remainder;
    let instruction_input = &stage3.instruction_input;
    let shift = &stage3.shift;
    let registers_read_write = &stage4.registers_read_write;
    let instruction_read_raf = &stage5.instruction_read_raf;
    let lookup_table_flags = instruction_read_raf.lookup_table_flags.clone();
    Ok(BytecodeReadRafAddressPhaseInputClaims {
        outer_unexpanded_pc: outer.unexpanded_pc,
        outer_imm: outer.imm,
        outer_op_flags,
        outer_pc: outer.pc,
        product_jump: product.jump_flag,
        product_branch: product.branch_flag,
        product_write_lookup_output_to_rd: product.write_lookup_output_to_rd,
        product_virtual_instruction: product.virtual_instruction,
        instruction_input_imm: instruction_input.imm,
        shift_unexpanded_pc: shift.unexpanded_pc,
        left_operand_is_rs1_value: instruction_input.left_operand_is_rs1,
        left_operand_is_pc: instruction_input.left_operand_is_pc,
        right_operand_is_rs2_value: instruction_input.right_operand_is_rs2,
        right_operand_is_imm: instruction_input.right_operand_is_imm,
        is_noop: shift.is_noop,
        shift_virtual_instruction: shift.is_virtual,
        shift_is_first_in_sequence: shift.is_first_in_sequence,
        shift_pc: shift.pc,
        rd_wa_read_write: registers_read_write.rd_wa,
        rs1_ra: registers_read_write.rs1_ra,
        rs2_ra: registers_read_write.rs2_ra,
        rd_wa_val_evaluation: stage5.registers_val_evaluation.rd_wa,
        instruction_raf_flag: instruction_read_raf.instruction_raf_flag,
        lookup_table_flags,
    })
}

/// Wire the prior-proof opening *points* the address-phase input claim binds. The
/// input claim reads only opening *values*, so the points are unused; every field
/// is an empty point.
///
/// WARNING: this must NOT be replaced by a generated
/// `#[sumcheck_batch(empty_input_points)]` constructor. Unlike the other stages'
/// all-empty input points, `outer_op_flags` is a length-`CIRCUIT_FLAGS.len()` outer
/// vec of empty points (one per circuit flag), which `Default::default()` would
/// shrink to a length-0 vec — desynchronizing the per-flag opening wiring.
pub fn bytecode_read_raf_address_phase_input_points_from_upstream<F: Field>(
) -> BytecodeReadRafAddressPhaseInputClaims<Vec<F>> {
    BytecodeReadRafAddressPhaseInputClaims {
        outer_unexpanded_pc: Vec::new(),
        outer_imm: Vec::new(),
        outer_op_flags: vec![Vec::new(); CIRCUIT_FLAGS.len()],
        outer_pc: Vec::new(),
        product_jump: Vec::new(),
        product_branch: Vec::new(),
        product_write_lookup_output_to_rd: Vec::new(),
        product_virtual_instruction: Vec::new(),
        instruction_input_imm: Vec::new(),
        shift_unexpanded_pc: Vec::new(),
        left_operand_is_rs1_value: Vec::new(),
        left_operand_is_pc: Vec::new(),
        right_operand_is_rs2_value: Vec::new(),
        right_operand_is_imm: Vec::new(),
        is_noop: Vec::new(),
        shift_virtual_instruction: Vec::new(),
        shift_is_first_in_sequence: Vec::new(),
        shift_pc: Vec::new(),
        rd_wa_read_write: Vec::new(),
        rs1_ra: Vec::new(),
        rs2_ra: Vec::new(),
        rd_wa_val_evaluation: Vec::new(),
        instruction_raf_flag: Vec::new(),
        lookup_table_flags: Vec::new(),
    }
}

pub struct BytecodeReadRafAddressPhase<F: Field> {
    symbolic: relations::bytecode::ReadRafAddressPhase,
    /// `NUM_BYTECODE_VAL_STAGES` in committed-program mode, else 0.
    num_val_stages: usize,
    _field: core::marker::PhantomData<F>,
}

impl<F: Field> BytecodeReadRafAddressPhase<F> {
    pub fn new(dimensions: BytecodeReadRafDimensions, num_val_stages: usize) -> Self {
        Self {
            symbolic: relations::bytecode::ReadRafAddressPhase::new(dimensions),
            num_val_stages,
            _field: core::marker::PhantomData,
        }
    }
}

impl<F: Field> ConcreteSumcheck<F> for BytecodeReadRafAddressPhase<F> {
    type Symbolic = relations::bytecode::ReadRafAddressPhase;

    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }

    fn wire_output_openings(&self) -> std::collections::BTreeSet<JoltOpeningId> {
        // Committed-program mode absorbs the staged `BytecodeValStage` columns
        // beyond the output-`Expr` set (the address-phase intermediate); their
        // constraining fold happens in stage 6b's bytecode claim reduction.
        let mut openings = self.symbolic().expected_output_openings::<F>();
        openings
            .extend((0..self.num_val_stages).map(bytecode_reduction::bytecode_val_stage_opening));
        openings
    }

    fn derive_opening_points(
        &self,
        sumcheck_point: &[F],
        _input_points: &BytecodeReadRafAddressPhaseInputClaims<Vec<F>>,
    ) -> Result<BytecodeReadRafAddressPhaseOutputClaims<Vec<F>>, VerifierError> {
        // `bytecode_r_address` is the reversed address sumcheck point; the
        // intermediate and every staged Val column open there.
        let r_address = sumcheck_point.iter().rev().copied().collect::<Vec<_>>();
        Ok(BytecodeReadRafAddressPhaseOutputClaims {
            intermediate: r_address.clone(),
            val_stages: vec![r_address; self.num_val_stages],
        })
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::stages::relations::draw_recording::{record, DrawEvent};
    use jolt_field::Fr;
    use jolt_lookup_tables::{LookupTableKind, XLEN as RISCV_XLEN};
    use jolt_riscv::NUM_CIRCUIT_FLAGS;
    use jolt_transcript::Transcript;

    // The address phase has the only multi-field `Challenges` (gamma + five stage
    // gammas), so it exercises that the default draws one `challenge_scalar` per
    // field in declaration order. Each inline draw is a `challenge_scalar_powers(..)`
    // whose single squeeze's degree-1 power equals that squeezed scalar, so the
    // default's six `challenge_scalar` squeezes reproduce the inline byte stream
    // (six squeezes) and the six stored values. The cycle and committed variants are
    // single-field and use the same default path.
    #[test]
    fn default_draw_challenges_matches_inline_bytecode_address_gammas() {
        let relation =
            BytecodeReadRafAddressPhase::<Fr>::new(BytecodeReadRafDimensions::new(3, 4, 2), 0);

        // Inline: six `challenge_scalar_powers(..)`, each contributing its
        // degree-1 power.
        let (inline_events, inline_gammas) = record(|t| {
            [
                t.challenge_scalar_powers(8)[1],
                t.challenge_scalar_powers(2 + NUM_CIRCUIT_FLAGS)[1],
                t.challenge_scalar_powers(4)[1],
                t.challenge_scalar_powers(9)[1],
                t.challenge_scalar_powers(3)[1],
                t.challenge_scalar_powers(2 + LookupTableKind::<RISCV_XLEN>::COUNT)[1],
            ]
        });
        let (draw_events, challenges) = record(|t| relation.draw_challenges(t).unwrap());

        // Six squeezes in the same order, byte-for-byte.
        assert_eq!(draw_events, inline_events);
        assert_eq!(
            draw_events,
            (1..=6).map(DrawEvent::Squeeze).collect::<Vec<_>>()
        );
        // Each field stores the corresponding inline degree-1 power.
        assert_eq!(
            [
                challenges.gamma,
                challenges.stage1_gamma,
                challenges.stage2_gamma,
                challenges.stage3_gamma,
                challenges.stage4_gamma,
                challenges.stage5_gamma,
            ],
            inline_gammas,
        );
    }
}
