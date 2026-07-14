//! The stage 6a bytecode read-RAF address-phase sumcheck instance.
//!
//! The **address phase** binds the `log_k` address variables. Its input claim is
//! the gamma-folded bind of the entire prior proof (every stage-1..5 opening plus
//! the two PC claims), wired by
//! [`bytecode_read_raf_address_phase_input_values_from_upstream`]. Its output is
//! the staged `BytecodeReadRafAddrClaim` intermediate (consumed by the stage-6b
//! cycle phase) followed, in committed mode, by the `BytecodeValStage` openings.
//!
//! Under the `akita` feature the symbolic swaps to the lattice address phase,
//! whose input fold additionally consumes the `IncVirtualization` store

#[cfg(not(feature = "akita"))]
use jolt_claims::protocols::jolt::relations;
pub use jolt_claims::protocols::jolt::relations::bytecode::{
    BytecodeReadRafAddressPhaseInputClaims, BytecodeReadRafAddressPhaseOutputClaims,
};
use jolt_claims::protocols::jolt::{
    geometry::{
        bytecode::BytecodeReadRafDimensions, claim_reductions::bytecode as bytecode_reduction,
    },
    JoltOpeningId,
};
use jolt_claims::SymbolicSumcheck;
use jolt_field::Field;

use crate::stages::relations::{ConcreteSumcheck, SumcheckInputPoints};
use crate::stages::{
    stage1::Stage1BatchOutputClaims, stage2::Stage2BatchOutputClaims, stage3::Stage3OutputClaims,
    stage4::Stage4OutputClaims, stage5::Stage5OutputClaims,
};
use crate::VerifierError;

#[cfg(not(feature = "akita"))]
type AddressPhaseSymbolic = relations::bytecode::ReadRafAddressPhase;
#[cfg(feature = "akita")]
type AddressPhaseSymbolic =
    jolt_claims::protocols::jolt::lattice::relations::read_raf::LatticeReadRafAddressPhase;

/// Wire the prior-proof opening *values* the address-phase input claim binds
/// (every stage-1..5 opening folded by the `read_raf_address_phase` input `Expr`,
/// plus the two PC claims). Each Spartan-outer circuit flag is a direct
/// field-for-field read from the stage-1 outer remainder; the input claim reads
/// only values, so the consumed input *points* are the generated all-empty
/// `empty_input_points`.
pub fn bytecode_read_raf_address_phase_input_values_from_upstream<F: Field>(
    stage1: &Stage1BatchOutputClaims<F>,
    stage2: &Stage2BatchOutputClaims<F>,
    stage3: &Stage3OutputClaims<F>,
    stage4: &Stage4OutputClaims<F>,
    stage5: &Stage5OutputClaims<F>,
) -> BytecodeReadRafAddressPhaseInputClaims<F> {
    let outer = &stage1.outer_remainder;
    let product = &stage2.product_remainder;
    let instruction_input = &stage3.instruction_input;
    let shift = &stage3.shift;
    let registers_read_write = &stage4.registers_read_write;
    let instruction_read_raf = &stage5.instruction_read_raf;
    BytecodeReadRafAddressPhaseInputClaims {
        outer_unexpanded_pc: outer.unexpanded_pc,
        outer_imm: outer.imm,
        outer_add_operands: outer.add_operands,
        outer_subtract_operands: outer.subtract_operands,
        outer_multiply_operands: outer.multiply_operands,
        outer_load: outer.load,
        outer_store: outer.store,
        outer_jump: outer.jump,
        outer_write_lookup_output_to_rd: outer.write_lookup_output_to_rd,
        outer_virtual_instruction: outer.virtual_instruction,
        outer_assert: outer.assert,
        outer_do_not_update_unexpanded_pc: outer.do_not_update_unexpanded_pc,
        outer_advice: outer.advice,
        outer_is_compressed: outer.is_compressed,
        outer_is_first_in_sequence: outer.is_first_in_sequence,
        outer_is_last_in_sequence: outer.is_last_in_sequence,
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
        lookup_table_flags: instruction_read_raf.lookup_table_flags.clone(),
    }
}

pub struct BytecodeReadRafAddressPhase<F: Field> {
    symbolic: AddressPhaseSymbolic,
    /// The staged-val count in committed-program mode, else 0.
    num_val_stages: usize,
    _field: core::marker::PhantomData<F>,
}

impl<F: Field> BytecodeReadRafAddressPhase<F> {
    pub fn new(dimensions: BytecodeReadRafDimensions, num_val_stages: usize) -> Self {
        Self {
            symbolic: AddressPhaseSymbolic::new(dimensions),
            num_val_stages,
            _field: core::marker::PhantomData,
        }
    }
}

impl<F: Field> ConcreteSumcheck<F> for BytecodeReadRafAddressPhase<F> {
    type Symbolic = AddressPhaseSymbolic;

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
        _input_points: &SumcheckInputPoints<F, Self>,
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
