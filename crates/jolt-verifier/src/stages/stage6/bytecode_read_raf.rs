//! The stage 6 bytecode read-RAF sumcheck instances.
//!
//! This module currently provides the stage-6a **address phase**. The cycle phase
//! (stage 6b) is deferred to the wiring step: its expected output depends on the
//! bytecode-table public values (`read_raf_public_values`, which needs the
//! preprocessing bytecode rows) and, in committed-program mode, consumes the
//! staged `BytecodeValStage` openings inside its *output* expression — both of
//! which are cleanest to finalize against the live `verify()`/prover wiring.
//!
//! The address phase binds the `log_k` address variables. Its input claim is the
//! gamma-folded bind of the entire prior proof (every stage-1..5 opening plus the
//! two PC claims); that 25-opening formula already lives in the single-sourced
//! [`stage6_bytecode_read_raf_address_input`] helper, so this relation takes the
//! precomputed value and overrides [`ConcreteSumcheck::input_claim`] rather than
//! restating the bind as a 25-field `InputClaims`. Its output is the staged
//! `BytecodeReadRafAddrClaim` intermediate (consumed by the cycle phase) followed,
//! in committed mode, by the `BytecodeValStage` openings.
//!
//! [`stage6_bytecode_read_raf_address_input`]: super::verify::stage6_bytecode_read_raf_address_input

use jolt_claims::protocols::jolt::relations;
pub use jolt_claims::protocols::jolt::relations::bytecode::{
    BytecodeReadRafAddressPhaseChallenges, BytecodeReadRafAddressPhaseInputClaims,
    BytecodeReadRafAddressPhaseOutputClaims, BytecodeReadRafCyclePhaseChallenges,
    BytecodeReadRafCyclePhaseCommittedChallenges, BytecodeReadRafInputClaims,
    BytecodeReadRafOutputClaims,
};
use jolt_claims::protocols::jolt::{
    geometry::{
        bytecode::{
            self, BytecodeReadRafCommittedEvaluationInputs, BytecodeReadRafDimensions,
            BytecodeReadRafEvaluationInputs,
        },
        claim_reductions::bytecode::bytecode_val_stage_opening,
        dimensions::committed_address_chunks,
        spartan::outer_opening,
    },
    BytecodeReadRafChallenge, JoltChallengeId, JoltDerivedId, JoltRelationId,
    JoltVirtualPolynomial,
};
use jolt_claims::{SumcheckChallenges, SymbolicSumcheck};
use jolt_field::Field;
use jolt_riscv::{JoltInstructionRow, CIRCUIT_FLAGS};

use super::verify::{
    stage6_bytecode_read_raf_expected_output, Stage6BytecodeReadRafExpectedOutputInputs,
};
use crate::stages::relations::{ConcreteSumcheck, OutputClaims};
use crate::stages::{
    stage1::Stage1ClearOutput, stage2::Stage2ClearOutput, stage3::Stage3ClearOutput,
    stage4::Stage4ClearOutput, stage5::Stage5ClearOutput,
};
use crate::VerifierError;

/// Wire the prior-proof opening *values* the address-phase input claim binds
/// (every stage-1..5 opening folded by the `read_raf_address_phase` input `Expr`,
/// plus the two PC claims). (Verifier-side constructor for the moved
/// [`BytecodeReadRafAddressPhaseInputClaims`].)
pub fn bytecode_read_raf_address_phase_input_values_from_upstream<F: Field>(
    stage1: &Stage1ClearOutput<F>,
    stage2: &Stage2ClearOutput<F>,
    stage3: &Stage3ClearOutput<F>,
    stage4: &Stage4ClearOutput<F>,
    stage5: &Stage5ClearOutput<F>,
) -> Result<BytecodeReadRafAddressPhaseInputClaims<F>, VerifierError> {
    let outer = &stage1.output_values.outer_remainder;
    let outer_op_flags = CIRCUIT_FLAGS
        .iter()
        .map(|flag| {
            let id = outer_opening(JoltVirtualPolynomial::OpFlags(*flag));
            outer
                .resolve_output(&id)
                .ok_or(VerifierError::MissingOpeningClaim { id })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let product = &stage2.output_values.product_remainder;
    let instruction_input = &stage3.output_values.instruction_input;
    let shift = &stage3.output_values.shift;
    let registers_read_write = &stage4.output_values.registers_read_write;
    let instruction_read_raf = &stage5.output_values.instruction_read_raf;
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
        rd_wa_val_evaluation: stage5.output_values.registers_val_evaluation.rd_wa,
        instruction_raf_flag: instruction_read_raf.instruction_raf_flag,
        lookup_table_flags,
    })
}

/// Wire the prior-proof opening *points* the address-phase input claim binds. The
/// input claim reads only opening *values*, so the points are unused; every field
/// is an empty point. (Verifier-side constructor for the moved
/// [`BytecodeReadRafAddressPhaseInputClaims<Vec<F>>`].)
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

/// The `BytecodeReadRafAddrClaim` intermediate *value* consumed from the address
/// phase. (Verifier-side constructor for the moved [`BytecodeReadRafInputClaims`].)
pub fn bytecode_read_raf_input_values_from_upstream<F: Field>(
    address_phase_value: F,
) -> BytecodeReadRafInputClaims<F> {
    BytecodeReadRafInputClaims {
        address_phase: address_phase_value,
    }
}

/// The `BytecodeReadRafAddrClaim` intermediate *point* consumed from the address
/// phase.
pub fn bytecode_read_raf_input_points_from_upstream<F: Field>(
    address_phase_point: Vec<F>,
) -> BytecodeReadRafInputClaims<Vec<F>> {
    BytecodeReadRafInputClaims {
        address_phase: address_phase_point,
    }
}

/// Construction inputs for the full-program bytecode cycle relation. The bytecode
/// rows are borrowed from preprocessing; the points/gammas are the verifier's
/// per-stage cycle bindings and Fiat-Shamir gammas. `stage_cycle_points` /
/// `stage_gammas` are indexed by stage (1..=5) in order.
pub struct BytecodeReadRafCycleInputs<'a, F: Field> {
    pub dimensions: BytecodeReadRafDimensions,
    pub bytecode: &'a [JoltInstructionRow],
    pub r_address: Vec<F>,
    pub stage_cycle_points: [Vec<F>; 5],
    pub register_read_write_point: Vec<F>,
    pub register_val_evaluation_point: Vec<F>,
    pub entry_bytecode_index: usize,
    pub stage_gammas: [Vec<F>; 5],
    pub committed_chunk_bits: usize,
}

/// The stage-6b bytecode read-RAF cycle phase, full-program mode.
///
/// Its expected output is the bytecode-table public values evaluated at
/// `(r_address, r_cycle)` folded against the committed `BytecodeRa` product — the
/// same quantity `read_raf`'s output expression computes. Rather than resolve each
/// public id individually (which would recompute the `O(2^log_k)` table fold once
/// per public), it OVERRIDES [`ConcreteSumcheck::expected_output`] to evaluate the
/// public values once and reuse the shared
/// [`stage6_bytecode_read_raf_expected_output`] helper.
///
/// Committed-program mode — which folds the staged `BytecodeValStage` openings into
/// the output expression and draws its publics from a committed evaluation — stays
/// on the verifier's existing committed helper for now.
pub struct BytecodeReadRaf<'a, F: Field> {
    symbolic: relations::bytecode::ReadRafCyclePhase,
    inputs: BytecodeReadRafCycleInputs<'a, F>,
}

impl<'a, F: Field> BytecodeReadRaf<'a, F> {
    pub fn new(inputs: BytecodeReadRafCycleInputs<'a, F>) -> Self {
        Self {
            symbolic: relations::bytecode::ReadRafCyclePhase::new(inputs.dimensions),
            inputs,
        }
    }

    /// The `log_t`-variable cycle suffix of a produced `BytecodeRa` opening point
    /// (`chunk ++ r_cycle`).
    fn r_cycle<'p>(&self, opening_point: &'p [F]) -> Result<&'p [F], VerifierError> {
        let log_t = self.inputs.dimensions.log_t();
        opening_point
            .get(opening_point.len() - log_t..)
            .ok_or_else(|| public_input_failed("bytecode cycle opening point shorter than log_t"))
    }
}

fn public_input_failed(reason: impl ToString) -> VerifierError {
    VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::BytecodeReadRaf,
        reason: reason.to_string(),
    }
}

impl<F: Field> ConcreteSumcheck<F> for BytecodeReadRaf<'_, F> {
    type Symbolic = relations::bytecode::ReadRafCyclePhase;

    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }

    fn derive_opening_points(
        &self,
        sumcheck_point: &[F],
        _input_points: &BytecodeReadRafInputClaims<Vec<F>>,
    ) -> Result<BytecodeReadRafOutputClaims<Vec<F>>, VerifierError> {
        let r_cycle = sumcheck_point.iter().rev().copied().collect::<Vec<_>>();
        let bytecode_ra =
            committed_address_chunks(&self.inputs.r_address, self.inputs.committed_chunk_bits)
                .into_iter()
                .map(|chunk| [chunk.as_slice(), r_cycle.as_slice()].concat())
                .collect();
        Ok(BytecodeReadRafOutputClaims { bytecode_ra })
    }

    fn expected_output(
        &self,
        _input_points: &BytecodeReadRafInputClaims<Vec<F>>,
        output_values: &BytecodeReadRafOutputClaims<F>,
        output_points: &BytecodeReadRafOutputClaims<Vec<F>>,
        challenges: &BytecodeReadRafCyclePhaseChallenges<F>,
    ) -> Result<F, VerifierError> {
        let gamma = challenges
            .resolve_challenge(&JoltChallengeId::from(BytecodeReadRafChallenge::Gamma))
            .ok_or(VerifierError::MissingStageClaimChallenge {
                id: JoltChallengeId::from(BytecodeReadRafChallenge::Gamma),
            })?;
        let opening_point = output_points
            .bytecode_ra()
            .first()
            .ok_or_else(|| public_input_failed("bytecode cycle produced no openings"))?;
        let r_cycle = self.r_cycle(opening_point)?;
        let public_values =
            bytecode::read_raf_public_values::<F>(BytecodeReadRafEvaluationInputs {
                bytecode: self.inputs.bytecode,
                r_address: &self.inputs.r_address,
                r_cycle,
                stage_cycle_points: [
                    &self.inputs.stage_cycle_points[0],
                    &self.inputs.stage_cycle_points[1],
                    &self.inputs.stage_cycle_points[2],
                    &self.inputs.stage_cycle_points[3],
                    &self.inputs.stage_cycle_points[4],
                ],
                register_read_write_point: &self.inputs.register_read_write_point,
                register_val_evaluation_point: &self.inputs.register_val_evaluation_point,
                entry_bytecode_index: self.inputs.entry_bytecode_index,
                stage1_gammas: &self.inputs.stage_gammas[0],
                stage2_gammas: &self.inputs.stage_gammas[1],
                stage3_gammas: &self.inputs.stage_gammas[2],
                stage4_gammas: &self.inputs.stage_gammas[3],
                stage5_gammas: &self.inputs.stage_gammas[4],
            })
            .map_err(public_input_failed)?;
        stage6_bytecode_read_raf_expected_output(Stage6BytecodeReadRafExpectedOutputInputs {
            dimensions: self.inputs.dimensions,
            public_values: &public_values,
            bytecode_ra: &output_values.bytecode_ra,
            gamma,
        })
    }
}

/// Construction inputs for the committed-program bytecode cycle relation.
pub struct BytecodeReadRafCommittedCycleInputs<F: Field> {
    pub dimensions: BytecodeReadRafDimensions,
    pub r_address: Vec<F>,
    pub stage_cycle_points: [Vec<F>; 5],
    pub entry_bytecode_index: usize,
    pub committed_chunk_bits: usize,
    /// The staged `BytecodeValStage` opening values from the address phase.
    pub val_stages: Vec<F>,
}

/// The stage-6b bytecode read-RAF cycle phase, committed-program mode.
///
/// Mirrors [`BytecodeReadRaf`] but folds the staged `BytecodeValStage` openings
/// into the output expression and draws its publics from a committed bytecode
/// evaluation (`read_raf_committed_public_values`) rather than the full bytecode
/// table. Like the full-mode relation it OVERRIDES
/// [`ConcreteSumcheck::expected_output`]: the staged Val openings are inputs mixed
/// into the output, and the committed public values are evaluated once.
pub struct BytecodeReadRafCommitted<F: Field> {
    symbolic: relations::bytecode::ReadRafCyclePhaseCommitted,
    dimensions: BytecodeReadRafDimensions,
    r_address: Vec<F>,
    stage_cycle_points: [Vec<F>; 5],
    entry_bytecode_index: usize,
    committed_chunk_bits: usize,
    val_stages: Vec<F>,
}

impl<F: Field> BytecodeReadRafCommitted<F> {
    pub fn new(inputs: BytecodeReadRafCommittedCycleInputs<F>) -> Self {
        Self {
            symbolic: relations::bytecode::ReadRafCyclePhaseCommitted::new(inputs.dimensions),
            dimensions: inputs.dimensions,
            r_address: inputs.r_address,
            stage_cycle_points: inputs.stage_cycle_points,
            entry_bytecode_index: inputs.entry_bytecode_index,
            committed_chunk_bits: inputs.committed_chunk_bits,
            val_stages: inputs.val_stages,
        }
    }

    fn r_cycle<'p>(&self, opening_point: &'p [F]) -> Result<&'p [F], VerifierError> {
        let log_t = self.dimensions.log_t();
        opening_point
            .get(opening_point.len() - log_t..)
            .ok_or_else(|| public_input_failed("bytecode cycle opening point shorter than log_t"))
    }
}

impl<F: Field> ConcreteSumcheck<F> for BytecodeReadRafCommitted<F> {
    type Symbolic = relations::bytecode::ReadRafCyclePhaseCommitted;

    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }

    fn derive_opening_points(
        &self,
        sumcheck_point: &[F],
        _input_points: &BytecodeReadRafInputClaims<Vec<F>>,
    ) -> Result<BytecodeReadRafOutputClaims<Vec<F>>, VerifierError> {
        let r_cycle = sumcheck_point.iter().rev().copied().collect::<Vec<_>>();
        let bytecode_ra = committed_address_chunks(&self.r_address, self.committed_chunk_bits)
            .into_iter()
            .map(|chunk| [chunk.as_slice(), r_cycle.as_slice()].concat())
            .collect();
        Ok(BytecodeReadRafOutputClaims { bytecode_ra })
    }

    fn expected_output(
        &self,
        _input_points: &BytecodeReadRafInputClaims<Vec<F>>,
        output_values: &BytecodeReadRafOutputClaims<F>,
        output_points: &BytecodeReadRafOutputClaims<Vec<F>>,
        challenges: &BytecodeReadRafCyclePhaseCommittedChallenges<F>,
    ) -> Result<F, VerifierError> {
        let opening_point = output_points
            .bytecode_ra()
            .first()
            .map(Vec::as_slice)
            .ok_or_else(|| public_input_failed("bytecode cycle produced no openings"))?;
        let r_cycle = self.r_cycle(opening_point)?;
        let public_values = bytecode::read_raf_committed_public_values::<F>(
            BytecodeReadRafCommittedEvaluationInputs {
                r_address: &self.r_address,
                r_cycle,
                stage_cycle_points: [
                    &self.stage_cycle_points[0],
                    &self.stage_cycle_points[1],
                    &self.stage_cycle_points[2],
                    &self.stage_cycle_points[3],
                    &self.stage_cycle_points[4],
                ],
                entry_bytecode_index: self.entry_bytecode_index,
            },
        );
        let output_openings = bytecode::read_raf_output_openings(self.dimensions);
        self.symbolic().output_expression::<F>().try_evaluate(
            |id| {
                for (stage, value) in self.val_stages.iter().enumerate() {
                    if *id == bytecode_val_stage_opening(stage) {
                        return Ok(*value);
                    }
                }
                for (index, opening_id) in output_openings.bytecode_ra.iter().enumerate() {
                    if *id == *opening_id {
                        return output_values
                            .bytecode_ra
                            .get(index)
                            .copied()
                            .ok_or(VerifierError::MissingOpeningClaim { id: *id });
                    }
                }
                Err(VerifierError::MissingOpeningClaim { id: *id })
            },
            |id| {
                challenges
                    .resolve_challenge(id)
                    .ok_or(VerifierError::MissingStageClaimChallenge { id: *id })
            },
            |id| match id {
                JoltDerivedId::BytecodeReadRaf(public_id) => public_values
                    .value(*public_id)
                    .ok_or(VerifierError::MissingStageClaimDerived { id: *id }),
                _ => Err(VerifierError::MissingStageClaimDerived { id: *id }),
            },
        )
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

        // Inline (stage6/verify.rs L214-221): six `challenge_scalar_powers(..)`,
        // each contributing its degree-1 power.
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
