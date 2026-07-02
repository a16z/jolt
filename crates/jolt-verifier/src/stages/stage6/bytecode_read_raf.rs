//! The stage 6 bytecode read-RAF sumcheck instances.
//!
//! The stage-6a **address phase** binds the `log_k` address variables. Its input
//! claim is the gamma-folded bind of the entire prior proof (every stage-1..5
//! opening plus the two PC claims), wired by
//! [`bytecode_read_raf_address_phase_input_values_from_upstream`]. Its output is
//! the staged `BytecodeReadRafAddrClaim` intermediate (consumed by the cycle
//! phase) followed, in committed mode, by the `BytecodeValStage` openings.
//!
//! The stage-6b **cycle phase** dispatches at runtime over full-program mode
//! ([`BytecodeReadRaf`]) and committed-program mode ([`BytecodeReadRafCommitted`])
//! through [`BytecodeReadRafCycle`], whose `ConcreteSumcheck` impl is anchored on
//! the committed symbolic (see the invariant note on the impl).

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
            BytecodeReadRafPublicValues, BytecodeReadRafStageValueInputs,
        },
        claim_reductions::bytecode::bytecode_val_stage_opening,
        dimensions::committed_address_chunks,
        spartan::outer_opening,
    },
    JoltDerivedId, JoltRelationId, JoltVirtualPolynomial,
};
use jolt_claims::{SumcheckChallenges, SymbolicSumcheck};
use jolt_field::Field;
use jolt_poly::EqPolynomial;
use jolt_riscv::{JoltInstructionRow, CIRCUIT_FLAGS};

use super::verify::{
    stage6_bytecode_read_raf_expected_output, Stage6BytecodeReadRafExpectedOutputInputs,
};
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
/// phase.
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

/// Clear-only aux for the full-program cycle relation's bytecode-table fold:
/// the borrowed table rows plus the register points and per-stage gammas that
/// weight each row. Consumed at construction ([`BytecodeReadRaf::new`] folds the
/// table against `eq(r_address)` immediately), so nothing borrowed is stored and
/// the relation stays lifetime-free.
pub struct BytecodeReadRafTableFoldInputs<'a, F: Field> {
    pub bytecode: &'a [JoltInstructionRow],
    pub register_read_write_point: &'a [F],
    pub register_val_evaluation_point: &'a [F],
    /// Per-stage (1..=5) Fiat-Shamir gamma powers.
    pub stage_gammas: [&'a [F]; 5],
}

/// Construction inputs for the full-program bytecode cycle relation.
/// `stage_cycle_points` are the verifier's per-stage (1..=5) cycle bindings.
/// `table_fold` is `Some` only in clear mode — ZK never runs `expected_output`,
/// so it skips the `O(2^log_k)` fold entirely.
pub struct BytecodeReadRafCycleInputs<'a, F: Field> {
    pub dimensions: BytecodeReadRafDimensions,
    pub r_address: Vec<F>,
    pub stage_cycle_points: [Vec<F>; 5],
    pub entry_bytecode_index: usize,
    pub committed_chunk_bits: usize,
    pub table_fold: Option<BytecodeReadRafTableFoldInputs<'a, F>>,
}

/// The stage-6b bytecode read-RAF cycle phase, full-program mode.
///
/// Its expected output is the bytecode-table public values evaluated at
/// `(r_address, r_cycle)` folded against the committed `BytecodeRa` product — the
/// same quantity `read_raf`'s output expression computes. The table depends only
/// on the address variables, so the `O(2^log_k)` fold against `eq(r_address)` runs
/// once at construction (clear mode only) and the cycle-dependent factors are
/// attached in [`ConcreteSumcheck::expected_output`], which it OVERRIDES to
/// evaluate the publics once and reuse the shared
/// [`stage6_bytecode_read_raf_expected_output`] helper.
pub struct BytecodeReadRaf<F: Field> {
    symbolic: relations::bytecode::ReadRafCyclePhase,
    dimensions: BytecodeReadRafDimensions,
    r_address: Vec<F>,
    stage_cycle_points: [Vec<F>; 5],
    entry_bytecode_index: usize,
    committed_chunk_bits: usize,
    /// The address-only bytecode-table fold: `Σ_row row_values[stage] *
    /// eq(r_address, row)` — the pre-cycle half of `read_raf_public_values`'
    /// `stage_values`. `None` in ZK, where `expected_output` never runs.
    stage_values_at_r_address: Option<[F; 5]>,
}

impl<F: Field> BytecodeReadRaf<F> {
    pub fn new(inputs: BytecodeReadRafCycleInputs<'_, F>) -> Result<Self, VerifierError> {
        let stage_values_at_r_address = inputs
            .table_fold
            .map(|fold| fold_stage_values(&inputs.r_address, fold))
            .transpose()?;
        Ok(Self {
            symbolic: relations::bytecode::ReadRafCyclePhase::new(inputs.dimensions),
            dimensions: inputs.dimensions,
            r_address: inputs.r_address,
            stage_cycle_points: inputs.stage_cycle_points,
            entry_bytecode_index: inputs.entry_bytecode_index,
            committed_chunk_bits: inputs.committed_chunk_bits,
            stage_values_at_r_address,
        })
    }

    /// The `log_t`-variable cycle suffix of a produced `BytecodeRa` opening point
    /// (`chunk ++ r_cycle`).
    fn r_cycle<'p>(&self, opening_point: &'p [F]) -> Result<&'p [F], VerifierError> {
        let log_t = self.dimensions.log_t();
        opening_point
            .get(opening_point.len() - log_t..)
            .ok_or_else(|| public_input_failed("bytecode cycle opening point shorter than log_t"))
    }
}

/// The address-only half of `read_raf_public_values`' `stage_values`: the
/// bytecode rows' per-stage values (shared `read_raf_stage_values` formula)
/// folded against `eq(r_address)`. The cycle-eq factors are attached later, at
/// `expected_output` time, so the fold can run before the cycle sumcheck.
fn fold_stage_values<F: Field>(
    r_address: &[F],
    fold: BytecodeReadRafTableFoldInputs<'_, F>,
) -> Result<[F; 5], VerifierError> {
    let expected_domain = 1usize
        .checked_shl(r_address.len() as u32)
        .ok_or_else(|| public_input_failed("bytecode address domain overflows"))?;
    if fold.bytecode.len() != expected_domain {
        return Err(public_input_failed(format!(
            "bytecode table has {} rows, expected the address domain {expected_domain}",
            fold.bytecode.len()
        )));
    }
    let address_eq_evals = EqPolynomial::<F>::evals(r_address, None);
    let row_values = bytecode::read_raf_stage_values(BytecodeReadRafStageValueInputs {
        bytecode: fold.bytecode,
        register_read_write_point: fold.register_read_write_point,
        register_val_evaluation_point: fold.register_val_evaluation_point,
        stage1_gammas: fold.stage_gammas[0],
        stage2_gammas: fold.stage_gammas[1],
        stage3_gammas: fold.stage_gammas[2],
        stage4_gammas: fold.stage_gammas[3],
        stage5_gammas: fold.stage_gammas[4],
    });
    let mut stage_values = [F::zero(); 5];
    for (row_values, eq_address) in row_values.into_iter().zip(address_eq_evals) {
        for (stage_value, row_value) in stage_values.iter_mut().zip(row_values) {
            *stage_value += row_value * eq_address;
        }
    }
    Ok(stage_values)
}

fn public_input_failed(reason: impl ToString) -> VerifierError {
    VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::BytecodeReadRaf,
        reason: reason.to_string(),
    }
}

impl<F: Field> ConcreteSumcheck<F> for BytecodeReadRaf<F> {
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
        challenges: &BytecodeReadRafCyclePhaseChallenges<F>,
    ) -> Result<F, VerifierError> {
        let opening_point = output_points
            .bytecode_ra()
            .first()
            .ok_or_else(|| public_input_failed("bytecode cycle produced no openings"))?;
        let r_cycle = self.r_cycle(opening_point)?;
        let stage_values_at_r_address = self
            .stage_values_at_r_address
            .ok_or_else(|| public_input_failed("bytecode table fold is unavailable"))?;
        // The cycle-dependent public factors (`stage_cycle_eqs`, the RAF terms,
        // `entry`) are exactly the committed-mode publics; combining them with the
        // construction-time address fold reproduces `read_raf_public_values`.
        let committed = bytecode::read_raf_committed_public_values::<F>(
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
        let mut stage_values = [F::zero(); 5];
        for ((stage_value, pre_cycle), stage_cycle_eq) in stage_values
            .iter_mut()
            .zip(stage_values_at_r_address)
            .zip(committed.stage_cycle_eqs)
        {
            *stage_value = pre_cycle * stage_cycle_eq;
        }
        let public_values = BytecodeReadRafPublicValues {
            stage_values,
            spartan_outer_raf: committed.spartan_outer_raf,
            spartan_shift_raf: committed.spartan_shift_raf,
            entry: committed.entry,
        };
        stage6_bytecode_read_raf_expected_output(Stage6BytecodeReadRafExpectedOutputInputs {
            dimensions: self.dimensions,
            public_values: &public_values,
            bytecode_ra: &output_values.bytecode_ra,
            gamma: challenges.gamma,
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
    /// Clear-only (empty in ZK, where `expected_output` never runs).
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

enum BytecodeReadRafCycleVariant<F: Field> {
    Full(BytecodeReadRaf<F>),
    Committed(BytecodeReadRafCommitted<F>),
}

/// The stage-6b bytecode read-RAF cycle relation, dispatching at runtime over
/// full-program mode ([`BytecodeReadRaf`]) and committed-program mode
/// ([`BytecodeReadRafCommitted`]). Lifetime-free so it can be a
/// `Stage6CyclePhaseSumchecks` member directly.
pub struct BytecodeReadRafCycle<F: Field> {
    /// The `ConcreteSumcheck` anchor symbolic (see the invariant on the impl).
    anchor: relations::bytecode::ReadRafCyclePhaseCommitted,
    variant: BytecodeReadRafCycleVariant<F>,
}

impl<F: Field> BytecodeReadRafCycle<F> {
    pub fn full(inputs: BytecodeReadRafCycleInputs<'_, F>) -> Result<Self, VerifierError> {
        Ok(Self {
            anchor: relations::bytecode::ReadRafCyclePhaseCommitted::new(inputs.dimensions),
            variant: BytecodeReadRafCycleVariant::Full(BytecodeReadRaf::new(inputs)?),
        })
    }

    pub fn committed(inputs: BytecodeReadRafCommittedCycleInputs<F>) -> Self {
        Self {
            anchor: relations::bytecode::ReadRafCyclePhaseCommitted::new(inputs.dimensions),
            variant: BytecodeReadRafCycleVariant::Committed(BytecodeReadRafCommitted::new(inputs)),
        }
    }
}

/// INVARIANT: this impl anchors `Symbolic` on the *committed* cycle symbolic for
/// both variants. That is sound because the two symbolics share `Inputs` /
/// `Outputs` / `rounds` / `degree` / `input_expression` (they differ only in the
/// `Challenges` type name and the output `Expr`), and every method that touches
/// the differing halves — `expected_output` (output `Expr`) and
/// `derive_opening_points` — is overridden to dispatch per variant, converting
/// the anchor's `Challenges` into the full variant's. It stays sound only while
/// those overrides stand and the batch does NOT enable `output_shape` (the
/// committed output `Expr` references the staged `BytecodeValStage` openings,
/// which the full mode never produces).
impl<F: Field> ConcreteSumcheck<F> for BytecodeReadRafCycle<F> {
    type Symbolic = relations::bytecode::ReadRafCyclePhaseCommitted;

    fn symbolic(&self) -> &Self::Symbolic {
        &self.anchor
    }

    fn derive_opening_points(
        &self,
        sumcheck_point: &[F],
        input_points: &BytecodeReadRafInputClaims<Vec<F>>,
    ) -> Result<BytecodeReadRafOutputClaims<Vec<F>>, VerifierError> {
        match &self.variant {
            BytecodeReadRafCycleVariant::Full(relation) => {
                relation.derive_opening_points(sumcheck_point, input_points)
            }
            BytecodeReadRafCycleVariant::Committed(relation) => {
                relation.derive_opening_points(sumcheck_point, input_points)
            }
        }
    }

    fn expected_output(
        &self,
        input_points: &BytecodeReadRafInputClaims<Vec<F>>,
        output_values: &BytecodeReadRafOutputClaims<F>,
        output_points: &BytecodeReadRafOutputClaims<Vec<F>>,
        challenges: &BytecodeReadRafCyclePhaseCommittedChallenges<F>,
    ) -> Result<F, VerifierError> {
        match &self.variant {
            BytecodeReadRafCycleVariant::Full(relation) => relation.expected_output(
                input_points,
                output_values,
                output_points,
                &BytecodeReadRafCyclePhaseChallenges {
                    gamma: challenges.gamma,
                },
            ),
            BytecodeReadRafCycleVariant::Committed(relation) => {
                relation.expected_output(input_points, output_values, output_points, challenges)
            }
        }
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
