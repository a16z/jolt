//! The stage 6b bytecode read-RAF cycle-phase sumcheck instance.
//!
//! The **cycle phase** dispatches at runtime over full-program mode
//! ([`BytecodeReadRaf`]) and committed-program mode ([`BytecodeReadRafCommitted`])
//! through [`BytecodeReadRafCycle`], whose `ConcreteSumcheck` impl is anchored on
//! the committed symbolic (see the invariant note on the impl). Its input claim is
//! the staged `BytecodeReadRafAddrClaim` intermediate produced by the stage-6a
//! address phase.

pub use jolt_claims::protocols::jolt::geometry::bytecode::READ_RAF_CYCLE_STAGES;
#[cfg(feature = "akita")]
pub use jolt_claims::protocols::jolt::lattice::relations::read_raf::LatticeBytecodeReadRafOutputClaims;
#[cfg(not(feature = "akita"))]
use jolt_claims::protocols::jolt::relations;
pub use jolt_claims::protocols::jolt::relations::bytecode::{
    BytecodeReadRafCyclePhaseChallenges, BytecodeReadRafCyclePhaseCommittedChallenges,
    BytecodeReadRafInputClaims, BytecodeReadRafOutputClaims,
};
use jolt_claims::protocols::jolt::{
    geometry::{
        bytecode::{
            self, BytecodeReadRafCommittedEvaluationInputs, BytecodeReadRafDimensions,
            BytecodeReadRafPublicValues, BytecodeReadRafStageValueInputs,
        },
        claim_reductions::bytecode::{bytecode_val_stage_opening, NUM_BYTECODE_VAL_STAGES},
        dimensions::committed_address_chunks,
    },
    BytecodeReadRafChallenge, JoltChallengeId, JoltDerivedId, JoltRelationId,
};
use jolt_claims::{SumcheckChallenges, SymbolicSumcheck};
use jolt_field::JoltField;
use jolt_poly::EqPolynomial;
use jolt_riscv::JoltInstructionRow;

#[cfg(feature = "field-inline")]
use crate::stages::field_inline_bytecode::FieldInlineBytecodeFold;
use crate::stages::relations::ConcreteSumcheck;
use crate::VerifierError;

#[cfg(not(feature = "akita"))]
type CycleSymbolic = relations::bytecode::ReadRafCyclePhase;
#[cfg(feature = "akita")]
type CycleSymbolic =
    jolt_claims::protocols::jolt::lattice::relations::read_raf::LatticeReadRafCyclePhase;
#[cfg(not(feature = "akita"))]
type CycleSymbolicCommitted = relations::bytecode::ReadRafCyclePhaseCommitted;
#[cfg(feature = "akita")]
type CycleSymbolicCommitted =
    jolt_claims::protocols::jolt::lattice::relations::read_raf::LatticeReadRafCyclePhaseCommitted;

/// The cycle-phase produced-claims type: `BytecodeRa` openings, plus (packed)
/// the `FusedInc` opening at the bound cycle point.
#[cfg(not(feature = "akita"))]
pub type BytecodeReadRafCycleOutputClaims<C> = BytecodeReadRafOutputClaims<C>;
#[cfg(feature = "akita")]
pub type BytecodeReadRafCycleOutputClaims<C> = LatticeBytecodeReadRafOutputClaims<C>;

/// Clear-only aux for the full-program cycle relation's bytecode-table fold:
/// the borrowed table rows plus the register points and per-stage gammas that
/// weight each row. Consumed at construction ([`BytecodeReadRaf::new`] folds the
/// table against `eq(r_address)` immediately), so nothing borrowed is stored and
/// the relation stays lifetime-free.
pub struct BytecodeReadRafTableFoldInputs<'a, F: JoltField> {
    pub bytecode: &'a [JoltInstructionRow],
    pub register_read_write_point: &'a [F],
    pub register_val_evaluation_point: &'a [F],
    /// Per-stage (1..=5) Fiat-Shamir gamma powers.
    pub stage_gammas: [&'a [F]; 5],
}

/// Construction inputs for the full-program bytecode cycle relation.
/// `stage_cycle_points` are the verifier's per-stage cycle bindings.
/// `table_fold` is `Some` only in clear mode — ZK never runs `expected_output`,
/// so it skips the `O(2^log_k)` fold entirely.
pub struct BytecodeReadRafCycleInputs<'a, F: JoltField> {
    pub dimensions: BytecodeReadRafDimensions,
    pub r_address: Vec<F>,
    pub stage_cycle_points: [Vec<F>; READ_RAF_CYCLE_STAGES],
    pub entry_bytecode_index: usize,
    pub committed_chunk_bits: usize,
    pub table_fold: Option<BytecodeReadRafTableFoldInputs<'a, F>>,
    /// Field-register access points and the stage-4/5 gamma powers.
    #[cfg(feature = "field-inline")]
    pub field_inline: FieldInlineBytecodeFold<F>,
}

fn cycle_symbolic(dimensions: BytecodeReadRafDimensions) -> CycleSymbolic {
    #[cfg(not(feature = "akita"))]
    {
        CycleSymbolic::new((dimensions, NUM_BYTECODE_VAL_STAGES))
    }
    #[cfg(feature = "akita")]
    {
        CycleSymbolic::new(dimensions)
    }
}

fn cycle_symbolic_committed(dimensions: BytecodeReadRafDimensions) -> CycleSymbolicCommitted {
    #[cfg(not(feature = "akita"))]
    {
        CycleSymbolicCommitted::new((dimensions, NUM_BYTECODE_VAL_STAGES))
    }
    #[cfg(feature = "akita")]
    {
        CycleSymbolicCommitted::new(dimensions)
    }
}

/// The stage-6b bytecode read-RAF cycle phase, full-program mode.
///
/// Its expected output is the bytecode-table public values evaluated at
/// `(r_address, r_cycle)` folded against the committed `BytecodeRa` product — the
/// same quantity `read_raf`'s output expression computes. The table depends only
/// on the address variables, so the `O(2^log_k)` fold against `eq(r_address)` runs
/// once at construction (clear mode only) and the cycle-dependent factors are
/// attached in [`ConcreteSumcheck::expected_output`], which it OVERRIDES to
/// evaluate the publics once and reuse the `expected_output_from_publics` helper.
#[derive(Clone)]
pub struct BytecodeReadRaf<F: JoltField> {
    symbolic: CycleSymbolic,
    dimensions: BytecodeReadRafDimensions,
    r_address: Vec<F>,
    stage_cycle_points: [Vec<F>; READ_RAF_CYCLE_STAGES],
    entry_bytecode_index: usize,
    committed_chunk_bits: usize,
    /// The address-only bytecode-table fold: each staged row value (five base
    /// stages, plus the lattice store stage — whose fold and complement feed
    /// the four fused-inc consumer stages) folded against
    /// `eq(r_address, row)` — the pre-cycle half of the read-raf publics.
    /// `None` in ZK, where `expected_output` never runs.
    stage_values_at_r_address: Option<FoldedStageValues<F>>,
    /// Field-register access points and the stage-4/5 gamma powers.
    #[cfg(feature = "field-inline")]
    field_inline: FieldInlineBytecodeFold<F>,
}

impl<F: JoltField> BytecodeReadRaf<F> {
    pub fn new(inputs: BytecodeReadRafCycleInputs<'_, F>) -> Result<Self, VerifierError> {
        let stage_values_at_r_address = inputs
            .table_fold
            .map(|fold| {
                fold_stage_values(
                    &inputs.r_address,
                    fold,
                    #[cfg(feature = "field-inline")]
                    &inputs.field_inline,
                )
            })
            .transpose()?;
        Ok(Self {
            symbolic: cycle_symbolic(inputs.dimensions),
            dimensions: inputs.dimensions,
            r_address: inputs.r_address,
            stage_cycle_points: inputs.stage_cycle_points,
            entry_bytecode_index: inputs.entry_bytecode_index,
            committed_chunk_bits: inputs.committed_chunk_bits,
            stage_values_at_r_address,
            #[cfg(feature = "field-inline")]
            field_inline: inputs.field_inline,
        })
    }

    #[cfg(feature = "field-inline")]
    fn field_inline_stage_values(&self, r_cycle: &[F]) -> Result<[F; 5], VerifierError> {
        let [stage1, stage2, stage3, read_write, val_evaluation] = self
            .stage_values_at_r_address
            .ok_or_else(|| public_input_failed("bytecode table fold is unavailable"))?
            .field_registers;
        Ok([
            stage1,
            stage2,
            stage3,
            read_write * EqPolynomial::<F>::mle(&self.field_inline.read_write_cycle, r_cycle),
            val_evaluation
                * EqPolynomial::<F>::mle(&self.field_inline.val_evaluation_cycle, r_cycle),
        ])
    }
}

#[derive(Clone, Copy)]
struct FoldedStageValues<F: JoltField> {
    ordinary: [F; NUM_BYTECODE_VAL_STAGES],
    #[cfg(feature = "field-inline")]
    field_registers: [F; 5],
}

/// The address-only half of the staged read-raf publics: the bytecode rows'
/// per-stage values (shared `read_raf_stage_values` formula, which carries
/// the lattice store stage as its last element) folded against
/// `eq(r_address)`. The cycle-eq factors are attached later, at
/// `expected_output` time, so the fold can run before the cycle sumcheck.
fn fold_stage_values<F: JoltField>(
    r_address: &[F],
    fold: BytecodeReadRafTableFoldInputs<'_, F>,
    #[cfg(feature = "field-inline")] field_inline: &FieldInlineBytecodeFold<F>,
) -> Result<FoldedStageValues<F>, VerifierError> {
    let expected_domain = u32::try_from(r_address.len())
        .ok()
        .and_then(|address_bits| 1usize.checked_shl(address_bits))
        .ok_or_else(|| public_input_failed("bytecode address domain overflows"))?;
    if fold.bytecode.len() != expected_domain {
        return Err(public_input_failed(format!(
            "bytecode table has {} rows, expected the address domain {expected_domain}",
            fold.bytecode.len()
        )));
    }
    let address_eq_evals = EqPolynomial::<F>::evals(r_address, None);
    let bytecode_rows = fold.bytecode;
    let row_values = bytecode::read_raf_stage_values(BytecodeReadRafStageValueInputs {
        bytecode: bytecode_rows,
        register_read_write_point: fold.register_read_write_point,
        register_val_evaluation_point: fold.register_val_evaluation_point,
        stage1_gammas: fold.stage_gammas[0],
        stage2_gammas: fold.stage_gammas[1],
        stage3_gammas: fold.stage_gammas[2],
        stage4_gammas: fold.stage_gammas[3],
        stage5_gammas: fold.stage_gammas[4],
    });
    let mut stage_values = [F::zero(); NUM_BYTECODE_VAL_STAGES];
    for (row_values, eq_address) in row_values.into_iter().zip(&address_eq_evals) {
        for (stage_value, row_value) in stage_values.iter_mut().zip(row_values) {
            *stage_value += row_value * *eq_address;
        }
    }
    #[cfg(feature = "field-inline")]
    let field_registers = {
        use jolt_claims::protocols::field_inline::geometry::bytecode::{
            read_raf_stage_values, FieldInlineBytecodeReadRafStageValueInputs,
        };
        let rows = read_raf_stage_values(FieldInlineBytecodeReadRafStageValueInputs {
            bytecode: bytecode_rows,
            field_register_read_write_point: &field_inline.read_write_address,
            field_register_val_evaluation_point: &field_inline.val_evaluation_address,
            stage4_gammas: &field_inline.gammas.stage4,
            stage5_gammas: &field_inline.gammas.stage5,
        });
        let mut values = [F::zero(); 5];
        for (row, eq_address) in rows.into_iter().zip(address_eq_evals) {
            for (value, row_value) in values.iter_mut().zip(row) {
                *value += row_value * eq_address;
            }
        }
        values
    };
    Ok(FoldedStageValues {
        ordinary: stage_values,
        #[cfg(feature = "field-inline")]
        field_registers,
    })
}

fn public_input_failed(reason: impl ToString) -> VerifierError {
    VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::BytecodeReadRaf,
        reason: reason.to_string(),
    }
}

/// The `log_t`-variable cycle suffix of a produced `BytecodeRa` opening point
/// (`chunk ++ r_cycle`).
fn r_cycle_suffix<F: JoltField>(log_t: usize, opening_point: &[F]) -> Result<&[F], VerifierError> {
    opening_point
        .len()
        .checked_sub(log_t)
        .and_then(|start| opening_point.get(start..))
        .ok_or_else(|| public_input_failed("bytecode cycle opening point shorter than log_t"))
}

/// Evaluate the full-program bytecode read-RAF output expression at the produced
/// `BytecodeRa` openings and public values.
#[cfg(not(feature = "akita"))]
#[expect(
    clippy::wildcard_enum_match_arm,
    reason = "fail-closed: ids not owned by this relation resolve to a missing-claim error"
)]
fn expected_output_from_publics<F: JoltField>(
    dimensions: BytecodeReadRafDimensions,
    public_values: &bytecode::BytecodeReadRafPublicValues<F>,
    bytecode_ra: &[F],
    gamma: F,
) -> Result<F, VerifierError> {
    let output_openings = bytecode::read_raf_output_openings(dimensions);
    if bytecode_ra.len() != output_openings.bytecode_ra.len() {
        return Err(VerifierError::StageClaimPublicInputFailed {
            stage: JoltRelationId::BytecodeReadRaf,
            reason: format!(
                "bytecode RA claim count mismatch: expected {}, got {}",
                output_openings.bytecode_ra.len(),
                bytecode_ra.len()
            ),
        });
    }
    let relation = relations::bytecode::ReadRaf::new(dimensions);
    relation.output_expression::<F>().try_evaluate(
        |id| {
            for (opening, value) in output_openings.bytecode_ra.iter().zip(bytecode_ra) {
                if *id == *opening {
                    return Ok(*value);
                }
            }
            Err(VerifierError::MissingOpeningClaim { id: (*id).into() })
        },
        |id| match id {
            JoltChallengeId::BytecodeReadRaf(BytecodeReadRafChallenge::Gamma) => Ok(gamma),
            _ => Err(VerifierError::MissingStageClaimChallenge { id: (*id).into() }),
        },
        |id| match id {
            JoltDerivedId::BytecodeReadRaf(public_id) => public_values
                .value(*public_id)
                .ok_or(VerifierError::MissingStageClaimDerived { id: (*id).into() }),
            _ => Err(VerifierError::MissingStageClaimDerived { id: (*id).into() }),
        },
    )
}

impl<F: JoltField> ConcreteSumcheck<F> for BytecodeReadRaf<F> {
    type Symbolic = CycleSymbolic;

    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }

    fn derive_opening_points(
        &self,
        sumcheck_point: &[F],
        _input_points: &BytecodeReadRafInputClaims<Vec<F>>,
    ) -> Result<BytecodeReadRafCycleOutputClaims<Vec<F>>, VerifierError> {
        let r_cycle = sumcheck_point.iter().rev().copied().collect::<Vec<_>>();
        derive_cycle_opening_points(&self.r_address, self.committed_chunk_bits, r_cycle)
    }

    #[cfg_attr(
        feature = "akita",
        expect(
            clippy::wildcard_enum_match_arm,
            reason = "fail-closed: ids not owned by this relation resolve to a missing-claim error"
        )
    )]
    fn expected_output(
        &self,
        _input_points: &BytecodeReadRafInputClaims<Vec<F>>,
        output_values: &BytecodeReadRafCycleOutputClaims<F>,
        output_points: &BytecodeReadRafCycleOutputClaims<Vec<F>>,
        challenges: &BytecodeReadRafCyclePhaseChallenges<F>,
    ) -> Result<F, VerifierError> {
        let opening_point = output_points
            .bytecode_ra()
            .first()
            .ok_or_else(|| public_input_failed("bytecode cycle produced no openings"))?;
        let r_cycle = r_cycle_suffix(self.dimensions.log_t(), opening_point)?;
        let stage_values_at_r_address = self
            .stage_values_at_r_address
            .ok_or_else(|| public_input_failed("bytecode table fold is unavailable"))?
            .ordinary;
        // The cycle-dependent public factors (`stage_cycle_eqs`, the RAF terms,
        // `entry`) are exactly the committed-mode publics; combining them with the
        // construction-time address fold reproduces the full-mode publics.
        let committed = bytecode::read_raf_committed_public_values::<F>(
            BytecodeReadRafCommittedEvaluationInputs {
                r_address: &self.r_address,
                r_cycle,
                stage_cycle_points: self.stage_cycle_points.each_ref().map(Vec::as_slice),
                entry_bytecode_index: self.entry_bytecode_index,
            },
        );
        // The base monolith publics carry the five gamma'd stages; the
        // lattice store fold feeds the fused-inc stage resolution below.
        let mut folded_stage_values = [F::zero(); bytecode::BYTECODE_STAGE_GAMMA_COUNTS.len()];
        for ((folded, stage_value), stage_cycle_eq) in folded_stage_values
            .iter_mut()
            .zip(&stage_values_at_r_address)
            .zip(&committed.stage_cycle_eqs)
        {
            *folded = *stage_value * *stage_cycle_eq;
        }
        let base_public_values = BytecodeReadRafPublicValues {
            stage_values: folded_stage_values,
            spartan_outer_raf: committed.spartan_outer_raf,
            spartan_shift_raf: committed.spartan_shift_raf,
            entry: committed.entry,
        };
        // The composed publics: the field-register stage values (already
        // cycle-weighted per stage) add onto the ordinary staged publics, so the same `Σ
        // γ^stage · StageValue(stage)` output fold carries both families under the existing
        // outer gamma powers.
        #[cfg(feature = "field-inline")]
        let base_public_values = {
            let field_inline_stage_values = self.field_inline_stage_values(r_cycle)?;
            let mut composed = base_public_values;
            for (stage_value, field_inline_value) in composed
                .stage_values
                .iter_mut()
                .zip(field_inline_stage_values)
            {
                *stage_value += field_inline_value;
            }
            composed
        };
        #[cfg(not(feature = "akita"))]
        {
            expected_output_from_publics(
                self.dimensions,
                &base_public_values,
                &output_values.bytecode_ra,
                challenges.gamma,
            )
        }
        // The packed fused-inc stages: the store fold (and its complement)
        // bound to the four consuming relations' cycle points, resolved
        // through the lattice cycle output expression against the `FusedInc`
        // opening.
        #[cfg(feature = "akita")]
        {
            // The four fused-inc consumer stages resolve from the staged store
            // fold (its complement for the register legs) against their own
            // cycle eqs; the `FusedInc` factor is the relation's own opening.
            let base_stages = NUM_BYTECODE_VAL_STAGES - 1;
            // The store stage is the last staged wire (index `base_stages`).
            let store_at_r_address = *stage_values_at_r_address
                .last()
                .ok_or_else(|| public_input_failed("bytecode stage fold is empty"))?;
            let fused_stage_value = |stage: usize| -> Result<F, VerifierError> {
                let address_fold = if stage < base_stages + 2 {
                    store_at_r_address
                } else {
                    F::one() - store_at_r_address
                };
                let cycle_eq = committed
                    .stage_cycle_eqs
                    .get(stage)
                    .ok_or_else(|| public_input_failed("missing fused stage cycle point"))?;
                Ok(address_fold * *cycle_eq)
            };
            let public_values = base_public_values;
            let output_openings = bytecode::read_raf_output_openings(self.dimensions);
            if output_values.bytecode_ra.len() != output_openings.bytecode_ra.len() {
                return Err(public_input_failed(format!(
                    "bytecode RA claim count mismatch: expected {}, got {}",
                    output_openings.bytecode_ra.len(),
                    output_values.bytecode_ra.len()
                )));
            }
            self.symbolic().output_expression::<F>().try_evaluate(
                |id| {
                    if *id == bytecode::fused_inc_read_raf_opening() {
                        return Ok(output_values.fused_inc);
                    }
                    for (opening_id, value) in output_openings
                        .bytecode_ra
                        .iter()
                        .zip(&output_values.bytecode_ra)
                    {
                        if *id == *opening_id {
                            return Ok(*value);
                        }
                    }
                    Err(VerifierError::MissingOpeningClaim { id: (*id).into() })
                },
                |id| match id {
                    JoltChallengeId::BytecodeReadRaf(BytecodeReadRafChallenge::Gamma) => {
                        Ok(challenges.gamma)
                    }
                    _ => Err(VerifierError::MissingStageClaimChallenge { id: (*id).into() }),
                },
                |id| match id {
                    JoltDerivedId::BytecodeReadRaf(
                        jolt_claims::protocols::jolt::BytecodeReadRafPublic::StageValue(stage),
                    ) if *stage >= base_stages => fused_stage_value(*stage),
                    JoltDerivedId::BytecodeReadRaf(public_id) => public_values
                        .value(*public_id)
                        .ok_or(VerifierError::MissingStageClaimDerived { id: (*id).into() }),
                    _ => Err(VerifierError::MissingStageClaimDerived { id: (*id).into() }),
                },
            )
        }
    }
}

// The dory-shaped composition pins (base input-claims struct, five stage points); the packed
// composition is covered by the prover's field-inline stage round-trips and the packed e2e
// suite.
#[cfg(all(test, feature = "field-inline", not(feature = "akita")))]
#[expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects,
    clippy::as_conversions,
    reason = "tests index their own fixed-size fixtures and use plain arithmetic on fixture data"
)]
mod field_inline_tests {
    use super::*;
    use crate::stages::field_inline_bytecode::{
        field_inline_stage_gamma_powers, FieldInlineBytecodeFold,
    };
    use jolt_claims::protocols::field_inline::geometry::bytecode as field_inline_geometry;
    use jolt_claims::protocols::field_inline::FIELD_REGISTERS_LOG_K;
    use jolt_claims::protocols::jolt::geometry::bytecode::BytecodeReadRafEvaluationInputs;
    use jolt_claims::protocols::jolt::relations::bytecode::BytecodeReadRafAddressPhaseChallenges;
    use jolt_field::{Fr, Ring};
    use jolt_riscv::{JoltInstructionKind as Kind, NormalizedOperands};

    fn fr(value: u64) -> Fr {
        Fr::from_u64(value)
    }

    fn point(start: u64, len: usize) -> Vec<Fr> {
        (0..len as u64).map(|i| fr(start + i)).collect()
    }

    fn bytecode_rows() -> Vec<JoltInstructionRow> {
        let mut rows = vec![JoltInstructionRow::default(); 4];
        rows[0] = JoltInstructionRow {
            instruction_kind: Kind::ADD,
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
        };
        rows[1] = JoltInstructionRow {
            instruction_kind: Kind::FIELD_MUL,
            operands: NormalizedOperands {
                rd: Some(1),
                rs1: Some(2),
                rs2: Some(3),
                imm: 0,
            },
            ..Default::default()
        };
        rows
    }

    fn address_challenges() -> BytecodeReadRafAddressPhaseChallenges<Fr> {
        BytecodeReadRafAddressPhaseChallenges {
            gamma: fr(501),
            stage1_gamma: fr(502),
            stage2_gamma: fr(503),
            stage3_gamma: fr(504),
            stage4_gamma: fr(505),
            stage5_gamma: fr(506),
        }
    }

    /// The composed full-mode `expected_output` equals the from-scratch fold: the ordinary
    /// full-program publics (evaluated through the one-shot monolith helper — a different
    /// assembly path than the relation's construction-time fold plus committed cycle publics)
    /// with the field-register publics added stage-for-stage, folded through the same
    /// output expression (spec: `field-inline-protocol.md`, "Stage 6 Composition" — the output
    /// stays the `BytecodeRa(i)` product, with public stage values augmented by the field-register
    /// evaluation).
    #[test]
    fn composed_expected_output_matches_from_scratch_public_fold() {
        let log_t = 2usize;
        let log_k = 2usize;
        let dimensions = BytecodeReadRafDimensions::new(log_t, log_k, 2);
        let r_address = point(10, log_k);
        let stage_cycle_points: [Vec<Fr>; READ_RAF_CYCLE_STAGES] =
            core::array::from_fn(|stage| point(20 + 10 * stage as u64, log_t));
        let register_read_write_point = point(70, 4);
        let register_val_evaluation_point = point(80, 4);
        let field_read_write_address = point(90, FIELD_REGISTERS_LOG_K);
        let field_read_write_cycle = point(100, log_t);
        let field_val_evaluation_address = point(110, FIELD_REGISTERS_LOG_K);
        let field_val_evaluation_cycle = point(120, log_t);
        let challenges = address_challenges();
        let stage_gammas = challenges.stage_gamma_powers();
        let field_gammas = field_inline_stage_gamma_powers(&challenges);
        let bytecode = bytecode_rows();
        let entry_bytecode_index = 1usize;

        let relation = BytecodeReadRaf::new(BytecodeReadRafCycleInputs {
            dimensions,
            r_address: r_address.clone(),
            stage_cycle_points: stage_cycle_points.clone(),
            entry_bytecode_index,
            committed_chunk_bits: 1,
            table_fold: Some(BytecodeReadRafTableFoldInputs {
                bytecode: &bytecode,
                register_read_write_point: &register_read_write_point,
                register_val_evaluation_point: &register_val_evaluation_point,
                stage_gammas: stage_gammas.each_ref().map(Vec::as_slice),
            }),
            field_inline: FieldInlineBytecodeFold {
                read_write_address: field_read_write_address.clone(),
                read_write_cycle: field_read_write_cycle.clone(),
                val_evaluation_address: field_val_evaluation_address.clone(),
                val_evaluation_cycle: field_val_evaluation_cycle.clone(),
                gammas: field_gammas.clone(),
            },
        })
        .unwrap();

        let sumcheck_point = point(130, log_t);
        let input_points = BytecodeReadRafInputClaims {
            address_phase: Vec::new(),
        };
        let output_points = relation
            .derive_opening_points(&sumcheck_point, &input_points)
            .unwrap();
        let output_values = BytecodeReadRafOutputClaims {
            bytecode_ra: vec![fr(601), fr(602)],
        };
        let cycle_challenges = BytecodeReadRafCyclePhaseChallenges {
            gamma: challenges.gamma,
        };
        let composed = relation
            .expected_output(
                &input_points,
                &output_values,
                &output_points,
                &cycle_challenges,
            )
            .unwrap();

        // From scratch: the one-shot monolith publics plus the field-register
        // publics, stage-for-stage, through the shared output fold.
        let r_cycle: Vec<Fr> = sumcheck_point.iter().rev().copied().collect();
        let mut publics = bytecode::read_raf_public_values(BytecodeReadRafEvaluationInputs {
            bytecode: &bytecode,
            r_address: &r_address,
            r_cycle: &r_cycle,
            stage_cycle_points: stage_cycle_points.each_ref().map(Vec::as_slice),
            register_read_write_point: &register_read_write_point,
            register_val_evaluation_point: &register_val_evaluation_point,
            entry_bytecode_index,
            stage1_gammas: &stage_gammas[0],
            stage2_gammas: &stage_gammas[1],
            stage3_gammas: &stage_gammas[2],
            stage4_gammas: &stage_gammas[3],
            stage5_gammas: &stage_gammas[4],
        })
        .unwrap();
        let field_publics = field_inline_geometry::read_raf_public_values(
            field_inline_geometry::FieldInlineBytecodeReadRafEvaluationInputs {
                bytecode: &bytecode,
                r_address: &r_address,
                r_cycle: &r_cycle,
                field_register_read_write_point: &field_read_write_address,
                field_register_read_write_cycle_point: &field_read_write_cycle,
                field_register_val_evaluation_point: &field_val_evaluation_address,
                field_register_val_evaluation_cycle_point: &field_val_evaluation_cycle,
                stage4_gammas: &field_gammas.stage4,
                stage5_gammas: &field_gammas.stage5,
            },
        )
        .unwrap();
        // The active field-inline row contributes to stages 4/5; a vanishing contribution
        // would make this pin vacuous.
        assert!(field_publics
            .stage_values
            .iter()
            .any(|value| *value != fr(0)));
        for (stage_value, field_value) in publics
            .stage_values
            .iter_mut()
            .zip(field_publics.stage_values)
        {
            *stage_value += field_value;
        }
        let expected = expected_output_from_publics(
            dimensions,
            &publics,
            &output_values.bytecode_ra,
            challenges.gamma,
        )
        .unwrap();

        assert_eq!(composed, expected);
    }
}

/// Derive the cycle-phase produced opening points: one `(chunk ++ r_cycle)`
/// point per committed `BytecodeRa` chunk, plus (packed) the `FusedInc` cycle
/// point.
fn derive_cycle_opening_points<F: JoltField>(
    r_address: &[F],
    committed_chunk_bits: usize,
    r_cycle: Vec<F>,
) -> Result<BytecodeReadRafCycleOutputClaims<Vec<F>>, VerifierError> {
    let bytecode_ra = committed_address_chunks(r_address, committed_chunk_bits)
        .into_iter()
        .map(|chunk| [chunk.as_slice(), r_cycle.as_slice()].concat())
        .collect();
    #[cfg(not(feature = "akita"))]
    {
        Ok(BytecodeReadRafOutputClaims { bytecode_ra })
    }
    #[cfg(feature = "akita")]
    {
        Ok(LatticeBytecodeReadRafOutputClaims {
            bytecode_ra,
            fused_inc: r_cycle,
        })
    }
}

/// Construction inputs for the committed-program bytecode cycle relation.
/// One cycle point per relation stage — five in base mode, nine on the packed
/// path (the four fused-inc consumer points follow the base five).
pub struct BytecodeReadRafCommittedCycleInputs<F: JoltField> {
    pub dimensions: BytecodeReadRafDimensions,
    pub r_address: Vec<F>,
    pub stage_cycle_points: [Vec<F>; READ_RAF_CYCLE_STAGES],
    pub entry_bytecode_index: usize,
    pub committed_chunk_bits: usize,
    /// The staged `BytecodeValClaim` opening values from the address phase.
    /// Clear-only (empty in ZK, where `expected_output` never runs).
    pub val_stages: Vec<F>,
}

/// The stage-6b bytecode read-RAF cycle phase, committed-program mode.
///
/// Mirrors [`BytecodeReadRaf`] but folds the staged `BytecodeValClaim` openings
/// into the output expression and draws its publics from a committed bytecode
/// evaluation (`read_raf_committed_public_values`) rather than the full bytecode
/// table. Like the full-mode relation it OVERRIDES
/// [`ConcreteSumcheck::expected_output`]: the staged Val openings are inputs mixed
/// into the output, and the committed public values are evaluated once.
#[derive(Clone)]
pub struct BytecodeReadRafCommitted<F: JoltField> {
    symbolic: CycleSymbolicCommitted,
    dimensions: BytecodeReadRafDimensions,
    r_address: Vec<F>,
    stage_cycle_points: [Vec<F>; READ_RAF_CYCLE_STAGES],
    entry_bytecode_index: usize,
    committed_chunk_bits: usize,
    val_stages: Vec<F>,
}

impl<F: JoltField> BytecodeReadRafCommitted<F> {
    pub fn new(inputs: BytecodeReadRafCommittedCycleInputs<F>) -> Self {
        Self {
            symbolic: cycle_symbolic_committed(inputs.dimensions),
            dimensions: inputs.dimensions,
            r_address: inputs.r_address,
            stage_cycle_points: inputs.stage_cycle_points,
            entry_bytecode_index: inputs.entry_bytecode_index,
            committed_chunk_bits: inputs.committed_chunk_bits,
            val_stages: inputs.val_stages,
        }
    }
}

impl<F: JoltField> ConcreteSumcheck<F> for BytecodeReadRafCommitted<F> {
    type Symbolic = CycleSymbolicCommitted;

    fn symbolic(&self) -> &Self::Symbolic {
        &self.symbolic
    }

    fn derive_opening_points(
        &self,
        sumcheck_point: &[F],
        _input_points: &BytecodeReadRafInputClaims<Vec<F>>,
    ) -> Result<BytecodeReadRafCycleOutputClaims<Vec<F>>, VerifierError> {
        let r_cycle = sumcheck_point.iter().rev().copied().collect::<Vec<_>>();
        derive_cycle_opening_points(&self.r_address, self.committed_chunk_bits, r_cycle)
    }

    #[expect(
        clippy::wildcard_enum_match_arm,
        reason = "fail-closed: ids not owned by this relation resolve to a missing-claim error"
    )]
    fn expected_output(
        &self,
        _input_points: &BytecodeReadRafInputClaims<Vec<F>>,
        output_values: &BytecodeReadRafCycleOutputClaims<F>,
        output_points: &BytecodeReadRafCycleOutputClaims<Vec<F>>,
        challenges: &BytecodeReadRafCyclePhaseCommittedChallenges<F>,
    ) -> Result<F, VerifierError> {
        let opening_point = output_points
            .bytecode_ra()
            .first()
            .map(Vec::as_slice)
            .ok_or_else(|| public_input_failed("bytecode cycle produced no openings"))?;
        let r_cycle = r_cycle_suffix(self.dimensions.log_t(), opening_point)?;
        let public_values = bytecode::read_raf_committed_public_values::<F>(
            BytecodeReadRafCommittedEvaluationInputs {
                r_address: &self.r_address,
                r_cycle,
                stage_cycle_points: self.stage_cycle_points.each_ref().map(Vec::as_slice),
                entry_bytecode_index: self.entry_bytecode_index,
            },
        );
        let output_openings = bytecode::read_raf_output_openings(self.dimensions);
        self.symbolic().output_expression::<F>().try_evaluate(
            |id| {
                #[cfg(feature = "akita")]
                if *id == bytecode::fused_inc_read_raf_opening() {
                    return Ok(output_values.fused_inc);
                }
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
                            .ok_or(VerifierError::MissingOpeningClaim { id: (*id).into() });
                    }
                }
                Err(VerifierError::MissingOpeningClaim { id: (*id).into() })
            },
            |id| {
                challenges
                    .resolve_challenge(id)
                    .ok_or(VerifierError::MissingStageClaimChallenge { id: (*id).into() })
            },
            |id| match id {
                JoltDerivedId::BytecodeReadRaf(public_id) => public_values
                    .value(*public_id)
                    .ok_or(VerifierError::MissingStageClaimDerived { id: (*id).into() }),
                _ => Err(VerifierError::MissingStageClaimDerived { id: (*id).into() }),
            },
        )
    }
}

#[derive(Clone)]
enum BytecodeReadRafCycleVariant<F: JoltField> {
    Full(BytecodeReadRaf<F>),
    Committed(BytecodeReadRafCommitted<F>),
}

/// The stage-6b bytecode read-RAF cycle relation, dispatching at runtime over
/// full-program mode ([`BytecodeReadRaf`]) and committed-program mode
/// ([`BytecodeReadRafCommitted`]). Lifetime-free so it can be a
/// `Stage6bSumchecks` member directly.
#[derive(Clone)]
pub struct BytecodeReadRafCycle<F: JoltField> {
    /// The `ConcreteSumcheck` anchor symbolic (see the invariant on the impl).
    anchor: CycleSymbolicCommitted,
    variant: BytecodeReadRafCycleVariant<F>,
}

impl<F: JoltField> BytecodeReadRafCycle<F> {
    pub fn full(inputs: BytecodeReadRafCycleInputs<'_, F>) -> Result<Self, VerifierError> {
        Ok(Self {
            anchor: cycle_symbolic_committed(inputs.dimensions),
            variant: BytecodeReadRafCycleVariant::Full(BytecodeReadRaf::new(inputs)?),
        })
    }

    pub fn committed(inputs: BytecodeReadRafCommittedCycleInputs<F>) -> Self {
        Self {
            anchor: cycle_symbolic_committed(inputs.dimensions),
            variant: BytecodeReadRafCycleVariant::Committed(BytecodeReadRafCommitted::new(inputs)),
        }
    }
}

impl<F: JoltField> BytecodeReadRafCycle<F> {
    pub fn dimensions(&self) -> BytecodeReadRafDimensions {
        match &self.variant {
            BytecodeReadRafCycleVariant::Full(relation) => relation.dimensions,
            BytecodeReadRafCycleVariant::Committed(relation) => relation.dimensions,
        }
    }

    pub fn r_address(&self) -> &[F] {
        match &self.variant {
            BytecodeReadRafCycleVariant::Full(relation) => &relation.r_address,
            BytecodeReadRafCycleVariant::Committed(relation) => &relation.r_address,
        }
    }

    pub fn stage_cycle_points(&self) -> &[Vec<F>; READ_RAF_CYCLE_STAGES] {
        match &self.variant {
            BytecodeReadRafCycleVariant::Full(relation) => &relation.stage_cycle_points,
            BytecodeReadRafCycleVariant::Committed(relation) => &relation.stage_cycle_points,
        }
    }

    pub fn entry_bytecode_index(&self) -> usize {
        match &self.variant {
            BytecodeReadRafCycleVariant::Full(relation) => relation.entry_bytecode_index,
            BytecodeReadRafCycleVariant::Committed(relation) => relation.entry_bytecode_index,
        }
    }

    pub fn committed_chunk_bits(&self) -> usize {
        match &self.variant {
            BytecodeReadRafCycleVariant::Full(relation) => relation.committed_chunk_bits,
            BytecodeReadRafCycleVariant::Committed(relation) => relation.committed_chunk_bits,
        }
    }

    /// The field-register opening points and gamma powers used by the cycle kernel.
    /// Full mode only: committed-program mode cannot anchor the field-inline selectors (see
    /// [`field_inline::committed_program_rejection`](crate::stages::stage6b::field_inline)),
    /// and the stage-6 batch build already rejects it, so this arm is fail-closed rather than
    /// reachable.
    #[cfg(feature = "field-inline")]
    pub fn field_inline_fold(&self) -> Result<&FieldInlineBytecodeFold<F>, VerifierError> {
        match &self.variant {
            BytecodeReadRafCycleVariant::Full(relation) => Ok(&relation.field_inline),
            BytecodeReadRafCycleVariant::Committed(_) => {
                Err(crate::stages::stage6b::field_inline::committed_program_rejection())
            }
        }
    }

    #[cfg(feature = "field-inline")]
    pub fn field_inline_stage_values_at_r_address(&self) -> Result<[F; 5], VerifierError> {
        match &self.variant {
            BytecodeReadRafCycleVariant::Full(relation) => relation
                .stage_values_at_r_address
                .map(|values| values.field_registers)
                .ok_or_else(|| public_input_failed("bytecode table fold is unavailable")),
            BytecodeReadRafCycleVariant::Committed(_) => {
                Err(crate::stages::stage6b::field_inline::committed_program_rejection())
            }
        }
    }

    /// The address-only bytecode-table fold at `r_address` — the constant
    /// `BytecodeValClaim` values the cycle kernel's tables carry. Full mode
    /// computes the fold at construction (clear only); committed mode's
    /// constants ARE the stage-6a staged raw values.
    pub fn stage_values_at_r_address(&self) -> Result<[F; NUM_BYTECODE_VAL_STAGES], VerifierError> {
        match &self.variant {
            BytecodeReadRafCycleVariant::Full(relation) => relation
                .stage_values_at_r_address
                .map(|values| values.ordinary)
                .ok_or_else(|| public_input_failed("bytecode table fold is unavailable")),
            BytecodeReadRafCycleVariant::Committed(relation) => {
                let staged: &[F] = &relation.val_stages;
                let mut stage_values = [F::zero(); NUM_BYTECODE_VAL_STAGES];
                if staged.len() != stage_values.len() {
                    return Err(public_input_failed(format!(
                        "expected {NUM_BYTECODE_VAL_STAGES} staged bytecode val stages, got {}",
                        staged.len()
                    )));
                }
                stage_values.copy_from_slice(staged);
                Ok(stage_values)
            }
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
/// those overrides stand and the batch keeps `no_output_shape` (the
/// committed output `Expr` references the staged `BytecodeValClaim` openings,
/// which the full mode never produces).
impl<F: JoltField> ConcreteSumcheck<F> for BytecodeReadRafCycle<F> {
    type Symbolic = CycleSymbolicCommitted;

    fn symbolic(&self) -> &Self::Symbolic {
        &self.anchor
    }

    fn derive_opening_points(
        &self,
        sumcheck_point: &[F],
        input_points: &BytecodeReadRafInputClaims<Vec<F>>,
    ) -> Result<BytecodeReadRafCycleOutputClaims<Vec<F>>, VerifierError> {
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
        output_values: &BytecodeReadRafCycleOutputClaims<F>,
        output_points: &BytecodeReadRafCycleOutputClaims<Vec<F>>,
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
