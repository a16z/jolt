//! The stage 6b bytecode read-RAF cycle-phase sumcheck instance.
//!
//! The **cycle phase** dispatches at runtime over full-program mode
//! ([`BytecodeReadRaf`]) and committed-program mode ([`BytecodeReadRafCommitted`])
//! through [`BytecodeReadRafCycle`], whose `ConcreteSumcheck` impl is anchored on
//! the committed symbolic (see the invariant note on the impl). Its input claim is
//! the staged `BytecodeReadRafAddrClaim` intermediate produced by the stage-6a
//! address phase.

#[cfg(feature = "akita")]
use jolt_claims::protocols::jolt::lattice::geometry::LatticeBytecodeReadRafPublicValues;
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
        claim_reductions::bytecode::bytecode_val_stage_opening,
        dimensions::committed_address_chunks,
    },
    BytecodeReadRafChallenge, JoltChallengeId, JoltDerivedId, JoltRelationId,
};
use jolt_claims::{SumcheckChallenges, SymbolicSumcheck};
use jolt_field::Field;
use jolt_poly::EqPolynomial;
use jolt_riscv::JoltInstructionRow;

use crate::stages::relations::ConcreteSumcheck;
use crate::stages::BYTECODE_VAL_STAGES;
use crate::VerifierError;

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
    pub stage_cycle_points: Vec<Vec<F>>,
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
/// evaluate the publics once and reuse the [`expected_output_from_publics`] helper.
pub struct BytecodeReadRaf<F: Field> {
    symbolic: relations::bytecode::ReadRafCyclePhase,
    dimensions: BytecodeReadRafDimensions,
    r_address: Vec<F>,
    stage_cycle_points: Vec<Vec<F>>,
    entry_bytecode_index: usize,
    committed_chunk_bits: usize,
    /// The address-only bytecode-table fold: the five per-stage row values and
    /// the store flag, each folded against `eq(r_address, row)` — the pre-cycle
    /// half of the read-raf publics (the store half feeds the packed sixth
    /// stage; the base output ignores it). `None` in ZK, where
    /// `expected_output` never runs.
    stage_values_at_r_address: Option<([F; 5], F)>,
}

impl<F: Field> BytecodeReadRaf<F> {
    pub fn new(inputs: BytecodeReadRafCycleInputs<'_, F>) -> Result<Self, VerifierError> {
        let stage_values_at_r_address = inputs
            .table_fold
            .map(|fold| fold_stage_values(&inputs.r_address, fold))
            .transpose()?;
        Ok(Self {
            symbolic: relations::bytecode::ReadRafCyclePhase::new((
                inputs.dimensions,
                BYTECODE_VAL_STAGES,
            )),
            dimensions: inputs.dimensions,
            r_address: inputs.r_address,
            stage_cycle_points: inputs.stage_cycle_points,
            entry_bytecode_index: inputs.entry_bytecode_index,
            committed_chunk_bits: inputs.committed_chunk_bits,
            stage_values_at_r_address,
        })
    }
}

/// The address-only half of `read_raf_public_values`' `stage_values`, plus the
/// store-flag fold (the lattice sixth stage's address half; the base relation
/// ignores it): the bytecode rows' per-stage values (shared
/// `read_raf_stage_values` formula) folded against `eq(r_address)`. The
/// cycle-eq factors are attached later, at `expected_output` time, so the fold
/// can run before the cycle sumcheck.
fn fold_stage_values<F: Field>(
    r_address: &[F],
    fold: BytecodeReadRafTableFoldInputs<'_, F>,
) -> Result<([F; 5], F), VerifierError> {
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
    let mut store_value = F::zero();
    for ((row_values, store), eq_address) in row_values.into_iter().zip(address_eq_evals) {
        for (stage_value, row_value) in stage_values.iter_mut().zip(row_values) {
            *stage_value += row_value * eq_address;
        }
        if store {
            store_value += eq_address;
        }
    }
    Ok((stage_values, store_value))
}

fn public_input_failed(reason: impl ToString) -> VerifierError {
    VerifierError::StageClaimPublicInputFailed {
        stage: JoltRelationId::BytecodeReadRaf,
        reason: reason.to_string(),
    }
}

/// The `log_t`-variable cycle suffix of a produced `BytecodeRa` opening point
/// (`chunk ++ r_cycle`).
fn r_cycle_suffix<F: Field>(log_t: usize, opening_point: &[F]) -> Result<&[F], VerifierError> {
    opening_point
        .get(opening_point.len() - log_t..)
        .ok_or_else(|| public_input_failed("bytecode cycle opening point shorter than log_t"))
}

/// Evaluate the full-program bytecode read-RAF output expression at the produced
/// `BytecodeRa` openings and public values.
#[cfg(not(feature = "akita"))]
fn expected_output_from_publics<F: Field>(
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
            for (index, opening) in output_openings.bytecode_ra.iter().enumerate() {
                if *id == *opening {
                    return Ok(bytecode_ra[index]);
                }
            }
            Err(VerifierError::MissingOpeningClaim { id: *id })
        },
        |id| match id {
            JoltChallengeId::BytecodeReadRaf(BytecodeReadRafChallenge::Gamma) => Ok(gamma),
            _ => Err(VerifierError::MissingStageClaimChallenge { id: *id }),
        },
        |id| match id {
            JoltDerivedId::BytecodeReadRaf(public_id) => public_values
                .value(*public_id)
                .ok_or(VerifierError::MissingStageClaimDerived { id: *id }),
            _ => Err(VerifierError::MissingStageClaimDerived { id: *id }),
        },
    )
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
        let r_cycle = r_cycle_suffix(self.dimensions.log_t(), opening_point)?;
        let (stage_values_at_r_address, store_at_r_address) = self
            .stage_values_at_r_address
            .ok_or_else(|| public_input_failed("bytecode table fold is unavailable"))?;
        // The cycle-dependent public factors (`stage_cycle_eqs`, the RAF terms,
        // `entry`) are exactly the committed-mode publics; combining them with the
        // construction-time address fold reproduces the full-mode publics.
        let committed = bytecode::read_raf_committed_public_values::<F>(
            BytecodeReadRafCommittedEvaluationInputs {
                r_address: &self.r_address,
                r_cycle,
                stage_cycle_points: self.stage_cycle_points.iter().map(Vec::as_slice).collect(),
                entry_bytecode_index: self.entry_bytecode_index,
            },
        );
        let mut stage_values = [F::zero(); 5];
        for ((stage_value, pre_cycle), stage_cycle_eq) in stage_values
            .iter_mut()
            .zip(stage_values_at_r_address)
            .zip(&committed.stage_cycle_eqs)
        {
            *stage_value = pre_cycle * *stage_cycle_eq;
        }
        let base_public_values = BytecodeReadRafPublicValues {
            stage_values,
            spartan_outer_raf: committed.spartan_outer_raf,
            spartan_shift_raf: committed.spartan_shift_raf,
            entry: committed.entry,
        };
        #[cfg(not(feature = "akita"))]
        {
            let _ = store_at_r_address;
            expected_output_from_publics(
                self.dimensions,
                &base_public_values,
                &output_values.bytecode_ra,
                challenges.gamma,
            )
        }
        // The packed sixth stage: the store fold bound to the
        // `IncVirtualization` cycle point (the sixth cycle eq), resolved
        // through the six-stage cycle output expression.
        #[cfg(feature = "akita")]
        {
            let store_value = store_at_r_address
                * *committed
                    .stage_cycle_eqs
                    .get(BYTECODE_VAL_STAGES - 1)
                    .ok_or_else(|| public_input_failed("missing store stage cycle point"))?;
            let public_values = LatticeBytecodeReadRafPublicValues {
                base: base_public_values,
                store_value,
            };
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
                    for (index, opening_id) in output_openings.bytecode_ra.iter().enumerate() {
                        if *id == *opening_id {
                            return Ok(output_values.bytecode_ra[index]);
                        }
                    }
                    Err(VerifierError::MissingOpeningClaim { id: *id })
                },
                |id| match id {
                    JoltChallengeId::BytecodeReadRaf(BytecodeReadRafChallenge::Gamma) => {
                        Ok(challenges.gamma)
                    }
                    _ => Err(VerifierError::MissingStageClaimChallenge { id: *id }),
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
}

/// Construction inputs for the committed-program bytecode cycle relation.
/// One cycle point per staged val — five in base mode, six on the packed path
/// (the sixth is the `IncVirtualization` store point); the staged-val count is
/// `stage_cycle_points.len()`.
pub struct BytecodeReadRafCommittedCycleInputs<F: Field> {
    pub dimensions: BytecodeReadRafDimensions,
    pub r_address: Vec<F>,
    pub stage_cycle_points: Vec<Vec<F>>,
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
    stage_cycle_points: Vec<Vec<F>>,
    entry_bytecode_index: usize,
    committed_chunk_bits: usize,
    val_stages: Vec<F>,
}

impl<F: Field> BytecodeReadRafCommitted<F> {
    pub fn new(inputs: BytecodeReadRafCommittedCycleInputs<F>) -> Self {
        Self {
            symbolic: relations::bytecode::ReadRafCyclePhaseCommitted::new((
                inputs.dimensions,
                inputs.stage_cycle_points.len(),
            )),
            dimensions: inputs.dimensions,
            r_address: inputs.r_address,
            stage_cycle_points: inputs.stage_cycle_points,
            entry_bytecode_index: inputs.entry_bytecode_index,
            committed_chunk_bits: inputs.committed_chunk_bits,
            val_stages: inputs.val_stages,
        }
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
        let r_cycle = r_cycle_suffix(self.dimensions.log_t(), opening_point)?;
        let public_values = bytecode::read_raf_committed_public_values::<F>(
            BytecodeReadRafCommittedEvaluationInputs {
                r_address: &self.r_address,
                r_cycle,
                stage_cycle_points: self.stage_cycle_points.iter().map(Vec::as_slice).collect(),
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
/// `Stage6bSumchecks` member directly.
pub struct BytecodeReadRafCycle<F: Field> {
    /// The `ConcreteSumcheck` anchor symbolic (see the invariant on the impl).
    anchor: relations::bytecode::ReadRafCyclePhaseCommitted,
    variant: BytecodeReadRafCycleVariant<F>,
}

impl<F: Field> BytecodeReadRafCycle<F> {
    pub fn full(inputs: BytecodeReadRafCycleInputs<'_, F>) -> Result<Self, VerifierError> {
        Ok(Self {
            anchor: relations::bytecode::ReadRafCyclePhaseCommitted::new((
                inputs.dimensions,
                BYTECODE_VAL_STAGES,
            )),
            variant: BytecodeReadRafCycleVariant::Full(BytecodeReadRaf::new(inputs)?),
        })
    }

    pub fn committed(inputs: BytecodeReadRafCommittedCycleInputs<F>) -> Self {
        Self {
            anchor: relations::bytecode::ReadRafCyclePhaseCommitted::new((
                inputs.dimensions,
                inputs.stage_cycle_points.len(),
            )),
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
/// those overrides stand and the batch keeps `no_output_shape` (the
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
