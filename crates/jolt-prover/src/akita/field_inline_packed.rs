//! The packed FR prover seam: limb-column extraction from the FR oracle via
//! the shared balanced-digit encoder, the stage-0 limb-object commit on the
//! trace setup, the reference kernel for the FR limb reconstruction member,
//! and the stage-8 packed opening. `mod.rs`, `stage0.rs`,
//! `reconstruction.rs`, `stage8.rs`, and `prover.rs` interact with the
//! packed FR protocol only through this module (the
//! `jolt_verifier::stages::stage8::field_inline_packed` seam's prover half).
//!
//! The reconstruction kernel materializes every limb column dense over the
//! `(digit-value ‖ cycle)` cell domain — the naive reference tier, like the
//! sibling reconstruction kernels; a sparse optimized kernel is the upgrade
//! path (the columns hold at most one hot cell per cycle).

use std::sync::Arc;

use jolt_akita::{TraceOneHotCommitment, TraceOneHotRows};
use jolt_claims::lattice::{balanced_inc_value, BalancedIncChunking};
use jolt_claims::protocols::field_inline::lattice::{
    canonical_limbs, column_selected_row, field_inc_limb_columns, recomposition_coefficient,
    FieldIncLimbPackingPlan, FieldIncLimbReconstructionOutputClaims,
};
use jolt_claims::protocols::field_inline::{
    FieldIncLimbReconstructionPublic, FieldInlineCommittedPolynomial, FieldInlineDerivedId,
    FieldInlinePolynomialId,
};
use jolt_claims::protocols::jolt::JoltOneHotConfig;
use jolt_field::JoltField;
use jolt_kernels::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};
use jolt_openings::CommitmentScheme;
use jolt_poly::{boolean_point_msb, BindingOrder, EqPolynomial, Polynomial, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_transcript::Transcript;
use jolt_verifier::stages::relations::{
    ConcreteSumcheck as _, ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckInputPoints,
    SumcheckOutputClaims, SumcheckOutputPoints,
};
use jolt_verifier::stages::stage8::field_inline_packed::{
    field_inc_limb_packed_claims, limb_plan, FieldIncLimbReconstructionInstance,
};
use jolt_verifier::stages::stage8::reconstruction::ReconstructionClearOutput;
use jolt_verifier::VerifierError;
use jolt_witness::{JoltWitnessPlane, WitnessError};

use super::reconstruction::ReferenceReconstruction;
use crate::ProverError;

fn commit_failed<F: JoltField>(reason: impl ToString) -> ProverError<F> {
    ProverError::Verifier(VerifierError::FinalOpeningVerificationFailed {
        reason: reason.to_string(),
    })
}

fn batch_failed<F: JoltField>(reason: impl ToString) -> ProverError<F> {
    ProverError::Verifier(VerifierError::FinalOpeningBatchFailed {
        reason: reason.to_string(),
    })
}

/// The committed packed FR limb object: its canonical plan, the commitment
/// the proof carries, and the retained opening hint stage 8 consumes.
pub struct FieldIncLimbOneHot<PCS: CommitmentScheme> {
    pub plan: FieldIncLimbPackingPlan,
    pub commitment: PCS::Output,
    pub hint: PCS::OpeningHint,
}

/// The per-cycle selected rows of every limb column, row-major over
/// `(cycle, column)` — the [`TraceOneHotRows`] source `commit_trace_one_hot`
/// consumes. Row zero means no committed entry (the digit-zero omission the
/// decode identity relies on), so a zero `FieldRdInc` cycle commits nothing.
struct FieldIncLimbRows {
    num_rows: usize,
    num_columns: usize,
    selected_rows: Vec<u8>,
}

impl TraceOneHotRows for FieldIncLimbRows {
    fn num_rows(&self) -> usize {
        self.num_rows
    }

    fn num_columns(&self) -> usize {
        self.num_columns
    }

    fn fill_row(&self, row: usize, selected_rows: &mut [u8]) {
        let start = row * self.num_columns;
        selected_rows.copy_from_slice(&self.selected_rows[start..start + self.num_columns]);
    }
}

/// The honest per-cycle limb-column rows off the FR oracle's `FieldRdInc`
/// values: the shared encoder's selected row per canonical column
/// ([`column_selected_row`] over [`canonical_limbs`]). `None` when
/// `FieldRdInc` is identically zero (nothing to commit — see
/// [`commit_field_inc_limbs`]).
fn assemble_field_inc_limb_rows<F: JoltField>(
    plan: &FieldIncLimbPackingPlan,
    log_t: usize,
    witness: &dyn JoltWitnessPlane<F>,
) -> Result<Option<Arc<FieldIncLimbRows>>, ProverError<F>> {
    let oracle =
        witness
            .field_inline()
            .ok_or(ProverError::Witness(WitnessError::UnavailableView {
                label: "packed field-inline limb commit oracle",
            }))?;
    let rd_inc: Vec<F> = oracle
        .oracle_table(FieldInlinePolynomialId::Committed(
            FieldInlineCommittedPolynomial::FieldRdInc,
        ))
        .map_err(ProverError::Witness)?;
    let num_rows = 1usize << log_t;
    if rd_inc.len() != num_rows {
        return Err(commit_failed(
            "the FR oracle's FieldRdInc table disagrees with the trace arity",
        ));
    }
    if rd_inc.iter().all(F::is_zero) {
        return Ok(None);
    }
    let chunking = BalancedIncChunking::new(plan.chunk_width()).map_err(commit_failed)?;
    let columns = plan.packing().ids();
    let num_columns = columns.len();
    let mut selected_rows = vec![0u8; num_rows * num_columns];
    for (value, row) in rd_inc
        .iter()
        .zip(selected_rows.chunks_exact_mut(num_columns))
    {
        let limbs = canonical_limbs(value);
        for (slot, column) in row.iter_mut().zip(columns) {
            let selected = column_selected_row(chunking, &limbs, *column).ok_or_else(|| {
                commit_failed(format!("{column:?} is not a field-inc limb column"))
            })?;
            // The plan's chunk width bounds every selected row below the
            // one-hot K, which is at most 256 on the packed axis.
            *slot = u8::try_from(selected).map_err(commit_failed)?;
        }
    }
    Ok(Some(Arc::new(FieldIncLimbRows {
        num_rows,
        num_columns,
        selected_rows,
    })))
}

/// Stage 0's packed FR commit: the limb object under the trace's own setup
/// (same packed arity by the norm-budget geometry) with the plan's layout
/// digest — the object [`jolt_verifier::stages::stage8::field_inline_packed`]
/// validates fail-closed.
///
/// `None` exactly when `FieldRdInc` is identically zero: the all-empty
/// one-hot object is unprovable under the catalogued Akita fold schedules,
/// and the verifier gates the object's presence on the stage-6b reduced
/// claim being nonzero — which the zero polynomial never produces (and a
/// nonzero one always does, up to the negligible Schwartz-Zippel miss).
pub fn commit_field_inc_limbs<F, PCS>(
    setup: &PCS::ProverSetup,
    one_hot_config: JoltOneHotConfig,
    log_t: usize,
    witness: &dyn JoltWitnessPlane<F>,
) -> Result<Option<FieldIncLimbOneHot<PCS>>, ProverError<F>>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F> + TraceOneHotCommitment,
{
    let plan = limb_plan::<F>(log_t, one_hot_config).map_err(ProverError::Verifier)?;
    let Some(rows) = assemble_field_inc_limb_rows::<F>(&plan, log_t, witness)? else {
        return Ok(None);
    };
    let (commitment, hint) = tracing::info_span!(
        "CommitmentScheme::commit_field_inc_limbs",
        packed_num_vars = plan.packing().packed_num_vars()
    )
    .in_scope(|| {
        PCS::commit_trace_one_hot(
            setup,
            plan.layout_digest(),
            plan.packing().slot_capacity(),
            rows,
        )
    })
    .map_err(commit_failed)?;
    Ok(Some(FieldIncLimbOneHot {
        plan,
        commitment,
        hint,
    }))
}

/// Stage 8's packed FR opening (after the auxiliary objects, mirroring the
/// verifier's order): the reconstruction member's per-column leaves reduce to
/// one physical claim on the transcript, opened natively from the stage-0
/// hint. Runs exactly when stage 0 committed the object (`FieldRdInc` not
/// identically zero — the verifier's presence gate).
pub fn open_field_inc_limbs<F, PCS, T>(
    setup: &PCS::ProverSetup,
    object: FieldIncLimbOneHot<PCS>,
    reconstruction: &ReconstructionClearOutput<F>,
    transcript: &mut T,
) -> Result<PCS::Proof, ProverError<F>>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
    T: Transcript<Challenge = F>,
{
    let values = reconstruction
        .output_values
        .field_inc_limbs
        .as_ref()
        .ok_or_else(|| batch_failed("the reconstruction phase produced no FR limb claims"))?;
    let points = reconstruction
        .output_points
        .field_inc_limbs
        .as_ref()
        .ok_or_else(|| batch_failed("the reconstruction phase produced no FR limb points"))?;
    let packed_claims = field_inc_limb_packed_claims(&object.plan, values, points)
        .map_err(ProverError::Verifier)?;
    let physical_claim = object
        .plan
        .packing()
        .reduce_claims(&packed_claims, transcript)
        .map_err(batch_failed)?;
    PCS::open_batch_from_hint(
        physical_claim.point.as_slice(),
        std::slice::from_ref(&physical_claim.value),
        setup,
        object.hint,
        transcript,
    )
    .map_err(batch_failed)
}

impl<F: JoltField> PrepareKernel<F, FieldIncLimbReconstructionInstance<F>>
    for ReferenceReconstruction
{
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, FieldIncLimbReconstructionInstance<F>>,
    ) -> Result<
        Box<dyn SumcheckKernel<F, Relation = FieldIncLimbReconstructionInstance<F>>>,
        KernelError<F>,
    > {
        let relation = inputs.relation;
        let shape = *relation.symbolic().shape();
        let rounds = relation.rounds();
        let r_reference = &inputs.challenges.r_reference;
        if r_reference.len() != rounds {
            return Err(KernelError::InvariantViolation {
                reason: "FR limb reference point arity disagrees with the cell domain",
            });
        }
        let r_cycle = inputs.points.rd_inc.as_slice();
        if r_cycle.len() != shape.log_t {
            return Err(KernelError::InvariantViolation {
                reason: "consumed FieldRdInc point arity disagrees with the trace domain",
            });
        }
        let chunking = BalancedIncChunking::new(shape.log_k_chunk).map_err(|_| {
            KernelError::InvariantViolation {
                reason: "FR limb chunk width is not a balanced-digit width",
            }
        })?;
        let column_ids =
            field_inc_limb_columns(&shape).map_err(|_| KernelError::InvariantViolation {
                reason: "FR limb shape admits no columns",
            })?;

        let oracle =
            witness
                .field_inline()
                .ok_or(KernelError::Witness(WitnessError::UnavailableView {
                    label: "FR limb reconstruction field-inline oracle",
                }))?;
        let rd_inc: Vec<F> = oracle.oracle_table(FieldInlinePolynomialId::Committed(
            FieldInlineCommittedPolynomial::FieldRdInc,
        ))?;
        let cycles = 1usize << shape.log_t;
        if rd_inc.len() != cycles {
            return Err(KernelError::TableSizeMismatch {
                table: "FieldRdInc".to_owned(),
                expected: cycles,
                got: rd_inc.len(),
            });
        }

        // The dense cell tables over the big-endian (digit-value ‖ cycle)
        // domain; LowToHigh binding reproduces the verifier's reversed-point
        // evaluations (the untrusted-advice kernel's convention).
        let cells = 1usize << rounds;
        let mut columns: Vec<Vec<F>> = vec![vec![F::zero(); cells]; column_ids.len()];
        for (cycle, value) in rd_inc.iter().enumerate() {
            let limbs = canonical_limbs(value);
            for (table, column) in columns.iter_mut().zip(&column_ids) {
                let selected = column_selected_row(chunking, &limbs, *column).ok_or(
                    KernelError::InvariantViolation {
                        reason: "canonical column list holds a non-limb polynomial",
                    },
                )?;
                if selected != 0 {
                    table[(selected << shape.log_t) | cycle] = F::one();
                }
            }
        }

        let eq_reference = eq_table(r_reference);
        let eq_cycle_base = eq_table(r_cycle);
        let digit_values: Vec<F> = (0..1usize << shape.log_k_chunk)
            .map(|digit| balanced_inc_value(&boolean_point_msb::<F>(shape.log_k_chunk, digit)))
            .collect();
        let cycle_mask = cycles - 1;
        let mut eq_cycle = vec![F::zero(); cells];
        let mut digit_value = vec![F::zero(); cells];
        for cell in 0..cells {
            eq_cycle[cell] = eq_cycle_base[cell & cycle_mask];
            digit_value[cell] = digit_values[cell >> shape.log_t];
        }

        let gamma = inputs.challenges.gamma;
        let mut booleanity_weights = Vec::with_capacity(column_ids.len());
        let mut decode_weights = Vec::with_capacity(column_ids.len());
        let mut gamma_power = F::one();
        for column in &column_ids {
            booleanity_weights.push(gamma_power);
            decode_weights.push(
                recomposition_coefficient::<F>(chunking, shape.limbs, *column).ok_or(
                    KernelError::InvariantViolation {
                        reason: "canonical column has no recomposition coefficient",
                    },
                )?,
            );
            gamma_power *= gamma;
        }
        // After the loop gamma_power is γ^C, the decode leg's shared weight.
        for weight in &mut decode_weights {
            *weight *= gamma_power;
        }

        Ok(Box::new(FieldIncLimbReconstructionKernel {
            relation: relation.clone(),
            columns: columns.into_iter().map(Polynomial::new).collect(),
            eq_reference: Polynomial::new(eq_reference),
            eq_cycle: Polynomial::new(eq_cycle),
            digit_value: Polynomial::new(digit_value),
            booleanity_weights,
            decode_weights,
            rounds_bound: 0,
        }))
    }
}

fn eq_table<F: JoltField>(point: &[F]) -> Vec<F> {
    EqPolynomial::new(point.to_vec()).evaluations()
}

/// The FR limb reconstruction member's hand kernel: per-column booleanity
/// legs against the reference point plus the balanced-digit decode leg,
///
/// `Σ_c γ^c · eq_ref · (Col_c² − Col_c) + γ^C · coeff_c · digit · eq_cyc · Col_c`
///
/// over the `(digit-value ‖ cycle)` cell domain (the FieldInline id family
/// cannot ride the jolt-keyed `NaiveSumcheckProver`, so the tables and the
/// expression are hand-held, like the FR claim-reduction kernels).
struct FieldIncLimbReconstructionKernel<F: JoltField> {
    relation: FieldIncLimbReconstructionInstance<F>,
    /// Dense cell tables in canonical column order.
    columns: Vec<Polynomial<F>>,
    eq_reference: Polynomial<F>,
    eq_cycle: Polynomial<F>,
    digit_value: Polynomial<F>,
    /// γ^c per column.
    booleanity_weights: Vec<F>,
    /// γ^C · recomposition coefficient per column.
    decode_weights: Vec<F>,
    rounds_bound: usize,
}

// Size arithmetic rather than a derive, like the sibling kernels.
#[cfg(feature = "allocative")]
impl<F: JoltField> allocative::Allocative for FieldIncLimbReconstructionKernel<F> {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let heap_bytes = |table: &Polynomial<F>| table.len() * size_of::<F>();
        let mut visitor = visitor.enter_self_sized::<Self>();
        visitor.visit_simple(
            allocative::Key::new("columns"),
            self.columns.iter().map(heap_bytes).sum::<usize>(),
        );
        for (key, table) in [
            (allocative::Key::new("eq_reference"), &self.eq_reference),
            (allocative::Key::new("eq_cycle"), &self.eq_cycle),
            (allocative::Key::new("digit_value"), &self.digit_value),
        ] {
            visitor.visit_simple(key, heap_bytes(table));
        }
        visitor.exit();
    }
}

impl<F: JoltField> FieldIncLimbReconstructionKernel<F> {
    fn remaining_rounds(&self) -> usize {
        self.relation.rounds() - self.rounds_bound
    }

    fn bind_tables(&mut self, challenge: F) {
        for table in self.columns.iter_mut().chain([
            &mut self.eq_reference,
            &mut self.eq_cycle,
            &mut self.digit_value,
        ]) {
            table.bind_with_order(challenge, BindingOrder::LowToHigh);
        }
        self.rounds_bound += 1;
    }

    fn require_fully_bound(&self) -> Result<(), SumcheckKernelError<F>> {
        match self.remaining_rounds() {
            0 => Ok(()),
            remaining => Err(SumcheckKernelError::NotFullyBound { remaining }),
        }
    }
}

impl<F: JoltField> ProveRounds<F> for FieldIncLimbReconstructionKernel<F> {
    fn num_rounds(&self) -> usize {
        self.relation.rounds()
    }

    fn prove_round(
        &mut self,
        bind: Option<F>,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if let Some(challenge) = bind {
            self.bind_tables(challenge);
        }
        let half = (1usize << self.remaining_rounds()) / 2;
        let degree = self.relation.degree();
        let order = BindingOrder::LowToHigh;
        let mut evals = Vec::with_capacity(degree + 1);
        for sample in 0..=degree {
            let point = F::from_u64(sample as u64);
            let sum = (0..half)
                .map(|y| {
                    let eq_reference = self
                        .eq_reference
                        .sumcheck_round_eval_with_order(y, point, order);
                    let decode = self
                        .digit_value
                        .sumcheck_round_eval_with_order(y, point, order)
                        * self
                            .eq_cycle
                            .sumcheck_round_eval_with_order(y, point, order);
                    self.columns
                        .iter()
                        .zip(&self.booleanity_weights)
                        .zip(&self.decode_weights)
                        .map(|((table, booleanity_weight), decode_weight)| {
                            let column = table.sumcheck_round_eval_with_order(y, point, order);
                            *booleanity_weight * eq_reference * (column * column - column)
                                + *decode_weight * decode * column
                        })
                        .sum::<F>()
                })
                .sum::<F>();
            evals.push(sum);
        }
        let round_sum = evals[0] + evals[1];
        if round_sum != previous_claim {
            return Err(SumcheckError::RoundCheckFailed {
                round,
                expected: previous_claim,
                actual: round_sum,
            });
        }
        Ok(UnivariatePoly::from_evals(&evals))
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind_tables(bind);
        Ok(())
    }
}

impl<F: JoltField> SumcheckKernel<F> for FieldIncLimbReconstructionKernel<F> {
    type Relation = FieldIncLimbReconstructionInstance<F>;

    fn output_claims(
        &mut self,
        _inputs: &SumcheckInputClaims<F, Self::Relation>,
    ) -> Result<SumcheckOutputClaims<F, Self::Relation>, SumcheckKernelError<F>> {
        self.require_fully_bound()?;
        Ok(FieldIncLimbReconstructionOutputClaims {
            columns: self.columns.iter().map(|table| table.evals()[0]).collect(),
        })
    }

    /// The public-table cross-checks: every bound public table's final value
    /// must equal the verifier's `derive_output_term` at the bound point (the
    /// same tie-down the FR claim-reduction kernels perform).
    fn validate_derived_tables(
        &self,
        relation: &Self::Relation,
        input_points: &SumcheckInputPoints<F, Self::Relation>,
        output_points: &SumcheckOutputPoints<F, Self::Relation>,
        challenges: &ConcreteSumcheckChallenges<F, Self::Relation>,
    ) -> Result<(), SumcheckKernelError<F>> {
        self.require_fully_bound()?;
        for (public, table) in [
            (
                FieldIncLimbReconstructionPublic::EqReference,
                &self.eq_reference,
            ),
            (FieldIncLimbReconstructionPublic::EqCycle, &self.eq_cycle),
            (
                FieldIncLimbReconstructionPublic::DigitValue,
                &self.digit_value,
            ),
        ] {
            let expected = relation.derive_output_term(
                &FieldInlineDerivedId::from(public),
                input_points,
                output_points,
                challenges,
            )?;
            let got = table.evals()[0];
            if got != expected {
                return Err(SumcheckKernelError::Verifier(
                    VerifierError::StageClaimSumcheckFailed {
                        stage: "FieldIncLimbReconstruction".to_string(),
                        reason: format!(
                            "{public:?} table bound to {got:?}, but derive_output_term gives \
                             {expected:?}"
                        ),
                    },
                ));
            }
        }
        Ok(())
    }
}
