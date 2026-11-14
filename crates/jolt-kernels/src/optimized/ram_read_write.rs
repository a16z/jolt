//! The optimized RAM read/write-checking (stage 2) kernel: the legacy
//! `RamReadWriteCheckingProver` ported onto the kernel seam.
//!
//! Same summand, variable order, and binding order as the reference kernel —
//! `eq(τ_low,j)·ra(k,j)·[(1+γ)val(k,j)+(γ+γ²C(k))inc(j)+γ³+γ⁴N(k)]` over the
//! joint `(address ‖ cycle)` domain, bound low-to-high — but the `(K × T)`
//! `ra`/`val` grids are never materialized:
//!
//! - **Sparse read-write matrix**: one entry per RAM access;
//!   `prev_val`/`next_val` checkpoints recover every implicit coefficient
//!   (see `rw_matrix`).
//! - **Gruen split-eq + Dao–Thaler factoring** for the `log_T` cycle rounds:
//!   cycle-first keeps the eq factor in `O(√T)` tables, and each cubic message is
//!   reconstructed from the quadratic factor's `[q(0), q_∞]` plus the
//!   running claim ([`jolt_poly::GruenRound::gruen_poly_deg_3`]).
//! - **Hint-based quadratic address rounds** on the address-major matrix
//!   against the bound `val_init` column (`s(1)` recovered from the claim).
//!
//! `val_init` is reconstructed from the trace and the `RamValFinal` oracle
//! (the witness plane does not expose the initial RAM state): an accessed
//! address's initial value is its first access's pre-value, an untouched
//! address's final value IS its initial value. Honest-prover data path; with
//! hint-anchored round messages a divergent witness surfaces at the driver's
//! final-claim check rather than a per-round check.
//!
//! Supports cycle-first and full address-first binding. Address-first keeps
//! cycle equality weights only through the last RAM access, alongside the sparse matrix.
//! At the cycle handoff, RAM selectors have a zero suffix and values have
//! a constant suffix equal to the final checkpoint; both remain implicit.
//! Mixed cycle/address schedules are explicitly unsupported.

use crate::optimized::support::eq_round_error;
use jolt_claims::protocols::jolt::geometry::ram::ram_inc;
use jolt_claims::protocols::jolt::{
    JoltDerivedId, JoltPolynomialId, JoltVirtualPolynomial, RamReadWritePublic, ReadWriteDimensions,
};
use jolt_field::JoltField;
use jolt_poly::{BindingOrder, GruenSplitEqPolynomial, Polynomial, TensorEqTable, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::{
    ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckInputPoints, SumcheckOutputClaims,
    SumcheckOutputPoints,
};
use jolt_verifier::stages::stage2::ram_read_write_checking::{
    RamReadWriteChecking, RamReadWriteOutputClaims,
};
use jolt_witness::JoltWitnessPlane;

use super::prefix_product::PrefixColumn;
use super::ram_events::RamEventLog;
use super::ram_trace::{ram_access_columns_bytes, RamAccessColumns};
use super::read_write::{AddressMasks, ReadWriteBatch, ReadWriteOrder};
use super::rw_matrix::{
    round0_bind, round0_bind_bytes, round0_event_bind, round0_event_q_at_one,
    round0_event_quadratic, round0_q_at_one, round0_quadratic_coefficients, AddressMajorMatrix,
    CycleMajorMatrix, SparseCycleColumn,
};
use super::streaming::FillMismatch;
use super::support::{map_indices, pin_derived_term_if_derived, GruenRoundMessage};
use super::OptimizedBackend;
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
enum Phase<F: JoltField> {
    AddressFirst {
        matrix: AddressMajorMatrix<F>,
        eq: Vec<F>,
        inc: PrefixColumn<F>,
        gruen: GruenSplitEqPolynomial<F>,
    },
    DenseCycle {
        ra: PrefixColumn<F>,
        val: PrefixColumn<F>,
        inc: PrefixColumn<F>,
        gruen: GruenSplitEqPolynomial<F>,
    },
    Round0 {
        columns: RamAccessColumns,
        inc: PrefixColumn<F>,
        gruen: GruenSplitEqPolynomial<F>,
    },
    Cycle {
        matrix: CycleMajorMatrix<F>,
        inc: PrefixColumn<F>,
        gruen: GruenSplitEqPolynomial<F>,
    },
    /// The streaming kernel's round 0, read straight off the event log.
    EventRound0 {
        log: RamEventLog,
        gruen: GruenSplitEqPolynomial<F>,
    },
    /// The streaming kernel's later cycle rounds: `inc` over its nonzero row
    /// pairs, both bound in place.
    SparseCycle {
        matrix: CycleMajorMatrix<F>,
        inc: SparseCycleColumn<F>,
        gruen: GruenSplitEqPolynomial<F>,
    },
    /// `inc` is bound at every cycle variable: one entry.
    Address {
        matrix: AddressMajorMatrix<F>,
        merged_eq: Vec<F>,
        inc: PrefixColumn<F>,
    },
    Done {
        merged_eq: Vec<F>,
        #[cfg_attr(feature = "allocative", allocative(skip))]
        final_ra: F,
        #[cfg_attr(feature = "allocative", allocative(skip))]
        final_val: F,
        #[cfg_attr(feature = "allocative", allocative(skip))]
        final_inc: F,
    },
}

#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
pub(crate) struct RamReadWriteKernel<F: JoltField> {
    phase: Option<Phase<F>>,
    val_init: Polynomial<F>,
    #[cfg_attr(feature = "allocative", allocative(skip))]
    batch: ReadWriteBatch<F>,
    mask: AddressMasks<F>,
    log_t: usize,
    log_k: usize,
}

impl<F: JoltField> Phase<F> {
    fn error() -> SumcheckError<F> {
        SumcheckError::MissingEvaluationSource {
            kind: "RAM read-write phase state",
        }
    }
}

const LATE_PURGE_CYCLE_ROUNDS: usize = 6;

impl<F: JoltField> RamReadWriteKernel<F> {
    /// The kernel from its first phase, batching with the relation's γ powers
    /// and address masks.
    fn new(
        inputs: &ProverInputs<'_, F, RamReadWriteChecking<F>>,
        phase: Phase<F>,
        val_init: Polynomial<F>,
        log_t: usize,
        log_k: usize,
    ) -> Result<Self, KernelError<F>> {
        Ok(Self {
            phase: Some(phase),
            val_init,
            batch: ReadWriteBatch::new(inputs.challenges.gamma),
            mask: AddressMasks::new(
                inputs.relation.write_protected_word_intervals(),
                inputs.relation.unmapped_word_intervals(),
            )?,
            log_t,
            log_k,
        })
    }

    fn ingest(&mut self, r: F, round: usize) -> Result<(), SumcheckError<F>> {
        self.phase = Some(match self.phase.take().ok_or_else(Phase::error)? {
            Phase::Round0 {
                columns,
                inc,
                gruen,
            } => {
                let matrix = round0_bind(&columns, r);
                crate::mem::drop_in_background_thread(columns);
                self.finish_cycle_bind(matrix, inc, gruen, r, round)?
            }
            Phase::Cycle {
                mut matrix,
                inc,
                gruen,
            } => {
                matrix.bind(r);
                self.finish_cycle_bind(matrix, inc, gruen, r, round)?
            }
            Phase::EventRound0 { log, gruen } => {
                let (matrix, inc) = round0_event_bind(&log, r).map_err(sparse_bind_failed)?;
                drop(log);
                self.finish_sparse_bind(matrix, inc, gruen, r, round)?
            }
            Phase::SparseCycle {
                mut matrix,
                mut inc,
                gruen,
            } => {
                matrix.bind_in_place(r).map_err(sparse_bind_failed)?;
                inc.bind(r).map_err(sparse_bind_failed)?;
                self.finish_sparse_bind(matrix, inc, gruen, r, round)?
            }
            Phase::Address {
                mut matrix,
                merged_eq,
                inc,
            } => {
                matrix.bind(r, &mut self.val_init);
                self.mask.bind(r);
                if round + 1 == self.log_t + self.log_k {
                    self.finish_address(matrix, merged_eq, inc.value(0))
                } else {
                    Phase::Address {
                        matrix,
                        merged_eq,
                        inc,
                    }
                }
            }
            Phase::AddressFirst {
                mut matrix,
                eq,
                inc,
                gruen,
            } => {
                matrix.bind(r, &mut self.val_init);
                self.mask.bind(r);
                if round + 1 == self.log_k {
                    self.mask.finish_address();
                    crate::mem::drop_in_background_thread(eq);
                    let (ra, val) = matrix.into_cycle_tables(self.val_init.evals()[0]);
                    Phase::DenseCycle {
                        ra,
                        val,
                        inc,
                        gruen,
                    }
                } else {
                    Phase::AddressFirst {
                        matrix,
                        eq,
                        inc,
                        gruen,
                    }
                }
            }
            Phase::DenseCycle {
                mut ra,
                mut val,
                mut inc,
                mut gruen,
            } => {
                let gruen_round = gruen.round_mut().ok_or_else(|| eq_round_error(round))?;
                let _ = ra.bind(r);
                let _ = val.bind(r);
                let _ = inc.bind(r);
                gruen_round.bind(r);
                if round + 1 == self.log_t + self.log_k {
                    Phase::Done {
                        merged_eq: vec![gruen.current_scalar()],
                        final_ra: ra.value(0),
                        final_val: val.value(0),
                        final_inc: inc.value(0),
                    }
                } else {
                    Phase::DenseCycle {
                        ra,
                        val,
                        inc,
                        gruen,
                    }
                }
            }
            Phase::Done { .. } => return Err(Phase::error()),
        });
        Ok(())
    }

    fn finish_cycle_bind(
        &self,
        matrix: CycleMajorMatrix<F>,
        mut inc: PrefixColumn<F>,
        mut gruen: GruenSplitEqPolynomial<F>,
        r: F,
        round: usize,
    ) -> Result<Phase<F>, SumcheckError<F>> {
        gruen
            .round_mut()
            .ok_or_else(|| eq_round_error(round))?
            .bind(r);
        let _ = inc.bind(r);
        let phase = if round + 1 == self.log_t {
            self.finish_cycle_rounds(matrix, gruen, inc)
        } else {
            Phase::Cycle { matrix, inc, gruen }
        };
        self.purge_after_cycle_bind(round);
        Ok(phase)
    }

    fn finish_sparse_bind(
        &self,
        matrix: CycleMajorMatrix<F>,
        inc: SparseCycleColumn<F>,
        mut gruen: GruenSplitEqPolynomial<F>,
        r: F,
        round: usize,
    ) -> Result<Phase<F>, SumcheckError<F>> {
        gruen
            .round_mut()
            .ok_or_else(|| eq_round_error(round))?
            .bind(r);
        let phase = if round + 1 == self.log_t {
            let inc = PrefixColumn {
                values: vec![inc.bound_value()],
                tail: [F::zero(); 2],
            };
            self.finish_cycle_rounds(matrix, gruen, inc)
        } else {
            Phase::SparseCycle { matrix, inc, gruen }
        };
        self.purge_after_cycle_bind(round);
        Ok(phase)
    }

    /// The handoff after the last cycle bind: `inc` is bound to one entry.
    fn finish_cycle_rounds(
        &self,
        matrix: CycleMajorMatrix<F>,
        gruen: GruenSplitEqPolynomial<F>,
        inc: PrefixColumn<F>,
    ) -> Phase<F> {
        let matrix = matrix.into_address_major();
        let merged_eq = vec![gruen.current_scalar()];
        crate::mem::drop_in_background_thread(gruen);
        if self.log_k == 0 {
            self.finish_address(matrix, merged_eq, inc.value(0))
        } else {
            Phase::Address {
                matrix,
                merged_eq,
                inc,
            }
        }
    }

    fn purge_after_cycle_bind(&self, round: usize) {
        if round == 0 || round == LATE_PURGE_CYCLE_ROUNDS || round + 1 == self.log_t {
            crate::mem::purge_retained_memory(self.log_t);
        }
    }

    fn finish_address(
        &self,
        matrix: AddressMajorMatrix<F>,
        merged_eq: Vec<F>,
        final_inc: F,
    ) -> Phase<F> {
        let (final_ra, final_val) = matrix.final_values(&self.val_init);
        Phase::Done {
            merged_eq,
            final_ra,
            final_val,
            final_inc,
        }
    }

    fn cycle_round_message(
        &self,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        let (gruen, [q_0, q_infty]) = match &self.phase {
            Some(Phase::Round0 {
                columns,
                inc,
                gruen,
            }) => {
                let e_in = gruen.e_in_current();
                let e_out = gruen.e_out_current();
                let in_bits = e_in.len().trailing_zeros() as usize;
                let in_mask = e_in.len() - 1;
                (
                    gruen,
                    round0_quadratic_coefficients(
                        columns,
                        |pair| e_out[pair >> in_bits] * e_in[pair & in_mask],
                        inc,
                        self.batch,
                        &self.mask,
                    ),
                )
            }
            Some(Phase::Cycle { matrix, inc, gruen }) => {
                let e_in = gruen.e_in_current();
                let e_out = gruen.e_out_current();
                let in_bits = e_in.len().trailing_zeros() as usize;
                let in_mask = e_in.len() - 1;
                (
                    gruen,
                    matrix.quadratic_coefficients(
                        |pair| e_out[pair >> in_bits] * e_in[pair & in_mask],
                        inc,
                        self.batch,
                        &self.mask,
                    ),
                )
            }
            Some(Phase::EventRound0 { log, gruen }) => {
                let e_in = gruen.e_in_current();
                let e_out = gruen.e_out_current();
                let in_bits = e_in.len().trailing_zeros() as usize;
                let in_mask = e_in.len() - 1;
                (
                    gruen,
                    round0_event_quadratic(
                        log,
                        |pair| e_out[pair >> in_bits] * e_in[pair & in_mask],
                        self.batch,
                        &self.mask,
                    ),
                )
            }
            Some(Phase::SparseCycle { matrix, inc, gruen }) => {
                let e_in = gruen.e_in_current();
                let e_out = gruen.e_out_current();
                let in_bits = e_in.len().trailing_zeros() as usize;
                let in_mask = e_in.len() - 1;
                (
                    gruen,
                    matrix.sparse_quadratic_coefficients(
                        |pair| e_out[pair >> in_bits] * e_in[pair & in_mask],
                        inc,
                        self.batch,
                        &self.mask,
                    ),
                )
            }
            Some(Phase::DenseCycle {
                ra,
                val,
                inc,
                gruen,
            }) => {
                // Address binding reduces each mask to one scalar. C scales the
                // increment and N folds into count, which affects q(0) only.
                let coefficients = self.batch.bound(self.mask.value(0));
                let quadratic = gruen.par_fold_out_in_prefix(
                    ra.values.len().div_ceil(2),
                    || [F::zero(); 2],
                    |acc, row, _x_in, e_in| {
                        let pair = |table: &PrefixColumn<F>| {
                            let (lo, hi) = table.pair(row);
                            [lo, hi - lo]
                        };
                        let ra = pair(ra);
                        let val = pair(val);
                        let inc = pair(inc);
                        acc[0] += e_in * ra[0] * coefficients.value(val[0], inc[0]);
                        acc[1] += e_in * ra[1] * coefficients.slope(val[1], inc[1]);
                    },
                    |_x_out, e_out, acc| acc.map(|value| e_out * value),
                    |a, b| [a[0] + b[0], a[1] + b[1]],
                );
                (gruen, quadratic)
            }
            _ => return Err(Phase::error()),
        };
        gruen
            .round()
            .ok_or_else(|| eq_round_error(round))?
            .checked_cubic(q_0, q_infty, previous_claim, round, || {
                let e_in = gruen.e_in_current();
                let e_out = gruen.e_out_current();
                let in_bits = e_in.len().trailing_zeros() as usize;
                let in_mask = e_in.len() - 1;
                let weight = |pair| e_out[pair >> in_bits] * e_in[pair & in_mask];
                match &self.phase {
                    Some(Phase::Round0 { columns, inc, .. }) => {
                        round0_q_at_one(columns, weight, inc, self.batch, &self.mask)
                    }
                    Some(Phase::Cycle { matrix, inc, .. }) => {
                        matrix.q_at_one(weight, inc, self.batch, &self.mask)
                    }
                    Some(Phase::EventRound0 { log, .. }) => {
                        round0_event_q_at_one(log, weight, self.batch, &self.mask)
                    }
                    Some(Phase::SparseCycle { matrix, inc, .. }) => {
                        matrix.sparse_q_at_one(weight, inc, self.batch, &self.mask)
                    }
                    Some(Phase::DenseCycle { ra, val, inc, .. }) => {
                        let coefficients = self.batch.bound(self.mask.value(0));
                        (0..ra.values.len().div_ceil(2)).fold(F::zero(), |sum, pair| {
                            let row = 2 * pair + 1;
                            sum + weight(pair)
                                * ra.value(row)
                                * coefficients.value(val.value(row), inc.value(row))
                        })
                    }
                    _ => unreachable!("cycle phase was checked above"),
                }
            })
    }

    fn address_round_message(
        &self,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        let (matrix, eq, inc) = match &self.phase {
            Some(Phase::Address {
                matrix,
                merged_eq,
                inc,
            }) => (matrix, merged_eq, inc),
            Some(Phase::AddressFirst {
                matrix, eq, inc, ..
            }) => (matrix, eq, inc),
            _ => return Err(Phase::error()),
        };
        let evals = matrix.address_round_evals(&self.val_init, inc, eq, self.batch, &self.mask);
        Ok(UnivariatePoly::from_evals_and_hint(previous_claim, &evals))
    }
}

impl<F: JoltField> ProveRounds<F> for RamReadWriteKernel<F> {
    fn num_rounds(&self) -> usize {
        self.log_t + self.log_k
    }

    fn prove_round(
        &mut self,
        bind: Option<F>,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if let Some(challenge) = bind {
            self.ingest(challenge, round - 1)?;
        }
        match &self.phase {
            Some(
                Phase::Round0 { .. }
                | Phase::Cycle { .. }
                | Phase::EventRound0 { .. }
                | Phase::SparseCycle { .. }
                | Phase::DenseCycle { .. },
            ) => self.cycle_round_message(round, previous_claim),
            Some(Phase::Address { .. } | Phase::AddressFirst { .. }) => {
                self.address_round_message(previous_claim)
            }
            Some(Phase::Done { .. }) | None => Err(Phase::error()),
        }
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.ingest(bind, self.num_rounds() - 1)
    }
}

impl<F: JoltField> SumcheckKernel<F> for RamReadWriteKernel<F> {
    type Relation = RamReadWriteChecking<F>;

    fn output_claims(
        &mut self,
        _inputs: &SumcheckInputClaims<F, Self::Relation>,
    ) -> Result<SumcheckOutputClaims<F, Self::Relation>, SumcheckKernelError<F>> {
        let Some(Phase::Done {
            final_ra,
            final_val,
            final_inc,
            ..
        }) = &self.phase
        else {
            return Err(SumcheckKernelError::NotFullyBound {
                remaining: self.num_rounds(),
            });
        };
        Ok(RamReadWriteOutputClaims {
            val: *final_val,
            ra: *final_ra,
            inc: *final_inc,
        })
    }

    /// The hand-maintained cycle-eq factor must equal the verifier's
    /// `EqCycle` scalar at the bound point — the same cross-check the naive
    /// tier runs on its tiled eq table.
    fn validate_derived_tables(
        &self,
        relation: &Self::Relation,
        input_points: &SumcheckInputPoints<F, Self::Relation>,
        output_points: &SumcheckOutputPoints<F, Self::Relation>,
        challenges: &ConcreteSumcheckChallenges<F, Self::Relation>,
    ) -> Result<(), SumcheckKernelError<F>> {
        let Some(Phase::Done { merged_eq, .. }) = &self.phase else {
            return Err(SumcheckKernelError::NotFullyBound {
                remaining: self.num_rounds(),
            });
        };
        let id = JoltDerivedId::from(RamReadWritePublic::EqCycle);
        pin_derived_term_if_derived(
            relation,
            id,
            input_points,
            output_points,
            challenges,
            merged_eq[0],
        )?;
        pin_derived_term_if_derived(
            relation,
            JoltDerivedId::from(RamReadWritePublic::WriteProtectedEqCycle),
            input_points,
            output_points,
            challenges,
            merged_eq[0] * self.mask.value(0).protected(),
        )?;
        pin_derived_term_if_derived(
            relation,
            JoltDerivedId::from(RamReadWritePublic::UnmappedEqCycle),
            input_points,
            output_points,
            challenges,
            merged_eq[0] * self.mask.value(0).unmapped(),
        )
    }
}

fn sparse_bind_failed<F: JoltField>(_: FillMismatch) -> SumcheckError<F> {
    SumcheckError::MissingEvaluationSource {
        kind: "RAM read-write sparse bind (entry counts disagree)",
    }
}

/// The relation's checked `(log_T, log_K, τ_low, binding order)`.
fn geometry<F: JoltField>(
    relation: &RamReadWriteChecking<F>,
) -> Result<(usize, usize, &[F], ReadWriteOrder), KernelError<F>> {
    let dimensions = relation.dimensions();
    let log_t = dimensions.log_t();
    let log_k = relation.ram_log_k();
    let tau_low = relation.product_tau_low();
    let order = ReadWriteOrder::new::<F>(dimensions)?;
    if log_t == 0 || dimensions.log_k() != log_k || tau_low.len() != log_t {
        return Err(KernelError::InvariantViolation {
            reason: "RAM read-write checking geometry is inconsistent",
        });
    }
    if log_t > 32 || log_k > 32 {
        return Err(KernelError::Unsupported {
            reason: "optimized RAM read-write checking packs indices as u32 \
                     (log_T, log_K ≤ 32)",
        });
    }
    Ok((log_t, log_k, tau_low, order))
}

impl<F: JoltField> PrepareKernel<F, RamReadWriteChecking<F>> for OptimizedBackend {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, RamReadWriteChecking<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = RamReadWriteChecking<F>>>, KernelError<F>>
    {
        let (log_t, log_k, tau_low, order) = geometry(inputs.relation)?;

        let columns = RamAccessColumns::collect_full(session, witness, log_t)?;
        super::ram_trace::validate_addresses(&columns.addresses, 1usize << log_k)?;

        let inc_rows = witness.shape(ram_inc().polynomial_id())?.rows();
        let inc = PrefixColumn {
            values: map_indices(columns.value_prefix_len(), |cycle| columns.increment(cycle)),
            tail: [F::zero(); 2],
        };
        let val_final = witness.oracle_table(JoltPolynomialId::Virtual(
            JoltVirtualPolynomial::RamValFinal,
        ))?;
        if inc_rows != 1usize << log_t || val_final.len() != 1usize << log_k {
            return Err(KernelError::InvariantViolation {
                reason: "RAM read-write witness tables disagree with the relation geometry",
            });
        }
        let val_init = Polynomial::new(columns.reconstruct_val_init(val_final));

        let gruen = GruenSplitEqPolynomial::new(tau_low, BindingOrder::LowToHigh);
        let phase = match order {
            ReadWriteOrder::CycleFirst => Phase::Round0 {
                columns,
                inc,
                gruen,
            },
            ReadWriteOrder::AddressFirst => {
                let matrix = AddressMajorMatrix::from_columns(&columns);
                Phase::AddressFirst {
                    matrix,
                    eq: {
                        let eq = TensorEqTable::new(tau_low);
                        map_indices(columns.addresses.prefix_len(), |row| eq.evaluate_index(row))
                    },
                    inc,
                    gruen,
                }
            }
        };
        Ok(Box::new(RamReadWriteKernel::new(
            &inputs, phase, val_init, log_t, log_k,
        )?))
    }
}

/// Bytes the RAM read-write kernel and RAM RAF evaluation hold in the same
/// stage-2 batch, over the `prefix_cycles` cycles before the trace's constant
/// suffix.
#[derive(Clone, Copy, Debug)]
pub struct RamReadWriteBytes {
    /// From the read-write kernel's preparation until a member of the batch
    /// first binds.
    pub prepared: usize,
    /// Through the read-write kernel's first bind and RAF's first bind after
    /// it.
    pub first_bind: usize,
}

impl RamReadWriteBytes {
    /// The read-write kernel's tables for `dimensions` under `prepare`'s
    /// `ReadWriteOrder` dispatch, with RAF's folded `ra` and unmap tables, or
    /// `None` past `usize::MAX`. A phase split `prepare` rejects fails with
    /// the same error.
    ///
    /// Both orders count the increment prefix (at most `prefix_cycles`
    /// entries), `val_init` (at least four slots), the split-eq tables
    /// ([`GruenSplitEqPolynomial::low_to_high_bytes`]) and RAF's two `K`-entry
    /// tables. Cycle-first adds one generation of access columns
    /// ([`ram_access_columns_bytes`]), queued for a background drop only after
    /// round 0 returns; its first bind adds the round-0 collect at its largest
    /// entry count (`round0_bind_bytes`), then one half-length copy of a RAF
    /// table. The increment prefix's first bind stays in its allocation. Address-first adds the eq prefix (at
    /// most `prefix_cycles` entries) and the matrix at its collected capacity
    /// (`AddressMajorMatrix::collected_bytes`), which `prepare` builds beside
    /// one generation of access columns before it drops the value columns;
    /// its first bind keeps the shared address column and holds the larger of
    /// the bind's collect (`AddressMajorMatrix::first_bind_bytes`, the old
    /// entries included) and its bound entries
    /// (`AddressMajorMatrix::bound_bytes`) beside one half-length copy of a
    /// `K`-entry table: `val_init`'s, or one of RAF's. RAF binds its tables
    /// one at a time after the read-write kernel under the optimized backend's
    /// sequential member scheduler. The cycle-table handoff of a two-address
    /// first bind fits in these bytes.
    ///
    /// Not counted: later binds, and the preparation transients other than
    /// the stored tables: the access columns' collection (the eager check
    /// reserves it separately), `val_init`'s `K` reconstruction flags,
    /// address-first's eq-table tensor factors, and RAF's fold staging and eq
    /// factors.
    pub fn new<F: JoltField>(
        dimensions: ReadWriteDimensions,
        prefix_cycles: usize,
    ) -> Result<Option<Self>, KernelError<F>> {
        let order = ReadWriteOrder::new::<F>(dimensions)?;
        Ok(Self::for_order::<F>(order, dimensions, prefix_cycles))
    }

    fn for_order<F: JoltField>(
        order: ReadWriteOrder,
        dimensions: ReadWriteDimensions,
        prefix_cycles: usize,
    ) -> Option<Self> {
        let (log_t, log_k) = (dimensions.log_t(), dimensions.log_k());
        let addresses = 1usize.checked_shl(u32::try_from(log_k).ok()?)?;
        let field = |entries: usize| entries.checked_mul(size_of::<F>());
        let half_copy = field(addresses / 2)?;
        let shared = [
            field(prefix_cycles)?,
            field(addresses.max(4))?,
            GruenSplitEqPolynomial::<F>::low_to_high_bytes(log_t)?,
            field(addresses.checked_mul(2)?)?,
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)?;
        match order {
            ReadWriteOrder::CycleFirst => {
                let prepared = shared.checked_add(ram_access_columns_bytes(prefix_cycles, 1)?)?;
                Some(Self {
                    prepared,
                    first_bind: prepared
                        .checked_add(round0_bind_bytes::<F>(prefix_cycles, log_k)?)?
                        .checked_add(half_copy)?,
                })
            }
            ReadWriteOrder::AddressFirst => {
                let eq = shared.checked_add(field(prefix_cycles)?)?;
                let bind = AddressMajorMatrix::<F>::first_bind_bytes(prefix_cycles)?.max(
                    AddressMajorMatrix::<F>::bound_bytes(prefix_cycles)?.checked_add(half_copy)?,
                );
                Some(Self {
                    prepared: eq
                        .checked_add(ram_access_columns_bytes(prefix_cycles, 1)?)?
                        .checked_add(AddressMajorMatrix::<F>::collected_bytes(prefix_cycles)?)?,
                    first_bind: eq
                        .checked_add(ram_access_columns_bytes(prefix_cycles, 0)?)?
                        .checked_add(bind)?,
                })
            }
        }
    }
}

/// The streaming stage-2 read-write slot. Cycle-first, round 0 reads the
/// [`RamEventLog`] directly, and its bind writes the half-length matrix and
/// a sparse `inc` column over the nonzero row pairs, which later rounds bind
/// in place, so the cycle rounds hold state that scales with the event
/// counts instead of `T`. Address-first keeps the default kernel's prefix-backed
/// `inc` and eq tables and builds its matrix from the log.
pub struct StreamingRamReadWrite;

impl<F: JoltField> PrepareKernel<F, RamReadWriteChecking<F>> for StreamingRamReadWrite {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, RamReadWriteChecking<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = RamReadWriteChecking<F>>>, KernelError<F>>
    {
        let (log_t, log_k, tau_low, order) = geometry(inputs.relation)?;
        let geometry_mismatch = KernelError::InvariantViolation {
            reason: "RAM read-write witness tables disagree with the relation geometry",
        };
        let log = RamEventLog::shared(session, witness, log_t, 1usize << log_k)?;
        let val_final = witness.oracle_table(JoltPolynomialId::Virtual(
            JoltVirtualPolynomial::RamValFinal,
        ))?;
        if val_final.len() != 1usize << log_k {
            return Err(geometry_mismatch);
        }

        let gruen = GruenSplitEqPolynomial::new(tau_low, BindingOrder::LowToHigh);
        let val_init = Polynomial::new(log.val_init(val_final));
        let phase = match order {
            ReadWriteOrder::CycleFirst => Phase::EventRound0 { log, gruen },
            ReadWriteOrder::AddressFirst => {
                let prefix = log
                    .events()
                    .last()
                    .map_or(0, |event| event.cycle as usize + 1);
                let mut values = vec![F::zero(); prefix];
                for event in log.events() {
                    values[event.cycle as usize] = F::from_i128(event.inc());
                }
                let inc = PrefixColumn {
                    values,
                    tail: [F::zero(); 2],
                };
                Phase::AddressFirst {
                    matrix: AddressMajorMatrix::from_events(&log).map_err(|_| {
                        KernelError::InvariantViolation {
                            reason: "RAM event entries disagree between the count and fill passes",
                        }
                    })?,
                    eq: {
                        let eq = TensorEqTable::new(tau_low);
                        map_indices(prefix, |row| eq.evaluate_index(row))
                    },
                    inc,
                    gruen,
                }
            }
        };
        Ok(Box::new(RamReadWriteKernel::new(
            &inputs, phase, val_init, log_t, log_k,
        )?))
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod tests {
    use common::jolt_device::MemoryLayout;
    use jolt_claims::protocols::jolt::geometry::ram::{ram_ra, ram_val};
    use jolt_claims::{OutputClaims, SumcheckChallenges, SymbolicSumcheck};
    use jolt_field::{Fr, One, Ring};
    use jolt_poly::EqPolynomial;
    use jolt_program::preprocess::ram::{write_protected_word_intervals, RamWordMasks};
    use jolt_verifier::stages::relations::ConcreteSumcheck;
    use jolt_verifier::stages::stage2::ram_read_write_checking::{
        RamReadWriteChallenges, RamReadWriteInputClaims,
    };

    use super::super::testing::{
        assert_parity, random_scalars, with_ram_fixture, with_ram_fixture_init, FixtureShape, RamOp,
    };
    use super::*;
    use crate::optimized::parity::ExceptionalEq;
    use crate::ReferenceBackend;

    fn indicator(intervals: &[(usize, usize)], address: usize) -> Fr {
        Fr::from_u64(u64::from(
            intervals
                .iter()
                .any(|&(start, end)| start <= address && address < end),
        ))
    }

    /// The input claim: the relation's output expression summed over the
    /// Boolean `(address, cycle)` domain, each public weight at its Boolean
    /// value, `eq(τ_low, j)` times `C(k)` or `N(k)` for a masked weight.
    fn dense_input_claim(
        witness: &dyn JoltWitnessPlane<Fr>,
        relation: &RamReadWriteChecking<Fr>,
        challenges: &RamReadWriteChallenges<Fr>,
    ) -> Fr {
        let log_t = relation.product_tau_low().len();
        let eq = EqPolynomial::new(relation.product_tau_low().to_vec()).evaluations();
        let ra: Vec<Fr> = witness.oracle_table(ram_ra().polynomial_id()).unwrap();
        let val: Vec<Fr> = witness.oracle_table(ram_val().polynomial_id()).unwrap();
        let inc: Vec<Fr> = witness.oracle_table(ram_inc().polynomial_id()).unwrap();
        let expression = relation.symbolic().output_expression::<Fr>();
        let mut claim = Fr::from_u64(0);
        for k in 0..1usize << relation.ram_log_k() {
            for (j, &eq_j) in eq.iter().enumerate() {
                let index = (k << log_t) | j;
                let outputs = RamReadWriteOutputClaims {
                    val: val[index],
                    ra: ra[index],
                    inc: inc[j],
                };
                claim += expression.evaluate(
                    |id| outputs.resolve_output(id).unwrap(),
                    |id| challenges.resolve_challenge(id).unwrap(),
                    |id| {
                        let JoltDerivedId::RamReadWrite(public) = id else {
                            unreachable!("not a RAM read-write public: {id:?}")
                        };
                        eq_j * match public {
                            RamReadWritePublic::EqCycle => Fr::one(),
                            RamReadWritePublic::WriteProtectedEqCycle => {
                                indicator(relation.write_protected_word_intervals(), k)
                            }
                            RamReadWritePublic::UnmappedEqCycle => {
                                indicator(relation.unmapped_word_intervals(), k)
                            }
                        }
                    },
                );
            }
        }
        claim
    }

    fn run_parity(shape: FixtureShape, ops: Vec<RamOp>) {
        run_parity_init(
            shape,
            Vec::new(),
            ops,
            &[(shape.log_t, shape.log_k()), (0, shape.log_k())],
        );
    }

    fn run_parity_init(
        shape: FixtureShape,
        init_words: Vec<u64>,
        ops: Vec<RamOp>,
        phase_splits: &[(usize, usize)],
    ) {
        run_parity_case(shape, init_words, ops, phase_splits, None, None);
    }

    fn run_parity_case(
        shape: FixtureShape,
        init_words: Vec<u64>,
        ops: Vec<RamOp>,
        phase_splits: &[(usize, usize)],
        exceptional: Option<ExceptionalEq>,
        masks: Option<RamWordMasks>,
    ) {
        with_ram_fixture_init(shape, init_words, ops, |witness| {
            let first_cycle = if phase_splits[0].0 == 0 {
                phase_splits[0].1
            } else {
                0
            };
            let binds = random_scalars(shape.log_t + shape.log_k(), 71);
            let tau_low = exceptional.map_or_else(
                || random_scalars(shape.log_t, 17),
                |case| case.point(shape.log_t, binds[first_cycle]),
            );
            let gamma = random_scalars(1, 23)[0];
            let claims = RamReadWriteInputClaims {
                ram_read_value: Fr::from_u64(0),
                ram_write_value: Fr::from_u64(0),
                load: Fr::from_u64(0),
                store: Fr::from_u64(0),
            };
            let points = RamReadWriteInputClaims::<Vec<Fr>>::default();
            let challenges = RamReadWriteChallenges { gamma };
            let masks = masks.unwrap_or_else(|| {
                if shape.ram_k >= 8 {
                    RamWordMasks {
                        write_protected: vec![(3, 6), (7, shape.ram_k)],
                        unmapped: vec![(1, 3), (6, 7)],
                    }
                } else {
                    RamWordMasks {
                        write_protected: vec![(0, 1)],
                        unmapped: vec![(1, shape.ram_k)],
                    }
                }
            });
            for &(phase1, phase2) in phase_splits {
                let relation = RamReadWriteChecking::<Fr>::new(
                    ReadWriteDimensions::new(shape.log_t, shape.log_k(), phase1, phase2),
                    shape.log_k(),
                    tau_low.clone(),
                    masks.clone(),
                );
                let input_claim = dense_input_claim(witness, &relation, &challenges);

                let slots: [&dyn PrepareKernel<Fr, RamReadWriteChecking<Fr>>; 2] =
                    [&OptimizedBackend, &StreamingRamReadWrite];
                for slot in slots {
                    let inputs = || ProverInputs {
                        relation: &relation,
                        claims: &claims,
                        points: &points,
                        challenges: &challenges,
                    };
                    let reference = PrepareKernel::<Fr, _>::prepare(
                        &ReferenceBackend,
                        &mut ProofSession::default(),
                        witness,
                        inputs(),
                    )
                    .unwrap();
                    let optimized = slot
                        .prepare(&mut ProofSession::default(), witness, inputs())
                        .unwrap();
                    assert_parity(reference, optimized, input_claim, &inputs(), 71);
                }
            }
        });
    }

    #[test]
    fn matches_reference_on_mixed_traffic() {
        run_parity(
            FixtureShape {
                log_t: 4,
                ram_k: 16,
            },
            vec![
                RamOp::Write { word: 3, post: 5 },
                RamOp::Read { word: 3 },
                RamOp::Write { word: 3, post: 9 },
                RamOp::Read { word: 7 },
                RamOp::None,
                RamOp::Write { word: 4, post: 2 },
                RamOp::Read { word: 3 },
                RamOp::Write { word: 7, post: 6 },
                RamOp::Read { word: 4 },
                RamOp::Write { word: 12, post: 1 },
            ],
        );
    }

    #[test]
    fn protected_io_mask_matches_reference_in_both_binding_orders() {
        let lowest = super::super::testing::fixture_lowest_address();
        let layout = MemoryLayout {
            trusted_advice_start: lowest,
            trusted_advice_end: lowest + 16,
            untrusted_advice_start: lowest + 16,
            untrusted_advice_end: lowest + 32,
            input_start: lowest + 32,
            input_end: lowest + 40,
            stack_end: lowest + 88,
            heap_end: lowest + 256,
            ..Default::default()
        };
        let intervals = write_protected_word_intervals(&[(7, 9)], &layout, 32).unwrap();
        assert_eq!(intervals, [(0, 5), (7, 9), (11, 27)]);
        let masks = RamWordMasks {
            write_protected: intervals,
            unmapped: vec![(5, 7), (29, 32)],
        };
        let shape = FixtureShape {
            log_t: 4,
            ram_k: 32,
        };
        run_parity_case(
            shape,
            Vec::new(),
            vec![
                RamOp::Read { word: 0 },
                RamOp::Write { word: 2, post: 17 },
                RamOp::Write { word: 4, post: 23 },
                RamOp::Write { word: 7, post: 29 },
                RamOp::Write { word: 11, post: 31 },
                RamOp::Write { word: 26, post: 37 },
                RamOp::Write { word: 28, post: 41 },
                RamOp::Read { word: 6 },
                RamOp::Write { word: 30, post: 43 },
                RamOp::Write { word: 2, post: 0 },
            ],
            &[(shape.log_t, shape.log_k()), (0, shape.log_k())],
            None,
            Some(masks),
        );
    }

    #[test]
    fn matches_reference_on_sparse_traffic() {
        // Long no-access gaps and a single hot address: exercises the
        // implicit-entry checkpoint paths on both matrix orientations.
        run_parity(
            FixtureShape { log_t: 5, ram_k: 8 },
            vec![
                RamOp::None,
                RamOp::None,
                RamOp::Write { word: 5, post: 11 },
                RamOp::None,
                RamOp::None,
                RamOp::None,
                RamOp::Read { word: 5 },
                RamOp::None,
                RamOp::None,
                RamOp::None,
                RamOp::None,
                RamOp::Write { word: 5, post: 3 },
            ],
        );
    }

    /// Stores whose address remaps to no word move `RamInc` without an
    /// access, beside and inside row pairs that hold one.
    #[test]
    fn matches_reference_with_unmapped_increments() {
        run_parity(
            FixtureShape {
                log_t: 4,
                ram_k: 16,
            },
            vec![
                RamOp::UnmappedWrite { pre: 3, post: 8 },
                RamOp::Write { word: 3, post: 5 },
                RamOp::Read { word: 3 },
                RamOp::UnmappedWrite { pre: 9, post: 2 },
                RamOp::None,
                RamOp::UnmappedWrite { pre: 1, post: 4 },
                RamOp::Write { word: 6, post: 7 },
            ],
        );
    }

    #[test]
    fn matches_reference_without_ram_traffic() {
        run_parity(FixtureShape { log_t: 3, ram_k: 4 }, vec![RamOp::None; 3]);
    }

    #[test]
    fn matches_reference_at_single_round_phase_boundaries() {
        for (log_t, ram_k) in [(1, 2), (1, 4), (2, 2)] {
            run_parity(FixtureShape { log_t, ram_k }, vec![RamOp::None]);
        }
    }

    /// Nonzero `val_init` with reads BEFORE the first write: the optimized
    /// `val_init` reconstruction must recover a read-first word's initial
    /// value from its first access's pre-value, a never-accessed nonzero
    /// word's from the final state, and stay in parity with the reference
    /// val grid through both phases. A RAM-silent prefix exercises the initial
    /// checkpoint in the address-first handoff.
    #[test]
    fn matches_reference_on_read_before_write_with_nonzero_val_init() {
        let shape = FixtureShape {
            log_t: 4,
            ram_k: 16,
        };
        run_parity_init(
            shape,
            vec![7, 5, 11],
            vec![
                RamOp::None,
                RamOp::Read { word: 2 },
                RamOp::Write { word: 2, post: 9 },
                RamOp::Read { word: 2 },
                RamOp::Read { word: 4 },
                RamOp::None,
                RamOp::Write { word: 6, post: 3 },
                RamOp::Read { word: 6 },
            ],
            &[
                (shape.log_t, shape.log_k()),
                (0, shape.log_k()),
                (shape.log_t / 2, 0),
            ],
        );
    }

    #[test]
    fn rejects_invalid_phase_split() {
        let shape = FixtureShape { log_t: 3, ram_k: 4 };
        with_ram_fixture(shape, vec![RamOp::None; 3], |witness| {
            let tau_low = random_scalars(shape.log_t, 17);
            let relation = RamReadWriteChecking::<Fr>::new(
                ReadWriteDimensions::new(
                    shape.log_t,
                    shape.log_k(),
                    shape.log_t - 1,
                    shape.log_k() + 1,
                ),
                shape.log_k(),
                tau_low,
                RamWordMasks::default(),
            );
            let claims = RamReadWriteInputClaims {
                ram_read_value: Fr::from_u64(0),
                ram_write_value: Fr::from_u64(0),
                load: Fr::from_u64(0),
                store: Fr::from_u64(0),
            };
            let points = RamReadWriteInputClaims::<Vec<Fr>>::default();
            let challenges = RamReadWriteChallenges {
                gamma: random_scalars(1, 23)[0],
            };
            let result = PrepareKernel::<Fr, _>::prepare(
                &OptimizedBackend,
                &mut ProofSession::default(),
                witness,
                ProverInputs {
                    relation: &relation,
                    claims: &claims,
                    points: &points,
                    challenges: &challenges,
                },
            );
            assert!(matches!(
                result.map(|_| ()),
                Err(KernelError::InvariantViolation { .. })
            ));
        });
    }
    #[test]
    fn matches_reference_at_exceptional_cycle_points_in_both_orders() {
        let shape = FixtureShape {
            log_t: 4,
            ram_k: 16,
        };
        for phase in [(shape.log_t, shape.log_k()), (0, shape.log_k())] {
            for case in ExceptionalEq::ALL {
                run_parity_case(
                    shape,
                    vec![0, 7, 3, 11],
                    vec![
                        RamOp::Read { word: 1 },
                        RamOp::None,
                        RamOp::Write { word: 1, post: 29 },
                        RamOp::Read { word: 3 },
                        RamOp::Write { word: 2, post: 37 },
                        RamOp::Read { word: 1 },
                    ],
                    &[phase],
                    Some(case),
                    None,
                );
            }
        }
    }

    /// Address-first, the first bind holds one half-length copy of a
    /// `K`-entry table at a time beside its bound entries. Over 2^13 cycles,
    /// 2^20 addresses and 4096 proof rows, with 136-byte entries, that 16 MiB
    /// copy outweighs the bind's collect, and the term is 118,284,976 bytes:
    /// above the 105,571,064 bytes with the collect leading instead, below the
    /// 135,062,192 bytes with two copies at once.
    #[test]
    fn address_first_first_bind_holds_one_half_copy() {
        let bytes = RamReadWriteBytes::new::<Fr>(ReadWriteDimensions::new(13, 20, 0, 20), 4096)
            .unwrap()
            .unwrap()
            .first_bind;
        assert_eq!(bytes, 118_284_976);
    }
}
