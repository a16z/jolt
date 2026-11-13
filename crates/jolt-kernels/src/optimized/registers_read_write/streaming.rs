//! The register read/write check (stage 4) of the streaming register family.
//!
//! Cycle-first binding keeps no per-cycle state: cycle rounds read the
//! witness's resident rows in task-partitioned passes ([`StreamedCycleState`])
//! until a level fits [`DENSE_TABLE_CAP`], and the two read-operand openings
//! come from one more row walk. Address-first binding keeps the default
//! kernel. Either way no `rd` indices carry to stage 5, whose value evaluation
//! ([`StreamingRegistersValEvaluation`](crate::optimized::registers_val_evaluation::StreamingRegistersValEvaluation))
//! reads the rows.

use jolt_claims::protocols::jolt::{JoltDerivedId, ReadWriteDimensions, RegistersReadWritePublic};
use jolt_field::{Accumulator, JoltField};
use jolt_poly::{BindingOrder, GruenSplitEqPolynomial, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::{
    ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckInputPoints, SumcheckOutputPoints,
};
use jolt_verifier::stages::stage4::registers_read_write_checking::{
    RegistersReadWriteChecking, RegistersReadWriteOutputClaims,
};
use jolt_witness::{JoltWitnessPlane, WitnessError};

use super::address::{OperandEq, RegisterAddressState};
use super::sparse::StreamedCycleState;
use super::{read_write_order, OptimizedRegistersReadWrite, RegisterCycleRow};
use crate::optimized::read_write::ReadWriteOrder;
use crate::optimized::streaming::{task_ranges, DENSE_TABLE_CAP, VISIT_ROWS};
use crate::optimized::support::{
    eq_round_error, pin_derived_term, try_par_sum_vecs, BundleStore, GruenRoundMessage,
    RoundChallenges,
};
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

// The operand walk weights each visit callback by one `e_hi` entry, so a
// callback of at most one block of aligned rows must lie inside that block.
const _: () = assert!(VISIT_ROWS.is_power_of_two());

pub struct StreamingRegistersReadWrite;

impl<F: JoltField> PrepareKernel<F, RegistersReadWriteChecking<F>> for StreamingRegistersReadWrite {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, RegistersReadWriteChecking<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = RegistersReadWriteChecking<F>>>, KernelError<F>>
    {
        let log_t = inputs.relation.register_dimensions().log_t();
        // Address-first binding keeps the default kernel, and so does a
        // two-cycle domain: its only cycle round is the counting pass, so no
        // round could fill a level.
        if read_write_order(&inputs)? == ReadWriteOrder::AddressFirst || log_t == 1 {
            return OptimizedRegistersReadWrite::prepare_kernel(None, witness, inputs);
        }
        let rows = BundleStore::resident(witness, log_t, "register read/write trace rows")?;
        Ok(Box::new(StreamingReadWriteKernel::new(
            rows,
            &inputs,
            DENSE_TABLE_CAP,
        )?))
    }
}

#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
struct StreamingReadWriteKernel<F: JoltField> {
    #[cfg_attr(feature = "allocative", allocative(skip))]
    dimensions: ReadWriteDimensions,
    cycle: StreamedCycleState<F>,
    gruen: GruenSplitEqPolynomial<F>,
    address: RegisterAddressState<F>,
    rows: BundleStore<RegisterCycleRow>,
    challenges: RoundChallenges<F>,
}

impl<F: JoltField> StreamingReadWriteKernel<F> {
    /// The cycle-first kernel over `rows`, whose inputs passed
    /// [`read_write_order`]; `cap` bounds the bytes of the filled level's
    /// entries and increments, production passing [`DENSE_TABLE_CAP`].
    fn new(
        rows: BundleStore<RegisterCycleRow>,
        inputs: &ProverInputs<'_, F, RegistersReadWriteChecking<F>>,
        cap: usize,
    ) -> Result<Self, KernelError<F>> {
        let dimensions = inputs.relation.register_dimensions();
        if dimensions.log_t() > u32::BITS as usize {
            return Err(KernelError::InvariantViolation {
                reason: "cycle count exceeds the seed entries' packed u32 row domain",
            });
        }
        Ok(Self {
            dimensions,
            cycle: StreamedCycleState::new(inputs.challenges.gamma, cap),
            gruen: GruenSplitEqPolynomial::new(
                &inputs.points.rd_write_value,
                BindingOrder::LowToHigh,
            ),
            address: RegisterAddressState::default(),
            rows,
            challenges: RoundChallenges::new(dimensions.log_t() + dimensions.log_k()),
        })
    }

    fn cycle_round_message(
        &mut self,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        let e_in = self.gruen.e_in_current();
        let e_out = self.gruen.e_out_current();
        let r = self.challenges.as_slice();
        let quadratic = self.cycle.quadratic(&self.rows, e_in, e_out, r)?;
        // A failed `q(1)` pass is held and returned, so its placeholder zero
        // never reaches a round message.
        let mut failure = None;
        let message = self
            .gruen
            .round()
            .ok_or_else(|| eq_round_error(round))?
            .checked_cubic(quadratic[0], quadratic[1], previous_claim, round, || {
                self.cycle
                    .q_at_one(&self.rows, e_in, e_out, r)
                    .unwrap_or_else(|error| {
                        failure = Some(error);
                        F::zero()
                    })
            });
        match failure {
            Some(error) => Err(error),
            None => message,
        }
    }

    fn bind(&mut self, r: F) -> Result<(), SumcheckError<F>> {
        let round = self.challenges.bound();
        let log_t = self.dimensions.log_t();
        if round < log_t {
            self.gruen
                .round_mut()
                .ok_or_else(|| eq_round_error(round))?
                .bind(r);
            self.cycle.bind(r);
        } else {
            self.address.bind(r);
        }
        self.challenges.push(r);

        if self.challenges.bound() == log_t {
            (
                self.address.ra,
                self.address.wa,
                self.address.val,
                self.address.inc_scalar,
            ) = self.cycle.take_dense(1usize << self.dimensions.log_k())?;
            self.address.eq_scalar = self.gruen.current_scalar();
        }
        Ok(())
    }

    /// `Σ_j [index_j hot] · eq(r_address, index_j) · eq(r_cycle, j)` for the
    /// two read operands in one walk over the rows, as the default kernel
    /// computes over its index columns ([`OperandEq`]). Tasks hold whole
    /// `e_hi` blocks, so each callback lies in one block.
    fn one_hot_operand_claims(
        &self,
        r_address: &[F],
        r_cycle: &[F],
    ) -> Result<[F; 2], WitnessError> {
        let eq = OperandEq::new(r_address, r_cycle);
        let block = 1usize << eq.cycle_bits_in_lo;
        let ranges = task_ranges(1 << r_cycle.len(), block);
        let claims = try_par_sum_vecs(ranges.len(), 2, |task| -> Result<_, WitnessError> {
            let mut claims = vec![F::zero(); 2];
            self.rows.visit(
                ranges[task].clone(),
                VISIT_ROWS.min(block),
                |start, rows| {
                    let mut sums = [F::Accumulator::default(); 2];
                    for (j, row) in (start..).zip(rows) {
                        if let Some((rs1, _)) = row.rs1 {
                            sums[0].add(eq.lo[eq.low_index(j, rs1)]);
                        }
                        if let Some((rs2, _)) = row.rs2 {
                            sums[1].add(eq.lo[eq.low_index(j, rs2)]);
                        }
                    }
                    let e_hi = eq.hi[start >> eq.cycle_bits_in_lo];
                    claims[0] += e_hi * sums[0].reduce();
                    claims[1] += e_hi * sums[1].reduce();
                    Ok(())
                },
            )?;
            Ok(claims)
        })?;
        Ok([claims[0], claims[1]])
    }
}

impl<F: JoltField> ProveRounds<F> for StreamingReadWriteKernel<F> {
    fn num_rounds(&self) -> usize {
        self.dimensions.read_write_rounds()
    }

    fn prove_round(
        &mut self,
        bind: Option<F>,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if let Some(challenge) = bind {
            self.bind(challenge)?;
        }
        if self.challenges.bound() < self.dimensions.log_t() {
            self.cycle_round_message(round, previous_claim)
        } else {
            self.address.round_message(round, previous_claim)
        }
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind)
    }
}

impl<F: JoltField> SumcheckKernel<F> for StreamingReadWriteKernel<F> {
    type Relation = RegistersReadWriteChecking<F>;

    fn output_claims(
        &mut self,
        _inputs: &SumcheckInputClaims<F, Self::Relation>,
    ) -> Result<RegistersReadWriteOutputClaims<F>, SumcheckKernelError<F>> {
        self.challenges.require_complete()?;
        let point = self
            .dimensions
            .read_write_opening_point(self.challenges.as_slice())
            .map_err(|_| SumcheckKernelError::InvariantViolation {
                reason: "invalid register read/write opening point",
            })?;
        let [rs1_ra, rs2_ra] = self
            .one_hot_operand_claims(&point.r_address, &point.r_cycle)
            .map_err(|_| SumcheckKernelError::InvariantViolation {
                reason: "register read/write operand walk re-extraction failed after the rounds",
            })?;
        Ok(RegistersReadWriteOutputClaims {
            registers_val: self.address.val[0],
            rs1_ra,
            rs2_ra,
            rd_wa: self.address.wa[0],
            rd_inc: self.address.inc_scalar,
        })
    }

    /// Pin the internally tracked eq factor to the verifier's scalar path:
    /// the fully bound Gruen scalar must equal `derive_output_term(EqCycle)`.
    fn validate_derived_tables(
        &self,
        relation: &Self::Relation,
        input_points: &SumcheckInputPoints<F, Self::Relation>,
        output_points: &SumcheckOutputPoints<F, Self::Relation>,
        challenges: &ConcreteSumcheckChallenges<F, Self::Relation>,
    ) -> Result<(), SumcheckKernelError<F>> {
        self.challenges.require_complete()?;
        pin_derived_term(
            relation,
            JoltDerivedId::from(RegistersReadWritePublic::EqCycle),
            input_points,
            output_points,
            challenges,
            self.address.eq_scalar,
        )
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests;
