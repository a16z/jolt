//! Optimized register read/write check (stage 4).
//!
//! Stores at most three sparse entries per cycle and combines reads as
//! `ra = γ·rs1_ra + γ²·rs2_ra`. Gruen factoring handles cycle rounds;
//! address rounds use three dense `K`-sized arrays.
//!
//! Cycle rounds 0 and 1 read the witness's row handle in task-partitioned
//! serial visits that regenerate their entries; no round holds `T/2` entries.
//! The second bind creates the quarter-cycle-domain field-coefficient entries
//! that later cycle binds compact in place. Parallel builds keep no seed
//! entries for a witness with a row handle: the second bind regenerates them
//! from the trace rows. The two read-operand openings come from one more row
//! walk.
//!
//! Supports cycle-first and full address-first binding.

use jolt_claims::protocols::jolt::geometry::dimensions::REGISTER_ADDRESS_BITS;
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

use super::read_write::ReadWriteOrder;
use super::support::{
    pin_derived_term, task_ranges, try_par_sum_vecs, GruenRoundMessage, KernelRows,
    RoundChallenges, VISIT_ROWS,
};
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

pub(crate) mod address;
mod address_first;
use address::{OperandEq, RegisterAddressState};
mod rows;
pub(crate) mod sparse;
#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test support module")]
pub(crate) mod test_support;
#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests;

pub(crate) use rows::RegisterCycleRow;

use address_first::AddressFirstKernel;
use rows::RegisterColumns;
use sparse::{CoeffLut, CycleState};

pub struct OptimizedRegistersReadWrite;

impl<F: JoltField> PrepareKernel<F, RegistersReadWriteChecking<F>> for OptimizedRegistersReadWrite {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, RegistersReadWriteChecking<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = RegistersReadWriteChecking<F>>>, KernelError<F>>
    {
        let dimensions = inputs.relation.register_dimensions();
        let order = ReadWriteOrder::new::<F>(dimensions)?;
        let log_t = dimensions.log_t();
        let log_k = dimensions.log_k();
        if log_t == 0 {
            return Err(KernelError::Unsupported {
                reason: "optimized registers read-write checking requires at least one cycle round",
            });
        }
        let r_cycle: &[F] = &inputs.points.rd_write_value;
        if r_cycle.len() != log_t {
            return Err(KernelError::InvariantViolation {
                reason: "registers read-write input point has the wrong variable count",
            });
        }
        if log_k != REGISTER_ADDRESS_BITS {
            return Err(KernelError::InvariantViolation {
                reason: "register read/write dimensions do not match the witness domain",
            });
        }
        if order == ReadWriteOrder::AddressFirst {
            return Ok(Box::new(AddressFirstKernel::prepare(witness, &inputs)?));
        }
        let rows = KernelRows::from_witness(witness, log_t, ROWS_LABEL)?;
        Ok(Box::new(ReadWriteKernel::new(witness, rows, &inputs)?))
    }
}

const ROWS_LABEL: &str = "register read/write trace rows";

#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
struct ReadWriteKernel<F: JoltField> {
    #[cfg_attr(feature = "allocative", allocative(skip))]
    dimensions: ReadWriteDimensions,
    cycle: CycleState<F>,
    gruen: GruenSplitEqPolynomial<F>,
    address: RegisterAddressState<F>,
    /// The witness plane owns these rows; it reports them itself.
    #[cfg_attr(feature = "allocative", allocative(skip))]
    rows: KernelRows<RegisterCycleRow>,
    challenges: RoundChallenges<F>,
}

impl<F: JoltField> ReadWriteKernel<F> {
    /// The cycle-first kernel: the second bind's seed cells and increments
    /// come from `witness`; rounds 0–1 and the operand walk read `rows`.
    fn new(
        witness: &dyn JoltWitnessPlane<F>,
        rows: KernelRows<RegisterCycleRow>,
        inputs: &ProverInputs<'_, F, RegistersReadWriteChecking<F>>,
    ) -> Result<Self, KernelError<F>> {
        let dimensions = inputs.relation.register_dimensions();
        let gamma = inputs.challenges.gamma;
        let gamma_sq = gamma * gamma;

        let (RegisterColumns { rd_inc }, seeds) =
            RegisterColumns::collect(witness, 1usize << dimensions.log_t())?;
        let cycle = CycleState::new(
            seeds,
            CoeffLut::new(vec![F::zero(), gamma, gamma_sq, gamma + gamma_sq]),
            CoeffLut::new(vec![F::zero(), F::one()]),
            rd_inc,
        );

        Ok(Self {
            dimensions,
            cycle,
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
        &self,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        let e_in = self.gruen.e_in_current();
        let e_out = self.gruen.e_out_current();
        let quadratic = self.cycle.quadratic(&self.rows, e_in, e_out)?;
        // `checked_cubic` takes an infallible endpoint; carry a row error out.
        let mut endpoint = Ok(());
        let message =
            self.gruen
                .checked_cubic(quadratic[0], quadratic[1], previous_claim, round, || {
                    self.cycle
                        .q_at_one(&self.rows, e_in, e_out)
                        .unwrap_or_else(|error| {
                            endpoint = Err(error);
                            F::zero()
                        })
                });
        endpoint?;
        message
    }

    fn bind(&mut self, r: F) -> Result<(), SumcheckError<F>> {
        let mut layout_transitioned = false;
        if self.challenges.bound() < self.dimensions.log_t() {
            self.gruen.bind(r);
            layout_transitioned = self.cycle.bind(r)?;
        } else {
            self.address.bind(r);
        }
        self.challenges.push(r);

        if self.challenges.bound() == self.dimensions.log_t() {
            // Replacing the state frees the entry allocation here rather
            // than at kernel drop.
            (
                self.address.ra,
                self.address.wa,
                self.address.val,
                self.address.inc_scalar,
            ) = self.cycle.take_dense(1usize << self.dimensions.log_k())?;
            self.address.eq_scalar = self.gruen.current_scalar();
        }

        if layout_transitioned {
            crate::mem::purge_retained_memory(self.dimensions.log_t());
        }
        Ok(())
    }

    /// `Σ_j [index_j hot] · eq(r_address, index_j) · eq(r_cycle, j)` for the
    /// two read operands in one walk over the rows — the direct MLE of a
    /// one-hot `(K × T)` grid at the bound point.
    ///
    /// Ports legacy `compute_rs2_ra_claim`: a 2-way split over the joint
    /// `(cycle ‖ address)` index keeps both eq tables at ~√(K·T). Big-endian
    /// joint point `[r_cycle ‖ r_address]`, joint index `(j << addr_bits) | k`.
    /// Tasks hold whole `e_hi` blocks, so each callback lies in one block.
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

impl<F: JoltField> ProveRounds<F> for ReadWriteKernel<F> {
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

impl<F: JoltField> SumcheckKernel<F> for ReadWriteKernel<F> {
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
