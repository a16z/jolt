//! Optimized register read/write check (stage 4).
//!
//! Stores at most three sparse entries per cycle and combines reads as
//! `ra = γ·rs1_ra + γ²·rs2_ra`. Gruen factoring handles cycle rounds;
//! address rounds use three dense `K`-sized arrays.
//!
//! `SeedEntry` omits the round-0 field value. The first challenge is held
//! without materializing `T/2`; the second bind creates the `T/4` indexed SoA
//! layout. Coefficients stay as LUT indices until the `u16` domain saturates.
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
use jolt_witness::JoltWitnessPlane;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::read_write::ReadWriteOrder;
use super::support::{pin_derived_term, GruenRoundMessage, RoundChallenges};
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

pub(crate) use rows::{RegisterCycleRow, SharedRdIndices};

use address_first::AddressFirstKernel;
use rows::CollectRegisterEntries;
use sparse::{CoeffLut, CycleState};

pub struct OptimizedRegistersReadWrite;

impl<F: JoltField> PrepareKernel<F, RegistersReadWriteChecking<F>> for OptimizedRegistersReadWrite {
    fn prepare(
        &self,
        session: &mut ProofSession,
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
            return Ok(Box::new(AddressFirstKernel::prepare(
                session, witness, &inputs,
            )?));
        }
        let cycles = 1usize << log_t;

        let gamma = inputs.challenges.gamma;
        let gamma_sq = gamma * gamma;

        // Sparse entry construction: one trace pass — the typed rows are
        // never materialized whole (80 bytes per cycle saved at the stage's
        // peak moment).
        let CollectRegisterEntries {
            entries,
            rs1_indices,
            rs2_indices,
            rd_indices,
            rd_inc,
        } = CollectRegisterEntries::collect(witness, cycles)?;
        let cycle = CycleState::new(
            entries,
            CoeffLut::new(vec![F::zero(), gamma, gamma_sq, gamma + gamma_sq]),
            CoeffLut::new(vec![F::zero(), F::one()]),
            rd_inc,
        );

        // Park the rd hot indices for the stage-5 val-evaluation kernel.
        session.park(SharedRdIndices(rd_indices));

        Ok(Box::new(ReadWriteKernel {
            dimensions,
            cycle,
            gruen: GruenSplitEqPolynomial::new(r_cycle, BindingOrder::LowToHigh),
            address: RegisterAddressState::default(),
            rs1_indices,
            rs2_indices,
            challenges: RoundChallenges::new(log_t + log_k),
        }))
    }
}

#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
struct ReadWriteKernel<F: JoltField> {
    #[cfg_attr(feature = "allocative", allocative(skip))]
    dimensions: ReadWriteDimensions,
    /// Sparse cycle-major entries, sorted by `(row, col)`; drained at the
    /// cycle→address transition.
    cycle: CycleState<F>,
    gruen: GruenSplitEqPolynomial<F>,
    address: RegisterAddressState<F>,
    rs1_indices: Vec<Option<u8>>,
    rs2_indices: Vec<Option<u8>>,
    challenges: RoundChallenges<F>,
}

impl<F: JoltField> ReadWriteKernel<F> {
    /// Cycle-round message via Gruen factoring: the quadratic inner factor's
    /// `[q(0), leading coefficient]` over the remaining cycle domain, wrapped
    /// into the exact cubic by `gruen_poly_deg_3`.
    fn cycle_round_message(
        &self,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        let e_in = self.gruen.e_in_current();
        let e_out = self.gruen.e_out_current();
        let quadratic = self.cycle.quadratic(e_in, e_out);
        self.gruen
            .checked_cubic(quadratic[0], quadratic[1], previous_claim, round, || {
                self.cycle
                    .q_at_one(self.gruen.e_in_current(), self.gruen.e_out_current())
            })
    }

    /// Bind the pending challenge: cycle rounds bind eq/inc and merge the
    /// sparse rows; the final cycle bind collapses to the K-sized dense
    /// address state; address rounds bind the three dense arrays.
    fn bind(&mut self, r: F) {
        let mut layout_transitioned = false;
        if self.challenges.bound() < self.dimensions.log_t() {
            self.gruen.bind(r);
            layout_transitioned = self.cycle.bind(r);
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
            ) = self.cycle.take_dense(1usize << self.dimensions.log_k());
            self.address.eq_scalar = self.gruen.current_scalar();
        }

        // Return replaced entry generations immediately.
        if layout_transitioned {
            crate::mem::purge_retained_memory(self.dimensions.log_t());
        }
    }

    /// `Σ_j [index_j hot] · eq(r_address, index_j) · eq(r_cycle, j)` for the
    /// two read operands in one walk — the direct MLE of a one-hot `(K × T)`
    /// grid at the bound point.
    ///
    /// Ports legacy `compute_rs2_ra_claim`: a 2-way split over the joint
    /// `(cycle ‖ address)` index keeps both eq tables at ~√(K·T). Big-endian
    /// joint point `[r_cycle ‖ r_address]`, joint index `(j << addr_bits) | k`.
    fn one_hot_operand_claims(&self, r_address: &[F], r_cycle: &[F]) -> (F, F) {
        let rs1_indices = &self.rs1_indices;
        let rs2_indices = &self.rs2_indices;
        let eq = OperandEq::new(r_address, r_cycle);
        let cycle_bits_in_lo = eq.cycle_bits_in_lo;
        let cycles_per_block = 1usize << cycle_bits_in_lo;
        let e_hi = &eq.hi;
        let e_lo = &eq.lo;

        let block_contribution = |idx_hi: usize| -> [F; 2] {
            let block_start = idx_hi << cycle_bits_in_lo;
            let block_end = core::cmp::min(block_start + cycles_per_block, rs1_indices.len());
            if block_start >= rs1_indices.len() {
                return [F::zero(); 2];
            }
            let mut sums = [F::Accumulator::default(), F::Accumulator::default()];
            for j in block_start..block_end {
                if let Some(rs1) = rs1_indices[j] {
                    sums[0].add(e_lo[eq.low_index(j, rs1)]);
                }
                if let Some(rs2) = rs2_indices[j] {
                    sums[1].add(e_lo[eq.low_index(j, rs2)]);
                }
            }
            let e_hi_eval = e_hi[idx_hi];
            [e_hi_eval * sums[0].reduce(), e_hi_eval * sums[1].reduce()]
        };

        #[cfg(feature = "parallel")]
        let claims = (0..e_hi.len())
            .into_par_iter()
            .map(block_contribution)
            .reduce(|| [F::zero(); 2], |a, b| [a[0] + b[0], a[1] + b[1]]);
        #[cfg(not(feature = "parallel"))]
        let claims = (0..e_hi.len())
            .map(block_contribution)
            .fold([F::zero(); 2], |a, b| [a[0] + b[0], a[1] + b[1]]);

        (claims[0], claims[1])
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
            self.bind(challenge);
        }
        if self.challenges.bound() < self.dimensions.log_t() {
            self.cycle_round_message(round, previous_claim)
        } else {
            self.address.round_message(round, previous_claim)
        }
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind);
        Ok(())
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
        let (rs1_ra, rs2_ra) = self.one_hot_operand_claims(&point.r_address, &point.r_cycle);
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
