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
//! Supports cycle-first and full address-first binding. The streaming
//! register family ([`StreamingRegistersReadWrite`]) runs the cycle-first
//! kernel over the witness's rows instead of collected entries
//! (`CycleSource`).

use crate::optimized::support::eq_round_error;
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
mod streaming;
#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test support module")]
pub(crate) mod test_support;
#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests;

pub(crate) use rows::{RegisterCycleRow, SharedRdIndices};
pub use streaming::StreamingRegistersReadWrite;

use address_first::AddressFirstKernel;
use rows::{CollectRegisterEntries, StreamedRows};
use sparse::{CycleState, StreamedCycleState};

pub struct OptimizedRegistersReadWrite;

impl OptimizedRegistersReadWrite {
    /// Peak bytes of the kernel `prepare` builds for `dimensions` over the
    /// `prefix_cycles` cycles before the trace's constant suffix, or `None`
    /// past `usize::MAX`, under the same `ReadWriteOrder` dispatch, with the
    /// per-cycle `rd` indices it parks for stage 5: the cycle-first sparse
    /// state (`CycleState::peak_bytes`) beside the `rs1` and `rs2` index
    /// columns, or the address-first tables (`AddressFirstKernel::peak_bytes`);
    /// none for a phase split `prepare` rejects before allocating. The parked
    /// indices' truncated copy is less than the increments' copy in the
    /// cycle-first preparation and than the address-first handoff. Not
    /// counted: the Gruen split-eq tables, the operand eq tables the output
    /// claims build, the round challenges, and the collects without
    /// random-access rows, which keep every cycle, or without `parallel`, with
    /// their per-chunk extraction buffers.
    pub fn peak_bytes<F: JoltField>(
        dimensions: ReadWriteDimensions,
        prefix_cycles: usize,
    ) -> Option<usize> {
        let column = prefix_cycles.checked_mul(size_of::<Option<u8>>())?;
        match ReadWriteOrder::new::<F>(dimensions) {
            Ok(ReadWriteOrder::CycleFirst) => column
                .checked_mul(3)?
                .checked_add(CycleState::<F>::peak_bytes(prefix_cycles)?),
            Ok(ReadWriteOrder::AddressFirst) => {
                column.checked_add(AddressFirstKernel::<F>::peak_bytes(prefix_cycles)?)
            }
            Err(_) => Some(0),
        }
    }
}

impl<F: JoltField> PrepareKernel<F, RegistersReadWriteChecking<F>> for OptimizedRegistersReadWrite {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, RegistersReadWriteChecking<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = RegistersReadWriteChecking<F>>>, KernelError<F>>
    {
        Self::prepare_kernel(Some(session), witness, inputs)
    }
}

impl OptimizedRegistersReadWrite {
    /// The default kernel for `inputs`. `rd_carry` is the session that
    /// receives the per-cycle `rd` indices the default stage-5 kernel
    /// reclaims ([`SharedRdIndices`]); `None` when stage 5 reads the rows
    /// itself.
    fn prepare_kernel<F: JoltField>(
        rd_carry: Option<&mut ProofSession>,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, RegistersReadWriteChecking<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = RegistersReadWriteChecking<F>>>, KernelError<F>>
    {
        if read_write_order(&inputs)? == ReadWriteOrder::AddressFirst {
            return Ok(Box::new(AddressFirstKernel::prepare(
                rd_carry, witness, &inputs,
            )?));
        }
        let cycles = 1usize << inputs.relation.register_dimensions().log_t();

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
        let state = CycleState::new(entries, inputs.challenges.gamma, rd_inc);

        if let Some(session) = rd_carry {
            session.park(SharedRdIndices::new(rd_indices, cycles));
        }

        Ok(Box::new(ReadWriteKernel::new(
            CycleSource::Collected {
                state,
                rs1_indices,
                rs2_indices,
            },
            &inputs,
        )))
    }
}

/// The binding order `inputs` select, once the checks every register
/// read/write kernel shares pass.
fn read_write_order<F: JoltField>(
    inputs: &ProverInputs<'_, F, RegistersReadWriteChecking<F>>,
) -> Result<ReadWriteOrder, KernelError<F>> {
    let dimensions = inputs.relation.register_dimensions();
    let order = ReadWriteOrder::new::<F>(dimensions)?;
    if dimensions.log_t() == 0 {
        return Err(KernelError::Unsupported {
            reason: "optimized registers read-write checking requires at least one cycle round",
        });
    }
    if inputs.points.rd_write_value.len() != dimensions.log_t() {
        return Err(KernelError::InvariantViolation {
            reason: "registers read-write input point has the wrong variable count",
        });
    }
    if dimensions.log_k() != REGISTER_ADDRESS_BITS {
        return Err(KernelError::InvariantViolation {
            reason: "register read/write dimensions do not match the witness domain",
        });
    }
    Ok(order)
}

/// The cycle rounds' state beside what the operand openings read.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
enum CycleSource<F: JoltField> {
    /// The trace's sparse entries, collected with its `rs1` and `rs2` index
    /// columns.
    Collected {
        state: CycleState<F>,
        rs1_indices: Vec<Option<u8>>,
        rs2_indices: Vec<Option<u8>>,
    },
    /// Passes over the witness's resident rows until a level is filled.
    Streamed {
        state: StreamedCycleState<F>,
        rows: StreamedRows,
    },
}

#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
struct ReadWriteKernel<F: JoltField> {
    #[cfg_attr(feature = "allocative", allocative(skip))]
    dimensions: ReadWriteDimensions,
    cycle: CycleSource<F>,
    gruen: GruenSplitEqPolynomial<F>,
    address: RegisterAddressState<F>,
    challenges: RoundChallenges<F>,
}

impl<F: JoltField> ReadWriteKernel<F> {
    /// The cycle-first kernel over `cycle`, for inputs that passed
    /// [`read_write_order`].
    fn new(
        cycle: CycleSource<F>,
        inputs: &ProverInputs<'_, F, RegistersReadWriteChecking<F>>,
    ) -> Self {
        let dimensions = inputs.relation.register_dimensions();
        Self {
            dimensions,
            cycle,
            gruen: GruenSplitEqPolynomial::new(
                &inputs.points.rd_write_value,
                BindingOrder::LowToHigh,
            ),
            address: RegisterAddressState::default(),
            challenges: RoundChallenges::new(dimensions.log_t() + dimensions.log_k()),
        }
    }

    fn cycle_round_message(
        &mut self,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        let e_in = self.gruen.e_in_current();
        let e_out = self.gruen.e_out_current();
        let r = self.challenges.as_slice();
        let quadratic = match &mut self.cycle {
            CycleSource::Collected { state, .. } => state.quadratic(e_in, e_out),
            CycleSource::Streamed { state, rows } => state.quadratic(rows, e_in, e_out, r)?,
        };
        // A failed streamed `q(1)` pass is held and returned, so its
        // placeholder zero never reaches a round message.
        let mut failure = None;
        let message = self
            .gruen
            .round()
            .ok_or_else(|| eq_round_error(round))?
            .checked_cubic(
                quadratic[0],
                quadratic[1],
                previous_claim,
                round,
                || match &self.cycle {
                    CycleSource::Collected { state, .. } => state.q_at_one(e_in, e_out),
                    CycleSource::Streamed { state, rows } => state
                        .q_at_one(rows, e_in, e_out, r)
                        .unwrap_or_else(|error| {
                            failure = Some(error);
                            F::zero()
                        }),
                },
            );
        match failure {
            Some(error) => Err(error),
            None => message,
        }
    }

    fn bind(&mut self, r: F) -> Result<(), SumcheckError<F>> {
        let round = self.challenges.bound();
        let mut layout_transitioned = false;
        if self.challenges.bound() < self.dimensions.log_t() {
            self.gruen
                .round_mut()
                .ok_or_else(|| eq_round_error(round))?
                .bind(r);
            layout_transitioned = match &mut self.cycle {
                CycleSource::Collected { state, .. } => state.bind(r),
                CycleSource::Streamed { state, .. } => state.bind(r),
            };
        } else {
            self.address.bind(r);
        }
        self.challenges.push(r);

        if self.challenges.bound() == self.dimensions.log_t() {
            let k = 1usize << self.dimensions.log_k();
            // Replacing the state frees the entry allocation here rather
            // than at kernel drop.
            (
                self.address.ra,
                self.address.wa,
                self.address.val,
                self.address.inc_scalar,
            ) = match &mut self.cycle {
                CycleSource::Collected { state, .. } => state.take_dense(k),
                CycleSource::Streamed { state, .. } => state.take_dense(k)?,
            };
            self.address.eq_scalar = self.gruen.current_scalar();
        }

        if layout_transitioned {
            crate::mem::purge_retained_memory(self.dimensions.log_t());
        }
        Ok(())
    }

    /// `Σ_j [index_j hot] · eq(r_address, index_j) · eq(r_cycle, j)` for the
    /// two read operands in one walk — the direct MLE of a one-hot `(K × T)`
    /// grid at the bound point.
    ///
    /// Ports legacy `compute_rs2_ra_claim`: a 2-way split over the joint
    /// `(cycle ‖ address)` index keeps both eq tables at ~√(K·T). Big-endian
    /// joint point `[r_cycle ‖ r_address]`, joint index `(j << addr_bits) | k`.
    /// A streamed source reads the operands from its rows instead of index
    /// columns ([`streaming::operand_claims`]).
    fn one_hot_operand_claims(
        &self,
        r_address: &[F],
        r_cycle: &[F],
    ) -> Result<(F, F), WitnessError> {
        let (rs1_indices, rs2_indices) = match &self.cycle {
            CycleSource::Collected {
                rs1_indices,
                rs2_indices,
                ..
            } => (rs1_indices, rs2_indices),
            CycleSource::Streamed { rows, .. } => {
                return streaming::operand_claims(rows, r_address, r_cycle);
            }
        };
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

        Ok((claims[0], claims[1]))
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
        let (rs1_ra, rs2_ra) = self
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
