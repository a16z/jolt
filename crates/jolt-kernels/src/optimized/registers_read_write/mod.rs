//! Optimized register read/write check (stage 4).
//!
//! Stores at most three sparse entries per cycle and combines reads as
//! `ra = γ·rs1_ra + γ²·rs2_ra`. Gruen factoring handles cycle rounds;
//! address rounds use three dense `K`-sized arrays.
//!
//! [`SeedEntry`] omits the round-0 field value. The first challenge is held
//! without materializing `T/2`; the second bind creates the `T/4` indexed SoA
//! layout. Coefficients stay as LUT indices until the `u16` domain saturates.
//!
//! Only the default read-write config is supported.

use jolt_claims::protocols::jolt::{JoltDerivedId, RegistersReadWritePublic};
use jolt_field::{Accumulator, JoltField};
use jolt_poly::{BindingOrder, EqPolynomial, GruenSplitEqPolynomial, UnivariatePoly};
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

use super::support::{bind_pairs, pin_derived_term, RoundChallenges};
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

mod rows;
mod sparse;
#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test support module")]
pub(crate) mod test_support;
#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests;

pub(crate) use rows::{RegisterCycleRow, SharedRdIndices};

use rows::CollectRegisterEntries;
use sparse::{CoeffLut, IncColumn, SparseEntries};

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
        // Same guard as the reference kernel: phase 1 must cover all cycle
        // rounds. The phase-2/phase-3 split of the address rounds is a legacy
        // data-structure choice with no effect on the round polynomials (the
        // default config sets phase 2 = all `log_K` address rounds), so it is
        // deliberately not constrained here.
        if dimensions.phase1_num_rounds() != dimensions.log_t() {
            return Err(KernelError::Unsupported {
                reason: "optimized registers read-write checking supports only the default \
                         read-write config (phase 1 = all cycle rounds)",
            });
        }
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
        let entries = SparseEntries::Seed {
            entries,
            ra_lut: CoeffLut::new(vec![F::zero(), gamma, gamma_sq, gamma + gamma_sq]),
            wa_lut: CoeffLut::new(vec![F::zero(), F::one()]),
        };

        // Park the rd hot indices for the stage-5 val-evaluation kernel.
        session.park(SharedRdIndices(rd_indices));

        Ok(Box::new(ReadWriteKernel {
            log_t,
            log_k,
            entries,
            gruen: GruenSplitEqPolynomial::new(r_cycle, BindingOrder::LowToHigh),
            inc: IncColumn::Raw(rd_inc),
            ra: Vec::new(),
            wa: Vec::new(),
            val: Vec::new(),
            eq_scalar: F::zero(),
            inc_scalar: F::zero(),
            rs1_indices,
            rs2_indices,
            challenges: RoundChallenges::new(log_t + log_k),
        }))
    }
}

#[cfg_attr(
    feature = "allocative",
    derive(allocative::Allocative),
    allocative(bound = "F: JoltField")
)]
struct ReadWriteKernel<F: JoltField> {
    log_t: usize,
    log_k: usize,
    /// Sparse cycle-major entries, sorted by `(row, col)`; drained at the
    /// cycle→address transition.
    entries: SparseEntries<F>,
    gruen: GruenSplitEqPolynomial<F>,
    inc: IncColumn<F>,
    // Address-phase dense state (K-sized), materialized at the transition.
    #[cfg_attr(feature = "allocative", allocative(visit = jolt_poly::visit_scalars))]
    ra: Vec<F>,
    #[cfg_attr(feature = "allocative", allocative(visit = jolt_poly::visit_scalars))]
    wa: Vec<F>,
    #[cfg_attr(feature = "allocative", allocative(visit = jolt_poly::visit_scalars))]
    val: Vec<F>,
    /// Fully bound `eq(r_cycle, ·)` — constant across the address rounds.
    #[cfg_attr(feature = "allocative", allocative(skip))]
    eq_scalar: F,
    /// Fully bound `rd_inc` — constant across the address rounds.
    #[cfg_attr(feature = "allocative", allocative(skip))]
    inc_scalar: F,
    rs1_indices: Vec<Option<u8>>,
    rs2_indices: Vec<Option<u8>>,
    challenges: RoundChallenges<F>,
}

impl<F: JoltField> ReadWriteKernel<F> {
    /// Cycle-round message via Gruen factoring: the quadratic inner factor's
    /// `[q(0), leading coefficient]` over the remaining cycle domain, wrapped
    /// into the exact cubic by `gruen_poly_deg_3`.
    fn cycle_round_message(&self, previous_claim: F) -> UnivariatePoly<F> {
        let e_in = self.gruen.e_in_current();
        let e_out = self.gruen.e_out_current();
        let quadratic = self.entries.quadratic(e_in, e_out, &self.inc);
        self.gruen
            .gruen_poly_deg_3(quadratic[0], quadratic[1], previous_claim)
    }

    /// Address-round message over the K-sized dense arrays. Cheap enough to
    /// sample all `degree + 1` points directly, so the naive tier's running
    /// claim self-check is kept.
    fn address_round_message(
        &self,
        round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        let half = self.ra.len() / 2;
        let mut evals = [F::zero(); 4];
        for y in 0..half {
            let pair = |table: &[F]| {
                let lo = table[2 * y];
                (lo, table[2 * y + 1] - lo)
            };
            let (ra_0, ra_m) = pair(&self.ra);
            let (wa_0, wa_m) = pair(&self.wa);
            let (val_0, val_m) = pair(&self.val);
            let (mut ra_t, mut wa_t, mut val_t) = (ra_0, wa_0, val_0);
            for eval in &mut evals {
                *eval += wa_t * (self.inc_scalar + val_t) + ra_t * val_t;
                ra_t += ra_m;
                wa_t += wa_m;
                val_t += val_m;
            }
        }
        let evals = evals.map(|eval| self.eq_scalar * eval);
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

    /// Bind the pending challenge: cycle rounds bind eq/inc and merge the
    /// sparse rows; the final cycle bind collapses to the K-sized dense
    /// address state; address rounds bind the three dense arrays.
    fn bind(&mut self, r: F) {
        let mut layout_transitioned = false;
        if self.challenges.bound() < self.log_t {
            self.gruen.bind(r);
            self.inc.bind(r);
            layout_transitioned = self.entries.bind(r);
        } else {
            for table in [&mut self.ra, &mut self.wa, &mut self.val] {
                bind_pairs(table, r);
            }
        }
        self.challenges.push(r);

        if self.challenges.bound() == self.log_t {
            // Replacing the state frees the entry allocation here rather
            // than at kernel drop.
            let entries = std::mem::replace(&mut self.entries, SparseEntries::Direct(Vec::new()));
            (self.ra, self.wa, self.val) = entries.into_dense(1usize << self.log_k);
            self.eq_scalar = self.gruen.current_scalar();
            self.inc_scalar = self.inc.final_scalar();
        }

        // Return replaced entry generations immediately.
        if layout_transitioned {
            crate::mem::purge_staging(self.log_t);
        }
    }

    /// The bound opening point, split as `(r_address, r_cycle)` — the same
    /// reversal `ReadWriteDimensions::read_write_opening_point` applies under
    /// the default config.
    fn bound_point(&self) -> (Vec<F>, Vec<F>) {
        let r_cycle: Vec<F> = self.challenges.as_slice()[..self.log_t]
            .iter()
            .rev()
            .copied()
            .collect();
        let r_address: Vec<F> = self.challenges.as_slice()[self.log_t..]
            .iter()
            .rev()
            .copied()
            .collect();
        (r_address, r_cycle)
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
        let log_t = r_cycle.len();
        let addr_bits = r_address.len();
        let n = log_t + addr_bits;
        let hi_bits = core::cmp::min(log_t, n.div_ceil(2));

        let r_joint: Vec<F> = r_cycle.iter().chain(r_address.iter()).copied().collect();
        let (r_hi, r_lo) = r_joint.split_at(hi_bits);
        let e_hi = EqPolynomial::<F>::evals(r_hi, None);
        let e_lo = EqPolynomial::<F>::evals(r_lo, None);

        let cycle_bits_in_lo = (n - hi_bits) - addr_bits;
        let cycles_per_block = 1usize << cycle_bits_in_lo;
        let cycle_lo_mask = cycles_per_block - 1;

        let block_contribution = |idx_hi: usize| -> [F; 2] {
            let block_start = idx_hi << cycle_bits_in_lo;
            let block_end = core::cmp::min(block_start + cycles_per_block, rs1_indices.len());
            if block_start >= rs1_indices.len() {
                return [F::zero(); 2];
            }
            let mut sums = [F::Accumulator::default(), F::Accumulator::default()];
            for j in block_start..block_end {
                let j_in_block = (j & cycle_lo_mask) << addr_bits;
                if let Some(rs1) = rs1_indices[j] {
                    sums[0].add(e_lo[j_in_block | rs1 as usize]);
                }
                if let Some(rs2) = rs2_indices[j] {
                    sums[1].add(e_lo[j_in_block | rs2 as usize]);
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
        self.log_t + self.log_k
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
        if self.challenges.bound() < self.log_t {
            Ok(self.cycle_round_message(previous_claim))
        } else {
            self.address_round_message(round, previous_claim)
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
        let (r_address, r_cycle) = self.bound_point();
        let (rs1_ra, rs2_ra) = self.one_hot_operand_claims(&r_address, &r_cycle);
        Ok(RegistersReadWriteOutputClaims {
            registers_val: self.val[0],
            rs1_ra,
            rs2_ra,
            rd_wa: self.wa[0],
            rd_inc: self.inc_scalar,
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
            self.eq_scalar,
        )
    }
}
