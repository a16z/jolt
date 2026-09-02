//! The optimized registers read/write-checking (stage 4) kernel: the legacy
//! prover's sparse-matrix algorithm behind the `PrepareKernel` seam.
//!
//! Byte-parity contract: identical round polynomials and output claims to the
//! reference kernel (`reference/registers_read_write.rs`), which sums the
//! summand over dense `2^(log_K + log_T)` register-major tables. This kernel
//! computes the same polynomials from the sparse structure of the one-hot
//! grids — field arithmetic is exact, so algebraic refactorings (eq
//! factoring, γ-combined ra, deferred-reduction accumulation) preserve every
//! wire byte.
//!
//! Carries forward the former sparse read/write-matrix optimizations:
//!
//! - **Sparse cycle-major matrix**: `rd_wa`/`rs1_ra`/`rs2_ra`/`Val` are
//!   represented by ≤ 3 entries per cycle (the touched registers) instead of
//!   three dense `K × T` grids. Between touches a register's value is
//!   constant, so a missing merge partner is inferred from its neighbor's
//!   raw `prev_val`/`next_val` (a constant slice binds to itself).
//! - **γ-combined read coefficient**: one `ra = γ·rs1_ra + γ²·rs2_ra` column
//!   per entry (exact by distributivity).
//! - **Gruen split-eq factoring** for the cycle rounds:
//!   `s(t) = l(t) · Σ_z E_out·E_in·inner(t, z)` via
//!   [`GruenSplitEqPolynomial::gruen_poly_deg_3`].
//! - **Small fixed K**: after the cycle rounds the state collapses to three
//!   `K = 2^REGISTER_ADDRESS_BITS` dense arrays plus two scalars (bound eq,
//!   bound inc); address rounds cost O(K).
//! - **Direct one-hot claims at extraction**: `rs1_ra(r)`/`rs2_ra(r)` are
//!   computed straight from the per-cycle indices with a 2-way split-eq walk
//!   (legacy's `compute_rs2_ra_claim`, applied to both operands — no γ⁻¹).
//!
//! - **Compact coefficient lookup tables** (legacy's
//!   `OneHotCoeffLookupTable`): entries carry a `u16` read index and `u8`
//!   write index instead of two field elements, and cycle rows fit in `u32` —
//!   40 bytes per Fp128 entry through the first three cycle binds, exactly
//!   where the entry count peaks (≤ 3·T). The
//!   tables square on each bind (all `b + r·(a − b)` pairs) and the entries
//!   combine indices, so every looked-up value equals the field element the
//!   direct representation would hold; entries deref to field coefficients
//!   when one more squaring would overflow the `u16` index domain.
//!
//! Like the reference kernel, only the default read-write config (phase 1 =
//! all cycle rounds, phase 2 = 0) is supported.

use jolt_claims::protocols::jolt::geometry::registers::rd_inc_read_write;
use jolt_claims::protocols::jolt::{JoltDerivedId, RegistersReadWritePublic};
use jolt_field::{Accumulator, JoltField};
use jolt_poly::{BindingOrder, EqPolynomial, GruenSplitEqPolynomial, Polynomial, UnivariatePoly};
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
use sparse::{
    bind_sparse_entries, sparse_quadratic, CoeffLut, OneHotCoeff, ReadWriteKernel, SparseEntries,
};

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
        if log_t >= 32 {
            return Err(KernelError::Unsupported {
                reason: "optimized registers read-write checking requires fewer than 2^32 cycles",
            });
        }
        let cycles = 1usize << log_t;

        let inc_table: Vec<F> = witness.oracle_table(rd_inc_read_write().polynomial_id())?;
        if inc_table.len() != cycles {
            return Err(KernelError::TableSizeMismatch {
                table: format!("{:?}", rd_inc_read_write()),
                expected: cycles,
                got: inc_table.len(),
            });
        }

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
        } = CollectRegisterEntries::collect(witness, cycles)?;
        let entries = SparseEntries::Indexed {
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
            inc: Polynomial::new(inc_table),
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

/// The sparse entries in their round-dependent coefficient representation:
/// `u16` LUT indices while the tables can still square (the first four cycle
/// rounds — the peak-memory window), direct field values after.
impl<F: JoltField> ReadWriteKernel<F> {
    /// Cycle-round message via Gruen factoring: the quadratic inner factor's
    /// `[q(0), leading coefficient]` over the remaining cycle domain, wrapped
    /// into the exact cubic by `gruen_poly_deg_3`.
    fn cycle_round_message(&self, previous_claim: F) -> UnivariatePoly<F> {
        let e_in = self.gruen.e_in_current();
        let e_out = self.gruen.e_out_current();
        let inc = self.inc.evals();
        let quadratic = match &self.entries {
            SparseEntries::Indexed {
                entries,
                ra_lut,
                wa_lut,
            } => sparse_quadratic(entries, ra_lut, wa_lut, e_in, e_out, inc),
            SparseEntries::Direct(entries) => {
                let unused = SparseEntries::unused_lut();
                sparse_quadratic(entries, &unused, &unused, e_in, e_out, inc)
            }
        };

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

    fn bind_sparse(&mut self, r: F) {
        // Dereference to direct field coefficients when one more table
        // squaring would overflow the u16 index domain (after the third
        // cycle bind under the seed sizes) — by then the entry count has
        // started merging down, so the wider entries no longer set the peak.
        let saturated = matches!(
            &self.entries,
            SparseEntries::Indexed { ra_lut, wa_lut, .. }
                if ra_lut.saturated() || wa_lut.saturated()
        );
        if saturated {
            if let SparseEntries::Indexed {
                entries,
                ra_lut,
                wa_lut,
            } = std::mem::replace(&mut self.entries, SparseEntries::Direct(Vec::new()))
            {
                self.entries =
                    SparseEntries::Direct(SparseEntries::deref(entries, &ra_lut, &wa_lut));
            }
        }
        match &mut self.entries {
            SparseEntries::Indexed {
                entries,
                ra_lut,
                wa_lut,
            } => {
                // Entries combine indices against the CURRENT table widths;
                // the tables then square so the combined indices address the
                // bound values.
                bind_sparse_entries(entries, r, ra_lut, wa_lut);
                ra_lut.bind(r);
                wa_lut.bind(r);
            }
            SparseEntries::Direct(entries) => {
                let unused = SparseEntries::unused_lut();
                bind_sparse_entries(entries, r, &unused, &unused);
            }
        }
    }

    /// Bind the pending challenge: cycle rounds bind eq/inc and merge the
    /// sparse rows; the final cycle bind collapses to the K-sized dense
    /// address state; address rounds bind the three dense arrays.
    fn bind(&mut self, r: F) {
        if self.challenges.bound() < self.log_t {
            self.gruen.bind(r);
            self.inc.bind_with_order(r, BindingOrder::LowToHigh);
            self.bind_sparse(r);
        } else {
            for table in [&mut self.ra, &mut self.wa, &mut self.val] {
                bind_pairs(table, r);
            }
        }
        self.challenges.push(r);

        if self.challenges.bound() == self.log_t {
            let k = 1usize << self.log_k;
            let mut ra = vec![F::zero(); k];
            let mut wa = vec![F::zero(); k];
            let mut val = vec![F::zero(); k];
            // Replacing the state frees the entry allocation here rather
            // than at kernel drop.
            match std::mem::replace(&mut self.entries, SparseEntries::Direct(Vec::new())) {
                SparseEntries::Indexed {
                    entries,
                    ra_lut,
                    wa_lut,
                } => {
                    for entry in entries {
                        debug_assert_eq!(entry.row, 0);
                        ra[entry.col as usize] = entry.ra.value(&ra_lut);
                        wa[entry.col as usize] = entry.wa.value(&wa_lut);
                        val[entry.col as usize] = entry.val;
                    }
                }
                SparseEntries::Direct(entries) => {
                    for entry in entries {
                        debug_assert_eq!(entry.row, 0);
                        ra[entry.col as usize] = entry.ra;
                        wa[entry.col as usize] = entry.wa;
                        val[entry.col as usize] = entry.val;
                    }
                }
            }
            self.ra = ra;
            self.wa = wa;
            self.val = val;
            self.eq_scalar = self.gruen.current_scalar();
            self.inc_scalar = self.inc.evals()[0];
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
}

impl<F: JoltField> ReadWriteKernel<F> {
    /// `Σ_j [index_j hot] · eq(r_address, index_j) · eq(r_cycle, j)` for the two
    /// read operands in one walk — the direct MLE of a one-hot `(K × T)` grid at
    /// the bound point.
    ///
    /// Ports legacy `compute_rs2_ra_claim`: a 2-way split over the joint
    /// `(cycle ‖ address)` index keeps both eq tables at ~√(K·T). Big-endian
    /// joint point `[r_cycle ‖ r_address]`, joint index `(j << addr_bits) | k`.
    fn one_hot_operand_claims(&self, r_address: &[F], r_cycle: &[F]) -> (F, F) {
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
            let block_end = core::cmp::min(block_start + cycles_per_block, self.rs1_indices.len());
            if block_start >= self.rs1_indices.len() {
                return [F::zero(); 2];
            }
            let mut sums = [F::Accumulator::default(), F::Accumulator::default()];
            for j in block_start..block_end {
                let j_in_block = (j & cycle_lo_mask) << addr_bits;
                if let Some(rs1) = self.rs1_indices[j] {
                    sums[0].add(e_lo[j_in_block | rs1 as usize]);
                }
                if let Some(rs2) = self.rs2_indices[j] {
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
        let id = JoltDerivedId::from(RegistersReadWritePublic::EqCycle);
        pin_derived_term(
            relation,
            id,
            input_points,
            output_points,
            challenges,
            self.eq_scalar,
        )
    }
}
