//! Optimized bytecode read+RAF kernels: the stage-6a address phase and the
//! stage-6b cycle phase, byte-parity twins of the reference kernels in
//! [`crate::reference::bytecode_read_raf`].
//!
//! Ported legacy techniques (`jolt-prover-legacy/src/zkvm/bytecode/read_raf_checking.rs`):
//!
//! - **Split-eq two-table pushforwards** (address phase): the per-stage
//!   `F_s(k) = Σ_{j: pc(j)=k} eq(r_cycle_s, j)` tables are accumulated as
//!   `Σ_{j_hi} E_hi_s[j_hi] · (Σ_{j_lo: pc=k} E_lo_s[j_lo])` — the inner sums
//!   are additions only and the five stages share one trace walk, so the eq
//!   tables cost `O(√T)` each instead of `O(T)` and the `O(T)` walk pays one
//!   PC lookup per cycle for all stages
//!   (legacy `BytecodeReadRafAddressSumcheckProver::initialize`).
//! - **Sparse one-hot RA** (cycle phase): the committed `BytecodeRa(i)`
//!   factors are materialized as `ra_i(j) = eq(chunk_i)[chunk_i(pc_j)]` from
//!   per-cycle mapped PCs (cold cycles are zero), never as `K x T` one-hot
//!   grids — the legacy `RaPolynomial` trick, `O(T)` per chunk
//!   (legacy `BytecodeReadRafCycleSumcheckProver::initialize`, the 1.13s
//!   member of legacy stage 6b at `2^23`).
//! - **Linear-leaf fusion** (cycle phase): the five `StageCycleEq · Val`
//!   pairs, both RAF terms (`int(r_address)`-scaled stage-1/3 eqs), and the
//!   entry selector all enter the summand linearly, so they collapse into ONE
//!   combined coefficient table `C(j)`; the summand becomes
//!   `Π_i ra_i(j) · C(j)` — `d+2` bound tables instead of the reference's
//!   `13`, with field-identical round messages (multilinear extension is
//!   linear in the table argument). The RAF weights fold as the legacy
//!   `raf_int_weight` does: stage 1 carries `γ⁵·int_r`, stage 3 `γ⁶·int_r`.
//! - **Eval-at-1 recovery** and **rayon walks**, both phases (see the module
//!   docs on [`crate::optimized`]).
//!
//! The two phases share one per-proof trace scan through the
//! [`ProofSession`]: whichever phase prepares first parks the packed
//! per-cycle PC rows; the other reclaims them (`state_or_insert_with` keyed
//! by the private [`PcRowsKey`] type — the modular equivalent of legacy's
//! shared `Arc<Vec<Cycle>>`).

use std::ops::Range;
use std::sync::Arc;

use jolt_claims::protocols::jolt::geometry::bytecode::{
    self, read_raf_stage_values, BytecodeReadRafStageValueInputs,
};
use jolt_claims::protocols::jolt::geometry::dimensions::{
    committed_address_chunks, REGISTER_ADDRESS_BITS,
};
use jolt_claims::protocols::jolt::JoltOpeningId;
use jolt_claims::OutputClaims;
use jolt_field::Field;
use jolt_poly::{IdentityPolynomial, MultilinearEvaluation, Polynomial, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::{
    ConcreteSumcheck, SumcheckInputClaims, SumcheckOutputClaims,
};
use jolt_verifier::stages::stage6a::bytecode_read_raf::{
    BytecodeReadRafAddressPhase, BytecodeReadRafAddressPhaseOutputClaims,
};
use jolt_verifier::stages::stage6b::bytecode_read_raf::BytecodeReadRafCycle;
use jolt_witness::witnesses::{BytecodePc, MappedPc, RaChunkSelector};
use jolt_witness::{JoltWitnessPlane, WitnessBundle};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::lazy_ra::{ChunkIndexSource, LazyFoldedRa};
use super::support::{
    bind_all, collect_rows, eq_table, gamma_powers_array, pair, par_sum_pair_groups,
    par_sum_pair_groups_reusing, round_poly_from_skipped_evals, scaled_eq_table, RoundProgress,
};
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

/// Sentinel for a cold cycle (no bytecode mapping) in [`PcRow::mapped_pc`].
const COLD: u32 = u32::MAX;

/// One cycle's packed bytecode PC facts: the pushforward slot (no-ops and
/// unmapped rows land on 0 — the address-phase convention) and the committed
/// one-hot hot index (unmapped rows are cold — the cycle-phase convention).
#[derive(Clone, Copy, Debug)]
pub(crate) struct PcRow {
    push_pc: u32,
    mapped_pc: u32,
}

/// The session key of the shared per-cycle PC scan.
struct PcRowsKey(Arc<Vec<PcRow>>);

#[cfg(feature = "allocative")]
crate::optimized::impl_allocative!(PcRowsKey, |rows| {
    crate::backend::arc_vec_heap_bytes(&rows.0)
});

#[derive(Clone, Copy, Debug, WitnessBundle)]
struct PcBundle {
    bytecode_pc: BytecodePc,
    mapped_pc: MappedPc,
}

impl PcRow {
    /// One trace scan per proof, shared by both phases through the session.
    fn shared<F: Field>(
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        cycles: usize,
    ) -> Result<Arc<Vec<PcRow>>, KernelError<F>> {
        if session.state::<PcRowsKey>().is_none() {
            let bundles: Vec<PcBundle> = collect_rows(witness, cycles)?;
            let pack = |bundle: &PcBundle| {
                let mapped = match bundle.mapped_pc.0 {
                    Some(pc) if pc as u32 as usize == pc && pc as u32 != COLD => pc as u32,
                    Some(_) => {
                        return Err(KernelError::InvariantViolation {
                            reason: "bytecode PC exceeds the packed u32 range",
                        })
                    }
                    None => COLD,
                };
                if bundle.bytecode_pc.0 as u32 as usize != bundle.bytecode_pc.0 {
                    return Err(KernelError::InvariantViolation {
                        reason: "bytecode PC exceeds the packed u32 range",
                    });
                }
                Ok(PcRow {
                    push_pc: bundle.bytecode_pc.0 as u32,
                    mapped_pc: mapped,
                })
            };
            #[cfg(feature = "parallel")]
            let rows = bundles
                .par_iter()
                .map(pack)
                .collect::<Result<Vec<_>, _>>()?;
            #[cfg(not(feature = "parallel"))]
            let rows = bundles.iter().map(pack).collect::<Result<Vec<_>, _>>()?;
            session.park(PcRowsKey(Arc::new(rows)));
        }
        let rows = session
            .state::<PcRowsKey>()
            .map(|key| Arc::clone(&key.0))
            .ok_or(KernelError::InvariantViolation {
                reason: "bytecode PC rows vanished from the session",
            })?;
        if rows.len() != cycles {
            return Err(KernelError::TableSizeMismatch {
                table: "bytecode cycle PC rows".to_owned(),
                expected: cycles,
                got: rows.len(),
            });
        }
        Ok(rows)
    }
}

/// The five per-stage cycle-eq pushforwards onto the bytecode address domain,
/// all stages in one trace walk over the split-eq two-table decomposition.
fn stage_pushforwards<F: Field>(
    stage_cycle_points: &[Vec<F>; 5],
    rows: &[PcRow],
    addresses: usize,
) -> [Vec<F>; 5] {
    let log_t = stage_cycle_points[0].len();
    let lo_bits = log_t / 2;
    let hi_bits = log_t - lo_bits;
    let in_len = 1usize << lo_bits;
    let out_len = 1usize << hi_bits;

    // Big-endian points split as eq(r, j) = eq(r[..hi], j_hi) · eq(r[hi..], j_lo)
    // with j = (j_hi << lo_bits) | j_lo.
    let e_hi: [Vec<F>; 5] = std::array::from_fn(|s| eq_table(&stage_cycle_points[s][..hi_bits]));
    let e_lo: [Vec<F>; 5] = std::array::from_fn(|s| eq_table(&stage_cycle_points[s][hi_bits..]));

    let block = |range: Range<usize>| -> [Vec<F>; 5] {
        let mut partial: [Vec<F>; 5] = std::array::from_fn(|_| vec![F::zero(); addresses]);
        let mut inner: [Vec<F>; 5] = std::array::from_fn(|_| vec![F::zero(); addresses]);
        let mut touched: Vec<usize> = Vec::with_capacity(in_len);
        // Exact membership marker for `touched`: a field-value test
        // (`inner[0][pc].is_zero()`) is not one — zero eq weights or
        // cancellation would re-push a PC and double-count it in the fold
        // below. Epochs make the reset per `j_hi` block free; the counter is
        // bounded by the block range length (≤ 2^hi_bits), far below u32.
        let mut seen: Vec<u32> = vec![0; addresses];
        let mut epoch = 0u32;
        for j_hi in range {
            for &k in &touched {
                for stage_inner in &mut inner {
                    stage_inner[k] = F::zero();
                }
            }
            touched.clear();
            epoch += 1;
            let base = j_hi << lo_bits;
            for j_lo in 0..in_len {
                let pc = rows[base + j_lo].push_pc as usize;
                if seen[pc] != epoch {
                    seen[pc] = epoch;
                    touched.push(pc);
                }
                for (stage_inner, stage_lo) in inner.iter_mut().zip(&e_lo) {
                    stage_inner[pc] += stage_lo[j_lo];
                }
            }
            for &k in &touched {
                for ((stage_partial, stage_inner), stage_hi) in
                    partial.iter_mut().zip(&inner).zip(&e_hi)
                {
                    stage_partial[k] += stage_hi[j_hi] * stage_inner[k];
                }
            }
        }
        partial
    };

    #[cfg(feature = "parallel")]
    {
        let num_threads = rayon::current_num_threads();
        let chunk = out_len.div_ceil(num_threads).max(1);
        (0..out_len)
            .into_par_iter()
            .step_by(chunk)
            .map(|start| block(start..(start + chunk).min(out_len)))
            .reduce(
                || std::array::from_fn(|_| vec![F::zero(); addresses]),
                |mut left, right| {
                    for (left, right) in left.iter_mut().zip(right) {
                        for (left, right) in left.iter_mut().zip(right) {
                            *left += right;
                        }
                    }
                    left
                },
            )
    }
    #[cfg(not(feature = "parallel"))]
    {
        block(0..out_len)
    }
}

/// Stage-6a address phase: `PrepareKernel` front of the optimized kernel.
pub struct OptimizedBytecodeReadRafAddress;

impl<F: Field> PrepareKernel<F, BytecodeReadRafAddressPhase<F>>
    for OptimizedBytecodeReadRafAddress
{
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, BytecodeReadRafAddressPhase<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = BytecodeReadRafAddressPhase<F>>>, KernelError<F>>
    {
        let relation = inputs.relation;
        let dimensions = relation.dimensions();
        let addresses = 1usize << dimensions.log_k();
        let cycles = 1usize << dimensions.log_t();

        let program = witness.program_preprocessing();
        if program.bytecode.bytecode.len() != addresses {
            return Err(KernelError::TableSizeMismatch {
                table: "bytecode stage values".to_owned(),
                expected: addresses,
                got: program.bytecode.bytecode.len(),
            });
        }
        let stage_gammas = inputs.challenges.stage_gamma_powers();
        let stage_values = read_raf_stage_values(BytecodeReadRafStageValueInputs {
            bytecode: &program.bytecode.bytecode,
            register_read_write_point: &relation.register_read_write_point()
                [..REGISTER_ADDRESS_BITS],
            register_val_evaluation_point: &relation.register_val_evaluation_point()
                [..REGISTER_ADDRESS_BITS],
            stage1_gammas: &stage_gammas[0],
            stage2_gammas: &stage_gammas[1],
            stage3_gammas: &stage_gammas[2],
            stage4_gammas: &stage_gammas[3],
            stage5_gammas: &stage_gammas[4],
        });

        let stage_cycle_points = relation.stage_cycle_points();
        for point in stage_cycle_points {
            if point.len() != dimensions.log_t() {
                return Err(KernelError::InvariantViolation {
                    reason: "bytecode stage cycle point has the wrong variable count",
                });
            }
        }
        let rows = PcRow::shared(session, witness, cycles)?;
        let entry_bytecode_index = relation.entry_bytecode_index();
        if entry_bytecode_index >= addresses
            || rows.iter().any(|row| row.push_pc as usize >= addresses)
        {
            return Err(KernelError::InvariantViolation {
                reason: "bytecode index outside the padded bytecode domain",
            });
        }

        let gamma_powers: [F; 8] = gamma_powers_array(inputs.challenges.gamma);

        let pushforwards =
            stage_pushforwards(stage_cycle_points, &rows, addresses).map(Polynomial::new);
        let values: [Polynomial<F>; 5] = std::array::from_fn(|s| {
            Polynomial::new(stage_values.iter().map(|row| row[s]).collect())
        });
        let int_table = Polynomial::new((0..addresses).map(|k| F::from_u64(k as u64)).collect());
        let one_hot = |index: usize| {
            let mut table = vec![F::zero(); addresses];
            table[index] = F::one();
            Polynomial::new(table)
        };

        Ok(Box::new(AddressKernel {
            progress: RoundProgress::new(relation.rounds()),
            committed_program: relation.committed_program(),
            stage_weights: std::array::from_fn(|s| gamma_powers[s]),
            entry_weight: gamma_powers[7],
            raf_weights: [
                gamma_powers[5],
                F::zero(),
                gamma_powers[4],
                F::zero(),
                F::zero(),
            ],
            pushforwards,
            values,
            int_table,
            entry_trace: one_hot(rows[0].push_pc as usize),
            entry_expected: one_hot(entry_bytecode_index),
        }))
    }
}

struct AddressKernel<F: Field> {
    progress: RoundProgress,
    committed_program: bool,
    stage_weights: [F; 5],
    entry_weight: F,
    /// Within-stage RAF `Int` weights: the overall γ⁵/γ⁶ RAF weights divided
    /// by the γ⁰/γ² stage weights (legacy `raf_int_weight`).
    raf_weights: [F; 5],
    pushforwards: [Polynomial<F>; 5],
    /// RAW stage-value tables — the RAF identity binds separately so
    /// committed mode can stage the raw bound `Val_s` wire claims.
    values: [Polynomial<F>; 5],
    int_table: Polynomial<F>,
    entry_trace: Polynomial<F>,
    entry_expected: Polynomial<F>,
}

#[cfg(feature = "allocative")]
crate::optimized::impl_field_allocative!(AddressKernel, |kernel| {
    use crate::backend::poly_heap_bytes;
    kernel
        .pushforwards
        .iter()
        .chain(&kernel.values)
        .map(poly_heap_bytes)
        .sum::<usize>()
        + poly_heap_bytes(&kernel.int_table)
        + poly_heap_bytes(&kernel.entry_trace)
        + poly_heap_bytes(&kernel.entry_expected)
});

impl<F: Field> AddressKernel<F> {
    fn bind(&mut self, challenge: F) {
        bind_all(
            self.pushforwards
                .iter_mut()
                .chain(self.values.iter_mut())
                .chain([
                    &mut self.int_table,
                    &mut self.entry_trace,
                    &mut self.entry_expected,
                ]),
            challenge,
        );
        self.progress.advance();
    }

    /// The summand's evaluations at `t ∈ {0, 2}` summed over group `y`.
    #[inline]
    fn group_evals(&self, y: usize) -> [F; 2] {
        let (int_lo, int_hi) = pair(&self.int_table, y);
        let int_delta = int_hi - int_lo;
        let int_at_2 = int_hi + int_delta;
        let mut out = [F::zero(); 2];
        for s in 0..5 {
            let (f_lo, f_hi) = pair(&self.pushforwards[s], y);
            let (v_lo, v_hi) = pair(&self.values[s], y);
            let raf = self.raf_weights[s];
            let val_at_0 = v_lo + raf * int_lo;
            let val_at_2 = (v_hi + v_hi - v_lo) + raf * int_at_2;
            out[0] += self.stage_weights[s] * f_lo * val_at_0;
            out[1] += self.stage_weights[s] * (f_hi + f_hi - f_lo) * val_at_2;
        }
        let (t_lo, t_hi) = pair(&self.entry_trace, y);
        let (e_lo, e_hi) = pair(&self.entry_expected, y);
        out[0] += self.entry_weight * t_lo * e_lo;
        out[1] += self.entry_weight * (t_hi + t_hi - t_lo) * (e_hi + e_hi - e_lo);
        out
    }
}

impl<F: Field> ProveRounds<F> for AddressKernel<F> {
    fn num_rounds(&self) -> usize {
        self.progress.total()
    }

    fn prove_round(
        &mut self,
        bind: Option<F>,
        _round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if let Some(challenge) = bind {
            self.bind(challenge);
        }
        let half = self.entry_trace.len() / 2;

        let evals = par_sum_pair_groups(half, 2, |acc, y| {
            let group = self.group_evals(y);
            acc[0] += group[0];
            acc[1] += group[1];
        });

        Ok(round_poly_from_skipped_evals(&evals, previous_claim))
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind);
        Ok(())
    }
}

impl<F: Field> SumcheckKernel<F> for AddressKernel<F> {
    type Relation = BytecodeReadRafAddressPhase<F>;

    fn output_claims(
        &mut self,
        _inputs: &SumcheckInputClaims<F, Self::Relation>,
    ) -> Result<BytecodeReadRafAddressPhaseOutputClaims<F>, SumcheckKernelError<F>> {
        self.progress.require_complete()?;
        let mut intermediate =
            self.entry_weight * self.entry_trace.evals()[0] * self.entry_expected.evals()[0];
        let bound_int = self.int_table.evals()[0];
        for s in 0..5 {
            intermediate += self.stage_weights[s]
                * self.pushforwards[s].evals()[0]
                * (self.values[s].evals()[0] + self.raf_weights[s] * bound_int);
        }
        let val_stages = if self.committed_program {
            self.values.iter().map(|table| table.evals()[0]).collect()
        } else {
            Vec::new()
        };
        Ok(BytecodeReadRafAddressPhaseOutputClaims {
            intermediate,
            val_stages,
        })
    }
}

/// Stage-6b cycle phase: `PrepareKernel` front of the optimized kernel.
pub struct OptimizedBytecodeReadRafCycle;

impl<F: Field> PrepareKernel<F, BytecodeReadRafCycle<F>> for OptimizedBytecodeReadRafCycle {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, BytecodeReadRafCycle<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = BytecodeReadRafCycle<F>>>, KernelError<F>>
    {
        let relation = inputs.relation;
        let dimensions = relation.dimensions();
        let r_address = relation.r_address();
        let stage_cycle_points = relation.stage_cycle_points();
        let cycles = 1usize << dimensions.log_t();
        let num_ra = dimensions.num_committed_ra_polys();

        let chunks = committed_address_chunks(r_address, relation.committed_chunk_bits());
        if chunks.len() != num_ra {
            return Err(KernelError::InvariantViolation {
                reason: "bytecode address chunk count disagrees with the committed RA count",
            });
        }
        let rows = PcRow::shared(session, witness, cycles)?;
        // This is the PC scan's last consumer: remove the session's copy so
        // the rows free at the lazy fold's materialization instead of living
        // to the end of the proof.
        let _ = session.take::<PcRowsKey>();

        // ra_i(j) = eq(chunk_i)[chunk_i(pc_j)] — the address fold of the
        // one-hot grid, served lazily off the sparse per-cycle indices for
        // the first four binds instead of `d × T` dense.
        let chunk_eqs: Vec<Vec<F>> = chunks.iter().map(|chunk| eq_table(chunk)).collect();
        let selectors = (0..num_ra)
            .map(|index| {
                RaChunkSelector::new(index, num_ra, relation.committed_chunk_bits())
                    .map_err(KernelError::from)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let ra = LazyFoldedRa::new(chunk_eqs, BytecodePcChunks { rows, selectors });

        // The combined coefficient table: every non-RA factor of the summand
        // is linear in one cycle table, so
        //   C(j) = Σ_s (γ^s·val_s + raf_s·int_r)·eq_s(j) + γ⁷·entry·[j = 0]
        // with raf_0 = γ⁵·int_r, raf_2 = γ⁶·int_r (SpartanOuterRaf rides the
        // stage-1 cycle point, SpartanShiftRaf the stage-3 one).
        let stage_values = relation.stage_values_at_r_address()?;
        let gamma_powers: [F; 8] = gamma_powers_array(inputs.challenges.gamma);
        let int_at_r_address = IdentityPolynomial::new(r_address.len()).evaluate(r_address);
        let mut stage_weights: [F; 5] = std::array::from_fn(|s| gamma_powers[s] * stage_values[s]);
        stage_weights[0] += gamma_powers[5] * int_at_r_address;
        stage_weights[2] += gamma_powers[6] * int_at_r_address;

        let mut combined = vec![F::zero(); cycles];
        for (point, weight) in stage_cycle_points.iter().zip(stage_weights) {
            if point.len() != dimensions.log_t() {
                return Err(KernelError::InvariantViolation {
                    reason: "bytecode stage cycle point has the wrong variable count",
                });
            }
            let scaled = scaled_eq_table(point, weight);
            #[cfg(feature = "parallel")]
            combined
                .par_iter_mut()
                .zip(scaled.par_iter())
                .for_each(|(acc, term)| *acc += *term);
            #[cfg(not(feature = "parallel"))]
            combined
                .iter_mut()
                .zip(scaled.iter())
                .for_each(|(acc, term)| *acc += *term);
        }
        let entry_scalar = eq_table(r_address)[relation.entry_bytecode_index()];
        combined[0] += gamma_powers[7] * entry_scalar;

        let output_openings = bytecode::read_raf_output_openings(dimensions).bytecode_ra;
        if output_openings.len() != num_ra {
            return Err(KernelError::InvariantViolation {
                reason: "bytecode RA output opening count disagrees with the committed RA count",
            });
        }

        Ok(Box::new(CycleKernel {
            progress: RoundProgress::new(relation.rounds()),
            degree: relation.degree(),
            ra,
            combined: Polynomial::new(combined),
            output_openings,
        }))
    }
}

/// Lazy-RA index source: chunk `i` of the per-cycle mapped bytecode PC,
/// cold on unmapped cycles, off the session-shared PC scan.
struct BytecodePcChunks {
    rows: Arc<Vec<PcRow>>,
    selectors: Vec<RaChunkSelector>,
}

impl ChunkIndexSource for BytecodePcChunks {
    fn num_polys(&self) -> usize {
        self.selectors.len()
    }

    fn cycles(&self) -> usize {
        self.rows.len()
    }

    #[inline]
    fn index(&self, i: usize, j: usize) -> Option<usize> {
        let row = &self.rows[j];
        if row.mapped_pc == COLD {
            return None;
        }
        Some(self.selectors[i].chunk_usize(row.mapped_pc as usize))
    }
}

struct CycleKernel<F: Field> {
    progress: RoundProgress,
    degree: usize,
    ra: LazyFoldedRa<F, BytecodePcChunks>,
    combined: Polynomial<F>,
    /// The produced `BytecodeRa` opening ids, in `read_raf_output_openings`
    /// order (index-aligned with `ra`).
    output_openings: Vec<JoltOpeningId>,
}

#[cfg(feature = "allocative")]
crate::optimized::impl_field_allocative!(CycleKernel, |kernel| {
    use crate::backend::{arc_vec_heap_bytes, poly_heap_bytes, vec_heap_bytes};
    kernel
        .ra
        .heap_bytes(|source| arc_vec_heap_bytes(&source.rows) + vec_heap_bytes(&source.selectors))
        + poly_heap_bytes(&kernel.combined)
        + vec_heap_bytes(&kernel.output_openings)
});

impl<F: Field> CycleKernel<F> {
    fn bind(&mut self, challenge: F) {
        bind_all([&mut self.combined], challenge);
        self.ra.bind(challenge);
        self.progress.advance();
    }

    /// The summand's evaluations at `t ∈ {0, 2, 3, .., degree}` summed over
    /// group `y`, written into `acc` (length `degree`); `ra_pairs` is the
    /// caller's per-group `(lo, hi)` scratch (each RA pair is gathered once
    /// per group, not once per sample point).
    #[inline]
    fn accumulate_group(&self, y: usize, acc: &mut [F], ra_pairs: &mut [(F, F)]) {
        let (c_lo, c_hi) = pair(&self.combined, y);
        let c_delta = c_hi - c_lo;
        for (i, slot) in ra_pairs.iter_mut().enumerate() {
            *slot = self.ra.lo_hi(i, y);
        }
        acc[0] += ra_pairs.iter().fold(c_lo, |acc, (lo, _)| acc * *lo);
        for (slot, t) in (2..=self.degree).enumerate() {
            let t_value = F::from_u64(t as u64);
            acc[slot + 1] += ra_pairs
                .iter()
                .fold(c_lo + t_value * c_delta, |acc, (lo, hi)| {
                    acc * (*lo + t_value * (*hi - *lo))
                });
        }
    }
}

impl<F: Field> ProveRounds<F> for CycleKernel<F> {
    fn num_rounds(&self) -> usize {
        self.progress.total()
    }

    fn prove_round(
        &mut self,
        bind: Option<F>,
        _round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if let Some(challenge) = bind {
            self.bind(challenge);
        }
        let half = self.combined.len() / 2;
        let slots = self.degree;
        let num_ra = self.ra.num_polys();

        let evals = par_sum_pair_groups_reusing(
            half,
            slots,
            || vec![(F::zero(), F::zero()); num_ra],
            |acc, ra_pairs, y| self.accumulate_group(y, acc, ra_pairs),
        );

        Ok(round_poly_from_skipped_evals(&evals, previous_claim))
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind);
        Ok(())
    }
}

impl<F: Field> SumcheckKernel<F> for CycleKernel<F> {
    type Relation = BytecodeReadRafCycle<F>;

    fn output_claims(
        &mut self,
        _inputs: &SumcheckInputClaims<F, Self::Relation>,
    ) -> Result<SumcheckOutputClaims<F, Self::Relation>, SumcheckKernelError<F>> {
        self.progress.require_complete()?;
        let ra = &self.ra;
        let output_openings = &self.output_openings;
        SumcheckOutputClaims::<F, Self::Relation>::from_opening_values(|id: &JoltOpeningId| {
            output_openings
                .iter()
                .position(|opening| opening == id)
                .map(|index| ra.value(index, 0))
        })
        .map_err(SumcheckKernelError::from)
    }
}

/// Byte-parity of both phases against the reference kernels, run as a PAIR
/// through one shared [`ProofSession`] so the parked per-cycle PC scan flows
/// the way production stages would exercise it.
///
/// Fixture honesty: `with_sample_backend` is the only witness plane
/// constructible without a `jolt-program` dependency. At its scale the
/// bytecode decomposition has a single committed chunk (`d = 1`), so the
/// cycle kernel's multi-factor `Π_i ra_i` loop runs with one factor; the
/// degree-`d+1` sampling, cold-cycle zeroing, entry/RAF fusion, and both
/// pushforward paths are exercised for real.
#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use jolt_claims::protocols::jolt::geometry::bytecode::BytecodeReadRafDimensions;
    use jolt_claims::protocols::jolt::relations::bytecode::BytecodeReadRafAddressPhaseChallenges;
    use jolt_claims::protocols::jolt::{JoltCommittedPolynomial, JoltPolynomialId};
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_verifier::stages::stage6a::bytecode_read_raf::{
        BytecodeReadRafAddressPhaseInputClaims, BytecodeStagePoints,
    };
    use jolt_verifier::stages::stage6b::bytecode_read_raf::{
        BytecodeReadRafCycleInputs, BytecodeReadRafCyclePhaseCommittedChallenges,
        BytecodeReadRafInputClaims, BytecodeReadRafTableFoldInputs,
    };
    use jolt_witness::testing::with_sample_backend;
    use jolt_witness::{JoltWitnessOracle, ProgramSource};

    use super::*;
    use crate::optimized::parity::{
        probe_input_claim, probe_one_hot_family, run_lockstep, synthetic_point,
    };
    use crate::ReferenceBackend;

    fn fr(value: u64) -> Fr {
        Fr::from_u64(value)
    }

    /// `stage_pushforwards` membership must not depend on field values: with
    /// a Boolean low-point coordinate, stage 0's eq weight is exactly zero on
    /// half the low domain, so a repeated PC whose first visit carried weight
    /// zero used to be pushed into `touched` twice and double-counted in
    /// every stage's fold.
    #[test]
    fn stage_pushforwards_membership_is_value_independent() {
        let log_t = 4usize;
        let addresses = 4usize;
        // Stage 0's low half becomes [1, r]: eq weights are literally zero
        // for j_lo ∈ {0, 1}. Stages 1–4 keep generic points.
        let mut stage_cycle_points: [Vec<Fr>; 5] =
            std::array::from_fn(|s| synthetic_point(log_t, 101 + s as u64));
        stage_cycle_points[0][2] = fr(1);

        // Block j_hi = 0 visits PC 2 first at zero stage-0 weight (j_lo = 0),
        // then again at j_lo = 2; later blocks repeat PCs generically.
        let pcs: [usize; 16] = [2, 0, 2, 1, 3, 3, 0, 2, 1, 1, 1, 1, 0, 3, 2, 0];
        let rows: Vec<PcRow> = pcs
            .iter()
            .map(|&pc| PcRow {
                push_pc: pc as u32,
                mapped_pc: 0,
            })
            .collect();

        // The naive pushforward over the full eq tables — the same monomials
        // the split accumulates, so equality is exact.
        let expected: [Vec<Fr>; 5] = std::array::from_fn(|s| {
            let eq: Vec<Fr> = eq_table(&stage_cycle_points[s]);
            let mut out = vec![fr(0); addresses];
            for (j, &pc) in pcs.iter().enumerate() {
                out[pc] += eq[j];
            }
            out
        });
        let got = stage_pushforwards(&stage_cycle_points, &rows, addresses);
        assert_eq!(got, expected);
    }

    fn run_pair(committed_program: bool) {
        with_sample_backend(|backend| {
            let log_t = JoltWitnessOracle::<Fr>::shape(
                backend,
                JoltPolynomialId::Committed(JoltCommittedPolynomial::RdInc),
            )
            .unwrap()
            .rows()
            .ilog2() as usize;
            let (bytecode_d, chunk_bits) =
                probe_one_hot_family(backend, JoltCommittedPolynomial::BytecodeRa, log_t);
            assert!(bytecode_d > 0, "fixture serves no BytecodeRa polynomials");
            let program = backend.program_preprocessing();
            let bytecode_len = program.bytecode.bytecode.len();
            assert!(bytecode_len.is_power_of_two());
            let log_k = bytecode_len.ilog2() as usize;
            let dimensions = BytecodeReadRafDimensions::new(log_t, log_k, bytecode_d);
            let entry_bytecode_index = 1.min(bytecode_len - 1);

            let stage_cycle_points: [Vec<Fr>; 5] =
                std::array::from_fn(|s| synthetic_point(log_t, 11 + s as u64));
            let stage_points = BytecodeStagePoints {
                stage_cycle_points: stage_cycle_points.clone(),
                register_read_write_point: synthetic_point(REGISTER_ADDRESS_BITS + log_t, 31),
                register_val_evaluation_point: synthetic_point(REGISTER_ADDRESS_BITS + log_t, 37),
                // Empty marks the base protocol's five-stage claim set; the
                // fused-inc points exist only on the packed (akita) wire.
                fused_inc_cycle_points: Vec::new(),
            };

            // ---- Stage 6a: address phase.
            let address_relation = BytecodeReadRafAddressPhase::new(
                dimensions,
                committed_program,
                stage_points.clone(),
                entry_bytecode_index,
            );
            let address_challenges = BytecodeReadRafAddressPhaseChallenges {
                gamma: fr(3),
                stage1_gamma: fr(5),
                stage2_gamma: fr(7),
                stage3_gamma: fr(11),
                stage4_gamma: fr(13),
                stage5_gamma: fr(17),
            };
            let address_claims = BytecodeReadRafAddressPhaseInputClaims::<Fr>::default();
            let address_input_points = BytecodeReadRafAddressPhaseInputClaims::<Vec<Fr>>::default();

            let mut session = ProofSession::default();
            let mut reference =
                <ReferenceBackend as PrepareKernel<Fr, BytecodeReadRafAddressPhase<Fr>>>::prepare(
                    &ReferenceBackend,
                    &mut session,
                    backend,
                    ProverInputs {
                        relation: &address_relation,
                        claims: &address_claims,
                        points: &address_input_points,
                        challenges: &address_challenges,
                    },
                )
                .unwrap();
            let mut optimized = OptimizedBytecodeReadRafAddress
                .prepare(
                    &mut session,
                    backend,
                    ProverInputs {
                        relation: &address_relation,
                        claims: &address_claims,
                        points: &address_input_points,
                        challenges: &address_challenges,
                    },
                )
                .unwrap();
            assert!(
                session.state::<PcRowsKey>().is_some(),
                "the address phase must park the shared PC scan"
            );

            let claim = probe_input_claim(reference.as_mut());
            let address_sumcheck_challenges = synthetic_point(log_k, 101);
            run_lockstep(
                reference.as_mut(),
                optimized.as_mut(),
                claim,
                &address_sumcheck_challenges,
            );
            let reference_outputs = reference.output_claims(&address_claims).unwrap();
            let optimized_outputs = optimized.output_claims(&address_claims).unwrap();
            assert_eq!(reference_outputs, optimized_outputs);
            assert_eq!(
                reference_outputs.val_stages.len(),
                if committed_program { 5 } else { 0 }
            );

            // ---- Stage 6b: cycle phase, from the same session (the parked
            // PC scan is reclaimed, mirroring the production 6a→6b flow) and
            // the production r_address wiring (the reversed 6a point).
            let r_address: Vec<Fr> = address_sumcheck_challenges.iter().rev().copied().collect();
            let stage_gammas = address_challenges.stage_gamma_powers();
            let cycle_relation = BytecodeReadRafCycle::full(BytecodeReadRafCycleInputs {
                dimensions,
                r_address,
                stage_cycle_points: stage_cycle_points.clone(),
                entry_bytecode_index,
                committed_chunk_bits: chunk_bits,
                table_fold: Some(BytecodeReadRafTableFoldInputs {
                    bytecode: &program.bytecode.bytecode,
                    register_read_write_point: &stage_points.register_read_write_point
                        [..REGISTER_ADDRESS_BITS],
                    register_val_evaluation_point: &stage_points.register_val_evaluation_point
                        [..REGISTER_ADDRESS_BITS],
                    stage_gammas: std::array::from_fn(|s| stage_gammas[s].as_slice()),
                }),
            })
            .unwrap();
            let cycle_challenges = BytecodeReadRafCyclePhaseCommittedChallenges { gamma: fr(19) };
            let cycle_claims = BytecodeReadRafInputClaims::<Fr>::default();
            let cycle_input_points = BytecodeReadRafInputClaims::<Vec<Fr>>::default();

            let mut reference =
                <ReferenceBackend as PrepareKernel<Fr, BytecodeReadRafCycle<Fr>>>::prepare(
                    &ReferenceBackend,
                    &mut session,
                    backend,
                    ProverInputs {
                        relation: &cycle_relation,
                        claims: &cycle_claims,
                        points: &cycle_input_points,
                        challenges: &cycle_challenges,
                    },
                )
                .unwrap();
            let mut optimized = OptimizedBytecodeReadRafCycle
                .prepare(
                    &mut session,
                    backend,
                    ProverInputs {
                        relation: &cycle_relation,
                        claims: &cycle_claims,
                        points: &cycle_input_points,
                        challenges: &cycle_challenges,
                    },
                )
                .unwrap();

            let claim = probe_input_claim(reference.as_mut());
            let cycle_sumcheck_challenges = synthetic_point(log_t, 211);
            run_lockstep(
                reference.as_mut(),
                optimized.as_mut(),
                claim,
                &cycle_sumcheck_challenges,
            );
            assert_eq!(
                reference.output_claims(&cycle_claims).unwrap(),
                optimized.output_claims(&cycle_claims).unwrap()
            );
        });
    }

    #[test]
    fn bytecode_phases_match_reference_through_a_shared_session() {
        run_pair(false);
    }

    #[test]
    fn bytecode_phases_match_reference_in_committed_program_mode() {
        run_pair(true);
    }
}
