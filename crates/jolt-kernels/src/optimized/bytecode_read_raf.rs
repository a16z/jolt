//! Optimized bytecode read+RAF kernels: the stage-6a address phase and the
//! stage-6b cycle phase, byte-parity twins of the reference kernels in
//! [`crate::reference::bytecode_read_raf`].
//!
//! Carries forward three optimizations from the former bytecode read/RAF prover:
//!
//! - **Split-eq two-table pushforwards** (address phase): the per-stage
//!   `F_s(k) = Σ_{j: pc(j)=k} eq(r_cycle_s, j)` tables are accumulated as
//!   `Σ_{j_hi} E_hi_s[j_hi] · (Σ_{j_lo: pc=k} E_lo_s[j_lo])` — the inner sums
//!   are additions only and the base stages share one trace walk, so the eq
//!   tables cost `O(√T)` each instead of `O(T)` and the `O(T)` walk pays one
//!   PC lookup per cycle for all stages. In the Akita protocol, four more
//!   pushforwards use the same walk with the fused-increment row weight
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
//! - **Lazy packed fused increment** (cycle phase, Akita): the extra stages
//!   remain `Π_i ra_i · FusedInc · C_fused`, preserving the relation's extra
//!   degree. `FusedInc` binds from shared sign-magnitude rows for three
//!   rounds and materializes at `T / 16` instead of allocating a dense field
//!   column at prepare.
//! - **Eval-at-1 recovery** and **rayon walks**, both phases (see the module
//!   docs on [`crate::optimized`]).
//!
//! Both phases read the stage-5 `InstructionCycleRow` carry their Booleanity
//! and RA-virtualization peers already retain, so neither protocol pays for a
//! second per-cycle PC cache. Both phases read the same total `BytecodePc`
//! column: the pushforward slot and the committed one-hot hot index are the
//! same value on every row.

mod streaming;

use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

use jolt_claims::protocols::jolt::geometry::bytecode::{
    self, read_raf_stage_values, BytecodeReadRafStageValueInputs,
};
use jolt_claims::protocols::jolt::geometry::claim_reductions::bytecode::NUM_BYTECODE_VAL_STAGES;
use jolt_claims::protocols::jolt::geometry::dimensions::{
    committed_address_chunks, REGISTER_ADDRESS_BITS,
};
use jolt_claims::protocols::jolt::JoltOpeningId;
use jolt_claims::OutputClaims;
use jolt_field::{Accumulator, JoltField};
#[cfg(feature = "akita")]
use jolt_poly::bind_prefix_in_place;
use jolt_poly::{
    EqPolynomial, IdentityPolynomial, MultilinearEvaluation, Polynomial, UnivariatePoly,
};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::{
    ConcreteSumcheck, SumcheckInputClaims, SumcheckOutputClaims,
};
use jolt_verifier::stages::stage6a::bytecode_read_raf::{
    BytecodeReadRafAddressPhase, BytecodeReadRafAddressPhaseOutputClaims,
};
use jolt_verifier::stages::stage6b::bytecode_read_raf::BytecodeReadRafCycle;
use jolt_witness::witnesses::RaChunkSelector;
use jolt_witness::{JoltWitnessPlane, WitnessError};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::constant_suffix::ConstantSuffix;
use super::eq_prefix::EqPrefix;
use super::instruction_read_raf::InstructionCycleRow;
use super::lazy_ra::{ChunkIndexSource, LazyFoldedRa};
use super::streaming::StreamedFoldedRa;
use super::support::{
    bind_all, eq_table, gamma_powers, map_reduce_chunks, merge_evals, pair, par_sum_pair_groups,
    par_sum_pair_groups_reusing, repeat_hash, repeats_often, round_poly_from_skipped_evals,
    RepeatCache, RoundProgress,
};
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

use streaming::{rows_unavailable, BytecodePcDigits};
pub use streaming::{StreamingBytecodeReadRafAddress, StreamingBytecodeReadRafCycle};

/// `row_at` serves the cycles up to `prefix_len`, whose row every later cycle
/// of the domain repeats.
fn stage_pushforwards<F: JoltField, R>(
    base_cycle_points: &[Vec<F>],
    weighted_cycle_points: &[Vec<F>],
    row_at: impl Fn(usize) -> Result<R, WitnessError> + Sync,
    prefix_len: usize,
    addresses: usize,
    pc: impl Fn(&R) -> usize + Sync,
    row_weight: impl Fn(&R) -> F + Sync,
) -> Result<Vec<Vec<F>>, WitnessError> {
    let Some(first_stage) = base_cycle_points
        .first()
        .or_else(|| weighted_cycle_points.first())
    else {
        return Ok(Vec::new());
    };
    let log_t = first_stage.len();
    let base_stages = base_cycle_points.len();
    let num_stages = base_stages + weighted_cycle_points.len();
    let finish = |mut outputs: Vec<Vec<F>>| -> Result<Vec<Vec<F>>, WitnessError> {
        let tail = row_at(prefix_len)?;
        let address = pc(&tail);
        for (stage, point) in base_cycle_points
            .iter()
            .chain(weighted_cycle_points)
            .enumerate()
        {
            let weight = if stage < base_stages {
                F::one()
            } else {
                row_weight(&tail)
            };
            outputs[stage][address] += EqPolynomial::suffix_sum(point, prefix_len) * weight;
        }
        Ok(outputs)
    };
    let lo_bits = log_t / 2;
    let hi_bits = log_t - lo_bits;
    let in_len = 1usize << lo_bits;

    // Big-endian points split as eq(r, j) = eq(r[..hi], j_hi) · eq(r[hi..], j_lo)
    // with j = (j_hi << lo_bits) | j_lo.
    let e_hi = base_cycle_points
        .iter()
        .chain(weighted_cycle_points)
        .map(|point| eq_table(&point[..hi_bits]))
        .collect::<Vec<_>>();
    let e_lo = base_cycle_points
        .iter()
        .chain(weighted_cycle_points)
        .map(|point| eq_table(&point[hi_bits..]))
        .collect::<Vec<_>>();

    // Routing and every worker accumulator are bounded independently of the bytecode
    // domain. Splitting a high-eq block is valid: its multiplier distributes over
    // the low-eq sums from each batch and fragment.
    const TILE_ADDRESSES: usize = 1024;
    const BATCH_ROWS: usize = 1 << 18;
    const FRAGMENT_ROWS: usize = 4096;

    // Small domains avoid routing overhead. Their direct accumulators share a
    // fixed per-call budget, so adding Rayon workers cannot multiply scratch
    // without bound even when the whole domain fits in one direct tile.
    const DIRECT_ADDRESSES: usize = 1 << 16;
    const SCRATCH_BYTES: usize = 256 << 20;
    let scratch_per_job = num_stages
        .saturating_mul(addresses)
        .saturating_mul(std::mem::size_of::<F>())
        .saturating_mul(2)
        .saturating_add(addresses)
        .saturating_add(addresses.min(in_len) * std::mem::size_of::<usize>())
        .saturating_add(
            num_stages
                .saturating_mul(2)
                .saturating_mul(std::mem::size_of::<Vec<F>>()),
        );
    if addresses <= DIRECT_ADDRESSES && scratch_per_job <= SCRATCH_BYTES {
        struct Tile<F> {
            partial: Vec<Vec<F>>,
            inner: Vec<Vec<F>>,
            seen: Vec<bool>,
            touched: Vec<usize>,
        }
        let tile = || Tile {
            partial: (0..num_stages)
                .map(|_| vec![F::zero(); addresses])
                .collect(),
            inner: (0..num_stages)
                .map(|_| vec![F::zero(); addresses])
                .collect(),
            seen: vec![false; addresses],
            touched: Vec::with_capacity(addresses.min(in_len)),
        };
        let accumulate = |mut tile: Tile<F>, hi: usize| -> Result<Tile<F>, WitnessError> {
            for address in tile.touched.drain(..) {
                for inner in &mut tile.inner {
                    inner[address] = F::zero();
                }
                tile.seen[address] = false;
            }
            for lo in 0..in_len.min(prefix_len - hi * in_len) {
                let row = row_at(hi * in_len + lo)?;
                let address = pc(&row);
                if !tile.seen[address] {
                    tile.seen[address] = true;
                    tile.touched.push(address);
                }
                let (base, weighted) = tile.inner.split_at_mut(base_stages);
                for (inner, eq) in base.iter_mut().zip(&e_lo[..base_stages]) {
                    inner[address] += eq[lo];
                }
                if !weighted.is_empty() {
                    let weight = row_weight(&row);
                    for (inner, eq) in weighted.iter_mut().zip(&e_lo[base_stages..]) {
                        inner[address] += eq[lo] * weight;
                    }
                }
            }
            for &address in &tile.touched {
                for ((partial, inner), eq) in tile.partial.iter_mut().zip(&tile.inner).zip(&e_hi) {
                    partial[address] += inner[address] * eq[hi];
                }
            }
            Ok(tile)
        };
        #[cfg(feature = "parallel")]
        {
            let merge = |mut left: Vec<Vec<F>>, right: Vec<Vec<F>>| {
                for (left, right) in left.iter_mut().zip(right) {
                    for (left, right) in left.iter_mut().zip(right) {
                        *left += right;
                    }
                }
                left
            };
            let out_len = prefix_len.div_ceil(in_len);
            let jobs = rayon::current_num_threads()
                .min(out_len.max(1))
                .min(SCRATCH_BYTES / scratch_per_job);
            let chunk = out_len.div_ceil(jobs).max(1);
            return finish(
                (0..out_len)
                    .into_par_iter()
                    .step_by(chunk)
                    .map(|start| {
                        (start..(start + chunk).min(out_len))
                            .try_fold(tile(), accumulate)
                            .map(|tile| tile.partial)
                    })
                    .try_reduce_with(|left, right| Ok(merge(left, right)))
                    .transpose()?
                    .unwrap_or_else(|| tile().partial),
            );
        }
        #[cfg(not(feature = "parallel"))]
        return finish(
            (0..prefix_len.div_ceil(in_len))
                .try_fold(tile(), accumulate)?
                .partial,
        );
    }

    struct RoutedRow<F> {
        cycle: usize,
        address: usize,
        weight: F,
    }

    let mut outputs = (0..num_stages)
        .map(|_| vec![F::zero(); addresses])
        .collect::<Vec<_>>();
    for batch in (0..prefix_len).step_by(BATCH_ROWS) {
        let mut tiles: HashMap<usize, Vec<RoutedRow<F>>> = HashMap::new();
        for cycle in batch..(batch + BATCH_ROWS).min(prefix_len) {
            let row = row_at(cycle)?;
            let address = pc(&row);
            tiles
                .entry(address / TILE_ADDRESSES)
                .or_default()
                .push(RoutedRow {
                    cycle,
                    address: address % TILE_ADDRESSES,
                    weight: if weighted_cycle_points.is_empty() {
                        F::zero()
                    } else {
                        row_weight(&row)
                    },
                });
        }
        let mut tiles = tiles.into_iter().collect::<Vec<_>>();
        tiles.sort_unstable_by_key(|(tile, _)| *tile);

        let accumulate_stage = |(stage, output): (usize, &mut Vec<F>)| {
            let mut remaining = output.as_mut_slice();
            let mut next_address = 0;
            let mut jobs = Vec::with_capacity(tiles.len());
            for (tile, entries) in &tiles {
                let start = tile * TILE_ADDRESSES;
                let width = TILE_ADDRESSES.min(addresses - start);
                let (_, tail) = remaining.split_at_mut(start - next_address);
                let (target, tail) = tail.split_at_mut(width);
                remaining = tail;
                next_address = start + width;
                jobs.push((target, entries.as_slice()));
            }
            let accumulate_tile = |(target, entries): (&mut [F], &[RoutedRow<F>])| {
                let width = target.len();
                struct TileAccumulator<F> {
                    partial: Vec<F>,
                    inner: Vec<F>,
                    seen: Vec<bool>,
                    touched: Vec<usize>,
                }
                let accumulator = || TileAccumulator {
                    partial: vec![F::zero(); width],
                    inner: vec![F::zero(); width],
                    seen: vec![false; width],
                    touched: Vec::with_capacity(width.min(entries.len())),
                };
                impl<F: JoltField> TileAccumulator<F> {
                    fn flush(&mut self, high_eq: F) {
                        for address in self.touched.drain(..) {
                            self.partial[address] += high_eq * self.inner[address];
                            self.inner[address] = F::zero();
                            self.seen[address] = false;
                        }
                    }
                }
                let fragment = |mut scratch: TileAccumulator<F>, entries: &[RoutedRow<F>]| {
                    let mut current_hi = entries[0].cycle / in_len;
                    for row in entries {
                        let hi = row.cycle / in_len;
                        if hi != current_hi {
                            scratch.flush(e_hi[stage][current_hi]);
                            current_hi = hi;
                        }
                        if !scratch.seen[row.address] {
                            scratch.seen[row.address] = true;
                            scratch.touched.push(row.address);
                        }
                        let value = e_lo[stage][row.cycle % in_len];
                        scratch.inner[row.address] += if stage < base_stages {
                            value
                        } else {
                            value * row.weight
                        };
                    }
                    scratch.flush(e_hi[stage][current_hi]);
                    scratch
                };
                #[cfg(feature = "parallel")]
                let merge = |mut left: Vec<F>, right: Vec<F>| {
                    for (left, right) in left.iter_mut().zip(right) {
                        *left += right;
                    }
                    left
                };
                #[cfg(feature = "parallel")]
                let partial = entries
                    .par_chunks(FRAGMENT_ROWS)
                    .fold(accumulator, fragment)
                    .map(|scratch| scratch.partial)
                    .reduce_with(merge);
                #[cfg(not(feature = "parallel"))]
                let partial = Some(
                    entries
                        .chunks(FRAGMENT_ROWS)
                        .fold(accumulator(), fragment)
                        .partial,
                );
                if let Some(partial) = partial {
                    for (output, value) in target.iter_mut().zip(partial) {
                        *output += value;
                    }
                }
            };
            #[cfg(feature = "parallel")]
            jobs.into_par_iter().for_each(accumulate_tile);
            #[cfg(not(feature = "parallel"))]
            jobs.into_iter().for_each(accumulate_tile);
        };
        #[cfg(feature = "parallel")]
        outputs
            .par_iter_mut()
            .enumerate()
            .for_each(accumulate_stage);
        #[cfg(not(feature = "parallel"))]
        outputs.iter_mut().enumerate().for_each(accumulate_stage);
    }
    finish(outputs)
}

/// Stage-6a address phase: `PrepareKernel` front of the optimized kernel.
pub struct OptimizedBytecodeReadRafAddress;

impl<F: JoltField> PrepareKernel<F, BytecodeReadRafAddressPhase<F>>
    for OptimizedBytecodeReadRafAddress
{
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, BytecodeReadRafAddressPhase<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = BytecodeReadRafAddressPhase<F>>>, KernelError<F>>
    {
        let dimensions = inputs.relation.dimensions();
        let addresses = 1usize << dimensions.log_k();
        let cycles = 1usize << dimensions.log_t();
        let rows = InstructionCycleRow::shared_for_next_consumer(session, witness, cycles)?;
        let representatives = rows.representatives();
        let push_pc = InstructionCycleRow::bytecode_pc;
        if representatives.iter().any(|row| push_pc(row) >= addresses) {
            return Err(KernelError::InvariantViolation {
                reason: "bytecode index outside the padded bytecode domain",
            });
        }
        #[cfg(not(feature = "akita"))]
        let row_weight = |_: &InstructionCycleRow| F::one();
        #[cfg(feature = "akita")]
        let row_weight = InstructionCycleRow::fused_inc::<F>;
        Ok(Box::new(AddressKernel::new(
            witness,
            &inputs,
            |cycle| Ok(representatives[cycle]),
            rows.prefix_len(),
            push_pc,
            row_weight,
        )?))
    }
}

impl<F: JoltField> AddressKernel<F> {
    /// `row_at` serves the cycles up to `prefix_len`, whose row every later
    /// cycle of the domain repeats. `pc` is a row's bytecode index, below the
    /// relation's address count, and `row_weight` its fused increment.
    fn new<R>(
        witness: &dyn JoltWitnessPlane<F>,
        inputs: &ProverInputs<'_, F, BytecodeReadRafAddressPhase<F>>,
        row_at: impl Fn(usize) -> Result<R, WitnessError> + Sync,
        prefix_len: usize,
        pc: impl Fn(&R) -> usize + Sync,
        row_weight: impl Fn(&R) -> F + Sync,
    ) -> Result<Self, KernelError<F>> {
        let relation = inputs.relation;
        let dimensions = relation.dimensions();
        let addresses = 1usize << dimensions.log_k();

        let program = witness.program_preprocessing();
        if program.bytecode.bytecode.len() != addresses {
            return Err(KernelError::TableSizeMismatch {
                table: "bytecode stage values".to_owned(),
                expected: addresses,
                got: program.bytecode.bytecode.len(),
            });
        }
        let stage_gammas = inputs.challenges.stage_gamma_powers();
        let bytecode_rows = &program.bytecode.bytecode;
        let stage_values = read_raf_stage_values(BytecodeReadRafStageValueInputs {
            bytecode: bytecode_rows,
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
        let fused_cycle_points = relation.fused_inc_cycle_points();
        for point in stage_cycle_points.iter().chain(fused_cycle_points) {
            if point.len() != dimensions.log_t() {
                return Err(KernelError::InvariantViolation {
                    reason: "bytecode stage cycle point has the wrong variable count",
                });
            }
        }
        let entry_bytecode_index = relation.entry_bytecode_index();
        if entry_bytecode_index >= addresses {
            return Err(KernelError::InvariantViolation {
                reason: "bytecode index outside the padded bytecode domain",
            });
        }

        let base_stages = stage_cycle_points.len();
        let num_stages = base_stages + fused_cycle_points.len();
        let gamma_powers = gamma_powers(inputs.challenges.gamma, num_stages + 3);

        // The field-inline extension's fold geometry: the field-register row values under
        // the extended per-stage gamma powers, each leg over its own cycle binding (see
        // the reference kernel's `FieldInlineAddressLegs`).
        #[cfg(feature = "field-inline")]
        let (field_inline_values, field_read_write_cycle, field_val_evaluation_cycle) = {
            use jolt_claims::protocols::field_inline::geometry::bytecode as field_inline_bytecode;
            use jolt_claims::protocols::field_inline::FIELD_REGISTERS_LOG_K;

            let geometry = relation.field_inline_geometry()?;
            if geometry.read_write_point.len() != FIELD_REGISTERS_LOG_K + dimensions.log_t()
                || geometry.val_evaluation_point.len() != FIELD_REGISTERS_LOG_K + dimensions.log_t()
            {
                return Err(KernelError::InvariantViolation {
                    reason: "field-inline opening point has the wrong variable count",
                });
            }
            let (read_write_address, read_write_cycle) =
                geometry.read_write_point.split_at(FIELD_REGISTERS_LOG_K);
            let (val_evaluation_address, val_evaluation_cycle) = geometry
                .val_evaluation_point
                .split_at(FIELD_REGISTERS_LOG_K);
            let gammas =
                jolt_verifier::stages::field_inline_bytecode::field_inline_stage_gamma_powers(
                    inputs.challenges,
                );
            let field_rows = field_inline_bytecode::read_raf_stage_values(
                field_inline_bytecode::FieldInlineBytecodeReadRafStageValueInputs {
                    bytecode: &program.bytecode.bytecode,
                    field_register_read_write_point: read_write_address,
                    field_register_val_evaluation_point: val_evaluation_address,
                    stage4_gammas: &gammas.stage4,
                    stage5_gammas: &gammas.stage5,
                },
            );
            let column =
                |s: usize| Polynomial::new(field_rows.iter().map(|row| row[s]).collect::<Vec<F>>());
            (
                [column(3), column(4)],
                read_write_cycle.to_vec(),
                val_evaluation_cycle.to_vec(),
            )
        };

        #[cfg(feature = "field-inline")]
        let active_legs = field_inline_values
            .into_iter()
            .zip([field_read_write_cycle, field_val_evaluation_cycle])
            .zip([gamma_powers[3], gamma_powers[4]])
            .filter_map(|((values, point), weight)| {
                values
                    .evals()
                    .iter()
                    .any(|value| !value.is_zero())
                    .then_some((values, point, weight))
            })
            .collect::<Vec<_>>();
        #[cfg(feature = "field-inline")]
        let composed_points: Vec<Vec<F>> = stage_cycle_points
            .iter()
            .cloned()
            .chain(active_legs.iter().map(|(_, point, _)| point.clone()))
            .collect();
        #[cfg(feature = "field-inline")]
        let walk_points: &[Vec<F>] = &composed_points;
        #[cfg(not(feature = "field-inline"))]
        let walk_points: &[Vec<F>] = stage_cycle_points;
        #[cfg_attr(not(feature = "field-inline"), expect(unused_mut))]
        let mut pushforwards = stage_pushforwards(
            walk_points,
            fused_cycle_points,
            &row_at,
            prefix_len,
            addresses,
            &pc,
            row_weight,
        )?;
        #[cfg(feature = "field-inline")]
        let field_inline = FieldInlineAddressLegs {
            legs: pushforwards
                .drain(base_stages..base_stages + active_legs.len())
                .zip(active_legs)
                .map(|(pushforward, (values, _, weight))| FieldInlineAddressLeg {
                    weight,
                    pushforward: Polynomial::new(pushforward),
                    values,
                })
                .collect(),
        };
        let pushforwards = pushforwards
            .into_iter()
            .map(Polynomial::new)
            .collect::<Vec<_>>();
        let values = (0..NUM_BYTECODE_VAL_STAGES)
            .map(|stage| Polynomial::new(stage_values.iter().map(|row| row[stage]).collect()))
            .collect::<Vec<_>>();
        let mut stage_values = (0..base_stages).map(StageVal::Table).collect::<Vec<_>>();
        if !fused_cycle_points.is_empty() {
            if fused_cycle_points.len() != bytecode::LATTICE_FUSED_INC_STAGES
                || NUM_BYTECODE_VAL_STAGES != base_stages + 1
            {
                return Err(KernelError::InvariantViolation {
                    reason: "packed bytecode read-raf stage shape is inconsistent",
                });
            }
            stage_values.extend([
                StageVal::Table(base_stages),
                StageVal::Table(base_stages),
                StageVal::Complement(base_stages),
                StageVal::Complement(base_stages),
            ]);
        }
        let mut raf_weights = vec![F::zero(); num_stages];
        raf_weights[0] = gamma_powers[num_stages];
        raf_weights[2] = gamma_powers[num_stages - 1];
        let int_table = Polynomial::new((0..addresses).map(|k| F::from_u64(k as u64)).collect());
        let one_hot = |index: usize| {
            let mut table = vec![F::zero(); addresses];
            table[index] = F::one();
            Polynomial::new(table)
        };

        Ok(Self {
            progress: RoundProgress::new(relation.rounds()),
            committed_program: relation.committed_program(),
            stage_weights: gamma_powers[..num_stages].to_vec(),
            entry_weight: gamma_powers[num_stages + 2],
            raf_weights,
            pushforwards,
            values,
            stage_values,
            int_table,
            entry_trace: one_hot(pc(&row_at(0)?)),
            entry_expected: one_hot(entry_bytecode_index),
            #[cfg(feature = "field-inline")]
            field_inline,
        })
    }
}

/// The two field-register access terms at the stage-4/5 cycle points and γ³/γ⁴
/// weights, sharing the ordinary bytecode address domain.
#[cfg(feature = "field-inline")]
#[cfg_attr(
    feature = "allocative",
    derive(allocative::Allocative),
    allocative(bound = "F: JoltField")
)]
struct FieldInlineAddressLegs<F: JoltField> {
    legs: Vec<FieldInlineAddressLeg<F>>,
}

#[cfg(feature = "field-inline")]
#[cfg_attr(
    feature = "allocative",
    derive(allocative::Allocative),
    allocative(bound = "F: JoltField")
)]
struct FieldInlineAddressLeg<F: JoltField> {
    weight: F,
    pushforward: Polynomial<F>,
    values: Polynomial<F>,
}

#[cfg(feature = "field-inline")]
impl<F: JoltField> FieldInlineAddressLegs<F> {
    fn bind(&mut self, challenge: F) {
        for leg in &mut self.legs {
            bind_all([&mut leg.pushforward, &mut leg.values], challenge);
        }
    }

    #[inline]
    fn group_evals(&self, y: usize) -> [F; 2] {
        let mut out = [F::zero(); 2];
        for leg in &self.legs {
            let (f_lo, f_hi) = pair(&leg.pushforward, y);
            let (v_lo, v_hi) = pair(&leg.values, y);
            out[0] += leg.weight * f_lo * v_lo;
            out[1] += leg.weight * (f_hi + f_hi - f_lo) * (v_hi + v_hi - v_lo);
        }
        out
    }

    fn bound_term(&self) -> F {
        self.legs
            .iter()
            .map(|leg| leg.weight * leg.pushforward.evals()[0] * leg.values.evals()[0])
            .sum()
    }
}

#[derive(Clone, Copy)]
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
enum StageVal {
    Table(usize),
    Complement(usize),
}

#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
struct AddressKernel<F: JoltField> {
    progress: RoundProgress,
    committed_program: bool,
    stage_weights: Vec<F>,
    #[cfg_attr(feature = "allocative", allocative(skip))]
    entry_weight: F,
    /// Within-stage RAF `Int` weights, divided by the stage batching weight.
    raf_weights: Vec<F>,
    pushforwards: Vec<Polynomial<F>>,
    /// RAW stage-value tables — the RAF identity binds separately so
    /// committed mode can stage the raw bound `Val_s` wire claims.
    values: Vec<Polynomial<F>>,
    stage_values: Vec<StageVal>,
    int_table: Polynomial<F>,
    entry_trace: Polynomial<F>,
    entry_expected: Polynomial<F>,
    #[cfg(feature = "field-inline")]
    field_inline: FieldInlineAddressLegs<F>,
}
impl<F: JoltField> AddressKernel<F> {
    #[inline]
    fn stage_pair(&self, stage: usize, y: usize) -> (F, F) {
        match self.stage_values[stage] {
            StageVal::Table(index) => pair(&self.values[index], y),
            StageVal::Complement(index) => {
                let (lo, hi) = pair(&self.values[index], y);
                (F::one() - lo, F::one() - hi)
            }
        }
    }

    fn bound_stage_value(&self, stage: usize) -> F {
        match self.stage_values[stage] {
            StageVal::Table(index) => self.values[index].evals()[0],
            StageVal::Complement(index) => F::one() - self.values[index].evals()[0],
        }
    }

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
        #[cfg(feature = "field-inline")]
        self.field_inline.bind(challenge);
        self.progress.advance();
    }

    #[inline]
    fn group_evals(&self, y: usize) -> [F; 2] {
        let (int_lo, int_hi) = pair(&self.int_table, y);
        let int_delta = int_hi - int_lo;
        let int_at_2 = int_hi + int_delta;
        let mut out = [F::zero(); 2];
        for s in 0..self.stage_values.len() {
            let (f_lo, f_hi) = pair(&self.pushforwards[s], y);
            let (v_lo, v_hi) = self.stage_pair(s, y);
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
        #[cfg(feature = "field-inline")]
        {
            let legs = self.field_inline.group_evals(y);
            out[0] += legs[0];
            out[1] += legs[1];
        }
        out
    }
}

impl<F: JoltField> ProveRounds<F> for AddressKernel<F> {
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

impl<F: JoltField> SumcheckKernel<F> for AddressKernel<F> {
    type Relation = BytecodeReadRafAddressPhase<F>;

    #[cfg_attr(
        not(feature = "field-inline"),
        expect(
            clippy::useless_conversion,
            reason = "field-inline selects composed claims and opening ids"
        )
    )]
    fn output_claims(
        &mut self,
        _inputs: &SumcheckInputClaims<F, Self::Relation>,
    ) -> Result<SumcheckOutputClaims<F, BytecodeReadRafAddressPhase<F>>, SumcheckKernelError<F>>
    {
        self.progress.require_complete()?;
        let mut intermediate =
            self.entry_weight * self.entry_trace.evals()[0] * self.entry_expected.evals()[0];
        let bound_int = self.int_table.evals()[0];
        for s in 0..self.stage_values.len() {
            intermediate += self.stage_weights[s]
                * self.pushforwards[s].evals()[0]
                * (self.bound_stage_value(s) + self.raf_weights[s] * bound_int);
        }
        #[cfg(feature = "field-inline")]
        {
            intermediate += self.field_inline.bound_term();
        }
        let val_stages = if self.committed_program {
            self.values.iter().map(|table| table.evals()[0]).collect()
        } else {
            Vec::new()
        };
        Ok(BytecodeReadRafAddressPhaseOutputClaims {
            intermediate,
            val_stages,
        }
        .into())
    }
}

/// The packed fused-increment cycle column, bound lazily while the bytecode
/// RA factors still retain their shared compact rows. The fourth bind
/// materializes only the nonconstant prefix divided by 16 and releases this handle.
#[cfg(feature = "akita")]
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
enum LazyFusedInc<F: JoltField> {
    Lazy {
        branch_weights: Vec<F>,
        rows: Arc<ConstantSuffix<InstructionCycleRow>>,
        prefix_len: usize,
        tail: F,
    },
    Prefix {
        values: Vec<F>,
        tail: F,
    },
}

#[cfg(feature = "akita")]
impl<F: JoltField> LazyFusedInc<F> {
    fn new(rows: Arc<ConstantSuffix<InstructionCycleRow>>, prefix_len: usize) -> Self {
        let tail = rows[rows.len() - 1].fused_inc::<F>();
        Self::Lazy {
            branch_weights: vec![F::one()],
            rows,
            prefix_len,
            tail,
        }
    }

    fn tail_value(&self) -> F {
        match self {
            Self::Lazy { tail, .. } | Self::Prefix { tail, .. } => *tail,
        }
    }

    #[inline]
    fn value(&self, index: usize) -> F {
        match self {
            Self::Lazy {
                branch_weights,
                rows,
                ..
            } => branch_weights
                .iter()
                .enumerate()
                .fold(F::zero(), |acc, (offset, weight)| {
                    acc + *weight * rows[index * branch_weights.len() + offset].fused_inc::<F>()
                }),
            Self::Prefix { values, tail } => values.get(index).copied().unwrap_or(*tail),
        }
    }

    #[inline]
    fn lo_hi(&self, row: usize) -> (F, F) {
        (self.value(2 * row), self.value(2 * row + 1))
    }

    fn bind(&mut self, challenge: F) {
        match self {
            Self::Lazy {
                branch_weights,
                rows,
                prefix_len,
                tail,
            } => {
                let one_minus = F::one() - challenge;
                let mut next = Vec::with_capacity(2 * branch_weights.len());
                next.extend(branch_weights.iter().map(|weight| one_minus * *weight));
                next.extend(branch_weights.iter().map(|weight| challenge * *weight));
                if next.len() < 16 {
                    *branch_weights = next;
                } else {
                    let len = prefix_len.div_ceil(next.len());
                    let evaluate = |index: usize| {
                        next.iter()
                            .enumerate()
                            .fold(F::zero(), |acc, (offset, weight)| {
                                acc + *weight * rows[index * next.len() + offset].fused_inc::<F>()
                            })
                    };
                    #[cfg(feature = "parallel")]
                    let evals = (0..len).into_par_iter().map(evaluate).collect();
                    #[cfg(not(feature = "parallel"))]
                    let evals = (0..len).map(evaluate).collect();
                    *self = Self::Prefix {
                        values: evals,
                        tail: *tail,
                    };
                }
            }
            Self::Prefix { values, tail } => {
                let _ = bind_prefix_in_place(values, challenge, *tail);
            }
        }
    }
}

/// Stage-6b cycle phase: `PrepareKernel` front of the optimized kernel.
pub struct OptimizedBytecodeReadRafCycle;

impl<F: JoltField> PrepareKernel<F, BytecodeReadRafCycle<F>> for OptimizedBytecodeReadRafCycle {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, BytecodeReadRafCycle<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = BytecodeReadRafCycle<F>>>, KernelError<F>>
    {
        let cycles = 1usize << inputs.relation.dimensions().log_t();
        Ok(Box::new(CycleKernel::new(
            &inputs,
            |chunk_eqs, selectors| {
                let rows = InstructionCycleRow::shared(session, witness, cycles)?;
                let prefix_len = rows.prefix_len().max(1);
                Ok(CycleRows {
                    #[cfg(feature = "akita")]
                    fused_inc: LazyFusedInc::new(Arc::clone(&rows), prefix_len),
                    ra: CycleRa::Lazy(LazyFoldedRa::new(
                        chunk_eqs,
                        BytecodePcChunks {
                            rows,
                            prefix_len,
                            selectors,
                        },
                    )),
                    prefix_len,
                })
            },
        )?))
    }
}

/// What a cycle kernel holds of the cycle rows.
struct CycleRows<F: JoltField> {
    ra: CycleRa<F>,
    /// The first cycle from which every row repeats, and at least 1: the
    /// coefficient tables are explicit up to it, cycle 0's entry term included.
    prefix_len: usize,
    #[cfg(feature = "akita")]
    fused_inc: LazyFusedInc<F>,
}

impl<F: JoltField> CycleKernel<F> {
    /// `rows` receives the eq table and the selector of each committed
    /// chunk of the address, in polynomial order.
    fn new(
        inputs: &ProverInputs<'_, F, BytecodeReadRafCycle<F>>,
        rows: impl FnOnce(Vec<Vec<F>>, Vec<RaChunkSelector>) -> Result<CycleRows<F>, KernelError<F>>,
    ) -> Result<Self, KernelError<F>> {
        let relation = inputs.relation;
        let dimensions = relation.dimensions();
        let r_address = relation.r_address();
        let stage_cycle_points = relation.stage_cycle_points();
        let num_ra = dimensions.num_committed_ra_polys();

        let chunks = committed_address_chunks(r_address, relation.committed_chunk_bits());
        if chunks.len() != num_ra {
            return Err(KernelError::InvariantViolation {
                reason: "bytecode address chunk count disagrees with the committed RA count",
            });
        }

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
        let CycleRows {
            ra,
            prefix_len,
            #[cfg(feature = "akita")]
            fused_inc,
        } = rows(chunk_eqs, selectors)?;

        // The combined coefficient table: every non-RA factor of the summand is linear
        // in one cycle table, so   C(j) = Σ_s (γ^s·val_s + raf_s·int_r)·eq_s(j) +
        // γ⁷·entry·[j = 0] with raf_0 = γ⁵·int_r, raf_2 = γ⁶·int_r (SpartanOuterRaf
        // rides the stage-1 cycle point, SpartanShiftRaf the stage-3 one). With
        // field-inline enabled, the stage-4/5 field-register legs ride the
        // field-register read-write / val-evaluation cycle sub-points at γ³/γ⁴ (the
        // reference kernel's composed pre-fold, term for term).
        let stage_values = relation.stage_values_at_r_address()?;
        let num_stages = stage_cycle_points.len();
        let base_stages = bytecode::BYTECODE_STAGE_GAMMA_COUNTS.len();
        let gamma_powers = gamma_powers(inputs.challenges.gamma, num_stages + 3);
        let int_at_r_address = IdentityPolynomial::new(r_address.len()).evaluate(r_address);
        let mut stage_weights = (0..base_stages)
            .map(|stage| gamma_powers[stage] * stage_values[stage])
            .collect::<Vec<_>>();
        stage_weights[0] += gamma_powers[num_stages] * int_at_r_address;
        stage_weights[2] += gamma_powers[num_stages + 1] * int_at_r_address;

        let eq_address: Vec<F> = eq_table(r_address);
        #[cfg(feature = "field-inline")]
        let (field_folds, field_read_write_cycle, field_val_evaluation_cycle) = {
            use jolt_claims::protocols::field_inline::FIELD_REGISTERS_LOG_K;

            let fold = relation.field_inline_fold()?;
            if fold.read_write_address.len() != FIELD_REGISTERS_LOG_K
                || fold.val_evaluation_address.len() != FIELD_REGISTERS_LOG_K
                || fold.read_write_cycle.len() != dimensions.log_t()
                || fold.val_evaluation_cycle.len() != dimensions.log_t()
            {
                return Err(KernelError::InvariantViolation {
                    reason: "field-inline bytecode fold points have the wrong variable counts",
                });
            }
            let field_folds = relation.field_inline_stage_values_at_r_address()?;
            (
                field_folds,
                fold.read_write_cycle.clone(),
                fold.val_evaluation_cycle.clone(),
            )
        };

        for point in stage_cycle_points {
            if point.len() != dimensions.log_t() {
                return Err(KernelError::InvariantViolation {
                    reason: "bytecode stage cycle point has the wrong variable count",
                });
            }
        }

        let mut terms: Vec<_> = stage_cycle_points[..base_stages]
            .iter()
            .cloned()
            .zip(stage_weights)
            .collect();
        // The field-inline stage-4/5 legs ride their own cycle sub-points at γ³/γ⁴. A
        // trace without field-inline activity folds both weights to zero (its
        // field-register operands are all absent), so these terms are skipped
        // exactly.
        #[cfg(feature = "field-inline")]
        for (point, weight) in [
            (
                field_read_write_cycle.as_slice(),
                gamma_powers[3] * field_folds[3],
            ),
            (
                field_val_evaluation_cycle.as_slice(),
                gamma_powers[4] * field_folds[4],
            ),
        ] {
            if weight.is_zero() {
                continue;
            }
            terms.push((point.to_vec(), weight));
        }
        let mut combined =
            EqPrefix::new(dimensions.log_t(), prefix_len, std::mem::take(&mut terms));
        let entry_scalar = eq_address[relation.entry_bytecode_index()];
        combined.add_to_first(gamma_powers[num_stages + 2] * entry_scalar);

        #[cfg(feature = "akita")]
        let fused_combined = {
            let store = stage_values[base_stages];
            let terms = (base_stages..num_stages)
                .map(|stage| {
                    let value = if stage < base_stages + 2 {
                        store
                    } else {
                        F::one() - store
                    };
                    (
                        stage_cycle_points[stage].clone(),
                        gamma_powers[stage] * value,
                    )
                })
                .collect();
            EqPrefix::new(dimensions.log_t(), prefix_len, terms)
        };

        let output_openings = bytecode::read_raf_output_openings(dimensions).bytecode_ra;
        if output_openings.len() != num_ra {
            return Err(KernelError::InvariantViolation {
                reason: "bytecode RA output opening count disagrees with the committed RA count",
            });
        }

        Ok(Self {
            progress: RoundProgress::new(relation.rounds()),
            degree: relation.degree(),
            ra,
            combined,
            #[cfg(feature = "akita")]
            fused_inc,
            #[cfg(feature = "akita")]
            fused_combined,
            output_openings,
        })
    }
}

/// The committed RA factors: the default family, lazy over the packed row
/// cache until it turns dense, or the streaming family until the bind that
/// gathers it into that family's dense state.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
enum CycleRa<F: JoltField> {
    Lazy(LazyFoldedRa<F, BytecodePcChunks>),
    Streamed(StreamedFoldedRa<F, BytecodePcDigits>),
}

impl<F: JoltField> CycleRa<F> {
    /// The default family, unless the factors are still streamed.
    fn lazy(&self) -> Option<&LazyFoldedRa<F, BytecodePcChunks>> {
        match self {
            Self::Lazy(ra) => Some(ra),
            Self::Streamed(_) => None,
        }
    }
}

/// Lazy-RA index source: chunk `i` of the per-cycle mapped bytecode PC, off
/// the shared stage-5 rows.
#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
struct BytecodePcChunks {
    rows: Arc<ConstantSuffix<InstructionCycleRow>>,
    prefix_len: usize,
    #[cfg_attr(feature = "allocative", allocative(visit = crate::backend::visit_heap_free_elements))]
    selectors: Vec<RaChunkSelector>,
}

/// Groups per map-reduce task of the four-factor walk and the short window.
const CYCLE_CHUNK: usize = 4096;

/// A window in which repeated PC groups are detected and merged.
#[derive(Clone, Copy)]
struct CycleGrouping {
    /// Distinct groups the merge cache holds, and the sampled block length.
    entries: usize,
    /// Groups per map-reduce task; repeated groups merge only within one task.
    #[cfg(feature = "parallel")]
    batch: usize,
    /// Fewest groups for which this window is sampled at all.
    min_groups: usize,
}

impl CycleGrouping {
    const LONG_ENTRIES: usize = 2048;
    const LONG_BATCH: usize = 16384;

    const SHORT: Self = Self {
        entries: 64,
        #[cfg(feature = "parallel")]
        batch: CYCLE_CHUNK,
        min_groups: 0,
    };
    /// Sampled only when a round fills at least one task. A higher gate keeps
    /// rounds with few tasks off this window, but on SHA-2 those rounds still
    /// win more from merging than they lose to parallelism.
    const LONG: Self = Self {
        entries: Self::LONG_ENTRIES,
        #[cfg(feature = "parallel")]
        batch: Self::LONG_BATCH,
        min_groups: Self::LONG_BATCH,
    };
}

impl BytecodePcChunks {
    /// Every column index is a function of this PC alone, so equal PC tuples
    /// select equal RA factors.
    #[inline]
    fn pc(&self, j: usize) -> usize {
        self.rows[j].bytecode_pc()
    }

    #[inline]
    fn group_hash(&self, y: usize, width: usize) -> u64 {
        repeat_hash((2 * width * y..2 * width * (y + 1)).map(|j| self.pc(j) as u64))
    }

    #[inline]
    fn same_group(&self, left: usize, right: usize, width: usize) -> bool {
        (0..2 * width)
            .all(|offset| self.pc(2 * width * left + offset) == self.pc(2 * width * right + offset))
    }

    /// The first window whose sample repeats often. A long loop can appear
    /// nonrepeating inside the short window.
    fn grouping_plan(&self, groups: usize, width: usize) -> Option<CycleGrouping> {
        [CycleGrouping::SHORT, CycleGrouping::LONG]
            .into_iter()
            .find(|plan| {
                groups >= plan.min_groups
                    && repeats_often(
                        groups,
                        plan.entries,
                        |y| self.group_hash(y, width),
                        |a, b| self.same_group(a, b, width),
                    )
            })
    }
}

impl ChunkIndexSource for BytecodePcChunks {
    fn constant_suffix_start(&self) -> usize {
        self.prefix_len
    }
    fn num_columns(&self) -> usize {
        self.selectors.len()
    }

    fn cycles(&self) -> usize {
        self.rows.len()
    }

    #[inline]
    fn index(&self, i: usize, j: usize) -> Option<usize> {
        Some(self.selectors[i].chunk_usize(self.pc(j)))
    }
}

#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
struct CycleKernel<F: JoltField> {
    progress: RoundProgress,
    degree: usize,
    ra: CycleRa<F>,
    combined: EqPrefix<F>,
    #[cfg(feature = "akita")]
    fused_inc: LazyFusedInc<F>,
    #[cfg(feature = "akita")]
    fused_combined: EqPrefix<F>,
    /// The produced `BytecodeRa` opening ids, in `read_raf_output_openings`
    /// order (index-aligned with `ra`).
    #[cfg_attr(feature = "allocative", allocative(visit = crate::backend::visit_heap_free_elements))]
    output_openings: Vec<JoltOpeningId>,
}
impl<F: JoltField> CycleKernel<F> {
    fn bind(&mut self, challenge: F) -> Result<(), WitnessError> {
        self.combined.bind(challenge);
        #[cfg(feature = "akita")]
        self.fused_combined.bind(challenge);
        match &mut self.ra {
            CycleRa::Lazy(ra) => ra.bind(challenge),
            CycleRa::Streamed(ra) => {
                if let Some(dense) = ra.bind(challenge)? {
                    self.ra = CycleRa::Lazy(dense);
                }
            }
        }
        #[cfg(feature = "akita")]
        self.fused_inc.bind(challenge);
        self.progress.advance();
        Ok(())
    }

    /// The summand's evaluations at `t ∈ {0, 2, 3, .., degree}` summed over
    /// group `y`, written into `acc` (length `degree`). `ra_pairs` is the
    /// caller's per-group scratch: each RA pair is gathered once per group,
    /// then held as the factor's running `(value, step)`, so every factor
    /// advances by one addition per sample point instead of a
    /// `t · (hi − lo)` multiplication.
    #[inline]
    fn accumulate_group(
        &self,
        ra: &LazyFoldedRa<F, BytecodePcChunks>,
        y: usize,
        acc: &mut [F],
        ra_pairs: &mut [(F, F)],
    ) {
        ra.lo_hi_all(y, ra_pairs);
        Self::accumulate_coefficient(self.coefficient(y), acc, ra_pairs);
    }

    /// `[C(0), C(1) - C(0), 2 * quadratic_coefficient(C)]`; each coordinate
    /// is linear in C, so adding these arrays adds coefficient polynomials.
    #[inline]
    fn coefficient(&self, y: usize) -> [F; 3] {
        let (c_lo, c_hi) = self.combined.pair(y);
        #[cfg(not(feature = "akita"))]
        {
            [c_lo, c_hi - c_lo, F::zero()]
        }
        #[cfg(feature = "akita")]
        {
            let (fused_coefficient_lo, fused_coefficient_hi) = self.fused_combined.pair(y);
            let (fused_inc_lo, fused_inc_hi) = self.fused_inc.lo_hi(y);
            let at_zero = c_lo + fused_inc_lo * fused_coefficient_lo;
            let at_one = c_hi + fused_inc_hi * fused_coefficient_hi;
            let leading =
                (fused_inc_hi - fused_inc_lo) * (fused_coefficient_hi - fused_coefficient_lo);
            [at_zero, at_one - at_zero, leading + leading]
        }
    }

    /// Adds the samples of `coefficient`'s polynomial times the factors whose
    /// `(lo, hi)` values `ra_pairs` holds.
    #[inline]
    fn accumulate_coefficient(coefficient: [F; 3], acc: &mut [F], ra_pairs: &mut [(F, F)]) {
        let [coefficient_at_zero, coefficient_delta, _second_difference] = coefficient;
        #[cfg(feature = "akita")]
        let (mut coefficient_delta, second_difference) = (coefficient_delta, _second_difference);
        for (lo, hi) in ra_pairs.iter_mut() {
            *hi -= *lo;
        }
        acc[0] += ra_pairs
            .iter()
            .fold(coefficient_at_zero, |acc, (value, _)| acc * *value);
        let mut coefficient = coefficient_at_zero + coefficient_delta + coefficient_delta;
        #[cfg(feature = "akita")]
        {
            coefficient += second_difference;
            coefficient_delta += second_difference + second_difference;
        }
        for (value, step) in ra_pairs.iter_mut() {
            *value += *step + *step;
        }
        for value in &mut acc[1..] {
            *value += ra_pairs
                .iter()
                .fold(coefficient, |acc, (factor, _)| acc * *factor);
            coefficient += coefficient_delta;
            #[cfg(feature = "akita")]
            {
                coefficient_delta += second_difference;
            }
            for (factor, step) in ra_pairs.iter_mut() {
                *factor += *step;
            }
        }
    }
}

impl<F: JoltField> CycleKernel<F> {
    fn grouped_evals(&self, half: usize) -> Option<Vec<F>> {
        let ra = self.ra.lazy()?;
        let LazyFoldedRa::Lazy { source, width, .. } = ra else {
            return None;
        };
        let width = *width;
        let plan = source.grouping_plan(half, width)?;
        // Equal ordered PC tuples have identical RA factors at this round. Only
        // their coefficient polynomials are added; fused increments may differ.
        // Every collision and batch end flushes the old contribution, and the
        // final drain leaves the cache empty for the worker's next batch. The
        // caches are local to this call, so they cannot survive a binding
        // challenge.
        let batch = |cache: &mut RepeatCache<(usize, [F; 3])>, range: Range<usize>| {
            let mut acc = vec![F::zero(); self.degree];
            let mut pairs = vec![(F::zero(), F::zero()); ra.num_polys()];
            for y in range {
                let entry = cache.entry(source.group_hash(y, width));
                let coefficients = self.coefficient(y);
                if let Some((old, sum)) = entry {
                    if source.same_group(*old, y, width) {
                        for (sum, value) in sum
                            .iter_mut()
                            .zip(coefficients)
                            .take(if cfg!(feature = "akita") { 3 } else { 2 })
                        {
                            *sum += value;
                        }
                        continue;
                    }
                    ra.lo_hi_all(*old, &mut pairs);
                    Self::accumulate_coefficient(*sum, &mut acc, &mut pairs);
                }
                *entry = Some((y, coefficients));
            }
            for (y, coefficients) in cache.drain() {
                ra.lo_hi_all(y, &mut pairs);
                Self::accumulate_coefficient(coefficients, &mut acc, &mut pairs);
            }
            acc
        };
        #[cfg(feature = "parallel")]
        let evals = (0..half.div_ceil(plan.batch))
            .into_par_iter()
            .map_init(
                || RepeatCache::new(plan.entries),
                |cache, c| batch(cache, c * plan.batch..((c + 1) * plan.batch).min(half)),
            )
            .reduce(|| vec![F::zero(); self.degree], merge_evals);
        #[cfg(not(feature = "parallel"))]
        let evals = batch(&mut RepeatCache::new(plan.entries), 0..half);
        Some(evals)
    }
}

impl<F: JoltField> CycleKernel<F> {
    // Pair the four linear RA factors into two quadratics. Their samples at
    // 0, 2, ..., degree need only additions after the initial products; the
    // final coefficient multiplication is accumulated before reduction.
    fn four_ra_evals(&self, half: usize) -> Option<Vec<F>> {
        fn quadratic<F: JoltField>(a: (F, F), b: (F, F)) -> [F; 4] {
            let zero = a.0 * b.0;
            let one = a.1 * b.1;
            let leading = (a.1 - a.0) * (b.1 - b.0);
            let second = leading + leading;
            let delta = one - zero + second;
            [zero, one + delta, delta, second]
        }
        let ra = self.ra.lazy()?;
        if ra.num_polys() != 4 {
            return None;
        }
        Some(map_reduce_chunks(
            half,
            CYCLE_CHUNK,
            |range| {
                let mut acc = vec![F::Accumulator::default(); self.degree];
                let mut pairs = [(F::zero(), F::zero()); 4];
                for y in range {
                    let [at_zero, mut delta, second] = self.coefficient(y);
                    ra.lo_hi_all(y, &mut pairs);
                    let mut coefficient = at_zero + delta + delta + second;
                    delta += second + second;
                    let mut left = quadratic(pairs[0], pairs[1]);
                    let mut right = quadratic(pairs[2], pairs[3]);
                    acc[0].fmadd(left[0] * right[0], at_zero);
                    for lane in &mut acc[1..] {
                        lane.fmadd(left[1] * right[1], coefficient);
                        coefficient += delta;
                        delta += second;
                        left[2] += left[3];
                        left[1] += left[2];
                        right[2] += right[3];
                        right[1] += right[2];
                    }
                }
                acc.into_iter().map(Accumulator::reduce).collect()
            },
            merge_evals,
            || vec![F::zero(); self.degree],
        ))
    }
}

impl<F: JoltField> ProveRounds<F> for CycleKernel<F> {
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
            self.bind(challenge).map_err(rows_unavailable)?;
        }
        let half = self.combined.prefix_len().div_ceil(2);
        let slots = self.degree;

        let (mut evals, ra_tail) = match &self.ra {
            CycleRa::Lazy(ra) => {
                let num_ra = ra.num_polys();
                let evals = self
                    .grouped_evals(half)
                    .or_else(|| self.four_ra_evals(half))
                    .unwrap_or_else(|| {
                        par_sum_pair_groups_reusing(
                            half,
                            slots,
                            || vec![(F::zero(), F::zero()); num_ra],
                            |acc, ra_pairs, y| self.accumulate_group(ra, y, acc, ra_pairs),
                        )
                    });
                (evals, ra.tail_values())
            }
            CycleRa::Streamed(ra) => (
                self.streamed_evals(ra).map_err(rows_unavailable)?,
                ra.tail_values(),
            ),
        };

        let (tail_lo, tail_hi) = self.combined.suffix_pair_sums(half);
        #[cfg(feature = "akita")]
        let (tail_lo, tail_hi) = {
            let (fused_lo, fused_hi) = self.fused_combined.suffix_pair_sums(half);
            let inc = self.fused_inc.tail_value();
            (tail_lo + inc * fused_lo, tail_hi + inc * fused_hi)
        };
        let ra_tail: F = ra_tail.iter().copied().product();
        let tail_lo = tail_lo * ra_tail;
        let tail_step = (tail_hi * ra_tail) - tail_lo;
        for (slot, value) in evals.iter_mut().enumerate() {
            let t = if slot == 0 { 0 } else { slot + 1 };
            *value += tail_lo + F::from_u64(t as u64) * tail_step;
        }
        Ok(round_poly_from_skipped_evals(&evals, previous_claim))
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind).map_err(rows_unavailable)
    }
}

impl<F: JoltField> SumcheckKernel<F> for CycleKernel<F> {
    type Relation = BytecodeReadRafCycle<F>;

    fn output_claims(
        &mut self,
        _inputs: &SumcheckInputClaims<F, Self::Relation>,
    ) -> Result<SumcheckOutputClaims<F, Self::Relation>, SumcheckKernelError<F>> {
        self.progress.require_complete()?;
        let CycleRa::Lazy(ra) = &self.ra else {
            return Err(SumcheckKernelError::InvariantViolation {
                reason: "bytecode RA factors streamed after full binding",
            });
        };
        let output_openings = &self.output_openings;
        SumcheckOutputClaims::<F, Self::Relation>::from_opening_values(|id: &JoltOpeningId| {
            #[cfg(feature = "akita")]
            if *id == bytecode::fused_inc_read_raf_opening() {
                return Some(self.fused_inc.value(0));
            }
            output_openings
                .iter()
                .position(|opening| opening == id)
                .map(|index| ra.value(index, 0))
        })
        .map_err(SumcheckKernelError::from)
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod stage_pushforward_tests;

/// Byte-parity of both phases against the reference kernels, run as a PAIR
/// through one shared [`ProofSession`] so the parked per-cycle rows flow the
/// way production stages would exercise them.
///
/// Fixture honesty: `with_sample_backend` is the only witness plane
/// constructible without a `jolt-program` dependency. At its scale the
/// bytecode decomposition has a single committed chunk (`d = 1`), so the
/// cycle kernel's multi-factor `Π_i ra_i` loop runs with one factor; the
/// degree-`d+1` sampling, cold-cycle zeroing, entry/RAF fusion, and both
/// pushforward paths are exercised for real.
#[cfg(all(test, not(feature = "akita")))]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests;

#[cfg(all(test, feature = "akita"))]
#[expect(clippy::unwrap_used, reason = "test module")]
mod akita_tests;

/// The cycle phase steps every factor of a group by its slope from one
/// sample point to the next; these tests recompute each sample from scratch.
#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod cycle_group_tests;

#[cfg(test)]
#[expect(clippy::unwrap_used)]
mod grouped_cycle_tests;
