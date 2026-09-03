//! The reference joint-opening kernel: eager-dense — it materializes every
//! table dense and simultaneously (a test oracle at harness scale, never a
//! performance path — an optimized backend returns lazy/sparse or
//! device-backed implementations).

use std::collections::BTreeMap;

use jolt_claims::protocols::jolt::geometry::committed_openings::final_opening_id;
use jolt_claims::protocols::jolt::{JoltCommittedPolynomial, TracePolynomialOrder};
use jolt_field::JoltField;
use jolt_poly::MultilinearPoly;
use jolt_utils::unsafe_allocate_zero_vec;
use jolt_witness::JoltWitnessPlane;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::views::dense_view;
use crate::commitment::CommitmentGrid;
use crate::opening::JointOpeningPolynomials;
use crate::{KernelError, ProofSession, ReferenceBackend};

impl<F: JoltField> JointOpeningPolynomials<F> for ReferenceBackend {
    // The backend-neutral `JointOpeningPolynomials::prepare` span lives at
    // the Dory stage-8 boundary (`crates/jolt-prover/src/dory/stages/stage8.rs`),
    // so every implementation inherits it — see the taxonomy's kernel-seam
    // contract.
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        polynomials: &[JoltCommittedPolynomial],
        mut precommitted_tables: BTreeMap<JoltCommittedPolynomial, Vec<F>>,
        grid: CommitmentGrid,
    ) -> Result<Vec<Box<dyn MultilinearPoly<F>>>, KernelError<F>> {
        let domain = 1usize << grid.total_vars;
        polynomials
            .iter()
            .map(|&polynomial| {
                let table = match precommitted_tables.remove(&polynomial) {
                    Some(table) => table,
                    None => dense_view(witness, final_opening_id(polynomial))?,
                };
                if table.len() > domain {
                    return Err(KernelError::TableSizeMismatch {
                        table: format!("{polynomial:?}"),
                        expected: domain,
                        got: table.len(),
                    });
                }
                let embedded = match polynomial {
                    JoltCommittedPolynomial::TrustedAdvice
                    | JoltCommittedPolynomial::UntrustedAdvice
                    | JoltCommittedPolynomial::BytecodeChunk(_)
                    | JoltCommittedPolynomial::ProgramImageInit => {
                        block_embed(&table, grid, polynomial)?
                    }
                    _ if grid.order == TracePolynomialOrder::AddressMajor => {
                        address_major_embed(&table, grid, polynomial)?
                    }
                    _ => {
                        let mut embedded: Vec<F> = unsafe_allocate_zero_vec(domain);
                        embedded[..table.len()].copy_from_slice(&table);
                        embedded
                    }
                };
                Ok(Box::new(embedded) as Box<dyn MultilinearPoly<F>>)
            })
            .collect()
    }
}

/// Embed a trace polynomial cycle-block-strided over the address-major grid:
/// a one-hot table's native `k · T + t` view permutes to `t · cycle_stride +
/// k · one_hot_stride`; a dense (per-cycle) table sits at each cycle block's
/// address slot zero.
fn address_major_embed<F: JoltField>(
    table: &[F],
    grid: CommitmentGrid,
    polynomial: JoltCommittedPolynomial,
) -> Result<Vec<F>, KernelError<F>> {
    let cycles = 1usize << grid.log_t;
    let cycle_stride = grid.cycle_stride();
    let one_hot_stride = grid.one_hot_stride();
    // `2^total_vars = cycles · cycle_stride`, so the grid splits exactly into
    // per-cycle blocks and each block's writes stay inside it
    // (`k · one_hot_stride < cycle_stride`) — the scatters below run as
    // disjoint per-block gathers.
    let mut embedded: Vec<F> = unsafe_allocate_zero_vec(1usize << grid.total_vars);
    match polynomial {
        JoltCommittedPolynomial::RdInc | JoltCommittedPolynomial::RamInc => {
            if table.len() > cycles {
                return Err(KernelError::TableSizeMismatch {
                    table: format!("{polynomial:?}"),
                    expected: cycles,
                    got: table.len(),
                });
            }
            #[cfg(feature = "parallel")]
            embedded
                .par_chunks_mut(cycle_stride)
                .zip(table.par_iter())
                .for_each(|(block, value)| block[0] = *value);
            #[cfg(not(feature = "parallel"))]
            for (cycle, value) in table.iter().enumerate() {
                embedded[cycle * cycle_stride] = *value;
            }
        }
        JoltCommittedPolynomial::InstructionRa(_)
        | JoltCommittedPolynomial::BytecodeRa(_)
        | JoltCommittedPolynomial::RamRa(_) => {
            let max_k = 1usize << grid.log_k_chunk;
            if !table.len().is_multiple_of(cycles) || table.len() / cycles > max_k {
                return Err(KernelError::InvalidGeometry {
                    reason: format!(
                        "one-hot table for {polynomial:?} ({} entries) is not a (K × {cycles}) \
                         grid with K at most {max_k}",
                        table.len()
                    ),
                });
            }
            let one_hot_k = table.len() / cycles;
            let fill_block = |cycle: usize, block: &mut [F]| {
                for k in 0..one_hot_k {
                    block[k * one_hot_stride] = table[k * cycles + cycle];
                }
            };
            #[cfg(feature = "parallel")]
            embedded
                .par_chunks_mut(cycle_stride)
                .enumerate()
                .for_each(|(cycle, block)| fill_block(cycle, block));
            #[cfg(not(feature = "parallel"))]
            embedded
                .chunks_mut(cycle_stride)
                .enumerate()
                .for_each(|(cycle, block)| fill_block(cycle, block));
        }
        _ => {
            return Err(KernelError::InvariantViolation {
                reason: "only trace polynomials embed address-major",
            });
        }
    }
    Ok(embedded)
}

/// Embed an advice polynomial's balanced matrix into the grid matrix's
/// top-left block: advice coefficient `row · 2^σ_a + col` lands at grid index
/// `row · 2^σ_main + col`.
fn block_embed<F: JoltField>(
    table: &[F],
    grid: CommitmentGrid,
    polynomial: JoltCommittedPolynomial,
) -> Result<Vec<F>, KernelError<F>> {
    if !table.len().is_power_of_two() {
        return Err(KernelError::TableSizeMismatch {
            table: format!("{polynomial:?}"),
            expected: table.len().next_power_of_two(),
            got: table.len(),
        });
    }
    let advice_vars = table.len().ilog2() as usize;
    let sigma_advice = advice_vars.div_ceil(2);
    let sigma_main = grid.total_vars.div_ceil(2);
    let mut embedded: Vec<F> = unsafe_allocate_zero_vec(1usize << grid.total_vars);
    // Row-block copies: advice row `r` (width `2^σ_a`, `σ_a ≤ σ_main` since
    // the caller checked `table.len() ≤ 2^total_vars`) lands at the head of
    // grid row `r`.
    #[cfg(feature = "parallel")]
    embedded
        .par_chunks_mut(1usize << sigma_main)
        .zip(table.par_chunks(1usize << sigma_advice))
        .for_each(|(grid_row, advice_row)| {
            grid_row[..advice_row.len()].copy_from_slice(advice_row);
        });
    #[cfg(not(feature = "parallel"))]
    embedded
        .chunks_mut(1usize << sigma_main)
        .zip(table.chunks(1usize << sigma_advice))
        .for_each(|(grid_row, advice_row)| {
            grid_row[..advice_row.len()].copy_from_slice(advice_row);
        });
    Ok(embedded)
}
