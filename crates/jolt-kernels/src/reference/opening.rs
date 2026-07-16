//! The reference joint-opening kernel: eager-dense — it materializes every
//! table dense and simultaneously (a test oracle at harness scale, never a
//! performance path — an optimized backend returns lazy/sparse or
//! device-backed implementations).

use std::collections::BTreeMap;

use jolt_claims::protocols::jolt::geometry::committed_openings::final_opening_id;
use jolt_claims::protocols::jolt::{JoltCommittedPolynomial, TracePolynomialOrder};
use jolt_field::Field;
use jolt_poly::MultilinearPoly;
use jolt_witness::protocols::jolt_vm::JoltVmNamespace;
use jolt_witness::WitnessProvider;

use super::views::dense_view;
use crate::commitment::CommitmentGrid;
use crate::opening::JointOpeningPolynomials;
use crate::{KernelError, ProofSession, ReferenceBackend};

impl<F: Field> JointOpeningPolynomials<F> for ReferenceBackend {
    fn prepare(
        &self,
        _session: &mut ProofSession,
        witness: &dyn WitnessProvider<F, JoltVmNamespace>,
        polynomials: &[JoltCommittedPolynomial],
        precommitted_tables: &BTreeMap<JoltCommittedPolynomial, Vec<F>>,
        grid: CommitmentGrid,
    ) -> Result<Vec<Box<dyn MultilinearPoly<F>>>, KernelError<F>> {
        let domain = 1usize << grid.total_vars;
        polynomials
            .iter()
            .map(|&polynomial| {
                let table = match precommitted_tables.get(&polynomial) {
                    Some(table) => table.clone(),
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
                        let mut table = table;
                        table.resize(domain, F::zero());
                        table
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
fn address_major_embed<F: Field>(
    table: &[F],
    grid: CommitmentGrid,
    polynomial: JoltCommittedPolynomial,
) -> Result<Vec<F>, KernelError<F>> {
    let cycles = 1usize << grid.log_t;
    let cycle_stride = grid.cycle_stride();
    let one_hot_stride = grid.one_hot_stride();
    let mut embedded = vec![F::zero(); 1usize << grid.total_vars];
    match polynomial {
        JoltCommittedPolynomial::RdInc | JoltCommittedPolynomial::RamInc => {
            if table.len() > cycles {
                return Err(KernelError::TableSizeMismatch {
                    table: format!("{polynomial:?}"),
                    expected: cycles,
                    got: table.len(),
                });
            }
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
            for k in 0..one_hot_k {
                for cycle in 0..cycles {
                    embedded[cycle * cycle_stride + k * one_hot_stride] = table[k * cycles + cycle];
                }
            }
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
fn block_embed<F: Field>(
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
    let column_mask = (1usize << sigma_advice) - 1;
    let sigma_main = grid.total_vars.div_ceil(2);
    let mut embedded = vec![F::zero(); 1usize << grid.total_vars];
    for (index, value) in table.iter().enumerate() {
        let row = index >> sigma_advice;
        let column = index & column_mask;
        embedded[(row << sigma_main) | column] = *value;
    }
    Ok(embedded)
}
