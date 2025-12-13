//! Shared utilities for RA (read-address) polynomials across all families.
//!
//! This module provides efficient computation of RA indices and G evaluations
//! that are shared across instruction, bytecode, and RAM polynomial families.
//!
//! ## Design Goals
//!
//! 1. **Single-pass trace iteration**: Compute all indices for all families in one pass
//! 2. **Cache locality**: All RA polynomials share the same eq table structure
//! 3. **Configurable delay binding**: Support delaying materialization for multiple rounds
//!
//! ## Two-Phase Architecture
//!
//! - **Phase 1**: Store shared eq table(s) and RA indices (compact representation)
//! - **Phase 2**: Materialize RA multilinear polynomials when needed
//!
//! ## SharedRaPolynomials
//!
//! Instead of storing N separate `RaPolynomial` each with their own eq table copy,
//! `SharedRaPolynomials` stores:
//! - ONE shared eq table (size K)
//! - `Vec<RaIndices>` (size T, non-transposed)
//!
//! This saves memory and improves cache locality when iterating through cycles.

use allocative::Allocative;

use crate::field::JoltField;
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding};
use crate::utils::thread::drop_in_background_thread;
use crate::zkvm::bytecode::BytecodePreprocessing;
use crate::zkvm::config::OneHotParams;
use crate::zkvm::instruction::LookupQuery;
use crate::zkvm::ram::remap_address;
use common::constants::XLEN;
use common::jolt_device::MemoryLayout;
use rayon::prelude::*;
use tracer::instruction::Cycle;

/// Maximum number of instruction RA chunks (lookup index splits into at most 32 chunks)
pub const MAX_INSTRUCTION_D: usize = 32;
/// Maximum number of bytecode RA chunks (PC splits into at most 6 chunks)
pub const MAX_BYTECODE_D: usize = 6;
/// Maximum number of RAM RA chunks (address splits into at most 8 chunks)
pub const MAX_RAM_D: usize = 8;

/// Stores all RA chunk indices for a single cycle.
/// Uses fixed-size arrays to avoid heap allocation in hot loops.
#[derive(Clone, Copy, Allocative)]
pub struct RaIndices {
    /// Instruction RA chunk indices (always present)
    pub instruction: [u16; MAX_INSTRUCTION_D],
    /// Bytecode RA chunk indices (always present)
    pub bytecode: [u16; MAX_BYTECODE_D],
    /// RAM RA chunk indices (None for non-memory cycles)
    pub ram: [Option<u16>; MAX_RAM_D],
}

impl RaIndices {
    /// Compute all RA chunk indices for a single cycle.
    #[inline]
    pub fn from_cycle(
        cycle: &Cycle,
        bytecode: &BytecodePreprocessing,
        memory_layout: &MemoryLayout,
        one_hot_params: &OneHotParams,
    ) -> Self {
        // Instruction indices from lookup index
        let lookup_index = LookupQuery::<XLEN>::to_lookup_index(cycle);
        let mut instruction = [0u16; MAX_INSTRUCTION_D];
        for i in 0..one_hot_params.instruction_d {
            instruction[i] = one_hot_params.lookup_index_chunk(lookup_index, i);
        }

        // Bytecode indices from PC
        let pc = bytecode.get_pc(cycle);
        let mut bytecode_arr = [0u16; MAX_BYTECODE_D];
        for i in 0..one_hot_params.bytecode_d {
            bytecode_arr[i] = one_hot_params.bytecode_pc_chunk(pc, i);
        }

        // RAM indices from remapped address (None for non-memory cycles)
        let address = cycle.ram_access().address() as u64;
        let remapped = remap_address(address, memory_layout);
        let mut ram = [None; MAX_RAM_D];
        for i in 0..one_hot_params.ram_d {
            ram[i] = remapped.map(|a| one_hot_params.ram_address_chunk(a, i));
        }

        Self {
            instruction,
            bytecode: bytecode_arr,
            ram,
        }
    }

    /// Extract the index for polynomial `poly_idx` in the unified ordering:
    /// [instruction_0..d, bytecode_0..d, ram_0..d]
    #[inline]
    pub fn get_index(&self, poly_idx: usize, one_hot_params: &OneHotParams) -> Option<u16> {
        let instruction_d = one_hot_params.instruction_d;
        let bytecode_d = one_hot_params.bytecode_d;

        if poly_idx < instruction_d {
            Some(self.instruction[poly_idx])
        } else if poly_idx < instruction_d + bytecode_d {
            Some(self.bytecode[poly_idx - instruction_d])
        } else {
            self.ram[poly_idx - instruction_d - bytecode_d]
        }
    }
}

/// Compute all H_indices for all families in parallel.
///
/// Returns H_indices in order: [instruction_0..d, bytecode_0..d, ram_0..d]
/// Each inner Vec has length trace.len() with the chunk index for that cycle.
///
/// This function parallelizes across trace cycles, computing all indices per cycle
/// and then transposing the results.
#[tracing::instrument(skip_all, name = "shared_ra_polys::compute_all_H_indices")]
pub fn compute_all_H_indices(
    trace: &[Cycle],
    bytecode: &BytecodePreprocessing,
    memory_layout: &MemoryLayout,
    one_hot_params: &OneHotParams,
) -> Vec<Vec<Option<u16>>> {
    let instruction_d = one_hot_params.instruction_d;
    let bytecode_d = one_hot_params.bytecode_d;
    let ram_d = one_hot_params.ram_d;
    let N = instruction_d + bytecode_d + ram_d;

    // Parallel computation: each cycle produces its RaIndices
    let all_indices: Vec<RaIndices> = trace
        .par_iter()
        .map(|cycle| RaIndices::from_cycle(cycle, bytecode, memory_layout, one_hot_params))
        .collect();

    // Transpose: for each polynomial, collect its index across all cycles
    (0..N)
        .into_par_iter()
        .map(|poly_idx| {
            all_indices
                .iter()
                .map(|indices| indices.get_index(poly_idx, one_hot_params))
                .collect()
        })
        .collect()
}

/// Compute all G evaluations for all families in parallel.
///
/// G_i(k) = Σ_j eq(r_cycle, j) · ra_i(k, j)
///
/// For one-hot RA polynomials, this simplifies to:
/// G_i(k) = Σ_{j: chunk_i(j) = k} eq_r_cycle[j]
///
/// Returns G in order: [instruction_0..d, bytecode_0..d, ram_0..d]
/// Each inner Vec has length k_chunk.
#[tracing::instrument(skip_all, name = "shared_ra_polys::compute_all_G")]
pub fn compute_all_G<F: JoltField>(
    trace: &[Cycle],
    bytecode: &BytecodePreprocessing,
    memory_layout: &MemoryLayout,
    one_hot_params: &OneHotParams,
    eq_r_cycle: &[F],
) -> Vec<Vec<F>> {
    let instruction_d = one_hot_params.instruction_d;
    let bytecode_d = one_hot_params.bytecode_d;
    let ram_d = one_hot_params.ram_d;
    let N = instruction_d + bytecode_d + ram_d;
    let K = one_hot_params.k_chunk;

    // Parallel fold over trace cycles
    // Each thread maintains local G arrays for all N polynomials
    trace
        .par_iter()
        .zip(eq_r_cycle.par_iter())
        .fold(
            || vec![vec![F::zero(); K]; N],
            |mut partial_G, (cycle, &eq_val)| {
                let indices =
                    RaIndices::from_cycle(cycle, bytecode, memory_layout, one_hot_params);

                // Instruction contributions
                for i in 0..instruction_d {
                    let k = indices.instruction[i] as usize;
                    partial_G[i][k] += eq_val;
                }

                // Bytecode contributions
                for i in 0..bytecode_d {
                    let k = indices.bytecode[i] as usize;
                    partial_G[instruction_d + i][k] += eq_val;
                }

                // RAM contributions (skip None indices)
                for i in 0..ram_d {
                    if let Some(k) = indices.ram[i] {
                        partial_G[instruction_d + bytecode_d + i][k as usize] += eq_val;
                    }
                }

                partial_G
            },
        )
        .reduce(
            || vec![vec![F::zero(); K]; N],
            |mut a, b| {
                for poly_idx in 0..N {
                    for k in 0..K {
                        a[poly_idx][k] += b[poly_idx][k];
                    }
                }
                a
            },
        )
}

// ============================================================================
// SharedRaPolynomials - Shared eq table for all RA polynomials
// ============================================================================

/// Shared RA polynomials that use a single eq table for all polynomials.
///
/// Instead of N separate `RaPolynomial` each with their own eq table copy,
/// this stores:
/// - ONE shared eq table (or split tables for later rounds)
/// - `Vec<RaIndices>` (size T, non-transposed)
///
/// This saves memory and improves cache locality.
#[derive(Allocative)]
pub enum SharedRaPolynomials<F: JoltField> {
    /// Round 1: Single shared eq table
    Round1(SharedRaRound1<F>),
    /// Round 2: Split into F_0, F_1
    Round2(SharedRaRound2<F>),
    /// Round 3: Split into F_00, F_01, F_10, F_11
    Round3(SharedRaRound3<F>),
    /// Round N: Fully materialized multilinear polynomials
    RoundN(Vec<MultilinearPolynomial<F>>),
}

/// Round 1 state: single shared eq table
#[derive(Allocative)]
pub struct SharedRaRound1<F: JoltField> {
    /// Shared eq table: F[k] = eq(r_address, k) for k in 0..K
    F: Vec<F>,
    /// RA indices for all cycles (non-transposed)
    indices: Vec<RaIndices>,
    /// Number of polynomials
    num_polys: usize,
    /// OneHotParams for index extraction
    #[allocative(skip)]
    one_hot_params: OneHotParams,
}

/// Round 2 state: split eq tables
#[derive(Allocative)]
pub struct SharedRaRound2<F: JoltField> {
    /// F_0[k] = eq(r_address, k) * eq(0, r0)
    F_0: Vec<F>,
    /// F_1[k] = eq(r_address, k) * eq(1, r0)
    F_1: Vec<F>,
    /// RA indices for all cycles
    indices: Vec<RaIndices>,
    num_polys: usize,
    #[allocative(skip)]
    one_hot_params: OneHotParams,
    binding_order: BindingOrder,
}

/// Round 3 state: further split eq tables
#[derive(Allocative)]
pub struct SharedRaRound3<F: JoltField> {
    F_00: Vec<F>,
    F_01: Vec<F>,
    F_10: Vec<F>,
    F_11: Vec<F>,
    indices: Vec<RaIndices>,
    num_polys: usize,
    #[allocative(skip)]
    one_hot_params: OneHotParams,
    binding_order: BindingOrder,
}

impl<F: JoltField> SharedRaPolynomials<F> {
    /// Create new SharedRaPolynomials from eq table and indices.
    pub fn new(eq_table: Vec<F>, indices: Vec<RaIndices>, one_hot_params: OneHotParams) -> Self {
        let num_polys =
            one_hot_params.instruction_d + one_hot_params.bytecode_d + one_hot_params.ram_d;
        Self::Round1(SharedRaRound1 {
            F: eq_table,
            indices,
            num_polys,
            one_hot_params,
        })
    }

    /// Get the number of polynomials
    pub fn num_polys(&self) -> usize {
        match self {
            Self::Round1(r) => r.num_polys,
            Self::Round2(r) => r.num_polys,
            Self::Round3(r) => r.num_polys,
            Self::RoundN(polys) => polys.len(),
        }
    }

    /// Get the current length (number of cycles / 2^rounds_so_far)
    pub fn len(&self) -> usize {
        match self {
            Self::Round1(r) => r.indices.len(),
            Self::Round2(r) => r.indices.len() / 2,
            Self::Round3(r) => r.indices.len() / 4,
            Self::RoundN(polys) => polys[0].len(),
        }
    }

    /// Get bound coefficient for polynomial `poly_idx` at position `j`
    #[inline]
    pub fn get_bound_coeff(&self, poly_idx: usize, j: usize) -> F {
        match self {
            Self::Round1(r) => r.get_bound_coeff(poly_idx, j),
            Self::Round2(r) => r.get_bound_coeff(poly_idx, j),
            Self::Round3(r) => r.get_bound_coeff(poly_idx, j),
            Self::RoundN(polys) => polys[poly_idx].get_bound_coeff(j),
        }
    }

    /// Get final sumcheck claim for polynomial `poly_idx`
    pub fn final_sumcheck_claim(&self, poly_idx: usize) -> F {
        match self {
            Self::RoundN(polys) => polys[poly_idx].final_sumcheck_claim(),
            _ => panic!("final_sumcheck_claim called before RoundN"),
        }
    }

    /// Bind with a challenge, transitioning to next round state.
    /// Consumes self and returns the new state.
    pub fn bind(self, r: F::Challenge, order: BindingOrder) -> Self {
        match self {
            Self::Round1(r1) => Self::Round2(r1.bind(r, order)),
            Self::Round2(r2) => Self::Round3(r2.bind(r, order)),
            Self::Round3(r3) => Self::RoundN(r3.bind(r, order)),
            Self::RoundN(mut polys) => {
                polys.par_iter_mut().for_each(|p| p.bind_parallel(r, order));
                Self::RoundN(polys)
            }
        }
    }

    /// Bind in place with a challenge, transitioning to next round state.
    pub fn bind_in_place(&mut self, r: F::Challenge, order: BindingOrder) {
        // Use take pattern: temporarily replace with RoundN(empty), then put back the real value
        let placeholder = Self::RoundN(vec![]);
        let current = std::mem::replace(self, placeholder);
        *self = current.bind(r, order);
    }
}

impl<F: JoltField> SharedRaRound1<F> {
    #[inline]
    fn get_bound_coeff(&self, poly_idx: usize, j: usize) -> F {
        self.indices[j]
            .get_index(poly_idx, &self.one_hot_params)
            .map_or(F::zero(), |k| self.F[k as usize])
    }

    fn bind(self, r0: F::Challenge, order: BindingOrder) -> SharedRaRound2<F> {
        let eq_0_r0 = EqPolynomial::mle(&[F::zero()], &[r0]);
        let eq_1_r0 = EqPolynomial::mle(&[F::one()], &[r0]);
        let F_0: Vec<F> = self.F.iter().map(|v| eq_0_r0 * v).collect();
        let F_1: Vec<F> = self.F.iter().map(|v| eq_1_r0 * v).collect();
        drop_in_background_thread(self.F);

        SharedRaRound2 {
            F_0,
            F_1,
            indices: self.indices,
            num_polys: self.num_polys,
            one_hot_params: self.one_hot_params,
            binding_order: order,
        }
    }
}

impl<F: JoltField> SharedRaRound2<F> {
    #[inline]
    fn get_bound_coeff(&self, poly_idx: usize, j: usize) -> F {
        match self.binding_order {
            BindingOrder::HighToLow => {
                let mid = self.indices.len() / 2;
                let h_0 = self.indices[j]
                    .get_index(poly_idx, &self.one_hot_params)
                    .map_or(F::zero(), |k| self.F_0[k as usize]);
                let h_1 = self.indices[mid + j]
                    .get_index(poly_idx, &self.one_hot_params)
                    .map_or(F::zero(), |k| self.F_1[k as usize]);
                h_0 + h_1
            }
            BindingOrder::LowToHigh => {
                let h_0 = self.indices[2 * j]
                    .get_index(poly_idx, &self.one_hot_params)
                    .map_or(F::zero(), |k| self.F_0[k as usize]);
                let h_1 = self.indices[2 * j + 1]
                    .get_index(poly_idx, &self.one_hot_params)
                    .map_or(F::zero(), |k| self.F_1[k as usize]);
                h_0 + h_1
            }
        }
    }

    fn bind(self, r1: F::Challenge, order: BindingOrder) -> SharedRaRound3<F> {
        assert_eq!(order, self.binding_order);
        let eq_0_r1 = EqPolynomial::mle(&[F::zero()], &[r1]);
        let eq_1_r1 = EqPolynomial::mle(&[F::one()], &[r1]);

        let mut F_00 = self.F_0.clone();
        let mut F_01 = self.F_0;
        let mut F_10 = self.F_1.clone();
        let mut F_11 = self.F_1;

        // Scale all four tables in parallel using nested joins
        rayon::join(
            || {
                rayon::join(
                    || F_00.par_iter_mut().for_each(|f| *f *= eq_0_r1),
                    || F_01.par_iter_mut().for_each(|f| *f *= eq_1_r1),
                )
            },
            || {
                rayon::join(
                    || F_10.par_iter_mut().for_each(|f| *f *= eq_0_r1),
                    || F_11.par_iter_mut().for_each(|f| *f *= eq_1_r1),
                )
            },
        );

        SharedRaRound3 {
            F_00,
            F_01,
            F_10,
            F_11,
            indices: self.indices,
            num_polys: self.num_polys,
            one_hot_params: self.one_hot_params,
            binding_order: order,
        }
    }
}

impl<F: JoltField> SharedRaRound3<F> {
    #[inline]
    fn get_bound_coeff(&self, poly_idx: usize, j: usize) -> F {
        match self.binding_order {
            BindingOrder::HighToLow => {
                let quarter = self.indices.len() / 4;
                let h_00 = self.indices[j]
                    .get_index(poly_idx, &self.one_hot_params)
                    .map_or(F::zero(), |k| self.F_00[k as usize]);
                let h_01 = self.indices[quarter + j]
                    .get_index(poly_idx, &self.one_hot_params)
                    .map_or(F::zero(), |k| self.F_01[k as usize]);
                let h_10 = self.indices[2 * quarter + j]
                    .get_index(poly_idx, &self.one_hot_params)
                    .map_or(F::zero(), |k| self.F_10[k as usize]);
                let h_11 = self.indices[3 * quarter + j]
                    .get_index(poly_idx, &self.one_hot_params)
                    .map_or(F::zero(), |k| self.F_11[k as usize]);
                h_00 + h_01 + h_10 + h_11
            }
            BindingOrder::LowToHigh => {
                let h_00 = self.indices[4 * j]
                    .get_index(poly_idx, &self.one_hot_params)
                    .map_or(F::zero(), |k| self.F_00[k as usize]);
                let h_01 = self.indices[4 * j + 1]
                    .get_index(poly_idx, &self.one_hot_params)
                    .map_or(F::zero(), |k| self.F_01[k as usize]);
                let h_10 = self.indices[4 * j + 2]
                    .get_index(poly_idx, &self.one_hot_params)
                    .map_or(F::zero(), |k| self.F_10[k as usize]);
                let h_11 = self.indices[4 * j + 3]
                    .get_index(poly_idx, &self.one_hot_params)
                    .map_or(F::zero(), |k| self.F_11[k as usize]);
                h_00 + h_01 + h_10 + h_11
            }
        }
    }

    #[tracing::instrument(skip_all, name = "SharedRaRound3::bind")]
    fn bind(self, r2: F::Challenge, order: BindingOrder) -> Vec<MultilinearPolynomial<F>> {
        assert_eq!(order, self.binding_order);

        // Materialize all polynomials in parallel
        let num_polys = self.num_polys;
        let indices = &self.indices;
        let one_hot_params = &self.one_hot_params;

        // Combine F tables: F_combined[k] = F_00[k]*eq(00,r) + F_01[k]*eq(01,r) + ...
        let eq_0_r2 = EqPolynomial::mle(&[F::zero()], &[r2]);
        let eq_1_r2 = EqPolynomial::mle(&[F::one()], &[r2]);
        
        let F_combined: Vec<F> = (0..self.F_00.len())
            .into_par_iter()
            .map(|k| {
                self.F_00[k] * eq_0_r2 + self.F_01[k] * eq_1_r2 
                    + self.F_10[k] * eq_0_r2 + self.F_11[k] * eq_1_r2
            })
            .collect();

        // For each polynomial, materialize from indices
        let new_len = indices.len() / 8;
        (0..num_polys)
            .into_par_iter()
            .map(|poly_idx| {
                let coeffs: Vec<F> = match order {
                    BindingOrder::LowToHigh => {
                        (0..new_len)
                            .map(|j| {
                                // Sum over 8 consecutive indices
                                (0..8)
                                    .map(|offset| {
                                        indices[8 * j + offset]
                                            .get_index(poly_idx, one_hot_params)
                                            .map_or(F::zero(), |k| F_combined[k as usize])
                                    })
                                    .sum()
                            })
                            .collect()
                    }
                    BindingOrder::HighToLow => {
                        let eighth = indices.len() / 8;
                        (0..new_len)
                            .map(|j| {
                                (0..8)
                                    .map(|seg| {
                                        indices[seg * eighth + j]
                                            .get_index(poly_idx, one_hot_params)
                                            .map_or(F::zero(), |k| F_combined[k as usize])
                                    })
                                    .sum()
                            })
                            .collect()
                    }
                };
                MultilinearPolynomial::from(coeffs)
            })
            .collect()
    }
}

/// Compute all RaIndices in parallel (non-transposed).
///
/// Returns one `RaIndices` per cycle.
#[tracing::instrument(skip_all, name = "shared_ra_polys::compute_ra_indices")]
pub fn compute_ra_indices(
    trace: &[Cycle],
    bytecode: &BytecodePreprocessing,
    memory_layout: &MemoryLayout,
    one_hot_params: &OneHotParams,
) -> Vec<RaIndices> {
    trace
        .par_iter()
        .map(|cycle| RaIndices::from_cycle(cycle, bytecode, memory_layout, one_hot_params))
        .collect()
}

#[cfg(test)]
mod tests {
    // TODO: Add tests for compute_all_H_indices
    // TODO: Add tests for compute_all_G
    // TODO: Add tests comparing parallel vs sequential results
    // TODO: Add tests for SharedRaPolynomials
}
