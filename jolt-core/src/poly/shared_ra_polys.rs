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
//! - One (small) eq table per polynomial (size K each; K is 16 or 256 in practice)
//! - A single `Vec<RaIndices>` (size T, non-transposed) shared by all polynomials
//!
//! This saves memory and improves cache locality when iterating through cycles.

use allocative::Allocative;
use ark_std::Zero;
use fixedbitset::FixedBitSet;

use std::sync::Arc;

use crate::field::JoltField;
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding};
use crate::utils::thread::unsafe_allocate_zero_vec;
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

/// Asserts that the one_hot_params dimensions are within bounds.
/// Call this once at the start of bulk operations to catch issues early.
#[inline]
pub fn assert_ra_bounds(one_hot_params: &OneHotParams) {
    assert!(
        one_hot_params.instruction_d <= MAX_INSTRUCTION_D,
        "instruction_d {} exceeds MAX_INSTRUCTION_D {}",
        one_hot_params.instruction_d,
        MAX_INSTRUCTION_D
    );
    assert!(
        one_hot_params.bytecode_d <= MAX_BYTECODE_D,
        "bytecode_d {} exceeds MAX_BYTECODE_D {}",
        one_hot_params.bytecode_d,
        MAX_BYTECODE_D
    );
    assert!(
        one_hot_params.ram_d <= MAX_RAM_D,
        "ram_d {} exceeds MAX_RAM_D {}",
        one_hot_params.ram_d,
        MAX_RAM_D
    );
}

/// Stores all RA chunk indices for a single cycle.
/// Uses fixed-size arrays to avoid heap allocation in hot loops.
#[derive(Clone, Copy, Default, Allocative)]
pub struct RaIndices {
    /// Instruction RA chunk indices (always present)
    pub instruction: [u8; MAX_INSTRUCTION_D],
    /// Bytecode RA chunk indices (always present)
    pub bytecode: [u8; MAX_BYTECODE_D],
    /// RAM RA chunk indices (None for non-memory cycles)
    pub ram: [Option<u8>; MAX_RAM_D],
}

impl std::ops::Add for RaIndices {
    type Output = Self;

    fn add(self, _rhs: Self) -> Self::Output {
        // This is only implemented to satisfy the Zero trait bound.
        // RaIndices should never actually be added together.
        unimplemented!("RaIndices::add is not meaningful; this impl exists only for Zero trait")
    }
}

/// Implement Zero trait for RaIndices to satisfy the trait bound for `unsafe_allocate_zero_vec`
impl Zero for RaIndices {
    fn zero() -> Self {
        // `unsafe_allocate_zero_vec` relies on the invariant that `Zero::zero()` is represented
        // by all-zero bytes. Constructing `[None; N]` can leave padding / unused enum payload
        // bytes uninitialized, which breaks that invariant (and is UB to inspect as bytes).
        //
        // All-zero is a valid bit-pattern for `RaIndices` (arrays of integers + `Option<u8>`),
        // so this is safe here.
        unsafe { core::mem::zeroed() }
    }

    fn is_zero(&self) -> bool {
        self.instruction.iter().all(|&x| x == 0)
            && self.bytecode.iter().all(|&x| x == 0)
            && self.ram.iter().all(|x| x.is_none())
    }
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
        // Debug assertions for bounds (use assert_ra_bounds once at bulk operation start)
        debug_assert!(
            one_hot_params.instruction_d <= MAX_INSTRUCTION_D,
            "instruction_d {} exceeds MAX_INSTRUCTION_D {}",
            one_hot_params.instruction_d,
            MAX_INSTRUCTION_D
        );
        debug_assert!(
            one_hot_params.bytecode_d <= MAX_BYTECODE_D,
            "bytecode_d {} exceeds MAX_BYTECODE_D {}",
            one_hot_params.bytecode_d,
            MAX_BYTECODE_D
        );
        debug_assert!(
            one_hot_params.ram_d <= MAX_RAM_D,
            "ram_d {} exceeds MAX_RAM_D {}",
            one_hot_params.ram_d,
            MAX_RAM_D
        );

        // Instruction indices from lookup index
        let lookup_index = LookupQuery::<XLEN>::to_lookup_index(cycle);
        let mut instruction = [0u8; MAX_INSTRUCTION_D];
        for i in 0..one_hot_params.instruction_d {
            instruction[i] = one_hot_params.lookup_index_chunk(lookup_index, i);
        }

        // Bytecode indices from PC
        let pc = bytecode.get_pc(cycle);
        let mut bytecode_arr = [0u8; MAX_BYTECODE_D];
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
    pub fn get_index(&self, poly_idx: usize, one_hot_params: &OneHotParams) -> Option<u8> {
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

/// Compute all G evaluations for all families in parallel using split-eq optimization.
///
/// G_i(k) = Σ_j eq(r_cycle, j) · ra_i(k, j)
///
/// For one-hot RA polynomials, this simplifies to:
/// G_i(k) = Σ_{j: chunk_i(j) = k} eq_r_cycle[j]
///
/// Uses a two-table split-eq: split `r_cycle` into MSB/LSB halves, compute `E_hi` and `E_lo`,
/// then `eq(r_cycle, c) = E_hi[c_hi] * E_lo[c_lo]` where `c = (c_hi << lo_bits) | c_lo`.
///
/// Returns G in order: [instruction_0..d, bytecode_0..d, ram_0..d]
/// Each inner Vec has length k_chunk.
#[tracing::instrument(skip_all, name = "shared_ra_polys::compute_all_G")]
pub fn compute_all_G<F: JoltField>(
    trace: &[Cycle],
    bytecode: &BytecodePreprocessing,
    memory_layout: &MemoryLayout,
    one_hot_params: &OneHotParams,
    r_cycle: &[F::Challenge],
) -> Vec<Vec<F>> {
    compute_all_G_impl::<F>(
        trace,
        bytecode,
        memory_layout,
        one_hot_params,
        r_cycle,
        None,
    )
}

/// Compute all G evaluations AND RA indices in a single pass over the trace.
///
/// This avoids traversing the trace twice when both G and ra_indices are needed.
///
/// Returns (G, ra_indices) where:
/// - G[i] = pushforward of ra_i over r_cycle (length k_chunk each)
/// - ra_indices[j] = RA chunk indices for cycle j
#[tracing::instrument(skip_all, name = "shared_ra_polys::compute_all_G_and_ra_indices")]
pub fn compute_all_G_and_ra_indices<F: JoltField>(
    trace: &[Cycle],
    bytecode: &BytecodePreprocessing,
    memory_layout: &MemoryLayout,
    one_hot_params: &OneHotParams,
    r_cycle: &[F::Challenge],
) -> (Vec<Vec<F>>, Arc<Vec<RaIndices>>) {
    let T = trace.len();
    let mut ra_indices: Vec<RaIndices> = unsafe_allocate_zero_vec(T);

    let G = compute_all_G_impl::<F>(
        trace,
        bytecode,
        memory_layout,
        one_hot_params,
        r_cycle,
        Some(&mut ra_indices),
    );

    (G, Arc::new(ra_indices))
}

/// Core implementation for computing G evaluations.
///
/// When `ra_indices` is `Some`, also writes RaIndices to the provided slice.
/// This is safe because each cycle index is visited exactly once (disjoint writes).
#[inline(always)]
fn compute_all_G_impl<F: JoltField>(
    trace: &[Cycle],
    bytecode: &BytecodePreprocessing,
    memory_layout: &MemoryLayout,
    one_hot_params: &OneHotParams,
    r_cycle: &[F::Challenge],
    ra_indices: Option<&mut [RaIndices]>,
) -> Vec<Vec<F>> {
    // Convert to usize for thread safety (usize is Send + Sync, raw pointers are not Sync)
    let ra_ptr_usize: usize = ra_indices.map(|s| s.as_mut_ptr() as usize).unwrap_or(0);
    // Verify bounds once at the start
    assert_ra_bounds(one_hot_params);

    let K = one_hot_params.k_chunk;
    let instruction_d = one_hot_params.instruction_d;
    let bytecode_d = one_hot_params.bytecode_d;
    let ram_d = one_hot_params.ram_d;
    let N = instruction_d + bytecode_d + ram_d;
    let T = trace.len();

    // Two-table split-eq:
    // EqPolynomial::evals uses big-endian bit order: r_cycle[0] is MSB, r_cycle[last] is LSB.
    // To get contiguous blocks in the cycle index, we split off the LSB half (suffix) as E_lo.
    let log_T = r_cycle.len();
    let lo_bits = log_T / 2;
    let hi_bits = log_T - lo_bits;
    let (r_hi, r_lo) = r_cycle.split_at(hi_bits);

    let (E_hi, E_lo) = rayon::join(
        || EqPolynomial::<F>::evals(r_hi),
        || EqPolynomial::<F>::evals(r_lo),
    );

    let in_len = E_lo.len(); // 2^lo_bits

    // Split E_hi into exactly num_threads chunks to minimize allocations
    // Each thread allocates ONE partial_G and processes its entire chunk.
    let num_threads = rayon::current_num_threads();
    let out_len = E_hi.len(); // 2^hi_bits
    let chunk_size = out_len.div_ceil(num_threads);

    // Parallel map over thread chunks using deferred reduction
    E_hi.par_chunks(chunk_size)
        .enumerate()
        .map(|(chunk_idx, chunk)| {
            // Allocate separate arrays per polynomial type
            let mut partial_instruction: Vec<Vec<F>> = (0..instruction_d)
                .map(|_| unsafe_allocate_zero_vec(K))
                .collect();
            let mut partial_bytecode: Vec<Vec<F>> = (0..bytecode_d)
                .map(|_| unsafe_allocate_zero_vec(K))
                .collect();
            let mut partial_ram: Vec<Vec<F>> =
                (0..ram_d).map(|_| unsafe_allocate_zero_vec(K)).collect();

            // Reusable local unreduced accumulators (5-limb) and touched flags
            let mut local_instruction: Vec<Vec<F::UnreducedMulU64>> = (0..instruction_d)
                .map(|_| unsafe_allocate_zero_vec(K))
                .collect();
            let mut local_bytecode: Vec<Vec<F::UnreducedMulU64>> = (0..bytecode_d)
                .map(|_| unsafe_allocate_zero_vec(K))
                .collect();
            let mut local_ram: Vec<Vec<F::UnreducedMulU64>> =
                (0..ram_d).map(|_| unsafe_allocate_zero_vec(K)).collect();
            let mut touched_instruction: Vec<FixedBitSet> =
                vec![FixedBitSet::with_capacity(K); instruction_d];
            let mut touched_bytecode: Vec<FixedBitSet> =
                vec![FixedBitSet::with_capacity(K); bytecode_d];
            let mut touched_ram: Vec<FixedBitSet> = vec![FixedBitSet::with_capacity(K); ram_d];

            let chunk_start = chunk_idx * chunk_size;
            for (local_idx, &e_hi) in chunk.iter().enumerate() {
                let c_hi = chunk_start + local_idx;
                let c_hi_base = c_hi * in_len;

                // Clear touched flags and local accumulators for this c_hi
                for i in 0..instruction_d {
                    for k in touched_instruction[i].ones() {
                        local_instruction[i][k] = Default::default();
                    }
                    touched_instruction[i].clear();
                }
                for i in 0..bytecode_d {
                    for k in touched_bytecode[i].ones() {
                        local_bytecode[i][k] = Default::default();
                    }
                    touched_bytecode[i].clear();
                }
                for i in 0..ram_d {
                    for k in touched_ram[i].ones() {
                        local_ram[i][k] = Default::default();
                    }
                    touched_ram[i].clear();
                }

                // Sequential over c_lo (contiguous cycles for this c_hi)
                for c_lo in 0..in_len {
                    let j = c_hi_base + c_lo;
                    if j >= T {
                        break;
                    }

                    // Get 4-limb unreduced representation
                    let add = E_lo[c_lo].to_unreduced();

                    let ra_idx =
                        RaIndices::from_cycle(&trace[j], bytecode, memory_layout, one_hot_params);

                    // Write ra_indices if collecting (disjoint write, each j visited once)
                    if ra_ptr_usize != 0 {
                        // SAFETY: Each j value is unique across all parallel iterations,
                        // so this write is to a disjoint index. No data race possible.
                        unsafe {
                            let ra_ptr = ra_ptr_usize as *mut RaIndices;
                            *ra_ptr.add(j) = ra_idx;
                        }
                    }

                    // InstructionRa contributions (unreduced accumulation)
                    for i in 0..instruction_d {
                        let k = ra_idx.instruction[i] as usize;
                        if !touched_instruction[i].contains(k) {
                            touched_instruction[i].insert(k);
                        }
                        local_instruction[i][k] += add;
                    }

                    // BytecodeRa contributions (unreduced accumulation)
                    for i in 0..bytecode_d {
                        let k = ra_idx.bytecode[i] as usize;
                        if !touched_bytecode[i].contains(k) {
                            touched_bytecode[i].insert(k);
                        }
                        local_bytecode[i][k] += add;
                    }

                    // RamRa contributions (may be None, unreduced accumulation)
                    for i in 0..ram_d {
                        if let Some(k) = ra_idx.ram[i] {
                            let k = k as usize;
                            if !touched_ram[i].contains(k) {
                                touched_ram[i].insert(k);
                            }
                            local_ram[i][k] += add;
                        }
                    }
                }

                // Barrett reduce and scale by E_hi[c_hi], only for touched indices
                for i in 0..instruction_d {
                    for k in touched_instruction[i].ones() {
                        let reduced = F::reduce_mul_u64(local_instruction[i][k]);
                        partial_instruction[i][k] += e_hi * reduced;
                    }
                }
                for i in 0..bytecode_d {
                    for k in touched_bytecode[i].ones() {
                        let reduced = F::reduce_mul_u64(local_bytecode[i][k]);
                        partial_bytecode[i][k] += e_hi * reduced;
                    }
                }
                for i in 0..ram_d {
                    for k in touched_ram[i].ones() {
                        let reduced = F::reduce_mul_u64(local_ram[i][k]);
                        partial_ram[i][k] += e_hi * reduced;
                    }
                }
            }

            // Combine into single Vec<Vec<F>> in order: instruction, bytecode, ram
            let mut result: Vec<Vec<F>> = Vec::with_capacity(N);
            result.extend(partial_instruction);
            result.extend(partial_bytecode);
            result.extend(partial_ram);
            result
        })
        .reduce(
            || (0..N).map(|_| unsafe_allocate_zero_vec::<F>(K)).collect(),
            |mut a, b| {
                for (a_poly, b_poly) in a.iter_mut().zip(b.iter()) {
                    a_poly
                        .par_iter_mut()
                        .zip(b_poly.par_iter())
                        .for_each(|(a_val, b_val)| *a_val += *b_val);
                }
                a
            },
        )
}

/// When the table round reaches this many groups, the next bind materializes to dense.
const SHARED_MATERIALIZE_THRESHOLD: usize = 16;

/// Shared RA polynomials using table-doubling state machine.
///
/// Instead of N separate `RaPolynomial` each with their own eq table copy,
/// stores per-polynomial eq tables and a single shared `Vec<RaIndices>`.
/// Each bind doubles the table groups; once the threshold is reached,
/// the next bind materializes to dense `MultilinearPolynomial`s.
#[derive(Allocative)]
pub enum SharedRaPolynomials<F: JoltField> {
    TableRound(SharedRaTableRound<F>),
    RoundN(Vec<MultilinearPolynomial<F>>),
}

/// Generic table round for SharedRaPolynomials with `n_groups` eq table groups.
///
/// `tables[group_idx][poly_idx][k]` — tables are in LowToHigh interleaving order.
#[derive(Allocative)]
pub struct SharedRaTableRound<F: JoltField> {
    tables: Vec<Vec<Vec<F>>>,
    indices: Arc<Vec<RaIndices>>,
    num_polys: usize,
    #[allocative(skip)]
    one_hot_params: OneHotParams,
    binding_order: BindingOrder,
}

impl<F: JoltField> Default for SharedRaTableRound<F> {
    fn default() -> Self {
        Self {
            tables: Vec::new(),
            indices: Arc::new(Vec::new()),
            num_polys: 0,
            one_hot_params: OneHotParams::default(),
            binding_order: BindingOrder::LowToHigh,
        }
    }
}

impl<F: JoltField> SharedRaPolynomials<F> {
    pub fn new(
        tables: Vec<Vec<F>>,
        indices: Arc<Vec<RaIndices>>,
        one_hot_params: OneHotParams,
    ) -> Self {
        let num_polys =
            one_hot_params.instruction_d + one_hot_params.bytecode_d + one_hot_params.ram_d;
        debug_assert!(
            tables.len() == num_polys,
            "SharedRaPolynomials::new: tables.len() = {}, expected num_polys = {}",
            tables.len(),
            num_polys
        );
        Self::TableRound(SharedRaTableRound {
            tables: vec![tables],
            indices,
            num_polys,
            one_hot_params,
            binding_order: BindingOrder::LowToHigh,
        })
    }

    pub fn new_instruction_only(
        tables: Vec<Vec<F>>,
        indices: Arc<Vec<RaIndices>>,
        one_hot_params: OneHotParams,
    ) -> Self {
        let num_polys = one_hot_params.instruction_d;
        debug_assert_eq!(tables.len(), num_polys);
        Self::TableRound(SharedRaTableRound {
            tables: vec![tables],
            indices,
            num_polys,
            one_hot_params,
            binding_order: BindingOrder::LowToHigh,
        })
    }

    pub fn num_polys(&self) -> usize {
        match self {
            Self::TableRound(t) => t.num_polys,
            Self::RoundN(polys) => polys.len(),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::TableRound(t) => t.len(),
            Self::RoundN(polys) => polys[0].len(),
        }
    }

    #[inline]
    pub fn get_bound_coeff(&self, poly_idx: usize, j: usize) -> F {
        match self {
            Self::TableRound(t) => t.get_bound_coeff(poly_idx, j),
            Self::RoundN(polys) => polys[poly_idx].get_bound_coeff(j),
        }
    }

    pub fn final_sumcheck_claim(&self, poly_idx: usize) -> F {
        match self {
            Self::RoundN(polys) => polys[poly_idx].final_sumcheck_claim(),
            _ => panic!("final_sumcheck_claim called before RoundN"),
        }
    }

    pub fn bind(self, r: F::Challenge, order: BindingOrder) -> Self {
        match self {
            Self::TableRound(t) => {
                if t.n_groups() >= SHARED_MATERIALIZE_THRESHOLD {
                    Self::RoundN(t.materialize(r, order))
                } else {
                    Self::TableRound(t.bind(r, order))
                }
            }
            Self::RoundN(mut polys) => {
                polys.par_iter_mut().for_each(|p| p.bind_parallel(r, order));
                Self::RoundN(polys)
            }
        }
    }

    pub fn bind_in_place(&mut self, r: F::Challenge, order: BindingOrder) {
        match self {
            Self::TableRound(t) => {
                if t.n_groups() >= SHARED_MATERIALIZE_THRESHOLD {
                    *self = Self::RoundN(std::mem::take(t).materialize(r, order));
                } else {
                    *self = Self::TableRound(std::mem::take(t).bind(r, order));
                }
            }
            Self::RoundN(polys) => {
                polys.par_iter_mut().for_each(|p| p.bind_parallel(r, order));
            }
        }
    }
}

impl<F: JoltField> SharedRaTableRound<F> {
    #[inline]
    fn n_groups(&self) -> usize {
        self.tables.len()
    }

    fn len(&self) -> usize {
        self.indices.len() / self.n_groups()
    }

    fn double_tables(tables: Vec<Vec<Vec<F>>>, r: F::Challenge) -> Vec<Vec<Vec<F>>> {
        let eq_0 = EqPolynomial::mle(&[F::zero()], &[r]);
        let eq_1 = EqPolynomial::mle(&[F::one()], &[r]);
        let n = tables.len();
        let mut doubled: Vec<Vec<Vec<F>>> = Vec::with_capacity(2 * n);
        for t in &tables {
            doubled.push(t.clone());
        }
        for t in tables {
            doubled.push(t);
        }
        let (lo, hi) = doubled.split_at_mut(n);
        rayon::join(
            || {
                lo.par_iter_mut().for_each(|group| {
                    group
                        .par_iter_mut()
                        .for_each(|t| t.par_iter_mut().for_each(|f| *f *= eq_0))
                })
            },
            || {
                hi.par_iter_mut().for_each(|group| {
                    group
                        .par_iter_mut()
                        .for_each(|t| t.par_iter_mut().for_each(|f| *f *= eq_1))
                })
            },
        );
        doubled
    }

    #[tracing::instrument(skip_all, name = "SharedRaTableRound::bind")]
    fn bind(self, r: F::Challenge, order: BindingOrder) -> Self {
        if self.n_groups() > 1 {
            assert_eq!(order, self.binding_order);
        }
        Self {
            tables: Self::double_tables(self.tables, r),
            indices: self.indices,
            num_polys: self.num_polys,
            one_hot_params: self.one_hot_params,
            binding_order: order,
        }
    }

    #[tracing::instrument(skip_all, name = "SharedRaTableRound::materialize")]
    fn materialize(self, r: F::Challenge, order: BindingOrder) -> Vec<MultilinearPolynomial<F>> {
        let binding_order = if self.n_groups() > 1 {
            assert_eq!(order, self.binding_order);
            self.binding_order
        } else {
            order
        };
        let tables = Self::double_tables(self.tables, r);
        let n_total = tables.len();
        let indices = &self.indices;
        let one_hot_params = &self.one_hot_params;
        let new_len = indices.len() / n_total;

        (0..self.num_polys)
            .into_par_iter()
            .map(|poly_idx| {
                let coeffs: Vec<F> = match binding_order {
                    BindingOrder::LowToHigh => (0..new_len)
                        .into_par_iter()
                        .map(|j| {
                            (0..n_total)
                                .map(|offset| {
                                    indices[n_total * j + offset]
                                        .get_index(poly_idx, one_hot_params)
                                        .map_or(F::zero(), |k| tables[offset][poly_idx][k as usize])
                                })
                                .sum()
                        })
                        .collect(),
                    BindingOrder::HighToLow => {
                        let n_bits = n_total.trailing_zeros() as usize;
                        let segment = indices.len() / n_total;
                        (0..new_len)
                            .into_par_iter()
                            .map(|j| {
                                (0..n_total)
                                    .map(|seg| {
                                        let table_idx =
                                            crate::poly::ra_poly::bit_reverse(seg, n_bits);
                                        indices[seg * segment + j]
                                            .get_index(poly_idx, one_hot_params)
                                            .map_or(F::zero(), |k| {
                                                tables[table_idx][poly_idx][k as usize]
                                            })
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

    #[inline]
    fn get_bound_coeff(&self, poly_idx: usize, j: usize) -> F {
        let n_groups = self.n_groups();
        match self.binding_order {
            BindingOrder::HighToLow => {
                let segment = self.indices.len() / n_groups;
                let n_bits = n_groups.trailing_zeros() as usize;
                (0..n_groups)
                    .map(|seg| {
                        let table_idx = crate::poly::ra_poly::bit_reverse(seg, n_bits);
                        self.indices[seg * segment + j]
                            .get_index(poly_idx, &self.one_hot_params)
                            .map_or(F::zero(), |k| self.tables[table_idx][poly_idx][k as usize])
                    })
                    .sum()
            }
            BindingOrder::LowToHigh => (0..n_groups)
                .map(|offset| {
                    self.indices[n_groups * j + offset]
                        .get_index(poly_idx, &self.one_hot_params)
                        .map_or(F::zero(), |k| self.tables[offset][poly_idx][k as usize])
                })
                .sum(),
        }
    }
}
