use std::sync::Arc;
use std::sync::Mutex;

use allocative::Allocative;
use ark_std::Zero;
use common::constants::REGISTER_COUNT;
use num::Integer;

use crate::field::JoltField;
use crate::field::OptimizedMul;
use crate::poly::multilinear_polynomial::MultilinearPolynomial;
use crate::poly::split_eq_poly::GruenSplitEqPolynomial;
use crate::poly::unipoly::UniPoly;
use crate::subprotocols::read_write_matrix::address_major::AddressMajorMatrixEntry;
use crate::subprotocols::read_write_matrix::cycle_major::CycleMajorMatrixEntry;
use crate::subprotocols::read_write_matrix::one_hot_coeffs::LookupTableIndex;
use crate::subprotocols::read_write_matrix::one_hot_coeffs::OneHotCoeff;
use crate::subprotocols::read_write_matrix::one_hot_coeffs::OneHotCoeffLookupTable;
use crate::subprotocols::read_write_matrix::ReadWriteMatrixAddressMajor;
use crate::subprotocols::read_write_matrix::ReadWriteMatrixCycleMajor;
use crate::utils::math::Math;
use crate::utils::thread::unsafe_allocate_zero_vec;
use rayon::prelude::*;
use tracer::instruction::Cycle;

/// Represents a non-zero entry in the ra(k, j) and Val(k, j) polynomials.
/// Conceptually, both ra and Val can be seen as K x T matrices.
///
/// # Memory Optimization: `prev_val`/`next_val` as `u64`
///
/// These fields store raw memory values (not bound coefficients) because:
/// - Phase 1 (cycle binding) always starts from initial memory state
/// - We never switch to cycle-major after binding address variables
/// - They're only converted to `F` when needed in arithmetic
///
/// This saves 48 bytes per entry (~35% reduction).
///
/// # Type Parameters
///
/// - `F`: The field type for coefficients.
///
/// Fields are ordered largest-first to minimize padding when `C` is small
/// (e.g. `LookupTableIndex(u16)`): `col: u8` packs into the tail padding
/// alongside `ra_coeff` and `wa_coeff`, keeping the struct at 64 bytes.
#[derive(Allocative, Debug, PartialEq, Clone, Copy, Default)]
pub struct RegistersCycleMajorEntry<F: JoltField, C: OneHotCoeff<F>> {
    /// The Val coefficient for this matrix entry.
    pub val_coeff: F,
    /// In round i, each ReadWriteEntry represents a coefficient
    ///   Val(k, j', r)
    /// which is some combination of Val(k, j', 00...0), ...
    /// Val(k, j', 11...1).
    /// `prev_val` contains the unbound coefficient before
    /// Val(k, j', 00...0) –– abusing notation, `prev_val` is
    /// Val(k, j'-1, 11...1)
    pub(crate) prev_val: u64,
    /// `next_val` contains the unbound coefficient after
    /// Val(k, j', 00...0) –– abusing notation, `next_val` is
    /// Val(k, j'+1, 00...0)
    pub(crate) next_val: u64,
    /// Row index (cycle count, row \in [0, T)).
    row: usize,
    /// Coefficient for the combined ra polynomial, equal to
    /// gamma * rs1_ra + gamma^2 * rs2_ra
    pub ra_coeff: C,
    pub wa_coeff: C,
    /// Column index (register index, col \in [0, K), K=128).
    col: u8,
}

impl<F: JoltField> ReadWriteMatrixCycleMajor<F, RegistersCycleMajorEntry<F, LookupTableIndex>> {
    /// Count how many distinct registers this cycle touches (0–3).
    #[inline]
    fn entry_count_for_cycle(cycle: &Cycle) -> u8 {
        let mut regs: [Option<_>; 3] = [None, None, None];
        let mut len = 0;

        if let Some((rs1, _)) = cycle.rs1_read() {
            if !regs[..len].contains(&Some(rs1)) {
                regs[len] = Some(rs1);
                len += 1;
            }
        }
        if let Some((rs2, _)) = cycle.rs2_read() {
            if !regs[..len].contains(&Some(rs2)) {
                regs[len] = Some(rs2);
                len += 1;
            }
        }
        if let Some((rd, ..)) = cycle.rd_write() {
            if !regs[..len].contains(&Some(rd)) {
                regs[len] = Some(rd);
                len += 1;
            }
        }

        len as u8
    }

    /// Fill the per-cycle entries into `out` (length 0–3), sorted by `col`.
    #[inline]
    fn fill_entries_for_cycle(
        row: usize,
        cycle: &Cycle,
        out: &mut [RegistersCycleMajorEntry<F, LookupTableIndex>],
    ) {
        debug_assert!(out.len() <= 3);
        let mut len = 0usize;

        if let Some((rs1, rs1_val)) = cycle.rs1_read() {
            out[len] = RegistersCycleMajorEntry {
                row,
                col: rs1,
                prev_val: rs1_val,
                next_val: rs1_val,
                val_coeff: F::from_u64(rs1_val),
                ra_coeff: LookupTableIndex(1),
                wa_coeff: LookupTableIndex(0),
            };
            len += 1;
        }

        if let Some((rs2, rs2_val)) = cycle.rs2_read() {
            if let Some(e) = out[..len].iter_mut().find(|e| e.column() as u8 == rs2) {
                e.ra_coeff = LookupTableIndex(3); // rs1_ra = rs2_ra = 1
            } else {
                out[len] = RegistersCycleMajorEntry {
                    row,
                    col: rs2,
                    prev_val: rs2_val,
                    next_val: rs2_val,
                    val_coeff: F::from_u64(rs2_val),
                    ra_coeff: LookupTableIndex(2),
                    wa_coeff: LookupTableIndex(0),
                };
                len += 1;
            }
        }

        if let Some((rd, rd_pre_val, rd_post_val)) = cycle.rd_write() {
            if let Some(e) = out[..len].iter_mut().find(|e| e.column() as u8 == rd) {
                // Same register is read and then written this cycle.
                e.wa_coeff = LookupTableIndex(1);
                e.next_val = rd_post_val;
            } else {
                out[len] = RegistersCycleMajorEntry {
                    row,
                    col: rd,
                    prev_val: rd_pre_val,
                    next_val: rd_post_val,
                    // val_coeff stores the value *before* any access at this cycle.
                    val_coeff: F::from_u64(rd_pre_val),
                    ra_coeff: LookupTableIndex(0),
                    wa_coeff: LookupTableIndex(1),
                };
                len += 1;
            }
        }

        debug_assert_eq!(len, out.len());

        // Sort by col for this row; len <= 3 so do a tiny manual sort.
        match len {
            0 | 1 => {}
            2 => {
                if out[0].column() > out[1].column() {
                    out.swap(0, 1);
                }
            }
            3 => {
                if out[0].column() > out[1].column() {
                    out.swap(0, 1);
                }
                if out[1].column() > out[2].column() {
                    out.swap(1, 2);
                }
                if out[0].column() > out[1].column() {
                    out.swap(0, 1);
                }
            }
            _ => unreachable!(),
        }
    }

    /// Creates a new `ReadWriteMatrixCycleMajor` to represent the ra, wa and Val polynomials
    /// for the registers read/write checking sumcheck.
    #[tracing::instrument(skip_all, name = "ReadWriteMatrixCycleMajor::new")]
    pub fn new(trace: &[Cycle], gamma: F) -> Self {
        // ---- Pass 1: per-cycle entry counts (parallel) ----
        let counts: Vec<u8> = trace
            .par_iter()
            .map(|cycle| Self::entry_count_for_cycle(cycle))
            .collect();

        // ---- Prefix sum: counts -> offsets (sequential, linear) ----
        let mut offsets: Vec<usize> = Vec::with_capacity(counts.len() + 1);
        offsets.push(0);
        let mut total: usize = 0;
        for &c in &counts {
            total += c as usize;
            offsets.push(total);
        }
        let total_entries = total;

        // ---- Allocate entries and set_len unsafely; we'll fill everything in pass 2 ----
        let mut entries: Vec<RegistersCycleMajorEntry<F, LookupTableIndex>> =
            Vec::with_capacity(total_entries);
        unsafe {
            entries.set_len(total_entries);
        }
        let entries_ptr = entries.as_mut_ptr() as usize;

        // ---- Pass 2: fill entries in parallel, disjoint slices per row ----
        let gamma_squared = gamma.square();
        trace.par_iter().enumerate().for_each(|(j, cycle)| {
            let count = counts[j] as usize;
            if count == 0 {
                return;
            }

            let start = offsets[j] as isize;
            let entries_ptr = entries_ptr as *mut RegistersCycleMajorEntry<F, LookupTableIndex>;
            unsafe {
                let dst = entries_ptr.offset(start);
                let slice = std::slice::from_raw_parts_mut(dst, count);
                Self::fill_entries_for_cycle(j, cycle, slice);
            }
        });

        ReadWriteMatrixCycleMajor {
            entries,
            ra_lookup_table: Some(OneHotCoeffLookupTable::new(vec![
                F::zero(),             // rs1_ra = 0, rs2_ra = 0
                gamma,                 // rs1_ra = 1, rs2_ra = 0
                gamma_squared,         // rs1_ra = 0, rs2_ra = 1
                gamma + gamma_squared, // rs1_ra = 1, rs2_ra = 1
            ])),
            wa_lookup_table: Some(OneHotCoeffLookupTable::new(vec![F::zero(), F::one()])),
            val_init: vec![F::zero(); REGISTER_COUNT as usize].into(),
        }
    }

    pub fn deref_coeffs(self) -> ReadWriteMatrixCycleMajor<F, RegistersCycleMajorEntry<F, F>> {
        let val_init = self.val_init;
        let ra_lookup_table = self.ra_lookup_table.unwrap();
        let wa_lookup_table = self.wa_lookup_table.unwrap();
        let entries = self
            .entries
            .into_par_iter()
            .map(|entry| RegistersCycleMajorEntry {
                row: entry.row,
                col: entry.col,
                prev_val: entry.prev_val,
                next_val: entry.next_val,
                val_coeff: entry.val_coeff,
                ra_coeff: ra_lookup_table[entry.ra_coeff],
                wa_coeff: wa_lookup_table[entry.wa_coeff],
            })
            .collect();

        ReadWriteMatrixCycleMajor {
            entries,
            ra_lookup_table: None,
            wa_lookup_table: None,
            val_init,
        }
    }
}

impl<F: JoltField, C: OneHotCoeff<F>> CycleMajorMatrixEntry<F> for RegistersCycleMajorEntry<F, C> {
    type AddressMajor = RegistersAddressMajorEntry<F>;

    fn row(&self) -> usize {
        self.row
    }

    fn column(&self) -> usize {
        self.col as usize
    }

    fn bind_entries(
        even: Option<&Self>,
        odd: Option<&Self>,
        r: F::Challenge,
        ra_lookup_table: Option<&OneHotCoeffLookupTable<F>>,
        wa_lookup_table: Option<&OneHotCoeffLookupTable<F>>,
    ) -> Self {
        match (even, odd) {
            (Some(even), Some(odd)) => {
                debug_assert!(even.row().is_even());
                debug_assert!(odd.row().is_odd());
                debug_assert_eq!(even.column(), odd.column());
                RegistersCycleMajorEntry {
                    row: even.row / 2,
                    col: even.col,
                    ra_coeff: OneHotCoeff::bind(
                        Some(&even.ra_coeff),
                        Some(&odd.ra_coeff),
                        r,
                        ra_lookup_table,
                    ),
                    wa_coeff: OneHotCoeff::bind(
                        Some(&even.wa_coeff),
                        Some(&odd.wa_coeff),
                        r,
                        wa_lookup_table,
                    ),
                    val_coeff: even.val_coeff + r.mul_0_optimized(odd.val_coeff - even.val_coeff),
                    prev_val: even.prev_val,
                    next_val: odd.next_val,
                }
            }
            (Some(even), None) => {
                // For ReadWriteMatrixCycleMajor, the absence of a matrix entry implies
                // that its coeff has not been bound yet.
                // The absence of an odd-row entry in the same column as even
                // means that its implicit Val coeff is even.next_val, and its implicit
                // ra/wa coeffs are 0.
                let odd_val_coeff = F::from_u64(even.next_val);
                RegistersCycleMajorEntry {
                    row: even.row / 2,
                    col: even.col,
                    ra_coeff: OneHotCoeff::bind(Some(&even.ra_coeff), None, r, ra_lookup_table),
                    wa_coeff: OneHotCoeff::bind(Some(&even.wa_coeff), None, r, wa_lookup_table),
                    val_coeff: even.val_coeff + r.mul_0_optimized(odd_val_coeff - even.val_coeff),
                    prev_val: even.prev_val,
                    next_val: even.next_val,
                }
            }
            (None, Some(odd)) => {
                // For ReadWriteMatrixCycleMajor, the absence of a matrix entry implies
                // that its coeff has not been bound yet.
                // The absence of an even-row entry in the same column as odd
                // means that its implicit Val coeff is odd.prev_val, and its implicit
                // ra coeff is 0.
                let even_val_coeff = F::from_u64(odd.prev_val);
                RegistersCycleMajorEntry {
                    row: odd.row / 2,
                    col: odd.col,
                    ra_coeff: OneHotCoeff::bind(None, Some(&odd.ra_coeff), r, ra_lookup_table),
                    wa_coeff: OneHotCoeff::bind(None, Some(&odd.wa_coeff), r, wa_lookup_table),
                    val_coeff: even_val_coeff + r.mul_0_optimized(odd.val_coeff - even_val_coeff),
                    prev_val: odd.prev_val,
                    next_val: odd.next_val,
                }
            }
            (None, None) => panic!("Both entries are None"),
        }
    }

    fn compute_evals(
        even: Option<&Self>,
        odd: Option<&Self>,
        inc_evals: [F; 2],
        _gamma: F,
        ra_lookup_table: Option<&OneHotCoeffLookupTable<F>>,
        wa_lookup_table: Option<&OneHotCoeffLookupTable<F>>,
    ) -> [F::Unreduced<8>; 2] {
        match (even, odd) {
            (Some(even), Some(odd)) => {
                debug_assert!(even.row().is_even());
                debug_assert!(odd.row().is_odd());
                debug_assert_eq!(even.column(), odd.column());
                let ra_evals =
                    OneHotCoeff::evals(Some(&even.ra_coeff), Some(&odd.ra_coeff), ra_lookup_table);
                let wa_evals =
                    OneHotCoeff::evals(Some(&even.wa_coeff), Some(&odd.wa_coeff), wa_lookup_table);
                let val_evals = [even.val_coeff, odd.val_coeff - even.val_coeff];
                [
                    ra_evals[0].mul_unreduced::<8>(val_evals[0])
                        + wa_evals[0].mul_unreduced::<8>(val_evals[0] + inc_evals[0]),
                    ra_evals[1].mul_unreduced::<8>(val_evals[1])
                        + wa_evals[1].mul_unreduced::<8>(val_evals[1] + inc_evals[1]),
                ]
            }
            (Some(even), None) => {
                let odd_val_coeff = F::from_u64(even.next_val);
                let ra_evals = OneHotCoeff::evals(Some(&even.ra_coeff), None, ra_lookup_table);
                let wa_evals = OneHotCoeff::evals(Some(&even.wa_coeff), None, wa_lookup_table);
                let val_evals = [even.val_coeff, odd_val_coeff - even.val_coeff];
                [
                    ra_evals[0].mul_unreduced::<8>(val_evals[0])
                        + wa_evals[0].mul_unreduced::<8>(val_evals[0] + inc_evals[0]),
                    ra_evals[1].mul_unreduced::<8>(val_evals[1])
                        + wa_evals[1].mul_unreduced::<8>(val_evals[1] + inc_evals[1]),
                ]
            }
            (None, Some(odd)) => {
                let even_val_coeff = F::from_u64(odd.prev_val);
                let ra_evals = OneHotCoeff::evals(None, Some(&odd.ra_coeff), ra_lookup_table);
                let wa_evals = OneHotCoeff::evals(None, Some(&odd.wa_coeff), wa_lookup_table);
                let val_evals = [even_val_coeff, odd.val_coeff - even_val_coeff];
                [
                    F::Unreduced::zero(),
                    ra_evals[1].mul_unreduced::<8>(val_evals[1])
                        + wa_evals[1].mul_unreduced::<8>(val_evals[1] + inc_evals[1]),
                ]
            }
            (None, None) => panic!("Both entries are None"),
        }
    }

    fn to_address_major(
        self,
        ra_lookup_table: Option<&OneHotCoeffLookupTable<F>>,
        wa_lookup_table: Option<&OneHotCoeffLookupTable<F>>,
    ) -> Self::AddressMajor {
        RegistersAddressMajorEntry {
            row: self.row,
            col: self.col,
            prev_val: F::from_u64(self.prev_val),
            next_val: F::from_u64(self.next_val),
            val_coeff: self.val_coeff,
            ra_coeff: self.ra_coeff.to_field(ra_lookup_table),
            wa_coeff: self.wa_coeff.to_field(wa_lookup_table),
        }
    }
}

impl<F: JoltField, C: OneHotCoeff<F>> ReadWriteMatrixCycleMajor<F, RegistersCycleMajorEntry<F, C>> {
    /// Materializes the ra, wa and Val polynomials represented by this `ReadWriteMatrixCycleMajor`.
    ///
    /// After partial binding of cycle and address variables, there are `K_prime` columns
    /// (remaining address positions) and `T_prime` rows (remaining cycle positions) in the matrix.
    /// This expands the sparse representation to dense polynomials of size `K_prime * T_prime`.
    ///
    /// The output layout is address-major: index(addr k, cycle t) = k * T_prime + t.
    /// This matches the layout expected by `phase3_compute_message`.
    ///
    /// When `T_prime == 1` (all cycle variables bound), this is equivalent to the simple case
    /// where each entry has `row == 0` and entries have distinct `col` values.
    #[tracing::instrument(skip_all, name = "ReadWriteMatrixCycleMajor::materialize")]
    pub fn materialize(self, K_prime: usize, T_prime: usize) -> [MultilinearPolynomial<F>; 3] {
        let len = K_prime * T_prime;

        // Initialize polynomials to zero
        // Layout: address-major, index(k, t) = k * T_prime + t
        let mut ra: Vec<F> = unsafe_allocate_zero_vec(len);
        let mut wa: Vec<F> = unsafe_allocate_zero_vec(len);
        let mut val: Vec<F> = unsafe_allocate_zero_vec(len);

        // Update ra, wa and val at positions where we have entries.
        // Index is col * T_prime + row (address-major layout).
        let ra_ptr = ra.as_mut_ptr() as usize;
        let wa_ptr = wa.as_mut_ptr() as usize;
        let val_ptr = val.as_mut_ptr() as usize;

        self.entries.into_par_iter().for_each(|entry| {
            debug_assert!(
                entry.row() < T_prime,
                "row {} >= T_prime {T_prime}",
                entry.row()
            );
            let idx = entry.column() * T_prime + entry.row();
            // SAFETY: Each entry has a unique (row, col) pair,
            // so writes to ra[idx] and val[idx] are disjoint across parallel iterations.
            unsafe {
                let ra_p = ra_ptr as *mut F;
                let wa_p = wa_ptr as *mut F;
                let val_p = val_ptr as *mut F;
                *ra_p.add(idx) = entry.ra_coeff.to_field(self.ra_lookup_table.as_ref());
                *wa_p.add(idx) = entry.wa_coeff.to_field(self.wa_lookup_table.as_ref());
                *val_p.add(idx) = entry.val_coeff;
            }
        });

        [ra.into(), wa.into(), val.into()]
    }

    pub fn compute_message(
        &self,
        inc: &MultilinearPolynomial<F>,
        gruen_eq: &GruenSplitEqPolynomial<F>,
        gamma: F,
        previous_claim: F,
    ) -> UniPoly<F> {
        // Compute quadratic coefficients using Gruen's optimization.
        // When E_in is fully bound (len <= 1), we use E_in_eval = 1 and num_x_in_bits = 0,
        // which makes the outer chunking degenerate to row pairs and skips the inner sum.
        let e_in = gruen_eq.E_in_current();
        let e_in_len = e_in.len();
        let num_x_in_bits = e_in_len.max(1).log_2(); // max(1) so log_2 of 0 or 1 gives 0
        let x_bitmask = (1 << num_x_in_bits) - 1;

        let quadratic_coeffs: [F; 2] = self
            .entries
            // Chunk by x_out (when E_in is bound, this is just row pairs)
            .par_chunk_by(|a, b| ((a.row() / 2) >> num_x_in_bits) == ((b.row() / 2) >> num_x_in_bits))
            .map(|entries| {
                let x_out = (entries[0].row() / 2) >> num_x_in_bits;
                let E_out_eval = gruen_eq.E_out_current()[x_out];

                let outer_sum_evals = entries
                    .par_chunk_by(|a, b| a.row() / 2 == b.row() / 2)
                    .map(|entries| {
                        let odd_row_start_index = entries.partition_point(|entry| entry.row().is_even());
                        let (even_row, odd_row) = entries.split_at(odd_row_start_index);
                        let j_prime = 2 * (entries[0].row() / 2);

                        // When E_in is fully bound, x_in = 0 and E_in_eval = 1
                        let E_in_eval = if e_in_len <= 1 {
                            F::one()
                        } else {
                            let x_in = (j_prime / 2) & x_bitmask;
                            e_in[x_in]
                        };

                        let inc_evals = {
                            let inc_0 = inc.get_bound_coeff(j_prime);
                            let inc_1 = inc.get_bound_coeff(j_prime + 1);
                            let inc_infty = inc_1 - inc_0;
                            [inc_0, inc_infty]
                        };

                        let inner_sum_evals = self.prover_message_contribution(
                            even_row,
                            odd_row,
                            inc_evals,
                            gamma,
                        );

                        [
                            E_in_eval.mul_unreduced::<9>(inner_sum_evals[0]),
                            E_in_eval.mul_unreduced::<9>(inner_sum_evals[1]),
                        ]
                    })
                    .reduce(
                        || [F::Unreduced::<9>::zero(); 2],
                        |running, new| [running[0] + new[0], running[1] + new[1]],
                    )
                    .map(F::from_montgomery_reduce);

                [
                    E_out_eval.mul_unreduced::<9>(outer_sum_evals[0]),
                    E_out_eval.mul_unreduced::<9>(outer_sum_evals[1]),
                ]
            })
            .reduce(
                || [F::Unreduced::<9>::zero(); 2],
                |running, new| [running[0] + new[0], running[1] + new[1]],
            )
            .map(F::from_montgomery_reduce);

        // Convert quadratic coefficients to cubic evaluations
        gruen_eq.gruen_poly_deg_3(quadratic_coeffs[0], quadratic_coeffs[1], previous_claim)
    }
}

/// Represents a non-zero entry in the ra(k, j) and Val(k, j) polynomials.
/// Conceptually, both ra and Val can be seen as K x T matrices.
///
#[derive(Allocative, Debug, PartialEq, Clone, Copy, Default)]
pub struct RegistersAddressMajorEntry<F: JoltField> {
    /// `prev_val` contains the unbound coefficient before
    /// Val(k, j', 00...0) –– abusing notation, `prev_val` is
    /// Val(k, j'-1, 11...1)
    pub(crate) prev_val: F,
    /// `next_val` contains the unbound coefficient after
    /// Val(k, j', 00...0) –– abusing notation, `next_val` is
    /// Val(k, j'+1, 00...0)
    pub(crate) next_val: F,
    /// The Val coefficient for this matrix entry.
    pub val_coeff: F,
    pub ra_coeff: F,
    pub wa_coeff: F,
    /// Row index (cycle count, row \in [0, T)).
    row: usize,
    /// Column index (register index, col \in [0, K), K=128).
    col: u8,
}

impl<F: JoltField> AddressMajorMatrixEntry<F> for RegistersAddressMajorEntry<F> {
    fn row(&self) -> usize {
        self.row
    }

    fn column(&self) -> usize {
        self.col as usize
    }

    fn prev_val(&self) -> F {
        self.prev_val
    }

    fn next_val(&self) -> F {
        self.next_val
    }

    fn bind_entries(
        even: Option<&Self>,
        odd: Option<&Self>,
        even_checkpoint: F,
        odd_checkpoint: F,
        r: F::Challenge,
    ) -> Self {
        match (even, odd) {
            (Some(even), Some(odd)) => {
                debug_assert!(even.column().is_even());
                debug_assert!(odd.column().is_odd());
                debug_assert_eq!(even.row(), odd.row());
                RegistersAddressMajorEntry {
                    row: even.row,
                    col: even.col / 2,
                    ra_coeff: even.ra_coeff + r.mul_01_optimized(odd.ra_coeff - even.ra_coeff),
                    wa_coeff: even.wa_coeff + r.mul_01_optimized(odd.wa_coeff - even.wa_coeff),
                    val_coeff: even.val_coeff + r.mul_0_optimized(odd.val_coeff - even.val_coeff),
                    prev_val: even.prev_val + r.mul_0_optimized(odd.prev_val - even.prev_val),
                    next_val: even.next_val + r.mul_0_optimized(odd.next_val - even.next_val),
                }
            }
            (Some(even), None) => {
                // For SparseMatrixPolynomial, the absence of a matrix entry implies
                // that its coeff has not been bound yet.
                // The absence of an odd-col entry in the same row as even
                // means that its implicit Val coeff is odd_checkpoint, and its implicit
                // ra coeff is 0.
                RegistersAddressMajorEntry {
                    row: even.row,
                    col: even.col / 2,
                    ra_coeff: (F::one() - r).mul_01_optimized(even.ra_coeff),
                    wa_coeff: (F::one() - r).mul_01_optimized(even.wa_coeff),
                    val_coeff: even.val_coeff + r.mul_0_optimized(odd_checkpoint - even.val_coeff),
                    prev_val: even.prev_val + r.mul_0_optimized(odd_checkpoint - even.prev_val),
                    next_val: even.next_val + r.mul_0_optimized(odd_checkpoint - even.next_val),
                }
            }
            (None, Some(odd)) => {
                // For SparseMatrixPolynomial, the absence of a matrix entry implies
                // that its coeff has not been bound yet.
                // The absence of an even-col entry in the same row as odd
                // means that its implicit Val coeff is even_checkpoint, and its implicit
                // ra coeff is 0.
                RegistersAddressMajorEntry {
                    row: odd.row,
                    col: odd.col / 2,
                    ra_coeff: r.mul_01_optimized(odd.ra_coeff),
                    wa_coeff: r.mul_01_optimized(odd.wa_coeff),
                    val_coeff: even_checkpoint + r.mul_0_optimized(odd.val_coeff - even_checkpoint),
                    prev_val: even_checkpoint + r.mul_0_optimized(odd.prev_val - even_checkpoint),
                    next_val: even_checkpoint + r.mul_0_optimized(odd.next_val - even_checkpoint),
                }
            }
            (None, None) => panic!("Both entries are None"),
        }
    }

    fn compute_evals(
        even: Option<&Self>,
        odd: Option<&Self>,
        even_checkpoint: F,
        odd_checkpoint: F,
        inc_eval: F,
        eq_eval: F,
        _gamma: F,
    ) -> [F::Unreduced<8>; 2] {
        match (even, odd) {
            (Some(even), Some(odd)) => {
                debug_assert!(even.column().is_even());
                debug_assert!(odd.column().is_odd());
                debug_assert_eq!(even.row(), odd.row());
                let ra_evals = [even.ra_coeff, odd.ra_coeff + odd.ra_coeff - even.ra_coeff];
                let wa_evals = [even.wa_coeff, odd.wa_coeff + odd.wa_coeff - even.wa_coeff];
                let val_evals = [
                    even.val_coeff,
                    odd.val_coeff + odd.val_coeff - even.val_coeff,
                ];
                [
                    eq_eval.mul_unreduced(
                        ra_evals[0] * val_evals[0] + wa_evals[0] * (val_evals[0] + inc_eval),
                    ),
                    eq_eval.mul_unreduced(
                        ra_evals[1] * val_evals[1] + wa_evals[1] * (val_evals[1] + inc_eval),
                    ),
                ]
            }
            (Some(even), None) => {
                // For SparseMatrixPolynomial, the absence of a matrix entry implies
                // that its coeff has not been bound yet.
                // The absence of an odd-row entry in the same column as even
                // means that its implicit Val coeff is odd_checkpoint, and its implicit
                // ra coeff is 0.
                let ra_evals = [even.ra_coeff, -even.ra_coeff];
                let wa_evals = [even.wa_coeff, -even.wa_coeff];
                let val_evals = [
                    even.val_coeff,
                    odd_checkpoint + odd_checkpoint - even.val_coeff,
                ];
                [
                    eq_eval.mul_unreduced(
                        ra_evals[0] * val_evals[0] + wa_evals[0] * (val_evals[0] + inc_eval),
                    ),
                    eq_eval.mul_unreduced(
                        ra_evals[1] * val_evals[1] + wa_evals[1] * (val_evals[1] + inc_eval),
                    ),
                ]
            }
            (None, Some(odd)) => {
                // For SparseMatrixPolynomial, the absence of a matrix entry implies
                // that its coeff has not been bound yet.
                // The absence of an even-row entry in the same column as odd
                // means that its implicit Val coeff is even_checkpoint, and its implicit
                // ra coeff is 0.
                let ra_evals = [F::zero(), odd.ra_coeff + odd.ra_coeff];
                let wa_evals = [F::zero(), odd.wa_coeff + odd.wa_coeff];
                let val_evals = [
                    even_checkpoint,
                    odd.val_coeff + odd.val_coeff - even_checkpoint,
                ];
                [
                    F::Unreduced::<8>::zero(), // ra_evals[0] is zero
                    eq_eval.mul_unreduced(
                        ra_evals[1] * val_evals[1] + wa_evals[1] * (val_evals[1] + inc_eval),
                    ),
                ]
            }
            (None, None) => panic!("Both entries are None"),
        }
    }
}

impl<F: JoltField> ReadWriteMatrixAddressMajor<F, RegistersAddressMajorEntry<F>> {
    /// Materializes the ra and Val polynomials represented by this `ReadWriteMatrixAddressMajor`.
    /// Some number of cycle and address variables have already been bound, so at this point
    /// there are `K_prime` columns and `T_prime` rows left in the matrix.
    #[tracing::instrument(skip_all, name = "ReadWriteMatrixAddressMajor::materialize")]
    pub fn materialize(self, K_prime: usize, T_prime: usize) -> [MultilinearPolynomial<F>; 3] {
        // Initialize ra, wa and Val to initial values
        let ra: Vec<Arc<Mutex<F>>> = (0..K_prime * T_prime)
            .into_par_iter()
            .map(|_| Arc::new(Mutex::new(F::zero())))
            .collect();
        let wa: Vec<Arc<Mutex<F>>> = (0..K_prime * T_prime)
            .into_par_iter()
            .map(|_| Arc::new(Mutex::new(F::zero())))
            .collect();
        let val: Vec<Arc<Mutex<F>>> = (0..K_prime * T_prime)
            .into_par_iter()
            .map(|_| Arc::new(Mutex::new(F::zero())))
            .collect();

        // Update some of the ra, wa and Val coefficients based on
        // matrix entries.
        self.entries
            .par_chunk_by(|a, b| a.column() == b.column())
            .for_each(|column| {
                let k = column[0].column();
                let mut current_val_coeff = F::zero();
                let mut column_iter = column.iter().peekable();
                for j in 0..T_prime {
                    let idx = k * T_prime + j;
                    if let Some(entry) = column_iter.next_if(|&entry| entry.row() == j) {
                        *ra[idx].lock().unwrap() = entry.ra_coeff;
                        *wa[idx].lock().unwrap() = entry.wa_coeff;
                        *val[idx].lock().unwrap() = entry.val_coeff;
                        current_val_coeff = entry.next_val();
                        continue;
                    }
                    *val[idx].lock().unwrap() = current_val_coeff;
                    continue;
                }
            });
        // Unwrap Arc<Mutex<F>> back into F
        let ra: Vec<F> = ra
            .into_par_iter()
            .map(|arc_mutex| *arc_mutex.lock().unwrap())
            .collect();
        let wa: Vec<F> = wa
            .into_par_iter()
            .map(|arc_mutex| *arc_mutex.lock().unwrap())
            .collect();
        let val: Vec<F> = val
            .into_par_iter()
            .map(|arc_mutex| *arc_mutex.lock().unwrap())
            .collect();
        // Convert Vec<F> to MultilinearPolynomial<F>
        [ra.into(), wa.into(), val.into()]
    }
}
