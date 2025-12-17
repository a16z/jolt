use std::sync::Arc;
use std::sync::Mutex;

use allocative::Allocative;
use ark_std::Zero;
use common::constants::REGISTER_COUNT;
use num::Integer;

use crate::field::JoltField;
use crate::field::OptimizedMul;
use crate::poly::multilinear_polynomial::MultilinearPolynomial;
use crate::subprotocols::read_write_matrix::address_major::AddressMajorMatrixEntry;
use crate::subprotocols::read_write_matrix::cycle_major::CycleMajorMatrixEntry;
use crate::subprotocols::read_write_matrix::ReadWriteMatrixAddressMajor;
use crate::subprotocols::read_write_matrix::ReadWriteMatrixCycleMajor;
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
#[derive(Allocative, Debug, PartialEq, Clone, Copy, Default)]
pub struct RegistersCycleMajorEntry<F: JoltField> {
    /// The row index. Before binding, row \in [0, T)
    pub row: usize,
    /// The column index. Before binding, col \in [0, K)
    pub col: u8,
    /// In round i, each ReadWriteEntry represents a coefficient
    ///   Val(k, j', r)
    /// which is some combination of Val(k, j', 00...0), ...
    /// Val(k, j', 11...1).
    /// `prev_val` contains the unbound coefficient before
    /// Val(k, j', 00...0) –– abusing notation, `prev_val` is
    /// Val(k, j'-1, 11...1)
    pub(crate) prev_val: u64,
    /// In round i, each ReadWriteEntry represents a coefficient
    ///   Val(k, j', r)
    /// which is some combination of Val(k, j', 00...0), ...
    /// Val(k, j', 11...1).
    /// `next_val` contains the unbound coefficient after
    /// Val(k, j', 00...0) –– abusing notation, `next_val` is
    /// Val(k, j'+1, 00...0)
    pub(crate) next_val: u64,
    /// The Val coefficient for this matrix entry.
    pub val_coeff: F,
    /// Coefficient for the combined ra polynomial, equal to
    /// gamma * rs1_ra + gamma^2 * rs2_ra
    pub ra_coeff: F,
    pub wa_coeff: F,
}

impl<F: JoltField> ReadWriteMatrixCycleMajor<F, RegistersCycleMajorEntry<F>> {
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
        out: &mut [RegistersCycleMajorEntry<F>],
        gamma: F,
        gamma_squared: F,
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
                ra_coeff: gamma,
                wa_coeff: F::zero(),
            };
            len += 1;
        }

        if let Some((rs2, rs2_val)) = cycle.rs2_read() {
            if let Some(e) = out[..len].iter_mut().find(|e| e.col == rs2) {
                e.ra_coeff += gamma_squared;
            } else {
                out[len] = RegistersCycleMajorEntry {
                    row,
                    col: rs2,
                    prev_val: rs2_val,
                    next_val: rs2_val,
                    val_coeff: F::from_u64(rs2_val),
                    ra_coeff: gamma_squared,
                    wa_coeff: F::zero(),
                };
                len += 1;
            }
        }

        if let Some((rd, rd_pre_val, rd_post_val)) = cycle.rd_write() {
            if let Some(e) = out[..len].iter_mut().find(|e| e.col == rd) {
                // Same register is read and then written this cycle.
                e.wa_coeff = F::one();
                e.next_val = rd_post_val;
            } else {
                out[len] = RegistersCycleMajorEntry {
                    row,
                    col: rd,
                    prev_val: rd_pre_val,
                    next_val: rd_post_val,
                    // val_coeff stores the value *before* any access at this cycle.
                    val_coeff: F::from_u64(rd_pre_val),
                    ra_coeff: F::zero(),
                    wa_coeff: F::one(),
                };
                len += 1;
            }
        }

        debug_assert_eq!(len, out.len());

        // Sort by col for this row; len <= 3 so do a tiny manual sort.
        match len {
            0 | 1 => {}
            2 => {
                if out[0].col > out[1].col {
                    out.swap(0, 1);
                }
            }
            3 => {
                if out[0].col > out[1].col {
                    out.swap(0, 1);
                }
                if out[1].col > out[2].col {
                    out.swap(1, 2);
                }
                if out[0].col > out[1].col {
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
        let mut entries: Vec<RegistersCycleMajorEntry<F>> = Vec::with_capacity(total_entries);
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
            let entries_ptr = entries_ptr as *mut RegistersCycleMajorEntry<F>;
            unsafe {
                let dst = entries_ptr.offset(start);
                let slice = std::slice::from_raw_parts_mut(dst, count);
                Self::fill_entries_for_cycle(j, cycle, slice, gamma, gamma_squared);
            }
        });

        ReadWriteMatrixCycleMajor {
            entries,
            val_init: vec![F::zero(); REGISTER_COUNT as usize].into(),
        }
    }
}

impl<F: JoltField> CycleMajorMatrixEntry<F> for RegistersCycleMajorEntry<F> {
    fn row(&self) -> usize {
        self.row
    }

    fn column(&self) -> usize {
        self.col as usize
    }

    fn bind_entries(even: Option<&Self>, odd: Option<&Self>, r: F::Challenge) -> Self {
        match (even, odd) {
            (Some(even), Some(odd)) => {
                debug_assert!(even.row.is_even());
                debug_assert!(odd.row.is_odd());
                debug_assert_eq!(even.col, odd.col);
                RegistersCycleMajorEntry {
                    row: even.row / 2,
                    col: even.col,
                    ra_coeff: even.ra_coeff + r.mul_01_optimized(odd.ra_coeff - even.ra_coeff),
                    wa_coeff: even.wa_coeff + r.mul_01_optimized(odd.wa_coeff - even.wa_coeff),
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
                    ra_coeff: (F::one() - r).mul_01_optimized(even.ra_coeff),
                    wa_coeff: (F::one() - r).mul_01_optimized(even.wa_coeff),
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
                    ra_coeff: r.mul_01_optimized(odd.ra_coeff),
                    wa_coeff: r.mul_01_optimized(odd.wa_coeff),
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
    ) -> [F::Unreduced<8>; 2] {
        match (even, odd) {
            (Some(even), Some(odd)) => {
                debug_assert!(even.row.is_even());
                debug_assert!(odd.row.is_odd());
                debug_assert_eq!(even.col, odd.col);
                let ra_evals = [even.ra_coeff, odd.ra_coeff - even.ra_coeff];
                let wa_evals = [even.wa_coeff, odd.wa_coeff - even.wa_coeff];
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
                let ra_evals = [even.ra_coeff, -even.ra_coeff];
                let wa_evals = [even.wa_coeff, -even.wa_coeff];
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
                let ra_evals = [F::zero(), odd.ra_coeff];
                let wa_evals = [F::zero(), odd.wa_coeff];
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
}

impl<F: JoltField> ReadWriteMatrixCycleMajor<F, RegistersCycleMajorEntry<F>> {
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
                *ra_p.add(idx) = entry.ra_coeff;
                *wa_p.add(idx) = entry.wa_coeff;
                *val_p.add(idx) = entry.val_coeff;
            }
        });

        [ra.into(), wa.into(), val.into()]
    }
}

/// Represents a non-zero entry in the ra(k, j) and Val(k, j) polynomials.
/// Conceptually, both ra and Val can be seen as K x T matrices.
///
/// # Type Parameters
///
/// - `F`: The field type for coefficients.
#[derive(Allocative, Debug, PartialEq, Clone, Copy, Default)]
pub struct RegistersAddressMajorEntry<F: JoltField> {
    /// The row index. Before binding, row \in [0, T)
    pub row: usize,
    /// The column index. Before binding, col \in [0, K)
    pub col: u8,
    /// In round i, each ReadWriteEntry represents a coefficient
    ///   Val(k, j', r)
    /// which is some combination of Val(k, j', 00...0), ...
    /// Val(k, j', 11...1).
    /// `prev_val` contains the unbound coefficient before
    /// Val(k, j', 00...0) –– abusing notation, `prev_val` is
    /// Val(k, j'-1, 11...1)
    pub(crate) prev_val: F,
    /// In round i, each ReadWriteEntry represents a coefficient
    ///   Val(k, j', r)
    /// which is some combination of Val(k, j', 00...0), ...
    /// Val(k, j', 11...1).
    /// `next_val` contains the unbound coefficient after
    /// Val(k, j', 00...0) –– abusing notation, `next_val` is
    /// Val(k, j'+1, 00...0)
    pub(crate) next_val: F,
    /// The Val coefficient for this matrix entry.
    pub val_coeff: F,
    pub ra_coeff: F,
    pub wa_coeff: F,
}

impl<F: JoltField> From<RegistersCycleMajorEntry<F>> for RegistersAddressMajorEntry<F> {
    fn from(entry: RegistersCycleMajorEntry<F>) -> Self {
        RegistersAddressMajorEntry {
            row: entry.row,
            col: entry.col,
            prev_val: F::from_u64(entry.prev_val),
            next_val: F::from_u64(entry.next_val),
            val_coeff: entry.val_coeff,
            ra_coeff: entry.ra_coeff,
            wa_coeff: entry.wa_coeff,
        }
    }
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
                debug_assert!(even.col.is_even());
                debug_assert!(odd.col.is_odd());
                debug_assert_eq!(even.row, odd.row);
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
                debug_assert!(even.col.is_even());
                debug_assert!(odd.col.is_odd());
                debug_assert_eq!(even.row, odd.row);
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
