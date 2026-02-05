//! Lookup table optimization for one-hot coefficient polynomials.
//!
//! Instead of storing full field elements for ra/wa coefficients, we store small indices
//! into a lookup table that grows during sumcheck binding. This saves memory when many
//! entries start from the same small set of possible values (e.g. 0 and 1).

use crate::field::JoltField;
use crate::field::OptimizedMul;
use crate::utils::math::Math;
use allocative::Allocative;
use rayon::prelude::*;

/// Lookup table for one-hot coefficient values that grows during sumcheck.
///
/// Starts with a small set of initial values (e.g., [0, gamma, gamma^2, gamma + gamma^2])
/// and expands by binding with random challenges. Saturates at MAX_LOOKUP_TABLE_SIZE.
#[derive(Allocative, Debug, Default, Clone)]
pub struct OneHotCoeffLookupTable<F: JoltField> {
    /// Grows exponentially with the number of sumcheck rounds
    lookup_table: Vec<F>,
    pub lookup_index_bitwidth: usize,
}

/// Maximum size of lookup table (2^16 entries).
const MAX_LOOKUP_TABLE_SIZE: usize = 1 << 16;

/// Index into a OneHotCoeffLookupTable, stored as u16 to save memory.
#[derive(Clone, Copy, Default, Allocative)]
pub struct LookupTableIndex(pub u16);

impl<F: JoltField> std::ops::Index<LookupTableIndex> for OneHotCoeffLookupTable<F> {
    type Output = F;

    fn index(&self, index: LookupTableIndex) -> &Self::Output {
        &self.lookup_table[index.0 as usize]
    }
}

impl<F: JoltField> OneHotCoeffLookupTable<F> {
    /// Creates a new lookup table with initial coefficient values.
    pub fn new(init_coeffs: Vec<F>) -> Self {
        let table_size = init_coeffs.len();
        debug_assert!(table_size.is_power_of_two());
        Self {
            lookup_table: init_coeffs,
            lookup_index_bitwidth: table_size.log_2(),
        }
    }

    /// Binds the lookup table with challenge `r`, doubling its size.
    ///
    /// Each new entry is computed as: b + r * (a - b) for all pairs (a, b).
    pub fn bind(&mut self, r: F::Challenge) {
        assert!(self.lookup_table.len() < MAX_LOOKUP_TABLE_SIZE);
        // Expand lookup table
        self.lookup_table = self
            .lookup_table
            .par_iter()
            .flat_map(|a| self.lookup_table.par_iter().map(|b| *b + r * (*a - b)))
            .collect();
    }

    /// Returns true if the table has reached maximum size and cannot grow further.
    pub fn is_saturated(&self) -> bool {
        self.lookup_table.len() >= MAX_LOOKUP_TABLE_SIZE
    }
}

/// Trait for coefficient types used in read-write checking matrices.
///
/// Implementors can be either field elements (F) or lookup table indices (LookupTableIndex).
pub trait OneHotCoeff<F: JoltField>: Send + Sync {
    /// Binds a pair of adjacent coefficients together with challenge `r`.
    fn bind(
        even: Option<&Self>,
        odd: Option<&Self>,
        r: F::Challenge,
        lookup_table: Option<&OneHotCoeffLookupTable<F>>,
    ) -> Self;

    /// Computes sumcheck evaluations [f(0), f(1) - f(0)] for a pair of adjacent coefficients.
    fn evals(
        even: Option<&Self>,
        odd: Option<&Self>,
        lookup_table: Option<&OneHotCoeffLookupTable<F>>,
    ) -> [F; 2];

    /// Converts the coefficient to a field element (dereferencing lookup table if needed).
    fn to_field(&self, lookup_table: Option<&OneHotCoeffLookupTable<F>>) -> F;
}

/// Direct field element implementation (no lookup table).
impl<F: JoltField> OneHotCoeff<F> for F {
    fn bind(
        even: Option<&Self>,
        odd: Option<&Self>,
        r: F::Challenge,
        _: Option<&OneHotCoeffLookupTable<F>>,
    ) -> Self {
        match (even, odd) {
            (Some(&even), Some(&odd)) => even + r.mul_0_optimized(odd - even),
            (Some(&even), None) => (F::one() - r).mul_1_optimized(even),
            (None, Some(&odd)) => r.mul_1_optimized(odd),
            (None, None) => panic!("Both entries are None"),
        }
    }

    fn evals(
        even: Option<&Self>,
        odd: Option<&Self>,
        _: Option<&OneHotCoeffLookupTable<F>>,
    ) -> [F; 2] {
        match (even, odd) {
            (Some(&even), Some(&odd)) => [even, odd - even],
            (Some(&even), None) => [even, -even],
            (None, Some(&odd)) => [F::zero(), odd],
            (None, None) => panic!("Both entries are None"),
        }
    }

    fn to_field(&self, _: Option<&OneHotCoeffLookupTable<F>>) -> F {
        *self
    }
}

/// Lookup table index implementation (memory-efficient for repeated values).
impl<F: JoltField> OneHotCoeff<F> for LookupTableIndex {
    fn bind(
        even: Option<&Self>,
        odd: Option<&Self>,
        _r: F::Challenge,
        lookup_table: Option<&OneHotCoeffLookupTable<F>>,
    ) -> Self {
        // The lookup table itself is bound to `_r` separately; we just combine indices
        let lookup_index_bitwidth = lookup_table.unwrap().lookup_table.len().log_2();
        debug_assert!(lookup_index_bitwidth <= 8);

        match (even, odd) {
            (Some(&even), Some(&odd)) => LookupTableIndex(odd.0 << lookup_index_bitwidth | even.0),
            (Some(&even), None) => even,
            (None, Some(&odd)) => LookupTableIndex(odd.0 << lookup_index_bitwidth),
            (None, None) => panic!("Both entries are None"),
        }
    }

    fn evals(
        even: Option<&Self>,
        odd: Option<&Self>,
        lookup_table: Option<&OneHotCoeffLookupTable<F>>,
    ) -> [F; 2] {
        let lookup_table = lookup_table.unwrap();
        match (even, odd) {
            (Some(&even), Some(&odd)) => {
                [lookup_table[even], lookup_table[odd] - lookup_table[even]]
            }
            (Some(&even), None) => [lookup_table[even], -lookup_table[even]],
            (None, Some(&odd)) => [F::zero(), lookup_table[odd]],
            (None, None) => panic!("Both entries are None"),
        }
    }

    fn to_field(&self, lookup_table: Option<&OneHotCoeffLookupTable<F>>) -> F {
        let lookup_table = lookup_table.unwrap();
        lookup_table[*self]
    }
}
