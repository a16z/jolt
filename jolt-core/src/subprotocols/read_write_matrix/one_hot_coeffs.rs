use crate::field::JoltField;
use crate::field::OptimizedMul;
use crate::utils::math::Math;
use allocative::Allocative;
use rayon::prelude::*;

#[derive(Allocative, Debug, Default, Clone)]
pub struct OneHotCoeffLookupTable<F: JoltField> {
    /// Grows exponentially with the number of sumcheck rounds
    lookup_table: Vec<F>,
    pub lookup_index_bitwidth: usize,
}

const MAX_LOOKUP_TABLE_SIZE: usize = 1 << 16;

#[derive(Clone, Copy, Default, Allocative)]
pub struct LookupTableIndex(pub u16);

impl<F: JoltField> std::ops::Index<LookupTableIndex> for OneHotCoeffLookupTable<F> {
    type Output = F;

    fn index(&self, index: LookupTableIndex) -> &Self::Output {
        &self.lookup_table[index.0 as usize]
    }
}

impl<F: JoltField> OneHotCoeffLookupTable<F> {
    pub fn new(init_coeffs: Vec<F>) -> Self {
        let table_size = init_coeffs.len();
        debug_assert!(table_size.is_power_of_two());
        Self {
            lookup_table: init_coeffs,
            lookup_index_bitwidth: table_size.log_2(),
        }
    }

    pub fn bind(&mut self, r: F::Challenge) {
        assert!(self.lookup_table.len() < MAX_LOOKUP_TABLE_SIZE);
        // Expand lookup table
        self.lookup_table = self
            .lookup_table
            .par_iter()
            .flat_map(|a| self.lookup_table.par_iter().map(|b| *b + r * (*a - b)))
            .collect();
    }

    pub fn is_saturated(&self) -> bool {
        self.lookup_table.len() >= MAX_LOOKUP_TABLE_SIZE
    }
}

pub trait OneHotCoeff<F: JoltField>: Send + Sync {
    fn bind(
        even: Option<&Self>,
        odd: Option<&Self>,
        r: F::Challenge,
        lookup_table: Option<&OneHotCoeffLookupTable<F>>,
    ) -> Self;

    fn evals(
        even: Option<&Self>,
        odd: Option<&Self>,
        lookup_table: Option<&OneHotCoeffLookupTable<F>>,
    ) -> [F; 2];

    fn to_field(&self, lookup_table: Option<&OneHotCoeffLookupTable<F>>) -> F;
}

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

impl<F: JoltField> OneHotCoeff<F> for LookupTableIndex {
    fn bind(
        even: Option<&Self>,
        odd: Option<&Self>,
        _r: F::Challenge,
        lookup_table: Option<&OneHotCoeffLookupTable<F>>,
    ) -> Self {
        // The lookup table itself is bound to `_r` separately
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
