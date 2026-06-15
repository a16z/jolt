use jolt_field::Field;

use crate::R1csKey;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[derive(Clone, Debug)]
pub struct R1csRowDotTable<F: Field> {
    row_count: usize,
    cycle_count: usize,
    a: Vec<F>,
    b: Vec<F>,
}

#[derive(Clone, Copy, Debug)]
pub struct R1csRowDotSlice<'a, F: Field> {
    pub a: &'a [F],
    pub b: &'a [F],
}

impl<F: Field> R1csRowDotTable<F> {
    #[tracing::instrument(skip_all, name = "R1csRowDotTable::compute_ab_prefix")]
    pub fn compute_ab_prefix(key: &R1csKey<F>, witness: &[F], row_count: usize) -> Self {
        assert!(
            row_count <= key.matrices.num_constraints,
            "row_count exceeds R1CS constraint count"
        );
        let expected = key.num_cycles * key.num_vars_padded;
        assert_eq!(
            witness.len(),
            expected,
            "R1CS witness length does not match key shape"
        );

        let total = key.num_cycles * row_count;
        let mut a = vec![F::zero(); total];
        let mut b = vec![F::zero(); total];
        compute_row_dots(key, witness, row_count, &mut a, &mut b);

        Self {
            row_count,
            cycle_count: key.num_cycles,
            a,
            b,
        }
    }

    #[inline]
    pub fn row_count(&self) -> usize {
        self.row_count
    }

    #[inline]
    pub fn cycle_count(&self) -> usize {
        self.cycle_count
    }

    #[inline]
    pub fn a(&self) -> &[F] {
        &self.a
    }

    #[inline]
    pub fn b(&self) -> &[F] {
        &self.b
    }

    #[inline]
    pub fn cycle(&self, cycle: usize) -> R1csRowDotSlice<'_, F> {
        assert!(cycle < self.cycle_count, "cycle index out of bounds");
        let start = cycle * self.row_count;
        let end = start + self.row_count;
        R1csRowDotSlice {
            a: &self.a[start..end],
            b: &self.b[start..end],
        }
    }
}

#[cfg(feature = "parallel")]
fn compute_row_dots<F: Field>(
    key: &R1csKey<F>,
    witness: &[F],
    row_count: usize,
    a: &mut [F],
    b: &mut [F],
) {
    a.par_chunks_mut(row_count)
        .zip(b.par_chunks_mut(row_count))
        .enumerate()
        .for_each(|(cycle, (a_chunk, b_chunk))| {
            let start = cycle * key.num_vars_padded;
            let witness_row = &witness[start..start + key.matrices.num_vars];
            for row in 0..row_count {
                a_chunk[row] = row_dot(&key.matrices.a[row], witness_row);
                b_chunk[row] = row_dot(&key.matrices.b[row], witness_row);
            }
        });
}

#[cfg(not(feature = "parallel"))]
fn compute_row_dots<F: Field>(
    key: &R1csKey<F>,
    witness: &[F],
    row_count: usize,
    a: &mut [F],
    b: &mut [F],
) {
    for cycle in 0..key.num_cycles {
        let witness_start = cycle * key.num_vars_padded;
        let row_start = cycle * row_count;
        let witness_row = &witness[witness_start..witness_start + key.matrices.num_vars];
        for row in 0..row_count {
            a[row_start + row] = row_dot(&key.matrices.a[row], witness_row);
            b[row_start + row] = row_dot(&key.matrices.b[row], witness_row);
        }
    }
}

#[inline]
fn row_dot<F: Field>(row: &[(usize, F)], witness: &[F]) -> F {
    let mut acc = F::zero();
    for &(variable, coefficient) in row {
        acc += coefficient * witness[variable];
    }
    acc
}
