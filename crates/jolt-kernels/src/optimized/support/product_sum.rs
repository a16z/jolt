//! Sum lower-degree rows before extending their evaluation grid.

use jolt_field::{Accumulator, JoltField};

use super::{
    extend_by_differences, quadratic_next, quadratic_product_samples, quartic_product_samples,
    quintic_product_samples,
};

pub(crate) struct TailFactorProducts<F: JoltField> {
    values: [F; 8],
    products: [F; 256],
}

impl<F: JoltField> TailFactorProducts<F> {
    pub(crate) fn new(values: &[F]) -> Option<Self> {
        let values: [F; 8] = values.try_into().ok()?;
        let mut products = [F::one(); 256];
        for mask in 1usize..256 {
            products[mask] = products[mask & (mask - 1)] * values[mask.trailing_zeros() as usize];
        }
        Some(Self { values, products })
    }
}

#[derive(Default)]
pub(crate) struct LowDegreeProductSum<F: JoltField> {
    lanes: [[F::Accumulator; 7]; 7],
    used: u8,
}

impl<F: JoltField> LowDegreeProductSum<F> {
    #[inline]
    pub(crate) fn try_accumulate(
        &mut self,
        evals: &[F],
        steps: &[F],
        tail: Option<&TailFactorProducts<F>>,
    ) -> bool {
        if evals.len() != 9 {
            return false;
        }
        debug_assert_eq!(steps.len(), evals.len());
        let degree = steps.iter().filter(|step| !step.is_zero()).count();
        if degree > 6 {
            return false;
        }
        let mut values = [F::zero(); 6];
        let mut slopes = [F::zero(); 6];
        let mut active = 0;
        let mut mask = 0usize;
        let mut other: Option<F> = None;
        for (index, (&value, &step)) in evals.iter().zip(steps).enumerate() {
            if step.is_zero() {
                if index != 0 && tail.is_some_and(|tail| value == tail.values[index - 1]) {
                    mask |= 1 << (index - 1);
                } else {
                    other = Some(other.map_or(value, |product| product * value));
                }
            } else {
                values[active] = value;
                slopes[active] = step;
                active += 1;
            }
        }
        let cached = tail.map_or_else(F::one, |tail| tail.products[mask]);
        let constant = match other {
            Some(product) if mask == 0 => product,
            Some(product) => product * cached,
            None => cached,
        };
        if constant.is_zero() {
            return true;
        }
        self.used |= 1 << degree;
        let lanes = &mut self.lanes[degree];
        if degree == 0 {
            lanes[0].add(constant);
            return true;
        }
        values[0] *= constant;
        slopes[0] *= constant;
        let mut samples = [F::zero(); 7];
        match degree {
            1 => samples[..2].copy_from_slice(&[values[0], slopes[0]]),
            2 => samples[..3].copy_from_slice(&quadratic_product_samples(
                [values[0], values[1]],
                [slopes[0], slopes[1]],
            )),
            3 => samples[..4].copy_from_slice(&cubic_product_samples(
                [values[0], values[1], values[2]],
                [slopes[0], slopes[1], slopes[2]],
            )),
            4 => samples[..5].copy_from_slice(&quartic_product_samples(
                [values[0], values[1], values[2], values[3]],
                [slopes[0], slopes[1], slopes[2], slopes[3]],
            )),
            5 => samples[..6].copy_from_slice(&quintic_product_samples(
                [values[0], values[1], values[2], values[3], values[4]],
                [slopes[0], slopes[1], slopes[2], slopes[3], slopes[4]],
            )),
            6 => {
                let left = cubic_product_samples(
                    [values[0], values[1], values[2]],
                    [slopes[0], slopes[1], slopes[2]],
                );
                let right = cubic_product_samples(
                    [values[3], values[4], values[5]],
                    [slopes[3], slopes[4], slopes[5]],
                );
                let mut left_grid = [F::zero(); 6];
                let mut right_grid = [F::zero(); 6];
                left_grid[..3].copy_from_slice(&left[..3]);
                right_grid[..3].copy_from_slice(&right[..3]);
                let mut scratch = [F::zero(); 6];
                extend_by_differences(&mut left_grid, 3, left[3], &mut scratch);
                extend_by_differences(&mut right_grid, 3, right[3], &mut scratch);
                for ((sample, left), right) in samples.iter_mut().zip(left_grid).zip(right_grid) {
                    *sample = left * right;
                }
                samples[6] = left[3] * right[3];
            }
            _ => unreachable!("degree was checked above"),
        }
        for (lane, value) in lanes[..=degree].iter_mut().zip(samples) {
            lane.add(value);
        }
        true
    }

    pub(crate) fn merge_into(self, lanes: &mut [F::Accumulator], scratch: &mut [F]) {
        if lanes.len() != 9 {
            return;
        }
        for (degree, sums) in self.lanes.into_iter().enumerate() {
            if self.used & (1 << degree) == 0 {
                continue;
            }
            let (grid, work) = scratch.split_at_mut(8);
            if degree == 0 {
                grid.fill(sums[0].reduce());
            } else {
                for (value, sum) in grid[..degree].iter_mut().zip(&sums) {
                    *value = sum.reduce();
                }
                extend_by_differences(grid, degree, sums[degree].reduce(), work);
            }
            for (lane, &value) in lanes[..8].iter_mut().zip(grid.iter()) {
                lane.add(value);
            }
        }
    }
}

#[inline]
fn cubic_product_samples<F: JoltField>(values: [F; 3], slopes: [F; 3]) -> [F; 4] {
    let q = quadratic_product_samples([values[0], values[1]], [slopes[0], slopes[1]]);
    let second = values[2] + slopes[2];
    [
        values[2] * q[0],
        second * q[1],
        (second + slopes[2]) * quadratic_next(q),
        slopes[2] * q[2],
    ]
}
