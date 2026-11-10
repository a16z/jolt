use super::EqPolynomial;
use crate::point::PointLike;
use jolt_field::Field;
use jolt_utils::Math;
use std::ops::{Mul, SubAssign};
impl<F: Field> EqPolynomial<F> {
    /// `result[j]` is the eq table over the `j` most-significant variables.
    ///
    /// Returns `n+1` tables where `result[0] = [scaling_factor]` (eq over 0 vars).
    #[tracing::instrument(skip_all, name = "EqPolynomial::evals_cached_msb")]
    pub fn evals_cached_msb<P: PointLike + ?Sized>(r: &P, scaling_factor: Option<F>) -> Vec<Vec<F>>
    where
        P::Coord: Send + Sync + Into<F>,
        F: Mul<P::Coord, Output = F> + SubAssign<F>,
    {
        let mut evals: Vec<Vec<F>> = (0..=r.num_vars())
            .map(|i| vec![scaling_factor.unwrap_or(F::one()); 1 << i])
            .collect();
        Self::fill_cached::<_, true>(r, &mut evals);
        evals
    }

    /// `result[j]` is the eq table over the `j` least-significant variables.
    ///
    /// `result[0] = [scaling_factor]`. Each new variable is the new MSB.
    pub fn evals_cached_lsb<P: PointLike + ?Sized>(r: &P, scaling_factor: Option<F>) -> Vec<Vec<F>>
    where
        P::Coord: Send + Sync + Into<F>,
        F: Mul<P::Coord, Output = F>,
    {
        let mut evals: Vec<Vec<F>> = (0..=r.num_vars())
            .map(|i| vec![scaling_factor.unwrap_or(F::one()); 1 << i])
            .collect();
        Self::fill_cached::<_, false>(r, &mut evals);
        evals
    }

    #[inline]
    pub(crate) fn fill_reserved_cached<P, const MSB: bool>(r: &P, tables: &mut [Vec<F>])
    where
        P: PointLike + ?Sized,
        P::Coord: Send + Sync + Into<F>,
        F: Mul<P::Coord, Output = F>,
    {
        let scale = tables[0][0];
        for (level, table) in tables.iter_mut().enumerate() {
            let len = level.pow2();
            // try_cached_buffers reserves every leaf before a fill is scheduled.
            debug_assert!(table.capacity() >= len);
            table.resize(len, scale);
        }
        Self::fill_cached::<_, MSB>(r, tables);
    }

    #[inline]
    pub(crate) fn fill_cached<P, const MSB: bool>(r: &P, evals: &mut [Vec<F>])
    where
        P: PointLike + ?Sized,
        P::Coord: Send + Sync + Into<F>,
        F: Mul<P::Coord, Output = F>,
    {
        if MSB {
            let mut size = 1;
            for j in 0..r.num_vars() {
                let r_j = r.msb(j);
                size *= 2;
                for i in (0..size).rev().step_by(2) {
                    let scalar = evals[j][i / 2];
                    evals[j + 1][i] = scalar * r_j;
                    evals[j + 1][i - 1] = scalar - evals[j + 1][i];
                }
            }
        } else {
            let mut size = 1;
            for j in 0..r.num_vars() {
                let r_j = r.msb(r.num_vars() - 1 - j);
                for i in 0..size {
                    let scalar = evals[j][i];
                    let multiple = 1 << j;
                    evals[j + 1][i + multiple] = scalar * r_j;
                    evals[j + 1][i] = scalar - evals[j + 1][i + multiple];
                }
                size *= 2;
            }
        }
    }
}
