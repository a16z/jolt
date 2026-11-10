use super::alloc::{check_layout, domain_len, EqAlloc, Global, TableInit};
use super::boolean_bits_msb;
use super::EqPolynomial;
use crate::point::{msb_index, Endianness, PointLike};
#[cfg(feature = "parallel")]
use crate::{PointRef, HIGH_TO_LOW};
use jolt_field::Field;
use jolt_utils::Math;
use std::ops::{Mul, SubAssign};
use std::panic::Location;

pub(super) const SERIAL_MAX_VARS: usize = 16;

#[cfg(feature = "parallel")]
const PARALLEL_SLAB_MIN_VARS: usize = 14;
#[cfg(feature = "parallel")]
const PARALLEL_SLAB_MAX_VARS: usize = 16;
#[cfg(feature = "parallel")]
const MINIMUM_SLAB_VARS: usize = 8;

impl<F: Field> EqPolynomial<F> {
    /// Computes `eq(x, y) = Π_i (x_i y_i + (1 - x_i)(1 - y_i))` for two slices.
    pub fn mle<C>(x: &[C], y: &[C]) -> F
    where
        C: Copy + Send + Sync + Into<F>,
        F: Mul<C, Output = F> + SubAssign<F>,
    {
        assert_eq!(x.len(), y.len());
        x.iter()
            .zip(y.iter())
            .map(|(x_i, y_i)| {
                let x: F = (*x_i).into();
                let y: F = (*y_i).into();
                x * y + (F::one() - x) * (F::one() - y)
            })
            .fold(F::one(), |acc, v| acc * v)
    }

    /// Computes `{ eq(r, x) : x ∈ {0,1}^n }` with optional scaling.
    ///
    /// Uses a serial or parallel path based on table size. Coordinate significance
    /// follows `PointLike::ENDIANNESS`; bare slices are high-to-low.
    #[track_caller]
    pub fn evals<P: PointLike + ?Sized>(r: &P, scaling_factor: Option<F>) -> Vec<F>
    where
        P::Coord: Send + Sync + Into<F>,
        F: Mul<P::Coord, Output = F> + SubAssign<F>,
    {
        match Self::evals_in(r, scaling_factor, Global) {
            Ok(v) => v,
            Err(never) => match never {},
        }
    }

    #[track_caller]
    #[tracing::instrument(skip_all, name = "EqPolynomial::evals")]
    pub fn evals_in<P: PointLike + ?Sized, A: EqAlloc>(
        r: &P,
        scaling_factor: Option<F>,
        policy: A,
    ) -> Result<Vec<F>, A::Error>
    where
        P::Coord: Send + Sync + Into<F>,
        F: Mul<P::Coord, Output = F> + SubAssign<F>,
    {
        let at = Location::caller();
        let len = domain_len(r.num_vars(), policy)?;
        check_layout::<F, _>(len, policy)?;
        policy.admit::<F>(len, at)?;
        let mut values = policy.table(len, TableInit::Zero)?;
        Self::fill_evals(r, scaling_factor, &mut values);
        Ok(values)
    }

    #[inline]
    pub(crate) fn fill_evals<P>(r: &P, scale: Option<F>, out: &mut [F])
    where
        P: PointLike + ?Sized,
        P::Coord: Send + Sync + Into<F>,
        F: Mul<P::Coord, Output = F> + SubAssign<F>,
    {
        out[0] = scale.unwrap_or(F::one());
        if r.num_vars() <= SERIAL_MAX_VARS {
            Self::fill_serial::<_, _, false>(r, out, &|x| x);
        } else {
            Self::fill_evals_parallel(r, out);
        }
    }

    #[cfg(test)]
    fn domain_len(num_vars: usize) -> usize {
        match domain_len(num_vars, Global) {
            Ok(v) => v,
            Err(never) => match never {},
        }
    }

    /// Computes `map(scale * eq(r, x))`, applying `map` once per output.
    #[track_caller]
    pub fn evals_mapped<P, G>(r: &P, scaling_factor: Option<F>, map: G) -> Vec<F>
    where
        P: PointLike + ?Sized,
        P::Coord: Send + Sync + Into<F>,
        F: Mul<P::Coord, Output = F>,
        G: Fn(F) -> F + Sync,
    {
        match Self::evals_mapped_in(r, scaling_factor, map, Global) {
            Ok(v) => v,
            Err(never) => match never {},
        }
    }

    #[track_caller]
    pub fn evals_mapped_in<P, G, A: EqAlloc>(
        r: &P,
        scaling_factor: Option<F>,
        map: G,
        policy: A,
    ) -> Result<Vec<F>, A::Error>
    where
        P: PointLike + ?Sized,
        P::Coord: Send + Sync + Into<F>,
        F: Mul<P::Coord, Output = F>,
        G: Fn(F) -> F + Sync,
    {
        let at = Location::caller();
        let len = domain_len(r.num_vars(), policy)?;
        check_layout::<F, _>(len, policy)?;
        policy.admit::<F>(len, at)?;
        let mut values = policy.table(len, TableInit::Zero)?;
        Self::evals_mapped_into(r, scaling_factor, map, &mut values);
        Ok(values)
    }

    fn evals_mapped_into<P, G>(r: &P, scale: Option<F>, map: G, out: &mut [F])
    where
        P: PointLike + ?Sized,
        P::Coord: Send + Sync + Into<F>,
        F: Mul<P::Coord, Output = F>,
        G: Fn(F) -> F + Sync,
    {
        out[0] = scale.unwrap_or(F::one());
        Self::fill_mapped(r, out, &map);
    }

    fn fill_mapped<P, G>(r: &P, values: &mut [F], map: &G)
    where
        P: PointLike + ?Sized,
        P::Coord: Send + Sync + Into<F>,
        F: Mul<P::Coord, Output = F>,
        G: Fn(F) -> F + Sync,
    {
        #[cfg(feature = "parallel")]
        if r.num_vars() >= PARALLEL_SLAB_MIN_VARS && rayon::current_num_threads() > 1 {
            if r.num_vars() <= PARALLEL_SLAB_MAX_VARS {
                Self::fill_slabs(r, values, map);
            } else {
                use rayon::prelude::*;
                Self::fill_parallel_coords(r.coords(), P::ENDIANNESS, values);
                values.par_iter_mut().for_each(|value| *value = map(*value));
            }
            return;
        }
        Self::fill_serial::<_, _, true>(r, values, map);
    }

    #[cfg(feature = "parallel")]
    fn fill_slabs<P, G>(r: &P, values: &mut [F], map: &G)
    where
        P: PointLike + ?Sized,
        P::Coord: Send + Sync + Into<F>,
        F: Mul<P::Coord, Output = F>,
        G: Fn(F) -> F + Sync,
    {
        use rayon::prelude::*;
        let outer_vars = (rayon::current_num_threads().next_power_of_two().ilog2() as usize)
            .min(r.num_vars() - MINIMUM_SLAB_VARS);
        let inner_vars = r.num_vars() - outer_vars;
        let slab_len = 1 << inner_vars;
        // Seeds occupy slab starts, avoiding an extra outer-table allocation.
        let mut size = 1;
        for j in 0..outer_vars {
            let coord = r.msb(j);
            size *= 2;
            for i in (0..size).rev().step_by(2) {
                let scalar = values[(i / 2) * slab_len];
                let at_one = scalar * coord;
                values[i * slab_len] = at_one;
                values[(i - 1) * slab_len] = scalar - at_one;
            }
        }
        // Capture only the coordinate slice: PointLike views need not be Sync.
        let coords = r.msb_range(outer_vars..r.num_vars());
        let coords = coords.coords();
        values.par_chunks_mut(slab_len).for_each(|slab| {
            if P::ENDIANNESS == HIGH_TO_LOW {
                Self::fill_serial::<_, _, true>(&PointRef::high_to_low(coords), slab, map);
            } else {
                Self::fill_serial::<_, _, true>(&PointRef::low_to_high(coords), slab, map);
            }
        });
    }

    /// Computes `eq(r, k)` over an aligned power-of-two block of Boolean indices.
    ///
    /// Returns values for `k = start_index..start_index + block_size` using the
    /// same index convention as [`Self::evals`], but only
    /// materializes the suffix table needed for the aligned block.
    pub fn evals_for_aligned_block<P: PointLike + ?Sized>(
        r: &P,
        start_index: usize,
        block_size: usize,
    ) -> Vec<F>
    where
        P::Coord: Send + Sync + Into<F>,
        F: Mul<P::Coord, Output = F> + SubAssign<F>,
    {
        assert!(block_size.is_power_of_two());
        assert_ne!(block_size, 0);
        assert_eq!(start_index % block_size, 0);

        let total_vars = r.num_vars();
        let block_vars = block_size.log_2();
        assert!(block_vars <= total_vars);

        let prefix_len = total_vars - block_vars;
        let prefix_value = start_index >> block_vars;
        let mut prefix_scale = F::one();
        for (k, bit) in boolean_bits_msb(prefix_len, prefix_value)
            .into_iter()
            .enumerate()
        {
            let r_i = r.msb(k).into();
            prefix_scale *= if bit { r_i } else { F::one() - r_i };
        }

        Self::evals(&r.msb_range(prefix_len..total_vars), Some(prefix_scale))
    }

    /// Chooses the largest aligned power-of-two block fitting the remaining range.
    ///
    /// Returns the block length and its eq table. This covers contiguous ranges
    /// with a small sequence of aligned-block tables without building the full
    /// domain table.
    pub fn evals_for_max_aligned_block<P: PointLike + ?Sized>(
        r: &P,
        start_index: usize,
        remaining_len: usize,
    ) -> (usize, Vec<F>)
    where
        P::Coord: Send + Sync + Into<F>,
        F: Mul<P::Coord, Output = F> + SubAssign<F>,
    {
        assert!(remaining_len > 0);
        let max_len_pow = if remaining_len.is_power_of_two() {
            remaining_len
        } else {
            remaining_len.next_power_of_two() >> 1
        };
        let align_pow = if start_index == 0 {
            1usize << r.num_vars()
        } else {
            1usize << start_index.trailing_zeros()
        };
        let block_size = max_len_pow.min(align_pow);
        (
            block_size,
            Self::evals_for_aligned_block(r, start_index, block_size),
        )
    }

    #[cfg(test)]
    #[inline]
    pub(crate) fn evals_serial<P: PointLike + ?Sized>(r: &P, scaling_factor: Option<F>) -> Vec<F>
    where
        P::Coord: Send + Sync + Into<F>,
        F: Mul<P::Coord, Output = F> + SubAssign<F>,
    {
        let mut evals = F::zeroed_vec(Self::domain_len(r.num_vars()));
        evals[0] = scaling_factor.unwrap_or(F::one());
        Self::fill_serial::<_, _, false>(r, &mut evals, &|x| x);
        evals
    }

    #[inline(always)]
    pub(super) fn fill_serial<P, G, const MAPPED: bool>(r: &P, evals: &mut [F], map: &G)
    where
        P: PointLike + ?Sized,
        P::Coord: Send + Sync + Into<F>,
        F: Mul<P::Coord, Output = F>,
        G: Fn(F) -> F + Sync,
    {
        if MAPPED && r.num_vars() == 0 {
            evals[0] = map(evals[0]);
        }
        let mut size = 1;
        for j in 0..r.num_vars() {
            let r_j = r.msb(j);
            size *= 2;
            for i in (0..size).rev().step_by(2) {
                let scalar = evals[i / 2];
                evals[i] = scalar * r_j;
                evals[i - 1] = scalar - evals[i];
                if MAPPED && j + 1 == r.num_vars() {
                    evals[i] = map(evals[i]);
                    evals[i - 1] = map(evals[i - 1]);
                }
            }
        }
    }

    /// Parallel eq table construction with optional scaling.
    ///
    /// Builds radix-four layers from least to most significant coordinates,
    /// using rayon to process the entries within each layer.
    #[cfg(test)]
    #[inline]
    pub(crate) fn evals_parallel<P: PointLike + ?Sized>(r: &P, scaling_factor: Option<F>) -> Vec<F>
    where
        P::Coord: Send + Sync + Into<F>,
        F: Mul<P::Coord, Output = F> + SubAssign<F>,
    {
        let mut evals = F::zeroed_vec(Self::domain_len(r.num_vars()));
        evals[0] = scaling_factor.unwrap_or(F::one());
        Self::fill_evals_parallel(r, &mut evals);
        evals
    }

    #[tracing::instrument(skip_all, name = "EqPolynomial::evals_parallel")]
    #[inline]
    fn fill_evals_parallel<P: PointLike + ?Sized>(r: &P, out: &mut [F])
    where
        P::Coord: Send + Sync + Into<F>,
        F: Mul<P::Coord, Output = F>,
    {
        Self::fill_parallel_coords(r.coords(), P::ENDIANNESS, out);
    }

    #[inline(always)]
    fn fill_parallel_coords<C>(r: &[C], endianness: Endianness, evals: &mut [F])
    where
        C: Copy + Send + Sync + Into<F>,
        F: Mul<C, Output = F>,
    {
        let mut size = 1;
        let mut i = r.len();

        while i >= 2 {
            let r_lo = r[msb_index(endianness, r.len(), i - 1)];
            let r_hi = r[msb_index(endianness, r.len(), i - 2)];
            i -= 2;

            let (q0, rest) = evals.split_at_mut(size);
            let (q1, rest) = rest.split_at_mut(size);
            let (q2, rest) = rest.split_at_mut(size);
            let (q3, _) = rest.split_at_mut(size);

            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;
                q0.par_iter_mut()
                    .zip(q1.par_iter_mut())
                    .zip(q2.par_iter_mut())
                    .zip(q3.par_iter_mut())
                    .for_each(|(((x, q1_out), q2_out), q3_out)| {
                        let with_lo = *x * r_lo;
                        let without_lo = *x - with_lo;

                        *q2_out = without_lo * r_hi;
                        *x = without_lo - *q2_out;
                        *q3_out = with_lo * r_hi;
                        *q1_out = with_lo - *q3_out;
                    });
            }

            #[cfg(not(feature = "parallel"))]
            {
                for j in 0..size {
                    let x_val = q0[j];
                    let with_lo = x_val * r_lo;
                    let without_lo = x_val - with_lo;

                    q2[j] = without_lo * r_hi;
                    q0[j] = without_lo - q2[j];
                    q3[j] = with_lo * r_hi;
                    q1[j] = with_lo - q3[j];
                }
            }

            size *= 4;
        }

        for k in (0..i).rev() {
            let r = &r[msb_index(endianness, r.len(), k)];
            let (evals_left, evals_right) = evals.split_at_mut(size);
            let (evals_right, _) = evals_right.split_at_mut(size);

            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;
                evals_left
                    .par_iter_mut()
                    .zip(evals_right.par_iter_mut())
                    .for_each(|(x, y)| {
                        *y = *x * *r;
                        *x -= *y;
                    });
            }

            #[cfg(not(feature = "parallel"))]
            {
                for i in 0..size {
                    evals_right[i] = evals_left[i] * *r;
                    evals_left[i] -= evals_right[i];
                }
            }

            size *= 2;
        }
    }
}
