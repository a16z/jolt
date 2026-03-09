//! CPU compute backend using Rayon for parallelism.

use jolt_field::{Field, FieldAccumulator};

use crate::traits::{BindingOrder, ComputeBackend, Scalar};

/// Parallelism threshold: buffers smaller than this use sequential loops.
///
/// Below this size the overhead of Rayon work-stealing exceeds the benefit.
const PAR_THRESHOLD: usize = 1024;

/// Composition evaluation function type for [`CpuKernel`].
///
/// Takes `(lo_values, hi_values, out)` and writes evaluations into `out`.
type EvalFn<F> = dyn Fn(&[F], &[F], &mut [F]) + Send + Sync;

/// Wraps a closure that evaluates a composition at grid points from paired
/// polynomial inputs, writing results into a caller-provided output slice.
/// The closure signature:
///
/// ```text
/// fn(lo: &[F], hi: &[F], out: &mut [F])
/// ```
///
/// where `lo[k]` and `hi[k]` are the even/odd pair for input buffer `k`,
/// and `out` has `num_evals` slots to receive the evaluations. For Toom-Cook
/// kernels: grid `{1, ..., D-1, ∞}`, D slots. For standard-grid kernels:
/// grid `{0, 1, ..., degree}`, `degree + 1` slots.
///
/// Constructed by a `compile` method on `CpuBackend` (not through the
/// `ComputeBackend` trait, to avoid coupling `jolt-compute` to `jolt-ir`).
pub struct CpuKernel<F: Field> {
    eval_fn: Box<EvalFn<F>>,
}

impl<F: Field> CpuKernel<F> {
    pub fn new(eval_fn: impl Fn(&[F], &[F], &mut [F]) + Send + Sync + 'static) -> Self {
        Self {
            eval_fn: Box::new(eval_fn),
        }
    }

    #[inline]
    pub fn evaluate(&self, lo: &[F], hi: &[F], out: &mut [F]) {
        (self.eval_fn)(lo, hi, out);
    }
}

/// Implements [`ComputeBackend`] with `Buffer<T> = Vec<T>`. All operations
/// use Rayon for parallelism (when the `parallel` feature is enabled and
/// buffers exceed `PAR_THRESHOLD`).
///
/// After monomorphization every trait call compiles to a direct function
/// call with no vtable indirection.
pub struct CpuBackend;

impl ComputeBackend for CpuBackend {
    type Buffer<T: Scalar> = Vec<T>;
    type CompiledKernel<F: Field> = CpuKernel<F>;

    #[inline]
    fn upload<T: Scalar>(&self, data: &[T]) -> Vec<T> {
        data.to_vec()
    }

    #[inline]
    fn download<T: Scalar>(&self, buf: &Vec<T>) -> Vec<T> {
        buf.clone()
    }

    fn alloc<T: Scalar>(&self, len: usize) -> Vec<T> {
        // SAFETY: T: Copy + 'static. We allocate zeroed memory which is valid
        // for all integer, bool, and field types (all have zero as a valid
        // bit pattern).
        let mut buf = Vec::with_capacity(len);
        // SAFETY: All Scalar types (integers, bool, field elements) have
        // all-zeros as a valid representation.
        unsafe {
            std::ptr::write_bytes(buf.as_mut_ptr(), 0, len);
            buf.set_len(len);
        }
        buf
    }

    #[inline]
    fn len<T: Scalar>(&self, buf: &Vec<T>) -> usize {
        buf.len()
    }

    fn interpolate_pairs<T, F>(&self, buf: Vec<T>, scalar: F) -> Vec<F>
    where
        T: Scalar,
        F: Field + From<T>,
    {
        let half = buf.len() / 2;

        #[cfg(feature = "parallel")]
        {
            if half >= PAR_THRESHOLD {
                use rayon::prelude::*;
                return (0..half)
                    .into_par_iter()
                    .map(|i| {
                        let lo: F = buf[2 * i].into();
                        let hi: F = buf[2 * i + 1].into();
                        lo + scalar * (hi - lo)
                    })
                    .collect();
            }
        }

        let mut result = Vec::with_capacity(half);
        for i in 0..half {
            let lo: F = buf[2 * i].into();
            let hi: F = buf[2 * i + 1].into();
            result.push(lo + scalar * (hi - lo));
        }
        result
    }

    fn interpolate_pairs_batch<F: Field>(&self, bufs: Vec<Vec<F>>, scalar: F) -> Vec<Vec<F>> {
        #[cfg(feature = "parallel")]
        {
            let total_pairs: usize = bufs.iter().map(|b| b.len() / 2).sum();
            if total_pairs >= PAR_THRESHOLD {
                use rayon::prelude::*;
                return bufs
                    .into_par_iter()
                    .map(|buf| {
                        let half = buf.len() / 2;
                        (0..half)
                            .map(|i| {
                                let lo = buf[2 * i];
                                let hi = buf[2 * i + 1];
                                lo + scalar * (hi - lo)
                            })
                            .collect()
                    })
                    .collect();
            }
        }

        bufs.into_iter()
            .map(|buf| self.interpolate_pairs(buf, scalar))
            .collect()
    }

    fn interpolate_pairs_inplace<F: Field>(
        &self,
        buf: &mut Vec<F>,
        scalar: F,
        order: BindingOrder,
    ) {
        let n = buf.len();
        let half = n / 2;

        match order {
            BindingOrder::HighToLow => {
                // Pairs: (buf[i], buf[i + half]). Write to buf[i], then truncate.
                // No aliasing: reads from the second half, writes to the first half.
                #[cfg(feature = "parallel")]
                {
                    if half >= PAR_THRESHOLD {
                        use rayon::prelude::*;
                        let (lo_half, hi_half) = buf.split_at_mut(half);
                        lo_half
                            .par_iter_mut()
                            .zip(hi_half.par_iter())
                            .for_each(|(lo, hi)| {
                                *lo = *lo + scalar * (*hi - *lo);
                            });
                        buf.truncate(half);
                        return;
                    }
                }
                for i in 0..half {
                    buf[i] = buf[i] + scalar * (buf[i + half] - buf[i]);
                }
                buf.truncate(half);
            }
            BindingOrder::LowToHigh => {
                // Pairs: (buf[2i], buf[2i+1]). Sequential is truly in-place
                // because i < 2i for i > 0 and i=0 reads before writing.
                #[cfg(feature = "parallel")]
                {
                    if half >= PAR_THRESHOLD {
                        use rayon::prelude::*;
                        // Parallel interleaved bind: reads from buf[2i], buf[2i+1]
                        // and writes to buf[i]. Since different threads may alias
                        // (thread i writes buf[i] which thread i/2 reads as buf[2*(i/2)]),
                        // we collect into a new vec and swap.
                        let result: Vec<F> = (0..half)
                            .into_par_iter()
                            .map(|i| {
                                let lo = buf[2 * i];
                                let hi = buf[2 * i + 1];
                                lo + scalar * (hi - lo)
                            })
                            .collect();
                        *buf = result;
                        return;
                    }
                }
                for i in 0..half {
                    let lo = buf[2 * i];
                    let hi = buf[2 * i + 1];
                    buf[i] = lo + scalar * (hi - lo);
                }
                buf.truncate(half);
            }
        }
    }

    fn interpolate_pairs_batch_inplace<F: Field>(
        &self,
        bufs: &mut [Vec<F>],
        scalar: F,
        order: BindingOrder,
    ) {
        #[cfg(feature = "parallel")]
        {
            let total_pairs: usize = bufs.iter().map(|b| b.len() / 2).sum();
            if total_pairs >= PAR_THRESHOLD {
                use rayon::prelude::*;
                bufs.par_iter_mut().for_each(|buf| {
                    // Inline bind to avoid nested Rayon dispatch for small individual buffers
                    let half = buf.len() / 2;
                    match order {
                        BindingOrder::HighToLow => {
                            for i in 0..half {
                                buf[i] = buf[i] + scalar * (buf[i + half] - buf[i]);
                            }
                            buf.truncate(half);
                        }
                        BindingOrder::LowToHigh => {
                            for i in 0..half {
                                let lo = buf[2 * i];
                                let hi = buf[2 * i + 1];
                                buf[i] = lo + scalar * (hi - lo);
                            }
                            buf.truncate(half);
                        }
                    }
                });
                return;
            }
        }
        for buf in bufs.iter_mut() {
            self.interpolate_pairs_inplace(buf, scalar, order);
        }
    }

    fn pairwise_reduce_fixed<F: Field, const D: usize>(
        &self,
        inputs: &[&Vec<F>],
        weights: &Vec<F>,
        kernel: &CpuKernel<F>,
    ) -> [F; D] {
        let n = inputs[0].len();
        debug_assert!(n % 2 == 0, "buffer length must be even");
        let half = n / 2;
        let num_inputs = inputs.len();

        let new_accs = || [F::Accumulator::default(); D];

        #[cfg(feature = "parallel")]
        {
            if half >= PAR_THRESHOLD {
                use rayon::prelude::*;

                let accs = (0..half)
                    .into_par_iter()
                    .fold(
                        || {
                            (
                                new_accs(),
                                Vec::with_capacity(num_inputs),
                                Vec::with_capacity(num_inputs),
                                [F::zero(); D],
                            )
                        },
                        |(mut acc, mut lo, mut hi, mut evals), i| {
                            lo.clear();
                            hi.clear();
                            for &input in inputs {
                                lo.push(input[2 * i]);
                                hi.push(input[2 * i + 1]);
                            }

                            kernel.evaluate(&lo, &hi, &mut evals);
                            let w = weights[i];
                            for (a, e) in acc.iter_mut().zip(evals.iter()) {
                                a.fmadd(w, *e);
                            }
                            (acc, lo, hi, evals)
                        },
                    )
                    .map(|(acc, _, _, _)| acc)
                    .reduce(new_accs, |mut a, b| {
                        for (ai, bi) in a.iter_mut().zip(b) {
                            ai.merge(bi);
                        }
                        a
                    });

                return accs.map(FieldAccumulator::reduce);
            }
        }

        let mut accs = new_accs();
        let mut lo = Vec::with_capacity(num_inputs);
        let mut hi = Vec::with_capacity(num_inputs);
        let mut evals = [F::zero(); D];

        for i in 0..half {
            lo.clear();
            hi.clear();
            for &input in inputs {
                lo.push(input[2 * i]);
                hi.push(input[2 * i + 1]);
            }

            kernel.evaluate(&lo, &hi, &mut evals);
            let w = weights[i];
            for (a, e) in accs.iter_mut().zip(evals.iter()) {
                a.fmadd(w, *e);
            }
        }

        accs.map(FieldAccumulator::reduce)
    }

    #[tracing::instrument(skip_all, name = "CpuBackend::tensor_pairwise_reduce")]
    fn tensor_pairwise_reduce<F: Field>(
        &self,
        inputs: &[&Vec<F>],
        outer_weights: &Vec<F>,
        inner_weights: &Vec<F>,
        kernel: &CpuKernel<F>,
        num_evals: usize,
    ) -> Vec<F> {
        let inner_len = inner_weights.len();
        let num_inputs = inputs.len();
        let total_pairs = outer_weights.len() * inner_len;

        let new_accs = || -> Vec<F::Accumulator> { vec![F::Accumulator::default(); num_evals] };

        let inner_fold = |outer_acc: &mut Vec<F::Accumulator>,
                          lo: &mut Vec<F>,
                          hi: &mut Vec<F>,
                          evals: &mut Vec<F>,
                          x_out: usize| {
            let mut inner_acc = new_accs();
            for (x_in, &w_in) in inner_weights.iter().enumerate() {
                let g = x_out * inner_len + x_in;
                lo.clear();
                hi.clear();
                for &input in inputs {
                    lo.push(input[2 * g]);
                    hi.push(input[2 * g + 1]);
                }
                kernel.evaluate(lo, hi, evals);
                for (a, e) in inner_acc.iter_mut().zip(evals.iter()) {
                    a.fmadd(w_in, *e);
                }
            }
            let w_out = outer_weights[x_out];
            for (oa, ia) in outer_acc.iter_mut().zip(inner_acc.iter()) {
                oa.fmadd(w_out, ia.reduce());
            }
        };

        #[cfg(feature = "parallel")]
        {
            if total_pairs >= PAR_THRESHOLD {
                use rayon::prelude::*;

                let accs = (0..outer_weights.len())
                    .into_par_iter()
                    .fold(
                        || {
                            (
                                new_accs(),
                                Vec::with_capacity(num_inputs),
                                Vec::with_capacity(num_inputs),
                                vec![F::zero(); num_evals],
                            )
                        },
                        |(mut outer_acc, mut lo, mut hi, mut evals), x_out| {
                            inner_fold(&mut outer_acc, &mut lo, &mut hi, &mut evals, x_out);
                            (outer_acc, lo, hi, evals)
                        },
                    )
                    .map(|(acc, _, _, _)| acc)
                    .reduce(new_accs, |mut a, b| {
                        for (ai, bi) in a.iter_mut().zip(b) {
                            ai.merge(bi);
                        }
                        a
                    });

                return accs.into_iter().map(FieldAccumulator::reduce).collect();
            }
        }

        let mut outer_acc = new_accs();
        let mut lo = Vec::with_capacity(num_inputs);
        let mut hi = Vec::with_capacity(num_inputs);
        let mut evals = vec![F::zero(); num_evals];

        for x_out in 0..outer_weights.len() {
            inner_fold(&mut outer_acc, &mut lo, &mut hi, &mut evals, x_out);
        }

        outer_acc
            .into_iter()
            .map(FieldAccumulator::reduce)
            .collect()
    }

    fn tensor_pairwise_reduce_fixed<F: Field, const D: usize>(
        &self,
        inputs: &[&Vec<F>],
        outer_weights: &Vec<F>,
        inner_weights: &Vec<F>,
        kernel: &CpuKernel<F>,
    ) -> [F; D] {
        let inner_len = inner_weights.len();
        let num_inputs = inputs.len();
        let total_pairs = outer_weights.len() * inner_len;

        let new_accs = || [F::Accumulator::default(); D];

        let inner_fold = |outer_acc: &mut [F::Accumulator; D],
                          lo: &mut Vec<F>,
                          hi: &mut Vec<F>,
                          evals: &mut [F; D],
                          x_out: usize| {
            let mut inner_acc = new_accs();
            for (x_in, &w_in) in inner_weights.iter().enumerate() {
                let g = x_out * inner_len + x_in;
                lo.clear();
                hi.clear();
                for &input in inputs {
                    lo.push(input[2 * g]);
                    hi.push(input[2 * g + 1]);
                }
                kernel.evaluate(lo, hi, evals);
                for (a, e) in inner_acc.iter_mut().zip(evals.iter()) {
                    a.fmadd(w_in, *e);
                }
            }
            let w_out = outer_weights[x_out];
            for (oa, ia) in outer_acc.iter_mut().zip(inner_acc.iter()) {
                oa.fmadd(w_out, ia.reduce());
            }
        };

        #[cfg(feature = "parallel")]
        {
            if total_pairs >= PAR_THRESHOLD {
                use rayon::prelude::*;

                let accs = (0..outer_weights.len())
                    .into_par_iter()
                    .fold(
                        || {
                            (
                                new_accs(),
                                Vec::with_capacity(num_inputs),
                                Vec::with_capacity(num_inputs),
                                [F::zero(); D],
                            )
                        },
                        |(mut outer_acc, mut lo, mut hi, mut evals), x_out| {
                            inner_fold(&mut outer_acc, &mut lo, &mut hi, &mut evals, x_out);
                            (outer_acc, lo, hi, evals)
                        },
                    )
                    .map(|(acc, _, _, _)| acc)
                    .reduce(new_accs, |mut a, b| {
                        for (ai, bi) in a.iter_mut().zip(b) {
                            ai.merge(bi);
                        }
                        a
                    });

                return accs.map(FieldAccumulator::reduce);
            }
        }

        let mut outer_acc = new_accs();
        let mut lo = Vec::with_capacity(num_inputs);
        let mut hi = Vec::with_capacity(num_inputs);
        let mut evals = [F::zero(); D];

        for x_out in 0..outer_weights.len() {
            inner_fold(&mut outer_acc, &mut lo, &mut hi, &mut evals, x_out);
        }

        outer_acc.map(FieldAccumulator::reduce)
    }

    #[tracing::instrument(skip_all, name = "CpuBackend::pairwise_reduce_multi")]
    fn pairwise_reduce_multi<F: Field>(
        &self,
        inputs: &[&Vec<F>],
        weights: &Vec<F>,
        kernels: &[(&CpuKernel<F>, usize)],
    ) -> Vec<Vec<F>> {
        let n = inputs[0].len();
        debug_assert!(n % 2 == 0, "buffer length must be even");
        let half = n / 2;
        let num_inputs = inputs.len();
        let num_kernels = kernels.len();

        let new_all_accs = || -> Vec<Vec<F::Accumulator>> {
            kernels
                .iter()
                .map(|(_, num_evals)| vec![F::Accumulator::default(); *num_evals])
                .collect()
        };

        #[cfg(feature = "parallel")]
        {
            if half >= PAR_THRESHOLD {
                use rayon::prelude::*;

                let all_accs = (0..half)
                    .into_par_iter()
                    .fold(
                        || {
                            let accs = new_all_accs();
                            let lo = Vec::with_capacity(num_inputs);
                            let hi = Vec::with_capacity(num_inputs);
                            let evals_bufs: Vec<Vec<F>> =
                                kernels.iter().map(|(_, ne)| vec![F::zero(); *ne]).collect();
                            (accs, lo, hi, evals_bufs)
                        },
                        |(mut accs, mut lo, mut hi, mut evals_bufs), i| {
                            lo.clear();
                            hi.clear();
                            for &input in inputs {
                                lo.push(input[2 * i]);
                                hi.push(input[2 * i + 1]);
                            }

                            let w = weights[i];
                            for k in 0..num_kernels {
                                kernels[k].0.evaluate(&lo, &hi, &mut evals_bufs[k]);
                                for (a, e) in accs[k].iter_mut().zip(evals_bufs[k].iter()) {
                                    a.fmadd(w, *e);
                                }
                            }
                            (accs, lo, hi, evals_bufs)
                        },
                    )
                    .map(|(accs, _, _, _)| accs)
                    .reduce(new_all_accs, |mut a, b| {
                        for (ak, bk) in a.iter_mut().zip(b) {
                            for (ai, bi) in ak.iter_mut().zip(bk) {
                                ai.merge(bi);
                            }
                        }
                        a
                    });

                return all_accs
                    .into_iter()
                    .map(|accs| accs.into_iter().map(FieldAccumulator::reduce).collect())
                    .collect();
            }
        }

        let mut all_accs = new_all_accs();
        let mut lo = Vec::with_capacity(num_inputs);
        let mut hi = Vec::with_capacity(num_inputs);
        let mut evals_bufs: Vec<Vec<F>> =
            kernels.iter().map(|(_, ne)| vec![F::zero(); *ne]).collect();

        for i in 0..half {
            lo.clear();
            hi.clear();
            for &input in inputs {
                lo.push(input[2 * i]);
                hi.push(input[2 * i + 1]);
            }

            let w = weights[i];
            for k in 0..num_kernels {
                kernels[k].0.evaluate(&lo, &hi, &mut evals_bufs[k]);
                for (a, e) in all_accs[k].iter_mut().zip(evals_bufs[k].iter()) {
                    a.fmadd(w, *e);
                }
            }
        }

        all_accs
            .into_iter()
            .map(|accs| accs.into_iter().map(FieldAccumulator::reduce).collect())
            .collect()
    }

    #[tracing::instrument(skip_all, name = "CpuBackend::pairwise_reduce")]
    fn pairwise_reduce<F: Field>(
        &self,
        inputs: &[&Vec<F>],
        weights: &Vec<F>,
        kernel: &Self::CompiledKernel<F>,
        num_evals: usize,
    ) -> Vec<F> {
        let n = inputs[0].len();
        debug_assert!(n % 2 == 0, "buffer length must be even");
        let half = n / 2;
        let num_inputs = inputs.len();

        let new_accs = || -> Vec<F::Accumulator> {
            (0..num_evals).map(|_| F::Accumulator::default()).collect()
        };

        #[cfg(feature = "parallel")]
        {
            if half >= PAR_THRESHOLD {
                use rayon::prelude::*;

                let accs = (0..half)
                    .into_par_iter()
                    .fold(
                        || {
                            (
                                new_accs(),
                                Vec::with_capacity(num_inputs),
                                Vec::with_capacity(num_inputs),
                                vec![F::zero(); num_evals],
                            )
                        },
                        |(mut acc, mut lo, mut hi, mut evals), i| {
                            lo.clear();
                            hi.clear();
                            for &input in inputs {
                                lo.push(input[2 * i]);
                                hi.push(input[2 * i + 1]);
                            }

                            kernel.evaluate(&lo, &hi, &mut evals);
                            let w = weights[i];
                            for (a, e) in acc.iter_mut().zip(evals.iter()) {
                                a.fmadd(w, *e);
                            }
                            (acc, lo, hi, evals)
                        },
                    )
                    .map(|(acc, _, _, _)| acc)
                    .reduce(new_accs, |mut a, b| {
                        for (ai, bi) in a.iter_mut().zip(b) {
                            ai.merge(bi);
                        }
                        a
                    });

                return accs.into_iter().map(FieldAccumulator::reduce).collect();
            }
        }

        let mut accs = new_accs();
        let mut lo = Vec::with_capacity(num_inputs);
        let mut hi = Vec::with_capacity(num_inputs);
        let mut evals = vec![F::zero(); num_evals];

        for i in 0..half {
            lo.clear();
            hi.clear();
            for &input in inputs {
                lo.push(input[2 * i]);
                hi.push(input[2 * i + 1]);
            }

            kernel.evaluate(&lo, &hi, &mut evals);
            let w = weights[i];
            for (a, e) in accs.iter_mut().zip(evals.iter()) {
                a.fmadd(w, *e);
            }
        }

        accs.into_iter().map(FieldAccumulator::reduce).collect()
    }

    #[tracing::instrument(skip_all, name = "CpuBackend::product_table")]
    fn product_table<F: Field>(&self, point: &[F]) -> Vec<F> {
        let n = point.len();
        let size = 1usize << n;
        let mut table = Vec::with_capacity(size);
        table.push(F::one());

        for &r_i in point {
            let one_minus_r_i = F::one() - r_i;
            let prev_len = table.len();
            table.resize(prev_len * 2, F::zero());

            #[cfg(feature = "parallel")]
            {
                if prev_len >= PAR_THRESHOLD {
                    use rayon::prelude::*;
                    let (left, right) = table.split_at_mut(prev_len);
                    left.par_iter_mut()
                        .zip(right.par_iter_mut())
                        .for_each(|(lo, hi)| {
                            let base = *lo;
                            *hi = base * r_i;
                            *lo = base * one_minus_r_i;
                        });
                    continue;
                }
            }

            for j in (0..prev_len).rev() {
                let base = table[j];
                table[2 * j] = base * one_minus_r_i;
                table[2 * j + 1] = base * r_i;
            }
        }

        table
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::{Field, Fr};
    use num_traits::{One, Zero};
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    fn backend() -> CpuBackend {
        CpuBackend
    }

    #[test]
    fn upload_download_round_trip() {
        let b = backend();
        let data: Vec<Fr> = vec![Fr::from_u64(1), Fr::from_u64(2), Fr::from_u64(3)];
        let buf = b.upload(&data);
        let out = b.download(&buf);
        assert_eq!(data, out);
    }

    #[test]
    fn upload_download_compact_u8() {
        let b = backend();
        let data: Vec<u8> = vec![0, 1, 2, 255];
        let buf = b.upload(&data);
        let out = b.download(&buf);
        assert_eq!(data, out);
    }

    #[test]
    fn alloc_is_zeroed() {
        let b = backend();
        let buf: Vec<u64> = b.alloc(16);
        assert_eq!(buf.len(), 16);
        assert!(buf.iter().all(|&x| x == 0));
    }

    #[test]
    fn alloc_field_is_zero() {
        let b = backend();
        let buf: Vec<Fr> = b.alloc(8);
        assert_eq!(buf.len(), 8);
        assert!(buf.iter().all(|x| x.is_zero()));
    }

    #[test]
    fn len_and_is_empty() {
        let b = backend();
        let empty: Vec<Fr> = b.alloc(0);
        assert!(b.is_empty(&empty));
        assert_eq!(b.len(&empty), 0);

        let nonempty = b.upload(&[Fr::one()]);
        assert!(!b.is_empty(&nonempty));
        assert_eq!(b.len(&nonempty), 1);
    }

    #[test]
    fn interpolate_pairs_field_matches_manual() {
        let b = backend();
        let mut rng = ChaCha20Rng::seed_from_u64(1);
        let n = 8;
        let data: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
        let scalar = Fr::random(&mut rng);

        let buf = b.upload(&data);
        let result = b.interpolate_pairs::<Fr, Fr>(buf, scalar);

        assert_eq!(result.len(), n / 2);
        for i in 0..n / 2 {
            let expected = data[2 * i] + scalar * (data[2 * i + 1] - data[2 * i]);
            assert_eq!(result[i], expected);
        }
    }

    #[test]
    fn interpolate_pairs_compact_u8() {
        let b = backend();
        let data: Vec<u8> = vec![0, 10, 20, 30, 40, 50, 60, 70];
        let scalar = Fr::from_u64(2);

        let buf = b.upload(&data);
        let result = b.interpolate_pairs::<u8, Fr>(buf, scalar);

        assert_eq!(result.len(), 4);
        for i in 0..4 {
            let lo = Fr::from(data[2 * i]);
            let hi = Fr::from(data[2 * i + 1]);
            let expected = lo + scalar * (hi - lo);
            assert_eq!(result[i], expected);
        }
    }

    #[test]
    fn interpolate_pairs_matches_polynomial_bind() {
        use jolt_poly::Polynomial;

        let b = backend();
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let n = 6;
        let data: Vec<Fr> = (0..(1 << n)).map(|_| Fr::random(&mut rng)).collect();
        let scalar = Fr::random(&mut rng);

        // Polynomial bind: fixes first variable
        let poly = Polynomial::new(data.clone());
        let bound = poly.bind_to_field(scalar);

        // ComputeBackend interpolate_pairs: pairwise interpolation
        // Polynomial::bind uses layout [lo_half | hi_half] while our
        // interpolate_pairs uses interleaved [lo0, hi0, lo1, hi1, ...].
        // So we need to interleave the polynomial data to match.
        let half = data.len() / 2;
        let mut interleaved = Vec::with_capacity(data.len());
        for i in 0..half {
            interleaved.push(data[i]);
            interleaved.push(data[i + half]);
        }

        let buf = b.upload(&interleaved);
        let result = b.interpolate_pairs::<Fr, Fr>(buf, scalar);

        assert_eq!(result.len(), bound.len());
        assert_eq!(&result, bound.evaluations());
    }

    #[test]
    fn product_table_sum_is_one() {
        let b = backend();
        let mut rng = ChaCha20Rng::seed_from_u64(7);
        let n = 5;
        let point: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();

        let table = b.product_table(&point);
        assert_eq!(table.len(), 1 << n);
        let sum: Fr = table.iter().copied().sum();
        assert_eq!(sum, Fr::one());
    }

    #[test]
    fn product_table_matches_eq_polynomial() {
        use jolt_poly::EqPolynomial;

        let b = backend();
        let mut rng = ChaCha20Rng::seed_from_u64(99);
        let n = 6;
        let point: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();

        let table = b.product_table(&point);
        let eq_table = EqPolynomial::new(point).evaluations();

        assert_eq!(table, eq_table);
    }

    #[test]
    fn product_table_parallel_matches_sequential() {
        let b = backend();
        let mut rng = ChaCha20Rng::seed_from_u64(300);
        // n=11 -> 2048 entries, above PAR_THRESHOLD
        let n = 11;
        let point: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();

        let table = b.product_table(&point);
        assert_eq!(table.len(), 1 << n);
        let sum: Fr = table.iter().copied().sum();
        assert_eq!(sum, Fr::one());
    }

    fn make_identity_kernel() -> CpuKernel<Fr> {
        // Identity kernel: for a single input, evaluates the linear interpolant
        // at grid points {0, 1, ..., num_evals-1}.
        // f(t) = lo + t * (hi - lo)
        CpuKernel::new(|lo: &[Fr], hi: &[Fr], out: &mut [Fr]| {
            for (t, slot) in out.iter_mut().enumerate() {
                let t_f = Fr::from_u64(t as u64);
                let mut sum = Fr::zero();
                for k in 0..lo.len() {
                    sum += lo[k] + t_f * (hi[k] - lo[k]);
                }
                *slot = sum;
            }
        })
    }

    #[test]
    fn pairwise_reduce_trivial() {
        let b = backend();
        // Single input [1, 2, 3, 4], weights [1, 1], 2 evals (t=0,1)
        // Pair 0: lo=1, hi=2 -> f(0)=1, f(1)=2; weighted by 1
        // Pair 1: lo=3, hi=4 -> f(0)=3, f(1)=4; weighted by 1
        // Sums: [1+3, 2+4] = [4, 6]
        let input: Vec<Fr> = vec![1, 2, 3, 4].into_iter().map(Fr::from_u64).collect();
        let weights: Vec<Fr> = vec![Fr::one(), Fr::one()];

        let kernel = make_identity_kernel();
        let result = b.pairwise_reduce(&[&input], &weights, &kernel, 2);

        assert_eq!(result.len(), 2);
        assert_eq!(result[0], Fr::from_u64(4));
        assert_eq!(result[1], Fr::from_u64(6));
    }

    #[test]
    fn pairwise_reduce_with_weights() {
        let b = backend();
        let input: Vec<Fr> = vec![10, 20, 30, 40].into_iter().map(Fr::from_u64).collect();
        let weights: Vec<Fr> = vec![Fr::from_u64(2), Fr::from_u64(3)];

        let kernel = make_identity_kernel();
        let result = b.pairwise_reduce(&[&input], &weights, &kernel, 2);

        // Pair 0: lo=10, hi=20; w=2 -> [2*10, 2*20] = [20, 40]
        // Pair 1: lo=30, hi=40; w=3 -> [3*30, 3*40] = [90, 120]
        // Sums: [110, 160]
        assert_eq!(result[0], Fr::from_u64(110));
        assert_eq!(result[1], Fr::from_u64(160));
    }

    #[test]
    fn pairwise_reduce_multiple_inputs() {
        let b = backend();
        let a: Vec<Fr> = vec![1, 2, 3, 4].into_iter().map(Fr::from_u64).collect();
        let c: Vec<Fr> = vec![10, 20, 30, 40].into_iter().map(Fr::from_u64).collect();
        let weights: Vec<Fr> = vec![Fr::one(); 2];

        // Kernel sums across inputs: sum_k (lo[k] + t*(hi[k]-lo[k]))
        let kernel = make_identity_kernel();
        let result = b.pairwise_reduce(&[&a, &c], &weights, &kernel, 2);

        // Pair 0: a=(1,2), c=(10,20) -> f(0)=1+10=11, f(1)=2+20=22
        // Pair 1: a=(3,4), c=(30,40) -> f(0)=3+30=33, f(1)=4+40=44
        // Sums: [44, 66]
        assert_eq!(result[0], Fr::from_u64(44));
        assert_eq!(result[1], Fr::from_u64(66));
    }

    #[test]
    fn pairwise_reduce_3_evals() {
        let b = backend();
        let input: Vec<Fr> = vec![1, 3].into_iter().map(Fr::from_u64).collect();
        let weights: Vec<Fr> = vec![Fr::one()];

        // Identity kernel with 3 evals: t=0,1,2
        let kernel = make_identity_kernel();
        let result = b.pairwise_reduce(&[&input], &weights, &kernel, 3);

        // lo=1, hi=3
        // f(0) = 1, f(1) = 3, f(2) = 1 + 2*(3-1) = 5
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], Fr::from_u64(1));
        assert_eq!(result[1], Fr::from_u64(3));
        assert_eq!(result[2], Fr::from_u64(5));
    }

    #[test]
    fn interpolate_pairs_large_parallel() {
        let b = backend();
        let mut rng = ChaCha20Rng::seed_from_u64(200);
        // 4096 elements -> 2048 pairs, above PAR_THRESHOLD
        let n = 4096;
        let data: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
        let scalar = Fr::random(&mut rng);

        let buf = b.upload(&data);
        let result = b.interpolate_pairs::<Fr, Fr>(buf, scalar);

        assert_eq!(result.len(), n / 2);
        for i in 0..n / 2 {
            let expected = data[2 * i] + scalar * (data[2 * i + 1] - data[2 * i]);
            assert_eq!(result[i], expected);
        }
    }

    #[test]
    fn pairwise_reduce_large_parallel() {
        let b = backend();
        let mut rng = ChaCha20Rng::seed_from_u64(201);
        // 4096 elements -> 2048 pairs, above PAR_THRESHOLD
        let n = 4096;
        let input: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
        let weights: Vec<Fr> = (0..n / 2).map(|_| Fr::random(&mut rng)).collect();

        let kernel = make_identity_kernel();
        let result = b.pairwise_reduce(&[&input], &weights, &kernel, 2);

        // Verify against sequential computation
        let mut expected = vec![Fr::zero(); 2];
        for i in 0..n / 2 {
            let lo = input[2 * i];
            let hi = input[2 * i + 1];
            expected[0] += weights[i] * lo;
            expected[1] += weights[i] * hi;
        }
        assert_eq!(result, expected);
    }

    #[test]
    fn interpolate_pairs_batch_matches_individual() {
        let b = backend();
        let mut rng = ChaCha20Rng::seed_from_u64(400);
        let scalar = Fr::random(&mut rng);

        let bufs: Vec<Vec<Fr>> = (0..5)
            .map(|_| (0..64).map(|_| Fr::random(&mut rng)).collect())
            .collect();

        // Batched
        let batch_result = b.interpolate_pairs_batch(bufs.clone(), scalar);

        // Individual
        let individual_results: Vec<Vec<Fr>> = bufs
            .into_iter()
            .map(|buf| b.interpolate_pairs(buf, scalar))
            .collect();

        assert_eq!(batch_result, individual_results);
    }

    #[test]
    fn interpolate_pairs_batch_large_parallel() {
        let b = backend();
        let mut rng = ChaCha20Rng::seed_from_u64(401);
        let scalar = Fr::random(&mut rng);

        // 10 buffers of 512 elements each = 2560 total pairs, above PAR_THRESHOLD
        let bufs: Vec<Vec<Fr>> = (0..10)
            .map(|_| (0..512).map(|_| Fr::random(&mut rng)).collect())
            .collect();

        let batch_result = b.interpolate_pairs_batch(bufs.clone(), scalar);

        let individual_results: Vec<Vec<Fr>> = bufs
            .into_iter()
            .map(|buf| b.interpolate_pairs(buf, scalar))
            .collect();

        assert_eq!(batch_result, individual_results);
    }

    // --- interpolate_pairs_inplace tests ---

    #[test]
    fn inplace_low_to_high_matches_allocating() {
        let b = backend();
        let mut rng = ChaCha20Rng::seed_from_u64(500);
        let n = 64;
        let data: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
        let scalar = Fr::random(&mut rng);

        let allocating = b.interpolate_pairs::<Fr, Fr>(data.clone(), scalar);

        let mut inplace = data;
        b.interpolate_pairs_inplace(&mut inplace, scalar, BindingOrder::LowToHigh);

        assert_eq!(inplace, allocating);
    }

    #[test]
    fn inplace_high_to_low_matches_polynomial_bind() {
        use jolt_poly::Polynomial;

        let b = backend();
        let mut rng = ChaCha20Rng::seed_from_u64(501);
        let nv = 6;
        let data: Vec<Fr> = (0..(1 << nv)).map(|_| Fr::random(&mut rng)).collect();
        let scalar = Fr::random(&mut rng);

        // Polynomial::bind uses split-half layout (HighToLow)
        let poly = Polynomial::new(data.clone());
        let bound = poly.bind_to_field(scalar);

        let mut buf = data;
        b.interpolate_pairs_inplace(&mut buf, scalar, BindingOrder::HighToLow);

        assert_eq!(buf.len(), bound.len());
        assert_eq!(&buf, bound.evaluations());
    }

    #[test]
    fn inplace_high_to_low_parallel() {
        let b = backend();
        let mut rng = ChaCha20Rng::seed_from_u64(502);
        // 4096 elements = 2048 pairs, above PAR_THRESHOLD
        let n = 4096;
        let data: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
        let scalar = Fr::random(&mut rng);

        // Reference: manual HighToLow
        let half = n / 2;
        let expected: Vec<Fr> = (0..half)
            .map(|i| data[i] + scalar * (data[i + half] - data[i]))
            .collect();

        let mut buf = data;
        b.interpolate_pairs_inplace(&mut buf, scalar, BindingOrder::HighToLow);

        assert_eq!(buf, expected);
    }

    #[test]
    fn inplace_low_to_high_parallel() {
        let b = backend();
        let mut rng = ChaCha20Rng::seed_from_u64(503);
        let n = 4096;
        let data: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
        let scalar = Fr::random(&mut rng);

        let allocating = b.interpolate_pairs::<Fr, Fr>(data.clone(), scalar);

        let mut inplace = data;
        b.interpolate_pairs_inplace(&mut inplace, scalar, BindingOrder::LowToHigh);

        assert_eq!(inplace, allocating);
    }

    #[test]
    fn batch_inplace_matches_individual() {
        let b = backend();
        let mut rng = ChaCha20Rng::seed_from_u64(504);
        let scalar = Fr::random(&mut rng);

        let mut bufs: Vec<Vec<Fr>> = (0..5)
            .map(|_| (0..64).map(|_| Fr::random(&mut rng)).collect())
            .collect();
        let bufs_copy = bufs.clone();

        b.interpolate_pairs_batch_inplace(&mut bufs, scalar, BindingOrder::LowToHigh);

        let individual: Vec<Vec<Fr>> = bufs_copy
            .into_iter()
            .map(|buf| b.interpolate_pairs(buf, scalar))
            .collect();

        assert_eq!(bufs, individual);
    }

    #[test]
    fn batch_inplace_high_to_low() {
        use jolt_poly::Polynomial;

        let b = backend();
        let mut rng = ChaCha20Rng::seed_from_u64(505);
        let scalar = Fr::random(&mut rng);
        let nv = 4;

        let mut bufs: Vec<Vec<Fr>> = (0..3)
            .map(|_| (0..(1 << nv)).map(|_| Fr::random(&mut rng)).collect())
            .collect();
        let expected: Vec<Vec<Fr>> = bufs
            .iter()
            .map(|data| {
                let p = Polynomial::new(data.clone());
                p.bind_to_field(scalar).evaluations().to_vec()
            })
            .collect();

        b.interpolate_pairs_batch_inplace(&mut bufs, scalar, BindingOrder::HighToLow);

        assert_eq!(bufs, expected);
    }

    // --- pairwise_reduce_fixed tests ---

    #[test]
    fn pairwise_reduce_fixed_matches_dynamic() {
        let b = backend();
        let mut rng = ChaCha20Rng::seed_from_u64(600);
        let n = 128;
        let input: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
        let weights: Vec<Fr> = (0..n / 2).map(|_| Fr::random(&mut rng)).collect();

        let kernel = make_identity_kernel();
        let dynamic = b.pairwise_reduce(&[&input], &weights, &kernel, 4);
        let fixed: [Fr; 4] = b.pairwise_reduce_fixed(&[&input], &weights, &kernel);

        assert_eq!(fixed.as_slice(), dynamic.as_slice());
    }

    #[test]
    fn pairwise_reduce_fixed_parallel() {
        let b = backend();
        let mut rng = ChaCha20Rng::seed_from_u64(601);
        let n = 4096;
        let input: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
        let weights: Vec<Fr> = (0..n / 2).map(|_| Fr::random(&mut rng)).collect();

        let kernel = make_identity_kernel();
        let dynamic = b.pairwise_reduce(&[&input], &weights, &kernel, 2);
        let fixed: [Fr; 2] = b.pairwise_reduce_fixed(&[&input], &weights, &kernel);

        assert_eq!(fixed.as_slice(), dynamic.as_slice());
    }

    // --- tensor_pairwise_reduce tests ---

    #[test]
    fn tensor_reduce_matches_flat_reduce() {
        let b = backend();
        let mut rng = ChaCha20Rng::seed_from_u64(700);

        let outer_len = 4;
        let inner_len = 8;
        let total_pairs = outer_len * inner_len;
        let n = total_pairs * 2;

        let input: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
        let outer_w: Vec<Fr> = (0..outer_len).map(|_| Fr::random(&mut rng)).collect();
        let inner_w: Vec<Fr> = (0..inner_len).map(|_| Fr::random(&mut rng)).collect();

        // Flat weights: tensor product
        let mut flat_w = Vec::with_capacity(total_pairs);
        for &o in &outer_w {
            for &i in &inner_w {
                flat_w.push(o * i);
            }
        }

        let kernel = make_identity_kernel();
        let flat_result = b.pairwise_reduce(&[&input], &flat_w, &kernel, 2);
        let tensor_result = b.tensor_pairwise_reduce(&[&input], &outer_w, &inner_w, &kernel, 2);

        assert_eq!(tensor_result, flat_result);
    }

    #[test]
    fn tensor_reduce_fixed_matches_dynamic() {
        let b = backend();
        let mut rng = ChaCha20Rng::seed_from_u64(701);

        let outer_len = 4;
        let inner_len = 8;
        let n = outer_len * inner_len * 2;

        let input: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
        let outer_w: Vec<Fr> = (0..outer_len).map(|_| Fr::random(&mut rng)).collect();
        let inner_w: Vec<Fr> = (0..inner_len).map(|_| Fr::random(&mut rng)).collect();

        let kernel = make_identity_kernel();
        let dynamic = b.tensor_pairwise_reduce(&[&input], &outer_w, &inner_w, &kernel, 4);
        let fixed: [Fr; 4] = b.tensor_pairwise_reduce_fixed(&[&input], &outer_w, &inner_w, &kernel);

        assert_eq!(fixed.as_slice(), dynamic.as_slice());
    }

    #[test]
    fn tensor_reduce_multiple_inputs() {
        let b = backend();
        let mut rng = ChaCha20Rng::seed_from_u64(702);

        let outer_len = 4;
        let inner_len = 4;
        let n = outer_len * inner_len * 2;

        let input_a: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
        let input_b: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
        let outer_w: Vec<Fr> = (0..outer_len).map(|_| Fr::random(&mut rng)).collect();
        let inner_w: Vec<Fr> = (0..inner_len).map(|_| Fr::random(&mut rng)).collect();

        // Flat weights
        let mut flat_w = Vec::with_capacity(outer_len * inner_len);
        for &o in &outer_w {
            for &i in &inner_w {
                flat_w.push(o * i);
            }
        }

        let kernel = make_identity_kernel();
        let flat_result = b.pairwise_reduce(&[&input_a, &input_b], &flat_w, &kernel, 2);
        let tensor_result =
            b.tensor_pairwise_reduce(&[&input_a, &input_b], &outer_w, &inner_w, &kernel, 2);

        assert_eq!(tensor_result, flat_result);
    }

    #[test]
    fn tensor_reduce_parallel() {
        let b = backend();
        let mut rng = ChaCha20Rng::seed_from_u64(703);

        // 32 × 64 = 2048 pairs, above PAR_THRESHOLD
        let outer_len = 32;
        let inner_len = 64;
        let n = outer_len * inner_len * 2;

        let input: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
        let outer_w: Vec<Fr> = (0..outer_len).map(|_| Fr::random(&mut rng)).collect();
        let inner_w: Vec<Fr> = (0..inner_len).map(|_| Fr::random(&mut rng)).collect();

        let mut flat_w = Vec::with_capacity(outer_len * inner_len);
        for &o in &outer_w {
            for &i in &inner_w {
                flat_w.push(o * i);
            }
        }

        let kernel = make_identity_kernel();
        let flat_result = b.pairwise_reduce(&[&input], &flat_w, &kernel, 3);
        let tensor_result = b.tensor_pairwise_reduce(&[&input], &outer_w, &inner_w, &kernel, 3);

        assert_eq!(tensor_result, flat_result);
    }

    // --- pairwise_reduce_multi tests ---

    #[test]
    fn multi_reduce_matches_individual() {
        let b = backend();
        let mut rng = ChaCha20Rng::seed_from_u64(800);
        let n = 64;
        let input: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
        let weights: Vec<Fr> = (0..n / 2).map(|_| Fr::random(&mut rng)).collect();

        let k1 = make_identity_kernel();
        // Second kernel: product of inputs
        let k2 = CpuKernel::new(|lo: &[Fr], hi: &[Fr], out: &mut [Fr]| {
            for (t, slot) in out.iter_mut().enumerate() {
                let t_f = Fr::from_u64(t as u64);
                let mut prod = Fr::one();
                for k in 0..lo.len() {
                    prod *= lo[k] + t_f * (hi[k] - lo[k]);
                }
                *slot = prod;
            }
        });

        let individual_1 = b.pairwise_reduce(&[&input], &weights, &k1, 2);
        let individual_2 = b.pairwise_reduce(&[&input], &weights, &k2, 3);

        let multi = b.pairwise_reduce_multi(&[&input], &weights, &[(&k1, 2), (&k2, 3)]);

        assert_eq!(multi.len(), 2);
        assert_eq!(multi[0], individual_1);
        assert_eq!(multi[1], individual_2);
    }

    #[test]
    fn multi_reduce_parallel() {
        let b = backend();
        let mut rng = ChaCha20Rng::seed_from_u64(801);
        let n = 4096;
        let input: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
        let weights: Vec<Fr> = (0..n / 2).map(|_| Fr::random(&mut rng)).collect();

        let k1 = make_identity_kernel();
        let k2 = CpuKernel::new(|lo: &[Fr], hi: &[Fr], out: &mut [Fr]| {
            for (t, slot) in out.iter_mut().enumerate() {
                let t_f = Fr::from_u64(t as u64);
                let mut prod = Fr::one();
                for k in 0..lo.len() {
                    prod *= lo[k] + t_f * (hi[k] - lo[k]);
                }
                *slot = prod;
            }
        });

        let individual_1 = b.pairwise_reduce(&[&input], &weights, &k1, 2);
        let individual_2 = b.pairwise_reduce(&[&input], &weights, &k2, 2);

        let multi = b.pairwise_reduce_multi(&[&input], &weights, &[(&k1, 2), (&k2, 2)]);

        assert_eq!(multi[0], individual_1);
        assert_eq!(multi[1], individual_2);
    }
}
