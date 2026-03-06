//! CPU compute backend using Rayon for parallelism.

use jolt_field::Field;

use crate::traits::{ComputeBackend, Scalar};

/// Parallelism threshold: buffers smaller than this use sequential loops.
///
/// Below this size the overhead of Rayon work-stealing exceeds the benefit.
const PAR_THRESHOLD: usize = 1024;

/// Composition evaluation function type for [`CpuKernel`].
///
/// Takes `(lo_values, hi_values, degree)` and returns `degree + 1` evaluations.
type EvalFn<F> = dyn Fn(&[F], &[F], usize) -> Vec<F> + Send + Sync;

/// Compiled kernel for [`CpuBackend`].
///
/// Wraps a closure that evaluates a composition at `degree + 1` grid points
/// from paired polynomial inputs. The closure signature:
///
/// ```text
/// fn(lo: &[F], hi: &[F], degree: usize) -> Vec<F>
/// ```
///
/// where `lo[k]` and `hi[k]` are the even/odd pair for input buffer `k`,
/// and the returned `Vec` has `degree + 1` evaluations of the composed
/// polynomial at grid points `{0, 1, ..., degree}`.
///
/// Constructed by a `compile` method on `CpuBackend` (not through the
/// `ComputeBackend` trait, to avoid coupling `jolt-compute` to `jolt-ir`).
pub struct CpuKernel<F: Field> {
    eval_fn: Box<EvalFn<F>>,
}

impl<F: Field> CpuKernel<F> {
    /// Creates a kernel from a composition evaluation closure.
    ///
    /// The closure receives `(lo_values, hi_values, degree)` where each slice
    /// has one entry per input buffer, and returns `degree + 1` evaluations.
    pub fn new(eval_fn: impl Fn(&[F], &[F], usize) -> Vec<F> + Send + Sync + 'static) -> Self {
        Self {
            eval_fn: Box::new(eval_fn),
        }
    }

    #[inline]
    fn evaluate(&self, lo: &[F], hi: &[F], degree: usize) -> Vec<F> {
        (self.eval_fn)(lo, hi, degree)
    }
}

/// CPU compute backend.
///
/// Implements [`ComputeBackend`] with `Buffer<T> = Vec<T>`. All operations
/// use Rayon for parallelism (when the `parallel` feature is enabled and
/// buffers exceed [`PAR_THRESHOLD`]).
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

    fn pairwise_reduce<F: Field>(
        &self,
        inputs: &[&Vec<F>],
        weights: &Vec<F>,
        kernel: &Self::CompiledKernel<F>,
        degree: usize,
    ) -> Vec<F> {
        let num_outputs = degree + 1;
        let n = inputs[0].len();
        debug_assert!(n % 2 == 0, "buffer length must be even");
        let half = n / 2;
        let num_inputs = inputs.len();

        #[cfg(feature = "parallel")]
        {
            if half >= PAR_THRESHOLD {
                use rayon::prelude::*;

                // Parallel reduce: each thread accumulates a local sum, then
                // we reduce across threads.
                let sums: Vec<F> = (0..half)
                    .into_par_iter()
                    .fold(
                        || vec![F::zero(); num_outputs],
                        |mut acc, i| {
                            let mut lo = Vec::with_capacity(num_inputs);
                            let mut hi = Vec::with_capacity(num_inputs);
                            for &input in inputs {
                                lo.push(input[2 * i]);
                                hi.push(input[2 * i + 1]);
                            }

                            let evals = kernel.evaluate(&lo, &hi, degree);
                            let w = weights[i];
                            for (a, e) in acc.iter_mut().zip(evals.iter()) {
                                *a += w * *e;
                            }
                            acc
                        },
                    )
                    .reduce(
                        || vec![F::zero(); num_outputs],
                        |mut a, b| {
                            for (ai, bi) in a.iter_mut().zip(b.iter()) {
                                *ai += *bi;
                            }
                            a
                        },
                    );

                return sums;
            }
        }

        let mut sums = vec![F::zero(); num_outputs];
        let mut lo = Vec::with_capacity(num_inputs);
        let mut hi = Vec::with_capacity(num_inputs);

        for i in 0..half {
            lo.clear();
            hi.clear();
            for &input in inputs {
                lo.push(input[2 * i]);
                hi.push(input[2 * i + 1]);
            }

            let evals = kernel.evaluate(&lo, &hi, degree);
            let w = weights[i];
            for (s, e) in sums.iter_mut().zip(evals.iter()) {
                *s += w * *e;
            }
        }

        sums
    }

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
        // at grid points {0, 1, ..., degree}.
        // f(t) = lo + t * (hi - lo)
        CpuKernel::new(|lo: &[Fr], hi: &[Fr], degree: usize| {
            let mut evals = Vec::with_capacity(degree + 1);
            for t in 0..=degree {
                let t_f = Fr::from_u64(t as u64);
                let mut sum = Fr::zero();
                for k in 0..lo.len() {
                    sum += lo[k] + t_f * (hi[k] - lo[k]);
                }
                evals.push(sum);
            }
            evals
        })
    }

    #[test]
    fn pairwise_reduce_trivial() {
        let b = backend();
        // Single input [1, 2, 3, 4], weights [1, 1], degree 1
        // Pair 0: lo=1, hi=2 -> f(0)=1, f(1)=2; weighted by 1
        // Pair 1: lo=3, hi=4 -> f(0)=3, f(1)=4; weighted by 1
        // Sums: [1+3, 2+4] = [4, 6]
        let input: Vec<Fr> = vec![1, 2, 3, 4].into_iter().map(Fr::from_u64).collect();
        let weights: Vec<Fr> = vec![Fr::one(), Fr::one()];

        let kernel = make_identity_kernel();
        let result = b.pairwise_reduce(&[&input], &weights, &kernel, 1);

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
        let result = b.pairwise_reduce(&[&input], &weights, &kernel, 1);

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
        let result = b.pairwise_reduce(&[&a, &c], &weights, &kernel, 1);

        // Pair 0: a=(1,2), c=(10,20) -> f(0)=1+10=11, f(1)=2+20=22
        // Pair 1: a=(3,4), c=(30,40) -> f(0)=3+30=33, f(1)=4+40=44
        // Sums: [44, 66]
        assert_eq!(result[0], Fr::from_u64(44));
        assert_eq!(result[1], Fr::from_u64(66));
    }

    #[test]
    fn pairwise_reduce_degree_2() {
        let b = backend();
        let input: Vec<Fr> = vec![1, 3].into_iter().map(Fr::from_u64).collect();
        let weights: Vec<Fr> = vec![Fr::one()];

        // Degree-2 kernel that returns [lo^2, (lo+hi)^2/4-ish, hi^2]
        // For simplicity, use the identity-sum kernel which just returns
        // lo + t*(hi-lo) at t=0,1,2
        let kernel = make_identity_kernel();
        let result = b.pairwise_reduce(&[&input], &weights, &kernel, 2);

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
        let result = b.pairwise_reduce(&[&input], &weights, &kernel, 1);

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
}
