#![expect(
    dead_code,
    reason = "implementation target: the address-phase bucket scans wire this once its kernels land"
)]

use cudarc::driver::PushKernelArg;

use super::context::CudaKernelContext;
use super::device::DeviceFrVec;
use super::error::CudaError;

pub const ACCUM_LIMBS: usize = 7;

pub(crate) fn alloc_slots(
    context: &CudaKernelContext,
    count: usize,
) -> Result<cudarc::driver::CudaSlice<u64>, CudaError> {
    context.alloc_u64(count * 2 * ACCUM_LIMBS)
}

pub(crate) fn finalize_slots(
    context: &CudaKernelContext,
    slots: &cudarc::driver::CudaSlice<u64>,
    count: usize,
) -> Result<DeviceFrVec, CudaError> {
    if slots.len() != count * 2 * ACCUM_LIMBS {
        return Err(CudaError::LengthMismatch {
            expected: count * 2 * ACCUM_LIMBS,
            got: slots.len(),
        });
    }
    let mut out = context.alloc(count)?;
    let count_arg = CudaKernelContext::count_of(count)?;
    let mut builder = context.stream().launch_builder(context.unr_reduce());
    let _ = builder.arg(slots);
    let _ = builder.arg(out.limbs_mut());
    let _ = builder.arg(&count_arg);
    // SAFETY: thread `b < count` reads its own `2 * ACCUM_LIMBS` lanes of `slots`,
    // whose length is checked above, and writes only `out[b]` of `count`; `out` is
    // a fresh allocation distinct from `slots`.
    let _ = unsafe { builder.launch(CudaKernelContext::launch_config(count_arg)) }?;
    context.stream().synchronize()?;
    Ok(out)
}

pub fn mul_u64_accumulate(
    context: &CudaKernelContext,
    values: &DeviceFrVec,
    multiplicands: &[u64],
    buckets: &[u32],
    bucket_count: usize,
) -> Result<DeviceFrVec, CudaError> {
    scatter(context, values, multiplicands, 1, buckets, bucket_count)
}

pub fn mul_u128_accumulate(
    context: &CudaKernelContext,
    values: &DeviceFrVec,
    multiplicands: &[u128],
    buckets: &[u32],
    bucket_count: usize,
) -> Result<DeviceFrVec, CudaError> {
    let mut words = Vec::with_capacity(multiplicands.len() * 2);
    for &value in multiplicands {
        words.push(value as u64);
        words.push((value >> 64) as u64);
    }
    scatter(context, values, &words, 2, buckets, bucket_count)
}

fn scatter(
    context: &CudaKernelContext,
    values: &DeviceFrVec,
    words: &[u64],
    mult_words: usize,
    buckets: &[u32],
    bucket_count: usize,
) -> Result<DeviceFrVec, CudaError> {
    let rows = buckets.len();
    if values.len() != rows || words.len() != rows * mult_words {
        return Err(CudaError::LengthMismatch {
            expected: rows,
            got: values.len().min(words.len() / mult_words.max(1)),
        });
    }
    if bucket_count == 0 {
        return Err(CudaError::InvariantViolation {
            reason: "an unreduced scatter needs at least one bucket",
        });
    }
    for &bucket in buckets {
        if bucket as usize >= bucket_count {
            return Err(CudaError::LengthMismatch {
                expected: bucket_count,
                got: bucket as usize,
            });
        }
    }

    let mut slots = alloc_slots(context, bucket_count)?;
    let device_words = context.upload_u64_slice(words)?;
    let device_buckets = context.upload_u32_slice(buckets)?;
    let count = CudaKernelContext::count_of(rows)?;
    let mult_words_arg = CudaKernelContext::count_of(mult_words)?;

    let mut builder = context.stream().launch_builder(context.unr_mul_scatter());
    let _ = builder.arg(values.limbs());
    let _ = builder.arg(&device_words);
    let _ = builder.arg(&mult_words_arg);
    let _ = builder.arg(&device_buckets);
    let _ = builder.arg(&mut slots);
    let _ = builder.arg(&count);
    // SAFETY: thread `j < rows` reads `values[j]` of `rows`,
    // `mults[j * mult_words .. + mult_words]` of `rows * mult_words`, and
    // `buckets[j]` of `rows`. It accumulates into
    // `slots[buckets[j] * 2 * ACCUM_LIMBS ..]`, in range because every bucket is
    // checked `< bucket_count` above and `slots` holds
    // `bucket_count * 2 * ACCUM_LIMBS` u64s. Concurrent accumulation into the
    // same bucket is `atomicAdd` on 32-bit-halved lanes, so no carry is lost.
    let _ = unsafe { builder.launch(CudaKernelContext::launch_config(count)) }?;
    context.stream().synchronize()?;

    finalize_slots(context, &slots, bucket_count)
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "test module: device operations fail loudly"
)]
mod tests {
    use jolt_field::{Fr, FromPrimitiveInt};
    use proptest::prelude::*;

    use super::super::context::shared_context;
    use super::super::testing::fr;
    use super::{mul_u128_accumulate, mul_u64_accumulate};

    fn host_scatter_u64(values: &[Fr], mults: &[u64], buckets: &[u32], count: usize) -> Vec<Fr> {
        let mut out = vec![Fr::from_u64(0); count];
        for ((value, &mult), &bucket) in values.iter().zip(mults).zip(buckets) {
            out[bucket as usize] += *value * Fr::from_u64(mult);
        }
        out
    }

    fn host_scatter_u128(values: &[Fr], mults: &[u128], buckets: &[u32], count: usize) -> Vec<Fr> {
        let mut out = vec![Fr::from_u64(0); count];
        for ((value, &mult), &bucket) in values.iter().zip(mults).zip(buckets) {
            out[bucket as usize] += *value * Fr::from_u128(mult);
        }
        out
    }

    proptest! {
        #[test]
        fn mul_u64_accumulate_matches_field_arithmetic(
            log_rows in 4usize..=12,
            seed in any::<u64>(),
            bucket_count in prop::sample::select(vec![1usize, 7, 256]),
        ) {
            let Some(context) = shared_context() else { return Ok(()); };
            let rows = 1usize << log_rows;
            let values: Vec<Fr> = (0..rows).map(|i| fr(seed + i as u64)).collect();
            let mults: Vec<u64> = (0..rows)
                .map(|i| {
                    let mixed = (i as u64)
                        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                        .wrapping_add(seed);
                    if mixed.is_multiple_of(5) { 0 } else { mixed }
                })
                .collect();
            let buckets: Vec<u32> = (0..rows)
                .map(|i| ((i as u64).wrapping_mul(2_654_435_761) % bucket_count as u64) as u32)
                .collect();

            let expected = host_scatter_u64(&values, &mults, &buckets, bucket_count);
            let uploaded = context.upload(&values).expect("upload values");
            let got = mul_u64_accumulate(context, &uploaded, &mults, &buckets, bucket_count)
                .expect("device mul_u64_accumulate")
                .to_host()
                .expect("download");
            prop_assert_eq!(got, expected);
        }

        #[test]
        fn mul_u128_accumulate_matches_field_arithmetic(
            log_rows in 4usize..=12,
            seed in any::<u64>(),
            bucket_count in prop::sample::select(vec![1usize, 7, 256]),
        ) {
            let Some(context) = shared_context() else { return Ok(()); };
            let rows = 1usize << log_rows;
            let values: Vec<Fr> = (0..rows).map(|i| fr(seed + i as u64)).collect();
            let mults: Vec<u128> = (0..rows)
                .map(|i| {
                    let mixed = (i as u64)
                        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                        .wrapping_add(seed);
                    if mixed.is_multiple_of(5) {
                        0
                    } else {
                        (u128::from(mixed) << 64) | u128::from(mixed.rotate_left(17))
                    }
                })
                .collect();
            let buckets: Vec<u32> = (0..rows)
                .map(|i| ((i as u64).wrapping_mul(2_654_435_761) % bucket_count as u64) as u32)
                .collect();

            let expected = host_scatter_u128(&values, &mults, &buckets, bucket_count);
            let uploaded = context.upload(&values).expect("upload values");
            let got = mul_u128_accumulate(context, &uploaded, &mults, &buckets, bucket_count)
                .expect("device mul_u128_accumulate")
                .to_host()
                .expect("download");
            prop_assert_eq!(got, expected);
        }
    }

    #[test]
    fn accumulator_survives_worst_case_magnitude() {
        let Some(context) = shared_context() else {
            return;
        };
        let rows = 1usize << 16;
        let values: Vec<Fr> = (0..rows).map(|i| fr(i as u64 + 1)).collect();
        let mults = vec![u128::MAX; rows];
        let buckets = vec![0u32; rows];
        let expected = host_scatter_u128(&values, &mults, &buckets, 1);
        let uploaded = context.upload(&values).expect("upload values");
        let got = mul_u128_accumulate(context, &uploaded, &mults, &buckets, 1)
            .expect("device mul_u128_accumulate")
            .to_host()
            .expect("download");
        assert_eq!(got, expected);
    }
}
