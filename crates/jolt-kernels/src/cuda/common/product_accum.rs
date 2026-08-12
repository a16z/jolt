#![cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "implementation target: step 2 (the RAM value-check Toom round message) is the first non-test caller"
    )
)]

use cudarc::driver::PushKernelArg;
use jolt_field::Fr;

use super::context::CudaKernelContext;
use super::device::DeviceFrVec;
use super::error::CudaError;

pub const PRODUCT_SLOTS: usize = 8;

pub(crate) fn alloc_slots(
    context: &CudaKernelContext,
    count: usize,
) -> Result<cudarc::driver::CudaSlice<u64>, CudaError> {
    context.alloc_u64(count * 2 * PRODUCT_SLOTS)
}

pub(crate) fn finalize_slots(
    context: &CudaKernelContext,
    slots: &cudarc::driver::CudaSlice<u64>,
    count: usize,
) -> Result<DeviceFrVec, CudaError> {
    if slots.len() != count * 2 * PRODUCT_SLOTS {
        return Err(CudaError::LengthMismatch {
            expected: count * 2 * PRODUCT_SLOTS,
            got: slots.len(),
        });
    }
    let mut out = context.alloc(count)?;
    let count_arg = CudaKernelContext::count_of(count)?;
    let mut builder = context.stream().launch_builder(context.pa_reduce());
    let _ = builder.arg(slots);
    let _ = builder.arg(out.limbs_mut());
    let _ = builder.arg(&count_arg);
    // SAFETY: thread `b < count` reads its own `2 * PRODUCT_SLOTS` lanes of
    // `slots`, whose length is checked above, and writes only `out[b]` of
    // `count`; `out` is a fresh allocation distinct from `slots`.
    let _ = unsafe { builder.launch(CudaKernelContext::launch_config(count_arg)) }?;
    context.stream().synchronize()?;
    Ok(out)
}

pub fn product_accumulate(
    context: &CudaKernelContext,
    left: &DeviceFrVec,
    right: &DeviceFrVec,
    buckets: &[u32],
    bucket_count: usize,
) -> Result<DeviceFrVec, CudaError> {
    let rows = buckets.len();
    if left.len() != rows || right.len() != rows {
        return Err(CudaError::LengthMismatch {
            expected: rows,
            got: left.len().min(right.len()),
        });
    }
    if bucket_count == 0 {
        return Err(CudaError::InvariantViolation {
            reason: "a product accumulation needs at least one bucket",
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
    let device_buckets = context.upload_u32_slice(buckets)?;
    let count = CudaKernelContext::count_of(rows)?;

    let mut builder = context.stream().launch_builder(context.pa_scatter());
    let _ = builder.arg(left.limbs());
    let _ = builder.arg(right.limbs());
    let _ = builder.arg(&device_buckets);
    let _ = builder.arg(&mut slots);
    let _ = builder.arg(&count);
    // SAFETY: thread `j < rows` reads `left[j]` and `right[j]`, both of `rows`
    // elements (checked above), and `buckets[j]` of `rows`. It accumulates into
    // `slots[buckets[j] * 2 * PRODUCT_SLOTS ..]`, in range because every bucket
    // is checked `< bucket_count` above and `slots` holds
    // `bucket_count * 2 * PRODUCT_SLOTS` u64s. Concurrent accumulation into the
    // same bucket is `atomicAdd` on 32-bit-halved lanes, so no carry is lost.
    let _ = unsafe { builder.launch(CudaKernelContext::launch_config(count)) }?;
    context.stream().synchronize()?;

    finalize_slots(context, &slots, bucket_count)
}

pub fn reduce_one(context: &CudaKernelContext, left: Fr, right: Fr) -> Result<Fr, CudaError> {
    let left = context.upload(&[left])?;
    let right = context.upload(&[right])?;
    product_accumulate(context, &left, &right, &[0], 1)?.first()
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test module: device operations fail loudly"
)]
mod tests {
    use ark_bn254::Fr as LegacyFr;
    use jolt_field::Fr;
    use jolt_prover_legacy::field::JoltField;
    use proptest::prelude::*;

    use super::super::context::shared_context;
    use super::super::testing::fr;
    use super::{product_accumulate, reduce_one};

    fn legacy_product(a: Fr, b: Fr) -> Fr {
        legacy_product_sum(&[(a, b)])
    }

    fn legacy_product_sum(pairs: &[(Fr, Fr)]) -> Fr {
        let mut accumulator = <LegacyFr as JoltField>::UnreducedProductAccum::default();
        for &(a, b) in pairs {
            accumulator += LegacyFr::from(a).mul_to_product_accum(LegacyFr::from(b));
        }
        Fr::from(<LegacyFr as JoltField>::reduce_product_accum(accumulator))
    }

    #[test]
    fn legacy_accumulator_agrees_with_plain_field_arithmetic() {
        let pairs: Vec<(Fr, Fr)> = (0..7u64).map(|i| (fr(i * 3 + 1), fr(i * 5 + 2))).collect();
        let mut plain = Fr::from(0u64);
        for &(a, b) in &pairs {
            plain += a * b;
        }
        assert_eq!(legacy_product_sum(&pairs), plain);
    }

    proptest! {
        #[test]
        fn single_product_matches_legacy(a in any::<u64>().prop_map(fr), b in any::<u64>().prop_map(fr)) {
            let Some(context) = shared_context() else { return Ok(()); };
            let expected = legacy_product(a, b);
            let got = reduce_one(context, a, b).expect("device product");
            prop_assert_eq!(got, expected);
        }
    }

    #[test]
    fn deep_accumulation_into_one_bucket_matches_legacy() {
        let Some(context) = shared_context() else {
            return;
        };
        let rows = 4096usize;
        let pairs: Vec<(Fr, Fr)> = (0..rows as u64)
            .map(|i| (fr(i * 2 + 1), fr(i * 3 + 2)))
            .collect();
        let buckets = vec![0u32; rows];
        let left = context
            .upload(&pairs.iter().map(|&(a, _)| a).collect::<Vec<_>>())
            .expect("upload left");
        let right = context
            .upload(&pairs.iter().map(|&(_, b)| b).collect::<Vec<_>>())
            .expect("upload right");
        let got = product_accumulate(context, &left, &right, &buckets, 1)
            .expect("device accumulate")
            .to_host()
            .expect("download");
        assert_eq!(got, vec![legacy_product_sum(&pairs)]);
    }

    proptest! {
        #[test]
        fn accumulated_products_match_legacy(
            seeds in proptest::collection::vec((any::<u64>(), any::<u64>(), 0u32..4), 1..40),
        ) {
            let Some(context) = shared_context() else { return Ok(()); };
            let bucket_count = 4usize;
            let pairs: Vec<(Fr, Fr)> =
                seeds.iter().map(|&(a, b, _)| (fr(a), fr(b))).collect();
            let buckets: Vec<u32> = seeds.iter().map(|&(_, _, k)| k).collect();

            let left = context
                .upload(&pairs.iter().map(|&(a, _)| a).collect::<Vec<_>>())
                .expect("upload left");
            let right = context
                .upload(&pairs.iter().map(|&(_, b)| b).collect::<Vec<_>>())
                .expect("upload right");

            let got = product_accumulate(context, &left, &right, &buckets, bucket_count)
                .expect("device accumulate")
                .to_host()
                .expect("download");

            let expected: Vec<Fr> = (0..bucket_count as u32)
                .map(|bucket| {
                    let members: Vec<(Fr, Fr)> = pairs
                        .iter()
                        .zip(&buckets)
                        .filter(|(_, &k)| k == bucket)
                        .map(|(&pair, _)| pair)
                        .collect();
                    legacy_product_sum(&members)
                })
                .collect();
            prop_assert_eq!(got, expected);
        }
    }
}
