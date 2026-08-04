#![expect(
    dead_code,
    reason = "implementation target: step 2c (the device address phase) is the consumer; only the tests exercise this module so far"
)]

use cudarc::driver::PushKernelArg;
use jolt_field::Field;

use super::context::CudaKernelContext;
use super::device::{require_fr_slice, DeviceFrVec};
use super::error::CudaError;
use super::suffixes::upload_lookup_bits;

pub const NUM_PREFIXES: usize = 46;

pub fn prefix_evaluate_batch<F: Field>(
    context: &CudaKernelContext,
    prefix: u32,
    checkpoints: &[F],
    bits: &[u128],
    lens: &[u8],
    suffix_len: usize,
) -> Result<DeviceFrVec, CudaError> {
    if bits.len() != lens.len() {
        return Err(CudaError::LengthMismatch {
            expected: bits.len(),
            got: lens.len(),
        });
    }
    if checkpoints.len() != NUM_PREFIXES {
        return Err(CudaError::LengthMismatch {
            expected: NUM_PREFIXES,
            got: checkpoints.len(),
        });
    }
    if prefix as usize >= NUM_PREFIXES {
        return Err(CudaError::LengthMismatch {
            expected: NUM_PREFIXES,
            got: prefix as usize,
        });
    }
    let mut out = context.alloc(bits.len())?;
    if bits.is_empty() {
        return Ok(out);
    }
    let checkpoints = context.upload(require_fr_slice(checkpoints)?)?;
    let device_bits = upload_lookup_bits(context, bits)?;
    let device_lens = context.upload_u8_slice(lens)?;
    let count = CudaKernelContext::count_of(bits.len())?;
    let suffix_len = CudaKernelContext::count_of(suffix_len)?;
    let mut builder = context.stream().launch_builder(context.pfx_eval_batch());
    let _ = builder.arg(checkpoints.limbs());
    let _ = builder.arg(&device_bits);
    let _ = builder.arg(&device_lens);
    let _ = builder.arg(&prefix);
    let _ = builder.arg(&suffix_len);
    let _ = builder.arg(out.limbs_mut());
    let _ = builder.arg(&count);
    // SAFETY: thread `i < count` reads `bits[2i]`/`bits[2i+1]` of a `2 * count`
    // buffer, `lens[i]` of `count`, and any of the `NUM_PREFIXES` checkpoints
    // (length-checked above), and writes only `out[i]` of `count`. `out` is a
    // distinct allocation and `prefix` is bounds-checked above.
    let _ = unsafe { builder.launch(CudaKernelContext::launch_config(count)) }?;
    context.stream().synchronize()?;
    Ok(out)
}

pub fn default_checkpoints(context: &CudaKernelContext) -> Result<DeviceFrVec, CudaError> {
    let mut out = context.alloc(NUM_PREFIXES)?;
    let count = CudaKernelContext::count_of(NUM_PREFIXES)?;
    let mut builder = context
        .stream()
        .launch_builder(context.pfx_default_checkpoints());
    let _ = builder.arg(out.limbs_mut());
    let _ = builder.arg(&count);
    // SAFETY: thread `i < NUM_PREFIXES` writes only `out[i]` of `NUM_PREFIXES`
    // elements and reads nothing but its own index. `out` is a fresh allocation.
    let _ = unsafe { builder.launch(CudaKernelContext::launch_config(count)) }?;
    context.stream().synchronize()?;
    Ok(out)
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test module: device operations fail loudly"
)]
mod tests {
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_lookup_tables::lookup_bits::LookupBits;
    use jolt_lookup_tables::tables::prefixes::{PrefixEval, Prefixes, ALL_PREFIXES};
    use proptest::prelude::*;
    use strum::EnumCount;

    use super::super::context::shared_context;
    use super::super::testing::fr;
    use super::{default_checkpoints, prefix_evaluate_batch, NUM_PREFIXES};

    const CHUNK_LEN: usize = 8;
    const ADDRESS_BITS: usize = 128;

    #[test]
    fn prefix_count_matches_the_rust_enum() {
        assert_eq!(
            NUM_PREFIXES,
            Prefixes::COUNT,
            "device prefix count is out of sync with jolt-lookup-tables",
        );
        assert_eq!(ALL_PREFIXES.len(), NUM_PREFIXES);
    }

    fn checkpoints(seed: u64) -> Vec<Fr> {
        ALL_PREFIXES
            .iter()
            .enumerate()
            .map(|(index, prefix)| {
                if seed == 0 {
                    prefix.default_checkpoint::<Fr>().value()
                } else {
                    fr(seed + index as u64)
                }
            })
            .collect()
    }

    fn host_evaluate(
        prefix: Prefixes,
        checkpoints: &[Fr],
        bits: &[u128],
        lens: &[u8],
        suffix_len: usize,
    ) -> Vec<Fr> {
        let wrapped: Vec<PrefixEval<Fr>> =
            checkpoints.iter().copied().map(PrefixEval::from).collect();
        bits.iter()
            .zip(lens)
            .map(|(&value, &len)| {
                prefix
                    .evaluate::<Fr>(&wrapped, LookupBits::new(value, len as usize), suffix_len)
                    .value()
            })
            .collect()
    }

    fn chunk_points() -> (Vec<u128>, Vec<u8>) {
        let bits = (0..1u128 << CHUNK_LEN).collect();
        let lens = vec![CHUNK_LEN as u8; 1 << CHUNK_LEN];
        (bits, lens)
    }

    proptest! {
        #[test]
        fn every_prefix_matches_the_rust_implementation(
            seed in 1u64..1_000_000,
            phase in 0usize..(ADDRESS_BITS / CHUNK_LEN),
        ) {
            let Some(context) = shared_context() else { return Ok(()); };
            let suffix_len = ADDRESS_BITS - (phase + 1) * CHUNK_LEN;
            let checkpoints = checkpoints(seed);
            let (bits, lens) = chunk_points();

            for (index, prefix) in ALL_PREFIXES.iter().enumerate() {
                let expected = host_evaluate(*prefix, &checkpoints, &bits, &lens, suffix_len);
                let got = prefix_evaluate_batch(
                    context, index as u32, &checkpoints, &bits, &lens, suffix_len,
                )
                .expect("device prefix_evaluate_batch")
                .to_host()
                .expect("download");
                prop_assert_eq!(
                    got,
                    expected,
                    "prefix {:?} (index {}) diverged at suffix_len {}",
                    prefix,
                    index,
                    suffix_len
                );
            }
        }
    }

    #[test]
    fn every_prefix_matches_from_default_checkpoints() {
        let Some(context) = shared_context() else {
            return;
        };
        let checkpoints = checkpoints(0);
        let (bits, lens) = chunk_points();
        for suffix_len in [0usize, CHUNK_LEN, ADDRESS_BITS - CHUNK_LEN] {
            for (index, prefix) in ALL_PREFIXES.iter().enumerate() {
                let expected = host_evaluate(*prefix, &checkpoints, &bits, &lens, suffix_len);
                let got = prefix_evaluate_batch(
                    context,
                    index as u32,
                    &checkpoints,
                    &bits,
                    &lens,
                    suffix_len,
                )
                .expect("device prefix_evaluate_batch")
                .to_host()
                .expect("download");
                assert_eq!(
                    got, expected,
                    "prefix {prefix:?} (index {index}) diverged at suffix_len {suffix_len}",
                );
            }
        }
    }

    #[test]
    fn default_checkpoints_match_the_rust_defaults() {
        let Some(context) = shared_context() else {
            return;
        };
        let expected: Vec<Fr> = ALL_PREFIXES
            .iter()
            .map(|prefix| prefix.default_checkpoint::<Fr>().value())
            .collect();
        let got = default_checkpoints(context)
            .expect("device default_checkpoints")
            .to_host()
            .expect("download");
        assert_eq!(got, expected);
        assert_ne!(
            expected[Prefixes::Eq as usize],
            Fr::from_u64(0),
            "the Eq checkpoint seeds a product family and must not default to zero",
        );
    }
}
