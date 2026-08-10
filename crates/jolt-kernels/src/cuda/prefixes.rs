#![expect(
    dead_code,
    reason = "implementation target: step 2c (the device address phase) is the consumer; only the tests exercise this module so far"
)]

use cudarc::driver::PushKernelArg;
use jolt_field::{Field, Fr};

use super::context::CudaKernelContext;
use super::device::{require_fr, require_fr_slice, DeviceFrVec};
use super::error::CudaError;
use super::suffixes::upload_lookup_bits;

pub const NUM_PREFIXES: usize = 46;
const ADDRESS_BITS: usize = 128;

pub fn prefix_mle_batch<F: Field>(
    context: &CudaKernelContext,
    prefix: u32,
    checkpoints: &[F],
    r_x: Option<F>,
    c: u32,
    bits: &[u128],
    lens: &[u8],
    round: usize,
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
    let b_len = lens.first().copied().unwrap_or(0) as usize;
    if lens.iter().any(|&len| len as usize != b_len) {
        return Err(CudaError::InvariantViolation {
            reason: "legacy's prefix MLE derives suffix_len from a single b.len per round, so \
                     every point in a batch must share it",
        });
    }
    let suffix_len = ADDRESS_BITS
        .checked_sub(round + b_len + 1)
        .ok_or(CudaError::InvariantViolation {
            reason: "the prefix MLE round and point width exceed the address width",
        })?;

    let mut out = context.alloc(bits.len())?;
    if bits.is_empty() {
        return Ok(out);
    }
    let device_checkpoints = context.upload(require_fr_slice(checkpoints)?)?;
    let challenge = context.upload(&[match r_x {
        Some(value) => require_fr(value)?,
        None => Fr::from(0u64),
    }])?;
    let device_bits = upload_lookup_bits(context, bits)?;
    let device_lens = context.upload_u8_slice(lens)?;
    let count = CudaKernelContext::count_of(bits.len())?;
    let has_r_x = u32::from(r_x.is_some());
    let round_arg = CudaKernelContext::count_of(round)?;
    let suffix_len_arg = CudaKernelContext::count_of(suffix_len)?;

    let mut builder = context.stream().launch_builder(context.pfx_mle_batch());
    let _ = builder.arg(device_checkpoints.limbs());
    let _ = builder.arg(&device_bits);
    let _ = builder.arg(&device_lens);
    let _ = builder.arg(&prefix);
    let _ = builder.arg(challenge.limbs());
    let _ = builder.arg(&has_r_x);
    let _ = builder.arg(&c);
    let _ = builder.arg(&round_arg);
    let _ = builder.arg(&suffix_len_arg);
    let _ = builder.arg(out.limbs_mut());
    let _ = builder.arg(&count);
    // SAFETY: thread `i < count` reads `bits[2i]`/`bits[2i+1]` of a `2 * count`
    // buffer, `lens[i]` of `count`, the single-element challenge buffer, and any
    // of the `NUM_PREFIXES` checkpoints (length-checked above); it writes only
    // `out[i]` of `count`. `out` is a distinct allocation and `prefix` is
    // bounds-checked above.
    let _ = unsafe { builder.launch(CudaKernelContext::launch_config(count)) }?;
    context.stream().synchronize()?;
    Ok(out)
}

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
    use ark_bn254::Fr as LegacyFr;
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_lookup_tables::lookup_bits::LookupBits;
    use jolt_lookup_tables::tables::prefixes::{PrefixEval, Prefixes, ALL_PREFIXES};
    use jolt_lookup_tables::XLEN as RISCV_XLEN;
    use jolt_prover_legacy::utils::lookup_bits::LookupBits as LegacyBits;
    use jolt_prover_legacy::zkvm::lookup_table::prefixes::{
        PrefixCheckpoint as LegacyCheckpoint, PrefixEval as LegacyPrefixEval,
        Prefixes as LegacyPrefixes,
    };
    use proptest::prelude::*;
    use strum::EnumCount;

    use super::super::context::shared_context;
    use super::super::testing::fr;
    use super::{default_checkpoints, prefix_evaluate_batch, prefix_mle_batch, NUM_PREFIXES};

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

    fn legacy_checkpoints(seed: u64, upto_round: usize) -> Vec<LegacyCheckpoint<LegacyFr>> {
        let mut checkpoints: Vec<LegacyCheckpoint<LegacyFr>> =
            vec![None.into(); LegacyPrefixes::COUNT];
        for pair in 0..upto_round / 2 {
            let round = 2 * pair + 1;
            let suffix_len = ADDRESS_BITS - (round / CHUNK_LEN + 1) * CHUNK_LEN;
            LegacyPrefixes::update_checkpoints::<RISCV_XLEN, LegacyFr, LegacyFr>(
                &mut checkpoints,
                legacy_fr(seed + 2 * pair as u64),
                legacy_fr(seed + 2 * pair as u64 + 1),
                round,
                suffix_len,
            );
        }
        checkpoints
    }

    fn chunk_points_of_len(len: usize) -> (Vec<u128>, Vec<u8>) {
        let bits = (0..1u128 << len).collect();
        let lens = vec![len as u8; 1 << len];
        (bits, lens)
    }

    fn legacy_fr(seed: u64) -> LegacyFr {
        LegacyFr::from(fr(seed))
    }

    fn legacy_value(eval: LegacyPrefixEval<LegacyFr>) -> Fr {
        let one = [eval];
        let slice: &[LegacyPrefixEval<LegacyFr>] = &one;
        Fr::from(slice[LegacyPrefixes::LowerWord])
    }

    fn legacy_prefix_mle(
        prefix: LegacyPrefixes,
        checkpoints: &[LegacyCheckpoint<LegacyFr>],
        r_x: Option<Fr>,
        c: u32,
        bits: &[u128],
        lens: &[u8],
        round: usize,
    ) -> Vec<Fr> {
        let r_x = r_x.map(LegacyFr::from);
        bits.iter()
            .zip(lens)
            .map(|(&value, &len)| {
                legacy_value(prefix.prefix_mle::<RISCV_XLEN, LegacyFr, LegacyFr>(
                    checkpoints,
                    r_x,
                    c,
                    LegacyBits::new(value, len as usize),
                    round,
                ))
            })
            .collect()
    }

    fn device_checkpoints(
        legacy: &[LegacyCheckpoint<LegacyFr>],
        defaults: &[Fr],
    ) -> Result<Vec<Fr>, TestCaseError> {
        prop_assert_eq!(legacy.len(), defaults.len());
        Ok(legacy
            .iter()
            .zip(defaults)
            .map(|(checkpoint, &default)| {
                let slice: &[LegacyCheckpoint<LegacyFr>] = std::slice::from_ref(checkpoint);
                slice[LegacyPrefixes::LowerWord].map_or(default, Fr::from)
            })
            .collect())
    }

    proptest! {
        #[test]
        fn every_prefix_mle_matches_legacy(
            seed in 1u64..1_000_000,
            round in 0usize..ADDRESS_BITS,
            c in prop::sample::select(vec![0u32, 2]),
        ) {
            let Some(context) = shared_context() else { return Ok(()); };
            let legacy = legacy_checkpoints(seed, round);
            let host_checkpoints = device_checkpoints(&legacy, &checkpoints(0))?;
            let (bits, lens) = chunk_points_of_len(CHUNK_LEN - 1 - round % CHUNK_LEN);

            let r_x = if round % 2 == 1 { Some(fr(seed + 977)) } else { None };

            for (index, prefix) in
                <LegacyPrefixes as strum::IntoEnumIterator>::iter().enumerate()
            {
                let expected = legacy_prefix_mle(
                    prefix, &legacy, r_x, c, &bits, &lens, round,
                );
                let got = prefix_mle_batch(
                    context,
                    index as u32,
                    &host_checkpoints,
                    r_x,
                    c,
                    &bits,
                    &lens,
                    round,
                )
                .expect("device prefix_mle_batch")
                .to_host()
                .expect("download");
                prop_assert_eq!(
                    got,
                    expected,
                    "prefix index {} diverged at round {} (c = {}, r_x = {})",
                    index,
                    round,
                    c,
                    r_x.is_some()
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
