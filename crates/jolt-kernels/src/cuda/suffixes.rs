#![expect(
    dead_code,
    reason = "implementation target: step 2c (the device address phase) is the consumer; only the tests exercise this module so far"
)]

use cudarc::driver::{CudaSlice, PushKernelArg};

use super::context::CudaKernelContext;
use super::error::CudaError;

pub const NUM_SUFFIXES: usize = 43;

pub fn suffix_mle_batch(
    context: &CudaKernelContext,
    suffix: u32,
    bits: &[u128],
    lens: &[u8],
) -> Result<Vec<u64>, CudaError> {
    if bits.len() != lens.len() {
        return Err(CudaError::LengthMismatch {
            expected: bits.len(),
            got: lens.len(),
        });
    }
    if bits.is_empty() {
        return Ok(Vec::new());
    }
    if suffix as usize >= NUM_SUFFIXES {
        return Err(CudaError::LengthMismatch {
            expected: NUM_SUFFIXES,
            got: suffix as usize,
        });
    }
    let device_bits = upload_lookup_bits(context, bits)?;
    let device_lens = context.upload_u8_slice(lens)?;
    let mut out = context.alloc_u64(bits.len())?;
    let count = CudaKernelContext::count_of(bits.len())?;
    let mut builder = context.stream().launch_builder(context.sfx_eval_batch());
    let _ = builder.arg(&device_bits);
    let _ = builder.arg(&device_lens);
    let _ = builder.arg(&suffix);
    let _ = builder.arg(&mut out);
    let _ = builder.arg(&count);
    // SAFETY: thread `i < count` reads `bits[2i]`/`bits[2i+1]` of a `2 * count`
    // buffer and `lens[i]` of `count`, and writes only `out[i]` of `count`. `out`
    // is a distinct allocation and `suffix` is bounds-checked above.
    let _ = unsafe { builder.launch(CudaKernelContext::launch_config(count)) }?;
    context.stream().synchronize()?;
    context.download_u64(&out)
}

pub(super) fn upload_lookup_bits(
    context: &CudaKernelContext,
    bits: &[u128],
) -> Result<CudaSlice<u64>, CudaError> {
    let mut raw = Vec::with_capacity(bits.len() * 2);
    for &value in bits {
        raw.push(value as u64);
        raw.push((value >> 64) as u64);
    }
    context.upload_u64_slice(&raw)
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test module: device operations fail loudly"
)]
mod tests {
    use jolt_lookup_tables::lookup_bits::LookupBits;
    use jolt_lookup_tables::tables::suffixes::Suffixes;
    use jolt_lookup_tables::tables::LookupTableKind;
    use proptest::prelude::*;
    use strum::EnumCount;

    use super::super::context::shared_context;
    use super::{suffix_mle_batch, NUM_SUFFIXES};

    const RISCV_XLEN: usize = 64;

    const SPLIT_SUFFIXES: [usize; 3] = [19, 20, 22];

    const SHIFT_BOUNDED_SUFFIXES: [usize; 5] = [7, 18, 24, 31, 32];

    fn min_len_for(index: usize) -> u8 {
        match index {
            19 | 22 => 6,
            20 => 5,
            _ => 0,
        }
    }

    fn max_len_for(index: usize) -> u8 {
        match index {
            32 => 62,
            index if SHIFT_BOUNDED_SUFFIXES.contains(&index) => 126,
            _ => 128,
        }
    }

    fn clamp(index: usize, len: u8) -> u8 {
        len.clamp(min_len_for(index), max_len_for(index))
    }

    #[test]
    fn suffix_count_matches_the_rust_enum() {
        assert_eq!(
            NUM_SUFFIXES,
            Suffixes::COUNT,
            "device suffix count is out of sync with jolt-lookup-tables",
        );
    }

    #[test]
    fn split_suffixes_are_the_ones_with_a_length_floor() {
        let floored: Vec<usize> = (0..Suffixes::COUNT)
            .filter(|index| min_len_for(*index) > 0)
            .collect();
        assert_eq!(floored, SPLIT_SUFFIXES.to_vec());
    }

    #[test]
    fn table_reachable_suffixes_are_a_strict_subset() {
        let reachable = table_reachable_suffixes();
        assert_eq!(reachable.len(), 40, "table-reachable suffix count changed");
        let unreachable: Vec<usize> = (0..Suffixes::COUNT)
            .filter(|index| !reachable.iter().any(|suffix| *suffix as usize == *index))
            .collect();
        assert_eq!(
            unreachable,
            vec![13, 22, 32],
            "unreachable suffix set changed"
        );
    }

    fn table_reachable_suffixes() -> Vec<Suffixes> {
        let mut seen: Vec<Suffixes> = Vec::new();
        for table in LookupTableKind::<RISCV_XLEN>::iter() {
            for &suffix in table.suffixes() {
                if !seen.contains(&suffix) {
                    seen.push(suffix);
                }
            }
        }
        seen.sort_by_key(|suffix| *suffix as u8);
        seen
    }

    fn all_suffixes() -> Vec<Suffixes> {
        let by_index = |index: usize| {
            table_reachable_suffixes()
                .into_iter()
                .find(|suffix| *suffix as usize == index)
        };
        (0..Suffixes::COUNT)
            .map(|index| match index {
                13 => Suffixes::GreaterThan,
                22 => Suffixes::RightShiftPadding,
                32 => Suffixes::LeftShiftWHelper,
                other => by_index(other).expect("every other index is table-reachable"),
            })
            .collect()
    }

    proptest! {
        #[test]
        fn every_suffix_matches_the_rust_implementation(
            raw in prop::collection::vec((any::<u128>(), 0u8..=128), 1..64),
        ) {
            let Some(context) = shared_context() else { return Ok(()); };
            let bits: Vec<u128> = raw.iter().map(|&(value, _)| value).collect();
            let lens: Vec<u8> = raw.iter().map(|&(_, len)| len).collect();

            for (index, suffix) in all_suffixes().into_iter().enumerate() {
                let (bits, lens): (Vec<u128>, Vec<u8>) = bits
                    .iter()
                    .zip(&lens)
                    .map(|(&value, &len)| (value, clamp(index, len)))
                    .unzip();
                let expected: Vec<u64> = bits
                    .iter()
                    .zip(&lens)
                    .map(|(&value, &len)| suffix.suffix_mle(LookupBits::new(value, len as usize)))
                    .collect();
                let got = suffix_mle_batch(context, index as u32, &bits, &lens)
                    .expect("device suffix_mle_batch");
                prop_assert_eq!(
                    got,
                    expected,
                    "suffix {:?} (index {}) diverged",
                    suffix,
                    index
                );
            }
        }
    }

    #[test]
    fn every_suffix_matches_at_boundary_lengths() {
        let Some(context) = shared_context() else {
            return;
        };
        let mut bits = Vec::new();
        let mut lens = Vec::new();
        for len in [0u8, 1, 2, 8, 63, 64, 65, 127, 128] {
            for value in [
                0u128,
                1,
                u128::MAX,
                u128::from(u64::MAX),
                u128::from(u64::MAX) << 1,
                0x5555_5555_5555_5555_5555_5555_5555_5555,
                0xAAAA_AAAA_AAAA_AAAA_AAAA_AAAA_AAAA_AAAA,
            ] {
                bits.push(value);
                lens.push(len);
            }
        }

        for (index, suffix) in all_suffixes().into_iter().enumerate() {
            let (bits, lens): (Vec<u128>, Vec<u8>) = bits
                .iter()
                .zip(&lens)
                .map(|(&value, &len)| (value, clamp(index, len)))
                .unzip();
            let expected: Vec<u64> = bits
                .iter()
                .zip(&lens)
                .map(|(&value, &len)| suffix.suffix_mle(LookupBits::new(value, len as usize)))
                .collect();
            let got = suffix_mle_batch(context, index as u32, &bits, &lens)
                .expect("device suffix_mle_batch");
            assert_eq!(got, expected, "suffix {suffix:?} (index {index}) diverged");
        }
    }
}
