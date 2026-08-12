#![expect(
    dead_code,
    reason = "implementation target: step 2 (the instruction read+RAF address phase) is the consumer; only the tests exercise this module so far"
)]

use cudarc::driver::{LaunchConfig, PushKernelArg};
use jolt_field::{Field, Fr, MulPow2};

use crate::cuda::common::context::{CudaKernelContext, BLOCK};
use crate::cuda::common::device::{require_fr_slice, DeviceFrVec, LIMBS};
use crate::cuda::common::error::CudaError;

pub const CHUNK_LEN: usize = 8;
pub const CHUNK_SIZE: usize = 1 << CHUNK_LEN;

#[derive(Clone, Copy)]
pub struct SuffixRow {
    pub chunk: u32,
    pub suffix_left: u64,
    pub suffix_right: u64,
    pub suffix_value: u128,
    pub raf_flag: bool,
}

pub struct DeviceSuffixBuckets {
    shift_half: DeviceFrVec,
    shift_full: DeviceFrVec,
    left: DeviceFrVec,
    right: DeviceFrVec,
    identity: DeviceFrVec,
}

impl DeviceSuffixBuckets {
    pub fn shift_half(&self) -> &DeviceFrVec {
        &self.shift_half
    }

    pub fn shift_full(&self) -> &DeviceFrVec {
        &self.shift_full
    }

    pub fn left(&self) -> &DeviceFrVec {
        &self.left
    }

    pub fn right(&self) -> &DeviceFrVec {
        &self.right
    }

    pub fn identity(&self) -> &DeviceFrVec {
        &self.identity
    }
}

pub fn init_q_raf<F: Field>(
    context: &CudaKernelContext,
    rows: &[SuffixRow],
    u_evals: &[F],
    suffix_len: usize,
) -> Result<DeviceSuffixBuckets, CudaError> {
    if rows.len() != u_evals.len() {
        return Err(CudaError::LengthMismatch {
            expected: rows.len(),
            got: u_evals.len(),
        });
    }
    let mut chunks = Vec::with_capacity(rows.len());
    let mut left = Vec::with_capacity(rows.len());
    let mut right = Vec::with_capacity(rows.len());
    let mut value = Vec::with_capacity(rows.len() * 2);
    let mut flags = Vec::with_capacity(rows.len());
    for row in rows {
        if row.chunk as usize >= CHUNK_SIZE {
            return Err(CudaError::LengthMismatch {
                expected: CHUNK_SIZE,
                got: row.chunk as usize,
            });
        }
        chunks.push(row.chunk);
        left.push(row.suffix_left);
        right.push(row.suffix_right);
        value.push(row.suffix_value as u64);
        value.push((row.suffix_value >> 64) as u64);
        flags.push(u8::from(row.raf_flag));
    }

    let chunks = context.upload_u32_slice(&chunks)?;
    let left = context.upload_u64_slice(&left)?;
    let right = context.upload_u64_slice(&right)?;
    let value = context.upload_u64_slice(&value)?;
    let flags = context.upload_u8_slice(&flags)?;
    let u_evals = context.upload(require_fr_slice(u_evals)?)?;

    let mut buckets = context.alloc(LANES * CHUNK_SIZE)?;
    let rows_count = CudaKernelContext::count_of(rows.len())?;
    let chunk_count = CudaKernelContext::count_of(CHUNK_SIZE)?;
    let mut builder = context.stream().launch_builder(context.ps_init_q_raf());
    let _ = builder.arg(&chunks);
    let _ = builder.arg(&left);
    let _ = builder.arg(&right);
    let _ = builder.arg(&value);
    let _ = builder.arg(&flags);
    let _ = builder.arg(u_evals.limbs());
    let _ = builder.arg(buckets.limbs_mut());
    let _ = builder.arg(&rows_count);
    let _ = builder.arg(&chunk_count);
    // SAFETY: block `bucket < CHUNK_SIZE` strides its threads over `rows`,
    // reading `chunks[r]`, `suffix_left[r]`, `suffix_right[r]`,
    // `suffix_value[2r]`/`[2r+1]`, `raf_flags[r]` and `u_evals[r]` — all inside
    // buffers of `rows` (or `2 * rows`) elements. Writes: thread 0 of each block
    // writes `buckets[lane * chunk_count + bucket]` for `lane < LANES`, one slot
    // per (lane, bucket) of `LANES * CHUNK_SIZE`. Shared memory is
    // `BLOCK * LIMBS` u64s, matching `shared_mem_bytes`, with `__syncthreads()`
    // between every tree level and after each lane's write.
    let _ = unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (chunk_count, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: BLOCK * LIMBS as u32 * size_of::<u64>() as u32,
        })
    }?;
    context.stream().synchronize()?;

    let one = Fr::from(1u64);
    let half_scale = context.upload(&[one.mul_pow_2(suffix_len / 2)])?;
    let full_scale = context.upload(&[one.mul_pow_2(suffix_len)])?;
    let mut builder = context.stream().launch_builder(context.ps_scale_shift());
    let _ = builder.arg(buckets.limbs_mut());
    let _ = builder.arg(half_scale.limbs());
    let _ = builder.arg(full_scale.limbs());
    let _ = builder.arg(&chunk_count);
    // SAFETY: thread `i < chunk_count` read-modify-writes exactly
    // `buckets[i]` and `buckets[chunk_count + i]` — the two shift lanes, one
    // thread per element so uncontended — plus the two single-element scale
    // buffers. `buckets` holds `LANES * CHUNK_SIZE` elements.
    let _ = unsafe { builder.launch(CudaKernelContext::launch_config(chunk_count)) }?;
    context.stream().synchronize()?;

    let mut lanes = Vec::with_capacity(LANES);
    for lane in 0..LANES {
        lanes.push(buckets.slice_elements(lane * CHUNK_SIZE, CHUNK_SIZE)?);
    }
    let mut lanes = lanes.into_iter();
    Ok(DeviceSuffixBuckets {
        shift_half: lanes.next().ok_or(EMPTY)?,
        shift_full: lanes.next().ok_or(EMPTY)?,
        left: lanes.next().ok_or(EMPTY)?,
        right: lanes.next().ok_or(EMPTY)?,
        identity: lanes.next().ok_or(EMPTY)?,
    })
}

const LANES: usize = 5;

const EMPTY: CudaError = CudaError::InvariantViolation {
    reason: "the suffix bucket kernel produced fewer lanes than expected",
};

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test module: device operations fail loudly"
)]
mod tests {
    use jolt_field::{Fr, FromPrimitiveInt, MulPow2};
    use proptest::prelude::*;

    use super::{init_q_raf, SuffixRow, CHUNK_SIZE};
    use crate::cuda::common::context::shared_context;
    use crate::cuda::common::testing::fr;

    struct HostBuckets {
        shift_half: Vec<Fr>,
        shift_full: Vec<Fr>,
        left: Vec<Fr>,
        right: Vec<Fr>,
        identity: Vec<Fr>,
    }

    fn host_init_q_raf(rows: &[SuffixRow], u_evals: &[Fr], suffix_len: usize) -> HostBuckets {
        let zero = || vec![Fr::from_u64(0); CHUNK_SIZE];
        let mut shift_half_raw = zero();
        let mut shift_full_raw = zero();
        let mut left = zero();
        let mut right = zero();
        let mut identity = zero();

        for (row, &u) in rows.iter().zip(u_evals) {
            let chunk = row.chunk as usize;
            if row.raf_flag {
                shift_full_raw[chunk] += u;
                if row.suffix_value != 0 {
                    identity[chunk] += u * Fr::from_u128(row.suffix_value);
                }
            } else {
                shift_half_raw[chunk] += u;
                if row.suffix_left != 0 {
                    left[chunk] += u * Fr::from_u64(row.suffix_left);
                }
                if row.suffix_right != 0 {
                    right[chunk] += u * Fr::from_u64(row.suffix_right);
                }
            }
        }

        HostBuckets {
            shift_half: shift_half_raw
                .into_iter()
                .map(|value| value.mul_pow_2(suffix_len / 2))
                .collect(),
            shift_full: shift_full_raw
                .into_iter()
                .map(|value| value.mul_pow_2(suffix_len))
                .collect(),
            left,
            right,
            identity,
        }
    }

    fn arb_rows(count: usize, seed: u64, suffix_len: usize) -> (Vec<SuffixRow>, Vec<Fr>) {
        let mask = if suffix_len >= 128 {
            u128::MAX
        } else {
            (1u128 << suffix_len) - 1
        };
        let rows: Vec<SuffixRow> = (0..count)
            .map(|index| {
                let mixed = (index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ seed;
                let value = u128::from(mixed) << 13 | u128::from(mixed >> 7);
                SuffixRow {
                    chunk: (mixed % CHUNK_SIZE as u64) as u32,
                    suffix_left: mixed & 0xFFFF_FFFF,
                    suffix_right: (mixed >> 32) & 0xFFFF_FFFF,
                    suffix_value: value & mask,
                    raf_flag: mixed.is_multiple_of(3),
                }
            })
            .collect();
        let u_evals = (0..count).map(|index| fr(index as u64 + seed)).collect();
        (rows, u_evals)
    }

    proptest! {
        #[test]
        fn init_q_raf_matches_the_host_scan(
            log_t in 4usize..=10,
            seed in any::<u64>(),
            suffix_len in prop::sample::select(vec![8usize, 16, 32, 64, 120]),
        ) {
            let Some(context) = shared_context() else { return Ok(()); };
            let (rows, u_evals) = arb_rows(1usize << log_t, seed, suffix_len);
            let expected = host_init_q_raf(&rows, &u_evals, suffix_len);
            let got = init_q_raf(context, &rows, &u_evals, suffix_len)
                .expect("device init_q_raf");

            for (label, device, host) in [
                ("shift_half", got.shift_half(), &expected.shift_half),
                ("shift_full", got.shift_full(), &expected.shift_full),
                ("left", got.left(), &expected.left),
                ("right", got.right(), &expected.right),
                ("identity", got.identity(), &expected.identity),
            ] {
                prop_assert_eq!(
                    &device.to_host().expect("download"),
                    host,
                    "{} bucket diverged",
                    label
                );
            }
        }
    }

    #[test]
    fn init_q_raf_handles_every_row_landing_in_one_bucket() {
        let Some(context) = shared_context() else {
            return;
        };
        let count = 4096usize;
        let rows: Vec<SuffixRow> = (0..count)
            .map(|index| SuffixRow {
                chunk: 7,
                suffix_left: 1,
                suffix_right: 2,
                suffix_value: 3,
                raf_flag: index.is_multiple_of(2),
            })
            .collect();
        let u_evals: Vec<Fr> = (0..count).map(|index| fr(index as u64)).collect();
        let expected = host_init_q_raf(&rows, &u_evals, 16);
        let got = init_q_raf(context, &rows, &u_evals, 16).expect("device init_q_raf");
        assert_eq!(
            got.identity().to_host().expect("download"),
            expected.identity,
        );
        assert_eq!(got.left().to_host().expect("download"), expected.left);
        assert_eq!(
            got.shift_half().to_host().expect("download"),
            expected.shift_half,
        );
    }
}
