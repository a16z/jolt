use cudarc::driver::{LaunchConfig, PushKernelArg};
use jolt_field::Fr;

use crate::cuda::common::context::{CudaKernelContext, BLOCK};
use crate::cuda::common::device::{fr_limbs, DeviceFrVec};
use crate::cuda::common::error::CudaError;

pub(super) fn bind_lanes(
    context: &CudaKernelContext,
    lanes: &mut [&mut DeviceFrVec],
    stride: usize,
    challenge: Fr,
) -> Result<(), CudaError> {
    if stride < 2 {
        for lane in lanes.iter_mut() {
            **lane = context.alloc(0)?;
        }
        return Ok(());
    }
    let half = stride / 2;
    let mut outputs = Vec::with_capacity(lanes.len());
    let mut counts = Vec::with_capacity(lanes.len());
    for lane in lanes.iter() {
        if !lane.len().is_multiple_of(stride) {
            return Err(CudaError::LengthMismatch {
                expected: stride,
                got: lane.len(),
            });
        }
        let count = (lane.len() / stride) * half;
        counts.push(u32::try_from(count).map_err(|_| CudaError::LengthMismatch {
            expected: u32::MAX as usize,
            got: count,
        })?);
        outputs.push(context.alloc(count)?);
    }
    let max_count = counts.iter().copied().max().unwrap_or(0);
    if max_count == 0 {
        for (lane, out) in lanes.iter_mut().zip(outputs) {
            **lane = out;
        }
        return Ok(());
    }

    let inputs = context.device_pointers(&lanes.iter().map(|lane| &**lane).collect::<Vec<_>>())?;
    let targets = context.device_pointers(&outputs.iter().collect::<Vec<_>>())?;
    let device_counts = context.upload_u32_slice(&counts)?;
    let limbs = fr_limbs(challenge);
    let lane_count = u32::try_from(lanes.len()).map_err(|_| CudaError::LengthMismatch {
        expected: u32::MAX as usize,
        got: lanes.len(),
    })?;
    let half_arg = CudaKernelContext::count_of(half)?;
    let stride_arg = CudaKernelContext::count_of(stride)?;
    let mut builder = context.stream().launch_builder(context.ap_bind_lanes());
    let _ = builder.arg(&inputs);
    let _ = builder.arg(&targets);
    let _ = builder.arg(&device_counts);
    let _ = builder.arg(&limbs[0]);
    let _ = builder.arg(&limbs[1]);
    let _ = builder.arg(&limbs[2]);
    let _ = builder.arg(&limbs[3]);
    let _ = builder.arg(&half_arg);
    let _ = builder.arg(&stride_arg);
    let _ = builder.arg(&max_count);
    // SAFETY: `blockIdx.y` indexes `lanes` because the grid's y extent is
    // `lane_count`, so `counts[lane]`, `in_ptrs[lane]` and `out_ptrs[lane]` are
    // all inside their `lanes.len()`-element uploads. A thread returns unless
    // `i < counts[lane]`, and `counts[lane] == (len / stride) * half` for that
    // lane, so `column = i / half < len / stride` and `b = i % half < half`: the
    // reads `in[column * stride + b]` and `in[column * stride + b + half]` are
    // inside that lane's `len` elements, and the write `out[i]` is inside its
    // freshly allocated `counts[lane]`. Every output is a distinct new
    // allocation, so no lane aliases another's input. The challenge travels by
    // value as four limbs, so no buffer backs it.
    let _ = unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (max_count.div_ceil(BLOCK).max(1), lane_count, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        })
    }?;
    for (lane, out) in lanes.iter_mut().zip(outputs) {
        **lane = out;
    }
    Ok(())
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test module: device operations fail loudly"
)]
mod tests {
    use jolt_field::Fr;

    use super::bind_lanes;
    use crate::cuda::common::context::shared_context;
    use crate::cuda::common::device::DeviceFrVec;
    use crate::cuda::common::testing::fr;

    fn host_bind(values: &[Fr], stride: usize, challenge: Fr) -> Vec<Fr> {
        let half = stride / 2;
        let mut out = Vec::with_capacity(values.len() / 2);
        for column in values.chunks_exact(stride) {
            for b in 0..half {
                let (lo, hi) = (column[b], column[b + half]);
                out.push(lo + challenge * (hi - lo));
            }
        }
        out
    }

    #[test]
    fn bind_lanes_matches_the_host_bind() {
        let Some(context) = shared_context() else {
            return;
        };
        const STRIDE: usize = 256;
        const ROUNDS: usize = 3;
        let widths = [37usize, 1, 1, 1, 1, 1, 1, 1];
        let mut host: Vec<Vec<Fr>> = widths
            .iter()
            .enumerate()
            .map(|(lane, &columns)| {
                (0..columns * STRIDE)
                    .map(|i| fr((lane as u64 + 1) * 7_001 + i as u64 + 1))
                    .collect()
            })
            .collect();
        let mut device: Vec<DeviceFrVec> = host
            .iter()
            .map(|values| context.upload(values).expect("upload a lane"))
            .collect();

        let mut stride = STRIDE;
        for round in 0..ROUNDS {
            let challenge = fr(0x00C0_FFEE + round as u64 * 91);
            {
                let mut lanes: Vec<&mut DeviceFrVec> = device.iter_mut().collect();
                bind_lanes(context, &mut lanes, stride, challenge).expect("batched bind");
            }
            for values in &mut host {
                *values = host_bind(values, stride, challenge);
            }
            stride /= 2;

            for (lane, (got, want)) in device.iter().zip(&host).enumerate() {
                assert_eq!(
                    got.to_host().expect("download a bound lane"),
                    *want,
                    "round {round} lane {lane}: the batched bind diverged from the host bind",
                );
            }
        }
        assert!(
            host.iter().all(|values| !values.is_empty())
                && host[0].iter().any(|value| *value != host[0][0]),
            "the fixture collapsed, so the comparison proves nothing",
        );
    }
}
