use cudarc::driver::{LaunchConfig, PushKernelArg};
use jolt_field::Fr;
use jolt_lookup_tables::lookup_bits::LookupBits;
use jolt_lookup_tables::tables::prefixes::{PrefixEval, ALL_PREFIXES};

use crate::cuda::common::context::{CudaKernelContext, BLOCK};
use crate::cuda::common::device::DeviceFrVec;
use crate::cuda::common::error::CudaError;

pub const NUM_PREFIXES: usize = ALL_PREFIXES.len();
pub const HINT_POINTS: usize = 2;

pub fn default_checkpoints(context: &CudaKernelContext) -> Result<DeviceFrVec, CudaError> {
    context.upload(
        &ALL_PREFIXES
            .iter()
            .map(|prefix| prefix.default_checkpoint::<Fr>().value())
            .collect::<Vec<_>>(),
    )
}

/// Materialize the bounded phase grid using the protocol's prefix definitions.
/// It has at most 256 entries per prefix, independent of trace length.
pub fn phase_tables(
    context: &CudaKernelContext,
    checkpoints: &DeviceFrVec,
    phase_bits: usize,
    suffix_len: usize,
) -> Result<DeviceFrVec, CudaError> {
    let checkpoints: Vec<PrefixEval<Fr>> =
        checkpoints.to_host()?.into_iter().map(Into::into).collect();
    if checkpoints.len() != NUM_PREFIXES {
        return Err(CudaError::LengthMismatch {
            expected: NUM_PREFIXES,
            got: checkpoints.len(),
        });
    }
    let mut tables = Vec::with_capacity(NUM_PREFIXES << phase_bits);
    for prefix in ALL_PREFIXES {
        tables.extend((0..1usize << phase_bits).map(|index| {
            prefix
                .evaluate(
                    &checkpoints,
                    LookupBits::new(index as u128, phase_bits),
                    suffix_len,
                )
                .value()
        }));
    }
    context.upload(&tables)
}

pub fn prefix_mle_round(
    context: &CudaKernelContext,
    tables: &DeviceFrVec,
    half: usize,
) -> Result<DeviceFrVec, CudaError> {
    if half == 0 || tables.len() != NUM_PREFIXES * 2 * half {
        return Err(CudaError::LengthMismatch {
            expected: NUM_PREFIXES * 2 * half,
            got: tables.len(),
        });
    }
    let mut out = context.alloc(HINT_POINTS * NUM_PREFIXES * half)?;
    let half_arg = CudaKernelContext::count_of(half)?;
    let prefixes = CudaKernelContext::count_of(NUM_PREFIXES)?;
    let mut builder = context.stream().launch_builder(context.pfx_mle_round());
    let _ = builder.arg(tables.limbs());
    let _ = builder.arg(&half_arg);
    let _ = builder.arg(&prefixes);
    let _ = builder.arg(out.limbs_mut());
    // SAFETY: each (prefix, index < half) reads the two halves of its
    // length-2*half row and writes one entry in each of the two output planes.
    let _ = unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (half_arg.div_ceil(BLOCK), prefixes, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        })
    }?;
    Ok(out)
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test fixture and CUDA failures")]
mod tests {
    use super::super::address_driver::bind_lanes;
    use super::*;
    use crate::cuda::common::context::shared_context;
    use jolt_field::{Ring, Zero};
    use jolt_poly::{BindingOrder, Polynomial};

    #[test]
    fn every_prefix_matches_the_live_table_through_all_phases() {
        let Some(context) = shared_context() else {
            return;
        };
        let mut checkpoints = default_checkpoints(context).unwrap();
        for phase in 0..16 {
            let mut device = phase_tables(context, &checkpoints, 8, 128 - 8 * (phase + 1)).unwrap();
            let mut reference: Vec<Polynomial<Fr>> = device
                .to_host()
                .unwrap()
                .chunks_exact(256)
                .map(|row| Polynomial::new(row.to_vec()))
                .collect();
            for round in 0..8 {
                let half = 128 >> round;
                let output = prefix_mle_round(context, &device, half)
                    .unwrap()
                    .to_host()
                    .unwrap();
                for (index, row) in reference.iter().enumerate() {
                    for b in 0..half {
                        let lo = row.evals()[b];
                        let hi = row.evals()[half + b];
                        assert_eq!(output[index * half + b], lo);
                        assert_eq!(output[(NUM_PREFIXES + index) * half + b], hi + hi - lo);
                    }
                }
                let challenge = match round {
                    0 => Fr::zero(),
                    1 => Fr::from_u64(1),
                    _ => Fr::from_u64((phase * 8 + round + 7) as u64),
                };
                bind_lanes(context, &mut [&mut device], half * 2, challenge).unwrap();
                for row in &mut reference {
                    row.bind_with_order(challenge, BindingOrder::HighToLow);
                }
            }
            assert_eq!(
                device.to_host().unwrap(),
                reference
                    .iter()
                    .map(|row| row.evals()[0])
                    .collect::<Vec<_>>()
            );
            checkpoints = device;
        }
    }
}
