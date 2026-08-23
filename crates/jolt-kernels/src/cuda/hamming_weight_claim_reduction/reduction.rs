use cudarc::driver::{LaunchConfig, PushKernelArg};
use jolt_field::Field;

use crate::cuda::common::context::{CudaKernelContext, BLOCK};
use crate::cuda::common::device::{fr_into, require_fr, require_fr_slice, DeviceFrVec, LIMBS};
use crate::cuda::common::error::CudaError;
use crate::cuda::common::one_hot_fold::{FoldTuning, OneHotShards};
use crate::cuda::common::primitives::reduce_lanes;

const LANES: usize = 2;

pub struct DeviceHammingWeightReduction {
    folded: DeviceFrVec,
    weights: DeviceFrVec,
    polys: usize,
    len: usize,
}

impl DeviceHammingWeightReduction {
    pub fn new<F: Field>(
        context: &CudaKernelContext,
        shards: &OneHotShards,
        cycle_point: &[F],
        address_point: &[F],
        virtualization_points: &[Vec<F>],
        gamma: F,
    ) -> Result<Self, CudaError> {
        let columns = shards.whole()?;
        let polys = columns.polys();
        let addresses = columns.addresses();
        let chunk_bits = addresses.ilog2() as usize;
        if virtualization_points.len() != polys || address_point.len() != chunk_bits {
            return Err(CudaError::InvariantViolation {
                reason: "the hamming reduction needs one virtualization point per polynomial and \
                         one address coordinate per chunk bit",
            });
        }
        if virtualization_points
            .iter()
            .any(|point| point.len() != chunk_bits)
        {
            return Err(CudaError::InvariantViolation {
                reason: "a hamming reduction virtualization point has the wrong variable count",
            });
        }

        let folded = shards.fold(cycle_point, FoldTuning::default())?;
        let booleanity = context.eq_evals(require_fr_slice(address_point)?)?;
        let mut virtualization = context.alloc(polys * addresses)?;
        for (index, point) in virtualization_points.iter().enumerate() {
            let table = context.eq_evals(require_fr_slice(point)?)?;
            context.copy_into(&mut virtualization, index * addresses, &table)?;
        }

        let mut powers = Vec::with_capacity(3 * polys);
        let mut power = F::one();
        for _ in 0..3 * polys {
            powers.push(power);
            power *= gamma;
        }
        let powers = context.upload(require_fr_slice(&powers)?)?;

        let mut weights = context.alloc(polys * addresses)?;
        let count = CudaKernelContext::count_of(polys * addresses)?;
        let poly_count = CudaKernelContext::count_of(polys)?;
        let address_count = CudaKernelContext::count_of(addresses)?;
        let mut builder = context.stream().launch_builder(context.hwr_weights());
        let _ = builder.arg(booleanity.limbs());
        let _ = builder.arg(virtualization.limbs());
        let _ = builder.arg(powers.limbs());
        let _ = builder.arg(weights.limbs_mut());
        let _ = builder.arg(&poly_count);
        let _ = builder.arg(&address_count);
        // SAFETY: thread `idx < polys * addresses` reads `booleanity[idx % addresses]`
        // of `addresses`, `virtualization[idx]` of `polys * addresses`, and
        // `powers[3p]`, `powers[3p + 1]`, `powers[3p + 2]` for `p = idx / addresses`,
        // inside `powers`'s `3 * polys` elements. It writes only `out[idx]` of
        // `polys * addresses`; index sets are disjoint across threads and `out` is a
        // fresh allocation.
        let _ = unsafe { builder.launch(CudaKernelContext::launch_config(count)) }?;
        context.stream().synchronize()?;

        Ok(Self {
            folded,
            weights,
            polys,
            len: addresses,
        })
    }

    pub fn round_lanes<F: Field>(&self, context: &CudaKernelContext) -> Result<(F, F), CudaError> {
        let half = self.len / 2;
        if half == 0 {
            return Err(CudaError::LengthMismatch {
                expected: 2,
                got: self.len,
            });
        }
        let half_count = CudaKernelContext::count_of(half)?;
        let blocks = half_count.div_ceil(BLOCK).max(1);
        let mut partials = context.alloc(LANES * blocks as usize)?;
        let poly_count = CudaKernelContext::count_of(self.polys)?;

        let mut builder = context.stream().launch_builder(context.hwr_message());
        let _ = builder.arg(self.folded.limbs());
        let _ = builder.arg(self.weights.limbs());
        let _ = builder.arg(&poly_count);
        let _ = builder.arg(&half_count);
        let _ = builder.arg(partials.limbs_mut());
        // SAFETY: thread `y < half` reads `folded[p * 2 * half + 2y]`,
        // `folded[p * 2 * half + 2y + 1]` and the same two slots of `weights` for
        // every `p < polys` — both buffers hold `polys * 2 * half` elements, since
        // `len == 2 * half` is the current row length of both — and writes only
        // `partials[lane * gridDim.x + blockIdx.x]` for `lane < 2`, of
        // `2 * blocks`. Shared memory is `BLOCK * LIMBS` u64s, matching
        // `shared_mem_bytes`, and the block reduction sits outside the `y < half`
        // guard so every thread reaches each `__syncthreads()`.
        let _ = unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: BLOCK * LIMBS as u32 * size_of::<u64>() as u32,
            })
        }?;
        context.stream().synchronize()?;

        let totals = reduce_lanes(
            context,
            partials,
            CudaKernelContext::count_of(LANES)?,
            blocks,
        )?;
        let host = totals.to_host()?;
        let unsupported = || CudaError::NotImplemented {
            kernel: "CUDA kernels support only the BN254 scalar field",
        };
        let at_one = fr_into(host[0]).ok_or_else(unsupported)?;
        let at_infinity = fr_into(host[1]).ok_or_else(unsupported)?;
        Ok((at_one, at_infinity))
    }

    pub fn bind<F: Field>(
        &mut self,
        context: &CudaKernelContext,
        challenge: F,
    ) -> Result<(), CudaError> {
        if self.len < 2 {
            return Err(CudaError::LengthMismatch {
                expected: 2,
                got: self.len,
            });
        }
        let challenge = require_fr(challenge)?;
        self.folded = context.bind_rows(&self.folded, self.len, challenge)?;
        self.weights = context.bind_rows(&self.weights, self.len, challenge)?;
        self.len /= 2;
        Ok(())
    }

    pub fn reduced_claims<F: Field>(&self) -> Result<Vec<F>, CudaError> {
        Self::row_values(&self.folded, self.len, self.polys)
    }

    pub fn weight_claims<F: Field>(&self) -> Result<Vec<F>, CudaError> {
        Self::row_values(&self.weights, self.len, self.polys)
    }

    fn row_values<F: Field>(
        table: &DeviceFrVec,
        len: usize,
        polys: usize,
    ) -> Result<Vec<F>, CudaError> {
        if len != 1 {
            return Err(CudaError::LengthMismatch {
                expected: 1,
                got: len,
            });
        }
        let host = table.to_host()?;
        if host.len() != polys {
            return Err(CudaError::LengthMismatch {
                expected: polys,
                got: host.len(),
            });
        }
        host.into_iter()
            .map(|value| {
                fr_into(value).ok_or(CudaError::NotImplemented {
                    kernel: "CUDA kernels support only the BN254 scalar field",
                })
            })
            .collect()
    }
}
