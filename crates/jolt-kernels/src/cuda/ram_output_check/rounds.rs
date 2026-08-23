use cudarc::driver::{LaunchConfig, PushKernelArg};
use jolt_field::Field;
use jolt_poly::{BindingOrder, UnivariatePoly};

use crate::cuda::common::context::{CudaKernelContext, BLOCK};
use crate::cuda::common::device::{fr_into, require_fr, require_fr_slice, DeviceFrVec, LIMBS};
use crate::cuda::common::error::CudaError;
use crate::cuda::common::primitives::reduce_lanes;
use crate::cuda::common::split_eq::DeviceSplitEq;

const LANES: usize = 2;

const ROWS: usize = 3;

pub struct DeviceOutputCheck<F: Field> {
    tables: DeviceFrVec,
    eq: DeviceSplitEq<F>,
    len: usize,
}

impl<F: Field> DeviceOutputCheck<F> {
    pub fn new(
        context: &CudaKernelContext,
        io_mask: &[F],
        val_final: &[F],
        val_io: &[F],
        address_point: &[F],
    ) -> Result<Self, CudaError> {
        let len = 1usize << address_point.len();
        if io_mask.len() != len || val_final.len() != len || val_io.len() != len {
            return Err(CudaError::LengthMismatch {
                expected: len,
                got: val_final.len(),
            });
        }

        let mut rows = Vec::with_capacity(ROWS * len);
        rows.extend_from_slice(io_mask);
        rows.extend_from_slice(val_final);
        rows.extend_from_slice(val_io);
        Ok(Self {
            tables: context.upload(require_fr_slice(&rows)?)?,
            eq: DeviceSplitEq::new(context, address_point, BindingOrder::LowToHigh)?,
            len,
        })
    }

    pub fn round_message(
        &self,
        context: &CudaKernelContext,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, CudaError> {
        let half = self.len / 2;
        if half == 0 {
            return Err(CudaError::LengthMismatch {
                expected: 2,
                got: self.len,
            });
        }
        let e_in_len = self.eq.e_in_len();
        if self.eq.e_out_current().len() * e_in_len != half {
            return Err(CudaError::LengthMismatch {
                expected: half,
                got: self.eq.e_out_current().len() * e_in_len,
            });
        }

        let half_count = CudaKernelContext::count_of(half)?;
        let blocks = half_count.div_ceil(BLOCK).max(1);
        let mut partials = context.alloc(LANES * blocks as usize)?;
        let e_in_arg = CudaKernelContext::count_of(e_in_len)?;
        let num_x_in_bits = e_in_len.max(1).ilog2();

        let mut builder = context.stream().launch_builder(context.roc_message());
        let _ = builder.arg(self.tables.limbs());
        let _ = builder.arg(&half_count);
        let _ = builder.arg(self.eq.e_in_current().limbs());
        let _ = builder.arg(&e_in_arg);
        let _ = builder.arg(self.eq.e_out_current().limbs());
        let _ = builder.arg(&num_x_in_bits);
        let _ = builder.arg(partials.limbs_mut());
        // SAFETY: thread `g < half` reads slots `2g` and `2g + 1` of each of the
        // three rows of `tables`, whose length is `3 * 2 * half` because `len` is
        // the current row length; plus `e_in[g & mask]` and
        // `e_out[g >> num_x_in_bits]`, both bounded because
        // `e_out.len() * e_in.len() == half` is checked above. It writes only
        // `partials[lane * gridDim.x + blockIdx.x]` for `lane < 2`, of
        // `2 * blocks`. Shared memory is `BLOCK * LIMBS` u64s, matching
        // `shared_mem_bytes`, and the block reduction sits outside the `g < half`
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
        let constant = fr_into(host[0]).ok_or_else(unsupported)?;
        let quadratic = fr_into(host[1]).ok_or_else(unsupported)?;
        Ok(self
            .eq
            .gruen_poly_deg_3(constant, quadratic, previous_claim))
    }

    pub fn bind(&mut self, context: &CudaKernelContext, challenge: F) -> Result<(), CudaError> {
        if self.len < 2 {
            return Err(CudaError::LengthMismatch {
                expected: 2,
                got: self.len,
            });
        }
        self.tables = context.bind_rows(&self.tables, self.len, require_fr(challenge)?)?;
        self.len /= 2;
        self.eq.bind(challenge);
        Ok(())
    }

    pub fn finals(&self) -> Result<[F; ROWS], CudaError> {
        if self.len != 1 {
            return Err(CudaError::LengthMismatch {
                expected: 1,
                got: self.len,
            });
        }
        let host = self.tables.to_host()?;
        if host.len() != ROWS {
            return Err(CudaError::LengthMismatch {
                expected: ROWS,
                got: host.len(),
            });
        }
        let unsupported = || CudaError::NotImplemented {
            kernel: "CUDA kernels support only the BN254 scalar field",
        };
        Ok([
            fr_into(host[0]).ok_or_else(unsupported)?,
            fr_into(host[1]).ok_or_else(unsupported)?,
            fr_into(host[2]).ok_or_else(unsupported)?,
        ])
    }
}
