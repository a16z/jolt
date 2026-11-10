//! Equality weights for a window with one representative of its constant tail.
//! Window offsets stay in the original cycle domain, including on later GPUs.

use cudarc::driver::PushKernelArg;
use jolt_field::Fr;
use jolt_poly::{EqPolynomial, TensorEqTable};

use crate::cuda::common::context::CudaKernelContext;
use crate::cuda::common::device::{fr_limbs, DeviceFrVec};
use crate::cuda::common::devices::CycleWindow;
use crate::cuda::common::error::CudaError;

pub(super) fn eq_weights(
    context: &CudaKernelContext,
    point: &[Fr],
    window: &CycleWindow,
    retained: usize,
) -> Result<DeviceFrVec, CudaError> {
    let len = 1usize
        .checked_shl(point.len() as u32)
        .ok_or(CudaError::InvariantViolation {
            reason: "instruction equality point exceeds the cycle index width",
        })?;
    if retained == 0
        || retained > window.len
        || window.start > len
        || window.len > len - window.start
    {
        return Err(CudaError::InvariantViolation {
            reason: "instruction equality prefix lies outside its cycle window",
        });
    }
    let eq = TensorEqTable::new(point);
    let e_in = context.upload(eq.e_in())?;
    let e_out = context.upload(eq.e_out())?;
    let tail = fr_limbs(
        EqPolynomial::suffix_sum(point, window.start + retained - 1)
            - EqPolynomial::suffix_sum(point, window.end()),
    );
    let start = CudaKernelContext::count_of(window.start)?;
    let count = CudaKernelContext::count_of(retained)?;
    let in_bits = eq.e_in().len().trailing_zeros();
    let mut output = context.alloc_fr_unset(retained)?;
    let mut builder = context.stream().launch_builder(context.cr_eq_prefix());
    let _ = builder.arg(e_in.limbs());
    let _ = builder.arg(e_out.limbs());
    let _ = builder.arg(&in_bits);
    let _ = builder.arg(&start);
    let _ = builder.arg(&count);
    for limb in &tail {
        let _ = builder.arg(limb);
    }
    let _ = builder.arg(output.limbs_mut());
    // SAFETY: the checked window lies in the tensor's domain. Each thread below
    // `count` writes one output; the last receives the entire suffix weight.
    let _ = unsafe { builder.launch(CudaKernelContext::launch_config(count)) }?;
    context.stream().synchronize()?;
    Ok(output)
}
