use cudarc::driver::{LaunchConfig, PushKernelArg};
use jolt_claims::protocols::jolt::geometry::dimensions::OUTER_UNISKIP_DOMAIN_SIZE;
use jolt_field::Field;
use jolt_poly::lagrange::{centered_lagrange_evals, interpolate_to_coeffs, poly_mul};
use jolt_poly::{EqPolynomial, UnivariatePoly};
use jolt_r1cs::constraint::ConstraintMatrices;
use jolt_r1cs::constraints::jolt::spartan_outer_row_weights;

use super::columns::{DeviceR1csInputs, LinearForms};
use crate::cuda::common::context::{CudaKernelContext, BLOCK};
use crate::cuda::common::dense_product::DeviceDenseProduct;
use crate::cuda::common::device::{require_fr_slice, LIMBS};
use crate::cuda::common::error::CudaError;

pub const EXTENDED_SIZE: usize = 2 * OUTER_UNISKIP_DOMAIN_SIZE - 1;

pub const STRIP: usize = 4;

pub fn extended_nodes() -> Vec<(usize, i64)> {
    let domain_start = -((OUTER_UNISKIP_DOMAIN_SIZE as i64 - 1) / 2);
    let extended_start = -((EXTENDED_SIZE as i64 - 1) / 2);
    let domain_end = domain_start + OUTER_UNISKIP_DOMAIN_SIZE as i64;
    (0..EXTENDED_SIZE)
        .filter_map(|position| {
            let node = extended_start + position as i64;
            (node < domain_start || node >= domain_end).then_some((position, node))
        })
        .collect()
}

pub fn node_field<F: Field>(node: i64) -> F {
    if node >= 0 {
        F::from_u64(node as u64)
    } else {
        -F::from_u64(node.unsigned_abs())
    }
}

pub fn push_forms<F: Field>(
    forms: &mut LinearForms,
    matrices: &ConstraintMatrices<F>,
    columns: &[usize],
    point: F,
    stream: F,
) -> Result<(), CudaError> {
    let weights =
        spartan_outer_row_weights(point, stream).map_err(|_| CudaError::InvariantViolation {
            reason: "the Spartan outer uni-skip domain is not a valid centered integer domain",
        })?;
    let weighted = matrices.weighted_columns(&weights, columns).map_err(|_| {
        CudaError::InvariantViolation {
            reason: "the Spartan outer constraint matrices do not span the relation's variables",
        }
    })?;
    let constants = matrices
        .public_column_contributions(&weights, 0, F::one())
        .map_err(|_| CudaError::InvariantViolation {
            reason: "the Spartan outer constraint matrices have no public column",
        })?;
    forms.push(&weighted.a, constants.a)?;
    forms.push(&weighted.b, constants.b)?;
    Ok(())
}

pub fn extended_evals<F: Field>(
    context: &CudaKernelContext,
    inputs: &DeviceR1csInputs,
    matrices: &ConstraintMatrices<F>,
    tau: &[F],
    log_t: usize,
) -> Result<Vec<F>, CudaError> {
    let nodes = extended_nodes();
    let variables = super::witness::VARIABLES;
    let columns: Vec<usize> = (1..=variables).collect();

    let mut forms = LinearForms::new();
    for &(_, node) in &nodes {
        let point = node_field::<F>(node);
        push_forms(&mut forms, matrices, &columns, point, F::zero())?;
        push_forms(&mut forms, matrices, &columns, point, F::one())?;
    }
    let device_forms = forms.upload(context)?;

    let (e_in, e_out, in_bits) = split_eq_tables(context, &tau[..=log_t])?;

    let cycles = inputs.cycles();
    let threads = cycles.div_ceil(STRIP);
    let blocks = u32::try_from(threads.div_ceil(BLOCK as usize).max(1)).map_err(|_| {
        CudaError::InvariantViolation {
            reason: "the Spartan outer uni-skip launch exceeds a u32 block count",
        }
    })?;
    let lanes = u32::try_from(nodes.len()).map_err(|_| CudaError::InvariantViolation {
        reason: "the Spartan outer uni-skip launch exceeds a u32 lane count",
    })?;
    let mut partials = context.alloc(nodes.len() * blocks as usize)?;

    let count = CudaKernelContext::count_of(cycles)?;
    let strip = CudaKernelContext::count_of(STRIP)?;
    let bits = CudaKernelContext::count_of(in_bits)?;
    let mut builder = context.stream().launch_builder(context.so_uniskip());
    let _ = builder.arg(inputs.narrow());
    let _ = builder.arg(inputs.wide());
    let _ = builder.arg(inputs.flags());
    device_forms.bind_args(&mut builder);
    let _ = builder.arg(e_in.limbs());
    let _ = builder.arg(e_out.limbs());
    let _ = builder.arg(&bits);
    let _ = builder.arg(&count);
    let _ = builder.arg(&strip);
    let _ = builder.arg(partials.limbs_mut());
    // SAFETY: thread `(blockIdx.x, threadIdx.x)` of node `blockIdx.y < lanes`
    // covers cycles `base .. base + STRIP` and breaks at `cycles`, so every
    // column read is inside `narrow`'s `cycles * NARROW`, `wide`'s
    // `cycles * WIDE * 2` and `flags`' `cycles` entries. Its two forms per stream
    // index `offsets`/`counts`/`constants` at `(node * 2 + stream) * 2 (+ 1)`,
    // below the `4 * lanes` forms uploaded above, and read `terms`/`coeffs`
    // within each form's own `count` entries. The eq lookup is bounded because
    // `in_bits` comes from `e_in`'s length and `(t << 1) | stream < 2 * cycles`.
    // It writes only `partials[node * blocks + blockIdx.x]` of
    // `lanes * blocks`. Shared memory is `BLOCK * LIMBS` u64s, matching
    // `shared_mem_bytes`, and the block reduction sits outside every early exit
    // so all threads reach each `__syncthreads()`.
    let _ = unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (blocks, lanes, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: BLOCK * LIMBS as u32 * size_of::<u64>() as u32,
        })
    }?;

    let totals = DeviceDenseProduct::reduce_lanes(context, partials, lanes, blocks)?;
    let host = totals.to_host()?;
    let mut values = vec![F::zero(); EXTENDED_SIZE];
    for (&(position, _), value) in nodes.iter().zip(&host) {
        values[position] =
            crate::cuda::common::device::fr_into(*value).ok_or(CudaError::NotImplemented {
                kernel: "CUDA kernels support only the BN254 scalar field",
            })?;
    }
    Ok(values)
}

pub fn split_eq_tables<F: Field>(
    context: &CudaKernelContext,
    point: &[F],
) -> Result<
    (
        crate::cuda::common::device::DeviceFrVec,
        crate::cuda::common::device::DeviceFrVec,
        usize,
    ),
    CudaError,
> {
    if point.is_empty() {
        return Err(CudaError::InvariantViolation {
            reason: "an eq factor pair needs at least one variable",
        });
    }
    let split = point.len() / 2;
    let in_bits = point.len() - split;
    let (outer, inner) = point.split_at(split);
    let e_out = context.upload(require_fr_slice(&EqPolynomial::<F>::evals(outer, None))?)?;
    let e_in = context.upload(require_fr_slice(&EqPolynomial::<F>::evals(inner, None))?)?;
    Ok((e_in, e_out, in_bits))
}

pub fn first_round_poly<F: Field>(
    extended: &[F],
    tau_high: F,
) -> Result<UnivariatePoly<F>, CudaError> {
    let domain_start = -((OUTER_UNISKIP_DOMAIN_SIZE as i64 - 1) / 2);
    let extended_start = -((EXTENDED_SIZE as i64 - 1) / 2);
    let kernel_values =
        centered_lagrange_evals::<F>(OUTER_UNISKIP_DOMAIN_SIZE, tau_high).map_err(|_| {
            CudaError::InvariantViolation {
                reason: "the Spartan outer uni-skip domain is not a valid centered integer domain",
            }
        })?;
    let kernel_coefficients = interpolate_to_coeffs(domain_start, &kernel_values);
    let t1_coefficients = interpolate_to_coeffs(extended_start, extended);
    Ok(UnivariatePoly::new(poly_mul(
        &kernel_coefficients,
        &t1_coefficients,
    )))
}
