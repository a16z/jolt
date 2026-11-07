//! Host-side integer forms for Spartan outer uni-skip.
//!
//! Each row supplies an A-side guard and a B-side value. CUDA derives compact
//! term descriptors from the canonical R1CS matrices; optimized evaluates the
//! corresponding expressions directly on typed witness rows. Construction
//! checks the supported layout, coefficient bounds, and guard range `[-3, 3]`
//! required by the device's integer arithmetic and packed guard keys.
//!
//! Bounded integer extension weights handle rows that overflow the device's
//! guard table; grouped rows use finite differences in `spartan_outer.cu`.

use cudarc::driver::{CudaSlice, LaunchArgs, PushKernelArg};
use jolt_claims::protocols::composed::r1cs::spartan_outer_row_weights;
use jolt_claims::protocols::jolt::geometry::dimensions::OUTER_UNISKIP_DOMAIN_SIZE;
use jolt_field::JoltField;
use jolt_poly::lagrange::{centered_domain_start, centered_lagrange_evals};
use jolt_r1cs::constraint::ConstraintMatrices;

use super::uniskip::{extended_nodes, node_field};
use super::witness::{KIND_FLAG, LAYOUT, VARIABLES};
use crate::cuda::common::context::CudaKernelContext;
use crate::cuda::common::device::LIMBS;
use crate::cuda::common::error::CudaError;
use crate::cuda::common::unreduced::ACCUM_LIMBS;

const GROUPS: usize = 32;
const DOMAIN: usize = OUTER_UNISKIP_DOMAIN_SIZE;
pub(super) const NODES: usize = DOMAIN - 1;
pub(super) const SHARED_BYTES: usize = GROUPS * size_of::<u32>()
    + ((GROUPS * DOMAIN + NODES) * 2 * ACCUM_LIMBS + GROUPS * NODES * LIMBS) * size_of::<u64>();

const _: () = assert!(SHARED_BYTES <= 48 * 1024);

// Source kind 3 in bits 8..9 denotes the implicit R1CS input z[0] = 1.
// The CUDA decoder supplies 1 directly, without loading a witness column.
const CONSTANT: u32 = 3 << 8;

pub(super) struct GuardForms {
    offsets: Vec<u32>,
    terms: Vec<u32>,
    extension: Vec<u32>,
}

impl GuardForms {
    pub(super) fn new<F: JoltField>(matrices: &ConstraintMatrices<F>) -> Result<Self, CudaError> {
        if DOMAIN != 10
            || matrices.num_vars <= VARIABLES
            || matrices
                .a
                .iter()
                .chain(&matrices.b)
                .flatten()
                .any(|&(column, coefficient)| column > VARIABLES && !coefficient.is_zero())
        {
            return Err(CudaError::NotImplemented {
                kernel: "CUDA guard grouping requires the ordinary ten-node Spartan outer domain",
            });
        }
        let mut forms = Self {
            offsets: vec![0],
            terms: Vec::new(),
            extension: Vec::new(),
        };
        let start = centered_domain_start(DOMAIN).map_err(|_| CudaError::InvariantViolation {
            reason: "the Spartan outer integer domain is invalid",
        })?;
        let columns: Vec<usize> = (1..=VARIABLES).collect();
        for stream in [F::zero(), F::one()] {
            for row in 0..DOMAIN {
                let weights =
                    spartan_outer_row_weights(node_field::<F>(start + row as i64), stream)
                        .map_err(|_| CudaError::InvariantViolation {
                            reason: "the Spartan outer row domain is invalid",
                        })?;
                let weighted = matrices.weighted_columns(&weights, &columns).map_err(|_| {
                    CudaError::InvariantViolation {
                        reason: "the Spartan outer matrix does not cover its input columns",
                    }
                })?;
                let constants = matrices
                    .public_column_contributions(&weights, 0, F::one())
                    .map_err(|_| CudaError::InvariantViolation {
                        reason: "the Spartan outer matrix has no constant column",
                    })?;
                forms.push(&weighted.a, constants.a, true)?;
                forms.push(&weighted.b, constants.b, false)?;
            }
        }
        for (_, node) in extended_nodes() {
            let weights = centered_lagrange_evals(DOMAIN, node_field::<F>(node)).map_err(|_| {
                CudaError::InvariantViolation {
                    reason: "the Spartan outer extension domain is invalid",
                }
            })?;
            let mut gain = 0i128;
            for weight in weights {
                let coefficient = Self::integer(weight, 1 << 20)?;
                gain += coefficient.abs();
                forms.extension.push(coefficient as i32 as u32);
            }
            if gain >= 1 << 20 {
                return Err(CudaError::NotImplemented {
                    kernel: "CUDA guard extension requires an absolute coefficient sum below 2^20",
                });
            }
        }
        Ok(forms)
    }

    fn integer<F: JoltField>(value: F, bound: i128) -> Result<i128, CudaError> {
        for (value, sign) in [(value, 1), (-value, -1)] {
            let bytes = value.to_bytes_le_vec();
            let mut word = [0u8; 16];
            let count = bytes.len().min(word.len());
            word[..count].copy_from_slice(&bytes[..count]);
            let magnitude = u128::from_le_bytes(word);
            if magnitude <= bound as u128 && bytes[count..].iter().all(|&b| b == 0) {
                return Ok(sign * magnitude as i128);
            }
        }
        Err(CudaError::NotImplemented {
            kernel: "CUDA guard grouping requires bounded integer coefficients",
        })
    }

    fn push<F: JoltField>(
        &mut self,
        weights: &[F],
        constant: F,
        guard: bool,
    ) -> Result<(), CudaError> {
        let mut minimum = 0i128;
        let mut maximum = 0i128;
        for (layout, value) in LAYOUT
            .into_iter()
            .zip(weights.iter().copied())
            .chain(std::iter::once((CONSTANT, constant)))
        {
            if value.is_zero() {
                continue;
            }
            let coefficient = Self::integer(value, 1 << 64)?;
            let magnitude = coefficient.unsigned_abs();
            if !magnitude.is_power_of_two()
                || (magnitude > 4 && !(layout == CONSTANT && magnitude == 1 << 64))
                || (guard && layout >> 8 != KIND_FLAG && layout != CONSTANT)
            {
                return Err(CudaError::NotImplemented {
                    kernel: "CUDA guard forms require small powers of two and flag-only guards",
                });
            }
            if guard {
                if layout == CONSTANT {
                    minimum += coefficient;
                    maximum += coefficient;
                } else {
                    minimum += coefficient.min(0);
                    maximum += coefficient.max(0);
                }
            }
            self.terms.push(
                layout | (u32::from(coefficient < 0) << 16) | (magnitude.trailing_zeros() << 24),
            );
        }
        if guard && (minimum < -3 || maximum > 3) {
            return Err(CudaError::NotImplemented {
                kernel: "CUDA guard keys require each guard in [-3, 3]",
            });
        }
        self.offsets.push(self.terms.len() as u32);
        Ok(())
    }

    pub(super) fn upload(
        &self,
        context: &CudaKernelContext,
    ) -> Result<DeviceGuardForms, CudaError> {
        Ok(DeviceGuardForms {
            offsets: context.upload_u32_slice(&self.offsets)?,
            terms: context.upload_u32_slice(&self.terms)?,
            extension: context.upload_u32_slice(&self.extension)?,
        })
    }
}

pub(super) struct DeviceGuardForms {
    offsets: CudaSlice<u32>,
    terms: CudaSlice<u32>,
    extension: CudaSlice<u32>,
}

impl DeviceGuardForms {
    pub(super) fn bind_args<'a>(&'a self, builder: &mut LaunchArgs<'a>) {
        let _ = builder.arg(&self.offsets);
        let _ = builder.arg(&self.terms);
        let _ = builder.arg(&self.extension);
    }
}
