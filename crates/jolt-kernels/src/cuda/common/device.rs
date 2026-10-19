//! Device field storage for BN254 `Fr` in Montgomery form: four consecutive
//! `u64` limbs per element, matching the host field's internal representation.

use ark_bn254::Fr as ArkFr;
use std::sync::Arc;

use ark_ff::BigInt;
use cudarc::driver::{CudaSlice, CudaStream};
use jolt_field::Fr;

use super::error::CudaError;
use super::xfer_stats::{self, Phase};

pub const LIMBS: usize = 4;

#[inline]
fn fr_to_limbs(value: Fr) -> [u64; LIMBS] {
    value.inner_limbs()
}

#[inline]
fn limbs_to_fr(limbs: [u64; LIMBS]) -> Fr {
    Fr::from(ArkFr::new_unchecked(BigInt(limbs)))
}

/// An owned field buffer paired with its CUDA stream. Reads and device copies
/// use the logical element count; the backing allocation may retain spare limbs.
pub struct DeviceFrVec {
    stream: Arc<CudaStream>,
    buffer: CudaSlice<u64>,
    len: usize,
}

impl DeviceFrVec {
    pub(crate) const fn from_parts(
        stream: Arc<CudaStream>,
        buffer: CudaSlice<u64>,
        len: usize,
    ) -> Self {
        Self {
            stream,
            buffer,
            len,
        }
    }

    pub const fn len(&self) -> usize {
        self.len
    }

    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub(crate) const fn limbs(&self) -> &CudaSlice<u64> {
        &self.buffer
    }

    pub(crate) const fn limbs_mut(&mut self) -> &mut CudaSlice<u64> {
        &mut self.buffer
    }

    pub fn to_host(&self) -> Result<Vec<Fr>, CudaError> {
        if self.len == 0 {
            return Ok(Vec::new());
        }
        let limbs = self.len * LIMBS;
        xfer_stats::timed(Phase::D2h, limbs * size_of::<u64>(), || {
            let raw = self.stream.clone_dtoh(&self.buffer.slice(0..limbs))?;
            Ok(raw
                .chunks_exact(LIMBS)
                .map(|limbs| limbs_to_fr([limbs[0], limbs[1], limbs[2], limbs[3]]))
                .collect())
        })
    }

    pub fn first(&self) -> Result<Fr, CudaError> {
        if self.len == 0 {
            return Err(CudaError::LengthMismatch {
                expected: 1,
                got: 0,
            });
        }
        xfer_stats::timed(Phase::D2h, LIMBS * size_of::<u64>(), || {
            let raw = self.stream.clone_dtoh(&self.buffer.slice(0..LIMBS))?;
            Ok(limbs_to_fr([raw[0], raw[1], raw[2], raw[3]]))
        })
    }

    pub fn try_clone(&self) -> Result<Self, CudaError> {
        let buffer = xfer_stats::timed(Phase::D2d, self.len * LIMBS * size_of::<u64>(), || {
            self.stream.clone_dtod(&self.buffer)
        })?;
        Ok(Self {
            stream: self.stream.clone(),
            buffer,
            len: self.len,
        })
    }
}

pub(crate) fn fill_staging(staging: &mut [u64], values: &[Fr]) {
    for (slot, &value) in staging.chunks_exact_mut(LIMBS).zip(values) {
        slot.copy_from_slice(&fr_to_limbs(value));
    }
}
