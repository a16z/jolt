use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream};
use jolt_field::{Fr, Limbs};

use super::error::CudaError;
use super::xfer_stats::{self, Phase};

pub const LIMBS: usize = 4;

#[inline]
fn fr_to_limbs(value: Fr) -> [u64; LIMBS] {
    value.inner_limbs().0
}

#[inline]
pub(crate) fn fr_limbs(value: Fr) -> [u64; LIMBS] {
    fr_to_limbs(value)
}

#[inline]
fn limbs_to_fr(limbs: [u64; LIMBS]) -> Fr {
    Fr::from_bigint_unchecked(Limbs(limbs))
}

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

    pub(crate) fn ordinal(&self) -> usize {
        self.buffer.ordinal()
    }

    pub const fn len(&self) -> usize {
        self.len
    }

    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[cfg(feature = "allocative")]
    pub fn device_bytes(&self) -> usize {
        self.buffer.len() * size_of::<u64>()
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

    pub fn to_limbs(&self) -> Result<Vec<u64>, CudaError> {
        if self.len == 0 {
            return Ok(Vec::new());
        }
        let limbs = self.len * LIMBS;
        xfer_stats::timed(Phase::D2h, limbs * size_of::<u64>(), || {
            Ok(self.stream.clone_dtoh(&self.buffer.slice(0..limbs))?)
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

    pub(crate) fn slice_elements(&self, offset: usize, len: usize) -> Result<Self, CudaError> {
        if offset + len > self.len {
            return Err(CudaError::LengthMismatch {
                expected: self.len,
                got: offset + len,
            });
        }
        let start = offset * LIMBS;
        let buffer = xfer_stats::timed(Phase::D2d, len * LIMBS * size_of::<u64>(), || {
            self.stream
                .clone_dtod(&self.buffer.slice(start..start + len * LIMBS))
        })?;
        Ok(Self {
            stream: self.stream.clone(),
            buffer,
            len,
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

pub(crate) fn require_fr_slice<F: jolt_field::Field>(values: &[F]) -> Result<&[Fr], CudaError> {
    as_fr_slice(values).ok_or(CudaError::NotImplemented {
        kernel: "CUDA kernels support only the BN254 scalar field",
    })
}

pub(crate) fn require_fr<F: jolt_field::Field>(value: F) -> Result<Fr, CudaError> {
    require_fr_slice(std::slice::from_ref(&value))?
        .first()
        .copied()
        .ok_or(CudaError::LengthMismatch {
            expected: 1,
            got: 0,
        })
}

pub fn as_fr_slice<F: jolt_field::Field>(values: &[F]) -> Option<&[Fr]> {
    if std::any::TypeId::of::<F>() != std::any::TypeId::of::<Fr>() {
        return None;
    }
    // SAFETY: the TypeId check above proves `F == Fr`, so the two slices have
    // identical layout, length, and validity. The lifetime is preserved.
    Some(unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<Fr>(), values.len()) })
}

pub fn fr_vec_into<F: jolt_field::Field>(values: Vec<Fr>) -> Option<Vec<F>> {
    if std::any::TypeId::of::<F>() != std::any::TypeId::of::<Fr>() {
        return None;
    }
    let mut values = std::mem::ManuallyDrop::new(values);
    // SAFETY: the TypeId check proves `F == Fr` — identical size and align —
    // so the allocation is valid to reconstitute as `Vec<F>` with the same
    // length and capacity. The original vector is not dropped.
    Some(unsafe {
        Vec::from_raw_parts(
            values.as_mut_ptr().cast::<F>(),
            values.len(),
            values.capacity(),
        )
    })
}

pub fn fr_into<F: jolt_field::Field>(value: Fr) -> Option<F> {
    if std::any::TypeId::of::<F>() != std::any::TypeId::of::<Fr>() {
        return None;
    }
    // SAFETY: the TypeId check proves `F == Fr`, so this reads an `Fr` as the
    // identical type. `Fr: Copy`, so the source needs no forgetting.
    Some(unsafe { std::ptr::read((&raw const value).cast::<F>()) })
}
