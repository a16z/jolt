use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, CudaView, LaunchConfig, PushKernelArg};

use super::context::{CudaKernelContext, BLOCK};
use jolt_field::Fr;

use super::device::DeviceFrVec;
use super::error::CudaError;
#[cfg(test)]
use super::pack::COLD;
use super::xfer_stats::{self, Phase};

pub const FQ_LIMBS: usize = 4;

pub const AFFINE_LIMBS: usize = 2 * FQ_LIMBS;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AffineLimbs {
    pub x: [u64; FQ_LIMBS],
    pub y: [u64; FQ_LIMBS],
    pub infinity: bool,
}

impl AffineLimbs {
    pub const IDENTITY: Self = Self {
        x: [0; FQ_LIMBS],
        y: [0; FQ_LIMBS],
        infinity: true,
    };
}

#[derive(Clone, Copy, Debug)]
pub struct JacobianLimbs {
    pub x: [u64; FQ_LIMBS],
    pub y: [u64; FQ_LIMBS],
    pub z: [u64; FQ_LIMBS],
}

impl JacobianLimbs {
    pub const IDENTITY: Self = Self {
        x: [0; FQ_LIMBS],
        y: [0; FQ_LIMBS],
        z: [0; FQ_LIMBS],
    };

    pub const fn is_identity(&self) -> bool {
        self.z[0] == 0 && self.z[1] == 0 && self.z[2] == 0 && self.z[3] == 0
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ResidentAxpy {
    pub a_offset: usize,
    pub b_offset: usize,
    pub out_offset: usize,
    pub count: usize,
}

pub struct DeviceG1Bases {
    stream: Arc<CudaStream>,
    limbs: CudaSlice<u64>,
    count: usize,
}

impl DeviceG1Bases {
    #[cfg(feature = "allocative")]
    pub fn device_bytes(&self) -> usize {
        self.limbs.len() * size_of::<u64>()
    }

    pub const fn count(&self) -> usize {
        self.count
    }

    pub(crate) const fn limbs(&self) -> &CudaSlice<u64> {
        &self.limbs
    }

    pub(crate) fn ordinal(&self) -> usize {
        self.limbs.ordinal()
    }

    pub fn to_host(&self) -> Result<Vec<AffineLimbs>, CudaError> {
        if self.count == 0 {
            return Ok(Vec::new());
        }
        let bytes = self.count * AFFINE_LIMBS * size_of::<u64>();
        let raw = xfer_stats::timed(Phase::D2h, bytes, || {
            Ok::<_, CudaError>(self.stream.clone_dtoh(&self.limbs)?)
        })?;
        Ok(raw
            .chunks_exact(AFFINE_LIMBS)
            .map(|chunk| {
                let mut x = [0u64; FQ_LIMBS];
                let mut y = [0u64; FQ_LIMBS];
                x.copy_from_slice(&chunk[..FQ_LIMBS]);
                y.copy_from_slice(&chunk[FQ_LIMBS..]);
                let infinity = x == [0; FQ_LIMBS] && y == [0; FQ_LIMBS];
                AffineLimbs { x, y, infinity }
            })
            .collect())
    }
}

const BATCH_INVERSE_TARGET_THREADS: usize = 8192;

const BATCH_INVERSE_MAX_CHUNK: usize = 128;

const POINT_BLOCK: u32 = 128;

const FR_SCALAR_BITS: usize = 254;

const MAX_BASE_INDEX: usize = 0x7fff_ffff;

const MAX_WINDOW_BITS: usize = 16;

const SMALL_SEGMENT_LIMIT: usize = 64;

const TARGET_REDUCE_BLOCKS: usize = 1024;

const G2_POINT_WORDS: usize = 6 * FQ_LIMBS;

const GLV_CONSTANTS: [u64; 14] = [
    0x163b_4843_cb4b_9a5e,
    0x149d_540f_d5e4_95cc,
    0x5398_fd03_00ff_6565,
    0x4cce_f014_a773_d2d2,
    0x0000_0000_0000_0002,
    0x8fa7_d32d_2faf_ba64,
    0x6eb9_c714_773a_6ef2,
    0xd91d_232e_c7e0_b3d7,
    0x0000_0000_0000_0002,
    0x8211_bbeb_7d4f_1128,
    0x6f4d_8248_eeb8_59fc,
    0x89d3_2568_94d2_13e3,
    0x0be4_e154_1221_250b,
    0x6f4d_8248_eeb8_59fd,
];

const GLV_SCALAR_BITS: usize = 128;

const GLV4_SCALAR_BITS: usize = 101;

fn glv_4d_table() -> (&'static [u64], &'static [u8]) {
    static TABLE: std::sync::OnceLock<(Vec<u64>, Vec<u8>)> = std::sync::OnceLock::new();
    let (magnitudes, signs) = TABLE.get_or_init(|| {
        let rows = jolt_crypto::ec::bn254::glv::constants::POWER_OF_2_DECOMPOSITIONS;
        let mut magnitudes = Vec::with_capacity(rows.len() * 8);
        let mut signs = Vec::with_capacity(rows.len() * 4);
        for (k0, k1, k2, k3, neg0, neg1, neg2, neg3) in rows {
            for component in [k0, k1, k2, k3] {
                magnitudes.push(component as u64);
                magnitudes.push((component >> 64) as u64);
            }
            for negative in [neg0, neg1, neg2, neg3] {
                signs.push(u8::from(negative));
            }
        }
        (magnitudes, signs)
    });
    (magnitudes, signs)
}

fn glv_4d_scalar(scalar: Fr) -> Result<([u64; 16], u32, usize), CudaError> {
    let ark = ark_bn254::Fr::from(scalar);
    let (coefficients, positive) = jolt_crypto::ec::bn254::glv::decomp_4d::decompose_scalar_4d(ark);
    let mut limbs = [0u64; 16];
    let mut signs = 0u32;
    let mut max_bits = 0usize;
    for (index, component) in coefficients.iter().enumerate() {
        let slot = limbs
            .get_mut(index * FQ_LIMBS..(index + 1) * FQ_LIMBS)
            .ok_or(CudaError::InvariantViolation {
                reason: "a 4D GLV component landed outside the coefficient buffer",
            })?;
        slot.copy_from_slice(&component.0);
        max_bits = max_bits.max(ark_ff::BigInteger::num_bits(component) as usize);
        if !positive.get(index).copied().unwrap_or(true) {
            signs |= 1u32 << index;
        }
    }
    Ok((limbs, signs, max_bits))
}

fn glv_frobenius_coefficients() -> &'static [u64] {
    static COEFFICIENTS: std::sync::OnceLock<Vec<u64>> = std::sync::OnceLock::new();
    COEFFICIENTS.get_or_init(|| {
        let coefficients = jolt_crypto::ec::bn254::glv::constants::get_frobenius_coefficients();
        let mut limbs = Vec::with_capacity(48);
        for value in [
            coefficients.psi1_coef2,
            coefficients.psi1_coef3,
            coefficients.psi2_coef2,
            coefficients.psi2_coef3,
            coefficients.psi3_coef2,
            coefficients.psi3_coef3,
        ] {
            limbs.extend_from_slice(&value.c0.0 .0);
            limbs.extend_from_slice(&value.c1.0 .0);
        }
        limbs
    })
}

fn glv_endomorphism_coefficient() -> [u64; FQ_LIMBS] {
    <ark_bn254::g1::Config as ark_ec::scalar_mul::glv::GLVConfig>::ENDO_COEFFS[0]
        .0
         .0
}

const G2_POINT_BLOCK: u32 = 64;

const BATCHED_WINDOW_BYTES: usize = 64 << 20;

struct DeviceScalars<'a> {
    values: &'a CudaSlice<u64>,
    limbs: usize,
}

struct IndexPlan<'a> {
    scalars: &'a CudaSlice<u64>,
    scalar_limbs: usize,
    signs: &'a CudaSlice<u8>,
    row_len: usize,
    buckets: usize,
    segments: usize,
    lane: usize,
    shifts: &'a [usize],
}

struct BucketIndex {
    indices: CudaSlice<u32>,
    offsets: CudaSlice<u32>,
    counts: CudaSlice<u32>,
    widest: usize,
}

struct PassPlan<'a> {
    bases: &'a DeviceG1Bases,
    scalars: &'a CudaSlice<u64>,
    scalar_limbs: usize,
    signs: &'a CudaSlice<u8>,
    row_len: usize,
    buckets: usize,
    rows: usize,
    lane: usize,
    shifts: &'a [usize],
}

struct SegmentPlan<'a> {
    indices: &'a CudaSlice<u32>,
    offsets: &'a CudaSlice<u32>,
    counts: &'a CudaSlice<u32>,
    segments: usize,
    widest: usize,
}

fn window_bits(row_len: usize) -> usize {
    if row_len < 32 {
        3
    } else {
        ((row_len as f64).ln() as usize + 2).min(MAX_WINDOW_BITS)
    }
}

fn flatten_affine(points: &[AffineLimbs]) -> Vec<u64> {
    let mut flat = Vec::with_capacity(points.len() * AFFINE_LIMBS);
    for point in points {
        if point.infinity {
            flat.extend_from_slice(&[0u64; AFFINE_LIMBS]);
        } else {
            flat.extend_from_slice(&point.x);
            flat.extend_from_slice(&point.y);
        }
    }
    flat
}

fn flatten_jacobian(points: &[JacobianLimbs]) -> Vec<u64> {
    let mut flat = Vec::with_capacity(points.len() * 3 * FQ_LIMBS);
    for point in points {
        flat.extend_from_slice(&point.x);
        flat.extend_from_slice(&point.y);
        flat.extend_from_slice(&point.z);
    }
    flat
}

fn take_limbs(chunk: &[u64]) -> [u64; FQ_LIMBS] {
    let mut limbs = [0u64; FQ_LIMBS];
    limbs.copy_from_slice(chunk);
    limbs
}

#[cfg(feature = "parallel")]
fn split_signed_magnitudes(rows: &[i128], magnitudes: &mut [u64], signs: &mut [u8]) -> bool {
    use rayon::prelude::*;

    magnitudes
        .par_iter_mut()
        .zip(signs.par_iter_mut())
        .zip(rows.par_iter())
        .map(|((magnitude, sign), &scalar)| {
            let value = scalar.unsigned_abs();
            *magnitude = u64::try_from(value).unwrap_or_default();
            *sign = u8::from(scalar.is_negative());
            value <= u128::from(u64::MAX)
        })
        .reduce(|| true, |a, b| a && b)
}

#[cfg(not(feature = "parallel"))]
fn split_signed_magnitudes(rows: &[i128], magnitudes: &mut [u64], signs: &mut [u8]) -> bool {
    let mut fits = true;
    for ((magnitude, sign), &scalar) in magnitudes.iter_mut().zip(signs.iter_mut()).zip(rows) {
        let value = scalar.unsigned_abs();
        *magnitude = u64::try_from(value).unwrap_or_default();
        *sign = u8::from(scalar.is_negative());
        fits &= value <= u128::from(u64::MAX);
    }
    fits
}

fn unflatten_jacobian(flat: &[u64]) -> Vec<JacobianLimbs> {
    flat.chunks_exact(3 * FQ_LIMBS)
        .map(|point| JacobianLimbs {
            x: take_limbs(&point[..FQ_LIMBS]),
            y: take_limbs(&point[FQ_LIMBS..2 * FQ_LIMBS]),
            z: take_limbs(&point[2 * FQ_LIMBS..]),
        })
        .collect()
}

fn unflatten_affine(flat: &[u64]) -> Vec<AffineLimbs> {
    flat.chunks_exact(AFFINE_LIMBS)
        .map(|point| {
            let x = take_limbs(&point[..FQ_LIMBS]);
            let y = take_limbs(&point[FQ_LIMBS..]);
            let infinity = x == [0; FQ_LIMBS] && y == [0; FQ_LIMBS];
            AffineLimbs { x, y, infinity }
        })
        .collect()
}

impl CudaKernelContext {
    pub fn upload_g1_bases(&self, bases: &[AffineLimbs]) -> Result<DeviceG1Bases, CudaError> {
        let flat = flatten_affine(bases);
        let limbs = if flat.is_empty() {
            self.alloc_u64(0)?
        } else {
            self.upload_u64_slice(&flat)?
        };
        Ok(DeviceG1Bases {
            stream: self.stream().clone(),
            limbs,
            count: bases.len(),
        })
    }

    fn fq_binary(
        &self,
        kernel: &cudarc::driver::CudaFunction,
        left: &[[u64; FQ_LIMBS]],
        right: &[[u64; FQ_LIMBS]],
    ) -> Result<Vec<[u64; FQ_LIMBS]>, CudaError> {
        if left.len() != right.len() {
            return Err(CudaError::LengthMismatch {
                expected: left.len(),
                got: right.len(),
            });
        }
        if left.is_empty() {
            return Ok(Vec::new());
        }
        let count = Self::count_of(left.len())?;
        let device_left = self.upload_u64_slice(left.as_flattened())?;
        let device_right = self.upload_u64_slice(right.as_flattened())?;
        let mut output = self.alloc_u64(left.len() * FQ_LIMBS)?;
        let mut builder = self.stream().launch_builder(kernel);
        let _ = builder.arg(&device_left);
        let _ = builder.arg(&device_right);
        let _ = builder.arg(&mut output);
        let _ = builder.arg(&count);
        // SAFETY: thread `i < count` reads only `left[i]`/`right[i]` and writes
        // only `out[i]` (4 limbs each); all three buffers hold
        // `count * FQ_LIMBS` u64s, the length equality is checked above, and
        // `out` is a fresh allocation distinct from both inputs. Threads with
        // `i >= count` return before any access.
        let _ = unsafe { builder.launch(Self::launch_config(count)) }?;
        self.stream().synchronize()?;
        Ok(self
            .download_u64(&output)?
            .chunks_exact(FQ_LIMBS)
            .map(take_limbs)
            .collect())
    }

    fn g1_binary(
        &self,
        kernel: &cudarc::driver::CudaFunction,
        left: &[JacobianLimbs],
        right_flat: &[u64],
        right_stride: usize,
    ) -> Result<Vec<JacobianLimbs>, CudaError> {
        if left.len() * right_stride != right_flat.len() {
            return Err(CudaError::LengthMismatch {
                expected: left.len() * right_stride,
                got: right_flat.len(),
            });
        }
        if left.is_empty() {
            return Ok(Vec::new());
        }
        let count = Self::count_of(left.len())?;
        let device_left = self.upload_u64_slice(&flatten_jacobian(left))?;
        let device_right = self.upload_u64_slice(right_flat)?;
        let mut output = self.alloc_u64(left.len() * 3 * FQ_LIMBS)?;
        let mut builder = self.stream().launch_builder(kernel);
        let _ = builder.arg(&device_left);
        let _ = builder.arg(&device_right);
        let _ = builder.arg(&mut output);
        let _ = builder.arg(&count);
        // SAFETY: thread `i < count` reads only its own point at `left[i]`
        // (12 limbs) and `right[i]` (`right_stride` limbs), and writes only
        // `out[i]` (12 limbs). The buffers hold `count * 12`,
        // `count * right_stride` and `count * 12` u64s respectively — the
        // right-hand length is checked above — and `out` is a fresh allocation
        // distinct from both inputs. Threads with `i >= count` return first.
        let _ = unsafe { builder.launch(Self::launch_config(count)) }?;
        self.stream().synchronize()?;
        Ok(unflatten_jacobian(&self.download_u64(&output)?))
    }
}

impl CudaKernelContext {
    pub fn fq_add(
        &self,
        left: &[[u64; FQ_LIMBS]],
        right: &[[u64; FQ_LIMBS]],
    ) -> Result<Vec<[u64; FQ_LIMBS]>, CudaError> {
        self.fq_binary(self.msm_fq_add(), left, right)
    }

    pub fn fq_sub(
        &self,
        left: &[[u64; FQ_LIMBS]],
        right: &[[u64; FQ_LIMBS]],
    ) -> Result<Vec<[u64; FQ_LIMBS]>, CudaError> {
        self.fq_binary(self.msm_fq_sub(), left, right)
    }

    pub fn fq_mul(
        &self,
        left: &[[u64; FQ_LIMBS]],
        right: &[[u64; FQ_LIMBS]],
    ) -> Result<Vec<[u64; FQ_LIMBS]>, CudaError> {
        self.fq_binary(self.msm_fq_mul(), left, right)
    }

    pub fn fq_batch_inverse(
        &self,
        values: &[[u64; FQ_LIMBS]],
    ) -> Result<Vec<[u64; FQ_LIMBS]>, CudaError> {
        if values.is_empty() {
            return Ok(Vec::new());
        }
        let device = self.upload_u64_slice(values.as_flattened())?;
        let mut output = self.alloc_u64(values.len() * FQ_LIMBS)?;
        self.launch_batch_inverse(&device, &mut output, values.len())?;
        Ok(self
            .download_u64(&output)?
            .chunks_exact(FQ_LIMBS)
            .map(take_limbs)
            .collect())
    }

    pub(crate) fn launch_batch_inverse(
        &self,
        values: &CudaSlice<u64>,
        output: &mut CudaSlice<u64>,
        len: usize,
    ) -> Result<(), CudaError> {
        if len == 0 {
            return Ok(());
        }
        let chunk = len
            .div_ceil(BATCH_INVERSE_TARGET_THREADS)
            .clamp(1, BATCH_INVERSE_MAX_CHUNK);
        let threads = Self::count_of(len.div_ceil(chunk))?;
        let count = Self::count_of(len)?;
        let chunk = Self::count_of(chunk)?;
        let mut builder = self.stream().launch_builder(self.msm_fq_batch_inverse());
        let _ = builder.arg(values);
        let _ = builder.arg(output);
        let _ = builder.arg(&count);
        let _ = builder.arg(&chunk);
        // SAFETY: thread `t` owns the half-open element range
        // `[t * chunk, min((t+1) * chunk, count))`, disjoint across threads and
        // clipped to `count`. It reads only `values` and read-modify-writes
        // only `out` inside that range (4 limbs per element; both buffers hold
        // `count * FQ_LIMBS` u64s), using `out` as its own prefix-product
        // scratch. `out` is a distinct allocation from `values`. Threads whose
        // range starts at or past `count` return before any access.
        let _ = unsafe { builder.launch(Self::launch_config(threads)) }?;
        self.stream().synchronize()?;
        Ok(())
    }

    pub fn g1_double(&self, points: &[JacobianLimbs]) -> Result<Vec<JacobianLimbs>, CudaError> {
        if points.is_empty() {
            return Ok(Vec::new());
        }
        let count = Self::count_of(points.len())?;
        let device = self.upload_u64_slice(&flatten_jacobian(points))?;
        let mut output = self.alloc_u64(points.len() * 3 * FQ_LIMBS)?;
        let mut builder = self.stream().launch_builder(self.msm_g1_double());
        let _ = builder.arg(&device);
        let _ = builder.arg(&mut output);
        let _ = builder.arg(&count);
        // SAFETY: thread `i < count` reads only `points[i]` and writes only
        // `out[i]` (12 limbs each); both buffers hold `count * 12` u64s and
        // `out` is a fresh allocation distinct from the input. Threads with
        // `i >= count` return before any access.
        let _ = unsafe { builder.launch(Self::launch_config(count)) }?;
        self.stream().synchronize()?;
        Ok(unflatten_jacobian(&self.download_u64(&output)?))
    }

    pub fn g1_add(
        &self,
        left: &[JacobianLimbs],
        right: &[JacobianLimbs],
    ) -> Result<Vec<JacobianLimbs>, CudaError> {
        self.g1_binary(
            self.msm_g1_add(),
            left,
            &flatten_jacobian(right),
            3 * FQ_LIMBS,
        )
    }

    pub fn g1_add_affine(
        &self,
        left: &[JacobianLimbs],
        right: &[AffineLimbs],
    ) -> Result<Vec<JacobianLimbs>, CudaError> {
        self.g1_binary(
            self.msm_g1_add_affine(),
            left,
            &flatten_affine(right),
            AFFINE_LIMBS,
        )
    }

    pub fn g1_batch_add_affine_pairs(
        &self,
        left: &[AffineLimbs],
        right: &[AffineLimbs],
    ) -> Result<Vec<AffineLimbs>, CudaError> {
        if left.len() != right.len() {
            return Err(CudaError::LengthMismatch {
                expected: left.len(),
                got: right.len(),
            });
        }
        if left.is_empty() {
            return Ok(Vec::new());
        }
        let count = Self::count_of(left.len())?;
        let device_left = self.upload_u64_slice(&flatten_affine(left))?;
        let device_right = self.upload_u64_slice(&flatten_affine(right))?;
        let mut denominators = self.alloc_u64(left.len() * FQ_LIMBS)?;
        let mut builder = self.stream().launch_builder(self.msm_affine_denominators());
        let _ = builder.arg(&device_left);
        let _ = builder.arg(&device_right);
        let _ = builder.arg(&mut denominators);
        let _ = builder.arg(&count);
        // SAFETY: thread `i < count` reads the x-coordinate of `left[i]` and
        // `right[i]` (limb group `2i` of buffers holding
        // `count * AFFINE_LIMBS` u64s) and writes only `denominators[i]` of a
        // `count * FQ_LIMBS` fresh allocation distinct from both inputs.
        // Threads with `i >= count` return before any access.
        let _ = unsafe { builder.launch(Self::launch_config(count)) }?;
        self.stream().synchronize()?;

        let mut inverses = self.alloc_u64(left.len() * FQ_LIMBS)?;
        self.launch_batch_inverse(&denominators, &mut inverses, left.len())?;

        let mut output = self.alloc_u64(left.len() * AFFINE_LIMBS)?;
        let mut builder = self.stream().launch_builder(self.msm_affine_combine());
        let _ = builder.arg(&device_left);
        let _ = builder.arg(&device_right);
        let _ = builder.arg(&inverses);
        let _ = builder.arg(&mut output);
        let _ = builder.arg(&count);
        // SAFETY: thread `i < count` reads `left[i]`, `right[i]`
        // (`AFFINE_LIMBS` u64s each) and `inverses[i]` (`FQ_LIMBS` u64s), and
        // writes only `out[i]` of a `count * AFFINE_LIMBS` fresh allocation
        // distinct from all three inputs. Threads with `i >= count` return
        // before any access.
        let _ = unsafe { builder.launch(Self::launch_config(count)) }?;
        self.stream().synchronize()?;
        Ok(unflatten_affine(&self.download_u64(&output)?))
    }
}

impl CudaKernelContext {
    pub fn msm_rows_fr(
        &self,
        bases: &DeviceG1Bases,
        rows: &DeviceFrVec,
        row_len: usize,
    ) -> Result<Vec<JacobianLimbs>, CudaError> {
        let canonical = self.canonical_scalars(rows)?;
        let signs = vec![0u8; rows.len()];
        let accumulator = self.pippenger_device(
            bases,
            DeviceScalars {
                values: &canonical,
                limbs: FQ_LIMBS,
            },
            &signs,
            rows.len(),
            row_len,
            FR_SCALAR_BITS,
        )?;
        Ok(unflatten_jacobian(&self.download_u64(&accumulator)?))
    }

    pub fn upload_raw_u64(&self, values: &[u64]) -> Result<CudaSlice<u64>, CudaError> {
        self.upload_u64_slice(values)
    }

    pub fn write_u64_range(
        &self,
        buffer: &mut CudaSlice<u64>,
        start: usize,
        values: &[u64],
    ) -> Result<(), CudaError> {
        if start + values.len() > buffer.len() {
            return Err(CudaError::LengthMismatch {
                expected: buffer.len(),
                got: start + values.len(),
            });
        }
        if values.is_empty() {
            return Ok(());
        }
        let mut target = buffer.slice_mut(start..start + values.len());
        xfer_stats::timed(Phase::H2d, size_of_val(values), || {
            Ok::<_, CudaError>(self.stream().memcpy_htod(values, &mut target)?)
        })
    }

    pub fn read_u64_range(
        &self,
        buffer: &CudaSlice<u64>,
        start: usize,
        end: usize,
    ) -> Result<Vec<u64>, CudaError> {
        if end > buffer.len() || start > end {
            return Err(CudaError::LengthMismatch {
                expected: buffer.len(),
                got: end,
            });
        }
        if start == end {
            return Ok(Vec::new());
        }
        let source = buffer.slice(start..end);
        xfer_stats::timed(Phase::D2h, (end - start) * size_of::<u64>(), || {
            Ok::<_, CudaError>(self.stream().clone_dtoh(&source)?)
        })
    }

    fn resident_span_end(
        offset: usize,
        count: usize,
        words: usize,
        buffer: &CudaSlice<u64>,
    ) -> Result<(), CudaError> {
        let end = offset
            .checked_add(count)
            .and_then(|last| last.checked_mul(words))
            .ok_or(CudaError::InvariantViolation {
                reason: "a resident point span overflowed",
            })?;
        if end > buffer.len() {
            return Err(CudaError::LengthMismatch {
                expected: buffer.len(),
                got: end,
            });
        }
        Ok(())
    }

    fn axpy_in_place(
        &self,
        kernel: &cudarc::driver::CudaFunction,
        words: usize,
        buffer: &mut CudaSlice<u64>,
        span: ResidentAxpy,
        scalar: Fr,
    ) -> Result<(), CudaError> {
        let ResidentAxpy {
            a_offset,
            b_offset,
            out_offset,
            count,
        } = span;
        if count == 0 {
            return Ok(());
        }
        for offset in [a_offset, b_offset, out_offset] {
            Self::resident_span_end(offset, count, words, buffer)?;
        }
        let coincides_or_avoids = |left: usize, right: usize| {
            left == right || left + count <= right || right + count <= left
        };
        if !coincides_or_avoids(out_offset, a_offset) || !coincides_or_avoids(out_offset, b_offset)
        {
            return Err(CudaError::InvariantViolation {
                reason: "a resident axpy output must coincide with an operand or avoid both",
            });
        }

        let device_scalar = self.canonical_scalars(&self.upload(&[scalar])?)?;
        let a = Self::count_of(a_offset)?;
        let b = Self::count_of(b_offset)?;
        let out = Self::count_of(out_offset)?;
        let points = Self::count_of(count)?;
        let bits = Self::count_of(FR_SCALAR_BITS)?;

        let mut builder = self.stream().launch_builder(kernel);
        let _ = builder.arg(buffer);
        let _ = builder.arg(&device_scalar);
        let _ = builder.arg(&a);
        let _ = builder.arg(&b);
        let _ = builder.arg(&out);
        let _ = builder.arg(&points);
        let _ = builder.arg(&bits);
        // SAFETY: thread `i < points` reads the `words` limbs at points
        // `a_offset + i` and `b_offset + i` of `buffer` and writes the same span
        // at point `out_offset + i`; all three spans are checked above to end
        // inside `buffer`. The three views alias one allocation, so `buf` is
        // passed once and un-`__restrict__`ed: the write span either coincides
        // exactly with an operand — in which case thread `i` has already loaded
        // both operand points into registers before it stores — or is disjoint
        // from both, which the alias check enforces. Either way no thread reads
        // a point another thread writes. The scalar is the single canonical
        // value and bit indices stay below `FR_SCALAR_BITS = 254`, so
        // `bit >> 6 <= 3` is inside its `FQ_LIMBS` limbs. Threads with
        // `i >= points` return first.
        let _ = unsafe { builder.launch(Self::launch_config(points)) }?;
        Ok(())
    }

    pub fn g1_axpy_in_place(
        &self,
        buffer: &mut CudaSlice<u64>,
        span: ResidentAxpy,
        scalar: Fr,
    ) -> Result<(), CudaError> {
        self.axpy_in_place(self.msm_g1_axpy(), 3 * FQ_LIMBS, buffer, span, scalar)
    }

    pub fn g2_axpy_in_place(
        &self,
        buffer: &mut CudaSlice<u64>,
        span: ResidentAxpy,
        scalar: Fr,
    ) -> Result<(), CudaError> {
        let ResidentAxpy {
            a_offset,
            b_offset,
            out_offset,
            count,
        } = span;
        if count == 0 {
            return Ok(());
        }
        let words = 6 * FQ_LIMBS;
        for offset in [a_offset, b_offset, out_offset] {
            Self::resident_span_end(offset, count, words, buffer)?;
        }
        let coincides_or_avoids = |left: usize, right: usize| {
            left == right || left + count <= right || right + count <= left
        };
        if !coincides_or_avoids(out_offset, a_offset) || !coincides_or_avoids(out_offset, b_offset)
        {
            return Err(CudaError::InvariantViolation {
                reason: "a resident axpy output must coincide with an operand or avoid both",
            });
        }

        let (limbs, signs, max_bits) = glv_4d_scalar(scalar)?;
        let coeffs = self.upload_u64_slice(&limbs)?;
        let frobenius = self.upload_u64_slice(glv_frobenius_coefficients())?;
        let a = Self::count_of(a_offset)?;
        let b = Self::count_of(b_offset)?;
        let out = Self::count_of(out_offset)?;
        let points = Self::count_of(count)?;
        let bits = Self::count_of(max_bits)?;

        let mut builder = self.stream().launch_builder(self.msm_g2_axpy_glv());
        let _ = builder.arg(buffer);
        let _ = builder.arg(&coeffs);
        let _ = builder.arg(&frobenius);
        let _ = builder.arg(&signs);
        let _ = builder.arg(&bits);
        let _ = builder.arg(&a);
        let _ = builder.arg(&b);
        let _ = builder.arg(&out);
        let _ = builder.arg(&points);
        // SAFETY: thread `i < points` reads the 24 limbs at points `a_offset + i`
        // and `b_offset + i` of `buffer` and writes the 24 limbs at
        // `out_offset + i`; all three spans are bounds-checked above, and the
        // output either coincides with an operand elementwise or is disjoint from
        // both, so no thread writes a slot another thread reads. It also reads the
        // 16 coefficient limbs and 48 Frobenius limbs, whose lengths are fixed by
        // `glv_4d_scalar` and `glv_frobenius_coefficients`. `max_bits` bounds the
        // ladder to bits present in the coefficients, all four of which hold four
        // limbs. Threads with `i >= points` return before any access.
        let _ = unsafe { builder.launch(Self::launch_config(points)) }?;
        Ok(())
    }

    pub fn g2_fixed_base_in_place(
        &self,
        buffer: &mut CudaSlice<u64>,
        base_offset: usize,
        out_offset: usize,
        scalars: &[Fr],
    ) -> Result<(), CudaError> {
        let count = scalars.len();
        if count == 0 {
            return Ok(());
        }
        let words = 6 * FQ_LIMBS;
        Self::resident_span_end(base_offset, 1, words, buffer)?;
        Self::resident_span_end(out_offset, count, words, buffer)?;
        if base_offset >= out_offset && base_offset < out_offset + count {
            return Err(CudaError::InvariantViolation {
                reason: "a resident fixed-base scaling must not write over its base",
            });
        }

        let device_scalars = self.canonical_scalars(&self.upload(scalars)?)?;
        let base = Self::count_of(base_offset)?;
        let out = Self::count_of(out_offset)?;
        let points = Self::count_of(count)?;
        let bits = Self::count_of(FR_SCALAR_BITS)?;

        let mut builder = self.stream().launch_builder(self.msm_g2_fixed_base());
        let _ = builder.arg(buffer);
        let _ = builder.arg(&device_scalars);
        let _ = builder.arg(&base);
        let _ = builder.arg(&out);
        let _ = builder.arg(&points);
        let _ = builder.arg(&bits);
        // SAFETY: thread `i < points` reads the `6 * FQ_LIMBS` limbs of the
        // single base point at `base_offset` and the `FQ_LIMBS` limbs of
        // `scalars[i]`, both checked above to lie inside their buffers, and
        // writes only point `out_offset + i`, whose span is checked and proven
        // disjoint from the base. `buf` is passed once, un-`__restrict__`ed,
        // because base and output live in one allocation. Bit indices stay
        // below `FR_SCALAR_BITS = 254`, so `bit >> 6 <= 3` is inside a scalar.
        // Threads with `i >= points` return first.
        let _ = unsafe { builder.launch(Self::launch_config(points)) }?;
        Ok(())
    }

    pub fn msm_rows_shared_scalars(
        &self,
        bases: &[JacobianLimbs],
        scalars: &[Fr],
        rows: usize,
    ) -> Result<Vec<JacobianLimbs>, CudaError> {
        if rows == 0 || scalars.is_empty() {
            return Err(CudaError::InvariantViolation {
                reason: "a shared-scalar row MSM needs at least one row and one term",
            });
        }
        if bases.len() != rows * scalars.len() {
            return Err(CudaError::LengthMismatch {
                expected: rows * scalars.len(),
                got: bases.len(),
            });
        }

        let terms = scalars.len();
        let flat = flatten_jacobian(bases);
        let device_bases = self.upload_u64_slice(&flat)?;
        let device_scalars = self.canonical_scalars(&self.upload(scalars)?)?;
        let mut output = self.alloc_u64(rows * 3 * FQ_LIMBS)?;

        let (split, signs) = self.glv_decompose_2d(&device_scalars, terms)?;
        let device_signs = self.upload_u8_slice(&signs)?;
        let beta = self.upload_u64_slice(&glv_endomorphism_coefficient())?;

        let rows_arg = Self::count_of(rows)?;
        let terms_arg = Self::count_of(terms)?;
        let bits_arg = Self::count_of(GLV_SCALAR_BITS)?;
        let block = terms.next_power_of_two().min(BLOCK as usize) as u32;
        let shared = block * 3 * FQ_LIMBS as u32 * size_of::<u64>() as u32;

        let mut builder = self
            .stream()
            .launch_builder(self.msm_shared_scalar_rows_glv());
        let _ = builder.arg(&device_bases);
        let _ = builder.arg(&split);
        let _ = builder.arg(&device_signs);
        let _ = builder.arg(&beta);
        let _ = builder.arg(&rows_arg);
        let _ = builder.arg(&terms_arg);
        let _ = builder.arg(&bits_arg);
        let _ = builder.arg(&mut output);
        // SAFETY: block `row = blockIdx.x < rows` (a block-uniform guard, so a
        // skipped block never reaches a `__syncthreads()`) reads, for each
        // `term` striding from `threadIdx.x` by `blockDim.x` below `terms`, the
        // `3 * FQ_LIMBS` limbs at `(term * rows + row) * 3 * FQ_LIMBS` inside
        // `bases`'s checked `rows * terms` points, and the `FQ_LIMBS` limbs at
        // `term * FQ_LIMBS` inside `scalars`'s `terms` canonical scalars. Bit
        // indices run below `FR_SCALAR_BITS = 254`, so `bit >> 6 <= 3` stays
        // inside a scalar. Shared memory is `blockDim.x * 3 * FQ_LIMBS` u64s,
        // matching `shared_mem_bytes`, and `blockDim.x` is a power of two so the
        // reduction tree covers the block; every thread reaches each
        // `__syncthreads()` because the strided loop and the tree sit outside
        // any early return. Only thread 0 writes, to the `3 * FQ_LIMBS` limbs at
        // `row * 3 * FQ_LIMBS` of the freshly allocated output — one slot per
        // block, distinct from both inputs.
        let _ = unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (rows_arg, 1, 1),
                block_dim: (block, 1, 1),
                shared_mem_bytes: shared,
            })
        }?;
        self.stream().synchronize()?;

        Ok(unflatten_jacobian(&self.download_u64(&output)?))
    }

    pub fn msm_rows_i128(
        &self,
        bases: &DeviceG1Bases,
        rows: &[i128],
        row_len: usize,
    ) -> Result<Vec<JacobianLimbs>, CudaError> {
        let (device, signs) = tracing::info_span!("cuda_msm_i128_stage", len = rows.len())
            .in_scope(|| {
                let mut magnitudes = vec![0u64; rows.len()];
                let mut signs = vec![0u8; rows.len()];
                if !split_signed_magnitudes(rows, &mut magnitudes, &mut signs) {
                    return Err(CudaError::InvariantViolation {
                        reason: "signed MSM scalars must fit in [-u64::MAX, u64::MAX]",
                    });
                }
                self.require_owned(bases.ordinal())?;
                Ok((self.upload_u64_slice(&magnitudes)?, signs))
            })?;
        let accumulator = self.pippenger_device(
            bases,
            DeviceScalars {
                values: &device,
                limbs: 1,
            },
            &signs,
            rows.len(),
            row_len,
            64,
        )?;
        tracing::info_span!("cuda_msm_i128_download", len = rows.len())
            .in_scope(|| Ok(unflatten_jacobian(&self.download_u64(&accumulator)?)))
    }

    #[cfg(test)]
    pub fn one_hot_chunk_sums(
        &self,
        bases: &DeviceG1Bases,
        hot: &[Option<usize>],
        one_hot_k: usize,
    ) -> Result<Vec<JacobianLimbs>, CudaError> {
        if one_hot_k == 0 {
            return Err(CudaError::InvariantViolation {
                reason: "one-hot chunk sums require a nonzero address count",
            });
        }
        let mut counts = vec![0u32; one_hot_k];
        for &row in hot {
            let Some(row) = row else { continue };
            if row >= one_hot_k {
                return Err(CudaError::InvariantViolation {
                    reason: "a one-hot address lies outside the declared address count",
                });
            }
            counts[row] += 1;
        }
        let mut offsets = Vec::with_capacity(one_hot_k);
        let mut running = 0u32;
        for &count in &counts {
            offsets.push(running);
            running += count;
        }
        let mut cursor = offsets.clone();
        let mut indices = vec![0u32; running as usize];
        for (column, row) in hot.iter().enumerate() {
            let Some(row) = *row else { continue };
            if column > MAX_BASE_INDEX {
                return Err(CudaError::InvariantViolation {
                    reason: "a one-hot column index exceeds the signed-index encoding",
                });
            }
            indices[cursor[row] as usize] = column as u32;
            cursor[row] += 1;
        }
        if bases.count() < hot.len() {
            return Err(CudaError::LengthMismatch {
                expected: hot.len(),
                got: bases.count(),
            });
        }
        self.segment_sums(bases, &indices, &offsets, &counts, one_hot_k)
    }

    #[cfg(test)]
    pub fn one_hot_rows(
        &self,
        bases: &DeviceG1Bases,
        hot: &[u32],
        one_hot_k: usize,
        chunk_len: usize,
    ) -> Result<Vec<JacobianLimbs>, CudaError> {
        if one_hot_k == 0 || chunk_len == 0 || !hot.len().is_multiple_of(chunk_len) {
            return Err(CudaError::LengthMismatch {
                expected: chunk_len,
                got: hot.len(),
            });
        }
        if bases.count() < chunk_len {
            return Err(CudaError::LengthMismatch {
                expected: chunk_len,
                got: bases.count(),
            });
        }
        let chunk_count = hot.len() / chunk_len;
        let segments = one_hot_k * chunk_count;
        let (indices, offsets, counts) =
            tracing::info_span!("cuda_commit_one_hot_csr", segments, cycles = hot.len()).in_scope(
                || {
                    let mut counts = vec![0u32; segments];
                    for (column, &row) in hot.iter().enumerate() {
                        if row == COLD {
                            continue;
                        }
                        let row = row as usize;
                        if row >= one_hot_k {
                            return Err(CudaError::InvariantViolation {
                                reason: "a one-hot address lies outside the declared address count",
                            });
                        }
                        counts[row * chunk_count + column / chunk_len] += 1;
                    }
                    let mut offsets = Vec::with_capacity(segments);
                    let mut running = 0u32;
                    for &count in &counts {
                        offsets.push(running);
                        running += count;
                    }
                    let mut cursor = offsets.clone();
                    let mut indices = vec![0u32; running as usize];
                    for (column, &row) in hot.iter().enumerate() {
                        if row == COLD {
                            continue;
                        }
                        let segment = row as usize * chunk_count + column / chunk_len;
                        indices[cursor[segment] as usize] = (column % chunk_len) as u32;
                        cursor[segment] += 1;
                    }
                    Ok((indices, offsets, counts))
                },
            )?;
        tracing::info_span!("cuda_commit_one_hot_segments", segments)
            .in_scope(|| self.segment_sums(bases, &indices, &offsets, &counts, segments))
    }

    pub fn one_hot_rows_device(
        &self,
        bases: &DeviceG1Bases,
        hot: &CudaView<'_, u32>,
        cycles: usize,
        one_hot_k: usize,
        chunk_len: usize,
    ) -> Result<Vec<JacobianLimbs>, CudaError> {
        if one_hot_k == 0 || chunk_len == 0 || !cycles.is_multiple_of(chunk_len) {
            return Err(CudaError::LengthMismatch {
                expected: chunk_len,
                got: cycles,
            });
        }
        if bases.count() < chunk_len {
            return Err(CudaError::LengthMismatch {
                expected: chunk_len,
                got: bases.count(),
            });
        }
        if hot.len() < cycles {
            return Err(CudaError::LengthMismatch {
                expected: cycles,
                got: hot.len(),
            });
        }
        self.require_owned(bases.ordinal())?;
        let chunk_count = cycles / chunk_len;
        let segments = one_hot_k * chunk_count;
        let cycle_count = Self::count_of(cycles)?;
        let chunk_len_arg = Self::count_of(chunk_len)?;
        let chunk_count_arg = Self::count_of(chunk_count)?;
        let one_hot_k_arg = Self::count_of(one_hot_k)?;

        let count_span =
            tracing::info_span!("cuda_commit_one_hot_count", segments, cycles).entered();
        let mut counts = self.alloc_u32(segments + 1)?;
        let mut builder = self.stream().launch_builder(self.msm_one_hot_count());
        let _ = builder.arg(hot);
        let _ = builder.arg(&cycle_count);
        let _ = builder.arg(&chunk_len_arg);
        let _ = builder.arg(&chunk_count_arg);
        let _ = builder.arg(&one_hot_k_arg);
        let _ = builder.arg(&mut counts);
        // SAFETY: thread `i < cycles` reads only `hot[i]`, of a buffer whose
        // length is checked against `cycles` above. A live address below
        // `one_hot_k` atomically increments
        // `counts[address * chunk_count + i / chunk_len]`, which is `< segments`
        // because `i / chunk_len < chunk_count`; an out-of-range address
        // increments the one extra trailing slot instead, and a cold cycle
        // increments nothing. `counts` holds `segments + 1` u32s zeroed by
        // `alloc_u32`, so every increment is in bounds and concurrent hits on
        // one counter are atomic.
        let _ = unsafe { builder.launch(Self::launch_config(cycle_count)) }?;
        self.stream().synchronize()?;
        drop(count_span);

        let (offsets, total, widest) = tracing::info_span!("cuda_commit_one_hot_scan", segments)
            .in_scope(|| {
                let histogram = self.download_u32(&counts)?;
                if histogram[segments] != 0 {
                    return Err(CudaError::InvariantViolation {
                        reason: "a one-hot address lies outside the declared address count",
                    });
                }
                let mut offsets = Vec::with_capacity(segments);
                let mut running = 0u32;
                let mut widest = 0u32;
                for &count in &histogram[..segments] {
                    offsets.push(running);
                    widest = widest.max(count);
                    running = running
                        .checked_add(count)
                        .ok_or(CudaError::InvariantViolation {
                            reason: "a one-hot segment plan holds more than u32::MAX entries",
                        })?;
                }
                Ok((
                    self.upload_u32_slice(&offsets)?,
                    running as usize,
                    widest as usize,
                ))
            })?;

        let scatter_span =
            tracing::info_span!("cuda_commit_one_hot_scatter", segments, entries = total).entered();
        let mut cursor = self.clone_u32(&offsets)?;
        let mut indices = self.alloc_u32(total.max(1))?;
        let mut builder = self.stream().launch_builder(self.msm_one_hot_scatter());
        let _ = builder.arg(hot);
        let _ = builder.arg(&cycle_count);
        let _ = builder.arg(&chunk_len_arg);
        let _ = builder.arg(&chunk_count_arg);
        let _ = builder.arg(&one_hot_k_arg);
        let _ = builder.arg(&mut cursor);
        let _ = builder.arg(&mut indices);
        // SAFETY: thread `i < cycles` reads `hot[i]` and, for a live in-range
        // address, atomically bumps `cursor[segment]` for the same `segment
        // < segments` the count kernel used, inside the `segments`-element copy
        // of the scan offsets. Because the cursor starts at the exclusive scan
        // of the very histogram those threads produced, the slots handed out are
        // exactly the `total` positions of `indices`, one per thread, so the
        // writes are disjoint and in bounds. Cold and out-of-range cycles were
        // excluded from the histogram and return without writing.
        let _ = unsafe { builder.launch(Self::launch_config(cycle_count)) }?;
        self.stream().synchronize()?;
        drop(scatter_span);

        let mut output = self.alloc_u64(segments * 3 * FQ_LIMBS)?;
        tracing::info_span!(
            "cuda_commit_one_hot_segments",
            segments,
            widest,
            entries = total
        )
        .in_scope(|| {
            self.launch_segment_sums(
                bases,
                SegmentPlan {
                    indices: &indices,
                    offsets: &offsets,
                    counts: &counts,
                    segments,
                    widest,
                },
                &mut output,
            )
        })?;
        tracing::info_span!("cuda_commit_one_hot_download", segments)
            .in_scope(|| Ok(unflatten_jacobian(&self.download_u64(&output)?)))
    }

    fn canonical_scalars(&self, values: &DeviceFrVec) -> Result<CudaSlice<u64>, CudaError> {
        let mut output = self.alloc_u64(values.len() * FQ_LIMBS)?;
        if values.is_empty() {
            return Ok(output);
        }
        let count = Self::count_of(values.len())?;
        let mut builder = self.stream().launch_builder(self.msm_from_montgomery());
        let _ = builder.arg(values.limbs());
        let _ = builder.arg(&mut output);
        let _ = builder.arg(&count);
        // SAFETY: thread `i < count` reads only `values[i]` and writes only
        // `out[i]` (4 limbs each); both buffers hold `count * LIMBS` u64s and
        // `out` is a fresh allocation distinct from the input. Threads with
        // `i >= count` return before any access.
        let _ = unsafe { builder.launch(Self::launch_config(count)) }?;
        self.stream().synchronize()?;
        Ok(output)
    }

    #[cfg(test)]
    fn segment_sums(
        &self,
        bases: &DeviceG1Bases,
        indices: &[u32],
        offsets: &[u32],
        counts: &[u32],
        segments: usize,
    ) -> Result<Vec<JacobianLimbs>, CudaError> {
        let device_indices = if indices.is_empty() {
            self.alloc_u32(1)?
        } else {
            self.upload_u32_slice(indices)?
        };
        let device_offsets = self.upload_u32_slice(offsets)?;
        let device_counts = self.upload_u32_slice(counts)?;
        let widest = counts.iter().copied().max().unwrap_or(0) as usize;
        let mut output = self.alloc_u64(segments * 3 * FQ_LIMBS)?;
        self.launch_segment_sums(
            bases,
            SegmentPlan {
                indices: &device_indices,
                offsets: &device_offsets,
                counts: &device_counts,
                segments,
                widest,
            },
            &mut output,
        )?;
        Ok(unflatten_jacobian(&self.download_u64(&output)?))
    }

    fn reduce_chunks(rows: usize, buckets: usize) -> usize {
        if rows == 0 || buckets <= 1 {
            return 1;
        }
        let saturated = (buckets - 1) / POINT_BLOCK as usize;
        let wanted = TARGET_REDUCE_BLOCKS.div_ceil(rows);
        saturated.min(wanted).max(1)
    }

    fn launch_bucket_reduce_chunked(
        &self,
        bucket_points: &CudaSlice<u64>,
        rows: usize,
        buckets: usize,
        chunks: usize,
        out: &mut CudaSlice<u64>,
    ) -> Result<(), CudaError> {
        if rows == 0 {
            return Ok(());
        }
        if buckets == 0 || chunks == 0 {
            return Err(CudaError::LengthMismatch {
                expected: 1,
                got: buckets.min(chunks),
            });
        }
        let rows_arg = Self::count_of(rows)?;
        let buckets_arg = Self::count_of(buckets)?;
        let chunks_arg = Self::count_of(chunks)?;
        let blocks = Self::count_of(rows * chunks)?;
        let shared = POINT_BLOCK * 3 * FQ_LIMBS as u32 * size_of::<u64>() as u32;
        let config = cudarc::driver::LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (POINT_BLOCK, 1, 1),
            shared_mem_bytes: shared,
        };

        if chunks == 1 {
            let mut builder = self
                .stream()
                .launch_builder(self.msm_bucket_reduce_chunked());
            let _ = builder.arg(bucket_points);
            let _ = builder.arg(&rows_arg);
            let _ = builder.arg(&buckets_arg);
            let _ = builder.arg(&chunks_arg);
            let _ = builder.arg(out);
            // SAFETY: block `b < rows * chunks` derives `row = b / chunks` and
            // returns when `row >= rows`. It reads only the `buckets` points of
            // its own row (`bucket_points[row * buckets ..]`, inside a
            // `rows * buckets` point buffer), each thread walking a disjoint
            // bucket sub-range of that block's chunk. The block reduces through
            // `POINT_BLOCK * 3 * LIMBS` u64s of dynamic shared memory, declared
            // here and matching the kernel's `extern __shared__`, with
            // `__syncthreads()` on both sides of every tree level. Only thread 0
            // writes, to `out[row * chunks + chunk]` (12 limbs) of a
            // `rows * chunks * 12` u64 buffer distinct from the input, so writes
            // are one per block.
            let _ = unsafe { builder.launch(config) }?;
            return Ok(());
        }

        let mut partials = self.alloc_u64(rows * chunks * 3 * FQ_LIMBS)?;
        let mut builder = self
            .stream()
            .launch_builder(self.msm_bucket_reduce_chunked());
        let _ = builder.arg(bucket_points);
        let _ = builder.arg(&rows_arg);
        let _ = builder.arg(&buckets_arg);
        let _ = builder.arg(&chunks_arg);
        let _ = builder.arg(&mut partials);
        // SAFETY: identical to the `chunks == 1` launch above, differing only in
        // that the output is the `rows * chunks * 12` u64 `partials` scratch
        // buffer rather than the caller's `rows * 12` output.
        let _ = unsafe { builder.launch(config) }?;

        let mut builder = self.stream().launch_builder(self.msm_point_rows_sum());
        let _ = builder.arg(&partials);
        let _ = builder.arg(&rows_arg);
        let _ = builder.arg(&chunks_arg);
        let _ = builder.arg(out);
        // SAFETY: block `row < rows` reads only `partials[row * chunks ..]`
        // (`chunks` points of a `rows * chunks` point buffer), striding by
        // `blockDim.x`, and reduces through the same `POINT_BLOCK * 3 * LIMBS`
        // u64 shared-memory tree with `__syncthreads()` on both sides of every
        // level. Only thread 0 writes, to `out[row]` (12 limbs) of a `rows * 12`
        // u64 buffer distinct from `partials`.
        let _ = unsafe {
            builder.launch(cudarc::driver::LaunchConfig {
                grid_dim: (rows_arg, 1, 1),
                block_dim: (POINT_BLOCK, 1, 1),
                shared_mem_bytes: shared,
            })
        }?;
        Ok(())
    }

    fn g2_reduce_and_fold(
        &self,
        bucket_points: &CudaSlice<u64>,
        rows: usize,
        buckets: usize,
        windows: usize,
        window_bits: usize,
    ) -> Result<CudaSlice<u64>, CudaError> {
        let lanes = windows * rows;
        let chunks = Self::reduce_chunks(lanes, buckets);
        let lanes_arg = Self::count_of(lanes)?;
        let buckets_arg = Self::count_of(buckets)?;
        let chunks_arg = Self::count_of(chunks)?;
        let shared = G2_POINT_BLOCK * G2_POINT_WORDS as u32 * size_of::<u64>() as u32;
        let mut window_points = self.alloc_u64(lanes * G2_POINT_WORDS)?;
        let mut partials = self.alloc_u64(lanes * chunks * G2_POINT_WORDS)?;

        let mut builder = self
            .stream()
            .launch_builder(self.msm_g2_bucket_reduce_chunked());
        let _ = builder.arg(bucket_points);
        let _ = builder.arg(&lanes_arg);
        let _ = builder.arg(&buckets_arg);
        let _ = builder.arg(&chunks_arg);
        let _ = builder.arg(if chunks == 1 {
            &mut window_points
        } else {
            &mut partials
        });
        // SAFETY: block `b < lanes * chunks` derives `lane = b / chunks`, returns
        // when `lane >= lanes`, and reads only the `buckets` G2 points of its own
        // lane inside the `lanes * buckets` point input, each thread walking a
        // disjoint bucket sub-range of that block's chunk. The block reduces
        // through `G2_POINT_BLOCK * G2_POINT_WORDS` u64s of dynamic shared
        // memory, declared here and matching the kernel's `extern __shared__`,
        // with `__syncthreads()` on both sides of every tree level. Only thread 0
        // writes, to element `lane * chunks + chunk` of the output, which is a
        // `lanes * chunks`-point buffer when `chunks > 1` and a `lanes`-point
        // buffer when `chunks == 1` (then `chunk` is always 0), distinct from the
        // input either way.
        let _ = unsafe {
            builder.launch(cudarc::driver::LaunchConfig {
                grid_dim: (Self::count_of(lanes * chunks)?, 1, 1),
                block_dim: (G2_POINT_BLOCK, 1, 1),
                shared_mem_bytes: shared,
            })
        }?;

        if chunks > 1 {
            let mut builder = self.stream().launch_builder(self.msm_g2_point_rows_sum());
            let _ = builder.arg(&partials);
            let _ = builder.arg(&lanes_arg);
            let _ = builder.arg(&chunks_arg);
            let _ = builder.arg(&mut window_points);
            // SAFETY: block `lane < lanes` reads only `partials[lane * chunks ..]`
            // (`chunks` G2 points of a `lanes * chunks` point buffer), striding by
            // `blockDim.x`, and reduces through the same shared-memory tree with
            // `__syncthreads()` on both sides of every level. Only thread 0
            // writes, to `out[lane]` of a `lanes`-point buffer distinct from
            // `partials`.
            let _ = unsafe {
                builder.launch(cudarc::driver::LaunchConfig {
                    grid_dim: (lanes_arg, 1, 1),
                    block_dim: (G2_POINT_BLOCK, 1, 1),
                    shared_mem_bytes: shared,
                })
            }?;
        }

        let rows_arg = Self::count_of(rows)?;
        let windows_arg = Self::count_of(windows)?;
        let window_bits_arg = Self::count_of(window_bits)?;
        let mut accumulator = self.alloc_u64(rows * G2_POINT_WORDS)?;
        let mut builder = self.stream().launch_builder(self.msm_g2_window_fold());
        let _ = builder.arg(&window_points);
        let _ = builder.arg(&rows_arg);
        let _ = builder.arg(&windows_arg);
        let _ = builder.arg(&window_bits_arg);
        let _ = builder.arg(&mut accumulator);
        // SAFETY: thread `row < rows` reads only `window_points[window * rows +
        // row]` for each `window < windows` (one G2 point each of a
        // `windows * rows` point buffer) and writes only `out[row]` of a `rows`
        // point buffer distinct from the input. Threads with `row >= rows` return
        // before any access.
        let _ = unsafe { builder.launch(Self::launch_config(rows_arg)) }?;
        Ok(accumulator)
    }

    pub(crate) fn g2_msm_in_place(
        &self,
        buffer: &mut CudaSlice<u64>,
        base_offset: usize,
        out_offset: usize,
        count: usize,
        scalars: &[Fr],
    ) -> Result<(), CudaError> {
        if count != scalars.len() {
            return Err(CudaError::LengthMismatch {
                expected: count,
                got: scalars.len(),
            });
        }
        Self::resident_span_end(out_offset, 1, G2_POINT_WORDS, buffer)?;
        if count == 0 {
            let zero = vec![0u64; G2_POINT_WORDS];
            return self.write_u64_range(buffer, out_offset * G2_POINT_WORDS, &zero);
        }
        Self::resident_span_end(base_offset, count, G2_POINT_WORDS, buffer)?;
        if 4 * count > MAX_BASE_INDEX {
            return Err(CudaError::InvariantViolation {
                reason: "a G2 MSM row is wider than the signed-index encoding allows",
            });
        }

        let terms = 4 * count;
        let window_bits = window_bits(terms);
        let buckets = 1usize << window_bits;
        let windows = GLV4_SCALAR_BITS.div_ceil(window_bits);
        let segments = windows * buckets;
        if !Self::batched_windows_g2(buckets, windows, terms) {
            return Err(CudaError::NotImplemented {
                kernel: "msm_g2_unbatched_windows",
            });
        }

        let mapped = self.g2_frobenius_span(buffer, base_offset, count)?;
        let (split, signs) = {
            let uploaded = self.upload(scalars)?;
            let canonical = self.canonical_scalars(&uploaded)?;
            self.glv_decompose_4d(&canonical, count)?
        };
        let mut lanes = Vec::with_capacity(signs.len() * windows);
        for _ in 0..windows {
            lanes.extend_from_slice(&signs);
        }
        let device_signs = self.upload_u8_slice(&lanes)?;
        let shifts: Vec<usize> = (0..windows).map(|window| window * window_bits).collect();
        let index = self.bucket_index_pass(IndexPlan {
            scalars: &split,
            scalar_limbs: FQ_LIMBS,
            signs: &device_signs,
            row_len: terms,
            buckets,
            segments,
            lane: terms,
            shifts: &shifts,
        })?;
        if index.widest > SMALL_SEGMENT_LIMIT {
            return Err(CudaError::NotImplemented {
                kernel: "msm_g2_wide_segments",
            });
        }

        let bases = mapped.slice(0..terms * G2_POINT_WORDS);
        let segments_arg = Self::count_of(segments)?;
        let mut bucket_points = self.alloc_u64(segments * G2_POINT_WORDS)?;
        let mut builder = self
            .stream()
            .launch_builder(self.msm_g2_segment_sum_small());
        let _ = builder.arg(&bases);
        let _ = builder.arg(&index.indices);
        let _ = builder.arg(&index.offsets);
        let _ = builder.arg(&index.counts);
        let _ = builder.arg(&segments_arg);
        let _ = builder.arg(&mut bucket_points);
        // SAFETY: thread `s < segments` reads `offsets[s]`/`counts[s]` and only
        // the `indices` window they delimit — the scatter builds those as a
        // partition of `indices` — and reads `bases` at the G2 points those
        // indices name, each masked to 31 bits and less than `count` because the
        // scatter stores `i % row_len` with `row_len = count`, inside the
        // `count`-point view checked above. It writes only `out[s]` of a
        // `segments`-point fresh allocation, one thread per segment. Threads with
        // `s >= segments` return first.
        let _ = unsafe { builder.launch(Self::launch_config(segments_arg)) }?;

        let accumulator =
            self.g2_reduce_and_fold(&bucket_points, 1, buckets, windows, window_bits)?;
        let source = accumulator.slice(0..G2_POINT_WORDS);
        let mut target =
            buffer.slice_mut(out_offset * G2_POINT_WORDS..(out_offset + 1) * G2_POINT_WORDS);
        self.stream().memcpy_dtod(&source, &mut target)?;
        Ok(())
    }

    fn batched_windows_g2(buckets: usize, windows: usize, len: usize) -> bool {
        let segments = windows.saturating_mul(buckets);
        let bucket_bytes = segments.saturating_mul(G2_POINT_WORDS * size_of::<u64>());
        let index_bytes = windows
            .saturating_mul(len)
            .saturating_mul(2 * size_of::<u32>());
        bucket_bytes.saturating_add(index_bytes) <= BATCHED_WINDOW_BYTES
    }

    fn glv_decompose_4d(
        &self,
        scalars: &CudaSlice<u64>,
        count: usize,
    ) -> Result<(CudaSlice<u64>, Vec<u8>), CudaError> {
        let mut out = self.alloc_u64(4 * count.max(1) * FQ_LIMBS)?;
        if count == 0 {
            return Ok((out, Vec::new()));
        }
        if scalars.len() < count * FQ_LIMBS {
            return Err(CudaError::LengthMismatch {
                expected: count * FQ_LIMBS,
                got: scalars.len(),
            });
        }
        let (magnitudes, table_signs) = glv_4d_table();
        let device_table = self.upload_u64_slice(magnitudes)?;
        let device_table_signs = self.upload_u8_slice(table_signs)?;
        let mut signs = self.upload_u8_slice(&vec![0u8; 4 * count])?;
        let count_arg = Self::count_of(count)?;
        let mut builder = self.stream().launch_builder(self.msm_glv_decompose_4d());
        let _ = builder.arg(scalars);
        let _ = builder.arg(&device_table);
        let _ = builder.arg(&device_table_signs);
        let _ = builder.arg(&count_arg);
        let _ = builder.arg(&mut out);
        let _ = builder.arg(&mut signs);
        // SAFETY: thread `i < count` reads only `scalars[i]` (4 limbs of a buffer
        // holding at least `count * FQ_LIMBS`, checked above) and rows
        // `0..254` of the two table buffers, whose lengths are `254 * 8` u64 and
        // `254 * 4` u8 by construction in `glv_4d_table`. It writes elements
        // `j * count + i` for `j < 4` of both a `4 * count * FQ_LIMBS` u64 output
        // and a `4 * count` u8 sign array, so writes are four disjoint slots per
        // thread in each and in bounds. Threads with `i >= count` return first.
        let _ = unsafe { builder.launch(Self::launch_config(count_arg)) }?;
        let hosted = self.download_u8(&signs)?;
        Ok((out, hosted))
    }

    fn g2_frobenius_span(
        &self,
        jacobian: &CudaSlice<u64>,
        offset: usize,
        count: usize,
    ) -> Result<CudaSlice<u64>, CudaError> {
        let end = offset
            .checked_add(count)
            .and_then(|last| last.checked_mul(G2_POINT_WORDS))
            .ok_or(CudaError::InvariantViolation {
                reason: "a G2 Frobenius span overflowed the buffer index space",
            })?;
        if end > jacobian.len() {
            return Err(CudaError::LengthMismatch {
                expected: end,
                got: jacobian.len(),
            });
        }
        let mut out = self.alloc_u64(4 * count.max(1) * G2_POINT_WORDS)?;
        if count == 0 {
            return Ok(out);
        }
        let span = jacobian.slice(offset * G2_POINT_WORDS..end);
        let coefficients = self.upload_u64_slice(glv_frobenius_coefficients())?;
        let count_arg = Self::count_of(count)?;
        let mut builder = self.stream().launch_builder(self.msm_g2_frobenius());
        let _ = builder.arg(&span);
        let _ = builder.arg(&coefficients);
        let _ = builder.arg(&count_arg);
        let _ = builder.arg(&mut out);
        // SAFETY: thread `i < count` reads the 24 limbs of point `i` of the
        // `count`-point view bounds-checked above plus the 48 coefficient limbs,
        // and writes points `power * count + i` for `power < 4` (24 limbs each) of
        // a `4 * count` point fresh allocation, so writes are four disjoint slots
        // per thread and in bounds. Threads with `i >= count` return first.
        let _ = unsafe { builder.launch(Self::launch_config(count_arg)) }?;
        Ok(out)
    }

    fn glv_decompose_2d(
        &self,
        scalars: &CudaSlice<u64>,
        count: usize,
    ) -> Result<(CudaSlice<u64>, Vec<u8>), CudaError> {
        let mut out = self.alloc_u64(2 * count.max(1) * FQ_LIMBS)?;
        if count == 0 {
            return Ok((out, Vec::new()));
        }
        if scalars.len() < count * FQ_LIMBS {
            return Err(CudaError::LengthMismatch {
                expected: count * FQ_LIMBS,
                got: scalars.len(),
            });
        }
        let constants = self.upload_u64_slice(&GLV_CONSTANTS)?;
        let mut signs = self.upload_u8_slice(&vec![0u8; 2 * count])?;
        let count_arg = Self::count_of(count)?;
        let mut builder = self.stream().launch_builder(self.msm_glv_decompose_2d());
        let _ = builder.arg(scalars);
        let _ = builder.arg(&constants);
        let _ = builder.arg(&count_arg);
        let _ = builder.arg(&mut out);
        let _ = builder.arg(&mut signs);
        // SAFETY: thread `i < count` reads only `scalars[i]` (4 limbs of a buffer
        // holding at least `count * FQ_LIMBS`, checked above) and the 14 constant
        // limbs, and writes elements `i` and `count + i` of both a
        // `2 * count * FQ_LIMBS` u64 output and a `2 * count` u8 sign array, so
        // writes are two disjoint slots per thread in each and in bounds. All
        // wide arithmetic is in thread-local arrays. Threads with `i >= count`
        // return before any access.
        let _ = unsafe { builder.launch(Self::launch_config(count_arg)) }?;
        let hosted = self.download_u8(&signs)?;
        Ok((out, hosted))
    }

    fn g1_endomorphism_span(
        &self,
        jacobian: &CudaSlice<u64>,
        offset: usize,
        count: usize,
    ) -> Result<CudaSlice<u64>, CudaError> {
        let words = 3 * FQ_LIMBS;
        let end = offset
            .checked_add(count)
            .and_then(|last| last.checked_mul(words))
            .ok_or(CudaError::InvariantViolation {
                reason: "a G1 endomorphism span overflowed the buffer index space",
            })?;
        if end > jacobian.len() {
            return Err(CudaError::LengthMismatch {
                expected: end,
                got: jacobian.len(),
            });
        }
        let mut out = self.alloc_u64(2 * count * words)?;
        if count == 0 {
            return Ok(out);
        }
        let span = jacobian.slice(offset * words..end);
        let beta = self.upload_u64_slice(&glv_endomorphism_coefficient())?;
        let count_arg = Self::count_of(count)?;
        let mut builder = self.stream().launch_builder(self.msm_g1_endomorphism());
        let _ = builder.arg(&span);
        let _ = builder.arg(&beta);
        let _ = builder.arg(&count_arg);
        let _ = builder.arg(&mut out);
        // SAFETY: thread `i < count` reads the 12 limbs of point `i` of the
        // `count`-point view bounds-checked above plus the 4 limbs of `beta`, and
        // writes points `i` and `count + i` (12 limbs each) of a `2 * count`
        // point fresh allocation, so writes are two disjoint slots per thread and
        // in bounds. Threads with `i >= count` return before any access.
        let _ = unsafe { builder.launch(Self::launch_config(count_arg)) }?;
        Ok(out)
    }

    pub fn normalise_g1_span(
        &self,
        jacobian: &CudaSlice<u64>,
        offset: usize,
        count: usize,
    ) -> Result<DeviceG1Bases, CudaError> {
        if count == 0 {
            return self.upload_g1_bases(&[]);
        }
        let words = 3 * FQ_LIMBS;
        let end = offset
            .checked_add(count)
            .and_then(|last| last.checked_mul(words))
            .ok_or(CudaError::InvariantViolation {
                reason: "a G1 span to normalise overflowed the buffer index space",
            })?;
        if end > jacobian.len() {
            return Err(CudaError::LengthMismatch {
                expected: end,
                got: jacobian.len(),
            });
        }
        let span = jacobian.slice(offset * words..end);
        let count_arg = Self::count_of(count)?;

        let mut packed = self.alloc_u64(count * FQ_LIMBS)?;
        let mut builder = self.stream().launch_builder(self.msm_jacobian_z());
        let _ = builder.arg(&span);
        let _ = builder.arg(&count_arg);
        let _ = builder.arg(&mut packed);
        // SAFETY: thread `i < count` reads the `z` limbs of point `i` of the
        // `count`-point view (12 limbs per point, bounds-checked above) and
        // writes only `out[i]` (4 limbs) of a `count * FQ_LIMBS` u64 fresh
        // allocation, one thread per element.
        let _ = unsafe { builder.launch(Self::launch_config(count_arg)) }?;

        let mut inverses = self.alloc_u64(count * FQ_LIMBS)?;
        self.launch_batch_inverse(&packed, &mut inverses, count)?;

        let mut affine = self.alloc_u64(count * AFFINE_LIMBS)?;
        let mut builder = self.stream().launch_builder(self.msm_jacobian_to_affine());
        let _ = builder.arg(&span);
        let _ = builder.arg(&inverses);
        let _ = builder.arg(&count_arg);
        let _ = builder.arg(&mut affine);
        // SAFETY: thread `i < count` reads the 12 limbs of point `i` of the same
        // bounds-checked view plus `inverses[i]` (4 limbs of a
        // `count * FQ_LIMBS` buffer) and writes only the two limb groups of
        // `out[i]` in a `count * AFFINE_LIMBS` u64 fresh allocation, one thread
        // per point.
        let _ = unsafe { builder.launch(Self::launch_config(count_arg)) }?;

        Ok(DeviceG1Bases {
            stream: self.stream().clone(),
            limbs: affine,
            count,
        })
    }

    pub(crate) fn g1_msm_in_place(
        &self,
        buffer: &mut CudaSlice<u64>,
        base_offset: usize,
        out_offset: usize,
        count: usize,
        scalars: &[Fr],
    ) -> Result<(), CudaError> {
        if count != scalars.len() {
            return Err(CudaError::LengthMismatch {
                expected: count,
                got: scalars.len(),
            });
        }
        let words = 3 * FQ_LIMBS;
        Self::resident_span_end(out_offset, 1, words, buffer)?;
        if count == 0 {
            let zero = vec![0u64; words];
            return self.write_u64_range(buffer, out_offset * words, &zero);
        }
        let bases = {
            let _span = tracing::info_span!("g1r_normalise", len = count).entered();
            let mapped = self.g1_endomorphism_span(buffer, base_offset, count)?;
            self.normalise_g1_span(&mapped, 0, 2 * count)?
        };
        let (split, signs) = {
            let _span = tracing::info_span!("g1r_scalars", len = count).entered();
            let uploaded = self.upload(scalars)?;
            let canonical = self.canonical_scalars(&uploaded)?;
            self.glv_decompose_2d(&canonical, count)?
        };
        let accumulator = {
            let _span = tracing::info_span!("g1r_pippenger", len = count).entered();
            self.pippenger_device(
                &bases,
                DeviceScalars {
                    values: &split,
                    limbs: FQ_LIMBS,
                },
                &signs,
                2 * count,
                2 * count,
                GLV_SCALAR_BITS,
            )?
        };
        let source = accumulator.slice(0..words);
        let mut target = buffer.slice_mut(out_offset * words..(out_offset + 1) * words);
        self.stream().memcpy_dtod(&source, &mut target)?;
        Ok(())
    }

    fn batched_windows(rows: usize, buckets: usize, windows: usize, len: usize) -> bool {
        let segments = windows.saturating_mul(rows).saturating_mul(buckets);
        let bucket_bytes = segments.saturating_mul(3 * FQ_LIMBS * size_of::<u64>());
        let index_bytes = windows
            .saturating_mul(len)
            .saturating_mul(2 * size_of::<u32>());
        windows > 1 && bucket_bytes.saturating_add(index_bytes) <= BATCHED_WINDOW_BYTES
    }

    fn bucket_index_pass(&self, plan: IndexPlan<'_>) -> Result<BucketIndex, CudaError> {
        let IndexPlan {
            scalars,
            scalar_limbs,
            signs,
            row_len,
            buckets,
            segments,
            lane,
            shifts,
        } = plan;
        let total = lane.saturating_mul(shifts.len());
        let count = Self::count_of(total)?;
        let lane_count = Self::count_of(lane)?;
        let row_len_arg = Self::count_of(row_len)?;
        let buckets_arg = Self::count_of(buckets)?;
        let mask = Self::count_of(buckets - 1)?;
        let limbs_arg = Self::count_of(scalar_limbs)?;

        let mut digits = self.alloc_u32(total)?;
        for (index, &shift_bits) in shifts.iter().enumerate() {
            let shift = Self::count_of(shift_bits)?;
            let mut lane_digits = digits.slice_mut(index * lane..(index + 1) * lane);
            let mut builder = self.stream().launch_builder(self.msm_digits());
            let _ = builder.arg(scalars);
            let _ = builder.arg(&lane_count);
            let _ = builder.arg(&limbs_arg);
            let _ = builder.arg(&shift);
            let _ = builder.arg(&mask);
            let _ = builder.arg(&mut lane_digits);
            // SAFETY: thread `i < lane_count` reads only `scalars[i]` (4 limbs of
            // a `lane * LIMBS` buffer) and writes only element `i` of the
            // `lane`-element view `digits[index * lane ..]`, so lanes are
            // disjoint and in bounds. The digit extraction reads at most limbs
            // `shift/64` and `shift/64 + 1`, both bounds-checked in the kernel.
            let _ = unsafe { builder.launch(Self::launch_config(lane_count)) }?;
        }

        let mut counts = self.alloc_u32(segments)?;
        let mut builder = self.stream().launch_builder(self.msm_bucket_count());
        let _ = builder.arg(&digits);
        let _ = builder.arg(&count);
        let _ = builder.arg(&row_len_arg);
        let _ = builder.arg(&buckets_arg);
        let _ = builder.arg(&mut counts);
        // SAFETY: thread `i < count` reads `digits[i]` and atomically increments
        // `counts[(i / row_len) * buckets + digit]`. The digit is masked to
        // `buckets - 1` by the digit kernel and `i / row_len < rows` because
        // `count = rows * row_len`, so the index stays inside the
        // `segments = rows * buckets` u32 allocation, which was zeroed by
        // `alloc_u32`. The increment is an atomic, so concurrent hits on one
        // counter are safe.
        let _ = unsafe { builder.launch(Self::launch_config(count)) }?;

        let offsets = self.exclusive_scan_u32_on_device(&counts, segments)?;
        let mut cursor = self.clone_u32(&offsets)?;
        let mut indices = self.alloc_u32(total.max(1))?;
        let mut builder = self.stream().launch_builder(self.msm_bucket_scatter());
        let _ = builder.arg(&digits);
        let _ = builder.arg(signs);
        let _ = builder.arg(&count);
        let _ = builder.arg(&row_len_arg);
        let _ = builder.arg(&buckets_arg);
        let _ = builder.arg(&mut cursor);
        let _ = builder.arg(&mut indices);
        // SAFETY: thread `i < count` reads `digits[i]` and `signs[i]` (`count`
        // elements each — the caller replicates `signs` once per shift lane),
        // atomically bumps `cursor[(i / row_len) * buckets + digit]` inside the
        // `segments` u32 copy of the scan offsets, and writes the returned slot
        // of `indices`. Because the cursor starts at the exclusive scan of the
        // same counts the previous kernel produced, the slots handed out are
        // exactly the `total` positions of `indices`, each to one thread — so
        // writes are disjoint and in bounds.
        let _ = unsafe { builder.launch(Self::launch_config(count)) }?;
        self.stream().synchronize()?;

        Ok(BucketIndex {
            indices,
            offsets,
            counts,
            widest: total.div_ceil(segments.max(1)).saturating_mul(4),
        })
    }

    fn pippenger_pass(
        &self,
        plan: PassPlan<'_>,
        out: &mut CudaSlice<u64>,
    ) -> Result<(), CudaError> {
        let PassPlan {
            bases,
            scalars,
            scalar_limbs,
            signs,
            row_len,
            buckets,
            rows,
            lane,
            shifts,
        } = plan;
        let total = lane.saturating_mul(shifts.len());
        if total == 0 {
            return Ok(());
        }
        let segments = rows * buckets;
        let index = self.bucket_index_pass(IndexPlan {
            scalars,
            scalar_limbs,
            signs,
            row_len,
            buckets,
            segments,
            lane,
            shifts,
        })?;

        let mut bucket_points = self.alloc_u64(segments * 3 * FQ_LIMBS)?;
        self.launch_segment_sums(
            bases,
            SegmentPlan {
                indices: &index.indices,
                offsets: &index.offsets,
                counts: &index.counts,
                segments,
                widest: index.widest,
            },
            &mut bucket_points,
        )?;

        self.launch_bucket_reduce_chunked(
            &bucket_points,
            rows,
            buckets,
            Self::reduce_chunks(rows, buckets),
            out,
        )
    }

    fn launch_window_fold(
        &self,
        window_points: &CudaSlice<u64>,
        rows: usize,
        windows: usize,
        window_bits: usize,
        out: &mut CudaSlice<u64>,
    ) -> Result<(), CudaError> {
        if rows == 0 || windows == 0 {
            return Ok(());
        }
        let rows_arg = Self::count_of(rows)?;
        let windows_arg = Self::count_of(windows)?;
        let window_bits_arg = Self::count_of(window_bits)?;
        let mut builder = self.stream().launch_builder(self.msm_window_fold());
        let _ = builder.arg(window_points);
        let _ = builder.arg(&rows_arg);
        let _ = builder.arg(&windows_arg);
        let _ = builder.arg(&window_bits_arg);
        let _ = builder.arg(out);
        // SAFETY: thread `row < rows` reads only `window_points[window * rows +
        // row]` for each `window < windows` (12 limbs each of a
        // `windows * rows * 12` u64 buffer) and writes only `out[row]` (12
        // limbs) of a `rows * 12` u64 buffer distinct from the input, so reads
        // stay in bounds and writes are one per thread. Threads with
        // `row >= rows` return before any access.
        let _ = unsafe { builder.launch(Self::launch_config(rows_arg)) }?;
        Ok(())
    }

    fn launch_segment_sums(
        &self,
        bases: &DeviceG1Bases,
        plan: SegmentPlan<'_>,
        output: &mut CudaSlice<u64>,
    ) -> Result<(), CudaError> {
        let SegmentPlan {
            indices,
            offsets,
            counts,
            segments,
            widest,
        } = plan;
        let segment_count = Self::count_of(segments)?;
        if segment_count == 0 {
            return Ok(());
        }
        if widest <= SMALL_SEGMENT_LIMIT {
            let mut builder = self.stream().launch_builder(self.msm_segment_sum_small());
            let _ = builder.arg(bases.limbs());
            let _ = builder.arg(indices);
            let _ = builder.arg(offsets);
            let _ = builder.arg(counts);
            let _ = builder.arg(&segment_count);
            let _ = builder.arg(output);
            // SAFETY: thread `s < segments` reads `offsets[s]`/`counts[s]` and
            // only the `indices` window they delimit — the host builds those as
            // a partition of `indices`, so windows are disjoint across threads
            // — and reads `bases` at the affine points those indices name
            // (masked to 31 bits, checked against `bases.count()` on the host).
            // It writes only `out[s]` (12 limbs) of a `segments * 12` u64 fresh
            // allocation, one thread per segment. Threads with `s >= segments`
            // return first. No shared memory and no barriers, which is the
            // whole point of this path.
            let _ = unsafe { builder.launch(Self::launch_config(segment_count)) }?;
            self.stream().synchronize()?;
            return Ok(());
        }
        let mut builder = self.stream().launch_builder(self.msm_segment_sum());
        let _ = builder.arg(bases.limbs());
        let _ = builder.arg(indices);
        let _ = builder.arg(offsets);
        let _ = builder.arg(counts);
        let _ = builder.arg(&segment_count);
        let _ = builder.arg(output);
        // SAFETY: block `s < segments` reads `offsets[s]`/`counts[s]` and only
        // the `indices` window they delimit — the host builds those as a
        // partition of `indices`, so windows are disjoint across blocks — and
        // reads `bases` at the affine points those indices name (every index is
        // masked to 31 bits and checked against `bases.count()` on the host).
        // The block reduces through `POINT_BLOCK * 3 * LIMBS` u64s of dynamic
        // shared memory, declared in the launch config and matching the
        // kernel's `extern __shared__`, with `__syncthreads()` on both sides of
        // every tree level. Only thread 0 writes, to `out[s]` (12 limbs) of a
        // `segments * 12` u64 fresh allocation, so writes are one per block and
        // non-aliasing.
        let _ = unsafe {
            builder.launch(cudarc::driver::LaunchConfig {
                grid_dim: (segment_count, 1, 1),
                block_dim: (POINT_BLOCK, 1, 1),
                shared_mem_bytes: POINT_BLOCK * 3 * FQ_LIMBS as u32 * size_of::<u64>() as u32,
            })
        }?;
        self.stream().synchronize()?;
        Ok(())
    }

    fn pippenger_device(
        &self,
        bases: &DeviceG1Bases,
        scalars: DeviceScalars<'_>,
        signs: &[u8],
        len: usize,
        row_len: usize,
        scalar_bits: usize,
    ) -> Result<CudaSlice<u64>, CudaError> {
        let DeviceScalars {
            values: scalars,
            limbs: scalar_limbs,
        } = scalars;
        if row_len == 0 || !len.is_multiple_of(row_len) {
            return Err(CudaError::LengthMismatch {
                expected: row_len,
                got: len,
            });
        }
        if bases.count() < row_len {
            return Err(CudaError::LengthMismatch {
                expected: row_len,
                got: bases.count(),
            });
        }
        if row_len > MAX_BASE_INDEX {
            return Err(CudaError::InvariantViolation {
                reason: "a Pippenger row is wider than the signed-index encoding allows",
            });
        }
        let rows = len / row_len;
        let window_bits = window_bits(row_len);
        let buckets = 1usize << window_bits;
        let windows = scalar_bits.div_ceil(window_bits);

        let mut accumulator = self.alloc_u64(rows * 3 * FQ_LIMBS)?;
        let rows_arg = Self::count_of(rows)?;

        if Self::batched_windows(rows, buckets, windows, len) {
            let _batched = tracing::info_span!(
                "cuda_pippenger_batched",
                rows,
                windows,
                buckets,
                lanes = signs.len() * windows
            )
            .entered();
            let mut lanes = Vec::with_capacity(signs.len() * windows);
            for _ in 0..windows {
                lanes.extend_from_slice(signs);
            }
            let device_lane_signs = self.upload_u8_slice(&lanes)?;
            let shifts: Vec<usize> = (0..windows).map(|window| window * window_bits).collect();
            let mut all_window_points = self.alloc_u64(windows * rows * 3 * FQ_LIMBS)?;
            self.pippenger_pass(
                PassPlan {
                    bases,
                    scalars,
                    scalar_limbs,
                    signs: &device_lane_signs,
                    row_len,
                    buckets,
                    rows: windows * rows,
                    lane: len,
                    shifts: &shifts,
                },
                &mut all_window_points,
            )?;
            self.launch_window_fold(
                &all_window_points,
                rows,
                windows,
                window_bits,
                &mut accumulator,
            )?;
            self.stream().synchronize()?;
            return Ok(accumulator);
        }

        let _looped =
            tracing::info_span!("cuda_pippenger_looped", rows, windows, buckets).entered();
        let device_signs = self.upload_u8_slice(signs)?;
        let mut window_points = self.alloc_u64(rows * 3 * FQ_LIMBS)?;
        for window in (0..windows).rev() {
            self.pippenger_pass(
                PassPlan {
                    bases,
                    scalars,
                    scalar_limbs,
                    signs: &device_signs,
                    row_len,
                    buckets,
                    rows,
                    lane: len,
                    shifts: &[window * window_bits],
                },
                &mut window_points,
            )?;
            self.stream().synchronize()?;

            let doublings = if window + 1 == windows {
                0
            } else {
                Self::count_of(window_bits)?
            };
            let mut builder = self.stream().launch_builder(self.msm_window_accumulate());
            let _ = builder.arg(&mut accumulator);
            let _ = builder.arg(&window_points);
            let _ = builder.arg(&rows_arg);
            let _ = builder.arg(&doublings);
            // SAFETY: thread `row < rows` read-modify-writes only
            // `accumulator[row]` and reads only `window[row]` (12 limbs each of
            // two distinct `rows * 12` u64 allocations), so both accesses are
            // one-per-thread and in bounds.
            let _ = unsafe { builder.launch(Self::launch_config(rows_arg)) }?;
            self.stream().synchronize()?;
        }

        Ok(accumulator)
    }
}

impl CudaKernelContext {
    pub(crate) fn zero_extend(
        &self,
        source: &DeviceFrVec,
        len: usize,
    ) -> Result<DeviceFrVec, CudaError> {
        let mut output = self.alloc(len)?;
        self.copy_into(&mut output, 0, source)?;
        Ok(output)
    }

    pub(crate) fn block_embed(
        &self,
        source: &DeviceFrVec,
        block_vars: usize,
        total_vars: usize,
    ) -> Result<DeviceFrVec, CudaError> {
        let mut output = self.alloc(1usize << total_vars)?;
        if source.is_empty() {
            return Ok(output);
        }
        let count = Self::count_of(source.len())?;
        let sigma_block = Self::count_of(block_vars.div_ceil(2))?;
        let sigma_main = Self::count_of(total_vars.div_ceil(2))?;
        let mut builder = self.stream().launch_builder(self.msm_block_embed());
        let _ = builder.arg(source.limbs());
        let _ = builder.arg(&sigma_block);
        let _ = builder.arg(&sigma_main);
        let _ = builder.arg(&count);
        let _ = builder.arg(output.limbs_mut());
        // SAFETY: thread `i < count` reads only `src[i]` and writes only
        // `out[(i >> sigma_block) << sigma_main | (i & mask)]`. That map is
        // injective (the row index and the `sigma_block`-wide column are
        // disjoint bit fields of the destination), and its image lies inside
        // `2^total_vars` because `block_vars <= total_vars` implies
        // `(count >> sigma_block) << sigma_main < 2^total_vars`. `out` is a
        // fresh zeroed allocation distinct from `src`.
        let _ = unsafe { builder.launch(Self::launch_config(count)) }?;
        self.stream().synchronize()?;
        Ok(output)
    }

    pub(crate) fn scatter_strided(
        &self,
        source: &DeviceFrVec,
        stride: usize,
        len: usize,
    ) -> Result<DeviceFrVec, CudaError> {
        let mut output = self.alloc(len)?;
        if source.is_empty() {
            return Ok(output);
        }
        if (source.len() - 1) * stride >= len {
            return Err(CudaError::LengthMismatch {
                expected: len,
                got: (source.len() - 1) * stride + 1,
            });
        }
        let count = Self::count_of(source.len())?;
        let stride_arg = Self::count_of(stride)?;
        let mut builder = self.stream().launch_builder(self.msm_scatter_strided());
        let _ = builder.arg(source.limbs());
        let _ = builder.arg(&stride_arg);
        let _ = builder.arg(&count);
        let _ = builder.arg(output.limbs_mut());
        // SAFETY: thread `i < count` reads only `src[i]` and writes only
        // `out[i * stride]`, which is injective in `i` and bounded by the check
        // above. `out` is a fresh zeroed allocation distinct from `src`.
        let _ = unsafe { builder.launch(Self::launch_config(count)) }?;
        self.stream().synchronize()?;
        Ok(output)
    }

    pub(crate) fn scatter_one_hot(
        &self,
        source: &DeviceFrVec,
        cycles: usize,
        cycle_stride: usize,
        one_hot_stride: usize,
        len: usize,
    ) -> Result<DeviceFrVec, CudaError> {
        let mut output = self.alloc(len)?;
        if source.is_empty() {
            return Ok(output);
        }
        if cycles == 0 || !source.len().is_multiple_of(cycles) {
            return Err(CudaError::LengthMismatch {
                expected: cycles,
                got: source.len(),
            });
        }
        let addresses = source.len() / cycles;
        let highest = (cycles - 1) * cycle_stride + (addresses - 1) * one_hot_stride;
        if highest >= len {
            return Err(CudaError::LengthMismatch {
                expected: len,
                got: highest + 1,
            });
        }
        let count = Self::count_of(source.len())?;
        let cycles_arg = Self::count_of(cycles)?;
        let cycle_stride_arg = Self::count_of(cycle_stride)?;
        let one_hot_stride_arg = Self::count_of(one_hot_stride)?;
        let mut builder = self.stream().launch_builder(self.msm_scatter_one_hot());
        let _ = builder.arg(source.limbs());
        let _ = builder.arg(&cycles_arg);
        let _ = builder.arg(&cycle_stride_arg);
        let _ = builder.arg(&one_hot_stride_arg);
        let _ = builder.arg(&count);
        let _ = builder.arg(output.limbs_mut());
        // SAFETY: thread `i < count` reads only `src[i]` and writes
        // `out[cycle * cycle_stride + address * one_hot_stride]` for the unique
        // `(address, cycle) = (i / cycles, i % cycles)`. The strides are the
        // grid's own powers of two with `cycle_stride = 2^k * one_hot_stride`
        // and `addresses <= 2^log_k_chunk`, so distinct `(address, cycle)`
        // pairs give distinct destinations; the largest is bounds-checked
        // above. `out` is a fresh zeroed allocation distinct from `src`.
        let _ = unsafe { builder.launch(Self::launch_config(count)) }?;
        self.stream().synchronize()?;
        Ok(output)
    }

    pub(crate) fn fold_rows(
        &self,
        table: &DeviceFrVec,
        left: &[Fr],
        sigma: usize,
    ) -> Result<Vec<Fr>, CudaError> {
        let columns = 1usize << sigma;
        if !table.len().is_multiple_of(columns) {
            return Err(CudaError::LengthMismatch {
                expected: columns,
                got: table.len(),
            });
        }
        let rows = table.len() / columns;
        if left.len() != rows {
            return Err(CudaError::LengthMismatch {
                expected: rows,
                got: left.len(),
            });
        }
        self.fold_rows_window(table, 0, table.len(), left, sigma)
    }

    pub(crate) fn fold_rows_window(
        &self,
        table: &DeviceFrVec,
        base: usize,
        len: usize,
        left: &[Fr],
        sigma: usize,
    ) -> Result<Vec<Fr>, CudaError> {
        let columns = 1usize << sigma;
        let mut output = self.alloc(columns)?;
        if len == 0 {
            return output.to_host();
        }
        let reach = base
            .checked_add(len)
            .ok_or(CudaError::LengthMismatch {
                expected: usize::MAX,
                got: len,
            })?
            .div_ceil(columns);
        if table.len() < len || left.len() < reach {
            return Err(CudaError::LengthMismatch {
                expected: reach,
                got: left.len().min(table.len()),
            });
        }
        let device_left = self.upload(left)?;
        let base_arg = u64::try_from(base).map_err(|_| CudaError::LengthMismatch {
            expected: u32::MAX as usize,
            got: base,
        })?;
        let len_arg = u64::try_from(len).map_err(|_| CudaError::LengthMismatch {
            expected: u32::MAX as usize,
            got: len,
        })?;
        let sigma_arg = u32::try_from(sigma).map_err(|_| CudaError::LengthMismatch {
            expected: u32::MAX as usize,
            got: sigma,
        })?;
        let rows_arg = Self::count_of(left.len())?;
        let columns_arg = Self::count_of(columns)?;
        let mut builder = self.stream().launch_builder(self.msm_fold_rows());
        let _ = builder.arg(table.limbs());
        let _ = builder.arg(device_left.limbs());
        let _ = builder.arg(&base_arg);
        let _ = builder.arg(&len_arg);
        let _ = builder.arg(&sigma_arg);
        let _ = builder.arg(&rows_arg);
        let _ = builder.arg(&columns_arg);
        let _ = builder.arg(output.limbs_mut());
        // SAFETY: thread `c < columns` reads `table[i]` for `i` stepping by
        // `columns` from its own residue class strictly below `len`, checked above
        // against `table`'s length, and `left[row]` for `row = (base + i) >> sigma`
        // only while `row < rows`, which is `left`'s own length. It writes only
        // `out[c]` of a `columns`-element fresh allocation distinct from both
        // inputs. Threads with `c >= columns` return before any access.
        let _ = unsafe { builder.launch(Self::launch_config(columns_arg)) }?;
        self.stream().synchronize()?;
        output.to_host()
    }

    pub(crate) fn one_hot_embed_device(
        &self,
        device_hot: &CudaSlice<u32>,
        cycles: usize,
        domain: usize,
    ) -> Result<DeviceFrVec, CudaError> {
        let mut output = self.alloc(domain)?;
        if cycles == 0 {
            return Ok(output);
        }
        if device_hot.len() < cycles {
            return Err(CudaError::LengthMismatch {
                expected: cycles,
                got: device_hot.len(),
            });
        }
        let count = Self::count_of(cycles)?;
        let cycles = u64::try_from(cycles).map_err(|_| CudaError::LengthMismatch {
            expected: u32::MAX as usize,
            got: count as usize,
        })?;
        let domain_arg = u64::try_from(domain).map_err(|_| CudaError::LengthMismatch {
            expected: u32::MAX as usize,
            got: domain,
        })?;
        let mut builder = self.stream().launch_builder(self.opening_one_hot_embed());
        let _ = builder.arg(device_hot);
        let _ = builder.arg(&cycles);
        let _ = builder.arg(&domain_arg);
        let _ = builder.arg(output.limbs_mut());
        // SAFETY: thread `t < cycles` reads only `hot[t]`, and writes at most one
        // element `out[address * cycles + t]` after checking that index against
        // `domain` — the length of the fresh zeroed allocation `out`. Each `t`
        // yields a distinct index, so no two threads write the same element, and
        // `out` aliases neither `hot` nor any input.
        let _ = unsafe { builder.launch(Self::launch_config(count)) }?;
        self.stream().synchronize()?;
        Ok(output)
    }

    pub(crate) fn one_hot_fold_window(
        &self,
        device_hot: &CudaSlice<u32>,
        cycles: usize,
        base: usize,
        len: usize,
        left: &[Fr],
        sigma: usize,
    ) -> Result<Vec<Fr>, CudaError> {
        let columns = 1usize << sigma;
        let mut output = self.alloc(columns)?;
        if cycles == 0 || len == 0 {
            return output.to_host();
        }
        if columns > cycles
            || base.checked_add(len).is_none_or(|end| end > cycles)
            || device_hot.len() < len
        {
            return Err(CudaError::LengthMismatch {
                expected: columns,
                got: cycles.min(device_hot.len()),
            });
        }
        let device_left = self.upload(left)?;
        let base = u64::try_from(base).map_err(|_| CudaError::LengthMismatch {
            expected: u32::MAX as usize,
            got: base,
        })?;
        let len_arg = u64::try_from(len).map_err(|_| CudaError::LengthMismatch {
            expected: u32::MAX as usize,
            got: len,
        })?;
        let cycles = u64::try_from(cycles).map_err(|_| CudaError::LengthMismatch {
            expected: u32::MAX as usize,
            got: cycles,
        })?;
        let columns_u64 = u64::try_from(columns).map_err(|_| CudaError::LengthMismatch {
            expected: u32::MAX as usize,
            got: columns,
        })?;
        let rows = u64::try_from(left.len()).map_err(|_| CudaError::LengthMismatch {
            expected: u32::MAX as usize,
            got: left.len(),
        })?;
        let sigma_arg = u32::try_from(sigma).map_err(|_| CudaError::LengthMismatch {
            expected: u32::MAX as usize,
            got: sigma,
        })?;
        let columns_arg = Self::count_of(columns)?;
        let mut builder = self.stream().launch_builder(self.opening_one_hot_fold());
        let _ = builder.arg(device_hot);
        let _ = builder.arg(device_left.limbs());
        let _ = builder.arg(&cycles);
        let _ = builder.arg(&base);
        let _ = builder.arg(&len_arg);
        let _ = builder.arg(&columns_u64);
        let _ = builder.arg(&sigma_arg);
        let _ = builder.arg(&rows);
        let _ = builder.arg(output.limbs_mut());
        // SAFETY: thread `c < columns` reads `hot[i]` for `i` stepping by
        // `columns` from its own residue class strictly below `len`, checked above
        // against `hot`'s length, and `left[row]` only after bounds-checking
        // `row < rows` against `left`'s own length. It writes only `out[c]` of a
        // `columns`-element fresh allocation distinct from both inputs. Threads
        // with `c >= columns` return before any access.
        let _ = unsafe { builder.launch(Self::launch_config(columns_arg)) }?;
        self.stream().synchronize()?;
        output.to_host()
    }
}

#[cfg(test)]
pub(crate) mod testing {
    use ark_bn254::{Fq, Fr as ArkFr, G1Affine, G1Projective};
    use ark_ec::{CurveGroup, PrimeGroup};
    use ark_ff::{BigInt, PrimeField};
    use jolt_field::Fr;
    use proptest::prelude::*;

    use super::{AffineLimbs, JacobianLimbs, FQ_LIMBS};

    pub fn fq_from_limbs(limbs: [u64; FQ_LIMBS]) -> Fq {
        Fq::new_unchecked(BigInt(limbs))
    }

    pub fn fq_limbs(value: Fq) -> [u64; FQ_LIMBS] {
        value.0 .0
    }

    pub fn ark_fr(value: Fr) -> ArkFr {
        ArkFr::new_unchecked(BigInt(value.inner_limbs().0))
    }

    pub fn affine_limbs(point: G1Affine) -> AffineLimbs {
        if point.infinity {
            return AffineLimbs::IDENTITY;
        }
        AffineLimbs {
            x: fq_limbs(point.x),
            y: fq_limbs(point.y),
            infinity: false,
        }
    }

    pub fn jacobian_limbs(point: G1Projective) -> JacobianLimbs {
        JacobianLimbs {
            x: fq_limbs(point.x),
            y: fq_limbs(point.y),
            z: fq_limbs(point.z),
        }
    }

    pub fn projective(point: JacobianLimbs) -> G1Projective {
        G1Projective::new_unchecked(
            fq_from_limbs(point.x),
            fq_from_limbs(point.y),
            fq_from_limbs(point.z),
        )
    }

    pub fn projectives(points: &[JacobianLimbs]) -> Vec<G1Projective> {
        points.iter().copied().map(projective).collect()
    }

    pub fn point(k: u64) -> G1Projective {
        G1Projective::generator() * ArkFr::from(k)
    }

    pub fn affine(k: u64) -> G1Affine {
        point(k).into_affine()
    }

    pub fn arb_fq() -> impl Strategy<Value = Fq> {
        any::<[u64; FQ_LIMBS]>().prop_map(|limbs| {
            let mut bytes = [0u8; 32];
            for (chunk, limb) in bytes.chunks_exact_mut(8).zip(limbs) {
                chunk.copy_from_slice(&limb.to_le_bytes());
            }
            Fq::from_le_bytes_mod_order(&bytes)
        })
    }

    pub fn arb_fr() -> impl Strategy<Value = Fr> {
        any::<[u64; FQ_LIMBS]>().prop_map(|limbs| {
            let mut bytes = [0u8; 32];
            for (chunk, limb) in bytes.chunks_exact_mut(8).zip(limbs) {
                chunk.copy_from_slice(&limb.to_le_bytes());
            }
            Fr::from_le_bytes_mod_order(&bytes)
        })
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test module: device operations fail loudly"
)]
mod tests {
    use ark_bn254::{Fr as ArkFr, G1Affine, G1Projective, G2Projective};
    use ark_ec::scalar_mul::glv::GLVConfig;
    use ark_ec::scalar_mul::variable_base::msm_i128;
    use ark_ec::{AdditiveGroup, AffineRepr, CurveGroup, PrimeGroup, VariableBaseMSM};
    use ark_ff::{Field, PrimeField, Zero};
    use jolt_crypto::ec::bn254::batch_addition::batch_g1_additions_multi_affine;
    use jolt_field::{Fr, FromPrimitiveInt};
    use proptest::collection::vec;
    use proptest::prelude::*;

    use super::testing::{
        affine, affine_limbs, arb_fq, arb_fr, ark_fr, fq_from_limbs, fq_limbs, jacobian_limbs,
        point, projective, projectives,
    };
    use super::{
        take_limbs, unflatten_jacobian, AffineLimbs, JacobianLimbs, FQ_LIMBS, GLV4_SCALAR_BITS,
    };
    use crate::cuda::common::context::{shared_context, CudaKernelContext};

    const MSM_ROWS: usize = 3;

    const MSM_ROW_LEN: usize = 8;

    const ONE_HOT_K: usize = 5;

    fn device() -> Option<&'static CudaKernelContext> {
        shared_context()
    }

    fn limbs(values: &[ark_bn254::Fq]) -> Vec<[u64; FQ_LIMBS]> {
        values.iter().copied().map(fq_limbs).collect()
    }

    fn arb_msm_scalar() -> impl Strategy<Value = i128> {
        (-(u64::MAX as i128))..=(u64::MAX as i128)
    }

    fn msm_bases() -> Vec<G1Affine> {
        (0..MSM_ROW_LEN).map(|i| affine(i as u64 + 3)).collect()
    }

    const CSR_GEOMETRIES: [(usize, usize, usize); 4] =
        [(16, 4, 5), (64, 16, 4), (4096, 2048, 8), (256, 256, 3)];

    fn csr_hot(cycles: usize, one_hot_k: usize, seed: u64) -> Vec<u32> {
        (0..cycles)
            .map(|cycle| {
                let mixed = (cycle as u64)
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    .wrapping_add(seed);
                if mixed.is_multiple_of(5) {
                    super::COLD
                } else {
                    1 + (mixed >> 17) as u32 % (one_hot_k as u32 - 1)
                }
            })
            .collect()
    }

    fn csr_segment_counts(hot: &[u32], one_hot_k: usize, chunk_len: usize) -> Vec<usize> {
        let chunk_count = hot.len() / chunk_len;
        let mut counts = vec![0usize; one_hot_k * chunk_count];
        for (column, &address) in hot.iter().enumerate() {
            if address != super::COLD {
                counts[address as usize * chunk_count + column / chunk_len] += 1;
            }
        }
        counts
    }

    #[test]
    fn fixture_csr_geometries_cover_both_segment_kernels() {
        let mut narrow = 0usize;
        let mut wide = 0usize;
        let mut empty_segments = 0usize;
        let mut multi_chunk = 0usize;
        for (index, &(cycles, chunk_len, one_hot_k)) in CSR_GEOMETRIES.iter().enumerate() {
            assert!(
                cycles.is_multiple_of(chunk_len) && one_hot_k > 1,
                "geometry {index}: the cycle count must tile into whole chunks over at least \
                 two addresses",
            );
            let hot = csr_hot(cycles, one_hot_k, 31 + index as u64);
            assert!(
                hot.contains(&super::COLD),
                "geometry {index}: no cold cycle, so a scatter that ignored the sentinel would \
                 pass",
            );
            let counts = csr_segment_counts(&hot, one_hot_k, chunk_len);
            let widest = counts.iter().copied().max().unwrap_or(0);
            if widest <= super::SMALL_SEGMENT_LIMIT {
                narrow += 1;
            } else {
                wide += 1;
            }
            if counts.contains(&0) {
                empty_segments += 1;
            }
            if cycles / chunk_len > 1 {
                multi_chunk += 1;
            }
        }
        assert!(
            narrow > 0 && wide > 0,
            "the geometries reach only one of the two segment-sum kernels ({narrow} narrow, \
             {wide} wide), so the widest-segment decision the device plan has to make is \
             untested",
        );
        assert!(
            empty_segments > 0,
            "no geometry leaves a segment empty, so an offsets/counts pair that mishandled a \
             zero-length window would pass",
        );
        assert!(
            multi_chunk > 0,
            "every geometry is a single chunk, so a key that dropped the chunk term would pass",
        );
    }

    fn one_hot_chunk() -> Vec<Option<usize>> {
        (0..MSM_ROW_LEN)
            .map(|column| {
                if column % 3 == 2 {
                    None
                } else {
                    Some((column * 5 + 1) % (ONE_HOT_K - 1))
                }
            })
            .collect()
    }

    fn columns_per_row(hot: &[Option<usize>]) -> Vec<Vec<usize>> {
        let mut rows = vec![Vec::new(); ONE_HOT_K];
        for (column, row) in hot.iter().copied().enumerate() {
            if let Some(row) = row {
                rows[row].push(column);
            }
        }
        rows
    }

    fn affine_pairs(count: usize, offset: u64) -> (Vec<G1Affine>, Vec<G1Affine>) {
        let left = (0..count)
            .map(|i| affine(offset + 2 * i as u64 + 1))
            .collect();
        let right = (0..count)
            .map(|i| affine(offset + 2 * i as u64 + 2))
            .collect();
        (left, right)
    }

    fn special_pairs() -> Vec<(G1Projective, G1Projective)> {
        let p = point(7);
        let q = point(11);
        vec![
            (p, q),
            (p, p),
            (p, -p),
            (p, G1Projective::zero()),
            (G1Projective::zero(), p),
            (G1Projective::zero(), G1Projective::zero()),
            (q, -q),
            (point(1), point(1)),
        ]
    }

    #[test]
    fn special_pairs_cover_every_addition_branch() {
        let pairs = special_pairs();
        assert!(
            pairs.iter().any(|&(a, b)| a == b && !a.is_zero()),
            "no pair has equal nonzero inputs, so a formula missing the doubling branch \
             would pass",
        );
        assert!(
            pairs
                .iter()
                .any(|&(a, b)| !a.is_zero() && (a + b).is_zero()),
            "no pair sums to infinity, so a formula that cannot produce the identity would pass",
        );
        assert!(
            pairs.iter().any(|&(a, _)| a.is_zero()),
            "no pair has the accumulator at infinity",
        );
        assert!(
            pairs.iter().any(|&(_, b)| b.is_zero()),
            "no pair has the addend at infinity",
        );
        assert!(
            pairs
                .iter()
                .any(|&(a, b)| !a.is_zero() && !b.is_zero() && a != b && !(a + b).is_zero()),
            "no pair is the ordinary generic case",
        );
    }

    #[test]
    fn fixture_msm_rows_and_one_hot_chunk_discriminate() {
        let bases = msm_bases();
        assert!(
            bases.iter().all(|base| !base.infinity),
            "an infinity base makes every scalar at that column irrelevant",
        );
        assert!(
            bases
                .iter()
                .enumerate()
                .all(|(i, base)| bases[i + 1..].iter().all(|other| other != base)),
            "repeated bases let a kernel that mixes up columns pass",
        );
        assert!(
            (MSM_ROWS * MSM_ROW_LEN) / bases.len() > 1,
            "the scalar vector spans one row, so a kernel that ignores the row stride \
             would pass",
        );

        let hot = one_hot_chunk();
        let rows = columns_per_row(&hot);
        assert!(
            hot.iter().any(Option::is_none),
            "no idle cycle, so a kernel that ignored the `None` case would pass",
        );
        assert!(
            rows.iter().any(|columns| columns.len() > 1),
            "no row accumulates two bases, so a broken segmented sum would pass",
        );
        assert!(
            rows.iter().any(Vec::is_empty),
            "every row is hit, so a kernel that never emits the identity would pass",
        );
    }

    #[test]
    fn batch_affine_pairs_respect_the_distinct_x_precondition() {
        let (left, right) = affine_pairs(16, 5);
        assert!(
            left.iter().zip(&right).all(|(a, b)| a.x != b.x),
            "a pair shares an x-coordinate, which makes the batch-inversion denominator zero \
             and yields a silently off-curve point",
        );
        assert!(
            left.iter().chain(&right).all(|point| !point.infinity),
            "an input is at infinity, which affine pair addition cannot represent",
        );
        assert!(
            special_pairs()
                .iter()
                .any(|&(a, b)| !a.is_zero() && !b.is_zero() && a == b),
            "the cases affine pair addition excludes (equal, inverse, infinity) must be covered \
             by the general Jacobian addition instead, and they are not",
        );
    }

    #[test]
    fn identity_limbs_read_back_as_the_group_identity() {
        assert!(JacobianLimbs::IDENTITY.is_identity());
        assert_eq!(projective(JacobianLimbs::IDENTITY), G1Projective::zero());
        assert_eq!(
            affine_limbs(<G1Affine as AffineRepr>::zero()),
            AffineLimbs::IDENTITY,
        );
    }

    #[test]
    fn g1_base_upload_round_trips() {
        let Some(context) = device() else {
            return;
        };
        let bases: Vec<AffineLimbs> = msm_bases()
            .into_iter()
            .map(affine_limbs)
            .chain(std::iter::once(affine_limbs(
                <G1Affine as AffineRepr>::zero(),
            )))
            .collect();
        let device_bases = context.upload_g1_bases(&bases).expect("upload bases");
        assert_eq!(device_bases.count(), bases.len());
        assert_eq!(device_bases.to_host().expect("download bases"), bases);
    }

    #[test]
    fn g1_add_matches_arkworks_at_special_cases() {
        let Some(context) = device() else {
            return;
        };
        let pairs = special_pairs();
        let left: Vec<JacobianLimbs> = pairs.iter().map(|&(a, _)| jacobian_limbs(a)).collect();
        let right: Vec<JacobianLimbs> = pairs.iter().map(|&(_, b)| jacobian_limbs(b)).collect();
        let expected: Vec<G1Projective> = pairs.iter().map(|&(a, b)| a + b).collect();
        let got = projectives(&context.g1_add(&left, &right).expect("device g1_add"));
        assert_eq!(got, expected);

        let right_affine: Vec<AffineLimbs> = pairs
            .iter()
            .map(|&(_, b)| affine_limbs(b.into_affine()))
            .collect();
        let got_affine = projectives(
            &context
                .g1_add_affine(&left, &right_affine)
                .expect("device g1_add_affine"),
        );
        assert_eq!(got_affine, expected);
    }

    proptest! {
        #[test]
        fn fq_add_matches_arkworks(left in vec(arb_fq(), 1..200)) {
            let Some(context) = device() else { return Ok(()); };
            let right: Vec<_> = left.iter().rev().copied().collect();
            let expected: Vec<_> = left.iter().zip(&right).map(|(&a, &b)| a + b).collect();
            let got: Vec<_> = context
                .fq_add(&limbs(&left), &limbs(&right))
                .expect("device fq_add")
                .into_iter()
                .map(fq_from_limbs)
                .collect();
            prop_assert_eq!(got, expected);
        }

        #[test]
        fn fq_sub_matches_arkworks(left in vec(arb_fq(), 1..200)) {
            let Some(context) = device() else { return Ok(()); };
            let right: Vec<_> = left.iter().rev().copied().collect();
            let expected: Vec<_> = left.iter().zip(&right).map(|(&a, &b)| a - b).collect();
            let got: Vec<_> = context
                .fq_sub(&limbs(&left), &limbs(&right))
                .expect("device fq_sub")
                .into_iter()
                .map(fq_from_limbs)
                .collect();
            prop_assert_eq!(got, expected);
        }

        #[test]
        fn fq_batch_inverse_matches_arkworks(values in vec(arb_fq(), 1..200)) {
            let Some(context) = device() else { return Ok(()); };
            let mut values = values;
            for (index, value) in values.iter_mut().enumerate() {
                if index % 7 == 0 {
                    *value = ark_bn254::Fq::zero();
                }
            }
            prop_assert!(
                values.iter().any(|value| value.is_zero()),
                "no zero in the input, so the pass-through-zero convention is untested",
            );
            let mut expected = values.clone();
            ark_ff::fields::batch_inversion(&mut expected);
            let got: Vec<_> = context
                .fq_batch_inverse(&limbs(&values))
                .expect("device fq_batch_inverse")
                .into_iter()
                .map(fq_from_limbs)
                .collect();
            prop_assert_eq!(got, expected);
        }

        #[test]
        fn fq_mul_matches_arkworks(left in vec(arb_fq(), 1..200)) {
            let Some(context) = device() else { return Ok(()); };
            let right: Vec<_> = left.iter().rev().copied().collect();
            let expected: Vec<_> = left.iter().zip(&right).map(|(&a, &b)| a * b).collect();
            let got: Vec<_> = context
                .fq_mul(&limbs(&left), &limbs(&right))
                .expect("device fq_mul")
                .into_iter()
                .map(fq_from_limbs)
                .collect();
            prop_assert_eq!(got, expected);
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(8))]

        #[test]
        fn g1_double_matches_arkworks(scalars in vec(1u64..1_000_000, 1..16)) {
            let Some(context) = device() else { return Ok(()); };
            let points: Vec<G1Projective> = scalars
                .iter()
                .copied()
                .map(point)
                .chain(std::iter::once(G1Projective::zero()))
                .collect();
            let expected: Vec<G1Projective> = points.iter().map(AdditiveGroup::double).collect();
            let inputs: Vec<JacobianLimbs> =
                points.iter().copied().map(jacobian_limbs).collect();
            let got = projectives(&context.g1_double(&inputs).expect("device g1_double"));
            prop_assert_eq!(got, expected);
        }

        #[test]
        fn g1_add_matches_arkworks(scalars in vec(1u64..1_000_000, 2..16)) {
            let Some(context) = device() else { return Ok(()); };
            let left: Vec<G1Projective> = scalars.iter().copied().map(point).collect();
            let right: Vec<G1Projective> = scalars.iter().rev().copied().map(point).collect();
            let expected: Vec<G1Projective> =
                left.iter().zip(&right).map(|(&a, &b)| a + b).collect();
            let got = projectives(
                &context
                    .g1_add(
                        &left.iter().copied().map(jacobian_limbs).collect::<Vec<_>>(),
                        &right.iter().copied().map(jacobian_limbs).collect::<Vec<_>>(),
                    )
                    .expect("device g1_add"),
            );
            prop_assert_eq!(got, expected);
        }

        #[test]
        fn g1_add_affine_matches_arkworks(scalars in vec(1u64..1_000_000, 2..16)) {
            let Some(context) = device() else { return Ok(()); };
            let left: Vec<G1Projective> = scalars.iter().copied().map(point).collect();
            let right: Vec<G1Affine> = scalars.iter().rev().copied().map(affine).collect();
            let expected: Vec<G1Projective> =
                left.iter().zip(&right).map(|(&a, &b)| a + b).collect();
            let got = projectives(
                &context
                    .g1_add_affine(
                        &left.iter().copied().map(jacobian_limbs).collect::<Vec<_>>(),
                        &right.iter().copied().map(affine_limbs).collect::<Vec<_>>(),
                    )
                    .expect("device g1_add_affine"),
            );
            prop_assert_eq!(got, expected);
        }

        #[test]
        fn g1_batch_add_affine_pairs_matches_arkworks(offset in 1u64..1_000) {
            let Some(context) = device() else { return Ok(()); };
            let (left, right) = affine_pairs(16, offset);
            let expected: Vec<G1Affine> = left
                .iter()
                .zip(&right)
                .map(|(&a, &b)| (G1Projective::from(a) + b).into_affine())
                .collect();
            let got: Vec<G1Affine> = context
                .g1_batch_add_affine_pairs(
                    &left.iter().copied().map(affine_limbs).collect::<Vec<_>>(),
                    &right.iter().copied().map(affine_limbs).collect::<Vec<_>>(),
                )
                .expect("device g1_batch_add_affine_pairs")
                .into_iter()
                .map(|point| {
                    G1Affine::new_unchecked(fq_from_limbs(point.x), fq_from_limbs(point.y))
                })
                .collect();
            prop_assert_eq!(got, expected);
        }

        #[test]
        fn msm_rows_fr_matches_arkworks(scalars in vec(arb_fr(), MSM_ROWS * MSM_ROW_LEN)) {
            let Some(context) = device() else { return Ok(()); };
            let bases = msm_bases();
            let ark_scalars: Vec<ArkFr> = scalars.iter().copied().map(ark_fr).collect();
            let expected: Vec<G1Projective> = ark_scalars
                .chunks(MSM_ROW_LEN)
                .map(|row| G1Projective::msm(&bases, row).expect("arkworks msm"))
                .collect();
            let device_bases = context
                .upload_g1_bases(&bases.iter().copied().map(affine_limbs).collect::<Vec<_>>())
                .expect("upload bases");
            let device_rows = context.upload(&scalars).expect("upload scalars");
            let got = projectives(
                &context
                    .msm_rows_fr(&device_bases, &device_rows, MSM_ROW_LEN)
                    .expect("device msm_rows_fr"),
            );
            prop_assert_eq!(got, expected);
        }

        #[test]
        fn msm_rows_i128_matches_arkworks(
            scalars in vec(arb_msm_scalar(), MSM_ROWS * MSM_ROW_LEN),
        ) {
            let Some(context) = device() else { return Ok(()); };
            let bases = msm_bases();
            let expected: Vec<G1Projective> = scalars
                .chunks(MSM_ROW_LEN)
                .map(|row| msm_i128::<G1Projective>(&bases, row, true))
                .collect();
            let definition: Vec<G1Projective> = scalars
                .chunks(MSM_ROW_LEN)
                .map(|row| {
                    row.iter()
                        .zip(&bases)
                        .map(|(&scalar, &base)| base * ark_fr(Fr::from_i128(scalar)))
                        .sum()
                })
                .collect();
            prop_assert_eq!(
                &expected,
                &definition,
                "the arkworks signed MSM disagrees with the sum-of-scalar-multiples it is \
                 supposed to compute, so it cannot serve as the oracle",
            );
            let device_bases = context
                .upload_g1_bases(&bases.iter().copied().map(affine_limbs).collect::<Vec<_>>())
                .expect("upload bases");
            let got = projectives(
                &context
                    .msm_rows_i128(&device_bases, &scalars, MSM_ROW_LEN)
                    .expect("device msm_rows_i128"),
            );
            prop_assert_eq!(got, expected);
        }

        #[test]
        fn one_hot_rows_matches_chunk_sums(offset in 1u64..1_000) {
            let Some(context) = device() else { return Ok(()); };
            let chunk_len = 4usize;
            let bases: Vec<G1Affine> =
                (0..chunk_len).map(|i| affine(i as u64 + offset)).collect();
            let hot = one_hot_chunk();
            let device_bases = context
                .upload_g1_bases(&bases.iter().copied().map(affine_limbs).collect::<Vec<_>>())
                .expect("upload bases");
            let expected: Vec<G1Projective> = {
                let chunks: Vec<Vec<JacobianLimbs>> = hot
                    .chunks(chunk_len)
                    .map(|chunk| {
                        context
                            .one_hot_chunk_sums(&device_bases, chunk, ONE_HOT_K)
                            .expect("device one_hot_chunk_sums")
                    })
                    .collect();
                (0..ONE_HOT_K)
                    .flat_map(|address| {
                        chunks
                            .iter()
                            .map(move |chunk| projective(chunk[address]))
                            .collect::<Vec<_>>()
                    })
                    .collect()
            };
            let words: Vec<u32> = hot
                .iter()
                .map(|row| row.map_or(super::COLD, |address| address as u32))
                .collect();
            let got = projectives(
                &context
                    .one_hot_rows(&device_bases, &words, ONE_HOT_K, chunk_len)
                    .expect("device one_hot_rows"),
            );
            prop_assert_eq!(
                got,
                expected,
                "the batched row form must equal the per-chunk form transposed"
            );
        }

        #[test]
        fn one_hot_rows_device_matches_the_host_csr(seed in any::<u64>()) {
            let Some(context) = device() else { return Ok(()); };
            for (index, &(cycles, chunk_len, one_hot_k)) in CSR_GEOMETRIES.iter().enumerate() {
                let hot = csr_hot(cycles, one_hot_k, seed ^ index as u64);
                let bases: Vec<G1Affine> = (0..chunk_len)
                    .map(|column| affine(column as u64 + 1))
                    .collect();
                let device_bases = context
                    .upload_g1_bases(&bases.iter().copied().map(affine_limbs).collect::<Vec<_>>())
                    .expect("upload bases");
                let device_hot = context
                    .upload_u32_slice(&hot)
                    .expect("upload the hot column");

                let expected = projectives(
                    &context
                        .one_hot_rows(&device_bases, &hot, one_hot_k, chunk_len)
                        .expect("host one_hot_rows"),
                );
                let got = projectives(
                    &context
                        .one_hot_rows_device(
                            &device_bases,
                            &device_hot.slice(..),
                            cycles,
                            one_hot_k,
                            chunk_len,
                        )
                        .expect("device one_hot_rows"),
                );
                prop_assert_eq!(
                    got,
                    expected,
                    "geometry {} (cycles {}, chunk_len {}, one_hot_k {}): the device segment \
                     plan diverged from the host one",
                    index,
                    cycles,
                    chunk_len,
                    one_hot_k
                );
            }
        }

        #[test]
        fn one_hot_rows_device_rejects_an_address_past_the_chunk(
            cycle in prop::sample::select(vec![0usize, 7, 63]),
        ) {
            let Some(context) = device() else { return Ok(()); };
            let (cycles, chunk_len, one_hot_k) = (64usize, 16usize, 4usize);
            let mut hot = csr_hot(cycles, one_hot_k, 5);
            hot[cycle] = one_hot_k as u32;
            let bases: Vec<G1Affine> = (0..chunk_len)
                .map(|column| affine(column as u64 + 1))
                .collect();
            let device_bases = context
                .upload_g1_bases(&bases.iter().copied().map(affine_limbs).collect::<Vec<_>>())
                .expect("upload bases");
            let device_hot = context
                .upload_u32_slice(&hot)
                .expect("upload the hot column");
            prop_assert!(
                context
                    .one_hot_rows(&device_bases, &hot, one_hot_k, chunk_len)
                    .is_err(),
                "the host CSR build accepted an address past the chunk, so it cannot say what \
                 the device one should do"
            );
            prop_assert!(
                context
                    .one_hot_rows_device(&device_bases, &device_hot.slice(..), cycles, one_hot_k, chunk_len)
                    .is_err(),
                "the device CSR build accepted an address past the chunk: the scatter drops such \
                 a cycle, so the commitment would silently omit it"
            );
        }

        #[test]
        fn one_hot_chunk_sums_matches_arkworks(offset in 1u64..1_000) {
            let Some(context) = device() else { return Ok(()); };
            let bases: Vec<G1Affine> =
                (0..MSM_ROW_LEN).map(|i| affine(i as u64 + offset)).collect();
            let hot = one_hot_chunk();
            let rows = columns_per_row(&hot);
            let expected: Vec<G1Projective> = batch_g1_additions_multi_affine(&bases, &rows)
                .into_iter()
                .zip(&rows)
                .map(|(sum, columns)| {
                    if columns.is_empty() {
                        G1Projective::zero()
                    } else {
                        sum.into()
                    }
                })
                .collect();
            let device_bases = context
                .upload_g1_bases(&bases.iter().copied().map(affine_limbs).collect::<Vec<_>>())
                .expect("upload bases");
            let got = projectives(
                &context
                    .one_hot_chunk_sums(&device_bases, &hot, ONE_HOT_K)
                    .expect("device one_hot_chunk_sums"),
            );
            prop_assert_eq!(got, expected);
        }
    }

    const RESIDENT_MSM_LENS: [usize; 6] = [1, 2, 255, 256, 1024, 4096];

    const GLV_LENS: [usize; 5] = [1, 2, 257, 1024, 4096];

    const FROBENIUS_LENS: [usize; 4] = [1, 2, 257, 1024];

    fn glv_lambda() -> ArkFr {
        <ark_bn254::g1::Config as GLVConfig>::LAMBDA
    }

    fn glv_bound() -> ArkFr {
        ArkFr::from(u128::MAX) + ArkFr::ONE
    }

    fn limbs_to_ark(limbs: &[u64]) -> ArkFr {
        ArkFr::from_bigint(ark_ff::BigInt(take_limbs(limbs))).expect("canonical limbs below Fr")
    }

    #[test]
    fn g1_endomorphism_span_matches_arkworks() {
        let Some(context) = device() else {
            return;
        };
        let lambda = glv_lambda();
        for count in GLV_LENS {
            for offset in [0usize, 3] {
                let lead = jacobian_span(offset, 7);
                let points = jacobian_span(count, 13);
                let mut flat = flat_jacobian(&lead);
                flat.extend_from_slice(&flat_jacobian(&points));
                let device_points = context.upload_raw_u64(&flat).expect("upload span");

                let raw = context
                    .g1_endomorphism_span(&device_points, offset, count)
                    .expect("endomorphism");
                let got = projectives(&unflatten_jacobian(
                    &context.download_u64(&raw).expect("download endomorphism"),
                ));

                assert_eq!(
                    got.len(),
                    2 * count,
                    "endomorphism should emit the original span then the mapped span"
                );
                let divergence = (0..count).position(|index| {
                    got[index].into_affine() != points[index].into_affine()
                        || got[count + index].into_affine()
                            != (points[index] * lambda).into_affine()
                });
                assert_eq!(
                    divergence, None,
                    "endomorphism diverged at count {count}, offset {offset}"
                );
            }
        }
    }

    #[test]
    fn glv_decompose_2d_matches_arkworks() {
        let Some(context) = device() else {
            return;
        };
        let lambda = glv_lambda();
        let bound = glv_bound();
        for count in GLV_LENS {
            let scalars: Vec<Fr> = (0..count)
                .map(|index| Fr::from_u64(index as u64 * 1_000_003 + 7))
                .collect();
            let uploaded = context.upload(&scalars).expect("upload scalars");
            let canonical = context
                .canonical_scalars(&uploaded)
                .expect("canonical scalars");
            let (split, signs) = context
                .glv_decompose_2d(&canonical, count)
                .expect("glv decompose");
            let raw = context.download_u64(&split).expect("download split");

            assert_eq!(signs.len(), 2 * count, "sign count at {count}");
            for index in 0..count {
                let first = limbs_to_ark(&raw[index * FQ_LIMBS..(index + 1) * FQ_LIMBS]);
                let second =
                    limbs_to_ark(&raw[(count + index) * FQ_LIMBS..(count + index + 1) * FQ_LIMBS]);
                let signed = |value: ArkFr, sign: u8| if sign == 0 { value } else { -value };
                let rebuilt =
                    signed(first, signs[index]) + signed(second, signs[count + index]) * lambda;
                assert_eq!(
                    rebuilt,
                    ark_fr(scalars[index]),
                    "glv split failed to rebuild scalar {index} at count {count}"
                );
                assert!(
                    first < bound && second < bound,
                    "glv component exceeded 2^128 at scalar {index}, count {count}"
                );
            }
        }
    }

    fn psi_eigenvalue() -> ArkFr {
        ArkFr::from_le_bytes_mod_order(&ark_ff::BigInteger::to_bytes_le(&ark_bn254::Fq::MODULUS))
    }

    #[test]
    fn g2_frobenius_span_matches_arkworks() {
        let Some(context) = device() else {
            return;
        };
        let lambda = psi_eigenvalue();
        for count in FROBENIUS_LENS {
            for offset in [0usize, 3] {
                let lead = g2_span(offset, 17);
                let points = g2_span(count, 23);
                let mut flat = flat_g2(&lead);
                flat.extend_from_slice(&flat_g2(&points));
                let device_points = context.upload_raw_u64(&flat).expect("upload span");

                let raw = context
                    .g2_frobenius_span(&device_points, offset, count)
                    .expect("frobenius");
                let hosted = context.download_u64(&raw).expect("download frobenius");
                assert_eq!(
                    hosted.len(),
                    4 * count * 24,
                    "frobenius should emit four spans at count {count}"
                );

                let mut power = ArkFr::ONE;
                for step in 0..4usize {
                    let got: Vec<G2Projective> = (0..count)
                        .map(|index| {
                            let slot = (step * count + index) * 24;
                            g2_from_limbs(&hosted[slot..slot + 24])
                        })
                        .collect();
                    let want: Vec<G2Projective> =
                        (0..count).map(|index| points[index] * power).collect();
                    let divergence = G2Projective::normalize_batch(&got)
                        .into_iter()
                        .zip(G2Projective::normalize_batch(&want))
                        .position(|(got, want)| got != want);
                    assert_eq!(
                        divergence, None,
                        "psi^{step} diverged at count {count}, offset {offset}"
                    );
                    power *= lambda;
                }
            }
        }
    }

    #[test]
    fn glv_decompose_4d_matches_arkworks() {
        let Some(context) = device() else {
            return;
        };
        let lambda = psi_eigenvalue();
        let bound = ArkFr::from(1u128 << (GLV4_SCALAR_BITS - 1));
        for count in GLV_LENS {
            let scalars: Vec<Fr> = (0..count)
                .map(|index| Fr::from_u64(index as u64 * 7_654_321 + 13))
                .collect();
            let uploaded = context.upload(&scalars).expect("upload scalars");
            let canonical = context
                .canonical_scalars(&uploaded)
                .expect("canonical scalars");
            let (split, signs) = context
                .glv_decompose_4d(&canonical, count)
                .expect("glv decompose 4d");
            let raw = context.download_u64(&split).expect("download split");

            assert_eq!(signs.len(), 4 * count, "sign count at {count}");
            for index in 0..count {
                let mut rebuilt = ArkFr::ZERO;
                let mut power = ArkFr::ONE;
                for step in 0..4usize {
                    let slot = (step * count + index) * FQ_LIMBS;
                    let magnitude = limbs_to_ark(&raw[slot..slot + FQ_LIMBS]);
                    assert!(
                        magnitude < bound,
                        "4d component {step} exceeded the GLV4_SCALAR_BITS bound at scalar {index}, count {count}"
                    );
                    if signs[step * count + index] == 0 {
                        rebuilt += magnitude * power;
                    } else {
                        rebuilt -= magnitude * power;
                    }
                    power *= lambda;
                }
                assert_eq!(
                    rebuilt,
                    ark_fr(scalars[index]),
                    "4d split failed to rebuild scalar {index} at count {count}"
                );
            }
        }
    }

    fn g2_span(count: usize, seed: u64) -> Vec<G2Projective> {
        let step = G2Projective::generator();
        let mut walk = step * ArkFr::from(seed + 1);
        (0..count)
            .map(|index| {
                walk += step;
                if (index as u64 + seed).is_multiple_of(41) {
                    G2Projective::zero()
                } else {
                    walk
                }
            })
            .collect()
    }

    fn flat_g2(points: &[G2Projective]) -> Vec<u64> {
        let mut flat = Vec::with_capacity(points.len() * 24);
        for value in points {
            for coordinate in [&value.x, &value.y, &value.z] {
                flat.extend_from_slice(&coordinate.c0.0 .0);
                flat.extend_from_slice(&coordinate.c1.0 .0);
            }
        }
        flat
    }

    fn g2_from_limbs(limbs: &[u64]) -> G2Projective {
        let fq2 = |chunk: &[u64]| {
            ark_bn254::Fq2::new(
                fq_from_limbs(take_limbs(&chunk[..FQ_LIMBS])),
                fq_from_limbs(take_limbs(&chunk[FQ_LIMBS..])),
            )
        };
        G2Projective::new_unchecked(fq2(&limbs[..8]), fq2(&limbs[8..16]), fq2(&limbs[16..24]))
    }

    #[test]
    fn resident_g2_msm_matches_arkworks() {
        let Some(context) = device() else {
            return;
        };
        for count in RESIDENT_MSM_LENS {
            let points = g2_span(count, 5);
            let scalars: Vec<Fr> = (0..count)
                .map(|index| {
                    if index.is_multiple_of(17) {
                        Fr::from_u64(0)
                    } else {
                        Fr::from_u64(index as u64 * 1_000_003 + 7)
                    }
                })
                .collect();
            let expected = points
                .iter()
                .zip(&scalars)
                .map(|(base, scalar)| *base * ark_fr(*scalar))
                .sum::<G2Projective>();

            let out_offset = count + 2;
            let mut flat = flat_g2(&points);
            flat.extend(std::iter::repeat_n(0u64, (out_offset + 1 - count) * 24));
            let mut buffer = context.upload_raw_u64(&flat).expect("upload arena");
            context
                .g2_msm_in_place(&mut buffer, 0, out_offset, count, &scalars)
                .expect("resident g2 msm");
            let raw = context.download_u64(&buffer).expect("download arena");
            let slot = out_offset * 24;
            let got = g2_from_limbs(raw.get(slot..slot + 24).expect("output slot"));

            assert_eq!(
                got.into_affine(),
                expected.into_affine(),
                "resident g2 msm diverged at count {count}"
            );
        }
    }

    fn jacobian_span(count: usize, seed: u64) -> Vec<G1Projective> {
        (0..count)
            .map(|index| {
                if (index as u64 + seed).is_multiple_of(41) {
                    G1Projective::zero()
                } else {
                    point(index as u64 + seed + 3)
                }
            })
            .collect()
    }

    fn flat_jacobian(points: &[G1Projective]) -> Vec<u64> {
        let mut flat = Vec::with_capacity(points.len() * 3 * FQ_LIMBS);
        for value in points {
            let limbs = jacobian_limbs(*value);
            flat.extend_from_slice(&limbs.x);
            flat.extend_from_slice(&limbs.y);
            flat.extend_from_slice(&limbs.z);
        }
        flat
    }

    #[test]
    fn normalise_g1_span_matches_arkworks() {
        let Some(context) = device() else {
            return;
        };
        for count in RESIDENT_MSM_LENS {
            for offset in [0usize, 3] {
                let lead = jacobian_span(offset, 7);
                let points = jacobian_span(count, 11);
                let mut flat = flat_jacobian(&lead);
                flat.extend_from_slice(&flat_jacobian(&points));
                let device_points = context.upload_raw_u64(&flat).expect("upload span");

                let expected: Vec<AffineLimbs> = points
                    .iter()
                    .map(|value| affine_limbs(value.into_affine()))
                    .collect();
                let got = context
                    .normalise_g1_span(&device_points, offset, count)
                    .expect("normalise span")
                    .to_host()
                    .expect("read affine bases");

                assert_eq!(
                    got.len(),
                    expected.len(),
                    "base count diverged at count {count}, offset {offset}"
                );
                let divergence = got
                    .iter()
                    .zip(&expected)
                    .position(|(got, expected)| got != expected);
                assert_eq!(
                    divergence, None,
                    "normalised span diverged at count {count}, offset {offset}"
                );
            }
        }
    }

    #[test]
    fn resident_g1_msm_matches_arkworks() {
        let Some(context) = device() else {
            return;
        };
        for count in RESIDENT_MSM_LENS {
            let points = jacobian_span(count, 5);
            let scalars: Vec<Fr> = (0..count)
                .map(|index| {
                    if index.is_multiple_of(17) {
                        Fr::from_u64(0)
                    } else {
                        Fr::from_u64(index as u64 * 1_000_003 + 7)
                    }
                })
                .collect();
            let expected = points
                .iter()
                .zip(&scalars)
                .map(|(base, scalar)| *base * ark_fr(*scalar))
                .sum::<G1Projective>();

            let out_offset = count + 2;
            let mut flat = flat_jacobian(&points);
            flat.extend(std::iter::repeat_n(
                0u64,
                (out_offset + 1 - count) * 3 * FQ_LIMBS,
            ));
            let mut buffer = context.upload_raw_u64(&flat).expect("upload arena");
            context
                .g1_msm_in_place(&mut buffer, 0, out_offset, count, &scalars)
                .expect("resident g1 msm");
            let raw = context.download_u64(&buffer).expect("download arena");
            let slot = out_offset * 3 * FQ_LIMBS;
            let got = projectives(&unflatten_jacobian(
                raw.get(slot..slot + 3 * FQ_LIMBS).expect("output slot"),
            ));

            assert_eq!(
                got.first().map(|point| point.into_affine()),
                Some(expected.into_affine()),
                "resident g1 msm diverged at count {count}"
            );
        }
    }

    const FOLD_SHAPES: [(usize, usize, usize); 5] =
        [(1, 26, 10), (1, 32, 8), (3, 22, 12), (5, 1, 16), (2, 85, 3)];

    #[test]
    fn window_fold_matches_arkworks() {
        let Some(context) = device() else {
            return;
        };
        for (rows, windows, window_bits) in FOLD_SHAPES {
            let mut points = Vec::with_capacity(rows * windows);
            for window in 0..windows {
                for row in 0..rows {
                    if (window + row) % 11 == 0 {
                        points.push(G1Projective::zero());
                    } else {
                        points.push(point((window * rows + row) as u64 + 5));
                    }
                }
            }
            let expected: Vec<G1Projective> = (0..rows)
                .map(|row| {
                    (0..windows)
                        .map(|window| {
                            let mut shift = ArkFr::from(1u64);
                            for _ in 0..window * window_bits {
                                shift = shift.double();
                            }
                            points[window * rows + row] * shift
                        })
                        .sum()
                })
                .collect();

            let mut flat = Vec::with_capacity(points.len() * 3 * FQ_LIMBS);
            for value in &points {
                let limbs = jacobian_limbs(*value);
                flat.extend_from_slice(&limbs.x);
                flat.extend_from_slice(&limbs.y);
                flat.extend_from_slice(&limbs.z);
            }
            let device_points = context.upload_raw_u64(&flat).expect("upload window points");
            let mut device_out = context
                .alloc_u64(rows * 3 * FQ_LIMBS)
                .expect("allocate fold output");
            context
                .launch_window_fold(&device_points, rows, windows, window_bits, &mut device_out)
                .expect("window fold");
            let got = projectives(&unflatten_jacobian(
                &context.download_u64(&device_out).expect("download fold"),
            ));

            assert_eq!(
                got, expected,
                "window fold diverged at rows {rows}, windows {windows}, window_bits {window_bits}"
            );
        }
    }

    const REDUCE_SHAPES: [(usize, usize, usize); 6] = [
        (1, 1024, 8),
        (1, 4096, 32),
        (3, 256, 4),
        (1, 128, 1),
        (2, 100, 7),
        (1, 1, 1),
    ];

    #[test]
    fn bucket_reduce_chunked_matches_arkworks() {
        let Some(context) = device() else {
            return;
        };
        for (rows, buckets, chunks) in REDUCE_SHAPES {
            let mut points = Vec::with_capacity(rows * buckets);
            for row in 0..rows {
                for bucket in 0..buckets {
                    if (row + bucket) % 23 == 0 {
                        points.push(G1Projective::zero());
                    } else {
                        points.push(point((row * buckets + bucket) as u64 + 7));
                    }
                }
            }
            let expected: Vec<G1Projective> = (0..rows)
                .map(|row| {
                    (1..buckets)
                        .map(|bucket| points[row * buckets + bucket] * ArkFr::from(bucket as u64))
                        .sum()
                })
                .collect();

            let mut flat = Vec::with_capacity(points.len() * 3 * FQ_LIMBS);
            for value in &points {
                let limbs = jacobian_limbs(*value);
                flat.extend_from_slice(&limbs.x);
                flat.extend_from_slice(&limbs.y);
                flat.extend_from_slice(&limbs.z);
            }
            let device_points = context.upload_raw_u64(&flat).expect("upload buckets");
            let mut device_out = context
                .alloc_u64(rows * 3 * FQ_LIMBS)
                .expect("allocate reduce output");
            context
                .launch_bucket_reduce_chunked(
                    &device_points,
                    rows,
                    buckets,
                    chunks,
                    &mut device_out,
                )
                .expect("chunked bucket reduce");
            let got = projectives(&unflatten_jacobian(
                &context.download_u64(&device_out).expect("download reduce"),
            ));

            assert_eq!(
                got, expected,
                "chunked bucket reduce diverged at rows {rows}, buckets {buckets}, chunks {chunks}"
            );
        }
    }
}
