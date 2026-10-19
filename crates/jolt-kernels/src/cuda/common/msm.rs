//! BN254 base-field and G1 arithmetic on the device.
//!
//! Row MSMs use Pippenger buckets built by device count/scan/scatter passes.
//! G1 window batches use a memory budget informed by free device memory and
//! fall back to sequential windows when the batch does not fit.
//! Signed increments retain separate native magnitudes and signs.
//!
//! Inputs and outputs use four Montgomery limbs per field element. Each GPU
//! thread inverts a chunk using its output range for prefix products; zero
//! entries remain zero.
//!
//! G1 doubling and addition keep Jacobian coordinates; mixed addition consumes
//! affine addends with all-zero device coordinates representing infinity.
//! Batched affine-pair addition instead inverts x-coordinate differences.
//! Callers must supply finite points with distinct x-coordinates in each pair;
//! this precondition is not checked by the wrapper or kernels.
//!
//! Affine base tables remain on their owning stream. Normalization extracts
//! Jacobian z-coordinates, batch-inverts them on the device, and converts the
//! selected span without downloading its coordinates.
//!
//! Segmented sums encode negation in the high bit of each base index. The host
//! checks index ranges and selects thread, warp, block, or size-class reductions.
//! One-hot chunks use Jacobian accumulators on CUDA; CPU streaming commitments
//! use batched affine addition.
//! Device one-hot rows count and scatter base indices with flat atomics or
//! shared-memory tallies; the host scans the histogram and selects segment classes.
//! Fully covered chunks can reconstruct a dominant bucket from the total base sum
//! minus the other buckets. Chunks containing cold cycles use direct segment sums.
//!
//! Weighted bucket sums split across blocks and merge their partials per row.
//! Window folding combines each row's results from the most significant window
//! down, with repeated doubling between windows.

use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, CudaView, LaunchConfig, PushKernelArg};

use super::context::{CudaKernelContext, BLOCK};
use super::device::DeviceFrVec;
use super::error::CudaError;
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

pub struct DeviceSegments {
    indices: CudaSlice<u32>,
    offsets: CudaSlice<u32>,
    counts: CudaSlice<u32>,
    segments: usize,
    widest: usize,
    classes: DeviceClasses,
}

impl DeviceSegments {
    pub const fn widest(&self) -> usize {
        self.widest
    }
}

pub(crate) struct DeviceClasses {
    ids: CudaSlice<u32>,
    fat: usize,
    thin: usize,
    fat_entries: usize,
    live_thin: usize,
}

pub(crate) struct HostClasses {
    ids: Vec<u32>,
    fat: usize,
    fat_entries: usize,
    live_thin: usize,
}

pub(crate) fn plan_classes(counts: &[u32]) -> HostClasses {
    let total: u64 = counts.iter().map(|&count| u64::from(count)).sum();
    let mean = total / counts.len().max(1) as u64;
    let limit = u32::try_from(mean.saturating_mul(2).max(CLASS_FLOOR as u64)).unwrap_or(u32::MAX);
    let mut ids = Vec::with_capacity(counts.len());
    let mut fat_entries = 0usize;
    for (segment, &count) in counts.iter().enumerate().filter(|(_, &c)| c > limit) {
        ids.push(u32::try_from(segment).unwrap_or(u32::MAX));
        fat_entries += count as usize;
    }
    let fat = ids.len();
    let mut live_thin = 0usize;
    for (segment, &count) in counts.iter().enumerate().filter(|(_, &c)| c <= limit) {
        ids.push(u32::try_from(segment).unwrap_or(u32::MAX));
        live_thin += usize::from(count > 0);
    }
    HostClasses {
        ids,
        fat,
        fat_entries,
        live_thin,
    }
}

struct OneHotComplements {
    counts: CudaSlice<u32>,
    targets: CudaSlice<u32>,
}

pub struct SignedColumn {
    magnitudes: CudaSlice<u64>,
    signs: CudaSlice<u8>,
    len: usize,
}

pub struct DeviceG1Bases {
    stream: Arc<CudaStream>,
    limbs: CudaSlice<u64>,
    count: usize,
}

impl DeviceG1Bases {
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

const MAX_BASE_INDEX: usize = 0x7fff_ffff;

const SMALL_SEGMENT_LIMIT: usize = 64;

const WARP_LANES: u32 = 32;

const WARP_WIDEST_LIMIT: usize = 4_096;

const WARP_SEGMENT_FLOOR: usize = 2_048;

const CLASS_FLOOR: usize = 1_024;

const FR_SCALAR_BITS: usize = 254;

const MAX_WINDOW_BITS: usize = 16;

const TARGET_REDUCE_BLOCKS: usize = 1024;

const BATCHED_WINDOW_BYTES: usize = 64 << 20;

const MAX_G1_BATCHED_WINDOW_BYTES: usize = 4 << 30;

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
    mode: SegmentMode,
    classes: Option<&'a DeviceClasses>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SegmentMode {
    Auto,
    Small,
    Heavy,
    Warp,
    Classes,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OneHotMode {
    Auto,
    Flat,
    Shared,
}

const ONE_HOT_SHARED_SLOTS: usize = 4_096;

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
        kernel: &CudaFunction,
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
        kernel: &CudaFunction,
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

    pub fn msm_rows_fr(
        &self,
        bases: &DeviceG1Bases,
        rows: &DeviceFrVec,
        row_len: usize,
    ) -> Result<Vec<JacobianLimbs>, CudaError> {
        let canonical = self.canonical_scalars(rows)?;
        let signs = self.alloc_u8(rows.len())?;
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

    pub fn signed_column(&self, rows: &[i128]) -> Result<SignedColumn, CudaError> {
        tracing::info_span!("cuda_msm_i128_stage", len = rows.len()).in_scope(|| {
            let mut magnitudes = vec![0u64; rows.len()];
            let mut signs = vec![0u8; rows.len()];
            if !split_signed_magnitudes(rows, &mut magnitudes, &mut signs) {
                return Err(CudaError::InvariantViolation {
                    reason: "signed MSM scalars must fit in [-u64::MAX, u64::MAX]",
                });
            }
            Ok(SignedColumn {
                magnitudes: self.upload_u64_slice(&magnitudes)?,
                signs: self.upload_u8_slice(&signs)?,
                len: rows.len(),
            })
        })
    }

    pub fn msm_rows_signed(
        &self,
        bases: &DeviceG1Bases,
        column: &SignedColumn,
        row_len: usize,
    ) -> Result<Vec<JacobianLimbs>, CudaError> {
        self.require_owned(bases.ordinal())?;
        let accumulator = self.pippenger_device(
            bases,
            DeviceScalars {
                values: &column.magnitudes,
                limbs: 1,
            },
            &column.signs,
            column.len,
            row_len,
            64,
        )?;
        tracing::info_span!("cuda_msm_i128_download", len = column.len)
            .in_scope(|| Ok(unflatten_jacobian(&self.download_u64(&accumulator)?)))
    }

    pub fn msm_rows_i128(
        &self,
        bases: &DeviceG1Bases,
        rows: &[i128],
        row_len: usize,
    ) -> Result<Vec<JacobianLimbs>, CudaError> {
        self.msm_rows_signed(bases, &self.signed_column(rows)?, row_len)
    }

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
        let plan = self.upload_segments(bases, &indices, &counts)?;
        self.segment_sums(bases, &plan, SegmentMode::Auto)
    }

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
        let (indices, counts) =
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
                    Ok((indices, counts))
                },
            )?;
        tracing::info_span!("cuda_commit_one_hot_segments", segments).in_scope(|| {
            let plan = self.upload_segments(bases, &indices, &counts)?;
            self.segment_sums(bases, &plan, SegmentMode::Auto)
        })
    }

    fn one_hot_complement_targets(
        counts: &mut [u32],
        chunk_count: usize,
        chunk_len: usize,
    ) -> Result<Vec<u32>, CudaError> {
        if chunk_count == 0 {
            return Ok(Vec::new());
        }
        let limit = (chunk_len / 2)
            .max(CLASS_FLOOR)
            .max(counts.len() / chunk_count);
        if limit >= chunk_len {
            return Ok(Vec::new());
        }
        let mut covered = vec![0u64; chunk_count];
        let mut dominant = vec![None; chunk_count];
        for (segment, &count) in counts.iter().enumerate() {
            let chunk = segment % chunk_count;
            covered[chunk] += u64::from(count);
            if count as usize > limit {
                dominant[chunk] = Some(segment);
            }
        }
        let mut targets = Vec::new();
        for (chunk, segment) in dominant.into_iter().enumerate() {
            // A cold cycle would make the total base sum an invalid complement.
            if covered[chunk] == chunk_len as u64 {
                if let Some(segment) = segment {
                    targets.push(Self::count_of(segment)?);
                    counts[segment] = 0;
                }
            }
        }
        Ok(targets)
    }

    pub(crate) const fn one_hot_slots(one_hot_k: usize, chunk_len: usize) -> usize {
        one_hot_k.saturating_mul((BLOCK as usize - 1) / chunk_len + 2)
    }

    pub(crate) const fn one_hot_shares(one_hot_k: usize, chunk_len: usize) -> bool {
        Self::one_hot_slots(one_hot_k, chunk_len) <= ONE_HOT_SHARED_SLOTS
    }

    fn one_hot_mode(
        mode: OneHotMode,
        one_hot_k: usize,
        chunk_len: usize,
    ) -> Result<OneHotMode, CudaError> {
        let shares = Self::one_hot_shares(one_hot_k, chunk_len);
        match mode {
            OneHotMode::Auto if shares => Ok(OneHotMode::Shared),
            OneHotMode::Auto | OneHotMode::Flat => Ok(OneHotMode::Flat),
            OneHotMode::Shared if shares => Ok(OneHotMode::Shared),
            OneHotMode::Shared => Err(CudaError::InvariantViolation {
                reason: "a one-hot histogram of this shape needs more per-block tally slots than \
                         shared memory holds",
            }),
        }
    }

    pub fn one_hot_rows_device(
        &self,
        bases: &DeviceG1Bases,
        hot: &CudaView<'_, u32>,
        cycles: usize,
        one_hot_k: usize,
        chunk_len: usize,
    ) -> Result<Vec<JacobianLimbs>, CudaError> {
        self.one_hot_rows_device_with(bases, hot, cycles, one_hot_k, chunk_len, OneHotMode::Auto)
    }

    pub fn one_hot_rows_device_with(
        &self,
        bases: &DeviceG1Bases,
        hot: &CudaView<'_, u32>,
        cycles: usize,
        one_hot_k: usize,
        chunk_len: usize,
        mode: OneHotMode,
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

        let mode = Self::one_hot_mode(mode, one_hot_k, chunk_len)?;

        let count_span =
            tracing::info_span!("cuda_commit_one_hot_count", segments, cycles, ?mode).entered();
        let slots = Self::one_hot_slots(one_hot_k, chunk_len);
        let slots_arg = Self::count_of(slots)?;
        let mut counts = self.alloc_u32(segments + 1)?;
        if mode == OneHotMode::Shared {
            let mut builder = self
                .stream()
                .launch_builder(self.msm_one_hot_count_shared());
            let _ = builder.arg(hot);
            let _ = builder.arg(&cycle_count);
            let _ = builder.arg(&chunk_len_arg);
            let _ = builder.arg(&chunk_count_arg);
            let _ = builder.arg(&one_hot_k_arg);
            let _ = builder.arg(&slots_arg);
            let _ = builder.arg(&mut counts);
            let mut config = Self::launch_config(cycle_count);
            config.shared_mem_bytes = Self::count_of(slots * size_of::<u32>())?;
            // SAFETY: the block is `BLOCK` wide, so thread `i < cycles` reads
            // only `hot[i]` of a buffer checked against `cycles` above. The
            // dynamic shared array holds exactly `slots` u32s, matching
            // `shared_mem_bytes`, and every index into it is
            // `(i / chunk_len - first) * one_hot_k + address` with
            // `address < one_hot_k` and `i` inside `[first * chunk_len,
            // first * chunk_len + BLOCK)`, so the chunk term is at most
            // `(BLOCK - 1) / chunk_len + 1` and the index stays below the
            // `one_hot_k * ((BLOCK - 1) / chunk_len + 2)` that `one_hot_slots`
            // allocates; the zeroing and merge loops stride the same range.
            // Both `__syncthreads()` sit outside every early return, so all
            // `BLOCK` threads reach them. The merge writes
            // `counts[address * chunk_count + chunk]` only for `chunk <
            // chunk_count`, and an out-of-range address hits the one trailing
            // slot, so every atomic lands inside the `segments + 1` u32s
            // `alloc_u32` zeroed.
            let _ = unsafe { builder.launch(config) }?;
        } else {
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
        }
        self.stream().synchronize()?;
        drop(count_span);

        let (offsets, classes, total, widest, complements) =
            tracing::info_span!("cuda_commit_one_hot_scan", segments).in_scope(|| {
                let mut histogram = self.download_u32(&counts)?;
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
                let targets = Self::one_hot_complement_targets(
                    &mut histogram[..segments],
                    chunk_count,
                    chunk_len,
                )?;
                let complements = if targets.is_empty() {
                    None
                } else {
                    widest = histogram[..segments].iter().copied().max().unwrap_or(0);
                    Some(OneHotComplements {
                        counts: self.upload_u32_slice(&histogram[..segments])?,
                        targets: self.upload_u32_slice(&targets)?,
                    })
                };
                let classes = self.upload_classes(&plan_classes(&histogram[..segments]))?;
                Ok((
                    self.upload_u32_slice(&offsets)?,
                    classes,
                    running as usize,
                    widest as usize,
                    complements,
                ))
            })?;

        let scatter_span =
            tracing::info_span!("cuda_commit_one_hot_scatter", segments, entries = total).entered();
        let mut cursor = self.clone_u32(&offsets)?;
        let mut indices = self.alloc_u32_unset(total.max(1))?;
        if mode == OneHotMode::Shared {
            let mut builder = self
                .stream()
                .launch_builder(self.msm_one_hot_scatter_shared());
            let _ = builder.arg(hot);
            let _ = builder.arg(&cycle_count);
            let _ = builder.arg(&chunk_len_arg);
            let _ = builder.arg(&chunk_count_arg);
            let _ = builder.arg(&one_hot_k_arg);
            let _ = builder.arg(&slots_arg);
            let _ = builder.arg(&mut cursor);
            let _ = builder.arg(&mut indices);
            let mut config = Self::launch_config(cycle_count);
            config.shared_mem_bytes = Self::count_of(2 * slots * size_of::<u32>())?;
            // SAFETY: the shared array holds `2 * slots` u32s, matching
            // `shared_mem_bytes`, split into a `slots`-element tally and a
            // `slots`-element reservation table; every index into either is the
            // same `(i / chunk_len - first) * one_hot_k + address` bounded by
            // `slots` as argued for the count kernel, or a strided sweep of
            // `0..slots`. All three `__syncthreads()` sit outside every early
            // return. A block reserves `tally[s]` consecutive positions per
            // segment with one `atomicAdd` on `cursor[segment]`, and because
            // `cursor` starts at the exclusive scan of the histogram those very
            // cycles produced, the reservations across all blocks exactly tile
            // `[offsets[segment], offsets[segment] + counts[segment])`. Each
            // thread then takes a distinct offset inside its own block's
            // reservation from the re-zeroed tally, so the `total` writes to
            // `indices` are disjoint and in bounds. Cold and out-of-range cycles
            // keep `slot == slots` and write nothing.
            let _ = unsafe { builder.launch(config) }?;
        } else {
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
        }
        self.stream().synchronize()?;
        drop(scatter_span);

        let mut output = self.alloc_u64(segments * 3 * FQ_LIMBS)?;
        tracing::info_span!(
            "cuda_commit_one_hot_segments",
            segments,
            widest,
            entries = total,
            fat = classes.fat,
            thin = classes.thin,
            fat_entries = classes.fat_entries,
            live_thin = classes.live_thin
        )
        .in_scope(|| {
            self.launch_segment_sums(
                bases,
                SegmentPlan {
                    indices: &indices,
                    offsets: &offsets,
                    counts: complements.as_ref().map_or(&counts, |plan| &plan.counts),
                    segments,
                    widest,
                    mode: SegmentMode::Auto,
                    classes: Some(&classes),
                },
                &mut output,
            )?;
            if let Some(plan) = &complements {
                self.complete_one_hot_buckets(
                    bases,
                    chunk_len,
                    chunk_count,
                    one_hot_k,
                    &plan.targets,
                    &mut output,
                )?;
            }
            Ok::<_, CudaError>(())
        })?;
        tracing::info_span!("cuda_commit_one_hot_download", segments)
            .in_scope(|| Ok(unflatten_jacobian(&self.download_u64(&output)?)))
    }

    fn complete_one_hot_buckets(
        &self,
        bases: &DeviceG1Bases,
        chunk_len: usize,
        chunk_count: usize,
        one_hot_k: usize,
        targets: &CudaSlice<u32>,
        output: &mut CudaSlice<u64>,
    ) -> Result<(), CudaError> {
        let base_count = Self::count_of(chunk_len)?;
        let base_indices: Vec<u32> = (0..base_count).collect();
        let tile_len = chunk_len.div_ceil(64);
        let counts: Vec<u32> = (0..chunk_len)
            .step_by(tile_len)
            .map(|start| Self::count_of(tile_len.min(chunk_len - start)))
            .collect::<Result<_, _>>()?;
        let plan = self.upload_segments(bases, &base_indices, &counts)?;
        let mut partials = self.alloc_u64(plan.segments * 3 * FQ_LIMBS)?;
        self.launch_segment_sums(
            bases,
            SegmentPlan {
                indices: &plan.indices,
                offsets: &plan.offsets,
                counts: &plan.counts,
                segments: plan.segments,
                widest: plan.widest,
                mode: SegmentMode::Heavy,
                classes: None,
            },
            &mut partials,
        )?;
        let mut total = self.alloc_u64(3 * FQ_LIMBS)?;
        let rows = 1u32;
        let tiles = Self::count_of(plan.segments)?;
        let mut builder = self.stream().launch_builder(self.msm_point_rows_sum());
        let _ = builder.arg(&partials);
        let _ = builder.arg(&rows);
        let _ = builder.arg(&tiles);
        let _ = builder.arg(&mut total);
        // SAFETY: one block reads the tiles produced above and reduces through
        // POINT_BLOCK * 3 * LIMBS shared u64s. All threads reach the barriers;
        // thread 0 alone writes the fresh, single-point total.
        let _ = unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (POINT_BLOCK, 1, 1),
                shared_mem_bytes: POINT_BLOCK * 3 * FQ_LIMBS as u32 * size_of::<u64>() as u32,
            })
        }?;

        let target_count = Self::count_of(targets.len())?;
        let chunks = Self::count_of(chunk_count)?;
        let addresses = Self::count_of(one_hot_k)?;
        let mut builder = self.stream().launch_builder(self.msm_one_hot_complement());
        let _ = builder.arg(&total);
        let _ = builder.arg(targets);
        let _ = builder.arg(&target_count);
        let _ = builder.arg(&chunks);
        let _ = builder.arg(&addresses);
        let _ = builder.arg(output);
        // SAFETY: targets contains at most one segment per fully covered chunk.
        // Each thread reads only other addresses in its own chunk, then writes
        // its target. Thus no thread reads a segment another thread writes,
        // and all segment indices are below one_hot_k * chunk_count.
        let _ = unsafe { builder.launch(Self::launch_config(target_count)) }?;
        self.stream().synchronize()?;
        Ok(())
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
        let config = LaunchConfig {
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
            builder.launch(LaunchConfig {
                grid_dim: (rows_arg, 1, 1),
                block_dim: (POINT_BLOCK, 1, 1),
                shared_mem_bytes: shared,
            })
        }?;
        Ok(())
    }

    fn batched_window_bytes(rows: usize, buckets: usize, windows: usize, len: usize) -> usize {
        let segments = windows.saturating_mul(rows).saturating_mul(buckets);
        // Include counts, offsets, cursors and scan scratch alongside the buckets.
        let bucket_bytes =
            segments.saturating_mul(3 * FQ_LIMBS * size_of::<u64>() + 4 * size_of::<u32>());
        let index_bytes = windows
            .saturating_mul(len)
            .saturating_mul(2 * size_of::<u32>() + size_of::<u8>());
        let output_bytes = windows
            .saturating_mul(rows)
            .saturating_mul(3 * FQ_LIMBS * size_of::<u64>());
        bucket_bytes
            .saturating_add(index_bytes)
            .saturating_add(output_bytes)
    }

    fn batched_window_budget(free: usize) -> usize {
        (free / 8).clamp(BATCHED_WINDOW_BYTES, MAX_G1_BATCHED_WINDOW_BYTES)
    }

    fn batched_windows(
        &self,
        rows: usize,
        buckets: usize,
        windows: usize,
        len: usize,
    ) -> Result<bool, CudaError> {
        let bytes = Self::batched_window_bytes(rows, buckets, windows, len);
        if windows <= 1 || bytes > MAX_G1_BATCHED_WINDOW_BYTES {
            return Ok(false);
        }
        if bytes <= BATCHED_WINDOW_BYTES {
            return Ok(true);
        }
        let (free, _) = self.stream().context().mem_get_info()?;
        Ok(bytes <= Self::batched_window_budget(free))
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

        let mut digits = self.alloc_u32_unset(total)?;
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
        let mut indices = self.alloc_u32_unset(total.max(1))?;
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

    pub fn upload_segments(
        &self,
        bases: &DeviceG1Bases,
        indices: &[u32],
        counts: &[u32],
    ) -> Result<DeviceSegments, CudaError> {
        let mut offsets = Vec::with_capacity(counts.len());
        let mut running = 0u32;
        let mut widest = 0u32;
        for &count in counts {
            offsets.push(running);
            widest = widest.max(count);
            running = running
                .checked_add(count)
                .ok_or(CudaError::InvariantViolation {
                    reason: "a segment plan holds more than u32::MAX entries",
                })?;
        }
        if running as usize != indices.len() {
            return Err(CudaError::LengthMismatch {
                expected: running as usize,
                got: indices.len(),
            });
        }
        let base_count = u32::try_from(bases.count()).unwrap_or(u32::MAX);
        if indices
            .iter()
            .any(|&index| (index & 0x7fff_ffff) >= base_count)
        {
            return Err(CudaError::InvariantViolation {
                reason: "a segment plan names a base outside the uploaded table",
            });
        }
        let classes = self.upload_classes(&plan_classes(counts))?;
        Ok(DeviceSegments {
            indices: if indices.is_empty() {
                self.alloc_u32(1)?
            } else {
                self.upload_u32_slice(indices)?
            },
            offsets: self.upload_u32_slice(&offsets)?,
            counts: self.upload_u32_slice(counts)?,
            segments: counts.len(),
            widest: widest as usize,
            classes,
        })
    }

    pub(crate) fn upload_classes(&self, plan: &HostClasses) -> Result<DeviceClasses, CudaError> {
        Ok(DeviceClasses {
            ids: if plan.ids.is_empty() {
                self.alloc_u32(1)?
            } else {
                self.upload_u32_slice(&plan.ids)?
            },
            fat: plan.fat,
            thin: plan.ids.len() - plan.fat,
            fat_entries: plan.fat_entries,
            live_thin: plan.live_thin,
        })
    }

    pub fn segment_sums(
        &self,
        bases: &DeviceG1Bases,
        plan: &DeviceSegments,
        mode: SegmentMode,
    ) -> Result<Vec<JacobianLimbs>, CudaError> {
        self.require_owned(bases.ordinal())?;
        let mut output = self.alloc_u64(plan.segments * 3 * FQ_LIMBS)?;
        self.launch_segment_sums(
            bases,
            SegmentPlan {
                indices: &plan.indices,
                offsets: &plan.offsets,
                counts: &plan.counts,
                segments: plan.segments,
                widest: plan.widest,
                mode,
                classes: Some(&plan.classes),
            },
            &mut output,
        )?;
        Ok(unflatten_jacobian(&self.download_u64(&output)?))
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
                mode: SegmentMode::Auto,
                classes: None,
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

    const fn segment_mode(segments: usize, widest: usize, classed: bool) -> SegmentMode {
        if widest <= SMALL_SEGMENT_LIMIT {
            SegmentMode::Small
        } else if classed {
            SegmentMode::Classes
        } else if widest <= WARP_WIDEST_LIMIT && segments >= WARP_SEGMENT_FLOOR {
            SegmentMode::Warp
        } else {
            SegmentMode::Heavy
        }
    }

    fn launch_classed_segments(
        &self,
        bases: &DeviceG1Bases,
        indices: &CudaSlice<u32>,
        offsets: &CudaSlice<u32>,
        counts: &CudaSlice<u32>,
        classes: &DeviceClasses,
        output: &mut CudaSlice<u64>,
    ) -> Result<(), CudaError> {
        let warps = POINT_BLOCK / WARP_LANES;
        let fat = Self::count_of(classes.fat)?;
        let thin = Self::count_of(classes.thin)?;
        let blocks = fat + thin.div_ceil(warps);
        if blocks == 0 {
            return Ok(());
        }
        let mut builder = self.stream().launch_builder(self.msm_segment_sum_classed());
        let _ = builder.arg(bases.limbs());
        let _ = builder.arg(indices);
        let _ = builder.arg(offsets);
        let _ = builder.arg(counts);
        let _ = builder.arg(&classes.ids);
        let _ = builder.arg(&fat);
        let _ = builder.arg(&thin);
        let _ = builder.arg(output);
        // SAFETY: `ids` holds the `fat + thin` segment ids the host wrote, the
        // fat class first, and the two classes partition the segments. A block
        // `b < fat` takes `ids[b]` and reduces that segment across all
        // `blockDim.x` threads through the `POINT_BLOCK * 3 * LIMBS` u64s of
        // dynamic shared memory declared here and matching the kernel's
        // `extern __shared__`, with `__syncthreads()` on both sides of every
        // tree level (the branch is block-uniform, so no thread skips a
        // barrier); only its thread 0 writes. Every other block gives each of
        // its warps slot `(b - fat) * warps + w`, returns unless that is
        // `< thin` — warp-uniform, so no live warp reaches a `__syncwarp()` its
        // siblings skipped — takes `ids[fat + slot]`, and reduces inside the
        // warp, reading only lanes of the same warp (`threadIdx.x + stride`
        // with `stride < WARP_LANES` and `lane < stride`); only its lane 0
        // writes. Either way the segment's `offsets`/`counts` entry delimits the
        // only `indices` window read, and the host builds those as a partition
        // of `indices`, so windows are disjoint across blocks and warps;
        // `bases` is read at the affine points those indices name (masked to 31
        // bits, checked against `bases.count()` on the host). Writes go to
        // `out[segment]` (12 limbs) of a `segments * 12` u64 allocation distinct
        // from every input, and the class partition makes exactly one writer per
        // slot.
        let _ = unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (POINT_BLOCK, 1, 1),
                shared_mem_bytes: POINT_BLOCK * 3 * FQ_LIMBS as u32 * size_of::<u64>() as u32,
            })
        }?;
        self.stream().synchronize()?;
        Ok(())
    }

    fn launch_warp_segments(
        &self,
        bases: &DeviceG1Bases,
        indices: &CudaSlice<u32>,
        offsets: &CudaSlice<u32>,
        counts: &CudaSlice<u32>,
        segments: u32,
        output: &mut CudaSlice<u64>,
    ) -> Result<(), CudaError> {
        let warps = POINT_BLOCK / WARP_LANES;
        let mut builder = self.stream().launch_builder(self.msm_segment_sum_warp());
        let _ = builder.arg(bases.limbs());
        let _ = builder.arg(indices);
        let _ = builder.arg(offsets);
        let _ = builder.arg(counts);
        let _ = builder.arg(&segments);
        let _ = builder.arg(output);
        // SAFETY: the warp `w` of block `b` owns segment `b * warps + w`, read
        // from `offsets`/`counts` and skipped entirely when it is `>= segments`
        // — a warp-uniform guard, so no live warp can reach a `__syncwarp()` its
        // siblings skipped. Its lanes read only the `indices` window that
        // segment delimits, which the host builds as a partition of `indices`
        // (whether the segments are the caller's or the pieces of a split), so
        // windows are disjoint across warps, and `bases` at the affine points
        // those indices name (masked to 31 bits, checked against
        // `bases.count()` on the host). Each lane owns `scratch[threadIdx.x]` of
        // the `POINT_BLOCK * 3 * LIMBS` u64s declared in the launch config and
        // matching the kernel's `extern __shared__`; the tree only ever reads
        // lanes of the same warp (`threadIdx.x + stride` with
        // `stride < WARP_LANES` and `lane < stride`), with `__syncwarp()` after
        // every level. Only lane 0 writes, to `out[segment]` (12 limbs) of a
        // `segments * 12` u64 allocation distinct from every input, so writes
        // are one per warp and non-aliasing.
        let _ = unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (segments.div_ceil(warps), 1, 1),
                block_dim: (POINT_BLOCK, 1, 1),
                shared_mem_bytes: POINT_BLOCK * 3 * FQ_LIMBS as u32 * size_of::<u64>() as u32,
            })
        }?;
        self.stream().synchronize()?;
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
            mode,
            classes,
        } = plan;
        let segment_count = Self::count_of(segments)?;
        if segment_count == 0 {
            return Ok(());
        }
        let mode = match mode {
            SegmentMode::Auto => Self::segment_mode(segments, widest, classes.is_some()),
            forced => forced,
        };
        if mode == SegmentMode::Classes {
            let classes = classes.ok_or(CudaError::InvariantViolation {
                reason: "the size-classed segment sum needs a host-built class plan",
            })?;
            if classes.fat + classes.thin != segments {
                return Err(CudaError::LengthMismatch {
                    expected: segments,
                    got: classes.fat + classes.thin,
                });
            }
            self.launch_classed_segments(bases, indices, offsets, counts, classes, output)?;
            return Ok(());
        }
        if mode == SegmentMode::Warp {
            return self.launch_warp_segments(
                bases,
                indices,
                offsets,
                counts,
                segment_count,
                output,
            );
        }
        if mode == SegmentMode::Small {
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
            // Small segments have one writer; larger ones are selected once
            // by their warp's ballot and only lane 0 writes the result. Every
            // one of the BLOCK = 256 threads, including out-of-range lanes
            // with zero counts, reaches the full-warp ballots and shuffles.
            // The launch bound matches BLOCK; no shared memory is required.
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
            builder.launch(LaunchConfig {
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
        signs: &CudaSlice<u8>,
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

        if self.batched_windows(rows, buckets, windows, len)? {
            let _batched = tracing::info_span!(
                "cuda_pippenger_batched",
                rows,
                windows,
                buckets,
                lanes = signs.len() * windows
            )
            .entered();
            let device_lane_signs = self.replicate_u8(signs, windows)?;
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
        let mut window_points = self.alloc_u64(rows * 3 * FQ_LIMBS)?;
        for window in (0..windows).rev() {
            self.pippenger_pass(
                PassPlan {
                    bases,
                    scalars,
                    scalar_limbs,
                    signs,
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
        ArkFr::new_unchecked(BigInt(value.inner_limbs()))
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
            Fr::from(ArkFr::from_le_bytes_mod_order(&bytes))
        })
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test module: device operations fail loudly"
)]
mod tests {
    use ark_bn254::{Fq, Fr as ArkFr, G1Affine, G1Projective};
    use ark_ec::scalar_mul::variable_base::msm_i128;
    use ark_ec::{AdditiveGroup, AffineRepr, CurveGroup, VariableBaseMSM};
    use ark_ff::{PrimeField, Zero};
    use jolt_crypto::ec::bn254::batch_addition::batch_g1_additions_multi_affine;
    use jolt_field::{Fr, Ring};
    use proptest::collection::vec;
    use proptest::prelude::*;

    use super::testing::{
        affine, affine_limbs, arb_fq, arb_fr, ark_fr, fq_from_limbs, fq_limbs, jacobian_limbs,
        point, projective, projectives,
    };
    use super::{
        unflatten_jacobian, AffineLimbs, JacobianLimbs, OneHotMode, SegmentMode, CLASS_FLOOR, COLD,
        FQ_LIMBS, ONE_HOT_SHARED_SLOTS, SMALL_SEGMENT_LIMIT,
    };
    use crate::cuda::common::context::{shared_context, CudaKernelContext};

    const MSM_ROWS: usize = 3;

    const MSM_ROW_LEN: usize = 8;

    const ONE_HOT_K: usize = 5;

    const CSR_GEOMETRIES: [(usize, usize, usize); 4] =
        [(16, 4, 5), (64, 16, 4), (4096, 2048, 8), (256, 256, 3)];

    const SEGMENT_BASES: usize = 64;

    const RESIDENT_MSM_LENS: [usize; 6] = [1, 2, 255, 256, 1024, 4096];

    fn device() -> Option<&'static CudaKernelContext> {
        shared_context()
    }

    fn limbs(values: &[Fq]) -> Vec<[u64; FQ_LIMBS]> {
        values.iter().copied().map(fq_limbs).collect()
    }

    fn arb_msm_scalar() -> impl Strategy<Value = i128> {
        (-(u64::MAX as i128))..=(u64::MAX as i128)
    }

    fn msm_bases() -> Vec<G1Affine> {
        (0..MSM_ROW_LEN).map(|i| affine(i as u64 + 3)).collect()
    }

    fn segment_shapes() -> Vec<Vec<u32>> {
        vec![
            vec![0, 1, 31, 32, 33, 63, 64, 100, 7],
            vec![0, 0, 0],
            vec![0, 500, 0],
            vec![3; 100],
            vec![1024, 0, 0, 0, 1],
            vec![65; 4],
            vec![1025],
            vec![4096],
            vec![2049, 1, 0, 3000, 1024],
            vec![9000, 1, 0, 4096, 5000],
            vec![16384, 512],
        ]
    }

    fn segment_indices(counts: &[u32], seed: u64) -> Vec<u32> {
        let total: usize = counts.iter().map(|&count| count as usize).sum();
        let mut state = seed | 1;
        (0..total)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let index = (state % SEGMENT_BASES as u64) as u32;
                index | (u32::from(state & (1 << 40) != 0) << 31)
            })
            .collect()
    }

    fn csr_hot(cycles: usize, one_hot_k: usize, seed: u64) -> Vec<u32> {
        (0..cycles)
            .map(|cycle| {
                let mixed = (cycle as u64)
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    .wrapping_add(seed);
                if mixed.is_multiple_of(5) {
                    COLD
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
            if address != COLD {
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
                hot.contains(&COLD),
                "geometry {index}: no cold cycle, so a scatter that ignored the sentinel would \
                 pass",
            );
            let counts = csr_segment_counts(&hot, one_hot_k, chunk_len);
            let widest = counts.iter().copied().max().unwrap_or(0);
            if widest <= SMALL_SEGMENT_LIMIT {
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

    #[test]
    fn fq_arithmetic_carries_across_limb_and_modulus_boundaries() {
        let Some(context) = device() else {
            return;
        };
        let mut values = vec![Fq::zero(), fq_from_limbs([1, 0, 0, 0])];
        let mut near_modulus = Fq::MODULUS.0;
        near_modulus[0] -= 1;
        values.push(fq_from_limbs(near_modulus));
        for end in 1..FQ_LIMBS {
            let mut below = [0; FQ_LIMBS];
            below[..end].fill(u64::MAX);
            let mut above = [0; FQ_LIMBS];
            above[end] = 1;
            values.extend([fq_from_limbs(below), fq_from_limbs(above)]);
        }
        let pairs: Vec<_> = values
            .iter()
            .flat_map(|&a| values.iter().map(move |&b| (a, b)))
            .collect();
        let left: Vec<_> = pairs.iter().map(|&(a, _)| fq_limbs(a)).collect();
        let right: Vec<_> = pairs.iter().map(|&(_, b)| fq_limbs(b)).collect();
        for (actual, expected) in [
            (
                context.fq_add(&left, &right).expect("add"),
                pairs
                    .iter()
                    .map(|&(a, b)| fq_limbs(a + b))
                    .collect::<Vec<_>>(),
            ),
            (
                context.fq_sub(&left, &right).expect("sub"),
                pairs
                    .iter()
                    .map(|&(a, b)| fq_limbs(a - b))
                    .collect::<Vec<_>>(),
            ),
            (
                context.fq_mul(&left, &right).expect("mul"),
                pairs
                    .iter()
                    .map(|&(a, b)| fq_limbs(a * b))
                    .collect::<Vec<_>>(),
            ),
        ] {
            assert_eq!(actual, expected);
        }
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
        fn classed_segment_sums_match_the_block_kernel(seed in any::<u64>()) {
            let Some(context) = device() else { return Ok(()); };
            let bases: Vec<AffineLimbs> = (0..SEGMENT_BASES)
                .map(|column| affine_limbs(affine(column as u64 + 1)))
                .collect();
            let device_bases = context.upload_g1_bases(&bases).expect("upload bases");
            let mut mixed = 0usize;
            for (index, counts) in segment_shapes().into_iter().enumerate() {
                let indices = segment_indices(&counts, seed ^ index as u64);
                let plan = context
                    .upload_segments(&device_bases, &indices, &counts)
                    .expect("upload segment plan");
                mixed += usize::from(plan.widest() > CLASS_FLOOR);
                let expected = projectives(
                    &context
                        .segment_sums(&device_bases, &plan, SegmentMode::Heavy)
                        .expect("block segment sums"),
                );
                let got = projectives(
                    &context
                        .segment_sums(&device_bases, &plan, SegmentMode::Classes)
                        .expect("classed segment sums"),
                );
                prop_assert_eq!(
                    got,
                    expected,
                    "shape {} ({} segments, widest {}): the classed kernels diverged from the \
                     block kernel",
                    index,
                    counts.len(),
                    plan.widest()
                );
            }
            prop_assert!(
                mixed > 0,
                "no shape exceeds CLASS_FLOOR, so the fat-class launch is untested"
            );
        }

        #[test]
        fn warp_segment_sums_match_the_block_kernels(seed in any::<u64>()) {
            let Some(context) = device() else { return Ok(()); };
            let bases: Vec<AffineLimbs> = (0..SEGMENT_BASES)
                .map(|column| affine_limbs(affine(column as u64 + 1)))
                .collect();
            let device_bases = context.upload_g1_bases(&bases).expect("upload bases");
            for (index, counts) in segment_shapes().into_iter().enumerate() {
                let indices = segment_indices(&counts, seed ^ index as u64);
                let plan = context
                    .upload_segments(&device_bases, &indices, &counts)
                    .expect("upload segment plan");
                let expected = projectives(
                    &context
                        .segment_sums(&device_bases, &plan, SegmentMode::Heavy)
                        .expect("block segment sums"),
                );
                let got = projectives(
                    &context
                        .segment_sums(&device_bases, &plan, SegmentMode::Warp)
                        .expect("warp segment sums"),
                );
                prop_assert_eq!(
                    got,
                    expected.clone(),
                    "shape {} ({} segments, widest {}): the warp kernel diverged from the \
                     block kernel",
                    index,
                    counts.len(),
                    plan.widest()
                );
                if plan.widest() <= SMALL_SEGMENT_LIMIT {
                    let small = projectives(
                        &context
                            .segment_sums(&device_bases, &plan, SegmentMode::Small)
                            .expect("thread segment sums"),
                    );
                    prop_assert_eq!(
                        small,
                        expected,
                        "shape {}: the thread-per-segment kernel diverged from the block kernel",
                        index
                    );
                }
            }
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
                .map(|row| row.map_or(COLD, |address| address as u32))
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
                for mode in [OneHotMode::Flat, OneHotMode::Shared] {
                    if mode == OneHotMode::Shared
                        && !CudaKernelContext::one_hot_shares(one_hot_k, chunk_len)
                    {
                        continue;
                    }
                    let got = projectives(
                        &context
                            .one_hot_rows_device_with(
                                &device_bases,
                                &device_hot.slice(..),
                                cycles,
                                one_hot_k,
                                chunk_len,
                                mode,
                            )
                            .expect("device one_hot_rows"),
                    );
                    prop_assert_eq!(
                        got,
                        expected.clone(),
                        "geometry {} (cycles {}, chunk_len {}, one_hot_k {}) in {:?} mode: the \
                         device segment plan diverged from the host one",
                        index,
                        cycles,
                        chunk_len,
                        one_hot_k,
                        mode
                    );
                }
            }
        }

        #[test]
        fn the_shared_one_hot_histogram_covers_every_commitment_grid(
            log_t in 8usize..33,
            log_k_chunk in prop::sample::select(vec![0usize, 4, 8]),
        ) {
            use crate::commitment::CommitmentGrid;
            use jolt_claims::protocols::jolt::TracePolynomialOrder;

            let grid = CommitmentGrid {
                total_vars: log_t + log_k_chunk,
                log_t,
                log_k_chunk,
                order: TracePolynomialOrder::CycleMajor,
            };
            let one_hot_k = 1usize << log_k_chunk;
            let chunk_len = grid.num_columns();
            prop_assert!(
                CudaKernelContext::one_hot_shares(one_hot_k, chunk_len),
                "a 2^{} trace with a {}-bit committed chunk needs {} tally slots, more than the \
                 {} the shared histogram holds, so the commit arm would silently fall back to \
                 the flat kernels",
                log_t,
                log_k_chunk,
                CudaKernelContext::one_hot_slots(one_hot_k, chunk_len),
                ONE_HOT_SHARED_SLOTS,
            );
        }

        #[test]
        fn the_shared_one_hot_histogram_declines_a_shape_it_cannot_hold(seed in any::<u64>()) {
            let Some(context) = device() else { return Ok(()); };
            let (cycles, chunk_len, one_hot_k) = (2048usize, 8usize, 256usize);
            prop_assert!(
                CudaKernelContext::one_hot_slots(one_hot_k, chunk_len) > ONE_HOT_SHARED_SLOTS,
                "this geometry no longer overflows the per-block tally, so it cannot exercise \
                 the fallback",
            );
            let hot = csr_hot(cycles, one_hot_k, seed);
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
                    .one_hot_rows_device_with(
                        &device_bases,
                        &device_hot.slice(..),
                        cycles,
                        one_hot_k,
                        chunk_len,
                        OneHotMode::Shared,
                    )
                    .is_err(),
                "the shared histogram accepted {} tally slots, more than the {} it can hold",
                CudaKernelContext::one_hot_slots(one_hot_k, chunk_len),
                ONE_HOT_SHARED_SLOTS,
            );
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
                "the automatic mode did not fall back to the flat histogram for {} tally slots",
                CudaKernelContext::one_hot_slots(one_hot_k, chunk_len),
            );
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
    }

    #[test]
    fn one_hot_complements_preserve_cold_chunks_and_base_prefixes() {
        let Some(context) = device() else { return };
        const CHUNK: usize = 4096;
        const CHUNKS: usize = 7;
        const K: usize = 4;
        let hot: Vec<u32> = (0..CHUNK * CHUNKS)
            .map(|cycle| {
                let column = cycle % CHUNK;
                match cycle / CHUNK {
                    0 => 0,
                    1 | 2 if cycle / CHUNK == 2 && column == CHUNK - 1 => COLD,
                    1 | 2 if column < 3 * CHUNK / 4 => 2,
                    1 | 2 => [0, 1, 3][column % 3],
                    3 => u32::from(column >= CHUNK / 2),
                    4 => COLD,
                    5 if column < 3 * CHUNK / 4 => 3,
                    5 => (column % 3) as u32,
                    _ => (column % K) as u32,
                }
            })
            .collect();
        let mut histogram = vec![0u32; K * CHUNKS];
        for (cycle, &address) in hot.iter().enumerate() {
            if address != COLD {
                histogram[address as usize * CHUNKS + cycle / CHUNK] += 1;
            }
        }
        assert_eq!(
            CudaKernelContext::one_hot_complement_targets(&mut histogram, CHUNKS, CHUNK)
                .expect("plan complements"),
            vec![0, (2 * CHUNKS + 1) as u32, (3 * CHUNKS + 5) as u32],
        );
        let device_hot = context.upload_u32_slice(&hot).expect("upload addresses");
        for balanced in [false, true] {
            // Known generator multiples provide an independent oracle; balanced
            // bases sum to identity, and the extra bases must never contribute.
            let weights: Vec<i64> = (0..CHUNK + 3)
                .map(|index| {
                    if index >= CHUNK {
                        10_000 + index as i64
                    } else if balanced && index >= CHUNK / 2 {
                        -((index - CHUNK / 2 + 1) as i64)
                    } else {
                        (index + 1) as i64
                    }
                })
                .collect();
            let bases: Vec<AffineLimbs> = weights
                .iter()
                .map(|&weight| {
                    let p = point(weight.unsigned_abs());
                    affine_limbs(if weight < 0 { -p } else { p }.into_affine())
                })
                .collect();
            let mut sums = vec![0i64; K * CHUNKS];
            for (cycle, &address) in hot.iter().enumerate() {
                if address != COLD {
                    sums[address as usize * CHUNKS + cycle / CHUNK] += weights[cycle % CHUNK];
                }
            }
            let expected: Vec<G1Projective> = sums
                .iter()
                .map(|&sum| {
                    let p = point(sum.unsigned_abs());
                    if sum < 0 {
                        -p
                    } else {
                        p
                    }
                })
                .collect();
            let device_bases = context.upload_g1_bases(&bases).expect("upload bases");
            for mode in [OneHotMode::Flat, OneHotMode::Shared] {
                let got = projectives(
                    &context
                        .one_hot_rows_device_with(
                            &device_bases,
                            &device_hot.as_view(),
                            hot.len(),
                            K,
                            CHUNK,
                            mode,
                        )
                        .expect("one-hot complement sums"),
                );
                assert_eq!(got, expected, "{mode:?}, balanced = {balanced}");
            }
        }
    }

    #[test]
    fn small_segment_sums_handle_skew_and_partial_warps() {
        let Some(context) = device() else { return };
        // Base i is i times the generator, so signed index sums are an independent oracle.
        let bases: Vec<AffineLimbs> = (0..SEGMENT_BASES)
            .map(|index| affine_limbs(affine(index as u64)))
            .collect();
        let device_bases = context.upload_g1_bases(&bases).expect("upload bases");
        for segments in [1, 31, 32, 33, 255, 256, 257] {
            let counts: Vec<u32> = (0..segments)
                .map(|index| {
                    if index + 1 == segments {
                        4097
                    } else {
                        [0, 1, 63, 64, 65, 66, 1024][index % 7]
                    }
                })
                .collect();
            let mut indices = Vec::new();
            let expected: Vec<G1Projective> = counts
                .iter()
                .enumerate()
                .map(|(segment, &count)| {
                    let mut sum = 0i64;
                    for term in 0..count {
                        let index = (term / 2 + segment as u32 + 1) % SEGMENT_BASES as u32;
                        let negate = if segment % 2 == 0 {
                            term % 2 != 0
                        } else {
                            term % 3 == 0
                        };
                        indices.push(index | (u32::from(negate) << 31));
                        sum += if negate {
                            -i64::from(index)
                        } else {
                            i64::from(index)
                        };
                    }
                    let result = point(sum.unsigned_abs());
                    if sum < 0 {
                        -result
                    } else {
                        result
                    }
                })
                .collect();
            let plan = context
                .upload_segments(&device_bases, &indices, &counts)
                .expect("upload mixed segment plan");
            let got = projectives(
                &context
                    .segment_sums(&device_bases, &plan, SegmentMode::Small)
                    .expect("adaptive small segment sums"),
            );
            assert_eq!(got, expected, "{segments} segments");
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
    fn batched_window_budget_reserves_device_headroom() {
        let mib = 1usize << 20;
        assert_eq!(CudaKernelContext::batched_window_budget(0), 64 * mib);
        assert_eq!(
            CudaKernelContext::batched_window_budget(1024 * mib),
            128 * mib
        );
        assert_eq!(
            CudaKernelContext::batched_window_budget(usize::MAX),
            4096 * mib
        );
        let large = CudaKernelContext::batched_window_bytes(1024, 2048, 6, 1 << 24);
        assert!(large > CudaKernelContext::batched_window_budget(8 << 30));
        assert!(large < CudaKernelContext::batched_window_budget(32 << 30));
        assert_eq!(
            CudaKernelContext::batched_window_bytes(usize::MAX, 2, 2, 1),
            usize::MAX
        );
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
