use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, PushKernelArg};

use super::context::CudaKernelContext;
use jolt_field::Fr;

use super::device::DeviceFrVec;
use super::error::CudaError;
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
        self.pippenger(
            bases,
            &canonical,
            &signs,
            rows.len(),
            row_len,
            FR_SCALAR_BITS,
        )
    }

    pub fn msm_rows_i128(
        &self,
        bases: &DeviceG1Bases,
        rows: &[i128],
        row_len: usize,
    ) -> Result<Vec<JacobianLimbs>, CudaError> {
        let mut magnitudes = Vec::with_capacity(rows.len() * FQ_LIMBS);
        let mut signs = Vec::with_capacity(rows.len());
        for &scalar in rows {
            let magnitude = scalar.unsigned_abs();
            if magnitude > u128::from(u64::MAX) {
                return Err(CudaError::InvariantViolation {
                    reason: "signed MSM scalars must fit in [-u64::MAX, u64::MAX]",
                });
            }
            magnitudes.extend_from_slice(&[magnitude as u64, 0, 0, 0]);
            signs.push(u8::from(scalar.is_negative()));
        }
        let device = self.upload_u64_slice(&magnitudes)?;
        self.pippenger(bases, &device, &signs, rows.len(), row_len, 64)
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
        self.segment_sums(bases, &indices, &offsets, &counts, one_hot_k)
    }

    pub fn one_hot_rows(
        &self,
        bases: &DeviceG1Bases,
        hot: &[Option<usize>],
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
        let mut counts = vec![0u32; segments];
        for (column, row) in hot.iter().enumerate() {
            let Some(row) = *row else { continue };
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
        for (column, row) in hot.iter().enumerate() {
            let Some(row) = *row else { continue };
            let segment = row * chunk_count + column / chunk_len;
            indices[cursor[segment] as usize] = (column % chunk_len) as u32;
            cursor[segment] += 1;
        }
        self.segment_sums(bases, &indices, &offsets, &counts, segments)
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

    fn pippenger(
        &self,
        bases: &DeviceG1Bases,
        scalars: &CudaSlice<u64>,
        signs: &[u8],
        len: usize,
        row_len: usize,
        scalar_bits: usize,
    ) -> Result<Vec<JacobianLimbs>, CudaError> {
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
        let segments = rows * buckets;
        let windows = scalar_bits.div_ceil(window_bits);

        let device_signs = self.upload_u8_slice(signs)?;
        let mut accumulator = self.alloc_u64(rows * 3 * FQ_LIMBS)?;
        let mut window_points = self.alloc_u64(rows * 3 * FQ_LIMBS)?;
        let count = Self::count_of(len)?;
        let row_len_arg = Self::count_of(row_len)?;
        let buckets_arg = Self::count_of(buckets)?;
        let rows_arg = Self::count_of(rows)?;
        let mask = Self::count_of(buckets - 1)?;

        for window in (0..windows).rev() {
            let shift = Self::count_of(window * window_bits)?;
            let mut digits = self.alloc_u32(len)?;
            let mut builder = self.stream().launch_builder(self.msm_digits());
            let _ = builder.arg(scalars);
            let _ = builder.arg(&count);
            let _ = builder.arg(&shift);
            let _ = builder.arg(&mask);
            let _ = builder.arg(&mut digits);
            // SAFETY: thread `i < count` reads only `scalars[i]` (4 limbs of a
            // `count * LIMBS` buffer) and writes only `digits[i]` of a `count`
            // u32 fresh allocation. The digit extraction reads at most limbs
            // `shift/64` and `shift/64 + 1`, both bounds-checked in the kernel.
            let _ = unsafe { builder.launch(Self::launch_config(count)) }?;

            let mut counts = self.alloc_u32(segments)?;
            let mut builder = self.stream().launch_builder(self.msm_bucket_count());
            let _ = builder.arg(&digits);
            let _ = builder.arg(&count);
            let _ = builder.arg(&row_len_arg);
            let _ = builder.arg(&buckets_arg);
            let _ = builder.arg(&mut counts);
            // SAFETY: thread `i < count` reads `digits[i]` and atomically
            // increments `counts[(i / row_len) * buckets + digit]`. The digit is
            // masked to `buckets - 1` in the previous kernel and `i / row_len <
            // rows`, so the index stays inside the `segments = rows * buckets`
            // u32 allocation, which was zeroed by `alloc_u32`. The increment is
            // an atomic, so concurrent hits on one counter are safe.
            let _ = unsafe { builder.launch(Self::launch_config(count)) }?;
            self.stream().synchronize()?;

            let (offsets, total) = self.exclusive_scan_with_total_u32(&counts, segments)?;
            let widest_bucket = self.download_u32(&counts)?.into_iter().max().unwrap_or(0) as usize;
            let mut cursor = self.clone_u32(&offsets)?;
            let mut indices = self.alloc_u32(total.max(1))?;
            let mut builder = self.stream().launch_builder(self.msm_bucket_scatter());
            let _ = builder.arg(&digits);
            let _ = builder.arg(&device_signs);
            let _ = builder.arg(&count);
            let _ = builder.arg(&row_len_arg);
            let _ = builder.arg(&buckets_arg);
            let _ = builder.arg(&mut cursor);
            let _ = builder.arg(&mut indices);
            // SAFETY: thread `i < count` reads `digits[i]` and `signs[i]`
            // (`count` elements each), atomically bumps
            // `cursor[(i / row_len) * buckets + digit]` inside the `segments`
            // u32 copy of the scan offsets, and writes the returned slot of
            // `indices`. Because the cursor starts at the exclusive scan of the
            // same counts the previous kernel produced, the slots handed out are
            // exactly the `total` positions of `indices`, each to one thread —
            // so writes are disjoint and in bounds.
            let _ = unsafe { builder.launch(Self::launch_config(count)) }?;
            self.stream().synchronize()?;

            let mut bucket_points = self.alloc_u64(segments * 3 * FQ_LIMBS)?;
            self.launch_segment_sums(
                bases,
                SegmentPlan {
                    indices: &indices,
                    offsets: &offsets,
                    counts: &counts,
                    segments,
                    widest: widest_bucket,
                },
                &mut bucket_points,
            )?;

            let mut builder = self
                .stream()
                .launch_builder(self.msm_bucket_reduce_parallel());
            let _ = builder.arg(&bucket_points);
            let _ = builder.arg(&rows_arg);
            let _ = builder.arg(&buckets_arg);
            let _ = builder.arg(&mut window_points);
            // SAFETY: block `row < rows` reads only the `buckets` points of its
            // own row (`bucket_points[row * buckets .. (row+1) * buckets]`,
            // inside a `segments = rows * buckets` element buffer), each thread
            // walking a disjoint bucket sub-range. The block reduces through
            // `POINT_BLOCK * 3 * LIMBS` u64s of dynamic shared memory, declared
            // in the launch config and matching the kernel's `extern
            // __shared__`, with `__syncthreads()` on both sides of every tree
            // level. Only thread 0 writes, to `out[row]` (12 limbs) of a
            // `rows * 12` u64 buffer distinct from the input, so writes are one
            // per block.
            let _ = unsafe {
                builder.launch(cudarc::driver::LaunchConfig {
                    grid_dim: (rows_arg, 1, 1),
                    block_dim: (POINT_BLOCK, 1, 1),
                    shared_mem_bytes: POINT_BLOCK * 3 * FQ_LIMBS as u32 * size_of::<u64>() as u32,
                })
            }?;
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

        Ok(unflatten_jacobian(&self.download_u64(&accumulator)?))
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
        let device_left = self.upload(left)?;
        let mut output = self.alloc(columns)?;
        let rows_arg = Self::count_of(rows)?;
        let columns_arg = Self::count_of(columns)?;
        let mut builder = self.stream().launch_builder(self.msm_fold_rows());
        let _ = builder.arg(table.limbs());
        let _ = builder.arg(device_left.limbs());
        let _ = builder.arg(&rows_arg);
        let _ = builder.arg(&columns_arg);
        let _ = builder.arg(output.limbs_mut());
        // SAFETY: thread `c < columns` reads `left[row]` and
        // `table[row * columns + c]` for every `row < rows` — inside buffers of
        // `rows` and `rows * columns` elements respectively — and writes only
        // `out[c]` of a `columns`-element fresh allocation distinct from both
        // inputs. Threads with `c >= columns` return before any access.
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
    use ark_bn254::{Fr as ArkFr, G1Affine, G1Projective};
    use ark_ec::scalar_mul::variable_base::msm_i128;
    use ark_ec::{AdditiveGroup, AffineRepr, CurveGroup, VariableBaseMSM};
    use ark_ff::Zero;
    use jolt_crypto::ec::bn254::batch_addition::batch_g1_additions_multi_affine;
    use jolt_field::{Fr, FromPrimitiveInt};
    use proptest::collection::vec;
    use proptest::prelude::*;

    use super::testing::{
        affine, affine_limbs, arb_fq, arb_fr, ark_fr, fq_from_limbs, fq_limbs, jacobian_limbs,
        point, projective, projectives,
    };
    use super::{AffineLimbs, JacobianLimbs, FQ_LIMBS};
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
            let got = projectives(
                &context
                    .one_hot_rows(&device_bases, &hot, ONE_HOT_K, chunk_len)
                    .expect("device one_hot_rows"),
            );
            prop_assert_eq!(
                got,
                expected,
                "the batched row form must equal the per-chunk form transposed"
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
}
