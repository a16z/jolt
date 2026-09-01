#![expect(
    dead_code,
    reason = "implementation target: the registers read-write vertical is the first non-test caller"
)]

use cudarc::driver::{CudaSlice, PushKernelArg};
use jolt_field::{Field, Fr, FromPrimitiveInt};
use jolt_poly::BindingOrder;

use super::context::{context_for, CudaKernelContext, BLOCK};
use super::device::{require_fr, DeviceFrVec, LIMBS};
use super::devices::{fan_out, DeviceTask};
use super::error::CudaError;
use super::split_eq::DeviceSplitEq;

pub const MAX_COEFFS: usize = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MatrixEntry {
    pub row: u32,
    pub col: u32,
    pub val_coeff: Fr,
    pub prev_val: u64,
    pub next_val: u64,
    pub coeffs: [Fr; MAX_COEFFS],
}

pub const MAX_COEFF_LUT_VALUES: usize = 1 << 16;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CoeffTables {
    pub values: [Vec<Fr>; MAX_COEFFS],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LutEntry {
    pub row: u32,
    pub col: u32,
    pub val_coeff: Fr,
    pub prev_val: u64,
    pub next_val: u64,
    pub coeff_index: [u16; MAX_COEFFS],
}

struct Segments {
    start: CudaSlice<u32>,
    even_end: CudaSlice<u32>,
    end: CudaSlice<u32>,
    pair: CudaSlice<u32>,
}

pub(crate) enum DeviceCoeffs {
    Direct(DeviceFrVec),
    Indexed {
        index: CudaSlice<u16>,
        luts: [DeviceCoeffLut; MAX_COEFFS],
    },
}

pub(crate) struct DeviceCoeffLut {
    values: DeviceFrVec,
}

impl DeviceCoeffLut {
    fn new(context: &CudaKernelContext, values: &[Fr]) -> Result<Self, CudaError> {
        if !values.len().is_power_of_two() || values.len() > MAX_COEFF_LUT_VALUES {
            return Err(CudaError::InvariantViolation {
                reason: "a coefficient lookup table holds a power-of-two count of values, at most \
                         one u16 index can address",
            });
        }
        if values.first() != Some(&Fr::from_u64(0)) {
            return Err(CudaError::InvariantViolation {
                reason: "a coefficient lookup table's first value must be zero: the merge's \
                         one-sided cases address it as the absent partner",
            });
        }
        Ok(Self {
            values: context.upload(values)?,
        })
    }

    fn placeholder(context: &CudaKernelContext) -> Result<Self, CudaError> {
        Ok(Self {
            values: context.alloc(1)?,
        })
    }

    fn bits(&self) -> u32 {
        self.values.len().trailing_zeros()
    }

    fn saturated(&self) -> bool {
        self.values.len() * self.values.len() > MAX_COEFF_LUT_VALUES
    }

    fn bind(&self, context: &CudaKernelContext, challenge: Fr) -> Result<Self, CudaError> {
        let len = self.values.len();
        if self.saturated() {
            return Err(CudaError::InvariantViolation {
                reason: "a saturated coefficient lookup table cannot square again: its indices \
                         would leave the u16 domain",
            });
        }
        let values_len = CudaKernelContext::count_of(len)?;
        let device_challenge = context.upload(&[challenge])?;
        let mut out = context.alloc(len * len)?;
        let mut builder = context.stream().launch_builder(context.rwm_square_lut());
        let _ = builder.arg(self.values.limbs());
        let _ = builder.arg(&values_len);
        let _ = builder.arg(device_challenge.limbs());
        let _ = builder.arg(out.limbs_mut());
        // SAFETY: thread `index < len * len` reads `values[index / len]` and
        // `values[index % len]` — both `< len` — plus the single-element
        // challenge, and writes only `out[index]` of `len * len` elements.
        // Input and output are distinct allocations.
        let _ =
            unsafe { builder.launch(CudaKernelContext::launch_config(values_len * values_len)) }?;
        context.stream().synchronize()?;
        Ok(Self { values: out })
    }
}

pub struct DeviceReadWriteMatrix {
    rows: CudaSlice<u32>,
    cols: CudaSlice<u32>,
    val_coeff: DeviceFrVec,
    prev_val: CudaSlice<u64>,
    next_val: CudaSlice<u64>,
    coeffs: DeviceFrVec,
    coeff_index: CudaSlice<u16>,
    luts: [DeviceCoeffLut; MAX_COEFFS],
    indexed: bool,
    wa_scale: DeviceFrVec,
    coeff_width: usize,
    entries: usize,
    rounds_bound: usize,
}

impl DeviceReadWriteMatrix {
    #[cfg(feature = "allocative")]
    pub fn device_bytes(&self) -> usize {
        (self.rows.len() + self.cols.len()) * size_of::<u32>()
            + (self.prev_val.len() + self.next_val.len()) * size_of::<u64>()
            + self.val_coeff.device_bytes()
            + self.coeffs.device_bytes()
            + self.coeff_index.len() * size_of::<u16>()
            + self
                .luts
                .iter()
                .map(|lut| lut.values.device_bytes())
                .sum::<usize>()
            + self.wa_scale.device_bytes()
    }

    pub fn new(
        context: &CudaKernelContext,
        entries: &[MatrixEntry],
        coeff_width: usize,
        wa_scale: Option<Fr>,
    ) -> Result<Self, CudaError> {
        if coeff_width == 0 || coeff_width > MAX_COEFFS {
            return Err(CudaError::InvariantViolation {
                reason: "a read-write matrix carries one or two one-hot coefficients per entry",
            });
        }
        if wa_scale.is_some() != (coeff_width == 1) {
            return Err(CudaError::InvariantViolation {
                reason: "a single-coefficient matrix derives its write lane by scaling its read lane, and only such a matrix may",
            });
        }
        Self::upload_entries(
            context,
            entries,
            coeff_width,
            wa_scale.unwrap_or(Fr::from_u64(0)),
            0,
        )
    }

    fn upload_entries(
        context: &CudaKernelContext,
        entries: &[MatrixEntry],
        coeff_width: usize,
        wa_scale: Fr,
        rounds_bound: usize,
    ) -> Result<Self, CudaError> {
        let count = entries.len();
        let mut rows = Vec::with_capacity(count);
        let mut cols = Vec::with_capacity(count);
        let mut val_coeff = Vec::with_capacity(count);
        let mut prev_val = Vec::with_capacity(count);
        let mut next_val = Vec::with_capacity(count);
        let mut coeffs = Vec::with_capacity(count * coeff_width);
        for entry in entries {
            rows.push(entry.row);
            cols.push(entry.col);
            val_coeff.push(entry.val_coeff);
            prev_val.push(entry.prev_val);
            next_val.push(entry.next_val);
            coeffs.extend_from_slice(&entry.coeffs[..coeff_width]);
        }
        Ok(Self {
            rows: context.upload_u32_slice(&rows)?,
            cols: context.upload_u32_slice(&cols)?,
            val_coeff: context.upload(&val_coeff)?,
            prev_val: context.upload_u64_slice(&prev_val)?,
            next_val: context.upload_u64_slice(&next_val)?,
            coeffs: context.upload(&coeffs)?,
            coeff_index: context.alloc_u16_unset(1)?,
            luts: [
                DeviceCoeffLut::placeholder(context)?,
                DeviceCoeffLut::placeholder(context)?,
            ],
            indexed: false,
            wa_scale: context.upload(&[wa_scale])?,
            coeff_width,
            entries: count,
            rounds_bound,
        })
    }

    pub fn new_indexed(
        context: &CudaKernelContext,
        entries: &[LutEntry],
        tables: &CoeffTables,
    ) -> Result<Self, CudaError> {
        let luts = Self::upload_luts(context, tables)?;
        let count = entries.len();
        let mut rows = Vec::with_capacity(count);
        let mut cols = Vec::with_capacity(count);
        let mut val_coeff = Vec::with_capacity(count);
        let mut prev_val = Vec::with_capacity(count);
        let mut next_val = Vec::with_capacity(count);
        let mut coeff_index = Vec::with_capacity(count * MAX_COEFFS);
        for entry in entries {
            for (lane, &index) in entry.coeff_index.iter().enumerate() {
                if usize::from(index) >= tables.values[lane].len() {
                    return Err(CudaError::LengthMismatch {
                        expected: tables.values[lane].len(),
                        got: usize::from(index),
                    });
                }
            }
            rows.push(entry.row);
            cols.push(entry.col);
            val_coeff.push(entry.val_coeff);
            prev_val.push(entry.prev_val);
            next_val.push(entry.next_val);
            coeff_index.extend_from_slice(&entry.coeff_index);
        }
        Ok(Self {
            rows: context.upload_u32_slice(&rows)?,
            cols: context.upload_u32_slice(&cols)?,
            val_coeff: context.upload(&val_coeff)?,
            prev_val: context.upload_u64_slice(&prev_val)?,
            next_val: context.upload_u64_slice(&next_val)?,
            coeffs: context.alloc(1)?,
            coeff_index: context.upload_u16_slice(&coeff_index)?,
            luts,
            indexed: true,
            wa_scale: context.upload(&[Fr::from_u64(0)])?,
            coeff_width: MAX_COEFFS,
            entries: count,
            rounds_bound: 0,
        })
    }

    pub(crate) fn upload_luts(
        context: &CudaKernelContext,
        tables: &CoeffTables,
    ) -> Result<[DeviceCoeffLut; MAX_COEFFS], CudaError> {
        let luts: Vec<DeviceCoeffLut> = tables
            .values
            .iter()
            .map(|values| DeviceCoeffLut::new(context, values))
            .collect::<Result<_, _>>()?;
        luts.try_into().map_err(|_| CudaError::LengthMismatch {
            expected: MAX_COEFFS,
            got: 0,
        })
    }

    pub fn coeff_tables(&self) -> Result<Option<CoeffTables>, CudaError> {
        if !self.indexed {
            return Ok(None);
        }
        let values: Vec<Vec<Fr>> = self
            .luts
            .iter()
            .map(|lut| lut.values.to_host())
            .collect::<Result<_, _>>()?;
        let values = values.try_into().map_err(|_| CudaError::LengthMismatch {
            expected: MAX_COEFFS,
            got: 0,
        })?;
        Ok(Some(CoeffTables { values }))
    }

    fn host_coeffs(&self, context: &CudaKernelContext) -> Result<Vec<Fr>, CudaError> {
        if !self.indexed {
            return self.coeffs.to_host();
        }
        let index = context.download_u16(&self.coeff_index)?;
        let tables: Vec<Vec<Fr>> = self
            .luts
            .iter()
            .map(|lut| lut.values.to_host())
            .collect::<Result<_, _>>()?;
        index
            .iter()
            .enumerate()
            .map(|(slot, &index)| {
                tables
                    .get(slot % self.coeff_width)
                    .and_then(|values| values.get(usize::from(index)))
                    .copied()
                    .ok_or(CudaError::LengthMismatch {
                        expected: self.coeff_width,
                        got: slot % self.coeff_width,
                    })
            })
            .collect()
    }

    fn require_direct(&self) -> Result<(), CudaError> {
        if self.indexed {
            return Err(CudaError::InvariantViolation {
                reason: "this read-write matrix operation needs field coefficients, and the \
                         entries still hold lookup-table indices",
            });
        }
        Ok(())
    }

    fn deref_coeffs(&mut self, context: &CudaKernelContext) -> Result<(), CudaError> {
        if !self.indexed {
            return Ok(());
        }
        let slots = CudaKernelContext::count_of(self.entries * self.coeff_width)?;
        let mut coeffs = context.alloc(self.entries * self.coeff_width)?;
        if slots > 0 {
            let mut builder = context.stream().launch_builder(context.rwm_deref_coeffs());
            let _ = builder.arg(&self.coeff_index);
            let _ = builder.arg(self.luts[0].values.limbs());
            let _ = builder.arg(self.luts[1].values.limbs());
            let _ = builder.arg(&slots);
            let _ = builder.arg(coeffs.limbs_mut());
            // SAFETY: thread `slot < slots` reads `coeff_index[slot]` of the
            // same `slots` elements and one value from the lane's table, whose
            // index `new_indexed` and the merge's index arithmetic both keep
            // inside that table, and writes only `coeffs[slot]`. Input and
            // output are distinct allocations.
            let _ = unsafe { builder.launch(CudaKernelContext::launch_config(slots)) }?;
            context.stream().synchronize()?;
        }
        self.coeffs = coeffs;
        self.coeff_index = context.alloc_u16_unset(1)?;
        self.luts = [
            DeviceCoeffLut::placeholder(context)?,
            DeviceCoeffLut::placeholder(context)?,
        ];
        self.indexed = false;
        Ok(())
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "the device-side construction hands over one buffer per matrix column"
    )]
    pub(crate) fn from_device_parts(
        context: &CudaKernelContext,
        rows: CudaSlice<u32>,
        cols: CudaSlice<u32>,
        val_coeff: DeviceFrVec,
        prev_val: CudaSlice<u64>,
        next_val: CudaSlice<u64>,
        coeffs: DeviceCoeffs,
        wa_scale: DeviceFrVec,
        coeff_width: usize,
        entries: usize,
    ) -> Result<Self, CudaError> {
        let (coeffs, coeff_index, luts, indexed) = match coeffs {
            DeviceCoeffs::Direct(coeffs) => (
                coeffs,
                context.alloc_u16_unset(1)?,
                [
                    DeviceCoeffLut::placeholder(context)?,
                    DeviceCoeffLut::placeholder(context)?,
                ],
                false,
            ),
            DeviceCoeffs::Indexed { index, luts } => (context.alloc(1)?, index, luts, true),
        };
        Ok(Self {
            rows,
            cols,
            val_coeff,
            prev_val,
            next_val,
            coeffs,
            coeff_index,
            luts,
            indexed,
            wa_scale,
            coeff_width,
            entries,
            rounds_bound: 0,
        })
    }

    pub const fn len(&self) -> usize {
        self.entries
    }

    pub const fn is_empty(&self) -> bool {
        self.entries == 0
    }

    pub const fn rounds_bound(&self) -> usize {
        self.rounds_bound
    }

    pub fn to_host(&self, context: &CudaKernelContext) -> Result<Vec<MatrixEntry>, CudaError> {
        let rows = context.download_u32(&self.rows)?;
        let cols = context.download_u32(&self.cols)?;
        let val_coeff = self.val_coeff.to_host()?;
        let prev_val = context.download_u64(&self.prev_val)?;
        let next_val = context.download_u64(&self.next_val)?;
        let coeffs = self.host_coeffs(context)?;
        let mut out = Vec::with_capacity(self.entries);
        for index in 0..self.entries {
            let mut entry_coeffs = [Fr::default(); MAX_COEFFS];
            for lane in 0..self.coeff_width {
                entry_coeffs[lane] = coeffs[index * self.coeff_width + lane];
            }
            out.push(MatrixEntry {
                row: rows[index],
                col: cols[index],
                val_coeff: val_coeff[index],
                prev_val: prev_val[index],
                next_val: next_val[index],
                coeffs: entry_coeffs,
            });
        }
        Ok(out)
    }

    fn segments(&self, context: &CudaKernelContext) -> Result<(usize, Segments), CudaError> {
        let entries = CudaKernelContext::count_of(self.entries)?;
        let mut flags = context.alloc_u32(self.entries)?;
        let mut builder = context.stream().launch_builder(context.rwm_segment_flags());
        let _ = builder.arg(&self.rows);
        let _ = builder.arg(&entries);
        let _ = builder.arg(&mut flags);
        // SAFETY: thread `index < entries` reads `rows[index]` and, when
        // `index > 0`, `rows[index - 1]`, then writes only `flags[index]`. Both
        // buffers hold `entries` u32s and are distinct allocations.
        let _ = unsafe { builder.launch(CudaKernelContext::launch_config(entries)) }?;
        context.stream().synchronize()?;

        let (device_ranks, segment_count) =
            context.exclusive_scan_with_total_u32(&flags, self.entries)?;

        let mut seg = Segments {
            start: context.alloc_u32(segment_count)?,
            even_end: context.alloc_u32(segment_count)?,
            end: context.alloc_u32(segment_count)?,
            pair: context.alloc_u32(segment_count)?,
        };
        let mut builder = context
            .stream()
            .launch_builder(context.rwm_segment_bounds());
        let _ = builder.arg(&self.rows);
        let _ = builder.arg(&flags);
        let _ = builder.arg(&device_ranks);
        let _ = builder.arg(&entries);
        let _ = builder.arg(&mut seg.start);
        let _ = builder.arg(&mut seg.even_end);
        let _ = builder.arg(&mut seg.end);
        let _ = builder.arg(&mut seg.pair);
        // SAFETY: thread `index < entries` returns unless it owns a segment head
        // (`flags[index]`), in which case it writes the four `seg_*` buffers at
        // `ranks[index]`, which is `< segment_count` because `ranks` is the
        // exclusive scan of `flags` and this index is a head. Its forward scans
        // are bounded by `entries`. One head per segment, so writes are
        // uncontended and every slot is written.
        let _ = unsafe { builder.launch(CudaKernelContext::launch_config(entries)) }?;
        context.stream().synchronize()?;
        Ok((segment_count, seg))
    }

    pub fn to_address_major(
        &self,
        context: &CudaKernelContext,
        val_init: &[Fr],
    ) -> Result<super::address_major_matrix::DeviceAddressMajorMatrix, CudaError> {
        if self.entries > val_init.len() {
            return Err(CudaError::InvariantViolation {
                reason: "transposing a read-write matrix needs its cycle variables fully bound",
            });
        }
        self.require_direct()?;
        super::address_major_matrix::DeviceAddressMajorMatrix::from_parts(
            context,
            context.clone_u32(&self.rows)?,
            context.clone_u32(&self.cols)?,
            &self.val_coeff,
            &self.prev_val,
            &self.next_val,
            &self.coeffs,
            val_init,
            &self.wa_scale,
            self.coeff_width,
            self.entries,
        )
    }

    pub fn quadratic_coeffs<F: jolt_field::Field>(
        &self,
        context: &CudaKernelContext,
        inc: &DeviceFrVec,
        eq: &super::split_eq::DeviceSplitEq<F>,
    ) -> Result<[F; 2], CudaError> {
        if self.entries == 0 {
            return Ok([F::from_u64(0), F::from_u64(0)]);
        }
        let (segment_count, seg) = self.segments(context)?;
        let segments = CudaKernelContext::count_of(segment_count)?;

        let e_in_len = CudaKernelContext::count_of(eq.e_in_len())?;
        let num_x_in_bits = eq.e_in_len().max(1).ilog2();
        let e_in = eq.e_in_current();
        let e_out = eq.e_out_current();

        let blocks = segments.div_ceil(BLOCK).max(1);
        let mut partials = context.alloc(2 * blocks as usize)?;
        let width = CudaKernelContext::count_of(self.coeff_width)?;

        let indexed = u32::from(self.indexed);
        let mut builder = context.stream().launch_builder(context.rwm_message());
        let _ = builder.arg(&self.cols);
        let _ = builder.arg(self.val_coeff.limbs());
        let _ = builder.arg(&self.prev_val);
        let _ = builder.arg(&self.next_val);
        let _ = builder.arg(self.coeffs.limbs());
        let _ = builder.arg(&width);
        let _ = builder.arg(&self.coeff_index);
        let _ = builder.arg(&indexed);
        let _ = builder.arg(self.luts[0].values.limbs());
        let _ = builder.arg(self.luts[1].values.limbs());
        let _ = builder.arg(&seg.start);
        let _ = builder.arg(&seg.even_end);
        let _ = builder.arg(&seg.end);
        let _ = builder.arg(&seg.pair);
        let _ = builder.arg(&segments);
        let _ = builder.arg(inc.limbs());
        let _ = builder.arg(e_in.limbs());
        let _ = builder.arg(&e_in_len);
        let _ = builder.arg(e_out.limbs());
        let _ = builder.arg(&num_x_in_bits);
        let _ = builder.arg(self.wa_scale.limbs());
        let _ = builder.arg(partials.limbs_mut());
        // SAFETY: thread `seg < segments` reads matrix data only between
        // `seg_start[seg]` and `seg_end[seg]` (confined to `entries` by the
        // bounds kernel), `inc[2 * pair]` and `inc[2 * pair + 1]` (in range
        // because `pair` indexes the half-sized cycle domain `inc` was built
        // over), `e_in[pair & mask]` and `e_out[pair >> num_x_in_bits]` — both
        // bounded by construction of `num_x_in_bits` from `e_in`'s length. When
        // `indexed` it reads each lane's coefficient through
        // `luts[lane][coeff_index[slot]]` instead of `coeffs[slot]`, and every
        // index in `coeff_index` addresses its own lane's table: construction
        // range-checks them and the merge's `(odd << bits) | even` stays inside
        // the squared table. It
        // writes only `partials[lane * gridDim.x + blockIdx.x]`, one slot per
        // (lane, block) of `2 * blocks`. Shared memory is `BLOCK * LIMBS` u64s,
        // matching `shared_mem_bytes`, and every `__syncthreads()` is reached by
        // all threads in the block because the reduction sits outside the
        // `seg < segments` guard.
        let _ = unsafe {
            builder.launch(cudarc::driver::LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: BLOCK * LIMBS as u32 * size_of::<u64>() as u32,
            })
        }?;
        context.stream().synchronize()?;

        let totals = super::primitives::reduce_lanes(context, partials, 2, blocks)?;
        let host = totals.to_host()?;
        let convert = |value: Fr| {
            super::device::fr_into(value).ok_or(CudaError::NotImplemented {
                kernel: "CUDA kernels support only the BN254 scalar field",
            })
        };
        Ok([convert(host[0])?, convert(host[1])?])
    }

    pub fn bind<F: jolt_field::Field>(
        &mut self,
        context: &CudaKernelContext,
        challenge: F,
    ) -> Result<(), CudaError> {
        if self.entries == 0 {
            self.rounds_bound += 1;
            return Ok(());
        }
        let challenge = super::device::require_fr(challenge)?;
        if self.indexed && self.luts.iter().any(DeviceCoeffLut::saturated) {
            self.deref_coeffs(context)?;
        }
        let (segment_count, seg) = self.segments(context)?;
        let segments = CudaKernelContext::count_of(segment_count)?;

        let mut counts = context.alloc_u32(segment_count)?;
        let mut builder = context.stream().launch_builder(context.rwm_count());
        let _ = builder.arg(&self.cols);
        let _ = builder.arg(&seg.start);
        let _ = builder.arg(&seg.even_end);
        let _ = builder.arg(&seg.end);
        let _ = builder.arg(&segments);
        let _ = builder.arg(&mut counts);
        // SAFETY: thread `seg < segments` reads its own bounds from the four
        // `seg_*` buffers (each `segments` u32s) and walks `cols` only between
        // `seg_start[seg]` and `seg_end[seg]`, which the bounds kernel confined
        // to `entries`. It writes only `counts[seg]`.
        let _ = unsafe { builder.launch(CudaKernelContext::launch_config(segments)) }?;
        context.stream().synchronize()?;

        let (device_offsets, bound_len) =
            context.exclusive_scan_with_total_u32(&counts, segment_count)?;

        let mut out_rows = context.alloc_u32(bound_len)?;
        let mut out_cols = context.alloc_u32(bound_len)?;
        let mut out_val = context.alloc(bound_len)?;
        let mut out_prev = context.alloc_u64(bound_len)?;
        let mut out_next = context.alloc_u64(bound_len)?;
        let mut out_coeffs = context.alloc(if self.indexed {
            1
        } else {
            bound_len * self.coeff_width
        })?;
        let mut out_coeff_index = context.alloc_u16_unset(if self.indexed {
            bound_len * self.coeff_width
        } else {
            1
        })?;
        let device_challenge = context.upload(&[challenge])?;
        let width = CudaKernelContext::count_of(self.coeff_width)?;
        let indexed = u32::from(self.indexed);
        let lut_bits = [self.luts[0].bits(), self.luts[1].bits()];

        let mut builder = context.stream().launch_builder(context.rwm_merge());
        let _ = builder.arg(&self.cols);
        let _ = builder.arg(self.val_coeff.limbs());
        let _ = builder.arg(&self.prev_val);
        let _ = builder.arg(&self.next_val);
        let _ = builder.arg(self.coeffs.limbs());
        let _ = builder.arg(&width);
        let _ = builder.arg(&self.coeff_index);
        let _ = builder.arg(&indexed);
        let _ = builder.arg(&lut_bits[0]);
        let _ = builder.arg(&lut_bits[1]);
        let _ = builder.arg(&seg.start);
        let _ = builder.arg(&seg.even_end);
        let _ = builder.arg(&seg.end);
        let _ = builder.arg(&seg.pair);
        let _ = builder.arg(&device_offsets);
        let _ = builder.arg(&segments);
        let _ = builder.arg(device_challenge.limbs());
        let _ = builder.arg(&mut out_rows);
        let _ = builder.arg(&mut out_cols);
        let _ = builder.arg(out_val.limbs_mut());
        let _ = builder.arg(&mut out_prev);
        let _ = builder.arg(&mut out_next);
        let _ = builder.arg(out_coeffs.limbs_mut());
        let _ = builder.arg(&mut out_coeff_index);
        // SAFETY: thread `seg < segments` reads inputs only between
        // `seg_start[seg]` and `seg_end[seg]` (confined to `entries` by the
        // bounds kernel) plus the single-element `challenge`, and writes exactly
        // `counts[seg]` output slots starting at `offsets[seg]`. Since `offsets`
        // is the exclusive scan of `counts` and `bound_len` is their total, the
        // per-segment output ranges are disjoint and all inside every `out_*`
        // buffer. `out_coeffs` is indexed `k * coeff_width + lane` over
        // `bound_len * coeff_width` elements, and when `indexed` the same
        // indexing applies to `out_coeff_index` instead, whose entries stay
        // inside the squared table because `(odd << bits) | even` with
        // `even, odd < 1 << bits` is below `1 << 2 * bits`. Inputs and outputs
        // are distinct allocations.
        let _ = unsafe { builder.launch(CudaKernelContext::launch_config(segments)) }?;
        context.stream().synchronize()?;

        self.rows = out_rows;
        self.cols = out_cols;
        self.val_coeff = out_val;
        self.prev_val = out_prev;
        self.next_val = out_next;
        self.coeffs = out_coeffs;
        self.coeff_index = out_coeff_index;
        self.entries = bound_len;
        self.rounds_bound += 1;
        if self.indexed {
            let bound: Vec<DeviceCoeffLut> = self
                .luts
                .iter()
                .map(|lut| lut.bind(context, challenge))
                .collect::<Result<_, _>>()?;
            self.luts = bound.try_into().map_err(|_| CudaError::LengthMismatch {
                expected: MAX_COEFFS,
                got: 0,
            })?;
        }
        Ok(())
    }

    pub(crate) fn concatenate_windows(
        context: &CudaKernelContext,
        windows: &[(usize, Self)],
    ) -> Result<Self, CudaError> {
        let (_, first) = windows.first().ok_or(CudaError::InvariantViolation {
            reason: "a windowed read-write matrix needs at least one window",
        })?;
        let coeff_width = first.coeff_width;
        let rounds_bound = first.rounds_bound;
        let mut entries: Vec<MatrixEntry> = Vec::new();
        for (row, (ordinal, window)) in windows.iter().enumerate() {
            if window.coeff_width != coeff_width || window.rounds_bound != rounds_bound {
                return Err(CudaError::InvariantViolation {
                    reason: "every read-write matrix window collapses from the same shape and the \
                             same number of bound rounds",
                });
            }
            let device = context_for(*ordinal).ok_or(CudaError::InvariantViolation {
                reason: "a read-write matrix window names an absent device",
            })?;
            window.require_direct()?;
            let mut host = window.to_host(device)?;
            if host.iter().any(|entry| entry.row != 0) {
                return Err(CudaError::InvariantViolation {
                    reason: "a read-write matrix window must bind its local cycle domain down to \
                             one row before it collapses",
                });
            }
            let row = u32::try_from(row).map_err(|_| CudaError::InvariantViolation {
                reason: "a read-write matrix window index exceeds the row word",
            })?;
            for entry in &mut host {
                entry.row = row;
            }
            entries.append(&mut host);
        }
        let wa_scale = first.wa_scale.first()?;
        Self::upload_entries(context, &entries, coeff_width, wa_scale, rounds_bound)
    }
}

pub(crate) struct CycleShard<F: Field> {
    pub(crate) ordinal: usize,
    pub(crate) matrix: DeviceReadWriteMatrix,
    pub(crate) inc: DeviceFrVec,
    pub(crate) eq: DeviceSplitEq<F>,
}

pub(crate) struct ShardedReadWriteMatrix<F: Field> {
    shards: Vec<CycleShard<F>>,
    collapsed: Option<(DeviceReadWriteMatrix, DeviceFrVec)>,
    local_rounds: usize,
    tail_rounds: usize,
}

impl<F: Field> ShardedReadWriteMatrix<F> {
    #[cfg(feature = "allocative")]
    pub(crate) fn device_bytes(&self) -> usize {
        self.shards
            .iter()
            .map(|shard| {
                shard.matrix.device_bytes() + shard.inc.device_bytes() + shard.eq.device_bytes()
            })
            .sum::<usize>()
            + self.collapsed.as_ref().map_or(0, |(matrix, inc)| {
                matrix.device_bytes() + inc.device_bytes()
            })
    }

    pub(crate) fn new(shards: Vec<CycleShard<F>>, log_t: usize) -> Result<Self, CudaError> {
        let count = shards.len();
        if count == 0 || !count.is_power_of_two() {
            return Err(CudaError::InvariantViolation {
                reason: "a sharded read-write matrix needs a power-of-two shard count",
            });
        }
        let tail_rounds = count.trailing_zeros() as usize;
        if tail_rounds > log_t {
            return Err(CudaError::InvariantViolation {
                reason: "a sharded read-write matrix cannot split more windows than cycle rounds",
            });
        }
        if count == 1 {
            let shard = shards
                .into_iter()
                .next()
                .ok_or(CudaError::InvariantViolation {
                    reason: "a single-shard read-write matrix lost its state",
                })?;
            return Ok(Self {
                shards: Vec::new(),
                collapsed: Some((shard.matrix, shard.inc)),
                local_rounds: log_t,
                tail_rounds: 0,
            });
        }
        Ok(Self {
            shards,
            collapsed: None,
            local_rounds: log_t - tail_rounds,
            tail_rounds,
        })
    }

    pub(crate) fn quadratic_coeffs(
        &self,
        whole_eq: &DeviceSplitEq<F>,
    ) -> Result<[F; 2], CudaError> {
        if let Some((matrix, inc)) = &self.collapsed {
            let context = context_for(0).ok_or(absent())?;
            return matrix.quadratic_coeffs(context, inc, whole_eq);
        }
        let tasks: Vec<DeviceTask<'_, [F; 2], CudaError>> = self
            .shards
            .iter()
            .map(|shard| {
                let task: DeviceTask<'_, [F; 2], CudaError> = Box::new(move || {
                    let context = context_for(shard.ordinal).ok_or(absent())?;
                    shard
                        .matrix
                        .quadratic_coeffs(context, &shard.inc, &shard.eq)
                });
                task
            })
            .collect();
        let mut total = [F::zero(), F::zero()];
        for part in fan_out(tasks)? {
            total[0] += part[0];
            total[1] += part[1];
        }
        Ok(total)
    }

    pub(crate) fn bind(&mut self, challenge: F, bound: usize) -> Result<(), CudaError> {
        if let Some((matrix, inc)) = &mut self.collapsed {
            let context = context_for(0).ok_or(absent())?;
            matrix.bind(context, challenge)?;
            *inc = context.bind(inc, require_fr(challenge)?, BindingOrder::LowToHigh)?;
            return Ok(());
        }
        let tasks: Vec<DeviceTask<'_, (), CudaError>> = self
            .shards
            .iter_mut()
            .map(|shard| {
                let task: DeviceTask<'_, (), CudaError> = Box::new(move || {
                    let context = context_for(shard.ordinal).ok_or(absent())?;
                    shard.matrix.bind(context, challenge)?;
                    shard.inc = context.bind(
                        &shard.inc,
                        require_fr(challenge)?,
                        BindingOrder::LowToHigh,
                    )?;
                    shard.eq.bind(challenge);
                    Ok(())
                });
                task
            })
            .collect();
        let _ = fan_out(tasks)?;
        if bound + 1 == self.local_rounds {
            self.collapse()?;
        }
        Ok(())
    }

    fn collapse(&mut self) -> Result<(), CudaError> {
        let context = context_for(0).ok_or(absent())?;
        let shards = std::mem::take(&mut self.shards);
        let mut scalars = Vec::with_capacity(shards.len());
        for shard in &shards {
            if shard.inc.len() != 1 {
                return Err(CudaError::LengthMismatch {
                    expected: 1,
                    got: shard.inc.len(),
                });
            }
            scalars.push(shard.inc.first()?);
        }
        let windows: Vec<(usize, DeviceReadWriteMatrix)> = shards
            .into_iter()
            .map(|shard| (shard.ordinal, shard.matrix))
            .collect();
        let matrix = DeviceReadWriteMatrix::concatenate_windows(context, &windows)?;
        let inc = context.upload(&scalars)?;
        self.collapsed = Some((matrix, inc));
        Ok(())
    }

    pub(crate) fn take_parts(&mut self) -> Option<(DeviceReadWriteMatrix, DeviceFrVec)> {
        self.collapsed.take()
    }
}

const fn absent() -> CudaError {
    CudaError::InvariantViolation {
        reason: "a sharded read-write matrix window names an absent device",
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test module: device operations and fixture errors fail loudly"
)]
mod tests {
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_poly::{BindingOrder, GruenSplitEqPolynomial, Polynomial};
    use proptest::prelude::*;

    use super::super::context::shared_context;
    use super::super::split_eq::DeviceSplitEq;
    use super::super::testing::{arb_point, fr, ram_matrix_cells};
    use super::{CoeffTables, DeviceReadWriteMatrix, LutEntry, MatrixEntry, MAX_COEFFS};
    use crate::cuda::registers_read_write::device_rows::register_coeff_tables;
    use crate::optimized::registers_read_write::sparse::CoeffLut;
    use crate::optimized::rw_matrix::{CycleMajorEntry, CycleMajorMatrix};

    const LOG_T: usize = 6;
    const RAM_K: usize = 32;
    const REGISTER_COEFF_WIDTH: usize = 2;
    const LUT_COLUMNS: u64 = 4;
    const INDEXED_ROUNDS: usize = 4;

    const MERGE_CASE_ROWS: [&[u32]; 8] = [
        &[1, 2],
        &[1, 2],
        &[0, 3],
        &[],
        &[],
        &[0, 1, 2],
        &[0, 2],
        &[1, 2],
    ];

    struct RamFixture {
        device: Vec<MatrixEntry>,
        optimized: Vec<CycleMajorEntry<Fr>>,
    }

    fn ram_fixture() -> RamFixture {
        let cells = ram_matrix_cells(LOG_T, RAM_K);
        let device = cells
            .iter()
            .map(|cell| MatrixEntry {
                row: cell.row as u32,
                col: cell.col as u32,
                val_coeff: Fr::from_u64(cell.prev_val),
                prev_val: cell.prev_val,
                next_val: cell.next_val,
                coeffs: [Fr::from_u64(1), Fr::from_u64(0)],
            })
            .collect();
        let optimized = cells
            .iter()
            .map(|cell| CycleMajorEntry {
                row: cell.row,
                col: cell.col,
                prev_val: cell.prev_val,
                next_val: cell.next_val,
                val: Fr::from_u64(cell.prev_val),
                ra: Fr::from_u64(1),
            })
            .collect();
        RamFixture { device, optimized }
    }

    fn entry_tuples(entries: &[MatrixEntry]) -> Vec<(u32, u32, Fr, u64, u64, Fr)> {
        entries
            .iter()
            .map(|entry| {
                (
                    entry.row,
                    entry.col,
                    entry.val_coeff,
                    entry.prev_val,
                    entry.next_val,
                    entry.coeffs[0],
                )
            })
            .collect()
    }

    fn optimized_tuples(entries: &[CycleMajorEntry<Fr>]) -> Vec<(u32, u32, Fr, u64, u64, Fr)> {
        entries
            .iter()
            .map(|entry| {
                (
                    entry.row as u32,
                    entry.col as u32,
                    entry.val,
                    entry.prev_val,
                    entry.next_val,
                    entry.ra,
                )
            })
            .collect()
    }

    #[test]
    fn device_ram_matrix_bind_matches_optimized_round_for_round() {
        let Some(context) = shared_context() else {
            return;
        };
        let fixture = ram_fixture();
        let mut expected = CycleMajorMatrix {
            entries: fixture.optimized,
        };
        let mut got = DeviceReadWriteMatrix::new(context, &fixture.device, 1, Some(fr(101)))
            .expect("device RAM matrix");

        assert_eq!(
            entry_tuples(&got.to_host(context).expect("download")),
            optimized_tuples(&expected.entries),
            "seeded entry sets disagree before any bind",
        );

        for round in 0..LOG_T {
            let challenge = fr(17 + round as u64);
            expected.bind(challenge);
            got.bind(context, challenge).expect("device bind");
            assert_eq!(
                entry_tuples(&got.to_host(context).expect("download")),
                optimized_tuples(&expected.entries),
                "entry sets diverged after binding round {round}",
            );
        }
    }

    #[test]
    fn sharded_ram_matrix_matches_the_whole_domain_round_for_round() {
        let Some(context) = shared_context() else {
            return;
        };
        let cycles = 1usize << LOG_T;
        let gamma = fr(103);
        let point: Vec<Fr> = (0..LOG_T).map(|i| fr(61 + i as u64)).collect();
        let inc_host: Vec<Fr> = (0..cycles).map(|j| fr(j as u64 * 5 + 3)).collect();

        for shards in [2usize, 4] {
            let fixture = ram_fixture();
            let mut whole = DeviceReadWriteMatrix::new(context, &fixture.device, 1, Some(gamma))
                .expect("whole device matrix");
            let mut whole_inc = context.upload(&inc_host).expect("upload inc");
            let mut whole_eq = DeviceSplitEq::<Fr>::new(context, &point, BindingOrder::LowToHigh)
                .expect("whole split-eq");

            let len = cycles / shards;
            let windows: Vec<super::CycleShard<Fr>> = (0..shards)
                .map(|shard| {
                    let base = (shard * len) as u32;
                    let entries: Vec<MatrixEntry> = fixture
                        .device
                        .iter()
                        .filter(|entry| (entry.row as usize) / len == shard)
                        .map(|entry| MatrixEntry {
                            row: entry.row - base,
                            ..*entry
                        })
                        .collect();
                    super::CycleShard {
                        ordinal: 0,
                        matrix: DeviceReadWriteMatrix::new(context, &entries, 1, Some(gamma))
                            .expect("window matrix"),
                        inc: context
                            .upload(&inc_host[shard * len..(shard + 1) * len])
                            .expect("upload window inc"),
                        eq: DeviceSplitEq::<Fr>::new_window(
                            context,
                            &point,
                            BindingOrder::LowToHigh,
                            shard,
                            shards,
                        )
                        .expect("window split-eq"),
                    }
                })
                .collect();
            let mut got =
                super::ShardedReadWriteMatrix::new(windows, LOG_T).expect("sharded matrix");
            let mut got_eq = DeviceSplitEq::<Fr>::new(context, &point, BindingOrder::LowToHigh)
                .expect("tail split-eq");

            for round in 0..LOG_T {
                let want: [Fr; 2] = whole
                    .quadratic_coeffs(context, &whole_inc, &whole_eq)
                    .expect("whole quadratic coeffs");
                let have: [Fr; 2] = got.quadratic_coeffs(&got_eq).expect("sharded coeffs");
                assert_eq!(
                    have, want,
                    "shards={shards} round {round}: a cycle window's segment sums must add to the \
                     whole domain's, with e_out sliced to the window and e_in shared",
                );

                let challenge = fr(19 + round as u64);
                whole.bind(context, challenge).expect("whole bind");
                whole_inc = context
                    .bind(&whole_inc, challenge, BindingOrder::LowToHigh)
                    .expect("bind whole inc");
                whole_eq.bind(challenge);
                got.bind(challenge, round).expect("sharded bind");
                got_eq.bind(challenge);
            }

            let (matrix, inc) = got.take_parts().expect("collapsed parts");
            assert_eq!(
                entry_tuples(&matrix.to_host(context).expect("download collapsed")),
                entry_tuples(&whole.to_host(context).expect("download whole")),
                "shards={shards}: the collapsed entry set diverged",
            );
            assert_eq!(
                inc.first().expect("collapsed inc"),
                whole_inc.first().expect("whole inc"),
                "shards={shards}: the collapsed increment diverged",
            );
            assert!(
                !matrix.is_empty(),
                "shards={shards}: an empty collapsed matrix would satisfy the comparison trivially",
            );
        }
    }

    struct LutFixture {
        direct: Vec<MatrixEntry>,
        indexed: Vec<LutEntry>,
        tables: CoeffTables,
    }

    fn lut_fixture(seed: u64, gamma: Fr) -> LutFixture {
        let tables = register_coeff_tables(gamma);
        let mut direct = Vec::new();
        let mut indexed = Vec::new();
        for row in 0..1u32 << LOG_T {
            let mix = (u64::from(row) + 1)
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add(seed);
            let mut cols: Vec<u32> = match MERGE_CASE_ROWS.get(row as usize) {
                Some(cols) => cols.to_vec(),
                None => (0..mix % 4)
                    .map(|slot| ((mix >> (5 * slot + 3)) % LUT_COLUMNS) as u32)
                    .collect(),
            };
            cols.sort_unstable();
            cols.dedup();
            for (slot, col) in cols.into_iter().enumerate() {
                let ra = ((mix >> (23 + 3 * slot)) % 4) as u16;
                let wa = if ra == 0 {
                    1
                } else {
                    ((mix >> (41 + slot)) % 2) as u16
                };
                let prev_val = (mix >> 11) % 97 + u64::from(row);
                let next_val = if wa == 1 { prev_val + 13 } else { prev_val };
                direct.push(MatrixEntry {
                    row,
                    col,
                    val_coeff: Fr::from_u64(prev_val),
                    prev_val,
                    next_val,
                    coeffs: [tables.values[0][ra as usize], tables.values[1][wa as usize]],
                });
                indexed.push(LutEntry {
                    row,
                    col,
                    val_coeff: Fr::from_u64(prev_val),
                    prev_val,
                    next_val,
                    coeff_index: [ra, wa],
                });
            }
        }
        assert!(
            direct.len() > (1 << LOG_T),
            "the fixture must place several entries per cycle pair to exercise the merge",
        );
        LutFixture {
            direct,
            indexed,
            tables,
        }
    }

    fn indexed_table_lengths(round: usize) -> Option<[usize; MAX_COEFFS]> {
        (round < INDEXED_ROUNDS).then(|| [1usize << (2 << round), 1usize << (1 << round)])
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(4))]
        #[test]
        fn indexed_coefficients_match_direct_field_coefficients_round_for_round(
            seed in any::<u64>(),
            gamma in any::<u64>().prop_map(fr),
            point in arb_point(LOG_T),
            challenges in arb_point(LOG_T),
        ) {
            let Some(context) = shared_context() else { return Ok(()); };
            let fixture = lut_fixture(seed, gamma);
            let mut expected =
                DeviceReadWriteMatrix::new(context, &fixture.direct, REGISTER_COEFF_WIDTH, None)
                    .expect("direct matrix");
            let mut got =
                DeviceReadWriteMatrix::new_indexed(context, &fixture.indexed, &fixture.tables)
                    .expect("indexed matrix");

            let cycles = 1usize << LOG_T;
            let inc_host: Vec<Fr> = (0..cycles).map(|j| fr(j as u64 * 5 + 3)).collect();
            let mut expected_inc = context.upload(&inc_host).expect("upload direct inc");
            let mut got_inc = context.upload(&inc_host).expect("upload indexed inc");
            let mut expected_eq =
                DeviceSplitEq::<Fr>::new(context, &point, BindingOrder::LowToHigh)
                    .expect("direct split-eq");
            let mut got_eq = DeviceSplitEq::<Fr>::new(context, &point, BindingOrder::LowToHigh)
                .expect("indexed split-eq");

            for (round, &challenge) in challenges.iter().enumerate() {
                let lengths = got
                    .coeff_tables()
                    .expect("indexed tables")
                    .map(|tables| tables.values.map(|lane| lane.len()));
                prop_assert_eq!(
                    lengths,
                    indexed_table_lengths(round),
                    "round {}: the coefficient representation left its schedule",
                    round,
                );
                prop_assert_eq!(
                    got.to_host(context).expect("download indexed"),
                    expected.to_host(context).expect("download direct"),
                    "round {}: entry sets diverged",
                    round,
                );
                let want: [Fr; 2] = expected
                    .quadratic_coeffs(context, &expected_inc, &expected_eq)
                    .expect("direct quadratic coeffs");
                let have: [Fr; 2] = got
                    .quadratic_coeffs(context, &got_inc, &got_eq)
                    .expect("indexed quadratic coeffs");
                prop_assert_eq!(
                    have,
                    want,
                    "round {}: quadratic coefficients diverged",
                    round,
                );

                expected.bind(context, challenge).expect("direct bind");
                got.bind(context, challenge).expect("indexed bind");
                expected_inc = context
                    .bind(&expected_inc, challenge, BindingOrder::LowToHigh)
                    .expect("bind direct inc");
                got_inc = context
                    .bind(&got_inc, challenge, BindingOrder::LowToHigh)
                    .expect("bind indexed inc");
                expected_eq.bind(challenge);
                got_eq.bind(challenge);
            }
        }
    }

    #[test]
    fn indexed_coefficient_tables_square_like_the_optimized_lut() {
        let Some(context) = shared_context() else {
            return;
        };
        let gamma = fr(101);
        let fixture = lut_fixture(11, gamma);
        let mut got =
            DeviceReadWriteMatrix::new_indexed(context, &fixture.indexed, &fixture.tables)
                .expect("indexed matrix");
        let mut expected: Vec<CoeffLut<Fr>> = fixture
            .tables
            .values
            .iter()
            .cloned()
            .map(CoeffLut::new)
            .collect();

        for round in 0..INDEXED_ROUNDS {
            let tables = got
                .coeff_tables()
                .expect("indexed tables")
                .expect("the matrix must still index its coefficients");
            for (lane, want) in expected.iter().enumerate() {
                assert_eq!(
                    tables.values[lane], want.values,
                    "round {round} lane {lane}: the device table diverged from the optimized LUT",
                );
            }
            let challenge = fr(29 + round as u64);
            got.bind(context, challenge).expect("indexed bind");
            if round + 1 < INDEXED_ROUNDS {
                for lut in &mut expected {
                    lut.bind(challenge);
                }
            }
        }
        assert!(
            got.coeff_tables().expect("indexed tables").is_none(),
            "one more squaring would overflow the u16 index, so the entries must hold field values",
        );
    }

    #[test]
    fn an_indexed_matrix_rejects_a_table_whose_zero_slot_is_not_zero() {
        let Some(context) = shared_context() else {
            return;
        };
        let gamma = fr(101);
        let fixture = lut_fixture(3, gamma);
        let mut tables = register_coeff_tables(gamma);
        tables.values[0][0] = Fr::from_u64(7);
        assert!(
            DeviceReadWriteMatrix::new_indexed(context, &fixture.indexed, &tables).is_err(),
            "index 0 is the absent merge partner, so its value must be zero",
        );
    }

    #[test]
    fn device_ram_quadratic_coeffs_match_optimized_round_for_round() {
        let Some(context) = shared_context() else {
            return;
        };
        let fixture = ram_fixture();
        let gamma = fr(103);
        let mut expected = CycleMajorMatrix {
            entries: fixture.optimized,
        };
        let mut got = DeviceReadWriteMatrix::new(context, &fixture.device, 1, Some(gamma))
            .expect("device RAM matrix");

        let inc_host: Vec<Fr> = (0..1usize << LOG_T).map(|j| fr(j as u64 * 5 + 3)).collect();
        let mut expected_inc = Polynomial::new(inc_host.clone());
        let mut got_inc = context.upload(&inc_host).expect("upload inc");

        let point: Vec<Fr> = (0..LOG_T).map(|i| fr(61 + i as u64)).collect();
        let mut expected_eq = GruenSplitEqPolynomial::new(&point, BindingOrder::LowToHigh);
        let mut got_eq = DeviceSplitEq::<Fr>::new(context, &point, BindingOrder::LowToHigh)
            .expect("device split-eq");

        for round in 0..LOG_T {
            let e_in = expected_eq.e_in_current();
            let e_out = expected_eq.e_out_current();
            let in_bits = e_in.len().trailing_zeros() as usize;
            let in_mask = e_in.len() - 1;
            let expected_coeffs = expected.quadratic_coefficients(
                |pair| e_out[pair >> in_bits] * e_in[pair & in_mask],
                &expected_inc,
                gamma,
            );
            let got_coeffs: [Fr; 2] = got
                .quadratic_coeffs(context, &got_inc, &got_eq)
                .expect("device quadratic coeffs");
            assert_eq!(
                got_coeffs, expected_coeffs,
                "RAM quadratic coefficients diverged at round {round}",
            );

            let challenge = fr(19 + round as u64);
            expected.bind(challenge);
            got.bind(context, challenge).expect("device bind");
            expected_inc.bind_with_order(challenge, BindingOrder::LowToHigh);
            got_inc = context
                .bind(&got_inc, challenge, BindingOrder::LowToHigh)
                .expect("bind device inc");
            expected_eq.bind(challenge);
            got_eq.bind(challenge);
        }
    }
}
