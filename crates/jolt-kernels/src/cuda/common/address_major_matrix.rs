use cudarc::driver::LaunchConfig;
use jolt_field::JoltField;
use jolt_poly::BindingOrder;

use cudarc::driver::{CudaSlice, PushKernelArg};
use jolt_field::Fr;

use super::context::{CudaKernelContext, BLOCK};
use super::device::{DeviceFrVec, LIMBS};
use super::error::CudaError;

struct Segments {
    start: CudaSlice<u32>,
    even_end: CudaSlice<u32>,
    end: CudaSlice<u32>,
    pair: CudaSlice<u32>,
}

pub struct DeviceAddressMajorMatrix {
    rows: CudaSlice<u32>,
    cols: CudaSlice<u32>,
    val_coeff: DeviceFrVec,
    prev_val: DeviceFrVec,
    next_val: DeviceFrVec,
    coeffs: DeviceFrVec,
    val_init: DeviceFrVec,
    wa_scale: DeviceFrVec,
    coeff_width: usize,
    entries: usize,
    rounds_bound: usize,
}

impl DeviceAddressMajorMatrix {
    #[cfg(feature = "allocative")]
    pub fn device_bytes(&self) -> usize {
        (self.rows.len() + self.cols.len()) * size_of::<u32>()
            + self.val_coeff.device_bytes()
            + self.prev_val.device_bytes()
            + self.next_val.device_bytes()
            + self.coeffs.device_bytes()
            + self.val_init.device_bytes()
            + self.wa_scale.device_bytes()
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "the cycle-major transpose hands over one buffer per matrix column"
    )]
    pub(crate) fn from_parts(
        context: &CudaKernelContext,
        rows: CudaSlice<u32>,
        cols: CudaSlice<u32>,
        val_coeff: &DeviceFrVec,
        prev_val: &CudaSlice<u64>,
        next_val: &CudaSlice<u64>,
        coeffs: &DeviceFrVec,
        val_init: &[Fr],
        wa_scale: &DeviceFrVec,
        coeff_width: usize,
        entries: usize,
    ) -> Result<Self, CudaError> {
        Ok(Self {
            rows,
            cols,
            val_coeff: val_coeff.try_clone()?,
            prev_val: Self::lift(context, prev_val, entries)?,
            next_val: Self::lift(context, next_val, entries)?,
            coeffs: coeffs.try_clone()?,
            val_init: context.upload(val_init)?,
            wa_scale: wa_scale.try_clone()?,
            coeff_width,
            entries,
            rounds_bound: 0,
        })
    }

    fn lift(
        context: &CudaKernelContext,
        raw: &CudaSlice<u64>,
        entries: usize,
    ) -> Result<DeviceFrVec, CudaError> {
        let mut out = context.alloc(entries)?;
        if entries == 0 {
            return Ok(out);
        }
        let count = CudaKernelContext::count_of(entries)?;
        let mut builder = context.stream().launch_builder(context.amm_lift());
        let _ = builder.arg(raw);
        let _ = builder.arg(&count);
        let _ = builder.arg(out.limbs_mut());
        // SAFETY: thread `i < entries` reads `raw[i]` and writes the `LIMBS`
        // field limbs at `out[i]`. `raw` holds `entries` u64s and `out` holds
        // `entries` field elements; they are distinct allocations.
        let _ = unsafe { builder.launch(CudaKernelContext::launch_config(count)) }?;
        context.stream().synchronize()?;
        Ok(out)
    }

    fn segments(&self, context: &CudaKernelContext) -> Result<(usize, Segments), CudaError> {
        let entries = CudaKernelContext::count_of(self.entries)?;
        let mut flags = context.alloc_u32(self.entries)?;
        let mut builder = context.stream().launch_builder(context.amm_segment_flags());
        let _ = builder.arg(&self.cols);
        let _ = builder.arg(&entries);
        let _ = builder.arg(&mut flags);
        // SAFETY: thread `index < entries` reads `cols[index]` and, when
        // `index > 0`, `cols[index - 1]`, then writes only `flags[index]`. Both
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
            .launch_builder(context.amm_segment_bounds());
        let _ = builder.arg(&self.cols);
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

    pub fn materialize(
        &self,
        context: &CudaKernelContext,
        k_prime: usize,
        t_prime: usize,
    ) -> Result<[DeviceFrVec; 3], CudaError> {
        if k_prime < self.val_init.len() {
            return Err(CudaError::InvariantViolation {
                reason: "materializing an address-major matrix needs a column per unbound address",
            });
        }
        let mut ra = context.alloc(k_prime * t_prime)?;
        let mut wa = context.alloc(k_prime * t_prime)?;
        let mut val = context.alloc(k_prime * t_prime)?;
        if t_prime > 0 {
            let cycles = CudaKernelContext::count_of(t_prime)?;
            let count = CudaKernelContext::count_of(self.val_init.len() * t_prime)?;
            let mut builder = context
                .stream()
                .launch_builder(context.amm_initial_values());
            let _ = builder.arg(self.val_init.limbs());
            let _ = builder.arg(&cycles);
            let _ = builder.arg(&count);
            let _ = builder.arg(val.limbs_mut());
            // SAFETY: i < initial.len() * cycles reads initial[i / cycles] and
            // writes val[i]; the checked output domain covers those elements.
            let _ = unsafe { builder.launch(CudaKernelContext::launch_config(count)) }?;
        }
        if self.entries == 0 || t_prime == 0 {
            return Ok([ra, wa, val]);
        }
        let entries = CudaKernelContext::count_of(self.entries)?;
        let width = CudaKernelContext::count_of(self.coeff_width)?;
        let cycles = CudaKernelContext::count_of(t_prime)?;

        let mut builder = context.stream().launch_builder(context.amm_materialize());
        let _ = builder.arg(&self.rows);
        let _ = builder.arg(&self.cols);
        let _ = builder.arg(self.val_coeff.limbs());
        let _ = builder.arg(self.next_val.limbs());
        let _ = builder.arg(self.coeffs.limbs());
        let _ = builder.arg(&width);
        let _ = builder.arg(&entries);
        let _ = builder.arg(&cycles);
        let _ = builder.arg(ra.limbs_mut());
        let _ = builder.arg(wa.limbs_mut());
        let _ = builder.arg(val.limbs_mut());
        // SAFETY: thread `n < entries` reads `rows`/`cols` at `n` and `n + 1`
        // (guarded), and its own `val_coeff`/`next_val`/`coeffs` slots. It returns
        // unless `row < t_prime`, and writes `ra`/`wa`/`val` only at
        // `col * t_prime + r` for `r` in `[row, fill_end)` with `fill_end <=
        // t_prime` — inside all three buffers because `col < val_init.len() <=
        // k_prime` (columns are `[0, val_init.len())` by construction and the
        // caller's `k_prime` was checked above). Writes are disjoint across
        // threads: entries have distinct `(col, row)` and each thread's fill range
        // stops at the next entry's row in the same column. Inputs and outputs are
        // distinct allocations.
        let _ = unsafe { builder.launch(CudaKernelContext::launch_config(entries)) }?;
        context.stream().synchronize()?;
        Ok([ra, wa, val])
    }

    pub fn round_evals<F: JoltField>(
        &self,
        context: &CudaKernelContext,
        inc: &DeviceFrVec,
        eq: &DeviceFrVec,
    ) -> Result<[F; 2], CudaError> {
        if self.entries == 0 {
            return Ok([F::from_u64(0), F::from_u64(0)]);
        }
        let (segment_count, seg) = self.segments(context)?;
        let segments = CudaKernelContext::count_of(segment_count)?;

        let blocks = segments.div_ceil(BLOCK).max(1);
        let mut partials = context.alloc(2 * blocks as usize)?;
        let width = CudaKernelContext::count_of(self.coeff_width)?;

        let mut builder = context.stream().launch_builder(context.amm_message());
        let _ = builder.arg(&self.rows);
        let _ = builder.arg(self.val_coeff.limbs());
        let _ = builder.arg(self.next_val.limbs());
        let _ = builder.arg(self.coeffs.limbs());
        let _ = builder.arg(&width);
        let _ = builder.arg(&seg.start);
        let _ = builder.arg(&seg.even_end);
        let _ = builder.arg(&seg.end);
        let _ = builder.arg(&seg.pair);
        let _ = builder.arg(&segments);
        let _ = builder.arg(self.val_init.limbs());
        let _ = builder.arg(inc.limbs());
        let _ = builder.arg(eq.limbs());
        let _ = builder.arg(self.wa_scale.limbs());
        let _ = builder.arg(partials.limbs_mut());
        // SAFETY: thread `seg < segments` reads matrix data only between
        // `seg_start[seg]` and `seg_end[seg]` (confined to `entries` by the
        // bounds kernel), `val_init[2 * pair]` / `[2 * pair + 1]` (in range
        // because `pair` is a column-pair index over the current column count and
        // `val_init` holds twice that), and `inc[row]` / `eq[row]` for rows read
        // out of that same confined range — in range because both polynomials are
        // dense over the cycle domain the rows index. It writes only
        // `partials[lane * gridDim.x + blockIdx.x]`, one slot per (lane, block) of
        // `2 * blocks`. Shared memory is `BLOCK * LIMBS` u64s, matching
        // `shared_mem_bytes`, and every `__syncthreads()` is reached by all
        // threads in the block because the reduction sits outside the
        // `seg < segments` guard.
        let _ = unsafe {
            builder.launch(LaunchConfig {
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

    pub fn bind<F: JoltField>(
        &mut self,
        context: &CudaKernelContext,
        challenge: F,
    ) -> Result<(), CudaError> {
        let challenge = super::device::require_fr(challenge)?;
        if self.entries == 0 {
            self.val_init = context.bind(&self.val_init, challenge, BindingOrder::LowToHigh)?;
            self.rounds_bound += 1;
            return Ok(());
        }
        let (segment_count, seg) = self.segments(context)?;
        let segments = CudaKernelContext::count_of(segment_count)?;

        let mut counts = context.alloc_u32(segment_count)?;
        let mut builder = context.stream().launch_builder(context.amm_count());
        let _ = builder.arg(&self.rows);
        let _ = builder.arg(&seg.start);
        let _ = builder.arg(&seg.even_end);
        let _ = builder.arg(&seg.end);
        let _ = builder.arg(&segments);
        let _ = builder.arg(&mut counts);
        // SAFETY: thread `seg < segments` reads its own bounds from the four
        // `seg_*` buffers (each `segments` u32s) and walks `rows` only between
        // `seg_start[seg]` and `seg_end[seg]`, which the bounds kernel confined
        // to `entries`. It writes only `counts[seg]`.
        let _ = unsafe { builder.launch(CudaKernelContext::launch_config(segments)) }?;
        context.stream().synchronize()?;

        let (device_offsets, bound_len) =
            context.exclusive_scan_with_total_u32(&counts, segment_count)?;

        let mut out_rows = context.alloc_u32(bound_len)?;
        let mut out_cols = context.alloc_u32(bound_len)?;
        let mut out_val = context.alloc(bound_len)?;
        let mut out_prev = context.alloc(bound_len)?;
        let mut out_next = context.alloc(bound_len)?;
        let mut out_coeffs = context.alloc(bound_len * self.coeff_width)?;
        let device_challenge = context.upload(&[challenge])?;
        let width = CudaKernelContext::count_of(self.coeff_width)?;

        let mut builder = context.stream().launch_builder(context.amm_merge());
        let _ = builder.arg(&self.rows);
        let _ = builder.arg(self.val_coeff.limbs());
        let _ = builder.arg(self.prev_val.limbs());
        let _ = builder.arg(self.next_val.limbs());
        let _ = builder.arg(self.coeffs.limbs());
        let _ = builder.arg(&width);
        let _ = builder.arg(&seg.start);
        let _ = builder.arg(&seg.even_end);
        let _ = builder.arg(&seg.end);
        let _ = builder.arg(&seg.pair);
        let _ = builder.arg(&device_offsets);
        let _ = builder.arg(&segments);
        let _ = builder.arg(device_challenge.limbs());
        let _ = builder.arg(self.val_init.limbs());
        let _ = builder.arg(&mut out_rows);
        let _ = builder.arg(&mut out_cols);
        let _ = builder.arg(out_val.limbs_mut());
        let _ = builder.arg(out_prev.limbs_mut());
        let _ = builder.arg(out_next.limbs_mut());
        let _ = builder.arg(out_coeffs.limbs_mut());
        // SAFETY: thread `seg < segments` reads inputs only between
        // `seg_start[seg]` and `seg_end[seg]` (confined to `entries` by the
        // bounds kernel), the single-element `challenge`, and
        // `val_init[2 * pair]` / `[2 * pair + 1]` — in range because `pair` is a
        // column-pair index over the current column count and `val_init` holds
        // twice that. It writes exactly `counts[seg]` output slots starting at
        // `offsets[seg]`; since `offsets` is the exclusive scan of `counts` and
        // `bound_len` their total, the per-segment ranges are disjoint and
        // inside every `out_*` buffer. Inputs and outputs are distinct
        // allocations.
        let _ = unsafe { builder.launch(CudaKernelContext::launch_config(segments)) }?;
        context.stream().synchronize()?;

        self.rows = out_rows;
        self.cols = out_cols;
        self.val_coeff = out_val;
        self.prev_val = out_prev;
        self.next_val = out_next;
        self.coeffs = out_coeffs;
        self.entries = bound_len;
        self.val_init = context.bind(&self.val_init, challenge, BindingOrder::LowToHigh)?;
        self.rounds_bound += 1;
        Ok(())
    }
}
