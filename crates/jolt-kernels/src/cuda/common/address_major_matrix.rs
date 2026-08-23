#![expect(
    dead_code,
    reason = "implementation target: the registers read-write phase-2 rounds are the first non-test caller"
)]

use cudarc::driver::{CudaSlice, PushKernelArg};
use jolt_field::{Fr, FromPrimitiveInt};

use super::context::{CudaKernelContext, BLOCK};
use super::device::{DeviceFrVec, LIMBS};
use super::error::CudaError;
use super::read_write_matrix::MAX_COEFFS;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AddressMajorEntry {
    pub row: u32,
    pub col: u32,
    pub val_coeff: Fr,
    pub prev_val: Fr,
    pub next_val: Fr,
    pub coeffs: [Fr; MAX_COEFFS],
}

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
    pub fn new(
        context: &CudaKernelContext,
        entries: &[AddressMajorEntry],
        val_init: &[Fr],
        coeff_width: usize,
        wa_scale: Option<Fr>,
    ) -> Result<Self, CudaError> {
        if coeff_width == 0 || coeff_width > MAX_COEFFS {
            return Err(CudaError::InvariantViolation {
                reason: "an address-major matrix carries one or two one-hot coefficients per entry",
            });
        }
        if wa_scale.is_some() != (coeff_width == 1) {
            return Err(CudaError::InvariantViolation {
                reason: "a single-coefficient matrix derives its write lane by scaling its read lane, and only such a matrix may",
            });
        }
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
            prev_val: context.upload(&prev_val)?,
            next_val: context.upload(&next_val)?,
            coeffs: context.upload(&coeffs)?,
            val_init: context.upload(val_init)?,
            wa_scale: context.upload(&[wa_scale.unwrap_or(Fr::from_u64(0))])?,
            coeff_width,
            entries: count,
            rounds_bound: 0,
        })
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

    pub const fn len(&self) -> usize {
        self.entries
    }

    pub const fn is_empty(&self) -> bool {
        self.entries == 0
    }

    pub const fn rounds_bound(&self) -> usize {
        self.rounds_bound
    }

    pub fn to_host(
        &self,
        context: &CudaKernelContext,
    ) -> Result<Vec<AddressMajorEntry>, CudaError> {
        let rows = context.download_u32(&self.rows)?;
        let cols = context.download_u32(&self.cols)?;
        let val_coeff = self.val_coeff.to_host()?;
        let prev_val = self.prev_val.to_host()?;
        let next_val = self.next_val.to_host()?;
        let coeffs = self.coeffs.to_host()?;
        let mut out = Vec::with_capacity(self.entries);
        for index in 0..self.entries {
            let mut entry_coeffs = [Fr::default(); MAX_COEFFS];
            for lane in 0..self.coeff_width {
                entry_coeffs[lane] = coeffs[index * self.coeff_width + lane];
            }
            out.push(AddressMajorEntry {
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

    pub fn round_evals<F: jolt_field::Field>(
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
        let challenge = super::device::require_fr(challenge)?;
        if self.entries == 0 {
            self.val_init = context.bind(
                &self.val_init,
                challenge,
                jolt_poly::BindingOrder::LowToHigh,
            )?;
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
        self.val_init = context.bind(
            &self.val_init,
            challenge,
            jolt_poly::BindingOrder::LowToHigh,
        )?;
        self.rounds_bound += 1;
        Ok(())
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test module: device operations and fixture errors fail loudly"
)]
mod tests {
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_poly::Polynomial;

    use super::super::context::shared_context;
    use super::super::testing::{fr, ram_matrix_cells};
    use super::{AddressMajorEntry, DeviceAddressMajorMatrix};
    use crate::optimized::rw_matrix::{AddressMajorMatrix, CycleMajorEntry, CycleMajorMatrix};

    const LOG_T: usize = 6;
    const RAM_K: usize = 32;
    const RAM_LOG_K: usize = 5;

    fn optimized_address_major() -> AddressMajorMatrix<Fr> {
        let entries = ram_matrix_cells(LOG_T, RAM_K)
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
        let mut cycle_major = CycleMajorMatrix { entries };
        for round in 0..LOG_T {
            cycle_major.bind(fr(31 + round as u64));
        }
        cycle_major.into_address_major()
    }

    fn device_seed(expected: &AddressMajorMatrix<Fr>) -> Vec<AddressMajorEntry> {
        expected
            .entries
            .iter()
            .map(|entry| AddressMajorEntry {
                row: entry.row as u32,
                col: entry.col as u32,
                val_coeff: entry.val,
                prev_val: entry.prev_val,
                next_val: entry.next_val,
                coeffs: [entry.ra, Fr::from_u64(0)],
            })
            .collect()
    }

    fn device_view(entries: &[AddressMajorEntry]) -> Vec<(u32, u32, Fr, Fr, Fr, Fr)> {
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

    fn optimized_view(matrix: &AddressMajorMatrix<Fr>) -> Vec<(u32, u32, Fr, Fr, Fr, Fr)> {
        matrix
            .entries
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

    fn val_init_values() -> Vec<Fr> {
        (0..RAM_K).map(|k| fr(k as u64 * 23 + 5)).collect()
    }

    #[test]
    fn address_major_ram_bind_matches_optimized_round_for_round() {
        let Some(context) = shared_context() else {
            return;
        };
        let mut expected = optimized_address_major();
        let val_init = val_init_values();
        let mut expected_val_init = Polynomial::new(val_init.clone());
        let mut got = DeviceAddressMajorMatrix::new(
            context,
            &device_seed(&expected),
            &val_init,
            1,
            Some(fr(103)),
        )
        .expect("device RAM address-major matrix");

        assert_eq!(
            device_view(&got.to_host(context).expect("download")),
            optimized_view(&expected),
            "seeded entry sets disagree before any bind",
        );

        for round in 0..RAM_LOG_K {
            let challenge = fr(71 + round as u64);
            expected.bind(challenge, &mut expected_val_init);
            got.bind(context, challenge).expect("device bind");
            assert_eq!(
                device_view(&got.to_host(context).expect("download")),
                optimized_view(&expected),
                "entry sets diverged after binding round {round}",
            );
        }
    }

    #[test]
    fn address_major_ram_message_matches_optimized_round_for_round() {
        let Some(context) = shared_context() else {
            return;
        };
        let gamma = fr(103);
        let mut expected = optimized_address_major();
        let val_init = val_init_values();
        let mut expected_val_init = Polynomial::new(val_init.clone());
        let mut got = DeviceAddressMajorMatrix::new(
            context,
            &device_seed(&expected),
            &val_init,
            1,
            Some(gamma),
        )
        .expect("device RAM address-major matrix");

        let inc_host: Vec<Fr> = (0..1usize << LOG_T).map(|j| fr(j as u64 * 5 + 3)).collect();
        let eq_host: Vec<Fr> = (0..1usize << LOG_T)
            .map(|j| fr(j as u64 * 13 + 11))
            .collect();
        let expected_inc = Polynomial::new(inc_host.clone());
        let expected_eq = Polynomial::new(eq_host.clone());
        let got_inc = context.upload(&inc_host).expect("upload inc");
        let got_eq = context.upload(&eq_host).expect("upload eq");

        for round in 0..RAM_LOG_K {
            let expected_evals = expected.address_round_evals(
                &expected_val_init,
                &expected_inc,
                &expected_eq,
                gamma,
            );
            let got_evals: [Fr; 2] = got
                .round_evals(context, &got_inc, &got_eq)
                .expect("device round evals");
            assert_eq!(
                got_evals, expected_evals,
                "RAM round evals diverged at round {round}",
            );

            let challenge = fr(71 + round as u64);
            expected.bind(challenge, &mut expected_val_init);
            got.bind(context, challenge).expect("device bind");
        }
    }
}
