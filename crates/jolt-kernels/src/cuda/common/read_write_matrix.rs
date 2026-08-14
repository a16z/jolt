#![expect(
    dead_code,
    reason = "implementation target: the registers read-write vertical is the first non-test caller"
)]

use cudarc::driver::{CudaSlice, PushKernelArg};
use jolt_field::Fr;

use super::context::{CudaKernelContext, BLOCK};
use super::device::{DeviceFrVec, LIMBS};
use super::error::CudaError;

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

struct Segments {
    start: CudaSlice<u32>,
    even_end: CudaSlice<u32>,
    end: CudaSlice<u32>,
    pair: CudaSlice<u32>,
}

pub struct DeviceReadWriteMatrix {
    rows: CudaSlice<u32>,
    cols: CudaSlice<u32>,
    val_coeff: DeviceFrVec,
    prev_val: CudaSlice<u64>,
    next_val: CudaSlice<u64>,
    coeffs: DeviceFrVec,
    coeff_width: usize,
    entries: usize,
    rounds_bound: usize,
}

impl DeviceReadWriteMatrix {
    pub fn new(
        context: &CudaKernelContext,
        entries: &[MatrixEntry],
        coeff_width: usize,
    ) -> Result<Self, CudaError> {
        if coeff_width == 0 || coeff_width > MAX_COEFFS {
            return Err(CudaError::InvariantViolation {
                reason: "a read-write matrix carries one or two one-hot coefficients per entry",
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
            prev_val: context.upload_u64_slice(&prev_val)?,
            next_val: context.upload_u64_slice(&next_val)?,
            coeffs: context.upload(&coeffs)?,
            coeff_width,
            entries: count,
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
        let coeffs = self.coeffs.to_host()?;
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

        let host_flags = context.download_u32(&flags)?;
        let mut ranks = Vec::with_capacity(host_flags.len());
        let mut running = 0u32;
        for flag in &host_flags {
            ranks.push(running);
            running += *flag;
        }
        let segment_count = running as usize;
        let device_ranks = context.upload_u32_slice(&ranks)?;

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
        super::address_major_matrix::DeviceAddressMajorMatrix::from_parts(
            context,
            context.clone_u32(&self.rows)?,
            context.clone_u32(&self.cols)?,
            &self.val_coeff,
            &self.prev_val,
            &self.next_val,
            &self.coeffs,
            val_init,
            self.coeff_width,
            self.entries,
        )
    }

    pub fn quadratic_coeffs<F: jolt_field::Field>(
        &self,
        context: &CudaKernelContext,
        inc: &DeviceFrVec,
        eq: &jolt_poly::GruenSplitEqPolynomial<F>,
    ) -> Result<[F; 2], CudaError> {
        if self.entries == 0 {
            return Ok([F::from_u64(0), F::from_u64(0)]);
        }
        let (segment_count, seg) = self.segments(context)?;
        let segments = CudaKernelContext::count_of(segment_count)?;

        let e_in_host = super::device::require_fr_slice(eq.e_in_current())?;
        let e_out_host = super::device::require_fr_slice(eq.e_out_current())?;
        let e_in_len = CudaKernelContext::count_of(e_in_host.len())?;
        let num_x_in_bits = e_in_host.len().max(1).ilog2();
        let e_in = context.upload(e_in_host)?;
        let e_out = context.upload(e_out_host)?;

        let blocks = segments.div_ceil(BLOCK).max(1);
        let mut partials = context.alloc(2 * blocks as usize)?;
        let width = CudaKernelContext::count_of(self.coeff_width)?;

        let mut builder = context.stream().launch_builder(context.rwm_message());
        let _ = builder.arg(&self.cols);
        let _ = builder.arg(self.val_coeff.limbs());
        let _ = builder.arg(&self.prev_val);
        let _ = builder.arg(&self.next_val);
        let _ = builder.arg(self.coeffs.limbs());
        let _ = builder.arg(&width);
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
        let _ = builder.arg(partials.limbs_mut());
        // SAFETY: thread `seg < segments` reads matrix data only between
        // `seg_start[seg]` and `seg_end[seg]` (confined to `entries` by the
        // bounds kernel), `inc[2 * pair]` and `inc[2 * pair + 1]` (in range
        // because `pair` indexes the half-sized cycle domain `inc` was built
        // over), `e_in[pair & mask]` and `e_out[pair >> num_x_in_bits]` — both
        // bounded by construction of `num_x_in_bits` from `e_in`'s length. It
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

        let totals =
            super::dense_product::DeviceDenseProduct::reduce_lanes(context, partials, 2, blocks)?;
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

        let host_counts = context.download_u32(&counts)?;
        let offsets = context.exclusive_scan_u32(&host_counts)?;
        let bound_len = offsets
            .last()
            .copied()
            .unwrap_or(0)
            .checked_add(host_counts.last().copied().unwrap_or(0))
            .ok_or(CudaError::InvariantViolation {
                reason: "bound read-write matrix length overflowed u32",
            })? as usize;
        let device_offsets = context.upload_u32_slice(&offsets)?;

        let mut out_rows = context.alloc_u32(bound_len)?;
        let mut out_cols = context.alloc_u32(bound_len)?;
        let mut out_val = context.alloc(bound_len)?;
        let mut out_prev = context.alloc_u64(bound_len)?;
        let mut out_next = context.alloc_u64(bound_len)?;
        let mut out_coeffs = context.alloc(bound_len * self.coeff_width)?;
        let device_challenge = context.upload(&[challenge])?;
        let width = CudaKernelContext::count_of(self.coeff_width)?;

        let mut builder = context.stream().launch_builder(context.rwm_merge());
        let _ = builder.arg(&self.cols);
        let _ = builder.arg(self.val_coeff.limbs());
        let _ = builder.arg(&self.prev_val);
        let _ = builder.arg(&self.next_val);
        let _ = builder.arg(self.coeffs.limbs());
        let _ = builder.arg(&width);
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
        // SAFETY: thread `seg < segments` reads inputs only between
        // `seg_start[seg]` and `seg_end[seg]` (confined to `entries` by the
        // bounds kernel) plus the single-element `challenge`, and writes exactly
        // `counts[seg]` output slots starting at `offsets[seg]`. Since `offsets`
        // is the exclusive scan of `counts` and `bound_len` is their total, the
        // per-segment output ranges are disjoint and all inside every `out_*`
        // buffer. `out_coeffs` is indexed `k * coeff_width + lane` over
        // `bound_len * coeff_width` elements. Inputs and outputs are distinct
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
        self.rounds_bound += 1;
        Ok(())
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    clippy::panic,
    reason = "test module: device operations and fixture errors fail loudly"
)]
mod tests {
    use ark_bn254::Fr as LegacyFr;
    use jolt_field::Fr;
    use jolt_field::FromPrimitiveInt;
    use jolt_prover_legacy::field::challenge::MontU128Challenge as LegacyChallenge;
    use jolt_prover_legacy::field::JoltField as LegacyJoltField;
    use jolt_prover_legacy::poly::multilinear_polynomial::{
        BindingOrder as LegacyBindingOrder, MultilinearPolynomial as LegacyMultilinear,
        PolynomialBinding as LegacyBinding,
    };
    use jolt_prover_legacy::poly::split_eq_poly::GruenSplitEqPolynomial as LegacyGruen;
    use jolt_prover_legacy::subprotocols::read_write_matrix::{
        CycleMajorMatrixEntry, ReadWriteMatrixCycleMajor, RegistersCycleMajorEntry,
    };
    use rand::rngs::StdRng;
    use rand::{RngCore, SeedableRng};
    use strum::IntoEnumIterator;
    use tracer::instruction::Cycle;

    use super::super::context::shared_context;
    use super::{DeviceReadWriteMatrix, MatrixEntry};

    const LOG_T: usize = 6;
    const COEFF_WIDTH: usize = 2;

    fn random_cycle(rng: &mut StdRng) -> Cycle {
        let variants: Vec<Cycle> = Cycle::iter().collect();
        for _ in 0..10_000 {
            let index = rng.next_u64() as usize % variants.len();
            let candidate = variants[index].random(rng);
            if jolt_prover_legacy::zkvm::instruction::JoltTraceCycle::try_new(&candidate).is_ok() {
                return candidate;
            }
        }
        panic!("no convertible cycle variant found");
    }

    fn trace(seed: u64) -> Vec<Cycle> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..1usize << LOG_T)
            .map(|_| random_cycle(&mut rng))
            .collect()
    }

    type LegacyMatrix =
        ReadWriteMatrixCycleMajor<LegacyFr, RegistersCycleMajorEntry<LegacyFr, LegacyFr>>;

    fn legacy_entries(matrix: &LegacyMatrix) -> Vec<(u32, u32, Fr, u64, u64, Fr, Fr)> {
        matrix
            .entries
            .iter()
            .map(|entry| {
                (
                    CycleMajorMatrixEntry::row(entry) as u32,
                    CycleMajorMatrixEntry::column(entry) as u32,
                    Fr::from(entry.val_coeff),
                    entry.prev_val,
                    entry.next_val,
                    Fr::from(entry.ra_coeff),
                    Fr::from(entry.wa_coeff),
                )
            })
            .collect()
    }

    fn device_entries(entries: &[MatrixEntry]) -> Vec<(u32, u32, Fr, u64, u64, Fr, Fr)> {
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
                    entry.coeffs[1],
                )
            })
            .collect()
    }

    fn seed_from(matrix: &LegacyMatrix) -> Vec<MatrixEntry> {
        matrix
            .entries
            .iter()
            .map(|entry| MatrixEntry {
                row: CycleMajorMatrixEntry::row(entry) as u32,
                col: CycleMajorMatrixEntry::column(entry) as u32,
                val_coeff: Fr::from(entry.val_coeff),
                prev_val: entry.prev_val,
                next_val: entry.next_val,
                coeffs: [Fr::from(entry.ra_coeff), Fr::from(entry.wa_coeff)],
            })
            .collect()
    }

    #[test]
    fn device_quadratic_coeffs_match_legacy_round_for_round() {
        let Some(context) = shared_context() else {
            return;
        };
        let gamma_raw = 101u64;
        let gamma = <LegacyFr as LegacyJoltField>::from_u64(gamma_raw);
        let mut legacy: LegacyMatrix = ReadWriteMatrixCycleMajor::<
            LegacyFr,
            RegistersCycleMajorEntry<LegacyFr, _>,
        >::new(&trace(7), gamma)
        .deref_coeffs();
        let mut device = DeviceReadWriteMatrix::new(context, &seed_from(&legacy), COEFF_WIDTH)
            .expect("device matrix");

        let inc_host: Vec<Fr> = (0..1usize << LOG_T)
            .map(|j| crate::cuda::common::testing::fr(j as u64 * 5 + 3))
            .collect();
        let mut legacy_inc = LegacyMultilinear::from(
            inc_host
                .iter()
                .map(|value| LegacyFr::from(*value))
                .collect::<Vec<_>>(),
        );
        let mut device_inc = context.upload(&inc_host).expect("upload inc");

        let point: Vec<Fr> = (0..LOG_T)
            .map(|i| crate::cuda::common::testing::fr(i as u64 * 11 + 7))
            .collect();
        let legacy_point: Vec<LegacyChallenge<LegacyFr>> = (0..LOG_T)
            .map(|i| LegacyChallenge::new(31 + i as u128))
            .collect();
        let mut legacy_eq =
            LegacyGruen::<LegacyFr>::new(&legacy_point, LegacyBindingOrder::LowToHigh);
        let mut device_eq = jolt_poly::GruenSplitEqPolynomial::<Fr>::new(
            &legacy_point
                .iter()
                .map(|c| Fr::from(LegacyFr::from(*c)))
                .collect::<Vec<_>>(),
            jolt_poly::BindingOrder::LowToHigh,
        );
        let _ = point;

        for round in 0..LOG_T {
            let expected = legacy.compute_message(
                &legacy_inc,
                &legacy_eq,
                gamma,
                <LegacyFr as LegacyJoltField>::from_u64(0),
            );
            let got_coeffs: [Fr; 2] = device
                .quadratic_coeffs(context, &device_inc, &device_eq)
                .expect("device quadratic coeffs");
            let got = device_eq.gruen_poly_deg_3(got_coeffs[0], got_coeffs[1], Fr::from_u64(0));
            let expected: Vec<Fr> = expected.coeffs.iter().map(|c| Fr::from(*c)).collect();
            assert_eq!(
                got.coefficients().to_vec(),
                expected,
                "round polynomials diverged at round {round}",
            );

            let challenge = LegacyChallenge::new(17 + round as u128);
            legacy.bind(challenge);
            device
                .bind(context, Fr::from(LegacyFr::from(challenge)))
                .expect("device bind");
            legacy_inc.bind_parallel(challenge, LegacyBindingOrder::LowToHigh);
            device_inc = context
                .bind(
                    &device_inc,
                    Fr::from(LegacyFr::from(challenge)),
                    jolt_poly::BindingOrder::LowToHigh,
                )
                .expect("bind device inc");
            legacy_eq.bind(challenge);
            device_eq.bind(Fr::from(LegacyFr::from(challenge)));
        }
    }

    #[test]
    fn device_matrix_bind_matches_legacy_round_for_round() {
        let Some(context) = shared_context() else {
            return;
        };
        let gamma = <LegacyFr as LegacyJoltField>::from_u64(101);
        let mut legacy: LegacyMatrix = ReadWriteMatrixCycleMajor::<
            LegacyFr,
            RegistersCycleMajorEntry<LegacyFr, _>,
        >::new(&trace(7), gamma)
        .deref_coeffs();

        let mut device = DeviceReadWriteMatrix::new(context, &seed_from(&legacy), COEFF_WIDTH)
            .expect("device matrix");

        assert_eq!(
            device_entries(&device.to_host(context).expect("download")),
            legacy_entries(&legacy),
            "seeded entry sets disagree before any bind",
        );

        for round in 0..LOG_T {
            let raw = 17u128 + round as u128;
            let challenge = LegacyChallenge::new(raw);
            legacy.bind(challenge);
            device
                .bind(context, Fr::from(LegacyFr::from(challenge)))
                .expect("device bind");
            assert_eq!(
                device_entries(&device.to_host(context).expect("download")),
                legacy_entries(&legacy),
                "entry sets diverged after binding round {round}",
            );
        }
    }
}
