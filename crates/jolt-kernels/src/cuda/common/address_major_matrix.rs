#![expect(
    dead_code,
    reason = "implementation target: the registers read-write phase-2 rounds are the first non-test caller"
)]

use cudarc::driver::{CudaSlice, PushKernelArg};
use jolt_field::Fr;

use super::context::CudaKernelContext;
use super::device::DeviceFrVec;
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
    ) -> Result<Self, CudaError> {
        if coeff_width == 0 || coeff_width > MAX_COEFFS {
            return Err(CudaError::InvariantViolation {
                reason: "an address-major matrix carries one or two one-hot coefficients per entry",
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

        let host_counts = context.download_u32(&counts)?;
        let offsets = context.exclusive_scan_u32(&host_counts)?;
        let bound_len = offsets
            .last()
            .copied()
            .unwrap_or(0)
            .checked_add(host_counts.last().copied().unwrap_or(0))
            .ok_or(CudaError::InvariantViolation {
                reason: "bound address-major matrix length overflowed u32",
            })? as usize;
        let device_offsets = context.upload_u32_slice(&offsets)?;

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
    clippy::panic,
    reason = "test module: device operations and fixture errors fail loudly"
)]
mod tests {
    use ark_bn254::Fr as LegacyFr;
    use jolt_field::{Fr, FromPrimitiveInt};
    use jolt_prover_legacy::field::challenge::MontU128Challenge as LegacyChallenge;
    use jolt_prover_legacy::field::JoltField as LegacyJoltField;
    use jolt_prover_legacy::subprotocols::read_write_matrix::{
        AddressMajorMatrixEntry, ReadWriteMatrixAddressMajor, ReadWriteMatrixCycleMajor,
        RegistersAddressMajorEntry, RegistersCycleMajorEntry,
    };
    use rand::rngs::StdRng;
    use rand::{RngCore, SeedableRng};
    use strum::IntoEnumIterator;
    use tracer::instruction::Cycle;

    use super::super::context::shared_context;
    use super::{AddressMajorEntry, DeviceAddressMajorMatrix};

    const LOG_T: usize = 6;
    const COEFF_WIDTH: usize = 2;
    const REGISTER_COUNT: usize = 128;

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

    type LegacyAddressMajor =
        ReadWriteMatrixAddressMajor<LegacyFr, RegistersAddressMajorEntry<LegacyFr>>;

    fn legacy_view(matrix: &LegacyAddressMajor) -> Vec<(u32, u32, Fr, Fr, Fr, Fr, Fr)> {
        matrix
            .entries
            .iter()
            .map(|entry| {
                (
                    AddressMajorMatrixEntry::row(entry) as u32,
                    AddressMajorMatrixEntry::column(entry) as u32,
                    Fr::from(entry.val_coeff),
                    Fr::from(AddressMajorMatrixEntry::prev_val(entry)),
                    Fr::from(AddressMajorMatrixEntry::next_val(entry)),
                    Fr::from(entry.ra_coeff),
                    Fr::from(entry.wa_coeff),
                )
            })
            .collect()
    }

    fn device_view(entries: &[AddressMajorEntry]) -> Vec<(u32, u32, Fr, Fr, Fr, Fr, Fr)> {
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

    #[test]
    fn address_major_binds_like_legacy_round_for_round() {
        let Some(context) = shared_context() else {
            return;
        };
        let mut rng = StdRng::seed_from_u64(17);
        let trace: Vec<Cycle> = (0..1usize << LOG_T)
            .map(|_| random_cycle(&mut rng))
            .collect();
        let gamma = <LegacyFr as LegacyJoltField>::from_u64(101);

        let cycle_major = ReadWriteMatrixCycleMajor::<
            LegacyFr,
            RegistersCycleMajorEntry<LegacyFr, _>,
        >::new(&trace, gamma);
        let mut legacy: LegacyAddressMajor = cycle_major.into();

        let seeded: Vec<AddressMajorEntry> = legacy
            .entries
            .iter()
            .map(|entry| AddressMajorEntry {
                row: AddressMajorMatrixEntry::row(entry) as u32,
                col: AddressMajorMatrixEntry::column(entry) as u32,
                val_coeff: Fr::from(entry.val_coeff),
                prev_val: Fr::from(AddressMajorMatrixEntry::prev_val(entry)),
                next_val: Fr::from(AddressMajorMatrixEntry::next_val(entry)),
                coeffs: [Fr::from(entry.ra_coeff), Fr::from(entry.wa_coeff)],
            })
            .collect();
        let val_init = vec![Fr::from_u64(0); REGISTER_COUNT];
        let mut device = DeviceAddressMajorMatrix::new(context, &seeded, &val_init, COEFF_WIDTH)
            .expect("device address-major matrix");

        assert_eq!(
            device_view(&device.to_host(context).expect("download")),
            legacy_view(&legacy),
            "seeded entry sets disagree before any bind",
        );

        for round in 0..REGISTER_COUNT.ilog2() as usize {
            let raw = 29u128 + round as u128;
            let challenge = LegacyChallenge::new(raw);
            legacy.bind(challenge);
            device
                .bind(context, Fr::from(LegacyFr::from(challenge)))
                .expect("device bind");
            assert_eq!(
                device_view(&device.to_host(context).expect("download")),
                legacy_view(&legacy),
                "entry sets diverged after binding round {round}",
            );
        }
    }
}
