use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use jolt_poly::EqPolynomial;
use jolt_utils::unsafe_allocate_zero_vec;
use jolt_verifier::stages::stage2::ram_read_write_checking::RamReadWriteChecking;
use jolt_witness::RandomAccessRows;

use super::*;

const CHUNK: usize = 1 << 14;
const CAPTURE_BYTES: usize = 512 << 20;

#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
struct Chunk {
    cycles: Vec<u32>,
    values: Vec<F>,
}

#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
pub(super) struct SparseSource {
    chunks: Vec<Chunk>,
    pub(super) addresses: Arc<RamAddresses>,
    #[cfg_attr(feature = "allocative", allocative(skip))]
    pub(super) source: RandomAccessRows,
    pub(super) cycles: usize,
    pub(super) records: usize,
    pub(super) prefix: usize,
}

pub(super) struct UploadedIncrements {
    pub(super) cycles: DeviceBuffer<u32>,
    pub(super) values: DeviceBuffer<F>,
}

/// Installed together with the value consumer; ordinary CPU preparation has no carry.
pub(crate) struct CaptureRamReadWrite {
    pub(super) min_prefix: usize,
}
impl PrepareKernel<F, RamReadWriteChecking<F>> for CaptureRamReadWrite {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, RamReadWriteChecking<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = RamReadWriteChecking<F>>>, KernelError<F>>
    {
        drop(session.take::<SparseSource>());
        let log_t = inputs.relation.dimensions().log_t();
        let Some(source) = witness.random_access() else {
            return OptimizedBackend.prepare(session, witness, inputs);
        };
        if !(6..=31).contains(&log_t) {
            return OptimizedBackend.prepare(session, witness, inputs);
        }
        OptimizedBackend::prepare_ram_with_increments(
            session,
            witness,
            inputs,
            |session, columns| {
                if columns.value_prefix_len() < self.min_prefix {
                    return map_indices(columns.value_prefix_len(), |cycle| {
                        columns.increment(cycle)
                    });
                }
                let (values, carry) =
                    SparseSource::capture(columns, source, 1 << log_t, CAPTURE_BYTES);
                if let Some(carry) = carry {
                    session.park(carry);
                }
                values
            },
        )
    }
}

impl SparseSource {
    #[tracing::instrument(name = "metal_ram_val_capture", skip_all, fields(records = tracing::field::Empty, admitted = tracing::field::Empty))]
    pub(super) fn capture(
        columns: &RamAccessColumns,
        source: RandomAccessRows,
        cycles: usize,
        cap: usize,
    ) -> (Vec<F>, Option<Self>) {
        let len = columns.value_prefix_len();
        let metadata = len.div_ceil(CHUNK) * size_of::<Chunk>();
        if metadata > cap || len > u32::MAX as usize {
            return (map_indices(len, |cycle| columns.increment(cycle)), None);
        }
        let remaining = AtomicUsize::new(cap - metadata);
        let overflow = AtomicBool::new(false);
        let mut values = Vec::<F>::with_capacity(len);
        let destination = &mut values.spare_capacity_mut()[..len];
        let produce = |(chunk, output): (usize, &mut [MaybeUninit<F>])| {
            let first = chunk * CHUNK;
            let mut count = 0;
            for (offset, dst) in output.iter_mut().enumerate() {
                let value = columns.increment::<F>(first + offset);
                let _ = dst.write(value);
                count += usize::from(value != F::zero());
            }
            let bytes = count * (size_of::<u32>() + size_of::<F>());
            if overflow.load(Ordering::Relaxed)
                || remaining
                    .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |left| {
                        left.checked_sub(bytes)
                    })
                    .is_err()
            {
                overflow.store(true, Ordering::Relaxed);
                return Chunk {
                    cycles: Vec::new(),
                    values: Vec::new(),
                };
            }
            let mut records = Chunk {
                cycles: Vec::with_capacity(count),
                values: Vec::with_capacity(count),
            };
            for (offset, value) in output.iter().enumerate() {
                // SAFETY: the loop above initialized every output before this scan.
                let value = unsafe { value.assume_init() };
                if value != F::zero() {
                    records.cycles.push((first + offset) as u32);
                    records.values.push(value);
                }
            }
            records
        };
        #[cfg(feature = "parallel")]
        let chunks: Vec<_> = destination
            .par_chunks_mut(CHUNK)
            .enumerate()
            .map(produce)
            .collect();
        #[cfg(not(feature = "parallel"))]
        let chunks: Vec<_> = destination
            .chunks_mut(CHUNK)
            .enumerate()
            .map(produce)
            .collect();
        // SAFETY: every disjoint chunk initialized all its entries even if capture failed.
        unsafe { values.set_len(len) };
        let admitted = !overflow.load(Ordering::Relaxed);
        let _ = tracing::Span::current().record("admitted", admitted);
        if !admitted {
            return (values, None);
        }
        let records = chunks.iter().map(|chunk| chunk.values.len()).sum();
        let prefix = chunks
            .iter()
            .rev()
            .find_map(|chunk| chunk.cycles.last())
            .map_or(0, |cycle| *cycle as usize + 1);
        let _ = tracing::Span::current().record("records", records);
        (
            values,
            Some(Self {
                chunks,
                addresses: Arc::clone(&columns.addresses),
                source,
                cycles,
                records,
                prefix,
            }),
        )
    }

    pub(super) fn upload(&self, device: &Device) -> Result<UploadedIncrements, SumcheckError<F>> {
        let mut cycles = DeviceBuffer::<MaybeUninit<u32>>::uninit(device, self.records)
            .map_err(compute_error)?;
        let mut values =
            DeviceBuffer::<MaybeUninit<F>>::uninit(device, self.records).map_err(compute_error)?;
        let cycle_output = cycles.write_uninit().map_err(compute_error)?;
        let value_output = values.write_uninit().map_err(compute_error)?;
        let mut offset = 0;
        for chunk in &self.chunks {
            for (dst, src) in cycle_output[offset..offset + chunk.cycles.len()]
                .iter_mut()
                .zip(&chunk.cycles)
            {
                let _ = dst.write(*src);
            }
            for (dst, src) in value_output[offset..offset + chunk.values.len()]
                .iter_mut()
                .zip(&chunk.values)
            {
                let _ = dst.write(*src);
            }
            offset += chunk.values.len();
        }
        // SAFETY: ordered chunks partition both destinations and initialize all elements.
        unsafe {
            Ok(UploadedIncrements {
                cycles: cycles.assume_init().map_err(compute_error)?,
                values: values.assume_init().map_err(compute_error)?,
            })
        }
    }

    /// Materialize only at the canonical RA switch. Fixed source chunks align
    /// with every admitted binding width, so parallel writers remain disjoint.
    pub(super) fn materialize(&self, weights: &[F]) -> Vec<F> {
        let width = weights.len();
        assert!(width.is_power_of_two() && width <= 16);
        let mut output = unsafe_allocate_zero_vec::<F>(self.prefix.div_ceil(width));
        let per_chunk = CHUNK / width;
        let fill = |(index, out): (usize, &mut [F])| {
            let chunk = &self.chunks[index];
            for (&cycle, &value) in chunk.cycles.iter().zip(&chunk.values) {
                let offset = cycle as usize % CHUNK;
                out[offset / width] += value * weights[offset % width];
            }
        };
        #[cfg(feature = "parallel")]
        output.par_chunks_mut(per_chunk).enumerate().for_each(fill);
        #[cfg(not(feature = "parallel"))]
        for chunk in output.chunks_mut(per_chunk).enumerate() {
            fill(chunk);
        }
        output
    }
}

pub(super) fn branch_weights(challenges: &[F]) -> Vec<F> {
    EqPolynomial::evals(&challenges.iter().rev().copied().collect::<Vec<_>>(), None)
}
