//! Point storage for one Dory opening on the calling host thread.
//!
//! G1 and G2 handles address append-only device buffers, with an immutable SRS
//! prefix cached on the host. Additional devices hold mirrors for distributed
//! vector work; a mirror allocation failure leaves the opening using fewer
//! devices. Dropping the arena guard releases the per-opening buffers.
//!
//! Large vector updates compute disjoint slices, then exchange their output
//! points. The mirrors remain complete because MSM and pairing consumers use
//! different partitions. Small updates run on each mirror to avoid exchanges.
//!
//! Once MSMs become small, their family retires secondary copies for the rest
//! of the opening, avoiding repeated broadcasts and mirrored vector updates. The
//! retirement cutoff is below the MSM split threshold so larger pairing
//! batches can still use the mirrors.
//!
//! After retirement, small MSMs wait until their results are needed. An
//! independent pairing batch copies just its inputs to GPU 1 and runs there
//! while GPU 0 computes the MSMs. Reads, mutations, and dependent requests
//! resolve pending MSMs first, preserving their input values and transcript
//! order. Pairing scratch is released after the batch; full mirrors stay retired.
//!
//! Dory's group traits cannot return storage errors. Handle failures therefore
//! poison this arena. `CudaDoryScheme::open` checks that flag after proving,
//! before resolving the final proof's handles.
//! Handles must stay within the opening and host thread that own their arena.
//! The offset representation does not enforce that ownership contract.

use std::cell::{Cell, RefCell};
use std::ops::Range;

use ark_bn254::{Fq, Fq2, G1Projective, G2Projective};
use ark_ff::BigInt;
use cudarc::driver::{CudaSlice, PushKernelArg};
use jolt_field::Fr;

use crate::cuda::common::context::{context_for, device_count, CudaKernelContext};
use crate::cuda::common::devices::{fan_out, DeviceTask};
use crate::cuda::common::error::CudaError;
use crate::cuda::common::msm::{ResidentAxpy, FQ_LIMBS};
use crate::cuda::common::xfer_stats::{self, Phase};

pub(super) const G1_WORDS: usize = 3 * FQ_LIMBS;

pub(super) const G2_WORDS: usize = 6 * FQ_LIMBS;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Family {
    G1,
    G2,
}

impl Family {
    const fn words(self) -> usize {
        match self {
            Self::G1 => G1_WORDS,
            Self::G2 => G2_WORDS,
        }
    }
}

struct Arena {
    buffers: Vec<CudaSlice<u64>>,
    words: usize,
    used: usize,
    capacity: usize,
    frozen: usize,
    frozen_host: Vec<u64>,
}

struct PairingInput {
    points: CudaSlice<u64>,
    offsets: Vec<usize>,
}

impl Arena {
    fn new(family: Family, capacity: usize) -> Result<Self, CudaError> {
        let words = family.words();
        let mut buffers = vec![device(0)?.alloc_u64(capacity * words)?];
        for ordinal in 1..device_count() {
            match device(ordinal).and_then(|context| context.alloc_u64(capacity * words)) {
                Ok(buffer) => buffers.push(buffer),
                Err(error) => {
                    tracing::warn!(
                        ?error,
                        ordinal,
                        "a Dory arena mirror did not allocate; the opening keeps {} device(s)",
                        buffers.len(),
                    );
                    break;
                }
            }
        }
        Ok(Self {
            buffers,
            words,
            used: 0,
            capacity,
            frozen: 0,
            frozen_host: Vec::new(),
        })
    }

    fn primary(&self) -> Result<&CudaSlice<u64>, CudaError> {
        self.buffers.first().ok_or(CudaError::InvariantViolation {
            reason: "a Dory arena holds no device buffer",
        })
    }

    const fn limbs(&self) -> usize {
        self.capacity * self.words
    }

    fn mirror(&mut self, start: usize, limbs: &[u64]) -> Result<(), CudaError> {
        for (ordinal, buffer) in self.buffers.iter_mut().enumerate() {
            device(ordinal)?.write_u64_range(buffer, start, limbs)?;
        }
        Ok(())
    }

    fn reserve(&mut self, count: usize) -> Result<usize, CudaError> {
        let end = self.used.saturating_add(count);
        if end > self.capacity {
            return Err(CudaError::LengthMismatch {
                expected: self.capacity,
                got: end,
            });
        }
        let offset = self.used;
        self.used = end;
        Ok(offset)
    }

    fn copy_pairing_input(
        &self,
        context: &CudaKernelContext,
        starts: impl IntoIterator<Item = usize>,
        count: usize,
    ) -> Result<PairingInput, CudaError> {
        let source = self.primary()?;
        let mut unique = Vec::new();
        let mut offsets = Vec::new();
        for start in starts {
            CudaKernelContext::resident_span_end(start, count, self.words, source)?;
            let index = if let Some(index) = unique.iter().position(|&offset| offset == start) {
                index
            } else {
                unique.push(start);
                unique.len() - 1
            };
            offsets.push(
                index
                    .checked_mul(count)
                    .ok_or(CudaError::InvariantViolation {
                        reason: "a Dory pairing input offset overflowed",
                    })?,
            );
        }
        let words = unique
            .len()
            .checked_mul(count)
            .and_then(|points| points.checked_mul(self.words))
            .ok_or(CudaError::InvariantViolation {
                reason: "a Dory pairing input buffer overflowed",
            })?;
        let mut points = context.alloc_u64(words)?;
        let copied = (|| {
            for (index, &start) in unique.iter().enumerate() {
                let source_start = start * self.words;
                let target_start = index * count * self.words;
                let words = count * self.words;
                xfer_stats::timed(Phase::D2d, words * size_of::<u64>(), || {
                    context.stream().memcpy_dtod(
                        &source.slice(source_start..source_start + words),
                        &mut points.slice_mut(target_start..target_start + words),
                    )
                })?;
            }
            Ok::<_, CudaError>(())
        })();
        // Cudarc does not record peer reads on the source allocation. Complete
        // all submitted copies, including on error, before an MSM can write it.
        let completed = context.stream().synchronize();
        copied?;
        completed?;
        Ok(PairingInput { points, offsets })
    }

    #[tracing::instrument(skip_all, name = "cuda_dory_exchange")]
    fn exchange(&mut self, offset: usize, chunks: &[Range<usize>]) -> Result<(), CudaError> {
        for (source_ordinal, chunk) in chunks.iter().enumerate() {
            let start = (offset + chunk.start) * self.words;
            let end = (offset + chunk.end) * self.words;
            let (before, tail) = self.buffers.split_at_mut(source_ordinal);
            let Some((source, after)) = tail.split_first_mut() else {
                return Err(CudaError::InvariantViolation {
                    reason: "a Dory vector chunk has no device mirror",
                });
            };
            let targets = before.iter_mut().enumerate().chain(
                after
                    .iter_mut()
                    .enumerate()
                    .map(|(index, buffer)| (source_ordinal + 1 + index, buffer)),
            );
            for (ordinal, target) in targets {
                let context = device(ordinal)?;
                let copied =
                    xfer_stats::timed(Phase::D2d, (end - start) * size_of::<u64>(), || {
                        context.stream().memcpy_dtod(
                            &source.slice(start..end),
                            &mut target.slice_mut(start..end),
                        )
                    });
                // Peer reads are not tracked on the source allocation by cudarc.
                // Complete them before a later update can overwrite the source;
                // serial destinations also avoid reverse readiness-event waits.
                let completed = context.stream().synchronize();
                copied?;
                completed?;
            }
        }
        Ok(())
    }
}

thread_local! {
    static G1_ARENA: RefCell<Option<Arena>> = const { RefCell::new(None) };
    static G2_ARENA: RefCell<Option<Arena>> = const { RefCell::new(None) };
    static PENDING_MSMS: RefCell<Vec<MsmRequest>> = const { RefCell::new(Vec::new()) };
    static POISONED: Cell<bool> = const { Cell::new(false) };
}

struct MsmRequest {
    family: Family,
    bases: Range<usize>,
    out: usize,
    scalars: Vec<Fr>,
}

impl MsmRequest {
    #[tracing::instrument(skip_all, name = "cuda_dory_pending_msm", fields(family = ?self.family, len = self.bases.len()))]
    fn run(&self, arena: &mut Arena) -> Result<(), CudaError> {
        let buffer = arena.buffers.first_mut().ok_or_else(closed)?;
        match self.family {
            Family::G1 => device(0)?.g1_msm_in_place(
                buffer,
                self.bases.start,
                self.out,
                self.bases.len(),
                &self.scalars,
            ),
            Family::G2 => device(0)?.g2_msm_in_place(
                buffer,
                self.bases.start,
                self.out,
                self.bases.len(),
                &self.scalars,
            ),
        }
    }
}

fn flush_msms() -> Result<(), CudaError> {
    let requests = PENDING_MSMS.with_borrow_mut(std::mem::take);
    for request in requests {
        with_arena(request.family, |arena| request.run(arena)).ok_or_else(closed)??;
    }
    Ok(())
}

pub(super) fn poison(reason: &'static str) {
    tracing::error!(
        reason,
        "the Dory arena was poisoned; the opening will be rejected"
    );
    POISONED.set(true);
}

pub(super) fn poisoned() -> bool {
    POISONED.get()
}

fn with_arena<R>(family: Family, act: impl FnOnce(&mut Arena) -> R) -> Option<R> {
    match family {
        Family::G1 => G1_ARENA.with_borrow_mut(|slot| slot.as_mut().map(act)),
        Family::G2 => G2_ARENA.with_borrow_mut(|slot| slot.as_mut().map(act)),
    }
}

const fn closed() -> CudaError {
    CudaError::InvariantViolation {
        reason: "the Dory arena is not open",
    }
}

fn device(ordinal: usize) -> Result<&'static CudaKernelContext, CudaError> {
    context_for(ordinal).ok_or(CudaError::NotImplemented {
        kernel: "no CUDA device is present for the Dory arenas",
    })
}

pub(super) struct ArenaGuard;

impl Drop for ArenaGuard {
    fn drop(&mut self) {
        PENDING_MSMS.with_borrow_mut(Vec::clear);
        G1_ARENA.with_borrow_mut(|slot| *slot = None);
        G2_ARENA.with_borrow_mut(|slot| *slot = None);
    }
}

pub(super) fn open(g1_capacity: usize, g2_capacity: usize) -> Result<ArenaGuard, CudaError> {
    let g1 = Arena::new(Family::G1, g1_capacity)?;
    let g2 = Arena::new(Family::G2, g2_capacity)?;
    G1_ARENA.with_borrow_mut(|slot| *slot = Some(g1));
    G2_ARENA.with_borrow_mut(|slot| *slot = Some(g2));
    PENDING_MSMS.with_borrow_mut(Vec::clear);
    POISONED.set(false);
    Ok(ArenaGuard)
}

pub(super) fn reserve(family: Family, count: usize) -> Result<usize, CudaError> {
    with_arena(family, |arena| arena.reserve(count)).ok_or_else(closed)?
}

/// Copy already-reserved points into a fresh contiguous span in every mirror.
pub(super) fn gather(family: Family, offsets: &[u32]) -> Result<usize, CudaError> {
    flush_msms()?;
    with_arena(family, |arena| {
        if offsets.iter().any(|&offset| offset as usize >= arena.used) {
            return Err(CudaError::InvariantViolation {
                reason: "a Dory gather referenced an unreserved point",
            });
        }
        let width = CudaKernelContext::count_of(arena.words)?;
        let words = CudaKernelContext::count_of(offsets.len().saturating_mul(arena.words))?;
        let first = arena.reserve(offsets.len())?;
        if offsets.is_empty() {
            return Ok(first);
        }
        for (ordinal, buffer) in arena.buffers.iter_mut().enumerate() {
            let context = device(ordinal)?;
            let order = context.upload_u32_slice(offsets)?;
            let (source, mut tail) = buffer.split_at_mut(first * arena.words);
            let source = source.as_view();
            let mut out = tail.slice_mut(..words as usize);
            let mut builder = context.stream().launch_builder(context.rwm_permute_words());
            let _ = builder.arg(&source);
            let _ = builder.arg(&order);
            let _ = builder.arg(&width);
            let _ = builder.arg(&words);
            let _ = builder.arg(&mut out);
            // SAFETY: checked source offsets precede the fresh reservation.
            // The views are disjoint and each thread copies one output word.
            let _ = unsafe { builder.launch(CudaKernelContext::launch_config(words)) }?;
        }
        Ok(first)
    })
    .ok_or_else(closed)?
}

pub(super) fn freeze(family: Family, count: usize, limbs: &[u64]) -> Result<(), CudaError> {
    flush_msms()?;
    with_arena(family, |arena| {
        if count * arena.words != limbs.len() {
            return Err(CudaError::LengthMismatch {
                expected: count * arena.words,
                got: limbs.len(),
            });
        }
        arena.frozen = count;
        arena.frozen_host = limbs.to_vec();
        Ok(())
    })
    .ok_or_else(closed)?
}

pub(super) fn write(family: Family, offset: usize, limbs: &[u64]) -> Result<(), CudaError> {
    flush_msms()?;
    with_arena(family, |arena| {
        if offset < arena.frozen {
            return Err(CudaError::InvariantViolation {
                reason: "a Dory arena write targeted the frozen setup prefix",
            });
        }
        if !limbs.len().is_multiple_of(arena.words) {
            return Err(CudaError::LengthMismatch {
                expected: arena.words,
                got: limbs.len(),
            });
        }
        let start = offset * arena.words;
        if start + limbs.len() > arena.limbs() {
            return Err(CudaError::LengthMismatch {
                expected: arena.limbs(),
                got: start + limbs.len(),
            });
        }
        arena.mirror(start, limbs)
    })
    .ok_or_else(closed)?
}

pub(super) fn read(family: Family, offset: usize, count: usize) -> Result<Vec<u64>, CudaError> {
    read_from(family, 0, offset, count)
}

fn read_from(
    family: Family,
    ordinal: usize,
    offset: usize,
    count: usize,
) -> Result<Vec<u64>, CudaError> {
    flush_msms()?;
    let context = device(ordinal)?;
    with_arena(family, |arena| {
        let start = offset * arena.words;
        let end = start + count * arena.words;
        if offset + count <= arena.frozen {
            return Ok(arena.frozen_host[start..end].to_vec());
        }
        if end > arena.limbs() {
            return Err(CudaError::LengthMismatch {
                expected: arena.limbs(),
                got: end,
            });
        }
        let buffer = arena
            .buffers
            .get(ordinal)
            .ok_or(CudaError::InvariantViolation {
                reason: "a Dory arena read named a device without a mirror",
            })?;
        context.read_u64_range(buffer, start, end)
    })
    .ok_or_else(closed)?
}

pub(super) fn axpy(family: Family, span: ResidentAxpy, scalar: Fr) -> Result<(), CudaError> {
    flush_msms()?;
    with_arena(family, |arena| {
        if span.out_offset < arena.frozen {
            return Err(CudaError::InvariantViolation {
                reason: "a resident axpy targeted the frozen setup prefix",
            });
        }
        if span.count == 0 {
            return Ok(());
        }
        // Validate the whole operation before splitting: a partial overlap
        // between inputs and output can otherwise disappear within each chunk.
        span.validate(arena.primary()?, arena.words)?;
        let split = arena.buffers.len() > 1 && span.count >= VECTOR_SPLIT_LEN;
        let chunks = if split {
            base_chunks(span.count, arena.buffers.len(), VECTOR_SPLIT_LEN)
        } else {
            vec![0..span.count; arena.buffers.len()]
        };
        for (ordinal, (buffer, chunk)) in arena.buffers.iter_mut().zip(&chunks).enumerate() {
            let context = device(ordinal)?;
            let part = ResidentAxpy {
                a_offset: span.a_offset + chunk.start,
                b_offset: span.b_offset + chunk.start,
                out_offset: span.out_offset + chunk.start,
                count: chunk.len(),
            };
            match family {
                Family::G1 => context.g1_axpy_in_place(buffer, part, scalar)?,
                Family::G2 => context.g2_axpy_in_place(buffer, part, scalar)?,
            }
        }
        if split {
            arena.exchange(span.out_offset, &chunks)?;
        }
        Ok(())
    })
    .ok_or_else(closed)?
}

pub(super) fn g1_msm(
    base_offset: usize,
    out_offset: usize,
    count: usize,
    scalars: &[Fr],
) -> Result<(), CudaError> {
    msm(Family::G1, base_offset, out_offset, count, scalars)
}

pub(super) fn g2_msm(
    base_offset: usize,
    out_offset: usize,
    count: usize,
    scalars: &[Fr],
) -> Result<(), CudaError> {
    msm(Family::G2, base_offset, out_offset, count, scalars)
}

const MSM_SPLIT_LEN: usize = 1 << 13;

const VECTOR_SPLIT_LEN: usize = 1 << 13;

const MIRROR_RETIRE_LEN: usize = 1 << 12;

fn base_chunks(count: usize, devices: usize, floor: usize) -> Vec<Range<usize>> {
    let devices = if devices < 2 || count < floor {
        1
    } else {
        devices.min(count.max(1))
    };
    let base = count / devices;
    let remainder = count % devices;
    let mut chunks = Vec::with_capacity(devices);
    let mut start = 0;
    for device in 0..devices {
        let len = base + usize::from(device < remainder);
        chunks.push(start..start + len);
        start += len;
    }
    chunks
}

fn msm(
    family: Family,
    base_offset: usize,
    out_offset: usize,
    count: usize,
    scalars: &[Fr],
) -> Result<(), CudaError> {
    if count > 0 && count < MIRROR_RETIRE_LEN && device_count() > 1 {
        if count != scalars.len() {
            return Err(CudaError::LengthMismatch {
                expected: count,
                got: scalars.len(),
            });
        }
        with_arena(family, |arena| {
            if out_offset < arena.frozen {
                return Err(CudaError::InvariantViolation {
                    reason: "a resident MSM targeted the frozen setup prefix",
                });
            }
            CudaKernelContext::resident_span_end(
                base_offset,
                count,
                arena.words,
                arena.primary()?,
            )?;
            CudaKernelContext::resident_span_end(out_offset, 1, arena.words, arena.primary()?)?;
            arena.buffers.truncate(1);
            Ok::<_, CudaError>(())
        })
        .ok_or_else(closed)??;
        let bases = base_offset..base_offset + count;
        let dependent = PENDING_MSMS.with_borrow(|requests| {
            requests.iter().any(|request| {
                request.family == family
                    && (bases.contains(&request.out)
                        || request.bases.contains(&out_offset)
                        || request.out == out_offset)
            })
        });
        if dependent {
            flush_msms()?;
        }
        PENDING_MSMS.with_borrow_mut(|requests| {
            requests.push(MsmRequest {
                family,
                bases,
                out: out_offset,
                scalars: scalars.to_vec(),
            });
        });
        return Ok(());
    }
    flush_msms()?;
    // Each device writes its partial into its own copy of `out_offset`.
    // `reduce` restores agreement before any mirror can be consumed.
    let split = with_arena(family, |arena| -> Result<bool, CudaError> {
        if out_offset < arena.frozen {
            return Err(CudaError::InvariantViolation {
                reason: "a resident MSM targeted the frozen setup prefix",
            });
        }
        if count < MIRROR_RETIRE_LEN {
            // `reduce` and vector updates leave the primary copy complete.
            // Removed mirrors are never recreated, even for later larger calls.
            arena.buffers.truncate(1);
        }
        let chunks = base_chunks(count, arena.buffers.len(), MSM_SPLIT_LEN);
        let split = chunks.len() > 1;
        let tasks: Vec<DeviceTask<'_, (), CudaError>> = chunks
            .into_iter()
            .zip(arena.buffers.iter_mut())
            .enumerate()
            .map(|(ordinal, (bases, buffer))| {
                let task: DeviceTask<'_, (), CudaError> = Box::new(move || {
                    let weights =
                        scalars
                            .get(bases.clone())
                            .ok_or(CudaError::InvariantViolation {
                                reason: "an MSM base chunk fell outside the scalar list",
                            })?;
                    let context = device(ordinal)?;
                    let start = base_offset + bases.start;
                    match family {
                        Family::G1 => {
                            context.g1_msm_in_place(buffer, start, out_offset, bases.len(), weights)
                        }
                        Family::G2 => {
                            context.g2_msm_in_place(buffer, start, out_offset, bases.len(), weights)
                        }
                    }
                });
                task
            })
            .collect();
        let _ = fan_out(tasks)?;
        Ok(split)
    })
    .ok_or_else(closed)??;
    if split {
        reduce(family, out_offset)
    } else {
        broadcast(family, out_offset, 1)
    }
}

fn reduce(family: Family, out_offset: usize) -> Result<(), CudaError> {
    let mut partials = Vec::with_capacity(mirrors(family));
    for ordinal in 0..mirrors(family) {
        partials.push(read_from(family, ordinal, out_offset, 1)?);
    }
    let total = match family {
        Family::G1 => g1_limbs(
            &partials
                .iter()
                .map(|limbs| g1_point(limbs))
                .fold(G1Projective::default(), |sum, point| sum + point),
        )
        .to_vec(),
        Family::G2 => g2_limbs(
            &partials
                .iter()
                .map(|limbs| g2_point(limbs))
                .fold(G2Projective::default(), |sum, point| sum + point),
        )
        .to_vec(),
    };
    write(family, out_offset, &total)
}

fn broadcast(family: Family, offset: usize, count: usize) -> Result<(), CudaError> {
    if mirrors(family) < 2 {
        return Ok(());
    }
    let limbs = read_from(family, 0, offset, count)?;
    with_arena(family, |arena| {
        let start = offset * arena.words;
        for (ordinal, buffer) in arena.buffers.iter_mut().enumerate().skip(1) {
            device(ordinal)?.write_u64_range(buffer, start, &limbs)?;
        }
        Ok(())
    })
    .ok_or_else(closed)?
}

pub(super) fn mirrors(family: Family) -> usize {
    with_arena(family, |arena| arena.buffers.len()).unwrap_or(0)
}

const MILLER_SPLIT_PAIRS: usize = 1 << 13;

fn lane_chunks(lanes: usize, devices: usize) -> Vec<Range<usize>> {
    let devices = devices.clamp(1, lanes.max(1));
    let base = lanes / devices;
    let remainder = lanes % devices;
    let mut chunks = Vec::with_capacity(devices);
    let mut start = 0;
    for device in 0..devices {
        let len = base + usize::from(device < remainder);
        chunks.push(start..start + len);
        start += len;
    }
    chunks
}

pub(super) fn multi_miller_batch(
    segments: &[(usize, usize)],
    count: usize,
) -> Result<Vec<u64>, CudaError> {
    let independent = count > 0
        && count < MIRROR_RETIRE_LEN
        && !segments.is_empty()
        && PENDING_MSMS.with_borrow(|requests| {
            !requests.is_empty()
                && requests.iter().all(|request| {
                    segments.iter().all(|&(g1, g2)| {
                        let start = match request.family {
                            Family::G1 => g1,
                            Family::G2 => g2,
                        };
                        !(start..start.saturating_add(count)).contains(&request.out)
                    })
                })
        });
    if independent && mirrors(Family::G1) == 1 && mirrors(Family::G2) == 1 {
        let _span = tracing::info_span!(
            "cuda_dory_tail_overlap",
            pairs = count,
            lanes = segments.len()
        )
        .entered();
        return G1_ARENA.with_borrow_mut(|g1| {
            G2_ARENA.with_borrow_mut(|g2| {
                let g1 = g1.as_mut().ok_or_else(closed)?;
                let g2 = g2.as_mut().ok_or_else(closed)?;
                let context = device(1)?;
                let ps = g1.copy_pairing_input(context, segments.iter().map(|&(p, _)| p), count)?;
                let qs = g2.copy_pairing_input(context, segments.iter().map(|&(_, q)| q), count)?;
                let spans: Vec<_> = ps.offsets.into_iter().zip(qs.offsets).collect();
                let requests = PENDING_MSMS.with_borrow_mut(std::mem::take);
                let tasks: Vec<DeviceTask<'_, Vec<u64>, CudaError>> = vec![
                    Box::new(move || {
                        for request in requests {
                            request.run(match request.family {
                                Family::G1 => g1,
                                Family::G2 => g2,
                            })?;
                        }
                        Ok(Vec::new())
                    }),
                    Box::new(move || {
                        context.multi_miller_batch(&ps.points, &qs.points, &spans, count)
                    }),
                ];
                Ok(fan_out(tasks)?.concat())
            })
        });
    }
    flush_msms()?;
    G1_ARENA.with_borrow(|g1| {
        G2_ARENA.with_borrow(|g2| match (g1.as_ref(), g2.as_ref()) {
            (Some(g1), Some(g2)) => miller_batch(g1, g2, segments, count),
            _ => Err(closed()),
        })
    })
}

fn miller_batch(
    g1: &Arena,
    g2: &Arena,
    segments: &[(usize, usize)],
    count: usize,
) -> Result<Vec<u64>, CudaError> {
    let pairs = g1.buffers.iter().zip(&g2.buffers);
    let mirrored: Vec<(&CudaSlice<u64>, &CudaSlice<u64>)> = pairs.collect();
    let split = mirrored.len() > 1
        && segments.len() > 1
        && segments.len().saturating_mul(count) >= MILLER_SPLIT_PAIRS;
    if !split {
        return device(0)?.multi_miller_batch(g1.primary()?, g2.primary()?, segments, count);
    }
    let tasks: Vec<DeviceTask<'_, Vec<u64>, CudaError>> =
        lane_chunks(segments.len(), mirrored.len())
            .into_iter()
            .zip(&mirrored)
            .enumerate()
            .map(|(ordinal, (lanes, &(g1, g2)))| {
                let task: DeviceTask<'_, Vec<u64>, CudaError> = Box::new(move || {
                    let lanes = segments.get(lanes).ok_or(CudaError::InvariantViolation {
                        reason: "a Dory Miller lane chunk fell outside the segment list",
                    })?;
                    device(ordinal)?.multi_miller_batch(g1, g2, lanes, count)
                });
                task
            })
            .collect();
    Ok(fan_out(tasks)?.concat())
}

pub(super) fn g2_fixed_base(
    base_offset: usize,
    out_offset: usize,
    scalars: &[Fr],
) -> Result<(), CudaError> {
    flush_msms()?;
    with_arena(Family::G2, |arena| {
        if out_offset < arena.frozen {
            return Err(CudaError::InvariantViolation {
                reason: "a resident fixed-base scaling targeted the frozen setup prefix",
            });
        }
        if scalars.is_empty() {
            return Ok(());
        }
        CudaKernelContext::validate_g2_fixed_base(
            arena.primary()?,
            base_offset,
            out_offset,
            scalars.len(),
        )?;
        let split = arena.buffers.len() > 1 && scalars.len() >= VECTOR_SPLIT_LEN;
        let chunks = if split {
            base_chunks(scalars.len(), arena.buffers.len(), VECTOR_SPLIT_LEN)
        } else {
            vec![0..scalars.len(); arena.buffers.len()]
        };
        for (ordinal, (buffer, chunk)) in arena.buffers.iter_mut().zip(&chunks).enumerate() {
            device(ordinal)?.g2_fixed_base_in_place(
                buffer,
                base_offset,
                out_offset + chunk.start,
                &scalars[chunk.clone()],
            )?;
        }
        if split {
            arena.exchange(out_offset, &chunks)?;
        }
        Ok(())
    })
    .ok_or_else(closed)?
}

fn fq(limbs: &[u64]) -> Fq {
    let mut value = [0u64; FQ_LIMBS];
    value.copy_from_slice(&limbs[..FQ_LIMBS]);
    Fq::new_unchecked(BigInt(value))
}

fn fq2(limbs: &[u64]) -> Fq2 {
    Fq2::new(fq(&limbs[..FQ_LIMBS]), fq(&limbs[FQ_LIMBS..2 * FQ_LIMBS]))
}

fn push_fq2(out: &mut [u64], value: &Fq2) {
    out[..FQ_LIMBS].copy_from_slice(&value.c0.0 .0);
    out[FQ_LIMBS..].copy_from_slice(&value.c1.0 .0);
}

pub(super) fn g1_limbs(point: &G1Projective) -> [u64; G1_WORDS] {
    let mut out = [0u64; G1_WORDS];
    out[..FQ_LIMBS].copy_from_slice(&point.x.0 .0);
    out[FQ_LIMBS..2 * FQ_LIMBS].copy_from_slice(&point.y.0 .0);
    out[2 * FQ_LIMBS..].copy_from_slice(&point.z.0 .0);
    out
}

pub(super) fn g1_point(limbs: &[u64]) -> G1Projective {
    G1Projective::new_unchecked(
        fq(&limbs[..FQ_LIMBS]),
        fq(&limbs[FQ_LIMBS..2 * FQ_LIMBS]),
        fq(&limbs[2 * FQ_LIMBS..]),
    )
}

pub(super) fn g2_limbs(point: &G2Projective) -> [u64; G2_WORDS] {
    let mut out = [0u64; G2_WORDS];
    let width = 2 * FQ_LIMBS;
    push_fq2(&mut out[..width], &point.x);
    push_fq2(&mut out[width..2 * width], &point.y);
    push_fq2(&mut out[2 * width..], &point.z);
    out
}

pub(super) fn g2_point(limbs: &[u64]) -> G2Projective {
    let width = 2 * FQ_LIMBS;
    G2Projective::new_unchecked(
        fq2(&limbs[..width]),
        fq2(&limbs[width..2 * width]),
        fq2(&limbs[2 * width..]),
    )
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test module: device storage failures fail loudly"
)]
mod tests {
    use ark_bn254::Fr;
    use ark_bn254::G1Affine;
    use ark_bn254::G1Projective;
    use ark_bn254::G2Affine;
    use ark_bn254::G2Projective;
    use ark_ec::PrimeGroup;
    use ark_ff::UniformRand;
    use dory::backends::arkworks::ArkFr;
    use dory::primitives::arithmetic::DoryRoutines;
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    use std::ops::Range;

    use ark_bn254::Bn254;
    use ark_ec::pairing::{MillerLoopOutput, Pairing};

    use dory::backends::arkworks::{ArkG1, ArkG2};
    use jolt_dory::{JoltG1Routines, JoltG2Routines};

    use super::super::curve::fq12;
    use super::super::handle::{span, span_g2, store_all, store_all_g2};
    use super::super::routines::{CudaG1Routines, CudaG2Routines};
    use super::{
        axpy, base_chunks, g2_fixed_base, lane_chunks, mirrors, multi_miller_batch, open, poisoned,
        read_from, Family, G1_WORDS, MILLER_SPLIT_PAIRS, MIRROR_RETIRE_LEN, MSM_SPLIT_LEN,
        VECTOR_SPLIT_LEN,
    };
    use super::{freeze, g1_limbs, g1_point, g2_limbs, g2_point, gather, read, reserve, write};
    use crate::cuda::common::context::{device_count, shared_context};
    use crate::cuda::common::error::CudaError;
    use crate::cuda::common::msm::ResidentAxpy;
    use crate::cuda::common::pairing::FQ12_LIMBS;

    #[test]
    fn gather_checks_bounds_and_preserves_every_mirror() {
        if shared_context().is_none() {
            return;
        }
        let _guard = open(4, 4).expect("open arena");
        let g1 = G1Projective::generator();
        let g2 = G2Projective::generator();
        for (family, a, b) in [
            (
                Family::G1,
                g1_limbs(&g1).to_vec(),
                g1_limbs(&(g1 + g1)).to_vec(),
            ),
            (
                Family::G2,
                g2_limbs(&g2).to_vec(),
                g2_limbs(&(g2 + g2)).to_vec(),
            ),
        ] {
            let source = [a.as_slice(), b.as_slice()].concat();
            let expected = [b.as_slice(), a.as_slice()].concat();
            assert_eq!(reserve(family, 2).expect("reserve source"), 0);
            write(family, 0, &source).expect("write source");
            assert!(matches!(
                gather(family, &[2]),
                Err(CudaError::InvariantViolation { .. })
            ));
            assert_eq!(gather(family, &[1, 0]).expect("gather reversed points"), 2);
            assert!(matches!(
                gather(family, &[0]),
                Err(CudaError::LengthMismatch { .. })
            ));
            assert_eq!(gather(family, &[]).expect("empty gather"), 4);
            assert_eq!(mirrors(family), device_count());
            for ordinal in 0..mirrors(family) {
                assert_eq!(read_from(family, ordinal, 0, 2).expect("source"), source);
                assert_eq!(read_from(family, ordinal, 2, 2).expect("gather"), expected);
            }
        }
        assert!(!poisoned());
    }

    #[test]
    fn reservations_frozen_prefixes_and_mirrors_preserve_point_storage() {
        if shared_context().is_none() {
            return;
        }
        let guard = open(4, 4).expect("open arena");
        let g1 = G1Projective::generator();
        let g2 = G2Projective::generator();
        for (family, first, second) in [
            (
                Family::G1,
                g1_limbs(&g1).to_vec(),
                g1_limbs(&(g1 + g1)).to_vec(),
            ),
            (
                Family::G2,
                g2_limbs(&g2).to_vec(),
                g2_limbs(&(g2 + g2)).to_vec(),
            ),
        ] {
            let points = [first.as_slice(), second.as_slice()].concat();
            assert_eq!(reserve(family, 2).expect("reserve first span"), 0);
            write(family, 0, &points).expect("write points");
            assert_eq!(mirrors(family), device_count());
            for ordinal in 0..mirrors(family) {
                assert_eq!(
                    read_from(family, ordinal, 0, 2).expect("read mirror"),
                    points
                );
            }
            freeze(family, 1, &first).expect("freeze setup prefix");
            assert_eq!(read(family, 0, 1).expect("read setup prefix"), first);
            assert!(matches!(
                write(family, 0, &second),
                Err(CudaError::InvariantViolation { .. })
            ));
            assert!(matches!(
                write(family, 2, &[0]),
                Err(CudaError::LengthMismatch { .. })
            ));
            assert!(matches!(
                write(family, 4, &second),
                Err(CudaError::LengthMismatch { .. })
            ));
            assert!(matches!(
                read(family, 4, 1),
                Err(CudaError::LengthMismatch { .. })
            ));
            assert!(matches!(
                freeze(family, 1, &[0]),
                Err(CudaError::LengthMismatch { .. })
            ));
            assert_eq!(
                read(family, 0, 1).expect("failed freeze preserves prefix"),
                first
            );
            assert!(matches!(
                reserve(family, 3),
                Err(CudaError::LengthMismatch { .. })
            ));
            assert_eq!(
                reserve(family, 2).expect("failed reservation preserves cursor"),
                2
            );
            assert!(matches!(
                reserve(family, 1),
                Err(CudaError::LengthMismatch { .. })
            ));
            write(family, 2, &points).expect("write remaining capacity");
            assert!(read(family, 4, 0).expect("empty end read").is_empty());
            for ordinal in 0..mirrors(family) {
                assert_eq!(
                    read_from(family, ordinal, 0, 4).expect("read full mirror"),
                    points.repeat(2)
                );
            }
            assert_eq!(g1_point(&g1_limbs(&g1)), g1);
            assert_eq!(g2_point(&g2_limbs(&g2)), g2);
        }
        assert!(!poisoned());
        drop(guard);
        for family in [Family::G1, Family::G2] {
            assert_eq!(mirrors(family), 0);
            assert!(matches!(
                read(family, 0, 1),
                Err(CudaError::InvariantViolation { .. })
            ));
            assert!(matches!(
                reserve(family, 1),
                Err(CudaError::InvariantViolation { .. })
            ));
        }
    }

    fn agree(family: Family, offset: usize, count: usize, what: &str) {
        let expected = read_from(family, 0, offset, count).expect("device 0 arena read");
        for ordinal in 1..mirrors(family) {
            assert_eq!(
                read_from(family, ordinal, offset, count).expect("mirror arena read"),
                expected,
                "{what}: device {ordinal}'s arena diverged from device 0",
            );
        }
    }

    fn scalars(count: usize, seed: u64) -> Vec<ArkFr> {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        (0..count).map(|_| ArkFr(Fr::rand(&mut rng))).collect()
    }

    #[test]
    fn distributed_axpy_preserves_mirrors_and_aliases() {
        if shared_context().is_none() {
            return;
        }
        let count = VECTOR_SPLIT_LEN + 3;
        let capacity = 1 + 3 * count;
        let _guard = open(capacity, capacity).expect("open arena");
        let scalar = scalars(1, 6_281)[0].0;
        let g1 = G1Projective::generator();
        let g2 = G2Projective::generator();
        let g1_pool: Vec<_> = (0..32).map(|i| g1 * Fr::from(i)).collect();
        let g2_pool: Vec<_> = (0..32).map(|i| g2 * Fr::from(i)).collect();

        for family in [Family::G1, Family::G2] {
            let initial: Vec<_> = (0..capacity)
                .flat_map(|i| match family {
                    Family::G1 => g1_limbs(&g1_pool[i % 32]).to_vec(),
                    Family::G2 => g2_limbs(&g2_pool[i % 32]).to_vec(),
                })
                .collect();
            assert_eq!(reserve(family, capacity).expect("reserve points"), 0);
            write(family, 0, &initial).expect("write points");
            freeze(family, 1, &initial[..family.words()]).expect("freeze prefix");
            for (a, b, out) in [
                (1, 1, 1),
                (1, 1 + count, 1),
                (1, 1 + count, 1 + count),
                (1, 1 + count, 1 + 2 * count),
                (1, 1, 1 + count),
            ] {
                write(family, 1, &initial[family.words()..]).expect("reset points");
                axpy(
                    family,
                    ResidentAxpy {
                        a_offset: a,
                        b_offset: b,
                        out_offset: out,
                        count,
                    },
                    scalar.into(),
                )
                .expect("distributed axpy");
                let coefficients: Vec<_> = (0..32)
                    .map(|i| {
                        Fr::from(((a + i) % 32) as u64) * scalar + Fr::from(((b + i) % 32) as u64)
                    })
                    .collect();
                let expected_g1: Vec<_> = coefficients.iter().map(|s| g1 * s).collect();
                let expected_g2: Vec<_> = coefficients.iter().map(|s| g2 * s).collect();
                assert_eq!(mirrors(family), device_count());
                for ordinal in 0..mirrors(family) {
                    let points = read_from(family, ordinal, 0, capacity).expect("read mirror");
                    for (i, limbs) in points.chunks_exact(family.words()).enumerate() {
                        let updated = (out..out + count).contains(&i);
                        match family {
                            Family::G1 => assert_eq!(
                                g1_point(limbs),
                                if updated {
                                    expected_g1[(i - out) % 32]
                                } else {
                                    g1_pool[i % 32]
                                },
                                "G1 mirror {ordinal}, point {i}",
                            ),
                            Family::G2 => assert_eq!(
                                g2_point(limbs),
                                if updated {
                                    expected_g2[(i - out) % 32]
                                } else {
                                    g2_pool[i % 32]
                                },
                                "G2 mirror {ordinal}, point {i}",
                            ),
                        }
                    }
                }
            }
            let before = read(family, 0, capacity).expect("read before rejection");
            for out_offset in [0, 1 + count.div_ceil(2), capacity] {
                assert!(axpy(
                    family,
                    ResidentAxpy {
                        a_offset: 1,
                        b_offset: 1,
                        out_offset,
                        count
                    },
                    scalar.into(),
                )
                .is_err());
            }
            for ordinal in 0..mirrors(family) {
                assert_eq!(
                    read_from(family, ordinal, 0, capacity).expect("unchanged mirror"),
                    before
                );
            }
        }
        assert!(!poisoned());
    }

    #[test]
    fn distributed_fixed_base_preserves_mirrors_and_rejects_overlap() {
        if shared_context().is_none() {
            return;
        }
        let count = VECTOR_SPLIT_LEN + 3;
        let _guard = open(1, count + 2).expect("open arena");
        let generator = G2Projective::generator();
        let weights = [
            Fr::from(0),
            Fr::from(1),
            -Fr::from(1),
            scalars(1, 6_283)[0].0,
        ];
        let expected: Vec<_> = weights.iter().map(|s| generator * s).collect();
        let scalars: Vec<_> = (0..count)
            .map(|i| weights[i % weights.len()].into())
            .collect();
        assert_eq!(reserve(Family::G2, count + 2).expect("reserve points"), 0);
        let prefix = [g2_limbs(&(-generator)), g2_limbs(&generator)].concat();
        write(Family::G2, 0, &prefix).expect("write prefix");
        freeze(Family::G2, 2, &prefix).expect("freeze prefix");
        g2_fixed_base(1, 2, &scalars).expect("distributed fixed-base update");
        assert_eq!(mirrors(Family::G2), device_count());
        for ordinal in 0..mirrors(Family::G2) {
            assert_eq!(
                read_from(Family::G2, ordinal, 0, 2).expect("prefix"),
                prefix
            );
            let points = read_from(Family::G2, ordinal, 2, count).expect("output");
            for (i, limbs) in points.chunks_exact(Family::G2.words()).enumerate() {
                assert_eq!(g2_point(limbs), expected[i % weights.len()]);
            }
        }
        let before = read(Family::G2, 0, count + 2).expect("read before rejection");
        for (base, out) in [(2 + count / 2, 2), (1, 0), (1, 3)] {
            assert!(g2_fixed_base(base, out, &scalars).is_err());
        }
        for ordinal in 0..mirrors(Family::G2) {
            assert_eq!(
                read_from(Family::G2, ordinal, 0, count + 2).expect("unchanged mirror"),
                before
            );
        }
        assert!(!poisoned());
    }

    #[test]
    fn base_chunks_decline_below_the_floor_and_partition_evenly_above_it() {
        for count in [0usize, 1, 7, MSM_SPLIT_LEN - 1] {
            for devices in [1usize, 2, 4] {
                assert_eq!(
                    base_chunks(count, devices, MSM_SPLIT_LEN),
                    vec![0..count],
                    "{count} base(s) over {devices} device(s) split below the floor",
                );
            }
        }
        for count in [MSM_SPLIT_LEN, MSM_SPLIT_LEN + 1, 3 * MSM_SPLIT_LEN + 5] {
            for devices in [1usize, 2, 3, 4, 8] {
                let chunks = base_chunks(count, devices, MSM_SPLIT_LEN);
                assert_eq!(
                    chunks.len(),
                    devices,
                    "{count} bases over {devices} device(s) produced {} chunk(s)",
                    chunks.len(),
                );
                let mut next = 0;
                for chunk in &chunks {
                    assert_eq!(chunk.start, next, "a gap at base {next}");
                    next = chunk.end;
                }
                assert_eq!(next, count, "the chunks do not cover the base list");
                let widths: Vec<usize> = chunks.iter().map(Range::len).collect();
                let spread = widths.iter().max().copied().unwrap_or(0)
                    - widths.iter().min().copied().unwrap_or(0);
                assert!(
                    spread <= 1,
                    "uneven base split across {devices}: {widths:?}"
                );
            }
        }
    }

    #[test]
    fn a_split_g1_msm_matches_reference_dory() {
        if shared_context().is_none() {
            return;
        }
        let len = MSM_SPLIT_LEN + 3;
        let Ok(guard) = open(len + 64, 64) else {
            return;
        };
        let mut rng = ChaCha20Rng::seed_from_u64(2_711);
        let bases: Vec<_> = (0..len)
            .map(|index| {
                if index % 37 == 0 {
                    G1Projective::default()
                } else {
                    G1Projective::rand(&mut rng)
                }
            })
            .collect();
        let weights = scalars(len, 2_713);
        let expected = JoltG1Routines::msm(
            &bases.iter().copied().map(ArkG1).collect::<Vec<_>>(),
            &weights,
        );

        let resident = store_all(&bases);
        let got = CudaG1Routines::msm(&resident, &weights).load();
        assert!(!poisoned(), "the split G1 MSM poisoned the arena");
        agree(
            Family::G1,
            span(&[CudaG1Routines::msm(&resident, &weights)]).expect("a G1 MSM output slot"),
            1,
            "after a split G1 MSM",
        );
        drop(guard);

        assert_eq!(
            G1Affine::from(got),
            G1Affine::from(expected.0),
            "a {len}-base split G1 MSM diverged from reference Dory",
        );
    }

    #[test]
    fn a_split_g2_msm_matches_reference_dory() {
        if shared_context().is_none() {
            return;
        }
        let len = MSM_SPLIT_LEN + 3;
        let Ok(guard) = open(64, len + 64) else {
            return;
        };
        let mut rng = ChaCha20Rng::seed_from_u64(2_719);
        let bases: Vec<_> = (0..len)
            .map(|index| {
                if index % 37 == 0 {
                    G2Projective::default()
                } else {
                    G2Projective::rand(&mut rng)
                }
            })
            .collect();
        let weights = scalars(len, 2_723);
        let expected = JoltG2Routines::msm(
            &bases.iter().copied().map(ArkG2).collect::<Vec<_>>(),
            &weights,
        );

        let resident = store_all_g2(&bases);
        let got = CudaG2Routines::msm(&resident, &weights).load();
        assert!(!poisoned(), "the split G2 MSM poisoned the arena");
        agree(
            Family::G2,
            span_g2(&[CudaG2Routines::msm(&resident, &weights)]).expect("a G2 MSM output slot"),
            1,
            "after a split G2 MSM",
        );
        drop(guard);

        assert_eq!(
            G2Affine::from(got),
            G2Affine::from(expected.0),
            "a {len}-base split G2 MSM diverged from reference Dory",
        );
    }

    #[test]
    fn lane_chunks_partition_every_lane_count_into_near_equal_runs() {
        for lanes in [1usize, 2, 3, 4, 5, 8, 17, 64] {
            for devices in [1usize, 2, 3, 4, 8, 64] {
                let chunks = lane_chunks(lanes, devices);
                assert!(
                    chunks.len() <= devices.min(lanes),
                    "{lanes} lane(s) over {devices} device(s) asked for {} chunk(s)",
                    chunks.len(),
                );
                let mut next = 0;
                for chunk in &chunks {
                    assert_eq!(
                        chunk.start, next,
                        "{lanes} lane(s) over {devices} device(s) leave a gap at lane {next}",
                    );
                    next = chunk.end;
                }
                assert_eq!(
                    next, lanes,
                    "{lanes} lane(s) over {devices} device(s) do not cover the lane list",
                );
                let widths: Vec<usize> = chunks.iter().map(Range::len).collect();
                let spread = widths.iter().max().copied().unwrap_or(0)
                    - widths.iter().min().copied().unwrap_or(0);
                assert!(
                    spread <= 1,
                    "{lanes} lane(s) over {devices} device(s) landed {spread} lanes apart: \
                     {widths:?}",
                );
            }
        }
    }

    #[test]
    fn a_split_miller_batch_matches_arkworks_lane_for_lane() {
        if shared_context().is_none() {
            return;
        }
        let lanes = 2 * device_count();
        let count = MILLER_SPLIT_PAIRS.div_ceil(lanes);
        let pairs = lanes * count;
        let Ok(guard) = open(pairs + 64, pairs + 64) else {
            return;
        };

        let mut rng = ChaCha20Rng::seed_from_u64(1_301);
        let ps: Vec<_> = (0..pairs).map(|_| G1Projective::rand(&mut rng)).collect();
        let qs: Vec<_> = (0..pairs).map(|_| G2Projective::rand(&mut rng)).collect();
        let g1 = store_all(&ps);
        let g2 = store_all_g2(&qs);
        let g1_base = span(&g1).expect("a contiguous G1 pair span");
        let g2_base = span_g2(&g2).expect("a contiguous G2 pair span");
        let segments: Vec<(usize, usize)> = (0..lanes)
            .map(|lane| (g1_base + lane * count, g2_base + lane * count))
            .collect();

        let limbs = multi_miller_batch(&segments, count).expect("a split Miller batch");
        assert!(!poisoned(), "the split Miller batch poisoned the arena");
        drop(guard);

        let got: Vec<_> = limbs.chunks_exact(FQ12_LIMBS).map(fq12).collect();
        assert_eq!(got.len(), lanes, "the split batch lost a lane");
        for (lane, value) in got.iter().enumerate() {
            let span = lane * count..(lane + 1) * count;
            let expected = Bn254::multi_miller_loop(ps[span.clone()].to_vec(), qs[span].to_vec()).0;
            assert_eq!(
                *value, expected,
                "lane {lane} of a {lanes}-lane x {count}-pair split batch diverged from arkworks",
            );
        }
    }

    #[test]
    fn small_round_msms_preserve_pairings_and_input_snapshots() {
        if shared_context().is_none() {
            return;
        }
        for count in [17usize, 129] {
            let _guard = open(4 * count + 16, 4 * count + 16).expect("arena");
            let g1 = G1Projective::generator();
            let g2 = G2Projective::generator();
            let p_scalars: Vec<_> = (0..2 * count).map(|i| Fr::from((i % 23) as u64)).collect();
            let q_scalars: Vec<_> = (0..2 * count).map(|i| Fr::from((i % 19) as u64)).collect();
            let ps: Vec<_> = p_scalars.iter().map(|s| g1 * s).collect();
            let qs: Vec<_> = q_scalars.iter().map(|s| g2 * s).collect();
            let hp = store_all(&ps);
            let hq = store_all_g2(&qs);
            let weights = scalars(count, 4_501);
            let p = CudaG1Routines::msm(&hp[..count], &weights);
            let q = CudaG2Routines::msm(&hq[..count], &weights);
            let p_start = span(&hp).expect("p span");
            let q_start = span_g2(&hq).expect("q span");
            let lanes = [(0, 0), (count, 0), (0, count), (count, count)];
            let segments: Vec<_> = lanes
                .iter()
                .map(|&(a, b)| (p_start + a, q_start + b))
                .collect();
            let paired = multi_miller_batch(&segments, count).expect("pairings with pending MSMs");
            assert_eq!(paired.len(), lanes.len() * FQ12_LIMBS);
            for ((a, b), limbs) in lanes.into_iter().zip(paired.chunks_exact(FQ12_LIMBS)) {
                let scalar: Fr = p_scalars[a..a + count]
                    .iter()
                    .zip(&q_scalars[b..b + count])
                    .map(|(p, q)| *p * q)
                    .sum();
                assert_eq!(
                    Bn254::final_exponentiation(MillerLoopOutput(fq12(limbs))).expect("final exp"),
                    Bn254::pairing(g1 * scalar, g2),
                );
            }
            let p_sum: Fr = p_scalars[..count]
                .iter()
                .zip(&weights)
                .map(|(p, s)| *p * s.0)
                .sum();
            let q_sum: Fr = q_scalars[..count]
                .iter()
                .zip(&weights)
                .map(|(q, s)| *q * s.0)
                .sum();
            assert_eq!(p.load(), g1 * p_sum);
            assert_eq!(q.load(), g2 * q_sum);

            let pending = CudaG1Routines::msm(&hp[..count], &weights);
            let dependent =
                multi_miller_batch(&[(span(&[pending]).expect("result"), q_start + 1)], 1)
                    .expect("pairing consumes an MSM");
            assert_eq!(
                Bn254::final_exponentiation(MillerLoopOutput(fq12(&dependent))).expect("final exp"),
                Bn254::pairing(g1 * p_sum, qs[1]),
            );
            let pending = CudaG2Routines::msm(&hq[..count], &weights);
            hq[1].overwrite(&(g2 + g2));
            assert_eq!(pending.load(), g2 * q_sum);
            assert_eq!(hq[1].load(), g2 + g2);

            let pending = CudaG1Routines::msm(&hp[..count], &weights);
            let dependent = CudaG1Routines::msm(&[pending], &weights[..1]);
            assert_eq!(dependent.load(), g1 * (p_sum * weights[0].0));

            let pending = CudaG1Routines::msm(&hp[..count], &weights);
            let _ = CudaG2Routines::msm(&hq[..count], &weights);
            assert!(multi_miller_batch(&[(usize::MAX, q_start)], 1).is_err());
            assert_eq!(pending.load(), g1 * p_sum);
            assert!(!poisoned());
        }
    }

    #[test]
    fn retired_arenas_preserve_later_group_operations() {
        if shared_context().is_none() || device_count() < 2 {
            return;
        }
        let len = MSM_SPLIT_LEN.max(MIRROR_RETIRE_LEN) + 1;
        let guard = open(4 * len + 64, 4 * len + 64).expect("open arena");
        let g1 = G1Projective::generator();
        let g2 = G2Projective::generator();
        let bases = store_all(&vec![g1; len]);
        let bases_g2 = store_all_g2(&vec![g2; len]);
        let weights = scalars(len, 1_927);
        let sum: Fr = weights.iter().map(|weight| weight.0).sum();

        assert_eq!(CudaG1Routines::msm(&bases, &weights).load(), g1 * sum);
        assert_eq!(CudaG2Routines::msm(&bases_g2, &weights).load(), g2 * sum);
        assert_eq!(mirrors(Family::G1), device_count());
        assert_eq!(mirrors(Family::G2), device_count());

        assert_eq!(
            CudaG1Routines::msm(&bases[..1], &weights[..1]).load(),
            g1 * weights[0].0
        );
        assert_eq!(mirrors(Family::G1), 1);
        assert_eq!(mirrors(Family::G2), device_count());
        let g1_offset = span(&bases).expect("G1 bases");
        let g2_offset = span_g2(&bases_g2).expect("G2 bases");
        let pair_count = MILLER_SPLIT_PAIRS / 2;
        let paired = multi_miller_batch(
            &[(g1_offset, g2_offset), (g1_offset + 1, g2_offset + 1)],
            pair_count,
        )
        .expect("pair across asymmetric arena counts");
        assert_eq!(paired.len(), 2 * FQ12_LIMBS);
        let expected_pairing = Bn254::pairing(g1 * Fr::from(pair_count as u64), g2);
        for lane in paired.chunks_exact(FQ12_LIMBS) {
            assert_eq!(
                Bn254::final_exponentiation(MillerLoopOutput(fq12(lane)))
                    .expect("final exponentiation"),
                expected_pairing
            );
        }

        assert_eq!(
            CudaG2Routines::msm(&bases_g2[..1], &weights[..1]).load(),
            g2 * weights[0].0
        );
        assert_eq!(mirrors(Family::G2), 1);
        let mut vs = store_all(&vec![g1; len]);
        let mut vs_g2 = store_all_g2(&vec![g2; len]);
        CudaG1Routines::fixed_scalar_mul_bases_then_add(&bases, &mut vs, &weights[0]);
        CudaG2Routines::fixed_scalar_mul_bases_then_add(&bases_g2, &mut vs_g2, &weights[1]);
        let updated_g1 = g1 * (Fr::from(1u64) + weights[0].0);
        let updated_g2 = g2 * (Fr::from(1u64) + weights[1].0);
        assert_eq!(CudaG1Routines::msm(&vs, &weights).load(), updated_g1 * sum);
        assert_eq!(
            CudaG2Routines::msm(&vs_g2, &weights).load(),
            updated_g2 * sum
        );

        let fixed = CudaG2Routines::fixed_base_vector_scalar_mul(&bases_g2[0], &weights[..3]);
        assert_eq!(fixed.len(), 3);
        for (point, weight) in fixed.iter().zip(&weights) {
            assert_eq!(point.load(), g2 * weight.0);
        }
        let gathered = gather(
            Family::G1,
            &[
                u32::try_from(span(&vs).expect("updated G1 span") + len - 1)
                    .expect("updated G1 offset fits u32"),
                u32::try_from(g1_offset).expect("G1 offset fits u32"),
            ],
        )
        .expect("gather after retirement");
        let copied = read(Family::G1, gathered, 2).expect("read gathered points");
        assert_eq!(
            copied
                .chunks_exact(G1_WORDS)
                .map(g1_point)
                .collect::<Vec<_>>(),
            vec![updated_g1, g1]
        );
        assert_eq!(mirrors(Family::G1), 1);
        assert_eq!(mirrors(Family::G2), 1);
        assert!(!poisoned());

        drop(guard);
        let _fresh = open(1, 1).expect("new opening");
        assert_eq!(mirrors(Family::G1), device_count());
        assert_eq!(mirrors(Family::G2), device_count());
    }

    #[test]
    fn every_device_mirror_holds_the_same_arena_bytes() {
        if shared_context().is_none() {
            return;
        }
        let len = 128usize;
        let Ok(guard) = open(8 * len, 8 * len) else {
            return;
        };
        assert_eq!(
            mirrors(Family::G1),
            device_count(),
            "the G1 arena did not mirror onto every device in the pool",
        );

        let mut rng = ChaCha20Rng::seed_from_u64(97);
        let g1: Vec<_> = (0..len).map(|_| G1Projective::rand(&mut rng)).collect();
        let g2: Vec<_> = (0..len).map(|_| G2Projective::rand(&mut rng)).collect();

        let bases = store_all(&g1);
        let mut vs = store_all(&g1);
        let bases_g2 = store_all_g2(&g2);
        let mut vs_g2 = store_all_g2(&g2);
        let base_offset = span(&bases).expect("a contiguous G1 base span");
        let vs_offset = span(&vs).expect("a contiguous G1 vs span");
        let g2_base_offset = span_g2(&bases_g2).expect("a contiguous G2 base span");
        let g2_vs_offset = span_g2(&vs_g2).expect("a contiguous G2 vs span");
        agree(Family::G1, base_offset, len, "after a G1 store");
        agree(Family::G2, g2_base_offset, len, "after a G2 store");

        let weights = scalars(len, 101);
        CudaG1Routines::fixed_scalar_mul_bases_then_add(&bases, &mut vs, &weights[0]);
        agree(Family::G1, vs_offset, len, "after a G1 axpy");
        CudaG2Routines::fixed_scalar_mul_bases_then_add(&bases_g2, &mut vs_g2, &weights[1]);
        agree(Family::G2, g2_vs_offset, len, "after a G2 axpy");

        let fixed = CudaG2Routines::fixed_base_vector_scalar_mul(&bases_g2[0], &weights[..8]);
        let fixed_offset = span_g2(&fixed).expect("a contiguous G2 fixed-base span");
        agree(Family::G2, fixed_offset, 8, "after a G2 fixed-base scaling");

        let product = CudaG1Routines::msm(&bases, &weights);
        agree(
            Family::G1,
            span(&[product]).expect("a G1 MSM output slot"),
            1,
            "after a G1 MSM",
        );
        let product_g2 = CudaG2Routines::msm(&bases_g2, &weights);
        agree(
            Family::G2,
            span_g2(&[product_g2]).expect("a G2 MSM output slot"),
            1,
            "after a G2 MSM",
        );

        assert!(!poisoned(), "the mirrored arena poisoned");
        drop(guard);
    }
}
