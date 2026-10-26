//! Point storage and handles for a Dory opening on the calling host thread.
//!
//! Unlike CPU group values, handles address append-only device buffers. Other
//! devices hold mirrors; a failed mirror allocation leaves fewer devices in use.
//! A frozen setup prefix is also cached on the host. The guard releases the
//! buffers, and handle storage failures set a thread-local poison flag.
//! Callers must keep handles within the owning arena's lifetime and thread;
//! the offset representation does not enforce that contract.

#[cfg(test)]
use ark_bn254::{Fq, G1Projective};
use ark_bn254::{Fq2, G2Projective};
#[cfg(test)]
use ark_ff::BigInt;
#[cfg(test)]
use cudarc::driver::CudaSlice;
#[cfg(test)]
use std::cell::{Cell, RefCell};
#[cfg(test)]
use std::ops::Range;

#[cfg(test)]
use jolt_field::Fr;

#[cfg(test)]
use crate::cuda::common::context::{context_for, device_count, CudaKernelContext};
#[cfg(test)]
use crate::cuda::common::devices::{fan_out, DeviceTask};
#[cfg(test)]
use crate::cuda::common::error::CudaError;
#[cfg(test)]
use crate::cuda::common::msm::ResidentAxpy;
use crate::cuda::common::msm::FQ_LIMBS;

#[cfg(test)]
pub(super) const G1_WORDS: usize = 3 * FQ_LIMBS;

pub(super) const G2_WORDS: usize = 6 * FQ_LIMBS;

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Family {
    G1,
    G2,
}

#[cfg(test)]
impl Family {
    const fn words(self) -> usize {
        match self {
            Self::G1 => G1_WORDS,
            Self::G2 => G2_WORDS,
        }
    }
}

#[cfg(test)]
struct Arena {
    buffers: Vec<CudaSlice<u64>>,
    words: usize,
    used: usize,
    capacity: usize,
    frozen: usize,
    frozen_host: Vec<u64>,
}

#[cfg(test)]
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

    const fn limbs(&self) -> usize {
        self.capacity * self.words
    }

    fn mirror(&mut self, start: usize, limbs: &[u64]) -> Result<(), CudaError> {
        for (ordinal, buffer) in self.buffers.iter_mut().enumerate() {
            device(ordinal)?.write_u64_range(buffer, start, limbs)?;
        }
        Ok(())
    }
}

#[cfg(test)]
thread_local! {
    static G1_ARENA: RefCell<Option<Arena>> = const { RefCell::new(None) };
    static G2_ARENA: RefCell<Option<Arena>> = const { RefCell::new(None) };
    static POISONED: Cell<bool> = const { Cell::new(false) };
}

#[cfg(test)]
pub(super) fn poison(reason: &'static str) {
    tracing::error!(
        reason,
        "the Dory arena was poisoned; the opening will be rejected"
    );
    POISONED.set(true);
}

#[cfg(test)]
pub(super) fn poisoned() -> bool {
    POISONED.get()
}

#[cfg(test)]
fn with_arena<R>(family: Family, act: impl FnOnce(&mut Arena) -> R) -> Option<R> {
    match family {
        Family::G1 => G1_ARENA.with_borrow_mut(|slot| slot.as_mut().map(act)),
        Family::G2 => G2_ARENA.with_borrow_mut(|slot| slot.as_mut().map(act)),
    }
}

#[cfg(test)]
const fn closed() -> CudaError {
    CudaError::InvariantViolation {
        reason: "the Dory arena is not open",
    }
}

#[cfg(test)]
fn device(ordinal: usize) -> Result<&'static CudaKernelContext, CudaError> {
    context_for(ordinal).ok_or(CudaError::NotImplemented {
        kernel: "no CUDA device is present for the Dory arenas",
    })
}

#[cfg(test)]
pub(super) struct ArenaGuard;

#[cfg(test)]
impl Drop for ArenaGuard {
    fn drop(&mut self) {
        G1_ARENA.with_borrow_mut(|slot| *slot = None);
        G2_ARENA.with_borrow_mut(|slot| *slot = None);
    }
}

#[cfg(test)]
pub(super) fn open(g1_capacity: usize, g2_capacity: usize) -> Result<ArenaGuard, CudaError> {
    let g1 = Arena::new(Family::G1, g1_capacity)?;
    let g2 = Arena::new(Family::G2, g2_capacity)?;
    G1_ARENA.with_borrow_mut(|slot| *slot = Some(g1));
    G2_ARENA.with_borrow_mut(|slot| *slot = Some(g2));
    POISONED.set(false);
    Ok(ArenaGuard)
}

#[cfg(test)]
pub(super) fn reserve(family: Family, count: usize) -> Result<usize, CudaError> {
    with_arena(family, |arena| {
        if arena.used + count > arena.capacity {
            return Err(CudaError::LengthMismatch {
                expected: arena.capacity,
                got: arena.used + count,
            });
        }
        let offset = arena.used;
        arena.used += count;
        Ok(offset)
    })
    .ok_or_else(closed)?
}

#[cfg(test)]
pub(super) fn freeze(family: Family, count: usize, limbs: &[u64]) -> Result<(), CudaError> {
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

#[cfg(test)]
pub(super) fn write(family: Family, offset: usize, limbs: &[u64]) -> Result<(), CudaError> {
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

#[cfg(test)]
pub(super) fn read(family: Family, offset: usize, count: usize) -> Result<Vec<u64>, CudaError> {
    read_from(family, 0, offset, count)
}

#[cfg(test)]
fn read_from(
    family: Family,
    ordinal: usize,
    offset: usize,
    count: usize,
) -> Result<Vec<u64>, CudaError> {
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

#[cfg(test)]
pub(super) fn axpy(family: Family, span: ResidentAxpy, scalar: Fr) -> Result<(), CudaError> {
    with_arena(family, |arena| {
        if span.out_offset < arena.frozen {
            return Err(CudaError::InvariantViolation {
                reason: "a resident axpy targeted the frozen setup prefix",
            });
        }
        for (ordinal, buffer) in arena.buffers.iter_mut().enumerate() {
            let context = device(ordinal)?;
            match family {
                Family::G1 => context.g1_axpy_in_place(buffer, span, scalar)?,
                Family::G2 => context.g2_axpy_in_place(buffer, span, scalar)?,
            }
        }
        Ok(())
    })
    .ok_or_else(closed)?
}

#[cfg(test)]
pub(super) fn g1_msm(
    base_offset: usize,
    out_offset: usize,
    count: usize,
    scalars: &[Fr],
) -> Result<(), CudaError> {
    msm(Family::G1, base_offset, out_offset, count, scalars)
}

#[cfg(test)]
pub(super) fn g2_msm(
    base_offset: usize,
    out_offset: usize,
    count: usize,
    scalars: &[Fr],
) -> Result<(), CudaError> {
    msm(Family::G2, base_offset, out_offset, count, scalars)
}

#[cfg(test)]
const MSM_SPLIT_LEN: usize = 1 << 13;

#[cfg(test)]
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

/// Each device reduces its own contiguous slice of the bases into ITS OWN copy
/// of `out_offset`, so the mirrors diverge for the duration of the fan-out;
/// `reduce` sums the partials and re-broadcasts the total before returning.
#[cfg(test)]
fn msm(
    family: Family,
    base_offset: usize,
    out_offset: usize,
    count: usize,
    scalars: &[Fr],
) -> Result<(), CudaError> {
    let split = with_arena(family, |arena| -> Result<bool, CudaError> {
        if out_offset < arena.frozen {
            return Err(CudaError::InvariantViolation {
                reason: "a resident MSM targeted the frozen setup prefix",
            });
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

#[cfg(test)]
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

#[cfg(test)]
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

#[cfg(test)]
pub(super) fn mirrors(family: Family) -> usize {
    with_arena(family, |arena| arena.buffers.len()).unwrap_or(0)
}

#[cfg(test)]
pub(super) fn g2_fixed_base(
    base_offset: usize,
    out_offset: usize,
    scalars: &[Fr],
) -> Result<(), CudaError> {
    with_arena(Family::G2, |arena| {
        if out_offset < arena.frozen {
            return Err(CudaError::InvariantViolation {
                reason: "a resident fixed-base scaling targeted the frozen setup prefix",
            });
        }
        for (ordinal, buffer) in arena.buffers.iter_mut().enumerate() {
            device(ordinal)?.g2_fixed_base_in_place(buffer, base_offset, out_offset, scalars)?;
        }
        Ok(())
    })
    .ok_or_else(closed)?
}

#[cfg(test)]
fn fq(limbs: &[u64]) -> Fq {
    let mut value = [0u64; FQ_LIMBS];
    value.copy_from_slice(&limbs[..FQ_LIMBS]);
    Fq::new_unchecked(BigInt(value))
}

#[cfg(test)]
fn fq2(limbs: &[u64]) -> Fq2 {
    Fq2::new(fq(&limbs[..FQ_LIMBS]), fq(&limbs[FQ_LIMBS..2 * FQ_LIMBS]))
}

fn push_fq2(out: &mut [u64], value: &Fq2) {
    out[..FQ_LIMBS].copy_from_slice(&value.c0.0 .0);
    out[FQ_LIMBS..].copy_from_slice(&value.c1.0 .0);
}

#[cfg(test)]
pub(super) fn g1_limbs(point: &G1Projective) -> [u64; G1_WORDS] {
    let mut out = [0u64; G1_WORDS];
    out[..FQ_LIMBS].copy_from_slice(&point.x.0 .0);
    out[FQ_LIMBS..2 * FQ_LIMBS].copy_from_slice(&point.y.0 .0);
    out[2 * FQ_LIMBS..].copy_from_slice(&point.z.0 .0);
    out
}

#[cfg(test)]
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

#[cfg(test)]
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
    use ark_bn254::{Fr, G1Affine, G1Projective, G2Affine, G2Projective};
    use ark_ec::PrimeGroup;
    use ark_ff::UniformRand;
    use dory::backends::arkworks::{ArkFr, ArkG1, ArkG2};
    use dory::primitives::arithmetic::DoryRoutines;
    use jolt_dory::{JoltG1Routines, JoltG2Routines};
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    use super::super::handle::{span, span_g2, store_all, store_all_g2};
    use super::super::routines::{CudaG1Routines, CudaG2Routines};

    use super::*;
    use crate::cuda::common::context::shared_context;

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
