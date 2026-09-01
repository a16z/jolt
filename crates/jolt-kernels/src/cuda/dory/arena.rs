use std::cell::{Cell, RefCell};
use std::ops::Range;

use ark_bn254::{Fq, Fq2, G1Projective, G2Projective};
use ark_ff::BigInt;
use cudarc::driver::CudaSlice;
use jolt_field::Fr;

use crate::cuda::common::context::{context_for, device_count, CudaKernelContext};
use crate::cuda::common::devices::{fan_out, DeviceTask};
use crate::cuda::common::error::CudaError;
use crate::cuda::common::msm::{ResidentAxpy, FQ_LIMBS};

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
}

thread_local! {
    static G1_ARENA: RefCell<Option<Arena>> = const { RefCell::new(None) };
    static G2_ARENA: RefCell<Option<Arena>> = const { RefCell::new(None) };
    static POISONED: Cell<bool> = const { Cell::new(false) };
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
        G1_ARENA.with_borrow_mut(|slot| *slot = None);
        G2_ARENA.with_borrow_mut(|slot| *slot = None);
    }
}

pub(super) fn open(g1_capacity: usize, g2_capacity: usize) -> Result<ArenaGuard, CudaError> {
    let g1 = Arena::new(Family::G1, g1_capacity)?;
    let g2 = Arena::new(Family::G2, g2_capacity)?;
    G1_ARENA.with_borrow_mut(|slot| *slot = Some(g1));
    G2_ARENA.with_borrow_mut(|slot| *slot = Some(g2));
    POISONED.set(false);
    Ok(ArenaGuard)
}

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

pub(super) fn read(family: Family, offset: usize, count: usize) -> Result<Vec<u64>, CudaError> {
    read_from(family, 0, offset, count)
}

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
    reason = "test module: device operations fail loudly"
)]
mod tests {
    use ark_ff::UniformRand;
    use dory::backends::arkworks::ArkFr;
    use dory::primitives::arithmetic::DoryRoutines;
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    use std::ops::Range;

    use ark_bn254::Bn254;
    use ark_ec::pairing::Pairing;

    use dory::backends::arkworks::{ArkG1, ArkG2};
    use jolt_dory::{JoltG1Routines, JoltG2Routines};

    use super::super::curve::fq12;
    use super::super::handle::{span, span_g2, store_all, store_all_g2};
    use super::super::routines::{CudaG1Routines, CudaG2Routines};
    use super::{
        base_chunks, lane_chunks, mirrors, multi_miller_batch, open, poisoned, read_from, Family,
        MILLER_SPLIT_PAIRS, MSM_SPLIT_LEN,
    };
    use crate::cuda::common::context::{device_count, shared_context};
    use crate::cuda::common::pairing::FQ12_LIMBS;

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
        (0..count)
            .map(|_| ArkFr(ark_bn254::Fr::rand(&mut rng)))
            .collect()
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
                    ark_bn254::G1Projective::default()
                } else {
                    ark_bn254::G1Projective::rand(&mut rng)
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
            ark_bn254::G1Affine::from(got),
            ark_bn254::G1Affine::from(expected.0),
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
                    ark_bn254::G2Projective::default()
                } else {
                    ark_bn254::G2Projective::rand(&mut rng)
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
            ark_bn254::G2Affine::from(got),
            ark_bn254::G2Affine::from(expected.0),
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
        let ps: Vec<_> = (0..pairs)
            .map(|_| ark_bn254::G1Projective::rand(&mut rng))
            .collect();
        let qs: Vec<_> = (0..pairs)
            .map(|_| ark_bn254::G2Projective::rand(&mut rng))
            .collect();
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
        let g1: Vec<_> = (0..len)
            .map(|_| ark_bn254::G1Projective::rand(&mut rng))
            .collect();
        let g2: Vec<_> = (0..len)
            .map(|_| ark_bn254::G2Projective::rand(&mut rng))
            .collect();

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
