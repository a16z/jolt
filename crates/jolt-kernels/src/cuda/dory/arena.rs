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
use crate::cuda::common::context::{context_for, device_count, CudaKernelContext};
#[cfg(test)]
use crate::cuda::common::error::CudaError;
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
pub(super) fn mirrors(family: Family) -> usize {
    with_arena(family, |arena| arena.buffers.len()).unwrap_or(0)
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
    use ark_ec::PrimeGroup;

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
}
