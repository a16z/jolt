use std::cell::{Cell, RefCell};

use ark_bn254::{Fq, Fq2, G1Projective, G2Projective};
use ark_ff::BigInt;
use cudarc::driver::CudaSlice;
use jolt_field::Fr;

use crate::cuda::common::context::{shared_context, CudaKernelContext};
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
    buffer: CudaSlice<u64>,
    words: usize,
    used: usize,
    capacity: usize,
    frozen: usize,
    frozen_host: Vec<u64>,
}

impl Arena {
    fn new(
        context: &CudaKernelContext,
        family: Family,
        capacity: usize,
    ) -> Result<Self, CudaError> {
        Ok(Self {
            buffer: context.alloc_u64(capacity * family.words())?,
            words: family.words(),
            used: 0,
            capacity,
            frozen: 0,
            frozen_host: Vec::new(),
        })
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

fn context() -> Result<&'static CudaKernelContext, CudaError> {
    shared_context().ok_or(CudaError::NotImplemented {
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
    let context = context()?;
    let g1 = Arena::new(context, Family::G1, g1_capacity)?;
    let g2 = Arena::new(context, Family::G2, g2_capacity)?;
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
    let context = context()?;
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
        if start + limbs.len() > arena.buffer.len() {
            return Err(CudaError::LengthMismatch {
                expected: arena.buffer.len(),
                got: start + limbs.len(),
            });
        }
        context.write_u64_range(&mut arena.buffer, start, limbs)
    })
    .ok_or_else(closed)?
}

pub(super) fn read(family: Family, offset: usize, count: usize) -> Result<Vec<u64>, CudaError> {
    let context = context()?;
    with_arena(family, |arena| {
        let start = offset * arena.words;
        let end = start + count * arena.words;
        if offset + count <= arena.frozen {
            return Ok(arena.frozen_host[start..end].to_vec());
        }
        if end > arena.buffer.len() {
            return Err(CudaError::LengthMismatch {
                expected: arena.buffer.len(),
                got: end,
            });
        }
        context.read_u64_range(&arena.buffer, start, end)
    })
    .ok_or_else(closed)?
}

pub(super) fn axpy(family: Family, span: ResidentAxpy, scalar: Fr) -> Result<(), CudaError> {
    let context = context()?;
    with_arena(family, |arena| {
        if span.out_offset < arena.frozen {
            return Err(CudaError::InvariantViolation {
                reason: "a resident axpy targeted the frozen setup prefix",
            });
        }
        match family {
            Family::G1 => context.g1_axpy_in_place(&mut arena.buffer, span, scalar),
            Family::G2 => context.g2_axpy_in_place(&mut arena.buffer, span, scalar),
        }
    })
    .ok_or_else(closed)?
}

pub(super) fn g1_msm(
    base_offset: usize,
    out_offset: usize,
    count: usize,
    scalars: &[Fr],
) -> Result<(), CudaError> {
    let context = context()?;
    with_arena(Family::G1, |arena| {
        if out_offset < arena.frozen {
            return Err(CudaError::InvariantViolation {
                reason: "a resident G1 MSM targeted the frozen setup prefix",
            });
        }
        context.g1_msm_in_place(&mut arena.buffer, base_offset, out_offset, count, scalars)
    })
    .ok_or_else(closed)?
}

pub(super) fn g2_msm(
    base_offset: usize,
    out_offset: usize,
    count: usize,
    scalars: &[Fr],
) -> Result<(), CudaError> {
    let context = context()?;
    with_arena(Family::G2, |arena| {
        if out_offset < arena.frozen {
            return Err(CudaError::InvariantViolation {
                reason: "a resident G2 MSM targeted the frozen setup prefix",
            });
        }
        context.g2_msm_in_place(&mut arena.buffer, base_offset, out_offset, count, scalars)
    })
    .ok_or_else(closed)?
}

pub(super) fn multi_miller_batch(
    segments: &[(usize, usize)],
    count: usize,
) -> Result<Vec<u64>, CudaError> {
    let context = context()?;
    G1_ARENA.with_borrow(|g1| {
        G2_ARENA.with_borrow(|g2| match (g1.as_ref(), g2.as_ref()) {
            (Some(g1), Some(g2)) => {
                context.multi_miller_batch(&g1.buffer, &g2.buffer, segments, count)
            }
            _ => Err(closed()),
        })
    })
}

pub(super) fn g2_fixed_base(
    base_offset: usize,
    out_offset: usize,
    scalars: &[Fr],
) -> Result<(), CudaError> {
    let context = context()?;
    with_arena(Family::G2, |arena| {
        if out_offset < arena.frozen {
            return Err(CudaError::InvariantViolation {
                reason: "a resident fixed-base scaling targeted the frozen setup prefix",
            });
        }
        context.g2_fixed_base_in_place(&mut arena.buffer, base_offset, out_offset, scalars)
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
