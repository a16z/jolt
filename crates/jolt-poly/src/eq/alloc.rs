//! Allocation policies for constructor-owned equality tables.
//!
//! Budgets exclude allocator slack, outer Vec headers, callbacks, Rayon bookkeeping,
//! and allocations made by later operations on a constructed table.
use core::convert::Infallible;
use jolt_field::Field;
use jolt_utils::alloc::ZeroAllocError;
use sealed::Sealed;
use std::{alloc::Layout, collections::TryReserveError, panic::Location};
use thiserror::Error;

pub(crate) mod sealed {
    pub trait Sealed {}
}

/// Sealed allocation behavior; table lengths and storage capacities are guaranteed.
pub trait EqAlloc: Sealed + Copy + Send + Sync {
    type Error: Send;
    /// Report invalid shape or arguments; `Global` panics and `Budget` returns the error.
    fn reject(self, error: EqTableError) -> Self::Error;
    /// Admit one family before allocating any of its buffers.
    fn admit<F: Field>(
        self,
        elements: usize,
        at: &'static Location<'static>,
    ) -> Result<(), Self::Error>;
    /// Allocate exactly `len` initialized elements after family admission.
    fn table<F: Field>(self, len: usize, init: TableInit<F>) -> Result<Vec<F>, Self::Error>;
    /// Return empty storage with capacity at least `len`.
    fn storage<T>(self, len: usize) -> Result<Vec<T>, Self::Error>;
}

pub enum TableInit<F> {
    Zero,
    Repeat(F),
}

#[derive(Clone, Copy)]
pub struct Global;

/// A per-family byte ceiling, not a running balance. Copies share no mutable state.
///
/// Flat and mapped builders admit their output, prefix builders admit only their
/// requested prefix, cached builders admit all levels together, and tensor builders
/// admit each factor separately. Bytes use `max(size_of::<F>(), 1)` per element.
/// Outer cache storage is fallible but excluded from the cap. Allocator slack,
/// callbacks, Rayon bookkeeping, and later operations on a table are not covered.
#[derive(Clone, Copy)]
pub struct Budget {
    pub max_bytes: usize,
}

#[derive(Debug, Error)]
pub enum AllocationCause {
    #[error("{0}")]
    Reserve(#[from] TryReserveError),
    #[error("zero allocation failed: {0:?}")]
    Zero(ZeroAllocError),
}

#[derive(Debug, Error)]
pub enum EqTableError {
    #[error("equality table capacity overflow ({num_vars} variables)")]
    DimensionOverflow { num_vars: usize },
    #[error("equality table capacity overflow ({elements} elements of {element_size} bytes)")]
    SizeOverflow {
        elements: usize,
        element_size: usize,
    },
    #[error("equality table requests {bytes} bytes, exceeding {max_bytes} at {at}")]
    OverBudget {
        bytes: usize,
        max_bytes: usize,
        at: &'static Location<'static>,
    },
    #[error("equality table allocation of {elements} elements failed: {cause}")]
    AllocationFailed {
        elements: usize,
        #[source]
        cause: AllocationCause,
    },
    #[error("invalid equality-table prefix (assertion failed: start <= len): expected at most {expected}, got {got}")]
    LengthMismatch { expected: usize, got: usize },
    #[error("index out of bounds: the len is {len} but the index is {index}")]
    IndexOutOfRange { index: usize, len: usize },
    #[error("assertion failed: range.start <= range.end && range.end <= self.0.len() (split {outer_vars}, variables {num_vars})")]
    InvalidSplit { outer_vars: usize, num_vars: usize },
}

impl Sealed for Global {}
impl EqAlloc for Global {
    type Error = Infallible;
    #[expect(
        clippy::panic,
        reason = "Global preserves infallible builders' argument and capacity panics"
    )]
    fn reject(self, error: EqTableError) -> Infallible {
        panic!("{error}")
    }
    #[inline]
    fn admit<F: Field>(self, _: usize, _: &'static Location<'static>) -> Result<(), Infallible> {
        Ok(())
    }
    #[inline]
    fn table<F: Field>(self, len: usize, init: TableInit<F>) -> Result<Vec<F>, Infallible> {
        Ok(match init {
            TableInit::Zero => F::zeroed_vec(len),
            TableInit::Repeat(value) => vec![value; len],
        })
    }
    #[inline]
    fn storage<T>(self, len: usize) -> Result<Vec<T>, Infallible> {
        Ok(Vec::with_capacity(len))
    }
}

impl Sealed for Budget {}
impl EqAlloc for Budget {
    type Error = EqTableError;
    fn reject(self, error: EqTableError) -> EqTableError {
        error
    }
    fn admit<F: Field>(
        self,
        elements: usize,
        at: &'static Location<'static>,
    ) -> Result<(), EqTableError> {
        let element_size = size_of::<F>().max(1);
        let bytes = elements
            .checked_mul(element_size)
            .ok_or(EqTableError::SizeOverflow {
                elements,
                element_size,
            })?;
        if bytes > self.max_bytes {
            return Err(EqTableError::OverBudget {
                bytes,
                max_bytes: self.max_bytes,
                at,
            });
        }
        Ok(())
    }
    fn table<F: Field>(self, len: usize, init: TableInit<F>) -> Result<Vec<F>, EqTableError> {
        match init {
            TableInit::Zero => {
                F::try_zeroed_vec(len).map_err(|cause| EqTableError::AllocationFailed {
                    elements: len,
                    cause: AllocationCause::Zero(cause),
                })
            }
            TableInit::Repeat(value) => {
                let mut table = self.storage(len)?;
                table.resize(len, value);
                Ok(table)
            }
        }
    }
    fn storage<T>(self, len: usize) -> Result<Vec<T>, EqTableError> {
        let mut values = Vec::new();
        values
            .try_reserve_exact(len)
            .map_err(|cause| EqTableError::AllocationFailed {
                elements: len,
                cause: AllocationCause::Reserve(cause),
            })?;
        Ok(values)
    }
}

pub(crate) fn domain_len<A: EqAlloc>(num_vars: usize, policy: A) -> Result<usize, A::Error> {
    u32::try_from(num_vars)
        .ok()
        .and_then(|bits| 1usize.checked_shl(bits))
        .ok_or_else(|| policy.reject(EqTableError::DimensionOverflow { num_vars }))
}

/// Check individual buffers, never the combined size of a multi-buffer family.
pub(crate) fn check_layout<T, A: EqAlloc>(elements: usize, policy: A) -> Result<(), A::Error> {
    Layout::array::<T>(elements).map(|_| ()).map_err(|_| {
        policy.reject(EqTableError::SizeOverflow {
            elements,
            element_size: size_of::<T>().max(1),
        })
    })
}
