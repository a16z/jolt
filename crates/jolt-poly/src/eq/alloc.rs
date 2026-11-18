//! Allocation policies for constructor-owned equality tables.
//!
//! Budgets exclude allocator slack, outer Vec headers, callbacks, Rayon bookkeeping,
//! and allocations made by later operations on a constructed table.
use core::convert::Infallible;
use jolt_field::Field;
use jolt_utils::alloc::ZeroAllocError;
use std::{alloc::Layout, collections::TryReserveError, panic::Location};
use thiserror::Error;

/// Allocation policy for the eq-table builders: [`Global`] or [`Budget`].
///
/// Sealed, with no methods callable outside jolt-poly: other crates can only pass
/// a policy to a builder, which admits each allocation family before allocating.
#[expect(
    private_bounds,
    reason = "Allocate seals EqAlloc and keeps its allocation methods private to jolt-poly"
)]
pub trait EqAlloc: Allocate + Copy + Send + Sync {
    type Error: Send;
}

/// The allocation methods behind [`EqAlloc`]; table lengths and storage
/// capacities are guaranteed.
pub(crate) trait Allocate {
    /// Report invalid shape or arguments; `Global` panics and `Budget` returns the error.
    fn reject(self, error: EqTableError) -> <Self as EqAlloc>::Error
    where
        Self: EqAlloc;
    /// Admit one family before allocating any of its buffers.
    fn admit<F: Field>(
        self,
        elements: usize,
        at: &'static Location<'static>,
    ) -> Result<(), <Self as EqAlloc>::Error>
    where
        Self: EqAlloc;
    /// Allocate exactly `len` initialized elements after family admission.
    fn table<F: Field>(
        self,
        len: usize,
        init: TableInit<F>,
    ) -> Result<Vec<F>, <Self as EqAlloc>::Error>
    where
        Self: EqAlloc;
    /// Return empty storage with capacity at least `len`.
    fn storage<T>(self, len: usize) -> Result<Vec<T>, <Self as EqAlloc>::Error>
    where
        Self: EqAlloc;
}

pub(crate) enum TableInit<F> {
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

/// The allocator's own error, displayed and chained unchanged.
#[derive(Debug, Error)]
pub enum AllocationCause {
    #[error(transparent)]
    Reserve(#[from] TryReserveError),
    #[error(transparent)]
    Zero(ZeroAllocError),
}

#[derive(Debug, Error)]
pub enum EqTableError {
    #[error("eq table over {num_vars} variables has more entries than usize can count")]
    DimensionOverflow { num_vars: usize },
    #[error(
        "eq table of {elements} entries of {element_size} bytes each is too large to allocate"
    )]
    SizeOverflow {
        elements: usize,
        element_size: usize,
    },
    /// `elements` counts every buffer the budget admits together: all levels of
    /// a cached table, or one tensor factor.
    #[error(
        "eq table of {elements} entries needs {bytes} bytes, over the {max_bytes}-byte budget (requested at {at})"
    )]
    OverBudget {
        elements: usize,
        bytes: usize,
        max_bytes: usize,
        at: &'static Location<'static>,
    },
    /// The allocator's error is the [`source`](std::error::Error::source).
    #[error("allocating {elements} elements of eq-table storage failed")]
    AllocationFailed {
        elements: usize,
        #[source]
        cause: AllocationCause,
    },
    #[error("eq-table prefix length {got} exceeds the table's {expected} entries")]
    LengthMismatch { expected: usize, got: usize },
    #[error("index {index} is outside the eq table's {len} entries")]
    IndexOutOfRange { index: usize, len: usize },
    #[error(
        "tensor split of {outer_vars} outer variables exceeds the point's {num_vars} variables"
    )]
    InvalidSplit { outer_vars: usize, num_vars: usize },
}

impl EqAlloc for Global {
    type Error = Infallible;
}
impl Allocate for Global {
    #[expect(
        clippy::panic,
        reason = "Global preserves infallible builders' argument and capacity panics"
    )]
    #[track_caller]
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

impl EqAlloc for Budget {
    type Error = EqTableError;
}
impl Allocate for Budget {
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
                elements,
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

// The panic paths below call `reject` outside closures: `#[track_caller]` does
// not reach into a closure, so Global would report this file instead.
#[track_caller]
pub(crate) fn domain_len<A: EqAlloc>(num_vars: usize, policy: A) -> Result<usize, A::Error> {
    match u32::try_from(num_vars)
        .ok()
        .and_then(|bits| 1usize.checked_shl(bits))
    {
        Some(len) => Ok(len),
        None => Err(policy.reject(EqTableError::DimensionOverflow { num_vars })),
    }
}

/// Check individual buffers, never the combined size of a multi-buffer family.
#[track_caller]
pub(crate) fn check_layout<T, A: EqAlloc>(elements: usize, policy: A) -> Result<(), A::Error> {
    if Layout::array::<T>(elements).is_err() {
        return Err(policy.reject(EqTableError::SizeOverflow {
            elements,
            element_size: size_of::<T>().max(1),
        }));
    }
    Ok(())
}
