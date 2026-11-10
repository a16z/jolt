use super::build::SERIAL_MAX_VARS;
use super::EqPolynomial;
use crate::PointLike;
use jolt_field::Field;
use jolt_utils::alloc::ZeroAllocError;
use std::alloc::Layout;
use std::ops::Mul;
use std::panic::Location;
use thiserror::Error;

/// Caps field-element storage across all buffers returned by one builder.
/// Cached-table headers are validated for capacity but excluded from the cap.
///
/// Fallible builders reserve every owned table buffer and point copy before
/// filling. AllocationFailed covers this storage. Rayon's pool and scheduler
/// bookkeeping allocations are outside the guarantee, as are callback allocations.
#[derive(Clone, Copy, Debug)]
pub struct EqBudget {
    pub max_bytes: usize,
}

#[derive(Debug, Error)]
pub enum EqTableError {
    #[error("equality table dimension {num_vars} overflows capacity")]
    DimensionOverflow { num_vars: usize },
    #[error("equality table needs {bytes} bytes, exceeding {max_bytes} at {at}")]
    OverBudget {
        bytes: usize,
        max_bytes: usize,
        at: &'static Location<'static>,
    },
    #[error("allocation failed for {elements} elements")]
    AllocationFailed { elements: usize },
    #[error("expected length at most {expected}, got {got}")]
    LengthMismatch { expected: usize, got: usize },
    #[error("index {index} is outside table of length {len}")]
    IndexOutOfRange { index: usize, len: usize },
    #[error("no equality-polynomial rounds remain")]
    RoundsExhausted,
}

#[inline]
pub(crate) fn domain_len(num_vars: usize) -> Result<usize, EqTableError> {
    u32::try_from(num_vars)
        .ok()
        .and_then(|bits| 1usize.checked_shl(bits))
        .ok_or(EqTableError::DimensionOverflow { num_vars })
}

#[inline]
pub(crate) fn cached_elements(num_vars: usize) -> Result<usize, EqTableError> {
    let len = domain_len(num_vars)?;
    len.checked_add(len - 1)
        .ok_or(EqTableError::DimensionOverflow { num_vars })
}

impl EqBudget {
    #[track_caller]
    #[inline]
    pub(crate) fn check<F>(self, elements: usize, num_vars: usize) -> Result<(), EqTableError> {
        let bytes = Layout::array::<F>(elements)
            .map_err(|_| EqTableError::DimensionOverflow { num_vars })?
            .size();
        if bytes > self.max_bytes {
            return Err(EqTableError::OverBudget {
                bytes,
                max_bytes: self.max_bytes,
                at: Location::caller(),
            });
        }
        Ok(())
    }
}

#[inline]
pub(crate) fn try_storage<T>(elements: usize, num_vars: usize) -> Result<Vec<T>, EqTableError> {
    let _layout =
        Layout::array::<T>(elements).map_err(|_| EqTableError::DimensionOverflow { num_vars })?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(elements)
        .map_err(|_| EqTableError::AllocationFailed { elements })?;
    Ok(values)
}

#[inline]
pub(crate) fn try_zeros<F: Field>(
    elements: usize,
    num_vars: usize,
) -> Result<Vec<F>, EqTableError> {
    F::try_zeroed_vec(elements).map_err(|error| match error {
        ZeroAllocError::CapacityOverflow => EqTableError::DimensionOverflow { num_vars },
        ZeroAllocError::AllocFailed { .. } => EqTableError::AllocationFailed { elements },
    })
}

impl<F: Field> EqPolynomial<F> {
    /// Fallible [`Self::evals`], capped before allocating any table storage.
    /// Reserves table storage fallibly before using the same fill and parallel
    /// dispatch as [`Self::evals`]. Rayon's runtime allocations are excluded;
    /// see [`EqBudget`].
    #[track_caller]
    pub fn try_evals<P>(r: &P, scale: Option<F>, budget: EqBudget) -> Result<Vec<F>, EqTableError>
    where
        P: PointLike + ?Sized,
        P::Coord: Send + Sync + Into<F>,
        F: Mul<P::Coord, Output = F>,
    {
        let len = domain_len(r.num_vars())?;
        budget.check::<F>(len, r.num_vars())?;
        let mut values = Self::try_evals_buffer(r.num_vars(), scale)?;
        Self::fill_evals(r, &mut values);
        Ok(values)
    }

    /// Fallible [`Self::evals_cached_msb`]; the cap covers `2^(n+1)-1` elements.
    #[track_caller]
    pub fn try_evals_cached_msb<P>(
        r: &P,
        scale: Option<F>,
        budget: EqBudget,
    ) -> Result<Vec<Vec<F>>, EqTableError>
    where
        P: PointLike + ?Sized,
        P::Coord: Send + Sync + Into<F>,
        F: Mul<P::Coord, Output = F>,
    {
        Self::try_cached::<_, true>(r, scale, budget)
    }

    /// Fallible [`Self::evals_cached_lsb`]; the cap covers `2^(n+1)-1` elements.
    #[track_caller]
    pub fn try_evals_cached_lsb<P>(
        r: &P,
        scale: Option<F>,
        budget: EqBudget,
    ) -> Result<Vec<Vec<F>>, EqTableError>
    where
        P: PointLike + ?Sized,
        P::Coord: Send + Sync + Into<F>,
        F: Mul<P::Coord, Output = F>,
    {
        Self::try_cached::<_, false>(r, scale, budget)
    }

    #[track_caller]
    fn try_cached<P, const MSB: bool>(
        r: &P,
        scale: Option<F>,
        budget: EqBudget,
    ) -> Result<Vec<Vec<F>>, EqTableError>
    where
        P: PointLike + ?Sized,
        P::Coord: Send + Sync + Into<F>,
        F: Mul<P::Coord, Output = F>,
    {
        let n = r.num_vars();
        budget.check::<F>(cached_elements(n)?, n)?;
        let mut tables = Self::try_cached_buffers(n, scale)?;
        Self::fill_reserved_cached::<_, MSB>(r, &mut tables);
        Ok(tables)
    }

    /// Fallible [`Self::evals_mapped`], with all table storage reserved before
    /// serial, slab or parallel filling. Rayon's runtime allocations are excluded;
    /// see [`EqBudget`].
    /// The callback's own allocations are its responsibility.
    #[track_caller]
    pub fn try_evals_mapped<P, G>(
        r: &P,
        scale: Option<F>,
        map: G,
        budget: EqBudget,
    ) -> Result<Vec<F>, EqTableError>
    where
        P: PointLike + ?Sized,
        P::Coord: Send + Sync + Into<F>,
        F: Mul<P::Coord, Output = F>,
        G: Fn(F) -> F + Sync,
    {
        let len = domain_len(r.num_vars())?;
        budget.check::<F>(len, r.num_vars())?;
        let mut values = try_zeros::<F>(len, r.num_vars())?;
        values[0] = scale.unwrap_or(F::one());
        Ok(Self::fill_mapped(r, values, &map))
    }

    #[inline]
    pub(crate) fn try_evals_buffer(
        num_vars: usize,
        scale: Option<F>,
    ) -> Result<Vec<F>, EqTableError> {
        let len = domain_len(num_vars)?;
        let scale = scale.unwrap_or(F::one());
        if num_vars <= SERIAL_MAX_VARS {
            let mut values = try_storage(len, num_vars)?;
            values.push(scale);
            Ok(values)
        } else {
            let mut values = try_zeros::<F>(len, num_vars)?;
            values[0] = scale;
            Ok(values)
        }
    }

    #[inline]
    pub(crate) fn try_cached_buffers(
        num_vars: usize,
        scale: Option<F>,
    ) -> Result<Vec<Vec<F>>, EqTableError> {
        let mut tables = try_storage(num_vars + 1, num_vars)?;
        for j in 0..=num_vars {
            tables.push(try_storage(1 << j, num_vars)?);
        }
        tables[0].push(scale.unwrap_or(F::one()));
        Ok(tables)
    }

    /// Validates the domain and prefix length without allocating.
    pub fn try_prefix_sum<P: PointLike<Coord = F> + ?Sized>(
        r: &P,
        len: usize,
    ) -> Result<F, EqTableError> {
        Self::check_prefix(r, len)?;
        Ok(Self::prefix_sum(r, len))
    }

    pub(super) fn check_prefix<P: PointLike + ?Sized>(
        r: &P,
        len: usize,
    ) -> Result<(), EqTableError> {
        let expected = domain_len(r.num_vars())?;
        if len > expected {
            return Err(EqTableError::LengthMismatch { expected, got: len });
        }
        Ok(())
    }
}
