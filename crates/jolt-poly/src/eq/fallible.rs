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

pub(crate) fn domain_len(num_vars: usize) -> Result<usize, EqTableError> {
    u32::try_from(num_vars)
        .ok()
        .and_then(|bits| 1usize.checked_shl(bits))
        .ok_or(EqTableError::DimensionOverflow { num_vars })
}

pub(crate) fn cached_elements(num_vars: usize) -> Result<usize, EqTableError> {
    let len = domain_len(num_vars)?;
    len.checked_add(len - 1)
        .ok_or(EqTableError::DimensionOverflow { num_vars })
}

impl EqBudget {
    #[track_caller]
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

pub(crate) fn try_storage<T>(elements: usize, num_vars: usize) -> Result<Vec<T>, EqTableError> {
    let _layout =
        Layout::array::<T>(elements).map_err(|_| EqTableError::DimensionOverflow { num_vars })?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(elements)
        .map_err(|_| EqTableError::AllocationFailed { elements })?;
    Ok(values)
}

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
    /// Fills serially to avoid Rayon's infallible runtime allocation paths.
    #[track_caller]
    pub fn try_evals<P>(r: &P, scale: Option<F>, budget: EqBudget) -> Result<Vec<F>, EqTableError>
    where
        P: PointLike + ?Sized,
        P::Coord: Send + Sync + Into<F>,
        F: Mul<P::Coord, Output = F>,
    {
        let len = domain_len(r.num_vars())?;
        budget.check::<F>(len, r.num_vars())?;
        let mut values = try_zeros::<F>(len, r.num_vars())?;
        values[0] = scale.unwrap_or(F::one());
        if r.num_vars() <= 16 {
            Self::fill_serial(r, &mut values, &|x| x);
        } else {
            Self::fill_parallel_coords::<_, false>(r.coords(), P::ENDIANNESS, &mut values);
        }
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
        Self::try_cached(r, scale, budget, true)
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
        Self::try_cached(r, scale, budget, false)
    }

    #[track_caller]
    fn try_cached<P>(
        r: &P,
        scale: Option<F>,
        budget: EqBudget,
        msb: bool,
    ) -> Result<Vec<Vec<F>>, EqTableError>
    where
        P: PointLike + ?Sized,
        P::Coord: Send + Sync + Into<F>,
        F: Mul<P::Coord, Output = F>,
    {
        let n = r.num_vars();
        budget.check::<F>(cached_elements(n)?, n)?;
        let mut tables = try_storage(n + 1, n)?;
        for j in 0..=n {
            tables.push(try_zeros::<F>(1 << j, n)?);
        }
        tables[0][0] = scale.unwrap_or(F::one());
        Self::fill_cached(r, &mut tables, msb);
        Ok(tables)
    }

    /// Fallible [`Self::evals_mapped`], filled serially as in [`Self::try_evals`].
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
        Self::fill_serial(r, &mut values, &map);
        Ok(values)
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
