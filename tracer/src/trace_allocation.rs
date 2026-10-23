//! Request checks for an explicit trace allocation limit
//! ([`TracerBackend::limited_for_row_domain`](crate::TracerBackend::limited_for_row_domain)).

use jolt_program::execution::{TraceAllocationError, TraceAllocationPhase};

/// What one trace-storage owner may request: the limit, less the bytes every
/// other selected buffer holds at its observed capacity.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TraceAllocationAllowance {
    limit_bytes: usize,
    other_live_bytes: usize,
}

impl TraceAllocationAllowance {
    pub(crate) fn new(limit_bytes: usize, other_live_bytes: usize) -> Self {
        Self {
            limit_bytes,
            other_live_bytes,
        }
    }

    /// Admits `new_request` bytes beside the `retained_here` bytes this owner
    /// keeps live until the request completes, such as the old buffer of a
    /// growth.
    pub(crate) fn check_request(
        self,
        phase: TraceAllocationPhase,
        retained_here: usize,
        new_request: usize,
    ) -> Result<(), TraceAllocationError> {
        let required_bytes = self
            .other_live_bytes
            .checked_add(retained_here)
            .and_then(|bytes| bytes.checked_add(new_request))
            .ok_or(TraceAllocationError::SizeOverflow { phase })?;
        if required_bytes > self.limit_bytes {
            return Err(TraceAllocationError::LimitExceeded {
                phase,
                limit_bytes: self.limit_bytes,
                required_bytes,
            });
        }
        Ok(())
    }
}

pub(crate) fn bytes_of<T>(
    count: usize,
    phase: TraceAllocationPhase,
) -> Result<usize, TraceAllocationError> {
    count
        .checked_mul(size_of::<T>())
        .ok_or(TraceAllocationError::SizeOverflow { phase })
}

pub(crate) fn sum_bytes(
    parts: impl IntoIterator<Item = usize>,
    phase: TraceAllocationPhase,
) -> Result<usize, TraceAllocationError> {
    parts
        .into_iter()
        .try_fold(0usize, usize::checked_add)
        .ok_or(TraceAllocationError::SizeOverflow { phase })
}

/// Capacity a limited buffer grows to when it must hold `required` elements:
/// the larger of `required` and twice `capacity`, so growth stays amortized.
pub(crate) fn growth_target(capacity: usize, required: usize) -> usize {
    required.max(capacity.saturating_mul(2))
}

/// `Vec::try_reserve_exact` reporting a refused request as `phase`'s
/// [`TraceAllocationError::AllocationFailed`].
pub(crate) fn reserve_exact<T>(
    buffer: &mut Vec<T>,
    additional: usize,
    phase: TraceAllocationPhase,
) -> Result<(), TraceAllocationError> {
    buffer
        .try_reserve_exact(additional)
        .map_err(|_| TraceAllocationError::AllocationFailed { phase })
}
