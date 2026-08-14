use std::mem::size_of;
use std::sync::atomic::{AtomicU64, Ordering};

use thiserror::Error;

use super::solinas::spartan_shift::{SpartanShiftGeometry, SpartanShiftResidentRows};

static NEXT_PRODUCER_GENERATION: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct AllocationReceipt {
    identity: usize,
    bytes: usize,
}

impl AllocationReceipt {
    fn new(identity: usize, bytes: usize) -> Result<Self, SpartanDenseOwnerError> {
        if identity == 0 {
            return Err(SpartanDenseOwnerError::MissingAllocationIdentity);
        }
        if bytes == 0 {
            return Err(SpartanDenseOwnerError::EmptyAllocation);
        }
        Ok(Self { identity, bytes })
    }

    fn validate(self, identity: usize, bytes: usize) -> Result<(), SpartanDenseOwnerError> {
        if self.identity != identity {
            return Err(SpartanDenseOwnerError::AllocationIdentityMismatch {
                expected: self.identity,
                got: identity,
            });
        }
        if self.bytes != bytes {
            return Err(SpartanDenseOwnerError::AllocationLengthMismatch {
                expected: self.bytes,
                got: bytes,
            });
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ShiftReceipt {
    generation: u64,
    rows: usize,
    device_registry_id: u64,
    allocations: [AllocationReceipt; 3],
    exact_current_flags: bool,
    row_extractions: usize,
    late_copy_dispatches: usize,
}

impl ShiftReceipt {
    fn co_produced(
        generation: u64,
        rows: &SpartanShiftResidentRows,
    ) -> Result<Self, SpartanDenseOwnerError> {
        let geometry = SpartanShiftGeometry::new(rows.len())
            .map_err(|_| SpartanDenseOwnerError::InvalidRows(rows.len()))?;
        let value_bytes = geometry
            .rows()
            .checked_mul(size_of::<u64>())
            .ok_or(SpartanDenseOwnerError::SizeOverflow)?;
        let flag_bytes = geometry
            .flag_words()
            .checked_mul(size_of::<super::solinas::spartan_shift::SpartanShiftFlagWord>())
            .ok_or(SpartanDenseOwnerError::SizeOverflow)?;
        let identities = rows.allocation_identities();
        let allocations = [
            AllocationReceipt::new(identities[0], value_bytes)?,
            AllocationReceipt::new(identities[1], value_bytes)?,
            AllocationReceipt::new(identities[2], flag_bytes)?,
        ];
        validate_unique_allocations(&allocations)?;
        Ok(Self {
            generation,
            rows: geometry.rows(),
            device_registry_id: rows.device_registry_id(),
            allocations,
            exact_current_flags: true,
            row_extractions: geometry.rows(),
            late_copy_dispatches: 0,
        })
    }

    fn validate(
        self,
        owner_generation: u64,
        rows: &SpartanShiftResidentRows,
        expected_rows: usize,
        expected_device_registry_id: u64,
    ) -> Result<(), SpartanDenseOwnerError> {
        if self.generation == 0 || self.generation != owner_generation {
            return Err(SpartanDenseOwnerError::GenerationMismatch {
                expected: owner_generation,
                got: self.generation,
            });
        }
        if self.rows != expected_rows || rows.len() != expected_rows {
            return Err(SpartanDenseOwnerError::RowCountMismatch {
                expected: expected_rows,
                got: rows.len(),
            });
        }
        if self.device_registry_id != expected_device_registry_id
            || rows.device_registry_id() != expected_device_registry_id
        {
            return Err(SpartanDenseOwnerError::DeviceMismatch {
                expected: expected_device_registry_id,
                got: rows.device_registry_id(),
            });
        }
        if !self.exact_current_flags {
            return Err(SpartanDenseOwnerError::UncertifiedCurrentNoop);
        }
        let geometry = SpartanShiftGeometry::new(expected_rows)
            .map_err(|_| SpartanDenseOwnerError::InvalidRows(expected_rows))?;
        let value_bytes = geometry
            .rows()
            .checked_mul(size_of::<u64>())
            .ok_or(SpartanDenseOwnerError::SizeOverflow)?;
        let flag_bytes = geometry
            .flag_words()
            .checked_mul(size_of::<super::solinas::spartan_shift::SpartanShiftFlagWord>())
            .ok_or(SpartanDenseOwnerError::SizeOverflow)?;
        for ((receipt, identity), bytes) in self
            .allocations
            .into_iter()
            .zip(rows.allocation_identities())
            .zip([value_bytes, value_bytes, flag_bytes])
        {
            receipt.validate(identity, bytes)?;
        }
        Ok(())
    }
}

pub(super) struct SpartanDenseShiftLease {
    rows: SpartanShiftResidentRows,
    receipt: ShiftReceipt,
    owner_generation: u64,
}

impl SpartanDenseShiftLease {
    pub(super) fn into_rows(
        self,
        expected_rows: usize,
        expected_device_registry_id: u64,
    ) -> Result<SpartanShiftResidentRows, SpartanDenseOwnerError> {
        self.receipt.validate(
            self.owner_generation,
            &self.rows,
            expected_rows,
            expected_device_registry_id,
        )?;
        Ok(self.rows)
    }
}

pub(super) struct SpartanDenseResidentOwner {
    generation: u64,
    shift_receipt: ShiftReceipt,
    shift_rows: Option<SpartanShiftResidentRows>,
}

impl SpartanDenseResidentOwner {
    pub(super) fn from_co_produced_shift(
        rows: SpartanShiftResidentRows,
    ) -> Result<Self, SpartanDenseOwnerError> {
        let generation = next_generation()?;
        let shift_receipt = ShiftReceipt::co_produced(generation, &rows)?;
        Ok(Self {
            generation,
            shift_receipt,
            shift_rows: Some(rows),
        })
    }

    pub(super) const fn generation(&self) -> u64 {
        self.generation
    }

    pub(super) const fn shift_row_extractions(&self) -> usize {
        self.shift_receipt.row_extractions
    }

    pub(super) const fn shift_late_copy_dispatches(&self) -> usize {
        self.shift_receipt.late_copy_dispatches
    }

    pub(super) fn shift_rows(&self) -> Option<&SpartanShiftResidentRows> {
        self.shift_rows.as_ref()
    }

    pub(super) fn take_shift_lease(&mut self) -> Option<SpartanDenseShiftLease> {
        self.shift_rows.take().map(|rows| SpartanDenseShiftLease {
            rows,
            receipt: self.shift_receipt,
            owner_generation: self.generation,
        })
    }
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for SpartanDenseResidentOwner {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        if let Some(rows) = &self.shift_rows {
            visitor.visit_simple(
                allocative::Key::new("device_shift_rows"),
                rows.resident_bytes(),
            );
        }
        visitor.exit();
    }
}

fn next_generation() -> Result<u64, SpartanDenseOwnerError> {
    NEXT_PRODUCER_GENERATION
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |generation| {
            generation.checked_add(1)
        })
        .map_err(|_| SpartanDenseOwnerError::GenerationExhausted)
}

fn validate_unique_allocations(
    allocations: &[AllocationReceipt],
) -> Result<(), SpartanDenseOwnerError> {
    for (index, allocation) in allocations.iter().enumerate() {
        if allocations[..index]
            .iter()
            .any(|previous| previous.identity == allocation.identity)
        {
            return Err(SpartanDenseOwnerError::DuplicateAllocationIdentity(
                allocation.identity,
            ));
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub(super) enum SpartanDenseOwnerError {
    #[error("Spartan dense producer generation counter is exhausted")]
    GenerationExhausted,
    #[error("Spartan dense receipt generation is {got}, expected {expected}")]
    GenerationMismatch { expected: u64, got: u64 },
    #[error("Spartan dense producer has invalid row count {0}")]
    InvalidRows(usize),
    #[error("Spartan dense row count is {got}, expected {expected}")]
    RowCountMismatch { expected: usize, got: usize },
    #[error("Spartan dense allocation has no identity")]
    MissingAllocationIdentity,
    #[error("Spartan dense allocation is empty")]
    EmptyAllocation,
    #[error("Spartan dense allocation identity is {got}, expected {expected}")]
    AllocationIdentityMismatch { expected: usize, got: usize },
    #[error("Spartan dense allocation has {got} bytes, expected {expected}")]
    AllocationLengthMismatch { expected: usize, got: usize },
    #[error("Spartan dense allocation identity {0} is used more than once")]
    DuplicateAllocationIdentity(usize),
    #[error("Spartan dense rows belong to Metal device {got}, expected {expected}")]
    DeviceMismatch { expected: u64, got: u64 },
    #[error("Spartan dense shift receipt does not certify current noop")]
    UncertifiedCurrentNoop,
    #[error("Spartan dense size arithmetic overflowed")]
    SizeOverflow,
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use super::*;
    use crate::metal::solinas::spartan_shift::SPARTAN_SHIFT_FLAG_ROWS_PER_WORD;

    #[test]
    fn allocation_receipt_checks_identity_and_length() {
        let receipt = AllocationReceipt::new(17, 64).unwrap();
        assert_eq!(receipt.validate(17, 64), Ok(()));
        assert!(matches!(
            receipt.validate(18, 64),
            Err(SpartanDenseOwnerError::AllocationIdentityMismatch { .. })
        ));
        assert!(matches!(
            receipt.validate(17, 32),
            Err(SpartanDenseOwnerError::AllocationLengthMismatch { .. })
        ));
    }

    #[test]
    fn producer_contract_uses_three_packed_flag_planes() {
        let rows: usize = 1 << 26;
        let flag_words = rows.div_ceil(SPARTAN_SHIFT_FLAG_ROWS_PER_WORD);
        assert_eq!(flag_words, 1 << 21);
        assert_eq!(flag_words * 12, 25_165_824);
    }
}
