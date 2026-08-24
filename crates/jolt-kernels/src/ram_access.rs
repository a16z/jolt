use std::mem::size_of;

use thiserror::Error;

pub(crate) const MAX_RETAINED_RAM_ACCESSES: usize = 1 << 18;

#[repr(C, align(8))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct RamAccessRecord {
    pub cycle: u32,
    pub address: u32,
    pub pre_value: u64,
    pub post_value: u64,
}

const _: [(); 24] = [(); size_of::<RamAccessRecord>()];

#[derive(Debug)]
pub(crate) struct RamAccessTape {
    log_t: usize,
    access_count: usize,
    records: Option<Vec<RamAccessRecord>>,
    increment_compatible: bool,
    ram_ra_compatible: bool,
    hamming_exact: bool,
}

#[cfg_attr(
    not(all(feature = "metal", target_os = "macos")),
    expect(
        dead_code,
        reason = "the compatibility certificates are consumed by the Metal RAM owner"
    )
)]
impl RamAccessTape {
    pub(crate) fn new(
        log_t: usize,
        access_count: usize,
        records: Option<Vec<RamAccessRecord>>,
        increment_compatible: bool,
        ram_ra_compatible: bool,
        hamming_exact: bool,
    ) -> Self {
        Self {
            log_t,
            access_count,
            records,
            increment_compatible,
            ram_ra_compatible,
            hamming_exact,
        }
    }

    pub(crate) const fn access_count(&self) -> usize {
        self.access_count
    }

    pub(crate) fn records(&self) -> Option<&[RamAccessRecord]> {
        self.records.as_deref()
    }

    pub(crate) const fn increment_compatible(&self) -> bool {
        self.increment_compatible
    }

    pub(crate) const fn ram_ra_compatible(&self) -> bool {
        self.ram_ra_compatible
    }

    pub(crate) const fn hamming_exact(&self) -> bool {
        self.hamming_exact
    }

    pub(crate) fn validate(
        &self,
        expected_log_t: usize,
        address_domain: usize,
    ) -> Result<(), RamAccessTapeError> {
        let _ = (self.increment_compatible, self.hamming_exact);
        if self.log_t != expected_log_t || self.log_t > u32::BITS as usize {
            return Err(RamAccessTapeError::WrongCycleDomain);
        }
        if !self.ram_ra_compatible {
            return Err(RamAccessTapeError::UnremappableAccess);
        }
        let Some(records) = &self.records else {
            return Ok(());
        };
        if records.len() != self.access_count {
            return Err(RamAccessTapeError::IncompleteRecords);
        }
        let cycle_domain = 1u64 << self.log_t;
        let mut previous = None;
        for record in records {
            if u64::from(record.cycle) >= cycle_domain {
                return Err(RamAccessTapeError::CycleOutOfRange);
            }
            if record.address == u32::MAX {
                return Err(RamAccessTapeError::ReservedAddress);
            }
            if record.address as usize >= address_domain {
                return Err(RamAccessTapeError::AddressOutOfRange);
            }
            if previous.is_some_and(|cycle| cycle >= record.cycle) {
                return Err(RamAccessTapeError::RecordsOutOfOrder);
            }
            previous = Some(record.cycle);
        }
        Ok(())
    }
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for RamAccessTape {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        if let Some(records) = &self.records {
            visitor.visit_simple(
                allocative::Key::new("records"),
                crate::backend::vec_heap_bytes(records),
            );
        }
        visitor.exit();
    }
}

#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
pub(crate) enum RamAccessTapeError {
    #[error("RAM access tape has the wrong cycle domain")]
    WrongCycleDomain,
    #[error("RAM access tape retained only part of its records")]
    IncompleteRecords,
    #[error("RAM access tape contains a cycle outside its domain")]
    CycleOutOfRange,
    #[error("RAM access tape contains an address outside its domain")]
    AddressOutOfRange,
    #[error("RAM access tape uses the reserved no-access address")]
    ReservedAddress,
    #[error("RAM access tape records are not strictly cycle ordered")]
    RecordsOutOfOrder,
    #[error("RAM access tape contains a nonzero raw access that did not remap")]
    UnremappableAccess,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(cycle: u32, address: u32) -> RamAccessRecord {
        RamAccessRecord {
            cycle,
            address,
            pre_value: u64::from(cycle),
            post_value: u64::from(cycle) + 1,
        }
    }

    #[test]
    fn validates_hard_geometry_and_unretained_tape() {
        let max = RamAccessTape::new(
            32,
            1,
            Some(vec![record(u32::MAX, u32::MAX - 1)]),
            true,
            true,
            true,
        );
        assert!(max.validate(32, u32::MAX as usize).is_ok());

        let unordered = RamAccessTape::new(
            3,
            2,
            Some(vec![record(2, 0), record(1, 0)]),
            true,
            true,
            true,
        );
        assert_eq!(
            unordered.validate(3, 1),
            Err(RamAccessTapeError::RecordsOutOfOrder)
        );
        let duplicate = RamAccessTape::new(
            3,
            2,
            Some(vec![record(1, 0), record(1, 1)]),
            true,
            true,
            true,
        );
        assert_eq!(
            duplicate.validate(3, 2),
            Err(RamAccessTapeError::RecordsOutOfOrder)
        );
        let bad_cycle = RamAccessTape::new(2, 1, Some(vec![record(4, 0)]), true, true, true);
        assert_eq!(
            bad_cycle.validate(2, 1),
            Err(RamAccessTapeError::CycleOutOfRange)
        );
        let bad_address = RamAccessTape::new(2, 1, Some(vec![record(0, 2)]), true, true, true);
        assert_eq!(
            bad_address.validate(2, 2),
            Err(RamAccessTapeError::AddressOutOfRange)
        );
        let sentinel = RamAccessTape::new(2, 1, Some(vec![record(0, u32::MAX)]), true, true, true);
        assert_eq!(
            sentinel.validate(2, 1usize << 32),
            Err(RamAccessTapeError::ReservedAddress)
        );
        let dense = RamAccessTape::new(26, 1 << 20, None, false, true, true);
        assert!(dense.validate(26, 1 << 13).is_ok());
        assert!(dense.records().is_none());

        let invalid_remap = RamAccessTape::new(3, 0, Some(vec![]), true, false, true);
        assert_eq!(
            invalid_remap.validate(3, 8),
            Err(RamAccessTapeError::UnremappableAccess)
        );
    }
}
