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

    pub(crate) fn census(
        &self,
        expected_log_t: usize,
        address_domain: usize,
        threadgroup_width: usize,
    ) -> Result<Option<RamAccessCensus>, RamAccessTapeError> {
        if threadgroup_width == 0 {
            return Err(RamAccessTapeError::ZeroThreadgroupWidth);
        }
        self.validate(expected_log_t, address_domain)?;
        let Some(records) = &self.records else {
            return Ok(None);
        };

        let mut nodes = records
            .iter()
            .map(|record| CensusNode {
                block: u64::from(record.cycle),
                address: record.address,
            })
            .collect::<Vec<_>>();
        let mut entries_per_level = Vec::with_capacity(self.log_t + 1);
        let mut groups_per_level = Vec::with_capacity(self.log_t + 1);
        let mut dispatches_per_level = Vec::with_capacity(self.log_t + 1);

        for level in 0..=self.log_t {
            entries_per_level.push(nodes.len());
            let (groups, dispatches) = group_counts(&nodes, threadgroup_width);
            groups_per_level.push(groups);
            dispatches_per_level.push(dispatches);
            if level != self.log_t {
                nodes = merge_parent_blocks(&nodes);
            }
        }

        Ok(Some(RamAccessCensus {
            entries: entries_per_level,
            groups: groups_per_level,
            dispatches: dispatches_per_level,
        }))
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

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RamAccessCensus {
    entries: Vec<usize>,
    groups: Vec<usize>,
    dispatches: Vec<usize>,
}

impl RamAccessCensus {
    pub(crate) fn entries_per_level(&self) -> &[usize] {
        &self.entries
    }

    pub(crate) fn groups_per_level(&self) -> &[usize] {
        &self.groups
    }

    pub(crate) fn dispatches_per_level(&self) -> &[usize] {
        &self.dispatches
    }
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for RamAccessCensus {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        for (name, values) in [
            ("entries_per_level", &self.entries),
            ("groups_per_level", &self.groups),
            ("dispatches_per_level", &self.dispatches),
        ] {
            visitor.visit_simple(
                allocative::Key::new(name),
                crate::backend::vec_heap_bytes(values),
            );
        }
        visitor.exit();
    }
}

#[derive(Clone, Copy)]
struct CensusNode {
    block: u64,
    address: u32,
}

fn group_counts(nodes: &[CensusNode], width: usize) -> (usize, usize) {
    let mut groups = 0;
    let mut dispatches = 0;
    let mut start = 0;
    while start < nodes.len() {
        let block = nodes[start].block;
        let mut end = start + 1;
        while end < nodes.len() && nodes[end].block == block {
            end += 1;
        }
        groups += 1;
        dispatches += (end - start).div_ceil(width);
        start = end;
    }
    (groups, dispatches)
}

fn merge_parent_blocks(nodes: &[CensusNode]) -> Vec<CensusNode> {
    let mut output = Vec::with_capacity(nodes.len());
    let mut start = 0;
    while start < nodes.len() {
        let parent = nodes[start].block >> 1;
        let first_child = nodes[start].block;
        let mut middle = start;
        while middle < nodes.len() && nodes[middle].block == first_child {
            middle += 1;
        }
        let mut end = middle;
        while end < nodes.len() && nodes[end].block >> 1 == parent {
            end += 1;
        }

        let mut left = start;
        let mut right = middle;
        while left < middle || right < end {
            let address = match (nodes.get(left).filter(|_| left < middle), nodes.get(right)) {
                (Some(left_node), Some(right_node)) if right < end => {
                    match left_node.address.cmp(&right_node.address) {
                        std::cmp::Ordering::Less => {
                            left += 1;
                            left_node.address
                        }
                        std::cmp::Ordering::Equal => {
                            left += 1;
                            right += 1;
                            left_node.address
                        }
                        std::cmp::Ordering::Greater => {
                            right += 1;
                            right_node.address
                        }
                    }
                }
                (Some(left_node), _) => {
                    left += 1;
                    left_node.address
                }
                (_, Some(right_node)) if right < end => {
                    right += 1;
                    right_node.address
                }
                _ => unreachable!("a parent block has at least one address"),
            };
            output.push(CensusNode {
                block: parent,
                address,
            });
        }
        start = end;
    }
    output
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
    #[error("RAM access census requires a nonzero threadgroup width")]
    ZeroThreadgroupWidth,
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "tests use fixed valid fixtures")]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use super::*;

    fn record(cycle: u32, address: u32) -> RamAccessRecord {
        RamAccessRecord {
            cycle,
            address,
            pre_value: u64::from(cycle),
            post_value: u64::from(cycle) + 1,
        }
    }

    fn independent_census(
        log_t: usize,
        records: &[RamAccessRecord],
        width: usize,
    ) -> RamAccessCensus {
        let mut entries = Vec::with_capacity(log_t + 1);
        let mut groups = Vec::with_capacity(log_t + 1);
        let mut dispatches = Vec::with_capacity(log_t + 1);
        for level in 0..=log_t {
            let mut blocks = BTreeMap::<u64, BTreeSet<u32>>::new();
            for record in records {
                let _ = blocks
                    .entry(u64::from(record.cycle) >> level)
                    .or_default()
                    .insert(record.address);
            }
            entries.push(blocks.values().map(BTreeSet::len).sum());
            groups.push(blocks.len());
            dispatches.push(
                blocks
                    .values()
                    .map(|addresses| addresses.len().div_ceil(width))
                    .sum(),
            );
        }
        RamAccessCensus {
            entries,
            groups,
            dispatches,
        }
    }

    #[test]
    fn merge_census_matches_independent_sets() {
        let fixtures = [
            vec![],
            vec![record(0, 0)],
            vec![record(0, 3), record(1, 3), record(7, 3)],
            vec![record(0, 1), record(1, 2), record(2, 1), record(3, 2)],
            vec![record(0, 0), record(1, 1), record(6, 2), record(7, 3)],
            vec![record(3, 0), record(4, 7)],
        ];
        for records in fixtures {
            let tape =
                RamAccessTape::new(3, records.len(), Some(records.clone()), true, true, true);
            for width in [1, 2, 3, 32] {
                assert_eq!(
                    tape.census(3, 8, width).unwrap().unwrap(),
                    independent_census(3, &records, width)
                );
            }
        }
    }

    #[test]
    fn validates_hard_geometry_and_dense_fallback() {
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
        assert_eq!(dense.census(26, 1 << 13, 256).unwrap(), None);
        assert_eq!(
            dense.census(26, 1 << 13, 0),
            Err(RamAccessTapeError::ZeroThreadgroupWidth)
        );

        let invalid_remap = RamAccessTape::new(3, 0, Some(vec![]), true, false, true);
        assert_eq!(
            invalid_remap.validate(3, 8),
            Err(RamAccessTapeError::UnremappableAccess)
        );
    }
}
