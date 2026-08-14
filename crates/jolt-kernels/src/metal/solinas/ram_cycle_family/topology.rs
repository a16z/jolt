use std::mem::size_of;

use thiserror::Error;

use super::owner::RamAccessRecord;

const NONE: u32 = u32::MAX;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LevelCensus {
    entries: u64,
    groups: u64,
    tiles: u64,
}

impl LevelCensus {
    pub const fn entries(self) -> u64 {
        self.entries
    }

    pub const fn groups(self) -> u64 {
        self.groups
    }

    pub const fn tiles(self) -> u64 {
        self.tiles
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RamRwMergeEvent {
    low_state: u32,
    high_state: u32,
    group_index: u32,
    parent_index: u32,
    low_absent_value: u64,
    high_absent_value: u64,
}

const _: [(); 32] = [(); size_of::<RamRwMergeEvent>()];

impl RamRwMergeEvent {
    pub fn low_state(self) -> Option<usize> {
        index(self.low_state)
    }

    pub fn high_state(self) -> Option<usize> {
        index(self.high_state)
    }

    pub const fn group_index(self) -> u32 {
        self.group_index
    }

    pub const fn parent_index(self) -> u32 {
        self.parent_index
    }

    pub const fn low_absent_value(self) -> u64 {
        self.low_absent_value
    }

    pub const fn high_absent_value(self) -> u64 {
        self.high_absent_value
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RamRwGroupEvent {
    low_group: u32,
    high_group: u32,
    parent_block: u32,
    parent_group: u32,
}

const _: [(); 16] = [(); size_of::<RamRwGroupEvent>()];

impl RamRwGroupEvent {
    pub fn low_group(self) -> Option<usize> {
        index(self.low_group)
    }

    pub fn high_group(self) -> Option<usize> {
        index(self.high_group)
    }

    pub const fn parent_block(self) -> u32 {
        self.parent_block
    }

    pub const fn parent_group(self) -> u32 {
        self.parent_group
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BlockMerge {
    low_state: u32,
    high_state: u32,
}

const _: [(); 8] = [(); size_of::<BlockMerge>()];

impl BlockMerge {
    pub fn low_state(self) -> Option<usize> {
        index(self.low_state)
    }

    pub fn high_state(self) -> Option<usize> {
        index(self.high_state)
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct LevelRange {
    start: u32,
    len: u32,
}

const _: [(); 8] = [(); size_of::<LevelRange>()];

impl LevelRange {
    fn slice<T>(self, values: &[T]) -> Result<&[T], TopologyError> {
        let start = self.start as usize;
        let end = start
            .checked_add(self.len as usize)
            .ok_or(TopologyError::Overflow)?;
        values
            .get(start..end)
            .ok_or(TopologyError::InvalidLevelRange)
    }
}

#[derive(Debug)]
pub struct RamRwMergeTopology {
    log_t: usize,
    events: Box<[RamRwMergeEvent]>,
    group_events: Box<[RamRwGroupEvent]>,
    event_levels: Box<[LevelRange]>,
    group_levels: Box<[LevelRange]>,
    census: Box<[LevelCensus]>,
    final_addresses: Box<[u32]>,
}

impl RamRwMergeTopology {
    pub const fn log_t(&self) -> usize {
        self.log_t
    }

    pub fn census(&self) -> &[LevelCensus] {
        &self.census
    }

    pub fn events_for_round(&self, round: usize) -> Result<&[RamRwMergeEvent], TopologyError> {
        self.event_levels
            .get(round)
            .ok_or(TopologyError::RoundOutOfRange { round })?
            .slice(&self.events)
    }

    pub fn final_addresses(&self) -> &[u32] {
        &self.final_addresses
    }

    pub fn group_events_for_round(
        &self,
        round: usize,
    ) -> Result<&[RamRwGroupEvent], TopologyError> {
        self.group_levels
            .get(round)
            .ok_or(TopologyError::RoundOutOfRange { round })?
            .slice(&self.group_events)
    }

    pub(crate) fn owned_heap_bytes(&self) -> usize {
        std::mem::size_of_val(self.events.as_ref())
            + std::mem::size_of_val(self.group_events.as_ref())
            + std::mem::size_of_val(self.event_levels.as_ref())
            + std::mem::size_of_val(self.group_levels.as_ref())
            + std::mem::size_of_val(self.census.as_ref())
            + std::mem::size_of_val(self.final_addresses.as_ref())
    }

    pub(crate) fn build(
        log_t: usize,
        records: &[RamAccessRecord],
        threadgroup_width: usize,
    ) -> Result<Self, TopologyError> {
        if threadgroup_width == 0 {
            return Err(TopologyError::ZeroThreadgroupWidth);
        }

        let mut current = records
            .iter()
            .enumerate()
            .map(|(index, record)| {
                Ok(RwBuildNode {
                    block: u64::from(record.cycle()),
                    address: record.address(),
                    previous_value: record.pre_value(),
                    next_value: record.post_value(),
                    state_index: checked_u32(index)?,
                    group_index: checked_u32(index)?,
                })
            })
            .collect::<Result<Vec<_>, TopologyError>>()?;

        let mut events = Vec::new();
        let mut group_events = Vec::new();
        let mut event_levels = Vec::with_capacity(log_t);
        let mut group_levels = Vec::with_capacity(log_t);
        let mut census = Vec::with_capacity(log_t.checked_add(1).ok_or(TopologyError::Overflow)?);
        census.push(LevelCensus {
            entries: checked_u64(current.len())?,
            groups: checked_u64(current.len())?,
            tiles: checked_u64(current.len())?,
        });

        for _ in 0..log_t {
            let event_start = checked_u32(events.len())?;
            let group_start = checked_u32(group_events.len())?;
            let mut next = Vec::with_capacity(current.len());
            let mut start = 0;
            let mut level_groups = 0usize;
            let mut level_tiles = 0usize;

            while start < current.len() {
                let first_node = current
                    .get(start)
                    .copied()
                    .ok_or(TopologyError::InvalidBuildState)?;
                let parent_block = first_node.block >> 1;
                let mut end = start.checked_add(1).ok_or(TopologyError::Overflow)?;
                while current
                    .get(end)
                    .is_some_and(|node| node.block >> 1 == parent_block)
                {
                    end += 1;
                }

                let first_block = first_node.block;
                let mut middle = start;
                while middle < end
                    && current
                        .get(middle)
                        .is_some_and(|node| node.block == first_block)
                {
                    middle += 1;
                }
                let first_is_low = first_block & 1 == 0;
                let (low, high) = if first_is_low {
                    (start..middle, middle..end)
                } else {
                    (start..start, start..end)
                };
                let low_group = low
                    .clone()
                    .next()
                    .and_then(|index| current.get(index))
                    .map_or(NONE, |node| node.group_index);
                let high_group = high
                    .clone()
                    .next()
                    .and_then(|index| current.get(index))
                    .map_or(NONE, |node| node.group_index);
                let parent_group = checked_u32(level_groups)?;
                group_events.push(RamRwGroupEvent {
                    low_group,
                    high_group,
                    parent_block: checked_u32_u64(parent_block)?,
                    parent_group,
                });

                let before = next.len();
                merge_address_group(
                    &current,
                    low,
                    high,
                    parent_block,
                    parent_group,
                    &mut events,
                    &mut next,
                )?;
                let group_entries = next
                    .len()
                    .checked_sub(before)
                    .ok_or(TopologyError::Overflow)?;
                level_tiles = level_tiles
                    .checked_add(group_entries.div_ceil(threadgroup_width))
                    .ok_or(TopologyError::Overflow)?;
                level_groups = level_groups.checked_add(1).ok_or(TopologyError::Overflow)?;
                start = end;
            }

            event_levels.push(LevelRange {
                start: event_start,
                len: checked_u32(events.len())?
                    .checked_sub(event_start)
                    .ok_or(TopologyError::Overflow)?,
            });
            group_levels.push(LevelRange {
                start: group_start,
                len: checked_u32(group_events.len())?
                    .checked_sub(group_start)
                    .ok_or(TopologyError::Overflow)?,
            });
            census.push(LevelCensus {
                entries: checked_u64(next.len())?,
                groups: checked_u64(level_groups)?,
                tiles: checked_u64(level_tiles)?,
            });
            current = next;
        }

        let final_addresses = current
            .iter()
            .map(|node| node.address)
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Ok(Self {
            log_t,
            events: events.into_boxed_slice(),
            group_events: group_events.into_boxed_slice(),
            event_levels: event_levels.into_boxed_slice(),
            group_levels: group_levels.into_boxed_slice(),
            census: census.into_boxed_slice(),
            final_addresses,
        })
    }
}

#[derive(Debug)]
pub struct RamBlockTopology {
    log_t: usize,
    leaf_cycles: Box<[u32]>,
    merges: Box<[BlockMerge]>,
    levels: Box<[LevelRange]>,
    census: Box<[LevelCensus]>,
}

impl RamBlockTopology {
    pub const fn log_t(&self) -> usize {
        self.log_t
    }

    pub fn leaf_cycles(&self) -> &[u32] {
        &self.leaf_cycles
    }

    pub fn census(&self) -> &[LevelCensus] {
        &self.census
    }

    pub fn merges_for_round(&self, round: usize) -> Result<&[BlockMerge], TopologyError> {
        self.levels
            .get(round)
            .ok_or(TopologyError::RoundOutOfRange { round })?
            .slice(&self.merges)
    }

    pub(crate) fn owned_heap_bytes(&self) -> usize {
        std::mem::size_of_val(self.leaf_cycles.as_ref())
            + std::mem::size_of_val(self.merges.as_ref())
            + std::mem::size_of_val(self.levels.as_ref())
            + std::mem::size_of_val(self.census.as_ref())
    }

    pub(crate) fn build(
        log_t: usize,
        records: &[RamAccessRecord],
        increment_cycles: &[u64],
        threadgroup_width: usize,
    ) -> Result<Self, TopologyError> {
        if threadgroup_width == 0 {
            return Err(TopologyError::ZeroThreadgroupWidth);
        }
        let leaf_cycles = union_cycles(records, increment_cycles)?;
        let mut current = leaf_cycles
            .iter()
            .enumerate()
            .map(|(index, cycle)| {
                Ok(BlockBuildNode {
                    block: u64::from(*cycle),
                    state_index: checked_u32(index)?,
                })
            })
            .collect::<Result<Vec<_>, TopologyError>>()?;
        let mut merges = Vec::new();
        let mut levels = Vec::with_capacity(log_t);
        let mut census = Vec::with_capacity(log_t.checked_add(1).ok_or(TopologyError::Overflow)?);
        census.push(block_census(current.len(), threadgroup_width)?);

        for _ in 0..log_t {
            let level_start = checked_u32(merges.len())?;
            let mut next = Vec::with_capacity(current.len());
            let mut index = 0;
            while index < current.len() {
                let first = current
                    .get(index)
                    .copied()
                    .ok_or(TopologyError::InvalidBuildState)?;
                let parent = first.block >> 1;
                let first_is_low = first.block & 1 == 0;
                let next_index = index.checked_add(1).ok_or(TopologyError::Overflow)?;
                let high_child = current
                    .get(next_index)
                    .copied()
                    .filter(|next| next.block >> 1 == parent);
                let (low_state, high_state, consumed) = if first_is_low && high_child.is_some() {
                    let high_state = high_child
                        .map(|node| node.state_index)
                        .ok_or(TopologyError::InvalidBuildState)?;
                    (first.state_index, high_state, 2)
                } else if first_is_low {
                    (first.state_index, NONE, 1)
                } else {
                    (NONE, first.state_index, 1)
                };
                let parent_index = checked_u32(next.len())?;
                merges.push(BlockMerge {
                    low_state,
                    high_state,
                });
                next.push(BlockBuildNode {
                    block: parent,
                    state_index: parent_index,
                });
                index = index.checked_add(consumed).ok_or(TopologyError::Overflow)?;
            }
            levels.push(LevelRange {
                start: level_start,
                len: checked_u32(merges.len())?
                    .checked_sub(level_start)
                    .ok_or(TopologyError::Overflow)?,
            });
            census.push(block_census(next.len(), threadgroup_width)?);
            current = next;
        }

        Ok(Self {
            log_t,
            leaf_cycles: leaf_cycles.into_boxed_slice(),
            merges: merges.into_boxed_slice(),
            levels: levels.into_boxed_slice(),
            census: census.into_boxed_slice(),
        })
    }
}

#[derive(Clone, Copy)]
struct RwBuildNode {
    block: u64,
    address: u32,
    previous_value: u64,
    next_value: u64,
    state_index: u32,
    group_index: u32,
}

#[derive(Clone, Copy)]
struct BlockBuildNode {
    block: u64,
    state_index: u32,
}

fn merge_address_group(
    current: &[RwBuildNode],
    low: std::ops::Range<usize>,
    high: std::ops::Range<usize>,
    parent_block: u64,
    parent_group: u32,
    events: &mut Vec<RamRwMergeEvent>,
    next: &mut Vec<RwBuildNode>,
) -> Result<(), TopologyError> {
    let mut left = low.start;
    let mut right = high.start;
    while left < low.end || right < high.end {
        let left_node = current.get(left).filter(|_| left < low.end).copied();
        let right_node = current.get(right).filter(|_| right < high.end).copied();
        let (low_node, high_node) = match (left_node, right_node) {
            (Some(low_node), Some(high_node)) => match low_node.address.cmp(&high_node.address) {
                std::cmp::Ordering::Less => {
                    left = left.checked_add(1).ok_or(TopologyError::Overflow)?;
                    (Some(low_node), None)
                }
                std::cmp::Ordering::Equal => {
                    left = left.checked_add(1).ok_or(TopologyError::Overflow)?;
                    right = right.checked_add(1).ok_or(TopologyError::Overflow)?;
                    (Some(low_node), Some(high_node))
                }
                std::cmp::Ordering::Greater => {
                    right = right.checked_add(1).ok_or(TopologyError::Overflow)?;
                    (None, Some(high_node))
                }
            },
            (Some(low_node), None) => {
                left = left.checked_add(1).ok_or(TopologyError::Overflow)?;
                (Some(low_node), None)
            }
            (None, Some(high_node)) => {
                right = right.checked_add(1).ok_or(TopologyError::Overflow)?;
                (None, Some(high_node))
            }
            (None, None) => return Err(TopologyError::EmptyParent),
        };
        if let (Some(low_node), Some(high_node)) = (low_node, high_node) {
            if low_node.next_value != high_node.previous_value {
                return Err(TopologyError::CheckpointDiscontinuity {
                    address: low_node.address,
                });
            }
        }
        let address = low_node
            .map(|node| node.address)
            .or_else(|| high_node.map(|node| node.address))
            .ok_or(TopologyError::EmptyParent)?;
        let previous_value = low_node
            .map(|node| node.previous_value)
            .or_else(|| high_node.map(|node| node.previous_value))
            .ok_or(TopologyError::EmptyParent)?;
        let next_value = high_node
            .map(|node| node.next_value)
            .or_else(|| low_node.map(|node| node.next_value))
            .ok_or(TopologyError::EmptyParent)?;
        let parent_index = checked_u32(next.len())?;
        events.push(RamRwMergeEvent {
            low_state: low_node.map_or(NONE, |node| node.state_index),
            high_state: high_node.map_or(NONE, |node| node.state_index),
            group_index: parent_group,
            parent_index,
            low_absent_value: high_node.map_or(0, |node| node.previous_value),
            high_absent_value: low_node.map_or(0, |node| node.next_value),
        });
        next.push(RwBuildNode {
            block: parent_block,
            address,
            previous_value,
            next_value,
            state_index: parent_index,
            group_index: parent_group,
        });
    }
    Ok(())
}

fn union_cycles(
    records: &[RamAccessRecord],
    increment_cycles: &[u64],
) -> Result<Vec<u32>, TopologyError> {
    let capacity = records
        .len()
        .checked_add(increment_cycles.len())
        .ok_or(TopologyError::Overflow)?;
    let mut output = Vec::with_capacity(capacity);
    let mut access = 0;
    let mut increment = 0;
    while access < records.len() || increment < increment_cycles.len() {
        let access_cycle = records.get(access).map(|record| u64::from(record.cycle()));
        let increment_cycle = increment_cycles.get(increment).copied();
        let cycle = match (access_cycle, increment_cycle) {
            (Some(access_cycle), Some(increment_cycle)) => {
                match access_cycle.cmp(&increment_cycle) {
                    std::cmp::Ordering::Less => {
                        access += 1;
                        access_cycle
                    }
                    std::cmp::Ordering::Greater => {
                        increment += 1;
                        increment_cycle
                    }
                    std::cmp::Ordering::Equal => {
                        access += 1;
                        increment += 1;
                        access_cycle
                    }
                }
            }
            (Some(access_cycle), None) => {
                access += 1;
                access_cycle
            }
            (None, Some(increment_cycle)) => {
                increment += 1;
                increment_cycle
            }
            (None, None) => break,
        };
        output.push(checked_u32_u64(cycle)?);
    }
    Ok(output)
}

fn block_census(entries: usize, threadgroup_width: usize) -> Result<LevelCensus, TopologyError> {
    Ok(LevelCensus {
        entries: checked_u64(entries)?,
        groups: checked_u64(entries)?,
        tiles: checked_u64(entries.div_ceil(threadgroup_width))?,
    })
}

fn index(value: u32) -> Option<usize> {
    (value != NONE).then_some(value as usize)
}

fn checked_u32(value: usize) -> Result<u32, TopologyError> {
    u32::try_from(value).map_err(|_| TopologyError::IndexTooLarge)
}

fn checked_u32_u64(value: u64) -> Result<u32, TopologyError> {
    u32::try_from(value).map_err(|_| TopologyError::IndexTooLarge)
}

fn checked_u64(value: usize) -> Result<u64, TopologyError> {
    u64::try_from(value).map_err(|_| TopologyError::Overflow)
}

#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
pub enum TopologyError {
    #[error("RAM topology requires a nonzero threadgroup width")]
    ZeroThreadgroupWidth,
    #[error("RAM topology arithmetic overflowed")]
    Overflow,
    #[error("RAM topology index exceeds the u32 ABI")]
    IndexTooLarge,
    #[error("RAM topology contains an empty parent")]
    EmptyParent,
    #[error("RAM topology has an invalid level range")]
    InvalidLevelRange,
    #[error("RAM topology builder reached an invalid internal state")]
    InvalidBuildState,
    #[error("RAM topology round {round} is out of range")]
    RoundOutOfRange { round: usize },
    #[error("RAM value checkpoints are discontinuous at address {address}")]
    CheckpointDiscontinuity { address: u32 },
}
