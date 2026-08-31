use std::mem::size_of;

use thiserror::Error;

use super::owner::RamAccessRecord;

const NONE: u32 = u32::MAX;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LevelCensus {
    entries: u64,
}

impl LevelCensus {
    copy_field_getters! { pub, { entries: u64 }}
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
pub struct RamBlockTopology {
    log_t: usize,
    leaf_cycles: Box<[u32]>,
    merges: Box<[BlockMerge]>,
    levels: Box<[LevelRange]>,
    census: Box<[LevelCensus]>,
}

impl RamBlockTopology {
    copy_field_getters! { pub, { log_t: usize }}

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
    ) -> Result<Self, TopologyError> {
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
        census.push(block_census(current.len())?);

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
            census.push(block_census(next.len())?);
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
struct BlockBuildNode {
    block: u64,
    state_index: u32,
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

fn block_census(entries: usize) -> Result<LevelCensus, TopologyError> {
    Ok(LevelCensus {
        entries: checked_u64(entries)?,
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
    #[error("RAM topology arithmetic overflowed")]
    Overflow,
    #[error("RAM topology index exceeds the u32 ABI")]
    IndexTooLarge,
    #[error("RAM topology has an invalid level range")]
    InvalidLevelRange,
    #[error("RAM topology builder reached an invalid internal state")]
    InvalidBuildState,
    #[error("RAM topology round {round} is out of range")]
    RoundOutOfRange { round: usize },
}
